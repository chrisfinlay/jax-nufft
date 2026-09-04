"""Forward and adjoint wgridder operators built on jax-finufft.

The operators take a :class:`~jax_nufft.planning.WGridderPlan` (built once via
:func:`~jax_nufft.planning.make_plan`) plus the per-call image / visibility
arrays, and dispatch the wgridder algorithm in pure JAX:

  * ``dirty2vis(plan, image)`` — forward operator (degridder).
  * ``vis2dirty(plan, vis, weights=None)`` — adjoint operator (gridder), with
    the optional ``1/n`` factor applied on the output.

Both functions are fully traceable through ``jax.jit``, ``jax.vmap``, and
``jax.grad``. Channel and w-plane traversal can each be configured to use
``scan`` (lower memory) or ``vmap`` (potentially faster on GPU). The defaults
are ``scan`` for both, which is the safer choice for medium-to-large problems.

Sign convention: matches ducc's ``explicit_degridder``, i.e.

    V(u, v, w) = sum_{l, m} I(l, m) * exp(-2 pi i (u l + v m)) * exp(+2 pi i w (n - 1))

with the optional ``1/n`` factor applied on the adjoint output (matching ducc's
``divide_by_n=True``).
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax_finufft import nufft1, nufft2
from jax_finufft.options import Opts

from jax_nufft.kernel import phi
from jax_nufft.planning import WGridderPlan

# Canonical strategy names introduced in v0.1.1. The bare ``scan`` /
# ``vmap`` names from v0.1 are accepted as deprecated aliases that map
# to the ``dense_*`` variants (the v0.1 algorithm). A future release
# will add ``windowed_scan`` / ``windowed_vmap`` for the per-plane
# windowed path; the dense path stays as the parity baseline.
WStrategy = Literal[
    "dense_scan",
    "dense_vmap",
    "windowed_scan",
    "windowed_vmap",
    "auto",
    "scan",
    "vmap",
]
ChannelStrategy = Literal["scan", "vmap"]

_CANONICAL_W_STRATEGIES = ("dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap")
_W_STRATEGY_ALIASES: dict[str, WStrategy] = {"scan": "dense_scan", "vmap": "dense_vmap"}

# Smallest epsilon FINUFFT can honour in double precision; asking for less
# makes it warn and clamp, and ``filterwarnings = ["error"]`` turns that into a
# hard failure.
_MIN_NUFFT_EPSILON = 1e-14


def _nufft_epsilon(epsilon: float) -> float:
    """Accuracy asked of the (u,v) NUFFT for a plan targeting ``epsilon``.

    The error a caller sees is the sum of two independent contributions: the
    w-direction kernel (sized by :func:`jax_nufft.kernel.kernel_params`) and
    the (u,v) NUFFT. Handing the caller's whole budget to *each* of them means
    the total can only be a multiple of it, so the NUFFT gets one extra digit,
    which is exactly one more cell of FINUFFT's own width rule
    ``W = ceil(-log10(eps/10))`` in each of u and v.

    That is not a paper margin. FINUFFT's epsilon is a target, not a bound, and
    on small transforms the achieved error runs above it: measured directly
    against an exact 2D DFT on the 16x16 / 24-point problem in
    ``tests/test_against_dft.py``, ``nufft2`` returns 1.9x eps at eps = 1e-4
    and 2.9x at 1e-6. With the whole budget spent there the library could not
    honour its own ``2 * eps`` contract against the exact DFT no matter how
    wide the w-kernel got (issue #9); with the extra digit the (u,v) term drops
    to ~0.3x eps and the w-kernel is the leading term again, as it should be.

    Cost: FINUFFT spreads with a kernel one cell wider in each direction, so
    the spreading work grows like ``((W+1)/W)^2`` (~30% at W = 7) with the
    upsampled grid and the FFTs unchanged.
    """
    return max(epsilon / 10.0, _MIN_NUFFT_EPSILON)


def _canonicalise_w_strategy(
    name: str,
    *,
    plan: WGridderPlan | None = None,
    is_adjoint: bool | None = None,
) -> WStrategy:
    """Resolve user-facing ``w_strategy`` to a canonical name.

    ``"auto"`` is resolved here -- in the public wrapper, before the JIT
    boundary -- via :func:`_auto_w_strategy`. The static arg fed into
    :func:`_dirty2vis_jit` / :func:`_vis2dirty_jit` is therefore always
    one of the four canonical names, so two callers using ``"auto"`` on
    the same plan still share a JIT cache entry with each other and with
    the explicit canonical caller.

    Emits :class:`DeprecationWarning` for the v0.1 names.
    """
    if name == "auto":
        if plan is None or is_adjoint is None:
            raise ValueError(
                "w_strategy='auto' must be resolved with plan + is_adjoint "
                "context; this is handled by dirty2vis / vis2dirty."
            )
        return _auto_w_strategy(plan, is_adjoint=is_adjoint)
    if name in _CANONICAL_W_STRATEGIES:
        return cast(WStrategy, name)
    canonical = _W_STRATEGY_ALIASES.get(name)
    if canonical is not None:
        warnings.warn(
            f"w_strategy={name!r} is deprecated; use {canonical!r} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return canonical
    raise ValueError(
        f"unknown w_strategy: {name!r}; expected one of "
        f"{(*_CANONICAL_W_STRATEGIES, 'auto', *_W_STRATEGY_ALIASES)}"
    )


# Below this many rows, the whole plane loop is short enough that spinning up
# an OpenMP thread pool per FINUFFT call isn't worth it regardless of
# strategy -- ``_resolve_nthreads`` overrides the strategy-family rule to `1`
# in that regime. 100k is a round number well above every review fixture
# (largest is GH200_large at 50k rows) and well below workloads where the
# scan-family per-plane cost would actually amortise the pool spin-up;
# tunable, and pinned by ``tests/test_nthreads_resolution.py``.
#
# This override applies to *every* strategy, including the vmap family, even
# though the vmap family's steady-state rule (above the cutoff) is
# `nthreads=0`. That is deliberate, not an oversight: measured directly with
# the repository's timing protocol (plan + 1 warm-up outside the timer,
# median of 9 calls, `.block_until_ready()`) at the row counts every fixture
# in this repo actually uses (400-600 rows for the CPU telescope
# fixtures, 50k for GH200_LARGE -- all below this cutoff),
# `nthreads=0` is *not* uniformly better for the vmap family here:
#
#   * `dense_vmap`: `nthreads=0` is faster on MWA_extended off30 (170.5ms vs
#     384.9ms forward, 289.4ms vs 506.7ms adjoint -- ~1.8-2.3x) and MeerKAT
#     off30 (12.0ms vs 23.3ms forward, 16.3ms vs 30.3ms adjoint -- ~1.9x),
#     but `nthreads=1` is faster on EDA2 zenith (1.28ms vs 1.51ms forward,
#     1.37ms vs 1.68ms adjoint -- small n_w means too little batched work to
#     amortise a multi-threaded pool spin-up).
#   * `windowed_vmap` on MWA_extended off30 goes the *other* way entirely:
#     `nthreads=1` is ~6-7x *faster* than `nthreads=0` (409ms vs 2915ms
#     forward, 470ms vs 2997ms adjoint). Windowing shrinks each plane's
#     per-call row count to `max_window_size`, so at this row count the
#     windowed vmap call is thread-pool-spin-up-bound the same way the scan
#     family is, not batch-parallelism-bound the way dense_vmap at large n_w
#     is -- the assumption that "vmap wants threads" turns out to be a
#     dense_vmap-at-large-n_w statement, not a strategy-family-wide one.
#
# Given that split verdict -- and that the worst case of forcing `1` below
# the cutoff (losing ~2x on dense_vmap at moderate n_w) is far smaller than
# the worst case of not forcing it (losing ~7x on windowed_vmap) -- the
# override stays strategy-blind. One consequence worth being explicit about:
# every fixture currently in this repository has n_rows well below this
# cutoff, so the vmap-family steady-state branch of ``_resolve_nthreads``
# (the `else: return 0` below) is never exercised by anything in
# ``tests/test_benchmark_against_ducc.py`` or the README's benchmark
# tables -- only by the direct unit tests in
# ``tests/test_nthreads_resolution.py`` and the JIT-boundary spy tests in
# ``tests/test_jax_integration.py``, both of which pass a plan built with
# `n_rows > _NTHREADS_SMALL_N_ROWS` explicitly to reach it.
_NTHREADS_SMALL_N_ROWS = 100_000


def _resolve_nthreads(
    nthreads: int | None,
    w_strategy: str,
    n_rows: int,
    *,
    plan: WGridderPlan | None = None,
    is_adjoint: bool | None = None,
) -> int:
    """Resolve the public ``nthreads: int | None = None`` default to an ``int``.

    Runs *before* the JIT boundary in :func:`dirty2vis` / :func:`vis2dirty`
    so the value that reaches ``_dirty2vis_jit`` / ``_vis2dirty_jit`` (whose
    ``static_argnames`` includes ``"nthreads"``) is always a concrete
    ``int`` -- never ``None`` -- and so two callers that both leave
    ``nthreads`` at its default still share a JIT cache entry.

    Rule (issue #24, R11/D4):

      * an explicit ``nthreads`` (any int, including ``0``, i.e. "let
        FINUFFT decide") always passes straight through unchanged. No
        strategy canonicalisation is attempted in this branch, so
        ``w_strategy="auto"`` with no ``plan`` / ``is_adjoint`` does *not*
        raise here -- that context is only needed to pick a default.
      * otherwise ``w_strategy`` is canonicalised via
        :func:`_canonicalise_w_strategy` (resolving ``"auto"`` and the
        deprecated ``"scan"`` / ``"vmap"`` aliases the same way the
        strategy dispatch itself does), and:
          - if ``n_rows < _NTHREADS_SMALL_N_ROWS``, the plane loop is short
            enough that spinning up a thread pool per call isn't worth it
            regardless of strategy -> ``1`` (this overrides the strategy
            family below);
          - else ``"dense_scan"`` / ``"windowed_scan"`` (the *scan* family,
            which re-enters FINUFFT once per w-plane, so ``nthreads > 1``
            just re-spins the whole OpenMP pool on every plane) -> ``1``;
          - else ``"dense_vmap"`` / ``"windowed_vmap"`` (the *vmap* family,
            one batched FINUFFT call across planes) -> ``0`` (let FINUFFT
            thread the batch).

    Measured on the review machine (Apple M-series, 10 cores), default
    ``nthreads=0`` vs. explicit ``nthreads=1`` on ``dense_scan``: MWA_extended
    off30 3343.6ms vs 696.0ms (4.80x), MeerKAT off30 207.5ms vs 41.5ms
    (5.01x), EDA2 zenith 36.6ms vs 4.3ms (8.49x). The batched ``dense_vmap``
    is the opposite story -- it benefits from threads -- so a flat default of
    ``1`` would regress it; hence the strategy-family split rather than a
    single number.
    """
    if nthreads is not None:
        return nthreads
    canonical = _canonicalise_w_strategy(w_strategy, plan=plan, is_adjoint=is_adjoint)
    if n_rows < _NTHREADS_SMALL_N_ROWS:
        return 1
    if canonical in ("dense_scan", "windowed_scan"):
        return 1
    return 0


# Padding-overhead cutoff for the CPU heuristic: above this, windowed
# traversal is assumed to waste more on padded plane rows than it saves by
# shortening them, so ``auto`` stays on ``dense_scan``.
#
# This was 5.0 while ``window_padding_overhead`` averaged over *nonzero*
# windows only. Averaging over all planes (the cost the windowed strategies
# actually pay) raises the number on any plan with empty planes, so the
# cutoff is restated on the new scale. Measured over the five review
# fixtures x two pointings x epsilon in {1e-3, 1e-6, 1e-9, 1e-12} (40 plans;
# `tests/test_padding_overhead.py` pins the summary):
#
#   * 38 of the 40 sit at or below 3.2 on either definition.
#   * The two exceptions are both MWA_extended off30, the one fixture where
#     the windowed adjoint is the measured CPU win (AGENTS.md section 9):
#     3.62-4.47 on the old scale, 5.11-5.58 on the new one. Leaving the
#     cutoff at 5.0 would have flipped exactly that cell to ``dense_scan``.
#
# 8.0 is ~1.43x above the largest measured value, matching the margin the
# old 5.0 had over the old scale's 4.47 -- i.e. still a guard against
# pathological w-distributions rather than a boundary any real fixture
# approaches. Nothing in this repository's calibration set reaches it, and
# the ``auto`` choice on all 40 plans is identical to the pre-redefinition
# one.
#
# Shape, not just scale, is the weaker part of this gate: what decides
# windowed-vs-dense is the *work ratio* ``max_window_size / n_rows``, which
# on the new definition is ``window_padding_overhead * w_kernel_width / n_w``
# up to edge effects -- so a plan with a high overhead and a very large
# ``n_w / w_kernel_width`` (MWA_extended off30: overhead 5.1, ratio 36,
# windowed touching 14% of dense's rows) is a windowed win despite the high
# overhead. Replacing the absolute cutoff with that ratio reproduces all 40
# calibration decisions, but puts three of them (MWA_compact off30, MeerKAT
# off30, GH200_large zenith) within 5% of the boundary, where a heuristic
# should not be. Left as an absolute cutoff until there are CPU
# measurements on the far side of it.
_CPU_PADDING_CUTOFF = 8.0


def _auto_w_strategy_cpu(plan: WGridderPlan, *, is_adjoint: bool) -> WStrategy:
    """CPU-tuned heuristic from ``docs/v0.1.2-plan.md`` Part 4.

    Thresholds calibrated from AGENTS.md §9 CPU measurements where:

      * the windowed adjoint only wins at off-zenith when ``n_w`` is at
        least a factor of two above the w-kernel width, and
      * the windowed forward never measurably beats dense on the v0.1.1
        algorithm, so we never auto-pick a windowed forward.

    A high ``window_padding_overhead`` means windowed traversal would
    waste enough cycles on padded plane rows that dense wins; see
    :data:`_CPU_PADDING_CUTOFF` for where the cutoff sits and why.

    The constant-w fast path collapses ``n_w`` to one, so the
    small-``n_w`` branch always picks ``dense_scan`` there.
    """
    if plan.n_w <= plan.w_kernel_width + 2:
        return "dense_scan"
    if plan.window_padding_overhead > _CPU_PADDING_CUTOFF:
        return "dense_scan"
    if is_adjoint and plan.n_w / plan.w_kernel_width > 2.0:
        return "windowed_scan"
    return "dense_scan"


# Empirical cutoffs from the v0.1.2 GH200 baseline sweep (160 cells,
# docs/benchmarks/v0.1.2-baseline-gpu.json). On GPU, scan variants are
# 5-30x slower than vmap (kernel-launch overhead dominates per-plane
# work); the windowed_vmap wins only on the GH200_large (50k-row)
# fixture where dense_vmap's per-plane n_rows*W^2 starts to bite.
_GPU_LARGE_N_ROWS = 10_000
# Unlike the CPU cutoff above, this one is unchanged by the padding-overhead
# redefinition: the gate can only bind on plans with ``n_rows >= 10_000``,
# which on this calibration set means GH200_large alone. Seven of its eight
# pointing/epsilon cells have no empty planes, so the definitions agree
# exactly; the zenith, epsilon=1e-9 cell is 7.7% empty and moves from 1.20 to
# 1.30. All eight remain well below the 3.0 cutoff (the off-zenith cells are
# 2.65-2.76), so the redefinition cannot change the selected strategy.
_GPU_PADDING_CUTOFF = 3.0
_GPU_FORWARD_RATIO_CUTOFF = 3.0


def _auto_w_strategy_gpu(plan: WGridderPlan, *, is_adjoint: bool) -> WStrategy:
    """GPU-tuned heuristic from the v0.1.2 GH200 baseline sweep.

    On GH200 (cuFINUFFT on Hopper), the four-way picture across 20
    (operator, fixture) cells is:

      * ``_scan`` variants are always 5-30x slower than ``_vmap``;
        never auto-pick a scan strategy on GPU.
      * ``dense_vmap`` wins everywhere on small-to-medium fixtures
        (MWA, MeerKAT, EDA2).
      * ``windowed_vmap`` wins on the 50k-row ``GH200_large`` fixture
        for the adjoint (both pointings) and the forward at zenith;
        the forward at off-zenith stays on dense because the higher
        ``n_w`` makes the windowed padding overhead outweigh the
        per-plane row-count saving.

    The four gates below pick the cell's winner in 20/20 cases on the
    Part 5.6 baseline, with the runner-up always within the 15%
    acceptance bar -- so wrong choices are bounded losses, not
    cliff-edges.
    """
    if plan.n_w <= plan.w_kernel_width + 2:
        # Small n_w (incl. constant-w fast path); either dense or
        # windowed is fine. dense_vmap is simpler and the data agrees.
        return "dense_vmap"
    if plan.window_padding_overhead > _GPU_PADDING_CUTOFF:
        # Windowed wastes too much per-plane work on padded slices;
        # 3.0 is the empirical break-even on GH200.
        return "dense_vmap"
    if is_adjoint and plan.n_rows >= _GPU_LARGE_N_ROWS:
        # Adjoint headline win: large-row plans always favour windowed
        # on GH200 (regardless of pointing).
        return "windowed_vmap"
    if (
        not is_adjoint
        and plan.n_rows >= _GPU_LARGE_N_ROWS
        and plan.n_w <= _GPU_FORWARD_RATIO_CUTOFF * plan.w_kernel_width
    ):
        # Forward windowed only wins at low ``n_w`` (zenith-like) on
        # large-row plans; off-zenith with high ``n_w`` stays dense.
        return "windowed_vmap"
    return "dense_vmap"


def _auto_w_strategy(plan: WGridderPlan, *, is_adjoint: bool) -> WStrategy:
    """Pick a canonical ``w_strategy`` for a given plan and operator.

    Dispatches to :func:`_auto_w_strategy_cpu` or
    :func:`_auto_w_strategy_gpu` based on ``jax.devices()[0].platform``.
    Both branches share the same input contract (only reads
    ``plan.n_w``, ``plan.w_kernel_width``,
    ``plan.window_padding_overhead`` -- plus ``plan.n_rows`` on the
    GPU branch -- and the ``is_adjoint`` flag), and both return one of
    :data:`_CANONICAL_W_STRATEGIES`.

    Parameters
    ----------
    plan:
        The plan returned by :func:`~jax_nufft.planning.make_plan`.
    is_adjoint:
        ``True`` for ``vis2dirty`` (gridder, adjoint), ``False`` for
        ``dirty2vis`` (degridder, forward).
    """
    platform = jax.devices()[0].platform
    if platform == "gpu":
        return _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint)
    return _auto_w_strategy_cpu(plan, is_adjoint=is_adjoint)


def _cast_to_plan_dtype(
    array: Array,
    plan: WGridderPlan,
    *,
    name: str,
    target: np.dtype[Any],
) -> Array:
    """Cast ``array`` up to ``target``, or refuse to cast it *down* (issue #11).

    The plan owns the precision (``make_plan(dtype=...)``), so an input that
    is narrower than the plan is simply promoted — that is lossless, and it
    keeps the common "float32 image, float64 plan" workflow working. An input
    that is *wider* is a different matter: silently truncating it would throw
    away precision the caller deliberately produced, so it is a ``TypeError``
    naming both dtypes.

    Both branches exist to keep jax-finufft's bare, message-less
    ``AssertionError()`` on a dtype mismatch away from the caller: it names
    neither the offending array nor the fix.
    """
    in_dtype = np.dtype(array.dtype)
    # Compare like with like: a complex input is measured against the plan's
    # complex dtype, a real one against its real dtype, so "wider" means more
    # bits *per component* rather than merely "complex vs real".
    reference = (
        plan.complex_dtype if jnp.issubdtype(in_dtype, jnp.complexfloating) else plan.real_dtype
    )
    if in_dtype.itemsize > reference.itemsize:
        raise TypeError(
            f"{name} has dtype {in_dtype}, which is wider than this plan's "
            f"{reference}: casting it down would silently discard precision. "
            f"Either narrow {name} to {reference} yourself, or build the plan at "
            f"that precision with make_plan(..., dtype=jnp.float64) (which needs "
            "jax_enable_x64)."
        )
    return array.astype(target)


def _prepare_image(image: Array, plan: WGridderPlan) -> Array:
    """Broadcast / cast the input image to ``(n_chan, n_l, n_m)`` complex.

    The output dtype is the *plan's* ``complex_dtype``, never the image's: an
    image narrower than the plan is cast up (see :func:`_cast_to_plan_dtype`),
    a wider one is rejected. Deriving it from the image instead is what used
    to hand jax-finufft mismatched arrays.

    Order matters here. The dtype check runs on the array the caller passed,
    *before* the 2-D broadcast: ``jnp.broadcast_to`` is the point where a raw
    numpy array enters JAX, and with ``jax_enable_x64`` off that entry
    silently narrows float64/complex128 to 32 bits. Checking afterwards would
    make a float32 plan accept a float64 numpy image on the 2-D path while
    correctly rejecting the same image on the 3-D path, which touches nothing
    before the check.
    """
    if image.ndim == 2:
        if image.shape != (plan.n_l, plan.n_m):
            raise ValueError(
                f"image shape {image.shape} does not match plan ({plan.n_l}, {plan.n_m})"
            )
    elif image.ndim == 3:
        if image.shape != (plan.n_chan, plan.n_l, plan.n_m):
            raise ValueError(
                f"image shape {image.shape} does not match plan "
                f"({plan.n_chan}, {plan.n_l}, {plan.n_m})"
            )
    else:
        raise ValueError(f"image must be 2- or 3-dimensional; got ndim={image.ndim}")
    image = _cast_to_plan_dtype(image, plan, name="image", target=plan.complex_dtype)
    if image.ndim == 2:
        image = jnp.broadcast_to(image, (plan.n_chan, plan.n_l, plan.n_m))
    return image


def _validate_vis(vis: Array, plan: WGridderPlan) -> Array:
    """Check the visibility shape and cast it to the plan's complex dtype.

    A real ``vis`` is auto-promoted to complex (the typical workflow where the
    caller built it with a zero imaginary part), and a narrower complex one is
    cast up to the plan's precision; a wider one raises ``TypeError``.
    """
    if vis.ndim != 2 or vis.shape != (plan.n_rows, plan.n_chan):
        raise ValueError(
            f"vis shape {vis.shape} does not match plan ({plan.n_rows}, {plan.n_chan})"
        )
    return _cast_to_plan_dtype(vis, plan, name="vis", target=plan.complex_dtype)


def _validate_weights(weights: Array | None, plan: WGridderPlan) -> Array | None:
    """Check the weight shape and dtype and cast the weights to the plan's real dtype.

    ``weights`` are **real** by contract (they multiply the visibilities
    before gridding, matching ducc's ``wgt``), so unlike the image and the
    visibilities they get a kind check before the width check: a complex
    ``weights`` array has no correct interpretation here. Casting it to
    ``plan.real_dtype`` would drop the imaginary part silently, and routing it
    through the usual "wider dtype" path would be worse still — the advice
    would be to narrow complex128 to complex64, which loses data without
    fixing anything. Reject it outright instead.

    Once known real, the usual rule applies: narrower is cast up to
    ``plan.real_dtype``, wider is a ``TypeError``.
    """
    if weights is None:
        return None
    if weights.shape != (plan.n_rows, plan.n_chan):
        raise ValueError(
            f"weights shape {weights.shape} does not match plan ({plan.n_rows}, {plan.n_chan})"
        )
    # Kind check, not a width check: real (or integer / boolean mask) weights
    # are all meaningful, complex ones are not.
    if jnp.issubdtype(weights.dtype, jnp.complexfloating):
        raise TypeError(
            f"weights must be a real array; got dtype {np.dtype(weights.dtype)}. They are "
            "multiplied into the visibilities before gridding (ducc's `wgt`), so a "
            "complex weight has no defined meaning — pass the real weights as "
            f"{plan.real_dtype} instead."
        )
    return _cast_to_plan_dtype(weights, plan, name="weights", target=plan.real_dtype)


def _w_relative(w_absolute: Array, plan: WGridderPlan) -> Array:
    """Re-express w (wavelengths) relative to the plan's w-range midpoint.

    Everything inside the w-plane loop -- plane centres, per-row w, the kernel
    argument z, and the nshift compensating phase -- works in this ``delta =
    w - w0`` coordinate, so no phase the loop exponentiates ever scales with
    the *absolute* w. The constant part that is factored out this way is
    ``plan.w0_screen`` (see ``planning.make_plan``).

    Kept as a subtraction at the JIT boundary rather than as extra pre-shifted
    plan leaves for two reasons: ``plan.w_centers`` and ``plan.uvw_lambda``
    keep their documented absolute-wavelength semantics (the window builder and
    the w-coverage tests read them), and in float64 the subtraction is exact --
    the operands are within one w-extent of each other, which is Sterbenz's
    condition -- so deferring it costs no accuracy. A float32 plan does lose
    resolution here, but only the resolution its absolute w already lacked
    (issue #13).
    """
    return w_absolute - plan.w0


def _apply_nshift_compensation(
    vis: Array,
    w_rel: Array,
    plan: WGridderPlan,
    *,
    conjugate: bool,
) -> Array:
    """Apply issue #16's per-visibility ``nshift`` compensating phase.

    The w-plane loop evaluates its image-domain phase on
    ``plan.n_minus_1_shifted = n - 1 + nshift`` rather than on ``n - 1``, which
    is what halves the plane count (see ``planning.make_plan``). The exact
    identity that licenses the substitution is

        exp(2πi d (n-1)) = exp(2πi d (n-1+nshift)) · exp(-2πi d nshift),

    so what the plane loop computes is the true answer *times*
    ``exp(+2πi d nshift)``; multiplying by ``exp(-2πi d nshift)`` restores it.
    That factor depends only on the visibility (through ``d``), never on the
    pixel or the plane, so it is applied once per visibility per call rather
    than once per plane: the whole point of the optimisation.

    ``d`` here is ``w_rel = w - plan.w0``, not the absolute w: the w-centring
    follow-up moved the absolute part into ``plan.w0_screen`` precisely so that
    this phase and the plane phase it cancels against are both small. Passing
    an absolute w here would reintroduce the cancellation bug.

    ``conjugate=False`` is the forward form ``exp(-2πi d nshift)``, applied to
    the summed **output** visibilities. ``conjugate=True`` is the adjoint form
    ``exp(+2πi d nshift)``, applied to the **input** visibilities before the
    plane loop -- the adjoint of a diagonal scaling on the forward's output is
    the conjugate diagonal on the adjoint's input, which is exactly what keeps
    ``<A x, y> == <x, A^H y>`` (``tests/test_adjoint.py`` and
    ``tests/test_nshift.py::test_dot_product_identity_survives_nshift``).

    ``vis`` and ``w_rel`` must be in the *same* row order; the windowed
    helpers therefore call this on their w-sorted arrays, before/after the
    sort_perm round trip rather than across it.

    ``plan.nshift`` is static, so the ``== 0.0`` early-out is a trace-time
    Python branch: the constant-w fast path (``nshift == 0.0`` by choice, see
    ``planning.make_plan``) emits no multiply at all.
    """
    if plan.nshift == 0.0:
        return vis
    two_pi = 2.0 * jnp.pi
    phase = (two_pi * plan.nshift) * w_rel
    sign = 1.0 if conjugate else -1.0
    return vis * jnp.exp((sign * 1j * phase).astype(vis.dtype))


def _channel_forward(
    image_c: Array,
    u_ft_c: Array,
    v_ft_c: Array,
    w_rel_c: Array,
    plan: WGridderPlan,
    opts: Opts,
    w_strategy: WStrategy,
) -> Array:
    """Forward operator for a single channel: image (n_l, n_m) -> vis (n_rows,).

    ``u_ft_c`` / ``v_ft_c`` are the precomputed FINUFFT input coordinates for
    this channel (``2π · pixsize_* · uvw_lambda[c, :, 0|1]``); ``w_rel_c`` is
    the per-channel w-component in wavelengths *relative to ``plan.w0``*
    (see :func:`_w_relative`), used for the kernel z, the plane phase and the
    nshift compensation. The absolute part of w is carried by
    ``plan.w0_screen`` and never enters this function.
    """
    two_pi = 2.0 * jnp.pi
    cdtype = image_c.dtype
    n_rows = u_ft_c.shape[0]

    def w_plane_contribution(w_k: Array) -> Array:
        # issue #16: the shifted grid, not plan.n_minus_1. The resulting
        # exp(+2πi w nshift) excess is removed from the *summed* output below.
        phase = (two_pi * w_k) * plan.n_minus_1_shifted  # (n_l, n_m), real
        shift = jnp.exp((1j * phase).astype(cdtype))
        image_k = image_c * shift / plan.phi_hat_n.astype(cdtype)
        vis_k = nufft2(
            image_k, u_ft_c, v_ft_c, iflag=-1, eps=_nufft_epsilon(plan.epsilon), opts=opts
        )
        # w-direction kernel applied at the visibility output
        z = (w_rel_c - w_k) / plan.w_kernel_scale
        kernel_w = phi(z, plan.beta).astype(cdtype)
        return vis_k * kernel_w

    if w_strategy == "dense_vmap":
        contributions = jax.vmap(w_plane_contribution)(plan.w_centers_rel)
        vis_c = jnp.sum(contributions, axis=0)
    elif w_strategy == "dense_scan":

        def step(acc: Array, w_k: Array) -> tuple[Array, None]:
            return acc + w_plane_contribution(w_k), None

        init = jnp.zeros((n_rows,), dtype=cdtype)
        vis_c, _ = jax.lax.scan(step, init, plan.w_centers_rel)
    else:
        raise ValueError(f"unknown w_strategy: {w_strategy!r}")

    # issue #16: once per visibility, *after* the plane loop -- the factor is
    # common to every plane, so pulling it out of the sum is both cheaper and
    # exact (the sum is linear in it).
    return _apply_nshift_compensation(vis_c, w_rel_c, plan, conjugate=False)


def _channel_forward_windowed(
    image_c: Array,
    u_finufft_c: Array,
    v_finufft_c: Array,
    w_rel_sorted_c: Array,
    window_start_c: Array,
    plan: WGridderPlan,
    opts: Opts,
    w_strategy: WStrategy,
) -> Array:
    """Windowed forward operator for a single channel.

    Each w-plane processes a contiguous slice (size ``max_window_size``) of
    the w-sorted visibilities and scatters the per-row contributions back
    into the original visibility order via ``plan.sort_perm``. Visibilities
    inside the slice but outside the kernel's natural support pick up
    ``phi(z) = 0`` automatically, so no explicit mask is needed.

    ``u_finufft_c`` / ``v_finufft_c`` are the precomputed FINUFFT coords
    for this channel in **input** row order; ``w_rel_sorted_c`` is the
    w-component in wavelengths relative to ``plan.w0``
    (see :func:`_w_relative`), already in sorted-row order
    (``plan.uvw_lambda_sorted[c, :, 2] - plan.w0``).

    ``w_strategy`` selects scan-over-planes (``windowed_scan``, low memory)
    or vmap-over-planes (``windowed_vmap``, higher memory, possibly faster
    on GPU).
    """
    two_pi = 2.0 * jnp.pi
    # v0.1.2 Part 3.3 (plan option (b)): sorted u/v FINUFFT coords aren't
    # stored on the plan to save (n_chan, n_rows) * 2 floats; gather them
    # once per channel from the unsorted plan.u_finufft / plan.v_finufft.
    u_sorted = u_finufft_c[plan.sort_perm]
    v_sorted = v_finufft_c[plan.sort_perm]
    w_rel_sorted = w_rel_sorted_c

    cdtype = image_c.dtype
    n_rows = plan.n_rows
    max_window_size = plan.max_window_size
    # ``dynamic_slice`` clamps out-of-bounds starts, but doing so silently
    # would change which rows the kernel sees on the right edge. Clamp
    # explicitly so the slice is always in-bounds.
    lo_max = max(n_rows - max_window_size, 0)

    def plane_to_window(lo_raw: Array, w_k: Array) -> tuple[Array, Array]:
        """Compute one w-plane's per-window contribution.

        Returns ``(lo, contrib)`` where ``lo`` is the clamped sorted-row
        start of the window and ``contrib`` is the ``(max_window_size,)``
        complex contribution in sorted-row order (i.e. aligned with
        ``plan.uvw_lambda_sorted[c, lo:lo+max_window_size]``).
        """
        lo = jnp.clip(lo_raw, 0, lo_max)

        u_k = jax.lax.dynamic_slice(u_sorted, (lo,), (max_window_size,))
        v_k = jax.lax.dynamic_slice(v_sorted, (lo,), (max_window_size,))
        w_rel_window = jax.lax.dynamic_slice(w_rel_sorted, (lo,), (max_window_size,))

        # issue #16: shifted grid; compensated once per visibility below.
        phase = (two_pi * w_k) * plan.n_minus_1_shifted
        shift = jnp.exp((1j * phase).astype(cdtype))
        image_k = image_c * shift / plan.phi_hat_n.astype(cdtype)

        contrib = nufft2(image_k, u_k, v_k, iflag=-1, eps=_nufft_epsilon(plan.epsilon), opts=opts)

        z = (w_rel_window - w_k) / plan.w_kernel_scale
        kernel_w = phi(z, plan.beta).astype(cdtype)
        return lo, contrib * kernel_w

    if w_strategy == "windowed_vmap":
        # vmap path materialises one (n_rows,) row-order vector per plane
        # and sums; unchanged from the v0.1.1 behaviour aside from the
        # plane_to_window factoring.
        def plane_to_full_rows(lo_raw: Array, w_k: Array) -> Array:
            lo, contrib = plane_to_window(lo_raw, w_k)
            rows_k = jax.lax.dynamic_slice(plan.sort_perm, (lo,), (max_window_size,))
            return jnp.zeros((n_rows,), dtype=cdtype).at[rows_k].add(contrib)

        contributions = jax.vmap(plane_to_full_rows)(window_start_c, plan.w_centers_rel)
        # issue #16: ``plane_to_full_rows`` has already scattered back to
        # *input* row order, so the compensating phase has to be built from
        # the input-order w -- the phase and the visibility it multiplies must
        # share a row order. Recover it from the sorted copy with the inverse
        # permutation (``sorted[i]`` belongs to input row ``sort_perm[i]``);
        # that is one (n_rows,) scatter per channel against the loop's n_w
        # NUFFTs, so it costs nothing and keeps the plan from carrying a
        # second w array for this path alone.
        w_rel_c = jnp.zeros_like(w_rel_sorted).at[plan.sort_perm].set(w_rel_sorted)
        return _apply_nshift_compensation(
            jnp.sum(contributions, axis=0), w_rel_c, plan, conjugate=False
        )

    # windowed_scan path: keep the carry in sorted-row order so each plane
    # touches only its (max_window_size,)-sized slice. The per-step
    # dynamic_slice + add + dynamic_update_slice is O(max_window_size); the
    # v0.1.1 code paid O(n_rows) per plane for a full-row zero + scatter.
    def step(vis_sorted_acc: Array, args: tuple[Array, Array]) -> tuple[Array, None]:
        lo_raw, w_k = args
        lo, contrib = plane_to_window(lo_raw, w_k)
        old = jax.lax.dynamic_slice(vis_sorted_acc, (lo,), (max_window_size,))
        new = old + contrib
        return jax.lax.dynamic_update_slice(vis_sorted_acc, new, (lo,)), None

    vis_sorted_init = jnp.zeros((n_rows,), dtype=cdtype)
    vis_sorted, _ = jax.lax.scan(step, vis_sorted_init, (window_start_c, plan.w_centers_rel))
    # issue #16: applied here, in sorted-row order, where ``vis_sorted`` and
    # ``w_rel_sorted`` already agree -- i.e. before the unsort rather than
    # across it.
    vis_sorted = _apply_nshift_compensation(vis_sorted, w_rel_sorted, plan, conjugate=False)
    # Unsort once: sorted[i] is the contribution for original row sort_perm[i].
    return jnp.empty_like(vis_sorted).at[plan.sort_perm].set(vis_sorted)


def _channel_adjoint(
    vis_c: Array,
    u_ft_c: Array,
    v_ft_c: Array,
    w_rel_c: Array,
    plan: WGridderPlan,
    opts: Opts,
    w_strategy: WStrategy,
) -> Array:
    """Adjoint operator for a single channel: vis (n_rows,) -> dirty (n_l, n_m).

    See :func:`_channel_forward` for the coord-arg convention.
    """
    two_pi = 2.0 * jnp.pi
    cdtype = vis_c.dtype

    # issue #16: the adjoint of the forward's per-visibility output scaling by
    # exp(-2πi w nshift) is the conjugate scaling on the *input*, applied once
    # here rather than inside the plane loop (see _apply_nshift_compensation).
    vis_c = _apply_nshift_compensation(vis_c, w_rel_c, plan, conjugate=True)

    def w_plane_contribution(w_k: Array) -> Array:
        z = (w_rel_c - w_k) / plan.w_kernel_scale
        kernel_w = phi(z, plan.beta).astype(cdtype)
        vis_k = vis_c * kernel_w
        # Adjoint of the type-2 NUFFT is type 1 with iflag = +1 (the conjugate
        # of iflag=-1 used in the forward).
        h_k = nufft1(
            (plan.n_l, plan.n_m),
            vis_k,
            u_ft_c,
            v_ft_c,
            iflag=+1,
            eps=_nufft_epsilon(plan.epsilon),
            opts=opts,
        )
        # Adjoint of the image-domain shift exp(+2 pi i w_k (n-1+nshift)) is
        # its conjugate; issue #16 evaluates it on the shifted grid, matching
        # the forward's plane phase exactly.
        phase = (two_pi * w_k) * plan.n_minus_1_shifted
        shift = jnp.exp((-1j * phase).astype(cdtype))
        return h_k * shift / plan.phi_hat_n.astype(cdtype)

    if w_strategy == "dense_vmap":
        contributions = jax.vmap(w_plane_contribution)(plan.w_centers_rel)
        return jnp.sum(contributions, axis=0)

    if w_strategy == "dense_scan":

        def step(acc: Array, w_k: Array) -> tuple[Array, None]:
            return acc + w_plane_contribution(w_k), None

        init = jnp.zeros((plan.n_l, plan.n_m), dtype=cdtype)
        result, _ = jax.lax.scan(step, init, plan.w_centers_rel)
        return result

    raise ValueError(f"unknown w_strategy: {w_strategy!r}")


@partial(
    jax.jit,
    static_argnames=("w_strategy", "channel_strategy", "nthreads"),
)
def _dirty2vis_jit(
    plan: WGridderPlan,
    image: Array,
    *,
    w_strategy: WStrategy,
    channel_strategy: ChannelStrategy,
    nthreads: int,
) -> Array:
    opts = Opts(nthreads=nthreads)

    # issue #16 follow-up: the w-range midpoint leaves the plane loop as a
    # constant image-domain phase screen. Applied here, once per call on the
    # (n_chan, n_l, n_m) image, rather than once per plane -- it depends on the
    # pixel but on neither the visibility nor the plane.
    image = image * plan.w0_screen

    if w_strategy in ("windowed_scan", "windowed_vmap"):
        # Windowed path: per-channel function takes precomputed FINUFFT
        # coords (input row order, gathered via sort_perm inside the helper)
        # plus the pre-permuted w-component and the per-channel window-start
        # table from the plan.
        w_rel_sorted = _w_relative(plan.uvw_lambda_sorted[..., 2], plan)  # (n_chan, n_rows)
        if channel_strategy == "vmap":
            vis_per_chan = jax.vmap(
                lambda im_c, u_c, v_c, w_s_c, ws_c: _channel_forward_windowed(
                    im_c, u_c, v_c, w_s_c, ws_c, plan, opts, w_strategy
                )
            )(
                image,
                plan.u_finufft,
                plan.v_finufft,
                w_rel_sorted,
                plan.window_start,
            )
        elif channel_strategy == "scan":

            def step_w(
                _: None,
                args: tuple[Array, Array, Array, Array, Array],
            ) -> tuple[None, Array]:
                im_c, u_c, v_c, w_s_c, ws_c = args
                return None, _channel_forward_windowed(
                    im_c, u_c, v_c, w_s_c, ws_c, plan, opts, w_strategy
                )

            _, vis_per_chan = jax.lax.scan(
                step_w,
                None,
                (
                    image,
                    plan.u_finufft,
                    plan.v_finufft,
                    w_rel_sorted,
                    plan.window_start,
                ),
            )
        else:
            raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")
        return vis_per_chan.T  # (n_rows, n_chan)

    w_rel = _w_relative(plan.uvw_lambda[..., 2], plan)  # (n_chan, n_rows)
    if channel_strategy == "vmap":
        vis_per_chan = jax.vmap(
            lambda im_c, u_c, v_c, w_c: _channel_forward(
                im_c, u_c, v_c, w_c, plan, opts, w_strategy
            )
        )(image, plan.u_finufft, plan.v_finufft, w_rel)
    elif channel_strategy == "scan":

        def step(_: None, args: tuple[Array, Array, Array, Array]) -> tuple[None, Array]:
            im_c, u_c, v_c, w_c = args
            return None, _channel_forward(im_c, u_c, v_c, w_c, plan, opts, w_strategy)

        _, vis_per_chan = jax.lax.scan(step, None, (image, plan.u_finufft, plan.v_finufft, w_rel))
    else:
        raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")

    return vis_per_chan.T  # (n_rows, n_chan)


def dirty2vis(
    plan: WGridderPlan,
    image: Array,
    *,
    w_strategy: WStrategy = "dense_scan",
    channel_strategy: ChannelStrategy = "scan",
    nthreads: int | None = None,
) -> Array:
    """Forward wgridder: image cube -> visibilities.

    Parameters
    ----------
    plan:
        Pre-built plan from :func:`~jax_nufft.planning.make_plan`.
    image:
        Either ``(n_chan, n_l, n_m)`` or ``(n_l, n_m)`` (broadcast across
        channels). Real or complex; real input is promoted to complex. The
        precision is the plan's, not the image's: an image narrower than
        ``plan.complex_dtype`` is cast up to it, a wider one raises
        ``TypeError`` (issue #11).
    w_strategy:
        ``"dense_scan"`` (default, low memory) or ``"dense_vmap"`` (potentially
        faster on GPU but allocates ``n_w * image_size`` peak memory).
        ``"windowed_scan"`` / ``"windowed_vmap"`` use the per-plane windowed
        path. ``"auto"`` resolves to a canonical name before the JIT boundary
        via :func:`_auto_w_strategy` (so cache sharing is preserved); see that
        helper's docstring for the heuristic. The bare names ``"scan"`` /
        ``"vmap"`` are accepted as deprecated aliases.
    channel_strategy:
        ``"scan"`` (default) or ``"vmap"`` for the channel loop.
    nthreads:
        Threads to pass to jax-finufft. Default ``None`` resolves (before the
        JIT boundary, via :func:`_resolve_nthreads`, so the JIT cache is
        still shared across callers that leave it at the default) to a
        strategy-aware choice: ``1`` for the ``*_scan`` strategies, which
        re-enter FINUFFT once per w-plane, so ``nthreads > 1`` just re-spins
        the whole OpenMP thread pool on every plane -- measured on the
        review machine (Apple M-series, 10 cores) at 4.80x-8.49x slower than
        ``nthreads=1`` for the pre-#24 default of ``0`` (MWA_extended off30
        3343.6ms vs 696.0ms, MeerKAT off30 207.5ms vs 41.5ms, EDA2 zenith
        36.6ms vs 4.3ms); ``0`` (let FINUFFT decide) for the ``*_vmap``
        strategies, whose single batched FINUFFT call over all w-planes does
        benefit from threads. Below ``_NTHREADS_SMALL_N_ROWS`` rows this
        override doesn't apply -- the plane loop is short enough that
        spinning up a pool at all isn't worth it, so every strategy gets
        ``1``. Pass an explicit ``int`` (including ``0``) to opt out.

    Returns
    -------
    vis:
        Complex array of shape ``(n_rows, n_chan)`` and dtype
        ``plan.complex_dtype``.
    """
    w_strategy = _canonicalise_w_strategy(w_strategy, plan=plan, is_adjoint=False)
    image = _prepare_image(image, plan)
    resolved_nthreads = _resolve_nthreads(nthreads, w_strategy, plan.n_rows)
    return _dirty2vis_jit(
        plan,
        image,
        w_strategy=w_strategy,
        channel_strategy=channel_strategy,
        nthreads=resolved_nthreads,
    )


def _channel_adjoint_windowed(
    vis_sorted_c: Array,
    u_finufft_c: Array,
    v_finufft_c: Array,
    w_rel_sorted_c: Array,
    window_start_c: Array,
    plan: WGridderPlan,
    opts: Opts,
    w_strategy: WStrategy,
) -> Array:
    """Windowed adjoint operator for a single channel.

    Mirrors :func:`_channel_forward_windowed`: per plane we take a
    contiguous slice of the w-sorted visibilities, apply the w-kernel
    weight (which zeros out padded entries automatically), run a 2D
    NUFFT type 1 to land an image, and accumulate. ``w_strategy``
    chooses scan-over-planes (``windowed_scan``) or vmap-over-planes
    (``windowed_vmap``).

    See :func:`_channel_forward_windowed` for the coord-arg convention.
    """
    two_pi = 2.0 * jnp.pi
    u_sorted = u_finufft_c[plan.sort_perm]
    v_sorted = v_finufft_c[plan.sort_perm]
    w_rel_sorted = w_rel_sorted_c

    cdtype = vis_sorted_c.dtype
    max_window_size = plan.max_window_size
    lo_max = max(plan.n_rows - max_window_size, 0)

    # issue #16: conjugate compensating phase on the input visibilities, once
    # per visibility before the plane loop. Both arrays are in sorted-row
    # order here, so no permutation is involved.
    vis_sorted_c = _apply_nshift_compensation(vis_sorted_c, w_rel_sorted, plan, conjugate=True)

    def plane_to_image(lo_raw: Array, w_k: Array) -> Array:
        lo = jnp.clip(lo_raw, 0, lo_max)

        u_k = jax.lax.dynamic_slice(u_sorted, (lo,), (max_window_size,))
        v_k = jax.lax.dynamic_slice(v_sorted, (lo,), (max_window_size,))
        w_rel_window = jax.lax.dynamic_slice(w_rel_sorted, (lo,), (max_window_size,))
        vis_k = jax.lax.dynamic_slice(vis_sorted_c, (lo,), (max_window_size,))

        z = (w_rel_window - w_k) / plan.w_kernel_scale
        kernel_w = phi(z, plan.beta).astype(cdtype)
        vis_k = vis_k * kernel_w

        h_k = nufft1(
            (plan.n_l, plan.n_m),
            vis_k,
            u_k,
            v_k,
            iflag=+1,
            eps=_nufft_epsilon(plan.epsilon),
            opts=opts,
        )
        # issue #16: shifted grid, matching the forward's plane phase.
        phase = (two_pi * w_k) * plan.n_minus_1_shifted
        shift = jnp.exp((-1j * phase).astype(cdtype))
        return h_k * shift / plan.phi_hat_n.astype(cdtype)

    if w_strategy == "windowed_vmap":
        contributions = jax.vmap(plane_to_image)(window_start_c, plan.w_centers_rel)
        return jnp.sum(contributions, axis=0)

    def step(dirty_acc: Array, args: tuple[Array, Array]) -> tuple[Array, None]:
        lo_raw, w_k = args
        return dirty_acc + plane_to_image(lo_raw, w_k), None

    dirty_init = jnp.zeros((plan.n_l, plan.n_m), dtype=cdtype)
    dirty_c, _ = jax.lax.scan(step, dirty_init, (window_start_c, plan.w_centers_rel))
    return dirty_c


@partial(
    jax.jit,
    static_argnames=("w_strategy", "channel_strategy", "nthreads", "apply_w_weights"),
)
def _vis2dirty_jit(
    plan: WGridderPlan,
    vis: Array,
    weights: Array | None,
    *,
    w_strategy: WStrategy,
    channel_strategy: ChannelStrategy,
    nthreads: int,
    apply_w_weights: bool,
) -> Array:
    opts = Opts(nthreads=nthreads)

    # Visibility input has shape (n_rows, n_chan). Channel-loop expects channel
    # axis first, so transpose once up front.
    vis_per_chan = vis.T  # (n_chan, n_rows)
    if apply_w_weights:
        # weights has shape (n_rows, n_chan); align to (n_chan, n_rows).
        weights_per_chan = weights.T.astype(vis_per_chan.dtype)  # type: ignore[union-attr]
        vis_per_chan = vis_per_chan * weights_per_chan

    if w_strategy in ("windowed_scan", "windowed_vmap"):
        # Apply sort_perm once per channel so windowed slices line up with
        # plan.window_start.
        vis_sorted_per_chan = vis_per_chan[:, plan.sort_perm]
        w_rel_sorted = _w_relative(plan.uvw_lambda_sorted[..., 2], plan)
        if channel_strategy == "vmap":
            dirty_per_chan = jax.vmap(
                lambda v_s_c, u_c, vv_c, w_s_c, ws_c: _channel_adjoint_windowed(
                    v_s_c, u_c, vv_c, w_s_c, ws_c, plan, opts, w_strategy
                )
            )(
                vis_sorted_per_chan,
                plan.u_finufft,
                plan.v_finufft,
                w_rel_sorted,
                plan.window_start,
            )
        elif channel_strategy == "scan":

            def step_w(
                _: None,
                args: tuple[Array, Array, Array, Array, Array],
            ) -> tuple[None, Array]:
                v_s_c, u_c, vv_c, w_s_c, ws_c = args
                return None, _channel_adjoint_windowed(
                    v_s_c, u_c, vv_c, w_s_c, ws_c, plan, opts, w_strategy
                )

            _, dirty_per_chan = jax.lax.scan(
                step_w,
                None,
                (
                    vis_sorted_per_chan,
                    plan.u_finufft,
                    plan.v_finufft,
                    w_rel_sorted,
                    plan.window_start,
                ),
            )
        else:
            raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")
    elif channel_strategy == "vmap":
        w_rel = _w_relative(plan.uvw_lambda[..., 2], plan)
        dirty_per_chan = jax.vmap(
            lambda v_c, u_c, vv_c, w_c: _channel_adjoint(
                v_c, u_c, vv_c, w_c, plan, opts, w_strategy
            )
        )(vis_per_chan, plan.u_finufft, plan.v_finufft, w_rel)
    elif channel_strategy == "scan":
        w_rel = _w_relative(plan.uvw_lambda[..., 2], plan)

        def step(_: None, args: tuple[Array, Array, Array, Array]) -> tuple[None, Array]:
            v_c, u_c, vv_c, w_c = args
            return None, _channel_adjoint(v_c, u_c, vv_c, w_c, plan, opts, w_strategy)

        _, dirty_per_chan = jax.lax.scan(
            step, None, (vis_per_chan, plan.u_finufft, plan.v_finufft, w_rel)
        )
    else:
        raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")

    # issue #16 follow-up: conjugate of the forward's w0 phase screen.
    #
    # Which side it belongs on follows from the adjoint relation, not from
    # trial and error. The forward is A = D . N . M, reading right to left:
    # M multiplies the image by the screen, N is the w-plane machinery, D
    # multiplies the output visibilities by the nshift compensation. Both M and
    # D are diagonal, so A^H = M^H . N^H . D^H with M^H = diag(conj(screen)) --
    # i.e. the screen's conjugate acts on the *image* end of the adjoint, which
    # is its output. D^H = diag(conj(compensation)) acts on the adjoint's input
    # and is already applied inside the channel helpers.
    #
    # It must land before ``.real`` below: conjugating a complex screen and then
    # discarding the imaginary part is not the same operation in the other
    # order, and the dot-product identity in tests/test_adjoint.py is what
    # would catch getting this wrong.
    dirty_per_chan = dirty_per_chan * jnp.conj(plan.w0_screen)

    # Apply 1/n on the output (matching ducc's divide_by_n=True), and take
    # the real part to land in real space.
    #
    # issue #16: this is ``plan.n_minus_1``, NOT ``plan.n_minus_1_shifted``,
    # and that is not an oversight. ``nshift`` is a bookkeeping device for the
    # w-phase only -- it is introduced and removed by an exact identity, and
    # the operator it defines is still the one whose output is divided by the
    # *physical* ``n = sqrt(1 - l² - m²)``. Using the shifted grid here would
    # divide by ``n + nshift``, a smooth per-pixel gain error of tens of
    # percent that no parity test's epsilon budget would absorb -- but it
    # would leave the plane count halved and every structural nshift test
    # green, which is exactly why it gets its own comment.
    n_grid = (plan.n_minus_1 + 1.0).astype(dirty_per_chan.real.dtype)
    safe_n = jnp.where(n_grid > 0.0, n_grid, 1.0)
    return jnp.where(n_grid > 0.0, dirty_per_chan.real / safe_n, 0.0)


def vis2dirty(
    plan: WGridderPlan,
    vis: Array,
    *,
    weights: Array | None = None,
    w_strategy: WStrategy = "dense_scan",
    channel_strategy: ChannelStrategy = "scan",
    nthreads: int | None = None,
) -> Array:
    """Adjoint wgridder: visibilities -> image cube (with 1/n factor).

    Parameters
    ----------
    plan:
        Pre-built plan from :func:`~jax_nufft.planning.make_plan`.
    vis:
        Complex array of shape ``(n_rows, n_chan)``. As for the image in
        :func:`dirty2vis`, the plan owns the precision: a narrower ``vis``
        (or a real one) is cast up to ``plan.complex_dtype``, a wider one
        raises ``TypeError`` (issue #11).
    weights:
        Optional real array of shape ``(n_rows, n_chan)``, multiplied into the
        visibilities before gridding (matches ducc's ``wgt`` argument). Cast
        to ``plan.real_dtype`` under the same rule; a complex ``weights``
        array is rejected with ``TypeError`` rather than silently losing its
        imaginary part.
    w_strategy:
        ``"dense_scan"`` (default), ``"dense_vmap"``, ``"windowed_scan"``,
        ``"windowed_vmap"``, or ``"auto"``; same semantics as in
        :func:`dirty2vis`. The bare names ``"scan"`` / ``"vmap"`` are accepted
        as deprecated aliases.
    channel_strategy:
        ``"scan"`` (default) or ``"vmap"``.
    nthreads:
        Threads to pass to jax-finufft. Default ``None`` resolves (before the
        JIT boundary, via :func:`_resolve_nthreads`, so the JIT cache is
        still shared across callers that leave it at the default) to a
        strategy-aware choice: ``1`` for the ``*_scan`` strategies, which
        re-enter FINUFFT once per w-plane, so ``nthreads > 1`` just re-spins
        the whole OpenMP thread pool on every plane -- measured on the
        review machine (Apple M-series, 10 cores) at 4.80x-8.49x slower than
        ``nthreads=1`` for the pre-#24 default of ``0`` (MWA_extended off30
        3343.6ms vs 696.0ms, MeerKAT off30 207.5ms vs 41.5ms, EDA2 zenith
        36.6ms vs 4.3ms); ``0`` (let FINUFFT decide) for the ``*_vmap``
        strategies, whose single batched FINUFFT call over all w-planes does
        benefit from threads. Below ``_NTHREADS_SMALL_N_ROWS`` rows this
        override doesn't apply -- the plane loop is short enough that
        spinning up a pool at all isn't worth it, so every strategy gets
        ``1``. Pass an explicit ``int`` (including ``0``) to opt out.

    Returns
    -------
    dirty:
        Real array of shape ``(n_chan, n_l, n_m)`` and dtype
        ``plan.real_dtype``.
    """
    w_strategy = _canonicalise_w_strategy(w_strategy, plan=plan, is_adjoint=True)
    vis = _validate_vis(vis, plan)
    weights = _validate_weights(weights, plan)
    apply_w = weights is not None
    resolved_nthreads = _resolve_nthreads(nthreads, w_strategy, plan.n_rows)
    return _vis2dirty_jit(
        plan,
        vis,
        weights if apply_w else jnp.zeros((), dtype=vis.real.dtype),
        w_strategy=w_strategy,
        channel_strategy=channel_strategy,
        nthreads=resolved_nthreads,
        apply_w_weights=apply_w,
    )


__all__ = ["dirty2vis", "vis2dirty"]
