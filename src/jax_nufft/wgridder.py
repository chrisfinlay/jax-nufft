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
        strategy dispatch itself does) *unless it is already one of the
        four canonical names*, in which case it is used as-is and the
        canonicalisation is skipped entirely -- see the note below on why
        that skip is a correctness guarantee rather than an optimisation.
        Then:
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

    One canonicalisation (issue #46)
    --------------------------------
    Since ``w_strategy`` also defaults to ``"auto"``, the common path is
    *both* defaults at once, and :func:`dirty2vis` / :func:`vis2dirty`
    resolve the strategy in the wrapper and hand the resolved name down to
    this function. Issue #46 asks for exactly one canonicalisation on that
    path, so the already-canonical branch below short-circuits rather than
    calling :func:`_canonicalise_w_strategy` a second time.

    That is a guarantee, not a micro-optimisation. Re-canonicalising a
    canonical name happens to be a no-op *today* -- the function is a fixed
    point on those four strings -- but relying on that makes an invariant
    of what is really an implementation detail of another function: any
    future change that gave ``_canonicalise_w_strategy`` a plan-dependent
    branch, or made it re-consult the heuristic, would silently start
    resolving twice and could disagree with the strategy the wrapper
    already committed to at the JIT boundary. Skipping the call removes
    that coupling instead of documenting it.

    Direct callers passing a raw name still work unchanged: ``"auto"`` and
    the deprecated aliases fall through to the full canonicalisation below,
    including its ``ValueError`` when ``"auto"`` arrives without ``plan`` /
    ``is_adjoint`` context.
    """
    if nthreads is not None:
        return nthreads
    if w_strategy in _CANONICAL_W_STRATEGIES:
        # Already resolved (the operators' path): use it as-is.
        canonical: WStrategy = cast(WStrategy, w_strategy)
    else:
        canonical = _canonicalise_w_strategy(w_strategy, plan=plan, is_adjoint=is_adjoint)
    if n_rows < _NTHREADS_SMALL_N_ROWS:
        return 1
    if canonical in ("dense_scan", "windowed_scan"):
        return 1
    return 0


# The CPU windowed/dense cutoff on ``plan.window_padding_overhead``, named
# rather than inlined since issue #43 moved the scale it is stated on.
#
# Through v0.1.2 this was a bare ``5.0`` sitting just above the *padded*-scale
# maximum of the review calibration grid (4.933, MWA_extended off30 at
# eps 1e-9), so it never bound on that grid: no cell reached it, and the branch
# below was exercised only by the stub plans in ``tests/test_auto_strategy.py``.
# "No cell of the grid" is the honest scope -- the grid is one draw per fixture
# (``synthetic_uvw(..., seed=0)``), and other draws of the same fixtures do
# cross both cutoffs. Over seeds 0-11, MWA_extended off30 at eps 1e-3 spans
# 4.81 to 6.68 on the padded scale and 5.78 to 8.09 on the corrected one,
# crossing 5.0 and 6.0 respectively on ten of the twelve. The seed sensitivity
# is issue #34; what matters here is that the two rules agree on all twelve,
# which ``test_auto_picks_survive_the_redefinition_across_seeds`` pins.
# Issue #43 moved the denominator onto the live rows, which raises the same
# grid's maximum to 5.784 in float64 and 5.792 in float32 -- both MWA_extended
# off30 at eps 1e-3. Left at 5.0 the cutoff would start binding, and the first
# thing it would catch is the four MWA_extended off30 adjoint cells, flipping
# them from ``windowed_scan`` to ``dense_scan`` against the measured CPU win
# recorded in AGENTS.md section 9. The metric changed; the measurements did
# not, so the cutoff has to move with it.
#
# 6.0 is the smallest round value that clears the grid, and it is just below
# where carrying the old cutoff across the change of scale lands. The worst
# padded-to-live inflation on the grid is 5.7843 / 4.8089 = 1.2028 (1.2043 on
# the float32 leg), so the proportional construction gives 5.0 * 1.2028 =
# 6.014, and 6.0 is that value rounded down to the nearest tenth -- the
# construction does not yield 6.0 exactly, it yields 6.014, and rounding down
# is what keeps the cutoff a round number without loosening it. So this is not
# a fresh calibration: it is the same 5.0, restated on the denominator that
# replaced the one it was written against. It sits 3.7% above the worst fixture
# where 5.0 sat 1.4% above the worst on the old scale, i.e. slightly the more
# permissive of the two; that direction is the safe one here, since crossing
# this cutoff can only take a plan off ``windowed_scan``, which AGENTS.md
# section 9 measures as the CPU win on exactly the fixtures that approach it.
#
# Pinned to this exact value by
# ``tests/test_padding_overhead.py::test_cpu_padding_cutoff_is_six_and_still_gates``,
# which also exercises the branch on both sides of it: a cutoff raised far
# enough to "clear the grid" trivially would disable the guard rather than
# restate it, and a lower bound alone cannot tell the two apart.
_CPU_PADDING_CUTOFF = 6.0


def _auto_w_strategy_cpu(plan: WGridderPlan, *, is_adjoint: bool) -> WStrategy:
    """CPU-tuned heuristic from ``docs/v0.1.2-plan.md`` Part 4.

    Thresholds calibrated from AGENTS.md §9 CPU measurements where:

      * the windowed adjoint only wins at off-zenith when ``n_w`` is at
        least a factor of two above the w-kernel width, and
      * the windowed forward never measurably beats dense on the v0.1.1
        algorithm, so we never auto-pick a windowed forward.

    A high ``window_padding_overhead`` means windowed traversal would
    waste enough cycles on padded plane rows that dense wins;
    :data:`_CPU_PADDING_CUTOFF` is conservative and no cell of the pinned
    calibration grid reaches it -- though other random draws of the same
    fixtures do, so this branch is live in practice rather than dead.

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
# slower than vmap in every one of that sweep's 160 scan/vmap pairs
# (kernel-launch overhead dominates per-plane work) -- by 1.45x to 32.7x,
# median 6.1x, recomputed from the JSON. The "never pick scan on GPU"
# rule rests on the *universality*, not on the size, of that gap;
# the windowed_vmap wins only on the GH200_large (50k-row)
# fixture where dense_vmap's per-plane n_rows*W^2 starts to bite.
_GPU_LARGE_N_ROWS = 10_000
# Unmoved by issue #43, unlike ``_CPU_PADDING_CUTOFF`` above -- and that is a
# measurement, not an omission. The redefinition raises every cell's overhead
# (by under 0.1% where the windows are wide, by up to 20% where they are
# narrow),
# but 3.0 happens to sit in a gap no cell crosses: over the whole calibration
# grid the nearest below it reads 2.906 after the change (EDA2 off30 at
# eps 1e-3, up from 2.599, the largest shift of any cell near the cutoff) and
# the nearest above reads 3.050 (MWA_compact off30 at eps 1e-3, up from 3.026).
# Every cell keeps the side it was on, on both precision legs, so the GH200
# sweep this number was fitted to still resolves to the same strategy in all 20
# of its cells and there is nothing to recalibrate.
_GPU_PADDING_CUTOFF = 3.0
_GPU_FORWARD_RATIO_CUTOFF = 3.0


def _auto_w_strategy_gpu(plan: WGridderPlan, *, is_adjoint: bool) -> WStrategy:
    """GPU-tuned heuristic from the v0.1.2 GH200 baseline sweep.

    On GH200 (cuFINUFFT on Hopper), the four-way picture across 20
    (operator, fixture) cells is:

      * ``_scan`` variants are always slower than ``_vmap`` -- in all
        160 scan/vmap pairs of the sweep, by 1.45x to 32.7x (median
        6.1x); never auto-pick a scan strategy on GPU.
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

    Kept as a subtraction here rather than as an extra pre-shifted plan leaf
    for two reasons: the plan's stored baselines keep their documented
    absolute semantics (``uvw_m`` in metres, scaled by ``inv_lambda`` -- see
    :func:`_channel_ft_coords`), and in float64 the subtraction is exact --
    the operands are within one w-extent of each other, which is Sterbenz's
    condition -- so deferring it costs no accuracy. A float32 plan does lose
    resolution here, but only the resolution its absolute w already lacked
    (issue #13).
    """
    return w_absolute - plan.w0


def _channel_ft_coords(
    uvw_m: Array,
    inv_lambda_c: Array,
    plan: WGridderPlan,
) -> tuple[Array, Array, Array]:
    """Derive one channel's ``(u_ft, v_ft, w_rel)`` from the stored leaves.

    issue #23: the plan stores the baselines exactly once, in metres and in
    whatever row order the caller hands this function (``plan.uvw_m``, or its
    ``sort_perm`` gather for the windowed helpers), plus one scalar per
    channel, ``inv_lambda[c] = freq[c] / c``. The three per-channel arrays the
    operators actually consume are rebuilt here instead of being stored:

        u_ft  = (2π · pixsize_l · inv_lambda[c]) · uvw_m[:, 0]
        v_ft  = (2π · pixsize_m · inv_lambda[c]) · uvw_m[:, 1]
        w     =  inv_lambda[c] · uvw_m[:, 2]

    ``pixsize_l`` / ``pixsize_m`` are static plan fields, so ``2π · pixsize_*``
    is folded at trace time and each line is one scalar-times-vector multiply:
    three multiplies per row per channel, invisible next to the ``n_w`` NUFFTs
    that follow. What it replaces is four ``(n_chan, n_rows)``-or-larger plan
    leaves -- ``uvw_lambda``, ``uvw_lambda_sorted``, ``u_finufft``,
    ``v_finufft`` -- i.e. 8 float64 per (channel, row) of which only these
    4 were ever read.

    Note the ordering: the two scalars are multiplied together *first* and the
    row vector last, so the per-row cost stays at a single multiply rather than
    two. That reassociates ``(2π · pixsize) · (inv_lambda · uvw)`` -- what
    v0.1.2 stored -- by a ulp or so, a rounding-level change to a FINUFFT input
    coordinate.

    The w that comes back is already relative to ``plan.w0``
    (:func:`_w_relative`); the plane loop uses no other form. The *product*
    ``inv_lambda[c] * uvw_m[:, 2]`` is bit-identical to the removed
    ``uvw_lambda[..., 2]`` -- same two operands, same single multiply -- but
    do not read that as "the w the operator uses is unchanged". It is not:
    XLA contracts this multiply with :func:`_w_relative`'s subtract into a
    single FMA, so the ``w`` reaching the plane loop differs in the last bits
    from the pre-issue-#23 ``stored product, then subtract``, on 18-22% of
    rows. That is the whole reason ``make_plan`` has to widen its window
    boundaries (see ``planning.window_boundary_margin``), and it is why the
    branch's operator output is not bit-identical to v0.1.2's: measured
    1.0e-12 forward and 9.1e-12 adjoint in relative L2, four orders above an
    ulp, ~1e-3 of the 1e-9-scale eps the contract is written against, and just
    under the 1e-11 strategy-equivalence bound.
    """
    two_pi = 2.0 * jnp.pi
    u_ft = (two_pi * plan.pixsize_l * inv_lambda_c) * uvw_m[:, 0]
    v_ft = (two_pi * plan.pixsize_m * inv_lambda_c) * uvw_m[:, 1]
    w_rel = _w_relative(inv_lambda_c * uvw_m[:, 2], plan)
    return u_ft, v_ft, w_rel


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
    uvw_m: Array,
    inv_lambda_c: Array,
    plan: WGridderPlan,
    opts: Opts,
    w_strategy: WStrategy,
) -> Array:
    """Forward operator for a single channel: image (n_l, n_m) -> vis (n_rows,).

    ``uvw_m`` is the plan's baselines in metres, in input row order, shared by
    every channel; ``inv_lambda_c`` is this channel's ``freq / c``. The FINUFFT
    input coordinates and the w-component in wavelengths *relative to
    ``plan.w0``* are derived from the pair by :func:`_channel_ft_coords`. The
    absolute part of w is carried by ``plan.w0_screen`` and never enters this
    function.
    """
    two_pi = 2.0 * jnp.pi
    cdtype = image_c.dtype
    u_ft_c, v_ft_c, w_rel_c = _channel_ft_coords(uvw_m, inv_lambda_c, plan)
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
    uvw_m_sorted: Array,
    inv_lambda_c: Array,
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

    ``uvw_m_sorted`` is ``plan.uvw_m[plan.sort_perm]`` -- the baselines in
    metres, in **sorted** row order. The sort is by w in metres, so that
    permutation is channel-independent and the caller gathers it once for the
    whole call rather than once per channel (issue #23; v0.1.2 already did the
    same for the u/v coordinates, one channel at a time). This channel's
    coordinates come from :func:`_channel_ft_coords` on that sorted array, so
    they land in sorted-row order too.

    ``w_strategy`` selects scan-over-planes (``windowed_scan``, low memory)
    or vmap-over-planes (``windowed_vmap``, higher memory, possibly faster
    on GPU).
    """
    two_pi = 2.0 * jnp.pi
    u_sorted, v_sorted, w_rel_sorted = _channel_ft_coords(uvw_m_sorted, inv_lambda_c, plan)

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
        ``uvw_m_sorted[lo:lo+max_window_size]``).
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
    uvw_m: Array,
    inv_lambda_c: Array,
    plan: WGridderPlan,
    opts: Opts,
    w_strategy: WStrategy,
) -> Array:
    """Adjoint operator for a single channel: vis (n_rows,) -> dirty (n_l, n_m).

    See :func:`_channel_forward` for the coord-arg convention.
    """
    two_pi = 2.0 * jnp.pi
    cdtype = vis_c.dtype
    u_ft_c, v_ft_c, w_rel_c = _channel_ft_coords(uvw_m, inv_lambda_c, plan)

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
        # Windowed path: the per-channel helper takes the *sorted* baselines in
        # metres plus this channel's inv_lambda and the per-channel
        # window-start table. issue #23: the sort key is w in metres, so the
        # gather is channel-independent and belongs here, outside the channel
        # loop -- one (n_rows, 3) gather per call, where the removed
        # ``uvw_lambda_sorted`` leaf paid (n_chan, n_rows, 3) of storage and
        # v0.1.2's u/v gathers paid two (n_rows,) gathers per channel.
        uvw_m_sorted = plan.uvw_m[plan.sort_perm]  # (n_rows, 3)
        if channel_strategy == "vmap":
            vis_per_chan = jax.vmap(
                lambda im_c, il_c, ws_c: _channel_forward_windowed(
                    im_c, uvw_m_sorted, il_c, ws_c, plan, opts, w_strategy
                )
            )(image, plan.inv_lambda, plan.window_start)
        elif channel_strategy == "scan":

            def step_w(
                _: None,
                args: tuple[Array, Array, Array],
            ) -> tuple[None, Array]:
                im_c, il_c, ws_c = args
                return None, _channel_forward_windowed(
                    im_c, uvw_m_sorted, il_c, ws_c, plan, opts, w_strategy
                )

            _, vis_per_chan = jax.lax.scan(
                step_w,
                None,
                (image, plan.inv_lambda, plan.window_start),
            )
        else:
            raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")
        return vis_per_chan.T  # (n_rows, n_chan)

    # Dense path: ``plan.uvw_m`` is closed over rather than scanned/mapped --
    # it is one array for every channel, not one per channel (issue #23).
    if channel_strategy == "vmap":
        vis_per_chan = jax.vmap(
            lambda im_c, il_c: _channel_forward(im_c, plan.uvw_m, il_c, plan, opts, w_strategy)
        )(image, plan.inv_lambda)
    elif channel_strategy == "scan":

        def step(_: None, args: tuple[Array, Array]) -> tuple[None, Array]:
            im_c, il_c = args
            return None, _channel_forward(im_c, plan.uvw_m, il_c, plan, opts, w_strategy)

        _, vis_per_chan = jax.lax.scan(step, None, (image, plan.inv_lambda))
    else:
        raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")

    return vis_per_chan.T  # (n_rows, n_chan)


def dirty2vis(
    plan: WGridderPlan,
    image: Array,
    *,
    w_strategy: WStrategy = "auto",
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
        ``"auto"`` (the shipped default since issue #46) resolves to one of
        the four canonical names before the JIT boundary via
        :func:`_auto_w_strategy` -- so cache sharing is preserved, and see
        that helper's docstring for the heuristic itself. The canonical
        names, all of which override the heuristic when passed explicitly,
        are ``"dense_scan"`` (low memory), ``"dense_vmap"`` (potentially
        faster on GPU but allocates ``n_w * image_size`` peak memory) and
        ``"windowed_scan"`` / ``"windowed_vmap"``, which use the per-plane
        windowed path. The bare names ``"scan"`` / ``"vmap"`` are accepted
        as deprecated aliases.

        The default was ``"dense_scan"`` through v0.1.2, which left the
        heuristic unreachable and, on GPU, made the shipped default the
        worst of the four choices: measured on one GH200 against ducc0 on
        72 Grace cores of the same node (eps 1e-6, float64, single
        channel), ``dense_scan`` runs 1.4-5.6x *slower* than ducc0 on five
        of the six cells of issue #46's table, where what ``"auto"`` picks
        there runs 1.4-6.3x faster on all six. The exception is the
        GH200_large off30 adjoint, where the old default was about 1.16x
        faster than ducc0 -- still 2.0x off the ``dense_vmap`` column of
        the same table.

        Passing ``w_strategy="dense_scan"`` restores the pre-#46 *code
        path*: the same strategy, so the same reduction order. It does not
        restore the pre-#46 *numbers*, because the release carrying #46
        also carries #16, #23 and #43, and #16's ``nshift`` centring moved
        the numbers on its own (the worst cell against the exact DFT went
        from 0.67x to 1.47x ``epsilon``). Scoped to the strategy change
        alone, the four strategies accumulate the w-planes in a different
        order and so agree to the strategy-equivalence bound (1e-11 in
        float64, ``tests/test_strategies_equivalent.py``) rather than
        bit-for-bit. Pinning ``w_strategy`` removes the strategy from a
        cross-version comparison and nothing else; reproducing an older
        release's output needs that release pinned.
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
    uvw_m_sorted: Array,
    inv_lambda_c: Array,
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
    u_sorted, v_sorted, w_rel_sorted = _channel_ft_coords(uvw_m_sorted, inv_lambda_c, plan)

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
        # plan.window_start; the baselines get the same permutation once for
        # the whole call (see _dirty2vis_jit's note -- the sort key is w in
        # metres, so it is channel-independent).
        vis_sorted_per_chan = vis_per_chan[:, plan.sort_perm]
        uvw_m_sorted = plan.uvw_m[plan.sort_perm]  # (n_rows, 3)
        if channel_strategy == "vmap":
            dirty_per_chan = jax.vmap(
                lambda v_s_c, il_c, ws_c: _channel_adjoint_windowed(
                    v_s_c, uvw_m_sorted, il_c, ws_c, plan, opts, w_strategy
                )
            )(vis_sorted_per_chan, plan.inv_lambda, plan.window_start)
        elif channel_strategy == "scan":

            def step_w(
                _: None,
                args: tuple[Array, Array, Array],
            ) -> tuple[None, Array]:
                v_s_c, il_c, ws_c = args
                return None, _channel_adjoint_windowed(
                    v_s_c, uvw_m_sorted, il_c, ws_c, plan, opts, w_strategy
                )

            _, dirty_per_chan = jax.lax.scan(
                step_w,
                None,
                (vis_sorted_per_chan, plan.inv_lambda, plan.window_start),
            )
        else:
            raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")
    elif channel_strategy == "vmap":
        dirty_per_chan = jax.vmap(
            lambda v_c, il_c: _channel_adjoint(v_c, plan.uvw_m, il_c, plan, opts, w_strategy)
        )(vis_per_chan, plan.inv_lambda)
    elif channel_strategy == "scan":

        def step(_: None, args: tuple[Array, Array]) -> tuple[None, Array]:
            v_c, il_c = args
            return None, _channel_adjoint(v_c, plan.uvw_m, il_c, plan, opts, w_strategy)

        _, dirty_per_chan = jax.lax.scan(step, None, (vis_per_chan, plan.inv_lambda))
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
    #
    # issue #23: ``plan.n_minus_1`` is now a property (``n_minus_1_shifted -
    # nshift``, both traced-leaf and static) rather than a leaf of its own, so
    # this is one image-sized subtract added to the graph -- once per call, not
    # once per plane. It is also the one place in either operator that reads a
    # derived accessor, and deliberately so: this factor is exactly what the
    # unshifted grid exists for.
    n_grid = (plan.n_minus_1 + 1.0).astype(dirty_per_chan.real.dtype)
    safe_n = jnp.where(n_grid > 0.0, n_grid, 1.0)
    return jnp.where(n_grid > 0.0, dirty_per_chan.real / safe_n, 0.0)


def vis2dirty(
    plan: WGridderPlan,
    vis: Array,
    *,
    weights: Array | None = None,
    w_strategy: WStrategy = "auto",
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
        ``"auto"`` (the shipped default since issue #46), ``"dense_scan"``,
        ``"dense_vmap"``, ``"windowed_scan"`` or ``"windowed_vmap"``; same
        semantics as in :func:`dirty2vis`, including that an explicit name
        overrides the heuristic, that ``"dense_scan"`` restores the pre-#46
        code path but not the pre-#46 numbers (other changes in the same
        release moved those), and that the strategies agree to 1e-11 in
        float64 rather than bit-for-bit. The bare names ``"scan"`` /
        ``"vmap"`` are accepted as deprecated aliases.

        The adjoint is where the new default also changes what a *CPU*
        caller gets: on the repository's off-zenith fixtures the CPU
        heuristic picks ``"windowed_scan"`` here where the old default was
        ``"dense_scan"``. Measured on a 10-core Apple M-series (eps 1e-6,
        float64, single channel, plan and warm-up outside the timer,
        median of 9 calls) that is 1.14-1.24x faster on MWA_extended off30
        (n_w=251) -- a range across two passes, 1.24x over five interleaved
        rounds and 1.14x over a nine-round paired re-measurement -- and
        indistinguishable, within a +-4% noise floor, on MWA_compact off30
        and MeerKAT off30, whose n_w is under 20. The forward is
        unaffected: the CPU heuristic never picks a windowed forward.
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
