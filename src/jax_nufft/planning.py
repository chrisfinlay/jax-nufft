"""Plan construction for the wgridder forward and adjoint operators.

Planning is a one-shot, *non-traced* preprocessing step that turns
``(uvw, freq, image_shape, pixsize, epsilon)`` into:

  * static integers (``n_w``, ``w_kernel_width``, image dimensions, ...);
  * the ``n_minus_1`` grid evaluated on the image, plus its ``nshift``-centred
    counterpart ``n_minus_1_shifted`` (issue #16);
  * the ``phi_hat_n`` correction (precomputed via :class:`PhiHatTable`);
  * the w-plane centres in wavelengths;
  * ``uvw_lambda`` for each channel.

These quantities are bundled into :class:`WGridderPlan`, which is a frozen
dataclass registered as a JAX pytree. The numerical fields (``uvw_lambda``,
``w_centers``, ``n_minus_1``, ``phi_hat_n``) become pytree leaves and are
traced normally by ``jax.jit``; the static fields (shape ints and Python
floats) live in the pytree aux_data, so they are part of the JIT cache key
without being treated as traced inputs.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import DTypeLike

from jax_nufft._utils import SPEED_OF_LIGHT
from jax_nufft.kernel import compute_phi_hat_table, kernel_params, phi_hat_oversample_for_w

# w-direction sampling step in n-1 units: ``dw = x0 / max|n-1|``, giving
# ``n_w_inner = ceil(w_extent * max|n-1| / x0)``. ducc uses
# ``dw = 0.5 / (ofactor * max|n-1+nshift|)`` where ``ofactor`` is the
# (u,v) oversampling ratio of the chosen kernel (2.0 for the FINUFFT
# ``sigma=2`` kernel that matches our ``(W, beta=2.30*W)`` choice). That
# corresponds to ``x0 = 1/(2*ofactor) = 0.25`` for our kernel.
#
# v0.1 used a ``W``-dependent ``x0 = 1/W`` (i.e. eta_max pinned at 0.5
# regardless of W) as a safety margin for the phi_hat correction; v0.1.1
# reverts to a fixed ``x0`` matching ducc, which reduces ``n_w`` by a
# factor of ``W/4`` and accepts a wider eta-range that phi_hat is well
# conditioned on with appropriately bumped oversample (see
# :func:`jax_nufft.kernel.phi_hat_oversample_for_w`).
W_OVERSAMPLE_X0 = 0.25

# Smallest ``epsilon`` a float32 plan can plausibly deliver. Measured in the
# 2026-09 review: with every plan array in single precision the relative L2
# error against a float64 reference floors at ~3.4e-5 across the review
# fixtures, i.e. roughly ``1e-5`` once the usual few-times-epsilon slack is
# taken into account. Requests below this floor are honoured structurally
# (the plan builds, the kernel width still grows) but cannot be met
# numerically, so ``make_plan`` warns instead of silently missing the target.
#
# The floor is a fixture-level figure, not a guarantee: single precision loses
# further accuracy with large ``|w|`` (the per-plane phase is exponentiated in
# float32 with no range reduction, so its rounding error grows with the phase
# magnitude — issue #13) and for pixels near the horizon (the adjoint rebuilds
# ``n`` from ``n - 1`` by a cancelling subtraction and then divides by it —
# issue #12). Neither affects a float64 plan.
FLOAT32_EPSILON_FLOOR = 1e-5

# Veltkamp splitting constant for :func:`_two_product`: 2**ceil(53/2) + 1.
_VELTKAMP_SPLIT = float(2**27 + 1)


def _two_product(a: npt.ArrayLike, b: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Exact float64 product: returns ``(p, err)`` with ``a * b == p + err``.

    Dekker's two-product via Veltkamp splitting. ``p`` is the rounded product
    and ``err`` the exactly-representable rounding residual, so the pair
    carries the full ~106-bit product of two float64s.

    We need this because ``np.longdouble`` is *not* a wider type everywhere:
    on arm64 (this project's primary dev platform) it is an alias for float64
    with a 52-bit mantissa, so "just use longdouble" silently does nothing.
    Dekker's algorithm needs only correctly-rounded float64 ``*`` and ``+``,
    which numpy guarantees per elementwise op -- it never contracts separate
    array operations into an FMA, which is the one thing that would break the
    ``ah * bh - p`` residual step.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return _two_product_f64(a_arr, b_arr)


def _two_product_f64(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``_two_product`` on arrays already known to be float64."""
    p = a * b
    a1 = _VELTKAMP_SPLIT * a
    a_hi = a1 - (a1 - a)
    a_lo = a - a_hi
    b1 = _VELTKAMP_SPLIT * b
    b_hi = b1 - (b1 - b)
    b_lo = b - b_hi
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, err


def _phase_turns_reduced(scale: float, x: np.ndarray) -> np.ndarray:
    """``(scale * x) mod 1``, in turns, accurate to full float64 precision.

    ``exp(2πi * scale * x)`` only depends on ``scale * x`` modulo 1, but
    forming that product in plain float64 throws the information away before
    the reduction can use it: at ``scale * x ~ 1e6`` an ulp is ~2e-10, so the
    fractional part -- the only part that matters -- is left with about six
    correct digits instead of sixteen. This is exactly the "range reduction"
    problem of [Higham2002] ch. 1.

    So take the product exactly (:func:`_two_product`) and subtract the
    nearest integer *before* collapsing back to a single float:
    ``p - rint(p)`` is exact (Sterbenz), and adding the residual ``err`` then
    lands a correctly-rounded fraction in ``[-0.5, 0.5]``. Accuracy is then
    set by the fraction's own magnitude, not by ``scale * x``'s.
    """
    p, err = _two_product(scale, x)
    return (p - np.rint(p)) + err


@dataclass(frozen=True)
class WGridderPlan:
    """Pre-computed data shared by :func:`dirty2vis` and :func:`vis2dirty`."""

    # ---- static metadata (aux_data) ----
    n_l: int
    n_m: int
    n_chan: int
    n_rows: int
    n_w: int
    w_kernel_width: int
    beta: float
    epsilon: float
    pixsize_l: float
    pixsize_m: float
    w_kernel_scale: float  # half-width of the w-direction kernel, in wavelengths
    # issue #16 n-1 centring offset (ducc's ``allow_nshift`` device), in the
    # same dimensionless units as ``n - 1``:
    #
    #     nshift = -(nm1_max + nm1_min) / 2
    #
    # so that ``n_minus_1_shifted = n_minus_1 + nshift`` is centred on zero and
    # ``max|n-1+nshift|`` -- the denominator of the plane spacing
    # ``dw = x0 / max|n-1|`` -- is halved for any image containing the phase
    # centre. Static because it is a plan-shape constant that also appears in
    # the per-call compensating phase; two plans differing only in ``nshift``
    # are genuinely different operators and must not share a JIT cache entry.
    # ``0.0`` on the constant-w fast path -- see ``make_plan``.
    nshift: float
    # issue #16 follow-up: the w-range midpoint ``(w_min + w_max) / 2`` over all
    # channels, in wavelengths. Everything inside the w-plane loop works with
    # ``delta = w - w0`` instead of ``w``, so the phases the loop exponentiates
    # are bounded by the w-*extent* rather than by the absolute w. The constant
    # part ``exp(2πi w0 (n-1))`` is factored out into the ``w0_screen`` leaf
    # below. Static for the same reason as ``nshift``: it changes the operator,
    # so it belongs in the JIT cache key.
    w0: float
    # v0.1.1 windowed-scan fields:
    # ``max_window_size`` is the worst-case live-window length across all
    # (channel, plane) pairs; ``window_padding_overhead`` is
    # ``max_window_size / mean_window_size`` and is purely diagnostic.
    max_window_size: int
    window_padding_overhead: float
    # v0.1.2 w-degeneracy metadata:
    # ``w_extent`` is ``max(w_lambda) - min(w_lambda)`` over all channels (in
    # wavelengths); ``is_constant_w`` is True iff ``w_extent == 0.0`` exactly.
    # Static so a future ``is_constant_w`` fast path can be selected
    # plan-side without re-tracing.
    w_extent: float
    is_constant_w: bool
    # issue #11 precision metadata: the dtype the plan was *requested* with
    # (``make_plan(dtype=...)``), not one inferred from the uvw/freq inputs.
    # ``real_dtype`` is the dtype of every floating plan leaf and of the
    # accepted ``weights``; ``complex_dtype`` is its complex counterpart and
    # governs the image / visibility dtype the operators accept and return.
    # Static, so a float32 and a float64 plan never share a JIT cache entry.
    real_dtype: np.dtype[Any]
    complex_dtype: np.dtype[Any]

    # ---- traced arrays (pytree leaves) ----
    uvw_lambda: Array = field()  # (n_chan, n_rows, 3) — input row order
    w_centers: Array = field()  # (n_w,)
    # issue #16 follow-up: the same plane centres measured from ``w0``, i.e.
    # ``w_centers - w0`` -- but built at small magnitude on the host rather than
    # by subtracting two ~|w0| numbers at call time. That distinction is the
    # whole point: ``w_centers`` is quantised to the float64 grid *at* |w0|
    # (ulp 1.2e-10 near 1e6), so recovering the offsets from it caps their
    # resolution at 1e-10 no matter how exact the subtraction is, and a 1e-10
    # error in a plane centre is a ~6e-10 phase error -- 300x the eps=1e-12
    # contract. Computed small, they carry full relative precision.
    # This is the array the plane loop iterates over; ``w_centers`` stays
    # absolute because the window builder and the w-coverage tests read it.
    w_centers_rel: Array = field()  # (n_w,)
    n_minus_1: Array = field()  # (n_l, n_m)
    # issue #16: ``n_minus_1 + nshift``. This is the grid the per-plane phase
    # ``exp(2πi w_k (n-1+nshift))`` and the ``phi_hat`` argument
    # ``eta = (n-1+nshift) * w_kernel_scale`` are evaluated on. ``n_minus_1``
    # itself stays, and is the *only* thing the adjoint's ``1/n`` output factor
    # may use -- ``n = n_minus_1 + 1``, not ``n_minus_1_shifted + 1``.
    n_minus_1_shifted: Array = field()  # (n_l, n_m)
    # issue #16 follow-up: ``exp(2πi * w0 * (n - 1))``, complex, the image-domain
    # phase screen carrying the w-range midpoint that the plane loop no longer
    # sees. Computed once on the host with exact range reduction
    # (:func:`_phase_turns_reduced`), which is the whole point: this is the one
    # place a large phase argument legitimately survives, and reducing it here
    # costs nothing at call time. The forward multiplies its image by it, the
    # adjoint multiplies its output image by the conjugate.
    w0_screen: Array = field()  # (n_l, n_m), complex_dtype
    phi_hat_n: Array = field()  # (n_l, n_m)
    # v0.1.1 windowed-scan support:
    sort_perm: Array = field()  # (n_rows,) int — argsort(uvw[:, 2]) ascending
    uvw_lambda_sorted: Array = field()  # (n_chan, n_rows, 3) — uvw_lambda[:, sort_perm, :]
    window_start: Array = field()  # (n_chan, n_w) int — start idx in sorted array
    window_size: Array = field()  # (n_chan, n_w) int — live window length per plane
    # v0.1.2 precomputed FINUFFT coordinates (option (b) from the v0.1.2 plan):
    # ``u_finufft[c, r] = 2π · pixsize_l · uvw_lambda[c, r, 0]`` and likewise
    # for ``v_finufft``. Sorted variants are NOT stored — the windowed helpers
    # gather via ``plan.sort_perm`` + ``dynamic_slice`` at scan time.
    u_finufft: Array = field()  # (n_chan, n_rows) — input row order
    v_finufft: Array = field()  # (n_chan, n_rows) — input row order

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self.n_l, self.n_m)


def _plan_aux(plan: WGridderPlan) -> tuple[Any, ...]:
    return (
        plan.n_l,
        plan.n_m,
        plan.n_chan,
        plan.n_rows,
        plan.n_w,
        plan.w_kernel_width,
        plan.beta,
        plan.epsilon,
        plan.pixsize_l,
        plan.pixsize_m,
        plan.w_kernel_scale,
        plan.nshift,
        plan.w0,
        plan.max_window_size,
        plan.window_padding_overhead,
        plan.w_extent,
        plan.is_constant_w,
        plan.real_dtype,
        plan.complex_dtype,
    )


def _plan_unflatten(aux: tuple[Any, ...], children: tuple[Array, ...]) -> WGridderPlan:
    (
        n_l,
        n_m,
        n_chan,
        n_rows,
        n_w,
        w_kernel_width,
        beta,
        epsilon,
        pixsize_l,
        pixsize_m,
        w_kernel_scale,
        nshift,
        w0,
        max_window_size,
        window_padding_overhead,
        w_extent,
        is_constant_w,
        real_dtype,
        complex_dtype,
    ) = aux
    (
        uvw_lambda,
        w_centers,
        w_centers_rel,
        n_minus_1,
        n_minus_1_shifted,
        w0_screen,
        phi_hat_n,
        sort_perm,
        uvw_lambda_sorted,
        window_start,
        window_size,
        u_finufft,
        v_finufft,
    ) = children
    return WGridderPlan(
        n_l=n_l,
        n_m=n_m,
        n_chan=n_chan,
        n_rows=n_rows,
        n_w=n_w,
        w_kernel_width=w_kernel_width,
        beta=beta,
        epsilon=epsilon,
        pixsize_l=pixsize_l,
        pixsize_m=pixsize_m,
        w_kernel_scale=w_kernel_scale,
        nshift=nshift,
        w0=w0,
        max_window_size=max_window_size,
        window_padding_overhead=window_padding_overhead,
        w_extent=w_extent,
        is_constant_w=is_constant_w,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        uvw_lambda=uvw_lambda,
        w_centers=w_centers,
        w_centers_rel=w_centers_rel,
        n_minus_1=n_minus_1,
        n_minus_1_shifted=n_minus_1_shifted,
        w0_screen=w0_screen,
        phi_hat_n=phi_hat_n,
        sort_perm=sort_perm,
        uvw_lambda_sorted=uvw_lambda_sorted,
        window_start=window_start,
        window_size=window_size,
        u_finufft=u_finufft,
        v_finufft=v_finufft,
    )


jax.tree_util.register_pytree_node(
    WGridderPlan,
    flatten_func=lambda p: (
        (
            p.uvw_lambda,
            p.w_centers,
            p.w_centers_rel,
            p.n_minus_1,
            p.n_minus_1_shifted,
            p.w0_screen,
            p.phi_hat_n,
            p.sort_perm,
            p.uvw_lambda_sorted,
            p.window_start,
            p.window_size,
            p.u_finufft,
            p.v_finufft,
        ),
        _plan_aux(p),
    ),
    unflatten_func=_plan_unflatten,
)


def _coerce_uvw_freq_dtype(
    uvw: np.ndarray, freq: np.ndarray, real_dtype: np.dtype[Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the shape/kind of ``uvw`` / ``freq`` and cast them to the plan dtype.

    Since issue #11 the plan's precision is *requested* through
    ``make_plan(dtype=...)`` rather than inherited from the inputs, so this
    helper no longer resolves a dtype: it only checks shapes and rejects
    non-real input, then casts unconditionally. Passing float64 ``uvw`` to a
    float32 plan is therefore explicitly allowed — planning inputs are
    metadata, not the user's data, and the cast is what the caller asked for
    by naming the plan dtype.
    """
    uvw_arr = np.asarray(uvw)
    freq_arr = np.asarray(freq)
    if uvw_arr.ndim != 2 or uvw_arr.shape[1] != 3:
        raise ValueError(f"uvw must have shape (N, 3); got {uvw_arr.shape}")
    if freq_arr.ndim != 1:
        raise ValueError(f"freq must have shape (Nchan,); got {freq_arr.shape}")
    # "b" (bool), "i"/"u" (integers) and "f" all cast losslessly enough for
    # coordinates; complex/object/str input is a mistake worth naming.
    for name, arr in (("uvw", uvw_arr), ("freq", freq_arr)):
        if arr.dtype.kind not in "bfiu":
            raise TypeError(f"{name} must be a real numeric array; got dtype {arr.dtype}")
    return uvw_arr.astype(real_dtype), freq_arr.astype(real_dtype)


def _resolve_plan_dtypes(dtype: DTypeLike) -> tuple[np.dtype[Any], np.dtype[Any]]:
    """Turn the requested ``dtype`` into the plan's ``(real, complex)`` pair.

    Also enforces the x64 guard of issue #11: float64 is only meaningful with
    ``jax.config.jax_enable_x64`` on. With x64 off (JAX's default) every array
    handed to ``jnp.asarray`` is truncated to float32 *silently*, so a
    "float64" plan would quietly be a float32 one whose error floors near
    3.4e-5. Refuse rather than degrade.

    This runs before any array is built or validated, because the guard has
    to fire before ``jnp.asarray`` gets a chance to truncate anything. The
    float32 accuracy *warning* deliberately does not live here — see
    :func:`_warn_if_below_float32_floor`.
    """
    real_dtype: np.dtype[Any] = np.dtype(dtype)
    complex_dtype: np.dtype[Any]
    if real_dtype == np.dtype(np.float32):
        complex_dtype = np.dtype(np.complex64)
    elif real_dtype == np.dtype(np.float64):
        complex_dtype = np.dtype(np.complex128)
    else:
        raise ValueError(
            f"unsupported plan dtype {real_dtype}; make_plan accepts jnp.float32 or jnp.float64"
        )

    # Ask JAX what it would actually do with a float64 array rather than
    # reading ``jax.config.jax_enable_x64``: ``canonicalize_dtype`` *is* the
    # step that silently truncates, so this cannot drift from the behaviour
    # being guarded against.
    x64_enabled = np.dtype(jax.dtypes.canonicalize_dtype(np.float64)) == np.dtype(np.float64)
    if real_dtype == np.dtype(np.float64) and not x64_enabled:
        raise ValueError(
            "make_plan(dtype=float64) requires jax_enable_x64, which is off in this "
            "process: JAX would truncate every plan array to float32 without "
            "telling you, and the accuracy would floor near 3.4e-5. Either enable "
            "it (jax.config.update('jax_enable_x64', True), or JAX_ENABLE_X64=1 in "
            "the environment, before the first JAX array is created), or pass "
            "dtype=jnp.float32 to opt into single precision deliberately "
            "(achievable epsilon ~ 1e-5)."
        )

    return real_dtype, complex_dtype


def _warn_if_below_float32_floor(real_dtype: np.dtype[Any], epsilon: float) -> None:
    """Warn when a float32 plan is asked for an ``epsilon`` it cannot reach.

    Called *after* every input-rejection check in ``make_plan``, with no
    exceptions -- ``uvw`` / ``freq`` shape and kind, ``epsilon``,
    ``image_shape``, ``pixsize_l`` / ``pixsize_m``, ``phi_hat_n_fine`` /
    ``phi_hat_oversample``, and (since issue #9's review follow-up)
    ``kernel_params(epsilon)``'s own ``ValueError`` for an epsilon below
    1e-14 -- and deliberately so: the suite (and many downstream users) run
    with ``filterwarnings = ["error"]``, which turns this into a raised
    ``UserWarning``. Emitting it earlier would mean a caller hits the wrong
    diagnostic for their actual bug: a malformed ``uvw`` at a tight epsilon
    would get an accuracy complaint instead of the shape error,
    ``dtype=jnp.float32`` at an epsilon ``kernel_params`` refuses outright
    (< 1e-14) would get this warning's "the plan will build" claim instead of
    the ``ValueError`` saying it can't, and the same is true of an invalid
    ``phi_hat_n_fine`` / ``phi_hat_oversample`` -- ``compute_phi_hat_table``
    (kernel.py) validates those too, but it is only called much later in
    ``make_plan``, well after this warning would already have fired. Only a
    plan that is otherwise buildable gets to hear about its accuracy.
    """
    if real_dtype == np.dtype(np.float32) and epsilon < FLOAT32_EPSILON_FLOOR:
        warnings.warn(
            f"epsilon={epsilon:g} is below the accuracy a float32 plan can deliver "
            "(measured relative-error floor ~3.4e-5, i.e. an achievable epsilon of "
            "about 1e-5). The plan will build, but the result will not reach the "
            "requested epsilon. Use dtype=jnp.float64 with jax_enable_x64 on for "
            "epsilon < 1e-5.",
            UserWarning,
            stacklevel=3,
        )


def make_plan(
    uvw: np.ndarray,
    freq: np.ndarray,
    image_shape: tuple[int, int],
    pixsize_l: float,
    pixsize_m: float,
    epsilon: float,
    *,
    dtype: DTypeLike = jnp.float64,
    phi_hat_n_fine: int = 4096,
    phi_hat_oversample: int | None = None,
    _force_generic: bool = False,
) -> WGridderPlan:
    """Build the wgridder plan for the given (uvw, freq, image, epsilon).

    The returned plan can be passed to :func:`jax_nufft.dirty2vis` and
    :func:`jax_nufft.vis2dirty`. All planning math runs on the host (numpy);
    the resulting numerical arrays live as JAX device arrays so that the
    JIT-compiled operators see them as constants.

    ``dtype`` (``jnp.float64``, the default, or ``jnp.float32``) sets the
    precision of the whole plan: ``uvw`` and ``freq`` are *cast* to it, every
    floating plan leaf carries it, and the operators accept and return the
    matching real/complex dtypes (see ``plan.real_dtype`` /
    ``plan.complex_dtype``). float64 requires ``jax_enable_x64``; float32
    warns for ``epsilon < 1e-5``, which single precision cannot reach.

    ``_force_generic`` is a private test-only escape hatch that skips the
    v0.1.2 constant-w fast path even when ``w_extent == 0``, building the
    generic-shape plan (``n_w == w_kernel_width + 1``) for direct
    fast-vs-generic comparison in tests. Not part of the public API; do
    not use in application code.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0; got {epsilon}")
    n_l, n_m = image_shape
    if n_l <= 0 or n_m <= 0:
        raise ValueError(f"image_shape must be positive; got {image_shape}")
    if pixsize_l <= 0 or pixsize_m <= 0:
        raise ValueError(f"pixsize_l and pixsize_m must be > 0; got ({pixsize_l}, {pixsize_m})")
    # Mirrors compute_phi_hat_table's own checks (kernel.py), but validated
    # here too: that function is only called later, after
    # _warn_if_below_float32_floor, so without this a bad phi_hat_n_fine /
    # phi_hat_oversample would reach the accuracy warning first -- see the
    # ordering comment below.
    if phi_hat_n_fine % 2 != 0:
        raise ValueError(f"phi_hat_n_fine must be even; got {phi_hat_n_fine}")
    if phi_hat_oversample is not None and phi_hat_oversample < 1:
        raise ValueError(f"phi_hat_oversample must be >= 1; got {phi_hat_oversample}")

    # Resolve precision before touching any array: the x64 guard has to fire
    # before ``jnp.asarray`` gets a chance to truncate anything.
    real_dtype, complex_dtype = _resolve_plan_dtypes(dtype)
    uvw_arr, freq_arr = _coerce_uvw_freq_dtype(uvw, freq, real_dtype)
    n_rows = uvw_arr.shape[0]
    n_chan = freq_arr.shape[0]

    # --- kernel parameters ---
    # Every input-rejection ValueError -- epsilon <= 0, image_shape,
    # pixsize, phi_hat_n_fine, phi_hat_oversample above, and the
    # epsilon-too-tight check inside kernel_params here -- must run before
    # _warn_if_below_float32_floor, with no exceptions: with
    # filterwarnings = ["error"] that warning becomes an exception too, and
    # if it fired first, a caller asking for dtype=jnp.float32 at an epsilon
    # kernel_params flatly cannot honour (< 1e-14), or at any epsilon with a
    # malformed phi_hat_n_fine / phi_hat_oversample, would see a UserWarning
    # claiming "the plan will build" instead of the ValueError that
    # actually applies. So this call -- and its result, reused below rather
    # than called again -- has to come first.
    w_kernel_width, beta = kernel_params(epsilon)
    # Only now, with every argument known good and the epsilon itself
    # confirmed reachable, complain about accuracy the requested precision
    # cannot deliver (see _warn_if_below_float32_floor).
    _warn_if_below_float32_floor(real_dtype, epsilon)

    # --- n - 1 grid (numpy, host-side) ---
    # For pixels inside the unit disc (l^2 + m^2 <= 1) this is the usual
    # n - 1 = sqrt(1 - l^2 - m^2) - 1 with values in [-1, 0]. For pixels
    # outside the disc we use ducc's analytic extension
    # n - 1 = -sqrt(l^2 + m^2 - 1) - 1 (values < -1), so that full-sky
    # imaging matches ducc's wgridder pixel-by-pixel.
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    ll = (i * pixsize_l)[:, None]
    mm = (j * pixsize_m)[None, :]
    eps_lm = ll * ll + mm * mm
    inside_disc = eps_lm <= 1.0
    inside_val = np.sqrt(np.where(inside_disc, 1.0 - eps_lm, 0.0)) - 1.0
    outside_val = -np.sqrt(np.where(inside_disc, 0.0, eps_lm - 1.0)) - 1.0
    n_minus_1_np = np.where(inside_disc, inside_val, outside_val).astype(real_dtype)

    # --- issue #16: centre the n-1 range (ducc's ``allow_nshift`` device) ---
    # The w-phase factor obeys the *exact* identity
    #
    #     exp(2πi w (n-1)) = exp(2πi w (n-1+s)) · exp(-2πi w s)
    #
    # for any constant ``s``. Only ``max|n-1|`` enters the plane spacing
    # ``dw = x0 / max|n-1|`` (see W_OVERSAMPLE_X0), so picking the ``s`` that
    # centres the ``n-1`` range on zero,
    #
    #     s = nshift = -(nm1_max + nm1_min) / 2,
    #
    # replaces ``max(|nm1_max|, |nm1_min|)`` by ``(nm1_max - nm1_min) / 2``.
    # For any image containing the phase centre ``nm1_max == 0`` exactly (the
    # centre pixel has n == 1) and ``nm1_min < 0``, so that is exactly half the
    # old denominator: ``dw`` doubles and ``n_w_inner`` halves. Images that do
    # not contain the phase centre do strictly better still.
    #
    # The extremes are taken over the *whole* grid, including the analytic
    # extension outside the unit disc (``n-1 = -sqrt(l²+m²-1) - 1 < -1``,
    # which is what dominates a wide-FoV fixture like EDA2's 120°) -- those
    # pixels are gridded like any other and their phase has to be represented
    # just as accurately.
    #
    # The price is the compensating factor ``exp(-2πi w nshift)``: a single
    # per-visibility complex multiply per call, applied to the forward output
    # and (conjugated) to the adjoint input in wgridder.py. It is *not* a
    # per-plane cost, which is why halving the plane count is a straight win.
    #
    # Note ``w_kernel_scale = dw * W / 2`` doubles alongside ``dw`` while
    # ``max|n-1+nshift|`` halves, so ``eta_max = max|n-1+nshift| *
    # w_kernel_scale`` is unchanged and the phi_hat table's conditioning is
    # exactly as before.
    nm1_max = float(np.max(n_minus_1_np))
    nm1_min = float(np.min(n_minus_1_np))
    nshift = -(nm1_max + nm1_min) / 2.0
    n_minus_1_shifted_np = (n_minus_1_np + nshift).astype(real_dtype)
    # Read the plane-spacing denominator off the two extremes rather than
    # recomputing ``max(abs(n_minus_1_shifted_np))``: it is by construction the
    # quantity ``nshift`` was chosen to minimise, and taking it from the same
    # two numbers keeps the two definitions from drifting under rounding.
    max_abs_nm1 = max(abs(nm1_max + nshift), abs(nm1_min + nshift))
    if max_abs_nm1 == 0.0:
        # Pathological: a 1x1 image at zenith (nm1_max == nm1_min == 0, so
        # nshift == 0 too). Force a tiny but non-zero value so that downstream
        # ratios are well-defined.
        max_abs_nm1 = 1e-12

    # --- per-channel uvw in wavelengths ---
    uvw_lambda_np = (uvw_arr[None, :, :] * (freq_arr / SPEED_OF_LIGHT)[:, None, None]).astype(
        real_dtype
    )
    # Worst-case w over all channels.
    w_lambda_all = uvw_lambda_np[..., 2]  # (Nchan, Nrow)
    w_min_all = float(np.min(w_lambda_all))
    w_max_all = float(np.max(w_lambda_all))
    w_extent = w_max_all - w_min_all
    if w_extent < 0:
        # Should be unreachable with a real telescope; guard anyway.
        raise AssertionError("internal: negative w-extent")

    # --- issue #16 follow-up: centre w as well as n-1 ---
    # nshift alone is not enough. After it, the phase the plane loop
    # exponentiates is ``2πi w (n-1+s)``, and at the phase centre ``n-1 == 0``
    # so that phase is ``2πi w s`` -- large whenever ``|w|`` is large -- while
    # the compensating factor is ``-2πi w s``. The two cancel, so the result is
    # a small number reconstructed from two large ones and the float64 rounding
    # error of the *large* phases survives into it ([Higham2002] ch. 1). Before
    # nshift this particular pixel was safe (``n-1 == 0`` made the plane phase
    # exactly zero however large ``w`` was), which is why nshift turned a
    # latent weakness into a measurable one. Measured on a delta image at the
    # phase centre over a 120-degree field at eps=1e-12, relative L2 in units
    # of eps, for w_lambda = offset + [-10, 10]:
    #
    #     offset        0      1e4      1e6
    #     pre-nshift  0.34     1.10    65.52
    #     nshift      0.58    15.70  1334.85
    #     + centring  0.57     0.57     0.57
    #
    # Note the last row is flat: with both w and n-1 centred the accuracy no
    # longer depends on the absolute w at all, which is stronger than what the
    # pre-nshift code gave.
    #
    # The cure is the same trick applied to w. Split off the range midpoint:
    #
    #     exp(2πi w (n-1)) = exp(2πi w0 (n-1)) · exp(2πi (w-w0) (n-1))
    #
    # The first factor depends on the pixel but on neither the visibility nor
    # the plane, so it leaves the loop entirely and becomes the ``w0_screen``
    # image-domain phase screen (applied once per call). The second uses
    # ``delta = w - w0``, bounded by half the w-extent, and combines with the
    # nshift identity to give
    #
    #     exp(2πi delta (n-1+s)) · exp(-2πi delta s)
    #
    # where *both* arguments are small. Note the plane spacing, ``n_w``,
    # ``w_centers`` and the window builder are untouched by this: only which
    # origin the phases are measured from changes.
    #
    # ``w_centers`` deliberately stays in *absolute* wavelengths -- it is the
    # documented plan field the window builder and the coverage tests read, and
    # the operators subtract ``w0`` at the JIT boundary instead. In float64
    # that subtraction is exact (Sterbenz: the two operands are within an
    # w-extent of each other), so nothing is lost by deferring it.
    w0 = (w_min_all + w_max_all) / 2.0

    is_constant_w_val = bool(w_extent == 0.0)
    use_fast_path = is_constant_w_val and not _force_generic

    if use_fast_path:
        # --- v0.1.2 constant-w fast path ---
        # All visibilities share a single w-value (in wavelengths, across all
        # channels). The dense path would set dw=0 and produce NaN at call
        # time via z=0/0; here we collapse to one w-plane at the constant
        # value. Math: V = NUFFT[I * exp(2πi w_const (n-1))]. No kernel
        # discretisation, no sum approximating an integral, so phi_hat_n
        # carries no correction (the operator code multiplies by
        # phi((w_lambda-w_k)/scale) = phi(0) = 1, and there is nothing to
        # invert).
        #
        # issue #16: this path deliberately takes ``nshift = 0.0``. nshift
        # exists to shrink ``n_w_inner = ceil(w_extent * max|n-1+nshift| / x0)``,
        # and here ``n_w`` is pinned at 1 by construction, ``phi_hat_n`` is
        # ones, and there is a single ``w_centers`` entry -- so it cannot buy
        # a single plane. It would only add one complex multiply per
        # visibility per call to the cheapest path in the library. Keeping it
        # at zero is *internally consistent*, which is the part that matters:
        # ``n_minus_1_shifted == n_minus_1``, so the plane phase below is the
        # unshifted one, and the compensating factor in wgridder.py is
        # exp(0) = 1 (that code skips the multiply entirely when
        # ``plan.nshift == 0.0``). Shifted phase and compensation are both
        # off, never one without the other.
        #
        # The w-centring follow-up makes that choice moot rather than merely
        # harmless: here ``w0 == w_min_all == w_max_all == w_centers[0]``, so
        # ``delta = w - w0`` is exactly 0 for every row and for the single
        # plane. The plane phase ``2π delta (n-1+s)`` and the compensation
        # ``-2π delta s`` are then both identically zero *whatever* nshift is,
        # and the entire w-dependence is carried by ``w0_screen`` -- which is
        # range-reduced on the host, so this path also stops exponentiating an
        # unreduced ``2π w_const (n-1)`` at call time. That is a strict
        # accuracy gain for constant-w data at large |w|.
        nshift = 0.0
        n_minus_1_shifted_np = n_minus_1_np
        w_constant = w_min_all  # = w_max_all by construction
        n_w = 1
        # Any nonzero ``w_kernel_scale`` makes ``z = (w_lambda - w_k)/scale``
        # well-defined; with z=0 the kernel weight phi(0, beta) = 1 by
        # construction so the value doesn't affect output.
        w_kernel_scale = 1.0
        w_centers_np = np.array([w_constant], dtype=real_dtype)
        # w0 == w_constant here, so the one plane sits exactly at the origin.
        w_centers_rel_np = np.zeros(1, dtype=real_dtype)
        phi_hat_n_np = np.ones((n_l, n_m), dtype=real_dtype)
        # Single-window builder: one window per channel, covering all rows.
        sort_perm_np = np.arange(n_rows, dtype=np.int32)
        uvw_lambda_sorted_np = uvw_lambda_np
        window_start_np = np.zeros((n_chan, 1), dtype=np.int32)
        window_size_np = np.full((n_chan, 1), n_rows, dtype=np.int32)
        max_window_size = max(n_rows, 1)
        window_padding_overhead = 1.0
    else:
        # --- number of w-planes ---
        # Sample w with step dw = x0 / max|n-1|, matching ducc's choice for
        # ofactor=2 kernels (see W_OVERSAMPLE_X0). This is independent of W.
        #
        # ``n_w = n_w_inner + W``: only the W half-widths of kernel overhang at
        # the two ends of the w-range depend on the kernel, so the plane count
        # grows by exactly one plane per unit of W. That is the whole cost of
        # issue #9's wider width rule, and it is a rounding error wherever the
        # w-range is what sets n_w: on the review fixtures, going from
        # eps = 1e-3 (W = 4) to eps = 1e-12 (W = 13) takes MWA_extended off30
        # from 492 to 501 planes (+1.8%) but MWA_compact zenith, where
        # n_w_inner is 1, from 5 to 14 (+180%). Per epsilon step the increase
        # is the width step itself: +1 plane at 1e-6, 1e-7, 1e-8 and +3 at
        # 1e-12 relative to the pre-#9 rule (worst relative case on the
        # fixtures: MWA_compact zenith at 1e-12, 11 -> 14 planes, +27%).
        x0 = W_OVERSAMPLE_X0
        n_w_inner = math.ceil(w_extent * max_abs_nm1 / x0)
        n_w_inner = max(n_w_inner, 1)  # always have at least one interior step
        n_w = n_w_inner + w_kernel_width

        # --- w-plane centres (spec sec 4.2 step 4) ---
        if w_extent == 0.0:
            # Reachable only via ``_force_generic`` on constant-w data
            # (the v0.1.2 fast path normally handles w_extent==0). Pick dw
            # matching the "one inner step" sampling we would use for the
            # smallest non-zero w_extent so the resulting plan is well-
            # defined (in particular, ``w_kernel_scale > 0`` avoids the
            # ``z = 0/0`` NaN that the dense operator would otherwise hit).
            dw = x0 / max_abs_nm1
        else:
            dw = w_extent / n_w_inner
        w_kernel_scale = dw * w_kernel_width / 2.0
        k = np.arange(n_w)
        # Build the centres *relative to w0* first, at small magnitude, then add
        # w0 back for the public absolute field. Doing it the other way round
        # (absolute first, subtract later) rounds each centre onto the float64
        # grid at |w0| and loses the low bits of the offset for good -- see the
        # ``w_centers_rel`` field comment. ``w_min_all - w0`` is exact
        # (Sterbenz), so the relative centres are as accurate as ``dw``.
        w_centers_rel_np = (w_min_all - w0) + (k - w_kernel_width / 2.0) * dw
        w_centers_np = (w0 + w_centers_rel_np).astype(real_dtype)
        w_centers_rel_np = w_centers_rel_np.astype(real_dtype)

        # --- phi_hat_n (precomputed on the n-1 grid) ---
        # Argument to phi_hat is eta = (n - 1) * scale, where scale is the kernel
        # half-width in wavelengths. With the v0.1.1 fixed-x0 sampling the
        # nominal eta_max is x0 * W / 2 = W/8 (W=4 -> 0.5, W=8 -> 1.0,
        # W=10 -> 1.25). We size the phi_hat oversample to keep cubic-Lagrange
        # interpolation accurate on that wider range.
        # issue #16: evaluated on the *shifted* grid, because the shifted
        # phase is what the per-plane kernel sum actually approximates.
        eta_n = n_minus_1_shifted_np * w_kernel_scale
        eta_max_request = max(float(np.max(np.abs(eta_n))), 1e-9)
        if phi_hat_oversample is None:
            phi_hat_oversample = phi_hat_oversample_for_w(w_kernel_width)
        phi_hat_table = compute_phi_hat_table(
            beta=beta,
            eta_max_request=eta_max_request,
            n_fine=phi_hat_n_fine,
            oversample=phi_hat_oversample,
        )
        # The image-domain correction needs a (W/2) factor to convert the discrete
        # w-plane sum used in the gridder into the continuous w-integral that
        # corresponds to the "literal sum" definition of the visibility (matching
        # ducc's dirty2vis). Concretely: sum_k phi((w-w_k)/scale) g(w_k) ~= (1/dw)
        # * integral phi((w-w')/scale) g(w') dw', and dw = 2*scale/W, so the
        # discrete sum picks up a (scale/dw) = W/2 multiplier relative to the
        # continuous-FT-based correction phi_hat(scale * (n-1)).
        phi_hat_dim = phi_hat_table.evaluate(eta_n)
        phi_hat_n_np = ((w_kernel_width / 2.0) * phi_hat_dim).astype(real_dtype)
        if not np.all(phi_hat_n_np > 0):
            raise ValueError(
                "phi_hat_n contains non-positive values; planning would produce "
                "infinite/garbage corrections. Try a larger oversample or a "
                "smaller epsilon."
            )

        # --- v0.1.1 windowed-scan builder ---
        # Sort visibilities by w in metres (frequency-independent). The same
        # permutation serves every channel because scaling by ``freq[c]/c`` is
        # strictly positive and so monotonic.
        sort_perm_np = np.argsort(uvw_arr[:, 2], kind="stable").astype(np.int32)
        uvw_lambda_sorted_np = uvw_lambda_np[:, sort_perm_np, :]

        # For each (channel, plane), the contributing rows are those with
        # ``|w_lambda - w_k| < W/2 * dw = w_kernel_scale`` (the kernel support
        # cutoff, where ``phi(z) = 0`` outside). After sorting, this is a
        # contiguous slice; ``searchsorted`` finds the boundaries.
        window_start_np = np.zeros((n_chan, n_w), dtype=np.int32)
        window_size_np = np.zeros((n_chan, n_w), dtype=np.int32)
        half_W_dw = w_kernel_scale  # = (W/2) * dw, the kernel support half-width
        # issue #16 follow-up: compute the boundaries in the SAME coordinate the
        # operators use for ``z``, i.e. relative to ``w0``, not in absolute
        # wavelengths. The two differ by the rounding of ``w0 + w_centers_rel``
        # onto the float64 grid at |w0|, and that is not negligible here: a row
        # sitting within an ulp of a window edge gets ``phi(z = +/-1) =
        # exp(-beta)`` from the dense path but is dropped entirely by a windowed
        # path whose boundary rounded the other way. exp(-beta) is ~1e-7 at W=7,
        # which is exactly the windowed-vs-dense mismatch tests/
        # test_boundary_planes.py measures at a 1e-12 tolerance.
        w_centers64 = w_centers_rel_np.astype(np.float64)
        for c in range(n_chan):
            w_lambda_c = uvw_lambda_sorted_np[c, :, 2].astype(np.float64) - w0
            # ``side="left"``  for lower bound, ``side="right"`` for upper bound
            # gives a half-open interval [lo, hi) of strictly-inside rows. Rows
            # exactly at ``w_k +/- half_W_dw`` have phi(z=+/-1) = exp(-beta),
            # numerically tiny but nonzero — including them costs at most one
            # extra row per side and avoids edge surprises.
            lo = np.searchsorted(w_lambda_c, w_centers64 - half_W_dw, side="left")
            hi = np.searchsorted(w_lambda_c, w_centers64 + half_W_dw, side="right")
            window_start_np[c] = lo.astype(np.int32)
            window_size_np[c] = (hi - lo).astype(np.int32)

        max_window_size = int(window_size_np.max(initial=0))
        # mean_window_size: ignore empty windows (entirely outside data range)
        # so that the diagnostic isn't dominated by edge planes.
        nonzero_windows = window_size_np[window_size_np > 0]
        if nonzero_windows.size:
            mean_window_size = float(nonzero_windows.mean())
            window_padding_overhead = max_window_size / mean_window_size
        else:
            window_padding_overhead = 1.0
        # Clamp max_window_size to at least 1 so the static dynamic_slice
        # shape is well-defined (e.g. n_rows >= 1 always).
        max_window_size = max(max_window_size, 1)

    # v0.1.2 Part 3: precompute the per-call (2π * pixsize) scaling on u, v
    # so the channel helpers can read them directly from the plan instead of
    # re-deriving them on every JIT invocation. Sorted variants are NOT
    # stored (option (b) in the v0.1.2 plan); the windowed helpers gather
    # via plan.sort_perm at scan time.
    # The explicit ``astype`` is not redundant: ``pixsize_*`` are Python
    # floats, and value-based promotion rules differ across numpy versions,
    # so pin the result to the plan's dtype rather than trusting the scalar
    # to stay weak.
    two_pi = 2.0 * np.pi
    u_finufft_np = ((two_pi * pixsize_l) * uvw_lambda_np[..., 0]).astype(real_dtype)
    v_finufft_np = ((two_pi * pixsize_m) * uvw_lambda_np[..., 1]).astype(real_dtype)

    # --- issue #16 follow-up: the w0 image-domain phase screen ---
    # ``exp(2πi w0 (n-1))``. This is the one place a large phase argument
    # legitimately remains, so it is the one place that gets exact range
    # reduction: ``_phase_turns_reduced`` takes ``w0 * (n-1)`` modulo 1 with a
    # double-double product, keeping full float64 accuracy in the fraction no
    # matter how large ``w0`` is. Doing this on the host is free -- it happens
    # once per plan, not once per call, and never inside the plane loop.
    #
    # Deliberately computed from the float64 ``n_minus_1`` grid rather than the
    # (possibly float32) plan leaf: the reduction is only worth doing if its
    # input carries full precision, and the result is cast to the plan's
    # complex dtype at the end.
    n_minus_1_f64 = np.where(inside_disc, inside_val, outside_val)
    w0_screen_np = np.exp(2j * np.pi * _phase_turns_reduced(w0, n_minus_1_f64)).astype(
        complex_dtype
    )

    return WGridderPlan(
        n_l=int(n_l),
        n_m=int(n_m),
        n_chan=int(n_chan),
        n_rows=int(n_rows),
        n_w=int(n_w),
        w_kernel_width=int(w_kernel_width),
        beta=float(beta),
        epsilon=float(epsilon),
        pixsize_l=float(pixsize_l),
        pixsize_m=float(pixsize_m),
        w_kernel_scale=float(w_kernel_scale),
        nshift=float(nshift),
        w0=float(w0),
        max_window_size=int(max_window_size),
        window_padding_overhead=float(window_padding_overhead),
        w_extent=float(w_extent),
        # ``is_constant_w`` reflects the plan SHAPE (n_w==1 collapse), not
        # the data shape. When ``_force_generic`` builds a generic plan
        # over constant-w data, this stays False so downstream selectors
        # (Part 4 auto, etc.) see the actual structure of this plan.
        is_constant_w=use_fast_path,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        uvw_lambda=jnp.asarray(uvw_lambda_np),
        w_centers=jnp.asarray(w_centers_np),
        w_centers_rel=jnp.asarray(w_centers_rel_np),
        n_minus_1=jnp.asarray(n_minus_1_np),
        n_minus_1_shifted=jnp.asarray(n_minus_1_shifted_np),
        w0_screen=jnp.asarray(w0_screen_np),
        phi_hat_n=jnp.asarray(phi_hat_n_np),
        sort_perm=jnp.asarray(sort_perm_np),
        uvw_lambda_sorted=jnp.asarray(uvw_lambda_sorted_np),
        window_start=jnp.asarray(window_start_np),
        window_size=jnp.asarray(window_size_np),
        u_finufft=jnp.asarray(u_finufft_np),
        v_finufft=jnp.asarray(v_finufft_np),
    )


__all__ = ["WGridderPlan", "make_plan"]
