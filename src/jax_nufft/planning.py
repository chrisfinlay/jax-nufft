"""Plan construction for the wgridder forward and adjoint operators.

Planning is a one-shot, *non-traced* preprocessing step that turns
``(uvw, freq, image_shape, pixsize, epsilon)`` into:

  * static integers (``n_w``, ``w_kernel_width``, image dimensions, ...);
  * the ``n_minus_1`` grid evaluated on the image;
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
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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

    # ---- traced arrays (pytree leaves) ----
    uvw_lambda: Array = field()  # (n_chan, n_rows, 3) — input row order
    w_centers: Array = field()  # (n_w,)
    n_minus_1: Array = field()  # (n_l, n_m)
    phi_hat_n: Array = field()  # (n_l, n_m)
    # v0.1.1 windowed-scan support:
    sort_perm: Array = field()  # (n_rows,) int — argsort(uvw[:, 2]) ascending
    uvw_lambda_sorted: Array = field()  # (n_chan, n_rows, 3) — uvw_lambda[:, sort_perm, :]
    window_start: Array = field()  # (n_chan, n_w) int — start idx in sorted array
    window_size: Array = field()  # (n_chan, n_w) int — live window length per plane

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
        plan.max_window_size,
        plan.window_padding_overhead,
        plan.w_extent,
        plan.is_constant_w,
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
        max_window_size,
        window_padding_overhead,
        w_extent,
        is_constant_w,
    ) = aux
    (
        uvw_lambda,
        w_centers,
        n_minus_1,
        phi_hat_n,
        sort_perm,
        uvw_lambda_sorted,
        window_start,
        window_size,
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
        max_window_size=max_window_size,
        window_padding_overhead=window_padding_overhead,
        w_extent=w_extent,
        is_constant_w=is_constant_w,
        uvw_lambda=uvw_lambda,
        w_centers=w_centers,
        n_minus_1=n_minus_1,
        phi_hat_n=phi_hat_n,
        sort_perm=sort_perm,
        uvw_lambda_sorted=uvw_lambda_sorted,
        window_start=window_start,
        window_size=window_size,
    )


jax.tree_util.register_pytree_node(
    WGridderPlan,
    flatten_func=lambda p: (
        (
            p.uvw_lambda,
            p.w_centers,
            p.n_minus_1,
            p.phi_hat_n,
            p.sort_perm,
            p.uvw_lambda_sorted,
            p.window_start,
            p.window_size,
        ),
        _plan_aux(p),
    ),
    unflatten_func=_plan_unflatten,
)


def _coerce_uvw_freq_dtype(
    uvw: np.ndarray, freq: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    """Resolve a single shared real dtype for uvw and freq."""
    uvw_arr = np.asarray(uvw)
    freq_arr = np.asarray(freq)
    if uvw_arr.ndim != 2 or uvw_arr.shape[1] != 3:
        raise ValueError(f"uvw must have shape (N, 3); got {uvw_arr.shape}")
    if freq_arr.ndim != 1:
        raise ValueError(f"freq must have shape (Nchan,); got {freq_arr.shape}")
    if uvw_arr.dtype.kind != "f" or freq_arr.dtype.kind != "f":
        # Promote integer/object inputs to float64.
        uvw_arr = uvw_arr.astype(np.float64)
        freq_arr = freq_arr.astype(np.float64)
    out_dtype = np.result_type(uvw_arr.dtype, freq_arr.dtype)
    return uvw_arr.astype(out_dtype), freq_arr.astype(out_dtype), out_dtype


def make_plan(
    uvw: np.ndarray,
    freq: np.ndarray,
    image_shape: tuple[int, int],
    pixsize_l: float,
    pixsize_m: float,
    epsilon: float,
    *,
    phi_hat_n_fine: int = 4096,
    phi_hat_oversample: int | None = None,
) -> WGridderPlan:
    """Build the wgridder plan for the given (uvw, freq, image, epsilon).

    The returned plan can be passed to :func:`jax_nufft.dirty2vis` and
    :func:`jax_nufft.vis2dirty`. All planning math runs on the host (numpy);
    the resulting numerical arrays live as JAX device arrays so that the
    JIT-compiled operators see them as constants.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0; got {epsilon}")
    n_l, n_m = image_shape
    if n_l <= 0 or n_m <= 0:
        raise ValueError(f"image_shape must be positive; got {image_shape}")
    if pixsize_l <= 0 or pixsize_m <= 0:
        raise ValueError(f"pixsize_l and pixsize_m must be > 0; got ({pixsize_l}, {pixsize_m})")

    uvw_arr, freq_arr, real_dtype = _coerce_uvw_freq_dtype(uvw, freq)
    n_rows = uvw_arr.shape[0]
    n_chan = freq_arr.shape[0]

    # --- kernel parameters ---
    w_kernel_width, beta = kernel_params(epsilon)

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
    max_abs_nm1 = float(np.max(np.abs(n_minus_1_np)))
    if max_abs_nm1 == 0.0:
        # Pathological: a 1x1 image at zenith. Force a tiny but non-zero value
        # so that downstream ratios are well-defined.
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

    # --- number of w-planes ---
    # Sample w with step dw = x0 / max|n-1|, matching ducc's choice for
    # ofactor=2 kernels (see W_OVERSAMPLE_X0). This is independent of W.
    x0 = W_OVERSAMPLE_X0
    n_w_inner = math.ceil(w_extent * max_abs_nm1 / x0)
    n_w_inner = max(n_w_inner, 1)  # always have at least one interior step
    n_w = n_w_inner + w_kernel_width

    # --- w-plane centres (spec sec 4.2 step 4) ---
    if n_w_inner == 0:
        # Degenerate: w-extent tiny relative to kernel; fall back to a single plane.
        dw = 1.0
    else:
        dw = w_extent / n_w_inner
    w_kernel_scale = dw * w_kernel_width / 2.0
    k = np.arange(n_w)
    w_centers_np = w_min_all + (k - w_kernel_width / 2.0) * dw
    w_centers_np = w_centers_np.astype(real_dtype)

    # --- phi_hat_n (precomputed on the n-1 grid) ---
    # Argument to phi_hat is eta = (n - 1) * scale, where scale is the kernel
    # half-width in wavelengths. With the v0.1.1 fixed-x0 sampling the
    # nominal eta_max is x0 * W / 2 = W/8 (W=4 -> 0.5, W=8 -> 1.0,
    # W=10 -> 1.25). We size the phi_hat oversample to keep cubic-Lagrange
    # interpolation accurate on that wider range.
    eta_n = n_minus_1_np * w_kernel_scale
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
    w_centers64 = w_centers_np.astype(np.float64)
    for c in range(n_chan):
        w_lambda_c = uvw_lambda_sorted_np[c, :, 2].astype(np.float64)
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
        max_window_size=int(max_window_size),
        window_padding_overhead=float(window_padding_overhead),
        w_extent=float(w_extent),
        is_constant_w=bool(w_extent == 0.0),
        uvw_lambda=jnp.asarray(uvw_lambda_np),
        w_centers=jnp.asarray(w_centers_np),
        n_minus_1=jnp.asarray(n_minus_1_np),
        phi_hat_n=jnp.asarray(phi_hat_n_np),
        sort_perm=jnp.asarray(sort_perm_np),
        uvw_lambda_sorted=jnp.asarray(uvw_lambda_sorted_np),
        window_start=jnp.asarray(window_start_np),
        window_size=jnp.asarray(window_size_np),
    )


__all__ = ["WGridderPlan", "make_plan"]
