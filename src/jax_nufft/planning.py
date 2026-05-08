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
from jax_nufft.kernel import compute_phi_hat_table, kernel_params

# Eta_max evaluated for phi_hat is ``x0 * W / 2`` (see the n_w_inner
# derivation in :func:`make_plan`). With ``x0 = 1 / W`` we get
# ``eta_max = 0.5``, which keeps the image-domain correction in the
# well-conditioned region of the exp-of-semicircle phi_hat. The spec
# nominally suggests ``x0 = 0.5`` (independent of W), but that pushes
# eta_max well outside the kernel's natural support and produces large
# aliasing errors for moderate-FoV / large-w-extent cases.
W_OVERSAMPLE_ETA_MAX = 0.5


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

    # ---- traced arrays (pytree leaves) ----
    uvw_lambda: Array = field()  # (n_chan, n_rows, 3)
    w_centers: Array = field()  # (n_w,)
    n_minus_1: Array = field()  # (n_l, n_m)
    phi_hat_n: Array = field()  # (n_l, n_m)

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
    ) = aux
    uvw_lambda, w_centers, n_minus_1, phi_hat_n = children
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
        uvw_lambda=uvw_lambda,
        w_centers=w_centers,
        n_minus_1=n_minus_1,
        phi_hat_n=phi_hat_n,
    )


jax.tree_util.register_pytree_node(
    WGridderPlan,
    flatten_func=lambda p: (
        (p.uvw_lambda, p.w_centers, p.n_minus_1, p.phi_hat_n),
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
    phi_hat_oversample: int = 32,
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
    # Pick x0 so that the image-domain argument eta = (n-1) * (dw*W/2) stays
    # within W_OVERSAMPLE_ETA_MAX. eta_max(x0) = x0 * W / 2, so set
    # x0 = 2 * eta_max / W, which gives the formula below.
    x0 = (2.0 * W_OVERSAMPLE_ETA_MAX) / w_kernel_width
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
    # half-width in wavelengths.
    eta_n = n_minus_1_np * w_kernel_scale
    eta_max_request = max(float(np.max(np.abs(eta_n))), 1e-9)
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
        uvw_lambda=jnp.asarray(uvw_lambda_np),
        w_centers=jnp.asarray(w_centers_np),
        n_minus_1=jnp.asarray(n_minus_1_np),
        phi_hat_n=jnp.asarray(phi_hat_n_np),
    )


__all__ = ["WGridderPlan", "make_plan"]
