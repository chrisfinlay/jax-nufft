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
from typing import Literal

import jax
import jax.numpy as jnp
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
    "dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap", "scan", "vmap"
]
ChannelStrategy = Literal["scan", "vmap"]

_CANONICAL_W_STRATEGIES = ("dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap")
_W_STRATEGY_ALIASES = {"scan": "dense_scan", "vmap": "dense_vmap"}


def _canonicalise_w_strategy(name: str) -> str:
    """Resolve user-facing ``w_strategy`` to a canonical name.

    Emits :class:`DeprecationWarning` for the v0.1 names.
    """
    if name in _CANONICAL_W_STRATEGIES:
        return name
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
        f"{_CANONICAL_W_STRATEGIES + tuple(_W_STRATEGY_ALIASES)}"
    )


def _real_to_complex_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """Promote a real dtype to its matching complex dtype."""
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return dtype
    if dtype == jnp.float32:
        return jnp.complex64
    if dtype == jnp.float64:
        return jnp.complex128
    raise TypeError(f"unsupported dtype for wgridder: {dtype!r}")


def _prepare_image(image: Array, plan: WGridderPlan) -> Array:
    """Broadcast / cast the input image to ``(n_chan, n_l, n_m)`` complex."""
    if image.ndim == 2:
        if image.shape != (plan.n_l, plan.n_m):
            raise ValueError(
                f"image shape {image.shape} does not match plan ({plan.n_l}, {plan.n_m})"
            )
        image = jnp.broadcast_to(image, (plan.n_chan, plan.n_l, plan.n_m))
    elif image.ndim == 3:
        if image.shape != (plan.n_chan, plan.n_l, plan.n_m):
            raise ValueError(
                f"image shape {image.shape} does not match plan "
                f"({plan.n_chan}, {plan.n_l}, {plan.n_m})"
            )
    else:
        raise ValueError(f"image must be 2- or 3-dimensional; got ndim={image.ndim}")
    cdtype = _real_to_complex_dtype(image.dtype)
    return image.astype(cdtype)


def _validate_vis(vis: Array, plan: WGridderPlan) -> Array:
    if vis.ndim != 2 or vis.shape != (plan.n_rows, plan.n_chan):
        raise ValueError(
            f"vis shape {vis.shape} does not match plan ({plan.n_rows}, {plan.n_chan})"
        )
    if not jnp.issubdtype(vis.dtype, jnp.complexfloating):
        # Auto-promote real visibilities to complex (matches user's typical
        # workflow where they constructed vis with zero imaginary part).
        cdtype = _real_to_complex_dtype(vis.dtype)
        vis = vis.astype(cdtype)
    return vis


def _validate_weights(weights: Array | None, plan: WGridderPlan) -> Array | None:
    if weights is None:
        return None
    if weights.shape != (plan.n_rows, plan.n_chan):
        raise ValueError(
            f"weights shape {weights.shape} does not match plan ({plan.n_rows}, {plan.n_chan})"
        )
    return weights


def _channel_forward(
    image_c: Array,
    uvw_c: Array,
    plan: WGridderPlan,
    opts: Opts,
    w_strategy: WStrategy,
) -> Array:
    """Forward operator for a single channel: image (n_l, n_m) -> vis (n_rows,)."""
    two_pi = 2.0 * jnp.pi
    u_ft = (two_pi * plan.pixsize_l) * uvw_c[:, 0]
    v_ft = (two_pi * plan.pixsize_m) * uvw_c[:, 1]
    w_lambda = uvw_c[:, 2]

    cdtype = image_c.dtype
    n_rows = uvw_c.shape[0]

    def w_plane_contribution(w_k: Array) -> Array:
        phase = (two_pi * w_k) * plan.n_minus_1  # (n_l, n_m), real
        shift = jnp.exp((1j * phase).astype(cdtype))
        image_k = image_c * shift / plan.phi_hat_n.astype(cdtype)
        vis_k = nufft2(image_k, u_ft, v_ft, iflag=-1, eps=plan.epsilon, opts=opts)
        # w-direction kernel applied at the visibility output
        z = (w_lambda - w_k) / plan.w_kernel_scale
        kernel_w = phi(z, plan.beta).astype(cdtype)
        return vis_k * kernel_w

    if w_strategy == "dense_vmap":
        contributions = jax.vmap(w_plane_contribution)(plan.w_centers)
        return jnp.sum(contributions, axis=0)

    if w_strategy == "dense_scan":

        def step(acc: Array, w_k: Array) -> tuple[Array, None]:
            return acc + w_plane_contribution(w_k), None

        init = jnp.zeros((n_rows,), dtype=cdtype)
        result, _ = jax.lax.scan(step, init, plan.w_centers)
        return result

    raise ValueError(f"unknown w_strategy: {w_strategy!r}")


def _channel_forward_windowed(
    image_c: Array,
    uvw_lambda_sorted_c: Array,
    window_start_c: Array,
    plan: WGridderPlan,
    opts: Opts,
) -> Array:
    """Windowed forward operator for a single channel.

    Each w-plane processes a contiguous slice (size ``max_window_size``) of
    the w-sorted visibilities and scatters the per-row contributions back
    into the original visibility order via ``plan.sort_perm``. Visibilities
    inside the slice but outside the kernel's natural support pick up
    ``phi(z) = 0`` automatically, so no explicit mask is needed.
    """
    two_pi = 2.0 * jnp.pi
    u_sorted = (two_pi * plan.pixsize_l) * uvw_lambda_sorted_c[:, 0]
    v_sorted = (two_pi * plan.pixsize_m) * uvw_lambda_sorted_c[:, 1]
    w_lambda_sorted = uvw_lambda_sorted_c[:, 2]

    cdtype = image_c.dtype
    n_rows = plan.n_rows
    max_window_size = plan.max_window_size
    # ``dynamic_slice`` clamps out-of-bounds starts, but doing so silently
    # would change which rows the kernel sees on the right edge. Clamp
    # explicitly so the slice is always in-bounds.
    lo_max = max(n_rows - max_window_size, 0)

    def per_plane(vis_acc: Array, args: tuple[Array, Array]) -> tuple[Array, None]:
        lo_raw, w_k = args
        lo = jnp.clip(lo_raw, 0, lo_max)

        u_k = jax.lax.dynamic_slice(u_sorted, (lo,), (max_window_size,))
        v_k = jax.lax.dynamic_slice(v_sorted, (lo,), (max_window_size,))
        w_k_lambda = jax.lax.dynamic_slice(w_lambda_sorted, (lo,), (max_window_size,))

        phase = (two_pi * w_k) * plan.n_minus_1
        shift = jnp.exp((1j * phase).astype(cdtype))
        image_k = image_c * shift / plan.phi_hat_n.astype(cdtype)

        contrib = nufft2(image_k, u_k, v_k, iflag=-1, eps=plan.epsilon, opts=opts)

        z = (w_k_lambda - w_k) / plan.w_kernel_scale
        kernel_w = phi(z, plan.beta).astype(cdtype)
        contrib = contrib * kernel_w

        rows_k = jax.lax.dynamic_slice(plan.sort_perm, (lo,), (max_window_size,))
        return vis_acc.at[rows_k].add(contrib), None

    vis_init = jnp.zeros((n_rows,), dtype=cdtype)
    vis_c, _ = jax.lax.scan(per_plane, vis_init, (window_start_c, plan.w_centers))
    return vis_c


def _channel_adjoint(
    vis_c: Array,
    uvw_c: Array,
    plan: WGridderPlan,
    opts: Opts,
    w_strategy: WStrategy,
) -> Array:
    """Adjoint operator for a single channel: vis (n_rows,) -> dirty (n_l, n_m)."""
    two_pi = 2.0 * jnp.pi
    u_ft = (two_pi * plan.pixsize_l) * uvw_c[:, 0]
    v_ft = (two_pi * plan.pixsize_m) * uvw_c[:, 1]
    w_lambda = uvw_c[:, 2]

    cdtype = vis_c.dtype

    def w_plane_contribution(w_k: Array) -> Array:
        z = (w_lambda - w_k) / plan.w_kernel_scale
        kernel_w = phi(z, plan.beta).astype(cdtype)
        vis_k = vis_c * kernel_w
        # Adjoint of the type-2 NUFFT is type 1 with iflag = +1 (the conjugate
        # of iflag=-1 used in the forward).
        h_k = nufft1((plan.n_l, plan.n_m), vis_k, u_ft, v_ft, iflag=+1, eps=plan.epsilon, opts=opts)
        # Adjoint of the image-domain shift exp(+2 pi i w_k (n-1)) is its conjugate.
        phase = (two_pi * w_k) * plan.n_minus_1
        shift = jnp.exp((-1j * phase).astype(cdtype))
        return h_k * shift / plan.phi_hat_n.astype(cdtype)

    if w_strategy == "dense_vmap":
        contributions = jax.vmap(w_plane_contribution)(plan.w_centers)
        return jnp.sum(contributions, axis=0)

    if w_strategy == "dense_scan":

        def step(acc: Array, w_k: Array) -> tuple[Array, None]:
            return acc + w_plane_contribution(w_k), None

        init = jnp.zeros((plan.n_l, plan.n_m), dtype=cdtype)
        result, _ = jax.lax.scan(step, init, plan.w_centers)
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

    if w_strategy == "windowed_scan":
        # Windowed path: per-channel function takes pre-permuted coords and
        # the per-channel window-start table from the plan.
        if channel_strategy == "vmap":
            vis_per_chan = jax.vmap(
                lambda im_c, uvw_s_c, ws_c: _channel_forward_windowed(
                    im_c, uvw_s_c, ws_c, plan, opts
                )
            )(image, plan.uvw_lambda_sorted, plan.window_start)
        elif channel_strategy == "scan":

            def step_w(_: None, args: tuple[Array, Array, Array]) -> tuple[None, Array]:
                im_c, uvw_s_c, ws_c = args
                return None, _channel_forward_windowed(im_c, uvw_s_c, ws_c, plan, opts)

            _, vis_per_chan = jax.lax.scan(
                step_w, None, (image, plan.uvw_lambda_sorted, plan.window_start)
            )
        else:
            raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")
        return vis_per_chan.T  # (n_rows, n_chan)

    if channel_strategy == "vmap":
        vis_per_chan = jax.vmap(
            lambda im_c, uvw_c: _channel_forward(im_c, uvw_c, plan, opts, w_strategy)
        )(image, plan.uvw_lambda)
    elif channel_strategy == "scan":

        def step(_: None, args: tuple[Array, Array]) -> tuple[None, Array]:
            im_c, uvw_c = args
            return None, _channel_forward(im_c, uvw_c, plan, opts, w_strategy)

        _, vis_per_chan = jax.lax.scan(step, None, (image, plan.uvw_lambda))
    else:
        raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")

    return vis_per_chan.T  # (n_rows, n_chan)


def dirty2vis(
    plan: WGridderPlan,
    image: Array,
    *,
    w_strategy: WStrategy = "dense_scan",
    channel_strategy: ChannelStrategy = "scan",
    nthreads: int = 0,
) -> Array:
    """Forward wgridder: image cube -> visibilities.

    Parameters
    ----------
    plan:
        Pre-built plan from :func:`~jax_nufft.planning.make_plan`.
    image:
        Either ``(n_chan, n_l, n_m)`` or ``(n_l, n_m)`` (broadcast across
        channels). Real or complex; real input is promoted to complex.
    w_strategy:
        ``"dense_scan"`` (default, low memory) or ``"dense_vmap"`` (potentially
        faster on GPU but allocates ``n_w * image_size`` peak memory). The bare
        names ``"scan"`` / ``"vmap"`` are accepted as deprecated aliases.
    channel_strategy:
        ``"scan"`` (default) or ``"vmap"`` for the channel loop.
    nthreads:
        Threads to pass to jax-finufft (0 = let FINUFFT decide).

    Returns
    -------
    vis:
        Complex array of shape ``(n_rows, n_chan)``.
    """
    w_strategy = _canonicalise_w_strategy(w_strategy)
    image = _prepare_image(image, plan)
    return _dirty2vis_jit(
        plan,
        image,
        w_strategy=w_strategy,
        channel_strategy=channel_strategy,
        nthreads=nthreads,
    )


def _channel_adjoint_windowed(
    vis_sorted_c: Array,
    uvw_lambda_sorted_c: Array,
    window_start_c: Array,
    plan: WGridderPlan,
    opts: Opts,
) -> Array:
    """Windowed adjoint operator for a single channel.

    Mirrors :func:`_channel_forward_windowed`: per plane we take a
    contiguous slice of the w-sorted visibilities, apply the w-kernel
    weight (which zeros out padded entries automatically), run a 2D
    NUFFT type 1 to land an image, and accumulate.
    """
    two_pi = 2.0 * jnp.pi
    u_sorted = (two_pi * plan.pixsize_l) * uvw_lambda_sorted_c[:, 0]
    v_sorted = (two_pi * plan.pixsize_m) * uvw_lambda_sorted_c[:, 1]
    w_lambda_sorted = uvw_lambda_sorted_c[:, 2]

    cdtype = vis_sorted_c.dtype
    max_window_size = plan.max_window_size
    lo_max = max(plan.n_rows - max_window_size, 0)

    def per_plane(dirty_acc: Array, args: tuple[Array, Array]) -> tuple[Array, None]:
        lo_raw, w_k = args
        lo = jnp.clip(lo_raw, 0, lo_max)

        u_k = jax.lax.dynamic_slice(u_sorted, (lo,), (max_window_size,))
        v_k = jax.lax.dynamic_slice(v_sorted, (lo,), (max_window_size,))
        w_k_lambda = jax.lax.dynamic_slice(w_lambda_sorted, (lo,), (max_window_size,))
        vis_k = jax.lax.dynamic_slice(vis_sorted_c, (lo,), (max_window_size,))

        z = (w_k_lambda - w_k) / plan.w_kernel_scale
        kernel_w = phi(z, plan.beta).astype(cdtype)
        vis_k = vis_k * kernel_w

        h_k = nufft1(
            (plan.n_l, plan.n_m), vis_k, u_k, v_k, iflag=+1, eps=plan.epsilon, opts=opts
        )
        phase = (two_pi * w_k) * plan.n_minus_1
        shift = jnp.exp((-1j * phase).astype(cdtype))
        contrib = h_k * shift / plan.phi_hat_n.astype(cdtype)
        return dirty_acc + contrib, None

    dirty_init = jnp.zeros((plan.n_l, plan.n_m), dtype=cdtype)
    dirty_c, _ = jax.lax.scan(per_plane, dirty_init, (window_start_c, plan.w_centers))
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

    if w_strategy == "windowed_scan":
        # Apply sort_perm once per channel so windowed slices line up with
        # plan.uvw_lambda_sorted / plan.window_start.
        vis_sorted_per_chan = vis_per_chan[:, plan.sort_perm]
        if channel_strategy == "vmap":
            dirty_per_chan = jax.vmap(
                lambda v_s_c, uvw_s_c, ws_c: _channel_adjoint_windowed(
                    v_s_c, uvw_s_c, ws_c, plan, opts
                )
            )(vis_sorted_per_chan, plan.uvw_lambda_sorted, plan.window_start)
        elif channel_strategy == "scan":

            def step_w(_: None, args: tuple[Array, Array, Array]) -> tuple[None, Array]:
                v_s_c, uvw_s_c, ws_c = args
                return None, _channel_adjoint_windowed(v_s_c, uvw_s_c, ws_c, plan, opts)

            _, dirty_per_chan = jax.lax.scan(
                step_w,
                None,
                (vis_sorted_per_chan, plan.uvw_lambda_sorted, plan.window_start),
            )
        else:
            raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")
    elif channel_strategy == "vmap":
        dirty_per_chan = jax.vmap(
            lambda v_c, uvw_c: _channel_adjoint(v_c, uvw_c, plan, opts, w_strategy)
        )(vis_per_chan, plan.uvw_lambda)
    elif channel_strategy == "scan":

        def step(_: None, args: tuple[Array, Array]) -> tuple[None, Array]:
            v_c, uvw_c = args
            return None, _channel_adjoint(v_c, uvw_c, plan, opts, w_strategy)

        _, dirty_per_chan = jax.lax.scan(step, None, (vis_per_chan, plan.uvw_lambda))
    else:
        raise ValueError(f"unknown channel_strategy: {channel_strategy!r}")

    # Apply 1/n on the output (matching ducc's divide_by_n=True), and take
    # the real part to land in real space.
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
    nthreads: int = 0,
) -> Array:
    """Adjoint wgridder: visibilities -> image cube (with 1/n factor).

    Parameters
    ----------
    plan:
        Pre-built plan from :func:`~jax_nufft.planning.make_plan`.
    vis:
        Complex array of shape ``(n_rows, n_chan)``.
    weights:
        Optional real array of shape ``(n_rows, n_chan)``, multiplied into the
        visibilities before gridding (matches ducc's ``wgt`` argument).
    w_strategy:
        ``"dense_scan"`` (default) or ``"dense_vmap"``; same semantics as in
        :func:`dirty2vis`. The bare names ``"scan"`` / ``"vmap"`` are accepted
        as deprecated aliases.
    channel_strategy:
        ``"scan"`` (default) or ``"vmap"``.
    nthreads:
        Threads to pass to jax-finufft (0 = let FINUFFT decide).

    Returns
    -------
    dirty:
        Real array of shape ``(n_chan, n_l, n_m)``.
    """
    w_strategy = _canonicalise_w_strategy(w_strategy)
    vis = _validate_vis(vis, plan)
    weights = _validate_weights(weights, plan)
    apply_w = weights is not None
    return _vis2dirty_jit(
        plan,
        vis,
        weights if apply_w else jnp.zeros((), dtype=vis.real.dtype),
        w_strategy=w_strategy,
        channel_strategy=channel_strategy,
        nthreads=nthreads,
        apply_w_weights=apply_w,
    )


__all__ = ["dirty2vis", "vis2dirty"]
