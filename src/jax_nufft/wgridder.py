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

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax_finufft import nufft1, nufft2
from jax_finufft.options import Opts

from jax_nufft.kernel import phi
from jax_nufft.planning import WGridderPlan

WStrategy = Literal["scan", "vmap"]
ChannelStrategy = Literal["scan", "vmap"]


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
            f"vis shape {vis.shape} does not match plan "
            f"({plan.n_rows}, {plan.n_chan})"
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
            f"weights shape {weights.shape} does not match plan "
            f"({plan.n_rows}, {plan.n_chan})"
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

    if w_strategy == "vmap":
        contributions = jax.vmap(w_plane_contribution)(plan.w_centers)
        return jnp.sum(contributions, axis=0)

    if w_strategy == "scan":
        def step(acc: Array, w_k: Array) -> tuple[Array, None]:
            return acc + w_plane_contribution(w_k), None

        init = jnp.zeros((n_rows,), dtype=cdtype)
        result, _ = jax.lax.scan(step, init, plan.w_centers)
        return result

    raise ValueError(f"unknown w_strategy: {w_strategy!r}")


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
        h_k = nufft1(
            (plan.n_l, plan.n_m), vis_k, u_ft, v_ft, iflag=+1, eps=plan.epsilon, opts=opts
        )
        # Adjoint of the image-domain shift exp(+2 pi i w_k (n-1)) is its conjugate.
        phase = (two_pi * w_k) * plan.n_minus_1
        shift = jnp.exp((-1j * phase).astype(cdtype))
        return h_k * shift / plan.phi_hat_n.astype(cdtype)

    if w_strategy == "vmap":
        contributions = jax.vmap(w_plane_contribution)(plan.w_centers)
        return jnp.sum(contributions, axis=0)

    if w_strategy == "scan":
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
    w_strategy: WStrategy = "scan",
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
    w_strategy, channel_strategy:
        ``"scan"`` (default, low memory) or ``"vmap"`` (potentially faster on GPU
        but allocates ``n_w * image_size`` peak memory).
    nthreads:
        Threads to pass to jax-finufft (0 = let FINUFFT decide).

    Returns
    -------
    vis:
        Complex array of shape ``(n_rows, n_chan)``.
    """
    image = _prepare_image(image, plan)
    return _dirty2vis_jit(
        plan,
        image,
        w_strategy=w_strategy,
        channel_strategy=channel_strategy,
        nthreads=nthreads,
    )


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

    if channel_strategy == "vmap":
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
    w_strategy: WStrategy = "scan",
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
    w_strategy, channel_strategy:
        ``"scan"`` (default) or ``"vmap"``; same semantics as in
        :func:`dirty2vis`.
    nthreads:
        Threads to pass to jax-finufft (0 = let FINUFFT decide).

    Returns
    -------
    dirty:
        Real array of shape ``(n_chan, n_l, n_m)``.
    """
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
