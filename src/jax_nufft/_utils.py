"""Internal helpers shared by the wgridder forward and adjoint operators."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

SPEED_OF_LIGHT = 299_792_458.0  # m/s, exact (CODATA / SI definition)


def n_minus_1_grid(n_l: int, n_m: int, pixsize_l: float, pixsize_m: float) -> Array:
    """Return ``sqrt(1 - l^2 - m^2) - 1`` evaluated on the (Nl, Nm) image grid.

    The grid follows ``l_i = (i - Nl/2) * pixsize_l`` for ``i = 0, ..., Nl-1``,
    matching the natural / image-centric ordering used by `jax-finufft`
    (modeord=0). Values that would fall outside the unit disc are clamped to 0
    inside the square root before subtracting 1, so the returned array equals
    ``-1`` on those pixels.
    """
    i = jnp.arange(n_l) - n_l // 2
    j = jnp.arange(n_m) - n_m // 2
    ll = i * pixsize_l
    mm = j * pixsize_m
    l2 = ll[:, None] ** 2
    m2 = mm[None, :] ** 2
    inside = jnp.maximum(1.0 - l2 - m2, 0.0)
    return jnp.sqrt(inside) - 1.0


def uvw_to_finufft_coords(
    uvw_lambda: Array, pixsize_l: float, pixsize_m: float
) -> tuple[Array, Array]:
    """Scale baseline coordinates (in wavelengths) to the FINUFFT [-pi, pi] range.

    FINUFFT type 1/2 expects non-uniform points in ``[-pi, pi)``. The natural
    image-domain wavenumber for a pixel at offset ``i' = i - Nl/2`` is
    ``k = i'``, so the discretised exponent ``-2 pi i u l = -2 pi i u (i' dl)``
    maps to a FINUFFT point coordinate ``x = 2 pi u dl``.
    """
    two_pi = 2.0 * jnp.pi
    u_finufft = two_pi * uvw_lambda[..., 0] * pixsize_l
    v_finufft = two_pi * uvw_lambda[..., 1] * pixsize_m
    return u_finufft, v_finufft


def uvw_meters_to_lambda(uvw_meters: Array, freq_hz: Array) -> Array:
    """Convert (Nrow, 3) uvw in metres + (Nchan,) frequencies to (Nchan, Nrow, 3) wavelengths."""
    factor = freq_hz / SPEED_OF_LIGHT
    return uvw_meters[None, :, :] * factor[:, None, None]
