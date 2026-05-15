"""Tiny-problem checks against the explicit DFT (single channel for now).

These tests use a small image and small ``Nrow`` so that we can afford the
full O(Nrow * Nl * Nm) reference DFT. They verify the *math* of the wgridder
end-to-end at the smallest non-trivial scale, independent of ducc.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan
from jax_nufft._utils import SPEED_OF_LIGHT

jax.config.update("jax_enable_x64", True)


def _reference_forward(
    image: np.ndarray, uvw: np.ndarray, freq: np.ndarray, pixsize_l: float, pixsize_m: float
) -> np.ndarray:
    """Direct DFT matching ducc's explicit_degridder sign convention.

    image: (n_chan, n_l, n_m) complex
    uvw:   (n_rows, 3) in metres
    freq:  (n_chan,) in Hz
    Returns: vis (n_rows, n_chan) complex.
    """
    n_chan, n_l, n_m = image.shape
    n_rows = uvw.shape[0]
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    ll = i * pixsize_l
    mm = j * pixsize_m
    LL, MM = np.meshgrid(ll, mm, indexing="ij")
    inside = np.maximum(1.0 - LL**2 - MM**2, 0.0)
    nm1 = np.sqrt(inside) - 1.0
    out = np.zeros((n_rows, n_chan), dtype=np.complex128)
    for c in range(n_chan):
        scale = freq[c] / SPEED_OF_LIGHT
        u = uvw[:, 0] * scale
        v = uvw[:, 1] * scale
        w = uvw[:, 2] * scale
        for r in range(n_rows):
            # Match ducc: phase = -2 pi i (u l + v m - w (n - 1))
            phase = -2j * np.pi * (u[r] * LL + v[r] * MM - w[r] * nm1)
            out[r, c] = np.sum(image[c] * np.exp(phase))
    return out


@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("w_strategy", ["dense_scan", "dense_vmap"])
def test_forward_matches_dft_single_channel_zenith(eps: float, w_strategy: str) -> None:
    """Tiny zenith problem: w in metres deliberately non-zero but small."""
    rng = np.random.default_rng(123)
    n_l = n_m = 16
    n_rows = 24
    pixsize = 0.005  # ~17 arcmin per pixel: small FoV, very mild w-effect

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-2.0, 2.0, size=n_rows)
    freq = np.array([1.4e9])

    image = rng.standard_normal((1, n_l, n_m)) + 1j * rng.standard_normal((1, n_l, n_m))

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy=w_strategy))
    vis_ref = _reference_forward(image, uvw, freq, pixsize, pixsize)

    err = np.linalg.norm(vis_jax - vis_ref) / np.linalg.norm(vis_ref)
    assert err < 10 * eps, f"relative error {err:.3e} exceeds 10*eps={10 * eps:.3e}"


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_forward_matches_dft_off_zenith(eps: float) -> None:
    """Tilted array so ``w`` and the n-1 phase actually do work."""
    rng = np.random.default_rng(7)
    n_l = n_m = 32
    n_rows = 48
    pixsize = 0.01  # ~34 arcmin/pixel

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-30.0, 30.0, size=n_rows)
    freq = np.array([1.0e9])

    image = rng.standard_normal((1, n_l, n_m)) + 1j * rng.standard_normal((1, n_l, n_m))

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image)))
    vis_ref = _reference_forward(image, uvw, freq, pixsize, pixsize)

    err = np.linalg.norm(vis_jax - vis_ref) / np.linalg.norm(vis_ref)
    assert err < 10 * eps, f"relative error {err:.3e} exceeds 10*eps={10 * eps:.3e}"


def test_forward_real_image_promotes_to_complex() -> None:
    """Real input should be auto-promoted to complex."""
    rng = np.random.default_rng(0)
    pixsize = 0.005
    n_l = n_m = 16
    uvw = rng.uniform(-50, 50, size=(20, 3))
    freq = np.array([1e9])
    image = rng.standard_normal((1, n_l, n_m))  # real
    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6)
    vis = dirty2vis(plan, jnp.asarray(image))
    assert jnp.iscomplexobj(vis)
    assert vis.shape == (20, 1)


def test_forward_2d_image_broadcasts_across_channels() -> None:
    """2D image should be broadcast across all channels."""
    rng = np.random.default_rng(1)
    pixsize = 0.005
    n_l = n_m = 8
    uvw = rng.uniform(-30, 30, size=(10, 3))
    freq = np.array([1e9, 1.5e9])
    image_2d = rng.standard_normal((n_l, n_m))
    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6)
    vis_2d = dirty2vis(plan, jnp.asarray(image_2d))
    image_3d = np.broadcast_to(image_2d, (2, n_l, n_m))
    vis_3d = dirty2vis(plan, jnp.asarray(image_3d))
    np.testing.assert_allclose(np.asarray(vis_2d), np.asarray(vis_3d), rtol=1e-12)
