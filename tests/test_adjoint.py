"""Adjoint operator: DFT comparison, dot-product test, weights pass-through."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT

jax.config.update("jax_enable_x64", True)


def _reference_adjoint(
    vis: np.ndarray,
    uvw: np.ndarray,
    freq: np.ndarray,
    image_shape: tuple[int, int],
    pixsize_l: float,
    pixsize_m: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Direct DFT adjoint matching ducc's explicit_gridder (with divide_by_n=True).

    vis:    (n_rows, n_chan) complex
    Returns: dirty (n_chan, n_l, n_m) real.
    """
    n_l, n_m = image_shape
    n_rows, n_chan = vis.shape
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    ll = i * pixsize_l
    mm = j * pixsize_m
    LL, MM = np.meshgrid(ll, mm, indexing="ij")
    inside = np.maximum(1.0 - LL**2 - MM**2, 0.0)
    nm1 = np.sqrt(inside) - 1.0
    n_grid = nm1 + 1.0
    out = np.zeros((n_chan, n_l, n_m), dtype=np.float64)
    for c in range(n_chan):
        scale = freq[c] / SPEED_OF_LIGHT
        u = uvw[:, 0] * scale
        v = uvw[:, 1] * scale
        w = uvw[:, 2] * scale
        for r in range(n_rows):
            phase = +2j * np.pi * (u[r] * LL + v[r] * MM - w[r] * nm1)
            v_eff = vis[r, c]
            if weights is not None:
                v_eff = v_eff * weights[r, c]
            out[c] += (v_eff * np.exp(phase)).real
    return np.where(n_grid > 0.0, out / np.maximum(n_grid, 1e-30), 0.0)


@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("w_strategy", ["scan", "vmap"])
def test_adjoint_matches_dft_zenith(eps: float, w_strategy: str) -> None:
    rng = np.random.default_rng(11)
    n_l = n_m = 16
    n_rows = 24
    pixsize = 0.005

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-2.0, 2.0, size=n_rows)
    freq = np.array([1.4e9])
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps)
    dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy=w_strategy))
    dirty_ref = _reference_adjoint(vis, uvw, freq, (n_l, n_m), pixsize, pixsize)

    err = np.linalg.norm(dirty_jax - dirty_ref) / np.linalg.norm(dirty_ref)
    assert err < 10 * eps, f"relative error {err:.3e} exceeds 10*eps={10 * eps:.3e}"


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_adjoint_matches_dft_off_zenith(eps: float) -> None:
    rng = np.random.default_rng(13)
    n_l = n_m = 32
    n_rows = 48
    pixsize = 0.01

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-30.0, 30.0, size=n_rows)
    freq = np.array([1.0e9])
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps)
    dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis)))
    dirty_ref = _reference_adjoint(vis, uvw, freq, (n_l, n_m), pixsize, pixsize)

    err = np.linalg.norm(dirty_jax - dirty_ref) / np.linalg.norm(dirty_ref)
    assert err < 10 * eps, f"relative error {err:.3e} exceeds 10*eps={10 * eps:.3e}"


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_dot_product_identity(eps: float) -> None:
    """Adjointness check for the wgridder pair, matching ducc's convention.

    With our chosen convention -- forward has no 1/n factor, adjoint applies
    1/n on the output and takes the real part -- the standard complex adjoint
    identity does not hold. The relation that holds for real x is

        Re(<A x, y>_C) = <n * x, A^* y>_R

    (the n multiplier on the RHS undoes the 1/n that A^* applies on its
    output relative to the literal adjoint of A).
    """
    rng = np.random.default_rng(33)
    n_l = n_m = 32
    n_rows = 48
    pixsize = 0.01

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-30.0, 30.0, size=n_rows)
    freq = np.array([1.4e9])

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps)

    image = rng.standard_normal((1, n_l, n_m))  # real
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )

    n_grid = np.asarray(plan.n_minus_1) + 1.0  # (n_l, n_m), real
    image_n = image * n_grid[None, :, :]

    Ax = np.asarray(dirty2vis(plan, jnp.asarray(image)))  # complex (n_rows, 1)
    Ay = np.asarray(vis2dirty(plan, jnp.asarray(vis)))  # real (1, n_l, n_m)

    lhs = np.vdot(Ax.ravel(), vis.ravel()).real  # Re(<A x, y>_C)
    rhs = float(np.vdot(image_n.ravel(), Ay.ravel()))  # <n * x, A^* y>_R

    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert rel_err < 100 * eps, f"dot-product relative error {rel_err:.3e}; lhs={lhs}, rhs={rhs}"


@pytest.mark.parametrize("eps", [1e-6])
def test_adjoint_weights_match_dft(eps: float) -> None:
    rng = np.random.default_rng(2)
    n_l = n_m = 32
    n_rows = 32
    pixsize = 0.01

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-30.0, 30.0, size=n_rows)
    freq = np.array([1.0e9])
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )
    weights = rng.uniform(0.1, 1.0, size=(n_rows, 1)).astype(np.float64)

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps)
    dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis), weights=jnp.asarray(weights)))
    dirty_ref = _reference_adjoint(vis, uvw, freq, (n_l, n_m), pixsize, pixsize, weights=weights)

    err = np.linalg.norm(dirty_jax - dirty_ref) / np.linalg.norm(dirty_ref)
    assert err < 10 * eps


def test_adjoint_validates_shapes() -> None:
    rng = np.random.default_rng(0)
    pixsize = 0.005
    n_l = n_m = 16
    uvw = rng.uniform(-30, 30, size=(20, 3))
    freq = np.array([1e9])
    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6)

    bad_vis = jnp.zeros((20, 2), dtype=jnp.complex128)  # wrong n_chan
    with pytest.raises(ValueError):
        vis2dirty(plan, bad_vis)

    good_vis = jnp.zeros((20, 1), dtype=jnp.complex128)
    bad_weights = jnp.ones((10, 1))
    with pytest.raises(ValueError):
        vis2dirty(plan, good_vis, weights=bad_weights)
