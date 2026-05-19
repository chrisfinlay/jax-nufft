"""Boundary-plane and pathological-w stress tests for the windowed strategies.

The windowed scheme assumes the contributing rows for any w-plane form a
*contiguous slice* of the w-sorted array. Most of the algorithmic interest
lives at the boundary planes (where the kernel support hangs off either
edge of the data range) and at pathologically clumped w-distributions
(where most planes have empty windows but one or two are densely packed).

These tests construct synthetic w-distributions that stress each regime
and confirm the windowed forward and adjoint agree with the dense
baseline.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty

jax.config.update("jax_enable_x64", True)


def _make_uvw(w_values: np.ndarray, *, seed: int) -> np.ndarray:
    """Build (n_rows, 3) uvw with given w values and random (u, v)."""
    rng = np.random.default_rng(seed)
    n_rows = w_values.shape[0]
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 2] = w_values
    return uvw


def _parity_check(
    uvw: np.ndarray,
    eps: float,
    *,
    n_pix: int = 32,
    pixsize: float = 0.01,
    freq_hz: float = 1.4e9,
) -> None:
    """Forward + adjoint dense-vs-windowed parity at the given epsilon."""
    freq = np.array([freq_hz])
    rng = np.random.default_rng(101)
    image = rng.standard_normal((n_pix, n_pix))
    vis = (
        rng.standard_normal((uvw.shape[0], 1)) + 1j * rng.standard_normal((uvw.shape[0], 1))
    ).astype(np.complex128)

    plan = make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, eps)

    vis_dense = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy="dense_scan"))
    vis_windowed = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy="windowed_scan"))
    fwd_err = np.linalg.norm(vis_windowed - vis_dense) / max(np.linalg.norm(vis_dense), 1e-30)
    # Forward: scatter-add reduces identical contributing rows in identical
    # order, so the windowed path is bit-equal to dense (zero error).
    assert fwd_err < 1e-12, f"forward windowed-vs-dense err {fwd_err:.3e}"

    dirty_dense = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy="dense_scan"))
    dirty_windowed = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy="windowed_scan"))
    adj_err = np.linalg.norm(dirty_windowed - dirty_dense) / max(np.linalg.norm(dirty_dense), 1e-30)
    # Adjoint: the NUFFT type 1 sums depend on the set of rows in the
    # batch; the dense/windowed reductions differ in summation order. We
    # accept up to 100*eps relative difference (well within the per-call
    # epsilon target of each implementation).
    assert adj_err < 100 * eps, f"adjoint windowed-vs-dense err {adj_err:.3e} (eps={eps:g})"


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_boundary_clumped_at_w_min(eps: float) -> None:
    """All rows clumped near w_min: only the lowest few planes have non-empty windows."""
    rng = np.random.default_rng(0)
    n_rows = 200
    # Clump around -25 with a tiny spread; add a small range so n_w > W.
    w_values = rng.normal(loc=-25.0, scale=0.1, size=n_rows)
    # Sprinkle a handful of outliers to extend the w-extent so n_w > W
    w_values[:5] = rng.uniform(-25.0, 25.0, size=5)
    _parity_check(_make_uvw(w_values, seed=1), eps=eps)


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_boundary_clumped_at_w_max(eps: float) -> None:
    """Mirror: rows clumped near w_max."""
    rng = np.random.default_rng(2)
    n_rows = 200
    w_values = rng.normal(loc=+25.0, scale=0.1, size=n_rows)
    w_values[:5] = rng.uniform(-25.0, 25.0, size=5)
    _parity_check(_make_uvw(w_values, seed=3), eps=eps)


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_boundary_symmetric_around_zero(eps: float) -> None:
    """Most contribution concentrated near w=0; edges sparsely populated."""
    rng = np.random.default_rng(4)
    n_rows = 300
    w_values = rng.normal(loc=0.0, scale=5.0, size=n_rows)
    _parity_check(_make_uvw(w_values, seed=5), eps=eps)


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_boundary_bimodal(eps: float) -> None:
    """Bimodal w-distribution: two clumps far apart with empty middle planes."""
    rng = np.random.default_rng(6)
    n_rows = 400
    half = n_rows // 2
    w_values = np.empty(n_rows)
    w_values[:half] = rng.normal(loc=-20.0, scale=0.5, size=half)
    w_values[half:] = rng.normal(loc=+20.0, scale=0.5, size=n_rows - half)
    _parity_check(_make_uvw(w_values, seed=7), eps=eps)


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_boundary_uniform_baseline(eps: float) -> None:
    """Uniform w-distribution: middle planes well-populated, edges thin (baseline)."""
    rng = np.random.default_rng(8)
    n_rows = 300
    w_values = rng.uniform(-30.0, 30.0, size=n_rows)
    _parity_check(_make_uvw(w_values, seed=9), eps=eps)


def test_small_nw_zenith_regression() -> None:
    """Near-zero w-extent (zenith with no z offsets) gives n_w in the W-only regime.

    The plan flags this regime as a risk: when ``n_w`` is close to ``W``,
    every plane is an "edge plane" and the windowed scheme has to still
    produce correct results.
    """
    rng = np.random.default_rng(10)
    n_rows = 64
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-30.0, 30.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-30.0, 30.0, size=n_rows)
    # Tiny w-extent so n_w is dominated by the kernel width.
    uvw[:, 2] = rng.uniform(-0.05, 0.05, size=n_rows)
    eps = 1e-6
    plan = make_plan(uvw, np.array([1.4e9]), (32, 32), 0.01, 0.01, eps)
    # In this regime n_w should be at most a couple of planes plus W.
    assert plan.n_w <= plan.w_kernel_width + 2

    image = rng.standard_normal((32, 32))
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )

    vis_dense = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy="dense_scan"))
    vis_windowed = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy="windowed_scan"))
    assert np.allclose(vis_dense, vis_windowed, atol=1e-12, rtol=1e-12)

    dirty_dense = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy="dense_scan"))
    dirty_windowed = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy="windowed_scan"))
    err = np.linalg.norm(dirty_windowed - dirty_dense) / np.linalg.norm(dirty_dense)
    assert err < 1e-5
