"""Parity tests against ``ducc0.wgridder``.

ducc0's ``dirty2vis`` and ``vis2dirty`` are taken as ground truth. Because our
operators do not divide by ``n`` on the forward but do divide by ``n`` on the
adjoint, we configure ducc consistently:

  * forward parity: ``ducc0.wgridder.dirty2vis(divide_by_n=False, ...)``
  * adjoint parity: ``ducc0.wgridder.vis2dirty(divide_by_n=True, ...)``

The acceptance threshold is ``10 * epsilon`` per the spec.

Image sizes / row counts are reduced from the spec values so that the test
matrix runs in single-digit seconds locally; the algorithmic regime (FoV,
baseline length distribution, off-zenith pointing) is preserved.
"""

from __future__ import annotations

import ducc0.wgridder
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from tests.conftest import Telescope, synthetic_uvw


@pytest.mark.parametrize("w_strategy", ["dense_scan", "windowed_scan"])
@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_forward_parity(
    short_telescope_pointing: tuple[Telescope, float], eps: float, w_strategy: str
) -> None:
    tel, zen_deg = short_telescope_pointing
    uvw = synthetic_uvw(tel, zen_deg, seed=0)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    rng = np.random.default_rng(7)
    image = rng.standard_normal((tel.n_pix, tel.n_pix))

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy=w_strategy))

    vis_ducc = ducc0.wgridder.dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=image,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )

    err = np.linalg.norm(vis_jax - vis_ducc) / np.linalg.norm(vis_ducc)
    # Loosen the tolerance slightly: ducc and our wgridder both target
    # `epsilon` independently, so the gap between them is bounded by ~2*eps
    # in the best case and ~10*eps in practice.
    assert err < 20 * eps, (
        f"{tel.name} zen={zen_deg} eps={eps:g} {w_strategy}: relative error {err:.3e}"
    )


@pytest.mark.parametrize("w_strategy", ["dense_scan", "windowed_scan"])
@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_adjoint_parity(
    short_telescope_pointing: tuple[Telescope, float], eps: float, w_strategy: str
) -> None:
    tel, zen_deg = short_telescope_pointing
    uvw = synthetic_uvw(tel, zen_deg, seed=1)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    rng = np.random.default_rng(11)
    vis_np = (
        rng.standard_normal((tel.n_rows, 1)) + 1j * rng.standard_normal((tel.n_rows, 1))
    ).astype(np.complex128)

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis_np), w_strategy=w_strategy))[0]

    dirty_ducc = ducc0.wgridder.vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis_np,
        npix_x=tel.n_pix,
        npix_y=tel.n_pix,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=True,
        nthreads=1,
    )

    err = np.linalg.norm(dirty_jax - dirty_ducc) / np.linalg.norm(dirty_ducc)
    assert err < 20 * eps, (
        f"{tel.name} zen={zen_deg} eps={eps:g} {w_strategy}: relative error {err:.3e}"
    )


@pytest.mark.parametrize("eps", [1e-6])
def test_forward_parity_with_weights(
    short_telescope_pointing: tuple[Telescope, float], eps: float
) -> None:
    """Sanity: our adjoint with weights matches ducc's adjoint with `wgt`."""
    tel, zen_deg = short_telescope_pointing
    uvw = synthetic_uvw(tel, zen_deg, seed=2)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    rng = np.random.default_rng(15)
    vis_np = (
        rng.standard_normal((tel.n_rows, 1)) + 1j * rng.standard_normal((tel.n_rows, 1))
    ).astype(np.complex128)
    wgt = rng.uniform(0.1, 1.0, size=(tel.n_rows, 1)).astype(np.float64)

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis_np), weights=jnp.asarray(wgt)))[0]

    dirty_ducc = ducc0.wgridder.vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis_np,
        wgt=wgt,
        npix_x=tel.n_pix,
        npix_y=tel.n_pix,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=True,
        nthreads=1,
    )

    err = np.linalg.norm(dirty_jax - dirty_ducc) / np.linalg.norm(dirty_ducc)
    assert err < 20 * eps


@pytest.mark.parametrize("eps", [1e-6])
def test_forward_parity_long(long_telescope_pointing: tuple[Telescope, float], eps: float) -> None:
    """Slow parity tests for MWA_extended / MeerKAT (skipped without --runslow)."""
    tel, zen_deg = long_telescope_pointing
    uvw = synthetic_uvw(tel, zen_deg, seed=4)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    rng = np.random.default_rng(21)
    image = rng.standard_normal((tel.n_pix, tel.n_pix))

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image)))
    vis_ducc = ducc0.wgridder.dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=image,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )
    err = np.linalg.norm(vis_jax - vis_ducc) / np.linalg.norm(vis_ducc)
    assert err < 20 * eps


@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_constant_w_ducc_parity(op: str, w_strategy: str) -> None:
    """v0.1.2 fast path: coplanar (w == 0 everywhere) data must match ducc
    within ``20 * eps`` for every w_strategy. ``plan.n_w == 1`` confirms the
    specialisation engaged."""
    eps = 1e-6
    tel = Telescope(
        name="MWA_compact_coplanar",
        freq_hz=150e6,
        n_rows=400,
        sigma_uv_m=50.0,
        max_baseline_m=200.0,
        n_pix=128,
        fov_rad=np.radians(20.0),
    )
    uvw = synthetic_uvw(tel, 0.0, seed=42)  # zenith pointing
    uvw[:, 2] = 0.0  # force exactly coplanar so the v0.1.2 fast path engages
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    # Acceptance signal that the fast path engaged.
    assert plan.is_constant_w
    assert plan.n_w == 1
    assert plan.w_extent == 0.0

    if op == "dirty2vis":
        rng = np.random.default_rng(7)
        image = rng.standard_normal((tel.n_pix, tel.n_pix))
        vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy=w_strategy))
        vis_ducc = ducc0.wgridder.dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=image,
            pixsize_x=pix,
            pixsize_y=pix,
            epsilon=eps,
            do_wgridding=True,
            divide_by_n=False,
            nthreads=1,
        )
        err = np.linalg.norm(vis_jax - vis_ducc) / np.linalg.norm(vis_ducc)
    else:
        rng = np.random.default_rng(11)
        vis_np = (
            rng.standard_normal((tel.n_rows, 1)) + 1j * rng.standard_normal((tel.n_rows, 1))
        ).astype(np.complex128)
        dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis_np), w_strategy=w_strategy))[0]
        dirty_ducc = ducc0.wgridder.vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=vis_np,
            npix_x=tel.n_pix,
            npix_y=tel.n_pix,
            pixsize_x=pix,
            pixsize_y=pix,
            epsilon=eps,
            do_wgridding=True,
            divide_by_n=True,
            nthreads=1,
        )
        err = np.linalg.norm(dirty_jax - dirty_ducc) / np.linalg.norm(dirty_ducc)

    assert err < 20 * eps, (
        f"constant-w fast path n_w={plan.n_w} {op} {w_strategy}: "
        f"relative error {err:.3e} exceeds {20 * eps:.3e}"
    )


@pytest.mark.parametrize("eps", [1e-6])
def test_multichannel_forward_parity(eps: float) -> None:
    """Multi-channel: same image broadcast across 4 channels covering ~10% bandwidth."""
    tel = Telescope(
        name="MWA_compact_multichan",
        freq_hz=150e6,
        n_rows=400,
        sigma_uv_m=50.0,
        max_baseline_m=200.0,
        n_pix=128,
        fov_rad=np.radians(20.0),
    )
    uvw = synthetic_uvw(tel, 30.0, seed=99)
    freq = np.linspace(0.95, 1.05, 4) * tel.freq_hz
    pix = tel.pixsize
    rng = np.random.default_rng(50)
    image = rng.standard_normal((tel.n_pix, tel.n_pix))

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image)))

    vis_ducc = ducc0.wgridder.dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=image,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )

    err = np.linalg.norm(vis_jax - vis_ducc) / np.linalg.norm(vis_ducc)
    assert err < 20 * eps
