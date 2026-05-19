"""Parity tests for the v0.1.2 constant-w fast path.

The default ``make_plan`` collapses coplanar data (``w_extent == 0``) to a
single-plane plan via the constant-w fast path. The ``_force_generic`` test
kwarg builds the generic-shape plan (``n_w == w_kernel_width + 1``) on the
same data so both can be evaluated and compared.

The generic-path-on-constant-w case is not a supported user code path; it
is built only for these tests. Without the v0.1.2 fast path it would hit
``z = (w_lambda - w_k)/0 = NaN`` at call time. ``make_plan`` defends
against that by picking a non-degenerate ``dw`` fallback when ``w_extent``
is exactly zero, so the resulting generic plan is runnable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty


_EPS = 1e-6
# Relative-norm tolerance: each path has wgridder error ~10*eps relative to
# truth, so the worst-case fast-vs-generic diff is ~20*eps. Keep the
# headroom modest but in line with tests/test_against_dft.py's per-strategy
# 10*eps bound for the dense path itself.
_TOL = 20 * _EPS


def _coplanar_uvw(n_rows: int = 96, seed: int = 1, w_const_m: float = 0.0) -> np.ndarray:
    """Random (u, v) baselines with a single constant w-value in metres."""
    rng = np.random.default_rng(seed)
    uvw = rng.normal(scale=120.0, size=(n_rows, 3))
    uvw[:, 2] = w_const_m
    return uvw


def _build_pair(uvw: np.ndarray):
    """Return ``(plan_fast, plan_generic)`` for the same uvw input."""
    freq = np.array([200e6])
    plan_fast = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(64, 64),
        pixsize_l=1e-3,
        pixsize_m=1e-3,
        epsilon=_EPS,
    )
    plan_generic = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(64, 64),
        pixsize_l=1e-3,
        pixsize_m=1e-3,
        epsilon=_EPS,
        _force_generic=True,
    )
    return plan_fast, plan_generic


@pytest.mark.parametrize("w_const_m", [0.0, 12.5])
def test_force_generic_builds_generic_shape_plan(w_const_m: float) -> None:
    """``_force_generic=True`` must build the generic-shape plan (n_w > 1)
    on constant-w data, with ``is_constant_w=False`` to signal that the
    fast path did not engage."""
    uvw = _coplanar_uvw(w_const_m=w_const_m)
    plan_fast, plan_generic = _build_pair(uvw)

    assert plan_fast.is_constant_w
    assert plan_fast.n_w == 1
    assert plan_fast.w_extent == 0.0

    assert not plan_generic.is_constant_w
    assert plan_generic.n_w == 1 + plan_generic.w_kernel_width
    assert plan_generic.w_extent == 0.0
    # Despite the constant-w data, w_kernel_scale must be strictly positive
    # so the operator's z = (w_lambda - w_k)/scale doesn't blow up to NaN.
    assert plan_generic.w_kernel_scale > 0


@pytest.mark.parametrize("w_const_m", [0.0, 12.5])
@pytest.mark.parametrize("w_strategy", ["dense_scan", "dense_vmap"])
def test_constant_w_fast_vs_generic_dirty2vis(w_const_m: float, w_strategy: str) -> None:
    """Forward operator agrees between fast and generic plans on the same
    coplanar uvw input within ``10 * eps``."""
    uvw = _coplanar_uvw(w_const_m=w_const_m)
    plan_fast, plan_generic = _build_pair(uvw)

    rng = np.random.default_rng(0)
    image = jnp.asarray(rng.standard_normal((64, 64)))

    vis_fast = np.asarray(dirty2vis(plan_fast, image, w_strategy=w_strategy))
    vis_generic = np.asarray(dirty2vis(plan_generic, image, w_strategy=w_strategy))

    assert np.all(np.isfinite(vis_fast)), "fast-path produced non-finite output"
    assert np.all(np.isfinite(vis_generic)), "generic-path produced non-finite output"
    err = np.linalg.norm(vis_fast - vis_generic) / np.linalg.norm(vis_generic)
    assert err < _TOL, f"relative error {err:.3e} exceeds {_TOL:.3e}"


@pytest.mark.parametrize("w_const_m", [0.0, 12.5])
@pytest.mark.parametrize("w_strategy", ["dense_scan", "dense_vmap"])
def test_constant_w_fast_vs_generic_vis2dirty(w_const_m: float, w_strategy: str) -> None:
    """Adjoint operator agrees between fast and generic plans on the same
    coplanar uvw input within ``10 * eps``."""
    uvw = _coplanar_uvw(w_const_m=w_const_m)
    plan_fast, plan_generic = _build_pair(uvw)

    rng = np.random.default_rng(1)
    n_rows = uvw.shape[0]
    vis = jnp.asarray(
        rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))
    )

    dirty_fast = np.asarray(vis2dirty(plan_fast, vis, w_strategy=w_strategy))
    dirty_generic = np.asarray(vis2dirty(plan_generic, vis, w_strategy=w_strategy))

    assert np.all(np.isfinite(dirty_fast))
    assert np.all(np.isfinite(dirty_generic))
    err = np.linalg.norm(dirty_fast - dirty_generic) / np.linalg.norm(dirty_generic)
    assert err < _TOL, f"relative error {err:.3e} exceeds {_TOL:.3e}"
