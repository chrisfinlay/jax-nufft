"""Adjoint operator: DFT comparison, dot-product test, weights pass-through.

The DFT-parity bounds here are the same accuracy contract as
``tests/test_against_dft.py``: the exact DFT is the definition of the answer,
so the adjoint must land within ``2 * eps`` of it too. Issue #9 tightened these
from ``10 * eps``, which was loose enough to hide a ``w_kernel_width`` that was
up to three cells short of the requested epsilon.

The dot-product identities below are a different quantity again -- a
reduction-order comparison between two operators (forward vs adjoint), not a
comparison against truth. Issue #10 tightened their bound from ``100 * eps``
(1e-4 at eps=1e-6, ten orders of magnitude looser than what the code
actually achieves) to the eps-independent ``1e-11``: the measured residual
is 1e-16 .. 7e-13 for jax-nufft on these fixtures (ducc0 itself: 1e-14 ..
8e-11, for comparison -- not asserted here since ducc0 is used only as a
black-box test oracle elsewhere).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT

jax.config.update("jax_enable_x64", True)

# Accuracy contract against the exact DFT (issue #9); mirrors
# ``tests/test_against_dft.py::DFT_TOL_FACTOR``.
DFT_TOL_FACTOR = 2.0


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
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
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
    assert err < DFT_TOL_FACTOR * eps, (
        f"relative error {err:.3e} exceeds {DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e}"
    )


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
    assert err < DFT_TOL_FACTOR * eps, (
        f"relative error {err:.3e} exceeds {DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e}"
    )


@pytest.mark.parametrize("eps", [1e-6])
def test_windowed_dot_product_identity(eps: float) -> None:
    """Adjoint relation must also hold for windowed forward/adjoint."""
    rng = np.random.default_rng(34)
    n_l = n_m = 32
    n_rows = 64
    pixsize = 0.01

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-30.0, 30.0, size=n_rows)
    freq = np.array([1.4e9])

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps)
    image = rng.standard_normal((1, n_l, n_m))
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )

    # ``n * x`` undoes the 1/n the adjoint applies on its output relative to
    # the literal adjoint of A (see test_dot_product_identity's docstring for
    # the full derivation). This form is only exact where n_grid > 0, i.e.
    # every pixel inside the unit disc; the fixture's small image is fully
    # inside it here, so no masking is needed. It will be replaced with the
    # cleaner divide_by_n-exposed form once that lands (a later issue).
    n_grid = np.asarray(plan.n_minus_1) + 1.0
    image_n = image * n_grid[None, :, :]

    Ax = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy="windowed_scan"))
    Ay = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy="windowed_scan"))

    lhs = np.vdot(Ax.ravel(), vis.ravel()).real
    rhs = float(np.vdot(image_n.ravel(), Ay.ravel()))
    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    # Issue #10: eps-independent bound, see the module docstring for the
    # measured residual behind 1e-11.
    assert rel_err < 1e-11, f"windowed dot-product rel err {rel_err:.3e}"


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_dot_product_identity(eps: float) -> None:
    """Adjointness check for the wgridder pair, matching ducc's convention.

    With our chosen convention -- forward has no 1/n factor, adjoint applies
    1/n on the output and takes the real part -- the standard complex adjoint
    identity does not hold. The relation that holds for real x is

        Re(<A x, y>_C) = <n * x, A^* y>_R

    (the n multiplier on the RHS undoes the 1/n that A^* applies on its
    output relative to the literal adjoint of A). This identity holds only
    where n_grid = n_minus_1 + 1 > 0, i.e. every pixel is inside the unit
    disc -- outside it ``n - 1`` is ducc's analytic extension (see
    ``planning.make_plan``), not the true ``n``, and the relation above
    breaks down. The fixture's image is fully inside the disc, so no
    masking is needed here.
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
    # Issue #10: eps-independent bound, see the module docstring for the
    # measured residual behind 1e-11.
    assert rel_err < 1e-11, f"dot-product relative error {rel_err:.3e}; lhs={lhs}, rhs={rhs}"


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
    assert err < DFT_TOL_FACTOR * eps


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
