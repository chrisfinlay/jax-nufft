"""Tests for the exp-of-semicircle kernel and its numerical FT."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft.kernel import (
    PhiHatTable,
    compute_phi_hat_table,
    kernel_params,
    phi,
    phi_numpy,
)


def test_kernel_params_basic() -> None:
    w, beta = kernel_params(1e-6)
    assert isinstance(w, int)
    assert w >= 2
    # Spec rule: W = ceil(-log10(eps) * 2/pi) + 2 = ceil(6 * 2/pi) + 2 = 4 + 2 = 6
    assert w == math.ceil(6 * 2 / math.pi) + 2
    assert beta == pytest.approx(2.30 * w)


def test_kernel_params_monotonic_in_epsilon() -> None:
    """Tighter accuracy must not give a smaller kernel width."""
    eps_list = [1e-2, 1e-4, 1e-6, 1e-8, 1e-12]
    widths = [kernel_params(e)[0] for e in eps_list]
    assert widths == sorted(widths)


def test_kernel_params_invalid_epsilon() -> None:
    with pytest.raises(ValueError):
        kernel_params(0.0)
    with pytest.raises(ValueError):
        kernel_params(-1e-6)
    with pytest.raises(ValueError):
        kernel_params(1.5)


def test_phi_basic_values() -> None:
    beta = 10.0
    # phi(0) = exp(beta * 0) = 1
    assert phi_numpy(np.array(0.0), beta) == pytest.approx(1.0)
    # phi(+/-1) = exp(beta * (-1)) = exp(-beta)
    assert phi_numpy(np.array(1.0), beta) == pytest.approx(math.exp(-beta), abs=1e-12)
    assert phi_numpy(np.array(-1.0), beta) == pytest.approx(math.exp(-beta), abs=1e-12)
    # phi outside |z| <= 1 is exactly 0
    assert phi_numpy(np.array(1.5), beta) == 0.0
    assert phi_numpy(np.array(-1.5), beta) == 0.0


def test_phi_jnp_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    z = rng.uniform(-1.5, 1.5, size=200)
    beta = 13.8  # 2.30 * 6
    expect = phi_numpy(z, beta)
    got = np.asarray(phi(jnp.asarray(z), beta))
    np.testing.assert_allclose(got, expect, rtol=1e-6, atol=1e-7)


def test_phi_is_jit_traceable() -> None:
    beta = 13.8
    fn = jax.jit(lambda z: phi(z, beta))
    out = fn(jnp.linspace(-1.2, 1.2, 17))
    assert out.shape == (17,)
    # phi(0) = 1
    centre = float(fn(jnp.array(0.0)))
    assert centre == pytest.approx(1.0, abs=1e-7)


def test_phi_hat_symmetry() -> None:
    """phi(z) is real and symmetric, so phi_hat(eta) is real and symmetric."""
    table = compute_phi_hat_table(beta=13.8, eta_max_request=2.0)
    # The table is built so that values[n//2] == phi_hat(0) and values[n//2 - k]
    # mirrors values[n//2 + k] up to FFT roundoff. Skip the very-edge-of-table
    # bin where wraparound can introduce tiny asymmetries.
    n = table.n
    centre = n // 2
    # Inspect a generous interior swath rather than just a few bins.
    span = min(centre, 1024)
    left = table.values[centre - span : centre]
    right = table.values[centre + 1 : centre + span + 1]
    np.testing.assert_allclose(left, right[::-1], rtol=1e-6, atol=1e-9)


def test_phi_hat_positive_in_range() -> None:
    table = compute_phi_hat_table(beta=13.8, eta_max_request=1.5)
    eta = np.linspace(-1.5, 1.5, 401)
    vals = table.evaluate(eta)
    assert np.all(vals > 0)


def test_phi_hat_against_direct_quadrature() -> None:
    """Compare FFT-based phi_hat against a high-resolution trapezoidal quadrature.

    This is the *defining* property of the table: it must approximate the
    continuous Fourier integral to better than the requested epsilon.
    """
    beta = 13.8
    table = compute_phi_hat_table(beta=beta, eta_max_request=1.0)
    # Direct quadrature with a very fine grid.
    n_quad = 200_001  # odd, so z=0 is hit
    z = np.linspace(-1.0, 1.0, n_quad)
    dz = z[1] - z[0]
    phi_z = phi_numpy(z, beta)
    eta_test = np.linspace(-1.0, 1.0, 21)
    direct = np.array(
        [np.sum(phi_z * np.exp(-2j * np.pi * e * z)).real * dz for e in eta_test]
    )
    interp = table.evaluate(eta_test)
    # Cubic interpolation on the FFT table should match direct quadrature
    # to well below 1e-6 in this regime (oversample=8, n_fine=4096).
    np.testing.assert_allclose(interp, direct, rtol=1e-6, atol=1e-7)


def test_phi_hat_table_evaluate_outside_range_raises() -> None:
    table = compute_phi_hat_table(beta=13.8, eta_max_request=1.0)
    with pytest.raises(ValueError):
        table.evaluate(np.array([table.eta_max * 2.0]))


def test_phi_hat_table_safety_floor_triggers() -> None:
    """If eta_max_request is far enough out, phi_hat decays below the floor."""
    # Large beta -> sharp kernel -> phi_hat decays slowly, so we instead use
    # small beta with large eta_max to drive phi_hat low.
    with pytest.raises(ValueError, match="phi_hat dropped"):
        compute_phi_hat_table(beta=2.0, eta_max_request=10.0, safety_floor=0.1)


def test_phi_hat_table_is_picklable_via_dataclass() -> None:
    """The table is a frozen dataclass so it composes with jax pytrees and caches."""
    table = compute_phi_hat_table(beta=13.8, eta_max_request=1.0)
    assert isinstance(table, PhiHatTable)
    # frozen=True means we can't mutate
    with pytest.raises(Exception):
        table.beta = 0.0  # type: ignore[misc]
