"""Tests for the exp-of-semicircle kernel and its numerical FT."""

from __future__ import annotations

import dataclasses
import itertools
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
    phi_hat_oversample_for_w,
    phi_numpy,
)
from jax_nufft.planning import W_OVERSAMPLE_X0


def test_kernel_params_basic() -> None:
    w, beta = kernel_params(1e-6)
    assert isinstance(w, int)
    assert w >= 2
    # FINUFFT rule at sigma = 2: W = ceil(-log10(eps/10)) = ceil(7) = 7.
    assert w == math.ceil(-math.log10(1e-6 / 10.0))
    assert w == 7
    assert beta == pytest.approx(2.30 * w)


# The FINUFFT width rule at upsampling factor sigma = 2 ([Barnett+2019] eq. 10,
# implemented in FINUFFT's ``setup_spreader``): ``W = ceil(-log10(eps/10))``,
# i.e. "one more than the number of digits requested". The expected integers are
# spelled out rather than only recomputed from the formula so that a future edit
# to ``kernel_params`` has to change this table too, deliberately.
#
# The rule the repo used before issue #9 -- ``W = ceil(-log10(eps)*2/pi) + 2``
# -> 4, 5, 6, 6, 7, 8, 9, 9, 10, 10 for the same epsilons -- is short by 0-3
# cells and, worse, maps 1e-5 and 1e-6 (and 1e-9/1e-10, 1e-11/1e-12) onto the
# *same* width, so asking for a tighter epsilon bought no accuracy at all.
_EXPECTED_WIDTHS: dict[float, int] = {
    1e-3: 4,
    1e-4: 5,
    1e-5: 6,
    1e-6: 7,
    1e-7: 8,
    1e-8: 9,
    1e-9: 10,
    1e-10: 11,
    1e-11: 12,
    1e-12: 13,
}


@pytest.mark.parametrize(("eps", "expected_w"), sorted(_EXPECTED_WIDTHS.items(), reverse=True))
def test_kernel_params_follows_finufft_width_rule(eps: float, expected_w: int) -> None:
    """``W = ceil(-log10(eps/10))`` at sigma = 2, with beta = 2.30 * W."""
    w, beta = kernel_params(eps)
    assert w == expected_w, f"eps={eps:g}: expected W={expected_w}, got {w}"
    # Same value stated as the formula, so the table above and the rule cannot
    # drift apart silently.
    assert w == math.ceil(-math.log10(eps / 10.0))
    # The module documents beta = 2.30 * W ([Barnett+2019] eq. 10 for sigma = 2);
    # the wider kernel must carry the matching shape parameter, otherwise the
    # aliasing floor exp(-pi*W*sqrt(1 - 1/sigma)) does not move with W.
    assert beta == pytest.approx(2.30 * expected_w)


def test_kernel_params_width_increases_by_one_per_digit() -> None:
    """One extra cell per requested digit -- no plateaus.

    This is the property the old rule violated: ``ceil(-log10(eps)*2/pi) + 2``
    grows by 2/pi < 1 per digit, so consecutive epsilons repeatedly collapsed
    onto one width.
    """
    eps_list = [10.0**-k for k in range(3, 13)]
    widths = [kernel_params(e)[0] for e in eps_list]
    deltas = [b - a for a, b in itertools.pairwise(widths)]
    assert deltas == [1] * len(deltas), f"widths {widths} do not step by one per digit"


def test_kernel_params_rejects_unreachable_epsilon() -> None:
    """Below 1e-14 the rule would ask for W > 15, beyond what the table and the
    float64 kernel can deliver, so ``kernel_params`` must refuse rather than
    silently return a width it cannot honour."""
    # 1e-14 is the tightest supported request (W = 15).
    assert kernel_params(1e-14)[0] == 15
    with pytest.raises(ValueError):
        kernel_params(1e-15)
    with pytest.raises(ValueError):
        kernel_params(1e-16)


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
    direct = np.array([np.sum(phi_z * np.exp(-2j * np.pi * e * z)).real * dz for e in eta_test])
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


# The full schedule, W = 4 through 15 (the whole range kernel_params can ever
# produce), pinned to concrete integers rather than re-derived from the
# formula. This is deliberate, and it is the cheap half of the regression
# guard for the schedule calibrated in phi_hat_oversample_for_w's docstring:
# it costs nothing (no phi_hat table is built), but it is also the one that
# actually catches a regression back to the schedule issue #9's review found
# one doubling too generous (``min(2048, 128 * 2 ** (w_kernel_width - 9))``,
# equivalently a flat cap at 1024 for every W >= 13) -- the numerical
# off-node tests below only run W up to 13 by default and W up to 14 with
# --runslow (see the comment on test_phi_hat_interpolation_error_off_node_slow
# for why W = 15's reference table is left out of even that leg), so a
# regressed cap at W = 15 alone would not be caught numerically anywhere in
# this suite without this pin.
_EXPECTED_OVERSAMPLE: dict[int, int] = {
    4: 32,
    5: 64,
    6: 64,
    7: 64,
    8: 64,
    9: 128,
    10: 128,
    11: 256,
    12: 512,
    13: 1024,
    14: 2048,
    15: 4096,
}


def test_phi_hat_oversample_schedule_is_pinned() -> None:
    """Pin the full ``phi_hat_oversample_for_w`` schedule, W = 4 through 15."""
    got = {w: phi_hat_oversample_for_w(w) for w in _EXPECTED_OVERSAMPLE}
    assert got == _EXPECTED_OVERSAMPLE, got


# W = 11 and 13 are the widths the FINUFFT rule asks for at eps = 1e-10 and
# 1e-12 (issue #9). They are included here so that widening the width rule
# cannot outrun ``phi_hat_oversample_for_w``: eta_max = x0 * W / 2 grows with W,
# and if the table's oversample stops keeping up, this is where it shows.
@pytest.mark.parametrize("w", [4, 6, 8, 10, 11, 13])
def test_phi_hat_conditioning_at_v011_eta_max(w: int) -> None:
    """phi_hat must stay above the safety floor over the full v0.1.1 eta range.

    With the W-independent ``x0 = W_OVERSAMPLE_X0`` sampling, the maximum
    eta encountered by the image-domain correction is ``x0 * W / 2``. This
    grows with W, so the recommended oversample is W-dependent. Here we
    verify that for each W the resulting phi_hat table is well-conditioned
    and that cubic-Lagrange interpolation matches a high-resolution
    quadrature.
    """
    beta = 2.30 * w
    eta_max = W_OVERSAMPLE_X0 * w / 2.0
    table = compute_phi_hat_table(
        beta=beta,
        eta_max_request=eta_max,
        oversample=phi_hat_oversample_for_w(w),
    )
    eta_grid = np.linspace(-eta_max, eta_max, 401)
    interp = table.evaluate(eta_grid)
    # Conditioning: phi_hat must not collapse on the supported range.
    assert np.all(interp > 0)
    # Interpolation accuracy: compare against a fine direct quadrature.
    n_quad = 200_001
    z = np.linspace(-1.0, 1.0, n_quad)
    dz = z[1] - z[0]
    phi_z = phi_numpy(z, beta)
    eta_check = np.linspace(-eta_max, eta_max, 17)
    direct = np.array([np.sum(phi_z * np.exp(-2j * np.pi * e * z)).real * dz for e in eta_check])
    interp_check = table.evaluate(eta_check)
    rel_err = np.max(np.abs(interp_check - direct) / np.abs(direct))
    # Cubic-Lagrange on a 1/(2*oversample)-spaced grid should comfortably
    # beat 1e-7 on the smooth phi_hat for our oversample defaults.
    assert rel_err < 1e-7, f"W={w}: phi_hat interp rel_err = {rel_err:.3e}"


def _phi_hat_interpolation_error_off_node(w: int) -> None:
    """Shared body for the fast and ``--runslow`` off-node interpolation checks.

    ``test_phi_hat_conditioning_at_v011_eta_max`` above samples
    ``linspace(-eta_max, eta_max, 17)``, and with ``eta_max = x0 * W / 2`` and
    an eta step of ``1 / (2 * oversample)`` every one of those points lands
    exactly on a table node, where cubic Lagrange reproduces the node value by
    construction. It therefore measures the table's quadrature error and not
    its interpolation error, which is what issue #9 tripped over: at W = 13 the
    flat oversample of 128 carried a 5.9e-11 off-node error -- invisible to
    that test -- and ``phi_hat_n`` is *divided* into the image, so it showed up
    one-for-one as 1.4e-11 relative in the operator output, 14x the requested
    eps = 1e-12.

    The reference here is the same table at 4x the oversample: its own
    interpolation error is 1/256 of the one under test (the error goes like
    ``eta_step**4``), so the difference is the error of the coarse table to
    better than a percent, and the shared ``n_fine`` discretisation cancels.
    """
    eps = 10.0 ** -(w - 1)  # the epsilon this width is chosen for (kernel_params)
    beta = 2.30 * w
    eta_max = W_OVERSAMPLE_X0 * w / 2.0
    oversample = phi_hat_oversample_for_w(w)
    table = compute_phi_hat_table(beta=beta, eta_max_request=eta_max, oversample=oversample)
    fine = compute_phi_hat_table(beta=beta, eta_max_request=eta_max, oversample=4 * oversample)
    # Deliberately off-node: an odd number of points over the range plus a
    # small irrational-ish offset, so no sample coincides with a table node.
    eta = np.linspace(-eta_max, eta_max, 1999) + 1e-4
    eta = eta[np.abs(eta) <= eta_max]
    rel_err = float(
        np.max(np.abs(table.evaluate(eta) - fine.evaluate(eta)) / np.abs(fine.evaluate(eta)))
    )
    assert rel_err < eps / 10.0, (
        f"W={w} (eps={eps:.0e}, oversample={oversample}): phi_hat interpolation "
        f"error {rel_err:.3e} exceeds eps/10={eps / 10.0:.3e}"
    )


@pytest.mark.parametrize("w", [4, 8, 10])
def test_phi_hat_interpolation_error_off_node(w: int) -> None:
    """Default-suite leg: widths up to 10, where both tables stay small.

    At W = 10 the 4x-oversample reference is ``phi_hat_oversample_for_w(10) *
    4 = 512``, an ``n_fft = n_fine * 512 ~= 2.1M``-element table (tens of MB).
    W in {11, 12, 13, 14} moved to
    :func:`test_phi_hat_interpolation_error_off_node_slow` below (and W = 15
    is skipped there too -- see that test's docstring): at W = 13 the
    reference alone is oversample 4096 (``n_fft ~= 16.8M``), and building it
    alongside the table under test previously reached multiple GB of
    transient host memory -- in the *default* suite, on every pull request.
    See that test and ``phi_hat_oversample_for_w`` for the memory accounting
    (issue #9 follow-up).
    """
    _phi_hat_interpolation_error_off_node(w)


@pytest.mark.parametrize("w", [11, 12, 13, 14])
def test_phi_hat_interpolation_error_off_node_slow(w: int, pytestconfig: pytest.Config) -> None:
    """``--runslow`` leg: widths where the 4x-oversample reference gets big.

    Gated on the same ``--runslow`` flag as the long-telescope parity tests
    (rather than the fixture-based gating in ``conftest.py``, which only
    covers ``long_telescope_pointing``) so an ordinary CI runner doing
    ``pytest -q`` never has to hold the W = 13 pair (table + oversample-4096
    reference) in memory at once. Skipped, not xfailed, without the flag --
    this is a cost gate, not a correctness gate.

    W = 15 (eps = 1e-14, the tightest ``kernel_params`` allows) is left out
    even here: its reference is oversample 16384 (``n_fft = n_fine * 16384
    ~= 67M``), and measured with ``/usr/bin/time -l`` this one parametrised
    case alone pushes ``pytest -q --runslow tests/test_kernel.py`` from
    ~1.2 GB to ~5.5 GB peak memory footprint (~2.3 GB with only W <= 14) --
    on top of the ducc parity legs --runslow also runs, that risks OOMing a
    standard CI runner. W = 15's *numerical* off-node check is therefore
    skipped everywhere; the schedule regression guard for it is
    ``test_phi_hat_oversample_schedule_is_pinned`` above (cheap, and it does
    cover W = 15), backed by these measured numbers (against a common,
    far-finer independent reference, eps/10 = 1e-15): os=1024 -> 1.08e-14
    (fails), os=2048 -> 1.16e-15 (fails, marginally), os=4096 -> 6.57e-16
    (passes) -- confirming the schedule's os=4096 at W = 15 is both
    necessary and sufficient without re-deriving it inside a test.
    """
    if not pytestconfig.getoption("--runslow"):
        pytest.skip("needs --runslow")
    _phi_hat_interpolation_error_off_node(w)


def test_phi_hat_table_is_picklable_via_dataclass() -> None:
    """The table is a frozen dataclass so it composes with jax pytrees and caches."""
    table = compute_phi_hat_table(beta=13.8, eta_max_request=1.0)
    assert isinstance(table, PhiHatTable)
    # frozen=True means we can't mutate
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.beta = 0.0  # type: ignore[misc]
