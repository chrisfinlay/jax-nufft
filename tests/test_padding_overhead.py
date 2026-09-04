"""What ``plan.window_padding_overhead`` measures, and the grid behind the cutoffs.

``window_padding_overhead`` is the factor by which a windowed strategy's
row-work exceeds the irreducible minimum. Each ``(channel, plane)`` step
slices a *static* ``max_window_size`` rows out of the w-sorted array (the
shape has to be static for ``lax.scan`` / ``vmap``), so a windowed traversal
touches ``n_chan * n_w * max_window_size`` rows. Only some of those carry a
nonzero kernel weight, and issue #43 is about which ones:

    window_padding_overhead = n_chan * n_w * max_window_size / live_row_count

``live_row_count`` counts the ``(channel, plane, row)`` incidences with
``|w_lambda - w_k| <= w_kernel_scale`` -- i.e. ``|z| <= 1``, the support of
``kernel.phi`` -- measured on the *unpadded* window. The builder then widens
each window by ``window_boundary_margin`` and by one further row at each end
(``lo - 1`` / ``hi + 1``, clamped into range). Those extra rows exist so the
host's window provably contains every row the *device* puts inside support
despite FMA contraction; they are real work, but they carry ``phi = 0``
exactly, so they belong in the numerator, never in the denominator.

In practice the clamp dominates: ``window_boundary_margin`` is a few ulps of
the absolute w scale, so it seldom has a row to catch. Measured over the
forty-cell float64 calibration grid it adds one or two rows on twelve cells
and none on the other twenty-eight, so the live-vs-padded gap is essentially
the two clamp rows per (channel, plane) -- on MWA_extended off30 at eps 1e-3,
the fixture the regression test below uses, it is exactly that and the margin
catches nothing. Both are excluded on principle regardless of how many rows
they happen to move: a padded row is one the *device* might place inside
support under FMA contraction, not one the kernel gives weight. That the gap
is mostly the clamp is also why it matters most where the windows are
narrowest.

Before #43 the denominator was ``window_size.mean()`` over the *padded*
windows, which counted the padding as irreducible work and so understated the
waste -- by up to 16.9% on the repository's calibration fixtures, worst
exactly where the windows are narrowest and the padding therefore relatively
largest. The ``+/-1`` clamp also guarantees ``window_size >= 1``, so a plane
with no live rows at all was indistinguishable from one holding a single row;
``empty_plane_count`` is what makes that visible again.

Two consequences this module pins:

* the ratio is bounded below by 1.0, with equality attained in the degenerate
  single-plane (constant-w) case; and
* restating the diagnostic must not move a single ``auto`` decision on the
  review calibration grid -- which is what forces ``_CPU_PADDING_CUTOFF``
  upward, since the corrected metric reads above the shipped ``5.0`` on the
  MWA_extended off-zenith cells where AGENTS.md section 9 records the windowed
  adjoint as the measured CPU win.
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from jax_nufft.kernel import phi_numpy
from jax_nufft.planning import WGridderPlan, make_plan, window_boundary_margin
from jax_nufft.wgridder import (
    _CPU_PADDING_CUTOFF,
    _GPU_PADDING_CUTOFF,
    _auto_w_strategy_cpu,
    _auto_w_strategy_gpu,
)
from tests.conftest import (
    EDA2,
    GH200_LARGE,
    MEERKAT,
    MWA_COMPACT,
    MWA_EXTENDED,
    X64,
    Telescope,
    requires_x64,
    synthetic_uvw,
)

# The plan dtype has to be requested explicitly (issue #11): a default float64
# plan raises under ``JAX_ENABLE_X64=0``, and this module runs on both legs.
_REAL_DTYPE: DTypeLike = jnp.float64 if X64 else jnp.float32

# ``filterwarnings = error`` plus ``make_plan``'s float32 accuracy warning below
# eps ~ 1e-5 means anything finer than this is float64-only.
_F32_SAFE_EPSILON = 1e-3


def _plan_for_uvw(
    uvw: np.ndarray,
    freq: np.ndarray,
    image_shape: tuple[int, int],
    pixsize: float,
    epsilon: float,
) -> WGridderPlan:
    return make_plan(
        uvw,
        freq,
        image_shape,
        pixsize,
        pixsize,
        epsilon=epsilon,
        dtype=_REAL_DTYPE,
    )


def _uvw_and_freq(telescope: Telescope, zenith_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    return synthetic_uvw(telescope, zenith_angle_deg, seed=0), np.array([telescope.freq_hz])


def _plan_for(telescope: Telescope, zenith_angle_deg: float, epsilon: float) -> WGridderPlan:
    uvw, freq = _uvw_and_freq(telescope, zenith_angle_deg)
    return _plan_for_uvw(uvw, freq, (telescope.n_pix, telescope.n_pix), telescope.pixsize, epsilon)


# The calibration grid builds 40 plans and two tests sweep it (one of them
# twice, forward and adjoint). GH200_large's plan holds ~135 MB of 2048^2
# leaves, so the cache is deliberately tiny: it exists to collapse the
# immediately-repeated build of the *same* cell, not to hold the grid.
@lru_cache(maxsize=2)
def _cached_plan(telescope: Telescope, zenith_angle_deg: float, epsilon: float) -> WGridderPlan:
    return _plan_for(telescope, zenith_angle_deg, epsilon)


# --- independent references --------------------------------------------------


def _w_rel_per_channel(plan: WGridderPlan, chan: int) -> np.ndarray:
    """``w`` in wavelengths relative to ``w0``, in the plan's own convention.

    ``make_plan`` narrows ``uvw`` to ``real_dtype`` before anything else
    (``_coerce_uvw_freq_dtype``), so the float64 array the caller passed in is
    *not* the array the builder measured against on the float32 leg -- read
    the ``uvw_m`` leaf instead. Everything downstream of that is float64 in
    both legs.
    """
    w_m = np.asarray(plan.uvw_m)[:, 2]
    inv_lambda = np.asarray(plan.inv_lambda)
    return (w_m * inv_lambda[chan]).astype(np.float64) - plan.w0


def _independent_live_sizes(plan: WGridderPlan) -> np.ndarray:
    """``(n_chan, n_w)`` count of rows inside each plane's *unpadded* support.

    Deliberately not a second copy of the builder's algorithm: the builder
    sorts by w and binary-searches two boundaries per plane, exploiting the
    fact that support is a contiguous run of the sorted array. This counts a
    dense ``(n_w, n_rows)`` boolean mask over the rows in **input order**, so
    it shares neither the sort, the ``searchsorted`` calls, nor the
    contiguity assumption. It also never sees ``window_boundary_margin`` or
    the ``lo - 1`` / ``hi + 1`` clamp, which is the whole point: those rows
    must not reach the denominator.

    The interval is closed at both ends, matching ``side="left"`` /
    ``side="right"`` and matching ``kernel.phi``, which returns a nonzero
    ``exp(-beta)`` at ``|z| == 1`` exactly.
    """
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)
    scale = plan.w_kernel_scale
    out = np.zeros((plan.n_chan, plan.n_w), dtype=np.int64)
    for c in range(plan.n_chan):
        w_rel = _w_rel_per_channel(plan, c)
        for k in range(plan.n_w):
            out[c, k] = np.count_nonzero(
                (w_rel >= centres[k] - scale) & (w_rel <= centres[k] + scale)
            )
    return out


def _kernel_weighted_incidence_count(plan: WGridderPlan) -> int:
    """``(channel, plane, row)`` incidences whose w-kernel weight is nonzero.

    This is the *semantic* definition of the denominator -- the work a
    windowed traversal cannot avoid -- expressed through the same
    ``kernel.phi`` the operators call, rather than through any interval
    arithmetic. It is a slightly different float computation from
    :func:`_independent_live_sizes` (it forms ``z = (w - w_k) / scale`` and
    lets the division round) so the two can disagree by a row or two on
    fixtures that place a row exactly on ``|z| == 1``; the callers below only
    use it where that does not happen, and say so.
    """
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)
    total = 0
    for c in range(plan.n_chan):
        w_rel = _w_rel_per_channel(plan, c)
        for k in range(plan.n_w):
            z = (w_rel - centres[k]) / plan.w_kernel_scale
            total += int(np.count_nonzero(phi_numpy(z, plan.beta)))
    return total


def _independent_padded_sizes(plan: WGridderPlan) -> np.ndarray:
    """``(n_chan, n_w)`` window lengths *as the operators slice them*.

    i.e. the pre-#43 denominator: the live support widened by
    ``window_boundary_margin`` and then by one row at each end. Reproduced
    here so the tests below can show that those rows are excluded, and so the
    equivalence test can recompute the old rule instead of pinning a table of
    strategy names. Pinned independently by
    ``test_planning.py::test_window_builder_matches_independent_reference``;
    the ``max_window_size`` cross-check below is a cheap guard that this copy
    has not drifted from the builder it stands in for.
    """
    n_rows = plan.n_rows
    sort_perm = np.asarray(plan.sort_perm)
    w_m = np.asarray(plan.uvw_m)[:, 2]
    w_m_sorted = w_m[sort_perm]
    inv_lambda = np.asarray(plan.inv_lambda)
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)
    # ``window_boundary_margin`` is sized from the absolute w range, which
    # ``make_plan`` forms in ``real_dtype`` from the narrowed uvw -- so form it
    # the same way here, or the float32 leg gets a different margin.
    ends_lo = inv_lambda * float(w_m.min())
    ends_hi = inv_lambda * float(w_m.max())
    margin = window_boundary_margin(
        plan.real_dtype,
        float(np.min(np.minimum(ends_lo, ends_hi))),
        float(np.max(np.maximum(ends_lo, ends_hi))),
        plan.w_extent,
    )
    out = np.zeros((plan.n_chan, plan.n_w), dtype=np.int64)
    for c in range(plan.n_chan):
        w_lambda_c = (w_m_sorted * inv_lambda[c]).astype(np.float64) - plan.w0
        lo = np.searchsorted(w_lambda_c, centres - plan.w_kernel_scale - margin, side="left")
        hi = np.searchsorted(w_lambda_c, centres + plan.w_kernel_scale + margin, side="right")
        out[c] = np.minimum(hi + 1, n_rows) - np.maximum(lo - 1, 0)
    assert int(out.max()) == plan.max_window_size, (
        "the padded-window reference has drifted from make_plan's builder"
    )
    return out


def _windowed_row_work(plan: WGridderPlan) -> int:
    """Rows a windowed traversal touches: one static slice per (channel, plane)."""
    return plan.n_chan * plan.n_w * plan.max_window_size


# --- the identity ------------------------------------------------------------


@pytest.mark.parametrize("zenith_angle_deg", [0.0, 30.0])
def test_padding_overhead_is_windowed_work_over_live_work(zenith_angle_deg: float) -> None:
    """``window_padding_overhead == n_chan * n_w * max_window_size / live_row_count``.

    The denominator is taken from :func:`_independent_live_sizes`, not from
    ``plan.live_row_count``, so this is a check of the number and not a
    restatement of the formula. ``epsilon=1e-3`` so it runs on both precision
    legs.
    """
    plan = _plan_for(MWA_EXTENDED, zenith_angle_deg, _F32_SAFE_EPSILON)
    live_sizes = _independent_live_sizes(plan)
    expected_live = int(live_sizes.sum())

    # Non-vacuity: a plan whose planes each hold every row would make the
    # identity true for the padded denominator too.
    assert plan.n_w > 1
    assert plan.max_window_size < plan.n_rows
    assert expected_live > 0

    assert plan.live_row_count == expected_live, (
        "live_row_count must count only rows inside the unpadded kernel support"
    )
    assert plan.empty_plane_count == int(np.count_nonzero(live_sizes == 0))

    assert plan.window_padding_overhead == pytest.approx(_windowed_row_work(plan) / expected_live)
    # A windowed traversal can never touch fewer rows than the live ones it
    # has to visit, so this is a real overhead factor.
    assert plan.window_padding_overhead >= 1.0

    # The same denominator expressed through the kernel the operators
    # actually apply: a row is live iff ``phi`` gives it a nonzero weight.
    # The tolerance is for rows sitting exactly on ``|z| == 1``, where forming
    # ``z`` by division rounds to the other side of the support edge; the
    # measured worst case over the whole forty-cell grid below, in both
    # precision legs, is three rows. Against the ~490-row gap between the live
    # and padded denominators on this fixture that is not slack the defect
    # could hide in.
    assert abs(_kernel_weighted_incidence_count(plan) - plan.live_row_count) <= 4

    # The padded windows are strictly bigger. How much bigger is the whole
    # issue, and is pinned at the pointing where it is large, below.
    assert plan.live_row_count < int(_independent_padded_sizes(plan).sum())


def test_live_row_count_is_the_kernel_width_times_the_rows() -> None:
    """A structural cross-check that shares nothing with the builder at all.

    Planes are spaced ``dw`` apart and the support half-width is ``W/2 * dw``,
    so every row is live in ``W`` planes give or take one boundary effect --
    independent of how clumped w is, which is why this holds on both fixtures
    below while their overheads differ by a factor of three. A denominator
    that had absorbed the padding would exceed ``n_rows * W`` by the margin
    and clamp, i.e. by roughly ``2 * n_w`` rows.
    """
    for telescope, zenith_angle_deg in ((MWA_EXTENDED, 30.0), (MWA_COMPACT, 0.0)):
        plan = _plan_for(telescope, zenith_angle_deg, _F32_SAFE_EPSILON)
        nominal = plan.n_chan * plan.n_rows * plan.w_kernel_width
        # Measured deviation over the whole forty-cell grid, both precision
        # legs: at most two rows per channel. Kept tight on purpose -- a
        # tolerance of order ``n_w`` would let a one-sided clamp through.
        assert abs(plan.live_row_count - nominal) <= 4 * plan.n_chan, (
            f"{telescope.name}: live_row_count {plan.live_row_count} is not ~n_rows * W = {nominal}"
        )


# --- the regression guard: the padding is excluded ---------------------------


def test_padding_rows_are_excluded_from_the_denominator() -> None:
    """The ``+/-1`` clamp rows must not count as irreducible work.

    This is the issue #43 regression guard. Reverting the denominator to
    ``window_size.sum()`` (the padded windows) fails every assertion here:
    the strict inequality, the separation of the two ratios, and the identity.

    MWA_extended *off-zenith* specifically. The padding is two rows per
    (channel, plane) plus the margin, so it only matters where the windows are
    narrow: at 30 degrees this fixture has 248 planes of ~10 live rows each and
    the padded denominator is 20% too large, while at zenith it has 11 planes
    of ~218 and the same padding is 0.5% -- too small a signal to gate
    anything, which is why that pointing is covered by the identity test above
    and not here.
    """
    telescope, zenith_angle_deg = MWA_EXTENDED, 30.0
    plan = _plan_for(telescope, zenith_angle_deg, _F32_SAFE_EPSILON)

    live_sizes = _independent_live_sizes(plan)
    padded_sizes = _independent_padded_sizes(plan)
    live_total = int(live_sizes.sum())
    padded_total = int(padded_sizes.sum())

    # Preconditions, asserted rather than skipped so fixture drift is loud.
    # The padded windows contain the live ones and are strictly bigger, by
    # enough that the two denominators are not within rounding of each other.
    assert np.all(padded_sizes >= live_sizes)
    assert padded_total > live_total * 1.15, (
        "fixture no longer carries enough padding for this test to gate anything"
    )

    assert plan.live_row_count == live_total
    assert plan.live_row_count < padded_total, (
        f"live_row_count ({plan.live_row_count}) counts padded rows: the "
        f"padded windows hold {padded_total} rows, the live ones {live_total}"
    )

    windowed = _windowed_row_work(plan)
    padded_overhead = windowed / padded_total
    assert plan.window_padding_overhead == pytest.approx(windowed / live_total)
    assert plan.window_padding_overhead > padded_overhead * 1.15, (
        f"window_padding_overhead ({plan.window_padding_overhead:.4f}) is at "
        f"the padded-denominator value ({padded_overhead:.4f}); the "
        f"boundary_margin / +-1 clamp rows are back in the denominator"
    )


def test_empty_planes_are_counted_even_though_the_clamp_hides_them() -> None:
    """``empty_plane_count`` sees the planes ``window_size >= 1`` cannot.

    MWA_extended at 30 degrees is the fixture with many narrow windows and a
    long tail of planes holding no rows at all. The precondition is asserted,
    not skipped: if the fixture ever stops producing empty planes this must
    fail loudly rather than pass vacuously.
    """
    telescope, zenith_angle_deg = MWA_EXTENDED, 30.0
    plan = _plan_for(telescope, zenith_angle_deg, _F32_SAFE_EPSILON)

    live_sizes = _independent_live_sizes(plan)
    empty = int(np.count_nonzero(live_sizes == 0))
    assert empty > 0, (
        "MWA_extended off30 must have planes with no live rows, or the "
        "emptiness assertions below gate nothing"
    )
    assert plan.empty_plane_count == empty

    # The reason a padded count cannot report this: the insurance clamp gives
    # every window at least one row, so emptiness is invisible downstream of it.
    padded_sizes = _independent_padded_sizes(plan)
    assert padded_sizes[live_sizes == 0].min() >= 1
    assert int(np.count_nonzero(padded_sizes == 0)) == 0


@requires_x64
@pytest.mark.parametrize(
    ("epsilon", "expected_empty"), [(1e-3, 87), (1e-6, 62), (1e-9, 45), (1e-12, 32)]
)
def test_empty_plane_count_matches_the_measured_table(epsilon: float, expected_empty: int) -> None:
    """The measured empty-plane table on the issue's worst fixture.

    ``requires_x64``: epsilon below ~1e-5 cannot be asked of a float32 plan
    without a warning, and ``filterwarnings = error`` makes that a failure.
    The float32 leg covers the same fixture at ``_F32_SAFE_EPSILON`` above,
    where it reads 90 rather than 87 -- rows on a support boundary land on
    the other side once ``inv_lambda`` is single precision, which is why this
    exact table is float64-only.
    """
    plan = _plan_for(MWA_EXTENDED, 30.0, epsilon)
    assert plan.empty_plane_count == expected_empty
    assert plan.empty_plane_count == int(np.count_nonzero(_independent_live_sizes(plan) == 0))


# --- bounds ------------------------------------------------------------------


def test_padding_overhead_lower_bound_is_attained() -> None:
    """1.0 exactly in the degenerate single-plane case.

    Constant ``w`` collapses the plan to one plane holding every row
    (``planning.py``'s fast path): every ``(channel, row)`` incidence is live,
    ``max_window_size == n_rows``, and windowed traversal wastes nothing.
    """
    rng = np.random.default_rng(0)
    uvw = np.zeros((128, 3))
    uvw[:, :2] = rng.uniform(-100.0, 100.0, size=(128, 2))
    freq = np.array([1.4e9, 1.5e9])
    plan = _plan_for_uvw(uvw, freq, (64, 64), 2e-3, _F32_SAFE_EPSILON)

    assert plan.n_w == 1
    assert plan.max_window_size == plan.n_rows
    assert plan.live_row_count == plan.n_chan * plan.n_rows
    assert plan.empty_plane_count == 0
    assert plan.window_padding_overhead == 1.0
    assert plan.window_padding_overhead == _windowed_row_work(plan) / plan.live_row_count


def test_padding_overhead_is_never_below_one() -> None:
    """The bound holds across the fixtures, not only in the degenerate case."""
    for telescope in (EDA2, MWA_COMPACT, MWA_EXTENDED, MEERKAT):
        for zenith_angle_deg in (0.0, 30.0):
            plan = _plan_for(telescope, zenith_angle_deg, _F32_SAFE_EPSILON)
            assert plan.window_padding_overhead >= 1.0
            assert plan.live_row_count <= _windowed_row_work(plan)


# --- the calibration grid ----------------------------------------------------
#
# The grid the CPU and GPU padding cutoffs are calibrated on: five review
# fixtures x two pointings x four epsilons. Corrected-metric measurements on
# this grid (float64; the float32 leg agrees to <0.3% at eps=1e-3):
#
#   * every cell except MWA_extended off30 reads 1.077 - 3.173;
#   * MWA_extended off30 reads 5.173 (eps 1e-12) to 5.784 (eps 1e-3), against
#     4.712 - 4.933 on the padded denominator -- an understatement of up to
#     16.9%, worst where the windows are narrowest.
#
# That last row is what moves ``_CPU_PADDING_CUTOFF``. The shipped 5.0 sat
# just above the padded-scale maximum of 4.933; on the corrected scale the
# same cells read up to 5.784 (5.791 on the float32 leg), so leaving the
# cutoff at 5.0 would flip the four MWA_extended off30 adjoint cells from
# ``windowed_scan`` to ``dense_scan`` -- contradicting the measured CPU win
# in AGENTS.md section 9. 6.0 is the smallest value that both clears the grid
# and preserves the old cutoff's proportional headroom: 5.0 times the worst
# measured padded->live inflation on the grid (5.784 / 4.809 = 1.203) is
# 6.014.
#
# ``_GPU_PADDING_CUTOFF`` does *not* move: no cell crosses 3.0 in either
# direction (the three cells near it -- MWA_compact off30, MeerKAT off30 at
# eps 1e-3 -- are above it on both scales, and EDA2 off30 is below on both).

_FIXTURES = [EDA2, MWA_COMPACT, MWA_EXTENDED, MEERKAT, GH200_LARGE]
_POINTINGS = [0.0, 30.0]
_EPSILONS = [1e-3, 1e-6, 1e-9, 1e-12]
_GRID = [
    pytest.param(
        telescope,
        zenith_angle_deg,
        epsilon,
        id=f"{telescope.name}-za{int(zenith_angle_deg)}-eps{epsilon:.0e}",
        # Only eps >= 1e-5 can be asked of a float32 plan without tripping
        # make_plan's accuracy warning (and filterwarnings = error). The
        # remaining cells still give the float32 leg ten of the forty.
        marks=() if epsilon >= 1e-5 else (requires_x64,),
    )
    for epsilon in _EPSILONS
    for telescope in _FIXTURES
    for zenith_angle_deg in _POINTINGS
]

# The one fixture / pointing whose corrected overhead lands above the shipped
# 5.0 -- and the one where the windowed adjoint is the measured CPU win
# (AGENTS.md section 9), so it must stay on ``windowed_scan``.
_HIGH_OVERHEAD_FIXTURE = MWA_EXTENDED.name
_HIGH_OVERHEAD_POINTING = 30.0
# Measured min/max over the grid with a little rounding room -- not a
# tolerance on a converging quantity.
_ORDINARY_OVERHEAD_RANGE = (1.0, 3.3)
_HIGH_OVERHEAD_RANGE = (5.0, 5.9)


def test_cpu_padding_cutoff_clears_the_corrected_grid_maximum() -> None:
    """The cutoff has to be restated upward, and 6.0 is the measured floor.

    Stated as its own test so the number is auditable without running the
    forty-cell sweep: the grid maximum is 5.784 (float64) / 5.791 (float32),
    and the guard is only a guard if it sits above every review fixture.
    """
    assert _CPU_PADDING_CUTOFF >= 6.0, (
        "the corrected window_padding_overhead reaches 5.79 on MWA_extended "
        "off30; a cutoff at or below that flips the measured CPU win to "
        "dense_scan"
    )
    # The GPU cutoff is unchanged by the redefinition; pin that it did not
    # get dragged along.
    assert _GPU_PADDING_CUTOFF == 3.0


@pytest.mark.parametrize(("telescope", "zenith_angle_deg", "epsilon"), _GRID)
def test_calibration_grid_padding_overhead(
    telescope: Telescope, zenith_angle_deg: float, epsilon: float
) -> None:
    """Each grid cell's corrected overhead sits in its measured band.

    The bands are what changed: on the padded denominator the MWA_extended
    off30 cells read 4.71 - 4.93 and would miss ``_HIGH_OVERHEAD_RANGE``
    entirely.
    """
    plan = _cached_plan(telescope, zenith_angle_deg, epsilon)
    overhead = plan.window_padding_overhead

    is_high = (
        telescope.name == _HIGH_OVERHEAD_FIXTURE and zenith_angle_deg == _HIGH_OVERHEAD_POINTING
    )
    lo, hi = _HIGH_OVERHEAD_RANGE if is_high else _ORDINARY_OVERHEAD_RANGE
    assert lo <= overhead <= hi, f"{overhead:.4f} outside the measured band ({lo}, {hi})"

    # The guard is a guard: no review fixture reaches it at any epsilon.
    assert overhead < _CPU_PADDING_CUTOFF


@pytest.mark.parametrize("is_adjoint", [False, True], ids=["forward", "adjoint"])
@pytest.mark.parametrize(("telescope", "zenith_angle_deg", "epsilon"), _GRID)
def test_calibration_grid_auto_picks_survive_the_redefinition(
    telescope: Telescope,
    zenith_angle_deg: float,
    epsilon: float,
    is_adjoint: bool,
) -> None:
    """Redefining the diagnostic must not move a single ``auto`` decision.

    The expectation is recomputed here from the pre-#43 rule -- the padded
    denominator, against the old 5.0 / 3.0 cutoffs -- rather than written down
    as a table of strategy names, so what is asserted is *equivalence*: not
    which strategy each cell gets, but that redefining the diagnostic and
    restating the cutoff moved none of them. The bite is the four
    MWA_extended off30 adjoint cells: their corrected overhead is 5.17 - 5.79
    against a padded 4.71 - 4.93, so any ``_CPU_PADDING_CUTOFF`` at or below
    5.79 flips them to ``dense_scan`` and this fails.
    """
    plan = _cached_plan(telescope, zenith_angle_deg, epsilon)

    padded_sizes = _independent_padded_sizes(plan)
    nonzero = padded_sizes[padded_sizes > 0]
    previous_overhead = float(padded_sizes.max()) / float(nonzero.mean())

    n_w, width = plan.n_w, plan.w_kernel_width
    if n_w <= width + 2:
        expected_cpu, expected_gpu = "dense_scan", "dense_vmap"
    elif previous_overhead > 5.0:
        expected_cpu, expected_gpu = "dense_scan", "dense_vmap"
    else:
        expected_cpu = "windowed_scan" if is_adjoint and n_w / width > 2.0 else "dense_scan"
        if previous_overhead > 3.0 or plan.n_rows < 10_000:
            expected_gpu = "dense_vmap"
        elif is_adjoint or n_w <= 3.0 * width:
            expected_gpu = "windowed_vmap"
        else:
            expected_gpu = "dense_vmap"

    assert _auto_w_strategy_cpu(plan, is_adjoint=is_adjoint) == expected_cpu
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == expected_gpu
