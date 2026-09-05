"""What ``plan.window_padding_overhead`` measures, and the grid behind the cutoffs.

``window_padding_overhead`` is the factor by which a windowed strategy's
row-work exceeds the irreducible minimum. Each ``(channel, plane)`` step
slices a *static* ``max_window_size`` rows out of the w-sorted array (the
shape has to be static for ``lax.scan`` / ``vmap``), so a windowed traversal
touches ``n_chan * n_w * max_window_size`` rows. Only some of those lie inside
a plane's kernel support, and issue #43 is about which ones:

    window_padding_overhead = n_chan * n_w * max_window_size / live_row_count

``live_row_count`` counts the ``(channel, plane, row)`` incidences with
``|w_lambda - w_k| <= w_kernel_scale``, measured on the *unpadded* window and
on the host's own w. That is the **nominal support**, and the name ``live`` is
shorthand for it -- not for "the incidences ``kernel.phi`` weights", which is a
close but genuinely different set. The operators derive their own w inside the
JIT (``wgridder._channel_ft_coords``, where XLA may contract the multiply and
the ``- w0`` into one FMA, and where a float32 plan runs the whole chain in
single precision), form ``z = fl(fl(w - w_k) / S)``, and test ``|z| <= 1``.
Measured against that compiled expression over the calibration grid, the two
counts differ on 20 of the 40 cells in float64 and 7 of the 10 in float32,
never by more than 3 incidences and never by more than 0.19%.

The nominal count is what the denominator of a work ratio wants. Three
incidences in several thousand is far below the resolution the ratio is read
at, an exact census would cost an ``(n_w, n_rows)`` ``phi`` evaluation per
channel at plan time, and it would in any case be a property of a compiled
executable rather than of the plan. Nothing in this module claims otherwise:
:func:`_kernel_weighted_incidence_count` is a *host-side* cross-check on the
nominal count, and says so.

The builder then widens each window by ``window_boundary_margin`` and by one
further row at each end (``lo - 1`` / ``hi + 1``, clamped into range). Those
extra rows exist so the host's window provably contains every row the *device*
puts inside support despite FMA contraction; they are real work, and they lie
outside nominal support, so they belong in the numerator and not in the
denominator. Not equally far outside it, though: a clamp row is outside by two
whole rows and ``phi`` does return zero for it, whereas a margin row is
outside by a few ulps and ``phi`` may not -- that possibility is the margin's
entire reason for existing.

In practice the clamp dominates: ``window_boundary_margin`` is a few ulps of
the absolute w scale, so it seldom has a row to catch. Measured over the
forty-cell float64 calibration grid it adds one or two rows on twelve cells
and none on the other twenty-eight, so the live-vs-padded gap is essentially
the two clamp rows per (channel, plane) -- on MWA_extended off30 at eps 1e-3,
the fixture the regression test below uses, it is exactly that and the margin
catches nothing. Both are excluded on principle regardless of how many rows
they happen to move: a padded row is outside nominal support, which is the
predicate being counted, and a margin row is one the *device* might well place
inside support under FMA contraction -- that possibility being the margin's
reason to exist. That the gap
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

**Two plan geometries (issue #17).** The Hermitian w-sign fold halves
``w_extent``, hence ``n_w``, hence the whole window layout, so every measured
number here belongs to one geometry or the other. The default is the folded
one, because that is what ``make_plan`` builds for a caller who says nothing;
the unfolded leg is kept where it is cheap (the calibration grid and the
empty-plane table are swept in both) and is *required* by the two equivalence
tests, whose subject is the v0.1.2 rule against the shipped one -- a comparison
between two rules that were both calibrated on the unfolded geometry. Which
setting each call site takes, and why, is recorded at ``_plan_for_uvw``.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache, partial

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
    *,
    hermitian: bool,
) -> WGridderPlan:
    """Build a plan at this module's precision leg.

    ``hermitian`` (issue #17) has no default here on purpose. The fold halves
    ``w_extent``, hence ``n_w``, hence ``max_window_size`` and the whole window
    layout, so every measured number in this module belongs to one geometry or
    the other and no call site may leave the question open. The two geometries
    it selects between are:

    * ``True`` -- what ships. The calibration grid's overhead bands, the
      empty-plane table and every structural identity below are measured on
      this one, because it is what a caller who does not pass ``hermitian``
      gets.
    * ``False`` -- the pre-#17 geometry. Kept where the *subject* of the test
      is issue #43's change of metric rather than the plan the library builds:
      the two equivalence tests compare the v0.1.2 rule against the shipped
      one, and both were written against the unfolded geometry (see the note
      above ``test_calibration_grid_auto_picks_survive_the_redefinition``).
    """
    return make_plan(
        uvw,
        freq,
        image_shape,
        pixsize,
        pixsize,
        epsilon=epsilon,
        dtype=_REAL_DTYPE,
        hermitian=hermitian,
    )


def _uvw_and_freq(telescope: Telescope, zenith_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    return synthetic_uvw(telescope, zenith_angle_deg, seed=0), np.array([telescope.freq_hz])


def _plan_for(
    telescope: Telescope, zenith_angle_deg: float, epsilon: float, *, hermitian: bool = True
) -> WGridderPlan:
    """A review-fixture plan; ``hermitian`` defaults to the shipped geometry."""
    uvw, freq = _uvw_and_freq(telescope, zenith_angle_deg)
    return _plan_for_uvw(
        uvw,
        freq,
        (telescope.n_pix, telescope.n_pix),
        telescope.pixsize,
        epsilon,
        hermitian=hermitian,
    )


# The calibration grid builds 40 plans per geometry and three tests sweep it
# (one of them twice, forward and adjoint). GH200_large's plan holds ~135 MB of
# 2048^2 leaves, so the cache is deliberately tiny: it exists to collapse the
# immediately-repeated build of the *same* cell, not to hold the grid.
# ``hermitian`` is part of the key, not a default, for the reason
# ``_plan_for_uvw`` gives.
@lru_cache(maxsize=2)
def _cached_plan(
    telescope: Telescope, zenith_angle_deg: float, epsilon: float, hermitian: bool
) -> WGridderPlan:
    return _plan_for(telescope, zenith_angle_deg, epsilon, hermitian=hermitian)


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

    The interval is closed at both ends, matching the builder's
    ``side="left"`` / ``side="right"`` pair. Closed is the right choice
    because ``kernel.phi`` returns a nonzero ``exp(-beta)`` at an *exactly*
    computed ``|z| == 1``; it does not follow that the operators weight every
    row this counts, since they form ``z`` by a division that rounds -- see
    the module docstring for the measured size of that gap.
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
    """Host-side ``phi_numpy`` cross-check on the nominal-support count.

    Restates the denominator through ``kernel.phi_numpy`` instead of through
    interval arithmetic: it forms ``z = (w - w_k) / scale`` in numpy and lets
    the division round, where :func:`_independent_live_sizes` compares against
    the interval endpoints directly. Two ways of asking the same host-side
    question, so a systematic error in one shows up as a disagreement.

    **It is not a proxy for what the operators weight, and must not be read as
    one.** The device derives its own w (``wgridder._channel_ft_coords``, FMA
    contraction, single precision throughout on a float32 plan); this runs
    numpy on the plan's leaves. Measured over the calibration grid against the
    JIT-compiled operator expression, this helper differs from it on 10 of the
    40 float64 cells and 7 of the 10 float32 ones -- for instance MeerKAT off30
    at eps 1e-3, where the nominal count is 2401, this helper gives 2399 and
    the compiled expression gives 2400. On the float32 leg it reproduces
    :func:`_independent_live_sizes` exactly on every cell and so carries no
    information about the device at all.

    The residual host-side disagreement with the interval count -- at most
    three incidences over the grid, from rows sitting within a rounding of
    ``|z| == 1`` -- is what the tolerance at the one call site allows for.
    """
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)
    total = 0
    for c in range(plan.n_chan):
        w_rel = _w_rel_per_channel(plan, c)
        for k in range(plan.n_w):
            z = (w_rel - centres[k]) / plan.w_kernel_scale
            total += int(np.count_nonzero(phi_numpy(z, plan.beta)))
    return total


def _boundary_margin_for(plan: WGridderPlan) -> float:
    """The builder's ``window_boundary_margin`` for this plan, in wavelengths.

    Factored out of :func:`_independent_padded_sizes` so the margin-exclusion
    test below can widen a boundary by exactly the amount the builder does.
    ``window_boundary_margin`` is sized from the absolute w range, which
    ``make_plan`` forms in ``real_dtype`` from the narrowed uvw -- so form it
    the same way here, or the float32 leg gets a different margin.
    """
    w_m = np.asarray(plan.uvw_m)[:, 2]
    inv_lambda = np.asarray(plan.inv_lambda)
    ends_lo = inv_lambda * float(w_m.min())
    ends_hi = inv_lambda * float(w_m.max())
    return window_boundary_margin(
        plan.real_dtype,
        float(np.min(np.minimum(ends_lo, ends_hi))),
        float(np.max(np.maximum(ends_lo, ends_hi))),
        plan.w_extent,
    )


def _independent_margin_widened_sizes(plan: WGridderPlan) -> np.ndarray:
    """``(n_chan, n_w)`` counts inside the support widened by the margin only.

    The middle term of the three the builder could divide by: wider than
    :func:`_independent_live_sizes` (no margin, no clamp), narrower than
    :func:`_independent_padded_sizes` (margin *and* clamp). It exists so that
    :func:`test_margin_rows_are_excluded_from_the_denominator` can separate
    the two widenings -- against the padded reference alone, an implementation
    that reused the builder's margin-widened ``lo`` / ``hi`` for the live
    count would still look correct.

    Counted with a dense mask in input order, like the live reference and for
    the same reason.
    """
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)
    edge = plan.w_kernel_scale + _boundary_margin_for(plan)
    out = np.zeros((plan.n_chan, plan.n_w), dtype=np.int64)
    for c in range(plan.n_chan):
        w_rel = _w_rel_per_channel(plan, c)
        for k in range(plan.n_w):
            out[c, k] = np.count_nonzero(
                (w_rel >= centres[k] - edge) & (w_rel <= centres[k] + edge)
            )
    return out


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
    margin = _boundary_margin_for(plan)
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


# The pre-#43 rule, kept whole so the equivalence tests can *recompute* what
# v0.1.2 would have decided instead of pinning a table of strategy names. The
# cutoffs are the shipped v0.1.2 literals and must not be restated in terms of
# ``_CPU_PADDING_CUTOFF``: the whole point is to compare the new rule against
# the old one as it was, so if this drifts to track the constant the tests
# below stop comparing two rules and start comparing one rule with itself.
_PREVIOUS_CPU_PADDING_CUTOFF = 5.0
_PREVIOUS_GPU_PADDING_CUTOFF = 3.0


def _pre_43_padded_overhead(plan: WGridderPlan) -> float:
    """``max_window_size / mean(nonzero window_size)`` -- the v0.1.2 metric."""
    padded_sizes = _independent_padded_sizes(plan)
    nonzero = padded_sizes[padded_sizes > 0]
    return float(padded_sizes.max()) / float(nonzero.mean())


def _pre_43_auto_picks(plan: WGridderPlan, *, is_adjoint: bool) -> tuple[str, str]:
    """``(cpu, gpu)`` strategy the v0.1.2 heuristic would have picked.

    A transcription of ``_auto_w_strategy_cpu`` / ``_auto_w_strategy_gpu`` as
    they stood before issue #43, reading the padded metric against the old
    cutoffs. Everything except the padding branch is unchanged by #43 and is
    reproduced here only so the comparison is against the complete old rule.
    """
    previous_overhead = _pre_43_padded_overhead(plan)
    n_w, width = plan.n_w, plan.w_kernel_width
    if n_w <= width + 2:
        return "dense_scan", "dense_vmap"
    if previous_overhead > _PREVIOUS_CPU_PADDING_CUTOFF:
        return "dense_scan", "dense_vmap"
    expected_cpu = "windowed_scan" if is_adjoint and n_w / width > 2.0 else "dense_scan"
    if previous_overhead > _PREVIOUS_GPU_PADDING_CUTOFF or plan.n_rows < 10_000:
        expected_gpu = "dense_vmap"
    elif is_adjoint or n_w <= 3.0 * width:
        expected_gpu = "windowed_vmap"
    else:
        expected_gpu = "dense_vmap"
    return expected_cpu, expected_gpu


# --- the identity ------------------------------------------------------------
#
# ``(zenith_angle_deg, hermitian)`` cells for the identity test below. Three,
# not four: MWA_extended *zenith* at eps 1e-3 loses its strict-subset windows
# under issue #17's fold (n_w 11 -> 8 against W = 4, i.e. n_w_inner drops to
# exactly W, so every plane's window spans every row and ``max_window_size ==
# n_rows``). That is precisely the degenerate regime the test's own
# non-vacuity precondition exists to exclude, so the cell is not run -- and the
# premise for not running it is asserted, not assumed, by
# ``test_the_two_denominator_guards_pin_their_geometry_for_a_reason``.
_IDENTITY_CELLS = [
    pytest.param(0.0, False, id="za0-unfolded"),
    pytest.param(30.0, False, id="za30-unfolded"),
    pytest.param(30.0, True, id="za30-folded"),
]


@pytest.mark.parametrize(("zenith_angle_deg", "hermitian"), _IDENTITY_CELLS)
def test_padding_overhead_is_windowed_work_over_live_work(
    zenith_angle_deg: float, hermitian: bool
) -> None:
    """``window_padding_overhead == n_chan * n_w * max_window_size / live_row_count``.

    The denominator is taken from :func:`_independent_live_sizes`, not from
    ``plan.live_row_count``, so this is a check of the number and not a
    restatement of the formula. ``epsilon=1e-3`` so it runs on both precision
    legs.

    Swept over both plan geometries where both are non-degenerate (issue #17);
    see ``_IDENTITY_CELLS`` for the one cell that is not.
    """
    plan = _plan_for(MWA_EXTENDED, zenith_angle_deg, _F32_SAFE_EPSILON, hermitian=hermitian)
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

    # The same denominator restated through ``phi_numpy`` rather than through
    # interval endpoints -- a second host-side route to the same number, not a
    # statement about what the operators weight (see the helper's docstring).
    # The tolerance is for rows sitting within a rounding of ``|z| == 1``,
    # where forming ``z`` by division puts them on the other side of the
    # support edge; the measured worst case over the whole forty-cell grid
    # below, in both precision legs, is three rows. Against the ~490-row gap
    # between the live and padded denominators on this fixture that is not
    # slack the defect could hide in.
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


# The channel spread for the multi-channel test below. Wide enough that the
# per-channel w in wavelengths -- and so every channel's window layout --
# genuinely differs: w scales with ``freq``, so the top channel's w-range is
# twice the bottom's and the two disagree about which planes are occupied.
_MULTI_CHAN_FREQ_FACTORS = np.array([1.0, 1.3, 1.6, 2.0])


def test_accumulators_sum_over_every_channel() -> None:
    """Both scalars accumulate across channels, not just the last one.

    Every other reference in this module runs on a single-channel plan, and
    the multi-channel assertion in ``test_planning.py`` takes its expectation
    from ``plan.live_row_count`` itself, so it is circular on exactly this
    point. Changing the builder's ``live_row_count += ...`` to ``=`` -- an
    easy slip, since the loop body assigns rather than accumulates
    everywhere else -- drops all but the last channel and passes both.

    ``epsilon=1e-3`` so this runs on the float32 leg too.
    """
    uvw = synthetic_uvw(MWA_EXTENDED, 30.0, seed=0)
    freq = MWA_EXTENDED.freq_hz * _MULTI_CHAN_FREQ_FACTORS
    plan = _plan_for_uvw(
        uvw,
        freq,
        (MWA_EXTENDED.n_pix, MWA_EXTENDED.n_pix),
        MWA_EXTENDED.pixsize,
        _F32_SAFE_EPSILON,
        hermitian=True,
    )
    live_sizes = _independent_live_sizes(plan)

    # Preconditions, asserted rather than skipped. The channels must lay their
    # windows out differently, or a plan with four identical channels would be
    # a single-channel plan wearing a hat and would gate nothing about the
    # channel loop.
    assert plan.n_chan == len(_MULTI_CHAN_FREQ_FACTORS) > 1
    assert not all(np.array_equal(live_sizes[0], live_sizes[c]) for c in range(1, plan.n_chan)), (
        "the channels have identical window layouts; widen _MULTI_CHAN_FREQ_FACTORS"
    )
    per_channel_live = live_sizes.sum(axis=1)
    per_channel_empty = np.count_nonzero(live_sizes == 0, axis=1)

    assert plan.live_row_count == int(live_sizes.sum())
    assert plan.empty_plane_count == int(per_channel_empty.sum())

    # Say it the other way round as well, so a failure names the defect rather
    # than only the mismatch. Note that the per-channel live totals are all
    # close to ``n_rows * W`` whatever the channel does with its planes -- the
    # invariant the structural test above rests on -- and on some legs they are
    # equal outright (float32 here gives 2400, 2400, 2400, 2401). So for
    # ``live_row_count`` it is the factor of n_chan that separates accumulating
    # from assigning, not a spread between the channels; for
    # ``empty_plane_count`` the per-channel values differ outright.
    assert plan.live_row_count != int(per_channel_live[-1]), (
        "live_row_count equals the last channel's count alone -- the builder "
        "is assigning where it should accumulate"
    )
    assert plan.empty_plane_count != int(per_channel_empty[-1]), (
        "empty_plane_count equals the last channel's count alone -- same slip"
    )
    assert plan.window_padding_overhead == pytest.approx(
        _windowed_row_work(plan) / plan.live_row_count
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

    ``hermitian=False`` (issue #17), and for a reason about *signal* rather
    than about legacy: the padding is two clamp rows per (channel, plane), so
    the gap this test measures scales with ``n_w`` -- which the fold halves.
    On the folded plan the same fixture reads 2654 padded against 2400 live,
    a 10.6% gap, against 20% unfolded, and the preconditions below (and the
    1.15x separation of the two ratios) are sized for the larger one. The
    quantity being guarded -- which rows count as irreducible -- is a property
    of the builder and not of the geometry, so the guard belongs on whichever
    review fixture shows it most loudly. That the folded gap really is 10.6%
    is asserted by
    ``test_the_two_denominator_guards_pin_their_geometry_for_a_reason`` rather
    than left as a claim in this docstring.
    """
    telescope, zenith_angle_deg = MWA_EXTENDED, 30.0
    plan = _plan_for(telescope, zenith_angle_deg, _F32_SAFE_EPSILON, hermitian=False)

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


def test_the_two_denominator_guards_pin_their_geometry_for_a_reason() -> None:
    """Both premises the two guards above rest on, asserted rather than claimed.

    Those guards are the only tests in this module that do *not* run on the
    shipped folded geometry, and each has a stated reason. A stated reason that
    nothing checks is a comment that goes stale silently, so both are measured
    here. If either stops holding -- because a plane count moves, or because
    the fold changes -- this fails and the two guards can be moved onto the
    folded geometry, which is where they would rather be.

    1. ``test_padding_overhead_is_windowed_work_over_live_work`` has no
       ``(zenith, folded)`` cell because the fold takes MWA_extended zenith at
       eps 1e-3 down to ``n_w_inner == W``: every plane's window then spans
       every row, which is exactly the degeneracy its non-vacuity precondition
       excludes.
    2. ``test_padding_rows_are_excluded_from_the_denominator`` stays unfolded
       because the padded-to-live gap it needs is two clamp rows per
       ``(channel, plane)``, so it shrinks with ``n_w``: 20% unfolded, 10.6%
       folded, against a 1.15x precondition.

    ``epsilon=1e-3`` throughout, so this runs on both precision legs.
    """
    # (1) the folded zenith plan has no strict-subset windows.
    zenith_folded = _plan_for(MWA_EXTENDED, 0.0, _F32_SAFE_EPSILON, hermitian=True)
    assert zenith_folded.n_w - zenith_folded.w_kernel_width <= zenith_folded.w_kernel_width, (
        f"MWA_extended zenith folded now has n_w_inner="
        f"{zenith_folded.n_w - zenith_folded.w_kernel_width} against W="
        f"{zenith_folded.w_kernel_width}: the windows may be strict subsets again, so the "
        "identity test can take its (zenith, folded) cell back"
    )
    assert zenith_folded.max_window_size == zenith_folded.n_rows

    # (2) the folded off30 plan carries less than the 15% padding signal the
    # exclusion guard's preconditions are sized for.
    off30_folded = _plan_for(MWA_EXTENDED, 30.0, _F32_SAFE_EPSILON, hermitian=True)
    live = int(_independent_live_sizes(off30_folded).sum())
    padded = int(_independent_padded_sizes(off30_folded).sum())
    assert 0 < padded - live < 0.15 * live, (
        f"MWA_extended off30 folded now carries {padded} padded rows against {live} live "
        f"({padded / live:.3f}x): if that has grown past 1.15x the exclusion guard can move "
        "onto the shipped geometry"
    )
    # It must still be a real gap: the two clamp rows per (channel, plane) are
    # there on the folded plan too, just half as many of them.
    assert padded - live == pytest.approx(2 * off30_folded.n_w * off30_folded.n_chan, rel=0.05)


# --- the margin exclusion, on a fixture that carries a margin row ------------
#
# The regression test above runs on MWA_extended off30, where the margin
# catches nothing at either precision and the whole live-vs-padded gap is the
# +/-1 clamp. So it gates only half of the exclusion: an implementation that
# reused the builder's margin-widened ``lo`` / ``hi`` for the live count --
# dropping the clamp but keeping the margin -- passes every assertion in it.
# No repository fixture separates the two, because the margin is a few ulps
# wide and nothing lands in it by chance. This fixture puts a row there on
# purpose.

_MARGIN_FIXTURE_N_ROWS = 200
_MARGIN_FIXTURE_HALF_EXTENT = 400.0  # metres, so w spans [-400, 400]
_MARGIN_FIXTURE_PIXSIZE = 2e-3
_MARGIN_FIXTURE_FREQ = np.array([1.4e9])
# float64 on the x64 leg AND float32, since the margin scales with the plan
# dtype's unit roundoff and a construction that works at one precision proves
# nothing about the other. A float64 plan cannot be requested with x64 off
# (issue #11), so that leg covers float32 only.
_MARGIN_DTYPES: tuple[DTypeLike, ...] = (jnp.float64, jnp.float32) if X64 else (jnp.float32,)


def _margin_fixture_uvw() -> np.ndarray:
    """Baselines with w spread evenly over a fixed range; row order is w order."""
    rng = np.random.default_rng(7)
    uvw = np.zeros((_MARGIN_FIXTURE_N_ROWS, 3))
    uvw[:, :2] = rng.uniform(-100.0, 100.0, size=(_MARGIN_FIXTURE_N_ROWS, 2))
    uvw[:, 2] = np.linspace(
        -_MARGIN_FIXTURE_HALF_EXTENT, _MARGIN_FIXTURE_HALF_EXTENT, _MARGIN_FIXTURE_N_ROWS
    )
    return uvw


def _w_rel_of(plan: WGridderPlan, w_metres: float, chan: int = 0) -> float:
    """``w`` in the plan's relative coordinate, derived as the builder derives it.

    Narrow to ``real_dtype``, multiply *in* ``real_dtype``, widen, subtract
    ``w0`` -- the order matters at the ulp scale this test works at.
    """
    w_m = np.asarray([w_metres], dtype=plan.real_dtype)
    inv_lambda = np.asarray(plan.inv_lambda)
    return float((w_m * inv_lambda[chan]).astype(np.float64)[0] - plan.w0)


def _plan_with_a_row_in_the_margin_band(
    real_dtype: DTypeLike,
) -> tuple[WGridderPlan, int, float]:
    """Build a plan holding one row strictly between support and support+margin.

    Returns ``(plan, plane_index, distance_from_that_plane's_centre)``.

    The construction is two-pass because the target depends on the geometry it
    is placed into: build once to read ``w_kernel_scale``, ``w_centers_rel``,
    ``w0`` and the margin off the plan, aim at the middle of the band below an
    *interior* plane's lower edge, then walk the representable grid of the
    row's w-in-metres with ``nextafter`` until the value the builder would
    compute lands inside the band. The band is about four ulps of that grid
    wide -- ``window_boundary_margin`` is ``4u`` of the absolute w scale --
    so a handful of candidates is always enough, but which one lands is not
    predictable in closed form, hence the search.

    An interior plane is chosen so the planted row is not a w-extreme: the
    second plan's ``w0``, ``dw``, ``n_w`` and centres are then identical to
    the first's, which is asserted, and the target computed against the first
    plan is therefore still valid in the second.

    The *lower* edge specifically, because ``window_start`` is built from the
    lower boundary alone (``max(lo - 1, 0)``). A row planted below it moves a
    number the plan actually stores, which is what lets the test below assert
    against ``plan.window_start`` rather than against a reconstruction.
    """
    base_uvw = _margin_fixture_uvw()
    image_shape = (64, 64)
    build = partial(
        make_plan,
        freq=_MARGIN_FIXTURE_FREQ,
        image_shape=image_shape,
        pixsize_l=_MARGIN_FIXTURE_PIXSIZE,
        pixsize_m=_MARGIN_FIXTURE_PIXSIZE,
        epsilon=_F32_SAFE_EPSILON,
        dtype=real_dtype,
        # hermitian=False for the same reason as ``_plan_for_uvw`` above, and
        # one more specific to this fixture: it plants a row at a chosen
        # w-in-metres a few ulps below an interior plane edge, and issue #17's
        # fold would move that row to |w| and re-place every plane, so the
        # planted value would no longer land in the band it was searched for.
        hermitian=False,
    )
    probe = build(base_uvw)
    assert probe.n_w > 2, "the fixture must have interior planes to plant a row below"

    scale = probe.w_kernel_scale
    margin = _boundary_margin_for(probe)
    assert margin > 0.0, "a zero margin would make this test vacuous"
    centre = float(np.asarray(probe.w_centers_rel, dtype=np.float64)[probe.n_w // 2])
    inv_lambda = float(np.asarray(probe.inv_lambda)[0])

    # Aim at the middle of the band, then walk outward one representable
    # value at a time in each direction.
    ideal_metres = (centre - scale - margin / 2.0 + probe.w0) / inv_lambda
    planted: float | None = None
    for direction in (np.inf, -np.inf):
        candidate = np.float64(ideal_metres)
        for _ in range(64):
            offset = abs(_w_rel_of(probe, float(candidate)) - centre)
            if scale < offset <= scale + margin:
                planted = float(candidate)
                break
            candidate = np.nextafter(candidate, direction)
        if planted is not None:
            break
    assert planted is not None, (
        f"no representable w lands in the margin band for {np.dtype(real_dtype).name}: "
        f"scale={scale!r}, margin={margin!r}. The band is ~4 ulps of the w grid, so this "
        "means the margin or the w scale has changed shape, not that the test was unlucky"
    )

    uvw = base_uvw.copy()
    uvw[_MARGIN_FIXTURE_N_ROWS // 2, 2] = planted
    plan = build(uvw)
    assert (
        plan.n_w == probe.n_w
        and plan.w0 == probe.w0
        and plan.w_kernel_scale == probe.w_kernel_scale
        and np.array_equal(np.asarray(plan.w_centers_rel), np.asarray(probe.w_centers_rel))
    ), "planting the row moved the plan geometry; it was supposed to be interior"

    return plan, probe.n_w // 2, abs(_w_rel_of(plan, planted) - centre)


@pytest.mark.parametrize(
    "real_dtype", _MARGIN_DTYPES, ids=[np.dtype(d).name for d in _MARGIN_DTYPES]
)
def test_margin_rows_are_excluded_from_the_denominator(real_dtype: DTypeLike) -> None:
    """A row inside ``boundary_margin`` of an edge is padding, not live work.

    The other half of the issue #43 exclusion, and the half no repository
    fixture reaches. Reverting ``live_row_count`` to the builder's
    margin-widened ``lo`` / ``hi`` -- keeping the margin, dropping only the
    clamp -- passes ``test_padding_rows_are_excluded_from_the_denominator``
    and fails here.

    Both directions are asserted, and both against a real plan field:

    * ``plan.live_row_count`` must *exclude* the planted row, and
    * ``plan.window_start`` must *include* it -- the production window really
      is widened by the margin, not merely reported as if it were.

    The second matters because the first alone is satisfied by a builder with
    no margin at all: the two would then agree by both being nominal. It is
    also the only assertion in this module that reads ``window_start``, so it
    is what keeps the fixture honest about which quantity it is exercising.
    ``tests/test_boundary_planes.py::test_windowed_dense_parity_at_window_edge``
    independently fails if the margin is deleted from the builder, via the
    windowed-vs-dense value mismatch rather than the index; the two are
    complementary, and neither makes the other redundant. (Its neighbour
    ``test_window_boundary_margin_covers_host_device_gap`` does *not* cover
    that mutation -- it derives the margin from ``window_boundary_margin``
    directly and never looks at whether the builder applied it.)

    The preconditions are asserted rather than skipped: if the construction
    stops landing a row in the band, that must fail loudly instead of passing
    vacuously.
    """
    plan, plane, offset = _plan_with_a_row_in_the_margin_band(real_dtype)
    scale = plan.w_kernel_scale
    margin = _boundary_margin_for(plan)

    # Precondition: the planted row is strictly outside kernel support, and
    # inside the margin the builder widens by.
    assert scale < offset <= scale + margin, (
        f"planted row sits at {offset - scale:.3e} past the edge, outside the "
        f"margin band (0, {margin:.3e}]"
    )
    # Said through the kernel rather than through the interval: at this |z|,
    # computed on the host, ``phi`` returns exactly zero. That is a host-side
    # statement -- the device forms its own ``z`` and this is a row the margin
    # exists to catch, so it is precisely the case where the device may
    # disagree. Which is the point: the row is excluded from the denominator
    # because it is outside *nominal* support, not because the kernel can be
    # shown to ignore it.
    assert float(phi_numpy(np.array([offset / scale]), plan.beta)[0]) == 0.0

    live_sizes = _independent_live_sizes(plan)
    margin_sizes = _independent_margin_widened_sizes(plan)
    padded_sizes = _independent_padded_sizes(plan)

    # Non-vacuity: the margin genuinely moves this plane's boundary, by
    # exactly the one row that was planted there.
    assert margin_sizes[0, plane] == live_sizes[0, plane] + 1, (
        "the fixture no longer carries a margin row, so the assertions below "
        "cannot distinguish the live bounds from the margin-widened ones"
    )
    assert padded_sizes[0, plane] >= margin_sizes[0, plane]

    # ... and the window the operators actually slice really does reach the
    # planted row. ``window_start`` is built from the lower boundary alone
    # (``max(lo - 1, 0)``), which is why the row was planted below an edge:
    # the two candidate starts -- with the margin and without it -- differ by
    # exactly one row at this plane, and the plan stores the widened one.
    # Reconstructing the two *expectations* is unavoidable; the subject of the
    # assertion is ``plan.window_start``, a stored leaf, so a builder that
    # dropped ``boundary_margin`` fails here rather than agreeing with a
    # reference that dropped it too.
    sorted_w_rel = _w_rel_per_channel(plan, 0)[np.asarray(plan.sort_perm)]
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)
    start_with_margin = np.maximum(
        np.searchsorted(sorted_w_rel, centres - scale - margin, side="left") - 1, 0
    )
    start_without_margin = np.maximum(
        np.searchsorted(sorted_w_rel, centres - scale, side="left") - 1, 0
    )
    assert start_with_margin[plane] == start_without_margin[plane] - 1, (
        "the planted row does not move this plane's lower boundary, so the "
        "window_start assertions below cannot tell the margin from its absence"
    )
    assert start_with_margin[plane] > 0, (
        "the planted row is at the very start of the sorted array, where the "
        "-1 clamp flattens both candidates to 0 and hides the difference"
    )
    window_start = np.asarray(plan.window_start)[0]
    assert np.array_equal(window_start, start_with_margin), (
        "plan.window_start does not match the margin-widened bounds: the "
        "builder is not applying boundary_margin"
    )
    assert not np.array_equal(window_start, start_without_margin)

    # The gate on the denominator. Both totals are available to a wrong
    # implementation; only the live one is correct.
    assert plan.live_row_count == int(live_sizes.sum())
    assert plan.live_row_count < int(margin_sizes.sum()) < int(padded_sizes.sum())
    assert plan.window_padding_overhead == pytest.approx(
        _windowed_row_work(plan) / int(live_sizes.sum())
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


# ``empty_plane_count`` on MWA_extended off30, float64, by geometry and epsilon.
# Measured on this branch; see the test below for what the two columns say.
_EXPECTED_EMPTY_PLANES: dict[bool, dict[float, int]] = {
    False: {1e-3: 87, 1e-6: 62, 1e-9: 45, 1e-12: 32},
    True: {1e-3: 31, 1e-6: 18, 1e-9: 9, 1e-12: 3},
}


@requires_x64
@pytest.mark.parametrize("hermitian", [False, True], ids=["unfolded", "folded"])
@pytest.mark.parametrize("epsilon", [1e-3, 1e-6, 1e-9, 1e-12])
def test_empty_plane_count_matches_the_measured_table(epsilon: float, hermitian: bool) -> None:
    """The measured empty-plane table on the issue's worst fixture.

    ``requires_x64``: epsilon below ~1e-5 cannot be asked of a float32 plan
    without a warning, and ``filterwarnings = error`` makes that a failure.
    The float32 leg covers the same fixture at ``_F32_SAFE_EPSILON`` above,
    where it reads 90 unfolded and 30 folded rather than 87 and 31 -- rows on a
    support boundary land on the other side once ``inv_lambda`` is single
    precision, which is why this exact table is float64-only.

    Both geometries (issue #17). The folded column is what ships and the
    unfolded one is the pre-#17 measurement, kept because the *shape* of the
    two columns is the interesting part: folding roughly halves ``n_w`` on this
    fixture but cuts the empty planes by far more than half (87 -> 31, 32 -> 3),
    because the emptiness was concentrated in the tail of planes covering the
    sparse negative-w wing that the fold reflects onto the positive one. An
    implementation that halved ``n_w`` without actually merging the two wings
    would keep the unfolded proportion of empty planes and fail here.
    """
    expected = _EXPECTED_EMPTY_PLANES[hermitian][epsilon]
    plan = _plan_for(MWA_EXTENDED, 30.0, epsilon, hermitian=hermitian)
    assert plan.empty_plane_count == expected
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
    plan = _plan_for_uvw(uvw, freq, (64, 64), 2e-3, _F32_SAFE_EPSILON, hermitian=True)

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
# fixtures x two pointings x four epsilons. Swept in **both** plan geometries
# (issue #17), because the fold moves every cell and only one of the two is
# what ships.
#
# Corrected-metric measurements, float64. The float32 leg runs ten of these
# cells (eps 1e-3) and agrees with the float64 figures to within 0.19%
# unfolded (worst EDA2 zenith) and 0.13% folded (worst EDA2 off30, 2.6169
# against 2.6202); the high-overhead cell agrees to 0.13% unfolded and 0.04%
# folded, which is why the bands below are per-geometry but not per-precision:
#
#                        every cell except      MWA_extended off30
#                        MWA_extended off30
#   hermitian=False        1.077 - 3.173          5.173 - 5.784
#   hermitian=True         1.077 - 2.765          4.944 - 5.076
#
# The unfolded column is what moved ``_CPU_PADDING_CUTOFF`` in issue #43. The
# shipped 5.0 sat just above the padded-scale maximum of 4.933; on the
# corrected scale the same cells read up to 5.784 (5.791 on the float32 leg),
# so leaving the cutoff at 5.0 would have flipped the four MWA_extended off30
# adjoint cells from ``windowed_scan`` to ``dense_scan`` -- contradicting the
# measured CPU win in AGENTS.md section 9. 6.0 is the smallest value that both
# cleared that grid and preserved the old cutoff's proportional headroom:
# 5.0 times the worst measured padded->live inflation (5.784 / 4.809 = 1.203)
# is 6.014.
#
# The folded column is the shipped geometry and it does **not** move the
# cutoff. #43 changed the metric's *denominator*, which is why 5.0 had to be
# restated on the new scale; #17 changes neither the metric nor its scale, only
# where the fixtures sit under it -- and they all move down, so 6.0 goes from
# clearing the worst cell by 3.7% to clearing it by 18.2%. Re-fitting it onto
# the folded maximum would be a fresh calibration on one draw rather than a
# restatement, and the seed sweep at the bottom of this file is why that would
# be a bad one: folded, MWA_extended off30 at eps 1e-3 spans 5.076 to 6.466
# over seeds 0-11 in float64 (5.074 to 6.469 in float32) and crosses 6.0 on
# four of the twelve in both legs, so a cutoff fitted to
# the seed-0 draw (the gentlest of the twelve, as it is unfolded too) would
# bind on a third of them -- on exactly the fixture AGENTS.md section 9
# measures the windowed adjoint win on. See the derivation comment on
# ``_CPU_PADDING_CUTOFF`` in ``wgridder.py``.
#
# ``_GPU_PADDING_CUTOFF`` does not move either, and on the folded grid it stops
# being reachable at all: unfolded, three cells sit above 3.0 (MWA_compact
# off30 3.050 and MeerKAT off30 3.172 at eps 1e-3, plus the MWA_extended off30
# column); folded, the maximum outside MWA_extended off30 is 2.765, so only
# that one fixture is above the cutoff. Both cells that cross are on plans of
# 600 rows, which the GPU heuristic sends to ``dense_vmap`` on the row-count
# gate before the padding branch is consulted, so no GPU pick depends on it
# either way.

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
# tolerance on a converging quantity. Keyed by ``hermitian`` (issue #17): the
# fold is not a perturbation of these numbers, it moves the high band down by
# roughly 0.7 and pulls the ordinary band's ceiling from 3.17 to 2.77, so one
# pair of bands wide enough for both geometries would gate neither.
_ORDINARY_OVERHEAD_RANGE = {False: (1.0, 3.3), True: (1.0, 2.9)}
_HIGH_OVERHEAD_RANGE = {False: (5.0, 5.9), True: (4.8, 5.2)}


def test_cpu_padding_cutoff_is_six_and_still_gates() -> None:
    """The cutoff is 6.0 exactly, and it still switches the strategy.

    Stated as its own test so the number is auditable without running the
    forty-cell sweep. Two halves, and the second is the one that matters:

    * **6.0 exactly.** On the *unfolded* geometry the corrected metric reaches
      5.784 (float64) / 5.792 (float32) on MWA_extended off30, so anything at
      or below that flips the measured CPU win to ``dense_scan``. That is a
      *lower* bound, and a lower bound alone is satisfied by 100.0 -- which
      would clear the grid by switching the guard off. The upper bound is the
      derivation: the worst padded-to-live inflation on the grid is
      5.7843 / 4.8089 = 1.2028, so carrying the shipped 5.0 across the change
      of scale gives 6.014, and 6.0 is that rounded down to the nearest tenth.
      The construction yields 6.014, not 6.0; 6.0 is the round number just
      below it, which is what keeps the restatement from loosening the cutoff.

      Issue #17's fold does not re-open that derivation, and the reason is
      that #43 changed the metric's *denominator* while #17 changes only the
      geometry measured under it: there is no change of scale to restate
      across a second time. What it does change is the headroom -- the folded
      grid's maximum is 5.076, so 6.0 now clears the worst shipped cell by
      18.2% where it cleared the unfolded one by 3.7%. See the grid comment
      above for why that headroom is not re-fitted onto 5.076.
    * **It still gates.** A constant no branch can reach is not a cutoff, so
      exercise the branch on both sides of the value rather than trusting that
      the comparison is still wired up.
    """
    assert _CPU_PADDING_CUTOFF == 6.0
    # The GPU cutoff is unchanged by the redefinition; pin that it did not
    # get dragged along.
    assert _GPU_PADDING_CUTOFF == 3.0

    # A real plan that clears every earlier branch, so the padding comparison
    # is what decides: on the shipped folded geometry n_w = 131 against W = 4,
    # i.e. past both `W + 2` and the adjoint's `n_w / W > 2` (it was 248
    # unfolded, so the fold halves the margin over those gates without coming
    # close to closing it). Only the overhead is substituted, so this is a test
    # of the branch and not of the fixture.
    plan = _plan_for(MWA_EXTENDED, _HIGH_OVERHEAD_POINTING, _F32_SAFE_EPSILON)
    assert plan.n_w > plan.w_kernel_width + 2
    assert plan.n_w / plan.w_kernel_width > 2.0

    below = dataclasses.replace(plan, window_padding_overhead=_CPU_PADDING_CUTOFF - 0.001)
    at = dataclasses.replace(plan, window_padding_overhead=_CPU_PADDING_CUTOFF)
    above = dataclasses.replace(plan, window_padding_overhead=_CPU_PADDING_CUTOFF + 0.001)
    assert _auto_w_strategy_cpu(below, is_adjoint=True) == "windowed_scan"
    assert _auto_w_strategy_cpu(at, is_adjoint=True) == "windowed_scan"  # strict >
    assert _auto_w_strategy_cpu(above, is_adjoint=True) == "dense_scan"


@pytest.mark.parametrize("hermitian", [False, True], ids=["unfolded", "folded"])
@pytest.mark.parametrize(("telescope", "zenith_angle_deg", "epsilon"), _GRID)
def test_calibration_grid_padding_overhead(
    telescope: Telescope, zenith_angle_deg: float, epsilon: float, hermitian: bool
) -> None:
    """Each grid cell's corrected overhead sits in its measured band.

    Two things the bands pin, and both directions matter:

    * on the padded denominator the MWA_extended off30 cells read 4.71 - 4.93
      and would miss ``_HIGH_OVERHEAD_RANGE[False]`` entirely -- that is issue
      #43's redefinition;
    * the ``True`` leg is the shipped geometry (issue #17). Its bands are
      *narrower* and lower, so they are a real re-measurement rather than the
      unfolded ones with slack added: the folded high band is 4.944 - 5.076
      against an unfolded 5.173 - 5.784, and a plan that silently stopped
      folding would read outside it.

    Both legs are swept because the grid is cheap (host-side plan building
    only) and because the cutoff derivation quotes numbers from each.
    """
    plan = _cached_plan(telescope, zenith_angle_deg, epsilon, hermitian)
    overhead = plan.window_padding_overhead

    is_high = (
        telescope.name == _HIGH_OVERHEAD_FIXTURE and zenith_angle_deg == _HIGH_OVERHEAD_POINTING
    )
    band = _HIGH_OVERHEAD_RANGE if is_high else _ORDINARY_OVERHEAD_RANGE
    lo, hi = band[hermitian]
    assert lo <= overhead <= hi, (
        f"{overhead:.4f} outside the measured band ({lo}, {hi}) for hermitian={hermitian}"
    )

    # The guard is a guard: no cell of *this* grid reaches it at any epsilon.
    # Scoped to the grid deliberately -- other draws of these same fixtures do
    # cross it, which ``test_auto_picks_survive_the_redefinition_across_seeds``
    # relies on and issue #34 is about.
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

    **``hermitian=False``, and this is the one place in the module where that
    is the right geometry rather than a legacy one.** What is compared here is
    two *rules*: the one v0.1.2 shipped and the one this repository ships. The
    v0.1.2 rule was written, and its 5.0 measured, on the unfolded plan
    geometry -- the only geometry that existed then -- and #43 restated it onto
    the corrected denominator on that same geometry. Running the comparison on
    plans neither rule was calibrated against would not be a stronger test of
    the redefinition; it would be a test of a rule pair on data neither was
    fitted to, which is a different (and unasserted) claim.

    That the two rules do come apart off that geometry is worth recording
    rather than leaving implicit. Measured on folded plans over seeds 0-11 of
    MWA_extended off30 at eps 1e-3, they disagree on the adjoint pick for five
    of the twelve: the padded-to-live inflation the 5.0 -> 6.0 restatement was
    built on is 1.20 unfolded but only 1.11 folded, so the proportional
    restatement stops being equivalence-preserving. Every one of those five
    disagreements is the shipped rule keeping ``windowed_scan`` where the
    v0.1.2 rule would have taken it to ``dense_scan`` -- the direction AGENTS.md
    section 9 measures as the CPU win on that exact fixture. The forty-cell
    folded grid shows no disagreement at all (checked; seed 0 is below both
    cutoffs on every cell).
    """
    plan = _cached_plan(telescope, zenith_angle_deg, epsilon, False)
    expected_cpu, expected_gpu = _pre_43_auto_picks(plan, is_adjoint=is_adjoint)

    assert _auto_w_strategy_cpu(plan, is_adjoint=is_adjoint) == expected_cpu
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == expected_gpu


# The seed sweep below. The forty-cell grid above is a single draw of each
# fixture (``synthetic_uvw(..., seed=0)``), and on MWA_extended off30 the draw
# moves the metric a long way: measured over seeds 0-11 at eps 1e-3 the
# corrected overhead spans 5.784 to 8.090 against a seed-0 value of 5.784, so
# ten of the twelve sit *above* ``_CPU_PADDING_CUTOFF`` where seed 0 sits below
# it. That the pinned grid happens to draw the gentlest seed is issue #34's
# subject, not this module's; what belongs here is that the equivalence this
# change promises survives the spread rather than being an artefact of one
# draw.
_EQUIVALENCE_SEEDS = tuple(range(12))


def test_auto_picks_survive_the_redefinition_across_seeds() -> None:
    """The two rules agree on every seed, on both sides of both cutoffs.

    The central claim of issue #43 is that redefining the metric and restating
    the cutoff moves no ``auto`` decision. On the pinned grid that claim is
    tested against one draw per fixture, and on the fixture that matters most
    the seed-0 draw is the one that stays below the cutoff -- so the grid
    exercises the "below" branch and never the "above" one. This sweeps the
    draw and asserts only the agreement, never which strategy a seed gets:
    which side of a cutoff a random draw lands on is a property of the draw.

    Cheap enough for the fast leg (twelve 600-row plans) and precision-
    independent: the two rules agree on all twelve seeds in float64 and in
    float32, so no ``requires_x64``.

    ``hermitian=False`` for the reason given at
    ``test_calibration_grid_auto_picks_survive_the_redefinition`` -- both rules
    being compared were calibrated on the unfolded geometry -- and that comment
    also records what the same sweep measures on folded plans (the two rules
    part company on five of the twelve, always in the direction of keeping the
    windowed adjoint).
    """
    fired_new = fired_old = 0
    for seed in _EQUIVALENCE_SEEDS:
        uvw = synthetic_uvw(MWA_EXTENDED, _HIGH_OVERHEAD_POINTING, seed=seed)
        plan = _plan_for_uvw(
            uvw,
            np.array([MWA_EXTENDED.freq_hz]),
            (MWA_EXTENDED.n_pix, MWA_EXTENDED.n_pix),
            MWA_EXTENDED.pixsize,
            _F32_SAFE_EPSILON,
            hermitian=False,
        )
        fired_new += plan.window_padding_overhead > _CPU_PADDING_CUTOFF
        fired_old += _pre_43_padded_overhead(plan) > _PREVIOUS_CPU_PADDING_CUTOFF
        for is_adjoint in (False, True):
            expected_cpu, expected_gpu = _pre_43_auto_picks(plan, is_adjoint=is_adjoint)
            got_cpu = _auto_w_strategy_cpu(plan, is_adjoint=is_adjoint)
            got_gpu = _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint)
            assert got_cpu == expected_cpu, (
                f"seed {seed}, {'adjoint' if is_adjoint else 'forward'}: the "
                f"corrected metric ({plan.window_padding_overhead:.4f} vs "
                f"cutoff {_CPU_PADDING_CUTOFF}) picks {got_cpu!r} where the "
                f"pre-#43 rule ({_pre_43_padded_overhead(plan):.4f} vs 5.0) "
                f"picks {expected_cpu!r}"
            )
            assert got_gpu == expected_gpu, f"seed {seed}: GPU pick moved"

    # Non-vacuity: the sweep has to straddle the cutoff, or "the two rules
    # agree" would only be a statement about the branch neither of them takes.
    # Both rules fire on ten of the twelve seeds and clear on two, in both
    # precision legs -- the counts are asserted loosely, since which seeds land
    # where is not the property under test.
    assert 0 < fired_new < len(_EQUIVALENCE_SEEDS), (
        f"the corrected metric crossed its cutoff on {fired_new} of "
        f"{len(_EQUIVALENCE_SEEDS)} seeds; the sweep must cover both branches"
    )
    assert 0 < fired_old < len(_EQUIVALENCE_SEEDS), (
        f"the pre-#43 metric crossed its cutoff on {fired_old} of "
        f"{len(_EQUIVALENCE_SEEDS)} seeds; the sweep must cover both branches"
    )
