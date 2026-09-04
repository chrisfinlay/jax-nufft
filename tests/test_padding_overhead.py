"""What ``plan.window_padding_overhead`` guarantees, and the grid behind the cutoffs.

``window_padding_overhead`` is defined in ``planning.py`` as

    max_window_size / window_size.mean()

with the mean over **every** ``(channel, plane)`` pair, empty planes
included. That is not a free choice of averaging convention: it is the
one number the windowed strategies actually pay. Each ``(channel, plane)``
step slices a *static* ``max_window_size`` rows out of the w-sorted array
(the shape has to be static for ``lax.scan`` / ``vmap``), so a windowed
traversal touches ``n_chan * n_w * max_window_size`` rows, while only
``window_size.sum()`` of them carry a nonzero kernel weight. The ratio of
those two is exactly the quantity above -- see
:func:`test_padding_overhead_is_the_windowed_work_ratio`, which asserts
the identity rather than restating the formula.

Two things follow, and they are what the rest of this module pins:

* it is bounded below by 1.0, with equality iff every plane's window has
  the same length; and
* on a peaked w-distribution it tends to rise as the plane grid refines --
  new planes between the peaks are empty while the widest window stays near
  a clump size. Small dips remain possible when window boundaries move.

The second point is why the definition changed in v0.1.3. Averaging over
nonzero windows only -- the pre-v0.1.3 convention -- systematically
understated the waste: a well-resolved clump produces all-or-nothing
windows whose nonzero mean sits right at the peak, so the ratio stays near
1.0 exactly where the padding is worst, and reads *lower* for a clumped
plan than for a uniform one. ``tests/test_planning.py``'s
``test_window_builder_clumped_distribution`` walks that resolution sweep.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from jax_nufft.planning import WGridderPlan, make_plan
from jax_nufft.wgridder import (
    _CPU_PADDING_CUTOFF,
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

# The plan dtype has to be requested explicitly (issue #11): a default
# float64 plan raises under ``JAX_ENABLE_X64=0``, which is a leg this module
# runs on.
_REAL_DTYPE: DTypeLike = jnp.float64 if X64 else jnp.float32


def _plan_for_uvw(
    uvw: np.ndarray,
    freq_hz: float,
    image_shape: tuple[int, int],
    pixsize: float,
    epsilon: float,
) -> WGridderPlan:
    return make_plan(
        uvw,
        np.array([freq_hz]),
        image_shape,
        pixsize,
        pixsize,
        epsilon=epsilon,
        dtype=_REAL_DTYPE,
    )


def _plan_for(
    telescope: Telescope, zenith_angle_deg: float, epsilon: float
) -> WGridderPlan:
    return _plan_for_uvw(
        synthetic_uvw(telescope, zenith_angle_deg, seed=0),
        telescope.freq_hz,
        (telescope.n_pix, telescope.n_pix),
        telescope.pixsize,
        epsilon,
    )


# --- the definition itself ---------------------------------------------------


@pytest.mark.parametrize("zenith_angle_deg", [0.0, 30.0])
def test_padding_overhead_is_the_windowed_work_ratio(zenith_angle_deg: float) -> None:
    """``window_padding_overhead`` == windowed row-work / irreducible row-work.

    ``epsilon=1e-3`` so this runs on both precision legs (a float32 plan
    warns below ~1e-5, and ``filterwarnings = error`` turns that into a
    failure).
    """
    plan = _plan_for(MWA_EXTENDED, zenith_angle_deg, 1e-3)
    window_size = np.asarray(plan.window_size)

    # The off-zenith case is specifically a regression fixture for including
    # empty windows in the mean. Keep that precondition explicit so fixture
    # drift cannot make the new and old definitions coincide and leave the
    # test vacuous. (At zenith, float64 has one empty edge window but float32
    # rounding makes it nonempty, so that is not a precision-stable guard.)
    if zenith_angle_deg == 30.0:
        assert np.any(window_size == 0)

    windowed_row_work = plan.n_chan * plan.n_w * plan.max_window_size
    irreducible_row_work = int(window_size.sum())
    assert plan.window_padding_overhead == pytest.approx(
        windowed_row_work / irreducible_row_work
    )
    # Equivalently: the max over the mean taken across ALL planes.
    assert plan.window_padding_overhead == pytest.approx(
        window_size.max() / window_size.mean()
    )
    # A max is never below a mean, so the ratio is a real overhead factor.
    assert plan.window_padding_overhead >= 1.0


def test_padding_overhead_is_one_when_every_window_is_equal() -> None:
    """The lower bound is attained, not merely approached.

    Constant ``w`` collapses the plan to a single plane holding every row
    (``planning.py``'s fast path), which is the degenerate case of "all
    windows the same length": windowed traversal wastes nothing.
    """
    rng = np.random.default_rng(0)
    uvw = np.zeros((128, 3))
    uvw[:, :2] = rng.uniform(-100.0, 100.0, size=(128, 2))
    plan = _plan_for_uvw(uvw, 1.4e9, (64, 64), 2e-3, 1e-3)

    assert plan.n_w == 1
    assert plan.max_window_size == plan.n_rows
    assert plan.window_padding_overhead == 1.0


# --- the calibration grid behind the cutoffs ---------------------------------
#
# ``_CPU_PADDING_CUTOFF`` moved from 5.0 to 8.0 when the mean stopped
# excluding empty windows. These two tests are what make that number
# auditable: the grid they sweep is the one quoted in its comment, so a
# planning change that pushes a real fixture towards the guard fails here
# rather than silently flipping an ``auto`` choice.
#
# ``requires_x64``: the sweep spans epsilon down to 1e-12, which a float32
# plan cannot be asked for without a warning (and ``filterwarnings = error``
# makes that a failure). The geometry it measures is precision-independent
# anyway -- both legs agree to <0.2% on every cell.

_FIXTURES = [EDA2, MWA_COMPACT, MWA_EXTENDED, MEERKAT, GH200_LARGE]
_POINTINGS = [0.0, 30.0]
_EPSILONS = [1e-3, 1e-6, 1e-9, 1e-12]
_GRID = [(t, za, eps) for eps in _EPSILONS for t in _FIXTURES for za in _POINTINGS]
_GRID_IDS = [f"{t.name}-za{int(za)}-eps{eps:.0e}" for t, za, eps in _GRID]

# The one fixture whose padding overhead lands above the old 5.0 cutoff on
# the new scale -- and the one where the windowed adjoint is the measured
# CPU win (AGENTS.md section 9), so it must stay on ``windowed_scan``.
_HIGH_OVERHEAD_FIXTURE = MWA_EXTENDED.name
_HIGH_OVERHEAD_POINTING = 30.0
# Every other cell sits here; the two bounds are the measured min/max with
# a little rounding room, not a tolerance on a converging quantity.
_ORDINARY_OVERHEAD_RANGE = (1.0, 3.2)
_HIGH_OVERHEAD_RANGE = (5.0, 5.6)


@requires_x64
@pytest.mark.parametrize("telescope,zenith_angle_deg,epsilon", _GRID, ids=_GRID_IDS)
def test_calibration_grid_padding_overhead_stays_below_cutoff(
    telescope: Telescope, zenith_angle_deg: float, epsilon: float
) -> None:
    plan = _plan_for(telescope, zenith_angle_deg, epsilon)
    overhead = plan.window_padding_overhead

    is_high = (
        telescope.name == _HIGH_OVERHEAD_FIXTURE
        and zenith_angle_deg == _HIGH_OVERHEAD_POINTING
    )
    lo, hi = _HIGH_OVERHEAD_RANGE if is_high else _ORDINARY_OVERHEAD_RANGE
    assert lo <= overhead <= hi

    # The guard is a guard: no review fixture comes near it at any epsilon.
    assert overhead < _CPU_PADDING_CUTOFF


@requires_x64
@pytest.mark.parametrize("telescope,zenith_angle_deg,epsilon", _GRID, ids=_GRID_IDS)
@pytest.mark.parametrize("is_adjoint", [False, True], ids=["forward", "adjoint"])
def test_calibration_grid_auto_picks_survive_the_redefinition(
    telescope: Telescope,
    zenith_angle_deg: float,
    epsilon: float,
    is_adjoint: bool,
) -> None:
    """Redefining the diagnostic must not move a single ``auto`` decision.

    The expectation is recomputed here from the pre-v0.1.3 rule (nonzero-mean
    overhead against the old 5.0 / 3.0 cutoffs) rather than written down as a
    table of strategy names, so what is asserted is *equivalence*: the point
    is not which strategy each cell gets, it is that redefining the
    diagnostic and restating the cutoff moved none of them.
    """
    plan = _plan_for(telescope, zenith_angle_deg, epsilon)
    window_size = np.asarray(plan.window_size)
    nonzero = window_size[window_size > 0]
    previous_overhead = float(window_size.max()) / float(nonzero.mean())

    n_w, width = plan.n_w, plan.w_kernel_width
    if n_w <= width + 2:
        expected_cpu, expected_gpu = "dense_scan", "dense_vmap"
    elif previous_overhead > 5.0:
        expected_cpu, expected_gpu = "dense_scan", "dense_vmap"
    else:
        expected_cpu = (
            "windowed_scan" if is_adjoint and n_w / width > 2.0 else "dense_scan"
        )
        if previous_overhead > 3.0 or plan.n_rows < 10_000:
            expected_gpu = "dense_vmap"
        elif is_adjoint or n_w <= 3.0 * width:
            expected_gpu = "windowed_vmap"
        else:
            expected_gpu = "dense_vmap"

    assert _auto_w_strategy_cpu(plan, is_adjoint=is_adjoint) == expected_cpu
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == expected_gpu
