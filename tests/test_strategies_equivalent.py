"""Strategy-equivalence regression net (issue #10 / T1).

AGENTS.md section 3: the four ``w_strategy`` values and the two
``channel_strategy`` values differ *only* in how the w-plane / channel loops
are structured (scan vs vmap, dense vs windowed-contiguous-slice) -- they are
mathematically identical operators that merely sum in a different order.
That makes cross-strategy agreement a property we can check independently of
any external oracle (ducc0, the exact DFT): every one of the eight
``(w_strategy, channel_strategy)`` combinations must land on (almost) the
same forward visibilities and the same adjoint image for the same plan.

This is the regression net every later engine change (chunked vmap,
windowed-bucketing rewrites, a new strategy) leans on: those changes are
free to alter reduction order, but never the answer. A bug that silently
drops rows, misaligns per-channel arrays, or corrupts one strategy's inner
loop shows up here even when every strategy still passes its own DFT/ducc0
parity bound (those bounds are wide enough -- see ``test_against_dft.py`` /
``test_against_ducc.py`` -- to hide a small per-strategy regression).

Precision
---------
Written precision-aware from the start: the plan is built with
``dtype=real_dtype`` (the session fixture from ``tests/conftest.py``), so
this module runs unmodified whether or not ``jax_enable_x64`` is on, and it
is deliberately *not* added to ``conftest.collect_ignore`` -- it is meant to
be the first module that grows the float32 CI leg rather than shrink the
backlog list. Under float32, ``eps`` values below
``planning.FLOAT32_EPSILON_FLOOR`` (1e-5) make ``make_plan`` emit a
``UserWarning`` about the unreachable accuracy floor; with the suite's
``filterwarnings = ["error"]`` that becomes an exception, so those plans are
built inside ``pytest.warns`` rather than skipped -- the *strategies still
have to agree with each other* even though none of them can hit 1e-6 or
1e-8 accuracy in float32.

Bound
-----
Issue #10 measured the worst-case pairwise disagreement between any two of
the eight strategy combinations, across the three ``short_telescope_pointing``
fixtures x the three ``eps`` values below, forward and adjoint (2026-09
review machine, macOS arm64, 3 channels):

  float64: forward 0.0 (bit-equal -- the same scatter-add / independent
           per-channel loop in every strategy, matching the "forward is
           bit-equal windowed-vs-dense" note in test_boundary_planes.py);
           adjoint 2.0e-13 (MWA_compact off30, eps=1e-6). Both comfortably
           inside the 1e-11 bound shared with test_boundary_planes.py and
           test_adjoint.py.
  float32: forward 0.0; adjoint 1.21e-7 (MWA_compact off30, eps=1e-8).
           float32 has essentially no headroom above its own rounding
           floor, so it gets its own bound rather than reusing the float64
           constant: 10x the measured worst case, per issue #10's rule for
           bounds that don't hold at 1e-11 on this machine.
"""

from __future__ import annotations

import contextlib
import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._types import ChannelStrategy, WStrategy
from jax_nufft.planning import FLOAT32_EPSILON_FLOOR
from tests.conftest import Telescope, synthetic_uvw, tol

W_STRATEGIES: tuple[WStrategy, ...] = (
    "dense_scan",
    "dense_vmap",
    "windowed_scan",
    "windowed_vmap",
)
CHANNEL_STRATEGIES: tuple[ChannelStrategy, ...] = ("scan", "vmap")
_STRATEGY_COMBOS: tuple[tuple[WStrategy, ChannelStrategy], ...] = tuple(
    itertools.product(W_STRATEGIES, CHANNEL_STRATEGIES)
)

EPS_VALUES = (1e-4, 1e-6, 1e-8)
N_CHAN = 3

# See the "Bound" section of the module docstring for the measurements
# behind these two constants.
STRATEGY_TOL = tol(1e-11, 1.3e-6)


def _per_channel_freq(tel: Telescope) -> np.ndarray:
    """Three distinct channel frequencies spanning +/-5% of the telescope's.

    Distinct (not repeated) frequencies give each channel its own
    ``uvw_lambda`` and hence its own w-window placement, so the windowed
    strategies' per-channel window-start table (``plan.window_start``,
    ``(n_chan, n_w)``) is exercised across channels rather than trivially
    identical for all three.
    """
    return tel.freq_hz * np.array([0.95, 1.0, 1.05])


def _build_problem(
    tel: Telescope,
    zen_deg: float,
    eps: float,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
):
    """Build a plan plus a real per-channel image and complex per-row vis.

    ``dtype=real_dtype`` is the load-bearing precision-aware bit: it makes
    the plan (and hence every downstream array) float32 under
    ``JAX_ENABLE_X64=0`` instead of raising ``make_plan``'s x64-off guard.
    """
    uvw = synthetic_uvw(tel, zen_deg, seed=0)
    freq = _per_channel_freq(tel)
    pix = tel.pixsize

    # Below the float32 accuracy floor, make_plan warns (issue #11); under
    # filterwarnings=["error"] that becomes an exception unless caught. The
    # warning is expected and not itself under test here -- what's under
    # test is that the strategies still agree even though none of them can
    # reach eps in this regime.
    warns_ctx = (
        pytest.warns(UserWarning, match="below the accuracy")
        if (real_dtype == jnp.float32 and eps < FLOAT32_EPSILON_FLOOR)
        else contextlib.nullcontext()
    )
    with warns_ctx:
        plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps, dtype=real_dtype)

    rng = np.random.default_rng(7)
    # Per-channel images (not a broadcast 2D image): each of the N_CHAN
    # channels gets its own independent random image, matching issue #10's
    # "3 channels, per-channel images" instruction so the channel axis is
    # genuinely exercised rather than trivially identical across channels.
    image_np = rng.standard_normal((N_CHAN, tel.n_pix, tel.n_pix))
    vis_np = (
        rng.standard_normal((tel.n_rows, N_CHAN)) + 1j * rng.standard_normal((tel.n_rows, N_CHAN))
    ).astype(np.complex128)

    image = jnp.asarray(image_np, dtype=real_dtype)
    vis = jnp.asarray(vis_np, dtype=complex_dtype)
    return plan, image, vis


def _pairwise_max_rel_err(
    results: dict[tuple[WStrategy, ChannelStrategy], np.ndarray],
) -> tuple[
    float, tuple[tuple[WStrategy, ChannelStrategy], tuple[WStrategy, ChannelStrategy]] | None
]:
    """Largest relative L2 disagreement between any two strategy results, and which pair."""
    worst = 0.0
    worst_pair = None
    for key_a, key_b in itertools.combinations(results.keys(), 2):
        a, b = results[key_a], results[key_b]
        err = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        if err > worst:
            worst = err
            worst_pair = (key_a, key_b)
    return worst, worst_pair


@pytest.mark.parametrize("eps", EPS_VALUES)
def test_strategies_agree_pairwise(
    short_telescope_pointing: tuple[Telescope, float],
    eps: float,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """All eight (w_strategy, channel_strategy) combinations agree with each
    other, forward and adjoint, to within STRATEGY_TOL.

    One plan/image/vis triple is built per (fixture, eps); every strategy
    combination is evaluated against *that same input*, so any discrepancy
    is attributable to the strategy implementation, not to random input
    variation.
    """
    tel, zen_deg = short_telescope_pointing
    plan, image, vis = _build_problem(tel, zen_deg, eps, real_dtype, complex_dtype)

    forward: dict[tuple[WStrategy, ChannelStrategy], np.ndarray] = {}
    adjoint: dict[tuple[WStrategy, ChannelStrategy], np.ndarray] = {}
    for w_strategy, channel_strategy in _STRATEGY_COMBOS:
        forward[(w_strategy, channel_strategy)] = np.asarray(
            dirty2vis(plan, image, w_strategy=w_strategy, channel_strategy=channel_strategy)
        )
        adjoint[(w_strategy, channel_strategy)] = np.asarray(
            vis2dirty(plan, vis, w_strategy=w_strategy, channel_strategy=channel_strategy)
        )

    fwd_err, fwd_pair = _pairwise_max_rel_err(forward)
    assert fwd_err < STRATEGY_TOL, (
        f"{tel.name} zen={zen_deg:g} eps={eps:g}: forward strategies disagree, "
        f"worst pair {fwd_pair} rel err {fwd_err:.3e} >= {STRATEGY_TOL:.3e}"
    )

    adj_err, adj_pair = _pairwise_max_rel_err(adjoint)
    assert adj_err < STRATEGY_TOL, (
        f"{tel.name} zen={zen_deg:g} eps={eps:g}: adjoint strategies disagree, "
        f"worst pair {adj_pair} rel err {adj_err:.3e} >= {STRATEGY_TOL:.3e}"
    )
