"""Clumped w-distributions and multi-channel constant-w (issue #15).

Four fixtures live here, and they exist because of an axis every other fixture
in this repository holds constant.

**The clumped track.** ``tests/conftest.py``'s ``synthetic_uvw`` produces a
symmetric, unimodal w-distribution at every pointing -- a Gaussian ``z`` offset
at zenith, a rotated Gaussian ``u`` off it. So the *shape* of the
w-distribution is constant across the whole review fixture set: unimodal, with
no dense core to concentrate the planes and no sparse tail to stretch them.
``tests/test_boundary_planes.py`` does build clumped distributions, but it only
compares the windowed path against the dense one on the same plan, on a toy
geometry (32^2, uniform ``(u, v)``, one channel). So no *clumped* fixture had
ever been held against an oracle outside the operator itself.
``conftest.clumped_track`` closes exactly that gap -- and only that gap.

What this module does **not** claim
-----------------------------------
The stronger-sounding version of the argument -- that a same-strategy
comparison cannot see a shared-mode error, so only an independent
implementation can -- does not survive measurement, and is not why these tests
are here. Two measurements taken on this machine (macOS arm64, jax 0.9.2,
ducc0 0.41.0), each a single mutation under ``src/`` run against the
*pre-existing* suite, ``origin/main`` at 1311c97, ``pytest -q --runslow``
(1520 passed unmutated):

* ``planning.py``'s ``w_kernel_scale = dw * W / 2`` widened by 2% -- a
  shared-mode w-kernel error the dense and windowed paths make equally:
  **223 tests fail**, 41 of them in ``tests/test_against_ducc.py`` at the same
  ``3 * eps`` external-oracle contract, on the Gaussian fixtures. All 13
  ``tests/test_boundary_planes.py`` tests pass. So the windowed-vs-dense
  *comparison* is blind to it; the suite around it plainly is not.
* ``wgridder.py``'s ``_nufft_epsilon`` giving the (u, v) NUFFT the whole
  budget instead of a tenth of it (``eps / 10`` -> ``eps``), a shared-mode
  accuracy loss both operators and all four strategies make identically:
  **48 tests fail, every one of them an exact-DFT comparison** (16
  ``test_against_dft``, 21 ``test_hermitian``, 8 ``test_adjoint``, 2
  ``test_nshift``, 1 ``test_divide_by_n``). Not one ducc0 parity test fails,
  on any fixture, at any strategy. On this branch it is 52, the four extra
  being the eps=1e-6 adjoint cells of
  :func:`test_small_clumped_geometry_matches_the_exact_dft` below.

The second measurement is the one that shaped this module. For shared-mode
*accuracy* the ordering runs opposite to the folklore: the exact-DFT contract
at ``2 * eps`` is the sharp instrument and a ducc0 comparison at ``3 * eps``
has too much headroom to notice, which is what AGENTS.md section 6 already
says ("DFT parity catches shared-mode bugs"). So the clumped geometry gets an
exact-DFT cell of its own
(:func:`test_small_clumped_geometry_matches_the_exact_dft`) and not only ducc0
cells. The two are complementary and neither is redundant: the DFT reference is
``O(n_rows * n_pix^2)`` and affordable only on a 24^2 / 48-row geometry, while
the ducc0 cells run the clumped distribution at full telescope scale, where
``n_w`` reaches 81 and 214 and the plane bookkeeping is what is under test.

**Multi-channel constant w.** ``plan.is_constant_w`` is ``w_extent == 0`` where
``w_extent`` is measured in *wavelengths, over all channels*. Data with a single
constant ``w`` in metres therefore takes the ``n_w == 1`` fast path at one
channel and the full generic path at three, because ``w * freq[c] / c`` differs
per channel. ``tests/test_constant_w.py`` covers only the single-channel case
(and reaches the generic path solely through the ``_force_generic`` test
kwarg), so the configuration a real multi-frequency observation of coplanar
data actually produces -- generic path, w-distribution collapsed onto ``n_chan``
discrete spikes -- was untested. That fixture cannot say anything about window
*bounds*, for a structural reason recorded at
:func:`test_multi_channel_constant_w_does_not_take_the_fast_path`.

**Multi-channel spread w.** This module's window-bound instrument, and the only
one in the repository that varies the *per-channel* window derivation: three
channels spanning a factor of two in frequency, ``n_w = 68`` with
``max_window_size`` 40 of 96 rows, so each channel's window is a strict subset
of the rows and the three subsets differ. Deriving every channel's bounds from
``inv_lambda[0]`` -- a defect that reaches only two plan-bookkeeping tests
elsewhere in the suite -- fails its eight windowed cells. The pre-existing
multi-channel windowed fixtures span 1.25 and 1.105 in fractional bandwidth,
where that defect is near-inert.

**The small clumped geometry.** A 24^2 / 48-row clumped track, small enough to
afford an exact ``O(n_rows * n_pix^2)`` DFT reference at ``2 * eps``.

Oracle
------
ducc0's public Python API, as in ``tests/test_against_ducc.py``, at that
module's ``3 * eps`` contract, plus a local exact-DFT reference at
``2 * eps``. The exact-DFT references in ``tests/test_against_dft.py`` and
``tests/test_adjoint.py`` are deliberately *not* imported, and are written out
again here instead: those modules call
``jax.config.update("jax_enable_x64", True)`` at import time, which would turn
the ``JAX_ENABLE_X64=0`` leg back into a float64 one for every module collected
after this. This module is precision-aware from the start (AGENTS.md section 6)
and must never be added to ``conftest.collect_ignore``.

The constant-w module gets a second, ducc0-independent oracle as well: at one
channel the same data *is* ``is_constant_w``, so each channel of the generic
multi-channel result can be checked against a single-channel fast-path plan --
a genuinely different branch of ``make_plan`` (one plane, no w-kernel, no
windows).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
from jax_nufft.planning import WGridderPlan
from tests.conftest import X64, Telescope, clumped_track, requires_x64, synthetic_uvw

ducc0_wgridder = pytest.importorskip("ducc0.wgridder")

# The ducc0 parity contract, identical to tests/test_against_ducc.py, and used
# on BOTH precision legs.
#
# There is no separate float32 factor. AGENTS.md section 6 says a bound that
# does not hold on a platform is measured there and set at 10x the measurement;
# the premise of that rule is a bound that does not hold, and here it holds
# with room to spare. Measured on this machine (macOS arm64, jax 0.9.2,
# ducc0 0.41.0, ``clumped_track`` seed 0, image/vis seed 7, shipped
# ``hermitian=True``, worst over all four ``w_strategy`` values), relative L2
# against the double-precision ducc0 oracle:
#
#     fixture / eps            float64 fwd   float64 adj   float32 fwd   float32 adj
#     EDA2         1e-4          8.858e-05     4.814e-05     8.890e-05     4.889e-05
#     EDA2         1e-6          9.463e-07     5.408e-07         --            --
#     MWA_extended 1e-4          6.200e-05     6.228e-05     7.183e-05     7.309e-05
#     MWA_extended 1e-6          7.481e-07     7.359e-07         --            --
#
# The unfolded leg, which this factor also governs since the parity cells are
# parametrised over ``hermitian``, lands lower: 0.474x-0.847x over both
# precisions, and the spread-w cells at 0.719x-0.770x.
#
# i.e. 0.95x eps at worst on the float64 leg (EDA2 forward at eps=1e-6) and
# 0.89x on the float32 one (EDA2 forward at eps=1e-4), worst over both folds -- the two legs land on
# top of each other, so this fixture is not float32-limited at eps=1e-4 and
# 3.0 is asserted on both, exactly as ``tests/test_dtype.py:84-93`` does on the
# same evidence for the off-zenith float32 fixtures.
DUCC_TOL_FACTOR = 3.0

# The exact-DFT accuracy contract, identical to tests/test_against_dft.py.
DFT_TOL_FACTOR = 2.0

# eps=1e-6 is below the float32 floor and ``make_plan`` warns about it, so the
# tighter cell is float64-only. eps=1e-4 runs on both legs.
_EPS_CELLS = [1e-4, pytest.param(1e-6, marks=requires_x64)]

_ALL_W_STRATEGIES = ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]

# Issue #17's Hermitian fold is the axis the first round of this module left
# frozen at the shipped default, and it is not a small one: ``make_plan`` folds
# ``w -> |w|`` before all plan geometry, so the two settings are different
# discretisations of the clumped track, not two spellings of one. Measured here
# at eps=1e-6, float64, seed 0 (True / False): EDA2 ``n_w`` 81 / 151 with 13 /
# 60 empty planes and ``max_window_size`` 389 / 196 of 400 rows; MWA_extended
# 214 / 461, 90 / 312 empty, 579 / 291 of 600.
_HERMITIAN_CELLS = [True, False]


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error, in float64 whatever precision the operands arrive in.

    Widened to complex128 unconditionally so the same helper serves the complex
    forward and the real adjoint without a cast that would silently discard an
    imaginary part.
    """
    x = np.asarray(a, dtype=np.complex128)
    y = np.asarray(b, dtype=np.complex128)
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))


def _data(tel: Telescope) -> tuple[np.ndarray, np.ndarray]:
    """``(image, vis)`` for ``tel``, in float64.

    Always float64, whatever the plan's precision: these arrays also feed the
    double-precision ducc0 oracle, and the operator call narrows its own copy.
    Generating them once in the widest precision keeps both sides of the
    comparison looking at the same numbers on both legs.
    """
    rng = np.random.default_rng(7)
    image = rng.standard_normal((tel.n_pix, tel.n_pix))
    vis = rng.standard_normal((tel.n_rows, 1)) + 1j * rng.standard_normal((tel.n_rows, 1))
    return image, vis


# ---------------------------------------------------------------------------
# 1. The fixture is what it claims to be
# ---------------------------------------------------------------------------


def test_clumped_track_changes_only_the_w_column(clumped_track_telescope: Telescope) -> None:
    """``clumped_track`` differs from the zenith fixture in ``w`` and nothing else.

    This is what licenses every comparison below to attribute a difference to
    the w-distribution: the ``(u, v)`` columns are bit-identical to
    ``synthetic_uvw(tel, 0.0, seed)``'s, so no second geometry change is
    smuggled in alongside.
    """
    tel = clumped_track_telescope
    clumped = clumped_track(tel, seed=0)
    zenith = synthetic_uvw(tel, 0.0, seed=0)

    assert clumped.shape == zenith.shape == (tel.n_rows, 3)
    np.testing.assert_array_equal(clumped[:, :2], zenith[:, :2])
    assert not np.array_equal(clumped[:, 2], zenith[:, 2])


def _excess_kurtosis(w: np.ndarray) -> float:
    d = np.asarray(w, dtype=np.float64) - np.mean(w)
    return float(np.mean(d**4) / np.std(d) ** 4 - 3.0)


def _clump_and_tail_fractions(w: np.ndarray) -> tuple[float, float]:
    """``(in_band, tail)`` fractions of ``|w|`` about its median.

    ``in_band`` is the fraction of rows within 5% of the ``|w|`` span of the
    median, ``tail`` the fraction more than 10% of the span above it. Stated on
    ``|w|`` because that is what a ``hermitian=True`` plan is built from
    (``planning.make_plan`` folds ``w -> |w|`` before all plan geometry), and
    about the median rather than a fitted mode so the statistic has no free
    parameters.
    """
    a = np.abs(np.asarray(w, dtype=np.float64))
    span = float(a.max() - a.min())
    centre = float(np.median(a))
    in_band = float(np.mean(np.abs(a - centre) <= 0.05 * span))
    tail = float(np.mean(a > centre + 0.10 * span))
    return in_band, tail


def test_clumped_track_w_is_clumped_where_the_gaussian_fixtures_are_not(
    clumped_track_telescope: Telescope,
) -> None:
    """The distribution really is clumps plus a sparse tail -- before *and* after the fold.

    Without this the generator could quietly decay into another Gaussian --
    exactly the failure this module exists to rule out -- and every parity
    assertion below would still pass while measuring nothing new. The test is
    stated as a comparison against the Gaussian fixture of the same telescope
    rather than as absolute thresholds, so it keeps its meaning if the
    generator's parameters are retuned.

    Two statements, because the parity cells below run at both ``hermitian``
    settings and the two settings are built from different columns:

    * ``hermitian=False`` builds the plane grid from the raw ``w`` column,
      which is *bimodal*: excess kurtosis is the standard indicator and it
      separates the families by sign, a two-point distribution sitting at the
      -2 floor and a Gaussian at 0. Measured here (seed 0): EDA2 clumped
      **-1.942**, MWA_extended clumped -1.903; EDA2 zenith +0.392, EDA2 off30
      -0.075, MWA_extended off30 +5.786.
    * ``hermitian=True`` -- the shipped default -- folds ``w -> |w|`` before all
      plan geometry (``planning.py``'s fold), so the plan does **not** see a
      bimodal column at all: it sees one dense clump at 0.4 * max_baseline plus
      a sparse tail running out to about 2.1x that. Excess kurtosis of ``|w|``
      is **+46.74** (EDA2) and +52.79 (MWA_extended) -- heavy-tailed, not
      bimodal -- so the kurtosis guard is worthless there and the fraction of
      rows in the clump is asserted instead. Measured: 0.9675 of EDA2's rows
      and 0.9650 of MWA_extended's lie within 5% of the ``|w|`` span of the
      median, against 0.245 (EDA2 zenith), 0.190 (EDA2 off30), 0.165 /
      0.415 (MWA_extended). Stable across seeds 0-4: 0.960-0.968.
    """
    tel = clumped_track_telescope
    w_clumped = clumped_track(tel, seed=0)[:, 2]

    # (a) the unfolded column, which a hermitian=False plan is built from.
    clumped_k = _excess_kurtosis(w_clumped)
    assert clumped_k < -1.5, (
        f"{tel.name}: clumped-track w has excess kurtosis {clumped_k:.3f}; a bimodal "
        "distribution sits near the -2 floor. The generator is no longer producing "
        "two clumps, so every hermitian=False parity test in this module has silently "
        "become another Gaussian one"
    )
    for angle in (0.0, 30.0):
        gauss_k = _excess_kurtosis(synthetic_uvw(tel, angle, seed=0)[:, 2])
        assert gauss_k > -1.0, (
            f"{tel.name} at {angle} deg: the Gaussian fixture's w has excess kurtosis "
            f"{gauss_k:.3f}. If synthetic_uvw ever became bimodal itself, this module "
            "would stop covering an axis the rest of the suite does not"
        )

    # (b) the folded |w|, which the shipped hermitian=True plan is built from.
    in_band, tail = _clump_and_tail_fractions(w_clumped)
    assert in_band > 0.9, (
        f"{tel.name}: only {in_band:.3f} of the folded |w| lies in the clump; the "
        "hermitian=True plan is built from |w| and needs a dense clump there, or "
        "every default-fold parity cell in this module is running on an ordinary "
        "spread-out w column"
    )
    assert 0.0 < tail < 0.1, (
        f"{tel.name}: {tail:.3f} of the folded |w| lies above the clump; the fixture "
        "needs a non-empty but sparse tail -- it is the tail that sets the w-extent, "
        "and the gap between it and the clump that leaves planes empty"
    )
    for angle in (0.0, 30.0):
        gauss_in_band, _ = _clump_and_tail_fractions(synthetic_uvw(tel, angle, seed=0)[:, 2])
        assert gauss_in_band < 0.6, (
            f"{tel.name} at {angle} deg: {gauss_in_band:.3f} of the Gaussian fixture's "
            "folded |w| is already concentrated in one band, so the clumped fixture is "
            "no longer covering a distribution shape the rest of the suite does not"
        )


@pytest.mark.parametrize("hermitian", _HERMITIAN_CELLS)
def test_the_clumped_plan_is_a_harder_windowed_plan_than_the_gaussian_one(
    clumped_track_telescope: Telescope, real_dtype, hermitian: bool
) -> None:
    """The clumped geometry costs the windowed path what a Gaussian one does not.

    The quantity is the ratio of the row-work a windowed traversal does to the
    row-work it cannot avoid (AGENTS.md section 4). A clumped w-distribution
    makes the two diverge when every plane pays the widest window:
    ``max_window_size`` is set by the clump while most planes hold nothing.
    Pinning the ordering is what makes the parity tests below a statement about
    a regime, not just about one more random fixture.

    issue #26 moves the assertion onto the un-bucketed form of the ratio,
    ``n_chan * n_w * max_window_size / live_row_count``, which is the one the
    regime claim is about -- and which stays computable, because #26 leaves
    every term of it on the plan. Bucketing gives each plane its own slice
    length, so the plan that had the most padding to lose loses the most of it
    and the *reported* overhead orders the other way round. Measured on this
    machine at eps=1e-4, EDA2, seed 0 (clumped / Gaussian off30), un-bucketed
    then reported:

        float64, hermitian=True   15.326 / 2.6177  ->  1.1505 / 1.2754
        float64, hermitian=False  14.595 / 2.6487  ->  1.2044 / 1.3563
        float32, hermitian=True   15.334 / 2.6177  ->  1.1511 / 1.2754
        float32, hermitian=False  14.595 / 2.6487  ->  1.2044 / 1.3563

    with 22 / 0, 76 / 0, 23 / 0 and 76 / 0 empty planes. The ``> 2x`` guard
    below therefore has a ~5.5x margin on the un-bucketed ratio, and the
    bucketed one is asserted for what it now is: bucketing removes far more of
    the clumped plan's padding (13.3x and 12.1x) than of the Gaussian one's
    (2.05x and 1.95x).

    Both ``hermitian`` settings, because the fold changes the plane grid rather
    than the summation order.
    """
    tel = clumped_track_telescope
    freq = np.array([tel.freq_hz])
    shape = (tel.n_pix, tel.n_pix)
    # eps=1e-4 rather than 1e-6 so this runs on the float32 leg too: the plan
    # geometry is what is under test, and a float32 plan builds its windows
    # from float32 w values, which is where a boundary-margin regression would
    # show first.
    kw = dict(
        image_shape=shape,
        pixsize_l=tel.pixsize,
        pixsize_m=tel.pixsize,
        epsilon=1e-4,
        dtype=real_dtype,
        hermitian=hermitian,
    )

    plan_clumped = make_plan(uvw=clumped_track(tel, seed=0), freq=freq, **kw)
    plan_gauss = make_plan(uvw=synthetic_uvw(tel, 30.0, seed=0), freq=freq, **kw)

    assert plan_clumped.empty_plane_count > 0
    assert plan_gauss.empty_plane_count == 0

    def unbucketed(plan: WGridderPlan) -> float:
        """Issue #43's overhead: one ``max_window_size`` slice per (channel, plane)."""
        return plan.n_chan * plan.n_w * plan.max_window_size / plan.live_row_count

    assert unbucketed(plan_clumped) > 2.0 * unbucketed(plan_gauss), (
        f"{tel.name} (hermitian={hermitian}): clumped un-bucketed padding overhead "
        f"{unbucketed(plan_clumped):.2f} vs Gaussian {unbucketed(plan_gauss):.2f} -- "
        "the clumped fixture is no longer stressing the windowed traversal any "
        "harder than the Gaussian one"
    )

    # issue #26: and the clumped plan is where bucketing pays, by a wide
    # margin, for exactly the reason the ordering above holds.
    clumped_gain = unbucketed(plan_clumped) / plan_clumped.window_padding_overhead
    gauss_gain = unbucketed(plan_gauss) / plan_gauss.window_padding_overhead
    assert clumped_gain > 2.0 * gauss_gain, (
        f"{tel.name} (hermitian={hermitian}): bucketing removed {clumped_gain:.2f}x "
        f"of the clumped plan's padded row-work and {gauss_gain:.2f}x of the "
        "Gaussian one's"
    )
    for plan in (plan_clumped, plan_gauss):
        assert plan.window_padding_overhead >= 1.0


# ---------------------------------------------------------------------------
# 2. ducc0 parity on the clumped track
# ---------------------------------------------------------------------------


def _ducc_forward(uvw, freq, image, pixsize, eps):
    return ducc0_wgridder.dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=np.ascontiguousarray(image, dtype=np.float64),
        pixsize_x=pixsize,
        pixsize_y=pixsize,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )


def _ducc_adjoint(uvw, freq, vis, n_pix, pixsize, eps):
    """``(n_chan, n_pix, n_pix)`` adjoint reference, one ducc0 call per channel.

    ducc0's ``vis2dirty`` **sums over channels** into a single image, while
    jax-nufft returns one image per channel, so a multi-channel call is not a
    reference for this operator at all -- it is a reference for the channel sum.
    Calling it per channel is what makes the comparison mean the same thing on
    one channel and on three.
    """
    vis = np.ascontiguousarray(vis, dtype=np.complex128)
    return np.stack(
        [
            ducc0_wgridder.vis2dirty(
                uvw=uvw,
                freq=np.asarray(freq[c : c + 1], dtype=np.float64),
                vis=np.ascontiguousarray(vis[:, c : c + 1]),
                npix_x=n_pix,
                npix_y=n_pix,
                pixsize_x=pixsize,
                pixsize_y=pixsize,
                epsilon=eps,
                do_wgridding=True,
                divide_by_n=True,
                nthreads=1,
            )
            for c in range(vis.shape[1])
        ]
    )


@pytest.mark.parametrize("hermitian", _HERMITIAN_CELLS)
@pytest.mark.parametrize("w_strategy", _ALL_W_STRATEGIES)
@pytest.mark.parametrize("eps", _EPS_CELLS)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_clumped_track_matches_ducc(
    clumped_track_telescope: Telescope,
    op: str,
    eps: float,
    w_strategy: str,
    hermitian: bool,
    real_dtype,
    complex_dtype,
) -> None:
    """Forward and adjoint parity on a clumped track, for all four traversals.

    All four ``w_strategy`` values, not the usual dense/windowed pair -- but
    not because the windows here are unusual. They are the opposite: measured
    ``max_window_size / n_rows`` at eps=1e-6 is 0.965-0.998 on all four
    telescopes at the shipped ``hermitian=True`` (EDA2 389/400, MWA_extended
    579/600, MWA_compact 599/600, MeerKAT 597/600) and 0.485-0.503 unfolded,
    against 0.147-0.258 for the off-zenith Gaussian fixtures the suite already
    runs. Clumping *widens* windows, because the clump sets the extent while
    most planes stay empty. These are among the most dense-like plans in the
    repository, and a 2% narrowing of the padded slice bounds passes all 95
    cells of this module while failing 49 tests elsewhere.

    What the four traversals cover here is plane *placement and accumulation*
    at the deepest stacks in the repository -- ``n_w`` 214 folded / 461
    unfolded against 134 for the deepest Gaussian, with 90 / 312 empty planes
    and an un-bucketed padding overhead of 29.50 against 4.94 (1.21 against
    1.38 on the metric issue #26 reports) -- and the ``vmap`` variants
    place their planes through a different (batched) composition than the
    ``scan`` ones. Window-bound coverage lives in
    :func:`test_multi_channel_spread_w_matches_ducc` (0.417 / 0.260).

    Both ``hermitian`` settings, because the fold is applied *before* all plan
    geometry, so it is not a summation-order variation on one plan but a second
    plan: on EDA2 at eps=1e-6 the unfolded plan is 151 planes with 60 empty
    against the folded 81 with 13. Both meet the same ``3 * eps`` ducc0 bound,
    ducc0 being given the unfolded uvw either way.
    """
    tel = clumped_track_telescope
    uvw = clumped_track(tel, seed=0)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    image, vis = _data(tel)

    plan = make_plan(
        uvw,
        freq,
        (tel.n_pix, tel.n_pix),
        pix,
        pix,
        eps,
        dtype=real_dtype,
        hermitian=hermitian,
    )
    # The fixture must actually reach the clumped regime on *this* plan, or the
    # parity statement is about an ordinary one. Empty planes are the direct
    # signature: the Gaussian fixtures of the same telescope have none.
    assert plan.n_w > 2 * plan.w_kernel_width, plan.n_w
    assert plan.empty_plane_count > 0, (
        f"{tel.name} clumped at eps={eps:g} (hermitian={hermitian}) has no empty "
        "w-planes: the clumping did not reach the plan and this cell duplicates the "
        "Gaussian fixtures"
    )

    if op == "dirty2vis":
        got = np.asarray(
            dirty2vis(plan, jnp.asarray(image, dtype=real_dtype), w_strategy=w_strategy)
        )
        want = _ducc_forward(uvw, freq, image, pix, eps)
    else:
        got = np.asarray(
            vis2dirty(plan, jnp.asarray(vis, dtype=complex_dtype), w_strategy=w_strategy)
        )
        want = _ducc_adjoint(uvw, freq, vis, tel.n_pix, pix, eps)

    err = _rel(got, want)
    bound = DUCC_TOL_FACTOR * eps
    assert err < bound, (
        f"{tel.name} clumped eps={eps:g} {op} {w_strategy} hermitian={hermitian}: "
        f"relative error {err:.3e} exceeds {bound:.3e}"
    )


@pytest.mark.parametrize("hermitian", _HERMITIAN_CELLS)
@pytest.mark.parametrize("w_strategy", ["dense_scan", "windowed_scan"])
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_clumped_track_matches_ducc_long(
    long_clumped_track_telescope: Telescope,
    op: str,
    w_strategy: str,
    hermitian: bool,
    real_dtype,
    complex_dtype,
) -> None:
    """The 256-pixel clumped fixtures (skipped without ``--runslow``).

    MWA_extended clumped is the extreme cell. Measured on this machine at
    eps=1e-6, float64, seed 0: ``n_w = 214`` folded against the Gaussian off30
    fixture's 134, with 90 of those planes empty and an un-bucketed padding
    overhead of 29.50 against 4.94 -- 1.21 against 1.38 on the bucketed metric
    issue #26 reports, both re-measured on this branch; unfolded it is 461
    planes with 312 empty.
    """
    eps = 1e-6 if X64 else 1e-4
    tel = long_clumped_track_telescope
    uvw = clumped_track(tel, seed=0)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    image, vis = _data(tel)

    plan = make_plan(
        uvw,
        freq,
        (tel.n_pix, tel.n_pix),
        pix,
        pix,
        eps,
        dtype=real_dtype,
        hermitian=hermitian,
    )
    assert plan.n_w > 2 * plan.w_kernel_width, plan.n_w
    assert plan.empty_plane_count > 0, (
        f"{tel.name} clumped at eps={eps:g} (hermitian={hermitian}) has no empty "
        "w-planes: the clumping did not reach the plan and this cell duplicates the "
        "Gaussian fixtures"
    )

    if op == "dirty2vis":
        got = np.asarray(
            dirty2vis(plan, jnp.asarray(image, dtype=real_dtype), w_strategy=w_strategy)
        )
        want = _ducc_forward(uvw, freq, image, pix, eps)
    else:
        got = np.asarray(
            vis2dirty(plan, jnp.asarray(vis, dtype=complex_dtype), w_strategy=w_strategy)
        )
        want = _ducc_adjoint(uvw, freq, vis, tel.n_pix, pix, eps)

    err = _rel(got, want)
    bound = DUCC_TOL_FACTOR * eps
    assert err < bound, (
        f"{tel.name} clumped eps={eps:g} {op} {w_strategy} hermitian={hermitian}: "
        f"relative error {err:.3e} exceeds {bound:.3e}"
    )


# ---------------------------------------------------------------------------
# 2b. The exact DFT on a small clumped geometry
# ---------------------------------------------------------------------------
#
# See "What this module does not claim" in the module docstring: the ducc0
# cells above cannot see a shared-mode accuracy regression that the exact-DFT
# contract catches immediately, so the clumped geometry needs a DFT cell of its
# own. It has to be tiny -- the reference is O(n_rows * n_pix^2) with a Python
# row loop -- so this is 48 rows on a 24^2 image, and the full-scale clumped
# coverage stays with the ducc0 cells above.
#
# The generator is ``conftest.clumped_track``'s shape, written out locally
# because ``clumped_track`` is tied to a ``Telescope``'s ``(u, v)`` columns and
# row count, and there is no 48-row telescope. Measured on this machine at
# float64, seed 5: ``hermitian=True`` gives n_w = 31 / 33 / 35 at
# eps = 1e-4 / 1e-6 / 1e-8 with 8 / 5 / 2 empty planes; ``hermitian=False``
# gives 53 / 55 / 57 with 25 / 19 / 13 empty and max_window_size 25 / 25 / 26
# of 48 rows. 23 of the 48 rows have w < 0, so the fold has work to do.
_DFT_N_PIX = 24
_DFT_N_ROWS = 48
_DFT_PIXSIZE = 0.02  # ~69 arcmin/pixel; max l^2 + m^2 = 0.115, well inside the disc
_DFT_FREQ = np.array([1.0e9])
_DFT_MAX_W_M = 100.0


def _small_clumped_geometry() -> np.ndarray:
    """``(48, 3)`` uvw in metres with ``clumped_track``'s w-distribution shape."""
    rng = np.random.default_rng(5)
    uvw = np.zeros((_DFT_N_ROWS, 3))
    uvw[:, 0] = rng.uniform(-100.0, 100.0, size=_DFT_N_ROWS)
    uvw[:, 1] = rng.uniform(-100.0, 100.0, size=_DFT_N_ROWS)
    n_tail = round(0.08 * _DFT_N_ROWS)
    n_clump = _DFT_N_ROWS - n_tail
    n_low = n_clump // 2
    w = np.empty(_DFT_N_ROWS)
    w[:n_low] = rng.normal(-0.4 * _DFT_MAX_W_M, 0.002 * _DFT_MAX_W_M, size=n_low)
    w[n_low:n_clump] = rng.normal(+0.4 * _DFT_MAX_W_M, 0.002 * _DFT_MAX_W_M, size=n_clump - n_low)
    w[n_clump:] = rng.uniform(-0.9 * _DFT_MAX_W_M, 0.9 * _DFT_MAX_W_M, size=n_tail)
    uvw[:, 2] = rng.permutation(w)
    return uvw


def _dft_lmn_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(l, m, n - 1)`` on the small image grid, matching ``planning.make_plan``.

    The fixture's FoV keeps every pixel inside the unit disc (max ``l^2 + m^2``
    is 0.115), which the caller asserts, so the analytic extension
    ``tests/test_against_dft.py::reference_lmn_grids`` needs for EDA2's
    120-degree field is not reproduced here.
    """
    i = np.arange(_DFT_N_PIX) - _DFT_N_PIX // 2
    ll, mm = np.meshgrid(i * _DFT_PIXSIZE, i * _DFT_PIXSIZE, indexing="ij")
    return ll, mm, np.sqrt(1.0 - ll * ll - mm * mm) - 1.0


def _dft_forward(image: np.ndarray, uvw: np.ndarray) -> np.ndarray:
    """Direct DFT, ducc's ``explicit_degridder`` sign convention, ``divide_by_n=False``.

    ``V(u, v, w) = sum_lm I(l, m) exp(-2 pi i (u l + v m - w (n - 1)))`` -- the
    sign convention recorded in AGENTS.md section 1, written out here rather
    than imported from ``tests/test_against_dft.py`` (see the module docstring:
    that module switches x64 on at import time).
    """
    ll, mm, nm1 = _dft_lmn_grids()
    scale = float(_DFT_FREQ[0]) / SPEED_OF_LIGHT
    u, v, w = uvw[:, 0] * scale, uvw[:, 1] * scale, uvw[:, 2] * scale
    out = np.zeros((uvw.shape[0], 1), dtype=np.complex128)
    for r in range(uvw.shape[0]):
        phase = -2j * np.pi * (u[r] * ll + v[r] * mm - w[r] * nm1)
        out[r, 0] = np.sum(image * np.exp(phase))
    return out


def _dft_adjoint(vis: np.ndarray, uvw: np.ndarray) -> np.ndarray:
    """Direct DFT adjoint, ducc's ``explicit_gridder`` with ``divide_by_n=True``."""
    ll, mm, nm1 = _dft_lmn_grids()
    scale = float(_DFT_FREQ[0]) / SPEED_OF_LIGHT
    u, v, w = uvw[:, 0] * scale, uvw[:, 1] * scale, uvw[:, 2] * scale
    out = np.zeros((_DFT_N_PIX, _DFT_N_PIX), dtype=np.float64)
    for r in range(uvw.shape[0]):
        phase = +2j * np.pi * (u[r] * ll + v[r] * mm - w[r] * nm1)
        out += (vis[r, 0] * np.exp(phase)).real
    return out / (nm1 + 1.0)


@requires_x64
@pytest.mark.parametrize("hermitian", _HERMITIAN_CELLS)
@pytest.mark.parametrize("w_strategy", ["dense_scan", "windowed_scan"])
@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_small_clumped_geometry_matches_the_exact_dft(
    op: str, eps: float, w_strategy: str, hermitian: bool
) -> None:
    """A clumped w-distribution against the definition of the answer, at ``2 * eps``.

    The one test in this module that can see a shared-mode accuracy loss at
    all. The module docstring records the measurement that motivates it:
    handing the whole epsilon budget to the (u, v) NUFFT
    (``_nufft_epsilon``'s ``eps / 10`` -> ``eps``) fails 48 pre-existing tests,
    all of them exact-DFT comparisons, and not one ducc0 parity test anywhere
    in the repository -- so the ducc0 cells above, clumped or not, cannot stand
    in for this.

    ``float64`` only: ``2 * eps`` at these epsilons is far below the single
    precision floor, which is why the whole of
    ``tests/test_against_dft.py`` is in ``conftest.collect_ignore`` on the
    x64-off leg.

    Measured on this machine (seed 5, image/vis seed 7, worst over both
    strategies and both fold settings): 0.47-0.58x eps at eps=1e-4,
    0.67-0.76x at 1e-6, and 0.98-1.11x at 1e-8, against the 2x contract.
    Under that ``_nufft_epsilon`` mutation the same cells measure 1.12-1.51x,
    1.52-2.11x and 1.17-1.24x, so it is the four eps=1e-6 ``vis2dirty`` cells
    that actually cross the bound -- a real detection with a thin margin, not a
    10x one. The regression is 10x in the (u, v) NUFFT's share of the budget,
    and that share is not what dominates this fixture's total error; nothing
    weaker than a DFT contract sees even this much.
    """
    uvw = _small_clumped_geometry()
    ll, mm, _ = _dft_lmn_grids()
    assert float((ll * ll + mm * mm).max()) < 1.0, (
        "the DFT reference here assumes every pixel is inside the unit disc; "
        "widen it to the analytic extension if the fixture's FoV grows"
    )
    n_negative = int((uvw[:, 2] < 0).sum())
    assert 0 < n_negative < _DFT_N_ROWS, (
        f"{n_negative} of {_DFT_N_ROWS} rows have w < 0: the hermitian=True leg needs "
        "the fold to have something to do, or it is the hermitian=False leg renamed"
    )

    rng = np.random.default_rng(7)
    image = rng.standard_normal((_DFT_N_PIX, _DFT_N_PIX))
    vis = rng.standard_normal((_DFT_N_ROWS, 1)) + 1j * rng.standard_normal((_DFT_N_ROWS, 1))

    plan = make_plan(
        uvw,
        _DFT_FREQ,
        (_DFT_N_PIX, _DFT_N_PIX),
        _DFT_PIXSIZE,
        _DFT_PIXSIZE,
        eps,
        hermitian=hermitian,
    )
    # Same anti-vacuity guards as the ducc0 cells: this has to be a clumped
    # plan, not a small one that happens to share the generator.
    assert plan.n_w > 2 * plan.w_kernel_width, plan.n_w
    assert plan.empty_plane_count > 0, (
        f"the small clumped geometry at eps={eps:g} (hermitian={hermitian}) has no "
        "empty w-planes: it is no longer a clumped fixture"
    )

    if op == "dirty2vis":
        got = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy=w_strategy))
        want = _dft_forward(image, uvw)
    else:
        got = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy=w_strategy))[0]
        want = _dft_adjoint(vis, uvw)

    err = _rel(got, want)
    bound = DFT_TOL_FACTOR * eps
    assert err < bound, (
        f"small clumped geometry eps={eps:g} {op} {w_strategy} hermitian={hermitian}: "
        f"relative error {err:.3e} exceeds {bound:.3e} (n_w={plan.n_w})"
    )


# ---------------------------------------------------------------------------
# 3. Multi-channel: constant w, and a spread w for the window bounds
# ---------------------------------------------------------------------------

# ``w == 12.5 m`` on every row -- the value tests/test_constant_w.py uses for
# its non-zero coplanar case -- with three channels spanning a factor of two.
# In metres the w-extent is zero; in wavelengths it is
# ``12.5 * (f_max - f_min) / c``, which is not, so ``is_constant_w`` is False
# and the generic path engages *on its own*, with no ``_force_generic``.
_CONST_W_METRES = 12.5
_CONST_W_FREQ = np.array([1.0e9, 1.5e9, 2.0e9])


def _constant_w_case(n_rows: int = 96, n_pix: int = 64) -> tuple[np.ndarray, float, int]:
    rng = np.random.default_rng(1)
    uvw = rng.normal(scale=120.0, size=(n_rows, 3))
    uvw[:, 2] = _CONST_W_METRES
    return uvw, 1e-3, n_pix


def test_multi_channel_constant_w_does_not_take_the_fast_path(real_dtype) -> None:
    """Constant w in metres is *not* constant w in wavelengths across channels.

    The precondition for everything below, asserted rather than assumed: if a
    planning change ever made ``is_constant_w`` true here, the two parity tests
    that follow would silently stop exercising the generic path they exist to
    cover.

    The second assertion records why this fixture is a *dense*-path fixture.
    Every row has the same w in metres, so for any (channel, plane) the window
    either contains every row or none of them: ``sort_perm`` is over a constant
    array and the two ``searchsorted`` boundaries land together. ("None" is
    exact for the nominal support that ``empty_plane_count`` counts; the padded
    slice is then widened to a single row by the ``+/-1`` clamp in
    ``planning.py``, which does not change the conclusion below.) Some plane
    always covers each channel's single spike, so ``max_window_size`` is
    ``n_rows`` no matter how the geometry is retuned -- and a
    ``dynamic_slice`` of ``n_rows`` rows out of ``n_rows`` is the dense
    traversal exactly. That is a structural fact about constant-w data, not a
    tuning accident: a larger FoV or a larger ``_CONST_W_METRES`` moves ``n_w``
    but cannot move this. So the ducc0 cells below run the dense strategies
    only, and the windowed slice bounds on multi-channel data are covered by
    :func:`test_multi_channel_spread_w_matches_ducc` instead, whose w column is
    spread and whose ``max_window_size`` is 40 of 96 rows.
    """
    uvw, pixsize, n_pix = _constant_w_case()
    # eps=1e-4 and the active precision, so the float32 leg covers this too:
    # ``w_extent`` is formed from ``inv_lambda`` cast to the plan's real dtype,
    # and it is exactly the per-channel difference that has to survive that cast.
    eps = 1e-4
    plan = make_plan(uvw, _CONST_W_FREQ, (n_pix, n_pix), pixsize, pixsize, eps, dtype=real_dtype)

    assert plan.n_chan == 3
    assert not plan.is_constant_w, (
        "a single constant w in metres became is_constant_w over three channels: "
        "w in wavelengths is w * freq[c] / c and must differ per channel"
    )
    assert plan.w_extent > 0.0
    assert plan.n_w > 1

    assert plan.max_window_size == plan.n_rows, (
        f"max_window_size is {plan.max_window_size} of {plan.n_rows} rows on "
        "exactly-constant-w data. That is not supposed to be possible (see this "
        "docstring), and if it has become possible the windowed strategies belong "
        "back in the parametrisation below"
    )

    # One channel at a time, the very same rows *do* collapse to the fast path.
    for freq_hz in _CONST_W_FREQ:
        single = make_plan(
            uvw, np.array([freq_hz]), (n_pix, n_pix), pixsize, pixsize, eps, dtype=real_dtype
        )
        assert single.is_constant_w and single.n_w == 1 and single.w_extent == 0.0


@requires_x64
@pytest.mark.parametrize("w_strategy", ["dense_scan", "dense_vmap"])
@pytest.mark.parametrize("channel_strategy", ["scan", "vmap"])
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_multi_channel_constant_w_matches_ducc(
    op: str, channel_strategy: str, w_strategy: str
) -> None:
    """The generic path on constant-w data must still track ducc0.

    ``tests/test_constant_w.py`` compares the generic path only against the
    *fast* path on the same data, and only at one channel. What is new here is
    the w-distribution the generic path is handed: ``n_chan`` discrete spikes,
    one per channel, with nothing between them.

    Dense strategies only. On exactly-constant-w data a windowed cell is the
    dense computation reached through the windowed spelling -- ``max_window_size
    == n_rows``, asserted and argued in
    :func:`test_multi_channel_constant_w_does_not_take_the_fast_path` -- so
    running the windowed pair here would look like window-bound coverage
    without being any. Measured at eps=1e-6: this plan is ``n_w = 8``, which is
    also why it is not a plane-count probe; the deep-stack coverage in this
    module is the clumped track (``n_w`` 81 and 214) and, on the x64-off leg,
    ``tests/test_dtype.py``'s off30 fixtures at 132.
    """
    eps = 1e-6
    uvw, pixsize, n_pix = _constant_w_case()
    rng = np.random.default_rng(7)
    image = rng.standard_normal((n_pix, n_pix))
    vis = (
        rng.standard_normal((uvw.shape[0], 3)) + 1j * rng.standard_normal((uvw.shape[0], 3))
    ).astype(np.complex128)

    plan = make_plan(uvw, _CONST_W_FREQ, (n_pix, n_pix), pixsize, pixsize, eps)
    assert not plan.is_constant_w

    if op == "dirty2vis":
        got = np.asarray(
            dirty2vis(
                plan,
                jnp.asarray(image),
                w_strategy=w_strategy,
                channel_strategy=channel_strategy,
            )
        )
        want = _ducc_forward(uvw, _CONST_W_FREQ, image, pixsize, eps)
    else:
        got = np.asarray(
            vis2dirty(
                plan,
                jnp.asarray(vis),
                w_strategy=w_strategy,
                channel_strategy=channel_strategy,
            )
        )
        want = _ducc_adjoint(uvw, _CONST_W_FREQ, vis, n_pix, pixsize, eps)

    err = _rel(got, want)
    assert err < DUCC_TOL_FACTOR * eps, (
        f"multi-channel constant-w {op} {w_strategy}/{channel_strategy}: relative "
        f"error {err:.3e} exceeds {DUCC_TOL_FACTOR * eps:.3e}"
    )
    # Per channel too: an aggregate norm over three channels is dominated by
    # the loudest, so a single mis-planned channel can hide inside a good total
    # -- and per-channel w in wavelengths is exactly what differs here.
    for c in range(3):
        got_c = got[:, c] if op == "dirty2vis" else got[c]
        want_c = want[:, c] if op == "dirty2vis" else want[c]
        err_c = _rel(got_c, want_c)
        assert err_c < DUCC_TOL_FACTOR * eps, (
            f"multi-channel constant-w {op} channel {c} "
            f"(freq={_CONST_W_FREQ[c]:.3g} Hz): relative error {err_c:.3e}"
        )


# The multi-channel fixture that *can* test a window bound. Same three channels
# as the constant-w case, but with the w column spread instead of collapsed, so
# each (channel, plane) window is a strict subset of the rows and the windows of
# different channels cover different rows. Measured on this machine at
# eps=1e-6, float64, seed 1: ``n_w = 68`` at ``w_kernel_width = 7``,
# ``max_window_size = 40`` of 96 rows, 49 empty planes,
# ``window_padding_overhead = 4.05``.
_SPREAD_W_N_ROWS = 96
_SPREAD_W_N_PIX = 64
_SPREAD_W_PIXSIZE = 4e-3


def _spread_w_case() -> np.ndarray:
    rng = np.random.default_rng(1)
    uvw = rng.normal(scale=120.0, size=(_SPREAD_W_N_ROWS, 3))
    uvw[:, 2] = rng.normal(scale=120.0, size=_SPREAD_W_N_ROWS)
    return uvw


@requires_x64
@pytest.mark.parametrize("w_strategy", _ALL_W_STRATEGIES)
@pytest.mark.parametrize("channel_strategy", ["scan", "vmap"])
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_multi_channel_spread_w_matches_ducc(
    op: str, channel_strategy: str, w_strategy: str
) -> None:
    """Per-channel window bounds on a genuinely windowed multi-channel plan.

    ``plan.window_start`` is ``(n_chan, n_w)`` and each channel's boundaries are
    found by scaling the shared w-in-metres by *that channel's*
    ``inv_lambda[c]`` (AGENTS.md section 4). A single-channel test cannot tell
    ``inv_lambda[c]`` from ``inv_lambda[0]``, and neither can a multi-channel
    test whose windows all span every row -- which is what
    ``_constant_w_case`` produces and why the constant-w cells above are dense
    only.

    This fixture is the missing one: three channels spanning a factor of two in
    frequency, ``max_window_size`` 40 of 96 rows, 68 planes of which 49 are
    empty. The per-channel assertion is the load-bearing half -- a window built
    at the wrong channel's scale drops rows from one channel while leaving the
    others intact, and an aggregate norm over three channels dilutes that.

    Measured on this machine at eps=1e-6, float64, seed 1 (worst over all four
    ``w_strategy`` x both ``channel_strategy``): 0.72x eps forward and 0.75x
    adjoint aggregate, 0.75x / 0.77x per channel, against the 3x contract.

    Checked to be non-vacuous by mutation, not by assertion alone: deriving
    every channel's window boundaries from ``inv_lambda_np[0]`` instead of
    ``inv_lambda_np[c]`` in ``planning.py``'s window builder fails the eight
    ``windowed_*`` cells here. Before this fixture existed that mutation was
    caught only by two plan-bookkeeping tests
    (``test_padding_overhead::test_accumulators_sum_over_every_channel`` and
    ``test_planning::test_window_builder_matches_independent_reference``), and
    by no numerical-oracle test anywhere in the repository.
    """
    eps = 1e-6
    uvw = _spread_w_case()
    n_pix = _SPREAD_W_N_PIX
    pixsize = _SPREAD_W_PIXSIZE
    rng = np.random.default_rng(7)
    image = rng.standard_normal((n_pix, n_pix))
    vis = (
        rng.standard_normal((_SPREAD_W_N_ROWS, 3)) + 1j * rng.standard_normal((_SPREAD_W_N_ROWS, 3))
    ).astype(np.complex128)

    plan = make_plan(uvw, _CONST_W_FREQ, (n_pix, n_pix), pixsize, pixsize, eps)
    assert plan.n_chan == 3
    # The anti-vacuity guard: if every window spans every row, ``windowed_*``
    # is ``dense_*`` under another name and these cells stop being about window
    # bounds at all.
    assert plan.max_window_size < plan.n_rows, (
        f"max_window_size is {plan.max_window_size} of {plan.n_rows} rows: every "
        "window holds every row, so the windowed cells here are the dense ones "
        "again and no per-channel window bound is under test"
    )
    assert plan.n_w > 2 * plan.w_kernel_width, plan.n_w

    if op == "dirty2vis":
        got = np.asarray(
            dirty2vis(
                plan,
                jnp.asarray(image),
                w_strategy=w_strategy,
                channel_strategy=channel_strategy,
            )
        )
        want = _ducc_forward(uvw, _CONST_W_FREQ, image, pixsize, eps)
    else:
        got = np.asarray(
            vis2dirty(
                plan,
                jnp.asarray(vis),
                w_strategy=w_strategy,
                channel_strategy=channel_strategy,
            )
        )
        want = _ducc_adjoint(uvw, _CONST_W_FREQ, vis, n_pix, pixsize, eps)

    bound = DUCC_TOL_FACTOR * eps
    err = _rel(got, want)
    assert err < bound, (
        f"multi-channel spread-w {op} {w_strategy}/{channel_strategy}: relative "
        f"error {err:.3e} exceeds {bound:.3e}"
    )
    for c in range(3):
        got_c = got[:, c] if op == "dirty2vis" else got[c]
        want_c = want[:, c] if op == "dirty2vis" else want[c]
        err_c = _rel(got_c, want_c)
        assert err_c < bound, (
            f"multi-channel spread-w {op} {w_strategy}/{channel_strategy} channel {c} "
            f"(freq={_CONST_W_FREQ[c]:.3g} Hz): relative error {err_c:.3e} exceeds "
            f"{bound:.3e}"
        )


@requires_x64
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_multi_channel_constant_w_matches_the_per_channel_fast_path(op: str) -> None:
    """Each channel of the generic result equals the ``n_w == 1`` fast path's.

    A ducc0-independent oracle for the same claim, and a much sharper one: at
    one channel this data *is* ``is_constant_w``, so the fast path -- one plane,
    no w-kernel, no windows, no ``phi_hat`` correction -- computes the same
    answer through an entirely different branch of ``make_plan``. The bound is
    the sum of the two paths' own budgets, the same ``3 * eps`` reasoning
    ``tests/test_constant_w.py`` uses.
    """
    eps = 1e-6
    uvw, pixsize, n_pix = _constant_w_case()
    rng = np.random.default_rng(7)
    image = rng.standard_normal((n_pix, n_pix))
    vis = (
        rng.standard_normal((uvw.shape[0], 3)) + 1j * rng.standard_normal((uvw.shape[0], 3))
    ).astype(np.complex128)

    multi = make_plan(uvw, _CONST_W_FREQ, (n_pix, n_pix), pixsize, pixsize, eps)
    assert not multi.is_constant_w

    if op == "dirty2vis":
        got = np.asarray(dirty2vis(multi, jnp.asarray(image)))
    else:
        got = np.asarray(vis2dirty(multi, jnp.asarray(vis)))

    for c, freq_hz in enumerate(_CONST_W_FREQ):
        single = make_plan(uvw, np.array([freq_hz]), (n_pix, n_pix), pixsize, pixsize, eps)
        assert single.is_constant_w and single.n_w == 1
        if op == "dirty2vis":
            want = np.asarray(dirty2vis(single, jnp.asarray(image)))[:, 0]
            got_c = got[:, c]
        else:
            want = np.asarray(vis2dirty(single, jnp.asarray(vis[:, c : c + 1])))[0]
            got_c = got[c]
        err = _rel(got_c, want)
        assert err < DUCC_TOL_FACTOR * eps, (
            f"multi-channel constant-w {op}: channel {c} (freq={freq_hz:.3g} Hz) "
            f"differs from the single-channel constant-w fast path by {err:.3e}, "
            f"above {DUCC_TOL_FACTOR * eps:.3e}"
        )
