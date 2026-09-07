"""Clumped w-distributions and multi-channel constant-w, against ducc0 (issue #15).

Two fixtures live here, and both exist because of an axis every other fixture in
this repository holds constant.

**The clumped track.** ``tests/conftest.py``'s ``synthetic_uvw`` produces a
symmetric, unimodal w-distribution at every pointing -- a Gaussian ``z`` offset
at zenith, a rotated Gaussian ``u`` off it. So the *shape* of the
w-distribution is constant across the whole review fixture set, and it is
constant at the one shape the windowed strategies have least to say about.
``tests/test_boundary_planes.py`` does build clumped distributions, but it only
compares the windowed path against the dense one on the same plan: a
reduction-order comparison between two spellings of one operator, blind by
construction to anything the two share -- a mis-placed plane centre, a plane
grid that under-resolves a clump. That is the shared-mode failure mode AGENTS.md
section 6 records from issue #16, and it is why "windowed == dense on a clumped
fixture" is not evidence that either is right. ``conftest.clumped_track`` is the
same clumped geometry held against an **external** oracle instead.

**Multi-channel constant w.** ``plan.is_constant_w`` is ``w_extent == 0`` where
``w_extent`` is measured in *wavelengths, over all channels*. Data with a single
constant ``w`` in metres therefore takes the ``n_w == 1`` fast path at one
channel and the full generic path at three, because ``w * freq[c] / c`` differs
per channel. ``tests/test_constant_w.py`` covers only the single-channel case
(and reaches the generic path solely through the ``_force_generic`` test
kwarg), so the configuration a real multi-frequency observation of coplanar
data actually produces -- generic path, w-distribution collapsed onto ``n_chan``
discrete spikes -- was untested.

Oracle
------
ducc0's public Python API, as in ``tests/test_against_ducc.py``, at that
module's ``3 * eps`` contract. The exact-DFT references are deliberately *not*
imported: ``tests/test_against_dft.py`` and ``tests/test_adjoint.py`` call
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
from tests.conftest import X64, Telescope, clumped_track, requires_x64, synthetic_uvw, tol

ducc0_wgridder = pytest.importorskip("ducc0.wgridder")

# The ducc0 parity contract, identical to tests/test_against_ducc.py.
DUCC_TOL_FACTOR = 3.0

# The float32 leg gets its own factor rather than reusing 3.0, per AGENTS.md
# section 6: a bound that does not hold on a leg is measured there and set at
# 10x the measurement, not loosened to whatever passes. What the measurement
# actually says here is that this fixture is *not* float32-limited at eps=1e-4
# -- the two legs land on top of each other (macOS arm64, jax 0.9.2,
# ducc0 0.41.0, clumped_track seed 0, all four strategies):
#
#     fixture / eps            float64 fwd   float64 adj   float32 fwd   float32 adj
#     EDA2         1e-4          8.86e-05      4.81e-05      8.89e-05      4.89e-05
#     EDA2         1e-6          9.46e-07      5.41e-07          --            --
#     MWA_extended 1e-4          6.20e-05      6.23e-05      7.18e-05      7.31e-05
#     MWA_extended 1e-6          7.48e-07      7.36e-07          --            --
#
# i.e. 0.89x eps at worst on either leg, against a 3x contract. 10x the worst
# float32 measurement is 8.9e-4, so the float32 factor is 10 and the float64
# one stays at the repo's 3. The gap between them is headroom for a platform
# whose FINUFFT bins sort differently (Linux CI), not for a real difference
# this fixture has shown.
DUCC_TOL_FACTOR_F32 = 10.0

# eps=1e-6 is below the float32 floor and ``make_plan`` warns about it, so the
# tighter cell is float64-only. eps=1e-4 runs on both legs.
_EPS_CELLS = [1e-4, pytest.param(1e-6, marks=requires_x64)]

_ALL_W_STRATEGIES = ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]


def _tol(eps: float) -> float:
    return tol(DUCC_TOL_FACTOR, DUCC_TOL_FACTOR_F32) * eps


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


def test_clumped_track_w_is_bimodal_where_the_gaussian_fixtures_are_not(
    clumped_track_telescope: Telescope,
) -> None:
    """The distribution really is two clumps plus a sparse tail.

    Without this the generator could quietly decay into another Gaussian --
    exactly the failure this module exists to rule out -- and every parity
    assertion below would still pass while measuring nothing new. The test is
    stated as a comparison against the Gaussian fixture of the same telescope
    rather than as absolute thresholds, so it keeps its meaning if the
    generator's parameters are retuned.
    """
    tel = clumped_track_telescope
    w_clumped = clumped_track(tel, seed=0)[:, 2]

    # Excess kurtosis is the standard bimodality indicator and it separates the
    # two families by sign: a two-point distribution sits at the -2 floor, a
    # Gaussian at 0. Measured on the fixtures: clumped -1.94, EDA2 zenith +0.39,
    # EDA2 off30 -0.08, MWA_extended off30 +5.79. The thresholds below are set
    # between those, not at them.
    def _excess_kurtosis(w: np.ndarray) -> float:
        d = np.asarray(w, dtype=np.float64) - np.mean(w)
        return float(np.mean(d**4) / np.std(d) ** 4 - 3.0)

    clumped_k = _excess_kurtosis(w_clumped)
    assert clumped_k < -1.5, (
        f"{tel.name}: clumped-track w has excess kurtosis {clumped_k:.3f}; a bimodal "
        "distribution sits near the -2 floor. The generator is no longer producing "
        "two clumps, so every parity test in this module has silently become "
        "another Gaussian one"
    )
    for angle in (0.0, 30.0):
        gauss_k = _excess_kurtosis(synthetic_uvw(tel, angle, seed=0)[:, 2])
        assert gauss_k > -1.0, (
            f"{tel.name} at {angle} deg: the Gaussian fixture's w has excess kurtosis "
            f"{gauss_k:.3f}. If synthetic_uvw ever became bimodal itself, this module "
            "would stop covering an axis the rest of the suite does not"
        )

    # A sparse tail spanning most of the range, not a pair of isolated deltas:
    # the rows between the clumps are what force the empty planes.
    lo, hi = w_clumped.min(), w_clumped.max()
    mid_band = np.abs(w_clumped - w_clumped.mean()) < 0.15 * (hi - lo)
    assert 0 < int(mid_band.sum()) < 0.1 * w_clumped.size, (
        f"{tel.name}: {int(mid_band.sum())} of {w_clumped.size} rows lie in the "
        "middle of the w-range; the fixture needs a non-empty but sparse tail there"
    )


def test_the_clumped_plan_is_a_harder_windowed_plan_than_the_gaussian_one(
    clumped_track_telescope: Telescope, real_dtype
) -> None:
    """The clumped geometry costs the windowed path what a Gaussian one does not.

    ``window_padding_overhead`` is the ratio of the row-work a windowed
    traversal does (``n_chan * n_w * max_window_size``) to the row-work it
    cannot avoid (AGENTS.md section 4). A bimodal w-distribution makes the two
    diverge: ``max_window_size`` is set by a clump while most planes hold
    nothing. Pinning the ordering is what makes the parity tests below a
    statement about a regime, not just about one more random fixture.
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
    )

    plan_clumped = make_plan(uvw=clumped_track(tel, seed=0), freq=freq, **kw)
    plan_gauss = make_plan(uvw=synthetic_uvw(tel, 30.0, seed=0), freq=freq, **kw)

    assert plan_clumped.empty_plane_count > 0
    assert plan_gauss.empty_plane_count == 0

    assert plan_clumped.window_padding_overhead > 2.0 * plan_gauss.window_padding_overhead, (
        f"{tel.name}: clumped padding overhead {plan_clumped.window_padding_overhead:.2f} "
        f"vs Gaussian {plan_gauss.window_padding_overhead:.2f} -- the clumped fixture is "
        "no longer stressing the windowed traversal any harder than the Gaussian one"
    )


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


@pytest.mark.parametrize("w_strategy", _ALL_W_STRATEGIES)
@pytest.mark.parametrize("eps", _EPS_CELLS)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_clumped_track_matches_ducc(
    clumped_track_telescope: Telescope,
    op: str,
    eps: float,
    w_strategy: str,
    real_dtype,
    complex_dtype,
) -> None:
    """Forward and adjoint parity on a clumped track, for all four traversals.

    All four ``w_strategy`` values, not the usual dense/windowed pair: this is
    the fixture where the windowed slice bounds are least like the dense loop's,
    and the ``vmap`` variants place their planes through a different (batched)
    composition than the ``scan`` ones.
    """
    tel = clumped_track_telescope
    uvw = clumped_track(tel, seed=0)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    image, vis = _data(tel)

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps, dtype=real_dtype)
    # The fixture must actually reach the clumped regime on *this* plan, or the
    # parity statement is about an ordinary one. Empty planes are the direct
    # signature: the Gaussian fixtures of the same telescope have none.
    assert plan.n_w > 2 * plan.w_kernel_width, plan.n_w
    assert plan.empty_plane_count > 0, (
        f"{tel.name} clumped at eps={eps:g} has no empty w-planes: the clumping did "
        "not reach the plan and this cell duplicates the Gaussian fixtures"
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
    bound = _tol(eps)
    assert err < bound, (
        f"{tel.name} clumped eps={eps:g} {op} {w_strategy}: relative error {err:.3e} "
        f"exceeds {bound:.3e}"
    )


@pytest.mark.parametrize("w_strategy", ["dense_scan", "windowed_scan"])
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_clumped_track_matches_ducc_long(
    long_clumped_track_telescope: Telescope,
    op: str,
    w_strategy: str,
    real_dtype,
    complex_dtype,
) -> None:
    """The 256-pixel clumped fixtures (skipped without ``--runslow``).

    MWA_extended clumped is the extreme cell: ``n_w = 214`` at eps=1e-6 against
    the Gaussian off30 fixture's 134, with 90 of those planes empty and a
    padding overhead of 29.5 against 4.94.
    """
    eps = 1e-6 if X64 else 1e-4
    tel = long_clumped_track_telescope
    uvw = clumped_track(tel, seed=0)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    image, vis = _data(tel)

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps, dtype=real_dtype)
    assert plan.n_w > 2 * plan.w_kernel_width, plan.n_w
    assert plan.empty_plane_count > 0, (
        f"{tel.name} clumped at eps={eps:g} has no empty w-planes: the clumping did "
        "not reach the plan and this cell duplicates the Gaussian fixtures"
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
    bound = _tol(eps)
    assert err < bound, (
        f"{tel.name} clumped eps={eps:g} {op} {w_strategy}: relative error {err:.3e} "
        f"exceeds {bound:.3e}"
    )


# ---------------------------------------------------------------------------
# 3. Multi-channel constant w: the generic path on a w-extent it did not ask for
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

    # One channel at a time, the very same rows *do* collapse to the fast path.
    for freq_hz in _CONST_W_FREQ:
        single = make_plan(
            uvw, np.array([freq_hz]), (n_pix, n_pix), pixsize, pixsize, eps, dtype=real_dtype
        )
        assert single.is_constant_w and single.n_w == 1 and single.w_extent == 0.0


@requires_x64
@pytest.mark.parametrize("w_strategy", _ALL_W_STRATEGIES)
@pytest.mark.parametrize("channel_strategy", ["scan", "vmap"])
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_multi_channel_constant_w_matches_ducc(
    op: str, channel_strategy: str, w_strategy: str
) -> None:
    """The generic path on constant-w data must still track ducc0.

    ``tests/test_constant_w.py`` compares the generic path only against the
    *fast* path on the same data, and only for ``dense_scan`` / ``dense_vmap``
    at one channel. Both windowed strategies on a w-distribution that is
    ``n_chan`` discrete spikes -- every row of a channel in the same window or
    in none -- were reached by nothing.
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
