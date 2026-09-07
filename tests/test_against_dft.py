"""Tiny-problem checks against the explicit DFT.

These tests use a small image and small ``Nrow`` so that we can afford the
full O(Nrow * Nl * Nm) reference DFT. They verify the *math* of the wgridder
end-to-end at the smallest non-trivial scale, independent of ducc.

The exact DFT is the *definition* of the answer, so the acceptance bound here
is the accuracy contract of the whole library: ``err < 2 * eps``. The former
``10 * eps`` allowance predated the FINUFFT width rule and was loose enough to
hide a ``w_kernel_width`` that was up to three cells short of what the
requested epsilon needs (see issue #9): the measured ratio was ~4x eps at
1e-6..1e-8 and several hundred x eps at 1e-12, while ducc0 stays at or below
0.24x eps on the same inputs. Two is a deliberately small constant -- it
allows for the reference DFT's own conditioning and for the fact that jax and
FINUFFT each target epsilon independently -- but not for a systematically
under-provisioned kernel.
"""

from __future__ import annotations

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
from jax_nufft.kernel import kernel_params
from tests.conftest import MWA_COMPACT, reference_lmn_grids, synthetic_uvw
from tests.test_adjoint import _reference_adjoint

jax.config.update("jax_enable_x64", True)

# Accuracy contract against the exact DFT (issue #9). See the module docstring.
DFT_TOL_FACTOR = 2.0


def _reference_forward(
    image: np.ndarray, uvw: np.ndarray, freq: np.ndarray, pixsize_l: float, pixsize_m: float
) -> np.ndarray:
    """Direct DFT matching ducc's explicit_degridder sign convention.

    image: (n_chan, n_l, n_m) complex
    uvw:   (n_rows, 3) in metres
    freq:  (n_chan,) in Hz
    Returns: vis (n_rows, n_chan) complex.
    """
    n_chan, n_l, n_m = image.shape
    n_rows = uvw.shape[0]
    LL, MM, nm1 = reference_lmn_grids((n_l, n_m), pixsize_l, pixsize_m)
    out = np.zeros((n_rows, n_chan), dtype=np.complex128)
    for c in range(n_chan):
        scale = freq[c] / SPEED_OF_LIGHT
        u = uvw[:, 0] * scale
        v = uvw[:, 1] * scale
        w = uvw[:, 2] * scale
        for r in range(n_rows):
            # Match ducc: phase = -2 pi i (u l + v m - w (n - 1))
            phase = -2j * np.pi * (u[r] * LL + v[r] * MM - w[r] * nm1)
            out[r, c] = np.sum(image[c] * np.exp(phase))
    return out


@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("hermitian", [False, True])
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_forward_matches_dft_single_channel_zenith(
    eps: float, w_strategy: str, hermitian: bool
) -> None:
    """Tiny zenith problem: w in metres deliberately non-zero but small.

    Parametrised over issue #17's Hermitian w-sign fold, which folds every row
    with ``w < 0`` onto ``(-u, -v, -w)`` with a conjugated value. The fold is an
    *exact* identity for a real sky, so both settings must meet the same
    ``2 * eps`` contract against the exact DFT -- there is no accuracy budget to
    spend on it, and a fold that were merely "close" would show up here first.

    The image is real, which is the fold's precondition (``hermitian=True``
    plans refuse a complex image; see
    ``tests/test_hermitian.py::test_dirty2vis_rejects_a_complex_image_on_a_
    folded_plan``). Complex-image forward parity is still covered at full
    strength by ``test_forward_matches_dft_off_zenith`` and
    ``test_multi_channel_matches_dft_forward_and_adjoint`` below, both of which
    pin ``hermitian=False`` explicitly.

    ``hermitian`` is passed explicitly in both legs so this test says the same
    thing whichever way ``make_plan``'s default is set.
    """
    rng = np.random.default_rng(123)
    n_l = n_m = 16
    n_rows = 24
    pixsize = 0.005  # ~17 arcmin per pixel: small FoV, very mild w-effect

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-2.0, 2.0, size=n_rows)
    freq = np.array([1.4e9])

    image = rng.standard_normal((1, n_l, n_m))

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps, hermitian=hermitian)
    # The fixture must give the fold something to do, or the hermitian=True leg
    # is the hermitian=False leg under another name.
    assert 0 < int((uvw[:, 2] < 0).sum()) < n_rows
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy=w_strategy))
    vis_ref = _reference_forward(image.astype(np.complex128), uvw, freq, pixsize, pixsize)

    err = np.linalg.norm(vis_jax - vis_ref) / np.linalg.norm(vis_ref)
    assert err < DFT_TOL_FACTOR * eps, (
        f"relative error {err:.3e} exceeds {DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e} "
        f"(hermitian={hermitian})"
    )


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_forward_matches_dft_off_zenith(eps: float) -> None:
    """Tilted array so ``w`` and the n-1 phase actually do work.

    Complex image, so ``hermitian=False`` is passed explicitly (issue #17: the
    conjugate-symmetry fold holds for a real sky only, and a folded plan refuses
    a complex image rather than returning a silently wrong answer). Pinning it
    here keeps this test meaning the same thing whichever way the default is
    set, and keeps a full-strength complex-image forward parity check in the
    suite now that the zenith test above runs on a real image.
    """
    rng = np.random.default_rng(7)
    n_l = n_m = 32
    n_rows = 48
    pixsize = 0.01  # ~34 arcmin/pixel

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-30.0, 30.0, size=n_rows)
    freq = np.array([1.0e9])

    image = rng.standard_normal((1, n_l, n_m)) + 1j * rng.standard_normal((1, n_l, n_m))

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps, hermitian=False)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image)))
    vis_ref = _reference_forward(image, uvw, freq, pixsize, pixsize)

    err = np.linalg.norm(vis_jax - vis_ref) / np.linalg.norm(vis_ref)
    assert err < DFT_TOL_FACTOR * eps, (
        f"relative error {err:.3e} exceeds {DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e}"
    )


# ---------------------------------------------------------------------------
# Multi-channel parity (issue #23; partial cover for issue #14)
# ---------------------------------------------------------------------------
#
# Everything above this point is single-channel, and until issue #23 that was
# a thin but tolerable gap: the plan stored one fully-formed coordinate array
# per channel, so "channel c" was a slice index and little else. Issue #23
# replaced those arrays with a per-channel *scalar*, ``inv_lambda[c] =
# freq[c] / c``, that the operators multiply through inside the channel loop.
# That is new machinery on the per-channel axis, and a single-channel test
# cannot distinguish ``inv_lambda[c]`` from ``inv_lambda[0]`` -- nor a channel
# loop that scans the image and the scalar out of step, nor a transposed
# output -- because with one channel every one of those bugs is the identity.
#
# The reference is the exact DFT at the usual ``2 * eps`` contract, not another
# jax-nufft call: comparing strategies against each other cannot catch this
# either, since all four call the same per-channel helper and would inherit the
# same wrong scalar (the shared-mode failure AGENTS.md sec 6 records from issue
# #16). Issue #14 tracks fuller per-channel and multi-channel coverage; this is
# the slice of it that issue #23 makes load-bearing.
#
# Three channels spanning a factor of four in frequency, and an image that
# differs per channel, so the wrong scalar or the wrong slice is a gross error
# rather than a tolerance argument.
_MULTI_CHAN_FREQ = np.array([0.7e9, 1.4e9, 2.8e9])


@functools.cache
def _multi_channel_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """The shared multi-channel fixture: ``(uvw, image, vis, freq, pixsize)``.

    Cached so every parametrisation below measures the same inputs, and so the
    two row-loop DFT references (which are epsilon- and strategy-independent)
    are computed against identical data each time.
    """
    rng = np.random.default_rng(20231)
    n_l = n_m = 16
    n_rows = 24
    pixsize = 0.006

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-60.0, 60.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-60.0, 60.0, size=n_rows)
    # Real w content, off zenith, so the w-plane machinery (whose plane
    # spacing and window placement are per-channel) actually does work.
    uvw[:, 2] = rng.uniform(-25.0, 25.0, size=n_rows) + 40.0

    freq = _MULTI_CHAN_FREQ
    n_chan = freq.shape[0]
    image = rng.standard_normal((n_chan, n_l, n_m)) + 1j * rng.standard_normal((n_chan, n_l, n_m))
    vis = (
        rng.standard_normal((n_rows, n_chan)) + 1j * rng.standard_normal((n_rows, n_chan))
    ).astype(np.complex128)
    return uvw, image, vis, freq, pixsize


@pytest.mark.parametrize("channel_strategy", ["scan", "vmap"])
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_multi_channel_matches_dft_forward_and_adjoint(
    w_strategy: str, channel_strategy: str
) -> None:
    """Three distinct frequencies, forward and adjoint, against the exact DFT.

    Both channel strategies are covered because they are two different ways of
    walking the same new per-channel scalar -- ``scan`` carries
    ``plan.inv_lambda`` as a scan input alongside the image, ``vmap`` maps over
    it -- and a mismatch between the two axes only shows up with more than one
    channel. Likewise both strategy families: the windowed helpers take the
    sort_perm gather of ``uvw_m`` (shared across channels) and this channel's
    scalar as separate arguments, which is a different composition from the
    dense path's.

    ``hermitian=False`` (issue #17) because this image is complex, which the
    conjugate-symmetry fold does not apply to; passed explicitly so the test
    means the same thing whichever way ``make_plan``'s default is set. The
    multi-channel case *with* the fold is
    ``tests/test_hermitian.py::test_multi_channel_dft_parity_under_the_fold``,
    which mirrors these eight strategy combinations on a real image.
    """
    eps = 1e-6
    uvw, image, vis, freq, pixsize = _multi_channel_case()
    n_chan, n_l, n_m = image.shape

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps, hermitian=False)
    assert plan.n_chan == n_chan

    vis_jax = np.asarray(
        dirty2vis(
            plan,
            jnp.asarray(image),
            w_strategy=w_strategy,
            channel_strategy=channel_strategy,
        )
    )
    dirty_jax = np.asarray(
        vis2dirty(
            plan,
            jnp.asarray(vis),
            w_strategy=w_strategy,
            channel_strategy=channel_strategy,
        )
    )

    vis_ref = _reference_forward(image, uvw, freq, pixsize, pixsize)
    dirty_ref = _reference_adjoint(vis, uvw, freq, (n_l, n_m), pixsize, pixsize)

    fwd_err = np.linalg.norm(vis_jax - vis_ref) / np.linalg.norm(vis_ref)
    adj_err = np.linalg.norm(dirty_jax - dirty_ref) / np.linalg.norm(dirty_ref)
    assert fwd_err < DFT_TOL_FACTOR * eps, (
        f"forward relative error {fwd_err:.3e} exceeds "
        f"{DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e} "
        f"({w_strategy}, channel_strategy={channel_strategy})"
    )
    assert adj_err < DFT_TOL_FACTOR * eps, (
        f"adjoint relative error {adj_err:.3e} exceeds "
        f"{DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e} "
        f"({w_strategy}, channel_strategy={channel_strategy})"
    )

    # Per channel as well as in aggregate: a norm over all three channels is
    # dominated by the loudest, so a single mis-scaled channel could hide
    # inside an otherwise-good total.
    for c in range(n_chan):
        c_err = np.linalg.norm(vis_jax[:, c] - vis_ref[:, c]) / np.linalg.norm(vis_ref[:, c])
        assert c_err < DFT_TOL_FACTOR * eps, (
            f"forward channel {c} (freq={freq[c]:.3g} Hz) relative error {c_err:.3e} "
            f"exceeds {DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e}"
        )
        c_err = np.linalg.norm(dirty_jax[c] - dirty_ref[c]) / np.linalg.norm(dirty_ref[c])
        assert c_err < DFT_TOL_FACTOR * eps, (
            f"adjoint channel {c} (freq={freq[c]:.3g} Hz) relative error {c_err:.3e} "
            f"exceeds {DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e}"
        )


# Every decade from 1e-3 to 1e-12, not just a subsample: the old rule
# ``ceil(-log10(eps)*2/pi) + 2`` collapsed the adjacent pairs 1e-5/1e-6,
# 1e-9/1e-10 and 1e-11/1e-12 onto the same ``W`` (see kernel.py, kernel_params
# docstring), and a list that skips one member of each pair (as the previous
# ``[1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]`` did) can't exercise the exact
# regression it exists to catch.
_TRACKING_EPS = [10.0**-k for k in range(3, 13)]


@pytest.mark.parametrize("hermitian", [False, True])
def test_accuracy_tracks_epsilon(hermitian: bool) -> None:
    """Error must follow ``epsilon`` down, not plateau (issue #9).

    A single test rather than a parametrised one because the interesting
    assertion is *across* epsilon values: with the old width rule
    ``W = ceil(-log10(eps) * 2/pi) + 2``, eps=1e-5 and eps=1e-6 both mapped to
    W=6 and produced the *same* error, so asking for a tighter epsilon bought
    nothing. ``_TRACKING_EPS`` covers every decade from 1e-3 to 1e-12 so this
    exercises the exact adjacent pairs that used to collapse (1e-5/1e-6,
    1e-9/1e-10, 1e-11/1e-12), not just a subsample that happens to include one
    member of each. The per-epsilon bound is the same ``2 * eps`` contract as
    the rest of the file.

    The real plateau guard is the width check at the end: ``kernel_params``
    must hand back a strictly wider kernel for every adjacent decade, since
    that -- not the measured error -- is what the old rule actually violated.
    A *measured*-error monotonicity assertion was tried first and dropped: a
    tighter kernel is not guaranteed to measure a smaller error on every
    fixture (a tighter approximation can reorder floating-point cancellation
    and land marginally worse), so asserting it risks becoming exactly the
    kind of assertion this repo's rules say not to weaken to make green.
    Asserting the width step directly tests the thing that must not
    regress and nothing else.

    MWA_compact off30 (128 px, 600 rows) is the smallest review fixture with
    real w-content, and the row-loop DFT reference over 128^2 pixels costs a
    fraction of a second, so this stays in the default (non-``--runslow``) run.

    Parametrised over issue #17's Hermitian fold, which cuts ``n_w`` on this
    fixture from 17 to 12 at eps=1e-6. The plateau this test exists to catch is
    a *kernel-width* one, and the fold changes the plane count without changing
    the width -- so if the fold ever bought its planes by under-resolving the
    w-direction rather than by shrinking the range it covers, the folded leg is
    where it would show, as an error that stops tracking epsilon while the
    unfolded leg keeps tracking it. The image is real, which the fold requires;
    the visibilities are complex, which the adjoint always allows.
    """
    tel = MWA_COMPACT
    uvw = synthetic_uvw(tel, 30.0, seed=0)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    shape = (tel.n_pix, tel.n_pix)
    rng = np.random.default_rng(7)
    image = rng.standard_normal((1, *shape))
    vis = (rng.standard_normal((tel.n_rows, 1)) + 1j * rng.standard_normal((tel.n_rows, 1))).astype(
        np.complex128
    )

    # References are epsilon-independent, so compute each exactly once.
    vis_ref = _reference_forward(image.astype(np.complex128), uvw, freq, pix, pix)
    dirty_ref = _reference_adjoint(vis, uvw, freq, shape, pix, pix)

    for eps in _TRACKING_EPS:
        plan = make_plan(uvw, freq, shape, pix, pix, eps, hermitian=hermitian)
        vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image)))
        dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis)))
        e_f = float(np.linalg.norm(vis_jax - vis_ref) / np.linalg.norm(vis_ref))
        e_a = float(np.linalg.norm(dirty_jax - dirty_ref) / np.linalg.norm(dirty_ref))
        assert e_f < DFT_TOL_FACTOR * eps, (
            f"forward eps={eps:g}: relative error {e_f:.3e} is {e_f / eps:.2f}x eps "
            f"(W={plan.w_kernel_width}, n_w={plan.n_w}, hermitian={hermitian})"
        )
        assert e_a < DFT_TOL_FACTOR * eps, (
            f"adjoint eps={eps:g}: relative error {e_a:.3e} is {e_a / eps:.2f}x eps "
            f"(W={plan.w_kernel_width}, n_w={plan.n_w}, hermitian={hermitian})"
        )

    # The actual plateau guard: kernel_params()[0] (the width W) must step up
    # by exactly one for every adjacent decade in _TRACKING_EPS. This is what
    # the old rule violated (three collapsed pairs across eps=1e-3..1e-12);
    # it is cheap (no plan / no NUFFT call) and, unlike a measured-error
    # comparison, deterministic.
    widths = [kernel_params(eps)[0] for eps in _TRACKING_EPS]
    width_deltas = [b - a for a, b in itertools.pairwise(widths)]
    assert width_deltas == [1] * len(width_deltas), (
        f"kernel width must increase by exactly one per decade from "
        f"eps={_TRACKING_EPS[0]:g} to eps={_TRACKING_EPS[-1]:g}; got widths={widths}"
    )


def test_forward_real_image_promotes_to_complex() -> None:
    """Real input should be auto-promoted to complex."""
    rng = np.random.default_rng(0)
    pixsize = 0.005
    n_l = n_m = 16
    uvw = rng.uniform(-50, 50, size=(20, 3))
    freq = np.array([1e9])
    image = rng.standard_normal((1, n_l, n_m))  # real
    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6)
    vis = dirty2vis(plan, jnp.asarray(image))
    assert jnp.iscomplexobj(vis)
    assert vis.shape == (20, 1)


def test_forward_2d_image_broadcasts_across_channels() -> None:
    """2D image should be broadcast across all channels."""
    rng = np.random.default_rng(1)
    pixsize = 0.005
    n_l = n_m = 8
    uvw = rng.uniform(-30, 30, size=(10, 3))
    freq = np.array([1e9, 1.5e9])
    image_2d = rng.standard_normal((n_l, n_m))
    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6)
    vis_2d = dirty2vis(plan, jnp.asarray(image_2d))
    image_3d = np.broadcast_to(image_2d, (2, n_l, n_m))
    vis_3d = dirty2vis(plan, jnp.asarray(image_3d))
    np.testing.assert_allclose(np.asarray(vis_2d), np.asarray(vis_3d), rtol=1e-12)


# ---------------------------------------------------------------------------
# Image-geometry coverage: odd, non-square, anisotropic (issue #14)
# ---------------------------------------------------------------------------
#
# Until issue #20 every fixture in this repository was square (``n_l == n_m``)
# with isotropic pixels (``pixsize_l == pixsize_m``). On such a plan the whole
# image grid is transpose-symmetric *bit for bit* -- ``max|nm1 - nm1.T| ==
# 0.0`` -- so every quantity indexed on that grid (the ``n - 1`` plane phase,
# ``phi_hat_n``, the ``w0`` screen, the ``1/n`` diagonal, the NUFFT output
# shape) is free to confuse its two axes and no test can see it. Issue #20
# measured the consequence on the one diagonal it added: transposing it is a
# one-token change that passed the entire 1255-test suite and gives 1.022
# relative error against ducc0 on an anisotropic plan.
#
# This block is the general case rather than one diagonal, and it is here (in
# the DFT file) rather than in the ducc0 file for one hard reason: ducc0's
# public API refuses an odd image size outright --
# ``ducc0.wgridder.dirty2vis(dirty=np.zeros((33, 33)), ...)`` raises
# ``RuntimeError: ... Assertion failure / nx_dirty must be even`` on 0.41.0, and
# so does ``vis2dirty(npix_x=33, ...)``. Non-square *even* sizes it accepts
# fine. So odd sizes have no oracle but the exact DFT, which is in any case the
# definition of the answer and the tighter bound (``2 * eps`` rather than
# ``3 * eps``).
#
# The cells, and what each one can falsify that the others cannot:
#
#   * ``(32, 32) / (0.01, 0.01)`` -- the control. Square and isotropic, i.e.
#     the blind geometry every pre-#20 fixture had. It falsifies no axis
#     confusion at all and is here so a failure that shows up on *every* cell
#     is legible as "the operator is broken" rather than "the geometry is".
#   * ``(33, 33)`` -- odd, square, isotropic. Exercises the ``i - n // 2``
#     grid convention where the centre pixel is exact and the grid is not
#     symmetric about zero (``-16 .. +16`` rather than ``-16 .. +15``), and the
#     FFT sizes FINUFFT picks from an odd extent. ducc0 cannot check this at
#     all.
#   * ``(32, 48)`` and ``(48, 32)`` -- non-square, isotropic, and each other's
#     transpose. A grid-indexed array that swaps its axes cannot even broadcast
#     here, so these catch an axis swap as a *shape* error, which is a better
#     failure than a numeric one. Both orders because a swap that happens to be
#     compensated in one order shows up in the other.
#   * ``(32, 32) / (0.012, 0.006)`` -- **square and anisotropic: the sharp
#     one.** A transposed grid-indexed array is still shape-valid here, so
#     nothing but the number can catch it. This is the geometry issue #20
#     measured the 1.022 error on, and it is the cell that makes this block
#     more than a shape check. It is not in issue #14's stated list, which
#     jumps straight from square-isotropic to non-square; without it the whole
#     parametrisation would hold "the transpose is shape-valid" constant at
#     False for every asymmetric cell.
#   * ``(31, 45) / (0.01, 0.007)`` -- odd *and* non-square *and* anisotropic,
#     with two odd extents that share no factor. The compound case.
#   * ``(32, 48) / (0.012, 0.006)`` -- even, non-square and anisotropic at
#     once, with ``pixsize_l * n_l`` deliberately *not* equal to
#     ``pixsize_m * n_m`` so the physical field of view is asymmetric too, not
#     just the sampling.
#
# One epsilon (1e-6) per cell, and three ``w_strategy`` values -- not one.
#
# The first version of this block pinned ``w_strategy="dense_scan"`` on both
# operators and called that "the default". It is not the default: the default
# is ``"auto"`` (issue #46), and on *every one* of the seven geometries above
# ``_auto_w_strategy`` resolves the adjoint to ``windowed_scan`` on the CPU
# backend (the forward it does resolve to ``dense_scan``). So the block steered
# around the one adjoint path a user actually gets, and the consequence was
# measurable: with ``dense_scan`` pinned, both
#
#     nufft1((plan.n_m, plan.n_l), ...)          in _channel_adjoint_windowed
#     dirty_init = jnp.zeros((plan.n_m, plan.n_l), ...)   ditto
#
# survived the *entire* suite, while the identical mutation on the dense
# adjoint's ``nufft1`` was caught by ``[nonsquare_32x48]``. The site the tests
# reached was checked and the site they did not reach was not.
#
# Hence ``_GEOMETRY_W_STRATEGIES``, and why it is those three:
#
#   * ``"auto"`` is what ships, so at least one leg per geometry has to be the
#     operator a caller who passes no ``w_strategy`` actually runs.
#   * ``"dense_scan"`` and ``"windowed_scan"`` pin the two *helper families*
#     explicitly, so this block does not become hostage to how the heuristic
#     resolves. That matters concretely: ``_auto_w_strategy`` dispatches on
#     ``jax.devices()[0].platform`` and the GPU branch never picks a ``_scan``
#     variant at all, so on GPU the ``"auto"`` leg alone would cover neither of
#     the two scan-only ``jnp.zeros((n_l, n_m))`` accumulators. The two
#     explicit legs cover all four operator-internal grid-shape sites
#     (``wgridder.py`` lines 1190/1214 dense, 1516/1537 windowed) on every
#     platform. For the same reason this block does *not* assert what
#     ``"auto"`` resolves to -- that is the heuristic's business and it is
#     platform-dependent; what is asserted is that both families are reached
#     whatever it picks.
#
# The two ``*_vmap`` variants are deliberately omitted, and that omission is
# the one axis this block still holds constant. It is safe by inspection
# rather than by hope: the four grid-shape literals above live in the shared
# prologue of ``_channel_adjoint`` / ``_channel_adjoint_windowed`` or in their
# ``scan`` accumulators, and the ``vmap`` branches allocate no image-shaped
# array of their own (they ``jnp.sum`` a mapped stack). So a ``*_vmap`` leg
# would re-execute the same grid indexing at 7 more cells apiece and could not
# falsify anything the ``*_scan`` legs do not. Scan-vs-vmap equivalence itself
# is crossed with everything in ``tests/test_strategies_equivalent.py``.
#
# 300 rows with w in +/-40 m so the w-plane machinery is genuinely engaged
# rather than collapsing onto the constant-w fast path.
_GEOMETRY_EPS = 1e-6
_GEOMETRY_W_STRATEGIES = ["auto", "dense_scan", "windowed_scan"]
_GEOMETRY_CASES = [
    pytest.param((32, 32), (0.01, 0.01), id="even_square_isotropic_control"),
    pytest.param((33, 33), (0.01, 0.01), id="odd_square"),
    pytest.param((32, 48), (0.01, 0.01), id="nonsquare_32x48"),
    pytest.param((48, 32), (0.01, 0.01), id="nonsquare_48x32"),
    pytest.param((32, 32), (0.012, 0.006), id="square_anisotropic"),
    pytest.param((31, 45), (0.01, 0.007), id="odd_nonsquare_anisotropic"),
    pytest.param((32, 48), (0.012, 0.006), id="even_nonsquare_anisotropic"),
]


@functools.cache
def _geometry_uvw_freq() -> tuple[np.ndarray, np.ndarray]:
    """The one (uvw, freq) every geometry cell shares.

    Shared deliberately: the only thing that varies across the cells above is
    the image geometry, so holding the baselines fixed makes a per-cell failure
    attributable to the geometry and nothing else.
    """
    rng = np.random.default_rng(4141)
    n_rows = 300
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-120.0, 120.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-120.0, 120.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-40.0, 40.0, size=n_rows)
    return uvw, np.array([1.0e9])


@functools.cache
def _geometry_data(
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """``(image, vis)`` for one geometry: complex image, complex visibilities."""
    n_l, n_m = image_shape
    uvw, _ = _geometry_uvw_freq()
    n_rows = uvw.shape[0]
    rng = np.random.default_rng(909 + n_l * 1000 + n_m)
    image = rng.standard_normal((1, n_l, n_m)) + 1j * rng.standard_normal((1, n_l, n_m))
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )
    return image, vis


@functools.cache
def _geometry_references(
    image_shape: tuple[int, int], pixsize: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """The exact-DFT ``(vis_ref, dirty_ref)`` for one geometry.

    Cached because the references are row-loop DFTs and depend only on the
    geometry -- not on ``w_strategy``, which the test is parametrised over.
    Without the cache each geometry would recompute both references once per
    strategy leg for an identical answer.
    """
    uvw, freq = _geometry_uvw_freq()
    image, vis = _geometry_data(image_shape)
    pixsize_l, pixsize_m = pixsize
    return (
        _reference_forward(image, uvw, freq, pixsize_l, pixsize_m),
        _reference_adjoint(vis, uvw, freq, image_shape, pixsize_l, pixsize_m),
    )


@pytest.mark.parametrize("w_strategy", _GEOMETRY_W_STRATEGIES)
@pytest.mark.parametrize("image_shape, pixsize", _GEOMETRY_CASES)
def test_geometry_matches_dft_forward_and_adjoint(
    image_shape: tuple[int, int], pixsize: tuple[float, float], w_strategy: str
) -> None:
    """Odd / non-square / anisotropic image grids against the exact DFT.

    Both operators, because they index the grid independently: the forward
    multiplies the image by ``exp(2 pi i w_k (n - 1 + nshift)) / phi_hat_n`` and
    hands FINUFFT an ``(n_l, n_m)`` array, while the adjoint asks ``nufft1`` for
    an ``(n_l, n_m)`` output, accumulates into a ``jnp.zeros((n_l, n_m))``, and
    then applies the same screen plus the ``1/n`` diagonal. A swap on one side
    is not a swap on the other, and -- see ``_GEOMETRY_W_STRATEGIES`` -- a swap
    in the *windowed* adjoint is not a swap in the dense one either: those are
    two separate ``nufft1`` call sites and two separate accumulators.

    ``hermitian=False`` with a complex image, so the forward is exercised at
    full strength (a folded plan refuses a complex image). The fold acts on the
    w axis and the row order, which is orthogonal to the l/m axis assignment
    this block is about, and it is crossed with everything else in
    ``tests/test_hermitian.py``.
    """
    eps = _GEOMETRY_EPS
    n_l, n_m = image_shape
    pixsize_l, pixsize_m = pixsize
    uvw, freq = _geometry_uvw_freq()
    image, vis = _geometry_data(image_shape)

    plan = make_plan(uvw, freq, image_shape, pixsize_l, pixsize_m, eps, hermitian=False)
    assert (plan.n_l, plan.n_m) == image_shape
    # The plan must be genuinely w-dependent, or the w-plane loop -- which is
    # where every grid-indexed phase is applied -- collapses to a single plane
    # and the geometry is being checked against a much weaker operator.
    assert not plan.is_constant_w
    assert plan.n_w > 1

    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy=w_strategy))
    dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy=w_strategy))
    assert vis_jax.shape == (uvw.shape[0], 1)
    assert dirty_jax.shape == (1, n_l, n_m)

    vis_ref, dirty_ref = _geometry_references(image_shape, pixsize)

    fwd_err = np.linalg.norm(vis_jax - vis_ref) / np.linalg.norm(vis_ref)
    adj_err = np.linalg.norm(dirty_jax - dirty_ref) / np.linalg.norm(dirty_ref)
    assert fwd_err < DFT_TOL_FACTOR * eps, (
        f"forward shape={image_shape} pixsize=({pixsize_l:g}, {pixsize_m:g}) "
        f"w_strategy={w_strategy}: relative error {fwd_err:.3e} exceeds "
        f"{DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e}"
    )
    assert adj_err < DFT_TOL_FACTOR * eps, (
        f"adjoint shape={image_shape} pixsize=({pixsize_l:g}, {pixsize_m:g}) "
        f"w_strategy={w_strategy}: relative error {adj_err:.3e} exceeds "
        f"{DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e}"
    )


def test_the_geometry_cases_can_actually_falsify_an_axis_swap() -> None:
    """The parametrisation above must not hold "symmetric grid" constant.

    Every cell in ``_GEOMETRY_CASES`` runs the same two operator calls against
    the same reference, so the only thing that makes the block worth its run
    time is that the *set* of geometries breaks the symmetries a square
    isotropic grid has. That is a property of the parameter list, not of any
    one test, so it is asserted here rather than left as a comment nobody
    re-checks when a cell is added or dropped.

    Three separate statements, because they fail an axis swap in three
    different ways and the block needs all three:

    1. at least one **odd** extent -- the grid convention and the FFT sizes,
       and the only regime with no ducc0 oracle at all;
    2. at least one **non-square** shape, where a transposed grid-indexed array
       is a shape error rather than a wrong number, *and* both orders of one
       such pair;
    3. at least one **square anisotropic** cell, where a transposed
       grid-indexed array is still shape-valid and only the number can catch
       it. This is the one issue #20 measured a 1.022 relative error on, and
       the one whose absence would leave the block a pure shape check.

    The last one is checked twice over: the parameter list has to contain such
    a cell, and the plan built from it has to have a grid that really is not
    transpose-symmetric. ``plan.n_minus_1`` is bit-for-bit equal to its own
    transpose on every square isotropic fixture in this repository, which is
    exactly why a positive measurement is worth making here.
    """
    shapes = [tuple(p.values[0]) for p in _GEOMETRY_CASES]
    pixsizes = [tuple(p.values[1]) for p in _GEOMETRY_CASES]

    odd = [s for s in shapes if s[0] % 2 or s[1] % 2]
    assert odd, "no odd image extent: ducc0 cannot check odd sizes, so this file must"

    nonsquare = {s for s in shapes if s[0] != s[1]}
    assert nonsquare, "no non-square shape: an axis swap would stay shape-valid everywhere"
    assert any((b, a) in nonsquare for a, b in nonsquare), (
        "non-square shapes come in one orientation only; include a transposed pair so a "
        "swap that is compensated in one order is still caught in the other"
    )

    square_aniso = [
        (s, p) for s, p in zip(shapes, pixsizes, strict=True) if s[0] == s[1] and p[0] != p[1]
    ]
    assert square_aniso, (
        "no square anisotropic cell: with only non-square asymmetry every axis swap fails as "
        "a shape error, and a grid-indexed quantity built with pixsize_l and pixsize_m "
        "exchanged -- which is shape-valid on any grid -- would go untested"
    )

    uvw, freq = _geometry_uvw_freq()
    for shape, pixsize in square_aniso:
        plan = make_plan(uvw, freq, shape, pixsize[0], pixsize[1], _GEOMETRY_EPS, hermitian=False)
        nm1 = np.asarray(plan.n_minus_1)
        asymmetry = float(np.max(np.abs(nm1 - nm1.T)))
        assert asymmetry > 0.0, (
            f"shape={shape} pixsize={pixsize}: plan.n_minus_1 equals its own transpose "
            f"(max|nm1 - nm1.T| = {asymmetry:g}), so a diagonal indexed [m, l] instead of "
            "[l, m] is literally the same array and this cell gates nothing"
        )

    # And the control cell must be blind, or "it passes on the control too" is
    # not the diagnostic this block relies on it being.
    control = [
        (s, p) for s, p in zip(shapes, pixsizes, strict=True) if s[0] == s[1] and p[0] == p[1]
    ]
    assert control, "no square isotropic control cell to attribute a universal failure against"


def test_the_geometry_block_reaches_both_w_strategy_families() -> None:
    """The geometry cells must not all run the same operator path.

    This is the parametrisation-integrity guard for the *other* axis of the
    block, and it exists because that axis was silently wrong once already: the
    cells originally pinned ``w_strategy="dense_scan"`` on both operators and
    a comment called it "the default". It is not -- ``"auto"`` is -- and on
    every geometry here ``auto`` resolves the adjoint to ``windowed_scan`` on
    CPU. The dense and windowed adjoints are *different call sites*
    (``wgridder.py`` 1190/1214 and 1516/1537: two ``nufft1`` output shapes and
    two ``jnp.zeros((n_l, n_m))`` accumulators), so pinning one family left the
    other's two sites reachable by no test in the suite. Transposing either of
    them passed all 1480 tests.

    Three claims, each of which would have caught that:

    1. ``"auto"`` really is the shipped default of both operators, so the leg
       that runs it is the operator a caller who passes nothing actually gets.
       Read off the signatures rather than assumed, since the point is to
       notice if the default moves again.
    2. The list carries an explicit member of *both* families, so coverage of
       the two call-site pairs does not depend on how the heuristic resolves
       -- which is platform-dependent (``_auto_w_strategy`` dispatches on the
       backend and its GPU branch never picks a ``_scan`` variant).
    3. The explicit members are ``_scan``, not ``_vmap``: the two accumulators
       above exist only in the scan branches, so a ``*_vmap`` pair would not
       reach them.
    """
    import inspect

    for op in (dirty2vis, vis2dirty):
        default = inspect.signature(op).parameters["w_strategy"].default
        assert default == "auto", (
            f"{op.__name__}'s default w_strategy is {default!r}, not 'auto'; this block "
            "parametrises over 'auto' precisely so one leg per geometry is the operator a "
            "caller gets by passing nothing -- update _GEOMETRY_W_STRATEGIES to match"
        )
    assert "auto" in _GEOMETRY_W_STRATEGIES

    explicit = [s for s in _GEOMETRY_W_STRATEGIES if s != "auto"]
    assert any(s.startswith("dense_") for s in explicit), (
        "no explicit dense strategy: the dense adjoint's nufft1 output shape and scan "
        "accumulator would be covered only if the auto heuristic happened to pick dense"
    )
    assert any(s.startswith("windowed_") for s in explicit), (
        "no explicit windowed strategy: the windowed adjoint's nufft1 output shape and scan "
        "accumulator are a second, separate pair of call sites -- transposing either passed "
        "the entire suite while this block pinned dense_scan"
    )
    for family in ("dense", "windowed"):
        assert f"{family}_scan" in explicit, (
            f"{family}_scan missing: the image-shaped accumulator in the {family} adjoint is "
            f"allocated only on the scan branch, so {family}_vmap does not reach it"
        )
