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
from tests.conftest import MWA_COMPACT, synthetic_uvw
from tests.test_adjoint import _reference_adjoint

jax.config.update("jax_enable_x64", True)

# Accuracy contract against the exact DFT (issue #9). See the module docstring.
DFT_TOL_FACTOR = 2.0


def reference_lmn_grids(
    image_shape: tuple[int, int], pixsize_l: float, pixsize_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(l, m, n - 1)`` on the image grid, matching ``planning.make_plan``.

    Inside the unit disc this is the usual ``n - 1 = sqrt(1 - l^2 - m^2) - 1``.
    Outside it (reachable for wide-FoV fixtures such as EDA2's 120-degree
    field) we use the same analytic extension as ducc and
    :func:`jax_nufft.planning.make_plan`, ``n - 1 = -sqrt(l^2 + m^2 - 1) - 1``.
    Clipping to ``n - 1 = -1`` there instead would make the reference disagree
    with the operator under test by O(1) on the corner pixels, which has
    nothing to do with the gridding accuracy we are trying to measure.
    """
    n_l, n_m = image_shape
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    ll, mm = np.meshgrid(i * pixsize_l, j * pixsize_m, indexing="ij")
    r2 = ll * ll + mm * mm
    inside_disc = r2 <= 1.0
    inside_val = np.sqrt(np.where(inside_disc, 1.0 - r2, 0.0)) - 1.0
    outside_val = -np.sqrt(np.where(inside_disc, 0.0, r2 - 1.0)) - 1.0
    return ll, mm, np.where(inside_disc, inside_val, outside_val)


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
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_forward_matches_dft_single_channel_zenith(eps: float, w_strategy: str) -> None:
    """Tiny zenith problem: w in metres deliberately non-zero but small."""
    rng = np.random.default_rng(123)
    n_l = n_m = 16
    n_rows = 24
    pixsize = 0.005  # ~17 arcmin per pixel: small FoV, very mild w-effect

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-2.0, 2.0, size=n_rows)
    freq = np.array([1.4e9])

    image = rng.standard_normal((1, n_l, n_m)) + 1j * rng.standard_normal((1, n_l, n_m))

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy=w_strategy))
    vis_ref = _reference_forward(image, uvw, freq, pixsize, pixsize)

    err = np.linalg.norm(vis_jax - vis_ref) / np.linalg.norm(vis_ref)
    assert err < DFT_TOL_FACTOR * eps, (
        f"relative error {err:.3e} exceeds {DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e}"
    )


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_forward_matches_dft_off_zenith(eps: float) -> None:
    """Tilted array so ``w`` and the n-1 phase actually do work."""
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

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps)
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
    """
    eps = 1e-6
    uvw, image, vis, freq, pixsize = _multi_channel_case()
    n_chan, n_l, n_m = image.shape

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps)
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


def test_accuracy_tracks_epsilon() -> None:
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
        plan = make_plan(uvw, freq, shape, pix, pix, eps)
        vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image)))
        dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis)))
        e_f = float(np.linalg.norm(vis_jax - vis_ref) / np.linalg.norm(vis_ref))
        e_a = float(np.linalg.norm(dirty_jax - dirty_ref) / np.linalg.norm(dirty_ref))
        assert e_f < DFT_TOL_FACTOR * eps, (
            f"forward eps={eps:g}: relative error {e_f:.3e} is {e_f / eps:.2f}x eps "
            f"(W={plan.w_kernel_width}, n_w={plan.n_w})"
        )
        assert e_a < DFT_TOL_FACTOR * eps, (
            f"adjoint eps={eps:g}: relative error {e_a:.3e} is {e_a / eps:.2f}x eps "
            f"(W={plan.w_kernel_width}, n_w={plan.n_w})"
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
