"""Parity tests against ``ducc0.wgridder``.

ducc0's ``dirty2vis`` and ``vis2dirty`` are taken as ground truth. Because our
operators do not divide by ``n`` on the forward but do divide by ``n`` on the
adjoint, we configure ducc consistently:

  * forward parity: ``ducc0.wgridder.dirty2vis(divide_by_n=False, ...)``
  * adjoint parity: ``ducc0.wgridder.vis2dirty(divide_by_n=True, ...)``

The acceptance threshold is ``3 * epsilon`` (issue #9). ducc0 itself lands at
roughly ``0.1 * epsilon`` against the exact DFT, and jax-nufft is held to
``2 * epsilon`` there (``tests/test_against_dft.py``), so the gap between the
two implementations is dominated by jax-nufft's own error and a factor of
three is all the headroom the contract needs. The previous ``20 * epsilon``
threshold was chosen when the w-kernel width rule under-provisioned by up to
three cells and would not have caught it.

Image sizes / row counts are reduced from the spec values so that the test
matrix runs in single-digit seconds locally; the algorithmic regime (FoV,
baseline length distribution, off-zenith pointing) is preserved.
"""

from __future__ import annotations

import functools

import ducc0.wgridder
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from tests.conftest import Telescope, synthetic_uvw
from tests.test_adjoint import DFT_TOL_FACTOR, _reference_adjoint

# ducc0 parity contract (issue #9). See the module docstring.
DUCC_TOL_FACTOR = 3.0


@pytest.mark.parametrize("w_strategy", ["dense_scan", "windowed_scan"])
@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_forward_parity(
    short_telescope_pointing: tuple[Telescope, float], eps: float, w_strategy: str
) -> None:
    tel, zen_deg = short_telescope_pointing
    uvw = synthetic_uvw(tel, zen_deg, seed=0)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    rng = np.random.default_rng(7)
    image = rng.standard_normal((tel.n_pix, tel.n_pix))

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy=w_strategy))

    vis_ducc = ducc0.wgridder.dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=image,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )

    err = np.linalg.norm(vis_jax - vis_ducc) / np.linalg.norm(vis_ducc)
    # ducc and our wgridder both target `epsilon` independently, so the gap
    # between them is bounded by the sum of their individual errors: ducc is
    # ~0.1*eps and we are contracted to 2*eps, hence 3*eps.
    assert err < DUCC_TOL_FACTOR * eps, (
        f"{tel.name} zen={zen_deg} eps={eps:g} {w_strategy}: relative error {err:.3e}"
    )


@pytest.mark.parametrize("w_strategy", ["dense_scan", "windowed_scan"])
@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_adjoint_parity(
    short_telescope_pointing: tuple[Telescope, float], eps: float, w_strategy: str
) -> None:
    tel, zen_deg = short_telescope_pointing
    uvw = synthetic_uvw(tel, zen_deg, seed=1)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    rng = np.random.default_rng(11)
    vis_np = (
        rng.standard_normal((tel.n_rows, 1)) + 1j * rng.standard_normal((tel.n_rows, 1))
    ).astype(np.complex128)

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis_np), w_strategy=w_strategy))[0]

    dirty_ducc = ducc0.wgridder.vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis_np,
        npix_x=tel.n_pix,
        npix_y=tel.n_pix,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=True,
        nthreads=1,
    )

    err = np.linalg.norm(dirty_jax - dirty_ducc) / np.linalg.norm(dirty_ducc)
    assert err < DUCC_TOL_FACTOR * eps, (
        f"{tel.name} zen={zen_deg} eps={eps:g} {w_strategy}: relative error {err:.3e}"
    )


@pytest.mark.parametrize("eps", [1e-6])
def test_forward_parity_with_weights(
    short_telescope_pointing: tuple[Telescope, float], eps: float
) -> None:
    """Sanity: our adjoint with weights matches ducc's adjoint with `wgt`."""
    tel, zen_deg = short_telescope_pointing
    uvw = synthetic_uvw(tel, zen_deg, seed=2)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    rng = np.random.default_rng(15)
    vis_np = (
        rng.standard_normal((tel.n_rows, 1)) + 1j * rng.standard_normal((tel.n_rows, 1))
    ).astype(np.complex128)
    wgt = rng.uniform(0.1, 1.0, size=(tel.n_rows, 1)).astype(np.float64)

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis_np), weights=jnp.asarray(wgt)))[0]

    dirty_ducc = ducc0.wgridder.vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis_np,
        wgt=wgt,
        npix_x=tel.n_pix,
        npix_y=tel.n_pix,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=True,
        nthreads=1,
    )

    err = np.linalg.norm(dirty_jax - dirty_ducc) / np.linalg.norm(dirty_ducc)
    assert err < DUCC_TOL_FACTOR * eps


@pytest.mark.parametrize("eps", [1e-6])
def test_forward_parity_long(long_telescope_pointing: tuple[Telescope, float], eps: float) -> None:
    """Slow parity tests for MWA_extended / MeerKAT (skipped without --runslow)."""
    tel, zen_deg = long_telescope_pointing
    uvw = synthetic_uvw(tel, zen_deg, seed=4)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    rng = np.random.default_rng(21)
    image = rng.standard_normal((tel.n_pix, tel.n_pix))

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image)))
    vis_ducc = ducc0.wgridder.dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=image,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )
    err = np.linalg.norm(vis_jax - vis_ducc) / np.linalg.norm(vis_ducc)
    assert err < DUCC_TOL_FACTOR * eps


@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_constant_w_ducc_parity(op: str, w_strategy: str) -> None:
    """v0.1.2 fast path: coplanar (w == 0 everywhere) data must match ducc
    within ``DUCC_TOL_FACTOR * eps`` for every w_strategy. ``plan.n_w == 1`` confirms the
    specialisation engaged."""
    eps = 1e-6
    tel = Telescope(
        name="MWA_compact_coplanar",
        freq_hz=150e6,
        n_rows=400,
        sigma_uv_m=50.0,
        max_baseline_m=200.0,
        n_pix=128,
        fov_rad=np.radians(20.0),
    )
    uvw = synthetic_uvw(tel, 0.0, seed=42)  # zenith pointing
    uvw[:, 2] = 0.0  # force exactly coplanar so the v0.1.2 fast path engages
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    # Acceptance signal that the fast path engaged.
    assert plan.is_constant_w
    assert plan.n_w == 1
    assert plan.w_extent == 0.0

    if op == "dirty2vis":
        rng = np.random.default_rng(7)
        image = rng.standard_normal((tel.n_pix, tel.n_pix))
        vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy=w_strategy))
        vis_ducc = ducc0.wgridder.dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=image,
            pixsize_x=pix,
            pixsize_y=pix,
            epsilon=eps,
            do_wgridding=True,
            divide_by_n=False,
            nthreads=1,
        )
        err = np.linalg.norm(vis_jax - vis_ducc) / np.linalg.norm(vis_ducc)
    else:
        rng = np.random.default_rng(11)
        vis_np = (
            rng.standard_normal((tel.n_rows, 1)) + 1j * rng.standard_normal((tel.n_rows, 1))
        ).astype(np.complex128)
        dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis_np), w_strategy=w_strategy))[0]
        dirty_ducc = ducc0.wgridder.vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=vis_np,
            npix_x=tel.n_pix,
            npix_y=tel.n_pix,
            pixsize_x=pix,
            pixsize_y=pix,
            epsilon=eps,
            do_wgridding=True,
            divide_by_n=True,
            nthreads=1,
        )
        err = np.linalg.norm(dirty_jax - dirty_ducc) / np.linalg.norm(dirty_ducc)

    assert err < DUCC_TOL_FACTOR * eps, (
        f"constant-w fast path n_w={plan.n_w} {op} {w_strategy}: "
        f"relative error {err:.3e} exceeds {DUCC_TOL_FACTOR * eps:.3e}"
    )


@pytest.mark.parametrize("eps", [1e-6])
def test_multichannel_forward_parity(eps: float) -> None:
    """Multi-channel: same image broadcast across 4 channels covering ~10% bandwidth."""
    tel = Telescope(
        name="MWA_compact_multichan",
        freq_hz=150e6,
        n_rows=400,
        sigma_uv_m=50.0,
        max_baseline_m=200.0,
        n_pix=128,
        fov_rad=np.radians(20.0),
    )
    uvw = synthetic_uvw(tel, 30.0, seed=99)
    freq = np.linspace(0.95, 1.05, 4) * tel.freq_hz
    pix = tel.pixsize
    rng = np.random.default_rng(50)
    image = rng.standard_normal((tel.n_pix, tel.n_pix))

    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, eps)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image)))

    vis_ducc = ducc0.wgridder.dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=image,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )

    err = np.linalg.norm(vis_jax - vis_ducc) / np.linalg.norm(vis_ducc)
    assert err < DUCC_TOL_FACTOR * eps


# ---------------------------------------------------------------------------
# Multi-channel coverage against ducc0 (issue #14)
# ---------------------------------------------------------------------------
#
# What was missing here, precisely. ``test_multichannel_forward_parity`` above
# is the only multi-channel cell in this file: it is a *forward* test, it
# broadcasts one 2-D image across all four channels, and it passes no weights.
# So three things went unchecked against the second implementation:
#
#   1. the **adjoint** with more than one channel, and with per-channel
#      ``weights`` (ducc's ``wgt``, an ``(n_rows, n_chan)`` array) -- the
#      forward has no weights argument at all, so the weight path had exactly
#      one ducc0 cell in the suite (``test_forward_parity_with_weights``,
#      single channel despite its name);
#   2. a forward whose image *content differs per channel*, which a broadcast
#      2-D image cannot express -- with one plane shared by every channel, an
#      operator that read image plane 0 for every channel would be the
#      identity;
#   3. either of those on a grid that is not transpose-symmetric.
#
# Both tests below therefore run on a **non-square, anisotropic** image. That
# costs nothing (ducc0 accepts any even ``npix_x``/``npix_y`` and any
# ``pixsize_x``/``pixsize_y``; only *odd* sizes are refused -- see
# ``tests/test_against_dft.py``'s geometry block) and it means the channel-axis
# statements below are made on a geometry where an l/m axis swap is also
# visible, rather than on the square isotropic grid that hides one bit for bit.
# The two use transposed shapes, ``(64, 48)`` and ``(48, 64)``, for the same
# reason the DFT block does.
_MULTICHAN_TEL = Telescope(
    name="MWA_compact_multichan_wide",
    freq_hz=150e6,
    n_rows=300,
    sigma_uv_m=50.0,
    max_baseline_m=200.0,
    n_pix=64,
    fov_rad=np.radians(20.0),
)
# 20% fractional bandwidth: wide enough that ``inv_lambda[c]`` really differs
# between channels (0.9 to 1.1 is a factor 1.22 in every baseline in
# wavelengths, so the w-plane windows land in different places per channel),
# narrow enough to stay one plausible band.
_MULTICHAN_FREQ = np.linspace(0.9, 1.1, 4) * _MULTICHAN_TEL.freq_hz


def test_multichannel_adjoint_parity() -> None:
    """Four channels, per-channel visibilities *and* weights, against ducc0.

    ducc0's ``vis2dirty`` sums over channels -- it returns one ``(npix_x,
    npix_y)`` image for an ``(n_rows, n_chan)`` visibility block -- whereas
    ``jax_nufft.vis2dirty`` returns the per-channel stack. So the ducc0
    statement here is necessarily about the **sum**, and that is worth being
    honest about: a sum over channels is invariant under permuting them, so
    this half cannot see a channel *mis-association* on its own. It can see a
    wrong per-channel scalar, a dropped or double-counted channel, and a wrong
    weight application, because all of those change the total.

    The per-channel half is what pins the association, and it needs the exact
    DFT (``tests/test_adjoint.py::_reference_adjoint``, the same reference the
    rest of the accuracy contract is written against) because ducc0 has no
    per-channel output to compare against. Hence the two bounds: ``3 * eps``
    for the ducc0 sum (two implementations, each with its own budget) and the
    tighter ``2 * eps`` per channel against the definition.

    Weights are drawn per ``(row, channel)`` and *not* normalised, so a channel
    whose weights were taken from a neighbouring column changes the answer
    rather than rescaling it.
    """
    eps = 1e-6
    tel = _MULTICHAN_TEL
    freq = _MULTICHAN_FREQ
    n_chan = freq.shape[0]
    shape = (64, 48)
    pixsize_l = tel.pixsize
    pixsize_m = 0.65 * pixsize_l

    uvw = synthetic_uvw(tel, 30.0, seed=77)
    rng = np.random.default_rng(2024)
    vis_np = (
        rng.standard_normal((tel.n_rows, n_chan)) + 1j * rng.standard_normal((tel.n_rows, n_chan))
    ).astype(np.complex128)
    wgt = rng.uniform(0.1, 2.0, size=(tel.n_rows, n_chan)).astype(np.float64)
    # Per-channel weights must actually differ per channel, or "with weights"
    # is "with one weight vector broadcast" and the (row, channel) indexing of
    # the weight array goes untested.
    assert not np.allclose(wgt[:, 0], wgt[:, 1])

    plan = make_plan(uvw, freq, shape, pixsize_l, pixsize_m, eps, hermitian=True)
    assert plan.n_chan == n_chan
    assert not plan.is_constant_w, "the w-plane loop must be engaged for this to mean anything"

    dirty_jax = np.asarray(
        vis2dirty(plan, jnp.asarray(vis_np), weights=jnp.asarray(wgt))
    )  # (n_chan, n_l, n_m)
    assert dirty_jax.shape == (n_chan, *shape)

    dirty_ducc = ducc0.wgridder.vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis_np,
        wgt=wgt,
        npix_x=shape[0],
        npix_y=shape[1],
        pixsize_x=pixsize_l,
        pixsize_y=pixsize_m,
        epsilon=eps,
        do_wgridding=True,
        divide_by_n=True,
        nthreads=1,
    )
    assert dirty_ducc.shape == shape

    summed = dirty_jax.sum(axis=0)
    err = np.linalg.norm(summed - dirty_ducc) / np.linalg.norm(dirty_ducc)
    assert err < DUCC_TOL_FACTOR * eps, (
        f"channel-summed adjoint with weights, shape={shape} "
        f"pixsize=({pixsize_l:.6g}, {pixsize_m:.6g}): relative error {err:.3e} exceeds "
        f"{DUCC_TOL_FACTOR:g}*eps={DUCC_TOL_FACTOR * eps:.3e} against ducc0"
    )

    dirty_ref = _reference_adjoint(vis_np, uvw, freq, shape, pixsize_l, pixsize_m, weights=wgt)
    for c in range(n_chan):
        c_err = np.linalg.norm(dirty_jax[c] - dirty_ref[c]) / np.linalg.norm(dirty_ref[c])
        assert c_err < DFT_TOL_FACTOR * eps, (
            f"adjoint channel {c} (freq={freq[c]:.6g} Hz) relative error {c_err:.3e} exceeds "
            f"{DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e} against the exact DFT -- the "
            "channel-summed ducc0 comparison above is blind to a channel mis-association, so "
            "this is the half that pins it"
        )


_PER_CHANNEL_TEL = Telescope(
    name="MWA_compact_per_channel_images",
    freq_hz=150e6,
    n_rows=300,
    sigma_uv_m=50.0,
    max_baseline_m=200.0,
    n_pix=48,
    fov_rad=np.radians(20.0),
)
_PER_CHANNEL_FREQ = np.array([0.9, 1.0, 1.1]) * _PER_CHANNEL_TEL.freq_hz
_PER_CHANNEL_SHAPE = (48, 64)


def _per_channel_forward_case() -> tuple[np.ndarray, np.ndarray, float, float]:
    """``(uvw, image, pixsize_l, pixsize_m)`` for the per-channel forward test.

    The image is ``(n_chan, n_l, n_m)`` with independent content in each plane
    and is **real**, which is both what ducc0's ``dirty`` argument accepts and
    the precondition of the plan's default Hermitian fold.
    """
    tel = _PER_CHANNEL_TEL
    uvw = synthetic_uvw(tel, 30.0, seed=88)
    rng = np.random.default_rng(31337)
    image = rng.standard_normal((_PER_CHANNEL_FREQ.shape[0], *_PER_CHANNEL_SHAPE))
    return uvw, image, tel.pixsize, 0.55 * tel.pixsize


@functools.cache
def _per_channel_ducc_reference() -> np.ndarray:
    """ducc0's answer for the per-channel image case, one channel at a time.

    ducc0's ``dirty2vis`` takes a single 2-D ``dirty``, so a multi-channel call
    necessarily broadcasts one image across every channel and cannot express
    per-channel content at all. Calling it once per channel -- image plane
    ``c`` with the single frequency ``freq[c]``, on the same ``uvw`` -- is the
    black-box way to get the per-channel answer out of it, and it makes the
    channel association an explicit statement rather than something the sum
    happens to be consistent with: column ``c`` of the multi-channel jax output
    must equal ducc0 run on plane ``c`` at frequency ``c``, for every ``c``.

    Cached because the eight strategy cells below all compare against it.
    """
    uvw, image, pixsize_l, pixsize_m = _per_channel_forward_case()
    columns = [
        ducc0.wgridder.dirty2vis(
            uvw=uvw,
            freq=_PER_CHANNEL_FREQ[c : c + 1],
            dirty=np.ascontiguousarray(image[c]),
            pixsize_x=pixsize_l,
            pixsize_y=pixsize_m,
            epsilon=1e-6,
            do_wgridding=True,
            divide_by_n=False,
            nthreads=1,
        )[:, 0]
        for c in range(_PER_CHANNEL_FREQ.shape[0])
    ]
    return np.stack(columns, axis=1)  # (n_rows, n_chan)


@pytest.mark.parametrize("channel_strategy", ["scan", "vmap"])
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_multichannel_forward_per_channel_images(w_strategy: str, channel_strategy: str) -> None:
    """A 3-D image with different content per channel, against ducc0 per channel.

    Every other multi-channel forward cell in this file broadcasts a single 2-D
    image, which makes the channel axis of the image trivially constant: an
    operator that fed image plane 0 to every channel, or that walked the image
    and ``plan.inv_lambda`` out of step, would return exactly the right answer.
    Here the three planes are independent draws, so either of those is a gross
    error.

    All four ``w_strategy`` values and both ``channel_strategy`` values,
    because the channel loop and the w-plane loop are composed differently in
    each: the windowed helpers take the shared ``sort_perm`` gather of
    ``plan.uvw_m`` plus this channel's scalar and this channel's row of
    ``plan.window_start`` as three separate arguments, which is a different
    per-channel composition from the dense path's two.

    The geometry is non-square (48 x 64) and anisotropic
    (``pixsize_m = 0.55 * pixsize_l``), so this doubles as ducc0 parity on an
    asymmetric grid for every strategy -- the square isotropic fixtures used
    elsewhere in this file cannot distinguish an ``[l, m]``-indexed
    grid quantity from an ``[m, l]``-indexed one.
    """
    eps = 1e-6
    uvw, image, pixsize_l, pixsize_m = _per_channel_forward_case()
    n_chan = _PER_CHANNEL_FREQ.shape[0]

    plan = make_plan(uvw, _PER_CHANNEL_FREQ, _PER_CHANNEL_SHAPE, pixsize_l, pixsize_m, eps)
    assert plan.n_chan == n_chan
    assert not plan.is_constant_w

    vis_jax = np.asarray(
        dirty2vis(
            plan,
            jnp.asarray(image),
            w_strategy=w_strategy,
            channel_strategy=channel_strategy,
        )
    )
    assert vis_jax.shape == (uvw.shape[0], n_chan)

    vis_ducc = _per_channel_ducc_reference()
    err = np.linalg.norm(vis_jax - vis_ducc) / np.linalg.norm(vis_ducc)
    assert err < DUCC_TOL_FACTOR * eps, (
        f"per-channel images, {w_strategy}/{channel_strategy}: relative error {err:.3e} "
        f"exceeds {DUCC_TOL_FACTOR:g}*eps={DUCC_TOL_FACTOR * eps:.3e} against ducc0"
    )
    # Per channel as well as in aggregate: the total is dominated by whichever
    # channel is loudest, so one mis-scaled or mis-associated channel can hide
    # inside an otherwise-good norm.
    for c in range(n_chan):
        c_err = np.linalg.norm(vis_jax[:, c] - vis_ducc[:, c]) / np.linalg.norm(vis_ducc[:, c])
        assert c_err < DUCC_TOL_FACTOR * eps, (
            f"per-channel images, {w_strategy}/{channel_strategy}, channel {c} "
            f"(freq={_PER_CHANNEL_FREQ[c]:.6g} Hz): relative error {c_err:.3e} exceeds "
            f"{DUCC_TOL_FACTOR:g}*eps={DUCC_TOL_FACTOR * eps:.3e}"
        )


def test_the_per_channel_image_planes_are_actually_distinct() -> None:
    """The fixture above must not be a broadcast image under another name.

    ``test_multichannel_forward_per_channel_images`` exists precisely because
    every other multi-channel cell here shares one image plane across channels.
    If its planes were ever made equal -- by a seed change, a reshape, or a
    ``np.broadcast_to`` -- the eight strategy cells would keep passing and stop
    testing anything the broadcast case does not already cover, silently. So
    the distinctness is asserted rather than assumed.

    The frequencies get the same treatment: at equal frequencies every channel
    shares one ``inv_lambda`` and a mis-association still lands on the right
    scalar.
    """
    _, image, _, _ = _per_channel_forward_case()
    n_chan = image.shape[0]
    assert n_chan > 1
    for a in range(n_chan):
        for b in range(a + 1, n_chan):
            assert not np.allclose(image[a], image[b]), (
                f"image planes {a} and {b} are equal: this fixture is a broadcast 2-D image "
                "in disguise and cannot distinguish per-channel content"
            )
    assert len(set(_PER_CHANNEL_FREQ.tolist())) == n_chan, (
        "channel frequencies must be distinct, or every channel shares one inv_lambda and "
        "the channel association goes untested"
    )
