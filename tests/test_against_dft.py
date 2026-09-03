"""Tiny-problem checks against the explicit DFT (single channel for now).

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
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


_TRACKING_EPS = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


def test_accuracy_tracks_epsilon() -> None:
    """Error must follow ``epsilon`` down, not plateau (issue #9).

    A single test rather than a parametrised one because the interesting
    assertion is *across* epsilon values: with the old width rule
    ``W = ceil(-log10(eps) * 2/pi) + 2``, eps=1e-5 and eps=1e-6 both mapped to
    W=6 and produced the *same* error, so asking for a tighter epsilon bought
    nothing. The monotonicity check below is the regression guard for that; the
    per-epsilon bound is the same ``2 * eps`` contract as the rest of the file.

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

    fwd_err: list[float] = []
    adj_err: list[float] = []
    for eps in _TRACKING_EPS:
        plan = make_plan(uvw, freq, shape, pix, pix, eps)
        vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image)))
        dirty_jax = np.asarray(vis2dirty(plan, jnp.asarray(vis)))
        e_f = float(np.linalg.norm(vis_jax - vis_ref) / np.linalg.norm(vis_ref))
        e_a = float(np.linalg.norm(dirty_jax - dirty_ref) / np.linalg.norm(dirty_ref))
        fwd_err.append(e_f)
        adj_err.append(e_a)
        assert e_f < DFT_TOL_FACTOR * eps, (
            f"forward eps={eps:g}: relative error {e_f:.3e} is {e_f / eps:.2f}x eps "
            f"(W={plan.w_kernel_width}, n_w={plan.n_w})"
        )
        assert e_a < DFT_TOL_FACTOR * eps, (
            f"adjoint eps={eps:g}: relative error {e_a:.3e} is {e_a / eps:.2f}x eps "
            f"(W={plan.w_kernel_width}, n_w={plan.n_w})"
        )

    # Strict monotonicity: every tightening of epsilon must actually buy
    # accuracy. Equal consecutive errors mean two epsilon values collapsed onto
    # the same kernel width.
    for name, errs in (("forward", fwd_err), ("adjoint", adj_err)):
        for k in range(1, len(_TRACKING_EPS)):
            assert errs[k] < errs[k - 1], (
                f"{name}: error did not improve going from eps={_TRACKING_EPS[k - 1]:g} "
                f"({errs[k - 1]:.3e}) to eps={_TRACKING_EPS[k]:g} ({errs[k]:.3e})"
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
