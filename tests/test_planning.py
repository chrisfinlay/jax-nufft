"""Tests for plan construction (Nw, w-plane centres, kernel correction)."""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import jax_nufft
from jax_nufft import dirty2vis, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
from jax_nufft.kernel import kernel_params
from jax_nufft.planning import (
    W_OVERSAMPLE_X0,
    WGridderPlan,
    make_plan,
    window_boundary_margin,
)
from jax_nufft.wgridder import _channel_ft_coords
from tests.conftest import (
    EDA2,
    MEERKAT,
    MWA_COMPACT,
    MWA_EXTENDED,
    Telescope,
    requires_x64,
    synthetic_uvw,
)


def _baseline_uvw(n_rows: int = 50, max_baseline: float = 100.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uvw = rng.normal(scale=max_baseline / 3, size=(n_rows, 3))
    # Truncate to max_baseline as a soft envelope.
    norms = np.linalg.norm(uvw, axis=1, keepdims=True)
    uvw = uvw / np.maximum(norms / max_baseline, 1.0)
    return uvw


def test_plan_basic_shapes() -> None:
    uvw = _baseline_uvw(n_rows=20, max_baseline=80.0)
    freq = np.array([100e6, 110e6, 120e6])
    plan = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(64, 64),
        pixsize_l=1.0e-3,
        pixsize_m=1.0e-3,
        epsilon=1e-6,
    )
    assert plan.n_l == 64
    assert plan.n_m == 64
    assert plan.n_chan == 3
    assert plan.n_rows == 20
    assert plan.uvw_lambda.shape == (3, 20, 3)
    assert plan.w_centers.shape == (plan.n_w,)
    assert plan.n_minus_1.shape == (64, 64)
    assert plan.phi_hat_n.shape == (64, 64)
    assert plan.beta > 0
    assert plan.w_kernel_width >= 2


def test_plan_kernel_params_match_eps() -> None:
    plan = make_plan(
        uvw=_baseline_uvw(),
        freq=np.array([1.4e9]),
        image_shape=(32, 32),
        pixsize_l=1e-4,
        pixsize_m=1e-4,
        epsilon=1e-7,
    )
    expected_w, expected_beta = kernel_params(1e-7)
    assert plan.w_kernel_width == expected_w
    assert plan.beta == pytest.approx(expected_beta)


# ---------------------------------------------------------------------------
# issue #23 (M2/M5/R9): uvw_lambda / uvw_lambda_sorted / u_finufft / v_finufft
# stored 8 float64 values per (channel, row), of which only 4 are ever read
# (uvw_lambda[..., 2], uvw_lambda_sorted[..., 2], u_finufft, v_finufft). The
# fix stores ``uvw_m`` (metres, once, input row order) and ``inv_lambda =
# freq / c`` (once, per channel) and derives the per-channel FINUFFT
# coordinates and the relative w inside the JIT -- three multiplies per row
# per channel. ``test_plan_uvw_lambda_correct`` and
# ``test_plan_finufft_coords_match_uvw_lambda`` (their v0.1.2-Part-3
# predecessors) asserted the *stored* per-channel arrays were correct; since
# those arrays no longer exist as plan leaves, what has to be gated instead is
# the derivation that replaced them.
#
# So this calls the PRODUCTION derivation, ``jax_nufft.wgridder.
# _channel_ft_coords`` -- the one and only implementation, shared by
# ``_channel_forward``, ``_channel_adjoint`` and both windowed helpers -- and
# compares all three of its outputs against references built here from
# ``uvw`` (metres) and ``freq`` (Hz) alone. An earlier draft of this test
# re-implemented the formula locally and checked *that* against the reference,
# which gated nothing: a swapped u/v axis, a dropped 2*pi, the wrong column of
# ``uvw_m`` or an absolute-instead-of-relative w in the real helper would all
# have left it green. The strategy-equivalence suite cannot catch such a bug
# either, because all four strategies call this same helper and would inherit
# the same wrong coordinates -- the shared-mode failure AGENTS.md sec 6
# records from issue #16, where the dot-product identity stayed green at every
# offset while the forward was catastrophically wrong.
#
# The references below therefore import nothing from ``src/``: they redo the
# arithmetic from the raw inputs, following this file's ``_nm1_extremes``
# convention, so a bug shared between the production helper and a test helper
# cannot hide behind it.
# ---------------------------------------------------------------------------


def _independent_ft_coords(
    uvw: np.ndarray, freq: np.ndarray, pixsize_l: float, pixsize_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Ground truth for ``(u_ft, v_ft, w_lambda, w0)``.

    The three arrays are ``(n_chan, n_rows)``; ``w0`` is the w-range midpoint
    over every (channel, row), which is what ``_channel_ft_coords`` subtracts
    to produce the relative w the plane loop uses.

    Deliberately built from ``uvw`` / ``freq`` / ``pixsize`` only -- no
    WGridderPlan field, old or new, and nothing imported from ``src/``.
    """
    inv_lambda = freq / SPEED_OF_LIGHT  # (n_chan,)
    u_ft = (2.0 * np.pi * pixsize_l) * np.outer(inv_lambda, uvw[:, 0])
    v_ft = (2.0 * np.pi * pixsize_m) * np.outer(inv_lambda, uvw[:, 1])
    w_lambda = np.outer(inv_lambda, uvw[:, 2])
    w0 = (float(np.min(w_lambda)) + float(np.max(w_lambda))) / 2.0
    return u_ft, v_ft, w_lambda, w0


@jax.jit
def _jit_channel_ft_coords(plan: WGridderPlan, inv_lambda_c: Array) -> tuple[Array, Array, Array]:
    """Call the production per-channel derivation under ``jax.jit``.

    Thin on purpose: the point of this test is that the assertion below runs
    the *shipped* helper, on the plan's own leaves, through the same tracing
    machinery the operators use -- not a transcription of it.
    """
    return _channel_ft_coords(plan.uvw_m, inv_lambda_c, plan)


@requires_x64
def test_channel_ft_coords_match_independent_reference() -> None:
    """``wgridder._channel_ft_coords`` must rebuild, from ``plan.uvw_m``
    (metres) and ``plan.inv_lambda`` (freq / c), exactly the (u_ft, v_ft)
    FINUFFT input coordinates and the relative w that the removed
    ``uvw_lambda`` / ``u_finufft`` / ``v_finufft`` leaves used to store.

    All three outputs are checked, per channel, against
    :func:`_independent_ft_coords`. Checking only ``u_ft`` would miss a
    swapped axis; checking only the magnitudes would miss the ``2*pi *
    pixsize`` factor; and checking an absolute w would miss the ``- w0``
    that makes the plane loop's phases small (issue #16's follow-up), which
    is why the third output is compared against ``w_lambda - w0`` with ``w0``
    recomputed here rather than read off the plan.

    Tolerances. ``plan.w0`` is compared bit for bit against the independently
    computed midpoint -- it is a plain min/max/average of the same float64
    products. The relative w is *not*: the helper computes
    ``(inv_lambda[c] * uvw_m[:, 2]) - w0``, and XLA is free to contract that
    multiply-then-subtract into a single FMA, one rounding where the numpy
    reference does two. The gap is bounded by half an ulp of the *absolute* w,
    not of the (much smaller) relative one, so the bound is an ``atol`` scaled
    by ``max|w_lambda|`` rather than an ``rtol``. ``u_ft`` / ``v_ft``
    additionally fold in ``2*pi * pixsize``, and the helper's grouping
    (``(2*pi * pixsize_l * inv_lambda[c]) * uvw_m[:, 0]``) reassociates that
    product differently from the reference's (``(2*pi * pixsize_l) *
    (inv_lambda * uvw)``, via ``np.outer``) -- multiplication is commutative
    but not associative in floating point, so the two can differ by a couple
    of ulps. Both bounds stay ~12 orders of magnitude below what an axis swap,
    a sign flip, a dropped ``2*pi``, an absolute-instead-of-relative w or a
    wrong-channel bug would produce on this fixture.

    Three properties of the fixture make those bugs visible, and none is
    incidental: ``pixsize_l != pixsize_m`` (so swapping the u and v scalings
    shows up), two widely separated channels (so a scaling that only looks
    right at one frequency shows up -- but *not* a wrong-channel read: this
    test hands the helper ``plan.inv_lambda[c]`` itself and the helper does no
    channel indexing, so the channel wiring in ``_dirty2vis_jit`` /
    ``_vis2dirty_jit`` is gated by
    ``test_multi_channel_matches_dft_forward_and_adjoint``, not here), and a
    large constant w offset, giving
    ``|w0| ~ 1.1e4`` wavelengths -- about 5x the relative-w spread, and some
    14 orders of magnitude above the tolerance the relative w is checked to,
    so returning the absolute w instead of ``w - w0`` is a gross failure
    rather than a rounding argument.
    """
    uvw = _baseline_uvw(n_rows=12, max_baseline=80.0)
    # Push the whole array off zenith by 2 km so the w-range midpoint is far
    # from zero (see the docstring): without this, w0 lands near 0 on a
    # symmetric fixture and returning the absolute w would be indistinguishable
    # from returning the relative one.
    uvw = uvw + np.array([0.0, 0.0, 2000.0])
    freq = np.array([1.4e9, 2.0e9])
    pixsize_l = 1.3e-3
    pixsize_m = 1.7e-3
    plan = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(32, 32),
        pixsize_l=pixsize_l,
        pixsize_m=pixsize_m,
        epsilon=1e-6,
    )

    assert plan.uvw_m.shape == (plan.n_rows, 3)
    assert plan.inv_lambda.shape == (plan.n_chan,)
    np.testing.assert_array_equal(np.asarray(plan.uvw_m), uvw)
    np.testing.assert_array_equal(np.asarray(plan.inv_lambda), freq / SPEED_OF_LIGHT)

    expected_u, expected_v, expected_w, expected_w0 = _independent_ft_coords(
        uvw, freq, pixsize_l, pixsize_m
    )
    assert plan.w0 == expected_w0, (
        f"plan.w0 {plan.w0!r} is not the w-range midpoint {expected_w0!r} over all "
        "(channel, row); the relative-w comparison below would then be self-consistent "
        "but wrong"
    )
    rtol = 20.0 * np.finfo(np.float64).eps
    # See the docstring: bound the relative w by ulps of the absolute w, since
    # an FMA-contracted (a*b - c) rounds once where the reference rounds twice.
    w_atol = 8.0 * np.finfo(np.float64).eps * float(np.max(np.abs(expected_w)))
    # The fixture must actually separate absolute from relative w, or the w
    # assertion below is vacuous: returning the absolute w would be an error of
    # |w0|, and that has to sit far above the tolerance it is measured against.
    assert abs(expected_w0) > 1e6 * w_atol

    for c in range(plan.n_chan):
        u_ft, v_ft, w_rel = _jit_channel_ft_coords(plan, plan.inv_lambda[c])
        np.testing.assert_allclose(
            np.asarray(w_rel), expected_w[c] - expected_w0, rtol=0.0, atol=w_atol
        )
        np.testing.assert_allclose(np.asarray(u_ft), expected_u[c], rtol=rtol, atol=0.0)
        np.testing.assert_allclose(np.asarray(v_ft), expected_v[c], rtol=rtol, atol=0.0)

    # Windowed path: the helpers feed the *same* production function a
    # sort_perm gather of uvw_m rather than a stored sorted array (issue #23
    # removed uvw_lambda_sorted on the same logic v0.1.2 used to drop the
    # sorted u/v coordinates). Check that composition end to end, again
    # through the production helper.
    sort_perm = np.asarray(plan.sort_perm)
    assert sorted(sort_perm.tolist()) == list(range(plan.n_rows))
    uvw_m_sorted = jnp.asarray(np.asarray(plan.uvw_m)[sort_perm])
    for c in range(plan.n_chan):
        u_s, v_s, w_s = _channel_ft_coords(uvw_m_sorted, plan.inv_lambda[c], plan)
        np.testing.assert_allclose(
            np.asarray(w_s), (expected_w[c] - expected_w0)[sort_perm], rtol=0.0, atol=w_atol
        )
        np.testing.assert_allclose(np.asarray(u_s), expected_u[c][sort_perm], rtol=rtol, atol=0.0)
        np.testing.assert_allclose(np.asarray(v_s), expected_v[c][sort_perm], rtol=rtol, atol=0.0)


@requires_x64
def test_plan_removed_leaf_names_stay_readable_via_backcompat() -> None:
    """``uvw_lambda``, ``n_minus_1`` and ``w_centers`` are read directly by
    ``tests/test_adjoint.py``, ``tests/test_dtype.py`` and
    ``tests/test_nshift.py`` -- suites issue #23 must leave untouched (it is
    "a pure storage change"). So the fix cannot simply delete these
    attributes: it must keep them readable (e.g. as ``@property`` computed
    from the surviving leaves) while removing them from the pytree leaves
    that ``jax.jit`` traces and that ``test_plan_footprint`` counts.

    This is the one place that pins both halves of that contract at once:
    correct values (this test), and NOT a leaf (test_plan_footprint /
    test_plan_leaves_are_exactly_the_expected_fields, which would fail if
    these were still counted).
    """
    uvw = _baseline_uvw(n_rows=10, max_baseline=60.0)
    freq = np.array([1e9, 2e9])
    plan = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(32, 32),
        pixsize_l=1e-3,
        pixsize_m=1e-3,
        epsilon=1e-6,
    )

    # uvw_lambda: same formula test_plan_uvw_lambda_correct used to check
    # directly against the (now removed) stored leaf.
    expected_uvw_lambda = uvw[None, :, :] * (freq[:, None, None] / SPEED_OF_LIGHT)
    np.testing.assert_allclose(np.asarray(plan.uvw_lambda), expected_uvw_lambda, rtol=1e-12)

    # n_minus_1 / n_minus_1_shifted: pin the *relationship* issue #23 relies
    # on to derive whichever of the pair is removed from the other plus the
    # static nshift, rather than re-deriving the n-1 grid independently a
    # second time (AGENTS.md sec 4 / planning.py already state the identity;
    # this is the contract the implementation must preserve, not a fresh
    # correctness check of the grid itself -- that is test_plan_nm1_
    # nonpositive_inside_disc and the nshift geometry tests below).
    np.testing.assert_allclose(
        np.asarray(plan.n_minus_1_shifted),
        np.asarray(plan.n_minus_1) + plan.nshift,
        rtol=0.0,
        atol=0.0,
    )

    # w_centers / w_centers_rel: likewise, w_centers == w0 + w_centers_rel.
    np.testing.assert_allclose(
        np.asarray(plan.w_centers),
        plan.w0 + np.asarray(plan.w_centers_rel),
        rtol=1e-9,
        atol=0.0,
    )

    # And neither uvw_lambda, n_minus_1 nor w_centers may be a pytree leaf:
    # a back-compat property must not smuggle the removed array back onto
    # the device / into the JIT cache key. Identity, not equality: a leaf
    # that happens to hold an equal value would still defeat the point.
    leaves = jax.tree_util.tree_leaves(plan)
    assert not any(leaf is plan.uvw_lambda for leaf in leaves)
    assert not any(leaf is plan.n_minus_1 for leaf in leaves)
    assert not any(leaf is plan.w_centers for leaf in leaves)


def test_plan_w_centers_span_data() -> None:
    """w-plane centres must extend symmetrically beyond the data range."""
    uvw = _baseline_uvw(n_rows=200, max_baseline=300.0)
    freq = np.array([200e6, 250e6])
    plan = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(128, 128),
        pixsize_l=1e-3,
        pixsize_m=1e-3,
        epsilon=1e-5,
    )
    w_lambda = uvw[None, :, :] * (freq[:, None, None] / SPEED_OF_LIGHT)
    w_min = float(np.min(w_lambda[..., 2]))
    w_max = float(np.max(w_lambda[..., 2]))
    centres = np.asarray(plan.w_centers)
    # The first / last centres should fall just outside the data range,
    # by half the kernel width times the spacing.
    assert centres[0] < w_min
    assert centres[-1] > w_max
    # Spacing is uniform.
    spacings = np.diff(centres)
    np.testing.assert_allclose(spacings, spacings[0], rtol=1e-6)


def test_plan_nw_scales_with_w_extent() -> None:
    """Doubling max baseline should ~double the inner w-plane count."""
    eps = 1e-6
    freq = np.array([1.4e9])
    image_shape = (256, 256)
    pixsize = 5e-4

    uvw_short = _baseline_uvw(n_rows=200, max_baseline=200.0, seed=0)
    uvw_long = _baseline_uvw(n_rows=200, max_baseline=400.0, seed=0) * 2.0
    plan_short = make_plan(uvw_short, freq, image_shape, pixsize, pixsize, eps)
    plan_long = make_plan(uvw_long, freq, image_shape, pixsize, pixsize, eps)
    # Inner w-plane count is n_w - W_k. The "long" plan should have ~2x more.
    inner_short = plan_short.n_w - plan_short.w_kernel_width
    inner_long = plan_long.n_w - plan_long.w_kernel_width
    assert inner_long > inner_short
    # Allow some slack since both are computed with ceil().
    assert inner_long >= 1.5 * inner_short - 2


def test_plan_zero_w_extent_is_handled() -> None:
    """All-zero w (perfectly zenith, coplanar array) collapses to a single
    plane via the v0.1.2 constant-w fast path.

    Before v0.1.2 the dense path would produce ``dw=0`` and ``w_kernel_scale=0``,
    which yielded NaN at call time via ``z = 0/0``. The fast path replaces
    that with ``n_w=1`` and a unit phi_hat correction.
    """
    uvw = _baseline_uvw(n_rows=20, max_baseline=50.0)
    uvw[:, 2] = 0.0
    plan = make_plan(
        uvw=uvw,
        freq=np.array([200e6]),
        image_shape=(64, 64),
        pixsize_l=1e-3,
        pixsize_m=1e-3,
        epsilon=1e-6,
    )
    assert plan.is_constant_w
    assert plan.w_extent == 0.0
    assert plan.n_w == 1
    assert np.all(np.isfinite(np.asarray(plan.w_centers)))


def test_constant_w_collapses_n_w() -> None:
    """Non-zero constant w (snapshot at fixed pointing) also triggers the
    fast path and pins ``w_centers[0]`` to the constant w-value in wavelengths.
    """
    uvw = _baseline_uvw(n_rows=64, max_baseline=400.0)
    # Replace the w column with a non-zero constant in metres.
    w_const_m = 12.5
    uvw[:, 2] = w_const_m
    freq_hz = np.array([200e6])
    plan = make_plan(
        uvw=uvw,
        freq=freq_hz,
        image_shape=(64, 64),
        pixsize_l=1e-3,
        pixsize_m=1e-3,
        epsilon=1e-6,
    )
    # The plan-level invariant.
    assert plan.is_constant_w == (plan.w_extent == 0.0)
    assert plan.is_constant_w
    assert plan.n_w == 1
    # In wavelengths, the constant value is w_m * freq / c. Single channel
    # here so the per-channel and worst-case values coincide.
    w_const_lambda = w_const_m * float(freq_hz[0]) / SPEED_OF_LIGHT
    np.testing.assert_allclose(np.asarray(plan.w_centers), [w_const_lambda], rtol=0.0, atol=1e-9)
    # phi_hat_n is unity for the fast path (no correction needed).
    np.testing.assert_allclose(np.asarray(plan.phi_hat_n), 1.0)
    # Windowed metadata: single window per channel covering all rows.
    # (window_size itself is issue #23's removed diagnostic-only leaf --
    # never read at call time -- so the "covers all n_rows" property is
    # pinned via max_window_size, still a static field, instead.)
    assert plan.max_window_size == plan.n_rows
    assert np.asarray(plan.window_start).shape == (1, 1)
    assert int(np.asarray(plan.window_start)[0, 0]) == 0


def test_plan_phi_hat_n_strictly_positive() -> None:
    plan = make_plan(
        uvw=_baseline_uvw(n_rows=200, max_baseline=300.0),
        freq=np.array([200e6, 250e6]),
        image_shape=(128, 128),
        pixsize_l=1e-3,
        pixsize_m=1e-3,
        epsilon=1e-5,
    )
    phi_hat = np.asarray(plan.phi_hat_n)
    assert np.all(phi_hat > 0)


def test_plan_nm1_nonpositive_inside_disc() -> None:
    """n - 1 must be <= 0 everywhere on a Nyquist-sampled image."""
    plan = make_plan(
        uvw=_baseline_uvw(),
        freq=np.array([200e6]),
        image_shape=(64, 64),
        pixsize_l=1e-3,
        pixsize_m=1e-3,
        epsilon=1e-6,
    )
    nm1 = np.asarray(plan.n_minus_1)
    assert np.all(nm1 <= 0.0)
    # The centre pixel is exactly at l=m=0, so n-1=0 there.
    assert nm1[plan.n_l // 2, plan.n_m // 2] == pytest.approx(0.0)


def _nm1_extremes(
    image_shape: tuple[int, int], pixsize_l: float, pixsize_m: float
) -> tuple[float, float]:
    """Independently recompute ``(max, min)`` of ``n - 1`` over the image grid.

    Deliberately a *separate* implementation from ``make_plan``'s (this
    repository's convention -- see ``tests/test_against_dft.py::reference_lmn_grids``
    and ``tests/test_adjoint.py::_reference_adjoint`` for the same pattern), so
    that a bug shared between ``make_plan`` and this helper cannot hide behind
    an ``nshift`` test that trusts the code it is checking. Matches ducc's
    analytic extension outside the unit disc (``n - 1 = -sqrt(l^2+m^2-1) - 1``),
    which ``make_plan`` uses so full-sky images (large ``l^2+m^2 > 1`` region,
    e.g. EDA2's 120-degree FoV) get a well-defined ``n - 1`` everywhere.
    """
    n_l, n_m = image_shape
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    ll = (i * pixsize_l)[:, None]
    mm = (j * pixsize_m)[None, :]
    r2 = ll * ll + mm * mm
    inside_disc = r2 <= 1.0
    inside_val = np.sqrt(np.where(inside_disc, 1.0 - r2, 0.0)) - 1.0
    outside_val = -np.sqrt(np.where(inside_disc, 0.0, r2 - 1.0)) - 1.0
    nm1 = np.where(inside_disc, inside_val, outside_val)
    return float(nm1.max()), float(nm1.min())


# A spread of fixtures for the nshift geometry check: two small synthetic
# baselines (zenith and off30, narrow FoV -- the whole image sits inside the
# unit disc, so nm1_max == 0 exactly and nshift == -nm1_min/2), plus EDA2's
# 120-degree FoV at both pointings, where a large fraction of the image lies
# *outside* the unit disc and nm1_max/nm1_min both come from the analytic
# extension. Also MWA_extended/MeerKAT off30, the fixtures issue #16 quotes
# numbers for.
_NSHIFT_GEOMETRY_FIXTURES: tuple[tuple[Telescope, float], ...] = (
    (EDA2, 0.0),
    (EDA2, 30.0),
    (MWA_COMPACT, 0.0),
    (MWA_COMPACT, 30.0),
    (MWA_EXTENDED, 30.0),
    (MEERKAT, 30.0),
)


def _fixture_id(values: tuple[Telescope, float]) -> str:
    tel, ang = values
    return f"{tel.name}_{'zenith' if ang == 0.0 else f'off{int(ang)}'}"


@pytest.mark.parametrize(
    ("telescope", "zenith_angle_deg"),
    _NSHIFT_GEOMETRY_FIXTURES,
    ids=[_fixture_id(v) for v in _NSHIFT_GEOMETRY_FIXTURES],
)
def test_nshift_matches_geometry(telescope: Telescope, zenith_angle_deg: float) -> None:
    """issue #16: ``nshift == -(nm1_max + nm1_min) / 2``, exactly.

    The w-phase identity ``exp(2*pi*i*w*(n-1)) == exp(2*pi*i*w*(n-1+s)) *
    exp(-2*pi*i*w*s)`` holds for *any* constant ``s``; the plan is required to
    pick the specific ``s`` that centres the shifted ``n-1`` range around zero,
    which is what halves ``max|n-1+s|`` (and therefore the plane count) for any
    image containing the phase centre. This test recomputes ``nm1_max`` /
    ``nm1_min`` independently (see ``_nm1_extremes``) rather than trusting
    ``make_plan``'s own grid, so a bug shared between the two cannot hide.
    """
    uvw = synthetic_uvw(telescope, zenith_angle_deg, seed=0)
    freq = np.array([telescope.freq_hz])
    image_shape = (telescope.n_pix, telescope.n_pix)
    plan = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=image_shape,
        pixsize_l=telescope.pixsize,
        pixsize_m=telescope.pixsize,
        epsilon=1e-6,
    )

    nm1_max, nm1_min = _nm1_extremes(image_shape, telescope.pixsize, telescope.pixsize)
    expected_nshift = -(nm1_max + nm1_min) / 2.0
    assert plan.nshift == pytest.approx(expected_nshift, abs=1e-9)


def test_plan_invalid_inputs() -> None:
    uvw = _baseline_uvw()
    freq = np.array([200e6])
    with pytest.raises(ValueError):
        make_plan(uvw, freq, (64, 64), 1e-3, 1e-3, epsilon=0.0)
    with pytest.raises(ValueError):
        make_plan(uvw, freq, (64, 64), -1e-3, 1e-3, epsilon=1e-6)
    with pytest.raises(ValueError):
        make_plan(uvw, freq, (0, 64), 1e-3, 1e-3, epsilon=1e-6)
    with pytest.raises(ValueError):
        make_plan(uvw[..., :2], freq, (64, 64), 1e-3, 1e-3, epsilon=1e-6)


# ---------------------------------------------------------------------------
# The plan's pytree contract, field by field (AGENTS.md sec 4)
# ---------------------------------------------------------------------------
#
# These two tables are the machine-checkable half of AGENTS.md sec 4's
# plan-field checklist. They are written out by hand rather than derived from
# ``dataclasses.fields(WGridderPlan)`` on purpose: deriving them would make the
# test agree with whatever the dataclass happens to say, which is exactly the
# drift it exists to catch. Adding a field to ``WGridderPlan`` without also
# adding it here -- and to ``_plan_aux`` / ``register_pytree_node`` --  must
# fail, because that combination is the "silent pytree corruption" sec 4 warns
# about: a leaf missing from the flatten tuple is silently frozen into the
# traced computation, and a static field missing from the aux data silently
# shares a JIT cache entry across two genuinely different operators.
#
# Leaves, in flatten order -- issue #23 (M2/M5/R9) post-change set.
#
# Removed outright (never read at call time, or read only through a value
# trivially reconstructible from what remains):
#   * uvw_lambda, uvw_lambda_sorted, u_finufft, v_finufft -- replaced by
#     uvw_m (metres, once) + inv_lambda (freq/c, once); the per-channel
#     coordinates are derived inside the JIT (three multiplies per row per
#     channel) instead of pre-materialised per channel on the host. The
#     windowed path gathers uvw_m via sort_perm the same way it already
#     gathers u_finufft/v_finufft, so no separate sorted leaf is needed.
#   * window_size -- never read inside _dirty2vis_jit / _vis2dirty_jit (grep
#     confirms; only window_start and the static max_window_size are), only
#     used at plan time to compute the (already-static) window_padding_
#     overhead and in this file's own tests. Not reintroduced in any form.
#
# Collapsed (this issue's own review finding, not in the original issue #23
# text, which predates issue #16): #16 added an *exact* redundancy in each
# of these pairs -- n_minus_1_shifted = n_minus_1 + nshift (nshift static)
# and w_centers = w_centers_rel + w0 (w0 static) -- so storing both leaves in
# a pair is the same dead weight this issue targets. Exactly one of each pair
# survives as a leaf; the other becomes a derived quantity, still readable as
# a same-named attribute (a handful of *other*, untouched suites --
# test_adjoint.py, test_dtype.py, test_nshift.py -- read plan.n_minus_1 /
# plan.w_centers / plan.uvw_lambda directly, so "pure storage change" means
# these keep working, just not as pytree leaves any more; see
# test_plan_removed_leaf_names_stay_readable_via_backcompat).
#
# n_minus_1_shifted and w_centers_rel are pinned below as the survivors:
# both are read inside the w-plane loop (every plane, both operators, in
# _channel_forward*/_channel_adjoint*), while n_minus_1 is read exactly once
# (the adjoint's final 1/n factor) and w_centers (absolute) is not read at
# call time at all (only by tests / the window-builder contract docs) --  so
# recomputing the *other* member of each pair on demand is the cheaper
# direction. This is a naming choice for testability, not a constraint from
# the issue: if the implementation picks the other survivor in either pair,
# this tuple (and the two field names in test_plan_footprint's docstring)
# need the corresponding one-line rename in the same commit (AGENTS.md sec 8
# rule 4's plan-field checklist covers exactly this).
_EXPECTED_LEAF_FIELDS: tuple[str, ...] = (
    "uvw_m",  # issue #23 -- replaces uvw_lambda / uvw_lambda_sorted
    "inv_lambda",  # issue #23 -- replaces the per-channel scaling baked into uvw_lambda
    "w_centers_rel",  # issue #16 follow-up; survives its redundancy with w_centers (#23)
    "n_minus_1_shifted",  # issue #16; survives its redundancy with n_minus_1 (#23)
    "w0_screen",  # issue #16 follow-up
    "phi_hat_n",
    "sort_perm",
    "window_start",
)

# Static (aux_data) fields, each with a way to produce a *different* value of
# the same kind. The value only has to be distinguishable -- the mutated plans
# below are flattened, never evaluated -- so an inconsistent one (n_l + 1 with
# unchanged arrays) is fine and keeps the probes one-liners.
_STATIC_FIELD_PROBES: tuple[tuple[str, Callable[[Any], Any]], ...] = (
    ("n_l", lambda v: v + 1),
    ("n_m", lambda v: v + 1),
    ("n_chan", lambda v: v + 1),
    ("n_rows", lambda v: v + 1),
    ("n_w", lambda v: v + 1),
    ("w_kernel_width", lambda v: v + 1),
    ("beta", lambda v: v + 1.0),
    ("epsilon", lambda v: v * 10.0),
    ("pixsize_l", lambda v: v * 2.0),
    ("pixsize_m", lambda v: v * 2.0),
    ("w_kernel_scale", lambda v: v + 1.0),
    ("nshift", lambda v: v + 1.0),  # issue #16
    ("w0", lambda v: v + 1.0),  # issue #16 follow-up
    ("max_window_size", lambda v: v + 1),
    ("window_padding_overhead", lambda v: v + 1.0),
    # issue #43: the two ints that replaced the padded ``window_size`` mean as
    # the diagnostic's denominator. They are STATIC on purpose -- issue #23
    # (PR #42) removed the per-(channel, plane) ``window_size`` leaf to cut
    # plan memory, and the fix for #43 must not reintroduce a per-plane array
    # in any form. ``_EXPECTED_LEAF_FIELDS`` stays at eight entries.
    ("live_row_count", lambda v: v + 1),
    ("empty_plane_count", lambda v: v + 1),
    ("w_extent", lambda v: v + 1.0),
    ("is_constant_w", lambda v: not v),
    ("real_dtype", lambda v: np.dtype(np.float32)),
    ("complex_dtype", lambda v: np.dtype(np.complex64)),
)


def _reference_plan() -> WGridderPlan:
    return make_plan(
        uvw=_baseline_uvw(n_rows=10),
        freq=np.array([200e6]),
        image_shape=(32, 32),
        pixsize_l=1e-3,
        pixsize_m=1e-3,
        epsilon=1e-6,
    )


@requires_x64
def test_plan_leaves_are_exactly_the_expected_fields() -> None:
    """Every pytree leaf is a named plan field, in order, and nothing else.

    Comparing against ``[getattr(plan, name) for name in _EXPECTED_LEAF_FIELDS]``
    by *identity* pins membership and order at once. A bare
    ``len(leaves) == N`` -- which is all this used to check -- passes just as
    happily when one leaf is swapped for another, or when a newly added leaf
    displaces an existing one in the flatten tuple, so it gates nothing.
    """
    plan = _reference_plan()
    leaves = jax.tree_util.tree_leaves(plan)
    expected = [getattr(plan, name) for name in _EXPECTED_LEAF_FIELDS]

    assert len(leaves) == len(_EXPECTED_LEAF_FIELDS), (
        f"expected {len(_EXPECTED_LEAF_FIELDS)} leaves "
        f"{_EXPECTED_LEAF_FIELDS}, got {len(leaves)} -- a field was added to "
        "WGridderPlan without updating register_pytree_node, _plan_unflatten, "
        "_EXPECTED_LEAF_FIELDS and AGENTS.md sec 4 together"
    )
    for name, got, want in zip(_EXPECTED_LEAF_FIELDS, leaves, expected, strict=True):
        assert got is want, f"leaf out of order or wrong field at {name!r}"

    # The issue #11 dtype metadata is *static*, so it must not show up here.
    assert not any(isinstance(leaf, np.dtype) for leaf in leaves)


@requires_x64
@pytest.mark.parametrize("field_name", _EXPECTED_LEAF_FIELDS)
def test_each_plan_leaf_round_trips_and_is_really_a_leaf(field_name: str) -> None:
    """Each named leaf survives flatten/unflatten *and* is actually traced.

    Two distinct failures are covered. Dropping the field from
    ``register_pytree_node``'s children tuple makes the mutation below change
    no leaf at all (it would be baked into the aux data instead), and dropping
    it from ``_plan_unflatten`` makes the round-trip lose it.
    """
    plan = _reference_plan()
    leaves, treedef = jax.tree_util.tree_flatten(plan)

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, WGridderPlan)
    assert getattr(rebuilt, field_name) is getattr(plan, field_name), (
        f"{field_name} did not survive the pytree round-trip"
    )

    # Perturb this leaf only: exactly one leaf must differ.
    mutated = dataclasses.replace(plan, **{field_name: getattr(plan, field_name) + 1})
    mutated_leaves = jax.tree_util.tree_leaves(mutated)
    assert len(mutated_leaves) == len(leaves)
    differing = [
        name
        for name, a, b in zip(_EXPECTED_LEAF_FIELDS, mutated_leaves, leaves, strict=True)
        if a is not b
    ]
    assert differing == [field_name], (
        f"changing {field_name} should change exactly that one leaf; changed {differing}. "
        "An empty list means the field is missing from the flatten_func children tuple"
    )
    # Structure is unchanged: perturbing a leaf's *value* must not move it into
    # the aux data (that would re-trigger a JIT recompile on every new value).
    assert jax.tree_util.tree_structure(mutated) == treedef


@requires_x64
@pytest.mark.parametrize(
    ("field_name", "perturb"), _STATIC_FIELD_PROBES, ids=[n for n, _ in _STATIC_FIELD_PROBES]
)
def test_each_static_plan_field_is_in_the_aux_data(
    field_name: str, perturb: Callable[[Any], Any]
) -> None:
    """Each static field is part of the treedef, i.e. of the JIT cache key.

    The swap goes through ``dataclasses.replace`` rather than a second
    ``make_plan(...)`` call *on purpose*. A separately built plan would differ
    in several aux entries at once (asking for float32 also forces a different
    epsilon, hence a different kernel width, beta, n_w and w_kernel_scale), so
    its treedef would compare unequal even if the field under test had been
    dropped from ``_plan_aux`` entirely -- and the assertion would prove
    nothing. Here every other aux entry and every leaf is identical by
    construction, so this fails if and only if this field is missing from
    ``_plan_aux``.
    """
    plan = _reference_plan()
    leaves, treedef = jax.tree_util.tree_flatten(plan)

    mutated = dataclasses.replace(plan, **{field_name: perturb(getattr(plan, field_name))})
    assert getattr(mutated, field_name) != getattr(plan, field_name), (
        "the probe must actually change the value, or this test is vacuous"
    )

    mutated_leaves = jax.tree_util.tree_leaves(mutated)
    assert all(a is b for a, b in zip(mutated_leaves, leaves, strict=True)), (
        f"changing {field_name} alone must not disturb any leaf, or the treedef "
        "comparison below would not isolate the aux data"
    )
    assert jax.tree_util.tree_structure(mutated) != treedef, (
        f"{field_name} must be part of the pytree aux_data (_plan_aux), or two "
        f"plans differing only in {field_name} would share a treedef / JIT cache "
        "entry -- exactly the 'silent pytree corruption' AGENTS.md sec 4 warns about"
    )


@requires_x64
def test_plan_is_a_jax_pytree() -> None:
    """The plan can flow through pytree-aware transforms (jit, vmap, etc.)."""
    plan = _reference_plan()

    leaves, treedef = jax.tree_util.tree_flatten(plan)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, WGridderPlan)
    # Static fields preserved exactly.
    assert rebuilt.n_l == plan.n_l
    assert rebuilt.n_w == plan.n_w
    assert rebuilt.beta == plan.beta
    # issue #11: real_dtype / complex_dtype are aux data, so they survive the
    # round-trip unchanged (AGENTS.md sec 4 plan-field checklist).
    assert rebuilt.real_dtype == plan.real_dtype
    assert rebuilt.complex_dtype == plan.complex_dtype
    assert np.dtype(plan.real_dtype) == np.dtype(jnp.float64)
    assert np.dtype(plan.complex_dtype) == np.dtype(jnp.complex128)

    # And we can read it through a jit'd function.
    @jax.jit
    def total_phi_hat(p: WGridderPlan) -> jax.Array:
        return jnp.sum(p.phi_hat_n)

    out = float(total_phi_hat(plan))
    assert np.isfinite(out)
    assert out > 0


def test_window_builder_basic() -> None:
    """sort_perm sorts by w; per-plane windows are monotonic and in-bounds.

    The coordinate cross-check this test used to do here (``uvw_lambda_sorted
    matches uvw_lambda[:, sort_perm, :]``) moved to
    ``test_plan_derived_channel_coords_match_independent_reference`` -- that
    leaf is gone under issue #23, and the windowed path's w-in-sorted-order
    is now a gather of ``uvw_m`` by ``sort_perm`` rather than a stored array.
    Likewise the exact per-window bound (``window_start + window_size <=
    n_rows``) needed the removed ``window_size`` leaf; see
    ``test_window_builder_matches_independent_reference`` below for the
    exact-match replacement (which pins ``window_start`` more tightly than
    this ever did).
    """
    rng = np.random.default_rng(0)
    n_rows = 400
    uvw = rng.normal(scale=80.0, size=(n_rows, 3))
    freq = np.array([1.0e9, 1.2e9])

    plan = make_plan(uvw, freq, (128, 128), 1e-3, 1e-3, epsilon=1e-6)

    sort_perm = np.asarray(plan.sort_perm)
    # Permutation property: every index appears exactly once.
    assert sorted(sort_perm.tolist()) == list(range(n_rows))
    # Applying sort_perm yields ascending w in metres.
    w_sorted = uvw[sort_perm, 2]
    assert np.all(np.diff(w_sorted) >= 0)

    window_start = np.asarray(plan.window_start)
    assert window_start.shape == (plan.n_chan, plan.n_w)

    # Window start is monotonic in k (planes scan ascending in w).
    for c in range(plan.n_chan):
        assert np.all(np.diff(window_start[c]) >= 0)
    # All windows start within [0, n_rows].
    assert np.all(window_start >= 0)
    assert np.all(window_start <= plan.n_rows)
    # Padding overhead >= 1 by construction.
    assert plan.window_padding_overhead >= 1.0


def _independent_window_bounds(
    uvw: np.ndarray, freq: np.ndarray, plan: WGridderPlan
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute ``(window_start, window_size)`` independently of make_plan's
    own builder (planning.py's ``for c in range(n_chan): ... np.searchsorted``
    loop), from raw ``uvw`` / ``freq`` plus the *static* ``w_kernel_scale`` /
    ``w0`` fields (unaffected by issue #23) and the ``sort_perm`` /
    ``w_centers_rel`` leaves (also unaffected -- both survive this issue).
    Written fresh rather than imported, per this file's ``_nm1_extremes``
    convention, so a shared bug cannot hide behind it.

    Stands in for the removed ``window_size`` leaf, which was diagnostic-only
    (never read at call time -- see ``_EXPECTED_LEAF_FIELDS``'s comment) and
    is not reintroduced by issue #23 in any form.
    """
    sort_perm = np.asarray(plan.sort_perm)
    n_rows = uvw.shape[0]
    w_m_sorted = uvw[sort_perm, 2].astype(np.float64)
    w_centers_rel64 = np.asarray(plan.w_centers_rel, dtype=np.float64)
    half_w_dw = plan.w_kernel_scale
    n_chan, n_w = plan.n_chan, plan.n_w
    # The builder widens each boundary before searching, and then by one row at
    # each end, so that a window is guaranteed to contain every row the *device*
    # puts inside kernel support even though the device's FMA-contracted w
    # differs from the host's in the last bits (AGENTS.md sec 4's window-builder
    # invariant; test_windowed_dense_parity_at_window_edge is the value gate).
    # Reproduced here because it is part of the contract this reference exists
    # to pin -- an implementation that dropped the widening would be a silent
    # parity regression, so this must not quietly accept one.
    w_abs = np.outer(np.asarray(freq, dtype=np.float64) / SPEED_OF_LIGHT, uvw[:, 2])
    # The production helper, not a transcription of it: this reference exists to
    # pin window_start exactly, so it has to widen by the same number make_plan
    # widened by, whatever that number is.
    margin = window_boundary_margin(
        plan.real_dtype, float(w_abs.min()), float(w_abs.max()), plan.w_extent
    )
    window_start = np.zeros((n_chan, n_w), dtype=np.int64)
    window_size = np.zeros((n_chan, n_w), dtype=np.int64)
    for c in range(n_chan):
        w_lambda_c = w_m_sorted * (float(freq[c]) / SPEED_OF_LIGHT) - plan.w0
        lo = np.searchsorted(w_lambda_c, w_centers_rel64 - half_w_dw - margin, side="left")
        hi = np.searchsorted(w_lambda_c, w_centers_rel64 + half_w_dw + margin, side="right")
        lo = np.maximum(lo - 1, 0)
        hi = np.minimum(hi + 1, n_rows)
        window_start[c] = lo
        window_size[c] = hi - lo
    return window_start, window_size


@pytest.mark.parametrize(
    "freq",
    [
        pytest.param(np.array([1.4e9]), id="single_channel"),
        # Widely split, descending freq, so the widest window is in channel 1
        # and channel 0 is strictly narrower (241 vs 250 rows on this fixture).
        # ``max_window_size`` sizes every windowed ``dynamic_slice``, so a
        # builder that maxed over channel 0 alone would undersize the slice and
        # silently drop rows from the other channels' windows -- a real-value
        # error, not a diagnostic one. The split has to be this wide: on
        # narrower pairs every window saturates at n_rows in both channels, so
        # the bug survives. Verified by mutating ``window_size_np.max()`` to
        # ``window_size_np[0].max()``, which this case catches and the
        # single-channel one does not.
        pytest.param(np.array([2.0e9, 0.5e9]), id="max_in_channel_1"),
    ],
)
def test_window_builder_matches_independent_reference(freq: np.ndarray) -> None:
    """``plan.window_start`` must match an independently-computed reference
    exactly, and ``plan.max_window_size`` must be the max over *every*
    (channel, plane) window -- the window sizes themselves stopped being a plan
    leaf under issue #23.
    """
    rng = np.random.default_rng(1)
    n_rows = 250
    uvw = rng.normal(scale=120.0, size=(n_rows, 3))

    plan = make_plan(uvw, freq, (128, 128), 5e-4, 5e-4, epsilon=1e-6)
    expected_start, expected_size = _independent_window_bounds(uvw, freq, plan)

    np.testing.assert_array_equal(np.asarray(plan.window_start), expected_start)
    assert plan.max_window_size == int(expected_size.max())

    # The predecessor test bounded ``window_size.sum()`` above and below by
    # ``n_rows * W``. Those bounds are not reproduced here: ``expected_size`` is
    # this file's own reference array, so asserting on its sum would check the
    # reference against itself, and ``_independent_window_bounds`` transcribes
    # the same searchsorted logic -- a builder that systematically widened its
    # windows would widen the reference with it and stay green. Anchor the
    # aggregate on something make_plan computes instead.
    #
    # issue #43: the denominator is ``live_row_count``, the incidences inside
    # the *unpadded* support, so ``expected_size`` (which carries the margin
    # and the +/-1 clamp) is the wrong array to divide by and is strictly
    # larger. ``tests/test_padding_overhead.py`` is where the live count is
    # pinned against a reference that does not go through ``searchsorted`` at
    # all; here it is enough that the identity holds and that the padding is
    # visibly excluded.
    assert plan.window_padding_overhead == pytest.approx(
        plan.n_chan * plan.n_w * plan.max_window_size / plan.live_row_count
    )
    assert plan.live_row_count < int(expected_size.sum())


def _clumped_and_uniform_uvw(n_rows: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Two w-distributions with the same u, v scale: two tight clumps, and flat.

    Drawn from one generator in this order so the arrays are exactly the ones
    the pre-#43 version of the test below used.
    """
    rng = np.random.default_rng(2)
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100, 100, n_rows)
    uvw[:, 1] = rng.uniform(-100, 100, n_rows)
    half = n_rows // 2
    uvw[:half, 2] = rng.normal(loc=-30.0, scale=0.5, size=half)
    uvw[half:, 2] = rng.normal(loc=+30.0, scale=0.5, size=n_rows - half)
    return uvw, rng.uniform(-60.0, 60.0, size=(n_rows, 3))


# The resolutions this sweep runs at. issue #16 added a ``sqrt(2)`` factor to a
# single hard-coded pixel size here so that a NON-INVARIANT assertion kept
# passing: "clumped overhead > uniform overhead" is simply false below a
# resolution crossover, and the fudge picked a point on the true side of it.
# issue #43 replaces the fudge with the crossover itself, which is analytic.
#
# Write ``n_rows * W`` for the live incidence count (every row is live in W
# planes, whatever the distribution -- ``live_row_count`` is 2800/2801 at every
# resolution below, against 400 * 7). Then
#
#   overhead = n_w * max_window_size / (n_rows * W)
#
# and the two distributions differ only in ``max_window_size``:
#
#   * two equal clumps, each narrow against the kernel support: the widest
#     window holds one whole clump, ``n_rows / 2``, so
#     ``overhead_clumped ~ n_w / (2W)``;
#   * uniform over the w-extent: the widest window holds its share of the
#     inner planes, ``n_rows * W / (n_w - W)``, so
#     ``overhead_uniform ~ n_w / (n_w - W)``.
#
# Those are equal at ``n_w - W == 2W``, i.e. ``n_w == 3W``. Below it the
# clumped plan is genuinely the flatter of the two and the assertion *should*
# fail; above it the clumped plan's peak dominates and the ordering is real.
# Measured on this fixture at W=7 (eps=1e-6), so the crossover is n_w = 21:
#
#   pixsize   n_w clumped/uniform   overhead clumped/uniform
#   2.0e-3     10 / 12               1.428 / 1.714   (below crossover)
#   2.8e-3     12 / 17               1.714 / 1.681   (below; the #16 fudge)
#   4.0e-3     17 / 26               1.220 / 1.448   (below crossover)
#   5.0e-3     23 / 36               1.650 / 1.427
#   8.0e-3     47 / 83               3.374 / 1.423
#   1.2e-2    101 / 187              7.248 / 1.669
#   1.5e-2    163 / 304             11.697 / 1.954
#
# Larger pixels mean a wider field, hence a larger ``max|n-1+nshift|``, hence
# more planes -- so the sweep runs *up* in pixel size to get above ``3W``.
_CLUMPED_SWEEP_PIXSIZES = (5e-3, 6e-3, 8e-3, 1e-2, 1.2e-2, 1.5e-2)


@pytest.mark.parametrize("pixsize", _CLUMPED_SWEEP_PIXSIZES)
def test_window_builder_clumped_distribution(pixsize: float) -> None:
    """Above the ``n_w = 3W`` crossover, clumped w really does pad more.

    The precondition is asserted rather than skipped: if a planning change
    moves ``n_w`` back below the crossover, this fixture stops measuring what
    the test claims and that must fail loudly instead of silently passing at
    one lucky resolution.
    """
    uvw, uvw_uniform = _clumped_and_uniform_uvw()
    freq = np.array([1.4e9])
    plan_clumped = make_plan(uvw, freq, (64, 64), pixsize, pixsize, epsilon=1e-6)
    plan_uniform = make_plan(uvw_uniform, freq, (64, 64), pixsize, pixsize, epsilon=1e-6)

    width = plan_clumped.w_kernel_width
    assert plan_clumped.n_w > 3 * width, (
        f"pixsize={pixsize} puts the clumped plan at n_w={plan_clumped.n_w}, "
        f"below the 3W={3 * width} crossover -- the ordering asserted below is "
        "not an invariant there"
    )
    assert plan_uniform.n_w > 3 * width

    # The live incidence count is ~n_rows * W for both, so the ordering is
    # entirely a statement about the widest window. Pin that so a failure
    # says which half moved.
    for plan in (plan_clumped, plan_uniform):
        nominal = plan.n_rows * plan.w_kernel_width
        assert abs(plan.live_row_count - nominal) <= plan.n_w

    assert plan_clumped.window_padding_overhead > plan_uniform.window_padding_overhead

    # The clumped plan is the one with dead planes; the uniform one has none
    # at these resolutions. Under the pre-#43 definition both read zero,
    # because the +/-1 clamp gives every window at least one row.
    assert plan_clumped.empty_plane_count > 0
    assert plan_uniform.empty_plane_count == 0


def test_plan_sample_consistency() -> None:
    """Cross-check between the kernel scale and the spec's x0 oversampling rule."""
    uvw = _baseline_uvw(n_rows=500, max_baseline=400.0)
    freq = np.array([1.4e9])
    plan = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(256, 256),
        pixsize_l=2e-4,
        pixsize_m=2e-4,
        epsilon=1e-6,
    )
    # dw * max|nm1| / x0 should be (close to) the inner w-plane count.
    w_lambda = uvw * (freq[0] / SPEED_OF_LIGHT)
    w_extent = float(np.max(w_lambda[:, 2]) - np.min(w_lambda[:, 2]))
    inner = plan.n_w - plan.w_kernel_width
    if inner > 0:
        dw = w_extent / inner
        # issue #16: the plane-spacing denominator is max|n-1+nshift|, i.e.
        # the *shifted* grid -- reading plan.n_minus_1 here would measure the
        # pre-nshift quantity and land a factor of ~2 off. Same assertion,
        # same +/-1 band; only the leaf it is read from moved.
        max_nm1 = float(np.max(np.abs(np.asarray(plan.n_minus_1_shifted))))
        # Sampling: inner ~ ceil(w_extent * max|nm1| / x0) with the v0.1.1
        # W-independent x0 = W_OVERSAMPLE_X0.
        oversamp_check = w_extent * max_nm1 / W_OVERSAMPLE_X0
        # Allow ceil rounding plus a small margin.
        assert oversamp_check <= inner + 1
        assert oversamp_check >= inner - 1
        # And the kernel half-width matches dw * W/2.
        assert plan.w_kernel_scale == pytest.approx(dw * plan.w_kernel_width / 2.0)
        # And eta_max sits at x0 * W / 2 = W * W_OVERSAMPLE_X0 / 2.
        eta_max = max_nm1 * plan.w_kernel_scale
        assert eta_max <= (W_OVERSAMPLE_X0 * plan.w_kernel_width / 2.0) + 1e-9


# ---------------------------------------------------------------------------
# issue #23 (M2/M5/R9): plan memory footprint
# ---------------------------------------------------------------------------


def _footprint_fixture_uvw(n_rows: int, max_baseline: float, seed: int = 0) -> np.ndarray:
    """A generic, non-degenerate uvw distribution with real w-extent (so the
    fixture exercises the generic multi-plane path, not the constant-w fast
    path), reused by the footprint and host-RSS tests below."""
    rng = np.random.default_rng(seed)
    uvw = rng.normal(scale=max_baseline / 3, size=(n_rows, 3))
    norms = np.linalg.norm(uvw, axis=1, keepdims=True)
    return uvw / np.maximum(norms / max_baseline, 1.0)


def _footprint_bound_bytes(n_chan: int, n_rows: int, n_w: int, n_l: int, n_m: int) -> int:
    """Upper bound on the summed ``nbytes`` of every ``WGridderPlan`` pytree
    leaf, post issue #23, for a float64 plan.

    Derived leaf by leaf from ``_EXPECTED_LEAF_FIELDS`` (real_dtype=float64,
    8 B; sort_perm/window_start are int32, 4 B; w0_screen is complex128,
    16 B):

        uvw_m               (n_rows, 3)    float64     24 * n_rows
        inv_lambda          (n_chan,)      float64      8 * n_chan
        w_centers_rel       (n_w,)         float64      8 * n_w
        n_minus_1_shifted   (n_l, n_m)     float64      8 * n_l * n_m
        w0_screen           (n_l, n_m)     complex128  16 * n_l * n_m
        phi_hat_n           (n_l, n_m)     float64      8 * n_l * n_m
        sort_perm           (n_rows,)      int32        4 * n_rows
        window_start        (n_chan, n_w)  int32        4 * n_chan * n_w
                                                        -------------------
        total = 28*n_rows + 8*n_chan + 8*n_w + 32*n_l*n_m + 4*n_chan*n_w

    This is an *exact* target, not a loose ceiling: every term above is
    achieved by exactly one leaf, so there is no slack for a removed leaf
    (or a redundant duplicate of n_minus_1/w_centers) to hide in -- the
    smallest of the leaves this issue removes (window_start's discarded
    twin, window_size, at 4*n_chan*n_w bytes) alone is enough to push the
    total over this bound for any fixture with n_w > 0.

    This corrects the issue body's own formula (``28*n_rows + 8*n_chan +
    4*n_chan*n_w + 2*16*n_pix**2``), which predates issue #16: #16 added
    ``n_minus_1_shifted`` (a second (n_l, n_m) float64 array, 8 B/pixel) and
    ``w0_screen`` (a NEW (n_l, n_m) complex128 phase-screen array, 16 B/pixel)
    on top of the pre-#16 ``n_minus_1`` + ``phi_hat_n`` pair. Before removing
    the n_minus_1/n_minus_1_shifted redundancy that leaves 3 real arrays (24
    B/pixel) + 1 complex array (16 B/pixel) = 40 B/pixel, not the issue's
    assumed 32; after removing it (this issue's own extension of the
    original scope, see _EXPECTED_LEAF_FIELDS's comment) it is 2 real (16
    B/pixel) + 1 complex (16 B/pixel) = 32 B/pixel -- 32 * n_l * n_m above,
    which happens to numerically match the issue's stale constant even
    though the leaf set it was computed from is different.
    """
    return 28 * n_rows + 8 * n_chan + 8 * n_w + 32 * n_l * n_m + 4 * n_chan * n_w


@requires_x64
def test_plan_footprint() -> None:
    """Sum of leaf ``nbytes`` for a 16-channel, 10k-row plan must be at or
    under the issue #23 target (``_footprint_bound_bytes``) -- tight enough
    that leaving *any* single removed leaf in place (``uvw_lambda``,
    ``uvw_lambda_sorted``, ``u_finufft``, ``v_finufft``, ``window_size``, or
    a redundant ``n_minus_1`` / ``w_centers`` duplicate) fails it.

    This exact fixture (n_chan=16, n_rows=10_000, image 256x256, MWA_extended
    pixsize, eps=1e-6, seed=0) is the PR's reported before/after number; see
    the test-writing report for the measured before value.
    """
    n_rows = 10_000
    n_chan = 16
    uvw = _footprint_fixture_uvw(n_rows, max_baseline=5000.0, seed=0)
    freq = np.linspace(140e6, 160e6, n_chan)
    plan = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(256, 256),
        pixsize_l=MWA_EXTENDED.pixsize,
        pixsize_m=MWA_EXTENDED.pixsize,
        epsilon=1e-6,
    )

    leaves = jax.tree_util.tree_leaves(plan)
    total_bytes = sum(np.asarray(leaf).nbytes for leaf in leaves)
    bound = _footprint_bound_bytes(plan.n_chan, plan.n_rows, plan.n_w, plan.n_l, plan.n_m)

    assert total_bytes <= bound, (
        f"plan leaf footprint {total_bytes} B ({total_bytes / 1e6:.3f} MB) exceeds the issue "
        f"#23 target of {bound} B ({bound / 1e6:.3f} MB) for n_chan={plan.n_chan}, "
        f"n_rows={plan.n_rows}, n_w={plan.n_w}, image={plan.n_l}x{plan.n_m} -- a removed leaf "
        "(uvw_lambda / uvw_lambda_sorted / u_finufft / v_finufft / window_size) or a redundant "
        "n_minus_1 / w_centers duplicate is still a pytree leaf on this plan"
    )


# ---------------------------------------------------------------------------
# issue #23: make_plan's *host* cost must scale with n_rows + n_chan, not with
# n_chan * n_rows
# ---------------------------------------------------------------------------
#
# ``test_plan_footprint`` above gates the plan's device leaves. This gates the
# host side: before issue #23 ``make_plan`` built ``uvw_lambda``,
# ``uvw_lambda_sorted``, ``u_finufft`` and ``v_finufft`` as transient numpy
# arrays -- 64 B per (channel, row) -- on top of the leaves it stored.
#
# The instrument is a *difference*, not an absolute peak, because an absolute
# peak cannot gate this property. ``make_plan``'s largest single allocation is
# the zero-padded FFT inside ``compute_phi_hat_table``, and its size depends
# only on ``epsilon``: at eps=1e-6 the table is n_fft = 4096 * 64 = 262144
# points, so the FFT input is 2.1 MB and its complex output 4.2 MB, and the
# call peaks near 14.7 MB of traced allocation -- identically so for a 32x32,
# 200-row, 2-channel plan whose leaves total 50 KB. Any absolute bound tight
# enough to catch a per-(channel, row) array is therefore already blown by the
# kernel table, and any bound loose enough to admit the table gates nothing.
#
# So build two plans that differ in *nothing but* ``n_chan`` -- same epsilon
# (hence the same kernel table and the same n_w), same image shape, same rows,
# same frequency endpoints -- and measure the gap. The table, the image-sized
# arrays, the uvw input and the interpreter baseline are identical in both and
# subtract out; what is left is exactly the quantity this issue changed.
#
# Two deliberate choices about *how* it is measured:
#
#   * ``tracemalloc``, not process RSS. RSS answers "did the allocator ask the
#     OS for new pages", which depends on whether previously freed pages of the
#     right size happen to still be held -- so the same code measures 0.03 MB
#     or 22 MB on consecutive calls in one process, and a gate built on it goes
#     intermittently red on a runner with different allocator behaviour.
#     ``tracemalloc`` counts bytes *requested*, including numpy's data
#     allocations (numpy registers them with tracemalloc), and reproduces to
#     within about 1 KB run to run.
#   * one plan per fresh interpreter. Both probes then pay identical one-time
#     costs (JAX import, numpy's FFT twiddle cache for this n_fft), which is
#     what lets them cancel; measuring both in one process would charge those
#     to whichever ran first.
#
# Measured with this probe (float64, eps=1e-6, 100k rows, 256^2, n_w=528):
#
#     n_chan            2          32        delta
#     pre-#23     29.57 MB   240.72 MB   211.15 MB
#     post-#23    14.66 MB    14.66 MB      ~900 B
#
# i.e. the pre-#23 gap is the 64 B per (channel, row) this issue removed, and
# the post-#23 gap is a few hundred bytes of per-channel scalars.

_HOST_ALLOC_PROBE = """
import sys
import tracemalloc

n_chan, n_rows, n_pix = (int(a) for a in sys.argv[1:4])
pixsize = float(sys.argv[4])
epsilon = float(sys.argv[5])

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

import jax_nufft
from jax_nufft.planning import make_plan

# The same fixture as _footprint_fixture_uvw (seed 0, max_baseline 5000 m),
# written out here rather than imported: this runs in a bare interpreter, and
# importing the test module would drag pytest and this whole file into it.
max_baseline = 5000.0
rng = np.random.default_rng(0)
uvw = rng.normal(scale=max_baseline / 3, size=(n_rows, 3))
uvw = uvw / np.maximum(np.linalg.norm(uvw, axis=1, keepdims=True) / max_baseline, 1.0)
freq = np.linspace(140e6, 160e6, n_chan)

# Start tracing *after* the inputs exist: uvw and freq are the caller's data,
# not make_plan's cost, and uvw is n_chan-independent anyway.
tracemalloc.start()
plan = make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, epsilon)
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()

print(peak, plan.n_w, jax_nufft.__file__)
"""


def _peak_host_alloc_bytes(
    n_chan: int, n_rows: int, n_pix: int, pixsize: float, epsilon: float
) -> tuple[int, int]:
    """Run one ``make_plan`` in a fresh interpreter; return ``(peak_bytes, n_w)``.

    The probe reports the ``jax_nufft.__file__`` it resolved and this asserts it
    against the parent's. Setting ``cwd`` to the repository root is *not* what
    makes the import land on the right copy -- there is no ``jax_nufft/``
    directory at the root, so the import succeeds through the editable install's
    ``.pth`` entry, and a non-editable install of some other version would be
    measured just as happily and pass. Only comparing the resolved paths rules
    that out.

    A caveat on the instrument, since it bounds what this can catch:
    ``tracemalloc``'s peak is a *maximum* over the traced window, so a
    per-(channel, row) transient that is allocated and freed entirely before
    ``compute_phi_hat_table``'s FFT plateau would not raise it and would go
    unseen. What the difference measures is peak-to-peak, which is what the
    memory budget is about; it is not a leak detector.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _HOST_ALLOC_PROBE,
            str(n_chan),
            str(n_rows),
            str(n_pix),
            repr(pixsize),
            repr(epsilon),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"host-allocation probe failed for n_chan={n_chan}:\n{completed.stdout}\n{completed.stderr}"
    )
    peak_str, n_w_str, probe_module = completed.stdout.split()[-3:]
    expected_module = str(pathlib.Path(jax_nufft.__file__).resolve())
    assert str(pathlib.Path(probe_module).resolve()) == expected_module, (
        f"the probe imported jax_nufft from {probe_module!r}, but this test process has it "
        f"at {expected_module!r} -- the subprocess is measuring a different installation, "
        "so its numbers say nothing about the code under test"
    )
    return int(peak_str), int(n_w_str)


def test_make_plan_host_cost_is_independent_of_n_chan_times_n_rows(
    pytestconfig: pytest.Config,
) -> None:
    """issue #23: doubling the channel count must not cost ``n_rows`` of host
    memory per added channel.

    The two plans differ only in ``n_chan`` (2 vs 32), so the bound is written
    in terms of what may *legitimately* scale with the channel count after this
    issue: the ``(n_chan, n_w)`` int32 window tables (``window_start``, its
    plan-time ``window_size`` twin, and any staging copy of either), plus a few
    scalars per channel. Nothing proportional to ``n_chan * n_rows`` is
    allowed, which is the whole point -- the pre-#23 code exceeds this bound by
    two orders of magnitude (see the table above).

    Marked slow (needs ``--runslow``): it spawns two interpreters and builds a
    100k-row plan in each.
    """
    if not pytestconfig.getoption("--runslow"):
        pytest.skip("needs --runslow")

    n_rows = 100_000
    n_pix = 256
    epsilon = 1e-6
    n_chan_lo, n_chan_hi = 2, 32

    lo_peak, lo_n_w = _peak_host_alloc_bytes(
        n_chan_lo, n_rows, n_pix, MWA_EXTENDED.pixsize, epsilon
    )
    hi_peak, hi_n_w = _peak_host_alloc_bytes(
        n_chan_hi, n_rows, n_pix, MWA_EXTENDED.pixsize, epsilon
    )
    # Same epsilon and the same frequency endpoints (np.linspace keeps 140 and
    # 160 MHz whatever the count), so the w-extent, the plane spacing and n_w
    # are identical. If they are not, the two plans differ in more than n_chan
    # and the difference below is not measuring what it claims to.
    assert lo_n_w == hi_n_w, (
        f"the two probe plans must differ only in n_chan, but n_w is {lo_n_w} at "
        f"n_chan={n_chan_lo} and {hi_n_w} at n_chan={n_chan_hi}"
    )

    delta_chan = n_chan_hi - n_chan_lo
    # Per added channel: four (n_chan, n_w) int32 tables' worth of headroom
    # (twice what make_plan actually builds) plus 64 B of scalars.
    per_channel_bound = 16 * hi_n_w + 64
    # x4 on top of that, plus a flat 512 KB, to absorb interpreter-level noise
    # -- still ~140x below the pre-#23 gap.
    bound = 4 * delta_chan * per_channel_bound + 512 * 1024
    # What n_chan * n_rows scaling costs: 8 float64 per (channel, row), the
    # uvw_lambda / uvw_lambda_sorted / u_finufft / v_finufft set issue #23
    # removed.
    naive = 64 * n_rows * delta_chan

    delta = hi_peak - lo_peak
    assert delta <= bound, (
        f"make_plan's peak host allocation grew by {delta} B ({delta / 1e6:.2f} MB) going from "
        f"n_chan={n_chan_lo} to n_chan={n_chan_hi} at n_rows={n_rows}, above the "
        f"{bound} B ({bound / 1e6:.2f} MB) allowed for per-channel bookkeeping. Something in "
        f"make_plan is again allocating per (channel, row): n_chan * n_rows scaling would cost "
        f"about {naive / 1e6:.0f} MB here, and the measured growth is "
        f"{100.0 * delta / naive:.1f}% of that"
    )


# ---------------------------------------------------------------------------
# issue #23: no operator path may materialise the per-(channel, row) array
# ---------------------------------------------------------------------------
#
# ``plan.uvw_lambda`` is a compatibility accessor: reading it rebuilds the
# ``(n_chan, n_rows, 3)`` array this issue exists to delete -- 3.9 GB at 64
# channels x 1M rows. Until now that rule was prose in AGENTS.md sec 4 and in
# the property's own docstring, and prose is not a gate: reintroducing the read
# in ``_dirty2vis_jit``'s *default* path (dense_scan, channel_strategy="scan")
# leaves the entire suite green, because ``test_plan_footprint`` counts stored
# leaves and the host-cost probe measures ``make_plan``, and neither sees an
# array conjured at call time.
#
# So gate it where it is observable: in the lowered IR. Every strategy pair is
# swept, not just the default, because the four w-strategies and two channel
# strategies reach the coordinates through different code
# (``_channel_forward`` / ``_channel_adjoint`` take ``plan.uvw_m`` directly;
# the windowed helpers take a ``sort_perm`` gather of it), so a regression can
# hide in one and not the others.


_UVW_LAMBDA_PROBE_N_CHAN = 5
_UVW_LAMBDA_PROBE_N_ROWS = 257
_UVW_LAMBDA_PROBE_N_PIX = 16

# StableHLO tensor types: ``tensor<5x257xf64>``, ``tensor<257x3xcomplex<f64>>``.
_TENSOR_TYPE_RE = re.compile(r"tensor<([0-9]+(?:x[0-9]+)*)x(complex<[a-z0-9]+>|[a-z][a-z0-9]*)>")


def _lowered_tensors(text: str) -> list[tuple[tuple[int, ...], str]]:
    """Every ``tensor<...>`` in a StableHLO module, as ``(dims, element type)``.

    Parsed structurally rather than matched as a substring so the check does not
    depend on the element type or on whichever dimension order the compiler
    happened to pick.
    """
    out: list[tuple[tuple[int, ...], str]] = []
    for dims, elem in _TENSOR_TYPE_RE.findall(text):
        out.append((tuple(int(d) for d in dims.split("x")), elem))
    return out


@requires_x64
@pytest.mark.parametrize("channel_strategy", ["scan", "vmap"])
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_no_operator_path_materialises_per_channel_row_coordinates(
    w_strategy: str, channel_strategy: str
) -> None:
    """No operator may build a *real* tensor of order ``n_chan * n_rows``.

    The ban is on element membership, not on a literal shape, because issue #23
    deleted four leaves of two different ranks: ``uvw_lambda`` and
    ``uvw_lambda_sorted`` at ``(n_chan, n_rows, 3)``, and ``u_finufft`` /
    ``v_finufft`` at ``(n_chan, n_rows)``. The rank-2 pair is half the 64 bytes
    per (channel, row) this issue removes -- about 1.0 GB of transient at 64
    channels x 1M rows -- so a gate that only looks for the rank-3 form misses
    half the regression, and misses it silently: reintroducing ``u_finufft`` /
    ``v_finufft`` live leaves the whole suite green while adding 56% to the
    lowered temp size on a 16-channel, 20k-row problem. Flagging any tensor
    whose dimensions include both ``n_chan`` and ``n_rows`` catches ``(5, 257)``,
    ``(257, 5)``, ``(257, 5, 3)`` and the rank-3 form alike.

    The rule has two clauses, because two different things have to be caught.
    Dimension membership catches every array that wears the shape -- ``(5, 257)``,
    ``(257, 5)``, ``(257, 5, 3)``. Element count catches the ones that do not:
    the flat carrier ``(n_chan * n_rows,)`` and the flat cube
    ``(n_chan * n_rows, 3)``, which is what a single-FINUFFT-call-over-all-
    channels rewrite builds and which contains neither ``n_chan`` nor
    ``n_rows`` as a dimension at all. Both clauses are checked on the *squeezed*
    shape so a broadcast axis cannot disguise either.

    Restricted to *real* element types, and that restriction is load-bearing
    rather than incidental: the visibility cube genuinely is ``n_chan * n_rows``
    and legitimately appears as ``(257, 5)`` and ``(5, 257)`` complex tensors in
    both operators. Coordinates are real, visibilities are complex, so the
    element type separates them exactly. The one real array of that order in the
    API is ``weights``, which is why this probe leaves it at ``None``; a
    weighted call would legitimately show ``(257, 5)`` in f64.

    ``n_rows = 257`` and ``n_chan = 5`` are both unlike every other dimension in
    the problem -- a 16x16 image, the coordinate axis of 3, ``n_w`` in the tens
    -- so a tensor carrying either can only have come from the per-row
    baselines or the per-channel scaling, and ``n_chan`` cannot be confused with
    the length-3 coordinate axis.

    ``channel_strategy="vmap"`` is held to the dimension-membership clause only
    -- the element-count clause is scan-only -- and that is a real, stated
    weakening: a *flattened* carrier is not caught under vmap. It is not a hole
    being papered over, though. ``jax.vmap`` over the channel axis
    *is* the request to batch the per-channel work, so it lifts
    ``_channel_ft_coords``'s three ``(n_rows,)`` outputs to ``(n_chan, n_rows)``
    by construction; pairing it with a ``*_vmap`` w-strategy lifts the kernel
    weights to ``(n_chan, n_w, n_rows)`` for the same reason. No implementation
    of those strategies can avoid either, and AGENTS.md sec 5 already prices
    them as "allocates n_chan x per-channel transient memory". What no strategy
    ever needs is the array that *also* carries the length-3 baseline axis, so
    the shaped cube stays banned in all eight; its flattened form is caught
    under ``"scan"`` only. The default ``"scan"`` path -- the one the memory
    argument in issue #23 is about -- is held to the full rule.
    """
    n_chan = _UVW_LAMBDA_PROBE_N_CHAN
    n_rows = _UVW_LAMBDA_PROBE_N_ROWS
    n_pix = _UVW_LAMBDA_PROBE_N_PIX

    rng = np.random.default_rng(23)
    uvw = rng.normal(scale=200.0, size=(n_rows, 3))
    freq = np.linspace(1.0e9, 2.0e9, n_chan)
    plan = make_plan(uvw, freq, (n_pix, n_pix), 2e-3, 2e-3, epsilon=1e-6)
    image = jnp.asarray(rng.standard_normal((n_chan, n_pix, n_pix)))
    vis = jnp.asarray(
        rng.standard_normal((n_rows, n_chan)) + 1j * rng.standard_normal((n_rows, n_chan))
    )

    for name, fn, arg in (("dirty2vis", dirty2vis, image), ("vis2dirty", vis2dirty, vis)):
        lowered = jax.jit(
            lambda p, x, _fn=fn: _fn(
                p, x, w_strategy=w_strategy, channel_strategy=channel_strategy, nthreads=1
            )
        ).lower(plan, arg)
        tensors = _lowered_tensors(lowered.as_text())
        # Sanity: the baselines themselves must be in there, or this is looking
        # at the wrong module and every assertion below passes vacuously.
        assert any(sorted(dims) == sorted((n_rows, 3)) for dims, _ in tensors), (
            f"{name} ({w_strategy}, {channel_strategy}): no (n_rows, 3) tensor in the "
            "lowered IR at all -- the probe is not seeing the operator it thinks it is"
        )

        def banned(dims: tuple[int, ...], elem: str) -> bool:
            if elem.startswith("complex"):
                return False  # the visibility cube; see the docstring
            # Squeeze degenerate axes: vmap leaves the batched coordinates as
            # ``(n_chan, 1, n_rows)``, which is the rank-2 form wearing a
            # broadcast axis.
            squeezed = [d for d in dims if d != 1]
            # Element count, not just dimension membership. A carrier that never
            # takes the (n_chan, n_rows) *shape* -- what a future "concatenate
            # every channel's points into one FINUFFT call" rewrite builds, by
            # gather, so no rank-2 intermediate ever exists -- costs exactly the
            # same memory and walks straight through a membership test.
            #
            # A multiple, not equality, because the flattened *cube* is
            # ``(n_chan * n_rows, 3)``: three times the element count, and
            # ``(1285, 3)`` contains neither 5 nor 257, so both the earlier
            # clauses miss it. Measured on that mutant: all eight
            # parametrisations passed while ``temp_size_in_bytes`` went from
            # 6.437 MB to 20.546 MB, +219%, on a 16-channel 20k-row fixture. A
            # quotient cap of 3 covers the flat form, the two-array u/v split
            # and the cube, and stops short of anything large enough to be a
            # legitimately per-plane or per-pixel array.
            elements = math.prod(squeezed)
            if (
                channel_strategy == "scan"
                and elements % (n_chan * n_rows) == 0
                and elements // (n_chan * n_rows) <= 3
            ):
                return True
            if not (n_chan in squeezed and n_rows in squeezed):
                return False
            if channel_strategy == "scan":
                return True
            # Under channel vmap the rank-2 batched coordinates are inherent,
            # and so is ``(n_chan, n_w, n_rows)`` when the w-plane loop is
            # vmapped too -- that is the kernel-weight array, which dense_vmap
            # is defined to materialise. The coordinate *cube* is not inherent
            # to anything, so it stays banned: it is the one that also carries
            # the length-3 baseline axis.
            return 3 in squeezed

        offenders = [(dims, elem) for dims, elem in tensors if banned(dims, elem)]
        assert not offenders, (
            f"{name} ({w_strategy}, channel_strategy={channel_strategy}) materialises "
            f"tensor<{'x'.join(str(d) for d in offenders[0][0])}x{offenders[0][1]}> -- a real "
            f"array of order n_chan * n_rows. That is a per-(channel, row) coordinate array "
            "(plan.uvw_lambda, or a revived u_finufft / v_finufft, or an equivalent broadcast "
            "of plan.uvw_m by plan.inv_lambda) being built in an operator path: up to 3.9 GB "
            "at 64 channels x 1M rows, and the whole point of issue #23. Read uvw_m and "
            "inv_lambda and derive per channel via _channel_ft_coords instead"
        )
