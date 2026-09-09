"""Tests for plan construction (Nw, w-plane centres, kernel correction)."""

from __future__ import annotations

import dataclasses
import decimal
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
from jax.typing import DTypeLike

import jax_nufft
from jax_nufft import dirty2vis, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
from jax_nufft.kernel import kernel_params
from jax_nufft.planning import (
    MAX_WINDOW_BUCKETS,
    W_OVERSAMPLE_X0,
    WGridderPlan,
    _n_minus_1_grid,
    bucket_window_sizes,
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


def _as_planned_uvw(uvw: np.ndarray, plan: WGridderPlan) -> np.ndarray:
    """The baselines in the orientation ``plan`` planned against.

    issue #17: a ``hermitian=True`` plan folds every row with ``w < 0`` onto
    ``(-u, -v, -w)`` before it computes the w-range, the sort permutation and
    the window boundaries, so every reference in this file that predicts one of
    those from the *raw* ``uvw`` has to fold it the same way first. Recomputed
    from the raw input and the plan's static ``hermitian`` flag -- deliberately
    NOT read off ``plan.uvw_m``, which is the array several of those references
    exist to check.

    Written so this file says the same thing whichever way ``make_plan``'s
    ``hermitian`` default is set. The fold's own contract (the sign rule, the
    per-row byte cost, the plane-count saving, the conjugation) is gated in
    ``tests/test_hermitian.py``; here it is only a coordinate convention the
    existing plan-structure references have to respect.
    """
    if not plan.hermitian:
        return np.asarray(uvw)
    sign = np.where(np.asarray(uvw)[:, 2] < 0, -1.0, 1.0)
    return np.asarray(uvw) * sign[:, None]


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
    # ``_as_planned_uvw``: ``plan.uvw_lambda`` broadcasts the baselines the plan
    # STORED, which issue #17's fold may have sign-flipped per row.
    expected_uvw_lambda = _as_planned_uvw(uvw, plan)[None, :, :] * (
        freq[:, None, None] / SPEED_OF_LIGHT
    )
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
    w_lambda = _as_planned_uvw(uvw, plan)[None, :, :] * (freq[:, None, None] / SPEED_OF_LIGHT)
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

    The inside-disc branch is the cancellation-free ``-r2 / (sqrt(1 - r2) + 1)``
    rather than ``sqrt(1 - r2) - 1``: see
    ``tests/conftest.py::reference_lmn_grids`` for why an oracle may not carry
    the ``ulp(1)/2`` absolute error that issue #12 removed from the operator.
    Independence of *convention* was never meant to mean a different *rounding
    error*. (``_nm1_naive_pre_issue_12`` below is the one deliberate exception:
    it is a frozen historical record, not an oracle.)
    """
    n_l, n_m = image_shape
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    ll = (i * pixsize_l)[:, None]
    mm = (j * pixsize_m)[None, :]
    r2 = ll * ll + mm * mm
    inside_disc = r2 <= 1.0
    x = np.where(inside_disc, r2, 0.0)
    inside_val = -x / (np.sqrt(1.0 - x) + 1.0)
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


# ---------------------------------------------------------------------------
# issue #12: cancellation-free ``n - 1``
# ---------------------------------------------------------------------------
# ``planning._n_minus_1_grid`` evaluates ``n - 1 = sqrt(1 - l^2 - m^2) - 1``
# inside the unit disc. That subtraction cancels catastrophically as
# ``l^2 + m^2 -> 0``: the phase centre pixel is exactly ``n - 1 = 0`` and its
# neighbours are ``-pixsize^2 / 2``, so for a fine enough pixel the true value
# sits far below ``ulp(1)`` while the formula can only return a multiple of
# ``ulp(1) / 2 = 1.11e-16``. The cancellation-free rewrite is
# ``sqrt(1 - x) - 1 = -x / (sqrt(1 - x) + 1)`` ([Higham2002] Ch. 1), which
# never forms the difference. The analytic extension used *outside* the disc,
# ``n - 1 = -sqrt(l^2 + m^2 - 1) - 1``, has no cancellation and is untouched.
#
# All measurements in this section were taken on 2026-09-08 at ``ef509c5``,
# CPython 3.12 / numpy on macOS arm64 (Apple M-series), against the
# ``decimal``-based oracle below. The float64 arithmetic they characterise is
# IEEE-754 correctly-rounded ``+ - * /`` and ``sqrt`` throughout, so the
# numbers are platform-independent even though the machine is not.

# The exact oracle's working precision, in significant decimal digits. Sixty is
# ~44 digits past float64, so the oracle contributes nothing measurable to any
# error below; it is not a tuned number.
_NM1_ORACLE_PREC = 60


def _nm1_exact(n_l: int, n_m: int, pixsize_l: float, pixsize_m: float) -> np.ndarray:
    """``n - 1`` on the image grid to ~60 significant digits, as ``Decimal``.

    **Not** ``numpy.longdouble``, which is what issue #12's definition of done
    names. ``np.longdouble`` is 80-bit extended on Linux x86-64 and 128-bit on
    Linux aarch64, but it is a plain **alias for float64 on macOS arm64** --
    measured here: ``np.finfo(np.longdouble).eps == 2.220446049250313e-16``,
    bit-for-bit ``np.finfo(np.float64).eps``. On that platform a longdouble
    "oracle" for this quantity is the cancellation-free float64 form itself, so
    it scores that form a perfect 0.0 by construction and the gate below would
    be checking the implementation against a copy of itself. ``decimal`` is
    stdlib, exact by construction, and the same on every platform.

    The grid convention is restated here rather than imported (this
    repository's convention -- see ``_nm1_extremes`` above and
    ``tests/conftest.py::reference_lmn_grids``), including the ``// 2``
    floor-division pixel-centre offset that issue #14 pinned.

    ``Decimal(float)`` is exact, so the pixel coordinates ``(i - n_l // 2) *
    pixsize`` are the exact real products of an exact integer and the exact
    binary value of ``pixsize`` -- deliberately *not* the float64-rounded
    coordinate the implementation forms. Rounding of the coordinate itself is
    therefore part of what is measured, which is why the wide-field absolute
    errors below (1.6e-15 on EDA2) are an order of magnitude above the
    ``ulp(1)``-scale errors of the narrow-field grids: on EDA2 ``l`` reaches
    1.05 and ``ulp(l)`` is already 2.2e-16 before any square root is taken.

    **Every** arithmetic operation below goes through ``ctx`` explicitly,
    including the negations. A bare unary ``-`` on a ``Decimal`` is a
    context-sensitive operation performed under the *ambient thread* context,
    not under ``ctx``: measured 2026-09-09, ``len(r2.as_tuple().digits)`` is
    60 while ``len((-r2).as_tuple().digits)`` is 28 -- ``decimal``'s default
    ``prec``. Written that way this "60-digit" oracle ran at 28 digits (still
    12 past float64, so no assertion here ever moved, but it silently inherited
    a global that a ``decimal.setcontext`` anywhere in the process could
    rescale). Checked against an independent exact ``fractions.Fraction``
    oracle (exact rational pixel coordinates, square roots by ``math.isqrt``
    scaled by 10^200) on the EDA2, MeerKAT and 1e-8 grids: with the bare
    unary minus the relative residual is 2.95e-28 to 4.41e-28, with
    ``ctx.minus`` it is 8.6e-60 to 1.6e-59, i.e. at the declared working
    precision.
    """
    ctx = decimal.Context(prec=_NM1_ORACLE_PREC)
    one = decimal.Decimal(1)
    d_l = decimal.Decimal(pixsize_l)
    d_m = decimal.Decimal(pixsize_m)
    out = np.empty((n_l, n_m), dtype=object)
    for a in range(n_l):
        ll = ctx.multiply(decimal.Decimal(a - n_l // 2), d_l)
        l2 = ctx.multiply(ll, ll)
        for b in range(n_m):
            mm = ctx.multiply(decimal.Decimal(b - n_m // 2), d_m)
            r2 = ctx.add(l2, ctx.multiply(mm, mm))
            if r2 <= one:
                # Cancellation-free by construction: the oracle may not be
                # written in the form under test.
                out[a, b] = ctx.divide(ctx.minus(r2), ctx.add(ctx.sqrt(ctx.subtract(one, r2)), one))
            else:
                out[a, b] = ctx.subtract(ctx.minus(ctx.sqrt(ctx.subtract(r2, one))), one)
    return out


def _nm1_naive_pre_issue_12(n_l: int, n_m: int, pixsize_l: float, pixsize_m: float) -> np.ndarray:
    """``n - 1`` by the pre-issue-#12 arithmetic, ``sqrt(1 - l^2 - m^2) - 1``, float64.

    **Frozen on purpose.** This is a historical record of the formula
    ``planning._n_minus_1_grid`` used before issue #12, not a second statement
    of the current convention, and it must NOT be updated when ``planning``
    changes. It is the one deliberate exception to this repository's rule that
    an oracle restates the convention independently: every *other* ``n - 1``
    under ``tests/`` was moved to the cancellation-free spelling with the
    operator (see ``tests/conftest.py::reference_lmn_grids``). **Four** tests
    below need it, and all four go quiet, not red, if it is "kept in sync":

    * ``test_nm1_relative_accuracy_across_pixel_scales`` needs a witness that a
      given ``pixsize`` really does trigger the cancellation, rather than
      assuming it from a docstring; synced, six of its seven rows fail on
      ``naive_clears_gate`` and the EDA2 control row passes;
    * ``test_nm1_absolute_error_stays_at_ulp_scale``'s naive arm becomes a
      duplicate of its live arm and passes;
    * ``test_nm1_outside_disc_branch_is_bit_identical`` compares the untouched
      outside branch, which is bit-identical either way, and passes;
    * ``test_plan_n_minus_1_does_not_move_at_imaging_pixel_scales`` -- the cell
      that carries this PR's central claim -- becomes
      ``|plan.n_minus_1 - plan.n_minus_1|`` and passes as a **tautology**.

    Measured 2026-09-09 by mutating this helper's inside branch to the stable
    form and running ``tests/test_planning.py``: **11 failed** -- the direct
    freeze cell, all seven rows of the pixel-scale cell (including the EDA2
    control row, which the old one-sided ``pytest.approx(v, rel=1.0)`` let
    through), both rows of the exact-zero cell, and the downstream w-phase
    cell's anti-vacuity assertion. As the section was first written the same
    mutation cost 6 failures, all in one cell and none of them in the cell that
    most needs the freeze.
    """
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    ll = (i * pixsize_l)[:, None]
    mm = (j * pixsize_m)[None, :]
    r2 = ll * ll + mm * mm
    inside_disc = r2 <= 1.0
    inside_val = np.sqrt(np.where(inside_disc, 1.0 - r2, 0.0)) - 1.0
    outside_val = -np.sqrt(np.where(inside_disc, 0.0, r2 - 1.0)) - 1.0
    return np.where(inside_disc, inside_val, outside_val)


def _nm1_horizon_gap(n_l: int, n_m: int, pixsize_l: float, pixsize_m: float) -> float:
    """``min |1 - (l^2 + m^2)|`` over the grid -- how close a pixel gets to the horizon."""
    return float(
        np.min(
            np.abs(
                1.0
                - (
                    ((np.arange(n_l) - n_l // 2) * pixsize_l)[:, None] ** 2
                    + ((np.arange(n_m) - n_m // 2) * pixsize_m)[None, :] ** 2
                )
            )
        )
    )


def _nm1_inside_disc(n_l: int, n_m: int, pixsize_l: float, pixsize_m: float) -> np.ndarray:
    """The ``l^2 + m^2 <= 1`` mask on the same grid."""
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    r2 = ((i * pixsize_l)[:, None]) ** 2 + ((j * pixsize_m)[None, :]) ** 2
    return r2 <= 1.0


def _nm1_max_ulp_error(got: np.ndarray, exact: np.ndarray) -> float:
    """Max over pixels of ``|got - exact| / ulp(|exact|)``, i.e. distance from correctly rounded.

    The elementwise companion to :func:`_nm1_errors_vs_exact`'s two whole-grid
    numbers. A whole-grid *absolute* bound is blind to a defect proportional to
    the pixel's own value, and a whole-grid *relative* bound is dominated by
    whichever pixel is smallest; this is the per-pixel statement, in the only
    unit in which "as good as float64 allows" means anything.

    The phase-centre pixel is skipped: ``exact`` is zero there, ``ulp(0)`` is
    the smallest subnormal, and every implementation returns exactly +-0.0, so
    the ratio is either 0/tiny or undefined.
    """
    worst = decimal.Decimal(0)
    n_l, n_m = exact.shape
    with decimal.localcontext(decimal.Context(prec=_NM1_ORACLE_PREC)):
        for a in range(n_l):
            for b in range(n_m):
                ref = exact[a, b]
                if ref == 0:
                    continue
                ulp = decimal.Decimal(float(np.spacing(abs(float(ref)))))
                worst = max(worst, abs(decimal.Decimal(float(got[a, b])) - ref) / ulp)
    return float(worst)


def _nm1_errors_vs_exact(got: np.ndarray, exact: np.ndarray) -> tuple[float, float]:
    """``(max absolute error, max relative error)`` of ``got`` against the oracle.

    The relative error skips the phase-centre pixel, where the exact value is
    zero and every implementation returns exactly zero: a relative error is not
    defined there, and including it as ``0/0`` or as an absolute error would
    quietly dilute the maximum.

    The subtraction, ``abs`` and division below are all context-sensitive, so
    the whole loop runs inside ``decimal.localcontext`` at
    ``_NM1_ORACLE_PREC``. Without it they ran at ``decimal``'s ambient default
    of 28 digits (see :func:`_nm1_exact`) and this function would have inherited
    whatever precision an unrelated ``decimal.setcontext`` anywhere in the
    process had left behind -- a gate that rescales itself without failing.
    """
    absmax = decimal.Decimal(0)
    relmax = decimal.Decimal(0)
    n_l, n_m = exact.shape
    with decimal.localcontext(decimal.Context(prec=_NM1_ORACLE_PREC)):
        for a in range(n_l):
            for b in range(n_m):
                ref = exact[a, b]
                err = abs(decimal.Decimal(float(got[a, b])) - ref)
                absmax = max(absmax, err)
                if ref != 0:
                    relmax = max(relmax, err / abs(ref))
    return float(absmax), float(relmax)


# (id, n_l, n_m, pixsize_l, pixsize_m, naive_rel_err, naive_clears_1e_12)
#
# The pixel scale is the axis this issue lives on, and it is the axis every CI
# fixture holds nearly constant: the seven review fixtures span 1.0e-4 to
# 3.3e-2 rad, so a test written on any of them measures at most 1.5e-09. The
# last two rows are VLBI-realistic (1e-6 rad is 0.21 arcsec, 1e-8 rad is 2.1
# mas) and are where the defect is unmissable. Relative errors are the whole
# grid against ``_nm1_exact``, measured 2026-09-08 at ``ef509c5``:
#
#     grid                     pixsize (rad)  innermost |n-1|   naive rel    stable rel
#     EDA2 120 deg FoV / 64      3.2725e-02       5.3560e-04    3.8342e-14    1.5421e-15
#     MWA 25 deg FoV / 128       3.4088e-03       5.8101e-06    2.7877e-12    3.8038e-16
#     (round number)             5.0000e-03       1.2500e-05    4.4372e-12    3.7280e-16
#     MeerKAT 1.5 deg / 256      1.0227e-04       5.2291e-09    1.4634e-09    4.0013e-16
#     odd 63x65, pm = pl / 5     5e-3 / 1e-3      5.0000e-07    6.0167e-11    3.7280e-16
#     (VLBI, 0.21 arcsec)        1.0000e-06       5.0000e-13    8.8901e-05    3.7305e-16
#     (VLBI, 2.1 mas)            1.0000e-08       5.0000e-17    1.2204e+00    3.5479e-16
#
# The naive column is ``2 * ulp(1) / pixsize^2`` to within a factor of two, as
# it must be: the absolute error is pinned at half an ulp of 1.0 (see
# ``test_nm1_absolute_error_stays_at_ulp_scale``) while the innermost value it
# is divided by shrinks as ``pixsize^2 / 2``.
_NM1_PIXEL_SCALES: tuple[tuple[str, int, int, float, float, float, bool], ...] = (
    (
        "EDA2_120deg_pixsize",
        64,
        64,
        math.radians(120.0) / 64,
        math.radians(120.0) / 64,
        3.8342e-14,
        True,
    ),
    (
        "MWA_25deg_pixsize",
        64,
        64,
        math.radians(25.0) / 128,
        math.radians(25.0) / 128,
        2.7877e-12,
        False,
    ),
    ("pixsize_5e-3", 64, 64, 5e-3, 5e-3, 4.4372e-12, False),
    (
        "MeerKAT_1.5deg_pixsize",
        64,
        64,
        math.radians(1.5) / 256,
        math.radians(1.5) / 256,
        1.4634e-09,
        False,
    ),
    ("odd_63x65_anisotropic", 63, 65, 5e-3, 1e-3, 6.0167e-11, False),
    ("pixsize_1e-6_vlbi", 64, 64, 1e-6, 1e-6, 8.8901e-05, False),
    ("pixsize_1e-8_vlbi", 64, 64, 1e-8, 1e-8, 1.2204e00, False),
)

# The gate itself, from issue #12's revised definition of done. The stable form
# measures 3.5e-16 to 1.5e-15 on every row above -- i.e. its relative error is
# at the float64 rounding floor and does *not* grow as the pixel shrinks -- so
# 1e-12 clears it by at least 650x on the worst row and is nowhere near a
# tuned threshold.
_NM1_REL_GATE = 1e-12

# How far every row of ``_NM1_PIXEL_SCALES`` must keep its closest pixel from
# the horizon ``l^2 + m^2 = 1``, where ``n`` is infinitely ill-conditioned in
# ``r2`` and no rewrite of ``n - 1`` can reach ``_NM1_REL_GATE``. Reachability
# is proved, not assumed, by
# ``test_the_horizon_precondition_rejects_the_grid_it_is_meant_to_reject``.
_NM1_HORIZON_GAP_MIN = 1e-3


def test_the_pre_issue_12_helper_is_still_frozen() -> None:
    """``_nm1_naive_pre_issue_12`` must keep the *old* arithmetic, not the new.

    Four cells read that helper and all four go **quiet, not red**, if someone
    "keeps it in sync" with ``planning`` -- see its docstring for the list and
    for what each one degenerates into. The most important of them,
    ``test_plan_n_minus_1_does_not_move_at_imaging_pixel_scales``, becomes
    ``|x - x|`` and passes as a tautology. So the freeze is stated here,
    directly and next to the helper, rather than left to be inferred.

    The two assertions are the two halves of "frozen": the value at one pixel
    where the two formulas provably differ, and whole-grid inequality.
    ``(32, 33)`` on the 1e-8 grid is the sharpest such pixel -- the true value
    is -5.0000000000000005e-17 and the pre-#12 formula can only return a
    multiple of ``ulp(1)/2``, so it returns exactly -1.1102230246251565e-16
    (measured 2026-09-09).
    """
    frozen = _nm1_naive_pre_issue_12(64, 64, 1e-8, 1e-8)
    assert frozen[32, 33] == -1.1102230246251565e-16, (
        f"_nm1_naive_pre_issue_12 returned {frozen[32, 33]!r} at (32, 33) on the 1e-8 "
        f"grid, not the frozen -1.1102230246251565e-16 (= -ulp(1)/2, the cancelled "
        f"value). This helper is a historical record of pre-issue-#12 arithmetic and "
        f"must not be updated when planning changes"
    )
    assert not np.array_equal(frozen, _n_minus_1_grid(64, 64, 1e-8, 1e-8)), (
        "_nm1_naive_pre_issue_12 now agrees bit-for-bit with _n_minus_1_grid; four "
        "cells in this section silently stop testing anything, including "
        "test_plan_n_minus_1_does_not_move_at_imaging_pixel_scales, which becomes a "
        "tautology"
    )


def test_nm1_is_exactly_minus_one_on_the_exact_horizon() -> None:
    """A pixel landing exactly on ``l^2 + m^2 == 1`` must give ``n - 1 == -1``, bitwise.

    The horizon is where the two branches meet, and they meet exactly: at
    ``x == 1`` the inside branch is ``-1 / (sqrt(0) + 1) == -1`` and the outside
    branch is ``-sqrt(0) - 1 == -1``, with no rounding on either side. Nothing
    else in this section reaches that pixel -- every row of
    ``_NM1_PIXEL_SCALES`` is required by its own horizon precondition to stay
    ``1e-3`` clear of it -- so without this cell the boundary is untested.

    A 5x5 grid at ``pixsize = 0.5`` puts four pixels on it in **binary-exact**
    arithmetic: ``(+-1.0)^2 + 0.0^2`` is exactly 1.0, no rounding involved, so
    this is not a near-miss dressed up as a boundary (measured 2026-09-09:
    ``r2 == 1.0`` at exactly (0,2), (2,0), (2,4), (4,2)).

    What it catches: narrowing the mask that builds ``x`` from ``<=`` to ``<``
    -- an entirely plausible edit, since the *selector* is ``<=`` -- sets
    ``x = 0`` at those pixels and returns ``-0.0`` instead of ``-1.0``. That
    corrupts their w-phase and, worse, makes ``n = (n-1) + 1`` equal ``1.0``
    instead of ``0.0``, so ``wgridder._disc_mask_and_safe_n`` starts treating
    the horizon as *interior* and divides by it under ``divide_by_n=True``.
    Verified 2026-09-09: with that mutation ``tests/test_planning.py`` reports
    **1 failed, 105 passed** -- this cell and nothing else. Before it existed
    the mutation was silent.
    """
    n_pix = 5
    pixsize = 0.5
    i = np.arange(n_pix) - n_pix // 2
    r2 = ((i * pixsize)[:, None]) ** 2 + ((i * pixsize)[None, :]) ** 2
    on_horizon = r2 == 1.0
    assert int(on_horizon.sum()) == 4, (
        f"expected the 5x5 / pixsize 0.5 grid to put 4 pixels exactly on l^2 + m^2 = 1, "
        f"found {int(on_horizon.sum())}; this cell is vacuous without them"
    )

    got = _n_minus_1_grid(n_pix, n_pix, pixsize, pixsize)
    assert np.array_equal(got[on_horizon], np.full(4, -1.0)), (
        f"n - 1 on the exact horizon is {got[on_horizon]!r}, not exactly -1.0. Both "
        f"branches evaluate to -1 there with no rounding, so this is a mask or a "
        f"branch-selection defect, not a precision one; at -0.0 the pixel's w-phase is "
        f"wrong and n = (n-1) + 1 becomes 1.0 instead of 0.0, which puts the horizon "
        f"inside wgridder._disc_mask_and_safe_n's division mask"
    )
    n_grid = got[on_horizon] + 1.0
    assert np.array_equal(n_grid, np.zeros(4)) and not np.any(np.signbit(n_grid)), (
        f"n = (n - 1) + 1 on the exact horizon is {n_grid!r}, not +0.0; "
        f"_disc_mask_and_safe_n masks on n > 0 and must exclude these pixels"
    )


@pytest.mark.parametrize(
    ("n_l", "n_m", "pixsize_l", "pixsize_m", "naive_rel_err", "naive_clears_gate"),
    [case[1:] for case in _NM1_PIXEL_SCALES],
    ids=[case[0] for case in _NM1_PIXEL_SCALES],
)
def test_nm1_relative_accuracy_across_pixel_scales(
    n_l: int,
    n_m: int,
    pixsize_l: float,
    pixsize_m: float,
    naive_rel_err: float,
    naive_clears_gate: bool,
) -> None:
    """issue #12: ``n - 1`` must be accurate to 1e-12 *relative*, at any pixel scale.

    Three assertions per row, in this order, because the third one is only
    worth anything if the first two hold:

    1. the frozen pre-#12 formula reproduces the relative error recorded in
       ``_NM1_PIXEL_SCALES``, within an explicit **two-sided** factor of two
       (``0.5 * v <= measured <= 2 * v``) -- the numbers are deterministic
       IEEE-754, the factor is slack for a platform whose ``sqrt`` is not
       correctly rounded, not for drift, and the lower half is the half that
       catches a helper someone "kept in sync";
    2. whether that row *can* discriminate is asserted, not assumed:
       ``naive_clears_gate`` says whether the old formula already passes
       ``_NM1_REL_GATE``, and it is ``True`` on exactly one row;
    3. ``planning._n_minus_1_grid`` passes the gate.

    Row 2 is the anti-vacuity row and the point of the table. **EDA2's own
    pixel size cannot see this bug**: at 3.27e-2 rad the innermost ``|n - 1|``
    is 5.3560e-04, 2.4e12 ulps of 1.0, and the pre-#12 formula already
    measures 3.8342e-14. A test written on the repository's widest fixture --
    the natural place to put it -- would have passed before the rewrite and
    after it, pinning nothing. Within the regime this table covers (every
    pixel comfortably inside the disc, or EDA2's wide field) the pre-#12
    error tracks ``2 * ulp(1) / pixsize^2``, which crosses 1e-12 at ~2e-2 rad;
    every one of the seven review fixtures except EDA2 is below that, and the
    two VLBI rows are far below.

    Fails on ``ef509c5`` on the six rows with ``naive_clears_gate=False``,
    worst at ``pixsize_1e-8_vlbi``: the true innermost ``|n - 1|`` is 5.0e-17,
    the naive form can only return 0.0 or 1.11e-16 there, and it returns the
    latter -- a relative error of 1.22, i.e. 122%.

    The horizon precondition below keeps this cell honest in the other
    direction. ``n`` is infinitely ill-conditioned in ``r2 = l^2 + m^2`` at the
    disc edge (``dn / d(r2) = -1 / 2n``), so a pixel that lands on ``r2 = 1``
    carries a relative error of order ``sqrt(ulp(1))`` no matter how ``n - 1``
    is written: at ``pixsize = 2.5e-2`` on 64^2, pixel (0, 8) sits at
    ``l = -0.8, m = -0.6`` and **both** forms measure 4.3644e-09 there
    (2026-09-08). That is a property of the square root, not a cancellation
    this issue can remove, so such a row would fail the gate for ever. Every
    row of ``_NM1_PIXEL_SCALES`` keeps ``min|1 - r2|`` at 1.9e-3 (EDA2) or
    above -- 0.95 or more on the six narrow-field rows -- and the assertion
    below is what stops a future row from being added inside that trap and
    mistaken for a real failure.
    """
    horizon_gap = _nm1_horizon_gap(n_l, n_m, pixsize_l, pixsize_m)
    assert horizon_gap > _NM1_HORIZON_GAP_MIN, (
        f"this grid has a pixel at |1 - (l^2 + m^2)| = {horizon_gap:.4e}, on the disc "
        f"edge where n is ill-conditioned in r2 and no algebraic rewrite of n - 1 can "
        f"reach {_NM1_REL_GATE:.0e} relative; pick a pixel size whose grid stays clear "
        f"of the horizon"
    )

    exact = _nm1_exact(n_l, n_m, pixsize_l, pixsize_m)

    naive = _nm1_naive_pre_issue_12(n_l, n_m, pixsize_l, pixsize_m)
    _, measured_naive_rel = _nm1_errors_vs_exact(naive, exact)
    # An explicit two-sided band, NOT ``pytest.approx(naive_rel_err, rel=1.0)``.
    # ``approx`` takes the *larger* of its relative and absolute tolerances and
    # its default ``abs`` is 1e-12, so on the EDA2 row (3.83e-14) the effective
    # window was [0, 1.04e-12] -- 27x the recorded value, with no lower bound at
    # all. Every row accepted 0.0, i.e. the assertion could not detect the drift
    # it exists to detect, in the direction that matters (a helper "kept in
    # sync" makes this number *smaller*, not larger).
    assert 0.5 * naive_rel_err <= measured_naive_rel <= 2.0 * naive_rel_err, (
        f"the recorded pre-#12 relative error {naive_rel_err:.4e} is no longer what the "
        f"frozen formula produces ({measured_naive_rel:.4e}); the table in "
        f"_NM1_PIXEL_SCALES describes float64 arithmetic and should not move"
    )
    assert (measured_naive_rel <= _NM1_REL_GATE) is naive_clears_gate, (
        f"this row's discriminating power changed: the pre-#12 formula measures "
        f"{measured_naive_rel:.4e} against a gate of {_NM1_REL_GATE:.0e}, so "
        f"naive_clears_gate should be {measured_naive_rel <= _NM1_REL_GATE}, not "
        f"{naive_clears_gate}"
    )

    _, rel = _nm1_errors_vs_exact(_n_minus_1_grid(n_l, n_m, pixsize_l, pixsize_m), exact)
    assert rel <= _NM1_REL_GATE, (
        f"_n_minus_1_grid is {rel:.4e} relative against the exact grid at "
        f"pixsize ({pixsize_l:.4e}, {pixsize_m:.4e}), over the gate of "
        f"{_NM1_REL_GATE:.0e}. The pre-#12 formula measures {measured_naive_rel:.4e} "
        f"here; use -x / (sqrt(1 - x) + 1) for the inside-disc branch"
    )


@pytest.mark.parametrize("pixsize", [1e-9, 1e-10], ids=["pixsize_1e-9", "pixsize_1e-10"])
def test_nm1_is_never_exactly_zero_off_the_phase_centre(pixsize: float) -> None:
    """Issue #12's premise -- "exactly 0 for the innermost pixels" -- was true, below 1e-9.

    The issue body says ``n - 1`` is "exactly 0 for the innermost pixels in
    float32". PR #70's first correction called that false; it is false about
    *float32* -- ``_n_minus_1_grid`` computes in float64 whatever the plan
    dtype, and narrowing preserves an exact zero rather than creating one --
    but the underlying claim is right one order of magnitude below the finest
    row of ``_NM1_PIXEL_SCALES``, for a reason the issue did not give: float64
    cancellation, not float32 storage.

    Count of **off-centre** pixels where ``n - 1`` is exactly ``0.0``, 64^2
    grid, measured 2026-09-09 (identical after narrowing to float32):

        pixsize   pre-#12   cancellation-free
        1e-8            0                   0
        1e-9          176                   0
        1e-10        4095 (all)             0
        1e-12        4095 (all)             0

    At 0.2 mas the pre-#12 formula collapsed 176 pixels of the grid to a
    constant; at 0.02 mas, the entire off-centre grid. The cancellation-free
    form returns -5e-19 at (32, 33) at pixsize 1e-9 and is never exactly zero
    at any scale. This is a stronger statement than the relative gate above --
    a value of exactly zero has infinite relative error and destroys the pixel
    ordering, not just its precision -- and it is the assertion that matches
    what the issue actually asked for.
    """
    n_pix = 64
    got = _n_minus_1_grid(n_pix, n_pix, pixsize, pixsize)
    off_centre = np.ones((n_pix, n_pix), dtype=bool)
    off_centre[n_pix // 2, n_pix // 2] = False

    frozen = _nm1_naive_pre_issue_12(n_pix, n_pix, pixsize, pixsize)
    n_zero_before = int((frozen[off_centre] == 0.0).sum())
    assert n_zero_before > 0, (
        f"the pre-#12 formula returns no exact zeros off the phase centre at pixsize "
        f"{pixsize:.0e}, so this row no longer demonstrates the collapse it exists to "
        f"demonstrate (measured 176 at 1e-9 and 4095 at 1e-10 on 2026-09-09)"
    )
    assert not np.any(got[off_centre] == 0.0), (
        f"_n_minus_1_grid returns exactly 0.0 at "
        f"{int((got[off_centre] == 0.0).sum())} off-centre pixels at pixsize "
        f"{pixsize:.0e}; the pre-#12 formula returned {n_zero_before}. An exact zero "
        f"is not a rounding error, it is the loss of the pixel's identity in the "
        f"w-phase"
    )
    assert not np.any(got.astype(np.float32)[off_centre] == 0.0), (
        f"narrowing to float32 introduces exact zeros off the phase centre at pixsize "
        f"{pixsize:.0e}; the float64 grid has none, so this is the float32 storage "
        f"claim in issue #12's body and it would be a separate defect"
    )


def test_the_horizon_precondition_rejects_the_grid_it_is_meant_to_reject() -> None:
    """The ``min|1 - r2| > 1e-3`` precondition must be reachable, and it must be right.

    Same shape as ``tests/test_window_bucketing.py::
    test_the_vacuity_guard_rejects_the_fixtures_it_is_meant_to_reject``: a guard
    that never fires on any current row is indistinguishable from a guard that
    cannot fire, so one grid that *does* trip it is asserted here.

    ``pixsize = 2.5e-2`` on 64^2 is that grid. Pixel (0, 8) sits at
    ``l = -0.8, m = -0.6`` and ``l^2 + m^2`` lands within one ulp of 1.0:
    measured 2026-09-09, ``min|1 - r2| = 2.2204e-16``, thirteen orders of
    magnitude inside the ``1e-3`` threshold.

    The second assertion is the reason the guard exists rather than a bug: at
    the horizon ``dn/d(r2) = -1/(2n)`` diverges, so ``n - 1`` inherits
    ``~sqrt(ulp(1))`` of relative error from the pixel coordinate no matter how
    it is written, and **both** forms measure 4.3644e-09 there (2026-09-09) --
    3600x over ``_NM1_REL_GATE``. A row placed there would fail for ever and
    read as a defect in ``_n_minus_1_grid``, which is what the guard prevents.
    """
    n_pix = 64
    pixsize = 2.5e-2
    gap = _nm1_horizon_gap(n_pix, n_pix, pixsize, pixsize)
    assert gap <= _NM1_HORIZON_GAP_MIN, (
        f"the pixsize 2.5e-2 grid no longer trips the horizon precondition "
        f"(min|1 - r2| = {gap:.4e} against a threshold of {_NM1_HORIZON_GAP_MIN:.0e}); "
        f"without a grid that trips it, the precondition in "
        f"test_nm1_relative_accuracy_across_pixel_scales is untested"
    )

    exact = _nm1_exact(n_pix, n_pix, pixsize, pixsize)
    _, naive_rel = _nm1_errors_vs_exact(
        _nm1_naive_pre_issue_12(n_pix, n_pix, pixsize, pixsize), exact
    )
    _, live_rel = _nm1_errors_vs_exact(_n_minus_1_grid(n_pix, n_pix, pixsize, pixsize), exact)
    assert naive_rel > _NM1_REL_GATE and live_rel > _NM1_REL_GATE, (
        f"on the horizon grid the pre-#12 form measures {naive_rel:.4e} and the "
        f"cancellation-free form {live_rel:.4e}; the precondition exists because "
        f"BOTH exceed {_NM1_REL_GATE:.0e} there (4.3644e-09 each, 2026-09-09), so if "
        f"either now clears the gate the precondition is over-broad and should be "
        f"narrowed rather than kept"
    )


# (id, plan dtype, epsilon, relative gate). The float64 row is issue #12's
# definition of done. The float32 row is the *other* half of the same claim and
# the one the PR text got backwards: ``_n_minus_1_grid`` is float64 whatever the
# plan dtype, so the helper is unaffected by ``dtype`` -- but ``plan.n_minus_1``
# is the narrowed leaf, and below ~1e-5 rad it inherited the float64
# cancellation wholesale. Measured 2026-09-09, 64^2, whole grid against
# ``_nm1_exact``, ``ef509c5`` -> ``ec469ff``:
#
#     pixsize   float64 before   float64 after   float32 before   float32 after
#     3.27e-2       2.4563e-13      1.6894e-13       3.8798e-05      3.8798e-05
#     5.00e-3       4.4372e-12      2.9350e-14       1.0678e-05      1.0678e-05
#     1.00e-6       8.8901e-05      5.6830e-14       8.8901e-05      2.2122e-05
#     1.00e-8       1.2204e+00      5.4815e-14       1.2204e+00      2.6784e-05
#
# i.e. at 1e-8 the float32 plan leaf improves by a factor of 45,000, which is
# the largest measurable effect this change has anywhere. The float32 floor of
# ~2.7e-05 is issue #23's ``n_minus_1_shifted - nshift`` round trip in single
# precision, not this issue, which is why the two gates differ by eight orders
# of magnitude.
#
# What this does NOT do is add coverage to the ``JAX_ENABLE_X64=0`` CI leg:
# this whole module is in ``tests/conftest.py``'s ``collect_ignore`` there
# (it builds default float64 plans and imports modules that switch x64 back
# on). ``dtype=jnp.float32`` under an x64-enabled jax exercises the narrowing
# and the single-precision ``n_minus_1_shifted - nshift`` round trip, which is
# the whole of what this issue can affect at float32; a cell that also wanted
# the x64=0 *config* would have to live in a module that leg collects.
_NM1_PLAN_DTYPE_CASES: tuple[tuple[str, DTypeLike, float, float], ...] = (
    ("float64", jnp.float64, 1e-6, _NM1_REL_GATE),
    # epsilon 1e-4: make_plan warns below 1e-5 for a float32 plan, and the leaf
    # measured here does not depend on epsilon (checked: identical at 1e-6).
    ("float32", jnp.float32, 1e-4, 1e-4),
)


@requires_x64
@pytest.mark.parametrize(
    ("dtype", "epsilon", "gate"),
    [case[1:] for case in _NM1_PLAN_DTYPE_CASES],
    ids=[case[0] for case in _NM1_PLAN_DTYPE_CASES],
)
def test_plan_n_minus_1_relative_accuracy_at_vlbi_pixel_scale(
    dtype: DTypeLike, epsilon: float, gate: float
) -> None:
    """issue #12's definition of done, on the plan leaf rather than the helper.

    ``plan.n_minus_1`` is not ``_n_minus_1_grid``'s output: since issue #23 it
    is the property ``n_minus_1_shifted - nshift``, so it carries an extra
    ``ulp(nshift)`` of absolute error on top. That matters here because
    ``nshift`` scales with the grid, not with the innermost pixel: at
    ``pixsize = 1e-8`` on 64^2, ``nshift`` is 5.1237e-14 and ``ulp(nshift)`` is
    6.3109e-30, which is a sixth of an ulp of 1.0 in absolute terms but sits
    against an innermost ``|n - 1|`` of 5.0e-17. So the round trip alone spends
    most of a 1e-12 relative budget, and the gate cannot simply be inherited
    from the helper's 3.5479e-16 -- it is measured through the plan.

    Measured 2026-09-09, 64^2, epsilon as parametrised, whole grid against
    ``_nm1_exact``:

        dtype      plan.n_minus_1 at ef509c5   with the stable inside branch
        float64                      1.2204                        5.4815e-14
        float32                      1.2204                        2.6784e-05

    18x of headroom after the rewrite on the float64 row, 3.7x on the float32
    one, against 1.2e12 / 1.2e4 times over their gates before it. See
    ``_NM1_PLAN_DTYPE_CASES`` for the full pixel-scale x dtype table and for
    why the two gates are eight orders of magnitude apart.

    **Grid size is an axis this cell deliberately does not sweep, and it is
    not free.** The ``nshift`` round trip grows with the extent while the
    innermost ``|n - 1|`` does not, so the float64 measurement degrades as the
    grid doubles -- measured 2026-09-09 at pixsize 1e-8: 64^2 5.4815e-14
    (nshift 5.120e-14), 128^2 1.9762e-13, 256^2 3.0728e-13, and **512^2
    3.7317e-12, which fails this 1e-12 gate**. That failure would belong to
    issue #23's ``n_minus_1_shifted - nshift`` representation, not to #12: the
    helper itself measures 3.5e-16 at every one of those sizes. 64^2 is the
    sized-down review geometry and is what this cell uses; a maintainer who
    raises ``n_pix`` here should expect a red cell at 512 and should not read
    it as a regression in ``_n_minus_1_grid``.
    """
    n_pix = 64
    pixsize = 1e-8
    plan = make_plan(
        uvw=_baseline_uvw(),
        freq=np.array([200e6]),
        image_shape=(n_pix, n_pix),
        pixsize_l=pixsize,
        pixsize_m=pixsize,
        epsilon=epsilon,
        dtype=dtype,
    )
    exact = _nm1_exact(n_pix, n_pix, pixsize, pixsize)
    _, rel = _nm1_errors_vs_exact(np.asarray(plan.n_minus_1), exact)
    assert rel <= gate, (
        f"plan.n_minus_1 ({np.dtype(dtype).name}) is {rel:.4e} relative against the "
        f"exact grid at pixsize {pixsize:.0e} rad, over the gate of {gate:.0e} "
        f"(measured 1.2204 at both dtypes before issue #12's rewrite; 5.4815e-14 "
        f"float64 / 2.6784e-05 float32 after it)"
    )


@pytest.mark.parametrize(
    ("n_l", "n_m", "pixsize_l", "pixsize_m"),
    [case[1:5] for case in _NM1_PIXEL_SCALES],
    ids=[case[0] for case in _NM1_PIXEL_SCALES],
)
def test_nm1_absolute_error_stays_at_ulp_scale(
    n_l: int, n_m: int, pixsize_l: float, pixsize_m: float
) -> None:
    """The absolute error must stay at ulp scale, and the live form within 3 ulps.

    ``n - 1`` reaches the operators only through ``exp(2*pi*i*w*(n-1))`` and
    through ``n = (n-1) + 1``, and both depend on the **absolute** error, not
    the relative one. So this cell holds the absolute error at ulp scale for
    the frozen pre-#12 formula *and* for the live one -- a future rewrite that
    bought relative accuracy at the cost of absolute accuracy would be a real
    regression that no accuracy test in this suite would catch, and the first
    two assertions are what stand in its way.

    The third assertion is stronger and applies to the live form only: on a
    grid wholly inside the disc it must be within **three ulps of its own
    value** at every pixel, i.e. essentially correctly rounded. A flat absolute
    bound cannot see a defect that scales with the value -- ``np.nextafter``
    applied to every inside pixel is a real one-ulp degradation, and it changes
    the whole-grid absolute error of the 1e-6 row from 1.06e-25 to 1.2e-28,
    which no absolute bound in this file would notice. Measured elementwise
    ulp distance from ``_nm1_exact`` (2026-09-09):

        row                    live    with the one-ulp nextafter mutation
        MWA 25 deg / 128       2.599                                 3.067
        pixsize 5e-3           2.756                                 3.756
        MeerKAT 1.5 deg / 256  2.588                                 3.037
        odd 63x65 anisotropic  2.398                                 3.356
        pixsize 1e-6           2.193                                 3.056
        pixsize 1e-8           2.353                                 3.312
        EDA2 (wide field)      7.284                                 7.284

    Three is not tuned to those numbers, it is the forward bound of the
    expression: ``x = fl(l^2 + m^2)`` is within one ulp, and ``1 - x``,
    ``sqrt``, ``+ 1`` and the divide each add at most half a one, with the
    condition number of ``n - 1`` in ``x`` equal to 1 for small ``x``. Every
    operation is IEEE-754 correctly rounded, so the measurements above are
    bit-reproducible on any platform, not just this one.

    EDA2 is exempted from the ulp bound and given 10: its worst pixel is
    dominated by the rounding of the pixel *coordinate* (``l`` reaches 1.05, so
    ``ulp(l)`` is 2.2e-16 before any square root), which both forms share and
    which the mutation does not touch -- 7.284 either way. It is a
    characterisation row here, not the discriminating one.

    Measured 2026-09-08 at ``ef509c5``, whole grid against ``_nm1_exact``:

        grid                       naive abs      stable abs
        EDA2 120 deg FoV / 64     1.6173e-15      1.6173e-15
        every other row above     ~8.15e-17       <= 4.03e-18

    Hence the two-tier bound. EDA2 is an order of magnitude worse than the
    narrow-field grids and identically so in both forms, because its error
    comes from somewhere else entirely: with ``l`` running to 1.05 the
    float64-rounded pixel coordinate is already 2.2e-16 off, and the
    outside-disc branch amplifies that, so the figure is a property of the
    grid construction that this issue does not touch (and that
    ``test_nm1_outside_disc_branch_is_bit_identical`` pins exactly). A flat
    2e-16 bound is the tempting form of this test and it is wrong: it fails on
    the repository's own wide-field fixture, before the rewrite as well as
    after.
    """
    # ``max|n - 1|`` never exceeds 1 on a grid inside the disc and grows past
    # it on a wide field, so this is one ulp of the grid's own dynamic range,
    # times a factor of ~2.5 of headroom over the worst measurement.
    inside_only = bool(_nm1_inside_disc(n_l, n_m, pixsize_l, pixsize_m).all())
    bound = 2e-16 if inside_only else 4e-15
    ulp_bound = 3.0 if inside_only else 10.0

    exact = _nm1_exact(n_l, n_m, pixsize_l, pixsize_m)
    naive_abs, _ = _nm1_errors_vs_exact(
        _nm1_naive_pre_issue_12(n_l, n_m, pixsize_l, pixsize_m), exact
    )
    assert naive_abs <= bound, (
        f"the pre-#12 formula's absolute error is {naive_abs:.4e}, over {bound:.0e}; "
        f"this arm characterises frozen float64 arithmetic and should not move"
    )

    got = _n_minus_1_grid(n_l, n_m, pixsize_l, pixsize_m)
    got_abs, _ = _nm1_errors_vs_exact(got, exact)
    assert got_abs <= bound, (
        f"_n_minus_1_grid's absolute error is {got_abs:.4e}, over {bound:.0e}, at "
        f"pixsize ({pixsize_l:.4e}, {pixsize_m:.4e}). Relative accuracy near the phase "
        f"centre may not be bought with absolute accuracy: the w-phase "
        f"exp(2*pi*i*w*(n-1)) and the adjoint's n = (n-1) + 1 both read the absolute "
        f"value (pre-#12 form: {naive_abs:.4e})"
    )

    worst_ulps = _nm1_max_ulp_error(got, exact)
    assert worst_ulps <= ulp_bound, (
        f"_n_minus_1_grid is {worst_ulps:.3f} ulps from correctly rounded at pixsize "
        f"({pixsize_l:.4e}, {pixsize_m:.4e}), over the bound of {ulp_bound:g}. This is "
        f"the elementwise arm: a defect that scales with the pixel value (an ulp of a "
        f"1e-4-scale number, say) is invisible to the whole-grid absolute bound above "
        f"but not to this one"
    )


def test_nm1_outside_disc_branch_is_bit_identical() -> None:
    """Outside the unit disc, ``n - 1`` must not change by a single bit.

    ``-sqrt(l^2 + m^2 - 1) - 1`` is a sum of two same-sign quantities: there is
    no cancellation to remove, so issue #12's rewrite has no business touching
    it. The branch is not a corner case here -- EDA2's 120-degree field puts
    **1155 of 4096 pixels** outside the disc (measured 2026-09-08), reaching
    ``n - 1 = -2.0924``, and those pixels set ``nshift`` and therefore the
    plane count for the whole plan (``test_nshift_matches_geometry``).

    Bit-identity rather than a tolerance, because these pixels are exactly
    where the plan's dynamic range lives: on this grid ``max|n-1|`` outside the
    disc is 2.0924 against 1.0 inside, ``nshift`` is 1.0462, and a change of
    even one ulp there moves ``dw`` and can move ``n_w``. The absolute error of
    this branch against the exact grid is 1.6173e-15 -- see
    ``test_nm1_absolute_error_stays_at_ulp_scale`` for why that is the pixel
    coordinate's rounding and not the branch's -- and it must be *the same*
    1.6173e-15 after the rewrite, not a different value inside a tolerance.

    Passes on ``ef509c5``; it is a guard on the rewrite, not a demonstration of
    the defect.
    """
    n_pix = 64
    pixsize = math.radians(120.0) / n_pix  # EDA2
    outside = ~_nm1_inside_disc(n_pix, n_pix, pixsize, pixsize)
    assert int(outside.sum()) == 1155, (
        f"expected EDA2's 64^2 / 120-degree grid to put 1155 pixels outside the unit "
        f"disc, found {int(outside.sum())}; this cell is vacuous without them"
    )

    got = _n_minus_1_grid(n_pix, n_pix, pixsize, pixsize)
    frozen = _nm1_naive_pre_issue_12(n_pix, n_pix, pixsize, pixsize)
    assert np.array_equal(got[outside], frozen[outside]), (
        "the outside-disc analytic extension changed. "
        f"max |delta| = {float(np.max(np.abs(got[outside] - frozen[outside]))):.4e}; "
        "issue #12's rewrite applies to the inside-disc branch only"
    )
    assert float(np.min(got[outside])) == pytest.approx(-2.0924, abs=1e-4)


# (id, n_l, n_m, pixsize_l, pixsize_m) -- the imaging-realistic geometries, i.e.
# the pixel sizes of the review fixtures plus one odd/anisotropic grid. The two
# VLBI rows are deliberately absent: nothing in this repository images at 2 mas,
# and the claim being pinned is about the configurations that exist.
_NM1_IMAGING_GEOMETRIES: tuple[tuple[str, int, int, float, float], ...] = tuple(
    case[:5] for case in _NM1_PIXEL_SCALES if "vlbi" not in case[0]
)


@requires_x64
@pytest.mark.parametrize(
    ("n_l", "n_m", "pixsize_l", "pixsize_m"),
    [case[1:] for case in _NM1_IMAGING_GEOMETRIES],
    ids=[case[0] for case in _NM1_IMAGING_GEOMETRIES],
)
def test_plan_n_minus_1_does_not_move_at_imaging_pixel_scales(
    n_l: int, n_m: int, pixsize_l: float, pixsize_m: float
) -> None:
    """issue #12 bounds how far ``plan.n_minus_1`` moves: half an ulp of 1.0, times 1.5.

    This cell bounds the **size of the change**, and that is all it does. It is
    explicitly *not* an argument that nothing downstream moves -- that
    inference is false, and the two cells that state the truth instead are
    ``test_nm1_downstream_phase_accuracy_at_large_w`` (the change is visible
    downstream, it scales as ``2 pi w`` times this bound, and the new form is
    the more accurate of the two) and the re-run accuracy sweep in
    ``tests/test_accuracy_sweep.py``.

    For the record, measured 2026-09-09 over 78 plans (the eight review
    fixtures plus five synthetic geometries x both dtypes x epsilon 1e-4/1e-6 x
    hermitian on/off) and their 312 operator outputs, ``ef509c5`` vs
    ``ec469ff``: 60 of 78 plans and 224 of 312 outputs change **bitwise**;
    ``plan.nshift`` moves on 38 of 78 (worst 3.17e-17 absolute, 3.70e-13
    relative, on MeerKAT zenith). What does *not* move, on 0 of 78, is every
    structural quantity -- ``n_w``, ``w_kernel_width``, ``beta``,
    ``w_kernel_scale``, ``w0``, ``w_extent``, the whole window layout,
    ``sort_perm``, ``flip_sign``, ``uvw_m``, ``inv_lambda``,
    ``w_centers_rel``. "It moves no measured number" was the wrong claim; this
    bound plus that structural invariance is the right one.

    "Before" is the frozen ``_nm1_naive_pre_issue_12``, so the comparison keeps
    working once ``planning`` no longer contains that formula.

    Measured 2026-09-09, float64 plan, epsilon 1e-6, max over the whole grid of
    ``|plan.n_minus_1 - pre_issue_12|``:

        grid                       today     with the stable inside branch
        EDA2 120 deg FoV / 64    1.1102e-16                    1.1102e-16
        MWA 25 deg / 128                0                      8.0665e-17
        pixsize 5e-3                    0                      8.0665e-17
        MeerKAT 1.5 deg / 256           0                      8.1481e-17
        odd 63x65 anisotropic           0                      8.1532e-17

    The bound is ``1.5 * (0.5 * np.spacing(1.0))`` = 1.6653e-16, not the
    ``np.spacing(1.0)`` = 2.2204e-16 this cell was first written with. Half an
    ulp of 1.0 is what the cancellation can produce and no more -- inside the
    disc ``sqrt(...) - 1`` is a Sterbenz-exact subtraction, so the whole of the
    pre-#12 error comes from rounding ``1 - r2``, which is bounded by
    ``ulp(1)/2`` -- and EDA2 attains it exactly at 1.1102e-16. The remaining
    1.5x is for issue #23's ``nshift`` round trip, which is what makes EDA2's
    figure non-zero *before* the rewrite as well as after (``nshift`` is 1.0462
    there while ``|n - 1|`` reaches 2.0924, the condition
    ``plan.n_minus_1``'s docstring names as breaking the error-free-subtraction
    argument).

    Note what this bound is still blind to, by construction: it compares
    against a formula whose own error is ``ulp(1)/2``, so a defect smaller than
    that -- ``np.nextafter`` on every inside pixel, say -- cannot show up here
    at all. ``test_nm1_absolute_error_stays_at_ulp_scale``'s elementwise ulp
    arm is where that is caught.
    """
    plan = make_plan(
        uvw=_baseline_uvw(),
        freq=np.array([200e6]),
        image_shape=(n_l, n_m),
        pixsize_l=pixsize_l,
        pixsize_m=pixsize_m,
        epsilon=1e-6,
    )
    moved = np.abs(
        np.asarray(plan.n_minus_1) - _nm1_naive_pre_issue_12(n_l, n_m, pixsize_l, pixsize_m)
    )
    worst = float(moved.max())
    bound = 1.5 * (0.5 * float(np.spacing(1.0)))
    assert worst <= bound, (
        f"plan.n_minus_1 moved by {worst:.4e} against the pre-issue-#12 grid at pixsize "
        f"({pixsize_l:.4e}, {pixsize_m:.4e}), over {bound:.4e} (1.5 x half an ulp of "
        f"1.0). Half an ulp is the most the pre-#12 cancellation could produce, so a "
        f"move past this is a change in kind, not in rounding: re-measure the "
        f"downstream phase (test_nm1_downstream_phase_accuracy_at_large_w) and re-run "
        f"the accuracy sweep"
    )


# The downstream fixture: a fine-pixel image at a large common w offset. 16^2
# pixels and 8 rows is enough for the effect (which is per-pixel, not
# statistical) and keeps the row-at-a-time DFT reference below a millisecond.
_NM1_W_PHASE_N_PIX = 16
_NM1_W_PHASE_N_ROWS = 8
_NM1_W_PHASE_PIXSIZE = 1e-6  # 0.21 arcsec, the VLBI row of _NM1_PIXEL_SCALES
_NM1_W_PHASE_W0 = 1e6  # wavelengths


def _nm1_w_phase_problem() -> tuple[np.ndarray, np.ndarray]:
    """``(uvw in metres, freq)`` for the large-w downstream cell.

    ``freq`` is the speed of light so that ``uvw`` in metres *is* ``uvw`` in
    wavelengths, exactly as ``tests/test_nshift.py::_w0_offset_problem`` does,
    which makes ``_NM1_W_PHASE_W0`` directly the w offset under test.
    """
    rng = np.random.default_rng(11)
    uvw = np.zeros((_NM1_W_PHASE_N_ROWS, 3))
    uvw[:, 0] = rng.uniform(-50.0, 50.0, _NM1_W_PHASE_N_ROWS)
    uvw[:, 1] = rng.uniform(-50.0, 50.0, _NM1_W_PHASE_N_ROWS)
    uvw[:, 2] = _NM1_W_PHASE_W0 + rng.uniform(-10.0, 10.0, _NM1_W_PHASE_N_ROWS)
    return uvw, np.array([SPEED_OF_LIGHT])


def _nm1_dft_forward(
    image: np.ndarray, uvw: np.ndarray, pixsize: float, nm1: np.ndarray
) -> np.ndarray:
    """Forward DFT on the given ``n - 1`` grid, row at a time, float64.

    ``V = sum_lm I exp(-2i pi (u l + v m - w (n-1)))`` -- AGENTS.md section 1's
    sign convention, written from it rather than imported (the row loop is the
    same shape as ``tests/test_against_dft.py::_reference_forward``).

    **Range reduction is deliberately not done here, and that was checked
    rather than assumed.** A large ``w`` normally makes a naive
    ``exp(2i pi w x)`` reference worthless -- ``ulp(2 pi * 1e6)`` is 9.3e-10 --
    which is why ``planning._phase_turns_reduced`` exists. It does not bite on
    *this* fixture because the pixel scale is 1e-6 rad: ``|n - 1|`` never
    exceeds 1.3e-10 on a 16^2 grid, so ``w (n - 1)`` is at most ~5e-7 turns and
    the products are nowhere near a rounding cliff. Measured 2026-09-09 by
    rebuilding the whole cell on an exactly ``Fraction``-reduced reference: the
    pre-#12 number is 1.9853e-10 either way, and the cancellation-free one is
    1.97e-20 (plain) against 7.90e-20 (exact) -- both twenty orders of
    magnitude below what the assertions compare. A future edit that widens the
    field or coarsens the pixel here would need the reduction back.
    """
    n_l, n_m = image.shape
    ll = ((np.arange(n_l) - n_l // 2) * pixsize)[:, None]
    mm = ((np.arange(n_m) - n_m // 2) * pixsize)[None, :]
    out = np.zeros(uvw.shape[0], dtype=np.complex128)
    for r in range(uvw.shape[0]):
        u, v, w = (float(c) for c in uvw[r])
        out[r] = np.sum(image * np.exp(-2j * np.pi * (u * ll + v * mm - w * nm1)))
    return out


@requires_x64
def test_nm1_downstream_phase_accuracy_at_large_w() -> None:
    """The rewrite *is* visible downstream, and it is visible as an improvement.

    This is the cell that replaces the inference
    ``test_plan_n_minus_1_does_not_move_at_imaging_pixel_scales`` used to make.
    That cell shows ``n - 1`` moves by at most half an ulp of 1.0; it does
    **not** follow that nothing downstream moves. ``n - 1`` enters as
    ``exp(2i pi w (n - 1))``, so an absolute error ``delta`` becomes a phase
    error of ``2 pi w delta`` -- unbounded in ``w``. Measured on this fixture
    (16^2 at pixsize 1e-6, 2026-09-09), relative L2 of the DFT against a
    reference built on ``_nm1_exact``:

        w offset     pre-#12 n - 1     cancellation-free n - 1
        0                  9.46e-16                        0.0
        1e4                1.99e-12                        0.0
        1e6                1.99e-10                   1.97e-20

    i.e. exactly linear in ``w``, as ``2 pi w delta`` with ``delta`` the
    ``8.3e-17`` absolute error the pre-#12 form carries at this pixel scale.
    1e6 wavelengths is not a stress value: at 1.4 GHz it is a 214 km baseline.

    Three assertions:

    1. the pre-#12 form's downstream error clears ``1e-11`` -- the anti-vacuity
       assertion, and the direct falsification of "it moves no measured number
       anywhere in the suite";
    2. the live form's is at least 1e6 times smaller;
    3. ``dirty2vis`` itself, at ``epsilon = 1e-12``, stays inside the sweep's
       ``2 x epsilon`` contract against the same reference (measured 0.40x eps).

    Assertion 3 is what makes 1 and 2 about the operator rather than about a
    numpy expression: the plan really does carry ``_n_minus_1_grid``'s output
    into the phase, at a ``w`` where getting it wrong would show.
    """
    n_pix = _NM1_W_PHASE_N_PIX
    pixsize = _NM1_W_PHASE_PIXSIZE
    uvw, freq = _nm1_w_phase_problem()
    image = np.random.default_rng(7).standard_normal((n_pix, n_pix))

    exact_grid = np.array(
        [[float(v) for v in row] for row in _nm1_exact(n_pix, n_pix, pixsize, pixsize)]
    )
    reference = _nm1_dft_forward(image, uvw, pixsize, exact_grid)
    norm = float(np.linalg.norm(reference))

    def rel(nm1: np.ndarray) -> float:
        got = _nm1_dft_forward(image, uvw, pixsize, nm1)
        return float(np.linalg.norm(got - reference)) / norm

    naive_rel = rel(_nm1_naive_pre_issue_12(n_pix, n_pix, pixsize, pixsize))
    live_rel = rel(_n_minus_1_grid(n_pix, n_pix, pixsize, pixsize))

    assert naive_rel > 1e-11, (
        f"the pre-#12 n - 1 costs only {naive_rel:.4e} relative at w = "
        f"{_NM1_W_PHASE_W0:.0e} on this fixture, so this cell no longer shows that the "
        f"issue-#12 rewrite is observable downstream and assertion 2 below is vacuous"
    )
    assert live_rel <= naive_rel / 1e6, (
        f"the cancellation-free n - 1 costs {live_rel:.4e} relative at w = "
        f"{_NM1_W_PHASE_W0:.0e}, against {naive_rel:.4e} for the pre-#12 form -- less "
        f"than the 1e6x improvement issue #12 buys. The w-phase reads the ABSOLUTE "
        f"error of n - 1, multiplied by 2*pi*w"
    )

    eps = 1e-12
    plan = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(n_pix, n_pix),
        pixsize_l=pixsize,
        pixsize_m=pixsize,
        epsilon=eps,
    )
    got = np.asarray(dirty2vis(plan, jnp.asarray(image[None, :, :])))[:, 0]
    operator_rel = float(np.linalg.norm(got - reference)) / norm
    assert operator_rel <= 2.0 * eps, (
        f"dirty2vis is {operator_rel:.4e} ({operator_rel / eps:.2f}x eps) against an "
        f"exact-n-1 DFT at w = {_NM1_W_PHASE_W0:.0e} and pixsize "
        f"{pixsize:.0e}, outside the accuracy sweep's 2x epsilon contract (measured "
        f"0.40x eps at ec469ff)"
    )


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


def test_plan_rejects_zero_rows() -> None:
    """An empty ``uvw`` is named as such, not reported as a numpy reduction error.

    Both empty cases were already impossible -- the w-extent block takes
    ``np.min`` over the rows -- but they surfaced as numpy's "zero-size array
    to reduction operation minimum which has no identity", which says nothing
    about which argument was wrong. Issue #43 made this worth pinning: the
    padding-overhead code downstream has a ``live_row_count == 0`` branch, and
    the reason that branch is defensive rather than reachable is precisely
    that ``n_rows >= 1`` is guaranteed here.
    """
    freq = np.array([200e6])
    with pytest.raises(ValueError, match="at least one row"):
        make_plan(np.zeros((0, 3)), freq, (64, 64), 1e-3, 1e-3, epsilon=1e-6)


def test_plan_rejects_zero_channels() -> None:
    """An empty ``freq`` likewise: there is no transform over no channels."""
    with pytest.raises(ValueError, match="at least one channel"):
        make_plan(_baseline_uvw(), np.zeros(0), (64, 64), 1e-3, 1e-3, epsilon=1e-6)


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
    # issue #17: the Hermitian w-sign fold's per-row sign, (n_rows,) int8 in
    # {+1, -1}. This is the NINTH leaf and the only one this issue may add: the
    # fold needs per-row information, which is inherently (n_rows,), but it must
    # cost ONE BYTE per row and must never acquire a channel axis (the sign of w
    # is frequency-independent, so a (n_chan, n_rows) form would carry no extra
    # information while reintroducing exactly the per-(channel, row) allocation
    # issue #23 deleted). The dtype, the shape and the byte cost are gated in
    # tests/test_hermitian.py; this entry pins the *name* and the leaf count.
    # If the implementation names it differently (a boolean mask, say), rename
    # here and in test_plan_footprint's docstring in the same commit --
    # AGENTS.md sec 4's plan-field checklist covers exactly that.
    "flip_sign",
    # issue #26: the plane order that makes each channel's window-size buckets
    # a contiguous rank range, (n_chan, n_w) int32 -- the same shape and dtype
    # as ``window_start``, and per (channel, plane) rather than per
    # (channel, row), which is the axis issue #23 removed. It is the TENTH leaf
    # and the only one that issue may add: the bucket *table* itself is static
    # (it is a set of compile-time slice lengths), but which planes are in which
    # bucket is data the windowed adjoint indexes with.
    #
    # **Its position in this tuple is load-bearing and is why it is written
    # after ``flip_sign`` rather than beside ``window_start``, which is where it
    # belongs by subject.** Every leaf is an entry parameter of every lowered
    # operator, so a leaf inserted in the middle renumbers every parameter after
    # it; appending keeps the eight ``ab7fbbd`` parameters at the ``ab7fbbd``
    # indices, which is what makes the windowed forward's optimised HLO
    # comparable to ``ab7fbbd``'s at all. See the field's comment in
    # ``planning.py`` and
    # ``tests/test_window_bucketing.py::test_the_windowed_forward_is_ab7fbbds_program_plus_one_unused_leaf``.
    "window_plane_order",
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
    # issue #26: the adjoint's bucketed twin of the line above. Static for the
    # same reason and diagnostic in the same way -- it gates the adjoint leg of
    # ``wgridder._padding_overhead`` and nothing else.
    ("window_padding_overhead_adjoint", lambda v: v + 1.0),
    # issue #26: the per-channel window sizes and the bucket table. Static
    # because each bucket's slice length is a compile-time ``dynamic_slice``
    # shape -- two plans that bucket differently emit different programs and
    # must not share a JIT cache entry.
    ("max_window_size_per_chan", lambda v: tuple(x + 1 for x in v)),
    ("window_buckets", lambda v: (((1, 1),), *v[1:])),
    # issue #43: the two ints that replaced the padded ``window_size`` mean as
    # the diagnostic's denominator. They are STATIC on purpose -- issue #23
    # (PR #42) removed the per-(channel, plane) ``window_size`` leaf to cut
    # plan memory, and the fix for #43 must not reintroduce a per-plane array
    # in any form. ``_EXPECTED_LEAF_FIELDS`` stays at eight entries.
    ("live_row_count", lambda v: v + 1),
    ("empty_plane_count", lambda v: v + 1),
    ("w_extent", lambda v: v + 1.0),
    ("is_constant_w", lambda v: not v),
    # issue #17: the Hermitian fold is a plan-shape *and* operator change (the
    # plane grid moves, and a subset of rows is conjugated), so a folded and an
    # unfolded plan over the same data must not share a JIT cache entry.
    ("hermitian", lambda v: not v),
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

    # EVERY static field survives with its value intact, not a hand-picked
    # few. ``_STATIC_FIELD_PROBES`` above establishes that each of these is in
    # the aux data at all -- it compares treedefs, which a *permutation* of
    # ``_plan_unflatten``'s unpacking survives untouched, since the aux tuple
    # is the same tuple either way. Only reading the values back catches one.
    # (A permutation of two fields holding equal values is still invisible;
    # the issue #43 pair is not such a case, see below.)
    for field_name, _ in _STATIC_FIELD_PROBES:
        assert getattr(rebuilt, field_name) == getattr(plan, field_name), (
            f"{field_name} did not survive tree_unflatten -- check that "
            "_plan_aux and _plan_unflatten unpack the aux tuple in the same order"
        )
    # Non-vacuity for the pair issue #43 added, which are adjacent in the aux
    # tuple and both plain ints: their values must differ, or swapping them in
    # _plan_unflatten would round-trip cleanly.
    assert plan.live_row_count != plan.empty_plane_count
    # issue #11: real_dtype / complex_dtype are aux data, so they survive the
    # round-trip unchanged (AGENTS.md sec 4 plan-field checklist).
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
    # Applying sort_perm yields ascending w in metres -- in the orientation the
    # plan sorted, i.e. after issue #17's fold if this plan applied one.
    w_sorted = _as_planned_uvw(uvw, plan)[sort_perm, 2]
    assert np.all(np.diff(w_sorted) >= 0), (
        "sort_perm does not sort the baselines the plan actually planned against. "
        "issue #17: with hermitian=True the sort key is the FOLDED w, so a sort_perm "
        "taken before the fold leaves the sorted array non-monotonic and every "
        "searchsorted window boundary built from it meaningless -- which the windowed "
        "strategies only notice on a fixture where max_window_size < n_rows"
    )

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

    issue #17: the builder sorts and searches the baselines *as the plan stores
    them*, i.e. after the Hermitian fold if the plan applied one, so the raw
    ``uvw`` is folded here first (``_as_planned_uvw``). This is what makes the
    exact ``window_start`` match below a live gate on the fold's interaction
    with the window builder rather than a comparison of two different
    coordinate systems: the fold changes the sort key, hence ``sort_perm``,
    hence every window boundary.
    """
    sort_perm = np.asarray(plan.sort_perm)
    n_rows = uvw.shape[0]
    uvw = _as_planned_uvw(uvw, plan)
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
    # the *unpadded* nominal support, so ``expected_size`` (which carries the margin
    # and the +/-1 clamp) is the wrong array to divide by and is strictly
    # larger. ``tests/test_padding_overhead.py`` is where the live count is
    # pinned against a reference that does not go through ``searchsorted`` at
    # all; here it is enough that the identity holds and that the padding is
    # visibly excluded.
    #
    # issue #26 adds a second numerator and leaves the first alone. The
    # forward's ``window_padding_overhead`` is still
    # ``n_chan * n_w * max_window_size`` over the live count, because the
    # windowed forward still slices ``max_window_size`` per plane; the
    # *adjoint* slices its plane's bucket length, and
    # ``window_padding_overhead_adjoint`` is the sum of those. The adjoint's
    # numerator is rebuilt here from this file's own reference sizes and the
    # plan's declared bucket table -- which keeps the identity a check on a
    # number rather than a restatement of ``make_plan``'s expression. The table
    # has to cover the reference (no plane bucketed below its own window, which
    # would silently drop rows the dense path weights) and its widest length
    # has to be that channel's widest reference window.
    padded_work = 0
    for c, buckets in enumerate(plan.window_buckets):
        lengths = np.array([length for length, _ in buckets])
        counts = np.array([count for _, count in buckets])
        assert int(counts.sum()) == plan.n_w
        expanded = np.repeat(lengths, counts)
        assert np.all(expanded >= np.sort(expected_size[c])), (
            f"channel {c}: a plane is bucketed below its own window length"
        )
        assert int(lengths.max()) == int(expected_size[c].max())
        padded_work += int(expanded.sum())

    assert plan.window_padding_overhead == pytest.approx(
        plan.n_chan * plan.n_w * plan.max_window_size / plan.live_row_count
    )
    assert plan.window_padding_overhead_adjoint == pytest.approx(padded_work / plan.live_row_count)
    assert padded_work < plan.n_chan * plan.n_w * plan.max_window_size, (
        "the bucketed row-work equals the pre-#26 one on a plan whose window "
        "sizes vary, so nothing was bucketed"
    )
    assert plan.live_row_count < int(expected_size.sum())


@pytest.mark.parametrize("n_distinct", [1025, 3000, 5000])
def test_the_bucket_dp_degrades_gracefully_past_its_edge_cap(n_distinct: int) -> None:
    """The ``_BUCKET_DP_MAX_EDGES`` branch, which no plan in this repository reaches.

    ``bucket_window_sizes`` is ``O(MAX_WINDOW_BUCKETS * m^2)`` in the number
    ``m`` of *distinct* padded window sizes, and it caps ``m`` at
    ``_BUCKET_DP_MAX_EDGES`` candidate cut points so that a plan with thousands
    of distinct window lengths degrades to a restricted search rather than to a
    quadratic plan-build time. Measured over every plan this repository builds
    (five telescopes x two pointings x four epsilons x both geometries, plus
    the clumped tracks), the worst ``m`` is 77 and the worst ``n_w`` is 467, so
    that branch is dead code as far as every other test here is concerned --
    which is exactly why it needs a cell of its own, entered through the public
    function and not by lowering the constant.

    What is asserted is what the branch's docstring promises: restricting the
    candidate edges shrinks the *search space* only, so the result is still a
    valid bucketing (ascending distinct lengths, counts summing to ``m``, and
    every size covered by the length of the class it lands in -- a plane
    bucketed below its own window would silently drop rows the dense path
    weights), merely not provably optimal. The optimum it is compared against
    is computed by the same function with the cap lifted.

    Measured on this machine (uniformly drawn distinct sizes, seeds 0 / 1 / 2),
    restricted cost against unrestricted: 4,789,394 vs 4,789,394 (m = 1025,
    exactly optimal), 40,246,634 vs 40,244,172 (m = 3000, +0.0061%) and
    109,947,623 vs 109,926,955 (m = 5000, +0.0188%); wall time 10.8 / 11.0 /
    11.1 ms restricted against 10.9 / 49.5 / 113.5 ms unrestricted, i.e. the
    cap does the flattening it exists for. The gate below is 1% rather than
    those figures: the claim is "a valid bucketing at a bounded cost", and
    pinning 0.0188% would be pinning the draw.
    """
    rng = np.random.default_rng({1025: 0, 3000: 1, 5000: 2}[n_distinct])
    sizes = np.unique(rng.integers(1, 20 * n_distinct, size=n_distinct * 3))[:n_distinct]
    assert sizes.size == n_distinct, "the draw did not yield enough distinct sizes"

    buckets, order = bucket_window_sizes(sizes)

    lengths = [length for length, _ in buckets]
    counts = [count for _, count in buckets]
    assert 1 <= len(buckets) <= MAX_WINDOW_BUCKETS
    assert lengths == sorted(lengths) and len(set(lengths)) == len(lengths)
    assert sum(counts) == n_distinct
    assert sorted(order.tolist()) == list(range(n_distinct))
    np.testing.assert_array_equal(sizes[order], np.sort(sizes))
    assert np.all(np.repeat(lengths, counts) >= np.sort(sizes)), (
        "the restricted search returned a bucket shorter than a plane it holds"
    )

    restricted = sum(length * count for length, count in buckets)
    cap = jax_nufft.planning._BUCKET_DP_MAX_EDGES
    assert n_distinct > cap, (
        f"vacuous cell: {n_distinct} distinct sizes is inside the "
        f"{cap}-edge cap, so the restricted branch is not taken"
    )
    # Lift the cap by asking the same DP over every candidate edge; the cap is a
    # module constant the function reads, so this is the only way in.
    saved = cap
    jax_nufft.planning._BUCKET_DP_MAX_EDGES = n_distinct
    try:
        unrestricted_buckets, _ = bucket_window_sizes(sizes)
    finally:
        jax_nufft.planning._BUCKET_DP_MAX_EDGES = saved
    unrestricted = sum(length * count for length, count in unrestricted_buckets)

    assert restricted >= unrestricted, (
        "the restricted search beat the full one, which is impossible: its "
        "candidate edges are a subset"
    )
    assert restricted <= 1.01 * unrestricted, (
        f"the {cap}-edge search costs {restricted} against an optimum of "
        f"{unrestricted} ({restricted / unrestricted:.6f}x) on {n_distinct} "
        "distinct sizes"
    )


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

    issue #26 moves the claim onto the quantity it was derived for. The
    crossover analysis above is about ``n_w * max_window_size``, i.e. the work
    an *un-bucketed* windowed traversal does, and that is what the ordering is
    asserted on; ``window_padding_overhead`` is now the work the bucketed
    traversal does, and on this fixture the ordering there is the other way
    round -- bucketing helps the clumped plan far more than the uniform one,
    which is the whole point of it. Measured at eps 1e-6, float64,
    ``hermitian=False``, seed 2 (the ``pixsize`` sweep in order), un-bucketed
    against bucketed:

        pixsize   n_w c/u    clumped              uniform
        5.0e-3     23 / 36    1.6505 -> 1.0111    1.4266 -> 1.1353
        6.0e-3     29 / 49    2.0810 -> 1.0154    1.3470 -> 1.1228
        8.0e-3     47 / 83    3.3739 -> 1.0454    1.4229 -> 1.1575
        1.0e-2     71 / 128   5.0968 -> 1.1207    1.5080 -> 1.1782
        1.2e-2    101 / 187   7.2478 -> 1.1714    1.6690 -> 1.2213
        1.5e-2    163 / 304  11.6969 -> 1.3135    1.9536 -> 1.3299

    So the two statements asserted below are (a) the pre-#26 ordering, on the
    pre-#26 quantity, and (b) that bucketing removes strictly more of the
    clumped plan's padding than of the uniform plan's -- which is what makes
    (a) stop showing up in the reported metric, stated on the same two plans
    rather than left as an explanation.
    """
    uvw, uvw_uniform = _clumped_and_uniform_uvw()
    freq = np.array([1.4e9])
    # hermitian=False (issue #17), and not incidentally: this fixture's clumped
    # w-distribution is two equal clumps at -30 and +30 m, which is exactly the
    # symmetry the Hermitian fold exists to collapse. Folded, the two clumps
    # become one, ``max_window_size`` halves, and the analytic crossover table
    # above -- measured on the two-clump geometry -- stops describing the
    # fixture. The fold's effect on this distribution is a real (and welcome)
    # one, but it is not what this test measures, and re-deriving the crossover
    # for the folded geometry would be a different test. Pinned explicitly so
    # this holds whichever way make_plan's default is set.
    plan_clumped = make_plan(uvw, freq, (64, 64), pixsize, pixsize, epsilon=1e-6, hermitian=False)
    plan_uniform = make_plan(
        uvw_uniform, freq, (64, 64), pixsize, pixsize, epsilon=1e-6, hermitian=False
    )

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

    def unbucketed(plan: WGridderPlan) -> float:
        """Issue #43's overhead: one ``max_window_size`` slice per (channel, plane)."""
        return plan.n_chan * plan.n_w * plan.max_window_size / plan.live_row_count

    assert unbucketed(plan_clumped) > unbucketed(plan_uniform)

    # ``window_padding_overhead`` *is* that expression -- issue #26 buckets the
    # windowed adjoint only, and reports the bucketed ratio under its own name
    # -- so pin that too, or the ordering above would be asserted on a quantity
    # nothing else in this test names.
    for plan in (plan_clumped, plan_uniform):
        assert plan.window_padding_overhead == pytest.approx(unbucketed(plan), rel=1e-12)

    # issue #26: both plans keep an adjoint overhead of at least one -- padded
    # work cannot fall below live work -- and the clumped plan, which is the one
    # whose padding the crossover analysis says is worst, is the one bucketing
    # helps most.
    for plan in (plan_clumped, plan_uniform):
        assert plan.window_padding_overhead_adjoint >= 1.0
        assert plan.window_padding_overhead_adjoint < unbucketed(plan)

    clumped_gain = unbucketed(plan_clumped) / plan_clumped.window_padding_overhead_adjoint
    uniform_gain = unbucketed(plan_uniform) / plan_uniform.window_padding_overhead_adjoint
    assert clumped_gain > uniform_gain, (
        f"bucketing removed {clumped_gain:.2f}x of the clumped plan's padded "
        f"row-work and {uniform_gain:.2f}x of the uniform plan's; the clumped "
        "fixture is the one with the padding to remove"
    )

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
    w_lambda = _as_planned_uvw(uvw, plan) * (freq[0] / SPEED_OF_LIGHT)
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
        window_plane_order  (n_chan, n_w)  int32        4 * n_chan * n_w
        flip_sign           (n_rows,)      int8         1 * n_rows
                                                        -------------------
        total = 29*n_rows + 8*n_chan + 8*n_w + 32*n_l*n_m + 8*n_chan*n_w

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

    issue #17 adds the ``flip_sign`` term, and exactly one byte of it: the
    Hermitian w-sign fold needs per-row sign information, which is inherently
    ``(n_rows,)``, so 28 B/row becomes 29 B/row. It must stay one byte -- an
    int32 leaf would make it 32 and a float64 one 36, and a per-(channel, row)
    form would put ``n_chan`` bytes on every row, which is the allocation
    issue #23 exists to have deleted. ``tests/test_hermitian.py`` gates the
    per-row and per-channel slopes directly; this bound gates the total.

    issue #26 adds the ``window_plane_order`` term, and it is the second and
    last ``4 * n_chan * n_w``: bucketing needs to know which planes are in
    which size class, which is inherently per (channel, plane). Per *plane*,
    not per row -- on the 16-channel, 10k-row fixture below it is 8,512 B
    against 290,000 B of row-shaped leaves (measured), and the axis issue #23
    exists to have deleted is ``n_rows``, which this term does not carry.
    """
    return 29 * n_rows + 8 * n_chan + 8 * n_w + 32 * n_l * n_m + 8 * n_chan * n_w


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
