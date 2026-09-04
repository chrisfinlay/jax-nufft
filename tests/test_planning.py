"""Tests for plan construction (Nw, w-plane centres, kernel correction)."""

from __future__ import annotations

import dataclasses
import math
import threading
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import psutil
import pytest
from jax import Array

from jax_nufft._utils import SPEED_OF_LIGHT
from jax_nufft.kernel import kernel_params
from jax_nufft.planning import W_OVERSAMPLE_X0, WGridderPlan, make_plan
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
# coordinates and w (wavelengths) inside the JIT -- three multiplies per row
# per channel. ``test_plan_uvw_lambda_correct`` and
# ``test_plan_finufft_coords_match_uvw_lambda`` (their v0.1.2-Part-3
# predecessors) asserted the *stored* per-channel arrays were correct; since
# those arrays no longer exist as plan leaves, the same property is now
# gated by reconstructing the derivation as a small jitted helper here and
# checking it against a reference computed independently of both the old and
# the new plan fields, straight from (uvw metres, freq Hz, pixsize) --
# following this file's ``_nm1_extremes`` convention so a bug shared between
# make_plan's real derivation and this helper cannot hide behind it.
# ---------------------------------------------------------------------------


def _independent_ft_coords(
    uvw: np.ndarray, freq: np.ndarray, pixsize_l: float, pixsize_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground truth for (u_ft, v_ft, w_lambda), (n_chan, n_rows) each.

    Deliberately not built from any WGridderPlan field, old or new.
    """
    inv_lambda = freq / SPEED_OF_LIGHT  # (n_chan,)
    u_ft = (2.0 * np.pi * pixsize_l) * np.outer(inv_lambda, uvw[:, 0])
    v_ft = (2.0 * np.pi * pixsize_m) * np.outer(inv_lambda, uvw[:, 1])
    w_lambda = np.outer(inv_lambda, uvw[:, 2])
    return u_ft, v_ft, w_lambda


@jax.jit
def _jit_derive_channel_ft_coords(
    uvw_m: Array, inv_lambda_c: Array, pixsize_l: Array, pixsize_m: Array
) -> tuple[Array, Array, Array]:
    """The formula issue #23 requires inside ``_channel_forward`` /
    ``_channel_adjoint`` for one channel: ``u_ft_c = 2π · pixsize_l ·
    inv_lambda[c] · uvw_m[:, 0]``, and likewise for ``v_ft`` / ``w_lambda``.
    This is the derivation under test, written fresh here -- it is not a call
    into ``jax_nufft.wgridder``, which does not implement it (that is the
    implementation agent's job; this pins the contract it must satisfy).
    """
    two_pi = 2.0 * jnp.pi
    u_ft = (two_pi * pixsize_l * inv_lambda_c) * uvw_m[:, 0]
    v_ft = (two_pi * pixsize_m * inv_lambda_c) * uvw_m[:, 1]
    w_lambda = inv_lambda_c * uvw_m[:, 2]
    return u_ft, v_ft, w_lambda


@requires_x64
def test_plan_derived_channel_coords_match_independent_reference() -> None:
    """``plan.uvw_m`` (metres) + ``plan.inv_lambda`` (freq / c) must carry
    enough information to reconstruct, per channel, the exact (u_ft, v_ft)
    FINUFFT input coordinates and the w-component in wavelengths that the
    removed ``uvw_lambda`` / ``u_finufft`` / ``v_finufft`` leaves used to
    store directly.

    Tolerances: ``w_lambda`` is a single float64 multiply
    (``inv_lambda[c] * uvw_m[:, 2]``) in both the derivation and the
    independent reference, and IEEE-754 multiplication is exactly
    commutative, so it is checked to bit-for-bit equality. ``u_ft`` / ``v_ft``
    additionally fold in ``2π · pixsize``, and the derivation's grouping
    (``(2π · pixsize_l · inv_lambda[c]) · uvw_m[:, 0]``) reassociates that
    product differently than the reference's (``(2π · pixsize_l) · (inv_lambda
    · uvw)``, via ``np.outer``) -- multiplication is commutative but not
    associative in floating point, so the two can differ by a couple of ulps.
    ``rtol`` below is ~20x float64 eps, generous for a two-multiply
    reassociation and far tighter than an axis swap, sign flip, or
    wrong-channel bug would need to slip through.
    """
    uvw = _baseline_uvw(n_rows=12, max_baseline=80.0)
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

    expected_u, expected_v, expected_w = _independent_ft_coords(uvw, freq, pixsize_l, pixsize_m)
    rtol = 20.0 * np.finfo(np.float64).eps

    for c in range(plan.n_chan):
        u_ft, v_ft, w_lambda = _jit_derive_channel_ft_coords(
            plan.uvw_m, plan.inv_lambda[c], plan.pixsize_l, plan.pixsize_m
        )
        np.testing.assert_array_equal(np.asarray(w_lambda), expected_w[c])
        np.testing.assert_allclose(np.asarray(u_ft), expected_u[c], rtol=rtol, atol=0.0)
        np.testing.assert_allclose(np.asarray(v_ft), expected_v[c], rtol=rtol, atol=0.0)

    # Windowed path (issue #16's helpers gather via plan.sort_perm rather than
    # storing a separate (n_chan, n_rows, 3) sorted array; issue #23 removes
    # the last remaining sorted leaf, uvw_lambda_sorted, on the same logic).
    # The sorted w-component the windowed helpers need is therefore a gather
    # of uvw_m by sort_perm, not a stored per-channel array -- check the two
    # compose correctly.
    sort_perm = np.asarray(plan.sort_perm)
    assert sorted(sort_perm.tolist()) == list(range(plan.n_rows))
    uvw_m_sorted = np.asarray(plan.uvw_m)[sort_perm]
    inv_lambda = np.asarray(plan.inv_lambda)
    for c in range(plan.n_chan):
        w_sorted = inv_lambda[c] * uvw_m_sorted[:, 2]
        np.testing.assert_array_equal(w_sorted, expected_w[c][sort_perm])


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
    w_m_sorted = uvw[sort_perm, 2].astype(np.float64)
    w_centers_rel64 = np.asarray(plan.w_centers_rel, dtype=np.float64)
    half_w_dw = plan.w_kernel_scale
    n_chan, n_w = plan.n_chan, plan.n_w
    window_start = np.zeros((n_chan, n_w), dtype=np.int64)
    window_size = np.zeros((n_chan, n_w), dtype=np.int64)
    for c in range(n_chan):
        w_lambda_c = w_m_sorted * (float(freq[c]) / SPEED_OF_LIGHT) - plan.w0
        lo = np.searchsorted(w_lambda_c, w_centers_rel64 - half_w_dw, side="left")
        hi = np.searchsorted(w_lambda_c, w_centers_rel64 + half_w_dw, side="right")
        window_start[c] = lo
        window_size[c] = hi - lo
    return window_start, window_size


def test_window_builder_matches_independent_reference() -> None:
    """``plan.window_start`` must match an independently-computed reference
    exactly, and the reference's window sizes (not a plan leaf any more,
    issue #23) recover the same aggregate property the old
    ``test_window_builder_sum_matches_expected`` pinned: each row lies in
    ``W`` consecutive plane-windows (interior case), so ``sum_k
    window_size[c, k]`` is close to ``n_rows * W``.
    """
    rng = np.random.default_rng(1)
    n_rows = 250
    uvw = rng.normal(scale=120.0, size=(n_rows, 3))
    freq = np.array([1.4e9])

    plan = make_plan(uvw, freq, (128, 128), 5e-4, 5e-4, epsilon=1e-6)
    expected_start, expected_size = _independent_window_bounds(uvw, freq, plan)

    np.testing.assert_array_equal(np.asarray(plan.window_start), expected_start)
    assert plan.max_window_size == int(expected_size.max())

    W = plan.w_kernel_width
    total = int(expected_size.sum())
    # See the removed test's comment for the slack derivation: searchsorted's
    # half-open [side="left", side="right") interval includes rows exactly on
    # a window edge, so each of the n_chan * n_w windows can pick up at most
    # one extra row at each end.
    assert total <= plan.n_rows * W + 2 * plan.n_chan * plan.n_w
    assert total >= plan.n_rows * (W - 1)


# Pixel size for test_window_builder_clumped_distribution. The bare 2e-3 this
# test used before issue #16 is scaled by sqrt(2) so that the *plan geometry*
# it measures is bit-for-bit the geometry it was written against.
#
# Why: nshift halves ``max|n-1|`` -> ``dw = x0 / max|n-1|`` doubles -> the
# w-window count halves and each window widens. On a 64x64 image
# ``max|n-1|`` is ``1024 * pixsize^2`` before the shift and ``512 * pixsize^2``
# after, so ``pixsize * sqrt(2)`` restores the pre-#16 value exactly, giving
# back the same ``n_w`` (12 clumped / 17 uniform), the same ``window_size``
# rows, and therefore the same two padding-overhead numbers (1.7137 / 1.6757).
# Only the sampling resolution is retuned -- the clumped-vs-uniform w geometry
# that is actually under test is untouched, and the assertion is unchanged.
#
# Left at 2e-3 the fixture drops to 3 inner planes with a kernel half-width
# (344) wider than the whole w extent (295), i.e. every plane sees every row
# and the clumped/uniform contrast the assertion is about no longer exists.
# That contrast is in any case not monotone in resolution (measured on this
# fixture: it holds at inner = 5/10 and at 182/347, but not at 10/19 or
# 40/76, because ``window_padding_overhead`` excludes empty windows and a
# well-resolved clumped distribution is all-or-nothing windows with overhead
# ~1.0). Making this a robust invariant rather than a fixture-specific
# heuristic is out of scope here; see the note in the issue #16 PR.
_CLUMPED_PIXSIZE = 2e-3 * math.sqrt(2.0)


def test_window_builder_clumped_distribution() -> None:
    """A clumped w-distribution should produce a high padding overhead."""
    rng = np.random.default_rng(2)
    n_rows = 400
    # Two tight clumps in w: padding overhead should be large because most
    # planes have ~0 rows while the two clump-overlapping planes hold many.
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100, 100, n_rows)
    uvw[:, 1] = rng.uniform(-100, 100, n_rows)
    half = n_rows // 2
    uvw[:half, 2] = rng.normal(loc=-30.0, scale=0.5, size=half)
    uvw[half:, 2] = rng.normal(loc=+30.0, scale=0.5, size=n_rows - half)
    freq = np.array([1.4e9])
    plan_clumped = make_plan(uvw, freq, (64, 64), _CLUMPED_PIXSIZE, _CLUMPED_PIXSIZE, epsilon=1e-6)

    uvw_uniform = rng.uniform(-60.0, 60.0, size=(n_rows, 3))
    plan_uniform = make_plan(
        uvw_uniform, freq, (64, 64), _CLUMPED_PIXSIZE, _CLUMPED_PIXSIZE, epsilon=1e-6
    )

    # Padding overhead should be noticeably higher for the clumped case.
    assert plan_clumped.window_padding_overhead > plan_uniform.window_padding_overhead


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


def _peak_rss_delta_during(fn: Callable[[], None], poll_interval: float = 0.002) -> int:
    """Sample this process's RSS (``psutil``) throughout a call to ``fn``,
    returning peak-minus-baseline.

    ``psutil`` has no cross-platform "peak RSS since a point in time"
    accessor (``Process.memory_info().rss`` is a snapshot), so this polls it
    from a background thread at ``poll_interval`` granularity while ``fn``
    runs. Coarse, but more than sufficient to catch multi-MB transient
    allocations held for longer than a few milliseconds -- exactly the shape
    of the cost this issue targets (Python-level numpy copies that outlive
    their use within a single ``make_plan`` call).
    """
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def _poll() -> None:
        nonlocal peak
        while not stop.is_set():
            rss = process.memory_info().rss
            if rss > peak:
                peak = rss
            stop.wait(poll_interval)

    sampler = threading.Thread(target=_poll, daemon=True)
    sampler.start()
    try:
        fn()
    finally:
        stop.set()
        sampler.join()
    return max(peak - baseline, 0)


def test_make_plan_peak_host_rss_bounded_slow(pytestconfig: pytest.Config) -> None:
    """issue #23: ``make_plan`` currently materialises the per-channel uvw
    arrays (``uvw_lambda``, ``uvw_lambda_sorted``, ``u_finufft``,
    ``v_finufft``) as transient host numpy arrays *in addition to* the ones
    it stores, and loops over channels in Python for ``searchsorted``. This
    pins the host-side (as opposed to ``test_plan_footprint``'s device-leaf)
    cost of that at a stated multiple of the issue's target footprint bound
    -- 3x, from the issue body's own gate ("make_plan peak host RSS delta <
    3x that (psutil, marked slow)", where "that" is the footprint bound
    computed the sentence before it, not whatever a given plan happens to
    measure).

    Marked slow (needs ``--runslow``): RSS sampling is coarser and noisier
    than a value assertion, and this fixture is bigger than the rest of the
    default suite.
    """
    if not pytestconfig.getoption("--runslow"):
        pytest.skip("needs --runslow")

    n_rows = 10_000
    n_chan = 16
    uvw = _footprint_fixture_uvw(n_rows, max_baseline=5000.0, seed=0)
    freq = np.linspace(140e6, 160e6, n_chan)

    def _build() -> None:
        make_plan(
            uvw=uvw,
            freq=freq,
            image_shape=(256, 256),
            pixsize_l=MWA_EXTENDED.pixsize,
            pixsize_m=MWA_EXTENDED.pixsize,
            epsilon=1e-6,
        )

    # One warm-up call outside the timed/sampled region (AGENTS.md sec 7's
    # timing protocol): the first JAX call in a process pays one-time
    # backend-init cost that has nothing to do with make_plan's own
    # transient host allocation, and would otherwise pollute the sample.
    # It also gives us plan.n_w for the bound formula.
    warmup_plan = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(256, 256),
        pixsize_l=MWA_EXTENDED.pixsize,
        pixsize_m=MWA_EXTENDED.pixsize,
        epsilon=1e-6,
    )
    bound = 3 * _footprint_bound_bytes(
        warmup_plan.n_chan, warmup_plan.n_rows, warmup_plan.n_w, warmup_plan.n_l, warmup_plan.n_m
    )

    peak_delta = _peak_rss_delta_during(_build)
    assert peak_delta <= bound, (
        f"make_plan peak host RSS delta {peak_delta} B ({peak_delta / 1e6:.2f} MB) exceeds 3x "
        f"the issue #23 footprint target ({bound} B / {bound / 1e6:.2f} MB) -- make_plan is "
        "materialising more transient host copies than the issue #23 budget allows"
    )
