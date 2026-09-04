"""Tests for plan construction (Nw, w-plane centres, kernel correction)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_plan_uvw_lambda_correct() -> None:
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
    expected = uvw[None, :, :] * (freq[:, None, None] / SPEED_OF_LIGHT)
    np.testing.assert_allclose(np.asarray(plan.uvw_lambda), expected, rtol=1e-6)


def test_plan_finufft_coords_match_uvw_lambda() -> None:
    """v0.1.2 Part 3.1: ``u_finufft`` / ``v_finufft`` must equal
    ``2π · pixsize_* · uvw_lambda[..., axis]`` so the channel helpers can
    read them directly from the plan instead of recomputing per call."""
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
    expected_u = (2.0 * np.pi * pixsize_l) * np.asarray(plan.uvw_lambda)[..., 0]
    expected_v = (2.0 * np.pi * pixsize_m) * np.asarray(plan.uvw_lambda)[..., 1]
    np.testing.assert_allclose(np.asarray(plan.u_finufft), expected_u, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(plan.v_finufft), expected_v, rtol=1e-12)
    # And shape matches uvw_lambda's leading dimensions.
    assert plan.u_finufft.shape == (plan.n_chan, plan.n_rows)
    assert plan.v_finufft.shape == (plan.n_chan, plan.n_rows)


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
    assert plan.max_window_size == plan.n_rows
    assert np.asarray(plan.window_start).shape == (1, 1)
    assert int(np.asarray(plan.window_size)[0, 0]) == plan.n_rows


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
# Leaves, in flatten order.
_EXPECTED_LEAF_FIELDS: tuple[str, ...] = (
    "uvw_lambda",
    "w_centers",
    "w_centers_rel",  # issue #16 follow-up
    "n_minus_1",
    "n_minus_1_shifted",  # issue #16
    "w0_screen",  # issue #16 follow-up
    "phi_hat_n",
    "sort_perm",
    "uvw_lambda_sorted",
    "window_start",
    "window_size",
    "u_finufft",  # v0.1.2 Part 3.1
    "v_finufft",  # v0.1.2 Part 3.1
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
    """sort_perm sorts by w; per-plane windows are monotonic and in-bounds."""
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

    # uvw_lambda_sorted matches uvw_lambda[:, sort_perm, :].
    uvw_lambda = np.asarray(plan.uvw_lambda)
    uvw_lambda_sorted = np.asarray(plan.uvw_lambda_sorted)
    np.testing.assert_allclose(uvw_lambda_sorted, uvw_lambda[:, sort_perm, :])

    window_start = np.asarray(plan.window_start)
    window_size = np.asarray(plan.window_size)
    assert window_start.shape == (plan.n_chan, plan.n_w)
    assert window_size.shape == (plan.n_chan, plan.n_w)

    # Window start is monotonic in k (planes scan ascending in w).
    for c in range(plan.n_chan):
        assert np.all(np.diff(window_start[c]) >= 0)
    # All windows stay within [0, n_rows].
    assert np.all(window_start >= 0)
    assert np.all(window_start + window_size <= plan.n_rows)
    # max_window_size matches the per-(c, k) max.
    assert plan.max_window_size == int(window_size.max())
    # Padding overhead >= 1 by construction.
    assert plan.window_padding_overhead >= 1.0


def test_window_builder_sum_matches_expected() -> None:
    """sum_k window_size[c, k] equals n_rows * W (each row contributes to W planes)."""
    rng = np.random.default_rng(1)
    n_rows = 250
    uvw = rng.normal(scale=120.0, size=(n_rows, 3))
    freq = np.array([1.4e9])

    plan = make_plan(uvw, freq, (128, 128), 5e-4, 5e-4, epsilon=1e-6)
    window_size = np.asarray(plan.window_size)
    W = plan.w_kernel_width
    # Each visibility lies in exactly W consecutive plane-windows (interior
    # case). Edge planes may pick up fewer when the kernel support hangs off
    # the end of the data range, so the sum is bounded above by n_rows * W
    # and below by n_rows * (W - 1) for our test geometry.
    total = int(window_size.sum())
    # The upper bound carries the builder's own documented slack: it uses
    # ``searchsorted(..., "left")`` / ``"right"``, which *includes* a row lying
    # exactly on a window edge, so each of the ``n_chan * n_w`` windows can pick
    # up at most one extra row at each end. That inclusion is not a wart to be
    # tolerated -- it is required for windowed/dense parity, because the dense
    # path gives such a row ``phi(z = +/-1) = exp(-beta)`` (~1e-7 at W=7, i.e.
    # far above tests/test_boundary_planes.py's 1e-12 tolerance) and a windowed
    # path that dropped it would disagree by exactly that much.
    #
    # Exact edge hits used to be vanishingly unlikely because the boundaries
    # were computed in absolute wavelengths; since issue #16's follow-up they
    # are computed relative to ``w0`` (so that they agree with the operators'
    # own ``z``), where the arithmetic is "rounder" and ties do occur -- this
    # fixture has exactly one. The bound below is still tight enough to catch a
    # builder that widened its windows systematically (the slack is 1.6% here).
    assert total <= plan.n_rows * W + 2 * plan.n_chan * plan.n_w
    assert total >= plan.n_rows * (W - 1)


# --- w-clumping and the padding diagnostic ----------------------------------
#
# Two tight w-clumps 60 m apart, against a uniform w-distribution of the same
# row count. The property under test is that
# ``plan.window_padding_overhead`` -- the factor by which a windowed
# traversal's row-work exceeds the irreducible work; see
# ``tests/test_padding_overhead.py`` for the definition -- reports the
# clumped geometry as the more expensive one.
#
# That is only true once the plane grid resolves the clumps, and the
# crossover is exact rather than a matter of taste. Write ``f`` for the
# fraction of rows in the largest clump and ``W`` for the kernel width. Once
# the kernel support is narrower than the clump separation, the widest window
# holds one whole clump, ``f * n_rows``, while the mean over all ``n_w``
# planes is ``n_rows * W / n_w`` (every row lands in ``W`` planes), so
#
#     overhead_clumped  ~=  f * n_w / W.
#
# A uniform distribution has near-equal windows apart from the ``W``-plane
# ramp at each end, giving ``overhead_uniform ~= n_w / (n_w - W)``, which
# decays to 1 as the grid refines. With ``f = 1/2`` the two cross at exactly
# ``n_w = 3W``. Below that the *clumped* plan is the flatter of the two --
# every plane sees both clumps, so its windows are uniformly full -- and the
# assertion legitimately fails. That is a property of the geometry, not a
# defect in the diagnostic, so the sweep below starts above the crossover
# (at ``n_w >= 4W``, asserted per case) and the assertion holds at every
# resolution from there up.
#
# Measured on this fixture (epsilon 1e-6, W = 7):
#
#     pixsize   n_w    clumped   f*n_w/W    uniform
#     6e-3       29      2.071     2.071      1.312
#     8e-3       47      3.357     3.357      1.393
#     1.2e-2    101      7.212     7.214      1.602
#     1.6e-2    189     13.428    13.500      1.769
#
# The old nonzero-window mean produced 1.142 / 1.214 / 1.571 / 2.131 for the
# same four plans: it understates the true waste by up to 6x, and it puts
# the clumped plan *below* the uniform one at three of the four. That is
# what issue #16 ran into, and why it reached for a fixture tweak (a pixsize
# scaled by sqrt(2), restoring the pre-nshift plane count so the two
# hard-coded numbers came back) rather than an invariant. The tweak is gone;
# this sweep replaces it.
_CLUMP_PIXSIZES = (6e-3, 8e-3, 1.2e-2, 1.6e-2)
_CLUMP_ROWS = 400
_CLUMP_OFFSET_M = 30.0
_CLUMP_SIGMA_M = 0.5
_CLUMP_PEAK_FRACTION = 0.5  # two equal clumps
_CLUMP_FREQ = np.array([1.4e9])


def _clumped_and_uniform_uvw() -> tuple[np.ndarray, np.ndarray]:
    """``(clumped, uniform)`` uvw arrays sharing a draw order with the seed."""
    rng = np.random.default_rng(2)
    n_rows = _CLUMP_ROWS
    clumped = np.zeros((n_rows, 3))
    clumped[:, 0] = rng.uniform(-100, 100, n_rows)
    clumped[:, 1] = rng.uniform(-100, 100, n_rows)
    half = n_rows // 2
    clumped[:half, 2] = rng.normal(-_CLUMP_OFFSET_M, _CLUMP_SIGMA_M, half)
    clumped[half:, 2] = rng.normal(+_CLUMP_OFFSET_M, _CLUMP_SIGMA_M, n_rows - half)
    uniform = rng.uniform(-2.0 * _CLUMP_OFFSET_M, 2.0 * _CLUMP_OFFSET_M, size=(n_rows, 3))
    return clumped, uniform


def _clump_plan(uvw: np.ndarray, pixsize: float) -> WGridderPlan:
    return make_plan(uvw, _CLUMP_FREQ, (64, 64), pixsize, pixsize, epsilon=1e-6)


@pytest.mark.parametrize("pixsize", _CLUMP_PIXSIZES)
def test_window_builder_clumped_distribution(pixsize: float) -> None:
    """A clumped w-distribution pays more windowed padding than a uniform one."""
    clumped_uvw, uniform_uvw = _clumped_and_uniform_uvw()
    plan_clumped = _clump_plan(clumped_uvw, pixsize)
    plan_uniform = _clump_plan(uniform_uvw, pixsize)

    # Preconditions, asserted rather than skipped: if a planning change moves
    # this fixture below the crossover the sweep stops testing what it claims
    # to, and that should be a failure, not a silently vacuous pass.
    width = plan_clumped.w_kernel_width
    assert plan_clumped.n_w >= 4 * width
    # Kernel support narrower than the clump separation, so no single plane
    # can see both clumps -- the condition behind ``f * n_w / W``.
    lam = _CLUMP_FREQ[0] / SPEED_OF_LIGHT
    assert plan_clumped.w_kernel_scale < _CLUMP_OFFSET_M * lam

    assert plan_clumped.window_padding_overhead > plan_uniform.window_padding_overhead

    # ... and it is high for the reason claimed: one clump fills the widest
    # window while the empty planes between them drag the mean down.
    predicted = _CLUMP_PEAK_FRACTION * plan_clumped.n_w / width
    assert plan_clumped.window_padding_overhead == pytest.approx(predicted, rel=0.02)
    assert plan_clumped.max_window_size == pytest.approx(
        _CLUMP_PEAK_FRACTION * _CLUMP_ROWS, rel=0.05
    )

    # The uniform plan stays O(1): its windows differ only by the end ramp.
    assert plan_uniform.window_padding_overhead < 2.0


def test_window_padding_overhead_grows_as_clumps_resolve() -> None:
    """Past the crossover, refining the grid raises the clumped overhead.

    Once each clump sits inside its own windows, further refinement only
    inserts empty planes between them: ``n_w`` grows, ``max_window_size``
    stays pinned to one clump, and a windowed traversal really does pay more
    per useful row. The diagnostic has to say so and keep separating from
    the uniform plan, which stays O(1) throughout. The old nonzero-window
    mean rose too, but so slowly (1.14 -> 2.13 across this sweep) that it
    stayed below the uniform plan for most of it.

    Below the crossover the overhead can legitimately fall -- a plane that
    stops straddling both clumps halves ``max_window_size`` -- so the sweep
    stays in the resolved regime.
    """
    clumped_uvw, uniform_uvw = _clumped_and_uniform_uvw()
    clumped = [
        _clump_plan(clumped_uvw, px).window_padding_overhead for px in _CLUMP_PIXSIZES
    ]
    uniform = [
        _clump_plan(uniform_uvw, px).window_padding_overhead for px in _CLUMP_PIXSIZES
    ]

    assert clumped == sorted(clumped)
    assert clumped[0] > 1.0
    # Roughly a doubling per resolution step here, so an order of magnitude
    # over the sweep; the uniform plan meanwhile stays within a factor of 2
    # of no padding at all.
    assert clumped[-1] > 5.0 * clumped[0]
    assert all(1.0 <= value < 2.0 for value in uniform)


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
