"""nshift regression tests (issue #16): plane-count ratchet, w-plane coverage,
and the compensating-phase exactness contract.

``make_plan`` samples w with ``dw = x0 / max|n-1|`` ([Arras+2021] eq. 12). The
w-phase factor obeys the exact identity

    exp(2*pi*i*w*(n-1)) = exp(2*pi*i*w*(n-1+s)) * exp(-2*pi*i*w*s)

for any constant ``s``. Choosing ``s = nshift = -(nm1_max + nm1_min) / 2``
halves ``max|n-1+s|`` for any image containing the phase centre, so the plane
spacing doubles and ``n_w_inner`` roughly halves; the compensating factor
``exp(-2*pi*i*w*nshift)`` is applied once per visibility per call (forward:
multiplied onto the output; adjoint: the conjugate multiplied onto the input,
matching AGENTS.md sec 1's sign convention).

``tests/test_planning.py::test_nshift_matches_geometry`` and
``::test_plan_is_a_jax_pytree`` cover the plan-construction side of this
(item 1: ``nshift`` matches the geometry; item 5: the plan-field checklist).
This module covers what the *value* of nshift actually buys and costs:

  * item 2 -- the w-plane count on the review fixtures actually drops
    (``test_n_w_drops_on_review_fixtures``);
  * item 3 -- the shifted plane centres still cover every visibility's
    ``w_lambda`` with the kernel's half-width of margin
    (``test_w_centers_span_data_after_shift``);
  * item 4 -- the compensating phase is exact: dropping it, flipping its
    sign, doubling it, or applying it to the wrong operator must not be able
    to hide behind the kernel's own error budget
    (``test_forward_dft_parity_survives_nshift``,
    ``test_adjoint_dft_parity_survives_nshift``,
    ``test_dot_product_identity_survives_nshift``).

The DFT references below are deliberately *not* imported from
``tests/test_against_dft.py`` / ``tests/test_adjoint.py``: both of those call
``jax.config.update("jax_enable_x64", True)`` at import time (so the DFT
reference math itself always runs in float64), which is why
``tests/conftest.py``'s ``collect_ignore`` has to exclude them -- and every
module that imports them -- from the float32 leg. Duplicating the small
loop-based reference here (same pattern as
``tests/test_adjoint.py::_reference_adjoint`` already duplicating
``tests/test_against_dft.py``'s) keeps this module import-clean under
``JAX_ENABLE_X64=0`` without adding anything to that list, which is meant
only to shrink.
"""

from __future__ import annotations

import contextlib

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
from jax_nufft.planning import FLOAT32_EPSILON_FLOOR
from tests.conftest import (
    EDA2,
    MEERKAT,
    MWA_COMPACT,
    MWA_EXTENDED,
    Telescope,
    requires_x64,
    synthetic_uvw,
)

# Accuracy contract against the exact DFT (issue #9; AGENTS.md sec 6). Same
# bound as tests/test_against_dft.py / tests/test_adjoint.py: nshift must not
# cost any of the headroom already measured there.
DFT_TOL_FACTOR = 2.0

# eps-independent bound on the forward/adjoint dot-product identity (issue
# #10; AGENTS.md sec 6). The compensating phases are exact conjugates of one
# another, so this must hold exactly as tightly as it did before nshift.
DOT_PRODUCT_TOL = 1e-11

_REVIEW_EPS = 1e-6


def _make_plan_precision_aware(
    uvw: np.ndarray,
    freq: np.ndarray,
    image_shape: tuple[int, int],
    pixsize_l: float,
    pixsize_m: float,
    eps: float,
    real_dtype: DTypeLike,
):
    """``make_plan`` wrapper shared by the ``n_w`` / ``w_centers`` checks below.

    ``dtype=real_dtype`` (the session fixture from ``tests/conftest.py``) is
    what makes these two tests run under both the float64 and float32 legs
    instead of tripping ``make_plan``'s x64-off guard (issue #11); ``_REVIEW_EPS``
    (1e-6) is below ``FLOAT32_EPSILON_FLOOR`` (1e-5), so under float32
    ``make_plan`` emits the expected accuracy-floor ``UserWarning`` -- caught
    here exactly as ``tests/test_strategies_equivalent.py::_build_problem``
    does, since it is not itself under test: plane *count* and *coverage* are
    dtype-independent structural properties, unlike the accuracy these two
    tests don't otherwise measure.
    """
    warns_ctx = (
        pytest.warns(UserWarning, match="below the accuracy")
        if (real_dtype == jnp.float32 and eps < FLOAT32_EPSILON_FLOOR)
        else contextlib.nullcontext()
    )
    with warns_ctx:
        return make_plan(uvw, freq, image_shape, pixsize_l, pixsize_m, eps, dtype=real_dtype)


def _fixture_id(values: tuple[Telescope, float]) -> str:
    tel, ang = values
    return f"{tel.name}_{'zenith' if ang == 0.0 else f'off{int(ang)}'}"


# --------------------------------------------------------------------------
# Item 2: the plane count actually drops
# --------------------------------------------------------------------------

# n_w measured on the seven review fixtures at eps=1e-6, against commit
# 19efe19 (the tip of issue-16-nshift immediately before this change) --
# i.e. make_plan's behaviour with no nshift at all. Reproduce with:
#
#   make_plan(synthetic_uvw(tel, angle, seed=0), np.array([tel.freq_hz]),
#             (tel.n_pix, tel.n_pix), tel.pixsize, tel.pixsize, 1e-6).n_w
#
# This is the "before" column for the PR's before/after table.
_PRE_CHANGE_N_W: dict[str, int] = {
    "EDA2_zenith": 20,
    "MWA_compact_zenith": 8,
    "MWA_compact_off30": 26,
    "MWA_extended_zenith": 21,
    "MWA_extended_off30": 495,
    "MeerKAT_zenith": 9,
    "MeerKAT_off30": 30,
}

# Upper bound on n_w *after* nshift. MWA_extended_off30's bound is issue
# #16's own number: floor(0.55 * 495) = 272 (predicted by the "halve
# max|n-1+s|" rule: ~251). The other four bounds below are each set with a
# few planes of slack above the same rule's prediction (EDA2_zenith ~14,
# MWA_compact_off30 ~17, MWA_extended_zenith ~14, MeerKAT_off30 ~19) while
# staying strictly below the pre-change n_w above, so this is a real
# ratchet rather than a trivially-satisfiable one. MWA_compact_zenith and
# MeerKAT_zenith are omitted from this table (see the fallback branch in
# the test below): their n_w_inner is already pinned at the "always have at
# least one interior step" floor in make_plan, so halving max|n-1+s| does
# not change n_w at all -- there is nothing to ratchet for them.
_MAX_N_W_AFTER_NSHIFT: dict[str, int] = {
    "EDA2_zenith": 16,
    "MWA_compact_off30": 20,
    "MWA_extended_zenith": 17,
    "MWA_extended_off30": 272,
    "MeerKAT_off30": 23,
}

_SEVEN_REVIEW_FIXTURES: tuple[tuple[Telescope, float], ...] = (
    (EDA2, 0.0),
    (MWA_COMPACT, 0.0),
    (MWA_COMPACT, 30.0),
    (MWA_EXTENDED, 0.0),
    (MWA_EXTENDED, 30.0),
    (MEERKAT, 0.0),
    (MEERKAT, 30.0),
)


@pytest.mark.parametrize(
    ("telescope", "zenith_angle_deg"),
    _SEVEN_REVIEW_FIXTURES,
    ids=[_fixture_id(v) for v in _SEVEN_REVIEW_FIXTURES],
)
def test_n_w_drops_on_review_fixtures(
    telescope: Telescope, zenith_angle_deg: float, real_dtype: DTypeLike
) -> None:
    """issue #16 item 2: n_w must actually shrink on the review fixtures.

    Ratchets against the pre-nshift baseline hard-coded in
    ``_PRE_CHANGE_N_W`` (measured against commit 19efe19) rather than
    comparing two live plans, so this fails today with "n_w unchanged"
    (the current code has no nshift at all) and would keep failing if a
    future change quietly regressed the plane count back up.
    """
    name = _fixture_id((telescope, zenith_angle_deg))
    uvw = synthetic_uvw(telescope, zenith_angle_deg, seed=0)
    freq = np.array([telescope.freq_hz])
    plan = _make_plan_precision_aware(
        uvw,
        freq,
        (telescope.n_pix, telescope.n_pix),
        telescope.pixsize,
        telescope.pixsize,
        _REVIEW_EPS,
        real_dtype,
    )

    pre = _PRE_CHANGE_N_W[name]
    print(f"{name}: n_w {pre} -> {plan.n_w}")

    bound = _MAX_N_W_AFTER_NSHIFT.get(name)
    if bound is None:
        # MWA_compact_zenith / MeerKAT_zenith: n_w_inner is already at the
        # "at least one interior step" floor, so there is nothing to
        # ratchet -- just confirm nshift never makes n_w *grow*.
        assert plan.n_w <= pre
        return
    assert plan.n_w <= bound, (
        f"{name}: n_w={plan.n_w} did not drop below {bound} (pre-nshift "
        f"baseline was {pre}, measured against commit 19efe19)"
    )


# --------------------------------------------------------------------------
# Item 3: w_centers still span the data after the shift
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("telescope", "zenith_angle_deg"),
    [(EDA2, 0.0), (MWA_EXTENDED, 30.0), (MEERKAT, 30.0)],
    ids=["EDA2_zenith", "MWA_extended_off30", "MeerKAT_off30"],
)
def test_w_centers_span_data_after_shift(
    telescope: Telescope, zenith_angle_deg: float, real_dtype: DTypeLike
) -> None:
    """issue #16 item 3: after halving the plane-spacing denominator, the
    w-plane centres must still cover every visibility's ``w_lambda`` with
    (at least) the kernel's half-width of margin -- not just have the right
    *count*. Getting ``n_w`` right while botching the ``w_min_all`` / ``dw``
    bookkeeping that places the centres could still leave data outside
    ``[centers[0] - margin, centers[-1] + margin]``, silently corrupting
    the edge planes; that would not show up in a test that only checks
    ``n_w`` (item 2) or ``nshift`` (item 1) in isolation.
    """
    uvw = synthetic_uvw(telescope, zenith_angle_deg, seed=0)
    freq = np.array([telescope.freq_hz])
    plan = _make_plan_precision_aware(
        uvw,
        freq,
        (telescope.n_pix, telescope.n_pix),
        telescope.pixsize,
        telescope.pixsize,
        _REVIEW_EPS,
        real_dtype,
    )
    # On these three fixtures the shift is structurally significant (see
    # test_planning.py::test_nshift_matches_geometry). Asserting it here
    # forces this test to fail loudly -- missing field -- rather than
    # silently passing against a plan that has no nshift applied at all;
    # the coverage property below already holds unconditionally of nshift
    # (AGENTS.md sec 4's window-builder invariant), so without this line
    # the test would prove nothing about the shift specifically.
    assert plan.nshift != 0.0

    w_lambda = np.asarray(plan.uvw_lambda)[..., 2]
    w_min = float(w_lambda.min())
    w_max = float(w_lambda.max())
    centers = np.asarray(plan.w_centers)
    margin = plan.w_kernel_scale  # (W/2) * dw -- the kernel support half-width

    assert centers[0] - margin <= w_min, (
        f"plane centres start at {centers[0]:.6g} (margin {margin:.3g}) but "
        f"data reaches w_lambda={w_min:.6g}"
    )
    assert centers[-1] + margin >= w_max, (
        f"plane centres end at {centers[-1]:.6g} (margin {margin:.3g}) but "
        f"data reaches w_lambda={w_max:.6g}"
    )


# --------------------------------------------------------------------------
# Item 4: the compensating phase is exact
# --------------------------------------------------------------------------
#
# A single small, fast, off-zenith problem, reused by all three tests below.
# It is engineered (not just off-zenith) to make w * nshift range over many
# full cycles of 2*pi: nshift ~= 0.235 here (nm1_min ~= -0.471, and the image
# is centred so nm1_max == 0 exactly), and w_lambda ranges roughly
# +/-15..64, so w * nshift ranges roughly -15..+15. A dropped, wrong-signed,
# or doubled compensating factor introduces a phase error of magnitude
# 2*pi*w*nshift or 4*pi*w*nshift that varies across all 40 rows' different w
# values -- there is no single s for which that error is simultaneously
# near a multiple of 2*pi for every row, so it cannot hide inside the
# kernel's own eps budget. The image stays entirely inside the unit disc
# (min n_grid ~= 0.53 > 0), matching the "no masking needed" precondition of
# tests/test_adjoint.py::test_dot_product_identity's derivation.


def _nshift_sensitive_problem() -> tuple[np.ndarray, np.ndarray, tuple[int, int], float, float]:
    rng = np.random.default_rng(42)
    n_l = n_m = 24
    n_rows = 40
    pixsize = 0.05
    eps = 1e-6

    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-20.0, 20.0, size=n_rows)
    freq = np.array([1.0e9])
    return uvw, freq, (n_l, n_m), pixsize, eps


def _reference_lmn_grids(
    image_shape: tuple[int, int], pixsize_l: float, pixsize_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(l, m, n - 1)`` on the image grid, matching ``planning.make_plan``.

    See ``tests/test_against_dft.py::reference_lmn_grids`` for the same
    computation (duplicated here -- see the module docstring for why).
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

    Unshifted -- i.e. exactly the target the nshift identity guarantees a
    correct implementation still reproduces. See
    ``tests/test_against_dft.py::_reference_forward`` for the same
    computation (duplicated here -- see the module docstring for why).
    """
    n_chan, n_l, n_m = image.shape
    n_rows = uvw.shape[0]
    ll, mm, nm1 = _reference_lmn_grids((n_l, n_m), pixsize_l, pixsize_m)
    out = np.zeros((n_rows, n_chan), dtype=np.complex128)
    for c in range(n_chan):
        scale = freq[c] / SPEED_OF_LIGHT
        u = uvw[:, 0] * scale
        v = uvw[:, 1] * scale
        w = uvw[:, 2] * scale
        for r in range(n_rows):
            phase = -2j * np.pi * (u[r] * ll + v[r] * mm - w[r] * nm1)
            out[r, c] = np.sum(image[c] * np.exp(phase))
    return out


def _reference_adjoint(
    vis: np.ndarray,
    uvw: np.ndarray,
    freq: np.ndarray,
    image_shape: tuple[int, int],
    pixsize_l: float,
    pixsize_m: float,
) -> np.ndarray:
    """Direct DFT adjoint matching ducc's explicit_gridder (divide_by_n=True).

    Unshifted, for the same reason as ``_reference_forward`` above. See
    ``tests/test_adjoint.py::_reference_adjoint`` for the same computation
    (duplicated here -- see the module docstring for why).
    """
    n_l, n_m = image_shape
    n_rows, n_chan = vis.shape
    ll, mm, nm1 = _reference_lmn_grids(image_shape, pixsize_l, pixsize_m)
    n_grid = nm1 + 1.0
    out = np.zeros((n_chan, n_l, n_m), dtype=np.float64)
    for c in range(n_chan):
        scale = freq[c] / SPEED_OF_LIGHT
        u = uvw[:, 0] * scale
        v = uvw[:, 1] * scale
        w = uvw[:, 2] * scale
        for r in range(n_rows):
            phase = +2j * np.pi * (u[r] * ll + v[r] * mm - w[r] * nm1)
            out[c] += (vis[r, c] * np.exp(phase)).real
    return np.where(n_grid > 0.0, out / np.maximum(n_grid, 1e-30), 0.0)


@requires_x64
def test_forward_dft_parity_survives_nshift() -> None:
    """issue #16 item 4: a wrong-signed, dropped, or doubled compensating
    phase on the *forward* operator must not be able to hide.

    The w-phase identity nshift relies on is exact, so a correct
    implementation reproduces the unshifted exact DFT to exactly the
    ``2 * eps`` contract used by tests/test_against_dft.py -- unchanged by
    nshift, just measured here on a fixture engineered to make w * nshift
    large (see the section docstring above).
    """
    uvw, freq, image_shape, pixsize, eps = _nshift_sensitive_problem()
    rng = np.random.default_rng(7)
    image = rng.standard_normal((1, *image_shape))

    plan = make_plan(uvw, freq, image_shape, pixsize, pixsize, eps)
    # Confirms the fixture is actually exercising a large shift, and forces
    # a loud "missing field" failure until nshift lands.
    assert abs(plan.nshift) > 0.1

    got = np.asarray(dirty2vis(plan, jnp.asarray(image)))
    want = _reference_forward(image.astype(np.complex128), uvw, freq, pixsize, pixsize)

    err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    assert err < DFT_TOL_FACTOR * eps, (
        f"relative error {err:.3e} exceeds {DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e} "
        "-- a compensating-phase bug (dropped / wrong sign / doubled) would show up exactly "
        "like this"
    )


@requires_x64
def test_adjoint_dft_parity_survives_nshift() -> None:
    """Mirrors ``test_forward_dft_parity_survives_nshift`` for the adjoint.

    The adjoint applies the *conjugate* compensating factor to its input
    ``vis`` (AGENTS.md sec 1 sign convention); this is the check that would
    catch the compensation being applied with the forward's sign instead of
    its own, or to the wrong operator entirely.
    """
    uvw, freq, image_shape, pixsize, eps = _nshift_sensitive_problem()
    rng = np.random.default_rng(8)
    n_rows = uvw.shape[0]
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )

    plan = make_plan(uvw, freq, image_shape, pixsize, pixsize, eps)
    assert abs(plan.nshift) > 0.1

    got = np.asarray(vis2dirty(plan, jnp.asarray(vis)))
    want = _reference_adjoint(vis, uvw, freq, image_shape, pixsize, pixsize)

    err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    assert err < DFT_TOL_FACTOR * eps, (
        f"relative error {err:.3e} exceeds {DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e} "
        "-- a compensating-phase bug (dropped / wrong sign / doubled / wrong operator) would "
        "show up exactly like this"
    )


@requires_x64
def test_dot_product_identity_survives_nshift() -> None:
    """issue #16 item 4: forward/adjoint adjointness must hold exactly as
    tightly as before nshift.

    The forward's ``exp(-2*pi*i*w*nshift)`` and the adjoint's
    ``exp(+2*pi*i*w*nshift)`` are exact conjugates of one another, so this
    eps-independent bound (unchanged from
    tests/test_adjoint.py::test_dot_product_identity, see that test's
    docstring for the full derivation of the identity used here) must still
    hold: applying the compensation to only one operator, with mismatched
    signs, or twice on one side, breaks the conjugate-pair cancellation and
    this identity is exactly the thing that would catch it, even if each
    operator's own DFT parity happened to look fine in isolation.
    """
    uvw, freq, image_shape, pixsize, eps = _nshift_sensitive_problem()
    rng = np.random.default_rng(9)
    n_rows = uvw.shape[0]

    plan = make_plan(uvw, freq, image_shape, pixsize, pixsize, eps)
    assert abs(plan.nshift) > 0.1

    image = rng.standard_normal((1, *image_shape))
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )

    # n_grid multiplier undoes the adjoint's 1/n (see
    # tests/test_adjoint.py::test_dot_product_identity for the derivation);
    # valid without masking because _nshift_sensitive_problem keeps every
    # pixel inside the unit disc (see the section docstring above).
    n_grid = np.asarray(plan.n_minus_1) + 1.0
    image_n = image * n_grid[None, :, :]

    Ax = np.asarray(dirty2vis(plan, jnp.asarray(image)))
    Ay = np.asarray(vis2dirty(plan, jnp.asarray(vis)))

    lhs = np.vdot(Ax.ravel(), vis.ravel()).real
    rhs = float(np.vdot(image_n.ravel(), Ay.ravel()))
    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert rel_err < DOT_PRODUCT_TOL, (
        f"dot-product relative error {rel_err:.3e}; lhs={lhs}, rhs={rhs}"
    )
