"""Hermitian w-sign fold (issue #17): plan structure, plane count, correctness.

For a real sky the visibility function is conjugate-symmetric ([TMS2017] ch. 3;
[Arras+2021] sec. 2 eq. 19 states it for the w-stacking measurement equation):

    V(-u, -v, -w) = conj(V(u, v, w)).

So every row with ``w < 0`` can be folded onto ``(-u, -v, -w)`` and its value
conjugated. The w-range becomes ``[0, max|w|]`` instead of ``[min w, max w]``,
which halves the w-extent -- and with it the inner plane count
``n_w_inner = ceil(w_extent * max|n-1+nshift| / x0)`` -- on any roughly
symmetric w-distribution.

Three things about the fold shape every test in this module:

* **It is not a reduction-order change.** ``hermitian=True`` and
  ``hermitian=False`` are two *different discretisations* of the same
  continuous operator: different ``n_w``, different plane centres, different
  kernel weights. They therefore agree to the accuracy contract
  (``2 * eps`` each against the exact DFT, hence ``4 * eps`` against each
  other by the triangle inequality), **not** to the 1e-11 strategy-equivalence
  bound of AGENTS.md sec 6, which is reserved for mathematically identical
  operators that differ only in summation order. Measured worst case over the
  seven review fixtures x eps in {1e-4, 1e-6, 1e-8, 1e-10}, forward and
  adjoint: 2.15x eps (MWA_extended off30, eps=1e-10). The 1e-11 bound *does*
  apply across the four ``w_strategy`` values at a fixed ``hermitian``, which
  is what ``test_all_strategies_agree_under_hermitian`` asserts.

* **The forward conjugates its output, the adjoint conjugates its input.**
  Forward: the machinery evaluates ``V(-u,-v,-w)`` on a folded row, so the
  answer is its conjugate. Adjoint: the summand for a folded row obeys
  ``vis_r * phase_r = conj(conj(vis_r) * phase_folded_r)`` and the output takes
  the real part, so feeding ``conj(vis_r)`` through the folded machinery is
  exact. Getting these two the wrong way round is the subtle bug this module
  exists to catch, and it is why ``test_fold_conjugates_the_output_of_folded_
  rows`` / ``..._the_input_of_folded_rows`` check the *flipped* and *unflipped*
  row groups separately rather than only in aggregate.

* **The per-row cost is one byte.** Issue #23 (PR #42) cut the plan from
  4.1 GB to 30 MB on a 64-channel 1M-row plan by deleting every per-(channel,
  row) array, and ``tests/test_planning.py`` gates that hard. The fold needs
  per-row sign information, which is inherently ``(n_rows,)`` -- that is
  allowed, but it must be **one byte per row** (``flip_sign``, int8) and it
  must never acquire a channel axis: the sign of ``w`` is frequency-independent
  (``freq/c > 0``), so a per-channel flip array is both redundant and a
  regression of issue #23.

Nothing in this module depends on what ``make_plan``'s ``hermitian`` **default**
is: every plan is built with the argument passed explicitly. The one test that
does touch the default -- ``test_default_plan_is_never_silently_wrong_on_a_
complex_image`` -- accepts either policy and only forbids the unsafe outcome.

The DFT references below are deliberately *not* imported from
``tests/test_against_dft.py`` / ``tests/test_adjoint.py``: both of those call
``jax.config.update("jax_enable_x64", True)`` at import time, which is why
``tests/conftest.py``'s ``collect_ignore`` has to exclude them -- and every
module importing them -- from the float32 leg. Duplicating the small loop-based
reference here (exactly as ``tests/test_nshift.py`` already does) keeps this
module import-clean under ``JAX_ENABLE_X64=0`` without adding anything to a list
that is only meant to shrink.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
from jax_nufft.kernel import kernel_params
from jax_nufft.planning import FLOAT32_EPSILON_FLOOR, W_OVERSAMPLE_X0, WGridderPlan
from tests.conftest import (
    EDA2,
    MEERKAT,
    MWA_COMPACT,
    MWA_EXTENDED,
    Telescope,
    requires_x64,
    synthetic_uvw,
)

# Accuracy contract against the exact DFT (issue #9; AGENTS.md sec 6). The fold
# is an exact identity, so it must not cost any of the headroom already
# measured there.
DFT_TOL_FACTOR = 2.0

# ``hermitian=True`` vs ``hermitian=False``. NOT the 1e-11 strategy-equivalence
# bound: the two are different discretisations, not different summation orders
# (see the module docstring). The triangle inequality on the ``2 * eps``
# contract each of them meets gives ``4 * eps``; measured worst case 2.15x eps,
# so the headroom is ~1.9x. If this bound is ever exceeded, the DFT contract
# itself is at risk and the fix is not to loosen this number.
CROSS_PATH_TOL_FACTOR = 2.0 * DFT_TOL_FACTOR

# eps-independent strategy-equivalence bound (issue #10; AGENTS.md sec 6). This
# one *is* a reduction-order comparison: same plan, same fold, four ways of
# walking the same w-planes.
STRATEGY_TOL = 1e-11

# The eight leaves the plan had before this issue (issue #23 / #16 shapes).
# ``flip_sign`` is issue #17's addition and is checked against this set by
# name-agnostic difference, so the tests below still gate the byte cost if the
# implementation names the new leaf something else.
_PRE_ISSUE_17_LEAVES: tuple[str, ...] = (
    "uvw_m",
    "inv_lambda",
    "w_centers_rel",
    "n_minus_1_shifted",
    "w0_screen",
    "phi_hat_n",
    "sort_perm",
    "window_start",
)

# Bytes per row the plan's leaves may cost, post issue #17:
#   uvw_m      (n_rows, 3) real_dtype   24 (float64)
#   sort_perm  (n_rows,)   int32         4
#   flip_sign  (n_rows,)   int8          1
_MAX_LEAF_BYTES_PER_ROW_F64 = 29

_REVIEW_EPS = 1e-6

_SEVEN_REVIEW_FIXTURES: tuple[tuple[Telescope, float], ...] = (
    (EDA2, 0.0),
    (MWA_COMPACT, 0.0),
    (MWA_COMPACT, 30.0),
    (MWA_EXTENDED, 0.0),
    (MWA_EXTENDED, 30.0),
    (MEERKAT, 0.0),
    (MEERKAT, 30.0),
)


def _fixture_id(values: tuple[Telescope, float]) -> str:
    tel, ang = values
    return f"{tel.name}_{'zenith' if ang == 0.0 else f'off{int(ang)}'}"


_REVIEW_IDS = [_fixture_id(v) for v in _SEVEN_REVIEW_FIXTURES]

# ``n_w`` at eps=1e-6 (W=7) on the seven review fixtures, measured on this
# branch with a working fold. Both columns are pinned so that a regression in
# EITHER direction is caught: the ``hermitian=False`` column would move if the
# fold leaked into the unfolded path, and the ``hermitian=True`` column is the
# issue's own deliverable.
#
#   fixture                n_w off   n_w on   n_w ratio   w_extent ratio
#   EDA2_zenith                 14       11       0.786            0.527
#   MWA_compact_zenith           8        8       1.000            0.532
#   MWA_compact_off30           17       12       0.706            0.519
#   MWA_extended_zenith         14       11       0.786            0.532
#   MWA_extended_off30         251      134       0.534            0.518
#   MeerKAT_zenith               8        8       1.000            0.532
#   MeerKAT_off30               19       13       0.684            0.519
#
# Read the two ratio columns together: the *w-extent* halves on all seven, which
# is the property the fold actually delivers, but ``n_w = n_w_inner + W`` adds a
# flat ``W = 7`` planes that the fold cannot touch. Where ``n_w_inner`` is small
# that constant dominates the total, so the 0.55x gate on the full ``n_w`` is
# only reachable where ``n_w_inner >> W`` -- MWA_extended off30, the fixture the
# issue is written about. Gating the full ``n_w`` at 0.55x on the other six
# would be gating the kernel width, not the fold; they are gated on the extent
# and on the inner count instead. See ``test_plane_count_halves_on_review_
# fixtures``.
_N_W_HERMITIAN_OFF: dict[str, int] = {
    "EDA2_zenith": 14,
    "MWA_compact_zenith": 8,
    "MWA_compact_off30": 17,
    "MWA_extended_zenith": 14,
    "MWA_extended_off30": 251,
    "MeerKAT_zenith": 8,
    "MeerKAT_off30": 19,
}
_N_W_HERMITIAN_ON: dict[str, int] = {
    "EDA2_zenith": 11,
    "MWA_compact_zenith": 8,
    "MWA_compact_off30": 12,
    "MWA_extended_zenith": 11,
    "MWA_extended_off30": 134,
    "MeerKAT_zenith": 8,
    "MeerKAT_off30": 13,
}

# The fraction of the unfolded w-extent the folded one must not exceed. 0.55
# rather than 0.5 because the synthetic w-distributions are only approximately
# symmetric (measured 0.518-0.532 across the seven).
W_EXTENT_RATIO_GATE = 0.55


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_plan_precision_aware(
    uvw: np.ndarray,
    freq: np.ndarray,
    image_shape: tuple[int, int],
    pixsize_l: float,
    pixsize_m: float,
    eps: float,
    real_dtype: DTypeLike,
    *,
    hermitian: bool,
) -> WGridderPlan:
    """``make_plan`` wrapper that runs under both precision legs.

    Same device as ``tests/test_nshift.py::_make_plan_precision_aware``: plane
    *count*, plane *coverage* and plan *structure* are dtype-independent
    properties, so the structural tests below run on the float32 leg too;
    ``_REVIEW_EPS`` (1e-6) is under ``FLOAT32_EPSILON_FLOOR`` (1e-5), so
    ``make_plan`` emits its accuracy-floor ``UserWarning`` there, which
    ``filterwarnings = ["error"]`` would otherwise turn into a failure.
    """
    warns_ctx = (
        pytest.warns(UserWarning, match="below the accuracy")
        if (real_dtype == jnp.float32 and eps < FLOAT32_EPSILON_FLOOR)
        else contextlib.nullcontext()
    )
    with warns_ctx:
        return make_plan(
            uvw,
            freq,
            image_shape,
            pixsize_l,
            pixsize_m,
            eps,
            dtype=real_dtype,
            hermitian=hermitian,
        )


def _nm1_grid(n_l: int, n_m: int, pixsize_l: float, pixsize_m: float) -> np.ndarray:
    """``n - 1`` on the image grid, including ducc's analytic extension.

    Written out here rather than imported from ``jax_nufft.planning`` so the
    ``n_w`` prediction in :func:`_independent_plan_geometry` is an independent
    reference rather than a restatement of the code under test.
    """
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    ll = (i * pixsize_l)[:, None]
    mm = (j * pixsize_m)[None, :]
    r2 = ll * ll + mm * mm
    inside = r2 <= 1.0
    return np.where(
        inside,
        np.sqrt(np.where(inside, 1.0 - r2, 0.0)) - 1.0,
        -np.sqrt(np.where(inside, 0.0, r2 - 1.0)) - 1.0,
    )


def _independent_plan_geometry(
    uvw: np.ndarray,
    freq: np.ndarray,
    image_shape: tuple[int, int],
    pixsize_l: float,
    pixsize_m: float,
    epsilon: float,
    *,
    hermitian: bool,
    real_dtype: DTypeLike = np.float64,
) -> tuple[float, float, int]:
    """``(nshift, w_extent, n_w)`` computed from scratch in numpy.

    Reimplements the whole chain issue #16's ``nshift`` and issue #17's fold
    both feed into:

        nshift      = -(max(n-1) + min(n-1)) / 2                       (#16)
        w           = (freq/c) * (|w_m| if folding else w_m)           (#17)
        n_w_inner   = max(ceil((max w - min w) * max|n-1+nshift| / x0), 1)
        n_w         = n_w_inner + W

    so a plan that quietly drops *either* device -- the fold, or the nshift
    centring -- lands on a different ``n_w`` than this predicts. That is what
    makes it a composition test rather than two independent ones.

    ``real_dtype`` is applied where ``make_plan`` applies it -- the ``n-1``
    grid, the baselines, ``freq/c`` -- so the prediction is exact on the float32
    leg too; the reductions themselves run in float64, as they do there.
    """
    dt = np.dtype(real_dtype)
    nm1 = np.asarray(_nm1_grid(*image_shape, pixsize_l, pixsize_m), dtype=dt)
    nshift = -(float(nm1.max()) + float(nm1.min())) / 2.0
    max_abs_nm1 = max(abs(float(nm1.max()) + nshift), abs(float(nm1.min()) + nshift))

    w_m = np.asarray(uvw[:, 2], dtype=dt)
    if hermitian:
        w_m = np.abs(w_m)
    inv_lambda = (np.asarray(freq, dtype=dt) / SPEED_OF_LIGHT).astype(dt)
    w = inv_lambda[:, None] * w_m[None, :]
    w_extent = float(w.max()) - float(w.min())

    kernel_width, _ = kernel_params(epsilon)
    n_w_inner = max(math.ceil(w_extent * max_abs_nm1 / W_OVERSAMPLE_X0), 1)
    return nshift, w_extent, n_w_inner + kernel_width


def _reference_forward(
    image: np.ndarray,
    uvw: np.ndarray,
    freq: np.ndarray,
    pixsize_l: float,
    pixsize_m: float,
) -> np.ndarray:
    """Exact DFT forward in the repo's sign convention (AGENTS.md sec 1).

    Mirrors ``tests/test_against_dft.py::_reference_forward``; duplicated for
    the import-hygiene reason given in the module docstring.
    """
    n_chan, n_l, n_m = image.shape
    n_rows = uvw.shape[0]
    nm1 = _nm1_grid(n_l, n_m, pixsize_l, pixsize_m)
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    ll = (i * pixsize_l)[:, None]
    mm = (j * pixsize_m)[None, :]
    out = np.zeros((n_rows, n_chan), dtype=np.complex128)
    for c in range(n_chan):
        scale = freq[c] / SPEED_OF_LIGHT
        u, v, w = (uvw[:, k] * scale for k in range(3))
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
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Exact DFT adjoint with ``divide_by_n=True``.

    Mirrors ``tests/test_adjoint.py::_reference_adjoint``, including its use of
    the *clipped* ``n - 1`` (no analytic extension) and its ``n > 0`` mask.
    """
    n_l, n_m = image_shape
    n_rows, n_chan = vis.shape
    i = np.arange(n_l) - n_l // 2
    j = np.arange(n_m) - n_m // 2
    ll = (i * pixsize_l)[:, None]
    mm = (j * pixsize_m)[None, :]
    nm1 = np.sqrt(np.maximum(1.0 - ll**2 - mm**2, 0.0)) - 1.0
    n_grid = nm1 + 1.0
    out = np.zeros((n_chan, n_l, n_m), dtype=np.float64)
    for c in range(n_chan):
        scale = freq[c] / SPEED_OF_LIGHT
        u, v, w = (uvw[:, k] * scale for k in range(3))
        for r in range(n_rows):
            phase = +2j * np.pi * (u[r] * ll + v[r] * mm - w[r] * nm1)
            v_eff = vis[r, c] if weights is None else vis[r, c] * weights[r, c]
            out[c] += (v_eff * np.exp(phase)).real
    return np.where(n_grid > 0.0, out / np.maximum(n_grid, 1e-30), 0.0)


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _folded_rows(plan: WGridderPlan) -> np.ndarray:
    """Boolean mask of the rows the plan folded, from its ``flip_sign`` leaf."""
    return np.asarray(plan.flip_sign) < 0


def _asymmetric_w_case(
    n_rows: int = 45,
    n_pix: int = 16,
    seed: int = 517,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """A deliberately *asymmetric* w-distribution: ``(uvw, image, vis, freq, pixsize)``.

    Two properties matter and both are asserted at the use sites rather than
    trusted here:

      * a substantial minority of rows have ``w < 0``, so the fold is exercised;
      * that minority is not half, so "conjugate the folded rows" and
        "conjugate the *un*folded rows" are not related by any symmetry of the
        fixture -- a test on a 50/50 split could in principle be passed by the
        complementary bug on a lucky cancellation, and this one cannot.

    The image is real (the fold's precondition) and the visibilities are
    complex with an imaginary part of the same magnitude as the real part, so a
    dropped or misplaced conjugation is an O(1) error rather than a small one.
    """
    rng = np.random.default_rng(seed)
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-120.0, 120.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-120.0, 120.0, size=n_rows)
    # Skewed towards positive w: about a third of the rows land below zero.
    uvw[:, 2] = rng.uniform(-20.0, 40.0, size=n_rows)
    freq = np.array([1.0e9])
    image = rng.standard_normal((1, n_pix, n_pix))
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )
    return uvw, image, vis, freq, 0.01


def _windowing_w_case(
    n_rows: int = 300,
    n_pix: int = 16,
    seed: int = 3117,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Like :func:`_asymmetric_w_case`, but wide enough that the windows bite.

    ``_asymmetric_w_case`` is 45 rows over a narrow w-range, so every window
    holds every row (``max_window_size == n_rows``) and the windowed strategies
    degenerate into the dense ones: they still exercise the *row order* the fold
    is applied in -- the scatter, the unsort, the ``sort_perm`` gather -- but
    not ``window_start`` itself. A bug that builds the window boundaries from
    the *unfolded* w is then invisible to every strategy comparison, which is
    not a hypothetical: measured, that mutation passes the whole of
    ``tests/test_strategies_equivalent.py`` (whose short fixtures are also
    non-windowing here) and is caught only by
    ``tests/test_planning.py::test_window_builder_basic``.

    This fixture is 300 rows with w in ``[-120, 300]`` metres: 85 rows fold, and
    the widest window holds 166-231 of the 300 rows depending on epsilon, so a
    misplaced boundary really does drop rows. The use sites assert
    ``max_window_size < n_rows`` rather than trusting that.
    """
    rng = np.random.default_rng(seed)
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-150.0, 150.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-150.0, 150.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-120.0, 300.0, size=n_rows)
    freq = np.array([1.0e9])
    image = rng.standard_normal((1, n_pix, n_pix))
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )
    return uvw, image, vis, freq, 0.01


_FOLD_CASES = {"narrow": _asymmetric_w_case, "windowing": _windowing_w_case}


def _row_bytes_and_other_bytes(plan: WGridderPlan) -> tuple[int, int]:
    """Split the plan's leaf bytes into row-proportional and everything else.

    ``other`` is the exact sum of the non-row leaves as ``_footprint_bound_bytes``
    in ``tests/test_planning.py`` accounts for them (``inv_lambda``,
    ``w_centers_rel``, the three image-sized arrays, ``window_start``), so the
    remainder is precisely what the plan spends per row. Computed from the plan's
    own static fields rather than from leaf names, so a renamed or added row leaf
    still shows up in the row half.
    """
    total = sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(plan))
    real_itemsize = plan.real_dtype.itemsize
    other = (
        real_itemsize * plan.n_chan  # inv_lambda
        + real_itemsize * plan.n_w  # w_centers_rel
        + real_itemsize * plan.n_l * plan.n_m  # n_minus_1_shifted
        + plan.complex_dtype.itemsize * plan.n_l * plan.n_m  # w0_screen
        + real_itemsize * plan.n_l * plan.n_m  # phi_hat_n
        + 4 * plan.n_chan * plan.n_w  # window_start (int32)
    )
    return total - other, other


# ---------------------------------------------------------------------------
# 1. plan structure: the fold leaf and the static option
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hermitian", [False, True])
def test_flip_sign_leaf_is_one_signed_byte_per_row(hermitian: bool, real_dtype: DTypeLike) -> None:
    """``flip_sign`` is ``(n_rows,)`` int8 in ``{+1, -1}``, and marks ``w < 0``.

    Four separate regressions are covered, and all four have to be, because
    each is individually silent:

      * a per-*channel* flip array (``(n_chan, n_rows)``) -- correct output,
        ``n_chan`` bytes per row instead of one, and a direct regression of
        issue #23. The sign of ``w`` is frequency-independent
        (``inv_lambda = freq/c > 0`` is monotonic), so the channel axis carries
        no information at all.
      * a float leaf -- correct output, 8 bytes per row instead of one.
      * folding the wrong sign (``w > 0``). That is *mathematically* just as
        valid -- the identity is symmetric, and the resulting visibilities are
        identical -- so no accuracy test anywhere can distinguish it. What
        makes it a bug is that it contradicts the documented plan semantics
        (``[0, max|w|]``, ``plan.uvw_m[:, 2] >= 0``), which the next test and
        the ``w_extent`` gates are written against. It is pinned here, where
        the convention lives, and nowhere else.
      * folding when ``hermitian=False`` -- checked by the ``False`` leg, which
        requires every sign to be ``+1``.

    ``n_chan = 5`` and ``n_rows = 257`` are mutually prime and unlike every
    other dimension in the problem, so ``(n_rows,)`` cannot be confused with a
    per-channel or per-plane array (same device as
    ``tests/test_planning.py::test_no_operator_path_materialises_per_channel_
    row_coordinates``).
    """
    n_rows, n_chan = 257, 5
    rng = np.random.default_rng(17)
    uvw = rng.normal(scale=300.0, size=(n_rows, 3))
    freq = np.linspace(1.0e9, 2.0e9, n_chan)
    plan = _make_plan_precision_aware(
        uvw, freq, (16, 16), 2e-3, 2e-3, _REVIEW_EPS, real_dtype, hermitian=hermitian
    )

    flip = np.asarray(plan.flip_sign)
    assert flip.shape == (n_rows,), (
        f"flip_sign has shape {flip.shape}, expected ({n_rows},). A ({n_chan}, {n_rows}) "
        "array is a per-(channel, row) leaf -- exactly what issue #23 removed -- and the "
        "w sign is frequency-independent, so the channel axis carries no information"
    )
    assert flip.dtype == np.int8, (
        f"flip_sign has dtype {flip.dtype}, expected int8: the fold needs ONE BYTE per "
        "row (issue #23's memory budget), and a float array costs eight"
    )
    assert set(np.unique(flip).tolist()) <= {-1, 1}, (
        f"flip_sign must contain only +1 / -1; got {np.unique(flip).tolist()}"
    )

    expected = (
        np.where(uvw[:, 2] < 0, -1, 1).astype(np.int8)
        if hermitian
        else np.ones(n_rows, dtype=np.int8)
    )
    np.testing.assert_array_equal(
        flip,
        expected,
        err_msg=(
            "flip_sign does not mark exactly the rows with w < 0"
            if hermitian
            else "hermitian=False must fold nothing: every flip_sign entry must be +1"
        ),
    )
    # The fixture has to contain both kinds of row, or the comparison above is
    # satisfied by any constant array.
    assert 0 < int((uvw[:, 2] < 0).sum()) < n_rows, "fixture has no mixed-sign w"


@pytest.mark.parametrize("hermitian", [False, True])
def test_stored_baselines_are_the_folded_ones(hermitian: bool, real_dtype: DTypeLike) -> None:
    """``plan.uvw_m == flip_sign[:, None] * uvw``, all three columns.

    The fold is ``(u, v, w) -> (-u, -v, -w)``: negating ``w`` alone is *not* the
    symmetry, it is a different (and wrong) operator, and it would still halve
    the w-extent and still pass every plane-count gate in this file. Checking
    all three columns is what separates the two.

    The ``w >= 0`` assertion pins the ``[0, max|w|]`` convention the issue
    specifies; see the previous test on why that cannot be inferred from any
    accuracy measurement.
    """
    n_rows = 257
    rng = np.random.default_rng(1717)
    uvw = rng.normal(scale=300.0, size=(n_rows, 3))
    freq = np.array([1.4e9])
    plan = _make_plan_precision_aware(
        uvw, freq, (16, 16), 2e-3, 2e-3, _REVIEW_EPS, real_dtype, hermitian=hermitian
    )

    flip = np.asarray(plan.flip_sign).astype(np.float64)
    stored = np.asarray(plan.uvw_m).astype(np.float64)
    expected = np.asarray(uvw, dtype=plan.real_dtype).astype(np.float64) * flip[:, None]
    np.testing.assert_allclose(
        stored,
        expected,
        rtol=0,
        atol=0,
        err_msg=(
            "plan.uvw_m is not the sign-folded input. If only the w column matches, the "
            "implementation negated w without negating u and v: that is not the Hermitian "
            "symmetry V(-u,-v,-w) = conj(V(u,v,w)) and it silently corrupts the (u,v) NUFFT"
        ),
    )
    if hermitian:
        assert np.all(stored[:, 2] >= 0.0), (
            "after the fold every stored w must be >= 0 (the issue's [0, max|w|] range). "
            "Folding on w > 0 instead of w < 0 is mathematically equivalent but contradicts "
            "the documented convention this repo's plan semantics are written against"
        )
        assert int((uvw[:, 2] < 0).sum()) > 0, "fixture folded nothing"


@requires_x64
def test_hermitian_is_static_plan_metadata() -> None:
    """Two plans differing only in ``hermitian`` must not share a JIT cache entry.

    ``hermitian`` changes the operator (different plane grid, and a conjugation
    on a subset of rows), so it belongs in the pytree aux data, i.e. in the
    treedef. The mutation goes through ``dataclasses.replace`` rather than a
    second ``make_plan`` for the reason ``tests/test_planning.py::test_each_
    static_plan_field_is_in_the_aux_data`` gives: two separately-built plans
    differ in leaf *shapes* too, and shapes are not part of a treedef, so
    comparing them would prove nothing about where ``hermitian`` lives.
    """
    rng = np.random.default_rng(4)
    uvw = rng.normal(scale=200.0, size=(64, 3))
    plan = make_plan(uvw, np.array([1.4e9]), (32, 32), 1e-3, 1e-3, 1e-6, hermitian=True)

    mutated = dataclasses.replace(plan, hermitian=False)
    assert mutated.hermitian is not plan.hermitian
    leaves = jax.tree_util.tree_leaves(plan)
    assert all(a is b for a, b in zip(jax.tree_util.tree_leaves(mutated), leaves, strict=True))
    assert jax.tree_util.tree_structure(mutated) != jax.tree_util.tree_structure(plan), (
        "hermitian must be part of the pytree aux_data (_plan_aux), or a folded and an "
        "unfolded plan over the same data share a treedef and therefore a JIT cache entry"
    )


@requires_x64
def test_fold_costs_at_most_one_byte_per_row() -> None:
    """The plan's row-proportional leaf bytes stay at 29 B/row (float64).

    Issue #23 cut a 64-channel 1M-row plan from 4.1 GB to 30 MB by deleting
    every per-(channel, row) array; ``tests/test_planning.py``'s footprint and
    host-cost gates keep it there. This issue is allowed to add per-*row*
    information and nothing more, so the budget is one byte on top of the 28
    (``uvw_m`` 24 + ``sort_perm`` 4) already spent.

    Measured as bytes-per-row rather than as a total, and at ``n_chan = 16``, so
    the number is unaffected by the image and plane terms and so a per-channel
    flip array shows up as 16x the allowance rather than as a few percent of a
    total. An int32 flip leaf lands at 32 B/row and a float64 one at 36 B/row;
    both fail.
    """
    n_rows, n_chan = 4_000, 16
    rng = np.random.default_rng(23)
    uvw = rng.normal(scale=1500.0, size=(n_rows, 3))
    freq = np.linspace(140e6, 160e6, n_chan)

    per_row: dict[bool, float] = {}
    for hermitian in (False, True):
        plan = make_plan(uvw, freq, (64, 64), 1e-3, 1e-3, 1e-6, hermitian=hermitian)
        row_bytes, _ = _row_bytes_and_other_bytes(plan)
        per_row[hermitian] = row_bytes / n_rows
        assert per_row[hermitian] <= _MAX_LEAF_BYTES_PER_ROW_F64, (
            f"hermitian={hermitian}: plan leaves cost {per_row[hermitian]:.2f} B/row, above "
            f"the {_MAX_LEAF_BYTES_PER_ROW_F64} B/row budget (uvw_m 24 + sort_perm 4 + one "
            "byte of fold sign). A float64 flip leaf costs 8 B/row and a per-channel one "
            f"costs {n_chan} B/row -- both are regressions of issue #23"
        )
    assert per_row[True] - per_row[False] <= 1.0, (
        f"turning the fold on added {per_row[True] - per_row[False]:.2f} B/row; at most one "
        "byte per row may depend on the hermitian setting"
    )


@requires_x64
def test_fold_adds_nothing_per_channel() -> None:
    """Leaf bytes must not grow with ``n_chan * n_rows``.

    ``test_fold_costs_at_most_one_byte_per_row`` bounds the per-row cost at a
    fixed channel count; this bounds the channel *slope* at a fixed row count,
    which is the shape a per-(channel, row) flip array actually takes. The two
    are not the same gate: a ``(n_chan, n_rows)`` int8 array at ``n_chan = 1``
    passes the first one exactly.

    The allowance is what may legitimately scale with the channel count after
    issue #23 -- one ``inv_lambda`` scalar and one ``(n_w,)`` row of
    ``window_start`` per channel -- times four for slack. A per-(channel, row)
    byte array costs ``n_rows`` per channel, three orders of magnitude more.
    """
    n_rows = 20_000
    rng = np.random.default_rng(43)
    uvw = rng.normal(scale=1500.0, size=(n_rows, 3))

    totals = {}
    for n_chan in (2, 32):
        freq = np.linspace(140e6, 160e6, n_chan)
        plan = make_plan(uvw, freq, (64, 64), 1e-3, 1e-3, 1e-6, hermitian=True)
        totals[n_chan] = (
            sum(int(np.asarray(x).nbytes) for x in jax.tree_util.tree_leaves(plan)),
            plan.n_w,
        )
    (lo_bytes, lo_n_w), (hi_bytes, hi_n_w) = totals[2], totals[32]
    assert lo_n_w == hi_n_w, "the two probe plans must differ only in n_chan"

    delta_chan = 30
    bound = 4 * delta_chan * (8 + 4 * hi_n_w + 64)
    naive = n_rows * delta_chan  # one byte per (channel, row)
    delta = hi_bytes - lo_bytes
    assert delta <= bound, (
        f"plan leaves grew by {delta} B going from 2 to 32 channels at n_rows={n_rows}, above "
        f"the {bound} B allowed for per-channel bookkeeping. A per-(channel, row) fold array "
        f"would cost about {naive} B here, and the measured growth is {100 * delta / naive:.1f}% "
        "of that. The sign of w is frequency-independent: the fold is per row, never per "
        "(channel, row)"
    )


# ---------------------------------------------------------------------------
# 2. the plane count actually drops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("telescope", "zenith_angle_deg"), _SEVEN_REVIEW_FIXTURES, ids=_REVIEW_IDS)
def test_plane_count_halves_on_review_fixtures(
    telescope: Telescope, zenith_angle_deg: float, real_dtype: DTypeLike
) -> None:
    """The fold must actually buy planes on every review fixture.

    Three assertions per fixture, in increasing distance from what the fold
    controls, because the headline "n_w halves" is only true where the kernel
    width is negligible:

      1. **w-extent** ratio <= 0.55. This is the fold's direct effect and it
         holds on all seven (measured 0.518-0.532).
      2. **inner plane count** ``n_w - W`` roughly halves:
         ``inner_on <= 0.55 * inner_off + 1``. The ``+1`` is the ceiling in
         ``n_w_inner = ceil(...)``, which costs a whole plane when the count is
         small (7 -> 4 on EDA2 zenith is 0.571, not 0.5, and no correct
         implementation can do better). On MWA_compact zenith and MeerKAT
         zenith ``inner_off`` is already 1 -- ``make_plan``'s "always at least
         one interior step" floor -- so this clause is vacuous there and only
         clause 1 gates them; the assertion message says so rather than letting
         a green tick imply coverage it does not have.
      3. **total ``n_w``** must not exceed the pinned per-fixture measurement,
         in *both* hermitian settings. Pinning the ``hermitian=False`` column
         too is what stops a regression that "improves" the ratio by inflating
         the baseline.

    The 0.55x gate on the *total* ``n_w`` -- the issue's own deliverable -- is
    asserted separately, on the one fixture where it is reachable, in
    ``test_n_w_meets_the_issue_gate_on_mwa_extended_off30``.
    """
    name = _fixture_id((telescope, zenith_angle_deg))
    uvw = synthetic_uvw(telescope, zenith_angle_deg, seed=0)
    freq = np.array([telescope.freq_hz])
    shape = (telescope.n_pix, telescope.n_pix)
    kw = (shape, telescope.pixsize, telescope.pixsize, _REVIEW_EPS, real_dtype)
    plan_off = _make_plan_precision_aware(uvw, freq, *kw, hermitian=False)
    plan_on = _make_plan_precision_aware(uvw, freq, *kw, hermitian=True)

    print(
        f"{name}: n_w {plan_off.n_w} -> {plan_on.n_w} "
        f"({plan_on.n_w / plan_off.n_w:.3f}x), w_extent {plan_off.w_extent:.6g} -> "
        f"{plan_on.w_extent:.6g} ({plan_on.w_extent / plan_off.w_extent:.3f}x), "
        f"{int(_folded_rows(plan_on).sum())}/{telescope.n_rows} rows folded"
    )

    # The fixture must actually have both signs of w, or every claim below is
    # about a fold that never happened.
    n_folded = int(_folded_rows(plan_on).sum())
    assert 0 < n_folded < telescope.n_rows, (
        f"{name}: {n_folded}/{telescope.n_rows} rows folded -- the fixture has no mixed-sign "
        "w, so it cannot demonstrate anything about the fold"
    )

    extent_ratio = plan_on.w_extent / plan_off.w_extent
    assert extent_ratio <= W_EXTENT_RATIO_GATE, (
        f"{name}: folded w-extent is {extent_ratio:.3f}x the unfolded one, above the "
        f"{W_EXTENT_RATIO_GATE} gate. The fold is not reaching the w-range the plane grid is "
        f"built from ({plan_off.w_extent:.6g} -> {plan_on.w_extent:.6g})"
    )

    width = plan_off.w_kernel_width
    assert plan_on.w_kernel_width == width
    inner_off = plan_off.n_w - width
    inner_on = plan_on.n_w - width
    if inner_off > 1:
        assert inner_on <= 0.55 * inner_off + 1, (
            f"{name}: inner plane count {inner_off} -> {inner_on}; the fold must roughly halve "
            f"it (bound {0.55 * inner_off + 1:.1f}, the +1 absorbing the ceil in n_w_inner)"
        )
    else:
        assert inner_on == inner_off == 1, (
            f"{name}: inner_off is at make_plan's one-interior-step floor, so the inner-count "
            f"clause gates nothing here; the w-extent clause above is what covers this fixture"
        )

    assert plan_off.n_w <= _N_W_HERMITIAN_OFF[name], (
        f"{name}: hermitian=False n_w is {plan_off.n_w}, above the pinned "
        f"{_N_W_HERMITIAN_OFF[name]} -- the unfolded baseline moved, so the ratio above is "
        "measured against the wrong thing"
    )
    assert plan_on.n_w <= _N_W_HERMITIAN_ON[name], (
        f"{name}: hermitian=True n_w is {plan_on.n_w}, above the pinned {_N_W_HERMITIAN_ON[name]}"
    )


def test_n_w_meets_the_issue_gate_on_mwa_extended_off30(real_dtype: DTypeLike) -> None:
    """``n_w <= 0.55 * n_w(hermitian=False)`` on the issue's headline fixture.

    This is issue #17's stated deliverable and the second of the two 2x factors
    separating ducc0's 88 planes from jax-nufft's 494 on MWA_extended off30
    (nshift, issue #16, was the first). It is asserted here and not on the other
    six fixtures because ``n_w = n_w_inner + W`` carries a flat ``W`` planes the
    fold cannot touch: at eps=1e-6, ``W = 7`` against an ``n_w_inner`` of 1-12
    on the other six, so a 0.55x gate on their totals would be a gate on the
    kernel width. Here ``n_w_inner`` is 244, ``W`` is 3% of the total, and the
    measured ratio is 0.534.
    """
    tel = MWA_EXTENDED
    uvw = synthetic_uvw(tel, 30.0, seed=0)
    freq = np.array([tel.freq_hz])
    kw = ((tel.n_pix, tel.n_pix), tel.pixsize, tel.pixsize, _REVIEW_EPS, real_dtype)
    off = _make_plan_precision_aware(uvw, freq, *kw, hermitian=False).n_w
    on = _make_plan_precision_aware(uvw, freq, *kw, hermitian=True).n_w
    print(f"MWA_extended_off30: n_w {off} -> {on} ({on / off:.3f}x)")
    assert on <= 0.55 * off, (
        f"MWA_extended off30: n_w went {off} -> {on} ({on / off:.3f}x), which does not reach "
        "the 0.55x the issue is written for. Without this the change buys nothing"
    )


@pytest.mark.parametrize(("telescope", "zenith_angle_deg"), _SEVEN_REVIEW_FIXTURES, ids=_REVIEW_IDS)
@pytest.mark.parametrize("hermitian", [False, True])
def test_fold_composes_with_the_nshift_centring(
    telescope: Telescope, zenith_angle_deg: float, hermitian: bool, real_dtype: DTypeLike
) -> None:
    """``nshift`` (issue #16) and the fold (issue #17) must both still be in effect.

    Both devices shrink the plane count and both act on the same expression
    ``n_w_inner = ceil(w_extent * max|n-1+nshift| / x0)`` -- #16 halves the
    second factor, #17 halves the first -- so it is entirely possible to land a
    fold that quietly undoes the centring (for instance by recomputing the
    ``n-1`` extremes after the fold, or by taking ``nshift = 0`` on the folded
    path) and still see ``n_w`` drop. The plane count alone cannot tell the two
    apart.

    :func:`_independent_plan_geometry` recomputes ``nshift``, the w-extent and
    ``n_w`` from the raw inputs in numpy, so it pins the *product* of the two
    effects: dropping either one lands on a different ``n_w``. ``nshift`` is
    additionally required to be bit-identical between the folded and unfolded
    plans -- the fold changes the baselines, never the image geometry.
    """
    uvw = synthetic_uvw(telescope, zenith_angle_deg, seed=0)
    freq = np.array([telescope.freq_hz])
    shape = (telescope.n_pix, telescope.n_pix)
    kw = (shape, telescope.pixsize, telescope.pixsize, _REVIEW_EPS, real_dtype)
    plan = _make_plan_precision_aware(uvw, freq, *kw, hermitian=hermitian)
    reference = _make_plan_precision_aware(uvw, freq, *kw, hermitian=False)

    nshift_exp, extent_exp, n_w_exp = _independent_plan_geometry(
        uvw,
        freq,
        shape,
        telescope.pixsize,
        telescope.pixsize,
        _REVIEW_EPS,
        hermitian=hermitian,
        real_dtype=real_dtype,
    )
    assert plan.nshift == pytest.approx(nshift_exp, rel=1e-12, abs=1e-12), (
        f"nshift is {plan.nshift!r}, expected {nshift_exp!r} from the image geometry alone "
        "(issue #16). The fold must not disturb the n-1 centring"
    )
    assert plan.nshift == reference.nshift, (
        "nshift differs between the folded and unfolded plans; it depends only on the image "
        "grid, so the fold has reached something it should not have"
    )
    assert plan.w_extent == pytest.approx(extent_exp, rel=1e-6), (
        f"w_extent is {plan.w_extent!r}, expected {extent_exp!r} for hermitian={hermitian}"
    )
    assert plan.n_w == n_w_exp, (
        f"n_w is {plan.n_w}, but ceil(w_extent * max|n-1+nshift| / x0) + W predicts {n_w_exp} "
        f"for hermitian={hermitian}. The two plane-count reductions (#16's nshift and #17's "
        "fold) must compose: this fails if either is missing or if one has undone the other"
    )


# ---------------------------------------------------------------------------
# 3. correctness: the two paths agree, and both match the exact DFT
# ---------------------------------------------------------------------------


@requires_x64
@pytest.mark.parametrize(("telescope", "zenith_angle_deg"), _SEVEN_REVIEW_FIXTURES, ids=_REVIEW_IDS)
def test_folded_and_unfolded_paths_agree_on_the_review_fixtures(
    telescope: Telescope, zenith_angle_deg: float
) -> None:
    """Forward and adjoint, folded vs unfolded, on all seven review fixtures.

    The bound is ``4 * eps`` and not 1e-11: see the module docstring. Both paths
    are separately held to ``2 * eps`` against the exact DFT elsewhere
    (``tests/test_against_dft.py``, ``tests/test_adjoint.py``, both parametrised
    over ``hermitian`` by this issue), so ``4 * eps`` here is the triangle
    inequality on that contract rather than a fitted number. Measured worst case
    2.15x eps at eps=1e-10; at the 1e-6 used here, 1.34x.

    The forward is run on a *real* image, which is the fold's precondition; the
    adjoint on complex visibilities, where the fold is always allowed because
    the output is the real part.
    """
    eps = _REVIEW_EPS
    uvw = synthetic_uvw(telescope, zenith_angle_deg, seed=0)
    freq = np.array([telescope.freq_hz])
    shape = (telescope.n_pix, telescope.n_pix)
    rng = np.random.default_rng(7)
    image = jnp.asarray(rng.standard_normal((1, *shape)))
    vis = jnp.asarray(
        rng.standard_normal((telescope.n_rows, 1)) + 1j * rng.standard_normal((telescope.n_rows, 1))
    )
    kw = dict(
        image_shape=shape,
        pixsize_l=telescope.pixsize,
        pixsize_m=telescope.pixsize,
        epsilon=eps,
    )
    plan_off = make_plan(uvw, freq, hermitian=False, **kw)
    plan_on = make_plan(uvw, freq, hermitian=True, **kw)
    assert int(_folded_rows(plan_on).sum()) > 0, "fixture folded no rows"

    fwd_err = _rel_l2(
        np.asarray(dirty2vis(plan_on, image, w_strategy="dense_scan")),
        np.asarray(dirty2vis(plan_off, image, w_strategy="dense_scan")),
    )
    adj_err = _rel_l2(
        np.asarray(vis2dirty(plan_on, vis, w_strategy="dense_scan")),
        np.asarray(vis2dirty(plan_off, vis, w_strategy="dense_scan")),
    )
    bound = CROSS_PATH_TOL_FACTOR * eps
    assert fwd_err < bound, (
        f"forward: folded vs unfolded relative error {fwd_err:.3e} ({fwd_err / eps:.2f}x eps) "
        f"exceeds {bound:.3e}"
    )
    assert adj_err < bound, (
        f"adjoint: folded vs unfolded relative error {adj_err:.3e} ({adj_err / eps:.2f}x eps) "
        f"exceeds {bound:.3e}"
    )


@requires_x64
@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_fold_conjugates_the_output_of_the_folded_rows(eps: float, w_strategy: str) -> None:
    """Forward: the conjugation goes on the **output**, on exactly the folded rows.

    The machinery evaluates ``V(-u, -v, -w)`` for a folded row, so the value it
    returns must be conjugated before it reaches the caller. Three ways to get
    this wrong -- drop the conjugation, apply it to the complementary rows, or
    apply it on the adjoint's side instead -- all produce an O(1) error against
    the exact DFT on a fixture with a mixed-sign w, which is what this measures.

    The two row groups are scored *separately*, not just in aggregate: an
    aggregate norm over a fixture that is 1/3 folded is dominated by the
    unfolded 2/3, so a conjugation applied to the wrong subset shows a smaller
    total error than it should and the failure message would blame the wrong
    thing. Splitting them means the message names which group is wrong.

    All four w-strategies are enumerated rather than sampled: the dense path
    conjugates in input row order, the windowed scan path after its unsort, and
    the windowed vmap path after its scatter -- three different row orders, and
    a fold applied in the wrong one is a permutation bug that only that
    strategy sees.
    """
    uvw, image, _, freq, pixsize = _asymmetric_w_case()
    shape = image.shape[1:]
    plan = make_plan(uvw, freq, shape, pixsize, pixsize, eps, hermitian=True)

    folded = _folded_rows(plan)
    n_folded, n_rows = int(folded.sum()), uvw.shape[0]
    assert 0 < n_folded < n_rows, "fixture must contain both folded and unfolded rows"
    assert n_folded != n_rows - n_folded, (
        "the fixture must not be an even split, or 'conjugate the folded rows' and "
        "'conjugate the unfolded rows' are indistinguishable by symmetry"
    )

    got = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy=w_strategy))
    ref = _reference_forward(image.astype(np.complex128), uvw, freq, pixsize, pixsize)

    bound = DFT_TOL_FACTOR * eps
    for label, mask in (("folded", folded), ("unfolded", ~folded)):
        err = _rel_l2(got[mask], ref[mask])
        assert err < bound, (
            f"{w_strategy}: forward relative error on the {label} rows ({int(mask.sum())} of "
            f"{n_rows}) is {err:.3e}, above {DFT_TOL_FACTOR:g}*eps={bound:.3e}. The forward "
            "must conjugate the OUTPUT of the rows it folded (and only those); an error on "
            "the folded rows alone means the conjugation is missing or on the wrong side, "
            "and an error on the unfolded rows alone means it is applied to the complement"
        )


@requires_x64
@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_fold_conjugates_the_input_of_the_folded_rows(eps: float, w_strategy: str) -> None:
    """Adjoint: the conjugation goes on the **input**, on exactly the folded rows.

    The mirror of the forward test, and the half where the mistake is easiest
    to make invisible: the adjoint's output is real, so "conjugate the output
    instead" is a no-op that silently degrades to "no conjugation at all".
    Feeding ``conj(vis_r)`` through the folded machinery is what makes
    ``Re[vis_r * phase_r] = Re[conj(vis_r) * phase_folded_r]`` exact.

    Scored on the whole image (the adjoint has no row axis in its output), so
    the row-group split of the forward test is replaced by a fixture guard: the
    visibilities have an imaginary part comparable to their real part, so a
    missing conjugation cannot cancel.

    All four strategies, for the same row-order reason as the forward: the
    windowed adjoint conjugates in *sorted* row order, which needs
    ``flip_sign[sort_perm]`` rather than ``flip_sign``, and getting that gather
    wrong is invisible to the dense path.
    """
    uvw, _, vis, freq, pixsize = _asymmetric_w_case()
    shape = (16, 16)
    plan = make_plan(uvw, freq, shape, pixsize, pixsize, eps, hermitian=True)

    folded = _folded_rows(plan)
    assert 0 < int(folded.sum()) < uvw.shape[0]
    assert np.min(np.abs(vis.imag)) > 0.0, "visibilities must be genuinely complex"

    got = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy=w_strategy))
    ref = _reference_adjoint(vis, uvw, freq, shape, pixsize, pixsize)
    err = _rel_l2(got, ref)
    assert err < DFT_TOL_FACTOR * eps, (
        f"{w_strategy}: adjoint relative error {err:.3e} exceeds "
        f"{DFT_TOL_FACTOR:g}*eps={DFT_TOL_FACTOR * eps:.3e}. The adjoint must conjugate the "
        "INPUT visibilities of the folded rows before the plane loop; conjugating its output "
        "instead is a no-op (the output is real), i.e. the same failure as dropping it"
    )


@requires_x64
def test_swapping_the_two_conjugation_sides_breaks_adjointness() -> None:
    """The dot-product identity pins forward-output vs adjoint-input jointly.

    ``<A x, y> == <x, A^H y>`` is what forces the two conjugations to be on
    opposite sides: the adjoint of "conjugate the output on a row subset" is
    "conjugate the input on that same subset", and any other pairing --
    conjugating on the same side twice, or on complementary subsets -- breaks
    the identity even where each operator separately looks plausible.

    This is deliberately a *different instrument* from the two DFT tests above:
    those compare each operator against truth one at a time and would both pass
    if the fold were dropped from both operators at once on a fixture with no
    folded rows, whereas this one relates them to each other. Together they pin
    each side and the pairing. Bound: the eps-independent 1e-11 of AGENTS.md
    sec 6 -- both sides are the same operator here, so reduction order is the
    only legitimate difference.
    """
    eps = 1e-8
    uvw, image, vis, freq, pixsize = _asymmetric_w_case()
    shape = image.shape[1:]
    plan = make_plan(uvw, freq, shape, pixsize, pixsize, eps, hermitian=True)
    assert int(_folded_rows(plan).sum()) > 0

    x = jnp.asarray(image)
    y = jnp.asarray(vis)
    ax = np.asarray(dirty2vis(plan, x, w_strategy="dense_scan"))
    aty = np.asarray(vis2dirty(plan, y, w_strategy="dense_scan"))

    # The identity in this repo's convention (see
    # ``tests/test_adjoint.py::test_dot_product_identity``): the forward carries
    # no 1/n and the adjoint applies it on its output, so for real ``x``
    #
    #     Re(<A x, y>_C) == <n * x, A^* y>_R,
    #
    # the ``n`` on the right undoing the adjoint's output factor. Valid only
    # where ``n > 0``; the fixture's 16x16 image at pixsize 0.01 is entirely
    # inside the unit disc, so no masking is needed.
    n_grid = np.asarray(plan.n_minus_1) + 1.0
    assert np.all(n_grid > 0.0), "fixture image must lie inside the unit disc"
    lhs = float(np.vdot(ax.ravel(), np.asarray(vis).ravel()).real)
    rhs = float(np.vdot((np.asarray(image) * n_grid[None]).ravel(), aty.ravel()))
    residual = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert residual < STRATEGY_TOL, (
        f"dot-product residual {residual:.3e} exceeds {STRATEGY_TOL:g}. The forward must "
        "conjugate the OUTPUT of the folded rows and the adjoint the INPUT of the same rows; "
        "putting both on the same side, or on complementary subsets, breaks exactly this"
    )


@requires_x64
@pytest.mark.parametrize("channel_strategy", ["scan", "vmap"])
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_multi_channel_dft_parity_under_the_fold(w_strategy: str, channel_strategy: str) -> None:
    """Three frequencies, forward and adjoint, folded, against the exact DFT.

    The fold is per row and channel-independent by construction (``freq/c > 0``
    is monotonic, so every channel folds the same rows), and the folded
    baselines have to serve every channel. A single-channel test cannot see a
    fold applied to channel 0's coordinates only, nor a ``flip_sign`` gathered
    against the wrong axis; with three channels spanning a factor of four in
    frequency and a different image per channel, either is a gross error.

    All eight strategy pairs, for the reason
    ``tests/test_against_dft.py::test_multi_channel_matches_dft_forward_and_adjoint``
    gives: the channel loop and the w-plane loop compose differently in each,
    and the windowed helpers reach the coordinates through a ``sort_perm``
    gather the dense ones do not.
    """
    eps = 1e-6
    rng = np.random.default_rng(1701)
    n_l = n_m = 16
    n_rows = 24
    pixsize = 0.006
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-60.0, 60.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-60.0, 60.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-35.0, 25.0, size=n_rows)
    freq = np.array([0.7e9, 1.4e9, 2.8e9])
    n_chan = freq.shape[0]
    image = rng.standard_normal((n_chan, n_l, n_m))  # real: the fold's precondition
    vis = (
        rng.standard_normal((n_rows, n_chan)) + 1j * rng.standard_normal((n_rows, n_chan))
    ).astype(np.complex128)

    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, eps, hermitian=True)
    assert plan.n_chan == n_chan
    assert 0 < int(_folded_rows(plan).sum()) < n_rows

    vis_jax = np.asarray(
        dirty2vis(
            plan, jnp.asarray(image), w_strategy=w_strategy, channel_strategy=channel_strategy
        )
    )
    dirty_jax = np.asarray(
        vis2dirty(plan, jnp.asarray(vis), w_strategy=w_strategy, channel_strategy=channel_strategy)
    )
    vis_ref = _reference_forward(image.astype(np.complex128), uvw, freq, pixsize, pixsize)
    dirty_ref = _reference_adjoint(vis, uvw, freq, (n_l, n_m), pixsize, pixsize)

    bound = DFT_TOL_FACTOR * eps
    # Per channel as well as in aggregate: a norm over three channels is
    # dominated by the loudest, so a single mis-folded channel could hide.
    for c in range(n_chan):
        f_err = _rel_l2(vis_jax[:, c], vis_ref[:, c])
        a_err = _rel_l2(dirty_jax[c], dirty_ref[c])
        assert f_err < bound, (
            f"forward channel {c} (freq={freq[c]:.3g} Hz, {w_strategy}, {channel_strategy}): "
            f"relative error {f_err:.3e} exceeds {bound:.3e} under the Hermitian fold"
        )
        assert a_err < bound, (
            f"adjoint channel {c} (freq={freq[c]:.3g} Hz, {w_strategy}, {channel_strategy}): "
            f"relative error {a_err:.3e} exceeds {bound:.3e} under the Hermitian fold"
        )


# ---------------------------------------------------------------------------
# 4. complex images: the one case the fold may not be applied to
# ---------------------------------------------------------------------------


@requires_x64
def test_dirty2vis_rejects_a_complex_image_on_a_folded_plan() -> None:
    """A complex image on a ``hermitian=True`` plan must raise, naming the fix.

    The Hermitian identity holds for a *real* sky only, and the folded plan
    cannot be rescued at call time: its w-planes span ``[0, max|w|]``, so a row
    restored to a negative w would fall outside every plane's kernel support
    and be gridded as (approximately) nothing -- a silently wrong answer, not a
    slow one. Computing the complex case correctly on a folded plan means
    running the whole operator twice, once per component, which is a 2x cost the
    caller did not ask for and cannot see.

    So the contract is: refuse, and name ``hermitian=False`` in the message so
    the caller knows the one-line fix. The check is on the dtype of the array
    the caller passed, *before* ``_prepare_image`` promotes it, so a
    complex-dtype image with a zero imaginary part is refused too -- the rule is
    a property of the input's type, not of its values, and a value-dependent
    check could not be made under ``jax.jit`` at all.
    """
    rng = np.random.default_rng(99)
    n_pix, n_rows = 16, 24
    uvw = rng.normal(scale=80.0, size=(n_rows, 3))
    freq = np.array([1.4e9])
    plan = make_plan(uvw, freq, (n_pix, n_pix), 0.005, 0.005, 1e-6, hermitian=True)

    complex_image = jnp.asarray(
        rng.standard_normal((1, n_pix, n_pix)) + 1j * rng.standard_normal((1, n_pix, n_pix))
    )
    with pytest.raises(ValueError, match="hermitian=False") as excinfo:
        dirty2vis(plan, complex_image)
    assert "hermitian" in str(excinfo.value)

    # Same refusal for a complex dtype whose imaginary part happens to be zero:
    # the check is on the type, not on the values.
    zero_imag = jnp.asarray(rng.standard_normal((1, n_pix, n_pix))).astype(jnp.complex128)
    assert jnp.iscomplexobj(zero_imag)
    with pytest.raises(ValueError, match="hermitian=False"):
        dirty2vis(plan, zero_imag)

    # And a real image on the same plan is accepted.
    real_image = jnp.asarray(rng.standard_normal((1, n_pix, n_pix)))
    assert dirty2vis(plan, real_image).shape == (n_rows, 1)


@requires_x64
def test_dirty2vis_accepts_a_complex_image_on_an_unfolded_plan() -> None:
    """``hermitian=False`` keeps the complex-image path working, to DFT parity.

    The refusal above must be scoped to folded plans: ``hermitian=False`` is the
    escape hatch the error message points at, so it has to actually work, and
    work to the same ``2 * eps`` contract as everything else. A refusal that
    fired on every plan would pass a test that only checked ``pytest.raises``.
    """
    eps = 1e-6
    rng = np.random.default_rng(100)
    n_pix, n_rows = 16, 24
    pixsize = 0.005
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-60.0, 60.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-60.0, 60.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-15.0, 15.0, size=n_rows)
    freq = np.array([1.4e9])
    image = rng.standard_normal((1, n_pix, n_pix)) + 1j * rng.standard_normal((1, n_pix, n_pix))

    plan = make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, eps, hermitian=False)
    got = np.asarray(dirty2vis(plan, jnp.asarray(image)))
    ref = _reference_forward(image, uvw, freq, pixsize, pixsize)
    err = _rel_l2(got, ref)
    assert err < DFT_TOL_FACTOR * eps, (
        f"hermitian=False must keep the complex-image forward at DFT parity; got {err:.3e}"
    )


@requires_x64
def test_default_plan_is_never_silently_wrong_on_a_complex_image() -> None:
    """Whatever ``make_plan``'s ``hermitian`` default is, this must not be silent.

    The default is the implementer's call and nothing else in this module
    depends on it. What is *not* negotiable is the failure mode: a plan built
    with no ``hermitian`` argument, handed a complex image, must either return
    an answer that meets the ``2 * eps`` DFT contract or refuse with a message
    naming ``hermitian=False``. Returning visibilities that are wrong by O(1)
    -- what a folded plan does if the check is missing -- is the one outcome
    this forbids, and it is the outcome that a default flipped to ``True``
    without the guard produces on every existing complex-image caller in this
    repository.
    """
    eps = 1e-6
    rng = np.random.default_rng(101)
    n_pix, n_rows = 16, 24
    pixsize = 0.005
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-60.0, 60.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-60.0, 60.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-25.0, 25.0, size=n_rows)
    freq = np.array([1.4e9])
    image = rng.standard_normal((1, n_pix, n_pix)) + 1j * rng.standard_normal((1, n_pix, n_pix))

    plan = make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, eps)
    assert int((uvw[:, 2] < 0).sum()) > 0, "fixture must have rows the fold would move"
    try:
        got = np.asarray(dirty2vis(plan, jnp.asarray(image)))
    except ValueError as exc:
        assert "hermitian=False" in str(exc), (
            "a complex image on the default plan was refused, which is fine, but the message "
            f"must name the fix (hermitian=False); got: {exc}"
        )
        return
    ref = _reference_forward(image, uvw, freq, pixsize, pixsize)
    err = _rel_l2(got, ref)
    assert err < DFT_TOL_FACTOR * eps, (
        f"the default plan accepted a complex image and returned a result {err:.3e} off the "
        f"exact DFT ({err / eps:.1f}x eps). If the default is hermitian=True, dirty2vis must "
        "REFUSE a complex image with a message naming hermitian=False rather than folding it"
    )


@requires_x64
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_vis2dirty_folds_complex_visibilities_unconditionally(w_strategy: str) -> None:
    """The adjoint may always fold: its output is the real part.

    ``vis2dirty`` has no real-input precondition to check -- the visibilities
    are complex by contract -- because ``Re[vis * phase] = Re[conj(vis) *
    phase_folded]`` holds identically. So it must *not* acquire the forward's
    refusal, it must use the folded plan (the plane count is halved for it too),
    and its answer must be unchanged.

    Three separate things are asserted, because "it does fold" and "the answer
    is right" are different claims and only the pair is worth anything: the plan
    it runs on has actually folded rows and a strictly smaller ``n_w``; no
    exception is raised; and the output matches both the exact DFT (``2 * eps``)
    and the unfolded path (``4 * eps``).
    """
    eps = 1e-6
    uvw, _, vis, freq, pixsize = _asymmetric_w_case()
    shape = (16, 16)
    kw = (shape, pixsize, pixsize, eps)
    plan_on = make_plan(uvw, freq, *kw, hermitian=True)
    plan_off = make_plan(uvw, freq, *kw, hermitian=False)

    assert int(_folded_rows(plan_on).sum()) > 0, "fixture folded no rows"
    assert plan_on.n_w < plan_off.n_w, (
        f"the adjoint is not getting a folded plan: n_w is {plan_on.n_w} with the fold on and "
        f"{plan_off.n_w} with it off, so there is no plane-count saving to verify"
    )

    got = np.asarray(vis2dirty(plan_on, jnp.asarray(vis), w_strategy=w_strategy))
    ref = _reference_adjoint(vis, uvw, freq, shape, pixsize, pixsize)
    unfolded = np.asarray(vis2dirty(plan_off, jnp.asarray(vis), w_strategy=w_strategy))

    err_dft = _rel_l2(got, ref)
    err_path = _rel_l2(got, unfolded)
    assert err_dft < DFT_TOL_FACTOR * eps, (
        f"{w_strategy}: folded adjoint is {err_dft:.3e} off the exact DFT "
        f"({err_dft / eps:.2f}x eps)"
    )
    assert err_path < CROSS_PATH_TOL_FACTOR * eps, (
        f"{w_strategy}: folded vs unfolded adjoint differ by {err_path:.3e} "
        f"({err_path / eps:.2f}x eps)"
    )


# ---------------------------------------------------------------------------
# 5. weights, strategies, traceability
# ---------------------------------------------------------------------------


@requires_x64
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_weights_are_unaffected_by_the_fold(w_strategy: str) -> None:
    """``vis2dirty(weights=...)`` is untouched by the conjugation.

    Weights are real and multiply the visibilities before gridding (ducc's
    ``wgt``), and ``conj(vis) * wgt == conj(vis * wgt)`` for real ``wgt`` -- so
    the fold and the weighting commute and the weights need no sign handling at
    all. The bug this forbids is an implementation that "helpfully" applies
    ``flip_sign`` to the weights as well, which negates them on the folded rows
    and is a sign error the unweighted tests cannot see.

    The weights are deliberately non-uniform, of mixed magnitude, and include an
    exact zero: a constant weight vector would be reproduced by a plain scale
    factor, and a strictly positive one would let a sign error on a small subset
    hide in the norm.

    Checked against the exact DFT *with the same weights* (2 * eps), not only
    against the unfolded path, so both paths mishandling the weights identically
    is caught too.
    """
    eps = 1e-6
    uvw, _, vis, freq, pixsize = _asymmetric_w_case()
    shape = (16, 16)
    n_rows = uvw.shape[0]
    rng = np.random.default_rng(555)
    weights = rng.uniform(0.1, 4.0, size=(n_rows, 1))
    weights[3, 0] = 0.0
    weights[7, 0] = 25.0

    plan_on = make_plan(uvw, freq, shape, pixsize, pixsize, eps, hermitian=True)
    plan_off = make_plan(uvw, freq, shape, pixsize, pixsize, eps, hermitian=False)
    assert int(_folded_rows(plan_on).sum()) > 0

    got = np.asarray(
        vis2dirty(plan_on, jnp.asarray(vis), weights=jnp.asarray(weights), w_strategy=w_strategy)
    )
    ref = _reference_adjoint(vis, uvw, freq, shape, pixsize, pixsize, weights=weights)
    unfolded = np.asarray(
        vis2dirty(plan_off, jnp.asarray(vis), weights=jnp.asarray(weights), w_strategy=w_strategy)
    )

    err_dft = _rel_l2(got, ref)
    assert err_dft < DFT_TOL_FACTOR * eps, (
        f"{w_strategy}: weighted folded adjoint is {err_dft:.3e} off the weighted exact DFT "
        f"({err_dft / eps:.2f}x eps). Weights are real and commute with the conjugation; the "
        "fold must not touch them"
    )
    err_path = _rel_l2(got, unfolded)
    assert err_path < CROSS_PATH_TOL_FACTOR * eps, (
        f"{w_strategy}: weighted folded vs unfolded adjoint differ by {err_path:.3e}"
    )

    # A zero weight must still zero its row's contribution under the fold: the
    # comparisons above are norms, and one row of 45 at a tolerance of 2e-6 is
    # not what pins this.
    weights_dropped = weights.copy()
    weights_dropped[0, 0] = 0.0
    dropped = np.asarray(
        vis2dirty(
            plan_on,
            jnp.asarray(vis),
            weights=jnp.asarray(weights_dropped),
            w_strategy=w_strategy,
        )
    )
    ref_dropped = _reference_adjoint(
        vis, uvw, freq, shape, pixsize, pixsize, weights=weights_dropped
    )
    assert _rel_l2(dropped, ref_dropped) < DFT_TOL_FACTOR * eps
    assert not np.allclose(dropped, got), (
        "zeroing a weight changed nothing, so the weights are not reaching the folded path"
    )


@requires_x64
@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("case", sorted(_FOLD_CASES))
def test_all_strategies_agree_under_the_fold(eps: float, case: str) -> None:
    """All eight strategy combinations agree to 1e-11 with the fold on.

    Unlike ``hermitian`` on vs off, this *is* a reduction-order comparison --
    one plan, one fold, four ways of walking the same w-planes crossed with two
    ways of walking the channels -- so the eps-independent 1e-11 bound of
    AGENTS.md sec 6 applies unchanged. The fold introduces a per-row
    conjugation that each strategy applies in a different row order (input order
    in the dense path, sorted order in the windowed adjoint, post-scatter in the
    windowed vmap forward), which is exactly the kind of thing that can be right
    in one strategy and wrong in another.

    All eight combinations are enumerated and compared *pairwise*, not each
    against a chosen reference: a bug shared by two strategies and absent from a
    third is visible either way, but a bug in the reference itself would make a
    star comparison green.

    Two fixtures, and the second is not redundant. On ``"narrow"`` every window
    holds every row (``max_window_size == n_rows``), so the windowed strategies
    exercise the fold's *row order* -- the ``sort_perm`` gather, the unsort, the
    scatter -- but not ``window_start``; on ``"windowing"`` the widest window
    holds two thirds of the rows, so a boundary built from the unfolded w drops
    rows and shows up here. The precondition is asserted below rather than
    assumed, because it is exactly the kind of fixture property that silently
    stops holding when a plane count moves.
    """
    uvw, image, vis, freq, pixsize = _FOLD_CASES[case]()
    shape = image.shape[1:]
    plan = make_plan(uvw, freq, shape, pixsize, pixsize, eps, hermitian=True)
    assert int(_folded_rows(plan).sum()) > 0
    if case == "windowing":
        assert plan.max_window_size < plan.n_rows, (
            f"the 'windowing' fixture must have strict-subset windows, but "
            f"max_window_size={plan.max_window_size} == n_rows={plan.n_rows} at eps={eps:g}: "
            "the windowed strategies then degenerate into the dense ones and this "
            "parametrisation gates nothing beyond the 'narrow' case"
        )

    combos = list(
        itertools.product(
            ("dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"), ("scan", "vmap")
        )
    )
    forwards = {}
    adjoints = {}
    for w_strategy, channel_strategy in combos:
        key = f"{w_strategy}/{channel_strategy}"
        forwards[key] = np.asarray(
            dirty2vis(
                plan,
                jnp.asarray(image),
                w_strategy=w_strategy,
                channel_strategy=channel_strategy,
            )
        )
        adjoints[key] = np.asarray(
            vis2dirty(
                plan,
                jnp.asarray(vis),
                w_strategy=w_strategy,
                channel_strategy=channel_strategy,
            )
        )

    for label, results in (("forward", forwards), ("adjoint", adjoints)):
        keys = list(results)
        for a, b in itertools.combinations(keys, 2):
            err = _rel_l2(results[a], results[b])
            assert err < STRATEGY_TOL, (
                f"{label}: {a} and {b} differ by {err:.3e} under the Hermitian fold, above the "
                f"{STRATEGY_TOL:g} strategy-equivalence bound. The strategies apply the fold's "
                "conjugation in different row orders; one of them has it wrong"
            )


@requires_x64
def test_folded_operators_stay_traceable() -> None:
    """``jit`` / ``vmap`` / ``grad`` still work through the fold.

    The conjugation is applied per row from a traced ``flip_sign`` leaf, so it
    has to be a ``jnp.where`` (or equivalent), never a Python ``if`` on a traced
    value and never a boolean-mask index. Both of those raise under ``jit``, and
    neither is caught by any accuracy test -- the eager path they work in is the
    one every other test in this file uses.

    ``grad`` is included because ``jnp.conj`` is not holomorphic: it is fine
    here (the image is real and the adjoint's output is real, so the composite
    is a real-to-real map), but a formulation that made it otherwise would fail
    here and nowhere else.
    """
    eps = 1e-6
    uvw, image, vis, freq, pixsize = _asymmetric_w_case()
    shape = image.shape[1:]
    plan = make_plan(uvw, freq, shape, pixsize, pixsize, eps, hermitian=True)
    assert int(_folded_rows(plan).sum()) > 0

    x = jnp.asarray(image)
    y = jnp.asarray(vis)

    jitted_fwd = jax.jit(lambda p, im: dirty2vis(p, im, w_strategy="dense_scan"))
    jitted_adj = jax.jit(lambda p, v: vis2dirty(p, v, w_strategy="dense_scan"))
    np.testing.assert_allclose(
        np.asarray(jitted_fwd(plan, x)),
        np.asarray(dirty2vis(plan, x, w_strategy="dense_scan")),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(jitted_adj(plan, y)),
        np.asarray(vis2dirty(plan, y, w_strategy="dense_scan")),
        rtol=1e-12,
        atol=0,
    )

    # vmap over a batch of images / visibility sets.
    batch_x = jnp.stack([x, 2.0 * x])
    batched = jax.vmap(lambda im: dirty2vis(plan, im, w_strategy="dense_scan"))(batch_x)
    np.testing.assert_allclose(
        np.asarray(batched[1]), 2.0 * np.asarray(batched[0]), rtol=1e-10, atol=0
    )

    # grad through both operators.
    def loss(im: jax.Array) -> jax.Array:
        return jnp.sum(jnp.abs(dirty2vis(plan, im, w_strategy="dense_scan")) ** 2)

    g = jax.grad(loss)(x)
    assert g.shape == x.shape
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.linalg.norm(np.asarray(g)) > 0.0

    def adj_loss(v: jax.Array) -> jax.Array:
        return jnp.sum(vis2dirty(plan, v, w_strategy="dense_scan") ** 2)

    g2 = jax.grad(adj_loss, holomorphic=False)(y.real)
    assert np.all(np.isfinite(np.asarray(g2)))
