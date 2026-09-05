"""``divide_by_n`` on both operators (issue #20): the exact adjoint pair.

Through v0.1.2 the ``1/n`` factor was hard-wired: ``dirty2vis`` never applied
it (ducc's ``divide_by_n=False``) and ``vis2dirty`` always did
(``divide_by_n=True``). That mixed pair is *not* an adjoint pair. The identity
that survives it is ``Re<A x, y> = <n x, A^H y>``, and even that holds only
inside the unit disc: outside it the forward evaluates the analytic extension
``n - 1 = -sqrt(l^2 + m^2 - 1) - 1`` (so ``n < 0``) while the adjoint zeroes
those pixels. On the EDA2 full-sky fixture (64^2 at a 120-degree FoV, 1155 of
4096 pixels outside the disc) that leaves the mixed pair off by a relative
2.7e-1 in the ``n x`` form and 6.3e-1 in the plain form -- measured here, and
reproduced with ducc0 called the same way. Anyone using ``vis2dirty`` as the
gradient of ``dirty2vis`` on a wide field gets that error in the gradient.

Issue #20 exposes the flag on both operators, keyword-only and static, with
the *defaults unchanged* (``False`` forward, ``True`` adjoint). With **equal**
flags the pair is exactly adjoint:

    Re<A x, y> = <x, A^H y>                       (both flags, any field)

measured at 1.3e-15 .. 1.9e-12 across the fixtures below, hence the
eps-independent ``1e-11`` bound this module gates it at -- the same bound
``tests/test_adjoint.py`` uses for the reduction-order comparison it makes.
Issue #21's ``custom_vjp`` needs exactly that pair, so these tests are its
contract too.

What each flag means, asserted separately from the identity in section 4:

  * forward with ``divide_by_n=True`` multiplies the image by ``1/n`` inside
    the disc and by **zero** outside -- the output is then insensitive to
    every pixel outside it;
  * adjoint with ``divide_by_n=False`` returns the analytic-extension result
    *without* the ``1/n``, so those pixels carry their (large) values rather
    than zeros.

Both are checked against ducc0 (public API only, black-box oracle) at the
repo's ``3 * eps`` bound in section 5, and the outside-disc values against an
exact DFT in the repo's sign convention.

Precision: the identity, the default check, both flag-semantics checks and
the traceability check are parametrised over both legs via ``_PRECISIONS``;
the float64 entry carries ``requires_x64`` so ``JAX_ENABLE_X64=0`` skips it
rather than failing, and the float32 entry runs in both legs. Float32 plans
are built with ``dtype=jnp.float32`` and ``epsilon = 1e-5``, and the identity
is gated at ``1e-6`` there (measured 1.8e-8 .. 7.2e-8). The remaining tests
are float64-only and say so with ``requires_x64``: the signature checks carry
no precision at all, while the ducc0 and exact-DFT comparisons and the
strategy / fold matrices are stated against bounds (``3 * eps`` at eps=1e-6,
``2 * eps``, and the 1e-11 strategy-equivalence bound) that single precision
cannot reach -- restating them at a float32 tolerance would gate nothing.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
from tests.conftest import EDA2, MWA_COMPACT, Telescope, requires_x64, synthetic_uvw

# The dot-product bound, matching tests/test_adjoint.py (issue #10). Measured
# residual under equal flags on these fixtures: 1.3e-15 .. 1.9e-12.
DOT_TOL_F64 = 1e-11
# Single precision: measured 1.8e-8 .. 7.2e-8 on EDA2 full-sky.
DOT_TOL_F32 = 1e-6
# ducc0 parity contract, mirroring tests/test_against_ducc.py::DUCC_TOL_FACTOR.
DUCC_TOL_FACTOR = 3.0
# Exact-DFT contract, mirroring tests/test_adjoint.py::DFT_TOL_FACTOR.
DFT_TOL_FACTOR = 2.0
# Folded vs unfolded agreement, mirroring
# tests/test_hermitian.py::CROSS_PATH_TOL_FACTOR (issue #17): the triangle
# inequality on the 2*eps DFT contract each path meets separately.
CROSS_PATH_TOL_FACTOR = 4.0

W_STRATEGIES = ("dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap")
FLAG_VALUES = (False, True)

# Both precision legs. The float64 entry is skipped (not failed) under
# JAX_ENABLE_X64=0; the float32 entry runs in *both* legs, since a float32
# plan is legal either way.
_PRECISIONS = [
    pytest.param(jnp.float64, 1e-6, DOT_TOL_F64, id="float64", marks=requires_x64),
    pytest.param(jnp.float32, 1e-5, DOT_TOL_F32, id="float32"),
]


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Problem:
    plan: Any
    uvw: np.ndarray
    freq: np.ndarray
    pixsize: float
    shape: tuple[int, int]
    image: np.ndarray  # (n_l, n_m) real, in the plan's real dtype
    vis: np.ndarray  # (n_rows, 1) complex, in the plan's complex dtype
    eps: float


_CACHE: dict[tuple, _Problem] = {}


def _problem(
    tel: Telescope,
    zenith_angle_deg: float,
    *,
    eps: float,
    dtype: DTypeLike = jnp.float64,
    hermitian: bool = True,
    uvw_seed: int = 0,
    data_seed: int = 7,
) -> _Problem:
    """Build (and cache) a plan plus a real image and complex visibilities.

    Cached because several tests below want the same plan and building one is
    pure host work; the arrays handed out are read-only by convention (every
    caller that modifies one takes a copy first).
    """
    real_dtype = np.dtype(jnp.dtype(dtype))
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    key = (tel.name, zenith_angle_deg, eps, real_dtype.name, hermitian, uvw_seed, data_seed)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    uvw = synthetic_uvw(tel, zenith_angle_deg, seed=uvw_seed)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    shape = (tel.n_pix, tel.n_pix)
    rng = np.random.default_rng(data_seed)
    image = rng.standard_normal(shape).astype(real_dtype)
    vis = (rng.standard_normal((tel.n_rows, 1)) + 1j * rng.standard_normal((tel.n_rows, 1))).astype(
        complex_dtype
    )
    plan = make_plan(uvw, freq, shape, pix, pix, eps, dtype=dtype, hermitian=hermitian)
    problem = _Problem(
        plan=plan,
        uvw=uvw,
        freq=freq,
        pixsize=pix,
        shape=shape,
        image=image,
        vis=vis,
        eps=eps,
    )
    _CACHE[key] = problem
    return problem


def _n_grid(plan: Any) -> np.ndarray:
    """``n = (n - 1) + 1`` on the plan's own grid, in the plan's own dtype.

    Formed the way the operators form it -- ``plan.n_minus_1`` plus one, in
    ``plan.real_dtype`` -- so the ``n_grid > 0`` disc mask this module uses is
    the mask the operators use, bit for bit, on a float32 plan as well.
    """
    nm1 = np.asarray(plan.n_minus_1)
    return np.asarray(nm1 + nm1.dtype.type(1.0), dtype=np.float64)


def _outside_disc(plan: Any) -> np.ndarray:
    return _n_grid(plan) <= 0.0


def _forward(problem: _Problem, image: np.ndarray, **kwargs: Any) -> np.ndarray:
    return np.asarray(dirty2vis(problem.plan, jnp.asarray(image), **kwargs))


def _adjoint(problem: _Problem, **kwargs: Any) -> np.ndarray:
    return np.asarray(vis2dirty(problem.plan, jnp.asarray(problem.vis), **kwargs))


def _dot_product_residual(
    problem: _Problem,
    *,
    image: np.ndarray | None = None,
    image_rhs: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[float, float, float]:
    """Relative residual of ``Re<A x, y> == <x_rhs, A^H y>``.

    ``image_rhs`` defaults to ``image``: passing a different array is how the
    mixed-default tests state the ``n x`` correction on the right-hand side
    only. Everything is accumulated in float64 so the residual measures the
    operators, not the test's own summation.
    """
    x = problem.image if image is None else image
    rhs_x = x if image_rhs is None else image_rhs
    vis = problem.vis
    ax = _forward(problem, x, **kwargs).astype(np.complex128)
    ay = _adjoint(problem, **kwargs).astype(np.float64)
    lhs = complex(np.vdot(ax.ravel(), vis.astype(np.complex128).ravel())).real
    rhs = float(np.vdot(np.asarray(rhs_x, dtype=np.float64).ravel(), ay.ravel()))
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs)), lhs, rhs


def _reference_adjoint_no_divide(
    vis: np.ndarray,
    uvw: np.ndarray,
    freq: np.ndarray,
    shape: tuple[int, int],
    pixsize: float,
) -> np.ndarray:
    """Exact DFT adjoint **without** the ``1/n`` and without the disc mask.

    The repo's sign convention (AGENTS.md section 1) with ``n - 1`` taken on
    the analytic extension outside the unit disc, i.e. what ``vis2dirty(...,
    divide_by_n=False)`` is supposed to return everywhere. Same structure as
    ``tests/test_adjoint.py::_reference_adjoint``, minus its final division
    and mask; written independently of ``src/`` so it is an oracle rather than
    a restatement.
    """
    n_l, n_m = shape
    ll = (np.arange(n_l) - n_l // 2) * pixsize
    mm = (np.arange(n_m) - n_m // 2) * pixsize
    lgrid, mgrid = np.meshgrid(ll, mm, indexing="ij")
    rho2 = lgrid**2 + mgrid**2
    inside = rho2 <= 1.0
    nm1 = np.where(
        inside,
        np.sqrt(np.where(inside, 1.0 - rho2, 0.0)) - 1.0,
        -np.sqrt(np.where(inside, 0.0, rho2 - 1.0)) - 1.0,
    )
    out = np.zeros(shape, dtype=np.float64)
    scale = freq[0] / SPEED_OF_LIGHT
    u = uvw[:, 0] * scale
    v = uvw[:, 1] * scale
    w = uvw[:, 2] * scale
    vis64 = vis.astype(np.complex128)
    for r in range(uvw.shape[0]):
        phase = 2j * np.pi * (u[r] * lgrid + v[r] * mgrid - w[r] * nm1)
        out += (vis64[r, 0] * np.exp(phase)).real
    return out


def _eda2_full_sky(dtype: DTypeLike = jnp.float64, eps: float = 1e-6, **kw: Any) -> _Problem:
    """EDA2 at a 120-degree FoV: the fixture whose image runs past the disc.

    Every test that says "outside the disc" uses this one, and asserts the
    region is non-empty and carries signal before asserting anything about it
    -- a narrow-field fixture would make those tests vacuously true.
    """
    return _problem(EDA2, 0.0, eps=eps, dtype=dtype, **kw)


def _assert_outside_disc_is_loaded(problem: _Problem) -> np.ndarray:
    """Guard: the fixture must actually have signal outside the unit disc."""
    outside = _outside_disc(problem.plan)
    n_out = int(outside.sum())
    assert 0 < n_out < outside.size, (
        f"fixture has {n_out} of {outside.size} pixels outside the unit disc: this test "
        "says nothing unless the outside-disc region is non-empty and not the whole image"
    )
    signal = float(np.linalg.norm(np.asarray(problem.image, dtype=np.float64)[outside]))
    assert signal > 0.0, "the outside-disc pixels must carry signal, not zeros"
    return outside


# ---------------------------------------------------------------------------
# 1. the declared defaults (issue #20 keeps today's behaviour by default)
# ---------------------------------------------------------------------------


def _declared(fn: Callable[..., Any]) -> inspect.Parameter:
    params = inspect.signature(fn).parameters
    assert "divide_by_n" in params, (
        f"{fn.__name__} has no divide_by_n parameter; issue #20 exposes it on **both** "
        "operators, and the pair is only adjoint when both accept it"
    )
    return params["divide_by_n"]


def test_dirty2vis_declares_divide_by_n_false_and_keyword_only() -> None:
    """The forward's declared default is the bool ``False``, keyword-only.

    ``is False`` rather than ``== False`` or ``not ...``: a truthy/falsy
    non-bool (``0``, ``None``) would normalise to the same behaviour through
    any ``bool()`` the implementation applies, so behaviour alone cannot see
    it -- but it is wrong in the public signature and, since the flag is a
    static JIT argument, it changes the cache key's type. Keyword-only because
    positional acceptance would silently reinterpret an existing
    ``dirty2vis(plan, image, w_strategy)``-style call.
    """
    param = _declared(dirty2vis)
    assert param.default is False, (
        f"dirty2vis's declared divide_by_n default is {param.default!r}, not the bool False. "
        "Issue #20 must not change what a caller who says nothing gets: the forward has "
        "never applied the 1/n factor (ducc's divide_by_n=False) and the existing DFT and "
        "ducc parity tests are written against that."
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"dirty2vis's divide_by_n is {param.kind}, not keyword-only"
    )


def test_vis2dirty_declares_divide_by_n_true_and_keyword_only() -> None:
    """The adjoint's declared default is the bool ``True``, keyword-only."""
    param = _declared(vis2dirty)
    assert param.default is True, (
        f"vis2dirty's declared divide_by_n default is {param.default!r}, not the bool True. "
        "The adjoint has always applied 1/n on its output (ducc's divide_by_n=True); "
        "issue #20 exposes the flag without moving the default."
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"vis2dirty's divide_by_n is {param.kind}, not keyword-only"
    )


def test_the_declared_defaults_are_not_swapped() -> None:
    """Both defaults in one assertion, because the pair is the claim.

    The two operators' defaults are deliberately *different*, which is exactly
    the shape of edit that gets made backwards. Pinning them one file-section
    apart lets a swap pass one test while failing another; this states the
    pair.
    """
    fwd = _declared(dirty2vis).default
    adj = _declared(vis2dirty).default
    assert (fwd, adj) == (False, True), (
        f"divide_by_n defaults are (dirty2vis={fwd!r}, vis2dirty={adj!r}); issue #20 ships "
        "(False, True) -- the mixed pair that reproduces pre-#20 behaviour. Swapping them "
        "would silently change every existing caller's answer by a factor of n."
    )
    assert fwd is False and adj is True


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_omitting_the_flag_reproduces_the_declared_default(
    op: str, dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """Behavioural half of the default check, on a fixture where it can fail.

    The signature test above cannot see an implementation that declares the
    right default and then ignores it, or one that forwards a constant to the
    JIT boundary. So: the no-argument call must be bit-identical to the
    explicit default and materially different from the other value. EDA2
    full-sky is the fixture that makes "materially different" true -- on a
    narrow field with ``n ~ 1`` the two values are close, and this test would
    pass while gating nothing.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    _assert_outside_disc_is_loaded(problem)
    if op == "dirty2vis":
        default = _forward(problem, problem.image)
        same = _forward(problem, problem.image, divide_by_n=False)
        other = _forward(problem, problem.image, divide_by_n=True)
    else:
        default = _adjoint(problem)
        same = _adjoint(problem, divide_by_n=True)
        other = _adjoint(problem, divide_by_n=False)
    np.testing.assert_array_equal(
        default,
        same,
        err_msg=f"{op} with no divide_by_n differs from the explicit declared default",
    )
    contrast = np.linalg.norm(default - other) / np.linalg.norm(other)
    assert contrast > 1e-2, (
        f"{op}'s two divide_by_n values differ by only {contrast:.3e} on EDA2 full-sky: "
        "either the flag is being ignored or this fixture cannot tell the two apart, and "
        "the equality assertion above is then vacuous"
    )


# ---------------------------------------------------------------------------
# 2. the headline gate: an exact adjoint pair under equal flags
# ---------------------------------------------------------------------------


_IDENTITY_FIXTURES = [
    # (telescope, zenith angle, expects pixels outside the unit disc)
    pytest.param(EDA2, 0.0, True, id="EDA2_zenith_full_sky"),
    pytest.param(MWA_COMPACT, 0.0, False, id="MWA_compact_zenith"),
    pytest.param(MWA_COMPACT, 30.0, False, id="MWA_compact_off30"),
]


@pytest.mark.parametrize("tel, zenith_deg, has_outside_disc", _IDENTITY_FIXTURES)
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("dtype, eps, dot_tol", _PRECISIONS)
def test_dot_product_identity_under_equal_flags(
    tel: Telescope,
    zenith_deg: float,
    has_outside_disc: bool,
    divide_by_n: bool,
    dtype: DTypeLike,
    eps: float,
    dot_tol: float,
) -> None:
    """``Re<A x, y> = <x, A^H y>``, for **both** flag values, per fixture.

    This is issue #20's definition of done and issue #21's precondition: a
    ``custom_vjp`` is only correct if the pair is exactly adjoint, and it is
    exactly adjoint only with equal flags. Asserted per fixture rather than
    aggregated, so the wide-field case (EDA2 at 120 degrees, where the mixed
    default pair is off by 2.7e-1) cannot be averaged out by the narrow ones.

    No ``n`` correction anywhere in this identity: that factor is an artefact
    of the *mixed* pair and is what section 3 below pins separately.
    """
    problem = _problem(tel, zenith_deg, eps=eps, dtype=dtype)
    outside = int(_outside_disc(problem.plan).sum())
    assert (outside > 0) == has_outside_disc, (
        f"{tel.name} zen={zenith_deg} has {outside} pixels outside the unit disc, "
        f"expected {'some' if has_outside_disc else 'none'}: the fixture set must keep "
        "covering both regimes for this parametrisation to mean anything"
    )
    residual, lhs, rhs = _dot_product_residual(problem, divide_by_n=divide_by_n)
    assert residual < dot_tol, (
        f"{tel.name} zen={zenith_deg} divide_by_n={divide_by_n} dtype={np.dtype(jnp.dtype(dtype)).name}: "
        f"dot-product residual {residual:.3e} exceeds {dot_tol:.1e} "
        f"(lhs={lhs!r}, rhs={rhs!r}). With equal flags the pair must be exactly adjoint -- "
        "this is what issue #21's custom_vjp is allowed to assume."
    )


@requires_x64
def test_the_mixed_default_pair_is_not_an_adjoint_pair_on_the_full_sky() -> None:
    """Why the flag exists, pinned as a measurement rather than prose.

    The defaults are unchanged by issue #20, so on a wide field they still do
    **not** form an adjoint pair -- neither in the plain form nor with the
    ``n x`` correction, because outside the unit disc the forward evaluates
    the analytic extension while the adjoint zeroes those pixels. Measured
    here: 6.3e-1 plain, 2.7e-1 with ``n x``. Equal flags on the same fixture
    and the same data land at 1e-15 .. 1e-13.

    If someone "fixes" the wide-field gradient by flipping a default instead
    of adding the flag, this test is what says the defaults moved.
    """
    problem = _eda2_full_sky()
    _assert_outside_disc_is_loaded(problem)
    n_grid = _n_grid(problem.plan)
    image = np.asarray(problem.image, dtype=np.float64)

    plain, _, _ = _dot_product_residual(problem)
    corrected, _, _ = _dot_product_residual(problem, image_rhs=image * n_grid)
    assert plain > 1e-2 and corrected > 1e-2, (
        f"the default (mixed) pair now looks adjoint on EDA2 full-sky: plain residual "
        f"{plain:.3e}, n*x residual {corrected:.3e}. That is not something issue #20 "
        "delivers -- it keeps the defaults as they were and fixes the pair via equal "
        "flags. A default was probably changed."
    )
    for flag in FLAG_VALUES:
        equal, _, _ = _dot_product_residual(problem, divide_by_n=flag)
        assert equal < DOT_TOL_F64, (
            f"equal flags divide_by_n={flag} on the same fixture: residual {equal:.3e}"
        )


# ---------------------------------------------------------------------------
# 3. the documented behaviour of the mixed default pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tel, zenith_deg, mask_to_disc",
    [
        pytest.param(MWA_COMPACT, 0.0, False, id="MWA_compact_zenith_whole_image"),
        pytest.param(MWA_COMPACT, 30.0, False, id="MWA_compact_off30_whole_image"),
        pytest.param(EDA2, 0.0, True, id="EDA2_full_sky_disc_masked"),
    ],
)
@requires_x64
def test_mixed_default_pair_keeps_the_documented_n_correction(
    tel: Telescope, zenith_deg: float, mask_to_disc: bool
) -> None:
    """``Re<A x, y> = <n x, A^H y>`` for the defaults, restricted to the disc.

    This is the identity ``tests/test_adjoint.py`` asserts today, kept here so
    that issue #20 pins the documented behaviour of the *defaults* rather than
    silently changing it. Neither operator is passed a flag: the point is what
    a caller who says nothing gets.

    On EDA2 the image is masked to the unit disc first, which is precisely the
    restriction the docstring claims -- with the mask the identity holds at
    4e-16, without it at 2.7e-1 (the test above).
    """
    problem = _problem(tel, zenith_deg, eps=1e-6)
    n_grid = _n_grid(problem.plan)
    image = np.asarray(problem.image, dtype=np.float64)
    if mask_to_disc:
        assert int((n_grid <= 0.0).sum()) > 0, "EDA2 must have pixels outside the disc"
        image = np.where(n_grid > 0.0, image, 0.0)
    else:
        assert int((n_grid <= 0.0).sum()) == 0, (
            f"{tel.name} was expected to lie entirely inside the unit disc"
        )
    residual, lhs, rhs = _dot_product_residual(problem, image=image, image_rhs=image * n_grid)
    assert residual < DOT_TOL_F64, (
        f"{tel.name} zen={zenith_deg}: default-pair residual {residual:.3e} in the n*x form "
        f"exceeds {DOT_TOL_F64:.1e} (lhs={lhs!r}, rhs={rhs!r}). The defaults' documented "
        "behaviour has changed."
    )


# ---------------------------------------------------------------------------
# 4. what each flag *means*, independent of the identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_forward_with_divide_by_n_ignores_every_pixel_outside_the_disc(
    dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """``divide_by_n=True`` multiplies by **zero** outside the unit disc.

    Stated as insensitivity, which is the sharp version: two images that
    differ only outside the disc must give bit-identical visibilities under
    ``True``. The same pair under ``False`` must differ materially -- without
    that half the test would also pass on a forward that zeroed the whole
    image, or on a fixture with nothing outside the disc.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    outside = _assert_outside_disc_is_loaded(problem)
    rng = np.random.default_rng(2024)
    perturbed = np.array(problem.image, copy=True)
    perturbed[outside] += rng.standard_normal(int(outside.sum())).astype(perturbed.dtype) * 5.0

    with_flag = _forward(problem, problem.image, divide_by_n=True)
    with_flag_perturbed = _forward(problem, perturbed, divide_by_n=True)
    np.testing.assert_array_equal(
        with_flag,
        with_flag_perturbed,
        err_msg=(
            "dirty2vis(divide_by_n=True) changed when only the pixels OUTSIDE the unit "
            "disc changed. Those pixels must be multiplied by zero (n < 0 there, so 1/n "
            "is not a gain the measurement equation defines); leaving them at the "
            "analytic extension is what breaks the adjoint pair on wide fields."
        ),
    )
    without_flag = _forward(problem, problem.image, divide_by_n=False)
    without_flag_perturbed = _forward(problem, perturbed, divide_by_n=False)
    contrast = np.linalg.norm(without_flag - without_flag_perturbed) / np.linalg.norm(without_flag)
    assert contrast > 1e-2, (
        f"the control leg moved by only {contrast:.3e}: divide_by_n=False must stay "
        "sensitive to the outside-disc pixels (it uses the analytic extension there), "
        "otherwise the insensitivity asserted above is a property of the fixture"
    )
    assert np.linalg.norm(with_flag - without_flag) / np.linalg.norm(without_flag) > 1e-2, (
        "the two flag values give the same forward output on a full-sky fixture"
    )


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_forward_with_divide_by_n_applies_one_over_n_inside_the_disc(
    dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """Inside the disc the flag is exactly a ``1/n`` pre-multiplication.

    The companion to the test above: insensitivity outside would also be
    satisfied by a forward that zeroed everything, so the inside-disc half has
    to be pinned too. Compared against the *other* flag value fed the explicitly
    scaled image, so the two paths differ only in where the factor is applied.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    _assert_outside_disc_is_loaded(problem)
    n_grid = _n_grid(problem.plan)
    scaled = np.where(
        n_grid > 0.0,
        np.asarray(problem.image, dtype=np.float64) / np.where(n_grid > 0.0, n_grid, 1.0),
        0.0,
    ).astype(problem.image.dtype)

    got = _forward(problem, problem.image, divide_by_n=True).astype(np.complex128)
    want = _forward(problem, scaled, divide_by_n=False).astype(np.complex128)
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    tol = 1e-12 if np.dtype(jnp.dtype(dtype)) == np.float64 else 1e-5
    assert err < tol, (
        f"dirty2vis(divide_by_n=True) differs by {err:.3e} from dirty2vis on an image "
        f"pre-multiplied by the masked 1/n (tol {tol:.1e}). The flag must apply 1/n to the "
        "IMAGE inside the disc -- applying n instead, or applying it to the output "
        "visibilities, both land here."
    )


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_adjoint_without_divide_by_n_keeps_the_outside_disc_pixels(
    dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """``divide_by_n=False`` returns the analytic extension, not zeros.

    The default (``True``) zeroes every pixel with ``n <= 0`` -- there is no
    ``1/n`` to apply there -- and those zeros are exactly what breaks the
    mixed pair against a forward that keeps evaluating the extension. With the
    flag off the adjoint must return those values.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    outside = _assert_outside_disc_is_loaded(problem)

    divided = _adjoint(problem, divide_by_n=True)[0]
    undivided = _adjoint(problem, divide_by_n=False)[0]

    np.testing.assert_array_equal(
        divided[outside],
        np.zeros(int(outside.sum()), dtype=divided.dtype),
        err_msg="divide_by_n=True must still zero the pixels outside the unit disc",
    )
    inside_scale = float(np.max(np.abs(divided)))
    outside_signal = float(np.max(np.abs(undivided[outside])))
    assert outside_signal > 1e-3 * inside_scale, (
        f"vis2dirty(divide_by_n=False) returned {outside_signal:.3e} at most outside the "
        f"unit disc against an in-disc scale of {inside_scale:.3e}: those pixels are still "
        "being zeroed, so the flag is not returning the analytic-extension result and the "
        "pair stays non-adjoint on wide fields"
    )


@requires_x64
def test_adjoint_without_divide_by_n_matches_the_exact_dft_outside_the_disc() -> None:
    """The outside-disc values are *the right* values, not merely non-zero.

    Checked against an exact DFT written in the repo's sign convention with
    ``n - 1`` on the analytic extension and no division -- an oracle
    independent of ``src/``. Held to the repo's ``2 * eps`` DFT contract, on
    the outside-disc pixels alone (they carry about half the image norm here,
    so restricting the ratio to them does not inflate it).
    """
    eps = 1e-6
    problem = _problem(EDA2, 0.0, eps=eps, uvw_seed=1, data_seed=11)
    outside = _outside_disc(problem.plan)
    assert int(outside.sum()) > 0

    got = _adjoint(problem, divide_by_n=False)[0]
    want = _reference_adjoint_no_divide(
        problem.vis, problem.uvw, problem.freq, problem.shape, problem.pixsize
    )
    err_out = np.linalg.norm(got[outside] - want[outside]) / np.linalg.norm(want[outside])
    assert err_out < DFT_TOL_FACTOR * eps, (
        f"outside-disc relative error {err_out:.3e} exceeds {DFT_TOL_FACTOR * eps:.3e}: "
        "vis2dirty(divide_by_n=False) does not reproduce the analytic-extension adjoint "
        "there"
    )
    err_all = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err_all < DFT_TOL_FACTOR * eps, (
        f"whole-image relative error {err_all:.3e} exceeds {DFT_TOL_FACTOR * eps:.3e}"
    )


@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_the_two_adjoint_flag_values_differ_by_exactly_one_over_n_inside_the_disc(
    dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """Inside the disc, ``False`` is ``True`` times ``n``: nothing else moved.

    Pins that the flag changes the output factor and *only* the output factor
    -- an implementation that (say) also dropped the ``w0`` screen conjugation
    on the ``False`` path would satisfy the outside-disc test above and fail
    here.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    n_grid = _n_grid(problem.plan)
    inside = n_grid > 0.0
    assert int(inside.sum()) > 0

    divided = _adjoint(problem, divide_by_n=True)[0].astype(np.float64)
    undivided = _adjoint(problem, divide_by_n=False)[0].astype(np.float64)
    got = undivided[inside] / n_grid[inside]
    want = divided[inside]
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    tol = 1e-12 if np.dtype(jnp.dtype(dtype)) == np.float64 else 1e-5
    assert err < tol, (
        f"inside the disc, vis2dirty(divide_by_n=False)/n differs from "
        f"vis2dirty(divide_by_n=True) by {err:.3e} (tol {tol:.1e}): the flag changed more "
        "than the 1/n output factor"
    )


# ---------------------------------------------------------------------------
# 5. ducc0 parity for the two new flag combinations (black-box oracle)
# ---------------------------------------------------------------------------


@requires_x64
@pytest.mark.parametrize("w_strategy", ["dense_scan", "windowed_scan"])
@pytest.mark.parametrize("eps", [1e-4, 1e-6])
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_ducc_parity_for_the_new_flag_combinations(
    short_telescope_pointing: tuple[Telescope, float],
    op: str,
    eps: float,
    w_strategy: str,
) -> None:
    """``dirty2vis(divide_by_n=True)`` and ``vis2dirty(divide_by_n=False)``.

    The two combinations the operators could not express before issue #20,
    against ducc0 configured the same way, at the repo's ``3 * eps`` bound
    (``tests/test_against_ducc.py``). ducc0 is used through its public Python
    API only. The fixture set includes EDA2 at 120 degrees, so the parity is
    asserted on a field that runs past the unit disc, where the two libraries
    have to agree about the outside-disc convention as well as the accuracy.
    """
    ducc0_wgridder = pytest.importorskip("ducc0.wgridder")
    tel, zen_deg = short_telescope_pointing
    problem = _problem(tel, zen_deg, eps=eps, uvw_seed=0, data_seed=7)
    pix = problem.pixsize
    common = dict(
        uvw=problem.uvw,
        freq=problem.freq,
        pixsize_x=pix,
        pixsize_y=pix,
        epsilon=eps,
        do_wgridding=True,
        nthreads=1,
    )
    if op == "dirty2vis":
        got = _forward(problem, problem.image, divide_by_n=True, w_strategy=w_strategy)
        want = ducc0_wgridder.dirty2vis(
            dirty=np.ascontiguousarray(problem.image, dtype=np.float64),
            divide_by_n=True,
            **common,
        )
    else:
        got = _adjoint(problem, divide_by_n=False, w_strategy=w_strategy)[0]
        want = ducc0_wgridder.vis2dirty(
            vis=problem.vis,
            npix_x=problem.shape[0],
            npix_y=problem.shape[1],
            divide_by_n=False,
            **common,
        )
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err < DUCC_TOL_FACTOR * eps, (
        f"{tel.name} zen={zen_deg} eps={eps:g} {w_strategy} {op}: relative error {err:.3e} "
        f"exceeds {DUCC_TOL_FACTOR:g}*eps={DUCC_TOL_FACTOR * eps:.3e} for the new "
        "divide_by_n combination"
    )


# ---------------------------------------------------------------------------
# 6. composition with the flags already in the plan
# ---------------------------------------------------------------------------


@requires_x64
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("hermitian", [False, True])
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_both_flag_values_compose_with_every_strategy_and_fold_setting(
    op: str, w_strategy: str, hermitian: bool, divide_by_n: bool
) -> None:
    """All four ``w_strategy`` values x both ``hermitian`` values x both flags.

    Enumerated, not sampled: the sets are ``W_STRATEGIES`` (four) and
    ``hermitian`` (both), because "the flag composes with the w_strategy
    family" is a universally quantified claim and one witness would not gate
    it.

    Two references, because the two axes are different kinds of equivalence
    and the repo already prices them differently:

      * against ``dense_scan`` **at the same fold setting**: the four
        strategies are the same operator in a different accumulation order, so
        the 1e-11 strategy-equivalence bound applies
        (``tests/test_strategies_equivalent.py``);
      * against ``dense_scan`` on an ``hermitian=False`` plan: folded and
        unfolded are different approximations of the same operator and agree
        to ``4 * eps``, the triangle inequality on the ``2 * eps`` DFT
        contract each meets (``tests/test_hermitian.py``,
        ``CROSS_PATH_TOL_FACTOR``).

    Both are needed. A flag ignored on *every* folded strategy would keep the
    first check green (the same-fold reference is wrong in the same way) and
    is caught only by the second; a flag mishandled by one strategy is caught
    only by the first. The fold is the interesting axis: it conjugates the
    forward's *output* and the adjoint's *input*, while ``divide_by_n`` acts
    on the image end of both.
    """
    eps = 1e-6
    unfolded = _problem(EDA2, 0.0, eps=eps, hermitian=False)
    problem = _problem(EDA2, 0.0, eps=eps, hermitian=hermitian)
    assert problem.plan.hermitian is hermitian
    n_negative = int((problem.uvw[:, 2] < 0).sum())
    assert 0 < n_negative < problem.uvw.shape[0], (
        "the fixture must have mixed-sign w or the hermitian=True leg is the "
        "hermitian=False leg under another name"
    )
    if op == "dirty2vis":

        def run(p: _Problem, strategy: str) -> np.ndarray:
            return _forward(p, p.image, divide_by_n=divide_by_n, w_strategy=strategy).astype(
                np.complex128
            )
    else:

        def run(p: _Problem, strategy: str) -> np.ndarray:
            return _adjoint(p, divide_by_n=divide_by_n, w_strategy=strategy).astype(np.float64)

    got = run(problem, w_strategy)
    same_fold = run(problem, "dense_scan")
    cross_fold = run(unfolded, "dense_scan")

    label = f"{op} w_strategy={w_strategy} hermitian={hermitian} divide_by_n={divide_by_n}"
    strategy_err = np.linalg.norm(got - same_fold) / np.linalg.norm(same_fold)
    assert strategy_err < DOT_TOL_F64, (
        f"{label}: relative difference {strategy_err:.3e} against dense_scan on the same "
        f"plan exceeds the {DOT_TOL_F64:.1e} strategy-equivalence bound -- divide_by_n "
        "must compose with every w_strategy, not just the dense ones."
    )
    fold_err = np.linalg.norm(got - cross_fold) / np.linalg.norm(cross_fold)
    fold_bound = CROSS_PATH_TOL_FACTOR * eps
    assert fold_err < fold_bound, (
        f"{label}: relative difference {fold_err:.3e} against dense_scan on an unfolded "
        f"plan exceeds {fold_bound:.3e} ({CROSS_PATH_TOL_FACTOR:g}*eps) -- the flag must "
        "mean the same thing on a folded plan as on an unfolded one, and be applied on "
        "the image side of the fold's per-row conjugation rather than skipped there."
    )


@requires_x64
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("hermitian", [False, True])
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_the_pair_stays_adjoint_for_every_strategy_and_fold_setting(
    w_strategy: str, hermitian: bool, divide_by_n: bool
) -> None:
    """The identity itself, over the full strategy x fold x flag enumeration.

    Section 2 gates the identity on the shipped defaults for those two axes;
    this gates it everywhere, which is what issue #21 needs -- a ``custom_vjp``
    is chosen per call, and a caller who passes ``windowed_vmap`` on a folded
    plan must get the same guarantee as one who passes nothing.
    """
    problem = _problem(EDA2, 0.0, eps=1e-6, hermitian=hermitian)
    residual, lhs, rhs = _dot_product_residual(
        problem, divide_by_n=divide_by_n, w_strategy=w_strategy
    )
    assert residual < DOT_TOL_F64, (
        f"w_strategy={w_strategy} hermitian={hermitian} divide_by_n={divide_by_n}: "
        f"dot-product residual {residual:.3e} exceeds {DOT_TOL_F64:.1e} "
        f"(lhs={lhs!r}, rhs={rhs!r})"
    )


# ---------------------------------------------------------------------------
# 7. the flag is static: JIT / grad traceability (issue #21's precondition)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("dtype, eps, _dot_tol", _PRECISIONS)
def test_both_flag_values_stay_traceable_through_jit_and_grad(
    divide_by_n: bool, dtype: DTypeLike, eps: float, _dot_tol: float
) -> None:
    """Both operators, both flag values, under ``jax.jit`` and ``jax.grad``.

    Issue #21 replaces the reverse-mode rule with a ``custom_vjp`` built on
    this pair, so the flag has to survive tracing (i.e. be a static argument,
    not a traced one) and produce finite gradients at both values.
    """
    problem = _eda2_full_sky(dtype=dtype, eps=eps)
    image = jnp.asarray(problem.image)
    vis = jnp.asarray(problem.vis)

    fwd = jax.jit(lambda im: dirty2vis(problem.plan, im, divide_by_n=divide_by_n))
    adj = jax.jit(lambda v: vis2dirty(problem.plan, v, divide_by_n=divide_by_n))
    assert np.all(np.isfinite(np.asarray(fwd(image))))
    assert np.all(np.isfinite(np.asarray(adj(vis))))

    grad_fwd = jax.grad(lambda im: jnp.sum(jnp.abs(fwd(im)) ** 2))(image)
    grad_adj = jax.grad(lambda v: jnp.sum(adj(v) ** 2))(vis.real)
    assert np.all(np.isfinite(np.asarray(grad_fwd)))
    assert np.all(np.isfinite(np.asarray(grad_adj)))
    assert float(np.linalg.norm(np.asarray(grad_fwd, dtype=np.float64))) > 0.0
