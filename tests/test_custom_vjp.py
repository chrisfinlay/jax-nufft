"""Issue #21: reverse mode through the wgridder must cost one adjoint call.

What this module gates
----------------------
``jax.grad`` through the scan strategies costs ``O(n_w * image)`` today,
because reverse-mode AD saves one residual per w-plane. Measured with
``jax.jit(fn).lower(...).compile().memory_analysis().temp_size_in_bytes`` on
this repository at ``HEAD`` before the change (CPU, float64, ``eps = 1e-6``,
``nthreads=1``, ``divide_by_n=True`` on both operators):

===========================  ==============  =========  ==========  =======
fixture                      strategy        forward    grad        ratio
===========================  ==============  =========  ==========  =======
MWA_compact off30 (128^2)    dense_scan       0.534 MB    7.213 MB   13.5x
MWA_compact off30 (128^2)    windowed_scan    0.544 MB    7.338 MB   13.5x
MWA_compact off30 (128^2)    dense_vmap       3.261 MB    6.349 MB    1.95x
MWA_compact off30 (128^2)    windowed_vmap    3.549 MB    6.695 MB    1.89x
MWA_extended off30 (256^2)   dense_scan       2.11 MB   285.47 MB   135.5x
MWA_extended off30 (256^2)   windowed_scan    2.11 MB   284.86 MB   134.9x
1024^2 / 20k rows            dense_scan      33.87 MB   897.83 MB    26.5x
===========================  ==============  =========  ==========  =======

and after the prototype that binds the operators as linear primitives whose
transposes are each other:

===========================  ==============  =========  ==========  =======
fixture                      strategy        forward    grad        ratio
===========================  ==============  =========  ==========  =======
MWA_compact off30 (128^2)    dense_scan       0.534 MB    0.796 MB    1.49x
MWA_compact off30 (128^2)    windowed_scan    0.544 MB    0.806 MB    1.48x
MWA_extended off30 (256^2)   dense_scan       2.11 MB     3.16 MB     1.50x
1024^2 / 20k rows            dense_scan      33.87 MB    50.65 MB     1.50x
===========================  ==============  =========  ==========  =======

**The ``*_vmap`` rows are not a gate and never were.** Their forward already
allocates ``n_w * image``, so their gradient ratio was 1.82-2.10 *before* the
change: issue #21's definition of done says "grad temp <= 2x forward for all
four strategies", and on two of the four that sentence is already true at
``HEAD``. They are kept here as a regression guard at **1.5x**, and the gate
that carries the issue is the 2.0x on the two ``*_scan`` strategies, where the
measured before/after is 13.5x -> 1.49x.

1.5 rather than the 2.5 this module first carried, which could not have failed:
the pre-change ratios on this fixture are 1.89-1.95x on ``dirty2vis`` and
2.07-2.11x on ``vis2dirty`` (both precision legs), so a complete revert of the
change left all four vmap cells inside 2.5 and the "regression guard" gated
nothing. After the change they read 0.98-1.18x, so 1.5 sits with 27% headroom
above the worst measured cell (``windowed_vmap`` / ``vis2dirty``, float32) and
below every pre-change one.

Why a separate module rather than more of ``test_jax_integration.py``: this is
a contract about a *rule*, not about plumbing. It has to pin the cotangent
convention (which JAX gets to choose, and which #20 measured), the static
configuration the backward runs with (invisible in every number), and the
memory (invisible in every value). Those are one subject.

Precision
---------
Written precision-aware from the start (AGENTS.md sec 6): every numeric test is
parametrised over ``test_divide_by_n.py``'s ``_IDENTITY_PRECISIONS`` table and
takes its tolerance from it, so the float32 entry runs in **both** legs (a
float32 plan is legal under x64) and the bound follows the parametrised dtype
rather than the run's own precision. The tests that are not parametrised --
memory, and the two spy tests -- build their plan at the run's precision.
This module must never be added to ``conftest.collect_ignore``.

The conventions this module pins, and how they were established
---------------------------------------------------------------
All four are *measured*, on a 16^2 / 40-row ``hermitian=False`` plan at
``eps = 1e-8``, float64, against a dense matrix ``A`` built column by column
from ``dirty2vis`` itself. Writing ``A`` for the forward operator:

* ``jax.vjp(dirty2vis)(y)`` is ``A^T y`` -- the **plain transpose**, no
  conjugation -- restricted to its real part when the image is real. Measured
  1.5e-16 relative against ``A^T y``, against 1.47 for ``A^H y`` and 1.32 for
  ``conj(A^T y)``.
* ``vis2dirty(v)`` is ``Re(A^H v)`` (1.3e-15).
* hence ``jax.vjp(dirty2vis)(y) == vis2dirty(conj(y))`` for a real image --
  issue #20's rule, pinned in
  ``test_divide_by_n.py::test_the_reverse_mode_vjp_is_vis2dirty_of_the_
  conjugated_cotangent`` and *not* repeated here.
* ``jax.vjp(vis2dirty)(u) == conj(dirty2vis(u))`` for a real cotangent image
  (0.0 exactly; 1.37 against the unconjugated form).

The complex-image case is the one #21's sketch gets wrong. The sketch's ``bwd``
returns ``_dirty2vis_adjoint(conj(ct))`` -- that is ``A^H conj(ct)``, i.e.
``conj(A^T ct)`` -- and casts it to the image dtype. For a **real** image that
is right by accident, since ``Re(conj z) == Re(z)``; for a **complex** image it
is the conjugate of the answer, wrong by a relative 1.32-1.49 on the fixture
below. See ``test_the_complex_image_cotangent_is_the_plain_transpose``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.extend as jex
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from jax.typing import DTypeLike

import jax_nufft.wgridder as wgridder
from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
from tests.conftest import (
    MWA_COMPACT,
    X64,
    Telescope,
    reference_lmn_grids,
    requires_x64,
    tol,
)
from tests.test_divide_by_n import (
    _ANISO_GEOMETRIES,
    _BOUNDARY_PIXSIZE,
    _IDENTITY_PRECISIONS,
    CHANNEL_STRATEGIES,
    FLAG_VALUES,
    IDENTITY_TOL_F64,
    W_STRATEGIES,
    _n_grid,
    _problem,
)
from tests.test_divide_by_n import EDA2 as EDA2  # re-export for readability

# ---------------------------------------------------------------------------
# tolerances
# ---------------------------------------------------------------------------

# Every numeric assertion in this module is an identity that is exact on paper,
# so the residual is pure round-off and the bound comes from
# ``test_divide_by_n``'s ``_IDENTITY_PRECISIONS`` table (1e-11 in float64,
# 5e-6 in float32) rather than from anything measured here. Note that the
# float32 entry of that table runs in BOTH precision legs -- a float32 plan is
# legal under x64 -- so the tolerance has to follow the parametrised dtype,
# never the run's own precision.

# The gradient-memory gates. See the module docstring for the before/after
# table these come from: 13.5x -> 1.49x on the scan strategies (so 2.0 sits
# between the two with 6.7x margin on the failing side), and 1.89-2.11 ->
# 0.98-1.18 on the vmap ones, where 1.5 is a regression guard rather than a
# gate because their forward already carries the n_w * image allocation.
GRAD_MEMORY_FACTOR_SCAN = 2.0
GRAD_MEMORY_FACTOR_VMAP = 1.5

SCAN_STRATEGIES = ("dense_scan", "windowed_scan")


def _factor(w_strategy: str) -> float:
    return GRAD_MEMORY_FACTOR_SCAN if w_strategy.endswith("_scan") else GRAD_MEMORY_FACTOR_VMAP


def _active_dtype() -> DTypeLike:
    """The precision the run itself was configured with (conftest's ``X64``)."""
    return jnp.float64 if X64 else jnp.float32


def _rel(a: Any, b: Any) -> float:
    """Relative L2 difference, in float64 regardless of the operand precision."""
    a64 = np.asarray(a, dtype=np.complex128).ravel()
    b64 = np.asarray(b, dtype=np.complex128).ravel()
    denom = np.linalg.norm(b64)
    assert denom > 0.0, "reference is identically zero; the comparison is vacuous"
    return float(np.linalg.norm(a64 - b64) / denom)


def _complex_adjoint(plan: Any, v: Any, **kw: Any) -> Any:
    """``A^H v`` (complex) built from the **public** adjoint alone.

    ``vis2dirty`` returns ``Re(A^H v)``. Since ``Re(A^H (-i v)) = Re(-i A^H v)
    = Im(A^H v)``, two public calls recover the full complex adjoint without
    reaching into any private helper -- so the reference this module compares
    gradients against is not the implementation of the gradient.
    """
    re = vis2dirty(plan, v, **kw)
    im = vis2dirty(plan, v * jnp.asarray(-1j, dtype=v.dtype), **kw)
    return re + 1j * im


def _transpose_reference(plan: Any, y: Any, **kw: Any) -> Any:
    """``A^T y = conj(A^H conj(y))`` -- what ``jax.vjp`` must return."""
    return jnp.conj(_complex_adjoint(plan, jnp.conj(y), **kw))


def _temp_bytes(fn: Any, *args: Any) -> int:
    """XLA transient memory for ``fn`` at these argument shapes (AGENTS.md sec 6)."""
    analysis = jax.jit(fn).lower(*args).compile().memory_analysis()
    if analysis is None:  # pragma: no cover - backend without the analysis
        pytest.skip("this backend does not expose memory_analysis()")
    return int(analysis.temp_size_in_bytes)


# ---------------------------------------------------------------------------
# 1. memory -- the point of the issue
# ---------------------------------------------------------------------------


def _memory_cell(
    problem: Any,
    *,
    op: str,
    w_strategy: str,
    channel_strategy: str,
    divide_by_n: bool = True,
) -> tuple[int, int]:
    """``(forward_temp, grad_temp)`` for one operator / strategy cell."""
    kw = dict(
        w_strategy=w_strategy,
        channel_strategy=channel_strategy,
        nthreads=1,
        divide_by_n=divide_by_n,
    )
    plan = problem.plan
    if op == "dirty2vis":
        arg = jnp.asarray(problem.image)

        def call(x: Any) -> Any:
            return dirty2vis(plan, x, **kw)

        def loss(x: Any) -> Any:
            return jnp.sum(jnp.abs(call(x)) ** 2)
    else:
        arg = jnp.asarray(problem.vis)

        def call(x: Any) -> Any:
            return vis2dirty(plan, x, **kw)

        def loss(x: Any) -> Any:
            return jnp.sum(call(x) ** 2)

        arg = arg.real  # a real vis keeps grad's own dtype off the comparison

    return _temp_bytes(call, arg), _temp_bytes(jax.grad(loss), arg)


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_gradient_memory_is_within_a_small_factor_of_the_forward(op: str, w_strategy: str) -> None:
    """``grad`` must not allocate one residual per w-plane.

    The gate is a *ratio* against the same call's forward, not a pinned byte
    count: absolute figures move with XLA, the platform and the precision leg,
    while "the backward is one adjoint call" is a statement about scaling.

    Measured on MWA_compact off30 (float64, eps 1e-6, ``nthreads=1``):
    13.5x before, 1.48-1.49x after on the two scan strategies. The vmap
    strategies read 1.89-2.11x before (``vis2dirty`` being the pair above 2.0)
    and 0.98-1.18x after -- see the module docstring for why they cannot gate
    this issue, and why their guard sits at 1.5.
    """
    dtype = _active_dtype()
    problem = _problem(MWA_COMPACT, 30.0, eps=tol(1e-6, 1e-5), dtype=dtype)
    fwd, grad = _memory_cell(problem, op=op, w_strategy=w_strategy, channel_strategy="scan")
    factor = _factor(w_strategy)

    if w_strategy in SCAN_STRATEGIES:
        # Non-vacuity: on this fixture a backward that saved one image-sized
        # residual per w-plane could not possibly fit inside the gate. Without
        # this the assertion below could pass on a plan too small to tell the
        # two regimes apart.
        plan = problem.plan
        per_plane = plan.n_w * plan.n_l * plan.n_m * np.dtype(plan.complex_dtype).itemsize
        assert per_plane > factor * fwd, (
            f"{op} ({w_strategy}): the fixture is too small to falsify anything -- "
            f"n_w * image = {per_plane / 1e6:.3f} MB is already inside "
            f"{factor} x forward ({factor * fwd / 1e6:.3f} MB), so a per-plane "
            "residual backward would pass the gate below"
        )

    assert grad <= factor * fwd, (
        f"{op} ({w_strategy}, channel_strategy=scan): grad transient memory is "
        f"{grad / 1e6:.3f} MB against a forward of {fwd / 1e6:.3f} MB "
        f"({grad / fwd:.2f}x, gate {factor}x) on a plan with n_w = "
        f"{problem.plan.n_w}. That is the O(n_w * image) reverse-mode residual "
        "stack issue #21 exists to remove: the backward pass must be one call to "
        "the adjoint operator, not a transposed replay of the w-plane loop."
    )


@pytest.mark.parametrize("channel_strategy", CHANNEL_STRATEGIES)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_gradient_memory_holds_with_several_channels(op: str, channel_strategy: str) -> None:
    """The same gate at ``n_chan == 2``, over both channel strategies.

    A single-channel plan cannot see a backward that runs the channel loop
    under ``vmap`` while the forward scans it: ``channel_strategy="vmap"``
    costs ``n_chan`` x the per-channel transient (AGENTS.md sec 5), which is
    invisible at ``n_chan == 1``. Measured on EDA2 zenith at ``n_chan = 2``,
    ``dense_scan``: 7.6x (dirty2vis) and 9.2x (vis2dirty) before the change,
    1.25x and 1.31x after, with ``channel_strategy="scan"``; 3.9x before and
    1.12-1.27x after with ``channel_strategy="vmap"``. Unlike the
    single-channel table above, the ``vmap`` channel loop is a live gate here
    -- it was 3.9x before.
    """
    dtype = _active_dtype()
    problem = _problem(EDA2, 0.0, eps=tol(1e-6, 1e-5), dtype=dtype, n_chan=2)
    assert problem.plan.n_chan == 2
    fwd, grad = _memory_cell(
        problem, op=op, w_strategy="dense_scan", channel_strategy=channel_strategy
    )
    factor = GRAD_MEMORY_FACTOR_SCAN
    assert grad <= factor * fwd, (
        f"{op} (dense_scan, channel_strategy={channel_strategy}, n_chan=2): grad "
        f"transient memory {grad / 1e6:.3f} MB against forward {fwd / 1e6:.3f} MB "
        f"({grad / fwd:.2f}x, gate {factor}x)"
    )


@requires_x64
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("w_strategy", SCAN_STRATEGIES)
def test_gradient_memory_at_the_review_fixtures(
    long_telescope_pointing: tuple[Telescope, float], op: str, w_strategy: str
) -> None:
    """The 256^2 fixtures issue #21 quotes: 135x before, 1.50x after.

    Gated behind ``--runslow`` through the ``long_telescope_pointing`` fixture.
    This is the cell the issue's headline number comes from (MWA_extended
    off30, n_w = 134), and the reason the fast-leg test above uses a 128^2
    plan is only run time.
    """
    telescope, zenith_angle_deg = long_telescope_pointing
    problem = _problem(telescope, zenith_angle_deg, eps=1e-6, dtype=jnp.float64)
    fwd, grad = _memory_cell(problem, op=op, w_strategy=w_strategy, channel_strategy="scan")
    assert grad <= GRAD_MEMORY_FACTOR_SCAN * fwd, (
        f"{telescope.name} at {zenith_angle_deg} deg, {op} ({w_strategy}), "
        f"n_w = {problem.plan.n_w}: grad transient memory {grad / 1e6:.2f} MB "
        f"against forward {fwd / 1e6:.2f} MB ({grad / fwd:.1f}x, gate "
        f"{GRAD_MEMORY_FACTOR_SCAN}x)"
    )


# ---------------------------------------------------------------------------
# 2. the cotangent convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_the_complex_image_cotangent_is_the_plain_transpose(
    w_strategy: str, divide_by_n: bool, dtype: DTypeLike, eps: float, exact_tol: float
) -> None:
    """``jax.vjp(dirty2vis)(y) == A^T y`` for a **complex** image.

    This is the half of the convention that issue #20's test cannot reach: it
    differentiates a real image, where the answer is ``Re(A^T y)`` and the two
    candidate rules -- ``A^T y`` and ``conj(A^T y)`` -- agree. On a complex
    image they do not, and issue #21's sketch returns the wrong one of the two:
    its ``bwd`` is ``_dirty2vis_adjoint(conj(ct))``, which is ``A^H conj(ct)``,
    which is ``conj(A^T ct)``.

    The reference is built from the public ``vis2dirty`` alone (see
    :func:`_complex_adjoint`), so this does not compare an implementation
    against itself. The two contrast assertions are what make the tolerance
    meaningful: on this fixture the wrong conventions are 1.3-1.5 away in
    relative L2 while returning entirely finite, plausible-looking gradients.

    ``hermitian=False``: the fold is a real-sky identity and ``dirty2vis``
    refuses a complex image on a folded plan (issue #17), so the complex leg of
    the convention can only be asked on an unfolded one. The folded leg is
    covered by the real-image test in ``test_divide_by_n.py``.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype, hermitian=False)
    kw = dict(divide_by_n=divide_by_n, w_strategy=w_strategy, nthreads=1)
    rng = np.random.default_rng(21)
    shape = problem.image.shape
    image = jnp.asarray(
        (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            problem.plan.complex_dtype
        )
    )
    y = jnp.asarray(problem.vis)

    _, vjp_fn = jax.vjp(lambda im: dirty2vis(problem.plan, im, **kw), image)
    got = vjp_fn(y)[0]
    want = _transpose_reference(problem.plan, y, **kw)[0]

    err = _rel(got, want)
    assert err < exact_tol, (
        f"{w_strategy}, divide_by_n={divide_by_n}: the reverse-mode cotangent of a "
        f"COMPLEX image differs by {err:.3e} (tol {exact_tol:.1e}) from A^T y = "
        "conj(A^H conj(y)). JAX transposes without conjugating, so the backward pass "
        "owes conj(_dirty2vis_adjoint(conj(ct))) here -- issue #21's sketch returns "
        "_dirty2vis_adjoint(conj(ct)) unconjugated, which is right for a real image "
        "only because Re(conj z) == Re z."
    )

    conjugated = _rel(got, jnp.conj(want))
    assert conjugated > 1e-2, (
        f"conj(A^T y) is only {conjugated:.3e} from the vjp on this fixture, so it "
        "cannot tell the sketch's rule from the correct one and the assertion above "
        "pins nothing"
    )
    hermitian_adjoint = _rel(got, _complex_adjoint(problem.plan, y, **kw)[0])
    assert hermitian_adjoint > 1e-2, (
        f"A^H y is only {hermitian_adjoint:.3e} from the vjp on this fixture, so a "
        "backward that dropped the input conjugation would pass unnoticed"
    )


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("hermitian", [False, True])
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_the_vis2dirty_cotangent_is_the_conjugated_forward(
    w_strategy: str,
    hermitian: bool,
    divide_by_n: bool,
    dtype: DTypeLike,
    eps: float,
    exact_tol: float,
) -> None:
    """``jax.vjp(vis2dirty)(u) == conj(dirty2vis(u))`` for a real cotangent image.

    The mirror of issue #20's rule, and the one the *second* half of #21 has to
    reproduce: ``vis2dirty`` is ``Re(A^H .)``, whose transpose against a real
    image ``u`` is ``conj(A u)``. Measured 0.0 exactly against ``conj(A u)`` and
    1.37-1.42 against ``A u`` -- so, as on the forward side, forgetting the
    conjugation is a 1.4-relative error that still returns finite numbers.

    The cotangent of ``vis2dirty``'s output is a **real** image, which is why
    ``dirty2vis`` can be the whole of the rule even on a folded plan: the fold
    refuses complex images, and no cotangent that appears here is one.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype, hermitian=hermitian)
    kw = dict(divide_by_n=divide_by_n, w_strategy=w_strategy, nthreads=1)
    vis = jnp.asarray(problem.vis)
    cotangent = jnp.asarray(problem.image)[None].astype(problem.plan.real_dtype)

    _, vjp_fn = jax.vjp(lambda v: vis2dirty(problem.plan, v, **kw), vis)
    got = vjp_fn(cotangent)[0]
    forward = dirty2vis(problem.plan, cotangent[0], **kw)

    err = _rel(got, jnp.conj(forward))
    assert err < exact_tol, (
        f"{w_strategy}, hermitian={hermitian}, divide_by_n={divide_by_n}: "
        f"jax.vjp(vis2dirty) differs by {err:.3e} (tol {exact_tol:.1e}) from "
        "conj(dirty2vis(cotangent)). The transpose of Re(A^H .) is conj(A .); "
        "dropping that conjugation is a ~1.4 relative error that still returns "
        "finite gradients."
    )
    contrast = _rel(got, forward)
    assert contrast > 1e-2, (
        f"the unconjugated dirty2vis is only {contrast:.3e} from the vjp on this "
        "fixture, so it cannot tell the convention from its absence"
    )


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("channel_strategy", CHANNEL_STRATEGIES)
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_the_gradient_of_half_the_squared_norm_is_the_normal_equations(
    w_strategy: str,
    channel_strategy: str,
    divide_by_n: bool,
    dtype: DTypeLike,
    eps: float,
    exact_tol: float,
) -> None:
    """``grad(0.5 ||A x||^2) == Re(A^H A x)``, computed with the adjoint operator.

    The identity every downstream optimisation loop actually relies on, and the
    one #21's definition of done names at 1e-11. It is checked against a
    composition of the two *public* operators at equal ``divide_by_n``, so it
    constrains the backward pass against the shipped adjoint rather than
    against another copy of itself.

    Parametrised over both loops because the backward is separate code once
    #21 lands: nothing else forces the gradient of a ``windowed_vmap`` forward
    to be a ``windowed_vmap`` adjoint.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype)
    kw = dict(
        divide_by_n=divide_by_n,
        w_strategy=w_strategy,
        channel_strategy=channel_strategy,
        nthreads=1,
    )
    image = jnp.asarray(problem.image)

    def loss(im: Any) -> Any:
        return 0.5 * jnp.sum(jnp.abs(dirty2vis(problem.plan, im, **kw)) ** 2)

    got = jax.grad(loss)(image)
    want = vis2dirty(problem.plan, dirty2vis(problem.plan, image, **kw), **kw)[0]

    err = _rel(got, want)
    assert err < exact_tol, (
        f"{w_strategy}/{channel_strategy}, divide_by_n={divide_by_n}: "
        f"grad(0.5||A x||^2) differs by {err:.3e} (tol {exact_tol:.1e}) from "
        "vis2dirty(dirty2vis(x)) at the same flag. Both sides are Re(A^H A x); a "
        "backward that used a different divide_by_n, a different strategy family or "
        "the wrong conjugation lands elsewhere."
    )


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_the_gradient_of_a_complex_image_loss_is_the_conjugated_normal_equations(
    w_strategy: str, divide_by_n: bool, dtype: DTypeLike, eps: float, exact_tol: float
) -> None:
    """``grad(0.5 ||A x||^2)(x) == conj(A^H A x)`` for a **complex** image.

    The Wirtinger half of the identity above, and the one #22 names. JAX's
    convention for ``f: C^n -> R`` is the *conjugate* Wirtinger gradient,
    ``g = dL/d(Re x) - i dL/d(Im x)`` -- which is what makes ``x - lr * g`` a
    descent step, and what the finite-difference smoke test in
    ``test_jax_integration.py`` checks componentwise. Writing ``x = u + iv`` and
    ``w = A^H A x``, a perturbation gives ``dL = Re<w, dx>``, so
    ``dL/du = Re(w)`` and ``dL/dv = Im(w)``, hence ``g = Re(w) - i Im(w) =
    conj(w)``.

    So the complex case is *not* the real one with the ``Re`` dropped: it is the
    real one conjugated. On a real image ``g = Re(A^H A x)`` and the conjugation
    is invisible, which is why the test above cannot reach this.

    ``A^H`` comes from :func:`_complex_adjoint`, two calls to the public
    ``vis2dirty`` -- so this constrains the backward pass against the shipped
    adjoint, not against another copy of itself. ``hermitian=False`` because
    ``dirty2vis`` refuses a complex image on a folded plan (#17).

    Round-off row: the identity is exact on paper, so the bound is
    ``_IDENTITY_PRECISIONS`` (1e-11 float64 / 5e-6 float32) rather than
    anything measured here.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype, hermitian=False)
    kw = dict(divide_by_n=divide_by_n, w_strategy=w_strategy, nthreads=1)
    image = jnp.asarray(problem.image)
    x = (image + 0.5j * image[::-1]).astype(problem.plan.complex_dtype)

    def loss(z: Any) -> Any:
        vis = dirty2vis(problem.plan, z, **kw)
        return 0.5 * jnp.sum(vis.real**2 + vis.imag**2)

    got = jax.grad(loss)(x)
    normal = _complex_adjoint(problem.plan, dirty2vis(problem.plan, x, **kw), **kw)

    err = _rel(got, jnp.conj(normal))
    assert err < exact_tol, (
        f"{w_strategy}, divide_by_n={divide_by_n}: grad(0.5||Ax||^2) differs by "
        f"{err:.3e} (tol {exact_tol:.1e}) from conj(A^H A x). JAX returns the "
        "conjugate Wirtinger gradient, so the complex case is the real one "
        "conjugated, not the real one with the Re dropped."
    )

    # Without these the tolerance pins nothing: on a real image all three
    # candidates coincide, so the cell has to show they are separated here.
    unconjugated = _rel(got, normal)
    assert unconjugated > 1e-2, (
        f"A^H A x is only {unconjugated:.3e} from the gradient on this fixture, so "
        "this cell cannot tell the conjugate Wirtinger convention from its absence"
    )
    real_part = _rel(got, jnp.real(normal).astype(got.dtype))
    assert real_part > 1e-2, (
        f"Re(A^H A x) -- the real-image answer -- is only {real_part:.3e} away, so "
        "this cell adds nothing over the real-image identity above"
    )


# ---------------------------------------------------------------------------
# the derivative reference: JAX's own VJP of an exact DFT
# ---------------------------------------------------------------------------

# Every other gradient test in this module compares the backward pass against
# something built from the library's *own* forward and adjoint. That pins the
# transpose algebra exactly (those are identities, good to 1e-11) but it is
# blind by construction to a shared-mode error: if the forward is wrong and the
# adjoint is wrong the same way, ``A^H A`` is still self-consistent and every
# identity above still holds.
#
# The exact DFT is the independent statement of what the operator *is*, and
# ``jax.vjp`` of it is the independent statement of what its derivative is --
# differentiated by JAX through a two-line sum rather than by a hand-written
# transpose rule. This is the same ordering the issue #15 review established for
# the forward operator: the DFT at ``2 * eps`` catches shared-mode errors that a
# library-against-library comparison at a looser bound cannot see. It applies to
# the gradients for exactly the same reason.
#
# Approximation-gap row, not a round-off one: the wgridder approximates the DFT
# to the requested epsilon and its transpose inherits that same approximation,
# so the bound here is ``DFT_GRAD_TOL_FACTOR * eps`` and not
# ``_IDENTITY_PRECISIONS``. Measured below.

_DFT_N_PIX = 32
_DFT_N_ROWS = 48
_DFT_PIXSIZE = 1e-2
_DFT_W_METRES = 60.0
_DFT_FREQ = np.array([1.4e9])

# The fixture has to have real w depth, or this cell degenerates into a test of
# the 2-D NUFFT. The first version of it used 16^2 / 24 rows / +-3 m of w, which
# gave ``n_w - w_kernel_width == 1``: the plane count was ENTIRELY kernel
# overhang, the w-phase ``2 pi w (n - 1)`` spanned 0.137 rad against 9.3 turns
# for ``2 pi u l``, and a 10x coarsening of the w-plane spacing
# (``W_OVERSAMPLE_X0`` 0.25 -> 2.5) failed 206 of the 240 cells in this module
# while leaving all 24 of these unchanged to every printed digit. The cell
# advertised as the one able to see a shared-mode error was the only cell in the
# module that could not see that one. Both guards in the test body exist to stop
# that returning silently.
#
# At +-60 m of w on a 32^2 / 1e-2 rad grid, over both fold settings and all
# three eps: n_w 20-37 against w_kernel_width 7, so 15-28 inner planes; and
# max_window_size 13-32 of 48 rows, a strict subset everywhere, so the windowed
# strategies are a different computation from the dense ones rather than the
# same one spelled differently. +-60 is the smallest spread at which BOTH hold
# on all six cells: at +-30 the folded plan at eps=1e-8 has max_window_size 48
# of 48, the fold having halved the w extent. The guards in the test body are
# what found that, so they stay.

# Measured on this machine (macOS arm64, jax 0.9.2, float64), relative L2 of
# the wgridder cotangent against jax.vjp of the exact DFT, worst over the four
# w_strategy values and both divide_by_n settings:
#
#     eps       real folded    real unfolded   complex unfolded
#     1e-4      0.49x          0.49x           0.50x
#     1e-6      0.69x          0.73x           0.71x
#     1e-8      0.75x          0.86x           0.83x
#
# 0.86x eps at worst (4.881e-05 / 6.945e-07 / 7.520e-09 folded; 4.909e-05 /
# 7.311e-07 / 8.627e-09 unfolded real; 5.005e-05 / 7.076e-07 / 8.339e-09
# complex). Higher than the 0.21-0.33x the degenerate fixture gave, and it
# should be: the w-kernel now contributes to the error instead of being one
# plane's worth of overhang. It sits at the top of the 0.15x-0.84x band
# tests/test_against_dft.py measures for the forward.
#
# Approximation-gap row, so per the #20 taxonomy it is backend-stable; 2.0
# leaves 2.3x of margin, matching the factor test_against_dft uses for the
# forward. (The round-off rows in this module sit at ``_IDENTITY_PRECISIONS``
# instead, and those are the ones that move between backends.)
DFT_GRAD_TOL_FACTOR = 2.0


def _dft_geometry() -> tuple[np.ndarray, np.ndarray]:
    """A small (u, v, w) set with genuine w depth, and a matching image."""
    rng = np.random.default_rng(5)
    uvw = np.zeros((_DFT_N_ROWS, 3))
    uvw[:, 0] = rng.uniform(-50.0, 50.0, size=_DFT_N_ROWS)
    uvw[:, 1] = rng.uniform(-50.0, 50.0, size=_DFT_N_ROWS)
    uvw[:, 2] = rng.uniform(-_DFT_W_METRES, _DFT_W_METRES, size=_DFT_N_ROWS)
    image = rng.standard_normal((_DFT_N_PIX, _DFT_N_PIX))
    return uvw, image


def _dft_forward_jnp(image: Any, uvw: np.ndarray, *, divide_by_n: bool = False) -> Any:
    """``V = sum_lm I(l, m) [/ n] exp(-2 pi i (u l + v m - w (n - 1)))``, in ``jnp``.

    Written from the README's sign convention and :func:`reference_lmn_grids`
    -- the grid helper is itself written from the README rather than from
    ``planning`` -- so neither the operator nor the geometry is taken from the
    code under test. The only thing imported from ``src`` is the speed of light.
    Small enough (48 x 32 x 32) that ``jax.vjp`` through it is free.

    ``divide_by_n`` is the plain image-side ``1 / n`` of #20. This fixture needs
    no disc handling for it: at 32 pixels of 1e-2 rad the largest ``l^2 + m^2``
    is 0.051, so ``n >= 0.974`` everywhere and the ``n == 0`` branch that
    ``_disc_mask_and_safe_n`` exists for is never reached. Outside-disc
    behaviour is covered by ``tests/test_divide_by_n.py``.
    """
    ll, mm, nm1 = reference_lmn_grids((_DFT_N_PIX, _DFT_N_PIX), _DFT_PIXSIZE, _DFT_PIXSIZE)
    scale = _DFT_FREQ[0] / SPEED_OF_LIGHT
    u = jnp.asarray(uvw[:, 0] * scale)[:, None, None]
    v = jnp.asarray(uvw[:, 1] * scale)[:, None, None]
    w = jnp.asarray(uvw[:, 2] * scale)[:, None, None]
    phase = -2j * jnp.pi * (u * jnp.asarray(ll) + v * jnp.asarray(mm) - w * jnp.asarray(nm1))
    weighted = image / jnp.asarray(nm1 + 1.0) if divide_by_n else image
    return jnp.sum(weighted[None] * jnp.exp(phase), axis=(1, 2))[:, None]


# ``hermitian=True`` only with a real image: ``dirty2vis`` refuses a complex one
# on a folded plan (#17, the fold is a real-sky identity). Both are run for the
# real image so this cell exercises the SHIPPED default plan, not only the
# unfolded one.
_DFT_IMAGE_FOLD_CELLS = [
    pytest.param(False, True, id="real-folded"),
    pytest.param(False, False, id="real-unfolded"),
    pytest.param(True, False, id="complex-unfolded"),
]


@requires_x64
@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
@pytest.mark.parametrize("complex_image, hermitian", _DFT_IMAGE_FOLD_CELLS)
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_the_cotangent_matches_jax_vjp_of_the_exact_dft(
    w_strategy: str, complex_image: bool, hermitian: bool, divide_by_n: bool, eps: float
) -> None:
    """``jax.vjp(dirty2vis)`` against ``jax.vjp`` of the exact DFT.

    The only gradient test in the repository whose reference is not built from
    the library's own operators. Both the operator and its derivative come from
    outside: the DFT is a direct sum written from the README, and its VJP is
    produced by JAX differentiating that sum, not by any transpose rule of ours.

    So this is the cell that can see a *shared-mode* error -- a forward and an
    adjoint wrong in the same way, which every ``A^H A`` identity in this module
    accepts by construction. Measured: giving the (u, v) NUFFT the whole epsilon
    budget fails 16 of these cells and moves no other cell in the module;
    coarsening the w-plane spacing 10x fails them too, on this fixture (it did
    not on the degenerate one this replaces -- see the note above the
    constants).

    It is looser than the identities (``2 * eps``, an approximation gap, against
    their exact ``1e-11``) and that is the trade: they are exact and
    self-referential, this one is approximate and independent, and neither
    substitutes for the other.

    ``divide_by_n`` is parametrised because it is the image-side diagonal (#20)
    and the only independent reference for the *gradient* of that diagonal is
    here -- with it pinned off, a fault in ``_disc_mask_and_safe_n`` reaches no
    new cell in this module.
    """
    uvw, image_np = _dft_geometry()
    plan = make_plan(
        uvw,
        _DFT_FREQ,
        (_DFT_N_PIX, _DFT_N_PIX),
        _DFT_PIXSIZE,
        _DFT_PIXSIZE,
        eps,
        hermitian=hermitian,
    )

    # Anti-vacuity, both of them measured failures of the first version of this
    # fixture rather than hypotheticals. Without the first, the plane count is
    # all kernel overhang and no w-direction error can show up at all; without
    # the second, every window holds every row and the four w_strategy values
    # are one computation under four names.
    inner = plan.n_w - plan.w_kernel_width
    assert inner > 1, (
        f"n_w={plan.n_w} with w_kernel_width={plan.w_kernel_width} leaves "
        f"{inner} inner plane(s): the w axis is kernel overhang only, and this "
        "cell has degenerated into a test of the 2-D NUFFT"
    )
    assert plan.max_window_size < plan.n_rows, (
        f"max_window_size is {plan.max_window_size} of {plan.n_rows} rows, so "
        "windowed_* and dense_* are the same computation and this cell's "
        "w_strategy axis is quantified over a constant"
    )

    kw = dict(divide_by_n=divide_by_n, w_strategy=w_strategy, nthreads=1)
    if complex_image:
        x = jnp.asarray(image_np + 0.5j * image_np[::-1]).astype(plan.complex_dtype)
    else:
        x = jnp.asarray(image_np).astype(plan.real_dtype)

    rng = np.random.default_rng(17)
    ct = jnp.asarray(
        rng.standard_normal((_DFT_N_ROWS, 1)) + 1j * rng.standard_normal((_DFT_N_ROWS, 1))
    ).astype(plan.complex_dtype)

    _, vjp_op = jax.vjp(lambda z: dirty2vis(plan, z, **kw), x)
    _, vjp_dft = jax.vjp(lambda z: _dft_forward_jnp(z, uvw, divide_by_n=divide_by_n), x)
    got = vjp_op(ct)[0]
    want = vjp_dft(ct)[0]

    err = _rel(got, want)
    assert err < DFT_GRAD_TOL_FACTOR * eps, (
        f"{w_strategy}, complex_image={complex_image}, hermitian={hermitian}, "
        f"divide_by_n={divide_by_n}, eps={eps:.0e}: the wgridder cotangent "
        f"differs by {err:.3e} ({err / eps:.2f}x eps, bound "
        f"{DFT_GRAD_TOL_FACTOR:.1f}x) from jax.vjp of the exact DFT. Unlike the "
        "A^H A identities above, this reference shares no code with the operator, "
        "so a forward and adjoint wrong in the same way fail here and nowhere else."
    )


@requires_x64
@pytest.mark.parametrize("geometry", _ANISO_GEOMETRIES)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_the_cotangent_is_indexed_l_by_m_and_not_m_by_l(op: str, geometry: dict) -> None:
    """The backward pass on a grid that is *not* transpose-symmetric.

    Every telescope fixture in this repository is square with isotropic pixels,
    and on such a plan ``plan.n_minus_1`` is transpose-symmetric bit for bit --
    so a backward pass that transposed its image-side diagonal would be a
    literal no-op everywhere else in this module. ``_ANISO_GEOMETRIES``
    (borrowed from ``test_divide_by_n.py``, which established the same blind
    spot for the primal) is the minimum that falsifies it: one square grid with
    anisotropic pixels, one non-square grid.

    ``divide_by_n=True`` on both sides, since that is the flag that puts a
    geometry-indexed diagonal into the operator at all.
    """
    exact_tol = IDENTITY_TOL_F64
    problem = _problem(EDA2, 0.0, eps=1e-6, dtype=jnp.float64, **geometry)
    kw = dict(divide_by_n=True, w_strategy="dense_scan", nthreads=1)
    if op == "dirty2vis":
        image = jnp.asarray(problem.image)
        y = jnp.asarray(problem.vis)
        _, vjp_fn = jax.vjp(lambda im: dirty2vis(problem.plan, im, **kw), image)
        got = vjp_fn(y)[0]
        want = vis2dirty(problem.plan, jnp.conj(y), **kw)[0]
    else:
        vis = jnp.asarray(problem.vis)
        cotangent = jnp.asarray(problem.image)[None]
        _, vjp_fn = jax.vjp(lambda v: vis2dirty(problem.plan, v, **kw), vis)
        got = vjp_fn(cotangent)[0]
        want = jnp.conj(dirty2vis(problem.plan, cotangent[0], **kw))

    err = _rel(got, want)
    assert err < exact_tol, (
        f"{op} on {problem.shape} pixels of {problem.pixsize_l:.4g} x "
        f"{problem.pixsize_m:.4g}: the reverse-mode cotangent differs by {err:.3e} "
        f"(tol {exact_tol:.1e}) from the adjoint of the same operator. On a square "
        "isotropic grid n is transpose-symmetric bit-for-bit, so an l/m axis swap in "
        "the backward pass is invisible; this geometry is what makes it visible."
    )


# ---------------------------------------------------------------------------
# 3. the transforms that must keep working
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_forward_mode_jvp_is_the_operator_applied_to_the_tangent(
    w_strategy: str, op: str, dtype: DTypeLike, eps: float, exact_tol: float
) -> None:
    """``jvp(A, x, t) == (A x, A t)`` -- both operators are linear.

    Forward mode is not a nicety here, it is the discriminator between the two
    implementation routes issue #21 offers. ``jax.custom_vjp`` supplies a
    reverse rule and *removes* forward mode: ``jax.jvp`` through a
    ``custom_vjp`` function raises ``TypeError: can't apply forward-mode
    autodiff (jvp) to a custom_vjp function``. Both operators support
    ``jax.jvp`` today, and the issue's own definition of done asks for
    ``check_grads(modes=("fwd", "rev"))`` and a finite ``jax.hessian``, so the
    plain ``custom_vjp`` of the sketch cannot satisfy it. The linear-primitive
    route (``ad.primitive_transposes``, which the issue calls "preferred") does:
    the JVP of a linear map is the map itself.

    Asserted as an exact identity rather than against finite differences: the
    tangent of a linear operator *is* the operator, and it was measured
    bit-for-bit equal (0.0) on the prototype.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype)
    kw = dict(divide_by_n=True, w_strategy=w_strategy, nthreads=1)
    rng = np.random.default_rng(210)
    if op == "dirty2vis":
        fn = lambda x: dirty2vis(problem.plan, x, **kw)  # noqa: E731
        primal = jnp.asarray(problem.image)
        tangent = jnp.asarray(
            rng.standard_normal(problem.image.shape).astype(problem.plan.real_dtype)
        )
    else:
        fn = lambda x: vis2dirty(problem.plan, x, **kw)  # noqa: E731
        primal = jnp.asarray(problem.vis)
        tangent = jnp.asarray(
            (
                rng.standard_normal(problem.vis.shape) + 1j * rng.standard_normal(problem.vis.shape)
            ).astype(problem.plan.complex_dtype)
        )

    out, tangent_out = jax.jvp(fn, (primal,), (tangent,))
    # Relative L2, not elementwise: computing the primal alongside the tangent
    # changes XLA's fusion, and ``vis2dirty``'s scatter-add is not bit-stable
    # under that (the same reason test_jax_integration compares it with
    # allclose rather than array_equal). Measured worst case 1.5e-06 in
    # float32, 6e-16 in float64.
    err_primal = _rel(out, fn(primal))
    assert err_primal < exact_tol, (
        f"{op} ({w_strategy}): jax.jvp's primal output differs by {err_primal:.3e} "
        f"(tol {exact_tol:.1e}) from the same call made on its own"
    )
    err = _rel(tangent_out, fn(tangent))
    assert err < exact_tol, (
        f"{op} ({w_strategy}): the forward-mode tangent differs by {err:.3e} "
        f"(tol {exact_tol:.1e}) from the operator applied to the tangent. Both "
        "operators are linear in their non-plan argument, so their JVP is "
        "themselves."
    )


@requires_x64
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_check_grads_passes_in_both_modes(op: str, w_strategy: str) -> None:
    """``jax.test_util.check_grads(modes=("fwd", "rev"), order=1)``.

    Issue #21's definition of done, verbatim. It is a numerical-differentiation
    check, so it runs on the float64 leg only; the exact-identity tests above
    carry the float32 leg.
    """
    problem = _problem(EDA2, 0.0, eps=1e-6, dtype=jnp.float64)
    kw = dict(divide_by_n=True, w_strategy=w_strategy, nthreads=1)
    if op == "dirty2vis":
        fn = lambda x: dirty2vis(problem.plan, x, **kw)  # noqa: E731
        args = (jnp.asarray(problem.image),)
    else:
        fn = lambda x: vis2dirty(problem.plan, x, **kw)  # noqa: E731
        args = (jnp.asarray(problem.vis),)
    check_grads(fn, args, order=1, modes=("fwd", "rev"), eps=1e-4, rtol=2e-4, atol=2e-4)


# ``check_grads`` differentiates numerically, so every cell below is float64
# only. The axes here are the ones the cell above holds constant: it runs a real
# image on a folded plan with ``divide_by_n=True`` and no weights, which is one
# corner of the operators' input space.


@requires_x64
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
@pytest.mark.parametrize("hermitian", [False, True])
@pytest.mark.parametrize("divide_by_n", FLAG_VALUES)
def test_check_grads_on_the_flags_the_default_cell_holds_constant(
    divide_by_n: bool, hermitian: bool, w_strategy: str
) -> None:
    """``check_grads`` across ``divide_by_n`` x ``hermitian``, both operators.

    ``test_check_grads_passes_in_both_modes`` pins one corner: ``divide_by_n``
    true on a folded plan. The four cells here that repeat that corner are left
    in rather than special-cased out -- the grid is cheap and a hole in it would
    be worse than an overlap. Both flags change the operator rather than the way
    it is traversed -- ``divide_by_n`` multiplies by a real image-side diagonal
    (#20) and ``hermitian`` folds the w axis and multiplies rows by
    ``flip_sign`` (#17) -- so each is a distinct linear map whose transpose has
    to be right on its own account.
    """
    problem = _problem(EDA2, 0.0, eps=1e-6, dtype=jnp.float64, hermitian=hermitian)
    kw = dict(divide_by_n=divide_by_n, w_strategy=w_strategy, nthreads=1)
    for fn, args in (
        (lambda x: dirty2vis(problem.plan, x, **kw), (jnp.asarray(problem.image),)),
        (lambda v: vis2dirty(problem.plan, v, **kw), (jnp.asarray(problem.vis),)),
    ):
        check_grads(fn, args, order=1, modes=("fwd", "rev"), eps=1e-4, rtol=2e-4, atol=2e-4)


@requires_x64
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_check_grads_on_a_complex_image(w_strategy: str) -> None:
    """``check_grads`` for ``dirty2vis`` on a **complex** image.

    The forward is C-linear in the image but the cotangent rule differs between
    the real and complex cases -- ``A^T y`` against ``conj(A^T y)`` -- and they
    agree on a real image because ``Re(conj z) == Re z``. So a numerical check
    on a real image cannot see the convention at all; this one can.

    ``hermitian=False``: the fold is a real-sky identity and ``dirty2vis``
    refuses a complex image on a folded plan (#17), so this is the only setting
    the complex leg can be asked at.
    """
    problem = _problem(EDA2, 0.0, eps=1e-6, dtype=jnp.float64, hermitian=False)
    kw = dict(divide_by_n=True, w_strategy=w_strategy, nthreads=1)
    image = jnp.asarray(problem.image) + 0.5j * jnp.asarray(problem.image)[::-1]
    fn = lambda x: dirty2vis(problem.plan, x, **kw)  # noqa: E731
    check_grads(fn, (image,), order=1, modes=("fwd", "rev"), eps=1e-4, rtol=2e-4, atol=2e-4)


@requires_x64
@pytest.mark.parametrize("argnum", [0, 1])
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_check_grads_with_weights(w_strategy: str, argnum: int) -> None:
    """``check_grads`` for ``vis2dirty`` with weights, in ``vis`` and in ``weights``.

    #21 moved the ``weights`` multiply out of the primitive and into the Python
    wrapper, so the two arguments reach the backward pass by different routes:
    ``vis`` through the primitive's transpose rule, ``weights`` through XLA's
    own differentiation of the multiply. ``argnum=1`` is the only cell in this
    module that exercises the second route numerically.
    """
    problem = _problem(EDA2, 0.0, eps=1e-6, dtype=jnp.float64)
    kw = dict(divide_by_n=True, w_strategy=w_strategy, nthreads=1)
    vis = jnp.asarray(problem.vis)
    rng = np.random.default_rng(11)
    weights = jnp.asarray(rng.uniform(0.25, 4.0, size=vis.shape).astype(problem.plan.real_dtype))
    if argnum == 0:
        fn = lambda v: vis2dirty(problem.plan, v, weights=weights, **kw)  # noqa: E731
        args = (vis,)
    else:
        fn = lambda w: vis2dirty(problem.plan, vis, weights=w, **kw)  # noqa: E731
        args = (weights,)
    check_grads(fn, args, order=1, modes=("fwd", "rev"), eps=1e-4, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_linear_transpose_agrees_with_the_reverse_mode_cotangent(
    op: str, dtype: DTypeLike, eps: float, exact_tol: float
) -> None:
    """``jax.linear_transpose`` must work and must agree with ``jax.vjp``.

    This does **not** work at ``HEAD``: the channel loop scans over the image,
    and transposing a ``scan`` whose ``xs`` is the undefined primal raises
    ``TypeError: Error interpreting argument to scan as a JAX value ... type
    ValAccum``. So it is a new capability rather than a regression guard, and
    it is here because it is the cheapest test that distinguishes an operator
    JAX *knows* is linear from one it merely differentiates: ``custom_vjp``
    keeps ``jax.grad`` working and leaves ``linear_transpose`` failing exactly
    as it fails today.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype)
    kw = dict(divide_by_n=True, w_strategy="dense_scan", nthreads=1)
    if op == "dirty2vis":
        fn = lambda x: dirty2vis(problem.plan, x, **kw)  # noqa: E731
        primal = jnp.asarray(problem.image)
        cotangent = jnp.asarray(problem.vis)
    else:
        fn = lambda x: vis2dirty(problem.plan, x, **kw)  # noqa: E731
        primal = jnp.asarray(problem.vis)
        cotangent = jnp.asarray(problem.image)[None].astype(problem.plan.real_dtype)

    _, vjp_fn = jax.vjp(fn, primal)
    from_vjp = vjp_fn(cotangent)[0]
    from_transpose = jax.linear_transpose(fn, primal)(cotangent)[0]
    err = _rel(from_transpose, from_vjp)
    assert err < exact_tol, (
        f"{op}: jax.linear_transpose differs by {err:.3e} (tol {exact_tol:.1e}) "
        "from jax.vjp. Both are the transpose of the same linear operator."
    )


@requires_x64
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_higher_order_autodiff_stays_finite(op: str) -> None:
    """``jax.hessian`` of a tiny problem: finite, symmetric, and non-zero.

    Issue #21's definition of done again. The Hessian of ``0.5 ||A x||^2`` is
    ``Re(A^H A)`` -- constant, symmetric and positive semi-definite -- so this
    also pins that the second derivative did not quietly become zero, which is
    what a backward pass with no rule of its own would produce.

    ``jax.hessian`` is ``jacfwd(jacrev(...))``: it needs a JVP rule for
    whatever the reverse pass emitted. A ``custom_vjp`` has none, so this test
    is the second of the two that route cannot pass.
    """
    dtype = jnp.float64
    problem = _problem(EDA2, 0.0, eps=1e-4, dtype=dtype, shape=(8, 8))
    kw = dict(divide_by_n=True, w_strategy="dense_scan", nthreads=1)
    if op == "dirty2vis":

        def loss(x: Any) -> Any:
            return 0.5 * jnp.sum(jnp.abs(dirty2vis(problem.plan, x, **kw)) ** 2)

        arg = jnp.asarray(problem.image)
    else:

        def loss(x: Any) -> Any:
            return 0.5 * jnp.sum(vis2dirty(problem.plan, x, **kw) ** 2)

        arg = jnp.asarray(problem.vis).real

    hessian = np.asarray(jax.hessian(loss)(arg), dtype=np.float64)
    assert np.all(np.isfinite(hessian)), f"{op}: jax.hessian produced non-finite entries"
    n = int(np.prod(arg.shape))
    flat = hessian.reshape(n, n)
    assert np.linalg.norm(flat) > 0.0, (
        f"{op}: the Hessian of 0.5||A x||^2 is identically zero, which means the "
        "second derivative was lost rather than computed"
    )
    asymmetry = np.abs(flat - flat.T).max() / np.abs(flat).max()
    assert asymmetry < 1e-9, (
        f"{op}: the Hessian of a quadratic is Re(A^H A) and must be symmetric; "
        f"measured asymmetry {asymmetry:.3e}"
    )


def _dense_operator(plan: Any, shape: tuple[int, int], **kw: Any) -> np.ndarray:
    """``A`` as a dense ``(n_rows * n_chan, n_pix)`` matrix, from ``dirty2vis``.

    Column ``j`` is the operator applied to the ``j``-th unit image, so the
    matrix is built from the *public* forward and carries no assumption about
    how the gradient is implemented. The module docstring's four conventions
    were established the same way.
    """
    n_pix = int(np.prod(shape))
    columns = []
    for j in range(n_pix):
        basis = np.zeros(n_pix, dtype=np.float64)
        basis[j] = 1.0
        column = dirty2vis(plan, jnp.asarray(basis.reshape(shape), dtype=plan.real_dtype), **kw)
        columns.append(np.asarray(column, dtype=np.complex128).ravel())
    return np.stack(columns, axis=1)


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_second_order_reverse_mode_is_the_operator_again(
    op: str, dtype: DTypeLike, eps: float, exact_tol: float
) -> None:
    """``grad(grad(...))``: the transpose of the transpose is the operator.

    The one transform that reaches the *transpose of the transpose rule*, and
    the reason this is a separate test from the Hessian one above.
    ``jax.hessian`` is ``jacfwd(jacrev(...))``, so its outer pass needs a JVP
    rule for what the inner reverse pass emitted and never transposes it; a
    second **reverse** pass does. Instrumenting the operators over the whole
    suite records zero calls to that rule without this test, and replacing its
    body with ``raise NotImplementedError`` leaves all 1448 fast tests green --
    while the ``dirty2vis`` leg here raises immediately. (The ``vis2dirty`` leg
    is the symmetric second-order check rather than a second cover of that
    rule: its two reverse passes go through the *other* two transposes, and it
    survives that mutation. Both are kept, since "which rule does a given leg
    reach" is exactly the thing a later change can move.)

    The scalar parametrisation is what forces the second pass through the
    operator rather than through the loss: ``f(s) = 0.5 ||A (s x)||^2`` is
    ``0.5 s^2 ||A x||^2``, so ``f''(s) = ||A x||^2`` exactly, and the reference
    is that norm taken on a **dense** ``A`` built column by column from the
    public forward. A rule that transposed to the wrong member of the pair --
    the plausible mistake, the two being adjacent lines -- is refused at
    abstract evaluation, the shape it produces being the opposite end of the
    operator.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype, shape=(8, 8))
    kw = dict(divide_by_n=True, w_strategy="dense_scan", nthreads=1)
    dense = _dense_operator(problem.plan, (8, 8), **kw)

    if op == "dirty2vis":
        image = jnp.asarray(problem.image)

        def f(s: Any) -> Any:
            return 0.5 * jnp.sum(jnp.abs(dirty2vis(problem.plan, s * image, **kw)) ** 2)

        want = float(np.linalg.norm(dense @ np.asarray(image, dtype=np.float64).ravel()) ** 2)
    else:
        vis = jnp.asarray(problem.vis)

        def f(s: Any) -> Any:
            return 0.5 * jnp.sum(vis2dirty(problem.plan, s * vis, **kw) ** 2)

        adjoint = np.real(dense.conj().T @ np.asarray(vis, dtype=np.complex128).ravel())
        want = float(np.linalg.norm(adjoint) ** 2)

    got = float(jax.grad(jax.grad(f))(jnp.asarray(1.0, dtype=problem.plan.real_dtype)))
    err = abs(got - want) / abs(want)
    assert err < exact_tol, (
        f"{op}: the second derivative of 0.5||A (s x)||^2 in s is {got:.6e}, "
        f"{err:.3e} away (tol {exact_tol:.1e}) from the ||A x||^2 = {want:.6e} a dense "
        "A gives. Both reverse passes are transposes of this operator; the second one "
        "transposes the first one's rule."
    )


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
def test_the_weights_gradient_survives_the_primitive_boundary(
    dtype: DTypeLike, eps: float, exact_tol: float
) -> None:
    """``grad`` with respect to ``vis2dirty``'s ``weights``.

    #21 moved the ``weights`` multiply out of the operator body and into the
    JIT wrapper, on the grounds that a visibility-side diagonal composes on the
    outside and so keeps its gradient "for free" through the multiply. That is
    a claim about a code path nothing exercised: wrapping ``weights`` in
    ``jax.lax.stop_gradient`` inside the wrapper passes all 1448 fast tests.

    The reference is the derivative written out by hand from the two public
    operators. With ``g = Re(A^H (v * w))`` and ``L(w) = sum g^2``,
    ``dL/dw_j = 2 Re(v_j conj((A g)_j))`` -- one adjoint and one forward call,
    neither of which is the code under test.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype)
    kw = dict(divide_by_n=True, w_strategy="dense_scan", nthreads=1)
    vis = jnp.asarray(problem.vis)
    weights = jnp.asarray(
        np.random.default_rng(2121)
        .uniform(0.3, 2.0, size=problem.vis.shape)
        .astype(problem.plan.real_dtype)
    )

    def loss(w: Any) -> Any:
        return jnp.sum(vis2dirty(problem.plan, vis, weights=w, **kw) ** 2)

    got = jax.grad(loss)(weights)
    gridded = vis2dirty(problem.plan, vis * weights.astype(vis.dtype), **kw)
    want = 2.0 * jnp.real(jnp.conj(dirty2vis(problem.plan, gridded, **kw)) * vis)

    err = _rel(got, want)
    assert err < exact_tol, (
        f"the gradient with respect to weights differs by {err:.3e} (tol "
        f"{exact_tol:.1e}) from 2 Re(v conj(A g)) with g = vis2dirty(v * w). The "
        "weights multiply sits outside the primitive precisely so that this works "
        "without a rule of its own."
    )


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_vmap_over_a_batch_of_gradients(
    op: str, dtype: DTypeLike, eps: float, exact_tol: float
) -> None:
    """``jax.vmap`` over a batch, of the operator and of its gradient.

    ``tests/test_jax_integration.py`` already covers ``vmap`` of the primal;
    what is new after #21 is that the *backward* pass has to carry a batching
    rule of its own. Both are asserted against an explicit Python loop, so a
    batching rule that silently transposed or broadcast the batch axis is
    caught rather than merely "not crashing".
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype)
    kw = dict(divide_by_n=True, w_strategy="dense_scan", nthreads=1)
    rng = np.random.default_rng(2100)
    batch = 3
    if op == "dirty2vis":
        fn = lambda x: dirty2vis(problem.plan, x, **kw)  # noqa: E731
        xs = jnp.asarray(
            rng.standard_normal((batch, *problem.image.shape)).astype(problem.plan.real_dtype)
        )
    else:
        fn = lambda x: vis2dirty(problem.plan, x, **kw)  # noqa: E731
        xs = jnp.asarray(
            (
                rng.standard_normal((batch, *problem.vis.shape))
                + 1j * rng.standard_normal((batch, *problem.vis.shape))
            ).astype(problem.plan.complex_dtype)
        )

    def scalar_loss(x: Any) -> Any:
        out = fn(x)
        return jnp.sum(jnp.abs(out) ** 2) if jnp.iscomplexobj(out) else jnp.sum(out**2)

    batched = jax.vmap(fn)(xs)
    looped = jnp.stack([fn(x) for x in xs])
    assert _rel(batched, looped) < exact_tol, f"{op}: vmap of the primal disagrees with a loop"

    batched_grad = jax.vmap(jax.grad(scalar_loss))(xs)
    looped_grad = jnp.stack([jax.grad(scalar_loss)(x) for x in xs])
    err = _rel(batched_grad, looped_grad)
    assert err < exact_tol, (
        f"{op}: vmap of the gradient differs by {err:.3e} (tol {exact_tol:.1e}) from "
        "the same gradients computed one at a time"
    )


def _walk_eqns(jaxpr: Any) -> Any:
    """Every equation of a jaxpr, including those inside its sub-jaxprs."""
    for eqn in jaxpr.eqns:
        yield eqn
        for param in eqn.params.values():
            items = param if isinstance(param, (tuple, list)) else [param]
            for item in items:
                if isinstance(item, jex.core.ClosedJaxpr):
                    yield from _walk_eqns(item.jaxpr)
                elif isinstance(item, jex.core.Jaxpr):
                    yield from _walk_eqns(item)


_PRIMITIVE_NAME = {
    "dirty2vis": "jax_nufft_dirty2vis",
    "vis2dirty": "jax_nufft_vis2dirty",
}


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_vmap_binds_the_operator_once_rather_than_looping(op: str) -> None:
    """``vmap`` must stay a *batched* call, not become ``B`` serial ones.

    The test above compares ``vmap`` against a Python loop, so it passes just
    as happily if the implementation *is* a loop. That is not a hypothetical:
    the obvious batching rule for a primitive is ``jax.lax.map`` over the batch
    axis, it is correct, and it costs 1.5x at ``B = 2`` rising to 3.7x at
    ``B = 16`` (EDA2 zenith 64^2, ``dense_scan``, median of 9) because the
    batch then buys nothing. Before #21 JAX batched these operators natively
    and no test would have seen that go away.

    So the property is asserted where it is exact rather than by timing: the
    lowered jaxpr must contain the operator's primitive **once**, carrying the
    batch in its ``batch_shape`` param, rather than once per element inside a
    ``scan``. That is the same statement as "the batch is one call" and it does
    not depend on the machine.
    """
    dtype = _active_dtype()
    problem = _problem(EDA2, 0.0, eps=tol(1e-6, 1e-5), dtype=dtype)
    kw = dict(divide_by_n=True, w_strategy="dense_scan", nthreads=1)
    batch = 3
    if op == "dirty2vis":
        fn = lambda x: dirty2vis(problem.plan, x, **kw)  # noqa: E731
        xs = jnp.zeros((batch, *problem.image.shape), dtype=problem.plan.real_dtype)
    else:
        fn = lambda x: vis2dirty(problem.plan, x, **kw)  # noqa: E731
        xs = jnp.zeros((batch, *problem.vis.shape), dtype=problem.plan.complex_dtype)

    jaxpr = jax.make_jaxpr(jax.vmap(fn))(xs).jaxpr
    binds = [eq for eq in _walk_eqns(jaxpr) if eq.primitive.name == _PRIMITIVE_NAME[op]]
    assert len(binds) == 1, (
        f"{op}: vmap over a batch of {batch} lowered to {len(binds)} binds of "
        f"{_PRIMITIVE_NAME[op]}; a batched call is one bind"
    )
    assert binds[0].params["batch_shape"] == (batch,), (
        f"{op}: the single bind carries batch_shape="
        f"{binds[0].params['batch_shape']!r}, so the batch axis went somewhere other "
        "than into the operator -- a per-element loop rather than a batched call"
    )


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_the_gradient_of_a_batched_loss_is_the_batched_transpose(
    op: str, dtype: DTypeLike, eps: float, exact_tol: float
) -> None:
    """``grad`` of a loss that ``vmap``s the operator -- mini-batch imaging.

    Every other ``vmap`` test in this repository is ``vmap(grad(...))``, with
    the batching *outside* the differentiation; the transpose rules then run on
    batch tracers and their own params are empty. A census over the float64
    suite makes that concrete: 174 transpose binds, every one of them with
    ``batch_shape=()``. The opposite ordering -- ``grad`` of something that
    contains a ``vmap`` of the operator, which is what
    ``sum_b ||A x_b - v_b||^2`` is -- is the only one that puts a batch into a
    transpose rule's params, and it was reached by nothing. Re-binding with
    ``batch_shape=()`` instead of ``**params`` in any of the three rules passes
    the rest of the suite and fails here, with the operand shape and the plan's
    shape in the message.

    The batch is **nested and unequal** (2, 5) for a second reason: nested
    ``vmap`` was otherwise reached only at inner size 1 (via ``jax.hessian``),
    so the leading-axis flattening in ``_apply_batched`` never ran on a
    non-degenerate nest. At (2, 5) a swapped reshape is a shape error, and any
    permutation of the ten elements is a value error against the loop below.

    The reference is an explicit Python loop of ten independent gradients, so
    nothing about the batched path is assumed.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype)
    kw = dict(divide_by_n=True, w_strategy="dense_scan", nthreads=1)
    outer, inner = 2, 5
    rng = np.random.default_rng(2105)
    if op == "dirty2vis":
        fn = lambda x: dirty2vis(problem.plan, x, **kw)  # noqa: E731
        xs = jnp.asarray(
            rng.standard_normal((outer, inner, *problem.image.shape)).astype(
                problem.plan.real_dtype
            )
        )
    else:
        fn = lambda x: vis2dirty(problem.plan, x, **kw)  # noqa: E731
        shape = (outer, inner, *problem.vis.shape)
        xs = jnp.asarray(
            (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
                problem.plan.complex_dtype
            )
        )

    def element_loss(x: Any) -> Any:
        out = fn(x)
        return jnp.sum(jnp.abs(out) ** 2) if jnp.iscomplexobj(out) else jnp.sum(out**2)

    def batched_loss(batch: Any) -> Any:
        return jnp.sum(jax.vmap(jax.vmap(element_loss))(batch))

    got = jax.grad(batched_loss)(xs)
    want = jnp.stack([jnp.stack([jax.grad(element_loss)(x) for x in row]) for row in xs])

    assert got.shape == xs.shape, (
        f"{op}: the gradient of a (2, 5)-batched loss came back shaped {got.shape} "
        f"against an input of {xs.shape}"
    )
    err = _rel(got, want)
    assert err < exact_tol, (
        f"{op}: the gradient of a loss over a nested (2, 5) batch differs by "
        f"{err:.3e} (tol {exact_tol:.1e}) from the same ten gradients taken one at a "
        "time. The backward pass of a batched call is the batched transpose -- one "
        "adjoint call over the whole batch, at the forward's own batch_shape."
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_the_operators_work_under_disable_jit(op: str) -> None:
    """``jax.disable_jit()``: the one public route to the primitives' ``def_impl``.

    Before #21 this came for free -- the operators were ordinary traced Python,
    so disabling ``jit`` just ran them op by op. A primitive without an
    evaluation rule raises ``NotImplementedError: Evaluation rule for
    'jax_nufft_dirty2vis' not implemented`` instead, and deleting all three
    ``def_impl`` registrations leaves the rest of the suite green. Debugging
    with ``disable_jit`` is a normal thing to do to a library like this one, so
    it is gated rather than left to the reader.
    """
    dtype = _active_dtype()
    problem = _problem(EDA2, 0.0, eps=tol(1e-6, 1e-5), dtype=dtype)
    kw = dict(divide_by_n=True, w_strategy="dense_scan", nthreads=1)
    arg = jnp.asarray(problem.image if op == "dirty2vis" else problem.vis)
    fn = dirty2vis if op == "dirty2vis" else vis2dirty

    want = fn(problem.plan, arg, **kw)
    with jax.disable_jit():
        got = fn(problem.plan, arg, **kw)

    # ``dtype`` is the run's own precision here, so the bound follows it: the
    # two calls are the same arithmetic in a different execution mode, and were
    # measured exactly equal.
    bound = tol(IDENTITY_TOL_F64, 5e-6)
    err = _rel(got, want)
    assert err < bound, (
        f"{op} under jax.disable_jit() differs by {err:.3e} (tol {bound:.1e}) from the jitted call"
    )


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_the_gradient_survives_being_jitted_from_either_side(
    op: str, dtype: DTypeLike, eps: float, exact_tol: float
) -> None:
    """The gradient must be the same with ``jax.jit`` outside it and inside it.

    Two shapes, because they stress different halves of the wiring.

    ``jax.jit(jax.grad(...))`` with the **plan as an argument** hands the rule a
    plan whose leaves are tracers rather than concrete arrays -- a rule that
    captured a concrete plan behaves differently there.

    ``jax.grad`` of a function that contains an inner ``jax.jit`` of the
    operator is the shape that actually catches it: the issue's sketch builds
    its ``custom_vjp`` inside the public wrapper and closes over everything the
    wrapper computed, including ``vis2dirty``'s placeholder ``weights`` array,
    which under an inner jit is a tracer. Measured on the sketch: ``TypeError:
    No constant handler for type: DynamicJaxprTracer``, on ``vis2dirty`` only.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype)
    kw = dict(divide_by_n=True, w_strategy="dense_scan", nthreads=1)
    if op == "dirty2vis":

        def scalar(plan: Any, x: Any) -> Any:
            return jnp.sum(jnp.abs(dirty2vis(plan, x, **kw)) ** 2)

        arg = jnp.asarray(problem.image)
    else:

        def scalar(plan: Any, x: Any) -> Any:
            return jnp.sum(vis2dirty(plan, x, **kw) ** 2)

        arg = jnp.asarray(problem.vis).real

    eager = jax.grad(scalar, argnums=1)(problem.plan, arg)

    jitted_outside = jax.jit(jax.grad(scalar, argnums=1))(problem.plan, arg)
    err = _rel(jitted_outside, eager)
    assert err < exact_tol, (
        f"{op}: the gradient under an outer jit (plan arriving as a tracer) differs "
        f"by {err:.3e} from the eager one"
    )

    inner = jax.jit(lambda x: scalar(problem.plan, x))
    jitted_inside = jax.grad(inner)(arg)
    err = _rel(jitted_inside, eager)
    assert err < exact_tol, (
        f"{op}: differentiating through an inner jax.jit of the operator gives a "
        f"gradient {err:.3e} away from the eager one"
    )


# ---------------------------------------------------------------------------
# 4. the backward pass runs the forward's static configuration
# ---------------------------------------------------------------------------


_CHANNEL_HELPERS = (
    "_channel_forward",
    "_channel_forward_windowed",
    "_channel_adjoint",
    "_channel_adjoint_windowed",
)


def _spy_on_channel_helpers(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[tuple]]:
    """Record ``(w_strategy, nthreads, channel-loop transform)`` per helper call.

    The four per-channel helpers are where a strategy actually becomes work,
    and each takes ``(..., plan, opts, w_strategy)`` positionally, so this reads
    the configuration the *backward* pass runs with without depending on how
    #21 is wired. The third field is the type name of the traced image /
    visibility argument: ``BatchTracer`` under ``channel_strategy="vmap"``,
    ``DynamicJaxprTracer`` under ``"scan"`` -- which is how the channel loop's
    shape becomes observable at all (it is not one of the helper's arguments).
    """
    calls: dict[str, list[tuple]] = {name: [] for name in _CHANNEL_HELPERS}

    def wrap(name: str) -> Any:
        original = getattr(wgridder, name)

        def spy(*args: Any, **kwargs: Any) -> Any:
            calls[name].append((args[-1], args[-2].nthreads, type(args[0]).__name__))
            return original(*args, **kwargs)

        return spy

    for name in _CHANNEL_HELPERS:
        monkeypatch.setattr(wgridder, name, wrap(name))
    # The public operators are jitted, so a warm cache would skip the tracing
    # this spy observes.
    jax.clear_caches()
    return calls


_FORWARD_HELPER = {
    "dense_scan": "_channel_forward",
    "dense_vmap": "_channel_forward",
    "windowed_scan": "_channel_forward_windowed",
    "windowed_vmap": "_channel_forward_windowed",
}
_ADJOINT_HELPER = {
    "dense_scan": "_channel_adjoint",
    "dense_vmap": "_channel_adjoint",
    "windowed_scan": "_channel_adjoint_windowed",
    "windowed_vmap": "_channel_adjoint_windowed",
}


@pytest.mark.parametrize("channel_strategy", CHANNEL_STRATEGIES)
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_the_backward_pass_runs_the_forwards_static_configuration(
    monkeypatch: pytest.MonkeyPatch, w_strategy: str, channel_strategy: str
) -> None:
    """The backward of ``dirty2vis`` is the adjoint *at the same settings*.

    A backward pass that quietly ran ``dense_scan`` while the forward ran
    ``windowed_vmap`` would still return the right numbers -- the four
    strategies agree to 1e-11 (AGENTS.md sec 6) -- and would have entirely the
    wrong memory, which is the only thing issue #21 is buying. So the static
    configuration has to be observed directly rather than inferred from a
    value.

    Three things are pinned: the strategy *family* (which of the two adjoint
    helpers runs), the ``w_strategy`` string it is handed, and the ``nthreads``
    inside its ``Opts``. ``nthreads=3`` is passed explicitly because the
    default resolves to ``1`` for the scan strategies (issue #24), so a
    backward that rebuilt its own options from the default would be
    indistinguishable at the default.

    At ``HEAD`` this test fails: reverse mode transposes the *forward's*
    w-plane loop, so no adjoint helper is called at all.
    """
    calls = _spy_on_channel_helpers(monkeypatch)
    dtype = _active_dtype()
    problem = _problem(EDA2, 0.0, eps=tol(1e-6, 1e-5), dtype=dtype, n_chan=2)
    image = jnp.asarray(problem.image)
    cotangent = jnp.asarray(problem.vis)

    _, vjp_fn = jax.vjp(
        lambda im: dirty2vis(
            problem.plan,
            im,
            divide_by_n=True,
            w_strategy=w_strategy,
            channel_strategy=channel_strategy,
            nthreads=3,
        ),
        image,
    )
    jax.block_until_ready(vjp_fn(cotangent))

    expected_tracer = "BatchTracer" if channel_strategy == "vmap" else "DynamicJaxprTracer"
    forward_helper = _FORWARD_HELPER[w_strategy]
    adjoint_helper = _ADJOINT_HELPER[w_strategy]
    other_adjoint = (set(_ADJOINT_HELPER.values()) - {adjoint_helper}).pop()

    assert calls[forward_helper], (
        f"the forward never reached {forward_helper} at w_strategy={w_strategy}; the "
        "spy is not seeing the operator it thinks it is"
    )
    assert calls[adjoint_helper], (
        f"the reverse-mode pass of dirty2vis({w_strategy}) never called "
        f"{adjoint_helper}. Issue #21's backward pass is one call to the adjoint "
        "operator at the forward's own settings -- a transposed replay of the "
        "forward's w-plane loop (what HEAD does) is what costs O(n_w * image)."
    )
    assert not calls[other_adjoint], (
        f"the backward pass of dirty2vis({w_strategy}) called {other_adjoint}, i.e. "
        f"the {'windowed' if 'windowed' in other_adjoint else 'dense'} traversal, "
        f"while the forward ran {w_strategy}. The numbers would still agree to "
        "1e-11 and the memory would not."
    )
    for got_strategy, got_nthreads, got_tracer in calls[adjoint_helper]:
        assert got_strategy == w_strategy, (
            f"the backward pass ran w_strategy={got_strategy!r} where the forward ran "
            f"{w_strategy!r}"
        )
        assert got_nthreads == 3, (
            f"the backward pass built Opts(nthreads={got_nthreads}) where the forward "
            "was given nthreads=3; the backward must close over the forward's static "
            "configuration, not re-resolve its own"
        )
        assert got_tracer == expected_tracer, (
            f"the backward pass ran its channel loop as "
            f"{'vmap' if got_tracer == 'BatchTracer' else 'scan'} where the forward ran "
            f"{channel_strategy}; channel vmap costs n_chan x the per-channel transient "
            "(AGENTS.md sec 5), so this is a memory bug that no value can see"
        )


@pytest.mark.parametrize("channel_strategy", CHANNEL_STRATEGIES)
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_the_backward_pass_of_vis2dirty_runs_the_forward_operator(
    monkeypatch: pytest.MonkeyPatch, w_strategy: str, channel_strategy: str
) -> None:
    """The mirror: reverse mode through ``vis2dirty`` runs ``dirty2vis``.

    Same reasoning and the same three assertions, with the roles of the two
    helper families swapped.
    """
    calls = _spy_on_channel_helpers(monkeypatch)
    dtype = _active_dtype()
    problem = _problem(EDA2, 0.0, eps=tol(1e-6, 1e-5), dtype=dtype, n_chan=2)
    vis = jnp.asarray(problem.vis)
    cotangent = jnp.asarray(problem.image).astype(problem.plan.real_dtype)

    _, vjp_fn = jax.vjp(
        lambda v: vis2dirty(
            problem.plan,
            v,
            divide_by_n=True,
            w_strategy=w_strategy,
            channel_strategy=channel_strategy,
            nthreads=3,
        ),
        vis,
    )
    jax.block_until_ready(vjp_fn(cotangent))

    expected_tracer = "BatchTracer" if channel_strategy == "vmap" else "DynamicJaxprTracer"
    forward_helper = _FORWARD_HELPER[w_strategy]
    other_forward = (set(_FORWARD_HELPER.values()) - {forward_helper}).pop()

    assert calls[_ADJOINT_HELPER[w_strategy]], "the primal adjoint never ran; spy is misplaced"
    assert calls[forward_helper], (
        f"the reverse-mode pass of vis2dirty({w_strategy}) never called "
        f"{forward_helper}. The transpose of Re(A^H .) is conj(A .), i.e. one call to "
        "the forward operator."
    )
    assert not calls[other_forward], (
        f"the backward pass of vis2dirty({w_strategy}) called {other_forward} rather "
        f"than the {w_strategy} traversal"
    )
    for got_strategy, got_nthreads, got_tracer in calls[forward_helper]:
        assert got_strategy == w_strategy, (
            f"the backward pass ran w_strategy={got_strategy!r} where the forward ran "
            f"{w_strategy!r}"
        )
        assert got_nthreads == 3, (
            f"the backward pass built Opts(nthreads={got_nthreads}) where the caller "
            "asked for nthreads=3"
        )
        assert got_tracer == expected_tracer, (
            f"the backward pass ran its channel loop as "
            f"{'vmap' if got_tracer == 'BatchTracer' else 'scan'} where the forward ran "
            f"{channel_strategy}"
        )


# ---------------------------------------------------------------------------
# 5. the n == 0 boundary, in the backward pass specifically
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype, eps, exact_tol", _IDENTITY_PRECISIONS)
@pytest.mark.parametrize("w_strategy", W_STRATEGIES)
def test_gradients_stay_finite_on_the_unit_circle(
    w_strategy: str, dtype: DTypeLike, eps: float, exact_tol: float
) -> None:
    """``n == 0`` exactly: the *backward* pass must divide by ``safe_n`` too.

    ``test_divide_by_n.py::test_pixels_exactly_on_the_unit_circle_stay_finite_
    in_value_and_gradient`` established this fixture and the failure mode: with
    ``safe_n`` replaced by the raw ``n``, every primal stays finite and only the
    gradient NaNs, because the untaken branch of a ``jnp.where`` still
    contributes in reverse mode. That mutation passed 1272 tests.

    After #21 the backward pass is separate code, so it can reintroduce exactly
    that defect on its own -- the primal path keeps its guard and the gradient
    loses it. Hence this is not the same test: it asserts on the gradients that
    #21's rule produces, over all four strategies, and it asserts the
    *value* of the forward cotangent against the adjoint as well, so a backward
    that "fixed" the NaN by zeroing the whole disc boundary is caught too.
    """
    problem = _problem(EDA2, 0.0, eps=eps, dtype=dtype, pixsize_l=_BOUNDARY_PIXSIZE)
    n_grid = _n_grid(problem.plan)
    on_circle = n_grid == 0.0
    assert int(on_circle.sum()) > 0, (
        f"this fixture has no pixel with n == 0 exactly (min|n| = "
        f"{np.abs(n_grid).min():.3e}); every assertion below is vacuous"
    )
    kw = dict(divide_by_n=True, w_strategy=w_strategy, nthreads=1)
    image = jnp.asarray(problem.image)
    vis = jnp.asarray(problem.vis)

    _, vjp_fwd = jax.vjp(lambda im: dirty2vis(problem.plan, im, **kw), image)
    cotangent_image = np.asarray(vjp_fwd(vis)[0], dtype=np.float64)
    _, vjp_adj = jax.vjp(lambda v: vis2dirty(problem.plan, v, **kw), vis)
    cotangent_vis = np.asarray(vjp_adj(jnp.asarray(problem.image)[None]), dtype=np.complex128)

    n_bad = int((~np.isfinite(cotangent_image)).sum()) + int((~np.isfinite(cotangent_vis)).sum())
    assert n_bad == 0, (
        f"{w_strategy}: {n_bad} non-finite entries in the reverse-mode cotangents on a "
        f"plan with {int(on_circle.sum())} pixels exactly on the unit circle, while "
        "both primals are finite. That is a division by an unguarded n in the "
        "BACKWARD pass -- either an unmasked 1/n written there, or the untaken branch "
        "of a jnp.where whose infinity reverse mode multiplies by zero. The backward "
        "must divide by _disc_mask_and_safe_n's safe_n, exactly as the primal does; "
        "writing its own 1/n is the one way this defect can appear on one side only."
    )
    assert float(np.linalg.norm(cotangent_image)) > 0.0
    # ... and the values are still the adjoint's, not merely finite.
    want = np.asarray(vis2dirty(problem.plan, jnp.conj(vis), **kw)[0], dtype=np.float64)
    err = _rel(cotangent_image, want)
    assert err < exact_tol, (
        f"{w_strategy}: on the disc-boundary fixture the forward's cotangent differs by "
        f"{err:.3e} from vis2dirty(conj(y)) -- finite, but not the adjoint"
    )
    np.testing.assert_array_equal(
        cotangent_image[on_circle],
        np.zeros(int(on_circle.sum())),
        err_msg=(
            "divide_by_n=True excludes the n == 0 pixels from the operator, so the "
            "gradient with respect to them is exactly zero; a non-zero entry means the "
            "backward pass used a mask of n >= 0"
        ),
    )
