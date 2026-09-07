"""Issue #25: ``w_strategy="chunked"`` with a static ``w_chunk`` -- the memory/compute knob.

**Status: this module is written against a feature that does not exist yet.**
Every test here fails at ``c803ac2``. They are the acceptance gates for issue
#25 and the implementing change is what turns them green; see "What fails
today, and how" at the foot of this docstring for the target list.

What the feature is
-------------------
A fifth (and sixth) canonical ``w_strategy``: pad the w-plane grid to a
multiple of a static ``w_chunk``, ``lax.scan`` over the chunks, and ``vmap``
over the ``w_chunk`` planes *inside* a chunk -- one FINUFFT plan and sort per
chunk with ``n_transf = w_chunk``. Transient memory is then ``~2 * w_chunk *
image`` instead of ``~2 * n_w * image``.

Why it is a *curve* and not a third point
-----------------------------------------
The point of the issue -- stated in the issue's own scope note -- is that
``w_chunk`` lets a caller choose where on the memory/compute curve to sit,
rather than picking between ``dense_scan`` (minimal memory, ~20x slower on
GPU) and ``dense_vmap`` (fastest, 30 GB at realistic size). Item 2 of the
implementation plan is what makes it a curve: ``dense_scan`` is
``chunked(w_chunk=1)`` and ``dense_vmap`` is ``chunked(w_chunk=n_w)``, with
the old names kept as aliases. So this module tests the *endpoints as
identities* (section 3) and the *interior as a monotone memory curve*
(section 4), not merely that one chunk size returns the right numbers.

Measurements this module's bounds are derived from
--------------------------------------------------
All taken on the review machine (macOS arm64, 10 cores, CPU backend, pixi env
``test``) at ``c803ac2``, float64, ``eps = 1e-6``, ``nthreads=1``,
``n_chan=1``, the shipped ``hermitian=True``, using AGENTS.md section 6's
memory protocol
(``jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes``).
``image`` below is ``n_l * n_m * itemsize(complex128)`` bytes.

=============================  =====  ===========  ==================  ====================  ====================
fixture                          n_w   image (B)    dense_scan fwd=adj  dense_vmap forward    dense_vmap adjoint
=============================  =====  ===========  ==================  ====================  ====================
EDA2 zenith 64^2 / 400 rows       11       65,536    137,608 (2.10 x)      791,296 (12.07 x)     791,296 (12.07 x)
MWA_compact off30 128^2 / 600     12      262,144    534,024 (2.04 x)    3,260,928 (12.44 x)   3,145,728 (12.00 x)
MWA_extended off30 256^2 / 600   134    1,048,576  2,106,888 (2.01 x)  141,795,584 (135.23 x) 281,018,368 (268.00 x)
MeerKAT off30 256^2 / 600         13    1,048,576  2,106,888 (2.01 x)   13,756,288 (13.12 x)  13,631,488 (13.00 x)
=============================  =====  ===========  ==================  ====================  ====================

Three things follow, and they are the whole basis of section 4:

1. **The definition-of-done's bare ``2 * w_chunk * image`` bound is not
   satisfiable at small ``w_chunk``.** ``dense_scan`` -- which is exactly what
   ``chunked(1)`` has to be -- already costs 2.01-2.10 x image, i.e. *above*
   ``2 * 1 * image``. The four rows above fit ``2 * image + 16 * n_rows + 136``
   to the byte, so the missing term is the row-sized and constant traffic that
   every strategy pays. This module therefore gates at
   ``2 * w_chunk * image + temp(dense_scan on the same plan and operator)``,
   with the scan floor *measured* on the fixture at run time rather than
   modelled, and separately checks that the bound so formed is still tighter
   than today's ``dense_vmap`` (otherwise it would gate nothing).
2. **``dense_vmap`` scales with ``n_w``, and that is the control.**
   MWA_extended off30 and MeerKAT off30 have the *same* 256^2 image and the
   *same* 600 rows and differ only in ``n_w`` (134 against 13, a 10.3x ratio).
   ``dense_vmap``'s temp moves by 10.31x on the forward and 20.62x on the
   adjoint across that pair. ``chunked`` at a fixed ``w_chunk`` must not:
   that pair, at fixed ``w_chunk``, is the direct statement of "scales with
   ``w_chunk``, not with ``n_w``" -- see
   ``test_temp_memory_tracks_w_chunk_not_n_w``.
3. **The end-to-end separation is enormous, so the monotone-curve gate has
   room.** ``dense_vmap / dense_scan`` on MWA_extended off30 is 67.3x
   (forward) and 133.4x (adjoint); a gate at 10x is far inside that.

Gradient memory, same conditions (``divide_by_n=True``):

===============================  ==============  =============  =============  ======
fixture                          strategy        forward        grad           ratio
===============================  ==============  =============  =============  ======
MWA_extended off30, dirty2vis    dense_scan        2,106,888      3,155,592     1.50x
MWA_extended off30, vis2dirty    dense_scan        2,106,888      2,126,152     1.01x
MWA_extended off30, dirty2vis    dense_vmap      141,795,584    281,018,368     1.98x
MWA_extended off30, vis2dirty    dense_vmap      281,018,368    281,661,568     1.00x
MWA_compact off30, dirty2vis     dense_scan          534,024        796,296     1.49x
===============================  ==============  =============  =============  ======

which is where section 5's ``2.0x`` gate comes from -- the same constant
``tests/test_custom_vjp.py`` uses (``GRAD_MEMORY_FACTOR_SCAN``), for the same
reason: issue #21 bound both operators as linear primitives whose transposes
are each other, so the backward of a ``chunked`` forward must be a
``chunked`` adjoint *at the same ``w_chunk``*. If ``w_chunk`` is not in
``wgridder._PRIMITIVE_STATIC`` the backward can silently run a different
chunk size, which is invisible in every value and is exactly the memory the
issue is buying.

Plane counts, so the fixtures are known not to degenerate
---------------------------------------------------------
``n_w`` on the fixtures this module uses (3 channels, seed 0; float32 and
float64 agree):

  EDA2 zenith          hermitian False/True: 12/9 (1e-4), 14/11 (1e-6), 16/13 (1e-8)
  MWA_compact zenith   6/6, 8/8, 10/10
  MWA_compact off30    15/11, 17/13, 19/15
  MWA_extended off30   261/138, 263/140, 265/142

Two consequences the tests assert rather than assume. Every short fixture has
``n_w >= 6``, so a chunk set built from ``n_w`` genuinely contains interior
sizes; and every short fixture has ``n_w < 32``, so the **spec default**
``w_chunk = 32`` is always the ``w_chunk > n_w`` path on CI-sized plans. That
path is not an edge case to be tolerated -- it is what every default call on a
small plan does, and ``chunked(n_w) == dense_vmap`` forces it to exist,
because a user pinning ``w_chunk=32`` must get an answer on a plan with
``n_w = 11``.

What is *not* claimed here
--------------------------
Timing. The definition of done's "``chunked(32)`` within 1.2x of
``dense_vmap``" and the issue comment's "must not spend more than ~1.2x of
the 1.66x/2.05x margin over ducc0" are wall-clock gates on a quiet machine;
this repository puts those behind ``--runtiming`` / ``--runbench`` (AGENTS.md
sections 6-7), and they belong in the benchmark suites named by item 4 of the
implementation plan, not in a correctness module. Nothing here measures time.

Nor does this module say anything about what ``w_strategy="auto"`` should
resolve to. Retuning the heuristic is issue #34's, and
``tests/test_default_w_strategy.py`` owns the resolved-pick table.

Precision
---------
Written precision-aware from the start (AGENTS.md section 6): plans are built
with ``dtype=real_dtype``, tolerances come from ``tol(f64, f32)``, and the
memory gates are expressed in units of the plan's *own* complex itemsize, so
they hold on the float32 leg without new constants. This module must never be
added to ``conftest.collect_ignore``.

What fails today, and how
-------------------------
At ``c803ac2`` there is no ``chunked`` strategy and no ``w_chunk`` keyword, so
``dirty2vis(..., w_chunk=8)`` raises ``TypeError: got an unexpected keyword
argument``. Every call in this module goes through :func:`_chunked_call`,
which turns that into a ``pytest.fail`` naming the issue and the missing
piece, so the failure list reads as a specification rather than as a wall of
``TypeError``. The two structural tests in section 1 fail against
``wgridder._CANONICAL_W_STRATEGIES`` / ``wgridder._PRIMITIVE_STATIC``
directly.
"""

from __future__ import annotations

import contextlib
import inspect
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

import jax_nufft.wgridder as wgridder
from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft.planning import FLOAT32_EPSILON_FLOOR
from jax_nufft.wgridder import _dirty2vis_jit, _vis2dirty_jit
from tests.conftest import (
    MEERKAT,
    MWA_COMPACT,
    MWA_EXTENDED,
    X64,
    Telescope,
    synthetic_uvw,
    tol,
)

# ---------------------------------------------------------------------------
# 0. names, chunk sizes, tolerances -- all the constants in one place
# ---------------------------------------------------------------------------

# The spec value from issue #25's implementation plan item 1: "a static
# ``w_chunk: int = 32`` argument (part of the JIT key)". Pinned rather than
# read off the signature so that the signature test below is a statement about
# the API rather than a tautology.
DEFAULT_W_CHUNK = 32

# The names are *discovered* from the canonical list rather than hard-coded,
# because the issue does not fix the spelling of the windowed variant (item 3
# says only "the same chunking over the windowed slices"). Discovery keeps this
# module from failing for a cosmetic reason after the feature lands, while the
# fallbacks below are what make it fail *today* with a message that names the
# strategy the implementer has to add.


def _discover(*, windowed: bool) -> list[str]:
    names = list(getattr(wgridder, "_CANONICAL_W_STRATEGIES", ()))
    return [n for n in names if "chunk" in n and (("windowed" in n) == windowed)]


CHUNKED: str = (_discover(windowed=False) or ["chunked"])[0]
WINDOWED_CHUNKED: str = (_discover(windowed=True) or ["windowed_chunked"])[0]

EPS_VALUES = (1e-4, 1e-6, 1e-8)
N_CHAN = 3

# Reduction-order agreement between mathematically identical operators, so the
# eps-independent bound from AGENTS.md section 6's tolerance table -- the same
# constant ``tests/test_strategies_equivalent.py`` carries, and for the same
# reason: chunking changes only the order in which w-planes are summed (the
# padding planes carry zero kernel weight and contribute exactly nothing).
STRATEGY_TOL = tol(1e-11, 1.3e-6)

# For comparisons between two calls that *should* route to the same compiled
# computation (the ``chunked(1) == dense_scan`` / ``chunked(n_w) ==
# dense_vmap`` identities). On a bit-reproducible backend those are exactly
# equal and this module asserts exact equality; XLA:GPU reductions are not
# run-to-run deterministic (measured in ``tests/test_divide_by_n.py``: 1.6e-16
# float64 / 7.9e-08 float32 on a GH200 for calls sharing one executable), so
# off CPU the identity is asserted at round-off instead. See
# :func:`_assert_same_executable_values`.
SAME_EXECUTABLE_TOL = tol(1e-11, 1e-4)

# Gradient memory: the backward must stay within this factor of its own
# forward. Same constant and same measurement basis as
# ``tests/test_custom_vjp.py::GRAD_MEMORY_FACTOR_SCAN`` -- measured 1.49-1.50x
# for ``dense_scan`` and 0.98-1.98x for ``dense_vmap`` (module docstring).
GRAD_MEMORY_FACTOR = 2.0

# The monotone-curve separation gate (section 4). Measured ``dense_vmap /
# dense_scan`` on MWA_extended off30: 67.3x forward, 133.4x adjoint. A 10x gate
# therefore has 6.7x of headroom on the tighter of the two while being far
# above any plausible constant-factor wobble.
CURVE_SEPARATION = 10.0

# Monotonicity slack. The curve is asserted non-decreasing in ``w_chunk`` to
# within 1%: XLA is free to lay out two neighbouring chunk sizes with slightly
# different scratch padding, and this module is claiming a trend, not a formula.
MONOTONE_SLACK = 0.99


def _chunk_sizes(n_w: int) -> tuple[int, ...]:
    """A chunk set that spans the curve for a plan with ``n_w`` planes.

    ``{1, 2, n_w // 2, n_w - 1, n_w, max(32, n_w + 3)}``: both endpoints (the
    two identities), two interior points, and one size strictly above ``n_w``.
    ``n_w - 1`` is in the set specifically because ``n_w % (n_w - 1) == 1`` for
    every ``n_w >= 3`` -- it is the element that *guarantees* the padding path
    is exercised with more than one chunk, rather than hoping some fixture
    happens to be indivisible. ``max(32, n_w + 3)`` is the spec default 32 on
    every CI-sized plan (all of which have ``n_w < 32``) and stays above ``n_w``
    if a future fixture is larger.

    ``test_chunk_sizes_span_the_curve`` pins these properties directly, so a
    later edit to this helper that quietly made every size a divisor of ``n_w``
    would fail there rather than silently emptying the padding coverage out of
    every equivalence test in the module.
    """
    return tuple(
        sorted({1, 2, max(1, n_w // 2), max(1, n_w - 1), n_w, max(DEFAULT_W_CHUNK, n_w + 3)})
    )


# ---------------------------------------------------------------------------
# fixtures and small helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Problem:
    plan: Any
    image: Any
    vis: Any
    eps: float


_CACHE: dict[tuple, _Problem] = {}


def _problem(
    tel: Telescope,
    zenith_angle_deg: float,
    *,
    eps: float,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
    hermitian: bool = True,
    n_chan: int = N_CHAN,
) -> _Problem:
    """Plan + real per-channel image + complex visibilities, cached per key.

    Modelled on ``tests/test_strategies_equivalent.py::_build_problem``: the
    channel frequencies are distinct (+/-5% of the telescope's), so each
    channel gets its own w in wavelengths and hence its own window placement,
    and the image is a genuine per-channel stack rather than a broadcast 2-D
    plane.

    Below the float32 accuracy floor ``make_plan`` warns (issue #11); under the
    suite's ``filterwarnings = ["error"]`` that would abort, so the plan is
    built inside ``pytest.warns`` in that regime. The warning is not what is
    under test -- the strategies still have to agree with each other even where
    none of them can reach the requested epsilon.
    """
    key = (tel.name, zenith_angle_deg, eps, str(real_dtype), hermitian, n_chan)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    uvw = synthetic_uvw(tel, zenith_angle_deg, seed=0)
    if n_chan == 1:
        freq = np.array([tel.freq_hz])
    else:
        freq = tel.freq_hz * np.linspace(0.95, 1.05, n_chan)
    pix = tel.pixsize

    warns_ctx: Any = contextlib.nullcontext()
    if np.dtype(jnp.dtype(real_dtype)) == np.float32 and eps < FLOAT32_EPSILON_FLOOR:
        warns_ctx = pytest.warns(UserWarning, match="below the accuracy")
    with warns_ctx:
        plan = make_plan(
            uvw,
            freq,
            (tel.n_pix, tel.n_pix),
            pix,
            pix,
            eps,
            dtype=real_dtype,
            hermitian=hermitian,
        )

    rng = np.random.default_rng(7)
    image_shape = (tel.n_pix, tel.n_pix) if n_chan == 1 else (n_chan, tel.n_pix, tel.n_pix)
    image = jnp.asarray(rng.standard_normal(image_shape), dtype=real_dtype)
    vis = jnp.asarray(
        rng.standard_normal((tel.n_rows, n_chan)) + 1j * rng.standard_normal((tel.n_rows, n_chan)),
        dtype=complex_dtype,
    )
    problem = _Problem(plan=plan, image=image, vis=vis, eps=eps)
    _CACHE[key] = problem
    return problem


def _active_dtypes() -> tuple[DTypeLike, DTypeLike]:
    """The run's own precision, for the tests that are not parametrised over it."""
    return (jnp.float64, jnp.complex128) if X64 else (jnp.float32, jnp.complex64)


def _rel(a: Any, b: Any) -> float:
    """Relative L2 difference, computed in float64 whatever the operands are."""
    a64 = np.asarray(a, dtype=np.complex128).ravel()
    b64 = np.asarray(b, dtype=np.complex128).ravel()
    denom = float(np.linalg.norm(b64))
    assert denom > 0.0, "reference is identically zero; the comparison would be vacuous"
    assert np.all(np.isfinite(a64)), "left-hand result contains non-finite values"
    return float(np.linalg.norm(a64 - b64) / denom)


def _image_bytes(plan: Any) -> int:
    """One complex image plane, in the plan's own precision."""
    return int(plan.n_l) * int(plan.n_m) * int(np.dtype(plan.complex_dtype).itemsize)


def _temp_bytes(fn: Callable[..., Any], *args: Any) -> int:
    """XLA transient memory for ``fn`` at these argument shapes (AGENTS.md sec 6)."""
    analysis = jax.jit(fn).lower(*args).compile().memory_analysis()
    if analysis is None:  # pragma: no cover - backend without the analysis
        pytest.skip("this backend does not expose memory_analysis()")
    return int(analysis.temp_size_in_bytes)


_MISSING_FEATURE = (
    "issue #25 is not implemented: {what}. Expected a canonical w_strategy "
    "{chunked!r} (and {windowed!r} for the windowed variant, implementation-plan "
    "item 3) plus a keyword-only, static `w_chunk: int = 32` on both operators, "
    "carried in wgridder._PRIMITIVE_STATIC so the backward pass runs the "
    "forward's own chunk size. Underlying error: {exc}"
)


def _op_fn(op: str) -> Callable[..., Any]:
    assert op in ("dirty2vis", "vis2dirty"), op
    return dirty2vis if op == "dirty2vis" else vis2dirty


def _chunked_call(
    op: str,
    plan: Any,
    arg: Any,
    *,
    w_chunk: int,
    w_strategy: str | None = None,
    **kw: Any,
) -> Any:
    """Call one operator with the chunked strategy, failing *descriptively* today.

    Every chunked call in this module goes through here. Without it the whole
    module reports ``TypeError: dirty2vis() got an unexpected keyword argument
    'w_chunk'`` and the implementer has to reconstruct the specification from
    the tracebacks; with it each failure names the missing piece. It also
    keeps ``test_invalid_w_chunk_is_rejected`` honest: that test asks for a
    ``ValueError`` mentioning ``w_chunk``, and today's ``TypeError`` message
    *also* mentions ``w_chunk``, so it could pass vacuously if the valid call
    it makes first were not routed through this helper.

    ``nthreads`` defaults to ``1`` here, matching :func:`_forward` /
    :func:`_adjoint`, and that is load-bearing for the endpoint identities in
    section 3 rather than a tidiness choice. Issue #24 resolves the default
    ``nthreads=None`` differently for the scan and the vmap families (``1``
    against ``0``, "let FINUFFT decide"), so a ``chunked`` call left at the
    default could be handed a different ``Opts`` from the ``dense_scan`` call it
    is being compared against -- a different FINUFFT reduction order, and an
    identity asserted at bit level would fail for a reason that has nothing to
    do with chunking. Pinning both sides at ``1`` takes that axis out.
    """
    fn = _op_fn(op)
    strategy = CHUNKED if w_strategy is None else w_strategy
    kw.setdefault("nthreads", 1)
    try:
        return fn(plan, arg, w_strategy=strategy, w_chunk=w_chunk, **kw)
    except TypeError as exc:  # no w_chunk keyword at all
        pytest.fail(
            _MISSING_FEATURE.format(
                what=f"{op}() does not accept a `w_chunk` keyword",
                chunked=CHUNKED,
                windowed=WINDOWED_CHUNKED,
                exc=exc,
            )
        )
    except ValueError as exc:  # keyword exists, strategy name does not
        pytest.fail(
            _MISSING_FEATURE.format(
                what=f"{op}() rejects w_strategy={strategy!r}",
                chunked=CHUNKED,
                windowed=WINDOWED_CHUNKED,
                exc=exc,
            )
        )


def _assert_same_executable_values(got: Any, want: Any, *, label: str) -> None:
    """Assert an identity between two calls that should be the same computation.

    Exactly equal on a bit-reproducible backend, at ``SAME_EXECUTABLE_TOL``
    elsewhere. The reasoning is ``tests/test_divide_by_n.py``'s, measured
    there: on CPU two calls sharing one executable differ by exactly 0.0 on
    both precision legs, while a GH200 differs by 1.6e-16 (float64) and 7.9e-08
    (float32) because XLA:GPU reductions are not run-to-run deterministic. A
    CPU-only bit-equality assertion would be asserting a property of the
    backend, not of the library.
    """
    if jax.default_backend() == "cpu":
        np.testing.assert_array_equal(
            np.asarray(got),
            np.asarray(want),
            err_msg=(
                f"{label}: the two calls must be the *same computation* on the CPU "
                "backend, not merely close. Issue #25 item 2 makes the old strategy "
                "name an alias of the chunked one, so the values are bit-identical "
                "unless the alias takes a genuinely different code path."
            ),
        )
        return
    err = _rel(got, want)
    assert err < SAME_EXECUTABLE_TOL, (
        f"{label}: {err:.3e} >= {SAME_EXECUTABLE_TOL:.1e}. These two calls are "
        "supposed to be the same computation (issue #25 item 2), so the only "
        "difference allowed is the backend's own reduction non-determinism."
    )


# ---------------------------------------------------------------------------
# 1. structure: the strategy names, the keyword, and the static configuration
# ---------------------------------------------------------------------------


def test_chunked_is_a_canonical_w_strategy() -> None:
    """``chunked`` is a canonical name, not an alias and not ``auto``-only.

    ``_CANONICAL_W_STRATEGIES`` is the list every static-argument path in
    ``wgridder.py`` validates against, and membership is what makes the name
    usable as a JIT static argument at all.
    """
    names = list(getattr(wgridder, "_CANONICAL_W_STRATEGIES", ()))
    found = _discover(windowed=False)
    assert len(found) == 1, (
        "issue #25 implementation-plan item 1: expected exactly one canonical "
        "dense chunked w_strategy (the issue names it 'chunked'), found "
        f"{found} in _CANONICAL_W_STRATEGIES = {names}"
    )


def test_windowed_chunked_is_a_canonical_w_strategy() -> None:
    """The windowed variant of the same chunking -- implementation-plan item 3.

    Item 3 is not decoration: ``windowed_vmap`` today materialises ``n_w``
    images exactly as ``dense_vmap`` does (measured on MWA_extended off30:
    142.1 MB forward, 281.0 MB adjoint, against ``windowed_scan``'s 2.11 MB),
    so the windowed family has the same all-or-nothing choice the issue exists
    to remove.
    """
    names = list(getattr(wgridder, "_CANONICAL_W_STRATEGIES", ()))
    found = _discover(windowed=True)
    assert len(found) == 1, (
        "issue #25 implementation-plan item 3: expected exactly one canonical "
        "windowed chunked w_strategy, found "
        f"{found} in _CANONICAL_W_STRATEGIES = {names}"
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_both_operators_take_a_keyword_only_w_chunk_defaulting_to_32(op: str) -> None:
    """``w_chunk`` is keyword-only on both operators and defaults to 32.

    Keyword-only because every other static knob on these operators is
    (``divide_by_n``, ``w_strategy``, ``channel_strategy``, ``nthreads``), and
    a positional fifth argument would be a silent hazard next to ``weights``
    on ``vis2dirty``. The default is the issue's own spec value; it is pinned
    here because it is also the value the definition of done's memory gate is
    written against, and because on every CI-sized fixture (``n_w < 32``, see
    the module docstring) it is the ``w_chunk > n_w`` path.
    """
    params = inspect.signature(_op_fn(op)).parameters
    assert "w_chunk" in params, (
        f"issue #25: {op}() has no `w_chunk` parameter. Its signature is ({', '.join(params)})."
    )
    param = params["w_chunk"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{op}(): w_chunk must be keyword-only, got {param.kind}"
    )
    assert param.default == DEFAULT_W_CHUNK, (
        f"{op}(): w_chunk default is {param.default!r}, expected "
        f"{DEFAULT_W_CHUNK} (issue #25 implementation-plan item 1)"
    )


def test_w_chunk_is_part_of_the_primitives_static_configuration() -> None:
    """``w_chunk`` travels with ``w_strategy`` in ``_PRIMITIVE_STATIC``.

    AGENTS.md section 5: the three linear primitives from issue #21 re-bind
    with ``**params`` untouched, so "the backward runs the forward's own static
    configuration" holds *by construction* for everything in that tuple and by
    nothing at all for anything outside it. A backward that chunked differently
    from its forward would agree with it to 1e-11 on the values and have
    entirely the wrong memory -- which is the only thing this issue buys. So
    this is a real gate, not a spelling check.
    """
    static = tuple(getattr(wgridder, "_PRIMITIVE_STATIC", ()))
    assert "w_chunk" in static, (
        "issue #25: 'w_chunk' must be in wgridder._PRIMITIVE_STATIC "
        f"(currently {static}), so the transpose rules carry it into the "
        "backward pass alongside w_strategy. Without it, grad of a "
        f"{CHUNKED!r} forward can silently run a different chunk size: the "
        "values still agree to 1e-11 and the memory claim is void."
    )


def test_chunk_sizes_span_the_curve() -> None:
    """:func:`_chunk_sizes` really does cover both endpoints, the interior, the
    padding path and the over-size path.

    Every equivalence test below draws its chunk sizes from that helper, so if
    it ever degenerated -- say to a set of divisors of ``n_w`` -- the padding
    path would quietly stop being tested everywhere at once and nothing would
    say so. This is the guard against that.

    ``n_w = 6`` is the smallest plane count any fixture in this module produces
    (MWA_compact zenith at eps=1e-4; see the module docstring's table), so the
    properties are checked from there upwards.
    """
    for n_w in (6, 9, 11, 13, 19, 134):
        sizes = _chunk_sizes(n_w)
        assert 1 in sizes, f"n_w={n_w}: the chunked(1) == dense_scan endpoint is missing"
        assert n_w in sizes, f"n_w={n_w}: the chunked(n_w) == dense_vmap endpoint is missing"
        interior = [s for s in sizes if 1 < s < n_w]
        assert len(interior) >= 2, f"n_w={n_w}: only {interior} in the interior of the curve"
        padding = [s for s in interior if n_w % s != 0]
        assert padding, (
            f"n_w={n_w}: every interior chunk size {interior} divides n_w, so the "
            "padding path (n_w % w_chunk != 0) would not be exercised at all"
        )
        assert [s for s in sizes if s > n_w], f"n_w={n_w}: no chunk size above n_w"


# ---------------------------------------------------------------------------
# 2. numerical equivalence
# ---------------------------------------------------------------------------


def _forward(plan: Any, image: Any, **kw: Any) -> Any:
    return dirty2vis(plan, image, nthreads=1, **kw)


def _adjoint(plan: Any, vis: Any, **kw: Any) -> Any:
    return vis2dirty(plan, vis, nthreads=1, **kw)


@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("hermitian", [False, True])
def test_chunked_agrees_with_dense_scan_at_every_chunk_size(
    short_telescope_pointing: tuple[Telescope, float],
    eps: float,
    hermitian: bool,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """The core equivalence: ``chunked`` at every chunk size is ``dense_scan``.

    Both operators, both fold settings, all three epsilons, both precision legs
    (the module is precision-aware), and a chunk set that includes both
    endpoints, two interior sizes, a size that does *not* divide ``n_w`` and a
    size above ``n_w``. Chunking changes only the order in which w-planes are
    accumulated, so the bound is the eps-independent reduction-order constant
    from AGENTS.md section 6, not a multiple of ``eps``.

    Anti-vacuity, asserted rather than assumed: the plan must have enough
    planes for the chunk set to be interesting (``n_w >= 4``), and at least one
    chunk size used here must be a *proper* divisor-failing size -- more than
    one chunk, with a partial last one. Without those two lines a fixture that
    happened to produce ``n_w = 1`` or ``n_w = 2`` would make every cell here
    the same computation compared against itself.
    """
    tel, zen_deg = short_telescope_pointing
    problem = _problem(
        tel,
        zen_deg,
        eps=eps,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        hermitian=hermitian,
    )
    plan = problem.plan
    n_w = int(plan.n_w)
    assert n_w >= 4, (
        f"{tel.name} zen={zen_deg:g} eps={eps:g}: n_w = {n_w} is too small for the "
        "chunk set to contain distinct interior sizes -- this cell would compare a "
        "computation against itself"
    )
    sizes = _chunk_sizes(n_w)
    partial = [s for s in sizes if 1 < s < n_w and n_w % s != 0]
    assert partial, (
        f"{tel.name} zen={zen_deg:g} eps={eps:g}: no chunk size in {sizes} leaves a "
        f"partial last chunk on n_w = {n_w}, so the padding path is untested here"
    )

    case = f"{tel.name} zen={zen_deg:g} eps={eps:g} hermitian={hermitian} n_w={n_w}"

    ref_fwd = np.asarray(_forward(plan, problem.image, w_strategy="dense_scan"))
    ref_adj = np.asarray(_adjoint(plan, problem.vis, w_strategy="dense_scan"))
    assert np.all(np.isfinite(ref_fwd)) and np.all(np.isfinite(ref_adj)), (
        f"{case}: the dense_scan reference itself is non-finite"
    )

    for w_chunk in sizes:
        got_fwd = np.asarray(_chunked_call("dirty2vis", plan, problem.image, w_chunk=w_chunk))
        err = _rel(got_fwd, ref_fwd)
        assert err < STRATEGY_TOL, (
            f"{case}: forward {CHUNKED}(w_chunk={w_chunk}) disagrees with dense_scan "
            f"by {err:.3e} (bound {STRATEGY_TOL:.1e}). n_w % w_chunk = "
            f"{n_w % w_chunk}, so this cell "
            + ("exercises" if n_w % w_chunk else "does not exercise")
            + " the padded last chunk."
        )

        got_adj = np.asarray(_chunked_call("vis2dirty", plan, problem.vis, w_chunk=w_chunk))
        err = _rel(got_adj, ref_adj)
        assert err < STRATEGY_TOL, (
            f"{case}: adjoint {CHUNKED}(w_chunk={w_chunk}) disagrees with dense_scan "
            f"by {err:.3e} (bound {STRATEGY_TOL:.1e}). n_w % w_chunk = "
            f"{n_w % w_chunk}."
        )


@pytest.mark.parametrize("channel_strategy", ["scan", "vmap"])
def test_chunked_agrees_with_dense_scan_on_both_channel_strategies(
    channel_strategy: str,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """The chunk loop and the channel loop compose, both ways round.

    ``channel_strategy`` is an independent axis (AGENTS.md section 5) and
    ``"vmap"`` costs ``n_chan`` x the per-channel transient. A chunked
    implementation that indexed the plane axis relative to the wrong leading
    axis would be right at ``n_chan = 1`` and wrong here; the fixture carries
    three *distinct* channel frequencies so the channels are not
    interchangeable.
    """
    problem = _problem(
        MWA_COMPACT,
        30.0,
        eps=tol(1e-6, 1e-5),
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    plan = problem.plan
    assert N_CHAN >= 2, "N_CHAN must be above 1 or the channel axis is trivial"
    assert int(plan.n_chan) == N_CHAN
    n_w = int(plan.n_w)
    w_chunk = max(2, n_w - 1)
    assert n_w % w_chunk != 0, f"n_w={n_w}, w_chunk={w_chunk}: no partial chunk to test"

    ref_fwd = np.asarray(
        _forward(plan, problem.image, w_strategy="dense_scan", channel_strategy=channel_strategy)
    )
    got_fwd = np.asarray(
        _chunked_call(
            "dirty2vis", plan, problem.image, w_chunk=w_chunk, channel_strategy=channel_strategy
        )
    )
    err = _rel(got_fwd, ref_fwd)
    assert err < STRATEGY_TOL, (
        f"forward {CHUNKED}(w_chunk={w_chunk}) with channel_strategy="
        f"{channel_strategy!r} disagrees with dense_scan by {err:.3e}"
    )

    ref_adj = np.asarray(
        _adjoint(plan, problem.vis, w_strategy="dense_scan", channel_strategy=channel_strategy)
    )
    got_adj = np.asarray(
        _chunked_call(
            "vis2dirty", plan, problem.vis, w_chunk=w_chunk, channel_strategy=channel_strategy
        )
    )
    err = _rel(got_adj, ref_adj)
    assert err < STRATEGY_TOL, (
        f"adjoint {CHUNKED}(w_chunk={w_chunk}) with channel_strategy="
        f"{channel_strategy!r} disagrees with dense_scan by {err:.3e}"
    )


# ---------------------------------------------------------------------------
# 3. the endpoints, as identities -- what makes it a curve and not a third point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize(
    "alias, chunk_of",
    [
        ("dense_scan", lambda n_w: 1),
        ("dense_vmap", lambda n_w: n_w),
    ],
    ids=["dense_scan_is_chunked_1", "dense_vmap_is_chunked_n_w"],
)
def test_the_old_strategy_names_are_the_endpoints_of_the_chunked_curve(
    op: str,
    alias: str,
    chunk_of: Callable[[int], int],
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """Implementation-plan item 2: ``dense_scan == chunked(1)`` and
    ``dense_vmap == chunked(n_w)``.

    This is the test that distinguishes "a curve the caller can sit anywhere
    on" from "a third strategy alongside the other four". The issue comment is
    explicit that item 2 "should not be dropped for expediency", and the reason
    it matters operationally is that ``w_chunk`` only means anything as a
    position on a continuum whose ends are the two strategies whose cost is
    already characterised.

    Asserted as an *identity* rather than at the 1e-11 strategy bound: if the
    old name is an alias for the chunked path, the two calls are the same
    computation and are bit-identical on a deterministic backend. See
    :func:`_assert_same_executable_values` for the non-CPU case.
    """
    problem = _problem(
        _short_fixture(),
        30.0,
        eps=tol(1e-6, 1e-5),
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    plan = problem.plan
    n_w = int(plan.n_w)
    w_chunk = chunk_of(n_w)
    # The two endpoints must actually be different points, or this test would
    # pass on a plan where every strategy is the same computation anyway.
    assert n_w > 1, f"n_w = {n_w}: the two endpoints of the curve coincide"

    arg = problem.image if op == "dirty2vis" else problem.vis
    want = _op_fn(op)(plan, arg, w_strategy=alias, nthreads=1)
    got = _chunked_call(op, plan, arg, w_chunk=w_chunk)
    _assert_same_executable_values(
        got,
        want,
        label=(
            f"{op}: {CHUNKED}(w_chunk={w_chunk}) against {alias!r} on a plan with "
            f"n_w = {n_w} (issue #25 implementation-plan item 2)"
        ),
    )


def _short_fixture() -> Telescope:
    """MWA_compact -- the short fixture with the largest ``n_w`` at off30.

    Chosen so the two endpoints of the curve (``w_chunk = 1`` and ``w_chunk =
    n_w``) are as far apart as the fast suite allows: ``n_w = 13`` at eps=1e-6
    with the shipped fold and this module's three channels, against 8 at
    zenith. On the single-channel version of the same plan (``n_w = 12``) the
    ``dense_vmap`` / ``dense_scan`` transient ratio is 6.1x (3,260,928 /
    534,024 bytes, forward, float64 -- module docstring), so the identity below
    is between two genuinely different memory regimes rather than between two
    spellings of one.
    """
    return MWA_COMPACT


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_the_endpoint_identities_hold_for_the_windowed_family_too(
    op: str,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """``windowed_scan == windowed_chunked(1)`` and ``windowed_vmap ==
    windowed_chunked(n_w)`` -- implementation-plan item 3's half of item 2.

    Run on MWA_extended off30 rather than on a short fixture, and the reason is
    the one this repository has been bitten by: on every short fixture the
    windows hold essentially every row (``max_window_size`` 597-600 of 600 at
    off30, 400 of 400 on EDA2 zenith), so ``windowed_*`` and ``dense_*`` are the
    same computation and a windowed test on them gates nothing about windowing.
    MWA_extended off30 has ``max_window_size = 167`` of 600 rows at eps=1e-6,
    which is a real contiguous slice; the assertion below pins that the fixture
    is still in that regime.
    """
    real, complex_ = real_dtype, complex_dtype
    problem = _problem(
        MWA_EXTENDED,
        30.0,
        eps=tol(1e-6, 1e-5),
        real_dtype=real,
        complex_dtype=complex_,
        n_chan=1,
    )
    plan = problem.plan
    n_w = int(plan.n_w)
    assert int(plan.max_window_size) < int(plan.n_rows) // 2, (
        f"max_window_size = {plan.max_window_size} of {plan.n_rows} rows: the windows "
        "hold most of the array, so the windowed strategies are the dense ones under "
        "another name and this test would not be about windowing at all"
    )
    assert n_w > 1

    arg = problem.image if op == "dirty2vis" else problem.vis
    for alias, w_chunk in (("windowed_scan", 1), ("windowed_vmap", n_w)):
        want = _op_fn(op)(plan, arg, w_strategy=alias, nthreads=1)
        got = _chunked_call(op, plan, arg, w_chunk=w_chunk, w_strategy=WINDOWED_CHUNKED)
        _assert_same_executable_values(
            got,
            want,
            label=f"{op}: {WINDOWED_CHUNKED}(w_chunk={w_chunk}) against {alias!r}",
        )


def test_windowed_chunked_agrees_with_dense_scan_at_every_chunk_size(
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """The windowed chunked variant is the same operator as ``dense_scan``.

    Compared against ``dense_scan`` rather than against ``windowed_scan``,
    deliberately: the windowed family's contract is that every window contains
    every row the dense path gives a non-zero kernel weight (AGENTS.md section
    4's window-builder invariant), and a chunked rewrite of the windowed
    traversal is exactly the kind of change that can drop a boundary row. A
    windowed-against-windowed comparison would be blind to a row dropped from
    both.

    On MWA_extended off30, where the windows are 167 of 600 rows -- asserted,
    for the reason given in the test above.
    """
    problem = _problem(
        MWA_EXTENDED,
        30.0,
        eps=tol(1e-6, 1e-5),
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        n_chan=1,
    )
    plan = problem.plan
    n_w = int(plan.n_w)
    assert int(plan.max_window_size) < int(plan.n_rows) // 2, (
        f"max_window_size = {plan.max_window_size} of {plan.n_rows}: windows hold most "
        "rows, so this is not a windowed test"
    )
    # A deliberately smaller chunk set than :func:`_chunk_sizes` gives, because
    # this fixture is a 256^2 / 134-plane plan that is actually *executed*
    # (unlike the memory tests, which only lower and compile). Four sizes still
    # cover everything the property needs: both endpoints, one interior chunk
    # that leaves a partial last one (134 % 8 = 6), and the n_w - 1 size that
    # makes the padding a single plane. The assertion below pins that the
    # padding path is present rather than trusting the arithmetic above.
    sizes = (1, 8, n_w - 1, n_w)
    assert [s for s in sizes if 1 < s < n_w and n_w % s != 0], (
        f"no partial-chunk size in {sizes} for n_w = {n_w}"
    )

    ref_fwd = np.asarray(_forward(plan, problem.image, w_strategy="dense_scan"))
    ref_adj = np.asarray(_adjoint(plan, problem.vis, w_strategy="dense_scan"))
    for w_chunk in sizes:
        got = np.asarray(
            _chunked_call(
                "dirty2vis", plan, problem.image, w_chunk=w_chunk, w_strategy=WINDOWED_CHUNKED
            )
        )
        err = _rel(got, ref_fwd)
        assert err < STRATEGY_TOL, (
            f"forward {WINDOWED_CHUNKED}(w_chunk={w_chunk}) vs dense_scan: {err:.3e} "
            f">= {STRATEGY_TOL:.1e} (n_w = {n_w}, n_w % w_chunk = {n_w % w_chunk})"
        )
        got = np.asarray(
            _chunked_call(
                "vis2dirty", plan, problem.vis, w_chunk=w_chunk, w_strategy=WINDOWED_CHUNKED
            )
        )
        err = _rel(got, ref_adj)
        assert err < STRATEGY_TOL, (
            f"adjoint {WINDOWED_CHUNKED}(w_chunk={w_chunk}) vs dense_scan: {err:.3e} "
            f">= {STRATEGY_TOL:.1e} (n_w = {n_w}, n_w % w_chunk = {n_w % w_chunk})"
        )


# ---------------------------------------------------------------------------
# 4. the memory curve -- the point of the issue
# ---------------------------------------------------------------------------


def _chunked_temp(
    op: str, plan: Any, arg: Any, *, w_chunk: int, w_strategy: str | None = None
) -> int:
    def call(x: Any) -> Any:
        return _chunked_call(op, plan, x, w_chunk=w_chunk, w_strategy=w_strategy, nthreads=1)

    return _temp_bytes(call, arg)


def _plain_temp(op: str, plan: Any, arg: Any, *, w_strategy: str) -> int:
    def call(x: Any) -> Any:
        return _op_fn(op)(plan, x, w_strategy=w_strategy, nthreads=1)

    return _temp_bytes(call, arg)


_MEMORY_CURVE_CHUNKS = (1, 2, 4, 8, 16, 32, 64)


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_temp_memory_is_a_monotone_curve_in_w_chunk(op: str) -> None:
    """Transient memory rises with ``w_chunk`` and spans a wide range.

    The knob has to *be* a knob. Two things are asserted: the curve is
    non-decreasing in ``w_chunk`` (to within 1%, ``MONOTONE_SLACK`` -- this is a
    trend, not a formula), and its two ends differ by at least
    ``CURVE_SEPARATION`` = 10x, so a "chunked" implementation that quietly
    materialised every plane whatever ``w_chunk`` said -- flat curve, correct
    values, 30 GB -- fails here.

    Measured basis for the 10x (module docstring): on this fixture today
    ``dense_vmap / dense_scan`` is 67.3x on the forward and 133.4x on the
    adjoint, and those are the two ends of this curve by item 2. Run on
    MWA_extended off30 because that is the only fixture in the fast suite whose
    ``n_w`` (134 at 1 channel, eps=1e-6) exceeds the largest chunk sampled;
    the assertion below pins that.
    """
    real, complex_ = _active_dtypes()
    problem = _problem(
        MWA_EXTENDED,
        30.0,
        eps=tol(1e-6, 1e-5),
        real_dtype=real,
        complex_dtype=complex_,
        n_chan=1,
    )
    plan = problem.plan
    n_w = int(plan.n_w)
    chunks = (*[c for c in _MEMORY_CURVE_CHUNKS if c < n_w], n_w)
    assert len(chunks) >= 5 and max(_MEMORY_CURVE_CHUNKS) < n_w, (
        f"n_w = {n_w} is not larger than the sampled chunk sizes "
        f"{_MEMORY_CURVE_CHUNKS}: the curve would have no interior and this test "
        "could not tell a real chunked traversal from a flat one"
    )

    arg = problem.image if op == "dirty2vis" else problem.vis
    temps = [_chunked_temp(op, plan, arg, w_chunk=c) for c in chunks]
    pairs = list(zip(chunks, temps, strict=True))
    table = ", ".join(f"{c}:{t / 1e6:.2f}MB" for c, t in pairs)

    for (c_lo, t_lo), (c_hi, t_hi) in itertools.pairwise(pairs):
        assert t_hi >= MONOTONE_SLACK * t_lo, (
            f"{op}: transient memory fell from w_chunk={c_lo} ({t_lo / 1e6:.2f} MB) to "
            f"w_chunk={c_hi} ({t_hi / 1e6:.2f} MB). The curve must be non-decreasing: "
            f"larger chunks hold more planes live at once. Full curve: {table}"
        )

    ratio = temps[-1] / temps[0]
    assert ratio >= CURVE_SEPARATION, (
        f"{op}: w_chunk barely moves transient memory -- {ratio:.2f}x between "
        f"w_chunk=1 ({temps[0] / 1e6:.2f} MB) and w_chunk={chunks[-1]} "
        f"({temps[-1] / 1e6:.2f} MB), gate {CURVE_SEPARATION}x. On this fixture the "
        "two ends are dense_scan and dense_vmap, measured 67.3x (forward) and "
        f"133.4x (adjoint) apart at c803ac2. Full curve: {table}"
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("w_chunk", [1, 2, 4, 8, 16, 32])
def test_temp_memory_stays_under_two_w_chunk_images(op: str, w_chunk: int) -> None:
    """The definition-of-done bound: ``temp <= 2 * w_chunk * image``, plus a floor.

    The floor is not a fudge. ``chunked(1)`` *is* ``dense_scan`` (item 2), and
    ``dense_scan``'s own transient is 2.01-2.10 x image on the four fixtures in
    the module docstring -- already above ``2 * 1 * image``, because every
    strategy also pays row-sized and constant traffic (the four rows fit
    ``2 * image + 16 * n_rows + 136`` exactly). So the bound asserted here is

        temp(chunked, w_chunk) <= 2 * w_chunk * image + temp(dense_scan)

    with ``temp(dense_scan)`` measured on this same plan and operator at run
    time, so the floor tracks the fixture and the precision leg instead of
    being a constant someone chose.

    The bound is still a real gate: the assertion below first checks it is
    *tighter than today's* ``dense_vmap`` transient on the same cell, which is
    what would otherwise make the whole thing vacuous. At ``w_chunk = 32`` on
    MWA_extended off30 that is 69.2 MB against a measured 141.8 MB (forward)
    and 281.0 MB (adjoint) -- 2.0x and 4.1x of margin. This is the CI-sized
    statement of the issue comment's realistic-size gate, where the same
    formula reads 6.3 GB against a measured 29.6 GB at 3600^2.
    """
    real, complex_ = _active_dtypes()
    problem = _problem(
        MWA_EXTENDED,
        30.0,
        eps=tol(1e-6, 1e-5),
        real_dtype=real,
        complex_dtype=complex_,
        n_chan=1,
    )
    plan = problem.plan
    n_w = int(plan.n_w)
    assert n_w > w_chunk, (
        f"n_w = {n_w} is not above w_chunk = {w_chunk}: chunking would be doing "
        "nothing on this cell and the bound would be trivially satisfied by "
        "materialising every plane"
    )

    arg = problem.image if op == "dirty2vis" else problem.vis
    image = _image_bytes(plan)
    floor = _plain_temp(op, plan, arg, w_strategy="dense_scan")
    bound = 2 * w_chunk * image + floor
    vmap_temp = _plain_temp(op, plan, arg, w_strategy="dense_vmap")
    assert bound < vmap_temp, (
        f"{op} at w_chunk={w_chunk}: the bound {bound / 1e6:.2f} MB is not below the "
        f"dense_vmap transient {vmap_temp / 1e6:.2f} MB it is supposed to improve on, "
        "so this cell gates nothing"
    )

    temp = _chunked_temp(op, plan, arg, w_chunk=w_chunk)
    assert temp <= bound, (
        f"{op} {CHUNKED}(w_chunk={w_chunk}) on {MWA_EXTENDED.name} off30 (n_w = "
        f"{n_w}, image = {image / 1e6:.2f} MB): transient {temp / 1e6:.2f} MB exceeds "
        f"2 * w_chunk * image + dense_scan floor = {bound / 1e6:.2f} MB "
        f"({temp / image:.1f} images against a budget of {bound / image:.1f}). "
        f"dense_vmap on the same cell is {vmap_temp / 1e6:.2f} MB; a chunked "
        "traversal that still materialises every plane lands there."
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_temp_memory_tracks_w_chunk_not_n_w(op: str) -> None:
    """At a fixed ``w_chunk``, transient memory is the same on a 134-plane plan
    as on a 13-plane one.

    This is the claim in its cleanest form, and it needs two fixtures that
    differ in ``n_w`` **and in nothing else that memory depends on**.
    MWA_extended off30 and MeerKAT off30 are exactly that pair: both 256^2,
    both 600 rows, both single-channel here, ``n_w`` 134 against 13 at
    eps=1e-6 float64.

    The control makes it non-vacuous. ``dense_vmap`` -- whose transient *does*
    scale with ``n_w`` -- moves by a measured 10.31x (forward) and 20.62x
    (adjoint) across that same pair; the test asserts the control ratio is at
    least 5x on the run's own precision before asking anything of ``chunked``.
    Without the control this test would pass unchanged on two fixtures that had
    accidentally converged to the same ``n_w``, which is precisely the
    "quantified over an axis every fixture holds constant" failure this
    repository keeps finding.
    """
    real, complex_ = _active_dtypes()
    eps = tol(1e-6, 1e-5)
    big = _problem(MWA_EXTENDED, 30.0, eps=eps, real_dtype=real, complex_dtype=complex_, n_chan=1)
    small = _problem(MEERKAT, 30.0, eps=eps, real_dtype=real, complex_dtype=complex_, n_chan=1)

    assert _image_bytes(big.plan) == _image_bytes(small.plan), (
        "the two fixtures must have the same image size, or the comparison confounds "
        "n_w with image bytes"
    )
    assert int(big.plan.n_rows) == int(small.plan.n_rows), (
        "the two fixtures must have the same row count, or the comparison confounds "
        "n_w with row-sized traffic"
    )
    n_w_ratio = int(big.plan.n_w) / int(small.plan.n_w)
    assert n_w_ratio >= 5.0, (
        f"n_w = {big.plan.n_w} against {small.plan.n_w} ({n_w_ratio:.1f}x): the two "
        "fixtures no longer differ enough in plane count for this comparison to mean "
        "anything"
    )

    big_arg = big.image if op == "dirty2vis" else big.vis
    small_arg = small.image if op == "dirty2vis" else small.vis

    control_big = _plain_temp(op, big.plan, big_arg, w_strategy="dense_vmap")
    control_small = _plain_temp(op, small.plan, small_arg, w_strategy="dense_vmap")
    control_ratio = control_big / control_small
    assert control_ratio >= 5.0, (
        f"{op}: the dense_vmap control only moves {control_ratio:.2f}x between "
        f"n_w = {big.plan.n_w} and n_w = {small.plan.n_w} ({control_big / 1e6:.2f} MB "
        f"vs {control_small / 1e6:.2f} MB). Measured 10.31x (forward) / 20.62x "
        "(adjoint) at c803ac2. Without a moving control this test cannot tell "
        "'independent of n_w' from 'nothing here depends on anything'."
    )

    w_chunk = 8
    assert w_chunk < int(small.plan.n_w), (
        f"w_chunk = {w_chunk} must be below the smaller fixture's n_w "
        f"({small.plan.n_w}), or the small side is really chunked(n_w) = dense_vmap"
    )
    chunk_big = _chunked_temp(op, big.plan, big_arg, w_chunk=w_chunk)
    chunk_small = _chunked_temp(op, small.plan, small_arg, w_chunk=w_chunk)
    ratio = chunk_big / chunk_small
    assert ratio <= 1.5, (
        f"{op} at w_chunk={w_chunk}: transient memory is {chunk_big / 1e6:.2f} MB on "
        f"the n_w={big.plan.n_w} fixture against {chunk_small / 1e6:.2f} MB on the "
        f"n_w={small.plan.n_w} one ({ratio:.2f}x, gate 1.5x), while the two fixtures "
        f"are otherwise identical (256^2, {big.plan.n_rows} rows, 1 channel). "
        f"dense_vmap moves {control_ratio:.2f}x across the same pair. Transient "
        "memory must follow w_chunk, not n_w -- that is the whole issue."
    )


# ---------------------------------------------------------------------------
# 5. gradients
# ---------------------------------------------------------------------------


def _loss(op: str, plan: Any, **kw: Any) -> Callable[[Any], Any]:
    if op == "dirty2vis":
        return lambda x: jnp.sum(jnp.abs(dirty2vis(plan, x, nthreads=1, **kw)) ** 2)
    return lambda x: jnp.sum(vis2dirty(plan, x, nthreads=1, **kw) ** 2)


def _chunked_loss(op: str, plan: Any, *, w_chunk: int, **kw: Any) -> Callable[[Any], Any]:
    if op == "dirty2vis":
        return lambda x: jnp.sum(
            jnp.abs(_chunked_call("dirty2vis", plan, x, w_chunk=w_chunk, nthreads=1, **kw)) ** 2
        )
    return lambda x: jnp.sum(
        _chunked_call("vis2dirty", plan, x, w_chunk=w_chunk, nthreads=1, **kw) ** 2
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_the_gradient_through_chunked_matches_the_gradient_through_dense_scan(op: str) -> None:
    """Reverse mode still works, and gives the same answer.

    Issue #21 bound both operators as linear primitives whose transposes are
    each other, so a new strategy is differentiable only if it is reachable
    through those primitives. A chunked path that bypassed them would still
    produce numbers -- JAX would differentiate the scan-of-vmap directly -- and
    would cost one image-sized residual per plane, which is what #21 removed.
    The value check here is the cheap half; the memory check below is the half
    that would catch that.
    """
    real, complex_ = _active_dtypes()
    problem = _problem(
        _short_fixture(),
        30.0,
        eps=tol(1e-6, 1e-5),
        real_dtype=real,
        complex_dtype=complex_,
    )
    plan = problem.plan
    n_w = int(plan.n_w)
    w_chunk = max(2, n_w - 1)
    assert 1 < w_chunk < n_w and n_w % w_chunk != 0, (
        f"n_w = {n_w}, w_chunk = {w_chunk}: the gradient is not being taken through "
        "a genuinely chunked, genuinely padded traversal"
    )

    arg = problem.image if op == "dirty2vis" else problem.vis.real
    want = np.asarray(jax.grad(_loss(op, plan, w_strategy="dense_scan"))(arg))
    got = np.asarray(jax.grad(_chunked_loss(op, plan, w_chunk=w_chunk))(arg))
    err = _rel(got, want)
    assert err < STRATEGY_TOL, (
        f"grad of {op} through {CHUNKED}(w_chunk={w_chunk}) differs from grad through "
        f"dense_scan by {err:.3e} (bound {STRATEGY_TOL:.1e}) on a plan with n_w = {n_w}"
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_gradient_memory_through_chunked_stays_bounded_by_the_chunk_size(op: str) -> None:
    """The backward is one chunked call at the *forward's* ``w_chunk``.

    Two gates, and they fail for different reasons.

    * ``grad <= 2.0 x`` its own forward. The constant is
      ``tests/test_custom_vjp.py``'s ``GRAD_MEMORY_FACTOR_SCAN``, measured 1.50x
      (``dirty2vis``) and 1.01x (``vis2dirty``) for ``dense_scan`` on this very
      fixture. A reverse pass that replayed the w-plane loop instead of calling
      the transposed primitive lands at ``O(n_w * image)`` -- 135.5x before
      issue #21 on this fixture -- and fails immediately.
    * ``grad`` also stays under the *forward's own* chunk-size budget,
      ``2 * (2 * w_chunk * image + dense_scan floor)``. This is the gate that
      catches a backward which is a single legitimate call but at the wrong
      chunk size: ``w_chunk`` missing from ``_PRIMITIVE_STATIC`` would let the
      transpose default to 32, or to ``n_w``, with values agreeing to 1e-11 and
      the memory silently back at ``dense_vmap``'s.

    Non-vacuity, asserted: ``n_w * image`` must exceed the gate, or a per-plane
    residual backward would fit inside it and the test would gate nothing.
    """
    real, complex_ = _active_dtypes()
    problem = _problem(
        MWA_EXTENDED,
        30.0,
        eps=tol(1e-6, 1e-5),
        real_dtype=real,
        complex_dtype=complex_,
        n_chan=1,
    )
    plan = problem.plan
    n_w = int(plan.n_w)
    w_chunk = 8
    assert n_w > 4 * w_chunk, (
        f"n_w = {n_w} must be well above w_chunk = {w_chunk} for the two regimes to "
        "be distinguishable"
    )

    arg = problem.image if op == "dirty2vis" else problem.vis.real
    fwd = _chunked_temp(op, plan, arg, w_chunk=w_chunk)
    grad = _temp_bytes(jax.grad(_chunked_loss(op, plan, w_chunk=w_chunk)), arg)

    image = _image_bytes(plan)
    per_plane = n_w * image
    assert per_plane > GRAD_MEMORY_FACTOR * fwd, (
        f"{op}: n_w * image = {per_plane / 1e6:.2f} MB already fits inside "
        f"{GRAD_MEMORY_FACTOR} x forward ({GRAD_MEMORY_FACTOR * fwd / 1e6:.2f} MB), so "
        "a per-plane-residual backward would pass the ratio gate below"
    )
    assert grad <= GRAD_MEMORY_FACTOR * fwd, (
        f"{op} {CHUNKED}(w_chunk={w_chunk}): grad transient {grad / 1e6:.2f} MB "
        f"against a forward of {fwd / 1e6:.2f} MB ({grad / fwd:.2f}x, gate "
        f"{GRAD_MEMORY_FACTOR}x) on a plan with n_w = {n_w}"
    )

    floor = _plain_temp(op, plan, arg, w_strategy="dense_scan")
    budget = GRAD_MEMORY_FACTOR * (2 * w_chunk * image + floor)
    vmap_temp = _plain_temp(op, plan, arg, w_strategy="dense_vmap")
    assert budget < vmap_temp, (
        f"{op}: the gradient budget {budget / 1e6:.2f} MB is not below dense_vmap's "
        f"forward transient {vmap_temp / 1e6:.2f} MB, so it cannot detect a backward "
        "that ran at w_chunk = n_w"
    )
    assert grad <= budget, (
        f"{op} {CHUNKED}(w_chunk={w_chunk}): grad transient {grad / 1e6:.2f} MB "
        f"exceeds 2 x (2 * w_chunk * image + scan floor) = {budget / 1e6:.2f} MB. The "
        "backward must be one chunked call at the forward's own w_chunk -- check that "
        "'w_chunk' is in wgridder._PRIMITIVE_STATIC."
    )


# ---------------------------------------------------------------------------
# 6. jit / vmap / disable_jit, and the cache key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_chunked_works_under_disable_jit(op: str) -> None:
    """``jax.disable_jit()`` is the one public route to the primitives'
    ``def_impl``; the chunked strategy has to survive it.

    Mirrors ``tests/test_custom_vjp.py::test_the_operators_work_under_disable_jit``.
    Debugging a wgridder call with ``disable_jit`` is a normal thing to do, and
    a strategy that only exists inside a ``jax.jit`` trace would fail here and
    nowhere else.
    """
    real, complex_ = _active_dtypes()
    problem = _problem(
        _short_fixture(), 30.0, eps=tol(1e-6, 1e-5), real_dtype=real, complex_dtype=complex_
    )
    plan = problem.plan
    w_chunk = max(2, int(plan.n_w) - 1)
    arg = problem.image if op == "dirty2vis" else problem.vis

    want = _chunked_call(op, plan, arg, w_chunk=w_chunk)
    with jax.disable_jit():
        got = _chunked_call(op, plan, arg, w_chunk=w_chunk)
    err = _rel(got, want)
    bound = tol(1e-11, 5e-6)
    assert err < bound, (
        f"{op} with {CHUNKED}(w_chunk={w_chunk}) under jax.disable_jit() differs by "
        f"{err:.3e} (bound {bound:.1e}) from the jitted call"
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_chunked_survives_jit_and_vmap(op: str) -> None:
    """``jit`` from the outside, and ``vmap`` over a batch of inputs.

    ``vmap`` matters more than it looks: issue #21's primitives carry a
    ``batch_shape`` param and the batching rule pushes the mapped axis into it
    so a batched call stays *one* call. A chunked lowering that assumed a
    particular rank -- image ``(n_chan, n_l, n_m)``, no batch axis -- would
    break exactly here, and the values are compared against the same operator
    applied one batch element at a time so a wrong axis shows up as a
    disagreement rather than as a shape error only.
    """
    real, complex_ = _active_dtypes()
    problem = _problem(
        _short_fixture(),
        30.0,
        eps=tol(1e-6, 1e-5),
        real_dtype=real,
        complex_dtype=complex_,
        n_chan=1,
    )
    plan = problem.plan
    w_chunk = max(2, int(plan.n_w) - 1)
    base = problem.image if op == "dirty2vis" else problem.vis
    batch = jnp.stack([base * (1.0 + 0.25 * k) for k in range(3)])

    def one(x: Any) -> Any:
        return _chunked_call(op, plan, x, w_chunk=w_chunk)

    stacked = jnp.stack([one(batch[k]) for k in range(batch.shape[0])])

    got_vmap = jax.vmap(one)(batch)
    err = _rel(got_vmap, stacked)
    assert err < STRATEGY_TOL, (
        f"{op}: vmap over a batch of 3 with {CHUNKED}(w_chunk={w_chunk}) differs from "
        f"the same three calls taken one at a time by {err:.3e}"
    )

    got_jit = jax.jit(jax.vmap(one))(batch)
    err = _rel(got_jit, stacked)
    assert err < STRATEGY_TOL, (
        f"{op}: jit(vmap(...)) with {CHUNKED}(w_chunk={w_chunk}) differs from the "
        f"unbatched loop by {err:.3e}"
    )


@pytest.mark.parametrize(
    "op, jit_fn", [("dirty2vis", _dirty2vis_jit), ("vis2dirty", _vis2dirty_jit)]
)
def test_w_chunk_is_part_of_the_jit_cache_key(op: str, jit_fn: Any) -> None:
    """Two different ``w_chunk`` values must compile to two different executables.

    ``w_chunk`` changes the shape of every intermediate in the plane loop, so it
    cannot be a traced value; the issue says so ("part of the JIT key"). If it
    leaked through as a Python int captured by closure, the *second* chunk size
    would silently reuse the first one's executable -- same numbers, wrong
    memory, and no test that compares values could ever see it.

    Measured the way ``tests/test_default_w_strategy.py`` measures the mirror
    property for ``auto``: ``_clear_cache()`` then ``_cache_size()`` on the
    jitted inner function.
    """
    real, complex_ = _active_dtypes()
    problem = _problem(
        _short_fixture(), 30.0, eps=tol(1e-6, 1e-5), real_dtype=real, complex_dtype=complex_
    )
    plan = problem.plan
    n_w = int(plan.n_w)
    chunk_a, chunk_b = 2, max(3, n_w - 1)
    # Two *distinct* interior chunk sizes, or the cache-size comparison below
    # would be asking whether one static argument compiles twice.
    assert chunk_a != chunk_b, f"n_w = {n_w} leaves only one usable chunk size"
    assert chunk_a < n_w and chunk_b < n_w, (
        f"n_w = {n_w}: chunk sizes {chunk_a} and {chunk_b} must both be below it, so "
        "neither is silently the chunked(n_w) == dense_vmap endpoint"
    )

    arg = problem.image if op == "dirty2vis" else problem.vis

    jit_fn._clear_cache()
    jax.block_until_ready(_chunked_call(op, plan, arg, w_chunk=chunk_a))
    after_a = jit_fn._cache_size()
    assert after_a == 1, f"sanity: expected one cache entry after the first call, got {after_a}"

    jax.block_until_ready(_chunked_call(op, plan, arg, w_chunk=chunk_a))
    after_repeat = jit_fn._cache_size()
    assert after_repeat == 1, (
        f"{op}: repeating the same w_chunk={chunk_a} call added a cache entry "
        f"({after_a} -> {after_repeat}); the static argument is not hashing stably"
    )

    jax.block_until_ready(_chunked_call(op, plan, arg, w_chunk=chunk_b))
    after_b = jit_fn._cache_size()
    assert after_b == 2, (
        f"{op}: w_chunk={chunk_a} then w_chunk={chunk_b} produced {after_b} cache "
        "entries, expected 2. w_chunk must be a static argument at the JIT boundary "
        "(issue #25 implementation-plan item 1), or the second chunk size reuses the "
        "first one's executable and the memory curve is a fiction."
    )


# ---------------------------------------------------------------------------
# 7. validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize(
    "bad", [0, -1, -32, 2.5, "8", None], ids=["zero", "minus1", "minus32", "float", "str", "none"]
)
def test_invalid_w_chunk_is_rejected(op: str, bad: Any) -> None:
    """``w_chunk`` must be a positive ``int``; anything else raises.

    ``0`` and negatives are the ones that matter operationally: ``w_chunk=0``
    would pad the plane grid to a multiple of zero, and a silently clamped
    value would leave the caller believing they had chosen a point on the
    memory curve while sitting somewhere else entirely. Non-integers are
    rejected because the value is a *static* shape, not a quantity.

    The test makes a **valid** chunked call first, through
    :func:`_chunked_call`. That is load-bearing rather than tidy: today's
    ``TypeError: dirty2vis() got an unexpected keyword argument 'w_chunk'``
    both raises ``TypeError`` and contains the string ``w_chunk``, so without
    the valid call this test would pass at ``c803ac2`` while the feature does
    not exist.
    """
    real, complex_ = _active_dtypes()
    problem = _problem(
        _short_fixture(), 30.0, eps=tol(1e-6, 1e-5), real_dtype=real, complex_dtype=complex_
    )
    plan = problem.plan
    arg = problem.image if op == "dirty2vis" else problem.vis

    # Fails descriptively at c803ac2; after the feature lands this is the proof
    # that the keyword exists, so the rejection below is about the *value*.
    jax.block_until_ready(_chunked_call(op, plan, arg, w_chunk=2))

    with pytest.raises((ValueError, TypeError), match="w_chunk"):
        _op_fn(op)(plan, arg, w_strategy=CHUNKED, w_chunk=bad, nthreads=1)
