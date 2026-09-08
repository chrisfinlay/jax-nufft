"""Unit tests for the ``nthreads`` resolution rule (issue #24, R11/D4).

With the pre-#24 default ``nthreads=0``, every per-plane FINUFFT call in the
``*_scan`` strategies re-spins the whole OpenMP pool: measured on this
machine, a bare ``dirty2vis(..., w_strategy="dense_scan")`` at the default is
2.5-8x slower than the same call with ``nthreads=1`` (see the timing gate in
``tests/test_timing_nthreads.py`` and the PR description for numbers). The
fix is to make ``nthreads: int | None = None`` on both operators and resolve
``None`` *before* the JIT boundary (so two callers that both leave it at the
default still share a JIT cache entry) via a new ``_resolve_nthreads``
helper in ``jax_nufft.wgridder``:

  * an explicit ``nthreads`` (any int, including 0) always passes straight
    through -- resolution never runs, and ``w_strategy="auto"`` never needs
    ``plan`` / ``is_adjoint`` in that case;
  * otherwise the strategy family decides, from a canonical name: a member
    of ``_CANONICAL_W_STRATEGIES`` is used as-is, and anything else --
    ``"auto"`` and the deprecated ``"scan"``/``"vmap"`` aliases -- is put through
    ``_canonicalise_w_strategy`` and so resolves the same way it does for
    the strategy dispatch itself. (The already-canonical short-circuit is
    issue #46: with ``w_strategy`` also defaulting to ``"auto"``, the
    operators resolve in the wrapper and pass the result down, and
    resolving it a second time here would be a silent dependency on
    canonicalisation staying a fixed point. See
    ``tests/test_default_w_strategy.py::test_defaulted_call_canonicalises_exactly_once``.)
  * if ``n_rows`` is below the tunable ``_NTHREADS_SMALL_N_ROWS`` cutoff,
    the plane loop is short enough that spinning up a thread pool per call
    isn't worth it regardless of strategy -> ``1``;
  * otherwise ``"dense_scan"`` / ``"windowed_scan"`` (the ``scan`` family,
    which re-enters FINUFFT once per w-plane) -> ``1``, and
    ``"dense_vmap"`` / ``"windowed_vmap"`` (the ``vmap`` family, one batched
    FINUFFT call) -> ``0`` (measured to benefit from threads in the issue:
    0.27 vs 0.62 ms per transform);
  * and, since issue #25, ``"chunked"`` / ``"windowed_chunked"``, which do
    not belong to a family by name: they get ``1`` when their resolved
    ``w_chunk`` is ``1`` (one FINUFFT call per plane, exactly the scan
    family's situation) and ``0`` for any larger chunk (one batched call per
    chunk). That branch is pinned by :data:`_CHUNKED_RESOLUTION_GRID` below
    and by nothing else: instrumenting it and running the whole
    ``--runslow`` suite records **zero** hits, because every chunked call in
    the suite pins ``nthreads=1`` and no fixture reaches the 100k-row cutoff.

``_resolve_nthreads`` and ``_NTHREADS_SMALL_N_ROWS`` are imported *inside*
the ``resolve_nthreads`` / ``small_n_rows_cutoff`` fixtures rather than at
module scope, so if either is ever renamed or removed the breakage surfaces
as ordinary per-test failures instead of a module-collection error that
would abort the whole ``pytest`` run before anything executes.

The grid in :data:`_RESOLUTION_GRID` below hardcodes every expected value
rather than deriving it from the imported cutoff constant, precisely so a
later change to the rule -- quietly widening the vmap family, say, or moving
the cutoff -- is caught as a test failure instead of silently redefining
what "correct" means.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Protocol

import numpy as np
import pytest

from jax_nufft import make_plan
from jax_nufft.kernel import kernel_params
from jax_nufft.wgridder import _auto_w_strategy
from tests.conftest import requires_x64

# The kernel half-width a typical eps=1e-6 plan actually gets, derived rather
# than written out -- the same treatment ``tests/test_auto_strategy.py`` gives
# its own copy of this stub, and for the same reason: a literal here is exactly
# how the pre-issue-#9 width of 8 survived the rule change that made it 7.
_TYPICAL_EPS = 1e-6
_TYPICAL_W_KERNEL_WIDTH = kernel_params(_TYPICAL_EPS)[0]


class _ResolveNthreads(Protocol):
    def __call__(
        self,
        nthreads: int | None,
        w_strategy: str,
        n_rows: int,
        *,
        plan: object | None = ...,
        is_adjoint: bool | None = ...,
        w_chunk: int | None = ...,
    ) -> int: ...


@pytest.fixture
def resolve_nthreads() -> _ResolveNthreads:
    """Lazy import of the not-yet-implemented resolution helper.

    See the module docstring for why this is a fixture (deferred import)
    rather than a module-level ``from jax_nufft.wgridder import
    _resolve_nthreads``.
    """
    from jax_nufft.wgridder import _resolve_nthreads

    return _resolve_nthreads


@pytest.fixture
def small_n_rows_cutoff() -> int:
    """Lazy import of the not-yet-implemented tunable cutoff constant."""
    from jax_nufft.wgridder import _NTHREADS_SMALL_N_ROWS

    return _NTHREADS_SMALL_N_ROWS


def _stub_plan(
    *,
    n_w: int,
    w_kernel_width: int = _TYPICAL_W_KERNEL_WIDTH,
    window_padding_overhead: float = 1.0,
    n_rows: int = 600,
):
    """Minimal stand-in exposing the fields ``_auto_w_strategy`` reads.

    Mirrors ``tests/test_auto_strategy.py::_stub_plan`` -- kept local (not
    imported) so this module stays self-contained like the rest of the
    per-file test suite. The default width is a typical eps=1e-6 plan's, 7
    under the issue #9 rule ``W = ceil(-log10(eps / 10))``, and is *derived*
    from ``kernel_params`` rather than written out, so it cannot drift the way
    the literal 8 it replaced did. That eps=1e-6 really is such a plan is
    checked against ``make_plan`` by
    ``test_the_stub_default_width_is_the_real_eps_1e_6_width`` below.
    """
    return SimpleNamespace(
        n_w=n_w,
        w_kernel_width=w_kernel_width,
        window_padding_overhead=window_padding_overhead,
        # issue #26: the adjoint leg reads its own bucketed ratio. Mirrored at
        # the un-bucketed value, which is a real plan's degenerate case (one
        # bucket per channel) and keeps this stub saying "this plan has this
        # much padding, in whichever direction you ask". The split itself is
        # gated in ``tests/test_auto_strategy.py``.
        window_padding_overhead_adjoint=window_padding_overhead,
        n_rows=n_rows,
    )


@requires_x64
def test_the_stub_default_width_is_the_real_eps_1e_6_width() -> None:
    """The mirrored stub's default must be a real eps=1e-6 plan's width.

    The twin of ``tests/test_auto_strategy.py``'s check of the same name, and
    it has to build a plan to be worth anything. Comparing the stub default
    against ``kernel_params(1e-6)`` alone is a tautology now that the default
    is derived from it; what is not a tautology is that ``make_plan`` at
    eps=1e-6 really does hand back a kernel that wide, which is the claim the
    stub's docstring makes and the one that went stale for six PRs after issue
    #9 changed the width rule.

    ``@requires_x64`` only because a default-dtype ``make_plan`` raises with
    x64 off and eps=1e-6 is below the float32 floor. The rest of this module
    is precision-agnostic and stays out of ``conftest.collect_ignore``.
    """
    assert _TYPICAL_W_KERNEL_WIDTH == 7, _TYPICAL_W_KERNEL_WIDTH
    assert _stub_plan(n_w=10).w_kernel_width == _TYPICAL_W_KERNEL_WIDTH

    uvw = np.zeros((8, 3))
    uvw[:, 0] = np.linspace(-50.0, 50.0, 8)
    uvw[:, 2] = np.linspace(-5.0, 5.0, 8)
    plan = make_plan(uvw, np.array([1.4e9]), (16, 16), 0.005, 0.005, _TYPICAL_EPS)
    assert plan.w_kernel_width == _TYPICAL_W_KERNEL_WIDTH, (
        f"make_plan(eps={_TYPICAL_EPS:g}) gives w_kernel_width={plan.w_kernel_width}, "
        f"but _stub_plan defaults to {_TYPICAL_W_KERNEL_WIDTH}: every n_w / W ratio in "
        "this file is being computed against a width no real plan has"
    )


class _FakeDevice:
    """Stand-in for ``jax.devices()[0]``; only ``.platform`` is read."""

    def __init__(self, platform: str) -> None:
        self.platform = platform


def _patch_platform(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    import jax

    monkeypatch.setattr(jax, "devices", lambda *a, **kw: [_FakeDevice(platform)])


# -- the cutoff constant itself ---------------------------------------------


def test_small_n_rows_cutoff_is_100k(small_n_rows_cutoff: int) -> None:
    """Pin the tunable constant's value so a silent retune is visible in the
    diff of this test rather than only in the grid results below."""
    assert small_n_rows_cutoff == 100_000


# -- the full strategy x n_rows grid, canonical strategy names --------------
#
# (w_strategy, n_rows, expected_nthreads). Every row is a literal, not a
# derived value, per the issue's instruction that a later change to the
# rule must be caught explicitly.
_RESOLUTION_GRID: list[tuple[str, int, int]] = [
    # -- n_rows far below the cutoff: every strategy resolves to 1 --------
    ("dense_scan", 1, 1),
    ("dense_vmap", 1, 1),
    ("windowed_scan", 1, 1),
    ("windowed_vmap", 1, 1),
    ("dense_scan", 100, 1),
    ("dense_vmap", 100, 1),
    ("windowed_scan", 100, 1),
    ("windowed_vmap", 100, 1),
    ("dense_scan", 600, 1),  # a typical review-fixture n_rows
    ("dense_vmap", 600, 1),
    ("windowed_scan", 600, 1),
    ("windowed_vmap", 600, 1),
    ("dense_scan", 50_000, 1),  # GH200_large's n_rows, still below 1e5
    ("dense_vmap", 50_000, 1),
    ("windowed_scan", 50_000, 1),
    ("windowed_vmap", 50_000, 1),
    # -- just below the cutoff: still overridden to 1 ----------------------
    ("dense_scan", 99_999, 1),
    ("dense_vmap", 99_999, 1),
    ("windowed_scan", 99_999, 1),
    ("windowed_vmap", 99_999, 1),
    # -- at and above the cutoff: strategy family decides -------------------
    ("dense_scan", 100_000, 1),
    ("dense_vmap", 100_000, 0),
    ("windowed_scan", 100_000, 1),
    ("windowed_vmap", 100_000, 0),
    ("dense_scan", 100_001, 1),
    ("dense_vmap", 100_001, 0),
    ("windowed_scan", 100_001, 1),
    ("windowed_vmap", 100_001, 0),
    ("dense_scan", 500_000, 1),
    ("dense_vmap", 500_000, 0),
    ("windowed_scan", 500_000, 1),
    ("windowed_vmap", 500_000, 0),
]


@pytest.mark.parametrize(
    ("w_strategy", "n_rows", "expected"),
    _RESOLUTION_GRID,
    ids=[f"{s}-n{n}-want{e}" for s, n, e in _RESOLUTION_GRID],
)
def test_resolve_nthreads_none_grid(
    resolve_nthreads: _ResolveNthreads, w_strategy: str, n_rows: int, expected: int
) -> None:
    assert resolve_nthreads(None, w_strategy, n_rows) == expected


# -- the chunked strategies (issue #25): the family is w_chunk's, not the name's
#
# (w_strategy, w_chunk, n_rows, expected_nthreads). Same literal-values policy
# as the grid above. This is the *only* thing that executes the
# ``_CHUNKED_W_STRATEGIES`` branch of ``_resolve_nthreads``: the branch needs
# ``nthreads is None`` **and** ``n_rows >= _NTHREADS_SMALL_N_ROWS``, and every
# chunked call anywhere else in the suite pins ``nthreads=1`` (the operator
# tests deliberately, to take the thread count out of the endpoint identities)
# while no fixture in ``tests/conftest.py`` has 100k rows -- GH200_large, the
# largest, has 50k. Instrumented under ``pytest -q --runslow`` before these
# rows existed, the branch took 0 hits across the whole suite.
_CHUNKED_RESOLUTION_GRID: list[tuple[str, int | None, int, int]] = [
    # -- below the cutoff: the small-n_rows override wins for every chunk ----
    ("chunked", 1, 600, 1),
    ("chunked", 8, 600, 1),
    ("chunked", 32, 600, 1),
    ("windowed_chunked", 1, 600, 1),
    ("windowed_chunked", 32, 600, 1),
    ("chunked", 32, 99_999, 1),
    # -- at and above the cutoff: w_chunk decides ---------------------------
    #    w_chunk == 1 is one FINUFFT call per plane -> the scan family's 1
    ("chunked", 1, 100_000, 1),
    ("windowed_chunked", 1, 100_000, 1),
    ("chunked", 1, 500_000, 1),
    ("windowed_chunked", 1, 500_000, 1),
    #    any larger chunk is a batched call per chunk -> the vmap family's 0
    ("chunked", 2, 100_000, 0),
    ("chunked", 8, 100_000, 0),
    ("chunked", 32, 100_000, 0),
    ("chunked", 32, 100_001, 0),
    ("chunked", 128, 500_000, 0),
    ("windowed_chunked", 2, 100_000, 0),
    ("windowed_chunked", 32, 100_000, 0),
    #    a direct caller who never resolved w_chunk gets the vmap answer,
    #    matching the shipped default w_chunk = 32
    ("chunked", None, 100_000, 0),
    ("windowed_chunked", None, 500_000, 0),
]


@pytest.mark.parametrize(
    ("w_strategy", "w_chunk", "n_rows", "expected"),
    _CHUNKED_RESOLUTION_GRID,
    ids=[f"{s}-c{c}-n{n}-want{e}" for s, c, n, e in _CHUNKED_RESOLUTION_GRID],
)
def test_resolve_nthreads_chunked_grid(
    resolve_nthreads: _ResolveNthreads,
    w_strategy: str,
    w_chunk: int | None,
    n_rows: int,
    expected: int,
) -> None:
    """``chunked`` follows its ``w_chunk``, not its name (issue #25).

    The two chunked strategies span both families -- ``w_chunk = 1`` re-enters
    FINUFFT once per w-plane the way ``dense_scan`` does, and any larger chunk
    is one batched call per chunk the way ``dense_vmap`` is -- so the family
    rule the four older names are looked up by cannot be read off the name.
    """
    assert resolve_nthreads(None, w_strategy, n_rows, w_chunk=w_chunk) == expected


def test_an_explicit_nthreads_still_wins_over_the_chunked_rule(
    resolve_nthreads: _ResolveNthreads,
) -> None:
    """``w_chunk`` never overrides an explicit ``nthreads`` (issue #25 + #24).

    The pass-through branch is checked before anything strategy-shaped, and
    adding a strategy family that reads a second keyword must not have moved
    it. ``0`` is the interesting value: it is falsy, so a pass-through written
    as a truthiness test rather than an ``is not None`` test would silently
    fall through to the family rule and return ``1`` on the ``w_chunk = 1``
    row below.
    """
    for w_chunk in (1, 32):
        for explicit in (0, 1, 4, 16):
            got = resolve_nthreads(explicit, "chunked", 500_000, w_chunk=w_chunk)
            assert got == explicit, (
                f"nthreads={explicit} with w_chunk={w_chunk} resolved to {got}; an "
                "explicit thread count must pass straight through"
            )


# -- w_strategy="auto" resolves to a canonical strategy first --------------


def test_auto_below_cutoff_resolves_to_1_regardless_of_platform_heuristic(
    resolve_nthreads: _ResolveNthreads,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the cutoff, the small-n_rows override wins even when the
    platform heuristic would have picked a vmap-family strategy: pin GPU
    and a plan the GPU heuristic resolves to dense_vmap
    (``test_gpu_small_n_w_picks_dense_vmap`` in test_auto_strategy.py), but
    call with n_rows below the cutoff."""
    _patch_platform(monkeypatch, "gpu")
    plan = _stub_plan(n_w=7, w_kernel_width=6, n_rows=600)
    assert _auto_w_strategy(plan, is_adjoint=False) == "dense_vmap"  # sanity
    assert resolve_nthreads(None, "auto", 600, plan=plan, is_adjoint=False) == 1


def test_auto_above_cutoff_cpu_heuristic_resolves_to_scan_family(
    resolve_nthreads: _ResolveNthreads,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On CPU the auto heuristic only ever picks a scan-family strategy
    (see ``_auto_w_strategy_cpu``), so above the cutoff it must still
    resolve to 1 -- this exercises the "auto resolves to a canonical
    strategy first" contract on the scan-family branch."""
    _patch_platform(monkeypatch, "cpu")
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=1.4)
    assert _auto_w_strategy(plan, is_adjoint=True) == "windowed_scan"  # sanity
    assert resolve_nthreads(None, "auto", 500_000, plan=plan, is_adjoint=True) == 1


def test_auto_above_cutoff_gpu_heuristic_resolves_to_vmap_family(
    resolve_nthreads: _ResolveNthreads,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On GPU the auto heuristic can pick a vmap-family strategy; above the
    cutoff that must resolve to 0. Uses the same small-n_w plan as
    ``test_gpu_small_n_w_picks_dense_vmap``, but with a row count passed to
    ``resolve_nthreads`` that is above the nthreads cutoff (independent of
    ``plan.n_rows``, which only the GPU strategy heuristic reads)."""
    _patch_platform(monkeypatch, "gpu")
    plan = _stub_plan(n_w=7, w_kernel_width=6, n_rows=600)
    assert _auto_w_strategy(plan, is_adjoint=False) == "dense_vmap"  # sanity
    assert resolve_nthreads(None, "auto", 500_000, plan=plan, is_adjoint=False) == 0


def test_auto_without_plan_context_raises(resolve_nthreads: _ResolveNthreads) -> None:
    """Resolving ``nthreads=None`` with ``w_strategy="auto"`` needs plan +
    is_adjoint context to canonicalise the strategy first, same contract as
    ``_canonicalise_w_strategy`` itself."""
    with pytest.raises(ValueError, match=r"auto.*plan.*is_adjoint"):
        resolve_nthreads(None, "auto", 100)


# -- an explicit nthreads always short-circuits resolution ------------------


# Every ``w_strategy`` an explicit ``nthreads`` can arrive with -- the four
# canonical names, ``"auto"``, and the two deprecated v0.1 aliases. The
# pass-through claim is about *all* of them: the explicit branch returns
# before any strategy handling at all, so a name that could not even be
# resolved here (``"auto"`` without plan context) or that would warn if it
# were (``"scan"`` / ``"vmap"``) must still come straight back. Literal
# strings rather than an import, matching ``_RESOLUTION_GRID`` above and the
# module docstring's note on deferred imports.
_PASSTHROUGH_STRATEGIES: tuple[str, ...] = (
    "dense_scan",
    "dense_vmap",
    "windowed_scan",
    "windowed_vmap",
    "auto",
    "scan",
    "vmap",
)
# Literal row counts either side of the 100k cutoff, same convention as
# ``_RESOLUTION_GRID``: the explicit branch must return before the cutoff is
# consulted, so both regimes have to be covered to say so.
_PASSTHROUGH_N_ROWS: tuple[int, ...] = (1, 99_999, 100_000, 10_000_000)
# Includes 0 (the pre-#24 default, and the documented opt-out), 1 (what the
# scan family resolves to) and values that are neither, so a rule that
# happened to return the *right* number for one strategy cannot pass.
_PASSTHROUGH_VALUES: tuple[int, ...] = (0, 1, 2, 8, 72)


@pytest.mark.parametrize("explicit", _PASSTHROUGH_VALUES)
@pytest.mark.parametrize("n_rows", _PASSTHROUGH_N_ROWS)
@pytest.mark.parametrize("w_strategy", _PASSTHROUGH_STRATEGIES)
def test_explicit_nthreads_passes_through_unchanged(
    resolve_nthreads: _ResolveNthreads, w_strategy: str, n_rows: int, explicit: int
) -> None:
    """An explicit nthreads (including 0, i.e. "let FINUFFT decide") is
    never touched by the resolution rule, for any strategy/n_rows
    combination -- this is what makes ``nthreads=0`` still usable as an
    explicit opt-out of the new default.

    The claim is universal, so the parametrisation enumerates it rather than
    sampling it: every strategy name the function accepts, both sides of the
    row cutoff, and five explicit values. An earlier revision asserted two
    hand-picked ``(strategy, n_rows)`` pairs, which left a rule that
    special-cased any *other* strategy -- e.g. overriding ``windowed_scan``
    to ``1`` -- passing the entire suite.

    The alias rows carry a second assertion implicitly: the suite runs with
    ``filterwarnings = ["error"]``, so if the explicit branch ever stopped
    short-circuiting and canonicalised ``"scan"`` / ``"vmap"`` on the way
    past, the ``DeprecationWarning`` would fail these cases.
    """
    assert resolve_nthreads(explicit, w_strategy, n_rows) == explicit


def test_explicit_nthreads_bypasses_auto_resolution_entirely(
    resolve_nthreads: _ResolveNthreads,
) -> None:
    """An explicit nthreads must short-circuit before any strategy
    canonicalisation happens: ``w_strategy="auto"`` with no ``plan`` /
    ``is_adjoint`` must NOT raise when nthreads is given explicitly, unlike
    the ``nthreads=None`` case above.

    The grid above now covers ``"auto"`` too; this stays as the named
    statement of the contract, so the failure a wiring regression produces
    reads as "auto stopped bypassing resolution" rather than as one row of a
    140-cell table."""
    assert resolve_nthreads(4, "auto", 100) == 4
    assert resolve_nthreads(0, "auto", 10_000_000) == 0
