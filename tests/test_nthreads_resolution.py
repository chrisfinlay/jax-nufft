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
  * otherwise the strategy family decides, from a canonical name: one of
    the four canonical names is used as-is, and anything else -- ``"auto"``
    and the deprecated ``"scan"``/``"vmap"`` aliases -- is put through
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
    0.27 vs 0.62 ms per transform).

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

import pytest

from jax_nufft.wgridder import _auto_w_strategy


class _ResolveNthreads(Protocol):
    def __call__(
        self,
        nthreads: int | None,
        w_strategy: str,
        n_rows: int,
        *,
        plan: object | None = ...,
        is_adjoint: bool | None = ...,
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
    w_kernel_width: int = 8,
    window_padding_overhead: float = 1.0,
    n_rows: int = 600,
):
    """Minimal stand-in exposing the fields ``_auto_w_strategy`` reads.

    Mirrors ``tests/test_auto_strategy.py::_stub_plan`` -- kept local (not
    imported) so this module stays self-contained like the rest of the
    per-file test suite.
    """
    return SimpleNamespace(
        n_w=n_w,
        w_kernel_width=w_kernel_width,
        window_padding_overhead=window_padding_overhead,
        n_rows=n_rows,
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


@pytest.mark.parametrize("explicit", [0, 1, 2, 8])
def test_explicit_nthreads_passes_through_unchanged(
    resolve_nthreads: _ResolveNthreads, explicit: int
) -> None:
    """An explicit nthreads (including 0, i.e. "let FINUFFT decide") is
    never touched by the resolution rule, for any strategy/n_rows
    combination -- this is what makes ``nthreads=0`` still usable as an
    explicit opt-out of the new default."""
    assert resolve_nthreads(explicit, "dense_scan", 1) == explicit
    assert resolve_nthreads(explicit, "windowed_vmap", 10_000_000) == explicit


def test_explicit_nthreads_bypasses_auto_resolution_entirely(
    resolve_nthreads: _ResolveNthreads,
) -> None:
    """An explicit nthreads must short-circuit before any strategy
    canonicalisation happens: ``w_strategy="auto"`` with no ``plan`` /
    ``is_adjoint`` must NOT raise when nthreads is given explicitly, unlike
    the ``nthreads=None`` case above."""
    assert resolve_nthreads(4, "auto", 100) == 4
    assert resolve_nthreads(0, "auto", 10_000_000) == 0
