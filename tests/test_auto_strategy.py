"""Unit tests for :func:`jax_nufft.wgridder._auto_w_strategy`.

Part 4.1 acceptance: the helper is exercised in isolation with synthetic
plan objects, covering every branch of the CPU-derived heuristic. The
public-API wiring lands in Part 4.2 and gets its own integration test
there.

The heuristic only reads three fields off the plan (``n_w``,
``w_kernel_width``, ``window_padding_overhead``), so we deliberately
build lightweight ``SimpleNamespace`` stand-ins rather than running
``make_plan`` -- it keeps the tests deterministic, free of GPU /
jax-finufft dependencies, and fast.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jax_nufft.wgridder import _auto_w_strategy


def _stub_plan(
    *,
    n_w: int,
    w_kernel_width: int = 8,
    window_padding_overhead: float = 1.0,
):
    """Minimal stand-in exposing the three fields the heuristic reads.

    The defaults match a typical eps=1e-6 plan (w_kernel_width=8, no
    windowed padding waste).
    """
    return SimpleNamespace(
        n_w=n_w,
        w_kernel_width=w_kernel_width,
        window_padding_overhead=window_padding_overhead,
    )


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_small_n_w_picks_dense_scan(is_adjoint: bool) -> None:
    """When ``n_w`` is at or just above ``w_kernel_width``, the windowed
    path has nothing to amortise over and dense_scan always wins -- on
    both the forward and adjoint."""
    plan = _stub_plan(n_w=10, w_kernel_width=8)  # 8 + 2 boundary
    assert _auto_w_strategy(plan, is_adjoint=is_adjoint) == "dense_scan"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_constant_w_fast_path_picks_dense_scan(is_adjoint: bool) -> None:
    """The constant-w fast path collapses ``n_w`` to one; that case must
    pick dense_scan (the windowed strategies have no work to do)."""
    plan = _stub_plan(n_w=1, w_kernel_width=8)
    assert _auto_w_strategy(plan, is_adjoint=is_adjoint) == "dense_scan"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_high_padding_overhead_forces_dense_scan(is_adjoint: bool) -> None:
    """If the windowed plane builder paid >5x average overhead per plane,
    the windowed savings disappear -- dense_scan even for the adjoint
    at large n_w."""
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=7.5)
    assert _auto_w_strategy(plan, is_adjoint=is_adjoint) == "dense_scan"


def test_adjoint_large_n_w_picks_windowed_scan() -> None:
    """The headline win: adjoint with n_w / w_kernel_width > 2 and no
    pathological padding -> windowed_scan."""
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=1.4)
    assert plan.n_w / plan.w_kernel_width > 2.0  # sanity
    assert _auto_w_strategy(plan, is_adjoint=True) == "windowed_scan"


def test_forward_large_n_w_stays_dense_scan() -> None:
    """The v0.1.1 forward windowed path never beats dense; the
    heuristic must not auto-pick windowed_scan for ``is_adjoint=False``
    even at large ``n_w``. (If Part 1's sorted-order forward measurably
    changes this on CPU we revisit the heuristic.)"""
    plan = _stub_plan(n_w=500, w_kernel_width=8, window_padding_overhead=1.2)
    assert _auto_w_strategy(plan, is_adjoint=False) == "dense_scan"


def test_adjoint_just_at_ratio_boundary_stays_dense_scan() -> None:
    """The ratio cutoff is strict ``>``, not ``>=``: an adjoint with
    n_w == 2 * w_kernel_width must fall through to dense_scan."""
    plan = _stub_plan(n_w=16, w_kernel_width=8, window_padding_overhead=1.0)
    # 16 / 8 == 2.0, not > 2.0
    assert _auto_w_strategy(plan, is_adjoint=True) == "dense_scan"


def test_adjoint_just_above_ratio_boundary_picks_windowed_scan() -> None:
    """Just on the windowed side of the cutoff."""
    plan = _stub_plan(n_w=17, w_kernel_width=8, window_padding_overhead=1.0)
    # 17 / 8 > 2.0
    assert _auto_w_strategy(plan, is_adjoint=True) == "windowed_scan"


def test_padding_overhead_boundary_is_strict() -> None:
    """The padding cutoff is strict ``>``: at exactly 5.0 we still take
    the next branch and (because is_adjoint with large n_w) pick
    windowed_scan."""
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=5.0)
    assert _auto_w_strategy(plan, is_adjoint=True) == "windowed_scan"
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=5.001)
    assert _auto_w_strategy(plan, is_adjoint=True) == "dense_scan"


def test_helper_not_wired_into_public_api_yet() -> None:
    """Part 4.1 ships the helper but Part 4.2 does the public wiring.
    Until then, ``_canonicalise_w_strategy`` must NOT accept ``"auto"``.

    Catches an accidental partial wire-up in this same commit.
    """
    from jax_nufft.wgridder import _canonicalise_w_strategy

    with pytest.raises(ValueError, match="unknown w_strategy"):
        _canonicalise_w_strategy("auto")
