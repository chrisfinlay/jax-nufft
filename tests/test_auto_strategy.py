"""Unit + integration tests for the ``w_strategy="auto"`` machinery.

Part 4.1 covers :func:`jax_nufft.wgridder._auto_w_strategy_cpu` in
isolation with synthetic plan objects (every branch of the CPU-tuned
heuristic).
Part 4.2 covers :func:`jax_nufft.wgridder._canonicalise_w_strategy`'s
``"auto"`` handling and the public-wrapper wiring -- that ``dirty2vis``
and ``vis2dirty`` resolve ``"auto"`` *before* the JIT boundary so the
JIT cache is shared with the explicit canonical caller.
Part 6.3 covers :func:`jax_nufft.wgridder._auto_w_strategy_gpu` (the
GH200-tuned heuristic) and the platform dispatch in
:func:`jax_nufft.wgridder._auto_w_strategy`.

The heuristic helpers only read a handful of fields off the plan
(``n_w``, ``w_kernel_width``, ``window_padding_overhead`` -- plus
``n_rows`` on the GPU branch), so for the unit tests we build
lightweight ``SimpleNamespace`` stand-ins rather than running
``make_plan``. The integration tests use a small real plan so they
cover the public API end-to-end.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft.wgridder import (
    _CPU_PADDING_CUTOFF,
    _GPU_LARGE_N_ROWS,
    _auto_w_strategy,
    _auto_w_strategy_cpu,
    _auto_w_strategy_gpu,
    _canonicalise_w_strategy,
    _dirty2vis_jit,
    _vis2dirty_jit,
)


def _stub_plan(
    *,
    n_w: int,
    w_kernel_width: int = 8,
    window_padding_overhead: float = 1.0,
    n_rows: int = 600,
):
    """Minimal stand-in exposing the fields both heuristics read.

    The defaults match a typical eps=1e-6 plan (w_kernel_width=8, no
    windowed padding waste) on a small-row fixture. The GPU branch also
    reads ``n_rows``; CPU branch ignores it.
    """
    return SimpleNamespace(
        n_w=n_w,
        w_kernel_width=w_kernel_width,
        window_padding_overhead=window_padding_overhead,
        n_rows=n_rows,
    )


class _FakeDevice:
    """Stand-in for ``jax.devices()[0]`` whose only contract is
    ``.platform``. Used to make platform-dispatch tests deterministic
    regardless of where pytest is running."""

    def __init__(self, platform: str) -> None:
        self.platform = platform


def _patch_platform(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """Patch :func:`jax.devices` so :func:`_auto_w_strategy`'s
    ``jax.devices()[0].platform`` lookup returns ``platform``."""
    monkeypatch.setattr(jax, "devices", lambda *a, **kw: [_FakeDevice(platform)])


# -- Part 4.1: CPU heuristic ------------------------------------------------


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_cpu_small_n_w_picks_dense_scan(is_adjoint: bool) -> None:
    """When ``n_w`` is at or just above ``w_kernel_width``, the windowed
    path has nothing to amortise over and dense_scan always wins -- on
    both the forward and adjoint."""
    plan = _stub_plan(n_w=10, w_kernel_width=8)  # 8 + 2 boundary
    assert _auto_w_strategy_cpu(plan, is_adjoint=is_adjoint) == "dense_scan"


def test_cpu_small_n_w_gate_stops_at_plus_two() -> None:
    """The small-``n_w`` gate is ``n_w <= w_kernel_width + 2``, and the
    ``+ 2`` is load-bearing: at ``+ 3`` the adjoint must already be out of
    the gate and free to pick ``windowed_scan``.

    The case above pins the inside of the gate but not its edge, because at
    the stub's default ``w_kernel_width=8`` the two are indistinguishable:
    ``n_w = 11`` clears a ``+ 2`` gate only to fail the ratio test
    (``11 / 8 = 1.375``, not ``> 2``) and land on ``dense_scan`` anyway. The
    gate's width is observable only where the ratio test would *pass*, which
    needs ``w_kernel_width + 3 > 2 * w_kernel_width``, i.e. ``W < 3``.

    ``W = 2`` is not a contrived value: ``kernel_params(1e-1)`` returns
    width 2, so this is the regime a real plan at a very loose ``epsilon``
    lands in. At ``W = 2, n_w = 5`` the ratio is ``2.5 > 2``, so the adjoint
    pick is ``windowed_scan`` -- and a gate widened by one to ``+ 3`` would
    swallow it and return ``dense_scan`` instead. Without this case that
    off-by-one passes the whole suite.
    """
    plan = _stub_plan(n_w=5, w_kernel_width=2, window_padding_overhead=1.0)
    assert plan.n_w == plan.w_kernel_width + 3  # sanity: one past the gate
    assert plan.n_w / plan.w_kernel_width > 2.0  # sanity: the ratio test passes
    assert _auto_w_strategy_cpu(plan, is_adjoint=True) == "windowed_scan"
    # The forward has no windowed win to claim at any n_w, so it stays dense
    # either side of the gate; asserted so this case does not read as if the
    # gate were the only thing keeping it there.
    assert _auto_w_strategy_cpu(plan, is_adjoint=False) == "dense_scan"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_cpu_constant_w_fast_path_picks_dense_scan(is_adjoint: bool) -> None:
    """The constant-w fast path collapses ``n_w`` to one; that case must
    pick dense_scan (the windowed strategies have no work to do)."""
    plan = _stub_plan(n_w=1, w_kernel_width=8)
    assert _auto_w_strategy_cpu(plan, is_adjoint=is_adjoint) == "dense_scan"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_cpu_high_padding_overhead_forces_dense_scan(is_adjoint: bool) -> None:
    """Past the padding cutoff the windowed savings disappear -- dense_scan
    even for the adjoint at large n_w.

    Stated relative to ``_CPU_PADDING_CUTOFF`` rather than at a literal
    overhead: issue #43 moved the cutoff (the corrected metric reads on a
    different scale), and pinning a number here would only re-encode
    whichever one shipped last.
    """
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=_CPU_PADDING_CUTOFF * 1.5)
    assert _auto_w_strategy_cpu(plan, is_adjoint=is_adjoint) == "dense_scan"


def test_cpu_adjoint_large_n_w_picks_windowed_scan() -> None:
    """The headline win: adjoint with n_w / w_kernel_width > 2 and no
    pathological padding -> windowed_scan."""
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=1.4)
    assert plan.n_w / plan.w_kernel_width > 2.0  # sanity
    assert _auto_w_strategy_cpu(plan, is_adjoint=True) == "windowed_scan"


# ``(n_w, w_kernel_width, window_padding_overhead)`` points spanning the CPU
# forward space, every one of them chosen so the *adjoint* at the same point
# picks ``windowed_scan``: ratio above 2, padding under the cutoff, n_w past
# the small-n_w gate. That is what makes them a test of the forward rule
# rather than of the gates the two directions share -- the assertion below
# pins that the direction alone decides.
_CPU_FORWARD_POINTS = [
    (500, 8, 1.2),
    (200, 8, 1.4),
    (17, 8, 1.0),  # just past the ratio cutoff
    (50, 5, 5.9),  # padding just under _CPU_PADDING_CUTOFF
    (1000, 2, 1.0),  # narrow kernel, very large n_w
]


@pytest.mark.parametrize(
    ("n_w", "w_kernel_width", "padding"),
    _CPU_FORWARD_POINTS,
    ids=[f"n_w{n}-W{w}-pad{p}" for n, w, p in _CPU_FORWARD_POINTS],
)
def test_cpu_forward_large_n_w_stays_dense_scan(
    n_w: int, w_kernel_width: int, padding: float
) -> None:
    """The v0.1.1 forward windowed path never beats dense on CPU; the
    heuristic must not auto-pick windowed_scan for ``is_adjoint=False``
    even at large ``n_w``.

    "Never" is a claim about the whole forward space, so this enumerates
    several points in it rather than one. Each point is paired with an
    adjoint assertion that *does* come out ``windowed_scan``: without that,
    a mutant that dropped the ``is_adjoint`` term from the windowed gate
    would still pass, because every point would be failing some other gate.
    """
    plan = _stub_plan(n_w=n_w, w_kernel_width=w_kernel_width, window_padding_overhead=padding)
    assert _auto_w_strategy_cpu(plan, is_adjoint=True) == "windowed_scan", (
        "fixture precondition: the adjoint must pick windowed_scan here, or the "
        "forward assertion below is not testing the direction rule"
    )
    assert _auto_w_strategy_cpu(plan, is_adjoint=False) == "dense_scan"


def test_cpu_adjoint_just_at_ratio_boundary_stays_dense_scan() -> None:
    """The ratio cutoff is strict ``>``, not ``>=``: an adjoint with
    n_w == 2 * w_kernel_width must fall through to dense_scan."""
    plan = _stub_plan(n_w=16, w_kernel_width=8, window_padding_overhead=1.0)
    assert _auto_w_strategy_cpu(plan, is_adjoint=True) == "dense_scan"


def test_cpu_adjoint_just_above_ratio_boundary_picks_windowed_scan() -> None:
    """Just on the windowed side of the cutoff."""
    plan = _stub_plan(n_w=17, w_kernel_width=8, window_padding_overhead=1.0)
    assert _auto_w_strategy_cpu(plan, is_adjoint=True) == "windowed_scan"


def test_cpu_padding_overhead_boundary_is_strict() -> None:
    """The padding cutoff is strict ``>``: at exactly the cutoff we still take
    the next branch and (because is_adjoint with large n_w) pick
    windowed_scan."""
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=_CPU_PADDING_CUTOFF)
    assert _auto_w_strategy_cpu(plan, is_adjoint=True) == "windowed_scan"
    plan = _stub_plan(
        n_w=200, w_kernel_width=8, window_padding_overhead=_CPU_PADDING_CUTOFF + 0.001
    )
    assert _auto_w_strategy_cpu(plan, is_adjoint=True) == "dense_scan"


# -- Part 6.3: GPU heuristic ------------------------------------------------
# Cells refer to (op, fixture) rows of docs/benchmarks/v0.1.2-baseline-gpu.json.


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_gpu_small_n_w_picks_dense_vmap(is_adjoint: bool) -> None:
    """Small/constant-w plans favour dense_vmap on GPU; either choice
    is roughly equivalent at this size, picking dense keeps things
    simple. Matches the MWA_compact_zenith / EDA2_zenith cells.

    ``_stub_plan``'s default ``n_rows`` is 600, i.e. below
    ``_GPU_LARGE_N_ROWS``, and since issue #17 that is load-bearing rather
    than incidental -- see
    ``test_gpu_large_rows_do_not_take_the_small_n_w_shortcut``."""
    plan = _stub_plan(n_w=7, w_kernel_width=6)
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == "dense_vmap"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_gpu_constant_w_fast_path_picks_dense_vmap(is_adjoint: bool) -> None:
    """The constant-w fast path collapses ``n_w`` to one; pick dense
    on GPU same as on CPU.

    On a *small-row* plan (``_stub_plan``'s 600). A constant-w plan with a
    large row count falls through to ``windowed_vmap`` since issue #17, which
    is degenerate rather than a different answer --
    ``test_gpu_constant_w_large_rows_falls_through_to_a_degenerate_window``
    is where that is established."""
    plan = _stub_plan(n_w=1, w_kernel_width=6)
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == "dense_vmap"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_gpu_high_padding_overhead_forces_dense_vmap(is_adjoint: bool) -> None:
    """``window_padding_overhead > 3.0`` cancels the windowed win
    even on the large-row fixture, matching the MWA_extended_off30
    cell (n_w=515)."""
    plan = _stub_plan(n_w=100, w_kernel_width=6, window_padding_overhead=4.0, n_rows=50_000)
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == "dense_vmap"


# The two pointings the "regardless of pointing" claim is about: a
# zenith-like low ``n_w`` and an off-zenith-like high ``n_w``, matching the
# GH200_large_zenith and GH200_large_off30 vis2dirty cells. The forward
# splits between these two (see the two forward tests below); the adjoint
# must not.
_GPU_LARGE_ROW_POINTINGS = [(13, 1.5), (77, 2.5)]


@pytest.mark.parametrize(
    ("n_w", "padding"), _GPU_LARGE_ROW_POINTINGS, ids=["zenith_like", "off30_like"]
)
def test_gpu_adjoint_large_rows_picks_windowed_vmap(n_w: int, padding: float) -> None:
    """Adjoint headline win on GH200: large-row plans favour windowed
    regardless of pointing. Matches GH200_large_off30 and
    GH200_large_zenith vis2dirty cells.

    "Regardless of pointing" needs both pointings to say anything, so both
    are parametrised. The high-``n_w`` point is the load-bearing one: it is
    exactly where the *forward* falls back to ``dense_vmap``, so a mutant
    that applied the forward's ``n_w`` ratio cutoff to the adjoint as well
    would pass on the zenith-like point alone.
    """
    plan = _stub_plan(n_w=n_w, w_kernel_width=6, window_padding_overhead=padding, n_rows=50_000)
    assert _auto_w_strategy_gpu(plan, is_adjoint=True) == "windowed_vmap"


def test_gpu_forward_large_rows_low_n_w_picks_windowed_vmap() -> None:
    """Forward on GPU only wins on windowed at low ``n_w`` (zenith-like)
    on large-row plans. Matches the GH200_large_zenith dirty2vis cell
    (n_w=13)."""
    plan = _stub_plan(n_w=13, w_kernel_width=6, window_padding_overhead=1.5, n_rows=50_000)
    assert _auto_w_strategy_gpu(plan, is_adjoint=False) == "windowed_vmap"


def test_gpu_forward_large_rows_high_n_w_stays_dense_vmap() -> None:
    """Forward at high ``n_w`` on a large plan falls back to dense_vmap;
    matches the GH200_large_off30 dirty2vis cell where n_w=77 puts the
    ratio above the forward cutoff."""
    plan = _stub_plan(n_w=77, w_kernel_width=6, window_padding_overhead=2.5, n_rows=50_000)
    assert _auto_w_strategy_gpu(plan, is_adjoint=False) == "dense_vmap"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_gpu_small_rows_picks_dense_vmap(is_adjoint: bool) -> None:
    """Below the ``n_rows`` cutoff the windowed path's per-plane
    slice-size advantage disappears; matches every non-GH200_large
    fixture in the baseline (MWA, MeerKAT, EDA2 with n_rows in
    {400, 600})."""
    plan = _stub_plan(n_w=200, w_kernel_width=6, window_padding_overhead=1.5, n_rows=600)
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == "dense_vmap"


def test_gpu_n_rows_boundary_is_inclusive() -> None:
    """The 10_000 ``n_rows`` cutoff is ``>=``: at exactly 10_000 we
    take the windowed branch (when other gates pass)."""
    plan = _stub_plan(n_w=20, w_kernel_width=6, window_padding_overhead=1.5, n_rows=10_000)
    assert _auto_w_strategy_gpu(plan, is_adjoint=True) == "windowed_vmap"
    plan = _stub_plan(n_w=20, w_kernel_width=6, window_padding_overhead=1.5, n_rows=9_999)
    assert _auto_w_strategy_gpu(plan, is_adjoint=True) == "dense_vmap"


# The small-``n_w`` shortcut must not override the size-based windowed gates.
# ``n_w`` is swept over the whole range the shortcut covers -- its boundary
# ``W + 2``, one below it, and the ``n_w == 1`` floor -- because the regression
# this guards against arrived when a real fixture's ``n_w`` landed *on* the
# boundary, and a test at an interior point alone would not have caught the
# off-by-one variant of the fix.
_GPU_SHORTCUT_N_W = [8, 7, 1]


@pytest.mark.parametrize("n_w", _GPU_SHORTCUT_N_W, ids=lambda n: f"n_w{n}")
@pytest.mark.parametrize("is_adjoint", [False, True], ids=["forward", "adjoint"])
def test_gpu_large_rows_do_not_take_the_small_n_w_shortcut(n_w: int, is_adjoint: bool) -> None:
    """A large-row plan with small ``n_w`` reaches the windowed branches.

    Issue #17 made this reachable and a GH200 measurement made it a bug. The
    fold halves ``n_w``, which put ``GH200_large`` at zenith -- 50k rows,
    2048^2 -- onto exactly ``w_kernel_width + 2`` at every epsilon of the
    calibration grid, so the small-``n_w`` shortcut started firing on it and
    answered ``dense_vmap``. Timed on a GH200 against ``windowed_vmap``
    (median of 9): 27.5 vs 16.1 ms forward and 33.4 vs 17.1 ms adjoint at
    eps 1e-6, 28.0 vs 20.4 and 40.9 vs 26.2 ms at eps 1e-9 -- 1.37x to 1.95x
    slower. The shortcut's "either dense or windowed is fine" premise was
    measured on plans of a few hundred rows and is false in the large-row
    regime, which is precisely the regime the two gates it was pre-empting
    exist for.

    What this pins is the *condition*, and the control below -- the same plan
    one row under the cutoff -- pins that the small-row answer is still
    ``dense_vmap``. Note what that control does and does not catch: with the
    row condition in place the shortcut is redundant (both windowed returns
    are themselves gated on ``n_rows >= _GPU_LARGE_N_ROWS``, so a small-row
    plan reaches ``dense_vmap`` either way), so deleting the shortcut outright
    is a semantically *equivalent* program and passes here, as it should. The
    bug this file can and does catch is the condition going missing, which
    changes six answers below.
    """
    large = _stub_plan(
        n_w=n_w, w_kernel_width=6, window_padding_overhead=1.5, n_rows=_GPU_LARGE_N_ROWS
    )
    assert large.n_w <= large.w_kernel_width + 2, "the fixture must be inside the shortcut's range"
    assert _auto_w_strategy_gpu(large, is_adjoint=is_adjoint) == "windowed_vmap", (
        f"a {large.n_rows}-row plan with n_w={n_w} (W={large.w_kernel_width}) took the "
        "small-n_w shortcut to dense_vmap. Measured on a GH200, that pick is 1.37-1.95x "
        "slower than windowed_vmap on exactly this shape of plan; the shortcut is only "
        "valid below _GPU_LARGE_N_ROWS"
    )

    small = _stub_plan(
        n_w=n_w, w_kernel_width=6, window_padding_overhead=1.5, n_rows=_GPU_LARGE_N_ROWS - 1
    )
    assert _auto_w_strategy_gpu(small, is_adjoint=is_adjoint) == "dense_vmap", (
        "below the row cutoff the answer must stay dense_vmap -- it is what the baseline "
        "sweep measures for every small-row fixture in it, and the fix is to condition the "
        "shortcut on the row count, not to move the small-row answer"
    )


def test_gpu_constant_w_large_rows_falls_through_to_a_degenerate_window() -> None:
    """...and where a constant-w large-row plan lands is degenerate, not a retune.

    The row-count condition above means ``n_w == 1`` no longer short-circuits
    on a large-row plan either, so the constant-w fast path now resolves to
    ``windowed_vmap`` there. That is safe for a structural reason rather than a
    measured one, and the structure is what this asserts: the fast path sets
    ``max_window_size == n_rows`` with every ``window_start`` at zero, so the
    single plane's "window" is the entire row set and the windowed traversal
    performs the dense traversal's one NUFFT over the same rows. The two
    strategies are therefore the same work, which the output comparison
    confirms rather than assumes.

    A real ``make_plan`` plan, not a stub, because the claim is about what the
    constant-w branch builds. ``n_rows`` is exactly ``_GPU_LARGE_N_ROWS`` and
    the image is small, so the two transforms cost milliseconds.
    """
    n_rows = _GPU_LARGE_N_ROWS
    rng = np.random.default_rng(1717)
    uvw = np.zeros((n_rows, 3))
    uvw[:, :2] = rng.uniform(-200.0, 200.0, size=(n_rows, 2))
    uvw[:, 2] = 250.0  # constant w, and non-zero so the w-phase is real work
    plan = make_plan(uvw, np.array([1.4e9]), (32, 32), 3e-3, 3e-3, epsilon=1e-6)

    assert plan.is_constant_w and plan.n_w == 1, "the fixture must take the constant-w fast path"
    assert plan.n_rows >= _GPU_LARGE_N_ROWS

    for is_adjoint in (False, True):
        assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == "windowed_vmap"

    # The window is every row: that is what makes the pick a no-op.
    assert plan.max_window_size == plan.n_rows
    assert np.all(np.asarray(plan.window_start) == 0)

    image = jnp.asarray(rng.standard_normal((32, 32)))
    vis = jnp.asarray(
        (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
            np.complex128
        )
    )
    np.testing.assert_allclose(
        np.asarray(dirty2vis(plan, image, w_strategy="windowed_vmap")),
        np.asarray(dirty2vis(plan, image, w_strategy="dense_vmap")),
        rtol=1e-13,
        atol=0,
        err_msg="windowed and dense disagree on a single-plane plan, so the window is not "
        "in fact spanning every row",
    )
    np.testing.assert_allclose(
        np.asarray(vis2dirty(plan, vis, w_strategy="windowed_vmap")),
        np.asarray(vis2dirty(plan, vis, w_strategy="dense_vmap")),
        rtol=1e-13,
        atol=0,
    )


# -- Part 6.3: platform dispatcher ------------------------------------------


def test_dispatcher_routes_to_cpu_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_auto_w_strategy`` must dispatch to the CPU helper when
    ``jax.devices()[0].platform`` is anything other than ``"gpu"``."""
    _patch_platform(monkeypatch, "cpu")
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=1.4)
    # CPU branch picks windowed_scan here.
    assert _auto_w_strategy(plan, is_adjoint=True) == "windowed_scan"


def test_dispatcher_routes_to_gpu_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_auto_w_strategy`` must dispatch to the GPU helper when
    ``jax.devices()[0].platform == "gpu"``."""
    _patch_platform(monkeypatch, "gpu")
    plan = _stub_plan(n_w=77, w_kernel_width=6, window_padding_overhead=2.5, n_rows=50_000)
    # GPU branch picks windowed_vmap (large-row adjoint) here -- the
    # CPU branch would also pick windowed but with the scan variant, so
    # this case discriminates the dispatch.
    assert _auto_w_strategy(plan, is_adjoint=True) == "windowed_vmap"


def test_dispatcher_unknown_platform_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TPU / future platforms fall back to the CPU heuristic rather
    than failing. This means ``w_strategy="auto"`` is safe on platforms
    we haven't measured -- it picks the more conservative
    (lower-memory) variant."""
    _patch_platform(monkeypatch, "tpu")
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=1.4)
    assert _auto_w_strategy(plan, is_adjoint=True) == "windowed_scan"


# -- Part 4.2: _canonicalise_w_strategy("auto") wiring ----------------------


def test_canonicalise_auto_dispatches_to_helper_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_canonicalise_w_strategy("auto", plan, is_adjoint=False)`` must
    return exactly what ``_auto_w_strategy(plan, is_adjoint=False)``
    would. Pin to CPU branch so the test is deterministic across
    platforms."""
    _patch_platform(monkeypatch, "cpu")
    plan = _stub_plan(n_w=10, w_kernel_width=8)
    expected = _auto_w_strategy(plan, is_adjoint=False)
    assert _canonicalise_w_strategy("auto", plan=plan, is_adjoint=False) == expected


def test_canonicalise_auto_dispatches_to_helper_adjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same check on the adjoint branch where the CPU heuristic picks
    ``windowed_scan``."""
    _patch_platform(monkeypatch, "cpu")
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=1.4)
    expected = _auto_w_strategy(plan, is_adjoint=True)
    assert expected == "windowed_scan"  # sanity
    assert _canonicalise_w_strategy("auto", plan=plan, is_adjoint=True) == expected


def test_canonicalise_auto_requires_context() -> None:
    """``"auto"`` without ``plan`` + ``is_adjoint`` is a programmer error
    (the public wrappers always supply them); raise to catch wiring
    regressions early rather than silently returning a wrong canonical
    name."""
    with pytest.raises(ValueError, match=r"auto.*plan.*is_adjoint"):
        _canonicalise_w_strategy("auto")
    plan = _stub_plan(n_w=10)
    with pytest.raises(ValueError, match=r"auto.*plan.*is_adjoint"):
        _canonicalise_w_strategy("auto", plan=plan)
    with pytest.raises(ValueError, match=r"auto.*plan.*is_adjoint"):
        _canonicalise_w_strategy("auto", is_adjoint=False)


def test_canonicalise_canonical_names_still_pass_through() -> None:
    """The Part 4.2 signature change must not have regressed the
    existing canonical-name pass-through; plan + is_adjoint are
    optional for non-``"auto"`` inputs."""
    for name in ("dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"):
        assert _canonicalise_w_strategy(name) == name


# -- Part 4.2: end-to-end integration on a small real plan ------------------


def _small_offzenith_plan_and_arrays(seed: int = 0):
    """Build a small off-zenith plan where the CPU auto heuristic
    actively resolves to a non-default strategy on the adjoint.

    Empirically yields ~n_w=100 at w_kernel_width=6 (eps=1e-6). On GPU
    the heuristic picks dense_vmap (n_rows=192 is below the 10k cutoff);
    on CPU it picks windowed_scan on the adjoint. The integration test
    is platform-agnostic: whatever the heuristic resolves to, the
    explicit-vs-"auto" calls must produce bit-equal output and share a
    JIT cache entry.
    """
    rng = np.random.default_rng(seed)
    n_rows = 192
    uvw = rng.normal(scale=300.0, size=(n_rows, 3))
    uvw[:, 2] = rng.normal(scale=400.0, size=n_rows)
    freq = np.array([200e6])
    plan = make_plan(
        uvw=uvw,
        freq=freq,
        image_shape=(64, 64),
        pixsize_l=4e-3,
        pixsize_m=4e-3,
        epsilon=1e-6,
    )
    image = jnp.asarray(rng.standard_normal((64, 64)))
    vis = jnp.asarray(
        (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
            np.complex128
        )
    )
    return plan, image, vis


def test_dirty2vis_auto_matches_explicit_resolved() -> None:
    """``dirty2vis(..., w_strategy="auto")`` must resolve to the
    explicitly-resolved canonical name *before* the JIT boundary -- the
    load-bearing assertion is that the cache size does NOT grow when the
    auto call follows the explicit one (a recompile would mean the
    literal ``"auto"`` leaked into the static arg). dirty2vis is a
    type-2 NUFFT (interpolation) and is bit-reproducible, so the outputs
    are compared exactly."""
    plan, image, _ = _small_offzenith_plan_and_arrays(seed=3)
    resolved = _auto_w_strategy(plan, is_adjoint=False)

    _dirty2vis_jit._clear_cache()
    out_explicit = dirty2vis(plan, image, w_strategy=resolved)
    jax.block_until_ready(out_explicit)
    cache_after_explicit = _dirty2vis_jit._cache_size()

    out_auto = dirty2vis(plan, image, w_strategy="auto")
    jax.block_until_ready(out_auto)
    cache_after_auto = _dirty2vis_jit._cache_size()

    np.testing.assert_array_equal(np.asarray(out_auto), np.asarray(out_explicit))
    assert cache_after_auto == cache_after_explicit, (
        f"dirty2vis(...w_strategy='auto') triggered a recompile: "
        f"cache {cache_after_explicit} -> {cache_after_auto}. "
        f"'auto' must be resolved before the JIT boundary."
    )


def test_vis2dirty_auto_matches_explicit_resolved() -> None:
    """Same as above on the adjoint. The resolved canonical name is
    platform-dependent (windowed_scan on CPU, dense_vmap on GPU for
    this fixture's row count), so the test asserts the load-bearing
    semantics: shared JIT cache (no recompile). vis2dirty is a type-1
    NUFFT (parallel scatter-add) whose reduction order is not fixed
    across calls on a multithreaded CPU FINUFFT build, so the outputs
    are compared with ``allclose`` at rtol=1e-10 -- ~12x above the
    measured ~8e-12 run-to-run jitter, and far below the O(1) scale a
    mis-resolved strategy would produce."""
    plan, _, vis = _small_offzenith_plan_and_arrays(seed=5)
    resolved = _auto_w_strategy(plan, is_adjoint=True)

    _vis2dirty_jit._clear_cache()
    out_explicit = vis2dirty(plan, vis, w_strategy=resolved)
    jax.block_until_ready(out_explicit)
    cache_after_explicit = _vis2dirty_jit._cache_size()

    out_auto = vis2dirty(plan, vis, w_strategy="auto")
    jax.block_until_ready(out_auto)
    cache_after_auto = _vis2dirty_jit._cache_size()

    np.testing.assert_allclose(
        np.asarray(out_auto), np.asarray(out_explicit), rtol=1e-10, atol=1e-11
    )
    assert cache_after_auto == cache_after_explicit, (
        f"vis2dirty(...w_strategy='auto') triggered a recompile: "
        f"cache {cache_after_explicit} -> {cache_after_auto}."
    )
