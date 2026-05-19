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


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_cpu_constant_w_fast_path_picks_dense_scan(is_adjoint: bool) -> None:
    """The constant-w fast path collapses ``n_w`` to one; that case must
    pick dense_scan (the windowed strategies have no work to do)."""
    plan = _stub_plan(n_w=1, w_kernel_width=8)
    assert _auto_w_strategy_cpu(plan, is_adjoint=is_adjoint) == "dense_scan"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_cpu_high_padding_overhead_forces_dense_scan(is_adjoint: bool) -> None:
    """If the windowed plane builder paid >5x average overhead per plane,
    the windowed savings disappear -- dense_scan even for the adjoint
    at large n_w."""
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=7.5)
    assert _auto_w_strategy_cpu(plan, is_adjoint=is_adjoint) == "dense_scan"


def test_cpu_adjoint_large_n_w_picks_windowed_scan() -> None:
    """The headline win: adjoint with n_w / w_kernel_width > 2 and no
    pathological padding -> windowed_scan."""
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=1.4)
    assert plan.n_w / plan.w_kernel_width > 2.0  # sanity
    assert _auto_w_strategy_cpu(plan, is_adjoint=True) == "windowed_scan"


def test_cpu_forward_large_n_w_stays_dense_scan() -> None:
    """The v0.1.1 forward windowed path never beats dense on CPU; the
    heuristic must not auto-pick windowed_scan for ``is_adjoint=False``
    even at large ``n_w``."""
    plan = _stub_plan(n_w=500, w_kernel_width=8, window_padding_overhead=1.2)
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
    """The padding cutoff is strict ``>``: at exactly 5.0 we still take
    the next branch and (because is_adjoint with large n_w) pick
    windowed_scan."""
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=5.0)
    assert _auto_w_strategy_cpu(plan, is_adjoint=True) == "windowed_scan"
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=5.001)
    assert _auto_w_strategy_cpu(plan, is_adjoint=True) == "dense_scan"


# -- Part 6.3: GPU heuristic ------------------------------------------------
# Cells refer to (op, fixture) rows of docs/benchmarks/v0.1.2-baseline-gpu.json.


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_gpu_small_n_w_picks_dense_vmap(is_adjoint: bool) -> None:
    """Small/constant-w plans favour dense_vmap on GPU; either choice
    is roughly equivalent at this size, picking dense keeps things
    simple. Matches the MWA_compact_zenith / EDA2_zenith cells."""
    plan = _stub_plan(n_w=7, w_kernel_width=6)
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == "dense_vmap"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_gpu_constant_w_fast_path_picks_dense_vmap(is_adjoint: bool) -> None:
    """The constant-w fast path collapses ``n_w`` to one; pick dense
    on GPU same as on CPU."""
    plan = _stub_plan(n_w=1, w_kernel_width=6)
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == "dense_vmap"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_gpu_high_padding_overhead_forces_dense_vmap(is_adjoint: bool) -> None:
    """``window_padding_overhead > 3.0`` cancels the windowed win
    even on the large-row fixture, matching the MWA_extended_off30
    cell (n_w=515)."""
    plan = _stub_plan(
        n_w=100, w_kernel_width=6, window_padding_overhead=4.0, n_rows=50_000
    )
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == "dense_vmap"


def test_gpu_adjoint_large_rows_picks_windowed_vmap() -> None:
    """Adjoint headline win on GH200: large-row plans favour windowed
    regardless of pointing. Matches GH200_large_off30 and
    GH200_large_zenith vis2dirty cells."""
    plan = _stub_plan(
        n_w=77, w_kernel_width=6, window_padding_overhead=2.5, n_rows=50_000
    )
    assert _auto_w_strategy_gpu(plan, is_adjoint=True) == "windowed_vmap"


def test_gpu_forward_large_rows_low_n_w_picks_windowed_vmap() -> None:
    """Forward on GPU only wins on windowed at low ``n_w`` (zenith-like)
    on large-row plans. Matches the GH200_large_zenith dirty2vis cell
    (n_w=13)."""
    plan = _stub_plan(
        n_w=13, w_kernel_width=6, window_padding_overhead=1.5, n_rows=50_000
    )
    assert _auto_w_strategy_gpu(plan, is_adjoint=False) == "windowed_vmap"


def test_gpu_forward_large_rows_high_n_w_stays_dense_vmap() -> None:
    """Forward at high ``n_w`` on a large plan falls back to dense_vmap;
    matches the GH200_large_off30 dirty2vis cell where n_w=77 puts the
    ratio above the forward cutoff."""
    plan = _stub_plan(
        n_w=77, w_kernel_width=6, window_padding_overhead=2.5, n_rows=50_000
    )
    assert _auto_w_strategy_gpu(plan, is_adjoint=False) == "dense_vmap"


@pytest.mark.parametrize("is_adjoint", [False, True])
def test_gpu_small_rows_picks_dense_vmap(is_adjoint: bool) -> None:
    """Below the ``n_rows`` cutoff the windowed path's per-plane
    slice-size advantage disappears; matches every non-GH200_large
    fixture in the baseline (MWA, MeerKAT, EDA2 with n_rows in
    {400, 600})."""
    plan = _stub_plan(
        n_w=200, w_kernel_width=6, window_padding_overhead=1.5, n_rows=600
    )
    assert _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint) == "dense_vmap"


def test_gpu_n_rows_boundary_is_inclusive() -> None:
    """The 10_000 ``n_rows`` cutoff is ``>=``: at exactly 10_000 we
    take the windowed branch (when other gates pass)."""
    plan = _stub_plan(
        n_w=20, w_kernel_width=6, window_padding_overhead=1.5, n_rows=10_000
    )
    assert _auto_w_strategy_gpu(plan, is_adjoint=True) == "windowed_vmap"
    plan = _stub_plan(
        n_w=20, w_kernel_width=6, window_padding_overhead=1.5, n_rows=9_999
    )
    assert _auto_w_strategy_gpu(plan, is_adjoint=True) == "dense_vmap"


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
    plan = _stub_plan(
        n_w=77, w_kernel_width=6, window_padding_overhead=2.5, n_rows=50_000
    )
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
    assert (
        _canonicalise_w_strategy("auto", plan=plan, is_adjoint=False) == expected
    )


def test_canonicalise_auto_dispatches_to_helper_adjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same check on the adjoint branch where the CPU heuristic picks
    ``windowed_scan``."""
    _patch_platform(monkeypatch, "cpu")
    plan = _stub_plan(n_w=200, w_kernel_width=8, window_padding_overhead=1.4)
    expected = _auto_w_strategy(plan, is_adjoint=True)
    assert expected == "windowed_scan"  # sanity
    assert (
        _canonicalise_w_strategy("auto", plan=plan, is_adjoint=True) == expected
    )


def test_canonicalise_auto_requires_context() -> None:
    """``"auto"`` without ``plan`` + ``is_adjoint`` is a programmer error
    (the public wrappers always supply them); raise to catch wiring
    regressions early rather than silently returning a wrong canonical
    name."""
    with pytest.raises(ValueError, match="auto.*plan.*is_adjoint"):
        _canonicalise_w_strategy("auto")
    plan = _stub_plan(n_w=10)
    with pytest.raises(ValueError, match="auto.*plan.*is_adjoint"):
        _canonicalise_w_strategy("auto", plan=plan)
    with pytest.raises(ValueError, match="auto.*plan.*is_adjoint"):
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
        (
            rng.standard_normal((n_rows, 1))
            + 1j * rng.standard_normal((n_rows, 1))
        ).astype(np.complex128)
    )
    return plan, image, vis


def test_dirty2vis_auto_matches_explicit_resolved() -> None:
    """``dirty2vis(..., w_strategy="auto")`` must produce bit-equal
    output to the call that passes the explicitly-resolved canonical
    name, and the JIT cache size must NOT grow when the auto call comes
    after the explicit one (cache hit)."""
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
    this fixture's row count), so the test asserts only the
    load-bearing semantics: bit-equality and shared JIT cache."""
    plan, _, vis = _small_offzenith_plan_and_arrays(seed=5)
    resolved = _auto_w_strategy(plan, is_adjoint=True)

    _vis2dirty_jit._clear_cache()
    out_explicit = vis2dirty(plan, vis, w_strategy=resolved)
    jax.block_until_ready(out_explicit)
    cache_after_explicit = _vis2dirty_jit._cache_size()

    out_auto = vis2dirty(plan, vis, w_strategy="auto")
    jax.block_until_ready(out_auto)
    cache_after_auto = _vis2dirty_jit._cache_size()

    np.testing.assert_array_equal(np.asarray(out_auto), np.asarray(out_explicit))
    assert cache_after_auto == cache_after_explicit, (
        f"vis2dirty(...w_strategy='auto') triggered a recompile: "
        f"cache {cache_after_explicit} -> {cache_after_auto}."
    )
