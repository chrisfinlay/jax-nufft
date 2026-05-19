"""Unit + integration tests for the ``w_strategy="auto"`` machinery.

Part 4.1 covers :func:`jax_nufft.wgridder._auto_w_strategy` in isolation
with synthetic plan objects (every branch of the CPU-derived heuristic).
Part 4.2 covers :func:`jax_nufft.wgridder._canonicalise_w_strategy`'s
``"auto"`` handling and the public-wrapper wiring -- that ``dirty2vis``
and ``vis2dirty`` resolve ``"auto"`` *before* the JIT boundary so the
JIT cache is shared with the explicit canonical caller.

The heuristic helper only reads three fields off the plan (``n_w``,
``w_kernel_width``, ``window_padding_overhead``), so for the Part 4.1
unit tests we build lightweight ``SimpleNamespace`` stand-ins rather
than running ``make_plan``. The Part 4.2 integration test uses a small
real plan so it covers the public API end-to-end.
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
    _canonicalise_w_strategy,
    _dirty2vis_jit,
    _vis2dirty_jit,
)


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


# -- Part 4.2: _canonicalise_w_strategy("auto") wiring ----------------------


def test_canonicalise_auto_dispatches_to_helper_forward() -> None:
    """``_canonicalise_w_strategy("auto", plan, is_adjoint=False)`` must
    return exactly what ``_auto_w_strategy(plan, is_adjoint=False)``
    would. Use a stub plan in the dense-scan branch (smallest dependency)."""
    plan = _stub_plan(n_w=10, w_kernel_width=8)
    expected = _auto_w_strategy(plan, is_adjoint=False)
    assert (
        _canonicalise_w_strategy("auto", plan=plan, is_adjoint=False) == expected
    )


def test_canonicalise_auto_dispatches_to_helper_adjoint() -> None:
    """Same check on the adjoint branch where the heuristic picks
    ``windowed_scan``."""
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
    """Build a small off-zenith plan where the auto heuristic actively
    resolves to a non-default strategy on the adjoint.

    Parameters chosen so ``n_w`` is well above ``2 * w_kernel_width``
    (the heuristic picks ``windowed_scan`` on the adjoint, ``dense_scan``
    on the forward) while staying fast enough to run on CPU. Empirically
    this fixture yields n_w ~100 at w_kernel_width=6 (eps=1e-6).
    """
    rng = np.random.default_rng(seed)
    n_rows = 192
    # Wide (u, v) baselines + a wide w-extent gives a large n_w on a
    # moderately-sized image. pixsize_*=4e-3 with a 64x64 image keeps the
    # FOV broad enough that the wgridder allocates ~100 w-planes.
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

    # Bit-equality: the canonical name fed into JIT is identical, so the
    # compiled binary is shared and there is no numerical drift.
    np.testing.assert_array_equal(np.asarray(out_auto), np.asarray(out_explicit))
    assert cache_after_auto == cache_after_explicit, (
        f"dirty2vis(...w_strategy='auto') triggered a recompile: "
        f"cache {cache_after_explicit} -> {cache_after_auto}. "
        f"'auto' must be resolved before the JIT boundary."
    )


def test_vis2dirty_auto_matches_explicit_resolved() -> None:
    """Same as above on the adjoint, where the heuristic actually picks
    a different canonical name (``windowed_scan`` on this plan)."""
    plan, _, vis = _small_offzenith_plan_and_arrays(seed=5)
    resolved = _auto_w_strategy(plan, is_adjoint=True)
    # Smoke check that the integration test is exercising the branch we
    # care about; if this ever falls back to dense_scan, pick a fixture
    # that triggers windowed.
    assert resolved == "windowed_scan", (
        f"expected windowed_scan for this off-zenith fixture, got {resolved!r}"
    )

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
