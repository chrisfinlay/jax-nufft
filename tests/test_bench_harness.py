"""Unit tests for :mod:`tests.bench_harness`.

These are platform-agnostic: the harness has to behave correctly on CPU
because that's the only place CI runs by default. The GPU-specific
behaviour (``sync=True`` actually waiting on kernels) is exercised
implicitly by tests/test_benchmark_gpu.py.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import pytest

from tests.bench_harness import time_jax_callable


def _double(x):
    return jnp.sin(x) * 2.0


def test_time_jax_callable_basic_keys() -> None:
    x = jnp.linspace(0.0, 1.0, 1024)
    fn = jax.jit(_double)
    res = time_jax_callable(fn, x, warmup=2, iters=5)
    for key in (
        "median_s",
        "min_s",
        "p05_s",
        "p95_s",
        "mean_s",
        "stdev_s",
        "cv",
        "iters",
        "warmup",
        "sync",
        "samples_s",
    ):
        assert key in res, f"missing key {key!r}"
    assert res["iters"] == 5
    assert res["warmup"] == 2
    assert res["sync"] is True
    assert len(res["samples_s"]) == 5
    assert res["median_s"] > 0.0
    assert res["min_s"] <= res["median_s"] <= res["p95_s"]
    assert res["p05_s"] <= res["median_s"]
    # cv = stdev / mean; both >= 0, so cv >= 0 and finite.
    assert res["cv"] >= 0.0
    assert res["cv"] < float("inf")


def test_time_jax_callable_excludes_warmup_cost() -> None:
    """If the first call is artificially slow (a Python-side timer
    pretending to be a JIT compile), the timed iterations should not pay
    that cost. We stub fn with a counter that sleeps only the first
    ``warmup`` times to simulate compile-amortised behaviour.
    """
    call_count = {"n": 0}
    warmup = 3

    def slow_first_calls(x):
        call_count["n"] += 1
        if call_count["n"] <= warmup:
            time.sleep(0.01)  # 10 ms "compile"
        return x * 2.0

    x = jnp.array([1.0, 2.0, 3.0])
    # sync=False -- callable returns a jax array but we don't need block;
    # the slow_first_calls sleeps in Python, so the cost is captured by
    # time.perf_counter regardless.
    res = time_jax_callable(slow_first_calls, x, warmup=warmup, iters=10, sync=False)

    # All 10 timed iterations should be fast (no sleep): median << 1 ms.
    assert res["median_s"] < 1e-3, (
        f"timed iterations include warmup cost? median={res['median_s']:.4f}s"
    )
    # Sanity: stub really did sleep `warmup` times.
    assert call_count["n"] == warmup + 10


def test_time_jax_callable_iters_one_is_safe() -> None:
    """A single-iter call must not blow up on the stdev computation."""
    res = time_jax_callable(lambda: jnp.zeros(4), warmup=0, iters=1)
    assert res["iters"] == 1
    assert res["stdev_s"] == 0.0
    assert res["cv"] == 0.0
    assert res["median_s"] == res["min_s"] == res["p05_s"] == res["p95_s"]


def test_time_jax_callable_validates_inputs() -> None:
    with pytest.raises(ValueError, match="iters must be >= 1"):
        time_jax_callable(lambda x: x, jnp.zeros(2), iters=0)
    with pytest.raises(ValueError, match="warmup must be >= 0"):
        time_jax_callable(lambda x: x, jnp.zeros(2), warmup=-1)


def test_time_jax_callable_pytree_output_syncs() -> None:
    """``sync=True`` must work when fn returns a pytree (tuple/dict),
    not just a single array. jax.block_until_ready walks pytrees, so the
    harness should not raise.
    """

    @jax.jit
    def f(x):
        return {"sin": jnp.sin(x), "cos": jnp.cos(x)}

    x = jnp.linspace(0, 1, 64)
    res = time_jax_callable(f, x, warmup=1, iters=3, sync=True)
    assert res["iters"] == 3
    assert res["sync"] is True
