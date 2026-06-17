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

from tests.bench_harness import (
    capture_fingerprint,
    snapshot_hbm,
    time_jax_callable,
)


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


def test_capture_fingerprint_required_keys_present() -> None:
    """The fingerprint shape must be stable across CPU and GPU runs --
    GPU-only fields read as None on non-GPU hosts rather than missing
    keys -- so JSON consumers can rely on a fixed schema."""
    fp = capture_fingerprint()
    for key in (
        "jax_version",
        "jax_finufft_version",
        "jax_devices",
        "jax_default_platform",
        "nvidia_smi_gpus",
        "nvidia_smi_driver_version",
        "env_OMP_NUM_THREADS",
        "env_XLA_FLAGS",
        "python_version",
        "platform_machine",
    ):
        assert key in fp, f"missing fingerprint key {key!r}"
    # jax_version is always present (we import jax at module load).
    assert isinstance(fp["jax_version"], str)
    assert isinstance(fp["jax_devices"], list) and len(fp["jax_devices"]) >= 1
    # platform must be one of the JAX-recognised values.
    assert fp["jax_default_platform"] in {"cpu", "gpu", "tpu", "rocm"}, fp["jax_default_platform"]
    # hostname is opt-in.
    assert "hostname" not in fp


def test_capture_fingerprint_hostname_opt_in() -> None:
    fp = capture_fingerprint(include_hostname=True)
    assert "hostname" in fp
    assert isinstance(fp["hostname"], str) and fp["hostname"]


def test_capture_fingerprint_gpu_fields_consistent_with_backend() -> None:
    """If ``nvidia-smi`` is present (GPU host), it must report at least
    one GPU and the driver version; if absent (CPU-only host), both
    GPU-related fields must be None. Catches a partial-fail like
    ``nvidia-smi`` returning OK but the driver version query failing."""
    fp = capture_fingerprint()
    gpus = fp["nvidia_smi_gpus"]
    drv = fp["nvidia_smi_driver_version"]
    if gpus is None:
        assert drv is None
    else:
        assert isinstance(gpus, list) and gpus
        assert isinstance(drv, str) and drv


def test_snapshot_hbm_shape_or_none() -> None:
    """``snapshot_hbm`` must return either a stable-key dict (GPU host) or
    ``None`` (CPU host where ``device.memory_stats`` is missing). The
    schema of the dict must match the keys the GPU bench writes into the
    JSON, otherwise downstream consumers break silently."""
    snap = snapshot_hbm()
    if snap is None:
        # CPU host: confirm we are on CPU; if we report None on GPU
        # something has gone wrong in the wrapper.
        assert jax.default_backend() == "cpu", "snapshot_hbm returned None on a non-CPU backend"
        return
    assert set(snap.keys()) == {
        "bytes_in_use",
        "peak_bytes_in_use",
        "largest_alloc_size",
    }
    for k, v in snap.items():
        assert isinstance(v, int), f"{k} must be int, got {type(v).__name__}"
        assert v >= 0, f"{k} must be non-negative"


def test_snapshot_hbm_peak_is_monotonic() -> None:
    """``peak_bytes_in_use`` is documented as monotonic. If we allocate a
    bigger array than anything seen so far, the second snapshot must
    report a peak >= the first; allocating + freeing must not *decrease*
    the peak (free-then-snapshot still sees the high-water mark)."""
    snap0 = snapshot_hbm()
    if snap0 is None:
        pytest.skip("snapshot_hbm unsupported on this backend")
    # Allocate ~16 MB, force materialisation, then read again. The peak
    # must be >= snap0.peak. We deliberately do not assert equality with
    # snap0.bytes_in_use + payload size since JAX may have pre-existing
    # device buffers from earlier tests in the session.
    payload = jnp.ones((2048, 2048), dtype=jnp.float32)
    jax.block_until_ready(payload)
    snap1 = snapshot_hbm()
    assert snap1 is not None
    assert snap1["peak_bytes_in_use"] >= snap0["peak_bytes_in_use"]
    del payload
    snap2 = snapshot_hbm()
    assert snap2 is not None
    assert snap2["peak_bytes_in_use"] >= snap1["peak_bytes_in_use"]


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
