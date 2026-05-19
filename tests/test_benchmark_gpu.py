"""v0.1.2 Part 5.4: GPU benchmark suite (GH200 target).

Gated by ``--runbench-gpu`` and ``jax.default_backend() == 'gpu'``; the
conftest skip-marker discipline keeps CPU CI from collecting these.

Each parametrised case runs through :func:`tests.bench_harness.time_jax_callable`
to get steady-state timings with proper async sync (the standard
pytest-benchmark harness times kernel-launch on GPU, which is meaningless).
Results accumulate in a session-scoped fixture and land in a single JSON at
session teardown:

  ``$JAX_NUFFT_BENCH_OUTPUT`` (default ``/tmp/jax_nufft_bench_gpu.json``)

JSON schema is documented in ``docs/benchmarks/README.md``. ducc rows are
intentionally omitted from v0.1.2; a future merge script joins this with
the CPU baseline JSON.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from tests.bench_harness import (
    capture_fingerprint,
    snapshot_hbm,
    time_jax_callable,
)
from tests.conftest import Telescope, synthetic_uvw


# -- session-scoped accumulator + teardown ----------------------------------

# Default warm + iter counts tuned so the GPU bench suite finishes in
# roughly 20-30 minutes even on the GH200 large fixture, while still
# giving CV < 5% for the typical bench case. Override via env if needed.
_BENCH_EPSILON = 1e-6
_BENCH_WARMUP = int(os.environ.get("JAX_NUFFT_BENCH_WARMUP", "2"))
_BENCH_ITERS = int(os.environ.get("JAX_NUFFT_BENCH_ITERS", "10"))


@pytest.fixture(scope="session")
def gpu_bench_results(request) -> list[dict[str, Any]]:
    """Collect per-case bench rows; write a single JSON at session end.

    The fixture is yielded as a plain list so each test can ``.append(...)``
    a row dict. JSON output happens in the teardown.
    """
    rows: list[dict[str, Any]] = []

    def _finalise() -> None:
        out_path = os.environ.get(
            "JAX_NUFFT_BENCH_OUTPUT", "/tmp/jax_nufft_bench_gpu.json"
        )
        payload = {
            "fingerprint": capture_fingerprint(),
            "rows": rows,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

    request.addfinalizer(_finalise)
    return rows


# -- shared setup -----------------------------------------------------------


def _make_inputs(tel: Telescope, zen_deg: float, seed: int = 0):
    uvw = synthetic_uvw(tel, zen_deg, seed=seed)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, _BENCH_EPSILON)
    rng = np.random.default_rng(seed + 1)
    image = jnp.asarray(rng.standard_normal((tel.n_pix, tel.n_pix)))
    vis = jnp.asarray(
        (
            rng.standard_normal((tel.n_rows, 1))
            + 1j * rng.standard_normal((tel.n_rows, 1))
        ).astype(np.complex128)
    )
    return tel, plan, image, vis


def _common_row_fields(
    tel: Telescope,
    zen_deg: float,
    plan,
    op: str,
    w_strategy: str,
    channel_strategy: str,
    stats: dict,
    compile_s: float,
    hbm_pre: dict[str, int] | None,
    hbm_post: dict[str, int] | None,
) -> dict[str, Any]:
    # device.memory_stats() peak_bytes_in_use is monotonic across the
    # process. We record the raw post value plus pre snapshots so
    # downstream can compute the per-cell delta:
    #   transient_bytes = peak_post - bytes_in_use_pre   (if peak_post > peak_pre)
    # When peak_post == peak_pre, this cell's transient did NOT exceed any
    # earlier cell's; the bound is (peak_pre - bytes_in_use_pre).
    def _pull(snap: dict[str, int] | None, key: str) -> int | None:
        return None if snap is None else snap[key]

    return {
        "op": op,
        "w_strategy": w_strategy,
        "channel_strategy": channel_strategy,
        "fixture": (
            f"{tel.name}_{'zenith' if zen_deg == 0.0 else f'off{int(zen_deg)}'}"
        ),
        "n_chan": int(plan.n_chan),
        "n_rows": int(plan.n_rows),
        "n_pix": int(plan.n_l),
        "n_w": int(plan.n_w),
        "w_kernel_width": int(plan.w_kernel_width),
        "window_padding_overhead": float(plan.window_padding_overhead),
        "is_constant_w": bool(plan.is_constant_w),
        "median_s": stats["median_s"],
        "min_s": stats["min_s"],
        "p05_s": stats["p05_s"],
        "p95_s": stats["p95_s"],
        "mean_s": stats["mean_s"],
        "stdev_s": stats["stdev_s"],
        "cv": stats["cv"],
        "iters": stats["iters"],
        "warmup": stats["warmup"],
        "peak_hbm_bytes": _pull(hbm_post, "peak_bytes_in_use"),
        "bytes_in_use_pre": _pull(hbm_pre, "bytes_in_use"),
        "peak_bytes_in_use_pre": _pull(hbm_pre, "peak_bytes_in_use"),
        "bytes_in_use_post": _pull(hbm_post, "bytes_in_use"),
        "largest_alloc_size_post": _pull(hbm_post, "largest_alloc_size"),
        "compile_s": compile_s,
    }


def _measure_op(op: str, plan, image, vis, w_strategy: str, channel_strategy: str):
    """Run warmup outside the timer to get a compile_s estimate, then call
    :func:`time_jax_callable` for the timed iterations.

    Returns ``(stats_dict, compile_s)``.
    """
    if op == "dirty2vis":
        def fn():
            return dirty2vis(
                plan,
                image,
                w_strategy=w_strategy,
                channel_strategy=channel_strategy,
            )
    else:
        def fn():
            return vis2dirty(
                plan,
                vis,
                w_strategy=w_strategy,
                channel_strategy=channel_strategy,
            )

    # Snapshot HBM *before* the first call so per-cell deltas exclude any
    # session-resident plan / inputs that already lived on-device. Note
    # that the plan + image/vis arrays *were* placed on device by the
    # caller before _measure_op runs, so they count as "pre-resident"
    # baseline -- bytes_in_use_pre reflects that.
    hbm_pre = snapshot_hbm()

    # Time the very first call so we can report a rough compile_s.
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    first_call_s = time.perf_counter() - t0

    stats = time_jax_callable(fn, warmup=_BENCH_WARMUP, iters=_BENCH_ITERS)
    # The first call is dominated by XLA compile + cuFINUFFT plan setup;
    # subtracting steady-state median is a coarse but plan-doc-sanctioned
    # estimate of the compile cost.
    compile_s = max(first_call_s - stats["median_s"], 0.0)
    hbm_post = snapshot_hbm()
    return stats, compile_s, hbm_pre, hbm_post


# -- the parametrised matrix ------------------------------------------------


@pytest.mark.runbench_gpu
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
@pytest.mark.parametrize("channel_strategy", ["scan", "vmap"])
def test_bench_gpu_bench_pointing(
    op: str,
    w_strategy: str,
    channel_strategy: str,
    bench_telescope_pointing: tuple[Telescope, float],
    gpu_bench_results: list[dict[str, Any]],
) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, plan, image, vis = _make_inputs(tel, zen_deg, seed=7)
    stats, compile_s, hbm_pre, hbm_post = _measure_op(
        op, plan, image, vis, w_strategy, channel_strategy
    )
    gpu_bench_results.append(
        _common_row_fields(
            tel,
            zen_deg,
            plan,
            op,
            w_strategy,
            channel_strategy,
            stats,
            compile_s,
            hbm_pre,
            hbm_post,
        )
    )


@pytest.mark.runbench_gpu
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
@pytest.mark.parametrize("channel_strategy", ["scan", "vmap"])
def test_bench_gpu_large_pointing(
    op: str,
    w_strategy: str,
    channel_strategy: str,
    gh200_large_pointing: tuple[Telescope, float],
    gpu_bench_results: list[dict[str, Any]],
) -> None:
    tel, zen_deg = gh200_large_pointing
    _, plan, image, vis = _make_inputs(tel, zen_deg, seed=11)
    stats, compile_s, hbm_pre, hbm_post = _measure_op(
        op, plan, image, vis, w_strategy, channel_strategy
    )
    gpu_bench_results.append(
        _common_row_fields(
            tel,
            zen_deg,
            plan,
            op,
            w_strategy,
            channel_strategy,
            stats,
            compile_s,
            hbm_pre,
            hbm_post,
        )
    )
