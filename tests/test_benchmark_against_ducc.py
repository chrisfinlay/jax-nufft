"""Performance + memory benchmarks vs ducc0.wgridder across the four telescope configs.

Skipped by default. Run with::

    pixi run -e test pytest tests/test_benchmark_against_ducc.py \\
        --runbench -q --benchmark-group-by=param:bench_telescope_pointing

The grouping flag bundles ``test_jax_*`` (scan, vmap) and ``test_ducc_*``
tests sharing the same telescope ID into a single comparison table.

Notes on what the numbers do and don't mean:

* CI runner numbers are noisy and not directly comparable to a quiet
  workstation. Treat absolute numbers as ballparks.
* The first JAX call compiles; the benchmark runs a manual warmup so this
  doesn't pollute the median.
* Both implementations are run single-threaded by default to keep the
  comparison apples-to-apples on the wall-clock dimension.
* These are *small* telescope configs (64 - 256 pixels). Real workloads at
  1k - 4k pixels behave qualitatively the same but absolute timings differ.
* JAX-side variants are parametrised over ``w_strategy`` so that scan vs
  vmap cost can be compared directly; channel work is single-channel so
  ``channel_strategy`` doesn't matter here.
* ``test_memory_*`` records the peak resident-set delta during one call
  via psutil polling. RSS includes everything the process holds (Python
  interpreter, JAX caches, FINUFFT buffers); the *delta* gives a useful
  comparison point for relative memory pressure between strategies.
"""

from __future__ import annotations

import threading
import time

import ducc0.wgridder
import jax
import jax.numpy as jnp
import numpy as np
import psutil
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from tests.conftest import Telescope, synthetic_uvw

jax.config.update("jax_enable_x64", True)

BENCH_EPSILON = 1e-6
BENCH_NTHREADS = 1


def _setup_problem(tel: Telescope, zen_deg: float, *, seed: int = 7):
    uvw = synthetic_uvw(tel, zen_deg, seed=seed)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    rng = np.random.default_rng(seed + 1)
    image = rng.standard_normal((tel.n_pix, tel.n_pix))
    vis = (rng.standard_normal((tel.n_rows, 1)) + 1j * rng.standard_normal((tel.n_rows, 1))).astype(
        np.complex128
    )
    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, BENCH_EPSILON)
    return tel, uvw, freq, pix, image, vis, plan


# ---------------------------------------------------------------------------
# Time benchmarks (pytest-benchmark)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("w_strategy", ["scan", "vmap"])
def test_bench_jax_dirty2vis(
    benchmark,
    bench_telescope_pointing: tuple[Telescope, float],
    w_strategy: str,
) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, _, _, _, image, _, plan = _setup_problem(tel, zen_deg)
    image_j = jnp.asarray(image)

    # Warm up so JIT compile cost is not in the timed window.
    dirty2vis(plan, image_j, w_strategy=w_strategy).block_until_ready()

    benchmark.extra_info["telescope"] = tel.name
    benchmark.extra_info["n_pix"] = tel.n_pix
    benchmark.extra_info["n_rows"] = tel.n_rows
    benchmark.extra_info["n_w"] = plan.n_w
    benchmark.extra_info["w_strategy"] = w_strategy
    benchmark(lambda: dirty2vis(plan, image_j, w_strategy=w_strategy).block_until_ready())


def test_bench_ducc_dirty2vis(benchmark, bench_telescope_pointing: tuple[Telescope, float]) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, uvw, freq, pix, image, _, _ = _setup_problem(tel, zen_deg)

    benchmark.extra_info["telescope"] = tel.name
    benchmark.extra_info["n_pix"] = tel.n_pix
    benchmark.extra_info["n_rows"] = tel.n_rows
    benchmark(
        lambda: ducc0.wgridder.dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=image,
            pixsize_x=pix,
            pixsize_y=pix,
            epsilon=BENCH_EPSILON,
            do_wgridding=True,
            divide_by_n=False,
            nthreads=BENCH_NTHREADS,
            verbosity=0,
        )
    )


@pytest.mark.parametrize("w_strategy", ["scan", "vmap"])
def test_bench_jax_vis2dirty(
    benchmark,
    bench_telescope_pointing: tuple[Telescope, float],
    w_strategy: str,
) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, _, _, _, _, vis, plan = _setup_problem(tel, zen_deg)
    vis_j = jnp.asarray(vis)

    vis2dirty(plan, vis_j, w_strategy=w_strategy).block_until_ready()

    benchmark.extra_info["telescope"] = tel.name
    benchmark.extra_info["n_pix"] = tel.n_pix
    benchmark.extra_info["n_rows"] = tel.n_rows
    benchmark.extra_info["n_w"] = plan.n_w
    benchmark.extra_info["w_strategy"] = w_strategy
    benchmark(lambda: vis2dirty(plan, vis_j, w_strategy=w_strategy).block_until_ready())


def test_bench_ducc_vis2dirty(benchmark, bench_telescope_pointing: tuple[Telescope, float]) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, uvw, freq, pix, _, vis, _ = _setup_problem(tel, zen_deg)

    benchmark.extra_info["telescope"] = tel.name
    benchmark.extra_info["n_pix"] = tel.n_pix
    benchmark.extra_info["n_rows"] = tel.n_rows
    benchmark(
        lambda: ducc0.wgridder.vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=vis,
            npix_x=tel.n_pix,
            npix_y=tel.n_pix,
            pixsize_x=pix,
            pixsize_y=pix,
            epsilon=BENCH_EPSILON,
            do_wgridding=True,
            divide_by_n=True,
            nthreads=BENCH_NTHREADS,
            verbosity=0,
        )
    )


# ---------------------------------------------------------------------------
# Memory measurement (psutil polling, separate from pytest-benchmark)
# ---------------------------------------------------------------------------

# Module-level table that test_memory_* tests append to and a session-scoped
# fixture prints at the end. Lets us emit a single tidy summary instead of
# scattering per-test prints.
_MEMORY_RESULTS: list[dict[str, object]] = []


def _poll_peak_rss(
    process: psutil.Process,
    stop: threading.Event,
    peak_holder: list[int],
    poll_interval_s: float,
) -> None:
    """Background thread body: track the peak RSS until ``stop`` is set."""
    while not stop.is_set():
        rss = process.memory_info().rss
        if rss > peak_holder[0]:
            peak_holder[0] = rss
        time.sleep(poll_interval_s)


def _peak_rss_during(fn, *, poll_interval_s: float = 0.001, n_iters: int = 5) -> int:
    """Return the peak RSS *delta* (bytes) observed across ``n_iters`` calls of ``fn``.

    Strategy: poll psutil's RSS in a background thread while ``fn`` runs;
    record peak relative to a baseline taken just before the call. Repeat
    ``n_iters`` times and return the worst-case (peak across iterations).
    The thread polls every ``poll_interval_s`` seconds; for sub-millisecond
    calls the peak may be missed, so this is mostly meaningful for problems
    that take >= a few ms to run.
    """
    import gc

    process = psutil.Process()
    worst_delta = 0
    for _ in range(n_iters):
        # Encourage the previous iteration's transient allocations to drop
        # before re-measuring.
        gc.collect()
        baseline = process.memory_info().rss
        peak_holder = [baseline]
        stop = threading.Event()
        thread = threading.Thread(
            target=_poll_peak_rss,
            args=(process, stop, peak_holder, poll_interval_s),
            daemon=True,
        )
        thread.start()
        try:
            fn()
        finally:
            stop.set()
            thread.join()
        delta = peak_holder[0] - baseline
        if delta > worst_delta:
            worst_delta = delta
    return worst_delta


@pytest.mark.parametrize("w_strategy", ["scan", "vmap"])
def test_memory_jax_dirty2vis(
    bench_telescope_pointing: tuple[Telescope, float], w_strategy: str
) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, _, _, _, image, _, plan = _setup_problem(tel, zen_deg)
    image_j = jnp.asarray(image)

    # Warm up so JIT-compiled buffers are already cached and we measure the
    # incremental peak of the call rather than compile-time allocation.
    dirty2vis(plan, image_j, w_strategy=w_strategy).block_until_ready()

    peak = _peak_rss_during(
        lambda: dirty2vis(plan, image_j, w_strategy=w_strategy).block_until_ready()
    )
    _MEMORY_RESULTS.append(
        {
            "op": "dirty2vis",
            "impl": f"jax/{w_strategy}",
            "telescope": tel.name,
            "zen_deg": zen_deg,
            "n_w": plan.n_w,
            "n_pix": tel.n_pix,
            "peak_mb": peak / (1024 * 1024),
        }
    )


def test_memory_ducc_dirty2vis(bench_telescope_pointing: tuple[Telescope, float]) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, uvw, freq, pix, image, _, _ = _setup_problem(tel, zen_deg)

    def call() -> None:
        ducc0.wgridder.dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=image,
            pixsize_x=pix,
            pixsize_y=pix,
            epsilon=BENCH_EPSILON,
            do_wgridding=True,
            divide_by_n=False,
            nthreads=BENCH_NTHREADS,
            verbosity=0,
        )

    call()
    peak = _peak_rss_during(call)
    _MEMORY_RESULTS.append(
        {
            "op": "dirty2vis",
            "impl": "ducc0",
            "telescope": tel.name,
            "zen_deg": zen_deg,
            "n_w": "-",
            "n_pix": tel.n_pix,
            "peak_mb": peak / (1024 * 1024),
        }
    )


@pytest.mark.parametrize("w_strategy", ["scan", "vmap"])
def test_memory_jax_vis2dirty(
    bench_telescope_pointing: tuple[Telescope, float], w_strategy: str
) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, _, _, _, _, vis, plan = _setup_problem(tel, zen_deg)
    vis_j = jnp.asarray(vis)

    vis2dirty(plan, vis_j, w_strategy=w_strategy).block_until_ready()
    peak = _peak_rss_during(
        lambda: vis2dirty(plan, vis_j, w_strategy=w_strategy).block_until_ready()
    )
    _MEMORY_RESULTS.append(
        {
            "op": "vis2dirty",
            "impl": f"jax/{w_strategy}",
            "telescope": tel.name,
            "zen_deg": zen_deg,
            "n_w": plan.n_w,
            "n_pix": tel.n_pix,
            "peak_mb": peak / (1024 * 1024),
        }
    )


def test_memory_ducc_vis2dirty(bench_telescope_pointing: tuple[Telescope, float]) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, uvw, freq, pix, _, vis, _ = _setup_problem(tel, zen_deg)

    def call() -> None:
        ducc0.wgridder.vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=vis,
            npix_x=tel.n_pix,
            npix_y=tel.n_pix,
            pixsize_x=pix,
            pixsize_y=pix,
            epsilon=BENCH_EPSILON,
            do_wgridding=True,
            divide_by_n=True,
            nthreads=BENCH_NTHREADS,
            verbosity=0,
        )

    call()
    peak = _peak_rss_during(call)
    _MEMORY_RESULTS.append(
        {
            "op": "vis2dirty",
            "impl": "ducc0",
            "telescope": tel.name,
            "zen_deg": zen_deg,
            "n_w": "-",
            "n_pix": tel.n_pix,
            "peak_mb": peak / (1024 * 1024),
        }
    )


@pytest.fixture(scope="module", autouse=True)
def _print_memory_summary():
    """At the end of this module's tests, print a tidy memory comparison table."""
    yield
    if not _MEMORY_RESULTS:
        return
    # Sort: telescope, op, impl
    rows = sorted(
        _MEMORY_RESULTS,
        key=lambda r: (r["telescope"], r["zen_deg"], r["op"], str(r["impl"])),
    )
    headers = ["telescope", "pointing", "op", "impl", "n_w", "peak RSS (MB)"]
    fmt_row = lambda r: [  # noqa: E731
        r["telescope"],
        f"{int(r['zen_deg'])} deg",
        r["op"],
        r["impl"],
        str(r["n_w"]),
        f"{r['peak_mb']:.1f}",
    ]
    widths = [max(len(h), max(len(fmt_row(r)[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    print()
    print("Peak RSS delta during one call (psutil-polled, 1ms granularity)")
    print(sep)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
    print(sep)
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(fmt_row(r), widths, strict=True)))
    print(sep)
