"""Performance benchmarks vs ducc0.wgridder across the four telescope configs.

Skipped by default. Run with::

    pixi run -e test pytest tests/test_benchmark_against_ducc.py \\
        --runbench -q --benchmark-group-by=param

The ``--benchmark-group-by=param`` flag tells pytest-benchmark to group
``test_jax_*`` and ``test_ducc_*`` tests with the same telescope ID into a
single comparison table, so each (telescope, operator) appears as one row of
"jax-nufft" and one row of "ducc0" with a Min / Mean / OPS column.

Notes on what the numbers do and don't mean:

* CI runner numbers are noisy and not directly comparable to a quiet
  workstation. Treat absolute numbers as ballparks.
* The first JAX call compiles; the benchmark runs a manual warmup so this
  doesn't pollute the median.
* Both implementations are run single-threaded by default to keep the
  comparison apples-to-apples on the wall-clock dimension.
* These are *small* telescope configs (64 - 256 pixels). Real workloads at
  1k - 4k pixels behave qualitatively the same but absolute timings differ.
"""

from __future__ import annotations

import ducc0.wgridder
import jax
import jax.numpy as jnp
import numpy as np

from jax_nufft import dirty2vis, make_plan, vis2dirty
from tests.conftest import Telescope, synthetic_uvw

jax.config.update("jax_enable_x64", True)

# Single epsilon for the bench matrix; 1e-6 is the most common operating
# point and gives a representative kernel width / Nw for both libraries.
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


def test_bench_jax_dirty2vis(benchmark, bench_telescope_pointing: tuple[Telescope, float]) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, _, _, _, image, _, plan = _setup_problem(tel, zen_deg)
    image_j = jnp.asarray(image)

    # Warm up so the JIT compile time is not in the timed window.
    dirty2vis(plan, image_j).block_until_ready()

    benchmark.extra_info["telescope"] = tel.name
    benchmark.extra_info["n_pix"] = tel.n_pix
    benchmark.extra_info["n_rows"] = tel.n_rows
    benchmark.extra_info["n_w"] = plan.n_w
    benchmark(lambda: dirty2vis(plan, image_j).block_until_ready())


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


def test_bench_jax_vis2dirty(benchmark, bench_telescope_pointing: tuple[Telescope, float]) -> None:
    tel, zen_deg = bench_telescope_pointing
    _, _, _, _, _, vis, plan = _setup_problem(tel, zen_deg)
    vis_j = jnp.asarray(vis)

    vis2dirty(plan, vis_j).block_until_ready()

    benchmark.extra_info["telescope"] = tel.name
    benchmark.extra_info["n_pix"] = tel.n_pix
    benchmark.extra_info["n_rows"] = tel.n_rows
    benchmark.extra_info["n_w"] = plan.n_w
    benchmark(lambda: vis2dirty(plan, vis_j).block_until_ready())


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
