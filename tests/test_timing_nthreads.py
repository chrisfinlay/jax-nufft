"""Timing gate for issue #24 (R11, D4): the default `nthreads` must not be
dramatically slower than an explicit `nthreads=1` for `dense_scan`.

This is the issue's definition of done, made into an actual assertion
instead of a comment: with the pre-#24 default `nthreads=0`, every per-plane
FINUFFT call in `dense_scan` re-spins the whole OpenMP pool, which the issue
measured at 6-11x slower than `nthreads=1` on a 10-core Apple M-series
across its five timing fixtures. The gate here asserts <= 1.2x on one of
those fixtures (MWA_extended off30, the fixture with the largest measured
gap, so the effect is unambiguous even amid CI noise).

Wall-clock assertions are flaky on shared/noisy runners, so this module is
opt-in behind `--runtiming` (see `tests/conftest.py`), the same way
`--runbench` gates `tests/test_benchmark_against_ducc.py`. It must NOT run
in the default `pixi run -e dev pytest -q` suite. Run it explicitly with::

    pixi run -e dev pytest -q --runtiming tests/test_timing_nthreads.py -s

Timing protocol (matches AGENTS.md's / the issue's "Timing protocol"):
plan built and one warm-up call done outside the timed window, median of
>= 7 calls, each ending with `.block_until_ready()`. `time_jax_callable`
from `tests/bench_harness.py` implements exactly this discipline (its
`warmup` argument runs untimed calls first, `sync=True` blocks on every
timed call), so it is reused here rather than reinventing a timing loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan
from tests.bench_harness import time_jax_callable
from tests.conftest import MWA_EXTENDED, synthetic_uvw

jax.config.update("jax_enable_x64", True)

# Same accuracy budget as the rest of the benchmark suite
# (tests/test_benchmark_against_ducc.py::BENCH_EPSILON) -- the timing gate
# is about scheduling, not accuracy, so there is no reason to diverge.
_TIMING_EPSILON = 1e-6
# >= 1 warm-up call outside the timed window (JIT compile + FINUFFT plan
# setup), per the repo's timing protocol.
_TIMING_WARMUP = 2
# >= 7 timed calls per the repo's timing protocol; 9 for an odd median.
_TIMING_ITERS = 9
# Definition-of-done threshold from the issue.
_MAX_DEFAULT_OVER_EXPLICIT_RATIO = 1.2


@pytest.mark.runtiming
def test_dense_scan_default_nthreads_within_1_2x_of_explicit_nthreads_1() -> None:
    """dense_scan at the *default* nthreads must be within 1.2x of the same
    call with explicit nthreads=1, on MWA_extended off30 (600 rows, 256^2
    pixels) -- the fixture the issue measured at 3272ms vs 546ms (6x) on
    the review machine. `dirty2vis` is called without a `nthreads=` kwarg
    at all for the "default" arm, exercising the operator's real default
    parameter value rather than a hardcoded stand-in for it, so this test
    keeps working unchanged once the default itself changes from `0` to
    `None`.
    """
    tel = MWA_EXTENDED
    uvw = synthetic_uvw(tel, 30.0, seed=7)
    freq = np.array([tel.freq_hz])
    pix = tel.pixsize
    rng = np.random.default_rng(8)
    image = jnp.asarray(rng.standard_normal((tel.n_pix, tel.n_pix)))
    # Plan build (host-side numpy work) happens once, outside every timed
    # window below, and is shared by both arms.
    plan = make_plan(uvw, freq, (tel.n_pix, tel.n_pix), pix, pix, _TIMING_EPSILON)

    default_stats = time_jax_callable(
        lambda: dirty2vis(plan, image, w_strategy="dense_scan"),
        warmup=_TIMING_WARMUP,
        iters=_TIMING_ITERS,
        sync=True,
    )
    explicit_stats = time_jax_callable(
        lambda: dirty2vis(plan, image, w_strategy="dense_scan", nthreads=1),
        warmup=_TIMING_WARMUP,
        iters=_TIMING_ITERS,
        sync=True,
    )
    default_median = default_stats["median_s"]
    explicit_median = explicit_stats["median_s"]
    ratio = default_median / explicit_median

    print(
        f"\ndense_scan MWA_extended off30: default nthreads median="
        f"{default_median * 1e3:.2f}ms, nthreads=1 median={explicit_median * 1e3:.2f}ms, "
        f"ratio={ratio:.2f}x (gate: <= {_MAX_DEFAULT_OVER_EXPLICIT_RATIO}x)"
    )
    assert ratio <= _MAX_DEFAULT_OVER_EXPLICIT_RATIO, (
        f"dense_scan default-nthreads median={default_median * 1e3:.2f}ms is "
        f"{ratio:.2f}x the explicit-nthreads=1 median={explicit_median * 1e3:.2f}ms "
        f"(gate: <= {_MAX_DEFAULT_OVER_EXPLICIT_RATIO}x). Issue #24 requires "
        "nthreads=None to resolve to 1 for dense_scan before the JIT boundary."
    )
