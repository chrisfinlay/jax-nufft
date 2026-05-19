"""Async-aware timing harness for jax-nufft benchmarks.

The default :mod:`pytest-benchmark` harness assumes synchronous callables,
which makes it useless for JAX on GPU: dispatch returns immediately and the
benchmark ends up timing kernel-launch latency rather than kernel
completion. This module provides a small replacement that does the
appropriate ``block_until_ready`` and warmup discipline.

Public surface
--------------

* :func:`time_jax_callable` -- time a no-arg ``fn(*args)`` callable, return
  a dict of summary statistics.
* :func:`capture_fingerprint` -- (added in Part 5.2) capture a reproducible
  hardware/software fingerprint to embed in benchmark JSON.

This module is import-safe on CPU (no GPU-only deps, no jax-finufft
imports at module load).
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import jax


def _block(out: Any) -> Any:
    """Block until ``out`` (possibly a pytree of arrays) is materialised."""
    return jax.block_until_ready(out)


def time_jax_callable(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 3,
    iters: int = 20,
    sync: bool = True,
) -> dict[str, Any]:
    """Time a JAX-returning callable with proper async sync.

    Parameters
    ----------
    fn:
        The callable. Will be invoked as ``fn(*args)`` each iteration.
    *args:
        Positional arguments to ``fn``.
    warmup:
        Number of warmup iterations whose timings are discarded. Used to
        absorb the XLA compile + cuFINUFFT plan setup that the first call
        pays.
    iters:
        Number of timed iterations.
    sync:
        If True (default), call :func:`jax.block_until_ready` on the
        callable's output every iteration so that wall-clock includes
        kernel completion (the only meaningful measure on GPU). Set to
        False only when timing a CPU-only callable whose return is known
        to be a numpy array.

    Returns
    -------
    dict with keys:
        ``median_s``, ``min_s``, ``p05_s``, ``p95_s``, ``mean_s``,
        ``stdev_s``, ``cv``, ``iters``, ``warmup``, ``sync``,
        ``samples_s`` (list of per-iter wall-clocks in seconds).

    Notes
    -----
    Statistics over ``iters`` < 2 are degenerate; the function still
    returns sensible values (``stdev_s = 0.0``, ``cv = 0.0``) so callers
    can use it for sanity smoke checks without special-casing.
    """
    if iters < 1:
        raise ValueError(f"iters must be >= 1; got {iters}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0; got {warmup}")

    for _ in range(warmup):
        out = fn(*args)
        if sync:
            _block(out)

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        if sync:
            _block(out)
        samples.append(time.perf_counter() - t0)

    samples_sorted = sorted(samples)
    median = statistics.median(samples_sorted)
    mean = statistics.fmean(samples_sorted)
    if iters >= 2:
        stdev = statistics.stdev(samples_sorted)
    else:
        stdev = 0.0
    cv = stdev / mean if mean > 0 else 0.0
    # Percentiles by nearest-rank, robust on small iter counts.
    def _pctile(p: float) -> float:
        if iters == 1:
            return samples_sorted[0]
        # 0-indexed nearest-rank.
        k = max(0, min(iters - 1, int(round((p / 100.0) * (iters - 1)))))
        return samples_sorted[k]

    return {
        "median_s": median,
        "min_s": samples_sorted[0],
        "p05_s": _pctile(5.0),
        "p95_s": _pctile(95.0),
        "mean_s": mean,
        "stdev_s": stdev,
        "cv": cv,
        "iters": iters,
        "warmup": warmup,
        "sync": sync,
        "samples_s": samples,
    }
