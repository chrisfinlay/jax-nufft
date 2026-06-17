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
* :func:`snapshot_hbm` -- (Part 5.5) lightweight wrapper around
  ``jax.devices()[0].memory_stats()`` for HBM accounting in the GPU bench.

This module is import-safe on CPU (no GPU-only deps, no jax-finufft
imports at module load).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

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
        k = max(0, min(iters - 1, round((p / 100.0) * (iters - 1))))
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


def _run_cmd(args: list[str], timeout: float = 5.0) -> str | None:
    """Run a shell command, return stripped stdout or None if it failed.

    Used for ``nvidia-smi`` calls; deliberately tolerant -- a missing
    binary or a non-GPU host should not raise here.
    """
    if shutil.which(args[0]) is None:
        return None
    try:
        out = subprocess.run(
            args,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", errors="replace").strip() or None


def _nvidia_smi_list_gpus() -> list[str] | None:
    """Return parsed ``nvidia-smi -L`` lines, or None on non-GPU hosts."""
    raw = _run_cmd(["nvidia-smi", "-L"])
    if raw is None:
        return None
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _nvidia_smi_driver_version() -> str | None:
    """Driver version per nvidia-smi.

    ``--query-gpu`` returns one line per GPU. On a multi-GPU node they
    all share the same driver, so we return the first line (matching the
    v0.1.2 plan's "parse the first line" guidance). If a future system
    has heterogeneous driver versions, the per-GPU info still lives in
    ``nvidia_smi_gpus``.
    """
    raw = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    if raw is None:
        return None
    first_line = raw.splitlines()[0].strip()
    return first_line or None


def _jax_finufft_version() -> str | None:
    """Return ``jax_finufft.__version__`` if importable, else None.

    The fingerprint must work on hosts without jax_finufft installed
    (CI lint runs on CPU only and may skip the optional dep), so
    swallow ImportError silently.
    """
    try:
        import jax_finufft  # type: ignore
    except Exception:
        return None
    return getattr(jax_finufft, "__version__", None)


def capture_fingerprint(include_hostname: bool = False) -> dict[str, Any]:
    """Capture a reproducible hardware/software fingerprint.

    Designed so that running this on the same machine with the same env
    yields a byte-equal dict (modulo ``hostname`` which is opt-in). GPU-
    specific fields are recorded as ``None`` on hosts without
    ``nvidia-smi`` or jax_finufft so the fingerprint shape stays stable
    across CPU and GPU runs.

    Parameters
    ----------
    include_hostname:
        Whether to record ``socket.gethostname()`` under ``hostname``.
        Off by default so the fingerprint is comparable across machines
        with the same software stack; useful to flip on for internal
        provenance.

    Returns
    -------
    dict with the following keys (some may be ``None``):

    * ``jax_version``: ``jax.__version__``.
    * ``jax_finufft_version``: as above (or ``None``).
    * ``jax_devices``: list of ``repr(d)`` strings for ``jax.devices()``.
    * ``jax_default_platform``: ``jax.default_backend()`` (``"cpu"`` /
      ``"gpu"`` / etc.).
    * ``nvidia_smi_gpus``: parsed ``nvidia-smi -L`` lines, or ``None`` on
      non-GPU hosts.
    * ``nvidia_smi_driver_version``: parsed
      ``--query-gpu=driver_version`` output, or ``None``.
    * ``env_OMP_NUM_THREADS``, ``env_XLA_FLAGS``: relevant env vars
      (``None`` if unset).
    * ``python_version``: ``sys.version`` summary line.
    * ``platform_machine``: ``platform.machine()`` (``aarch64`` on
      GH200, ``x86_64`` on usual workstations).
    * ``hostname``: only if ``include_hostname=True``.
    """
    devices = jax.devices()
    fp: dict[str, Any] = {
        "jax_version": getattr(jax, "__version__", None),
        "jax_finufft_version": _jax_finufft_version(),
        "jax_devices": [repr(d) for d in devices],
        "jax_default_platform": jax.default_backend(),
        "nvidia_smi_gpus": _nvidia_smi_list_gpus(),
        "nvidia_smi_driver_version": _nvidia_smi_driver_version(),
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "env_XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "python_version": sys.version.splitlines()[0],
        "platform_machine": platform.machine(),
    }
    if include_hostname:
        fp["hostname"] = socket.gethostname()
    return fp


def snapshot_hbm(device: Any | None = None) -> dict[str, int] | None:
    """Return a small HBM snapshot from JAX's BFC allocator.

    Wraps ``device.memory_stats()`` and pulls the three keys the GPU bench
    needs for per-cell accounting: current live bytes, the session-wide
    peak, and the largest single allocation seen. The peak is *monotonic*
    across a process lifetime -- JAX does not expose a reset for it -- so
    consumers comparing cells should look at the *delta* between the
    pre-cell and post-cell snapshots rather than the raw post value.

    Parameters
    ----------
    device:
        A JAX device. Defaults to ``jax.devices()[0]``.

    Returns
    -------
    A dict with int keys ``bytes_in_use``, ``peak_bytes_in_use``,
    ``largest_alloc_size`` on platforms where the API is available;
    ``None`` on CPU or older JAX versions where ``memory_stats`` is
    missing / unsupported.
    """
    if device is None:
        device = jax.devices()[0]
    fn = getattr(device, "memory_stats", None)
    if fn is None:
        return None
    try:
        raw = fn()
    except Exception:
        return None
    if not raw:
        return None
    return {
        "bytes_in_use": int(raw.get("bytes_in_use", 0)),
        "peak_bytes_in_use": int(raw.get("peak_bytes_in_use", 0)),
        "largest_alloc_size": int(raw.get("largest_alloc_size", 0)),
    }


def _cli_main(argv: list[str]) -> int:
    """``python -m tests.bench_harness fingerprint`` entry point."""
    if len(argv) >= 2 and argv[1] == "fingerprint":
        include_hostname = "--hostname" in argv[2:]
        print(json.dumps(capture_fingerprint(include_hostname=include_hostname), indent=2))
        return 0
    print(
        "usage: python -m tests.bench_harness fingerprint [--hostname]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli_main(sys.argv))
