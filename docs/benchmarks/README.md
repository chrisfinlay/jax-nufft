# Benchmark JSON schema (v0.1.2+)

This directory stores time-series benchmark results plus the GH200
baseline used to track v0.1.2 performance regressions. The two kinds of
JSON files here are produced by different harnesses with different
shapes; this README documents both so a downstream diff/merge script can
consume either without guessing.

## v0.1.x CPU benchmark JSON: `v0.1.2-baseline-gh200.json`, `v0.1.2-part1.json`, ...

Produced by `pytest tests/test_benchmark_against_ducc.py --runbench
--bench-pointing=both --benchmark-json=<path>`. This is the standard
`pytest-benchmark` JSON schema. Each row in `benchmarks[*]` carries:

- `name`: `test_bench_<lib>_<op>[<fixture_id>-<w_strategy>]`
- `stats`: dict with `mean`, `median`, `stddev`, `min`, `max`, `rounds`
- `extra_info`: dict with at minimum `telescope`, `n_pix`, `n_rows`; for
  jax-nufft rows also `n_w`, `w_strategy`, `max_window_size`,
  `padding_overhead`.

ducc rows omit `n_w`, `w_strategy`, `max_window_size`,
`padding_overhead` because those concepts are specific to jax-nufft's
plan.

## v0.1.2+ GPU benchmark JSON: `v0.1.2-baseline-gpu.json` (Part 5.5+)

Produced by `pytest tests/test_benchmark_gpu.py --runbench-gpu` with the
output path in `$JAX_NUFFT_BENCH_OUTPUT`. Top-level shape:

```jsonc
{
  "fingerprint": { /* capture_fingerprint() output */ },
  "rows": [ /* one entry per parametrised case */ ]
}
```

### `fingerprint` (from `tests/bench_harness.capture_fingerprint()`)

| Key                          | Type             | Notes                                   |
|------------------------------|------------------|-----------------------------------------|
| `jax_version`                | `str`            | `jax.__version__`                       |
| `jax_finufft_version`        | `str` or `null`  | `null` if jax-finufft not installed     |
| `jax_devices`                | `list[str]`      | `repr(d)` for each `jax.devices()`      |
| `jax_default_platform`       | `"cpu"` / `"gpu"`| `jax.default_backend()`                 |
| `nvidia_smi_gpus`            | `list[str]`/null | Parsed `nvidia-smi -L`; null on CPU host|
| `nvidia_smi_driver_version`  | `str` or `null`  | Parsed `--query-gpu=driver_version`     |
| `env_OMP_NUM_THREADS`        | `str` or `null`  | Env var, raw string                     |
| `env_XLA_FLAGS`              | `str` or `null`  | Env var, raw string                     |
| `python_version`             | `str`            | First line of `sys.version`             |
| `platform_machine`           | `str`            | `aarch64` on GH200; `x86_64` elsewhere  |
| `hostname` (optional)        | `str`            | Only present when run with `--hostname` |

### `rows` (from `tests/test_benchmark_gpu.py`, Part 5.4+)

Each row is a dict produced by `time_jax_callable` plus parametrisation
metadata. Stable keys (the merge script in v0.1.3+ keys joins by these):

| Key                | Type    | Description                                                       |
|--------------------|---------|-------------------------------------------------------------------|
| `op`               | `str`   | `"dirty2vis"` or `"vis2dirty"`                                    |
| `w_strategy`       | `str`   | One of the four canonical names                                   |
| `channel_strategy` | `str`   | `"scan"` or `"vmap"`                                              |
| `fixture`          | `str`   | Fixture id (e.g. `"MWA_compact_zenith"`, `"gh200_large_pointing"`)|
| `n_chan`           | `int`   | From the fixture                                                  |
| `n_rows`           | `int`   | From the fixture                                                  |
| `n_pix`            | `int`   | Image side                                                        |
| `n_w`              | `int`   | From `plan.n_w`                                                   |
| `w_kernel_width`   | `int`   | From `plan.w_kernel_width` (the spreading-kernel half-width, set by `epsilon`). Added in Part 6.1 so the auto-strategy heuristic can be validated against the JSON without re-deriving plan-internal quantities. |
| `window_padding_overhead` | `float` | From `plan.window_padding_overhead` (= `max_window_size / mean_window_size`). Added in Part 6.1; gates the windowed-vs-dense choice in the heuristic. |
| `is_constant_w`    | `bool`  | From `plan.is_constant_w`                                         |
| `median_s`         | `float` | Time-harness median seconds                                       |
| `min_s`            | `float` | Time-harness min seconds                                          |
| `p05_s`, `p95_s`   | `float` | Time-harness percentiles                                          |
| `mean_s`           | `float` | Time-harness mean                                                 |
| `stdev_s`, `cv`    | `float` | Time-harness stdev / coefficient of variation                     |
| `iters`, `warmup`  | `int`   | Time-harness configuration used                                   |
| `peak_hbm_bytes`           | `int`/null | `device.memory_stats()["peak_bytes_in_use"]` measured *after* the cell's warmup + timed iters. Monotonic across the pytest session: equal across cells whose transient did not exceed any earlier cell's. |
| `bytes_in_use_pre`         | `int`/null | Live HBM at cell start (plan + inputs resident before the op runs). Subtract from `peak_hbm_bytes` to bound this cell's transient (only meaningful when `peak_hbm_bytes > peak_bytes_in_use_pre`). |
| `peak_bytes_in_use_pre`    | `int`/null | Session peak at cell start. If `peak_hbm_bytes == peak_bytes_in_use_pre` the cell's transient was bounded by some earlier cell; if greater, this cell pushed a new high-water mark. |
| `bytes_in_use_post`        | `int`/null | Live HBM at cell end (sanity check: should usually return to roughly `bytes_in_use_pre`). |
| `largest_alloc_size_post`  | `int`/null | Largest single allocation seen on this device, from `memory_stats()`. Also monotonic. |
| `compile_s`        | `float` | First-call wall-clock minus steady-state median                   |

Result schema constraints:

- Every row has a complete set of these keys; missing data is `null`,
  not an absent key. This lets future diff scripts work with a fixed
  shape.
- The schema may **only** grow over time; existing key names and types
  are stable.
- `samples_s` (the per-iter raw timings) is recorded but excluded from
  the merge-key set since CV and percentiles capture what matters.

## Re-running the fingerprint

```sh
pixi run -e gpu python -m tests.bench_harness fingerprint > /tmp/fp.json
# add --hostname for internal provenance
```

The `__main__` block is intentionally minimal so the harness stays
import-safe and unit-testable.
