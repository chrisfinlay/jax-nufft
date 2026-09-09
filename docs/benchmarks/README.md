# Benchmark JSON schema

This directory stores time-series benchmark results plus the GH200
baseline used to track v0.1.2 performance regressions. There are **four**
schemas here, produced by different harnesses with different shapes; this
README documents each so a downstream diff/merge script can consume any of
them without guessing.

| file | schema | section |
|---|---|---|
| `v0.1.2-baseline-gh200.json`, `v0.1.2-part1.json` | pytest-benchmark | below |
| `v0.1.2-baseline-gpu.json` | `{fingerprint, rows}` | below |
| `v0.2.0-vs-ducc0-gh200.json` | `{schema, provenance, sizing, suites, rows}` | below |
| `v0.2.0-memory-gh200.json` | `{schema, provenance, rows, w_chunk_sweep}` | below |

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

`padding_overhead` is `plan.window_padding_overhead`, whose **definition
changed in 0.2.0** (issue #43): the denominator moved from the mean of the
padded per-plane window lengths to `plan.live_row_count`, the incidences
inside the unpadded *nominal* kernel support (a host-side count — see the
`live_row_count` field comment in `planning.py` for how far it can sit from
what the compiled operators weight, and why that does not matter here). Every
JSON file in this directory that carries a `padding_overhead` column at all
— that is, the two pytest-benchmark files and `v0.1.2-baseline-gpu.json` —
was recorded through v0.1.2 and so holds the old scale, which reads 0 – 17%
lower on the same plan. (The two `v0.2.0-*` files record no padding column,
so the question does not arise for them.) The two are not convertible after the fact — the
conversion needs the per-`(channel, plane)` window lengths, which are
plan-time locals and have never been stored — so a `padding_overhead` value
is only comparable against another recorded by the same version. Compare
timings across the boundary freely; compare this column only within a
version.

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
metadata. Stable keys (the merge script in 0.2.0+ keys joins by these):

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
| `window_padding_overhead` | `float` | From `plan.window_padding_overhead`. Added in Part 6.1; gates the windowed-vs-dense choice in the heuristic. **Scale changed in 0.2.0** (issue #43): it is now `n_chan * n_w * max_window_size / live_row_count`, where it was `max_window_size / mean_window_size` over the *padded* windows through v0.1.2. Rows recorded before 0.2.0 — including `v0.1.2-baseline-gpu.json` — hold the old scale and cannot be converted; see the note under the CPU schema above. |
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

## v0.2.0 comparison JSON: `v0.2.0-vs-ducc0-gh200.json`

Assembled from the raw sweep recorded on the GH200 (see `provenance`), and
recomputed by `tests/test_benchmark_claims.py`. Top-level shape:

```jsonc
{
  "schema": "v0.2.0-vs-ducc0",
  "provenance": { /* machine, date, epsilon, dtype, protocol, ducc0_role */ },
  "sizing":     { /* the rule the problem sizes follow, and its inputs */ },
  "suites":     { /* what each value of rows[*].suite means */ },
  "rows":       [ /* one entry per (suite, fixture, implementation) */ ]
}
```

Each row carries `suite` (`"realistic"` or `"ci_sized"`), `fixture`, `impl`
(`"jax-nufft"` or `"ducc0"`), `device`, `nthreads` (ducc0 only; `null` for
jax-nufft), `n_pix`, `n_rows`, `n_w`, `w_kernel_width`, and the four timing
columns `dirty2vis_median_ms`, `vis2dirty_median_ms`, `dirty2vis_min_ms`,
`vis2dirty_min_ms`.

**How to compare.** A jax-nufft row is one measurement; the ducc0 rows for
the same fixture are a thread-count sweep. "jax-nufft is N times faster"
means against `min()` over that sweep — ducc0 at its own best measured
setting for that cell and operator. Comparing against a fixed thread count
instead flatters jax-nufft by up to 6x, since ducc0's best is never all 288
hardware threads.

`sizing` is not decoration: the `realistic` suite's `n_pix` and `n_rows` are
derived from the instrument parameters recorded there, and
`test_the_realistic_problem_sizes_reproduce_from_the_stated_sizing_rule`
recomputes all eight from the rule. The `ci_sized` suite is the unmodified
`tests/conftest.py` fixtures, kept so the effect of the sizing can be
inspected rather than taken on trust.

## v0.2.0 memory JSON: `v0.2.0-memory-gh200.json`

Top-level `{schema, provenance, rows, w_chunk_sweep}`.

`rows` holds one entry per (fixture, operator, implementation), each with
`n_pix`, `n_rows`, `n_w`, `w_strategy`, `w_chunk`, `nthreads` and:

| key | impl | meaning |
|---|---|---|
| `temp_mb` | jax-nufft | `memory_analysis().temp_size_in_bytes` for the compiled executable — a compiler-reported figure, independent of run order |
| `peak_mb` | both | jax-nufft: peak device HBM. ducc0: peak process RSS |
| `peak_pre_mb` | jax-nufft | device peak before the measured call |
| `rss_import_mb` | ducc0 | RSS after imports, before any problem data exists |
| `working_set_mb` | ducc0 | `peak_mb - rss_import_mb`: input + scratch + output without the interpreter, the quantity comparable with jax-nufft's `peak_mb` |

**One operator per process.** Both `peak_bytes_in_use` and `ru_maxrss` are
monotonic high-water marks for the life of a process, so two operators
measured in one process cannot be attributed separately — the second one's
rise reads zero whenever its transient stayed below the first one's peak. An
earlier sweep did exactly that and reported a zero adjoint delta for all
eight fixtures, plus an identical 72.0 MB forward figure for four fixtures
spanning 144² to 3600². Those numbers were discarded rather than published.
Any re-measurement must keep the one-operator-per-process discipline or it
will reproduce the same artefact.

**RSS is coarse.** Observed ducc0 values are multiples of about 36 MB, so
ratios against a ducc0 side below a few hundred MB carry a large relative
uncertainty. The `provenance.comparability` field says this too; read the
small ratios as approximate.

`w_chunk_sweep.rows` is a separate grid: `temp_mb` and `median_ms` for each
(fixture, operator) under every w-strategy, including `chunked` at several
chunk widths. This is the memory/compute dial the README documents.

Reading it needs one correction first: **when `w_chunk >= n_w` the chunked
strategy clamps to `dense_vmap`** and compiles to the same program, so several
rows share a `temp_mb` exactly. Ordering those tied rows by time and calling
the result an inversion measures run-to-run noise (about 1.4% here), not the
dial. Collapse equal `temp_mb` to one level first.

With that done, `tests/test_benchmark_claims.py` checks monotonicity across
all eight (fixture, operator) pairs, not only the one the README tabulates.
Seven are monotone. One is not, and is pinned rather than hidden:
`MWA_extended_zenith`'s adjoint at `n_w = 13`, where `chunked8` takes 4.1x the
scratch of `dense_scan` and is 2.9% slower. At `n_w = 13` a chunk of 8 is two
chunks of 7, so the dial barely engages; the trade is a usable one in the
regime where `n_w` substantially exceeds `w_chunk`, which is where it matters.
