# Running the benchmarks

The opt-in benchmark suite, how to invoke it, what its numbers include, and
the historical CPU comparisons. For the headline GH200-versus-ducc0 result
see [README.md](../README.md#performance).

<!--
Moved out of README.md by the 0.2.0 documentation pass. The text is unchanged:
every figure in it was verified against the code or the committed benchmark
JSON during issue #33, and several are recomputed by
tests/test_benchmark_claims.py, so it is moved rather than rewritten.
-->

## CPU benchmarks vs ducc0

The repository ships an opt-in benchmark suite that times *and* measures
peak memory of `dirty2vis` / `vis2dirty` against `ducc0.wgridder`, across
the four built-in telescope configs and both `w_strategy` choices.

The benchmarks are gated behind two flags so they don't run by default:

| Flag                          | What it does                                      |
|-------------------------------|---------------------------------------------------|
| `--runbench`                  | Enables the bench suite (otherwise all skipped).  |
| `--bench-pointing={zenith,off30,both}` | Default `zenith`. Picks which pointings run. |

The bench file contains four kinds of test, all parametrised over the
four telescopes and the chosen pointings:

| Test                             | Strategies                                              | What it measures   |
|----------------------------------|---------------------------------------------------------|--------------------|
| `test_bench_jax_dirty2vis`       | dense_scan, dense_vmap, windowed_scan, windowed_vmap    | wall-clock time    |
| `test_bench_ducc_dirty2vis`      | n/a                                                     | wall-clock time    |
| `test_bench_jax_vis2dirty`       | dense_scan, dense_vmap, windowed_scan, windowed_vmap    | wall-clock time    |
| `test_bench_ducc_vis2dirty`      | n/a                                                     | wall-clock time    |
| `test_memory_jax_dirty2vis`      | dense_scan, dense_vmap, windowed_scan, windowed_vmap    | peak RSS delta     |
| `test_memory_ducc_dirty2vis`     | n/a                                                     | peak RSS delta     |
| `test_memory_jax_vis2dirty`      | dense_scan, dense_vmap, windowed_scan, windowed_vmap    | peak RSS delta     |
| `test_memory_ducc_vis2dirty`     | n/a                                                     | peak RSS delta     |

The standard pytest `-k` filter is the usual way to narrow a run.
Pytest-benchmark's `--benchmark-group-by=param:bench_telescope_pointing`
groups results so each comparison table contains all the implementations
for one telescope-pointing. The `extra_info` row reports
`max_window_size` and `padding_overhead` alongside each timing.

### Common invocations

Time benchmarks, zenith only (fastest):

```sh
pixi run -e test pytest tests/test_benchmark_against_ducc.py \
    --runbench -k "not memory" \
    --benchmark-group-by=param:bench_telescope_pointing -q
```

Time benchmarks, off-zenith only:

```sh
pixi run -e test pytest tests/test_benchmark_against_ducc.py \
    --runbench --bench-pointing=off30 -k "not memory" \
    --benchmark-group-by=param:bench_telescope_pointing -q
```

Time benchmarks, full matrix (both pointings):

```sh
pixi run -e test pytest tests/test_benchmark_against_ducc.py \
    --runbench --bench-pointing=both -k "not memory" \
    --benchmark-group-by=param:bench_telescope_pointing -q
```

Memory only, all telescopes, both pointings (use `-s` so the summary
table printed by the autouse fixture isn't captured):

```sh
pixi run -e test pytest tests/test_benchmark_against_ducc.py \
    --runbench --bench-pointing=both --benchmark-disable -k memory -s
```

Single telescope (e.g. just MeerKAT), forward only, both strategies:

```sh
pixi run -e test pytest tests/test_benchmark_against_ducc.py \
    --runbench --bench-pointing=both \
    -k "MeerKAT and dirty2vis and not memory" \
    --benchmark-group-by=param:bench_telescope_pointing -q
```

vmap variants only, all telescopes, both pointings:

```sh
pixi run -e test pytest tests/test_benchmark_against_ducc.py \
    --runbench --bench-pointing=both \
    -k "(vmap or ducc) and not memory" \
    --benchmark-group-by=param:bench_telescope_pointing -q
```

(The `or ducc` clause keeps the ducc rows visible alongside the jax/vmap
rows for direct comparison.)

Windowed variants only, all telescopes, both pointings:

```sh
pixi run -e test pytest tests/test_benchmark_against_ducc.py \
    --runbench --bench-pointing=both \
    -k "(windowed or ducc) and not memory" \
    --benchmark-group-by=param:bench_telescope_pointing -q
```

Save and reload runs (handy on quiet machines):

```sh
pixi run -e test pytest tests/test_benchmark_against_ducc.py \
    --runbench --benchmark-save=baseline -k "not memory"

pixi run -e test pytest tests/test_benchmark_against_ducc.py \
    --runbench --benchmark-compare=0001_baseline -k "not memory"
```

### What the benchmark numbers include

The jax-nufft timings below are **steady-state per-call cost** &mdash;
the plan and the JIT compile are excluded from the timed window:

* `make_plan(...)` runs once in setup, then `plan` is reused across
  every benchmark iteration. This matches the usage pattern in an
  optimisation loop, where the plan is built once on init and the
  forward / adjoint are called every step.
* A warmup `dirty2vis(plan, image).block_until_ready()` runs once
  before `pytest-benchmark`'s timed loop, so the JIT compile is also
  excluded.

The one-time costs for MWA-extended off-zenith are:

| One-time cost                                  | Time     |
|-----------------------------------------------|----------|
| `make_plan` (host-side numpy, plus device-side `jnp.asarray` copies) | ~5 ms |
| First `dirty2vis(plan, ..., w_strategy=...)` (JIT compile + execute) | ~700 ms |
| First `vis2dirty(plan, ..., w_strategy=...)`  | ~700 ms  |

Each JIT cache entry is keyed on the static plan metadata
(`n_w`, `n_chan`, `n_rows`, `w_strategy`, etc.), so a fresh plan with
the same shape reuses the same compiled binary.

**ducc asymmetry:** ducc's public Python API (`ducc0.wgridder.dirty2vis`)
does not separate plan from execute &mdash; each call internally rebuilds
its bin sort, kernel selection, and other per-call state. The ducc
numbers below therefore include that per-call planning. In an
optimisation loop with fixed `(uvw, freq)`, jax-nufft amortises its
plan cost to nearly zero per step, while ducc pays its full
per-call cost on every step. Treat the ducc column as a fair
comparison against current ducc *usage* rather than against a
hypothetical "ducc with reused plan".

### Indicative numbers (Mac M-series CPU, eps=1e-6)

Median wall-clock time for `dirty2vis` / `vis2dirty`, from a full sweep of
`tests/test_benchmark_against_ducc.py --runbench --bench-pointing=both`
(160 benchmark cases, Apple M4, 10 cores, macOS arm64, 2026-09-04). Rerun
on your hardware before making strategy decisions &mdash; these are
CI-runner sized problems and absolute timings vary several-fold across
machines.

Regenerated for issue #24 (R11/D4), which found the previous version of
this table silently comparing jax at its old implicit default
(`nthreads=0`, every OpenMP core re-spun per w-plane call) against ducc's
explicit `nthreads=1` &mdash; not the "single-threaded" comparison the old
caption claimed. The two tables below give the honest comparison instead:
one with both sides pinned to the *same* explicit thread count, one with
both sides left at "let the library decide" (`nthreads=0` passed
explicitly to both jax and ducc).

Every fixture below has 400-600 rows, far under the 100k-row
`_NTHREADS_SMALL_N_ROWS` cutoff (see [`nthreads`](strategies.md#nthreads-issue-24-r11d4)
above), so jax's new strategy-aware default (`nthreads=None`) resolves to
`nthreads=1` for *all four* strategies on every problem in these tables,
vmap included &mdash; the strategy split (`0` for `dense_vmap` /
`windowed_vmap`) only takes effect above that cutoff, which nothing here
reaches. **The "matched `nthreads=1`" table below is therefore what
`dirty2vis` / `vis2dirty` produce today with no `nthreads=` argument at
all**, across every strategy shown. The "`nthreads=0`" table is an
explicit opt-out (`nthreads=0` passed by hand) or a preview of the
vmap-family steady state at larger row counts, not a second flavour of the
current default.

**Matched threads, jax and ducc both explicit `nthreads=1`**

Zenith pointing:

| Telescope     | dense_scan       | dense_vmap       | windowed_scan    | windowed_vmap    | ducc           |
|---------------|------------------|------------------|------------------|------------------|----------------|
| EDA2          | 4.2 / 4.6 ms     | 1.3 / 1.4 ms     | 4.5 / 4.2 ms     | 1.6 / 1.4 ms     | 0.9 / 1.3 ms   |
| MWA_compact   | 4.9 / 5.8 ms     | 2.5 / 2.8 ms     | 5.1 / 6.2 ms     | 2.8 / 3.6 ms     | 2.1 / 2.8 ms   |
| MWA_extended  | 37.1 / 40.0 ms   | 25.6 / 25.9 ms   | 42.2 / 38.3 ms   | 27.6 / 30.1 ms   | 13.6 / 15.5 ms |
| MeerKAT       | 14.2 / 22.5 ms   | 9.7 / 16.7 ms    | 16.0 / 23.4 ms   | 10.0 / 19.2 ms   | 9.1 / 11.8 ms  |

30-deg off-zenith pointing (`n_w` is much larger; this is where both v0.1.1
improvements have the most to bite into):

| Telescope     | dense_scan        | dense_vmap        | windowed_scan       | windowed_vmap       | ducc            |
|---------------|-------------------|-------------------|---------------------|---------------------|-----------------|
| EDA2          | 52.7 / 50.9 ms    | 14.0 / 13.7 ms    | 49.8 / 50.4 ms      | 11.9 / 13.6 ms      | 2.0 / 2.7 ms    |
| MWA_compact   | 16.2 / 22.9 ms    | 7.8 / 11.0 ms     | 18.8 / 20.4 ms      | 8.9 / 10.8 ms       | 2.7 / 3.6 ms    |
| MWA_extended  | 969.7 / 951.9 ms  | 594.2 / 664.5 ms  | 869.6 / 631.9 ms    | 562.4 / 504.0 ms    | 41.0 / 55.1 ms  |
| MeerKAT       | 49.3 / 81.7 ms    | 32.0 / 56.6 ms    | 53.3 / 83.2 ms      | 35.2 / 64.9 ms      | 12.2 / 15.6 ms  |

**Both sides `nthreads=0`** ("let the library decide", passed explicitly on
both sides &mdash; this is what *every* column looked like pre-#24, since
jax's old flat default *was* `0`; at the row counts in this table, none of
it is what jax's new default produces post-#24, including the vmap
columns, per the small-`n_rows` override explained above):

Zenith pointing:

| Telescope     | dense_scan       | dense_vmap       | windowed_scan    | windowed_vmap    | ducc           |
|---------------|------------------|------------------|------------------|------------------|----------------|
| EDA2          | 38.0 / 58.8 ms   | 2.1 / 2.7 ms     | 42.4 / 57.4 ms   | 29.6 / 45.2 ms   | 2.0 / 2.5 ms   |
| MWA_compact   | 36.1 / 29.5 ms   | 3.2 / 3.1 ms     | 37.3 / 29.5 ms   | 26.3 / 33.0 ms   | 2.7 / 3.7 ms   |
| MWA_extended  | 154.0 / 220.6 ms | 13.4 / 19.5 ms   | 155.9 / 237.0 ms | 133.4 / 210.3 ms | 7.1 / 10.3 ms  |
| MeerKAT       | 80.8 / 67.0 ms   | 8.0 / 8.1 ms     | 77.5 / 76.9 ms   | 61.7 / 58.7 ms   | 7.6 / 9.4 ms   |

30-deg off-zenith pointing:

| Telescope     | dense_scan          | dense_vmap        | windowed_scan       | windowed_vmap        | ducc            |
|---------------|----------------------|-------------------|---------------------|-----------------------|-----------------|
| EDA2          | 469.7 / 656.0 ms     | 9.6 / 20.0 ms     | 545.9 / 411.3 ms    | 384.0 / 299.7 ms      | 6.3 / 7.1 ms    |
| MWA_compact   | 100.0 / 145.8 ms     | 6.1 / 10.0 ms     | 99.3 / 133.4 ms     | 79.5 / 114.1 ms       | 3.4 / 4.3 ms    |
| MWA_extended  | 3377.8 / 3816.1 ms   | 238.2 / 476.5 ms  | 4300.6 / 4536.4 ms  | 3881.3 / 3270.1 ms    | 36.8 / 40.4 ms  |
| MeerKAT       | 206.2 / 217.0 ms     | 15.7 / 23.1 ms    | 210.4 / 245.7 ms    | 180.3 / 201.6 ms      | 8.0 / 11.8 ms   |

The gap between the two tables' scan columns is the effect issue #24
fixes: `dense_scan` / `windowed_scan` at `nthreads=0` re-spin the whole
OpenMP pool on every w-plane's FINUFFT call, which is why they are
2.7-13.7x slower than the matched-`nthreads=1` table above on the same
problem (e.g. MWA_extended off30 dense_scan: 3377.8ms at `nthreads=0` vs
969.7ms at `nthreads=1`, ~3.5x; the largest gaps are on EDA2, up to 13.7x
on zenith windowed_scan adjoint). This is exactly why the strategy-aware
default resolves `dense_scan` / `windowed_scan` to `1`, not `0`.

### Isolating Part 1 (standard n_w) vs Part 2 (windowed)

We checked out the v0.1.0 tag and ran the same benchmarks with the v0.1
`scan` / `vmap` strategies, then compared. With the v0.1 `scan` row as
the baseline:

  * **Part 1 only** = `v0.1 scan / v0.1.1 dense_scan` &mdash; fewer
    w-planes, same dense algorithm.
  * **Part 2 only** = `v0.1.1 dense_scan / v0.1.1 windowed_scan` &mdash;
    same `n_w`, switch to windowed.
  * **Combined** = `v0.1 scan / v0.1.1 windowed_scan`.

Scan-variant speedups (median-time ratios, higher is better):

| op        | telescope     | pointing | Part 1 only | Part 2 only | Combined |
|-----------|---------------|----------|-------------|-------------|----------|
| dirty2vis | EDA2          | zenith   | 1.35x       | 0.98x       | 1.32x    |
| dirty2vis | EDA2          | off30    | 1.52x       | 1.06x       | 1.61x    |
| dirty2vis | MWA_compact   | zenith   | 1.05x       | 0.99x       | 1.04x    |
| dirty2vis | MWA_compact   | off30    | 1.49x       | 0.95x       | 1.41x    |
| dirty2vis | MWA_extended  | zenith   | 1.40x       | 1.01x       | 1.42x    |
| dirty2vis | MWA_extended  | off30    | 1.62x       | 0.99x       | 1.60x    |
| dirty2vis | MeerKAT       | zenith   | 1.06x       | 1.00x       | 1.06x    |
| dirty2vis | MeerKAT       | off30    | 1.43x       | 0.96x       | 1.37x    |
| vis2dirty | EDA2          | zenith   | 1.88x       | 0.71x       | 1.33x    |
| vis2dirty | EDA2          | off30    | 1.50x       | 1.17x       | 1.75x    |
| vis2dirty | MWA_compact   | zenith   | 0.91x       | 1.17x       | 1.06x    |
| vis2dirty | MWA_compact   | off30    | 1.22x       | 1.53x       | 1.86x    |
| vis2dirty | MWA_extended  | zenith   | 1.12x       | 1.06x       | 1.19x    |
| vis2dirty | MWA_extended  | off30    | 1.22x       | 1.19x       | 1.45x    |
| vis2dirty | MeerKAT       | zenith   | 0.78x       | 1.15x       | 0.90x    |
| vis2dirty | MeerKAT       | off30    | 1.25x       | 1.05x       | 1.32x    |

vmap-variant speedups (`v0.1 vmap` &rarr; `v0.1.1 dense_vmap` &rarr;
`v0.1.1 windowed_vmap`):

| op        | telescope     | pointing | Part 1 only | Part 2 only | Combined |
|-----------|---------------|----------|-------------|-------------|----------|
| dirty2vis | EDA2          | zenith   | 1.29x       | 0.99x       | 1.28x    |
| dirty2vis | EDA2          | off30    | 1.49x       | 1.22x       | 1.81x    |
| dirty2vis | MWA_compact   | off30    | 1.34x       | 0.89x       | 1.19x    |
| dirty2vis | MWA_extended  | off30    | 1.48x       | 0.95x       | 1.40x    |
| dirty2vis | MeerKAT       | off30    | 1.36x       | 0.93x       | 1.26x    |
| vis2dirty | EDA2          | off30    | 1.42x       | 1.23x       | 1.74x    |
| vis2dirty | MWA_compact   | off30    | 1.28x       | 1.03x       | 1.32x    |
| vis2dirty | MWA_extended  | off30    | 1.45x       | 1.13x       | 1.64x    |
| vis2dirty | MeerKAT       | off30    | 1.45x       | 0.95x       | 1.38x    |

Reading the table:

* **Part 1 wins broadly** &mdash; 1.05x to 1.6x across most cases, with
  the largest gains at off-zenith where `n_w` was previously inflated
  most by the v0.1 `x0 = 1/W` choice. This matches the `W/4` theoretical
  FFT-count reduction: at eps=1e-6 the kernel width was `W = 6` under the
  pre-#9 rule in force when this comparison was run, giving an expected
  speedup of 1.5x. (The current rule, `W = ceil(-log10(epsilon / 10))`,
  gives `W = 7` at that epsilon; this whole section is a v0.1.0 to v0.1.1
  comparison and is kept at the widths of the time.)
* **Part 2 helps the adjoint** at off-zenith on most telescopes
  (1.05–1.53x). Forward is essentially flat: NUFFT type-2 per-point
  cost doesn't fall with slice size.
* **At zenith Part 2 is a wash** because `n_w` is already close to `W`,
  so `max_window_size ~ n_rows` and there's nothing to slice off.
* **Combined wins** reach 1.6x–1.86x on the off-zenith adjoint cases
  most users care about.

The MWA-extended off-zenith adjoint (the configuration the v0.1.1 plan
targeted) drops from **1056 ms** (v0.1 scan) to **872 ms** (Part 1
only) to **731 ms** (Part 1 + Part 2 windowed_scan) &mdash; a 1.45x
total speedup, with both parts each contributing ~1.2x.

