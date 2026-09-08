# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Figures quoted below are recomputed from the committed benchmark JSON by
`tests/test_benchmark_claims.py` where a JSON exists; the rest cite the pull request that measured
them.

## [Unreleased]

## [0.2.0] — unreleased

This release covers everything since **v0.1.1**. A v0.1.2 series was developed and merged but never
tagged or released, so its changes appear here for the first time; they are marked *(v0.1.2 series)*.

### Removed

- **Python 3.10 is no longer supported.** The floor is 3.11. *(v0.1.2 series)*

### Changed — breaking

- **The default `w_strategy` is now `"auto"`, not `"dense_scan"`.**
  `"auto"` resolves per call from the plan and the device platform, so the strategy a given call
  runs may differ from v0.1.1. Pass `w_strategy="dense_scan"` explicitly to keep the old behaviour.
  On one GH200, against ducc0 on the 72 Grace cores of the same node, the old default ran 1.4–5.6×
  *slower* than ducc0 on five of six measured cells, where the heuristic's pick runs 1.4–6.3×
  faster.
  ([#46](https://github.com/chrisfinlay/jax-nufft/issues/46),
  [PR #48](https://github.com/chrisfinlay/jax-nufft/pull/48))

- **The minimum supported JAX is now 0.6.0**, raised from the declared 0.5.0. The old floor was
  never correct: `src/` calls `jax.typeof`, which was exported in 0.6.0, so on the declared minimum
  every operator invoked through `jax.disable_jit()` raised `AttributeError`.
  ([#21](https://github.com/chrisfinlay/jax-nufft/issues/21),
  [PR #52](https://github.com/chrisfinlay/jax-nufft/pull/52))

- **Running with `jax_enable_x64` disabled is now explicit rather than silent.** Previously a
  float64 plan built with x64 off silently produced float32 results with a ~3.4e-5 error floor and
  no warning. Plans now carry an explicit dtype, mixed dtypes are cast or rejected, and the
  x64-off path is guarded.
  ([#11](https://github.com/chrisfinlay/jax-nufft/issues/11),
  [PR #37](https://github.com/chrisfinlay/jax-nufft/pull/37))

- **The default `nthreads` is now strategy-aware.** `nthreads=0` previously made the default
  strategy several times slower than single-threaded.
  ([#24](https://github.com/chrisfinlay/jax-nufft/issues/24),
  [PR #40](https://github.com/chrisfinlay/jax-nufft/pull/40))

- **Plan internals changed shape.** `uvw` is stored once in metres with per-channel coordinates
  derived inside JIT, and leaves that were never read have been dropped. Code that reached into
  `WGridderPlan` fields rather than using the public operators may need updating; the plan is not
  a stable public interface.
  ([#23](https://github.com/chrisfinlay/jax-nufft/issues/23),
  [PR #42](https://github.com/chrisfinlay/jax-nufft/pull/42))

- **`window_padding_overhead` now measures the *bucketed* row-work**, `sum(slice_length ×
  n_planes) / live_row_count` instead of `n_chan × n_w × max_window_size / live_row_count`.
  Values on this scale are not comparable with those from before this change; the old figure is
  still recomputable from the plan, because `max_window_size` is unchanged. Two things follow for
  code that reads the plan: `WGridderPlan` gains the static `window_buckets` and
  `max_window_size_per_chan` and a tenth leaf `window_plane_order`, and the `auto` selector's
  padding branch (`_CPU_PADDING_CUTOFF` 6.0, `_GPU_PADDING_CUTOFF` 3.0) no longer fires on any
  repository fixture, so plans that used to fall back to a dense strategy on a high padding figure
  now stay windowed.
  ([#26](https://github.com/chrisfinlay/jax-nufft/issues/26))

### Added

- **`divide_by_n` on both operators.** `dirty2vis` and `vis2dirty` each take a keyword-only
  `divide_by_n` flag applying the measurement equation's image-side `1/n`. With **equal** flags the
  pair is an exact adjoint; with the previous mixed defaults the dot-product residual was 0.63,
  against 1.3e-15 when matched.
  ([#20](https://github.com/chrisfinlay/jax-nufft/issues/20),
  [PR #51](https://github.com/chrisfinlay/jax-nufft/pull/51))

- **`w_strategy="auto"`**, a platform-aware heuristic choosing among `dense_scan`, `dense_vmap`,
  `windowed_scan` and `windowed_vmap`. Opt-in when introduced, and the default since
  [#46](https://github.com/chrisfinlay/jax-nufft/issues/46). *(v0.1.2 series)*

- **`w_strategy="chunked"` / `"windowed_chunked"` with a static `w_chunk` (default 32)**, making the
  w-plane loop a memory/compute curve instead of a choice between two points. The loop scans over
  chunks of at most `w_chunk` planes with a `vmap` inside each, so transient memory follows
  `w_chunk` rather than `n_w`. The four older names *are* points on that curve — `dense_scan` and
  `windowed_scan` are `w_chunk=1`, `dense_vmap` and `windowed_vmap` are `w_chunk=n_w` — and share
  its code, so a call at either end is bit-identical to the old name for it **at equal
  `nthreads`**. `w_chunk` is part of the JIT key and of the primitives' static configuration, so
  reverse mode chunks the way its forward did.

  The default `w_chunk = 32` exceeds `n_w` on most, not all, of this repository's fixtures.
  Measured over every telescope in `tests/conftest.py` at both pointings (seed 0, eps 1e-6,
  float64, hermitian, one channel): EDA2 11 / **56**, GH200_large 9 / 26, MWA_compact 8 / 12,
  MWA_extended 11 / **134**, MeerKAT 8 / 13 (zenith / off30). Two of the ten run a real chunk
  loop at the default, with padding; the other eight clamp to `dense_vmap`.

  Measured on MWA_extended off30 (256², 600 rows, `n_w = 134`, float64, eps 1e-6, `nthreads=1`,
  single channel, `memory_analysis().temp_size_in_bytes`), in units of one complex image:

  | | `dense_scan` | `chunked(8)` | `chunked(16)` | `chunked(32)` | `dense_vmap` |
  |---|---:|---:|---:|---:|---:|
  | forward | 2.01× | 9.08× | 16.15× | **28.26×** | 135.23× |
  | adjoint | 2.01× | 9.01× | 16.01× | **28.01×** | 268.00× |

  `chunked(32)` there runs 1.01–1.12× (forward) and 0.94–1.26× (adjoint) of `dense_vmap`'s time
  over **seven** interleaved passes on a 10-core Apple M-series. Read those to one significant
  figure: the suite's own control — `chunked(1)`, the same compiled program as `dense_scan` —
  spans 0.92–1.00× against it over the same passes, which is the instrument's resolution at this
  problem size. The memory rows are exact and reproduce to the byte.

  **On a GH200, where the issue was opened.** MWA_extended off30 at 3600² / 1 219 200 rows
  (`n_w = 140`, complex image 197.8 MB), eps 1e-6, float64, single channel, defaults for
  `hermitian`/`nthreads`; median of 5 with warm-up outside the timer:

  | | temp | vs `dense_vmap` | forward | adjoint |
  |---|---:|---:|---:|---:|
  | `dense_scan`  |    414 MB | **73.1× less** | 2.35× | 1.94× |
  | `chunked(8)`  |  1 929 MB | 15.7× less | 1.73× | 1.53× |
  | `chunked(16)` |  3 659 MB |  8.3× less | 1.35× | 1.24× |
  | `chunked(32)` |  6 256 MB |  4.8× less | **1.27×** | 1.17× |
  | `chunked(64)` | 10 367 MB |  2.9× less | 1.10× | 1.06× |
  | `dense_vmap`  | 30 290 MB |  1.0×      | 1.00× | 1.00× |

  The 30 GB that made this cell need a 96 GB device becomes **6.3 GB** at the default `w_chunk`.
  Note that the definition of done's GPU gate — "`chunked(32)` within 1.2× of `dense_vmap` on
  every cell" — is **breached on that forward cell at 1.27×**; its adjoint and every other cell
  measured pass. Two of the four GPU fixtures cannot inform the gate at all (MWA_extended zenith
  `n_w = 13`, MeerKAT off30 `n_w = 14`: `w_chunk = 32` clamps and reads 1.00× by construction);
  EDA2 off30 (150² / 4 896 000 rows, `n_w = 60`) is the other real cell and passes at 1.07× /
  1.01×. Full tables in `README.md` §`w_chunk`.

  Existing strategies are untouched **on every plan with `n_w > 1`**: the optimised HLO for all
  four older names, plus `auto`, is byte-identical before and after over both operators and four
  fixtures. The exception is the constant-w fast path (`n_w == 1`), where the plane loop tests
  `w_chunk >= n_w` before `w_chunk == 1` and so sends the *scan* names down the single-plane vmap
  branch instead of `lax.scan` — 24 of 82 optimised-HLO keys differ there, with bit-identical
  results on every cell and a strictly smaller program.

  **Not done:** implementation-plan item 4 — `chunked` cells at `w_chunk ∈ {8, 32, 128}` in the
  CPU and GPU benchmark suites. `tests/test_benchmark_claims.py` recovers `w_strategy` from the
  benchmark test *name* and classifies it with `rsplit("_", 1)[1]`, which raises on `"chunked"`
  and yields a third family on `"windowed_chunked"`, against pair counts and spreads pinned to
  three decimals from committed v0.1.2 JSONs. Teaching that layer about a family that is neither
  scan nor vmap is its own change; until then the curve is measured out-of-band and published
  above. Issue #25 is therefore **not fully closed** by this entry.
  ([#25](https://github.com/chrisfinlay/jax-nufft/issues/25))

- **The windowed strategies bucket their w-planes by window size.** Each channel's planes are
  sorted into at most four size classes, placed by an exact dynamic program over that channel's
  own padded window lengths, and each class is a sub-loop with its own static slice length — so a
  plane whose window holds 8 rows no longer reads 155. The class table is per channel, so a
  high-frequency channel (narrower windows) is no longer charged the plan-wide maximum.

  Padded row-work over irreducible row-work on the ten review cells (eps 1e-6, float64, seed 0,
  `hermitian=True`, `n_chan = 1`, CI fixture sizes), before → after:

  | fixture | before | after | classes |
  |---|---:|---:|---:|
  | EDA2 zenith | 1.5709 | 1.0368 | 4 |
  | EDA2 off30 | 2.5191 | 1.2424 | 4 |
  | MWA_compact zenith | 1.1431 | 1.0007 | 2 |
  | MWA_compact off30 | 1.7139 | 1.0467 | 4 |
  | MWA_extended zenith | 1.5714 | 1.0371 | 4 |
  | MWA_extended off30 | 4.9441 | **1.3768** | 4 |
  | MeerKAT zenith | 1.1429 | 1.0005 | 2 |
  | MeerKAT off30 | 1.8571 | 1.0690 | 4 |
  | GH200_large zenith | 1.2857 | 1.0000 | 4 |
  | GH200_large off30 | 2.7389 | 1.2861 | 4 |

  Runtime, CPU `nthreads=1`, macOS arm64 10-core, median of 9 interleaved calls with the plan and
  one warm-up outside the timer (AGENTS.md §6). MWA_extended off30 (`n_w = 134`): forward
  `windowed_vmap` 107.2 → 96.8 ms and `windowed_chunked(32)` 111.5 → 107.3 ms; adjoint
  `windowed_scan` 170.6 → 153.8 ms, `windowed_vmap` 145.0 → 128.4 ms, `windowed_chunked(32)`
  137.3 → 125.4 ms. MeerKAT off30 (`n_w = 13`): forward `windowed_vmap` 11.60 → 8.96 ms and
  `windowed_chunked(32)` 11.66 → 8.98 ms. Against the best dense strategy on the same cell,
  `windowed_vmap` and `windowed_chunked` land at 0.85–1.07× on all four (fixture, operator) cells;
  `windowed_scan` is 1.16–1.44×, which is the scan family's own standing gap on these two
  fixtures — `dense_scan` is 1.28–1.47× of the best dense strategy on the same runs — and not
  something bucketing moves.

  `windowed_vmap`'s forward also stopped materialising one `(n_rows,)` vector per plane, which
  cost `n_w × n_rows` whatever the windows held; it scatter-adds each class's
  `(n_planes, slice_length)` block into a sorted-row carry instead. Forward
  `temp_size_in_bytes` on a 4000-row, 16², 138-plane fixture (float64, eps 1e-6): 13,565,952 B →
  5,389,696 B from the scatter-add alone → 1,097,136 B with bucketing. A scan holds one class's
  slice at a time, so `windowed_scan`'s peak is set by the widest class and does not fall
  (128,032 B → 128,224 B on the same fixture).

  **The definition of done's two CPU timing gates, stated as measured rather than as met.**

  *Gate 1 — "windowed strategies within 1.2× of the best dense strategy on CPU single-thread for
  MWA_extended off30 and MeerKAT off30".* Re-measured on this branch under the same protocol (a
  fresh interleaved run on the same machine; it agrees with the runtime paragraph above to within
  2.6% on every ratio):
  `windowed_vmap` and `windowed_chunked` clear it on all four (fixture, operator) cells at
  0.85–1.06×, but `windowed_scan` reads 1.42× / 1.13× (MWA_extended off30 forward / adjoint) and
  1.36× / 1.24× (MeerKAT off30), so it is **above 1.2× on three of the four**. That is the scan
  family's own gap and not windowing: on the same runs `windowed_scan` is 0.87–0.98× of
  `dense_scan` — never slower than its own family's dense member — while `dense_scan` alone is
  1.30–1.46× of the best dense strategy. The gate as written is therefore measuring a difference
  between the scan and vmap families rather than the change under review. Read on the pick a caller
  actually gets, it is met: CPU `auto` resolves to a windowed strategy on exactly one of the four
  cells (MWA_extended off30 *adjoint*; the two forwards go to `dense_scan`, and MeerKAT off30's
  adjoint fails the `n_w / W > 2` gate at 1.86), and that cell reads **1.13×**.

  *Gate 2 — "faster than dense on the adjoint for a 20k-row problem" (`synthetic_uvw` with
  `GH200_LARGE`'s baseline parameters, `n_rows = 20_000`, `n_pix = 512`, off30).* Measured, same
  protocol, eps 1e-6, float64, seed 0, `hermitian=True`: `n_w = 25`, `max_window_size` 15,067 of
  20,000, buckets `((2568, 11), (5567, 4), (10266, 5), (15067, 5))`, overhead 1.2656 against an
  un-bucketed 2.6905. Adjoint, best dense/chunked 172.7 ms: `windowed_vmap` **141.7 ms (0.82×)**,
  `windowed_chunked(32)` 142.6 ms (0.83×), `windowed_scan` 179.5 ms (1.04×). Forward, best dense
  138.8 ms: `windowed_chunked(32)` 128.5 ms (0.93×), `windowed_vmap` 129.0 ms (0.93×),
  `windowed_scan` 154.1 ms (1.11×). **Gate met on the adjoint**, by 1.22×.

  **Not done:** the definition of done's GPU gate, which needs a CUDA jax-finufft build.

  **Not measured by any gate:** the multi-channel path. Every figure above is `n_chan = 1`, which
  is one bucket-table group and emits the pre-#26 program. A plan whose channels bucket differently
  compiles one body per group, and at a realistic ±5% frequency spread the group count *equals*
  `n_chan`. Measured on EDA2 off30 at `n_chan` 1 / 3 / 8 / 16, the compile time of a jitted
  `windowed_scan` `vis2dirty` is 0.13 / 0.29 / 0.63 / 1.10 s against a flat 0.07–0.09 s for
  `dense_scan` and 0.08–0.11 s for `windowed_scan` at `ab7fbbd`; run time is unchanged. The
  windowed adjoint's transient is below both baselines to eight channels and above them past that
  — 2,035,712 B at `n_chan = 16` against `ab7fbbd`'s 1,237,384 B (+64.5%) and `dense_scan`'s
  1,202,376 B (+69.3%) — because it concatenates one image cube per group. `dirty2vis` is
  unaffected (279,904 B against 1,243,904 B). Retuning `auto` for channel count is issue
  [#34](https://github.com/chrisfinlay/jax-nufft/issues/34).
  ([#26](https://github.com/chrisfinlay/jax-nufft/issues/26))

- **A constant-w fast path.** When every row shares one `w` in wavelengths, the plan collapses to a
  single plane (`plan.n_w == 1`, `plan.is_constant_w`). *(v0.1.2 series)*

- **A GPU benchmark suite** (`--runbench-gpu`) with HBM capture, plus committed GH200 baselines
  under `docs/benchmarks/`. *(v0.1.2 series)*

- **CI covers Python 3.13 and 3.14.** *(v0.1.2 series)*

### Fixed

- **`jax.grad` through either operator no longer costs `O(n_w · image)` memory.** Both operators are
  bound as linear primitives with `ad.primitive_transposes`, so gradients cost
  `O(image + n_rows)`. Measured peak XLA temp for `grad`:

  | fixture | `n_w` | forward | before | after |
  |---|---:|---:|---:|---:|
  | MWA_extended off30, 256² | 134 | 2.11 MB | 285.47 MB | **3.16 MB** |
  | 1024² / 20k rows | 25 | 33.87 MB | 897.83 MB | **50.65 MB** |

  Gradients also got **1.15× faster**, and the forward is bit-identical.
  ([#21](https://github.com/chrisfinlay/jax-nufft/issues/21),
  [PR #52](https://github.com/chrisfinlay/jax-nufft/pull/52))

- **The accuracy contract now holds.** The achieved error was 3–4.5× the requested epsilon at
  1e-6…1e-8 and 26–550× at 1e-10…1e-12. Adopting FINUFFT's w-kernel width rule brought this within
  the documented multiple, and the test tolerances that had hidden it were tightened to measured
  bounds.
  ([#9](https://github.com/chrisfinlay/jax-nufft/issues/9),
  [#10](https://github.com/chrisfinlay/jax-nufft/issues/10),
  [PR #38](https://github.com/chrisfinlay/jax-nufft/pull/38),
  [PR #39](https://github.com/chrisfinlay/jax-nufft/pull/39))

- **Benchmark figures quoted in prose are now recomputed from the committed JSON by a test**, after
  five hand-written citations were found to disagree with the data they cited.
  ([#49](https://github.com/chrisfinlay/jax-nufft/issues/49))

### Performance

- **The w-plane count is roughly halved, twice.** `nshift` centres the `n−1` range
  ([#16](https://github.com/chrisfinlay/jax-nufft/issues/16)), and negative-w rows are folded onto
  their Hermitian image for real skies
  ([#17](https://github.com/chrisfinlay/jax-nufft/issues/17)) — on MWA_extended off30, `n_w` 251 →
  134. Every measured GPU cell got faster.
  ([PR #41](https://github.com/chrisfinlay/jax-nufft/pull/41),
  [PR #50](https://github.com/chrisfinlay/jax-nufft/pull/50))

- **`window_padding_overhead` measures the right denominator**, so the reported figure is no longer
  inflated by counting zero-weight padding rows as irreducible work. Values on this scale are not
  comparable with those from v0.1.2 or earlier.
  ([#43](https://github.com/chrisfinlay/jax-nufft/issues/43),
  [PR #47](https://github.com/chrisfinlay/jax-nufft/pull/47))

### Internal

- Test coverage for odd, non-square and anisotropic images and per-channel geometry
  ([#14](https://github.com/chrisfinlay/jax-nufft/issues/14)); clumped and constant-w w-distributions
  against an external oracle, and the x64-off leg's first off-zenith numerical oracle
  ([#15](https://github.com/chrisfinlay/jax-nufft/issues/15)); exact-identity gradient tests plus an
  independent derivative reference built from `jax.vjp` of an exact DFT
  ([#22](https://github.com/chrisfinlay/jax-nufft/issues/22)).
- Native aarch64 CPU environments, split `gpu`/`gpu-dev` features, and pixi-based CI.
  *(v0.1.2 series)*

## [0.1.1]

See the git history; this file starts at 0.2.0.

[Unreleased]: https://github.com/chrisfinlay/jax-nufft/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/chrisfinlay/jax-nufft/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/chrisfinlay/jax-nufft/releases/tag/v0.1.1
