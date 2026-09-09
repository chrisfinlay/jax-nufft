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

- **The `auto` selector's padding branch reads a different field on the adjoint.**
  `window_padding_overhead` is unchanged — still `n_chan × n_w × max_window_size /
  live_row_count`, and now explicitly the *forward's* ratio — but the windowed adjoint buckets its
  plane slices (below), so its padded work is smaller and it is gated on a new
  `window_padding_overhead_adjoint`. Neither cutoff moved (`_CPU_PADDING_CUTOFF` 6.0,
  `_GPU_PADDING_CUTOFF` 3.0), but on the adjoint leg neither is reached by any repository fixture
  at any epsilon in either geometry (the adjoint maxima over the forty-cell calibration grid are
  1.6206 unfolded and 1.4133 folded), so an adjoint that used to fall back to a dense strategy on a
  high padding figure now stays windowed. The forward leg is untouched. `WGridderPlan` also gains
  the static `window_buckets` and `max_window_size_per_chan` and a tenth leaf `window_plane_order`,
  appended after `flip_sign` in the flatten order.
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

- **The windowed adjoint buckets its w-planes by window size.** Each channel's planes are
  sorted into at most four size classes, placed by an exact dynamic program over that channel's
  own padded window lengths, and each class is a sub-loop with its own static slice length — so a
  plane whose window holds 8 rows no longer reads 155. The class table is per channel, so a
  high-frequency channel (narrower windows) is no longer charged the plan-wide maximum.

  **The windowed forward is unchanged, deliberately.** Bucketing it as well was implemented,
  measured and reverted: on one GH200 (Daint, eps 1e-6, float64, `n_chan = 1`, realistic sizes,
  the `auto` path) it ran 20.3x slower on GH200_large zenith (2048²/50k, `n_w = 9`; 16.7 → 339.7
  ms), 58.4x slower on MeerKAT off30 (2700²/302400, `n_w = 14`; 32.9 → 1918.7 ms), and timed out
  at 420 s on MWA_extended zenith (3600²/1.22M) against 48.6 ms — while the adjoint over the same
  runs was 1.08-1.24x. `auto` resolves the forward to `windowed_vmap` on most realistic GPU plans,
  so this reached defaulting users. **The cause is not understood** and is left open as
  #65; two diagnosis-and-fix rounds did not move the GPU numbers. The forward therefore
  keeps the previous release's code, its `max_window_size` slice and its
  `window_padding_overhead` figure, and its optimised HLO is that code's: over 7 `w_strategy` x
  10 fixtures, all 70 forward modules differ from the previous revision in nothing but one
  appended unused parameter (the `window_plane_order` leaf) and the renumbering of the operand
  after it. No instruction differs. (The four GPU timings are the maintainer's, on hardware this
  repository's test suite does not have.)

  Padded row-work over irreducible row-work on the ten review cells (eps 1e-6, float64, seed 0,
  `hermitian=True`, `n_chan = 1`, CI fixture sizes) — the forward column is
  `window_padding_overhead`, unchanged, and the adjoint column is the new
  `window_padding_overhead_adjoint`:

  | fixture | forward | adjoint | classes |
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

  Adjoint runtime, CPU `nthreads=1`, macOS arm64 10-core, median of 11 calls with the strategies
  interleaved inside one process and one warm-up outside the timer (AGENTS.md §6), before →
  after. MWA_extended off30 (`n_w = 134`): `windowed_scan` 167.4 → 147.4 ms, `windowed_vmap`
  137.0 → 120.2 ms, `windowed_chunked(32)` 138.4 → 110.0 ms. MeerKAT off30 (`n_w = 13`):
  `windowed_scan` 16.71 → 16.36 ms, `windowed_vmap` 13.37 → 13.72 ms, `windowed_chunked(32)`
  13.15 → 14.14 ms — i.e. flat to 7.5% *slower* on that fixture, whose padding was only 1.86x to
  start with, so the sub-loops cost about what they save. Forward runtime is unchanged to within
  1% on every strategy of both fixtures, as it must be.

  Adjoint transient memory falls with the padding. Measured `vis2dirty` `temp_size_in_bytes` on a
  4000-row, 16², 138-plane fixture (float64, eps 1e-6, `max_window_size` 1072 of 4000 rows),
  before → after: `windowed_vmap` 2,932,224 → 1,038,464 B, `windowed_chunked(32)` 1,143,392 →
  787,616 B, `windowed_scan` 106,568 → 106,760 B (a scan holds one class's slice at a time, so
  its peak is the widest class and does not fall). Every `dirty2vis` transient on that fixture is
  byte-identical before and after, on all five strategies.

  **The definition of done's two CPU timing gates, stated as measured rather than as met.**

  *Gate 1 — "windowed strategies within 1.2x of the best dense strategy on CPU single-thread for
  MWA_extended off30 and MeerKAT off30".* Measured under the protocol above: `windowed_vmap` and
  `windowed_chunked` clear it on all four (fixture, operator) cells, at 0.85-1.16x, but
  `windowed_scan` reads 1.51x / 1.14x (MWA_extended off30 forward / adjoint) and 1.44x / 1.25x
  (MeerKAT off30), so it is **above 1.2x on three of the four**. That is the scan family's own
  gap and not windowing: `dense_scan` alone is 1.29-1.54x of the best dense strategy on the same
  runs, and `windowed_scan` is 0.88-0.98x of `dense_scan`. Three of those four readings are the
  *forward*, which this change does not touch, so on those the gate is measuring the previous
  release. Read on the pick a caller actually gets, it is met: CPU `auto` resolves to a windowed
  strategy on exactly one of the four cells (MWA_extended off30 **adjoint**; the two forwards go
  to `dense_scan`, and MeerKAT off30's adjoint fails the `n_w / W > 2` gate at 1.86), and that
  cell reads **1.14x**.

  *Gate 2 — "faster than dense on the adjoint for a 20k-row problem" (`synthetic_uvw` with
  `GH200_LARGE`'s baseline parameters, `n_rows = 20_000`, `n_pix = 512`, off30).* Measured, same
  protocol, two independent rounds per revision, eps 1e-6, float64, seed 0, `hermitian=True`:
  `n_w = 25`, `max_window_size` 15,067 of 20,000, buckets `((2568, 11), (5567, 4), (10266, 5),
  (15067, 5))`, adjoint overhead 1.2656 against a forward 2.6905. Adjoint, best dense 162.8 /
  164.5 ms: `windowed_vmap` **111.0 / 110.1 ms (0.68x)**, `windowed_chunked(32)` 110.5 / 111.5 ms
  (0.68x), `windowed_scan` 174.4 / 177.1 ms (1.07x). The same three strategies at the previous
  revision read 165.9 / 165.2, 166.8 / 166.3 and 180.8 / 180.1 ms against a best dense of 163.0 /
  162.5 — i.e. **bucketing is what takes this cell from 1.02x to 0.68x**. Gate met, by 1.47x.

  **Not done:** the definition of done's GPU gate, which needs a CUDA jax-finufft build; and the
  windowed forward, which is #65's.

  **Known defect, inherited rather than introduced.** The windowed forward's `windowed_chunked`
  branch accumulates a whole chunk into one shared `(n_rows,)` carry through a single scatter-add
  over its `(w_chunk, max_window_size)` index block. The windows in a chunk overlap physically —
  every visibility is inside the w-kernel's support on 7.00 planes on every fixture here — so
  those updates collide and XLA has to assume they can. Whether that costs anything on a GPU is
  the question the failed diagnosis above leaves open; the GH200 A/B does not answer it, because
  `w_chunk = 32` exceeds `n_w` on both of those fixtures and `windowed_chunked` degenerates to
  `windowed_vmap` there. It is declared by an `xfail(strict=True)` cell rather than fixed here;
  #65.

  **Not measured by any gate:** the multi-channel path. Every figure above is `n_chan = 1`, which
  is one bucket-table group and emits the previous release's program. A plan whose channels
  bucket differently compiles one adjoint body per group, and at a realistic ±5% frequency spread
  the group count *equals* `n_chan`. Measured on EDA2 off30 at `n_chan` 1 / 4 / 8 / 16 / 32, the
  compile time of a jitted `windowed_scan` `vis2dirty` is 0.083 / 0.224 / 0.397 / 0.656 / 1.231 s
  against a flat 0.047-0.073 s for `dense_scan` and 0.056-0.083 s at the previous revision. The
  windowed adjoint's transient is below both baselines to eight channels and above them past that
  — 2,035,712 B at `n_chan = 16` against 1,237,384 B before (+64.5%) and `dense_scan`'s
  1,202,376 B (+69.3%) — because it concatenates one image cube per group. `dirty2vis` does not
  group and is byte-identical at every channel count. Retuning `auto` for channel count is issue
  [#34](https://github.com/chrisfinlay/jax-nufft/issues/34).
  ([#26](https://github.com/chrisfinlay/jax-nufft/issues/26))

- **A constant-w fast path.** When every row shares one `w` in wavelengths, the plan collapses to a
  single plane (`plan.n_w == 1`, `plan.is_constant_w`). *(v0.1.2 series)*

- **A GPU benchmark suite** (`--runbench-gpu`) with HBM capture, plus committed GH200 baselines
  under `docs/benchmarks/`. *(v0.1.2 series)*

- **CI covers Python 3.13 and 3.14.** *(v0.1.2 series)*

- **A CI job that runs the declared JAX floor.** Every other job runs the single `jax` the pixi
  lockfile resolves, so the `jax>=` bound in `pyproject.toml` was exercised by nothing — which is
  how it came to say `0.5.0` while `src/` called `jax.typeof` (see the floor bump above). The
  `jax floor probe` job installs **jax alone** at the declared version and runs
  `tests/jax_floor_probe.py`, which reads the floor out of `pyproject.toml`, derives every
  module-level `jax.*` attribute chain `src/` and `tests/` touch by walking their ASTs (84 of
  them, measured on this branch; issue #53 records the earlier #21 review as having checked 18 by
  hand), imports and `getattr`s each, and then drives a 3×3-matmul miniature of the wgridder's
  primitive pattern through `jit`, `grad`, `jvp`, `linear_transpose`, `vmap` inside `grad`,
  `grad(grad(...))` and `disable_jit`. A `Call` terminates an attribute chain, so methods of a
  *returned* object — `jax.typeof(...).to_tangent_aval()`, `jax.jit(...).lower()`,
  `jnp.zeros(...).at[...]`, `.astype`, `.real`, `.reshape` — are outside the derived set and
  cannot be in it; the miniature calls all six for real instead, and which six is itself derived.
  Neither the version, the symbol list nor the method list is maintained by hand. With the floor
  reverted to `>=0.5.0` the job fails, naming `jax.typeof` as the one missing symbol of the 84.
  ([#53](https://github.com/chrisfinlay/jax-nufft/issues/53))

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

### Documentation

- **The GPU-versus-ducc0 and memory comparisons are now measured data in the tree**
  ([#33](https://github.com/chrisfinlay/jax-nufft/issues/33)). Both were previously hand-written
  markdown tables; `tests/test_benchmark_claims.py` recorded them as figures it could not
  recompute. `docs/benchmarks/v0.2.0-vs-ducc0-gh200.json` and `v0.2.0-memory-gh200.json` now hold
  the sweeps, and every figure the README prints from them is recomputed by a test.
  - Problem sizes are derived from instrument parameters (`pixsize = lambda / (3 B_max)`,
    `n_pix` the next even 5-smooth integer covering the field of view, `n_rows = 150 N_bl`)
    rather than taken from the CI fixtures, which are 400–600 rows at 64–256 pixels and measure
    overhead rather than the algorithm. The test recomputes all eight sizes from the rule.
  - ducc0 is compared at its own best measured thread count per cell. All 288 hardware threads
    is never its best setting, and costs it 1.9–6.0× against that best.
- **The accuracy sweep now runs in CI** ([#33](https://github.com/chrisfinlay/jax-nufft/issues/33)).
  Nothing in `.github/workflows` passed `--runsweep`, so the only evidence for the headline
  `2 * epsilon` contract ran solely when someone typed the flag locally. It costs 22 s.
- **The accuracy contract is scoped to the grid that measures it.** The sweep holds `w_strategy`
  at `dense_scan`, which is not the shipped default; cross-strategy agreement is pinned at
  `1e-11`, wider than the contract itself below `epsilon = 1e-10`. The `3 * epsilon` bound
  against ducc0 holds `epsilon` at {`1e-4`, `1e-6`}.
- **Corrected claims that the repository's own code or data falsified**
  ([#33](https://github.com/chrisfinlay/jax-nufft/issues/33)): the README reported the package
  version as v0.1.2 (it is `0.2.0.dev0`) and attributed shipped work to a "v0.1.3" that will
  never be tagged; the memory table reported 0 MB for three rows, an artefact of measuring two
  operators against one monotonic high-water mark; the padding-overhead ranges quoted an
  `epsilon = 1e-6` slice while naming a four-epsilon grid; "all four `w_strategy` choices"
  survived #25 adding two more; `_plane_chunk_grid(56, 32)` pads nothing where two docstrings
  said it pads a plane; and float32 was said to halve plan memory, which holds for
  image-dominated plans (0.501×) but not row-dominated ones (0.586×).
- `[tool.mypy]` and `[tool.ruff]` now target Python 3.11, matching `requires-python`.

## [0.1.1]

See the git history; this file starts at 0.2.0.

[Unreleased]: https://github.com/chrisfinlay/jax-nufft/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/chrisfinlay/jax-nufft/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/chrisfinlay/jax-nufft/releases/tag/v0.1.1
