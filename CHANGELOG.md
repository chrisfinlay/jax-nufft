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
  its code, so a call at either end is bit-identical to the old name for it. `w_chunk` is part of
  the JIT key and of the primitives' static configuration, so reverse mode chunks the way its
  forward did.

  Measured on MWA_extended off30 (256², 600 rows, `n_w = 134`, float64, eps 1e-6, `nthreads=1`,
  single channel, `memory_analysis().temp_size_in_bytes`), in units of one complex image:

  | | `dense_scan` | `chunked(8)` | `chunked(16)` | `chunked(32)` | `dense_vmap` |
  |---|---:|---:|---:|---:|---:|
  | forward | 2.01× | 9.08× | 16.15× | **28.26×** | 135.23× |
  | adjoint | 2.01× | 9.01× | 16.01× | **28.01×** | 268.00× |

  `chunked(32)` there runs 1.04–1.16× (forward) and 0.98–1.00× (adjoint) of `dense_vmap`'s time
  across two interleaved passes on a 10-core Apple M-series. Existing strategies are untouched: the
  optimised HLO for all four, plus `auto`, is byte-identical before and after over both operators
  and four fixtures.
  ([#25](https://github.com/chrisfinlay/jax-nufft/issues/25))

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
