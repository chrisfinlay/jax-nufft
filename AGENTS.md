# Repository guide for coding agents

This file is the orientation doc for Claude Code / other coding agents
picking up work on this repo. It covers what the project is, how the
code is laid out, the conventions and invariants you need to respect,
how to run tests / benchmarks, the history of design decisions taken
in `v0.1` and `v0.1.1`, and the performance plan for `v0.1.2`.

For end-user-facing documentation, see `README.md`. For the most recent
formal release plan, see `docs/v0.1.2-plan.md`.

---

## 1. What this project is

`jax-nufft` is a pure-JAX implementation of the wgridder algorithm for
radio interferometric imaging, built on `jax-finufft`. It exposes two
public operators:

* `dirty2vis(plan, image)` — forward (image &rarr; visibilities).
* `vis2dirty(plan, vis)` — adjoint (visibilities &rarr; image).

Both are fully traceable through `jax.jit`, `jax.vmap`, `jax.grad`. The
reference baseline for correctness is `ducc0.wgridder`. Headline
target is DFT parity within `10 * epsilon`; ducc parity, windowed
vs. dense, and adjoint reduction-order tolerances are looser — see
§6 for the full table.

The strategic value-add over ducc is **differentiability and the GPU
port via cuFINUFFT**, not raw CPU speed (ducc is consistently faster
on CPU). The intended user is a downstream JAX pipeline doing
optimisation / sampling / amortised inference over the wgridder.

**Sign convention** (matches ducc's `explicit_degridder` with
`divide_by_n=True`):

```
V(u, v, w) = Σ_{l,m} I(l, m) · exp(-2πi (u·l + v·m)) · exp(+2πi w (n-1))
```

The adjoint applies the conjugate phases and divides the output by
`n`. **Do not flip these signs** without re-deriving and updating
the ducc parity tests — and `vis2dirty` also takes an optional
`weights` arg (`(n_rows, n_chan)` real) that is multiplied into
visibilities before gridding (matches ducc's `wgt`).

---

## 2. Five-minute tour

```
jax-nufft/
├── README.md                 # user-facing docs, install, API, benchmarks
├── AGENTS.md                 # this file
├── LICENSE                   # Apache-2.0
├── pyproject.toml            # hatchling build, pytest config, ruff config
├── pixi.toml                 # pixi workspace + tasks (test, lint, format, …)
├── pixi.lock
├── .github/
│   └── workflows/test.yml    # CI: lint + fast suite on every push
├── docs/
│   ├── v0.1.1-plan.md        # formal v0.1.1 plan with rationale
│   └── v0.1.2-plan.md        # prioritised v0.1.2 performance plan
├── src/jax_nufft/
│   ├── __init__.py           # public API surface
│   ├── _version.py           # __version__
│   ├── _types.py             # Literal aliases (WStrategy, ChannelStrategy)
│   ├── _utils.py             # SPEED_OF_LIGHT, small helpers
│   ├── kernel.py             # exp-of-semicircle kernel + phi_hat table
│   ├── planning.py           # WGridderPlan dataclass + make_plan
│   └── wgridder.py           # dirty2vis / vis2dirty + per-channel helpers
└── tests/
    ├── conftest.py           # Telescope fixtures, synthetic_uvw, pytest options
    ├── test_smoke.py         # import smoke test
    ├── test_kernel.py        # phi, phi_hat, conditioning
    ├── test_planning.py      # plan shapes, window builder, pytree
    ├── test_against_dft.py   # forward DFT parity, parametrised over strategy
    ├── test_adjoint.py       # adjoint DFT parity + dot-product identity
    ├── test_against_ducc.py  # ducc parity across telescopes (slow tests gated)
    ├── test_boundary_planes.py  # windowed-vs-dense parity on edge cases
    ├── test_jax_integration.py  # jit / grad / vmap traceability
    └── test_benchmark_against_ducc.py  # opt-in pytest-benchmark suite
```

### Where to look first when picking up a task

| Task                                | Start here                                        |
|-------------------------------------|---------------------------------------------------|
| Change kernel choice                | `src/jax_nufft/kernel.py`                         |
| Tweak `n_w` or w-plane sampling     | `src/jax_nufft/planning.py` (see `W_OVERSAMPLE_X0`) |
| Add a new w-traversal strategy      | `src/jax_nufft/wgridder.py` + `_types.py`         |
| Add a parity test                   | `tests/test_against_ducc.py` or `_dft.py`         |
| Run benchmarks                      | `tests/test_benchmark_against_ducc.py` + README   |
| Add a benchmark fixture             | `tests/conftest.py` (telescope fixtures) + the benchmark file |
| Understand a design decision        | `docs/v0.1.1-plan.md` or `docs/v0.1.2-plan.md`    |

---

## 3. The wgridder algorithm, in 3 paragraphs

The radio interferometric measurement equation says each visibility
`V(u, v, w)` is the 3D Fourier transform of the sky brightness, sampled
at the baseline `(u, v, w)` in wavelengths. Direct evaluation is
`O(N_rows * N_pix^2)`; the wgridder reduces this to a stack of 2D
NUFFTs indexed by `w`-plane by approximating the `exp(2πi w (n-1))`
phase with a kernel-interpolation scheme.

The plan-time work picks the number of w-planes `n_w`, the kernel
half-width `W`, the kernel shape `beta`, the w-plane centres
`w_centers`, and a per-image correction `phi_hat_n` that compensates
for the kernel's image-domain footprint. The call-time work, for each
w-plane `k`, multiplies the image by the w-correction
`exp(2πi w_k (n-1)) / phi_hat_n`, runs a 2D NUFFT to land at the
visibilities, and weights each visibility by `phi((w-w_k) / scale)`.

The four `w_strategy` variants (`dense_scan` default, `dense_vmap`,
`windowed_scan`, `windowed_vmap`) differ only in how the w-plane loop
is structured (scan vs vmap) and whether each plane processes every
visibility (`dense_*`) or just a contiguous w-sorted slice (`windowed_*`).
All four are mathematically equivalent — they differ only in
floating-point reduction order.

For the underlying math (sign convention, phi_hat correction, kernel
parameters), see `README.md` &rarr; "Mathematical background" and the
docstrings in `src/jax_nufft/wgridder.py`.

---

## 4. The plan-then-call pattern (critical to get right)

```python
plan = make_plan(uvw, freq, image_shape, pixsize_l, pixsize_m, epsilon)
vis = dirty2vis(plan, image)            # JIT-cached
dirty = vis2dirty(plan, vis)            # JIT-cached separately
```

* `make_plan` is host-side (numpy) and returns a `WGridderPlan` — a
  frozen dataclass that's also a registered JAX pytree.
* Plan **static fields** (`n_l, n_m, n_chan, n_rows, n_w,
  w_kernel_width, beta, epsilon, pixsize_*, w_kernel_scale,
  max_window_size, window_padding_overhead, w_extent,
  is_constant_w`) live in the pytree aux_data and become part of
  the JIT cache key. This list must match `_plan_aux` in
  `planning.py` exactly — drift here is the most common source of
  "silent pytree corruption" bugs.
* Plan **traced fields** (`uvw_lambda, w_centers, n_minus_1,
  phi_hat_n, sort_perm, uvw_lambda_sorted, window_start,
  window_size, u_finufft, v_finufft`) are JAX device arrays.
  `u_finufft` / `v_finufft` are `(n_chan, n_rows)` precomputed
  FINUFFT-input coordinates (`2π · pixsize_* · uvw_lambda[..., axis]`,
  v0.1.2+). They add roughly `2 · n_chan · n_rows · sizeof(real)` to the
  plan's HBM footprint (about +33% on top of the dense `uvw_lambda*`
  arrays since each is now-half the shape). Sorted variants are not
  stored; the windowed helpers gather them via `plan.sort_perm` at
  scan time, trading half the memory for one gather per channel iter.
* This split is **load-bearing**: changing which fields are static vs
  traced affects JIT cache behaviour, error messages, and trace
  reuse. If you add a field, decide aux vs leaf deliberately and
  update both `_plan_aux` and the `register_pytree_node` flatten /
  unflatten functions.

The reused JIT cache means an optimisation loop that calls
`dirty2vis(plan, ...)` 1000 times with the same `plan` and same
`w_strategy` compiles once and runs the cached binary thereafter. The
benchmark numbers in the README reflect this steady-state regime.

### Invariant: window builder consistency

The windowed strategies rely on a contract between
`planning.make_plan` and `wgridder._channel_*_windowed`:

* `plan.sort_perm` is `argsort(uvw[:, 2])` (ascending, stable).
* `plan.uvw_lambda_sorted[c] = plan.uvw_lambda[c][sort_perm]`.
* `plan.window_start[c, k]` is the start index in the sorted array
  for the rows inside `[w_centers[k] - W/2 * dw, w_centers[k] + W/2 * dw]`.
* `plan.max_window_size` is a static int used as the
  `dynamic_slice` size — must be `>= max(plan.window_size)`.

If you touch any of these, run `tests/test_planning.py` and
`tests/test_boundary_planes.py` to confirm the contract still holds.

---

## 5. Strategies

| `w_strategy`      | Per-plane work          | Peak transient memory       | Notes                                          |
|-------------------|-------------------------|-----------------------------|------------------------------------------------|
| `dense_scan`      | `n_rows * W^2`          | `O(image_size + n_rows)`    | Default. v0.1 `"scan"` is a deprecated alias.  |
| `dense_vmap`      | `n_rows * W^2`          | `O(n_w * image_size)`       | v0.1 `"vmap"` is a deprecated alias.           |
| `windowed_scan`   | `max_window_size * W^2` | `O(image_size + n_rows)`    | v0.1.1; helps on adjoint when `n_w >> W`.      |
| `windowed_vmap`   | `max_window_size * W^2` | `O(n_w * image_size)`       | v0.1.1; rare wins, mostly for completeness.    |

The v0.1 names `"scan"` / `"vmap"` are accepted as deprecated aliases
that emit `DeprecationWarning` and resolve to their `dense_*`
counterparts (`_canonicalise_w_strategy` in `wgridder.py`). Plan to
remove these in v0.2.

`channel_strategy` is independently `"scan"` (default) or `"vmap"`.
The default `"scan"` is the safer choice across `n_chan` and memory
budgets; `"vmap"` can win on GPU with small `n_chan` because it
unrolls the channel loop into a single batched call, but it
allocates `n_chan` × per-channel transient memory. Don't change the
default without a GPU benchmark.

---

## 6. Testing conventions

```sh
pixi run -e test pytest                # fast unit tests, ~5 s
pixi run -e test pytest --runslow      # adds MWA_extended/MeerKAT parity
pixi run -e test pytest --runbench     # opt-in benchmarks (~2 min for one pointing)
pixi run -e dev lint                   # ruff check
pixi run -e dev format                 # ruff format (apply)
pixi run -e dev typecheck              # mypy (best-effort)
```

* `pyproject.toml` sets `filterwarnings = ["error"]` &mdash; any
  unexpected warning fails the test. If you intentionally emit a
  `DeprecationWarning`, test it with `pytest.warns(DeprecationWarning, …)`.
* Parity tolerances: `err < 10 * eps` for DFT parity and dense-vs-windowed
  forward; `err < 20 * eps` for ducc parity (ducc and jax both target
  `eps` independently so the gap is bounded by `~2*eps` to `~10*eps`).
  `err < 100 * eps` for dense-vs-windowed adjoint (different reduction
  order across NUFFT batches).
* Telescope fixtures live in `conftest.py`. `short_telescope_pointing`
  runs by default; `long_telescope_pointing` is gated behind
  `--runslow`. `bench_telescope_pointing` is gated behind `--runbench`.

When adding a feature, the right test files to update are:
- algorithmic correctness &rarr; `test_against_dft.py` (small problems)
- production correctness &rarr; `test_against_ducc.py` (telescope sweep)
- structural / API behaviour &rarr; `test_planning.py`,
  `test_jax_integration.py`
- regression on synthetic edge cases &rarr; `test_boundary_planes.py`

---

## 7. Benchmarking conventions

* `tests/test_benchmark_against_ducc.py` is gated behind `--runbench`.
* Plan and JIT compile are **excluded** from the timed window. The
  benchmark numbers represent per-call cost in a reuse-the-plan
  optimisation loop.
* ducc's Python API doesn't separate plan from execute, so ducc's
  numbers include its per-call planning. README explains the
  asymmetry.
* `--benchmark-group-by=param:bench_telescope_pointing` clusters all
  implementations (dense_scan, dense_vmap, windowed_scan,
  windowed_vmap, ducc) for one telescope-pointing into a single
  comparison table.
* `--bench-pointing={zenith,off30,both}` controls which pointings to
  run. Zenith is the cheap sanity sweep; `off30` is where the
  windowed gains show up.
* `extra_info` on each row reports `n_w`, `max_window_size`,
  `padding_overhead`, `w_strategy`, etc., for post-hoc analysis.

When proposing a perf change, the workflow is:

1. Run the benchmarks on the **current** state of the branch:
   ```sh
   pixi run -e test pytest tests/test_benchmark_against_ducc.py \
       --runbench --bench-pointing=both \
       -k "(bench_jax or bench_ducc) and not memory" \
       --benchmark-group-by=param:bench_telescope_pointing \
       --benchmark-json=/tmp/before.json -q
   ```
2. Apply the change.
3. Re-run with `--benchmark-json=/tmp/after.json`.
4. Diff. (`pytest-benchmark compare` works, or a small python script
   that loads both JSONs.)

This is exactly what was done to isolate Part 1 vs Part 2 in `v0.1.1`.

---

## 8. Verifying a change end-to-end

### Commit-per-step convention (applies to all plan docs)

When implementing a plan document (`docs/v0.1.X-plan.md`), follow
the per-step commit workflow:

1. **Every commit must leave `pixi run -e test pytest -q` green.**
   CI (`.github/workflows/test.yml`) runs on every push; a red
   commit blocks the next push.
2. **One sub-step = one commit.** Plan docs from v0.1.2 onwards
   include a "Sub-steps (one commit each)" section per Part with
   explicit file targets and acceptance gates. Honour those
   boundaries even if the next sub-step looks trivial — they exist
   so reverting any single commit leaves the repo working.
3. **Commit message format:** `v0.1.X Part N.M: <subject>` for
   per-step commits; `v0.1.X: <subject>` for cross-cutting commits
   (baseline, README, version bump).
4. **Plan-field changes require a same-commit checklist.** Any
   commit that adds or removes a field on `WGridderPlan` must, in
   the same commit, update: `_plan_aux`, the `register_pytree_node`
   flatten/unflatten in `planning.py`, AGENTS.md §4's static-vs-leaf
   list, and any test asserting the pytree leaf count (§11 "Don'ts"
   covers why).
5. **Don't squash per-step commits at PR time.** The history is the
   audit trail for which sub-step introduced any regression.

### Per-change checklist

The "ship a perf change" checklist:

1. `pixi run -e dev lint` &mdash; ruff clean.
2. `pixi run -e test pytest -q` &mdash; fast unit tests pass.
3. `pixi run -e test pytest -q --runslow` &mdash; slow ducc parity passes.
4. If a perf change, benchmark before/after as above.
5. If touching the JIT-cached path, eyeball compile time on a fresh
   plan (the script in the v0.1.1 README section "What the benchmark
   numbers include" measures this).
6. Update `README.md` if user-facing.
7. Update `AGENTS.md` (this file) if conventions or design decisions
   change.
8. For a tagged release: bump `src/jax_nufft/_version.py`, update the
   roadmap/history in this file to move the just-shipped items into
   §9 (history), and add a changelog entry to the corresponding
   `docs/v0.1.X-plan.md` "Release checklist" section.

---

## 9. History of decisions

### v0.1 (tag `v0.1.0`)

* Initial release: `make_plan` + `dirty2vis` + `vis2dirty`.
* `w_strategy` was `"scan"` (default) or `"vmap"`. Channel strategy
  identical.
* Plan sampling: `dw = (1/W) / max|n-1|` &mdash; W-dependent. Chosen
  empirically to keep `phi_hat` in its well-conditioned region after
  v0.1's first attempts hit safety floor errors at wider eta.
* Parity vs ducc: `< 10*eps` across the four built-in telescopes.

### v0.1.1 (tag `v0.1.1`, branch `v0.1.1`)

Two improvements (`docs/v0.1.1-plan.md` has the full motivation):

**Part 1: standard n_w.** Reverted v0.1's W-dependent oversampling to
a W-independent `x0 = 0.25` (matching ducc's `ofactor=2` kernels for
our FINUFFT `sigma=2` kernel choice). `n_w` shrinks to roughly `4/W`
of v0.1's value (since v0.1 used `x0 = 1/W` and `n_w_inner ∝ 1/x0`).
The phi_hat table compensates with a W-dependent oversample
(`phi_hat_oversample_for_w` in `kernel.py`).

**Part 2: windowed per-plane scan.** Visibilities are sorted by `w`
in `make_plan`; each w-plane processes only the contiguous slice of
the sorted array that falls inside the kernel support. New strategies
`windowed_scan` (low memory) and `windowed_vmap` (high memory). Plan
fields `sort_perm`, `uvw_lambda_sorted`, `window_start`, `window_size`,
`max_window_size`, `window_padding_overhead` support this. No explicit
mask: the kernel weight `phi(z)` is zero outside the support.

**Measured wins** (Mac M-series CPU, eps=1e-6):
* Part 1: 1.05–1.62x across telescopes/pointings (forward + adjoint).
* Part 2: ~flat on forward, 1.05–1.53x on adjoint at off-zenith.
* Combined: 1.6–1.86x at the off-zenith adjoint workloads.

**Renames.** `scan` &rarr; `dense_scan`, `vmap` &rarr; `dense_vmap`
(old names accepted as deprecated aliases for one release; remove in
v0.2).

### v0.1.2 (branch `feature/v0.1.2-perf-gpu`)

Performance release, GPU-focused. Full motivation in
`docs/v0.1.2-plan.md`; six Parts, landed in per-sub-step commits.

**Part 1: sorted-order windowed forward.** `_channel_forward_windowed`
accumulates in w-sorted order with contiguous-slice updates and
unsorts once at the end, instead of building a full `(n_rows,)` zero
vector and scattering per plane.

**Part 2: constant-w fast path.** `make_plan` collapses zero-w-extent
data to a single plane (`n_w == 1`, `is_constant_w == True`), giving a
~`W+1` win on coplanar / zenith-like workloads. All four `w_strategy`
values reduce to the same single-plane work in this regime.

**Part 3: precomputed FINUFFT coordinates.** `u_finufft` / `v_finufft`
are stored as plan leaves (option (b): dense-order only; windowed
helpers gather via `sort_perm`) rather than recomputed inside every
channel call. Pytree leaf count bumped accordingly.

**Part 4: `w_strategy="auto"`.** Opt-in value resolved to a canonical
name *before* the JIT boundary (so cache sharing is preserved), via
`_auto_w_strategy`. Defaults stay on `dense_scan`.

**Part 5: GPU benchmark suite.** `tests/bench_harness.py` (async-aware
timing, hardware fingerprint, HBM capture via `device.memory_stats()`)
plus `tests/test_benchmark_gpu.py`, gated behind `--runbench-gpu`.
Stable JSON schema documented in `docs/benchmarks/README.md`.

**Part 6: GPU sweep + platform-aware defaults.** 160-cell GH200 sweep
(`docs/benchmarks/v0.1.2-baseline-gpu.json`) showed `_scan` variants
5-30x slower than `_vmap`, `dense_vmap` winning 17/20 cells, and
`windowed_vmap` winning only the 50k-row `GH200_large` fixture.
`_auto_w_strategy` is now platform-aware (`jax.devices()[0].platform`);
unknown platforms fall back to the CPU heuristic.
`tests/test_auto_strategy_acceptance.py` asserts the GPU pick is within
15% of best on every cell.

**Measured wins.** CPU (Mac M-series, eps=1e-6): Parts 1-3 carry the
v0.1.1 windowed-adjoint gains forward; constant-w gives ~`W+1` on
coplanar data. GPU (GH200): `dense_vmap` / `windowed_vmap` dominate;
the auto heuristic picks the measured winner in 20/20 (op, fixture)
cells.

---

## 10. Roadmap

### Strategic intent

Future jax-nufft releases will **add other optimised NUFFT routines
built on `jax-finufft`** beyond the wgridder. Candidates:

* Type-3 NUFFT wrappers for direct non-uniform-to-non-uniform
  transforms (useful for visibility-domain re-gridding and for some
  inverse-problem formulations).
* A planned "build once, call many" wrapper around `jax-finufft`'s
  type-1/type-2 NUFFTs that pre-caches the JIT-compiled call
  alongside the FINUFFT plan, fitting the same plan-then-call
  pattern this codebase already uses for the wgridder.
* Pulsar-style 1D NUFFT helpers for time-series analysis (different
  problem domain, same underlying primitives).
* Polarisation-aware variants (currently flagged as out of scope for
  v1).
* Faceting + BDA support for the wgridder (out of scope for v1; major
  algorithmic additions, would be v0.2+).

The repo name and module layout (`jax_nufft.<feature>`) anticipate
this expansion. Treat `wgridder.py` as the first of N feature modules
rather than the whole package.

### Plan-doc template

Plan docs in this repo follow a consistent template — when writing the
next one (`v0.1.3-plan.md`, etc.) preserve the structure:
**Scope / Non-goals / Baseline / numbered Parts (Problem / Proposed
change / Tests / Expected impact / Risk) / Lower-priority follow-ups /
Release checklist**. The Parts are ordered by expected value;
implementation risk is recorded per-Part rather than driving the
ordering. The v0.1.2 plan (`docs/v0.1.2-plan.md`, shipped — see §9) is
a worked example.

### Other ideas (not yet sized)

These are seeds for later releases, not v0.1.2 candidates:

* **Reduce plan memory.** `plan.uvw_lambda_sorted` doubles the coord
  storage (`(n_chan, n_rows, 3)` floats). For very large
  `(n_chan, n_rows)` we could keep only `sort_perm` and apply it at
  scan time, trading one gather per scan iter for half the plan
  memory.
* **Plan-time pre-compilation cache (JIT-level).** Currently each
  `dirty2vis(plan, ..., w_strategy=...)` JIT-compiles on the first
  call. A `jax_nufft.compile_plan(plan, w_strategy=...)` helper that
  forces the compile at plan-time would let optimisation pipelines
  amortise compile cost into setup. This is distinct from the
  FINUFFT-plan reuse idea under "Strategic intent" above, which is
  about caching cuFINUFFT/FINUFFT plans (a C-library object) across
  calls — solving the same problem at a different layer.
* **Per-channel `n_w`.** Currently `n_w`, `dw`, `w_kernel_scale` are
  shared across channels (taken from the worst-case w-extent). For
  wide-bandwidth observations this can be wasteful at the low end of
  the band. The cost is a bigger API surface and more JIT cache
  entries.

---

## 11. Don'ts

* Don't change `phi_hat_oversample`'s default behaviour without
  re-running the phi_hat-conditioning tests in `tests/test_kernel.py`
  and confirming `min(phi_hat) > safety_floor` across `W in {4, 6, 8, 10}`.
  v0.1.1 picks the oversample as a function of `W` precisely because
  v0.1's constant-32 default broke conditioning at the wider eta
  range. (`safety_floor` is defined in `src/jax_nufft/kernel.py`.)
* Don't bypass `_canonicalise_w_strategy` by hard-coding the v0.1
  names in new code. They are user-input aliases only; the internal
  dispatch uses the canonical names.
* Don't add fields to `WGridderPlan` without updating BOTH
  `_plan_aux` (for static fields) AND the `register_pytree_node`
  flatten / unflatten (for traced fields). The pytree registration
  is positional; mismatching it produces silent corruption rather
  than a clean error. See §8 commit-per-step rule 4 for the full
  plan-field checklist (this don't is its load-bearing subset).
* Don't introduce mutable state (caches, module-level dicts, etc.)
  in the call path. The whole point of the plan-then-call API is that
  every per-call state is in the plan.
* Don't promote `windowed_*` to default `w_strategy` without
  CPU+GPU benchmark validation. The default is the value most users
  inherit by not specifying — changing it changes the behaviour of
  every existing caller.
