# Repository guide for coding agents

This file is the orientation doc for Claude Code / other coding agents
picking up work on this repo. It covers what the project is, how the
code is laid out, the conventions and invariants you need to respect,
how to run tests / benchmarks, the history of design decisions taken
in `v0.1` and `v0.1.1`, and an ideation list for `v0.1.2`.

For end-user-facing documentation, see `README.md`. For the most recent
formal release plan, see `docs/v0.1.1-plan.md`.

---

## 1. What this project is

`jax-nufft` is a pure-JAX implementation of the wgridder algorithm for
radio interferometric imaging, built on `jax-finufft`. It exposes two
public operators:

* `dirty2vis(plan, image)` — forward (image &rarr; visibilities).
* `vis2dirty(plan, vis)` — adjoint (visibilities &rarr; image).

Both are fully traceable through `jax.jit`, `jax.vmap`, `jax.grad`. The
reference baseline for correctness is `ducc0.wgridder`; we target
relative L2 error `< 10 * epsilon`.

The strategic value-add over ducc is **differentiability and the GPU
port via cuFINUFFT**, not raw CPU speed (ducc is consistently faster
on CPU). The intended user is a downstream JAX pipeline doing
optimisation / sampling / amortised inference over the wgridder.

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
├── .github/                  # CI
├── docs/
│   └── v0.1.1-plan.md        # formal v0.1.1 plan with rationale
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
| Understand a design decision        | `docs/v0.1.1-plan.md`                             |

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
* Plan **static fields** (`n_l, n_m, n_w, w_kernel_width, beta,
  epsilon, pixsize_*, w_kernel_scale, max_window_size,
  window_padding_overhead`) live in the pytree aux_data and become
  part of the JIT cache key.
* Plan **traced fields** (`uvw_lambda, w_centers, n_minus_1,
  phi_hat_n, sort_perm, uvw_lambda_sorted, window_start,
  window_size`) are JAX device arrays.
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

---

## 6. Testing conventions

```sh
pixi run -e test pytest                # 118 tests, ~5 s
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
our FINUFFT `sigma=2` kernel choice). `n_w` drops by `W/4`. The
phi_hat table compensates with a W-dependent oversample
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

### v0.1.2 ideation

Two concrete improvements to the current codebase that look ripe for
a v0.1.2 release. Both are independent and additive; either could be
shipped alone.

#### Idea 1: Auto-select `w_strategy` from plan diagnostics

The v0.1.1 benchmark sweep showed that **windowed strategies help on
the adjoint when `n_w >> W` and the w-distribution is reasonably
uniform**, while at zenith (`n_w ~ W`) windowed and dense are
essentially tied. Today users have to know this and pick by hand. A
small heuristic in `make_plan` (or in `dirty2vis` / `vis2dirty`)
could pick automatically:

```python
def _auto_w_strategy(plan, *, is_adjoint: bool) -> str:
    if plan.n_w <= plan.w_kernel_width + 2:
        return "dense_scan"               # too few planes to slice
    if plan.window_padding_overhead > 5.0:
        return "dense_scan"               # pathological w-distribution
    if is_adjoint and plan.n_w / plan.w_kernel_width > 2.0:
        return "windowed_scan"            # adjoint where windowing wins
    return "dense_scan"
```

Concrete tasks:
* Add `w_strategy="auto"` as a new accepted value that runs this
  heuristic at call time (cheap; just a Python `if`).
* Make it the default for `dirty2vis` / `vis2dirty` once the
  heuristic is benchmark-validated.
* Add a small benchmark-driven test that the heuristic doesn't pick a
  loser on any of the four telescopes x both pointings.
* Document `plan.window_padding_overhead` and the auto rules in the
  README.

Risk: low. The dense path is always a valid fallback, and the
heuristic only narrows choice — it doesn't change algorithmic
behaviour.

Expected impact: 1.2–1.5x adjoint speedup on the typical off-zenith
workloads, for users who haven't bothered to pick by hand.

#### Idea 2: GPU benchmark sweep + cuFINUFFT-aware defaults

All v0.1.1 strategy decisions are based on CPU measurements. The
`pixi.toml` already has a `gpu` feature gated on Linux+CUDA, but no
GPU benchmarks have been run. The most likely surprises:

* `windowed_vmap` may become the clear winner on GPU even though it's
  marginal on CPU, because the type-1/type-2 NUFFT throughput on
  cuFINUFFT scales much more favourably with batch size than CPU
  FINUFFT does, *and* the `n_w * image_size` transient memory that
  hurts CPU `vmap` variants is fine on consumer GPUs.
* The scatter-add `vis_acc.at[rows_k].add(contrib)` in the windowed
  forward must compile to an efficient XLA scatter on GPU. If it
  shows up hot in nsys profiling, alternatives are
  `jax.ops.segment_sum` (we have sorted indices, so this is a free
  upgrade) or a custom XLA op.

Concrete tasks:
* Provision a Linux+CUDA machine, install `pixi run -e gpu`, run the
  benchmark sweep at `--bench-pointing=both`.
* Compare GPU rankings vs CPU rankings; if the rankings invert,
  consider making the auto-select heuristic platform-aware
  (`jax.devices()[0].platform`).
* If scatter-add is hot, replace with `segment_sum`.

Risk: medium. The investigation may surface real cuFINUFFT
ergonomics issues (e.g., compile time per `(uvw, plan)` pair) that
need a workaround.

Expected impact: hard to predict in advance; the goal is to get a
defensible "this is the right default on GPU" answer rather than a
specific speedup.

### Other ideas (not yet sized)

These are seeds for later releases, not v0.1.2 candidates:

* **Reduce plan memory.** `plan.uvw_lambda_sorted` doubles the coord
  storage (`(n_chan, n_rows, 3)` floats). For very large
  `(n_chan, n_rows)` we could keep only `sort_perm` and apply it at
  scan time, trading one gather per scan iter for half the plan
  memory.
* **Plan-time pre-compilation cache.** Currently each
  `dirty2vis(plan, ..., w_strategy=...)` JIT-compiles on the first
  call. A `jax_nufft.compile_plan(plan, w_strategy=...)` helper that
  forces the compile at plan-time would let optimisation pipelines
  amortise compile cost into setup.
* **`segment_sum` everywhere.** Replace the scatter-add in
  `_channel_forward_windowed` with `jax.ops.segment_sum`. Already
  noted as a GPU candidate; may also help CPU.
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
  range.
* Don't bypass `_canonicalise_w_strategy` by hard-coding the v0.1
  names in new code. They are user-input aliases only; the internal
  dispatch uses the canonical names.
* Don't add fields to `WGridderPlan` without updating BOTH
  `_plan_aux` (for static fields) AND the `register_pytree_node`
  flatten / unflatten (for traced fields). The pytree registration
  is positional; mismatching it produces silent corruption rather
  than a clean error.
* Don't introduce mutable state (caches, module-level dicts, etc.)
  in the call path. The whole point of the plan-then-call API is that
  every per-call state is in the plan.
* Don't promote `windowed_*` to default `w_strategy` without
  CPU+GPU benchmark validation. The default is the value most users
  inherit by not specifying — changing it changes the behaviour of
  every existing caller.
