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
target is DFT parity within `2 * epsilon`; ducc parity, windowed
vs. dense, and adjoint reduction-order tolerances are looser — see
§6 for the full table. These contracts are for the default float64 plan
with `jax_enable_x64` enabled; `float32` / `complex64` inputs are
accuracy-limited and do not meet them — see issues #11 and #13.

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
│   ├── _utils.py             # SPEED_OF_LIGHT
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
half-width `W` (`ceil(-log10(eps/10))`, so `n_w = n_w_inner + W` grows
by one plane per requested digit), the kernel shape `beta`, the
w-plane centres `w_centers`, and a per-image correction `phi_hat_n`
that compensates for the kernel's image-domain footprint. The
caller's `epsilon` is a budget shared by two error sources, so the
(u,v) NUFFT is asked for `epsilon / 10` (`_nufft_epsilon` in
`wgridder.py`) and the w-kernel carries the rest. The call-time work, for each
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
  w_kernel_width, beta, epsilon, pixsize_*, w_kernel_scale, nshift, w0,
  max_window_size, window_padding_overhead, live_row_count,
  empty_plane_count, w_extent,
  is_constant_w, real_dtype, complex_dtype`) live in the pytree
  aux_data and become part of the JIT cache key. This list must
  match `_plan_aux` in `planning.py` exactly — drift here is the
  most common source of "silent pytree corruption" bugs.
  `real_dtype` / `complex_dtype` (issue #11) are `np.dtype` objects
  recording the precision requested via `make_plan(dtype=...)`;
  being aux data is what keeps a float32 and a float64 plan on
  separate JIT cache entries. `nshift` (issue #16) is the `n-1`
  centring offset `-(nm1_max + nm1_min) / 2`; it is `0.0` on the
  constant-w fast path, which cannot benefit from it. `w0` (issue #16
  follow-up) is the w-range midpoint: the plane loop works in
  `delta = w - w0` so no phase it exponentiates scales with the
  absolute w, and the constant part leaves as the `w0_screen` leaf.
* Plan **traced fields** (`uvw_m, inv_lambda, w_centers_rel,
  n_minus_1_shifted, w0_screen, phi_hat_n, sort_perm, window_start`
  — 8 leaves, in that flatten order) are JAX device arrays.
  `uvw_m` is `(n_rows, 3)` in **metres**, in input row order, and
  `inv_lambda` is `(n_chan,)` = `freq / c` (issue #23). Nothing per
  `(channel, row)` is stored: the per-channel FINUFFT coordinates and
  w in wavelengths are derived inside the JIT by
  `wgridder._channel_ft_coords` — `u_ft = (2π · pixsize_l ·
  inv_lambda[c]) · uvw_m[:, 0]`, likewise `v_ft`, and
  `w = inv_lambda[c] · uvw_m[:, 2]` — three multiplies per row per
  channel, invisible next to the `n_w` NUFFTs. That replaced
  `uvw_lambda` / `uvw_lambda_sorted` `(n_chan, n_rows, 3)` and
  `u_finufft` / `v_finufft` `(n_chan, n_rows)`: 12.98 MB → 2.42 MB of
  leaves for a 16-channel, 10k-row, 256² plan, 3.9 GB → 31 MB at
  64 channels × 1M rows. No *sorted* coordinate array is stored either;
  the sort is by w in metres, so the permutation is
  channel-independent and the windowed helpers gather
  `uvw_m[sort_perm]` once per call.
  `n_minus_1_shifted` (issue #16) is `n_minus_1 + nshift` and is the
  grid the per-plane phase and the `phi_hat` argument are evaluated
  on; the adjoint's `1/n` output factor must keep using the *unshifted*
  grid (`n = n_minus_1 + 1`) — using the shifted one there is a silent,
  test-passing gain error.
  `w_centers_rel` and `w0_screen` (issue #16 follow-up) are the
  plane centres measured from `w0` and the image-domain screen
  `exp(2πi·w0·(n-1))`. Both are built at plan time *at small
  magnitude* / with exact range reduction; deriving either from its
  absolute counterpart at call time reintroduces the large-|w|
  cancellation they exist to remove.
* **Non-leaf compatibility accessors** (issue #23): `plan.n_minus_1`
  (`n_minus_1_shifted - nshift`), `plan.w_centers` (`w0 +
  w_centers_rel`) and `plan.uvw_lambda` (`uvw_m[None] *
  inv_lambda[:, None, None]`) are `@property`, not fields. They are
  documented plan attributes that tests and downstream code read, but
  they are *not* pytree leaves and must not be treated as such.
  Nothing under `src/` may read `plan.uvw_lambda` in an operator path
  — materialising it is exactly the `(n_chan, n_rows, 3)` allocation
  issue #23 deleted, and doing so would undo the change while every
  test still passed. `window_size` was removed outright with no
  accessor: it was diagnostic-only, feeding the (static)
  `max_window_size` and `window_padding_overhead` at plan time.
  Issue #43 added `live_row_count` and `empty_plane_count` to that
  diagnostic and kept them **scalars** for the same reason — two static
  ints is the whole budget, and reintroducing a `(n_chan, n_w)` array
  in any form would undo #23. `tests/test_planning.py` pins the leaf
  set at eight entries, which is what makes that a checked constraint
  rather than an intention.
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

* `plan.sort_perm` is `argsort(uvw[:, 2])` (ascending, stable). The
  sort key is w in *metres*, so the same permutation serves every
  channel (`freq[c]/c > 0` is monotonic) — which is why no sorted
  coordinate array is stored and the helpers gather
  `plan.uvw_m[plan.sort_perm]` once per call (issue #23).
* `plan.window_start[c, k]` is the start index in the sorted array
  for the rows inside `[w_centers[k] - W/2 * dw, w_centers[k] + W/2 * dw]`.
  **Invariant: every window contains every row the dense path gives a
  non-zero kernel weight.** That is what makes the windowed and dense
  strategies agree, and it holds *by construction*, not by luck. The
  builder places its boundaries in the same *relative* coordinate
  (`w - w0`) the operator uses and forms the per-channel w in the
  plan's `real_dtype`, then widens each boundary by
  `boundary_margin` (load-bearing) and by one row at each end
  (redundant insurance against the margin's derivation). The widening is
  load-bearing since issue #23: the operator derives `w` as a
  multiply immediately followed by a subtract, which XLA may contract
  into a single FMA — one rounding where the builder's numpy steps
  round twice — so host and device values differ on a fifth to a
  quarter of rows (measured: ~23% on MWA_extended, ~18% on MeerKAT).
  Before #23 the device subtracted `w0` from a *stored, already
  rounded* product, so the two agreed bit for bit and no margin was
  needed. Do not "restore" bit-equality here; it is not achievable
  against a fusing compiler. Widen instead: over-inclusion is free
  (`phi(|z| > 1) = 0` contributes nothing), under-inclusion is a
  silent `phi(z = ±1) = exp(-beta)` (~1e-7 at W=7) windowed-vs-dense
  mismatch, ~1e-9 in relative L2 against a 1e-11 contract.
  `boundary_margin` must dominate **two** terms, not one: the
  host/device gap (~`3u · w_abs_scale`) *and* `2u · w_kernel_scale`,
  because the operator tests `|fl(fl(w − w_k) / S)| ≤ 1` rather than
  comparing `w` against the edge, so a row can be in support with an
  exact `|w − w_k|` up to `S(1 + 2u)`. That second term is not
  bounded by `w_abs_scale` in general; it is bounded exactly when a
  window is a strict subset of the rows, which is the only case where
  a drop is possible — see the derivation in `planning.py`, and
  re-derive it if `w_kernel_scale` or the plane count ever changes.
  Pinned by `test_windowed_dense_parity_at_window_edge`, whose
  fixture places **two adjacent rows** in the ulp-wide band where
  host and device disagree: one row is rescued by the `±1`-row
  widening on its own, so a one-row fixture gates the margin not at
  all.
* `plan.max_window_size` is a static int used as the
  `dynamic_slice` size — must be `>=` every window's length, and is
  computed from the widened windows above, so it already carries the
  two extra rows. The per-window lengths themselves are plan-time
  locals, not a leaf (issue #23 removed `window_size`).
* `plan.window_padding_overhead` is `n_chan * n_w * max_window_size /
  plan.live_row_count` (issue #43): the row-work a windowed traversal
  actually does — every `(channel, plane)` step slices a *static*
  `max_window_size` rows — over the row-work it cannot avoid.
  `live_row_count` is measured on the **unpadded** support, i.e. from
  a second pair of `searchsorted` calls per channel that see neither
  `boundary_margin` nor the `±1` clamp. That is the whole point: the
  widened rows are work, so they belong in the numerator, but they lie
  outside nominal support, so counting them as irreducible understates
  the waste. The two widenings are not outside it equally: a clamp row
  is outside by two whole rows and the kernel does ignore it, while a
  margin row is outside by a few ulps and the kernel may not — the
  margin exists because the device might place such a row inside
  support. Both are excluded regardless, the denominator being what the
  host can establish is irreducible. Note also that `live_row_count` is
  a host-side **nominal-support** count, not a census of applied
  weights — the name is shorthand, and slightly stronger-sounding than
  the measurement supports. The operators derive their own `w` in the
  JIT (FMA contraction; single precision throughout on a float32 plan)
  and test `|z| <= 1`; measured against that compiled expression over
  the calibration grid the two counts differ on 20 of 40 cells in
  float64 and 7 of 10 in float32, never by more than 3 incidences or
  0.19%. That is far below the resolution a work ratio is read at, and
  an exact census would have to be taken against a compiled executable,
  which does not exist at plan time. Through v0.1.2 the denominator
  was the mean of the *padded* window lengths and counted the widening
  as irreducible, understating the waste by up to 17% on the review
  fixtures. In practice it is the `±1` clamp that the
  correction removes — `boundary_margin` is a few ulps of the absolute
  w scale and catches a row on twelve of the forty float64 calibration
  cells, one or two rows each; on MWA_extended off30 at eps 1e-3 it
  catches none and the entire 487-row gap is the clamp (`2 · n_w =
  496`, less end-clipping). Both are excluded regardless — that split
  is a measurement, not a rule. The
  clamp is also why `empty_plane_count` has to be measured here rather
  than inferred downstream: it guarantees `window_size >= 1`, so a
  plane holding no rows is otherwise indistinguishable from one
  holding a single row.

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
pixi run -e test pytest --runsweep -s tests/test_accuracy_sweep.py  # exact-DFT accuracy sweep
pixi run -e test pytest --runbench     # opt-in benchmarks (~2 min for one pointing)
pixi run -e dev lint                   # ruff check
pixi run -e dev format                 # ruff format (apply)
pixi run -e dev typecheck              # mypy (best-effort)
```

* `pyproject.toml` sets `filterwarnings = ["error"]` &mdash; any
  unexpected warning fails the test. If you intentionally emit a
  `DeprecationWarning`, test it with `pytest.warns(DeprecationWarning, …)`.
* **Tolerance contract** (current as of issue #10; all bounds below are
  for the default float64 plan with `jax_enable_x64` enabled -- `float32`
  / `complex64` inputs are accuracy-limited and do not meet them, issues
  #11, #13):

  | Comparison                                   | Bound          | Measured (this repo, 2026-09 review)       |
  |-----------------------------------------------|----------------|---------------------------------------------|
  | vs. exact DFT (forward + adjoint)              | `err < 2 * eps`  | ~4x eps at `1e-6..1e-8` before #9's width-rule fix; `0.67x eps` worst cell after it; `1.47x eps` worst cell after #16's nshift centring (MWA_extended off30, eps=1e-12, adjoint) -- still inside the contract, but the headroom is now ~1.4x, not ~3x |
  | vs. ducc0 (forward + adjoint, constant-w fast path) | `err < 3 * eps`  | ducc0 lands at `~0.1 * eps` against the DFT; the `3 * eps` gap is dominated by our own `2 * eps` budget |
  | Strategy equivalence (`w_strategy` x `channel_strategy`, `tests/test_strategies_equivalent.py`) | `err < 1e-11` (float64), `err < 1.3e-6` (float32) | worst pairwise difference between any two of the eight combinations, 3 short fixtures x eps in {1e-4,1e-6,1e-8}: float64 2.0e-13 (MWA_compact off30, eps=1e-6); float32 1.21e-7 (MWA_compact off30, eps=1e-8) -- float32 bound is ~10x that measurement, see the module docstring |
  | Adjointness (dot-product identity, dense-vs-windowed adjoint) | `err < 1e-11`  | dot-product residual 1e-16 .. 7e-13 (ducc0 itself: 1e-14 .. 8e-11, for comparison only) |

  The DFT bound is an accuracy *contract*, not a comparison -- the DFT is
  the definition of the answer, whereas ducc0 is a second implementation
  with its own error budget. The strategy-equivalence and adjointness
  bounds are reduction-order comparisons between mathematically identical
  operators (AGENTS.md §3), so unlike the two above they are
  **eps-independent**: `1e-11` is roughly `100 * n_rows * eps_machine` and
  does not loosen as `eps` tightens. Issue #10 tightened both from
  `100 * eps` (1e-4 at eps=1e-6 -- ten orders of magnitude looser than
  what the code actually achieves, wide enough to hide a bug that dropped
  a whole window row from the adjoint) after measuring the numbers in the
  table above. Strategy equivalence gets a second, separate bound for
  float32 (`1.3e-6`) rather than reusing `1e-11`: float32 has essentially
  no headroom above its own rounding floor, so `1e-11` simply does not
  hold there. If a bound doesn't hold on a given platform (Linux CI sorts
  FINUFFT bins differently from macOS arm64), the fix is to measure the
  worst case there and set the bound at 10x it, not higher -- record the
  measurement in a comment, don't just loosen it to whatever passes.
* **Adjointness is necessary but not sufficient &mdash; it cannot
  replace DFT parity.** The dot-product identity
  `<A x, y> == <x, A^H y>` constrains the forward and the adjoint
  relative to *each other*, not either of them relative to the truth.
  Any error the two share &mdash; a common-mode error &mdash; cancels out
  of the identity and passes it cleanly while both operators are badly
  wrong. Issue #16's follow-up is the worked example: with a large common
  `w` offset the plane phase and the `nshift` compensating phase were
  cancelling catastrophically, the forward's relative L2 error against the
  exact DFT was `1335 * eps`, and the dot-product identity passed at
  **every** offset, because the adjoint made the identical rounding error
  in the identical place. Both sign-mutation checks on that branch tell
  the same story from the other side: mutations that break the *pairing*
  (flip one operator's compensation) are caught by adjointness
  immediately, while a mutation that moves both together is not.
  So: adjointness catches pairing bugs, DFT parity catches shared-mode
  bugs, and a change to the shared phase machinery needs both. When you
  add a new phase factor, add a DFT-parity test for it &mdash; see
  `tests/test_nshift.py::test_large_common_w_offset_stays_in_contract`,
  which uses a delta image at the phase centre (exact answer: 1 for every
  visibility, analytically) so that the residual *is* the phase error,
  with no reference implementation in the loop.
* Telescope fixtures live in `conftest.py`. `short_telescope_pointing`
  runs by default; `long_telescope_pointing` is gated behind
  `--runslow`. `bench_telescope_pointing` is gated behind `--runbench`.
* `tests/test_accuracy_sweep.py` is the reusable accuracy harness: the
  seven review fixtures x eight epsilon values x forward/adjoint,
  measured against the exact DFT and asserted at `<= 2 * eps`. It is
  gated behind `--runsweep` and skipped without `jax_enable_x64`.
  Whenever an issue says "re-run the accuracy sweep" it means
  `pytest --runsweep -s tests/test_accuracy_sweep.py`; the printed
  table is what goes into the PR description.
* **Precision leg.** `JAX_ENABLE_X64=0 pytest -q` runs the fast suite in
  float32 (`tests/conftest.py`'s `X64` / `tol()` / `real_dtype` /
  `complex_dtype`, issue #11). `conftest.collect_ignore` names the
  modules that are not yet precision-aware and are skipped at collection
  time under that leg -- a shrinking backlog, not a design; new test
  modules should be written precision-aware from the start (build the
  plan with `dtype=real_dtype`, scale tolerances with `tol(f64, f32)`)
  and never added to that list. `tests/test_strategies_equivalent.py` is
  the first module written this way.

When adding a feature, the right test files to update are:
- algorithmic correctness &rarr; `test_against_dft.py` (small problems)
- production correctness &rarr; `test_against_ducc.py` (telescope sweep)
- structural / API behaviour &rarr; `test_planning.py`,
  `test_jax_integration.py`
- regression on synthetic edge cases &rarr; `test_boundary_planes.py`
- cross-strategy regression (any change to the w-plane or channel loop
  structure) &rarr; `test_strategies_equivalent.py`

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
  re-running the phi_hat tests in `tests/test_kernel.py`:
  `test_phi_hat_oversample_schedule_is_pinned` (the full returned
  schedule, `W` from 4 through 15, pinned to concrete integers -- cheap,
  and the only test that covers `W = 15` at all);
  `test_phi_hat_conditioning_at_v011_eta_max` (`min(phi_hat) >
  safety_floor` across `W in {4, 6, 8, 10, 11, 13}`); and
  `test_phi_hat_interpolation_error_off_node` / its `--runslow`
  counterpart `test_phi_hat_interpolation_error_off_node_slow`
  (`< eps/10` off the table nodes; `W in {4, 8, 10}` run by default,
  `W in {11, 12, 13, 14}` gated behind `--runslow` because their
  4x-oversample reference tables get big -- see
  `phi_hat_oversample_for_w`'s docstring for the memory accounting;
  `W = 15` is skipped even there, its reference alone pushing the
  `--runslow` leg from ~1.2 GB to ~5.5 GB peak memory footprint, so the
  schedule pin above is its only regression guard). v0.1.1 picks the
  oversample as a function of `W` because v0.1's
  constant-32 default broke conditioning at the wider eta range; #9
  added the doublings above `W = 8` because `phi_hat_n` is divided
  into the image, so a 5.9e-11 table error at `W = 13` showed up as
  14x epsilon in the operator output. A later review found the
  original doubling-every-digit schedule one doubling too generous
  throughout (doubling every *second* digit is enough); the current
  rule in `phi_hat_oversample_for_w` roughly halves the table (and
  its build-time peak host memory) at each width above `W = 8`.
  Node-aligned
  sampling hides that, which is why the off-node test exists.
  (`safety_floor` is defined in `src/jax_nufft/kernel.py`.)
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
