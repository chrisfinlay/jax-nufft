# jax-nufft

JAX-native wgridder for radio interferometric imaging.

> **Status:** v0.1.1. API stable; planning-side changes only between
> v0.1 and v0.1.1, plus two new opt-in `w_strategy` values. v0.1.2 adds
> a third opt-in value, `"auto"`, which resolves to one of the four
> canonical strategies via a plan-based heuristic; defaults are unchanged.

## Overview

`jax-nufft` provides a pure-JAX implementation of the wgridder algorithm for
radio interferometric imaging, expressed as a stack of 2D non-uniform FFTs
indexed by w-plane. It is built on top of [`jax-finufft`][jaxfinufft] for the
underlying NUFFT primitives.

The package is aimed at radio interferometric pipelines that need to compose
the wgridder with other JAX-traceable operators &mdash; for example calibration,
self-cal, or amortised inference networks. Concretely:

- Full `jax.jit`, `jax.vmap`, and `jax.grad` traceability (forward and reverse
  mode).
- CPU execution out-of-the-box (Mac and Linux), and GPU execution via the
  cuFINUFFT-enabled `jax-finufft` build with no code changes.
- Multi-channel visibilities (`Nrow x Nchan`) with shared per-row `uvw` and
  per-channel frequency.
- Optional per-visibility weights of shape `(Nrow, Nchan)`.
- Output that matches `ducc0.wgridder` to within ~10x the requested accuracy
  `epsilon` for both `dirty2vis` and `vis2dirty`.

Polarisation handling is **out of scope for v1**.

## Installation

The package is intended to be installable via `pip` once published:

```sh
pip install jax-nufft           # CPU
pip install 'jax-nufft[gpu]'    # GPU (Linux + CUDA 12)
```

For development, the recommended workflow uses [pixi](https://pixi.sh/):

```sh
git clone https://github.com/chrisfinlay/jax-nufft.git
cd jax-nufft
pixi run -e test pytest         # run the full test suite
pixi run -e dev format          # format with ruff
```

The `test` environment additionally installs `ducc0`, used as a reference for
parity tests.

## Quick start

```python
import jax.numpy as jnp
import numpy as np

from jax_nufft import dirty2vis, make_plan, vis2dirty

# Synthetic problem
n_l = n_m = 128
n_rows = 500
freq = np.array([1.4e9])
pixsize = np.deg2rad(20.0) / n_l  # 20 deg FoV
rng = np.random.default_rng(0)
uvw = rng.normal(scale=80.0, size=(n_rows, 3))  # baselines in metres
image = rng.standard_normal((n_l, n_m))         # real dirty image

# 1. Build the plan once for this (uvw, freq, image_shape, epsilon).
plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6)

# 2. Forward operator: dirty image -> visibilities
vis = dirty2vis(plan, jnp.asarray(image))   # shape (n_rows, 1) complex

# 3. Adjoint operator: visibilities -> dirty image
dirty_back = vis2dirty(plan, vis)           # shape (1, n_l, n_m) real
```

## Mathematical background

### The radio interferometric measurement equation

For a sky brightness `B(l, m)` on the tangent plane and a single baseline with
coordinates `(u, v, w)` in wavelengths, the standard radio interferometric
measurement equation reads

```
V(u, v, w) = integral B(l, m) / n * exp(-2 pi i (u l + v m + w (n - 1))) dl dm
```

with `n = sqrt(1 - l^2 - m^2)`. The image is on a regular tangent-plane grid

```
l_i = (i - n_l / 2) * pixsize_l    for i = 0, ..., n_l - 1
```

(and similarly for `m`). It is assumed centred on the phase centre &mdash;
off-zenith pointing is the caller's responsibility (rotated `uvw`).

#### Sign convention

`jax-nufft` follows the same sign convention as ducc's `explicit_degridder`:

```
V(u, v, w) = sum_{l, m} B(l, m) * exp(-2 pi i (u l + v m)) * exp(+2 pi i w (n - 1))
```

That is: `exp(-2 pi i (u l + v m))` for the (u, v) part and `exp(+2 pi i w (n - 1))`
for the w part. Note the **plus** sign on the w-term &mdash; ducc uses
`-w (n - 1)` inside the parenthesis rather than `+w (n - 1)`, which is
algebraically the same flip.

### The wgridder algorithm

Direct evaluation of the visibility integral is `O(n_rows * n_l * n_m)` per
channel, which is prohibitive. The **wgridder** factorises it into a stack of
2D non-uniform FFTs, one per w-plane:

1. Discretise the `w` axis into `n_w` planes with centres `w_0, ..., w_{n_w-1}`.
2. For each plane k, perform a 2D NUFFT in `(u, v)` of the image multiplied by
   the image-domain w-shift `exp(+2 pi i w_k (n - 1))` and divided by the
   image-domain kernel correction `phi_hat(scale * (n - 1))`.
3. Multiply each per-plane visibility by the w-direction gridding kernel
   `phi((w_lambda - w_k) / scale)`.
4. Sum over w-planes.

Steps 2-3 implement a discrete approximation of the continuous w-direction
convolution; step 4 is the inverse NUFFT in w. The kernel `phi` and its
Fourier transform `phi_hat` are chosen as a matched pair so that the gridding
correction in image space cancels the kernel apodisation in w-space.

### Forward operator (`dirty2vis`)

Given an image `B` of shape `(n_chan, n_l, n_m)` (or `(n_l, n_m)` broadcast
across channels), the forward operator computes `vis` of shape
`(n_rows, n_chan)`:

For each channel `c`:

  1. `uvw_lambda = uvw * freq[c] / c`
  2. `u_finufft, v_finufft = 2*pi * uvw_lambda[:, 0:2] * pixsize`
  3. For each w-plane `k`:
     - `image_k = B[c] * exp(+2 pi i w_k (n - 1)) / phi_hat_n`
     - `vis_k = NUFFT2(image_k, u_finufft, v_finufft, iflag = -1, eps = epsilon)`
     - `vis_k = vis_k * phi((w_lambda - w_k) / w_kernel_scale)`
  4. `vis[:, c] = sum over k of vis_k`

w-plane traversal has four strategies (`dense_scan` default, `dense_vmap`,
`windowed_scan`, `windowed_vmap`). The dense variants evaluate every
visibility on every w-plane and rely on the kernel zeroing out non-
contributing rows; the windowed variants take a contiguous slice of
visibilities (after sorting by `w`) per plane, cutting the spread cost
to roughly `p * n_rows * W^3` where `p ~ 1.5-3` is the window padding
overhead. See *Strategy options* below for the trade-offs. Channel
traversal independently supports `scan` (default) or `vmap`.

### Adjoint operator (`vis2dirty`)

Given visibilities `V` of shape `(n_rows, n_chan)` and an image shape
`(n_l, n_m)`, the adjoint operator computes a real-valued dirty image of
shape `(n_chan, n_l, n_m)`:

For each channel `c`:

  1. `uvw_lambda`, `u_finufft`, `v_finufft` as for the forward operator.
  2. `vis_w = vis[:, c] * weights[:, c]` if weights are provided.
  3. For each w-plane `k`:
     - `vis_k = vis_w * phi((w_lambda - w_k) / w_kernel_scale)`
     - `H_k = NUFFT1((u_finufft, v_finufft), vis_k, image_shape, iflag = +1, eps = epsilon)`
     - `I_k = H_k * exp(-2 pi i w_k (n - 1)) / phi_hat_n`
  4. `dirty[c] = (sum over k of I_k).real / n`,
     where the `1/n` factor matches ducc's `divide_by_n=True` convention.

Pixels with `n <= 0` (i.e. outside the unit disc) are returned as exactly 0.

### Kernel choice

`jax-nufft` uses the **exp-of-semicircle** kernel introduced by Barnett,
Magland & af Klinteberg in the FINUFFT paper:

```
phi(z; beta) = exp(beta * (sqrt(1 - z^2) - 1))   for |z| <= 1
             = 0                                  otherwise
```

Parameters as a function of `epsilon`:

  - kernel half-width `W = ceil(-log10(epsilon) * 2 / pi) + 2`;
  - shape parameter `beta = 2.30 * W` (matches the FINUFFT default for
    upsampling factor sigma = 2).

`phi_hat`, the continuous Fourier transform of `phi`, has no closed form. We
compute it once at planning time on a regular grid via a zero-padded FFT, and
evaluate it at arbitrary `eta` values using 4-point Lagrange (cubic)
interpolation. The resulting per-pixel correction `phi_hat_n` is bundled into
the plan and treated as a JIT-time constant.

### Plan-then-call API

The wgridder requires several quantities that depend on `(uvw, freq,
image_shape, pixsize, epsilon)` but not on the image or visibility values: the
n-1 grid, the kernel correction, the number of w-planes, the w-plane centres.
`make_plan` precomputes those once and returns a `WGridderPlan` &mdash; a frozen
dataclass registered as a JAX pytree. The actual operators are then JIT-friendly
functions of `(plan, image)` or `(plan, vis)`:

```python
plan = make_plan(uvw, freq, (n_l, n_m), pixsize_l, pixsize_m, epsilon)
vis = jax.jit(dirty2vis)(plan, image)
```

If `(uvw, freq, image_shape, ...)` change between calls, the plan must be
rebuilt; if they stay the same, the same plan can be reused for an arbitrary
number of forward and adjoint calls.

## API reference

### `make_plan(uvw, freq, image_shape, pixsize_l, pixsize_m, epsilon, *, phi_hat_n_fine=4096, phi_hat_oversample=None) -> WGridderPlan`

Build the wgridder plan. Inputs are host-side numpy / jnp arrays (planning math
runs on the host); the resulting plan holds JAX device arrays.

`phi_hat_oversample=None` (the default) picks a width-dependent oversample
suitable for the kernel chosen by `epsilon` (32 / 64 / 128 for `W <= 4`,
`<= 8`, `> 8`); pass an explicit integer to override.

The returned plan also exposes `max_window_size` and
`window_padding_overhead` for callers that want to inspect whether the
windowed strategies will be efficient on a given uvw distribution.

### `dirty2vis(plan, image, *, w_strategy="dense_scan", channel_strategy="scan", nthreads=0) -> Array`

Forward operator. `image` may be `(n_chan, n_l, n_m)` or `(n_l, n_m)`
(broadcast across channels), real or complex. Output is complex
`(n_rows, n_chan)`.

### `vis2dirty(plan, vis, *, weights=None, w_strategy="dense_scan", channel_strategy="scan", nthreads=0) -> Array`

Adjoint operator. `vis` is complex `(n_rows, n_chan)`; optional `weights` is
real `(n_rows, n_chan)`. Output is real `(n_chan, n_l, n_m)` with the `1/n`
factor applied (matching ducc's `divide_by_n=True`).

### Strategy options

`w_strategy` selects how the w-plane loop is structured. There are four
canonical choices plus an opt-in `"auto"` resolver, and the two v0.1
names are kept as deprecated aliases:

| `w_strategy`      | Per-plane work               | Peak transient memory       | Notes                                          |
|-------------------|------------------------------|-----------------------------|------------------------------------------------|
| `"dense_scan"`    | `n_rows * W^2`               | `O(image_size + n_rows)`    | default; v0.1 `"scan"` is a deprecated alias.  |
| `"dense_vmap"`    | `n_rows * W^2`               | `O(n_w * image_size)`       | v0.1 `"vmap"` is a deprecated alias.           |
| `"windowed_scan"` | `max_window_size * W^2`      | `O(image_size + n_rows)`    | v0.1.1; helps on adjoint when `n_w >> W`.      |
| `"windowed_vmap"` | `max_window_size * W^2`      | `O(n_w * image_size)`       | v0.1.1; rare wins, mostly for completeness.    |
| `"auto"`          | resolves to one of the above | matches the resolved choice | v0.1.2; opt-in. Platform-aware heuristic, not the default. |

`channel_strategy` is independently `"scan"` (default) or `"vmap"`.

For the windowed strategies, the plan exposes
`plan.window_padding_overhead = max_window_size / mean_window_size` as a
diagnostic. Pathological `w`-distributions can drive this above ~3, at
which point dense strategies usually win on absolute time.

#### `w_strategy="auto"` (v0.1.2+, opt-in)

`"auto"` resolves to one of the four canonical names before the JIT
boundary, using `plan.n_w`, `plan.w_kernel_width`,
`plan.window_padding_overhead`, and (on GPU) `plan.n_rows`, plus the
operator's adjoint flag. Resolution happens in the public wrapper, so
an `"auto"` call shares a JIT cache entry with the explicit equivalent
on the same plan. The heuristic is **platform-aware**
(`jax.devices()[0].platform`):

- **CPU.** Conservative: never picks a windowed forward (no measured
  win on the v0.1.1 algorithm), and only picks `windowed_scan` on the
  adjoint when `n_w / w_kernel_width > 2` and the windowed padding
  overhead is below 5x. Otherwise `dense_scan`.
- **GPU** (tuned on the GH200 baseline sweep). Never picks a `_scan`
  variant (5-30x slower than `_vmap` there). Picks `windowed_vmap` only
  on large-row plans (`n_rows >= 10000`) with padding overhead below 3x
  -- on the adjoint at any pointing, and on the forward when `n_w` is
  low (`n_w <= 3 * w_kernel_width`). Otherwise `dense_vmap`.
- **Other platforms** (e.g. TPU) fall back to the CPU heuristic.

```python
# Recommended for new code: let the heuristic pick per call and per
# platform.
vis   = dirty2vis(plan, image, w_strategy="auto")
dirty = vis2dirty(plan, vis,   w_strategy="auto")

# For reproducing benchmarks, pin the strategy explicitly:
vis   = dirty2vis(plan, image, w_strategy="dense_vmap")
```

`"auto"` is **not** the default in v0.1.2 (defaults stay on
`"dense_scan"`); promotion to default is deferred to a later release.
The GPU gates are validated against `docs/benchmarks/v0.1.2-baseline-gpu.json`
by `tests/test_auto_strategy_acceptance.py`, which asserts the picked
strategy is within 15% of the best measured strategy for every
(operator, telescope) cell.

### Accuracy expectation

`dirty2vis` and `vis2dirty` match `ducc0.wgridder` (with matched
`divide_by_n` flags) to within ~10x the requested `epsilon`. For tighter
`epsilon`, you may need to bump `phi_hat_oversample` to keep the
phi_hat-table interpolation error below the wgridder accuracy floor.

## Performance notes

### GPU support

`jax-nufft` itself is platform-agnostic; the heavy lifting is delegated to
`jax-finufft` which dispatches to FINUFFT on CPU and cuFINUFFT on GPU. Switch
between the two by installing the matching `jax-finufft` extra:

- `pixi run -e default ...` &mdash; CPU FINUFFT.
- `pixi run -e gpu ...` &mdash; CUDA-enabled `jax-finufft` (Linux only).

### Scaling with `n_w` and image size

Per-channel forward / adjoint cost is roughly `n_w * (image_size + n_rows)`
plus the FFT and spreading work inside FINUFFT. `n_w` scales with
`baseline_max_lambda * max|n - 1|`, which means wider FoVs and longer
baselines produce more w-planes &mdash; expected wgridder behaviour.

### Constant-w fast path (v0.1.2+)

When every `uvw_lambda[:, :, 2]` entry is identical &mdash; e.g. a perfectly
coplanar array, snapshot data at fixed pointing, or any case where
`plan.w_extent == 0` after planning &mdash; `make_plan` collapses the
w-plane loop to a single plane at the constant w-value. Expected speedup
is roughly `w_kernel_width + 1` (one NUFFT instead of `W+1`), which is
about 7&times; for the default `epsilon = 1e-6` (`W = 6`).

The user-visible signal that the specialisation engaged is
`plan.n_w == 1` (and `plan.is_constant_w == True`). All four
`w_strategy` choices reduce to the same single-plane work in this
regime, so picking one vs another has no effect on output. Both
operators stay bit-identical across strategies in this case and match
ducc within `20 * epsilon`.

### CPU benchmarks vs ducc0

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

#### Common invocations

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

#### What the benchmark numbers include

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

#### Indicative numbers (Mac M-series CPU, eps=1e-6, single-threaded)

Median wall-clock time for `dirty2vis` / `vis2dirty`, taken from a
single sweep of the benchmark suite. Rerun on your hardware before
making strategy decisions &mdash; these are CI-runner sized problems and
absolute timings vary several-fold across machines.

**Zenith pointing**

| Telescope     | dense_scan       | dense_vmap       | windowed_scan    | windowed_vmap    | ducc           |
|---------------|------------------|------------------|------------------|------------------|----------------|
| EDA2          | 2.7 / 3.0 ms     | 1.0 / 1.2 ms     | 2.8 / 4.2 ms     | 1.0 / 1.2 ms     | 0.7 / 0.9 ms   |
| MWA_compact   | 2.7 / 3.8 ms     | 1.6 / 1.8 ms     | 2.7 / 3.3 ms     | 1.9 / 1.9 ms     | 1.7 / 1.9 ms   |
| MWA_extended  | 22.0 / 33.5 ms   | 15.6 / 20.2 ms   | 21.7 / 31.6 ms   | 16.9 / 22.4 ms   | 9.9 / 10.9 ms  |
| MeerKAT       | 8.8 / 13.9 ms    | 6.3 / 8.2 ms     | 8.8 / 12.0 ms    | 6.9 / 8.3 ms     | 7.5 / 8.3 ms   |

**30-deg off-zenith pointing** (`n_w` is much larger; this is where
both improvements have the most to bite into):

| Telescope     | dense_scan        | dense_vmap        | windowed_scan       | windowed_vmap       | ducc            |
|---------------|-------------------|-------------------|---------------------|---------------------|-----------------|
| EDA2          | 33.2 / 38.3 ms    | 9.4 / 12.9 ms     | 31.4 / 32.7 ms      | 7.7 / 10.5 ms       | 1.5 / 1.9 ms    |
| MWA_compact   | 10.2 / 13.8 ms    | 5.2 / 6.4 ms      | 10.8 / 9.0 ms       | 5.9 / 6.2 ms        | 2.1 / 2.5 ms    |
| MWA_extended  | 561 / 872 ms      | 394 / 546 ms      | 566 / 731 ms        | 417 / 482 ms        | 34.7 / 38.4 ms  |
| MeerKAT       | 32.9 / 44.8 ms    | 23.2 / 30.6 ms    | 34.4 / 42.5 ms      | 25.0 / 32.2 ms      | 10.2 / 10.9 ms  |

#### Isolating Part 1 (standard n_w) vs Part 2 (windowed)

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
  FFT-count reduction: at eps=1e-6 (W=6), expected speedup is 1.5x.
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

#### Memory

`vmap` variants are still consistently 1.4-3.5x faster than `scan` because
they reduce the per-iteration FINUFFT planning / setpts cost and let XLA
fuse work across w-planes. The cost is **memory**: `vmap` materialises
the full `(n_w, n_l, n_m)` stack of corrected images at once. For
MWA-extended off-zenith at `n_pix = 256`:

| Implementation             | peak RSS delta |
|----------------------------|----------------|
| jax/dense_scan dirty2vis   | 1.6 MB         |
| jax/dense_scan vis2dirty   | 0 MB           |
| **jax/dense_vmap dirty2vis** | **776 MB**   |
| **jax/dense_vmap vis2dirty** | **1.5 GB**   |
| ducc dirty2vis             | 0 MB           |
| ducc vis2dirty             | 0 MB           |

(Windowed variants sit between the two: `windowed_scan` is comparable to
`dense_scan` plus the sort-permutation tables; `windowed_vmap` is
comparable to `dense_vmap`.)

ducc remains the fastest CPU implementation in every regime; the
jax-nufft value proposition is differentiability and the GPU port
(where the wasted spread / FFT work parallelises cheaply).

### Picking a strategy

- **`dense_scan` (default)** keeps memory bounded at
  `O(image_size + n_rows)` regardless of `n_w` and `n_chan`. Recommended
  on CPU and as a safe baseline for any problem.
- **`dense_vmap`** allocates `O(n_w * image_size)` but is usually the
  fastest of the four on CPU at the tested scales.
- **`windowed_scan`** matches `dense_scan` memory and helps on the
  adjoint when `n_w >> W` and the `w`-distribution is reasonably
  uniform (`plan.window_padding_overhead < ~3`).
- **`windowed_vmap`** is the high-memory variant of the windowed path;
  marginal wins on most cases, kept primarily for GPU parity.

## Comparison with ducc

ducc's wgridder is a hand-tuned C++ implementation that uses a single 3D bin
sort over `(u, v, w)` and is currently CPU only. It is the reference for both
correctness and speed in CPU-side wgridding.

`jax-nufft` trades some constant factor relative to ducc for full JAX
integration: composability with `jax.jit`, `jax.grad`, and `jax.vmap`, plus
GPU support via cuFINUFFT. It is intended for use within JAX-native pipelines
(differentiable inverse problems, calibration, sampler-friendly forward
models), not as a faster CPU wgridder.

## Citations

The algorithm and kernel design draw from:

- P. Arras, M. Reinecke, R. Westermann, T. A. Enssli, "Efficient wide-field
  radio interferometry response," A&A 646 A58 (2021), arXiv:2010.10122.
- H. Ye, S. F. Gull, S. M. Tan, B. Nikolic, "Optimal gridding and degridding
  in radio interferometry imaging," MNRAS 510, 4110 (2022), arXiv:2110.03914.
- A. Barnett, J. Magland, L. af Klinteberg, "FINUFFT", SIAM J. Sci. Comput.
  41, C479 (2019).

## License

Apache-2.0. See [`LICENSE`](LICENSE) for the full text.

`ducc0` is used as a test reference under GPL-2.0+ but is not a runtime
dependency of `jax-nufft`.

[jaxfinufft]: https://github.com/flatironinstitute/jax-finufft
