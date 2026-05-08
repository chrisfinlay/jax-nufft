# jax-nufft

JAX-native wgridder for radio interferometric imaging.

> **Status:** v0.1 in active development. API may change.

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

w-plane traversal can be performed via `lax.scan` (default, lower memory) or
`vmap` (potentially faster on GPU). The same applies to channel traversal.

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

### `make_plan(uvw, freq, image_shape, pixsize_l, pixsize_m, epsilon, *, phi_hat_n_fine=4096, phi_hat_oversample=32) -> WGridderPlan`

Build the wgridder plan. Inputs are host-side numpy / jnp arrays (planning math
runs on the host); the resulting plan holds JAX device arrays.

### `dirty2vis(plan, image, *, w_strategy="scan", channel_strategy="scan", nthreads=0) -> Array`

Forward operator. `image` may be `(n_chan, n_l, n_m)` or `(n_l, n_m)`
(broadcast across channels), real or complex. Output is complex
`(n_rows, n_chan)`.

### `vis2dirty(plan, vis, *, weights=None, w_strategy="scan", channel_strategy="scan", nthreads=0) -> Array`

Adjoint operator. `vis` is complex `(n_rows, n_chan)`; optional `weights` is
real `(n_rows, n_chan)`. Output is real `(n_chan, n_l, n_m)` with the `1/n`
factor applied (matching ducc's `divide_by_n=True`).

### Strategy options

Both operators expose `w_strategy` and `channel_strategy`, each of which can
be `"scan"` (default, low memory) or `"vmap"` (potentially faster on GPU but
allocates `O(n_w * image_size)` peak memory for the w-loop).

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

### CPU benchmarks vs ducc0

The repository ships an opt-in benchmark suite that times `dirty2vis` and
`vis2dirty` against `ducc0.wgridder` for each of the four built-in telescope
configs:

```sh
pixi run -e test pytest tests/test_benchmark_against_ducc.py \
    --runbench --benchmark-group-by=param -q
```

Indicative numbers from a Mac (M-series, single-threaded, eps=1e-6):

| Telescope     | jax fwd | ducc fwd | jax adj | ducc adj |
|---------------|---------|----------|---------|----------|
| EDA2          |  3.5 ms |  0.7 ms  |  3.9 ms |  0.9 ms  |
| MWA_compact   |  2.7 ms |  1.7 ms  |  3.2 ms |  1.9 ms  |
| MWA_extended  | 27.3 ms |  9.6 ms  | 34.5 ms | 10.8 ms  |
| MeerKAT       |  8.3 ms |  7.5 ms  | 10.4 ms |  8.3 ms  |

`jax-nufft` is 1.1x to ~5x slower than ducc on CPU. The gap is largest for
small problems (where JAX dispatch overhead dominates) and shrinks as the
NUFFT compute grows. The intended use case for `jax-nufft` is differentiable
or vmap-able pipelines where ducc's CPU-only, opaque-to-JAX implementation
isn't usable; if you have a pure CPU forward / adjoint workload with no
need for autodiff, ducc remains the faster choice.

### `vmap` vs `scan`

- `"scan"` keeps memory bounded at `O(image_size + n_rows)` regardless of
  `n_w` and `n_chan` &mdash; recommended on CPU and for problems where `n_w`
  is large (say > 50).
- `"vmap"` allocates `O(n_w * image_size)` (or `O(n_chan * ...)` for
  channel-vmap) but exposes parallelism that GPUs can use directly.

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
