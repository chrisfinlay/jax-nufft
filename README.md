# jax-nufft

**A JAX-native wgridder for radio interferometric imaging** — differentiable,
GPU-capable, and built on [`jax-finufft`][jaxfinufft].

[![tests](https://github.com/chrisfinlay/jax-nufft/actions/workflows/test.yml/badge.svg)](https://github.com/chrisfinlay/jax-nufft/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)](pyproject.toml)
[![jax](https://img.shields.io/badge/jax-%E2%89%A5%200.6.0-important)](https://github.com/jax-ml/jax)
[![pypi](https://img.shields.io/pypi/v/jax-nufft)](https://pypi.org/project/jax-nufft/)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

```python
plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6)
vis   = dirty2vis(plan, image)   # sky  -> visibilities
dirty = vis2dirty(plan, vis)     # visibilities -> sky
```

Both operators are ordinary JAX functions: `jit` them, `vmap` them, take
`grad` through them, compose them with the rest of a calibration or imaging
pipeline.

---

## Contents

| | |
|---|---|
| [Why this exists](#why-this-exists) | what it is for, and what it is not |
| [Installation](#installation) · [Quick start](#quick-start) | getting running |
| [GPU setup](#gpu-install-jax-finufft-from-conda-forge-first) | **read before installing** — PyPI gives you a CPU-only backend |
| [How it works](#how-it-works) | the measurement equation and w-stacking |
| [API reference](#api-reference) | `make_plan`, `dirty2vis`, `vis2dirty` |
| [Accuracy](#accuracy) · [Precision](#precision) | what `epsilon` buys, and float32 |
| [Weighting](docs/weighting.md) | Briggs, tapering, flags, and the PSF |
| [Performance](#performance) | measured against ducc0 on a GH200 |
| [Choosing a strategy](#choosing-a-strategy) | speed / memory trade-offs |
| [Further reading](#further-reading) | the deep-dive documents |

> [!NOTE]
> **Current release: 0.2.0.** The API is stable. A v0.1.2 series was developed
> and merged but never tagged, so its work ships for the first time in 0.2.0 —
> see [`CHANGELOG.md`](CHANGELOG.md).
>
> Two defaults changed since v0.1.1: `w_strategy` now defaults to `"auto"`
> (was `"dense_scan"`), and `nthreads` defaults to `None`, resolving to a
> strategy-aware thread count instead of a flat `0`. Both are described under
> [Choosing a strategy](#choosing-a-strategy); pass explicit values to opt out.

## Why this exists

`jax-nufft` implements the wgridder algorithm — wide-field radio
interferometric imaging as a stack of 2D non-uniform FFTs indexed by w-plane —
in pure JAX.

It exists for pipelines that need the wgridder to be **one differentiable
operator among many**: calibration solvers, forward-modelling, variational
inference, anything where the measurement operator sits inside a larger
gradient computation. `ducc0` is faster on CPU and leaner in memory; what it
cannot do is participate in a JAX program.

**What you get**

- **Differentiable.** Both operators are linear JAX primitives with registered
  transposes, so `jvp`, `grad`, `vjp`, `hessian` and `check_grads` all work —
  not a `custom_vjp` wrapper that breaks forward mode. Gradient memory is
  0.98–1.98× a forward call depending on strategy, down from 12.5–135× before
  v0.2.0.
- **GPU-capable**, via cuFINUFFT, with no code change — and on a GH200 it is
  [1.5–11.2× faster than ducc0](#performance) at realistic problem sizes. This
  needs a CUDA `jax-finufft` from conda-forge; PyPI's is CPU-only, so
  [read the GPU setup](#gpu-install-jax-finufft-from-conda-forge-first) first.
- **Accurate to a contract**: within `2 * epsilon` of the exact DFT over a
  stated grid, and within `3 * epsilon` of `ducc0.wgridder`. See
  [Accuracy](#accuracy) for exactly what is measured.
- **Flexible geometry**: odd and non-square images, anisotropic pixels,
  multi-channel visibilities, optional per-visibility weights. `ducc0` requires
  even extents.
- **A memory dial**: [`w_chunk`](docs/strategies.md) trades transient memory
  against time monotonically, from all w-planes resident to one at a time.

**What you should know going in**

- It is **hungrier than ducc0** — roughly 3× to 54× the memory at the sizes
  measured. That is the honest cost of the speed, and
  [the memory section](#memory) says where it bites.
- Plans are **float64 by default** and need `jax_enable_x64`. See
  [Precision](#precision).
- The default operator pair is **not an exact adjoint** — a deliberate choice
  matching ducc0's defaults. Pass `divide_by_n=True` to both when you need one;
  see [Adjointness](docs/accuracy.md#adjointness-equal-flags-only).
- **Polarisation is out of scope** for v1. The operators are Stokes-I scalar.

## Installation

```sh
pip install jax-nufft
```

Requires Python 3.11+. For the unreleased tip:

```sh
pip install git+https://github.com/chrisfinlay/jax-nufft.git
```

### GPU: install `jax-finufft` from conda-forge *first*

> [!IMPORTANT]
> **`pip install jax-nufft` gives you a CPU-only backend, on a GPU machine
> too.** `jax-nufft` is platform-agnostic and does no CUDA work itself — all
> of it happens inside [`jax-finufft`][jaxfinufft] — and the `jax-finufft`
> wheels on PyPI are built without CUDA. Installing `jax-nufft` from PyPI
> therefore pulls a CPU `jax-finufft` and everything runs on the CPU, quietly
> and correctly and slowly.
>
> The CUDA builds live on **conda-forge**. Install one **before** `jax-nufft`.

```sh
# 1. A CUDA build of jax-finufft (conda-forge; Linux x86-64 and aarch64 only)
conda install -c conda-forge "jax-finufft=*=cuda*" "cuda-version>=12.0,<13"

# 2. then jax-nufft, leaving that backend in place
pip install --no-deps jax-nufft
```

`--no-deps` is the careful form: the requirement is already satisfied by the
conda package, and it stops pip from reaching for `jax` on its own and
replacing the CUDA-matched build. If you drop it, check `jax` afterwards.

There is **no `jax-nufft[gpu]` extra** — the CPU/CUDA choice belongs entirely
to `jax-finufft`, and an extra here could not express it.

Using pixi instead, the stanza is the one this repository's own `pixi.toml`
uses for its `gpu` feature:

```toml
[feature.gpu]
platforms = ["linux-64", "linux-aarch64"]
system-requirements = { cuda = "12.0" }

[feature.gpu.dependencies]
jax-finufft = { version = ">=1.3.0", build = "cuda*" }
cuda-version = ">=12.0,<13"
```

**Check which backend you got**, because nothing will tell you otherwise:

```python
import jax
print(jax.devices())            # CudaDevice(...) means the GPU is visible to JAX
import jax_finufft, importlib.metadata as m
print(m.version("jax-finufft"))  # and `conda list jax-finufft` shows cuda* vs cpu_*
```

`jax.devices()` reporting a CUDA device is necessary but not sufficient — JAX
can see the GPU while `jax-finufft` is still the CPU build, which is exactly
the failure this section exists to prevent. The build string is the thing to
check.

For development, the repository uses [pixi](https://pixi.sh/):

```sh
git clone https://github.com/chrisfinlay/jax-nufft.git
cd jax-nufft
pixi run -e test pytest              # unit tests (~6 min)
pixi run -e test pytest --runslow    # adds MWA_extended / MeerKAT ducc0 parity
pixi run -e dev lint                 # ruff check
pixi run -e dev format               # ruff format
pixi run -e dev typecheck            # mypy
```

The `test` environment additionally installs `ducc0`, used only as a black-box
reference oracle for parity tests.

## Quick start

```python
import jax
import jax.numpy as jnp
import numpy as np

from jax_nufft import dirty2vis, make_plan, vis2dirty

# Plans are float64 by default, which needs x64. This must run before the
# first JAX array is created.
jax.config.update("jax_enable_x64", True)

n_l = n_m = 128
n_rows = 500
freq = np.array([1.4e9])
pixsize = np.deg2rad(20.0) / n_l                # 20 degree field of view
rng = np.random.default_rng(0)
uvw = rng.normal(scale=80.0, size=(n_rows, 3))  # baselines in metres
image = rng.standard_normal((n_l, n_m))         # real sky

# Build the plan once per (uvw, freq, image_shape, epsilon) ...
plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6)

# ... then call the operators as often as you like.
vis = dirty2vis(plan, jnp.asarray(image))       # (500, 1)      complex128
dirty = vis2dirty(plan, vis)                    # (1, 128, 128) float64
```

Because both operators are differentiable, the normal equations are one line:

```python
loss = lambda x: 0.5 * jnp.sum(jnp.abs(dirty2vis(plan, x) - vis) ** 2)
gradient = jax.grad(loss)(image)                # (128, 128) float64
```

## How it works

### The measurement equation

For a sky brightness $B(l, m)$ on the tangent plane and a baseline with
coordinates $(u, v, w)$ in wavelengths:

$$
V(u, v, w) \;=\; \int \frac{B(l, m)}{n} \;
  \underbrace{e^{-2\pi i\,(u\,l \;+\; v\,m)}}_{\text{the 2D NUFFT}} \;
  \underbrace{e^{+2\pi i\,w\,(n - 1)}}_{\text{the w-screen}}
  \; \mathrm{d}l \; \mathrm{d}m ,
\qquad n = \sqrt{1 - l^2 - m^2} .
$$

The $w(n-1)$ term is what makes wide-field imaging hard: it couples the sky
coordinates into the exponent in a way that is not a 2D Fourier transform.

**Sign convention.** Note the **plus** on the w-term while the $(u, v)$ part
carries a minus. Collapsed into one exponential this library computes

$$
e^{-2\pi i\,\left(u\,l \;+\; v\,m \;-\; w\,(n - 1)\right)} ,
$$

which is `ducc0`'s `explicit_degridder` convention and is what
`tests/test_against_dft.py` checks against. Texts that write
$+w(n-1)$ *inside* the parenthesis are using the opposite overall sign, and
that form is **not** what this library implements: evaluated against the
operator it disagrees at relative $L_2$ of order 1, where the form above
agrees to $1.5 \times 10^{-12}$. If you are porting an expression in from
elsewhere, check this sign first — it is the easiest thing in the whole
library to get backwards.

**The image grid** is regular on the tangent plane, centred on the phase
centre:

$$
l_i = \left(i - \left\lfloor n_l / 2 \right\rfloor\right)\,\Delta l ,
\qquad i = 0, \ldots, n_l - 1
$$

and likewise for $m$. The offset is **floor** division, and that is
load-bearing: it matters only for odd $n_l$, where $\lfloor n_l/2 \rfloor$ puts
$l = 0$ on an exact pixel and $n_l/2$ would leave no pixel at the phase centre
at all and shift every pixel by half of one. The odd cells of
`tests/test_against_dft.py` pin it; before they existed, substituting `/` for
`//` passed the entire suite. See [AGENTS.md](AGENTS.md) §2 for the full
argument. Off-zenith pointing is the caller's responsibility — rotate `uvw`
before planning.

### w-stacking

The wgridder makes the $w$ term tractable by quantising it. Sort the
visibilities by $w$, lay down $n_w$ planes across the range, and on each plane
$w_k$ evaluate an ordinary 2D NUFFT of the sky pre-multiplied by that plane's
w-screen:

$$
V(u, v, w) \;\approx\; \sum_{k=1}^{n_w} \Phi_k(w) \cdot
  \mathrm{NUFFT}_{2\mathrm{D}}\!\left[\frac{B(l, m)}{n}\, e^{2\pi i\, w_k (n - 1)}\right](u, v)
$$

where $\Phi_k$ is a compact interpolation kernel in $w$ — each visibility is
reconstructed from the $W$ planes nearest its own $w$, so the sum is sparse.
The adjoint runs the same structure backwards.

Three quantities follow from `epsilon` and the data, and between them they
determine both cost and memory:

| symbol | plan field | meaning |
|---|---|---|
| $W$ | `w_kernel_width` | kernel half-width, $W = \lceil -\log_{10}(\epsilon/10) \rceil$ — 7 at the default $\epsilon = 10^{-6}$ |
| $\beta$ | `beta` | ES-kernel shape, $\beta = 2.30\,W$ |
| $n_w$ | `n_w` | number of w-planes, $n_w = n_w^{\text{inner}} + W$, growing with field of view and baseline length |

Per-channel cost is roughly $n_w \times (\text{image size} + n_{\text{rows}})$
plus the FFT and spreading work inside FINUFFT. `n_w` is the number to watch:
wider fields and longer baselines make more planes, and memory in the
all-planes-resident strategies scales with it directly.

The full derivation — including the two centrings that keep this numerically
well behaved, and the kernel — is in
[docs/algorithm.md](docs/algorithm.md).

Two savings are applied automatically. **Hermitian folding** (`hermitian=True`)
uses $V(-u,-v,-w) = \overline{V(u,v,w)}$ for a real sky to store every row at
$w \ge 0$, roughly halving the w-extent — on this repository's MWA_extended
off-zenith fixture it takes $n_w$ from 251 to 134. The **constant-w fast path**
collapses the plane loop entirely when every visibility shares one $w$
(`plan.is_constant_w`), worth about $W + 1$ NUFFTs.

### Plan, then call

Planning is host-side numpy: sorting by $w$, choosing $n_w$, building the
kernel tables and the per-plane window bounds. None of it depends on the image
or the visibilities, so it happens once:

```python
plan = make_plan(uvw, freq, image_shape, pixsize_l, pixsize_m, epsilon)
for _ in range(n_iterations):
    vis = dirty2vis(plan, image)     # plan reused; no replanning
```

`WGridderPlan` is a registered JAX pytree, so it crosses `jit` boundaries and
its array leaves live on device. Static metadata (`n_w`, `w_kernel_width`,
shapes) is in the treedef and so participates in the JIT cache key. Rebuild the
plan when `uvw`, `freq`, `image_shape`, `pixsize` or `epsilon` change; never
inside a training loop.

## API reference

The public surface is four names: `make_plan`, `dirty2vis`, `vis2dirty` and
the `WGridderPlan` type.

### `make_plan`

```python
make_plan(
    uvw, freq, image_shape, pixsize_l, pixsize_m, epsilon,
    *,
    dtype=jnp.float64,
    hermitian=True,
    phi_hat_n_fine=4096,
    phi_hat_oversample=None,
) -> WGridderPlan
```

| parameter | type | default | description |
|---|---|---|---|
| `uvw` | `(n_rows, 3)` array | — | baseline coordinates in **metres** |
| `freq` | `(n_chan,)` array | — | channel frequencies in Hz |
| `image_shape` | `(int, int)` | — | `(n_l, n_m)`; **odd and non-square are both allowed** |
| `pixsize_l`, `pixsize_m` | `float` | — | pixel scale in radians; may differ from each other |
| `epsilon` | `float` | — | requested relative accuracy; `1e-14` is the floor, below which `make_plan` raises. Looser than `1e-3` is accepted but outside the measured [accuracy grid](#accuracy) |
| `dtype` | dtype | `jnp.float64` | precision of the whole plan; see [Precision](#precision) |
| `hermitian` | `bool` | `True` | fold conjugate symmetry — halves the w-extent, real sky only |
| `phi_hat_n_fine` | `int` | `4096` | kernel-table resolution |
| `phi_hat_oversample` | `int \| None` | `None` | table oversample; `None` picks from `W` |

Inputs are host-side arrays — planning maths runs in numpy — and the returned
plan holds JAX device arrays.

> [!IMPORTANT]
> `hermitian=True` is valid for a **real sky only**. `dirty2vis` raises
> `ValueError` on a complex image from a folded plan, naming `hermitian=False`
> as the fix. The check is on dtype, so a complex array with zero imaginary
> part is refused too — pass `image.real`. `vis2dirty` is unrestricted: its
> output is a real part, so the fold's saving applies unconditionally.

**Useful plan attributes**

| attribute | meaning |
|---|---|
| `n_w`, `w_kernel_width`, `beta` | plane count and kernel parameters |
| `is_constant_w` | the single-plane fast path engaged |
| `image_shape`, `n_l`, `n_m`, `n_chan`, `n_rows` | problem shape |
| `real_dtype`, `complex_dtype` | dtypes the operators accept and return |
| `max_window_size`, `window_buckets` | row slices the windowed strategies take |
| `window_padding_overhead`, `..._adjoint` | how much of those slices is padding |
| `live_row_count`, `empty_plane_count` | diagnostics for whether windowing will pay |

### `dirty2vis` — sky to visibilities

```python
dirty2vis(
    plan, image,
    *,
    divide_by_n=False,
    w_strategy="auto",
    channel_strategy="scan",
    nthreads=None,
    w_chunk=32,
) -> Array
```

Takes `image` of shape `(n_l, n_m)` — real, unless the plan is unfolded —
and returns visibilities of shape `(n_rows, n_chan)`, complex.

### `vis2dirty` — visibilities to sky

```python
vis2dirty(
    plan, vis,
    *,
    divide_by_n=True,
    weights=None,
    w_strategy="auto",
    channel_strategy="scan",
    nthreads=None,
    w_chunk=32,
) -> Array
```

Takes `vis` of shape `(n_rows, n_chan)` complex and returns `(n_chan, n_l,
n_m)` real. `weights`, if given, is `(n_rows, n_chan)` real and is applied
before gridding — see [weighting](docs/weighting.md) for Briggs, tapering,
flags and the PSF.

### Shared keyword arguments

| argument | default | what it does |
|---|---|---|
| `divide_by_n` | `False` fwd, `True` adj | apply the $1/n$ factor. **The defaults differ**, matching ducc0, which means the shipped pair is *not* an exact adjoint — see [Adjointness](docs/accuracy.md#adjointness-equal-flags-only) |
| `w_strategy` | `"auto"` | how the w-plane loop is structured; `"auto"` resolves per call and per platform. See [Choosing a strategy](#choosing-a-strategy) |
| `channel_strategy` | `"scan"` | `"scan"` (bounded memory) or `"vmap"` (faster, memory grows with `n_chan`) |
| `nthreads` | `None` | `None` resolves to a strategy-aware count before the JIT boundary; pass an `int` (including `0`, meaning "let FINUFFT decide") to override |
| `w_chunk` | `32` | w-planes resident at once for the `chunked` strategies; the [memory dial](#memory) |

> [!WARNING]
> The `divide_by_n` defaults are asymmetric. If you are differentiating
> through the operator pair, or need $A^H A$ to be self-adjoint, pass
> `divide_by_n=True` to **both**. With the shipped defaults the dot-product
> residual on a wide field is `0.63`; with matched flags it is `1.3e-15`.

## Accuracy

Both operators land within **`2 * epsilon`** of the exact DFT — relative $L_2$,
in the sign convention above — over the grid described below. The worst cell
measured is `1.48 * epsilon`. Against `ducc0.wgridder` with matched
`divide_by_n` flags the bound is **`3 * epsilon`**, the sum of the two
implementations' error budgets.

This is a measurement over a finite grid, not a proof, so what the grid holds
constant is part of the claim:

| axis | covered |
|---|---|
| `epsilon` (vs the DFT) | eight points, `1e-3` … `1e-12` (`1e-9`, `1e-11` skipped) |
| `epsilon` (vs ducc0) | **`1e-4` and `1e-6` only** |
| fixtures | seven telescope/pointing combinations |
| `w_strategy` | **`dense_scan` only** — *not* the shipped default |
| everything else | one channel, `"scan"`, shipped `divide_by_n`, no weights, float64 |

The `w_strategy` row is the one that bites. A caller taking the default gets
`"auto"`, which resolves to `windowed_scan` for the adjoint on several
off-zenith fixtures — a strategy the sweep never measures. Cross-strategy
agreement is pinned separately at `1e-11`, which is wider than the contract
itself below `epsilon = 5e-12`. **So `2 * epsilon` is established for
`dense_scan`, and carries to the other strategies only down to about
`1e-10`.** If you need the tight bound at `1e-12`, pin
`w_strategy="dense_scan"`.

The sweep runs in CI. See [docs/accuracy.md](docs/accuracy.md) for the full
table and the reasoning.

## Precision

Plans are **float64** by default and require x64, enabled before the first JAX
array exists:

```python
jax.config.update("jax_enable_x64", True)   # or JAX_ENABLE_X64=1
```

Single precision is opt-in and reaches about `1e-4`:

```python
plan = make_plan(..., epsilon=1e-4, dtype=jnp.float32)
```

`make_plan` warns if you ask a float32 plan for an `epsilon` below `1e-5`,
which it cannot deliver. What float32 buys:

- **Operator scratch halves**, near-exactly — a ratio of 1.99987 to 1.99999
  across three size classes, two strategies and both operators.
- **The plan halves only when image-dominated** — 0.501× at 256² with 600 rows,
  but 0.586× at 150² with 4.9M rows, because `sort_perm` is `int32` and
  `flip_sign` is `int8` and neither follows the floating dtype.

## Performance

Measured on one GH200 node: jax-nufft on the H100, ducc0 on the same node's 72
Grace cores, at $\epsilon = 10^{-6}$ in float64. **jax-nufft is faster in all
sixteen (problem, operator) comparisons**, against ducc0 at the best thread
count measured for each — not at a fixed setting.

| | span | median |
|---|---|---|
| `dirty2vis` (forward) | 1.7× – 3.8× | 2.3× |
| `vis2dirty` (adjoint) | 1.5× – 11.2× | 2.6× |

The adjoint's span is two clusters rather than a continuum: five of the eight
problems sit between 1.5× and 2.9×, three between 8.8× and 11.2×, and nothing
lands in between — so "about 6×" describes no case that was measured.

| Telescope / pointing | n_pix | n_rows | `n_w` | `dirty2vis` | `vis2dirty` |
|----------------------|-------|-----------|------|-------------|-------------|
| EDA2 zenith          | 150   | 4,896,000 | 13   | 2.3×  | 1.9×  |
| EDA2 off30           | 150   | 4,896,000 | 60   | 3.5×  | 1.5×  |
| MWA_compact zenith   | 144   | 1,219,200 | 8    | 2.1×  | 2.9×  |
| MWA_compact off30    | 144   | 1,219,200 | 13   | 2.0×  | 2.2×  |
| MWA_extended zenith  | 3600  | 1,219,200 | 13   | 3.8×  | 11.2× |
| MWA_extended off30   | 3600  | 1,219,200 | 140  | 1.7×  | 2.1×  |
| MeerKAT zenith       | 2700  | 302,400   | 8    | 3.6×  | 10.7× |
| MeerKAT off30        | 2700  | 302,400   | 14   | 2.2×  | 8.8×  |

**Give ducc0 its best thread count.** All 288 hardware threads was *never*
ducc0's fastest setting in any of the sixteen — it ran 1.86× to 6.00× slower
than that comparison's best measured count, which was 64 in thirteen of them.
Benchmarking ducc0 at `os.cpu_count()` would have flattered jax-nufft by up to
a further 6×.

**How the problems are sized.** The built-in test fixtures are CI-sized — 400
to 600 rows at 64 to 256 pixels — which under-resolves the field of view and
measures overhead rather than the algorithm. These sizes come from instrument
parameters instead: $\Delta\theta = \lambda / (3 B_{\max})$, `n_pix` the next
even 5-smooth integer covering the field of view, and
$n_{\text{rows}} = 150 \cdot n_{\text{ant}}(n_{\text{ant}} - 1)/2$.
`tests/test_benchmark_claims.py` recomputes all eight from that rule.

Every figure above is recomputed from
[`docs/benchmarks/v0.2.0-vs-ducc0-gh200.json`](docs/benchmarks/v0.2.0-vs-ducc0-gh200.json)
by the test suite, so prose and data cannot drift apart. ducc0 is used only as
a black-box oracle through its public Python API.

### Memory

Speed is not the whole trade. On the same node and problems, **jax-nufft needs
more memory than ducc0 in every one of the sixteen cells**, by roughly 3× to
54×. Over the ten cells the instrument resolves cleanly the range is 2.9× to
54.4×, median 8.8×.

| Telescope / pointing | n_pix | `dirty2vis` | `vis2dirty` |
|----------------------|-------|-------------|-------------|
| MWA_compact zenith   | 144   | 5.9×†  | 3.0×†  |
| MWA_compact off30    | 144   | 9.1×†  | 2.8×†  |
| EDA2 zenith          | 150   | 8.9×   | 3.0×   |
| EDA2 off30           | 150   | 13.6×  | 2.9×   |
| MeerKAT zenith       | 2700  | 13.0×† | 6.2×   |
| MeerKAT off30        | 2700  | 19.3×† | 8.7×   |
| MWA_extended zenith  | 3600  | 17.6×  | 7.9×   |
| MWA_extended off30   | 3600  | **54.4×** | 37.1× |

† ducc0's own figure here is small enough to sit inside the measurement's
noise; read these as "several times more", not as the number printed. The ratio
is jax-nufft's peak device HBM over ducc0's peak RSS less the interpreter, and
the ducc0 side is the weaker instrument: RSS moves in ~36 MB steps and its
baseline varied 36–144 MB across 48 runs of an identical program, so each ducc0
figure is the median of three. What is *not* in doubt is the direction.

**Two levers, if it does not fit.** `w_chunk` sets how many w-planes are
resident at once, and trades memory against time monotonically — on the worst
cell above, 3600² with `n_w = 140`:

| `w_strategy` | scratch | vs `dense_vmap` | time |
|---|---|---|---|
| `dense_vmap` (what `auto` picks) | 29.6 GB | — | 1.00× |
| `chunked`, `w_chunk=32` | 6.1 GB | 4.8× less | 1.27× |
| `chunked`, `w_chunk=8` | 1.9 GB | 15.7× less | 1.73× |
| `dense_scan` | 0.40 GB | 73× less | 2.35× |

Less scratch costs more time with no inversion here, which is what makes the
table advice rather than trivia: pick the row that fits your card. That
monotonicity holds on seven of the eight (fixture, operator) pairs measured —
the exception is a 2.9% inversion at `n_w = 13`, where a chunk of 8 barely
engages the dial at all.

`w_chunk=16` takes that problem from *needs a 96 GB GH200* to *fits on a 16 GB
card* for 35% more time. And `dtype=jnp.float32` halves whatever remains. The
two compose: float32 with `w_chunk=16` puts the 30 GB cell under 2 GB.

## Choosing a strategy

`w_strategy="auto"` is the default and picks per call and per platform. Pass an
explicit value to pin it — an explicit choice always overrides the heuristic.

| strategy | transient memory | when |
|---|---|---|
| `dense_scan` | `O(image + n_rows)` | safe everywhere; the pre-0.2.0 default |
| `dense_vmap` | `O(n_w · image)` | fastest on GPU; needs the memory |
| `windowed_scan` | `O(image + n_rows)` | adjoint when `n_w >> W` |
| `windowed_vmap` | `O(n_w · image)` | large-row GPU adjoints |
| `chunked` | `O(w_chunk · image)` | **the dial** — anywhere between the two |
| `windowed_chunked` | `O(w_chunk · image)` | the windowed half of the dial |

Rules of thumb: on **GPU** never pick a scan variant — the vmap family is
faster in all 160 measured pairs, by 1.45× to 32.7× (median 6.1×). On **CPU**
`dense_scan` is the safe default and the heuristic only departs from it for the
adjoint when `n_w / W > 2`. If memory is the binding constraint, reach for
`chunked` before giving up on the GPU.

Full decision table, the heuristic's exact conditions, `nthreads` resolution
and the `w_chunk` curve: **[docs/strategies.md](docs/strategies.md)**.

## Comparison with ducc0

[`ducc0.wgridder`](https://gitlab.mpcdf.mpg.de/mtr/ducc) is the reference
implementation and the oracle this library is tested against.

| | ducc0 | jax-nufft |
|---|---|---|
| CPU speed | faster | — |
| CPU memory | much leaner | — |
| GPU | not supported | 1.5–11.2× faster than tuned ducc0 on CPU |
| Differentiable | no | yes, forward and reverse |
| Composable in JAX | no | yes — `jit`, `vmap`, `grad`, `scan` |
| Odd / non-square images | rejects odd extents | supported |

Use ducc0 for CPU imaging. Use `jax-nufft` when the wgridder has to live inside
a JAX program, or when there is a GPU to use.

> [!NOTE]
> ducc0 is GPL-2.0-or-later. It is used in this project **only** as a black-box
> test oracle through its public Python API; no ducc0 source is consulted, and
> nothing under `src/` imports it. The algorithm is implemented from the cited
> papers.

## Further reading

| document | contents |
|---|---|
| [docs/algorithm.md](docs/algorithm.md) | the factorisation, `nshift` and `w` centring, the kernel, and each operator step by step |
| [docs/weighting.md](docs/weighting.md) | natural, Briggs, tapering and flags — and why imaging weights are not likelihood weights |
| [docs/strategies.md](docs/strategies.md) | every `w_strategy`, the `auto` heuristic, `w_chunk`, `nthreads` |
| [docs/accuracy.md](docs/accuracy.md) | the accuracy grid, precision, adjointness and `divide_by_n` |
| [docs/benchmarking.md](docs/benchmarking.md) | running the benchmark suite, historical CPU comparisons |
| [docs/benchmarks/README.md](docs/benchmarks/README.md) | schemas of the committed benchmark JSON |
| [CHANGELOG.md](CHANGELOG.md) | what changed in each release |
| [AGENTS.md](AGENTS.md) | repository guide: conventions, history, invariants |

## Citations

The algorithm and its kernel come from:

- **[Arras+2021]** P. Arras, M. Reinecke, R. Westermann, T. A. Enßlin,
  *Efficient wide-field radio interferometry response*, A&A 646, A58 (2021),
  [arXiv:2010.10122](https://arxiv.org/abs/2010.10122) — w-stacking, plane
  placement, the modified ES kernel, and the cost model.
- **[Barnett+2019]** A. H. Barnett, J. Magland, L. af Klinteberg, *A parallel
  nonuniform fast Fourier transform library based on an "exponential of
  semicircle" kernel*, SIAM J. Sci. Comput. 41(5), C479 (2019),
  [arXiv:1808.06736](https://arxiv.org/abs/1808.06736) — the ES kernel,
  $\beta = 2.30\,W$, and the error decay behind the width rule.
- **[Ye+2022]** H. Ye, S. F. Gull, S. M. Tan, B. Nikolic, *Optimal gridding and
  degridding in radio interferometry imaging*, MNRAS 510, 4110 (2022),
  [arXiv:2110.03914](https://arxiv.org/abs/2110.03914) — kernel optimisation.
- **[TMS2017]** A. R. Thompson, J. M. Moran, G. W. Swenson, *Interferometry and
  Synthesis in Radio Astronomy*, 3rd ed., Springer (2017), Ch. 3 — conjugate
  symmetry and the shift theorem.

## Contributing

Issues and pull requests are welcome. `AGENTS.md` documents the repository's
conventions; the short version is that a quantitative claim in prose needs a
test or a committed measurement behind it, and a tolerance is never widened to
make a test pass.

## License

Apache-2.0 — see [LICENSE](LICENSE).

[jaxfinufft]: https://github.com/flatironinstitute/jax-finufft
