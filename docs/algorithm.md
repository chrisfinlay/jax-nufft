# The algorithm, and what the operators actually compute

Reference material for the wgridder as implemented here: the factorisation, the
two centrings that make it numerically well behaved, the kernel, and a
step-by-step of each operator. [README.md](../README.md) has the summary; this
is the level of detail needed to modify the implementation or to check it.

<!--
Moved out of README.md by the 0.2.0 documentation pass, unchanged. Every figure
in it was verified against the code during issue #33.
-->

## The `1/n` factor (`divide_by_n`, issue #20)

The measurement equation above carries a `B(l, m) / n` weighting that the
sign-convention formula omits. Both operators take a keyword-only, static
`divide_by_n` flag deciding whether to apply it, with ducc0's meaning and
ducc0's names:

| | `divide_by_n=False` | `divide_by_n=True` |
|---|---|---|
| `dirty2vis` (forward) | **default.** No `1/n`; the `exp(+2 pi i w (n - 1))` phase is evaluated on the analytic extension `n - 1 = -sqrt(l^2 + m^2 - 1) - 1` at pixels outside the unit disc too | multiplies the image by `1/n` inside the disc and by **zero** outside, so the visibilities become insensitive to every pixel outside it |
| `vis2dirty` (adjoint) | returns the analytic-extension result with no division and no mask, so outside-disc pixels carry their (large) values | **default.** Divides by `n` inside the disc and returns exactly `0` outside |

The defaults are the mixed pair `dirty2vis(divide_by_n=False)` /
`vis2dirty(divide_by_n=True)`. That is ducc0's default pairing and reproduces
every release of this library through v0.1.2 &mdash; but it is **not an
adjoint pair on a wide field**. See *Adjoint operator* below for the numbers
and the recommendation.

`n = sqrt(1 - l^2 - m^2)` here is always the *physical* `n` on the unshifted
grid, never the `nshift`-centred one the w-phase is evaluated on (issue #16).

**One deliberate deviation from ducc0, on the unit circle itself.** The disc
mask is `n > 0` strictly, so a pixel with `n == 0` exactly is excluded and
returned as `0`; ducc0 divides there and returns `inf`. `1/n` is undefined on
the circle, so neither is more correct as arithmetic, but a finite `0` is what
keeps values *and gradients* finite, and this library is built to be
differentiated. Everywhere else the two agree: on a grid deliberately built
with two such pixels (`n_pix · pixsize == 2`, dyadic pixsize — a 2-radian field
on a power-of-two grid), `vis2dirty(divide_by_n=True)` matches ducc0 to
3.39e-07 relative over the other 4094 pixels. A whole-image parity check at
that geometry returns `nan`, which is ducc0's `inf`, not a disagreement about
the other pixels. No such pixel arises unless the grid is constructed to have
one; the repository's other fixtures reach `min|n|` of 0.026 to 0.95.

## The wgridder algorithm

Direct evaluation of the visibility integral is `O(n_rows * n_l * n_m)` per
channel, which is prohibitive. The **wgridder** factorises it into a stack of
2D non-uniform FFTs, one per w-plane:

1. Discretise the `w` axis into `n_w` planes with centres `w_0, ..., w_{n_w-1}`.
2. Multiply the image by the `w0` phase screen `exp(+2 pi i w0 (n - 1))`,
   where `w0` is the midpoint of the w range.
3. For each plane k, perform a 2D NUFFT in `(u, v)` of the image multiplied by
   the image-domain w-shift `exp(+2 pi i d_k (n - 1 + nshift))` and divided by
   the image-domain kernel correction `phi_hat(scale * (n - 1 + nshift))`,
   where `d_k = w_k - w0`.
4. Multiply each per-plane visibility by the w-direction gridding kernel
   `phi((d_lambda - d_k) / scale)`, `d_lambda = w_lambda - w0`.
5. Sum over w-planes.
6. Multiply each visibility by the compensating phase
   `exp(-2 pi i d_lambda * nshift)`.

Steps 3-4 implement a discrete approximation of the continuous w-direction
convolution; step 5 is the inverse NUFFT in w. The kernel `phi` and its
Fourier transform `phi_hat` are chosen as a matched pair so that the gridding
correction in image space cancels the kernel apodisation in w-space.

Steps 3 and 6 together are the **`nshift` centring** (ducc's `allow_nshift`).
The w-phase obeys the exact identity

```
exp(2 pi i w (n - 1)) = exp(2 pi i w (n - 1 + s)) * exp(-2 pi i w s)
```

for any constant `s`, and only `max|n - 1|` enters the plane spacing (below).
Taking `s = nshift = -(max(n-1) + min(n-1)) / 2` centres the shifted range on
zero, halving `max|n - 1 + s|` for any image containing the phase centre. The
plane spacing therefore doubles and `n_w_inner` halves &mdash; on the review
fixtures, MWA_extended at 30 degrees off-zenith goes from 495 to 251 planes at
`epsilon = 1e-6`, and the Hermitian fold below takes the same fixture from 251
to 134 &mdash; while the compensating factor in step 6 costs a single
complex multiply per visibility per call, independent of the plane count. The
adjoint applies the conjugate factor to its *input* visibilities instead. Note
the adjoint's `1/n` output factor still uses the physical, *unshifted*
`n = sqrt(1 - l^2 - m^2)`.

Step 2 is the matching **`w` centring**, and it is what keeps the two phases in
step 3 and step 6 from cancelling catastrophically. Both are proportional to
`w`, so with an absolute `w` they grow without bound while their sum stays
small — at the phase centre (`n - 1 = 0`) they cancel *exactly*, so the result
is a small number reconstructed from two large ones and only the rounding error
survives. Splitting `w = w0 + d` the same way, via
`exp(2 pi i w (n-1)) = exp(2 pi i w0 (n-1)) * exp(2 pi i d (n-1))`, leaves the
loop working in `d`, bounded by the w-extent, while the constant `w0` part
becomes a per-pixel image-domain screen applied once per call. The screen's own
argument *is* large, so it is evaluated at plan time on the host with exact
range reduction (`w0 * (n-1)` reduced mod 1 in double-double arithmetic). The
adjoint multiplies its output image by the screen's conjugate. Measured on a
delta image at the phase centre with `w_lambda = 1e6 + [-10, 10]` and
`epsilon = 1e-12`, this takes the relative L2 error from 1335x epsilon to
0.57x, and makes it independent of the absolute `w`.

## Forward operator (`dirty2vis`)

Given an image `B` of shape `(n_chan, n_l, n_m)` (or `(n_l, n_m)` broadcast
across channels), the forward operator computes `vis` of shape
`(n_rows, n_chan)`:

For each channel `c`:

  1. `w_lambda = inv_lambda[c] * uvw_m[:, 2]`, where `uvw_m` is the plan's
     baselines in metres and `inv_lambda[c] = freq[c] / c`.
  2. `u_ft, v_ft = (2*pi * pixsize * inv_lambda[c]) * uvw_m[:, 0:2]`
  3. `B_c = B[c] * w0_screen`, `d_lambda = w_lambda - w0`.
  4. For each w-plane `k` (with `d_k = w_centers_rel[k]`):
     - `image_k = B_c * exp(+2 pi i d_k (n - 1 + nshift)) / phi_hat_n`
     - `vis_k = NUFFT2(image_k, u_ft, v_ft, iflag = -1, eps = max(epsilon / 10, 1e-14))`
     - `vis_k = vis_k * phi((d_lambda - d_k) / w_kernel_scale)`
  5. `vis[:, c] = (sum over k of vis_k) * exp(-2 pi i d_lambda * nshift)`

The `(u, v)` NUFFT is asked for `max(epsilon / 10, 1e-14)`, not `epsilon`
itself (`_nufft_epsilon` in `wgridder.py`): the caller's budget is shared
between the w-kernel and the `(u, v)` NUFFT, and FINUFFT's `eps` is a target
rather than a bound, so the NUFFT gets one extra digit of headroom (issue
#9). The same call appears in the adjoint operator below.

w-plane traversal has six strategies (`dense_scan`, `dense_vmap`,
`windowed_scan`, `windowed_vmap`, and `chunked` / `windowed_chunked`, which
take a chunk size and generalise the first four), one of the first four of
which the default `w_strategy="auto"` picks per plan and platform. The dense
variants evaluate every
visibility on every w-plane and rely on the kernel zeroing out non-
contributing rows; the windowed variants take a contiguous slice of
visibilities (after sorting by `w`) per plane, cutting the spread cost
to roughly `p * n_rows * W^3` where `p` is the window padding overhead —
1.08-5.08 over the forty-cell calibration grid for the forward, and 1.00-1.41 for the
adjoint, which 0.2.0 (#26) bucketed. See *Strategy options* below for the
trade-offs. Channel traversal independently supports `scan` (default) or
`vmap`.

## Adjoint operator (`vis2dirty`)

Given visibilities `V` of shape `(n_rows, n_chan)` and an image shape
`(n_l, n_m)`, the adjoint operator computes a real-valued dirty image of
shape `(n_chan, n_l, n_m)`:

For each channel `c`:

  1. `w_lambda`, `u_ft`, `v_ft` as for the forward operator.
  2. `vis_w = vis[:, c] * weights[:, c]` if weights are provided, then
     `vis_w = vis_w * exp(+2 pi i d_lambda * nshift)` &mdash; the conjugate of
     the forward's step 5, applied to the input rather than the output.
  3. For each w-plane `k` (with `d_k = w_centers_rel[k]`):
     - `vis_k = vis_w * phi((d_lambda - d_k) / w_kernel_scale)`
     - `H_k = NUFFT1((u_ft, v_ft), vis_k, image_shape, iflag = +1, eps = max(epsilon / 10, 1e-14))`
     - `I_k = H_k * exp(-2 pi i d_k (n - 1 + nshift)) / phi_hat_n`
  4. `dirty[c] = ((sum over k of I_k) * conj(w0_screen)).real / n`,
     where the `1/n` factor matches ducc's `divide_by_n=True` convention. The
     conjugate screen is the adjoint of the forward's step 3 and must be
     applied before taking the real part.

With `divide_by_n=True` (the default), pixels with `n <= 0` (i.e. outside the
unit disc) are returned as exactly 0. With `divide_by_n=False` step 4 becomes
`dirty[c] = ((sum over k of I_k) * conj(w0_screen)).real` &mdash; no division
and no mask, so those pixels carry the analytic-extension values instead. The
conjugate screen is applied on **both** settings; it is the adjoint of the
forward's `w0` screen, which `divide_by_n` does not touch.

## Kernel choice

`jax-nufft` uses the **exp-of-semicircle** kernel introduced by Barnett,
Magland & af Klinteberg in the FINUFFT paper:

```
phi(z; beta) = exp(beta * (sqrt(1 - z^2) - 1))   for |z| <= 1
             = 0                                  otherwise
```

Parameters as a function of `epsilon`:

  - kernel half-width `W = ceil(-log10(epsilon / 10))`, i.e. one cell per
    requested digit plus one. This is the practical rule of eq. 10 in the
    FINUFFT paper &mdash; "`W` is one more than the desired number of digits"
    &mdash; and is what FINUFFT itself implements in
    `src/spreadinterp.cpp::setup_spreader` at upsampling factor sigma = 2. The
    paper's Theorem 7 is what makes it one *per digit*: the kernel's aliasing
    error decays like `exp(-pi * W * gamma * sqrt(1 - 1/sigma))`, about
    `exp(-2.2 * W)` at sigma = 2. `epsilon` below `1e-14` is rejected (the rule
    would ask for `W > 15`).
  - shape parameter `beta = 2.30 * W` (the same equation's value for
    upsampling factor sigma = 2).

The plane count follows the width directly: `n_w = n_w_inner + W`, so one extra
plane per unit of `W`; `n_w_inner = ceil(w_extent * max|n - 1 + nshift| / x0)`
with `x0 = 0.25`, i.e. the `nshift` centring above halves it. The (u,v) NUFFT is asked for one digit more than the
caller's `epsilon`, so that the error the caller sees is dominated by the
w-kernel rather than by FINUFFT's own tolerance (which is a target, not a
bound, and runs 1-3x above `epsilon` on small transforms).

`phi_hat`, the continuous Fourier transform of `phi`, has no closed form. We
compute it once at planning time on a regular grid via a zero-padded FFT, and
evaluate it at arbitrary `eta` values using 4-point Lagrange (cubic)
interpolation. The resulting per-pixel correction `phi_hat_n` is bundled into
the plan and treated as a JIT-time constant. Since `phi_hat_n` is *divided*
into the image, the table's interpolation error lands one-for-one in the
output, so the grid is refined as `W` grows (see `phi_hat_oversample_for_w`).

## Plan-then-call API

The wgridder requires several quantities that depend on `(uvw, freq,
image_shape, pixsize, epsilon)` but not on the image or visibility values: the
`nshift`-centred n-1 grid, the kernel correction, the number of w-planes, the
w-plane centres relative to the w-range midpoint, and the `w0` phase screen.
The unshifted n-1 grid and the absolute plane centres are *derived* from those
on access rather than stored &mdash; see **Plan memory** below.
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

## Scaling with `n_w` and image size

Per-channel forward / adjoint cost is roughly `n_w * (image_size + n_rows)`
plus the FFT and spreading work inside FINUFFT. `n_w` scales with
`baseline_max_lambda * max|n - 1|`, which means wider FoVs and longer
baselines produce more w-planes &mdash; expected wgridder behaviour.

## Constant-w fast path (v0.1.2+)

When every visibility shares the same `w` in wavelengths &mdash; e.g. a perfectly
coplanar array, snapshot data at fixed pointing, or any case where
`plan.w_extent == 0` after planning &mdash; `make_plan` collapses the
w-plane loop to a single plane at the constant w-value. The expected
speedup is roughly `w_kernel_width + 1` (one NUFFT instead of `W+1`),
about 8&times; for the default `epsilon = 1e-6` (`W = 7`). That is a
count of NUFFTs, not a measurement: no timing for this path is committed
to the repository.

The user-visible signal that the specialisation engaged is
`plan.n_w == 1` (and `plan.is_constant_w == True`). All six
`w_strategy` choices reduce to the same single-plane work in this
regime, so the arithmetic each performs is the same. That makes the
outputs bit-identical **only if `nthreads` is pinned**: with the default
`nthreads=None` the strategies resolve to different thread counts, and
FINUFFT's reduction order changes with them — on a constant-`w` plan
above the large-row cutoff, `chunked(32)` and `dense_vmap` differ by a
relative `1.8e-13` for exactly this reason (see
[`nthreads`](strategies.md#nthreads-issue-24-r11d4)). Pass an explicit `nthreads` for
the identity to hold unconditionally. Either way both operators match
ducc within `3 * epsilon` (issue #9 tightened this from `20 * epsilon`,
the same DFT-width-rule fix behind the headline accuracy contract
above).

## GPU support

`jax-nufft` itself is platform-agnostic; the heavy lifting is delegated to
`jax-finufft` which dispatches to FINUFFT on CPU and cuFINUFFT on GPU. Switch
between the two by installing the matching `jax-finufft` extra:

- `pixi run -e default ...` &mdash; CPU FINUFFT.
- `pixi run -e gpu ...` &mdash; CUDA-enabled `jax-finufft` (Linux only).

## Full `make_plan` semantics

The prose that accompanied the `make_plan` signature before the README was
condensed: geometry rules, the Hermitian fold, the kernel table, and every
diagnostic the plan exposes.


Build the wgridder plan. Inputs are host-side numpy / jnp arrays (planning math
runs on the host); the resulting plan holds JAX device arrays.

`image_shape` is `(n_l, n_m)`, and **neither extent has to be even and the two
need not be equal**: `(33, 33)`, `(32, 48)` and `(31, 45)` are all valid, as
are independent `pixsize_l` and `pixsize_m`. This is a difference from ducc0,
whose `wgridder` asserts `nx_dirty must be even` and refuses an odd `npix_x` /
`npix_y` outright (it does accept non-square even shapes and anisotropic pixel
sizes). Odd and non-square grids are gated against the exact DFT in
`tests/test_against_dft.py`, which is the only available oracle for the odd
cases.

`dtype` fixes the precision of the whole plan — `uvw` and `freq` are cast to
it, and the operators accept and return the matching real / complex dtypes.
See [Precision](../README.md#precision) below.

`hermitian=True` (the default) applies the conjugate-symmetry fold. For a real
sky `V(-u, -v, -w) = conj(V(u, v, w))`, so every row with `w < 0` is stored at
`(-u, -v, -w)` and its sign recorded in `plan.flip_sign`; the plan's w-range
becomes `[0, max|w|]` instead of `[min w, max w]`, which halves the w-extent
and with it the inner w-plane count on any roughly symmetric w-distribution
(on this repository's MWA_extended off-zenith fixture at `epsilon = 1e-6`,
`n_w` goes 251 → 134). The operators put the conjugation back per row, so the
answer is unchanged.

The identity holds for a **real** sky only. `dirty2vis` therefore raises
`ValueError` on a complex image if the plan was folded, naming `hermitian=False`
as the fix; the check is on the array's dtype, so a complex array with a zero
imaginary part is refused too — pass `image.real`. `vis2dirty` has no such
restriction: its output is the real part, and `Re[v z] == Re[conj(v) conj(z)]`
identically, so the fold's plane-count saving applies to the adjoint
unconditionally. Pass `hermitian=False` for a complex sky, or to reproduce the
pre-fold plan geometry.

`phi_hat_oversample=None` (the default) picks a width-dependent oversample
suitable for the kernel chosen by `epsilon` (32 for `W <= 4`, 64 for `W <= 8`,
128 for `W in {9, 10}`, then one doubling per digit up to a cap of 4096); pass
an explicit integer to override.

The returned plan also exposes `max_window_size`,
`max_window_size_per_chan`, `window_buckets`, `window_padding_overhead`,
`window_padding_overhead_adjoint`, `live_row_count` and `empty_plane_count`
for callers that want to inspect whether the windowed strategies will be
efficient on a given uvw distribution. `window_buckets[c]` is channel `c`'s
w-planes sorted into at most four size classes as `((slice_length,
n_planes), ...)`; those lengths are the row slices the windowed **adjoint**
takes, and `max_window_size` — the largest of them over the whole plan — is
what the windowed **forward** takes for every plane. `live_row_count` is the
number of
`(channel, plane, row)` incidences inside a plane's nominal kernel support
(`|w - w_k| <= w_kernel_scale`, as the host computes `w`), and
`empty_plane_count` the number of `(channel, plane)` pairs holding none — a
long tail of empty planes is the signature of a clumped `w`-distribution, and
the reason the windowed strategies stop paying on one.

`live_row_count` is a nominal-support count, not a census of the weights the
operators apply. The operators derive their own `w` inside the JIT — where the
multiply and the `- w0` may contract into one FMA, and where a float32 plan
runs the whole chain in single precision — then test `|z| <= 1`. Measured
against that compiled expression over the review fixtures, the two counts
differ on 20 of 40 cells in float64 and 7 of 10 in float32, never by more than
3 incidences and never by more than 0.19%. `window_padding_overhead` is a work
ratio rather than a kernel-weight audit, so a disagreement of that size sits
well below the precision it is read at; an exact census would have to be taken
against a compiled executable, which does not exist at plan time.

**Plan memory.** The plan stores nothing per `(channel, row)`: the baselines
are kept once in metres (`plan.uvw_m`, `(n_rows, 3)`) next to one scalar per
channel (`plan.inv_lambda = freq / c`), and the per-channel `(u, v)` FINUFFT
coordinates and `w` in wavelengths are derived inside the JIT. A float64 plan
is therefore about `29 * n_rows + 8 * n_chan + 32 * n_l * n_m` bytes plus the
small `n_w`-sized arrays &mdash; 2.4 MB for 16 channels &times; 10k rows at
256&sup2;, 32 MB for 64 channels &times; 1M rows. (28 B/row before the
Hermitian fold's one-byte `plan.flip_sign`.) `plan.uvw_lambda`,
`plan.n_minus_1` and `plan.w_centers` remain readable as derived properties;
reading `plan.uvw_lambda` materialises the full `(n_chan, n_rows, 3)` array,
so it is for introspection, not for hot loops.


## Full `dirty2vis` semantics


Forward operator. `image` may be `(n_chan, n_l, n_m)` or `(n_l, n_m)`
(broadcast across channels), real or complex. Output is complex
`(n_rows, n_chan)`. A **complex** image needs a plan built with
`hermitian=False` &mdash; see `make_plan` above &mdash; and raises `ValueError`
otherwise.

`divide_by_n` (keyword-only, static) applies the measurement equation's `1/n`
factor to the image: `1/n` inside the unit disc, zero outside. The default
`False` omits it, matching ducc0 and every release through v0.1.2.


## Full `vis2dirty` semantics


Adjoint operator. `vis` is complex `(n_rows, n_chan)`; optional `weights` is
real `(n_rows, n_chan)`. Output is real `(n_chan, n_l, n_m)`.

`divide_by_n` (keyword-only, static) defaults to `True` here &mdash; the `1/n`
factor applied on the output, matching ducc's `divide_by_n=True`, with pixels
outside the unit disc returned as exactly 0 — and pixels exactly *on* it
returned as 0 too, where ducc0 returns `inf`. See *The `1/n` factor* above. `False` returns the
analytic-extension result with neither the division nor the mask.

The two defaults are deliberately different (ducc0's pairing) and are
**not an adjoint pair on a wide field**; pass `divide_by_n=True` to both for
imaging and for anything differentiating through the pair. See *Adjoint
operator* above for the measured residuals.
