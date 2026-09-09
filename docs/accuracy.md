# Accuracy, precision and adjointness

What `epsilon` buys, over which grid it was measured, what single precision
costs, and why the shipped operator pair is not an exact adjoint. See
[README.md](../README.md) for the summary.

<!--
Moved out of README.md by the 0.2.0 documentation pass. The text is unchanged:
every figure in it was verified against the code or the committed benchmark
JSON during issue #33, and several are recomputed by
tests/test_benchmark_claims.py, so it is moved rather than rewritten.
-->

## Accuracy expectation

`dirty2vis` and `vis2dirty` land within **`2 * epsilon`** of the exact DFT
&mdash; relative L2, in the sign convention above &mdash; forward and adjoint,
on the seven telescope fixtures of `tests/test_accuracy_sweep.py`. The measured
worst cell across that 112-cell matrix is `1.48 * epsilon` (MWA_extended off30,
`epsilon = 1e-12`, adjoint). Against `ducc0.wgridder` (with matched
`divide_by_n` flags) the bound is `3 * epsilon`, the sum of the two
implementations' budgets.

**What the grid covers, and what it holds constant.** This is a measurement
over a finite grid, not a proof, so the axes it does *not* vary are part of the
claim:

| axis | swept | held constant |
|---|---|---|
| `epsilon` (DFT bound) | eight points: `1e-3`, `1e-4`, `1e-5`, `1e-6`, `1e-7`, `1e-8`, `1e-10`, `1e-12` | `1e-9` and `1e-11` are skipped, only to keep the matrix at 112 cells |
| `epsilon` (ducc0 bound) | &mdash; | **`1e-4` and `1e-6` only.** Nothing above or below is checked against ducc0 |
| `w_strategy` | &mdash; | **`dense_scan` only**, which is *not* the shipped default |
| `channel_strategy`, `n_chan` | &mdash; | one channel, `"scan"` |
| `divide_by_n` | &mdash; | the shipped mixed pair (forward `False`, adjoint `True`) |
| `weights`, `hermitian` | &mdash; | `None`; `True` |
| precision | &mdash; | float64 with `jax_enable_x64` |

The `w_strategy` row is the one that matters in practice. A caller who takes
the default gets `"auto"`, which on CPU resolves to `windowed_scan` for the
adjoint on several off-zenith fixtures &mdash; a strategy this sweep never
measures. What connects them is `tests/test_strategies_equivalent.py`, which
pins cross-strategy agreement to `1e-11` at `epsilon` in {`1e-4`, `1e-6`,
`1e-8`}. That is looser than the contract itself at the tightest settings: at
`epsilon = 1e-12` the contract is `2e-12` and the transfer bound is `1e-11`,
5&times; wider. **So the `2 * epsilon` figure is established for `dense_scan`,
and carries to the other strategies only down to about `epsilon = 1e-10`.** If
you need the tight bound at `1e-12`, pin `w_strategy="dense_scan"`.

That worst cell was `0.67 * epsilon` before the `nshift` centring, so the
headroom against the `2 * epsilon` contract has genuinely narrowed, from about
3x to about 1.4x. The cause is structural rather than a regression in the
kernel: centring `n - 1` puts *both* ends of its range at the `eta` extreme
where the w-kernel's aliasing error peaks, where previously only one end was
there. It buys a halving of the w-plane count (see above), and every cell
still meets the contract &mdash; but the margin is now thin enough that a
future change spending more of the error budget should re-run the sweep and
expect to have to widen the kernel rather than assume the slack is there.

Both contracts are for the **default float64 plan with `jax_enable_x64`
enabled**. `float32` / `complex64` inputs are accuracy-limited and do not
meet either bound. See issue #11 for the `make_plan` guard (refusing a
float64 plan when `jax_enable_x64` is off, and warning on a float32 plan
below `epsilon = 1e-5`) and issue #13 for the underlying precision-vs-epsilon
tradeoff.

The contract holds with the default `phi_hat_oversample=None`, which
automatically sizes the phi_hat table to the width `kernel_params(epsilon)`
picks (see `phi_hat_oversample_for_w`). There is no public API to override
the kernel width itself -- it is derived from `epsilon` alone -- so ordinary
callers never need to pass `phi_hat_oversample` explicitly; it exists as an
escape hatch for testing the phi_hat table at a size other than the
schedule's default.

## Precision

The plan owns the precision. `make_plan(..., dtype=...)` takes `jnp.float64`
(the default) or `jnp.float32`; the choice sets the dtype of every plan array
and is exposed as `plan.real_dtype` / `plan.complex_dtype`.

**float64 (default) requires `jax_enable_x64`.** JAX ships with x64 *off*, in
which case it silently truncates every array to single precision — a plan that
looks like float64 but isn't. `make_plan` refuses to build one instead:

```python
import jax
jax.config.update("jax_enable_x64", True)   # before the first JAX array
# or: JAX_ENABLE_X64=1 in the environment
```

**Opting into float32.** Single precision roughly halves memory and is the
natural choice on GPUs where fp32 throughput dominates. The two halves of
"roughly" differ, and it is worth knowing which you are buying:

* **Operator scratch halves, essentially exactly** &mdash; a ratio of 1.99987 to
  1.99999 over three size classes, two strategies and both operators. The
  shortfall is a fixed overhead under 512 bytes that does not scale.
* **The plan halves only when it is image-dominated** &mdash; 0.501&times; at
  256&sup2; with 600 rows, but 0.586&times; at 150&sup2; with 4.9M rows. The
  per-pixel term goes 32 B to 16 B, but the per-row term only goes 29 B to
  17 B, because `sort_perm` is `int32` and `flip_sign` is `int8` and neither
  follows the floating dtype. Row-heavy plans keep the larger share.

Both are pinned in `tests/test_dtype.py`.

```python
plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-4,
                 dtype=jnp.float32)
vis = dirty2vis(plan, jnp.asarray(image, dtype=jnp.float32))  # complex64 out
```

This works with x64 either on or off; `uvw` and `freq` are cast to the plan
dtype, so float64 inputs are fine.

**Achievable accuracy in single precision is about `epsilon = 1e-5`.** The
measured relative-error floor across the telescope fixtures is ~3.4e-5, so
`epsilon = 1e-4` is comfortably reachable (parity with `ducc0` within `3 *
epsilon`) while `epsilon = 1e-6` is not. `make_plan` emits a `UserWarning` for
a float32 plan with `epsilon < 1e-5` rather than quietly missing the target.
Use float64 whenever you need more.

**Known limits of the float32 path.** The ~1e-5 figure is measured on the
telescope fixtures in `tests/conftest.py`. Two further effects degrade
single-precision accuracy beyond it and are not yet addressed:

- **Large absolute `w`.** The per-plane phase `2*pi*w_k*(n-1)` is formed and
  exponentiated directly in float32 with no range reduction, so its rounding
  error grows with `|w|`. Tracked in
  [#13](https://github.com/chrisfinlay/jax-nufft/issues/13).
- **Pixels near the horizon.** The adjoint reconstructs `n` from `n - 1` by a
  cancelling subtraction and then divides by it, so relative accuracy is lost
  as `n` approaches zero. Tracked in
  [#12](https://github.com/chrisfinlay/jax-nufft/issues/12).

Neither affects the default float64 path.

**Mixing dtypes.** Per-call arrays are measured against the plan: an image,
`vis` or `weights` array *narrower* than the plan is cast up (a float32 image
into a float64 plan gives a complex128 result), while a *wider* one raises
`TypeError` naming both dtypes rather than silently discarding the precision
you produced.

## Adjointness: equal flags only

`dirty2vis` and `vis2dirty` form an exact adjoint pair &mdash;

```
Re<A x, y> == <x, A^H y>          for real x
```

&mdash; **only when both are given the same `divide_by_n`.** Measured on the
EDA2 full-sky fixture (`tests/conftest.py`: 64&times;64 at a 120&deg; field of
view, 400 rows, `synthetic_uvw(EDA2, 0.0, seed=0)`), `epsilon = 1e-6`, float64,
with `tests/test_divide_by_n.py`'s image and visibilities:

| `dirty2vis` | `vis2dirty` | relative residual |
|---|---|---|
| `False` | `False` | **1.293e-13** |
| `True`  | `True`  | **1.327e-15** |
| `False` | `True` (the shipped defaults) | **0.630** |
| `True`  | `False` | 0.630 |

The shipped defaults are the mixed pair, so **they are not an adjoint pair on
a wide field**. Nor does the classical `n`-corrected form rescue them: on the
same fixture `Re<A x, y> == <n x, A^H y>` is off by **0.273**. The entire
failure lives in the **1155 of 4096 pixels outside the unit disc**, where the
forward at its default evaluates the analytic extension while the adjoint at
its default zeroes them &mdash; restrict the image to the disc and the mixed
pair satisfies the `n x` identity again, to **4.105e-16**. That is the `n`-
corrected form specifically: masked to the disc, the *plain* identity still
reads 7.306e-01, because the mixed pair differs from an adjoint pair by the
factor of `n` whether or not the image runs past the circle.

Narrow fields hide it, because they have no outside-disc pixels at all: on
MWA_compact the mixed pair's `n x` residual is 1.1794e-14 at zenith and
1.9360e-12 at 30&deg;.

**Recommendation: pass `divide_by_n=True` to both operators.** It is the
factor the measurement equation actually carries, and it is what any
gradient-based use of the pair needs:

```python
vis = dirty2vis(plan, image, divide_by_n=True)
dirty = vis2dirty(plan, vis, divide_by_n=True)     # exactly A^H of the above
```

Both equal-flag pairs are adjoint to far inside any tolerance in this library,
and the recommendation does **not** rest on which residual is smaller. Read
across the precisions, the ordering reverses: in float64 `True` measures
1.3e-15 against `False`'s 1.3e-13 on EDA2 full-sky, but in float32 `True` is
the *larger* residual on every fixture measured (7.2e-8 / 1.1e-7 / 1.8e-7
against 1.8e-8 / 2.9e-8 / 2.0e-8 on EDA2 zenith, MWA_compact zenith and
MWA_compact off30). That is what one would expect from a `1/n` diagonal, which
amplifies pixels near the horizon where `n -> 0` and so widens the dynamic
range a single-precision sum has to carry. These are dot-product residuals —
round-off in an identity that is exact on paper — and not a measurement of
either operator's conditioning; neither ordering should be read as one.

The defaults are left as they are for ducc0 compatibility and because moving
them would silently change every existing caller's answer by a factor of `n`.

The residuals above come from `tests/test_divide_by_n.py`. Its section-6
matrices enumerate every axis the operators dispatch on — both flag values,
all four `w_strategy` values, both `channel_strategy` values, both `hermitian`
settings and both precision legs, on a two-channel plan — across **three**
tests, none of which gates the claim alone:

1. the **dot-product identity**, which cannot see a defect applied
   consistently to both operators (that is just the other flag's pair, and
   still exactly adjoint);
2. a **strategy/fold comparison** whose reference stays pinned to
   `(dense_scan, scan)`, which catches such a defect only when the reference
   is spared by it;
3. a per-cell **semantic oracle** — `divide_by_n=True` against the same call
   with the flag off and a `1/n` built from `(l, m)` alone applied by hand —
   which is the only one that can fail when a wrong-but-self-adjoint diagonal
   is shared by the reference too.

The identity is scored **per channel**, not on the channel sum: the two
channel blocks are `+3909.9` and `−3837.0` on this fixture, so summing them
first leaves 1.86%, and the bound that then has to accommodate the inflation
is one whose honesty rests on a headroom judgement unrelated to the operators
— the one actually chosen that way passed a uniform 5e-5 one-sided error.

Bounds: `1e-11` in float64 and `5e-6` in float32 (`epsilon = 1e-5`). **Every
tolerance in the test module is measured on two backends — CPU (macOS arm64)
and one NVIDIA GH200** — rather than on whichever was to hand, and the reason
is that they are not all the same kind of number. Bounds that measure a real
approximation gap against an exact or independent reference (the DFT and ducc0
contracts, folded-vs-unfolded agreement) are backend-stable, agreeing to under
5% across the two machines. Bounds that measure round-off in a relation exact
on paper — this identity among them — are reduction order and nothing else,
and move by 8x to 780x. So on CPU the float32 identity measures 1.8e-7 and on
the GPU's `JAX_ENABLE_X64=0` leg 1.5e-6: CPU evidence alone overstates the
margin by 8x in exactly the population that looks safest. In float64 the CPU is
the worse backend instead, by 350x. What fixes the float32 value is not
headroom but detection — a uniform 5e-5 one-sided error still fails it by 34x
at the worst measured cell. The per-bound measurements for both backends, and
which kind each bound is, are tabulated in `tests/test_divide_by_n.py`'s module
docstring.

Two comparisons of a call against itself (the no-argument call against the
explicit default, and `divide_by_n=True` against a perturbation outside the
disc) are held at round-off rather than bit-equality for the same reason:
XLA:GPU reductions are not run-to-run deterministic, so one executable run
twice on identical inputs need not give identical bits. The exact claim those
comparisons used to carry — that the no-argument call routes to the declared
default — is asserted at the JIT boundary instead, where it is a property of
the dispatch rather than of the arithmetic.

