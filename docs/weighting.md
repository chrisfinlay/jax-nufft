# Weighting: natural, Briggs, tapering and flags

`jax-nufft` **takes** weights; it does not compute them. `vis2dirty(weights=)`
multiplies its argument into the visibilities before gridding — the same
contract as `ducc0`'s `wgt` — and what you put in that array is your imaging
policy, not the gridder's business.

This page covers what goes in it.

| | |
|---|---|
| [The contract](#the-contract) | shape, dtype, and why it is per-channel |
| [Flags](#flags) | zero weights, exactly |
| [UV tapering](#uv-tapering) | synthesising a Gaussian beam |
| [Briggs robust weighting](#briggs-robust-weighting) | uniform through natural |
| [Putting them together](#putting-them-together) | one array, and the PSF |
| [Imaging weights are not likelihood weights](#imaging-weights-are-not-likelihood-weights) | **read this if you are fitting** |

The code below is `tests/weighting_recipes.py`, verified by
`tests/test_weighting.py` — including a check that these blocks still match the
module, so a snippet here cannot rot silently.

## The contract

```
weights : (n_rows, n_chan) real, or None
```

Real by contract: weights multiply visibilities, and a complex weight has no
correct interpretation, so `vis2dirty` rejects one rather than guessing. An
array narrower than the plan's dtype is cast up.

**Why per-channel and not just per-row.** Both tapering and Briggs depend on
`u` and `v` *in wavelengths*, and those scale with frequency. A 200 MHz and a
1.4 GHz channel on the same baseline sit at different uv points, so they get
different weights. That is what the second axis is for:

```python
def uv_lambda(uvw: np.ndarray, freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(u, v)`` in wavelengths, shaped ``(n_rows, n_chan)`` like the weights.

    ``uvw`` is the same metres array ``make_plan`` takes.
    """
    inv_lam = np.asarray(freq) / C_M_S
    return uvw[:, 0:1] * inv_lam[None, :], uvw[:, 1:2] * inv_lam[None, :]
```

Only the adjoint takes weights. The forward does not, and for the ways this
library is normally used it does not need to — see
[Imaging weights are not likelihood weights](#imaging-weights-are-not-likelihood-weights).

## Flags

Set the weight to zero. That is not an approximation: measured against
rebuilding the problem with the flagged rows genuinely removed, a zeroed weight
matches to a **relative L2 of 2e-16** — bit-level agreement.

```python
w = w_natural.copy()
w[flagged] = 0.0
dirty = vis2dirty(plan, vis, weights=w)
```

One thing zeroing does *not* do is shrink the plan. `n_w` is set by the largest
`|w|` among the rows the plan was built from, so flagged rows still pay for
their w-planes. Because it is a maximum, this only bites when flagging removes
the whole extreme-`|w|` tail: dropping the five largest-`|w|` rows of
MWA_extended off30 takes `n_w` from 134 to 121, while flagging a *random* 20
per cent of rows changes it by nothing at all. If your flags correlate with
long baselines, building the plan from the unflagged rows is worth it.

## UV tapering

An image-plane Gaussian of FWHM $\theta$ is a uv-plane Gaussian of FWHM
$4\ln 2/(\pi\theta)$, so down-weighting by

$$
w_{\text{taper}}(u, v) \;=\; \exp\!\left(-\left(u^{2} + v^{2}\right)
  \frac{\pi^{2}\theta^{2}}{4\ln 2}\right)
$$

synthesises a beam of FWHM $\theta$:

```python
def gaussian_taper(u: np.ndarray, v: np.ndarray, fwhm_rad: float) -> np.ndarray:
    """Down-weight long baselines to synthesise a Gaussian beam of ``fwhm_rad``.

    An image-plane Gaussian of FWHM ``theta`` is a uv-plane Gaussian of FWHM
    ``4 ln2 / (pi theta)``, so the per-visibility factor is
    ``exp(-(u^2 + v^2) pi^2 theta^2 / (4 ln 2))``.

    ``u``, ``v`` in wavelengths; ``fwhm_rad`` in radians. The result is 1 at
    the origin and falls monotonically with baseline length, so it never
    increases a weight.
    """
    return np.exp(-(u**2 + v**2) * (np.pi**2 * fwhm_rad**2) / (4.0 * np.log(2.0)))
```

Verified by transforming the taper on a bare uv grid and measuring the
resulting beam rather than trusting the algebra — asked for 3, 5 and 8 pixel
beams, measured 3, 5 and 8. It is checked on a bare grid on purpose: real uv
coverage would convolve in a dirty beam and stop the measurement being about
the taper.

## Briggs robust weighting

Bin the natural weights onto a uv grid and divide each visibility down by how
crowded its own cell is, so densely sampled regions stop dominating:

```python
def briggs(
    u: np.ndarray,
    v: np.ndarray,
    w_natural: np.ndarray,
    n_pix: int,
    pixsize: float,
    robust: float = 0.0,
) -> np.ndarray:
    """Briggs robust weighting: ``robust = -2`` is ~uniform, ``+2`` is ~natural.

    The natural weights are binned onto a uv grid, and each visibility is
    divided down by how crowded its own cell is, so densely sampled regions
    stop dominating.

    The uv cell **must** be ``1 / (n_pix * pixsize)`` -- one over the field of
    view, matching the image grid. A different cell measures a density that
    corresponds to no image and the robustness parameter stops meaning what it
    is supposed to mean.

    Visibilities falling outside the grid keep their natural weight: they are
    beyond the image's uv extent, so no cell population is known for them.
    """
    du = 1.0 / (n_pix * pixsize)
    iu = np.rint(u / du).astype(int) + n_pix // 2
    iv = np.rint(v / du).astype(int) + n_pix // 2
    on_grid = (iu >= 0) & (iu < n_pix) & (iv >= 0) & (iv < n_pix)

    grid = np.zeros((n_pix, n_pix))
    np.add.at(grid, (iu[on_grid], iv[on_grid]), w_natural[on_grid])
    cell_weight = np.zeros_like(w_natural)
    cell_weight[on_grid] = grid[iu[on_grid], iv[on_grid]]

    f_sq = (5.0 * 10.0 ** (-robust)) ** 2 / (np.sum(grid**2) / np.sum(w_natural))
    return w_natural / (1.0 + cell_weight * f_sq)
```

`robust = -2` approaches uniform, `+2` is natural, `0` is the usual
compromise. On MWA_extended off30 the spread between the largest and smallest
weight comes out as:

| `robust` | spread | |
|---|---|---|
| `+2` | 1.0x | indistinguishable from natural |
| `0` | 74x | |
| `-2` | 7.4e5x | crowded cells crushed — uniform |

**The uv cell must be `1 / (n_pix * pixsize)`.** That is one over the field of
view, and it is the cell the image grid implies. Any other cell measures a
density belonging to no image, and `robust` stops meaning what it is defined to
mean.

Two conventions worth deciding deliberately, because implementations differ and
neither is wrong: whether the density count includes the Hermitian conjugate
points `(-u, -v)` — this is your computation, independent of the plan's
`hermitian` fold — and what to do with visibilities outside the grid. The
recipe above leaves those at their natural weight, on the grounds that no cell
population is known for them.

## Putting them together

They are independent factors on one array:

```python
u, v = uv_lambda(uvw, freq)
w = briggs(u, v, w_natural, n_pix, pixsize, robust=0.5)
w = w * gaussian_taper(u, v, fwhm_rad=np.deg2rad(15 / 3600))   # 15 arcsec
w[flagged] = 0.0

dirty = vis2dirty(plan, vis, weights=w) / w.sum()
psf   = vis2dirty(plan, jnp.ones_like(vis), weights=w) / w.sum()
```

Dividing by `w.sum()` is the conventional normalisation and makes the PSF peak
**exactly 1.0**, which is the cheapest check that your weights are doing what
you think. The PSF itself is just the adjoint applied to unit visibilities —
there is no separate entry point for it.

## Imaging weights are not likelihood weights

This is the part with no API to warn you about it, because `weights=` carries
both.

**Imaging.** Briggs and tapering are choices about beam shape and sidelobe
level. Pick whatever makes the map you want.

**Fitting.** If you are differentiating a $\chi^{2}$ in visibility space, the
weights are inverse noise variance — that is, **natural**, times flags. Briggs
weighting down-weights well-sampled uv cells relative to their actual noise, so
putting it in a likelihood biases the inference. It is not a free knob there.

And in that mode the weights do not go through the operator at all. They go in
the loss:

```python
resid = dirty2vis(plan, x) - vis_obs
loss  = jnp.sum(w * jnp.abs(resid) ** 2)
```

`jax.grad` of that produces $2 A^{H}(w \odot (Ax - v))$ — verified to a
relative L2 of 8e-14 — routing the weights through the adjoint's own `weights`
path for you. That is why the forward operator does not need a `weights`
argument to support weighted fitting.

---

See also: [README](../README.md) for the operator reference,
[accuracy.md](accuracy.md) for what `epsilon` buys and why the shipped operator
pair is not an exact adjoint.
