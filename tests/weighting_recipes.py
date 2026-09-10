"""Reference implementations of the visibility weighting schemes.

``jax-nufft`` takes weights, it does not compute them: ``vis2dirty(weights=)``
multiplies its ``(n_rows, n_chan)`` argument into the visibilities before
gridding, and what goes in that array is the caller's policy. Natural
weighting, Briggs robust weighting and a uv taper are all *policies* -- they
belong to the imaging choice, not to the gridder -- so they live here rather
than in ``src/``.

They live under ``tests/`` rather than in an examples directory because
``docs/weighting.md`` quotes them verbatim and
:mod:`tests.test_weighting` both pins their behaviour and checks that the
quoted code still matches this module. A recipe nobody runs rots; this one
fails the suite when it does.

Everything here is plain numpy. Weights are real, and the shape is
``(n_rows, n_chan)`` because ``u`` and ``v`` in wavelengths scale with
frequency: a multi-channel dataset genuinely has a different taper and a
different Briggs weight per channel.
"""

from __future__ import annotations

import numpy as np

#: Speed of light, m/s. Local so this module imports without jax_nufft.
C_M_S = 299792458.0


def uv_lambda(uvw: np.ndarray, freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(u, v)`` in wavelengths, shaped ``(n_rows, n_chan)`` like the weights.

    ``uvw`` is the same metres array ``make_plan`` takes.
    """
    inv_lam = np.asarray(freq) / C_M_S
    return uvw[:, 0:1] * inv_lam[None, :], uvw[:, 1:2] * inv_lam[None, :]


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
