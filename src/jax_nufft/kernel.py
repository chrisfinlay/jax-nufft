"""Exp-of-semicircle gridding kernel and its (numerical) Fourier transform.

The kernel used here matches the form introduced in Barnett, Magland &
af Klinteberg (2019), "FINUFFT", SIAM J. Sci. Comput. 41 C479:

    phi(z; beta) = exp(beta * (sqrt(1 - z^2) - 1))     for |z| <= 1
                = 0                                    otherwise

`phi` itself is evaluated inside the JIT-compiled hot path (to multiply
visibilities by the w-direction kernel), so the JAX ``phi`` is jnp-friendly.

`phi_hat` is the continuous Fourier transform of phi:

    phi_hat(eta; beta) = integral_{-1}^{+1} phi(z) exp(-2 pi i eta z) dz

It has no closed form for this kernel, so we compute it numerically once at
*planning time* (outside the JIT trace) via a zero-padded FFT, store it on a
regular eta grid, and use linear interpolation to evaluate it at arbitrary
(real) eta values. Because this happens before tracing, the precomputation
uses ``numpy``; the resulting interpolated values are then promoted to
``jax.numpy`` arrays and treated as JIT-time constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array


def phi_hat_oversample_for_w(w_kernel_width: int) -> int:
    """Recommended ``oversample`` for the phi_hat table at width ``W``.

    With v0.1.1's fixed ``x0`` sampling, the maximum eta evaluated by the
    image-domain correction is ``x0 * W / 2 = W / 8``, which grows with
    W. Bump the oversample so that cubic-Lagrange interpolation off the
    phi_hat table stays accurate (the interpolation error scales like
    ``eta_step**4``, with ``eta_step = 1 / (2 * oversample)``).

    Conditioning is not the binding constraint: the smallest ``phi_hat`` over
    ``|eta| <= x0 * W / 2`` is 8.2e-2 at W = 13 and 5.8e-2 at W = 15, four
    orders of magnitude above the table's 1e-6 safety floor. The binding
    constraint is *interpolation accuracy*, because ``phi_hat_n`` is divided
    into the image: whatever relative error the table carries lands
    one-for-one in the output. The requested accuracy for a width is
    ``eps = 10**-(W - 1)`` (:func:`kernel_params` inverted), so the table has
    to be good to roughly ``eps / 10``, and off-grid interpolation error
    measured against a 64x finer table is

        W      = 4        8        10       11       12       13
        os=32    4.9e-08  2.8e-08  2.3e-08  2.0e-08  1.8e-08  1.5e-08
        os=64    3.1e-09  1.7e-09  1.4e-09  1.3e-09  1.1e-09  9.4e-10
        os=128   1.9e-10  1.1e-10  8.8e-11  7.8e-11  6.8e-11  5.9e-11
        os=512   7.5e-13  4.1e-13  3.5e-13  3.1e-13  2.7e-13  2.3e-13

    i.e. a factor 16 per doubling (``eta_step**4`` with
    ``eta_step = 1 / (2 * oversample)``) against a requirement that tightens
    by 10 per digit -- so the oversample doubles once per digit above W = 9
    and keeps a growing margin. Issue #9 added the steps above 128: at
    eps = 1e-12 (W = 13) the old flat 128 left a 5.9e-11 table error, which
    showed up end-to-end as 1.4e-11 relative -- 14x the requested epsilon,
    with the w-kernel itself already well inside its budget. The cap at 2048
    is where the table's own float64 noise (~1e-16 relative) takes over;
    building it costs about 190 ms and 270 MB of transient host memory at
    plan time, and only for eps <= 1e-12 requests.

    Note this is *not* measured by ``test_phi_hat_conditioning_at_v011_eta_max``
    in the test suite: that test's check points land exactly on table nodes,
    where cubic Lagrange is exact by construction.
    """
    if w_kernel_width <= 4:
        return 32
    if w_kernel_width <= 8:
        return 64
    # W >= 9: one doubling per requested digit, capped where float64 noise wins.
    return min(2048, 128 * 2 ** (w_kernel_width - 9))


def kernel_params(epsilon: float) -> tuple[int, float]:
    """Return ``(W, beta)`` for the requested accuracy.

    The width follows the practical rule of [Barnett+2019] eq. 10 -- "W is one
    more than the desired number of digits" -- i.e.

        W = ceil(-log10(epsilon / 10))

    which is what FINUFFT itself implements in ``src/spreadinterp.cpp::
    setup_spreader`` at upsampling factor ``sigma = 2``. ``beta = 2.30 * W`` is
    the matching shape parameter from the same equation.

    The rule is not a fudge factor: [Barnett+2019] Theorem 7 bounds the
    aliasing error of the exp-of-semicircle kernel by
    ``exp(-pi * W * gamma * sqrt(1 - 1/sigma))``, which at ``sigma = 2`` and the
    standard ``gamma`` is ``~exp(-2.2 * W)`` -- exactly one decimal digit per
    unit of ``W``, with no margin. So the number of cells has to be the number
    of digits requested, plus one for the constants the bound drops.

    Before issue #9 this used ``W = ceil(-log10(eps) * 2/pi) + 2``, which grows
    by only ``2/pi < 1`` per digit and was therefore 0, 0, 0, 1, 1, 1, 2, 3
    cells short across ``eps = 1e-3 .. 1e-12``; it also mapped 1e-5 and 1e-6
    (and 1e-9/1e-10, 1e-11/1e-12) onto the same width, so tightening epsilon
    across those pairs bought no accuracy at all. Measured against the exact
    DFT the shortfall was 3-4.4x epsilon at 1e-6..1e-8 and 26-550x at
    1e-10..1e-12; with this rule every cell of the accuracy sweep lands within
    2x (see ``tests/test_accuracy_sweep.py``).

    jax-finufft may still pick a slightly different ``W`` internally for the
    (u,v) directions; what matters here is that the w-direction kernel is
    provisioned for the epsilon the caller asked for.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1); got {epsilon!r}")
    # Below 1e-14 the rule asks for W > 15. That is past what the phi_hat table
    # and the float64 kernel can honour (and past what the (u,v) NUFFT can
    # deliver anyway), so refuse rather than silently return a width that does
    # not buy the requested accuracy.
    if epsilon < 1e-14:
        raise ValueError(
            f"epsilon={epsilon!r} is below the smallest supported value 1e-14 "
            "(the width rule would ask for W > 15)"
        )
    w = math.ceil(-math.log10(epsilon / 10.0))
    # Clip to a sensible minimum: W < 2 has effectively no kernel support.
    w = max(w, 2)
    beta = 2.30 * w
    return w, beta


def phi(z: Array, beta: float) -> Array:
    """Exp-of-semicircle kernel evaluated at ``z`` (jnp-compatible).

    Returns 0 outside ``|z| <= 1`` so that callers can apply it on the full
    range of differences ``(w_lambda - w_k) / scale`` without explicit masking.
    """
    z2 = z * z
    inside = jnp.maximum(1.0 - z2, 0.0)
    return jnp.where(z2 <= 1.0, jnp.exp(beta * (jnp.sqrt(inside) - 1.0)), 0.0)


def phi_numpy(z: np.ndarray, beta: float) -> np.ndarray:
    """Numpy version of :func:`phi`, used inside the planning-time precompute."""
    z2 = z * z
    inside = np.maximum(1.0 - z2, 0.0)
    out = np.where(z2 <= 1.0, np.exp(beta * (np.sqrt(inside) - 1.0)), 0.0)
    return out


@dataclass(frozen=True)
class PhiHatTable:
    """Pre-computed values of ``phi_hat`` on a regular eta grid.

    The grid covers symmetric range ``[-eta_max, +eta_max]`` with uniform
    spacing ``eta_step``. Linear interpolation off this grid is implemented by
    :meth:`evaluate`.
    """

    beta: float
    eta_step: float
    eta_max: float
    values: np.ndarray  # shape (N,), real, indexed so that values[N//2] == phi_hat(0)

    @property
    def n(self) -> int:
        return self.values.shape[0]

    def evaluate(self, eta: np.ndarray) -> np.ndarray:
        """Cubic Lagrange interpolation of ``phi_hat`` at the requested eta values.

        Uses the four nearest grid samples (two on each side of the requested
        point), which gives ``O(eta_step^4)`` accuracy on smooth functions. This
        keeps the table small while supporting tight ``epsilon`` requests.

        Raises ``ValueError`` if any |eta| exceeds the table range.
        """
        eta_arr = np.asarray(eta, dtype=np.float64)
        if np.any(np.abs(eta_arr) > self.eta_max):
            raise ValueError(
                f"eta value {np.max(np.abs(eta_arr)):.6g} outside phi_hat table "
                f"range [-{self.eta_max:.6g}, {self.eta_max:.6g}]"
            )
        idx_float = eta_arr / self.eta_step + (self.n // 2)
        # Use the 4-point stencil (i1-1, i1, i1+1, i1+2) where i1 = floor(idx_float)
        # is the lower-left neighbour. Clamp so the stencil stays inside the table.
        i1 = np.floor(idx_float).astype(np.int64)
        i1 = np.clip(i1, 1, self.n - 3)
        t = idx_float - i1
        v_m1 = self.values[i1 - 1]
        v_0 = self.values[i1]
        v_p1 = self.values[i1 + 1]
        v_p2 = self.values[i1 + 2]
        # Lagrange basis on (-1, 0, 1, 2) evaluated at t:
        #   L_{-1}(t) = -t (t-1)(t-2) / 6
        #   L_0(t)    = (t+1)(t-1)(t-2) / 2
        #   L_{+1}(t) = -(t+1) t (t-2) / 2
        #   L_{+2}(t) = (t+1) t (t-1) / 6
        l_m1 = -t * (t - 1.0) * (t - 2.0) / 6.0
        l_0 = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0
        l_p1 = -(t + 1.0) * t * (t - 2.0) / 2.0
        l_p2 = (t + 1.0) * t * (t - 1.0) / 6.0
        return l_m1 * v_m1 + l_0 * v_0 + l_p1 * v_p1 + l_p2 * v_p2


def compute_phi_hat_table(
    beta: float,
    eta_max_request: float,
    n_fine: int = 4096,
    oversample: int = 16,
    safety_floor: float = 1e-6,
) -> PhiHatTable:
    """Compute ``phi_hat`` on a fine eta grid via zero-padded FFT.

    Parameters
    ----------
    beta:
        Kernel shape parameter.
    eta_max_request:
        Largest |eta| at which ``phi_hat`` will subsequently be evaluated. The
        returned table is guaranteed to cover at least this range, plus a small
        margin so that endpoint interpolation is safe.
    n_fine:
        Number of samples used to discretise ``phi(z)`` on ``z in [-1, 1)``.
    oversample:
        FFT zero-pad factor, controlling the eta-grid spacing
        (``deta = 1 / (n_fine * oversample * dz)``).
    safety_floor:
        Smallest acceptable value of ``phi_hat`` over the kept range; if any
        sample drops below this, raise ``ValueError`` (dividing by such a small
        value would amplify noise drastically).
    """
    if oversample < 1:
        raise ValueError(f"oversample must be >= 1; got {oversample}")
    if n_fine % 2 != 0:
        raise ValueError(f"n_fine must be even; got {n_fine}")

    n_fft = int(n_fine * oversample)
    if n_fft % 2 != 0:
        # n_fft is even because n_fine is even and oversample is integer >= 1.
        raise AssertionError("expected n_fft to be even")

    dz = 2.0 / n_fine

    # Build phi sampled on the centered, zero-padded grid of length n_fft. The
    # padded array has phi(z) on the central n_fine samples and zeros elsewhere,
    # with z=0 sitting exactly at index n_fft // 2.
    pad = (n_fft - n_fine) // 2
    z_inner = (np.arange(n_fine) - n_fine // 2) * dz
    phi_inner = phi_numpy(z_inner, beta)
    phi_centered = np.zeros(n_fft, dtype=np.float64)
    phi_centered[pad : pad + n_fine] = phi_inner

    # Approximate phi_hat(eta_k) = integral phi(z) exp(-2 pi i eta_k z) dz
    # via the trapezoidal-equivalent DFT with grid spacing dz. ifftshift moves
    # z=0 to index 0; fft does the sum; fftshift puts negative-eta first.
    phi_for_fft = np.fft.ifftshift(phi_centered)
    fft_out = np.fft.fft(phi_for_fft)
    phi_hat_complex = np.fft.fftshift(fft_out) * dz

    # phi(z) is real and symmetric, so phi_hat is real to within FFT rounding.
    phi_hat = phi_hat_complex.real

    eta_step = 1.0 / (n_fft * dz)
    eta_max_table = (n_fft // 2) * eta_step

    if eta_max_request > eta_max_table:
        raise ValueError(
            f"requested eta_max={eta_max_request:.6g} exceeds table range "
            f"{eta_max_table:.6g}; increase oversample or n_fine"
        )

    # Inspect just the range we actually care about for the safety check.
    keep_mask = np.abs(np.arange(n_fft) - n_fft // 2) * eta_step <= eta_max_request
    if np.any(phi_hat[keep_mask] < safety_floor):
        worst = float(np.min(phi_hat[keep_mask]))
        raise ValueError(
            f"phi_hat dropped to {worst:.3g} (< safety_floor={safety_floor:.3g}) "
            f"within |eta| <= {eta_max_request:.6g}; try increasing oversample or beta"
        )

    return PhiHatTable(
        beta=beta,
        eta_step=eta_step,
        eta_max=eta_max_table,
        values=phi_hat,
    )
