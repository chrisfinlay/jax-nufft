"""Exact-DFT accuracy sweep over the seven review fixtures (issue #9).

This is the harness every later issue means when it says "re-run the accuracy
sweep". It answers one question, for the whole review matrix at once: *when a
caller asks for ``epsilon``, how much error do they actually get?*

    seven fixtures  x  eight epsilon values  x  {forward, adjoint}

The fixtures are the review set: EDA2 zenith; MWA_compact zenith and off30;
MWA_extended zenith and off30; MeerKAT zenith and off30 -- built exactly as
``tests/conftest.py``'s ``synthetic_uvw(tel, angle, seed=0)`` with a single
channel at ``tel.freq_hz``, a ``tel.n_pix``-square image at ``tel.pixsize``,
and image / visibility data from ``np.random.default_rng(7)``.

The metric is the relative L2 error ``norm(a - b) / norm(b)`` against the
*exact DFT* in the repository's sign convention (AGENTS.md section 1) --
not against ducc0. ducc0 is an independent implementation with its own error
budget; the DFT is the definition of the answer, which is what makes "achieved
error <= 2 x epsilon" a contract rather than a comparison.

Cost and gating
---------------
The sweep is gated behind ``--runsweep`` (see ``tests/conftest.py``) because it
is minutes, not seconds: 112 plan-and-call combinations plus fourteen exact
DFTs. It is also inherently a float64 test -- at float32 the DFT reference is
worth roughly 1e-7 relative and the sweep's tightest cells ask for 1e-12 -- so
it skips outright when ``jax_enable_x64`` is off.

The references are vectorised over row chunks and cached per fixture, so each
256-pixel fixture's DFT costs a few seconds once rather than a minute per
epsilon. Correctness of that vectorisation is not taken on trust:
``test_chunked_references_match_row_loop`` (which runs in the default suite,
in milliseconds) pins it against the row-at-a-time
``tests/test_against_dft.py::_reference_forward`` and
``tests/test_adjoint.py::_reference_adjoint`` that the parity tests use.

Reading the output
------------------
Every cell prints its row as it is measured, and the full table is printed at
module teardown, so ``pytest --runsweep -s tests/test_accuracy_sweep.py``
leaves the before/after numbers in the CI log for a human to diff.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
from tests.conftest import (
    EDA2,
    MEERKAT,
    MWA_COMPACT,
    MWA_EXTENDED,
    Telescope,
    synthetic_uvw,
)
from tests.test_adjoint import _reference_adjoint
from tests.test_against_dft import _reference_forward, reference_lmn_grids

# A parallel branch is adding a shared ``requires_x64`` marker to conftest;
# until that lands this module carries its own copy, and the two get reconciled
# at rebase time.
requires_x64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="the accuracy sweep measures down to eps=1e-12 and needs jax_enable_x64",
)

# The accuracy contract this sweep enforces: the achieved relative L2 error
# against the exact DFT must not exceed twice the requested epsilon, in either
# direction, on any fixture. Two is headroom for the reference's own
# conditioning, not for an under-provisioned kernel; ducc0 sits at or below
# 0.24x epsilon on the same inputs.
MAX_ERROR_RATIO = 2.0

# The full epsilon ladder from the review. 1e-9 and 1e-11 are omitted only to
# keep the matrix at 112 cells; 1e-5/1e-6 and 1e-10/1e-12 are the interesting
# neighbours because the pre-issue-#9 width rule collapsed pairs of them onto a
# single kernel width.
SWEEP_EPS: tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12)

# The seven review fixtures, as (telescope, zenith angle in degrees).
SWEEP_FIXTURES: tuple[tuple[Telescope, float], ...] = (
    (EDA2, 0.0),
    (MWA_COMPACT, 0.0),
    (MWA_COMPACT, 30.0),
    (MWA_EXTENDED, 0.0),
    (MWA_EXTENDED, 30.0),
    (MEERKAT, 0.0),
    (MEERKAT, 30.0),
)

# ``dense_scan`` is the reference traversal. The other three differ from it only
# in floating-point reduction order (AGENTS.md section 5) and are already held
# against it by tests/test_boundary_planes.py and the per-strategy DFT parity in
# tests/test_against_dft.py, so sweeping all four would quadruple the runtime to
# measure the same number four times.
SWEEP_STRATEGY = "dense_scan"

# Rows per chunk in the vectorised DFT references. At 256^2 pixels one chunk of
# 64 rows is a 64 x 65536 complex128 temporary (~67 MB); larger chunks stop
# paying for themselves once they fall out of cache.
_ROW_CHUNK = 64


def _fixture_id(telescope: Telescope, zenith_angle_deg: float) -> str:
    suffix = "zenith" if zenith_angle_deg == 0.0 else f"off{int(zenith_angle_deg)}"
    return f"{telescope.name}_{suffix}"


# --------------------------------------------------------------------------
# Vectorised exact-DFT references
# --------------------------------------------------------------------------
# Same math as tests/test_against_dft.py::_reference_forward and
# tests/test_adjoint.py::_reference_adjoint -- the row loop is simply lifted
# into a matmul over a chunk of rows so a 256^2 fixture takes seconds instead
# of minutes. ``test_chunked_references_match_row_loop`` holds the two forms
# together.


def reference_forward_chunked(
    image: np.ndarray,
    uvw: np.ndarray,
    freq: np.ndarray,
    pixsize_l: float,
    pixsize_m: float,
    row_chunk: int = _ROW_CHUNK,
) -> np.ndarray:
    """Chunk-vectorised twin of ``test_against_dft._reference_forward``."""
    n_chan, n_l, n_m = image.shape
    n_rows = uvw.shape[0]
    ll, mm, nm1 = reference_lmn_grids((n_l, n_m), pixsize_l, pixsize_m)
    ll_f, mm_f, nm1_f = ll.ravel(), mm.ravel(), nm1.ravel()
    out = np.zeros((n_rows, n_chan), dtype=np.complex128)
    for c in range(n_chan):
        scale = freq[c] / SPEED_OF_LIGHT
        uvw_lambda = uvw * scale
        image_flat = image[c].ravel().astype(np.complex128)
        for start in range(0, n_rows, row_chunk):
            stop = min(start + row_chunk, n_rows)
            u = uvw_lambda[start:stop, 0, None]
            v = uvw_lambda[start:stop, 1, None]
            w = uvw_lambda[start:stop, 2, None]
            # phase = -2 pi i (u l + v m - w (n - 1)), one row per visibility.
            phase = -2j * np.pi * (u * ll_f + v * mm_f - w * nm1_f)
            out[start:stop, c] = np.exp(phase) @ image_flat
    return out


def reference_adjoint_chunked(
    vis: np.ndarray,
    uvw: np.ndarray,
    freq: np.ndarray,
    image_shape: tuple[int, int],
    pixsize_l: float,
    pixsize_m: float,
    weights: np.ndarray | None = None,
    row_chunk: int = _ROW_CHUNK,
) -> np.ndarray:
    """Chunk-vectorised twin of ``test_adjoint._reference_adjoint``."""
    n_l, n_m = image_shape
    n_rows, n_chan = vis.shape
    _ll, _mm, nm1 = reference_lmn_grids(image_shape, pixsize_l, pixsize_m)
    ll_f, mm_f, nm1_f = _ll.ravel(), _mm.ravel(), nm1.ravel()
    n_grid = nm1 + 1.0
    out = np.zeros((n_chan, n_l * n_m), dtype=np.complex128)
    for c in range(n_chan):
        scale = freq[c] / SPEED_OF_LIGHT
        uvw_lambda = uvw * scale
        v_eff = vis[:, c] if weights is None else vis[:, c] * weights[:, c]
        for start in range(0, n_rows, row_chunk):
            stop = min(start + row_chunk, n_rows)
            u = uvw_lambda[start:stop, 0, None]
            v = uvw_lambda[start:stop, 1, None]
            w = uvw_lambda[start:stop, 2, None]
            phase = +2j * np.pi * (u * ll_f + v * mm_f - w * nm1_f)
            out[c] += v_eff[start:stop] @ np.exp(phase)
    dirty = out.real.reshape(n_chan, n_l, n_m)
    # Same 1/n on the output as ducc's divide_by_n=True, with the pixels whose
    # analytic-extension n is non-positive zeroed rather than divided.
    return np.where(n_grid > 0.0, dirty / np.maximum(n_grid, 1e-30), 0.0)


# --------------------------------------------------------------------------
# Per-fixture data + cached references
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _FixtureData:
    """Inputs and exact-DFT outputs for one review fixture.

    The references depend only on the fixture, never on epsilon, so they are
    computed once and reused across all eight epsilon values.
    """

    name: str
    uvw: np.ndarray
    freq: np.ndarray
    image: np.ndarray  # (1, n_pix, n_pix) real
    vis: np.ndarray  # (n_rows, 1) complex
    image_shape: tuple[int, int]
    pixsize: float
    vis_ref: np.ndarray
    dirty_ref: np.ndarray


_FIXTURE_CACHE: dict[str, _FixtureData] = {}


def _fixture_data(telescope: Telescope, zenith_angle_deg: float) -> _FixtureData:
    name = _fixture_id(telescope, zenith_angle_deg)
    cached = _FIXTURE_CACHE.get(name)
    if cached is not None:
        return cached

    uvw = synthetic_uvw(telescope, zenith_angle_deg, seed=0)
    freq = np.array([telescope.freq_hz])
    image_shape = (telescope.n_pix, telescope.n_pix)
    pixsize = telescope.pixsize
    # One generator for both the image and the visibilities, per the review's
    # fixture definition.
    rng = np.random.default_rng(7)
    image = rng.standard_normal((1, *image_shape))
    vis = (
        rng.standard_normal((telescope.n_rows, 1)) + 1j * rng.standard_normal((telescope.n_rows, 1))
    ).astype(np.complex128)

    data = _FixtureData(
        name=name,
        uvw=uvw,
        freq=freq,
        image=image,
        vis=vis,
        image_shape=image_shape,
        pixsize=pixsize,
        vis_ref=reference_forward_chunked(image.astype(np.complex128), uvw, freq, pixsize, pixsize),
        dirty_ref=reference_adjoint_chunked(vis, uvw, freq, image_shape, pixsize, pixsize),
    )
    _FIXTURE_CACHE[name] = data
    return data


# --------------------------------------------------------------------------
# Result table
# --------------------------------------------------------------------------

_HEADER = f"{'fixture':<22}{'eps':>9}{'dir':>9}{'error':>12}{'x eps':>10}{'n_w':>7}{'W':>4}"


@dataclass(frozen=True)
class _Row:
    fixture: str
    eps: float
    direction: str
    error: float
    n_w: int
    width: int

    def render(self) -> str:
        return (
            f"{self.fixture:<22}{self.eps:>9.0e}{self.direction:>9}"
            f"{self.error:>12.3e}{self.error / self.eps:>10.2f}{self.n_w:>7}{self.width:>4}"
        )


_ROWS: list[_Row] = []


@pytest.fixture(scope="module", autouse=True)
def _print_sweep_table() -> Iterator[None]:
    """Dump the assembled table once the module is done.

    Individual rows are also printed as they are measured, so a cell that fails
    still reports its number in the failure's captured stdout; this final block
    is the copy-pasteable version for the PR description.
    """
    yield
    if not _ROWS:
        return
    print("\n\n=== exact-DFT accuracy sweep ===")
    print(_HEADER)
    print("-" * len(_HEADER))
    for row in _ROWS:
        print(row.render())
    worst = max(_ROWS, key=lambda r: r.error / r.eps)
    print("-" * len(_HEADER))
    print(
        f"worst cell: {worst.fixture} eps={worst.eps:.0e} {worst.direction} "
        f"-> {worst.error / worst.eps:.2f}x eps (bound {MAX_ERROR_RATIO:g}x)"
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_chunked_references_match_row_loop() -> None:
    """The vectorised references must reproduce the row-at-a-time originals.

    Cheap on purpose (8x8 image, 6 rows, small FoV) so it runs in the default
    suite: it is what licenses the sweep to use the fast references instead of
    the ones the parity tests are written against.
    """
    rng = np.random.default_rng(3)
    n_l = n_m = 8
    n_rows = 6
    pixsize = 0.01
    uvw = np.column_stack(
        [
            rng.uniform(-80.0, 80.0, size=n_rows),
            rng.uniform(-80.0, 80.0, size=n_rows),
            rng.uniform(-20.0, 20.0, size=n_rows),
        ]
    )
    freq = np.array([1.0e9, 1.4e9])
    image = rng.standard_normal((2, n_l, n_m)) + 1j * rng.standard_normal((2, n_l, n_m))
    vis = (rng.standard_normal((n_rows, 2)) + 1j * rng.standard_normal((n_rows, 2))).astype(
        np.complex128
    )

    # row_chunk=4 does not divide n_rows=6, so the ragged final chunk is
    # exercised too.
    np.testing.assert_allclose(
        reference_forward_chunked(image, uvw, freq, pixsize, pixsize, row_chunk=4),
        _reference_forward(image, uvw, freq, pixsize, pixsize),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reference_adjoint_chunked(vis, uvw, freq, (n_l, n_m), pixsize, pixsize, row_chunk=4),
        _reference_adjoint(vis, uvw, freq, (n_l, n_m), pixsize, pixsize),
        rtol=1e-12,
        atol=1e-12,
    )


@requires_x64
@pytest.mark.runsweep
@pytest.mark.parametrize("direction", ["forward", "adjoint"])
@pytest.mark.parametrize("eps", SWEEP_EPS, ids=lambda e: f"eps{e:.0e}")
@pytest.mark.parametrize(
    ("telescope", "zenith_angle_deg"),
    SWEEP_FIXTURES,
    ids=[_fixture_id(t, a) for t, a in SWEEP_FIXTURES],
)
def test_accuracy_sweep(
    telescope: Telescope, zenith_angle_deg: float, eps: float, direction: str
) -> None:
    """One cell of the sweep: measured error must be within ``2 * epsilon``."""
    data = _fixture_data(telescope, zenith_angle_deg)
    plan = make_plan(data.uvw, data.freq, data.image_shape, data.pixsize, data.pixsize, eps)

    if direction == "forward":
        got = np.asarray(dirty2vis(plan, jnp.asarray(data.image), w_strategy=SWEEP_STRATEGY))
        want = data.vis_ref
    else:
        got = np.asarray(vis2dirty(plan, jnp.asarray(data.vis), w_strategy=SWEEP_STRATEGY))
        want = data.dirty_ref

    error = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    row = _Row(
        fixture=data.name,
        eps=eps,
        direction=direction,
        error=error,
        n_w=plan.n_w,
        width=plan.w_kernel_width,
    )
    _ROWS.append(row)
    print(row.render())

    assert error <= MAX_ERROR_RATIO * eps, (
        f"{data.name} eps={eps:.0e} {direction}: relative error {error:.3e} is "
        f"{error / eps:.2f}x eps, above the {MAX_ERROR_RATIO:g}x contract "
        f"(W={plan.w_kernel_width}, n_w={plan.n_w})"
    )
