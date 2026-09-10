"""The weighting recipes, and the figures ``docs/weighting.md`` quotes for them.

Two jobs. The first is to pin what the recipes actually do -- the taper's
Fourier pair, Briggs's two limits -- because the document states those as
numbers and AGENTS.md section 11 says a quantitative claim in prose needs a
test behind it.

The second is the one a recipe usually lacks: ``docs/weighting.md`` quotes
:mod:`tests.weighting_recipes` verbatim, and
:func:`test_the_document_quotes_the_module_it_says_it_quotes` fails when the
two drift apart. A code sample that nothing executes is a code sample that
stops working quietly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import MWA_EXTENDED, synthetic_uvw
from tests.weighting_recipes import briggs, gaussian_taper, uv_lambda

_DOC = Path(__file__).resolve().parent.parent / "docs" / "weighting.md"


# --------------------------------------------------------------------------
# The uv taper
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fwhm_px", [3.0, 5.0, 8.0])
def test_the_taper_synthesises_the_beam_width_it_was_asked_for(fwhm_px: float) -> None:
    """Transform the taper and measure the beam, rather than trusting the algebra.

    This is a property of the Fourier pair alone, so it is checked on a bare
    grid rather than through the operators: the uv coverage of a real array
    would convolve in a dirty beam and stop the measurement being about the
    taper. Getting the 4*ln2/(pi*theta) factor wrong -- dropping the 4, or
    confusing FWHM with sigma -- moves the measured width by enough that these
    assertions catch it.
    """
    n_pix, pixsize = 512, np.deg2rad(0.002)
    fwhm_rad = fwhm_px * pixsize

    du = 1.0 / (n_pix * pixsize)
    axis = (np.arange(n_pix) - n_pix // 2) * du
    u_grid, v_grid = np.meshgrid(axis, axis, indexing="ij")

    taper = gaussian_taper(u_grid, v_grid, fwhm_rad)
    beam = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(taper))))
    profile = beam[:, n_pix // 2] / beam.max()

    above_half = np.flatnonzero(profile >= 0.5)
    measured_px = above_half[-1] - above_half[0] + 1
    assert abs(measured_px - fwhm_px) <= 1.0, (
        f"a taper asked for a {fwhm_px} px FWHM beam synthesised {measured_px} px. "
        "docs/weighting.md states the uv factor as "
        "exp(-(u^2+v^2) pi^2 theta^2 / (4 ln 2)); a width this far out means "
        "that pair is wrong."
    )


def test_the_taper_never_increases_a_weight() -> None:
    """It is a down-weighting: 1 at the origin, monotone in baseline length."""
    u = np.linspace(0.0, 5e3, 64)[:, None]
    v = np.zeros_like(u)
    factor = gaussian_taper(u, v, np.deg2rad(0.01))
    assert factor[0, 0] == pytest.approx(1.0)
    assert np.all(np.diff(factor[:, 0]) <= 0.0)
    assert np.all((factor > 0.0) & (factor <= 1.0))


# --------------------------------------------------------------------------
# Briggs
# --------------------------------------------------------------------------


def _fixture_uv():
    tel = MWA_EXTENDED
    uvw = synthetic_uvw(tel, 30.0, seed=0)
    u, v = uv_lambda(uvw, np.array([tel.freq_hz]))
    return tel, u, v, np.ones((tel.n_rows, 1))


def test_briggs_spans_natural_at_plus_two_and_uniform_at_minus_two() -> None:
    """The two limits the robustness parameter is defined by.

    Quantified as the spread max/min rather than absolute weights, because the
    scheme is scale-free: what distinguishes natural from uniform is whether
    crowded cells are suppressed relative to sparse ones, not the overall
    normalisation.
    """
    tel, u, v, w_nat = _fixture_uv()
    spreads = {}
    for robust in (2.0, 0.0, -2.0):
        w = briggs(u, v, w_nat, tel.n_pix, tel.pixsize, robust=robust)
        assert np.all(w > 0.0) and np.all(w <= w_nat), f"robust={robust} left the range"
        spreads[robust] = w.max() / w.min()

    # +2 is natural to within a fraction of a percent: every weight survives.
    assert spreads[2.0] < 1.01, (
        f"robust=+2 spread {spreads[2.0]:.3f}x; docs/weighting.md calls it "
        "indistinguishable from natural weighting."
    )
    # Ordered, and -2 crushes the dense cells by orders of magnitude.
    assert spreads[-2.0] > spreads[0.0] > spreads[2.0]
    assert spreads[-2.0] > 1e3, (
        f"robust=-2 spread only {spreads[-2.0]:.1f}x; it should approach uniform "
        "weighting, where a cell's weight goes as one over its population."
    )


def test_briggs_suppresses_the_crowded_cells_not_arbitrary_ones() -> None:
    """The down-weighting has to track uv density, or it is not Briggs."""
    tel, u, v, w_nat = _fixture_uv()
    w = briggs(u, v, w_nat, tel.n_pix, tel.pixsize, robust=-2.0)

    du = 1.0 / (tel.n_pix * tel.pixsize)
    iu = np.rint(u / du).astype(int) + tel.n_pix // 2
    iv = np.rint(v / du).astype(int) + tel.n_pix // 2
    on_grid = (iu >= 0) & (iu < tel.n_pix) & (iv >= 0) & (iv < tel.n_pix)
    grid = np.zeros((tel.n_pix, tel.n_pix))
    np.add.at(grid, (iu[on_grid], iv[on_grid]), w_nat[on_grid])
    population = grid[iu[on_grid], iv[on_grid]]

    # Rank correlation: denser cell -> smaller weight, monotonically.
    order = np.argsort(population)
    ranked = w[on_grid][order]
    assert np.all(np.diff(ranked) <= 1e-12), (
        "a visibility in a more crowded cell came out with a larger Briggs "
        "weight than one in a sparser cell"
    )


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------


def test_the_document_quotes_the_module_it_says_it_quotes() -> None:
    """docs/weighting.md's python blocks must still be this module's source.

    The usual failure of a documented recipe is silent: the module is fixed,
    the snippet in the prose is not, and a reader copies code that stopped
    working some releases ago. Each function the document shows is compared
    against the real definition.
    """
    assert _DOC.exists(), f"{_DOC} is missing"
    doc = _DOC.read_text()
    source = (Path(__file__).resolve().parent / "weighting_recipes.py").read_text()

    quoted = re.findall(r"^```python\n(.*?)^```", doc, re.M | re.S)
    assert quoted, "docs/weighting.md has no python blocks to check"

    checked = 0
    for block in quoted:
        for match in re.finditer(r"^def (\w+)\(", block, re.M):
            name = match.group(1)
            if f"\ndef {name}(" not in source:
                continue  # illustrative helper, not one of the recipes
            body = block[match.start() :]
            for line in body.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                assert stripped in source, (
                    f"docs/weighting.md shows a line in {name}() that is not in "
                    f"tests/weighting_recipes.py any more:\n    {stripped}\n"
                    "The document and the module have drifted; update the block."
                )
            checked += 1
    assert checked >= 2, f"only {checked} recipe(s) cross-checked against the module"
