"""Shared pytest fixtures: synthetic telescope uvw + pointings.

Each ``Telescope`` describes a synthetic observing setup: the uv distribution
parameters, the image size / FoV, and the central frequency. ``synthetic_uvw``
turns those parameters (plus a chosen pointing) into a ``(n_rows, 3)`` uvw
array in metres, with controllable w-content for both zenith and 30-degree
off-zenith cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import numpy as np
import pytest

# Switch x64 on globally so the parity tests have headroom at eps=1e-8.
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Telescope:
    name: str
    freq_hz: float
    n_rows: int
    sigma_uv_m: float
    max_baseline_m: float
    n_pix: int
    fov_rad: float

    @property
    def pixsize(self) -> float:
        return self.fov_rad / self.n_pix


# Smaller image sizes than the spec (which lists 256/512/1024) so tests stay
# CI-friendly. The algorithmic regime (large baselines, off-zenith pointing,
# wide FoV for low-freq instruments) is preserved; only the pixel count and
# ``n_rows`` are reduced. Production accuracy at full size is the concern of
# downstream benchmarks, not unit tests.
EDA2 = Telescope(
    name="EDA2",
    freq_hz=200e6,
    n_rows=400,
    sigma_uv_m=12.0,
    max_baseline_m=35.0,
    n_pix=64,
    fov_rad=math.radians(120.0),  # full-sky-ish
)
MWA_COMPACT = Telescope(
    name="MWA_compact",
    freq_hz=150e6,
    n_rows=600,
    sigma_uv_m=50.0,
    max_baseline_m=200.0,
    n_pix=128,
    fov_rad=math.radians(25.0),
)
MWA_EXTENDED = Telescope(
    name="MWA_extended",
    freq_hz=150e6,
    n_rows=600,
    sigma_uv_m=800.0,
    max_baseline_m=5300.0,
    n_pix=256,
    fov_rad=math.radians(25.0),
)
MEERKAT = Telescope(
    name="MeerKAT",
    freq_hz=1.28e9,
    n_rows=600,
    sigma_uv_m=2000.0,
    max_baseline_m=8000.0,
    n_pix=256,
    fov_rad=math.radians(1.5),
)


def synthetic_uvw(
    telescope: Telescope,
    zenith_angle_deg: float,
    seed: int,
) -> np.ndarray:
    """Generate ``(n_rows, 3)`` uvw in metres for ``telescope`` at the given pointing.

    Strategy:
      * ``(u, v)`` is drawn from a 2D Gaussian with ``sigma = sigma_uv_m`` and
        truncated to the max baseline.
      * A small ``z`` antenna offset (~3 % of sigma_uv_m) is added so the
        zenith case still has *some* w-extent; this keeps every code path
        (including the w-direction kernel) exercised.
      * For a non-zero zenith angle, a tilt rotation is applied that mixes the
        u-direction baseline length into w. At 30 degrees this puts roughly
        ``sin(30) ~ 0.5`` of the u-baseline into w.
    """
    rng = np.random.default_rng(seed)
    sigma = telescope.sigma_uv_m
    n_rows = telescope.n_rows

    # Bivariate Gaussian in u, v.
    uv = rng.normal(scale=sigma, size=(n_rows, 2))
    radii = np.linalg.norm(uv, axis=1, keepdims=True)
    # Sparse outer component: pull a few baselines towards max_baseline_m so the
    # tail of the distribution actually reaches it.
    outer_n = max(1, n_rows // 25)
    outer_idx = rng.choice(n_rows, size=outer_n, replace=False)
    direction = uv[outer_idx] / np.maximum(
        np.linalg.norm(uv[outer_idx], axis=1, keepdims=True), 1e-9
    )
    uv[outer_idx] = direction * telescope.max_baseline_m
    # Soft-truncate the rest at max_baseline_m.
    radii = np.linalg.norm(uv, axis=1, keepdims=True)
    uv = uv * np.minimum(1.0, telescope.max_baseline_m / np.maximum(radii, 1e-9))

    z = rng.normal(scale=sigma * 0.03, size=n_rows)

    if zenith_angle_deg == 0.0:
        return np.column_stack([uv[:, 0], uv[:, 1], z])

    theta = math.radians(zenith_angle_deg)
    u_in = uv[:, 0]
    u_new = u_in * math.cos(theta) - z * math.sin(theta)
    z_new = u_in * math.sin(theta) + z * math.cos(theta)
    return np.column_stack([u_new, uv[:, 1], z_new])


_SHORT_TELESCOPES = [EDA2, MWA_COMPACT]
_LONG_TELESCOPES = [MWA_EXTENDED, MEERKAT]


def _telescope_pointing_id(values):
    tel, ang = values
    return f"{tel.name}_zenith" if ang == 0 else f"{tel.name}_off{int(ang)}"


@pytest.fixture(
    params=[
        (EDA2, 0.0),
        (MWA_COMPACT, 0.0),
        (MWA_COMPACT, 30.0),
    ],
    ids=lambda v: _telescope_pointing_id(v),
)
def short_telescope_pointing(request) -> tuple[Telescope, float]:
    return request.param


@pytest.fixture(
    params=[
        (MWA_EXTENDED, 0.0),
        (MWA_EXTENDED, 30.0),
        (MEERKAT, 0.0),
        (MEERKAT, 30.0),
    ],
    ids=lambda v: _telescope_pointing_id(v),
)
def long_telescope_pointing(request) -> tuple[Telescope, float]:
    return request.param


# All four telescopes (zenith only) for benchmarking. Off-zenith roughly
# doubles n_w; running both pointings would double the bench duration
# without changing the qualitative ranking.
@pytest.fixture(
    params=[
        (EDA2, 0.0),
        (MWA_COMPACT, 0.0),
        (MWA_EXTENDED, 0.0),
        (MEERKAT, 0.0),
    ],
    ids=lambda v: _telescope_pointing_id(v),
)
def bench_telescope_pointing(request) -> tuple[Telescope, float]:
    return request.param


def pytest_collection_modifyitems(config, items):
    """Mark slow / benchmark tests so they are skipped without their flag."""
    skip_slow = pytest.mark.skip(reason="needs --runslow")
    skip_bench = pytest.mark.skip(reason="needs --runbench")
    runslow = config.getoption("--runslow", default=False)
    runbench = config.getoption("--runbench", default=False)
    for item in items:
        if "long_telescope_pointing" in item.fixturenames and not runslow:
            item.add_marker(skip_slow)
        if "bench_telescope_pointing" in item.fixturenames and not runbench:
            item.add_marker(skip_bench)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow telescope parity tests (MWA_extended, MeerKAT).",
    )
    parser.addoption(
        "--runbench",
        action="store_true",
        default=False,
        help="Run benchmark suite comparing jax-nufft to ducc0.",
    )
