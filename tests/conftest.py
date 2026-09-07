"""Shared pytest fixtures: synthetic telescope uvw + pointings, precision switch.

Each ``Telescope`` describes a synthetic observing setup: the uv distribution
parameters, the image size / FoV, and the central frequency. ``synthetic_uvw``
turns those parameters (plus a chosen pointing) into a ``(n_rows, 3)`` uvw
array in metres, with controllable w-content for both zenith and 30-degree
off-zenith cases.

Precision
---------
The suite runs in float64 by default (see the ``JAX_ENABLE_X64`` block below);
``JAX_ENABLE_X64=0`` in the environment selects the single-precision leg, which
is what a user gets from JAX out of the box. The public handles are:

``X64``
    Module-level ``bool`` recording the resolved setting.
``requires_x64``
    A plain ``pytest.mark.skipif`` marker for float64-only tests; import it with
    ``from tests.conftest import requires_x64``.
``tol(f64, f32)``
    Returns ``f64`` under x64 and ``f32`` otherwise, so precision-scaled
    tolerances are written once here instead of being reinvented per module.
``precision`` / ``real_dtype`` / ``complex_dtype``
    Session-scoped fixtures giving the active precision as a string
    (``"float64"`` / ``"float32"``) and as the matching ``jnp`` dtypes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

# --- precision switch (issue #11) -------------------------------------------
# Default float64 so the ducc/DFT parity tests keep their headroom at eps=1e-8;
# ``JAX_ENABLE_X64=0`` (also ``false`` / ``False``) selects the float32 leg.
# This must run before any JAX array is created, hence its position at the top
# of the file.
X64: bool = os.environ.get("JAX_ENABLE_X64", "1").strip().lower() not in ("0", "false")
jax.config.update("jax_enable_x64", X64)

# A marker object, not a collection hook: test modules apply it per test with
# ``@requires_x64`` so a float32 run reports SKIPPED rather than FAILED.
requires_x64 = pytest.mark.skipif(
    not X64,
    reason="needs jax_enable_x64 (this run has JAX_ENABLE_X64=0, i.e. float32)",
)


def tol(f64: float, f32: float) -> float:
    """Pick a tolerance for the active precision: ``f64`` under x64, else ``f32``."""
    return f64 if X64 else f32


@pytest.fixture(scope="session")
def precision() -> str:
    """``"float64"`` or ``"float32"`` — the precision this run was configured with."""
    return "float64" if X64 else "float32"


@pytest.fixture(scope="session")
def real_dtype() -> DTypeLike:
    """The real dtype matching the active precision."""
    return jnp.float64 if X64 else jnp.float32


@pytest.fixture(scope="session")
def complex_dtype() -> DTypeLike:
    """The complex dtype matching the active precision."""
    return jnp.complex128 if X64 else jnp.complex64


# --- what the float32 leg covers today (issue #11) --------------------------
# This is an explicit BACKLOG, not a design. Under ``JAX_ENABLE_X64=0`` the
# modules listed below are skipped at collection time because they are not
# precision-aware yet: each either builds a default (float64) plan, which
# ``make_plan`` now refuses with a ``ValueError`` when x64 is off, or calls
# ``jax.config.update("jax_enable_x64", True)`` at import time, which would
# silently turn the whole "float32" run back into a float64 one and make the
# leg meaningless.
#
# The list was determined empirically (run the leg, see what breaks), not by
# guessing, and it is expected to SHRINK: later issues in the 2026-09 review
# plan parametrise these modules over precision -- using the ``precision`` /
# ``real_dtype`` / ``complex_dtype`` fixtures and ``tol(f64, f32)`` above
# instead of hard-coding float64 -- and drop them from here one at a time, one
# module per issue, until the list is empty. Nothing should ever be added.
#
# ``tests/test_dtype.py`` is deliberately absent: it is the module this leg
# exists to run today. So is every module that is already precision-agnostic
# (pure-host kernel/plan structure, strategy selection, benchmark harness).
collect_ignore: list[str] = []
if not X64:
    collect_ignore = [
        # Call ``jax.config.update("jax_enable_x64", True)`` at import time, so
        # merely collecting them turns the float32 run back into a float64 one
        # for every module that follows. (They also assert DFT / ducc parity
        # down to eps=1e-8, which single precision cannot reach.)
        "test_adjoint.py",
        "test_against_dft.py",
        "test_benchmark_against_ducc.py",
        "test_boundary_planes.py",
        "test_jax_integration.py",
        # Build default (float64) plans with no ``dtype=`` argument: 62 tests
        # across these four now stop at the new ``make_plan`` ValueError.
        "test_against_ducc.py",
        "test_auto_strategy.py",
        "test_constant_w.py",
        # Both: it builds default (float64) plans, and it imports the DFT
        # references from ``test_adjoint`` / ``test_against_dft``, whose
        # import-time ``jax.config.update("jax_enable_x64", True)`` would
        # switch this leg back to float64 for everything collected after it.
        "test_accuracy_sweep.py",
        "test_planning.py",
    ]


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
# v0.1.2 Part 5.4: a GH200-class fixture for the GPU bench suite. Sized so
# the transient dense_vmap allocation (n_w * n_pix^2 complex64) is on the
# order of ~10-20 GB -- big enough to demand real HBM bandwidth, far below
# the GH200's 96 GB so we don't OOM. Off-zenith pointing produces ~n_w in
# the low hundreds with these parameters.
GH200_LARGE = Telescope(
    name="GH200_large",
    freq_hz=1.4e9,
    n_rows=50_000,
    sigma_uv_m=4000.0,
    max_baseline_m=12_000.0,
    n_pix=2048,
    fov_rad=math.radians(2.0),
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


# --- clumped w-distributions (issue #15) ------------------------------------
# ``synthetic_uvw`` produces a *symmetric, unimodal* w-distribution at every
# pointing: at zenith w is the Gaussian ``z`` offset, and off-zenith it is a
# rotation that mixes a Gaussian ``u`` into it. Every telescope fixture in this
# repository -- and therefore every ducc0- and DFT-parity assertion in it --
# holds the shape of the w-distribution constant at "Gaussian". That is an axis
# held constant, and it is precisely the axis the windowed strategies exist to
# exploit: their whole premise is that the rows contributing to a plane are a
# contiguous slice of the w-sorted array, which is interesting exactly when the
# slices are wildly unequal.
#
# ``tests/test_boundary_planes.py`` does build clumped w-distributions, but it
# only ever compares the windowed path against the dense one on the same plan.
# That is a reduction-order comparison between two spellings of the same
# operator: an error the two share -- a mis-placed plane centre, a plane grid
# that under-resolves a clump -- cancels out of it and passes cleanly, which is
# the shared-mode failure mode AGENTS.md section 6 records from issue #16. It
# also runs on a toy geometry (32^2, uniform (u, v), one channel), not on a
# telescope fixture. So nothing in the suite has ever held a clumped
# w-distribution against an *external* oracle.
#
# ``clumped_track`` closes that gap. The (u, v) columns are taken verbatim from
# ``synthetic_uvw(telescope, 0.0, seed)``, so the *only* thing that differs
# from the zenith fixture of the same telescope is the w column -- which is
# what makes a failure attributable to the w-distribution rather than to a
# second, incidentally different geometry.
#
# The distribution is the one issue #15 asks for and the one a short
# hour-angle-blocked track produces: two dense clumps at +/- ``clump_offset``
# of the array's longest baseline, plus a sparse uniform tail that sets the
# w-extent (and hence the plane count) while populating almost none of the
# planes between the clumps. Measured against the matching ``synthetic_uvw``
# off30 plan (float64, eps=1e-6, shipped ``hermitian=True`` default):
#
#     fixture           n_w   empty planes   window_padding_overhead
#     EDA2 clumped       81        13                 11.25
#     EDA2 off30         56         0                  2.52
#     MWA_extended clu. 214        90                 29.50
#     MWA_extended off30 134       18                  4.94
#
# It is a w-distribution stress fixture, not a physically simulated track: as
# in ``tests/test_boundary_planes.py`` the w column is drawn independently of
# (u, v). |w| stays under the array's longest baseline, which is the only
# physical bound that matters for the plane grid.
_CLUMP_OFFSET_FRAC = 0.4
_CLUMP_WIDTH_FRAC = 0.002
_CLUMP_TAIL_FRACTION = 0.04
_CLUMP_TAIL_SPAN_FRAC = 0.9


def clumped_track(
    telescope: Telescope,
    seed: int,
    *,
    clump_offset_frac: float = _CLUMP_OFFSET_FRAC,
    clump_width_frac: float = _CLUMP_WIDTH_FRAC,
    tail_fraction: float = _CLUMP_TAIL_FRACTION,
    tail_span_frac: float = _CLUMP_TAIL_SPAN_FRAC,
) -> np.ndarray:
    """``(n_rows, 3)`` uvw in metres with a bimodal, heavily clumped w column.

    The ``(u, v)`` columns are exactly ``synthetic_uvw(telescope, 0.0, seed)``'s;
    only ``w`` differs, and it is drawn as

      * ``1 - tail_fraction`` of the rows split evenly between two narrow
        Gaussian clumps at ``+/- clump_offset_frac * max_baseline_m``, with
        standard deviation ``clump_width_frac * max_baseline_m``;
      * the remaining ``tail_fraction`` uniform over
        ``+/- tail_span_frac * max_baseline_m`` -- the sparse tail that sets
        the w-extent, and hence ``n_w``, while leaving the planes between the
        clumps almost or entirely empty.

    The result is shuffled, so w is uncorrelated with row index and the sort
    permutation is a genuine reordering rather than a near-identity.

    A dedicated RNG stream (``seed + 977``) draws the w column, so the ``(u, v)``
    columns are bit-identical to the zenith fixture built from the same seed.
    """
    uvw = synthetic_uvw(telescope, 0.0, seed=seed).copy()
    rng = np.random.default_rng(seed + 977)
    n_rows = uvw.shape[0]
    baseline = telescope.max_baseline_m

    n_tail = round(tail_fraction * n_rows)
    n_clump = n_rows - n_tail
    n_low = n_clump // 2

    w = np.empty(n_rows)
    w[:n_low] = rng.normal(
        loc=-clump_offset_frac * baseline, scale=clump_width_frac * baseline, size=n_low
    )
    w[n_low:n_clump] = rng.normal(
        loc=+clump_offset_frac * baseline,
        scale=clump_width_frac * baseline,
        size=n_clump - n_low,
    )
    w[n_clump:] = rng.uniform(-tail_span_frac * baseline, tail_span_frac * baseline, size=n_tail)
    uvw[:, 2] = rng.permutation(w)
    return uvw


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


# Which telescopes the clumped generator is worth running on. Clumping the w
# column only reaches the *plan* when the plane count is well above the kernel
# width: below that every window already spans essentially every row and the
# distribution's shape stops mattering. Measured at eps=1e-6, float64, shipped
# ``hermitian=True``, ``clumped_track`` vs ``synthetic_uvw(tel, 30.0)``:
#
#     telescope       n_w clumped / off30   padding overhead clumped / off30
#     EDA2                  81 / 56                 11.25 / 2.52
#     MWA_extended         214 / 134                29.50 / 4.94
#     MWA_compact           15 / 12                  2.14 / 1.71
#     MeerKAT               17 / 13                  1.86 / 1.86
#
# EDA2 gets there on a 120-degree field (large ``max|n-1|``) and MWA_extended on
# 5.3 km baselines (large w-extent in wavelengths); MWA_compact and MeerKAT have
# neither and sit at ``n_w`` under ``2 * w_kernel_width`` however the w column is
# drawn. Running them here would add cells that cannot fail for the reason this
# fixture exists, and would force the "is it really clumped" guard in
# ``tests/test_clumped_track.py`` down to a threshold that no longer says
# anything. ``tests/test_boundary_planes.py`` keeps the small-``n_w`` clumped
# cases.
_CLUMPED_SHORT_TELESCOPES = [EDA2]
_CLUMPED_LONG_TELESCOPES = [MWA_EXTENDED]


@pytest.fixture(params=_CLUMPED_SHORT_TELESCOPES, ids=lambda t: f"{t.name}_clumped")
def clumped_track_telescope(request) -> Telescope:
    """A short telescope whose uvw comes from :func:`clumped_track` (issue #15).

    Deliberately *not* a ``(telescope, angle)`` pair: the clumped fixture has no
    pointing -- it replaces the w column outright -- so a pointing parameter
    would produce identical data twice.
    """
    return request.param


@pytest.fixture(params=_CLUMPED_LONG_TELESCOPES, ids=lambda t: f"{t.name}_clumped")
def long_clumped_track_telescope(request) -> Telescope:
    """The 256-pixel clumped fixture; gated behind ``--runslow`` like
    ``long_telescope_pointing``."""
    return request.param


# All four telescopes for benchmarking, both pointings. The
# ``--bench-pointing`` flag (default ``zenith``) controls which subset is
# actually run; the full param list lives here so pytest's collection logic
# can attach proper IDs even when only a subset is selected.
@pytest.fixture(
    params=[
        (EDA2, 0.0),
        (EDA2, 30.0),
        (MWA_COMPACT, 0.0),
        (MWA_COMPACT, 30.0),
        (MWA_EXTENDED, 0.0),
        (MWA_EXTENDED, 30.0),
        (MEERKAT, 0.0),
        (MEERKAT, 30.0),
    ],
    ids=lambda v: _telescope_pointing_id(v),
)
def bench_telescope_pointing(request) -> tuple[Telescope, float]:
    return request.param


@pytest.fixture(
    params=[
        (GH200_LARGE, 0.0),
        (GH200_LARGE, 30.0),
    ],
    ids=lambda v: _telescope_pointing_id(v),
)
def gh200_large_pointing(request) -> tuple[Telescope, float]:
    """GH200-sized fixture for the v0.1.2 GPU bench suite. Only used by
    ``tests/test_benchmark_gpu.py``; gated by ``--runbench-gpu`` and a GPU
    backend so accidental CPU collection doesn't try to allocate the
    multi-GB transient arrays."""
    return request.param


_BENCH_POINTING_FILTERS: dict[str, set[float]] = {
    "zenith": {0.0},
    "off30": {30.0},
    "both": {0.0, 30.0},
}


def _jax_platform() -> str:
    """Detect the JAX default platform without importing at conftest top.

    Used to gate ``--runbench-gpu`` tests so they skip cleanly on CPU
    machines instead of failing inside cuFINUFFT.
    """
    try:
        import jax
    except Exception:  # pragma: no cover
        return "cpu"
    return jax.default_backend()


def pytest_collection_modifyitems(config, items):
    """Mark slow / benchmark tests so they are skipped without their flag."""
    skip_slow = pytest.mark.skip(reason="needs --runslow")
    skip_sweep = pytest.mark.skip(reason="needs --runsweep")
    skip_bench = pytest.mark.skip(reason="needs --runbench")
    skip_bench_gpu_flag = pytest.mark.skip(reason="needs --runbench-gpu")
    skip_bench_gpu_platform = pytest.mark.skip(
        reason="runbench_gpu requires jax.default_backend() == 'gpu'"
    )
    skip_timing = pytest.mark.skip(reason="needs --runtiming")
    runslow = config.getoption("--runslow", default=False)
    runsweep = config.getoption("--runsweep", default=False)
    runbench = config.getoption("--runbench", default=False)
    runbench_gpu = config.getoption("--runbench-gpu", default=False)
    runtiming = config.getoption("--runtiming", default=False)
    bench_pointing = config.getoption("--bench-pointing", default="zenith")
    allowed_angles = _BENCH_POINTING_FILTERS[bench_pointing]
    skip_off_pointing = pytest.mark.skip(
        reason=f"--bench-pointing={bench_pointing} excludes this combination"
    )
    platform = _jax_platform()
    # Fixtures whose presence means "this item is a --runslow item".
    needs_slow = {"long_telescope_pointing", "long_clumped_track_telescope"}
    for item in items:
        is_bench_item = "bench_telescope_pointing" in item.fixturenames
        is_runbench_gpu = "runbench_gpu" in item.keywords
        # The accuracy sweep is its own opt-in axis: it is neither a parity
        # test nor a benchmark, and it is expensive (7 fixtures x 8 epsilon
        # x 2 directions, each with an exact-DFT reference). Gate it first
        # and skip the rest of the checks for those items.
        if "runsweep" in item.keywords and not runsweep:
            item.add_marker(skip_sweep)
            continue
        # The nthreads timing gate (issue #24) is wall-clock-based and thus
        # flaky on shared/noisy CI runners, same rationale as --runbench;
        # it gets its own flag rather than piggybacking on --runbench so it
        # can be run in isolation without the full (slower) benchmark suite.
        if "runtiming" in item.keywords and not runtiming:
            item.add_marker(skip_timing)
            continue
        if needs_slow & set(item.fixturenames) and not runslow:
            item.add_marker(skip_slow)
        if is_runbench_gpu:
            if not runbench_gpu:
                item.add_marker(skip_bench_gpu_flag)
                continue
            if platform != "gpu":
                item.add_marker(skip_bench_gpu_platform)
                continue
            # --runbench-gpu tests are gated only by their own flag +
            # platform; don't apply --runbench gating below.
        elif is_bench_item and not runbench:
            item.add_marker(skip_bench)
            continue
        if is_bench_item:
            tel_pointing = item.callspec.params.get("bench_telescope_pointing")
            if tel_pointing is not None and tel_pointing[1] not in allowed_angles:
                item.add_marker(skip_off_pointing)


def pytest_configure(config):
    """Register custom markers so ``pytest -m`` is happy and PYTHONDEVMODE
    doesn't print a warning."""
    config.addinivalue_line(
        "markers",
        "runbench_gpu: opt-in GPU benchmark suite "
        "(needs --runbench-gpu and jax.default_backend() == 'gpu')",
    )
    config.addinivalue_line(
        "markers",
        "runsweep: opt-in exact-DFT accuracy sweep (needs --runsweep)",
    )
    config.addinivalue_line(
        "markers",
        "runtiming: opt-in nthreads timing gate, issue #24 (needs --runtiming)",
    )


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
    parser.addoption(
        "--runbench-gpu",
        action="store_true",
        default=False,
        help=(
            "Run the v0.1.2 GPU benchmark suite (tests/test_benchmark_gpu.py). "
            "Tests are also gated on jax.default_backend() == 'gpu' so a CPU "
            "host produces SKIPPED, not FAILED."
        ),
    )
    parser.addoption(
        "--bench-pointing",
        choices=("zenith", "off30", "both"),
        default="zenith",
        help=(
            "Which pointings to include in the benchmark suite. "
            "'zenith' is the default; 'off30' adds w-extent and roughly doubles n_w; "
            "'both' runs each telescope twice."
        ),
    )
    parser.addoption(
        "--runsweep",
        action="store_true",
        default=False,
        help=(
            "Run the exact-DFT accuracy sweep (tests/test_accuracy_sweep.py): the "
            "seven review fixtures x eight epsilon values x forward/adjoint. Every "
            "issue that says 're-run the accuracy sweep' means this flag. Add -s to "
            "see the printed ratio table."
        ),
    )
    parser.addoption(
        "--runtiming",
        action="store_true",
        default=False,
        help=(
            "Run the issue #24 nthreads timing gate (tests/test_timing_nthreads.py): "
            "asserts dense_scan's default nthreads is within 1.2x of an explicit "
            "nthreads=1 on the same machine. Wall-clock based, so it is opt-in and "
            "excluded from the default suite the same way --runbench is."
        ),
    )
