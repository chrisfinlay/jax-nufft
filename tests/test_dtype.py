"""Plan/array dtype contract: the x64-off guard and mixed-precision inputs (issue #11).

Two legs:

* **x64 off.** ``jax_enable_x64`` is process-global and ``tests/conftest.py``
  turns it on for the suite, so the single-precision path is exercised in a
  *child interpreter* launched with ``JAX_ENABLE_X64=0``. The child must not
  import ``tests.conftest`` (that would flip the setting back on); it gets its
  fixture data through an ``.npz`` file written by the parent and reports back
  as JSON on stdout. Requirements checked there: the default (float64) plan is
  refused with a clear ``ValueError``; an explicit ``dtype=jnp.float32`` plan
  tracks ducc0 at ``epsilon=1e-4``; and asking for ``epsilon=1e-6`` in float32
  warns about the achievable floor instead of silently missing it.
* **x64 on.** Inputs narrower than the plan are cast (and agree with the
  all-float64 answer), inputs wider than the plan raise ``TypeError`` naming
  both dtypes, and in no case does jax-finufft's bare ``AssertionError()``
  escape to the user.

The oracle is ducc0's public Python API only (``dirty2vis`` / ``vis2dirty``),
used as a black box exactly as in ``tests/test_against_ducc.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import ducc0.wgridder
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft.planning import WGridderPlan
from tests.conftest import (
    EDA2,
    MEERKAT,
    MWA_COMPACT,
    MWA_EXTENDED,
    Telescope,
    requires_x64,
    synthetic_uvw,
    tol,
)

# EDA2 zenith: 400 rows, 64x64 pixels, 120-degree FoV. The smallest review
# fixture, so the whole module (including one child interpreter) stays well
# under a minute.
_SEED = 0
_IMAGE_SEED = 7
_VIS_SEED = 11
_WEIGHT_SEED = 13


def _fixture_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """``(uvw, freq, image, vis, pixsize)`` for the EDA2 zenith fixture."""
    uvw = synthetic_uvw(EDA2, 0.0, seed=_SEED)
    freq = np.array([EDA2.freq_hz])
    image = np.random.default_rng(_IMAGE_SEED).standard_normal((EDA2.n_pix, EDA2.n_pix))
    rng = np.random.default_rng(_VIS_SEED)
    vis = rng.standard_normal((EDA2.n_rows, 1)) + 1j * rng.standard_normal((EDA2.n_rows, 1))
    return uvw, freq, image, vis, EDA2.pixsize


def _weights() -> np.ndarray:
    """Strictly positive per-visibility weights, ``(n_rows, 1)`` real."""
    rng = np.random.default_rng(_WEIGHT_SEED)
    return rng.uniform(0.5, 2.0, size=(EDA2.n_rows, 1))


# The three off30 review fixtures (issue #15). Everything else in this module
# runs on EDA2 at *zenith*, whose float32 plan is nine w-planes deep; these put
# the x64-off leg at 10, 132 and 11 planes. The w-plane loop is the only part of
# the operator whose cost and error both scale with the plane count, and until
# these landed no float32 run in this repository had ever taken more than nine
# passes through it.
_OFF30_TELESCOPES: tuple[Telescope, ...] = (MWA_COMPACT, MWA_EXTENDED, MEERKAT)

# Measured float32 relative error against the double-precision ducc0 oracle at
# eps=1e-4 (macOS arm64, jax 0.9.2, ducc0 0.41.0; uvw seed 0, image seed 7,
# vis seed 11, weights seed 13; shipped ``hermitian=True`` and the default
# ``w_strategy="auto"``), forward / adjoint:
#
#     MWA_compact off30   n_w=10    0.50x eps / 0.54x eps
#     MWA_extended off30  n_w=132   0.73x eps / 0.70x eps
#     MeerKAT off30       n_w=11    0.57x eps / 0.58x eps
#
# So the repo's 3*eps ducc0 contract holds on the float32 leg on these
# fixtures, with the same headroom the zenith probe has; it is asserted at 3
# rather than loosened.
_OFF30_EPS = 1e-4


def _off30_arrays(tel: Telescope) -> dict[str, np.ndarray]:
    """Fixture arrays for ``tel`` at 30 degrees off zenith, ready for ``np.savez``."""
    rng = np.random.default_rng(_IMAGE_SEED)
    vis_rng = np.random.default_rng(_VIS_SEED)
    wgt_rng = np.random.default_rng(_WEIGHT_SEED)
    return {
        "name": np.asarray(f"{tel.name}_off30"),
        "uvw": synthetic_uvw(tel, 30.0, seed=_SEED),
        "freq": np.array([tel.freq_hz]),
        "image": rng.standard_normal((tel.n_pix, tel.n_pix)),
        "vis": vis_rng.standard_normal((tel.n_rows, 1))
        + 1j * vis_rng.standard_normal((tel.n_rows, 1)),
        "weights": wgt_rng.uniform(0.5, 2.0, size=(tel.n_rows, 1)),
        "pixsize": np.asarray(tel.pixsize),
    }


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error ``norm(a - b) / norm(b)``."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _dtype_is(value: Any, expected: Any) -> bool:
    """Compare dtype metadata regardless of scalar-type vs ``np.dtype`` form."""
    return np.dtype(value) == np.dtype(expected)


# ---------------------------------------------------------------------------
# Leg 0: the conftest precision infrastructure describes reality
# ---------------------------------------------------------------------------


def test_precision_fixtures_match_the_active_configuration(
    precision: str, real_dtype: Any, complex_dtype: Any
) -> None:
    """Self-check for the conftest precision switch that later issues build on."""
    expected = "float64" if jax.config.jax_enable_x64 else "float32"
    assert precision == expected
    assert _dtype_is(real_dtype, jnp.float64 if expected == "float64" else jnp.float32)
    assert _dtype_is(complex_dtype, jnp.complex128 if expected == "float64" else jnp.complex64)
    # ``tol`` hands out the f64 value under x64 and the f32 value otherwise.
    assert tol(2.0, 1.0) == (2.0 if expected == "float64" else 1.0)


# ---------------------------------------------------------------------------
# Leg 1: jax_enable_x64 off (child interpreter)
# ---------------------------------------------------------------------------

# NOTE: kept deliberately dependency-light and free of any ``tests.*`` import.
_CHILD_SCRIPT = '''
"""Probe the wgridder dtype contract with jax_enable_x64 off.

Run as ``python child.py eda2.npz [off30.npz ...]`` with ``JAX_ENABLE_X64=0``
in the environment. The first npz drives the dtype-contract probes (a)-(d);
every further npz is an extra geometry probed for ducc0 parity only, keyed by
the ``name`` it carries. Emits one JSON object on stdout, prefixed by a marker
line.
"""

import json
import sys
import warnings

import ducc0.wgridder
import jax
import jax.numpy as jnp
import numpy as np

from jax_nufft import dirty2vis, make_plan, vis2dirty

data = np.load(sys.argv[1])
uvw = data["uvw"]
freq = data["freq"]
image = data["image"]
vis = data["vis"]
weights = data["weights"]
pixsize = float(data["pixsize"])
n_pix = int(image.shape[0])

out = {
    "x64": bool(jax.config.jax_enable_x64),
    "float32_error": None,
    "adjoint_error": None,
    "warn_probe_error": None,
}


def record(caught):
    return [{"category": w.category.__name__, "message": str(w.message)} for w in caught]


def describe(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


# (a) The default (float64) plan must refuse to build silently in float32.
try:
    make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, 1e-4)
except Exception as exc:
    out["default_error"] = describe(exc)
else:
    out["default_error"] = None

# (b) An explicit float32 plan at eps=1e-4 must work, warn about nothing, and
#     track the ducc0 oracle (which runs in double). Every step is guarded so a
#     missing feature is reported as data rather than killing the run: (a) and
#     (c) still get their own verdicts.
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, 1e-4, dtype=jnp.float32)
    out["warnings_at_1e_4"] = record(caught)
    out["plan_leaf_dtypes"] = sorted({str(leaf.dtype) for leaf in jax.tree_util.tree_leaves(plan)})
    out["plan_real_dtype"] = str(np.dtype(plan.real_dtype))
    out["plan_complex_dtype"] = str(np.dtype(plan.complex_dtype))

    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image, dtype=jnp.float32)))
    out["vis_dtype"] = str(vis_jax.dtype)
    vis_ducc = ducc0.wgridder.dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=image,
        pixsize_x=pixsize,
        pixsize_y=pixsize,
        epsilon=1e-4,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )
    out["rel_err_1e_4"] = float(np.linalg.norm(vis_jax - vis_ducc) / np.linalg.norm(vis_ducc))
except Exception as exc:
    out["float32_error"] = describe(exc)

# (b2) The adjoint has to work in float32 too, weights included -- otherwise
#      the x64-off CI leg would stay green with vis2dirty completely broken.
#      Same plan, same epsilon, same 3*eps bound against the double oracle.
try:
    plan = make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, 1e-4, dtype=jnp.float32)
    dirty_jax = np.asarray(
        vis2dirty(
            plan,
            jnp.asarray(vis, dtype=jnp.complex64),
            weights=jnp.asarray(weights, dtype=jnp.float32),
        )
    )[0]
    out["dirty_dtype"] = str(dirty_jax.dtype)
    dirty_ducc = ducc0.wgridder.vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=vis,
        wgt=weights,
        npix_x=n_pix,
        npix_y=n_pix,
        pixsize_x=pixsize,
        pixsize_y=pixsize,
        epsilon=1e-4,
        do_wgridding=True,
        divide_by_n=True,
        nthreads=1,
    )
    out["adjoint_rel_err_1e_4"] = float(
        np.linalg.norm(dirty_jax - dirty_ducc) / np.linalg.norm(dirty_ducc)
    )
except Exception as exc:
    out["adjoint_error"] = describe(exc)

# (c) Asking float32 for eps=1e-6 is below the achievable floor: warn.
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, 1e-6, dtype=jnp.float32)
    out["warnings_at_1e_6"] = record(caught)
except Exception as exc:
    out["warn_probe_error"] = describe(exc)

# (d) A RAW NUMPY float64 2-D image into a float32 plan must still be refused.
#     This case only exists with x64 off: `jnp.broadcast_to` is where a numpy
#     array enters JAX, and that entry silently narrows float64 to float32, so
#     broadcasting before the dtype check would make the 2-D path accept an
#     image the 3-D path rejects.
try:
    plan = make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, 1e-4, dtype=jnp.float32)
    result = dirty2vis(plan, np.asarray(image, dtype=np.float64))
except Exception as exc:
    out["wide_numpy_image_error"] = describe(exc)
else:
    out["wide_numpy_image_error"] = None
    out["wide_numpy_image_dtype"] = str(np.asarray(result).dtype)

# (e) Off-zenith geometries (issue #15). Everything above runs on EDA2 at
#     zenith, whose float32 plan is NINE w-planes deep -- so the entire
#     single-precision CI leg has been exercising the w-plane machinery at a
#     plane count the dense and windowed traversals cannot be told apart at.
#     The three off30 review fixtures put it at 10, 132 and 11 planes, and
#     MWA_extended off30 is the one that matters: 132 float32 planes, each
#     accumulating into the same grid. Forward and adjoint, against the double
#     oracle at the same 3*eps as the zenith probe.
out["off30"] = {}
for path in sys.argv[2:]:
    d = np.load(path)
    name = str(d["name"])
    entry = {"error": None}
    out["off30"][name] = entry
    try:
        u = d["uvw"]
        f = d["freq"]
        img = d["image"]
        v = d["vis"]
        wgt = d["weights"]
        px = float(d["pixsize"])
        npx = int(img.shape[0])
        pl = make_plan(u, f, (npx, npx), px, px, 1e-4, dtype=jnp.float32)
        entry["n_w"] = int(pl.n_w)
        entry["w_kernel_width"] = int(pl.w_kernel_width)
        entry["real_dtype"] = str(np.dtype(pl.real_dtype))

        vj = np.asarray(dirty2vis(pl, jnp.asarray(img, dtype=jnp.float32)))
        entry["vis_dtype"] = str(vj.dtype)
        vd = ducc0.wgridder.dirty2vis(
            uvw=u,
            freq=f,
            dirty=img,
            pixsize_x=px,
            pixsize_y=px,
            epsilon=1e-4,
            do_wgridding=True,
            divide_by_n=False,
            nthreads=1,
        )
        entry["forward_rel_err"] = float(np.linalg.norm(vj - vd) / np.linalg.norm(vd))

        dj = np.asarray(
            vis2dirty(
                pl,
                jnp.asarray(v, dtype=jnp.complex64),
                weights=jnp.asarray(wgt, dtype=jnp.float32),
            )
        )[0]
        entry["dirty_dtype"] = str(dj.dtype)
        dd = ducc0.wgridder.vis2dirty(
            uvw=u,
            freq=f,
            vis=v,
            wgt=wgt,
            npix_x=npx,
            npix_y=npx,
            pixsize_x=px,
            pixsize_y=px,
            epsilon=1e-4,
            do_wgridding=True,
            divide_by_n=True,
            nthreads=1,
        )
        entry["adjoint_rel_err"] = float(np.linalg.norm(dj - dd) / np.linalg.norm(dd))
    except Exception as exc:
        entry["error"] = describe(exc)

print("<<<JSON>>>" + json.dumps(out))
'''

_JSON_MARKER = "<<<JSON>>>"


@pytest.fixture(scope="module")
def x64_off_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run ``_CHILD_SCRIPT`` once with ``JAX_ENABLE_X64=0`` and parse its JSON."""
    tmp: Path = tmp_path_factory.mktemp("x64_off")
    uvw, freq, image, vis, pixsize = _fixture_arrays()
    npz = tmp / "fixture.npz"
    np.savez(
        npz,
        uvw=uvw,
        freq=freq,
        image=image,
        vis=vis,
        weights=_weights(),
        pixsize=np.asarray(pixsize),
    )
    off30_paths = []
    for tel in _OFF30_TELESCOPES:
        path = tmp / f"{tel.name}_off30.npz"
        np.savez(path, **_off30_arrays(tel))
        off30_paths.append(str(path))
    script = tmp / "x64_off_child.py"
    script.write_text(_CHILD_SCRIPT)

    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "0"
    proc = subprocess.run(
        [sys.executable, str(script), str(npz), *off30_paths],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, (
        "the JAX_ENABLE_X64=0 child interpreter crashed outside its guarded probes"
        f"\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    marker_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_JSON_MARKER)]
    assert len(marker_lines) == 1, f"child produced no JSON payload:\n{proc.stdout}"
    report: dict[str, Any] = json.loads(marker_lines[0][len(_JSON_MARKER) :])
    # Sanity: the child really did run without x64, otherwise the whole leg is
    # meaningless.
    assert report["x64"] is False
    return report


def test_x64_off_default_dtype_raises(x64_off_report: dict[str, Any]) -> None:
    """With x64 off, the default float64 plan must fail loudly, not degrade."""
    err = x64_off_report["default_error"]
    assert err is not None, (
        "make_plan built a plan with jax_enable_x64 off: every leaf silently "
        "becomes float32 and the error floors near 3.4e-5 with no warning"
    )
    assert err["type"] == "ValueError", err
    # The message must say what is wrong and how to opt into single precision.
    assert "jax_enable_x64" in err["message"], err["message"]
    assert "dtype=jnp.float32" in err["message"], err["message"]


def test_x64_off_float32_plan_matches_ducc(x64_off_report: dict[str, Any]) -> None:
    """An explicit float32 plan is a supported configuration at eps=1e-4."""
    assert x64_off_report["float32_error"] is None, (
        f"the float32 probe failed in the child: {x64_off_report['float32_error']}"
    )
    assert x64_off_report["warnings_at_1e_4"] == [], (
        "eps=1e-4 is inside the float32 floor; it must not warn"
    )
    assert x64_off_report["plan_real_dtype"] == "float32"
    assert x64_off_report["plan_complex_dtype"] == "complex64"
    # Exact allowlist, so a float64 / complex128 leaf leaking into a float32
    # plan still fails here: float32 (numerics), int32 (sort_perm / window
    # tables), and complex64 for the one genuinely complex leaf, ``w0_screen``
    # (issue #16 follow-up: the precomputed exp(2i*pi*w0*(n-1)) phase screen).
    # ``int8`` is issue #17's ``flip_sign``, the Hermitian fold's per-row sign:
    # one byte per row and no more (the exact width is gated in
    # ``tests/test_hermitian.py::test_flip_sign_leaf_is_one_signed_byte_per_row``;
    # what this list adds is that it must not drag a wider *floating* leaf into
    # a float32 plan). Sorted, so "int8" follows "int32".
    assert x64_off_report["plan_leaf_dtypes"] == ["complex64", "float32", "int32", "int8"]
    assert x64_off_report["vis_dtype"] == "complex64"
    rel = x64_off_report["rel_err_1e_4"]
    assert rel < 3 * 1e-4, f"float32 plan vs ducc0 at eps=1e-4: relative error {rel:.3e}"


def test_x64_off_float32_adjoint_matches_ducc(x64_off_report: dict[str, Any]) -> None:
    """The x64-off leg must exercise ``vis2dirty`` too, weights included.

    Everything else under ``JAX_ENABLE_X64=0`` is skipped by ``collect_ignore``
    in ``tests/conftest.py``, so without this the single-precision CI job would
    stay green with the adjoint completely broken.
    """
    assert x64_off_report["adjoint_error"] is None, (
        f"the float32 adjoint probe failed in the child: {x64_off_report['adjoint_error']}"
    )
    assert x64_off_report["dirty_dtype"] == "float32"
    rel = x64_off_report["adjoint_rel_err_1e_4"]
    assert rel < 3 * 1e-4, f"float32 adjoint vs ducc0 at eps=1e-4: relative error {rel:.3e}"


def test_x64_off_float32_plan_rejects_a_raw_float64_numpy_image(
    x64_off_report: dict[str, Any],
) -> None:
    """The 2-D broadcast must not launder a wide numpy image into the plan.

    With x64 off, entering JAX narrows float64 to float32 silently, so a
    ``jnp.broadcast_to`` ahead of the dtype check would let the 2-D path
    accept exactly the image the 3-D path refuses.
    """
    err = x64_off_report["wide_numpy_image_error"]
    assert err is not None, (
        "a raw numpy float64 image was accepted by a float32 plan; the 2-D path "
        f"broadcast before checking the dtype (result dtype "
        f"{x64_off_report.get('wide_numpy_image_dtype')})"
    )
    assert err["type"] == "TypeError", err
    assert "float64" in err["message"], err["message"]
    assert "float32" in err["message"], err["message"]


def test_x64_off_float32_plan_warns_below_the_floor(x64_off_report: dict[str, Any]) -> None:
    """eps=1e-6 is unreachable in float32 — say so instead of silently missing it."""
    assert x64_off_report["warn_probe_error"] is None, (
        f"the eps=1e-6 float32 probe failed in the child: {x64_off_report['warn_probe_error']}"
    )
    user_warnings = [
        w for w in x64_off_report["warnings_at_1e_6"] if w["category"] == "UserWarning"
    ]
    assert len(user_warnings) == 1, x64_off_report["warnings_at_1e_6"]
    message = user_warnings[0]["message"]
    assert "float32" in message, message
    # The achievable floor has to be named so the user can act on it.
    assert "1e-5" in message, message


@pytest.mark.parametrize("tel", _OFF30_TELESCOPES, ids=lambda t: f"{t.name}_off30")
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_x64_off_float32_matches_ducc_off_zenith(
    x64_off_report: dict[str, Any], tel: Telescope, op: str
) -> None:
    """The x64-off leg must run off-zenith geometry, not only EDA2 at zenith.

    Every other probe in this module runs on EDA2 zenith, whose float32 plan has
    ``n_w = 9`` -- fewer planes than ``w_kernel_width + 5``. At that plane count
    the w-plane loop is nearly degenerate: a bug in plane placement, in the
    per-plane phase, or in the accumulation across planes has almost nothing to
    accumulate over, and single precision has almost nothing to lose. Plane
    count is an axis the single-precision leg held constant, and MWA_extended
    off30 moves it to 132.

    Not a windowed-vs-dense or float32-vs-float64 comparison: those are internal
    and would agree with each other while both drifted. The oracle is
    double-precision ducc0, at the repo's 3*eps.
    """
    key = f"{tel.name}_off30"
    report = x64_off_report["off30"][key]
    assert report["error"] is None, f"the {key} float32 probe failed in the child: {report}"
    assert report["real_dtype"] == "float32"
    assert report["vis_dtype"] == "complex64"
    assert report["dirty_dtype"] == "float32"

    err = report["forward_rel_err"] if op == "dirty2vis" else report["adjoint_rel_err"]
    bound = 3 * _OFF30_EPS
    assert err < bound, (
        f"{key} float32 {op} vs ducc0 at eps={_OFF30_EPS:g}: relative error "
        f"{err:.3e} exceeds {bound:.3e} (n_w={report['n_w']})"
    )


def test_the_x64_off_leg_reaches_a_deep_w_plane_stack(x64_off_report: dict[str, Any]) -> None:
    """At least one x64-off fixture must be many planes deep.

    The anti-vacuity guard for the test above: if a planning change collapsed
    every off30 float32 plan back to the zenith fixture's handful of planes,
    those cells would keep passing while covering nothing the zenith probe did
    not already cover. ``10 * w_kernel_width`` is well clear of the shipped
    numbers (MWA_extended off30 is 132 planes at W=5) and well clear of the
    other two fixtures (10 and 11), so it is a statement about the deep cell.
    """
    deepest = max(
        (r for r in x64_off_report["off30"].values() if r["error"] is None),
        key=lambda r: r["n_w"],
        default=None,
    )
    assert deepest is not None
    assert deepest["n_w"] > 10 * deepest["w_kernel_width"], (
        f"the deepest x64-off fixture has only n_w={deepest['n_w']} planes at "
        f"W={deepest['w_kernel_width']}: the single-precision leg is back to "
        "exercising the w-plane loop at a plane count that cannot distinguish "
        "the traversals"
    )


# ---------------------------------------------------------------------------
# Leg 2: jax_enable_x64 on (this interpreter)
# ---------------------------------------------------------------------------


def _plan(dtype: Any, epsilon: float = 1e-4) -> WGridderPlan:
    uvw, freq, _image, _vis, pixsize = _fixture_arrays()
    return make_plan(
        uvw,
        freq,
        (EDA2.n_pix, EDA2.n_pix),
        pixsize,
        pixsize,
        epsilon,
        dtype=dtype,
    )


@requires_x64
def test_plan_dtype_metadata_defaults_to_float64() -> None:
    """The default plan is float64 and advertises both dtypes as metadata."""
    plan = _plan(jnp.float64)
    assert _dtype_is(plan.real_dtype, jnp.float64)
    assert _dtype_is(plan.complex_dtype, jnp.complex128)
    assert plan.uvw_lambda.dtype == jnp.float64
    assert plan.n_minus_1.dtype == jnp.float64
    # The default argument must give the same plan as the explicit request.
    uvw, freq, _image, _vis, pixsize = _fixture_arrays()
    default_plan = make_plan(uvw, freq, (EDA2.n_pix, EDA2.n_pix), pixsize, pixsize, 1e-4)
    assert _dtype_is(default_plan.real_dtype, jnp.float64)


@requires_x64
def test_float32_plan_is_float32_even_with_x64_on() -> None:
    """``dtype`` drives the plan, not the dtype of the (float64) uvw/freq inputs."""
    plan = _plan(jnp.float32)
    assert _dtype_is(plan.real_dtype, jnp.float32)
    assert _dtype_is(plan.complex_dtype, jnp.complex64)
    float_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(plan) if jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    assert float_leaves, "plan has no floating-point leaves"
    assert all(leaf.dtype == jnp.float32 for leaf in float_leaves), [
        str(leaf.dtype) for leaf in float_leaves
    ]


@requires_x64
def test_float32_plan_warns_below_the_float32_floor() -> None:
    """``pytest.warns`` rather than escape: the suite runs with filterwarnings=error."""
    uvw, freq, _image, _vis, pixsize = _fixture_arrays()
    with pytest.warns(UserWarning, match="1e-5"):
        make_plan(
            uvw,
            freq,
            (EDA2.n_pix, EDA2.n_pix),
            pixsize,
            pixsize,
            1e-6,
            dtype=jnp.float32,
        )


@requires_x64
def test_float32_plan_does_not_warn_at_reachable_epsilon() -> None:
    """eps=1e-4 is inside the float32 floor; filterwarnings=error catches a stray warn."""
    plan = _plan(jnp.float32, epsilon=1e-4)
    assert _dtype_is(plan.real_dtype, jnp.float32)


@requires_x64
def test_float32_plan_matches_ducc_at_1e_4() -> None:
    """A float32 plan is usable under x64 too: it still tracks the double oracle."""
    uvw, freq, image, _vis, pixsize = _fixture_arrays()
    plan = _plan(jnp.float32, epsilon=1e-4)
    vis_jax = np.asarray(dirty2vis(plan, jnp.asarray(image, dtype=jnp.float32)))
    assert vis_jax.dtype == np.complex64
    vis_ducc = ducc0.wgridder.dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=image,
        pixsize_x=pixsize,
        pixsize_y=pixsize,
        epsilon=1e-4,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )
    err = _rel_err(vis_jax, vis_ducc)
    assert err < 3 * 1e-4, f"float32 plan vs ducc0 at eps=1e-4: relative error {err:.3e}"


@requires_x64
def test_float64_plan_casts_a_float32_image() -> None:
    """A narrower image is cast up to the plan's complex dtype, not rejected."""
    _uvw, _freq, image, _vis, _pixsize = _fixture_arrays()
    plan = _plan(jnp.float64)
    vis_ref = np.asarray(dirty2vis(plan, jnp.asarray(image)))
    vis_cast = dirty2vis(plan, jnp.asarray(image, dtype=jnp.float32))
    assert vis_cast.dtype == jnp.complex128, "the plan's precision governs the output"
    err = _rel_err(np.asarray(vis_cast), vis_ref)
    assert err < 1e-6, f"float32 image into a float64 plan: relative error {err:.3e}"


@requires_x64
def test_float64_plan_casts_a_complex64_vis() -> None:
    """Same contract on the adjoint: complex64 vis into a float64 plan is cast."""
    _uvw, _freq, _image, vis, _pixsize = _fixture_arrays()
    plan = _plan(jnp.float64)
    dirty_ref = np.asarray(vis2dirty(plan, jnp.asarray(vis)))
    dirty_cast = vis2dirty(plan, jnp.asarray(vis, dtype=jnp.complex64))
    assert dirty_cast.dtype == jnp.float64
    err = _rel_err(np.asarray(dirty_cast), dirty_ref)
    assert err < 1e-6, f"complex64 vis into a float64 plan: relative error {err:.3e}"


@requires_x64
def test_float32_plan_rejects_a_float64_image() -> None:
    """A *wider* image would silently lose the user's precision: refuse it."""
    _uvw, _freq, image, _vis, _pixsize = _fixture_arrays()
    plan = _plan(jnp.float32)
    image64 = jnp.asarray(image)
    assert image64.dtype == jnp.float64
    with pytest.raises(TypeError) as excinfo:
        dirty2vis(plan, image64)
    message = str(excinfo.value)
    assert "float64" in message, message
    assert "float32" in message, message


@requires_x64
def test_float32_plan_rejects_a_complex128_vis() -> None:
    """Adjoint counterpart: complex128 vis into a float32 (complex64) plan."""
    _uvw, _freq, _image, vis, _pixsize = _fixture_arrays()
    plan = _plan(jnp.float32)
    vis128 = jnp.asarray(vis)
    assert vis128.dtype == jnp.complex128
    with pytest.raises(TypeError) as excinfo:
        vis2dirty(plan, vis128)
    message = str(excinfo.value)
    assert "complex128" in message, message
    assert "complex64" in message, message


@requires_x64
@pytest.mark.parametrize("plan_dtype", [jnp.float64, jnp.float32])
def test_complex_weights_are_rejected(plan_dtype: Any) -> None:
    """``weights`` are real by contract (ducc's ``wgt``): refuse a complex array.

    Casting to the plan's real dtype would silently drop the imaginary part,
    and treating it as a merely "too wide" dtype would advise narrowing to a
    complex dtype, which loses the same data. Neither is acceptable, so the
    kind check comes first and names the offending dtype.
    """
    _uvw, _freq, _image, vis, _pixsize = _fixture_arrays()
    plan = _plan(plan_dtype)
    complex_weights = jnp.ones((EDA2.n_rows, 1), dtype=jnp.complex64)
    with pytest.raises(TypeError) as excinfo:
        vis2dirty(plan, jnp.asarray(vis, dtype=plan.complex_dtype), weights=complex_weights)
    message = str(excinfo.value)
    assert "complex64" in message, message
    assert "real" in message, message


@requires_x64
def test_narrower_real_weights_are_cast_up() -> None:
    """The complex rejection must not break the narrower-is-cast-up contract."""
    _uvw, _freq, _image, vis, _pixsize = _fixture_arrays()
    plan = _plan(jnp.float64)
    weights = _weights()
    dirty_ref = np.asarray(vis2dirty(plan, jnp.asarray(vis), weights=jnp.asarray(weights)))
    dirty_cast = vis2dirty(
        plan,
        jnp.asarray(vis),
        weights=jnp.asarray(weights, dtype=jnp.float32),
    )
    assert dirty_cast.dtype == jnp.float64, "the plan's precision governs the output"
    err = _rel_err(np.asarray(dirty_cast), dirty_ref)
    assert err < 1e-6, f"float32 weights into a float64 plan: relative error {err:.3e}"


@requires_x64
def test_malformed_uvw_beats_the_float32_epsilon_warning() -> None:
    """Argument validation must run before the accuracy warning.

    With ``filterwarnings = ["error"]`` an eagerly-emitted ``UserWarning``
    would mask the shape error that actually describes the caller's bug.
    """
    uvw, freq, _image, _vis, pixsize = _fixture_arrays()
    with pytest.raises(ValueError, match=r"uvw must have shape"):
        make_plan(
            uvw[:, :2],
            freq,
            (EDA2.n_pix, EDA2.n_pix),
            pixsize,
            pixsize,
            1e-6,
            dtype=jnp.float32,
        )


@requires_x64
def test_unreachable_epsilon_beats_the_float32_epsilon_warning() -> None:
    """``kernel_params``'s own rejection must also beat the accuracy warning.

    Same class of ordering bug as the malformed-``uvw`` case above, surfaced
    by the rebase onto issue #11: ``epsilon=1e-15`` is both below
    ``FLOAT32_EPSILON_FLOOR`` (so ``_warn_if_below_float32_floor`` would fire)
    and below the 1e-14 floor ``kernel_params`` refuses outright. With
    ``filterwarnings = ["error"]`` the warning would raise first if it ran
    first, and its own text ("the plan will build") would be false -- the
    plan cannot build at this epsilon at all, in any dtype. ``make_plan``
    must raise the ``ValueError`` naming the unreachable epsilon, not a
    ``UserWarning`` about accuracy.
    """
    uvw, freq, _image, _vis, pixsize = _fixture_arrays()
    with pytest.raises(ValueError, match=r"1e-14"):
        make_plan(
            uvw,
            freq,
            (EDA2.n_pix, EDA2.n_pix),
            pixsize,
            pixsize,
            1e-15,
            dtype=jnp.float32,
        )


@requires_x64
def test_malformed_phi_hat_oversample_beats_the_float32_epsilon_warning() -> None:
    """The two phi_hat-table overrides must also beat the accuracy warning.

    Same class of ordering bug as the two cases above, this time for
    ``phi_hat_oversample`` / ``phi_hat_n_fine``: they are validated inside
    ``compute_phi_hat_table`` (kernel.py), but that function is only called
    much later in ``make_plan``, well after ``_warn_if_below_float32_floor``.
    ``epsilon=1e-6`` alone is below ``FLOAT32_EPSILON_FLOOR`` (so the
    accuracy warning would fire), and ``phi_hat_oversample=0`` is invalid.
    With ``filterwarnings = ["error"]`` the warning would raise first if it
    ran first, again with its false "the plan will build" claim.
    ``make_plan`` must raise the ``ValueError`` naming the invalid
    oversample, not a ``UserWarning`` about accuracy.
    """
    uvw, freq, _image, _vis, pixsize = _fixture_arrays()
    with pytest.raises(ValueError, match=r"phi_hat_oversample"):
        make_plan(
            uvw,
            freq,
            (EDA2.n_pix, EDA2.n_pix),
            pixsize,
            pixsize,
            1e-6,
            dtype=jnp.float32,
            phi_hat_oversample=0,
        )


_MIXED_CASES = (
    "float64_plan_float32_image",
    "float64_plan_complex64_vis",
    "float32_plan_float64_image",
    "float32_plan_complex128_vis",
)


@requires_x64
@pytest.mark.parametrize("case", _MIXED_CASES)
def test_mixed_dtypes_never_leak_a_bare_assertion(case: str) -> None:
    """jax-finufft raises a message-less ``AssertionError()`` on a dtype mismatch.

    Whatever we do with mixed dtypes — cast or refuse — that assertion must
    never reach the caller: it names neither the offending array nor the fix.
    """
    _uvw, _freq, image, vis, _pixsize = _fixture_arrays()
    plan64 = _plan(jnp.float64)
    plan32 = _plan(jnp.float32)
    calls = {
        "float64_plan_float32_image": lambda: dirty2vis(
            plan64, jnp.asarray(image, dtype=jnp.float32)
        ),
        "float64_plan_complex64_vis": lambda: vis2dirty(plan64, jnp.asarray(vis, jnp.complex64)),
        "float32_plan_float64_image": lambda: dirty2vis(plan32, jnp.asarray(image)),
        "float32_plan_complex128_vis": lambda: vis2dirty(plan32, jnp.asarray(vis)),
    }
    try:
        result = calls[case]()
    except Exception as exc:  # the exception TYPE is what this test is about
        assert not isinstance(exc, AssertionError), (
            f"{case}: jax-finufft's bare AssertionError escaped to the caller"
        )
        assert isinstance(exc, TypeError), f"{case}: expected TypeError, got {exc!r}"
        assert str(exc).strip(), f"{case}: the dtype error must carry a message"
    else:
        assert np.all(np.isfinite(np.asarray(result))), f"{case}: produced non-finite output"


# --------------------------------------------------------------------------
# What float32 buys: exactly half the memory (issue #33).
#
# The README offers single precision as the cheapest way to halve a problem's
# footprint, so the factor is a documented number and needs a test rather than
# an appeal to "floats are half the width" -- which would not settle it, since
# the plan also carries int32 index tables whose size does not change with the
# floating dtype, and a scratch buffer dominated by those would not halve.
# --------------------------------------------------------------------------


# The largest non-scaling remainder observed over the cases below, rounded up to
# a round number, and the weakest ratio it implies. Both are quoted in README.md.
_HALVING_FIXED_OVERHEAD_BYTES = 512
_HALVING_MIN_RATIO = 1.999

_HALVING_SIZES = (
    # (telescope, n_pix, n_rows) -- one image-dominated, one row-dominated, and
    # one CI-sized, so the ratio is not established on a single size class. See
    # the sizing rule in docs/benchmarks/v0.2.0-vs-ducc0-gh200.json.
    pytest.param(MEERKAT, 540, 60_480, id="MeerKAT_image_heavy"),
    pytest.param(EDA2, 150, 244_800, id="EDA2_row_heavy"),
    pytest.param(MWA_EXTENDED, 256, 600, id="MWA_extended_ci_sized"),
)


def _scratch_bytes(plan: WGridderPlan, op: str, w_strategy: str, w_chunk: int, n_rows: int) -> int:
    """Compiler-reported scratch for one compiled operator.

    ``temp_size_in_bytes`` is a property of the executable, so it needs no
    device and no run, and it is unaffected by whatever else the process has
    done -- unlike a high-water mark such as ``peak_bytes_in_use``.
    """
    rng = np.random.default_rng(7)
    n_pix = plan.image_shape[0]
    if op == "dirty2vis":
        data = jnp.asarray(rng.standard_normal((n_pix, n_pix)), plan.real_dtype)

        def fn(d):
            return dirty2vis(plan, d, w_strategy=w_strategy, w_chunk=w_chunk)
    else:
        raw = rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))
        data = jnp.asarray(raw, plan.complex_dtype)

        def fn(d):
            return vis2dirty(plan, d, w_strategy=w_strategy, w_chunk=w_chunk)

    return jax.jit(fn).lower(data).compile().memory_analysis().temp_size_in_bytes


@requires_x64
@pytest.mark.parametrize("tel,n_pix,n_rows", _HALVING_SIZES)
@pytest.mark.parametrize("w_strategy", ["dense_scan", "chunked"])
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_a_float32_plan_needs_half_the_scratch_of_a_float64_one(
    tel: Telescope, n_pix: int, n_rows: int, w_strategy: str, op: str
) -> None:
    """The README's "float32 halves it", to the precision the README states.

    Quantified over three size classes, two strategies and both operators,
    because the interesting way for this to be false is for it to hold on the
    shape the claim was written against and not on another.
    """
    from dataclasses import replace

    sized = replace(tel, n_pix=n_pix, n_rows=n_rows)
    uvw = synthetic_uvw(sized, 30.0, seed=0)
    freq = np.array([sized.freq_hz])
    kwargs = dict(
        uvw=uvw,
        freq=freq,
        image_shape=(n_pix, n_pix),
        pixsize_l=sized.pixsize,
        pixsize_m=sized.pixsize,
        # 1e-4 is reachable in single precision; 1e-6 is not, and asking for it
        # would warn and compare two different kernel widths.
        epsilon=1e-4,
    )
    wide = _scratch_bytes(
        make_plan(**kwargs, dtype=jnp.float64), op, w_strategy, w_chunk=8, n_rows=n_rows
    )
    narrow = _scratch_bytes(
        make_plan(**kwargs, dtype=jnp.float32), op, w_strategy, w_chunk=8, n_rows=n_rows
    )
    assert narrow > 0 and wide > 0, "a compiled operator with no scratch is not a real case"

    # Not exactly 2x: a small fixed remainder does not scale with the floating
    # dtype, so the halving is asymptotic rather than exact. Measured at 96-264
    # bytes over these twelve cells, against scratch of 2 MB to 39 MB. Both
    # halves of the bound matter -- the size of the remainder is what makes
    # "halves it" honest, and the ratio is what the README quotes.
    remainder = 2 * narrow - wide
    assert 0 <= remainder <= _HALVING_FIXED_OVERHEAD_BYTES, (
        f"float64 scratch is {wide} B and float32 {narrow} B, leaving "
        f"{remainder} B that does not scale with the floating dtype. README.md "
        f"describes the halving as exact up to a fixed overhead of at most "
        f"{_HALVING_FIXED_OVERHEAD_BYTES} B; a remainder growing with the "
        "problem would mean some part of the working set (the plan's int32 "
        "index tables, say) is being counted in the documented factor when it "
        "does not shrink."
    )
    ratio = wide / narrow
    assert ratio >= _HALVING_MIN_RATIO, (
        f"float32 saves a factor of {ratio:.6f}, below the {_HALVING_MIN_RATIO} "
        "README.md quotes as 'halves it'."
    )


# The plan is a different question from the scratch, and has a different
# answer. Its floating leaves halve, but ``sort_perm`` is int32 and
# ``flip_sign`` is int8, so a plan whose size is dominated by per-row tables
# saves noticeably less than one dominated by the image grid. README.md quotes
# both ends; a single fixture would only ever show one of them.
_PLAN_RATIO_CASES = (
    pytest.param(MWA_EXTENDED, 256, 600, 0.501, id="image_dominated"),
    pytest.param(EDA2, 150, 4_896_000, 0.586, id="row_dominated"),
)


@requires_x64
@pytest.mark.parametrize("tel,n_pix,n_rows,expected", _PLAN_RATIO_CASES)
def test_what_float32_saves_on_the_plan_depends_on_its_shape(
    tel: Telescope, n_pix: int, n_rows: int, expected: float
) -> None:
    """Both ends of the range README.md quotes for the plan.

    The interesting failure is not the ratio drifting a little; it is someone
    reading "float32 halves memory", applying it to a row-heavy plan, and
    budgeting 41% less than the plan will take.
    """
    from dataclasses import replace

    sized = replace(tel, n_pix=n_pix, n_rows=n_rows)
    uvw = synthetic_uvw(sized, 30.0, seed=0)
    freq = np.array([sized.freq_hz])

    def plan_bytes(dtype: Any) -> int:
        plan = make_plan(
            uvw,
            freq,
            (n_pix, n_pix),
            sized.pixsize,
            sized.pixsize,
            1e-4,
            dtype=dtype,
        )
        return sum(leaf.nbytes for leaf in jax.tree.leaves(plan) if hasattr(leaf, "nbytes"))

    ratio = plan_bytes(jnp.float32) / plan_bytes(jnp.float64)
    assert round(ratio, 3) == expected, (
        f"a float32 plan is {ratio:.3f} of the float64 one here; README.md's "
        f"Precision section quotes {expected}. The two ends of that range are "
        "the point of the sentence -- an image-dominated plan halves, a "
        "row-dominated one does not -- so a change in either end changes the "
        "advice."
    )
