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
from tests.conftest import EDA2, requires_x64, synthetic_uvw, tol

# EDA2 zenith: 400 rows, 64x64 pixels, 120-degree FoV. The smallest review
# fixture, so the whole module (including one child interpreter) stays well
# under a minute.
_SEED = 0
_IMAGE_SEED = 7
_VIS_SEED = 11


def _fixture_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """``(uvw, freq, image, vis, pixsize)`` for the EDA2 zenith fixture."""
    uvw = synthetic_uvw(EDA2, 0.0, seed=_SEED)
    freq = np.array([EDA2.freq_hz])
    image = np.random.default_rng(_IMAGE_SEED).standard_normal((EDA2.n_pix, EDA2.n_pix))
    rng = np.random.default_rng(_VIS_SEED)
    vis = rng.standard_normal((EDA2.n_rows, 1)) + 1j * rng.standard_normal((EDA2.n_rows, 1))
    return uvw, freq, image, vis, EDA2.pixsize


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

Run as ``python child.py fixture.npz`` with ``JAX_ENABLE_X64=0`` in the
environment. Emits one JSON object on stdout, prefixed by a marker line.
"""

import json
import sys
import warnings

import ducc0.wgridder
import jax
import jax.numpy as jnp
import numpy as np

from jax_nufft import dirty2vis, make_plan

data = np.load(sys.argv[1])
uvw = data["uvw"]
freq = data["freq"]
image = data["image"]
pixsize = float(data["pixsize"])
n_pix = int(image.shape[0])

out = {"x64": bool(jax.config.jax_enable_x64), "float32_error": None, "warn_probe_error": None}


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

# (c) Asking float32 for eps=1e-6 is below the achievable floor: warn.
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, 1e-6, dtype=jnp.float32)
    out["warnings_at_1e_6"] = record(caught)
except Exception as exc:
    out["warn_probe_error"] = describe(exc)

print("<<<JSON>>>" + json.dumps(out))
'''

_JSON_MARKER = "<<<JSON>>>"


@pytest.fixture(scope="module")
def x64_off_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run ``_CHILD_SCRIPT`` once with ``JAX_ENABLE_X64=0`` and parse its JSON."""
    tmp: Path = tmp_path_factory.mktemp("x64_off")
    uvw, freq, image, _vis, pixsize = _fixture_arrays()
    npz = tmp / "fixture.npz"
    np.savez(npz, uvw=uvw, freq=freq, image=image, pixsize=np.asarray(pixsize))
    script = tmp / "x64_off_child.py"
    script.write_text(_CHILD_SCRIPT)

    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "0"
    proc = subprocess.run(
        [sys.executable, str(script), str(npz)],
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
    # Only float32 (numerics) and int32 (sort_perm / window tables) leaves.
    assert x64_off_report["plan_leaf_dtypes"] == ["float32", "int32"]
    assert x64_off_report["vis_dtype"] == "complex64"
    rel = x64_off_report["rel_err_1e_4"]
    assert rel < 3 * 1e-4, f"float32 plan vs ducc0 at eps=1e-4: relative error {rel:.3e}"


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
