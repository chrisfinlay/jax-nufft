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
