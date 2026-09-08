"""Keep ``tests/jax_floor_probe.py`` honest, and the three floor declarations agreed.

The probe itself runs in its own CI job against ``jax`` pinned to the declared
floor (see ``.github/workflows/test.yml``); this module runs the same two checks
against whatever ``jax`` the pixi environment resolved. That is not redundant.
The probe's real failure mode is not "a symbol is missing" but "the scanner
returned nothing and the job passed vacuously" -- a derived list that derives
zero symbols is green at every floor, forever. So the tests below pin what the
scan must contain, and pin the alias forms it has to understand, in the
environment that actually gets run on every push.

``test_pixi_features_declare_the_same_floor`` covers the other half of issue
#21's original bug: the floor is declared in *three* places (``pyproject.toml``
and both pixi CPU/GPU features) and only the ``pyproject.toml`` one is what the
probe job installs. Two of the three agreeing is the same silence the probe
exists to break.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from tests.jax_floor_probe import (
    REPO_ROOT,
    ProbeError,
    check_primitive_pattern,
    check_symbols,
    declared_floor,
    jax_symbols,
)

# Anchors for the derived set: names that ``src/jax_nufft/wgridder.py`` cannot
# stop using without the primitive pattern itself changing. They are **not** the
# authority on what gets probed -- the AST scan is, and it currently finds many
# more -- and they are not a transcription of the 18 symbols issue #53 records
# the #21 review as having checked by hand, since that list is not recorded
# anywhere in the repository. They exist for one purpose: a scan that silently
# stops finding things must fail here rather than pass everywhere. Written out
# by hand deliberately, because a second derived list would fail in the same
# way as the first.
ANCHOR_SYMBOLS = frozenset(
    {
        "jax.Array",
        "jax.core.ShapedArray",
        "jax.disable_jit",
        "jax.extend.core.Primitive",
        "jax.grad",
        "jax.interpreters.ad.Zero",
        "jax.interpreters.ad.is_undefined_primal",
        "jax.interpreters.ad.primitive_jvps",
        "jax.interpreters.ad.primitive_transposes",
        "jax.interpreters.batching.not_mapped",
        "jax.interpreters.batching.primitive_batchers",
        "jax.interpreters.mlir.lower_fun",
        "jax.interpreters.mlir.register_lowering",
        "jax.jit",
        "jax.jvp",
        "jax.linear_transpose",
        "jax.tree_util.register_pytree_node",
        "jax.typeof",
    }
)


def test_declared_floor_is_a_version_string() -> None:
    floor = declared_floor()
    assert floor[0].isdigit(), floor
    assert floor.count(".") >= 1, floor


def test_declared_floor_requires_a_lower_bound(tmp_path: Path) -> None:
    """A ``jax`` requirement with no ``>=`` is a floor the probe cannot exercise."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["jax", "numpy>=1.24"]\n')
    with pytest.raises(ProbeError, match="no '>=' lower bound"):
        declared_floor(pyproject)


def test_declared_floor_is_not_confused_by_jax_finufft(tmp_path: Path) -> None:
    """``jax-finufft>=1.3.0`` also starts with "jax" and must not be read as the floor."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["jax-finufft>=1.3.0", "jax>=0.6.0"]\n')
    assert declared_floor(pyproject) == "0.6.0"


def test_pixi_features_declare_the_same_floor() -> None:
    """``pyproject.toml`` and both pixi features must agree on the jax floor.

    Issue #21 had to raise all three. Only the ``pyproject.toml`` one is what
    the probe job installs, so a pixi feature left behind would be a floor no
    CI job exercises -- the exact hole this issue closes.
    """
    floor = declared_floor()
    with (REPO_ROOT / "pixi.toml").open("rb") as handle:
        pixi = tomllib.load(handle)
    declared = {
        f"feature.{name}": pixi["feature"][name]["dependencies"]["jax"] for name in ("cpu", "gpu")
    }
    assert declared == {"feature.cpu": f">={floor}", "feature.gpu": f">={floor}"}


def test_symbol_scan_still_finds_the_anchors() -> None:
    """The derived set must be a strict superset of the anchors.

    This is the anti-vacuity gate. A scanner that returns an empty set -- an
    alias form it stops understanding, a directory that moves -- makes the CI
    probe green at every jax version ever released, and nothing else in the
    repository would notice.
    """
    derived = jax_symbols()
    assert ANCHOR_SYMBOLS <= derived, sorted(ANCHOR_SYMBOLS - derived)
    assert len(derived) > len(ANCHOR_SYMBOLS)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import jax\njax.typeof(x)\n", "jax.typeof"),
        ("import jax.numpy as jnp\njnp.exp(x)\n", "jax.numpy.exp"),
        ("import jax.extend as jex\njex.core.Primitive('p')\n", "jax.extend.core.Primitive"),
        ("from jax.interpreters import ad\nad.Zero(a)\n", "jax.interpreters.ad.Zero"),
        ("from jax import grad\ngrad(f)\n", "jax.grad"),
        ("import jax.numpy\njax.numpy.fft.fftn(x)\n", "jax.numpy.fft.fftn"),
        # A ``Call`` ends the chain, so an instance attribute of the result is
        # not mistaken for a module attribute of ``jax.numpy``.
        ("import jax.numpy as jnp\njnp.array(x).shape\n", "jax.numpy.array"),
    ],
)
def test_scanner_understands_each_alias_form(tmp_path: Path, source: str, expected: str) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text(source)
    assert expected in jax_symbols(tmp_path, scan_dirs=("src",))


def test_scanner_ignores_non_jax_roots(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text(
        "import jax\nimport numpy as np\nnp.exp(plan.uvw_m)\njax.jit(f)\n"
    )
    assert jax_symbols(tmp_path, scan_dirs=("src",)) == {"jax", "jax.jit"}


def test_every_derived_jax_symbol_resolves() -> None:
    """The installed jax has every symbol ``src/`` and ``tests/`` touch.

    Trivially true here (the pixi env resolves a current jax); the point is that
    the same function is what the floor job runs, so it cannot rot unnoticed.
    """
    assert len(check_symbols()) >= len(ANCHOR_SYMBOLS)


def test_primitive_pattern_behaves() -> None:
    """The miniature of the wgridder's three linear primitives, on this jax."""
    check_primitive_pattern()
