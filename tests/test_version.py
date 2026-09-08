"""The package must report one version, not two (issue #59).

Before this, ``pyproject.toml`` carried a static ``[project] version`` that
hatchling read and ``src/jax_nufft/_version.py`` carried a second literal that
nothing read. The v0.1.2 series bumped ``_version.py`` alone and was never
tagged, so ``jax_nufft.__version__`` reported ``0.1.2`` while the wheel, the
sdist and ``pip show`` all reported ``0.1.1``, and neither matched a tag. A
user could not name the version they were running.

Nothing caught it because nothing read both. These tests do.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import tomllib

import jax_nufft

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
# PEP 440 release segment plus the optional pre/post/dev parts this project
# might plausibly use. Deliberately not a full PEP 440 grammar: the point is to
# reject a placeholder like "0.0.0" or a stray "v" prefix, not to validate
# every legal spelling.
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+|\.dev\d+)?$")


def test_the_dunder_version_matches_the_installed_metadata() -> None:
    """``jax_nufft.__version__`` and the installed distribution agree.

    This is the assertion that would have caught #59 the day it was
    introduced. It compares the value the *library* reports against the value
    the *package manager* reports, which are the two numbers a user sees.
    """
    installed = metadata.version("jax-nufft")
    assert jax_nufft.__version__ == installed, (
        f"jax_nufft.__version__ is {jax_nufft.__version__!r} but the installed "
        f"distribution is {installed!r}. These are the two versions a user can "
        "see -- the library's own and the one `pip show` reports -- and they "
        "must not drift. See pyproject.toml's [tool.hatch.version]."
    )


def test_pyproject_takes_its_version_from_the_module_rather_than_a_literal() -> None:
    """The build backend must read ``_version.py``, not a second literal.

    The equality above holds trivially if someone re-adds a static
    ``[project] version`` and happens to set it to the same string; it would
    then drift again at the next bump. This pins the *mechanism*: there is one
    source and the build reads it.
    """
    config = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    project = config["project"]

    assert "version" not in project, (
        "pyproject.toml declares a static [project] version again. That is the "
        "second literal that caused #59; the version must come from "
        "src/jax_nufft/_version.py via [tool.hatch.version]."
    )
    assert "version" in project.get("dynamic", []), (
        "[project] must declare version as dynamic so hatchling resolves it "
        "from [tool.hatch.version]."
    )
    hatch_version = config["tool"]["hatch"]["version"]
    assert hatch_version["path"] == "src/jax_nufft/_version.py", (
        f"[tool.hatch.version] reads {hatch_version.get('path')!r}; it must read "
        "src/jax_nufft/_version.py, the module that defines __version__."
    )


def test_the_version_is_a_plausible_release_string() -> None:
    """Not a placeholder, and not carrying a ``v`` prefix.

    Cheap, but it is the difference between shipping ``0.2.0`` and shipping
    ``v0.2.0`` or a leftover ``0.0.0``, both of which sort and compare wrongly
    for anyone pinning the dependency.
    """
    assert _VERSION_RE.match(jax_nufft.__version__), (
        f"__version__ is {jax_nufft.__version__!r}, which is not a plain PEP 440 "
        "release string (expected e.g. '0.2.0', optionally with a pre/post/dev "
        "suffix, and never a leading 'v')."
    )
