# The single source of truth for the package version: hatchling reads this file
# via [tool.hatch.version] in pyproject.toml, so the wheel, the sdist,
# ``pip show`` and ``jax_nufft.__version__`` cannot disagree (issue #59).
#
# Development between releases carries a ``.dev0`` suffix, dropped here and
# nowhere else when a release is cut. The value before 0.2.0 was 0.1.2, naming
# a version that was developed and merged but never tagged, so the package
# reported a number matching no release; that is what #59 fixed by making this
# file the single source, and what .github/scripts/check_release_version.py now
# enforces against the git tag at publish time.
__version__ = "0.2.0"
