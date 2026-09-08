# The single source of truth for the package version: hatchling reads this file
# via [tool.hatch.version] in pyproject.toml, so the wheel, the sdist,
# ``pip show`` and ``jax_nufft.__version__`` cannot disagree (issue #59).
#
# ``.dev0`` because 0.2.0 is not released yet. The previous value, 0.1.2, named
# a version that was developed and merged but never tagged, so the package
# reported a number matching no release; ``0.2.0.dev0`` sorts before 0.2.0 and
# says plainly that this is work in progress. The release step is to drop the
# suffix here, and nowhere else.
__version__ = "0.2.0.dev0"
