"""Forward and adjoint wgridder operators.

The full implementation lands in steps 4-7 of the plan; this module currently
exposes the public API names so that downstream code can import them, but
calling either function raises ``NotImplementedError``.
"""

from __future__ import annotations


def dirty2vis(*args, **kwargs):  # noqa: D401 - public API placeholder
    """Forward wgridder: image cube -> visibilities. Not yet implemented."""
    raise NotImplementedError("dirty2vis will be implemented in step 4 of the plan.")


def vis2dirty(*args, **kwargs):  # noqa: D401 - public API placeholder
    """Adjoint wgridder: visibilities -> image cube. Not yet implemented."""
    raise NotImplementedError("vis2dirty will be implemented in step 5 of the plan.")
