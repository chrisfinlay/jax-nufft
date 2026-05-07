"""Trivial import-level smoke tests so the scaffold can pass CI immediately."""

from __future__ import annotations

import jax_nufft


def test_version_string() -> None:
    assert isinstance(jax_nufft.__version__, str)
    assert jax_nufft.__version__


def test_public_api_names() -> None:
    assert hasattr(jax_nufft, "dirty2vis")
    assert hasattr(jax_nufft, "vis2dirty")
