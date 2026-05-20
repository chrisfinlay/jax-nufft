"""Type aliases used across the jax-nufft public API."""

from __future__ import annotations

from typing import Literal

from jax import Array

RealArray = Array
ComplexArray = Array

# Strategy for traversing w-planes inside dirty2vis / vis2dirty.
# v0.1.1 renamed ``scan`` -> ``dense_scan`` and ``vmap`` -> ``dense_vmap``,
# anticipating ``windowed_scan`` / ``windowed_vmap`` in a follow-up. The
# old names are accepted as deprecated aliases for one release. v0.1.2
# adds ``"auto"`` which the public wrappers resolve to a canonical name
# before reaching the JIT boundary.
WStrategy = Literal[
    "dense_scan",
    "dense_vmap",
    "windowed_scan",
    "windowed_vmap",
    "auto",
    "scan",
    "vmap",
]

# Strategy for traversing channels.
ChannelStrategy = Literal["vmap", "scan"]
