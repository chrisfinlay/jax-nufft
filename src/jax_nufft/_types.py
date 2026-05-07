"""Type aliases used across the jax-nufft public API."""

from __future__ import annotations

from typing import Literal

from jax import Array

RealArray = Array
ComplexArray = Array

# Strategy for traversing w-planes inside dirty2vis / vis2dirty.
WStrategy = Literal["vmap", "scan"]

# Strategy for traversing channels.
ChannelStrategy = Literal["vmap", "scan"]
