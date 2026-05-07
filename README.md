# jax-nufft

JAX-native wgridder for radio interferometric imaging.

> **Status:** v0.1 in active development. API may change.

## Overview

`jax-nufft` provides a pure-JAX implementation of the wgridder algorithm for
radio interferometric imaging, expressed as a stack of 2D non-uniform FFTs
indexed by w-plane. It is built on top of [`jax-finufft`][jaxfinufft] for the
underlying NUFFT primitives.

The package is designed for radio interferometric pipelines that need to compose
the wgridder with other JAX-traceable operators — for example calibration,
self-cal, or amortised inference networks. Concretely, it supports:

- Full `jax.jit`, `jax.vmap`, and `jax.grad` traceability (forward and reverse
  mode).
- CPU execution out-of-the-box (Mac and Linux), and GPU execution via
  cuFINUFFT-enabled `jax-finufft` builds without any code changes.
- Multi-channel visibilities (`Nrow × Nchan`) with shared per-row `uvw` and
  per-channel frequency.
- Optional per-visibility weights of shape `(Nrow, Nchan)`.
- Output that matches `ducc0.wgridder` to within ~10× the requested accuracy
  `epsilon`, both forward (`dirty2vis`) and adjoint (`vis2dirty`).

Polarisation handling is **out of scope for v1**.

## Installation

> Detailed installation instructions follow once v0.1 is released. The package
> is intended to be installable via `pip install jax-nufft`, with a `pixi`
> workflow for reproducible development environments.

## Quick start

> A worked end-to-end example will be added once the v0.1 API is stable.

## Documentation

Mathematical background, API reference, and performance notes will be added
during v0.1 development. See `prompts/` and inline docstrings for now.

## License

Apache-2.0. See [`LICENSE`](LICENSE) for the full text.

`ducc0` is used as a test reference under GPL-2.0+ but is not a runtime
dependency of `jax-nufft`.

[jaxfinufft]: https://github.com/flatironinstitute/jax-finufft
