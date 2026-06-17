"""JAX traceability tests: jit, grad (forward + reverse), vmap.

Spec sec 7.3: the wgridder operators must compose with jax.jit, jax.grad, and
jax.vmap without falling back to host execution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty

jax.config.update("jax_enable_x64", True)


def _tiny_setup(seed: int = 0):
    """Small problem used by all the integration tests."""
    rng = np.random.default_rng(seed)
    n_l = n_m = 16
    n_rows = 24
    pixsize = 0.005
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-3.0, 3.0, size=n_rows)
    freq = np.array([1.4e9])
    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6)
    image = rng.standard_normal((1, n_l, n_m)) + 1j * rng.standard_normal((1, n_l, n_m))
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )
    return plan, jnp.asarray(image), jnp.asarray(vis)


def test_jit_idempotent_with_eager() -> None:
    """Wrapping the already-jitted operators with another jax.jit must not change output.

    ``dirty2vis`` (a type-2 NUFFT / interpolation) is bit-reproducible,
    so it is checked exactly. ``vis2dirty`` (a type-1 NUFFT / spreading)
    accumulates via a parallel scatter-add whose reduction order is not
    fixed across calls on a multithreaded FINUFFT CPU build, so it is
    checked with ``allclose`` at a tolerance just above the measured
    run-to-run jitter (max ~8e-12 relative over 30 trials on the 72-core
    Grace CPU; a real jit-vs-eager bug would be orders of magnitude
    larger). rtol=1e-10 leaves ~12x headroom over that floor.
    """
    plan, image, vis = _tiny_setup(0)

    eager_vis = dirty2vis(plan, image)
    jitted_vis = jax.jit(dirty2vis)(plan, image)
    np.testing.assert_array_equal(np.asarray(jitted_vis), np.asarray(eager_vis))

    eager_dirty = vis2dirty(plan, vis)
    jitted_dirty = jax.jit(vis2dirty)(plan, vis)
    np.testing.assert_allclose(
        np.asarray(jitted_dirty), np.asarray(eager_dirty), rtol=1e-10, atol=1e-11
    )


def test_grad_of_dirty2vis_finite_difference() -> None:
    """jax.grad of a real scalar built from dirty2vis matches finite differences."""
    plan, image, _ = _tiny_setup(1)
    image_real = image.real  # use real-valued image so the loss is purely a function of image_real

    def loss(im_real):
        vis = dirty2vis(plan, im_real)
        return jnp.sum(vis.real**2 + vis.imag**2)

    grad_fn = jax.grad(loss)
    g_jax = np.asarray(grad_fn(image_real))

    # Pick a few random pixels for finite differences (full-grid FD is too slow).
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(image_real.size, size=8, replace=False)
    image_flat = np.asarray(image_real).ravel()
    h = 1e-5
    fd = np.zeros_like(image_flat)
    for k in sample_idx:
        bumped = image_flat.copy()
        bumped[k] += h
        plus = float(loss(jnp.asarray(bumped.reshape(image_real.shape))))
        bumped[k] -= 2 * h
        minus = float(loss(jnp.asarray(bumped.reshape(image_real.shape))))
        fd[k] = (plus - minus) / (2 * h)
    fd_at_sample = fd[sample_idx]
    grad_at_sample = g_jax.ravel()[sample_idx]
    np.testing.assert_allclose(grad_at_sample, fd_at_sample, rtol=1e-4, atol=1e-6)


def test_grad_of_vis2dirty_finite_difference() -> None:
    """Reverse-mode AD through vis2dirty matches finite differences in vis.

    For a real-valued loss with a complex input, jax.grad returns the
    "conjugate Wirtinger" gradient ``g = dL/d(re) - i * dL/d(im)``, so
    ``g.real`` matches a real-direction FD and ``-g.imag`` matches an
    imag-direction FD.
    """
    plan, _, vis = _tiny_setup(2)

    def loss(v):
        dirty = vis2dirty(plan, v)
        return jnp.sum(dirty**2)

    g_jax = np.asarray(jax.grad(loss)(vis.astype(jnp.complex128)))

    rng = np.random.default_rng(43)
    sample_idx = rng.choice(vis.size, size=6, replace=False)
    vis_flat = np.asarray(vis).ravel()
    h = 1e-5
    fd_re_list = []
    fd_im_list = []
    for k in sample_idx:
        bumped = vis_flat.copy()
        bumped[k] += h
        plus_re = float(loss(jnp.asarray(bumped.reshape(vis.shape))))
        bumped[k] -= 2 * h
        minus_re = float(loss(jnp.asarray(bumped.reshape(vis.shape))))
        fd_re_list.append((plus_re - minus_re) / (2 * h))

        bumped = vis_flat.copy()
        bumped[k] += 1j * h
        plus_im = float(loss(jnp.asarray(bumped.reshape(vis.shape))))
        bumped[k] -= 2j * h
        minus_im = float(loss(jnp.asarray(bumped.reshape(vis.shape))))
        fd_im_list.append((plus_im - minus_im) / (2 * h))

    fd_re = np.asarray(fd_re_list)
    fd_im = np.asarray(fd_im_list)
    g_at_sample = g_jax.ravel()[sample_idx]
    np.testing.assert_allclose(g_at_sample.real, fd_re, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(-g_at_sample.imag, fd_im, rtol=1e-3, atol=1e-6)


def test_vmap_over_image_batch() -> None:
    """vmap over a stack of 4 images stacks vis correctly."""
    plan, image, _ = _tiny_setup(3)
    batch_n = 4
    rng = np.random.default_rng(99)
    images = jnp.stack(
        [
            jnp.asarray(rng.standard_normal(image.shape) + 1j * rng.standard_normal(image.shape))
            for _ in range(batch_n)
        ],
        axis=0,
    )

    vis_vmap = jax.vmap(lambda im: dirty2vis(plan, im))(images)
    assert vis_vmap.shape == (batch_n, plan.n_rows, plan.n_chan)

    # Compare against independent calls.
    vis_loop = jnp.stack([dirty2vis(plan, images[k]) for k in range(batch_n)], axis=0)
    np.testing.assert_allclose(np.asarray(vis_vmap), np.asarray(vis_loop), rtol=1e-10)


def test_vmap_over_vis_batch() -> None:
    """vmap over a stack of vis arrays produces correctly stacked dirty output."""
    plan, _, vis = _tiny_setup(4)
    batch_n = 3
    rng = np.random.default_rng(101)
    vis_batch = jnp.stack(
        [
            jnp.asarray(
                rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape)
            ).astype(jnp.complex128)
            for _ in range(batch_n)
        ],
        axis=0,
    )

    dirty_vmap = jax.vmap(lambda v: vis2dirty(plan, v))(vis_batch)
    assert dirty_vmap.shape == (batch_n, plan.n_chan, plan.n_l, plan.n_m)
    dirty_loop = jnp.stack([vis2dirty(plan, vis_batch[k]) for k in range(batch_n)], axis=0)
    np.testing.assert_allclose(np.asarray(dirty_vmap), np.asarray(dirty_loop), rtol=1e-10)


def test_grad_through_pipeline_works() -> None:
    """Sanity: grad through both dirty2vis and vis2dirty composed (closed loop)."""
    plan, image, _ = _tiny_setup(5)
    image_real = image.real

    def loss(im_real):
        vis = dirty2vis(plan, im_real)
        round_trip = vis2dirty(plan, vis)
        return jnp.sum((round_trip - im_real) ** 2)

    g = jax.grad(loss)(image_real)
    assert g.shape == image_real.shape
    assert jnp.all(jnp.isfinite(g))


def test_jit_static_strategy_args() -> None:
    """w_strategy / channel_strategy must be static (passing them through jit)."""
    plan, image, _ = _tiny_setup(6)

    @jax.jit
    def vmap_call(im):
        return dirty2vis(plan, im, w_strategy="dense_vmap", channel_strategy="vmap")

    out = vmap_call(image)
    assert out.shape == (plan.n_rows, plan.n_chan)


def test_w_strategy_aliases_emit_deprecation() -> None:
    """v0.1 names ``scan``/``vmap`` still work but warn."""
    plan, image, _ = _tiny_setup(6)
    with pytest.warns(DeprecationWarning, match=r"scan.*deprecated"):
        out = dirty2vis(plan, image, w_strategy="scan")
    assert out.shape == (plan.n_rows, plan.n_chan)
    with pytest.warns(DeprecationWarning, match=r"vmap.*deprecated"):
        out = dirty2vis(plan, image, w_strategy="vmap")
    assert out.shape == (plan.n_rows, plan.n_chan)
