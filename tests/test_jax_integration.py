"""JAX traceability tests: jit, grad (forward + reverse), vmap.

Spec sec 7.3: the wgridder operators must compose with jax.jit, jax.grad, and
jax.vmap without falling back to host execution.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_nufft.wgridder as wgridder
from jax_nufft import dirty2vis, make_plan, vis2dirty

jax.config.update("jax_enable_x64", True)


def _setup_with_n_rows(n_rows: int, seed: int = 0):
    """Like ``_tiny_setup`` but with a caller-chosen ``n_rows``, so tests can
    straddle ``_NTHREADS_SMALL_N_ROWS`` deliberately. ``n_l``/``n_m`` stay
    tiny regardless of ``n_rows`` -- ``make_plan`` is host-side numpy work
    (fast even at 100k+ rows, see the boundary tests below), and the tests
    that use this stub out the JIT call entirely, so image/plan size never
    drives execution cost."""
    rng = np.random.default_rng(seed)
    n_l = n_m = 16
    pixsize = 0.005
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-3.0, 3.0, size=n_rows)
    freq = np.array([1.4e9])
    # hermitian=False (issue #17): the image below is complex, which the
    # conjugate-symmetry fold does not apply to -- a folded plan refuses a
    # complex image rather than returning a silently wrong answer. These tests
    # are about jit/vmap/grad plumbing and the nthreads resolution, not about
    # the fold, so they keep their complex image and pin the setting explicitly
    # rather than depending on make_plan's default. Traceability of the FOLDED
    # operators is covered by
    # tests/test_hermitian.py::test_folded_operators_stay_traceable.
    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6, hermitian=False)
    image = rng.standard_normal((1, n_l, n_m)) + 1j * rng.standard_normal((1, n_l, n_m))
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )
    return plan, jnp.asarray(image), jnp.asarray(vis)


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
    # hermitian=False (issue #17): the image below is complex, which the
    # conjugate-symmetry fold does not apply to -- a folded plan refuses a
    # complex image rather than returning a silently wrong answer. These tests
    # are about jit/vmap/grad plumbing and the nthreads resolution, not about
    # the fold, so they keep their complex image and pin the setting explicitly
    # rather than depending on make_plan's default. Traceability of the FOLDED
    # operators is covered by
    # tests/test_hermitian.py::test_folded_operators_stay_traceable.
    plan = make_plan(uvw, freq, (n_l, n_m), pixsize, pixsize, epsilon=1e-6, hermitian=False)
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


def test_grad_matches_finite_differences_smoke() -> None:
    """One cheap central difference, kept for documentation (issue #22).

    This replaces two tests that checked ``jax.grad`` against finite differences
    at ``rtol=1e-4`` on eight sampled pixels and ``rtol=1e-3`` on six. Central
    differences at ``h=1e-5`` truncate at ``O(h^2) = 1e-10`` in exact arithmetic
    and lose about half the mantissa to cancellation, so those bounds were some
    seven orders looser than the quantity they measured -- wide enough to admit a
    wrong conjugation, which is exactly the defect #21's first prototype had.

    What actually pins the gradients now, all exact identities rather than
    numerical differentiation, in ``tests/test_custom_vjp.py``:

    * ``test_the_gradient_of_half_the_squared_norm_is_the_normal_equations``
      and ``..._of_a_complex_image_loss_is_the_conjugated_normal_equations`` --
      ``grad(0.5||Ax||^2)`` against ``Re(A^H A x)`` and ``conj(A^H A x)``, at
      1e-11, with contrast assertions that separate the candidate conventions.
    * ``test_the_complex_image_cotangent_is_the_plain_transpose`` and
      ``test_the_vis2dirty_cotangent_is_the_conjugated_forward`` -- the
      conjugation convention itself.
    * the ``check_grads`` cells, ``modes=("fwd", "rev")`` over both flags, a
      complex image, and weights.

    This one stays because a reader wants to see, once and concretely, that the
    gradient is the derivative of the thing the operator computes -- and because
    it is the only place the Wirtinger sign convention is stated in the units a
    numerical check works in: ``g.real`` against a real-direction difference and
    ``-g.imag`` against an imaginary-direction one, ``g`` being the conjugate
    Wirtinger gradient ``dL/d(re) - i dL/d(im)``.

    (This module is in ``conftest.collect_ignore`` when ``JAX_ENABLE_X64=0``, so
    this cell runs on the float64 leg only.)
    """
    plan, _, vis = _tiny_setup(2)

    def loss(v):
        return jnp.sum(vis2dirty(plan, v) ** 2)

    g = np.asarray(jax.grad(loss)(vis.astype(jnp.complex128)))

    k = 3
    h = 1e-5
    flat = np.asarray(vis).ravel()

    def bumped(delta):
        out = flat.copy()
        out[k] += delta
        return float(loss(jnp.asarray(out.reshape(vis.shape))))

    fd_re = (bumped(h) - bumped(-h)) / (2 * h)
    fd_im = (bumped(1j * h) - bumped(-1j * h)) / (2 * h)

    np.testing.assert_allclose(g.ravel()[k].real, fd_re, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(-g.ravel()[k].imag, fd_im, rtol=1e-4, atol=1e-6)


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


# The channel-strategy comparison bound (issue #14). ``scan`` and ``vmap``
# walk the *same* per-channel helper in the same order over the same inputs;
# they are two lowerings of one reduction, not two reductions, so what
# separates them is floating point and nothing else. 1e-11 is the repo's
# eps-independent strategy-equivalence bound (AGENTS.md sec 6), reused here
# rather than reinvented.
CHANNEL_STRATEGY_TOL = 1e-11


def _multichannel_setup(seed: int = 0):
    """A small **multi-channel, non-square** problem with per-channel images.

    ``_tiny_setup`` is single-channel and square, which is exactly the
    configuration in which the two ``channel_strategy`` values cannot be told
    apart: with one channel a scan over a length-1 axis and a vmap over a
    length-1 axis produce the same graph up to naming, and neither can
    mis-associate a channel because there is only one thing to associate.
    Three channels at distinct frequencies, with independent image content per
    channel, is the smallest fixture where the association is falsifiable.

    ``n_l != n_m`` for the same reason, one axis over: a square grid cannot
    distinguish an ``[l, m]``-indexed quantity from an ``[m, l]``-indexed one,
    and this module's other fixtures are all 16 x 16.
    """
    rng = np.random.default_rng(seed)
    n_l, n_m = 12, 20
    n_rows = 32
    n_chan = 3
    pixsize_l = 0.005
    pixsize_m = 0.008
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-8.0, 8.0, size=n_rows)
    freq = np.array([0.9, 1.0, 1.15]) * 1.4e9
    plan = make_plan(uvw, freq, (n_l, n_m), pixsize_l, pixsize_m, epsilon=1e-6, hermitian=False)
    image = rng.standard_normal((n_chan, n_l, n_m)) + 1j * rng.standard_normal((n_chan, n_l, n_m))
    vis = (
        rng.standard_normal((n_rows, n_chan)) + 1j * rng.standard_normal((n_rows, n_chan))
    ).astype(np.complex128)
    return plan, jnp.asarray(image), jnp.asarray(vis)


def test_jit_static_strategy_args() -> None:
    """w_strategy / channel_strategy must be static (passing them through jit),
    and ``channel_strategy="vmap"`` must agree with ``"scan"`` *numerically*.

    The shape assertion is the original content of this test and is kept: it is
    what says the strategy names survived the ``jit`` boundary as static
    arguments rather than being traced. On its own, though, it is a weak
    statement about ``vmap`` -- the output shape of ``dirty2vis`` is derived
    from the plan, not from the channel loop (since issue #21 it is literally
    the primitive's abstract eval, ``(n_rows, n_chan)``), so a ``vmap`` branch
    that computed the wrong numbers, or the right numbers in the wrong channel
    order, would return the right shape and pass.

    So the value comparison is the substance. It needs a multi-channel plan
    (see ``_multichannel_setup``): at ``n_chan == 1`` the two branches are
    indistinguishable by construction and this assertion would hold for any
    implementation of either. Both operators, since the two channel loops are
    written separately -- the forward maps/scans over ``(image,
    plan.inv_lambda)`` and transposes the stacked result, the adjoint over
    ``(vis_per_chan, plan.inv_lambda)`` and does not.

    Related but not the same: ``tests/test_strategies_equivalent.py`` compares
    all eight ``(w_strategy, channel_strategy)`` combinations pairwise on the
    telescope fixtures, so the numeric claim here is not the suite's only one.
    What this cell adds is the claim *inside* ``jit`` on a non-square grid, and
    a home for it next to the staticness assertion it strengthens.
    """
    plan, image, _ = _tiny_setup(6)

    @jax.jit
    def vmap_call(im):
        return dirty2vis(plan, im, w_strategy="dense_vmap", channel_strategy="vmap")

    out = vmap_call(image)
    assert out.shape == (plan.n_rows, plan.n_chan)

    mc_plan, mc_image, mc_vis = _multichannel_setup(6)
    assert mc_plan.n_chan > 1, "a single-channel plan cannot tell the two branches apart"
    assert mc_plan.n_l != mc_plan.n_m

    # ``static_argnums=1`` rather than two separate closures: passing the
    # strategy name *through* ``jit`` as a static argument is the same claim
    # the shape assertion above makes, now made once per branch.
    @functools.partial(jax.jit, static_argnums=1)
    def forward(im, channel_strategy):
        return dirty2vis(mc_plan, im, w_strategy="dense_vmap", channel_strategy=channel_strategy)

    @functools.partial(jax.jit, static_argnums=1)
    def adjoint(v, channel_strategy):
        return vis2dirty(mc_plan, v, w_strategy="dense_vmap", channel_strategy=channel_strategy)

    fwd_scan = np.asarray(forward(mc_image, "scan"))
    fwd_vmap = np.asarray(forward(mc_image, "vmap"))
    adj_scan = np.asarray(adjoint(mc_vis, "scan"))
    adj_vmap = np.asarray(adjoint(mc_vis, "vmap"))

    assert fwd_vmap.shape == (mc_plan.n_rows, mc_plan.n_chan)
    assert adj_vmap.shape == (mc_plan.n_chan, mc_plan.n_l, mc_plan.n_m)

    # ``channel_axis`` differs between the two: ``dirty2vis`` returns
    # ``(n_rows, n_chan)`` and ``vis2dirty`` ``(n_chan, n_l, n_m)``.
    for name, got, want, channel_axis in (
        ("dirty2vis", fwd_vmap, fwd_scan, -1),
        ("vis2dirty", adj_vmap, adj_scan, 0),
    ):
        err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        assert err < CHANNEL_STRATEGY_TOL, (
            f'{name}: channel_strategy="vmap" differs from "scan" by {err:.3e} '
            f"(tol {CHANNEL_STRATEGY_TOL:.1e}) on a {mc_plan.n_chan}-channel "
            f"{mc_plan.n_l}x{mc_plan.n_m} plan -- the two are the same reduction in the "
            "same order, so anything above the rounding floor is a wiring difference"
        )
        # Per channel too: the norm above is dominated by the loudest channel,
        # and a permuted channel axis is exactly the defect that hides there
        # when the per-channel magnitudes are comparable.
        got_by_chan = np.moveaxis(got, channel_axis, 0)
        want_by_chan = np.moveaxis(want, channel_axis, 0)
        for c in range(mc_plan.n_chan):
            c_err = float(
                np.linalg.norm(got_by_chan[c] - want_by_chan[c]) / np.linalg.norm(want_by_chan[c])
            )
            assert c_err < CHANNEL_STRATEGY_TOL, (
                f'{name}: channel_strategy="vmap" differs from "scan" by {c_err:.3e} '
                f"on channel {c} (tol {CHANNEL_STRATEGY_TOL:.1e})"
            )


@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_dirty2vis_nthreads_invariant(w_strategy: str) -> None:
    """``nthreads`` is a scheduling knob, not a semantic one: dirty2vis at
    ``nthreads=None`` (issue #24's new default) and ``nthreads=2`` must
    match ``nthreads=1`` exactly to within 1e-11, across every w_strategy.

    dirty2vis is a type-2 NUFFT (interpolation), which has no thread-order-
    dependent reduction, so in principle this could be checked exactly; the
    1e-11 bound is kept for parity with the adjoint check below and with
    the issue's stated contract. Against the current default (``nthreads:
    int = 0``, no ``None`` handling), the ``nthreads=None`` case fails
    immediately with a jax-finufft ``Opts`` validation error rather than a
    numerical mismatch -- that failure is the point: this test cannot pass
    until ``nthreads: int | None = None`` resolves before the JIT boundary.
    """
    plan, image, _ = _tiny_setup(11)
    reference = np.asarray(dirty2vis(plan, image, w_strategy=w_strategy, nthreads=1))
    for nthreads in (None, 2):
        out = np.asarray(dirty2vis(plan, image, w_strategy=w_strategy, nthreads=nthreads))
        np.testing.assert_allclose(
            out,
            reference,
            atol=1e-11,
            rtol=0,
            err_msg=(
                f"dirty2vis(nthreads={nthreads}) vs nthreads=1 mismatch "
                f"for w_strategy={w_strategy!r}"
            ),
        )


@pytest.mark.parametrize(
    "w_strategy", ["dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap"]
)
def test_vis2dirty_nthreads_invariant(w_strategy: str) -> None:
    """Adjoint counterpart of ``test_dirty2vis_nthreads_invariant``.

    vis2dirty is a type-1 NUFFT (parallel scatter-add), so its reduction
    order is not fixed across calls on a multithreaded FINUFFT CPU build --
    ``nthreads=None``/``1``/``2`` are expected to differ only at that
    floor, which ``test_jit_idempotent_with_eager`` above measured at up to
    ~8e-12 relative on a 72-core build; 1e-11 leaves headroom over that
    while still catching a strategy/threading wiring bug (which would be
    O(1), not O(1e-11)).
    """
    plan, _, vis = _tiny_setup(12)
    reference = np.asarray(vis2dirty(plan, vis, w_strategy=w_strategy, nthreads=1))
    for nthreads in (None, 2):
        out = np.asarray(vis2dirty(plan, vis, w_strategy=w_strategy, nthreads=nthreads))
        np.testing.assert_allclose(
            out,
            reference,
            atol=1e-11,
            rtol=0,
            err_msg=(
                f"vis2dirty(nthreads={nthreads}) vs nthreads=1 mismatch "
                f"for w_strategy={w_strategy!r}"
            ),
        )


def test_w_strategy_aliases_emit_deprecation() -> None:
    """v0.1 names ``scan``/``vmap`` still work but warn."""
    plan, image, _ = _tiny_setup(6)
    with pytest.warns(DeprecationWarning, match=r"scan.*deprecated"):
        out = dirty2vis(plan, image, w_strategy="scan")
    assert out.shape == (plan.n_rows, plan.n_chan)
    with pytest.warns(DeprecationWarning, match=r"vmap.*deprecated"):
        out = dirty2vis(plan, image, w_strategy="vmap")
    assert out.shape == (plan.n_rows, plan.n_chan)


# -- boundary tests: the exact ``nthreads`` reaching the JIT boundary -------
#
# ``test_dirty2vis_nthreads_invariant`` / ``test_vis2dirty_nthreads_invariant``
# above only check that different ``nthreads`` values produce numerically
# equivalent output -- exactly the property that would still hold if the
# wrapper ignored resolution entirely and always forwarded a fixed value
# (e.g. ``1``). They also only ever use ``_tiny_setup``'s 24-row plan, which
# is far below ``_NTHREADS_SMALL_N_ROWS`` (100_000), so the vmap-family arm
# of the resolution rule (-> ``0``) is never exercised through the public
# wrappers, only through the isolated ``_resolve_nthreads`` unit tests in
# ``tests/test_nthreads_resolution.py``. The tests below close that gap by
# spying on the internal ``_dirty2vis_jit`` / ``_vis2dirty_jit`` functions
# (whose ``nthreads`` is a ``static_argname``) and asserting the *exact*
# value ``dirty2vis`` / ``vis2dirty`` hand them, across the cases the
# resolution rule branches on. The JIT/FINUFFT call itself is stubbed out
# (returning a dummy array) so a 100k+-row "above cutoff" plan costs only
# the host-side ``make_plan`` build (~tens of ms), not a real transform.


def _spy_jit(monkeypatch: pytest.MonkeyPatch, jit_name: str) -> list[dict]:
    """Replace ``jax_nufft.wgridder.<jit_name>`` with a stub that records
    every call's kwargs and returns a dummy array (the public wrapper
    returns the JIT function's result directly, with no further
    processing, so any return value is fine)."""
    calls: list[dict] = []

    def stub(*args: object, **kwargs: object) -> jax.Array:
        calls.append(kwargs)
        return jnp.zeros(())

    monkeypatch.setattr(wgridder, jit_name, stub)
    return calls


def test_dirty2vis_default_nthreads_reaches_jit_scan_below_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan-family strategy on a plan below the cutoff must resolve to
    ``nthreads=1`` at the point it reaches ``_dirty2vis_jit``."""
    calls = _spy_jit(monkeypatch, "_dirty2vis_jit")
    plan, image, _ = _tiny_setup(6)  # 24 rows, well below the 100_000 cutoff
    dirty2vis(plan, image, w_strategy="dense_scan")
    assert len(calls) == 1
    assert calls[0]["nthreads"] == 1


def test_dirty2vis_default_nthreads_reaches_jit_vmap_above_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vmap-family strategy on a plan *above* the cutoff must resolve to
    ``nthreads=0`` at the point it reaches ``_dirty2vis_jit`` -- the one
    combination the numerical-invariance tests above cannot exercise."""
    calls = _spy_jit(monkeypatch, "_dirty2vis_jit")
    n_rows = wgridder._NTHREADS_SMALL_N_ROWS + 1
    plan, image, _ = _setup_with_n_rows(n_rows, seed=6)
    dirty2vis(plan, image, w_strategy="dense_vmap")
    assert len(calls) == 1
    assert calls[0]["nthreads"] == 0


def test_vis2dirty_default_nthreads_reaches_jit_scan_below_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjoint counterpart of the scan-below-cutoff test above."""
    calls = _spy_jit(monkeypatch, "_vis2dirty_jit")
    plan, _, vis = _tiny_setup(7)
    vis2dirty(plan, vis, w_strategy="windowed_scan")
    assert len(calls) == 1
    assert calls[0]["nthreads"] == 1


def test_vis2dirty_default_nthreads_reaches_jit_vmap_above_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjoint counterpart of the vmap-above-cutoff test above."""
    calls = _spy_jit(monkeypatch, "_vis2dirty_jit")
    n_rows = wgridder._NTHREADS_SMALL_N_ROWS + 1
    plan, _, vis = _setup_with_n_rows(n_rows, seed=7)
    vis2dirty(plan, vis, w_strategy="windowed_vmap")
    assert len(calls) == 1
    assert calls[0]["nthreads"] == 0


def test_dirty2vis_default_nthreads_reaches_jit_auto_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``w_strategy="auto"`` must canonicalise (via ``plan``/``is_adjoint``)
    and resolve all the way through to a concrete ``nthreads`` at the JIT
    boundary, not raise and not leak ``None`` through. On CPU, ``"auto"``
    always resolves to a scan-family strategy (``_auto_w_strategy_cpu``),
    and this plan is below the cutoff either way, so ``1`` is expected
    regardless of which scan variant is picked."""
    calls = _spy_jit(monkeypatch, "_dirty2vis_jit")
    plan, image, _ = _tiny_setup(6)
    dirty2vis(plan, image, w_strategy="auto")
    assert len(calls) == 1
    assert calls[0]["nthreads"] == 1


def test_dirty2vis_default_nthreads_reaches_jit_deprecated_scan_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deprecated ``"scan"`` alias (-> ``dense_scan``) must resolve the
    same as its canonical name at the JIT boundary."""
    calls = _spy_jit(monkeypatch, "_dirty2vis_jit")
    plan, image, _ = _tiny_setup(6)
    with pytest.warns(DeprecationWarning, match=r"scan.*deprecated"):
        dirty2vis(plan, image, w_strategy="scan")
    assert len(calls) == 1
    assert calls[0]["nthreads"] == 1


def test_dirty2vis_default_nthreads_reaches_jit_deprecated_vmap_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deprecated ``"vmap"`` alias (-> ``dense_vmap``) must resolve the
    same as its canonical name at the JIT boundary -- on a plan above the
    cutoff, so this is distinguishable from the (also correct) ``1`` a
    below-cutoff plan would give."""
    calls = _spy_jit(monkeypatch, "_dirty2vis_jit")
    n_rows = wgridder._NTHREADS_SMALL_N_ROWS + 1
    plan, image, _ = _setup_with_n_rows(n_rows, seed=6)
    with pytest.warns(DeprecationWarning, match=r"vmap.*deprecated"):
        dirty2vis(plan, image, w_strategy="vmap")
    assert len(calls) == 1
    assert calls[0]["nthreads"] == 0


def test_dirty2vis_explicit_nthreads_reaches_jit_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``nthreads`` must reach ``_dirty2vis_jit`` completely
    untouched by the resolution rule -- including on a vmap-family,
    above-cutoff plan where the default would resolve to ``0``, to prove
    the explicit value (here ``5``) isn't coincidentally matching a
    resolved one."""
    calls = _spy_jit(monkeypatch, "_dirty2vis_jit")
    n_rows = wgridder._NTHREADS_SMALL_N_ROWS + 1
    plan, image, _ = _setup_with_n_rows(n_rows, seed=6)
    dirty2vis(plan, image, w_strategy="dense_vmap", nthreads=5)
    assert len(calls) == 1
    assert calls[0]["nthreads"] == 5
