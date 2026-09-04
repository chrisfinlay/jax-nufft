"""Boundary-plane and pathological-w stress tests for the windowed strategies.

The windowed scheme assumes the contributing rows for any w-plane form a
*contiguous slice* of the w-sorted array. Most of the algorithmic interest
lives at the boundary planes (where the kernel support hangs off either
edge of the data range) and at pathologically clumped w-distributions
(where most planes have empty windows but one or two are densely packed).

These tests construct synthetic w-distributions that stress each regime
and confirm the windowed forward and adjoint agree with the dense
baseline.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._utils import SPEED_OF_LIGHT
from jax_nufft.planning import window_boundary_margin

jax.config.update("jax_enable_x64", True)


def _make_uvw(w_values: np.ndarray, *, seed: int) -> np.ndarray:
    """Build (n_rows, 3) uvw with given w values and random (u, v)."""
    rng = np.random.default_rng(seed)
    n_rows = w_values.shape[0]
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-100.0, 100.0, size=n_rows)
    uvw[:, 2] = w_values
    return uvw


def _parity_check(
    uvw: np.ndarray,
    eps: float,
    *,
    n_pix: int = 32,
    pixsize: float = 0.01,
    freq_hz: float = 1.4e9,
) -> None:
    """Forward + adjoint dense-vs-windowed parity at the given epsilon."""
    freq = np.array([freq_hz])
    rng = np.random.default_rng(101)
    image = rng.standard_normal((n_pix, n_pix))
    vis = (
        rng.standard_normal((uvw.shape[0], 1)) + 1j * rng.standard_normal((uvw.shape[0], 1))
    ).astype(np.complex128)

    plan = make_plan(uvw, freq, (n_pix, n_pix), pixsize, pixsize, eps)

    vis_dense = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy="dense_scan"))
    vis_windowed = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy="windowed_scan"))
    fwd_err = np.linalg.norm(vis_windowed - vis_dense) / max(np.linalg.norm(vis_dense), 1e-30)
    # Forward: scatter-add reduces identical contributing rows in identical
    # order, so the windowed path is bit-equal to dense (zero error).
    assert fwd_err < 1e-12, f"forward windowed-vs-dense err {fwd_err:.3e}"

    dirty_dense = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy="dense_scan"))
    dirty_windowed = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy="windowed_scan"))
    adj_err = np.linalg.norm(dirty_windowed - dirty_dense) / max(np.linalg.norm(dirty_dense), 1e-30)
    # Adjoint: the NUFFT type 1 sums depend on the set of rows in the
    # batch; the dense/windowed reductions differ in summation order, so
    # this is a reduction-order comparison, not an accuracy contract, and
    # the bound below is deliberately eps-independent rather than scaled by
    # the requested epsilon. Issue #10 measured the actual worst-case
    # pairwise disagreement between the four w_strategy values across these
    # boundary fixtures at 2e-12 (eps=1e-8) and <=5e-14 (eps=1e-6); the
    # former ``100 * eps`` bound (1e-4 at eps=1e-6) was ten orders of
    # magnitude looser than that and would not have caught a bug that
    # dropped a whole window row from the adjoint. 1e-11 is roughly
    # ``100 * n_rows * eps_machine`` and stays flat as eps tightens.
    assert adj_err < 1e-11, f"adjoint windowed-vs-dense err {adj_err:.3e} (eps={eps:g})"


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_boundary_clumped_at_w_min(eps: float) -> None:
    """All rows clumped near w_min: only the lowest few planes have non-empty windows."""
    rng = np.random.default_rng(0)
    n_rows = 200
    # Clump around -25 with a tiny spread; add a small range so n_w > W.
    w_values = rng.normal(loc=-25.0, scale=0.1, size=n_rows)
    # Sprinkle a handful of outliers to extend the w-extent so n_w > W
    w_values[:5] = rng.uniform(-25.0, 25.0, size=5)
    _parity_check(_make_uvw(w_values, seed=1), eps=eps)


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_boundary_clumped_at_w_max(eps: float) -> None:
    """Mirror: rows clumped near w_max."""
    rng = np.random.default_rng(2)
    n_rows = 200
    w_values = rng.normal(loc=+25.0, scale=0.1, size=n_rows)
    w_values[:5] = rng.uniform(-25.0, 25.0, size=5)
    _parity_check(_make_uvw(w_values, seed=3), eps=eps)


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_boundary_symmetric_around_zero(eps: float) -> None:
    """Most contribution concentrated near w=0; edges sparsely populated."""
    rng = np.random.default_rng(4)
    n_rows = 300
    w_values = rng.normal(loc=0.0, scale=5.0, size=n_rows)
    _parity_check(_make_uvw(w_values, seed=5), eps=eps)


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_boundary_bimodal(eps: float) -> None:
    """Bimodal w-distribution: two clumps far apart with empty middle planes."""
    rng = np.random.default_rng(6)
    n_rows = 400
    half = n_rows // 2
    w_values = np.empty(n_rows)
    w_values[:half] = rng.normal(loc=-20.0, scale=0.5, size=half)
    w_values[half:] = rng.normal(loc=+20.0, scale=0.5, size=n_rows - half)
    _parity_check(_make_uvw(w_values, seed=7), eps=eps)


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_boundary_uniform_baseline(eps: float) -> None:
    """Uniform w-distribution: middle planes well-populated, edges thin (baseline)."""
    rng = np.random.default_rng(8)
    n_rows = 300
    w_values = rng.uniform(-30.0, 30.0, size=n_rows)
    _parity_check(_make_uvw(w_values, seed=9), eps=eps)


def test_small_nw_zenith_regression() -> None:
    """Near-zero w-extent (zenith with no z offsets) gives n_w in the W-only regime.

    The plan flags this regime as a risk: when ``n_w`` is close to ``W``,
    every plane is an "edge plane" and the windowed scheme has to still
    produce correct results.
    """
    rng = np.random.default_rng(10)
    n_rows = 64
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-30.0, 30.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-30.0, 30.0, size=n_rows)
    # Tiny w-extent so n_w is dominated by the kernel width.
    uvw[:, 2] = rng.uniform(-0.05, 0.05, size=n_rows)
    eps = 1e-6
    plan = make_plan(uvw, np.array([1.4e9]), (32, 32), 0.01, 0.01, eps)
    # In this regime n_w should be at most a couple of planes plus W.
    assert plan.n_w <= plan.w_kernel_width + 2

    image = rng.standard_normal((32, 32))
    vis = (rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1))).astype(
        np.complex128
    )

    vis_dense = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy="dense_scan"))
    vis_windowed = np.asarray(dirty2vis(plan, jnp.asarray(image), w_strategy="windowed_scan"))
    assert np.allclose(vis_dense, vis_windowed, atol=1e-12, rtol=1e-12)

    dirty_dense = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy="dense_scan"))
    dirty_windowed = np.asarray(vis2dirty(plan, jnp.asarray(vis), w_strategy="windowed_scan"))
    err = np.linalg.norm(dirty_windowed - dirty_dense) / np.linalg.norm(dirty_dense)
    # Same eps-independent bound as ``_parity_check`` above (issue #10); see
    # its comment for the measured values behind 1e-11.
    assert err < 1e-11


# ---------------------------------------------------------------------------
# The window-builder invariant at an ulp of a window edge (issue #23)
# ---------------------------------------------------------------------------
#
# The windowed strategies are correct only if every window contains every row
# the dense path gives a non-zero kernel weight. Which rows those are is
# decided on the *device*, by ``wgridder._channel_ft_coords``; which rows the
# window holds is decided on the *host*, by ``make_plan``'s searchsorted. So
# the invariant lives on the seam between two evaluations of the same
# quantity, and issue #23 moved that seam.
#
# Before #23 the plan stored the per-channel w and the device merely subtracted
# ``w0`` from it, so host and device agreed bit for bit and the invariant was
# free. Since #23 the device derives w as a multiply immediately followed by a
# subtract, which XLA contracts into a single FMA -- one rounding where the
# host's numpy steps round twice. The two now differ in the last bits on a
# large fraction of rows (~23% on MWA_extended, ~18% on MeerKAT), and a row
# landing in that ulp-wide band on the wrong side of a window edge is kept by
# the dense path at ``phi(z = +/-1) = exp(-beta)`` and dropped entirely by the
# windowed one. Measured on the fixture below before the builder was widened:
# 3.694e-09 forward and 2.480e-09 adjoint against this file's 1e-11 contract,
# where ``origin/main`` is 1.5e-17 / 2.4e-15 on the identical inputs.
#
# make_plan now widens each boundary by ``boundary_margin`` (an upper bound on
# that host/device gap) and by one row at each end, which makes the invariant
# hold by construction: over-inclusion is free, since a row inside the slice
# but outside kernel support gets ``phi(|z| > 1) = 0``.


def _device_w_sorted(plan: object, uvw: np.ndarray, chan: int = 0) -> np.ndarray:
    """The relative w the *operator* computes, in sorted-row order.

    This is the production helper, not a transcription of it: which rows the
    dense path weights is decided by exactly this function, so the invariant
    below has to be checked against it and nothing else.
    """
    from jax_nufft.wgridder import _channel_ft_coords

    perm = np.asarray(plan.sort_perm)  # type: ignore[attr-defined]
    return np.asarray(
        jax.jit(_channel_ft_coords)(
            jnp.asarray(uvw[perm]),
            plan.inv_lambda[chan],  # type: ignore[attr-defined]
            plan,
        )[2]
    )


def _rows_dropped_by_windows(plan: object, uvw: np.ndarray) -> int:
    """Rows the dense path weights but the windowed slice cannot reach.

    Zero is the window-builder invariant (AGENTS.md sec 4). Mirrors the
    operator's own slice arithmetic, including ``_channel_*_windowed``'s clamp
    of the start index into ``[0, n_rows - max_window_size]``.
    """
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)  # type: ignore[attr-defined]
    scale = plan.w_kernel_scale  # type: ignore[attr-defined]
    starts = np.asarray(plan.window_start)  # type: ignore[attr-defined]
    size = plan.max_window_size  # type: ignore[attr-defined]
    n_rows = plan.n_rows  # type: ignore[attr-defined]
    dropped = 0
    for chan in range(plan.n_chan):  # type: ignore[attr-defined]
        w = _device_w_sorted(plan, uvw, chan)
        for k in range(centres.shape[0]):
            inside = np.nonzero(np.abs((w - centres[k]) / scale) <= 1.0)[0]
            if inside.size == 0:
                continue
            lo = min(max(int(starts[chan, k]), 0), max(n_rows - size, 0))
            dropped += int(np.sum((inside < lo) | (inside >= lo + size)))
    return dropped


def _unwidened_builder_drops(plan: object, uvw: np.ndarray, freq: np.ndarray) -> int:
    """Rows the *pre-fix* window builder would drop on this fixture.

    Mirrors ``make_plan``'s searchsorted loop as it stood before the widening --
    no ``boundary_margin``, no extra row at either end -- and counts rows that
    the production ``_channel_ft_coords`` puts inside kernel support but that
    those narrower windows could not reach.

    This is what selects the fixture, and it has to be, because the obvious
    criterion is circular: searching for a ``w`` that makes the *current*
    builder drop a row finds nothing once the builder is correct, and the test
    then skips or -- worse -- silently falls back to a fixture that gates
    nothing. Simulating the defective builder here keeps the search meaningful
    on a fixed build: it picks a fixture that provably defeats the old
    algorithm, and the assertions then check that the real one survives it.
    """
    perm = np.asarray(plan.sort_perm)  # type: ignore[attr-defined]
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)  # type: ignore[attr-defined]
    scale = plan.w_kernel_scale  # type: ignore[attr-defined]
    n_rows = plan.n_rows  # type: ignore[attr-defined]
    w_m_sorted = uvw[perm, 2].astype(np.float64)

    los, sizes = [], []
    for chan in range(plan.n_chan):  # type: ignore[attr-defined]
        w_host = (w_m_sorted * (float(freq[chan]) / SPEED_OF_LIGHT)).astype(np.float64) - plan.w0  # type: ignore[attr-defined]
        lo = np.searchsorted(w_host, centres - scale, side="left")
        hi = np.searchsorted(w_host, centres + scale, side="right")
        los.append(lo)
        sizes.append(hi - lo)
    max_window = max(1, int(np.max(np.asarray(sizes))))

    dropped = 0
    for chan in range(plan.n_chan):  # type: ignore[attr-defined]
        w_dev = _device_w_sorted(plan, uvw, chan)
        for k in range(centres.shape[0]):
            inside = np.nonzero(np.abs((w_dev - centres[k]) / scale) <= 1.0)[0]
            if inside.size == 0:
                continue
            lo = min(max(int(los[chan][k]), 0), max(n_rows - max_window, 0))
            dropped += int(np.sum((inside < lo) | (inside >= lo + max_window)))
    return dropped


def _find_invariant_breaking_w(
    plan: object, uvw: np.ndarray, freq: np.ndarray, plan_kwargs: dict, n_step: int = 12
) -> tuple[float, np.ndarray] | None:
    """Search for a fixture that defeats the *unwidened* window builder.

    Three conditions have to line up, which is why this searches rather than
    hard-codes. First the value must sit in the ulp-wide band where the host's
    two-rounding w and the device's FMA-contracted w fall on opposite sides of
    a window edge -- only there can the builder and the operator disagree about
    membership at all. Second, the row must not be rescued by padding: the slice
    is ``max_window_size`` long while the window is usually shorter, so most
    straddling rows are covered anyway. Third, the candidate is spliced into
    **one and then two adjacent** interior rows.

    That third condition is not a refinement, it is the point. The two halves of
    the widening are independent, and a single straddling row is rescued
    unconditionally by ``lo = max(lo - 1, 0)``: with ``boundary_margin`` deleted
    the one-row fixture still passes, so a search that only ever splices one row
    leaves the half the code calls provable completely unpinned. Two rows
    sharing the same straddling ``w`` exhaust the one-row slack, and the margin
    has to carry it.

    Returns ``(overhang, spliced_uvw)`` -- the amount by which the selected
    row's host w falls outside the bare boundary, and the fixture -- or ``None``
    when no candidate qualifies
    -- e.g. on a platform whose compiler does not contract the multiply-subtract,
    where host and device agree exactly and the invariant is trivially safe.
    """
    from jax_nufft.wgridder import _channel_ft_coords

    invl = float(np.asarray(plan.inv_lambda)[0])  # type: ignore[attr-defined]
    w0 = plan.w0  # type: ignore[attr-defined]
    scale = plan.w_kernel_scale  # type: ignore[attr-defined]
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)  # type: ignore[attr-defined]
    w_lo, w_hi = float(np.min(uvw[:, 2])), float(np.max(uvw[:, 2]))

    candidates: list[float] = []
    edges: list[tuple[str, float]] = []
    for k in range(1, centres.shape[0] - 1):
        for side, bound in (("lo", centres[k] - scale), ("hi", centres[k] + scale)):
            x = (bound + w0) / invl
            for _ in range(n_step // 2):
                x = np.nextafter(x, -np.inf)
            for _ in range(n_step):
                candidates.append(float(x))
                edges.append((side, float(bound)))
                x = np.nextafter(x, np.inf)

    cand = np.asarray(candidates, dtype=np.float64)
    host = (cand * invl).astype(np.float64) - w0
    zeros = np.zeros_like(cand)
    device = np.asarray(
        jax.jit(_channel_ft_coords)(
            jnp.asarray(np.stack([zeros, zeros, cand], axis=1)),
            plan.inv_lambda[0],  # type: ignore[attr-defined]
            plan,
        )[2]
    )

    # Take the *largest* straddle, not the first. Straddling only requires the
    # overhang to exceed zero, and the first candidate encountered exercised
    # 8.3% of the margin -- so the fixture was gating a fraction of what it
    # could. Ordering by |bound - host| costs nothing: the qualifying test
    # below is unchanged, it just runs on the most demanding candidates first.
    overhangs = np.abs(np.asarray([b for _, b in edges]) - host)
    order_by_overhang = np.argsort(-overhangs)

    order = np.argsort(uvw[:, 2])
    for i in order_by_overhang:
        side, bound = edges[i]
        # "lo": searchsorted(..., "left") drops rows below the bound, but the
        # dense path keeps any row the device puts at or above it. "hi" mirrors.
        straddles = host[i] < bound <= device[i] if side == "lo" else host[i] > bound >= device[i]
        if not straddles or not (w_lo < cand[i] < w_hi):
            continue
        # Splice into interior rows -- strictly inside the existing w-range, so
        # w_min / w_max and every plan quantity derived from them are untouched
        # and the only change is which side of one edge those rows are on.
        # ``n_splice = 2`` uses adjacent sorted positions, which stay adjacent in
        # the rebuilt plan since they share the same w.
        for n_splice in (2, 1):
            for first in (1, 2, len(order) // 2):
                if first + n_splice >= len(order):
                    continue  # never touch the max-w row
                spliced = uvw.copy()
                spliced[order[first : first + n_splice], 2] = float(cand[i])
                candidate_plan = make_plan(spliced, freq, **plan_kwargs)
                if candidate_plan.w0 != plan.w0 or candidate_plan.n_w != plan.n_w:  # type: ignore[attr-defined]
                    continue
                if _unwidened_builder_drops(candidate_plan, spliced, freq):
                    return float(overhangs[i]), spliced
    return None


def test_windowed_dense_parity_at_window_edge() -> None:
    """A row within an ulp of a window edge must not break windowed/dense parity.

    The fixture is adversarial and built in two passes: plan the base geometry,
    then search (:func:`_find_invariant_breaking_w`) for a ``w`` that both lands
    in the host/device disagreement band at a window edge *and* is not rescued
    by ``max_window_size`` padding, splice it into an interior row and re-plan.
    The splice leaves ``w0``, ``n_w`` and ``w_centers_rel`` untouched -- asserted
    below -- so the only thing that changed is which side of one window edge a
    single row sits on.

    Large ``|w0|`` (~2.3e5 wavelengths against a w-extent of ~4e3) is what makes
    the host/device gap reach an edge at all: the gap is bounded by an ulp of
    the *absolute* w, so a zenith fixture with ``w0 ~ 0`` cannot exhibit this
    however the rows fall. Windows also have to be a strict subset of the rows,
    or the windowed path trivially sees everything.

    Measured on this fixture before ``make_plan``'s boundaries were widened:
    3.694e-09 forward and 2.480e-09 adjoint, against the 1e-11 contract this
    file uses throughout and ``origin/main``'s 1.5e-17 / 2.4e-15 on identical
    inputs.

    If the search finds nothing the test does not quietly fall back to the base
    fixture -- it fails or skips, explicitly. See the ``found is None`` branch.
    """
    n_rows = 60
    freq = np.array([1.0e9])
    n_pix, pixsize, eps = 32, 0.004, 1e-6
    plan_kwargs = {
        "image_shape": (n_pix, n_pix),
        "pixsize_l": pixsize,
        "pixsize_m": pixsize,
        "epsilon": eps,
    }

    rng = np.random.default_rng(0)
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-120.0, 120.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-120.0, 120.0, size=n_rows)
    uvw[:, 2] = 5.0e4 + rng.uniform(-2.0e3, 2.0e3, size=n_rows)

    base = make_plan(uvw, freq, **plan_kwargs)
    assert base.max_window_size < n_rows, (
        "the windows must be a strict subset of the rows, or a dropped row "
        "cannot show up in the first place"
    )

    # The premise of the whole fixture, measured rather than assumed: does this
    # compiler's w disagree with the builder's at all? On every fixture in this
    # repository it disagrees on 18-22% of rows.
    perm = np.asarray(base.sort_perm)
    host_w = (uvw[perm, 2] * float(np.asarray(base.inv_lambda)[0])).astype(np.float64) - base.w0
    disagreement = float(np.max(np.abs(_device_w_sorted(base, uvw) - host_w)))

    found = _find_invariant_breaking_w(base, uvw, freq, plan_kwargs)
    if found is None:
        # Two very different situations, and conflating them is how a test like
        # this rots into a permanent skip.
        assert disagreement == 0.0, (
            f"host and device disagree by up to {disagreement:.3e} wavelengths on this "
            "fixture, so a straddling value should exist, but the search found none. "
            "The search has gone stale (plane geometry, candidate window, or splice "
            "positions), not the platform -- this test is no longer gating anything"
        )
        pytest.skip(  # pragma: no cover - platform-dependent
            "host and device compute w bit-for-bit identically on every row of this "
            "fixture, so there is no straddling value to build the adversarial case "
            "from and this test gates nothing on this platform. Note this does NOT "
            "mean the invariant is safe here: the 2u*S term in boundary_margin's "
            "derivation does not come from FMA contraction at all. That half is "
            "gated by test_window_boundary_margin_covers_host_device_gap"
        )
    selected_overhang, uvw_adv = found
    # The fixture has to exercise a real fraction of the margin, not merely a
    # non-zero one: a straddle needs only overhang > 0, and the first candidate
    # the search used to return exercised 8.3% of it. The scale to compare
    # against is an ulp of the absolute w, which is what the host/device
    # disagreement is bounded by in the first place.
    ulp_w_abs = float(
        np.spacing(abs(base.w0) + 0.5 * base.w_extent)  # type: ignore[arg-type]
    )
    assert selected_overhang >= 0.25 * ulp_w_abs, (
        f"the selected row overhangs its window edge by {selected_overhang:.3e} "
        f"wavelengths, under a quarter of ulp(|w|) = {ulp_w_abs:.3e}. The search is "
        "settling for a marginal straddle, so the fixture exercises far less of "
        "boundary_margin than it could"
    )
    # Never let the fixture silently degrade. Two conditions, because they fail
    # differently: a zero-row splice would run every assertion below against the
    # plain base plan, and a *one*-row splice is rescued unconditionally by the
    # builder's ``lo - 1`` and so gates ``boundary_margin`` not at all (measured:
    # one row with the margin off gives dropped=0 and this test passes).
    n_spliced = int(np.sum(uvw_adv[:, 2] != uvw[:, 2]))
    assert n_spliced >= 2, (
        f"the search returned a {n_spliced}-row splice; at least 2 adjacent rows must "
        "share the straddling w or the +/-1-row half of the widening rescues the "
        "fixture on its own and boundary_margin goes ungated"
    )

    plan = make_plan(uvw_adv, freq, **plan_kwargs)
    assert plan.w0 == base.w0
    assert plan.n_w == base.n_w
    np.testing.assert_array_equal(np.asarray(plan.w_centers_rel), np.asarray(base.w_centers_rel))

    # Measured up front so it can be quoted as the *diagnosis* alongside the
    # parity numbers below, which are the contract the library actually owes.
    dropped = _rows_dropped_by_windows(plan, uvw_adv)
    why = (
        f"{dropped} row(s) that the dense path weights at phi(z = +/-1) = exp(-beta) lie "
        "outside the windowed slice. See AGENTS.md sec 4's window-builder invariant: "
        "make_plan must widen its boundaries, because the operator's FMA-contracted w "
        "does not agree bit-for-bit with the builder's two-rounding one"
    )

    rng = np.random.default_rng(7)
    image = jnp.asarray(rng.standard_normal((n_pix, n_pix)))
    vis = jnp.asarray(rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1)))

    vis_dense = np.asarray(dirty2vis(plan, image, w_strategy="dense_scan"))
    dirty_dense = np.asarray(vis2dirty(plan, vis, w_strategy="dense_scan"))
    for strategy in ("windowed_scan", "windowed_vmap"):
        vis_win = np.asarray(dirty2vis(plan, image, w_strategy=strategy))
        dirty_win = np.asarray(vis2dirty(plan, vis, w_strategy=strategy))
        fwd = np.linalg.norm(vis_win - vis_dense) / np.linalg.norm(vis_dense)
        adj = np.linalg.norm(dirty_win - dirty_dense) / np.linalg.norm(dirty_dense)
        assert fwd < 1e-11, f"dirty2vis dense-vs-{strategy} rel L2 {fwd:.3e} > 1e-11: {why}"
        assert adj < 1e-11, f"vis2dirty dense-vs-{strategy} rel L2 {adj:.3e} > 1e-11: {why}"

    # The invariant itself, checked against the production coordinate helper.
    # Strictly stronger than the parity assertions above: a dropped row whose
    # kernel weight happens to underflow the norms would slip past those.
    assert dropped == 0, why


# ---------------------------------------------------------------------------
# The *magnitude* of boundary_margin (issue #23)
# ---------------------------------------------------------------------------

# Fixtures chosen to be strict-subset (some window holds fewer than every row,
# which is the only regime where a row can be dropped) and cheap. The w spread
# is fixed in *metres* while the offset varies, which decouples |w0| from n_w:
# a large offset raises the absolute-w scale, and hence the host/device gap,
# without exploding the plane count.
_MARGIN_CASES = [
    # (tag, dtype, w offset in metres, w spread in metres, freq, epsilon)
    ("float64 edge-test geometry", np.float64, 5.0e4, 2.0e3, np.array([1.0e9]), 1e-6),
    ("float64 |w0| ~ 0", np.float64, 0.0, 8.0e3, np.array([1.4e9]), 1e-8),
    ("float64 |w0| 5e5 m", np.float64, 5.0e5, 2.0e3, np.array([1.0e9]), 1e-6),
    ("float64 3 channels", np.float64, 0.0, 8.0e3, np.array([1.2e9, 1.4e9, 1.6e9]), 1e-6),
    ("float32 |w0| ~ 0", np.float32, 0.0, 8.0e3, np.array([1.4e9]), 1e-3),
    ("float32 |w0| 5e7 m", np.float32, 5.0e7, 8.0e3, np.array([1.4e9]), 1e-3),
]

# Minimum acceptable ratio of ``boundary_margin`` to the largest overhang any
# in-support row actually needs. Measured worst over these cases is 0.130 of
# the margin, i.e. 7.7x headroom (float32, |w0| 5e7 m); the worst float64 case
# is 0.110, i.e. 9.1x. Tripping below 3x leaves a factor of ~2.5 of slack while
# still failing outright if the coefficient is cut to a half, a quarter or an
# eighth of its shipped value.
_MIN_MARGIN_HEADROOM = 3.0


def _edge_probe_w_metres(
    plan: object,
    inv_lambda_c: float,
    w_lo: float,
    w_hi: float,
    n_step: int = 20,
    max_planes: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """``(w_metres, plane_index)`` hugging every interior window edge.

    The requirement this file measures is a property of the *arithmetic*, not of
    any particular row set, and in float64 waiting for random rows to land in
    the host/device disagreement band does not work: the band is ~1e-16
    relative, so 120 rows essentially never fall in it. Measured, every one of
    the 120 strict-subset float64 plans in a 311-plan sweep reports a
    requirement of exactly zero -- which is why cutting the margin coefficient
    8x *for float64 only* left the entire suite green.

    So walk to the edges instead of waiting for them, exactly as
    :func:`_find_invariant_breaking_w` does. The walk is in the plan's own
    dtype, because that is the grid ``plan.uvw_m`` lives on and a float64 walk
    would collapse onto a handful of distinct float32 values.

    Restricted to ``(w_lo, w_hi)``: an edge outside the data range belongs to a
    plane whose window is empty, so a row there cannot exist and counting it
    would make the gate conservative for no reason.
    """
    dtype = plan.real_dtype  # type: ignore[attr-defined]
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)  # type: ignore[attr-defined]
    scale = plan.w_kernel_scale  # type: ignore[attr-defined]
    out: list[float] = []
    planes: list[int] = []
    # A subsample of the interior planes, evenly spaced. The quantity being
    # measured is a property of the arithmetic at *an* edge, not of any
    # particular plane, so probing all ~600 planes of a wide-w plan costs
    # 25x the time for no more coverage (and one XLA compile per distinct
    # probe count).
    interior = np.arange(1, max(centres.shape[0] - 1, 2))
    if interior.size > max_planes:
        interior = interior[np.linspace(0, interior.size - 1, max_planes).astype(int)]
    for k in interior:
        k = int(k)
        for bound in (centres[k] - scale, centres[k] + scale):
            x = np.asarray((bound + plan.w0) / inv_lambda_c, dtype=dtype)  # type: ignore[attr-defined]
            for _ in range(n_step // 2):
                x = np.nextafter(x, np.asarray(-np.inf, dtype=dtype))
            for _ in range(n_step):
                if w_lo < float(x) < w_hi:
                    out.append(float(x))
                    # Each probe is only ever interesting against the plane whose
                    # edge produced it. Pairing every probe with every plane is a
                    # (30k, 600) matrix per channel and runs the process out of
                    # memory for nothing.
                    planes.append(k)
                x = np.nextafter(x, np.asarray(np.inf, dtype=dtype))
    return np.asarray(out, dtype=dtype), np.asarray(planes, dtype=np.intp)


def _margin_requirement(uvw: np.ndarray, freq: np.ndarray, dtype: type, eps: float) -> tuple:
    """``(requirement, margin, plan)`` for one fixture.

    The requirement is the largest distance by which a row that the *production*
    ``_channel_ft_coords`` places inside kernel support (``|z| <= 1``) falls
    outside the *unwidened* host boundary ``[w_k - S, w_k + S]``. That is
    exactly what ``boundary_margin`` has to cover: any such row would be dropped
    by a builder that searched at the bare boundary.

    Both sides are computed the way production computes them, which took two
    corrections to get right and is the whole reliability of this gate:

      * the **host** side forms the product in the plan's ``real_dtype`` before
        widening, as ``make_plan`` does. A float64 host value reports zero
        overhang on every float32 fixture.
      * the **device** side is fed ``plan.uvw_m`` -- already cast to
        ``real_dtype`` -- not the caller's raw float64 ``uvw``. Feeding the raw
        array measures a float32 host against a float64 device, a gap that never
        occurs at runtime, and inflates the answer about 2x.

    The margin comes from ``window_boundary_margin``, the same call
    ``make_plan`` makes, so this measures the number actually applied rather
    than a restatement of the formula.
    """
    from jax_nufft.wgridder import _channel_ft_coords

    plan = make_plan(uvw, freq, (32, 32), 0.004, 0.004, eps, dtype=dtype)
    perm = np.asarray(plan.sort_perm)
    centres = np.asarray(plan.w_centers_rel, dtype=np.float64)
    scale = plan.w_kernel_scale
    inv_lambda = np.asarray(plan.inv_lambda)
    uvw_m = np.asarray(plan.uvw_m)

    w_abs = np.outer(inv_lambda.astype(np.float64), uvw[:, 2])
    margin = window_boundary_margin(
        plan.real_dtype, float(w_abs.min()), float(w_abs.max()), plan.w_extent
    )
    w_lo, w_hi = float(uvw_m[:, 2].min()), float(uvw_m[:, 2].max())

    def overhang_of(host: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """How far outside the bare ``[centre - S, centre + S]`` the host sits."""
        return np.maximum(np.maximum(centre - scale - host, host - centre - scale), 0.0)

    requirement = 0.0
    for chan in range(plan.n_chan):
        probes, probe_plane = _edge_probe_w_metres(plan, float(inv_lambda[chan]), w_lo, w_hi)
        rows_m = np.concatenate([uvw_m[perm, 2], probes])
        block = np.zeros((rows_m.size, 3), dtype=plan.real_dtype)
        block[:, 2] = rows_m
        host = (rows_m * inv_lambda[chan]).astype(np.float64) - plan.w0
        # Under ``jax.jit``, not eagerly. Op-by-op execution does not contract
        # the multiply-subtract into an FMA at all, so an eager call measures a
        # device value that agrees with the host bit-for-bit and reports a
        # requirement of exactly zero -- i.e. it silently measures the wrong
        # thing on the very dtype this gate exists for.
        device = np.asarray(
            jax.jit(_channel_ft_coords)(jnp.asarray(block), plan.inv_lambda[chan], plan)[2],
            dtype=np.float64,
        )
        n_fixture = perm.size

        # Fixture rows: every plane, but only 120 of them.
        host_rows, dev_rows = host[:n_fixture], device[:n_fixture]
        in_support = np.abs((dev_rows[:, None] - centres[None, :]) / scale) <= 1.0
        if in_support.any():
            over = overhang_of(host_rows[:, None], centres[None, :])
            requirement = max(requirement, float(over[in_support].max()))

        # Edge probes: each against its own plane only.
        if probes.size:
            host_p, dev_p = host[n_fixture:], device[n_fixture:]
            centre_p = centres[probe_plane]
            hit = np.abs((dev_p - centre_p) / scale) <= 1.0
            if hit.any():
                over_p = overhang_of(host_p, centre_p)
                requirement = max(requirement, float(over_p[hit].max()))
    return requirement, margin, plan


def test_window_boundary_margin_covers_host_device_gap() -> None:
    """``boundary_margin`` must exceed what the rows actually need, with headroom.

    ``test_windowed_dense_parity_at_window_edge`` gates that the margin is
    non-zero -- it fails with the margin deleted -- but nothing gated its
    *magnitude*, which is the substance of the derivation in ``planning.py``.
    Measured: with the coefficient at 0.5, 1.0 or 1.5 instead of 4.0, that test
    and the entire fast suite stayed green. A fixture cannot close that gap,
    because on any fixture whose rows are not adversarially placed the margin
    sits orders of magnitude below the row spacing and a smaller one changes no
    window at all.

    So measure the requirement directly rather than provoking it: for each
    strict-subset plan, the largest amount by which a row the operator puts
    inside kernel support falls outside the bare ``[w_k - S, w_k + S]``
    boundary. That is the quantity the margin exists to cover, it is
    well-defined on every plan, and it scales with the same ``u * w_abs_scale``
    the derivation is written in.

    The assertion is on the ratio, not just the sign, so that a future change to
    ``w_kernel_scale`` or the plane count which erodes the headroom fails loudly
    instead of silently spending it.
    """
    ratios = []
    for tag, dtype, offset_m, spread_m, freq, eps in _MARGIN_CASES:
        rng = np.random.default_rng(0)
        n_rows = 120
        uvw = np.zeros((n_rows, 3))
        uvw[:, 0] = rng.uniform(-120.0, 120.0, size=n_rows)
        uvw[:, 1] = rng.uniform(-120.0, 120.0, size=n_rows)
        uvw[:, 2] = offset_m + rng.uniform(-spread_m, spread_m, size=n_rows)

        requirement, margin, plan = _margin_requirement(uvw, freq, dtype, eps)
        assert plan.max_window_size < plan.n_rows, (
            f"{tag}: every window already spans every row, so no row can be dropped and "
            "this case gates nothing -- pick a fixture with more planes"
        )
        assert margin > 0.0, f"{tag}: boundary_margin is zero"
        ratios.append((requirement / margin, tag, requirement, margin, dtype))

    # Per *dtype*, not just overall. float64 is the dtype the whole tolerance
    # contract is written in, and it is the one where a requirement of zero is
    # easy to measure by accident: the disagreement band is ~1e-16 relative, so
    # random rows never land in it and only the edge probes reach it. With the
    # float64 cases reporting zero, cutting the coefficient 8x for float64 alone
    # left the entire suite green -- so "some case is non-zero" is not enough.
    for want in (np.float64, np.float32):
        best = max(r[0] for r in ratios if r[4] is want)
        assert best > 0.0, (
            f"every {np.dtype(want).name} case measures a requirement of exactly zero, so "
            f"this gate would pass with boundary_margin set to zero for {np.dtype(want).name}. "
            "The edge probes are not reaching the host/device disagreement band -- check "
            "_edge_probe_w_metres, and that the device side runs under jax.jit"
        )

    worst_ratio, worst_tag, worst_req, worst_margin, _ = max(ratios)
    assert worst_ratio <= 1.0 / _MIN_MARGIN_HEADROOM, (
        f"boundary_margin has only {1.0 / worst_ratio:.2f}x headroom on {worst_tag} "
        f"(rows need {worst_req:.3e} wavelengths of widening, margin is {worst_margin:.3e}), "
        f"below the {_MIN_MARGIN_HEADROOM:g}x this gate requires. Either "
        "WINDOW_BOUNDARY_MARGIN_EPS has been cut, or a change to w_kernel_scale / the "
        "plane count has invalidated the derivation at its use site in planning.py -- "
        "re-derive it rather than raising this bound"
    )
