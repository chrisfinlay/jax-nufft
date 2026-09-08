"""Issue #26: per-channel window sizes and w-plane size bucketing.

**Status: this module is written against a feature that does not exist yet.**
The structural half of it fails at ``ab7fbbd``; the equivalence half passes
today and is the regression net the change has to keep green. "What fails
today, and how" at the foot of this docstring is the implementer's target
list.

What the feature is
-------------------
Three things, from the issue's implementation plan:

2. **Per-channel window sizes.** ``plan.max_window_size`` is one number for
   the whole plan, so every ``(channel, plane)`` slice is padded to the
   widest window *any* channel has. Channels differ: their w in wavelengths
   is ``freq[c]/c * w_metres``, so a high-frequency channel spreads the same
   baselines over more planes and gets narrower windows.
3. **Bucketing.** Sort a channel's planes into a small number of size
   classes and give each class its own static slice length, so a plane whose
   window holds 24 rows does not slice 1072.
4. **Report the effective ratio.** ``plan.window_padding_overhead`` becomes
   the work bucketing actually does over the work it cannot avoid.

The plan surface this module pins
---------------------------------
The issue fixes the *behaviour* and not the spelling, so the two new plan
attributes are looked up through :func:`_bucket_layout` /
:func:`_max_window_size_per_chan`, which try a few plausible names and fail
with a message naming the contract when none is present. What they have to
mean is fixed:

``plan.window_buckets``
    Static, ``n_chan`` entries, one per channel. Entry ``c`` is that
    channel's buckets as ``(slice_length, n_planes)`` pairs, ascending in
    ``slice_length``, the lengths distinct, ``sum(n_planes) == plan.n_w``.
    Per-channel and not global: a global bucket table would make item 2
    unreachable, because then every channel's widest bucket is the widest
    bucket in the plan.
``plan.max_window_size_per_chan``
    ``(n_chan,)`` static ints, channel ``c``'s widest padded window. May be
    a property over ``window_buckets``; this module does not care which.
``plan.max_window_size``
    **Unchanged**: still one int, still the global maximum, still the size
    of the widest slice any strategy takes. It is what the pre-#26 metric is
    recomputed from below, so the baseline stays visible after the change.
``plan.window_padding_overhead``
    Redefined from ``n_chan * n_w * max_window_size / live_row_count``
    (issue #43) to ``sum over channels and buckets of
    slice_length * n_planes / live_row_count``. Same denominator, same
    meaning -- padded row-work over irreducible row-work -- and it collapses
    to the #43 form when every channel has exactly one bucket, which is
    where the plan builder is today.

Measured starting point
-----------------------
All numbers in this module were measured on ``ab7fbbd`` (2026-09, macOS
arm64 10-core, CPU backend, ``pixi run -e test``, jax 0.9.2), float64,
``epsilon = 1e-6``, ``synthetic_uvw(..., seed=0)``, ``n_chan = 1``, the
shipped ``hermitian=True``, at the CI fixture sizes in ``tests/conftest.py``
-- *not* at the realistic sizes the issue's second comment also tabulates.

===================  ======  =====  ======  =============  ==============  =========
fixture              n_rows    n_w     mws  mws / n_rows   overhead today  <=4-bucket
===================  ======  =====  ======  =============  ==============  =========
EDA2 zenith             400     11     400          1.000          1.5709     1.0368
EDA2 off30              400     56     126          0.315          2.5191     1.2424
MWA_compact zenith      600      8     600          1.000          1.1431     1.0007
MWA_compact off30       600     12     600          1.000          1.7139     1.0467
MWA_extended zenith     600     11     600          1.000          1.5714     1.0371
MWA_extended off30      600    134     155          0.258          4.9441     1.3768
MeerKAT zenith          600      8     600          1.000          1.1429     1.0005
MeerKAT off30           600     13     600          1.000          1.8571     1.0690
GH200_large zenith    50000      9   50000          1.000          1.2857     1.0000
GH200_large off30     50000     26   36870          0.737          2.7389     1.2861
===================  ======  =====  ======  =============  ==============  =========

The "<=4-bucket" column is the *achievable* effective overhead: the minimum
of ``sum(slice_length * n_planes) / live_row_count`` over every partition of
that plan's padded window sizes into at most four size classes, computed by
exact dynamic programming over the sorted sizes. It is what makes the
definition-of-done's ``<= 1.5`` gate a target rather than a hope -- the worst
cell reaches 1.3768, i.e. 8% of headroom under the gate, and the gate is
therefore asserted at 1.5 exactly as the issue states it.

Two conclusions from that table that the issue does not have:

* **The issue's own suggested policy for item 2 does not meet its own gate.**
  Rounding each window up to a power of two gives, on the ten cells above in
  order: 1.3088, 1.3417, 1.7075, 1.7486, 1.7414, 1.5891, 1.7071, 1.6424,
  1.3166, 1.5397. Seven of the ten are above 1.5, and on five of them the
  metric comes out *worse than it is today* -- the five 600-row cells all
  round to a 1024 class, so MWA_compact zenith reads 1.7075 against today's
  1.1431 and MeerKAT zenith 1.7071 against 1.1429. Bucket boundaries have to
  be placed against the plan's own size distribution, not on a fixed grid.
* **``GH200_large`` off30 is a third fixture with strictly-subset windows.**
  The issue's second comment names EDA2 off30 and MWA_extended off30 as the
  only two; ``GH200_large`` off30 has ``max_window_size / n_rows = 0.737``
  and belongs on that list. The other seven cells above have
  ``max_window_size == n_rows``.

The vacuity trap, and the guard against it
------------------------------------------
Seven of the ten cells pad every window to the full row count, so on those
plans ``windowed_*`` and ``dense_*`` are the same computation and *nothing*
about window bounds can be demonstrated. Every test below that asserts a
bucketing or window-bound property calls :func:`_assert_windows_are_strict`,
which pins ``plan.max_window_size < plan.n_rows`` and, where the claim is
about size classes, that the padded window sizes actually vary. Without that
guard these tests would pass on a plan where the feature does nothing --
the failure mode issue #15 and issue #22 shipped.

The fixtures that carry the guard here are EDA2 off30 (single- and
four-channel) and :data:`NARROW`, a synthetic telescope defined below
specifically so the row-slice term dominates transient memory.

Two measurement caveats found while writing this module
-------------------------------------------------------
* **Never compare an eager call against a jitted one.** On :data:`NARROW`
  (4000 rows, 16^2, eps 1e-6, float64) ``jax.jit(jax.vmap(f))(x)`` and
  ``jax.vmap(f)(x)`` differ by a relative 4.09e-10 for ``dense_scan`` and
  4.09e-10 for ``windowed_scan`` -- the *same* discrepancy for both, i.e. a
  property of the execution mode and not of the strategy, but four orders of
  magnitude above the 1e-11 strategy bound. Compared jit-to-jit the two
  strategies agree to 2.41e-15 on the same call. Every comparison below is
  within one execution mode.
* **``windowed_scan``'s peak transient memory will not fall with
  bucketing**, and the issue's item 5 framing ("the windowed strategies' temp
  bytes should fall") is too broad. A scan holds one plane's slice at a time,
  so its peak is set by the *largest* bucket, which is still
  ``max_window_size``. Only the strategies that hold every plane's slice at
  once pay ``sum over planes``; that is ``windowed_vmap``, and it is the one
  the memory gate below is written on. Measured on :data:`NARROW`, forward,
  ``temp_size_in_bytes``: ``windowed_vmap`` 13,565,952 against
  ``windowed_scan`` 128,032 and ``dense_scan`` 72,328.

What fails today, and how
-------------------------
At ``ab7fbbd``:

* every test that calls :func:`_bucket_layout` or
  :func:`_max_window_size_per_chan` fails with :data:`_MISSING_FEATURE`,
  because ``WGridderPlan`` has no such attribute --
  ``test_the_bucket_layout_is_well_formed`` (4 cells),
  ``test_the_reported_overhead_is_the_effective_bucketed_ratio``,
  ``test_equivalence_still_holds_once_the_windows_are_actually_bucketed``
  (2 cells) and
  ``test_per_channel_window_sizes_are_used_rather_than_one_global_max``;
* ``test_effective_padding_overhead_is_at_most_one_and_a_half`` fails on
  seven of its ten cells on the number itself (1.5709, 2.5191, 1.7139,
  1.5714, 4.9441, 1.8571, 2.7389), and passes on MWA_compact zenith
  (1.1431), MeerKAT zenith (1.1429) and GH200_large zenith (1.2857);
* ``test_the_lowered_windowed_loop_takes_more_than_one_slice_length`` fails
  on all six of its cells, but not all six the same way: the three
  ``dirty2vis`` cells fail on "still takes a single slice length [1072]",
  while the three ``vis2dirty`` ones find ``{1, 1072}`` -- the extra ``1``
  being the adjoint's ``flip_sign`` row gather and not a window -- so they
  clear that assertion and fail at :func:`_bucket_layout` instead. See the
  test's own docstring;
* ``test_interleaved_channel_groups_are_reassembled_in_channel_order`` fails
  on all six of its cells at :func:`_bucket_layout`, that being how it reads
  the channel grouping;
* ``test_windowed_vmap_temp_memory_falls_with_bucketing`` fails at
  13,565,952 bytes against a 6,782,976-byte bound;
* ``test_no_two_planes_accumulate_into_one_row_vector_in_the_forward``
  splits: its ``windowed_vmap`` cell **passes** (``ab7fbbd``'s forward gives
  each plane a private row vector, uncapped) and its ``windowed_chunked``
  cell **fails** on a ``(28, 1072)`` shared-carry scatter. That failure is a
  pre-existing defect in ``ab7fbbd``'s chunked forward and not something this
  issue introduced -- it is invisible in the coordinator's GPU A/B only
  because ``w_chunk = 32`` exceeds ``n_w`` on both of its fixtures, so the
  chunked strategy degenerates to the vmap one there. Both cells fail at
  ``f3a7f49``, the state of this branch before that test existed;
* the equivalence, gradient, vmap/jit and vacuity-guard tests **pass** today
  (measured: 41 passed, 29 failed of the module's 70 cells at ``ab7fbbd``)
  and are regression protection: they carry the vacuity guard so that they
  are still meaningful once bucketing lands, but they assert nothing that is
  false before it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft.planning import WGridderPlan
from jax_nufft.wgridder import _CANONICAL_W_STRATEGIES, DEFAULT_W_CHUNK
from tests.conftest import (
    EDA2,
    GH200_LARGE,
    MEERKAT,
    MWA_COMPACT,
    MWA_EXTENDED,
    X64,
    Telescope,
    requires_x64,
    synthetic_uvw,
    tol,
)

# The padded-window reference is ``tests/test_padding_overhead.py``'s, not a
# second copy of it: that helper is already the independent statement of the
# builder's window lengths (it re-derives ``lo``/``hi`` from the plan leaves
# and asserts its own maximum equals ``plan.max_window_size``), and it is
# pinned in its own module by
# ``test_planning.py::test_window_builder_matches_independent_reference``.
# Importing it here is safe on both precision legs -- that module is not in
# ``conftest.collect_ignore`` and does not touch ``jax_enable_x64``.
from tests.test_padding_overhead import _independent_padded_sizes

# --- module-wide constants ---------------------------------------------------

# float32 cannot reach 1e-6 (``planning.FLOAT32_EPSILON_FLOOR`` is 1e-5) and
# ``make_plan`` warns below it, which the suite's ``filterwarnings = ["error"]``
# turns into an exception. Choosing an epsilon each leg can actually deliver
# keeps every plan in this module warning-free, so no call site needs a
# ``pytest.warns`` wrapper. Every *measured number* in this module is float64 at
# 1e-6 and its test carries ``@requires_x64``; the float32 leg runs the
# structural and equivalence tests at 1e-3.
EPSILON: float = tol(1e-6, 1e-3)

_REAL_DTYPE: DTypeLike = jnp.float64 if X64 else jnp.float32
_COMPLEX_DTYPE: DTypeLike = jnp.complex128 if X64 else jnp.complex64

# Reduction-order agreement between mathematically identical operators.
# Bucketing changes which rows a plane slices, never which rows carry a
# nonzero kernel weight, so it is a reduction-order change and AGENTS.md
# section 6's eps-independent strategy-equivalence row is the right bound for
# it.
#
# float64 takes that row's constant unchanged, and has four orders of headroom
# under it: measured over this module's whole float64 grid (four equivalence
# cells x three windowed strategies x two operators, plus the jit/vmap and
# gradient cells) the worst disagreement with ``dense_scan`` is 9.223e-15,
# on NARROW off30's adjoint.
#
# float32 does **not** take the shared 1.3e-6. That constant was measured by
# issue #10 on 600-row, three-channel fixtures; :data:`NARROW` has 4000 rows,
# so its adjoint accumulates over a longer single-precision reduction and the
# same comparison lands above it. Measured over this module's whole float32
# grid (eps 1e-3, the same cells as above): equivalence worst 2.614e-6
# (NARROW off30 adjoint, ``windowed_scan``; ``windowed_vmap`` 2.446e-6,
# ``windowed_chunked`` 2.611e-6, and every EDA2 cell at or below 3.952e-7),
# jit+vmap worst 7.139e-7, gradient worst 1.630e-6. The bound is 10x the worst
# of those, which is AGENTS.md section 6's rule for a bound that does not hold
# as written -- measure the worst case and set it at 10x, not at whatever
# passes.
STRATEGY_TOL: float = tol(1e-11, 2.6e-5)

# The definition-of-done's gate, quoted verbatim from issue #26.
PADDING_OVERHEAD_GATE = 1.5

WINDOWED_STRATEGIES: tuple[str, ...] = tuple(
    name for name in _CANONICAL_W_STRATEGIES if name.startswith("windowed")
)

_MISSING_FEATURE = (
    "issue #26 is not implemented: {what}. Expected {names} on WGridderPlan. "
    "``window_buckets`` is a static tuple with one entry per channel; entry c "
    "is that channel's planes bucketed by padded window length, as "
    "((slice_length, n_planes), ...) ascending in slice_length with distinct "
    "lengths and sum(n_planes) == plan.n_w. ``max_window_size_per_chan`` is a "
    "(n_chan,) tuple of ints, channel c's widest padded window (a property "
    "over window_buckets is fine). ``plan.max_window_size`` stays the global "
    "int it is today, and ``plan.window_padding_overhead`` becomes "
    "sum(slice_length * n_planes) / plan.live_row_count."
)


# --- fixtures ----------------------------------------------------------------

# A synthetic telescope built for the memory gate, because no repo fixture can
# carry it. ``windowed_vmap``'s transient is dominated by the plane stack of
# images on every one of the eight review cells: measured at eps 1e-6,
# float64, seed 0, hermitian=True, forward, ``n_w * image`` is 74.53% of the
# transient on EDA2 zenith, 88.62% / 88.64% / 88.64% on EDA2 off30 and the two
# MWA_compact cells, and 96.90% / 98.86% / 96.90% / 96.90% on the four
# 256^2 cells. Bucketing cannot move that term, so a memory gate written on any
# of them would be measuring the wrong thing.
#
# This fixture reuses MWA_extended's uv distribution -- which is what produces
# the narrow windows (measured ``max_window_size / n_rows`` 0.258 there, 0.268
# here) -- with 4000 rows instead of 600 and a 16^2 image instead of 256^2, so
# the ratio inverts. Measured at eps 1e-6, float64, off30, seed 0,
# hermitian=True: n_w = 138, max_window_size = 1072, live_row_count = 28001,
# overhead 5.2832, 85 distinct padded window sizes spanning 2 to 1072 with a
# median of 24, no empty planes. One image plane is 4096 bytes, so
# ``n_w * image`` is 565,248 bytes -- 4.17% of ``windowed_vmap``'s measured
# 13,565,952-byte forward transient, the rest of which is
# ``n_w * max_window_size`` row-shaped traffic at a measured 91.70 bytes per
# (plane, row).
NARROW = Telescope(
    name="i26_narrow",
    freq_hz=150e6,
    n_rows=4000,
    sigma_uv_m=800.0,
    max_baseline_m=5300.0,
    n_pix=16,
    fov_rad=math.radians(25.0),
)

# The four-channel fixture for item 2. The frequency factors are deliberately
# *not* the +/-5% spread ``test_strategies_equivalent`` and
# ``test_chunked_strategy`` use: at +/-5% the per-channel window maxima on
# EDA2 off30 are 129 / 124 / 122 (measured, three channels), a 6% spread that
# a global maximum barely wastes anything on. At 1.0 / 1.3 / 1.6 / 2.0 they
# are 126 / 103 / 84 / 72 (measured, eps 1e-6, float64, hermitian=True), a
# 1.75x spread, and the global maximum costs a measured 4.7250 against 3.6094
# for per-channel maxima alone -- i.e. per-channel sizing alone removes 23.6%
# of the padded work before any bucketing. That is what "measurably wasteful"
# has to mean for this test not to be vacuous on the channel axis, which is
# the axis issue #26 item 2 is about.
MULTI_CHAN_FREQ_FACTORS = (1.0, 1.3, 1.6, 2.0)

# (telescope, zenith_angle_deg) -> ``window_padding_overhead`` at ``ab7fbbd``,
# float64, eps 1e-6, seed 0, hermitian=True, n_chan=1, CI fixture sizes. This
# is the *pre-bucketing* value and stays recomputable after the change as
# ``n_chan * n_w * max_window_size / live_row_count`` (issue #43's definition,
# every term of which this issue leaves in place), so it is asserted below as
# a baseline: if the plan geometry drifts, that assertion fires rather than
# the gate silently becoming easier.
_BASELINE_OVERHEAD_AT_AB7FBBD: dict[tuple[str, float], float] = {
    ("EDA2", 0.0): 1.5709,
    ("EDA2", 30.0): 2.5191,
    ("MWA_compact", 0.0): 1.1431,
    ("MWA_compact", 30.0): 1.7139,
    ("MWA_extended", 0.0): 1.5714,
    ("MWA_extended", 30.0): 4.9441,
    ("MeerKAT", 0.0): 1.1429,
    ("MeerKAT", 30.0): 1.8571,
    ("GH200_large", 0.0): 1.2857,
    ("GH200_large", 30.0): 2.7389,
}

_REPO_FIXTURE_CELLS: tuple[tuple[Telescope, float], ...] = (
    (EDA2, 0.0),
    (EDA2, 30.0),
    (MWA_COMPACT, 0.0),
    (MWA_COMPACT, 30.0),
    (MWA_EXTENDED, 0.0),
    (MWA_EXTENDED, 30.0),
    (MEERKAT, 0.0),
    (MEERKAT, 30.0),
    (GH200_LARGE, 0.0),
    (GH200_LARGE, 30.0),
)

_CELL_IDS = tuple(
    f"{tel.name}-{'zenith' if za == 0.0 else 'off30'}" for tel, za in _REPO_FIXTURE_CELLS
)


def _plan(
    telescope: Telescope,
    zenith_angle_deg: float,
    *,
    freq_factors: tuple[float, ...] = (1.0,),
    hermitian: bool = True,
    epsilon: float | None = None,
) -> WGridderPlan:
    """A plan at this module's precision leg, ``seed=0``, square image/pixel."""
    uvw = synthetic_uvw(telescope, zenith_angle_deg, seed=0)
    freq = telescope.freq_hz * np.asarray(freq_factors, dtype=np.float64)
    return make_plan(
        uvw,
        freq,
        (telescope.n_pix, telescope.n_pix),
        telescope.pixsize,
        telescope.pixsize,
        epsilon=EPSILON if epsilon is None else epsilon,
        dtype=_REAL_DTYPE,
        hermitian=hermitian,
    )


# ``GH200_large``'s plan holds ~135 MB of 2048^2 leaves and 50k rows, so the
# cache is small on purpose: it collapses an immediately repeated build of the
# same cell, it is not meant to hold the whole parametrisation.
@lru_cache(maxsize=3)
def _cached_plan(
    telescope: Telescope,
    zenith_angle_deg: float,
    freq_factors: tuple[float, ...],
    hermitian: bool,
) -> WGridderPlan:
    return _plan(telescope, zenith_angle_deg, freq_factors=freq_factors, hermitian=hermitian)


def _inputs(plan: WGridderPlan, telescope: Telescope) -> tuple[Any, Any]:
    """A real per-channel image stack and complex visibilities for ``plan``."""
    rng = np.random.default_rng(7)
    n_chan = plan.n_chan
    shape = (
        (telescope.n_pix, telescope.n_pix)
        if n_chan == 1
        else (n_chan, telescope.n_pix, telescope.n_pix)
    )
    image = jnp.asarray(rng.standard_normal(shape), dtype=_REAL_DTYPE)
    vis = jnp.asarray(
        rng.standard_normal((plan.n_rows, n_chan))
        + 1j * rng.standard_normal((plan.n_rows, n_chan)),
        dtype=_COMPLEX_DTYPE,
    )
    return image, vis


# --- feature discovery -------------------------------------------------------

_BUCKET_ATTRS = ("window_buckets", "window_size_buckets", "plane_size_buckets")
_PER_CHAN_ATTRS = (
    "max_window_size_per_chan",
    "max_window_size_per_channel",
    "max_window_sizes",
)


def _bucket_layout(plan: WGridderPlan) -> tuple[tuple[tuple[int, int], ...], ...]:
    """The per-channel bucket table, normalised, or a descriptive failure.

    Returns ``n_chan`` entries of ``((slice_length, n_planes), ...)``. A
    single-channel plan is allowed to store the flat pair list without the
    outer channel axis; anything else must already be per-channel, because a
    global table cannot express item 2.
    """
    for name in _BUCKET_ATTRS:
        raw = getattr(plan, name, None)
        if raw is None:
            continue
        entries = tuple(raw)
        if entries and all(
            isinstance(e, (tuple, list)) and len(e) == 2 and isinstance(e[0], int) for e in entries
        ):
            # A flat ((length, count), ...) list: only unambiguous at n_chan == 1.
            if plan.n_chan != 1:
                pytest.fail(
                    f"plan.{name} is a flat bucket list on an n_chan={plan.n_chan} "
                    "plan; issue #26 item 2 needs one bucket table per channel, "
                    "otherwise every channel's widest bucket is the plan's widest "
                    "bucket and per-channel sizing buys nothing."
                )
            entries = (entries,)
        out = tuple(
            tuple((int(length), int(count)) for length, count in per_chan) for per_chan in entries
        )
        assert len(out) == plan.n_chan, (
            f"plan.{name} has {len(out)} entries for an n_chan={plan.n_chan} plan"
        )
        return out
    pytest.fail(
        _MISSING_FEATURE.format(
            what="the plan carries no bucket table",
            names=" / ".join(f"`{n}`" for n in _BUCKET_ATTRS),
        )
    )


def _max_window_size_per_chan(plan: WGridderPlan) -> tuple[int, ...]:
    """``(n_chan,)`` per-channel maximum padded window length.

    Derived from :func:`_bucket_layout` when the plan does not expose it
    directly -- the two are the same information and the issue does not say
    which one is the field.
    """
    for name in _PER_CHAN_ATTRS:
        raw = getattr(plan, name, None)
        if raw is None:
            continue
        out = tuple(int(v) for v in raw)
        assert len(out) == plan.n_chan, (
            f"plan.{name} has {len(out)} entries for an n_chan={plan.n_chan} plan"
        )
        return out
    return tuple(max(length for length, _ in per_chan) for per_chan in _bucket_layout(plan))


def _effective_padded_work(plan: WGridderPlan) -> int:
    """Rows a bucketed windowed traversal touches: one slice per (channel, plane)."""
    return sum(length * count for per_chan in _bucket_layout(plan) for length, count in per_chan)


def _unbucketed_padded_work(plan: WGridderPlan) -> int:
    """Rows the pre-#26 windowed traversal touches -- issue #43's numerator."""
    return plan.n_chan * plan.n_w * plan.max_window_size


def _unbucketed_overhead(plan: WGridderPlan) -> float:
    """Issue #43's ``window_padding_overhead``, recomputed from plan fields.

    Every term survives issue #26 by the contract in the module docstring, so
    this stays computable after the change and is what the baseline table is
    asserted against.
    """
    return _unbucketed_padded_work(plan) / plan.live_row_count


# --- anti-vacuity guards -----------------------------------------------------


def _assert_windows_are_strict(plan: WGridderPlan, *, sizes_must_vary: bool = True) -> None:
    """Refuse to make a window-bound claim on a plan whose windows hold every row.

    Seven of the ten repo fixture cells (module docstring) have
    ``max_window_size == n_rows``: on those the windowed and dense strategies
    are literally the same computation and a bucketing assertion is
    unfalsifiable. This is the guard the issue's second comment asks for, and
    ``sizes_must_vary`` adds the second half of it -- a plan whose padded
    window sizes are all equal has nothing to bucket either.
    """
    assert plan.max_window_size < plan.n_rows, (
        f"vacuous fixture: max_window_size ({plan.max_window_size}) == n_rows "
        f"({plan.n_rows}), so every window spans every row and this plan cannot "
        "demonstrate anything about window bounds. Use EDA2 off30, "
        "MWA_extended off30, GH200_large off30 or the NARROW fixture."
    )
    if sizes_must_vary:
        sizes = _independent_padded_sizes(plan)
        distinct = len(set(sizes.ravel().tolist()))
        assert distinct > 1, (
            f"vacuous fixture: all {sizes.size} padded window sizes are equal, "
            "so there is nothing for bucketing to separate."
        )


# --- lowering probe ----------------------------------------------------------

# The windowed slice is invisible in the jaxpr -- both operators are bound as
# opaque primitives (``_dirty2vis_p`` / ``_vis2dirty_p``, issue #21), so the
# w-plane loop only appears once they are lowered. In the lowered StableHLO it
# takes one of two forms depending on whether the loop scans or vmaps:
#
#   dynamic_slice %arg2, %11, sizes = [1072] : (tensor<4000xf64>, ...)
#   "stablehlo.gather"(%4, %22) <{... slice_sizes = array<i64: 1072>}>
#       : (tensor<4000xf64>, tensor<138x1xi32>) -> tensor<138x1072xf64>
#
# Both are matched below, and both are anchored on a *one-dimensional*
# ``tensor<n_rows x ...>`` operand so that the ``slice_sizes = array<i64: 1, 3>``
# gather on the ``(n_rows, 3)`` ``uvw_m`` leaf is not mistaken for a window.
_DYNAMIC_SLICE = r"dynamic_slice[^\n]*sizes = \[(\d+)\][^\n]*\(tensor<{n}x[a-z]"
_GATHER_SLICE = r"slice_sizes = array<i64: (\d+)>[^\n]*\(tensor<{n}x[a-z]"


def _window_slice_lengths(text: str, n_rows: int) -> set[int]:
    """Distinct row-slice lengths the lowered program takes out of a row array."""
    found: set[int] = set()
    for pattern in (_DYNAMIC_SLICE, _GATHER_SLICE):
        found.update(int(m) for m in re.findall(pattern.format(n=n_rows), text))
    return found


def _lowered_text(fn: Callable[..., Any], *args: Any) -> str:
    return jax.jit(fn).lower(*args).as_text()


def _temp_bytes(fn: Callable[..., Any], *args: Any) -> int:
    """XLA transient memory for ``fn`` at these argument shapes (AGENTS.md sec 6)."""
    analysis = jax.jit(fn).lower(*args).compile().memory_analysis()
    if analysis is None:  # pragma: no cover - backend without the analysis
        pytest.skip("this backend does not expose memory_analysis()")
    return int(analysis.temp_size_in_bytes)


def _call(op: str, plan: WGridderPlan, arg: Any, *, w_strategy: str, **kw: Any) -> Any:
    fn = dirty2vis if op == "dirty2vis" else vis2dirty
    if "chunk" in w_strategy:
        kw.setdefault("w_chunk", DEFAULT_W_CHUNK)
    return fn(plan, arg, w_strategy=w_strategy, **kw)


def _rel(a: Any, b: Any) -> float:
    """Relative L2 difference, computed in float64 whatever the operands are."""
    a64 = np.asarray(a, dtype=np.complex128).ravel()
    b64 = np.asarray(b, dtype=np.complex128).ravel()
    denom = float(np.linalg.norm(b64))
    assert denom > 0.0, "reference is identically zero; the comparison would be vacuous"
    assert np.all(np.isfinite(a64)), "left-hand result contains non-finite values"
    return float(np.linalg.norm(a64 - b64) / denom)


# =============================================================================
# 1. the gate: effective padding overhead <= 1.5 on the repo fixtures
# =============================================================================


@requires_x64
@pytest.mark.parametrize(("telescope", "zenith_angle_deg"), _REPO_FIXTURE_CELLS, ids=_CELL_IDS)
def test_effective_padding_overhead_is_at_most_one_and_a_half(
    telescope: Telescope, zenith_angle_deg: float
) -> None:
    """Issue #26's definition of done, with the pre-change value pinned beside it.

    Three assertions, in the order that makes a failure readable:

    1. the *pre-bucketing* metric, recomputed from the plan's own fields as
       issue #43 defines it, still equals the value measured at ``ab7fbbd``
       (module docstring table). This is the regression tripwire: it is what
       says the plan geometry has not moved under the gate;
    2. the reported ``window_padding_overhead`` is no worse than that. A
       bucketing that made the ratio go up is a bug, and on the three cells
       already under the gate (MWA_compact zenith 1.1431, MeerKAT zenith
       1.1429, GH200_large zenith 1.2857) this is the only thing the gate
       itself would check;
    3. the reported ratio meets the ``<= 1.5`` gate.

    Measured at ``ab7fbbd``, float64, eps 1e-6, seed 0, hermitian=True,
    n_chan=1: assertion 3 fails on seven of these ten cells -- everything
    except the three named above -- and the worst is MWA_extended off30 at
    4.9441. Assertion 1 passes today by construction:
    ``window_padding_overhead`` *is* the pre-bucketing metric at ``ab7fbbd``.
    """
    plan = _cached_plan(telescope, zenith_angle_deg, (1.0,), True)
    baseline = _BASELINE_OVERHEAD_AT_AB7FBBD[(telescope.name, zenith_angle_deg)]

    assert _unbucketed_overhead(plan) == pytest.approx(baseline, rel=1e-3), (
        "the un-bucketed padding overhead has moved off its measured baseline; "
        "the plan geometry changed, so the gate below is no longer being read "
        "against the fixture it was calibrated on"
    )
    assert plan.window_padding_overhead <= _unbucketed_overhead(plan) * (1.0 + 1e-9), (
        f"bucketing made the padding overhead worse: "
        f"{plan.window_padding_overhead} > {_unbucketed_overhead(plan)}"
    )
    assert plan.window_padding_overhead <= PADDING_OVERHEAD_GATE, (
        f"{telescope.name} at {zenith_angle_deg} deg: effective padding overhead "
        f"{plan.window_padding_overhead:.4f} exceeds issue #26's gate of "
        f"{PADDING_OVERHEAD_GATE}. Measured achievable with at most four size "
        "classes on this cell: see the module docstring's last column."
    )


@requires_x64
def test_the_reported_overhead_is_the_effective_bucketed_ratio() -> None:
    """Pin the metric's definition, which issue #43 already moved once.

    ``window_padding_overhead`` has had two denominators and now gets a new
    numerator, so the tests that read it have to say which one they mean. The
    contract asserted here is

        window_padding_overhead
            == sum over channels and buckets of slice_length * n_planes
               / live_row_count

    with the denominator left exactly as issue #43 built it (nominal support,
    excluding both ``window_boundary_margin`` and the ``+/-1`` clamp).

    Also asserted: on a plan whose windows genuinely vary the new numerator is
    strictly smaller than issue #43's ``n_chan * n_w * max_window_size``, so
    the redefinition is not a rename. Measured at ``ab7fbbd`` on EDA2 off30,
    float64, eps 1e-6, hermitian=True, n_chan=1: the old numerator is
    56 * 126 = 7056 rows against a live count of 2801, i.e. the 2.5191 in the
    table, while the per-plane floor (every plane its own length) is 1.035 and
    at most four size classes reach 1.2424.
    """
    plan = _cached_plan(EDA2, 30.0, (1.0,), True)
    _assert_windows_are_strict(plan)

    effective = _effective_padded_work(plan)
    assert plan.window_padding_overhead == pytest.approx(
        effective / plan.live_row_count, rel=1e-12
    ), (
        "window_padding_overhead is not the effective bucketed ratio: expected "
        f"{effective} / {plan.live_row_count} = "
        f"{effective / plan.live_row_count!r}, got {plan.window_padding_overhead!r}"
    )
    assert effective < _unbucketed_padded_work(plan), (
        "the bucketed numerator equals the un-bucketed one on a plan whose "
        "window sizes vary, so nothing was bucketed"
    )
    assert plan.window_padding_overhead >= 1.0, (
        "padded work cannot be below live work; the lower bound "
        "tests/test_padding_overhead.py pins must survive the redefinition"
    )


@requires_x64
@pytest.mark.parametrize(
    ("telescope", "freq_factors"),
    [
        (EDA2, (1.0,)),
        (EDA2, MULTI_CHAN_FREQ_FACTORS),
        (NARROW, (1.0,)),
        (MWA_EXTENDED, (1.0,)),
    ],
    ids=["EDA2-1chan", "EDA2-4chan", "NARROW-1chan", "MWA_extended-1chan"],
)
def test_the_bucket_layout_is_well_formed(
    telescope: Telescope, freq_factors: tuple[float, ...]
) -> None:
    """Structural invariants of the bucket table, including the one that matters.

    The one that matters is the last: sorting a channel's padded window sizes
    ascending and expanding its buckets ascending, the ``i``-th bucket length
    must be ``>=`` the ``i``-th window size. That is exactly "no plane is
    assigned to a bucket too short to hold its window", stated without needing
    the plan to expose the plane-to-bucket assignment -- if it held anywhere,
    that plane would silently drop rows the dense path weights, which is the
    failure the equivalence tests below would then catch numerically.

    The rest are the shape contract: one entry per channel, ascending and
    distinct lengths, positive counts, ``sum(counts) == n_w`` (every plane is
    in exactly one bucket), each channel's widest bucket equal to its
    independently recomputed widest padded window, and the global maximum of
    those equal to ``plan.max_window_size``, which this issue leaves alone.

    Measured at ``ab7fbbd``, float64, eps 1e-6, hermitian=True: the four cells
    here have (n_chan, n_w, max_window_size, n_rows, distinct padded sizes) of
    (1, 56, 126, 400, 38), (4, 105, 126, 400, 95), (1, 138, 1072, 4000, 85)
    and (1, 134, 155, 600, 47).
    """
    plan = _plan(telescope, 30.0, freq_factors=freq_factors)
    _assert_windows_are_strict(plan)

    layout = _bucket_layout(plan)
    padded = _independent_padded_sizes(plan)
    per_chan_max = _max_window_size_per_chan(plan)

    assert len(layout) == plan.n_chan
    for c, buckets in enumerate(layout):
        assert buckets, f"channel {c} has no buckets"
        lengths = [length for length, _ in buckets]
        counts = [count for _, count in buckets]
        assert lengths == sorted(lengths), f"channel {c}: bucket lengths are not ascending"
        assert len(set(lengths)) == len(lengths), f"channel {c}: duplicate bucket lengths"
        assert all(length >= 1 for length in lengths), f"channel {c}: non-positive length"
        assert all(count >= 1 for count in counts), f"channel {c}: empty bucket"
        assert sum(counts) == plan.n_w, (
            f"channel {c}: buckets cover {sum(counts)} planes, plan.n_w is {plan.n_w}"
        )
        assert max(lengths) == per_chan_max[c] == int(padded[c].max()), (
            f"channel {c}: widest bucket {max(lengths)}, reported per-channel max "
            f"{per_chan_max[c]}, independently recomputed {int(padded[c].max())}"
        )

        expanded = np.repeat(np.array(lengths), np.array(counts))
        assert np.all(expanded >= np.sort(padded[c])), (
            f"channel {c}: some plane is bucketed below its own padded window "
            "length, which would drop rows the dense path weights"
        )

    assert max(per_chan_max) == plan.max_window_size, (
        "plan.max_window_size must stay the global maximum padded window length"
    )


# =============================================================================
# 2. numerical equivalence -- against dense_scan, never against windowed_scan
# =============================================================================

# The oracle is ``dense_scan`` and not ``windowed_scan`` on purpose. A
# windowed-against-windowed comparison shares the window bounds with the thing
# under test, so a boundary row dropped from both sides cancels out of the
# difference and the comparison passes while both are wrong; issue #15's
# review established that and issue #22's DFT cell followed it. ``dense_scan``
# gives every plane every row and so has no window bound to share.
_EQUIVALENCE_CELLS = [
    pytest.param(EDA2, (1.0,), True, id="EDA2-1chan-folded"),
    pytest.param(EDA2, MULTI_CHAN_FREQ_FACTORS, True, id="EDA2-4chan-folded"),
    pytest.param(EDA2, MULTI_CHAN_FREQ_FACTORS, False, id="EDA2-4chan-unfolded"),
    pytest.param(NARROW, (1.0,), True, id="NARROW-1chan-folded"),
]


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("w_strategy", WINDOWED_STRATEGIES)
@pytest.mark.parametrize(("telescope", "freq_factors", "hermitian"), _EQUIVALENCE_CELLS)
def test_the_bucketed_windowed_path_agrees_with_dense_scan(
    telescope: Telescope,
    freq_factors: tuple[float, ...],
    hermitian: bool,
    w_strategy: str,
    op: str,
) -> None:
    """Bucketing changes the reduction order and nothing else.

    Every cell here carries :func:`_assert_windows_are_strict`, so the
    windowed path under test really is slicing a strict subset of the rows and
    the comparison is not two spellings of the same computation.

    Bound: :data:`STRATEGY_TOL`, whose float64 half is AGENTS.md section 6's
    eps-independent strategy-equivalence constant and whose float32 half is
    10x this module's own measured worst case (see the constant for why the
    shared 1.3e-6 does not survive a 4000-row fixture). Measured at
    ``ab7fbbd`` (float64, eps 1e-6, seed 0, all three windowed strategies,
    both operators), worst cell by cell:

    * EDA2 off30 4-channel folded: forward 3.957e-17, adjoint 2.704e-15;
    * EDA2 off30 4-channel unfolded: forward 4.530e-17, adjoint 2.178e-15;
    * NARROW off30: forward 4.177e-17, adjoint 9.223e-15.

    i.e. four orders of magnitude of headroom under the bound, and the bound
    is the shared one rather than a fitted number.
    """
    plan = _plan(telescope, 30.0, freq_factors=freq_factors, hermitian=hermitian)
    _assert_windows_are_strict(plan)
    image, vis = _inputs(plan, telescope)
    arg = image if op == "dirty2vis" else vis

    reference = _call(op, plan, arg, w_strategy="dense_scan")
    got = _call(op, plan, arg, w_strategy=w_strategy)
    assert _rel(got, reference) < STRATEGY_TOL, (
        f"{w_strategy} {op} disagrees with dense_scan by {_rel(got, reference):.3e} "
        f"on {telescope.name} off30 (hermitian={hermitian}, n_chan={plan.n_chan}, "
        f"max_window_size={plan.max_window_size} of {plan.n_rows} rows)"
    )


@requires_x64
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_equivalence_still_holds_once_the_windows_are_actually_bucketed(op: str) -> None:
    """The same comparison, with the bucketing asserted to be live.

    :func:`test_the_bucketed_windowed_path_agrees_with_dense_scan` passes at
    ``ab7fbbd`` and is deliberately written so that it does -- it is the
    regression net, and a regression net that cannot run before the change
    protects nothing. This test is the same comparison with the extra
    assertion that the plan really did split its planes into more than one
    size class, so that after issue #26 lands there is at least one cell whose
    *subject* is the bucketed path rather than the unbucketed one it currently
    exercises. It fails today on the bucket table being absent.
    """
    plan = _plan(EDA2, 30.0, freq_factors=MULTI_CHAN_FREQ_FACTORS)
    _assert_windows_are_strict(plan)
    image, vis = _inputs(plan, EDA2)
    arg = image if op == "dirty2vis" else vis

    layout = _bucket_layout(plan)
    assert any(len(buckets) > 1 for buckets in layout), (
        "no channel of this plan has more than one bucket, so the comparison "
        "below would be against the unbucketed path"
    )

    reference = _call(op, plan, arg, w_strategy="dense_scan")
    for w_strategy in WINDOWED_STRATEGIES:
        got = _call(op, plan, arg, w_strategy=w_strategy)
        assert _rel(got, reference) < STRATEGY_TOL, (
            f"{w_strategy} {op} disagrees with dense_scan by "
            f"{_rel(got, reference):.3e} on a bucketed plan"
        )


# ``freq`` factors chosen so that the channels which share a bucket table are
# *not* contiguous in channel index. ``make_plan`` does not require a sorted
# ``freq`` and nothing downstream sorts it, but every other multi-channel
# fixture in this repository passes an ascending one
# (``linspace(0.95, 1.05, 4)``, ``[0.9, 1.0, 1.1]``,
# :data:`MULTI_CHAN_FREQ_FACTORS`, ``[1e9, 1.5e9]``) -- and window size is
# monotone in frequency, so with an ascending ``freq`` equal bucket tables
# always fall in contiguous runs and the channel-group reassembly is the
# identity permutation. Measured: 256 repository-shaped plans (four
# telescopes x two pointings x both ``hermitian`` settings x eps
# {1e-6, 1e-3} x eight frequency-factor sets, including all four this
# repository uses) plus 480 random ascending draws of two to five channels
# spanning 0.6x to 2.5x the telescope frequency -- the identity on every one
# of the 736. This fixture is the smallest departure from that --
# channels 0 and 2 at the telescope frequency and channel 1 at twice it, so
# the groups come out ``[(0, 2), (1,)]`` and their concatenation is
# ``[0, 2, 1]``.
_NON_MONOTONE_FREQ_FACTORS = (1.0, 2.0, 1.0)


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("w_strategy", WINDOWED_STRATEGIES)
def test_interleaved_channel_groups_are_reassembled_in_channel_order(
    w_strategy: str, op: str
) -> None:
    """Channels that share a bucket table need not be adjacent in the plan.

    A bucket's slice length is a static shape, so the channel axis can only be
    mapped over channels that bucket identically; the implementation therefore
    groups channels by bucket table, runs each group, concatenates the results
    **in group order** and permutes them back into channel order. That last
    step is a no-op on every fixture this repository can build, which is
    exactly why it needs a cell of its own: with this test deselected,
    deleting the permutation leaves the whole ``--runslow`` suite
    byte-identical, pass for pass and skip for skip, because with an ascending
    ``freq`` the groups are already contiguous ascending runs. See
    :data:`_NON_MONOTONE_FREQ_FACTORS` for how far that was checked.

    The implementation skips the permutation when it *is* the identity, which
    makes the reordering branch rarer still -- on every repository fixture the
    fast path is the one taken and the ``argsort`` never runs. So this cell is
    what pins both halves of that branch, and it was checked against both
    mutations: deleting the reassembly outright (``return out``
    unconditionally), and inverting the identity test so the fast path and the
    reorder swap places. Each fails all six cells here, with the readings
    below, and neither is caught anywhere else in the suite.

    The comparison is per channel and not on the stacked result, because a
    permutation of the channel axis is invisible to a norm over the whole
    array only if the channels happen to be equal -- and, more usefully,
    because a per-channel report names *which* channels were swapped.

    Measured on this fixture (EDA2 off30, ``freq = f * [1.0, 2.0, 1.0]``,
    seed 0, eps 1e-6, float64, ``hermitian=True``, ``max_window_size`` 126 of
    400 rows, per-channel maxima 126 / 72 / 126, groups ``[(0, 2), (1,)]``),
    worst per-channel disagreement with ``dense_scan`` over the three windowed
    strategies: forward 1.323e-16, adjoint 3.464e-15. Under either mutation
    channels 1 and 2 come back swapped and all six cells fail, at the same
    readings both times -- forward ``[1.323e-16, 1.477e+00, 1.373e+00]`` and
    adjoint ``[2.888e-15, 1.415e+00, 1.562e+00]``, channel 0 (which is first
    in its own group and therefore lands in the right slot either way) being
    the one that stays correct.
    """
    plan = _plan(EDA2, 30.0, freq_factors=_NON_MONOTONE_FREQ_FACTORS)
    _assert_windows_are_strict(plan)
    assert plan.n_chan == len(_NON_MONOTONE_FREQ_FACTORS)

    # The grouping, derived from the public plan surface rather than imported:
    # channels in first-appearance order of their bucket table.
    by_table: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for chan, table in enumerate(_bucket_layout(plan)):
        by_table.setdefault(table, []).append(chan)
    grouped_order = [chan for chans in by_table.values() for chan in chans]
    assert len(by_table) > 1, (
        "vacuous fixture: every channel of this plan has the same bucket table, "
        "so there is only one group and no reassembly happens at all "
        f"({_bucket_layout(plan)})"
    )
    assert grouped_order != sorted(grouped_order), (
        "vacuous fixture: the bucket-table groups are already contiguous ascending "
        f"runs of channel index ({grouped_order}), so the reassembly this test is "
        "about is the identity permutation and deleting it would change nothing"
    )

    image, vis = _inputs(plan, EDA2)
    arg = image if op == "dirty2vis" else vis
    reference = _call(op, plan, arg, w_strategy="dense_scan")
    got = _call(op, plan, arg, w_strategy=w_strategy)

    # ``dirty2vis`` puts the channel axis last ((n_rows, n_chan)) and
    # ``vis2dirty`` first ((n_chan, n_l, n_m)); take the channel either way.
    def channel(x: Any, chan: int) -> Any:
        return x[..., chan] if op == "dirty2vis" else x[chan]

    per_chan = [_rel(channel(got, c), channel(reference, c)) for c in range(plan.n_chan)]
    assert max(per_chan) < STRATEGY_TOL, (
        f"{w_strategy} {op} disagrees with dense_scan per channel by "
        f"{[f'{v:.3e}' for v in per_chan]} on a plan whose bucket-table groups are "
        f"{[tuple(chans) for chans in by_table.values()]} -- channels that share a "
        "bucket table are not contiguous here, so a result concatenated in group "
        "order and not permuted back comes out with those channels transposed"
    )


# =============================================================================
# 3. per-channel window sizes (implementation-plan item 2)
# =============================================================================


@requires_x64
def test_per_channel_window_sizes_are_used_rather_than_one_global_max() -> None:
    """A global maximum is measurably wasteful on a plan whose channels differ.

    Fixture: EDA2 off30, seed 0, four channels at 1.0 / 1.3 / 1.6 / 2.0 times
    the telescope frequency, eps 1e-6, float64, hermitian=True. Measured at
    ``ab7fbbd``: n_w = 105, n_rows = 400, ``max_window_size`` 126 (so the
    windows are a strict subset, 0.315 of the rows), per-channel maxima
    126 / 103 / 84 / 72, ``live_row_count`` 11200, and
    ``window_padding_overhead`` 4.7250.

    Charging every channel its own maximum instead of the global one gives
    ``105 * (126 + 103 + 84 + 72) / 11200 = 3.6094``, i.e. per-channel sizing
    alone removes 23.6% of the padded row-work before any bucketing. That is
    the number this test asserts against: whatever the bucketing does on top,
    the reported overhead may not be worse than per-channel maxima alone,
    which is the weakest statement that is false if a global maximum is still
    being charged.

    The channel spread is load-bearing and is why this fixture does not use
    the +/-5% frequencies the other strategy modules use: measured on EDA2
    off30 at 0.95 / 1.0 / 1.05 the per-channel maxima are 129 / 124 / 122 and
    the same comparison is 2.6336 against 2.7179, a 3% difference that would
    make this test nearly vacuous on its own axis.
    """
    plan = _plan(EDA2, 30.0, freq_factors=MULTI_CHAN_FREQ_FACTORS)
    _assert_windows_are_strict(plan)
    assert plan.n_chan == len(MULTI_CHAN_FREQ_FACTORS)

    padded = _independent_padded_sizes(plan)
    independent_per_chan = tuple(int(v) for v in padded.max(axis=1))
    assert len(set(independent_per_chan)) == plan.n_chan, (
        "vacuous fixture: two channels share a maximum window size, so a global "
        f"maximum costs them nothing ({independent_per_chan})"
    )
    assert max(independent_per_chan) >= 1.5 * min(independent_per_chan), (
        "vacuous fixture: the per-channel window maxima are too close together "
        f"for a global maximum to be measurably wasteful ({independent_per_chan})"
    )

    assert _max_window_size_per_chan(plan) == independent_per_chan, (
        "the plan's per-channel window maxima disagree with an independent "
        f"recomputation: {_max_window_size_per_chan(plan)} vs {independent_per_chan}"
    )

    per_chan_work = plan.n_w * sum(independent_per_chan)
    assert per_chan_work < _unbucketed_padded_work(plan)
    assert _effective_padded_work(plan) <= per_chan_work, (
        "the padded row-work is above what per-channel maxima alone would cost "
        f"({_effective_padded_work(plan)} > {per_chan_work}), so some channel is "
        "still being charged the plan-wide maximum"
    )


# =============================================================================
# 4. bucketing actually buckets, at the level of the emitted program
# =============================================================================


@requires_x64
@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("w_strategy", WINDOWED_STRATEGIES)
def test_the_lowered_windowed_loop_takes_more_than_one_slice_length(
    w_strategy: str, op: str
) -> None:
    """Rows outside a bucket's slice are never touched -- read off the lowering.

    "Untouched" has no numerical signature: a row outside a plane's window
    gets a zero kernel weight anyway, which is precisely why bucketing is
    allowed to skip it. The claim is about *work*, so it is asserted where
    work is visible -- the lowered StableHLO, in which a slice of length
    ``K`` out of the ``(n_rows,)`` w-sorted arrays reads exactly ``K`` rows.
    See :func:`_window_slice_lengths` for the two forms it takes.

    Measured at ``ab7fbbd`` on the NARROW fixture (off30, seed 0, eps 1e-6,
    float64, hermitian=True, n_w = 138, max_window_size = 1072 of 4000 rows,
    85 distinct padded window sizes from 2 to 1072 with a median of 24): all
    three windowed strategies lower ``dirty2vis`` to exactly one distinct
    slice length, ``{1072}``, and ``vis2dirty`` to ``{1, 1072}``.
    ``dense_scan`` lowers to none at all in either direction, which is the
    control that the probe is reading the windowed slice and not something
    incidental.

    The stray ``1`` is not a window: it is the adjoint's
    ``plan.flip_sign[plan.sort_perm]``, a whole-row gather that lowers with
    ``slice_sizes = array<i64: 1>`` out of ``tensor<4000xi8>`` and which
    :func:`_window_slice_lengths` cannot tell from a one-row window. So at
    ``ab7fbbd`` the three ``vis2dirty`` cells reach the ``> 1`` assertion and
    pass it for the wrong reason, then fail at :func:`_bucket_layout` on the
    absent bucket table -- which is why the ``found <= bucket_lengths``
    assertion at the foot of this test matters: 1 is not a bucket length on
    this plan (its narrowest window is 2), so a probe reading that gather
    after bucketing would fail there. The shipped source keeps the probe
    honest at the source instead, by gathering ``flip_sign`` through an
    ``(n_rows, 1)`` view so that every whole-row gather of a plan leaf lowers
    at rank 2 and the only rank-1 row slice left is a w-plane window; the
    alternative was to anchor this module's regex on the operand's element
    type (``tensor<{n}x(?:f|complex)``) rather than on ``[a-z]``. Nothing but
    a comment in ``wgridder.py`` enforces the convention that was chosen.

    After bucketing the set must be the plan's own bucket lengths -- a length
    the plan does not claim as a bucket is work nothing accounts for -- and
    must have more than one element on this fixture.
    """
    plan = _plan(NARROW, 30.0)
    _assert_windows_are_strict(plan)
    image, vis = _inputs(plan, NARROW)
    arg = image if op == "dirty2vis" else vis

    dense = _window_slice_lengths(
        _lowered_text(lambda x: _call(op, plan, x, w_strategy="dense_scan"), arg),
        plan.n_rows,
    )
    assert dense == set(), (
        "control failed: dense_scan lowers to a row slice, so the probe below is "
        f"not specific to the windowed loop (found {sorted(dense)})"
    )

    found = _window_slice_lengths(
        _lowered_text(lambda x: _call(op, plan, x, w_strategy=w_strategy), arg),
        plan.n_rows,
    )
    assert found, (
        f"probe found no row slice in the lowered {w_strategy} {op}; either the "
        "windowed loop stopped slicing the w-sorted arrays or the StableHLO "
        "spelling this probe matches has changed"
    )

    assert len(found) > 1, (
        f"{w_strategy} {op} still takes a single slice length {sorted(found)} on a "
        f"plan with {len(set(_independent_padded_sizes(plan).ravel().tolist()))} "
        "distinct padded window sizes: every plane is reading the widest window's "
        "worth of rows"
    )
    assert max(found) <= plan.max_window_size

    bucket_lengths = {length for per_chan in _bucket_layout(plan) for length, _ in per_chan}
    assert found <= bucket_lengths, (
        f"{w_strategy} {op} slices lengths {sorted(found - bucket_lengths)} that the "
        f"plan does not declare as buckets (declared: {sorted(bucket_lengths)})"
    )


# =============================================================================
# 5. memory
# =============================================================================

# ``windowed_vmap`` forward transient on the NARROW fixture at ``ab7fbbd``
# (macOS arm64 CPU backend, float64, eps 1e-6, seed 0, hermitian=True,
# n_chan=1, AGENTS.md section 6's memory protocol). The comparison numbers on
# the same call: dense_scan 72,328, windowed_scan 128,032, windowed_chunked(32)
# 1,272,104, dense_vmap 17,664,000.
_WINDOWED_VMAP_FORWARD_TEMP_AT_AB7FBBD = 13_565_952

# The bound, derived rather than chosen. On this fixture the transient splits
# into a plane-stack term ``n_w * image = 138 * 4096 = 565,248`` bytes (4.17%),
# which bucketing does not touch, and a row-slice term of 92.0% or more which
# scales with the padded row-work. Section 1's gate caps the effective overhead
# at 1.5 against a measured 5.2832 here, so the row-slice term may keep at most
# 1.5 / 5.2832 = 28.4% of its size and the modelled total is
# 0.0417 + 0.9583 * 0.284 = 31.4% of today's. The gate is set at 50%, i.e. 1.6x
# the model, so it is a statement about the direction and rough size of the
# change and not a fit to a predicted byte count. Today's value is 100% of
# itself and fails it by 2x.
#
# Shipped: 3,134,168 B, 23.1% of ``ab7fbbd``. Two terms, not one -- the
# row-slice work the model is about, and 2,048,000 B of
# ``wgridder._FORWARD_ROW_LANES * n_rows`` row vectors, which is what keeps the
# per-plane accumulate collision-free (see
# ``test_no_two_planes_accumulate_into_one_row_vector_in_the_forward``). The
# lane term is why the gate is not tighter than the model: at an uncapped one
# lane per plane, which is ``ab7fbbd``'s form, NARROW's widest bucket alone
# would put 83 of them live.
_MEMORY_GATE_FRACTION = 0.5


@requires_x64
def test_windowed_vmap_temp_memory_falls_with_bucketing() -> None:
    """The strategy that holds every plane's slice at once pays sum, not max.

    ``windowed_vmap`` is the right subject and the only one: a scan holds one
    plane's slice at a time, so its peak is the *largest* bucket, which is
    still ``max_window_size``. Measured at ``ab7fbbd`` on this fixture the two
    forward transients are 13,565,952 (vmap) and 128,032 (scan) bytes, and the
    vmap number is 91.70 bytes per ``(plane, row)`` of
    ``n_w * max_window_size = 138 * 1072 = 147,936`` -- i.e. it tracks the
    padded row-work almost exactly, which is what makes a bound derivable from
    section 1's gate. See the constants above for the derivation.

    ``windowed_scan`` is asserted here too, but only not to *regress*: nothing
    in issue #26 should make the one-plane-at-a-time peak bigger, and if
    bucketing were implemented by materialising several differently-sized
    copies of the sorted arrays it would.
    """
    plan = _plan(NARROW, 30.0)
    _assert_windows_are_strict(plan)
    image, _ = _inputs(plan, NARROW)

    vmap_temp = _temp_bytes(
        lambda x: _call("dirty2vis", plan, x, w_strategy="windowed_vmap"), image
    )
    plane_stack = plan.n_w * plan.n_l * plan.n_m * np.dtype(plan.complex_dtype).itemsize
    assert vmap_temp > plane_stack, (
        "sanity: the measured transient is below one plane stack, so this "
        "fixture is no longer row-slice dominated and the bound below is not "
        "derived from anything"
    )

    bound = int(_MEMORY_GATE_FRACTION * _WINDOWED_VMAP_FORWARD_TEMP_AT_AB7FBBD)
    assert vmap_temp <= bound, (
        f"windowed_vmap forward transient is {vmap_temp} bytes, above the "
        f"{bound}-byte bound derived from the {PADDING_OVERHEAD_GATE} padding "
        f"gate and the {_WINDOWED_VMAP_FORWARD_TEMP_AT_AB7FBBD} bytes measured "
        "at ab7fbbd"
    )

    scan_temp = _temp_bytes(
        lambda x: _call("dirty2vis", plan, x, w_strategy="windowed_scan"), image
    )
    assert scan_temp <= 1.1 * 128_032, (
        f"windowed_scan forward transient rose to {scan_temp} bytes from the "
        "128,032 measured at ab7fbbd; a scan's peak is one bucket, so bucketing "
        "should leave it where it is"
    )


# A ``stablehlo.scatter`` with its attribute dict, its region, and the operand
# type tuple that follows it. The region is matched non-greedily up to the
# ``}) : (`` that closes it, which is the only place that sequence occurs.
_SCATTER_OP = re.compile(
    r'"stablehlo\.scatter"\([^)]*\)\s*<\{(?P<attrs>.*?)\}>\s*\(\{.*?\}\)\s*:\s*'
    r"\((?P<types>[^)]*)\)\s*->",
    re.S,
)
_TENSOR_DIMS = re.compile(r"tensor<([0-9x]*)x?([a-z][^>]*)>")


def _forward_scatters(text: str) -> list[tuple[str, tuple[int, ...]]]:
    """``(attribute text, updates shape)`` for every scatter in the lowered IR."""
    found: list[tuple[str, tuple[int, ...]]] = []
    for match in _SCATTER_OP.finditer(text):
        tensors = _TENSOR_DIMS.findall(match.group("types"))
        assert len(tensors) >= 3, (
            "a stablehlo.scatter takes (operand, indices, updates); this one parsed "
            f"as {tensors!r}, so the probe is not reading the IR it thinks it is"
        )
        dims = tuple(int(d) for d in tensors[-1][0].split("x") if d)
        found.append((match.group("attrs"), dims))
    return found


@pytest.mark.parametrize("w_strategy", ["windowed_vmap", "windowed_chunked"])
def test_no_two_planes_accumulate_into_one_row_vector_in_the_forward(w_strategy: str) -> None:
    """The forward's plane accumulate must be collision-free *in the lowering*.

    The windows of different w-planes overlap, physically: at eps 1e-6 every
    visibility is inside the w-kernel's support on exactly 7.00 planes, on all
    ten (telescope, pointing) cells of ``tests/conftest.py`` and on both of
    :data:`NARROW`, and the padded windows this issue buckets cover each sorted
    row 7.00-9.70 times on average and 8-81 times at the worst row. So a
    forward that accumulates every plane into one shared ``(n_rows,)`` vector
    is asking XLA to combine several writes per element, and XLA has to assume
    they can collide.

    On a CPU that costs nothing measurable. On a GPU it is the difference
    between the operator being usable and not: measured on one GH200 (Daint,
    eps 1e-6, float64, ``n_chan = 1``, realistic sizes), the shared-carry form
    ran this strategy's forward at 351.9 ms against 38.9 ms for the
    collision-free form on GH200_large off30 (2048^2, 50k rows, ``n_w = 26``)
    and at 1842.4 ms against 33.0 ms on MeerKAT off30 (2700^2, 302400 rows,
    ``n_w = 14``) -- 9.05x and 55.76x, and on *less* NUFFT work, because the
    bucketing had cut the padded overhead from 2.7389 to 1.2861 and from
    2.0000 to 1.1158 on those two plans. The adjoint, which accumulates an
    image and never a row-indexed carry, moved by 1.09x and 1.15x.

    **This cell exists because no other kind of test in this repository can
    see that.** Every value test passes either way -- the two forms are the
    same arithmetic in a different reduction order. Every CPU timing passes
    either way: measured on the review machine (macOS arm64, nthreads=1,
    median of 11 interleaved rounds, both forms imported into one process so
    their samples alternate), the forward ratio of this form to the
    shared-carry one is 0.985 / 1.067 / 0.712 / 0.683 for ``windowed_vmap`` on
    MWA_extended off30 / MeerKAT off30 / EDA2 off30 / :data:`NARROW` off30 --
    i.e. the difference the GPU reads as 9-56x reads on a CPU as noise, in
    whichever direction the fixture happens to fall. And the transient-memory
    gate above passes either way. The lowering is the only CPU-visible signal,
    so the lowering is what is asserted.

    What is asserted, precisely: every ``stablehlo.scatter`` in the lowered
    forward writes updates that XLA can place without combining, by being one
    of

    * marked ``unique_indices = true`` -- no two updates share an element;
    * carrying ``input_batching_dims`` -- each batch element owns its own slice
      of the operand, which is the shape a ``vmap`` over per-plane row vectors
      takes (and the shape ``ab7fbbd`` had, one lane per plane and uncapped);
    * or writing exactly ``n_rows`` elements, which is the single sorted-to-
      input row permutation every windowed forward ends with.

    That is a structural proxy and not a proof of speed -- a scatter can carry
    batching dims and still collide *within* a batch element, and there is no
    GPU in this suite to time. It is exact about the thing that regressed: an
    accumulate whose updates are one ``(n_planes, slice_length)`` block per
    bucket, unbatched and not unique, matches none of the three.

    Non-vacuity is checked two ways: the fixture's windows are a strict subset
    of the rows (:func:`_assert_windows_are_strict`), and the plane accumulate
    has to actually be in the IR -- at least one scatter writing a rank-2 block
    -- so that a rewrite which lost it would fail here rather than pass
    silently.
    """
    plan = _plan(NARROW, 30.0)
    _assert_windows_are_strict(plan)
    image, _ = _inputs(plan, NARROW)
    n_rows = int(plan.n_rows)

    text = _lowered_text(lambda x: _call("dirty2vis", plan, x, w_strategy=w_strategy), image)
    scatters = _forward_scatters(text)
    assert scatters, (
        "no stablehlo.scatter in the lowered forward at all -- the probe is not "
        "seeing the operator it thinks it is"
    )
    assert any(len(dims) >= 2 for _, dims in scatters), (
        "no scatter writes a rank-2 block, so the per-plane accumulate is not in "
        f"this IR and there is nothing here to gate. Shapes seen: "
        f"{[dims for _, dims in scatters]}"
    )

    offenders = [
        dims
        for attrs, dims in scatters
        if "unique_indices = true" not in attrs
        and "input_batching_dims" not in attrs
        and dims != (n_rows,)
    ]
    assert not offenders, (
        f"{w_strategy} forward: {len(offenders)} scatter(s) with updates of shape "
        f"{offenders} are neither marked unique, nor per-plane batched, nor the "
        f"({n_rows},) row permutation. That is the shape of a shared row "
        "accumulator, which every plane's window writes into and which cost "
        "9.05x-55.76x on a GH200 -- see this test's docstring."
    )


# =============================================================================
# 6. jit, vmap, grad, both precisions, both hermitian settings
# =============================================================================


@pytest.mark.parametrize("hermitian", [True, False], ids=["folded", "unfolded"])
@pytest.mark.parametrize("w_strategy", WINDOWED_STRATEGIES)
def test_the_windowed_path_survives_jit_and_vmap(w_strategy: str, hermitian: bool) -> None:
    """``jit(vmap(op))`` over a batch of images still matches ``dense_scan``.

    Compared jit-to-jit throughout: on this fixture an eager ``vmap`` and a
    jitted one differ by a measured 4.09e-10 relative, identically for
    ``dense_scan`` and ``windowed_scan``, which is a property of the execution
    mode and would swamp the 1e-11 strategy bound if the two sides were
    mixed. Measured jit-to-jit at ``ab7fbbd`` on NARROW off30, float64,
    eps 1e-6, batch of 2: 2.41e-15.

    Runs on both precision legs (float32 at eps 1e-3, see :data:`EPSILON`) and
    on both plan geometries. The vacuity guard is what keeps it meaningful:
    the batched call is over a plan whose windows are a strict subset of the
    rows, so a batching rule that lost the per-plane window bounds would show
    up here.
    """
    plan = _plan(NARROW, 30.0, hermitian=hermitian)
    _assert_windows_are_strict(plan)
    image, _ = _inputs(plan, NARROW)
    batch = jnp.stack([image, 2.0 * image])

    reference = jax.jit(jax.vmap(lambda x: _call("dirty2vis", plan, x, w_strategy="dense_scan")))(
        batch
    )
    got = jax.jit(jax.vmap(lambda x: _call("dirty2vis", plan, x, w_strategy=w_strategy)))(batch)
    assert _rel(got, reference) < STRATEGY_TOL


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("w_strategy", WINDOWED_STRATEGIES)
def test_gradients_through_the_windowed_path_match_dense_scan(w_strategy: str, op: str) -> None:
    """Reverse mode through a bucketed loop is still the other operator's forward.

    Both operators are linear primitives whose transposes are each other
    (issue #21), so a change to the plane loop's slicing has to leave the
    gradient alone. Measured at ``ab7fbbd`` on EDA2 off30, four channels,
    float64, eps 1e-6, hermitian=True, ``loss = sum |op(x)|^2``: the three
    windowed strategies' forward gradients differ from ``dense_scan``'s by
    1.301e-13, 1.251e-13 and 1.301e-13 -- reduction-order noise, two orders
    under the 1e-11 bound.
    """
    plan = _plan(EDA2, 30.0, freq_factors=MULTI_CHAN_FREQ_FACTORS)
    _assert_windows_are_strict(plan)
    image, vis = _inputs(plan, EDA2)
    arg = image if op == "dirty2vis" else vis

    def loss(strategy: str) -> Callable[[Any], Any]:
        def inner(x: Any) -> Any:
            return jnp.sum(jnp.abs(_call(op, plan, x, w_strategy=strategy)) ** 2)

        return inner

    reference = jax.jit(jax.grad(loss("dense_scan")))(arg)
    got = jax.jit(jax.grad(loss(w_strategy)))(arg)
    assert _rel(got, reference) < STRATEGY_TOL


def test_the_vacuity_guard_rejects_the_fixtures_it_is_meant_to_reject() -> None:
    """The guard is load-bearing, so it gets its own test.

    Every window-bound claim in this module is only as good as
    :func:`_assert_windows_are_strict`, and a guard that silently stopped
    guarding would restore exactly the defect issue #26's second comment warns
    about. MeerKAT at zenith is one of the seven repo cells measured at
    ``max_window_size == n_rows`` (600 of 600 rows, at eps 1e-6 float64 where
    ``n_w`` is 8 and at eps 1e-3 float32 where it is 5), so the guard must
    reject it; EDA2 off30 is one of the three measured strictly below (126 of
    400 in float64, 79 of 400 in float32), so it must accept it.
    """
    with pytest.raises(AssertionError, match="vacuous fixture"):
        _assert_windows_are_strict(_plan(MEERKAT, 0.0))
    _assert_windows_are_strict(_plan(EDA2, 30.0))
