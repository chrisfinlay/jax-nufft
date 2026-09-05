"""The shipped ``w_strategy`` default is ``"auto"`` on both operators (issue #46).

Through v0.1.2 both public operators defaulted to ``w_strategy="dense_scan"``
and the ``"auto"`` heuristic added in Part 4 was opt-in, i.e. never reached
unless a caller asked for it by name. On GPU that made the shipped default the
*worst* available choice: measured on one GH200 against ducc0 on 72 Grace
cores (issue #46's table), ``dense_scan`` runs 1.4-5.6x slower than ducc0 on
the same node in five of that table's six cells, while the strategy
``"auto"`` picks there runs 1.4-6.3x *faster* in all six. (The sixth cell,
the GH200_large off30 adjoint, had the old default about 1.16x faster than
ducc0 -- still 2.0x off the ``dense_vmap`` column of the same table.) The heuristic was fine; only
its reachability was broken.

What this module pins
---------------------
* the *signature* default of both operators is ``"auto"`` -- checked with
  :func:`inspect.signature`, not only through behaviour, so "changed one
  operator and forgot the other" is named as such;
* leaving ``w_strategy`` unset actually takes the ``"auto"`` path at call
  time (checked at the JIT boundary, with the platform pinned so the
  resolved name is one the old default could never produce);
* an explicit ``w_strategy`` is still forwarded untouched -- including
  ``"dense_scan"``, which issue #46 promises restores the old *code path*
  (not the old numbers -- other changes in the same release moved those);
* what ``"auto"`` resolves to on every repository CPU fixture, forward and
  adjoint, as a literal table: a later change to the heuristic must break a
  test rather than silently move the shipped default;
* the ``nthreads`` interaction from issue #24: ``_resolve_nthreads`` derives
  its default from the *resolved* strategy and is never handed the literal
  ``"auto"`` by the operators, a defaulted call canonicalises exactly *once*
  (issue #46's checkbox -- ``_resolve_nthreads`` short-circuits on an
  already-canonical name rather than resolving a second time), and
  both-defaults gives the same ``nthreads`` as passing the resolved strategy
  explicitly;
* JIT cache sharing: the default call and the explicit resolved call hit the
  same compiled executable, not merely equal numbers.

Numerical equivalence of the new default against the four explicit
strategies is *not* here: ``tests/test_strategies_equivalent.py`` owns the
1e-11 strategy-equivalence bound. It has no default-call entrant, and
deliberately so -- ``"auto"`` resolves before the JIT boundary, so such an
entrant is bit-identical to the explicit strategy it resolves to and gates
nothing; see the comment on ``STRATEGY_TOL`` there. What makes that
reasoning sound is pinned below, by
``test_default_shares_the_compiled_executable_with_the_explicit_resolved``.

Platform pinning
----------------
Almost every behavioural test below patches ``jax.devices`` (via
``_patch_platform``, the same trick ``tests/test_auto_strategy.py`` uses) so
the heuristic's platform branch is deterministic wherever pytest runs. This
is load-bearing for more than determinism: on CPU the forward heuristic
always resolves to ``dense_scan``, which is exactly the *old* default, so a
CPU-only behavioural test could not tell "default is auto" from "default is
dense_scan" at all. Pinning ``"gpu"`` makes the two answers different
(``dense_vmap``) and the assertion discriminating. Only the heuristic reads
the patched platform -- the transforms themselves still run on whatever
backend pytest has, which is why the JIT-cache tests below execute real
FINUFFT calls under the patch.

Precision
---------
Precision-aware from the start (so it is *not* in
``conftest.collect_ignore``): plans are built with ``dtype=real_dtype`` and
``epsilon`` comes from ``tol(1e-6, 1e-4)`` -- above ``planning``'s float32
accuracy floor of 1e-5, so the float32 leg does not trip
``filterwarnings = ["error"]`` on the accuracy warning. The resolved-strategy
table is precision-keyed because ``eps`` and ``dtype`` change ``n_w`` and
``w_kernel_width``, which are the heuristic's inputs.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

import jax_nufft.wgridder as wgridder
from jax_nufft import dirty2vis, make_plan, vis2dirty
from jax_nufft._types import WStrategy
from jax_nufft.wgridder import (
    _auto_w_strategy,
    _canonicalise_w_strategy,
    _dirty2vis_jit,
    _resolve_nthreads,
    _vis2dirty_jit,
)
from tests.conftest import (
    EDA2,
    GH200_LARGE,
    MEERKAT,
    MWA_COMPACT,
    MWA_EXTENDED,
    X64,
    Telescope,
    synthetic_uvw,
    tol,
)

CANONICAL: tuple[WStrategy, ...] = (
    "dense_scan",
    "dense_vmap",
    "windowed_scan",
    "windowed_vmap",
)

# Above the float32 accuracy floor (1e-5) on the float32 leg; see the module
# docstring's Precision section.
EPSILON = tol(1e-6, 1e-4)


# -- platform pinning -------------------------------------------------------


class _FakeDevice:
    """Stand-in for ``jax.devices()[0]``; only ``.platform`` is read."""

    def __init__(self, platform: str) -> None:
        self.platform = platform


def _patch_platform(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """Make ``_auto_w_strategy``'s ``jax.devices()[0].platform`` lookup
    return ``platform``. Mirrors ``tests/test_auto_strategy.py``."""
    monkeypatch.setattr(jax, "devices", lambda *a, **kw: [_FakeDevice(platform)])


# -- problems ---------------------------------------------------------------


def _offzenith_problem(real_dtype: DTypeLike, complex_dtype: DTypeLike, seed: int = 3):
    """A small off-zenith plan (192 rows, 64^2) with enough w-extent that the
    heuristic has a real choice to make.

    At eps=1e-6/float64 this gives n_w=56 at w_kernel_width=7, so the CPU
    heuristic picks ``dense_scan`` forward / ``windowed_scan`` adjoint and the
    GPU heuristic picks ``dense_vmap`` for both -- three distinct answers, so
    a test that pins the platform can discriminate all of them.
    """
    rng = np.random.default_rng(seed)
    n_rows = 192
    uvw = rng.normal(scale=300.0, size=(n_rows, 3))
    uvw[:, 2] = rng.normal(scale=400.0, size=n_rows)
    plan = make_plan(
        uvw=uvw,
        freq=np.array([200e6]),
        image_shape=(64, 64),
        pixsize_l=4e-3,
        pixsize_m=4e-3,
        epsilon=EPSILON,
        dtype=real_dtype,
    )
    image = jnp.asarray(rng.standard_normal((64, 64)), dtype=real_dtype)
    vis = jnp.asarray(
        rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1)),
        dtype=complex_dtype,
    )
    return plan, image, vis


def _plan_above_nthreads_cutoff(real_dtype: DTypeLike, complex_dtype: DTypeLike, seed: int = 4):
    """A plan with ``n_rows`` just above ``_NTHREADS_SMALL_N_ROWS``.

    The thread-default rule collapses every strategy family to ``1`` below
    that cutoff, so this is the only regime in which "the thread default came
    from the resolved strategy" is observable at all: above it the vmap
    family resolves to ``0`` and the scan family to ``1``. Image size stays
    tiny -- ``make_plan`` is host-side numpy and the tests that use this stub
    out the JIT call entirely.
    """
    rng = np.random.default_rng(seed)
    n_rows = wgridder._NTHREADS_SMALL_N_ROWS + 1
    uvw = np.zeros((n_rows, 3))
    uvw[:, 0] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 1] = rng.uniform(-50.0, 50.0, size=n_rows)
    uvw[:, 2] = rng.uniform(-3.0, 3.0, size=n_rows)
    plan = make_plan(
        uvw=uvw,
        freq=np.array([1.4e9]),
        image_shape=(16, 16),
        pixsize_l=0.005,
        pixsize_m=0.005,
        epsilon=EPSILON,
        dtype=real_dtype,
    )
    image = jnp.asarray(rng.standard_normal((16, 16)), dtype=real_dtype)
    vis = jnp.asarray(
        rng.standard_normal((n_rows, 1)) + 1j * rng.standard_normal((n_rows, 1)),
        dtype=complex_dtype,
    )
    return plan, image, vis


def _spy_jit(monkeypatch: pytest.MonkeyPatch, jit_name: str) -> list[dict]:
    """Replace ``jax_nufft.wgridder.<jit_name>`` with a stub recording every
    call's kwargs (``w_strategy`` and ``nthreads`` among them, both
    ``static_argnames``) and returning a dummy array.

    Same device as ``tests/test_jax_integration.py``'s boundary tests: the
    public wrapper returns the JIT function's result unchanged, so any return
    value is fine, and no FINUFFT work is done.
    """
    calls: list[dict] = []

    def stub(*args: object, **kwargs: object) -> jax.Array:
        calls.append(kwargs)
        return jnp.zeros(())

    monkeypatch.setattr(wgridder, jit_name, stub)
    return calls


_OPERATORS: dict[str, tuple[Callable[..., Any], str, bool]] = {
    # name -> (public callable, name of the jit function it calls, is_adjoint)
    "dirty2vis": (dirty2vis, "_dirty2vis_jit", False),
    "vis2dirty": (vis2dirty, "_vis2dirty_jit", True),
}


def _call(op: str, plan, image, vis, **kwargs):
    """Call ``dirty2vis(plan, image, ...)`` or ``vis2dirty(plan, vis, ...)``."""
    fn = _OPERATORS[op][0]
    return fn(plan, vis if op == "vis2dirty" else image, **kwargs)


# -- 1. the signature default ----------------------------------------------


def _signature_default(fn: Callable[..., Any]) -> object:
    return inspect.signature(fn).parameters["w_strategy"].default


def test_dirty2vis_signature_default_is_auto() -> None:
    """The *signature* default, not a value patched at call time: a caller
    reading ``help(dirty2vis)`` or binding the signature must see ``"auto"``."""
    got = _signature_default(dirty2vis)
    assert got == "auto", f"dirty2vis signature default is {got!r}, expected 'auto' (issue #46)"


def test_vis2dirty_signature_default_is_auto() -> None:
    """Adjoint counterpart."""
    got = _signature_default(vis2dirty)
    assert got == "auto", f"vis2dirty signature default is {got!r}, expected 'auto' (issue #46)"


def test_both_operators_share_the_same_default() -> None:
    """Changing one operator's default and not the other is the specific
    half-done edit this test exists to name."""
    fwd = _signature_default(dirty2vis)
    adj = _signature_default(vis2dirty)
    assert fwd == adj == "auto", (
        f"w_strategy defaults disagree or are not 'auto': "
        f"dirty2vis={fwd!r}, vis2dirty={adj!r}. Issue #46 changes both."
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_default_takes_the_auto_path_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
    op: str,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """Behavioural half of the default check: with the platform pinned to
    ``"gpu"``, ``"auto"`` resolves to a vmap-family strategy, which the old
    ``dense_scan`` default could never produce. Assert the exact
    ``w_strategy`` the wrapper hands the JIT boundary when the caller passes
    no ``w_strategy`` at all."""
    _patch_platform(monkeypatch, "gpu")
    plan, image, vis = _offzenith_problem(real_dtype, complex_dtype)
    _, jit_name, is_adjoint = _OPERATORS[op]
    expected = _auto_w_strategy(plan, is_adjoint=is_adjoint)
    assert expected in ("dense_vmap", "windowed_vmap"), (
        f"sanity: the GPU heuristic should pick a vmap-family strategy here, got {expected!r}"
    )

    calls = _spy_jit(monkeypatch, jit_name)
    _call(op, plan, image, vis)
    assert len(calls) == 1
    assert calls[0]["w_strategy"] == expected, (
        f"{op} with no w_strategy forwarded {calls[0]['w_strategy']!r}; the default must "
        f"resolve through 'auto' to {expected!r} on this (pinned) platform"
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_default_resolves_for_its_own_direction(
    monkeypatch: pytest.MonkeyPatch,
    op: str,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """The default must be resolved with the operator's *own* ``is_adjoint``.

    ``_canonicalise_w_strategy`` takes ``is_adjoint`` as an argument, so
    ``dirty2vis`` passing ``True`` (or ``vis2dirty`` passing ``False``) is a
    one-character edit that no test above would notice: on the pinned GPU
    platform this plan resolves to ``dense_vmap`` in both directions. Pin CPU
    instead, where the same plan resolves to ``dense_scan`` forward and
    ``windowed_scan`` adjoint, and assert each operator forwards its own
    direction's pick.

    The ``fwd_pick != adj_pick`` sanity assertion is what keeps this test from
    quietly gating nothing if a future heuristic change makes the two
    directions agree on this fixture: it fails loudly instead.
    """
    _patch_platform(monkeypatch, "cpu")
    plan, image, vis = _offzenith_problem(real_dtype, complex_dtype)
    fwd_pick = _auto_w_strategy(plan, is_adjoint=False)
    adj_pick = _auto_w_strategy(plan, is_adjoint=True)
    assert fwd_pick != adj_pick, (
        "this test needs a plan whose CPU forward and adjoint picks differ; both "
        f"came out {fwd_pick!r}, so it would pass under a swapped is_adjoint. "
        "Pick a different fixture rather than deleting the check."
    )

    _, jit_name, is_adjoint = _OPERATORS[op]
    expected = adj_pick if is_adjoint else fwd_pick
    calls = _spy_jit(monkeypatch, jit_name)
    _call(op, plan, image, vis)
    assert len(calls) == 1
    assert calls[0]["w_strategy"] == expected, (
        f"{op} at its default forwarded {calls[0]['w_strategy']!r}, the pick for the "
        f"*other* direction; it must resolve 'auto' with is_adjoint={is_adjoint}"
    )


# -- 2. explicit values are still honoured ----------------------------------


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
@pytest.mark.parametrize("w_strategy", CANONICAL)
def test_explicit_strategy_is_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
    op: str,
    w_strategy: WStrategy,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """Every *canonical* member of ``WStrategy``, passed explicitly, reaches
    the JIT boundary unchanged -- ``"dense_scan"`` included, which is the
    promise
    that an explicit ``dense_scan`` restores the pre-#46 *code path*. (Not
    the pre-#46 numbers: #16, #23 and #43 land in the same release and #16
    moved the numbers on its own, so pinning the strategy removes the
    strategy change from a cross-version comparison and nothing else.)

    The platform is pinned to ``"gpu"``, where ``"auto"`` resolves to
    ``dense_vmap`` for this plan, so an implementation that resolved
    unconditionally (ignoring what the caller passed) differs from a correct
    one on three of the four parametrisations rather than none.

    The other three members of ``WStrategy`` are deliberately not here:
    ``"auto"`` and the deprecated ``"scan"`` / ``"vmap"`` aliases are
    *supposed* to be rewritten on the way through, so "unchanged" is not
    the contract for them. ``test_explicit_auto_matches_the_default`` and
    ``tests/test_auto_strategy.py`` cover those.
    """
    _patch_platform(monkeypatch, "gpu")
    plan, image, vis = _offzenith_problem(real_dtype, complex_dtype)
    _, jit_name, _ = _OPERATORS[op]
    calls = _spy_jit(monkeypatch, jit_name)
    _call(op, plan, image, vis, w_strategy=w_strategy)
    assert len(calls) == 1
    assert calls[0]["w_strategy"] == w_strategy, (
        f"{op}(w_strategy={w_strategy!r}) was overridden to "
        f"{calls[0]['w_strategy']!r} by the default resolution"
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_explicit_auto_matches_the_default(
    monkeypatch: pytest.MonkeyPatch,
    op: str,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """Passing ``w_strategy="auto"`` explicitly and leaving it unset must be
    the same call: same resolved strategy *and* same resolved nthreads at the
    JIT boundary."""
    _patch_platform(monkeypatch, "gpu")
    plan, image, vis = _offzenith_problem(real_dtype, complex_dtype)
    _, jit_name, _ = _OPERATORS[op]
    calls = _spy_jit(monkeypatch, jit_name)
    _call(op, plan, image, vis, w_strategy="auto")
    _call(op, plan, image, vis)
    assert len(calls) == 2
    assert calls[0]["w_strategy"] == calls[1]["w_strategy"]
    assert calls[0]["nthreads"] == calls[1]["nthreads"]


# -- 3. what "auto" resolves to on the repository CPU fixtures --------------
#
# Literal expectations, not derived from the heuristic: the point is that a
# later retune (issue #34) has to edit this table, in a diff a reviewer can
# see, rather than silently move the shipped default. Keyed by precision
# because ``eps`` (1e-6 vs 1e-4) and ``dtype`` change ``n_w`` and
# ``w_kernel_width``, which are the heuristic's inputs.
#
# Measured on this branch with the platform pinned to CPU:
#   float64/1e-6: EDA2 zen n_w=14 W=7 pad=2.00, MWA_compact zen n_w=8 pad=1.14,
#     MWA_compact off30 n_w=17 pad=2.37, MWA_extended zen n_w=14 pad=2.00,
#     MWA_extended off30 n_w=251 pad=5.26, MeerKAT zen n_w=8 pad=1.14,
#     MeerKAT off30 n_w=19 pad=2.61.
#   float32/1e-4: same fixtures at W=5, n_w 12/6/15/12/249/6/17.
# The pattern in both legs: the forward never leaves ``dense_scan`` (the CPU
# heuristic has no forward windowed win to claim -- v0.1.1 Part 2 measured
# "~flat on forward"), and the adjoint switches to ``windowed_scan`` where the
# w-extent is large enough for the windowed slice to pay
# (n_w / w_kernel_width > 2).
#
# What that switch is worth on current code is *smaller* than the 1.05-1.53x
# AGENTS.md section 9 records for v0.1.1 Part 2, and that range should not be
# quoted here: its endpoints are MWA_compact off30 (1.53x) and MeerKAT off30
# (1.05x), and both now time flat (1.01x and 0.99x, inside a +-4% noise
# floor). Only MWA_extended off30 still shows a win, 1.14-1.24x. See the
# README's `w_strategy="auto"` section for the measurement. None of that
# changes what this table asserts -- which strategy is picked, not what it
# earns -- but a reader should not take the pick as evidence of the old
# range.
_CPU_FIXTURES: list[tuple[Telescope, float]] = [
    (EDA2, 0.0),
    (MWA_COMPACT, 0.0),
    (MWA_COMPACT, 30.0),
    (MWA_EXTENDED, 0.0),
    (MWA_EXTENDED, 30.0),
    (MEERKAT, 0.0),
    (MEERKAT, 30.0),
]

_EXPECTED_CPU_AUTO_F64: dict[tuple[str, float], tuple[str, str]] = {
    # (telescope name, zenith angle) -> (forward pick, adjoint pick)
    ("EDA2", 0.0): ("dense_scan", "dense_scan"),
    ("MWA_compact", 0.0): ("dense_scan", "dense_scan"),
    ("MWA_compact", 30.0): ("dense_scan", "windowed_scan"),
    ("MWA_extended", 0.0): ("dense_scan", "dense_scan"),
    ("MWA_extended", 30.0): ("dense_scan", "windowed_scan"),
    ("MeerKAT", 0.0): ("dense_scan", "dense_scan"),
    ("MeerKAT", 30.0): ("dense_scan", "windowed_scan"),
}
_EXPECTED_CPU_AUTO_F32: dict[tuple[str, float], tuple[str, str]] = {
    # Only EDA2 zenith and MWA_extended zenith differ from the float64 leg:
    # the kernel width is the *denominator* of the heuristic's ratio, so the
    # narrower float32 kernel (W=5 against 7) raises n_w / w_kernel_width past
    # the adjoint's cutoff of 2 at those two fixtures -- 14/7 = 2.0 in float64,
    # which the strict ``>`` rejects, against 12/5 = 2.4 in float32.
    ("EDA2", 0.0): ("dense_scan", "windowed_scan"),
    ("MWA_compact", 0.0): ("dense_scan", "dense_scan"),
    ("MWA_compact", 30.0): ("dense_scan", "windowed_scan"),
    ("MWA_extended", 0.0): ("dense_scan", "windowed_scan"),
    ("MWA_extended", 30.0): ("dense_scan", "windowed_scan"),
    ("MeerKAT", 0.0): ("dense_scan", "dense_scan"),
    ("MeerKAT", 30.0): ("dense_scan", "windowed_scan"),
}
_EXPECTED_CPU_AUTO = _EXPECTED_CPU_AUTO_F64 if X64 else _EXPECTED_CPU_AUTO_F32


@pytest.mark.parametrize(
    ("telescope", "zenith_deg"),
    _CPU_FIXTURES,
    ids=[f"{t.name}_{'zenith' if a == 0 else f'off{int(a)}'}" for t, a in _CPU_FIXTURES],
)
@pytest.mark.parametrize("is_adjoint", [False, True], ids=["forward", "adjoint"])
def test_cpu_default_resolves_to_expected_strategy(
    monkeypatch: pytest.MonkeyPatch,
    telescope: Telescope,
    zenith_deg: float,
    is_adjoint: bool,
    real_dtype: DTypeLike,
) -> None:
    """The resolved default for every repository CPU fixture, both directions.

    Host-side only (``make_plan`` plus the heuristic, no transform), which is
    why this is not gated behind ``--runslow`` even for the long-baseline
    fixtures: nothing here runs FINUFFT.
    """
    _patch_platform(monkeypatch, "cpu")
    uvw = synthetic_uvw(telescope, zenith_deg, seed=0)
    plan = make_plan(
        uvw=uvw,
        freq=np.array([telescope.freq_hz]),
        image_shape=(telescope.n_pix, telescope.n_pix),
        pixsize_l=telescope.pixsize,
        pixsize_m=telescope.pixsize,
        epsilon=EPSILON,
        dtype=real_dtype,
    )
    expected = _EXPECTED_CPU_AUTO[(telescope.name, zenith_deg)][1 if is_adjoint else 0]
    resolved = _canonicalise_w_strategy("auto", plan=plan, is_adjoint=is_adjoint)
    assert resolved == expected, (
        f"{telescope.name} zen={zenith_deg:g} "
        f"{'adjoint' if is_adjoint else 'forward'} (n_w={plan.n_w}, "
        f"W={plan.w_kernel_width}, padding_overhead="
        f"{plan.window_padding_overhead:.4f}): the default now resolves to "
        f"{resolved!r}, but this table says {expected!r}. If the heuristic was "
        f"retuned on purpose (issue #34), update the table with the measurement."
    )


# -- 4. the GPU case, behind the repository's GPU gate ----------------------
#
# Literal expectations, exactly as for the CPU table above: "a vmap-family
# strategy" is not enough. The GPU heuristic has two vmap answers and the
# choice between them is the whole content of three of its four gates, so an
# ``in ("dense_vmap", "windowed_vmap")`` assertion passes under a heuristic
# mutated to return the wrong one of the two -- which is how the earlier
# revision of this test failed to gate the branch it was written for.
#
# Measured on a real GH200 (issue #46's hardware) and reproduced here from
# the plan alone, which is host-side and therefore checkable off-GPU:
#
#   MWA_extended off30  n_w=251  fwd dense_vmap  adj dense_vmap
#   MeerKAT off30       n_w=19   fwd dense_vmap  adj dense_vmap
#   GH200_large off30   n_w=43   fwd dense_vmap  adj windowed_vmap
#
# GH200_large is the interesting row and the reason it is worth its size: it
# is the only fixture where the two directions *disagree*, so it is what pins
# the ``is_adjoint and n_rows >= _GPU_LARGE_N_ROWS -> windowed_vmap`` gate.
# Without it nothing in the suite reaches that branch on real hardware.
#
# The picks are the same on both precision legs (float64/eps 1e-6 and
# float32/eps 1e-4), so unlike the CPU table this one is not precision-keyed.
# The cell closest to a boundary is MeerKAT off30 in float32, whose padding
# overhead reads 2.992 against a 3.0 cutoff; crossing it would send the
# adjoint to ``dense_vmap``, which is what this table already expects, so
# even that crossing would not silently change an answer here.
_EXPECTED_GPU_AUTO: list[tuple[Telescope, str, str]] = [
    # telescope (all at 30 deg off-zenith) -> (forward pick, adjoint pick)
    (MWA_EXTENDED, "dense_vmap", "dense_vmap"),
    (MEERKAT, "dense_vmap", "dense_vmap"),
    (GH200_LARGE, "dense_vmap", "windowed_vmap"),
]


def _gpu_plan(tel: Telescope, real_dtype: DTypeLike):
    """Plan for ``tel`` at 30 deg off-zenith. Host-side numpy only."""
    return make_plan(
        uvw=synthetic_uvw(tel, 30.0, seed=0),
        freq=np.array([tel.freq_hz]),
        image_shape=(tel.n_pix, tel.n_pix),
        pixsize_l=tel.pixsize,
        pixsize_m=tel.pixsize,
        epsilon=EPSILON,
        dtype=real_dtype,
    )


@pytest.mark.runbench_gpu
@pytest.mark.parametrize(
    ("telescope", "expected_forward", "expected_adjoint"),
    _EXPECTED_GPU_AUTO,
    ids=[t.name for t, _, _ in _EXPECTED_GPU_AUTO],
)
def test_gpu_default_resolves_to_the_expected_strategy(
    telescope: Telescope,
    expected_forward: str,
    expected_adjoint: str,
    real_dtype: DTypeLike,
) -> None:
    """On a real GPU backend the default resolves to the *named* strategy.

    Issue #46's measurements: on a GH200, ``dense_scan`` (the old default)
    is 1.4-5.6x *slower* than ducc0 on 72 Grace cores in five of that
    table's six cells (the GH200_large off30 adjoint is the exception, at
    about 1.16x faster than ducc0, though still 2.0x off that table's
    ``dense_vmap`` column), while the strategy ``"auto"`` picks is
    1.4-6.3x faster in all six. Independently, the 160-cell GH200 sweep
    behind ``docs/benchmarks/v0.1.2-baseline-gpu.json`` has the scan family
    slower than the vmap family in *every* scan/vmap pair it contains,
    by 1.45x to 32.7x with a median of 6.1x. A scan-family pick is the
    headline regression this catches, but the exact-name assertion also
    catches picking the wrong member of the vmap family -- see the table
    comment above.

    Unlike the CPU tests, this does *not* patch the platform: the point is
    what a user on real GPU hardware gets. Gated by
    ``@pytest.mark.runbench_gpu``, the repository's only GPU gate
    (``--runbench-gpu`` plus ``jax.default_backend() == "gpu"``, enforced in
    ``tests/conftest.py``), so it skips cleanly on CPU.

    Cost: this builds a plan and runs the heuristic, and runs no transform.
    That matters for the ``GH200_large`` parametrisation -- 2048^2 with 50k
    rows -- whose plan is a few hundred MB of host arrays but whose gridding
    would be seconds of GPU time per call. The strategy resolution is
    host-side and needs neither, so it is not paid here. The end-to-end
    check below deliberately uses only the small fixture.
    """
    assert jax.default_backend() == "gpu", "the runbench_gpu gate should guarantee this"
    plan = _gpu_plan(telescope, real_dtype)
    for is_adjoint, expected in ((False, expected_forward), (True, expected_adjoint)):
        resolved = _canonicalise_w_strategy("auto", plan=plan, is_adjoint=is_adjoint)
        assert resolved == expected, (
            f"{telescope.name} off30 {'adjoint' if is_adjoint else 'forward'} "
            f"(n_w={plan.n_w}, W={plan.w_kernel_width}, n_rows={plan.n_rows}, "
            f"padding_overhead={plan.window_padding_overhead:.4f}): the GPU default "
            f"resolved to {resolved!r}, but this table says {expected!r}. A scan-family "
            "answer is slower than every vmap answer in all 160 pairs of the GH200 "
            "sweep (1.45-32.7x); the other vmap answer is a "
            "bounded loss but still not what was measured. If the heuristic was "
            "retuned on purpose (issue #34), update the table with the measurement."
        )


@pytest.mark.runbench_gpu
def test_gpu_default_matches_the_explicit_resolved_end_to_end(
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """...and the operators actually forward the resolved strategy on GPU.

    The resolution test above is host-side; this runs the real transform so
    there is one end-to-end statement about the GPU path rather than only a
    statement about the heuristic. Uses ``MWA_extended`` off30 alone -- the
    headline row of the issue's table (n_w=251) at 256^2 and 600 rows, so
    four transforms here are cheap; ``GH200_large`` is deliberately excluded.
    """
    assert jax.default_backend() == "gpu", "the runbench_gpu gate should guarantee this"
    plan = _gpu_plan(MWA_EXTENDED, real_dtype)
    rng = np.random.default_rng(0)
    image = jnp.asarray(
        rng.standard_normal((MWA_EXTENDED.n_pix, MWA_EXTENDED.n_pix)), dtype=real_dtype
    )
    vis = jnp.asarray(
        rng.standard_normal((MWA_EXTENDED.n_rows, 1))
        + 1j * rng.standard_normal((MWA_EXTENDED.n_rows, 1)),
        dtype=complex_dtype,
    )
    fwd_default = np.asarray(dirty2vis(plan, image))
    fwd_explicit = np.asarray(
        dirty2vis(
            plan, image, w_strategy=_canonicalise_w_strategy("auto", plan=plan, is_adjoint=False)
        )
    )
    np.testing.assert_allclose(fwd_default, fwd_explicit, rtol=tol(1e-11, 1e-5), atol=0)
    adj_default = np.asarray(vis2dirty(plan, vis))
    adj_explicit = np.asarray(
        vis2dirty(
            plan, vis, w_strategy=_canonicalise_w_strategy("auto", plan=plan, is_adjoint=True)
        )
    )
    np.testing.assert_allclose(adj_default, adj_explicit, rtol=tol(1e-11, 1e-5), atol=0)


# -- 5. the nthreads interaction (issue #24 x issue #46) --------------------


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_resolve_nthreads_is_never_handed_the_literal_auto(
    monkeypatch: pytest.MonkeyPatch,
    op: str,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """``_resolve_nthreads`` derives the thread default from the strategy
    *family*, so it must see the resolved name, never ``"auto"``.

    With both defaults in play the operators canonicalise first and pass the
    result down; this spies on the boundary between them and asserts the
    exact string handed over. An implementation that forwarded the raw
    ``"auto"`` would either blow up (no ``plan``/``is_adjoint`` context) or
    -- if it also forwarded the context -- resolve a second time, which this
    names as the defect rather than letting it pass because the answer
    happened to come out the same.
    """
    _patch_platform(monkeypatch, "gpu")
    plan, image, vis = _offzenith_problem(real_dtype, complex_dtype)
    _, jit_name, is_adjoint = _OPERATORS[op]
    expected = _auto_w_strategy(plan, is_adjoint=is_adjoint)

    seen: list[tuple[Any, ...]] = []
    real_resolve = wgridder._resolve_nthreads

    def spy(nthreads, w_strategy, n_rows, **kwargs):  # type: ignore[no-untyped-def]
        seen.append((nthreads, w_strategy, n_rows, kwargs))
        return real_resolve(nthreads, w_strategy, n_rows, **kwargs)

    monkeypatch.setattr(wgridder, "_resolve_nthreads", spy)
    _spy_jit(monkeypatch, jit_name)
    _call(op, plan, image, vis)

    assert len(seen) == 1, f"{op} called _resolve_nthreads {len(seen)} times, expected once"
    _, w_strategy_arg, _, _ = seen[0]
    assert w_strategy_arg != "auto", (
        f"{op} handed _resolve_nthreads the unresolved literal 'auto'; the thread "
        "default must come from the resolved strategy"
    )
    assert w_strategy_arg == expected, (
        f"{op} handed _resolve_nthreads {w_strategy_arg!r}, but 'auto' resolves to "
        f"{expected!r} for this plan/direction"
    )


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_defaulted_call_canonicalises_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    op: str,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """Issue #46's "one canonicalisation" checkbox, gated rather than argued.

    With ``w_strategy`` and ``nthreads`` both defaulting, the resolution is
    reachable from two places: the operator's own wrapper, and
    ``_resolve_nthreads``, which canonicalises whatever name it is handed.
    Before issue #46 closed this, the pair ran it twice per defaulted call --
    the second pass on an already-canonical name.

    Two passes are *harmless today* only because canonicalisation is a fixed
    point on the four canonical names, which
    ``test_canonicalisation_is_idempotent`` pins separately. That makes the
    second pass a correctness dependency on another function's internals: a
    future ``_canonicalise_w_strategy`` with a plan-dependent branch, or one
    that re-consulted the heuristic, would resolve twice and could hand
    ``_resolve_nthreads`` a different strategy from the one the wrapper had
    already committed to at the JIT boundary. So the count itself is the
    contract, not just the answer.

    ``_resolve_nthreads`` reaches ``_canonicalise_w_strategy`` through the
    module global, so patching the module attribute observes both call sites
    from one spy.

    Scope: this goes through a real operator call, so it exercises whichever
    canonical name the fixture resolves to and no others.
    ``test_resolve_nthreads_never_canonicalises_a_canonical_name`` below is
    the per-name statement of the same property, and is what catches a
    short-circuit that covers only some of the four.
    """
    _patch_platform(monkeypatch, "gpu")
    plan, image, vis = _offzenith_problem(real_dtype, complex_dtype)
    _, jit_name, is_adjoint = _OPERATORS[op]
    expected = _auto_w_strategy(plan, is_adjoint=is_adjoint)

    seen: list[tuple[str, Any]] = []
    real_canonicalise = wgridder._canonicalise_w_strategy

    def spy(name, **kwargs):  # type: ignore[no-untyped-def]
        seen.append((name, kwargs.get("is_adjoint")))
        return real_canonicalise(name, **kwargs)

    monkeypatch.setattr(wgridder, "_canonicalise_w_strategy", spy)
    _spy_jit(monkeypatch, jit_name)
    _call(op, plan, image, vis)  # both defaults

    assert len(seen) == 1, (
        f"{op} at both defaults canonicalised {len(seen)} times ({seen}), expected once. "
        "The wrapper resolves 'auto'; _resolve_nthreads must use that resolved name "
        "as-is rather than canonicalising it again."
    )
    name, seen_is_adjoint = seen[0]
    assert name == "auto", f"the single canonicalisation should be of 'auto', got {name!r}"
    assert seen_is_adjoint is is_adjoint, (
        f"{op} resolved 'auto' with is_adjoint={seen_is_adjoint!r}, expected {is_adjoint!r}"
    )
    # ...and the one resolution still produces the right answer, so this cannot
    # pass by skipping the resolution altogether.
    assert real_canonicalise("auto", plan=plan, is_adjoint=is_adjoint) == expected


@pytest.mark.parametrize("w_strategy", CANONICAL)
@pytest.mark.parametrize(
    "n_rows",
    [wgridder._NTHREADS_SMALL_N_ROWS - 1, wgridder._NTHREADS_SMALL_N_ROWS + 1],
    ids=["below_row_cutoff", "above_row_cutoff"],
)
def test_resolve_nthreads_never_canonicalises_a_canonical_name(
    monkeypatch: pytest.MonkeyPatch,
    w_strategy: WStrategy,
    n_rows: int,
) -> None:
    """The short-circuit itself, stated over *every* canonical name.

    This is the general form of the requirement
    ``test_defaulted_call_canonicalises_exactly_once`` checks end-to-end.
    That test goes through a real operator call, so it can only ever
    exercise whichever strategy its fixture happens to resolve to -- one of
    the four. A short-circuit written to cover only that one (or only the
    vmap family, or only the name the fixture produces) would satisfy it
    while leaving the other three defaulted paths canonicalising twice.

    So assert the property directly and per name: hand ``_resolve_nthreads``
    an already-canonical strategy and require that
    ``_canonicalise_w_strategy`` is not called *at all*. Nothing here depends
    on the heuristic, so a retune under issue #34 cannot make it stale, and
    no plan or JAX work is needed.

    Both sides of ``_NTHREADS_SMALL_N_ROWS`` are covered: the short-circuit
    currently sits above that branch, and this keeps it gated if the two are
    ever reordered.

    ``_resolve_nthreads`` reaches ``_canonicalise_w_strategy`` through the
    module global, so patching the module attribute observes it however
    ``_resolve_nthreads`` itself was imported.
    """
    calls: list[str] = []
    real_canonicalise = wgridder._canonicalise_w_strategy

    def spy(name, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(name)
        return real_canonicalise(name, **kwargs)

    monkeypatch.setattr(wgridder, "_canonicalise_w_strategy", spy)
    got = _resolve_nthreads(None, w_strategy, n_rows)

    assert calls == [], (
        f"_resolve_nthreads(None, {w_strategy!r}, {n_rows}) canonicalised {calls}; "
        "an already-canonical name must be used as-is. A short-circuit that covers "
        "only some of the four canonical names leaves the rest resolving twice on "
        "the defaulted path (issue #46)."
    )
    # ...and the answer is still the right one, so this cannot pass by
    # short-circuiting past the thread rule as well as past canonicalisation.
    expected = 1 if (n_rows < wgridder._NTHREADS_SMALL_N_ROWS or "scan" in w_strategy) else 0
    assert got == expected, (
        f"_resolve_nthreads(None, {w_strategy!r}, {n_rows}) = {got}, expected {expected}"
    )
    # The canonical path must also not have acquired a plan/is_adjoint
    # dependency: the call above passed neither.


@pytest.mark.parametrize("w_strategy", CANONICAL)
def test_resolve_nthreads_needs_no_plan_context_for_canonical_names(
    w_strategy: WStrategy,
) -> None:
    """The short-circuit above must not have made the canonical path depend on
    ``plan`` / ``is_adjoint``: a canonical name resolves with neither."""
    assert _resolve_nthreads(None, w_strategy, 500_000) in (0, 1)


@pytest.mark.parametrize(("alias", "canonical"), [("scan", "dense_scan"), ("vmap", "dense_vmap")])
def test_resolve_nthreads_still_resolves_deprecated_aliases(alias: str, canonical: str) -> None:
    """The raw-name path is still live for direct callers.

    Skipping canonicalisation for already-canonical names must not skip it
    for the v0.1 aliases: they still resolve (and still warn) here, and reach
    the same thread default their ``dense_*`` counterparts do. Run above
    ``_NTHREADS_SMALL_N_ROWS`` so the two families give different answers and
    the assertion is not satisfied by the small-``n_rows`` override.
    """
    n_rows = wgridder._NTHREADS_SMALL_N_ROWS + 1
    with pytest.warns(DeprecationWarning):
        got = _resolve_nthreads(None, alias, n_rows)
    assert got == _resolve_nthreads(None, canonical, n_rows)


def test_resolve_nthreads_still_rejects_auto_without_context() -> None:
    """The other half of the raw-name path: ``"auto"`` with no ``plan`` /
    ``is_adjoint`` must still raise rather than fall through the
    already-canonical short-circuit."""
    with pytest.raises(ValueError, match="must be resolved with plan"):
        _resolve_nthreads(None, "auto", 500_000)


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_both_defaults_give_the_same_nthreads_as_the_explicit_resolved_strategy(
    monkeypatch: pytest.MonkeyPatch,
    op: str,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """The combination the issue asks about: ``w_strategy`` *and* ``nthreads``
    both left unset.

    Run above ``_NTHREADS_SMALL_N_ROWS`` with the platform pinned to ``"gpu"``
    -- the only regime where the two strategy families give different thread
    defaults (vmap -> 0, scan -> 1), so "derived from the resolved strategy"
    is observable at all. The literal ``0`` is asserted, not just equality
    between the two calls: with the pre-#46 ``dense_scan`` default both calls
    would agree on ``1`` and an equality-only test would pass.
    """
    _patch_platform(monkeypatch, "gpu")
    plan, image, vis = _plan_above_nthreads_cutoff(real_dtype, complex_dtype)
    _, jit_name, is_adjoint = _OPERATORS[op]
    resolved = _auto_w_strategy(plan, is_adjoint=is_adjoint)
    assert resolved in ("dense_vmap", "windowed_vmap"), (
        f"sanity: expected a vmap-family pick on the pinned GPU platform, got {resolved!r}"
    )

    calls = _spy_jit(monkeypatch, jit_name)
    _call(op, plan, image, vis)  # both defaults
    _call(op, plan, image, vis, w_strategy=resolved)  # explicit strategy, default nthreads
    assert len(calls) == 2
    assert calls[0]["nthreads"] == calls[1]["nthreads"] == 0, (
        f"{op}: both-defaults gave nthreads={calls[0]['nthreads']}, explicit "
        f"{resolved!r} gave {calls[1]['nthreads']}; the vmap family's default is 0"
    )
    assert calls[0]["w_strategy"] == calls[1]["w_strategy"] == resolved


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("is_adjoint", [False, True], ids=["forward", "adjoint"])
def test_canonicalisation_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    is_adjoint: bool,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """Resolving twice cannot change the answer.

    Since issue #46 a defaulted operator call resolves exactly once -- the
    wrapper canonicalises and ``_resolve_nthreads`` short-circuits on the
    already-canonical name, which
    ``test_defaulted_call_canonicalises_exactly_once`` gates. Idempotence is
    therefore no longer load-bearing on that path, and this test is not a
    substitute for that one. It stays because the property is still relied
    on wherever a canonical name is re-offered to the resolver -- direct
    callers, and any future refactor that moves the canonicalisation -- and
    because it is what makes the short-circuit safe to add rather than a
    behaviour change.

    Note this also catches a heuristic that returns a *deprecated alias*
    (``"scan"`` / ``"vmap"``) instead of a canonical name: the second
    canonicalisation would emit ``DeprecationWarning``, which
    ``filterwarnings = ["error"]`` turns into a failure here.
    """
    _patch_platform(monkeypatch, platform)
    plan, _, _ = _offzenith_problem(real_dtype, complex_dtype)
    once = _canonicalise_w_strategy("auto", plan=plan, is_adjoint=is_adjoint)
    assert once in CANONICAL, f"'auto' resolved to the non-canonical name {once!r}"
    twice = _canonicalise_w_strategy(once, plan=plan, is_adjoint=is_adjoint)
    thrice = _canonicalise_w_strategy(twice, plan=plan, is_adjoint=is_adjoint)
    assert twice == once, f"canonicalising {once!r} again gave {twice!r}"
    assert thrice == once, f"a third canonicalisation of {once!r} gave {thrice!r}"


@pytest.mark.parametrize("name", CANONICAL)
def test_canonical_names_are_fixed_points(name: WStrategy) -> None:
    """Same property stated on the canonical names directly, with no plan
    context -- the form ``_resolve_nthreads`` relies on when the operators
    hand it an already-resolved strategy."""
    assert _canonicalise_w_strategy(name) == name


@pytest.mark.parametrize("is_adjoint", [False, True], ids=["forward", "adjoint"])
def test_double_resolution_cannot_change_the_thread_default(
    monkeypatch: pytest.MonkeyPatch,
    is_adjoint: bool,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """``_resolve_nthreads`` given the raw ``"auto"`` (resolving it itself)
    and given the already-resolved name must agree.

    This is the invariant that makes the operators' "canonicalise, then pass
    the result down" ordering safe, and equally makes it safe for a future
    refactor to move the canonicalisation -- but not to skip resolving at
    all. Run above the row cutoff so the two strategy families are
    distinguishable.
    """
    _patch_platform(monkeypatch, "gpu")
    plan, _, _ = _plan_above_nthreads_cutoff(real_dtype, complex_dtype)
    n_rows = plan.n_rows
    resolved = _auto_w_strategy(plan, is_adjoint=is_adjoint)
    from_auto = _resolve_nthreads(None, "auto", n_rows, plan=plan, is_adjoint=is_adjoint)
    from_resolved = _resolve_nthreads(None, resolved, n_rows)
    assert from_auto == from_resolved == 0, (
        f"nthreads from 'auto' = {from_auto}, from the resolved {resolved!r} = "
        f"{from_resolved}; both must be the vmap family's 0 here"
    )


# -- 6. JIT cache sharing ---------------------------------------------------


@pytest.mark.parametrize("op", ["dirty2vis", "vis2dirty"])
def test_default_shares_the_compiled_executable_with_the_explicit_resolved(
    monkeypatch: pytest.MonkeyPatch,
    op: str,
    real_dtype: DTypeLike,
    complex_dtype: DTypeLike,
) -> None:
    """``"auto"`` resolves *before* the JIT boundary, so the default call and
    the explicit resolved call must hit the same compiled executable -- not
    merely produce equal numbers.

    Measured the way ``tests/test_auto_strategy.py`` already measures it: the
    jitted inner function's ``_cache_size()`` after a ``_clear_cache()``. One
    entry after both calls means one lowering/compilation; two would mean the
    default reached the boundary as a different static argument (the literal
    ``"auto"``, or an unresolved-but-different name), i.e. every user paying a
    second compile and losing the shared cache.

    The platform is pinned to ``"gpu"`` so the resolved name (``dense_vmap``)
    differs from the pre-#46 default: without that, on CPU the forward
    resolves to ``dense_scan`` and a stale default would share the cache for
    the wrong reason. Only the heuristic reads the pinned platform; the
    transforms below really execute on whatever backend pytest has.
    """
    _patch_platform(monkeypatch, "gpu")
    plan, image, vis = _offzenith_problem(real_dtype, complex_dtype)
    jit_fn = _dirty2vis_jit if op == "dirty2vis" else _vis2dirty_jit
    is_adjoint = _OPERATORS[op][2]
    resolved = _auto_w_strategy(plan, is_adjoint=is_adjoint)

    jit_fn._clear_cache()
    out_explicit = _call(op, plan, image, vis, w_strategy=resolved)
    jax.block_until_ready(out_explicit)
    size_after_explicit = jit_fn._cache_size()
    assert size_after_explicit == 1, (
        f"sanity: expected exactly one cache entry after the explicit call, "
        f"got {size_after_explicit}"
    )

    out_default = _call(op, plan, image, vis)
    jax.block_until_ready(out_default)
    size_after_default = jit_fn._cache_size()

    assert size_after_default == size_after_explicit, (
        f"{op} at its default triggered a recompile: cache "
        f"{size_after_explicit} -> {size_after_default}. The default must resolve "
        f"to {resolved!r} before the JIT boundary so it shares the executable."
    )
    # Same executable, same inputs: the outputs are bit-identical, so this is
    # an exact comparison rather than a tolerance.
    np.testing.assert_array_equal(np.asarray(out_default), np.asarray(out_explicit))
