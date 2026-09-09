"""Issue #49: recompute every prose citation of ``docs/benchmarks/*.json``.

The repository quotes aggregate figures from its own committed benchmark JSON
in ``src/`` comments, ``README.md``, ``AGENTS.md``, ``docs/v0.1.2-plan.md`` and
test docstrings, and those citations are written by hand. Three of them were
wrong when #46 was reviewed, each falsifiable by reading JSON that had been in
the tree the whole time. This module closes that loop: every cited aggregate is
recomputed here from the JSON, and each case states the claim *in the form the
prose makes it* -- a range, a bound, a count, an "in every cell" -- so a failure
names the sentence that went stale rather than only a number.

**Where the figures live.** :data:`CITATIONS` is the single place the numbers
are written down. Each entry carries the sentence being checked and the files
that make it, so a re-measurement updates the citation and the assertion in one
edit. Prose that quotes one of these aggregates should point here.

**What is deliberately not covered.** Three families of quoted figure cannot be
recomputed from anything in the tree, and are recorded rather than proxied,
followed by one thing this module asserts on data it cannot check the scale of:

* Issue #46's GH200-versus-ducc0 table ("1.4-5.6x slower than ducc0 on five of
  the table's six cells", "1.4-6.3x faster in all six", the "about 1.16x
  faster" exception) -- quoted in ``README.md``'s strategy section,
  ``AGENTS.md`` sec 5, the ``dirty2vis`` / ``vis2dirty`` docstrings and
  ``tests/test_default_w_strategy.py``. Those timings exist only as the
  markdown table in ``README.md``; no JSON was committed for them. Recomputing
  them from that table (by hand, 2026-09) reproduces every one of the four
  figures, but a test would be asserting the README against itself.

  The v0.2.0 comparison that replaces it in the README's performance section
  does not have this problem: issue #33 committed the sweep behind it as
  ``docs/benchmarks/v0.2.0-vs-ducc0-gh200.json``, and the cases from
  :func:`test_the_realistic_forward_speedup_spans_the_cited_range` onwards
  recompute every figure the new prose states. This bullet stays because the
  *old* table's numbers are still quoted in the four sites above and still
  cannot be checked against anything.
* The post-#43 ``window_padding_overhead`` figures in ``wgridder.py``'s
  ``_GPU_PADDING_CUTOFF`` comment. ``docs/benchmarks/README.md`` records that
  every committed JSON carries the *pre*-#43 scale and that the conversion
  needs plan-time locals that were never stored, so the new-scale numbers
  cannot come from the tree at all.
* ``README.md``'s "Indicative numbers" tables and every Apple M-series timing
  in ``AGENTS.md`` sec 5 / sec 9, including the #24 ``nthreads`` ranges. The
  committed CPU JSON is aarch64 Grace/GH200 from 2026-05-18; those tables are a
  different machine and a different date, with no JSON behind them.

And one figure that is checked here but whose *comparability* rests on a
judgement this module cannot make for itself:

* The memory ratio in :func:`test_jax_needs_more_memory_than_ducc0_in_every_cell_and_by_how_much`
  divides jax-nufft's peak device HBM by ducc0's peak process RSS less the
  interpreter's footprint. Both sides cover input, scratch and output, which is
  what makes the division meaningful, but they are different instruments on
  different hardware, and RSS is quantised in steps of about 36 MB. For the
  four cells whose ducc0 side reads 108 MB or less that quantisation is a large
  fraction of the value, so those ratios carry an uncertainty this module
  asserts nothing about. The arithmetic is pinned; the interpretation is stated
  in the JSON's ``comparability`` field and in the README beside the number.

And one hole that is *not* a missing case but a mismatch of scales, recorded
here because the passing assertion hides it:

* :func:`test_the_heuristic_picks_the_measured_winner_in_every_cell_and_slice`
  feeds each row's ``window_padding_overhead`` into today's
  ``_auto_w_strategy_gpu``, which compares it against
  ``_GPU_PADDING_CUTOFF = 3.0``. But ``docs/benchmarks/README.md`` records that
  every committed JSON carries the **pre-#43** padding scale, which "reads 0 -
  17% lower on the same plan" and is not convertible after the fact. So that
  case is checking old-scale data against a new-scale gate, and no assertion
  here can see the difference.

  Bounding the exposure at ``new = old / 0.83`` (measured 2026-09): two of the
  20 cells cross the 3.0 cutoff, ``dirty2vis`` and ``vis2dirty`` on
  ``GH200_large_off30``, both 2.588 -> up to 3.118. Eighteen keep their pick
  either way and so does the ``dirty2vis`` one, whose ``n_w`` of 77 already
  sends it to ``dense_vmap`` through the forward-ratio gate. **One flips:**
  ``vis2dirty/GH200_large_off30`` takes the padding gate instead of the
  large-row adjoint gate and picks ``dense_vmap`` where the measured winner is
  ``windowed_vmap`` -- which would falsify both the "20/20 cells" claim and
  "``windowed_vmap`` wins the adjoint at both pointings", while this module
  stayed green, because it reads 2.588. Inherited from #43 rather than
  introduced by #49, and fixable only by re-running the sweep on the current
  scale.

Pure-Python and fast: it reads JSON and replays the host-side heuristic, and
runs no kernels.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jax_nufft.wgridder import _auto_w_strategy_gpu

_BENCH_DIR = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
_GPU_BASELINE = _BENCH_DIR / "v0.1.2-baseline-gpu.json"
_CPU_BASELINE = _BENCH_DIR / "v0.1.2-baseline-gh200.json"
_CPU_PART1 = _BENCH_DIR / "v0.1.2-part1.json"
_REALISTIC = _BENCH_DIR / "v0.2.0-vs-ducc0-gh200.json"
_MEMORY = _BENCH_DIR / "v0.2.0-memory-gh200.json"

_W_STRATEGIES = ("dense_scan", "dense_vmap", "windowed_scan", "windowed_vmap")
_CPU_FIXTURES = (
    "EDA2_zenith",
    "EDA2_off30",
    "MWA_compact_zenith",
    "MWA_compact_off30",
    "MWA_extended_zenith",
    "MWA_extended_off30",
    "MeerKAT_zenith",
    "MeerKAT_off30",
)
_OPS = ("dirty2vis", "vis2dirty")


# --------------------------------------------------------------------------
# The cited figures, in one place.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """One aggregate quoted in prose, with the sentence and where it is made.

    ``figures`` holds the numbers exactly as the prose states them, so editing
    a citation means editing this table and nothing else.
    """

    claim: str
    sites: tuple[str, ...]
    source: Path
    figures: dict[str, Any] = field(default_factory=dict)


CITATIONS: dict[str, Citation] = {
    "pair_population": Citation(
        claim=(
            "'that sweep's 160 scan/vmap pairs' -- the population is every "
            "(scan-family w_strategy, vmap-family w_strategy) pair drawn within "
            "an (op, fixture, channel_strategy) group: 20 cells x 2 channel "
            "strategies x 2 x 2 = 160."
        ),
        sites=(
            "src/jax_nufft/wgridder.py (_GPU_LARGE_N_ROWS comment; _auto_w_strategy_gpu docstring)",
            "README.md (Strategies -> GPU bullet)",
            "AGENTS.md sec 9 (Part 6 correction)",
            "tests/test_auto_strategy_acceptance.py (test_gpu_auto_never_picks_a_scan_variant)",
            "tests/test_default_w_strategy.py (test_gpu_default_resolves_to_the_expected_strategy)",
        ),
        source=_GPU_BASELINE,
        figures={"n_pairs": 160, "n_cells": 20},
    ),
    "scan_always_slower": Citation(
        claim=(
            "'the scan family is slower in every one' of the 160 pairs -- the "
            "universality the 'never auto-pick a scan strategy on GPU' rule "
            "rests on, as opposed to the size of the gap."
        ),
        sites=(
            "src/jax_nufft/wgridder.py (_GPU_LARGE_N_ROWS comment; _auto_w_strategy_gpu docstring)",
            "README.md (Strategies -> GPU bullet)",
            "AGENTS.md sec 9 (Part 6 correction)",
            "tests/test_auto_strategy_acceptance.py",
            "tests/test_default_w_strategy.py",
        ),
        source=_GPU_BASELINE,
        figures={"slower_in_all": True},
    ),
    "pair_spread": Citation(
        claim=(
            "'by 1.45x to 32.7x (median 6.1x)', stated to three decimals in "
            "AGENTS.md sec 9 as 'span 1.448x to 32.660x with a median of "
            "6.096x'."
        ),
        sites=(
            "src/jax_nufft/wgridder.py (_GPU_LARGE_N_ROWS comment; _auto_w_strategy_gpu docstring)",
            "README.md (Strategies -> GPU bullet)",
            "AGENTS.md sec 9 (Part 6 correction)",
            "tests/test_auto_strategy_acceptance.py",
            "tests/test_default_w_strategy.py",
        ),
        source=_GPU_BASELINE,
        figures={
            "min": 1.448,
            "max": 32.660,
            "median": 6.096,
            # The same three rounded the way src/, README.md and the test
            # docstrings quote them. A second editorial step with its own
            # chance to drift, so it is written down here rather than inline
            # in the assertion: this table is meant to be the single place
            # the numbers live.
            "rounded": (1.45, 32.7, 6.1),
        },
    ),
    "pairs_below_5x": Citation(
        claim=(
            "'72 of them are below 5x' -- the count that killed the '5-30x' "
            "claim the sweep was originally written up with."
        ),
        sites=(
            "AGENTS.md sec 9 (Part 6 correction)",
            "tests/test_auto_strategy_acceptance.py",
        ),
        source=_GPU_BASELINE,
        figures={"below_5x": 72},
    ),
    "no_subset_is_5_to_30x": Citation(
        claim=(
            "'no subset of the sweep (off-zenith only, excluding GH200_large, "
            "best-of-family per cell), taken on its own, yields a 5-30x "
            "range'; and the parenthetical that makes 'taken on its own' "
            "load-bearing -- 'conjoining all three restrictions does give a "
            "population lying inside 5-30x, 10.376x to 29.393x over eight "
            "cells'."
        ),
        sites=("AGENTS.md sec 9 (Part 6 correction)",),
        source=_GPU_BASELINE,
        figures={
            "low": 5.0,
            "high": 30.0,
            "named_subsets": (
                "off-zenith only",
                "excluding GH200_large",
                "best-of-family per cell",
            ),
            # (min, max, n) of the conjunction, to the three decimals
            # AGENTS.md sec 9 states it in. This is itself a hand-written
            # aggregate of the JSON in prose, so it is pinned like the rest.
            "conjunction": (10.376, 29.393, 8),
        },
    ),
    "dense_vmap_cell_wins": Citation(
        claim="'dense_vmap winning 17/20 cells'.",
        sites=("AGENTS.md sec 9 (Part 6)",),
        source=_GPU_BASELINE,
        figures={"wins": 17, "cells": 20},
    ),
    "windowed_vmap_cell_wins": Citation(
        claim=(
            "'windowed_vmap winning only the 50k-row GH200_large fixture' "
            "(AGENTS.md), stated cell by cell in wgridder.py as 'the adjoint "
            "(both pointings) and the forward at zenith'."
        ),
        sites=(
            "AGENTS.md sec 9 (Part 6)",
            "src/jax_nufft/wgridder.py (_auto_w_strategy_gpu docstring)",
        ),
        source=_GPU_BASELINE,
        figures={
            "cells": (
                ("dirty2vis", "GH200_large_zenith"),
                ("vis2dirty", "GH200_large_off30"),
                ("vis2dirty", "GH200_large_zenith"),
            )
        },
    ),
    "heuristic_picks_the_winner": Citation(
        claim=(
            "'the four gates below pick the cell's winner in 20/20 cases ... "
            "on both the scan and vmap channel-strategy slices' (wgridder.py); "
            "'the auto heuristic picks the measured winner in 20/20 (op, "
            "fixture) cells' (AGENTS.md)."
        ),
        sites=(
            "src/jax_nufft/wgridder.py (_auto_w_strategy_gpu docstring)",
            "AGENTS.md sec 9 (Part 6, Measured wins)",
        ),
        source=_GPU_BASELINE,
        figures={"cells": 20, "channel_slices": 40},
    ),
    "runner_up_gap": Citation(
        claim=(
            "'the runner-up sits 1.07x to 7.31x behind the winner (median "
            "1.86x) and only one of the 20 cells is within 15% ... taken as 40 "
            "separate per-channel slices the range is 1.07x to 7.36x and two "
            "are within 15%'. This replaced the wrong claim that the runner-up "
            "is always inside the 15% acceptance bar. Also issue #49's own "
            "'worst 7.22x slower', which is the same gap restricted to the "
            "channel_strategy == 'scan' half of those 40 slices."
        ),
        sites=(
            "src/jax_nufft/wgridder.py (_auto_w_strategy_gpu docstring)",
            "tests/test_default_w_strategy.py (assertion message)",
            "issue #49 (body: 'worst 7.22x slower')",
        ),
        source=_GPU_BASELINE,
        figures={
            "min": 1.07,
            "max": 7.31,
            "median": 1.86,
            "within_15pct": 1,
            "slice_min": 1.07,
            "slice_max": 7.36,
            "slice_within_15pct": 2,
            # The two channel halves taken separately, to four decimals,
            # because the difference between them is the whole of what the
            # issue's 7.22x and wgridder.py's 7.36x disagree about. Measured
            # 2026-09 under #49.
            "scan_channel": (1.0847, 7.2205, 1.8620, 20),
            "vmap_channel": (1.0738, 7.3605, 1.8557, 20),
            "scan_channel_worst_cell": ("dirty2vis", "EDA2_off30"),
            "scan_channel_worst_ms": (152.649, 21.141),
        },
    ),
    "acceptance_bar": Citation(
        claim=(
            "'asserts the picked strategy is within 15% of the best measured "
            "strategy for every (operator, telescope) cell' -- and wgridder.py's "
            "reading of why that is not reassuring: 'a bound the heuristic "
            "meets trivially here by picking the best one every time'."
        ),
        sites=(
            "README.md (Strategies -> the GPU gates are validated ...)",
            "AGENTS.md sec 9 (Part 6)",
            "src/jax_nufft/wgridder.py (_auto_w_strategy_gpu docstring)",
            "tests/test_auto_strategy_acceptance.py",
        ),
        source=_GPU_BASELINE,
        figures={"bar": 1.15, "actual_worst": 1.0},
    ),
    "plan_dense_vmap_sweeps_cpu_json": Citation(
        claim=(
            "'dense_vmap wins on every fixture x operator on GPU' (baseline "
            "table) and 'dense_vmap is the winner on every fixture x op' "
            "(Part 1 table)."
        ),
        sites=("docs/v0.1.2-plan.md (Baseline summary; Part 1 GH200 snapshot)",),
        source=_CPU_BASELINE,
        # The population both plan-doc tables are quantified over: 8 fixtures
        # x 2 operators x 4 strategies, one row per cell in each table. Every
        # CPU case derives its cells from the JSON and checks them against
        # this, rather than iterating it (see :func:`_cpu_cells`).
        figures={
            "cells": 16,
            "fixtures": _CPU_FIXTURES,
            "ops": _OPS,
            "w_strategies": _W_STRATEGIES,
        },
    ),
    "plan_long_scan_worst_case": Citation(
        claim=(
            "'Off-zenith MWA_extended pushes n_w to 515 and exposes the "
            "dense_scan / windowed_scan worst case (>=0.7-1.1 s ...)'."
        ),
        sites=("docs/v0.1.2-plan.md (Reading the baseline at a glance)",),
        source=_CPU_BASELINE,
        figures={"low_s": 0.7, "high_s": 1.1},
    ),
    "plan_gpu_loses_to_ducc": Citation(
        claim=(
            "'On the small or low-w fixtures (EDA2 zenith/off30, MWA_compact "
            "off30 forward) the GPU loses to ducc'."
        ),
        sites=("docs/v0.1.2-plan.md (Reading the table)",),
        source=_CPU_PART1,
        figures={
            "cells": (
                ("dirty2vis", "EDA2_zenith"),
                ("vis2dirty", "EDA2_zenith"),
                ("dirty2vis", "EDA2_off30"),
                ("vis2dirty", "EDA2_off30"),
                ("dirty2vis", "MWA_compact_off30"),
            )
        },
    ),
    "plan_large_fixture_speedup": Citation(
        claim=(
            "'On the larger / more w-spread fixtures (MeerKAT, MWA_extended) "
            "jax-nufft on H100 beats single-thread ducc by 1.29-7.45x ... six "
            "of those eight cells sit in 3-7x'. Corrected under issue #49: the "
            "sentence read '3-7x for both operators', which two of the eight "
            "cells fall outside."
        ),
        sites=("docs/v0.1.2-plan.md (Reading the table)",),
        source=_CPU_PART1,
        figures={
            "min": 1.29,
            "max": 7.45,
            "in_3_to_7": 6,
            "cells": 8,
            "below": ("dirty2vis", "MWA_extended_off30"),
            "above": ("vis2dirty", "MWA_extended_zenith"),
        },
    ),
    "plan_dense_scan_regression": Citation(
        claim=(
            "The Part 1 per-Part diff row's 'dense_scan worst regression'. "
            "Corrected under issue #49 to '+42.3% on the MeerKAT zenith adjoint "
            "(a noisy cell Part 1 cannot reach), +6.0% worst of the other "
            "fifteen'; it read '+5.2% (within noise floor of untouched paths)', "
            "which is only the third largest of the sixteen."
        ),
        sites=("docs/v0.1.2-plan.md (Per-Part diff vs baseline)",),
        source=_CPU_PART1,
        figures={
            "worst_pct": 42.3,
            "worst_cell": ("vis2dirty", "MeerKAT_zenith"),
            "runner_up_pct": 6.0,
            "runner_up_cell": ("vis2dirty", "EDA2_zenith"),
            "cells": 16,
        },
    ),
    "plan_windowed_scan_regression": Citation(
        claim=(
            "'windowed_scan off30 dirty2vis | +2.7% to +10.7% (MWA_ext off30 +2.7% with n_w=515)'."
        ),
        sites=("docs/v0.1.2-plan.md (Per-Part diff vs baseline)",),
        source=_CPU_PART1,
        figures={
            "low_pct": 2.7,
            "high_pct": 10.7,
            "low_cell": "MWA_extended_off30",
            "cells": 4,
        },
    ),
    # ----------------------------------------------------------------------
    # v0.2.0: the realistic-size comparison against ducc0 (issue #33).
    #
    # The GH200-versus-ducc0 figures used to be a hand-written markdown table
    # with no JSON behind it, recorded in this module's docstring as one of the
    # three families that could not be recomputed. They now can be: the sweep is
    # committed as docs/benchmarks/v0.2.0-vs-ducc0-gh200.json.
    # ----------------------------------------------------------------------
    "realistic_forward_speedup": Citation(
        claim=(
            "'the forward is 1.7x to 3.8x faster than ducc0' at realistic sizes -- "
            "over the eight (telescope, pointing) cells, each against ducc0's own "
            "best measured thread count for that cell and operator."
        ),
        sites=("README.md (Performance notes -> GPU vs ducc0)",),
        source=_REALISTIC,
        figures={"min": 1.663, "max": 3.814, "median": 2.259, "cells": 8, "rounded": (1.7, 3.8)},
    ),
    "realistic_adjoint_speedup": Citation(
        claim=(
            "'the adjoint is 1.5x to 11.2x faster', and that the spread is not a "
            "continuum: five of the eight cells sit between 1.5x and 2.9x, and "
            "three -- MWA_extended zenith, MeerKAT at both pointings -- sit "
            "between 8.9x and 11.2x. Quoting only the range would suggest a "
            "typical figure of about 6x, which no cell shows."
        ),
        sites=("README.md (Performance notes -> GPU vs ducc0)",),
        source=_REALISTIC,
        figures={
            "min": 1.541,
            "max": 11.217,
            "median": 2.562,
            "cells": 8,
            "rounded": (1.5, 11.2),
            # The empty band, as thresholds rather than as cluster endpoints:
            # a rounded endpoint cannot double as a comparison bound, because
            # the cell whose value rounds to it falls the wrong side.
            "gap": (3.0, 8.0),
            "cluster_low": (1.541, 2.938, 5),
            "cluster_high": (8.850, 11.217, 3),
            "high_cells": ("MWA_extended_zenith", "MeerKAT_off30", "MeerKAT_zenith"),
        },
    ),
    "jax_faster_in_every_realistic_cell": Citation(
        claim=(
            "'faster in every one' of the sixteen (cell, operator) comparisons -- "
            "the universality, as opposed to the size of the gap."
        ),
        sites=("README.md (Performance notes -> GPU vs ducc0)",),
        source=_REALISTIC,
        figures={"comparisons": 16, "faster_in_all": True},
    ),
    "ducc0_thread_optimum": Citation(
        claim=(
            "'ducc0 is never fastest with all 288 hardware threads; at 288 it runs "
            "1.9x to 6.0x slower than its own best measured setting'. Stated this "
            "way on purpose: the thread grid is not the same for every fixture "
            "(EDA2 was measured at 8/16/32/64/128/288, MWA_extended at "
            "16/32/64/96/128/288), so the best *measured* count is a lower bound on "
            "tuning, not a located optimum. What the data does establish is that "
            "288 never wins and by how much it loses."
        ),
        sites=(
            "README.md (Performance notes -> GPU vs ducc0)",
            "README.md (nthreads)",
        ),
        source=_REALISTIC,
        figures={
            "n_hardware_threads": 288,
            "best_never_288": True,
            "penalty_min": 1.863,
            "penalty_max": 5.996,
            "penalty_rounded": (1.9, 6.0),
            "best_counts": {32: 2, 64: 13, 96: 1},
        },
    ),
    "realistic_sizing_rule": Citation(
        claim=(
            "the 'realistic' suite's problem sizes are not hand-picked: n_pix and "
            "n_rows both follow from stated instrument parameters through "
            "pixsize = lambda / (3 * B_max), n_pix = next even 5-smooth integer >= "
            "fov / pixsize, and n_rows = 150 * n_ant * (n_ant - 1) / 2. All eight "
            "numbers reproduce from the rule."
        ),
        sites=(
            "README.md (Performance notes -> how the benchmark problems are sized)",
            "docs/benchmarks/v0.2.0-vs-ducc0-gh200.json (sizing block)",
        ),
        source=_REALISTIC,
        figures={
            "n_pix": {"EDA2": 150, "MWA_compact": 144, "MWA_extended": 3600, "MeerKAT": 2700},
            "n_rows": {
                "EDA2": 4_896_000,
                "MWA_compact": 1_219_200,
                "MWA_extended": 1_219_200,
                "MeerKAT": 302_400,
            },
        },
    ),
    "memory_vs_ducc0": Citation(
        claim=(
            "'jax-nufft needs 2.7x to 54x the memory ducc0 does, and more in every "
            "one of the sixteen cells; the median is 8.5x'. The ratio is peak device "
            "HBM against ducc0's peak RSS less the interpreter's own footprint -- "
            "see the JSON's 'comparability' note, including that RSS quantisation "
            "makes the small-fixture ratios approximate."
        ),
        sites=("README.md (Performance notes -> memory)",),
        source=_MEMORY,
        figures={
            "min": 2.7,
            "max": 54.0,
            "median": 8.5,
            "cells": 16,
            "heavier_in_all": True,
            "min_cell": ("MWA_compact_off30", "vis2dirty"),
            "max_cell": ("MWA_extended_off30", "dirty2vis"),
        },
    ),
    "w_chunk_dial": Citation(
        claim=(
            "the memory/compute curve quoted for MWA_extended off30's forward "
            "(3600 pixels square, n_w = 140): dense_vmap 30.3 GB at 1.00x time, "
            "w_chunk=32 6.3 GB at 1.27x, w_chunk=8 1.9 GB at 1.73x, dense_scan "
            "0.41 GB at 2.35x. Monotone in both columns -- less scratch costs more "
            "time, with no inversion -- which is what makes it usable as a dial."
        ),
        sites=(
            "README.md (w_chunk: the memory/compute knob)",
            "README.md (Performance notes -> memory)",
        ),
        source=_MEMORY,
        figures={
            "fixture": "MWA_extended_off30",
            "op": "dirty2vis",
            "gb": {"dense_vmap": 29.6, "chunked32": 6.1, "chunked8": 1.9, "dense_scan": 0.40},
            "time_ratio": {
                "dense_vmap": 1.00,
                "chunked32": 1.27,
                "chunked8": 1.73,
                "dense_scan": 2.35,
            },
            "monotone": True,
        },
    ),
}

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"benchmark JSON not present at {path}")
    return json.loads(path.read_text())


def _gpu_rows() -> list[dict]:
    payload = _load(_GPU_BASELINE)
    if payload["fingerprint"]["jax_default_platform"] != "gpu":
        pytest.skip("baseline JSON was not captured on a GPU backend")
    return list(payload["rows"])


def _gpu_cells() -> dict[tuple[str, str], list[dict]]:
    """Group the GPU sweep by (op, fixture): the 20 'cells' the prose counts."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in _gpu_rows():
        cells[(row["op"], row["fixture"])].append(row)
    return dict(cells)


def _w_family(w_strategy: str) -> str:
    """``dense_scan`` -> ``scan``: the axis 'scan variants vs vmap' names."""
    return w_strategy.rsplit("_", 1)[1]


def _scan_vmap_pairs() -> list[tuple[tuple[str, str, str], str, str, float]]:
    """The 160 pairs the prose quotes, as ``(group, scan_w, vmap_w, ratio)``.

    A pair is one scan-family ``w_strategy`` against one vmap-family
    ``w_strategy`` measured in the same ``(op, fixture, channel_strategy)``
    group -- 2 x 2 per group, 40 groups. It gives
    ``(n, min, max, median, count below 5x)`` of
    ``(160, 1.4480, 32.6601, 6.0960, 72)``, which is what the prose quotes.

    Of the four obvious pairings, it is the only one that reproduces those
    figures; the other three were measured while writing this module and give
    (80, 1.4546, 32.6601, 5.4874, 36) pairing within a family at matched
    channel strategy, (160, 1.4539, 33.0505, 5.4827, 72) pairing within a
    family across channel strategies, and (320, 1.4477, 33.0505, 6.0960, 144)
    for the full per-cell cross product.

    **The endpoints are what identify it.** A fifth pairing -- every
    scan-family row of a cell against that cell's ``channel_strategy ==
    "vmap"`` vmap-family rows -- reproduces three of the four figures
    (n = 160, median 6.0960, 72 below 5x) and differs only in the span,
    1.4573 .. 33.0505. So a citation quoting only "160 pairs, median 6.1x"
    would not pin the definition; the ``1.448x``/``32.660x`` endpoints
    ``pair_spread`` asserts to three decimals are what does.
    """
    groups: dict[tuple[str, str, str], dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: {"scan": [], "vmap": []}
    )
    for row in _gpu_rows():
        key = (row["op"], row["fixture"], row["channel_strategy"])
        groups[key][_w_family(row["w_strategy"])].append((row["w_strategy"], row["median_s"]))
    pairs = []
    for key, families in sorted(groups.items()):
        for (scan_w, scan_s), (vmap_w, vmap_s) in product(families["scan"], families["vmap"]):
            pairs.append((key, scan_w, vmap_w, scan_s / vmap_s))
    return pairs


def _cpu_rows(path: Path) -> dict[tuple[str, str, str, str | None], dict]:
    """Index a pytest-benchmark JSON by ``(lib, op, fixture, w_strategy)``.

    ducc rows carry no ``w_strategy``; their key ends in ``None``.
    """
    out: dict[tuple[str, str, str, str | None], dict] = {}
    for bench in _load(path)["benchmarks"]:
        name = bench["name"]
        head, _, tail = name.partition("[")
        lib_op = head.removeprefix("test_bench_")
        lib, _, op = lib_op.partition("_")
        fixture, _, w_strategy = tail.rstrip("]").partition("-")
        out[(lib, op, fixture, w_strategy or None)] = bench["stats"]
    return out


def _cpu_cells(path: Path) -> tuple[tuple[str, str], ...]:
    """The ``(op, fixture)`` cells a CPU JSON *actually contains*, validated.

    Derived from the parsed benchmark names rather than from
    :data:`_CPU_FIXTURES`, and that is the whole point of the helper. Every
    CPU claim below is quantified over "every fixture x operator"; if the
    iteration comes from a literal tuple then ``len(cells) == 16`` is a
    statement about the tuple, not about the sweep, and a fixture appearing
    in either CPU JSON changes nothing anywhere. (Measured, 2026-09: cloning
    the ``EDA2_zenith`` rows of both CPU JSONs into a ninth fixture whose
    ``windowed_vmap`` is 1000x faster -- which makes
    ``docs/v0.1.2-plan.md``'s "dense_vmap wins on every fixture x operator"
    false -- fired **0** of the 18 cases while the CPU half iterated
    :data:`_CPU_FIXTURES`. Cloning a 21st cell into the GPU JSON fired 8, so
    the blindness was the CPU half's alone. With this helper the same CPU
    experiment fires all six CPU cases.)

    So the population is derived here, then asserted against the cited one,
    and every CPU case calls this instead of iterating a constant.
    """
    cited = CITATIONS["plan_dense_vmap_sweeps_cpu_json"].figures
    stats = _cpu_rows(path)
    cells = tuple(sorted({(op, fixture) for lib, op, fixture, _ in stats if lib == "jax"}))
    want = tuple(sorted(product(cited["ops"], cited["fixtures"])))
    assert cells == want, (
        f"{path.name} now measures {len(cells)} (op, fixture) cells, not the "
        f"{len(want)} docs/v0.1.2-plan.md's tables have a row for. Added: "
        f"{sorted(set(cells) - set(want))}; missing: {sorted(set(want) - set(cells))}."
        f"{_cite('plan_dense_vmap_sweeps_cpu_json')}"
    )
    # Every cell a complete 4-strategy block plus its ducc reference, or the
    # "winner" and "best/ducc" columns below compare different-sized menus.
    for op, fixture in cells:
        seen = {w for lib, o, f, w in stats if lib == "jax" and (o, f) == (op, fixture)}
        assert seen == set(cited["w_strategies"]), (
            f"{path.name}: {op}/{fixture} is not a complete "
            f"{len(cited['w_strategies'])}-strategy block: {sorted(seen)}"
        )
        assert ("ducc", op, fixture, None) in stats, (
            f"{path.name}: {op}/{fixture} has no ducc reference row, so its "
            "best/ducc ratio cannot be formed."
        )
    return cells


def _cite(key: str) -> str:
    """Render a citation for a failure message: sentence, then where it lives."""
    citation = CITATIONS[key]
    sites = "\n    ".join(citation.sites)
    return f"\n  claim: {citation.claim}\n  cited in:\n    {sites}\n  data: {citation.source.name}"


# --------------------------------------------------------------------------
# The GPU sweep (docs/benchmarks/v0.1.2-baseline-gpu.json)
# --------------------------------------------------------------------------


def test_the_sweep_has_twenty_cells_and_a_hundred_and_sixty_scan_vmap_pairs() -> None:
    """The denominator every other GPU claim is quoted over.

    If the sweep grows a fixture or a strategy, "160 pairs" and "20 cells" stop
    being true before any ratio does, and every sentence below inherits the
    error silently. So the population is asserted first and separately.
    """
    cited = CITATIONS["pair_population"].figures
    cells = _gpu_cells()
    pairs = _scan_vmap_pairs()
    assert len(cells) == cited["n_cells"], (
        f"the sweep now has {len(cells)} (op, fixture) cells, not "
        f"{cited['n_cells']}.{_cite('pair_population')}"
    )
    assert len(pairs) == cited["n_pairs"], (
        f"the sweep now yields {len(pairs)} scan/vmap pairs, not "
        f"{cited['n_pairs']}.{_cite('pair_population')}"
    )
    # Every cell must be a complete 4 x 2 block, or "best of family" and
    # "runner-up" below would be comparing different-sized menus per cell.
    for cell, rows in sorted(cells.items()):
        seen = {(r["w_strategy"], r["channel_strategy"]) for r in rows}
        assert seen == set(product(_W_STRATEGIES, ("scan", "vmap"))), (
            f"{cell[0]}/{cell[1]} is not a complete 4 w_strategy x 2 "
            f"channel_strategy block: {sorted(seen)}"
        )


def test_the_scan_family_is_slower_in_every_one_of_the_160_pairs() -> None:
    """The universality claim, which is what the "never scan on GPU" rule rests on.

    Stated as an "in every pair" rather than as a size, because the size is
    what went wrong: the sweep was originally written up as "5-30x" and 72 of
    the 160 pairs are below 5x (see the case below).
    """
    pairs = _scan_vmap_pairs()
    faster = [(g, s, v, r) for g, s, v, r in pairs if r <= 1.0]
    assert not faster, (
        f"{len(faster)} of {len(pairs)} scan/vmap pairs have the scan family at "
        f"or faster than the vmap family, e.g. {faster[:3]}. The rule 'never "
        f"auto-pick a scan strategy on GPU' is justified by this being empty."
        f"{_cite('scan_always_slower')}"
    )


def test_the_scan_vmap_pairs_span_the_cited_range_and_median() -> None:
    """The quoted spread, to the three decimals AGENTS.md sec 9 states it in."""
    cited = CITATIONS["pair_spread"].figures
    ratios = [r for _, _, _, r in _scan_vmap_pairs()]
    got = {
        "min": round(min(ratios), 3),
        "max": round(max(ratios), 3),
        "median": round(statistics.median(ratios), 3),
    }
    assert got == {k: cited[k] for k in ("min", "max", "median")}, (
        f"the 160 scan/vmap pairs now span {got['min']}x to {got['max']}x with "
        f"a median of {got['median']}x; the prose says {cited['min']}x to "
        f"{cited['max']}x, median {cited['median']}x.{_cite('pair_spread')}"
    )
    # The same figures rounded the way src/, README.md and the test docstrings
    # quote them. Asserted separately: rounding 1.448 to "1.45x" is a second
    # editorial step and has its own chance to drift.
    rounded = (round(min(ratios), 2), round(max(ratios), 1), round(statistics.median(ratios), 1))
    assert rounded == cited["rounded"], (
        f"the coarse restatement is now {rounded[0]}x to {rounded[1]}x (median "
        f"{rounded[2]}x); src/, README.md and the test docstrings quote "
        f"'{cited['rounded'][0]}x to {cited['rounded'][1]}x (median "
        f"{cited['rounded'][2]}x)'.{_cite('pair_spread')}"
    )


def test_seventy_two_of_the_160_pairs_are_below_five_times() -> None:
    """The count that falsified the original "5-30x" write-up."""
    cited = CITATIONS["pairs_below_5x"].figures
    ratios = [r for _, _, _, r in _scan_vmap_pairs()]
    below = sum(1 for r in ratios if r < 5.0)
    assert below == cited["below_5x"], (
        f"{below} of {len(ratios)} scan/vmap pairs are below 5x; the correction "
        f"in AGENTS.md sec 9 says {cited['below_5x']}.{_cite('pairs_below_5x')}"
    )


def test_no_named_subset_of_the_sweep_yields_the_five_to_thirty_range() -> None:
    """AGENTS.md sec 9 names three subsets and says none produces "5-30x".

    Each is checked on its own, which is how the sentence enumerates them, and
    the check is *containment*: a subset reproduces the discarded "5-30x"
    write-up if it lies inside [5, 30], because that is what would let someone
    re-derive the range from this data. An earlier form of this case asserted
    only that the endpoints did not round to exactly (5.0, 30.0), which is a
    measure-zero coincidence: measured 2026-09, 200 random re-measurements of
    the JSON (per-row lognormal noise, sigma 0.02 to 3.00) and 20 000 uniform
    rescales of the scan family over s in [0.001, 20.000] fired it zero times,
    and it is immune to a uniform rescale by construction, since a rescale
    leaves each subset's max/min ratio alone and none of the three is the 6.0
    that form needs (they are 13.2752, 15.7924 and 11.6752).

    Containment is quiet under the same noise -- zero fires on those same 200
    draws -- but for a reason, not by construction: "best-of-family per cell"
    already satisfies the upper end at 29.3932x and is held out of [5, 30]
    only by its 2.5176x low end, so a regime in which the scan penalty became
    more uniform reaches it. Compressing the log-ratios toward 10x does
    exactly that and fires this case, where the old form still does not.

    The conjunction is evaluated as a population of its own, because
    AGENTS.md sec 9 now spends a parenthetical on it and quotes three figures
    for it. That parenthetical is a hand-written aggregate of the JSON, which
    is the exact thing this module exists to stop, so it is pinned to
    ``(min, max, n)`` rather than merely described. Note the direction:
    the three named subsets must *not* be contained, the conjunction *is*.

    "The whole sweep" is deliberately not in the list. It is not one of the
    three subsets the sentence names, and its non-containment is already
    pinned harder elsewhere -- ``pair_spread`` fixes its endpoints at
    1.448x/32.660x to three decimals.
    """
    cited = CITATIONS["no_subset_is_5_to_30x"].figures

    def by_group(rows: list[dict]) -> list[float]:
        groups: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
            lambda: {"scan": [], "vmap": []}
        )
        for row in rows:
            key = (row["op"], row["fixture"], row["channel_strategy"])
            groups[key][_w_family(row["w_strategy"])].append(row["median_s"])
        return [s / v for g in groups.values() for s, v in product(g["scan"], g["vmap"])]

    def best_of_family(rows: list[dict]) -> list[float]:
        cells: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: {"scan": [], "vmap": []}
        )
        for row in rows:
            cells[(row["op"], row["fixture"])][_w_family(row["w_strategy"])].append(row["median_s"])
        return [min(g["scan"]) / min(g["vmap"]) for g in cells.values()]

    rows = _gpu_rows()
    off_zenith = [r for r in rows if r["fixture"].endswith("off30")]
    no_gh200_large = [r for r in rows if not r["fixture"].startswith("GH200_large")]

    subsets = {
        "off-zenith only": by_group(off_zenith),
        "excluding GH200_large": by_group(no_gh200_large),
        "best-of-family per cell": best_of_family(rows),
    }
    assert tuple(subsets) == cited["named_subsets"], (
        f"the subsets evaluated here, {tuple(subsets)}, are no longer the ones "
        f"AGENTS.md sec 9 names, {cited['named_subsets']}."
        f"{_cite('no_subset_is_5_to_30x')}"
    )
    for name, ratios in subsets.items():
        lo, hi = min(ratios), max(ratios)
        assert not (cited["low"] <= lo and hi <= cited["high"]), (
            f"the subset {name!r} now spans {lo:.3f}x to {hi:.3f}x over "
            f"{len(ratios)} values, which lies inside the '{cited['low']:g}-"
            f"{cited['high']:g}x' the sweep was originally written up with. "
            f"AGENTS.md sec 9 says no named subset, taken on its own, yields "
            f"that range.{_cite('no_subset_is_5_to_30x')}"
        )

    # The conjunction of all three, which the same paragraph quotes and which
    # *is* inside [5, 30] -- so it is pinned by its figures, not by
    # non-containment.
    conjoined = best_of_family(
        [r for r in off_zenith if not r["fixture"].startswith("GH200_large")]
    )
    got = (round(min(conjoined), 3), round(max(conjoined), 3), len(conjoined))
    assert got == cited["conjunction"], (
        f"conjoining all three named restrictions now gives {got[0]}x to "
        f"{got[1]}x over {got[2]} cells; AGENTS.md sec 9's parenthetical says "
        f"{cited['conjunction'][0]}x to {cited['conjunction'][1]}x over "
        f"{cited['conjunction'][2]} cells.{_cite('no_subset_is_5_to_30x')}"
    )
    # The two below add nothing while the pin above holds, and that is the
    # point: a re-measurement reaches this module as an edit to
    # ``conjunction``, and these are the two things AGENTS.md sec 9 asserts
    # about the conjunction that such an edit could quietly falsify -- that it
    # lies inside 5-30x at all, and that its endpoints are not the quoted 5
    # and 30.
    lo, hi = min(conjoined), max(conjoined)
    assert cited["low"] <= lo and hi <= cited["high"], (
        f"the conjunction now spans {lo:.3f}x to {hi:.3f}x, which is *not* "
        f"inside {cited['low']:g}-{cited['high']:g}x. AGENTS.md sec 9 says it "
        f"is, and that is the whole reason the sentence says 'taken on its "
        f"own'.{_cite('no_subset_is_5_to_30x')}"
    )
    assert (round(lo, 1), round(hi, 1)) != (cited["low"], cited["high"]), (
        f"the conjunction now spans {lo:.3f}x to {hi:.3f}x, whose endpoints "
        f"round to exactly {cited['low']:g} and {cited['high']:g}: it does "
        f"reproduce the discarded write-up after all, and AGENTS.md sec 9's "
        f"'its endpoints are still not 5 and 30' is false."
        f"{_cite('no_subset_is_5_to_30x')}"
    )


def test_dense_vmap_wins_seventeen_of_the_twenty_cells() -> None:
    """AGENTS.md sec 9's headline count for the sweep."""
    cited = CITATIONS["dense_vmap_cell_wins"].figures
    winners = [
        min(rows, key=lambda r: r["median_s"])["w_strategy"] for rows in _gpu_cells().values()
    ]
    wins = winners.count("dense_vmap")
    assert (wins, len(winners)) == (cited["wins"], cited["cells"]), (
        f"dense_vmap wins {wins} of {len(winners)} cells; AGENTS.md sec 9 says "
        f"{cited['wins']}/{cited['cells']}.{_cite('dense_vmap_cell_wins')}"
    )


def test_windowed_vmap_wins_exactly_the_three_gh200_large_cells() -> None:
    """ "...winning only the 50k-row GH200_large fixture".

    An "only" claim, so both halves are asserted: nothing outside GH200_large
    wins, and the three cells that do are the ones wgridder.py names (the
    adjoint at both pointings and the forward at zenith). The fourth
    GH200_large cell -- the off-zenith forward -- goes to dense_vmap, which is
    the asymmetry the docstring explains by the higher n_w.
    """
    cited = CITATIONS["windowed_vmap_cell_wins"].figures
    won = tuple(
        sorted(
            cell
            for cell, rows in _gpu_cells().items()
            if min(rows, key=lambda r: r["median_s"])["w_strategy"] == "windowed_vmap"
        )
    )
    assert won == cited["cells"], (
        f"windowed_vmap now wins {won}, not {cited['cells']}.{_cite('windowed_vmap_cell_wins')}"
    )
    assert all(fixture.startswith("GH200_large") for _, fixture in won), (
        f"windowed_vmap wins outside the 50k-row GH200_large fixture: {won}."
        f"{_cite('windowed_vmap_cell_wins')}"
    )


def test_the_heuristic_picks_the_measured_winner_in_every_cell_and_slice() -> None:
    """20/20 cells, and 40/40 per-channel slices.

    The per-channel half is the part worth asserting separately: picking the
    winner of a cell aggregated over channel strategies would be compatible
    with losing one of the two slices it is aggregated from.
    """
    cited = CITATIONS["heuristic_picks_the_winner"].figures
    cell_hits = 0
    slice_hits = 0
    slices = 0
    misses = []
    for (op, fixture), rows in sorted(_gpu_cells().items()):
        # The plan-shaped inputs are constant across a cell's eight rows;
        # asserted rather than assumed, since reading them off one row is what
        # tests/test_auto_strategy_acceptance.py does too.
        for key in ("n_w", "w_kernel_width", "window_padding_overhead", "n_rows"):
            assert len({r[key] for r in rows}) == 1, f"{op}/{fixture}: {key} varies within a cell"
        plan = SimpleNamespace(
            n_w=rows[0]["n_w"],
            w_kernel_width=rows[0]["w_kernel_width"],
            window_padding_overhead=rows[0]["window_padding_overhead"],
            # issue #26 split the padding field by direction; these rows are a
            # pre-#26 sweep and record only the un-bucketed number, which is
            # what the forward still reads. Mirroring it keeps this a replay of
            # the recorded sweep rather than a re-derivation of how the current
            # planner would bucket a plan the JSON does not describe.
            window_padding_overhead_adjoint=rows[0]["window_padding_overhead"],
            n_rows=rows[0]["n_rows"],
        )
        picked = _auto_w_strategy_gpu(plan, is_adjoint=op == "vis2dirty")
        winner = min(rows, key=lambda r: r["median_s"])["w_strategy"]
        cell_hits += picked == winner
        if picked != winner:
            misses.append((op, fixture, picked, winner))
        for channel in ("scan", "vmap"):
            sub = [r for r in rows if r["channel_strategy"] == channel]
            slices += 1
            slice_winner = min(sub, key=lambda r: r["median_s"])["w_strategy"]
            slice_hits += picked == slice_winner
            if picked != slice_winner:
                misses.append((op, f"{fixture} ({channel} channels)", picked, slice_winner))
    assert (cell_hits, slice_hits, slices) == (
        cited["cells"],
        cited["channel_slices"],
        cited["channel_slices"],
    ), (
        f"the heuristic picks the winner in {cell_hits}/{cited['cells']} cells "
        f"and {slice_hits}/{slices} channel slices, not 20/20 and 40/40. "
        f"Misses: {misses}.{_cite('heuristic_picks_the_winner')}"
    )


def test_the_runner_up_is_not_within_the_fifteen_percent_bar() -> None:
    """The corrected form of "the runner-up is always within the 15% bar".

    That sentence was backwards, and it mattered: it made a wrong pick look
    like a bounded loss. Both populations wgridder.py quotes are checked -- the
    four strategies each aggregated by their better channel strategy, and the
    40 per-channel slices taken separately.

    The 40 slices are *also* checked as their two halves, because that is
    where issue #49's own "worst 7.22x slower" comes from and it was written
    up as not reproducing. It reproduces: restricted to the
    ``channel_strategy == "scan"`` half the gap is 1.0847x to 7.2205x over 20
    cells (median 1.8620x), and the worst cell is ``dirty2vis`` /
    ``EDA2_off30`` at ``windowed_vmap`` 152.649 ms over ``dense_vmap``
    21.141 ms. The ``vmap`` half is 1.0738x to 7.3605x, which is the 7.36x the
    tree quotes; the union of the two is the 40-slice line. So 7.22x, 7.31x
    and 7.36x are three different populations, all three correct: 7.31x
    aggregates each strategy by its better channel before ranking, while 7.22x
    and 7.36x are the same per-slice ranking taken over one channel half or
    the other. Worth pinning, since "which half" is what the apparent
    arithmetic disagreement turned out to be.
    """
    cited = CITATIONS["runner_up_gap"].figures
    per_cell = []
    per_slice = []
    per_channel: dict[str, list[float]] = {"scan": [], "vmap": []}
    worst_by_channel: dict[str, tuple[float, tuple[str, str], str, float, str, float]] = {}
    for cell, rows in sorted(_gpu_cells().items()):
        best_by_w: dict[str, float] = {}
        for row in rows:
            w = row["w_strategy"]
            best_by_w[w] = min(best_by_w.get(w, row["median_s"]), row["median_s"])
        ordered = sorted(best_by_w.values())
        per_cell.append(ordered[1] / ordered[0])
        for channel in ("scan", "vmap"):
            sub = sorted(
                (r["median_s"], r["w_strategy"]) for r in rows if r["channel_strategy"] == channel
            )
            gap = sub[1][0] / sub[0][0]
            per_slice.append(gap)
            per_channel[channel].append(gap)
            if gap > worst_by_channel.get(channel, (0.0,))[0]:
                worst_by_channel[channel] = (
                    gap,
                    cell,
                    sub[1][1],
                    sub[1][0] * 1e3,
                    sub[0][1],
                    sub[0][0] * 1e3,
                )

    got = (
        round(min(per_cell), 2),
        round(max(per_cell), 2),
        round(statistics.median(per_cell), 2),
        sum(1 for r in per_cell if r <= 1.15),
    )
    want = (cited["min"], cited["max"], cited["median"], cited["within_15pct"])
    assert got == want, (
        f"across the 20 cells the runner-up now trails by {got[0]}x to "
        f"{got[1]}x (median {got[2]}x) with {got[3]} inside 15%; the prose says "
        f"{want[0]}x to {want[1]}x (median {want[2]}x) with {want[3]} inside."
        f"{_cite('runner_up_gap')}"
    )
    got_slice = (
        round(min(per_slice), 2),
        round(max(per_slice), 2),
        sum(1 for r in per_slice if r <= 1.15),
    )
    want_slice = (cited["slice_min"], cited["slice_max"], cited["slice_within_15pct"])
    assert got_slice == want_slice, (
        f"across the 40 per-channel slices the runner-up now trails by "
        f"{got_slice[0]}x to {got_slice[1]}x with {got_slice[2]} inside 15%; "
        f"the prose says {want_slice[0]}x to {want_slice[1]}x with "
        f"{want_slice[2]} inside.{_cite('runner_up_gap')}"
    )
    for channel in ("scan", "vmap"):
        gaps = per_channel[channel]
        got_channel = (
            round(min(gaps), 4),
            round(max(gaps), 4),
            round(statistics.median(gaps), 4),
            len(gaps),
        )
        want_channel = cited[f"{channel}_channel"]
        assert got_channel == want_channel, (
            f"the {channel}-channel half of the 40 slices now runs "
            f"{got_channel[0]}x to {got_channel[1]}x (median {got_channel[2]}x) "
            f"over {got_channel[3]} cells, not {want_channel[0]}x to "
            f"{want_channel[1]}x (median {want_channel[2]}x) over "
            f"{want_channel[3]}.{_cite('runner_up_gap')}"
        )
    _, worst_cell, runner_up_w, runner_up_ms, winner_w, winner_ms = worst_by_channel["scan"]
    got_worst = (worst_cell, round(runner_up_ms, 3), round(winner_ms, 3))
    want_worst = (
        cited["scan_channel_worst_cell"],
        cited["scan_channel_worst_ms"][0],
        cited["scan_channel_worst_ms"][1],
    )
    assert got_worst == want_worst, (
        f"the widest scan-channel gap -- issue #49's 'worst 7.22x slower' -- "
        f"is now {worst_cell}, {runner_up_w} {runner_up_ms:.3f} ms over "
        f"{winner_w} {winner_ms:.3f} ms, not {want_worst}."
        f"{_cite('runner_up_gap')}"
    )


def test_the_auto_pick_meets_the_fifteen_percent_bar_by_being_the_best() -> None:
    """The acceptance bar README.md and AGENTS.md quote, plus its slack.

    ``tests/test_auto_strategy_acceptance.py`` already asserts the bar. What is
    asserted here is the sentence wgridder.py adds about it: the bar is met
    "trivially ... by picking the best one every time", i.e. the worst cell's
    ratio is 1.0x and not merely under 1.15x. If a retune ever spends real
    slack against that bar, this is the case that says so.

    Asserted exactly rather than to two decimals: "picking the best one" makes
    the picked row *be* the min row, so the ratio is the same float divided by
    itself and the tolerance buys nothing except room for a pick up to 0.4%
    off the best to still read as "the best one every time".
    """
    cited = CITATIONS["acceptance_bar"].figures
    worst = 0.0
    worst_cell = None
    for (op, fixture), rows in sorted(_gpu_cells().items()):
        plan = SimpleNamespace(
            n_w=rows[0]["n_w"],
            w_kernel_width=rows[0]["w_kernel_width"],
            window_padding_overhead=rows[0]["window_padding_overhead"],
            # issue #26 split the padding field by direction; these rows are a
            # pre-#26 sweep and record only the un-bucketed number, which is
            # what the forward still reads. Mirroring it keeps this a replay of
            # the recorded sweep rather than a re-derivation of how the current
            # planner would bucket a plan the JSON does not describe.
            window_padding_overhead_adjoint=rows[0]["window_padding_overhead"],
            n_rows=rows[0]["n_rows"],
        )
        picked = _auto_w_strategy_gpu(plan, is_adjoint=op == "vis2dirty")
        picked_best = min(r["median_s"] for r in rows if r["w_strategy"] == picked)
        ratio = picked_best / min(r["median_s"] for r in rows)
        if ratio > worst:
            worst, worst_cell = ratio, (op, fixture)
    assert worst <= cited["bar"], (
        f"{worst_cell} is {worst:.3f}x off the best strategy, over the "
        f"{cited['bar']}x acceptance bar.{_cite('acceptance_bar')}"
    )
    assert worst == cited["actual_worst"], (
        f"the heuristic no longer picks the outright best strategy in every "
        f"cell: worst is {worst_cell} at {worst:.6f}x. The prose reads the "
        f"{cited['bar']}x bar as met trivially; at {worst:.6f}x it is not."
        f"{_cite('acceptance_bar')}"
    )


# --------------------------------------------------------------------------
# The CPU JSON cited by docs/v0.1.2-plan.md
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "path", "stat"),
    [
        ("Baseline summary", _CPU_BASELINE, "median"),
        ("Part 1 GH200 snapshot", _CPU_PART1, "mean"),
    ],
)
def test_dense_vmap_wins_every_cpu_json_cell(label: str, path: Path, stat: str) -> None:
    """Both plan-doc tables say dense_vmap wins every fixture x operator.

    The statistic differs between them -- the baseline table is captioned
    "median ms" and the Part 1 table "mean wall-clock" -- so each is recomputed
    on the statistic its own caption names.

    The cells come from :func:`_cpu_cells`, i.e. from the JSON, so "every
    fixture x operator" is quantified over what the sweep measured rather than
    over a tuple in this file.
    """
    cited = CITATIONS["plan_dense_vmap_sweeps_cpu_json"].figures
    stats = _cpu_rows(path)
    cells = _cpu_cells(path)
    losses = []
    for op, fixture in cells:
        times = {w: stats[("jax", op, fixture, w)][stat] for w in cited["w_strategies"]}
        winner = min(times, key=lambda w: times[w])
        if winner != "dense_vmap":
            losses.append((op, fixture, winner))
    assert len(cells) == cited["cells"], (
        f"{label}: the sweep now has {len(cells)} (op, fixture) cells, not the "
        f"{cited['cells']} the table has rows for."
        f"{_cite('plan_dense_vmap_sweeps_cpu_json')}"
    )
    assert not losses, (
        f"docs/v0.1.2-plan.md's {label} says dense_vmap wins every fixture x "
        f"operator, but on {stat} it loses in {losses}."
        f"{_cite('plan_dense_vmap_sweeps_cpu_json')}"
    )


def test_the_long_scan_worst_case_is_seven_tenths_to_one_and_a_bit_seconds() -> None:
    """ ">=0.7-1.1 s on a single H100 for a 256^2 image", MWA_extended off30.

    Asserted on the endpoints rounded to the one decimal the sentence quotes.
    The unrounded span is 0.724-1.117 s, so a containment test would reject the
    citation over the upper end's third digit -- which is the citation being
    written to one decimal, not the citation being wrong.
    """
    cited = CITATIONS["plan_long_scan_worst_case"].figures
    stats = _cpu_rows(_CPU_BASELINE)
    ops = sorted({op for op, fixture in _cpu_cells(_CPU_BASELINE)})
    times = {
        (op, w): stats[("jax", op, "MWA_extended_off30", w)]["median"]
        for op, w in product(ops, ("dense_scan", "windowed_scan"))
    }
    lo, hi = min(times.values()), max(times.values())
    assert (round(lo, 1), round(hi, 1)) == (cited["low_s"], cited["high_s"]), (
        f"the MWA_extended off30 scan-family worst case now spans {lo:.3f}-"
        f"{hi:.3f} s, which does not round to the cited {cited['low_s']}-"
        f"{cited['high_s']} s: { {k: round(v, 4) for k, v in times.items()} }"
        f"{_cite('plan_long_scan_worst_case')}"
    )


def _part1_best_over_ducc() -> dict[tuple[str, str], float]:
    """Part 1's ``best/ducc`` column: ducc mean over the best jax mean.

    Over the cells :func:`_cpu_cells` derives from the JSON, so a fixture
    entering the sweep enters this column too.
    """
    strategies = CITATIONS["plan_dense_vmap_sweeps_cpu_json"].figures["w_strategies"]
    stats = _cpu_rows(_CPU_PART1)
    out = {}
    for op, fixture in _cpu_cells(_CPU_PART1):
        best = min(stats[("jax", op, fixture, w)]["mean"] for w in strategies)
        out[(op, fixture)] = stats[("ducc", op, fixture, None)]["mean"] / best
    return out


def test_the_gpu_loses_to_ducc_on_exactly_the_named_small_fixtures() -> None:
    """ "On the small or low-w fixtures ... the GPU loses to ducc".

    An enumeration, so it is checked as a set equality rather than as
    "these ones lose": a sixth losing cell would leave the sentence's list
    incomplete without contradicting any of the cells it does list.
    """
    cited = CITATIONS["plan_gpu_loses_to_ducc"].figures
    losing = tuple(sorted(cell for cell, ratio in _part1_best_over_ducc().items() if ratio < 1.0))
    assert losing == tuple(sorted(cited["cells"])), (
        f"the cells where jax on H100 loses to single-thread ducc are now "
        f"{losing}, not the {tuple(sorted(cited['cells']))} the plan doc lists."
        f"{_cite('plan_gpu_loses_to_ducc')}"
    )


def test_the_large_fixture_speedup_over_ducc_spans_the_corrected_range() -> None:
    """The corrected form of "3-7x for both operators" (issue #49).

    The original sentence quantified 3-7x over MeerKAT and MWA_extended, both
    pointings, both operators -- eight cells, two of which are outside it: the
    MWA_extended off30 forward at 1.29x, well below, and the MWA_extended
    zenith adjoint at 7.45x, above. The corrected sentence states the true span
    and how many cells sit in 3-7x, and both halves are asserted here so that
    neither can rot alone.
    """
    cited = CITATIONS["plan_large_fixture_speedup"].figures
    ratios = {
        cell: ratio
        for cell, ratio in _part1_best_over_ducc().items()
        if cell[1].startswith(("MeerKAT", "MWA_extended"))
    }
    assert len(ratios) == cited["cells"], f"{len(ratios)} cells, not {cited['cells']}"
    lo_cell = min(ratios, key=lambda c: ratios[c])
    hi_cell = max(ratios, key=lambda c: ratios[c])
    got = (round(ratios[lo_cell], 2), round(ratios[hi_cell], 2))
    assert got == (cited["min"], cited["max"]), (
        f"the MeerKAT / MWA_extended cells now span {got[0]}x to {got[1]}x, not "
        f"the cited {cited['min']}x to {cited['max']}x."
        f"{_cite('plan_large_fixture_speedup')}"
    )
    assert (lo_cell, hi_cell) == (cited["below"], cited["above"]), (
        f"the two cells the corrected sentence names as falling outside 3-7x "
        f"are now {lo_cell} (low) and {hi_cell} (high), not {cited['below']} "
        f"and {cited['above']}.{_cite('plan_large_fixture_speedup')}"
    )
    inside = sum(1 for r in ratios.values() if 3.0 <= r <= 7.0)
    assert inside == cited["in_3_to_7"], (
        f"{inside} of the {len(ratios)} cells sit in 3-7x, not the cited "
        f"{cited['in_3_to_7']}.{_cite('plan_large_fixture_speedup')}"
    )


def _part1_vs_baseline(w_strategy: str) -> dict[tuple[str, str], float]:
    """Part 1 median over baseline median, as a percentage change per cell.

    Both files are asked for their own cells and the two are required to
    agree, so a fixture added to one sweep and not the other is a failure here
    rather than a silently dropped column.
    """
    base = _cpu_rows(_CPU_BASELINE)
    part1 = _cpu_rows(_CPU_PART1)
    cells = _cpu_cells(_CPU_BASELINE)
    assert cells == _cpu_cells(_CPU_PART1), (
        "the baseline and Part 1 CPU sweeps no longer measure the same cells, "
        "so a per-Part diff cannot be formed over them."
    )
    return {
        (op, fixture): 100.0
        * (
            part1[("jax", op, fixture, w_strategy)]["median"]
            / base[("jax", op, fixture, w_strategy)]["median"]
            - 1.0
        )
        for op, fixture in cells
    }


def test_the_dense_scan_worst_regression_is_the_meerkat_zenith_adjoint() -> None:
    """The corrected "dense_scan worst regression" row (issue #49).

    The plan doc's per-Part diff read "+5.2% (within noise floor of untouched
    paths)". Recomputed over all sixteen dense_scan cells, +5.2% is the *third*
    largest change: MeerKAT zenith adjoint moved +42.3% (12.675 -> 18.035 ms
    median) and EDA2 zenith adjoint +6.0%. Part 1 changed only the windowed
    forward and cannot reach a dense_scan adjoint, and that cell's own
    dispersion is wide (part-1 stddev 2.08 ms over 43 rounds against the
    baseline's 0.47 ms over 77), so this is a noisy cell rather than a Part 1
    effect -- but "worst regression: +5.2%" is not what the committed data says,
    and the corrected sentence names both.
    """
    cited = CITATIONS["plan_dense_scan_regression"].figures
    deltas = _part1_vs_baseline("dense_scan")
    assert len(deltas) == cited["cells"], f"{len(deltas)} cells, not {cited['cells']}"
    ordered = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)
    (worst_cell, worst), (second_cell, second) = ordered[0], ordered[1]
    assert (worst_cell, round(worst, 1)) == (cited["worst_cell"], cited["worst_pct"]), (
        f"the worst dense_scan change from baseline to Part 1 is now "
        f"{worst_cell} at {worst:+.1f}%, not {cited['worst_cell']} at "
        f"{cited['worst_pct']:+.1f}%.{_cite('plan_dense_scan_regression')}"
    )
    assert (second_cell, round(second, 1)) == (cited["runner_up_cell"], cited["runner_up_pct"]), (
        f"the worst of the other fifteen dense_scan cells is now {second_cell} "
        f"at {second:+.1f}%, not {cited['runner_up_cell']} at "
        f"{cited['runner_up_pct']:+.1f}%.{_cite('plan_dense_scan_regression')}"
    )


def test_the_windowed_scan_off_zenith_forward_regression_spans_the_cited_range() -> None:
    """ "+2.7% to +10.7% (MWA_ext off30 +2.7% with n_w=515)".

    Both endpoints and the fixture the low end is attributed to, because the
    parenthetical is the part a reader checks: it names which cell is +2.7%.
    """
    cited = CITATIONS["plan_windowed_scan_regression"].figures
    deltas = {
        fixture: delta
        for (op, fixture), delta in _part1_vs_baseline("windowed_scan").items()
        if op == "dirty2vis" and fixture.endswith("off30")
    }
    assert len(deltas) == cited["cells"], f"{len(deltas)} off30 cells, not {cited['cells']}"
    low_fixture = min(deltas, key=lambda f: deltas[f])
    high_fixture = max(deltas, key=lambda f: deltas[f])
    got = (round(deltas[low_fixture], 1), round(deltas[high_fixture], 1))
    assert got == (cited["low_pct"], cited["high_pct"]), (
        f"the off-zenith windowed_scan forward now spans {got[0]:+.1f}% to "
        f"{got[1]:+.1f}%, not the cited {cited['low_pct']:+.1f}% to "
        f"{cited['high_pct']:+.1f}%.{_cite('plan_windowed_scan_regression')}"
    )
    assert low_fixture == cited["low_cell"], (
        f"the {cited['low_pct']:+.1f}% end is now {low_fixture}, not the "
        f"{cited['low_cell']} the parenthetical names."
        f"{_cite('plan_windowed_scan_regression')}"
    )


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_every_committed_benchmark_json_is_covered_by_a_case() -> None:
    """A new JSON in docs/benchmarks/ is a new thing prose can cite wrongly.

    The guard against a whole *file* arriving uncovered: someone commits a
    fresh sweep, quotes an aggregate of it in a comment, and nothing here
    notices because no case names that file. New *rows* inside an already
    covered file are a different failure and are caught elsewhere -- the GPU
    half by ``test_the_sweep_has_twenty_cells_...``, the CPU half by
    :func:`_cpu_cells` -- which is what this docstring used to claim for
    itself and does not reach.
    """
    committed = {p.name for p in sorted(_BENCH_DIR.glob("*.json"))}
    covered = {c.source.name for c in CITATIONS.values()}
    missing = committed - covered
    assert not missing, (
        f"{sorted(missing)} is committed under docs/benchmarks/ but no entry of "
        "tests/test_benchmark_claims.py::CITATIONS recomputes anything from it. "
        "Either add a case for the aggregates its prose cites, or record it in "
        "this module's docstring as not cited."
    )
    assert covered <= committed, (
        f"CITATIONS names {sorted(covered - committed)}, which is not committed "
        "under docs/benchmarks/."
    )


# --------------------------------------------------------------------------
# v0.2.0: the realistic-size sweep against ducc0
# (docs/benchmarks/v0.2.0-vs-ducc0-gh200.json)
#
# Every ducc0 row in a cell is one thread count. "ducc0's best" throughout
# means the fastest thread count *measured for that cell and operator*, which
# is the only fair comparison available: quoting ducc0 at a fixed setting would
# either flatter it or handicap it depending on the setting chosen.
# --------------------------------------------------------------------------

_REALISTIC_OPS = ("dirty2vis", "vis2dirty")


def _realistic_cells(suite: str = "realistic") -> dict[str, list[dict]]:
    """Group one suite of the v0.2.0 sweep by fixture."""
    cells: dict[str, list[dict]] = defaultdict(list)
    for row in _load(_REALISTIC)["rows"]:
        if row["suite"] == suite:
            cells[row["fixture"]].append(row)
    return dict(cells)


def _speedups(op: str, suite: str = "realistic") -> dict[str, float]:
    """jax-nufft's speedup over the best measured ducc0 thread count, per cell."""
    stat = f"{op}_median_ms"
    out = {}
    for fixture, rows in _realistic_cells(suite).items():
        jax_rows = [r for r in rows if r["impl"] == "jax-nufft"]
        ducc_rows = [r for r in rows if r["impl"] == "ducc0"]
        assert len(jax_rows) == 1, f"{fixture} has {len(jax_rows)} jax-nufft rows, expected 1"
        assert ducc_rows, f"{fixture} has no ducc0 rows to compare against"
        out[fixture] = min(r[stat] for r in ducc_rows) / jax_rows[0][stat]
    return out


def test_the_realistic_forward_speedup_spans_the_cited_range() -> None:
    """The forward's spread is tight; the README quotes it as its own range."""
    cited = CITATIONS["realistic_forward_speedup"].figures
    ratios = _speedups("dirty2vis")
    assert len(ratios) == cited["cells"], (
        f"{len(ratios)} realistic cells, prose counts {cited['cells']}."
        f"{_cite('realistic_forward_speedup')}"
    )
    got = (
        round(min(ratios.values()), 3),
        round(max(ratios.values()), 3),
        round(statistics.median(ratios.values()), 3),
    )
    assert got == (cited["min"], cited["max"], cited["median"]), (
        f"the forward now spans {got[0]}x to {got[1]}x (median {got[2]}x); the "
        f"prose says {cited['min']}x to {cited['max']}x (median "
        f"{cited['median']}x).{_cite('realistic_forward_speedup')}"
    )
    rounded = (round(min(ratios.values()), 1), round(max(ratios.values()), 1))
    assert rounded == cited["rounded"], (
        f"README.md quotes '{cited['rounded'][0]}x to {cited['rounded'][1]}x'; "
        f"the data rounds to {rounded[0]}x to {rounded[1]}x."
        f"{_cite('realistic_forward_speedup')}"
    )


def test_the_realistic_adjoint_speedup_is_two_clusters_not_a_continuum() -> None:
    """The shape of the adjoint's spread, not just its ends.

    A bare "1.5x to 11.2x" invites the reader to average it to about 6x, and no
    cell is anywhere near 6x. The claim the README makes is the two clusters and
    the empty gap between them, so that is what is checked -- including that the
    gap really is empty, which is the part a new fixture could quietly break.
    """
    cited = CITATIONS["realistic_adjoint_speedup"].figures
    ratios = _speedups("vis2dirty")
    got = (
        round(min(ratios.values()), 3),
        round(max(ratios.values()), 3),
        round(statistics.median(ratios.values()), 3),
    )
    assert got == (cited["min"], cited["max"], cited["median"]), (
        f"the adjoint now spans {got[0]}x to {got[1]}x (median {got[2]}x); the "
        f"prose says {cited['min']}x to {cited['max']}x (median "
        f"{cited['median']}x).{_cite('realistic_adjoint_speedup')}"
    )

    low_min, low_max, low_n = cited["cluster_low"]
    high_min, high_max, high_n = cited["cluster_high"]
    gap_lo, gap_hi = cited["gap"]
    low = {f: r for f, r in ratios.items() if r < gap_lo}
    high = {f: r for f, r in ratios.items() if r > gap_hi}
    assert len(low) == low_n and len(high) == high_n, (
        f"the clusters now hold {len(low)} and {len(high)} cells; the prose says "
        f"{low_n} and {high_n}.{_cite('realistic_adjoint_speedup')}"
    )
    assert len(low) + len(high) == len(ratios), (
        f"a cell has landed in the empty band {gap_lo}x-{gap_hi}x between the two "
        f"clusters: {sorted(set(ratios) - set(low) - set(high))}. The prose "
        f"describes the spread as two clusters with nothing between them."
        f"{_cite('realistic_adjoint_speedup')}"
    )
    assert (round(min(low.values()), 3), round(max(low.values()), 3)) == (low_min, low_max)
    assert (round(min(high.values()), 3), round(max(high.values()), 3)) == (high_min, high_max)
    assert tuple(sorted(high)) == tuple(sorted(cited["high_cells"])), (
        f"the cells above {high_min}x are now {tuple(sorted(high))}; the prose "
        f"names {tuple(sorted(cited['high_cells']))}."
        f"{_cite('realistic_adjoint_speedup')}"
    )


def test_jax_is_faster_than_ducc0_in_every_realistic_comparison() -> None:
    """The universality the README's headline rests on, separately from the gap."""
    cited = CITATIONS["jax_faster_in_every_realistic_cell"].figures
    losses = [
        (op, fixture, ratio)
        for op in _REALISTIC_OPS
        for fixture, ratio in _speedups(op).items()
        if ratio <= 1.0
    ]
    n = sum(len(_speedups(op)) for op in _REALISTIC_OPS)
    assert n == cited["comparisons"], (
        f"{n} comparisons, prose counts {cited['comparisons']}."
        f"{_cite('jax_faster_in_every_realistic_cell')}"
    )
    assert not losses, (
        f"ducc0 is at least as fast in {losses}; the README says jax-nufft is "
        f"faster in every one.{_cite('jax_faster_in_every_realistic_cell')}"
    )


def test_ducc0_is_never_fastest_with_all_288_hardware_threads() -> None:
    """Why the README tells users to tune ``nthreads`` rather than max it.

    Deliberately two assertions with different strengths. That 288 never wins is
    exact and holds however coarse the grid is. The size of the penalty is only
    against the best count *measured*, and the grid differs per fixture, so it
    bounds the loss from below rather than locating the optimum.
    """
    cited = CITATIONS["ducc0_thread_optimum"].figures
    best_counts: dict[int, int] = defaultdict(int)
    penalties = []
    at_288_wins = []
    for op in _REALISTIC_OPS:
        stat = f"{op}_median_ms"
        for fixture, rows in _realistic_cells().items():
            ducc_rows = [r for r in rows if r["impl"] == "ducc0"]
            best = min(ducc_rows, key=lambda r: r[stat])
            at_288 = [r for r in ducc_rows if r["nthreads"] == cited["n_hardware_threads"]]
            assert at_288, f"{fixture}/{op} has no {cited['n_hardware_threads']}-thread row"
            best_counts[best["nthreads"]] += 1
            if best["nthreads"] == cited["n_hardware_threads"]:
                at_288_wins.append((fixture, op))
            penalties.append(at_288[0][stat] / best[stat])

    assert not at_288_wins, (
        f"{cited['n_hardware_threads']} threads is now the best measured setting "
        f"for {at_288_wins}; the README says it never is."
        f"{_cite('ducc0_thread_optimum')}"
    )
    got = (round(min(penalties), 3), round(max(penalties), 3))
    assert got == (cited["penalty_min"], cited["penalty_max"]), (
        f"running ducc0 at {cited['n_hardware_threads']} threads now costs "
        f"{got[0]}x to {got[1]}x its best measured setting; the prose says "
        f"{cited['penalty_min']}x to {cited['penalty_max']}x."
        f"{_cite('ducc0_thread_optimum')}"
    )
    assert (round(min(penalties), 1), round(max(penalties), 1)) == cited["penalty_rounded"]
    assert dict(best_counts) == {int(k): v for k, v in cited["best_counts"].items()}, (
        f"the winning thread counts are now {dict(sorted(best_counts.items()))}; "
        f"the citation records {cited['best_counts']}."
        f"{_cite('ducc0_thread_optimum')}"
    )


def test_the_realistic_problem_sizes_reproduce_from_the_stated_sizing_rule() -> None:
    """The sizes are derived, not chosen -- so recompute them from the rule.

    This is the guard against the benchmark quietly drifting to whatever size
    happened to be convenient: both dimensions follow from published instrument
    parameters, and if a future re-measurement changes a size without changing
    the rule, this fails.
    """
    cited = CITATIONS["realistic_sizing_rule"].figures
    payload = _load(_REALISTIC)
    sizing = payload["sizing"]
    c_m_s = 299_792_458.0

    def next_even_5_smooth(x: float) -> int:
        n = math.ceil(x)
        if n % 2:
            n += 1
        while True:
            m = n
            for p in (2, 3, 5):
                while m % p == 0:
                    m //= p
            if m == 1:
                return n
            n += 2

    rows = {r["fixture"]: r for r in payload["rows"] if r["suite"] == "realistic"}
    for name, params in sizing["telescopes"].items():
        lam = c_m_s / params["freq_hz"]
        pixsize = lam / (3.0 * params["max_baseline_m"])
        n_pix = next_even_5_smooth(math.radians(params["fov_deg"]) / pixsize)
        n_ant = params["n_ant"]
        n_rows = 150 * n_ant * (n_ant - 1) // 2

        assert n_pix == cited["n_pix"][name], (
            f"the sizing rule gives n_pix = {n_pix} for {name}; the citation "
            f"records {cited['n_pix'][name]}.{_cite('realistic_sizing_rule')}"
        )
        assert n_rows == cited["n_rows"][name], (
            f"the sizing rule gives n_rows = {n_rows} for {name}; the citation "
            f"records {cited['n_rows'][name]}.{_cite('realistic_sizing_rule')}"
        )
        # ... and that the rows actually measured were the size the rule asks for.
        for pointing in ("zenith", "off30"):
            row = rows.get(f"{name}_{pointing}")
            assert row is not None, f"no realistic row for {name}_{pointing}"
            assert (row["n_pix"], row["n_rows"]) == (n_pix, n_rows), (
                f"{name}_{pointing} was measured at n_pix={row['n_pix']}, "
                f"n_rows={row['n_rows']}, but the rule asks for {n_pix} and "
                f"{n_rows}.{_cite('realistic_sizing_rule')}"
            )


# --------------------------------------------------------------------------
# v0.2.0: memory (docs/benchmarks/v0.2.0-memory-gh200.json)
# --------------------------------------------------------------------------


def _memory_ratios() -> dict[tuple[str, str], float]:
    """jax-nufft peak device HBM over ducc0's interpreter-corrected peak RSS."""
    rows = _load(_MEMORY)["rows"]
    keyed = {(r["fixture"], r["op"], r["impl"]): r for r in rows}
    out = {}
    for (fixture, op, impl), row in keyed.items():
        if impl != "jax-nufft":
            continue
        ducc = keyed[(fixture, op, "ducc0")]
        out[(fixture, op)] = row["peak_mb"] / ducc["working_set_mb"]
    return out


def test_jax_needs_more_memory_than_ducc0_in_every_cell_and_by_how_much() -> None:
    """The cost side of the trade the README states next to the speedups."""
    cited = CITATIONS["memory_vs_ducc0"].figures
    ratios = _memory_ratios()
    assert len(ratios) == cited["cells"], (
        f"{len(ratios)} memory cells, prose counts {cited['cells']}.{_cite('memory_vs_ducc0')}"
    )
    lighter = {k: v for k, v in ratios.items() if v <= 1.0}
    assert not lighter, (
        f"jax-nufft is no heavier than ducc0 in {sorted(lighter)}; the README "
        f"says it is heavier in every cell.{_cite('memory_vs_ducc0')}"
    )
    got = (
        round(min(ratios.values()), 1),
        round(max(ratios.values()), 1),
        round(statistics.median(ratios.values()), 1),
    )
    assert got == (cited["min"], cited["max"], cited["median"]), (
        f"the memory ratio now spans {got[0]}x to {got[1]}x (median {got[2]}x); "
        f"the prose says {cited['min']}x to {cited['max']}x (median "
        f"{cited['median']}x).{_cite('memory_vs_ducc0')}"
    )
    assert min(ratios, key=lambda k: ratios[k]) == tuple(cited["min_cell"])
    assert max(ratios, key=lambda k: ratios[k]) == tuple(cited["max_cell"]), (
        "the heaviest cell has moved; the README names "
        f"{tuple(cited['max_cell'])}.{_cite('memory_vs_ducc0')}"
    )


def test_the_w_chunk_dial_trades_memory_for_time_monotonically() -> None:
    """``w_chunk`` is documented as a dial, which requires it to behave like one.

    Two separate properties. The quoted waypoints are what the README's table
    prints. Monotonicity is the stronger claim and the one that makes the table
    advice rather than trivia: sorting the strategies by scratch must sort them
    by time the other way, with no inversion, or "pick your point on the curve"
    is not sound guidance.
    """
    cited = CITATIONS["w_chunk_dial"].figures
    sweep = [
        r
        for r in _load(_MEMORY)["w_chunk_sweep"]["rows"]
        if r["fixture"] == cited["fixture"] and r["op"] == cited["op"]
    ]
    assert sweep, f"no w_chunk sweep rows for {cited['fixture']}/{cited['op']}"
    by_strategy = {r["w_strategy"]: r for r in sweep}
    baseline = by_strategy["dense_vmap"]

    for strategy, gb in cited["gb"].items():
        got = round(by_strategy[strategy]["temp_mb"] / 1024.0, 1)
        assert got == gb, (
            f"{strategy} now needs {got} GB of scratch; the README's table says "
            f"{gb} GB.{_cite('w_chunk_dial')}"
        )
    for strategy, ratio in cited["time_ratio"].items():
        got = round(by_strategy[strategy]["median_ms"] / baseline["median_ms"], 2)
        assert got == ratio, (
            f"{strategy} now runs at {got}x the dense_vmap time; the README's "
            f"table says {ratio}x.{_cite('w_chunk_dial')}"
        )

    ordered = sorted(sweep, key=lambda r: r["temp_mb"])
    times = [r["median_ms"] for r in ordered]
    inversions = [
        (ordered[i]["w_strategy"], ordered[i + 1]["w_strategy"])
        for i in range(len(times) - 1)
        if times[i] < times[i + 1]
    ]
    assert not inversions and cited["monotone"], (
        f"the memory/time curve is not monotone: {inversions} each use more "
        f"scratch than their predecessor and are also faster, so 'less memory "
        f"costs more time' does not hold as stated.{_cite('w_chunk_dial')}"
    )


def test_every_citation_is_recomputed_by_a_case_in_this_module() -> None:
    """Naming a file in CITATIONS is not the same as checking anything in it.

    ``test_every_committed_benchmark_json_is_covered_by_a_case`` is satisfied by
    a ``Citation`` that merely points at a file, so on its own it would let a new
    sweep be "covered" by an entry no assertion ever reads. Every key must reach
    a ``_cite(...)`` call, which only appears in failure messages of cases that
    assert something.
    """
    source = Path(__file__).read_text()
    unreferenced = sorted(
        k for k in CITATIONS if f"_cite('{k}')" not in source and f'_cite("{k}")' not in source
    )
    assert not unreferenced, (
        f"{unreferenced} are entries of CITATIONS that no case in this module "
        "cites in a failure message, so nothing recomputes them. Add a case, or "
        "drop the entry."
    )
