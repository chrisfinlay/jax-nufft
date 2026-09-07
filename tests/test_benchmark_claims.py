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
recomputed from anything in the tree, and are recorded rather than proxied:

* Issue #46's GH200-versus-ducc0 table ("1.4-5.6x slower than ducc0 on five of
  the table's six cells", "1.4-6.3x faster in all six", the "about 1.16x
  faster" exception) -- quoted in ``README.md``'s strategy section,
  ``AGENTS.md`` sec 5, the ``dirty2vis`` / ``vis2dirty`` docstrings and
  ``tests/test_default_w_strategy.py``. Those timings exist only as the
  markdown table in ``README.md``; no JSON was committed for them. Recomputing
  them from that table (by hand, 2026-09) reproduces every one of the four
  figures, but a test would be asserting the README against itself.
* The post-#43 ``window_padding_overhead`` figures in ``wgridder.py``'s
  ``_GPU_PADDING_CUTOFF`` comment. ``docs/benchmarks/README.md`` records that
  every committed JSON carries the *pre*-#43 scale and that the conversion
  needs plan-time locals that were never stored, so the new-scale numbers
  cannot come from the tree at all.
* ``README.md``'s "Indicative numbers" tables and every Apple M-series timing
  in ``AGENTS.md`` sec 5 / sec 9, including the #24 ``nthreads`` ranges. The
  committed CPU JSON is aarch64 Grace/GH200 from 2026-05-18; those tables are a
  different machine and a different date, with no JSON behind them.

Pure-Python and fast: it reads JSON and replays the host-side heuristic, and
runs no kernels.
"""

from __future__ import annotations

import json
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
        figures={"min": 1.448, "max": 32.660, "median": 6.096},
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
            "range'."
        ),
        sites=("AGENTS.md sec 9 (Part 6 correction)",),
        source=_GPU_BASELINE,
        figures={"low": 5.0, "high": 30.0},
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
            "is always inside the 15% acceptance bar."
        ),
        sites=(
            "src/jax_nufft/wgridder.py (_auto_w_strategy_gpu docstring)",
            "tests/test_default_w_strategy.py (assertion message)",
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
        figures={"cells": 16},
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
    group -- 2 x 2 per group, 40 groups. This is the *only* one of the four
    obvious pairings that reproduces the quoted figures; the alternatives were
    measured while writing this module and give (n, min, max, median, count
    below 5x) of (80, 1.4546, 32.6601, 5.4874, 36) pairing within a family at
    matched channel strategy, (160, 1.4539, 33.0505, 5.4827, 72) pairing within
    a family across channel strategies, and (320, 1.4477, 33.0505, 6.0960, 144)
    for the full per-cell cross product. Only this one gives
    (160, 1.4480, 32.6601, 6.0960, 72).
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
    assert (round(min(ratios), 2), round(max(ratios), 1), round(statistics.median(ratios), 1)) == (
        1.45,
        32.7,
        6.1,
    ), (
        "the one-decimal restatement '1.45x to 32.7x (median 6.1x)' no longer "
        f"rounds from the data.{_cite('pair_spread')}"
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

    Each is checked on its own, which is how the sentence enumerates them. A
    subset "yields a 5-30x range" only if its endpoints are the quoted ones, so
    what is asserted is that neither endpoint lands on 5 or 30 -- not merely
    that some value escapes the interval, which is a weaker and less useful
    statement.

    Measured while writing this (2026-09): conjoining all three named
    restrictions at once *does* give a population lying inside [5, 30]
    (10.376x to 29.393x over 8 cells). Its endpoints are still not 5 and 30, so
    the sentence holds; AGENTS.md now says "taken on its own" so the stronger
    reading is not left standing.
    """
    cited = CITATIONS["no_subset_is_5_to_30x"].figures
    all_pairs = _scan_vmap_pairs()

    def by_group(rows: list[dict]) -> list[float]:
        groups: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
            lambda: {"scan": [], "vmap": []}
        )
        for row in rows:
            key = (row["op"], row["fixture"], row["channel_strategy"])
            groups[key][_w_family(row["w_strategy"])].append(row["median_s"])
        return [s / v for g in groups.values() for s, v in product(g["scan"], g["vmap"])]

    rows = _gpu_rows()
    best_of_family: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"scan": [], "vmap": []}
    )
    for row in rows:
        best_of_family[(row["op"], row["fixture"])][_w_family(row["w_strategy"])].append(
            row["median_s"]
        )

    subsets = {
        "the whole sweep": [r for _, _, _, r in all_pairs],
        "off-zenith only": by_group([r for r in rows if r["fixture"].endswith("off30")]),
        "excluding GH200_large": by_group(
            [r for r in rows if not r["fixture"].startswith("GH200_large")]
        ),
        "best-of-family per cell": [
            min(g["scan"]) / min(g["vmap"]) for g in best_of_family.values()
        ],
    }
    for name, ratios in subsets.items():
        lo, hi = round(min(ratios), 1), round(max(ratios), 1)
        assert (lo, hi) != (cited["low"], cited["high"]), (
            f"the subset {name!r} now spans {lo}x to {hi}x, which *is* the "
            f"'5-30x' the sweep was originally written up with. AGENTS.md sec 9 "
            f"says no named subset reproduces it.{_cite('no_subset_is_5_to_30x')}"
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
    """
    cited = CITATIONS["runner_up_gap"].figures
    per_cell = []
    per_slice = []
    for rows in _gpu_cells().values():
        best_by_w: dict[str, float] = {}
        for row in rows:
            w = row["w_strategy"]
            best_by_w[w] = min(best_by_w.get(w, row["median_s"]), row["median_s"])
        ordered = sorted(best_by_w.values())
        per_cell.append(ordered[1] / ordered[0])
        for channel in ("scan", "vmap"):
            sub = sorted(r["median_s"] for r in rows if r["channel_strategy"] == channel)
            per_slice.append(sub[1] / sub[0])

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


def test_the_auto_pick_meets_the_fifteen_percent_bar_by_being_the_best() -> None:
    """The acceptance bar README.md and AGENTS.md quote, plus its slack.

    ``tests/test_auto_strategy_acceptance.py`` already asserts the bar. What is
    asserted here is the sentence wgridder.py adds about it: the bar is met
    "trivially ... by picking the best one every time", i.e. the worst cell's
    ratio is 1.00x and not merely under 1.15x. If a retune ever spends real
    slack against that bar, this is the case that says so.
    """
    cited = CITATIONS["acceptance_bar"].figures
    worst = 0.0
    worst_cell = None
    for (op, fixture), rows in sorted(_gpu_cells().items()):
        plan = SimpleNamespace(
            n_w=rows[0]["n_w"],
            w_kernel_width=rows[0]["w_kernel_width"],
            window_padding_overhead=rows[0]["window_padding_overhead"],
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
    assert round(worst, 2) == cited["actual_worst"], (
        f"the heuristic no longer picks the outright best strategy in every "
        f"cell: worst is {worst_cell} at {worst:.3f}x. The prose reads the "
        f"{cited['bar']}x bar as met trivially; at {worst:.3f}x it is not."
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
    """
    cited = CITATIONS["plan_dense_vmap_sweeps_cpu_json"].figures
    stats = _cpu_rows(path)
    losses = []
    cells = 0
    for fixture, op in product(_CPU_FIXTURES, _OPS):
        times = {w: stats[("jax", op, fixture, w)][stat] for w in _W_STRATEGIES}
        cells += 1
        winner = min(times, key=lambda w: times[w])
        if winner != "dense_vmap":
            losses.append((op, fixture, winner))
    assert cells == cited["cells"], f"{label}: {cells} cells, not {cited['cells']}"
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
    times = {
        (op, w): stats[("jax", op, "MWA_extended_off30", w)]["median"]
        for op, w in product(_OPS, ("dense_scan", "windowed_scan"))
    }
    lo, hi = min(times.values()), max(times.values())
    assert (round(lo, 1), round(hi, 1)) == (cited["low_s"], cited["high_s"]), (
        f"the MWA_extended off30 scan-family worst case now spans {lo:.3f}-"
        f"{hi:.3f} s, which does not round to the cited {cited['low_s']}-"
        f"{cited['high_s']} s: { {k: round(v, 4) for k, v in times.items()} }"
        f"{_cite('plan_long_scan_worst_case')}"
    )


def _part1_best_over_ducc() -> dict[tuple[str, str], float]:
    """Part 1's ``best/ducc`` column: ducc mean over the best jax mean."""
    stats = _cpu_rows(_CPU_PART1)
    out = {}
    for fixture, op in product(_CPU_FIXTURES, _OPS):
        best = min(stats[("jax", op, fixture, w)]["mean"] for w in _W_STRATEGIES)
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
    """Part 1 median over baseline median, as a percentage change per cell."""
    base = _cpu_rows(_CPU_BASELINE)
    part1 = _cpu_rows(_CPU_PART1)
    return {
        (op, fixture): 100.0
        * (
            part1[("jax", op, fixture, w_strategy)]["median"]
            / base[("jax", op, fixture, w_strategy)]["median"]
            - 1.0
        )
        for fixture, op in product(_CPU_FIXTURES, _OPS)
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

    This is the one guard against the failure mode the module cannot otherwise
    see: someone commits a fresh sweep, quotes an aggregate of it in a comment,
    and nothing here notices because no case names that file.
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
