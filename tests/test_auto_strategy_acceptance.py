"""Part 6.4: GH200 baseline acceptance test for the GPU auto-heuristic.

The v0.1.2 plan's Part 6 acceptance criterion: on GH200, the strategy
picked by ``_auto_w_strategy`` must be within 15% of the best measured
strategy for the same (operator, telescope) cell.

This test encodes that against the committed baseline JSON
(``docs/benchmarks/v0.1.2-baseline-gpu.json``). It is pure-Python -- it
reads measured medians from the JSON and replays the heuristic on the
per-row plan parameters; it does NOT run any kernels, so it is fast and
runs on any host (no GPU required).

The heuristic resolves ``w_strategy`` only; ``channel_strategy`` is an
independent user choice. So for each cell we compare the *best channel
variant* of the auto-picked ``w_strategy`` against the *global best*
across all (w_strategy, channel_strategy) rows. This answers the
question the heuristic is responsible for: "did it pick a w_strategy
that, at its best channel setting, lands within 15% of optimal?".
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from jax_nufft.wgridder import _auto_w_strategy_gpu

_BASELINE = Path(__file__).resolve().parent.parent / "docs/benchmarks/v0.1.2-baseline-gpu.json"
_ACCEPTANCE_FACTOR = 1.15


def _load_payload() -> dict:
    if not _BASELINE.exists():
        pytest.skip(f"GH200 baseline JSON not present at {_BASELINE}")
    return json.loads(_BASELINE.read_text())


def _cells() -> list[tuple[str, str, list[dict]]]:
    """Group baseline rows by (op, fixture). Returns a list of
    ``(op, fixture, rows)`` so the test can parametrise over cells."""
    payload = _load_payload()
    if payload["fingerprint"]["jax_default_platform"] != "gpu":
        pytest.skip("baseline JSON was not captured on a GPU backend")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in payload["rows"]:
        groups[(row["op"], row["fixture"])].append(row)
    return [(op, fix, rows) for (op, fix), rows in sorted(groups.items())]


def _plan_stub_from_row(row: dict) -> SimpleNamespace:
    """Reconstruct the heuristic's plan-input view from a baseline row."""
    return SimpleNamespace(
        n_w=row["n_w"],
        w_kernel_width=row["w_kernel_width"],
        window_padding_overhead=row["window_padding_overhead"],
        n_rows=row["n_rows"],
    )


_CELLS = _cells()


@pytest.mark.parametrize(
    "op,fixture,rows",
    _CELLS,
    ids=[f"{op}-{fix}" for op, fix, _ in _CELLS],
)
def test_gpu_auto_within_15pct_of_best(op: str, fixture: str, rows: list[dict]) -> None:
    is_adjoint = op == "vis2dirty"
    plan = _plan_stub_from_row(rows[0])
    picked = _auto_w_strategy_gpu(plan, is_adjoint=is_adjoint)

    # Best median achievable with the auto-picked w_strategy (over the
    # two channel_strategy variants), vs the global best in this cell.
    picked_rows = [r for r in rows if r["w_strategy"] == picked]
    assert picked_rows, f"auto picked {picked!r} but no such row exists for {op}/{fixture}"
    auto_best = min(r["median_s"] for r in picked_rows)
    global_best = min(r["median_s"] for r in rows)
    best_row = min(rows, key=lambda r: r["median_s"])

    ratio = auto_best / global_best
    assert ratio <= _ACCEPTANCE_FACTOR, (
        f"{op}/{fixture}: auto picked {picked!r} "
        f"({auto_best * 1e3:.2f} ms) but best is "
        f"{best_row['w_strategy']!r}/{best_row['channel_strategy']} "
        f"({global_best * 1e3:.2f} ms) -- {ratio:.2f}x, over the "
        f"{_ACCEPTANCE_FACTOR:.2f}x acceptance bar."
    )


def test_gpu_auto_never_picks_a_scan_variant() -> None:
    """Belt-and-braces: on the GH200 baseline the scan variants are
    5-30x slower than vmap, so the GPU heuristic must never resolve to
    a scan strategy on any cell. A scan pick would be a >5x regression
    that the 15% check above would also catch, but this gives a
    clearer failure message."""
    for op, fixture, rows in _CELLS:
        is_adjoint = op == "vis2dirty"
        picked = _auto_w_strategy_gpu(_plan_stub_from_row(rows[0]), is_adjoint=is_adjoint)
        assert picked in ("dense_vmap", "windowed_vmap"), (
            f"{op}/{fixture}: GPU heuristic picked scan variant {picked!r}"
        )
