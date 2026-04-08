#!/usr/bin/env python3
"""
run_mega_sweep.py — Massive IBIT credit spread parameter sweep.

Grid: 7×4×4×4×4×2×3×3×3 = 96,768 combinations
Train: 2024-04-01 → 2025-03-31  (IBIT options live 2024-11-19, so ~4mo effective)
Test:  2025-04-01 → 2026-03-31  (~12mo)

Gate 2:
  - avg annualized return ≥ 50%
  - max drawdown (worst of train/test) < 30% (i.e. > -30%)
  - overfit score ≥ 0.60  (test_ann / train_ann)

Output:
  output/mega_sweep_leaderboard.json  — all completed results, sorted by composite score
  output/mega_sweep_state.json        — checkpoint every 100 combos

Usage:
    python3 scripts/run_mega_sweep.py
    python3 scripts/run_mega_sweep.py --workers 12
    python3 scripts/run_mega_sweep.py --resume        # skip already-done combos
    python3 scripts/run_mega_sweep.py --dry-run       # print grid size and exit
    python3 scripts/run_mega_sweep.py --top 20        # show top 20 after run
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR     = ROOT / "output"
LEADERBOARD    = OUTPUT_DIR / "mega_sweep_leaderboard.json"
STATE_PATH     = OUTPUT_DIR / "mega_sweep_state.json"

TRAIN_START = "2024-04-01"
TRAIN_END   = "2025-03-31"
TEST_START  = "2025-04-01"
TEST_END    = "2026-03-31"

# Gate 2 thresholds
GATE2_AVG_ANN = 50.0    # %
GATE2_MAX_DD  = -30.0   # max drawdown must be > this (less negative)
GATE2_OVERFIT = 0.60    # test_ann / train_ann

CHECKPOINT_EVERY = 100  # save state every N completed combos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid
# ─────────────────────────────────────────────────────────────────────────────

GRID = {
    "dte":            [0, 1, 3, 5, 7, 10, 14],
    "otm_pct":        [0.03, 0.05, 0.08, 0.10],
    "spread_width":   [3, 5, 7, 10],
    "profit_target":  [0.30, 0.50, 0.65, 0.80],
    "stop_loss_mult": [1.5, 2.0, 2.5, 3.0],
    "direction":      ["adaptive", "iron_condor"],
    "max_concurrent": [3, 5, 8],
    "risk_pct":       [0.15, 0.20, 0.30],
    "kelly_fraction": [0.5, 0.75, 1.0],
}


def build_all_combos() -> List[Dict[str, Any]]:
    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = []
    for i, vals in enumerate(itertools.product(*values)):
        combos.append({"_id": i, **dict(zip(keys, vals))})
    return combos


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def build_config(params: Dict[str, Any]) -> Dict[str, Any]:
    dte = int(params["dte"])
    return {
        "compound":        True,
        "direction":       params["direction"],
        "dte_target":      dte,
        "dte_min":         0 if dte == 0 else 1,
        "dte_max":         dte + 10 if dte > 0 else 1,
        "otm_pct":         float(params["otm_pct"]),
        "spread_width":    float(params["spread_width"]),
        "profit_target":   float(params["profit_target"]),
        "stop_loss_mult":  float(params["stop_loss_mult"]),
        "risk_pct":        float(params["risk_pct"]),
        "max_contracts":   100,
        "max_concurrent":  int(params["max_concurrent"]),
        "kelly_fraction":  float(params["kelly_fraction"]),
        "min_credit_pct":  5.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Worker (runs in subprocess — no shared state)
# ─────────────────────────────────────────────────────────────────────────────

def _run_combo(args: Tuple) -> Dict[str, Any]:
    """Run one parameter combo through train + test periods. Called in subprocess."""
    params, train_start, train_end, test_start, test_end = args

    # Import here so each subprocess gets its own copy
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backtest.ibit_backtester import IBITBacktester

    combo_id = params["_id"]
    p = {k: v for k, v in params.items() if k != "_id"}
    config = build_config(p)

    try:
        bt = IBITBacktester(config)
        tr = bt.run(train_start, train_end)
        te = bt.run(test_start, test_end)
    except Exception as exc:
        return {
            "id": combo_id,
            "params": p,
            "error": str(exc),
            "passes_gate2": False,
            "composite_score": -999.0,
        }

    train_ann = tr.get("ann_return", 0.0) or 0.0
    test_ann  = te.get("ann_return", 0.0) or 0.0
    train_dd  = tr.get("max_drawdown", 0.0) or 0.0
    test_dd   = te.get("max_drawdown", 0.0) or 0.0
    worst_dd  = min(train_dd, test_dd)
    avg_ann   = (train_ann + test_ann) / 2.0

    # Overfit score: clamp to [-1, 2]
    if train_ann > 1.0:   # avoid division by tiny numbers
        overfit = max(-1.0, min(2.0, test_ann / train_ann))
    elif train_ann > 0:
        overfit = max(-1.0, min(2.0, test_ann / max(train_ann, 1.0)))
    else:
        overfit = -1.0

    # Composite: reward high avg return that generalises (penalise overfit < 1)
    # composite = avg_ann × min(overfit, 1.0) — so overfit > 1.0 doesn't inflate score
    composite = avg_ann * min(max(overfit, 0.0), 1.0) if avg_ann > 0 else avg_ann

    passes = (
        avg_ann   >= GATE2_AVG_ANN and
        worst_dd  >  GATE2_MAX_DD  and
        overfit   >= GATE2_OVERFIT
    )

    return {
        "id":              combo_id,
        "params":          p,
        "train_ann":       round(train_ann, 2),
        "test_ann":        round(test_ann, 2),
        "avg_ann":         round(avg_ann, 2),
        "train_dd":        round(train_dd, 2),
        "test_dd":         round(test_dd, 2),
        "worst_dd":        round(worst_dd, 2),
        "train_trades":    tr.get("total_trades", 0),
        "test_trades":     te.get("total_trades", 0),
        "train_win_rate":  round(tr.get("win_rate", 0.0), 1),
        "test_win_rate":   round(te.get("win_rate", 0.0), 1),
        "overfit_score":   round(overfit, 3),
        "composite_score": round(composite, 2),
        "passes_gate2":    passes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_state() -> Tuple[List[Dict], set]:
    """Return (results_list, done_ids)."""
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                state = json.load(f)
            results = state.get("results", [])
            done_ids = {r["id"] for r in results}
            return results, done_ids
        except Exception:
            pass
    return [], set()


def _save_state(results: List[Dict], elapsed_s: float, total: int):
    tmp = str(STATE_PATH) + ".tmp"
    payload = {
        "saved_at":   datetime.now().isoformat(),
        "elapsed_s":  round(elapsed_s, 1),
        "total":      total,
        "done":       len(results),
        "gate2_hits": sum(1 for r in results if r.get("passes_gate2")),
        "results":    results,
    }
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, str(STATE_PATH))


def _save_leaderboard(results: List[Dict]):
    """Save leaderboard sorted by composite_score descending."""
    ranked = sorted(results, key=lambda r: r.get("composite_score", -999), reverse=True)
    tmp = str(LEADERBOARD) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ranked, f, indent=2)
    os.replace(tmp, str(LEADERBOARD))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IBIT mega parameter sweep")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker processes (default: 8)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed combos from state file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print grid size and exit")
    parser.add_argument("--top", type=int, default=0,
                        help="Print top N results after sweep and exit")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore existing state and start fresh")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_combos = build_all_combos()
    total = len(all_combos)

    log.info("=== IBIT Mega Parameter Sweep ===")
    log.info("Grid:      %d combinations", total)
    log.info("Train:     %s → %s", TRAIN_START, TRAIN_END)
    log.info("Test:      %s → %s", TEST_START, TEST_END)
    log.info("Workers:   %d", args.workers)
    log.info("Gate 2:    avg_ann≥%.0f%%  worst_dd>%.0f%%  overfit≥%.2f",
             GATE2_AVG_ANN, GATE2_MAX_DD, GATE2_OVERFIT)

    if args.dry_run:
        log.info("Dry run — exiting")
        for k, v in GRID.items():
            log.info("  %s: %d values", k, len(v))
        return

    # Load checkpoint
    results: List[Dict] = []
    done_ids: set = set()

    if not args.reset and (args.resume or STATE_PATH.exists()):
        results, done_ids = _load_state()
        if done_ids:
            log.info("Resuming: %d already done, %d remaining",
                     len(done_ids), total - len(done_ids))

    # Show top results mode
    if args.top > 0 and LEADERBOARD.exists():
        with open(LEADERBOARD) as f:
            ranked = json.load(f)
        log.info("=== TOP %d RESULTS ===", args.top)
        for i, r in enumerate(ranked[:args.top]):
            gate = "✓" if r.get("passes_gate2") else " "
            log.info(
                "[%d] %s id=%d avg_ann=%.1f%% worst_dd=%.1f%% overfit=%.2f "
                "comp=%.1f  train=%d/%d trades",
                i + 1, gate, r["id"],
                r.get("avg_ann", 0), r.get("worst_dd", 0),
                r.get("overfit_score", 0), r.get("composite_score", 0),
                r.get("train_trades", 0), r.get("test_trades", 0),
            )
            log.info("     params: %s", r.get("params", {}))
        return

    # Filter out done
    todo = [c for c in all_combos if c["_id"] not in done_ids]
    log.info("Starting:  %d combos to process", len(todo))

    if not todo:
        log.info("All combos done! See %s", LEADERBOARD)
        _save_leaderboard(results)
        return

    # Build task args
    task_args = [
        (combo, TRAIN_START, TRAIN_END, TEST_START, TEST_END)
        for combo in todo
    ]

    start_t = time.time()
    done_this_run = 0
    gate2_hits = sum(1 for r in results if r.get("passes_gate2"))
    batch: List[Dict] = []

    log.info("")
    log.info("Launching %d workers...", args.workers)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_combo, a): a[0]["_id"] for a in task_args}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            batch.append(result)
            done_this_run += 1

            if result.get("passes_gate2"):
                gate2_hits += 1

            total_done = len(done_ids) + done_this_run
            elapsed = time.time() - start_t
            rate = done_this_run / elapsed if elapsed > 0 else 0
            remaining = total - total_done
            eta_min = remaining / (rate * 60) if rate > 0 else 0

            # Progress every 100 combos
            if done_this_run % 100 == 0 or done_this_run == len(todo):
                best_so_far = max((r.get("composite_score", -999) for r in results), default=0)
                log.info(
                    "[%d/%d] %.1f%% | %.1f/s | ETA %.0fm | gate2=%d | best=%.1f%%",
                    total_done, total,
                    100.0 * total_done / total,
                    rate, eta_min,
                    gate2_hits, best_so_far,
                )

            # Checkpoint every CHECKPOINT_EVERY combos
            if len(batch) >= CHECKPOINT_EVERY:
                _save_state(results, time.time() - start_t, total)
                _save_leaderboard(results)
                batch.clear()

    # Final save
    _save_state(results, time.time() - start_t, total)
    _save_leaderboard(results)

    elapsed_min = (time.time() - start_t) / 60
    log.info("")
    log.info("=== SWEEP COMPLETE ===")
    log.info("Total combos:   %d", total)
    log.info("Gate 2 passes:  %d  (%.1f%%)", gate2_hits, 100 * gate2_hits / total)
    log.info("Elapsed:        %.1f min", elapsed_min)
    log.info("Leaderboard:    %s", LEADERBOARD)

    # Show top 10
    ranked = sorted(results, key=lambda r: r.get("composite_score", -999), reverse=True)
    log.info("")
    log.info("=== TOP 10 ===")
    for i, r in enumerate(ranked[:10]):
        gate = "✓ GATE2" if r.get("passes_gate2") else "      "
        log.info(
            "[%d] %s  avg_ann=%.1f%%  worst_dd=%.1f%%  overfit=%.2f  comp=%.1f",
            i + 1, gate,
            r.get("avg_ann", 0), r.get("worst_dd", 0),
            r.get("overfit_score", 0), r.get("composite_score", 0),
        )
        p = r.get("params", {})
        log.info(
            "     dte=%s  otm=%.0f%%  w=$%s  pt=%.0f%%  sl=%.1fx  dir=%s  "
            "conc=%s  risk=%.0f%%  kelly=%.2f",
            p.get("dte"), (p.get("otm_pct", 0) * 100),
            p.get("spread_width"), (p.get("profit_target", 0) * 100),
            p.get("stop_loss_mult"), p.get("direction"),
            p.get("max_concurrent"), (p.get("risk_pct", 0) * 100),
            p.get("kelly_fraction"),
        )


if __name__ == "__main__":
    main()
