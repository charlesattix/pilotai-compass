#!/usr/bin/env python3
"""
IBIT Aggressive Parameter Sweep — Iron Condors, Short-DTE, Multi-Position, Kelly.

Grid:
  DTE          : [0, 1, 3]
  OTM          : [0.03, 0.05]
  Spread width : [3, 5]
  Direction    : iron_condor (both sides)
  Concurrent   : [5, 8, 10]
  Risk pct     : [0.20, 0.30, 0.40]
  Kelly frac   : [0.75, 1.0]
  Profit target: [0.30, 0.50, 0.65]
  Stop loss    : [1.5, 2.0, 3.0]x
  → 1,944 combos × 3 runs each

Data:
  Full : 2024-11-19 → 2026-03-20  (options data start)
  Train: 2024-11-19 → 2025-03-31  (~4.5 months)
  Test : 2025-04-01 → 2026-03-20  (~12 months)

Gate 2:
  avg_ann_return ≥ 50%
  max_drawdown   > -30%   (i.e. better than -30%)
  overfit_score  ≥ 0.60   (test_ann / train_ann)
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.ibit_backtester import IBITBacktester

# ─────────────────────────────────────────────────────────────────────────────
# Sweep grid
# ─────────────────────────────────────────────────────────────────────────────

DTE_VALUES        = [0, 1, 3]
OTM_VALUES        = [0.03, 0.05]
WIDTH_VALUES      = [3.0, 5.0]
CONCURRENT_VALUES = [5, 8, 10]
RISK_VALUES       = [0.20, 0.30, 0.40]
KELLY_VALUES      = [0.75, 1.0]
PT_VALUES         = [0.30, 0.50, 0.65]   # profit_target fraction
SL_VALUES         = [1.5, 2.0, 3.0]      # stop_loss_mult

# Data windows (options data starts 2024-11-19)
FULL_START  = "2024-11-19"
FULL_END    = "2026-03-20"
TRAIN_START = "2024-11-19"
TRAIN_END   = "2025-03-31"
TEST_START  = "2025-04-01"
TEST_END    = "2026-03-20"

# Gate 2 thresholds
GATE2_MIN_ANN    = 50.0    # % annualized
GATE2_MAX_DD     = -30.0   # % (must be >= this)
GATE2_MIN_OVERFIT = 0.60

# Output
OUTPUT_DIR        = ROOT / "output"
LEADERBOARD_FILE  = OUTPUT_DIR / "mega_sweep_aggressive_leaderboard.json"
STATE_FILE        = OUTPUT_DIR / "mega_sweep_aggressive_state.json"
LOG_FILE          = OUTPUT_DIR / "mega_sweep_aggressive.log"

MIN_CREDIT_PCT = 3.0   # aggressive threshold — more trades

# ─────────────────────────────────────────────────────────────────────────────
# Overfit score
# ─────────────────────────────────────────────────────────────────────────────

def overfit_score(train_ann: float, test_ann: float) -> float:
    """test_ann / train_ann clamped to [-1, 2]. 0 when train <= 0."""
    if abs(train_ann) < 0.01:
        return 0.0
    score = test_ann / train_ann
    return round(max(-1.0, min(2.0, score)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Build combos
# ─────────────────────────────────────────────────────────────────────────────

def build_combos():
    combos = []
    for dte, otm, width, conc, risk, kelly, pt, sl in itertools.product(
        DTE_VALUES, OTM_VALUES, WIDTH_VALUES,
        CONCURRENT_VALUES, RISK_VALUES, KELLY_VALUES,
        PT_VALUES, SL_VALUES,
    ):
        dte_min = 0 if dte <= 1 else max(0, dte - 2)
        dte_max = max(dte + 5, 7)   # catch at least a few expirations around target
        combos.append({
            "direction":        "iron_condor",
            "compound":         True,
            "otm_pct":          otm,
            "call_otm_pct":     None,   # symmetric
            "spread_width":     width,
            "call_spread_width": None,  # symmetric
            "min_credit_pct":   MIN_CREDIT_PCT,
            "dte_target":       dte,
            "dte_min":          dte_min,
            "dte_max":          dte_max,
            "profit_target":    pt,
            "stop_loss_mult":   sl,
            "risk_pct":         risk,
            "max_contracts":    500,
            "max_concurrent":   conc,
            "kelly_fraction":   kelly,
            "kelly_min_trades": 5,      # faster Kelly warmup for short-dated
            "same_day_reentry": True,   # aggressive: re-enter same day
            "regime_filter":    "none",
        })
    return combos


ALL_COMBOS = build_combos()
TOTAL      = len(ALL_COMBOS)


# ─────────────────────────────────────────────────────────────────────────────
# Single combo runner
# ─────────────────────────────────────────────────────────────────────────────

def run_combo(cfg: dict) -> dict:
    """Run full + train + test. Returns merged result dict."""
    full_cfg  = {**cfg, "starting_capital": 100_000.0}
    train_cfg = {**cfg, "starting_capital": 100_000.0}
    test_cfg  = {**cfg, "starting_capital": 100_000.0}

    full  = IBITBacktester(config=full_cfg).run(FULL_START,  FULL_END)
    train = IBITBacktester(config=train_cfg).run(TRAIN_START, TRAIN_END)
    test  = IBITBacktester(config=test_cfg).run(TEST_START,  TEST_END)

    ov = overfit_score(train["ann_return"], test["ann_return"])

    passes = (
        full["ann_return"]   >= GATE2_MIN_ANN
        and full["max_drawdown"] >= GATE2_MAX_DD
        and ov               >= GATE2_MIN_OVERFIT
        and not full["ruin"]
    )

    return {
        # Identity
        "dte":           cfg["dte_target"],
        "otm_pct":       cfg["otm_pct"],
        "spread_width":  cfg["spread_width"],
        "max_concurrent": cfg["max_concurrent"],
        "risk_pct":      cfg["risk_pct"],
        "kelly_fraction": cfg["kelly_fraction"],
        "profit_target": cfg["profit_target"],
        "stop_loss_mult": cfg["stop_loss_mult"],
        # Full period
        "ann_return":    round(full["ann_return"],    2),
        "return_pct":    round(full["return_pct"],    2),
        "max_drawdown":  round(full["max_drawdown"],  2),
        "total_trades":  full["total_trades"],
        "win_rate":      round(full["win_rate"],       2),
        "profit_factor": round(full["profit_factor"], 4),
        "ending_capital": round(full["ending_capital"], 0),
        # Train / test
        "train_ann":     round(train["ann_return"],   2),
        "train_return":  round(train["return_pct"],   2),
        "train_trades":  train["total_trades"],
        "test_ann":      round(test["ann_return"],    2),
        "test_return":   round(test["return_pct"],    2),
        "test_trades":   test["total_trades"],
        "overfit_score": ov,
        # Gate verdict
        "passes_gate2":  passes,
        "ruin":          bool(full["ruin"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# State / leaderboard I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_completed_index": -1, "total_runs": 0, "gate2_passes": 0}


def save_state(state: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_leaderboard() -> list:
    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE) as f:
            return json.load(f)
    return []


def save_leaderboard(entries: list) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(entries, key=lambda x: x.get("ann_return", -999), reverse=True)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(ranked, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="a"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    log = logging.getLogger("aggressive_sweep")

    state       = load_state()
    leaderboard = load_leaderboard()

    resume_from  = state["last_completed_index"] + 1
    gate2_passes = state.get("gate2_passes", 0)
    total_runs   = state.get("total_runs", 0)

    log.info("=" * 65)
    log.info("IBIT Aggressive Sweep  (%d combos total)", TOTAL)
    log.info("Full : %s → %s", FULL_START, FULL_END)
    log.info("Train: %s → %s  |  Test: %s → %s", TRAIN_START, TRAIN_END, TEST_START, TEST_END)
    log.info("Gate2: ann≥%.0f%%  DD>%.0f%%  overfit≥%.2f", GATE2_MIN_ANN, GATE2_MAX_DD, GATE2_MIN_OVERFIT)
    log.info("Resuming from index %d  |  leaderboard: %d entries", resume_from, len(leaderboard))
    log.info("=" * 65)

    t0 = time.time()

    for idx in range(resume_from, TOTAL):
        cfg     = ALL_COMBOS[idx]
        t_combo = time.time()

        try:
            result = run_combo(cfg)
        except Exception as exc:
            log.error("FAILED idx=%d  %s", idx, exc)
            result = {
                "combo_index": idx,
                "dte": cfg["dte_target"],
                "otm_pct": cfg["otm_pct"],
                "spread_width": cfg["spread_width"],
                "max_concurrent": cfg["max_concurrent"],
                "risk_pct": cfg["risk_pct"],
                "kelly_fraction": cfg["kelly_fraction"],
                "profit_target": cfg["profit_target"],
                "stop_loss_mult": cfg["stop_loss_mult"],
                "error": str(exc),
                "ann_return": -999,
                "passes_gate2": False,
            }

        result["combo_index"]     = idx
        result["run_timestamp"]   = datetime.utcnow().isoformat()
        result["elapsed_s"]       = round(time.time() - t_combo, 3)

        leaderboard.append(result)
        total_runs += 1

        if result.get("passes_gate2"):
            gate2_passes += 1
            log.info(
                "★ GATE2 #%d  idx=%d  DTE=%d OTM=%.2f W=%.0f C=%d R=%.0f%% K=%.2f "
                "PT=%.0f%% SL=%.1f  ann=%.1f%% dd=%.1f%% ov=%.2f",
                gate2_passes, idx,
                result["dte"], result["otm_pct"], result["spread_width"],
                result["max_concurrent"], result["risk_pct"] * 100,
                result["kelly_fraction"], result["profit_target"] * 100,
                result["stop_loss_mult"],
                result.get("ann_return", 0),
                result.get("max_drawdown", 0),
                result.get("overfit_score", 0),
            )

        combos_done = idx - resume_from + 1
        if combos_done % 50 == 0 or idx == TOTAL - 1:
            elapsed  = time.time() - t0
            remaining = TOTAL - idx - 1
            rate     = combos_done / elapsed if elapsed > 0 else 1
            eta_min  = remaining / rate / 60 if rate > 0 else 0

            log.info(
                "[%d/%d] idx=%d | gate2=%d | %.2f/s | ETA %.1f min",
                combos_done, TOTAL, idx, gate2_passes, rate, eta_min,
            )
            save_state({
                "last_completed_index": idx,
                "total_runs":   total_runs,
                "gate2_passes": gate2_passes,
                "elapsed_sec":  round(elapsed, 1),
                "eta_min":      round(eta_min, 1),
                "timestamp":    datetime.utcnow().isoformat(),
                "complete":     idx == TOTAL - 1,
            })
            save_leaderboard(leaderboard)

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    log.info("=" * 65)
    log.info("SWEEP COMPLETE  —  %d combos  in %.1f s  (%.2f/s)",
             total_runs - state.get("total_runs", 0),
             elapsed_total,
             (total_runs - state.get("total_runs", 0)) / elapsed_total if elapsed_total > 0 else 0)
    log.info("Gate2 passes: %d / %d  (%.1f%%)",
             gate2_passes, total_runs,
             gate2_passes / total_runs * 100 if total_runs else 0)
    log.info("Leaderboard → %s", LEADERBOARD_FILE)

    top = sorted(
        [e for e in leaderboard if "error" not in e and not e.get("ruin")],
        key=lambda x: x.get("ann_return", -999),
        reverse=True,
    )[:20]

    if top:
        log.info("")
        log.info("TOP 20 by annualized return:")
        log.info("  %-3s  %-3s  %-5s  %-4s  %-4s  %-5s  %-5s  %-5s  %-5s  %-8s  %-8s  %-8s  %-5s  G2",
                 "DTE", "OTM", "Width", "Con", "Risk", "Kelly", "PT%", "SL",
                 "Trd", "ann%", "dd%", "overfit", "trainA")
        for rank, e in enumerate(top, 1):
            log.info(
                "  #%-3d DTE=%-2d OTM=%.2f W=%.0f C=%-2d R=%.0f%% K=%.2f PT=%.0f%% SL=%.1f "
                "T=%-4d %+7.1f%% %6.1f%% ov=%.2f tr=%.1f%%  %s",
                rank,
                e.get("dte", 0), e.get("otm_pct", 0), e.get("spread_width", 0),
                e.get("max_concurrent", 0), e.get("risk_pct", 0) * 100,
                e.get("kelly_fraction", 0), e.get("profit_target", 0) * 100,
                e.get("stop_loss_mult", 0),
                e.get("total_trades", 0),
                e.get("ann_return", 0),
                e.get("max_drawdown", 0),
                e.get("overfit_score", 0),
                e.get("train_ann", 0),
                "✓" if e.get("passes_gate2") else "✗",
            )


if __name__ == "__main__":
    main()
