#!/usr/bin/env python3
"""
IBIT Crypto Credit Spread Parameter Sweep — SECOND HALF (combos 800-1599).

Maps IBIT sweep params → BTCCreditSpreadBacktester (Deribit BTC real data 2020-2024).
Saves progress every 50 combos so the run is fully recoverable.

Gate 2 criteria:
  avg_annual_return >= 12%
  max_drawdown > -15%
  overfit_score >= 0.70  (test_return / train_return)

Train period : 2020-2022
Test period  : 2023-2024
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.crypto_param_sweep import FULL_SWEEP
from backtest.btc_credit_spread_backtester import BTCCreditSpreadBacktester

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_YEARS: List[int] = [2020, 2021, 2022]
TEST_YEARS:  List[int] = [2023, 2024]
ALL_YEARS:   List[int] = [2020, 2021, 2022, 2023, 2024]

LEADERBOARD_FILE = ROOT / "output" / "leaderboard_b.json"
STATE_FILE        = ROOT / "output" / "optimization_state_b.json"
LOG_FILE          = ROOT / "output" / "sweep_b.log"

SWEEP_START = 800
SWEEP_END   = 1600   # exclusive

# Gate 2 criteria
GATE2_MIN_AVG_RETURN = 12.0   # % per year (simple avg of year_stats returns)
GATE2_MAX_DRAWDOWN   = -15.0  # % (must be >= this, i.e. better than -15%)
GATE2_MIN_OVERFIT    = 0.70   # test_return / train_return

# IBIT spread_width ($) → BTC spread_width_pct mapping
WIDTH_TO_PCT = {1: 0.03, 2: 0.05, 3: 0.08}


# ---------------------------------------------------------------------------
# Parameter mapping
# ---------------------------------------------------------------------------

def params_to_btc_config(params: dict) -> dict:
    """Map IBIT sweep params to BTCCreditSpreadBacktester config.

    Key translations:
      target_delta   → otm_pct  (delta ≈ OTM fraction for puts)
      target_dte     → dte_target
      profit_target  → profit_target_pct  (divide by 100)
      stop_loss_*    → stop_loss_multiplier (same)
      spread_width   → spread_width_pct via WIDTH_TO_PCT
      Regime scales  → ignored (BTC backtester is regime-agnostic)
    """
    width = params.get("spread_width", 2)
    return {
        "starting_capital":     100_000.0,
        "otm_pct":              params["target_delta"],
        "spread_width_pct":     WIDTH_TO_PCT.get(width, 0.05),
        "min_credit_pct":       float(params.get("min_credit_pct", 8.0)),
        "stop_loss_multiplier": params["stop_loss_multiplier"],
        "profit_target_pct":    params["profit_target"] / 100.0,
        "dte_target":           params["target_dte"],
        "dte_min":              params.get("min_dte", max(2, params["target_dte"] - 7)),
        "risk_per_trade_pct":   params.get("max_risk_per_trade", 5.0) / 100.0,
        "max_contracts":        int(params.get("max_contracts", 10)),
        "compound":             bool(params.get("compound", True)),
        "commission_rate":      0.0003,
        "slippage_btc":         0.0002,
    }


# ---------------------------------------------------------------------------
# Overfit score
# ---------------------------------------------------------------------------

def compute_overfit_score(train_return: float, test_return: float) -> float:
    """Return test_return / train_return, clamped to [0, 2].

    Logic:
      - If train_return <= 0 and test_return <= 0: coin flip, return 0.5
      - If train_return <= 0 and test_return >  0: great generalization → 1.5
      - If train_return <= 0 and test_return <= 0: both bad → 0.0
      - Otherwise: ratio, clamped to [0, 2]
    """
    if train_return <= 0:
        if test_return > 0:
            return 1.0   # improved out-of-sample — treat as neutral
        return 0.0       # both negative
    ratio = test_return / train_return
    return round(max(0.0, min(2.0, ratio)), 4)


# ---------------------------------------------------------------------------
# Single combo runner
# ---------------------------------------------------------------------------

def run_combo(params: dict) -> dict:
    """Run full + train + test backtest for one param set. Returns result dict."""
    btc_cfg = params_to_btc_config(params)

    # Full 5-year run
    full = BTCCreditSpreadBacktester(config=btc_cfg).run(ALL_YEARS)

    # Train period (2020-2022) — normalized starting capital
    train_cfg = {**btc_cfg, "starting_capital": 100_000.0}
    train = BTCCreditSpreadBacktester(config=train_cfg).run(TRAIN_YEARS)

    # Test period (2023-2024) — normalized starting capital (fair OOS comparison)
    test_cfg = {**btc_cfg, "starting_capital": 100_000.0}
    test = BTCCreditSpreadBacktester(config=test_cfg).run(TEST_YEARS)

    overfit = compute_overfit_score(train["return_pct"], test["return_pct"])

    # Per-year avg return from the full run
    yr_returns = [
        full["year_stats"].get(y, {}).get("return_pct", 0.0)
        for y in ALL_YEARS
    ]
    avg_annual = sum(yr_returns) / len(yr_returns)

    passes = (
        avg_annual >= GATE2_MIN_AVG_RETURN
        and full["max_drawdown"] >= GATE2_MAX_DRAWDOWN
        and overfit >= GATE2_MIN_OVERFIT
        and not full.get("ruin_triggered", False)
    )

    year_stats_out = {}
    for y in ALL_YEARS:
        ys = full["year_stats"].get(y, {})
        year_stats_out[str(y)] = {
            "return_pct":   round(ys.get("return_pct", 0.0), 2),
            "max_drawdown": round(ys.get("max_drawdown", 0.0), 2),
            "trade_count":  ys.get("trade_count", 0),
            "win_rate":     round(ys.get("win_rate", 0.0), 2),
            "pnl_usd":      round(ys.get("pnl_usd", 0.0), 0),
        }

    return {
        # Identity
        "target_dte":           params["target_dte"],
        "target_delta":         params["target_delta"],
        "profit_target":        params["profit_target"],
        "stop_loss_multiplier": params["stop_loss_multiplier"],
        "regime_profile":       params["regime_profile"],
        "spread_width":         params.get("spread_width", 2),
        # Full-period metrics
        "return_pct":           round(full["return_pct"], 2),
        "avg_annual_return":    round(avg_annual, 2),
        "max_drawdown":         round(full["max_drawdown"], 2),
        "total_trades":         full["total_trades"],
        "win_rate":             round(full["win_rate"], 2),
        "profit_factor":        round(full["profit_factor"], 2),
        "ending_capital":       round(full["ending_capital"], 0),
        # Train/test
        "train_return":         round(train["return_pct"], 2),
        "test_return":          round(test["return_pct"], 2),
        "overfit_score":        overfit,
        # Per-year detail
        "year_stats":           year_stats_out,
        # Gate verdict
        "passes_gate2":         passes,
        "ruin_triggered":       bool(full.get("ruin_triggered", False)),
    }


# ---------------------------------------------------------------------------
# State / leaderboard I/O
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_completed_index": SWEEP_START - 1, "total_runs": 0, "gate2_passes": 0}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_leaderboard() -> list:
    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE) as f:
            return json.load(f)
    return []


def save_leaderboard(entries: list) -> None:
    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(entries, key=lambda x: x.get("avg_annual_return", -999), reverse=True)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(ranked, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Dual output: console + file
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
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
    log = logging.getLogger("sweep_b")

    state       = load_state()
    leaderboard = load_leaderboard()

    resume_from    = state["last_completed_index"] + 1
    gate2_passes   = state.get("gate2_passes", 0)
    total_runs     = state.get("total_runs", 0)
    already_done   = len(leaderboard)

    log.info("=" * 60)
    log.info("IBIT Param Sweep — SECOND HALF (combos %d–%d)", SWEEP_START, SWEEP_END - 1)
    log.info("Resuming from index %d  |  already done: %d", resume_from, already_done)
    log.info("Gate2 criteria: avg≥%.0f%%, dd>%.0f%%, overfit≥%.2f",
             GATE2_MIN_AVG_RETURN, GATE2_MAX_DRAWDOWN, GATE2_MIN_OVERFIT)
    log.info("=" * 60)

    t0 = time.time()

    for abs_idx in range(resume_from, SWEEP_END):
        local_idx = abs_idx - SWEEP_START
        params = FULL_SWEEP[abs_idx]

        t_combo = time.time()
        try:
            result = run_combo(params)
        except Exception as exc:
            log.error("FAILED idx=%d  %s", abs_idx, exc)
            result = {
                "combo_index":    abs_idx,
                "target_dte":     params.get("target_dte"),
                "target_delta":   params.get("target_delta"),
                "profit_target":  params.get("profit_target"),
                "stop_loss_multiplier": params.get("stop_loss_multiplier"),
                "regime_profile": params.get("regime_profile"),
                "error":          str(exc),
                "avg_annual_return": -999,
                "passes_gate2":   False,
            }

        result["combo_index"]     = abs_idx
        result["run_timestamp"]   = datetime.utcnow().isoformat()
        result["combo_elapsed_s"] = round(time.time() - t_combo, 3)

        leaderboard.append(result)
        total_runs += 1

        if result.get("passes_gate2"):
            gate2_passes += 1
            log.info(
                "★ GATE2 #%d  idx=%d  DTE=%d δ=%.2f PT=%d SL=%.1f %-16s "
                "avg=%.1f%% dd=%.1f%% overfit=%.2f",
                gate2_passes, abs_idx,
                params["target_dte"], params["target_delta"],
                params["profit_target"], params["stop_loss_multiplier"],
                params["regime_profile"],
                result.get("avg_annual_return", 0),
                result.get("max_drawdown", 0),
                result.get("overfit_score", 0),
            )

        # Checkpoint every 50 combos
        combos_done = abs_idx - resume_from + 1
        if combos_done % 50 == 0 or abs_idx == SWEEP_END - 1:
            elapsed = time.time() - t0
            remaining = SWEEP_END - abs_idx - 1
            rate = combos_done / elapsed if elapsed > 0 else 1
            eta_min = remaining / rate / 60 if rate > 0 else 0

            log.info(
                "[%d/%d] idx=%d | gate2=%d | %.1f/s | ETA %.1f min",
                local_idx + 1, SWEEP_END - SWEEP_START,
                abs_idx, gate2_passes, rate, eta_min,
            )

            save_state({
                "last_completed_index": abs_idx,
                "total_runs":    total_runs,
                "gate2_passes":  gate2_passes,
                "elapsed_sec":   round(elapsed, 1),
                "eta_min":       round(eta_min, 1),
                "timestamp":     datetime.utcnow().isoformat(),
                "complete":      abs_idx == SWEEP_END - 1,
            })
            save_leaderboard(leaderboard)

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    log.info("=" * 60)
    log.info("SWEEP COMPLETE")
    log.info("Combos run this session: %d", total_runs - state.get("total_runs", 0))
    log.info("Gate2 passes total: %d / %d  (%.1f%%)",
             gate2_passes, total_runs,
             gate2_passes / total_runs * 100 if total_runs else 0)
    log.info("Elapsed: %.1f s", elapsed_total)
    log.info("Leaderboard: %s", LEADERBOARD_FILE)

    # Top 15
    log.info("")
    log.info("TOP 15 by avg annual return:")
    log.info("  %-4s  %-5s  %-5s  %-4s  %-4s  %-16s  %-8s  %-8s  %-8s  G2",
             "DTE", "delta", "PT%", "SL", "rank", "profile",
             "avg_ret%", "max_dd%", "overfit")
    top = sorted(
        [e for e in leaderboard if "error" not in e],
        key=lambda x: x.get("avg_annual_return", -999),
        reverse=True,
    )[:15]
    for rank, e in enumerate(top, 1):
        log.info(
            "  #%-3d  DTE=%2d  δ=%.2f  PT=%2d  SL=%.1f  %-16s  %+7.1f%%  %7.1f%%  %.3f  %s",
            rank,
            e.get("target_dte", 0), e.get("target_delta", 0),
            e.get("profit_target", 0), e.get("stop_loss_multiplier", 0),
            e.get("regime_profile", "?"),
            e.get("avg_annual_return", 0),
            e.get("max_drawdown", 0),
            e.get("overfit_score", 0),
            "✓" if e.get("passes_gate2") else "✗",
        )


if __name__ == "__main__":
    main()
