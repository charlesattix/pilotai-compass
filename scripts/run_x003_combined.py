#!/usr/bin/env python3
"""
run_x003_combined.py — X-003 Combined Test: COMPASS C-001 + ML threshold

Runs COMPASS C-001 (macro_sizing=true) backtest per year, enriches trades
with market features, scores with ML model, and filters by confidence
threshold.  Recomputes portfolio stats from the ML-filtered trade set.

Acceptance criteria:
  Combined return >= max(COMPASS-only, ML-only)
  Combined DD     <= min(COMPASS-only, ML-only)

Usage:
    PYTHONPATH=. python3 scripts/run_x003_combined.py [--threshold 0.55]
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from compass.collect_training_data import enrich_trades, _load_full_market_data
from compass.signal_model import SignalModel
from engine.portfolio_backtester import PortfolioBacktester
from scripts.run_phase4_validate import (
    CompassBacktester,
    build_strategy_params,
    evaluate_gates,
    load_yaml_config,
    _forward_fill_to_daily,
    _load_macro_scores,
    _load_rrg_breadth,
    STARTING_CAPITAL,
    TICKERS,
    YEARS,
)
from strategies import STRATEGY_REGISTRY

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_PATH = "ml/models/signal_model_20260321.joblib"

# ML feature columns (must match model training order)
ML_FEATURE_COLS = [
    "day_of_week", "days_since_last_trade", "dte_at_entry",
    "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
    "vix", "vix_percentile_20d", "vix_percentile_50d", "vix_percentile_100d",
    "iv_rank",
    "dist_from_ma20_pct", "dist_from_ma50_pct", "dist_from_ma80_pct", "dist_from_ma200_pct",
    "ma20_slope_ann_pct", "ma50_slope_ann_pct",
    "realized_vol_atr20", "realized_vol_5d", "realized_vol_10d", "realized_vol_20d",
    "regime_bear", "regime_bull", "regime_crash", "regime_high_vol", "regime_low_vol",
    "strategy_type_CS", "strategy_type_IC", "strategy_type_SS",
]


def _one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot columns for regime and strategy_type."""
    for regime in ["bear", "bull", "crash", "high_vol", "low_vol"]:
        df[f"regime_{regime}"] = (df["regime"] == regime).astype(int)
    for stype in ["CS", "IC", "SS"]:
        df[f"strategy_type_{stype}"] = (df["strategy_type"] == stype).astype(int)
    return df


def run_year_with_trades(
    cs_params: Dict, ss_params: Dict, year: int,
    compass_config: Dict, macro_daily: Dict, rrg_daily: Dict,
) -> Tuple[Dict, "CompassBacktester"]:
    """Run one year and return both results and backtester (for trade access)."""
    cs_cls = STRATEGY_REGISTRY["credit_spread"]
    ss_cls = STRATEGY_REGISTRY["straddle_strangle"]

    bt = CompassBacktester(
        strategies=[("credit_spread", cs_cls(dict(cs_params))),
                    ("straddle_strangle", ss_cls(dict(ss_params)))],
        tickers=TICKERS,
        start_date=datetime(year, 1, 1),
        end_date=datetime(year, 12, 31),
        starting_capital=STARTING_CAPITAL,
        max_positions=12,
        max_positions_per_strategy=5,
        compass_config=compass_config,
        macro_scores_daily=macro_daily,
        rrg_breadth_daily=rrg_daily,
    )
    raw = bt.run()
    combined = raw.get("combined", raw)

    result = {
        "return_pct": combined.get("return_pct", 0),
        "max_drawdown": combined.get("max_drawdown", 0),
        "total_trades": combined.get("total_trades", 0),
        "win_rate": combined.get("win_rate", 0),
        "sharpe_ratio": combined.get("sharpe_ratio", 0),
        "compass_stats": bt.get_compass_stats(),
    }
    return result, bt


def compute_drawdown_from_trades(trades: List[Dict], capital: float) -> float:
    """Compute max drawdown from a list of trade dicts (sorted by exit_date).

    Uses a simplified approach: running equity curve from trade PnL events.
    """
    if not trades:
        return 0.0

    equity = capital
    peak = capital
    max_dd = 0.0

    for t in sorted(trades, key=lambda x: x["exit_date"]):
        equity += t["pnl"]
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak * 100 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return round(max_dd, 2)


def main():
    parser = argparse.ArgumentParser(description="X-003 Combined Test")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="ML confidence threshold (default 0.55)")
    args = parser.parse_args()
    threshold = args.threshold

    print("=" * 70)
    print(f"  X-003 COMBINED TEST: COMPASS C-001 + ML threshold {threshold}")
    print("=" * 70)

    # ── Load COMPASS data ─────────────────────────────────────────────────
    print("\nLoading COMPASS data...")
    macro_weekly = _load_macro_scores()
    rrg_weekly = _load_rrg_breadth()
    macro_daily = _forward_fill_to_daily(macro_weekly)
    rrg_daily = _forward_fill_to_daily(rrg_weekly)
    print(f"  Macro scores: {len(macro_daily)} daily entries")
    print(f"  RRG breadth:  {len(rrg_daily)} daily entries")

    # ── Load ML model ─────────────────────────────────────────────────────
    print("\nLoading ML model...")
    model = SignalModel()
    if not model.load(Path(MODEL_PATH).name):
        print("ERROR: Failed to load ML model")
        sys.exit(1)
    print(f"  Model loaded: {MODEL_PATH}")
    print(f"  Features: {len(model.feature_names)}")

    # ── Load C-001 config ─────────────────────────────────────────────────
    config_path = ROOT / "configs" / "compass_c001.yaml"
    config = load_yaml_config(config_path)
    cs_params, ss_params = build_strategy_params(config)
    compass_cfg = config.get("compass", {})

    # ── Load full market data for enrichment ──────────────────────────────
    print("\nLoading market data for trade enrichment...")
    full_spy, full_vix = _load_full_market_data()
    full_spy_closes = full_spy["Close"]

    # ── Run C-001 backtest per year + enrich + score ──────────────────────
    print(f"\nRunning C-001 backtest with ML filter (threshold={threshold})...")
    print("-" * 70)

    all_trades_enriched = []  # All C-001 trades with ML scores
    yearly_c001 = {}          # C-001 raw results (no ML filter)

    for year in YEARS:
        t0 = time.time()
        print(f"  {year}...", end=" ", flush=True)

        # Run C-001 backtest
        result, bt = run_year_with_trades(
            cs_params, ss_params, year, compass_cfg, macro_daily, rrg_daily,
        )
        yearly_c001[str(year)] = result

        # Enrich trades with market features
        trades = enrich_trades(bt, year, spy_closes=full_spy_closes, vix_series=full_vix)

        if trades:
            df = pd.DataFrame(trades)
            df = _one_hot_encode(df)

            # Build feature matrix
            for col in ML_FEATURE_COLS:
                if col not in df.columns:
                    df[col] = 0.0
            features_df = df[ML_FEATURE_COLS].fillna(0.0)

            # Score with ML model
            probabilities = model.predict_batch(features_df)
            df["ml_probability"] = probabilities
            df["ml_pass"] = probabilities >= threshold

            trades_kept = df[df["ml_pass"]].to_dict("records")
            trades_rejected = len(df) - len(trades_kept)
        else:
            trades_kept = []
            trades_rejected = 0

        all_trades_enriched.extend(
            df.to_dict("records") if trades else []
        )

        elapsed = time.time() - t0
        print(f"C-001: {result['return_pct']:+.1f}%  {result['total_trades']} trades  "
              f"ML kept: {len(trades_kept)}/{result['total_trades']}  "
              f"({elapsed:.0f}s)")

    # ── Compute combined (ML-filtered) results ────────────────────────────
    print("\n" + "=" * 70)
    print("  COMBINED RESULTS (COMPASS C-001 + ML)")
    print("=" * 70)

    df_all = pd.DataFrame(all_trades_enriched)
    df_kept = df_all[df_all["ml_pass"] == True].copy()

    yearly_combined = {}
    for year in YEARS:
        year_trades = df_kept[df_kept["year"] == year]
        if len(year_trades) == 0:
            yearly_combined[str(year)] = {
                "return_pct": 0, "max_drawdown": 0, "total_trades": 0,
                "win_rate": 0, "trades_kept": 0, "trades_rejected": 0,
            }
            continue

        total_pnl = year_trades["pnl"].sum()
        return_pct = round(total_pnl / STARTING_CAPITAL * 100, 2)
        n_trades = len(year_trades)
        n_wins = year_trades["win"].sum()
        win_rate = round(n_wins / n_trades * 100, 2) if n_trades > 0 else 0

        # Drawdown from filtered trades
        max_dd = compute_drawdown_from_trades(
            year_trades.to_dict("records"), STARTING_CAPITAL
        )

        # Count rejected
        year_all = df_all[df_all["year"] == year]
        n_rejected = len(year_all) - n_trades

        yearly_combined[str(year)] = {
            "return_pct": return_pct,
            "max_drawdown": max_dd,
            "total_trades": n_trades,
            "win_rate": win_rate,
            "trades_kept": n_trades,
            "trades_rejected": n_rejected,
        }

        c001_ret = yearly_c001[str(year)]["return_pct"]
        delta = return_pct - c001_ret
        print(f"  {year}: C-001={c001_ret:+.1f}%  Combined={return_pct:+.1f}%  "
              f"Δ={delta:+.1f}%  trades={n_trades}/{len(year_all)}  WR={win_rate:.0f}%")

    # ── Compute summary stats ─────────────────────────────────────────────
    rets = [yearly_combined[str(y)]["return_pct"] for y in YEARS]
    dds = [yearly_combined[str(y)]["max_drawdown"] for y in YEARS]
    trades_list = [yearly_combined[str(y)]["total_trades"] for y in YEARS]

    summary = {
        "experiment_id": "X-003",
        "description": f"COMPASS C-001 (macro_sizing) + ML threshold {threshold}",
        "ml_threshold": threshold,
        "yearly": yearly_combined,
        "avg_return": round(sum(rets) / len(rets), 2),
        "min_return": round(min(rets), 2),
        "max_return": round(max(rets), 2),
        "worst_dd": round(min(dds), 2),
        "total_trades": sum(trades_list),
        "years_profitable": sum(1 for r in rets if r > 0),
        "year_2022_return": yearly_combined.get("2022", {}).get("return_pct", 0),
    }

    # Evaluate gates
    gates = evaluate_gates(summary)
    summary["gates"] = gates

    # ── C-001 baseline summary (for comparison) ───────────────────────────
    c001_rets = [yearly_c001[str(y)]["return_pct"] for y in YEARS]
    c001_dds = [yearly_c001[str(y)]["max_drawdown"] for y in YEARS]
    c001_summary = {
        "avg_return": round(sum(c001_rets) / len(c001_rets), 2),
        "worst_dd": round(min(c001_dds), 2),
        "total_trades": sum(yearly_c001[str(y)]["total_trades"] for y in YEARS),
    }

    # ── Acceptance criteria ───────────────────────────────────────────────
    # Combined must beat max(C-001 alone, ML-only) on return
    # and beat min(C-001 alone, ML-only) on DD
    c001_avg = c001_summary["avg_return"]
    c001_dd = c001_summary["worst_dd"]

    combined_avg = summary["avg_return"]
    combined_dd = summary["worst_dd"]

    print(f"\n  SUMMARY:")
    print(f"    C-001 alone:  avg={c001_avg:+.1f}%  worst_dd={c001_dd:.1f}%  "
          f"trades={c001_summary['total_trades']}")
    print(f"    Combined:     avg={combined_avg:+.1f}%  worst_dd={combined_dd:.1f}%  "
          f"trades={summary['total_trades']}")
    print(f"    Delta:        avg={combined_avg - c001_avg:+.1f}%  "
          f"dd={combined_dd - c001_dd:+.1f}%")
    print(f"    ROBUST score: {gates['robust_score']}")
    print(f"    All gates:    {'PASS' if gates['all_pass'] else 'FAIL'}")
    print(f"    6/6 profit:   {'PASS' if gates['profitable_6_6'] else 'FAIL'}")
    print(f"    Trades>=250:  {'PASS' if gates['trades_pass'] else 'FAIL'}")
    print(f"    2022>=5%:     {'PASS' if gates['year_2022_pass'] else 'FAIL'}")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "generated_at": datetime.now().isoformat(),
        "ml_model": MODEL_PATH,
        "ml_threshold": threshold,
        "c001_baseline": {
            "yearly": yearly_c001,
            **c001_summary,
        },
        "combined": summary,
        "acceptance": {
            "combined_return_vs_c001": combined_avg - c001_avg,
            "combined_dd_vs_c001": combined_dd - c001_dd,
        },
    }

    out_path = ROOT / "results" / "x003_combined_results.json"
    out_path.parent.mkdir(exist_ok=True)
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
