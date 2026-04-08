#!/usr/bin/env python3
"""
Phase 4: ML Backtest Sweep — Confidence Threshold Evaluation

All 4 ML gates passed (Phase 3). Now test whether ML trade filtering
improves on the baseline strategy by sweeping confidence thresholds.

Experiments:
  M-001: ML confidence threshold 0.40
  M-002: ML confidence threshold 0.45
  M-003: ML confidence threshold 0.50
  M-004: ML confidence threshold 0.55
  M-005: ML confidence threshold 0.60

Walk-forward structure (same as Phase 3):
  Fold 1: train 2020-2022, test 2023
  Fold 2: train 2020-2023, test 2024
  Fold 3: train 2020-2024, test 2025

Metrics per experiment:
  - Avg annual return (% on $100k base)
  - Max drawdown (%)
  - Trade count
  - Win rate
  - Yearly breakdown (return, trades, wins, max DD)

Usage:
    cd /home/node/openclaw/workspace/pilotai-credit-spreads
    PYTHONPATH=. python3 scripts/phase4_ml_sweep.py
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. pip install xgboost")
    sys.exit(1)

from sklearn.metrics import roc_auc_score

# Reuse enrichment from ml_walkforward
from scripts.ml_walkforward import (
    FEATURE_COLS,
    FOLDS,
    TARGET_COL,
    XGB_PARAMS,
    enrich_dataframe,
    _prepare_xy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4_ml_sweep")

# ── Paths ────────────────────────────────────────────────────────────────────
COMBINED_CSV = ROOT / "compass" / "training_data_combined.csv"
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "phase4_ml_sweep_results.json"

# ── Configuration ────────────────────────────────────────────────────────────
STARTING_CAPITAL = 100_000
CONFIDENCE_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]
EXPERIMENT_IDS = ["M-001", "M-002", "M-003", "M-004", "M-005"]


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward training & prediction
# ═══════════════════════════════════════════════════════════════════════════

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Train walk-forward XGBoost and get OOS predictions for each test year.

    Returns DataFrame with original data + 'ml_confidence' column for OOS rows.
    """
    df = df.copy()
    df["ml_confidence"] = np.nan  # Will be filled for OOS predictions

    for fold in FOLDS:
        fold_name = fold["name"]
        train_years = fold["train_years"]
        test_year = fold["test_year"]

        train_mask = df["year"].isin(train_years)
        test_mask = df["year"] == test_year

        train_df = df[train_mask]
        test_df = df[test_mask]

        logger.info(
            "%s: train=%d trades (%s), test=%d trades (%d)",
            fold_name, len(train_df), train_years, len(test_df), test_year,
        )

        if len(train_df) < 20 or len(test_df) < 5:
            logger.warning("%s: insufficient data, skipping", fold_name)
            continue

        X_train, y_train = _prepare_xy(train_df.copy())
        X_test, y_test = _prepare_xy(test_df.copy())

        # Train XGBoost
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        # Get test predictions
        y_proba = model.predict_proba(X_test)[:, 1]

        # AUC for logging
        auc = roc_auc_score(y_test, y_proba)
        logger.info("%s: AUC=%.3f, n_predictions=%d", fold_name, auc, len(y_proba))

        # Store predictions
        df.loc[test_mask, "ml_confidence"] = y_proba

    # Report coverage
    oos_count = df["ml_confidence"].notna().sum()
    logger.info("OOS predictions: %d / %d trades", oos_count, len(df))

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio metrics computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_max_drawdown(pnl_series: pd.Series) -> float:
    """Compute max drawdown from a series of trade PnLs.

    Returns max drawdown as a negative percentage of starting capital.
    """
    if len(pnl_series) == 0:
        return 0.0

    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    return round(float(max_dd / STARTING_CAPITAL * 100), 2)


def compute_yearly_metrics(trades: pd.DataFrame, year: int) -> Dict:
    """Compute metrics for a single year."""
    yr_trades = trades[trades["year"] == year].sort_values("entry_date")
    n_trades = len(yr_trades)

    if n_trades == 0:
        return {
            "year": year,
            "return_pct": 0.0,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
        }

    total_pnl = float(yr_trades["pnl"].sum())
    wins = int(yr_trades["win"].sum())
    return_pct = round(total_pnl / STARTING_CAPITAL * 100, 2)
    max_dd = compute_max_drawdown(yr_trades["pnl"].reset_index(drop=True))
    win_rate = round(wins / n_trades * 100, 1)

    return {
        "year": year,
        "return_pct": return_pct,
        "trades": n_trades,
        "wins": wins,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "total_pnl": round(total_pnl, 2),
    }


def compute_portfolio_metrics(trades: pd.DataFrame, years: List[int]) -> Dict:
    """Compute full portfolio metrics across specified years."""
    yearly = {}
    for yr in years:
        yearly[str(yr)] = compute_yearly_metrics(trades, yr)

    # Aggregate
    yearly_returns = [yearly[str(yr)]["return_pct"] for yr in years]
    total_trades = sum(yearly[str(yr)]["trades"] for yr in years)
    total_wins = sum(yearly[str(yr)]["wins"] for yr in years)
    worst_dd = min(yearly[str(yr)]["max_drawdown"] for yr in years)

    avg_return = round(float(np.mean(yearly_returns)), 2)
    win_rate = round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0.0
    years_profitable = sum(1 for r in yearly_returns if r > 0)

    # Max drawdown across all years (track sequential PnL)
    all_trades = trades[trades["year"].isin(years)].sort_values("entry_date")
    overall_max_dd = compute_max_drawdown(all_trades["pnl"].reset_index(drop=True))

    return {
        "avg_return_pct": avg_return,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": win_rate,
        "years_profitable": f"{years_profitable}/{len(years)}",
        "worst_year_dd": worst_dd,
        "overall_max_dd": overall_max_dd,
        "yearly": yearly,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Sweep execution
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep(df: pd.DataFrame) -> Dict:
    """Run the full confidence threshold sweep."""
    oos_years = [2023, 2024, 2025]
    oos_df = df[df["year"].isin(oos_years)].copy()

    results = {}

    # ── Baseline: no ML filter ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE (no ML filter)")
    logger.info("=" * 60)

    baseline = compute_portfolio_metrics(oos_df, oos_years)
    baseline["experiment_id"] = "BASELINE"
    baseline["description"] = "No ML filtering — all trades accepted"
    baseline["confidence_threshold"] = None

    logger.info(
        "  avg_return=%.2f%%, trades=%d, win_rate=%.1f%%, max_dd=%.2f%%",
        baseline["avg_return_pct"],
        baseline["total_trades"],
        baseline["win_rate"],
        baseline["overall_max_dd"],
    )
    results["baseline"] = baseline

    # ── ML-filtered experiments ─────────────────────────────────────────────
    experiments = []

    for exp_id, threshold in zip(EXPERIMENT_IDS, CONFIDENCE_THRESHOLDS):
        logger.info("")
        logger.info("=" * 60)
        logger.info("%s: ML confidence threshold = %.2f", exp_id, threshold)
        logger.info("=" * 60)

        # Filter: keep trades where ML confidence >= threshold
        filtered = oos_df[oos_df["ml_confidence"] >= threshold].copy()
        rejected = len(oos_df) - len(filtered)

        logger.info(
            "  Accepted: %d / %d trades (rejected %d, %.1f%%)",
            len(filtered), len(oos_df), rejected,
            rejected / len(oos_df) * 100 if len(oos_df) > 0 else 0,
        )

        metrics = compute_portfolio_metrics(filtered, oos_years)
        metrics["experiment_id"] = exp_id
        metrics["description"] = f"ML confidence threshold >= {threshold}"
        metrics["confidence_threshold"] = threshold
        metrics["trades_rejected"] = rejected
        metrics["rejection_rate_pct"] = round(
            rejected / len(oos_df) * 100 if len(oos_df) > 0 else 0, 1
        )

        # Delta vs baseline
        metrics["delta_avg_return"] = round(
            metrics["avg_return_pct"] - baseline["avg_return_pct"], 2
        )
        metrics["delta_win_rate"] = round(
            metrics["win_rate"] - baseline["win_rate"], 1
        )
        metrics["delta_max_dd"] = round(
            metrics["overall_max_dd"] - baseline["overall_max_dd"], 2
        )

        logger.info(
            "  avg_return=%.2f%% (delta=%.2f%%), trades=%d, "
            "win_rate=%.1f%% (delta=%.1f%%), max_dd=%.2f%%",
            metrics["avg_return_pct"], metrics["delta_avg_return"],
            metrics["total_trades"],
            metrics["win_rate"], metrics["delta_win_rate"],
            metrics["overall_max_dd"],
        )

        experiments.append(metrics)

    results["experiments"] = experiments

    # ── Confidence distribution analysis ────────────────────────────────────
    confidence_vals = oos_df["ml_confidence"].dropna()
    results["confidence_distribution"] = {
        "mean": round(float(confidence_vals.mean()), 4),
        "median": round(float(confidence_vals.median()), 4),
        "std": round(float(confidence_vals.std()), 4),
        "min": round(float(confidence_vals.min()), 4),
        "max": round(float(confidence_vals.max()), 4),
        "pct_above_50": round(
            float((confidence_vals >= 0.50).mean() * 100), 1
        ),
    }

    # ── Win rate by confidence bucket ───────────────────────────────────────
    oos_with_conf = oos_df[oos_df["ml_confidence"].notna()].copy()
    buckets = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]
    bucket_analysis = []
    for lo, hi in buckets:
        mask = (oos_with_conf["ml_confidence"] >= lo) & (oos_with_conf["ml_confidence"] < hi)
        bucket_trades = oos_with_conf[mask]
        n = len(bucket_trades)
        if n > 0:
            wr = round(float(bucket_trades["win"].mean() * 100), 1)
            avg_pnl = round(float(bucket_trades["pnl"].mean()), 2)
        else:
            wr = 0.0
            avg_pnl = 0.0
        bucket_analysis.append({
            "range": f"[{lo:.1f}, {hi:.2f})" if hi < 1 else f"[{lo:.1f}, 1.00]",
            "trades": n,
            "win_rate": wr,
            "avg_pnl": avg_pnl,
        })
    results["confidence_buckets"] = bucket_analysis

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Phase 4: ML Backtest Sweep — Confidence Threshold Evaluation")
    logger.info("=" * 60)

    # Load training data
    if not COMBINED_CSV.exists():
        logger.error("Training data not found: %s", COMBINED_CSV)
        sys.exit(1)

    df = pd.read_csv(COMBINED_CSV)
    logger.info("Loaded %d trades from %s", len(df), COMBINED_CSV)
    logger.info("Years: %s", sorted(df["year"].unique()))
    logger.info("Strategy types: %s", df["strategy_type"].value_counts().to_dict())

    # Enrich with Phase 3 features
    df = enrich_dataframe(df)

    # Train walk-forward models and get OOS predictions
    logger.info("")
    logger.info("Training walk-forward XGBoost models...")
    df = train_and_predict(df)

    # Run confidence threshold sweep
    logger.info("")
    logger.info("Running confidence threshold sweep...")
    results = run_sweep(df)

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 4 SWEEP COMPLETE")
    logger.info("=" * 60)

    baseline = results["baseline"]
    logger.info(
        "BASELINE: avg_return=%.2f%%, trades=%d, win_rate=%.1f%%, max_dd=%.2f%%",
        baseline["avg_return_pct"], baseline["total_trades"],
        baseline["win_rate"], baseline["overall_max_dd"],
    )
    logger.info("")

    best_exp = None
    best_improvement = -999

    for exp in results["experiments"]:
        improvement = exp["delta_avg_return"]
        marker = " <-- BEST" if improvement == max(
            e["delta_avg_return"] for e in results["experiments"]
        ) else ""
        logger.info(
            "  %s (threshold=%.2f): avg_return=%.2f%% (delta=%+.2f%%), "
            "trades=%d, win_rate=%.1f%%, max_dd=%.2f%%%s",
            exp["experiment_id"], exp["confidence_threshold"],
            exp["avg_return_pct"], exp["delta_avg_return"],
            exp["total_trades"], exp["win_rate"],
            exp["overall_max_dd"], marker,
        )
        if improvement > best_improvement:
            best_improvement = improvement
            best_exp = exp

    logger.info("")

    # Confidence bucket analysis
    logger.info("Confidence Bucket Analysis (actual win rates vs ML confidence):")
    for bucket in results["confidence_buckets"]:
        logger.info(
            "  %s: n=%d, win_rate=%.1f%%, avg_pnl=$%.0f",
            bucket["range"], bucket["trades"], bucket["win_rate"], bucket["avg_pnl"],
        )

    # Verdict
    logger.info("")
    if best_exp and best_improvement > 0:
        logger.info(
            "VERDICT: %s (threshold=%.2f) IMPROVES on baseline by +%.2f%% avg return",
            best_exp["experiment_id"], best_exp["confidence_threshold"],
            best_improvement,
        )
    else:
        logger.info("VERDICT: No ML threshold improves on baseline avg return")

    logger.info("Elapsed: %.1f seconds", elapsed)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "description": "Phase 4 ML Backtest Sweep — Confidence Threshold Evaluation",
        "starting_capital": STARTING_CAPITAL,
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        "oos_years": [2023, 2024, 2025],
        "n_total_trades": len(df),
        "n_oos_trades": int(df[df["year"].isin([2023, 2024, 2025])].shape[0]),
        "model_config": XGB_PARAMS,
        "features_used": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "baseline": results["baseline"],
        "experiments": results["experiments"],
        "confidence_distribution": results["confidence_distribution"],
        "confidence_buckets": results["confidence_buckets"],
        "verdict": {
            "best_experiment": best_exp["experiment_id"] if best_exp else None,
            "best_threshold": best_exp["confidence_threshold"] if best_exp else None,
            "improvement_pct": best_improvement if best_improvement > 0 else 0,
            "improves_on_baseline": best_improvement > 0,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info("Results saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
