#!/usr/bin/env python3
"""
ML Walk-Forward Evaluation (Phase 3)

Anchored walk-forward XGBoost training with G1-G4 gate evaluation.

Folds:
  Fold 1: train 2020-2022, test 2023
  Fold 2: train 2020-2023, test 2024
  Fold 3: train 2020-2024, test 2025

Gates:
  G1: AUC > 0.55 on ALL folds
  G2: Calibration error < 0.10
  G3: Feature importance stable across folds (top-5 overlap >= 3)
  G4: No single feature > 40% importance

Output: results/ml_walkforward_results.json

Usage:
    cd /home/node/openclaw/workspace/pilotai-credit-spreads
    PYTHONPATH=. python3 scripts/ml_walkforward.py
"""

import json
import logging
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from compass.regime import RegimeClassifier
from engine.portfolio_backtester import PortfolioBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_walkforward")

# ── Paths ────────────────────────────────────────────────────────────────────
COMBINED_CSV = ROOT / "compass" / "training_data_combined.csv"
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "ml_walkforward_results.json"

# ── Walk-forward folds ───────────────────────────────────────────────────────
FOLDS = [
    {"name": "fold_1", "train_years": [2020, 2021, 2022], "test_year": 2023},
    {"name": "fold_2", "train_years": [2020, 2021, 2022, 2023], "test_year": 2024},
    {"name": "fold_3", "train_years": [2020, 2021, 2022, 2023, 2024], "test_year": 2025},
]

# ── Features to use ─────────────────────────────────────────────────────────
# Numeric features from training data (after new additions and dead drops)
FEATURE_COLS = [
    # Timing
    "dte_at_entry", "hold_days", "day_of_week", "days_since_last_trade",
    # Regime & signals
    "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
    # VIX
    "vix", "iv_rank",
    # Price & MAs
    "dist_from_ma20_pct", "dist_from_ma50_pct",
    "dist_from_ma80_pct", "dist_from_ma200_pct",
    # Volatility
    "realized_vol_atr20", "realized_vol_5d", "realized_vol_10d", "realized_vol_20d",
    # Trade structure
    "net_credit", "otm_pct", "contracts",
    # NEW Phase 3 features
    "credit_to_width_ratio",
    "regime_duration_days",
    "vix_change_5d",
]

# Dead features (dropped in Phase 3):
# - spread_width: constant per config
# - vix_percentile_20d/50d/100d: redundant with iv_rank
# - ma20_slope_ann_pct, ma50_slope_ann_pct: noisy

TARGET_COL = "win"


# ═══════════════════════════════════════════════════════════════════════════
# Feature enrichment
# ═══════════════════════════════════════════════════════════════════════════

def _load_market_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load SPY + VIX data via backtester data loader (2018-2025)."""
    logger.info("Loading market data for feature enrichment...")
    bt = PortfolioBacktester(
        strategies=[],
        tickers=["SPY"],
        start_date=datetime(2018, 1, 1),
        end_date=datetime(2025, 12, 31),
        starting_capital=100_000,
    )
    bt._load_data()
    spy = bt._price_data.get("SPY")
    vix = bt._vix_series
    if spy is None or spy.empty:
        raise RuntimeError("Failed to load SPY data")
    if vix is None or vix.empty:
        raise RuntimeError("Failed to load VIX data")
    logger.info("Market data: SPY=%d rows, VIX=%d rows", len(spy), len(vix))
    return spy, vix


def _compute_regime_durations(
    spy_data: pd.DataFrame, vix_series: pd.Series,
) -> Dict[pd.Timestamp, int]:
    """Compute regime duration in days for every trading day.

    Returns {date: number of consecutive days in current regime}.
    """
    clf = RegimeClassifier()
    regime_series = clf.classify_series(spy_data, vix_series)

    durations = {}
    streak = 0
    prev_regime = None
    for date, regime in regime_series.items():
        if regime == prev_regime:
            streak += 1
        else:
            streak = 1
            prev_regime = regime
        durations[pd.Timestamp(date)] = streak

    return durations


def _compute_vix_changes(vix_series: pd.Series) -> Dict[pd.Timestamp, float]:
    """Compute 5-day VIX change for each trading day.

    Returns {date: vix_today - vix_5_days_ago}.
    """
    changes = {}
    vix_vals = vix_series.dropna()
    for i in range(5, len(vix_vals)):
        date = vix_vals.index[i]
        changes[pd.Timestamp(date)] = float(vix_vals.iloc[i] - vix_vals.iloc[i - 5])
    return changes


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add new Phase 3 features to the training DataFrame."""
    df = df.copy()

    # 1. credit_to_width_ratio (from existing columns)
    df["credit_to_width_ratio"] = np.where(
        (df["spread_width"].notna()) & (df["spread_width"] > 0),
        df["net_credit"].abs() / df["spread_width"],
        np.nan,
    )

    # 2. regime_duration_days and 3. vix_change_5d (from market data)
    spy_data, vix_series = _load_market_data()
    regime_durations = _compute_regime_durations(spy_data, vix_series)
    vix_changes = _compute_vix_changes(vix_series)

    # Map to trade entry dates
    entry_dates = pd.to_datetime(df["entry_date"])

    regime_dur_vals = []
    vix_chg_vals = []
    for dt in entry_dates:
        ts = pd.Timestamp(dt)
        # Find closest prior date in regime_durations
        dur = regime_durations.get(ts)
        if dur is None:
            prior = [k for k in regime_durations if k <= ts]
            dur = regime_durations[max(prior)] if prior else 1
        regime_dur_vals.append(dur)

        chg = vix_changes.get(ts)
        if chg is None:
            prior = [k for k in vix_changes if k <= ts]
            chg = vix_changes[max(prior)] if prior else 0.0
        vix_chg_vals.append(chg)

    df["regime_duration_days"] = regime_dur_vals
    df["vix_change_5d"] = vix_chg_vals

    logger.info(
        "Enriched %d trades: credit_to_width mean=%.3f, "
        "regime_dur mean=%.1f, vix_change_5d mean=%.2f",
        len(df),
        df["credit_to_width_ratio"].mean(),
        df["regime_duration_days"].mean(),
        df["vix_change_5d"].mean(),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# XGBoost walk-forward
# ═══════════════════════════════════════════════════════════════════════════

XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
}


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract feature matrix X and label vector y from DataFrame."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning("Missing feature columns (will be zero-filled): %s", missing)
        for col in missing:
            df[col] = 0.0
    X = df[FEATURE_COLS].copy()
    # Fill NaN with 0 (XGBoost handles this well)
    X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    y = df[TARGET_COL].values.astype(int)
    return X, y


def run_fold(
    df: pd.DataFrame, train_years: List[int], test_year: int, fold_name: str,
) -> Dict:
    """Run a single walk-forward fold.

    Returns dict with AUC, calibration error, feature importances, etc.
    """
    train_df = df[df["year"].isin(train_years)]
    test_df = df[df["year"] == test_year]

    logger.info(
        "%s: train=%d trades (%s), test=%d trades (%d)",
        fold_name, len(train_df), train_years, len(test_df), test_year,
    )

    if len(train_df) < 20 or len(test_df) < 5:
        logger.error("%s: insufficient data", fold_name)
        return {"error": "insufficient data"}

    X_train, y_train = _prepare_xy(train_df)
    X_test, y_test = _prepare_xy(test_df)

    # Train XGBoost
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    # Predict probabilities on test set
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test > 0.5).astype(int)

    # Also predict on train for comparison
    y_proba_train = model.predict_proba(X_train)[:, 1]

    # AUC
    test_auc = roc_auc_score(y_test, y_proba_test)
    train_auc = roc_auc_score(y_train, y_proba_train)

    # Calibration error: mean absolute difference between predicted prob
    # and actual outcome in probability bins
    cal_error = _calibration_error(y_test, y_proba_test, n_bins=10)

    # Feature importance (gain-based)
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top5 = [name for name, _ in sorted_imp[:5]]
    max_imp = sorted_imp[0][1] if sorted_imp else 0.0

    # Accuracy
    accuracy = float((y_pred_test == y_test).mean())
    test_wr = float(y_test.mean())

    logger.info(
        "%s results: AUC=%.3f (train=%.3f), cal_err=%.3f, acc=%.3f, "
        "test_WR=%.1f%%, top5=%s, max_imp=%.3f",
        fold_name, test_auc, train_auc, cal_error, accuracy,
        test_wr * 100, top5, max_imp,
    )

    return {
        "fold": fold_name,
        "train_years": train_years,
        "test_year": test_year,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_auc": round(train_auc, 4),
        "test_auc": round(test_auc, 4),
        "calibration_error": round(cal_error, 4),
        "accuracy": round(accuracy, 4),
        "test_win_rate": round(test_wr, 4),
        "top5_features": top5,
        "max_feature_importance": round(max_imp, 4),
        "feature_importances": {k: round(v, 4) for k, v in sorted_imp},
    }


def _calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions into n_bins equal-width bins, computes |avg_predicted - avg_actual|
    for each bin, weighted by bin size.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_pred = y_prob[mask].mean()
        avg_actual = y_true[mask].mean()
        ece += (n_in_bin / total) * abs(avg_pred - avg_actual)
    return ece


# ═══════════════════════════════════════════════════════════════════════════
# G1-G4 gate evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_gates(fold_results: List[Dict]) -> Dict:
    """Evaluate G1-G4 gates on walk-forward results."""
    valid_folds = [f for f in fold_results if "error" not in f]
    n_folds = len(valid_folds)

    if n_folds == 0:
        return {"passed": False, "reason": "no valid folds"}

    # G1: AUC > 0.55 on ALL folds
    aucs = [f["test_auc"] for f in valid_folds]
    g1_pass = all(a > 0.55 for a in aucs)
    g1 = {
        "gate": "G1_AUC",
        "threshold": 0.55,
        "values": aucs,
        "min_auc": min(aucs),
        "passed": g1_pass,
    }

    # G2: Calibration error < 0.10 on all folds
    cal_errors = [f["calibration_error"] for f in valid_folds]
    g2_pass = all(c < 0.10 for c in cal_errors)
    g2 = {
        "gate": "G2_Calibration",
        "threshold": 0.10,
        "values": cal_errors,
        "max_cal_error": max(cal_errors),
        "passed": g2_pass,
    }

    # G3: Feature importance stable across folds (top-5 overlap >= 3)
    top5_sets = [set(f["top5_features"]) for f in valid_folds]
    if len(top5_sets) >= 2:
        # Pairwise overlap
        overlaps = []
        for i in range(len(top5_sets)):
            for j in range(i + 1, len(top5_sets)):
                overlaps.append(len(top5_sets[i] & top5_sets[j]))
        min_overlap = min(overlaps)
        # Overall: intersection of all folds' top-5
        common_top5 = top5_sets[0]
        for s in top5_sets[1:]:
            common_top5 = common_top5 & s
        g3_pass = min_overlap >= 3
    else:
        min_overlap = 5
        common_top5 = top5_sets[0] if top5_sets else set()
        g3_pass = True

    g3 = {
        "gate": "G3_Stability",
        "threshold": "top-5 pairwise overlap >= 3",
        "min_pairwise_overlap": min_overlap,
        "common_top5": sorted(common_top5),
        "all_top5": [sorted(s) for s in top5_sets],
        "passed": g3_pass,
    }

    # G4: No single feature > 40% importance
    max_imps = [f["max_feature_importance"] for f in valid_folds]
    g4_pass = all(m <= 0.40 for m in max_imps)
    g4 = {
        "gate": "G4_NoFeatureDominance",
        "threshold": 0.40,
        "values": max_imps,
        "max_across_folds": max(max_imps),
        "passed": g4_pass,
    }

    all_pass = g1_pass and g2_pass and g3_pass and g4_pass

    return {
        "all_gates_passed": all_pass,
        "n_folds": n_folds,
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "G4": g4,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("ML Walk-Forward Evaluation (Phase 3)")
    logger.info("=" * 60)

    # Load training data
    if not COMBINED_CSV.exists():
        logger.error("Training data not found: %s", COMBINED_CSV)
        sys.exit(1)

    df = pd.read_csv(COMBINED_CSV)
    logger.info("Loaded %d trades from %s", len(df), COMBINED_CSV)
    logger.info("Years: %s", sorted(df["year"].unique()))
    logger.info("Strategy types: %s", df["strategy_type"].value_counts().to_dict())

    # Enrich with new features
    df = enrich_dataframe(df)

    # Run walk-forward folds
    fold_results = []
    for fold in FOLDS:
        result = run_fold(
            df,
            train_years=fold["train_years"],
            test_year=fold["test_year"],
            fold_name=fold["name"],
        )
        fold_results.append(result)

    # Evaluate G1-G4 gates
    gates = evaluate_gates(fold_results)

    # Summary
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("WALK-FORWARD EVALUATION COMPLETE")
    logger.info("=" * 60)

    for g_key in ["G1", "G2", "G3", "G4"]:
        g = gates[g_key]
        status = "PASS" if g["passed"] else "FAIL"
        logger.info("  %s: %s — %s", g_key, status, g["gate"])

    logger.info("")
    logger.info(
        "  ALL GATES: %s",
        "PASSED" if gates["all_gates_passed"] else "FAILED",
    )
    logger.info("  Elapsed: %.1f seconds", elapsed)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "data_source": str(COMBINED_CSV),
        "n_trades": len(df),
        "features_used": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "folds": fold_results,
        "gates": gates,
        "elapsed_seconds": round(elapsed, 1),
    }

    def _json_default(obj):
        """Convert numpy types for JSON serialization."""
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info("Results saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
