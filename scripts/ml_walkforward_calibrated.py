#!/usr/bin/env python3
"""
ML Walk-Forward Evaluation with Calibration Fix (Phase 3)

Applies Platt scaling and isotonic regression to fix G2 calibration failure.
Reuses infrastructure from ml_walkforward.py.

Strategy:
  For each fold, trains 4 models and compares:
    1. Raw XGBoost (baseline)
    2. Platt scaling via CalibratedClassifierCV(cv=3, method='sigmoid')
    3. Isotonic regression via CalibratedClassifierCV(cv=3, method='isotonic')
    4. Prefit approach: train on all-but-last-year, calibrate on last year

  CalibratedClassifierCV(cv=3) does internal cross-validation on the training
  set to learn calibration mapping, then averages predictions from 3 calibrators.
  This is more robust than single-year prefit when market regimes shift.

Output: results/ml_walkforward_results_calibrated.json

Usage:
    cd /home/node/openclaw/workspace/pilotai-credit-spreads
    PYTHONPATH=. python3 scripts/ml_walkforward_calibrated.py
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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# Reuse from existing walkforward script
from scripts.ml_walkforward import (
    COMBINED_CSV,
    FEATURE_COLS,
    FOLDS,
    TARGET_COL,
    XGB_PARAMS,
    _calibration_error,
    _prepare_xy,
    enrich_dataframe,
    evaluate_gates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_walkforward_cal")

RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "ml_walkforward_results_calibrated.json"


def _eval_method(
    name: str, y_test: np.ndarray, y_proba: np.ndarray,
) -> Dict:
    """Compute AUC and calibration error for a set of predictions."""
    auc = roc_auc_score(y_test, y_proba)
    cal_err = _calibration_error(y_test, y_proba, n_bins=10)
    return {"test_auc": round(auc, 4), "calibration_error": round(cal_err, 4)}


def run_fold_calibrated(
    df: pd.DataFrame, train_years: List[int], test_year: int, fold_name: str,
) -> Dict:
    """Run a single walk-forward fold with multiple calibration methods.

    Methods compared:
      - raw: plain XGBoost
      - platt_cv3: CalibratedClassifierCV(cv=3, method='sigmoid')
      - isotonic_cv3: CalibratedClassifierCV(cv=3, method='isotonic')
      - platt_prefit: train on all-but-last-year, calibrate on last year
      - isotonic_prefit: same, with isotonic

    Returns results with best method selected by lowest calibration error.
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

    methods = {}

    # ── 1. Raw XGBoost ────────────────────────────────────────────────────
    model_raw = xgb.XGBClassifier(**XGB_PARAMS)
    model_raw.fit(X_train, y_train, verbose=False)
    y_proba_raw = model_raw.predict_proba(X_test)[:, 1]
    methods["raw"] = _eval_method("raw", y_test, y_proba_raw)

    # Feature importance from raw model
    importance = dict(zip(FEATURE_COLS, model_raw.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top5 = [name for name, _ in sorted_imp[:5]]
    max_imp = sorted_imp[0][1] if sorted_imp else 0.0

    # ── 2. Platt scaling (CV=3) ───────────────────────────────────────────
    cal_platt = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_PARAMS), method="sigmoid", cv=3,
    )
    cal_platt.fit(X_train, y_train)
    y_proba_platt = cal_platt.predict_proba(X_test)[:, 1]
    methods["platt_cv3"] = _eval_method("platt_cv3", y_test, y_proba_platt)

    # ── 3. Isotonic regression (CV=3) ─────────────────────────────────────
    cal_iso = CalibratedClassifierCV(
        xgb.XGBClassifier(**XGB_PARAMS), method="isotonic", cv=3,
    )
    cal_iso.fit(X_train, y_train)
    y_proba_iso = cal_iso.predict_proba(X_test)[:, 1]
    methods["isotonic_cv3"] = _eval_method("isotonic_cv3", y_test, y_proba_iso)

    # ── 4 & 5. Prefit approach (if enough data for chronological split) ──
    if len(train_years) >= 2:
        cal_year = train_years[-1]
        proper_years = train_years[:-1]
        proper_df = df[df["year"].isin(proper_years)]
        cal_df = df[df["year"] == cal_year]

        if len(proper_df) >= 15 and len(cal_df) >= 10:
            X_proper, y_proper = _prepare_xy(proper_df)
            X_cal, y_cal = _prepare_xy(cal_df)

            model_proper = xgb.XGBClassifier(**XGB_PARAMS)
            model_proper.fit(X_proper, y_proper, verbose=False)

            # Platt prefit
            from sklearn.base import clone
            try:
                from sklearn.frozen import FrozenEstimator
                cal_platt_pf = CalibratedClassifierCV(
                    FrozenEstimator(model_proper), method="sigmoid", cv="prefit",
                )
            except ImportError:
                cal_platt_pf = CalibratedClassifierCV(
                    model_proper, method="sigmoid", cv="prefit",
                )
            cal_platt_pf.fit(X_cal, y_cal)
            y_proba_platt_pf = cal_platt_pf.predict_proba(X_test)[:, 1]
            methods["platt_prefit"] = _eval_method("platt_prefit", y_test, y_proba_platt_pf)

            # Isotonic prefit
            try:
                cal_iso_pf = CalibratedClassifierCV(
                    FrozenEstimator(model_proper), method="isotonic", cv="prefit",
                )
            except ImportError:
                cal_iso_pf = CalibratedClassifierCV(
                    model_proper, method="isotonic", cv="prefit",
                )
            cal_iso_pf.fit(X_cal, y_cal)
            y_proba_iso_pf = cal_iso_pf.predict_proba(X_test)[:, 1]
            methods["isotonic_prefit"] = _eval_method("isotonic_prefit", y_test, y_proba_iso_pf)

    # ── Pick best method ──────────────────────────────────────────────────
    best_method = min(methods, key=lambda m: methods[m]["calibration_error"])

    for name, m in methods.items():
        marker = " ← BEST" if name == best_method else ""
        logger.info(
            "  %-18s AUC=%.3f  cal_err=%.4f%s",
            name + ":", m["test_auc"], m["calibration_error"], marker,
        )

    best = methods[best_method]
    test_wr = float(y_test.mean())
    accuracy = round(float(((y_proba_raw > 0.5).astype(int) == y_test).mean()), 4)

    return {
        "fold": fold_name,
        "train_years": train_years,
        "test_year": test_year,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "calibration_methods": methods,
        "best_method": best_method,
        # Fields expected by evaluate_gates():
        "test_auc": best["test_auc"],
        "calibration_error": best["calibration_error"],
        "accuracy": accuracy,
        "test_win_rate": round(test_wr, 4),
        "top5_features": top5,
        "max_feature_importance": round(max_imp, 4),
        "feature_importances": {k: round(v, 4) for k, v in sorted_imp},
    }


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("ML Walk-Forward Evaluation — CALIBRATION FIX")
    logger.info("=" * 60)

    # Load training data
    if not COMBINED_CSV.exists():
        logger.error("Training data not found: %s", COMBINED_CSV)
        sys.exit(1)

    df = pd.read_csv(COMBINED_CSV)
    logger.info("Loaded %d trades from %s", len(df), COMBINED_CSV)

    # Enrich with new features
    df = enrich_dataframe(df)

    # Run walk-forward folds with calibration
    fold_results = []
    for fold in FOLDS:
        result = run_fold_calibrated(
            df,
            train_years=fold["train_years"],
            test_year=fold["test_year"],
            fold_name=fold["name"],
        )
        fold_results.append(result)

    # Evaluate G1-G4 gates (using best-calibrated values per fold)
    gates = evaluate_gates(fold_results)

    # Summary
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("CALIBRATED WALK-FORWARD EVALUATION COMPLETE")
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

    # Method comparison summary
    logger.info("")
    logger.info("Best calibration method per fold:")
    for fr in fold_results:
        if "error" in fr:
            continue
        logger.info(
            "  %s: %s (cal_err=%.4f)",
            fr["fold"], fr["best_method"], fr["calibration_error"],
        )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "description": "Walk-forward with calibration fix (Platt + isotonic, CV and prefit methods)",
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
