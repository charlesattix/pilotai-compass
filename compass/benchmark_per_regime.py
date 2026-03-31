"""
Per-Regime Benchmark: Ensemble vs XGBoost
==========================================

Runs full walk-forward validation for both XGBoost (single model) and
EnsembleSignalModel, then breaks out-of-sample predictions down by market
regime (and by strategy_type x regime).

Answers: Should we use ML selectively — e.g., only in certain regimes?

Output: compass/benchmark_results_per_regime.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root on sys.path ────────────────────────────────────────────
HERE = Path(__file__).parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Import submodules directly to avoid compass/__init__.py pulling in
# heavy dependencies (requests, etc.) that aren't available here.
import importlib.util as _ilu

def _import_module(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Pre-register stubs for transitive imports that __init__ would trigger
for _stub in ["compass"]:
    if _stub not in sys.modules:
        import types
        sys.modules[_stub] = types.ModuleType(_stub)

# shared.indicators and shared.types must load first
_shared = ROOT / "shared"
_import_module("shared", str(_shared / "__init__.py"))
_import_module("shared.indicators", str(_shared / "indicators.py"))
_import_module("shared.types", str(_shared / "types.py"))

# Now load the two compass submodules we actually need
_import_module("compass.feature_pipeline", str(HERE / "feature_pipeline.py"))
_import_module("compass.ensemble_signal_model", str(HERE / "ensemble_signal_model.py"))

from compass.ensemble_signal_model import EnsembleSignalModel
from compass.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
DATA_PATH = HERE / "training_data_combined.csv"
OUTPUT_PATH = HERE / "benchmark_results_per_regime.json"

MIN_REGIME_SAMPLES_FOR_AUC = 10   # need both classes + at least this many
MIN_TRAIN_SAMPLES = 40

_XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
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


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> Optional[float]:
    """Compute AUC; return None when both classes aren't present."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_proba))


def _regime_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    return_pct: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute classification metrics for a slice of predictions."""
    n = len(y_true)
    if n == 0:
        return {"n": 0}

    base_win_rate = float(y_true.mean())
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    auc = _safe_auc(y_true, y_proba)

    # "ML win rate": win rate among trades the model predicted as wins
    predicted_wins_mask = y_pred == 1
    ml_win_rate = (
        float(y_true[predicted_wins_mask].mean())
        if predicted_wins_mask.sum() > 0
        else None
    )
    n_predicted_wins = int(predicted_wins_mask.sum())

    result: Dict[str, Any] = {
        "n": n,
        "base_win_rate": round(base_win_rate, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "n_predicted_wins": n_predicted_wins,
        "ml_win_rate": round(ml_win_rate, 4) if ml_win_rate is not None else None,
        "win_rate_lift": (
            round(ml_win_rate - base_win_rate, 4)
            if ml_win_rate is not None
            else None
        ),
    }

    # Mean return of ML-filtered trades vs all trades
    if return_pct is not None:
        result["mean_return_all"] = round(float(np.mean(return_pct)), 4)
        if predicted_wins_mask.sum() > 0:
            result["mean_return_ml_filtered"] = round(
                float(np.mean(return_pct[predicted_wins_mask])), 4
            )
        else:
            result["mean_return_ml_filtered"] = None

    return result


def _breakdown(
    records: pd.DataFrame,
    group_col: str,
    model_col: str,
    min_n: int = 5,
) -> Dict[str, Any]:
    """Group records by group_col and compute metrics per group per model."""
    out: Dict[str, Any] = {}
    for group_val, grp in records.groupby(group_col):
        out[str(group_val)] = {}
        for model_name, mgrp in grp.groupby(model_col):
            y_true = mgrp["true_label"].values
            y_pred = mgrp["prediction"].values
            y_proba = mgrp["probability"].values
            ret = mgrp["return_pct"].values if "return_pct" in mgrp else None
            if len(y_true) < min_n:
                out[str(group_val)][str(model_name)] = {
                    "n": len(y_true),
                    "note": "too_few_samples",
                }
            else:
                out[str(group_val)][str(model_name)] = _regime_metrics(
                    y_true, y_pred, y_proba, ret
                )
    return out


# ── Walk-forward core ────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expanding-window walk-forward: train on all years < test_year.

    Returns a DataFrame of per-trade OOS records with columns:
        entry_date, year, regime, strategy_type, return_pct,
        true_label, model, prediction, probability
    """
    pipeline = FeaturePipeline()

    # Build full feature matrix once (consistent columns)
    features_full = pipeline.transform(df)
    feature_cols = list(features_full.columns)

    df = df.reset_index(drop=True)
    features_full = features_full.reset_index(drop=True)

    years = sorted(df["year"].unique())
    print(f"  Years in data: {years}")

    all_records: List[Dict] = []

    for fold_idx, test_year in enumerate(years[1:], start=1):
        train_years = [y for y in years if y < test_year]
        train_mask = df["year"].isin(train_years)
        test_mask = df["year"] == test_year

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        if n_train < MIN_TRAIN_SAMPLES:
            print(f"  Fold {fold_idx} (test={test_year}): skip — only {n_train} train samples")
            continue

        print(f"  Fold {fold_idx} (train={train_years}, test={test_year}): "
              f"n_train={n_train}, n_test={n_test}")

        X_train = features_full.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "win"].values.astype(int)
        X_test = features_full.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, "win"].values.astype(int)

        feat_train_df = features_full.loc[train_mask, feature_cols]
        feat_test_df = features_full.loc[test_mask, feature_cols]

        test_meta = df.loc[test_mask, ["entry_date", "year", "regime", "strategy_type", "return_pct"]].copy()

        # ── XGBoost ───────────────────────────────────────────────────
        xgb_model = xgb.XGBClassifier(**_XGB_PARAMS)
        # Use a small inner val split for early stopping, but don't leak test
        from sklearn.model_selection import train_test_split as tts
        if n_train > 80:
            X_tr2, X_val2, y_tr2, y_val2 = tts(
                X_train, y_train, test_size=0.15, random_state=42
            )
            xgb_model.fit(X_tr2, y_tr2, eval_set=[(X_val2, y_val2)], verbose=False)
        else:
            xgb_model.fit(X_train, y_train)

        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = (xgb_proba > 0.5).astype(int)

        # ── EnsembleSignalModel ───────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            ens_model = EnsembleSignalModel(model_dir=tmpdir)
            ens_model.train(feat_train_df, y_train, save_model=False)
            ens_proba = ens_model.predict_batch(feat_test_df)

        ens_pred = (ens_proba > 0.5).astype(int)

        # ── Collect records ───────────────────────────────────────────
        for i, idx in enumerate(test_meta.index):
            row = test_meta.loc[idx]
            base = {
                "fold": fold_idx,
                "entry_date": str(row["entry_date"]),
                "year": int(row["year"]),
                "regime": str(row["regime"]),
                "strategy_type": str(row["strategy_type"]),
                "return_pct": float(row["return_pct"]) if pd.notna(row["return_pct"]) else 0.0,
                "true_label": int(y_test[i]),
            }
            all_records.append({**base, "model": "xgboost",
                                 "prediction": int(xgb_pred[i]),
                                 "probability": float(xgb_proba[i])})
            all_records.append({**base, "model": "ensemble",
                                 "prediction": int(ens_pred[i]),
                                 "probability": float(ens_proba[i])})

    return pd.DataFrame(all_records)


# ── Analysis ─────────────────────────────────────────────────────────────

def build_results(records: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate OOS records into per-regime (and regime x strategy) metrics."""

    results: Dict[str, Any] = {}

    # ── 1. Overall aggregate ──────────────────────────────────────────
    print("\nComputing overall metrics...")
    overall: Dict[str, Any] = {}
    for model_name, mgrp in records.groupby("model"):
        overall[model_name] = _regime_metrics(
            mgrp["true_label"].values,
            mgrp["prediction"].values,
            mgrp["probability"].values,
            mgrp["return_pct"].values,
        )
    results["overall"] = overall

    # ── 2. Per-regime ─────────────────────────────────────────────────
    print("Computing per-regime metrics...")
    results["by_regime"] = _breakdown(records, "regime", "model", min_n=5)

    # ── 3. Per-year ───────────────────────────────────────────────────
    print("Computing per-year metrics...")
    results["by_year"] = _breakdown(records, "year", "model", min_n=3)

    # ── 4. Regime x strategy_type ─────────────────────────────────────
    print("Computing regime x strategy_type metrics...")
    records["regime_x_strategy"] = records["regime"] + "_x_" + records["strategy_type"]
    results["by_regime_x_strategy"] = _breakdown(
        records, "regime_x_strategy", "model", min_n=5
    )

    # ── 5. Ensemble lift over XGBoost per regime ──────────────────────
    print("Computing ensemble lift over XGBoost per regime...")
    lift: Dict[str, Any] = {}
    regimes = records["regime"].unique()
    for regime in regimes:
        reg_records = records[records["regime"] == regime]
        xgb_m = reg_records[reg_records["model"] == "xgboost"]
        ens_m = reg_records[reg_records["model"] == "ensemble"]

        if len(xgb_m) < 5:
            lift[str(regime)] = {"note": "too_few_samples", "n": len(xgb_m)}
            continue

        xgb_acc = float(accuracy_score(xgb_m["true_label"], xgb_m["prediction"]))
        ens_acc = float(accuracy_score(ens_m["true_label"], ens_m["prediction"]))

        xgb_auc = _safe_auc(xgb_m["true_label"].values, xgb_m["probability"].values)
        ens_auc = _safe_auc(ens_m["true_label"].values, ens_m["probability"].values)

        xgb_wr = float(xgb_m["true_label"].mean())  # same as base win rate
        # ML-filtered win rate: trades ensemble predicted as wins
        ens_wins_mask = ens_m["prediction"].values == 1
        ens_ml_wr = (
            float(ens_m["true_label"].values[ens_wins_mask].mean())
            if ens_wins_mask.sum() > 0 else None
        )
        xgb_wins_mask = xgb_m["prediction"].values == 1
        xgb_ml_wr = (
            float(xgb_m["true_label"].values[xgb_wins_mask].mean())
            if xgb_wins_mask.sum() > 0 else None
        )

        lift[str(regime)] = {
            "n": len(xgb_m),
            "base_win_rate": round(xgb_wr, 4),
            "xgboost_accuracy": round(xgb_acc, 4),
            "ensemble_accuracy": round(ens_acc, 4),
            "accuracy_delta": round(ens_acc - xgb_acc, 4),
            "xgboost_auc": round(xgb_auc, 4) if xgb_auc is not None else None,
            "ensemble_auc": round(ens_auc, 4) if ens_auc is not None else None,
            "auc_delta": (
                round(ens_auc - xgb_auc, 4)
                if ens_auc is not None and xgb_auc is not None else None
            ),
            "xgboost_ml_win_rate": round(xgb_ml_wr, 4) if xgb_ml_wr is not None else None,
            "ensemble_ml_win_rate": round(ens_ml_wr, 4) if ens_ml_wr is not None else None,
            "ml_win_rate_delta": (
                round(ens_ml_wr - xgb_ml_wr, 4)
                if ens_ml_wr is not None and xgb_ml_wr is not None else None
            ),
        }
    results["ensemble_lift_over_xgboost"] = lift

    # ── 6. Sample summary ─────────────────────────────────────────────
    n_trades = len(records) // 2  # records has 2 rows per trade (one per model)
    results["meta"] = {
        "total_oos_trades": n_trades,
        "regime_counts": records[records["model"] == "xgboost"]["regime"].value_counts().to_dict(),
        "strategy_counts": records[records["model"] == "xgboost"]["strategy_type"].value_counts().to_dict(),
        "year_counts": records[records["model"] == "xgboost"]["year"].value_counts().sort_index().to_dict(),
        "folds_run": int(records["fold"].nunique()),
    }

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Per-Regime Benchmark: Ensemble vs XGBoost")
    print("=" * 60)

    print(f"\nLoading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    print(f"  {len(df)} trades, years {df['year'].min()}-{df['year'].max()}")
    print(f"  Regime distribution: {df['regime'].value_counts().to_dict()}")

    print("\nRunning walk-forward validation...")
    records = run_walk_forward(df)

    if records.empty:
        print("ERROR: No OOS records produced. Check data and fold configuration.")
        return

    n_oos = len(records) // 2
    print(f"\nCollected {n_oos} OOS trade predictions across "
          f"{records['fold'].nunique()} folds")

    print("\nBuilding regime analysis...")
    results = build_results(records)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n── OVERALL RESULTS ──────────────────────────────────────")
    for model, m in results["overall"].items():
        print(f"  {model:12s}  acc={m['accuracy']:.3f}  "
              f"auc={m.get('auc', 'N/A')}  "
              f"ml_win_rate={m.get('ml_win_rate','N/A')}")

    print("\n── ENSEMBLE LIFT OVER XGBOOST BY REGIME ─────────────────")
    for regime, lt in results["ensemble_lift_over_xgboost"].items():
        if "note" in lt:
            print(f"  {regime:12s}  n={lt['n']}  ({lt['note']})")
        else:
            delta = lt.get("accuracy_delta", 0)
            auc_d = lt.get("auc_delta")
            print(f"  {regime:12s}  n={lt['n']:3d}  "
                  f"base_wr={lt['base_win_rate']:.3f}  "
                  f"acc_delta={delta:+.3f}  "
                  f"auc_delta={auc_d if auc_d is not None else 'N/A'}")

    # ── Serialize ─────────────────────────────────────────────────────
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
