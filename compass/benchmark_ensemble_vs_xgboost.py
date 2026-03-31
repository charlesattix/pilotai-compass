"""
Benchmark: EnsembleSignalModel vs standalone XGBoost on combined training data.

Compares:
  - Standalone XGBoost (same hyperparameters as SignalModel.train)
  - EnsembleSignalModel (XGBoost + RandomForest + ExtraTrees, walk-forward weighted)

Both models are evaluated using the same WalkForwardValidator — expanding-window,
year-by-year chronological splits — on compass/training_data_combined.csv (428 trades,
2020-2025, mixed CS/IC/SS strategy types).

Features: FeaturePipeline.transform() is applied before validation to produce
stationary, normalized inputs (z-scored prices, ratio-based trade structure,
domain-aware imputation).

Usage:
    python -m compass.benchmark_ensemble_vs_xgboost
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as exc:
    print(f"ERROR: XGBoost is required. pip install xgboost\n{exc}", file=sys.stderr)
    sys.exit(1)

from sklearn.base import BaseEstimator

from compass.ensemble_signal_model import EnsembleSignalModel
from compass.feature_pipeline import FeaturePipeline
from compass.walk_forward import WalkForwardValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("compass/training_data_combined.csv")
RESULTS_PATH = Path("compass/benchmark_results_combined.json")


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn-compatible wrapper for EnsembleSignalModel
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleAdapter(BaseEstimator):
    """Thin sklearn-compatible wrapper around EnsembleSignalModel.

    Allows EnsembleSignalModel to be used inside WalkForwardValidator, which
    expects a sklearn fit/predict/predict_proba interface and uses clone() to
    create fresh instances per fold.

    Parameters
    ----------
    feature_names : list of str
        Column names corresponding to the numpy feature matrix columns.
        Must be set before fit() is called; clone() will preserve this.
    """

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        self.feature_names = feature_names

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleAdapter":
        tmp_dir = tempfile.mkdtemp(prefix="ensemble_bench_")
        self._model = EnsembleSignalModel(model_dir=tmp_dir)

        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]
        features_df = pd.DataFrame(X, columns=names)

        self._model.train(
            features_df=features_df,
            labels=y,
            calibrate=True,
            save_model=False,  # No disk I/O during benchmark
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]
        features_df = pd.DataFrame(X, columns=names)
        proba1 = self._model.predict_batch(features_df)
        return np.column_stack([1.0 - proba1, proba1])


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply FeaturePipeline and reconstruct a WalkForwardValidator-compatible DataFrame.

    The WalkForwardValidator expects a DataFrame with:
      - ``entry_date`` (date column for year-based splits)
      - ``win`` (binary target)
      - ``return_pct`` (per-trade return for Signal Sharpe)
      - feature columns referenced by ``numeric_features``

    We append the pipeline-transformed feature columns to these metadata columns.

    Returns
    -------
    df_out : DataFrame ready for WalkForwardValidator.run()
    feature_names : list of column names produced by the pipeline
    """
    pipeline = FeaturePipeline()
    features_clean = pipeline.transform(df)
    feature_names = list(features_clean.columns)

    meta = df[["entry_date", "win", "return_pct"]].copy().reset_index(drop=True)
    feat = features_clean.reset_index(drop=True)

    df_out = pd.concat([meta, feat], axis=1)
    return df_out, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────────────────────

def run_xgboost(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    """Run walk-forward validation with standalone XGBoost."""
    logger.info("=" * 60)
    logger.info("RUNNING: Standalone XGBoost walk-forward")
    logger.info("=" * 60)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=200,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )

    validator = WalkForwardValidator(
        model=model,
        numeric_features=feature_names,
        categorical_features=[],  # Pipeline already one-hot encoded categoricals
        min_train_samples=30,
    )

    t0 = time.perf_counter()
    results = validator.run(df)
    results["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    results["model_name"] = "XGBoost"
    return results


def run_ensemble(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    """Run walk-forward validation with EnsembleSignalModel."""
    logger.info("=" * 60)
    logger.info("RUNNING: EnsembleSignalModel walk-forward")
    logger.info("=" * 60)

    model = EnsembleAdapter(feature_names=feature_names)

    validator = WalkForwardValidator(
        model=model,
        numeric_features=feature_names,
        categorical_features=[],  # Pipeline already one-hot encoded categoricals
        min_train_samples=30,
    )

    t0 = time.perf_counter()
    results = validator.run(df)
    results["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    results["model_name"] = "EnsembleSignalModel"
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(xgb_results: Dict, ens_results: Dict) -> None:
    """Print formatted head-to-head comparison table."""
    xgb_agg = xgb_results["aggregate"]
    ens_agg = ens_results["aggregate"]

    metrics = [
        ("Accuracy",      "accuracy_mean",      "accuracy_std"),
        ("Precision",     "precision_mean",      "precision_std"),
        ("Recall",        "recall_mean",         "recall_std"),
        ("Brier Score",   "brier_score_mean",    "brier_score_std"),
        ("AUC",           "auc_mean",            "auc_std"),
        ("Signal Sharpe", "signal_sharpe_mean",  "signal_sharpe_std"),
    ]

    print("\n" + "=" * 84)
    print("  ENSEMBLE vs XGBOOST — combined dataset (428 trades, 2020-2025, mixed CS/IC/SS)")
    print("=" * 84)
    print(f"  {'Metric':<18}  {'XGBoost':>18}  {'Ensemble':>18}  {'Δ (Ens-XGB)':>12}  Winner")
    print("  " + "─" * 78)

    for label, mean_key, std_key in metrics:
        xv = xgb_agg.get(mean_key)
        ev = ens_agg.get(mean_key)

        if xv is None or ev is None:
            print(f"  {label:<18}  {'N/A':>18}  {'N/A':>18}  {'N/A':>12}")
            continue

        xs = xgb_agg.get(std_key) or 0.0
        es = ens_agg.get(std_key) or 0.0
        delta = ev - xv

        # Brier: lower is better; all others: higher is better
        if mean_key == "brier_score_mean":
            wins_ensemble = delta < -0.001
            wins_xgboost  = delta >  0.001
        else:
            wins_ensemble = delta >  0.001
            wins_xgboost  = delta < -0.001

        if wins_ensemble:
            winner = ">>> Ensemble"
        elif wins_xgboost:
            winner = "    XGBoost"
        else:
            winner = "    TIE"

        print(
            f"  {label:<18}  {xv:>7.4f} ±{xs:<7.4f}  "
            f"{ev:>7.4f} ±{es:<7.4f}  "
            f"{delta:>+11.4f}  {winner}"
        )

    print("  " + "─" * 78)
    n_folds_x = xgb_results["n_folds"]
    n_folds_e = ens_results["n_folds"]
    oos_x = xgb_agg["total_oos_samples"]
    oos_e = ens_agg["total_oos_samples"]
    print(
        f"\n  Folds: XGBoost={n_folds_x}  Ensemble={n_folds_e}  |  "
        f"OOS samples: XGBoost={oos_x}  Ensemble={oos_e}"
    )
    print(
        f"  Elapsed: XGBoost={xgb_results['elapsed_seconds']}s  "
        f"Ensemble={ens_results['elapsed_seconds']}s"
    )

    # Per-fold AUC table
    print("\n" + "─" * 84)
    print("  PER-FOLD AUC DETAIL")
    print("─" * 84)
    print(f"  {'Fold':<6}  {'Test Period':<28}  {'N-train':>7}  {'N-test':>6}  "
          f"{'XGB AUC':>9}  {'Ens AUC':>9}  {'Δ AUC':>8}")
    print("  " + "─" * 78)

    for xf, ef in zip(xgb_results["folds"], ens_results["folds"]):
        xauc = xf.get("auc")
        eauc = ef.get("auc")
        xauc_s = f"{xauc:.4f}" if xauc is not None else "   N/A"
        eauc_s = f"{eauc:.4f}" if eauc is not None else "   N/A"
        delta_s = f"{eauc - xauc:+.4f}" if (xauc is not None and eauc is not None) else "   N/A"
        period = ef.get("test_period", "")
        n_tr   = ef.get("n_train", 0)
        n_te   = ef.get("n_test", 0)
        print(
            f"  {xf['fold']:<6}  {period:<28}  {n_tr:>7}  {n_te:>6}  "
            f"{xauc_s:>9}  {eauc_s:>9}  {delta_s:>8}"
        )

    print()


def determine_verdict(xgb_agg: Dict, ens_agg: Dict) -> Dict[str, str]:
    """Determine winner on AUC, Brier score, and Signal Sharpe."""
    def cmp(xv, ev, higher_is_better: bool, threshold: float = 0.001) -> str:
        if xv is None or ev is None:
            return "N/A"
        delta = ev - xv
        if higher_is_better:
            if delta > threshold:
                return "Ensemble"
            if delta < -threshold:
                return "XGBoost"
        else:
            if delta < -threshold:
                return "Ensemble"
            if delta > threshold:
                return "XGBoost"
        return "TIE"

    return {
        "auc":          cmp(xgb_agg.get("auc_mean"),           ens_agg.get("auc_mean"),           higher_is_better=True),
        "brier_score":  cmp(xgb_agg.get("brier_score_mean"),   ens_agg.get("brier_score_mean"),   higher_is_better=False),
        "signal_sharpe":cmp(xgb_agg.get("signal_sharpe_mean"), ens_agg.get("signal_sharpe_mean"), higher_is_better=True, threshold=0.05),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Loading %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info(
        "Loaded %d trades | years: %s | strategy types: %s",
        len(df),
        sorted(df["year"].unique().tolist()) if "year" in df.columns else "?",
        df["strategy_type"].value_counts().to_dict() if "strategy_type" in df.columns else "?",
    )

    # Apply FeaturePipeline for stationary, normalized features
    logger.info("Applying FeaturePipeline.transform()...")
    df_pipeline, feature_names = build_pipeline_df(df)
    logger.info("Pipeline → %d features: %s", len(feature_names), feature_names)

    # ── Run benchmarks ──────────────────────────────────────────────────
    xgb_results = run_xgboost(df_pipeline, feature_names)
    ens_results = run_ensemble(df_pipeline, feature_names)

    # ── Print comparison ────────────────────────────────────────────────
    print_comparison(xgb_results, ens_results)

    verdict = determine_verdict(xgb_results["aggregate"], ens_results["aggregate"])
    print("VERDICT — does Ensemble beat XGBoost?")
    print(f"  AUC:           {verdict['auc']}")
    print(f"  Brier Score:   {verdict['brier_score']}")
    print(f"  Signal Sharpe: {verdict['signal_sharpe']}")
    print()

    # ── Save JSON results ───────────────────────────────────────────────
    years = sorted(int(y) for y in df["year"].unique()) if "year" in df.columns else []
    output = {
        "metadata": {
            "dataset": str(DATA_PATH),
            "n_trades": len(df),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "years": years,
            "strategy_type_counts": (
                df["strategy_type"].value_counts().to_dict()
                if "strategy_type" in df.columns else {}
            ),
        },
        "xgboost": {
            "model_name": xgb_results["model_name"],
            "aggregate": xgb_results["aggregate"],
            "folds": xgb_results["folds"],
            "elapsed_seconds": xgb_results["elapsed_seconds"],
            "n_folds": xgb_results["n_folds"],
        },
        "ensemble": {
            "model_name": ens_results["model_name"],
            "aggregate": ens_results["aggregate"],
            "folds": ens_results["folds"],
            "elapsed_seconds": ens_results["elapsed_seconds"],
            "n_folds": ens_results["n_folds"],
        },
        "verdict": verdict,
    }

    RESULTS_PATH.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Results saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
