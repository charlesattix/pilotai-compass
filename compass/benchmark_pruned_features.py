"""
Benchmark: Pruned features vs full feature set (walk-forward, 5-fold).

Removes features identified as harmful or noise by the ablation analysis
(experiments/feature_importance_report.md):

  Harmful (hurt AUC when present):
    - vix_percentile_20d (-0.009)  → already absent from pipeline
    - otm_pct (-0.007)             → already absent from pipeline
    - contracts (-0.006)           → pipeline equiv: contracts_log (PRUNED)
    - ma20_slope_ann_pct (-0.005)  → already absent from pipeline

  Noise (zero gain + zero permutation importance):
    - regime_bear, regime_bull, regime_crash, regime_high_vol, regime_low_vol
    - strategy_type_IC, strategy_type_SS
    - spread_type_bear_call, spread_type_unknown
    - spread_width      → already absent from pipeline
    - day_of_week       → already absent from pipeline

Total pipeline features: 31 → 21 after pruning (10 removed).

Usage::

    python3 -m compass.benchmark_pruned_features
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
RESULTS_PATH = Path("experiments/pruned_features_benchmark.json")
REPORT_PATH = Path("experiments/pruned_features_benchmark.md")

# ── Features to prune from the pipeline output ──────────────────────────

# Harmful: pipeline equivalent of raw "contracts" feature
HARMFUL_PIPELINE_FEATURES = [
    "contracts_log",
]

# Noise: zero-importance one-hot dummies
NOISE_PIPELINE_FEATURES = [
    "regime_bear",
    "regime_bull",
    "regime_crash",
    "regime_high_vol",
    "regime_low_vol",
    "strategy_type_IC",
    "strategy_type_SS",
    "spread_type_bear_call",
    "spread_type_unknown",
]

PRUNE_LIST = HARMFUL_PIPELINE_FEATURES + NOISE_PIPELINE_FEATURES

# Features already absent from pipeline (for documentation)
ALREADY_PRUNED_BY_PIPELINE = [
    "vix_percentile_20d",  # harmful
    "otm_pct",             # harmful
    "ma20_slope_ann_pct",  # harmful
    "spread_width",        # noise
    "day_of_week",         # noise
]


# ── Pipeline + feature preparation ───────────────────────────────────────


def build_pipeline_df(
    df: pd.DataFrame,
    prune: Optional[List[str]] = None,
) -> tuple:
    """Apply FeaturePipeline and optionally prune features.

    Always starts from the full 31-feature pipeline output (pruned=False)
    so this benchmark can compare full vs pruned side-by-side.

    Returns (df_out, feature_names).
    """
    pipeline = FeaturePipeline(pruned=False)
    features_clean = pipeline.transform(df)

    if prune:
        to_drop = [c for c in prune if c in features_clean.columns]
        features_clean = features_clean.drop(columns=to_drop)

    feature_names = list(features_clean.columns)
    meta = df[["entry_date", "win", "return_pct"]].copy().reset_index(drop=True)
    feat = features_clean.reset_index(drop=True)
    df_out = pd.concat([meta, feat], axis=1)
    return df_out, feature_names


# ── Sklearn adapter for Ensemble ─────────────────────────────────────────


class EnsembleAdapter(BaseEstimator):
    """Sklearn-compatible wrapper for EnsembleSignalModel."""

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        self.feature_names = feature_names

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleAdapter":
        tmp_dir = tempfile.mkdtemp(prefix="pruned_bench_")
        self._model = EnsembleSignalModel(model_dir=tmp_dir)
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]
        features_df = pd.DataFrame(X, columns=names)
        self._model.train(features_df=features_df, labels=y, calibrate=True, save_model=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]
        features_df = pd.DataFrame(X, columns=names)
        proba1 = self._model.predict_batch(features_df)
        return np.column_stack([1.0 - proba1, proba1])


# ── Benchmark runners ────────────────────────────────────────────────────


XGB_PARAMS = dict(
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


def run_xgboost(df: pd.DataFrame, feature_names: List[str], label: str) -> Dict[str, Any]:
    """Run walk-forward XGBoost benchmark."""
    logger.info("RUNNING: %s (%d features)", label, len(feature_names))
    model = xgb.XGBClassifier(**XGB_PARAMS)
    validator = WalkForwardValidator(
        model=model,
        numeric_features=feature_names,
        categorical_features=[],
        min_train_samples=30,
    )
    t0 = time.perf_counter()
    results = validator.run(df)
    results["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    results["model_name"] = label
    results["n_features"] = len(feature_names)
    results["feature_names"] = feature_names
    return results


def run_ensemble(df: pd.DataFrame, feature_names: List[str], label: str) -> Dict[str, Any]:
    """Run walk-forward Ensemble benchmark."""
    logger.info("RUNNING: %s (%d features)", label, len(feature_names))
    model = EnsembleAdapter(feature_names=feature_names)
    validator = WalkForwardValidator(
        model=model,
        numeric_features=feature_names,
        categorical_features=[],
        min_train_samples=30,
    )
    t0 = time.perf_counter()
    results = validator.run(df)
    results["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    results["model_name"] = label
    results["n_features"] = len(feature_names)
    results["feature_names"] = feature_names
    return results


# ── Report generation ────────────────────────────────────────────────────


def generate_markdown(
    full_xgb: Dict, pruned_xgb: Dict,
    full_ens: Optional[Dict], pruned_ens: Optional[Dict],
    full_features: List[str], pruned_features: List[str],
) -> str:
    """Generate the comparison markdown report."""
    fa = full_xgb["aggregate"]
    pa = pruned_xgb["aggregate"]

    lines = [
        "# Pruned Features Benchmark",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Dataset:** `compass/training_data_combined.csv` (428 trades, 2020-2025)",
        f"**Validation:** 5-fold walk-forward (year-based expanding window)",
        "",
        "## Feature Pruning Summary",
        "",
        f"- Original pipeline features: {len(full_features)}",
        f"- Pruned pipeline features: {len(pruned_features)}",
        f"- Features removed: {len(full_features) - len(pruned_features)}",
        "",
        "### Removed (present in pipeline):",
        "",
    ]
    for f in HARMFUL_PIPELINE_FEATURES:
        lines.append(f"- `{f}` (harmful — hurts AUC)")
    for f in NOISE_PIPELINE_FEATURES:
        lines.append(f"- `{f}` (noise — zero importance)")
    lines.append("")
    lines.append("### Already absent from pipeline (pruned by FeaturePipeline):")
    lines.append("")
    for f in ALREADY_PRUNED_BY_PIPELINE:
        lines.append(f"- `{f}`")

    # XGBoost comparison
    lines.extend([
        "",
        "## XGBoost: Full vs Pruned",
        "",
        "| Metric | Full (31 feat) | Pruned (21 feat) | Delta |",
        "|--------|---------------|-----------------|-------|",
    ])
    metrics = [
        ("AUC", "auc_mean", "auc_std", True),
        ("Accuracy", "accuracy_mean", "accuracy_std", True),
        ("Precision", "precision_mean", "precision_std", True),
        ("Recall", "recall_mean", "recall_std", True),
        ("Brier Score", "brier_score_mean", "brier_score_std", False),
        ("Signal Sharpe", "signal_sharpe_mean", "signal_sharpe_std", True),
    ]
    xgb_improved = False
    for label, mean_k, std_k, higher_better in metrics:
        fv = fa.get(mean_k, 0)
        pv = pa.get(mean_k, 0)
        fs = fa.get(std_k, 0)
        ps = pa.get(std_k, 0)
        delta = pv - fv
        if label == "AUC" and delta > 0.001:
            xgb_improved = True
        sign = "+" if delta > 0 else ""
        lines.append(
            f"| {label} | {fv:.4f} +/- {fs:.4f} | {pv:.4f} +/- {ps:.4f} | {sign}{delta:.4f} |"
        )

    # Per-fold AUC
    lines.extend(["", "### Per-Fold AUC"])
    lines.append("")
    lines.append("| Fold | Test Year | Full AUC | Pruned AUC | Delta |")
    lines.append("|------|-----------|----------|------------|-------|")
    for ff, pf in zip(full_xgb["folds"], pruned_xgb["folds"]):
        fauc = ff.get("auc", 0)
        pauc = pf.get("auc", 0)
        delta = pauc - fauc if fauc and pauc else 0
        period = pf.get("test_period", f"Fold {pf['fold']}")
        sign = "+" if delta > 0 else ""
        lines.append(f"| {pf['fold']} | {period} | {fauc:.4f} | {pauc:.4f} | {sign}{delta:.4f} |")

    # Ensemble comparison (if run)
    if full_ens and pruned_ens:
        ea_full = full_ens["aggregate"]
        ea_pruned = pruned_ens["aggregate"]
        lines.extend([
            "",
            "## Ensemble: Full vs Pruned",
            "",
            "| Metric | Full (31 feat) | Pruned (21 feat) | Delta |",
            "|--------|---------------|-----------------|-------|",
        ])
        for label, mean_k, std_k, higher_better in metrics:
            fv = ea_full.get(mean_k, 0)
            pv = ea_pruned.get(mean_k, 0)
            fs = ea_full.get(std_k, 0)
            ps = ea_pruned.get(std_k, 0)
            delta = pv - fv
            sign = "+" if delta > 0 else ""
            lines.append(
                f"| {label} | {fv:.4f} +/- {fs:.4f} | {pv:.4f} +/- {ps:.4f} | {sign}{delta:.4f} |"
            )

    # Verdict
    auc_delta = pa.get("auc_mean", 0) - fa.get("auc_mean", 0)
    lines.extend([
        "",
        "## Verdict",
        "",
        f"- XGBoost AUC delta: **{auc_delta:+.4f}** ({'IMPROVED' if auc_delta > 0.001 else 'COMPARABLE' if abs(auc_delta) < 0.005 else 'WORSE'})",
    ])
    if full_ens and pruned_ens:
        ens_delta = pruned_ens["aggregate"].get("auc_mean", 0) - full_ens["aggregate"].get("auc_mean", 0)
        lines.append(
            f"- Ensemble AUC delta: **{ens_delta:+.4f}** ({'IMPROVED' if ens_delta > 0.001 else 'COMPARABLE' if abs(ens_delta) < 0.005 else 'WORSE'})"
        )
    lines.append(f"- Feature reduction: 31 → 21 ({10} features removed, 32% reduction)")
    lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Loading %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d trades", len(df))

    # Build full and pruned feature sets
    df_full, full_features = build_pipeline_df(df, prune=None)
    df_pruned, pruned_features = build_pipeline_df(df, prune=PRUNE_LIST)
    logger.info("Full features (%d): %s", len(full_features), full_features)
    logger.info("Pruned features (%d): %s", len(pruned_features), pruned_features)
    logger.info("Removed: %s", [f for f in full_features if f not in pruned_features])

    # Run XGBoost benchmarks
    full_xgb = run_xgboost(df_full, full_features, "XGBoost (Full 31)")
    pruned_xgb = run_xgboost(df_pruned, pruned_features, "XGBoost (Pruned 21)")

    # Compare XGBoost AUC
    full_auc = full_xgb["aggregate"]["auc_mean"]
    pruned_auc = pruned_xgb["aggregate"]["auc_mean"]
    delta_auc = pruned_auc - full_auc
    logger.info(
        "XGBoost AUC: full=%.4f → pruned=%.4f (delta=%+.4f)",
        full_auc, pruned_auc, delta_auc,
    )

    # Run Ensemble if pruned AUC improved
    full_ens = None
    pruned_ens = None
    if delta_auc > -0.005:  # Run ensemble if pruned didn't significantly regress
        logger.info("Pruned AUC acceptable — running Ensemble benchmark too")
        full_ens = run_ensemble(df_full, full_features, "Ensemble (Full 31)")
        pruned_ens = run_ensemble(df_pruned, pruned_features, "Ensemble (Pruned 21)")
        ens_full_auc = full_ens["aggregate"]["auc_mean"]
        ens_pruned_auc = pruned_ens["aggregate"]["auc_mean"]
        logger.info(
            "Ensemble AUC: full=%.4f → pruned=%.4f (delta=%+.4f)",
            ens_full_auc, ens_pruned_auc, ens_pruned_auc - ens_full_auc,
        )
    else:
        logger.info("Pruned AUC regressed by %.4f — skipping Ensemble", delta_auc)

    # Save JSON
    output = {
        "metadata": {
            "dataset": str(DATA_PATH),
            "n_trades": len(df),
            "full_features": full_features,
            "pruned_features": pruned_features,
            "removed_features": [f for f in full_features if f not in pruned_features],
        },
        "xgboost_full": {"aggregate": full_xgb["aggregate"], "folds": full_xgb["folds"]},
        "xgboost_pruned": {"aggregate": pruned_xgb["aggregate"], "folds": pruned_xgb["folds"]},
    }
    if full_ens and pruned_ens:
        output["ensemble_full"] = {"aggregate": full_ens["aggregate"], "folds": full_ens["folds"]}
        output["ensemble_pruned"] = {"aggregate": pruned_ens["aggregate"], "folds": pruned_ens["folds"]}

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2, default=str))
    logger.info("JSON → %s", RESULTS_PATH)

    # Generate markdown report
    md = generate_markdown(full_xgb, pruned_xgb, full_ens, pruned_ens, full_features, pruned_features)
    REPORT_PATH.write_text(md)
    logger.info("Report → %s", REPORT_PATH)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PRUNED FEATURE BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"  XGBoost AUC: {full_auc:.4f} (full) → {pruned_auc:.4f} (pruned) | Δ={delta_auc:+.4f}")
    if full_ens and pruned_ens:
        print(f"  Ensemble AUC: {full_ens['aggregate']['auc_mean']:.4f} (full) → "
              f"{pruned_ens['aggregate']['auc_mean']:.4f} (pruned) | "
              f"Δ={pruned_ens['aggregate']['auc_mean'] - full_ens['aggregate']['auc_mean']:+.4f}")
    print(f"  Features: {len(full_features)} → {len(pruned_features)}")
    print(f"  Report: {REPORT_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
