"""
Walk-forward feature importance analysis for COMPASS signal models.

Runs chronological expanding-window validation (same splits as
``compass.walk_forward``) but extracts per-fold feature importances from
XGBoost (gain-based) and sklearn permutation importance. Aggregates across
folds to produce **stable** rankings that reflect real out-of-sample signal
rather than in-sample overfitting.

Key outputs:
    1. Per-feature **mean importance** and **stability score** (inverse CV
       across folds — high stability means the feature is consistently
       important, not just in one fold).
    2. Permutation importance on held-out folds (tests actual AUC impact
       when the feature is shuffled — immune to XGBoost's gain bias toward
       high-cardinality features).
    3. Ablation AUC: walk-forward AUC with each feature removed, quantifying
       the marginal contribution to discriminative power.
    4. Markdown report with tables, pruning recommendations, and plots.

Usage::

    from compass.feature_importance import run_feature_importance_analysis
    report = run_feature_importance_analysis("compass/training_data_combined.csv")
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from compass.feature_pipeline import FeaturePipeline
from compass.walk_forward import (
    CATEGORICAL_FEATURES,
    DATE_COL,
    NUMERIC_FEATURES,
    RETURN_COL,
    TARGET_COL,
    prepare_features,
)
from shared.indicators import sanitize_features

logger = logging.getLogger(__name__)

# ── XGBoost defaults (match SignalModel.train) ────────────────────────────

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


def _build_model():
    """Instantiate the default XGBoost classifier."""
    import xgboost as xgb
    return xgb.XGBClassifier(**_XGB_PARAMS)


# ── Walk-forward feature importance extraction ────────────────────────────


def _walk_forward_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_train_samples: int = 30,
    n_permutation_repeats: int = 10,
) -> Dict[str, Any]:
    """Run walk-forward validation extracting per-fold feature importances.

    Returns dict with:
        fold_results:      List of per-fold dicts (auc, gain_importance, perm_importance)
        feature_cols:      Canonical feature column order
        n_folds:           Number of folds evaluated
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    years = sorted(df[DATE_COL].dt.year.unique())

    if len(years) < 2:
        raise ValueError(f"Need ≥2 years for walk-forward; got {years}")

    # Build full feature matrix once for consistent one-hot columns
    features_full = prepare_features(
        df,
        numeric_features=[c for c in NUMERIC_FEATURES if c in df.columns],
        categorical_features=CATEGORICAL_FEATURES,
    )
    actual_cols = list(features_full.columns)

    fold_results: List[Dict[str, Any]] = []

    for fold_idx in range(len(years) - 1):
        train_years = years[:fold_idx + 1]
        test_year = years[fold_idx + 1]

        train_mask = df[DATE_COL].dt.year.isin(train_years)
        test_mask = df[DATE_COL].dt.year == test_year

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < min_train_samples or n_test < 5:
            logger.info("Fold %d: skipping (train=%d, test=%d)", fold_idx, n_train, n_test)
            continue

        X_train = features_full.loc[train_mask].values
        y_train = df.loc[train_mask, TARGET_COL].values.astype(int)
        X_test = features_full.loc[test_mask].values
        y_test = df.loc[test_mask, TARGET_COL].values.astype(int)

        # Need both classes in train and test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.info("Fold %d: skipping, single class in train or test", fold_idx)
            continue

        # Train fresh model
        model = _build_model()
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        # 1. XGBoost gain-based importance
        gain_raw = model.get_booster().get_score(importance_type="gain")
        gain_importance = np.zeros(len(actual_cols))
        for fname, score in gain_raw.items():
            # XGBoost uses f0, f1, ... naming
            if fname.startswith("f"):
                idx = int(fname[1:])
                if idx < len(actual_cols):
                    gain_importance[idx] = score

        # Normalize to sum to 1
        total = gain_importance.sum()
        if total > 0:
            gain_importance = gain_importance / total

        # 2. Permutation importance on test set
        perm_result = permutation_importance(
            model, X_test, y_test,
            scoring="roc_auc",
            n_repeats=n_permutation_repeats,
            random_state=42,
        )
        perm_importance = perm_result.importances_mean

        fold_results.append({
            "fold": fold_idx,
            "train_years": [int(y) for y in train_years],
            "test_year": int(test_year),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "auc": round(auc, 4),
            "gain_importance": gain_importance,
            "perm_importance": perm_importance,
        })

        logger.info(
            "Fold %d [train %s → test %d]: AUC=%.4f, n_train=%d, n_test=%d",
            fold_idx, train_years, test_year, auc, n_train, n_test,
        )

    return {
        "fold_results": fold_results,
        "feature_cols": actual_cols,
        "n_folds": len(fold_results),
    }


# ── Ablation analysis ────────────────────────────────────────────────────


def _ablation_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    baseline_aucs: List[float],
    min_train_samples: int = 30,
) -> Dict[str, float]:
    """Compute AUC drop when each feature is removed (leave-one-out ablation).

    Runs the walk-forward loop once per feature, excluding that feature each
    time.  The AUC difference from the full-model baseline measures marginal
    contribution.

    Returns dict: {feature_name: mean_auc_drop} (positive = feature helps).
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    years = sorted(df[DATE_COL].dt.year.unique())

    baseline_mean = np.mean(baseline_aucs) if baseline_aucs else 0.5

    results: Dict[str, float] = {}

    for drop_col in feature_cols:
        remaining = [c for c in feature_cols if c != drop_col]
        if not remaining:
            continue

        fold_aucs: List[float] = []

        for fold_idx in range(len(years) - 1):
            train_years = years[:fold_idx + 1]
            test_year = years[fold_idx + 1]

            train_mask = df[DATE_COL].dt.year.isin(train_years)
            test_mask = df[DATE_COL].dt.year == test_year

            if train_mask.sum() < min_train_samples or test_mask.sum() < 5:
                continue

            features_full = prepare_features(
                df,
                numeric_features=[c for c in NUMERIC_FEATURES if c in df.columns],
                categorical_features=CATEGORICAL_FEATURES,
            )

            # Drop the target feature column
            if drop_col in features_full.columns:
                features_ablated = features_full.drop(columns=[drop_col])
            else:
                continue

            X_train = features_ablated.loc[train_mask].values
            y_train = df.loc[train_mask, TARGET_COL].values.astype(int)
            X_test = features_ablated.loc[test_mask].values
            y_test = df.loc[test_mask, TARGET_COL].values.astype(int)

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            model = _build_model()
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_proba = model.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_proba))

        if fold_aucs:
            ablated_mean = np.mean(fold_aucs)
            results[drop_col] = round(baseline_mean - ablated_mean, 4)
        else:
            results[drop_col] = 0.0

    return results


# ── Aggregation ──────────────────────────────────────────────────────────


def aggregate_importances(
    wf_results: Dict[str, Any],
) -> pd.DataFrame:
    """Aggregate per-fold importances into a ranked summary DataFrame.

    Returns DataFrame with columns:
        feature, gain_mean, gain_std, gain_stability,
        perm_mean, perm_std, perm_stability, rank_gain, rank_perm
    """
    folds = wf_results["fold_results"]
    cols = wf_results["feature_cols"]
    n_folds = len(folds)

    if n_folds == 0:
        return pd.DataFrame()

    gains = np.array([f["gain_importance"] for f in folds])
    perms = np.array([f["perm_importance"] for f in folds])

    gain_mean = gains.mean(axis=0)
    gain_std = gains.std(axis=0, ddof=1) if n_folds > 1 else np.zeros(len(cols))

    perm_mean = perms.mean(axis=0)
    perm_std = perms.std(axis=0, ddof=1) if n_folds > 1 else np.zeros(len(cols))

    # Stability: inverse coefficient of variation.
    # High stability = consistent importance across folds.
    # CV = std/mean; stability = 1 / (1 + CV) → in [0, 1].
    def _stability(mean_arr, std_arr):
        result = np.zeros(len(mean_arr))
        for i in range(len(mean_arr)):
            if mean_arr[i] > 0 and std_arr[i] >= 0:
                cv = std_arr[i] / mean_arr[i]
                result[i] = 1.0 / (1.0 + cv)
            elif mean_arr[i] == 0 and std_arr[i] == 0:
                result[i] = 0.0
            else:
                result[i] = 0.0
        return result

    gain_stability = _stability(gain_mean, gain_std)
    perm_stability = _stability(np.maximum(perm_mean, 0), perm_std)

    df = pd.DataFrame({
        "feature": cols,
        "gain_mean": np.round(gain_mean, 6),
        "gain_std": np.round(gain_std, 6),
        "gain_stability": np.round(gain_stability, 4),
        "perm_mean": np.round(perm_mean, 6),
        "perm_std": np.round(perm_std, 6),
        "perm_stability": np.round(perm_stability, 4),
    })

    # Rankings (1 = most important)
    df["rank_gain"] = df["gain_mean"].rank(ascending=False, method="min").astype(int)
    df["rank_perm"] = df["perm_mean"].rank(ascending=False, method="min").astype(int)

    # Composite rank: average of gain and perm ranks
    df["rank_composite"] = ((df["rank_gain"] + df["rank_perm"]) / 2).round(1)

    return df.sort_values("rank_composite").reset_index(drop=True)


# ── Report generation ────────────────────────────────────────────────────


def generate_report(
    summary_df: pd.DataFrame,
    wf_results: Dict[str, Any],
    ablation: Dict[str, float],
    dataset_path: str,
    n_trades: int,
) -> str:
    """Generate a markdown feature importance report."""
    folds = wf_results["fold_results"]
    aucs = [f["auc"] for f in folds]
    mean_auc = np.mean(aucs) if aucs else 0.0
    std_auc = np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0

    lines: List[str] = []

    lines.append("# Feature Importance Analysis — Walk-Forward")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Dataset:** `{dataset_path}` ({n_trades} trades)")
    lines.append(f"**Folds:** {len(folds)} (year-based expanding window)")
    lines.append(f"**Baseline WF AUC:** {mean_auc:.4f} +/- {std_auc:.4f}")
    lines.append("")

    # Per-fold summary
    lines.append("## 1. Walk-Forward Fold Summary")
    lines.append("")
    lines.append("| Fold | Train Years | Test Year | Train N | Test N | AUC |")
    lines.append("|------|-------------|-----------|---------|--------|-----|")
    for f in folds:
        train_str = ", ".join(str(y) for y in f["train_years"])
        lines.append(
            f"| {f['fold']} | {train_str} | {f['test_year']} "
            f"| {f['n_train']} | {f['n_test']} | {f['auc']:.4f} |"
        )
    lines.append("")

    # Feature rankings
    lines.append("## 2. Feature Importance Rankings")
    lines.append("")
    lines.append("Ranked by **composite** (average of XGBoost gain rank and permutation importance rank).")
    lines.append("**Stability** measures consistency across folds (1.0 = identical in every fold).")
    lines.append("")
    lines.append(
        "| Rank | Feature | Gain Mean | Gain Stability | Perm Mean | Perm Stability | Composite |"
    )
    lines.append(
        "|------|---------|-----------|----------------|-----------|----------------|-----------|"
    )
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['rank_composite']:.0f} | {row['feature']} "
            f"| {row['gain_mean']:.4f} | {row['gain_stability']:.2f} "
            f"| {row['perm_mean']:.4f} | {row['perm_stability']:.2f} "
            f"| {row['rank_composite']:.1f} |"
        )
    lines.append("")

    # Signal vs noise classification
    n_features = len(summary_df)
    signal_mask = (summary_df["perm_mean"] > 0.005) | (summary_df["gain_mean"] > 0.02)
    noise_mask = (summary_df["perm_mean"] <= 0.001) & (summary_df["gain_mean"] < 0.01)
    ambiguous_mask = ~signal_mask & ~noise_mask

    signal_features = summary_df[signal_mask]["feature"].tolist()
    noise_features = summary_df[noise_mask]["feature"].tolist()
    ambiguous_features = summary_df[ambiguous_mask]["feature"].tolist()

    lines.append("## 3. Signal vs Noise Classification")
    lines.append("")
    lines.append(f"**Signal features** ({len(signal_features)}): consistently contribute to AUC")
    for f in signal_features:
        row = summary_df[summary_df["feature"] == f].iloc[0]
        lines.append(f"  - `{f}` (gain={row['gain_mean']:.4f}, perm={row['perm_mean']:.4f})")
    lines.append("")

    lines.append(f"**Ambiguous features** ({len(ambiguous_features)}): mixed signal, keep for now")
    for f in ambiguous_features:
        row = summary_df[summary_df["feature"] == f].iloc[0]
        lines.append(f"  - `{f}` (gain={row['gain_mean']:.4f}, perm={row['perm_mean']:.4f})")
    lines.append("")

    lines.append(f"**Noise features** ({len(noise_features)}): candidates for pruning")
    for f in noise_features:
        row = summary_df[summary_df["feature"] == f].iloc[0]
        lines.append(f"  - `{f}` (gain={row['gain_mean']:.4f}, perm={row['perm_mean']:.4f})")
    lines.append("")

    # Ablation analysis
    lines.append("## 4. Ablation Analysis")
    lines.append("")
    lines.append("AUC drop when each feature is **removed** from the model.")
    lines.append("Positive = feature helps; negative = feature hurts (removing it improves AUC).")
    lines.append("")
    lines.append("| Feature | AUC Drop | Verdict |")
    lines.append("|---------|----------|---------|")
    for feat, drop in sorted(ablation.items(), key=lambda x: -x[1]):
        if drop > 0.005:
            verdict = "KEEP"
        elif drop > -0.005:
            verdict = "NEUTRAL"
        else:
            verdict = "PRUNE"
        lines.append(f"| {feat} | {drop:+.4f} | {verdict} |")
    lines.append("")

    # Pruning recommendations
    prune_candidates = [f for f, d in ablation.items() if d <= -0.003]
    neutral_candidates = [f for f in noise_features if f not in prune_candidates]

    lines.append("## 5. Pruning Recommendations")
    lines.append("")

    if prune_candidates:
        lines.append(f"**Recommended to remove** ({len(prune_candidates)} features):")
        lines.append("These features actively **hurt** AUC when present (negative ablation drop):")
        for f in prune_candidates:
            lines.append(f"  - `{f}` (AUC improves by {-ablation.get(f, 0):.4f} when removed)")
    else:
        lines.append("**No features actively hurt AUC** — ablation shows all features are neutral or positive.")

    lines.append("")

    if neutral_candidates:
        lines.append(f"**Consider removing** ({len(neutral_candidates)} features):")
        lines.append("Near-zero importance and permutation impact — likely noise:")
        for f in neutral_candidates:
            lines.append(f"  - `{f}`")

    lines.append("")
    lines.append("## 6. Methodology Notes")
    lines.append("")
    lines.append("- **Walk-forward validation**: Year-based expanding window (no data leakage)")
    lines.append("- **Gain importance**: XGBoost's built-in metric measuring total gain from splits on each feature")
    lines.append("- **Permutation importance**: Measures AUC drop when feature values are shuffled in the test set")
    lines.append("- **Stability score**: 1 / (1 + coefficient_of_variation) across folds")
    lines.append("- **Ablation**: Full walk-forward re-run with each feature excluded")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by compass/feature_importance.py*")

    return "\n".join(lines)


# ── Main entry point ─────────────────────────────────────────────────────


def run_feature_importance_analysis(
    csv_path: str,
    output_dir: str = "experiments",
    min_train_samples: int = 30,
    n_permutation_repeats: int = 10,
    run_ablation: bool = True,
) -> str:
    """Run the full feature importance analysis and generate a report.

    Args:
        csv_path: Path to training data CSV.
        output_dir: Directory for the markdown report.
        min_train_samples: Minimum samples per fold.
        n_permutation_repeats: Repeats for permutation importance.
        run_ablation: Whether to run leave-one-out ablation (slow but informative).

    Returns:
        Path to the generated report file.
    """
    logger.info("Loading training data from %s", csv_path)
    df = pd.read_csv(csv_path)
    n_trades = len(df)
    logger.info("Loaded %d trades, %d columns", n_trades, len(df.columns))

    # Step 1: Walk-forward importance extraction
    logger.info("Running walk-forward importance extraction...")
    wf_results = _walk_forward_importance(
        df,
        feature_cols=NUMERIC_FEATURES,
        min_train_samples=min_train_samples,
        n_permutation_repeats=n_permutation_repeats,
    )
    logger.info("Completed %d folds", wf_results["n_folds"])

    # Step 2: Aggregate across folds
    summary_df = aggregate_importances(wf_results)

    # Step 3: Ablation analysis
    ablation: Dict[str, float] = {}
    if run_ablation and wf_results["n_folds"] > 0:
        logger.info("Running ablation analysis (this may take a while)...")
        baseline_aucs = [f["auc"] for f in wf_results["fold_results"]]
        ablation = _ablation_analysis(
            df, wf_results["feature_cols"], baseline_aucs, min_train_samples,
        )
        logger.info("Ablation complete for %d features", len(ablation))

    # Step 4: Generate report
    report = generate_report(summary_df, wf_results, ablation, csv_path, n_trades)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "feature_importance_report.md"
    report_path.write_text(report)
    logger.info("Report saved to %s", report_path)

    # Also save summary CSV
    csv_out = out_path / "feature_importance_summary.csv"
    summary_df.to_csv(csv_out, index=False)
    logger.info("Summary CSV saved to %s", csv_out)

    return str(report_path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    csv = sys.argv[1] if len(sys.argv) > 1 else "compass/training_data_combined.csv"
    run_feature_importance_analysis(csv)
