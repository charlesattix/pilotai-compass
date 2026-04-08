#!/usr/bin/env python3
"""
CS-Only Clean vs Legacy Feature Pipeline Analysis
===================================================
Filters training data to credit spread (CS) trades only, then compares
the clean feature pipeline (z-scored, ratio-based) against the legacy
pipeline (raw prices, 0-fill imputation) on walk-forward AUC and accuracy.

Key question: when the model can't lean on strategy_type_CS as a crutch,
how well does each pipeline find signal among CS trades alone?

Outputs:
  reports/cs_only_clean_analysis.json   — structured results
  reports/cs_only_clean_analysis.md     — markdown summary
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("cs_analysis")

CSV_PATH = ROOT / "compass" / "training_data_combined.csv"
REPORTS_DIR = ROOT / "reports"
JSON_OUT = REPORTS_DIR / "cs_only_clean_analysis.json"
MD_OUT = REPORTS_DIR / "cs_only_clean_analysis.md"


# ── Legacy feature list (matches the all-strategy retroactive run) ────────
LEGACY_NUMERIC = [
    "dte_at_entry", "day_of_week", "days_since_last_trade",
    "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
    "vix", "vix_percentile_20d", "vix_percentile_50d", "vix_percentile_100d",
    "iv_rank", "spy_price",
    "dist_from_ma20_pct", "dist_from_ma50_pct", "dist_from_ma80_pct", "dist_from_ma200_pct",
    "ma20_slope_ann_pct", "ma50_slope_ann_pct",
    "realized_vol_atr20", "realized_vol_5d", "realized_vol_10d", "realized_vol_20d",
]

# Categoricals to one-hot (for legacy: regime + spread_type, no strategy_type since CS-only)
LEGACY_CATEGORICALS = ["regime", "spread_type"]


def build_legacy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the legacy feature matrix — raw values, 0-fill, no normalization."""
    parts = []

    # Numerics
    num = pd.DataFrame(index=df.index)
    for col in LEGACY_NUMERIC:
        if col in df.columns:
            num[col] = df[col].fillna(0.0).astype(float)
        else:
            num[col] = 0.0
    parts.append(num)

    # Categoricals
    for col in LEGACY_CATEGORICALS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            parts.append(dummies)

    from shared.indicators import sanitize_features
    result = pd.concat(parts, axis=1)
    result[:] = sanitize_features(result.values.astype(np.float64))
    return result


def build_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the clean feature matrix using FeaturePipeline.

    For CS-only, we exclude strategy_type from categoricals (it's constant)
    and keep regime + spread_type.
    """
    from compass.feature_pipeline import FeaturePipeline
    pipeline = FeaturePipeline(categorical_features=["regime", "spread_type"])
    return pipeline.transform(df)


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train XGBoost and return metrics."""
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.75, colsample_bytree=0.7,
            gamma=2, min_child_weight=8, reg_alpha=0.5, reg_lambda=2.0,
            eval_metric="logloss", random_state=42,
            use_label_encoder=False,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.75, random_state=42,
        )

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score, brier_score_loss, precision_score,
        recall_score, roc_auc_score,
    )

    model.fit(X_train, y_train)

    # Calibrate
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), cv=3, method="sigmoid")
    except ImportError:
        calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
    calibrated.fit(X_train, y_train)

    # Predict on test
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_prob)

    # Feature importance
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]

    # Confidence analysis: what would gating do?
    confidence = np.abs(y_prob - 0.5) * 2
    threshold = 0.30
    gated_mask = confidence >= threshold
    n_gated_out = (~gated_mask).sum()
    if gated_mask.sum() > 0:
        gated_acc = accuracy_score(y_test[gated_mask], y_pred[gated_mask])
        gated_wr = y_test[gated_mask].mean()
    else:
        gated_acc = 0.0
        gated_wr = 0.0

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "brier": float(brier),
        "n_gated_out": int(n_gated_out),
        "gated_accuracy": float(gated_acc),
        "gated_win_rate": float(gated_wr),
        "top_5_features": [
            {"name": str(X_train.columns[i]), "importance": float(importances[i])}
            for i in top_idx
        ],
        "model": model,
        "probabilities": y_prob.tolist(),
    }


def run_walk_forward(df: pd.DataFrame, pipeline_name: str, feature_builder):
    """Run chronological walk-forward validation on CS-only data.

    Folds:
      - Train on all years up to Y-1, test on year Y
      - Only years with enough test data (>=10 trades)
    """
    years = sorted(df["year"].unique())
    # Need at least 2 years of training data before testing
    test_years = [y for y in years if (df["year"] < y).sum() >= 30 and (df["year"] == y).sum() >= 10]

    log.info("\n  %s pipeline (%d test folds: %s)", pipeline_name, len(test_years), test_years)

    fold_results = []
    all_probs = []
    all_true = []

    for test_year in test_years:
        train_mask = df["year"] < test_year
        test_mask = df["year"] == test_year

        train_df = df[train_mask]
        test_df = df[test_mask]

        # Build features on full data (sorted by date), then split
        full_df = pd.concat([train_df, test_df])
        full_features = feature_builder(full_df)

        X_train = full_features.iloc[:len(train_df)]
        X_test = full_features.iloc[len(train_df):]
        y_train = train_df["win"].values.astype(int)
        y_test = test_df["win"].values.astype(int)

        result = train_and_evaluate(X_train, y_train, X_test, y_test)
        all_probs.extend(result["probabilities"])
        all_true.extend(y_test.tolist())

        fold_info = {
            "test_year": int(test_year),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
            "train_win_rate": float(y_train.mean()),
            "test_win_rate": float(y_test.mean()),
            "auc": round(result["auc"], 4),
            "accuracy": round(result["accuracy"], 4),
            "precision": round(result["precision"], 4),
            "recall": round(result["recall"], 4),
            "brier": round(result["brier"], 4),
            "n_gated_out": result["n_gated_out"],
            "gated_accuracy": round(result["gated_accuracy"], 4),
            "gated_win_rate": round(result["gated_win_rate"], 4),
            "top_5_features": result["top_5_features"],
        }
        fold_results.append(fold_info)

        log.info("    %d: train=%d test=%d | AUC=%.4f Acc=%.4f Brier=%.4f | "
                 "gated: %d/%d kept (WR=%.1f%%)",
                 test_year, len(X_train), len(X_test),
                 result["auc"], result["accuracy"], result["brier"],
                 len(X_test) - result["n_gated_out"], len(X_test),
                 result["gated_win_rate"] * 100)

    # Pooled calibration from all OOS predictions
    all_probs_arr = np.array(all_probs)
    all_true_arr = np.array(all_true)
    calibration_bins = []
    for lo in np.arange(0, 1.0, 0.1):
        hi = lo + 0.1
        mask = (all_probs_arr >= lo) & (all_probs_arr < hi)
        if mask.sum() >= 3:
            calibration_bins.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "predicted_mean": round(float(all_probs_arr[mask].mean()), 4),
                "actual_fraction": round(float(all_true_arr[mask].mean()), 4),
                "count": int(mask.sum()),
            })

    avg_auc = round(np.mean([f["auc"] for f in fold_results]), 4)
    avg_acc = round(np.mean([f["accuracy"] for f in fold_results]), 4)
    avg_brier = round(np.mean([f["brier"] for f in fold_results]), 4)

    return {
        "pipeline": pipeline_name,
        "n_features": fold_results[0]["n_features"] if fold_results else 0,
        "folds": fold_results,
        "avg_auc": avg_auc,
        "avg_accuracy": avg_acc,
        "avg_brier": avg_brier,
        "calibration_bins": calibration_bins,
    }


def generate_markdown(results: dict) -> str:
    """Generate a concise markdown summary."""
    legacy = results["legacy_pipeline"]
    clean = results["clean_pipeline"]
    cs = results["cs_data_summary"]

    md = f"""# CS-Only Clean vs Legacy Feature Pipeline Analysis

**Generated:** {results['generated']}
**Data:** {cs['n_trades']} credit spread trades, {cs['year_range']}
**Baseline win rate:** {cs['win_rate']:.1%} ({cs['n_wins']}/{cs['n_trades']})
**Total P&L:** ${cs['total_pnl']:,.0f}

## Why CS-Only?

The all-strategy model's top feature was `strategy_type_CS` — it learned "CS wins, SS loses"
rather than finding signal within CS trades. This analysis removes that crutch by training
exclusively on CS trades, forcing the model to find real predictive features.

## Head-to-Head Results

| Metric | Legacy Pipeline | Clean Pipeline | Delta |
|--------|:--------------:|:--------------:|:-----:|
| Avg AUC | {legacy['avg_auc']:.4f} | {clean['avg_auc']:.4f} | {clean['avg_auc'] - legacy['avg_auc']:+.4f} |
| Avg Accuracy | {legacy['avg_accuracy']:.4f} | {clean['avg_accuracy']:.4f} | {clean['avg_accuracy'] - legacy['avg_accuracy']:+.4f} |
| Avg Brier | {legacy['avg_brier']:.4f} | {clean['avg_brier']:.4f} | {clean['avg_brier'] - legacy['avg_brier']:+.4f} |
| Features | {legacy['n_features']} | {clean['n_features']} | {clean['n_features'] - legacy['n_features']:+d} |

## Per-Fold Detail

### Legacy Pipeline ({legacy['n_features']} features)
| Year | Train | Test | AUC | Accuracy | Brier | Gated WR |
|------|------:|-----:|----:|---------:|------:|---------:|
"""
    for f in legacy["folds"]:
        md += f"| {f['test_year']} | {f['n_train']} | {f['n_test']} | {f['auc']:.4f} | {f['accuracy']:.4f} | {f['brier']:.4f} | {f['gated_win_rate']:.1%} |\n"

    md += f"""
### Clean Pipeline ({clean['n_features']} features)
| Year | Train | Test | AUC | Accuracy | Brier | Gated WR |
|------|------:|-----:|----:|---------:|------:|---------:|
"""
    for f in clean["folds"]:
        md += f"| {f['test_year']} | {f['n_train']} | {f['n_test']} | {f['auc']:.4f} | {f['accuracy']:.4f} | {f['brier']:.4f} | {f['gated_win_rate']:.1%} |\n"

    # Best fold feature importances
    md += "\n## Top Features (CS-Only)\n\n"
    for pipeline_data, name in [(legacy, "Legacy"), (clean, "Clean")]:
        best_fold = max(pipeline_data["folds"], key=lambda f: f["auc"])
        md += f"### {name} Pipeline (best fold: {best_fold['test_year']})\n"
        for i, feat in enumerate(best_fold["top_5_features"], 1):
            bar = "#" * int(feat["importance"] * 100)
            md += f"{i}. `{feat['name']}` — {feat['importance']:.4f} {bar}\n"
        md += "\n"

    # Calibration
    md += "## Calibration (Pooled OOS)\n\n"
    for pipeline_data, name in [(legacy, "Legacy"), (clean, "Clean")]:
        if pipeline_data.get("calibration_bins"):
            md += f"### {name}\n| Bin | Predicted | Actual | Count |\n|-----|:---------:|:------:|------:|\n"
            for b in pipeline_data["calibration_bins"]:
                md += f"| {b['bin']} | {b['predicted_mean']:.3f} | {b['actual_fraction']:.3f} | {b['count']} |\n"
            md += "\n"

    # Gating analysis
    md += "## ML Gating Impact (30% confidence threshold)\n\n"
    md += "| Year | Pipeline | Trades In | Gated Out | Kept WR | Base WR |\n"
    md += "|------|----------|:---------:|:---------:|:-------:|:-------:|\n"
    for pipeline_data, name in [(legacy, "Legacy"), (clean, "Clean")]:
        for f in pipeline_data["folds"]:
            md += (f"| {f['test_year']} | {name} | {f['n_test']} | {f['n_gated_out']} | "
                   f"{f['gated_win_rate']:.1%} | {f['test_win_rate']:.1%} |\n")

    md += f"\n## Verdict\n\n**{results['verdict']}**\n\n"
    md += f"AUC delta (clean - legacy): {results['auc_delta_avg']:+.4f}\n"

    return md


def main():
    log.info("=" * 70)
    log.info("  CS-Only Clean vs Legacy Feature Pipeline Analysis")
    log.info("=" * 70)

    # Load and filter
    df = pd.read_csv(CSV_PATH)
    df["year"] = df["year"].astype(int)
    cs = df[df["strategy_type"] == "CS"].copy().sort_values("entry_date").reset_index(drop=True)

    log.info("\nData: %d CS trades (of %d total), years %d-%d",
             len(cs), len(df), cs["year"].min(), cs["year"].max())
    log.info("CS win rate: %.1f%% | Total PnL: $%+.0f",
             cs["win"].mean() * 100, cs["pnl"].sum())

    per_year = cs.groupby("year").agg(
        trades=("win", "count"),
        wins=("win", "sum"),
        pnl=("pnl", "sum"),
    )
    per_year["wr"] = per_year["wins"] / per_year["trades"]
    for year, row in per_year.iterrows():
        log.info("  %d: %d trades, WR=%.1f%%, PnL=$%+.0f",
                 year, row["trades"], row["wr"] * 100, row["pnl"])

    cs_summary = {
        "n_trades": len(cs),
        "n_wins": int(cs["win"].sum()),
        "win_rate": float(cs["win"].mean()),
        "total_pnl": float(cs["pnl"].sum()),
        "year_range": f"{cs['year'].min()}-{cs['year'].max()}",
        "per_year": {
            str(y): {"trades": int(r["trades"]), "wins": int(r["wins"]),
                      "pnl": round(float(r["pnl"]), 2), "win_rate": round(float(r["wr"]), 4)}
            for y, r in per_year.iterrows()
        },
    }

    # Run walk-forward for each pipeline
    log.info("\nRunning walk-forward validation...")

    legacy_results = run_walk_forward(cs, "legacy", build_legacy_features)
    clean_results = run_walk_forward(cs, "clean", build_clean_features)

    auc_delta = clean_results["avg_auc"] - legacy_results["avg_auc"]
    acc_delta = clean_results["avg_accuracy"] - legacy_results["avg_accuracy"]
    brier_delta = clean_results["avg_brier"] - legacy_results["avg_brier"]

    if auc_delta > 0.005:
        verdict = f"Clean pipeline wins on CS-only trades (AUC +{auc_delta:.4f})"
    elif auc_delta < -0.005:
        verdict = f"Legacy pipeline wins on CS-only trades (AUC {auc_delta:+.4f})"
    else:
        verdict = f"Pipelines are equivalent on CS-only trades (AUC delta {auc_delta:+.4f})"

    results = {
        "generated": datetime.utcnow().isoformat(),
        "data_source": str(CSV_PATH),
        "filter": "strategy_type == CS",
        "cs_data_summary": cs_summary,
        "legacy_pipeline": {
            "description": "Raw features: spy_price, vix (absolute), contracts (raw), 0-fill imputation",
            **legacy_results,
        },
        "clean_pipeline": {
            "description": "Z-scored prices, ratio features, domain-aware imputation, log-contracts",
            **clean_results,
        },
        "auc_delta_avg": round(auc_delta, 4),
        "accuracy_delta_avg": round(acc_delta, 4),
        "brier_delta_avg": round(brier_delta, 4),
        "verdict": verdict,
    }

    # Save JSON (strip non-serializable model objects)
    save_results = json.loads(json.dumps(results, default=str))
    REPORTS_DIR.mkdir(exist_ok=True)
    JSON_OUT.write_text(json.dumps(save_results, indent=2))
    log.info("\nJSON → %s", JSON_OUT)

    # Save markdown
    md = generate_markdown(results)
    MD_OUT.write_text(md)
    log.info("MD   → %s", MD_OUT)

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("  RESULTS (CS-only, %d trades)", len(cs))
    log.info("=" * 70)
    log.info("  %-18s AUC=%.4f  Acc=%.4f  Brier=%.4f  (%d features)",
             "Legacy:", legacy_results["avg_auc"], legacy_results["avg_accuracy"],
             legacy_results["avg_brier"], legacy_results["n_features"])
    log.info("  %-18s AUC=%.4f  Acc=%.4f  Brier=%.4f  (%d features)",
             "Clean:", clean_results["avg_auc"], clean_results["avg_accuracy"],
             clean_results["avg_brier"], clean_results["n_features"])
    log.info("  %-18s AUC %+.4f  Acc %+.4f  Brier %+.4f",
             "Delta:", auc_delta, acc_delta, brier_delta)
    log.info("  Verdict: %s", verdict)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
