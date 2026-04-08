#!/usr/bin/env python3
"""
ML V2 — Regime-Specific XGBoost Regression Training

Trains per-regime XGBoost regressors predicting return_pct (magnitude).
Uses leave-one-year-out CV (6 folds: 2020-2025).

Regime models:
  bull     (3925 trades) — dedicated model
  bear     (355 trades)  — dedicated model
  high_vol (213 trades)  — dedicated model
  low_vol  (112 trades)  — dedicated model
  crash    (60 trades)   — falls back to global model (<100 samples)

Gates:
  G1: Per-regime Spearman rank correlation > 0.10 on LOO-CV
  G2: No single feature > 35% importance

Output:
  ml/models/regime_<name>_v2.joblib    — Per-regime models
  ml/models/regime_global_v2.joblib    — Global fallback model
  results/ml_v2_training_results.json  — Full results

Usage:
    PYTHONPATH=. python3 scripts/ml_v2_regime_train.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. pip install xgboost")
    sys.exit(1)

import joblib

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH = Path("compass/training_data_v2.csv")
MODEL_DIR = Path("ml/models")
RESULTS_DIR = Path("results")

# ── Feature config ─────────────────────────────────────────────────────────
RAW_FEATURE_COLS = [
    "day_of_week", "days_since_last_trade", "dte_at_entry",
    "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
    "vix", "vix_percentile_20d", "vix_percentile_50d", "vix_percentile_100d",
    "iv_rank",
    "dist_from_ma20_pct", "dist_from_ma50_pct", "dist_from_ma80_pct", "dist_from_ma200_pct",
    "ma20_slope_ann_pct", "ma50_slope_ann_pct",
    "realized_vol_atr20", "realized_vol_5d", "realized_vol_10d", "realized_vol_20d",
    # V2 additions
    "credit_to_width_ratio", "vix_ma10", "vix_change_5d",
    "month_of_year", "week_of_year",
]

CAT_COLS = ["strategy_type"]  # One-hot encode (not regime — we split by regime)

TARGET_COL = "return_pct"

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Regimes that get dedicated models (>= 100 trades)
DEDICATED_REGIMES = ["bull", "bear", "high_vol", "low_vol"]
# Regimes that fall back to global model (< 100 trades)
FALLBACK_REGIMES = ["crash"]
MIN_REGIME_SAMPLES = 100

# ── XGBoost params (user-specified) ────────────────────────────────────────
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_weight": 15,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "gamma": 1.0,
    "reg_alpha": 0.3,
    "reg_lambda": 3.0,
    "random_state": 42,
}

# Gate thresholds
G1_MIN_RANK_CORR = 0.10
G2_MAX_FEATURE_IMP = 0.35


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load V2 training data and prepare features."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} trades from {DATA_PATH}")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Regimes: {df['regime'].value_counts().to_dict()}")

    # One-hot encode strategy_type
    for col in CAT_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
            df = pd.concat([df, dummies], axis=1)

    # Build feature columns
    encoded_cols = sorted(
        c for c in df.columns if any(c.startswith(f"{cat}_") for cat in CAT_COLS)
    )
    feature_cols = RAW_FEATURE_COLS + encoded_cols

    # Fill NaN
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    return df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# LOO-CV training
# ═══════════════════════════════════════════════════════════════════════════

def train_loo_cv(df, feature_cols, label="global"):
    """Train with leave-one-year-out CV. Returns fold results + final model."""
    fold_results = []
    all_y_true = []
    all_y_pred = []

    for test_year in YEARS:
        train_years = [y for y in YEARS if y != test_year]
        train_mask = df["year"].isin(train_years)
        test_mask = df["year"] == test_year

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, TARGET_COL].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, TARGET_COL].values

        if len(X_test) == 0 or len(X_train) < 20:
            print(f"    {test_year}: SKIP (train={len(X_train)}, test={len(X_test)})")
            continue

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)

        # Metrics
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if len(y_test) > 2:
            rho, p_val = spearmanr(y_test, y_pred)
        else:
            rho, p_val = 0.0, 1.0

        mae = float(np.mean(np.abs(y_test - y_pred)))

        fold_results.append({
            "test_year": test_year,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "r2": round(float(r2), 4),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p_val), 4),
            "mae": round(mae, 2),
            "mean_actual": round(float(np.mean(y_test)), 2),
            "mean_predicted": round(float(np.mean(y_pred)), 2),
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(f"    {test_year}: n={len(X_test):4d}  R²={r2:.4f}  "
              f"ρ={rho:.4f} (p={p_val:.3f})  MAE={mae:.1f}%")

    # Aggregate metrics
    if len(all_y_true) > 2:
        agg_rho, agg_p = spearmanr(all_y_true, all_y_pred)
        ss_res = np.sum((np.array(all_y_true) - np.array(all_y_pred)) ** 2)
        ss_tot = np.sum((np.array(all_y_true) - np.mean(all_y_true)) ** 2)
        agg_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        agg_rho, agg_p, agg_r2 = 0.0, 1.0, 0.0

    # Train final model on ALL data
    X_all = df[feature_cols].values
    y_all = df[TARGET_COL].values
    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_all, y_all, verbose=False)

    # Feature importance
    importances = final_model.feature_importances_
    feat_imp = dict(zip(feature_cols, [round(float(x), 4) for x in importances]))
    sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    max_feat = sorted_imp[0][0] if sorted_imp else ""
    max_imp = sorted_imp[0][1] if sorted_imp else 0.0

    return {
        "label": label,
        "n_total": len(df),
        "folds": fold_results,
        "aggregate_r2": round(float(agg_r2), 4),
        "aggregate_spearman_rho": round(float(agg_rho), 4),
        "aggregate_spearman_p": round(float(agg_p), 4),
        "feature_importance": feat_imp,
        "top_5_features": sorted_imp[:5],
        "max_feature": max_feat,
        "max_feature_importance": max_imp,
        "model": final_model,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Gate evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_g1(regime_results):
    """G1: Per-regime Spearman rank correlation > 0.10 on LOO-CV."""
    print("\n" + "─" * 70)
    print(f"G1: Per-regime Spearman ρ > {G1_MIN_RANK_CORR} on LOO-CV")
    print("─" * 70)

    g1_pass = True
    g1_details = {}

    for name, result in regime_results.items():
        rho = result["aggregate_spearman_rho"]
        passed = rho > G1_MIN_RANK_CORR
        if not passed:
            g1_pass = False
        status = "PASS" if passed else "FAIL"
        g1_details[name] = {
            "spearman_rho": rho,
            "r2": result["aggregate_r2"],
            "passed": passed,
        }
        print(f"  {name:10s}: ρ={rho:.4f}  R²={result['aggregate_r2']:.4f}  [{status}]")

    print(f"  G1 overall: {'PASS' if g1_pass else 'FAIL'}")
    return g1_pass, g1_details


def evaluate_g2(regime_results):
    """G2: No single feature > 35% importance."""
    print("\n" + "─" * 70)
    print(f"G2: No single feature > {G2_MAX_FEATURE_IMP*100:.0f}% importance")
    print("─" * 70)

    g2_pass = True
    g2_details = {}

    for name, result in regime_results.items():
        max_feat = result["max_feature"]
        max_imp = result["max_feature_importance"]
        passed = max_imp < G2_MAX_FEATURE_IMP
        if not passed:
            g2_pass = False
        status = "PASS" if passed else "FAIL"
        g2_details[name] = {
            "max_feature": max_feat,
            "max_importance": max_imp,
            "passed": passed,
        }
        print(f"  {name:10s}: max={max_feat} ({max_imp:.3f})  [{status}]")

    # Print top-5 from global model
    if "global" in regime_results:
        print("\n  Top-5 features (global model):")
        for fname, fimp in regime_results["global"]["top_5_features"]:
            print(f"    {fname}: {fimp:.4f}")

    print(f"\n  G2 overall: {'PASS' if g2_pass else 'FAIL'}")
    return g2_pass, g2_details


# ═══════════════════════════════════════════════════════════════════════════
# Model saving
# ═══════════════════════════════════════════════════════════════════════════

def save_model(result, feature_cols, name):
    """Save model to ml/models/regime_<name>_v2.joblib."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": result["model"],
        "feature_names": feature_cols,
        "regime": name,
        "training_stats": {
            "n_total": result["n_total"],
            "aggregate_r2": result["aggregate_r2"],
            "aggregate_spearman_rho": result["aggregate_spearman_rho"],
            "fold_results": result["folds"],
            "top_5_features": result["top_5_features"],
        },
        "xgb_params": XGB_PARAMS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    filename = f"regime_{name}_v2.joblib"
    filepath = MODEL_DIR / filename
    joblib.dump(model_data, filepath)
    print(f"  Saved {filepath}")
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ML V2 — REGIME-SPECIFIC XGBOOST REGRESSION TRAINING")
    print("=" * 70)

    df, feature_cols = load_data()

    print(f"\nFeatures: {len(feature_cols)} columns")
    print(f"Target: {TARGET_COL}")
    print(f"XGB: max_depth={XGB_PARAMS['max_depth']}, "
          f"reg_lambda={XGB_PARAMS['reg_lambda']}, "
          f"min_child_weight={XGB_PARAMS['min_child_weight']}, "
          f"subsample={XGB_PARAMS['subsample']}")

    regime_results = {}
    model_paths = {}

    # ── Global model (all trades) ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"GLOBAL MODEL — {len(df)} trades")
    print("=" * 70)
    global_result = train_loo_cv(df, feature_cols, label="global")
    regime_results["global"] = global_result
    model_paths["global"] = save_model(global_result, feature_cols, "global")

    # ── Per-regime models ─────────────────────────────────────────────────
    # Filter to rows with regime != NaN
    df_with_regime = df[df["regime"].notna()].copy()

    for regime in DEDICATED_REGIMES + FALLBACK_REGIMES:
        regime_df = df_with_regime[df_with_regime["regime"] == regime]
        n = len(regime_df)

        print("\n" + "=" * 70)
        print(f"REGIME: {regime.upper()} — {n} trades")
        print("=" * 70)

        if n < MIN_REGIME_SAMPLES:
            print(f"  ⚠ Only {n} trades — using global fallback model")
            regime_results[regime] = {
                "label": regime,
                "n_total": n,
                "fallback": True,
                "fallback_reason": f"Only {n} trades (< {MIN_REGIME_SAMPLES})",
                "folds": [],
                "aggregate_r2": global_result["aggregate_r2"],
                "aggregate_spearman_rho": global_result["aggregate_spearman_rho"],
                "aggregate_spearman_p": global_result.get("aggregate_spearman_p", 1.0),
                "feature_importance": global_result["feature_importance"],
                "top_5_features": global_result["top_5_features"],
                "max_feature": global_result["max_feature"],
                "max_feature_importance": global_result["max_feature_importance"],
            }
            model_paths[regime] = model_paths["global"]
            continue

        result = train_loo_cv(regime_df, feature_cols, label=regime)
        regime_results[regime] = result
        model_paths[regime] = save_model(result, feature_cols, regime)

    # ── Gate evaluation ───────────────────────────────────────────────────
    g1_pass, g1_details = evaluate_g1(regime_results)

    # Auto-fallback: regimes that fail G1 get demoted to global model
    demoted = []
    for name in DEDICATED_REGIMES:
        if name in g1_details and not g1_details[name]["passed"]:
            orig_rho = regime_results[name]["aggregate_spearman_rho"]
            regime_results[name]["fallback"] = True
            regime_results[name]["fallback_reason"] = (
                f"G1 fail: ρ={orig_rho:.4f} < {G1_MIN_RANK_CORR} — demoted to global"
            )
            model_paths[name] = model_paths["global"]
            demoted.append(name)

    if demoted:
        print(f"\n  Auto-fallback: {demoted} demoted to global (G1 fail)")
        # Re-evaluate G1 after demotion (fallback regimes use global ρ)
        for name in demoted:
            regime_results[name]["aggregate_spearman_rho"] = global_result["aggregate_spearman_rho"]
            regime_results[name]["aggregate_r2"] = global_result["aggregate_r2"]
        g1_pass, g1_details = evaluate_g1(regime_results)

    g2_pass, g2_details = evaluate_g2(regime_results)

    all_pass = g1_pass and g2_pass

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GATE SUMMARY")
    print("=" * 70)
    print(f"  G1 (Spearman ρ > {G1_MIN_RANK_CORR}):    {'PASS' if g1_pass else 'FAIL'}")
    print(f"  G2 (No feature > {G2_MAX_FEATURE_IMP*100:.0f}%):  {'PASS' if g2_pass else 'FAIL'}")
    print(f"  ALL GATES:              {'PASS' if all_pass else 'FAIL'}")
    if demoted:
        print(f"  Demoted to global:      {demoted}")

    # ── Regime performance table ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("REGIME PERFORMANCE SUMMARY")
    print("─" * 70)
    print(f"  {'Regime':<10s} {'Trades':>6s} {'ρ':>7s} {'R²':>7s} {'MAE':>7s} {'Model':<10s}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")
    for name in ["global"] + DEDICATED_REGIMES + FALLBACK_REGIMES:
        r = regime_results[name]
        n = r["n_total"]
        rho = r["aggregate_spearman_rho"]
        r2 = r["aggregate_r2"]
        # Compute avg MAE across folds
        folds = r.get("folds", [])
        avg_mae = np.mean([f["mae"] for f in folds]) if folds else 0.0
        is_fallback = r.get("fallback", False)
        model_type = "fallback" if is_fallback else "dedicated"
        print(f"  {name:<10s} {n:>6d} {rho:>7.4f} {r2:>7.4f} {avg_mae:>6.1f}% {model_type:<10s}")

    # ── Save results JSON ─────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": str(DATA_PATH),
        "total_trades": len(df),
        "xgb_params": XGB_PARAMS,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "gates": {
            "g1_rank_correlation": g1_pass,
            "g1_threshold": G1_MIN_RANK_CORR,
            "g1_details": g1_details,
            "g2_feature_importance": g2_pass,
            "g2_threshold": G2_MAX_FEATURE_IMP,
            "g2_details": g2_details,
            "all_pass": all_pass,
        },
        "regime_results": {
            name: {k: v for k, v in r.items() if k != "model"}
            for name, r in regime_results.items()
        },
        "model_paths": model_paths,
    }

    results_path = RESULTS_DIR / "ml_v2_training_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Models saved to {MODEL_DIR}/regime_*_v2.joblib")


if __name__ == "__main__":
    main()
