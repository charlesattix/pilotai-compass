#!/usr/bin/env python3
"""
ML V2 Improved Walk-Forward Blind Test — 5 Targeted Fixes

Root cause of original FAIL: G6 walk-forward ratio 0.34 (calibration +65% vs blind +22%).
Five fixes to reduce calibration-to-blind gap:

  Fix 1: Linear sizing (replace 5-tier lookup) — only 2 params
  Fix 2: LOO-CV sizing calibration — honest held-out calibration avg
  Fix 3: Production model retrained on all 6 years (post-validation only)
  Fix 4: Merge bear + high_vol → "defensive" regime (more samples)
  Fix 5: Ensemble of 5 models per regime (reduce single-model variance)

Output: results/ml_v2_improved_blind.json

Usage:
    PYTHONPATH=. python3 scripts/ml_v2_improved_blind.py
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
from scipy.stats import spearmanr

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. pip install xgboost")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from compass.collect_training_data import (
    enrich_trades,
    run_year_backtest_exp401,
    _load_full_market_data,
    _compute_ma,
    STARTING_CAPITAL,
)
from scripts.ml_v2_backtest import (
    add_v2_features,
    compute_year_stats,
    compute_sharpe,
    V2_FEATURE_COLS,
)
from scripts.ml_v2_regime_train import (
    XGB_PARAMS,
    RAW_FEATURE_COLS,
    MIN_REGIME_SAMPLES,
    G1_MIN_RANK_CORR,
    CAT_COLS,
    TARGET_COL,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "compass" / "training_data_v2.csv"
RESULTS_PATH = ROOT / "results" / "ml_v2_improved_blind.json"
ORIGINAL_RESULTS_PATH = ROOT / "results" / "ml_v2_walkforward_blind.json"

TRAIN_YEARS = [2020, 2021, 2022, 2023]
BLIND_YEARS = [2024, 2025]
ALL_YEARS = TRAIN_YEARS + BLIND_YEARS

# ── Fix 1: Linear sizing constants ────────────────────────────────────────
SIZING_BASE = 0.25
SIZING_SCALE = 0.75

# ── Fix 4: Regime remapping ──────────────────────────────────────────────
REGIME_REMAP = {"bear": "defensive", "high_vol": "defensive"}
DEDICATED_REGIMES_V2 = ["bull", "defensive", "low_vol"]
FALLBACK_REGIMES_V2 = ["crash"]

# ── Fix 5: Ensemble seeds ────────────────────────────────────────────────
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337]


# ═══════════════════════════════════════════════════════════════════════════
# Fix 4: Regime remapping
# ═══════════════════════════════════════════════════════════════════════════

def remap_regime(regime: str) -> str:
    """Map bear/high_vol → 'defensive', others unchanged."""
    if pd.isna(regime):
        return regime
    return REGIME_REMAP.get(regime, regime)


# ═══════════════════════════════════════════════════════════════════════════
# Fix 5: Ensemble training
# ═══════════════════════════════════════════════════════════════════════════

def train_ensemble(X: np.ndarray, y: np.ndarray, seeds: List[int]) -> list:
    """Train N XGBoost models with different random_state, return list."""
    models = []
    for seed in seeds:
        params = dict(XGB_PARAMS)
        params["random_state"] = seed
        model = xgb.XGBRegressor(**params)
        model.fit(X, y, verbose=False)
        models.append(model)
    return models


def predict_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    """Average predictions from N models."""
    preds = np.column_stack([m.predict(X) for m in models])
    return preds.mean(axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# Fix 1: Linear sizing
# ═══════════════════════════════════════════════════════════════════════════

def linear_sizing(predictions: pd.Series, calibration_preds: np.ndarray) -> pd.Series:
    """Map predictions to multipliers via percentile rank in calibration CDF.

    multiplier = SIZING_BASE + SIZING_SCALE * percentile_rank
    Returns multipliers in [SIZING_BASE, SIZING_BASE + SIZING_SCALE].
    """
    sorted_cal = np.sort(calibration_preds)
    n_cal = len(sorted_cal)

    def _percentile_rank(pred):
        # Fraction of calibration predictions <= this prediction
        rank = np.searchsorted(sorted_cal, pred, side="right")
        return rank / n_cal

    percentiles = predictions.apply(_percentile_rank)
    multipliers = SIZING_BASE + SIZING_SCALE * percentiles
    return multipliers


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode and build feature column list."""
    for col in CAT_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
            df = pd.concat([df, dummies], axis=1)

    encoded_cols = sorted(
        c for c in df.columns if any(c.startswith(f"{cat}_") for cat in CAT_COLS)
    )
    feature_cols = RAW_FEATURE_COLS + encoded_cols

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    return df, feature_cols


def load_training_csv(years: List[int]) -> Tuple[pd.DataFrame, List[str]]:
    """Load training CSV, filter to specified years, prepare features."""
    df = pd.read_csv(DATA_PATH)
    total = len(df)
    df = df[df["year"].isin(years)].copy()

    # Remap regimes (Fix 4)
    df["regime"] = df["regime"].apply(remap_regime)

    print(f"  Loaded {total} total trades, filtered to {len(df)} (years {years})")
    print(f"  Regimes after remap: {df['regime'].value_counts().to_dict()}")

    df, feature_cols = prepare_features(df)
    return df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# Score trades with ensemble models
# ═══════════════════════════════════════════════════════════════════════════

def score_trades_ensemble(df: pd.DataFrame, models: Dict,
                          feature_cols: List[str]) -> pd.Series:
    """Score each trade with its regime-specific ensemble.

    models: {regime: ensemble_list} where ensemble_list is list of XGBRegressors.
    Falls back to 'global' ensemble if regime not found.
    """
    predictions = pd.Series(0.0, index=df.index)

    for i, row in df.iterrows():
        regime = row.get("regime", None)
        if regime and regime in models:
            ensemble = models[regime]
        elif "global" in models:
            ensemble = models["global"]
        else:
            continue

        features = []
        for col in feature_cols:
            val = row.get(col, 0.0)
            features.append(float(val) if pd.notna(val) else 0.0)

        X = np.array([features])
        pred = predict_ensemble(ensemble, X)[0]
        predictions.at[i] = float(pred)

    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# Fix 2 + 5: LOO-CV training with ensembles
# ═══════════════════════════════════════════════════════════════════════════

def train_regime_ensembles_loo(
    df_train: pd.DataFrame,
    train_years: List[int],
    feature_cols: List[str],
) -> Tuple[Dict[int, pd.Series], Dict[int, Dict], Dict[str, list], Dict]:
    """LOO-CV with ensemble models per regime.

    For each fold year k:
      - Train global + regime ensembles on years != k
      - Score year k trades → store LOO predictions
      - Save fold models for honest calibration scoring

    Also train final ensemble on all train_years.

    Returns:
        fold_predictions: {year: pd.Series of predictions (CSV trades)}
        fold_models_by_year: {year: {regime: ensemble_list}} for honest scoring
        final_models: {regime: ensemble_list} trained on all train_years
        training_info: dict with fold stats and routing
    """
    fold_predictions = {}
    fold_models_by_year = {}
    fold_stats = {}
    training_info = {"folds": {}, "routing": {}}

    for test_year in train_years:
        print(f"\n  LOO fold: held-out year {test_year}")
        train_mask = df_train["year"] != test_year
        test_mask = df_train["year"] == test_year
        df_fold_train = df_train[train_mask]
        df_fold_test = df_train[test_mask].copy()

        if len(df_fold_test) == 0:
            print(f"    SKIP: no trades in {test_year}")
            continue

        # Train global ensemble
        X_train_all = df_fold_train[feature_cols].values
        y_train_all = df_fold_train[TARGET_COL].values
        global_ensemble = train_ensemble(X_train_all, y_train_all, ENSEMBLE_SEEDS)
        fold_models = {"global": global_ensemble}

        # Train regime-specific ensembles
        for regime in DEDICATED_REGIMES_V2:
            regime_df = df_fold_train[df_fold_train["regime"] == regime]
            n = len(regime_df)
            if n < MIN_REGIME_SAMPLES:
                fold_models[regime] = global_ensemble
                print(f"    {regime}: fallback ({n} < {MIN_REGIME_SAMPLES})")
                continue

            X_r = regime_df[feature_cols].values
            y_r = regime_df[TARGET_COL].values
            regime_ensemble = train_ensemble(X_r, y_r, ENSEMBLE_SEEDS)

            # G1 check: Spearman on held-out year for this regime
            regime_test = df_fold_test[df_fold_test["regime"] == regime]
            if len(regime_test) > 10:
                X_test_r = regime_test[feature_cols].values
                y_test_r = regime_test[TARGET_COL].values
                y_pred_r = predict_ensemble(regime_ensemble, X_test_r)
                rho, _ = spearmanr(y_test_r, y_pred_r) if len(y_test_r) > 2 else (0.0, 1.0)
                if rho < G1_MIN_RANK_CORR:
                    fold_models[regime] = global_ensemble
                    print(f"    {regime}: G1 fail rho={rho:.4f} -> fallback")
                    continue

            fold_models[regime] = regime_ensemble
            print(f"    {regime}: dedicated ({n} trades)")

        # Fallback regimes
        for regime in FALLBACK_REGIMES_V2:
            fold_models[regime] = global_ensemble

        # Save fold models for honest calibration scoring
        fold_models_by_year[test_year] = fold_models

        # Score held-out year (CSV trades)
        preds = score_trades_ensemble(df_fold_test, fold_models, feature_cols)
        fold_predictions[test_year] = preds

        # Fold stats
        y_true = df_fold_test[TARGET_COL].values
        y_pred = preds.values
        rho, p_val = spearmanr(y_true, y_pred) if len(y_true) > 2 else (0.0, 1.0)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        print(f"    Fold {test_year}: n={len(df_fold_test)} rho={rho:.4f} MAE={mae:.1f}%")

        fold_stats[test_year] = {
            "n_test": len(df_fold_test),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p_val), 4),
            "mae": round(mae, 2),
        }
        training_info["folds"][str(test_year)] = fold_stats[test_year]

    # ── Train final ensembles on ALL train_years ─────────────────────────
    print(f"\n  Training final ensembles on all {train_years}...")
    X_all = df_train[feature_cols].values
    y_all = df_train[TARGET_COL].values
    final_global = train_ensemble(X_all, y_all, ENSEMBLE_SEEDS)
    final_models = {"global": final_global}

    for regime in DEDICATED_REGIMES_V2:
        regime_df = df_train[df_train["regime"] == regime]
        n = len(regime_df)
        if n < MIN_REGIME_SAMPLES:
            final_models[regime] = final_global
            training_info["routing"][regime] = "fallback"
            print(f"    {regime}: fallback ({n} < {MIN_REGIME_SAMPLES})")
            continue

        X_r = regime_df[feature_cols].values
        y_r = regime_df[TARGET_COL].values
        regime_ensemble = train_ensemble(X_r, y_r, ENSEMBLE_SEEDS)

        # G1 check on aggregate LOO predictions for this regime
        all_true = []
        all_pred = []
        for yr, preds in fold_predictions.items():
            yr_mask = df_train["year"] == yr
            regime_mask = df_train["regime"] == regime
            combined = yr_mask & regime_mask
            if combined.sum() > 0:
                all_true.extend(df_train.loc[combined, TARGET_COL].values.tolist())
                all_pred.extend(preds[combined].values.tolist())
        if len(all_true) > 10:
            rho, _ = spearmanr(all_true, all_pred)
            if rho < G1_MIN_RANK_CORR:
                final_models[regime] = final_global
                training_info["routing"][regime] = f"fallback (G1 rho={rho:.4f})"
                print(f"    {regime}: G1 fail rho={rho:.4f} -> fallback")
                continue

        final_models[regime] = regime_ensemble
        training_info["routing"][regime] = "dedicated"
        print(f"    {regime}: dedicated ({n} trades)")

    for regime in FALLBACK_REGIMES_V2:
        final_models[regime] = final_global
        training_info["routing"][regime] = "fallback"

    training_info["routing"]["global"] = "dedicated"

    return fold_predictions, fold_models_by_year, final_models, training_info


# ═══════════════════════════════════════════════════════════════════════════
# Run backtests and enrich
# ═══════════════════════════════════════════════════════════════════════════

def run_and_enrich_years(years: List[int],
                         full_spy_closes: pd.Series,
                         full_vix: pd.Series) -> Tuple[Dict, Dict]:
    """Run EXP-401, enrich trades for given years.

    Returns (baseline_yearly, enriched_dfs) — predictions NOT yet applied.
    """
    baseline_yearly = {}
    enriched_dfs = {}

    for year in years:
        t0 = time.time()
        print(f"  {year}...", end=" ", flush=True)

        bt, combined = run_year_backtest_exp401(year)
        baseline_yearly[year] = {
            "return_pct": round(combined.get("return_pct", 0), 2),
            "total_trades": combined.get("total_trades", 0),
            "win_rate": round(combined.get("win_rate", 0), 2),
            "max_drawdown": round(combined.get("max_drawdown", 0), 2),
            "total_pnl": round(combined.get("total_pnl", 0), 2),
        }

        trades = enrich_trades(bt, year, spy_closes=full_spy_closes, vix_series=full_vix)
        df = add_v2_features(trades, full_vix)

        if not df.empty:
            df = df.sort_values("exit_date").reset_index(drop=True)
            # Remap regimes (Fix 4)
            df["regime"] = df["regime"].apply(remap_regime)

        enriched_dfs[year] = df
        elapsed = time.time() - t0
        bl = baseline_yearly[year]
        print(f"ret={bl['return_pct']:+.1f}%  trades={len(df)}  ({elapsed:.0f}s)")

    return baseline_yearly, enriched_dfs


# ═══════════════════════════════════════════════════════════════════════════
# Gate evaluation (reused from original, adapted)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_blind_gates(baseline_yearly: Dict, ml_yearly: Dict,
                         calibration_avg: float) -> Dict:
    """Evaluate G3-G6 on blind years only."""
    bl_rets = {y: baseline_yearly[y]["return_pct"] for y in BLIND_YEARS}
    ml_rets = {y: ml_yearly[str(y)]["return_pct"] for y in BLIND_YEARS}

    bl_avg = sum(bl_rets.values()) / len(BLIND_YEARS)
    ml_avg = sum(ml_rets.values()) / len(BLIND_YEARS)

    # G3: ML blind avg >= baseline
    g3 = ml_avg >= bl_avg
    g3_detail = f"ML blind avg={ml_avg:.2f}% vs baseline={bl_avg:.2f}%"

    # G4: no blind year > 5% worse
    g4 = True
    g4_worst = 0.0
    for y in BLIND_YEARS:
        delta = ml_rets[y] - bl_rets[y]
        if delta < g4_worst:
            g4_worst = delta
        if delta < -5.0:
            g4 = False
    g4_detail = f"worst blind year delta={g4_worst:.2f}%"

    # G5: no negative flip
    g5 = True
    for y in BLIND_YEARS:
        if bl_rets[y] > 0 and ml_rets[y] < 0:
            g5 = False
    g5_detail = ("no blind year negative when baseline positive" if g5
                 else "FAIL: ML negative while baseline positive")

    # G6: walk-forward ratio
    if calibration_avg > 0:
        wf_ratio = ml_avg / calibration_avg
    else:
        wf_ratio = 0.0
    g6 = wf_ratio > 0.5
    g6_detail = (f"blind/calibration = {ml_avg:.2f}/{calibration_avg:.2f} "
                 f"= {wf_ratio:.3f}")

    all_pass = g3 and g4 and g5 and g6

    return {
        "g3_avg_return": {"pass": g3, "detail": g3_detail},
        "g4_no_year_5pct_worse": {"pass": g4, "detail": g4_detail},
        "g5_no_negative_flip": {"pass": g5, "detail": g5_detail},
        "g6_walkforward_ratio": {
            "pass": g6, "detail": g6_detail, "ratio": round(wf_ratio, 3),
        },
        "all_pass": all_pass,
    }


# ═══════════════════════════════════════════════════════════════════════════
# JSON encoder
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("  ML V2 IMPROVED WALK-FORWARD BLIND TEST")
    print("  5 Fixes: linear sizing, LOO-CV calibration, production retrain,")
    print("           defensive regime merge, 5-seed ensemble")
    print("  Train: 2020-2023 | Blind: 2024-2025")
    print("=" * 70)

    # ── Step 1: Load market data ──────────────────────────────────────────
    print("\nLoading market data...")
    full_spy, full_vix = _load_full_market_data()
    full_spy_closes = full_spy["Close"]

    # ── Step 2: Run all backtests (6 years) ───────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: RUN BACKTESTS + ENRICH (ALL 6 YEARS)")
    print("=" * 70)

    cal_baseline, cal_enriched = run_and_enrich_years(
        TRAIN_YEARS, full_spy_closes, full_vix
    )
    blind_baseline, blind_enriched = run_and_enrich_years(
        BLIND_YEARS, full_spy_closes, full_vix
    )

    # ── Step 3: Load training CSV for model training ──────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: TRAIN REGIME ENSEMBLES WITH LOO-CV (2020-2023 ONLY)")
    print("=" * 70)

    df_train_csv, feature_cols = load_training_csv(TRAIN_YEARS)

    # STRICT: verify no 2024/2025 data
    assert df_train_csv["year"].max() <= 2023, "DATA LEAK: found rows from 2024+"

    fold_predictions, fold_models_by_year, final_models, training_info = \
        train_regime_ensembles_loo(df_train_csv, TRAIN_YEARS, feature_cols)

    # ── Step 4: LOO-CV calibration (Fix 1 + 2) ───────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: LOO-CV CALIBRATION WITH LINEAR SIZING")
    print("=" * 70)

    # Collect ALL LOO predictions from calibration years
    all_loo_preds = []
    for year in TRAIN_YEARS:
        if year in fold_predictions:
            all_loo_preds.extend(fold_predictions[year].values.tolist())
    calibration_cdf = np.array(all_loo_preds)
    print(f"\n  Calibration CDF: {len(calibration_cdf)} predictions")
    print(f"  Prediction range: [{calibration_cdf.min():.1f}%, {calibration_cdf.max():.1f}%]")

    # Apply linear sizing to each calibration year using LOO fold models (honest)
    cal_ml_yearly = {}
    cal_returns = []

    for year in TRAIN_YEARS:
        df = cal_enriched[year]
        if df.empty or year not in fold_models_by_year:
            cal_ml_yearly[str(year)] = {
                "return_pct": 0.0, "total_trades": 0, "win_rate": 0.0,
                "max_drawdown": 0.0, "total_pnl": 0.0,
            }
            cal_returns.append(0.0)
            continue

        # Score backtest trades with the FOLD model that did NOT see this year
        # This is the honest LOO-CV calibration
        fold_model = fold_models_by_year[year]
        preds_bt = score_trades_ensemble(df, fold_model, feature_cols)

        # Verify no NaN
        n_nan = preds_bt.isna().sum()
        assert n_nan == 0, f"NaN predictions in calibration {year}: {n_nan}"

        # Apply linear sizing with calibration CDF
        multipliers_bt = linear_sizing(preds_bt, calibration_cdf)

        # Verify multiplier range
        assert multipliers_bt.min() >= SIZING_BASE - 0.001, \
            f"Multiplier below floor: {multipliers_bt.min()}"
        assert multipliers_bt.max() <= SIZING_BASE + SIZING_SCALE + 0.001, \
            f"Multiplier above ceiling: {multipliers_bt.max()}"

        stats = compute_year_stats(df, multipliers_bt, STARTING_CAPITAL)
        cal_ml_yearly[str(year)] = stats
        cal_returns.append(stats["return_pct"])

        bl = cal_baseline[year]
        delta = stats["return_pct"] - bl["return_pct"]
        print(f"  {year}: baseline={bl['return_pct']:+.1f}%  ML={stats['return_pct']:+.1f}%  "
              f"delta={delta:+.1f}%  avg_mult={multipliers_bt.mean():.2f}  "
              f"trades={len(df)} (LOO-honest)")

    calibration_avg = round(sum(cal_returns) / len(cal_returns), 2)
    print(f"\n  Calibration avg (LOO-honest): {calibration_avg:+.1f}%")

    # ── Step 5: Blind test (2024-2025) ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: BLIND TEST ON 2024-2025 (LOCKED MODEL + CDF)")
    print("=" * 70)

    blind_ml_yearly = {}
    for year in BLIND_YEARS:
        df = blind_enriched[year]
        if df.empty:
            blind_ml_yearly[str(year)] = {
                "return_pct": 0.0, "total_trades": 0, "win_rate": 0.0,
                "max_drawdown": 0.0, "total_pnl": 0.0,
            }
            continue

        # Score with FINAL model (trained on 2020-2023)
        preds = score_trades_ensemble(df, final_models, feature_cols)

        # Verify no NaN
        n_nan = preds.isna().sum()
        assert n_nan == 0, f"NaN predictions in blind {year}: {n_nan}"

        # Apply linear sizing with LOCKED calibration CDF
        multipliers = linear_sizing(preds, calibration_cdf)

        # Verify multiplier range
        assert multipliers.min() >= SIZING_BASE - 0.001
        assert multipliers.max() <= SIZING_BASE + SIZING_SCALE + 0.001

        stats = compute_year_stats(df, multipliers, STARTING_CAPITAL)
        blind_ml_yearly[str(year)] = stats

        bl = blind_baseline[year]
        delta = stats["return_pct"] - bl["return_pct"]
        print(f"  {year}: baseline={bl['return_pct']:+.1f}%  ML={stats['return_pct']:+.1f}%  "
              f"delta={delta:+.1f}%  avg_mult={multipliers.mean():.2f}  "
              f"avg_pred={preds.mean():+.1f}%")

    # Blind aggregate
    blind_rets = [blind_ml_yearly[str(y)]["return_pct"] for y in BLIND_YEARS]
    blind_avg = round(sum(blind_rets) / len(blind_rets), 2)
    blind_sharpe = compute_sharpe(blind_rets)
    bl_blind_avg = round(
        sum(blind_baseline[y]["return_pct"] for y in BLIND_YEARS) / len(BLIND_YEARS), 2
    )

    print(f"\n  Blind aggregate: ML avg={blind_avg:+.1f}%  baseline avg={bl_blind_avg:+.1f}%")

    # ── Step 6: Gate evaluation ───────────────────────────────────────────
    gates = evaluate_blind_gates(blind_baseline, blind_ml_yearly, calibration_avg)

    print("\n" + "=" * 70)
    print("GATE EVALUATION (BLIND YEARS ONLY)")
    print("=" * 70)
    for gname, gval in gates.items():
        if gname == "all_pass":
            continue
        status = "PASS" if gval["pass"] else "FAIL"
        print(f"  {gname}: {status} -- {gval['detail']}")

    verdict = "PASS" if gates["all_pass"] else "FAIL"
    print(f"\n  VERDICT: {verdict}")

    # ── Step 7: Production model (Fix 3) ──────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: PRODUCTION MODEL (RETRAIN ON ALL 6 YEARS)")
    print("=" * 70)

    df_all_csv, all_feature_cols = load_training_csv(ALL_YEARS)
    X_prod = df_all_csv[all_feature_cols].values
    y_prod = df_all_csv[TARGET_COL].values
    prod_global = train_ensemble(X_prod, y_prod, ENSEMBLE_SEEDS)
    prod_models = {"global": prod_global}

    prod_regime_stats = {}
    for regime in DEDICATED_REGIMES_V2:
        regime_df = df_all_csv[df_all_csv["regime"] == regime]
        n = len(regime_df)
        if n < MIN_REGIME_SAMPLES:
            prod_models[regime] = prod_global
            prod_regime_stats[regime] = {"n": n, "model": "fallback"}
            continue
        X_r = regime_df[all_feature_cols].values
        y_r = regime_df[TARGET_COL].values
        prod_models[regime] = train_ensemble(X_r, y_r, ENSEMBLE_SEEDS)
        prod_regime_stats[regime] = {"n": n, "model": "dedicated"}
        print(f"  {regime}: {n} trades (dedicated)")

    for regime in FALLBACK_REGIMES_V2:
        prod_models[regime] = prod_global
        prod_regime_stats[regime] = {"n": 0, "model": "fallback"}

    print(f"  Production model trained on {len(df_all_csv)} trades (all 6 years)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  {'Year':>6s}  {'Baseline':>10s}  {'ML':>10s}  {'Delta':>8s}  {'Phase':<12s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*12}")

    for year in TRAIN_YEARS:
        bl = cal_baseline[year]["return_pct"]
        ml = cal_ml_yearly[str(year)]["return_pct"]
        delta = ml - bl
        print(f"  {year:>6d}  {bl:>+9.1f}%  {ml:>+9.1f}%  {delta:>+7.1f}%  calibration")

    for year in BLIND_YEARS:
        bl = blind_baseline[year]["return_pct"]
        ml = blind_ml_yearly[str(year)]["return_pct"]
        delta = ml - bl
        print(f"  {year:>6d}  {bl:>+9.1f}%  {ml:>+9.1f}%  {delta:>+7.1f}%  BLIND")

    print(f"\n  Calibration avg (LOO): {calibration_avg:+.1f}%  |  Blind avg: {blind_avg:+.1f}%")
    print(f"  Walk-forward ratio: {gates['g6_walkforward_ratio']['ratio']:.3f}")
    print(f"  Verdict: {verdict}")

    # ── Load original results for comparison ──────────────────────────────
    comparison = {}
    if ORIGINAL_RESULTS_PATH.exists():
        with open(ORIGINAL_RESULTS_PATH) as f:
            orig = json.load(f)
        orig_cal_avg = orig.get("calibration", {}).get("best_avg_return", 0)
        orig_blind_avg = orig.get("blind_test", {}).get("aggregate", {}).get("ml_avg_return", 0)
        orig_wf = orig.get("blind_test", {}).get("gates", {}).get(
            "g6_walkforward_ratio", {}
        ).get("ratio", 0)
        orig_verdict = orig.get("verdict", "N/A")

        comparison = {
            "original": {
                "calibration_avg": orig_cal_avg,
                "blind_avg": orig_blind_avg,
                "wf_ratio": orig_wf,
                "verdict": orig_verdict,
            },
            "improved": {
                "calibration_avg": calibration_avg,
                "blind_avg": blind_avg,
                "wf_ratio": gates["g6_walkforward_ratio"]["ratio"],
                "verdict": verdict,
            },
        }

        print("\n" + "=" * 70)
        print("COMPARISON: ORIGINAL vs IMPROVED")
        print("=" * 70)
        print(f"  {'Metric':<25s}  {'Original':>12s}  {'Improved':>12s}")
        print(f"  {'-'*25}  {'-'*12}  {'-'*12}")
        print(f"  {'Calibration avg':.<25s}  {orig_cal_avg:>+11.1f}%  {calibration_avg:>+11.1f}%")
        print(f"  {'Blind avg':.<25s}  {orig_blind_avg:>+11.1f}%  {blind_avg:>+11.1f}%")
        print(f"  {'WF ratio':.<25s}  {orig_wf:>12.3f}  {gates['g6_walkforward_ratio']['ratio']:>12.3f}")
        print(f"  {'Verdict':.<25s}  {orig_verdict:>12s}  {verdict:>12s}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now().isoformat(),
        "description": "Improved walk-forward blind test with 5 fixes",
        "fixes_applied": [
            "linear_sizing", "loo_cv_calibration", "production_retrain",
            "defensive_regime", "ensemble_5_seeds",
        ],
        "training": {
            "years": TRAIN_YEARS,
            "n_trades_csv": len(df_train_csv),
            "feature_cols": feature_cols,
            "ensemble_seeds": ENSEMBLE_SEEDS,
            "regime_remap": REGIME_REMAP,
            "dedicated_regimes": DEDICATED_REGIMES_V2,
            "fallback_regimes": FALLBACK_REGIMES_V2,
            "routing": training_info.get("routing", {}),
            "loo_folds": training_info.get("folds", {}),
        },
        "calibration": {
            "years": TRAIN_YEARS,
            "baseline": {str(y): cal_baseline[y] for y in TRAIN_YEARS},
            "ml": cal_ml_yearly,
            "calibration_avg": calibration_avg,
            "calibration_cdf_stats": {
                "n": len(calibration_cdf),
                "mean": round(float(calibration_cdf.mean()), 2),
                "std": round(float(calibration_cdf.std()), 2),
                "min": round(float(calibration_cdf.min()), 2),
                "max": round(float(calibration_cdf.max()), 2),
                "p25": round(float(np.percentile(calibration_cdf, 25)), 2),
                "p50": round(float(np.percentile(calibration_cdf, 50)), 2),
                "p75": round(float(np.percentile(calibration_cdf, 75)), 2),
            },
            "sizing_params": {
                "base": SIZING_BASE,
                "scale": SIZING_SCALE,
                "multiplier_range": [SIZING_BASE, SIZING_BASE + SIZING_SCALE],
            },
        },
        "blind_test": {
            "years": BLIND_YEARS,
            "baseline": {str(y): blind_baseline[y] for y in BLIND_YEARS},
            "ml": blind_ml_yearly,
            "aggregate": {
                "ml_avg_return": blind_avg,
                "baseline_avg_return": bl_blind_avg,
                "ml_sharpe": blind_sharpe,
            },
            "gates": gates,
        },
        "production_model": {
            "years": ALL_YEARS,
            "n_trades": len(df_all_csv),
            "regime_stats": prod_regime_stats,
        },
        "verdict": verdict,
        "comparison_to_original": comparison,
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)

    elapsed = time.time() - t_start
    print(f"\n  Results saved to {RESULTS_PATH}")
    print(f"  Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
