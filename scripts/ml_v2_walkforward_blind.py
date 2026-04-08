#!/usr/bin/env python3
"""
ML V2 Walk-Forward Blind Test — Regime-Specific Confidence Sizing

TRUE out-of-sample validation:
  Phase 1: Train XGBoost on 2020-2023 ONLY (LOO-CV within those 4 years)
  Phase 2: Calibrate sizing profiles on 2020-2023 ONLY
  Phase 3: Blind test on 2024-2025 with LOCKED model + profile

Models held in memory only — does NOT overwrite ml/models/ files.

Output: results/ml_v2_walkforward_blind.json

Usage:
    PYTHONPATH=. python3 scripts/ml_v2_walkforward_blind.py
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
    score_trades,
    get_multiplier,
    compute_year_stats,
    compute_sharpe,
    SIZING_PROFILES,
    V2_FEATURE_COLS,
)
from scripts.ml_v2_regime_train import (
    XGB_PARAMS,
    RAW_FEATURE_COLS,
    MIN_REGIME_SAMPLES,
    G1_MIN_RANK_CORR,
    CAT_COLS,
    TARGET_COL,
    DEDICATED_REGIMES,
    FALLBACK_REGIMES,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "compass" / "training_data_v2.csv"
RESULTS_PATH = ROOT / "results" / "ml_v2_walkforward_blind.json"

TRAIN_YEARS = [2020, 2021, 2022, 2023]
BLIND_YEARS = [2024, 2025]
ALL_YEARS = TRAIN_YEARS + BLIND_YEARS


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Train on 2020-2023 ONLY
# ═══════════════════════════════════════════════════════════════════════════

def load_training_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load V2 training CSV, filter to 2020-2023, prepare features."""
    df = pd.read_csv(DATA_PATH)
    total = len(df)

    # STRICT: only training years
    df = df[df["year"].isin(TRAIN_YEARS)].copy()
    assert df["year"].max() <= 2023, "DATA LEAK: found rows from 2024+"
    assert len(df[df["year"] > 2023]) == 0, "DATA LEAK: post-2023 rows present"

    print(f"  Loaded {total} total trades, filtered to {len(df)} (2020-2023 only)")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Regimes: {df['regime'].value_counts().to_dict()}")

    # One-hot encode strategy_type
    for col in CAT_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
            df = pd.concat([df, dummies], axis=1)

    # Build feature columns (same order as ml_v2_regime_train.py)
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


def train_loo_cv_walk(df: pd.DataFrame, feature_cols: List[str],
                      label: str = "global") -> Dict:
    """LOO-CV within 2020-2023 (4 folds). Returns fold stats + trained model."""
    fold_results = []
    all_y_true = []
    all_y_pred = []

    for test_year in TRAIN_YEARS:
        train_mask = (df["year"] != test_year)
        test_mask = (df["year"] == test_year)

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

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        rho, p_val = spearmanr(y_test, y_pred) if len(y_test) > 2 else (0.0, 1.0)
        mae = float(np.mean(np.abs(y_test - y_pred)))

        fold_results.append({
            "test_year": test_year,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "r2": round(float(r2), 4),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p_val), 4),
            "mae": round(mae, 2),
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(f"    {test_year}: n={len(X_test):4d}  R2={r2:.4f}  "
              f"rho={rho:.4f} (p={p_val:.3f})  MAE={mae:.1f}%")

    # Aggregate
    if len(all_y_true) > 2:
        agg_rho, _ = spearmanr(all_y_true, all_y_pred)
    else:
        agg_rho = 0.0

    # Final model: train on ALL 2020-2023
    X_all = df[feature_cols].values
    y_all = df[TARGET_COL].values
    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_all, y_all, verbose=False)

    # Feature importance
    importances = final_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    return {
        "label": label,
        "n_total": len(df),
        "folds": fold_results,
        "aggregate_spearman_rho": round(float(agg_rho), 4),
        "top_5_features": [(f, round(float(v), 4)) for f, v in feat_imp[:5]],
        "model": final_model,
        "feature_names": feature_cols,
    }


def train_regime_models(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Tuple]:
    """Train global + regime-specific models on 2020-2023. Returns {regime: (model, feat_names)}."""
    print("\n  GLOBAL MODEL")
    global_result = train_loo_cv_walk(df, feature_cols, "global")

    models = {"global": (global_result["model"], global_result["feature_names"])}
    training_info = {"global": global_result}

    df_with_regime = df[df["regime"].notna()].copy()

    for regime in DEDICATED_REGIMES + FALLBACK_REGIMES:
        regime_df = df_with_regime[df_with_regime["regime"] == regime]
        n = len(regime_df)

        print(f"\n  REGIME: {regime.upper()} ({n} trades)")

        if n < MIN_REGIME_SAMPLES:
            print(f"    Fallback to global ({n} < {MIN_REGIME_SAMPLES})")
            models[regime] = models["global"]
            training_info[regime] = {
                "label": regime, "n_total": n, "fallback": True,
                "fallback_reason": f"Only {n} trades",
            }
            continue

        result = train_loo_cv_walk(regime_df, feature_cols, regime)

        # G1 gate: rho < threshold → fallback to global
        if result["aggregate_spearman_rho"] < G1_MIN_RANK_CORR:
            print(f"    G1 FAIL: rho={result['aggregate_spearman_rho']:.4f} < {G1_MIN_RANK_CORR} -> fallback")
            models[regime] = models["global"]
            training_info[regime] = {
                "label": regime, "n_total": n, "fallback": True,
                "fallback_reason": f"G1 fail: rho={result['aggregate_spearman_rho']:.4f}",
                "folds": result["folds"],
                "aggregate_spearman_rho": result["aggregate_spearman_rho"],
            }
        else:
            print(f"    G1 PASS: rho={result['aggregate_spearman_rho']:.4f}")
            models[regime] = (result["model"], result["feature_names"])
            training_info[regime] = result

    return models, training_info


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 & 3: Run backtests, enrich, score, evaluate
# ═══════════════════════════════════════════════════════════════════════════

def run_and_enrich_years(years: List[int], models: Dict,
                         full_spy_closes: pd.Series, full_vix: pd.Series
                         ) -> Tuple[Dict, Dict]:
    """Run EXP-401, enrich trades, score with ML for given years.

    Returns (baseline_yearly, enriched_dfs).
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
            df["predicted_return"] = score_trades(df, models)

        enriched_dfs[year] = df
        elapsed = time.time() - t0
        bl = baseline_yearly[year]
        pred_mean = df["predicted_return"].mean() if not df.empty else 0
        print(f"ret={bl['return_pct']:+.1f}%  trades={len(df)}  "
              f"avg_pred={pred_mean:+.1f}%  ({elapsed:.0f}s)")

    return baseline_yearly, enriched_dfs


def sweep_profiles(years: List[int], enriched_dfs: Dict,
                   baseline_yearly: Dict) -> Dict:
    """Sweep sizing profiles over given years. Returns {profile_name: {yearly, aggregate}}."""
    results = {}

    for profile_name, profile_tiers in SIZING_PROFILES.items():
        yearly_stats = {}
        for year in years:
            df = enriched_dfs[year]
            if df.empty:
                yearly_stats[year] = {
                    "return_pct": 0.0, "total_trades": 0, "win_rate": 0.0,
                    "max_drawdown": 0.0, "total_pnl": 0.0,
                }
                continue

            multipliers = df["predicted_return"].apply(
                lambda x: get_multiplier(x, profile_tiers)
            )
            yearly_stats[year] = compute_year_stats(df, multipliers, STARTING_CAPITAL)

        rets = [yearly_stats[y]["return_pct"] for y in years]
        avg_ret = round(sum(rets) / len(rets), 2) if rets else 0.0
        worst_dd = round(min(yearly_stats[y]["max_drawdown"] for y in years), 2) if years else 0.0
        sharpe = compute_sharpe(rets)

        results[profile_name] = {
            "yearly": {str(y): yearly_stats[y] for y in years},
            "aggregate": {
                "avg_return": avg_ret,
                "worst_dd": worst_dd,
                "years_profitable": sum(1 for r in rets if r > 0),
                "sharpe": sharpe,
            },
        }

    return results


def evaluate_blind_gates(baseline_yearly: Dict, ml_yearly: Dict,
                         calibration_avg: float) -> Dict:
    """Evaluate G3-G6 on blind years only.

    G3: blind avg return >= baseline blind avg return
    G4: no blind year > 5% worse than baseline
    G5: no blind year negative when baseline is positive
    G6: walk-forward ratio = blind avg / calibration avg > 0.5
    """
    bl_rets = {y: baseline_yearly[y]["return_pct"] for y in BLIND_YEARS}
    ml_rets = {y: ml_yearly[str(y)]["return_pct"] for y in BLIND_YEARS}

    bl_avg = sum(bl_rets.values()) / len(BLIND_YEARS)
    ml_avg = sum(ml_rets.values()) / len(BLIND_YEARS)

    # G3
    g3 = ml_avg >= bl_avg
    g3_detail = f"ML blind avg={ml_avg:.2f}% vs baseline={bl_avg:.2f}%"

    # G4
    g4 = True
    g4_worst = 0.0
    for y in BLIND_YEARS:
        delta = ml_rets[y] - bl_rets[y]
        if delta < g4_worst:
            g4_worst = delta
        if delta < -5.0:
            g4 = False
    g4_detail = f"worst blind year delta={g4_worst:.2f}%"

    # G5
    g5 = True
    for y in BLIND_YEARS:
        if bl_rets[y] > 0 and ml_rets[y] < 0:
            g5 = False
    g5_detail = "no blind year negative when baseline positive" if g5 else "FAIL: ML negative while baseline positive"

    # G6: walk-forward ratio
    if calibration_avg > 0:
        wf_ratio = ml_avg / calibration_avg
    else:
        wf_ratio = 0.0
    g6 = wf_ratio > 0.5
    g6_detail = f"blind/calibration = {ml_avg:.2f}/{calibration_avg:.2f} = {wf_ratio:.3f}"

    all_pass = g3 and g4 and g5 and g6

    return {
        "g3_avg_return": {"pass": g3, "detail": g3_detail},
        "g4_no_year_5pct_worse": {"pass": g4, "detail": g4_detail},
        "g5_no_negative_flip": {"pass": g5, "detail": g5_detail},
        "g6_walkforward_ratio": {"pass": g6, "detail": g6_detail, "ratio": round(wf_ratio, 3)},
        "all_pass": all_pass,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("  ML V2 WALK-FORWARD BLIND TEST")
    print("  Train: 2020-2023 | Blind: 2024-2025")
    print("=" * 70)

    # ── Phase 1: Train on 2020-2023 ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: TRAIN MODELS ON 2020-2023 ONLY")
    print("=" * 70)

    df_train, feature_cols = load_training_data()
    models, training_info = train_regime_models(df_train, feature_cols)

    # Summarize routing
    routing = {}
    for regime in ["global"] + DEDICATED_REGIMES + FALLBACK_REGIMES:
        info = training_info.get(regime, {})
        is_fallback = info.get("fallback", False)
        routing[regime] = "fallback" if is_fallback else "dedicated"
    print(f"\n  Model routing: {routing}")

    # ── Load market data ──────────────────────────────────────────────────
    print("\nLoading market data...")
    full_spy, full_vix = _load_full_market_data()
    full_spy_closes = full_spy["Close"]

    # ── Phase 2: Calibrate on 2020-2023 ───────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: CALIBRATE SIZING ON 2020-2023")
    print("=" * 70)

    print("\nRunning EXP-401 baseline + ML scoring for calibration years...")
    cal_baseline, cal_enriched = run_and_enrich_years(
        TRAIN_YEARS, models, full_spy_closes, full_vix
    )

    # Verify no NaN predictions
    cal_nan = sum(
        cal_enriched[y]["predicted_return"].isna().sum()
        for y in TRAIN_YEARS if not cal_enriched[y].empty
    )
    print(f"\n  Calibration NaN predictions: {cal_nan}")
    assert cal_nan == 0, f"NaN predictions in calibration: {cal_nan}"

    # Sweep profiles on calibration years
    print("\n  Sweeping sizing profiles on 2020-2023...")
    cal_profiles = sweep_profiles(TRAIN_YEARS, cal_enriched, cal_baseline)

    for pname, pdata in cal_profiles.items():
        agg = pdata["aggregate"]
        bl_avg = round(sum(cal_baseline[y]["return_pct"] for y in TRAIN_YEARS) / len(TRAIN_YEARS), 2)
        print(f"    {pname:14s}: avg={agg['avg_return']:+.1f}% (baseline={bl_avg:+.1f}%)  "
              f"worst_dd={agg['worst_dd']:.1f}%  Sharpe={agg['sharpe']}")

    # Pick best profile by avg return on calibration years
    best_profile = max(
        cal_profiles,
        key=lambda p: cal_profiles[p]["aggregate"]["avg_return"],
    )
    calibration_avg = cal_profiles[best_profile]["aggregate"]["avg_return"]
    print(f"\n  LOCKED profile: {best_profile} (calibration avg={calibration_avg:+.1f}%)")

    # ── Phase 3: Blind test on 2024-2025 ──────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: BLIND TEST ON 2024-2025 (LOCKED MODEL + PROFILE)")
    print("=" * 70)

    print("\nRunning EXP-401 baseline + ML scoring for blind years...")
    blind_baseline, blind_enriched = run_and_enrich_years(
        BLIND_YEARS, models, full_spy_closes, full_vix
    )

    # Verify no NaN predictions
    blind_nan = sum(
        blind_enriched[y]["predicted_return"].isna().sum()
        for y in BLIND_YEARS if not blind_enriched[y].empty
    )
    print(f"\n  Blind NaN predictions: {blind_nan}")
    assert blind_nan == 0, f"NaN predictions in blind test: {blind_nan}"

    # Apply LOCKED profile to blind years
    locked_tiers = SIZING_PROFILES[best_profile]
    blind_ml_yearly = {}

    print(f"\n  Applying LOCKED profile '{best_profile}' to blind years:")
    for year in BLIND_YEARS:
        df = blind_enriched[year]
        if df.empty:
            blind_ml_yearly[str(year)] = {
                "return_pct": 0.0, "total_trades": 0, "win_rate": 0.0,
                "max_drawdown": 0.0, "total_pnl": 0.0,
            }
            continue

        multipliers = df["predicted_return"].apply(
            lambda x: get_multiplier(x, locked_tiers)
        )
        stats = compute_year_stats(df, multipliers, STARTING_CAPITAL)
        blind_ml_yearly[str(year)] = stats

        bl = blind_baseline[year]
        delta = stats["return_pct"] - bl["return_pct"]
        avg_mult = multipliers.mean()
        print(f"    {year}: baseline={bl['return_pct']:+.1f}%  ML={stats['return_pct']:+.1f}%  "
              f"delta={delta:+.1f}%  avg_mult={avg_mult:.2f}")

    # Blind aggregate
    blind_rets = [blind_ml_yearly[str(y)]["return_pct"] for y in BLIND_YEARS]
    blind_avg = round(sum(blind_rets) / len(blind_rets), 2)
    blind_sharpe = compute_sharpe(blind_rets)
    bl_blind_avg = round(sum(blind_baseline[y]["return_pct"] for y in BLIND_YEARS) / len(BLIND_YEARS), 2)

    print(f"\n  Blind aggregate: ML avg={blind_avg:+.1f}%  baseline avg={bl_blind_avg:+.1f}%")

    # ── Gate evaluation (blind years only) ────────────────────────────────
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

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  {'Year':>6s}  {'Baseline':>10s}  {'ML':>10s}  {'Delta':>8s}  {'Phase':<12s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*12}")

    for year in TRAIN_YEARS:
        bl = cal_baseline[year]["return_pct"]
        ml = cal_profiles[best_profile]["yearly"][str(year)]["return_pct"]
        delta = ml - bl
        print(f"  {year:>6d}  {bl:>+9.1f}%  {ml:>+9.1f}%  {delta:>+7.1f}%  calibration")

    for year in BLIND_YEARS:
        bl = blind_baseline[year]["return_pct"]
        ml = blind_ml_yearly[str(year)]["return_pct"]
        delta = ml - bl
        print(f"  {year:>6d}  {bl:>+9.1f}%  {ml:>+9.1f}%  {delta:>+7.1f}%  BLIND")

    print(f"\n  Calibration avg: {calibration_avg:+.1f}%  |  Blind avg: {blind_avg:+.1f}%")
    print(f"  Walk-forward ratio: {gates['g6_walkforward_ratio']['ratio']:.3f}")
    print(f"  Profile: {best_profile}  |  Verdict: {verdict}")

    # ── Save results ──────────────────────────────────────────────────────
    class _Enc(json.JSONEncoder):
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

    # Build serializable training info (strip model objects)
    training_serial = {}
    for regime, info in training_info.items():
        entry = {k: v for k, v in info.items() if k not in ("model", "feature_names")}
        # Convert top_5_features tuples to lists
        if "top_5_features" in entry:
            entry["top_5_features"] = [[f, v] for f, v in entry["top_5_features"]]
        training_serial[regime] = entry

    output = {
        "generated_at": datetime.now().isoformat(),
        "description": "Walk-forward blind test: train 2020-2023, blind 2024-2025",
        "training": {
            "years": TRAIN_YEARS,
            "n_trades": len(df_train),
            "regime_models": training_serial,
            "routing": routing,
        },
        "calibration": {
            "years": TRAIN_YEARS,
            "baseline": {str(y): cal_baseline[y] for y in TRAIN_YEARS},
            "profiles": {
                pname: pdata for pname, pdata in cal_profiles.items()
            },
            "best_profile": best_profile,
            "best_avg_return": calibration_avg,
        },
        "blind_test": {
            "years": BLIND_YEARS,
            "locked_profile": best_profile,
            "baseline": {str(y): blind_baseline[y] for y in BLIND_YEARS},
            "ml": blind_ml_yearly,
            "aggregate": {
                "ml_avg_return": blind_avg,
                "baseline_avg_return": bl_blind_avg,
                "ml_sharpe": blind_sharpe,
            },
            "gates": gates,
        },
        "verdict": verdict,
        "sizing_profiles": {
            name: [(t, m) for t, m in tiers]
            for name, tiers in SIZING_PROFILES.items()
        },
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=_Enc)

    elapsed = time.time() - t_start
    print(f"\n  Results saved to {RESULTS_PATH}")
    print(f"  Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
