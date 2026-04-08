#!/usr/bin/env python3
"""
ML V2 Backtest — Regime-Specific Confidence Sizing

Post-hoc PnL scaling: runs EXP-401 baseline per year, enriches trades with
features, scores with regime-specific V2 XGBoost regressors, maps predicted
return_pct to sizing multiplier, and recomputes portfolio stats.

Sweeps 3 multiplier profiles (conservative/balanced/aggressive).
Evaluates G3-G6 gates vs EXP-401 baseline.

Usage:
    PYTHONPATH=. python3 scripts/ml_v2_backtest.py
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from compass.collect_training_data import (
    enrich_trades,
    run_year_backtest_exp401,
    _load_full_market_data,
    _compute_ma,
    STARTING_CAPITAL,
    YEARS,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Model paths ───────────────────────────────────────────────────────────
TRAINING_RESULTS_PATH = ROOT / "results" / "ml_v2_training_results.json"
MODEL_DIR = ROOT / "ml" / "models"
RESULTS_PATH = ROOT / "results" / "ml_v2_backtest_results.json"

# ── Sizing multiplier profiles ────────────────────────────────────────────
# predicted return → sizing multiplier
SIZING_PROFILES = {
    "conservative": [
        (15.0, 1.10),   # predicted >= 15%
        (5.0,  1.00),   # 5-15%
        (0.0,  0.75),   # 0-5%
        (-10.0, 0.50),  # -10 to 0%
        (None,  0.35),  # < -10% (floor)
    ],
    "balanced": [
        (15.0, 1.25),
        (5.0,  1.00),
        (0.0,  0.75),
        (-10.0, 0.50),
        (None,  0.25),
    ],
    "aggressive": [
        (15.0, 1.50),
        (5.0,  1.00),
        (0.0,  0.50),
        (-10.0, 0.25),
        (None,  0.10),
    ],
}

# V2 feature columns (must match training order from ml_v2_training_results.json)
V2_FEATURE_COLS = [
    "day_of_week", "days_since_last_trade", "dte_at_entry",
    "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
    "vix", "vix_percentile_20d", "vix_percentile_50d", "vix_percentile_100d",
    "iv_rank",
    "dist_from_ma20_pct", "dist_from_ma50_pct", "dist_from_ma80_pct", "dist_from_ma200_pct",
    "ma20_slope_ann_pct", "ma50_slope_ann_pct",
    "realized_vol_atr20", "realized_vol_5d", "realized_vol_10d", "realized_vol_20d",
    "credit_to_width_ratio", "vix_ma10", "vix_change_5d",
    "month_of_year", "week_of_year",
    "strategy_type_CS", "strategy_type_DS", "strategy_type_IC",
    "strategy_type_MS", "strategy_type_SS",
]


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_regime_models() -> Dict[str, Tuple]:
    """Load V2 regime models and build routing dict.

    Returns: {regime_name: (model, feature_names)}
    """
    with open(TRAINING_RESULTS_PATH) as f:
        training_results = json.load(f)

    model_paths = training_results["model_paths"]
    models = {}
    loaded_files = {}

    for regime, rel_path in model_paths.items():
        full_path = ROOT / rel_path
        if not full_path.exists():
            print(f"  WARNING: model not found: {full_path}")
            continue

        # Cache — don't reload the same file
        if rel_path not in loaded_files:
            data = joblib.load(full_path)
            loaded_files[rel_path] = (data["model"], data["feature_names"])

        models[regime] = loaded_files[rel_path]

    print(f"  Loaded {len(loaded_files)} unique model files for {len(models)} regimes")
    for regime, (_, feats) in models.items():
        print(f"    {regime}: {model_paths[regime]}")

    return models


# ═══════════════════════════════════════════════════════════════════════════
# V2 feature enrichment
# ═══════════════════════════════════════════════════════════════════════════

def add_v2_features(trades: List[Dict], vix_series: pd.Series) -> pd.DataFrame:
    """Add V2-specific features that enrich_trades() doesn't produce.

    Computes: credit_to_width_ratio, vix_ma10, vix_change_5d,
              month_of_year, week_of_year, strategy_type one-hot.
    """
    df = pd.DataFrame(trades)
    if df.empty:
        return df

    # credit_to_width_ratio
    nc = df["net_credit"].fillna(0)
    sw = df["spread_width"].fillna(1).replace(0, 1)
    df["credit_to_width_ratio"] = (nc / sw).round(4)

    # month_of_year, week_of_year from entry_date
    entry_dates = pd.to_datetime(df["entry_date"])
    df["month_of_year"] = entry_dates.dt.month
    df["week_of_year"] = entry_dates.dt.isocalendar().week.astype(int)

    # vix_ma10 and vix_change_5d
    df["vix_ma10"] = 0.0
    df["vix_change_5d"] = 0.0
    if vix_series is not None:
        for i, row in df.iterrows():
            entry_ts = pd.Timestamp(row["entry_date"])
            ma10 = _compute_ma(vix_series, entry_ts, 10)
            df.at[i, "vix_ma10"] = round(ma10, 2) if ma10 is not None else 0.0

            hist = vix_series.loc[vix_series.index <= entry_ts]
            if len(hist) >= 6:
                df.at[i, "vix_change_5d"] = round(
                    float(hist.iloc[-1]) - float(hist.iloc[-6]), 2
                )

    # One-hot strategy_type
    for stype in ["CS", "DS", "IC", "MS", "SS"]:
        df[f"strategy_type_{stype}"] = (df["strategy_type"] == stype).astype(float)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# ML scoring
# ═══════════════════════════════════════════════════════════════════════════

def score_trades(df: pd.DataFrame, models: Dict) -> pd.Series:
    """Score each trade with its regime-specific V2 model.

    Returns predicted return_pct per trade.
    """
    predictions = pd.Series(0.0, index=df.index)

    for i, row in df.iterrows():
        regime = row.get("regime", None)
        if regime and regime in models:
            model, feat_names = models[regime]
        elif "global" in models:
            model, feat_names = models["global"]
        else:
            continue

        # Build feature vector in model's expected order
        features = []
        for col in feat_names:
            val = row.get(col, 0.0)
            features.append(float(val) if pd.notna(val) else 0.0)

        pred = model.predict(np.array([features]))[0]
        predictions.at[i] = float(pred)

    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# Sizing multiplier
# ═══════════════════════════════════════════════════════════════════════════

def get_multiplier(predicted_return: float, profile: List[Tuple]) -> float:
    """Map predicted return_pct to sizing multiplier."""
    for threshold, mult in profile:
        if threshold is None:
            return mult
        if predicted_return >= threshold:
            return mult
    return profile[-1][1]


# ═══════════════════════════════════════════════════════════════════════════
# Stats computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_year_stats(trades_df: pd.DataFrame, multipliers: pd.Series,
                       capital: float) -> Dict:
    """Compute adjusted stats for one year given PnL multipliers."""
    if trades_df.empty:
        return {
            "return_pct": 0.0, "total_trades": 0, "win_rate": 0.0,
            "max_drawdown": 0.0, "total_pnl": 0.0,
        }

    adjusted_pnl = trades_df["pnl"] * multipliers
    total_pnl = float(adjusted_pnl.sum())
    return_pct = round(total_pnl / capital * 100, 2)

    wins = (adjusted_pnl > 0).sum()
    n = len(trades_df)
    win_rate = round(wins / n * 100, 2) if n > 0 else 0.0

    # Max drawdown from cumulative PnL (trades sorted by exit_date)
    sorted_pnl = adjusted_pnl.values
    equity = capital
    peak = capital
    max_dd = 0.0
    for p in sorted_pnl:
        equity += p
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak * 100 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return {
        "return_pct": return_pct,
        "total_trades": n,
        "win_rate": win_rate,
        "max_drawdown": round(max_dd, 2),
        "total_pnl": round(total_pnl, 2),
    }


def compute_sharpe(yearly_returns: List[float]) -> float:
    """Compute Sharpe ratio from annual returns (risk-free = 0)."""
    if len(yearly_returns) < 2:
        return 0.0
    arr = np.array(yearly_returns)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return round(float(arr.mean() / std), 2)


# ═══════════════════════════════════════════════════════════════════════════
# Gate evaluation (G3-G6)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_gates(baseline_yearly: Dict, ml_yearly: Dict, ml_agg: Dict) -> Dict:
    """Evaluate G3-G6 gates.

    G3: avg return >= baseline avg return
    G4: no single year more than 5% worse than baseline year
    G5: 2022 adjusted return >= baseline 2022 return
    G6: simplified ROBUST score >= 0.70
    """
    baseline_rets = {y: baseline_yearly[y]["return_pct"] for y in YEARS}
    ml_rets = {y: ml_yearly[y]["return_pct"] for y in YEARS}

    baseline_avg = sum(baseline_rets.values()) / len(YEARS)
    ml_avg = ml_agg["avg_return"]

    # G3: avg return >= baseline
    g3 = ml_avg >= baseline_avg
    g3_detail = f"ML avg={ml_avg:.2f}% vs baseline={baseline_avg:.2f}%"

    # G4: no year more than 5% worse
    g4 = True
    g4_worst = 0.0
    for y in YEARS:
        delta = ml_rets[y] - baseline_rets[y]
        if delta < g4_worst:
            g4_worst = delta
        if delta < -5.0:
            g4 = False
    g4_detail = f"worst year delta={g4_worst:.2f}%"

    # G5: 2022 >= baseline 2022
    g5 = ml_rets[2022] >= baseline_rets[2022]
    g5_detail = f"ML 2022={ml_rets[2022]:.2f}% vs baseline={baseline_rets[2022]:.2f}%"

    # G6: simplified ROBUST score
    years_profitable = sum(1 for r in ml_rets.values() if r > 0)
    a_score = years_profitable / len(YEARS)

    # Walk-forward: test years 2023/2024/2025, train = prior years
    # Use ratio of test return to mean of training returns
    wf_ratios = []
    for test_yr in [2023, 2024, 2025]:
        train_rets = [ml_rets[y] for y in YEARS if y < test_yr]
        if train_rets and np.mean(train_rets) > 0:
            ratio = ml_rets[test_yr] / np.mean(train_rets)
            wf_ratios.append(min(ratio, 2.0))  # cap at 2x
    b_score = float(np.median(wf_ratios)) if wf_ratios else 0.0
    b_score = min(b_score, 1.0)

    # Sensitivity: how stable across profiles (computed externally, use 1.0 placeholder)
    c_score = 1.0  # will be overridden by caller with cross-profile stability

    robust = a_score * 0.25 + b_score * 0.30 + c_score * 0.25 + 0.10 + 0.10
    g6 = robust >= 0.70
    g6_detail = f"ROBUST={robust:.3f} (A={a_score:.2f} B={b_score:.2f} C={c_score:.2f})"

    return {
        "g3_avg_return": {"pass": g3, "detail": g3_detail},
        "g4_no_year_5pct_worse": {"pass": g4, "detail": g4_detail},
        "g5_2022_not_worse": {"pass": g5, "detail": g5_detail},
        "g6_robust_score": {"pass": g6, "detail": g6_detail, "score": round(robust, 3)},
        "all_pass": g3 and g4 and g5 and g6,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("  ML V2 BACKTEST — REGIME-SPECIFIC CONFIDENCE SIZING")
    print("=" * 70)

    # ── Step 1: Load models ───────────────────────────────────────────────
    print("\nStep 1: Loading V2 regime models...")
    models = load_regime_models()

    # ── Step 2: Load market data ──────────────────────────────────────────
    print("\nStep 2: Loading market data...")
    full_spy, full_vix = _load_full_market_data()
    full_spy_closes = full_spy["Close"]

    # ── Step 3: Run EXP-401 baseline + enrich + score ─────────────────────
    print("\nStep 3: Running EXP-401 baseline per year + ML scoring...")
    print("-" * 70)

    baseline_yearly = {}
    all_enriched_dfs = {}

    for year in YEARS:
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

        # Enrich trades
        trades = enrich_trades(bt, year, spy_closes=full_spy_closes, vix_series=full_vix)
        df = add_v2_features(trades, full_vix)

        if not df.empty:
            # Sort by exit_date for drawdown computation
            df = df.sort_values("exit_date").reset_index(drop=True)
            # Score with ML
            df["predicted_return"] = score_trades(df, models)

        all_enriched_dfs[year] = df

        elapsed = time.time() - t0
        n_trades = len(df)
        bl = baseline_yearly[year]
        pred_mean = df["predicted_return"].mean() if not df.empty else 0
        print(f"ret={bl['return_pct']:+.1f}%  trades={n_trades}  "
              f"avg_pred={pred_mean:+.1f}%  ({elapsed:.0f}s)")

    # ── Verification: baseline totals ─────────────────────────────────────
    total_trades = sum(baseline_yearly[y]["total_trades"] for y in YEARS)
    total_enriched = sum(len(all_enriched_dfs[y]) for y in YEARS)
    print(f"\n  Baseline total: {total_trades} trades, enriched: {total_enriched}")

    nan_preds = sum(
        all_enriched_dfs[y]["predicted_return"].isna().sum()
        for y in YEARS if not all_enriched_dfs[y].empty
    )
    print(f"  NaN predictions: {nan_preds}")

    # ── Step 4: Apply sizing profiles ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("Step 4: Applying sizing multiplier profiles")
    print("=" * 70)

    profile_results = {}

    for profile_name, profile_tiers in SIZING_PROFILES.items():
        print(f"\n  --- {profile_name.upper()} ---")

        yearly_stats = {}
        for year in YEARS:
            df = all_enriched_dfs[year]
            if df.empty:
                yearly_stats[year] = {
                    "return_pct": 0.0, "total_trades": 0, "win_rate": 0.0,
                    "max_drawdown": 0.0, "total_pnl": 0.0,
                }
                continue

            multipliers = df["predicted_return"].apply(
                lambda x: get_multiplier(x, profile_tiers)
            )
            stats = compute_year_stats(df, multipliers, STARTING_CAPITAL)
            yearly_stats[year] = stats

            bl_ret = baseline_yearly[year]["return_pct"]
            delta = stats["return_pct"] - bl_ret
            avg_mult = multipliers.mean()
            print(f"    {year}: baseline={bl_ret:+.1f}%  ML={stats['return_pct']:+.1f}%  "
                  f"delta={delta:+.1f}%  avg_mult={avg_mult:.2f}")

        # Aggregate
        rets = [yearly_stats[y]["return_pct"] for y in YEARS]
        dds = [yearly_stats[y]["max_drawdown"] for y in YEARS]
        avg_ret = round(sum(rets) / len(rets), 2)
        worst_dd = round(min(dds), 2)
        years_prof = sum(1 for r in rets if r > 0)
        sharpe = compute_sharpe(rets)

        agg = {
            "avg_return": avg_ret,
            "worst_dd": worst_dd,
            "years_profitable": years_prof,
            "sharpe": sharpe,
            "total_trades": sum(yearly_stats[y]["total_trades"] for y in YEARS),
        }

        # Gates
        gates = evaluate_gates(baseline_yearly, yearly_stats, agg)

        profile_results[profile_name] = {
            "yearly": {str(y): yearly_stats[y] for y in YEARS},
            "aggregate": agg,
            "gates": gates,
        }

        bl_avg = round(sum(baseline_yearly[y]["return_pct"] for y in YEARS) / len(YEARS), 2)
        print(f"    AGG: avg={avg_ret:+.1f}% (baseline={bl_avg:+.1f}%)  "
              f"worst_dd={worst_dd:.1f}%  {years_prof}/6 profitable  "
              f"Sharpe={sharpe}  gates={'PASS' if gates['all_pass'] else 'FAIL'}")

    # ── Cross-profile sensitivity for G6 ──────────────────────────────────
    profile_avgs = [profile_results[p]["aggregate"]["avg_return"]
                    for p in SIZING_PROFILES]
    if len(profile_avgs) > 1 and np.mean(profile_avgs) != 0:
        cv = np.std(profile_avgs) / abs(np.mean(profile_avgs))
        c_score = max(0, 1.0 - cv)  # lower CV = more stable = higher score
    else:
        c_score = 1.0

    # Update G6 with real sensitivity score
    for pname, presult in profile_results.items():
        g6 = presult["gates"]["g6_robust_score"]
        old_robust = g6["score"]
        # Recompute with real c_score
        yearly_rets = {y: presult["yearly"][str(y)]["return_pct"] for y in YEARS}
        years_prof = sum(1 for r in yearly_rets.values() if r > 0)
        a_score = years_prof / len(YEARS)

        wf_ratios = []
        for test_yr in [2023, 2024, 2025]:
            train_rets = [yearly_rets[y] for y in YEARS if y < test_yr]
            if train_rets and np.mean(train_rets) > 0:
                ratio = yearly_rets[test_yr] / np.mean(train_rets)
                wf_ratios.append(min(ratio, 2.0))
        b_score = min(float(np.median(wf_ratios)), 1.0) if wf_ratios else 0.0

        robust = a_score * 0.25 + b_score * 0.30 + c_score * 0.25 + 0.10 + 0.10
        g6["score"] = round(robust, 3)
        g6["pass"] = robust >= 0.70
        g6["detail"] = f"ROBUST={robust:.3f} (A={a_score:.2f} B={b_score:.2f} C={c_score:.2f})"
        presult["gates"]["all_pass"] = all(
            presult["gates"][k]["pass"]
            for k in ["g3_avg_return", "g4_no_year_5pct_worse",
                      "g5_2022_not_worse", "g6_robust_score"]
        )

    # ── Pick best profile ─────────────────────────────────────────────────
    best_name = max(
        profile_results,
        key=lambda p: (
            profile_results[p]["gates"]["all_pass"],
            profile_results[p]["aggregate"]["avg_return"],
            -abs(profile_results[p]["aggregate"]["worst_dd"]),
        ),
    )

    # ── Step 5: Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    bl_avg = round(sum(baseline_yearly[y]["return_pct"] for y in YEARS) / len(YEARS), 2)
    bl_worst_dd = round(min(baseline_yearly[y]["max_drawdown"] for y in YEARS), 2)

    print(f"\n  EXP-401 Baseline:  avg={bl_avg:+.1f}%  worst_dd={bl_worst_dd:.1f}%")
    for pname in SIZING_PROFILES:
        agg = profile_results[pname]["aggregate"]
        gates = profile_results[pname]["gates"]
        marker = " <-- BEST" if pname == best_name else ""
        print(f"  {pname:14s}:  avg={agg['avg_return']:+.1f}%  "
              f"worst_dd={agg['worst_dd']:.1f}%  Sharpe={agg['sharpe']}  "
              f"gates={'PASS' if gates['all_pass'] else 'FAIL'}{marker}")

    best = profile_results[best_name]
    print(f"\n  Best profile: {best_name}")
    print(f"  Gates:")
    for gname, gval in best["gates"].items():
        if gname == "all_pass":
            continue
        print(f"    {gname}: {'PASS' if gval['pass'] else 'FAIL'} — {gval['detail']}")
    print(f"    all_pass: {'PASS' if best['gates']['all_pass'] else 'FAIL'}")

    # ── Step 6: Save results ──────────────────────────────────────────────
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

    output = {
        "generated_at": datetime.now().isoformat(),
        "baseline": {
            "yearly": {str(y): baseline_yearly[y] for y in YEARS},
            "avg_return": bl_avg,
            "worst_dd": bl_worst_dd,
            "total_trades": total_trades,
        },
        "profiles": profile_results,
        "best_profile": best_name,
        "best_gates": best["gates"],
        "model_paths": {
            regime: str(ROOT / path)
            for regime, path in json.load(open(TRAINING_RESULTS_PATH))["model_paths"].items()
        },
        "cross_profile_sensitivity": round(c_score, 3),
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
