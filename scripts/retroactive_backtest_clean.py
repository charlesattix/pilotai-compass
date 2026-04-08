#!/usr/bin/env python3
"""
Retroactive ML Backtest with Clean Feature Pipeline
=====================================================
Runs walk-forward ML backtesting using the clean FeaturePipeline
(z-scored prices, ratio features, domain-aware imputation) instead
of raw price/dollar features.

Compares clean-pipeline results against the legacy raw-feature baseline
to quantify the impact of feature stationarity fixes.

Data sources (in priority order):
  1. Fresh retroactive backtest via IronVault + options_cache.db
     (requires data/options_cache.db — run on Charles's Mac Studio)
  2. Existing training CSV at compass/training_data_combined.csv
     (produced by a prior retroactive run — always available in repo)

Usage:
    python scripts/retroactive_backtest_clean.py
    python scripts/retroactive_backtest_clean.py --data compass/training_data_combined.csv
    python scripts/retroactive_backtest_clean.py --fresh --years 2020-2025

When --fresh is passed, the script runs the full retroactive pipeline:
  1. Load options_cache.db via IronVault
  2. Run backtester per year to generate trades
  3. Enrich trades with market context (collect_training_data.enrich_trades)
  4. Apply FeaturePipeline transforms
  5. Walk-forward ML train/eval

Without --fresh, it reads the pre-existing CSV and applies FeaturePipeline
in-place (suitable for CI or machines without options_cache.db).
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost required — pip install xgboost")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from compass.feature_pipeline import FeaturePipeline
from shared.indicators import sanitize_features

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────

DEFAULT_DATA = PROJECT_ROOT / "compass" / "training_data_combined.csv"
OPTIONS_CACHE = PROJECT_ROOT / "data" / "options_cache.db"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Walk-forward folds (anchored expanding window) ──────────────────────────

FOLDS = [
    {"train_years": [2020, 2021, 2022], "test_year": 2023},
    {"train_years": [2020, 2021, 2022, 2023], "test_year": 2024},
    {"train_years": [2020, 2021, 2022, 2023, 2024], "test_year": 2025},
]

# ── Legacy (raw) feature columns — same as ml_walkforward_train.py ──────────

LEGACY_RAW_FEATURES = [
    'day_of_week', 'days_since_last_trade', 'dte_at_entry',
    'rsi_14', 'momentum_5d_pct', 'momentum_10d_pct',
    'vix', 'vix_percentile_20d', 'vix_percentile_50d', 'vix_percentile_100d',
    'iv_rank',
    'dist_from_ma20_pct', 'dist_from_ma50_pct', 'dist_from_ma80_pct', 'dist_from_ma200_pct',
    'ma20_slope_ann_pct', 'ma50_slope_ann_pct',
    'realized_vol_atr20', 'realized_vol_5d', 'realized_vol_10d', 'realized_vol_20d',
]

LEGACY_CAT_COLS = ['strategy_type', 'regime']

LABEL_COL = 'win'

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.08,
    'n_estimators': 150,
    'min_child_weight': 8,
    'subsample': 0.75,
    'colsample_bytree': 0.7,
    'gamma': 2,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'random_state': 42,
    'eval_metric': 'logloss',
}


# ── Data loading ────────────────────────────────────────────────────────────

def load_csv_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d trades from %s", len(df), path)
    return df


def try_fresh_retroactive(years: List[int], ticker: str = "SPY") -> Optional[pd.DataFrame]:
    """Attempt a fresh retroactive backtest using IronVault + options_cache.db.

    Returns enriched trade DataFrame, or None if options_cache.db is unavailable.
    """
    if not OPTIONS_CACHE.exists():
        logger.warning("options_cache.db not found at %s — cannot run fresh retroactive", OPTIONS_CACHE)
        return None

    try:
        from shared.iron_vault import IronVault
        from backtest.backtester import Backtester
        from compass.collect_training_data import enrich_trades
    except ImportError as e:
        logger.warning("Cannot import backtest modules: %s", e)
        return None

    hd = IronVault.instance()
    all_trades = []

    for year in years:
        logger.info("Running retroactive backtest for %d...", year)
        config = {
            "starting_capital": 100_000,
            "direction": "both",
            "spread_width": 12,
            "min_dte": 15,
            "max_dte": 25,
            "target_dte": 15,
            "otm_pct": 0.02,
            "max_risk_per_trade": 8.5,
            "profit_target": 55,
            "stop_loss_multiplier": 1.25,
            "regime_mode": "combo",
        }
        bt = Backtester(config, historical_data=hd, otm_pct=0.02)
        bt.run_backtest(ticker, datetime(year, 1, 1), datetime(year, 12, 31))
        enriched = enrich_trades(bt, year)
        all_trades.extend(enriched)

    if not all_trades:
        return None

    return pd.DataFrame(all_trades)


# ── Feature preparation ─────────────────────────────────────────────────────

def prepare_legacy_features(df: pd.DataFrame) -> tuple:
    """Prepare features using the legacy (raw) pipeline.

    Returns (feature_cols, df_with_dummies).
    """
    out = df.copy()
    for col in LEGACY_CAT_COLS:
        if col in out.columns:
            dummies = pd.get_dummies(out[col], prefix=col, dtype=float)
            out = pd.concat([out, dummies], axis=1)

    encoded_cols = sorted(
        c for c in out.columns
        if any(c.startswith(f"{cat}_") for cat in LEGACY_CAT_COLS)
    )
    feature_cols = LEGACY_RAW_FEATURES + encoded_cols
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return feature_cols, out


def prepare_clean_features(df: pd.DataFrame) -> tuple:
    """Prepare features using the new FeaturePipeline (z-scored, ratios).

    Returns (feature_cols, transformed_df_aligned_with_original_index).
    """
    pipeline = FeaturePipeline()
    transformed = pipeline.transform(df)
    feature_cols = list(transformed.columns)
    return feature_cols, transformed


# ── Walk-forward evaluation ─────────────────────────────────────────────────

def walk_forward_eval(
    df: pd.DataFrame,
    feature_cols: List[str],
    features_df: pd.DataFrame,
    label_col: str = LABEL_COL,
    folds: list = None,
) -> List[Dict]:
    """Run anchored walk-forward evaluation.

    Returns list of per-fold result dicts.
    """
    if folds is None:
        folds = FOLDS

    results = []
    for fold in folds:
        train_years = fold["train_years"]
        test_year = fold["test_year"]

        train_mask = df["year"].isin(train_years)
        test_mask = df["year"] == test_year

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.warning("Fold %s→%d: no data, skipping", train_years, test_year)
            continue

        X_train = features_df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, label_col].values
        X_test = features_df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, label_col].values

        X_train = sanitize_features(X_train.astype(np.float64))
        X_test = sanitize_features(X_test.astype(np.float64))

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.warning("Fold %s→%d: degenerate labels, skipping", train_years, test_year)
            continue

        # Train with early stopping on validation split
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train,
        )
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = float(roc_auc_score(y_test, y_proba))
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))

        # Feature importance
        importances = dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
        top_feat = max(importances, key=importances.get)

        results.append({
            "test_year": test_year,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "train_win_rate": float(y_train.mean()),
            "test_win_rate": float(y_test.mean()),
            "auc": auc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "top_feature": top_feat,
            "top_feature_importance": importances[top_feat],
            "feature_importances": importances,
        })

    return results


# ── Reporting ───────────────────────────────────────────────────────────────

def _bar(label, width=70):
    print(f"\n{'─' * width}")
    print(f"  {label}")
    print(f"{'─' * width}")


def print_fold_results(label: str, folds: List[Dict]):
    _bar(f"WALK-FORWARD: {label}")
    header = f"  {'Year':>6s}  {'N_tr':>5s}  {'N_te':>5s}  {'AUC':>7s}  {'Acc':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'Top Feature':<30s}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    for f in folds:
        print(
            f"  {f['test_year']:>6d}  {f['n_train']:>5d}  {f['n_test']:>5d}  "
            f"{f['auc']:>7.4f}  {f['accuracy']:>7.4f}  {f['precision']:>7.4f}  "
            f"{f['recall']:>7.4f}  {f['top_feature']:<30s}"
        )
    if folds:
        aucs = [f["auc"] for f in folds]
        accs = [f["accuracy"] for f in folds]
        print(f"  {'─' * (len(header) - 2)}")
        print(f"  {'AVG':>6s}  {'':>5s}  {'':>5s}  {np.mean(aucs):>7.4f}  {np.mean(accs):>7.4f}")


def print_comparison(legacy_folds: List[Dict], clean_folds: List[Dict]):
    _bar("COMPARISON: Legacy (raw) vs Clean (z-scored/ratios)")
    header = f"  {'Year':>6s}  {'Legacy AUC':>11s}  {'Clean AUC':>10s}  {'Delta':>8s}  {'Legacy Acc':>11s}  {'Clean Acc':>10s}  {'Delta':>8s}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    legacy_map = {f["test_year"]: f for f in legacy_folds}
    clean_map = {f["test_year"]: f for f in clean_folds}

    auc_deltas = []
    acc_deltas = []

    for year in sorted(set(legacy_map) | set(clean_map)):
        l = legacy_map.get(year)
        c = clean_map.get(year)
        if l and c:
            auc_d = c["auc"] - l["auc"]
            acc_d = c["accuracy"] - l["accuracy"]
            auc_deltas.append(auc_d)
            acc_deltas.append(acc_d)
            sign_a = "+" if auc_d >= 0 else ""
            sign_c = "+" if acc_d >= 0 else ""
            print(
                f"  {year:>6d}  {l['auc']:>11.4f}  {c['auc']:>10.4f}  {sign_a}{auc_d:>7.4f}  "
                f"{l['accuracy']:>11.4f}  {c['accuracy']:>10.4f}  {sign_c}{acc_d:>7.4f}"
            )

    if auc_deltas:
        print(f"  {'─' * (len(header) - 2)}")
        avg_auc_d = np.mean(auc_deltas)
        avg_acc_d = np.mean(acc_deltas)
        sign_a = "+" if avg_auc_d >= 0 else ""
        sign_c = "+" if avg_acc_d >= 0 else ""
        print(f"  {'AVG':>6s}  {'':>11s}  {'':>10s}  {sign_a}{avg_auc_d:>7.4f}  "
              f"{'':>11s}  {'':>10s}  {sign_c}{avg_acc_d:>7.4f}")

    print()
    if auc_deltas:
        verdict = "CLEAN WINS" if np.mean(auc_deltas) > 0 else "LEGACY WINS" if np.mean(auc_deltas) < -0.01 else "TIED"
        print(f"  Verdict: {verdict} (avg AUC delta: {sign_a}{np.mean(auc_deltas):.4f})")


def save_report(
    legacy_folds: List[Dict],
    clean_folds: List[Dict],
    data_source: str,
    output_path: Path,
):
    """Save comparison report as JSON."""
    report = {
        "generated": datetime.now().isoformat(),
        "data_source": data_source,
        "folds": [f["test_year"] for f in FOLDS],
        "legacy_pipeline": {
            "description": "Raw features: spy_price, vix (absolute), contracts (raw), 0-fill imputation",
            "n_features": len(LEGACY_RAW_FEATURES),
            "folds": [{k: v for k, v in f.items() if k != "feature_importances"} for f in legacy_folds],
            "avg_auc": round(np.mean([f["auc"] for f in legacy_folds]), 4) if legacy_folds else None,
            "avg_accuracy": round(np.mean([f["accuracy"] for f in legacy_folds]), 4) if legacy_folds else None,
        },
        "clean_pipeline": {
            "description": "Z-scored prices, ratio features, domain-aware imputation, log-contracts",
            "n_features": None,  # filled below
            "folds": [{k: v for k, v in f.items() if k != "feature_importances"} for f in clean_folds],
            "avg_auc": round(np.mean([f["auc"] for f in clean_folds]), 4) if clean_folds else None,
            "avg_accuracy": round(np.mean([f["accuracy"] for f in clean_folds]), 4) if clean_folds else None,
        },
    }

    if clean_folds:
        report["clean_pipeline"]["n_features"] = len(clean_folds[0].get("feature_importances", {}))

    # Deltas
    if legacy_folds and clean_folds:
        auc_deltas = []
        for lf, cf in zip(legacy_folds, clean_folds):
            auc_deltas.append(cf["auc"] - lf["auc"])
        report["auc_delta_avg"] = round(np.mean(auc_deltas), 4)
        report["verdict"] = "clean_wins" if np.mean(auc_deltas) > 0 else "legacy_wins" if np.mean(auc_deltas) < -0.01 else "tied"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", output_path)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retroactive ML backtest: clean vs legacy features")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Training CSV path")
    parser.add_argument("--fresh", action="store_true", help="Run fresh retroactive via options_cache.db")
    parser.add_argument("--years", default="2020-2025", help="Years for fresh retroactive")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "retroactive_clean_vs_legacy.json")
    args = parser.parse_args()

    # 1. Load data
    if args.fresh:
        parts = args.years.split("-")
        years = list(range(int(parts[0]), int(parts[1]) + 1))
        df = try_fresh_retroactive(years)
        if df is None:
            print("\n  options_cache.db not available. Falling back to CSV data.")
            print("  To run a fresh retroactive backtest:")
            print(f"    1. Copy options_cache.db to {OPTIONS_CACHE}")
            print("    2. Re-run with: python scripts/retroactive_backtest_clean.py --fresh\n")
            if not args.data.exists():
                print(f"  ERROR: CSV fallback not found at {args.data}")
                sys.exit(1)
            df = load_csv_data(args.data)
            data_source = f"csv_fallback:{args.data}"
        else:
            data_source = f"fresh_retroactive:{args.years}"
    else:
        if not args.data.exists():
            print(f"ERROR: training data not found at {args.data}")
            sys.exit(1)
        df = load_csv_data(args.data)
        data_source = f"csv:{args.data}"

    print(f"\n  Data source: {data_source}")
    print(f"  Trades: {len(df)}")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Win rate: {df[LABEL_COL].mean():.3f}")

    # 2. Run legacy pipeline
    _bar("LEGACY PIPELINE (raw features)")
    legacy_cols, legacy_df = prepare_legacy_features(df)
    print(f"  Features: {len(legacy_cols)}")

    t0 = time.time()
    legacy_folds = walk_forward_eval(df, legacy_cols, legacy_df)
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print_fold_results("Legacy (raw features)", legacy_folds)

    # 3. Run clean pipeline
    _bar("CLEAN PIPELINE (z-scored / ratios)")
    clean_cols, clean_df = prepare_clean_features(df)
    # Align index — FeaturePipeline.transform preserves the input index
    clean_df_aligned = clean_df.reindex(df.index)
    print(f"  Features: {len(clean_cols)}")

    t0 = time.time()
    clean_folds = walk_forward_eval(df, clean_cols, clean_df_aligned)
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print_fold_results("Clean (z-scored / ratios)", clean_folds)

    # 4. Comparison
    if legacy_folds and clean_folds:
        print_comparison(legacy_folds, clean_folds)

    # 5. Save report
    save_report(legacy_folds, clean_folds, data_source, args.output)
    print(f"\n  Report: {args.output}")

    # 6. Note for Charles
    if not OPTIONS_CACHE.exists():
        _bar("NOTE: Fresh retroactive backtest requires options_cache.db")
        print("  The comparison above uses the existing training CSV.")
        print("  For a full retroactive backtest on real options data:")
        print()
        print("    1. On the Mac Studio, ensure options_cache.db is populated:")
        print("       python scripts/backfill_polygon_cache.py --years 2020-2025")
        print()
        print(f"   2. Copy to this machine:")
        print(f"       scp mac-studio:~/pilotai/data/options_cache.db {OPTIONS_CACHE}")
        print()
        print("    3. Re-run with --fresh flag:")
        print("       python scripts/retroactive_backtest_clean.py --fresh --years 2020-2025")
        print()

    print()


if __name__ == "__main__":
    main()
