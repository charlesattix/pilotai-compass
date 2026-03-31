"""
CS-Only Benchmark: XGBoost vs Ensemble vs Baseline (No Model)

Walk-forward validation on credit-spread (CS) trades only.
Answers the key question: does ML meaningfully improve trade selection
over the 84% base win rate?

Results saved to compass/benchmark_results_cs_only.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import xgboost as xgb

# Add project root to path for shared/ imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.indicators import sanitize_features

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "training_data_combined.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_results_cs_only.json")

XGB_PARAMS = {
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
    "verbosity": 0,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 5,   # relaxed from 10 for smaller CS dataset
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

ET_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

MIN_TRAIN_SAMPLES = 20  # lower threshold for CS-only (smaller dataset)


# ── Feature preparation ──────────────────────────────────────────────────────

def _zscore_column(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score, clipped to [-4, 4]."""
    roll_mean = series.expanding(min_periods=1).mean()
    roll_std = series.expanding(min_periods=1).std()
    if len(series) > window:
        rm_fixed = series.rolling(window, min_periods=max(10, window // 3)).mean()
        rs_fixed = series.rolling(window, min_periods=max(10, window // 3)).std()
        mask = rm_fixed.notna()
        roll_mean = roll_mean.where(~mask, rm_fixed)
        roll_std = roll_std.where(~mask, rs_fixed)
    roll_std = roll_std.replace(0, np.nan)
    z = (series - roll_mean) / roll_std
    return z.fillna(0.0).clip(-4.0, 4.0)


def prepare_cs_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a clean feature matrix for CS-only data.

    Inlines the FeaturePipeline transforms to avoid circular imports via
    compass/__init__.py (which requires requests/SQLAlchemy etc).

    strategy_type is constant (all CS) — dropped to avoid trivial feature.
    spread_type is one-hot encoded (bull/bear spread direction may add signal).
    """
    out = df.copy()

    # Z-score non-stationary price levels
    if "spy_price" in out.columns:
        out["spy_price_zscore"] = _zscore_column(out["spy_price"].astype(float))
    else:
        out["spy_price_zscore"] = 0.0

    if "vix" in out.columns:
        out["vix_zscore"] = _zscore_column(out["vix"].astype(float))
    else:
        out["vix_zscore"] = 0.0

    # VIX change as % of current VIX (not raw points)
    if "vix_change_5d" in out.columns and "vix" in out.columns:
        vix = out["vix"].replace(0, np.nan)
        out["vix_change_5d_pct"] = (out["vix_change_5d"] / vix * 100).fillna(0.0)
    else:
        out["vix_change_5d_pct"] = 0.0

    # Log-transform contracts
    if "contracts" in out.columns:
        out["contracts_log"] = np.sign(out["contracts"]) * np.log1p(out["contracts"].abs().fillna(0))
    else:
        out["contracts_log"] = 0.0

    # Trade structure ratios (replace dollar amounts)
    def credit_to_width(row):
        sw, nc = row.get("spread_width", 0), row.get("net_credit", 0)
        if pd.isna(sw) or pd.isna(nc) or sw <= 0:
            return 0.0
        return nc / sw

    def loss_to_width(row):
        sw, ml = row.get("spread_width", 0), row.get("max_loss_per_unit", 0)
        if pd.isna(sw) or pd.isna(ml) or sw <= 0:
            return 0.0
        return ml / sw

    out["credit_to_width"] = out.apply(credit_to_width, axis=1)
    out["loss_to_width"] = out.apply(loss_to_width, axis=1)

    # Numeric features (stationary)
    NUMERIC = [
        "days_since_last_trade",
        "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
        "vix_zscore", "vix_change_5d_pct",
        "vix_percentile_50d", "vix_percentile_100d", "iv_rank",
        "spy_price_zscore",
        "dist_from_ma20_pct", "dist_from_ma50_pct", "dist_from_ma80_pct",
        "dist_from_ma200_pct", "ma50_slope_ann_pct",
        "realized_vol_atr20", "realized_vol_20d",
        "credit_to_width", "loss_to_width", "contracts_log",
        "dte_at_entry", "hold_days",
    ]

    # Domain-aware imputation defaults
    IMPUTE = {
        "iv_rank": 50.0, "vix_percentile_50d": 50.0, "vix_percentile_100d": 50.0,
        "rsi_14": 50.0, "realized_vol_atr20": 20.0, "realized_vol_20d": 20.0,
    }

    num_df = pd.DataFrame(index=out.index)
    for col in NUMERIC:
        if col in out.columns:
            default = IMPUTE.get(col, 0.0)
            num_df[col] = out[col].fillna(default).astype(float)
        else:
            num_df[col] = IMPUTE.get(col, 0.0)

    # Categoricals: regime and spread_type (strategy_type is constant=CS, skip)
    parts = [num_df]
    for cat in ["regime", "spread_type"]:
        if cat in out.columns:
            dummies = pd.get_dummies(out[cat], prefix=cat, dummy_na=False)
            parts.append(dummies)

    result = pd.concat(parts, axis=1)
    # Sanitize any remaining inf/nan — build new DataFrame to avoid dtype issues
    arr = result.values.astype(np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return pd.DataFrame(arr, columns=result.columns, index=result.index)


# ── Sklearn wrapper for the 3-model ensemble ────────────────────────────────

class EnsembleWrapper(BaseEstimator):
    """Sklearn-compatible wrapper around XGB + RF + ET ensemble.

    Implements fit/predict/predict_proba so it can be used anywhere a
    standard sklearn estimator is expected (including clone()).
    """

    def __init__(self):
        self.models_: Optional[Dict[str, Any]] = None
        self.weights_: Optional[Dict[str, float]] = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleWrapper":
        X = sanitize_features(X)
        base_models = [
            ("xgboost", xgb.XGBClassifier(**XGB_PARAMS)),
            ("random_forest", RandomForestClassifier(**RF_PARAMS)),
            ("extra_trees", ExtraTreesClassifier(**ET_PARAMS)),
        ]

        # Walk-forward weights using 4-fold (smaller dataset)
        n_folds = 4
        fold_size = len(y) // n_folds
        model_aucs: Dict[str, List[float]] = {name: [] for name, _ in base_models}

        if fold_size >= 10:
            for k in range(1, n_folds):
                X_tr, y_tr = X[:fold_size * k], y[:fold_size * k]
                X_val, y_val = X[fold_size * k:fold_size * (k + 1)], y[fold_size * k:fold_size * (k + 1)]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                    continue
                for name, base_est in base_models:
                    try:
                        est = clone(base_est)
                        est.fit(X_tr, y_tr)
                        proba = est.predict_proba(X_val)[:, 1]
                        if len(np.unique(y_val)) == 2:
                            model_aucs[name].append(roc_auc_score(y_val, proba))
                    except Exception:
                        pass

        # Compute weights from AUC edge
        mean_aucs = {
            name: float(np.mean(scores)) if scores else 0.5
            for name, scores in model_aucs.items()
        }
        edges = {name: max(0.0, auc - 0.5) for name, auc in mean_aucs.items()}
        total_edge = sum(edges.values())
        if total_edge < 1e-9:
            weights = {name: 1.0 / len(base_models) for name, _ in base_models}
        else:
            weights = {name: edge / total_edge for name, edge in edges.items()}
        self.weights_ = weights

        # Train final models on full training set
        self.models_ = {}
        for name, base_est in base_models:
            est = clone(base_est)
            est.fit(X, y)
            self.models_[name] = est

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = sanitize_features(X)
        weighted_sum = np.zeros(X.shape[0])
        total_w = 0.0
        for name, model in self.models_.items():
            w = self.weights_.get(name, 0.0)
            if w <= 0:
                continue
            proba = model.predict_proba(X)[:, 1]
            weighted_sum += w * proba
            total_w += w
        if total_w < 1e-9:
            n = len(self.models_)
            for model in self.models_.values():
                weighted_sum += model.predict_proba(X)[:, 1]
            return np.column_stack([1 - weighted_sum / n, weighted_sum / n])
        p1 = weighted_sum / total_w
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ── Walk-forward validation (year-based expanding window) ───────────────────

def walk_forward_cs(
    df: pd.DataFrame,
    features: pd.DataFrame,
    model: BaseEstimator,
    model_name: str,
) -> Dict[str, Any]:
    """Run walk-forward validation on CS-only data.

    Returns per-fold and aggregate metrics.
    """
    df = df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    years = sorted(df["entry_date"].dt.year.unique())

    if len(years) < 2:
        raise ValueError(f"Need >= 2 years, got {years}")

    fold_results = []
    all_labels, all_preds, all_probas, all_returns = [], [], [], []

    for fold_idx in range(len(years) - 1):
        train_years = years[: fold_idx + 1]
        test_year = years[fold_idx + 1]

        train_mask = df["entry_date"].dt.year.isin(train_years)
        test_mask = df["entry_date"].dt.year == test_year

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < MIN_TRAIN_SAMPLES:
            logger.info("Fold %d: skipping, %d train samples < %d min", fold_idx, n_train, MIN_TRAIN_SAMPLES)
            continue
        if n_test == 0:
            continue

        X_train = sanitize_features(features.loc[train_mask].values)
        y_train = df.loc[train_mask, "win"].values.astype(int)
        X_test = sanitize_features(features.loc[test_mask].values)
        y_test = df.loc[test_mask, "win"].values.astype(int)
        test_returns = df.loc[test_mask, "return_pct"].values

        # Base win rate this fold
        base_wr = float(y_test.mean())

        # Train and predict
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        if hasattr(fold_model, "predict_proba"):
            y_proba = fold_model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        brier = brier_score_loss(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else None

        # Signal Sharpe: returns on trades model predicts as wins
        signal_sharpe = None
        signal_mask = y_proba > 0.5
        if signal_mask.sum() >= 2:
            sig_ret = test_returns[signal_mask]
            m, s = np.mean(sig_ret), np.std(sig_ret, ddof=1)
            if s > 0:
                signal_sharpe = float(m / s * np.sqrt(52))

        # "Filter lift": WR on model-selected trades vs base WR
        selected_wr = float(y_test[signal_mask].mean()) if signal_mask.sum() > 0 else base_wr
        filter_lift = selected_wr - base_wr

        # Trades skipped (predicted loss)
        n_skipped = int((y_proba <= 0.5).sum())
        n_selected = int(signal_mask.sum())

        train_dates = df.loc[train_mask, "entry_date"]
        test_dates = df.loc[test_mask, "entry_date"]

        fold = {
            "fold": fold_idx,
            "train_years": [int(y) for y in train_years],
            "test_year": int(test_year),
            "train_period": f"{train_dates.min().date()} → {train_dates.max().date()}",
            "test_period": f"{test_dates.min().date()} → {test_dates.max().date()}",
            "n_train": int(n_train),
            "n_test": int(n_test),
            "base_win_rate": round(base_wr, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "brier_score": round(brier, 4),
            "auc": round(auc, 4) if auc is not None else None,
            "signal_sharpe": round(signal_sharpe, 4) if signal_sharpe is not None else None,
            "n_selected": n_selected,
            "n_skipped": n_skipped,
            "selected_win_rate": round(selected_wr, 4),
            "filter_lift": round(filter_lift, 4),
        }
        fold_results.append(fold)
        all_labels.append(y_test)
        all_preds.append(y_pred)
        all_probas.append(y_proba)
        all_returns.append(test_returns)

        logger.info(
            "[%s] Fold %d (test=%d) | base_wr=%.3f acc=%.3f prec=%.3f brier=%.4f "
            "auc=%s sharpe=%s lift=%+.3f",
            model_name, fold_idx, test_year,
            base_wr, acc, prec, brier,
            f"{auc:.3f}" if auc else "N/A",
            f"{signal_sharpe:.3f}" if signal_sharpe else "N/A",
            filter_lift,
        )

    # Aggregate across folds
    def mean_std(vals):
        v = [x for x in vals if x is not None]
        return (round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)) if len(v) > 1 else (round(float(v[0]), 4) if v else None, 0.0)

    all_labels_cat = np.concatenate(all_labels)
    all_preds_cat = np.concatenate(all_preds)
    all_probas_cat = np.concatenate(all_probas)
    all_returns_cat = np.concatenate(all_returns)

    oos_acc = accuracy_score(all_labels_cat, all_preds_cat)
    oos_brier = brier_score_loss(all_labels_cat, all_probas_cat)
    oos_auc = roc_auc_score(all_labels_cat, all_probas_cat) if len(np.unique(all_labels_cat)) == 2 else None
    oos_base_wr = float(all_labels_cat.mean())
    oos_selected = all_probas_cat > 0.5
    oos_selected_wr = float(all_labels_cat[oos_selected].mean()) if oos_selected.sum() > 0 else oos_base_wr

    # OOS signal Sharpe on all OOS trades
    oos_sharpe = None
    if oos_selected.sum() >= 2:
        sig_ret = all_returns_cat[oos_selected]
        m, s = np.mean(sig_ret), np.std(sig_ret, ddof=1)
        if s > 0:
            oos_sharpe = float(m / s * np.sqrt(52))

    aggregate = {
        "accuracy_mean": mean_std([f["accuracy"] for f in fold_results])[0],
        "accuracy_std": mean_std([f["accuracy"] for f in fold_results])[1],
        "precision_mean": mean_std([f["precision"] for f in fold_results])[0],
        "brier_mean": mean_std([f["brier_score"] for f in fold_results])[0],
        "auc_mean": mean_std([f["auc"] for f in fold_results])[0],
        "signal_sharpe_mean": mean_std([f["signal_sharpe"] for f in fold_results])[0],
        "filter_lift_mean": mean_std([f["filter_lift"] for f in fold_results])[0],
        "oos_accuracy": round(oos_acc, 4),
        "oos_brier": round(oos_brier, 4),
        "oos_auc": round(oos_auc, 4) if oos_auc else None,
        "oos_base_win_rate": round(oos_base_wr, 4),
        "oos_selected_win_rate": round(oos_selected_wr, 4),
        "oos_filter_lift": round(oos_selected_wr - oos_base_wr, 4),
        "oos_signal_sharpe": round(oos_sharpe, 4) if oos_sharpe else None,
        "n_folds": len(fold_results),
        "total_oos_samples": int(len(all_labels_cat)),
    }

    return {"model": model_name, "folds": fold_results, "aggregate": aggregate}


# ── Baseline: no model (always predict win) ──────────────────────────────────

def compute_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """Baseline: always trade every CS signal (no filter). Per-year breakdown."""
    df = df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    years = sorted(df["entry_date"].dt.year.unique())

    yearly = []
    for yr in years:
        mask = df["entry_date"].dt.year == yr
        yr_df = df[mask]
        wr = float(yr_df["win"].mean())
        n = int(len(yr_df))
        mean_ret = float(yr_df["return_pct"].mean())
        yearly.append({
            "year": int(yr),
            "n_trades": n,
            "win_rate": round(wr, 4),
            "mean_return_pct": round(mean_ret, 4),
        })

    overall_wr = float(df["win"].mean())
    overall_ret = float(df["return_pct"].mean())

    # Baseline Sharpe (all trades, annualized)
    m = df["return_pct"].mean()
    s = df["return_pct"].std(ddof=1)
    baseline_sharpe = float(m / s * np.sqrt(52)) if s > 0 else None

    return {
        "model": "baseline_no_filter",
        "overall_win_rate": round(overall_wr, 4),
        "overall_mean_return_pct": round(overall_ret, 4),
        "overall_sharpe": round(baseline_sharpe, 4) if baseline_sharpe else None,
        "by_year": yearly,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== CS-Only Benchmark: XGBoost vs Ensemble vs Baseline ===")

    # 1. Load data, filter to CS only
    df_all = pd.read_csv(DATA_PATH)
    df = df_all[df_all["strategy_type"] == "CS"].copy().reset_index(drop=True)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    logger.info("CS trades: %d  |  overall WR: %.4f", len(df), df["win"].mean())
    logger.info("Years: %s", sorted(df["entry_date"].dt.year.unique().tolist()))

    # 2. Build features using FeaturePipeline
    logger.info("Building features via FeaturePipeline...")
    features = prepare_cs_features(df)
    logger.info("Feature matrix: %s  |  columns: %s", features.shape, list(features.columns))

    # 3. Baseline (no model)
    logger.info("\n--- BASELINE (no filter) ---")
    baseline = compute_baseline(df)
    logger.info("Overall WR: %.4f  |  mean return: %.2f%%  |  Sharpe: %s",
                baseline["overall_win_rate"],
                baseline["overall_mean_return_pct"],
                baseline["overall_sharpe"])

    # 4. XGBoost walk-forward
    logger.info("\n--- XGBoost Walk-Forward ---")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_results = walk_forward_cs(df, features, xgb_model, "xgboost")

    # 5. Ensemble walk-forward
    logger.info("\n--- Ensemble (XGB+RF+ET) Walk-Forward ---")
    ensemble_model = EnsembleWrapper()
    ens_results = walk_forward_cs(df, features, ensemble_model, "ensemble")

    # 6. Comparison summary
    logger.info("\n=== RESULTS SUMMARY ===")
    for res in [xgb_results, ens_results]:
        agg = res["aggregate"]
        logger.info(
            "[%s] OOS acc=%.4f brier=%.4f auc=%s base_wr=%.4f selected_wr=%.4f lift=%+.4f sharpe=%s",
            res["model"],
            agg["oos_accuracy"],
            agg["oos_brier"],
            f"{agg['oos_auc']:.4f}" if agg["oos_auc"] else "N/A",
            agg["oos_base_win_rate"],
            agg["oos_selected_win_rate"],
            agg["oos_filter_lift"],
            f"{agg['oos_signal_sharpe']:.4f}" if agg["oos_signal_sharpe"] else "N/A",
        )

    # 7. Save results
    output = {
        "meta": {
            "description": "CS-only walk-forward benchmark: XGBoost vs Ensemble vs Baseline",
            "date_run": "2026-03-26",
            "n_cs_trades": int(len(df)),
            "overall_cs_win_rate": round(float(df["win"].mean()), 4),
            "data_file": "compass/training_data_combined.csv",
            "feature_count": int(features.shape[1]),
            "features_used": list(features.columns),
            "years": sorted(df["entry_date"].dt.year.unique().tolist()),
        },
        "baseline": baseline,
        "xgboost": xgb_results,
        "ensemble": ens_results,
        "comparison": {
            "baseline_wr": round(float(df["win"].mean()), 4),
            "xgb_selected_wr": xgb_results["aggregate"]["oos_selected_win_rate"],
            "ens_selected_wr": ens_results["aggregate"]["oos_selected_win_rate"],
            "xgb_lift": xgb_results["aggregate"]["oos_filter_lift"],
            "ens_lift": ens_results["aggregate"]["oos_filter_lift"],
            "xgb_oos_auc": xgb_results["aggregate"]["oos_auc"],
            "ens_oos_auc": ens_results["aggregate"]["oos_auc"],
            "xgb_sharpe": xgb_results["aggregate"]["oos_signal_sharpe"],
            "ens_sharpe": ens_results["aggregate"]["oos_signal_sharpe"],
            "baseline_sharpe": baseline["overall_sharpe"],
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("\nResults saved to %s", OUTPUT_PATH)
    logger.info("\n=== KEY FINDINGS ===")
    cmp = output["comparison"]
    logger.info("Baseline WR (take all trades): %.1f%%", cmp["baseline_wr"] * 100)
    logger.info("XGBoost selected WR:           %.1f%% (lift %+.1f%%)",
                cmp["xgb_selected_wr"] * 100, cmp["xgb_lift"] * 100)
    logger.info("Ensemble selected WR:          %.1f%% (lift %+.1f%%)",
                cmp["ens_selected_wr"] * 100, cmp["ens_lift"] * 100)
    logger.info("XGBoost OOS AUC: %s", cmp["xgb_oos_auc"])
    logger.info("Ensemble OOS AUC: %s", cmp["ens_oos_auc"])


if __name__ == "__main__":
    main()
