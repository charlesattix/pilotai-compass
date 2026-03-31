"""
Tier 1 Feature Benchmark: Original vs Original+Tier1 features
==============================================================

Rebuilds training_data_combined.csv with 5 new Tier 1 features, then
runs walk-forward validation for both feature sets with XGBoost and
EnsembleSignalModel.

New features (commit 05c533c):
  vix_contango_ratio    VIX3M / VIX spot  (VRP proxy; contango >1 = good for sellers)
  spy_tlt_corr_20d      SPY/TLT 20-day rolling correlation (risk-on/risk-off regime)
  hyg_lqd_ratio         HYG / LQD ratio   (credit stress; falling = worsening HY spreads)
  hyg_lqd_ratio_5d_chg  5-day % change in HYG/LQD ratio
  days_to_opex          Calendar days to next 3rd-Friday OPEX  (replaces is_opex_week)
  opening_gap_pct       (today_open - prev_close) / prev_close × 100

Usage:
    cd /home/node/openclaw/workspace/pilotai-credit-spreads
    PYTHONPATH=. python3 compass/benchmark_tier1_features.py

Outputs:
    compass/training_data_combined_tier1.csv   — augmented dataset
    compass/benchmark_tier1_results.json       — raw metrics
    compass/benchmark_tier1_features.md        — analysis report
"""

from __future__ import annotations

import calendar as _cal
import importlib.util as _ilu
import json
import logging
import sys
import tempfile
import types
from datetime import date as _date
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

# ── Project root on sys.path ─────────────────────────────────────────────
HERE = Path(__file__).parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Import compass submodules directly to avoid __init__.py side-effects
def _import_module(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

for _stub in ["compass"]:
    if _stub not in sys.modules:
        sys.modules[_stub] = types.ModuleType(_stub)

_shared = ROOT / "shared"
_import_module("shared", str(_shared / "__init__.py"))
_import_module("shared.indicators", str(_shared / "indicators.py"))
_import_module("shared.types", str(_shared / "types.py"))
_import_module("compass.feature_pipeline", str(HERE / "feature_pipeline.py"))
_import_module("compass.ensemble_signal_model", str(HERE / "ensemble_signal_model.py"))

from compass.ensemble_signal_model import EnsembleSignalModel
from compass.feature_pipeline import FeaturePipeline, _zscore_column, _IMPUTATION_DEFAULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tier1_benchmark")

# ── Paths ────────────────────────────────────────────────────────────────
COMBINED_PATH = HERE / "training_data_combined.csv"
TIER1_PATH    = HERE / "training_data_combined_tier1.csv"
JSON_PATH     = HERE / "benchmark_tier1_results.json"
MD_PATH       = HERE / "benchmark_tier1_features.md"

# ── Walk-forward config ──────────────────────────────────────────────────
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
MIN_TRAIN_SAMPLES = 40


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Download market data for Tier 1 features
# ═══════════════════════════════════════════════════════════════════════════

def _download_yf(tickers: list, start: str, end: str) -> Dict[str, pd.Series]:
    """Download adjusted close prices for multiple tickers. Returns {ticker: Series}."""
    logger.info("Downloading %s from %s to %s via yfinance...", tickers, start, end)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    result: Dict[str, pd.Series] = {}
    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                close = raw["Close"][ticker].dropna()
                result[ticker] = close
            except KeyError:
                logger.warning("No Close data for %s", ticker)
                result[ticker] = pd.Series(dtype=float)
        # Also grab Open for SPY gap calculation
        try:
            result["SPY_Open"] = raw["Open"]["SPY"].dropna()
        except KeyError:
            result["SPY_Open"] = pd.Series(dtype=float)
    else:
        # Single ticker case
        ticker = tickers[0]
        result[ticker] = raw["Close"].dropna()
        if "Open" in raw.columns:
            result[f"{ticker}_Open"] = raw["Open"].dropna()
    return result


def download_tier1_data(start: str = "2018-01-01", end: str = "2026-01-01") -> Dict[str, pd.Series]:
    """Download all data needed for Tier 1 feature computation."""
    tickers = ["^VIX3M", "SPY", "TLT", "HYG", "LQD"]
    data = _download_yf(tickers, start, end)
    logger.info(
        "Downloaded: VIX3M=%d rows, SPY=%d, TLT=%d, HYG=%d, LQD=%d",
        len(data.get("^VIX3M", [])),
        len(data.get("SPY", [])),
        len(data.get("TLT", [])),
        len(data.get("HYG", [])),
        len(data.get("LQD", [])),
    )
    return data


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Compute Tier 1 features for each trade entry date
# ═══════════════════════════════════════════════════════════════════════════

def _days_to_next_opex(entry_date_str: str) -> int:
    """Return integer days from entry_date until the next 3rd-Friday OPEX."""
    today = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
    for delta_months in range(3):
        m = today.month + delta_months
        y = today.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        monthly = _cal.monthcalendar(y, m)
        fridays = [week[_cal.FRIDAY] for week in monthly if week[_cal.FRIDAY] != 0]
        if len(fridays) < 3:
            continue
        opex_day = _date(y, m, fridays[2])
        if opex_day > today:
            return (opex_day - today).days
    return 0


def _lookup_at_date(series: pd.Series, date_str: str, n_prior: int = 1) -> Optional[float]:
    """Get the value n_prior trading days before or on date_str."""
    ts = pd.Timestamp(date_str)
    prior = series.loc[series.index <= ts]
    if len(prior) < n_prior:
        return None
    return float(prior.iloc[-n_prior])


def compute_tier1_features(df: pd.DataFrame, mkt: Dict[str, pd.Series]) -> pd.DataFrame:
    """Add Tier 1 features to the training DataFrame.

    Computes features at each trade's entry_date using historical market data.
    Missing data → NaN (handled by pipeline imputation downstream).

    Args:
        df:  Raw training DataFrame from training_data_combined.csv
        mkt: Dict of price Series from download_tier1_data()

    Returns:
        df with 6 new columns appended.
    """
    out = df.copy()

    vix3m_s = mkt.get("^VIX3M", pd.Series(dtype=float))
    spy_s    = mkt.get("SPY",    pd.Series(dtype=float))
    spy_open = mkt.get("SPY_Open", pd.Series(dtype=float))
    tlt_s    = mkt.get("TLT",    pd.Series(dtype=float))
    hyg_s    = mkt.get("HYG",    pd.Series(dtype=float))
    lqd_s    = mkt.get("LQD",    pd.Series(dtype=float))

    # Pre-compute SPY / TLT daily return Series for rolling correlation
    spy_rets = spy_s.pct_change() if len(spy_s) > 1 else pd.Series(dtype=float)
    tlt_rets = tlt_s.pct_change() if len(tlt_s) > 1 else pd.Series(dtype=float)

    n = len(out)
    vix_contango_ratios    = np.full(n, np.nan)
    spy_tlt_corrs          = np.full(n, np.nan)
    hyg_lqd_ratios         = np.full(n, np.nan)
    hyg_lqd_ratio_5d_chgs  = np.full(n, np.nan)
    days_to_opex_arr       = np.zeros(n, dtype=float)
    opening_gap_pcts       = np.full(n, np.nan)

    for i, row in enumerate(out.itertuples(index=False)):
        d = row.entry_date  # string "YYYY-MM-DD"
        ts = pd.Timestamp(d)

        # ── 1. vix_contango_ratio = VIX3M / VIX ───────────────────────
        vix3m_val = _lookup_at_date(vix3m_s, d)
        vix_val   = row.vix  # already in the CSV
        if vix3m_val is not None and vix_val and vix_val > 0:
            vix_contango_ratios[i] = vix3m_val / vix_val
        elif vix_val and vix_val > 0 and hasattr(row, "realized_vol_20d") and row.realized_vol_20d:
            # Fallback: VIX / realized_vol (variance risk premium proxy)
            rv = float(row.realized_vol_20d) if row.realized_vol_20d else None
            if rv and rv > 0:
                vix_contango_ratios[i] = vix_val / rv

        # ── 2. spy_tlt_corr_20d ────────────────────────────────────────
        spy_window = spy_rets.loc[spy_rets.index <= ts].tail(20)
        tlt_window = tlt_rets.loc[tlt_rets.index <= ts].tail(20)
        # Align on common index
        aligned = pd.concat(
            [spy_window.rename("spy"), tlt_window.rename("tlt")], axis=1
        ).dropna()
        if len(aligned) >= 10:
            corr = aligned["spy"].corr(aligned["tlt"])
            if not np.isnan(corr):
                spy_tlt_corrs[i] = corr

        # ── 3. hyg_lqd_ratio and 5d change ────────────────────────────
        hyg_now = _lookup_at_date(hyg_s, d)
        lqd_now = _lookup_at_date(lqd_s, d)
        if hyg_now is not None and lqd_now is not None and lqd_now > 0:
            ratio_now = hyg_now / lqd_now
            hyg_lqd_ratios[i] = ratio_now
            # 5-day prior: offset=6 (5 trading days back)
            hyg_5d = _lookup_at_date(hyg_s, d, n_prior=6)
            lqd_5d = _lookup_at_date(lqd_s, d, n_prior=6)
            if hyg_5d is not None and lqd_5d is not None and lqd_5d > 0:
                ratio_5d_ago = hyg_5d / lqd_5d
                if ratio_5d_ago > 0:
                    hyg_lqd_ratio_5d_chgs[i] = (ratio_now / ratio_5d_ago - 1) * 100

        # ── 4. days_to_opex ────────────────────────────────────────────
        days_to_opex_arr[i] = float(_days_to_next_opex(d))

        # ── 5. opening_gap_pct ─────────────────────────────────────────
        # today's SPY open vs previous close
        spy_close_arr = spy_s.loc[spy_s.index <= ts]
        spy_open_arr  = spy_open.loc[spy_open.index <= ts]
        if len(spy_close_arr) >= 2 and len(spy_open_arr) >= 1:
            prev_close = float(spy_close_arr.iloc[-2])
            today_open = float(spy_open_arr.iloc[-1])
            if prev_close > 0:
                opening_gap_pcts[i] = (today_open - prev_close) / prev_close * 100

    out["vix_contango_ratio"]    = vix_contango_ratios
    out["spy_tlt_corr_20d"]      = spy_tlt_corrs
    out["hyg_lqd_ratio"]         = hyg_lqd_ratios
    out["hyg_lqd_ratio_5d_chg"]  = hyg_lqd_ratio_5d_chgs
    out["days_to_opex"]          = days_to_opex_arr
    out["opening_gap_pct"]       = opening_gap_pcts

    logger.info(
        "Tier 1 fill rates: vix_contango=%.1f%% spy_tlt_corr=%.1f%% "
        "hyg_lqd=%.1f%% hyg_lqd_5d=%.1f%% days_to_opex=100.0%% gap=%.1f%%",
        (~np.isnan(vix_contango_ratios)).mean() * 100,
        (~np.isnan(spy_tlt_corrs)).mean() * 100,
        (~np.isnan(hyg_lqd_ratios)).mean() * 100,
        (~np.isnan(hyg_lqd_ratio_5d_chgs)).mean() * 100,
        (~np.isnan(opening_gap_pcts)).mean() * 100,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Feature pipeline with Tier 1 additions
# ═══════════════════════════════════════════════════════════════════════════

# Tier 1 feature normalization choices:
#   vix_contango_ratio  → z-score (level drifts; ratio can be non-stationary over regimes)
#   spy_tlt_corr_20d    → pass-through (already bounded [-1, 1])
#   hyg_lqd_ratio       → z-score (level non-stationary; e.g. different regimes)
#   hyg_lqd_ratio_5d_chg → pass-through (already a %)
#   days_to_opex        → pass-through (bounded 0-21)
#   opening_gap_pct     → pass-through (already a %)

TIER1_NUMERIC_FEATURES = [
    "vix_contango_ratio_zscore",   # z-scored version
    "spy_tlt_corr_20d",
    "hyg_lqd_ratio_zscore",        # z-scored version
    "hyg_lqd_ratio_5d_chg",
    "days_to_opex",
    "opening_gap_pct",
]

TIER1_IMPUTATION_DEFAULTS = {
    "vix_contango_ratio_zscore": 0.0,    # neutral (z=0)
    "spy_tlt_corr_20d":          0.0,    # zero correlation = neutral
    "hyg_lqd_ratio_zscore":      0.0,    # neutral
    "hyg_lqd_ratio_5d_chg":      0.0,    # no change
    "days_to_opex":              10.0,   # mid-cycle
    "opening_gap_pct":           0.0,    # no gap
}


class FeaturePipelineTier1(FeaturePipeline):
    """Extends FeaturePipeline with Tier 1 features.

    Applies z-scoring to the non-stationary Tier 1 level features
    (vix_contango_ratio, hyg_lqd_ratio) and passes through bounded features
    (spy_tlt_corr_20d, hyg_lqd_ratio_5d_chg, days_to_opex, opening_gap_pct).
    """

    @staticmethod
    def default_numeric_features_tier1() -> List[str]:
        """Original features + Tier 1 features."""
        return FeaturePipeline.default_numeric_features() + TIER1_NUMERIC_FEATURES

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Base pipeline transform + Tier 1 feature additions."""
        # First apply base transforms (z-score VIX/SPY, log contracts, etc.)
        out_raw = df.copy()

        # Z-score the level-based Tier 1 features before calling super
        if "vix_contango_ratio" in out_raw.columns:
            s = out_raw["vix_contango_ratio"].astype(float)
            out_raw["vix_contango_ratio_zscore"] = _zscore_column(s)
        else:
            out_raw["vix_contango_ratio_zscore"] = 0.0

        if "hyg_lqd_ratio" in out_raw.columns:
            s = out_raw["hyg_lqd_ratio"].astype(float)
            out_raw["hyg_lqd_ratio_zscore"] = _zscore_column(s)
        else:
            out_raw["hyg_lqd_ratio_zscore"] = 0.0

        # Pass-through features: just copy into expected names
        for col in ["spy_tlt_corr_20d", "hyg_lqd_ratio_5d_chg", "days_to_opex", "opening_gap_pct"]:
            if col not in out_raw.columns:
                out_raw[col] = 0.0

        # Temporarily override numeric_features to include Tier 1 features
        original_nf = self.numeric_features
        self.numeric_features = self.default_numeric_features_tier1()
        result = super().transform(out_raw)
        self.numeric_features = original_nf

        # Apply Tier 1-specific imputation defaults (base imputation already ran)
        for col, default in TIER1_IMPUTATION_DEFAULTS.items():
            if col in result.columns:
                result[col] = result[col].fillna(default)

        return result


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Walk-forward validation
# ═══════════════════════════════════════════════════════════════════════════

def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_proba))


def _metrics(y_true, y_pred, y_proba, ret=None) -> Dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {"n": 0}
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    auc  = _safe_auc(y_true, y_proba)
    base_wr = float(y_true.mean())
    pred_win_mask = y_pred == 1
    ml_wr = float(y_true[pred_win_mask].mean()) if pred_win_mask.sum() > 0 else None
    r = {
        "n": n,
        "base_win_rate": round(base_wr, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "ml_win_rate": round(ml_wr, 4) if ml_wr is not None else None,
        "win_rate_lift": round(ml_wr - base_wr, 4) if ml_wr is not None else None,
    }
    if ret is not None:
        r["mean_return_all"] = round(float(np.mean(ret)), 4)
        r["mean_return_ml_filtered"] = (
            round(float(np.mean(ret[pred_win_mask])), 4)
            if pred_win_mask.sum() > 0 else None
        )
    return r


def run_walk_forward(df: pd.DataFrame, pipeline: FeaturePipeline, label: str) -> pd.DataFrame:
    """Expanding-window walk-forward, returning per-trade OOS records."""
    logger.info("Walk-forward [%s] — building feature matrix...", label)
    features_full = pipeline.transform(df)
    feature_cols  = list(features_full.columns)
    df = df.reset_index(drop=True)
    features_full = features_full.reset_index(drop=True)

    years = sorted(df["year"].unique())
    logger.info("  Years: %s, feature_dim=%d", years, len(feature_cols))

    records: List[Dict] = []

    for fold_idx, test_year in enumerate(years[1:], start=1):
        train_mask = df["year"] < test_year
        test_mask  = df["year"] == test_year
        n_train = int(train_mask.sum())
        n_test  = int(test_mask.sum())

        if n_train < MIN_TRAIN_SAMPLES:
            logger.info("  Fold %d (test=%d): skipped (only %d train)", fold_idx, test_year, n_train)
            continue

        logger.info("  Fold %d (test=%d): n_train=%d n_test=%d", fold_idx, test_year, n_train, n_test)

        X_train = features_full.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "win"].values.astype(int)
        X_test  = features_full.loc[test_mask, feature_cols].values
        y_test  = df.loc[test_mask, "win"].values.astype(int)

        feat_train_df = features_full.loc[train_mask, feature_cols]
        feat_test_df  = features_full.loc[test_mask, feature_cols]
        test_meta     = df.loc[test_mask, ["entry_date", "year", "regime",
                                           "strategy_type", "return_pct"]].copy()

        # XGBoost
        xgb_model = xgb.XGBClassifier(**_XGB_PARAMS)
        if n_train > 80:
            from sklearn.model_selection import train_test_split as tts
            Xtr, Xval, ytr, yval = tts(X_train, y_train, test_size=0.15, random_state=42)
            xgb_model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        else:
            xgb_model.fit(X_train, y_train)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred  = (xgb_proba > 0.5).astype(int)

        # EnsembleSignalModel
        with tempfile.TemporaryDirectory() as tmpdir:
            ens = EnsembleSignalModel(model_dir=tmpdir)
            ens.train(feat_train_df, y_train, save_model=False)
            ens_proba = ens.predict_batch(feat_test_df)
        ens_pred = (ens_proba > 0.5).astype(int)

        for i, idx in enumerate(test_meta.index):
            row = test_meta.loc[idx]
            base = {
                "fold": fold_idx,
                "feature_set": label,
                "entry_date": str(row["entry_date"]),
                "year": int(row["year"]),
                "regime": str(row.get("regime", "unknown")),
                "strategy_type": str(row.get("strategy_type", "CS")),
                "return_pct": float(row["return_pct"]) if pd.notna(row.get("return_pct")) else 0.0,
                "true_label": int(y_test[i]),
            }
            records.append({**base, "model": "xgboost",
                            "prediction": int(xgb_pred[i]),
                            "probability": float(xgb_proba[i])})
            records.append({**base, "model": "ensemble",
                            "prediction": int(ens_pred[i]),
                            "probability": float(ens_proba[i])})

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Aggregate results
# ═══════════════════════════════════════════════════════════════════════════

def build_results(orig_recs: pd.DataFrame, tier1_recs: pd.DataFrame) -> Dict[str, Any]:
    """Compute metrics for both feature sets and both models."""
    results: Dict[str, Any] = {}

    for label, recs in [("original", orig_recs), ("tier1", tier1_recs)]:
        results[label] = {}
        for model, mrecs in recs.groupby("model"):
            y_true  = mrecs["true_label"].values
            y_pred  = mrecs["prediction"].values
            y_proba = mrecs["probability"].values
            ret     = mrecs["return_pct"].values
            results[label][model] = {
                "overall": _metrics(y_true, y_pred, y_proba, ret),
                "by_year": {},
                "by_regime": {},
            }
            for year, grp in mrecs.groupby("year"):
                results[label][model]["by_year"][str(year)] = _metrics(
                    grp["true_label"].values, grp["prediction"].values,
                    grp["probability"].values, grp["return_pct"].values,
                )
            for regime, grp in mrecs.groupby("regime"):
                if len(grp) >= 5:
                    results[label][model]["by_regime"][str(regime)] = _metrics(
                        grp["true_label"].values, grp["prediction"].values,
                        grp["probability"].values, grp["return_pct"].values,
                    )

    # Compute deltas (tier1 - original) for each model
    results["delta"] = {}
    for model in ["xgboost", "ensemble"]:
        orig_auc  = results["original"][model]["overall"].get("auc")
        t1_auc    = results["tier1"][model]["overall"].get("auc")
        orig_acc  = results["original"][model]["overall"].get("accuracy")
        t1_acc    = results["tier1"][model]["overall"].get("accuracy")
        orig_lift = results["original"][model]["overall"].get("win_rate_lift")
        t1_lift   = results["tier1"][model]["overall"].get("win_rate_lift")
        results["delta"][model] = {
            "auc_delta":       round(t1_auc  - orig_auc,  4) if (t1_auc  and orig_auc)  else None,
            "acc_delta":       round(t1_acc  - orig_acc,  4) if (t1_acc  and orig_acc)  else None,
            "lift_delta":      round(t1_lift - orig_lift, 4) if (t1_lift and orig_lift) else None,
            "orig_auc":        orig_auc,
            "tier1_auc":       t1_auc,
            "orig_accuracy":   orig_acc,
            "tier1_accuracy":  t1_acc,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Report generation
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v, fmt=".4f"):
    if v is None:
        return "—"
    return f"{v:{fmt}}"

def _delta_marker(d):
    if d is None:
        return "—"
    sign = "▲" if d > 0 else ("▼" if d < 0 else "–")
    return f"{sign} {abs(d):.4f}"


def build_report(results: Dict[str, Any], tier1_df: pd.DataFrame) -> str:
    lines = [
        "# Tier 1 Feature Benchmark: Original vs Original + Tier 1",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Training set:** {len(tier1_df)} trades, "
        f"{tier1_df['entry_date'].min()} – {tier1_df['entry_date'].max()}",
        f"**Feature sets compared:** original ({len(FeaturePipeline.default_numeric_features())} features) "
        f"vs tier1 ({len(FeaturePipeline.default_numeric_features()) + len(TIER1_NUMERIC_FEATURES)} features)",
        "",
        "---",
        "",
        "## 1. Dataset Augmentation",
        "",
    ]

    # Fill rate table
    tier1_cols = ["vix_contango_ratio", "spy_tlt_corr_20d", "hyg_lqd_ratio",
                  "hyg_lqd_ratio_5d_chg", "days_to_opex", "opening_gap_pct"]
    lines.append("| Feature | Non-null | Fill % | Min | Median | Max |")
    lines.append("|---------|---------|--------|-----|--------|-----|")
    for col in tier1_cols:
        if col in tier1_df.columns:
            s = tier1_df[col].dropna()
            pct = len(s) / len(tier1_df) * 100
            lines.append(
                f"| `{col}` | {len(s)} | {pct:.1f}% | "
                f"{_fmt(s.min() if len(s) else None, '.3f')} | "
                f"{_fmt(s.median() if len(s) else None, '.3f')} | "
                f"{_fmt(s.max() if len(s) else None, '.3f')} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 2. Overall Walk-Forward Results",
        "",
        "| Model | Feature Set | N OOS | AUC | Accuracy | Base WR | ML WR | Lift |",
        "|-------|-------------|-------|-----|----------|---------|-------|------|",
    ]

    for label in ["original", "tier1"]:
        for model in ["xgboost", "ensemble"]:
            m = results[label][model]["overall"]
            lines.append(
                f"| {model} | {label} | {m.get('n', 0)} "
                f"| {_fmt(m.get('auc'))} "
                f"| {_fmt(m.get('accuracy'))} "
                f"| {_fmt(m.get('base_win_rate'))} "
                f"| {_fmt(m.get('ml_win_rate'))} "
                f"| {_fmt(m.get('win_rate_lift'))} |"
            )

    lines += [
        "",
        "## 3. Tier 1 Delta vs Original",
        "",
        "| Model | ΔAUC | ΔAccuracy | ΔWR Lift | Original AUC | Tier1 AUC |",
        "|-------|------|-----------|----------|-------------|-----------|",
    ]
    for model in ["xgboost", "ensemble"]:
        d = results["delta"][model]
        lines.append(
            f"| {model} | {_delta_marker(d.get('auc_delta'))} "
            f"| {_delta_marker(d.get('acc_delta'))} "
            f"| {_delta_marker(d.get('lift_delta'))} "
            f"| {_fmt(d.get('orig_auc'))} "
            f"| {_fmt(d.get('tier1_auc'))} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Per-Year Breakdown",
        "",
        "| Year | Model | Feat Set | N | AUC | Accuracy | Lift |",
        "|------|-------|----------|---|-----|----------|------|",
    ]
    all_years = sorted(set(
        list(results["original"]["ensemble"]["by_year"].keys()) +
        list(results["tier1"]["ensemble"]["by_year"].keys())
    ))
    for year in all_years:
        for model in ["xgboost", "ensemble"]:
            for label in ["original", "tier1"]:
                m = results[label][model]["by_year"].get(str(year), {})
                if m.get("n", 0) == 0:
                    continue
                lines.append(
                    f"| {year} | {model} | {label} | {m.get('n', 0)} "
                    f"| {_fmt(m.get('auc'))} "
                    f"| {_fmt(m.get('accuracy'))} "
                    f"| {_fmt(m.get('win_rate_lift'))} |"
                )

    lines += [
        "",
        "---",
        "",
        "## 5. Per-Regime Breakdown",
        "",
        "| Regime | Model | Feat Set | N | AUC | Accuracy | Lift |",
        "|--------|-------|----------|---|-----|----------|------|",
    ]
    all_regimes = sorted(set(
        list(results["original"]["ensemble"]["by_regime"].keys()) +
        list(results["tier1"]["ensemble"]["by_regime"].keys())
    ))
    for regime in all_regimes:
        for model in ["xgboost", "ensemble"]:
            for label in ["original", "tier1"]:
                m = results[label][model]["by_regime"].get(str(regime), {})
                if m.get("n", 0) == 0:
                    continue
                lines.append(
                    f"| {regime} | {model} | {label} | {m.get('n', 0)} "
                    f"| {_fmt(m.get('auc'))} "
                    f"| {_fmt(m.get('accuracy'))} "
                    f"| {_fmt(m.get('win_rate_lift'))} |"
                )

    lines += ["", "---", "", "## 6. Interpretation", ""]

    # Auto-interpret the deltas
    xgb_auc_delta = results["delta"]["xgboost"].get("auc_delta")
    ens_auc_delta = results["delta"]["ensemble"].get("auc_delta")
    xgb_tier1_auc = results["tier1"]["xgboost"]["overall"].get("auc")
    ens_tier1_auc = results["tier1"]["ensemble"]["overall"].get("auc")
    xgb_orig_auc  = results["original"]["xgboost"]["overall"].get("auc")
    ens_orig_auc  = results["original"]["ensemble"]["overall"].get("auc")

    def _verdict(delta, thresh_good=0.005, thresh_great=0.015):
        if delta is None:
            return "inconclusive"
        if delta >= thresh_great:
            return "**strong improvement**"
        elif delta >= thresh_good:
            return "**modest improvement**"
        elif delta > -thresh_good:
            return "**neutral** (within noise)"
        else:
            return "**degraded** (features may be noisy)"

    lines.append(f"- XGBoost: Tier 1 ΔAUC = {_fmt(xgb_auc_delta, '+.4f')} → {_verdict(xgb_auc_delta)}")
    lines.append(f"- Ensemble: Tier 1 ΔAUC = {_fmt(ens_auc_delta, '+.4f')} → {_verdict(ens_auc_delta)}")
    lines.append("")

    # Best model + feature set
    combos = [
        ("xgboost",  "original", xgb_orig_auc),
        ("xgboost",  "tier1",    xgb_tier1_auc),
        ("ensemble", "original", ens_orig_auc),
        ("ensemble", "tier1",    ens_tier1_auc),
    ]
    combos = [(m, l, a) for m, l, a in combos if a is not None]
    if combos:
        best = max(combos, key=lambda x: x[2])
        lines.append(f"**Best combination: {best[0]} + {best[1]} features (AUC = {best[2]:.4f})**")
        lines.append("")

    lines += [
        "### What each Tier 1 feature adds",
        "",
        "| Feature | Signal | Prior expected impact |",
        "|---------|--------|----------------------|",
        "| `vix_contango_ratio` | VIX term structure (VRP proxy) | High — directly prices the option-selling edge |",
        "| `spy_tlt_corr_20d` | Equity-bond correlation regime | Medium — identifies 2022-style joint-selloff regime |",
        "| `hyg_lqd_ratio` | Credit stress level | Medium-High — leads equity vol by 2-5 days |",
        "| `hyg_lqd_ratio_5d_chg` | Credit stress direction | Medium — rate of change more actionable than level |",
        "| `days_to_opex` | Gamma proximity to OPEX | Medium — replaces binary is_opex_week |",
        "| `opening_gap_pct` | Overnight sentiment | Low-Medium — daily signal, high noise |",
        "",
        "---",
        "",
        f"*Generated by compass/benchmark_tier1_features.py — {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("TIER 1 FEATURE BENCHMARK")
    print("=" * 70)

    # Step 1: Load base training data
    logger.info("Loading base training data from %s", COMBINED_PATH)
    df_base = pd.read_csv(COMBINED_PATH)
    df_base["entry_date"] = df_base["entry_date"].astype(str)
    logger.info("Base dataset: %d trades, years %s", len(df_base),
                sorted(df_base["year"].unique()))

    # Step 2: Download Tier 1 market data
    date_min = df_base["entry_date"].min()
    # Go back further for warmup (rolling correlations need history)
    warmup_start = str(int(date_min[:4]) - 2) + date_min[4:]
    mkt_data = download_tier1_data(start=warmup_start, end="2026-01-01")

    # Step 3: Augment with Tier 1 features
    logger.info("Computing Tier 1 features for %d trades...", len(df_base))
    df_tier1 = compute_tier1_features(df_base, mkt_data)
    df_tier1.to_csv(TIER1_PATH, index=False)
    logger.info("Saved augmented dataset to %s", TIER1_PATH)

    # Step 4: Build feature matrices and run walk-forward
    print("\n--- Running walk-forward: ORIGINAL features ---")
    pipeline_orig  = FeaturePipeline()
    orig_records   = run_walk_forward(df_base,   pipeline_orig,  "original")

    print("\n--- Running walk-forward: TIER 1 features ---")
    pipeline_tier1 = FeaturePipelineTier1()
    tier1_records  = run_walk_forward(df_tier1, pipeline_tier1, "tier1")

    # Step 5: Aggregate
    print("\n--- Aggregating results ---")
    results = build_results(orig_records, tier1_records)

    # Save JSON
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results JSON to %s", JSON_PATH)

    # Step 6: Generate report
    report = build_report(results, df_tier1)
    with open(MD_PATH, "w") as f:
        f.write(report)
    logger.info("Saved report to %s", MD_PATH)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for model in ["xgboost", "ensemble"]:
        d = results["delta"][model]
        print(f"  {model:12s}: "
              f"orig_AUC={d.get('orig_auc', 0):.4f}  "
              f"tier1_AUC={d.get('tier1_auc', 0):.4f}  "
              f"Δ={d.get('auc_delta', 0):+.4f}")
    print(f"\nOutputs:")
    print(f"  {TIER1_PATH}")
    print(f"  {JSON_PATH}")
    print(f"  {MD_PATH}")


if __name__ == "__main__":
    main()
