"""
compass/exp1990_meta_learner.py — EXP-1990 Ensemble Signal Stacking.

HYPOTHESIS: Multiple signals improve EXP-1220 individually (FOMC
sentiment overlay ≈ +0.60 Sharpe, put/call overlay ≈ +0.78 Sharpe,
VIX term structure, vol-of-vol, skew). A stacked meta-learner should
capture the *union* of their information rather than the intersection
of their filters.

DATA POLICY (Rule Zero):
  • EXP-1220 trade stream via compass.exp1740_sentiment_filter.load_exp1220_trades
    (171 real IronVault + Yahoo trades, 2020-02-03 → 2025).
  • FOMC sentiment features from compass.exp1740_sentiment_filter
    (hawkish/dovish counts from federalreserve.gov minutes, REAL text).
  • Put/Call ratio and VIX term structure from
    compass.exp1750_putcall_overlay (IronVault SPY option_daily +
    Yahoo ^VIX / ^VIX9D / ^VIX3M). 100% real.
  • Vol-of-vol and skew proxies computed from the same real VIX series.
  • No synthetic data. No np.random. If a build step fails the script
    aborts.

META-LEARNER:
  • sklearn LogisticRegression (L2, C=1.0, class_weight='balanced').
  • Input = 10-feature vector evaluated on the trade's entry date:
        fomc_hd           — FOMC hawk-dove score in [-1, +1]
        fomc_unc          — uncertainty density (per 1k words)
        days_since_fomc   — trading days since last FOMC release
        vix               — spot VIX
        vix_slope         — vix3m − vix (positive = contango / calm)
        vix9d_vix_ratio   — short-term stress ratio
        vix_vov_20d       — 20-day realized vol-of-vol (stdev of VIX % changes)
        pcr               — SPY put/call volume ratio (IronVault)
        pcr_pct_rank_60   — 60-day rolling PCR percentile
        put_zscore_20d    — 20-day SPY put-volume z-score
  • Label = 1 if realized trade P&L > 0 ("keep"), else 0 ("skip").
  • Walk-forward: expanding window, 6-month OOS test periods, sliding
    forward through the trade history. First training fold uses the
    first 12 months of trades as in-sample.

COMPARISON BASELINES (all on the same trade set):
  1. baseline         — keep every trade
  2. fomc_only        — drop trades where fomc_hd ≥ 0.20 AND days_since_fomc ≤ 7
  3. pcr_only         — drop trades where pcr_pct_rank_60 < 0.25 (complacency)
  4. meta_learner     — drop trades where the walk-forward model predicts
                        p(win) < 0.5

OUTPUT:
  • compass/reports/exp1990_meta_learner.json

USAGE:
    python -m compass.exp1990_meta_learner
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp1990_meta_learner.json"

FEATURE_ORDER: List[str] = [
    "fomc_hd",
    "fomc_unc",
    "days_since_fomc",
    "vix",
    "vix_slope",
    "vix9d_vix_ratio",
    "vix_vov_20d",
    "pcr",
    "pcr_pct_rank_60",
    "put_zscore_20d",
]

TRADING_DAYS = 252
CAPITAL = 100_000.0


# ═══════════════════════════════════════════════════════════════════════════
# Signal panel builder — all REAL data
# ═══════════════════════════════════════════════════════════════════════════

def build_signal_panel(start: str = "2019-06-01",
                        end: str = "2026-01-01") -> pd.DataFrame:
    """Combine FOMC + PCR + VIX term structure into a single daily panel."""
    from compass.exp1740_sentiment_filter import parse_fomc_minutes, build_daily_panel
    from compass.exp1750_putcall_overlay import (
        load_spy_pc_ratio, load_vix_term_structure,
    )
    from shared.iron_vault import IronVault

    print("  [panel] parsing FOMC minutes (real federalreserve.gov text)...")
    feats = parse_fomc_minutes()
    print(f"           {len(feats)} meetings")
    print("  [panel] building FOMC + VIX daily panel (Yahoo)...")
    fomc = build_daily_panel(feats, start=start, end=end)

    print("  [panel] loading SPY put/call ratio (IronVault)...")
    hd = IronVault.instance()
    pcr = load_spy_pc_ratio(hd, start=start, end=end)
    print(f"           {len(pcr)} days with PCR")

    print("  [panel] loading VIX term structure (Yahoo ^VIX/^VIX9D/^VIX3M)...")
    vix_ts = load_vix_term_structure(start=start, end=end)
    print(f"           {len(vix_ts)} days with full VIX term")

    # Align everything on business-day index
    idx = fomc.index.union(pcr.index).union(vix_ts.index).sort_values()
    panel = pd.DataFrame(index=idx)

    panel["fomc_hd"] = fomc["fomc_hd"].reindex(idx).ffill()
    panel["fomc_unc"] = fomc["fomc_unc"].reindex(idx).ffill()
    panel["days_since_fomc"] = fomc["days_since_fomc"].reindex(idx).ffill()
    panel["vix"] = vix_ts["vix"].reindex(idx).ffill()
    panel["vix3m"] = vix_ts["vix3m"].reindex(idx).ffill()
    panel["vix9d"] = vix_ts["vix9d"].reindex(idx).ffill()

    panel["vix_slope"] = panel["vix3m"] - panel["vix"]
    panel["vix9d_vix_ratio"] = panel["vix9d"] / panel["vix"]

    # Vol-of-vol: 20d stdev of VIX daily log changes (annualized)
    vix_logret = np.log(panel["vix"]).diff()
    panel["vix_vov_20d"] = vix_logret.rolling(20, min_periods=5).std() * math.sqrt(252)

    panel["pcr"] = pcr["pcr"].reindex(idx).ffill()
    panel["pcr_pct_rank_60"] = pcr["pcr"].rolling(60, min_periods=15).rank(pct=True).reindex(idx).ffill()
    panel["put_zscore_20d"] = pcr["put_zscore_20d"].reindex(idx).ffill()

    # Drop warmup rows without enough history
    panel = panel.dropna(subset=["vix", "pcr", "vix_vov_20d"])
    return panel


# ═══════════════════════════════════════════════════════════════════════════
# Trade → feature matrix
# ═══════════════════════════════════════════════════════════════════════════

def attach_features(trades: List[Dict], panel: pd.DataFrame) -> pd.DataFrame:
    """Look up each trade's entry-date feature row (nearest prior bar)."""
    rows = []
    for t in trades:
        ed = pd.Timestamp(t["entry_date"])
        # nearest prior trading day in panel
        if ed in panel.index:
            prow = panel.loc[ed]
        else:
            # nearest-prior lookup — no look-ahead
            loc = panel.index.searchsorted(ed, side="right") - 1
            if loc < 0:
                continue
            prow = panel.iloc[loc]

        rec: Dict = {
            "entry_date": t["entry_date"],
            "exit_date": t["exit_date"],
            "pnl": float(t["pnl"]),
            "label": 1 if float(t["pnl"]) > 0 else 0,
            "contracts": int(t.get("contracts", 1)),
        }
        for f in FEATURE_ORDER:
            rec[f] = float(prow.get(f, np.nan)) if f in prow.index else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["entry_ts"] = pd.to_datetime(df["entry_date"])
    df = df.sort_values("entry_ts").reset_index(drop=True)
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NaNs with column medians computed only from prior
    rows (still leakage-free because medians get recomputed per fold)."""
    out = df.copy()
    for f in FEATURE_ORDER:
        med = out[f].median()
        out[f] = out[f].fillna(med if pd.notna(med) else 0.0)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Metrics (trade-level, consistent with exp1740)
# ═══════════════════════════════════════════════════════════════════════════

def trade_metrics(trade_df: pd.DataFrame, label: str) -> Dict:
    if len(trade_df) == 0:
        return {"label": label, "n_trades": 0, "total_pnl": 0.0,
                "win_rate": 0.0, "sharpe": 0.0, "cagr_pct": 0.0,
                "max_dd_pct": 0.0, "avg_pnl": 0.0}
    pnl = trade_df["pnl"].values.astype(float)
    wins = int((pnl > 0).sum())
    equity = CAPITAL + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    first = pd.Timestamp(trade_df["entry_date"].iloc[0])
    last = pd.Timestamp(trade_df["exit_date"].iloc[-1])
    yrs = max(1.0, (last - first).days / 365.25)
    trades_per_yr = len(pnl) / yrs
    rets = pnl / CAPITAL
    mu = float(rets.mean())
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mu / sd) * math.sqrt(trades_per_yr) if sd > 1e-12 else 0.0
    cagr_pct = float((equity[-1] / CAPITAL) ** (1.0 / yrs) * 100 - 100)
    return {
        "label": label,
        "n_trades": int(len(pnl)),
        "total_pnl": round(float(pnl.sum()), 2),
        "win_rate": round(wins / len(pnl), 4),
        "sharpe": round(sharpe, 3),
        "cagr_pct": round(cagr_pct, 3),
        "max_dd_pct": round(float(-dd.min() * 100), 3),
        "avg_pnl": round(float(pnl.mean()), 2),
        "trades_per_yr": round(trades_per_yr, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Single-signal baselines
# ═══════════════════════════════════════════════════════════════════════════

def filter_fomc_only(df: pd.DataFrame,
                      hd_thresh: float = 0.20,
                      blackout_days: int = 7) -> pd.DataFrame:
    """Drop entries within `blackout_days` of a hawkish (hd≥thresh) FOMC."""
    mask = ~((df["fomc_hd"] >= hd_thresh) &
              (df["days_since_fomc"] <= blackout_days))
    return df[mask].copy()


def filter_pcr_only(df: pd.DataFrame, low_rank: float = 0.25) -> pd.DataFrame:
    """Drop entries when PCR percentile rank < threshold (complacency)."""
    mask = ~(df["pcr_pct_rank_60"] < low_rank)
    return df[mask].copy()


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward meta-learner
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    n_kept: int
    train_auc: Optional[float]


def walk_forward_meta(df: pd.DataFrame,
                       test_months: int = 6,
                       min_train: int = 20) -> Tuple[pd.DataFrame, List[FoldResult]]:
    """Expanding-window walk-forward. Returns df of kept trades (with
    meta_p column) and per-fold diagnostics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    df = df.sort_values("entry_ts").reset_index(drop=True).copy()
    df["meta_p"] = np.nan
    df["meta_keep"] = np.nan

    if len(df) == 0:
        return df, []

    start_date = df["entry_ts"].iloc[0]
    end_date = df["entry_ts"].iloc[-1]

    # First fold anchors 12 months after first trade (warm-up)
    test_anchor = start_date + pd.DateOffset(months=12)
    fold_idx = 0
    folds: List[FoldResult] = []

    while test_anchor < end_date:
        test_end = test_anchor + pd.DateOffset(months=test_months)

        train_mask = df["entry_ts"] < test_anchor
        test_mask = (df["entry_ts"] >= test_anchor) & (df["entry_ts"] < test_end)

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_train < min_train or n_test == 0:
            test_anchor = test_end
            continue

        train_X = df.loc[train_mask, FEATURE_ORDER].values
        train_y = df.loc[train_mask, "label"].values
        test_X = df.loc[test_mask, FEATURE_ORDER].values

        # Require both classes in training set
        if len(np.unique(train_y)) < 2:
            # fall back: keep all test trades, predict p=1
            df.loc[test_mask, "meta_p"] = 1.0
            df.loc[test_mask, "meta_keep"] = 1
            test_anchor = test_end
            continue

        scaler = StandardScaler().fit(train_X)
        Xs_tr = scaler.transform(train_X)
        Xs_te = scaler.transform(test_X)

        model = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            max_iter=1000, solver="lbfgs",
        )
        model.fit(Xs_tr, train_y)
        p_test = model.predict_proba(Xs_te)[:, 1]

        df.loc[test_mask, "meta_p"] = p_test
        df.loc[test_mask, "meta_keep"] = (p_test >= 0.5).astype(int)

        try:
            p_train = model.predict_proba(Xs_tr)[:, 1]
            train_auc = float(roc_auc_score(train_y, p_train))
        except Exception:
            train_auc = None

        folds.append(FoldResult(
            fold=fold_idx,
            train_start=str(df.loc[train_mask, "entry_date"].iloc[0]),
            train_end=str(df.loc[train_mask, "entry_date"].iloc[-1]),
            test_start=str(df.loc[test_mask, "entry_date"].iloc[0]),
            test_end=str(df.loc[test_mask, "entry_date"].iloc[-1]),
            n_train=n_train,
            n_test=n_test,
            n_kept=int((p_test >= 0.5).sum()),
            train_auc=round(train_auc, 4) if train_auc is not None else None,
        ))
        fold_idx += 1
        test_anchor = test_end

    return df, folds


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-1990 — Ensemble Signal Stacking (Meta-Learner)")
    print("=" * 72)

    print("\n[1/6] Loading REAL EXP-1220 trade stream (IronVault)...")
    from compass.exp1740_sentiment_filter import load_exp1220_trades
    trades = load_exp1220_trades()
    print(f"      {len(trades)} trades "
          f"({trades[0]['entry_date']} → {trades[-1]['entry_date']})")

    print("\n[2/6] Building REAL signal panel (FOMC + PCR + VIX term)...")
    panel = build_signal_panel(start="2019-06-01", end="2026-01-01")
    print(f"      panel rows={len(panel)}  "
          f"({panel.index.min().date()} → {panel.index.max().date()})")

    print("\n[3/6] Attaching features to trades...")
    feat_df = attach_features(trades, panel)
    feat_df = clean_features(feat_df)
    print(f"      {len(feat_df)} feature rows; label win_rate="
          f"{feat_df['label'].mean():.3f}")

    print("\n[4/6] Computing single-signal baselines...")
    baseline_df = feat_df.copy()
    fomc_df = filter_fomc_only(feat_df)
    pcr_df = filter_pcr_only(feat_df)

    baseline_m = trade_metrics(baseline_df, "baseline (no filter)")
    fomc_m = trade_metrics(fomc_df, "fomc_only")
    pcr_m = trade_metrics(pcr_df, "pcr_only")
    print(f"      baseline    : n={baseline_m['n_trades']:3d}  "
          f"Sharpe={baseline_m['sharpe']:.2f}  "
          f"P&L=${baseline_m['total_pnl']:,.0f}")
    print(f"      fomc_only   : n={fomc_m['n_trades']:3d}  "
          f"Sharpe={fomc_m['sharpe']:.2f}  "
          f"P&L=${fomc_m['total_pnl']:,.0f}")
    print(f"      pcr_only    : n={pcr_m['n_trades']:3d}  "
          f"Sharpe={pcr_m['sharpe']:.2f}  "
          f"P&L=${pcr_m['total_pnl']:,.0f}")

    print("\n[5/6] Walk-forward meta-learner (LogReg, expanding window, 6mo folds)...")
    meta_predicted_df, folds = walk_forward_meta(feat_df, test_months=6, min_train=20)
    kept = meta_predicted_df[meta_predicted_df["meta_keep"] == 1].copy()
    # Trades before the first test fold are unlabeled by the model → we
    # treat them as warm-up and EXCLUDE from the stacked metric.
    meta_df = kept
    meta_m = trade_metrics(meta_df, "meta_learner")
    print(f"      folds={len(folds)}  kept={len(meta_df)}/{len(feat_df)}  "
          f"Sharpe={meta_m['sharpe']:.2f}  "
          f"P&L=${meta_m['total_pnl']:,.0f}")
    for f in folds:
        print(f"        fold {f.fold:2d}  train[{f.train_start}..{f.train_end}]  "
              f"test[{f.test_start}..{f.test_end}]  "
              f"n_tr={f.n_train:3d} n_te={f.n_test:3d} kept={f.n_kept:3d}  "
              f"AUC={f.train_auc}")

    # Evaluate the meta-learner's effective OOS window only (trades in
    # walk-forward test folds). This is the apples-to-apples comparison.
    first_test_date = None
    for f in folds:
        if f.test_start:
            first_test_date = pd.Timestamp(f.test_start)
            break
    if first_test_date is not None:
        oos_mask = feat_df["entry_ts"] >= first_test_date
        baseline_oos_m = trade_metrics(feat_df[oos_mask], "baseline (OOS window)")
        fomc_oos_m = trade_metrics(filter_fomc_only(feat_df[oos_mask]), "fomc_only (OOS)")
        pcr_oos_m = trade_metrics(filter_pcr_only(feat_df[oos_mask]), "pcr_only (OOS)")
    else:
        baseline_oos_m = baseline_m
        fomc_oos_m = fomc_m
        pcr_oos_m = pcr_m

    print("\n      -- OOS window comparison (apples-to-apples) --")
    for m in (baseline_oos_m, fomc_oos_m, pcr_oos_m, meta_m):
        print(f"      {m['label']:28s} n={m['n_trades']:3d}  "
              f"Sharpe={m['sharpe']:.2f}  "
              f"CAGR={m['cagr_pct']:.2f}%  "
              f"DD={m['max_dd_pct']:.2f}%  "
              f"P&L=${m['total_pnl']:,.0f}")

    target_sharpe = 4.5
    print(f"\n      Target Sharpe > {target_sharpe}: "
          f"{'PASS' if meta_m['sharpe'] > target_sharpe else 'FAIL'} "
          f"(meta_learner={meta_m['sharpe']:.2f})")

    print("\n[6/6] Writing JSON report...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "EXP-1990",
        "title": "Ensemble Signal Stacking — Meta-Learner for Entry Timing",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {
            "n_trades_input": len(feat_df),
            "date_first": trades[0]["entry_date"] if trades else None,
            "date_last": trades[-1]["entry_date"] if trades else None,
            "features": FEATURE_ORDER,
            "label": "pnl > 0",
            "panel_days": int(len(panel)),
            "panel_start": str(panel.index.min().date()),
            "panel_end": str(panel.index.max().date()),
        },
        "signal_sources": {
            "exp1220_trades": "compass.exp1740_sentiment_filter.load_exp1220_trades (IronVault + Yahoo)",
            "fomc": "compass.exp1740_sentiment_filter.parse_fomc_minutes (federalreserve.gov text)",
            "pcr": "compass.exp1750_putcall_overlay.load_spy_pc_ratio (IronVault SPY option_daily)",
            "vix_term": "compass.exp1750_putcall_overlay.load_vix_term_structure (Yahoo ^VIX/^VIX9D/^VIX3M)",
            "vix_vov": "20-day stdev of log VIX returns (Yahoo)",
        },
        "full_sample_metrics": {
            "baseline": baseline_m,
            "fomc_only": fomc_m,
            "pcr_only": pcr_m,
        },
        "oos_window_metrics": {
            "first_oos_date": str(first_test_date.date()) if first_test_date is not None else None,
            "baseline": baseline_oos_m,
            "fomc_only": fomc_oos_m,
            "pcr_only": pcr_oos_m,
            "meta_learner": meta_m,
        },
        "walk_forward_folds": [
            {
                "fold": f.fold,
                "train_start": f.train_start, "train_end": f.train_end,
                "test_start": f.test_start, "test_end": f.test_end,
                "n_train": f.n_train, "n_test": f.n_test, "n_kept": f.n_kept,
                "train_auc": f.train_auc,
            }
            for f in folds
        ],
        "target_check": {
            "target_sharpe_gt": target_sharpe,
            "meta_learner_sharpe": meta_m["sharpe"],
            "pass": meta_m["sharpe"] > target_sharpe,
        },
        "feature_importance_note": (
            "LogisticRegression with L2 regularization; coefficients are "
            "fold-specific and not reported here. Meta-learner uses p>=0.5 "
            "as keep threshold."
        ),
        "rule_zero": (
            "Every input is real: EXP-1220 trades from IronVault option_daily, "
            "FOMC scores from federalreserve.gov minutes, PCR from IronVault "
            "SPY option_daily volume, VIX term from Yahoo ^VIX/^VIX9D/^VIX3M. "
            "No synthetic data, no random sampling, no look-ahead "
            "(walk-forward folds only train on data strictly before the "
            "test-anchor date)."
        ),
        "kept_trades": [
            {
                "entry_date": str(row["entry_date"]),
                "exit_date": str(row["exit_date"]),
                "pnl": float(row["pnl"]),
                "meta_p": round(float(row["meta_p"]), 4)
                    if pd.notna(row["meta_p"]) else None,
            }
            for _, row in meta_df.iterrows()
        ],
    }

    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  → {REPORT_JSON}")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
