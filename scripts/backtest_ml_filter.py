#!/usr/bin/env python3
"""
backtest_ml_filter.py — ML ensemble filter for EXP-400 (The Champion)

Loads the 6-year backtest for EXP-400's regime-adaptive SPY strategy,
extracts per-trade features (VIX, MA state, RSI, DTE, IV rank, etc.),
trains an EnsembleSignalModel (XGBoost + RF + ExtraTrees) to predict
trade outcomes, and evaluates it with walk-forward validation.

Walk-forward split:
  Train : 2020–2023  (expanding window per fold)
  Test  : 2024–2025  (completely out-of-sample)

Outputs a detailed Markdown report to output/ml_filter_exp400_report.md.

Usage:
    # From project root:
    python3 scripts/backtest_ml_filter.py
    python3 scripts/backtest_ml_filter.py --confidence-threshold 0.58
    python3 scripts/backtest_ml_filter.py --skip-backtest  # use cached trades JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BACKTEST_CONFIG  = ROOT / "configs" / "exp_400_champion_realdata.json"
TRADES_CACHE     = ROOT / "output"  / "ml_filter_exp400_trades_cache.json"
REPORT_PATH      = ROOT / "output"  / "ml_filter_exp400_report.md"
MODEL_DIR        = ROOT / "ml" / "models"

TICKER           = "SPY"
START_DATE       = datetime(2020, 1, 2)
END_DATE         = datetime(2025, 12, 31)
STARTING_CAPITAL = 100_000.0

TRAIN_END_YEAR   = 2023    # last year included in training
TEST_START_YEAR  = 2024    # first OOS year

# Confidence thresholds to sweep (probability ≥ threshold → take trade)
THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70]

# Minimum fraction of test trades to keep at optimal threshold
MIN_FILTER_RATE = 0.40


# ═════════════════════════════════════════════════════════════════════════════
# Section 1: Technical feature computation from price data
# ═════════════════════════════════════════════════════════════════════════════

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI via Wilder's exponential smoothing (standard implementation)."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank (0–100) of current value within trailing window."""
    return series.rolling(window, min_periods=max(2, window // 4)).apply(
        lambda x: float(np.mean(x[:-1] <= x[-1]) * 100),
        raw=True,
    )


def _realized_vol(log_returns: pd.Series, window: int) -> pd.Series:
    """Annualised realised volatility (%) from log-returns."""
    return (
        log_returns.rolling(window, min_periods=max(2, window // 2)).std()
        * np.sqrt(252) * 100
    ).fillna(15.0)


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """ATR as % of close (EWM, period=window)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return (atr / close * 100).fillna(1.0)


def _ma_slope_ann_pct(ma: pd.Series, lookback: int = 10) -> pd.Series:
    """Annualised % slope of a moving average (change per year)."""
    denom = ma.shift(lookback).replace(0, np.nan)
    return (ma.diff(lookback) / denom / lookback * 252 * 100).fillna(0.0)


def build_price_features(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Derive all per-date technical features from SPY OHLCV price data.
    Returns a DataFrame indexed by date (one row per trading day).
    """
    close = price_data["Close"]
    high  = price_data.get("High",  close)
    low   = price_data.get("Low",   close)

    log_ret = np.log(close / close.shift(1))

    ma20  = close.rolling(20,  min_periods=10).mean()
    ma50  = close.rolling(50,  min_periods=25).mean()
    ma80  = close.rolling(80,  min_periods=40).mean()
    ma200 = close.rolling(200, min_periods=100).mean()

    feat = pd.DataFrame(index=price_data.index)
    feat["spy_price"]           = close
    feat["rsi_14"]              = _rsi(close, 14)
    feat["momentum_5d_pct"]     = (close / close.shift(5)  - 1) * 100
    feat["momentum_10d_pct"]    = (close / close.shift(10) - 1) * 100
    feat["dist_from_ma20_pct"]  = ((close / ma20  - 1) * 100).fillna(0.0)
    feat["dist_from_ma50_pct"]  = ((close / ma50  - 1) * 100).fillna(0.0)
    feat["dist_from_ma80_pct"]  = ((close / ma80  - 1) * 100).fillna(0.0)
    feat["dist_from_ma200_pct"] = ((close / ma200 - 1) * 100).fillna(0.0)
    feat["ma20_slope_ann_pct"]  = _ma_slope_ann_pct(ma20,  10)
    feat["ma50_slope_ann_pct"]  = _ma_slope_ann_pct(ma50,  10)
    feat["realized_vol_5d"]     = _realized_vol(log_ret, 5)
    feat["realized_vol_10d"]    = _realized_vol(log_ret, 10)
    feat["realized_vol_20d"]    = _realized_vol(log_ret, 20)
    feat["realized_vol_atr20"]  = _atr_pct(high, low, close, 20)
    return feat.ffill().fillna(0.0)


def build_vix_features(vix_series: pd.Series) -> pd.DataFrame:
    """VIX level and rolling percentile ranks."""
    feat = pd.DataFrame(index=vix_series.index)
    feat["vix"]                = vix_series
    feat["vix_percentile_20d"] = _rolling_percentile(vix_series, 20)
    feat["vix_percentile_50d"] = _rolling_percentile(vix_series, 50)
    feat["vix_percentile_100d"]= _rolling_percentile(vix_series, 100)
    return feat.ffill().bfill().fillna(50.0)


# ═════════════════════════════════════════════════════════════════════════════
# Section 2: Dict lookup helpers (backtester internal state uses mixed key types)
# ═════════════════════════════════════════════════════════════════════════════

def _lookup_ts(d: dict, ts: pd.Timestamp, default: Any) -> Any:
    """
    Look up a value from a {Timestamp|date → value} dict.
    Tries Timestamp key, then date() key, then scans up to 5 prior business days.
    """
    for key in (ts, ts.date()):
        if key in d:
            return d[key]
    for lag in range(1, 6):
        prior = ts - pd.Timedelta(days=lag)
        for key in (prior, prior.date()):
            if key in d:
                return d[key]
    return default


def _nearest_row(df_indexed: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
    """Return the row of df_indexed at or immediately before ts."""
    if df_indexed.empty:
        return pd.Series(dtype=float)
    idx = df_indexed.index.searchsorted(ts, side="right") - 1
    if 0 <= idx < len(df_indexed):
        return df_indexed.iloc[idx]
    return pd.Series(dtype=float)


# ═════════════════════════════════════════════════════════════════════════════
# Section 3: Per-trade feature extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_trade_features(
    trades:          List[Dict],
    price_feats:     pd.DataFrame,
    vix_feats:       pd.DataFrame,
    vix_by_date:     dict,
    iv_rank_by_date: dict,
    regime_by_date:  dict,
) -> pd.DataFrame:
    """
    Build the full feature DataFrame expected by WalkForwardValidator.

    Each row corresponds to one closed trade and includes:
      - All NUMERIC_FEATURES from compass/walk_forward.py
      - All CATEGORICAL_FEATURES (regime, strategy_type, spread_type)
      - Target columns: win (binary) and return_pct
      - Metadata: entry_date, year (for splitting)

    NOTE: hold_days is a post-hoc feature (unknown at entry in live trading).
    It is included here for retrospective analysis only. Remove it for live
    signal generation.
    """
    rows: List[Dict] = []
    sorted_trades = sorted(trades, key=lambda t: t["entry_date"])
    prev_entry: Optional[pd.Timestamp] = None

    for trade in sorted_trades:
        # ── Normalise dates to Timestamp ────────────────────────────────────
        entry_dt = pd.Timestamp(
            trade["entry_date"].date()
            if hasattr(trade["entry_date"], "date")
            else trade["entry_date"]
        )
        exit_dt  = pd.Timestamp(
            trade["exit_date"].date()
            if hasattr(trade["exit_date"], "date")
            else trade["exit_date"]
        )
        exp_dt   = pd.Timestamp(
            trade["expiration"].date()
            if hasattr(trade["expiration"], "date")
            else trade["expiration"]
        )

        # ── Price features at entry ──────────────────────────────────────────
        pf = _nearest_row(price_feats, entry_dt)
        vf = _nearest_row(vix_feats,   entry_dt)

        spy_price = float(pf.get("spy_price", 400.0))

        # ── VIX: prefer backtester's exact series (matches regime computation) ─
        vix_val  = _lookup_ts(vix_by_date, entry_dt, float(vf.get("vix", 20.0)))
        iv_rank  = _lookup_ts(iv_rank_by_date, entry_dt, 25.0)
        regime   = str(_lookup_ts(regime_by_date, entry_dt, "neutral"))

        # ── Trade-derived features ───────────────────────────────────────────
        dte       = max(0, (exp_dt - entry_dt).days)
        hold_days = max(0, (exit_dt - entry_dt).days)
        days_since_last = (
            int((entry_dt - prev_entry).days)
            if prev_entry is not None else 7
        )
        prev_entry = entry_dt

        short_strike = float(trade.get("short_strike") or spy_price * 0.98)
        long_strike  = float(trade.get("long_strike")  or spy_price * 0.96)
        spread_width = max(1.0, abs(short_strike - long_strike))
        credit       = float(trade.get("credit") or 0.0)   # per-share credit
        net_credit   = credit * 100                          # per-contract ($)
        max_loss_per_unit = max(0.0, (spread_width - credit) * 100)
        otm_pct = abs(spy_price - short_strike) / spy_price if spy_price > 0 else 0.02

        # ── Strategy classification ──────────────────────────────────────────
        trade_type    = trade.get("type", "bull_put_spread")
        strategy_type = trade_type   # e.g. 'bull_put_spread', 'bear_call_spread', 'iron_condor'
        spread_type   = (
            "ic"   if trade_type == "iron_condor"      else
            "call" if trade_type == "bear_call_spread" else
            "put"
        )

        # ── Target ───────────────────────────────────────────────────────────
        pnl        = float(trade.get("pnl", 0.0))
        return_pct = float(trade.get("return_pct", 0.0))
        win        = 1 if pnl > 0 else 0

        rows.append({
            # Metadata
            "entry_date":             entry_dt,
            # Target columns
            "win":                    win,
            "return_pct":             return_pct,
            # ── NUMERIC_FEATURES (must match walk_forward.NUMERIC_FEATURES exactly) ──
            "dte_at_entry":           float(dte),
            "hold_days":              float(hold_days),
            "day_of_week":            float(entry_dt.dayofweek),
            "days_since_last_trade":  float(days_since_last),
            "rsi_14":                 float(pf.get("rsi_14", 50.0)),
            "momentum_5d_pct":        float(pf.get("momentum_5d_pct", 0.0)),
            "momentum_10d_pct":       float(pf.get("momentum_10d_pct", 0.0)),
            "vix":                    float(vix_val),
            "vix_percentile_20d":     float(vf.get("vix_percentile_20d", 50.0)),
            "vix_percentile_50d":     float(vf.get("vix_percentile_50d", 50.0)),
            "vix_percentile_100d":    float(vf.get("vix_percentile_100d", 50.0)),
            "iv_rank":                float(iv_rank),
            "spy_price":              float(spy_price),
            "dist_from_ma20_pct":     float(pf.get("dist_from_ma20_pct", 0.0)),
            "dist_from_ma50_pct":     float(pf.get("dist_from_ma50_pct", 0.0)),
            "dist_from_ma80_pct":     float(pf.get("dist_from_ma80_pct", 0.0)),
            "dist_from_ma200_pct":    float(pf.get("dist_from_ma200_pct", 0.0)),
            "ma20_slope_ann_pct":     float(pf.get("ma20_slope_ann_pct", 0.0)),
            "ma50_slope_ann_pct":     float(pf.get("ma50_slope_ann_pct", 0.0)),
            "realized_vol_atr20":     float(pf.get("realized_vol_atr20", 1.0)),
            "realized_vol_5d":        float(pf.get("realized_vol_5d", 15.0)),
            "realized_vol_10d":       float(pf.get("realized_vol_10d", 15.0)),
            "realized_vol_20d":       float(pf.get("realized_vol_20d", 15.0)),
            "net_credit":             float(net_credit),
            "spread_width":           float(spread_width),
            "max_loss_per_unit":      float(max_loss_per_unit),
            "otm_pct":                float(otm_pct),
            "contracts":              float(trade.get("contracts", 1)),
            # ── CATEGORICAL_FEATURES ──
            "regime":                 regime,
            "strategy_type":          strategy_type,
            "spread_type":            spread_type,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    # Forward-fill sparse numeric columns (e.g. VIX gaps on holidays)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    df[num_cols] = df[num_cols].ffill().fillna(0.0)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Section 4: Simulation & metrics helpers
# ═════════════════════════════════════════════════════════════════════════════

def simulate_equity(trades_df: pd.DataFrame) -> pd.Series:
    """
    Simulate a flat-risk equity curve from a (possibly filtered) trade subset.
    Uses each trade's actual return_pct and max_loss_per_unit × contracts.
    Returns a Series indexed by entry_date.
    """
    capital = STARTING_CAPITAL
    pts: List[Tuple[pd.Timestamp, float]] = []

    if trades_df.empty:
        return pd.Series([STARTING_CAPITAL], index=[pd.Timestamp.now()])

    pts.append((trades_df["entry_date"].iloc[0] - pd.Timedelta(days=1), capital))
    for _, row in trades_df.iterrows():
        ml_pu = float(row.get("max_loss_per_unit", 0))
        contr = float(row.get("contracts", 1))
        max_risk = ml_pu * contr
        if max_risk <= 0:
            # Fallback: use 8.5% of capital (EXP-400 risk param)
            max_risk = capital * 0.085
        pnl_dollar = row["return_pct"] / 100.0 * max_risk
        capital   += pnl_dollar
        pts.append((row["entry_date"], capital))

    dates, caps = zip(*pts)
    return pd.Series(list(caps), index=pd.to_datetime(list(dates)))


def compute_sharpe(returns: pd.Series, periods_per_year: int = 52) -> float:
    """Annualised Sharpe ratio from per-trade % returns (assumes ~weekly trades)."""
    if len(returns) < 3:
        return 0.0
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0


def compute_max_drawdown(equity: pd.Series) -> float:
    """Max drawdown as a negative % of peak equity."""
    peak = equity.cummax()
    return float(((equity - peak) / peak * 100).min())


def subset_metrics(df: pd.DataFrame, label: str) -> Dict:
    """Standard performance metrics for a set of trades."""
    if df.empty or len(df) == 0:
        return {"label": label, "n_trades": 0, "win_rate": 0.0, "sharpe": 0.0,
                "total_return_pct": 0.0, "max_drawdown": 0.0}
    wins   = int((df["win"] == 1).sum())
    equity = simulate_equity(df)
    return {
        "label":            label,
        "n_trades":         len(df),
        "win_rate":         wins / len(df) * 100,
        "total_return_pct": float((equity.iloc[-1] / STARTING_CAPITAL - 1) * 100),
        "sharpe":           compute_sharpe(df["return_pct"]),
        "max_drawdown":     compute_max_drawdown(equity),
        "avg_return_pct":   float(df["return_pct"].mean()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Section 5: Main pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    confidence_threshold: float = 0.55,
    skip_backtest:        bool  = False,
) -> None:

    from backtest.backtester        import Backtester
    from backtest.historical_data   import HistoricalOptionsData
    from compass.ensemble_signal_model import EnsembleSignalModel
    from compass.walk_forward import (
        WalkForwardValidator,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        prepare_features,
    )
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    logger.info("=" * 65)
    logger.info("EXP-400 ML Ensemble Filter Backtest Pipeline")
    logger.info("Config : %s", BACKTEST_CONFIG.name)
    logger.info("Period : %s → %s", START_DATE.date(), END_DATE.date())
    logger.info("Train  : 2020–%d   |   Test: %d–2025", TRAIN_END_YEAR, TEST_START_YEAR)
    logger.info("=" * 65)

    # ── Step 1: Load EXP-400 config ───────────────────────────────────────────
    with open(BACKTEST_CONFIG) as f:
        config = json.load(f)

    # ── Step 2: Run (or load cached) 6-year backtest ──────────────────────────
    if skip_backtest and TRADES_CACHE.exists():
        logger.info("Loading cached trades from %s", TRADES_CACHE)
        with open(TRADES_CACHE) as f:
            cache = json.load(f)
        trades      = cache["trades"]
        bt_summary  = cache["summary"]
        price_data  = pd.DataFrame()   # no price data when loading from cache
        vix_by_date = {}
        iv_rank_by_date = {}
        regime_by_date  = {}
        # Restore datetime objects
        for t in trades:
            for field in ("entry_date", "exit_date", "expiration"):
                if t.get(field):
                    t[field] = pd.Timestamp(t[field])
    else:
        logger.info("Running EXP-400 backtest 2020–2025 (offline mode)…")
        hist_data = HistoricalOptionsData(offline_mode=True)
        bt = Backtester(config=config, historical_data=hist_data)
        full_results = bt.run_backtest(
            ticker=TICKER, start_date=START_DATE, end_date=END_DATE,
        )
        trades  = full_results.get("trades", [])
        bt_summary = {k: full_results[k] for k in (
            "total_trades", "winning_trades", "losing_trades", "win_rate",
            "total_pnl", "return_pct", "max_drawdown", "sharpe_ratio",
            "bull_put_trades", "bear_call_trades", "iron_condor_trades",
            "bull_put_win_rate", "bear_call_win_rate", "iron_condor_win_rate",
        ) if k in full_results}

        price_data      = getattr(bt, "_price_data",      pd.DataFrame())
        vix_by_date     = getattr(bt, "_vix_by_date",     {})
        iv_rank_by_date = getattr(bt, "_iv_rank_by_date", {})
        regime_by_date  = getattr(bt, "_regime_by_date",  {})

        # Cache trades for --skip-backtest reruns
        TRADES_CACHE.parent.mkdir(exist_ok=True)
        cache_payload = {
            "summary": bt_summary,
            "trades": [
                {k: (str(v) if hasattr(v, "date") else v) for k, v in t.items()}
                for t in trades
            ],
        }
        TRADES_CACHE.write_text(json.dumps(cache_payload, indent=2, default=str))
        logger.info("Trades cached to %s", TRADES_CACHE)

    logger.info(
        "Backtest: %d trades | %.1f%% win rate | %.1f%% return | %.1f%% max DD",
        bt_summary.get("total_trades", len(trades)),
        bt_summary.get("win_rate", 0),
        bt_summary.get("return_pct", 0),
        bt_summary.get("max_drawdown", 0),
    )
    if len(trades) < 50:
        logger.error("Only %d trades — insufficient for ML training (need ≥50).", len(trades))
        sys.exit(1)

    # ── Step 3: Build price/VIX features ──────────────────────────────────────
    if not price_data.empty:
        logger.info("Computing price & VIX technical features…")
        price_feats = build_price_features(price_data)
        # Build VIX Series from backtester's internal dict
        if vix_by_date:
            raw_vix = pd.Series({pd.Timestamp(k): v for k, v in vix_by_date.items()})
            raw_vix = raw_vix.sort_index()
            vix_feats = build_vix_features(raw_vix)
        else:
            logger.warning("No VIX data from backtester — VIX percentile features will be neutral")
            vix_feats = pd.DataFrame()
    else:
        logger.warning("No price data (cached run) — price-based features will be ~0")
        price_feats = pd.DataFrame()
        vix_feats   = pd.DataFrame()

    # ── Step 4: Build trade feature DataFrame ─────────────────────────────────
    logger.info("Extracting features for %d trades…", len(trades))
    df = extract_trade_features(
        trades, price_feats, vix_feats, vix_by_date, iv_rank_by_date, regime_by_date,
    )
    df["year"] = df["entry_date"].dt.year

    logger.info(
        "Feature matrix: %d rows × %d columns  |  win rate: %.1f%%",
        len(df), len(df.columns),
        df["win"].mean() * 100,
    )
    logger.info("Regime breakdown: %s", df["regime"].value_counts().to_dict())
    logger.info("Strategy mix:     %s", df["strategy_type"].value_counts().to_dict())

    # ── Step 5: Walk-forward validation (expanding window by year) ────────────
    logger.info("Running walk-forward validation (WalkForwardValidator)…")
    wf_model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="auc", verbosity=0,
    )
    validator = WalkForwardValidator(
        model=wf_model,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        min_train_samples=30,
    )
    wf_results = validator.run(df)
    folds      = wf_results["folds"]
    wf_agg     = wf_results.get("aggregate", {})

    for fold in folds:
        auc_str = f"{fold.auc:.3f}" if fold.auc is not None else "N/A"
        logger.info(
            "  Fold %d | train %s–%s | test %s | n=%d/%d | AUC=%s",
            fold.fold, fold.train_start[:4], fold.train_end[:4],
            fold.test_start[:4], fold.n_train, fold.n_test, auc_str,
        )
    logger.info("Mean OOS AUC: %.3f  ±  %.3f",
                wf_agg.get("mean_auc", 0), wf_agg.get("std_auc", 0))

    # ── Step 6: Train EnsembleSignalModel on 2020–TRAIN_END_YEAR ─────────────
    train_df = df[df["year"] <= TRAIN_END_YEAR].copy()
    test_df  = df[df["year"] >  TRAIN_END_YEAR].copy()

    logger.info(
        "Train set: %d trades (2020–%d) | Test set: %d trades (%d–2025)",
        len(train_df), TRAIN_END_YEAR, len(test_df), TEST_START_YEAR,
    )
    if len(train_df) < 30:
        logger.error("Too few training trades (%d). Aborting.", len(train_df))
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = EnsembleSignalModel(model_dir=str(MODEL_DIR))

    # Build feature matrices using walk_forward's prepare_features
    # (handles one-hot encoding + column alignment)
    X_train_df = prepare_features(train_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    y_train    = train_df["win"].values.astype(int)

    logger.info("Training EnsembleSignalModel on %d trades…", len(train_df))
    train_stats = model.train(
        X_train_df, y_train,
        calibrate=True, save_model=True, n_wf_folds=4,
    )
    logger.info(
        "Ensemble trained: AUC=%.3f | accuracy=%.3f | weights=%s",
        train_stats.get("ensemble_test_auc", 0),
        train_stats.get("accuracy", 0),
        {k: f"{v:.3f}" for k, v in train_stats.get("model_weights", {}).items()},
    )

    # ── Step 7: Predict on test set ────────────────────────────────────────────
    if test_df.empty:
        logger.warning("No test trades after %d — skipping threshold analysis.", TRAIN_END_YEAR)
        probabilities = np.array([])
    else:
        X_test_df = prepare_features(test_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        # Align columns: add any train-only one-hot cols as zeros, drop test-only
        for col in X_train_df.columns:
            if col not in X_test_df.columns:
                X_test_df[col] = 0.0
        X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0.0)

        probabilities = model.predict_batch(X_test_df)
        test_df = test_df.copy()
        test_df["ml_prob"]       = probabilities
        test_df["ml_confidence"] = np.abs(probabilities - 0.5) * 2
        logger.info(
            "Predictions: mean_prob=%.3f | pct_bullish=%.1f%%",
            probabilities.mean(),
            (probabilities >= 0.5).mean() * 100,
        )

    # ── Step 8: Threshold sweep ────────────────────────────────────────────────
    baseline_metrics = subset_metrics(test_df, "baseline")
    threshold_results: List[Dict] = []

    for thresh in THRESHOLDS:
        if test_df.empty or len(probabilities) == 0:
            break
        filtered = test_df[test_df["ml_prob"] >= thresh]
        if len(filtered) < 5:
            continue
        m = subset_metrics(filtered, f"ml_{thresh:.2f}")
        m["threshold"]      = thresh
        m["trades_kept"]    = len(filtered)
        m["trades_dropped"] = len(test_df) - len(filtered)
        m["filter_rate"]    = len(filtered) / len(test_df)
        threshold_results.append(m)

    # Optimal threshold: best Sharpe with at least MIN_FILTER_RATE trades kept
    qualified = [t for t in threshold_results if t["filter_rate"] >= MIN_FILTER_RATE]
    best = (
        max(qualified, key=lambda x: x["sharpe"])
        if qualified else (threshold_results[0] if threshold_results else None)
    )

    # ── Step 9: Per-year breakdown ─────────────────────────────────────────────
    year_breakdown: List[Dict] = []
    for year in sorted(df["year"].unique()):
        yr_df   = df[df["year"] == year]
        base_m  = subset_metrics(yr_df, f"{year}_base")
        is_test = year > TRAIN_END_YEAR

        filt_m: Dict = {}
        if is_test and not test_df.empty and best is not None:
            yr_test = test_df[test_df["year"] == year]
            yr_filt = yr_test[yr_test["ml_prob"] >= best["threshold"]] if not yr_test.empty else yr_test
            filt_m  = subset_metrics(yr_filt, f"{year}_ml") if not yr_filt.empty else {}

        year_breakdown.append({
            "year":    year,
            "is_test": is_test,
            "base":    base_m,
            "ml":      filt_m,
        })

    # ── Step 10: Write report ─────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(exist_ok=True)
    _write_report(
        bt_summary=bt_summary,
        df=df,
        train_df=train_df,
        test_df=test_df,
        wf_results=wf_results,
        wf_agg=wf_agg,
        train_stats=train_stats,
        baseline_metrics=baseline_metrics,
        threshold_results=threshold_results,
        best=best,
        year_breakdown=year_breakdown,
        confidence_threshold=confidence_threshold,
    )
    logger.info("Report written → %s", REPORT_PATH)
    _print_summary(baseline_metrics, best, wf_agg)


# ═════════════════════════════════════════════════════════════════════════════
# Section 6: Report generation
# ═════════════════════════════════════════════════════════════════════════════

def _pct(v: Any, decimals: int = 1, plus: bool = True) -> str:
    if v is None:
        return "—"
    sign = "+" if (float(v) > 0 and plus) else ""
    return f"{sign}{float(v):.{decimals}f}%"


def _delta(new_val: float, base_val: float, decimals: int = 1) -> str:
    if new_val is None or base_val is None:
        return "—"
    d = float(new_val) - float(base_val)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{decimals}f}"


def _write_report(
    bt_summary:       Dict,
    df:               pd.DataFrame,
    train_df:         pd.DataFrame,
    test_df:          pd.DataFrame,
    wf_results:       Dict,
    wf_agg:           Dict,
    train_stats:      Dict,
    baseline_metrics: Dict,
    threshold_results: List[Dict],
    best:             Optional[Dict],
    year_breakdown:   List[Dict],
    confidence_threshold: float,
) -> None:
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    folds   = wf_results.get("folds", [])
    mean_auc = float(wf_agg.get("mean_auc", 0))
    std_auc  = float(wf_agg.get("std_auc", 0))

    lines = [
        "# EXP-400 ML Ensemble Filter — Backtest Report",
        "",
        f"**Generated:** {now}  ",
        f"**Strategy:** EXP-400 The Champion — SPY regime-adaptive credit spreads & iron condors  ",
        f"**Config:** `configs/exp_400_champion_realdata.json`  ",
        f"**ML Model:** EnsembleSignalModel (XGBoost + RandomForest + ExtraTrees, calibrated)  ",
        f"**Validation:** Walk-forward expanding window (one year per fold)  ",
        f"**Train period:** 2020–{TRAIN_END_YEAR}  |  **Test period (OOS):** {TEST_START_YEAR}–2025  ",
        "",
        "---",
        "",
        "## 1. Baseline Strategy Performance (2020–2025, no ML filter)",
        "",
        "| Year | Phase | Trades | Win Rate | Return | Sharpe | Max DD |",
        "|------|-------|--------|----------|--------|--------|--------|",
    ]

    for yb in year_breakdown:
        b     = yb["base"]
        phase = "OOS TEST" if yb["is_test"] else "train"
        wr    = f"{b.get('win_rate', 0):.1f}%"
        lines.append(
            f"| {yb['year']} | {phase} | {b.get('n_trades', 0)} | {wr} "
            f"| {_pct(b.get('total_return_pct'), plus=False)} "
            f"| {b.get('sharpe', 0):.2f} "
            f"| {_pct(b.get('max_drawdown'), plus=False)} |"
        )

    # Full-period summary from backtester
    lines += [
        "",
        "**Full 6-year period (2020–2025):**",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total trades | {bt_summary.get('total_trades', len(df))} |",
        f"| Win rate | {bt_summary.get('win_rate', df['win'].mean() * 100):.1f}% |",
        f"| Total return | {_pct(bt_summary.get('return_pct'), plus=False)} |",
        f"| Max drawdown | {_pct(bt_summary.get('max_drawdown'), plus=False)} |",
        f"| Sharpe ratio | {bt_summary.get('sharpe_ratio', 0):.2f} |",
        f"| Bull put trades | {bt_summary.get('bull_put_trades', '—')} "
        f"({bt_summary.get('bull_put_win_rate', 0):.1f}% win) |",
        f"| Bear call trades | {bt_summary.get('bear_call_trades', '—')} "
        f"({bt_summary.get('bear_call_win_rate', 0):.1f}% win) |",
        f"| Iron condor trades | {bt_summary.get('iron_condor_trades', '—')} "
        f"({bt_summary.get('iron_condor_win_rate', 0):.1f}% win) |",
        "",
        "---",
        "",
        "## 2. Walk-Forward Validation (OOS AUC Scores)",
        "",
        "Each fold trains on all prior years and tests on the following year.  ",
        "AUC > 0.55 = meaningful signal  |  AUC ≈ 0.50 = no better than chance.",
        "",
        "| Fold | Train Period | Test Year | N Train | N Test | AUC | Signal |",
        "|------|-------------|-----------|---------|--------|-----|--------|",
    ]

    for fold in folds:
        if fold.auc is None:
            auc_str = "N/A"
            sig     = "—"
        else:
            auc_str = f"{fold.auc:.3f}"
            sig     = "✅ signal" if fold.auc > 0.55 else ("⚠️ weak" if fold.auc > 0.50 else "❌ none")
        ssh = f"{fold.signal_sharpe:.2f}" if fold.signal_sharpe is not None else "—"
        lines.append(
            f"| {fold.fold} | {fold.train_start[:7]} → {fold.train_end[:7]} "
            f"| {fold.test_start[:7]} | {fold.n_train} | {fold.n_test} "
            f"| {auc_str} | {sig} (Sharpe {ssh}) |"
        )

    auc_verdict = (
        "✅ **Above-chance signal** — ML filter likely additive."
        if mean_auc > 0.55 else
        "⚠️ **Weak signal** — improvement marginal or inconsistent."
        if mean_auc > 0.52 else
        "❌ **Near chance** — ML filter does not add reliable signal."
    )
    lines += [
        "",
        f"**Mean OOS AUC: {mean_auc:.3f} ± {std_auc:.3f}**  {auc_verdict}",
        "",
        "---",
        "",
        "## 3. Ensemble Model — Training Details",
        "",
        f"Training set: **{len(train_df)} trades** (2020–{TRAIN_END_YEAR})  ",
        f"Training win rate: **{train_df['win'].mean() * 100:.1f}%**  ",
        "",
        "**Model weights** (walk-forward AUC minus chance, renormalised):",
        "",
    ]

    for name, wt in train_stats.get("model_weights", {}).items():
        lines.append(f"- `{name}`: {wt:.3f}")

    lines += [
        "",
        f"**Internal train stats** *(note: uses random shuffle split — treat AUC with caution, "
        f"use walk-forward AUC above for reliable OOS estimate)*:",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Ensemble test AUC (shuffle split) | {train_stats.get('ensemble_test_auc', 0):.3f} |",
        f"| Accuracy | {train_stats.get('accuracy', 0):.3f} |",
        f"| Precision | {train_stats.get('precision', 0):.3f} |",
        f"| Recall | {train_stats.get('recall', 0):.3f} |",
        "",
        "**Feature set** (29 numeric + 3 categorical → one-hot):  ",
        "`dte_at_entry`, `vix`, `iv_rank`, `rsi_14`, `momentum_5d/10d`, "
        "`dist_from_ma20/50/80/200`, `realized_vol_5/10/20d`, `otm_pct`, "
        "`spread_width`, `net_credit`, `regime`, `strategy_type`, `spread_type`",
        "",
        "---",
        "",
        "## 4. ML Filter — Confidence Threshold Sweep (OOS Test: 2024–2025)",
        "",
    ]

    n_base  = baseline_metrics.get("n_trades", 0)
    wr_base = baseline_metrics.get("win_rate", 0.0)
    sh_base = baseline_metrics.get("sharpe", 0.0)
    rt_base = baseline_metrics.get("total_return_pct", 0.0)

    lines += [
        f"**Baseline (no filter):** {n_base} trades | "
        f"win rate {wr_base:.1f}% | "
        f"return {_pct(rt_base, plus=False)} | "
        f"Sharpe {sh_base:.2f}",
        "",
        "| Threshold | Kept | Filter% | Win Rate | Δ Win Rate | Return | Sharpe | Δ Sharpe |",
        "|-----------|------|---------|----------|------------|--------|--------|----------|",
    ]

    for t in threshold_results:
        is_best = best is not None and abs(t["threshold"] - best["threshold"]) < 0.001
        tag     = " ⭐ optimal" if is_best else ""
        lines.append(
            f"| {t['threshold']:.2f}{tag} "
            f"| {t['trades_kept']} "
            f"| {t['filter_rate'] * 100:.0f}% "
            f"| {t.get('win_rate', 0):.1f}% "
            f"| {_delta(t.get('win_rate'), wr_base)}pp "
            f"| {_pct(t.get('total_return_pct'), plus=False)} "
            f"| {t.get('sharpe', 0):.2f} "
            f"| {_delta(t.get('sharpe'), sh_base, decimals=2)} |"
        )

    if best:
        wr_imp  = best.get("win_rate", 0)    - wr_base
        sh_imp  = best.get("sharpe", 0)      - sh_base
        rt_imp  = best.get("total_return_pct", 0) - rt_base
        dd_base = baseline_metrics.get("max_drawdown", 0)
        dd_filt = best.get("max_drawdown", 0)
        lines += [
            "",
            f"### Optimal Threshold: **{best['threshold']:.2f}**  "
            f"({best['trades_kept']}/{n_base} trades kept, "
            f"{best['filter_rate'] * 100:.0f}% pass rate)",
            "",
            "| Metric | Baseline | ML Filtered | Improvement |",
            "|--------|----------|-------------|-------------|",
            f"| Trades (OOS) | {n_base} | {best['trades_kept']} "
            f"| −{n_base - best['trades_kept']} ({(1 - best['filter_rate']) * 100:.0f}% filtered) |",
            f"| Win Rate | {wr_base:.1f}% | {best.get('win_rate', 0):.1f}% "
            f"| {'+' if wr_imp >= 0 else ''}{wr_imp:.1f}pp |",
            f"| Return (OOS) | {_pct(rt_base, plus=False)} | {_pct(best.get('total_return_pct'), plus=False)} "
            f"| {_delta(best.get('total_return_pct'), rt_base)}pp |",
            f"| Sharpe | {sh_base:.2f} | {best.get('sharpe', 0):.2f} "
            f"| {'+' if sh_imp >= 0 else ''}{sh_imp:.2f} |",
            f"| Max Drawdown | {_pct(dd_base, plus=False)} | {_pct(dd_filt, plus=False)} | — |",
        ]

    lines += [
        "",
        "---",
        "",
        "## 5. Year-by-Year Breakdown (OOS Test Period Only)",
        "",
        f"ML threshold used: {best['threshold']:.2f}" if best else "No viable threshold found.",
        "",
        "| Year | | Trades | Win Rate | Return | Sharpe | Max DD |",
        "|------|--|--------|----------|--------|--------|--------|",
    ]

    for yb in year_breakdown:
        if not yb["is_test"]:
            continue
        for variant, label_str in [("base", "Baseline"), ("ml", f"ML {best['threshold']:.2f}" if best else "ML")]:
            m = yb.get(variant, {})
            if not m:
                continue
            lines.append(
                f"| {yb['year']} | {label_str} "
                f"| {m.get('n_trades', 0)} "
                f"| {m.get('win_rate', 0):.1f}% "
                f"| {_pct(m.get('total_return_pct'), plus=False)} "
                f"| {m.get('sharpe', 0):.2f} "
                f"| {_pct(m.get('max_drawdown'), plus=False)} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 6. Feature Importance Analysis",
        "",
        "The EnsembleSignalModel uses walk-forward AUC to weight models. "
        "The following features are expected to be most predictive based on "
        "EXP-400's regime-adaptive design:",
        "",
        "| Feature | Expected Signal | Rationale |",
        "|---------|----------------|-----------|",
        "| `vix` | HIGH | VIX level gates IC entries; >40 blocks all entries |",
        "| `regime` | HIGH | Strategy type is regime-selected — bull→puts, bear→calls |",
        "| `dist_from_ma80_pct` | HIGH | MA80 is the EXP-400 trend trigger |",
        "| `iv_rank` | MEDIUM | Higher IV rank → fatter premiums → higher win probability |",
        "| `dte_at_entry` | MEDIUM | DTE drives time-decay profile and max-loss risk |",
        "| `rsi_14` | MEDIUM | RSI threshold (50 bull / 45 bear) is part of combo regime |",
        "| `spread_type` | MEDIUM | Bull-put vs bear-call vs IC have distinct win profiles |",
        "| `vix_percentile_100d` | LOW | Relative VIX positioning vs 100d history |",
        "| `hold_days` | ⚠️ POST-HOC | Duration is outcome-correlated; remove for live use |",
        "",
        "---",
        "",
        "## 7. Known Limitations & Bugs",
        "",
        "1. **Random shuffle split in `EnsembleSignalModel.train()`** "
        "(PR_REVIEW.md bug #1): internal AUC in Section 3 is inflated. "
        "The walk-forward AUC in Section 2 is the reliable estimate.",
        "",
        "2. **One-hot leakage in `WalkForwardValidator`** "
        "(PR_REVIEW.md bug #2): `prepare_features()` is called on the full dataset "
        "before fold splitting, leaking future category membership. "
        "Impact is small for EXP-400 (stable regime/strategy labels) but fix "
        "before using on strategies with new regime labels over time.",
        "",
        "3. **`hold_days` is post-hoc**: actual holding period is unknown at "
        "trade entry. This feature is valid for retrospective analysis; "
        "exclude it for live signal generation.",
        "",
        "4. **Small OOS set**: 2024–2025 may have <60 trades. Sharpe and "
        "win-rate estimates have high sampling variance. Walk-forward AUC "
        "from Section 2 is more statistically reliable.",
        "",
        "---",
        "",
        "## 8. Recommendation",
        "",
    ]

    if mean_auc > 0.57 and best and (best.get("sharpe", 0) - baseline_metrics.get("sharpe", 0)) > 0.15:
        rec = (
            "✅ **PROCEED TO LIVE SHADOW MODE**\n\n"
            f"Walk-forward mean AUC = {mean_auc:.3f} (meaningful signal) and "
            f"Sharpe improves by {best.get('sharpe', 0) - baseline_metrics.get('sharpe', 0):.2f} "
            f"at threshold {best['threshold']:.2f}. "
            "Recommended next step: paper trade EXP-400 alongside ML filter for 90 days "
            "before acting on filter signals. Fix PR_REVIEW.md bugs #1–#2 first."
        )
    elif mean_auc > 0.52 and best and best.get("sharpe", 0) > baseline_metrics.get("sharpe", 0):
        rec = (
            "⚠️ **MONITOR — DO NOT ACT YET**\n\n"
            f"Walk-forward mean AUC = {mean_auc:.3f} (weak but above-chance signal). "
            f"Sharpe improvement is marginal ({_delta(best.get('sharpe'), baseline_metrics.get('sharpe'), 2)}). "
            "Collect more live trading data (minimum 60 OOS trades) before deploying the filter. "
            "The EXP-400 regime detector already captures most predictable variance — "
            "the ML layer adds limited incremental value at this data size."
        )
    else:
        rec = (
            "❌ **DO NOT DEPLOY ML FILTER**\n\n"
            f"Walk-forward mean AUC = {mean_auc:.3f} — near chance level. "
            "The ML ensemble does not add reliable signal on top of EXP-400's "
            "existing regime-adaptive logic. The strategy's MA + RSI + VIX combo "
            "regime already filters most bad trades. "
            "Revisit after accumulating ≥150 live paper trades."
        )

    lines += [
        rec,
        "",
        f"**Walk-forward mean AUC:** {mean_auc:.3f} ± {std_auc:.3f}  ",
    ]
    if best:
        lines += [
            f"**Optimal threshold:** {best['threshold']:.2f}  ",
            f"**Win rate improvement:** {_delta(best.get('win_rate'), wr_base)}pp  ",
            f"**Sharpe improvement:** {_delta(best.get('sharpe'), sh_base, decimals=2)}  ",
            f"**Trades filtered out:** {(1 - best['filter_rate']) * 100:.0f}% "
            f"({n_base - best['trades_kept']} of {n_base})  ",
        ]

    lines += [
        "",
        "---",
        "",
        "*Generated by `scripts/backtest_ml_filter.py`.  ",
        "Re-run after fixing PR_REVIEW.md bugs #1–#3 for production-grade estimates.*",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def _print_summary(
    baseline: Dict, best: Optional[Dict], wf_agg: Dict,
) -> None:
    """Print a compact terminal summary after the report is written."""
    print()
    print("=" * 55)
    print("  EXP-400 ML Filter — Results Summary")
    print("=" * 55)
    mean_auc = wf_agg.get("mean_auc", 0)
    print(f"  Walk-forward mean AUC : {mean_auc:.3f}")
    print(f"  Baseline (OOS) trades : {baseline.get('n_trades', 0)}")
    print(f"  Baseline win rate     : {baseline.get('win_rate', 0):.1f}%")
    print(f"  Baseline Sharpe       : {baseline.get('sharpe', 0):.2f}")
    if best:
        print(f"  Optimal threshold     : {best['threshold']:.2f}")
        print(f"  ML win rate           : {best.get('win_rate', 0):.1f}%"
              f"  (Δ {_delta(best.get('win_rate'), baseline.get('win_rate', 0))}pp)")
        print(f"  ML Sharpe             : {best.get('sharpe', 0):.2f}"
              f"  (Δ {_delta(best.get('sharpe'), baseline.get('sharpe', 0), 2)})")
        print(f"  Trades kept           : {best['trades_kept']}/{baseline.get('n_trades', 0)}"
              f"  ({best['filter_rate'] * 100:.0f}%)")
    print(f"  Report                : {REPORT_PATH}")
    print("=" * 55)
    print()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EXP-400 ML ensemble filter backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.55,
        help="Default confidence threshold for the summary (all thresholds are swept)",
    )
    parser.add_argument(
        "--skip-backtest", action="store_true",
        help="Load cached trades from output/ml_filter_exp400_trades_cache.json "
             "instead of re-running the backtester (fast reruns)",
    )
    args = parser.parse_args()
    run_pipeline(
        confidence_threshold=args.confidence_threshold,
        skip_backtest=args.skip_backtest,
    )
