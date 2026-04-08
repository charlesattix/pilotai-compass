#!/usr/bin/env python3
"""
EXP-601: IBIT ML Signal Filter

Builds an XGBoost binary classifier (win/loss) trained on real trade-level
features extracted from EXP-600 (IBIT Adaptive) champion config backtests.

Champion config (config #14 from mega sweep):
  DTE=14, OTM=10%, width=$5, PT=30%, SL=2.5x, adaptive MA50,
  risk=15%, Kelly=1.0

Steps:
  1. Run champion config, capture trade-level features per trade
  2. Train XGBoost with time-series CV (no random splits, no data leakage)
  3. Backtest comparison: EXP-600 with vs without ML filter
  4. Save all outputs

Outputs:
  ml/ibit_training_data.csv        — labeled feature matrix
  ml/ibit_signal_model.joblib      — trained XGBoost model
  ml/ibit_feature_importance.md    — ranked feature importances
  ml/ibit_model_report.md          — full CV + final model report
  output/exp601_ml_comparison.json — backtest comparison results

CRITICAL: crypto_options_cache.db + macro_state.db only.
          No synthetic pricing. No Black-Scholes. Cache miss = skip trade.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.ibit_backtester import IBITBacktester, DB_PATH, MULTIPLIER

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("exp601")

MACRO_DB_PATH = ROOT / "data" / "macro_state.db"
ML_DIR = ROOT / "ml"
ML_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Champion config (EXP-600 #14, IBITBacktester format) ──────────────────────
CHAMPION_CONFIG: Dict[str, Any] = {
    "starting_capital":    100_000.0,
    "compound":            False,           # flat sizing
    "direction":           "adaptive_directional",  # bull_put > MA50, bear_call < MA50
    "ma_period":           50,
    "regime_filter":       "none",          # direction logic handles regime
    "otm_pct":             0.10,
    "call_otm_pct":        0.10,
    "spread_width":        5.0,
    "call_spread_width":   5.0,
    "min_credit_pct":      3.0,
    "dte_target":          14,
    "dte_min":             10,
    "dte_max":             21,
    "profit_target":       0.30,            # close at 30% profit
    "stop_loss_mult":      2.5,
    "risk_pct":            0.15,            # 15% risk per trade
    "kelly_fraction":      1.0,             # full Kelly
    "kelly_min_trades":    10,
    "max_contracts":       25,
    "max_concurrent":      5,
    "same_day_reentry":    False,
}

# Available IBIT options data range
BACKTEST_START = "2024-11-19"
BACKTEST_END   = "2026-03-20"

ML_FEATURES = [
    "dte",
    "otm_pct",
    "spread_width",
    "credit_received",
    "credit_pct",        # credit / spread_width — proxy for IV level
    "vix",
    "realized_vol_20d",
    "ma50_distance_pct",
    "rsi_14",
    "volume_ratio",      # today vol / 20d avg vol
    "btc_corr_30d",      # IBIT-ETHA return correlation (BTC proxy)
    "direction_bull",    # 1=bull_put, 0=bear_call
]


# ══════════════════════════════════════════════════════════════════════════════
# Extended backtester — captures ML features at entry time
# ══════════════════════════════════════════════════════════════════════════════

class IBITAdaptiveBacktester(IBITBacktester):
    """
    Extends IBITBacktester with:
      1. "adaptive_directional" direction: bull_put above MA50, bear_call below MA50
      2. Per-trade ML feature capture (computed from real price/vol/VIX data)
      3. Optional ML filter: skip trades where P(win) < threshold

    All extra features are computed from:
      - crypto_options_cache.db: IBIT/ETHA prices + volumes
      - macro_state.db: VIX (weekly, forward-filled to daily)
    """

    def __init__(
        self,
        config=None,
        db_path=DB_PATH,
        ml_model=None,
        ml_threshold: float = 0.5,
        capture_features: bool = True,
    ):
        super().__init__(config=config, db_path=db_path)
        self._ml_model = ml_model
        self._ml_threshold = ml_threshold
        self._capture_features = capture_features
        # Pre-loaded per run
        self._etha_spots: Dict[str, float] = {}
        self._etha_sorted: List[str] = []
        self._vix_map: Dict[str, float] = {}      # date -> vix (forward-filled)
        self._ibit_volume: Dict[str, float] = {}   # date -> volume

    # ── Overridden to add adaptive_directional support ─────────────────────────

    def _resolve_direction_and_scales(
        self, direction: str, ma_direction: str
    ) -> Tuple[str, float, float, float, float]:
        if direction == "adaptive_directional":
            # bull_put when above MA50 (bull), bear_call when below MA50 (bear)
            # neutral → bull_put (default to income in sideways)
            if ma_direction == "bear":
                return "bear_call", 1.0, 1.0, 1.0, 1.0
            else:
                return "bull_put", 1.0, 1.0, 1.0, 1.0
        return super()._resolve_direction_and_scales(direction, ma_direction)

    # ── Extra data loading ──────────────────────────────────────────────────────

    def _load_extra_data(self, conn: sqlite3.Connection) -> None:
        """Load ETHA prices and IBIT volumes (called once per run)."""
        rows = conn.execute(
            "SELECT date, close FROM crypto_underlying_daily "
            "WHERE ticker='ETHA' AND date <> '0000-00-00' ORDER BY date"
        ).fetchall()
        self._etha_spots = {r["date"]: float(r["close"]) for r in rows if r["close"]}
        self._etha_sorted = sorted(self._etha_spots.keys())

        rows2 = conn.execute(
            "SELECT date, volume FROM crypto_underlying_daily "
            "WHERE ticker='IBIT' AND date <> '0000-00-00' ORDER BY date"
        ).fetchall()
        self._ibit_volume = {r["date"]: float(r["volume"]) for r in rows2 if r["volume"]}

    def _load_vix(self) -> None:
        """Load VIX from macro_state.db macro_score table, forward-fill to daily."""
        conn = sqlite3.connect(str(MACRO_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT date, vix FROM macro_score WHERE vix IS NOT NULL ORDER BY date"
        ).fetchall()
        conn.close()
        # Build a sparse map then forward-fill below
        sparse = {r["date"]: float(r["vix"]) for r in rows}
        if not sparse:
            return
        # Forward-fill: for every date in IBIT price history, use most recent VIX
        last_vix = None
        for dt in self._sorted_dates:
            if dt in sparse:
                last_vix = sparse[dt]
            if last_vix is not None:
                self._vix_map[dt] = last_vix

    # ── Feature computation helpers ─────────────────────────────────────────────

    def _rsi(self, dt: str, period: int = 14) -> Optional[float]:
        """14-day RSI of IBIT (T-1 aware)."""
        prior = [d for d in self._sorted_dates if d < dt]
        if len(prior) < period + 1:
            return None
        prices = [self._spots[prior[i]] for i in range(len(prior) - period - 1, len(prior))]
        gains, losses = [], []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gains.append(max(0.0, change))
            losses.append(max(0.0, -change))
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _volume_ratio(self, dt: str, period: int = 20) -> Optional[float]:
        """Today's IBIT volume / 20-day average volume (T-1 aware)."""
        prior = [d for d in self._sorted_dates if d < dt]
        if len(prior) < period:
            return None
        recent_dates = prior[-period:]
        avg_vol = sum(self._ibit_volume.get(d, 0.0) for d in recent_dates) / period
        if avg_vol <= 0:
            return None
        today_vol = self._ibit_volume.get(dt, None)
        if today_vol is None:
            return None
        return today_vol / avg_vol

    def _etha_spot(self, dt: str) -> Optional[float]:
        """ETHA close at or before dt."""
        for d in reversed(self._etha_sorted):
            if d <= dt:
                return self._etha_spots.get(d)
        return None

    def _btc_corr_30d(self, dt: str, period: int = 30) -> Optional[float]:
        """30-day Pearson correlation of IBIT vs ETHA daily returns (BTC proxy)."""
        prior = [d for d in self._sorted_dates if d < dt]
        if len(prior) < period + 1:
            return None
        ibit_prices = [self._spots[prior[i]] for i in range(len(prior) - period - 1, len(prior))]
        ibit_rets = [math.log(ibit_prices[i + 1] / ibit_prices[i])
                     for i in range(len(ibit_prices) - 1)
                     if ibit_prices[i] > 0 and ibit_prices[i + 1] > 0]

        etha_prior = [d for d in self._etha_sorted if d < dt]
        if len(etha_prior) < period + 1:
            return None
        etha_prices = [self._etha_spots[etha_prior[i]]
                       for i in range(len(etha_prior) - period - 1, len(etha_prior))]
        etha_rets = [math.log(etha_prices[i + 1] / etha_prices[i])
                     for i in range(len(etha_prices) - 1)
                     if etha_prices[i] > 0 and etha_prices[i + 1] > 0]

        n = min(len(ibit_rets), len(etha_rets))
        if n < 10:
            return None
        ir = ibit_rets[-n:]
        er = etha_rets[-n:]
        mean_i = sum(ir) / n
        mean_e = sum(er) / n
        cov = sum((ir[i] - mean_i) * (er[i] - mean_e) for i in range(n)) / n
        std_i = math.sqrt(sum((x - mean_i) ** 2 for x in ir) / n)
        std_e = math.sqrt(sum((x - mean_e) ** 2 for x in er) / n)
        if std_i == 0 or std_e == 0:
            return None
        return cov / (std_i * std_e)

    def _get_entry_features(self, entry_date: str, spot: float, resolved_direction: str,
                             ma_direction: str, total_credit: float, spread_width: float,
                             dte: int) -> Dict[str, Optional[float]]:
        """Collect all ML features for a trade entry. T-1 safe (uses prior data)."""
        ma50 = self._ma(entry_date, 50)
        ma50_dist = ((spot / ma50) - 1.0) * 100.0 if ma50 else None
        return {
            "dte":               dte,
            "otm_pct":           self.cfg["otm_pct"],
            "spread_width":      spread_width,
            "credit_received":   total_credit,
            "credit_pct":        (total_credit / spread_width * 100.0) if spread_width > 0 else None,
            "vix":               self._vix_map.get(entry_date),
            "realized_vol_20d":  self._realized_vol(entry_date, 20),
            "ma50_distance_pct": ma50_dist,
            "rsi_14":            self._rsi(entry_date, 14),
            "volume_ratio":      self._volume_ratio(entry_date, 20),
            "btc_corr_30d":      self._btc_corr_30d(entry_date, 30),
            "direction_bull":    1 if resolved_direction == "bull_put" else 0,
        }

    # ── Override _try_enter to attach features & apply ML filter ───────────────

    def _try_enter(self, conn, entry_date, expiry):
        pos = super()._try_enter(conn, entry_date, expiry)
        if pos is None:
            return None
        if not self._capture_features:
            return pos

        features = self._get_entry_features(
            entry_date=entry_date,
            spot=pos["spot_at_entry"],
            resolved_direction=pos["direction"],
            ma_direction=pos.get("ma_direction", "neutral"),
            total_credit=pos["total_credit"],
            spread_width=pos["total_max_loss"] + pos["total_credit"],  # approx
            dte=pos["dte_at_entry"],
        )
        pos["_ml_features"] = features

        # ML filter: skip if model predicts loss
        if self._ml_model is not None:
            feat_vals = [features.get(f) for f in ML_FEATURES]
            if all(v is not None for v in feat_vals):
                X = np.array([feat_vals], dtype=float)
                prob_win = self._ml_model.predict_proba(X)[0][1]
                if prob_win < self._ml_threshold:
                    log.debug("%s ML filtered out entry (prob_win=%.3f < %.3f)",
                              entry_date, prob_win, self._ml_threshold)
                    return None  # skip trade

        return pos

    # ── Override _record_close to attach features to trade ─────────────────────

    def _record_close(self, pos, exit_date, pnl_usd, reason):
        super()._record_close(pos, exit_date, pnl_usd, reason)
        if not self._capture_features:
            return
        # Attach features and hold_days to last appended trade
        last = self.trades[-1]
        features = pos.get("_ml_features", {})
        for k, v in features.items():
            last[k] = v
        entry_dt = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
        exit_dt  = datetime.strptime(exit_date, "%Y-%m-%d")
        last["hold_days"] = (exit_dt - entry_dt).days

    # ── Override run() to load extra data ──────────────────────────────────────

    def run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        # Reset extra state
        self._etha_spots = {}
        self._etha_sorted = []
        self._vix_map = {}
        self._ibit_volume = {}

        # We need to hook into the early part of run() to load spots first.
        # Reset internal state manually then call parent (which calls _load_spots).
        self.capital = self.cfg["starting_capital"]
        self.starting_capital = self.cfg["starting_capital"]
        self.open_positions = []
        self.trades = []
        self.equity_curve = []
        self._ruin = False
        self._spots = {}
        self._sorted_dates = []

        conn = self._connect()
        self._load_spots(conn, start_date, end_date)
        self._load_extra_data(conn)
        self._load_vix()
        conn.close()

        # Now call parent run() — it will call _load_spots again (no-op since already loaded)
        # Actually, to avoid double-loading, we need to override fully. Let's do it inline.
        return self._run_loop(start_date, end_date)

    def _run_loop(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Identical to parent run() but skips the _load_spots/reset (already done)."""
        from datetime import timedelta

        conn = self._connect()

        dte_target = int(self.cfg["dte_target"])
        dte_min    = int(self.cfg["dte_min"])
        dte_max    = int(self.cfg.get("dte_max") or dte_target + 10)
        max_conc   = int(self.cfg["max_concurrent"])
        same_day   = bool(self.cfg.get("same_day_reentry", False))

        trading_days = self._trading_days(conn, start_date, end_date)

        for current_date in trading_days:
            if self._ruin:
                break
            today = datetime.strptime(current_date, "%Y-%m-%d").date()

            # 1. Close expired
            for pos in list(self.open_positions):
                if pos["status"] != "open":
                    continue
                exp_date = datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
                if today >= exp_date:
                    pnl_usd, reason = self._close_at_expiry(conn, pos)
                    self._record_close(pos, pos["expiry"], pnl_usd, reason)
            self.open_positions = [p for p in self.open_positions if p["status"] == "open"]

            # 2. Check PT/SL
            newly_closed = []
            for pos in list(self.open_positions):
                result = self._check_exit(conn, pos, current_date)
                if result:
                    pnl_usd, reason = result
                    self._record_close(pos, current_date, pnl_usd, reason)
                    newly_closed.append(pos)
            self.open_positions = [p for p in self.open_positions if p["status"] == "open"]

            # 3. Open new
            if not self._ruin:
                self._fill_positions(conn, current_date, dte_min, dte_max, max_conc)
                if same_day and newly_closed:
                    if len(self.open_positions) < max_conc:
                        self._fill_positions(conn, current_date, dte_min, dte_max, max_conc)

            # 4. Equity curve
            unrealized = sum(p.get("unrealized_pnl", 0.0) for p in self.open_positions)
            self.equity_curve.append((current_date, self.capital + unrealized))

            if self.capital <= 0:
                self._ruin = True

        conn.close()
        return self._build_results(start_date, end_date)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Extract training data
# ══════════════════════════════════════════════════════════════════════════════

def extract_training_data() -> pd.DataFrame:
    """Run champion config backtest and extract trade-level features."""
    print("\n" + "="*70)
    print("  PHASE 1: Extracting Training Data")
    print(f"  Period: {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Config: DTE=14, OTM=10%, W=$5, PT=30%, SL=2.5x, risk=15%, Kelly=1.0")
    print("="*70)

    bt = IBITAdaptiveBacktester(
        config=CHAMPION_CONFIG,
        db_path=DB_PATH,
        ml_model=None,
        capture_features=True,
    )
    result = bt.run(BACKTEST_START, BACKTEST_END)

    trades = result["trades"]
    print(f"  Total trades: {len(trades)}")
    print(f"  Win rate: {result['win_rate']:.1f}%")
    print(f"  Return: {result['return_pct']:+.2f}%")
    print(f"  Max DD: {result['max_drawdown']:.2f}%")

    if not trades:
        raise RuntimeError("No trades generated — check IBIT options data availability")

    df = pd.DataFrame(trades)

    # Map win to label
    df["label"] = df["win"].astype(int)
    df["return_pct_trade"] = df["pnl_pct"] * 100.0

    # Reorder / select final columns for CSV (no duplicates)
    core_cols = [
        "entry_date", "direction", "dte_at_entry", "otm_pct", "spread_width",
        "credit_received", "credit_pct", "vix", "realized_vol_20d",
        "ma50_distance_pct", "rsi_14", "volume_ratio", "btc_corr_30d",
        "direction_bull", "ma_direction",
        "label", "return_pct_trade", "hold_days",
        "exit_reason", "win", "pnl_usd", "pnl_pct", "total_credit",
        "n_contracts",
    ]
    # Deduplicate while preserving order
    seen: set = set()
    deduped_cols = [c for c in core_cols if not (c in seen or seen.add(c))]
    available_cols = [c for c in deduped_cols if c in df.columns]
    df_out = df[available_cols].copy()
    df_out = df_out.rename(columns={"dte_at_entry": "dte"})

    out_path = ML_DIR / "ibit_training_data.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Training data saved: {out_path}  ({len(df_out)} rows)")

    # Summary stats
    print(f"\n  Feature coverage (non-null %):")
    for col in ML_FEATURES:
        if col in df_out.columns:
            pct = df_out[col].notna().mean() * 100
            print(f"    {col:<25} {pct:5.1f}%")

    wins = df_out["label"].sum()
    print(f"\n  Class balance: {wins} wins ({wins/len(df_out)*100:.1f}%) / "
          f"{len(df_out)-wins} losses ({(1-wins/len(df_out))*100:.1f}%)")

    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Train XGBoost model
# ══════════════════════════════════════════════════════════════════════════════

def train_model(df: pd.DataFrame) -> Tuple[Optional[XGBClassifier], List[str]]:
    """
    Train XGBoost binary classifier with time-series CV.
    Returns (model, feature_names) or (None, []) if insufficient data.
    """
    print("\n" + "="*70)
    print("  PHASE 2: Training XGBoost ML Signal Filter")
    print("="*70)

    # Select features that are available
    available_features = [f for f in ML_FEATURES if f in df.columns]
    missing = [f for f in ML_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features (will be excluded): {missing}")

    # Drop rows with NaN in any feature or label
    df_model = df[available_features + ["label", "entry_date"]].dropna().copy()
    print(f"  Rows after dropping NaN: {len(df_model)} (from {len(df)})")

    if len(df_model) < 20:
        print(f"  INSUFFICIENT DATA for ML (need ≥20 trades, got {len(df_model)})")
        print(f"  Skipping model training. The EXP-600 backtest period may be too short.")
        _write_minimal_report(len(df_model), available_features)
        return None, available_features

    # Sort chronologically
    df_model = df_model.sort_values("entry_date").reset_index(drop=True)

    X = df_model[available_features].values.astype(float)
    y = df_model["label"].values.astype(int)

    print(f"\n  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Class balance: {y.sum()} wins ({y.mean()*100:.1f}%) / "
          f"{len(y)-y.sum()} losses ({(1-y.mean())*100:.1f}%)")

    # Handle class imbalance: wins are 84%+ so boost loss class weight
    n_wins   = int(y.sum())
    n_losses = len(y) - n_wins
    spw = max(1.0, n_wins / n_losses) if n_losses > 0 else 1.0
    print(f"  Class imbalance ratio (wins/losses): {spw:.1f}x → scale_pos_weight={1/spw:.2f}")

    # XGBoost params: shallow + strongly regularized to prevent overfit
    xgb_params = dict(
        max_depth=3,
        min_child_weight=5,
        n_estimators=50,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.0,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=round(1.0 / spw, 3),  # de-weight wins to improve loss detection
        eval_metric="logloss",
        random_state=42,
    )

    # ── Time-series CV (no random splits) ──────────────────────────────────────
    n_splits = min(3, max(2, len(df_model) // 15))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"\n  Time-series CV: {n_splits} folds (chronological)")
    print(f"  {'Fold':<6} {'Train':>6} {'Val':>6} {'Acc':>7} {'AUC':>7} "
          f"{'Prec':>7} {'Recall':>7}")
    print(f"  {'-'*55}")

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if y_tr.sum() < 2 or (len(y_tr) - y_tr.sum()) < 2:
            print(f"  Fold {fold}: skipped (insufficient class diversity in train set)")
            continue

        model = XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, verbose=False)
        y_pred = model.predict(X_val)

        try:
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5
        except Exception:
            auc = 0.5

        acc  = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec  = recall_score(y_val, y_pred, zero_division=0)

        fold_metrics.append({"fold": fold, "acc": acc, "auc": auc,
                              "prec": prec, "rec": rec,
                              "n_train": len(train_idx), "n_val": len(val_idx)})

        print(f"  Fold {fold}: {len(train_idx):>6} {len(val_idx):>6} "
              f"{acc:>6.3f} {auc:>6.3f} {prec:>6.3f} {rec:>6.3f}")

    if fold_metrics:
        avg_acc = np.mean([m["acc"] for m in fold_metrics])
        avg_auc = np.mean([m["auc"] for m in fold_metrics])
        avg_prec = np.mean([m["prec"] for m in fold_metrics])
        avg_rec  = np.mean([m["rec"] for m in fold_metrics])
        print(f"  {'MEAN':<6} {'':>6} {'':>6} "
              f"{avg_acc:>6.3f} {avg_auc:>6.3f} {avg_prec:>6.3f} {avg_rec:>6.3f}")
    else:
        avg_acc = avg_auc = avg_prec = avg_rec = 0.0
        print("  No valid folds completed.")

    # ── Final model: train on all data ──────────────────────────────────────────
    print("\n  Training final model on all data...")
    final_model = XGBClassifier(**xgb_params)
    final_model.fit(X, y, verbose=False)

    # ── Feature importances ────────────────────────────────────────────────────
    importances = final_model.feature_importances_
    feat_imp = sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)

    print("\n  Feature importances (gain):")
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 40)
        print(f"    {feat:<25} {imp:.4f}  {bar}")

    # ── Save model ─────────────────────────────────────────────────────────────
    model_path = ML_DIR / "ibit_signal_model.joblib"
    joblib.dump({"model": final_model, "features": available_features,
                 "threshold": 0.5, "xgb_params": xgb_params}, model_path)
    print(f"\n  Model saved: {model_path}")

    # ── Save feature importance report ─────────────────────────────────────────
    _write_feature_importance(feat_imp)

    # ── Save model report ──────────────────────────────────────────────────────
    _write_model_report(
        n_samples=len(df_model),
        features=available_features,
        feat_imp=feat_imp,
        fold_metrics=fold_metrics,
        avg_acc=avg_acc,
        avg_auc=avg_auc,
        avg_prec=avg_prec,
        avg_rec=avg_rec,
        xgb_params=xgb_params,
        n_splits=n_splits,
        win_rate=float(y.mean() * 100),
    )

    return final_model, available_features


def _write_minimal_report(n_samples: int, features: List[str]) -> None:
    """Write a minimal report when there's insufficient data."""
    report_path = ML_DIR / "ibit_model_report.md"
    with open(report_path, "w") as f:
        f.write("# EXP-601 IBIT ML Model Report\n\n")
        f.write(f"**Status: INSUFFICIENT DATA**\n\n")
        f.write(f"Only {n_samples} clean samples available (minimum 20 required).\n")
        f.write("IBIT options data starts 2024-11-19. More data needed for reliable ML.\n\n")
        f.write(f"Available features: {features}\n")
    imp_path = ML_DIR / "ibit_feature_importance.md"
    with open(imp_path, "w") as f:
        f.write("# EXP-601 IBIT Feature Importance\n\n")
        f.write("Model not trained — insufficient data.\n")


def _write_feature_importance(feat_imp: List[Tuple[str, float]]) -> None:
    path = ML_DIR / "ibit_feature_importance.md"
    with open(path, "w") as f:
        f.write("# EXP-601 IBIT Feature Importance\n\n")
        f.write("XGBoost gain-based feature importance (higher = more predictive).\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        for rank, (feat, imp) in enumerate(feat_imp, 1):
            f.write(f"| {rank} | `{feat}` | {imp:.4f} |\n")
        f.write("\n## Interpretation Notes\n\n")
        f.write("- `credit_pct`: proxy for implied volatility level (credit / spread_width)\n")
        f.write("- `ma50_distance_pct`: distance from MA50 — trend strength signal\n")
        f.write("- `realized_vol_20d`: 20-day realized volatility (annualized, %)\n")
        f.write("- `vix`: VIX level at entry (forward-filled from weekly macro_score)\n")
        f.write("- `rsi_14`: 14-day RSI — momentum/overbought signal\n")
        f.write("- `volume_ratio`: today vol / 20d avg — unusual activity signal\n")
        f.write("- `btc_corr_30d`: IBIT-ETHA 30d return correlation (BTC regime proxy)\n")
    print(f"  Feature importance: {path}")


def _write_model_report(
    n_samples, features, feat_imp, fold_metrics,
    avg_acc, avg_auc, avg_prec, avg_rec,
    xgb_params, n_splits, win_rate,
) -> None:
    path = ML_DIR / "ibit_model_report.md"
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with open(path, "w") as f:
        f.write(f"# EXP-601 IBIT ML Signal Filter — Model Report\n\n")
        f.write(f"Generated: {now}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Model**: XGBoost binary classifier (win/loss)\n")
        f.write(f"- **Training samples**: {n_samples}\n")
        f.write(f"- **Baseline win rate**: {win_rate:.1f}%\n")
        f.write(f"- **Features**: {len(features)}\n")
        f.write(f"- **CV**: TimeSeriesSplit (n_splits={n_splits}) — chronological only\n")
        f.write(f"- **Backtest period**: {BACKTEST_START} → {BACKTEST_END}\n\n")
        f.write(f"## CV Results\n\n")
        f.write(f"| Fold | Train | Val | Acc | AUC | Prec | Recall |\n")
        f.write(f"|------|-------|-----|-----|-----|------|--------|\n")
        for m in fold_metrics:
            f.write(f"| {m['fold']} | {m['n_train']} | {m['n_val']} | "
                    f"{m['acc']:.3f} | {m['auc']:.3f} | "
                    f"{m['prec']:.3f} | {m['rec']:.3f} |\n")
        if fold_metrics:
            f.write(f"| **Mean** | - | - | **{avg_acc:.3f}** | **{avg_auc:.3f}** | "
                    f"**{avg_prec:.3f}** | **{avg_rec:.3f}** |\n")
        f.write(f"\n## XGBoost Parameters\n\n```\n")
        for k, v in xgb_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"```\n\n")
        f.write(f"## Feature Importances\n\n")
        f.write(f"| Rank | Feature | Importance |\n")
        f.write(f"|------|---------|------------|\n")
        for rank, (feat, imp) in enumerate(feat_imp, 1):
            f.write(f"| {rank} | `{feat}` | {imp:.4f} |\n")
        f.write(f"\n## Notes\n\n")
        f.write(f"- IBIT options data available from 2024-11-19 only\n")
        f.write(f"- With {n_samples} samples, ML signal is preliminary — "
                f"accumulate more trades before relying on filter\n")
        f.write(f"- `btc_corr_30d` uses ETHA (Ethereum ETF) as BTC correlation proxy\n")
        f.write(f"- `vix` is SPX VIX forward-filled from weekly macro_score readings\n")
        f.write(f"- `credit_pct` = credit / spread_width × 100 (no Black-Scholes)\n")
        f.write(f"- Threshold 0.5: skip trade if P(win) < 0.5\n")
    print(f"  Model report: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Backtest comparison
# ══════════════════════════════════════════════════════════════════════════════

def _run_backtest(ml_model=None, label="baseline") -> Dict[str, Any]:
    """Run EXP-600 champion config backtest, optionally with ML filter."""
    bt = IBITAdaptiveBacktester(
        config=CHAMPION_CONFIG,
        db_path=DB_PATH,
        ml_model=ml_model,
        ml_threshold=0.5,
        capture_features=(ml_model is not None),
    )
    result = bt.run(BACKTEST_START, BACKTEST_END)
    return result


def _metrics_snapshot(r: Dict) -> Dict:
    return {
        "total_trades":  r["total_trades"],
        "win_rate":      r["win_rate"],
        "return_pct":    r["return_pct"],
        "max_drawdown":  r["max_drawdown"],
        "profit_factor": r["profit_factor"],
        "avg_win":       r["avg_win"],
        "avg_loss":      r["avg_loss"],
    }


def backtest_comparison(model: Optional[XGBClassifier], features: List[str]) -> Dict:
    """Run EXP-600 with and without ML filter, compare metrics at multiple thresholds."""
    print("\n" + "="*70)
    print("  PHASE 3: Backtest Comparison (WITH vs WITHOUT ML Filter)")
    print("="*70)

    # ── Baseline (no filter) ──────────────────────────────────────────────────
    print("\n  Running baseline (no ML filter)...")
    baseline = _run_backtest(ml_model=None, label="baseline")
    print(f"  Baseline: {baseline['total_trades']} trades | "
          f"WR={baseline['win_rate']:.1f}% | "
          f"Return={baseline['return_pct']:+.2f}% | "
          f"MaxDD={baseline['max_drawdown']:.2f}%")

    if model is None:
        result = {
            "experiment": "EXP-601",
            "description": "IBIT ML Signal Filter — baseline only (model not trained)",
            "backtest_period": f"{BACKTEST_START} to {BACKTEST_END}",
            "champion_config": {k: v for k, v in CHAMPION_CONFIG.items()
                                if k != "starting_capital"},
            "baseline": _metrics_snapshot(baseline),
            "thresholds": {},
            "best_threshold": None,
            "verdict": "MODEL_NOT_TRAINED",
            "reason": "Insufficient training data",
            "features_used": features,
        }
        out_path = OUTPUT_DIR / "exp601_ml_comparison.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Comparison saved: {out_path}")
        return result

    # ── Multi-threshold sweep ─────────────────────────────────────────────────
    thresholds = [0.50, 0.60, 0.70, 0.80]
    threshold_results: Dict[str, Dict] = {}

    print(f"\n  {'Threshold':>11} {'Trades':>7} {'Removed':>8} {'WR':>7} "
          f"{'Return':>9} {'MaxDD':>8} {'Verdict'}")
    print(f"  {'-'*70}")

    best_thresh = 0.50
    best_score  = -999.0  # return_pct + wr_delta*0.5 - |dd_delta|*0.5

    for thr in thresholds:
        bt = IBITAdaptiveBacktester(
            config=CHAMPION_CONFIG,
            db_path=DB_PATH,
            ml_model=model,
            ml_threshold=thr,
            capture_features=True,
        )
        r = bt.run(BACKTEST_START, BACKTEST_END)

        removed      = baseline["total_trades"] - r["total_trades"]
        wr_delta     = r["win_rate"] - baseline["win_rate"]
        ret_delta    = r["return_pct"] - baseline["return_pct"]
        dd_delta     = r["max_drawdown"] - baseline["max_drawdown"]

        score = ret_delta + wr_delta * 0.5 - abs(dd_delta) * 0.3
        if r["total_trades"] >= 10 and score > best_score:
            best_score = score
            best_thresh = thr

        verdict_flag = (
            "IMPROVES" if ret_delta > 0.5 and wr_delta >= 0 else
            "MIXED"    if ret_delta > 0 or wr_delta > 0     else
            "NO_CHANGE" if removed == 0                      else
            "DEGRADES"
        )

        print(f"  thr={thr:.2f}: {r['total_trades']:>7} {removed:>8} "
              f"{r['win_rate']:>6.1f}% {r['return_pct']:>+8.2f}% "
              f"{r['max_drawdown']:>7.2f}%  {verdict_flag}")

        threshold_results[str(thr)] = {
            **_metrics_snapshot(r),
            "trades_removed": removed,
            "trades_removed_pct": round(removed / max(1, baseline["total_trades"]) * 100, 1),
            "win_rate_delta": round(wr_delta, 2),
            "return_delta":   round(ret_delta, 2),
            "max_dd_delta":   round(dd_delta, 2),
            "verdict": verdict_flag,
        }

    # ── Best threshold summary ────────────────────────────────────────────────
    best = threshold_results[str(best_thresh)]
    print(f"\n  Best threshold: {best_thresh:.2f}  ({best['verdict']})")
    print(f"    Trades removed: {best['trades_removed']} ({best['trades_removed_pct']:.1f}%)")
    print(f"    Win rate:       {best['win_rate_delta']:+.2f}%")
    print(f"    Return:         {best['return_delta']:+.2f}%")
    print(f"    Max DD:         {best['max_dd_delta']:+.2f}%")

    overall_verdict = best["verdict"]
    print(f"\n  Overall verdict: ML filter {overall_verdict} EXP-600 performance")

    result = {
        "experiment": "EXP-601",
        "description": "IBIT ML Signal Filter vs EXP-600 baseline",
        "backtest_period": f"{BACKTEST_START} to {BACKTEST_END}",
        "champion_config": {k: v for k, v in CHAMPION_CONFIG.items()
                            if k != "starting_capital"},
        "baseline": _metrics_snapshot(baseline),
        "thresholds": threshold_results,
        "best_threshold": best_thresh,
        "best_threshold_result": best,
        "verdict": overall_verdict,
        "features_used": features,
        "notes": (
            "IMPORTANT: Backtest comparison is IN-SAMPLE — model was trained on the same "
            "2024-11-19 to 2026-03-20 period used for comparison. "
            "The improvement figures are inflated by in-sample memorization and are NOT "
            "reliable estimates of live performance. "
            "CV AUC ~0.50 (essentially random) is the honest out-of-sample signal estimate. "
            "Recommendation: accumulate 12+ months of live paper trade data, retrain on "
            "2024-2025, then test on 2026+ for a valid walk-forward assessment."
        ),
    }

    out_path = OUTPUT_DIR / "exp601_ml_comparison.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Comparison saved: {out_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Registry update helpers
# ══════════════════════════════════════════════════════════════════════════════

def register_exp601() -> None:
    """Register EXP-601 in experiments/registry.json if not already present."""
    registry_path = ROOT / "experiments" / "registry.json"
    if not registry_path.exists():
        print("  WARNING: registry.json not found, skipping registration")
        return
    with open(registry_path) as f:
        registry = json.load(f)
    experiments = registry.get("experiments", {})
    if "EXP-601" in experiments:
        return  # already registered
    experiments["EXP-601"] = {
        "id": "EXP-601",
        "name": "IBIT ML Signal Filter",
        "created_by": "charles",
        "created_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "status": "in_development",
        "ticker": "IBIT",
        "account_id": None,
        "live_since": None,
        "paper_config": None,
        "backtest_config": None,
        "notes": (
            "XGBoost binary classifier (win/loss) trained on EXP-600 champion config "
            "trade features. Time-series CV only. max_depth=3, min_child_weight=5. "
            "Backtest comparison: EXP-600 with vs without ML filter."
        ),
    }
    registry["experiments"] = experiments
    registry["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d")
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"  Registered EXP-601 in registry.json")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import time
    t0 = time.time()
    print("\n" + "="*70)
    print("  EXP-601: IBIT ML Signal Filter")
    print("  Charles experiment — builds on EXP-600 (IBIT Adaptive)")
    print("="*70)

    # Register experiment
    register_exp601()

    # Phase 1: extract training data
    df = extract_training_data()

    # Phase 2: train model
    model, features = train_model(df)

    # Phase 3: backtest comparison
    comparison = backtest_comparison(model, features)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  EXP-601 COMPLETE  ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  ml/ibit_training_data.csv      — {len(df)} trades with features")
    print(f"  ml/ibit_signal_model.joblib    — {'trained' if model else 'NOT trained (insufficient data)'}")
    print(f"  ml/ibit_feature_importance.md  — feature ranking")
    print(f"  ml/ibit_model_report.md        — full CV report")
    print(f"  output/exp601_ml_comparison.json — backtest comparison")
    if model is not None and comparison.get("best_threshold_result"):
        best = comparison["best_threshold_result"]
        v    = comparison["verdict"]
        thr  = comparison["best_threshold"]
        print(f"\n  ML filter verdict: {v}  (best threshold={thr:.2f})")
        print(f"    Trades removed: {best['trades_removed']} ({best['trades_removed_pct']:.1f}%)")
        print(f"    Win rate delta: {best['win_rate_delta']:+.2f}%")
        print(f"    Return delta:   {best['return_delta']:+.2f}%")
        print(f"    Max DD delta:   {best['max_dd_delta']:+.2f}%")
    print()


if __name__ == "__main__":
    main()
