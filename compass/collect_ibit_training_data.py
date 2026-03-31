"""
IBIT ML Training Data Collection for EXP-601.

Enriches IBIT trade-level backtest results with crypto-native features from
``compass.ibit_features``.  Works with the trade output from IBITBacktester
(backtest/ibit_backtester.py) which stores trades in ``crypto_options_cache.db``.

Unlike the SPY-centric ``collect_training_data.py``, this collector:
  - Uses crypto annualization (365 days)
  - Adds IBIT-ETHA correlation features
  - Adds BTC perpetual funding rate features
  - Adds Fear & Greed Index and composite score components
  - Sources IBIT OHLCV from the crypto options cache DB, not Polygon REST

Output:
    compass/training_data_ibit.csv — Chronological trade-level dataset with
    both trade-structure features (DTE, OTM%, credit) and crypto-native
    features (vol regime, correlations, funding, FGI).

Usage:
    # With pre-collected trade data in a DataFrame:
    from compass.collect_ibit_training_data import enrich_ibit_trades
    enriched = enrich_ibit_trades(trades_df, ibit_prices, etha_prices, ...)

    # Standalone (reads from backtest DB):
    python3 -m compass.collect_ibit_training_data
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.ibit_features import IBITFeatureEngine

logger = logging.getLogger(__name__)


# ── Feature columns output by this collector ─────────────────────────────

# Trade-structure features (from the backtester trade object)
TRADE_STRUCTURE_COLS = [
    "entry_date",
    "exit_date",
    "direction",        # "bull_put" | "bear_call"
    "dte",
    "otm_pct",
    "spread_width",
    "credit_received",
    "credit_pct",       # credit / spread_width * 100
    "contracts",
    "exit_reason",
    "hold_days",
    "pnl",
    "return_pct",
    "win",
]

# Market context at entry (generic)
MARKET_CONTEXT_COLS = [
    "ibit_price",
    "rsi_14",
    "ma50_distance_pct",
    "volume_ratio",       # today vol / 20d avg vol
    "realized_vol_20d",
]


def _compute_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Wilder's RSI from a price list. Returns None if insufficient data."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent = deltas[-period * 3:]  # use extra history for EMA warmup

    gains = [max(0, d) for d in recent]
    losses = [max(0, -d) for d in recent]

    # Wilder's smoothed average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - 100.0 / (1.0 + rs), 2)


def _compute_ma_distance(prices: List[float], period: int) -> Optional[float]:
    """Percent distance from simple moving average."""
    if len(prices) < period:
        return None
    ma = sum(prices[-period:]) / period
    if ma == 0:
        return None
    return round((prices[-1] - ma) / ma * 100, 4)


def _compute_volume_ratio(volumes: List[float], window: int = 20) -> Optional[float]:
    """Current volume / trailing average."""
    if not volumes or len(volumes) < window:
        return None
    avg = sum(volumes[-window:]) / window
    if avg == 0:
        return None
    return round(volumes[-1] / avg, 4)


def _compute_returns_vol(prices: List[float], window: int = 20) -> Optional[float]:
    """Annualized returns-based vol (equity convention: 252 days for IBIT as stock)."""
    if len(prices) < window + 1:
        return None
    rets = [prices[i] / prices[i - 1] - 1 for i in range(len(prices) - window, len(prices))]
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
    return round(math.sqrt(var) * math.sqrt(252) * 100, 4)


def enrich_ibit_trades(
    trades: pd.DataFrame,
    ibit_closes: pd.Series,
    etha_closes: Optional[pd.Series] = None,
    spy_closes: Optional[pd.Series] = None,
    ibit_volumes: Optional[pd.Series] = None,
    ibit_highs: Optional[pd.Series] = None,
    ibit_lows: Optional[pd.Series] = None,
    funding_history: Optional[pd.DataFrame] = None,
    fgi_by_date: Optional[Dict] = None,
    dominance_by_date: Optional[Dict] = None,
) -> pd.DataFrame:
    """Enrich IBIT trade rows with crypto-native ML features.

    Args:
        trades: DataFrame with at least 'entry_date', 'pnl', and trade-structure
                columns (dte, otm_pct, spread_width, credit_received, direction, etc).
        ibit_closes: IBIT daily close prices (DatetimeIndex).
        etha_closes: ETHA daily close prices (DatetimeIndex), or None.
        spy_closes:  SPY daily close prices (DatetimeIndex), or None.
        ibit_volumes: IBIT daily volumes (DatetimeIndex), or None.
        ibit_highs:   IBIT daily highs (DatetimeIndex), or None.
        ibit_lows:    IBIT daily lows (DatetimeIndex), or None.
        funding_history: DataFrame with 'date' and 'funding_rate' columns, or None.
        fgi_by_date:  {date_str: float} Fear & Greed Index values, or None.
        dominance_by_date: {date_str: float} BTC dominance % values, or None.

    Returns:
        DataFrame with original trade columns plus all IBIT feature columns.
    """
    engine = IBITFeatureEngine()
    enriched_rows = []

    for _, trade in trades.iterrows():
        entry_date = pd.Timestamp(trade["entry_date"])

        # Gather price history up to entry date (no lookahead)
        ibit_hist = ibit_closes.loc[ibit_closes.index < entry_date].tolist()

        etha_hist = None
        if etha_closes is not None:
            etha_hist = etha_closes.loc[etha_closes.index < entry_date].tolist()

        spy_hist = None
        if spy_closes is not None:
            spy_hist = spy_closes.loc[spy_closes.index < entry_date].tolist()

        vol_hist = None
        if ibit_volumes is not None:
            vol_hist = ibit_volumes.loc[ibit_volumes.index < entry_date].tolist()

        high_hist = None
        if ibit_highs is not None:
            high_hist = ibit_highs.loc[ibit_highs.index < entry_date].tolist()

        low_hist = None
        if ibit_lows is not None:
            low_hist = ibit_lows.loc[ibit_lows.index < entry_date].tolist()

        # Funding rates up to entry date
        funding_rates_list = None
        if funding_history is not None and not funding_history.empty:
            mask = funding_history["date"] < str(entry_date.date())
            if mask.any():
                funding_rates_list = funding_history.loc[mask, "funding_rate"].tolist()

        # FGI and dominance for entry date (published before market open)
        fgi = None
        if fgi_by_date:
            date_str = str(entry_date.date())
            fgi = fgi_by_date.get(date_str)

        dominance = None
        if dominance_by_date:
            date_str = str(entry_date.date())
            dominance = dominance_by_date.get(date_str)

        # Compute all crypto-native features
        if len(ibit_hist) < 8:
            # Not enough IBIT history; skip crypto features
            crypto_features = {name: None for name in engine.feature_names}
        else:
            crypto_features = engine.compute(
                ibit_prices=ibit_hist,
                etha_prices=etha_hist if etha_hist and len(etha_hist) >= 8 else None,
                spy_prices=spy_hist if spy_hist and len(spy_hist) >= 8 else None,
                ibit_volumes=vol_hist,
                ibit_highs=high_hist,
                ibit_lows=low_hist,
                funding_rates=funding_rates_list,
                fear_greed_index=fgi,
                btc_dominance=dominance,
            )

        # Generic market context features
        rsi = _compute_rsi(ibit_hist) if len(ibit_hist) > 14 else None
        ma50_dist = _compute_ma_distance(ibit_hist, 50) if len(ibit_hist) >= 50 else None
        vol_ratio = _compute_volume_ratio(vol_hist) if vol_hist and len(vol_hist) >= 20 else None
        rv_20d = _compute_returns_vol(ibit_hist) if len(ibit_hist) >= 21 else None
        ibit_price = ibit_hist[-1] if ibit_hist else None

        # Build row: trade structure + market context + crypto features
        row = dict(trade)
        row.update({
            "ibit_price": round(ibit_price, 2) if ibit_price else None,
            "rsi_14": rsi,
            "ma50_distance_pct": ma50_dist,
            "volume_ratio": vol_ratio,
            "realized_vol_20d": rv_20d,
        })
        row.update(crypto_features)

        # Ensure win label
        if "win" not in row:
            row["win"] = 1 if row.get("pnl", 0) > 0 else 0

        enriched_rows.append(row)

    return pd.DataFrame(enriched_rows)
