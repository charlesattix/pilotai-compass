"""
Intraday feature engineering for signal enhancement.

Computes 12 features from 1-minute bar data and quote snapshots at
signal-generation time.  Designed to augment the daily feature set
used by the production ensemble (EXP-860).

Feature groups:
  - Price-based (4): VWAP deviation, intraday return/range, distance from high
  - Volume-based (3): relative volume, acceleration, buy/sell imbalance
  - Microstructure (3): bid-ask spread, spread vs avg, quote imbalance
  - Momentum confirmation (2): 5-min momentum, alignment with daily

Usage::

    from compass.intraday_features import IntradayFeatureEngine
    engine = IntradayFeatureEngine()
    features = engine.compute(bars_1min, quote_snapshot, daily_context)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "vwap_deviation_pct",
    "intraday_return_pct",
    "intraday_range_pct",
    "distance_from_high_pct",
    "relative_volume",
    "volume_acceleration",
    "buy_sell_imbalance",
    "bid_ask_spread_bps",
    "spread_vs_avg",
    "quote_imbalance",
    "momentum_5min",
    "momentum_alignment",
]


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class BarData:
    """Single 1-minute bar."""

    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class QuoteSnapshot:
    """Level-1 quote at signal time."""

    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float


@dataclass
class DailyContext:
    """Daily-level context for alignment features."""

    daily_open: float
    daily_momentum_5d: float = 0.0
    avg_daily_volume: float = 1_000_000.0
    avg_spread_20d_bps: float = 3.0


@dataclass
class IntradayFeatures:
    """All 12 intraday features."""

    vwap_deviation_pct: float = 0.0
    intraday_return_pct: float = 0.0
    intraday_range_pct: float = 0.0
    distance_from_high_pct: float = 0.0
    relative_volume: float = 1.0
    volume_acceleration: float = 0.0
    buy_sell_imbalance: float = 0.0
    bid_ask_spread_bps: float = 0.0
    spread_vs_avg: float = 1.0
    quote_imbalance: float = 0.0
    momentum_5min: float = 0.0
    momentum_alignment: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in FEATURE_NAMES}

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in FEATURE_NAMES])


# ── Individual feature computations ──────────────────────────────────────


def compute_vwap(bars: List[BarData]) -> float:
    """Volume-weighted average price from bars."""
    if not bars:
        return 0.0
    total_pv = sum(b.close * b.volume for b in bars)
    total_v = sum(b.volume for b in bars)
    return total_pv / total_v if total_v > 0 else bars[-1].close


def compute_vwap_deviation(current_price: float, vwap: float) -> float:
    """(price - VWAP) / VWAP × 100."""
    if vwap <= 0:
        return 0.0
    return (current_price - vwap) / vwap * 100


def compute_intraday_return(current_price: float, daily_open: float) -> float:
    """Return from open to now in percent."""
    if daily_open <= 0:
        return 0.0
    return (current_price - daily_open) / daily_open * 100


def compute_intraday_range(bars: List[BarData], daily_open: float) -> float:
    """(intraday high - low) / open × 100."""
    if not bars or daily_open <= 0:
        return 0.0
    high = max(b.high for b in bars)
    low = min(b.low for b in bars)
    return (high - low) / daily_open * 100


def compute_distance_from_high(current_price: float, bars: List[BarData]) -> float:
    """How far below the day's high in percent."""
    if not bars:
        return 0.0
    intraday_high = max(b.high for b in bars)
    if intraday_high <= 0:
        return 0.0
    return (intraday_high - current_price) / intraday_high * 100


def compute_relative_volume(
    bars: List[BarData],
    avg_daily_volume: float,
    expected_fraction: float = 0.5,
) -> float:
    """Current cumulative volume / expected volume at this time of day."""
    if not bars or avg_daily_volume <= 0:
        return 1.0
    current_vol = sum(b.volume for b in bars)
    expected = avg_daily_volume * expected_fraction
    return current_vol / expected if expected > 0 else 1.0


def compute_volume_acceleration(bars: List[BarData], lookback: int = 15) -> float:
    """Rate of change in volume over last N bars."""
    if len(bars) < lookback + 5:
        return 0.0
    recent = sum(b.volume for b in bars[-lookback:])
    prior = sum(b.volume for b in bars[-lookback * 2:-lookback])
    if prior <= 0:
        return 0.0
    return (recent - prior) / prior


def compute_buy_sell_imbalance(bars: List[BarData]) -> float:
    """Estimate buy/sell imbalance from bar direction.

    Uptick bar (close > open) → buy volume, downtick → sell volume.
    Returns ratio in [-1, 1].
    """
    if not bars:
        return 0.0
    buy_vol = sum(b.volume for b in bars if b.close >= b.open)
    sell_vol = sum(b.volume for b in bars if b.close < b.open)
    total = buy_vol + sell_vol
    if total <= 0:
        return 0.0
    return (buy_vol - sell_vol) / total


def compute_bid_ask_spread_bps(quote: QuoteSnapshot) -> float:
    """Bid-ask spread in basis points."""
    mid = (quote.bid + quote.ask) / 2
    if mid <= 0:
        return 0.0
    return (quote.ask - quote.bid) / mid * 10_000


def compute_spread_vs_avg(current_spread_bps: float, avg_spread_bps: float) -> float:
    """Current spread / historical average spread."""
    if avg_spread_bps <= 0:
        return 1.0
    return current_spread_bps / avg_spread_bps


def compute_quote_imbalance(quote: QuoteSnapshot) -> float:
    """(bid_size - ask_size) / (bid_size + ask_size).  Range [-1, 1]."""
    total = quote.bid_size + quote.ask_size
    if total <= 0:
        return 0.0
    return (quote.bid_size - quote.ask_size) / total


def compute_momentum_5min(bars: List[BarData], lookback: int = 5) -> float:
    """5-bar (5-min) return at signal time in percent."""
    if len(bars) < lookback + 1:
        return 0.0
    price_now = bars[-1].close
    price_ago = bars[-lookback - 1].close
    if price_ago <= 0:
        return 0.0
    return (price_now - price_ago) / price_ago * 100


def compute_momentum_alignment(
    intraday_return: float,
    daily_momentum: float,
) -> float:
    """1.0 if intraday direction agrees with daily momentum, 0.0 otherwise."""
    if abs(intraday_return) < 0.01 or abs(daily_momentum) < 0.01:
        return 0.5  # neutral
    return 1.0 if (intraday_return > 0) == (daily_momentum > 0) else 0.0


# ── Feature quality scoring ──────────────────────────────────────────────


def score_entry_quality(features: IntradayFeatures) -> float:
    """Score overall entry timing quality from intraday features.

    Returns 0-1 composite score.  Higher = better entry.
    """
    scores = []

    # VWAP: buying below VWAP is favorable for puts
    vwap_score = 0.5 + min(max(features.vwap_deviation_pct, -2), 2) / 4
    scores.append(vwap_score)

    # Volume: higher relative volume = more conviction
    vol_score = min(features.relative_volume / 1.5, 1.0)
    scores.append(vol_score)

    # Spread: tighter = better execution
    spread_score = max(0, 1.0 - features.spread_vs_avg * 0.3)
    scores.append(spread_score)

    # Momentum alignment: confirming = better
    scores.append(features.momentum_alignment)

    # Imbalance: positive buy imbalance = bullish (good for put credit spreads)
    imb_score = 0.5 + features.buy_sell_imbalance * 0.5
    scores.append(imb_score)

    return float(np.mean(scores))


# ── Core engine ──────────────────────────────────────────────────────────


class IntradayFeatureEngine:
    """Computes all 12 intraday features from market data."""

    def __init__(self, volume_lookback: int = 15, momentum_lookback: int = 5):
        self.volume_lookback = volume_lookback
        self.momentum_lookback = momentum_lookback

    def compute(
        self,
        bars: List[BarData],
        quote: QuoteSnapshot,
        context: DailyContext,
    ) -> IntradayFeatures:
        """Compute all 12 features from current market state."""
        if not bars:
            return IntradayFeatures()

        current_price = quote.last_price if quote.last_price > 0 else bars[-1].close
        vwap = compute_vwap(bars)
        intraday_ret = compute_intraday_return(current_price, context.daily_open)
        spread_bps = compute_bid_ask_spread_bps(quote)

        return IntradayFeatures(
            vwap_deviation_pct=compute_vwap_deviation(current_price, vwap),
            intraday_return_pct=intraday_ret,
            intraday_range_pct=compute_intraday_range(bars, context.daily_open),
            distance_from_high_pct=compute_distance_from_high(current_price, bars),
            relative_volume=compute_relative_volume(bars, context.avg_daily_volume),
            volume_acceleration=compute_volume_acceleration(bars, self.volume_lookback),
            buy_sell_imbalance=compute_buy_sell_imbalance(bars),
            bid_ask_spread_bps=spread_bps,
            spread_vs_avg=compute_spread_vs_avg(spread_bps, context.avg_spread_20d_bps),
            quote_imbalance=compute_quote_imbalance(quote),
            momentum_5min=compute_momentum_5min(bars, self.momentum_lookback),
            momentum_alignment=compute_momentum_alignment(
                intraday_ret, context.daily_momentum_5d,
            ),
        )

    def compute_from_dataframe(
        self,
        bars_df: pd.DataFrame,
        quote: QuoteSnapshot,
        context: DailyContext,
    ) -> IntradayFeatures:
        """Compute features from a DataFrame of 1-min bars.

        Expected columns: timestamp, open, high, low, close, volume.
        """
        bars = [
            BarData(
                timestamp=row.get("timestamp"),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for _, row in bars_df.iterrows()
        ]
        return self.compute(bars, quote, context)
