"""
Intraday momentum feature engineering and scalping signal generation.

Computes 12 momentum-specific features from 1-minute bars for detecting
short-duration (5-15 min) momentum bursts in SPY. Designed for 0-DTE
directional option purchases (EXP-1030).

Differs from intraday_features.py (EXP-1010) which targets credit spread
entry quality. This module targets momentum *continuation* signals.

Usage::

    from compass.intraday_momentum import MomentumScalper, ScalperConfig
    scalper = MomentumScalper(config)
    signal = scalper.evaluate(bars_1min)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "tick_momentum_5m",
    "tick_momentum_15m",
    "vwap_slope",
    "volume_surge",
    "order_flow_imbalance",
    "price_acceleration",
    "bid_ask_pressure",
    "spread_tightening",
    "momentum_consistency",
    "vwap_distance_bps",
    "rsi_5min",
    "tick_velocity",
]


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class Bar:
    """Single 1-minute OHLCV bar."""
    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ScalperConfig:
    """Scalping strategy configuration."""
    momentum_threshold_bps: float = 15.0  # 0.15% = 15 bps
    imbalance_threshold: float = 0.60
    profit_target_pct: float = 30.0
    stop_loss_pct: float = 50.0
    max_hold_bars: int = 15  # 15 minutes
    premium_risk: float = 200.0  # max $200 per trade
    min_volume_surge: float = 1.5
    rsi_overbought: float = 80.0
    rsi_oversold: float = 20.0


@dataclass
class MomentumFeatures:
    """All 12 momentum features."""
    tick_momentum_5m: float = 0.0
    tick_momentum_15m: float = 0.0
    vwap_slope: float = 0.0
    volume_surge: float = 0.0
    order_flow_imbalance: float = 0.0
    price_acceleration: float = 0.0
    bid_ask_pressure: float = 1.0
    spread_tightening: float = 0.0
    momentum_consistency: float = 0.0
    vwap_distance_bps: float = 0.0
    rsi_5min: float = 50.0
    tick_velocity: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {n: getattr(self, n) for n in FEATURE_NAMES}

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in FEATURE_NAMES])


@dataclass
class ScalpSignal:
    """Output signal from momentum evaluation."""
    triggered: bool
    direction: str  # "long", "short", "none"
    strength: float  # 0-1
    features: MomentumFeatures
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_price: float = 0.0


@dataclass
class ScalpTrade:
    """Completed scalp trade result."""
    entry_bar: int
    exit_bar: int
    direction: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_dollars: float
    hold_bars: int
    exit_reason: str  # "profit_target", "stop_loss", "time_stop"
    win: bool


# ── Feature computations ────────────────────────────────────────────────


def compute_tick_momentum(bars: List[Bar], lookback: int = 5) -> float:
    """Price change over lookback bars in basis points."""
    if len(bars) < lookback + 1:
        return 0.0
    p_now = bars[-1].close
    p_ago = bars[-lookback - 1].close
    if p_ago <= 0:
        return 0.0
    return (p_now - p_ago) / p_ago * 10_000


def compute_vwap_and_slope(bars: List[Bar], slope_lookback: int = 10) -> Tuple[float, float]:
    """Running VWAP and its slope (rate of change)."""
    if not bars:
        return 0.0, 0.0
    cum_pv = sum(b.close * b.volume for b in bars)
    cum_v = sum(b.volume for b in bars)
    vwap = cum_pv / cum_v if cum_v > 0 else bars[-1].close

    # Slope: VWAP change over last slope_lookback bars
    if len(bars) < slope_lookback + 1:
        return vwap, 0.0

    bars_early = bars[:-slope_lookback]
    if not bars_early:
        return vwap, 0.0
    cum_pv_early = sum(b.close * b.volume for b in bars_early)
    cum_v_early = sum(b.volume for b in bars_early)
    vwap_early = cum_pv_early / cum_v_early if cum_v_early > 0 else bars_early[-1].close

    slope = (vwap - vwap_early) / max(vwap_early, 0.01) * 10_000  # bps per lookback
    return vwap, slope


def compute_volume_surge(bars: List[Bar], recent: int = 5, baseline: int = 20) -> float:
    """Recent volume / baseline average volume."""
    if len(bars) < baseline:
        return 1.0
    recent_vol = sum(b.volume for b in bars[-recent:]) / max(recent, 1)
    baseline_vol = sum(b.volume for b in bars[-baseline:]) / max(baseline, 1)
    if baseline_vol <= 0:
        return 1.0
    return recent_vol / baseline_vol


def compute_order_flow_imbalance(bars: List[Bar], lookback: int = 5) -> float:
    """(uptick volume - downtick volume) / total volume over lookback."""
    if len(bars) < lookback:
        return 0.0
    recent = bars[-lookback:]
    buy = sum(b.volume for b in recent if b.close >= b.open)
    sell = sum(b.volume for b in recent if b.close < b.open)
    total = buy + sell
    if total <= 0:
        return 0.0
    return (buy - sell) / total


def compute_price_acceleration(bars: List[Bar], lookback: int = 5) -> float:
    """Second derivative of price: is momentum increasing or fading?"""
    if len(bars) < lookback * 2 + 1:
        return 0.0
    # Momentum now vs momentum lookback bars ago
    p = [b.close for b in bars]
    mom_now = p[-1] - p[-lookback - 1]
    mom_ago = p[-lookback - 1] - p[-2 * lookback - 1]
    base = abs(p[-1]) if p[-1] > 0 else 1.0
    return (mom_now - mom_ago) / base * 10_000


def compute_momentum_consistency(bars: List[Bar], lookback: int = 5) -> float:
    """Fraction of last N bars moving in the same direction as overall momentum."""
    if len(bars) < lookback + 1:
        return 0.0
    overall = bars[-1].close - bars[-lookback - 1].close
    if abs(overall) < 1e-10:
        return 0.5
    same_dir = sum(
        1 for i in range(-lookback, 0)
        if (bars[i].close - bars[i].open > 0) == (overall > 0)
    )
    return same_dir / lookback


def compute_rsi(bars: List[Bar], period: int = 5) -> float:
    """Fast RSI over period bars."""
    if len(bars) < period + 1:
        return 50.0
    changes = [bars[i].close - bars[i - 1].close for i in range(-period, 0)]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0
    if avg_loss <= 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def compute_tick_velocity(bars: List[Bar], lookback: int = 3) -> float:
    """Rate of price change acceleration (bps/bar)."""
    if len(bars) < lookback + 2:
        return 0.0
    changes = [(bars[i].close - bars[i - 1].close) for i in range(-lookback, 0)]
    if len(changes) < 2:
        return 0.0
    accel = [changes[i] - changes[i - 1] for i in range(1, len(changes))]
    base = abs(bars[-1].close) if bars[-1].close > 0 else 1.0
    return float(np.mean(accel)) / base * 10_000


# ── Feature engine ───────────────────────────────────────────────────────


def compute_all_features(bars: List[Bar]) -> MomentumFeatures:
    """Compute all 12 momentum features."""
    if len(bars) < 20:
        return MomentumFeatures()

    vwap, vwap_slope = compute_vwap_and_slope(bars)
    current = bars[-1].close
    vwap_dist = (current - vwap) / vwap * 10_000 if vwap > 0 else 0.0

    return MomentumFeatures(
        tick_momentum_5m=compute_tick_momentum(bars, 5),
        tick_momentum_15m=compute_tick_momentum(bars, 15),
        vwap_slope=vwap_slope,
        volume_surge=compute_volume_surge(bars),
        order_flow_imbalance=compute_order_flow_imbalance(bars),
        price_acceleration=compute_price_acceleration(bars),
        bid_ask_pressure=1.0,  # requires L1 data
        spread_tightening=0.0,  # requires L1 data
        momentum_consistency=compute_momentum_consistency(bars),
        vwap_distance_bps=vwap_dist,
        rsi_5min=compute_rsi(bars, 5),
        tick_velocity=compute_tick_velocity(bars),
    )


# ── Signal generation ────────────────────────────────────────────────────


def evaluate_signal(
    features: MomentumFeatures,
    config: ScalperConfig,
    current_price: float,
) -> ScalpSignal:
    """Evaluate whether current features trigger a scalp entry."""
    mom5 = features.tick_momentum_5m
    imbalance = features.order_flow_imbalance
    surge = features.volume_surge
    consistency = features.momentum_consistency
    rsi = features.rsi_5min
    accel = features.price_acceleration

    # Direction from momentum
    if mom5 > config.momentum_threshold_bps:
        direction = "long"
    elif mom5 < -config.momentum_threshold_bps:
        direction = "short"
    else:
        return ScalpSignal(False, "none", 0.0, features)

    # Confirmation checks
    checks_passed = 0
    total_checks = 5

    # 1. Order flow confirms direction
    if (direction == "long" and imbalance > config.imbalance_threshold) or \
       (direction == "short" and imbalance < -config.imbalance_threshold):
        checks_passed += 1

    # 2. Volume surge
    if surge >= config.min_volume_surge:
        checks_passed += 1

    # 3. Momentum consistency > 60%
    if consistency > 0.6:
        checks_passed += 1

    # 4. Not overextended (RSI filter)
    if direction == "long" and rsi < config.rsi_overbought:
        checks_passed += 1
    elif direction == "short" and rsi > config.rsi_oversold:
        checks_passed += 1

    # 5. Price acceleration positive (momentum growing, not fading)
    if (direction == "long" and accel > 0) or (direction == "short" and accel < 0):
        checks_passed += 1

    strength = checks_passed / total_checks
    triggered = checks_passed >= 3  # need 3 of 5 confirmations

    # Price targets
    move = current_price * config.profit_target_pct / 100 / 100
    if direction == "long":
        target = current_price + move
        stop = current_price - move * (config.stop_loss_pct / config.profit_target_pct)
    else:
        target = current_price - move
        stop = current_price + move * (config.stop_loss_pct / config.profit_target_pct)

    return ScalpSignal(
        triggered=triggered, direction=direction, strength=strength,
        features=features, entry_price=current_price,
        target_price=target, stop_price=stop,
    )


# ── Backtest engine ──────────────────────────────────────────────────────


def simulate_scalp_trade(
    bars: List[Bar],
    entry_bar_idx: int,
    direction: str,
    config: ScalperConfig,
) -> Optional[ScalpTrade]:
    """Simulate a single scalp trade from entry to exit."""
    if entry_bar_idx >= len(bars) - 1:
        return None

    entry_price = bars[entry_bar_idx].close
    premium = config.premium_risk
    target_pct = config.profit_target_pct / 100
    stop_pct = config.stop_loss_pct / 100

    for offset in range(1, config.max_hold_bars + 1):
        idx = entry_bar_idx + offset
        if idx >= len(bars):
            break

        bar = bars[idx]
        if direction == "long":
            move_pct = (bar.high - entry_price) / entry_price
            adverse_pct = (entry_price - bar.low) / entry_price
        else:
            move_pct = (entry_price - bar.low) / entry_price
            adverse_pct = (bar.high - entry_price) / entry_price

        # Check profit target (gamma amplifies: 2x for 0-DTE)
        gamma_mult = 2.0
        if move_pct * gamma_mult >= target_pct / 100:
            pnl_pct = target_pct
            pnl_dollars = premium * target_pct
            return ScalpTrade(
                entry_bar=entry_bar_idx, exit_bar=idx, direction=direction,
                entry_price=entry_price, exit_price=bar.close,
                pnl_pct=pnl_pct, pnl_dollars=pnl_dollars,
                hold_bars=offset, exit_reason="profit_target", win=True,
            )

        # Check stop loss
        if adverse_pct >= stop_pct / 100:
            pnl_pct = -stop_pct
            pnl_dollars = -premium * stop_pct
            return ScalpTrade(
                entry_bar=entry_bar_idx, exit_bar=idx, direction=direction,
                entry_price=entry_price, exit_price=bar.close,
                pnl_pct=pnl_pct, pnl_dollars=pnl_dollars,
                hold_bars=offset, exit_reason="stop_loss", win=False,
            )

    # Time stop — close at last bar, small loss from theta
    last_idx = min(entry_bar_idx + config.max_hold_bars, len(bars) - 1)
    last_bar = bars[last_idx]
    if direction == "long":
        final_move = (last_bar.close - entry_price) / entry_price
    else:
        final_move = (entry_price - last_bar.close) / entry_price

    # Time stop: theta decay eats ~20% of premium + directional
    theta_drag = -0.20
    pnl_pct = (final_move * 200 + theta_drag) * 100  # gamma mult
    pnl_dollars = premium * pnl_pct / 100
    return ScalpTrade(
        entry_bar=entry_bar_idx, exit_bar=last_idx, direction=direction,
        entry_price=entry_price, exit_price=last_bar.close,
        pnl_pct=pnl_pct, pnl_dollars=pnl_dollars,
        hold_bars=last_idx - entry_bar_idx, exit_reason="time_stop",
        win=pnl_dollars > 0,
    )


# ── Core scalper ─────────────────────────────────────────────────────────


class MomentumScalper:
    """Intraday momentum scalping engine."""

    def __init__(self, config: Optional[ScalperConfig] = None):
        self.config = config or ScalperConfig()

    def evaluate(self, bars: List[Bar]) -> ScalpSignal:
        """Evaluate current bars for scalp entry."""
        features = compute_all_features(bars)
        price = bars[-1].close if bars else 0.0
        return evaluate_signal(features, self.config, price)

    def backtest(self, bars: List[Bar], cooldown: int = 20) -> List[ScalpTrade]:
        """Backtest over a full day of 1-min bars."""
        trades: List[ScalpTrade] = []
        last_exit = -cooldown

        for i in range(30, len(bars) - self.config.max_hold_bars):
            if i - last_exit < cooldown:
                continue

            signal = self.evaluate(bars[:i + 1])
            if not signal.triggered:
                continue

            trade = simulate_scalp_trade(bars, i, signal.direction, self.config)
            if trade:
                trades.append(trade)
                last_exit = trade.exit_bar

        return trades
