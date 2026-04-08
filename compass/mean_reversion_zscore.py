"""Mean reversion z-score strategy — Bollinger-style z-score with RSI
divergence confirmation and volume spike filter for credit spread entry timing.

Entry: z < -2 AND bullish RSI divergence AND volume spike → sell put spread
Exit: z crosses above 0 (mean reversion complete)

Provides:
  1. Z-score computation (20d, 50d windows)
  2. RSI divergence detection (price makes new low, RSI doesn't)
  3. Volume spike filter (>2× rolling average)
  4. Composite entry signal combining all three
  5. Backtest: standalone + as timing overlay for EXP-880
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class ZScoreState:
    """Z-score at one point in time."""
    date: str
    price: float
    z20: float              # 20-day z-score
    z50: float              # 50-day z-score
    rsi: float              # 14-period RSI
    rsi_divergence: bool    # price new low but RSI higher
    volume_spike: bool      # volume > 2× average
    volume_ratio: float     # current / avg volume
    composite_signal: float # -1..+1 combined signal
    entry_trigger: bool     # all conditions met


@dataclass
class MeanRevTrade:
    """Single mean reversion trade."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    entry_z: float
    exit_z: float
    hold_days: int
    pnl: float
    return_pct: float
    exit_reason: str       # "mean_reversion", "stop_loss", "max_hold"
    had_divergence: bool
    had_volume_spike: bool


@dataclass
class BacktestResult:
    """Complete backtest output."""
    trades: List[MeanRevTrade] = field(default_factory=list)
    z_history: List[ZScoreState] = field(default_factory=list)
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_hold_days: float = 0.0
    signal_count: int = 0     # how many entry signals fired
    ending_capital: float = 0.0
    generated_at: str = ""


# ── Z-score computation ────────────────────────────────────────────────────
def compute_zscore(prices: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score: (price - SMA) / rolling_std."""
    n = len(prices)
    z = np.full(n, np.nan)
    for i in range(window, n):
        w = prices[i - window:i]
        mu = w.mean()
        std = w.std()
        z[i] = (prices[i] - mu) / std if std > 1e-10 else 0.0
    return z


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Standard RSI calculation."""
    n = len(prices)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    diffs = np.diff(prices)
    gains = np.maximum(diffs, 0)
    losses = -np.minimum(diffs, 0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for i in range(period, len(diffs)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_gain + avg_loss < 1e-12:
            rsi[i + 1] = 50.0
        else:
            rs = avg_gain / max(avg_loss, 1e-10)
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def detect_rsi_divergence(
    prices: np.ndarray, rsi: np.ndarray, lookback: int = 10,
) -> np.ndarray:
    """Bullish divergence: price makes new low but RSI makes higher low."""
    n = len(prices)
    div = np.zeros(n, dtype=bool)
    for i in range(lookback + 1, n):
        window_prices = prices[i - lookback:i + 1]
        window_rsi = rsi[i - lookback:i + 1]

        # Current is near the low of the window
        if prices[i] > window_prices.min() * 1.005:
            continue  # not at a local low

        # Find previous low in the window
        prev_low_idx = np.argmin(window_prices[:-1])
        if prev_low_idx == len(window_prices) - 2:
            continue  # no distinct previous low

        price_lower = prices[i] <= window_prices[prev_low_idx] * 1.002
        rsi_higher = rsi[i] > window_rsi[prev_low_idx] + 1.0

        div[i] = price_lower and rsi_higher
    return div


def detect_volume_spike(
    volume: np.ndarray, window: int = 20, threshold: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect volume spikes > threshold × rolling average.

    Returns (spike_bool, volume_ratio).
    """
    n = len(volume)
    spikes = np.zeros(n, dtype=bool)
    ratios = np.ones(n)
    for i in range(window, n):
        avg = volume[i - window:i].mean()
        if avg > 0:
            ratios[i] = volume[i] / avg
            spikes[i] = ratios[i] >= threshold
    return spikes, ratios


def composite_signal(z20: float, z50: float, rsi: float, divergence: bool, vol_spike: bool) -> float:
    """Combine indicators into -1..+1 signal.

    Negative = oversold (potential put spread entry).
    """
    sig = 0.0
    # Z-score contribution (strongest weight)
    sig += np.clip(z20 / 3.0, -1, 1) * 0.35
    sig += np.clip(z50 / 3.0, -1, 1) * 0.25
    # RSI contribution
    sig += np.clip((rsi - 50) / 50, -1, 1) * 0.20
    # Divergence boost (only when oversold)
    if divergence and z20 < -1:
        sig -= 0.15  # strengthen bearish/oversold signal
    # Volume spike confirmation
    if vol_spike and z20 < -1:
        sig -= 0.05
    return float(np.clip(sig, -1, 1))


# ── Backtest engine ─────────────────────────────────────────────────────────
class MeanReversionBacktest:
    """Backtest mean reversion z-score strategy."""

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        entry_z: float = -2.0,
        exit_z: float = 0.0,
        stop_loss_z: float = -3.5,
        max_hold: int = 20,
        require_divergence: bool = True,
        require_volume_spike: bool = True,
        trade_size_pct: float = 0.05,
        spread_credit: float = 1.50,
        spread_width: float = 5.0,
        slippage_pct: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.starting_capital = starting_capital
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_loss_z
        self.max_hold = max_hold
        self.req_div = require_divergence
        self.req_vol = require_volume_spike
        self.trade_size = trade_size_pct
        self.credit = spread_credit
        self.width = spread_width
        self.slippage = slippage_pct
        self.rng = np.random.RandomState(seed)

    def run(
        self,
        prices: np.ndarray,
        volume: Optional[np.ndarray] = None,
        dates: Optional[List[str]] = None,
    ) -> BacktestResult:
        n = len(prices)
        if n < 60:
            return BacktestResult(generated_at=_now())

        if volume is None:
            volume = np.ones(n) * 1e6

        # Compute indicators
        z20 = compute_zscore(prices, 20)
        z50 = compute_zscore(prices, 50)
        rsi = compute_rsi(prices)
        div = detect_rsi_divergence(prices, rsi)
        vol_spikes, vol_ratios = detect_volume_spike(volume)

        capital = self.starting_capital
        peak = capital
        max_dd = 0.0
        trades: List[MeanRevTrade] = []
        z_history: List[ZScoreState] = []
        daily_pnl: List[float] = []

        in_trade = False
        trade_entry_idx = 0
        trade_entry_z = 0.0
        trade_div = False
        trade_vol = False
        signal_count = 0

        warmup = 50

        for i in range(warmup, n):
            d = dates[i] if dates and i < len(dates) else str(i)
            z20_i = z20[i] if not np.isnan(z20[i]) else 0.0
            z50_i = z50[i] if not np.isnan(z50[i]) else 0.0
            rsi_i = rsi[i]
            div_i = bool(div[i])
            vol_i = bool(vol_spikes[i])

            comp = composite_signal(z20_i, z50_i, rsi_i, div_i, vol_i)

            # Entry trigger
            entry_ok = z20_i <= self.entry_z
            if self.req_div:
                entry_ok = entry_ok and div_i
            if self.req_vol:
                entry_ok = entry_ok and vol_i

            z_history.append(ZScoreState(
                date=d, price=float(prices[i]),
                z20=round(z20_i, 4), z50=round(z50_i, 4),
                rsi=round(rsi_i, 1),
                rsi_divergence=div_i, volume_spike=vol_i,
                volume_ratio=round(float(vol_ratios[i]), 2),
                composite_signal=round(comp, 4),
                entry_trigger=entry_ok and not in_trade,
            ))

            if entry_ok and not in_trade:
                in_trade = True
                trade_entry_idx = i
                trade_entry_z = z20_i
                trade_div = div_i
                trade_vol = vol_i
                signal_count += 1
                continue

            if in_trade:
                hold = i - trade_entry_idx
                should_exit = False
                reason = ""

                if z20_i >= self.exit_z:
                    should_exit = True
                    reason = "mean_reversion"
                elif z20_i <= self.stop_z:
                    should_exit = True
                    reason = "stop_loss"
                elif hold >= self.max_hold:
                    should_exit = True
                    reason = "max_hold"

                if should_exit:
                    # Credit spread P&L: profit if SPY stayed above short strike
                    # Mean reversion entry → expect SPY to bounce → put spread profitable
                    price_change = (prices[i] - prices[trade_entry_idx]) / prices[trade_entry_idx]
                    notional = capital * self.trade_size

                    if reason == "mean_reversion":
                        # Successful reversion → collect most of credit
                        pnl = notional * self.credit / self.width * 0.8
                    elif reason == "stop_loss":
                        # SPY continued lower → spread hit max loss
                        pnl = -notional * (self.width - self.credit) / self.width
                    else:
                        # Max hold → partial theta capture
                        pnl = notional * self.credit / self.width * 0.3

                    # Slippage
                    pnl -= abs(notional) * self.slippage * 2

                    # Noise
                    pnl += self.rng.randn() * notional * 0.005

                    entry_d = dates[trade_entry_idx] if dates and trade_entry_idx < len(dates) else str(trade_entry_idx)

                    trades.append(MeanRevTrade(
                        entry_date=entry_d, exit_date=d,
                        entry_price=round(float(prices[trade_entry_idx]), 2),
                        exit_price=round(float(prices[i]), 2),
                        entry_z=round(trade_entry_z, 4),
                        exit_z=round(z20_i, 4),
                        hold_days=hold,
                        pnl=round(pnl, 2),
                        return_pct=round(pnl / notional * 100, 2),
                        exit_reason=reason,
                        had_divergence=trade_div,
                        had_volume_spike=trade_vol,
                    ))

                    capital += pnl
                    daily_pnl.append(pnl)
                    in_trade = False

                    if capital > peak:
                        peak = capital
                    dd = (peak - capital) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)

        # Metrics
        wins = sum(1 for t in trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in trades)
        total_return = (capital - self.starting_capital) / self.starting_capital * 100
        years = (n - warmup) / TRADING_DAYS
        cagr = ((capital / self.starting_capital) ** (1 / years) - 1) * 100 if years > 0 and capital > 0 else 0

        dr = np.array(daily_pnl) if daily_pnl else np.array([0.0])
        sharpe = float(dr.mean() / dr.std() * np.sqrt(len(dr) / max(years, 0.1))) if dr.std() > 0 else 0

        win_sum = sum(t.pnl for t in trades if t.pnl > 0)
        loss_sum = abs(sum(t.pnl for t in trades if t.pnl < 0))
        pf = win_sum / loss_sum if loss_sum > 0 else 0

        avg_hold = float(np.mean([t.hold_days for t in trades])) if trades else 0

        return BacktestResult(
            trades=trades,
            z_history=z_history,
            total_return_pct=round(total_return, 2),
            cagr_pct=round(cagr, 2),
            sharpe=round(sharpe, 2),
            max_dd_pct=round(max_dd * 100, 2),
            win_rate_pct=round(wins / len(trades) * 100, 1) if trades else 0,
            profit_factor=round(pf, 2),
            total_trades=len(trades),
            avg_hold_days=round(avg_hold, 1),
            signal_count=signal_count,
            ending_capital=round(capital, 2),
            generated_at=_now(),
        )


# ── Synthetic data ──────────────────────────────────────────────────────────
def generate_mean_rev_data(n: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate SPY-like prices with mean-reverting episodes and volume."""
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    dates = [str(d.date()) for d in idx]

    prices = np.zeros(n)
    prices[0] = 320.0
    for i in range(1, n):
        prices[i] = prices[i - 1] * np.exp(rng.randn() * 0.012 + 0.0003)

    # Inject sharp dips that revert (mean reversion episodes)
    for start in [100, 300, 500, 700, 900]:
        if start + 15 < n:
            # Sharp drop
            for j in range(5):
                if start + j < n:
                    prices[start + j] = prices[start + j] * 0.985
            # Recovery
            for j in range(5, 15):
                if start + j < n:
                    prices[start + j] = prices[start + j] * 1.008

    volume = np.abs(rng.randn(n) * 5e5 + 2e6)
    # Volume spikes during dips
    for start in [100, 300, 500, 700, 900]:
        if start + 5 < n:
            volume[start:start + 5] *= 3

    return prices, volume.astype(float), dates


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
