"""
Order flow imbalance alpha — daily OFI from OHLCV proxy.

Close-location value (CLV) measures buy/sell pressure from OHLCV data.
Accumulation/distribution line, tick imbalance bars, and signal
generation (contrarian at extremes, trend-following at moderate levels).
Standalone backtest + EXP-880 filter overlay.

Usage::

    from compass.order_flow_alpha import OrderFlowAlpha
    ofa = OrderFlowAlpha(ohlcv_df)
    signals = ofa.compute_signals()
    bt = ofa.backtest()
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger_name = __name__


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class OFISnapshot:
    """Order flow imbalance at one point in time."""
    date: str
    clv: float                # close location value [-1, 1]
    ad_line: float            # accumulation/distribution line
    ofi: float                # volume-weighted OFI
    ofi_z: float              # z-score of OFI over lookback
    cum_delta: float          # cumulative delta proxy
    signal: str               # "buy", "sell", "neutral"
    signal_type: str           # "contrarian" or "trend"
    strength: float            # 0-1


@dataclass
class TickImbalanceBar:
    """One tick-imbalance bar (volume-clock)."""
    start_idx: int
    end_idx: int
    n_bars: int
    total_volume: float
    buy_volume: float
    sell_volume: float
    imbalance: float           # buy_vol - sell_vol
    vwap: float
    direction: str             # "buy_dominated" or "sell_dominated"


@dataclass
class FlowSignal:
    """Trading signal from order flow analysis."""
    date: str
    direction: int             # +1 buy, -1 sell, 0 neutral
    confidence: float          # 0-1
    signal_type: str           # "contrarian" or "trend"
    ofi_z: float
    cum_delta_z: float
    regime_compatible: bool    # True if compatible with market regime


@dataclass
class BacktestResult:
    """Backtest result for OFI strategy."""
    strategy: str
    n_signals: int
    n_correct: int
    accuracy: float
    sharpe: float
    total_pnl: float
    max_dd: float
    win_rate: float
    avg_return: float
    # Per signal type
    contrarian_accuracy: float
    trend_accuracy: float
    contrarian_n: int
    trend_n: int


@dataclass
class FilterResult:
    """Result of using OFI as filter for another strategy."""
    base_sharpe: float
    filtered_sharpe: float
    base_pnl: float
    filtered_pnl: float
    base_trades: int
    filtered_trades: int
    improvement_pct: float
    correlation_with_base: float


# ── Core computations ───────────────────────────────────────────────────


def close_location_value(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
) -> np.ndarray:
    """CLV = (close - low) / (high - low), mapped to [-1, 1].

    CLV > 0 → close near high (buy pressure)
    CLV < 0 → close near low (sell pressure)
    """
    rng = high - low
    rng = np.where(rng > 0, rng, 1.0)  # avoid div/0
    clv = 2.0 * (close - low) / rng - 1.0
    return np.clip(clv, -1.0, 1.0)


def accumulation_distribution(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Accumulation/Distribution line = cumsum(CLV × volume)."""
    clv = close_location_value(high, low, close)
    ad = np.cumsum(clv * volume)
    return ad


def volume_weighted_ofi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    volume: np.ndarray, lookback: int = 20,
) -> np.ndarray:
    """Volume-weighted OFI: rolling mean of CLV × volume."""
    clv = close_location_value(high, low, close)
    raw = clv * volume
    # Rolling mean
    ofi = np.full_like(raw, np.nan)
    for i in range(lookback - 1, len(raw)):
        ofi[i] = np.mean(raw[i - lookback + 1: i + 1])
    return ofi


def cumulative_delta(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Cumulative delta proxy: cumsum of signed volume.

    Buy volume = volume × (close - low) / (high - low)
    Sell volume = volume × (high - close) / (high - low)
    Delta = buy - sell = volume × CLV
    """
    clv = close_location_value(high, low, close)
    delta = clv * volume
    return np.cumsum(delta)


def compute_tick_imbalance_bars(
    close: np.ndarray, volume: np.ndarray,
    expected_ticks: int = 20,
) -> List[TickImbalanceBar]:
    """Compute tick-imbalance bars (volume-clock regime shifts).

    Groups bars until cumulative signed volume exceeds threshold.
    """
    n = len(close)
    if n < expected_ticks:
        return []

    # Estimate expected imbalance
    price_changes = np.diff(close)
    signed_vol = np.sign(price_changes) * volume[1:]
    avg_abs_imbalance = np.mean(np.abs(signed_vol)) * expected_ticks

    if avg_abs_imbalance < 1:
        avg_abs_imbalance = np.mean(volume) * expected_ticks * 0.1

    bars: List[TickImbalanceBar] = []
    start = 0
    cum_imb = 0.0
    total_vol = 0.0
    buy_vol = 0.0
    sell_vol = 0.0

    for i in range(1, n):
        tick_sign = 1.0 if close[i] >= close[i - 1] else -1.0
        v = volume[i]
        total_vol += v
        if tick_sign > 0:
            buy_vol += v
        else:
            sell_vol += v
        cum_imb += tick_sign * v

        if abs(cum_imb) >= avg_abs_imbalance or i == n - 1:
            vwap = np.mean(close[start:i + 1]) if i > start else close[i]
            bars.append(TickImbalanceBar(
                start_idx=start, end_idx=i,
                n_bars=i - start + 1,
                total_volume=total_vol,
                buy_volume=buy_vol, sell_volume=sell_vol,
                imbalance=cum_imb, vwap=float(vwap),
                direction="buy_dominated" if cum_imb > 0 else "sell_dominated",
            ))
            start = i + 1
            cum_imb = 0.0
            total_vol = 0.0
            buy_vol = 0.0
            sell_vol = 0.0

    return bars


# ── Signal generation ───────────────────────────────────────────────────


def generate_signals(
    ofi: np.ndarray,
    cum_delta: np.ndarray,
    lookback: int = 60,
    contrarian_z: float = 2.0,
    trend_z_min: float = 0.5,
    trend_z_max: float = 2.0,
) -> np.ndarray:
    """Generate directional signals from OFI z-scores.

    Returns array of +1 (buy), -1 (sell), 0 (neutral).
    - |z| > contrarian_z: contrarian (fade the extreme)
    - trend_z_min < |z| < trend_z_max: trend (follow the flow)
    """
    n = len(ofi)
    signals = np.zeros(n, dtype=int)
    signal_types = np.full(n, "", dtype=object)

    for i in range(lookback, n):
        window = ofi[max(0, i - lookback):i]
        valid = window[~np.isnan(window)]
        if len(valid) < 10:
            continue
        mean = np.mean(valid)
        std = np.std(valid)
        if std < 1e-10:
            continue
        z = (ofi[i] - mean) / std if not np.isnan(ofi[i]) else 0

        if z > contrarian_z:
            signals[i] = -1  # contrarian sell (overbought flow)
            signal_types[i] = "contrarian"
        elif z < -contrarian_z:
            signals[i] = +1  # contrarian buy (oversold flow)
            signal_types[i] = "contrarian"
        elif trend_z_min < z < trend_z_max:
            signals[i] = +1  # trend buy
            signal_types[i] = "trend"
        elif -trend_z_max < z < -trend_z_min:
            signals[i] = -1  # trend sell
            signal_types[i] = "trend"

    return signals


# ── Order flow alpha engine ─────────────────────────────────────────────


class OrderFlowAlpha:
    """Order flow imbalance alpha engine."""

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        lookback: int = 20,
        signal_lookback: int = 60,
        contrarian_z: float = 2.0,
        trend_z_min: float = 0.5,
        trend_z_max: float = 2.0,
        tick_imbalance_window: int = 20,
    ) -> None:
        self.ohlcv = ohlcv.copy()
        self.lookback = lookback
        self.signal_lookback = signal_lookback
        self.contrarian_z = contrarian_z
        self.trend_z_min = trend_z_min
        self.trend_z_max = trend_z_max
        self.tick_imbalance_window = tick_imbalance_window

        # Extract arrays
        self.high = ohlcv["high"].values.astype(float)
        self.low = ohlcv["low"].values.astype(float)
        self.close = ohlcv["close"].values.astype(float)
        self.volume = ohlcv["volume"].values.astype(float)
        self.n = len(ohlcv)

        # Computed
        self.clv: Optional[np.ndarray] = None
        self.ad: Optional[np.ndarray] = None
        self.ofi: Optional[np.ndarray] = None
        self.cum_delta_arr: Optional[np.ndarray] = None
        self.signals: Optional[np.ndarray] = None
        self.tick_bars: Optional[List[TickImbalanceBar]] = None
        self.snapshots: List[OFISnapshot] = []

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "OrderFlowAlpha":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls(df, **kwargs)

    def compute_signals(self) -> List[OFISnapshot]:
        """Run full OFI computation and signal generation."""
        self.clv = close_location_value(self.high, self.low, self.close)
        self.ad = accumulation_distribution(self.high, self.low, self.close, self.volume)
        self.ofi = volume_weighted_ofi(self.high, self.low, self.close, self.volume, self.lookback)
        self.cum_delta_arr = cumulative_delta(self.high, self.low, self.close, self.volume)
        self.signals = generate_signals(
            self.ofi, self.cum_delta_arr,
            self.signal_lookback, self.contrarian_z,
            self.trend_z_min, self.trend_z_max,
        )
        self.tick_bars = compute_tick_imbalance_bars(
            self.close, self.volume, self.tick_imbalance_window,
        )

        # Build snapshots
        self.snapshots = []
        for i in range(self.n):
            ofi_val = self.ofi[i] if not np.isnan(self.ofi[i]) else 0
            # z-score
            window = self.ofi[max(0, i - self.signal_lookback):i]
            valid = window[~np.isnan(window)]
            if len(valid) > 5:
                z = (ofi_val - np.mean(valid)) / max(np.std(valid), 1e-10)
            else:
                z = 0.0

            sig = self.signals[i]
            if sig == 0:
                signal_str = "neutral"
                sig_type = ""
                strength = 0.0
            else:
                signal_str = "buy" if sig > 0 else "sell"
                sig_type = "contrarian" if abs(z) > self.contrarian_z else "trend"
                strength = min(abs(z) / 3.0, 1.0)

            date_str = str(self.ohlcv.index[i]) if hasattr(self.ohlcv.index, '__getitem__') else str(i)
            self.snapshots.append(OFISnapshot(
                date=date_str, clv=float(self.clv[i]),
                ad_line=float(self.ad[i]), ofi=float(ofi_val),
                ofi_z=float(z), cum_delta=float(self.cum_delta_arr[i]),
                signal=signal_str, signal_type=sig_type,
                strength=strength,
            ))

        return self.snapshots

    def backtest(self) -> BacktestResult:
        """Backtest: OFI signal predicts next-day return."""
        if self.signals is None:
            self.compute_signals()

        # Next-day returns
        returns = np.diff(self.close) / self.close[:-1]
        signals = self.signals[:-1]  # align: signal[i] predicts return[i]

        n = len(returns)
        if n < 10:
            return BacktestResult("ofi_standalone", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Filter to non-zero signals
        mask = signals != 0
        sig_returns = returns[mask]
        sig_signals = signals[mask]

        if len(sig_returns) == 0:
            return BacktestResult("ofi_standalone", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Strategy returns: signal × next-day return
        strat_returns = sig_signals * sig_returns

        # Accuracy: did signal predict direction?
        correct = np.sum((sig_signals > 0) == (sig_returns > 0))
        accuracy = correct / len(sig_returns)

        # Sharpe
        sh = float(np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252)) if np.std(strat_returns) > 0 else 0

        # PnL (notional $100K)
        cap = 100_000
        total_pnl = float(np.sum(strat_returns) * cap)

        # Equity and DD
        equity = cap * (1 + np.cumsum(strat_returns))
        equity_full = np.concatenate([[cap], equity])
        peak = np.maximum.accumulate(equity_full)
        dd = float(np.min((equity_full - peak) / np.where(peak > 0, peak, 1)))

        wr = float((strat_returns > 0).mean())
        avg_ret = float(np.mean(strat_returns))

        # Per signal type
        contrarian_mask = np.array([
            abs(self._ofi_z_at(i)) > self.contrarian_z
            for i in np.where(mask)[0]
        ])
        trend_mask = ~contrarian_mask

        c_correct = np.sum(((sig_signals[contrarian_mask] > 0) == (sig_returns[contrarian_mask] > 0))) if contrarian_mask.any() else 0
        t_correct = np.sum(((sig_signals[trend_mask] > 0) == (sig_returns[trend_mask] > 0))) if trend_mask.any() else 0
        c_n = int(contrarian_mask.sum())
        t_n = int(trend_mask.sum())

        return BacktestResult(
            strategy="ofi_standalone",
            n_signals=len(sig_returns),
            n_correct=int(correct),
            accuracy=accuracy,
            sharpe=sh,
            total_pnl=total_pnl,
            max_dd=dd,
            win_rate=wr,
            avg_return=avg_ret,
            contrarian_accuracy=c_correct / c_n if c_n > 0 else 0,
            trend_accuracy=t_correct / t_n if t_n > 0 else 0,
            contrarian_n=c_n,
            trend_n=t_n,
        )

    def filter_strategy(
        self,
        trade_pnls: np.ndarray,
        trade_dates_idx: np.ndarray,
    ) -> FilterResult:
        """Use OFI as filter for another strategy (e.g., EXP-880).

        Only take trades where OFI signal is non-negative (not bearish flow).
        """
        if self.signals is None:
            self.compute_signals()

        base_pnl = float(trade_pnls.sum())
        base_n = len(trade_pnls)

        # Filter: keep trades where OFI signal >= 0 (neutral or buy)
        keep = np.array([
            self.signals[min(int(idx), self.n - 1)] >= 0
            for idx in trade_dates_idx
        ])

        filtered_pnls = trade_pnls[keep]
        filtered_pnl = float(filtered_pnls.sum())
        filtered_n = int(keep.sum())

        cap = 100_000
        base_rets = trade_pnls / cap
        filt_rets = filtered_pnls / cap

        base_sh = float(np.mean(base_rets) / np.std(base_rets) * np.sqrt(252)) if np.std(base_rets) > 0 else 0
        filt_sh = float(np.mean(filt_rets) / np.std(filt_rets) * np.sqrt(252)) if len(filt_rets) > 1 and np.std(filt_rets) > 0 else 0

        improvement = (filtered_pnl - base_pnl) / abs(base_pnl) * 100 if abs(base_pnl) > 0 else 0

        # Correlation between OFI signal and trade PnL
        ofi_at_trades = np.array([
            float(self.signals[min(int(idx), self.n - 1)])
            for idx in trade_dates_idx
        ])
        if np.std(ofi_at_trades) > 0 and np.std(trade_pnls) > 0:
            corr = float(np.corrcoef(ofi_at_trades, trade_pnls)[0, 1])
        else:
            corr = 0.0

        return FilterResult(
            base_sharpe=base_sh, filtered_sharpe=filt_sh,
            base_pnl=base_pnl, filtered_pnl=filtered_pnl,
            base_trades=base_n, filtered_trades=filtered_n,
            improvement_pct=improvement,
            correlation_with_base=corr,
        )

    def _ofi_z_at(self, idx: int) -> float:
        """Get OFI z-score at index."""
        if self.ofi is None or idx >= self.n:
            return 0
        window = self.ofi[max(0, idx - self.signal_lookback):idx]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            return 0
        val = self.ofi[idx]
        if np.isnan(val):
            return 0
        return float((val - np.mean(valid)) / max(np.std(valid), 1e-10))

    def get_current_state(self) -> Optional[OFISnapshot]:
        """Get the most recent OFI snapshot."""
        if self.snapshots:
            return self.snapshots[-1]
        return None
