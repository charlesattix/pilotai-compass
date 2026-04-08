"""
Dynamic leverage manager for EXP-1220 Tail Risk Protection.

Scales leverage based on real-time market conditions:
  - VIX level (spot stress)
  - VIX term structure (VIX / VIX3M ratio — inversion = stress)
  - 20-day realized volatility of SPY

Target: ~100% CAGR in normal conditions, cap DD at 12% in COVID-level crashes.

The manager operates on base 1x protected returns from TailRiskProtector
and applies a time-varying leverage multiplier.

All data from Yahoo Finance — no synthetic data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LeverageState:
    """Daily leverage decision."""
    date: object  # datetime-like
    leverage: float
    vix: float
    vix_ratio: float  # VIX / VIX3M
    realized_vol: float  # 20-day annualized
    regime: str  # calm, normal, elevated, crisis


@dataclass
class DynamicLeverageConfig:
    """Configuration for the dynamic leverage manager.

    Thresholds and leverage targets calibrated on 2020-2025 data.
    """
    # Target leverage in calm conditions (VIX < 15, normal term structure)
    target_leverage: float = 1.8

    # Minimum leverage (floor — never go below this)
    min_leverage: float = 0.3

    # VIX-based scaling thresholds
    vix_calm: float = 15.0       # Below this: full target leverage
    vix_normal: float = 20.0     # Below this: moderate reduction
    vix_elevated: float = 28.0   # Below this: significant reduction
    vix_crisis: float = 35.0     # Above this: minimum leverage

    # VIX term structure thresholds (VIX / VIX3M)
    # > 1.0 = inversion = stress; < 0.85 = contango = calm
    ts_contango: float = 0.90    # Healthy contango: boost allowed
    ts_flat: float = 1.0         # Flat: neutral
    ts_inverted: float = 1.10    # Inverted: reduce
    ts_deep_inversion: float = 1.25  # Deep inversion: minimum

    # Realized vol thresholds (annualized)
    rvol_low: float = 0.10       # Low vol: boost allowed
    rvol_normal: float = 0.16    # Normal: neutral
    rvol_high: float = 0.25      # High: reduce
    rvol_extreme: float = 0.40   # Extreme: minimum

    # Smoothing: exponential moving average on leverage changes
    # Prevents whipsaw from daily noise
    smoothing_halflife: int = 5  # days — higher = slower adjustment


# ═══════════════════════════════════════════════════════════════════════════
# Core Engine
# ═══════════════════════════════════════════════════════════════════════════


class DynamicLeverageManager:
    """Computes time-varying leverage from VIX, term structure, and realized vol.

    The leverage is the product of three independent scaling factors:
      leverage = target * vix_scale * ts_scale * rvol_scale

    Each factor maps a market observable to [0, 1] via piecewise linear ramps.
    The result is clamped to [min_leverage, target_leverage] and smoothed.

    Args:
        config: DynamicLeverageConfig with all thresholds.
    """

    def __init__(self, config: Optional[DynamicLeverageConfig] = None):
        self.cfg = config or DynamicLeverageConfig()

    def compute_leverage_series(
        self,
        vix: pd.Series,
        vix3m: pd.Series,
        spy_returns: pd.Series,
    ) -> List[LeverageState]:
        """Compute daily leverage decisions from real market data.

        Args:
            vix: Daily VIX close series.
            vix3m: Daily VIX3M close series.
            spy_returns: Daily SPY returns (for realized vol calculation).

        Returns:
            List of LeverageState, one per day.
        """
        # Align
        common = vix.index.intersection(vix3m.index).intersection(spy_returns.index)
        common = common.sort_values()
        vix = vix.reindex(common).ffill()
        vix3m = vix3m.reindex(common).ffill()
        spy_returns = spy_returns.reindex(common).fillna(0)

        # Realized vol: 20-day rolling std, annualized
        rvol = spy_returns.rolling(20, min_periods=10).std() * math.sqrt(TRADING_DAYS)
        rvol = rvol.fillna(0.15)  # Default to ~15% vol

        # VIX / VIX3M ratio
        vix_ratio = vix / vix3m.replace(0, 1)

        cfg = self.cfg
        states = []
        raw_leverage = []

        for dt in common:
            v = float(vix.loc[dt])
            vr = float(vix_ratio.loc[dt])
            rv = float(rvol.loc[dt])

            # 1. VIX scale: piecewise linear ramp
            vix_scale = self._ramp(v, cfg.vix_calm, cfg.vix_crisis)

            # 2. Term structure scale
            ts_scale = self._ramp(vr, cfg.ts_contango, cfg.ts_deep_inversion)

            # 3. Realized vol scale
            rvol_scale = self._ramp(rv, cfg.rvol_low, cfg.rvol_extreme)

            # Combined leverage = target * product of scales
            lev = cfg.target_leverage * vix_scale * ts_scale * rvol_scale
            lev = max(cfg.min_leverage, min(cfg.target_leverage, lev))

            # Classify regime
            if v < cfg.vix_calm and rv < cfg.rvol_normal:
                regime = "calm"
            elif v < cfg.vix_normal:
                regime = "normal"
            elif v < cfg.vix_elevated:
                regime = "elevated"
            else:
                regime = "crisis"

            raw_leverage.append(lev)
            states.append(LeverageState(
                date=dt, leverage=lev, vix=round(v, 1),
                vix_ratio=round(vr, 3), realized_vol=round(rv, 4),
                regime=regime,
            ))

        # Apply exponential smoothing to prevent whipsaw
        if cfg.smoothing_halflife > 0 and len(raw_leverage) > 1:
            alpha = 1 - math.exp(-math.log(2) / cfg.smoothing_halflife)
            smoothed = raw_leverage[0]
            for i in range(len(states)):
                smoothed = alpha * raw_leverage[i] + (1 - alpha) * smoothed
                smoothed = max(cfg.min_leverage, min(cfg.target_leverage, smoothed))
                states[i].leverage = round(smoothed, 4)

        return states

    @staticmethod
    def _ramp(value: float, low: float, high: float) -> float:
        """Piecewise linear ramp: 1.0 at low, 0.0 at high."""
        if value <= low:
            return 1.0
        if value >= high:
            return 0.0
        return 1.0 - (value - low) / (high - low)

    def apply_leverage(
        self,
        base_returns: np.ndarray,
        leverage_states: List[LeverageState],
    ) -> np.ndarray:
        """Apply dynamic leverage to base protected returns.

        Args:
            base_returns: 1x protected daily returns (from TailRiskProtector).
            leverage_states: Daily leverage decisions (same length).

        Returns:
            Leveraged daily returns.
        """
        if len(base_returns) != len(leverage_states):
            raise ValueError(
                f"Length mismatch: {len(base_returns)} returns vs {len(leverage_states)} states"
            )
        result = np.zeros(len(base_returns))
        for i, state in enumerate(leverage_states):
            result[i] = base_returns[i] * state.leverage
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_metrics(rets: np.ndarray) -> dict:
    """Compute full performance metrics from daily returns."""
    if len(rets) < 2:
        return {"cagr_pct": 0, "sharpe": 0, "max_dd_pct": 0, "calmar": 0,
                "sortino": 0, "vol_pct": 0, "total_ret_pct": 0, "n_days": 0}
    eq = np.cumprod(1 + rets)
    total = float(eq[-1] - 1)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1]) ** (1 / max(n_yr, 0.01)) - 1 if eq[-1] > 0 else 0
    mu, std = float(rets.mean()), float(rets.std())
    sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    down = rets[rets < 0]
    down_std = float(down.std()) if len(down) > 1 else std
    sortino = mu / down_std * math.sqrt(TRADING_DAYS) if down_std > 1e-12 else 0
    return {
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(dd * 100, 2),
        "calmar": round(calmar, 2),
        "sortino": round(sortino, 2),
        "vol_pct": round(std * math.sqrt(TRADING_DAYS) * 100, 2),
        "total_ret_pct": round(total * 100, 2),
        "n_days": len(rets),
    }


def yearly_metrics(rets: np.ndarray, dates: list) -> Dict[int, dict]:
    """Year-by-year metrics (skip pre-2020 warmup)."""
    by_yr: Dict[int, list] = {}
    for i, d in enumerate(dates):
        yr = d.year if hasattr(d, 'year') else d.year
        if yr < 2020:
            continue
        by_yr.setdefault(yr, []).append(rets[i])
    return {yr: compute_metrics(np.array(v)) for yr, v in sorted(by_yr.items())}


def regime_metrics(rets: np.ndarray, states: List[LeverageState]) -> Dict[str, dict]:
    """Metrics broken down by leverage regime."""
    buckets: Dict[str, list] = {}
    for i, s in enumerate(states):
        if i < len(rets):
            buckets.setdefault(s.regime, []).append(rets[i])
    return {
        regime: {
            **compute_metrics(np.array(vals)),
            "n_days": len(vals),
            "avg_leverage": round(np.mean([s.leverage for s in states if s.regime == regime]), 2),
        }
        for regime, vals in sorted(buckets.items())
    }
