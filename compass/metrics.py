"""
compass/metrics.py — Canonical portfolio metrics functions.

ALL Sharpe calculations in the codebase should use annualized_sharpe()
from this module. Do NOT compute Sharpe inline.

The correct Sharpe ratio formula:
    Sharpe = (mean(daily_returns) - rf_daily) / std(daily_returns) × √252

Common bug found in codebase (commit 1f0888a audit):
    WRONG:  (CAGR - rf_annual) / annualized_vol
    At 100%+ CAGR this inflates Sharpe by 1.4-1.6× because
    geometric return (CAGR) >> arithmetic mean at high return levels.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

TRADING_DAYS = 252
DEFAULT_RF_ANNUAL = 0.045  # 4.5% risk-free rate


def annualized_sharpe(
    daily_returns: np.ndarray,
    rf_annual: float = DEFAULT_RF_ANNUAL,
) -> float:
    """Correct annualized Sharpe ratio from daily returns.

    Formula: (mean(r) - rf/252) / std(r) × √252

    Args:
        daily_returns: Array of daily simple returns (e.g., 0.003 = +0.3%)
        rf_annual: Annual risk-free rate (default 4.5%)

    Returns:
        Annualized Sharpe ratio
    """
    if len(daily_returns) < 2:
        return 0.0
    r = np.asarray(daily_returns, dtype=np.float64)
    rf_daily = rf_annual / TRADING_DAYS
    excess_mean = float(np.mean(r)) - rf_daily
    std = float(np.std(r, ddof=0))
    if std < 1e-12:
        return 0.0
    return excess_mean / std * math.sqrt(TRADING_DAYS)


def sortino_ratio(
    daily_returns: np.ndarray,
    rf_annual: float = DEFAULT_RF_ANNUAL,
) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(daily_returns) < 2:
        return 0.0
    r = np.asarray(daily_returns, dtype=np.float64)
    rf_daily = rf_annual / TRADING_DAYS
    excess_mean = float(np.mean(r)) - rf_daily
    down = r[r < 0]
    if len(down) < 2:
        return annualized_sharpe(daily_returns, rf_annual)
    down_std = float(np.std(down, ddof=0))
    if down_std < 1e-12:
        return 0.0
    return excess_mean / down_std * math.sqrt(TRADING_DAYS)


def cagr(daily_returns: np.ndarray) -> float:
    """Compound Annual Growth Rate from daily returns."""
    if len(daily_returns) < 1:
        return 0.0
    eq = np.cumprod(1 + np.asarray(daily_returns, dtype=np.float64))
    if eq[-1] <= 0:
        return -1.0
    n_years = len(daily_returns) / TRADING_DAYS
    return float(eq[-1] ** (1 / max(n_years, 0.01)) - 1)


def max_drawdown(daily_returns: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction (e.g., 0.15 = 15%)."""
    if len(daily_returns) < 1:
        return 0.0
    eq = np.cumprod(1 + np.asarray(daily_returns, dtype=np.float64))
    hwm = np.maximum.accumulate(eq)
    dd = 1 - eq / hwm
    return float(dd.max())


def calmar_ratio(daily_returns: np.ndarray) -> float:
    """CAGR / max drawdown."""
    dd = max_drawdown(daily_returns)
    if dd < 1e-6:
        return 0.0
    return cagr(daily_returns) / dd


def annualized_vol(daily_returns: np.ndarray) -> float:
    """Annualized volatility from daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    return float(np.std(daily_returns, ddof=0)) * math.sqrt(TRADING_DAYS)


def full_metrics(daily_returns: np.ndarray, rf_annual: float = DEFAULT_RF_ANNUAL) -> Dict[str, float]:
    """Compute all standard metrics from daily returns.

    Returns dict with: cagr_pct, sharpe, max_dd_pct, calmar, sortino, vol_pct,
                        total_ret_pct, n_days
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    if len(r) < 2:
        return {"cagr_pct": 0, "sharpe": 0, "max_dd_pct": 0, "calmar": 0,
                "sortino": 0, "vol_pct": 0, "total_ret_pct": 0, "n_days": 0}

    eq = np.cumprod(1 + r)
    total = float(eq[-1] - 1)
    n_yr = len(r) / TRADING_DAYS
    c = eq[-1] ** (1 / max(n_yr, 0.01)) - 1 if eq[-1] > 0 else 0

    s = annualized_sharpe(r, rf_annual)
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    cal = c / dd if dd > 1e-6 else 0
    sort = sortino_ratio(r, rf_annual)
    vol = annualized_vol(r)

    return {
        "cagr_pct": round(c * 100, 2),
        "sharpe": round(s, 2),
        "max_dd_pct": round(dd * 100, 2),
        "calmar": round(cal, 2),
        "sortino": round(sort, 2),
        "vol_pct": round(vol * 100, 2),
        "total_ret_pct": round(total * 100, 2),
        "n_days": len(r),
    }
