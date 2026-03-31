"""
Realized volatility calculator for crypto ETF signals.

Provides annualized realized volatility from daily close prices and the
IV-minus-RV spread that quantifies premium richness ("the edge we sell").

Design notes:
    - All returns are log returns (ln(p_t / p_{t-1}))
    - Annualization uses 365 trading days (crypto trades 24/7)
    - compute_realized_vol returns the *most recent* window's vol, not a series
    - compute_iv_rv_spread is a pure arithmetic function; no data fetching here
"""

from __future__ import annotations

import math
from typing import List


_CRYPTO_DAYS_PER_YEAR = 365  # crypto trades 365 days/year, not 252


def compute_realized_vol(prices: List[float], window: int = 7) -> float:
    """Annualized realized volatility over the most recent ``window`` days.

    Args:
        prices: Daily close prices in chronological order (oldest first).
                Minimum length required: window + 1 (to form window log-returns).
        window: Look-back window in days. Common values: 7 (weekly), 30 (monthly).

    Returns:
        Annualized realized volatility as a decimal (e.g. 0.85 means 85%/yr).
        Returns 0.0 if there are not enough data points to compute.

    Raises:
        ValueError: If prices contains non-positive values.
    """
    if len(prices) < window + 1:
        return 0.0

    recent = prices[-(window + 1):]

    if any(p <= 0 for p in recent):
        raise ValueError("All prices must be positive.")

    log_returns = [
        math.log(recent[i] / recent[i - 1])
        for i in range(1, len(recent))
    ]

    n = len(log_returns)
    mean = sum(log_returns) / n
    variance = sum((r - mean) ** 2 for r in log_returns) / (n - 1) if n > 1 else 0.0

    daily_vol = math.sqrt(variance)
    annualized = daily_vol * math.sqrt(_CRYPTO_DAYS_PER_YEAR)
    return annualized


def compute_iv_rv_spread(iv: float, rv: float) -> float:
    """Implied-minus-realized volatility spread (the premium-selling edge).

    A positive spread means the market is pricing in more vol than has been
    realized — options are "rich" and selling premium has a positive carry edge.

    Args:
        iv: Current implied volatility (annualized decimal, e.g. 0.90 = 90%).
        rv: Realized volatility over a matching horizon (same units as iv).

    Returns:
        iv - rv as a decimal. Positive → premium rich; negative → premium cheap.
    """
    return iv - rv
