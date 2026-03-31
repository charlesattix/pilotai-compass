"""
compass/alpha_research.py — Alpha research framework with signal zoo and
automated ranking.

Systematically tests candidate alpha signals via walk-forward IC analysis,
measures marginal model improvement, estimates capacity/decay, and detects
feature interactions.

Provides:
  1. AlphaResearcher — walk-forward IC, marginal Sharpe, capacity & decay
  2. Signal zoo — 20+ pre-built candidate signals (momentum, mean-reversion,
     volatility, cross-asset, calendar, microstructure)
  3. Automated ranking — by OOS IC, turnover-adjusted IC, marginal Sharpe
  4. Feature interaction detection — pairwise synergy testing
  5. HTML report with ranking table, IC decay curves, interaction matrix

Usage:
    from compass.alpha_research import AlphaResearcher, SIGNAL_ZOO

    researcher = AlphaResearcher(prices, returns, vix)
    results = researcher.evaluate_zoo()
    html = researcher.generate_html(results)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SignalDefinition:
    """A candidate alpha signal."""
    name: str
    category: str  # momentum, mean_reversion, volatility, cross_asset, calendar, microstructure
    func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    description: str = ""


@dataclass
class ICResult:
    """Information coefficient analysis for one signal."""
    signal_name: str
    ic_mean: float  # mean Spearman IC across folds
    ic_std: float  # std of IC across folds
    icir: float  # IC information ratio = ic_mean / ic_std
    ic_by_fold: List[float]  # per-fold IC values


@dataclass
class SignalEvaluation:
    """Complete evaluation of one candidate signal."""
    signal_name: str
    category: str
    ic: ICResult
    turnover: float  # average daily turnover (fraction of signal that flips)
    turnover_adjusted_ic: float  # IC penalized by turnover
    marginal_sharpe: float  # marginal Sharpe contribution to portfolio
    ic_half_life: float  # decay half-life in days (inf if no decay)
    ic_decay_curve: List[float]  # IC at lag 1, 2, ..., max_lag
    capacity_score: float  # 0-100, higher = more capacity
    rank_ic: int = 0
    rank_turnover_ic: int = 0
    rank_marginal_sharpe: int = 0
    composite_rank: float = 0.0


@dataclass
class InteractionResult:
    """Pairwise interaction test between two signals."""
    signal_a: str
    signal_b: str
    joint_ic: float  # IC of product interaction term
    marginal_ic_a: float  # IC of signal A alone
    marginal_ic_b: float  # IC of signal B alone
    synergy: float  # joint_ic - max(marginal_ic_a, marginal_ic_b)
    correlation: float  # correlation between signals


@dataclass
class ResearchResult:
    """Complete alpha research output."""
    evaluations: List[SignalEvaluation]
    interactions: List[InteractionResult]
    top_signals: List[str]  # top N signal names by composite rank
    summary: Dict[str, Any] = field(default_factory=dict)


# ── Signal zoo: 20+ pre-built candidate signals ──────────────────────────────

def _safe_rolling(arr: np.ndarray, window: int, func: Callable) -> np.ndarray:
    """Apply a rolling function, filling leading NaNs with 0."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        result[i] = func(arr[i - window + 1:i + 1])
    return result


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


# ── Momentum signals ─────────────────────────────────────────────────────────

def sig_momentum_5d(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """5-day price momentum (return over last 5 days)."""
    result = np.zeros(len(prices))
    for i in range(5, len(prices)):
        result[i] = prices[i] / prices[i - 5] - 1
    return result


def sig_momentum_20d(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """20-day price momentum."""
    result = np.zeros(len(prices))
    for i in range(20, len(prices)):
        result[i] = prices[i] / prices[i - 20] - 1
    return result


def sig_momentum_60d(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """60-day (quarterly) price momentum."""
    result = np.zeros(len(prices))
    for i in range(60, len(prices)):
        result[i] = prices[i] / prices[i - 60] - 1
    return result


def sig_rsi_14(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """14-day RSI."""
    n = len(returns)
    result = np.full(n, 50.0)
    if n < 15:
        return result
    gains = np.where(returns > 0, returns, 0.0)
    losses = np.where(returns < 0, -returns, 0.0)
    avg_gain = np.mean(gains[:14])
    avg_loss = np.mean(losses[:14])
    for i in range(14, n):
        avg_gain = (avg_gain * 13 + gains[i]) / 14
        avg_loss = (avg_loss * 13 + losses[i]) / 14
        if avg_loss < 1e-12:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)
    return result


def sig_macd_histogram(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """MACD histogram (12/26/9)."""
    ema12 = _ema(prices, 12)
    ema26 = _ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    return macd_line - signal_line


def sig_acceleration(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Momentum acceleration: 5d momentum minus 20d momentum."""
    m5 = sig_momentum_5d(prices, returns, vix)
    m20 = sig_momentum_20d(prices, returns, vix)
    return m5 - m20


# ── Mean-reversion signals ───────────────────────────────────────────────────

def sig_mean_reversion_20d(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Z-score of price relative to 20-day SMA."""
    result = np.zeros(len(prices))
    for i in range(20, len(prices)):
        window = prices[i - 20:i]
        mu = np.mean(window)
        std = np.std(window)
        if std > 1e-12:
            result[i] = -(prices[i] - mu) / std  # negative: expect reversion
    return result


def sig_bollinger_pctb(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Bollinger %B (position within Bollinger Bands, inverted for mean-rev)."""
    result = np.zeros(len(prices))
    for i in range(20, len(prices)):
        window = prices[i - 20:i]
        mu = np.mean(window)
        std = np.std(window)
        if std > 1e-12:
            upper = mu + 2 * std
            lower = mu - 2 * std
            pctb = (prices[i] - lower) / (upper - lower)
            result[i] = -(pctb - 0.5)  # center at 0, negative = overbought
    return result


def sig_rsi_reversion(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """RSI-based mean reversion: short when RSI > 70, long when < 30."""
    rsi = sig_rsi_14(prices, returns, vix)
    return -(rsi - 50.0) / 50.0  # normalize to [-1, 1]


# ── Volatility signals ───────────────────────────────────────────────────────

def sig_realized_vol_ratio(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Ratio of 5-day to 20-day realized vol (vol regime shift)."""
    result = np.zeros(len(returns))
    for i in range(20, len(returns)):
        vol5 = np.std(returns[i - 5:i])
        vol20 = np.std(returns[i - 20:i])
        if vol20 > 1e-12:
            result[i] = vol5 / vol20 - 1.0
    return result


def sig_vix_zscore(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """VIX z-score relative to 60-day window."""
    result = np.zeros(len(vix))
    for i in range(60, len(vix)):
        window = vix[i - 60:i]
        mu = np.mean(window)
        std = np.std(window)
        if std > 1e-12:
            result[i] = (vix[i] - mu) / std
    return result


def sig_vix_term_structure(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """VIX change over 5 days (proxy for term structure slope)."""
    result = np.zeros(len(vix))
    for i in range(5, len(vix)):
        if vix[i - 5] > 1e-12:
            result[i] = (vix[i] - vix[i - 5]) / vix[i - 5]
    return result


def sig_vol_of_vol(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Volatility of VIX (20-day rolling std of VIX changes)."""
    vix_changes = np.diff(vix, prepend=vix[0])
    return _safe_rolling(vix_changes, 20, np.std)


# ── Cross-asset signals ──────────────────────────────────────────────────────

def sig_equity_vol_divergence(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Divergence between equity momentum and VIX (should be negatively correlated)."""
    m20 = sig_momentum_20d(prices, returns, vix)
    vz = sig_vix_zscore(prices, returns, vix)
    return m20 + vz  # positive = divergence (momentum up + VIX up = unusual)


def sig_risk_adjusted_momentum(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Momentum divided by realized vol (Sharpe-like signal)."""
    result = np.zeros(len(returns))
    for i in range(20, len(returns)):
        ret = prices[i] / prices[i - 20] - 1
        vol = np.std(returns[i - 20:i])
        if vol > 1e-12:
            result[i] = ret / vol
    return result


# ── Calendar signals ──────────────────────────────────────────────────────────

def sig_day_of_week(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Day of week effect: encode as sine wave (0-4 → cycle)."""
    n = len(prices)
    indices = np.arange(n)
    return np.sin(2 * np.pi * (indices % 5) / 5)


def sig_month_of_year(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Monthly seasonality: encode trading day as sine wave over ~252 cycle."""
    n = len(prices)
    indices = np.arange(n)
    return np.sin(2 * np.pi * (indices % 21) / 21)


def sig_end_of_month(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """End-of-month effect: 1.0 in last 3 days of 21-day month, else 0."""
    n = len(prices)
    indices = np.arange(n)
    day_in_month = indices % 21
    return np.where(day_in_month >= 18, 1.0, 0.0)


def sig_opex_week(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Options expiration week (3rd Friday effect): every ~21 days, days 14-18."""
    n = len(prices)
    indices = np.arange(n)
    day_in_month = indices % 21
    return np.where((day_in_month >= 14) & (day_in_month <= 18), 1.0, 0.0)


# ── Microstructure signals ────────────────────────────────────────────────────

def sig_overnight_gap(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Cumulative overnight gap indicator (proxy: extreme open moves)."""
    result = np.zeros(len(returns))
    for i in range(1, len(returns)):
        if abs(returns[i]) > 2 * np.std(returns[max(0, i - 20):i]) if i > 20 else abs(returns[i]) > 0.02:
            result[i] = returns[i]
    return result


def sig_volume_spike(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Absolute return spike proxy (large moves relative to recent vol)."""
    result = np.zeros(len(returns))
    for i in range(20, len(returns)):
        vol = np.std(returns[i - 20:i])
        if vol > 1e-12:
            result[i] = abs(returns[i]) / vol
    return result


def sig_return_skew_20d(prices: np.ndarray, returns: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """Rolling 20-day return skewness."""
    n = len(returns)
    result = np.zeros(n)
    for i in range(20, n):
        window = returns[i - 20:i]
        mu = np.mean(window)
        std = np.std(window)
        if std > 1e-12:
            result[i] = np.mean(((window - mu) / std) ** 3)
    return result


# ── Signal Zoo registry ───────────────────────────────────────────────────────

SIGNAL_ZOO: List[SignalDefinition] = [
    # Momentum (6)
    SignalDefinition("momentum_5d", "momentum", sig_momentum_5d, "5-day price momentum"),
    SignalDefinition("momentum_20d", "momentum", sig_momentum_20d, "20-day price momentum"),
    SignalDefinition("momentum_60d", "momentum", sig_momentum_60d, "60-day quarterly momentum"),
    SignalDefinition("rsi_14", "momentum", sig_rsi_14, "14-day RSI"),
    SignalDefinition("macd_histogram", "momentum", sig_macd_histogram, "MACD histogram (12/26/9)"),
    SignalDefinition("acceleration", "momentum", sig_acceleration, "Momentum acceleration (5d - 20d)"),
    # Mean-reversion (3)
    SignalDefinition("mean_reversion_20d", "mean_reversion", sig_mean_reversion_20d, "20-day z-score mean reversion"),
    SignalDefinition("bollinger_pctb", "mean_reversion", sig_bollinger_pctb, "Bollinger %B inverted"),
    SignalDefinition("rsi_reversion", "mean_reversion", sig_rsi_reversion, "RSI-based mean reversion"),
    # Volatility (4)
    SignalDefinition("realized_vol_ratio", "volatility", sig_realized_vol_ratio, "5d/20d realized vol ratio"),
    SignalDefinition("vix_zscore", "volatility", sig_vix_zscore, "VIX 60-day z-score"),
    SignalDefinition("vix_term_structure", "volatility", sig_vix_term_structure, "VIX 5-day change"),
    SignalDefinition("vol_of_vol", "volatility", sig_vol_of_vol, "20-day volatility of VIX"),
    # Cross-asset (2)
    SignalDefinition("equity_vol_divergence", "cross_asset", sig_equity_vol_divergence, "Equity momentum vs VIX divergence"),
    SignalDefinition("risk_adjusted_momentum", "cross_asset", sig_risk_adjusted_momentum, "Momentum / realized vol"),
    # Calendar (4)
    SignalDefinition("day_of_week", "calendar", sig_day_of_week, "Day-of-week sine encoding"),
    SignalDefinition("month_of_year", "calendar", sig_month_of_year, "Monthly seasonality sine"),
    SignalDefinition("end_of_month", "calendar", sig_end_of_month, "End-of-month flag"),
    SignalDefinition("opex_week", "calendar", sig_opex_week, "Options expiration week flag"),
    # Microstructure (3)
    SignalDefinition("overnight_gap", "microstructure", sig_overnight_gap, "Overnight gap indicator"),
    SignalDefinition("volume_spike", "microstructure", sig_volume_spike, "Absolute return spike"),
    SignalDefinition("return_skew_20d", "microstructure", sig_return_skew_20d, "20-day return skewness"),
]


# ── Core analytics ────────────────────────────────────────────────────────────

def compute_spearman_ic(signal: np.ndarray, forward_returns: np.ndarray) -> float:
    """Spearman rank correlation between signal and forward returns.

    Handles NaN and constant arrays gracefully.
    """
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    s = signal[mask]
    r = forward_returns[mask]
    if len(s) < 10:
        return 0.0
    if np.std(s) < 1e-12 or np.std(r) < 1e-12:
        return 0.0
    # Rank-based correlation
    s_rank = _rank_array(s)
    r_rank = _rank_array(r)
    return float(np.corrcoef(s_rank, r_rank)[0, 1])


def _rank_array(arr: np.ndarray) -> np.ndarray:
    """Rank array values (average rank for ties)."""
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    return ranks


def compute_walk_forward_ic(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    n_folds: int = 5,
) -> ICResult:
    """Walk-forward IC: expanding-window train, test on next fold.

    Returns per-fold IC and aggregate statistics.
    """
    n = len(signal)
    fold_size = n // n_folds
    if fold_size < 20:
        return ICResult(
            signal_name="", ic_mean=0.0, ic_std=0.0, icir=0.0, ic_by_fold=[],
        )

    ics: List[float] = []
    for fold in range(1, n_folds):
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, n)
        ic = compute_spearman_ic(
            signal[test_start:test_end],
            forward_returns[test_start:test_end],
        )
        ics.append(ic)

    ic_mean = float(np.mean(ics)) if ics else 0.0
    ic_std = float(np.std(ics)) if len(ics) > 1 else 0.0
    icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0

    return ICResult(
        signal_name="",
        ic_mean=round(ic_mean, 6),
        ic_std=round(ic_std, 6),
        icir=round(icir, 4),
        ic_by_fold=ics,
    )


def compute_turnover(signal: np.ndarray) -> float:
    """Average daily turnover: fraction of signal that changes sign."""
    valid = signal[np.isfinite(signal)]
    if len(valid) < 2:
        return 0.0
    signs = np.sign(valid)
    flips = np.sum(signs[1:] != signs[:-1])
    return float(flips / (len(valid) - 1))


def compute_ic_decay(
    signal: np.ndarray,
    returns: np.ndarray,
    max_lag: int = 20,
) -> Tuple[List[float], float]:
    """Compute IC at increasing forward lags to measure decay.

    Returns:
        (decay_curve, half_life)
        decay_curve: IC at lag 1, 2, ..., max_lag
        half_life: estimated half-life in days (inf if no decay)
    """
    decay_curve: List[float] = []
    for lag in range(1, max_lag + 1):
        if lag >= len(returns):
            decay_curve.append(0.0)
            continue
        fwd = np.zeros(len(returns))
        for i in range(len(returns) - lag):
            fwd[i] = np.sum(returns[i + 1:i + 1 + lag])
        ic = compute_spearman_ic(signal, fwd)
        decay_curve.append(ic)

    # Estimate half-life via log-linear regression on |IC|
    half_life = float("inf")
    if len(decay_curve) >= 3:
        abs_ic = np.array([abs(x) for x in decay_curve])
        valid = abs_ic > 1e-6
        if np.sum(valid) >= 3:
            log_ic = np.log(abs_ic[valid] + 1e-10)
            x = np.arange(1, len(abs_ic) + 1)[valid].astype(float)
            if len(x) >= 2:
                slope = float(np.polyfit(x, log_ic, 1)[0])
                if slope < -1e-6:
                    half_life = round(math.log(2) / abs(slope), 1)

    return decay_curve, half_life


def compute_marginal_sharpe(
    signal: np.ndarray,
    returns: np.ndarray,
    existing_signals: Optional[np.ndarray] = None,
) -> float:
    """Marginal Sharpe contribution of adding this signal.

    Computes Sharpe of signal-weighted returns, then subtracts Sharpe
    of existing signals if provided.
    """
    valid = np.isfinite(signal) & np.isfinite(returns)
    s = signal[valid]
    r = returns[valid]
    if len(s) < 20 or np.std(s) < 1e-12:
        return 0.0

    # Normalize signal to unit std
    s_norm = (s - np.mean(s)) / np.std(s)
    weighted = s_norm * r

    mean_w = np.mean(weighted)
    std_w = np.std(weighted)
    if std_w < 1e-12:
        return 0.0
    signal_sharpe = float(mean_w / std_w * math.sqrt(PERIODS_PER_YEAR))

    if existing_signals is not None:
        ex = existing_signals[valid]
        if np.std(ex) > 1e-12:
            ex_norm = (ex - np.mean(ex)) / np.std(ex)
            ex_weighted = ex_norm * r
            ex_mean = np.mean(ex_weighted)
            ex_std = np.std(ex_weighted)
            if ex_std > 1e-12:
                existing_sharpe = float(ex_mean / ex_std * math.sqrt(PERIODS_PER_YEAR))
                return signal_sharpe - existing_sharpe

    return signal_sharpe


def compute_capacity_score(turnover: float, ic_half_life: float) -> float:
    """Estimate capacity: lower turnover + longer half-life = more capacity.

    Score from 0-100.
    """
    # Turnover penalty: low turnover = high capacity
    turnover_score = max(0, 50 * (1.0 - turnover * 2))
    # Half-life bonus: longer decay = more capacity
    if math.isinf(ic_half_life) or ic_half_life > 100:
        hl_score = 50.0
    else:
        hl_score = min(50.0, ic_half_life * 2.5)
    return round(min(100, turnover_score + hl_score), 1)


# ── Interaction detection ─────────────────────────────────────────────────────

def test_interaction(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    forward_returns: np.ndarray,
) -> InteractionResult:
    """Test pairwise interaction between two signals.

    Computes IC of the product term (signal_a * signal_b) and compares
    to individual ICs to measure synergy.
    """
    mask = (
        np.isfinite(signal_a) & np.isfinite(signal_b)
        & np.isfinite(forward_returns)
    )
    a = signal_a[mask]
    b = signal_b[mask]
    fr = forward_returns[mask]

    if len(a) < 20:
        return InteractionResult("", "", 0.0, 0.0, 0.0, 0.0, 0.0)

    # Normalize before interaction
    a_std = np.std(a)
    b_std = np.std(b)
    a_norm = (a - np.mean(a)) / a_std if a_std > 1e-12 else np.zeros_like(a)
    b_norm = (b - np.mean(b)) / b_std if b_std > 1e-12 else np.zeros_like(b)

    interaction = a_norm * b_norm
    joint_ic = compute_spearman_ic(interaction, fr)
    ic_a = compute_spearman_ic(a, fr)
    ic_b = compute_spearman_ic(b, fr)
    synergy = joint_ic - max(abs(ic_a), abs(ic_b))

    corr = float(np.corrcoef(a_norm, b_norm)[0, 1]) if len(a) > 1 else 0.0

    return InteractionResult(
        signal_a="", signal_b="",
        joint_ic=round(joint_ic, 6),
        marginal_ic_a=round(ic_a, 6),
        marginal_ic_b=round(ic_b, 6),
        synergy=round(synergy, 6),
        correlation=round(corr, 4),
    )


# ── AlphaResearcher ──────────────────────────────────────────────────────────

class AlphaResearcher:
    """Systematically evaluates candidate alpha signals.

    Args:
        prices: numpy array of price levels (e.g. SPY close).
        returns: numpy array of daily returns (same length as prices).
        vix: numpy array of VIX levels (same length).
        forward_returns: Optional pre-computed forward returns. If None,
            uses 1-day forward returns from the returns array.
        n_folds: Number of walk-forward folds for IC analysis.
        max_lag: Maximum lag for IC decay curve.
        top_n: Number of top signals to report.
    """

    def __init__(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        vix: np.ndarray,
        forward_returns: Optional[np.ndarray] = None,
        n_folds: int = 5,
        max_lag: int = 20,
        top_n: int = 10,
    ):
        self.prices = np.asarray(prices, dtype=float)
        self.returns = np.asarray(returns, dtype=float)
        self.vix = np.asarray(vix, dtype=float)
        self.n = len(prices)
        self.n_folds = n_folds
        self.max_lag = max_lag
        self.top_n = top_n

        if len(returns) != self.n or len(vix) != self.n:
            raise ValueError("prices, returns, and vix must have same length")
        if self.n < 50:
            raise ValueError("Need at least 50 data points for alpha research")

        if forward_returns is not None:
            self.forward_returns = np.asarray(forward_returns, dtype=float)
        else:
            self.forward_returns = np.zeros(self.n)
            self.forward_returns[:-1] = returns[1:]

    def evaluate_signal(
        self,
        signal_def: SignalDefinition,
        existing_signals: Optional[np.ndarray] = None,
    ) -> SignalEvaluation:
        """Evaluate a single candidate signal."""
        signal = signal_def.func(self.prices, self.returns, self.vix)

        ic = compute_walk_forward_ic(signal, self.forward_returns, self.n_folds)
        ic.signal_name = signal_def.name

        turnover = compute_turnover(signal)
        turnover_penalty = max(0.5, 1.0 - turnover)
        turnover_adj_ic = ic.ic_mean * turnover_penalty

        marginal_sharpe = compute_marginal_sharpe(
            signal, self.returns, existing_signals
        )
        decay_curve, half_life = compute_ic_decay(
            signal, self.returns, self.max_lag
        )
        capacity = compute_capacity_score(turnover, half_life)

        return SignalEvaluation(
            signal_name=signal_def.name,
            category=signal_def.category,
            ic=ic,
            turnover=round(turnover, 4),
            turnover_adjusted_ic=round(turnover_adj_ic, 6),
            marginal_sharpe=round(marginal_sharpe, 4),
            ic_half_life=half_life,
            ic_decay_curve=decay_curve,
            capacity_score=capacity,
        )

    def evaluate_zoo(
        self,
        signals: Optional[List[SignalDefinition]] = None,
        test_interactions: bool = True,
        interaction_top_n: int = 5,
    ) -> ResearchResult:
        """Evaluate all signals in the zoo and rank them.

        Args:
            signals: Signal list to evaluate (defaults to SIGNAL_ZOO).
            test_interactions: Whether to test pairwise interactions.
            interaction_top_n: Number of top signals to test interactions for.

        Returns:
            ResearchResult with evaluations, interactions, and rankings.
        """
        signal_defs = signals or SIGNAL_ZOO
        evaluations: List[SignalEvaluation] = []

        for sd in signal_defs:
            try:
                ev = self.evaluate_signal(sd)
                evaluations.append(ev)
            except Exception as e:
                logger.warning("Signal %s failed: %s", sd.name, e)

        # Rank by IC (absolute value — direction-agnostic)
        by_ic = sorted(evaluations, key=lambda e: abs(e.ic.ic_mean), reverse=True)
        for i, ev in enumerate(by_ic):
            ev.rank_ic = i + 1

        # Rank by turnover-adjusted IC
        by_taic = sorted(evaluations, key=lambda e: abs(e.turnover_adjusted_ic), reverse=True)
        for i, ev in enumerate(by_taic):
            ev.rank_turnover_ic = i + 1

        # Rank by marginal Sharpe
        by_sharpe = sorted(evaluations, key=lambda e: e.marginal_sharpe, reverse=True)
        for i, ev in enumerate(by_sharpe):
            ev.rank_marginal_sharpe = i + 1

        # Composite rank = average of three ranks
        for ev in evaluations:
            ev.composite_rank = round(
                (ev.rank_ic + ev.rank_turnover_ic + ev.rank_marginal_sharpe) / 3, 2
            )

        evaluations.sort(key=lambda e: e.composite_rank)
        top_names = [e.signal_name for e in evaluations[:self.top_n]]

        # Interaction detection
        interactions: List[InteractionResult] = []
        if test_interactions and len(evaluations) >= 2:
            top_for_interaction = evaluations[:interaction_top_n]
            signal_cache: Dict[str, np.ndarray] = {}
            for sd in signal_defs:
                if sd.name in [e.signal_name for e in top_for_interaction]:
                    signal_cache[sd.name] = sd.func(
                        self.prices, self.returns, self.vix
                    )

            for i, ev_a in enumerate(top_for_interaction):
                for ev_b in top_for_interaction[i + 1:]:
                    sa = signal_cache.get(ev_a.signal_name)
                    sb = signal_cache.get(ev_b.signal_name)
                    if sa is not None and sb is not None:
                        ir = test_interaction(sa, sb, self.forward_returns)
                        ir.signal_a = ev_a.signal_name
                        ir.signal_b = ev_b.signal_name
                        interactions.append(ir)

        interactions.sort(key=lambda x: x.synergy, reverse=True)

        summary = {
            "n_signals": len(evaluations),
            "n_interactions": len(interactions),
            "top_signal": top_names[0] if top_names else "N/A",
            "best_ic": round(max(abs(e.ic.ic_mean) for e in evaluations), 4) if evaluations else 0,
            "best_synergy": round(interactions[0].synergy, 4) if interactions else 0,
            "categories": list(set(e.category for e in evaluations)),
        }

        logger.info(
            "Alpha research: %d signals evaluated, top=%s (IC=%.4f)",
            len(evaluations), summary["top_signal"], summary["best_ic"],
        )

        return ResearchResult(
            evaluations=evaluations,
            interactions=interactions,
            top_signals=top_names,
            summary=summary,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # HTML report
    # ─────────────────────────────────────────────────────────────────────────

    def generate_html(self, result: ResearchResult) -> str:
        """Generate HTML report with ranking table, IC decay, interactions."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        s = result.summary

        # ── Signal ranking table ─────────────────────────────────────────
        rank_rows = ""
        for ev in result.evaluations:
            ic_cls = "good" if abs(ev.ic.ic_mean) > 0.02 else "neutral"
            rank_rows += (
                f"<tr><td>{ev.composite_rank:.1f}</td>"
                f"<td><b>{ev.signal_name}</b></td>"
                f"<td>{ev.category}</td>"
                f"<td class='{ic_cls}'>{ev.ic.ic_mean:+.4f}</td>"
                f"<td>{ev.ic.icir:+.2f}</td>"
                f"<td>{ev.turnover_adjusted_ic:+.4f}</td>"
                f"<td>{ev.marginal_sharpe:+.2f}</td>"
                f"<td>{ev.turnover:.2%}</td>"
                f"<td>{ev.ic_half_life:.0f}d</td>"
                f"<td>{ev.capacity_score:.0f}</td></tr>\n"
            )

        # ── IC decay curves (SVG) ───────────────────────────────────────
        decay_svg = self._render_decay_curves(result.evaluations[:5])

        # ── Interaction matrix ───────────────────────────────────────────
        int_rows = ""
        for ir in result.interactions[:15]:
            syn_cls = "good" if ir.synergy > 0.005 else ("bad" if ir.synergy < -0.005 else "neutral")
            int_rows += (
                f"<tr><td>{ir.signal_a}</td><td>{ir.signal_b}</td>"
                f"<td>{ir.joint_ic:+.4f}</td>"
                f"<td>{ir.marginal_ic_a:+.4f}</td>"
                f"<td>{ir.marginal_ic_b:+.4f}</td>"
                f"<td class='{syn_cls}'>{ir.synergy:+.4f}</td>"
                f"<td>{ir.correlation:+.2f}</td></tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Alpha Research Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  .good {{ color: #16a34a; }}
  .bad {{ color: #dc2626; }}
  .neutral {{ color: #64748b; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.85em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1.5em; margin: 1.5em 0; overflow-x: auto; }}
  .section {{ margin-bottom: 2.5em; }}
  .legend {{ display: flex; gap: 1.5em; margin-top: 0.5em; font-size: 0.85em; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.3em; }}
  .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Alpha Research Report</h1>
<div class="meta">Generated: {now} | {s['n_signals']} signals | {s['n_interactions']} interactions tested</div>

<div class="kpi-row">
  <div class="kpi"><div class="value">{s['n_signals']}</div><div class="label">Signals Tested</div></div>
  <div class="kpi"><div class="value">{s['top_signal']}</div><div class="label">Top Signal</div></div>
  <div class="kpi"><div class="value">{s['best_ic']:.4f}</div><div class="label">Best |IC|</div></div>
  <div class="kpi"><div class="value">{s['best_synergy']:+.4f}</div><div class="label">Best Synergy</div></div>
</div>

<div class="section">
<h2>Signal Ranking</h2>
<table>
<thead><tr><th>Rank</th><th>Signal</th><th>Category</th><th>IC</th><th>ICIR</th><th>TO-adj IC</th><th>Marg. Sharpe</th><th>Turnover</th><th>Half-Life</th><th>Capacity</th></tr></thead>
<tbody>{rank_rows}</tbody>
</table>
</div>

<div class="section">
<h2>IC Decay Curves (Top 5)</h2>
<div class="chart">
{decay_svg}
<div class="legend">
{self._legend_items(result.evaluations[:5])}
</div>
</div>
</div>

<div class="section">
<h2>Feature Interactions</h2>
<table>
<thead><tr><th>Signal A</th><th>Signal B</th><th>Joint IC</th><th>IC(A)</th><th>IC(B)</th><th>Synergy</th><th>Corr</th></tr></thead>
<tbody>{int_rows}</tbody>
</table>
</div>

</body>
</html>"""
        return html

    @staticmethod
    def _render_decay_curves(evaluations: List[SignalEvaluation]) -> str:
        """SVG line chart of IC decay for top signals."""
        if not evaluations:
            return "<p>No data</p>"

        w, h = 600, 250
        pad_l, pad_r, pad_t, pad_b = 55, 15, 15, 30
        pw = w - pad_l - pad_r
        ph = h - pad_t - pad_b
        colors = ["#3b82f6", "#f59e0b", "#8b5cf6", "#10b981", "#ef4444"]

        all_vals = [v for ev in evaluations for v in ev.ic_decay_curve if ev.ic_decay_curve]
        if not all_vals:
            return "<p>No decay data</p>"
        y_min = min(min(all_vals), -0.01)
        y_max = max(max(all_vals), 0.01)
        y_range = max(y_max - y_min, 0.001)

        lines = ""
        for idx, ev in enumerate(evaluations):
            if not ev.ic_decay_curve:
                continue
            color = colors[idx % len(colors)]
            pts = []
            n = len(ev.ic_decay_curve)
            for i, val in enumerate(ev.ic_decay_curve):
                x = pad_l + (i / max(n - 1, 1)) * pw
                y = pad_t + (1 - (val - y_min) / y_range) * ph
                pts.append(f"{x:.1f},{y:.1f}")
            lines += (
                f'<polyline points="{" ".join(pts)}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>\n'
            )

        # Zero line
        zy = pad_t + (1 - (0 - y_min) / y_range) * ph
        zero = (
            f'<line x1="{pad_l}" y1="{zy:.1f}" x2="{w - pad_r}" '
            f'y2="{zy:.1f}" stroke="#94a3b8" stroke-width="0.5" '
            f'stroke-dasharray="4,3"/>\n'
        )

        # X axis label
        x_label = (
            f'<text x="{pad_l + pw / 2}" y="{h - 2}" text-anchor="middle" '
            f'font-size="11" fill="#334155">Forward Lag (days)</text>\n'
        )

        return (
            f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">\n'
            f'{zero}{lines}{x_label}</svg>'
        )

    @staticmethod
    def _legend_items(evaluations: List[SignalEvaluation]) -> str:
        colors = ["#3b82f6", "#f59e0b", "#8b5cf6", "#10b981", "#ef4444"]
        items = ""
        for idx, ev in enumerate(evaluations):
            c = colors[idx % len(colors)]
            items += (
                f'<div class="legend-item">'
                f'<div class="legend-swatch" style="background:{c}"></div>'
                f'{ev.signal_name}</div>\n'
            )
        return items


# ── Convenience ───────────────────────────────────────────────────────────────

def generate_report(
    prices: np.ndarray,
    returns: np.ndarray,
    vix: np.ndarray,
    output_path: str = "reports/alpha_research.html",
    **kwargs,
) -> ResearchResult:
    """One-call: evaluate signal zoo and write HTML report."""
    researcher = AlphaResearcher(prices, returns, vix, **kwargs)
    result = researcher.evaluate_zoo()
    html = researcher.generate_html(result)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info("Alpha research report written to %s", output_path)
    return result
