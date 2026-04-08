"""Signal decay half-life analyzer.

For each alpha signal, computes autocorrelation function, predictive R²
decay curve, information coefficient over horizons 1d–60d, fits
exponential decay to find half-life, and recommends optimal rebalance
frequency.

Pure-Python — no external dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs: List[float]) -> float:
    if len(xs) < 2: return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _correlation(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3: return 0.0
    mx, my = _mean(xs[:n]), _mean(ys[:n])
    sx, sy = _std(xs[:n]), _std(ys[:n])
    if sx == 0 or sy == 0: return 0.0
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n - 1)
    return max(-1, min(1, cov / (sx * sy)))

def _autocorrelation(xs: List[float], lag: int) -> float:
    if lag >= len(xs) or len(xs) < lag + 3: return 0.0
    return _correlation(xs[:len(xs)-lag], xs[lag:])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SignalSeries:
    """One named signal with daily values and corresponding returns."""
    name: str
    values: List[float]           # daily signal values
    forward_returns: List[float]  # daily forward returns (aligned)

HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 60]


# ---------------------------------------------------------------------------
# 1. Autocorrelation function
# ---------------------------------------------------------------------------

@dataclass
class ACFResult:
    """Autocorrelation function for one signal."""
    signal_name: str
    lags: List[int]
    autocorrelations: List[float]
    first_zero_crossing: Optional[int]  # first lag where ACF < 0


def compute_acf(series: SignalSeries, lags: Optional[List[int]] = None) -> ACFResult:
    """Compute autocorrelation at multiple lags."""
    lags = lags or HORIZONS
    acfs: List[float] = []
    first_zero: Optional[int] = None

    for lag in lags:
        ac = _autocorrelation(series.values, lag)
        acfs.append(round(ac, 4))
        if first_zero is None and ac <= 0:
            first_zero = lag

    return ACFResult(series.name, lags, acfs, first_zero)


# ---------------------------------------------------------------------------
# 2. Predictive R² decay curve
# ---------------------------------------------------------------------------

@dataclass
class R2DecayPoint:
    horizon: int
    r_squared: float
    n_obs: int


@dataclass
class R2DecayResult:
    signal_name: str
    points: List[R2DecayPoint]
    peak_r2: float
    peak_horizon: int


def compute_r2_decay(
    series: SignalSeries,
    horizons: Optional[List[int]] = None,
) -> R2DecayResult:
    """Compute R² of signal vs cumulative forward returns at each horizon."""
    horizons = horizons or HORIZONS
    n = len(series.values)
    points: List[R2DecayPoint] = []
    peak_r2 = 0.0
    peak_h = 1

    for h in horizons:
        # Compute h-day cumulative returns
        cum_rets: List[float] = []
        for i in range(n - h):
            cr = 1.0
            for j in range(h):
                if i + j < len(series.forward_returns):
                    cr *= (1 + series.forward_returns[i + j])
            cum_rets.append(cr - 1.0)

        m = min(len(series.values), len(cum_rets))
        if m < 10:
            points.append(R2DecayPoint(h, 0.0, m))
            continue

        corr = _correlation(series.values[:m], cum_rets[:m])
        r2 = corr ** 2
        points.append(R2DecayPoint(h, round(r2, 6), m))

        if r2 > peak_r2:
            peak_r2 = r2
            peak_h = h

    return R2DecayResult(series.name, points, round(peak_r2, 6), peak_h)


# ---------------------------------------------------------------------------
# 3. Information coefficient over horizons
# ---------------------------------------------------------------------------

@dataclass
class ICDecayPoint:
    horizon: int
    ic: float            # Pearson correlation (information coefficient)
    ic_abs: float
    n_obs: int


@dataclass
class ICDecayResult:
    signal_name: str
    points: List[ICDecayPoint]
    peak_ic: float
    peak_horizon: int
    ic_1d: float
    ic_5d: float
    ic_20d: float


def compute_ic_decay(
    series: SignalSeries,
    horizons: Optional[List[int]] = None,
) -> ICDecayResult:
    """Compute IC (Pearson correlation) at each forward horizon."""
    horizons = horizons or HORIZONS
    n = len(series.values)
    points: List[ICDecayPoint] = []
    peak_ic = 0.0
    peak_h = 1
    ic_map: Dict[int, float] = {}

    for h in horizons:
        cum_rets: List[float] = []
        for i in range(n - h):
            cr = 1.0
            for j in range(h):
                if i + j < len(series.forward_returns):
                    cr *= (1 + series.forward_returns[i + j])
            cum_rets.append(cr - 1.0)

        m = min(len(series.values), len(cum_rets))
        if m < 10:
            points.append(ICDecayPoint(h, 0.0, 0.0, m))
            ic_map[h] = 0.0
            continue

        ic = _correlation(series.values[:m], cum_rets[:m])
        points.append(ICDecayPoint(h, round(ic, 4), round(abs(ic), 4), m))
        ic_map[h] = ic

        if abs(ic) > abs(peak_ic):
            peak_ic = ic
            peak_h = h

    return ICDecayResult(
        series.name, points, round(peak_ic, 4), peak_h,
        round(ic_map.get(1, 0), 4),
        round(ic_map.get(5, 0), 4),
        round(ic_map.get(20, 0), 4),
    )


# ---------------------------------------------------------------------------
# 4. Exponential decay fit → half-life
# ---------------------------------------------------------------------------

@dataclass
class HalfLifeResult:
    signal_name: str
    half_life_days: float         # days until signal loses half its power
    decay_rate: float             # exponential decay rate per day
    r2_at_halflife: float
    fit_quality: float            # R² of the exponential fit itself
    category: str                 # "fast" (<5d), "medium" (5-20d), "slow" (>20d)


def fit_exponential_decay(
    ic_result: ICDecayResult,
) -> HalfLifeResult:
    """Fit |IC| = A × exp(-λt) to the decay curve, extract half-life."""
    points = [(p.horizon, p.ic_abs) for p in ic_result.points if p.ic_abs > 0.001]

    if len(points) < 2:
        return HalfLifeResult(ic_result.signal_name, float('inf'), 0.0, 0.0, 0.0, "slow")

    # Log-linear fit: ln(|IC|) = ln(A) - λt
    log_ics: List[float] = []
    ts: List[float] = []
    for t, ic in points:
        if ic > 0.001:
            log_ics.append(math.log(ic))
            ts.append(float(t))

    if len(ts) < 2:
        return HalfLifeResult(ic_result.signal_name, float('inf'), 0.0, 0.0, 0.0, "slow")

    # OLS: ln(IC) = a + b*t
    mt = _mean(ts)
    ml = _mean(log_ics)
    num = sum((ts[i] - mt) * (log_ics[i] - ml) for i in range(len(ts)))
    den = sum((ts[i] - mt) ** 2 for i in range(len(ts)))

    if den == 0:
        return HalfLifeResult(ic_result.signal_name, float('inf'), 0.0, 0.0, 0.0, "slow")

    b = num / den  # should be negative (decay)
    a = ml - b * mt

    decay_rate = -b  # positive decay rate
    if decay_rate <= 0:
        # Signal is increasing — no decay
        return HalfLifeResult(ic_result.signal_name, float('inf'), 0.0, 0.0, 0.0, "slow")

    half_life = math.log(2) / decay_rate

    # R² at half-life
    r2_hl = math.exp(a) * math.exp(-decay_rate * half_life)

    # Fit quality: R² of the log-linear fit
    predicted = [a + b * t for t in ts]
    ss_res = sum((log_ics[i] - predicted[i]) ** 2 for i in range(len(ts)))
    ss_tot = sum((l - ml) ** 2 for l in log_ics)
    fit_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    if half_life < 5:
        cat = "fast"
    elif half_life < 20:
        cat = "medium"
    else:
        cat = "slow"

    return HalfLifeResult(
        ic_result.signal_name,
        round(half_life, 1), round(decay_rate, 4),
        round(r2_hl, 4), round(max(0, fit_r2), 4), cat,
    )


# ---------------------------------------------------------------------------
# 5. Optimal rebalance recommendation
# ---------------------------------------------------------------------------

@dataclass
class RebalanceRecommendation:
    signal_name: str
    half_life_days: float
    optimal_rebalance_days: int   # ~ half-life (replace at 50% decay)
    category: str
    reasoning: str


def recommend_rebalance(hl: HalfLifeResult) -> RebalanceRecommendation:
    """Recommend rebalance frequency based on half-life."""
    if hl.half_life_days <= 3:
        freq = 1
        reason = "Fast-decaying signal: rebalance daily to capture alpha before it vanishes"
    elif hl.half_life_days <= 10:
        freq = max(1, int(hl.half_life_days * 0.7))
        reason = f"Medium-fast decay ({hl.half_life_days:.0f}d HL): rebalance every {freq}d"
    elif hl.half_life_days <= 30:
        freq = max(3, int(hl.half_life_days * 0.5))
        reason = f"Medium decay ({hl.half_life_days:.0f}d HL): rebalance weekly or bi-weekly"
    elif math.isfinite(hl.half_life_days):
        freq = max(5, int(hl.half_life_days * 0.4))
        reason = f"Slow decay ({hl.half_life_days:.0f}d HL): rebalance monthly"
    else:
        freq = 20
        reason = "No measurable decay: rebalance at discretion (monthly default)"

    return RebalanceRecommendation(
        hl.signal_name, hl.half_life_days, freq, hl.category, reason,
    )


# ---------------------------------------------------------------------------
# 6. Full analysis
# ---------------------------------------------------------------------------

@dataclass
class SignalDecayAnalysis:
    signal_name: str
    acf: ACFResult
    r2_decay: R2DecayResult
    ic_decay: ICDecayResult
    half_life: HalfLifeResult
    rebalance: RebalanceRecommendation


@dataclass
class FullDecayResult:
    n_signals: int
    analyses: List[SignalDecayAnalysis]
    ranking_by_half_life: List[Tuple[str, float]]
    ranking_by_peak_ic: List[Tuple[str, float]]
    fastest_signal: str
    slowest_signal: str
    avg_half_life: float


class SignalDecayEngine:
    """Orchestrates decay analysis across multiple signals."""

    def __init__(self, signals: List[SignalSeries]) -> None:
        self.signals = signals

    def analyse(self) -> FullDecayResult:
        analyses: List[SignalDecayAnalysis] = []

        for sig in self.signals:
            acf = compute_acf(sig)
            r2 = compute_r2_decay(sig)
            ic = compute_ic_decay(sig)
            hl = fit_exponential_decay(ic)
            reb = recommend_rebalance(hl)
            analyses.append(SignalDecayAnalysis(sig.name, acf, r2, ic, hl, reb))

        by_hl = sorted([(a.signal_name, a.half_life.half_life_days) for a in analyses],
                        key=lambda x: x[1])
        by_ic = sorted([(a.signal_name, abs(a.ic_decay.peak_ic)) for a in analyses],
                        key=lambda x: -x[1])

        finite_hls = [a.half_life.half_life_days for a in analyses
                       if math.isfinite(a.half_life.half_life_days)]

        fastest = by_hl[0][0] if by_hl else ""
        slowest = by_hl[-1][0] if by_hl else ""

        return FullDecayResult(
            n_signals=len(self.signals),
            analyses=analyses,
            ranking_by_half_life=by_hl,
            ranking_by_peak_ic=by_ic,
            fastest_signal=fastest,
            slowest_signal=slowest,
            avg_half_life=round(_mean(finite_hls), 1) if finite_hls else 0,
        )


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_test_signals(n_days: int = 500, seed: int = 1390) -> List[SignalSeries]:
    """Generate synthetic signals with known decay characteristics."""
    rng = random.Random(seed)
    returns = [rng.gauss(0.0003, 0.01) for _ in range(n_days)]

    signals: List[SignalSeries] = []

    # Fast-decay signal: autocorrelation drops quickly
    fast = []
    v = 0.0
    for r in returns:
        v = 0.3 * v + 0.7 * r * 50 + rng.gauss(0, 0.3)
        fast.append(v)
    signals.append(SignalSeries("ml_ensemble", fast, returns))

    # Medium-decay signal
    med = []
    v = 0.0
    for r in returns:
        v = 0.85 * v + 0.15 * r * 30 + rng.gauss(0, 0.2)
        med.append(v)
    signals.append(SignalSeries("regime_score", med, returns))

    # Slow-decay signal (momentum)
    slow = []
    v = 0.0
    for r in returns:
        v = 0.97 * v + 0.03 * r * 20 + rng.gauss(0, 0.1)
        slow.append(v)
    signals.append(SignalSeries("momentum_20d", slow, returns))

    # Very fast (noise-like)
    noise = [r * 100 + rng.gauss(0, 1) for r in returns]
    signals.append(SignalSeries("microstructure", noise, returns))

    # Moderate with high IC
    combo = [0.5 * fast[i] + 0.3 * med[i] + 0.2 * slow[i] for i in range(n_days)]
    signals.append(SignalSeries("sentiment", combo, returns))

    # Pure noise (no signal)
    pure_noise = [rng.gauss(0, 1) for _ in range(n_days)]
    signals.append(SignalSeries("random_noise", pure_noise, returns))

    return signals
