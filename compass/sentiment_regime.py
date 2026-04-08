"""Sentiment regime detector: composite fear/greed index with changepoint detection.

Combines put/call ratio, VIX term structure slope, SKEW index, and
credit spreads (HYG-TLT proxy) into a normalised composite.  Detects
regime shifts via CUSUM and generates contrarian timing signals.

Pure-Python — no external dependencies.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
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

def _zscore(x: float, xs: List[float]) -> float:
    if len(xs) < 5: return 0.0
    m = _mean(xs); s = _std(xs)
    return (x - m) / s if s > 0 else 0.0

def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def _percentile_rank(x: float, xs: List[float]) -> float:
    """Percentile rank of x within xs, returned in [0, 1]."""
    if not xs: return 0.5
    return sum(1 for v in xs if v <= x) / len(xs)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class SentimentObs:
    """One day of sentiment data."""
    date: str
    put_call_ratio: float = 0.85       # equity-only P/C ratio
    vix_term_slope: float = 0.05       # (VX2-VX1)/VX1, positive=contango
    skew_index: float = 130.0          # CBOE SKEW (100=no skew, >130=tail fear)
    credit_spread_bps: float = 350.0   # HYG-TLT spread proxy in bps
    spy_return_5d: float = 0.0         # for backtest labelling
    ground_truth_regime: Optional[str] = None


# ---------------------------------------------------------------------------
# 1. Component normalisation
# ---------------------------------------------------------------------------

@dataclass
class NormalisedComponents:
    """Normalised sentiment components, each in [-1, +1]."""
    date: str
    pc_score: float       # put/call: high ratio = fear (-1)
    vts_score: float      # VIX term: backwardation = fear (-1)
    skew_score: float     # SKEW: high = tail fear (-1)
    credit_score: float   # credit spread: wide = fear (-1)
    composite: float      # weighted average


def normalise_observation(
    obs: SentimentObs,
    history: List[SentimentObs],
    weights: Optional[Dict[str, float]] = None,
) -> NormalisedComponents:
    """Normalise components to [-1, +1] using rolling z-score then clamp."""
    w = weights or {"pc": 0.25, "vts": 0.30, "skew": 0.20, "credit": 0.25}

    pc_hist = [h.put_call_ratio for h in history]
    vts_hist = [h.vix_term_slope for h in history]
    skew_hist = [h.skew_index for h in history]
    credit_hist = [h.credit_spread_bps for h in history]

    # Higher P/C = more fear → invert so positive = greed
    pc_z = -_zscore(obs.put_call_ratio, pc_hist)
    pc_score = _clamp(pc_z / 2)

    # Positive slope = contango = calm; negative = backwardation = fear
    vts_z = _zscore(obs.vix_term_slope, vts_hist)
    vts_score = _clamp(vts_z / 2)

    # Higher SKEW = more tail fear → invert
    skew_z = -_zscore(obs.skew_index, skew_hist)
    skew_score = _clamp(skew_z / 2)

    # Wider credit spread = more fear → invert
    credit_z = -_zscore(obs.credit_spread_bps, credit_hist)
    credit_score = _clamp(credit_z / 2)

    composite = _clamp(
        w["pc"] * pc_score + w["vts"] * vts_score +
        w["skew"] * skew_score + w["credit"] * credit_score
    )

    return NormalisedComponents(
        obs.date, round(pc_score, 4), round(vts_score, 4),
        round(skew_score, 4), round(credit_score, 4), round(composite, 4),
    )


# ---------------------------------------------------------------------------
# 2. Sentiment regime classification
# ---------------------------------------------------------------------------

SENTIMENT_REGIMES = ["extreme_fear", "fear", "neutral", "greed", "extreme_greed"]


def classify_sentiment(composite: float) -> str:
    """Map composite [-1, +1] to sentiment regime."""
    if composite <= -0.6: return "extreme_fear"
    if composite <= -0.2: return "fear"
    if composite >= 0.6: return "extreme_greed"
    if composite >= 0.2: return "greed"
    return "neutral"


# ---------------------------------------------------------------------------
# 3. CUSUM changepoint detection
# ---------------------------------------------------------------------------

@dataclass
class CUSUMChangepoint:
    """Detected regime shift."""
    day: int
    date: str
    direction: str    # "bearish_shift" or "bullish_shift"
    cusum_value: float
    composite_before: float
    composite_after: float


def detect_cusum_changepoints(
    composites: List[float],
    dates: List[str],
    threshold: float = 1.5,
    drift: float = 0.02,
) -> List[CUSUMChangepoint]:
    """CUSUM (cumulative sum) changepoint detection on composite series.

    Detects upward and downward shifts when cumulative deviation from
    running mean exceeds *threshold* standard deviations.
    """
    if len(composites) < 20:
        return []

    changepoints: List[CUSUMChangepoint] = []
    s_pos = 0.0  # positive CUSUM
    s_neg = 0.0  # negative CUSUM
    running_mean = _mean(composites[:20])
    running_std = max(_std(composites[:20]), 0.01)

    for i in range(20, len(composites)):
        z = (composites[i] - running_mean) / running_std

        s_pos = max(0, s_pos + z - drift)
        s_neg = min(0, s_neg + z + drift)

        if s_pos > threshold:
            changepoints.append(CUSUMChangepoint(
                i, dates[i], "bullish_shift", round(s_pos, 3),
                round(composites[max(0, i-5)], 3), round(composites[i], 3),
            ))
            s_pos = 0.0

        if s_neg < -threshold:
            changepoints.append(CUSUMChangepoint(
                i, dates[i], "bearish_shift", round(s_neg, 3),
                round(composites[max(0, i-5)], 3), round(composites[i], 3),
            ))
            s_neg = 0.0

        # Update running stats (exponential)
        alpha = 0.02
        running_mean = (1 - alpha) * running_mean + alpha * composites[i]
        running_std = max(0.01, (1 - alpha) * running_std + alpha * abs(composites[i] - running_mean))

    return changepoints


# ---------------------------------------------------------------------------
# 4. Contrarian signals
# ---------------------------------------------------------------------------

@dataclass
class ContrarianSignal:
    """Timing signal from extreme sentiment."""
    day: int
    date: str
    regime: str
    composite: float
    signal: str         # "buy" (extreme fear) or "reduce" (extreme greed)
    strength: float     # 0 to 1
    cooldown_remaining: int


def generate_contrarian_signals(
    composites: List[float],
    dates: List[str],
    extreme_threshold: float = 0.5,
    cooldown: int = 10,
) -> List[ContrarianSignal]:
    """Generate buy/reduce signals at sentiment extremes."""
    signals: List[ContrarianSignal] = []
    last_signal_day = -cooldown - 1

    for i, (comp, date) in enumerate(zip(composites, dates)):
        remaining = max(0, cooldown - (i - last_signal_day))
        if remaining > 0:
            continue

        regime = classify_sentiment(comp)

        if comp <= -extreme_threshold:
            signals.append(ContrarianSignal(
                i, date, regime, round(comp, 3), "buy",
                round(min(1.0, abs(comp) / 0.8), 3), 0,
            ))
            last_signal_day = i
        elif comp >= extreme_threshold:
            signals.append(ContrarianSignal(
                i, date, regime, round(comp, 3), "reduce",
                round(min(1.0, comp / 0.8), 3), 0,
            ))
            last_signal_day = i

    return signals


# ---------------------------------------------------------------------------
# 5. Backtest as timing filter for EXP-880
# ---------------------------------------------------------------------------

@dataclass
class TimingBacktest:
    """Backtest results using sentiment as timing filter."""
    n_days: int
    n_signals: int
    always_in_return: float       # buy-and-hold
    sentiment_filtered_return: float
    improvement_pct: float
    fear_buy_accuracy: float      # % of fear-buy signals that were profitable
    greed_reduce_accuracy: float


def backtest_timing_filter(
    composites: List[float],
    spy_returns: List[float],
    extreme_threshold: float = 0.5,
) -> TimingBacktest:
    """Backtest: stay invested when neutral/greed, reduce in extreme greed,
    increase in extreme fear."""
    n = min(len(composites), len(spy_returns))
    if n < 20:
        return TimingBacktest(n, 0, 0, 0, 0, 0, 0)

    always_cum = 0.0
    filtered_cum = 0.0
    fear_buys: List[bool] = []
    greed_reduces: List[bool] = []

    for i in range(n):
        r = spy_returns[i]
        comp = composites[i]
        always_cum += r

        if comp <= -extreme_threshold:
            # Extreme fear → increase exposure (1.5x)
            filtered_cum += r * 1.5
            if i + 5 < n:
                fwd = sum(spy_returns[i+1:i+6])
                fear_buys.append(fwd > 0)
        elif comp >= extreme_threshold:
            # Extreme greed → reduce (0.5x)
            filtered_cum += r * 0.5
            if i + 5 < n:
                fwd = sum(spy_returns[i+1:i+6])
                greed_reduces.append(fwd < 0)
        else:
            filtered_cum += r

    improvement = (filtered_cum - always_cum) / max(abs(always_cum), 0.001) * 100

    return TimingBacktest(
        n_days=n,
        n_signals=len(fear_buys) + len(greed_reduces),
        always_in_return=round(always_cum, 4),
        sentiment_filtered_return=round(filtered_cum, 4),
        improvement_pct=round(improvement, 1),
        fear_buy_accuracy=round(_mean([1.0 if b else 0.0 for b in fear_buys]), 3) if fear_buys else 0,
        greed_reduce_accuracy=round(_mean([1.0 if b else 0.0 for b in greed_reduces]), 3) if greed_reduces else 0,
    )


# ---------------------------------------------------------------------------
# 6. Comparison: composite vs VIX-only
# ---------------------------------------------------------------------------

@dataclass
class VsVIXComparison:
    """Compare composite sentiment detection vs VIX alone."""
    composite_changepoints: int
    vix_only_changepoints: int
    composite_earlier_count: int
    avg_lead_days: float


def compare_vs_vix(
    composites: List[float],
    vix_values: List[float],
    dates: List[str],
) -> VsVIXComparison:
    """How many changepoints does composite detect earlier than VIX?"""
    comp_cps = detect_cusum_changepoints(composites, dates)

    # VIX-only: normalise VIX to [-1,1] and detect changepoints
    if len(vix_values) >= 20:
        vix_norm = []
        for i, v in enumerate(vix_values):
            hist = vix_values[max(0, i-60):i] if i >= 5 else vix_values[:max(i, 1)]
            z = -_zscore(v, hist)  # high VIX = fear = negative
            vix_norm.append(_clamp(z / 2))
        vix_cps = detect_cusum_changepoints(vix_norm, dates)
    else:
        vix_cps = []
        vix_norm = []

    # Count how often composite detected a shift before VIX
    earlier = 0
    lead_days: List[int] = []
    for ccp in comp_cps:
        for vcp in vix_cps:
            if ccp.direction == vcp.direction and abs(ccp.day - vcp.day) < 30:
                if ccp.day < vcp.day:
                    earlier += 1
                    lead_days.append(vcp.day - ccp.day)
                break

    return VsVIXComparison(
        len(comp_cps), len(vix_cps), earlier,
        round(_mean([float(d) for d in lead_days]), 1) if lead_days else 0,
    )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

@dataclass
class SentimentRegimeResult:
    n_days: int
    composites: List[float]
    regimes: List[str]
    regime_counts: Dict[str, int]
    components: List[NormalisedComponents]
    changepoints: List[CUSUMChangepoint]
    signals: List[ContrarianSignal]
    backtest: TimingBacktest
    vs_vix: VsVIXComparison
    current_regime: str
    current_composite: float


class SentimentRegimeEngine:
    """Orchestrates sentiment regime detection and analysis."""

    def __init__(self, observations: List[SentimentObs]) -> None:
        self.observations = observations

    def analyse(self) -> SentimentRegimeResult:
        components: List[NormalisedComponents] = []
        composites: List[float] = []
        regimes: List[str] = []

        for i, obs in enumerate(self.observations):
            history = self.observations[max(0, i-60):i] if i >= 5 else self.observations[:max(i, 1)]
            nc = normalise_observation(obs, history)
            components.append(nc)
            composites.append(nc.composite)
            regimes.append(classify_sentiment(nc.composite))

        dates = [o.date for o in self.observations]
        changepoints = detect_cusum_changepoints(composites, dates)
        signals = generate_contrarian_signals(composites, dates)
        spy_rets = [o.spy_return_5d for o in self.observations]
        bt = backtest_timing_filter(composites, spy_rets)

        vix_proxy = [-o.vix_term_slope * 10 + 20 for o in self.observations]  # rough VIX proxy
        vs_vix = compare_vs_vix(composites, vix_proxy, dates)

        regime_counts = dict(defaultdict(int))
        for r in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        return SentimentRegimeResult(
            n_days=len(self.observations),
            composites=composites, regimes=regimes,
            regime_counts=regime_counts, components=components,
            changepoints=changepoints, signals=signals,
            backtest=bt, vs_vix=vs_vix,
            current_regime=regimes[-1] if regimes else "neutral",
            current_composite=composites[-1] if composites else 0.0,
        )


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_test_data(n: int = 500, seed: int = 1250) -> List[SentimentObs]:
    """Generate synthetic sentiment data with regime structure."""
    rng = random.Random(seed)
    obs: List[SentimentObs] = []
    phases = [
        (100, 0.80, 0.04, 125, 320, 0.002),    # greed
        (50,  0.95, -0.01, 140, 450, -0.001),   # fear building
        (30,  1.20, -0.04, 155, 700, -0.008),   # extreme fear
        (80,  0.75, 0.05, 120, 300, 0.003),     # recovery greed
        (100, 0.85, 0.03, 130, 350, 0.001),     # neutral
        (40,  0.70, 0.06, 115, 280, 0.004),     # extreme greed
        (100, 0.90, 0.02, 132, 380, 0.001),     # neutral
    ]
    for n_days, pc, vts, skew, cs, spy in phases:
        for _ in range(n_days):
            if len(obs) >= n: break
            obs.append(SentimentObs(
                date=f"day-{len(obs)}",
                put_call_ratio=max(0.4, pc + rng.gauss(0, 0.06)),
                vix_term_slope=vts + rng.gauss(0, 0.015),
                skew_index=max(100, skew + rng.gauss(0, 5)),
                credit_spread_bps=max(150, cs + rng.gauss(0, 30)),
                spy_return_5d=spy + rng.gauss(0, 0.01),
            ))
    return obs[:n]
