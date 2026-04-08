"""Hidden Semi-Markov regime transition predictor.

Duration-dependent HMM that models expected regime durations, predicts
regime transitions 1-5 days ahead, and generates early-warning signals
for preemptive position adjustment.

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

def _logsumexp(vals: List[float]) -> float:
    if not vals: return -math.inf
    m = max(vals)
    if m == -math.inf: return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in vals))

def _log_gauss(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0: return -1e12
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2 * math.pi)


# ---------------------------------------------------------------------------
# Regimes and observation
# ---------------------------------------------------------------------------

REGIMES = ["bull", "bear", "sideways", "crisis"]
N_REGIMES = len(REGIMES)
_IDX = {r: i for i, r in enumerate(REGIMES)}


@dataclass
class MarketObs:
    """Daily market observation."""
    date: str
    ret_20d: float = 0.0
    ret_60d: float = 0.0
    rvol_20d: float = 0.12
    vix: float = 15.0
    credit_spread_bps: float = 350.0
    ground_truth: Optional[str] = None


# ---------------------------------------------------------------------------
# Duration distributions (Geometric/Negative-Binomial proxy)
# ---------------------------------------------------------------------------

@dataclass
class DurationParams:
    """Expected duration for one regime (geometric distribution)."""
    mean_duration: float    # expected days in this regime
    min_duration: int       # minimum before transition allowed
    hazard_rate: float      # P(exit | survived d days) = 1/mean after min


DEFAULT_DURATIONS: Dict[str, DurationParams] = {
    "bull":    DurationParams(mean_duration=120, min_duration=20, hazard_rate=0.008),
    "bear":    DurationParams(mean_duration=60,  min_duration=10, hazard_rate=0.017),
    "sideways":DurationParams(mean_duration=40,  min_duration=5,  hazard_rate=0.025),
    "crisis":  DurationParams(mean_duration=15,  min_duration=3,  hazard_rate=0.067),
}


def duration_survival_prob(regime: str, days_in_regime: int) -> float:
    """P(stay in regime | already been d days). Decreases with duration."""
    dp = DEFAULT_DURATIONS.get(regime, DEFAULT_DURATIONS["sideways"])
    if days_in_regime < dp.min_duration:
        return 0.98  # very unlikely to leave before minimum
    # Hazard increases slowly after min_duration
    excess = days_in_regime - dp.min_duration
    hazard = dp.hazard_rate * (1 + excess * 0.005)  # slight increase over time
    return max(0.5, 1.0 - hazard)


def duration_transition_prob(regime: str, days_in_regime: int) -> float:
    """P(transition out | already d days) = 1 - survival."""
    return 1.0 - duration_survival_prob(regime, days_in_regime)


# ---------------------------------------------------------------------------
# HSMM emission model
# ---------------------------------------------------------------------------

_EMISSION = {
    "bull":    [(0.03,0.04),(0.08,0.06),(0.12,0.04),(15,4),(300,80)],
    "bear":    [(-0.04,0.04),(-0.10,0.06),(0.18,0.05),(24,5),(450,120)],
    "sideways":[(0.005,0.02),(0.02,0.03),(0.09,0.03),(14,3),(320,70)],
    "crisis":  [(-0.15,0.08),(-0.25,0.10),(0.45,0.15),(50,15),(700,200)],
}

_BASE_TRANS = [
    [0.00, 0.35, 0.55, 0.10],  # bull→ (can't self-transition here; handled by duration)
    [0.40, 0.00, 0.35, 0.25],  # bear→
    [0.45, 0.25, 0.00, 0.30],  # sideways→
    [0.10, 0.30, 0.55, 0.05],  # crisis→
]


def _log_emission(obs: MarketObs, regime_idx: int) -> float:
    r = REGIMES[regime_idx]
    feats = [obs.ret_20d, obs.ret_60d, obs.rvol_20d, obs.vix, obs.credit_spread_bps]
    params = _EMISSION[r]
    return sum(_log_gauss(feats[k], params[k][0], params[k][1]) for k in range(len(feats)))


# ---------------------------------------------------------------------------
# HSMM forward pass
# ---------------------------------------------------------------------------

@dataclass
class HSMMState:
    """Internal state of the HSMM."""
    log_belief: List[float]
    current_regime: str
    days_in_regime: int
    history: List[str]
    transition_probs_history: List[Dict[str, float]]


class HSMMDetector:
    """Hidden Semi-Markov Model with duration-dependent transitions."""

    def __init__(self, min_hold: int = 3) -> None:
        self._log_belief = [-math.log(N_REGIMES)] * N_REGIMES
        self._current = "bull"
        self._days_in = 0
        self._min_hold = min_hold
        self._history: List[str] = []
        self._trans_prob_hist: List[Dict[str, float]] = []

    @property
    def current_regime(self) -> str:
        return self._current

    @property
    def days_in_regime(self) -> int:
        return self._days_in

    @property
    def history(self) -> List[str]:
        return list(self._history)

    def _duration_adjusted_trans(self) -> List[List[float]]:
        """Transition matrix adjusted by duration survival probability."""
        result = [[0.0] * N_REGIMES for _ in range(N_REGIMES)]
        for i, r in enumerate(REGIMES):
            p_stay = duration_survival_prob(r, self._days_in if r == self._current else 0)
            p_leave = 1.0 - p_stay
            result[i][i] = p_stay
            for j in range(N_REGIMES):
                if i != j:
                    result[i][j] = p_leave * _BASE_TRANS[i][j]
            # Normalise row
            total = sum(result[i])
            if total > 0:
                result[i] = [v / total for v in result[i]]
        return result

    def update(self, obs: MarketObs) -> Dict[str, float]:
        """Process one observation, return posterior probabilities."""
        trans = self._duration_adjusted_trans()
        log_trans = [[math.log(max(p, 1e-15)) for p in row] for row in trans]

        # Predict
        log_pred = []
        for j in range(N_REGIMES):
            terms = [self._log_belief[i] + log_trans[i][j] for i in range(N_REGIMES)]
            log_pred.append(_logsumexp(terms))

        # Emit
        log_up = []
        for j in range(N_REGIMES):
            log_up.append(log_pred[j] + _log_emission(obs, j))

        # Normalise
        lse = _logsumexp(log_up)
        self._log_belief = [v - lse for v in log_up]

        posterior = {REGIMES[i]: math.exp(self._log_belief[i]) for i in range(N_REGIMES)}

        # MAP with min-hold
        map_regime = max(posterior, key=posterior.get)
        if map_regime != self._current:
            if self._days_in < self._min_hold:
                map_regime = self._current
            else:
                self._current = map_regime
                self._days_in = 0

        self._days_in += 1
        self._history.append(self._current)
        self._trans_prob_hist.append(dict(posterior))

        return posterior

    def predict_transition(self, horizon: int = 5) -> Dict[str, float]:
        """P(regime at t+horizon) using duration-adjusted transition matrix."""
        lse = _logsumexp(self._log_belief)
        prob = [math.exp(b - lse) for b in self._log_belief]

        for step in range(horizon):
            trans = [[0.0] * N_REGIMES for _ in range(N_REGIMES)]
            for i, r in enumerate(REGIMES):
                d = self._days_in + step if r == self._current else step
                p_stay = duration_survival_prob(r, d)
                p_leave = 1.0 - p_stay
                trans[i][i] = p_stay
                for j in range(N_REGIMES):
                    if i != j:
                        trans[i][j] = p_leave * _BASE_TRANS[i][j]
                total = sum(trans[i])
                if total > 0:
                    trans[i] = [v / total for v in trans[i]]

            new_prob = [0.0] * N_REGIMES
            for j in range(N_REGIMES):
                for i in range(N_REGIMES):
                    new_prob[j] += prob[i] * trans[i][j]
            prob = new_prob

        return {REGIMES[i]: round(prob[i], 4) for i in range(N_REGIMES)}

    def transition_probability(self, horizon: int = 5) -> float:
        """P(regime changes within horizon days)."""
        forecast = self.predict_transition(horizon)
        return round(1.0 - forecast.get(self._current, 0), 4)

    def reset(self) -> None:
        self._log_belief = [-math.log(N_REGIMES)] * N_REGIMES
        self._current = "bull"
        self._days_in = 0
        self._history = []
        self._trans_prob_hist = []


# ---------------------------------------------------------------------------
# Early warning signals
# ---------------------------------------------------------------------------

@dataclass
class EarlyWarning:
    """Transition early-warning signal."""
    day: int
    date: str
    current_regime: str
    days_in_regime: int
    transition_prob_5d: float
    most_likely_next: str
    signal: str               # "warn_exit", "imminent_exit", "stable"
    strength: float


def generate_early_warnings(
    detector: HSMMDetector,
    observations: List[MarketObs],
    warn_threshold: float = 0.30,
    imminent_threshold: float = 0.50,
) -> Tuple[List[EarlyWarning], List[str]]:
    """Run detector on observations and generate early warnings."""
    warnings: List[EarlyWarning] = []
    regimes: List[str] = []

    for i, obs in enumerate(observations):
        posterior = detector.update(obs)
        regimes.append(detector.current_regime)

        tp5 = detector.transition_probability(5)
        forecast = detector.predict_transition(5)
        next_regime = max((r for r in forecast if r != detector.current_regime),
                          key=lambda r: forecast[r], default=detector.current_regime)

        if tp5 >= imminent_threshold:
            signal = "imminent_exit"
            strength = min(1.0, tp5 / 0.7)
        elif tp5 >= warn_threshold:
            signal = "warn_exit"
            strength = min(1.0, tp5 / 0.5)
        else:
            signal = "stable"
            strength = 0.0

        warnings.append(EarlyWarning(
            i, obs.date, detector.current_regime, detector.days_in_regime,
            tp5, next_regime, signal, round(strength, 3),
        ))

    return warnings, regimes


# ---------------------------------------------------------------------------
# Backtest: preemptive vs reactive
# ---------------------------------------------------------------------------

@dataclass
class SwitchBacktest:
    """Preemptive vs reactive regime switching comparison."""
    n_days: int
    reactive_transitions: int
    preemptive_transitions: int
    avg_lead_time: float          # days earlier for preemptive
    false_alarm_rate: float
    preemptive_pnl: float
    reactive_pnl: float
    pnl_improvement_pct: float


def backtest_switching(
    observations: List[MarketObs],
    warn_threshold: float = 0.30,
) -> SwitchBacktest:
    """Compare preemptive (HSMM early warning) vs reactive (lagged rule-based)."""
    n = len(observations)
    if n < 30:
        return SwitchBacktest(n, 0, 0, 0, 0, 0, 0, 0)

    # Preemptive: HSMM
    det_pre = HSMMDetector(min_hold=3)
    warnings, pre_regimes = generate_early_warnings(det_pre, observations, warn_threshold)

    # Reactive: simple rule-based (lagged by 5 days)
    det_react = HSMMDetector(min_hold=5)  # higher min_hold = more lag
    react_regimes: List[str] = []
    for obs in observations:
        det_react.update(obs)
        react_regimes.append(det_react.current_regime)

    # Count transitions
    pre_trans = sum(1 for i in range(1, n) if pre_regimes[i] != pre_regimes[i-1])
    react_trans = sum(1 for i in range(1, n) if react_regimes[i] != react_regimes[i-1])

    # Lead time: when preemptive detects a transition before reactive
    lead_times: List[int] = []
    false_alarms = 0
    for i in range(1, n):
        if pre_regimes[i] != pre_regimes[i-1]:
            # Find when reactive catches up
            found = False
            for j in range(i, min(i+20, n)):
                if react_regimes[j] == pre_regimes[i]:
                    lead_times.append(j - i)
                    found = True
                    break
            if not found:
                false_alarms += 1

    # PnL simulation: regime-dependent returns
    regime_mult = {"bull": 1.0, "sideways": 0.5, "bear": -0.3, "crisis": -1.0}

    pre_pnl = 0.0
    react_pnl = 0.0
    for i in range(n):
        # Use 5d forward return proxy from observations
        base_ret = observations[i].ret_20d / 20  # daily proxy

        pre_scale = regime_mult.get(pre_regimes[i], 0.5)
        react_scale = regime_mult.get(react_regimes[i], 0.5)

        pre_pnl += base_ret * max(0, pre_scale)
        react_pnl += base_ret * max(0, react_scale)

    total_warnings = sum(1 for w in warnings if w.signal != "stable")
    far = false_alarms / max(total_warnings, 1)
    improvement = (pre_pnl - react_pnl) / max(abs(react_pnl), 0.001) * 100

    return SwitchBacktest(
        n, react_trans, pre_trans,
        round(_mean([float(l) for l in lead_times]), 1) if lead_times else 0,
        round(far, 3),
        round(pre_pnl, 4), round(react_pnl, 4),
        round(improvement, 1),
    )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

@dataclass
class TransitionAnalysis:
    n_days: int
    warnings: List[EarlyWarning]
    regimes: List[str]
    backtest: SwitchBacktest
    n_imminent_warnings: int
    n_warn_warnings: int
    avg_days_in_regime: float
    regime_counts: Dict[str, int]
    current_regime: str
    current_days_in: int
    transition_prob_5d: float


class RegimeTransitionEngine:
    """Orchestrates HSMM transition prediction and analysis."""

    def __init__(self, observations: List[MarketObs]) -> None:
        self.observations = observations

    def analyse(self) -> TransitionAnalysis:
        detector = HSMMDetector(min_hold=3)
        warnings, regimes = generate_early_warnings(detector, self.observations)
        bt = backtest_switching(self.observations)

        n_imm = sum(1 for w in warnings if w.signal == "imminent_exit")
        n_warn = sum(1 for w in warnings if w.signal == "warn_exit")

        # Duration stats
        durations: List[int] = []
        if regimes:
            curr = regimes[0]; count = 1
            for i in range(1, len(regimes)):
                if regimes[i] == curr:
                    count += 1
                else:
                    durations.append(count)
                    curr = regimes[i]; count = 1
            durations.append(count)

        counts = dict(defaultdict(int))
        for r in regimes:
            counts[r] = counts.get(r, 0) + 1

        return TransitionAnalysis(
            n_days=len(self.observations),
            warnings=warnings, regimes=regimes, backtest=bt,
            n_imminent_warnings=n_imm, n_warn_warnings=n_warn,
            avg_days_in_regime=round(_mean([float(d) for d in durations]), 1) if durations else 0,
            regime_counts=counts,
            current_regime=detector.current_regime,
            current_days_in=detector.days_in_regime,
            transition_prob_5d=detector.transition_probability(5),
        )


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_test_data(n: int = 800, seed: int = 1440) -> List[MarketObs]:
    """Generate synthetic data with regime structure and durations."""
    rng = random.Random(seed)
    obs: List[MarketObs] = []
    phases = [
        (150, "bull",    0.03, 0.08, 0.12, 15, 300),
        (50,  "bear",   -0.04,-0.10, 0.20, 26, 460),
        (20,  "crisis", -0.15,-0.28, 0.50, 55, 750),
        (40,  "sideways",0.005,0.02, 0.09, 14, 320),
        (200, "bull",    0.03, 0.07, 0.11, 14, 290),
        (80,  "bear",   -0.03,-0.08, 0.22, 28, 500),
        (15,  "crisis", -0.12,-0.20, 0.40, 45, 680),
        (60,  "sideways",0.003,0.01, 0.08, 12, 310),
        (100, "bull",    0.02, 0.06, 0.10, 13, 300),
        (85,  "bear",   -0.03,-0.07, 0.19, 25, 440),
    ]
    for n_days, truth, r20, r60, rvol, vix_b, cs in phases:
        for _ in range(n_days):
            if len(obs) >= n: break
            obs.append(MarketObs(
                date=f"day-{len(obs)}",
                ret_20d=r20 + rng.gauss(0, abs(r20) * 0.3 + 0.005),
                ret_60d=r60 + rng.gauss(0, abs(r60) * 0.25 + 0.005),
                rvol_20d=max(0.04, rvol + rng.gauss(0, rvol * 0.2)),
                vix=max(9, vix_b + rng.gauss(0, vix_b * 0.15)),
                credit_spread_bps=max(100, cs + rng.gauss(0, cs * 0.1)),
                ground_truth=truth,
            ))
    return obs[:n]
