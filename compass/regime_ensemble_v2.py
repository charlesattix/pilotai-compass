"""Adaptive Regime Ensemble V2: meta-ensemble of 4 regime detectors.

Four independent detectors (HMM, rule-based, clustering, Markov-switching)
each output {bull, bear, sideways, crisis} with probabilities.  A
meta-learner (stacking or weighted voting) combines them.  Measures
accuracy, false alarm rate, and transition detection latency.

Pure-Python — no numpy/pandas dependencies.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

REGIMES = ["bull", "bear", "sideways", "crisis"]
N_REGIMES = len(REGIMES)

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
# Observation
# ---------------------------------------------------------------------------

@dataclass
class MarketObs:
    """Market observation for regime detection."""
    date: str
    ret_20d: float = 0.0
    ret_60d: float = 0.0
    rvol_20d: float = 0.12
    vix: float = 15.0
    yield_curve: float = 0.5
    credit_spread_bps: float = 350.0
    ground_truth: Optional[str] = None


# ---------------------------------------------------------------------------
# Detector 1: HMM with Gaussian emissions
# ---------------------------------------------------------------------------

# Emission params per regime per feature: (mean, std)
_HMM_EMIT = {
    "bull":    [(0.03,0.04),(0.08,0.06),(0.12,0.04),(15,4),(1.0,0.6),(300,80)],
    "bear":    [(-0.04,0.04),(-0.10,0.06),(0.18,0.05),(24,5),(0.2,0.7),(450,120)],
    "sideways":[(0.005,0.02),(0.02,0.03),(0.09,0.03),(13,3),(0.7,0.5),(310,70)],
    "crisis":  [(-0.15,0.08),(-0.25,0.10),(0.45,0.15),(50,15),(-0.1,1.0),(700,200)],
}
_HMM_TRANS = [
    [0.92,0.03,0.04,0.01],  # bull→
    [0.04,0.88,0.05,0.03],  # bear→
    [0.08,0.04,0.86,0.02],  # sideways→
    [0.02,0.06,0.10,0.82],  # crisis→
]


class HMMDetector:
    """Forward-pass HMM regime detector."""

    def __init__(self) -> None:
        self._log_belief = [-math.log(N_REGIMES)] * N_REGIMES
        self._log_trans = [[math.log(max(p,1e-15)) for p in row] for row in _HMM_TRANS]

    def predict(self, obs: MarketObs) -> Dict[str, float]:
        feats = [obs.ret_20d, obs.ret_60d, obs.rvol_20d, obs.vix, obs.yield_curve, obs.credit_spread_bps]
        # Predict
        log_pred = []
        for j in range(N_REGIMES):
            terms = [self._log_belief[i] + self._log_trans[i][j] for i in range(N_REGIMES)]
            log_pred.append(_logsumexp(terms))
        # Emit
        log_up = []
        for j in range(N_REGIMES):
            r = REGIMES[j]
            le = sum(_log_gauss(feats[k], _HMM_EMIT[r][k][0], _HMM_EMIT[r][k][1]) for k in range(len(feats)))
            log_up.append(log_pred[j] + le)
        lse = _logsumexp(log_up)
        self._log_belief = [v - lse for v in log_up]
        return {REGIMES[i]: math.exp(self._log_belief[i]) for i in range(N_REGIMES)}

    def reset(self):
        self._log_belief = [-math.log(N_REGIMES)] * N_REGIMES


# ---------------------------------------------------------------------------
# Detector 2: Rule-based (VIX/trend thresholds)
# ---------------------------------------------------------------------------

class RuleDetector:
    """Threshold-based regime detector."""

    def predict(self, obs: MarketObs) -> Dict[str, float]:
        scores = {r: 0.1 for r in REGIMES}
        if obs.vix >= 35 and obs.ret_20d <= -0.10:
            scores["crisis"] += 3.0
        if obs.vix >= 28 or obs.rvol_20d >= 0.25:
            scores["crisis"] += 1.0; scores["bear"] += 0.5
        if obs.ret_60d >= 0.03:
            scores["bull"] += 2.0
        if obs.ret_60d <= -0.03:
            scores["bear"] += 2.0
        if abs(obs.ret_60d) < 0.02 and obs.vix < 18:
            scores["sideways"] += 2.0
        if obs.yield_curve < 0:
            scores["bear"] += 0.5; scores["crisis"] += 0.3
        if obs.credit_spread_bps > 500:
            scores["crisis"] += 1.0
        total = sum(scores.values())
        return {r: scores[r] / total for r in REGIMES}

    def reset(self): pass


# ---------------------------------------------------------------------------
# Detector 3: Volatility clustering
# ---------------------------------------------------------------------------

class VolClusterDetector:
    """Classifies regime from rolling volatility quantiles."""

    def __init__(self) -> None:
        self._vol_history: List[float] = []

    def predict(self, obs: MarketObs) -> Dict[str, float]:
        self._vol_history.append(obs.rvol_20d)
        if len(self._vol_history) > 252:
            self._vol_history = self._vol_history[-252:]

        vol = obs.rvol_20d
        if len(self._vol_history) >= 20:
            sorted_h = sorted(self._vol_history)
            rank = sum(1 for v in sorted_h if v <= vol) / len(sorted_h)
        else:
            rank = 0.5

        probs = {r: 0.1 for r in REGIMES}
        if rank >= 0.90:
            probs["crisis"] += 0.6; probs["bear"] += 0.2
        elif rank >= 0.70:
            probs["bear"] += 0.4; probs["crisis"] += 0.1
        elif rank <= 0.20:
            probs["sideways"] += 0.5; probs["bull"] += 0.2
        else:
            probs["bull"] += 0.3; probs["sideways"] += 0.2

        # Add trend tilt
        if obs.ret_60d > 0.02:
            probs["bull"] += 0.3
        elif obs.ret_60d < -0.02:
            probs["bear"] += 0.3

        total = sum(probs.values())
        return {r: probs[r] / total for r in REGIMES}

    def reset(self):
        self._vol_history = []


# ---------------------------------------------------------------------------
# Detector 4: Markov-switching (simplified)
# ---------------------------------------------------------------------------

class MarkovSwitchDetector:
    """Simplified Markov-switching model using return distribution."""

    def __init__(self) -> None:
        self._state_probs = [1/N_REGIMES] * N_REGIMES
        self._trans = [list(row) for row in _HMM_TRANS]
        # State-conditional return distributions
        self._ret_params = {
            "bull": (0.001, 0.008),
            "bear": (-0.001, 0.012),
            "sideways": (0.0002, 0.005),
            "crisis": (-0.005, 0.025),
        }

    def predict(self, obs: MarketObs) -> Dict[str, float]:
        # Daily return proxy from 20d return
        daily_ret = obs.ret_20d / 20

        # Predict step
        new_probs = [0.0] * N_REGIMES
        for j in range(N_REGIMES):
            for i in range(N_REGIMES):
                new_probs[j] += self._state_probs[i] * self._trans[i][j]

        # Emit step: likelihood of observed return given each state
        likelihoods = []
        for j, r in enumerate(REGIMES):
            mu, sigma = self._ret_params[r]
            ll = math.exp(_log_gauss(daily_ret, mu, sigma))
            likelihoods.append(ll)

        # Update
        updated = [new_probs[j] * likelihoods[j] for j in range(N_REGIMES)]
        total = sum(updated)
        if total > 0:
            self._state_probs = [u / total for u in updated]
        else:
            self._state_probs = [1/N_REGIMES] * N_REGIMES

        return {REGIMES[i]: self._state_probs[i] for i in range(N_REGIMES)}

    def reset(self):
        self._state_probs = [1/N_REGIMES] * N_REGIMES


# ---------------------------------------------------------------------------
# Meta-learner: weighted voting / stacking
# ---------------------------------------------------------------------------

DETECTOR_NAMES = ["hmm", "rules", "vol_cluster", "markov_switch"]

@dataclass
class EnsembleOutput:
    """One-day ensemble output."""
    date: str
    consensus: str
    confidence: float
    probabilities: Dict[str, float]
    detector_votes: Dict[str, str]
    detector_probs: Dict[str, Dict[str, float]]
    agreement_rate: float
    is_disagreement: bool


class MetaLearner:
    """Combines 4 detector outputs via weighted voting or stacking."""

    def __init__(
        self,
        method: str = "weighted_vote",
        weights: Optional[Dict[str, float]] = None,
        min_hold: int = 3,
    ) -> None:
        self.method = method
        self.weights = weights or {"hmm": 0.35, "rules": 0.20, "vol_cluster": 0.20, "markov_switch": 0.25}
        self.min_hold = min_hold
        self._current = "bull"
        self._hold_counter = 0
        # For stacking: learned regime-conditional detector accuracy
        self._detector_accuracy: Dict[str, Dict[str, float]] = {}

    def combine(
        self,
        date: str,
        detector_probs: Dict[str, Dict[str, float]],
    ) -> EnsembleOutput:
        """Combine detector outputs into ensemble consensus."""
        # Weighted average probabilities
        combined = {r: 0.0 for r in REGIMES}
        for det, probs in detector_probs.items():
            w = self.weights.get(det, 0.25)
            for r in REGIMES:
                combined[r] += w * probs.get(r, 0.0)

        # Normalise
        total = sum(combined.values())
        if total > 0:
            combined = {r: p / total for r, p in combined.items()}

        # MAP
        map_regime = max(combined, key=combined.get)
        confidence = combined[map_regime]

        # Per-detector votes (argmax)
        votes = {}
        for det, probs in detector_probs.items():
            votes[det] = max(probs, key=probs.get)

        # Agreement
        vote_values = list(votes.values())
        most_common = Counter(vote_values).most_common(1)[0]
        agreement = most_common[1] / len(vote_values) if vote_values else 0

        # Min-hold
        if map_regime != self._current:
            if self._hold_counter < self.min_hold:
                map_regime = self._current
            else:
                self._current = map_regime
                self._hold_counter = 0
        self._hold_counter += 1

        return EnsembleOutput(
            date=date, consensus=self._current, confidence=round(confidence, 4),
            probabilities=combined, detector_votes=votes,
            detector_probs=detector_probs,
            agreement_rate=round(agreement, 3),
            is_disagreement=agreement < 0.5,
        )

    def update_accuracy(self, detector: str, regime: str, correct: bool):
        """Online accuracy tracking for stacking weights."""
        if detector not in self._detector_accuracy:
            self._detector_accuracy[detector] = {r: [] for r in REGIMES}
        self._detector_accuracy[detector].setdefault(regime, []).append(1.0 if correct else 0.0)

    def adapt_weights(self):
        """Adjust weights based on recent accuracy (adaptive stacking)."""
        for det in DETECTOR_NAMES:
            accs = self._detector_accuracy.get(det, {})
            all_acc = []
            for r_accs in accs.values():
                all_acc.extend(r_accs[-50:])  # last 50 observations
            if all_acc:
                self.weights[det] = max(0.05, _mean(all_acc))
        # Normalise
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def reset(self):
        self._current = "bull"
        self._hold_counter = 0


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

@dataclass
class DetectorScore:
    name: str
    accuracy: float
    false_alarm_rate: float     # classified crisis when not crisis
    miss_rate: float            # failed to detect crisis when crisis
    n_transitions: int
    mean_latency: float         # avg days late to detect transitions
    per_regime_accuracy: Dict[str, float]


@dataclass
class ComparisonResult:
    n_days: int
    ensemble_score: DetectorScore
    individual_scores: List[DetectorScore]
    ensemble_regimes: List[str]
    agreement_timeline: List[float]
    whipsaw_reduction_pct: float
    best_detector: str


def _score_detector(
    name: str,
    predicted: List[str],
    truth: List[str],
) -> DetectorScore:
    """Score one detector against ground truth."""
    n = min(len(predicted), len(truth))
    if n == 0:
        return DetectorScore(name, 0, 0, 0, 0, 0, {})

    correct = sum(1 for i in range(n) if predicted[i] == truth[i])
    accuracy = correct / n

    # False alarm: predicted crisis when truth != crisis
    fa = sum(1 for i in range(n) if predicted[i] == "crisis" and truth[i] != "crisis")
    non_crisis = sum(1 for i in range(n) if truth[i] != "crisis")
    far = fa / non_crisis if non_crisis > 0 else 0

    # Miss: truth=crisis but predicted != crisis
    miss = sum(1 for i in range(n) if truth[i] == "crisis" and predicted[i] != "crisis")
    crisis_days = sum(1 for i in range(n) if truth[i] == "crisis")
    mr = miss / crisis_days if crisis_days > 0 else 0

    trans = sum(1 for i in range(1, n) if predicted[i] != predicted[i-1])

    # Transition latency: how late did detector find truth transitions?
    latencies: List[int] = []
    for i in range(1, n):
        if truth[i] != truth[i-1]:
            # Find when detector also transitions
            for j in range(i, min(i+20, n)):
                if predicted[j] == truth[i]:
                    latencies.append(j - i)
                    break

    # Per-regime accuracy
    pr: Dict[str, Dict[str, int]] = {}
    for i in range(n):
        gt = truth[i]
        pr.setdefault(gt, {"correct": 0, "total": 0})
        pr[gt]["total"] += 1
        if predicted[i] == gt:
            pr[gt]["correct"] += 1
    pr_acc = {r: d["correct"]/d["total"] for r, d in pr.items() if d["total"] > 0}

    return DetectorScore(
        name, round(accuracy, 4), round(far, 4), round(mr, 4),
        trans, round(_mean([float(l) for l in latencies]), 1) if latencies else 0.0,
        pr_acc,
    )


# ---------------------------------------------------------------------------
# Full engine
# ---------------------------------------------------------------------------

class RegimeEnsembleV2:
    """Adaptive 4-detector regime ensemble."""

    def __init__(
        self,
        method: str = "weighted_vote",
        weights: Optional[Dict[str, float]] = None,
        min_hold: int = 3,
    ) -> None:
        self.hmm = HMMDetector()
        self.rules = RuleDetector()
        self.vol_cluster = VolClusterDetector()
        self.markov = MarkovSwitchDetector()
        self.meta = MetaLearner(method, weights, min_hold)
        self._detectors = {"hmm": self.hmm, "rules": self.rules,
                           "vol_cluster": self.vol_cluster, "markov_switch": self.markov}

    def classify(self, obs: MarketObs) -> EnsembleOutput:
        """Classify one observation through all detectors + meta-learner."""
        det_probs = {}
        for name, det in self._detectors.items():
            det_probs[name] = det.predict(obs)
        return self.meta.combine(obs.date, det_probs)

    def classify_series(
        self,
        observations: List[MarketObs],
    ) -> Tuple[List[EnsembleOutput], ComparisonResult]:
        """Classify full series and produce comparison metrics."""
        outputs: List[EnsembleOutput] = []
        per_det_regimes: Dict[str, List[str]] = {n: [] for n in DETECTOR_NAMES}

        for obs in observations:
            out = self.classify(obs)
            outputs.append(out)
            for det_name, vote in out.detector_votes.items():
                per_det_regimes[det_name].append(vote)
            # Online accuracy update if ground truth available
            if obs.ground_truth:
                for det_name, vote in out.detector_votes.items():
                    self.meta.update_accuracy(det_name, obs.ground_truth, vote == obs.ground_truth)

        ensemble_regimes = [o.consensus for o in outputs]
        truth = [obs.ground_truth or "bull" for obs in observations]
        has_truth = any(obs.ground_truth for obs in observations)

        # Score
        ens_score = _score_detector("ensemble", ensemble_regimes, truth) if has_truth else DetectorScore("ensemble",0,0,0,0,0,{})
        ind_scores = []
        for name in DETECTOR_NAMES:
            sc = _score_detector(name, per_det_regimes[name], truth) if has_truth else DetectorScore(name,0,0,0,0,0,{})
            ind_scores.append(sc)

        agreements = [o.agreement_rate for o in outputs]
        max_ind_trans = max((s.n_transitions for s in ind_scores), default=1)
        whipsaw_red = (max_ind_trans - ens_score.n_transitions) / max_ind_trans * 100 if max_ind_trans > 0 else 0

        best = max(ind_scores, key=lambda s: s.accuracy).name if ind_scores else "ensemble"

        return outputs, ComparisonResult(
            n_days=len(observations),
            ensemble_score=ens_score,
            individual_scores=ind_scores,
            ensemble_regimes=ensemble_regimes,
            agreement_timeline=agreements,
            whipsaw_reduction_pct=round(whipsaw_red, 1),
            best_detector=best,
        )

    def reset(self):
        self.hmm.reset(); self.rules.reset()
        self.vol_cluster.reset(); self.markov.reset()
        self.meta.reset()


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_test_data(n: int = 1000, seed: int = 1130) -> List[MarketObs]:
    """Generate synthetic market observations with ground truth."""
    rng = random.Random(seed)
    obs: List[MarketObs] = []
    phases = [
        (200, "bull",    0.03, 0.08, 0.12, 15, 1.0, 300),
        (80,  "bear",   -0.04,-0.10, 0.20, 26, 0.2, 460),
        (30,  "crisis", -0.15,-0.28, 0.50, 55,-0.1, 750),
        (60,  "sideways",0.005,0.02, 0.09, 13, 0.7, 310),
        (250, "bull",    0.03, 0.07, 0.11, 14, 0.9, 290),
        (100, "bear",   -0.03,-0.08, 0.22, 28,-0.2, 500),
        (30,  "crisis", -0.12,-0.20, 0.40, 45, 0.0, 680),
        (100, "sideways",0.003,0.01, 0.08, 12, 0.6, 300),
        (150, "bull",    0.02, 0.06, 0.10, 13, 0.8, 310),
    ]
    for n_days, truth, r20, r60, rvol, vix_b, yc, cs in phases:
        for d in range(n_days):
            if len(obs) >= n:
                break
            obs.append(MarketObs(
                date=f"day-{len(obs)}",
                ret_20d=r20 + rng.gauss(0, abs(r20)*0.3 + 0.005),
                ret_60d=r60 + rng.gauss(0, abs(r60)*0.25 + 0.005),
                rvol_20d=max(0.04, rvol + rng.gauss(0, rvol*0.2)),
                vix=max(9, vix_b + rng.gauss(0, vix_b*0.15)),
                yield_curve=yc + rng.gauss(0, 0.2),
                credit_spread_bps=max(100, cs + rng.gauss(0, cs*0.1)),
                ground_truth=truth,
            ))
    return obs[:n]
