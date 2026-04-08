"""Ensemble meta-learner V2: gradient-boosted stacking of 10+ signal generators.

Collects outputs from all existing signal modules, trains a gradient-
boosted meta-learner (pure-Python decision stumps), and outputs a
unified signal with confidence.  Walk-forward validated.

Pure-Python — no sklearn/xgboost dependencies.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable


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

def _sigmoid(x: float) -> float:
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x); return ex / (1.0 + ex)


# ---------------------------------------------------------------------------
# Signal collection
# ---------------------------------------------------------------------------

SIGNAL_NAMES = [
    "base_ensemble",      # EXP-880 ML ensemble P(win)
    "regime_score",       # regime detector output (-1 to +1)
    "momentum_20d",       # 20-day price momentum
    "ofi_score",          # order flow imbalance
    "calendar_signal",    # earnings/FOMC calendar
    "sentiment_composite",# sentiment regime composite
    "microstructure",     # VPIN / Kyle lambda signal
    "vol_surface",        # IV term structure slope
    "tail_risk",          # tail risk indicator
    "vix_term",           # VIX contango/backwardation score
    "credit_spread",      # credit spread z-score
    "breadth",            # market breadth signal
]
N_SIGNALS = len(SIGNAL_NAMES)


@dataclass
class SignalVector:
    """One observation: all signal values + forward return label."""
    date: str
    signals: Dict[str, float]
    label: float               # forward 5-day return (regression target)
    label_binary: int          # 1 if positive, 0 if negative


# ---------------------------------------------------------------------------
# Gradient-boosted stump meta-learner
# ---------------------------------------------------------------------------

@dataclass
class _Stump:
    feature: str
    threshold: float
    left_val: float
    right_val: float

    def predict(self, signals: Dict[str, float]) -> float:
        return self.left_val if signals.get(self.feature, 0) <= self.threshold else self.right_val


@dataclass
class GBConfig:
    n_estimators: int = 80
    learning_rate: float = 0.08
    n_thresholds: int = 10
    subsample: float = 0.75
    seed: int = 42


class GBMetaLearner:
    """Gradient-boosted decision stumps trained on signal vectors."""

    def __init__(self, config: Optional[GBConfig] = None) -> None:
        self.config = config or GBConfig()
        self._stumps: List[_Stump] = []
        self._bias: float = 0.0
        self._importance: Dict[str, float] = {}
        self._trained = False

    def train(self, data: List[SignalVector]) -> None:
        if not data: return
        cfg = self.config
        rng = random.Random(cfg.seed)
        n = len(data)

        pos = sum(1 for d in data if d.label_binary == 1)
        neg = n - pos
        self._bias = math.log(pos / neg) if pos > 0 and neg > 0 else 0.0

        raw = [self._bias] * n
        self._stumps = []
        self._importance = {s: 0.0 for s in SIGNAL_NAMES}

        for _ in range(cfg.n_estimators):
            residuals = [d.label_binary - _sigmoid(raw[i]) for i, d in enumerate(data)]
            k = max(1, int(n * cfg.subsample))
            indices = rng.sample(range(n), k)

            best_stump = None
            best_loss = math.inf

            for feat in SIGNAL_NAMES:
                vals = [data[idx].signals.get(feat, 0) for idx in indices]
                lo, hi = min(vals), max(vals)
                if lo == hi: continue
                step = (hi - lo) / (cfg.n_thresholds + 1)

                for t in range(cfg.n_thresholds):
                    thresh = lo + step * (t + 1)
                    left_r, right_r = [], []
                    for idx in indices:
                        v = data[idx].signals.get(feat, 0)
                        (left_r if v <= thresh else right_r).append(residuals[idx])
                    if not left_r or not right_r: continue
                    lv = _mean(left_r); rv = _mean(right_r)
                    loss = sum((residuals[idx] - (lv if data[idx].signals.get(feat, 0) <= thresh else rv)) ** 2 for idx in indices)
                    if loss < best_loss:
                        best_loss = loss
                        best_stump = _Stump(feat, thresh, lv, rv)

            if best_stump is None:
                mr = _mean([residuals[i] for i in indices])
                best_stump = _Stump(SIGNAL_NAMES[0], 0, mr, mr)

            self._stumps.append(best_stump)
            for i in range(n):
                raw[i] += cfg.learning_rate * best_stump.predict(data[i].signals)
            self._importance[best_stump.feature] = self._importance.get(best_stump.feature, 0) + 1

        total = sum(self._importance.values())
        if total > 0:
            self._importance = {k: v / total for k, v in self._importance.items()}
        self._trained = True

    def predict_proba(self, signals: Dict[str, float]) -> float:
        if not self._trained: return 0.5
        raw = self._bias
        for stump in self._stumps:
            raw += self.config.learning_rate * stump.predict(signals)
        return _sigmoid(raw)

    def predict(self, signals: Dict[str, float]) -> int:
        return 1 if self.predict_proba(signals) >= 0.5 else 0

    def feature_importance(self) -> Dict[str, float]:
        return dict(self._importance)


# ---------------------------------------------------------------------------
# Simple average baseline
# ---------------------------------------------------------------------------

def simple_average_signal(signals: Dict[str, float]) -> float:
    """Equal-weight average of all signals, mapped to [0, 1]."""
    vals = [signals.get(s, 0) for s in SIGNAL_NAMES]
    avg = _mean(vals)
    return _sigmoid(avg * 2)  # scale and squash


def best_individual_signal(
    data: List[SignalVector],
) -> Tuple[str, float]:
    """Find the single signal with highest label correlation."""
    labels = [d.label for d in data]
    best_name = SIGNAL_NAMES[0]
    best_corr = -1.0
    for sig in SIGNAL_NAMES:
        vals = [d.signals.get(sig, 0) for d in data]
        c = abs(_correlation(vals, labels))
        if c > best_corr:
            best_corr = c
            best_name = sig
    return best_name, round(best_corr, 4)


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

@dataclass
class WFFold:
    fold: int
    train_size: int
    test_size: int
    meta_auc: float
    avg_auc: float
    best_single_auc: float
    meta_accuracy: float


@dataclass
class WFResult:
    n_folds: int
    folds: List[WFFold]
    meta_mean_auc: float
    avg_mean_auc: float
    best_single_mean_auc: float
    meta_lift_vs_avg: float      # pct improvement
    meta_lift_vs_best: float


def _compute_auc(labels: List[int], probs: List[float]) -> float:
    if not labels: return 0.5
    pairs = sorted(zip(probs, labels))
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    cum_neg = 0; auc_sum = 0.0
    for _, lab in pairs:
        if lab == 0: cum_neg += 1
        else: auc_sum += cum_neg
    return auc_sum / (n_pos * n_neg)


def walk_forward_validate(
    data: List[SignalVector],
    n_folds: int = 5,
    config: Optional[GBConfig] = None,
) -> WFResult:
    """Walk-forward validation: expanding window."""
    n = len(data)
    fold_size = n // (n_folds + 1)
    folds: List[WFFold] = []

    for i in range(n_folds):
        train_end = fold_size + fold_size * i
        test_end = min(train_end + fold_size, n)
        if train_end >= n or test_end <= train_end: break

        train = data[:train_end]
        test = data[train_end:test_end]
        if len(test) < 10: break

        # Meta-learner
        ml = GBMetaLearner(config)
        ml.train(train)
        meta_probs = [ml.predict_proba(d.signals) for d in test]
        labels = [d.label_binary for d in test]
        meta_auc = _compute_auc(labels, meta_probs)

        # Simple average
        avg_probs = [simple_average_signal(d.signals) for d in test]
        avg_auc = _compute_auc(labels, avg_probs)

        # Best single
        best_sig, _ = best_individual_signal(train)
        best_probs = [_sigmoid(d.signals.get(best_sig, 0) * 2) for d in test]
        best_auc = _compute_auc(labels, best_probs)

        meta_acc = _mean([1.0 if (p >= 0.5) == (l == 1) else 0.0 for p, l in zip(meta_probs, labels)])

        folds.append(WFFold(i+1, len(train), len(test),
                             round(meta_auc, 4), round(avg_auc, 4),
                             round(best_auc, 4), round(meta_acc, 4)))

    meta_aucs = [f.meta_auc for f in folds]
    avg_aucs = [f.avg_auc for f in folds]
    best_aucs = [f.best_single_auc for f in folds]

    m_meta = _mean(meta_aucs)
    m_avg = _mean(avg_aucs)
    m_best = _mean(best_aucs)

    lift_avg = (m_meta - m_avg) / max(abs(m_avg), 0.001) * 100
    lift_best = (m_meta - m_best) / max(abs(m_best), 0.001) * 100

    return WFResult(len(folds), folds, round(m_meta, 4), round(m_avg, 4),
                     round(m_best, 4), round(lift_avg, 1), round(lift_best, 1))


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

@dataclass
class MetaLearnerResult:
    n_samples: int
    n_signals: int
    wf: WFResult
    feature_importance: Dict[str, float]
    top_5_signals: List[Tuple[str, float]]
    best_individual: Tuple[str, float]
    meta_beats_average: bool
    meta_beats_best_individual: bool


class MetaLearnerEngine:
    """Orchestrates meta-learner training, validation, and analysis."""

    def __init__(self, data: List[SignalVector], config: Optional[GBConfig] = None) -> None:
        self.data = data
        self.config = config or GBConfig()

    def analyse(self) -> MetaLearnerResult:
        wf = walk_forward_validate(self.data, config=self.config)

        # Train final model on all data for feature importance
        ml = GBMetaLearner(self.config)
        ml.train(self.data)
        fi = ml.feature_importance()
        top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]

        best_sig, best_corr = best_individual_signal(self.data)

        return MetaLearnerResult(
            n_samples=len(self.data), n_signals=N_SIGNALS,
            wf=wf, feature_importance=fi, top_5_signals=top5,
            best_individual=(best_sig, best_corr),
            meta_beats_average=wf.meta_mean_auc > wf.avg_mean_auc,
            meta_beats_best_individual=wf.meta_mean_auc > wf.best_single_mean_auc,
        )


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_test_data(n: int = 600, seed: int = 1340) -> List[SignalVector]:
    """Generate synthetic signal vectors with known importance structure."""
    rng = random.Random(seed)
    data: List[SignalVector] = []

    for i in range(n):
        signals = {}
        # Base ensemble is the most predictive
        base = rng.gauss(0, 1)
        signals["base_ensemble"] = base

        # Regime and sentiment have moderate predictive power
        regime = rng.gauss(0, 0.8)
        sentiment = rng.gauss(0, 0.7)
        signals["regime_score"] = regime
        signals["sentiment_composite"] = sentiment

        # Others have weak to moderate signal
        for sig in SIGNAL_NAMES:
            if sig not in signals:
                signals[sig] = rng.gauss(0, 0.5)

        # Label: weighted combo + noise
        true_signal = (0.4 * base + 0.2 * regime + 0.15 * sentiment +
                       0.1 * signals["vix_term"] + 0.05 * signals["tail_risk"])
        label = true_signal + rng.gauss(0, 0.8)
        label_binary = 1 if label > 0 else 0

        yr = 2020 + i // 120
        data.append(SignalVector(f"{yr}-{(i%12)+1:02d}-{(i%28)+1:02d}",
                                  signals, round(label, 4), label_binary))

    return data
