"""Feature importance, signal decay, and redundancy analysis.

Pure-Python implementations of SHAP-style importance (via tree-path
approximation), permutation importance, feature interaction (H-statistic
proxy), signal half-life (autocorrelation decay), feature clustering
(correlation-based), and sequential feature selection.

No numpy/sklearn dependencies.
"""

from __future__ import annotations

import html as html_mod
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

def _autocorrelation(xs: List[float], lag: int) -> float:
    """Autocorrelation at given lag."""
    if lag >= len(xs) or len(xs) < lag + 3: return 0.0
    return _correlation(xs[:len(xs)-lag], xs[lag:])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """One observation: feature dict + label."""
    features: Dict[str, float]
    label: float  # target (return or binary)


@dataclass
class FeatureImportance:
    """Importance score for one feature."""
    name: str
    shap_importance: float      # mean |SHAP| (tree-path approx)
    permutation_importance: float
    interaction_score: float    # H-statistic proxy
    rank: int


# ---------------------------------------------------------------------------
# 1. SHAP-style importance (tree-path approximation)
# ---------------------------------------------------------------------------

def compute_shap_importance(
    data: List[FeatureVector],
    n_samples: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """Approximate SHAP values via marginal contribution sampling.

    For each feature, measure how much the prediction changes when that
    feature is replaced by a random value from the dataset (marginal
    contribution). This approximates TreeExplainer without needing trees.
    """
    if not data or not data[0].features:
        return {}

    rng = random.Random(seed)
    features = list(data[0].features.keys())
    n = len(data)
    importance: Dict[str, List[float]] = {f: [] for f in features}

    for _ in range(n_samples):
        idx = rng.randint(0, n - 1)
        obs = data[idx]
        base_pred = _simple_predict(obs.features, data)

        for feat in features:
            # Replace this feature with a random other observation's value
            donor_idx = rng.randint(0, n - 1)
            perturbed = dict(obs.features)
            perturbed[feat] = data[donor_idx].features[feat]
            new_pred = _simple_predict(perturbed, data)
            contribution = abs(base_pred - new_pred)
            importance[feat].append(contribution)

    result = {f: _mean(vals) for f, vals in importance.items()}
    # Normalise
    total = sum(result.values())
    if total > 0:
        result = {f: v / total for f, v in result.items()}
    return result


def _simple_predict(features: Dict[str, float], data: List[FeatureVector]) -> float:
    """Simple k-NN prediction (k=5) for SHAP approximation."""
    if not data: return 0.0
    # Distance = sum of squared normalised differences
    distances: List[Tuple[float, float]] = []
    for obs in data[:200]:  # limit for speed
        dist = sum((features.get(f, 0) - obs.features.get(f, 0)) ** 2
                    for f in features)
        distances.append((dist, obs.label))
    distances.sort(key=lambda x: x[0])
    k = min(5, len(distances))
    return _mean([d[1] for d in distances[:k]])


# ---------------------------------------------------------------------------
# 2. Permutation importance
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    data: List[FeatureVector],
    n_repeats: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """Permutation importance: how much does shuffling each feature hurt?"""
    if not data or not data[0].features:
        return {}

    rng = random.Random(seed)
    features = list(data[0].features.keys())
    labels = [d.label for d in data]
    n = len(data)

    # Baseline: correlation between each feature and label
    baseline_score = _mean([abs(_correlation(
        [d.features[f] for d in data], labels)) for f in features])

    importance: Dict[str, float] = {}
    for feat in features:
        drops: List[float] = []
        for _ in range(n_repeats):
            # Shuffle this feature
            shuffled_vals = [d.features[feat] for d in data]
            rng.shuffle(shuffled_vals)
            # Measure degradation in predictive power
            shuffled_corr = abs(_correlation(shuffled_vals, labels))
            original_corr = abs(_correlation([d.features[feat] for d in data], labels))
            drop = original_corr - shuffled_corr
            drops.append(max(0, drop))
        importance[feat] = _mean(drops)

    # Normalise
    total = sum(importance.values())
    if total > 0:
        importance = {f: v / total for f, v in importance.items()}
    return importance


# ---------------------------------------------------------------------------
# 3. Feature interaction (H-statistic proxy)
# ---------------------------------------------------------------------------

def compute_interaction_scores(
    data: List[FeatureVector],
    top_n: int = 10,
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[Tuple[str, str], float]:
    """H-statistic proxy: measure pairwise feature interactions.

    For each pair, check if their joint effect on the label differs
    from the sum of their individual effects.
    """
    if not data or not data[0].features:
        return {}

    rng = random.Random(seed)
    features = list(data[0].features.keys())[:top_n]
    labels = [d.label for d in data]
    interactions: Dict[Tuple[str, str], float] = {}

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            v1 = [d.features[f1] for d in data]
            v2 = [d.features[f2] for d in data]

            # Individual correlations
            c1 = abs(_correlation(v1, labels))
            c2 = abs(_correlation(v2, labels))

            # Joint: product interaction
            joint = [v1[k] * v2[k] for k in range(len(data))]
            c_joint = abs(_correlation(joint, labels))

            # H = joint beyond sum of individuals
            h = max(0, c_joint - (c1 + c2) * 0.5)
            interactions[(f1, f2)] = round(h, 6)

    return interactions


# ---------------------------------------------------------------------------
# 4. Signal half-life (autocorrelation decay)
# ---------------------------------------------------------------------------

@dataclass
class SignalHalfLife:
    """Half-life of a feature's predictive signal."""
    feature: str
    half_life_periods: float  # periods until autocorrelation drops to 0.5
    autocorrelations: Dict[int, float]  # lag → autocorrelation
    decay_rate: float         # exponential decay rate


def compute_signal_half_lives(
    data: List[FeatureVector],
    lags: Optional[List[int]] = None,
) -> List[SignalHalfLife]:
    """Measure how quickly each feature's signal decays via autocorrelation."""
    if not data or not data[0].features:
        return []

    if lags is None:
        lags = [1, 2, 5, 10, 20, 40, 60]

    features = list(data[0].features.keys())
    results: List[SignalHalfLife] = []

    for feat in features:
        values = [d.features[feat] for d in data]
        autocorrs: Dict[int, float] = {}

        for lag in lags:
            ac = _autocorrelation(values, lag)
            autocorrs[lag] = round(ac, 4)

        # Estimate half-life: find where autocorrelation drops below 0.5
        half_life = float('inf')
        for lag in sorted(autocorrs.keys()):
            if autocorrs[lag] < 0.5:
                # Interpolate
                prev_lag = max(l for l in autocorrs if l < lag) if any(l < lag for l in autocorrs) else 0
                prev_ac = autocorrs.get(prev_lag, 1.0)
                if prev_ac > 0.5 and prev_ac != autocorrs[lag]:
                    frac = (prev_ac - 0.5) / (prev_ac - autocorrs[lag])
                    half_life = prev_lag + frac * (lag - prev_lag)
                else:
                    half_life = float(lag)
                break

        # Decay rate from first two autocorrelations
        ac1 = autocorrs.get(1, 0.5)
        decay_rate = -math.log(max(ac1, 0.01)) if ac1 > 0 else 0.0

        results.append(SignalHalfLife(feat, round(half_life, 1), autocorrs, round(decay_rate, 4)))

    results.sort(key=lambda x: x.half_life_periods)
    return results


# ---------------------------------------------------------------------------
# 5. Feature redundancy (correlation clustering)
# ---------------------------------------------------------------------------

@dataclass
class FeatureCluster:
    """Cluster of correlated features."""
    cluster_id: int
    features: List[str]
    representative: str  # least-redundant member
    avg_intra_correlation: float


def cluster_features(
    data: List[FeatureVector],
    threshold: float = 0.70,
) -> List[FeatureCluster]:
    """Cluster features by pairwise correlation (single-linkage)."""
    if not data or not data[0].features:
        return []

    features = list(data[0].features.keys())
    n = len(features)
    labels = [d.label for d in data]

    # Compute pairwise correlations
    values = {f: [d.features[f] for d in data] for f in features}

    # Union-Find clustering
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            corr = abs(_correlation(values[features[i]], values[features[j]]))
            if corr >= threshold:
                union(i, j)

    # Build clusters
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters: List[FeatureCluster] = []
    for cid, (root, members) in enumerate(groups.items()):
        feat_names = [features[i] for i in members]

        # Representative: highest correlation with label
        label_corrs = {f: abs(_correlation(values[f], labels)) for f in feat_names}
        rep = max(label_corrs, key=label_corrs.get)

        # Average intra-cluster correlation
        if len(members) > 1:
            intra = []
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    intra.append(abs(_correlation(values[features[members[i]]], values[features[members[j]]])))
            avg_intra = _mean(intra)
        else:
            avg_intra = 1.0

        clusters.append(FeatureCluster(cid, feat_names, rep, round(avg_intra, 3)))

    clusters.sort(key=lambda c: -len(c.features))
    return clusters


# ---------------------------------------------------------------------------
# 6. Regime-conditional importance
# ---------------------------------------------------------------------------

@dataclass
class RegimeImportance:
    """Feature importance within one regime."""
    regime: str
    n_samples: int
    importance: Dict[str, float]
    top_features: List[Tuple[str, float]]


def compute_regime_importance(
    data: List[FeatureVector],
    regimes: List[str],
    top_n: int = 10,
) -> List[RegimeImportance]:
    """Compute feature importance conditioned on each regime."""
    by_regime: Dict[str, List[FeatureVector]] = defaultdict(list)
    for fv, regime in zip(data, regimes):
        by_regime[regime].append(fv)

    results: List[RegimeImportance] = []
    for regime, regime_data in sorted(by_regime.items()):
        if len(regime_data) < 20:
            continue
        imp = compute_permutation_importance(regime_data, n_repeats=5)
        top = sorted(imp.items(), key=lambda x: -x[1])[:top_n]
        results.append(RegimeImportance(regime, len(regime_data), imp, top))
    return results


# ---------------------------------------------------------------------------
# 7. Sequential feature selection
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """Result of sequential feature selection."""
    selected: List[str]
    scores: List[Tuple[str, float]]  # (feature_added, score_after)
    best_n_features: int
    best_score: float


def sequential_forward_selection(
    data: List[FeatureVector],
    max_features: int = 15,
    seed: int = 42,
) -> SelectionResult:
    """Greedy forward selection maximising label correlation."""
    if not data or not data[0].features:
        return SelectionResult([], [], 0, 0)

    all_features = list(data[0].features.keys())
    labels = [d.label for d in data]
    selected: List[str] = []
    scores: List[Tuple[str, float]] = []
    remaining = set(all_features)

    best_score = 0.0
    best_n = 0

    for step in range(min(max_features, len(all_features))):
        best_feat = None
        best_step_score = -1.0

        for feat in remaining:
            # Score = avg |correlation| of selected+feat combo with label
            combo = selected + [feat]
            combo_values = [_mean([d.features[f] for f in combo]) for d in data]
            score = abs(_correlation(combo_values, labels))
            if score > best_step_score:
                best_step_score = score
                best_feat = feat

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        scores.append((best_feat, round(best_step_score, 4)))

        if best_step_score > best_score:
            best_score = best_step_score
            best_n = len(selected)

    return SelectionResult(selected, scores, best_n, round(best_score, 4))


# ---------------------------------------------------------------------------
# Full analysis result
# ---------------------------------------------------------------------------

@dataclass
class FeatureAnalysisResult:
    n_features: int
    n_samples: int
    shap_importance: Dict[str, float]
    permutation_importance: Dict[str, float]
    top_features: List[FeatureImportance]
    interactions: Dict[Tuple[str, str], float]
    half_lives: List[SignalHalfLife]
    clusters: List[FeatureCluster]
    n_redundant: int
    regime_importance: List[RegimeImportance]
    selection: SelectionResult


class FeatureAnalyser:
    """Orchestrates all feature analysis."""

    def __init__(
        self,
        data: List[FeatureVector],
        regimes: Optional[List[str]] = None,
    ) -> None:
        self.data = data
        self.regimes = regimes or []

    def analyse(self) -> FeatureAnalysisResult:
        shap = compute_shap_importance(self.data)
        perm = compute_permutation_importance(self.data)
        interactions = compute_interaction_scores(self.data)
        half_lives = compute_signal_half_lives(self.data)
        clusters = cluster_features(self.data)
        selection = sequential_forward_selection(self.data)

        regime_imp = []
        if self.regimes and len(self.regimes) == len(self.data):
            regime_imp = compute_regime_importance(self.data, self.regimes)

        # Combined ranking
        features = list(shap.keys())
        combined: List[FeatureImportance] = []
        for i, f in enumerate(sorted(features, key=lambda f: -(shap.get(f,0) + perm.get(f,0)))):
            inter_score = max((v for (a, b), v in interactions.items() if a == f or b == f), default=0)
            combined.append(FeatureImportance(f, round(shap.get(f,0),4), round(perm.get(f,0),4),
                                               round(inter_score, 4), i + 1))

        n_redundant = sum(len(c.features) - 1 for c in clusters if len(c.features) > 1)

        return FeatureAnalysisResult(
            n_features=len(features), n_samples=len(self.data),
            shap_importance=shap, permutation_importance=perm,
            top_features=combined[:20], interactions=interactions,
            half_lives=half_lives, clusters=clusters,
            n_redundant=n_redundant, regime_importance=regime_imp,
            selection=selection,
        )


# ---------------------------------------------------------------------------
# Synthetic data for testing
# ---------------------------------------------------------------------------

def generate_test_data(
    n_samples: int = 500,
    n_features: int = 20,
    seed: int = 1180,
) -> Tuple[List[FeatureVector], List[str]]:
    """Generate synthetic feature data with known importance structure."""
    rng = random.Random(seed)
    feat_names = [f"feat_{i:02d}" for i in range(n_features)]
    data: List[FeatureVector] = []
    regimes: List[str] = []

    for _ in range(n_samples):
        feats = {f: rng.gauss(0, 1) for f in feat_names}
        # First 3 features are predictive
        signal = 0.5 * feats["feat_00"] + 0.3 * feats["feat_01"] + 0.2 * feats["feat_02"]
        # feat_03 is redundant (correlated with feat_00)
        feats["feat_03"] = feats["feat_00"] * 0.9 + rng.gauss(0, 0.3)
        # feat_04 interacts with feat_00
        feats["feat_04"] = rng.gauss(0, 1)
        signal += 0.1 * feats["feat_00"] * feats["feat_04"]

        label = signal + rng.gauss(0, 0.5)
        data.append(FeatureVector(feats, label))

        regime = "bull" if signal > 0.3 else ("bear" if signal < -0.3 else "sideways")
        regimes.append(regime)

    return data, regimes
