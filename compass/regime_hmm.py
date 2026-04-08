"""
Market regime transition probabilities via Hidden Markov Model.

Fits a 4-state Gaussian HMM on continuous emissions (returns, VIX,
breadth) to learn regime dynamics and forecast transitions.

States: Bull(0), Sideways(1), Correction(2), Crisis(3)

Uses Baum-Welch EM — no hmmlearn dependency, pure numpy.
All methods work on pre-loaded data — no API calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
N_STATES = 4
STATE_NAMES = {0: "bull", 1: "sideways", 2: "correction", 3: "crisis"}


@dataclass
class HMMParams:
    n_states: int
    transition_matrix: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    initial_probs: np.ndarray
    log_likelihood: float = 0.0


@dataclass
class RegimeState:
    date: datetime
    state_probs: np.ndarray
    most_likely: int
    regime_name: str
    next_day_probs: np.ndarray
    transition_signal: float


@dataclass
class TransitionEvent:
    date: datetime
    from_state: str
    to_state: str
    lead_days: int
    probability: float


@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    n_transitions: int
    avg_lead_days: float
    accuracy: float
    transitions: List[TransitionEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gaussian HMM (pure numpy)
# ---------------------------------------------------------------------------

class GaussianHMM:
    def __init__(self, n_states: int = N_STATES, n_iter: int = 30):
        self.K = n_states
        self.n_iter = n_iter
        self.params: Optional[HMMParams] = None

    def _init(self, X: np.ndarray) -> HMMParams:
        N, D = X.shape
        K = self.K
        idx = np.argsort(X[:, 0])
        bs = N // K
        means = np.zeros((K, D))
        covs = np.zeros((K, D, D))
        for k in range(K):
            s, e = k * bs, (k + 1) * bs if k < K - 1 else N
            chunk = X[idx[s:e]]
            means[k] = chunk.mean(0)
            c = np.cov(chunk, rowvar=False)
            covs[k] = (c if c.ndim == 2 else np.array([[float(c)]])) + np.eye(D) * 1e-6
        pi = np.ones(K) / K
        A = np.full((K, K), 0.05)
        np.fill_diagonal(A, 0.85)
        A /= A.sum(1, keepdims=True)
        return HMMParams(K, A, means, covs, pi)

    @staticmethod
    def _log_gauss(x, mu, cov):
        D = len(mu)
        diff = x - mu
        try:
            ci = np.linalg.inv(cov)
            s, ld = np.linalg.slogdet(cov)
            if s <= 0:
                return -1e10
            return float(-0.5 * (D * np.log(2 * np.pi) + ld + diff @ ci @ diff))
        except np.linalg.LinAlgError:
            return -1e10

    def _forward(self, X, p):
        N, K = len(X), p.n_states
        la = np.full((N, K), -1e10)
        for k in range(K):
            la[0, k] = np.log(p.initial_probs[k] + 1e-300) + self._log_gauss(X[0], p.means[k], p.covariances[k])
        for t in range(1, N):
            for j in range(K):
                le = self._log_gauss(X[t], p.means[j], p.covariances[j])
                v = la[t - 1] + np.log(p.transition_matrix[:, j] + 1e-300)
                mx = v.max()
                la[t, j] = mx + np.log(np.exp(v - mx).sum()) + le
        mx = la[-1].max()
        ll = mx + np.log(np.exp(la[-1] - mx).sum())
        return la, float(ll)

    def _backward(self, X, p):
        N, K = len(X), p.n_states
        lb = np.full((N, K), -1e10)
        lb[-1] = 0.0
        for t in range(N - 2, -1, -1):
            for i in range(K):
                v = np.array([np.log(p.transition_matrix[i, j] + 1e-300) +
                               self._log_gauss(X[t + 1], p.means[j], p.covariances[j]) +
                               lb[t + 1, j] for j in range(K)])
                mx = v.max()
                lb[t, i] = mx + np.log(np.exp(v - mx).sum())
        return lb

    def fit(self, X: np.ndarray) -> HMMParams:
        N, D = X.shape
        K = self.K
        p = self._init(X)
        ll = -1e20
        for _ in range(self.n_iter):
            la, ll = self._forward(X, p)
            lb = self._backward(X, p)
            lg = la + lb
            mx = lg.max(1, keepdims=True)
            g = np.exp(lg - mx)
            g /= g.sum(1, keepdims=True) + 1e-300
            p.initial_probs = g[0] / (g[0].sum() + 1e-300)
            for i in range(K):
                for j in range(K):
                    s = sum(np.exp(la[t, i] + np.log(p.transition_matrix[i, j] + 1e-300) +
                                    self._log_gauss(X[t + 1], p.means[j], p.covariances[j]) +
                                    lb[t + 1, j] - ll) for t in range(N - 1))
                    p.transition_matrix[i, j] = s
                d = p.transition_matrix[i].sum()
                if d > 0:
                    p.transition_matrix[i] /= d
            for k in range(K):
                w = g[:, k]
                ws = w.sum() + 1e-300
                p.means[k] = (w[:, None] * X).sum(0) / ws
                diff = X - p.means[k]
                p.covariances[k] = (diff.T * w) @ diff / ws + np.eye(D) * 1e-6
        p.log_likelihood = ll
        self.params = p
        return p

    def predict_proba(self, X):
        la, _ = self._forward(X, self.params)
        lb = self._backward(X, self.params)
        lg = la + lb
        mx = lg.max(1, keepdims=True)
        g = np.exp(lg - mx)
        g /= g.sum(1, keepdims=True) + 1e-300
        return g

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_regime_data(n_days=1512, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    regime = np.zeros(n_days, dtype=int)
    if n_days > 120:
        regime[45:75] = 3; regime[75:120] = 2
    if n_days > 700:
        regime[504:560] = 2; regime[560:700] = 1
    sp = {0: (0.0006, 0.008, 16, 0.6), 1: (0.0001, 0.010, 22, 0.5),
          2: (-0.002, 0.015, 28, 0.35), 3: (-0.008, 0.035, 55, 0.15)}
    ret = np.array([rng.normal(sp[regime[i]][0], sp[regime[i]][1]) for i in range(n_days)])
    vix = np.array([max(10, sp[regime[i]][2] + rng.normal(0, sp[regime[i]][2] * 0.15)) for i in range(n_days)])
    brd = np.array([np.clip(sp[regime[i]][3] + rng.normal(0, 0.08), 0.05, 0.95) for i in range(n_days)])
    features = pd.DataFrame({"returns": ret, "vix": vix, "breadth": brd}, index=idx)
    return features, pd.Series(ret, index=idx)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RegimeTransitionEngine:
    def __init__(self, n_states=N_STATES, n_iter=20, threshold=0.3):
        self.n_states = n_states
        self.hmm = GaussianHMM(n_states, n_iter)
        self.threshold = threshold
        self._params: Optional[HMMParams] = None

    def fit(self, features: pd.DataFrame) -> HMMParams:
        self._params = self.hmm.fit(features.values)
        order = np.argsort(self._params.means[:, 0])[::-1]
        self._params.means = self._params.means[order]
        self._params.covariances = self._params.covariances[order]
        self._params.transition_matrix = self._params.transition_matrix[order][:, order]
        self._params.initial_probs = self._params.initial_probs[order]
        return self._params

    def assess(self, features: pd.DataFrame) -> List[RegimeState]:
        if self._params is None:
            self.fit(features)
        probs = self.hmm.predict_proba(features.values)
        states = []
        for i in range(len(features)):
            cp = probs[i]
            np_next = cp @ self._params.transition_matrix
            ml = int(cp.argmax())
            fav_now = cp[:2].sum()
            fav_next = np_next[:2].sum()
            sig = float(np.clip((fav_next - fav_now) / 0.3, -1, 1))
            states.append(RegimeState(features.index[i], cp, ml,
                                        STATE_NAMES.get(ml, f"s{ml}"), np_next, sig))
        return states

    def backtest(self, features: pd.DataFrame, returns: pd.Series,
                  cost=0.001) -> BacktestResult:
        states = self.assess(features)
        n = len(states)
        size_map = {"bull": 1.2, "sideways": 0.8, "correction": 0.4, "crisis": 0.0}
        rets = np.zeros(n)
        for i in range(1, n):
            s = states[i - 1]
            size = size_map.get(s.regime_name, 0.5)
            if s.transition_signal < -0.5:
                size *= 0.2
            elif s.transition_signal > 0.5:
                size *= 1.2
            rets[i] = float(returns.iloc[i]) * size * 0.1

        transitions = []
        for i in range(1, n):
            if states[i].most_likely != states[i - 1].most_likely:
                lead = 0
                for j in range(max(0, i - 30), i):
                    if abs(states[j].transition_signal) > self.threshold:
                        lead = i - j; break
                transitions.append(TransitionEvent(
                    states[i].date, states[i - 1].regime_name,
                    states[i].regime_name, lead, float(states[i].state_probs.max())))

        total = float(np.prod(1 + rets) - 1)
        n_yr = n / TRADING_DAYS
        annual = (1 + total) ** (1 / max(n_yr, 0.01)) - 1
        mu, std = float(rets.mean()), float(rets.std())
        sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
        eq = np.cumprod(1 + rets)
        dd = float((1 - eq / np.maximum.accumulate(eq)).max())
        avg_lead = float(np.mean([t.lead_days for t in transitions])) if transitions else 0
        correct = sum(1 for t in transitions if t.lead_days > 0)
        acc = correct / len(transitions) if transitions else 0

        return BacktestResult(total, annual, sharpe, dd, len(transitions), avg_lead, acc, transitions)

    @property
    def transition_matrix(self) -> Optional[np.ndarray]:
        return self._params.transition_matrix if self._params else None

    def generate_report(self, result: BacktestResult, states=None,
                         output_path="reports/regime_hmm.html") -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        regime_svg = ""
        if states and len(states) > 10:
            n = len(states)
            w, h = 750, 50
            colors = {0: "#059669", 1: "#2563eb", 2: "#d97706", 3: "#dc2626"}
            bw = w / max(n, 1)
            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                      f'style="border:1px solid #e2e8f0;border-radius:6px;margin:.5rem 0">']
            for i, s in enumerate(states):
                c = colors.get(s.most_likely, "#999")
                parts.append(f'<rect x="{i * bw:.1f}" y="0" width="{bw + .5:.1f}" height="{h - 16}" fill="{c}"/>')
            lx = 5
            for sid, name in STATE_NAMES.items():
                c = colors.get(sid, "#999")
                parts.append(f'<rect x="{lx}" y="{h - 12}" width="8" height="8" fill="{c}"/>')
                parts.append(f'<text x="{lx + 11}" y="{h - 4}" font-size="8" fill="#333">{name}</text>')
                lx += 80
            parts.append("</svg>")
            regime_svg = "\n".join(parts)

        tm_html = ""
        if self._params is not None:
            A = self._params.transition_matrix
            rows = []
            for i in range(self.n_states):
                cells = "".join(f"<td>{A[i, j]:.2f}</td>" for j in range(self.n_states))
                rows.append(f"<tr><td style='text-align:left'>{STATE_NAMES.get(i, i)}</td>{cells}</tr>")
            hdrs = "".join(f"<th>{STATE_NAMES.get(j, j)}</th>" for j in range(self.n_states))
            tm_html = (f'<h2>Transition Matrix</h2><table>'
                        f'<tr><th style="text-align:left">From / To</th>{hdrs}</tr>'
                        f'{"".join(rows)}</table>')

        r = result
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Regime HMM</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fff; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: .5rem; }}
h2 {{ color: #334155; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #f1f5f9; padding: 10px 12px; text-align: right; border-bottom: 2px solid #e2e8f0; }}
th:first-child {{ text-align: left; }}
td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #f1f5f9; }}
td:first-child {{ text-align: left; }}
.card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }}
</style></head><body>
<h1>EXP-1360-max: Regime Transition Probabilities (HMM)</h1>
<div class="card">
<p><strong>Sharpe:</strong> {r.sharpe:.2f} | <strong>Annual:</strong> {r.annual_return:.1%} |
<strong>Max DD:</strong> {r.max_drawdown:.1%} | <strong>Transitions:</strong> {r.n_transitions} |
<strong>Avg Lead:</strong> {r.avg_lead_days:.0f}d | <strong>Accuracy:</strong> {r.accuracy:.0%}</p>
</div>
{regime_svg}
{tm_html}
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
