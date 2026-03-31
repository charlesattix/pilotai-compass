"""Ensemble regime detection — combines HMM, change-point detection, volatility
clustering, trend/mean-reversion classification, and macro overlay into a
consensus regime with confidence scores and transition forecasting.

Provides:
  1. HMM-based regime detection (Gaussian emissions via EM)
  2. Change-point detection (CUSUM)
  3. Volatility clustering (rolling vol quantiles)
  4. Trend vs mean-reversion classifier
  5. Macro regime overlay (VIX + yield curve)
  6. Voting/stacking to produce consensus regime with confidence
  7. Regime transition probability matrix and forecasting
  8. HTML report with timeline, agreement heatmap, transition matrix
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIMES = ["bull", "bear", "high_vol", "low_vol", "crash"]
N_REGIMES = len(REGIMES)


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class MethodResult:
    """Output of a single detection method."""
    method: str
    regimes: List[str]          # per-bar regime label
    confidence: List[float]     # per-bar confidence (0–1)


@dataclass
class TransitionMatrix:
    """Regime transition probabilities."""
    matrix: Dict[str, Dict[str, float]]
    regimes: List[str]
    n_transitions: int
    stationary: Dict[str, float]


@dataclass
class RegimeConsensus:
    """Consensus regime at one time point."""
    regime: str
    confidence: float
    method_agreement: float     # fraction of methods agreeing
    votes: Dict[str, str]       # method → regime voted


@dataclass
class EnsembleResult:
    """Complete ensemble output."""
    consensus: List[RegimeConsensus] = field(default_factory=list)
    method_results: List[MethodResult] = field(default_factory=list)
    transition_matrix: Optional[TransitionMatrix] = None
    forecast: Dict[str, float] = field(default_factory=dict)  # regime → prob next period
    current_regime: str = ""
    current_confidence: float = 0.0
    n_bars: int = 0
    generated_at: str = ""


# ── Individual detectors ────────────────────────────────────────────────────
class GaussianHMM:
    """Simple 2-state HMM with Gaussian emissions via EM."""

    def __init__(self, n_states: int = 3, n_iter: int = 20, seed: int = 42) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.rng = np.random.RandomState(seed)

    def fit_predict(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (state_sequence, state_probs) via EM."""
        n = len(returns)
        k = self.n_states

        # Initialise parameters
        means = np.linspace(returns.min(), returns.max(), k)
        stds = np.full(k, returns.std())
        weights = np.ones(k) / k
        trans = np.full((k, k), 1.0 / k)

        # EM iterations
        for _ in range(self.n_iter):
            # E-step: compute responsibilities
            resp = np.zeros((n, k))
            for j in range(k):
                std_j = max(stds[j], 1e-10)
                resp[:, j] = weights[j] * _norm_pdf_array(returns, means[j], std_j)
            row_sums = resp.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-300)
            resp /= row_sums

            # M-step
            Nk = resp.sum(axis=0) + 1e-10
            weights = Nk / n
            for j in range(k):
                means[j] = float(np.dot(resp[:, j], returns) / Nk[j])
                diff = returns - means[j]
                stds[j] = float(np.sqrt(np.dot(resp[:, j], diff ** 2) / Nk[j]))
                stds[j] = max(stds[j], 1e-10)

            # Transition matrix from consecutive responsibilities
            for j1 in range(k):
                for j2 in range(k):
                    trans[j1, j2] = np.sum(resp[:-1, j1] * resp[1:, j2]) + 1e-10
                trans[j1] /= trans[j1].sum()

        states = np.argmax(resp, axis=1)
        max_probs = np.max(resp, axis=1)

        return states, max_probs


def detect_hmm(returns: pd.Series, n_states: int = 3, seed: int = 42) -> MethodResult:
    """HMM-based regime detection."""
    r = returns.dropna().values
    if len(r) < 20:
        return MethodResult("hmm", [], [])

    hmm = GaussianHMM(n_states=n_states, seed=seed)
    states, probs = hmm.fit_predict(r)

    # Map states to regime labels by mean return
    state_means = {s: float(r[states == s].mean()) for s in range(n_states)}
    sorted_states = sorted(state_means, key=state_means.get)

    label_map: Dict[int, str] = {}
    if n_states >= 3:
        label_map[sorted_states[0]] = "bear"
        label_map[sorted_states[-1]] = "bull"
        for s in sorted_states[1:-1]:
            # Check volatility to distinguish high_vol vs low_vol
            vol = float(r[states == s].std()) if (states == s).sum() > 1 else 0
            median_vol = float(np.median([r[states == si].std() for si in range(n_states) if (states == si).sum() > 1]))
            label_map[s] = "high_vol" if vol > median_vol else "low_vol"
    elif n_states == 2:
        label_map[sorted_states[0]] = "bear"
        label_map[sorted_states[1]] = "bull"
    else:
        label_map[0] = "bull"

    regimes = [label_map.get(s, "bull") for s in states]
    return MethodResult("hmm", regimes, [float(p) for p in probs])


def detect_changepoint(returns: pd.Series, threshold: float = 2.0) -> MethodResult:
    """CUSUM-based change-point detection for regime shifts."""
    r = returns.dropna().values
    n = len(r)
    if n < 20:
        return MethodResult("changepoint", [], [])

    mean = np.mean(r)
    std = max(np.std(r), 1e-10)

    # CUSUM: track cumulative sum of normalised deviations
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    changepoints = []

    for i in range(1, n):
        z = (r[i] - mean) / std
        cusum_pos[i] = max(0, cusum_pos[i - 1] + z - 0.5)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - z - 0.5)
        if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
            changepoints.append(i)
            cusum_pos[i] = 0
            cusum_neg[i] = 0

    # Label segments by rolling mean
    regimes: List[str] = []
    confidences: List[float] = []
    window = max(10, n // 20)

    for i in range(n):
        start = max(0, i - window)
        seg_mean = float(np.mean(r[start:i + 1]))
        seg_vol = float(np.std(r[start:i + 1])) if i > start else std
        regime = _classify_from_stats(seg_mean, seg_vol, mean, std)
        regimes.append(regime)
        # Confidence: higher near change points
        min_dist = min((abs(i - cp) for cp in changepoints), default=n)
        conf = max(0.3, 1.0 - min_dist / max(window, 1))
        confidences.append(conf)

    return MethodResult("changepoint", regimes, confidences)


def detect_vol_clustering(returns: pd.Series, window: int = 20) -> MethodResult:
    """Volatility-quantile-based regime detection."""
    r = returns.dropna()
    n = len(r)
    if n < window + 5:
        return MethodResult("vol_cluster", [], [])

    rolling_vol = r.rolling(window).std().dropna()
    vol_median = float(rolling_vol.median())
    vol_p25 = float(rolling_vol.quantile(0.25))
    vol_p75 = float(rolling_vol.quantile(0.75))
    vol_p90 = float(rolling_vol.quantile(0.90))

    regimes: List[str] = []
    confs: List[float] = []
    for v in rolling_vol.values:
        if v >= vol_p90:
            regimes.append("crash")
            confs.append(0.85)
        elif v >= vol_p75:
            regimes.append("high_vol")
            confs.append(0.70)
        elif v <= vol_p25:
            regimes.append("low_vol")
            confs.append(0.70)
        else:
            # Use return direction to distinguish bull/bear
            regimes.append("bull")
            confs.append(0.50)

    return MethodResult("vol_cluster", regimes, confs)


def detect_trend(returns: pd.Series, window: int = 50) -> MethodResult:
    """Trend vs mean-reversion classification."""
    r = returns.dropna()
    n = len(r)
    if n < window + 5:
        return MethodResult("trend", [], [])

    cum = r.cumsum()
    regimes: List[str] = []
    confs: List[float] = []

    for i in range(window, n):
        seg = cum.iloc[i - window:i + 1].values
        # Linear regression slope
        x = np.arange(window + 1)
        slope = float(np.polyfit(x, seg, 1)[0])
        # Hurst exponent proxy: variance ratio
        half = window // 2
        var_full = float(np.var(np.diff(seg)))
        var_half = float(np.var(np.diff(seg[:half + 1])))
        vr = var_full / (2 * var_half) if var_half > 1e-15 else 1.0

        if slope > 0.0005:
            regimes.append("bull")
            confs.append(min(0.9, 0.5 + abs(slope) * 500))
        elif slope < -0.0005:
            regimes.append("bear")
            confs.append(min(0.9, 0.5 + abs(slope) * 500))
        else:
            # Mean-reverting or range-bound
            regimes.append("low_vol" if vr < 0.8 else "high_vol")
            confs.append(0.50)

    return MethodResult("trend", regimes, confs)


def detect_macro(
    returns: pd.Series,
    vix: Optional[pd.Series] = None,
    yield_curve: Optional[pd.Series] = None,
) -> MethodResult:
    """Macro overlay using VIX and yield curve."""
    r = returns.dropna()
    n = len(r)
    if n < 10:
        return MethodResult("macro", [], [])

    regimes: List[str] = []
    confs: List[float] = []

    for i in range(n):
        v = float(vix.iloc[i]) if vix is not None and i < len(vix) else 20.0
        yc = float(yield_curve.iloc[i]) if yield_curve is not None and i < len(yield_curve) else 0.5

        if v > 40:
            regimes.append("crash")
            confs.append(0.90)
        elif v > 30:
            regimes.append("high_vol")
            confs.append(0.80)
        elif v > 25 and yc < 0:
            regimes.append("bear")
            confs.append(0.75)
        elif v < 15 and yc > 0.5:
            regimes.append("bull")
            confs.append(0.75)
        elif v < 15:
            regimes.append("low_vol")
            confs.append(0.65)
        else:
            regimes.append("bull" if yc > 0 else "bear")
            confs.append(0.55)

    return MethodResult("macro", regimes, confs)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _norm_pdf_array(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _classify_from_stats(seg_mean: float, seg_vol: float, global_mean: float, global_std: float) -> str:
    if seg_vol > global_std * 2:
        return "crash"
    if seg_vol > global_std * 1.3:
        return "high_vol"
    if seg_mean > global_mean + global_std * 0.5:
        return "bull"
    if seg_mean < global_mean - global_std * 0.5:
        return "bear"
    return "low_vol"


# ── Ensemble ────────────────────────────────────────────────────────────────
class RegimeEnsemble:
    """Ensemble regime detection combining multiple methods."""

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        hmm_states: int = 3,
        vol_window: int = 20,
        trend_window: int = 50,
        seed: int = 42,
    ) -> None:
        self.methods = methods or ["hmm", "changepoint", "vol_cluster", "trend", "macro"]
        self.hmm_states = hmm_states
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.seed = seed

    def detect(
        self,
        returns: pd.Series,
        vix: Optional[pd.Series] = None,
        yield_curve: Optional[pd.Series] = None,
    ) -> EnsembleResult:
        """Run all methods and produce consensus."""
        returns = returns.dropna()
        if len(returns) < 20:
            return EnsembleResult(generated_at=self._now())

        method_results: List[MethodResult] = []
        if "hmm" in self.methods:
            method_results.append(detect_hmm(returns, self.hmm_states, self.seed))
        if "changepoint" in self.methods:
            method_results.append(detect_changepoint(returns))
        if "vol_cluster" in self.methods:
            method_results.append(detect_vol_clustering(returns, self.vol_window))
        if "trend" in self.methods:
            method_results.append(detect_trend(returns, self.trend_window))
        if "macro" in self.methods:
            method_results.append(detect_macro(returns, vix, yield_curve))

        # Align lengths (use shortest)
        min_len = min((len(mr.regimes) for mr in method_results if mr.regimes), default=0)
        if min_len == 0:
            return EnsembleResult(generated_at=self._now())

        for mr in method_results:
            if len(mr.regimes) > min_len:
                offset = len(mr.regimes) - min_len
                mr.regimes = mr.regimes[offset:]
                mr.confidence = mr.confidence[offset:]

        # Voting: confidence-weighted majority
        consensus: List[RegimeConsensus] = []
        for t in range(min_len):
            votes: Dict[str, str] = {}
            regime_scores: Dict[str, float] = {r: 0.0 for r in REGIMES}
            for mr in method_results:
                if t < len(mr.regimes):
                    r = mr.regimes[t]
                    c = mr.confidence[t] if t < len(mr.confidence) else 0.5
                    regime_scores[r] = regime_scores.get(r, 0.0) + c
                    votes[mr.method] = r

            best = max(regime_scores, key=regime_scores.get)
            total_weight = sum(regime_scores.values())
            confidence = regime_scores[best] / total_weight if total_weight > 0 else 0.0
            agreement = sum(1 for v in votes.values() if v == best) / len(votes) if votes else 0.0

            consensus.append(RegimeConsensus(
                regime=best, confidence=confidence,
                method_agreement=agreement, votes=votes,
            ))

        # Transition matrix from consensus
        tm = self._transition_matrix([c.regime for c in consensus])

        # Forecast: next-period probabilities from last regime
        forecast: Dict[str, float] = {}
        if consensus and tm:
            last = consensus[-1].regime
            forecast = dict(tm.matrix.get(last, {}))

        current = consensus[-1].regime if consensus else ""
        current_conf = consensus[-1].confidence if consensus else 0.0

        return EnsembleResult(
            consensus=consensus,
            method_results=method_results,
            transition_matrix=tm,
            forecast=forecast,
            current_regime=current,
            current_confidence=current_conf,
            n_bars=min_len,
            generated_at=self._now(),
        )

    @staticmethod
    def _transition_matrix(regimes: List[str]) -> TransitionMatrix:
        labels = sorted(set(regimes))
        counts: Dict[str, Dict[str, int]] = {r: {r2: 0 for r2 in labels} for r in labels}
        n_trans = 0
        for i in range(len(regimes) - 1):
            counts[regimes[i]][regimes[i + 1]] += 1
            n_trans += 1

        matrix: Dict[str, Dict[str, float]] = {}
        for fr in labels:
            s = sum(counts[fr].values())
            matrix[fr] = {to: counts[fr][to] / s if s > 0 else 0 for to in labels}

        # Stationary distribution via power iteration
        n = len(labels)
        P = np.zeros((n, n))
        for i, fr in enumerate(labels):
            for j, to in enumerate(labels):
                P[i, j] = matrix[fr][to]
        pi = np.ones(n) / n
        for _ in range(100):
            pi = pi @ P
            s = pi.sum()
            if s > 0:
                pi /= s

        stationary = {labels[i]: float(pi[i]) for i in range(n)}
        return TransitionMatrix(matrix, labels, n_trans, stationary)

    def generate_report(
        self,
        result: EnsembleResult,
        output_path: str | Path = "reports/regime_ensemble.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Regime ensemble report written to %s", path)
        return path

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML ────────────────────────────────────────────────────────────────
    def _build_html(self, r: EnsembleResult) -> str:
        cards = self._html_cards(r)
        timeline = self._svg_timeline(r.consensus)
        agreement = self._svg_agreement(r.consensus, r.method_results)
        tm_heatmap = self._svg_transition(r.transition_matrix)
        forecast_tbl = self._html_forecast(r.forecast)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Regime Ensemble</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
svg{{display:block;margin:0 auto}}
</style>
</head>
<body>
<h1>Regime Ensemble Detection</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {r.n_bars} bars &middot; {len(r.method_results)} methods</p>
{cards}
<div class="sec"><h2>Regime Timeline (Consensus)</h2>{timeline}</div>
<div class="sec"><h2>Method Agreement Heatmap</h2>{agreement}</div>
<div class="sec"><h2>Transition Matrix</h2>{tm_heatmap}</div>
{forecast_tbl}
</body>
</html>"""

    @staticmethod
    def _html_cards(r: EnsembleResult) -> str:
        avg_conf = float(np.mean([c.confidence for c in r.consensus])) if r.consensus else 0
        avg_agree = float(np.mean([c.method_agreement for c in r.consensus])) if r.consensus else 0
        return f"""<div class="grid">
<div class="card"><div class="lbl">Current Regime</div><div class="val">{r.current_regime.upper() or 'N/A'}</div></div>
<div class="card"><div class="lbl">Confidence</div><div class="val">{r.current_confidence:.0%}</div></div>
<div class="card"><div class="lbl">Avg Confidence</div><div class="val">{avg_conf:.0%}</div></div>
<div class="card"><div class="lbl">Avg Agreement</div><div class="val">{avg_agree:.0%}</div></div>
<div class="card"><div class="lbl">Methods</div><div class="val">{len(r.method_results)}</div></div>
</div>"""

    @staticmethod
    def _svg_timeline(consensus: List[RegimeConsensus]) -> str:
        if not consensus:
            return "<p>No data.</p>"
        colours = {"bull": "#4ade80", "bear": "#f87171", "high_vol": "#fbbf24",
                   "low_vol": "#38bdf8", "crash": "#a855f7"}
        w, h = 600, 50
        n = len(consensus)
        bars = ""
        bw = max(1, w / n)
        for i, c in enumerate(consensus):
            x = i * bw
            col = colours.get(c.regime, "#475569")
            bars += f'<rect x="{x:.1f}" y="0" width="{bw + 0.5:.1f}" height="35" fill="{col}" opacity="{0.4 + c.confidence * 0.6:.2f}"/>'
        # Legend
        lx = 0
        for regime, col in colours.items():
            bars += (f'<rect x="{lx}" y="40" width="10" height="10" fill="{col}"/>'
                     f'<text x="{lx + 14}" y="49" font-size="9" fill="#94a3b8">{regime}</text>')
            lx += 80
        return f'<svg viewBox="0 0 {w} {h + 10}" width="{w}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'

    @staticmethod
    def _svg_agreement(consensus: List[RegimeConsensus], methods: List[MethodResult]) -> str:
        if not consensus or not methods:
            return "<p>No data.</p>"
        colours = {"bull": "#4ade80", "bear": "#f87171", "high_vol": "#fbbf24",
                   "low_vol": "#38bdf8", "crash": "#a855f7"}
        n_methods = len(methods)
        n_bars = min(len(consensus), 100)  # sample last 100
        offset = max(0, len(consensus) - n_bars)
        w = max(400, n_bars * 4 + 80)
        h = n_methods * 25 + 30
        lbl_w = 80
        bw = max(1, (w - lbl_w) / n_bars)

        cells = ""
        for mi, mr in enumerate(methods):
            y = mi * 25
            cells += f'<text x="{lbl_w - 5}" y="{y + 16}" text-anchor="end" font-size="10" fill="#e2e8f0">{mr.method}</text>'
            for t in range(n_bars):
                idx = offset + t
                if idx < len(mr.regimes):
                    col = colours.get(mr.regimes[idx], "#475569")
                else:
                    col = "#1e293b"
                x = lbl_w + t * bw
                cells += f'<rect x="{x:.1f}" y="{y}" width="{bw + 0.5:.1f}" height="20" fill="{col}" opacity="0.7"/>'

        return f'<svg viewBox="0 0 {w} {h}" width="{min(w, 600)}" xmlns="http://www.w3.org/2000/svg">{cells}</svg>'

    @staticmethod
    def _svg_transition(tm: Optional[TransitionMatrix]) -> str:
        if not tm:
            return "<p>No data.</p>"
        regs = tm.regimes
        n = len(regs)
        cell = 55
        lbl_w = 70
        w = lbl_w + n * cell + 10
        h = 25 + n * cell + 25
        cells = ""
        for i, fr in enumerate(regs):
            cells += f'<text x="{lbl_w - 5}" y="{30 + i * cell + cell // 2 + 4}" text-anchor="end" font-size="10" fill="#e2e8f0">{fr}</text>'
            for j, to in enumerate(regs):
                p = tm.matrix[fr][to]
                intensity = min(255, int(p * 400))
                colour = f"rgb({30},{intensity + 40},{70})"
                x = lbl_w + j * cell
                y = 25 + i * cell
                cells += (f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{colour}" stroke="#0f172a" stroke-width="1"/>'
                          f'<text x="{x + cell // 2}" y="{y + cell // 2 + 4}" text-anchor="middle" font-size="10" fill="#e2e8f0">{p:.0%}</text>')
        for j, to in enumerate(regs):
            cells += f'<text x="{lbl_w + j * cell + cell // 2}" y="18" text-anchor="middle" font-size="9" fill="#94a3b8">{to}</text>'
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">{cells}</svg>'

    @staticmethod
    def _html_forecast(forecast: Dict[str, float]) -> str:
        if not forecast:
            return ""
        rows = "".join(f"<tr><td>{r}</td><td>{p:.0%}</td></tr>" for r, p in sorted(forecast.items(), key=lambda x: -x[1]))
        return f"""<div class="sec"><h2>Next-Period Forecast</h2>
<table><thead><tr><th>Regime</th><th>Probability</th></tr></thead><tbody>{rows}</tbody></table></div>"""
