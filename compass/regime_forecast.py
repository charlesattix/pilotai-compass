"""
Regime forecaster — predicts next market regime using transition probabilities,
macro leading indicators, and regime feature momentum.

Uses an empirical Hidden Markov Model (transition matrix from observed regime
sequences) rather than fitted HMM parameters — more robust with limited data.

Forecasts:
  - 1-day: next trading day's regime
  - 1-week: dominant regime over the next 5 trading days
  - 1-month: dominant regime over the next 21 trading days

Calibration: compares forecasted vs realized transitions on historical data
to measure accuracy and identify systematic biases.

Usage::

    from compass.regime_forecast import RegimeForecaster
    forecaster = RegimeForecaster()
    forecaster.fit(regime_series)
    forecast = forecaster.predict(current_regime="bull", vix=18, vix3m=20)
"""

from __future__ import annotations

import base64
import io
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.regime import Regime

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "regime_forecast.html"

REGIME_NAMES = [r.value for r in Regime]  # bull, bear, high_vol, low_vol, crash
REGIME_TO_IDX = {r: i for i, r in enumerate(REGIME_NAMES)}
N_REGIMES = len(REGIME_NAMES)


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class RegimeForecast:
    """Probability distribution over next regime."""
    current_regime: str
    horizon: str                          # "1d", "1w", "1m"
    probabilities: Dict[str, float]       # regime → probability
    predicted_regime: str                 # argmax
    confidence: float                     # max probability
    macro_adjustments: Dict[str, float]   # adjustments applied


@dataclass
class CalibrationResult:
    """Forecast accuracy on historical data."""
    horizon: str
    n_predictions: int
    accuracy: float                       # fraction correct
    per_regime_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray          # (N_REGIMES, N_REGIMES)
    brier_score: float


# ── Transition matrix ────────────────────────────────────────────────────


def build_transition_matrix(
    regime_sequence: List[str],
    smoothing: float = 1.0,
) -> np.ndarray:
    """Build empirical transition probability matrix from observed regime sequence.

    Uses Laplace smoothing to handle unseen transitions.

    Args:
        regime_sequence: List of regime labels in chronological order.
        smoothing: Laplace smoothing count (default 1.0 = add-one).

    Returns:
        (N_REGIMES, N_REGIMES) row-stochastic matrix where M[i,j] = P(next=j|current=i).
    """
    counts = np.full((N_REGIMES, N_REGIMES), smoothing)

    for i in range(len(regime_sequence) - 1):
        curr = regime_sequence[i]
        nxt = regime_sequence[i + 1]
        if curr in REGIME_TO_IDX and nxt in REGIME_TO_IDX:
            counts[REGIME_TO_IDX[curr], REGIME_TO_IDX[nxt]] += 1

    # Normalize rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # prevent division by zero
    return counts / row_sums


def multi_step_forecast(
    transition_matrix: np.ndarray,
    current_regime: str,
    steps: int,
) -> np.ndarray:
    """Forecast regime distribution N steps ahead via matrix exponentiation.

    Returns probability vector over all regimes after *steps* transitions.
    """
    idx = REGIME_TO_IDX.get(current_regime, 0)
    state = np.zeros(N_REGIMES)
    state[idx] = 1.0

    mat_power = np.linalg.matrix_power(transition_matrix, steps)
    return state @ mat_power


# ── Macro adjustments ────────────────────────────────────────────────────


def compute_macro_adjustments(
    vix: Optional[float] = None,
    vix3m: Optional[float] = None,
    yield_spread: Optional[float] = None,
    credit_spread: Optional[float] = None,
    momentum_10d: Optional[float] = None,
) -> Dict[str, float]:
    """Compute regime probability adjustments from macro leading indicators.

    Returns dict of {regime: adjustment} where positive = increase probability.
    Adjustments are bounded to [-0.15, +0.15] per regime.
    """
    adj = {r: 0.0 for r in REGIME_NAMES}

    # VIX level
    if vix is not None:
        if vix > 35:
            adj["high_vol"] += 0.10
            adj["crash"] += 0.05
            adj["bull"] -= 0.10
        elif vix > 25:
            adj["high_vol"] += 0.05
            adj["bear"] += 0.03
            adj["bull"] -= 0.05
        elif vix < 15:
            adj["low_vol"] += 0.08
            adj["bull"] += 0.05
            adj["high_vol"] -= 0.05

    # VIX term structure (backwardation = stress)
    if vix is not None and vix3m is not None and vix3m > 0:
        ratio = vix / vix3m
        if ratio > 1.1:  # backwardation
            adj["high_vol"] += 0.08
            adj["crash"] += 0.05
            adj["bull"] -= 0.08
        elif ratio < 0.9:  # steep contango = calm
            adj["bull"] += 0.05
            adj["low_vol"] += 0.03

    # Yield curve (inverted = recession risk)
    if yield_spread is not None:
        if yield_spread < -0.5:
            adj["bear"] += 0.08
            adj["bull"] -= 0.05
        elif yield_spread > 1.0:
            adj["bull"] += 0.03

    # Credit spreads (wide = stress)
    if credit_spread is not None:
        if credit_spread > 5.0:
            adj["high_vol"] += 0.08
            adj["crash"] += 0.05
            adj["bull"] -= 0.08
        elif credit_spread < 3.0:
            adj["bull"] += 0.03

    # Momentum
    if momentum_10d is not None:
        if momentum_10d < -5:
            adj["bear"] += 0.05
            adj["bull"] -= 0.05
        elif momentum_10d > 5:
            adj["bull"] += 0.05
            adj["bear"] -= 0.03

    # Clamp adjustments
    for r in REGIME_NAMES:
        adj[r] = max(-0.15, min(0.15, adj[r]))

    return adj


def apply_adjustments(
    base_probs: np.ndarray,
    adjustments: Dict[str, float],
) -> np.ndarray:
    """Apply macro adjustments to base probabilities and re-normalize."""
    adjusted = base_probs.copy()
    for regime, delta in adjustments.items():
        idx = REGIME_TO_IDX.get(regime)
        if idx is not None:
            adjusted[idx] += delta

    # Clamp to non-negative and re-normalize
    adjusted = np.maximum(adjusted, 0.001)
    return adjusted / adjusted.sum()


# ── Calibration ──────────────────────────────────────────────────────────


def calibrate(
    regime_sequence: List[str],
    transition_matrix: np.ndarray,
    horizon_steps: int = 1,
) -> CalibrationResult:
    """Evaluate forecast accuracy on historical regime sequence.

    For each position i, predicts regime at i+horizon_steps using the
    transition matrix and compares to actual.
    """
    n = len(regime_sequence)
    if n < horizon_steps + 10:
        return CalibrationResult(
            horizon=f"{horizon_steps}d",
            n_predictions=0, accuracy=0.0,
            per_regime_accuracy={}, confusion_matrix=np.zeros((N_REGIMES, N_REGIMES)),
            brier_score=1.0,
        )

    correct = 0
    total = 0
    per_regime_correct: Dict[str, int] = Counter()
    per_regime_total: Dict[str, int] = Counter()
    confusion = np.zeros((N_REGIMES, N_REGIMES), dtype=int)
    brier_scores: List[float] = []

    for i in range(n - horizon_steps):
        curr = regime_sequence[i]
        actual = regime_sequence[i + horizon_steps]
        if curr not in REGIME_TO_IDX or actual not in REGIME_TO_IDX:
            continue

        probs = multi_step_forecast(transition_matrix, curr, horizon_steps)
        predicted = REGIME_NAMES[int(np.argmax(probs))]
        actual_idx = REGIME_TO_IDX[actual]

        total += 1
        per_regime_total[curr] += 1

        if predicted == actual:
            correct += 1
            per_regime_correct[curr] += 1

        confusion[REGIME_TO_IDX[curr], actual_idx] += 1

        # Brier score: mean squared error of probability vector
        target = np.zeros(N_REGIMES)
        target[actual_idx] = 1.0
        brier_scores.append(float(np.mean((probs - target) ** 2)))

    accuracy = correct / total if total > 0 else 0.0
    per_regime_acc = {
        r: per_regime_correct[r] / per_regime_total[r]
        if per_regime_total[r] > 0 else 0.0
        for r in REGIME_NAMES
    }

    horizon_label = {1: "1d", 5: "1w", 21: "1m"}.get(horizon_steps, f"{horizon_steps}d")

    return CalibrationResult(
        horizon=horizon_label,
        n_predictions=total,
        accuracy=round(accuracy, 4),
        per_regime_accuracy={k: round(v, 4) for k, v in per_regime_acc.items()},
        confusion_matrix=confusion,
        brier_score=round(float(np.mean(brier_scores)), 4) if brier_scores else 1.0,
    )


# ── RegimeForecaster ─────────────────────────────────────────────────────


class RegimeForecaster:
    """Regime forecaster using empirical transition probabilities and macro signals.

    Args:
        smoothing: Laplace smoothing for transition matrix.
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.transition_matrix: Optional[np.ndarray] = None
        self.regime_sequence: List[str] = []
        self.calibration_results: Dict[str, CalibrationResult] = {}
        self._fitted = False

    def fit(self, regime_sequence: List[str]) -> np.ndarray:
        """Fit transition matrix from observed regime sequence.

        Args:
            regime_sequence: Chronological list of regime labels.

        Returns:
            The fitted transition matrix.
        """
        self.regime_sequence = list(regime_sequence)
        self.transition_matrix = build_transition_matrix(regime_sequence, self.smoothing)
        self._fitted = True

        # Auto-calibrate on training data
        for steps, label in [(1, "1d"), (5, "1w"), (21, "1m")]:
            self.calibration_results[label] = calibrate(
                regime_sequence, self.transition_matrix, steps,
            )

        logger.info("Fitted transition matrix on %d observations", len(regime_sequence))
        return self.transition_matrix

    def predict(
        self,
        current_regime: str,
        horizon: str = "1d",
        vix: Optional[float] = None,
        vix3m: Optional[float] = None,
        yield_spread: Optional[float] = None,
        credit_spread: Optional[float] = None,
        momentum_10d: Optional[float] = None,
    ) -> RegimeForecast:
        """Predict next regime with probability distribution.

        Args:
            current_regime: Current observed regime.
            horizon: "1d", "1w", or "1m".
            vix, vix3m, yield_spread, credit_spread, momentum_10d:
                Optional macro indicators for adjustment.

        Returns:
            RegimeForecast with probability distribution and predicted regime.
        """
        if self.transition_matrix is None:
            # Uniform prior if not fitted
            probs = np.ones(N_REGIMES) / N_REGIMES
        else:
            steps = {"1d": 1, "1w": 5, "1m": 21}.get(horizon, 1)
            probs = multi_step_forecast(self.transition_matrix, current_regime, steps)

        # Apply macro adjustments
        macro_adj = compute_macro_adjustments(vix, vix3m, yield_spread, credit_spread, momentum_10d)
        adjusted = apply_adjustments(probs, macro_adj)

        predicted = REGIME_NAMES[int(np.argmax(adjusted))]
        confidence = float(np.max(adjusted))

        prob_dict = {REGIME_NAMES[i]: round(float(adjusted[i]), 4) for i in range(N_REGIMES)}

        return RegimeForecast(
            current_regime=current_regime,
            horizon=horizon,
            probabilities=prob_dict,
            predicted_regime=predicted,
            confidence=round(confidence, 4),
            macro_adjustments={k: round(v, 4) for k, v in macro_adj.items()},
        )

    def forecast_all_horizons(
        self, current_regime: str, **macro_kwargs,
    ) -> Dict[str, RegimeForecast]:
        """Forecast at all three horizons."""
        return {
            h: self.predict(current_regime, horizon=h, **macro_kwargs)
            for h in ("1d", "1w", "1m")
        }

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ── HTML Report ──────────────────────────────────────────────────

    def generate_report(
        self,
        current_regime: str = "bull",
        output: str = str(DEFAULT_OUTPUT),
        **macro_kwargs,
    ) -> str:
        """Generate HTML forecast report."""
        forecasts = self.forecast_all_horizons(current_regime, **macro_kwargs)
        charts = self._render_charts()
        html = self._build_html(current_regime, forecasts, charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Report written to %s", out)
        return str(out.resolve())

    def _render_charts(self) -> Dict[str, str]:
        import matplotlib
        matplotlib.use("Agg")
        charts: Dict[str, str] = {}
        if self.transition_matrix is not None:
            charts["heatmap"] = self._chart_transition_heatmap()
        if self.calibration_results:
            charts["accuracy"] = self._chart_calibration()
        return charts

    def _fig_to_b64(self, fig) -> str:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _chart_transition_heatmap(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        M = self.transition_matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(N_REGIMES))
        ax.set_yticks(range(N_REGIMES))
        ax.set_xticklabels(REGIME_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(REGIME_NAMES, fontsize=9)
        ax.set_xlabel("Next Regime")
        ax.set_ylabel("Current Regime")
        for i in range(N_REGIMES):
            for j in range(N_REGIMES):
                color = "white" if M[i, j] > 0.5 else "black"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Probability")
        ax.set_title("Regime Transition Probability Matrix", fontsize=12)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_calibration(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        horizons = ["1d", "1w", "1m"]
        accs = [self.calibration_results.get(h, CalibrationResult(h, 0, 0, {}, np.zeros((5, 5)), 1)).accuracy
                for h in horizons]
        briers = [self.calibration_results.get(h, CalibrationResult(h, 0, 0, {}, np.zeros((5, 5)), 1)).brier_score
                  for h in horizons]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        colors = ["#16a34a" if a > 0.5 else "#d97706" if a > 0.3 else "#dc2626" for a in accs]
        ax1.bar(horizons, [a * 100 for a in accs], color=colors, alpha=0.85)
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Forecast Accuracy by Horizon", fontsize=11)
        ax1.set_ylim(0, 100)
        ax1.axhline(20, color="gray", ls="--", lw=0.8, label="Random (20%)")
        ax1.legend(fontsize=8)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2.bar(horizons, briers, color="#2563eb", alpha=0.85)
        ax2.set_ylabel("Brier Score (lower=better)")
        ax2.set_title("Brier Score by Horizon", fontsize=11)
        ax2.set_ylim(0, 0.5)
        ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(
        self,
        current_regime: str,
        forecasts: Dict[str, RegimeForecast],
        charts: Dict[str, str],
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        def _img(key):
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ''

        # Forecast cards
        cards = ""
        for h in ("1d", "1w", "1m"):
            fc = forecasts.get(h)
            if not fc:
                continue
            horizon_label = {"1d": "Tomorrow", "1w": "Next Week", "1m": "Next Month"}[h]
            prob_bars = ""
            for r in REGIME_NAMES:
                p = fc.probabilities.get(r, 0)
                w = max(2, p * 100)
                color = {"bull": "#16a34a", "bear": "#dc2626", "high_vol": "#d97706",
                         "low_vol": "#2563eb", "crash": "#7c3aed"}.get(r, "#64748b")
                bold = "font-weight:700" if r == fc.predicted_regime else ""
                prob_bars += f'<div style="display:flex;align-items:center;margin:2px 0;{bold}"><span style="width:70px;font-size:0.85em">{r}</span><div style="background:{color};width:{w}%;height:16px;border-radius:3px;opacity:0.75"></div><span style="margin-left:6px;font-size:0.85em">{p:.0%}</span></div>'
            cards += f"""
            <div class="forecast-card">
              <h4>{horizon_label}</h4>
              <div class="pred"><strong>{fc.predicted_regime.upper()}</strong> ({fc.confidence:.0%})</div>
              {prob_bars}
            </div>"""

        # Calibration table
        cal_rows = ""
        for h in ("1d", "1w", "1m"):
            cr = self.calibration_results.get(h)
            if not cr:
                continue
            acc_cls = "good" if cr.accuracy > 0.5 else ""
            cal_rows += (
                f'<tr><td>{h}</td><td>{cr.n_predictions}</td>'
                f'<td class="{acc_cls}">{cr.accuracy:.1%}</td>'
                f'<td>{cr.brier_score:.4f}</td></tr>\n'
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Regime Forecast</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  h4 {{ margin: 0 0 0.3em; color: #475569; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left; border-bottom: 2px solid #cbd5e1; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  .forecast-cards {{ display: flex; gap: 1.5em; flex-wrap: wrap; margin: 1.5em 0; }}
  .forecast-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
                    padding: 1.2em; flex: 1; min-width: 250px; }}
  .pred {{ font-size: 1.2em; margin: 0.3em 0 0.8em; color: #1e293b; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Regime Forecast</h1>
<div class="meta">Current regime: <strong>{current_regime.upper()}</strong> · {len(self.regime_sequence)} historical observations · Generated {now}</div>

<h2>1. Current Forecast</h2>
<div class="forecast-cards">{cards}</div>

<h2>2. Transition Matrix</h2>
{_img("heatmap")}

<h2>3. Calibration</h2>
{_img("accuracy")}
<table>
<thead><tr><th>Horizon</th><th>Predictions</th><th>Accuracy</th><th>Brier Score</th></tr></thead>
<tbody>{cal_rows}</tbody>
</table>

<footer>Generated by <code>compass/regime_forecast.py</code></footer>
</body></html>"""
        return html
