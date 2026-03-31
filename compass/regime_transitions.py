"""
compass/regime_transitions.py — Regime transition detection and early warning.

Analyses sequences of regime classifications from RegimeClassifier to:
  1. Build empirical transition probability matrices
  2. Track regime durations and transition frequencies
  3. Generate early-warning signals when VIX/trend indicators approach
     regime boundary thresholds
  4. Produce an HTML report with regime timeline + transition heatmap

Does NOT depend on hmmlearn — uses observed frequencies from historical
regime sequences, which is more interpretable and avoids hidden-state
estimation artifacts on small datasets.

Usage::

    from compass.regime_transitions import RegimeTransitionDetector
    from compass.regime import RegimeClassifier

    classifier = RegimeClassifier()
    regimes = classifier.classify_series(spy_data, vix_series)

    detector = RegimeTransitionDetector()
    detector.fit(regimes)

    # Transition matrix
    print(detector.transition_matrix)

    # Early warning
    warning = detector.early_warning(current_vix=28.0, vix_5d_change=+6.0)

    # HTML report
    html = detector.generate_report(regimes, spy_data, vix_series)
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from compass.regime import Regime

logger = logging.getLogger(__name__)

# Canonical regime order for matrix display
REGIME_ORDER = [Regime.BULL, Regime.BEAR, Regime.HIGH_VOL, Regime.LOW_VOL, Regime.CRASH]
REGIME_LABELS = [r.value for r in REGIME_ORDER]


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class TransitionEvent:
    """A single regime transition."""
    date: pd.Timestamp
    from_regime: str
    to_regime: str
    duration_days: int  # how long the previous regime lasted


@dataclass
class EarlyWarning:
    """Early warning signal for an impending regime change."""
    current_regime: str
    likely_next: str
    probability: float  # 0-1, based on historical transition frequency
    trigger_reason: str
    severity: str  # "low", "medium", "high"


@dataclass
class RegimeSummary:
    """Summary statistics for one regime."""
    regime: str
    total_days: int
    n_episodes: int
    avg_duration: float
    median_duration: float
    min_duration: int
    max_duration: int
    pct_of_total: float


# ── Transition detector ──────────────────────────────────────────────────


class RegimeTransitionDetector:
    """Analyses regime sequences for transition patterns and early warnings.

    Attributes:
        transition_matrix: DataFrame with from-regime as rows, to-regime as
            columns, values = empirical transition probabilities.
        transition_counts: Raw count matrix (same shape).
        transitions: List of TransitionEvent objects.
        regime_summaries: Per-regime duration statistics.
    """

    def __init__(self) -> None:
        self.transition_matrix: pd.DataFrame = pd.DataFrame()
        self.transition_counts: pd.DataFrame = pd.DataFrame()
        self.transitions: List[TransitionEvent] = []
        self.regime_summaries: List[RegimeSummary] = []
        self._fitted = False

    def fit(self, regime_series: pd.Series) -> "RegimeTransitionDetector":
        """Fit the detector on a historical regime sequence.

        Args:
            regime_series: pd.Series of regime labels (str or Regime enum),
                indexed by date, sorted chronologically.

        Returns:
            self (for chaining).
        """
        regimes = regime_series.dropna()
        if len(regimes) < 2:
            logger.warning("Need at least 2 regime observations to fit")
            self._fitted = True
            return self

        # Normalise to string labels
        labels = [r.value if isinstance(r, Regime) else str(r) for r in regimes]
        dates = list(regimes.index)

        # Extract transitions
        self.transitions = []
        episode_start = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                duration = i - episode_start
                self.transitions.append(TransitionEvent(
                    date=dates[i],
                    from_regime=labels[i - 1],
                    to_regime=labels[i],
                    duration_days=duration,
                ))
                episode_start = i

        # Build count matrix
        all_labels = REGIME_LABELS
        counts = pd.DataFrame(0, index=all_labels, columns=all_labels)
        for t in self.transitions:
            if t.from_regime in counts.index and t.to_regime in counts.columns:
                counts.loc[t.from_regime, t.to_regime] += 1

        self.transition_counts = counts

        # Normalise to probabilities
        row_sums = counts.sum(axis=1)
        prob = counts.copy().astype(float)
        for regime in all_labels:
            if row_sums[regime] > 0:
                prob.loc[regime] = counts.loc[regime] / row_sums[regime]
            else:
                prob.loc[regime] = 0.0
        self.transition_matrix = prob.round(4)

        # Compute regime summaries
        self.regime_summaries = self._compute_summaries(labels, dates)

        self._fitted = True
        logger.info(
            "Fitted on %d days, %d transitions across %d regime types",
            len(labels), len(self.transitions), len(set(labels)),
        )
        return self

    def early_warning(
        self,
        current_regime: str = "bull",
        current_vix: float = 20.0,
        vix_5d_change: float = 0.0,
        regime_duration_days: int = 0,
    ) -> Optional[EarlyWarning]:
        """Generate an early warning if regime change seems likely.

        Uses a rule-based approach combining:
          1. Historical transition probability from current regime
          2. VIX proximity to regime boundary thresholds
          3. VIX rate of change (sharp moves signal instability)
          4. Duration exhaustion (regimes lasting much longer than average)

        Returns:
            EarlyWarning if conditions met, None otherwise.
        """
        if not self._fitted:
            return None

        # Most likely next regime from transition matrix
        if current_regime in self.transition_matrix.index:
            probs = self.transition_matrix.loc[current_regime]
            # Exclude self-transition
            other_probs = probs.drop(current_regime, errors="ignore")
            if other_probs.sum() > 0:
                likely_next = other_probs.idxmax()
                base_prob = float(other_probs.max())
            else:
                return None
        else:
            return None

        # Adjust probability based on current conditions
        adjusted_prob = base_prob
        reasons = []

        # VIX boundary proximity
        if current_regime == "bull" and current_vix > 22:
            proximity_boost = min(0.3, (current_vix - 22) / 20)
            adjusted_prob += proximity_boost
            reasons.append(f"VIX {current_vix:.0f} approaching bear threshold")
        elif current_regime == "bear" and current_vix < 22:
            proximity_boost = min(0.2, (22 - current_vix) / 15)
            adjusted_prob += proximity_boost
            reasons.append(f"VIX {current_vix:.0f} dropping toward bull zone")
        elif current_regime in ("bull", "low_vol") and current_vix > 28:
            adjusted_prob += 0.25
            reasons.append(f"VIX {current_vix:.0f} elevated for {current_regime}")

        # VIX rate of change
        if abs(vix_5d_change) > 5:
            adjusted_prob += min(0.2, abs(vix_5d_change) / 25)
            direction = "spiking" if vix_5d_change > 0 else "collapsing"
            reasons.append(f"VIX {direction} ({vix_5d_change:+.1f} pts / 5d)")

        # Duration exhaustion
        summaries_by_regime = {s.regime: s for s in self.regime_summaries}
        if current_regime in summaries_by_regime:
            avg_dur = summaries_by_regime[current_regime].avg_duration
            if avg_dur > 0 and regime_duration_days > avg_dur * 1.5:
                adjusted_prob += 0.1
                reasons.append(
                    f"Duration {regime_duration_days}d exceeds avg {avg_dur:.0f}d"
                )

        adjusted_prob = min(adjusted_prob, 1.0)

        # Severity thresholds
        if adjusted_prob >= 0.6:
            severity = "high"
        elif adjusted_prob >= 0.35:
            severity = "medium"
        elif adjusted_prob >= 0.15:
            severity = "low"
        else:
            return None  # below threshold

        return EarlyWarning(
            current_regime=current_regime,
            likely_next=likely_next,
            probability=round(adjusted_prob, 3),
            trigger_reason="; ".join(reasons) if reasons else "Historical transition pattern",
            severity=severity,
        )

    def get_transition_for(self, from_regime: str, to_regime: str) -> float:
        """Get transition probability from one regime to another."""
        if self.transition_matrix.empty:
            return 0.0
        if from_regime in self.transition_matrix.index and to_regime in self.transition_matrix.columns:
            return float(self.transition_matrix.loc[from_regime, to_regime])
        return 0.0

    # ── Report generation ─────────────────────────────────────────────

    def generate_report(
        self,
        regime_series: pd.Series,
        spy_close: Optional[pd.Series] = None,
        vix_series: Optional[pd.Series] = None,
    ) -> str:
        """Generate a standalone HTML report.

        Args:
            regime_series: The regime sequence used for fitting.
            spy_close: Optional SPY close prices for overlay.
            vix_series: Optional VIX series for overlay.

        Returns:
            HTML string.
        """
        if not self._fitted:
            self.fit(regime_series)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        n_days = len(regime_series)
        n_transitions = len(self.transitions)

        # Transition matrix heatmap
        matrix_html = self._render_transition_matrix()

        # Regime summaries table
        summary_html = self._render_summaries()

        # Timeline
        timeline_html = self._render_timeline(regime_series, spy_close, vix_series)

        # Recent transitions table
        recent_html = self._render_recent_transitions(20)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regime Transition Analysis</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:#f8fafc;color:#1e293b;line-height:1.5;padding:24px;max-width:1200px;margin:0 auto}}
h1{{font-size:1.6em;font-weight:700;margin-bottom:4px}}
h2{{font-size:1.15em;font-weight:600;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}}
.subtitle{{color:#64748b;font-size:0.9em;margin-bottom:20px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px 20px;
min-width:160px;flex:1;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.card-title{{font-size:0.78em;color:#64748b;text-transform:uppercase;letter-spacing:.5px}}
.card-value{{font-size:1.5em;font-weight:700}}
.card-sub{{font-size:0.8em;color:#94a3b8;margin-top:2px}}
table{{border-collapse:collapse;width:100%;font-size:0.88em;margin-bottom:16px}}
th{{background:#f1f5f9;padding:7px 10px;text-align:left;font-weight:600;border-bottom:2px solid #e2e8f0}}
td{{padding:6px 10px;border-bottom:1px solid #f1f5f9}}
.timeline{{display:flex;height:32px;border-radius:4px;overflow:hidden;margin:12px 0}}
.timeline-seg{{height:100%;display:inline-block}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:0.82em;margin-bottom:16px}}
.legend-item{{display:flex;align-items:center;gap:4px}}
.legend-dot{{width:12px;height:12px;border-radius:2px}}
hr{{margin:28px 0;border:none;border-top:1px solid #e2e8f0}}
</style>
</head>
<body>

<h1>Regime Transition Analysis</h1>
<p class="subtitle">{n_days:,} trading days &middot; {n_transitions} transitions &middot; Generated {now}</p>

<div class="cards">
<div class="card"><div class="card-title">Trading Days</div><div class="card-value">{n_days:,}</div></div>
<div class="card"><div class="card-title">Transitions</div><div class="card-value">{n_transitions}</div>
<div class="card-sub">{n_transitions / max(n_days, 1) * 252:.1f}/yr</div></div>
<div class="card"><div class="card-title">Regime Types</div><div class="card-value">{len(set(r.value if isinstance(r, Regime) else str(r) for r in regime_series.dropna()))}</div></div>
<div class="card"><div class="card-title">Avg Duration</div><div class="card-value">{self._avg_duration():.0f}d</div></div>
</div>

<h2>Regime Timeline</h2>
{timeline_html}

<h2>Transition Probability Matrix</h2>
<p style="font-size:0.85em;color:#64748b;margin-bottom:8px">
Row = current regime, column = next regime. Values = empirical probability of transitioning.
</p>
{matrix_html}

<h2>Regime Duration Summary</h2>
{summary_html}

<h2>Recent Transitions</h2>
{recent_html}

<hr>
<p style="font-size:0.75em;color:#94a3b8">Generated by <code>compass/regime_transitions.py</code></p>
</body>
</html>"""

    # ── Private helpers ───────────────────────────────────────────────

    def _compute_summaries(
        self, labels: List[str], dates: List,
    ) -> List[RegimeSummary]:
        """Compute per-regime duration statistics."""
        episodes: Dict[str, List[int]] = defaultdict(list)

        # Walk through labels tracking episode durations
        if not labels:
            return []
        current = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != current:
                episodes[current].append(i - start)
                current = labels[i]
                start = i
        episodes[current].append(len(labels) - start)

        total_days = len(labels)
        summaries = []
        for regime in REGIME_LABELS:
            durs = episodes.get(regime, [])
            n_days = sum(durs)
            summaries.append(RegimeSummary(
                regime=regime,
                total_days=n_days,
                n_episodes=len(durs),
                avg_duration=np.mean(durs) if durs else 0.0,
                median_duration=float(np.median(durs)) if durs else 0.0,
                min_duration=min(durs) if durs else 0,
                max_duration=max(durs) if durs else 0,
                pct_of_total=round(n_days / total_days * 100, 1) if total_days > 0 else 0.0,
            ))
        return summaries

    def _avg_duration(self) -> float:
        if not self.regime_summaries:
            return 0.0
        all_durs = [s.avg_duration for s in self.regime_summaries if s.avg_duration > 0]
        return np.mean(all_durs) if all_durs else 0.0

    def _render_transition_matrix(self) -> str:
        if self.transition_matrix.empty:
            return "<p><em>No transition data.</em></p>"
        labels = list(self.transition_matrix.index)
        header = "<th>From \\ To</th>" + "".join(f"<th>{l}</th>" for l in labels)
        rows = ""
        for r_label in labels:
            cells = f'<td style="font-weight:600">{r_label}</td>'
            for c_label in labels:
                val = self.transition_matrix.loc[r_label, c_label]
                count = int(self.transition_counts.loc[r_label, c_label])
                if r_label == c_label:
                    bg = "#f1f5f9"
                elif val > 0.4:
                    bg = "#fecaca"
                elif val > 0.2:
                    bg = "#fef3c7"
                elif val > 0:
                    bg = "#d1fae5"
                else:
                    bg = "#fff"
                cells += (
                    f'<td style="background:{bg};text-align:center">'
                    f'{val:.0%}<br><span style="font-size:0.75em;color:#94a3b8">n={count}</span></td>'
                )
            rows += f"<tr>{cells}</tr>"
        return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"

    def _render_summaries(self) -> str:
        if not self.regime_summaries:
            return "<p><em>No data.</em></p>"
        header = (
            "<tr><th>Regime</th><th>Days</th><th>Episodes</th>"
            "<th>Avg Duration</th><th>Median</th><th>Min</th><th>Max</th>"
            "<th>% of Total</th></tr>"
        )
        rows = ""
        for s in self.regime_summaries:
            if s.total_days == 0:
                continue
            rows += (
                f"<tr><td style='font-weight:600'>{s.regime}</td>"
                f"<td>{s.total_days:,}</td><td>{s.n_episodes}</td>"
                f"<td>{s.avg_duration:.1f}</td><td>{s.median_duration:.1f}</td>"
                f"<td>{s.min_duration}</td><td>{s.max_duration}</td>"
                f"<td>{s.pct_of_total:.1f}%</td></tr>"
            )
        return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"

    def _render_timeline(
        self,
        regime_series: pd.Series,
        spy_close: Optional[pd.Series],
        vix_series: Optional[pd.Series],
    ) -> str:
        """Render a color-coded regime timeline bar."""
        colors = {
            "bull": "#22c55e", "bear": "#ef4444", "high_vol": "#f59e0b",
            "low_vol": "#3b82f6", "crash": "#7c3aed",
        }
        labels = [r.value if isinstance(r, Regime) else str(r) for r in regime_series.dropna()]
        if not labels:
            return "<p><em>No regime data.</em></p>"

        # Build segments
        segments = []
        current = labels[0]
        count = 1
        for i in range(1, len(labels)):
            if labels[i] == current:
                count += 1
            else:
                segments.append((current, count))
                current = labels[i]
                count = 1
        segments.append((current, count))

        total = sum(c for _, c in segments)
        bar = ""
        for regime, n in segments:
            pct = n / total * 100
            color = colors.get(regime, "#94a3b8")
            title = f"{regime}: {n} days"
            bar += f'<div class="timeline-seg" style="width:{pct}%;background:{color}" title="{title}"></div>'

        legend = ""
        for regime, color in colors.items():
            legend += f'<div class="legend-item"><div class="legend-dot" style="background:{color}"></div>{regime}</div>'

        date_range = ""
        if len(regime_series) > 0:
            date_range = f"<p style='font-size:0.82em;color:#94a3b8'>{regime_series.index[0].strftime('%Y-%m-%d')} to {regime_series.index[-1].strftime('%Y-%m-%d')}</p>"

        return f'<div class="legend">{legend}</div><div class="timeline">{bar}</div>{date_range}'

    def _render_recent_transitions(self, n: int = 20) -> str:
        if not self.transitions:
            return "<p><em>No transitions recorded.</em></p>"
        recent = self.transitions[-n:]
        header = "<tr><th>Date</th><th>From</th><th>To</th><th>Duration (prev)</th></tr>"
        rows = ""
        for t in reversed(recent):
            rows += (
                f"<tr><td>{t.date.strftime('%Y-%m-%d')}</td>"
                f"<td>{t.from_regime}</td><td>{t.to_regime}</td>"
                f"<td>{t.duration_days}d</td></tr>"
            )
        return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"
