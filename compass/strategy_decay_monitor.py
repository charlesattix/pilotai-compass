"""
Strategy decay monitor — detects alpha decay and manages lifecycle transitions.

Components:
  - Rolling metrics:    Sharpe, hit-rate, avg P&L with configurable windows
  - CUSUM:              Cumulative-sum structural break detection on rolling Sharpe
  - Lifecycle:          emerging → mature → degrading → dead classification
  - Recent performance: composite score from Sharpe, hit rate, P&L
  - Kill score:         weighted composite (0-1) with keep/monitor/retire
  - HTML report:        rolling Sharpe chart, CUSUM timeline, lifecycle badge,
                        kill dashboard

This is READ-ONLY analysis.  No broker connections, no trade placement.

Usage::

    from compass.strategy_decay_monitor import StrategyDecayMonitor
    mon = StrategyDecayMonitor()
    report = mon.monitor("EXP-400", returns_series)
    mon.generate_report(report)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "strategy_decay.html"
TRADING_DAYS = 252


# ── Enums & dataclasses ──────────────────────────────────────────────────


class LifecyclePhase(str, Enum):
    EMERGING = "emerging"
    MATURE = "mature"
    DEGRADING = "degrading"
    DEAD = "dead"


@dataclass
class RollingMetrics:
    """Point-in-time rolling performance snapshot."""

    date: datetime
    sharpe: float
    hit_rate: float
    avg_pnl: float
    cumulative_pnl: float


@dataclass
class CUSUMResult:
    """CUSUM structural break detection on rolling Sharpe."""

    break_detected: bool
    break_index: Optional[int] = None
    break_date: Optional[datetime] = None
    cusum_series: Optional[np.ndarray] = None
    threshold: float = 0.0
    max_cusum: float = 0.0


@dataclass
class RecentPerformance:
    """Composite recent performance scoring."""

    window_days: int
    sharpe: float
    hit_rate: float
    avg_pnl: float
    total_pnl: float
    score: float  # 0-1 composite


@dataclass
class KillSignal:
    """Composite retirement recommendation."""

    strategy_name: str
    lifecycle: LifecyclePhase
    kill_score: float  # 0-1, higher = more urgent to retire
    sharpe_component: float
    hit_rate_component: float
    cusum_component: float
    pnl_trend_component: float
    reasons: List[str]
    confidence: float
    recommendation: str  # "keep", "monitor", "retire"


@dataclass
class MonitorResult:
    """Full output from strategy monitoring."""

    strategy_name: str
    rolling_metrics: List[RollingMetrics]
    cusum: CUSUMResult
    lifecycle: LifecyclePhase
    recent_performance: RecentPerformance
    kill_signal: KillSignal
    n_observations: int


# ── Rolling metrics engine ───────────────────────────────────────────────


def compute_rolling_sharpe(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling annualised Sharpe ratio."""
    n = len(returns)
    if n < window:
        return np.array([])

    out = np.empty(n - window + 1)
    for i in range(len(out)):
        chunk = returns[i : i + window]
        mu = chunk.mean()
        std = chunk.std(ddof=1)
        out[i] = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
    return out


def compute_rolling_hit_rate(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling hit rate (fraction of positive returns)."""
    n = len(returns)
    if n < window:
        return np.array([])

    out = np.empty(n - window + 1)
    for i in range(len(out)):
        chunk = returns[i : i + window]
        out[i] = (chunk > 0).sum() / window
    return out


# ── CUSUM on rolling Sharpe ──────────────────────────────────────────────


def cusum_on_sharpe(
    rolling_sharpe: np.ndarray,
    threshold: float = 3.0,
    dates: Optional[pd.DatetimeIndex] = None,
) -> CUSUMResult:
    """CUSUM structural break detection on rolling Sharpe series.

    Detects downward mean shift (declining alpha).  Uses negative CUSUM:
    S(t) = max(0, S(t-1) + (mu - x(t)) / sigma).
    """
    if len(rolling_sharpe) < 10:
        return CUSUMResult(break_detected=False, threshold=threshold)

    mu = float(rolling_sharpe.mean())
    sigma = float(rolling_sharpe.std(ddof=1))
    if sigma < 1e-12:
        return CUSUMResult(break_detected=False, threshold=threshold)

    n = len(rolling_sharpe)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = max(0.0, s[i - 1] + (mu - rolling_sharpe[i]) / sigma)

    max_s = float(s.max())
    break_idx = int(s.argmax()) if max_s > threshold else None
    break_dt = None
    if break_idx is not None and dates is not None and break_idx < len(dates):
        break_dt = dates[break_idx]

    return CUSUMResult(
        break_detected=max_s > threshold,
        break_index=break_idx,
        break_date=break_dt,
        cusum_series=s,
        threshold=threshold,
        max_cusum=max_s,
    )


# ── Lifecycle classification ─────────────────────────────────────────────


def classify_lifecycle(
    rolling_sharpe: np.ndarray,
    sharpe_threshold: float = 0.3,
) -> LifecyclePhase:
    """Classify strategy lifecycle from rolling Sharpe trajectory.

    EMERGING  — rising Sharpe, positive alpha, less data
    MATURE    — stable Sharpe above threshold
    DEGRADING — declining Sharpe or below threshold
    DEAD      — sustained negative Sharpe
    """
    if len(rolling_sharpe) < 5:
        return LifecyclePhase.EMERGING

    quarter = max(len(rolling_sharpe) // 4, 1)
    recent = rolling_sharpe[-quarter:]
    recent_mean = float(recent.mean())

    # Linear trend
    x = np.arange(len(rolling_sharpe), dtype=float)
    n = len(x)
    x_mean = x.mean()
    y_mean = rolling_sharpe.mean()
    slope = float(
        ((x - x_mean) * (rolling_sharpe - y_mean)).sum()
        / max(((x - x_mean) ** 2).sum(), 1e-12)
    )

    if recent_mean < 0 and slope < 0:
        return LifecyclePhase.DEAD
    if slope < -0.002 or recent_mean < sharpe_threshold:
        return LifecyclePhase.DEGRADING
    if slope > 0.002 and recent_mean > 0.5:
        return LifecyclePhase.EMERGING
    return LifecyclePhase.MATURE


# ── Recent performance scoring ───────────────────────────────────────────


def score_recent_performance(
    returns: pd.Series,
    window: int = 63,
) -> RecentPerformance:
    """Score recent strategy performance on 0-1 scale.

    Components (equally weighted):
      - Sharpe score: sigmoid-mapped annualised Sharpe
      - Hit rate score: linear 0.3-0.7 → 0-1
      - PnL score: based on positive/negative trend
    """
    recent = returns.iloc[-window:] if len(returns) >= window else returns
    if len(recent) < 2:
        return RecentPerformance(
            window_days=len(recent), sharpe=0.0, hit_rate=0.0,
            avg_pnl=0.0, total_pnl=0.0, score=0.0,
        )

    mu = float(recent.mean())
    std = float(recent.std(ddof=1))
    sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
    hit_rate = float((recent > 0).sum() / len(recent))
    total_pnl = float(recent.sum())

    # Sharpe score: sigmoid mapping [-2, 4] → [0, 1]
    sharpe_score = 1.0 / (1.0 + math.exp(-1.5 * (sharpe - 0.5)))

    # Hit rate score: linear [0.3, 0.7] → [0, 1]
    hr_score = max(0.0, min(1.0, (hit_rate - 0.3) / 0.4))

    # PnL score: based on sign and magnitude relative to volatility
    if std > 1e-12:
        pnl_z = mu / std * math.sqrt(len(recent))
        pnl_score = 1.0 / (1.0 + math.exp(-pnl_z))
    else:
        pnl_score = 0.5

    composite = (sharpe_score + hr_score + pnl_score) / 3.0

    return RecentPerformance(
        window_days=len(recent),
        sharpe=sharpe,
        hit_rate=hit_rate,
        avg_pnl=mu,
        total_pnl=total_pnl,
        score=composite,
    )


# ── Kill score computation ───────────────────────────────────────────────


def compute_kill_score(
    lifecycle: LifecyclePhase,
    cusum: CUSUMResult,
    recent: RecentPerformance,
    rolling_sharpe: np.ndarray,
    sharpe_threshold: float = 0.3,
    hit_rate_threshold: float = 0.40,
) -> Tuple[float, Dict[str, float], List[str]]:
    """Compute composite kill score (0-1).

    Components (weighted):
      - Sharpe component  (0.35): based on recent and rolling Sharpe
      - Hit rate component (0.20): below threshold contributes
      - CUSUM component   (0.25): structural break detection
      - PnL trend         (0.20): declining recent performance

    Returns:
        (kill_score, component_dict, reasons_list)
    """
    reasons: List[str] = []

    # 1) Sharpe component (0.35)
    sharpe_comp = 0.0
    if recent.sharpe < 0:
        sharpe_comp = 1.0
        reasons.append(f"negative_sharpe({recent.sharpe:.2f})")
    elif recent.sharpe < sharpe_threshold:
        sharpe_comp = 1.0 - recent.sharpe / sharpe_threshold
        reasons.append(f"low_sharpe({recent.sharpe:.2f})")
    # Check Sharpe trend
    if len(rolling_sharpe) >= 20:
        first_half = rolling_sharpe[: len(rolling_sharpe) // 2].mean()
        second_half = rolling_sharpe[len(rolling_sharpe) // 2 :].mean()
        if second_half < first_half * 0.5:
            sharpe_comp = max(sharpe_comp, 0.7)
            reasons.append("sharpe_declining_>50%")

    # 2) Hit rate component (0.20)
    hr_comp = 0.0
    if recent.hit_rate < hit_rate_threshold:
        hr_comp = 1.0 - recent.hit_rate / hit_rate_threshold
        reasons.append(f"low_hit_rate({recent.hit_rate:.1%})")

    # 3) CUSUM component (0.25)
    cusum_comp = 0.0
    if cusum.break_detected:
        # Scale by how far above threshold
        cusum_comp = min(1.0, cusum.max_cusum / (cusum.threshold * 2))
        reasons.append("structural_break_detected")

    # 4) PnL trend component (0.20)
    pnl_comp = 0.0
    if recent.total_pnl < 0:
        pnl_comp = min(1.0, abs(recent.total_pnl) / max(abs(recent.avg_pnl) * recent.window_days * 0.5, 1e-6))
        pnl_comp = min(pnl_comp, 1.0)
        reasons.append(f"negative_pnl({recent.total_pnl:.4f})")

    # Lifecycle override
    if lifecycle == LifecyclePhase.DEAD:
        reasons.append(f"lifecycle={lifecycle.value}")

    score = (
        0.35 * sharpe_comp
        + 0.20 * hr_comp
        + 0.25 * cusum_comp
        + 0.20 * pnl_comp
    )
    score = min(1.0, max(0.0, score))

    # Dead lifecycle floor
    if lifecycle == LifecyclePhase.DEAD:
        score = max(score, 0.7)

    components = {
        "sharpe": sharpe_comp,
        "hit_rate": hr_comp,
        "cusum": cusum_comp,
        "pnl_trend": pnl_comp,
    }
    return score, components, reasons


# ── Core monitor class ───────────────────────────────────────────────────


class StrategyDecayMonitor:
    """Detects strategy alpha decay and manages lifecycle transitions.

    Args:
        rolling_window: Window for rolling Sharpe / metrics (days).
        cusum_threshold: Std devs for CUSUM break detection.
        sharpe_threshold: Sharpe below this triggers kill contribution.
        hit_rate_threshold: Hit rate below this triggers kill contribution.
        recent_window: Window for recent performance scoring.
    """

    def __init__(
        self,
        rolling_window: int = 63,
        cusum_threshold: float = 3.0,
        sharpe_threshold: float = 0.3,
        hit_rate_threshold: float = 0.40,
        recent_window: int = 63,
    ) -> None:
        self.rolling_window = rolling_window
        self.cusum_threshold = cusum_threshold
        self.sharpe_threshold = sharpe_threshold
        self.hit_rate_threshold = hit_rate_threshold
        self.recent_window = recent_window

    # ── Rolling metrics ──────────────────────────────────────────────

    def compute_rolling_metrics(
        self,
        returns: pd.Series,
        window: Optional[int] = None,
    ) -> List[RollingMetrics]:
        """Compute rolling Sharpe, hit rate, avg P&L."""
        w = window or self.rolling_window
        vals = returns.values
        if len(vals) < w:
            return []

        sharpes = compute_rolling_sharpe(vals, w)
        hit_rates = compute_rolling_hit_rate(vals, w)
        cumsum = np.cumsum(vals)
        dates = returns.index[w - 1 :]

        results: List[RollingMetrics] = []
        for i in range(len(sharpes)):
            chunk = vals[i : i + w]
            results.append(
                RollingMetrics(
                    date=dates[i],
                    sharpe=float(sharpes[i]),
                    hit_rate=float(hit_rates[i]),
                    avg_pnl=float(chunk.mean()),
                    cumulative_pnl=float(cumsum[i + w - 1]),
                )
            )
        return results

    # ── CUSUM ────────────────────────────────────────────────────────

    def cusum_test(
        self,
        returns: pd.Series,
        threshold: Optional[float] = None,
    ) -> CUSUMResult:
        """Run CUSUM structural break detection on rolling Sharpe."""
        th = threshold or self.cusum_threshold
        w = self.rolling_window
        vals = returns.values

        rolling_sh = compute_rolling_sharpe(vals, w)
        if len(rolling_sh) < 10:
            return CUSUMResult(break_detected=False, threshold=th)

        dates = returns.index[w - 1 :] if hasattr(returns, "index") else None
        return cusum_on_sharpe(rolling_sh, th, dates)

    # ── Lifecycle ────────────────────────────────────────────────────

    def classify_lifecycle(self, returns: pd.Series) -> LifecyclePhase:
        """Classify strategy lifecycle phase."""
        rolling_sh = compute_rolling_sharpe(
            returns.values, self.rolling_window
        )
        return classify_lifecycle(rolling_sh, self.sharpe_threshold)

    # ── Recent performance ───────────────────────────────────────────

    def score_recent(self, returns: pd.Series) -> RecentPerformance:
        """Score recent performance."""
        return score_recent_performance(returns, self.recent_window)

    # ── Kill signal ──────────────────────────────────────────────────

    def kill_signal(
        self,
        strategy_name: str,
        returns: pd.Series,
    ) -> KillSignal:
        """Compute composite kill recommendation."""
        rolling_sh = compute_rolling_sharpe(
            returns.values, self.rolling_window
        )
        lifecycle = classify_lifecycle(rolling_sh, self.sharpe_threshold)
        cusum = self.cusum_test(returns)
        recent = self.score_recent(returns)

        score, comps, reasons = compute_kill_score(
            lifecycle,
            cusum,
            recent,
            rolling_sh,
            self.sharpe_threshold,
            self.hit_rate_threshold,
        )

        metrics = self.compute_rolling_metrics(returns)
        confidence = min(len(metrics) / 50.0, 1.0)

        if score >= 0.7:
            rec = "retire"
        elif score >= 0.4:
            rec = "monitor"
        else:
            rec = "keep"

        return KillSignal(
            strategy_name=strategy_name,
            lifecycle=lifecycle,
            kill_score=score,
            sharpe_component=comps["sharpe"],
            hit_rate_component=comps["hit_rate"],
            cusum_component=comps["cusum"],
            pnl_trend_component=comps["pnl_trend"],
            reasons=reasons,
            confidence=confidence,
            recommendation=rec,
        )

    # ── Full monitor ─────────────────────────────────────────────────

    def monitor(
        self,
        strategy_name: str,
        returns: pd.Series,
    ) -> MonitorResult:
        """Run full monitoring pipeline."""
        metrics = self.compute_rolling_metrics(returns)
        cusum = self.cusum_test(returns)
        lifecycle = self.classify_lifecycle(returns)
        recent = self.score_recent(returns)
        kill = self.kill_signal(strategy_name, returns)

        return MonitorResult(
            strategy_name=strategy_name,
            rolling_metrics=metrics,
            cusum=cusum,
            lifecycle=lifecycle,
            recent_performance=recent,
            kill_signal=kill,
            n_observations=len(returns),
        )

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        result: MonitorResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        """Generate self-contained HTML report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────

LIFECYCLE_COLORS = {
    LifecyclePhase.EMERGING: "#3fb950",
    LifecyclePhase.MATURE: "#58a6ff",
    LifecyclePhase.DEGRADING: "#d29922",
    LifecyclePhase.DEAD: "#f85149",
}

REC_COLORS = {"keep": "#3fb950", "monitor": "#d29922", "retire": "#f85149"}


def _svg_line(
    values: List[float],
    title: str,
    width: int = 700,
    height: int = 220,
    threshold: Optional[float] = None,
    color: str = "#58a6ff",
) -> str:
    """Generic SVG line chart."""
    if len(values) < 2:
        return ""

    n = len(values)
    pad = 55
    pw = width - 2 * pad
    ph = height - 70

    y_min = min(min(values), threshold or 0, 0)
    y_max = max(max(values), threshold or 0) * 1.15
    if y_max <= y_min:
        y_max = y_min + 0.1

    def tx(i: int) -> float:
        return pad + i / max(n - 1, 1) * pw

    def ty(v: float) -> float:
        return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart">',
        f'<text x="{width // 2}" y="20" text-anchor="middle" class="svg-title">{title}</text>',
    ]

    # Zero line
    if y_min < 0:
        zy = ty(0)
        parts.append(
            f'<line x1="{pad}" y1="{zy:.0f}" x2="{width - pad}" y2="{zy:.0f}" '
            f'stroke="#30363d" stroke-dasharray="3,3"/>'
        )

    # Threshold line
    if threshold is not None:
        thy = ty(threshold)
        parts.append(
            f'<line x1="{pad}" y1="{thy:.0f}" x2="{width - pad}" y2="{thy:.0f}" '
            f'stroke="#f85149" stroke-width="1" stroke-dasharray="4,3"/>'
        )
        parts.append(
            f'<text x="{width - pad + 3}" y="{thy + 4:.0f}" '
            f'class="svg-label" fill="#f85149">{threshold:.1f}</text>'
        )

    d = " ".join(
        f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
        for i, v in enumerate(values)
    )
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_cusum(cusum: CUSUMResult) -> str:
    """SVG CUSUM chart with threshold and break marker."""
    if cusum.cusum_series is None or len(cusum.cusum_series) < 2:
        return "<p class='meta'>Insufficient data for CUSUM chart.</p>"

    vals = cusum.cusum_series.tolist()
    n = len(vals)
    w, h = 700, 200
    pad = 55
    pw = w - 2 * pad
    ph = h - 70

    y_max = max(max(vals), cusum.threshold) * 1.2
    if y_max <= 0:
        y_max = 1.0

    def tx(i: int) -> float:
        return pad + i / max(n - 1, 1) * pw

    def ty(v: float) -> float:
        return 35 + (1 - v / y_max) * ph

    parts = [
        f'<svg viewBox="0 0 {w} {h}" class="chart">',
        f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">'
        f"CUSUM Structural Break Detection</text>",
    ]

    # Threshold
    thy = ty(cusum.threshold)
    parts.append(
        f'<line x1="{pad}" y1="{thy:.0f}" x2="{w - pad}" y2="{thy:.0f}" '
        f'stroke="#f85149" stroke-dasharray="4,3"/>'
    )
    parts.append(
        f'<text x="{w - pad + 3}" y="{thy + 4:.0f}" '
        f'class="svg-label" fill="#f85149">threshold</text>'
    )

    d = " ".join(
        f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
        for i, v in enumerate(vals)
    )
    parts.append(f'<path d="{d}" fill="none" stroke="#d29922" stroke-width="2"/>')

    if cusum.break_detected and cusum.break_index is not None:
        bx = tx(cusum.break_index)
        by = ty(vals[cusum.break_index])
        parts.append(f'<circle cx="{bx:.0f}" cy="{by:.0f}" r="5" fill="#f85149"/>')

    parts.append("</svg>")
    return "\n".join(parts)


def _kill_dashboard(ks: KillSignal) -> str:
    """Kill score dashboard with component bars."""
    rec_color = REC_COLORS.get(ks.recommendation, "#8b949e")
    lc_color = LIFECYCLE_COLORS.get(ks.lifecycle, "#8b949e")

    def _bar(label: str, value: float, weight: str) -> str:
        pct = value * 100
        return (
            f'<div class="kill-row">'
            f'<span class="label">{label} ({weight})</span>'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.0f}%"></div></div>'
            f'<span class="value">{value:.2f}</span>'
            f"</div>"
        )

    reasons_html = ""
    if ks.reasons:
        items = "".join(f"<li>{r}</li>" for r in ks.reasons)
        reasons_html = f'<ul class="reasons">{items}</ul>'

    return f"""
    <div class="kill-dashboard">
      <div class="kill-header">
        <div class="kill-score" style="color:{rec_color}">{ks.kill_score:.2f}</div>
        <div class="kill-label">Kill Score</div>
        <div class="kill-rec" style="background:{rec_color}">{ks.recommendation.upper()}</div>
      </div>
      <div class="kill-meta">
        <span>Lifecycle: <span class="phase" style="background:{lc_color}">{ks.lifecycle.value.upper()}</span></span>
        <span>Confidence: {ks.confidence:.0%}</span>
      </div>
      {_bar("Sharpe", ks.sharpe_component, "35%")}
      {_bar("CUSUM", ks.cusum_component, "25%")}
      {_bar("Hit Rate", ks.hit_rate_component, "20%")}
      {_bar("PnL Trend", ks.pnl_trend_component, "20%")}
      {reasons_html}
    </div>"""


def _recent_card(rp: RecentPerformance) -> str:
    return f"""
    <div class="card">
      <h3>Recent Performance ({rp.window_days}d)</h3>
      <div class="metrics-grid">
        <div><span class="label">Sharpe</span><span class="value">{rp.sharpe:.2f}</span></div>
        <div><span class="label">Hit Rate</span><span class="value">{rp.hit_rate:.1%}</span></div>
        <div><span class="label">Avg P&amp;L</span><span class="value">{rp.avg_pnl:.6f}</span></div>
        <div><span class="label">Total P&amp;L</span><span class="value">{rp.total_pnl:.4f}</span></div>
        <div><span class="label">Score</span><span class="value">{rp.score:.2f}</span></div>
      </div>
    </div>"""


def _build_html(result: MonitorResult) -> str:
    metrics = result.rolling_metrics
    cusum = result.cusum
    ks = result.kill_signal
    rp = result.recent_performance

    sharpe_vals = [m.sharpe for m in metrics] if metrics else []
    hit_vals = [m.hit_rate for m in metrics] if metrics else []
    pnl_vals = [m.cumulative_pnl for m in metrics] if metrics else []

    sharpe_svg = _svg_line(
        sharpe_vals, "Rolling Sharpe Ratio", threshold=0.3, color="#58a6ff"
    )
    cusum_svg = _svg_cusum(cusum)
    hit_svg = _svg_line(hit_vals, "Rolling Hit Rate", threshold=0.4, color="#3fb950")
    pnl_svg = _svg_line(pnl_vals, "Cumulative P&L", color="#d29922")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Strategy Decay Monitor: {result.strategy_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .top-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; }}
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .metrics-grid .label {{ color: #8b949e; font-size: 0.85em; }}
  .metrics-grid .value {{ color: #f0f6fc; font-weight: 600; }}
  .kill-dashboard {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
                     padding: 24px; }}
  .kill-header {{ text-align: center; margin-bottom: 16px; }}
  .kill-score {{ font-size: 3.5em; font-weight: 800; }}
  .kill-label {{ color: #8b949e; font-size: 1.1em; }}
  .kill-rec {{ display: inline-block; padding: 4px 16px; border-radius: 12px;
               color: #fff; font-weight: 700; margin-top: 8px; }}
  .kill-meta {{ display: flex; justify-content: space-between; margin-bottom: 12px; }}
  .kill-row {{ display: flex; align-items: center; gap: 8px; margin: 6px 0; }}
  .kill-row .label {{ width: 140px; text-align: right; color: #8b949e; font-size: 0.85em; }}
  .kill-row .value {{ width: 50px; font-weight: 600; font-size: 0.85em; }}
  .bar-bg {{ flex: 1; height: 10px; background: #21262d; border-radius: 5px; }}
  .bar-fill {{ height: 100%; background: #f85149; border-radius: 5px; }}
  .phase {{ display: inline-block; padding: 2px 10px; border-radius: 10px;
            color: #fff; font-weight: 700; font-size: 0.85em; }}
  .chart {{ width: 100%; max-width: 750px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  .svg-label {{ fill: #8b949e; font-size: 10px; }}
  ul.reasons {{ color: #8b949e; font-size: 0.9em; margin-top: 12px; }}
</style>
</head>
<body>
<h1>Strategy Decay Monitor: {result.strategy_name}</h1>
<p class="meta">{result.n_observations} observations &middot;
   Rolling window: {len(sharpe_vals)} points</p>

<div class="top-grid">
  {_kill_dashboard(ks)}
  {_recent_card(rp)}
</div>

<h2>Rolling Sharpe</h2>
{sharpe_svg}

<h2>CUSUM Break Detection</h2>
{cusum_svg}

<h2>Rolling Hit Rate</h2>
{hit_svg}

<h2>Cumulative P&amp;L</h2>
{pnl_svg}

</body>
</html>"""
