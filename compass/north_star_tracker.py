"""
North Star progress tracker — monitors all MASTERPLAN targets.

Targets (from MASTERPLAN.md):
  - Annual return: 55%
  - Sharpe ratio: 6.0
  - Max drawdown: ≤ 30%

Tracks per-experiment and portfolio-level metrics, gap analysis,
trajectory projections, milestone history, improvement velocity.

All methods work on pre-loaded data — no network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

NORTH_STAR_TARGETS = {
    "annual_return": 0.55,
    "sharpe": 6.0,
    "max_drawdown": 0.30,
}


@dataclass
class MetricSnapshot:
    date: datetime
    annual_return: float
    sharpe: float
    max_drawdown: float


@dataclass
class GapAnalysis:
    metric: str
    current: float
    target: float
    gap: float
    pct_achieved: float
    is_met: bool


@dataclass
class Milestone:
    metric: str
    value: float
    date: datetime
    description: str


@dataclass
class TrajectoryProjection:
    metric: str
    current: float
    target: float
    improvement_rate: float    # per month
    months_to_target: float
    projected_date: Optional[str] = None


@dataclass
class ExperimentMetrics:
    experiment_id: str
    annual_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int = 0


@dataclass
class TrackerReport:
    current: MetricSnapshot
    gaps: List[GapAnalysis]
    milestones: List[Milestone]
    projections: List[TrajectoryProjection]
    experiments: List[ExperimentMetrics]
    velocity: Dict[str, float]
    overall_progress: float


class NorthStarTracker:
    """North Star progress tracker.

    Args:
        targets: Override MASTERPLAN targets.
    """

    def __init__(self, targets: Optional[Dict[str, float]] = None) -> None:
        self.targets = targets or dict(NORTH_STAR_TARGETS)
        self._snapshots: List[MetricSnapshot] = []
        self._milestones: List[Milestone] = []

    # ------------------------------------------------------------------
    # Record metrics
    # ------------------------------------------------------------------

    def record(
        self, annual_return: float, sharpe: float, max_drawdown: float,
        date: Optional[datetime] = None,
    ) -> MetricSnapshot:
        dt = date or datetime.now()
        snap = MetricSnapshot(dt, annual_return, sharpe, max_drawdown)
        self._snapshots.append(snap)
        self._check_milestones(snap)
        return snap

    def _check_milestones(self, snap: MetricSnapshot) -> None:
        for metric in ["annual_return", "sharpe", "max_drawdown"]:
            val = getattr(snap, metric)
            # Check if this is a new best
            prev_bests = [getattr(s, metric) for s in self._snapshots[:-1]]
            if not prev_bests:
                continue
            if metric == "max_drawdown":
                if val < min(prev_bests):
                    self._milestones.append(Milestone(
                        metric, val, snap.date, f"New best {metric}: {val:.2%}"))
            else:
                if val > max(prev_bests):
                    self._milestones.append(Milestone(
                        metric, val, snap.date, f"New best {metric}: {val:.2f}"))

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def compute_gaps(self, snapshot: MetricSnapshot) -> List[GapAnalysis]:
        gaps: List[GapAnalysis] = []
        for metric, target in self.targets.items():
            current = getattr(snapshot, metric)
            if metric == "max_drawdown":
                gap = current - target
                is_met = current <= target
                pct = target / current if current > 0 else 1.0
            else:
                gap = target - current
                is_met = current >= target
                pct = current / target if target > 0 else 0.0
            gaps.append(GapAnalysis(metric, current, target, gap, pct, is_met))
        return gaps

    # ------------------------------------------------------------------
    # Trajectory projection
    # ------------------------------------------------------------------

    def project_trajectory(self) -> List[TrajectoryProjection]:
        if len(self._snapshots) < 2:
            return [TrajectoryProjection(m, 0, t, 0, float("inf"))
                    for m, t in self.targets.items()]

        projections: List[TrajectoryProjection] = []
        n = len(self._snapshots)
        months = n / 21  # approximate trading days per month

        for metric, target in self.targets.items():
            first = getattr(self._snapshots[0], metric)
            last = getattr(self._snapshots[-1], metric)

            if metric == "max_drawdown":
                rate = (first - last) / max(months, 1)  # decreasing is good
                remaining = last - target
                months_to = remaining / rate if rate > 1e-8 else float("inf")
            else:
                rate = (last - first) / max(months, 1)
                remaining = target - last
                months_to = remaining / rate if rate > 1e-8 else float("inf")

            projections.append(TrajectoryProjection(
                metric, last, target, rate,
                max(0, months_to),
            ))
        return projections

    # ------------------------------------------------------------------
    # Improvement velocity
    # ------------------------------------------------------------------

    def improvement_velocity(self) -> Dict[str, float]:
        """Monthly improvement rate for each metric."""
        if len(self._snapshots) < 2:
            return {m: 0.0 for m in self.targets}
        n = len(self._snapshots)
        months = max(n / 21, 1)
        vel: Dict[str, float] = {}
        for metric in self.targets:
            first = getattr(self._snapshots[0], metric)
            last = getattr(self._snapshots[-1], metric)
            if metric == "max_drawdown":
                vel[metric] = (first - last) / months  # positive = improving
            else:
                vel[metric] = (last - first) / months
        return vel

    # ------------------------------------------------------------------
    # Per-experiment analysis
    # ------------------------------------------------------------------

    @staticmethod
    def experiment_metrics(
        experiment_returns: Dict[str, pd.Series],
    ) -> List[ExperimentMetrics]:
        results: List[ExperimentMetrics] = []
        for eid, rets in experiment_returns.items():
            if rets.empty or len(rets) < 10:
                results.append(ExperimentMetrics(eid, 0, 0, 0))
                continue
            mu = float(rets.mean())
            std = float(rets.std())
            sharpe = mu / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
            total = float((1 + rets).prod() - 1)
            n_years = len(rets) / TRADING_DAYS
            annual = (1 + total) ** (1 / max(n_years, 0.01)) - 1
            eq = (1 + rets).cumprod()
            dd = float((1 - eq / eq.expanding().max()).max())
            results.append(ExperimentMetrics(
                eid, annual, sharpe, dd, n_trades=len(rets)))
        results.sort(key=lambda e: e.sharpe, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def track(
        self,
        annual_return: float, sharpe: float, max_drawdown: float,
        experiment_returns: Optional[Dict[str, pd.Series]] = None,
        date: Optional[datetime] = None,
    ) -> TrackerReport:
        snap = self.record(annual_return, sharpe, max_drawdown, date)
        gaps = self.compute_gaps(snap)
        projections = self.project_trajectory()
        velocity = self.improvement_velocity()
        exps = self.experiment_metrics(experiment_returns) if experiment_returns else []
        progress = sum(1 for g in gaps if g.is_met) / len(gaps) if gaps else 0.0

        return TrackerReport(
            current=snap, gaps=gaps, milestones=self._milestones,
            projections=projections, experiments=exps,
            velocity=velocity, overall_progress=progress,
        )

    @property
    def snapshots(self) -> List[MetricSnapshot]:
        return list(self._snapshots)

    @property
    def milestones(self) -> List[Milestone]:
        return list(self._milestones)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, report: TrackerReport,
        output_path: str = "reports/north_star.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        c = report.current
        gap_rows = [
            f"<tr><td>{g.metric}</td><td>{g.current:.2%}</td><td>{g.target:.2%}</td>"
            f"<td>{g.gap:+.2%}</td><td>{g.pct_achieved:.0%}</td>"
            f"<td style='color:{'#27ae60' if g.is_met else '#e74c3c'}'>"
            f"{'MET' if g.is_met else 'GAP'}</td></tr>"
            for g in report.gaps
        ]
        proj_rows = [
            f"<tr><td>{p.metric}</td><td>{p.current:.2%}</td><td>{p.target:.2%}</td>"
            f"<td>{p.improvement_rate:+.4f}/mo</td>"
            f"<td>{p.months_to_target:.0f} months</td></tr>"
            for p in report.projections
        ]
        exp_rows = [
            f"<tr><td>{e.experiment_id}</td><td>{e.annual_return:.2%}</td>"
            f"<td>{e.sharpe:.2f}</td><td>{e.max_drawdown:.2%}</td>"
            f"<td>{e.n_trades}</td></tr>"
            for e in report.experiments
        ]
        ms_rows = [
            f"<tr><td>{m.date.strftime('%Y-%m-%d') if hasattr(m.date, 'strftime') else m.date}</td>"
            f"<td>{m.metric}</td><td>{m.value:.4f}</td>"
            f"<td>{m.description}</td></tr>"
            for m in report.milestones[-10:]
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>North Star Tracker</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
.progress {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
</style></head><body>
<h1>North Star Progress Tracker</h1>
<div class="summary">
<p class="progress">{report.overall_progress:.0%} targets met</p>
<p>Return: {c.annual_return:.2%} | Sharpe: {c.sharpe:.2f} | Max DD: {c.max_drawdown:.2%}</p>
</div>
<h2>Gap Analysis</h2>
<table><tr><th>Metric</th><th>Current</th><th>Target</th><th>Gap</th><th>% Achieved</th><th>Status</th></tr>
{''.join(gap_rows)}</table>
<h2>Trajectory Projections</h2>
<table><tr><th>Metric</th><th>Current</th><th>Target</th><th>Rate</th><th>ETA</th></tr>
{''.join(proj_rows)}</table>
{'<h2>Experiments</h2><table><tr><th>ID</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>Trades</th></tr>' + ''.join(exp_rows) + '</table>' if exp_rows else ''}
{'<h2>Milestones</h2><table><tr><th>Date</th><th>Metric</th><th>Value</th><th>Description</th></tr>' + ''.join(ms_rows) + '</table>' if ms_rows else ''}
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
