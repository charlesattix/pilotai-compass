"""
North Star target optimizer — identifies gaps and improvement opportunities.

Given current metrics and MASTERPLAN targets (55% annual, Sharpe 6.0,
DD ≤ 30%), identifies bottlenecks, runs parameter sweeps, evaluates
strategy combinations, and ranks improvement opportunities.

All methods work on pre-loaded data — no network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
NORTH_STAR = {"annual_return": 0.55, "sharpe": 6.0, "max_drawdown": 0.30}


@dataclass
class PerformanceGap:
    metric: str
    current: float
    target: float
    gap: float            # target - current (positive = shortfall)
    pct_achieved: float   # current / target
    is_met: bool


@dataclass
class SweepResult:
    param_name: str
    param_value: Any
    sharpe: float
    annual_return: float
    max_drawdown: float
    improvement_vs_baseline: float


@dataclass
class CombinationResult:
    strategies: List[str]
    weights: Dict[str, float]
    sharpe: float
    annual_return: float
    max_drawdown: float
    correlation: float


@dataclass
class ImprovementOpportunity:
    description: str
    category: str         # "parameter" | "combination" | "signal"
    estimated_impact: float
    feasibility: float    # 0-1 (higher = easier)
    priority_score: float


@dataclass
class TargetAnalysis:
    gaps: List[PerformanceGap]
    sweep_results: List[SweepResult] = field(default_factory=list)
    combinations: List[CombinationResult] = field(default_factory=list)
    opportunities: List[ImprovementOpportunity] = field(default_factory=list)
    overall_score: float = 0.0


class TargetOptimizer:
    """North Star target optimizer.

    Args:
        targets: Override North Star targets.
    """

    def __init__(self, targets: Optional[Dict[str, float]] = None) -> None:
        self.targets = targets or dict(NORTH_STAR)

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def compute_gaps(
        self, current: Dict[str, float],
    ) -> List[PerformanceGap]:
        """Identify gaps between current metrics and targets."""
        gaps: List[PerformanceGap] = []
        for metric, target in self.targets.items():
            cur = current.get(metric, 0.0)
            if metric == "max_drawdown":
                gap = cur - target  # positive = worse than target
                is_met = cur <= target
                pct = (target / cur) if cur > 0 else 1.0
            else:
                gap = target - cur
                is_met = cur >= target
                pct = cur / target if target > 0 else 0.0
            gaps.append(PerformanceGap(
                metric=metric, current=cur, target=target,
                gap=gap, pct_achieved=pct, is_met=is_met,
            ))
        return gaps

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------

    @staticmethod
    def parameter_sweep(
        evaluate_fn: Callable[[Any], Dict[str, float]],
        param_name: str,
        param_values: List[Any],
        baseline: Optional[Dict[str, float]] = None,
    ) -> List[SweepResult]:
        """Run a parameter sweep and collect metrics."""
        baseline_sharpe = baseline.get("sharpe", 0.0) if baseline else 0.0
        results: List[SweepResult] = []
        for val in param_values:
            try:
                metrics = evaluate_fn(val)
            except Exception:
                continue
            results.append(SweepResult(
                param_name=param_name, param_value=val,
                sharpe=metrics.get("sharpe", 0.0),
                annual_return=metrics.get("annual_return", 0.0),
                max_drawdown=metrics.get("max_drawdown", 0.0),
                improvement_vs_baseline=metrics.get("sharpe", 0.0) - baseline_sharpe,
            ))
        results.sort(key=lambda r: r.sharpe, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Strategy combination
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_combinations(
        strategy_returns: Dict[str, pd.Series],
        max_strategies: int = 4,
    ) -> List[CombinationResult]:
        """Test equal-weight combinations of strategies."""
        from itertools import combinations as itertools_combos
        names = list(strategy_returns.keys())
        results: List[CombinationResult] = []

        for n in range(2, min(len(names), max_strategies) + 1):
            for combo in itertools_combos(names, n):
                rets = pd.DataFrame({k: strategy_returns[k] for k in combo}).dropna()
                if len(rets) < 20:
                    continue
                w = {k: 1.0 / n for k in combo}
                port = rets.mean(axis=1)
                mu = float(port.mean()) * TRADING_DAYS
                std = float(port.std()) * np.sqrt(TRADING_DAYS)
                sharpe = mu / std if std > 1e-12 else 0.0
                eq = (1 + port).cumprod()
                dd = float((1 - eq / eq.expanding().max()).max())
                annual = float((1 + port.sum()) ** (TRADING_DAYS / len(port)) - 1) if len(port) > 0 else 0.0

                corr_matrix = rets.corr()
                n_assets = len(corr_matrix)
                avg_corr = 0.0
                if n_assets > 1:
                    upper = []
                    for i in range(n_assets):
                        for j in range(i + 1, n_assets):
                            upper.append(corr_matrix.iloc[i, j])
                    avg_corr = float(np.mean(upper))

                results.append(CombinationResult(
                    strategies=list(combo), weights=w,
                    sharpe=sharpe, annual_return=annual,
                    max_drawdown=dd, correlation=avg_corr,
                ))

        results.sort(key=lambda r: r.sharpe, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Required improvement per metric
    # ------------------------------------------------------------------

    @staticmethod
    def required_improvements(gaps: List[PerformanceGap]) -> Dict[str, float]:
        """Compute required improvement for each unmet target."""
        return {
            g.metric: g.gap
            for g in gaps if not g.is_met
        }

    # ------------------------------------------------------------------
    # Opportunity ranking
    # ------------------------------------------------------------------

    @staticmethod
    def rank_opportunities(
        sweep_results: List[SweepResult],
        combinations: List[CombinationResult],
    ) -> List[ImprovementOpportunity]:
        """Rank improvement opportunities by feasibility × impact."""
        opps: List[ImprovementOpportunity] = []

        # From sweeps
        for r in sweep_results[:5]:
            if r.improvement_vs_baseline > 0:
                impact = min(r.improvement_vs_baseline / 2.0, 1.0)
                feas = 0.8  # parameter changes are easy
                opps.append(ImprovementOpportunity(
                    description=f"Set {r.param_name}={r.param_value} (Sharpe +{r.improvement_vs_baseline:.2f})",
                    category="parameter", estimated_impact=impact,
                    feasibility=feas, priority_score=impact * feas,
                ))

        # From combinations
        for c in combinations[:5]:
            if c.sharpe > 2.0:
                impact = min(c.sharpe / 6.0, 1.0)
                feas = 0.5 if c.correlation < 0.3 else 0.3
                opps.append(ImprovementOpportunity(
                    description=f"Combine {'+'.join(c.strategies)} (Sharpe {c.sharpe:.2f}, corr {c.correlation:.2f})",
                    category="combination", estimated_impact=impact,
                    feasibility=feas, priority_score=impact * feas,
                ))

        opps.sort(key=lambda o: o.priority_score, reverse=True)
        return opps

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        current_metrics: Dict[str, float],
        strategy_returns: Optional[Dict[str, pd.Series]] = None,
        sweep_fn: Optional[Callable] = None,
        sweep_param: str = "",
        sweep_values: Optional[List] = None,
    ) -> TargetAnalysis:
        gaps = self.compute_gaps(current_metrics)
        sweeps: List[SweepResult] = []
        if sweep_fn and sweep_values:
            sweeps = self.parameter_sweep(sweep_fn, sweep_param, sweep_values, current_metrics)

        combos: List[CombinationResult] = []
        if strategy_returns:
            combos = self.evaluate_combinations(strategy_returns)

        opps = self.rank_opportunities(sweeps, combos)
        score = sum(1 for g in gaps if g.is_met) / len(gaps) if gaps else 0.0

        return TargetAnalysis(
            gaps=gaps, sweep_results=sweeps,
            combinations=combos, opportunities=opps,
            overall_score=score,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, analysis: TargetAnalysis,
        output_path: str = "reports/target_optimizer.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        gap_rows = [
            f"<tr><td>{g.metric}</td><td>{g.current:.2%}</td>"
            f"<td>{g.target:.2%}</td><td>{g.gap:+.2%}</td>"
            f"<td>{g.pct_achieved:.0%}</td>"
            f"<td>{'MET' if g.is_met else 'GAP'}</td></tr>"
            for g in analysis.gaps
        ]
        opp_rows = [
            f"<tr><td style='text-align:left'>{o.description}</td>"
            f"<td>{o.category}</td><td>{o.estimated_impact:.2f}</td>"
            f"<td>{o.feasibility:.2f}</td><td>{o.priority_score:.2f}</td></tr>"
            for o in analysis.opportunities[:10]
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Target Optimizer</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
</style></head><body>
<h1>North Star Target Optimizer</h1>
<div class="summary">
<p><strong>Score:</strong> {analysis.overall_score:.0%} targets met</p>
</div>
<h2>Gap Analysis</h2>
<table><tr><th>Metric</th><th>Current</th><th>Target</th><th>Gap</th><th>% Achieved</th><th>Status</th></tr>
{''.join(gap_rows)}</table>
<h2>Improvement Opportunities</h2>
<table><tr><th style='text-align:left'>Description</th><th>Category</th><th>Impact</th>
<th>Feasibility</th><th>Priority</th></tr>
{''.join(opp_rows)}</table>
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
