"""
North Star gap analyzer — measures where we stand vs targets.

Targets: 100% annual return, 12% max drawdown, 6.0 Sharpe, billions AUM.
Per target: current best, gap, bottlenecks, recommendations, score 0-100.
Monte Carlo confidence intervals, capacity analysis, strategy combination.

Usage::

    from compass.north_star_gap import NorthStarGapAnalyzer
    analyzer = NorthStarGapAnalyzer(experiment_results)
    results = analyzer.analyze()
    analyzer.generate_report()
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "north_star_gap.html"

TARGETS = {
    "annual_return": 1.0,    # 100%
    "max_drawdown": 0.12,    # 12%
    "sharpe": 6.0,
    "capacity_aum": 1e9,     # $1B
}


@dataclass
class ExperimentResult:
    name: str
    daily_returns: np.ndarray
    annual_return: float
    max_drawdown: float
    sharpe: float
    capacity_est: float      # estimated max AUM


@dataclass
class TargetGap:
    metric: str
    target: float
    current_best: float
    gap: float               # target - current (positive = below target)
    gap_pct: float           # gap as % of target
    score: float             # 0-100
    status: str              # "achieved", "close", "moderate", "far"
    best_experiment: str
    bottlenecks: List[str]
    recommendations: List[str]


@dataclass
class StrategyCombo:
    experiments: List[str]
    combined_return: float
    combined_dd: float
    combined_sharpe: float
    gap_to_north_star: float
    is_achievable: bool


@dataclass
class CapacityPoint:
    aum: float
    estimated_return: float
    estimated_sharpe: float
    estimated_dd: float
    feasible: bool


@dataclass
class MCProjection:
    metric: str
    current: float
    target: float
    p50_months_to_target: float
    p10_months: float
    p90_months: float
    probability_in_12m: float


@dataclass
class GapSummary:
    overall_score: float
    n_achieved: int
    n_close: int
    n_far: int
    biggest_gap_metric: str
    priority_actions: List[str]


class NorthStarGapAnalyzer:
    """Gap analysis vs North Star targets."""

    def __init__(
        self,
        experiments: List[ExperimentResult],
        targets: Optional[Dict[str, float]] = None,
        mc_sims: int = 1000,
        seed: int = 42,
    ) -> None:
        self.experiments = list(experiments)
        self.targets = targets or dict(TARGETS)
        self.mc_sims = mc_sims
        self.rng = np.random.RandomState(seed)

        self.gaps: List[TargetGap] = []
        self.combos: List[StrategyCombo] = []
        self.capacity: List[CapacityPoint] = []
        self.projections: List[MCProjection] = []
        self.summary: Optional[GapSummary] = None

    @classmethod
    def from_returns(
        cls, returns_dict: Dict[str, np.ndarray], **kwargs: Any,
    ) -> "NorthStarGapAnalyzer":
        experiments = []
        for name, rets in returns_dict.items():
            rets = np.asarray(rets, dtype=float)
            ann = float((1 + np.mean(rets)) ** 252 - 1) if len(rets) > 0 else 0
            eq = np.cumprod(1 + rets)
            pk = np.maximum.accumulate(eq)
            dd = float(np.min((eq - pk) / np.where(pk > 0, pk, 1))) if len(eq) > 0 else 0
            sh = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0
            cap = 1e8 / max(abs(dd), 0.01)
            experiments.append(ExperimentResult(name, rets, ann, abs(dd), sh, cap))
        return cls(experiments, **kwargs)

    def analyze(self) -> Dict[str, Any]:
        self.gaps = self._compute_gaps()
        self.combos = self._strategy_combinations()
        self.capacity = self._capacity_analysis()
        self.projections = self._mc_projections()
        self.summary = self._summarize()
        return {
            "gaps": self.gaps,
            "combos": self.combos,
            "capacity": self.capacity,
            "projections": self.projections,
            "summary": self.summary,
        }

    def _compute_gaps(self) -> List[TargetGap]:
        gaps = []
        # Annual return
        best_ret = max(self.experiments, key=lambda e: e.annual_return)
        target = self.targets["annual_return"]
        gap = target - best_ret.annual_return
        score = min(100, max(0, best_ret.annual_return / target * 100))
        gaps.append(TargetGap(
            "annual_return", target, best_ret.annual_return, gap,
            gap / target if target else 0, score,
            self._status(score), best_ret.name,
            self._bottlenecks_return(), self._recs_return(),
        ))

        # Max drawdown (lower is better)
        best_dd = min(self.experiments, key=lambda e: e.max_drawdown)
        target_dd = self.targets["max_drawdown"]
        dd_gap = best_dd.max_drawdown - target_dd
        dd_score = min(100, max(0, (1 - best_dd.max_drawdown / target_dd) * 100)) if best_dd.max_drawdown <= target_dd else max(0, 100 - dd_gap / target_dd * 100)
        gaps.append(TargetGap(
            "max_drawdown", target_dd, best_dd.max_drawdown, dd_gap,
            dd_gap / target_dd if target_dd else 0, dd_score,
            self._status(dd_score), best_dd.name,
            self._bottlenecks_dd(), self._recs_dd(),
        ))

        # Sharpe
        best_sh = max(self.experiments, key=lambda e: e.sharpe)
        target_sh = self.targets["sharpe"]
        sh_gap = target_sh - best_sh.sharpe
        sh_score = min(100, max(0, best_sh.sharpe / target_sh * 100))
        gaps.append(TargetGap(
            "sharpe", target_sh, best_sh.sharpe, sh_gap,
            sh_gap / target_sh if target_sh else 0, sh_score,
            self._status(sh_score), best_sh.name,
            self._bottlenecks_sharpe(), self._recs_sharpe(),
        ))

        # Capacity
        best_cap = max(self.experiments, key=lambda e: e.capacity_est)
        target_cap = self.targets["capacity_aum"]
        cap_gap = target_cap - best_cap.capacity_est
        cap_score = min(100, max(0, best_cap.capacity_est / target_cap * 100))
        gaps.append(TargetGap(
            "capacity_aum", target_cap, best_cap.capacity_est, cap_gap,
            cap_gap / target_cap if target_cap else 0, cap_score,
            self._status(cap_score), best_cap.name,
            ["Limited number of tradable instruments", "Market impact at scale"],
            ["Add more uncorrelated strategies", "Diversify across asset classes"],
        ))
        return gaps

    @staticmethod
    def _status(score: float) -> str:
        if score >= 90: return "achieved"
        if score >= 60: return "close"
        if score >= 30: return "moderate"
        return "far"

    def _bottlenecks_return(self) -> List[str]:
        return ["Signal alpha decay", "Execution slippage", "Position sizing constraints"]
    def _recs_return(self) -> List[str]:
        return ["Improve signal_model IC by 20%", "Add new alpha sources", "Optimize entry timing"]
    def _bottlenecks_dd(self) -> List[str]:
        return ["Tail event exposure", "Correlation spikes in stress", "Stop loss gaps"]
    def _recs_dd(self) -> List[str]:
        return ["Tighten adaptive stops", "Add crisis hedging", "Reduce position concentration"]
    def _bottlenecks_sharpe(self) -> List[str]:
        return ["Return volatility", "Signal noise", "Transaction costs"]
    def _recs_sharpe(self) -> List[str]:
        return ["Ensemble more signals", "Reduce turnover", "Improve regime detection"]

    def _strategy_combinations(self) -> List[StrategyCombo]:
        combos = []
        n = len(self.experiments)
        if n < 2:
            return combos
        # Try all pairs + full portfolio
        for i in range(n):
            for j in range(i + 1, n):
                combo = self._eval_combo([self.experiments[i], self.experiments[j]])
                combos.append(combo)
        if n >= 3:
            combos.append(self._eval_combo(self.experiments))
        return sorted(combos, key=lambda c: c.combined_sharpe, reverse=True)[:10]

    def _eval_combo(self, exps: List[ExperimentResult]) -> StrategyCombo:
        # Equal-weight combination
        max_len = min(len(e.daily_returns) for e in exps)
        if max_len < 10:
            return StrategyCombo([e.name for e in exps], 0, 0, 0, 1, False)
        combined = np.mean([e.daily_returns[:max_len] for e in exps], axis=0)
        ann = float((1 + np.mean(combined)) ** 252 - 1)
        eq = np.cumprod(1 + combined)
        pk = np.maximum.accumulate(eq)
        dd = float(abs(np.min((eq - pk) / np.where(pk > 0, pk, 1))))
        sh = float(np.mean(combined) / np.std(combined) * np.sqrt(252)) if np.std(combined) > 0 else 0
        gap = max(self.targets["annual_return"] - ann, 0) + max(dd - self.targets["max_drawdown"], 0)
        feasible = ann >= self.targets["annual_return"] * 0.5 and dd <= self.targets["max_drawdown"] * 2
        return StrategyCombo([e.name for e in exps], ann, dd, sh, gap, feasible)

    def _capacity_analysis(self) -> List[CapacityPoint]:
        points = []
        best = max(self.experiments, key=lambda e: e.sharpe)
        for aum in [1e6, 1e7, 1e8, 1e9, 1e10]:
            # Impact: returns degrade with sqrt(AUM)
            impact_drag = 0.001 * math.sqrt(aum / 1e6)
            adj_ret = best.annual_return * max(0, 1 - impact_drag)
            adj_sh = best.sharpe * max(0, 1 - impact_drag)
            adj_dd = best.max_drawdown * (1 + impact_drag * 0.5)
            feasible = adj_sh > 1.0 and adj_dd < 0.30
            points.append(CapacityPoint(aum, adj_ret, adj_sh, adj_dd, feasible))
        return points

    def _mc_projections(self) -> List[MCProjection]:
        projections = []
        for gap in self.gaps:
            if gap.status == "achieved":
                projections.append(MCProjection(
                    gap.metric, gap.current_best, gap.target, 0, 0, 0, 1.0,
                ))
                continue
            # Simple MC: project improvement rate
            improvement_rate = abs(gap.current_best) * 0.05 / 12  # 5%/yr improvement
            if improvement_rate < 1e-10:
                improvement_rate = abs(gap.target) * 0.01
            months = []
            for _ in range(self.mc_sims):
                val = gap.current_best
                for m in range(1, 121):
                    noise = self.rng.normal(0, improvement_rate * 0.5)
                    val += improvement_rate + noise
                    if gap.metric == "max_drawdown":
                        if val <= gap.target:
                            months.append(m); break
                    else:
                        if val >= gap.target:
                            months.append(m); break
                else:
                    months.append(120)
            months_arr = np.array(months)
            prob_12m = float((months_arr <= 12).mean())
            projections.append(MCProjection(
                gap.metric, gap.current_best, gap.target,
                float(np.median(months_arr)), float(np.percentile(months_arr, 10)),
                float(np.percentile(months_arr, 90)), prob_12m,
            ))
        return projections

    def _summarize(self) -> GapSummary:
        scores = [g.score for g in self.gaps]
        overall = float(np.mean(scores)) if scores else 0
        achieved = sum(1 for g in self.gaps if g.status == "achieved")
        close = sum(1 for g in self.gaps if g.status == "close")
        far = sum(1 for g in self.gaps if g.status in ("moderate", "far"))
        worst = min(self.gaps, key=lambda g: g.score) if self.gaps else None
        actions = []
        for g in sorted(self.gaps, key=lambda x: x.score):
            actions.extend(g.recommendations[:2])
        return GapSummary(overall, achieved, close, far,
                          worst.metric if worst else "", actions[:6])

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.summary is None:
            self.analyze()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return str(out.resolve())

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        return {"scorecard": self._chart_scorecard(), "capacity": self._chart_capacity(), "waterfall": self._chart_waterfall()}

    def _chart_scorecard(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.gaps: return ""
        names = [g.metric.replace("_", " ").title() for g in self.gaps]
        scores = [g.score for g in self.gaps]
        colors = ["#16a34a" if s >= 80 else "#f59e0b" if s >= 50 else "#dc2626" for s in scores]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(names, scores, color=colors, alpha=0.85)
        ax.set_xlim(0, 100); ax.set_xlabel("Score (0-100)")
        ax.set_title("North Star Scorecard", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_capacity(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.capacity: return ""
        aums = [c.aum for c in self.capacity]
        sharpes = [c.estimated_sharpe for c in self.capacity]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.semilogx(aums, sharpes, "o-", color="#3b82f6", lw=1.5)
        ax.axhline(1.0, color="#dc2626", ls="--", lw=0.8, label="Min viable Sharpe")
        ax.set_xlabel("AUM ($)"); ax.set_ylabel("Estimated Sharpe")
        ax.set_title("Capacity Curve", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_waterfall(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.gaps: return ""
        names = [g.metric.replace("_", " ").title() for g in self.gaps]
        gap_pcts = [g.gap_pct * 100 for g in self.gaps]
        colors = ["#dc2626" if g > 0 else "#16a34a" for g in gap_pcts]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(names, gap_pcts, color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Gap (% of target)"); ax.set_title("Gap to North Star", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        s = self.summary or GapSummary(0, 0, 0, 0, "", [])

        gap_rows = ""
        for g in self.gaps:
            cls = "good" if g.status == "achieved" else "bad" if g.status == "far" else ""
            fmt = f"${g.target:,.0f}" if g.metric == "capacity_aum" else f"{g.target:.0%}" if g.metric in ("annual_return", "max_drawdown") else f"{g.target:.1f}"
            cur_fmt = f"${g.current_best:,.0f}" if g.metric == "capacity_aum" else f"{g.current_best:.1%}" if g.metric in ("annual_return", "max_drawdown") else f"{g.current_best:.2f}"
            gap_rows += (f'<tr><td>{g.metric.replace("_"," ").title()}</td><td>{fmt}</td><td>{cur_fmt}</td>'
                        f'<td class="{cls}">{g.score:.0f}</td><td class="{cls}">{g.status}</td>'
                        f'<td>{g.best_experiment}</td></tr>\n')

        combo_rows = ""
        for c in self.combos[:5]:
            cls = "good" if c.is_achievable else ""
            combo_rows += (f'<tr><td>{" + ".join(c.experiments)}</td><td>{c.combined_return:+.1%}</td>'
                          f'<td>{c.combined_dd:.1%}</td><td>{c.combined_sharpe:.2f}</td>'
                          f'<td class="{cls}">{"Yes" if c.is_achievable else "No"}</td></tr>\n')
        if not combo_rows:
            combo_rows = '<tr><td colspan="5" style="text-align:center;color:#64748b">Need 2+ experiments</td></tr>'

        proj_rows = ""
        for p in self.projections:
            proj_rows += (f'<tr><td>{p.metric.replace("_"," ").title()}</td>'
                         f'<td>{p.p50_months_to_target:.0f}m</td><td>{p.p10_months:.0f}m</td>'
                         f'<td>{p.p90_months:.0f}m</td><td>{p.probability_in_12m:.0%}</td></tr>\n')

        action_list = "".join(f"<li>{a}</li>" for a in s.priority_actions) or "<li>No actions</li>"

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>North Star Gap Analysis</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }} h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }}
  .kpi-row {{ display:flex; gap:1.2em; flex-wrap:wrap; margin:1.5em 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em 1.5em; min-width:120px; flex:1; text-align:center; }}
  .kpi .value {{ font-size:1.5em; font-weight:700; }} .kpi .label {{ font-size:0.75em; color:#64748b; margin-top:0.2em; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:0.88em; }}
  th {{ background:#f1f5f9; padding:8px 10px; text-align:left; border-bottom:2px solid #cbd5e1; font-weight:600; }}
  td {{ padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:right; }} td:first-child {{ text-align:left; }}
  ul {{ margin:0.5em 0; padding-left:1.5em; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em; margin:1.5em 0; text-align:center; }}
  .chart img {{ max-width:100%; height:auto; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.8em; color:#94a3b8; }}
</style></head><body>
<h1>North Star Gap Analysis</h1>
<div class="meta">{len(self.experiments)} experiments &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value {"good" if s.overall_score >= 60 else "bad"}">{s.overall_score:.0f}</div><div class="label">Overall Score</div></div>
  <div class="kpi"><div class="value good">{s.n_achieved}</div><div class="label">Achieved</div></div>
  <div class="kpi"><div class="value">{s.n_close}</div><div class="label">Close</div></div>
  <div class="kpi"><div class="value bad">{s.n_far}</div><div class="label">Far</div></div>
</div>
<h2>1. North Star Scorecard</h2>{_img("scorecard")}
<table><thead><tr><th>Metric</th><th>Target</th><th>Current Best</th><th>Score</th><th>Status</th><th>Best Exp</th></tr></thead><tbody>{gap_rows}</tbody></table>
<h2>2. Gap Waterfall</h2>{_img("waterfall")}
<h2>3. Strategy Combinations</h2>
<table><thead><tr><th>Combination</th><th>Return</th><th>DD</th><th>Sharpe</th><th>Feasible</th></tr></thead><tbody>{combo_rows}</tbody></table>
<h2>4. Capacity Curve</h2>{_img("capacity")}
<h2>5. Monte Carlo Projections</h2>
<table><thead><tr><th>Metric</th><th>Median Months</th><th>P10</th><th>P90</th><th>P(12m)</th></tr></thead><tbody>{proj_rows}</tbody></table>
<h2>6. Priority Actions</h2><ul>{action_list}</ul>
<footer>Generated by <code>compass/north_star_gap.py</code></footer>
</body></html>"""
        return html
