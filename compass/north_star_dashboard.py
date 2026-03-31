from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Traffic-light enum (plain strings kept simple on purpose)
# ---------------------------------------------------------------------------
GREEN = "GREEN"
YELLOW = "YELLOW"
RED = "RED"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class NorthStarTargets:
    """Configurable North Star targets for the strategy."""

    annual_return_target: float = 0.55
    sharpe_target: float = 6.0
    max_dd_target: float = 0.30
    all_years_profitable: bool = True
    robustness_target: float = 0.70


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ExperimentStatus:
    """Status card for a single experiment."""

    name: str
    sharpe: float
    annual_return: float
    max_dd: float
    win_rate: float
    profit_factor: float
    yearly_returns: list[float]
    robustness_score: float
    all_years_profitable: bool
    traffic_lights: dict[str, str] = field(default_factory=dict)


@dataclass
class GapItem:
    """One row of the gap-analysis table."""

    metric: str
    target: float | bool
    current: float | bool
    gap: float | None
    message: str
    status: str


@dataclass
class PortfolioMetrics:
    """Blended portfolio-level metrics across experiments."""

    blended_sharpe: float
    blended_return: float
    worst_max_dd: float
    avg_win_rate: float
    avg_profit_factor: float
    avg_robustness: float
    all_years_profitable: bool


@dataclass
class NorthStarResult:
    """Complete dashboard result."""

    experiment_statuses: list[ExperimentStatus]
    portfolio_metrics: PortfolioMetrics
    gap_analysis: list[GapItem]
    overall_status: str  # GREEN / YELLOW / RED


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
class NorthStarDashboard:
    """Investor-quality North Star targets dashboard."""

    def __init__(self, targets: NorthStarTargets | None = None) -> None:
        self.targets = targets or NorthStarTargets()

    # ------------------------------------------------------------------
    # Traffic-light logic
    # ------------------------------------------------------------------
    @staticmethod
    def _traffic_light(value: float, target: float, higher_is_better: bool = True) -> str:
        """Return GREEN / YELLOW / RED for *value* vs *target*.

        For ``higher_is_better=True`` (default):
          * GREEN  if value >= target
          * YELLOW if value >= 80 % of target
          * RED    otherwise

        For ``higher_is_better=False`` (e.g. max drawdown where lower is
        better) the comparison is inverted: GREEN when value <= target, etc.
        """
        if higher_is_better:
            if value >= target:
                return GREEN
            if value >= target * 0.80:
                return YELLOW
            return RED
        else:
            # Lower is better (drawdown)
            if value <= target:
                return GREEN
            if value <= target * 1.20:
                return YELLOW
            return RED

    def _evaluate_traffic_lights(self, exp: dict[str, Any]) -> dict[str, str]:
        t = self.targets
        lights: dict[str, str] = {
            "annual_return": self._traffic_light(exp["annual_return"], t.annual_return_target),
            "sharpe": self._traffic_light(exp["sharpe"], t.sharpe_target),
            "max_dd": self._traffic_light(exp["max_dd"], t.max_dd_target, higher_is_better=False),
            "robustness": self._traffic_light(exp["robustness_score"], t.robustness_target),
        }
        yearly = exp.get("yearly_returns", [])
        all_profitable = all(r > 0 for r in yearly) if yearly else False
        if t.all_years_profitable:
            lights["all_years_profitable"] = GREEN if all_profitable else RED
        else:
            lights["all_years_profitable"] = GREEN
        return lights

    # ------------------------------------------------------------------
    # Per-experiment status card
    # ------------------------------------------------------------------
    def _build_experiment_status(self, name: str, exp: dict[str, Any]) -> ExperimentStatus:
        yearly = exp.get("yearly_returns", [])
        return ExperimentStatus(
            name=name,
            sharpe=float(exp["sharpe"]),
            annual_return=float(exp["annual_return"]),
            max_dd=float(exp["max_dd"]),
            win_rate=float(exp.get("win_rate", 0.0)),
            profit_factor=float(exp.get("profit_factor", 0.0)),
            yearly_returns=list(yearly),
            robustness_score=float(exp.get("robustness_score", 0.0)),
            all_years_profitable=all(r > 0 for r in yearly) if yearly else False,
            traffic_lights=self._evaluate_traffic_lights(exp),
        )

    # ------------------------------------------------------------------
    # Portfolio (blended) metrics
    # ------------------------------------------------------------------
    def _compute_portfolio_metrics(
        self, experiments: dict[str, dict[str, Any]]
    ) -> PortfolioMetrics:
        if not experiments:
            return PortfolioMetrics(
                blended_sharpe=0.0,
                blended_return=0.0,
                worst_max_dd=0.0,
                avg_win_rate=0.0,
                avg_profit_factor=0.0,
                avg_robustness=0.0,
                all_years_profitable=False,
            )
        sharpes = np.array([e["sharpe"] for e in experiments.values()])
        returns_ = np.array([e["annual_return"] for e in experiments.values()])
        dds = np.array([e["max_dd"] for e in experiments.values()])
        win_rates = np.array([e.get("win_rate", 0.0) for e in experiments.values()])
        pfs = np.array([e.get("profit_factor", 0.0) for e in experiments.values()])
        robs = np.array([e.get("robustness_score", 0.0) for e in experiments.values()])

        all_yearly: list[list[float]] = [e.get("yearly_returns", []) for e in experiments.values()]
        all_prof = all(
            all(r > 0 for r in yr) if yr else False for yr in all_yearly
        )

        return PortfolioMetrics(
            blended_sharpe=float(np.mean(sharpes)),
            blended_return=float(np.mean(returns_)),
            worst_max_dd=float(np.max(dds)),
            avg_win_rate=float(np.mean(win_rates)),
            avg_profit_factor=float(np.mean(pfs)),
            avg_robustness=float(np.mean(robs)),
            all_years_profitable=all_prof,
        )

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------
    def _gap_analysis(self, portfolio: PortfolioMetrics) -> list[GapItem]:
        t = self.targets
        items: list[GapItem] = []

        def _higher(metric_name: str, current: float, target: float) -> GapItem:
            gap = target - current
            light = self._traffic_light(current, target)
            if gap <= 0:
                msg = f"Target met ({current:.2%} >= {target:.2%})"
            else:
                msg = f"Need +{gap:.2%} to hit {target:.2%}"
            return GapItem(
                metric=metric_name,
                target=target,
                current=current,
                gap=round(gap, 6) if gap > 0 else 0.0,
                message=msg,
                status=light,
            )

        def _higher_abs(metric_name: str, current: float, target: float) -> GapItem:
            gap = target - current
            light = self._traffic_light(current, target)
            if gap <= 0:
                msg = f"Target met ({current:.2f} >= {target:.2f})"
            else:
                msg = f"Need +{gap:.2f} to hit {target:.2f}"
            return GapItem(
                metric=metric_name,
                target=target,
                current=current,
                gap=round(gap, 6) if gap > 0 else 0.0,
                message=msg,
                status=light,
            )

        def _lower(metric_name: str, current: float, target: float) -> GapItem:
            gap = current - target
            light = self._traffic_light(current, target, higher_is_better=False)
            if gap <= 0:
                msg = f"Target met ({current:.2%} <= {target:.2%})"
            else:
                msg = f"Need to reduce by {gap:.2%} to hit {target:.2%}"
            return GapItem(
                metric=metric_name,
                target=target,
                current=current,
                gap=round(gap, 6) if gap > 0 else 0.0,
                message=msg,
                status=light,
            )

        items.append(_higher("Annual Return", portfolio.blended_return, t.annual_return_target))
        items.append(_higher_abs("Sharpe Ratio", portfolio.blended_sharpe, t.sharpe_target))
        items.append(_lower("Max Drawdown", portfolio.worst_max_dd, t.max_dd_target))
        items.append(_higher("Robustness", portfolio.avg_robustness, t.robustness_target))

        # All years profitable
        ayp_light = GREEN if portfolio.all_years_profitable else RED
        ayp_msg = "All years profitable" if portfolio.all_years_profitable else "Not all years profitable"
        items.append(
            GapItem(
                metric="All Years Profitable",
                target=t.all_years_profitable,
                current=portfolio.all_years_profitable,
                gap=None,
                message=ayp_msg,
                status=ayp_light if t.all_years_profitable else GREEN,
            )
        )
        return items

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------
    @staticmethod
    def _overall_status(gap_items: list[GapItem]) -> str:
        statuses = [g.status for g in gap_items]
        if all(s == GREEN for s in statuses):
            return GREEN
        if any(s == RED for s in statuses):
            return RED
        return YELLOW

    # ------------------------------------------------------------------
    # Evaluate (main entry point without HTML)
    # ------------------------------------------------------------------
    def evaluate(self, experiments: dict[str, dict[str, Any]]) -> NorthStarResult:
        statuses = [
            self._build_experiment_status(name, data)
            for name, data in experiments.items()
        ]
        portfolio = self._compute_portfolio_metrics(experiments)
        gaps = self._gap_analysis(portfolio)
        overall = self._overall_status(gaps)
        return NorthStarResult(
            experiment_statuses=statuses,
            portfolio_metrics=portfolio,
            gap_analysis=gaps,
            overall_status=overall,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------
    @staticmethod
    def _light_color(status: str) -> str:
        return {"GREEN": "#00e676", "YELLOW": "#ffd600", "RED": "#ff1744"}.get(
            status, "#999"
        )

    @staticmethod
    def _light_emoji(status: str) -> str:
        return {"GREEN": "&#9679;", "YELLOW": "&#9679;", "RED": "&#9679;"}.get(
            status, "&#9679;"
        )

    def _render_summary_card(self, result: NorthStarResult) -> str:
        color = self._light_color(result.overall_status)
        pm = result.portfolio_metrics
        return f"""
        <div class="card summary-card" style="border-left: 6px solid {color};">
            <h2>
                <span style="color:{color}; font-size:1.4em;">{self._light_emoji(result.overall_status)}</span>
                Overall Status: {result.overall_status}
            </h2>
            <div class="metrics-grid">
                <div class="metric"><span class="label">Blended Sharpe</span><span class="value">{pm.blended_sharpe:.2f}</span></div>
                <div class="metric"><span class="label">Blended Return</span><span class="value">{pm.blended_return:.2%}</span></div>
                <div class="metric"><span class="label">Worst Max DD</span><span class="value">{pm.worst_max_dd:.2%}</span></div>
                <div class="metric"><span class="label">Avg Win Rate</span><span class="value">{pm.avg_win_rate:.2%}</span></div>
                <div class="metric"><span class="label">Avg Profit Factor</span><span class="value">{pm.avg_profit_factor:.2f}</span></div>
                <div class="metric"><span class="label">Avg Robustness</span><span class="value">{pm.avg_robustness:.2%}</span></div>
                <div class="metric"><span class="label">All Years Profitable</span><span class="value">{"Yes" if pm.all_years_profitable else "No"}</span></div>
            </div>
        </div>
        """

    def _render_experiment_card(self, es: ExperimentStatus) -> str:
        rows = ""
        metrics = [
            ("Sharpe", f"{es.sharpe:.2f}", es.traffic_lights.get("sharpe", "")),
            ("Annual Return", f"{es.annual_return:.2%}", es.traffic_lights.get("annual_return", "")),
            ("Max Drawdown", f"{es.max_dd:.2%}", es.traffic_lights.get("max_dd", "")),
            ("Win Rate", f"{es.win_rate:.2%}", ""),
            ("Profit Factor", f"{es.profit_factor:.2f}", ""),
            ("Robustness", f"{es.robustness_score:.2%}", es.traffic_lights.get("robustness", "")),
            ("All Years Profitable", "Yes" if es.all_years_profitable else "No",
             es.traffic_lights.get("all_years_profitable", "")),
        ]
        for label, value, light in metrics:
            color = self._light_color(light) if light else "#aaa"
            dot = f'<span style="color:{color};">{self._light_emoji(light)}</span>' if light else ""
            rows += f"<tr><td>{dot} {html.escape(label)}</td><td>{html.escape(value)}</td></tr>\n"
        safe_name = html.escape(es.name)
        return f"""
        <div class="card experiment-card">
            <h3>{safe_name}</h3>
            <table>{rows}</table>
        </div>
        """

    def _render_gap_table(self, gaps: list[GapItem]) -> str:
        rows = ""
        for g in gaps:
            color = self._light_color(g.status)
            dot = f'<span style="color:{color};">{self._light_emoji(g.status)}</span>'
            target_str = f"{g.target}" if isinstance(g.target, bool) else (
                f"{g.target:.2%}" if abs(g.target) < 10 else f"{g.target:.2f}"
            )
            current_str = f"{g.current}" if isinstance(g.current, bool) else (
                f"{g.current:.2%}" if abs(g.current) < 10 else f"{g.current:.2f}"
            )
            gap_str = "" if g.gap is None else (f"{g.gap:.2%}" if abs(g.gap) < 10 else f"{g.gap:.2f}")
            rows += (
                f"<tr><td>{dot} {html.escape(g.metric)}</td>"
                f"<td>{html.escape(target_str)}</td>"
                f"<td>{html.escape(current_str)}</td>"
                f"<td>{html.escape(gap_str)}</td>"
                f"<td>{html.escape(g.message)}</td></tr>\n"
            )
        return f"""
        <div class="card">
            <h2>Gap Analysis</h2>
            <table class="gap-table">
                <thead><tr><th>Metric</th><th>Target</th><th>Current</th><th>Gap</th><th>Action</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def generate_report(self, experiments: dict[str, dict[str, Any]]) -> str:
        """Evaluate experiments and return an investor-quality HTML report."""
        result = self.evaluate(experiments)

        experiment_cards = "\n".join(
            self._render_experiment_card(es) for es in result.experiment_statuses
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>North Star Dashboard</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: #121212;
        color: #e0e0e0;
        padding: 2rem;
    }}
    h1 {{
        text-align: center;
        font-size: 2rem;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }}
    .card {{
        background: #1e1e1e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,.4);
    }}
    .summary-card h2 {{
        margin-bottom: 1rem;
    }}
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 1rem;
    }}
    .metric {{
        display: flex;
        flex-direction: column;
        gap: .25rem;
    }}
    .metric .label {{ font-size: .85rem; color: #aaa; }}
    .metric .value {{ font-size: 1.2rem; font-weight: 600; }}
    .experiments-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
        gap: 1.5rem;
    }}
    .experiment-card h3 {{
        margin-bottom: .75rem;
        color: #ffffff;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    td, th {{
        text-align: left;
        padding: .45rem .6rem;
        border-bottom: 1px solid #333;
    }}
    th {{ color: #aaa; font-weight: 600; font-size: .85rem; }}
    .gap-table th {{ background: #262626; }}
    a {{ color: #82b1ff; }}
</style>
</head>
<body>
<h1>North Star Targets Dashboard</h1>
{self._render_summary_card(result)}
<h2 style="margin-bottom:1rem; color:#fff;">Experiments</h2>
<div class="experiments-grid">
{experiment_cards}
</div>
{self._render_gap_table(result.gap_analysis)}
</body>
</html>"""
