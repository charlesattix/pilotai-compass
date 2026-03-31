"""
Margin efficiency analyzer for credit spread portfolio.

Computes margin requirements per spread type, tracks utilization over time,
runs stress scenarios (VIX spikes), and produces a self-contained HTML report
at reports/margin_analysis.html.

This is READ-ONLY analysis.  No broker connections, no trade placement.

Usage::

    from compass.margin_analyzer import MarginAnalyzer
    analyzer = MarginAnalyzer(account_capital=100_000)
    analyzer.add_trades(trades_df)
    result = analyzer.analyze()
    MarginAnalyzer.generate_report(result)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "margin_analysis.html"

# ── Spread types and their margin formulas ───────────────────────────────

SPREAD_TYPES = ("credit_spread", "iron_condor", "straddle", "strangle")


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class MarginRequirement:
    """Margin requirement for a single position."""

    spread_type: str
    spread_width: float
    contracts: int
    premium_received: float
    margin_required: float
    underlying_price: float
    vix: float = 20.0

    @property
    def return_on_margin(self) -> float:
        if self.margin_required <= 0:
            return 0.0
        return self.premium_received / self.margin_required

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spread_type": self.spread_type,
            "spread_width": self.spread_width,
            "contracts": self.contracts,
            "premium_received": self.premium_received,
            "margin_required": self.margin_required,
            "return_on_margin": self.return_on_margin,
            "underlying_price": self.underlying_price,
            "vix": self.vix,
        }


@dataclass
class UtilizationSnapshot:
    """Margin utilization at a point in time."""

    date: pd.Timestamp
    total_margin_used: float
    account_capital: float
    n_positions: int

    @property
    def utilization_pct(self) -> float:
        if self.account_capital <= 0:
            return 0.0
        return self.total_margin_used / self.account_capital

    @property
    def buying_power_remaining(self) -> float:
        return max(0.0, self.account_capital - self.total_margin_used)


@dataclass
class ExperimentEfficiency:
    """Margin efficiency metrics for one experiment."""

    experiment: str
    total_margin_used: float
    total_premium: float
    total_pnl: float
    n_trades: int
    avg_return_on_margin: float
    margin_per_trade: float
    win_rate: float


@dataclass
class StressScenario:
    """Result of a margin stress test."""

    scenario_name: str
    vix_level: float
    vix_multiplier: float
    baseline_margin: float
    stressed_margin: float
    margin_increase_pct: float
    would_exceed_capital: bool
    capital_shortfall: float


@dataclass
class AnalysisResult:
    """Full result from margin analysis."""

    account_capital: float
    margin_requirements: List[MarginRequirement]
    utilization_history: List[UtilizationSnapshot]
    experiment_efficiency: List[ExperimentEfficiency]
    stress_scenarios: List[StressScenario]
    summary: Dict[str, Any]


# ── Margin calculation engine ────────────────────────────────────────────


def compute_margin(
    spread_type: str,
    spread_width: float,
    contracts: int,
    premium_received: float,
    underlying_price: float,
    vix: float = 20.0,
) -> float:
    """Compute margin requirement for a spread.

    Args:
        spread_type: One of SPREAD_TYPES.
        spread_width: Width between strikes in dollars.
        contracts: Number of contracts.
        premium_received: Total premium received in dollars.
        underlying_price: Current price of the underlying.
        vix: Current VIX level (affects naked/straddle margin).

    Returns:
        Margin required in dollars.
    """
    multiplier = contracts * 100

    if spread_type == "credit_spread":
        # Max loss = (spread_width - premium per share) * multiplier
        # Margin = spread_width * multiplier (standard Reg-T)
        margin = spread_width * multiplier
    elif spread_type == "iron_condor":
        # Margin = max of either side's spread width * multiplier
        # (only one side can lose, so margin = single side)
        margin = spread_width * multiplier
    elif spread_type == "straddle":
        # Naked-like margin: 20% of underlying + premium, scaled by VIX
        vix_factor = max(1.0, vix / 20.0)
        base = 0.20 * underlying_price * multiplier
        margin = (base + premium_received) * vix_factor
    elif spread_type == "strangle":
        # Similar to straddle but slightly less (OTM)
        vix_factor = max(1.0, vix / 20.0)
        base = 0.15 * underlying_price * multiplier
        margin = (base + premium_received) * vix_factor
    else:
        raise ValueError(f"Unknown spread_type: {spread_type!r}")

    return float(margin)


def compute_stressed_margin(
    margin_req: MarginRequirement,
    vix_multiplier: float,
) -> float:
    """Recompute margin under a stressed VIX scenario."""
    stressed_vix = margin_req.vix * vix_multiplier
    return compute_margin(
        spread_type=margin_req.spread_type,
        spread_width=margin_req.spread_width,
        contracts=margin_req.contracts,
        premium_received=margin_req.premium_received,
        underlying_price=margin_req.underlying_price,
        vix=stressed_vix,
    )


# ── Core analyzer ────────────────────────────────────────────────────────


class MarginAnalyzer:
    """Margin efficiency analyzer for options spread portfolios."""

    def __init__(self, account_capital: float = 100_000.0):
        if account_capital <= 0:
            raise ValueError("account_capital must be positive")
        self.account_capital = account_capital
        self._trades: Optional[pd.DataFrame] = None

    def add_trades(self, trades: pd.DataFrame) -> None:
        """Add trade data for analysis.

        Expected columns: date (or entry_date), spread_type, spread_width,
        contracts, premium, pnl, underlying_price, vix, experiment (optional).
        """
        required = {"spread_type", "spread_width", "contracts", "premium"}
        missing = required - set(trades.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self._trades = trades.copy()

    # ── Margin requirements ──────────────────────────────────────────

    def compute_all_margins(self) -> List[MarginRequirement]:
        """Compute margin for every trade."""
        if self._trades is None or self._trades.empty:
            return []

        results: List[MarginRequirement] = []
        for _, row in self._trades.iterrows():
            spread_type = str(row["spread_type"])
            spread_width = float(row["spread_width"])
            contracts = int(row["contracts"])
            premium = float(row["premium"])
            underlying = float(row.get("underlying_price", 450.0))
            vix = float(row.get("vix", 20.0))

            margin = compute_margin(
                spread_type, spread_width, contracts, premium, underlying, vix
            )
            results.append(
                MarginRequirement(
                    spread_type=spread_type,
                    spread_width=spread_width,
                    contracts=contracts,
                    premium_received=premium,
                    margin_required=margin,
                    underlying_price=underlying,
                    vix=vix,
                )
            )
        return results

    # ── Utilization tracking ─────────────────────────────────────────

    def compute_utilization_history(
        self, margin_reqs: List[MarginRequirement]
    ) -> List[UtilizationSnapshot]:
        """Track margin utilization over time."""
        if self._trades is None or self._trades.empty:
            return []

        date_col = "date" if "date" in self._trades.columns else "entry_date"
        if date_col not in self._trades.columns:
            return []

        df = self._trades.copy()
        df["_date"] = pd.to_datetime(df[date_col])
        df["_margin"] = [m.margin_required for m in margin_reqs]

        daily = df.groupby("_date").agg(
            total_margin=("_margin", "sum"),
            n_positions=("_margin", "count"),
        )

        snapshots: List[UtilizationSnapshot] = []
        for date, row in daily.iterrows():
            snapshots.append(
                UtilizationSnapshot(
                    date=pd.Timestamp(date),
                    total_margin_used=float(row["total_margin"]),
                    account_capital=self.account_capital,
                    n_positions=int(row["n_positions"]),
                )
            )
        return sorted(snapshots, key=lambda s: s.date)

    # ── Cross-experiment efficiency ──────────────────────────────────

    def compute_experiment_efficiency(
        self, margin_reqs: List[MarginRequirement]
    ) -> List[ExperimentEfficiency]:
        """Rank experiments by margin efficiency."""
        if self._trades is None or self._trades.empty:
            return []
        if "experiment" not in self._trades.columns:
            return []

        df = self._trades.copy()
        df["_margin"] = [m.margin_required for m in margin_reqs]
        df["_rom"] = [m.return_on_margin for m in margin_reqs]

        results: List[ExperimentEfficiency] = []
        for exp, grp in df.groupby("experiment"):
            pnl_col = "pnl" if "pnl" in grp.columns else None
            total_pnl = float(grp[pnl_col].sum()) if pnl_col else 0.0
            wins = int((grp[pnl_col] > 0).sum()) if pnl_col else 0
            n = len(grp)

            results.append(
                ExperimentEfficiency(
                    experiment=str(exp),
                    total_margin_used=float(grp["_margin"].sum()),
                    total_premium=float(grp["premium"].sum()),
                    total_pnl=total_pnl,
                    n_trades=n,
                    avg_return_on_margin=float(grp["_rom"].mean()),
                    margin_per_trade=float(grp["_margin"].mean()),
                    win_rate=wins / n if n > 0 else 0.0,
                )
            )
        return sorted(results, key=lambda e: e.avg_return_on_margin, reverse=True)

    # ── Stress scenarios ─────────────────────────────────────────────

    def run_stress_scenarios(
        self,
        margin_reqs: List[MarginRequirement],
        scenarios: Optional[List[Tuple[str, float]]] = None,
    ) -> List[StressScenario]:
        """Simulate margin impact under VIX stress scenarios.

        Args:
            margin_reqs: Current margin requirements.
            scenarios: List of (name, vix_multiplier) tuples.
                       Defaults to standard stress levels.
        """
        if not margin_reqs:
            return []

        if scenarios is None:
            scenarios = [
                ("Mild stress (VIX +25%)", 1.25),
                ("Moderate stress (VIX +50%)", 1.50),
                ("Severe stress (VIX 2x)", 2.0),
                ("Crisis (VIX 3x)", 3.0),
                ("Black swan (VIX 4x)", 4.0),
            ]

        baseline_total = sum(m.margin_required for m in margin_reqs)
        results: List[StressScenario] = []

        for name, mult in scenarios:
            stressed_total = sum(
                compute_stressed_margin(m, mult) for m in margin_reqs
            )
            increase_pct = (
                (stressed_total - baseline_total) / baseline_total
                if baseline_total > 0
                else 0.0
            )
            shortfall = max(0.0, stressed_total - self.account_capital)

            results.append(
                StressScenario(
                    scenario_name=name,
                    vix_level=margin_reqs[0].vix * mult,
                    vix_multiplier=mult,
                    baseline_margin=baseline_total,
                    stressed_margin=stressed_total,
                    margin_increase_pct=increase_pct,
                    would_exceed_capital=stressed_total > self.account_capital,
                    capital_shortfall=shortfall,
                )
            )
        return results

    # ── Buying power impact ──────────────────────────────────────────

    @staticmethod
    def buying_power_impact(
        account_capital: float,
        margin_reqs: List[MarginRequirement],
    ) -> Dict[str, Any]:
        """Compute buying power metrics."""
        total_margin = sum(m.margin_required for m in margin_reqs)
        remaining = max(0.0, account_capital - total_margin)
        utilization = total_margin / account_capital if account_capital > 0 else 0.0

        by_type: Dict[str, float] = {}
        for m in margin_reqs:
            by_type[m.spread_type] = by_type.get(m.spread_type, 0.0) + m.margin_required

        return {
            "account_capital": account_capital,
            "total_margin_used": total_margin,
            "buying_power_remaining": remaining,
            "utilization_pct": utilization,
            "margin_by_spread_type": by_type,
            "n_positions": len(margin_reqs),
            "avg_margin_per_position": total_margin / len(margin_reqs) if margin_reqs else 0.0,
        }

    # ── Main analyze ─────────────────────────────────────────────────

    def analyze(self) -> AnalysisResult:
        """Run full margin analysis."""
        margin_reqs = self.compute_all_margins()
        utilization = self.compute_utilization_history(margin_reqs)
        efficiency = self.compute_experiment_efficiency(margin_reqs)
        stress = self.run_stress_scenarios(margin_reqs)
        bp = self.buying_power_impact(self.account_capital, margin_reqs)

        total_margin = sum(m.margin_required for m in margin_reqs)
        total_premium = sum(m.premium_received for m in margin_reqs)
        avg_rom = (
            np.mean([m.return_on_margin for m in margin_reqs])
            if margin_reqs
            else 0.0
        )

        summary = {
            "account_capital": self.account_capital,
            "total_margin": total_margin,
            "total_premium": total_premium,
            "avg_return_on_margin": float(avg_rom),
            "n_positions": len(margin_reqs),
            "buying_power": bp,
        }

        return AnalysisResult(
            account_capital=self.account_capital,
            margin_requirements=margin_reqs,
            utilization_history=utilization,
            experiment_efficiency=efficiency,
            stress_scenarios=stress,
            summary=summary,
        )

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        result: AnalysisResult,
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


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.2f}"


def _utilization_svg(snapshots: List[UtilizationSnapshot]) -> str:
    """Inline SVG chart of margin utilization over time."""
    if not snapshots:
        return "<p>No utilization data available.</p>"

    w, h = 700, 300
    pad = 60
    utils = [s.utilization_pct for s in snapshots]
    n = len(utils)

    max_u = max(max(utils), 0.01)
    chart_w = w - 2 * pad
    chart_h = h - 2 * pad

    points = []
    for i, u in enumerate(utils):
        x = pad + (i / max(1, n - 1)) * chart_w if n > 1 else pad + chart_w / 2
        y = h - pad - (u / max_u) * chart_h
        points.append(f"{x:.1f},{y:.1f}")

    fill_points = [f"{pad},{h - pad}"] + points + [f"{pad + chart_w},{h - pad}"]

    line = f'<polyline points="{" ".join(points)}" fill="none" stroke="#58a6ff" stroke-width="2"/>'
    area = f'<polygon points="{" ".join(fill_points)}" fill="#58a6ff" opacity="0.15"/>'

    # Threshold line at 80%
    if max_u >= 0.8:
        y80 = h - pad - (0.8 / max_u) * chart_h
        threshold = (
            f'<line x1="{pad}" y1="{y80:.1f}" x2="{w - pad}" y2="{y80:.1f}" '
            f'stroke="#f85149" stroke-dasharray="5,5" stroke-width="1"/>'
            f'<text x="{w - pad + 5}" y="{y80:.1f}" class="svg-label" fill="#f85149">80%</text>'
        )
    else:
        threshold = ""

    return f"""
    <svg viewBox="0 0 {w} {h}" class="chart">
      <text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">
        Margin Utilization Over Time
      </text>
      <text x="15" y="{h // 2}" text-anchor="middle" class="svg-label"
            transform="rotate(-90,15,{h // 2})">Utilization</text>
      {area}
      {line}
      {threshold}
    </svg>"""


def _efficiency_table(experiments: List[ExperimentEfficiency]) -> str:
    if not experiments:
        return "<p>No experiment data available.</p>"
    rows = ""
    for e in experiments:
        rows += f"""<tr>
          <td>{e.experiment}</td>
          <td>{e.n_trades}</td>
          <td>{_fmt_dollar(e.margin_per_trade)}</td>
          <td>{_fmt_dollar(e.total_premium)}</td>
          <td>{_fmt_dollar(e.total_pnl)}</td>
          <td>{_fmt_pct(e.avg_return_on_margin)}</td>
          <td>{_fmt_pct(e.win_rate)}</td>
        </tr>"""
    return f"""
    <table class="data-table">
      <tr>
        <th>Experiment</th><th>Trades</th><th>Avg Margin</th>
        <th>Total Premium</th><th>Total P&amp;L</th>
        <th>Avg ROM</th><th>Win Rate</th>
      </tr>
      {rows}
    </table>"""


def _stress_table(scenarios: List[StressScenario]) -> str:
    if not scenarios:
        return "<p>No stress scenarios available.</p>"
    rows = ""
    for s in scenarios:
        status_cls = "danger" if s.would_exceed_capital else "ok"
        status = "EXCEEDS" if s.would_exceed_capital else "OK"
        rows += f"""<tr>
          <td>{s.scenario_name}</td>
          <td>{s.vix_level:.1f}</td>
          <td>{_fmt_dollar(s.baseline_margin)}</td>
          <td>{_fmt_dollar(s.stressed_margin)}</td>
          <td>{_fmt_pct(s.margin_increase_pct)}</td>
          <td class="{status_cls}">{status}</td>
          <td>{_fmt_dollar(s.capital_shortfall)}</td>
        </tr>"""
    return f"""
    <table class="data-table">
      <tr>
        <th>Scenario</th><th>VIX</th><th>Baseline</th>
        <th>Stressed</th><th>Increase</th>
        <th>Status</th><th>Shortfall</th>
      </tr>
      {rows}
    </table>"""


def _buying_power_card(bp: Dict[str, Any]) -> str:
    by_type = bp.get("margin_by_spread_type", {})
    type_rows = "".join(
        f"<tr><td>{t}</td><td>{_fmt_dollar(v)}</td></tr>"
        for t, v in sorted(by_type.items())
    )
    return f"""
    <div class="card">
      <h3>Buying Power</h3>
      <div class="metrics-grid">
        <div><span class="label">Capital</span><span class="value">{_fmt_dollar(bp['account_capital'])}</span></div>
        <div><span class="label">Margin Used</span><span class="value">{_fmt_dollar(bp['total_margin_used'])}</span></div>
        <div><span class="label">Remaining</span><span class="value">{_fmt_dollar(bp['buying_power_remaining'])}</span></div>
        <div><span class="label">Utilization</span><span class="value">{_fmt_pct(bp['utilization_pct'])}</span></div>
        <div><span class="label">Positions</span><span class="value">{bp['n_positions']}</span></div>
        <div><span class="label">Avg Margin/Pos</span><span class="value">{_fmt_dollar(bp['avg_margin_per_position'])}</span></div>
      </div>
      {f'<table class="weights"><tr><th>Type</th><th>Margin</th></tr>{type_rows}</table>' if type_rows else ''}
    </div>"""


def _recommendations(result: AnalysisResult) -> str:
    """Generate capital allocation recommendations."""
    recs: List[str] = []

    bp = result.summary.get("buying_power", {})
    util = bp.get("utilization_pct", 0)
    if util > 0.8:
        recs.append("Margin utilization above 80% — reduce position count or size to avoid margin calls.")
    elif util > 0.6:
        recs.append("Utilization at 60-80% — monitor closely, limited room for new positions.")
    elif util < 0.3:
        recs.append("Low utilization (<30%) — capital is underdeployed, consider adding positions.")

    # Check stress scenarios
    for s in result.stress_scenarios:
        if s.would_exceed_capital and s.vix_multiplier <= 2.0:
            recs.append(
                f"WARNING: {s.scenario_name} would exceed capital by "
                f"{_fmt_dollar(s.capital_shortfall)}. Reduce exposure."
            )
            break

    # Best experiment
    if result.experiment_efficiency:
        best = result.experiment_efficiency[0]
        recs.append(
            f"Most efficient experiment: {best.experiment} "
            f"(ROM: {_fmt_pct(best.avg_return_on_margin)}). "
            f"Consider increasing allocation."
        )

    if not recs:
        recs.append("Portfolio margin utilization appears healthy.")

    items = "".join(f"<li>{r}</li>" for r in recs)
    return f"<ul class='recs'>{items}</ul>"


def _build_html(result: AnalysisResult) -> str:
    bp = result.summary.get("buying_power", {})
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Margin Efficiency Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; margin: 16px 0; }}
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }}
  .metrics-grid .label {{ color: #8b949e; font-size: 0.85em; }}
  .metrics-grid .value {{ color: #f0f6fc; font-weight: 600; }}
  .chart {{ width: 100%; max-width: 750px; margin: 20px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 14px; }}
  .svg-label {{ fill: #8b949e; font-size: 11px; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
  table.data-table th, table.data-table td {{ padding: 8px 12px; text-align: left;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  table.weights {{ width: 100%; margin-top: 12px; border-collapse: collapse; }}
  table.weights th, table.weights td {{ padding: 4px 8px; text-align: left;
                                         border-bottom: 1px solid #21262d; }}
  table.weights th {{ color: #8b949e; }}
  .danger {{ color: #f85149; font-weight: 700; }}
  .ok {{ color: #3fb950; font-weight: 700; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  ul.recs {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
             padding: 16px 16px 16px 32px; }}
  ul.recs li {{ margin: 8px 0; }}
</style>
</head>
<body>
<h1>Margin Efficiency Analysis</h1>
<p class="meta">Account capital: {_fmt_dollar(result.account_capital)} &middot;
   {len(result.margin_requirements)} positions analyzed</p>

{_buying_power_card(bp)}

{_utilization_svg(result.utilization_history)}

<h2>Per-Experiment Efficiency</h2>
{_efficiency_table(result.experiment_efficiency)}

<h2>Stress Scenarios</h2>
{_stress_table(result.stress_scenarios)}

<h2>Recommendations</h2>
{_recommendations(result)}

</body>
</html>"""
