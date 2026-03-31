"""
Backtest reality checker -- detect biases, unrealistic assumptions, and
over-fitting indicators in backtest results.

Checks: look-ahead bias, survivorship bias, transaction cost realism,
fill realism, capacity, parameter sensitivity, OOS degradation,
overfitting ratio, complexity penalty.  Composite credibility 0-100.

Usage::

    from compass.backtest_reality import BacktestRealityChecker
    checker = BacktestRealityChecker(trades=df, returns=series, ...)
    result = checker.run_all()
    html = checker.generate_report()
"""

from __future__ import annotations

import html as _html
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of a single reality check."""

    name: str
    passed: bool
    score: float  # 0-100, higher = better
    detail: str


@dataclass
class BacktestRealityResult:
    """Aggregated output of all reality checks."""

    checks: List[CheckResult]
    credibility_score: float  # 0-100
    grade: str  # A / B / C / D / F
    recommendations: List[str]


# ---------------------------------------------------------------------------
# Weights for credibility score (must sum to 1.0)
# ---------------------------------------------------------------------------

_CHECK_WEIGHTS: Dict[str, float] = {
    "look_ahead_bias": 0.15,
    "survivorship_bias": 0.10,
    "transaction_cost_realism": 0.10,
    "fill_realism": 0.10,
    "capacity_check": 0.10,
    "parameter_sensitivity": 0.10,
    "oos_degradation": 0.15,
    "overfitting_ratio": 0.10,
    "complexity_penalty": 0.10,
}


def _grade_from_score(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# BacktestRealityChecker
# ---------------------------------------------------------------------------


class BacktestRealityChecker:
    """Run a suite of reality checks on backtest output.

    Parameters
    ----------
    trades : pd.DataFrame
        Must contain columns ``date`` (or ``entry_date`` + ``exit_date``),
        ``pnl``, ``entry_price``, ``exit_price``, ``quantity``.
    returns : pd.Series
        Daily strategy returns indexed by date.
    n_params_tested : int
        Number of free parameters / variants tested during optimisation.
    adv : float
        Average daily volume (shares or contracts) for the traded instrument.
    assumed_spread_bps : float
        Bid-ask spread assumed in the backtest (basis points).
    assumed_commission : float
        Per-trade commission assumed in the backtest.
    param_sweep : dict | None
        ``{param_name: [sharpe_at_val0, sharpe_at_val1, ...]}`` used by
        the parameter-sensitivity check to detect cliffs.
    is_sharpe : float | None
        In-sample Sharpe ratio (annualised).
    oos_sharpe : float | None
        Out-of-sample Sharpe ratio (annualised).
    """

    def __init__(
        self,
        trades: pd.DataFrame,
        returns: pd.Series,
        *,
        n_params_tested: int = 1,
        adv: float = 100_000.0,
        assumed_spread_bps: float = 1.0,
        assumed_commission: float = 1.0,
        param_sweep: Optional[Dict[str, List[float]]] = None,
        is_sharpe: Optional[float] = None,
        oos_sharpe: Optional[float] = None,
    ) -> None:
        self.trades = trades.copy()
        self.returns = returns.copy()
        self.n_params_tested = n_params_tested
        self.adv = adv
        self.assumed_spread_bps = assumed_spread_bps
        self.assumed_commission = assumed_commission
        self.param_sweep = param_sweep or {}
        self.is_sharpe = is_sharpe
        self.oos_sharpe = oos_sharpe

        self._result: Optional[BacktestRealityResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> BacktestRealityResult:
        """Execute every check and return the aggregated result."""
        checks: List[CheckResult] = [
            self.check_look_ahead_bias(),
            self.check_survivorship_bias(),
            self.check_transaction_cost_realism(),
            self.check_fill_realism(),
            self.check_capacity(),
            self.check_parameter_sensitivity(),
            self.check_oos_degradation(),
            self.check_overfitting_ratio(),
            self.check_complexity_penalty(),
        ]
        score = self._credibility_score(checks)
        grade = _grade_from_score(score)
        recommendations = self._build_recommendations(checks)
        self._result = BacktestRealityResult(
            checks=checks,
            credibility_score=score,
            grade=grade,
            recommendations=recommendations,
        )
        return self._result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_look_ahead_bias(self) -> CheckResult:
        """Flag trades where exit_date < entry_date."""
        df = self.trades
        bad = 0
        if "entry_date" in df.columns and "exit_date" in df.columns:
            entry = pd.to_datetime(df["entry_date"])
            exit_ = pd.to_datetime(df["exit_date"])
            bad = int((exit_ < entry).sum())
        if bad > 0:
            return CheckResult(
                "look_ahead_bias",
                False,
                0.0,
                f"{bad} trade(s) have exit_date before entry_date",
            )
        return CheckResult(
            "look_ahead_bias",
            True,
            100.0,
            "No look-ahead bias detected",
        )

    def check_survivorship_bias(self) -> CheckResult:
        """Warn if single ticker or data gaps > 30 days."""
        warnings: List[str] = []
        score = 100.0

        # Single ticker
        if "ticker" in self.trades.columns:
            n_tickers = self.trades["ticker"].nunique()
        else:
            n_tickers = 1  # assume single ticker if column missing
        if n_tickers <= 1:
            warnings.append("single ticker traded -- survivorship risk")
            score -= 30.0

        # Data gaps
        if len(self.returns) > 1:
            idx = pd.to_datetime(self.returns.index)
            diffs = idx.to_series().diff().dt.days.dropna()
            max_gap = float(diffs.max()) if len(diffs) > 0 else 0.0
            if max_gap > 30:
                warnings.append(
                    f"max data gap {max_gap:.0f} days (>30) -- possible survivorship"
                )
                score -= 30.0

        score = max(score, 0.0)
        passed = len(warnings) == 0
        detail = "; ".join(warnings) if warnings else "No survivorship bias indicators"
        return CheckResult("survivorship_bias", passed, score, detail)

    def check_transaction_cost_realism(self) -> CheckResult:
        """Compare assumed spread/commission vs typical values."""
        warnings: List[str] = []
        score = 100.0
        typical_spread_bps = 5.0
        typical_commission = 1.0

        if self.assumed_spread_bps < typical_spread_bps * 0.5:
            penalty = min(50.0, (typical_spread_bps - self.assumed_spread_bps) / typical_spread_bps * 100)
            score -= penalty
            warnings.append(
                f"assumed spread {self.assumed_spread_bps:.1f}bps is below "
                f"typical {typical_spread_bps:.1f}bps"
            )
        if self.assumed_commission < typical_commission * 0.5:
            score -= 20.0
            warnings.append(
                f"assumed commission ${self.assumed_commission:.2f} is below "
                f"typical ${typical_commission:.2f}"
            )

        score = max(score, 0.0)
        passed = score >= 70.0
        detail = "; ".join(warnings) if warnings else "Transaction costs are realistic"
        return CheckResult("transaction_cost_realism", passed, score, detail)

    def check_fill_realism(self) -> CheckResult:
        """Check if any trade size > 10% of ADV."""
        if self.adv <= 0:
            return CheckResult(
                "fill_realism", False, 0.0, "ADV is zero or negative"
            )
        qty = self.trades["quantity"].abs() if "quantity" in self.trades.columns else pd.Series(dtype=float)
        if len(qty) == 0:
            return CheckResult("fill_realism", True, 100.0, "No quantity data to check")

        ratio = qty / self.adv
        n_bad = int((ratio > 0.10).sum())
        max_ratio = float(ratio.max())
        if n_bad > 0:
            score = max(0.0, 100.0 - n_bad / max(len(qty), 1) * 100.0 * 2)
            return CheckResult(
                "fill_realism",
                False,
                score,
                f"{n_bad} trade(s) exceed 10% ADV (max {max_ratio:.1%})",
            )
        return CheckResult(
            "fill_realism",
            True,
            100.0,
            f"All trades within 10% ADV (max {max_ratio:.1%})",
        )

    def check_capacity(self) -> CheckResult:
        """Estimate max AUM from trade sizes vs ADV."""
        if self.adv <= 0:
            return CheckResult("capacity_check", False, 0.0, "ADV is zero or negative")

        if "quantity" in self.trades.columns and "entry_price" in self.trades.columns:
            notional = (self.trades["quantity"].abs() * self.trades["entry_price"].abs())
            avg_notional = float(notional.mean()) if len(notional) > 0 else 0.0
        else:
            avg_notional = 0.0

        # Estimate capacity as ADV * price * 2% participation
        if "entry_price" in self.trades.columns and len(self.trades) > 0:
            avg_price = float(self.trades["entry_price"].abs().mean())
        else:
            avg_price = 1.0
        max_aum = self.adv * avg_price * 0.02  # 2% participation ceiling

        if avg_notional > 0 and max_aum > 0:
            utilisation = avg_notional / max_aum
            score = max(0.0, min(100.0, (1.0 - utilisation) * 100.0))
        else:
            utilisation = 0.0
            score = 100.0

        passed = score >= 50.0
        detail = (
            f"avg notional ${avg_notional:,.0f}, "
            f"est. max AUM ${max_aum:,.0f}, "
            f"utilisation {utilisation:.1%}"
        )
        return CheckResult("capacity_check", passed, score, detail)

    def check_parameter_sensitivity(self) -> CheckResult:
        """Detect parameter cliffs (>50% Sharpe drop at neighbours)."""
        if not self.param_sweep:
            return CheckResult(
                "parameter_sensitivity",
                True,
                100.0,
                "No param_sweep provided; skipped",
            )

        cliff_params: List[str] = []
        for param, sharpes in self.param_sweep.items():
            if len(sharpes) < 2:
                continue
            arr = np.asarray(sharpes, dtype=float)
            for i in range(1, len(arr)):
                prev = arr[i - 1]
                curr = arr[i]
                if abs(prev) > 1e-9:
                    drop = (prev - curr) / abs(prev)
                    if drop > 0.50:
                        cliff_params.append(param)
                        break

        if cliff_params:
            score = max(0.0, 100.0 - len(cliff_params) * 25.0)
            return CheckResult(
                "parameter_sensitivity",
                False,
                score,
                f"Cliff detected in: {', '.join(cliff_params)}",
            )
        return CheckResult(
            "parameter_sensitivity",
            True,
            100.0,
            "No parameter cliffs detected",
        )

    def check_oos_degradation(self) -> CheckResult:
        """Flag if OOS Sharpe degrades > 30% vs IS."""
        if self.is_sharpe is None or self.oos_sharpe is None:
            return CheckResult(
                "oos_degradation",
                True,
                100.0,
                "IS/OOS Sharpe not provided; skipped",
            )
        if abs(self.is_sharpe) < 1e-9:
            return CheckResult(
                "oos_degradation",
                True,
                50.0,
                "IS Sharpe near zero; cannot compute degradation",
            )
        degradation = round((self.is_sharpe - self.oos_sharpe) / abs(self.is_sharpe), 10)
        if degradation > 0.30:
            score = max(0.0, 100.0 - degradation * 100.0)
            return CheckResult(
                "oos_degradation",
                False,
                score,
                f"OOS degradation {degradation:.0%} exceeds 30% threshold "
                f"(IS={self.is_sharpe:.2f}, OOS={self.oos_sharpe:.2f})",
            )
        score = max(0.0, 100.0 - degradation * 100.0)
        return CheckResult(
            "oos_degradation",
            True,
            score,
            f"OOS degradation {degradation:.0%} within 30% threshold "
            f"(IS={self.is_sharpe:.2f}, OOS={self.oos_sharpe:.2f})",
        )

    def check_overfitting_ratio(self) -> CheckResult:
        """n_params / n_trades; warn if > 0.1."""
        n_trades = len(self.trades)
        if n_trades == 0:
            return CheckResult(
                "overfitting_ratio",
                False,
                0.0,
                "No trades to evaluate",
            )
        ratio = self.n_params_tested / n_trades
        if ratio > 0.10:
            score = max(0.0, 100.0 - (ratio - 0.10) / 0.10 * 50.0)
            return CheckResult(
                "overfitting_ratio",
                False,
                score,
                f"params/trades ratio {ratio:.3f} exceeds 0.1 "
                f"({self.n_params_tested} params, {n_trades} trades)",
            )
        score = 100.0 - ratio / 0.10 * 30.0  # mild linear penalty
        return CheckResult(
            "overfitting_ratio",
            True,
            max(score, 0.0),
            f"params/trades ratio {ratio:.3f} is acceptable",
        )

    def check_complexity_penalty(self) -> CheckResult:
        """Score based on n_params_tested: fewer is better."""
        n = self.n_params_tested
        # Piecewise scoring
        if n <= 3:
            score = 100.0
        elif n <= 10:
            score = 100.0 - (n - 3) * 5.0  # 65 at n=10
        elif n <= 30:
            score = 65.0 - (n - 10) * 2.0  # 25 at n=30
        else:
            score = max(0.0, 25.0 - (n - 30) * 1.0)

        passed = score >= 50.0
        detail = f"{n} parameter(s) tested -- complexity score {score:.0f}"
        return CheckResult("complexity_penalty", passed, score, detail)

    # ------------------------------------------------------------------
    # Credibility score
    # ------------------------------------------------------------------

    def _credibility_score(self, checks: List[CheckResult]) -> float:
        total = 0.0
        weight_sum = 0.0
        for c in checks:
            w = _CHECK_WEIGHTS.get(c.name, 0.0)
            total += c.score * w
            weight_sum += w
        if weight_sum <= 0:
            return 0.0
        return total / weight_sum

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendations(checks: List[CheckResult]) -> List[str]:
        recs: List[str] = []
        for c in checks:
            if c.passed:
                continue
            if c.name == "look_ahead_bias":
                recs.append("Fix trades where exit_date precedes entry_date.")
            elif c.name == "survivorship_bias":
                recs.append("Add more tickers or fill data gaps to reduce survivorship risk.")
            elif c.name == "transaction_cost_realism":
                recs.append("Increase assumed spread/commission to realistic levels.")
            elif c.name == "fill_realism":
                recs.append("Reduce trade sizes or use a fill model that accounts for ADV.")
            elif c.name == "capacity_check":
                recs.append("Consider smaller position sizes to stay within capacity limits.")
            elif c.name == "parameter_sensitivity":
                recs.append("Smooth parameter cliffs; prefer robust parameter regions.")
            elif c.name == "oos_degradation":
                recs.append("Reduce model complexity to narrow the IS/OOS gap.")
            elif c.name == "overfitting_ratio":
                recs.append("Reduce the number of tested parameters relative to trade count.")
            elif c.name == "complexity_penalty":
                recs.append("Simplify the strategy by reducing free parameters.")
        return recs

    # ------------------------------------------------------------------
    # HTML report with SVG charts
    # ------------------------------------------------------------------

    def generate_report(self, output: Optional[str] = None) -> str:
        """Generate an HTML report.  Returns the HTML string.

        If *output* is given the HTML is also written to that path.
        """
        if self._result is None:
            self.run_all()
        assert self._result is not None

        res = self._result
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # -- credibility gauge (SVG) --
        gauge_svg = self._svg_gauge(res.credibility_score, res.grade)

        # -- checklist table --
        rows = ""
        for c in res.checks:
            icon = "&#9989;" if c.passed else "&#10060;"
            score_cls = "good" if c.score >= 70 else "warn" if c.score >= 40 else "bad"
            rows += (
                f"<tr>"
                f"<td>{icon}</td>"
                f"<td>{_html.escape(c.name)}</td>"
                f"<td class=\"{score_cls}\">{c.score:.0f}</td>"
                f"<td>{_html.escape(c.detail)}</td>"
                f"</tr>\n"
            )

        # -- tornado SVG --
        tornado_svg = self._svg_tornado()

        # -- degradation SVG --
        degrad_svg = self._svg_degradation()

        # -- recommendations --
        rec_items = "".join(
            f"<li>{_html.escape(r)}</li>" for r in res.recommendations
        ) or "<li>No recommendations -- all checks passed.</li>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Backtest Reality Check</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b;
}}
h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: .4em; }}
h2 {{ color: #334155; margin-top: 2em; }}
.meta {{ color: #64748b; font-size: .9em; margin-bottom: 1.5em; }}
.good {{ color: #16a34a; font-weight: 600; }}
.warn {{ color: #f59e0b; font-weight: 600; }}
.bad  {{ color: #dc2626; font-weight: 600; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .85em; }}
th {{ background: #f1f5f9; padding: 8px 10px; text-align: left; border-bottom: 2px solid #cbd5e1; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
.chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em; margin: 1.5em 0; text-align: center; }}
.gauge-section {{ text-align: center; margin: 1.5em 0; }}
ul {{ margin: .5em 0; padding-left: 1.5em; }}
footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
          font-size: .8em; color: #94a3b8; }}
</style>
</head>
<body>
<h1>Backtest Reality Check</h1>
<div class="meta">Generated {now} &middot; {len(self.trades)} trades &middot;
{len(self.returns)} return observations</div>

<div class="gauge-section">
{gauge_svg}
</div>

<h2>Checklist</h2>
<table>
<thead><tr><th></th><th>Check</th><th>Score</th><th>Detail</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>

<h2>Parameter Sensitivity</h2>
<div class="chart">
{tornado_svg}
</div>

<h2>Degradation Analysis</h2>
<div class="chart">
{degrad_svg}
</div>

<h2>Recommendations</h2>
<ul>
{rec_items}
</ul>

<footer>Generated by <code>compass/backtest_reality.py</code></footer>
</body>
</html>"""

        if output:
            p = Path(output)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html)

        return html

    # ------------------------------------------------------------------
    # SVG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_gauge(score: float, grade: str) -> str:
        """Semi-circle credibility gauge."""
        if score >= 85:
            colour = "#16a34a"
        elif score >= 70:
            colour = "#22c55e"
        elif score >= 55:
            colour = "#f59e0b"
        elif score >= 40:
            colour = "#f97316"
        else:
            colour = "#dc2626"

        # Arc from 180 to 0 degrees (left to right)
        frac = max(0.0, min(score / 100.0, 1.0))
        angle = math.pi * (1 - frac)  # radians from left
        ex = 100 + 80 * math.cos(angle)
        ey = 100 - 80 * math.sin(angle)
        large_arc = 1 if frac > 0.5 else 0

        return (
            f'<svg width="220" height="140" viewBox="0 0 220 140" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<path d="M 20 100 A 80 80 0 0 1 180 100" '
            f'fill="none" stroke="#e2e8f0" stroke-width="14" stroke-linecap="round"/>'
            f'<path d="M 20 100 A 80 80 0 {large_arc} 1 {ex:.1f} {ey:.1f}" '
            f'fill="none" stroke="{colour}" stroke-width="14" stroke-linecap="round"/>'
            f'<text x="100" y="95" text-anchor="middle" '
            f'font-size="28" font-weight="700" fill="{colour}">{score:.0f}</text>'
            f'<text x="100" y="120" text-anchor="middle" '
            f'font-size="16" font-weight="600" fill="#334155">Grade {grade}</text>'
            f'</svg>'
        )

    def _svg_tornado(self) -> str:
        """Tornado chart of param_sweep Sharpe values."""
        if not self.param_sweep:
            return "<em>No parameter sweep data provided.</em>"

        params = list(self.param_sweep.keys())
        bar_h = 28
        gap = 6
        total_h = (bar_h + gap) * len(params) + 40
        chart_w = 500
        mid_x = chart_w // 2
        max_range = 0.0

        # Compute ranges
        ranges: List[Tuple[str, float, float]] = []
        for p in params:
            vals = self.param_sweep[p]
            if len(vals) == 0:
                ranges.append((p, 0.0, 0.0))
                continue
            mn = float(np.min(vals))
            mx = float(np.max(vals))
            ranges.append((p, mn, mx))
            max_range = max(max_range, abs(mx), abs(mn))

        if max_range < 1e-9:
            max_range = 1.0
        scale = (mid_x - 60) / max_range

        bars = ""
        for i, (p, mn, mx) in enumerate(ranges):
            y = 30 + i * (bar_h + gap)
            x1 = mid_x + mn * scale
            x2 = mid_x + mx * scale
            bx = min(x1, x2)
            bw = max(abs(x2 - x1), 1)
            colour = "#dc2626" if (mx - mn) / max(abs(mx), 1e-9) > 0.5 else "#3b82f6"
            bars += (
                f'<rect x="{bx:.1f}" y="{y}" width="{bw:.1f}" height="{bar_h}" '
                f'fill="{colour}" rx="3"/>'
                f'<text x="5" y="{y + bar_h * 0.7}" font-size="12" '
                f'fill="#334155">{_html.escape(p)}</text>'
            )

        return (
            f'<svg width="{chart_w}" height="{total_h}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{mid_x}" y1="20" x2="{mid_x}" y2="{total_h - 10}" '
            f'stroke="#94a3b8" stroke-width="1"/>'
            f'{bars}'
            f'<text x="{mid_x}" y="15" text-anchor="middle" font-size="13" '
            f'font-weight="600" fill="#334155">Parameter Sensitivity (Sharpe)</text>'
            f'</svg>'
        )

    def _svg_degradation(self) -> str:
        """Bar chart comparing IS vs OOS Sharpe."""
        is_s = self.is_sharpe
        oos_s = self.oos_sharpe
        if is_s is None or oos_s is None:
            return "<em>IS/OOS Sharpe not provided.</em>"

        max_val = max(abs(is_s), abs(oos_s), 0.01)
        scale = 120.0 / max_val
        bar_w = 60
        chart_w = 300
        chart_h = 200
        base_y = 160

        def _bar(x: float, val: float, label: str, colour: str) -> str:
            h = abs(val) * scale
            y = base_y - h if val >= 0 else base_y
            return (
                f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" '
                f'fill="{colour}" rx="3"/>'
                f'<text x="{x + bar_w / 2}" y="{base_y + 18}" text-anchor="middle" '
                f'font-size="12" fill="#334155">{label}</text>'
                f'<text x="{x + bar_w / 2}" y="{y - 5:.1f}" text-anchor="middle" '
                f'font-size="12" font-weight="600" fill="{colour}">{val:.2f}</text>'
            )

        bars = _bar(60, is_s, "In-Sample", "#3b82f6")
        bars += _bar(180, oos_s, "OOS", "#f59e0b")

        return (
            f'<svg width="{chart_w}" height="{chart_h}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="40" y1="{base_y}" x2="280" y2="{base_y}" '
            f'stroke="#cbd5e1" stroke-width="1"/>'
            f'{bars}'
            f'<text x="150" y="15" text-anchor="middle" font-size="13" '
            f'font-weight="600" fill="#334155">IS vs OOS Sharpe</text>'
            f'</svg>'
        )
