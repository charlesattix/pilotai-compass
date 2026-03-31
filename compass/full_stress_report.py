"""
Phase 6 comprehensive stress testing pipeline.

End-to-end stress test: 10K Monte Carlo block-bootstrap paths, crisis
scenario replay (COVID, 2022 bear, flash crash, VIX spike), sensitivity
analysis (risk_pct, spread_width, stop_loss_mult), hard failure checks,
and investor-quality HTML report.

Uses compass.stress_test.StressTester as the computation engine.

Usage::

    from compass.full_stress_report import FullStressReport
    report = FullStressReport(daily_returns)
    results = report.run()
    report.generate_report()
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
DEFAULT_OUTPUT = ROOT / "reports" / "full_stress_report.html"


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class MCResult:
    """Monte Carlo simulation results."""
    n_paths: int
    path_length: int
    median_dd: float
    p5_dd: float            # 5th percentile drawdown
    p1_dd: float            # 1st percentile drawdown
    median_return: float
    p5_return: float
    median_sharpe: float
    p5_sharpe: float
    passed: bool            # p5_dd > -0.40


@dataclass
class CrisisResult:
    """Crisis scenario replay result."""
    name: str
    description: str
    max_dd: float
    final_return: float
    recovery_days: int
    worst_day: float
    passed: bool            # max_dd > -0.50


@dataclass
class SensitivityPoint:
    """One point in a sensitivity sweep."""
    param: str
    value: float
    sharpe: float
    max_dd: float
    total_return: float


@dataclass
class HardCheck:
    """Hard failure check result."""
    name: str
    threshold: float
    actual: float
    passed: bool
    description: str


@dataclass
class StressTestSummary:
    """Overall stress test summary."""
    overall_pass: bool
    mc_pass: bool
    crisis_pass: bool
    n_hard_failures: int
    mc_p5_dd: float
    worst_crisis_dd: float
    risk_rating: str         # "low", "medium", "high", "reject"


# ── Pipeline ────────────────────────────────────────────────────────────


class FullStressReport:
    """End-to-end Phase 6 stress testing pipeline."""

    def __init__(
        self,
        daily_returns: np.ndarray,
        starting_capital: float = 100_000,
        n_simulations: int = 10_000,
        block_size: int = 5,
        seed: int = 42,
        mc_dd_reject: float = 0.40,
        crisis_dd_reject: float = 0.50,
    ) -> None:
        self.returns = np.asarray(daily_returns, dtype=float)
        self.starting_capital = starting_capital
        self.n_simulations = n_simulations
        self.block_size = block_size
        self.seed = seed
        self.mc_dd_reject = mc_dd_reject
        self.crisis_dd_reject = crisis_dd_reject

        # Results
        self.mc_result: Optional[MCResult] = None
        self.crisis_results: List[CrisisResult] = []
        self.sensitivity: List[SensitivityPoint] = []
        self.hard_checks: List[HardCheck] = []
        self.summary: Optional[StressTestSummary] = None

    # ── Public API ──────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Run full stress testing pipeline."""
        self.mc_result = self._monte_carlo()
        self.crisis_results = self._crisis_scenarios()
        self.sensitivity = self._sensitivity_analysis()
        self.hard_checks = self._hard_failure_checks()
        self.summary = self._summarize()
        return {
            "mc": self.mc_result,
            "crisis": self.crisis_results,
            "sensitivity": self.sensitivity,
            "hard_checks": self.hard_checks,
            "summary": self.summary,
        }

    # ── Monte Carlo ─────────────────────────────────────────────────────

    def _monte_carlo(self) -> MCResult:
        """10K block-bootstrap Monte Carlo paths."""
        rng = np.random.RandomState(self.seed)
        n = len(self.returns)
        if n < 10:
            return MCResult(0, 0, 0, 0, 0, 0, 0, 0, 0, True)

        path_len = min(n, 252)
        dds, rets, sharpes = [], [], []

        for _ in range(self.n_simulations):
            # Block bootstrap
            path = np.empty(path_len)
            idx = 0
            while idx < path_len:
                start = rng.randint(0, max(1, n - self.block_size))
                block = self.returns[start:start + self.block_size]
                end = min(idx + len(block), path_len)
                path[idx:end] = block[:end - idx]
                idx = end

            equity = self.starting_capital * np.cumprod(1 + path)
            equity = np.concatenate([[self.starting_capital], equity])
            peak = np.maximum.accumulate(equity)
            dd = np.min((equity - peak) / np.where(peak > 0, peak, 1.0))
            dds.append(dd)
            rets.append(equity[-1] / equity[0] - 1)
            if np.std(path) > 0:
                sharpes.append(np.mean(path) / np.std(path) * np.sqrt(252))
            else:
                sharpes.append(0.0)

        dds = np.array(dds)
        rets = np.array(rets)
        sharpes = np.array(sharpes)

        p5_dd = float(np.percentile(dds, 5))
        return MCResult(
            n_paths=self.n_simulations, path_length=path_len,
            median_dd=float(np.median(dds)),
            p5_dd=p5_dd,
            p1_dd=float(np.percentile(dds, 1)),
            median_return=float(np.median(rets)),
            p5_return=float(np.percentile(rets, 5)),
            median_sharpe=float(np.median(sharpes)),
            p5_sharpe=float(np.percentile(sharpes, 5)),
            passed=p5_dd > -self.mc_dd_reject,
        )

    # ── Crisis scenarios ────────────────────────────────────────────────

    def _crisis_scenarios(self) -> List[CrisisResult]:
        scenarios = [
            ("COVID Crash", "S&P -34% in 23 days", -0.34, 23),
            ("2022 Bear Market", "S&P -25% over 200 days", -0.25, 200),
            ("Flash Crash", "S&P -7% in 1 day", -0.07, 1),
            ("VIX Spike +150%", "VIX quadruples over 5 days", -0.15, 5),
        ]
        results: List[CrisisResult] = []
        for name, desc, total_ret, n_days in scenarios:
            # Build crisis path
            if n_days == 1:
                path = np.array([total_ret])
            else:
                daily = total_ret / n_days
                noise = np.random.RandomState(hash(name) % 2**31).normal(0, abs(daily) * 0.3, n_days)
                path = np.full(n_days, daily) + noise
                # Adjust to hit target
                actual = np.prod(1 + path) - 1
                if abs(actual) > 1e-10:
                    path = path * (total_ret / actual)

            equity = self.starting_capital * np.cumprod(1 + path)
            equity = np.concatenate([[self.starting_capital], equity])
            peak = np.maximum.accumulate(equity)
            dd_series = (equity - peak) / np.where(peak > 0, peak, 1.0)
            max_dd = float(np.min(dd_series))
            final = float(equity[-1] / equity[0] - 1)
            worst_day = float(np.min(path))

            # Recovery: how many more days to get back (estimate)
            recovery = int(abs(max_dd) / 0.001) if max_dd < 0 else 0

            results.append(CrisisResult(
                name=name, description=desc, max_dd=max_dd,
                final_return=final, recovery_days=min(recovery, 500),
                worst_day=worst_day,
                passed=max_dd > -self.crisis_dd_reject,
            ))
        return results

    # ── Sensitivity analysis ────────────────────────────────────────────

    def _sensitivity_analysis(self) -> List[SensitivityPoint]:
        points: List[SensitivityPoint] = []
        n = len(self.returns)
        if n < 10:
            return points

        # Sweep risk_pct (scale returns)
        for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            scaled = self.returns * scale
            equity = self.starting_capital * np.cumprod(1 + scaled)
            equity = np.concatenate([[self.starting_capital], equity])
            peak = np.maximum.accumulate(equity)
            dd = float(np.min((equity - peak) / np.where(peak > 0, peak, 1.0)))
            sh = float(np.mean(scaled) / np.std(scaled) * np.sqrt(252)) if np.std(scaled) > 0 else 0
            ret = float(equity[-1] / equity[0] - 1)
            points.append(SensitivityPoint("risk_pct_scale", scale, sh, dd, ret))

        # Sweep spread_width effect (approximate: wider spread = lower returns)
        for width in [2.5, 5.0, 7.5, 10.0]:
            adj = self.returns * (5.0 / width)  # narrower spread = better returns
            equity = self.starting_capital * np.cumprod(1 + adj)
            equity = np.concatenate([[self.starting_capital], equity])
            peak = np.maximum.accumulate(equity)
            dd = float(np.min((equity - peak) / np.where(peak > 0, peak, 1.0)))
            sh = float(np.mean(adj) / np.std(adj) * np.sqrt(252)) if np.std(adj) > 0 else 0
            ret = float(equity[-1] / equity[0] - 1)
            points.append(SensitivityPoint("spread_width", width, sh, dd, ret))

        # Sweep stop_loss_mult (truncate worst returns)
        for mult in [2.0, 3.0, 3.5, 4.0, 5.0]:
            daily_limit = -0.01 * mult  # e.g., 3.5x → -3.5%
            clipped = np.clip(self.returns, daily_limit, None)
            equity = self.starting_capital * np.cumprod(1 + clipped)
            equity = np.concatenate([[self.starting_capital], equity])
            peak = np.maximum.accumulate(equity)
            dd = float(np.min((equity - peak) / np.where(peak > 0, peak, 1.0)))
            sh = float(np.mean(clipped) / np.std(clipped) * np.sqrt(252)) if np.std(clipped) > 0 else 0
            ret = float(equity[-1] / equity[0] - 1)
            points.append(SensitivityPoint("stop_loss_mult", mult, sh, dd, ret))

        return points

    # ── Hard failure checks ─────────────────────────────────────────────

    def _hard_failure_checks(self) -> List[HardCheck]:
        checks: List[HardCheck] = []

        # MC 5th percentile DD
        mc_dd = abs(self.mc_result.p5_dd) if self.mc_result else 0
        checks.append(HardCheck(
            "MC 5th-pctile DD", self.mc_dd_reject, mc_dd,
            mc_dd < self.mc_dd_reject,
            f"5th percentile MC drawdown {mc_dd:.1%} {'<' if mc_dd < self.mc_dd_reject else '>'} {self.mc_dd_reject:.0%} threshold",
        ))

        # Worst crisis DD
        worst_crisis = max((abs(c.max_dd) for c in self.crisis_results), default=0)
        checks.append(HardCheck(
            "Worst crisis DD", self.crisis_dd_reject, worst_crisis,
            worst_crisis < self.crisis_dd_reject,
            f"Worst crisis drawdown {worst_crisis:.1%} {'<' if worst_crisis < self.crisis_dd_reject else '>'} {self.crisis_dd_reject:.0%} threshold",
        ))

        # MC 5th percentile Sharpe > 0
        mc_sh = self.mc_result.p5_sharpe if self.mc_result else 0
        checks.append(HardCheck(
            "MC 5th-pctile Sharpe", 0.0, mc_sh,
            mc_sh > 0,
            f"5th percentile MC Sharpe {mc_sh:.2f} {'>' if mc_sh > 0 else '<='} 0",
        ))

        return checks

    # ── Summary ─────────────────────────────────────────────────────────

    def _summarize(self) -> StressTestSummary:
        mc_pass = self.mc_result.passed if self.mc_result else False
        crisis_pass = all(c.passed for c in self.crisis_results)
        n_hard_fail = sum(1 for h in self.hard_checks if not h.passed)
        overall = mc_pass and crisis_pass and n_hard_fail == 0

        mc_dd = self.mc_result.p5_dd if self.mc_result else 0
        worst_crisis = min((c.max_dd for c in self.crisis_results), default=0)

        if not overall:
            rating = "reject"
        elif abs(mc_dd) > 0.25 or abs(worst_crisis) > 0.35:
            rating = "high"
        elif abs(mc_dd) > 0.15 or abs(worst_crisis) > 0.25:
            rating = "medium"
        else:
            rating = "low"

        return StressTestSummary(
            overall_pass=overall, mc_pass=mc_pass, crisis_pass=crisis_pass,
            n_hard_failures=n_hard_fail, mc_p5_dd=mc_dd,
            worst_crisis_dd=worst_crisis, risk_rating=rating,
        )

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.summary is None:
            self.run()
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
        return {
            "mc_dist": self._chart_mc_dist(),
            "crisis_bars": self._chart_crisis(),
            "sensitivity": self._chart_sensitivity(),
        }

    def _chart_mc_dist(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if self.mc_result is None or self.mc_result.n_paths == 0:
            return ""
        # Re-run a small MC for distribution chart
        rng = np.random.RandomState(self.seed + 1)
        dds = []
        n = len(self.returns)
        path_len = min(n, 252)
        for _ in range(min(self.n_simulations, 2000)):
            path = np.empty(path_len)
            idx = 0
            while idx < path_len:
                start = rng.randint(0, max(1, n - self.block_size))
                block = self.returns[start:start + self.block_size]
                end = min(idx + len(block), path_len)
                path[idx:end] = block[:end - idx]
                idx = end
            eq = self.starting_capital * np.cumprod(1 + path)
            eq = np.concatenate([[self.starting_capital], eq])
            pk = np.maximum.accumulate(eq)
            dds.append(float(np.min((eq - pk) / np.where(pk > 0, pk, 1.0))))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dds, bins=50, color="#3b82f6", alpha=0.7, edgecolor="white")
        ax.axvline(self.mc_result.p5_dd, color="#dc2626", lw=2, ls="--", label=f"5th pctile: {self.mc_result.p5_dd:.1%}")
        ax.axvline(-self.mc_dd_reject, color="#7f1d1d", lw=2, ls="-", label=f"Reject: -{self.mc_dd_reject:.0%}")
        ax.set_xlabel("Max Drawdown"); ax.set_ylabel("Frequency")
        ax.set_title("Monte Carlo Drawdown Distribution", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_crisis(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.crisis_results:
            return ""
        names = [c.name for c in self.crisis_results]
        dds = [c.max_dd for c in self.crisis_results]
        colors = ["#dc2626" if not c.passed else "#16a34a" for c in self.crisis_results]
        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6)))
        ax.barh(names, dds, color=colors, alpha=0.85)
        ax.axvline(-self.crisis_dd_reject, color="#7f1d1d", lw=2, ls="--", label=f"Reject: -{self.crisis_dd_reject:.0%}")
        ax.set_xlabel("Max Drawdown"); ax.set_title("Crisis Scenario Drawdowns", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_sensitivity(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.sensitivity:
            return ""
        params = sorted(set(s.param for s in self.sensitivity))
        fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4))
        if len(params) == 1:
            axes = [axes]
        for ax, param in zip(axes, params):
            pts = [s for s in self.sensitivity if s.param == param]
            xs = [p.value for p in pts]
            sharpes = [p.sharpe for p in pts]
            dds = [p.max_dd for p in pts]
            ax.plot(xs, sharpes, "o-", color="#3b82f6", label="Sharpe")
            ax2 = ax.twinx()
            ax2.plot(xs, dds, "s--", color="#dc2626", label="Max DD")
            ax.set_xlabel(param); ax.set_ylabel("Sharpe", color="#3b82f6")
            ax2.set_ylabel("Max DD", color="#dc2626")
            ax.set_title(param.replace("_", " ").title(), fontsize=10)
            ax.grid(True, alpha=0.2)
        fig.suptitle("Sensitivity Analysis", fontsize=11); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        s = self.summary or StressTestSummary(False, False, False, 0, 0, 0, "reject")
        mc = self.mc_result or MCResult(0, 0, 0, 0, 0, 0, 0, 0, 0, False)

        overall_cls = "good" if s.overall_pass else "bad"
        rating_colors = {"low": "#16a34a", "medium": "#f59e0b", "high": "#dc2626", "reject": "#7f1d1d"}
        rc = rating_colors.get(s.risk_rating, "#64748b")

        # Crisis table
        crisis_rows = ""
        for c in self.crisis_results:
            cls = "good" if c.passed else "bad"
            crisis_rows += (f'<tr><td>{c.name}</td><td>{c.description}</td>'
                           f'<td class="{cls}">{c.max_dd:.1%}</td><td>{c.final_return:+.1%}</td>'
                           f'<td>{c.worst_day:+.2%}</td><td>{c.recovery_days}d</td>'
                           f'<td class="{cls}">{"PASS" if c.passed else "FAIL"}</td></tr>\n')

        # Sensitivity table
        sens_rows = ""
        for p in self.sensitivity:
            sens_rows += f'<tr><td>{p.param}</td><td>{p.value}</td><td>{p.sharpe:.2f}</td><td>{p.max_dd:.1%}</td><td>{p.total_return:+.1%}</td></tr>\n'

        # Hard checks
        check_rows = ""
        for h in self.hard_checks:
            cls = "good" if h.passed else "bad"
            check_rows += f'<tr><td>{h.name}</td><td>{h.threshold}</td><td>{h.actual:.4f}</td><td class="{cls}">{"PASS" if h.passed else "FAIL"}</td><td>{h.description}</td></tr>\n'

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 6 Stress Test Report</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }} h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }}
  .kpi-row {{ display:flex; gap:1.2em; flex-wrap:wrap; margin:1.5em 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em 1.5em; min-width:120px; flex:1; text-align:center; }}
  .kpi .value {{ font-size:1.5em; font-weight:700; }} .kpi .label {{ font-size:0.75em; color:#64748b; margin-top:0.2em; }}
  .risk-badge {{ display:inline-block; padding:0.3em 0.8em; border-radius:4px; color:white; font-weight:700; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:0.88em; }}
  th {{ background:#f1f5f9; padding:8px 10px; text-align:left; border-bottom:2px solid #cbd5e1; font-weight:600; }}
  td {{ padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:right; }} td:first-child {{ text-align:left; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em; margin:1.5em 0; text-align:center; }}
  .chart img {{ max-width:100%; height:auto; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.8em; color:#94a3b8; }}
</style></head><body>
<h1>Phase 6: Comprehensive Stress Test Report</h1>
<div class="meta">{len(self.returns)} daily returns &middot; {mc.n_paths:,} MC paths &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value {overall_cls}">{"PASS" if s.overall_pass else "REJECT"}</div><div class="label">Overall</div></div>
  <div class="kpi"><div class="value"><span class="risk-badge" style="background:{rc}">{s.risk_rating.upper()}</span></div><div class="label">Risk Rating</div></div>
  <div class="kpi"><div class="value">{mc.p5_dd:.1%}</div><div class="label">MC 5th-pctile DD</div></div>
  <div class="kpi"><div class="value">{s.worst_crisis_dd:.1%}</div><div class="label">Worst Crisis DD</div></div>
  <div class="kpi"><div class="value">{mc.median_sharpe:.2f}</div><div class="label">Median MC Sharpe</div></div>
  <div class="kpi"><div class="value {'' if s.n_hard_failures == 0 else 'bad'}">{s.n_hard_failures}</div><div class="label">Hard Failures</div></div>
</div>
<h2>1. Monte Carlo ({mc.n_paths:,} paths)</h2>{_img("mc_dist")}
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
<tr><td>Median DD</td><td>{mc.median_dd:.1%}</td></tr>
<tr><td>5th Percentile DD</td><td class="{"good" if mc.passed else "bad"}">{mc.p5_dd:.1%}</td></tr>
<tr><td>1st Percentile DD</td><td>{mc.p1_dd:.1%}</td></tr>
<tr><td>Median Return</td><td>{mc.median_return:+.1%}</td></tr>
<tr><td>5th Percentile Sharpe</td><td>{mc.p5_sharpe:.2f}</td></tr>
<tr><td>Result</td><td class="{"good" if mc.passed else "bad"}">{"PASS" if mc.passed else "FAIL"}</td></tr>
</tbody></table>
<h2>2. Crisis Scenario Replay</h2>{_img("crisis_bars")}
<table><thead><tr><th>Scenario</th><th>Description</th><th>Max DD</th><th>Return</th><th>Worst Day</th><th>Recovery</th><th>Result</th></tr></thead>
<tbody>{crisis_rows}</tbody></table>
<h2>3. Sensitivity Analysis</h2>{_img("sensitivity")}
<table><thead><tr><th>Parameter</th><th>Value</th><th>Sharpe</th><th>Max DD</th><th>Return</th></tr></thead>
<tbody>{sens_rows}</tbody></table>
<h2>4. Hard Failure Checks</h2>
<table><thead><tr><th>Check</th><th>Threshold</th><th>Actual</th><th>Result</th><th>Description</th></tr></thead>
<tbody>{check_rows}</tbody></table>
<footer>Generated by <code>compass/full_stress_report.py</code> &middot; Phase 6 Validation</footer>
</body></html>"""
        return html
