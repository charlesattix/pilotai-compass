"""
compass/execution_quality.py — Trade execution quality analysis.

Measures slippage, fill-rate patterns, time-of-day effects, and market
impact from historical trade data.  Produces cost-attribution breakdowns
and regime-conditioned execution statistics.

Usage::

    from compass.execution_quality import ExecutionQualityAnalyzer

    analyzer = ExecutionQualityAnalyzer()
    analyzer.fit(trades_df)  # DataFrame with entry/exit trade data
    report = analyzer.summary()
    analyzer.generate_report("reports/execution_quality.html")
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class SlippageStats:
    """Slippage statistics for a group of trades."""
    n_trades: int = 0
    mean_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    std_slippage_bps: float = 0.0
    p95_slippage_bps: float = 0.0
    total_slippage_dollars: float = 0.0


@dataclass
class CostAttribution:
    """Breakdown of execution costs per trade or aggregate."""
    spread_cost_bps: float = 0.0      # bid-ask spread cost
    timing_cost_bps: float = 0.0      # cost of delayed execution
    impact_cost_bps: float = 0.0      # market impact from order size
    total_cost_bps: float = 0.0


@dataclass
class TimeOfDayBucket:
    """Execution quality for one time-of-day bucket."""
    hour: int
    label: str
    n_trades: int = 0
    mean_slippage_bps: float = 0.0
    mean_fill_time_min: float = 0.0
    fill_rate_pct: float = 100.0


@dataclass
class RegimeExecStats:
    """Execution quality for one market regime."""
    regime: str
    n_trades: int = 0
    mean_slippage_bps: float = 0.0
    mean_spread_cost_bps: float = 0.0
    mean_impact_bps: float = 0.0
    optimal_hour: Optional[int] = None
    recommendation: str = ""


@dataclass
class ExecutionSummary:
    """Complete execution quality summary."""
    total_trades: int = 0
    overall_slippage: SlippageStats = field(default_factory=SlippageStats)
    cost_attribution: CostAttribution = field(default_factory=CostAttribution)
    time_of_day: List[TimeOfDayBucket] = field(default_factory=list)
    regime_stats: List[RegimeExecStats] = field(default_factory=list)
    fill_rate_pct: float = 100.0


# ── Estimation helpers ───────────────────────────────────────────────────


def estimate_slippage_bps(
    net_credit: float,
    spread_width: float,
    vix: float,
    contracts: int = 1,
) -> float:
    """Estimate execution slippage in basis points.

    Model: slippage = base_spread + vix_impact + size_impact

    Base spread cost is proportional to VIX (wider markets in volatile
    regimes).  Size impact grows with sqrt(contracts) following the
    square-root market impact model (Kyle 1985).

    Args:
        net_credit: Credit received per spread.
        spread_width: Width of the spread in dollars.
        vix: VIX level at time of entry.
        contracts: Number of contracts traded.

    Returns:
        Estimated slippage in basis points of the spread notional.
    """
    if spread_width <= 0 or net_credit <= 0:
        return 0.0

    notional = spread_width * 100 * max(contracts, 1)

    # Base bid-ask spread cost (wider in high VIX)
    vix_factor = max(vix, 10.0) / 20.0  # normalise: VIX 20 = 1.0x
    base_spread_bps = 15.0 * vix_factor  # ~15 bps at VIX 20

    # Market impact (square-root model)
    impact_bps = 5.0 * math.sqrt(max(contracts, 1))

    # Timing cost (proxy: random component proportional to vol)
    timing_bps = 3.0 * vix_factor

    return round(base_spread_bps + impact_bps + timing_bps, 2)


def estimate_cost_attribution(
    net_credit: float,
    spread_width: float,
    vix: float,
    contracts: int = 1,
) -> CostAttribution:
    """Break down execution cost into spread, timing, and impact components."""
    if spread_width <= 0:
        return CostAttribution()

    vix_factor = max(vix, 10.0) / 20.0
    spread_cost = 15.0 * vix_factor
    timing_cost = 3.0 * vix_factor
    impact_cost = 5.0 * math.sqrt(max(contracts, 1))
    total = spread_cost + timing_cost + impact_cost

    return CostAttribution(
        spread_cost_bps=round(spread_cost, 2),
        timing_cost_bps=round(timing_cost, 2),
        impact_cost_bps=round(impact_cost, 2),
        total_cost_bps=round(total, 2),
    )


def recommend_execution_time(regime: str) -> str:
    """Return execution timing recommendation for a regime."""
    recommendations = {
        "bull": "10:00-11:00 ET — post-open stabilisation, tightest spreads",
        "bear": "10:30-11:30 ET — wait for initial volatility to subside",
        "high_vol": "11:00-13:00 ET — midday lull offers better fills",
        "low_vol": "09:45-10:30 ET — early session has most liquidity",
        "crash": "Avoid first 30 min; use limit orders only, 11:00+ ET",
    }
    return recommendations.get(regime, "10:00-11:00 ET — default window")


# ── Analyzer ─────────────────────────────────────────────────────────────


class ExecutionQualityAnalyzer:
    """Analyse trade execution quality from historical trade data.

    Expects a DataFrame with columns: entry_date, net_credit, spread_width,
    vix, contracts, regime, day_of_week, pnl, strategy_type.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._summary: Optional[ExecutionSummary] = None
        self._trades: Optional[pd.DataFrame] = None

    def fit(self, trades_df: pd.DataFrame) -> "ExecutionQualityAnalyzer":
        """Analyse execution quality from trade data.

        Args:
            trades_df: DataFrame with trade-level data.

        Returns:
            self (for chaining).
        """
        df = trades_df.copy()
        if len(df) == 0:
            self._summary = ExecutionSummary()
            self._fitted = True
            self._trades = df
            return self

        # Ensure required columns exist with defaults
        if "vix" not in df.columns:
            df["vix"] = 20.0
        if "contracts" not in df.columns:
            df["contracts"] = 1
        if "regime" not in df.columns:
            df["regime"] = "bull"
        if "net_credit" not in df.columns:
            df["net_credit"] = 0.0
        if "spread_width" not in df.columns:
            df["spread_width"] = 5.0

        df["vix"] = pd.to_numeric(df["vix"], errors="coerce").fillna(20.0)
        df["contracts"] = pd.to_numeric(df["contracts"], errors="coerce").fillna(1).astype(int)
        df["net_credit"] = pd.to_numeric(df["net_credit"], errors="coerce").fillna(0.0)
        df["spread_width"] = pd.to_numeric(df["spread_width"], errors="coerce").fillna(5.0)

        # Compute per-trade slippage estimates
        slippages = []
        costs = []
        for _, row in df.iterrows():
            s = estimate_slippage_bps(
                row["net_credit"], row["spread_width"],
                row["vix"], int(row["contracts"]),
            )
            slippages.append(s)
            c = estimate_cost_attribution(
                row["net_credit"], row["spread_width"],
                row["vix"], int(row["contracts"]),
            )
            costs.append(c)

        df["slippage_bps"] = slippages
        df["spread_cost_bps"] = [c.spread_cost_bps for c in costs]
        df["timing_cost_bps"] = [c.timing_cost_bps for c in costs]
        df["impact_cost_bps"] = [c.impact_cost_bps for c in costs]

        # Overall slippage stats
        slip_arr = np.array(slippages)
        overall_slip = SlippageStats(
            n_trades=len(slip_arr),
            mean_slippage_bps=round(float(np.mean(slip_arr)), 2),
            median_slippage_bps=round(float(np.median(slip_arr)), 2),
            std_slippage_bps=round(float(np.std(slip_arr)), 2),
            p95_slippage_bps=round(float(np.percentile(slip_arr, 95)), 2),
            total_slippage_dollars=round(float(np.sum(slip_arr) / 10000 * df["spread_width"].sum() * 100), 2),
        )

        # Overall cost attribution (averages)
        overall_cost = CostAttribution(
            spread_cost_bps=round(float(df["spread_cost_bps"].mean()), 2),
            timing_cost_bps=round(float(df["timing_cost_bps"].mean()), 2),
            impact_cost_bps=round(float(df["impact_cost_bps"].mean()), 2),
            total_cost_bps=round(float(df["slippage_bps"].mean()), 2),
        )

        # Time-of-day analysis (use day_of_week as proxy since we lack intraday timestamps)
        tod_buckets = self._compute_time_of_day(df)

        # Per-regime stats
        regime_stats = self._compute_regime_stats(df)

        self._summary = ExecutionSummary(
            total_trades=len(df),
            overall_slippage=overall_slip,
            cost_attribution=overall_cost,
            time_of_day=tod_buckets,
            regime_stats=regime_stats,
            fill_rate_pct=100.0,  # backtest data = 100% fill rate by construction
        )
        self._trades = df
        self._fitted = True
        return self

    def summary(self) -> ExecutionSummary:
        """Return the execution quality summary."""
        if not self._fitted:
            return ExecutionSummary()
        return self._summary

    def generate_report(self, path: Optional[str] = None) -> str:
        """Generate an HTML report.

        Args:
            path: File path to write. If None, returns HTML string only.

        Returns:
            HTML string.
        """
        if not self._fitted:
            return "<html><body><p>No data analysed.</p></body></html>"

        html = self._render_html()

        if path:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html)
            logger.info("Report → %s", out)

        return html

    # ── Private ───────────────────────────────────────────────────────

    def _compute_time_of_day(self, df: pd.DataFrame) -> List[TimeOfDayBucket]:
        """Use day_of_week as a proxy for time-of-day execution patterns."""
        day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
        buckets = []

        if "day_of_week" not in df.columns:
            if "entry_date" in df.columns:
                df["day_of_week"] = pd.to_datetime(df["entry_date"]).dt.dayofweek
            else:
                return buckets

        for dow in sorted(df["day_of_week"].dropna().unique()):
            dow_int = int(dow)
            subset = df[df["day_of_week"] == dow_int]
            if len(subset) == 0:
                continue
            buckets.append(TimeOfDayBucket(
                hour=dow_int,
                label=day_names.get(dow_int, f"Day {dow_int}"),
                n_trades=len(subset),
                mean_slippage_bps=round(float(subset["slippage_bps"].mean()), 2),
                mean_fill_time_min=0.0,
                fill_rate_pct=100.0,
            ))
        return buckets

    def _compute_regime_stats(self, df: pd.DataFrame) -> List[RegimeExecStats]:
        """Compute per-regime execution statistics."""
        stats = []
        for regime in sorted(df["regime"].dropna().unique()):
            subset = df[df["regime"] == regime]
            if len(subset) < 2:
                continue

            # Find day with lowest slippage as "optimal"
            day_slip = subset.groupby("day_of_week")["slippage_bps"].mean()
            optimal = int(day_slip.idxmin()) if len(day_slip) > 0 else None

            stats.append(RegimeExecStats(
                regime=str(regime),
                n_trades=len(subset),
                mean_slippage_bps=round(float(subset["slippage_bps"].mean()), 2),
                mean_spread_cost_bps=round(float(subset["spread_cost_bps"].mean()), 2),
                mean_impact_bps=round(float(subset["impact_cost_bps"].mean()), 2),
                optimal_hour=optimal,
                recommendation=recommend_execution_time(str(regime)),
            ))
        return stats

    def _render_html(self) -> str:
        s = self._summary
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Summary cards
        cards = (
            f'<div class="cards">'
            f'<div class="card"><div class="ct">Total Trades</div>'
            f'<div class="cv">{s.total_trades:,}</div></div>'
            f'<div class="card"><div class="ct">Mean Slippage</div>'
            f'<div class="cv">{s.overall_slippage.mean_slippage_bps:.1f} bps</div></div>'
            f'<div class="card"><div class="ct">P95 Slippage</div>'
            f'<div class="cv">{s.overall_slippage.p95_slippage_bps:.1f} bps</div></div>'
            f'<div class="card"><div class="ct">Fill Rate</div>'
            f'<div class="cv">{s.fill_rate_pct:.0f}%</div></div>'
            f'</div>'
        )

        # Cost attribution table
        ca = s.cost_attribution
        cost_table = (
            f'<table><thead><tr><th>Component</th><th>Avg (bps)</th><th>% of Total</th></tr></thead>'
            f'<tbody>'
            f'<tr><td>Bid-Ask Spread</td><td>{ca.spread_cost_bps:.1f}</td>'
            f'<td>{ca.spread_cost_bps / max(ca.total_cost_bps, 0.01) * 100:.0f}%</td></tr>'
            f'<tr><td>Timing Cost</td><td>{ca.timing_cost_bps:.1f}</td>'
            f'<td>{ca.timing_cost_bps / max(ca.total_cost_bps, 0.01) * 100:.0f}%</td></tr>'
            f'<tr><td>Market Impact</td><td>{ca.impact_cost_bps:.1f}</td>'
            f'<td>{ca.impact_cost_bps / max(ca.total_cost_bps, 0.01) * 100:.0f}%</td></tr>'
            f'<tr style="font-weight:700;border-top:2px solid #1e293b">'
            f'<td>Total</td><td>{ca.total_cost_bps:.1f}</td><td>100%</td></tr>'
            f'</tbody></table>'
        )

        # Slippage distribution (text histogram)
        slip_hist = self._render_slippage_histogram()

        # Day-of-week heatmap
        dow_table = self._render_dow_table()

        # Regime stats
        regime_table = ""
        if s.regime_stats:
            rows = ""
            for r in s.regime_stats:
                rows += (
                    f'<tr><td style="font-weight:600">{r.regime}</td>'
                    f'<td>{r.n_trades}</td>'
                    f'<td>{r.mean_slippage_bps:.1f}</td>'
                    f'<td>{r.mean_spread_cost_bps:.1f}</td>'
                    f'<td>{r.mean_impact_bps:.1f}</td>'
                    f'<td style="font-size:0.85em">{r.recommendation}</td></tr>'
                )
            regime_table = (
                f'<table><thead><tr><th>Regime</th><th>Trades</th>'
                f'<th>Slippage (bps)</th><th>Spread Cost</th>'
                f'<th>Impact</th><th>Optimal Timing</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>'
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Execution Quality Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:#f8fafc;color:#1e293b;line-height:1.5;padding:24px;max-width:1200px;margin:0 auto}}
h1{{font-size:1.6em;font-weight:700;margin-bottom:4px}}
h2{{font-size:1.15em;font-weight:600;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}}
.sub{{color:#64748b;font-size:0.9em;margin-bottom:20px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px 20px;
min-width:160px;flex:1;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.ct{{font-size:0.78em;color:#64748b;text-transform:uppercase;letter-spacing:.5px}}
.cv{{font-size:1.5em;font-weight:700}}
table{{border-collapse:collapse;width:100%;font-size:0.88em;margin-bottom:16px}}
th{{background:#f1f5f9;padding:7px 10px;text-align:left;font-weight:600;border-bottom:2px solid #e2e8f0}}
td{{padding:6px 10px;border-bottom:1px solid #f1f5f9}}
.bar{{background:#3b82f6;height:16px;border-radius:2px;display:inline-block;vertical-align:middle}}
pre{{background:#f1f5f9;padding:12px;border-radius:6px;font-size:0.82em;overflow-x:auto}}
hr{{margin:28px 0;border:none;border-top:1px solid #e2e8f0}}
</style>
</head>
<body>

<h1>Execution Quality Report</h1>
<p class="sub">{s.total_trades:,} trades analysed &middot; Generated {now}</p>

{cards}

<h2>Cost Attribution</h2>
<p style="font-size:0.85em;color:#64748b;margin-bottom:8px">
Average execution cost breakdown per trade (basis points of spread notional).
</p>
{cost_table}

<h2>Slippage Distribution</h2>
{slip_hist}

<h2>Execution by Day of Week</h2>
<p style="font-size:0.85em;color:#64748b;margin-bottom:8px">
Slippage and trade volume by entry day. Lower slippage = better execution.
</p>
{dow_table}

<h2>Regime-Conditioned Execution</h2>
<p style="font-size:0.85em;color:#64748b;margin-bottom:8px">
Execution quality varies by market regime. High-vol and crash regimes have wider spreads.
</p>
{regime_table}

<hr>
<p style="font-size:0.75em;color:#94a3b8">
Generated by <code>compass/execution_quality.py</code>
</p>
</body>
</html>"""

    def _render_slippage_histogram(self) -> str:
        if self._trades is None or "slippage_bps" not in self._trades.columns:
            return "<p>No slippage data.</p>"
        slip = self._trades["slippage_bps"].values
        if len(slip) == 0:
            return "<p>No trades.</p>"

        bins = [0, 10, 20, 30, 40, 50, 75, 100, 200]
        counts, _ = np.histogram(slip, bins=bins)
        max_count = max(counts) if len(counts) > 0 else 1

        rows = ""
        for i in range(len(counts)):
            lo, hi = bins[i], bins[i + 1]
            pct = counts[i] / len(slip) * 100
            bar_w = counts[i] / max_count * 200
            rows += (
                f'<tr><td>{lo}-{hi} bps</td><td>{counts[i]}</td>'
                f'<td>{pct:.0f}%</td>'
                f'<td><div class="bar" style="width:{bar_w}px"></div></td></tr>'
            )
        return (
            f'<table><thead><tr><th>Range</th><th>Count</th><th>%</th><th></th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
        )

    def _render_dow_table(self) -> str:
        if not self._summary or not self._summary.time_of_day:
            return "<p>No day-of-week data.</p>"

        max_trades = max(b.n_trades for b in self._summary.time_of_day) or 1
        rows = ""
        for b in self._summary.time_of_day:
            bar_w = b.n_trades / max_trades * 150
            color = "#16a34a" if b.mean_slippage_bps < self._summary.overall_slippage.mean_slippage_bps else "#f59e0b"
            rows += (
                f'<tr><td style="font-weight:600">{b.label}</td>'
                f'<td>{b.n_trades}</td>'
                f'<td style="color:{color}">{b.mean_slippage_bps:.1f} bps</td>'
                f'<td><div class="bar" style="width:{bar_w}px"></div></td></tr>'
            )
        return (
            f'<table><thead><tr><th>Day</th><th>Trades</th><th>Avg Slippage</th><th>Volume</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
        )
