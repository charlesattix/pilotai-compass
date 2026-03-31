"""
Post-trade execution analytics.

Implementation shortfall, VWAP/TWAP benchmarks, timing analysis,
cost attribution, fill rate, venue analysis, speed analysis,
rolling quality score, and improvement recommendations.

All methods work on pre-loaded fill data — no broker connections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ShortfallResult:
    total_bps: float = 0.0
    delay_cost: float = 0.0
    trading_cost: float = 0.0
    opportunity_cost: float = 0.0


@dataclass
class BenchmarkResult:
    fill_price: float = 0.0
    vwap: float = 0.0
    twap: float = 0.0
    vs_vwap_bps: float = 0.0
    vs_twap_bps: float = 0.0


@dataclass
class TimingResult:
    actual_price: float = 0.0
    optimal_price: float = 0.0
    timing_cost_bps: float = 0.0
    was_optimal: bool = False


@dataclass
class CostAttribution:
    spread_cost: float = 0.0
    impact_cost: float = 0.0
    timing_cost: float = 0.0
    opportunity_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class VenueResult:
    venue: str
    n_fills: int = 0
    avg_price_improvement: float = 0.0
    fill_rate: float = 0.0
    avg_speed_ms: float = 0.0


@dataclass
class QualityScore:
    date: Optional[datetime] = None
    score: float = 0.0          # 0-100
    shortfall_component: float = 0.0
    timing_component: float = 0.0
    fill_rate_component: float = 0.0


@dataclass
class ExecutionReport:
    shortfall: ShortfallResult
    benchmark: BenchmarkResult
    timing: TimingResult
    cost_attr: CostAttribution
    venues: List[VenueResult]
    quality: QualityScore
    recommendations: List[str]


class ExecutionAnalytics:
    """Post-trade execution analytics engine."""

    # ------------------------------------------------------------------
    # Implementation shortfall
    # ------------------------------------------------------------------

    @staticmethod
    def implementation_shortfall(
        decision_price: float, arrival_price: float,
        fill_price: float, end_price: float,
        filled_qty: float, ordered_qty: float,
        side: str = "buy",
    ) -> ShortfallResult:
        sign = 1.0 if side == "buy" else -1.0
        unfilled = ordered_qty - filled_qty
        delay = sign * (arrival_price - decision_price) * filled_qty
        trading = sign * (fill_price - arrival_price) * filled_qty
        opportunity = sign * (end_price - decision_price) * unfilled
        total = delay + trading + opportunity
        ref = abs(decision_price * ordered_qty)
        bps = total / ref * 10000 if ref > 0 else 0.0
        return ShortfallResult(bps, delay, trading, opportunity)

    @staticmethod
    def shortfall_from_df(fills: pd.DataFrame) -> ShortfallResult:
        """Compute from DataFrame with standard columns."""
        req = {"decision_price", "arrival_price", "fill_price", "end_price",
               "fill_qty", "ordered_qty", "side"}
        if not req.issubset(fills.columns) or fills.empty:
            return ShortfallResult()
        total_filled = float(fills["fill_qty"].sum())
        avg_fill = float((fills["fill_price"] * fills["fill_qty"]).sum() / total_filled) if total_filled > 0 else 0.0
        return ExecutionAnalytics.implementation_shortfall(
            float(fills["decision_price"].iloc[0]),
            float(fills["arrival_price"].iloc[0]),
            avg_fill, float(fills["end_price"].iloc[-1]),
            total_filled, float(fills["ordered_qty"].iloc[0]),
            str(fills["side"].iloc[0]),
        )

    # ------------------------------------------------------------------
    # VWAP / TWAP benchmarks
    # ------------------------------------------------------------------

    @staticmethod
    def benchmark(
        fill_price: float, market_prices: pd.Series,
        market_volumes: pd.Series, side: str = "buy",
    ) -> BenchmarkResult:
        aligned = pd.DataFrame({"p": market_prices, "v": market_volumes}).dropna()
        if aligned.empty:
            return BenchmarkResult(fill_price)
        vwap = float((aligned["p"] * aligned["v"]).sum() / aligned["v"].sum())
        twap = float(aligned["p"].mean())
        sign = 1.0 if side == "buy" else -1.0
        vs_vwap = sign * (fill_price - vwap) / abs(vwap) * 10000 if vwap != 0 else 0.0
        vs_twap = sign * (fill_price - twap) / abs(twap) * 10000 if twap != 0 else 0.0
        return BenchmarkResult(fill_price, vwap, twap, vs_vwap, vs_twap)

    # ------------------------------------------------------------------
    # Timing analysis
    # ------------------------------------------------------------------

    @staticmethod
    def timing_analysis(
        fill_price: float, market_prices: pd.Series,
        side: str = "buy",
    ) -> TimingResult:
        if market_prices.empty:
            return TimingResult(fill_price)
        optimal = float(market_prices.min()) if side == "buy" else float(market_prices.max())
        sign = 1.0 if side == "buy" else -1.0
        cost = sign * (fill_price - optimal) / abs(optimal) * 10000 if optimal != 0 else 0.0
        return TimingResult(fill_price, optimal, cost, abs(cost) < 5)

    # ------------------------------------------------------------------
    # Cost attribution
    # ------------------------------------------------------------------

    @staticmethod
    def cost_attribution(
        shortfall: ShortfallResult,
        spread_bps: float = 0.0,
    ) -> CostAttribution:
        impact = shortfall.trading_cost
        timing = shortfall.delay_cost
        opp = shortfall.opportunity_cost
        spread = spread_bps
        total = spread + abs(impact) + abs(timing) + abs(opp)
        return CostAttribution(spread, abs(impact), abs(timing), abs(opp), total)

    # ------------------------------------------------------------------
    # Venue analysis
    # ------------------------------------------------------------------

    @staticmethod
    def venue_analysis(fills: pd.DataFrame) -> List[VenueResult]:
        req = {"venue", "fill_qty", "midprice", "fill_price"}
        if not req.issubset(fills.columns) or fills.empty:
            return []
        results: List[VenueResult] = []
        for venue, grp in fills.groupby("venue"):
            pi = float((grp["midprice"] - grp["fill_price"]).abs().mean())
            speed = float(grp["fill_time_ms"].mean()) if "fill_time_ms" in grp.columns else 0.0
            fr = 1.0
            if "orders_routed" in grp.columns:
                routed = float(grp["orders_routed"].sum())
                fr = len(grp) / routed if routed > 0 else 1.0
            results.append(VenueResult(str(venue), len(grp), pi, fr, speed))
        results.sort(key=lambda v: v.avg_price_improvement, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Rolling quality score
    # ------------------------------------------------------------------

    @staticmethod
    def quality_score(
        shortfall_bps: float, timing_bps: float,
        fill_rate: float = 1.0,
    ) -> QualityScore:
        sf_score = max(0, 100 - abs(shortfall_bps) * 5)
        tm_score = max(0, 100 - abs(timing_bps) * 3)
        fr_score = fill_rate * 100
        composite = sf_score * 0.4 + tm_score * 0.3 + fr_score * 0.3
        return QualityScore(
            score=min(100, composite),
            shortfall_component=sf_score,
            timing_component=tm_score,
            fill_rate_component=fr_score,
        )

    @staticmethod
    def rolling_quality(
        fills: pd.DataFrame, window: int = 20,
    ) -> List[QualityScore]:
        """Rolling quality score over fills."""
        if fills.empty or "shortfall_bps" not in fills.columns:
            return []
        results: List[QualityScore] = []
        for end in range(window, len(fills) + 1):
            chunk = fills.iloc[end - window:end]
            sf = float(chunk["shortfall_bps"].mean()) if "shortfall_bps" in chunk else 0
            tm = float(chunk["timing_bps"].mean()) if "timing_bps" in chunk else 0
            fr = float(chunk["fill_rate"].mean()) if "fill_rate" in chunk else 1.0
            qs = ExecutionAnalytics.quality_score(sf, tm, fr)
            results.append(qs)
        return results

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def recommend(
        shortfall: ShortfallResult, timing: TimingResult,
        venues: List[VenueResult],
    ) -> List[str]:
        recs: List[str] = []
        if abs(shortfall.total_bps) > 10:
            recs.append(f"High shortfall ({shortfall.total_bps:.1f}bps) — consider faster execution")
        if not timing.was_optimal:
            recs.append(f"Timing cost {timing.timing_cost_bps:.1f}bps — review entry timing")
        if venues:
            best = venues[0]
            worst = venues[-1]
            if best.avg_price_improvement > worst.avg_price_improvement * 2:
                recs.append(f"Route more to {best.venue} (best PI: {best.avg_price_improvement:.4f})")
        return recs

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        decision_price: float, arrival_price: float,
        fill_price: float, end_price: float,
        filled_qty: float, ordered_qty: float,
        side: str = "buy",
        market_prices: Optional[pd.Series] = None,
        market_volumes: Optional[pd.Series] = None,
        fills_df: Optional[pd.DataFrame] = None,
        spread_bps: float = 2.0,
    ) -> ExecutionReport:
        sf = self.implementation_shortfall(
            decision_price, arrival_price, fill_price, end_price,
            filled_qty, ordered_qty, side)

        bm = BenchmarkResult(fill_price)
        tm = TimingResult(fill_price)
        if market_prices is not None and market_volumes is not None:
            bm = self.benchmark(fill_price, market_prices, market_volumes, side)
            tm = self.timing_analysis(fill_price, market_prices, side)

        ca = self.cost_attribution(sf, spread_bps)
        venues = self.venue_analysis(fills_df) if fills_df is not None else []
        fr = filled_qty / ordered_qty if ordered_qty > 0 else 1.0
        qs = self.quality_score(sf.total_bps, tm.timing_cost_bps, fr)
        recs = self.recommend(sf, tm, venues)

        return ExecutionReport(sf, bm, tm, ca, venues, qs, recs)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_bar(labels, values, title, width=600, height=200, color="#2980b9"):
        if not values:
            return ""
        n = len(values)
        vmax = max(abs(v) for v in values) or 1
        pad_l = 90
        pw = width - pad_l - 20
        ph = height - 55
        bw = pw / max(n, 1) * 0.7
        gap = pw / max(n, 1)
        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="18" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">{title}</text>')
        for i in range(n):
            x = pad_l + i * gap + (gap - bw) / 2
            bh = abs(values[i]) / vmax * (ph - 30)
            y = 30 + ph - 30 - bh
            c = "#27ae60" if values[i] <= 0 else "#e74c3c"
            p.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{max(bh, 1):.0f}" fill="{c}" rx="3"/>')
            p.append(f'<text x="{x + bw / 2:.0f}" y="{height - 8:.0f}" text-anchor="middle" font-size="9" fill="#666">{labels[i]}</text>')
            p.append(f'<text x="{x + bw / 2:.0f}" y="{y - 3:.0f}" text-anchor="middle" font-size="9" fill="#333">{values[i]:.1f}</text>')
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self, report: ExecutionReport,
        output_path: str = "reports/execution_analytics.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf = report.shortfall
        ca = report.cost_attr
        cost_svg = self._svg_bar(
            ["Spread", "Impact", "Timing", "Opportunity"],
            [ca.spread_cost, ca.impact_cost, ca.timing_cost, ca.opportunity_cost],
            "Cost Attribution (bps)")
        venue_rows = [
            f"<tr><td style='text-align:left'>{v.venue}</td><td>{v.n_fills}</td>"
            f"<td>{v.avg_price_improvement:.6f}</td><td>{v.fill_rate:.1%}</td>"
            f"<td>{v.avg_speed_ms:.0f}</td></tr>"
            for v in report.venues
        ]
        rec_html = ""
        if report.recommendations:
            items = "".join(f"<li>{r}</li>" for r in report.recommendations)
            rec_html = f"<h2>Recommendations</h2><ul>{items}</ul>"

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Execution Analytics</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
</style></head><body>
<h1>Execution Analytics</h1>
<div class="summary">
<p><strong>Quality Score:</strong> {report.quality.score:.0f}/100 |
<strong>Shortfall:</strong> {sf.total_bps:.1f}bps |
<strong>vs VWAP:</strong> {report.benchmark.vs_vwap_bps:.1f}bps</p>
</div>
<h2>Cost Attribution</h2>
{cost_svg}
<table class="m"><tr><th>Shortfall</th><th>Delay</th><th>Trading</th><th>Opportunity</th></tr>
<tr><td>{sf.total_bps:.1f}bps</td><td>{sf.delay_cost:.4f}</td>
<td>{sf.trading_cost:.4f}</td><td>{sf.opportunity_cost:.4f}</td></tr></table>
{'<h2>Venue Analysis</h2><table><tr><th style="text-align:left">Venue</th><th>Fills</th><th>PI</th><th>Fill Rate</th><th>Speed</th></tr>' + ''.join(venue_rows) + '</table>' if venue_rows else ''}
{rec_html}
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
