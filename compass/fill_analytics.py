"""
Fill quality analytics engine — measures how well trades were executed.

Components:
  1. Implementation shortfall   (delay / trading / opportunity cost)
  2. Timing analysis            (optimal vs actual execution price)
  3. Venue analysis             (fill rate, avg price improvement by route)
  4. VWAP / TWAP benchmarks     (deviation from volume- / time-weighted avg)
  5. Slippage attribution       (by time-of-day, volatility bucket, order size)
  6. Execution scorecard        (per-strategy composite quality score)
  7. HTML report

All methods operate on pre-loaded fill data — no broker connections.
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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ShortfallDecomposition:
    """Implementation shortfall broken into components."""
    total_shortfall: float = 0.0
    delay_cost: float = 0.0        # decision → first fill
    trading_cost: float = 0.0      # first fill → last fill (market impact)
    opportunity_cost: float = 0.0  # unfilled portion
    total_bps: float = 0.0


@dataclass
class TimingAnalysis:
    """Execution timing quality."""
    actual_avg_price: float = 0.0
    optimal_price: float = 0.0     # best available during window
    vwap_price: float = 0.0
    twap_price: float = 0.0
    timing_cost: float = 0.0       # actual - optimal
    timing_cost_bps: float = 0.0


@dataclass
class VenueStats:
    """Fill quality by execution venue."""
    venue: str
    n_fills: int = 0
    total_volume: float = 0.0
    fill_rate: float = 0.0        # fills / orders routed
    avg_price_improvement: float = 0.0
    avg_fill_time_ms: float = 0.0


@dataclass
class BenchmarkComparison:
    """Fill price vs VWAP / TWAP benchmarks."""
    fill_price: float = 0.0
    vwap: float = 0.0
    twap: float = 0.0
    vs_vwap_bps: float = 0.0
    vs_twap_bps: float = 0.0
    side: str = "buy"              # buy or sell


@dataclass
class SlippageBucket:
    """Slippage for one attribution bucket."""
    bucket_label: str
    bucket_type: str               # "time_of_day" | "volatility" | "size"
    n_fills: int = 0
    avg_slippage_bps: float = 0.0
    total_slippage: float = 0.0
    avg_fill_size: float = 0.0


@dataclass
class ExecutionScorecard:
    """Composite execution quality for a strategy."""
    strategy: str
    n_trades: int = 0
    avg_shortfall_bps: float = 0.0
    avg_vs_vwap_bps: float = 0.0
    avg_slippage_bps: float = 0.0
    fill_rate: float = 0.0
    score: float = 0.0            # 0-100 composite


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class FillAnalytics:
    """Fill quality analytics engine.

    All methods accept DataFrames with standard column names. Required
    columns vary by method (documented in each docstring).
    """

    # ------------------------------------------------------------------
    # 1. Implementation shortfall
    # ------------------------------------------------------------------

    @staticmethod
    def implementation_shortfall(
        decision_price: float,
        arrival_price: float,
        avg_fill_price: float,
        end_price: float,
        filled_qty: float,
        ordered_qty: float,
        side: str = "buy",
    ) -> ShortfallDecomposition:
        """Perold-style implementation shortfall decomposition.

        Args:
            decision_price: Price when trading decision was made.
            arrival_price: Price when order reached market.
            avg_fill_price: Volume-weighted average fill price.
            end_price: Price at end of execution window.
            filled_qty: Quantity actually filled.
            ordered_qty: Total quantity ordered.
            side: 'buy' or 'sell'.
        """
        sign = 1.0 if side == "buy" else -1.0
        unfilled = ordered_qty - filled_qty

        delay = sign * (arrival_price - decision_price) * filled_qty
        trading = sign * (avg_fill_price - arrival_price) * filled_qty
        opportunity = sign * (end_price - decision_price) * unfilled

        total = delay + trading + opportunity
        ref = abs(decision_price * ordered_qty)
        total_bps = total / ref * 10000 if ref > 0 else 0.0

        return ShortfallDecomposition(
            total_shortfall=total,
            delay_cost=delay,
            trading_cost=trading,
            opportunity_cost=opportunity,
            total_bps=total_bps,
        )

    @staticmethod
    def shortfall_from_fills(fills: pd.DataFrame) -> ShortfallDecomposition:
        """Compute shortfall from a fills DataFrame.

        Required columns: decision_price, arrival_price, fill_price,
        end_price, fill_qty, ordered_qty, side.
        """
        required = {"decision_price", "arrival_price", "fill_price",
                     "end_price", "fill_qty", "ordered_qty", "side"}
        if not required.issubset(fills.columns) or fills.empty:
            return ShortfallDecomposition()

        total_filled = float(fills["fill_qty"].sum())
        total_ordered = float(fills["ordered_qty"].iloc[0])
        avg_fill = float((fills["fill_price"] * fills["fill_qty"]).sum() / total_filled) if total_filled > 0 else 0.0

        return FillAnalytics.implementation_shortfall(
            decision_price=float(fills["decision_price"].iloc[0]),
            arrival_price=float(fills["arrival_price"].iloc[0]),
            avg_fill_price=avg_fill,
            end_price=float(fills["end_price"].iloc[-1]),
            filled_qty=total_filled,
            ordered_qty=total_ordered,
            side=str(fills["side"].iloc[0]),
        )

    # ------------------------------------------------------------------
    # 2. Timing analysis
    # ------------------------------------------------------------------

    @staticmethod
    def timing_analysis(
        fill_prices: pd.Series,
        fill_volumes: pd.Series,
        market_prices: pd.Series,
        market_volumes: pd.Series,
        side: str = "buy",
    ) -> TimingAnalysis:
        """Compare actual execution to optimal / VWAP / TWAP.

        Args:
            fill_prices: Prices of actual fills.
            fill_volumes: Volumes of actual fills.
            market_prices: All market prices during execution window.
            market_volumes: Corresponding market volumes.
            side: 'buy' or 'sell'.
        """
        if fill_prices.empty or market_prices.empty:
            return TimingAnalysis()

        total_vol = float(fill_volumes.sum())
        actual_avg = float((fill_prices * fill_volumes).sum() / total_vol) if total_vol > 0 else 0.0

        optimal = float(market_prices.min()) if side == "buy" else float(market_prices.max())

        mkt_vol = market_volumes.sum()
        vwap = float((market_prices * market_volumes).sum() / mkt_vol) if mkt_vol > 0 else 0.0
        twap = float(market_prices.mean())

        sign = 1.0 if side == "buy" else -1.0
        timing_cost = sign * (actual_avg - optimal)
        ref = abs(optimal) if optimal != 0 else 1.0
        timing_bps = timing_cost / ref * 10000

        return TimingAnalysis(
            actual_avg_price=actual_avg,
            optimal_price=optimal,
            vwap_price=vwap,
            twap_price=twap,
            timing_cost=timing_cost,
            timing_cost_bps=timing_bps,
        )

    # ------------------------------------------------------------------
    # 3. Venue analysis
    # ------------------------------------------------------------------

    @staticmethod
    def venue_analysis(fills: pd.DataFrame) -> List[VenueStats]:
        """Fill quality breakdown by venue.

        Required columns: venue, fill_qty, midprice, fill_price.
        Optional: fill_time_ms, orders_routed.
        """
        required = {"venue", "fill_qty", "midprice", "fill_price"}
        if not required.issubset(fills.columns) or fills.empty:
            return []

        results: List[VenueStats] = []
        for venue, grp in fills.groupby("venue"):
            n = len(grp)
            total_vol = float(grp["fill_qty"].sum())
            avg_pi = float((grp["midprice"] - grp["fill_price"]).abs().mean())

            fr = 1.0
            if "orders_routed" in grp.columns:
                routed = float(grp["orders_routed"].sum())
                fr = n / routed if routed > 0 else 1.0

            avg_time = 0.0
            if "fill_time_ms" in grp.columns:
                avg_time = float(grp["fill_time_ms"].mean())

            results.append(VenueStats(
                venue=str(venue), n_fills=n, total_volume=total_vol,
                fill_rate=fr, avg_price_improvement=avg_pi,
                avg_fill_time_ms=avg_time,
            ))

        results.sort(key=lambda v: v.avg_price_improvement, reverse=True)
        return results

    # ------------------------------------------------------------------
    # 4. VWAP / TWAP benchmarks
    # ------------------------------------------------------------------

    @staticmethod
    def compute_vwap(prices: pd.Series, volumes: pd.Series) -> float:
        """Volume-weighted average price."""
        aligned = pd.DataFrame({"p": prices, "v": volumes}).dropna()
        total = aligned["v"].sum()
        return float((aligned["p"] * aligned["v"]).sum() / total) if total > 0 else 0.0

    @staticmethod
    def compute_twap(prices: pd.Series) -> float:
        """Time-weighted average price."""
        return float(prices.mean()) if not prices.empty else 0.0

    @staticmethod
    def benchmark_comparison(
        fill_price: float,
        market_prices: pd.Series,
        market_volumes: pd.Series,
        side: str = "buy",
    ) -> BenchmarkComparison:
        """Compare a fill price against VWAP and TWAP benchmarks."""
        vwap = FillAnalytics.compute_vwap(market_prices, market_volumes)
        twap = FillAnalytics.compute_twap(market_prices)

        sign = 1.0 if side == "buy" else -1.0
        vs_vwap = sign * (fill_price - vwap) / abs(vwap) * 10000 if vwap != 0 else 0.0
        vs_twap = sign * (fill_price - twap) / abs(twap) * 10000 if twap != 0 else 0.0

        return BenchmarkComparison(
            fill_price=fill_price, vwap=vwap, twap=twap,
            vs_vwap_bps=vs_vwap, vs_twap_bps=vs_twap, side=side,
        )

    @staticmethod
    def benchmark_fills(
        fills: pd.DataFrame,
        market_prices: pd.Series,
        market_volumes: pd.Series,
    ) -> List[BenchmarkComparison]:
        """Benchmark each fill in a DataFrame.

        Required columns: fill_price, side.
        """
        if fills.empty or "fill_price" not in fills.columns:
            return []
        results: List[BenchmarkComparison] = []
        for _, row in fills.iterrows():
            side = str(row.get("side", "buy"))
            bc = FillAnalytics.benchmark_comparison(
                float(row["fill_price"]), market_prices, market_volumes, side)
            results.append(bc)
        return results

    # ------------------------------------------------------------------
    # 5. Slippage attribution
    # ------------------------------------------------------------------

    @staticmethod
    def slippage_by_time_of_day(fills: pd.DataFrame) -> List[SlippageBucket]:
        """Attribute slippage to time-of-day buckets.

        Required columns: fill_time (datetime), slippage_bps, fill_qty.
        """
        if fills.empty or "fill_time" not in fills.columns:
            return []
        df = fills.copy()
        df["hour"] = pd.to_datetime(df["fill_time"]).dt.hour
        results: List[SlippageBucket] = []
        for hour, grp in df.groupby("hour"):
            results.append(SlippageBucket(
                bucket_label=f"{int(hour):02d}:00",
                bucket_type="time_of_day",
                n_fills=len(grp),
                avg_slippage_bps=float(grp["slippage_bps"].mean()),
                total_slippage=float(grp["slippage_bps"].sum()),
                avg_fill_size=float(grp["fill_qty"].mean()) if "fill_qty" in grp.columns else 0.0,
            ))
        return results

    @staticmethod
    def slippage_by_volatility(
        fills: pd.DataFrame, vol_col: str = "volatility", n_buckets: int = 3,
    ) -> List[SlippageBucket]:
        """Attribute slippage to volatility buckets.

        Required columns: slippage_bps, `vol_col`, fill_qty.
        """
        if fills.empty or vol_col not in fills.columns:
            return []
        df = fills.copy()
        labels = ["low", "medium", "high"][:n_buckets]
        df["vbucket"] = pd.qcut(df[vol_col], q=n_buckets, labels=labels, duplicates="drop")
        results: List[SlippageBucket] = []
        for bucket, grp in df.groupby("vbucket", observed=True):
            results.append(SlippageBucket(
                bucket_label=str(bucket),
                bucket_type="volatility",
                n_fills=len(grp),
                avg_slippage_bps=float(grp["slippage_bps"].mean()),
                total_slippage=float(grp["slippage_bps"].sum()),
                avg_fill_size=float(grp["fill_qty"].mean()) if "fill_qty" in grp.columns else 0.0,
            ))
        return results

    @staticmethod
    def slippage_by_size(
        fills: pd.DataFrame, n_buckets: int = 3,
    ) -> List[SlippageBucket]:
        """Attribute slippage to order size buckets.

        Required columns: slippage_bps, fill_qty.
        """
        if fills.empty or "fill_qty" not in fills.columns:
            return []
        df = fills.copy()
        labels = ["small", "medium", "large"][:n_buckets]
        df["sbucket"] = pd.qcut(df["fill_qty"], q=n_buckets, labels=labels, duplicates="drop")
        results: List[SlippageBucket] = []
        for bucket, grp in df.groupby("sbucket", observed=True):
            results.append(SlippageBucket(
                bucket_label=str(bucket),
                bucket_type="size",
                n_fills=len(grp),
                avg_slippage_bps=float(grp["slippage_bps"].mean()),
                total_slippage=float(grp["slippage_bps"].sum()),
                avg_fill_size=float(grp["fill_qty"].mean()),
            ))
        return results

    # ------------------------------------------------------------------
    # 6. Execution scorecard
    # ------------------------------------------------------------------

    @staticmethod
    def execution_scorecard(
        fills: pd.DataFrame,
        strategy_col: str = "strategy",
    ) -> List[ExecutionScorecard]:
        """Per-strategy execution quality scorecard.

        Required columns: strategy, slippage_bps, vs_vwap_bps.
        Optional: shortfall_bps, fill_rate.
        """
        if fills.empty or strategy_col not in fills.columns:
            return []

        results: List[ExecutionScorecard] = []
        for strat, grp in fills.groupby(strategy_col):
            n = len(grp)
            avg_slip = float(grp["slippage_bps"].mean()) if "slippage_bps" in grp.columns else 0.0
            avg_vwap = float(grp["vs_vwap_bps"].mean()) if "vs_vwap_bps" in grp.columns else 0.0
            avg_sf = float(grp["shortfall_bps"].mean()) if "shortfall_bps" in grp.columns else 0.0
            fr = float(grp["fill_rate"].mean()) if "fill_rate" in grp.columns else 1.0

            # Score: 100 - penalties.  Lower slippage/shortfall = higher score.
            penalty = abs(avg_slip) * 0.3 + abs(avg_vwap) * 0.3 + abs(avg_sf) * 0.2
            fill_bonus = fr * 20  # up to 20 points for fill rate
            score = max(0.0, min(100.0, 100.0 - penalty + fill_bonus - 20))

            results.append(ExecutionScorecard(
                strategy=str(strat), n_trades=n,
                avg_shortfall_bps=avg_sf, avg_vs_vwap_bps=avg_vwap,
                avg_slippage_bps=avg_slip, fill_rate=fr, score=score,
            ))

        results.sort(key=lambda s: s.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # 7. HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_bar(
        labels: List[str], values: List[float], title: str,
        width: int = 650, height: int = 200, color: str = "#2980b9",
    ) -> str:
        if not values:
            return ""
        n = len(values)
        abs_max = max(abs(v) for v in values) or 1.0
        pad_l, pad_b = 70, 40
        pw = width - pad_l - 20
        ph = height - 55 - pad_b
        bw = pw / max(n, 1) * 0.7
        gap = pw / max(n, 1)

        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="18" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">{title}</text>')

        # zero line
        base_y = 30 + ph
        if any(v < 0 for v in values):
            base_y = 30 + ph / 2
            p.append(f'<line x1="{pad_l}" y1="{base_y:.0f}" x2="{width - 20}" '
                     f'y2="{base_y:.0f}" stroke="#ccc" stroke-dasharray="3,3"/>')

        for i in range(n):
            x = pad_l + i * gap + (gap - bw) / 2
            v = values[i]
            bh = abs(v) / abs_max * (ph / 2 if any(vv < 0 for vv in values) else ph)
            c = "#27ae60" if v <= 0 else "#e74c3c"
            if all(vv >= 0 for vv in values):
                c = color
            y = base_y - bh if v >= 0 else base_y
            p.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" '
                     f'height="{max(bh, 1):.0f}" fill="{c}" rx="3"/>')
            p.append(f'<text x="{x + bw / 2:.0f}" y="{height - 8:.0f}" text-anchor="middle" '
                     f'font-size="9" fill="#666">{labels[i]}</text>')
            p.append(f'<text x="{x + bw / 2:.0f}" y="{y - 3:.0f}" text-anchor="middle" '
                     f'font-size="9" fill="#333">{v:+.1f}</text>')
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self,
        shortfall: Optional[ShortfallDecomposition] = None,
        timing: Optional[TimingAnalysis] = None,
        venues: Optional[List[VenueStats]] = None,
        benchmarks: Optional[List[BenchmarkComparison]] = None,
        slippage_tod: Optional[List[SlippageBucket]] = None,
        slippage_vol: Optional[List[SlippageBucket]] = None,
        slippage_size: Optional[List[SlippageBucket]] = None,
        scorecards: Optional[List[ExecutionScorecard]] = None,
        output_path: str = "reports/fill_analytics.html",
    ) -> str:
        """HTML report: cost breakdown, timing, venue, slippage."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Shortfall
        sf_html = ""
        sf_svg = ""
        if shortfall is not None:
            sf_svg = self._svg_bar(
                ["Delay", "Trading", "Opportunity"],
                [shortfall.delay_cost, shortfall.trading_cost, shortfall.opportunity_cost],
                "Implementation Shortfall Decomposition",
            )
            sf_html = f"""
<table class="m"><tr><th>Total</th><th>Delay</th><th>Trading</th>
<th>Opportunity</th><th>Total (bps)</th></tr>
<tr><td>{shortfall.total_shortfall:+.4f}</td><td>{shortfall.delay_cost:+.4f}</td>
<td>{shortfall.trading_cost:+.4f}</td><td>{shortfall.opportunity_cost:+.4f}</td>
<td>{shortfall.total_bps:+.1f}</td></tr></table>"""

        # Timing
        tm_html = ""
        if timing is not None:
            tm_html = f"""
<h2>Timing Analysis</h2>
<table class="m"><tr><th>Actual Avg</th><th>Optimal</th><th>VWAP</th>
<th>TWAP</th><th>Timing Cost</th><th>Timing (bps)</th></tr>
<tr><td>{timing.actual_avg_price:.4f}</td><td>{timing.optimal_price:.4f}</td>
<td>{timing.vwap_price:.4f}</td><td>{timing.twap_price:.4f}</td>
<td>{timing.timing_cost:+.4f}</td><td>{timing.timing_cost_bps:+.1f}</td></tr></table>"""

        # Venues
        vn_html = ""
        if venues:
            rows = [
                f"<tr><td style='text-align:left'>{v.venue}</td><td>{v.n_fills}</td>"
                f"<td>{v.total_volume:,.0f}</td><td>{v.fill_rate:.1%}</td>"
                f"<td>{v.avg_price_improvement:.6f}</td>"
                f"<td>{v.avg_fill_time_ms:.0f}</td></tr>"
                for v in venues
            ]
            vn_html = f"""
<h2>Venue Analysis</h2>
<table><tr><th style='text-align:left'>Venue</th><th>Fills</th><th>Volume</th>
<th>Fill Rate</th><th>Price Imp.</th><th>Avg Time (ms)</th></tr>
{''.join(rows)}</table>"""

        # Slippage charts
        slip_svg = ""
        if slippage_tod:
            labs = [s.bucket_label for s in slippage_tod]
            vals = [s.avg_slippage_bps for s in slippage_tod]
            slip_svg += '<h2>Slippage by Time of Day</h2>\n'
            slip_svg += self._svg_bar(labs, vals, "Avg Slippage (bps) by Hour")
        if slippage_vol:
            labs = [s.bucket_label for s in slippage_vol]
            vals = [s.avg_slippage_bps for s in slippage_vol]
            slip_svg += '<h2>Slippage by Volatility</h2>\n'
            slip_svg += self._svg_bar(labs, vals, "Avg Slippage (bps) by Vol Bucket")
        if slippage_size:
            labs = [s.bucket_label for s in slippage_size]
            vals = [s.avg_slippage_bps for s in slippage_size]
            slip_svg += '<h2>Slippage by Size</h2>\n'
            slip_svg += self._svg_bar(labs, vals, "Avg Slippage (bps) by Order Size")

        # Scorecards
        sc_html = ""
        if scorecards:
            rows = [
                f"<tr><td style='text-align:left'>{s.strategy}</td><td>{s.n_trades}</td>"
                f"<td>{s.avg_shortfall_bps:+.1f}</td><td>{s.avg_vs_vwap_bps:+.1f}</td>"
                f"<td>{s.avg_slippage_bps:+.1f}</td><td>{s.fill_rate:.1%}</td>"
                f"<td><strong>{s.score:.0f}</strong></td></tr>"
                for s in scorecards
            ]
            sc_html = f"""
<h2>Execution Scorecards</h2>
<table><tr><th style='text-align:left'>Strategy</th><th>Trades</th>
<th>Shortfall (bps)</th><th>vs VWAP (bps)</th><th>Slippage (bps)</th>
<th>Fill Rate</th><th>Score</th></tr>
{''.join(rows)}</table>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Fill Analytics</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
table.m {{ width: auto; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
</style></head><body>
<h1>Fill Quality Analytics</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>

<h2>Implementation Shortfall</h2>
{sf_svg}
{sf_html}
{tm_html}
{vn_html}
{slip_svg}
{sc_html}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Fill analytics report -> %s", path)
        return str(path)
