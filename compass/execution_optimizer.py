"""Execution Optimizer – pre-trade cost estimation, algorithm selection,
smart venue routing, real-time monitoring, and post-trade TCA."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

Urgency = Literal["low", "medium", "high", "critical"]
AlgoName = Literal["TWAP", "VWAP", "IS", "Iceberg"]


@dataclass
class VenueDef:
    """Definition of an execution venue."""
    name: str
    spread_bps: float
    fill_rate: float      # 0-1
    fee_bps: float
    rebate_bps: float
    latency_ms: float


@dataclass
class VenueAllocation:
    """Allocation result for a single venue."""
    venue: VenueDef
    weight: float          # 0-1, sums to 1.0 across all venues
    score: float


@dataclass
class PreTradeEstimate:
    """Almgren-Chriss pre-trade cost estimate."""
    market_impact_bps: float
    timing_risk_bps: float
    total_bps: float


@dataclass
class PostTradeTCA:
    """Post-trade transaction cost analysis vs benchmarks."""
    arrival_slippage_bps: float
    vwap_slippage_bps: float
    close_slippage_bps: float
    implementation_shortfall_bps: float


@dataclass
class ISDecomposition:
    """Implementation shortfall decomposition."""
    timing_cost_bps: float
    impact_cost_bps: float
    opportunity_cost_bps: float
    total_bps: float


@dataclass
class TimeRecommendation:
    """Time-of-day trading recommendation."""
    hour: int
    volatility_regime: str   # "high" or "low"
    recommended_algo: str
    reason: str


@dataclass
class TradeMonitor:
    """Real-time execution monitor."""
    order_qty: int
    filled_qty: int = 0
    avg_price: float = 0.0
    elapsed_time: float = 0.0   # minutes
    pre_trade_estimate_bps: float = 0.0
    arrival_price: float = 0.0
    aborted: bool = False
    abort_reason: str = ""

    @property
    def fill_pct(self) -> float:
        if self.order_qty == 0:
            return 0.0
        return self.filled_qty / self.order_qty

    @property
    def current_cost_bps(self) -> float:
        if self.arrival_price == 0.0:
            return 0.0
        return abs(self.avg_price - self.arrival_price) / self.arrival_price * 10_000

    def check_abort(self) -> bool:
        """Abort if realised cost exceeds 2x the pre-trade estimate."""
        if self.pre_trade_estimate_bps > 0 and self.current_cost_bps > 2 * self.pre_trade_estimate_bps:
            self.aborted = True
            self.abort_reason = (
                f"Cost {self.current_cost_bps:.1f} bps exceeds "
                f"2x estimate {self.pre_trade_estimate_bps:.1f} bps"
            )
        return self.aborted


@dataclass
class ExecutionResult:
    """Full execution optimiser result."""
    selected_algo: str
    venue_allocation: List[VenueAllocation]
    pre_trade_estimate: PreTradeEstimate
    post_trade_tca: Optional[PostTradeTCA] = None
    is_decomposition: Optional[ISDecomposition] = None
    time_recommendation: Optional[TimeRecommendation] = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Predefined venues
# ---------------------------------------------------------------------------

NYSE = VenueDef(name="NYSE", spread_bps=1.0, fill_rate=0.85, fee_bps=0.30, rebate_bps=0.20, latency_ms=0.5)
NASDAQ = VenueDef(name="NASDAQ", spread_bps=0.8, fill_rate=0.90, fee_bps=0.25, rebate_bps=0.25, latency_ms=0.3)
IEX = VenueDef(name="IEX", spread_bps=0.6, fill_rate=0.70, fee_bps=0.09, rebate_bps=0.0, latency_ms=1.0)
DARK_POOL = VenueDef(name="DARK_POOL", spread_bps=0.2, fill_rate=0.40, fee_bps=0.10, rebate_bps=0.05, latency_ms=2.0)

DEFAULT_VENUES: List[VenueDef] = [NYSE, NASDAQ, IEX, DARK_POOL]

# ---------------------------------------------------------------------------
# Time-of-day lookup table
# ---------------------------------------------------------------------------

_TOD_TABLE: Dict[int, dict] = {
    # hour -> (volatility_regime, recommended_algo, reason)
    9:  {"vol": "high",  "algo": "IS",      "reason": "Open auction / high volatility – use IS to minimise shortfall"},
    10: {"vol": "high",  "algo": "IS",      "reason": "Post-open volatility still elevated – IS preferred"},
    11: {"vol": "low",   "algo": "VWAP",    "reason": "Mid-morning calm – passive VWAP execution"},
    12: {"vol": "low",   "algo": "TWAP",    "reason": "Lunch lull – spread evenly with TWAP"},
    13: {"vol": "low",   "algo": "TWAP",    "reason": "Midday low volume – passive TWAP"},
    14: {"vol": "low",   "algo": "VWAP",    "reason": "Afternoon ramp beginning – VWAP follows volume curve"},
    15: {"vol": "high",  "algo": "IS",      "reason": "Approaching close – higher volatility, use IS"},
    16: {"vol": "high",  "algo": "IS",      "reason": "Closing auction – high volatility, aggressive IS"},
}


# ---------------------------------------------------------------------------
# ExecutionOptimizer
# ---------------------------------------------------------------------------

class ExecutionOptimizer:
    """Unified execution optimiser: cost estimation, algo selection,
    venue routing, monitoring, TCA, and reporting."""

    def __init__(self, venues: Optional[List[VenueDef]] = None) -> None:
        self.venues = venues if venues is not None else list(DEFAULT_VENUES)

    # ---- Pre-trade cost estimation (Almgren-Chriss) ----

    @staticmethod
    def estimate_cost(
        order_qty: int,
        adv: int,
        volatility: float,
        duration_minutes: float = 30.0,
        eta: float = 0.1,
    ) -> PreTradeEstimate:
        """Almgren-Chriss style pre-trade cost estimate.

        market_impact = eta * sqrt(participation_rate) * 10000  (bps)
        timing_risk   = sigma * sqrt(duration / 390) * 10000    (bps)
        """
        if adv <= 0 or order_qty <= 0:
            return PreTradeEstimate(
                market_impact_bps=0.0, timing_risk_bps=0.0, total_bps=0.0
            )
        participation = order_qty / adv
        market_impact_bps = eta * np.sqrt(participation) * 10_000
        timing_risk_bps = volatility * np.sqrt(duration_minutes / 390) * 10_000
        total_bps = market_impact_bps + timing_risk_bps
        return PreTradeEstimate(
            market_impact_bps=float(market_impact_bps),
            timing_risk_bps=float(timing_risk_bps),
            total_bps=float(total_bps),
        )

    # ---- Algorithm selection ----

    @staticmethod
    def select_algorithm(
        urgency: Urgency,
        order_qty: int,
        adv: int,
        volatility: float,
    ) -> str:
        """Select execution algorithm based on urgency and order characteristics."""
        participation = order_qty / adv if adv > 0 else 0.0

        if urgency == "critical":
            return "IS"
        if urgency == "high":
            return "IS" if volatility > 0.02 else "VWAP"
        if urgency == "medium":
            if participation > 0.05:
                return "Iceberg"
            return "VWAP"
        # low urgency
        if participation > 0.10:
            return "Iceberg"
        return "TWAP"

    # ---- Smart venue routing ----

    def score_venue(self, venue: VenueDef) -> float:
        """Score a venue: lower cost / better fill / lower latency is better.

        score = fill_rate / (cost_bps + latency_penalty)
        where cost_bps = spread + fee - rebate
              latency_penalty = latency_ms / 10  (normalised contribution)
        """
        cost_bps = venue.spread_bps + venue.fee_bps - venue.rebate_bps
        latency_penalty = venue.latency_ms / 10.0
        denominator = cost_bps + latency_penalty
        if denominator <= 0:
            denominator = 1e-6
        return venue.fill_rate / denominator

    def route_order(self, venues: Optional[List[VenueDef]] = None) -> List[VenueAllocation]:
        """Score venues and split order proportionally by score."""
        vlist = venues if venues is not None else self.venues
        if not vlist:
            return []

        scores = [(v, self.score_venue(v)) for v in vlist]
        total_score = sum(s for _, s in scores)
        if total_score <= 0:
            weight = 1.0 / len(scores)
            return [VenueAllocation(venue=v, weight=weight, score=s) for v, s in scores]

        return [
            VenueAllocation(venue=v, weight=s / total_score, score=s)
            for v, s in scores
        ]

    # ---- Real-time monitoring ----

    @staticmethod
    def create_monitor(
        order_qty: int,
        arrival_price: float,
        pre_trade_estimate_bps: float,
    ) -> TradeMonitor:
        return TradeMonitor(
            order_qty=order_qty,
            arrival_price=arrival_price,
            pre_trade_estimate_bps=pre_trade_estimate_bps,
        )

    @staticmethod
    def update_monitor(
        monitor: TradeMonitor,
        filled_qty: int,
        avg_price: float,
        elapsed_time: float,
    ) -> TradeMonitor:
        monitor.filled_qty = filled_qty
        monitor.avg_price = avg_price
        monitor.elapsed_time = elapsed_time
        monitor.check_abort()
        return monitor

    # ---- Post-trade TCA ----

    @staticmethod
    def compute_tca(
        avg_fill_price: float,
        arrival_price: float,
        vwap: float,
        close_price: float,
        side: Literal["buy", "sell"] = "buy",
    ) -> PostTradeTCA:
        """Compute post-trade TCA benchmarks. All results in bps.

        For buys: positive means we paid *more* than the benchmark (bad).
        For sells: positive means we received *less* than the benchmark (bad).
        """
        sign = 1.0 if side == "buy" else -1.0

        def _slip(bench: float) -> float:
            if bench == 0:
                return 0.0
            return sign * (avg_fill_price - bench) / bench * 10_000

        arrival_slip = _slip(arrival_price)
        vwap_slip = _slip(vwap)
        close_slip = _slip(close_price)
        impl_shortfall = arrival_slip  # IS = slippage vs arrival

        return PostTradeTCA(
            arrival_slippage_bps=float(arrival_slip),
            vwap_slippage_bps=float(vwap_slip),
            close_slippage_bps=float(close_slip),
            implementation_shortfall_bps=float(impl_shortfall),
        )

    # ---- IS decomposition ----

    @staticmethod
    def decompose_is(
        arrival_price: float,
        decision_price: float,
        avg_fill_price: float,
        close_price: float,
        filled_qty: int,
        order_qty: int,
        side: Literal["buy", "sell"] = "buy",
    ) -> ISDecomposition:
        """Decompose implementation shortfall into timing, impact and
        opportunity cost components.

        timing_cost    = (arrival_price - decision_price) / decision_price  (delay cost)
        impact_cost    = (avg_fill - arrival_price) / arrival_price         (our trading moved the market)
        opportunity_cost = unfilled_frac * (close - decision_price) / decision_price
        """
        sign = 1.0 if side == "buy" else -1.0

        if decision_price == 0:
            return ISDecomposition(0.0, 0.0, 0.0, 0.0)

        timing_bps = sign * (arrival_price - decision_price) / decision_price * 10_000
        impact_bps = 0.0
        if arrival_price != 0:
            impact_bps = sign * (avg_fill_price - arrival_price) / arrival_price * 10_000

        unfilled_frac = max(0.0, 1.0 - (filled_qty / order_qty)) if order_qty > 0 else 0.0
        opp_bps = sign * unfilled_frac * (close_price - decision_price) / decision_price * 10_000

        total = timing_bps + impact_bps + opp_bps
        return ISDecomposition(
            timing_cost_bps=float(timing_bps),
            impact_cost_bps=float(impact_bps),
            opportunity_cost_bps=float(opp_bps),
            total_bps=float(total),
        )

    # ---- Time-of-day recommendation ----

    @staticmethod
    def recommend_time(hour: int) -> TimeRecommendation:
        """Lookup best algorithm for a given hour of day (ET, 9-16)."""
        entry = _TOD_TABLE.get(hour)
        if entry is None:
            return TimeRecommendation(
                hour=hour,
                volatility_regime="unknown",
                recommended_algo="TWAP",
                reason="Outside regular trading hours – default TWAP",
            )
        return TimeRecommendation(
            hour=hour,
            volatility_regime=entry["vol"],
            recommended_algo=entry["algo"],
            reason=entry["reason"],
        )

    # ---- Full optimisation run ----

    def optimise(
        self,
        order_qty: int,
        adv: int,
        volatility: float,
        urgency: Urgency = "medium",
        duration_minutes: float = 30.0,
        eta: float = 0.1,
        hour: Optional[int] = None,
        # Post-trade fields (optional – supply for TCA)
        avg_fill_price: Optional[float] = None,
        arrival_price: Optional[float] = None,
        decision_price: Optional[float] = None,
        vwap: Optional[float] = None,
        close_price: Optional[float] = None,
        filled_qty: Optional[int] = None,
        side: Literal["buy", "sell"] = "buy",
    ) -> ExecutionResult:
        pre = self.estimate_cost(order_qty, adv, volatility, duration_minutes, eta)
        algo = self.select_algorithm(urgency, order_qty, adv, volatility)
        allocation = self.route_order()

        tca: Optional[PostTradeTCA] = None
        is_dec: Optional[ISDecomposition] = None
        time_rec: Optional[TimeRecommendation] = None

        if avg_fill_price is not None and arrival_price is not None:
            tca = self.compute_tca(
                avg_fill_price, arrival_price,
                vwap or arrival_price,
                close_price or arrival_price,
                side,
            )

        if (decision_price is not None and arrival_price is not None
                and avg_fill_price is not None and close_price is not None):
            is_dec = self.decompose_is(
                arrival_price, decision_price, avg_fill_price,
                close_price, filled_qty or order_qty, order_qty, side,
            )

        if hour is not None:
            time_rec = self.recommend_time(hour)

        return ExecutionResult(
            selected_algo=algo,
            venue_allocation=allocation,
            pre_trade_estimate=pre,
            post_trade_tca=tca,
            is_decomposition=is_dec,
            time_recommendation=time_rec,
        )

    # ---- HTML report ----

    def generate_report(self, result: ExecutionResult) -> str:
        """Generate an HTML scorecard report from an ExecutionResult."""
        h = html.escape

        venue_rows = ""
        for va in result.venue_allocation:
            venue_rows += (
                f"<tr><td>{h(va.venue.name)}</td>"
                f"<td>{va.score:.4f}</td>"
                f"<td>{va.weight * 100:.1f}%</td></tr>\n"
            )

        pre = result.pre_trade_estimate
        cost_section = (
            f"<tr><td>Market Impact</td><td>{pre.market_impact_bps:.2f} bps</td></tr>\n"
            f"<tr><td>Timing Risk</td><td>{pre.timing_risk_bps:.2f} bps</td></tr>\n"
            f"<tr><td><strong>Total Estimated</strong></td><td><strong>{pre.total_bps:.2f} bps</strong></td></tr>\n"
        )

        tca_section = ""
        if result.post_trade_tca:
            t = result.post_trade_tca
            tca_section = f"""
        <h2>Post-Trade TCA</h2>
        <table>
            <tr><th>Benchmark</th><th>Slippage</th></tr>
            <tr><td>Arrival</td><td>{t.arrival_slippage_bps:.2f} bps</td></tr>
            <tr><td>VWAP</td><td>{t.vwap_slippage_bps:.2f} bps</td></tr>
            <tr><td>Close</td><td>{t.close_slippage_bps:.2f} bps</td></tr>
            <tr><td><strong>Impl. Shortfall</strong></td><td><strong>{t.implementation_shortfall_bps:.2f} bps</strong></td></tr>
        </table>"""

        is_section = ""
        if result.is_decomposition:
            d = result.is_decomposition
            is_section = f"""
        <h2>IS Decomposition</h2>
        <table>
            <tr><th>Component</th><th>Cost</th></tr>
            <tr><td>Timing</td><td>{d.timing_cost_bps:.2f} bps</td></tr>
            <tr><td>Impact</td><td>{d.impact_cost_bps:.2f} bps</td></tr>
            <tr><td>Opportunity</td><td>{d.opportunity_cost_bps:.2f} bps</td></tr>
            <tr><td><strong>Total IS</strong></td><td><strong>{d.total_bps:.2f} bps</strong></td></tr>
        </table>"""

        time_section = ""
        if result.time_recommendation:
            tr = result.time_recommendation
            time_section = f"""
        <h2>Time-of-Day Recommendation</h2>
        <p>Hour: {tr.hour}:00 ET | Regime: {h(tr.volatility_regime)} |
        Algo: {h(tr.recommended_algo)}</p>
        <p>{h(tr.reason)}</p>"""

        algo_comparison = ""
        for algo_name in ("TWAP", "VWAP", "IS", "Iceberg"):
            marker = " (selected)" if algo_name == result.selected_algo else ""
            algo_comparison += f"<tr><td>{algo_name}{marker}</td></tr>\n"

        report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Execution Optimizer Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; margin: 10px 0; width: 100%; max-width: 600px; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
        th {{ background: #f0f0f0; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 24px; }}
        .scorecard {{ background: #f9f9f9; padding: 12px; border-radius: 6px; margin: 12px 0; }}
    </style>
</head>
<body>
    <h1>Execution Optimizer Report</h1>
    <p>Generated: {h(result.generated_at)}</p>

    <div class="scorecard">
        <h2>Scorecard</h2>
        <p><strong>Selected Algorithm:</strong> {h(result.selected_algo)}</p>
        <p><strong>Estimated Cost:</strong> {pre.total_bps:.2f} bps</p>
    </div>

    <h2>Pre-Trade Cost Breakdown</h2>
    <table>
        <tr><th>Component</th><th>Value</th></tr>
        {cost_section}
    </table>

    <h2>Algorithm Comparison</h2>
    <table>
        <tr><th>Algorithm</th></tr>
        {algo_comparison}
    </table>

    <h2>Venue Allocation</h2>
    <table>
        <tr><th>Venue</th><th>Score</th><th>Weight</th></tr>
        {venue_rows}
    </table>
    {tca_section}
    {is_section}
    {time_section}
</body>
</html>"""
        return report
