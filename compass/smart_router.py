from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data definitions
# ---------------------------------------------------------------------------

@dataclass
class VenueDef:
    """Definition of a single execution venue."""
    name: str
    venue_type: str  # "lit", "dark", "midpoint"
    avg_spread_bps: float
    fill_rate: float  # 0-1
    latency_ms: float
    rebate_bps: float
    fee_bps: float


@dataclass
class RoutingDecision:
    """A single leg of a routed order."""
    venue: VenueDef
    quantity: int
    score: float
    reason: str


@dataclass
class CostAttribution:
    """Per-venue cost breakdown for a routed leg."""
    venue_name: str
    quantity: int
    spread_cost_bps: float
    fee_cost_bps: float
    rebate_bps: float
    impact_cost_bps: float
    total_cost_bps: float


@dataclass
class QueueEstimate:
    """Estimated queue position and expected wait."""
    venue_name: str
    queue_position: int
    expected_wait_ms: float
    fill_probability: float


@dataclass
class AdverseSelectionEstimate:
    """Toxicity estimate for a venue fill."""
    venue_name: str
    spread_widening_bps: float
    toxicity_score: float  # 0-1
    expected_adverse_cost_bps: float


@dataclass
class RoutingResult:
    """Complete result of an order routing decision."""
    decisions: List[RoutingDecision]
    cost_attributions: List[CostAttribution]
    queue_estimates: List[QueueEstimate]
    adverse_selection: List[AdverseSelectionEstimate]
    total_cost_bps: float
    naive_cost_bps: float
    savings_bps: float


# ---------------------------------------------------------------------------
# Predefined venues
# ---------------------------------------------------------------------------

DEFAULT_VENUES: List[VenueDef] = [
    VenueDef(
        name="NYSE",
        venue_type="lit",
        avg_spread_bps=1.2,
        fill_rate=0.85,
        latency_ms=0.5,
        rebate_bps=0.25,
        fee_bps=0.30,
    ),
    VenueDef(
        name="NASDAQ",
        venue_type="lit",
        avg_spread_bps=1.1,
        fill_rate=0.88,
        latency_ms=0.3,
        rebate_bps=0.28,
        fee_bps=0.30,
    ),
    VenueDef(
        name="BATS",
        venue_type="lit",
        avg_spread_bps=1.0,
        fill_rate=0.82,
        latency_ms=0.4,
        rebate_bps=0.30,
        fee_bps=0.28,
    ),
    VenueDef(
        name="IEX",
        venue_type="lit",
        avg_spread_bps=1.3,
        fill_rate=0.75,
        latency_ms=0.8,
        rebate_bps=0.10,
        fee_bps=0.09,
    ),
    VenueDef(
        name="SIGMA-X",
        venue_type="dark",
        avg_spread_bps=0.5,
        fill_rate=0.35,
        latency_ms=2.0,
        rebate_bps=0.0,
        fee_bps=0.15,
    ),
    VenueDef(
        name="CROSSFINDER",
        venue_type="dark",
        avg_spread_bps=0.4,
        fill_rate=0.30,
        latency_ms=2.5,
        rebate_bps=0.0,
        fee_bps=0.18,
    ),
]


# ---------------------------------------------------------------------------
# SmartRouter
# ---------------------------------------------------------------------------

class SmartRouter:
    """Smart order routing engine.

    Scores venues on a weighted combination of liquidity, cost, and latency,
    then splits orders across the top venues while respecting dark-pool
    thresholds and time-of-day adjustments.
    """

    # Default scoring weights
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "liquidity": 0.40,
        "cost": 0.35,
        "latency": 0.25,
    }

    # Dark-pool parameters
    DEFAULT_DARK_SIZE_THRESHOLD: int = 500
    DEFAULT_DARK_SPREAD_SAVING_MIN_BPS: float = 0.3

    # Time-of-day windows (Eastern)
    OPEN_START: time = time(9, 30)
    OPEN_END: time = time(10, 0)
    CLOSE_START: time = time(15, 30)
    CLOSE_END: time = time(16, 0)

    def __init__(
        self,
        venues: Optional[List[VenueDef]] = None,
        weights: Optional[Dict[str, float]] = None,
        dark_size_threshold: int = DEFAULT_DARK_SIZE_THRESHOLD,
        dark_spread_saving_min_bps: float = DEFAULT_DARK_SPREAD_SAVING_MIN_BPS,
        current_time: Optional[time] = None,
    ) -> None:
        self.venues = venues if venues is not None else list(DEFAULT_VENUES)
        self.weights = weights if weights is not None else dict(self.DEFAULT_WEIGHTS)
        self.dark_size_threshold = dark_size_threshold
        self.dark_spread_saving_min_bps = dark_spread_saving_min_bps
        self._current_time = current_time

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    @property
    def current_time(self) -> time:
        if self._current_time is not None:
            return self._current_time
        return datetime.now().time()

    def is_open_or_close(self) -> bool:
        t = self.current_time
        return (self.OPEN_START <= t <= self.OPEN_END) or (
            self.CLOSE_START <= t <= self.CLOSE_END
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _liquidity_score(self, venue: VenueDef) -> float:
        """Higher fill rate => higher score (0-1)."""
        return float(np.clip(venue.fill_rate, 0.0, 1.0))

    def _cost_score(self, venue: VenueDef) -> float:
        """Lower net cost => higher score.  Net cost = spread + fee - rebate."""
        net_cost = venue.avg_spread_bps + venue.fee_bps - venue.rebate_bps
        # Normalise: assume max plausible net cost ~5 bps
        return float(np.clip(1.0 - net_cost / 5.0, 0.0, 1.0))

    def _latency_score(self, venue: VenueDef) -> float:
        """Lower latency => higher score.  Normalised against 5 ms ceiling."""
        return float(np.clip(1.0 - venue.latency_ms / 5.0, 0.0, 1.0))

    def score_venue(self, venue: VenueDef) -> float:
        """Compute weighted composite score for *venue*."""
        w = self.weights
        raw = (
            w["liquidity"] * self._liquidity_score(venue)
            + w["cost"] * self._cost_score(venue)
            + w["latency"] * self._latency_score(venue)
        )
        # Time-of-day adjustment: prefer IEX and dark pools at open/close
        if self.is_open_or_close():
            if venue.venue_type == "dark" or venue.name == "IEX":
                raw *= 1.15  # 15 % boost
            else:
                raw *= 0.90  # 10 % penalty for other lit venues
        return float(np.clip(raw, 0.0, 1.0))

    def score_all_venues(self) -> List[Tuple[VenueDef, float]]:
        """Return venues sorted by descending score."""
        scored = [(v, self.score_venue(v)) for v in self.venues]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Dark-pool decision
    # ------------------------------------------------------------------

    def should_use_dark(self, order_size: int) -> bool:
        """Decide whether dark-pool routing is appropriate."""
        if order_size < self.dark_size_threshold:
            return False
        lit_venues = [v for v in self.venues if v.venue_type == "lit"]
        dark_venues = [v for v in self.venues if v.venue_type in ("dark", "midpoint")]
        if not lit_venues or not dark_venues:
            return False
        avg_lit_spread = float(np.mean([v.avg_spread_bps for v in lit_venues]))
        avg_dark_spread = float(np.mean([v.avg_spread_bps for v in dark_venues]))
        avg_dark_fee = float(np.mean([v.fee_bps for v in dark_venues]))
        avg_lit_fee = float(np.mean([v.fee_bps for v in lit_venues]))
        spread_saving = avg_lit_spread - avg_dark_spread
        fee_delta = avg_dark_fee - avg_lit_fee  # dark fees minus lit net fees
        # rebate offsets on the lit side
        avg_lit_rebate = float(np.mean([v.rebate_bps for v in lit_venues]))
        fee_delta += avg_lit_rebate  # losing the rebate is a cost
        return spread_saving > (fee_delta + self.dark_spread_saving_min_bps)

    # ------------------------------------------------------------------
    # Order splitting
    # ------------------------------------------------------------------

    def split_order(
        self, total_qty: int, price: float = 100.0
    ) -> List[RoutingDecision]:
        """Split *total_qty* across venues proportionally to their scores."""
        scored = self.score_all_venues()
        use_dark = self.should_use_dark(total_qty)

        eligible: List[Tuple[VenueDef, float]] = []
        for v, s in scored:
            if v.venue_type in ("dark", "midpoint") and not use_dark:
                continue
            eligible.append((v, s))

        if not eligible:
            # Fallback: use all venues
            eligible = scored

        total_score = sum(s for _, s in eligible)
        if total_score == 0:
            # Equal split
            total_score = float(len(eligible))
            eligible = [(v, 1.0) for v, _ in eligible]

        decisions: List[RoutingDecision] = []
        remaining = total_qty
        for i, (venue, score) in enumerate(eligible):
            if i == len(eligible) - 1:
                qty = remaining
            else:
                qty = int(round(total_qty * score / total_score))
                qty = min(qty, remaining)
            if qty <= 0:
                continue
            remaining -= qty
            reason = "dark_preferred" if venue.venue_type in ("dark", "midpoint") else "lit_scored"
            if self.is_open_or_close() and (venue.venue_type == "dark" or venue.name == "IEX"):
                reason = "tod_adjusted"
            decisions.append(RoutingDecision(venue=venue, quantity=qty, score=score, reason=reason))
            if remaining <= 0:
                break

        return decisions

    # ------------------------------------------------------------------
    # Queue position estimation
    # ------------------------------------------------------------------

    def estimate_queue_position(
        self, venue: VenueDef, order_qty: int, depth: int = 10000
    ) -> QueueEstimate:
        """Simple queue-position model.

        Queue position = depth * (1 - fill_rate).
        Expected wait proportional to queue position and latency.
        """
        queue_pos = int(round(depth * (1.0 - venue.fill_rate)))
        expected_wait = venue.latency_ms + queue_pos * 0.01  # 0.01 ms per position
        fill_prob = venue.fill_rate * np.clip(order_qty / max(depth, 1), 0.0, 1.0)
        fill_prob = float(np.clip(fill_prob, 0.0, 1.0))
        return QueueEstimate(
            venue_name=venue.name,
            queue_position=queue_pos,
            expected_wait_ms=round(expected_wait, 4),
            fill_probability=round(fill_prob, 4),
        )

    # ------------------------------------------------------------------
    # Adverse selection cost model
    # ------------------------------------------------------------------

    def estimate_adverse_selection(
        self, venue: VenueDef, order_qty: int, depth: int = 10000
    ) -> AdverseSelectionEstimate:
        """Estimate toxicity based on expected spread widening after fill.

        Spread widening model: base widening proportional to order_qty/depth,
        amplified for dark pools (information leakage is lower but adverse
        selection per fill is higher due to informed flow).
        """
        qty_ratio = order_qty / max(depth, 1)
        base_widening = venue.avg_spread_bps * qty_ratio * 2.0
        if venue.venue_type in ("dark", "midpoint"):
            toxicity_multiplier = 1.5
        else:
            toxicity_multiplier = 1.0
        spread_widening = base_widening * toxicity_multiplier
        toxicity_score = float(np.clip(spread_widening / 5.0, 0.0, 1.0))
        adverse_cost = spread_widening * 0.5  # half the widening realized
        return AdverseSelectionEstimate(
            venue_name=venue.name,
            spread_widening_bps=round(spread_widening, 6),
            toxicity_score=round(toxicity_score, 6),
            expected_adverse_cost_bps=round(adverse_cost, 6),
        )

    # ------------------------------------------------------------------
    # Cost attribution
    # ------------------------------------------------------------------

    def attribute_cost(
        self, venue: VenueDef, quantity: int, depth: int = 10000
    ) -> CostAttribution:
        """Per-venue cost: spread + fee - rebate + market impact."""
        spread_cost = venue.avg_spread_bps
        fee_cost = venue.fee_bps
        rebate = venue.rebate_bps
        impact = self.estimate_adverse_selection(venue, quantity, depth).expected_adverse_cost_bps
        total = spread_cost + fee_cost - rebate + impact
        return CostAttribution(
            venue_name=venue.name,
            quantity=quantity,
            spread_cost_bps=round(spread_cost, 6),
            fee_cost_bps=round(fee_cost, 6),
            rebate_bps=round(rebate, 6),
            impact_cost_bps=round(impact, 6),
            total_cost_bps=round(total, 6),
        )

    # ------------------------------------------------------------------
    # Naive cost benchmark
    # ------------------------------------------------------------------

    def _naive_cost_bps(self, total_qty: int, depth: int = 10000) -> float:
        """Cost of sending everything to the first lit venue (worst-case naive)."""
        lit = [v for v in self.venues if v.venue_type == "lit"]
        if not lit:
            return 0.0
        # Pick the venue with the widest spread as the naive choice
        worst = max(lit, key=lambda v: v.avg_spread_bps + v.fee_bps - v.rebate_bps)
        ca = self.attribute_cost(worst, total_qty, depth)
        return ca.total_cost_bps

    # ------------------------------------------------------------------
    # Full routing pipeline
    # ------------------------------------------------------------------

    def route_order(
        self, total_qty: int, price: float = 100.0, depth: int = 10000
    ) -> RoutingResult:
        decisions = self.split_order(total_qty, price)
        cost_attributions: List[CostAttribution] = []
        queue_estimates: List[QueueEstimate] = []
        adverse_estimates: List[AdverseSelectionEstimate] = []

        weighted_cost = 0.0
        total_routed = 0
        for d in decisions:
            ca = self.attribute_cost(d.venue, d.quantity, depth)
            qe = self.estimate_queue_position(d.venue, d.quantity, depth)
            ae = self.estimate_adverse_selection(d.venue, d.quantity, depth)
            cost_attributions.append(ca)
            queue_estimates.append(qe)
            adverse_estimates.append(ae)
            weighted_cost += ca.total_cost_bps * d.quantity
            total_routed += d.quantity

        total_cost = weighted_cost / max(total_routed, 1)
        naive_cost = self._naive_cost_bps(total_qty, depth)
        savings = naive_cost - total_cost

        return RoutingResult(
            decisions=decisions,
            cost_attributions=cost_attributions,
            queue_estimates=queue_estimates,
            adverse_selection=adverse_estimates,
            total_cost_bps=round(total_cost, 6),
            naive_cost_bps=round(naive_cost, 6),
            savings_bps=round(savings, 6),
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        total_qty: int = 1000,
        price: float = 100.0,
        depth: int = 10000,
    ) -> str:
        """Generate an HTML report with venue comparison, routing decisions,
        and cost savings versus naive routing."""
        result = self.route_order(total_qty, price, depth)
        scored = self.score_all_venues()

        esc = html.escape

        lines: List[str] = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html lang='en'><head><meta charset='utf-8'>")
        lines.append("<title>Smart Router Report</title>")
        lines.append("<style>")
        lines.append("body{font-family:Arial,sans-serif;margin:20px;}")
        lines.append("table{border-collapse:collapse;width:100%;margin-bottom:24px;}")
        lines.append("th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;}")
        lines.append("th{background:#f4f4f4;}")
        lines.append("td:first-child,th:first-child{text-align:left;}")
        lines.append(".positive{color:green;} .negative{color:red;}")
        lines.append("h1,h2,h3{color:#333;}")
        lines.append("</style></head><body>")

        lines.append("<h1>Smart Order Routing Report</h1>")
        lines.append(f"<p>Order size: <b>{total_qty:,}</b> shares | "
                     f"Reference price: <b>${price:,.2f}</b> | "
                     f"Depth: <b>{depth:,}</b></p>")

        # ---- Venue comparison table ----
        lines.append("<h2>Venue Comparison</h2>")
        lines.append("<table><thead><tr>")
        for hdr in ["Venue", "Type", "Spread (bps)", "Fill Rate", "Latency (ms)",
                     "Rebate (bps)", "Fee (bps)", "Score"]:
            lines.append(f"<th>{hdr}</th>")
        lines.append("</tr></thead><tbody>")
        for venue, score in scored:
            lines.append("<tr>")
            lines.append(f"<td>{esc(venue.name)}</td>")
            lines.append(f"<td>{esc(venue.venue_type)}</td>")
            lines.append(f"<td>{venue.avg_spread_bps:.2f}</td>")
            lines.append(f"<td>{venue.fill_rate:.2%}</td>")
            lines.append(f"<td>{venue.latency_ms:.1f}</td>")
            lines.append(f"<td>{venue.rebate_bps:.2f}</td>")
            lines.append(f"<td>{venue.fee_bps:.2f}</td>")
            lines.append(f"<td>{score:.4f}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table>")

        # ---- Routing decisions ----
        lines.append("<h2>Routing Decisions</h2>")
        lines.append("<table><thead><tr>")
        for hdr in ["Venue", "Quantity", "Score", "Reason"]:
            lines.append(f"<th>{hdr}</th>")
        lines.append("</tr></thead><tbody>")
        for d in result.decisions:
            lines.append("<tr>")
            lines.append(f"<td>{esc(d.venue.name)}</td>")
            lines.append(f"<td>{d.quantity:,}</td>")
            lines.append(f"<td>{d.score:.4f}</td>")
            lines.append(f"<td>{esc(d.reason)}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table>")

        # ---- Cost attribution ----
        lines.append("<h2>Cost Attribution</h2>")
        lines.append("<table><thead><tr>")
        for hdr in ["Venue", "Qty", "Spread (bps)", "Fee (bps)", "Rebate (bps)",
                     "Impact (bps)", "Total (bps)"]:
            lines.append(f"<th>{hdr}</th>")
        lines.append("</tr></thead><tbody>")
        for ca in result.cost_attributions:
            lines.append("<tr>")
            lines.append(f"<td>{esc(ca.venue_name)}</td>")
            lines.append(f"<td>{ca.quantity:,}</td>")
            lines.append(f"<td>{ca.spread_cost_bps:.4f}</td>")
            lines.append(f"<td>{ca.fee_cost_bps:.4f}</td>")
            lines.append(f"<td>{ca.rebate_bps:.4f}</td>")
            lines.append(f"<td>{ca.impact_cost_bps:.4f}</td>")
            lines.append(f"<td>{ca.total_cost_bps:.4f}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table>")

        # ---- Queue estimates ----
        lines.append("<h2>Queue Position Estimates</h2>")
        lines.append("<table><thead><tr>")
        for hdr in ["Venue", "Queue Pos", "Wait (ms)", "Fill Prob"]:
            lines.append(f"<th>{hdr}</th>")
        lines.append("</tr></thead><tbody>")
        for qe in result.queue_estimates:
            lines.append("<tr>")
            lines.append(f"<td>{esc(qe.venue_name)}</td>")
            lines.append(f"<td>{qe.queue_position:,}</td>")
            lines.append(f"<td>{qe.expected_wait_ms:.2f}</td>")
            lines.append(f"<td>{qe.fill_probability:.4f}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table>")

        # ---- Adverse selection ----
        lines.append("<h2>Adverse Selection Estimates</h2>")
        lines.append("<table><thead><tr>")
        for hdr in ["Venue", "Spread Widening (bps)", "Toxicity", "Adverse Cost (bps)"]:
            lines.append(f"<th>{hdr}</th>")
        lines.append("</tr></thead><tbody>")
        for ae in result.adverse_selection:
            lines.append("<tr>")
            lines.append(f"<td>{esc(ae.venue_name)}</td>")
            lines.append(f"<td>{ae.spread_widening_bps:.4f}</td>")
            lines.append(f"<td>{ae.toxicity_score:.4f}</td>")
            lines.append(f"<td>{ae.expected_adverse_cost_bps:.4f}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table>")

        # ---- Summary ----
        savings_cls = "positive" if result.savings_bps >= 0 else "negative"
        lines.append("<h2>Cost Summary</h2>")
        lines.append("<table><tbody>")
        lines.append(f"<tr><td>Smart routing cost</td><td>{result.total_cost_bps:.4f} bps</td></tr>")
        lines.append(f"<tr><td>Naive routing cost</td><td>{result.naive_cost_bps:.4f} bps</td></tr>")
        lines.append(f"<tr><td>Savings</td><td class='{savings_cls}'>{result.savings_bps:.4f} bps</td></tr>")
        lines.append("</tbody></table>")

        lines.append("</body></html>")
        return "\n".join(lines)
