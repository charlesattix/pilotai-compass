"""
Trade execution simulator with realistic option order fill modeling.

Simulates slippage (fixed bps, proportional, volume-dependent), queue position
(time-priority vs pro-rata), partial fills, market impact decay, and latency.
Produces a self-contained HTML report at reports/execution_sim.html.

This is READ-ONLY simulation.  No broker connections, no trade placement.

Usage::

    from compass.execution_simulator import ExecutionSimulator
    sim = ExecutionSimulator()
    results = sim.simulate_orders(orders_df)
    ExecutionSimulator.generate_report(results)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "execution_sim.html"


# ── Enums ────────────────────────────────────────────────────────────────


class SlippageModel(Enum):
    FIXED_BPS = "fixed_bps"
    PROPORTIONAL = "proportional"
    VOLUME_DEPENDENT = "volume_dependent"


class QueueModel(Enum):
    TIME_PRIORITY = "time_priority"
    PRO_RATA = "pro_rata"


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class SlippageConfig:
    """Configuration for slippage modeling."""

    model: SlippageModel = SlippageModel.FIXED_BPS
    fixed_bps: float = 5.0
    proportional_factor: float = 0.1  # fraction of bid-ask spread
    volume_impact_factor: float = 0.5  # how much volume affects slippage
    base_spread_bps: float = 10.0  # base bid-ask spread in bps


@dataclass
class LatencyConfig:
    """Configuration for latency modeling."""

    base_latency_ms: float = 50.0
    jitter_ms: float = 20.0
    network_latency_ms: float = 5.0


@dataclass
class MarketImpactConfig:
    """Configuration for market impact modeling."""

    temporary_impact_bps: float = 3.0
    permanent_impact_bps: float = 1.0
    decay_half_life_seconds: float = 30.0


@dataclass
class OrderRequest:
    """A single order to simulate."""

    order_id: str
    side: str  # "buy" or "sell"
    price: float
    quantity: int
    spread_width: float = 5.0
    market_volume: int = 1000
    timestamp_ms: float = 0.0


@dataclass
class FillResult:
    """Result of simulating a single order fill."""

    order_id: str
    side: str
    requested_price: float
    requested_quantity: int
    filled_price: float
    filled_quantity: int
    slippage_bps: float
    slippage_dollars: float
    fill_ratio: float
    latency_ms: float
    queue_position: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float

    @property
    def is_partial_fill(self) -> bool:
        return self.filled_quantity < self.requested_quantity

    @property
    def is_complete_fill(self) -> bool:
        return self.filled_quantity == self.requested_quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "side": self.side,
            "requested_price": self.requested_price,
            "requested_quantity": self.requested_quantity,
            "filled_price": self.filled_price,
            "filled_quantity": self.filled_quantity,
            "slippage_bps": self.slippage_bps,
            "slippage_dollars": self.slippage_dollars,
            "fill_ratio": self.fill_ratio,
            "latency_ms": self.latency_ms,
            "queue_position": self.queue_position,
            "temporary_impact_bps": self.temporary_impact_bps,
            "permanent_impact_bps": self.permanent_impact_bps,
            "total_impact_bps": self.total_impact_bps,
            "is_partial_fill": self.is_partial_fill,
        }


@dataclass
class SimulationResult:
    """Full result from execution simulation."""

    fills: List[FillResult]
    summary: Dict[str, Any]
    slippage_config: SlippageConfig
    latency_config: LatencyConfig
    impact_config: MarketImpactConfig
    queue_model: QueueModel


# ── Computation engines ──────────────────────────────────────────────────


def compute_slippage_bps(
    config: SlippageConfig,
    order: OrderRequest,
    rng: np.random.RandomState,
) -> float:
    """Compute slippage in basis points for an order."""
    if config.model == SlippageModel.FIXED_BPS:
        noise = rng.normal(0, config.fixed_bps * 0.2)
        return max(0.0, config.fixed_bps + noise)

    elif config.model == SlippageModel.PROPORTIONAL:
        # Slippage proportional to spread width relative to price
        if order.price <= 0:
            return 0.0
        spread_bps = (order.spread_width / order.price) * 10_000
        base = spread_bps * config.proportional_factor
        noise = rng.normal(0, base * 0.2)
        return max(0.0, base + noise)

    elif config.model == SlippageModel.VOLUME_DEPENDENT:
        # Higher quantity relative to market volume → more slippage
        if order.market_volume <= 0:
            volume_ratio = 1.0
        else:
            volume_ratio = order.quantity / order.market_volume
        base = config.base_spread_bps * (
            1.0 + config.volume_impact_factor * math.sqrt(volume_ratio)
        )
        noise = rng.normal(0, base * 0.15)
        return max(0.0, base + noise)

    else:
        raise ValueError(f"Unknown slippage model: {config.model}")


def compute_queue_position(
    model: QueueModel,
    order: OrderRequest,
    rng: np.random.RandomState,
) -> float:
    """Compute queue position as a fraction [0, 1].

    0 = front of queue, 1 = back of queue.
    """
    if model == QueueModel.TIME_PRIORITY:
        # FIFO — earlier timestamp → better position
        # Simulate as random draw weighted by order size
        # (larger orders take longer to fill in FIFO)
        if order.market_volume <= 0:
            return rng.uniform(0.3, 1.0)
        size_penalty = min(1.0, order.quantity / order.market_volume)
        base = rng.uniform(0.0, 0.6)
        return min(1.0, base + size_penalty * 0.4)

    elif model == QueueModel.PRO_RATA:
        # Pro-rata — allocation proportional to order size
        if order.market_volume <= 0:
            return 0.5
        share = order.quantity / order.market_volume
        # Pro-rata gives larger orders proportionally more fills
        # but you still compete with other large orders
        return max(0.0, min(1.0, 1.0 - share + rng.normal(0, 0.1)))

    else:
        raise ValueError(f"Unknown queue model: {model}")


def compute_partial_fill(
    queue_position: float,
    requested_quantity: int,
    market_volume: int,
    rng: np.random.RandomState,
) -> int:
    """Determine filled quantity based on queue position and volume."""
    if requested_quantity <= 0:
        return 0

    # Base fill probability decreases with worse queue position
    fill_prob = max(0.1, 1.0 - queue_position * 0.7)

    # Volume ratio affects fill likelihood
    if market_volume > 0:
        vol_ratio = requested_quantity / market_volume
        if vol_ratio > 0.5:
            fill_prob *= 0.6  # large orders harder to fill completely
        elif vol_ratio > 0.2:
            fill_prob *= 0.85

    # Simulate per-contract fill
    fills = rng.binomial(requested_quantity, fill_prob)
    return max(1, int(fills))  # at least 1 contract fills


def compute_market_impact(
    config: MarketImpactConfig,
    order: OrderRequest,
    rng: np.random.RandomState,
) -> Tuple[float, float]:
    """Compute temporary and permanent market impact in bps.

    Returns:
        (temporary_impact_bps, permanent_impact_bps)
    """
    if order.market_volume <= 0:
        vol_ratio = 1.0
    else:
        vol_ratio = order.quantity / order.market_volume

    # Temporary impact scales with sqrt of volume ratio (square-root law)
    temp_base = config.temporary_impact_bps * math.sqrt(vol_ratio)
    temp_noise = rng.normal(0, temp_base * 0.2)
    temp_impact = max(0.0, temp_base + temp_noise)

    # Permanent impact is a fraction that doesn't decay
    perm_base = config.permanent_impact_bps * vol_ratio
    perm_noise = rng.normal(0, perm_base * 0.1)
    perm_impact = max(0.0, perm_base + perm_noise)

    return (temp_impact, perm_impact)


def compute_latency(
    config: LatencyConfig,
    rng: np.random.RandomState,
) -> float:
    """Compute order-to-fill latency in milliseconds."""
    jitter = abs(rng.normal(0, config.jitter_ms))
    return max(1.0, config.base_latency_ms + config.network_latency_ms + jitter)


def apply_impact_decay(
    temporary_impact_bps: float,
    elapsed_seconds: float,
    half_life: float,
) -> float:
    """Compute remaining temporary impact after elapsed time."""
    if half_life <= 0 or elapsed_seconds < 0:
        return temporary_impact_bps
    decay = 0.5 ** (elapsed_seconds / half_life)
    return temporary_impact_bps * decay


# ── Core simulator ───────────────────────────────────────────────────────


class ExecutionSimulator:
    """Simulates realistic option order execution."""

    def __init__(
        self,
        slippage_config: Optional[SlippageConfig] = None,
        latency_config: Optional[LatencyConfig] = None,
        impact_config: Optional[MarketImpactConfig] = None,
        queue_model: QueueModel = QueueModel.TIME_PRIORITY,
        seed: Optional[int] = None,
    ):
        self.slippage_config = slippage_config or SlippageConfig()
        self.latency_config = latency_config or LatencyConfig()
        self.impact_config = impact_config or MarketImpactConfig()
        self.queue_model = queue_model
        self.seed = seed

    def simulate_single(
        self,
        order: OrderRequest,
        rng: np.random.RandomState,
    ) -> FillResult:
        """Simulate execution of a single order."""
        # Slippage
        slip_bps = compute_slippage_bps(self.slippage_config, order, rng)

        # Queue position and partial fills
        queue_pos = compute_queue_position(self.queue_model, order, rng)
        filled_qty = compute_partial_fill(
            queue_pos, order.quantity, order.market_volume, rng
        )

        # Market impact
        temp_impact, perm_impact = compute_market_impact(
            self.impact_config, order, rng
        )
        total_impact = slip_bps + temp_impact + perm_impact

        # Latency
        latency = compute_latency(self.latency_config, rng)

        # Compute fill price
        slip_dollars_per_unit = order.price * slip_bps / 10_000
        impact_dollars_per_unit = order.price * (temp_impact + perm_impact) / 10_000

        if order.side == "buy":
            filled_price = order.price + slip_dollars_per_unit + impact_dollars_per_unit
        else:
            filled_price = order.price - slip_dollars_per_unit - impact_dollars_per_unit

        total_slip_dollars = abs(filled_price - order.price) * filled_qty * 100

        fill_ratio = filled_qty / order.quantity if order.quantity > 0 else 0.0

        return FillResult(
            order_id=order.order_id,
            side=order.side,
            requested_price=order.price,
            requested_quantity=order.quantity,
            filled_price=round(filled_price, 4),
            filled_quantity=filled_qty,
            slippage_bps=round(slip_bps, 2),
            slippage_dollars=round(total_slip_dollars, 2),
            fill_ratio=round(fill_ratio, 4),
            latency_ms=round(latency, 2),
            queue_position=round(queue_pos, 4),
            temporary_impact_bps=round(temp_impact, 2),
            permanent_impact_bps=round(perm_impact, 2),
            total_impact_bps=round(total_impact, 2),
        )

    def simulate_orders(
        self,
        orders: pd.DataFrame,
    ) -> SimulationResult:
        """Simulate execution for a batch of orders.

        Expected columns: order_id, side, price, quantity.
        Optional: spread_width, market_volume, timestamp_ms.
        """
        required = {"order_id", "side", "price", "quantity"}
        missing = required - set(orders.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        rng = np.random.RandomState(self.seed)
        fills: List[FillResult] = []

        for _, row in orders.iterrows():
            order = OrderRequest(
                order_id=str(row["order_id"]),
                side=str(row["side"]),
                price=float(row["price"]),
                quantity=int(row["quantity"]),
                spread_width=float(row.get("spread_width", 5.0)),
                market_volume=int(row.get("market_volume", 1000)),
                timestamp_ms=float(row.get("timestamp_ms", 0.0)),
            )
            fills.append(self.simulate_single(order, rng))

        summary = self._compute_summary(fills)

        return SimulationResult(
            fills=fills,
            summary=summary,
            slippage_config=self.slippage_config,
            latency_config=self.latency_config,
            impact_config=self.impact_config,
            queue_model=self.queue_model,
        )

    @staticmethod
    def _compute_summary(fills: List[FillResult]) -> Dict[str, Any]:
        """Compute aggregate statistics from fill results."""
        if not fills:
            return {
                "n_orders": 0,
                "n_complete": 0,
                "n_partial": 0,
                "avg_slippage_bps": 0.0,
                "median_slippage_bps": 0.0,
                "p95_slippage_bps": 0.0,
                "avg_fill_ratio": 0.0,
                "avg_latency_ms": 0.0,
                "total_slippage_dollars": 0.0,
                "avg_temp_impact_bps": 0.0,
                "avg_perm_impact_bps": 0.0,
            }

        slips = np.array([f.slippage_bps for f in fills])
        fill_ratios = np.array([f.fill_ratio for f in fills])
        latencies = np.array([f.latency_ms for f in fills])
        temp_impacts = np.array([f.temporary_impact_bps for f in fills])
        perm_impacts = np.array([f.permanent_impact_bps for f in fills])

        return {
            "n_orders": len(fills),
            "n_complete": sum(1 for f in fills if f.is_complete_fill),
            "n_partial": sum(1 for f in fills if f.is_partial_fill),
            "avg_slippage_bps": float(np.mean(slips)),
            "median_slippage_bps": float(np.median(slips)),
            "p95_slippage_bps": float(np.percentile(slips, 95)),
            "avg_fill_ratio": float(np.mean(fill_ratios)),
            "avg_latency_ms": float(np.mean(latencies)),
            "total_slippage_dollars": float(sum(f.slippage_dollars for f in fills)),
            "avg_temp_impact_bps": float(np.mean(temp_impacts)),
            "avg_perm_impact_bps": float(np.mean(perm_impacts)),
        }

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        result: SimulationResult,
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


def _fmt_bps(v: float) -> str:
    return f"{v:.2f} bps"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.2f}"


def _fmt_ms(v: float) -> str:
    return f"{v:.1f} ms"


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _histogram_svg(
    values: List[float], title: str, xlabel: str, n_bins: int = 25
) -> str:
    """Inline SVG histogram."""
    if not values:
        return f"<p>No data for {title}.</p>"

    w, h = 600, 300
    pad = 60

    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=n_bins)
    max_count = max(counts) if len(counts) > 0 else 1
    chart_w = w - 2 * pad
    chart_h = h - 2 * pad
    bar_w = chart_w / len(counts)

    bars = []
    for i, c in enumerate(counts):
        bh = (c / max_count) * chart_h if max_count > 0 else 0
        x = pad + i * bar_w
        y = h - pad - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" '
            f'height="{bh:.1f}" fill="#58a6ff" opacity="0.8"/>'
        )

    return f"""
    <svg viewBox="0 0 {w} {h}" class="chart">
      <text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">{title}</text>
      <text x="{w // 2}" y="{h - 5}" text-anchor="middle" class="svg-label">{xlabel}</text>
      <text x="15" y="{h // 2}" text-anchor="middle" class="svg-label"
            transform="rotate(-90,15,{h // 2})">Count</text>
      {"".join(bars)}
    </svg>"""


def _impact_decay_svg(config: MarketImpactConfig) -> str:
    """SVG chart showing temporary vs permanent impact decay."""
    w, h = 600, 250
    pad = 60
    n_points = 100
    max_seconds = config.decay_half_life_seconds * 5

    chart_w = w - 2 * pad
    chart_h = h - 2 * pad

    temp_base = config.temporary_impact_bps
    perm = config.permanent_impact_bps

    temp_points = []
    perm_points = []
    total_points = []

    for i in range(n_points):
        t = (i / (n_points - 1)) * max_seconds
        x = pad + (i / (n_points - 1)) * chart_w
        temp_val = apply_impact_decay(temp_base, t, config.decay_half_life_seconds)
        total_val = temp_val + perm
        max_val = temp_base + perm + 0.5

        y_temp = h - pad - (temp_val / max_val) * chart_h
        y_perm = h - pad - (perm / max_val) * chart_h
        y_total = h - pad - (total_val / max_val) * chart_h

        temp_points.append(f"{x:.1f},{y_temp:.1f}")
        perm_points.append(f"{x:.1f},{y_perm:.1f}")
        total_points.append(f"{x:.1f},{y_total:.1f}")

    return f"""
    <svg viewBox="0 0 {w} {h}" class="chart">
      <text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">
        Market Impact Decay
      </text>
      <text x="{w // 2}" y="{h - 5}" text-anchor="middle" class="svg-label">
        Time (seconds)
      </text>
      <polyline points="{" ".join(total_points)}" fill="none"
                stroke="#f0883e" stroke-width="2"/>
      <polyline points="{" ".join(temp_points)}" fill="none"
                stroke="#58a6ff" stroke-width="2" stroke-dasharray="5,3"/>
      <polyline points="{" ".join(perm_points)}" fill="none"
                stroke="#f85149" stroke-width="1.5" stroke-dasharray="2,4"/>
      <text x="{w - pad - 100}" y="45" class="svg-label" fill="#f0883e">Total</text>
      <text x="{w - pad - 100}" y="60" class="svg-label" fill="#58a6ff">Temporary</text>
      <text x="{w - pad - 100}" y="75" class="svg-label" fill="#f85149">Permanent</text>
    </svg>"""


def _fill_distribution_svg(fills: List[FillResult]) -> str:
    """SVG showing fill ratio distribution."""
    ratios = [f.fill_ratio for f in fills]
    return _histogram_svg(ratios, "Fill Ratio Distribution", "Fill Ratio", n_bins=20)


def _build_html(result: SimulationResult) -> str:
    s = result.summary
    fills = result.fills
    slips = [f.slippage_bps for f in fills]
    latencies = [f.latency_ms for f in fills]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Execution Simulation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
              gap: 12px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 12px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.85em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.3em; }}
  .chart {{ width: 100%; max-width: 700px; margin: 20px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 14px; }}
  .svg-label {{ fill: #8b949e; font-size: 11px; }}
  .config {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
             padding: 16px; margin: 16px 0; }}
  .config code {{ color: #f0883e; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>Trade Execution Simulation</h1>
<p class="meta">{s['n_orders']} orders simulated &middot;
   Slippage model: {result.slippage_config.model.value} &middot;
   Queue model: {result.queue_model.value}</p>

<div class="summary">
  <div class="stat"><div class="label">Avg Slippage</div>
    <div class="value">{_fmt_bps(s['avg_slippage_bps'])}</div></div>
  <div class="stat"><div class="label">P95 Slippage</div>
    <div class="value">{_fmt_bps(s['p95_slippage_bps'])}</div></div>
  <div class="stat"><div class="label">Total Cost</div>
    <div class="value">{_fmt_dollar(s['total_slippage_dollars'])}</div></div>
  <div class="stat"><div class="label">Avg Fill Ratio</div>
    <div class="value">{_fmt_pct(s['avg_fill_ratio'])}</div></div>
  <div class="stat"><div class="label">Complete Fills</div>
    <div class="value">{s['n_complete']}/{s['n_orders']}</div></div>
  <div class="stat"><div class="label">Avg Latency</div>
    <div class="value">{_fmt_ms(s['avg_latency_ms'])}</div></div>
  <div class="stat"><div class="label">Avg Temp Impact</div>
    <div class="value">{_fmt_bps(s['avg_temp_impact_bps'])}</div></div>
  <div class="stat"><div class="label">Avg Perm Impact</div>
    <div class="value">{_fmt_bps(s['avg_perm_impact_bps'])}</div></div>
</div>

<h2>Slippage Distribution</h2>
{_histogram_svg(slips, "Slippage Distribution", "Slippage (bps)")}

<h2>Fill Ratio Distribution</h2>
{_fill_distribution_svg(fills)}

<h2>Latency Distribution</h2>
{_histogram_svg(latencies, "Order-to-Fill Latency", "Latency (ms)")}

<h2>Market Impact Decay</h2>
{_impact_decay_svg(result.impact_config)}

<div class="config">
  <h3>Configuration</h3>
  <p>Slippage: <code>{result.slippage_config.model.value}</code>
     (fixed_bps={result.slippage_config.fixed_bps},
      proportional={result.slippage_config.proportional_factor},
      vol_impact={result.slippage_config.volume_impact_factor})</p>
  <p>Latency: <code>base={result.latency_config.base_latency_ms}ms,
     jitter={result.latency_config.jitter_ms}ms</code></p>
  <p>Impact: <code>temp={result.impact_config.temporary_impact_bps}bps,
     perm={result.impact_config.permanent_impact_bps}bps,
     half_life={result.impact_config.decay_half_life_seconds}s</code></p>
</div>

</body>
</html>"""
