"""Order management system – lifecycle tracking, smart routing, batching,
execution cost tracking, order replay, and kill switch for credit spread
portfolios.

Provides:
  1. Order lifecycle: pending → filled / partial / cancelled / rejected
  2. Smart order routing (limit vs market, aggression levels)
  3. Order batching and netting across experiments
  4. Execution cost tracking (commissions, exchange fees, ECN rebates)
  5. Order replay for debugging (reconstruct decision chain)
  6. Kill switch (cancel all open orders immediately)
  7. HTML report with order flow, fill rates, cost breakdown
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Enums ───────────────────────────────────────────────────────────────────
class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Aggression(str, Enum):
    PASSIVE = "passive"       # limit at bid/ask
    MODERATE = "moderate"     # limit at mid ± small offset
    AGGRESSIVE = "aggressive" # market or marketable limit


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class ExecutionCost:
    """Cost breakdown for a single fill."""
    commission: float = 0.0
    exchange_fee: float = 0.0
    ecn_rebate: float = 0.0
    slippage: float = 0.0

    @property
    def total(self) -> float:
        return self.commission + self.exchange_fee - self.ecn_rebate + self.slippage


@dataclass
class OrderEvent:
    """Single event in an order's lifecycle."""
    timestamp: str
    event_type: str       # "created", "submitted", "partial_fill", "filled", "cancelled", "rejected"
    detail: str = ""
    fill_qty: int = 0
    fill_price: float = 0.0


@dataclass
class Order:
    """A single order with full lifecycle tracking."""
    order_id: str
    experiment_id: str
    symbol: str
    side: str              # "buy" / "sell"
    order_type: str        # "limit" / "market"
    quantity: int
    limit_price: float = 0.0
    status: str = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    aggression: str = Aggression.MODERATE
    cost: ExecutionCost = field(default_factory=ExecutionCost)
    events: List[OrderEvent] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class BatchedOrder:
    """Netted order across multiple experiments."""
    symbol: str
    side: str
    net_quantity: int
    contributing_orders: List[str]   # order_ids
    experiments: List[str]


@dataclass
class CostSummary:
    """Aggregate execution cost summary."""
    total_commissions: float = 0.0
    total_exchange_fees: float = 0.0
    total_ecn_rebates: float = 0.0
    total_slippage: float = 0.0
    total_cost: float = 0.0
    n_orders: int = 0
    avg_cost_per_order: float = 0.0
    cost_by_type: Dict[str, float] = field(default_factory=dict)


@dataclass
class FillStats:
    """Fill rate and quality statistics."""
    total_orders: int = 0
    filled: int = 0
    partial: int = 0
    cancelled: int = 0
    rejected: int = 0
    pending: int = 0
    fill_rate: float = 0.0
    avg_fill_time_seconds: float = 0.0


@dataclass
class OrderManagerResult:
    """Complete order management report."""
    fill_stats: Optional[FillStats] = None
    cost_summary: Optional[CostSummary] = None
    batched_orders: List[BatchedOrder] = field(default_factory=list)
    order_type_dist: Dict[str, int] = field(default_factory=dict)
    n_orders: int = 0
    kill_switch_triggered: bool = False
    generated_at: str = ""


# ── Routing logic ───────────────────────────────────────────────────────────
@dataclass
class RoutingConfig:
    """Smart order routing configuration."""
    default_aggression: str = Aggression.MODERATE
    passive_offset_bps: float = 5.0      # limit offset from mid for passive
    moderate_offset_bps: float = 2.0     # for moderate
    max_market_order_qty: int = 50       # above this, force limit
    commission_per_contract: float = 0.65
    exchange_fee_per_contract: float = 0.30
    ecn_rebate_per_contract: float = 0.10


def route_order(
    side: str,
    quantity: int,
    mid_price: float,
    aggression: str = Aggression.MODERATE,
    config: Optional[RoutingConfig] = None,
) -> Tuple[str, float]:
    """Determine order type and limit price from routing logic.

    Returns (order_type, limit_price).
    """
    cfg = config or RoutingConfig()

    if aggression == Aggression.AGGRESSIVE or quantity <= 0:
        return (OrderType.MARKET, 0.0)

    if quantity > cfg.max_market_order_qty and aggression == Aggression.AGGRESSIVE:
        aggression = Aggression.MODERATE

    if aggression == Aggression.PASSIVE:
        offset = mid_price * cfg.passive_offset_bps / 10_000
    else:
        offset = mid_price * cfg.moderate_offset_bps / 10_000

    if side == Side.BUY:
        limit_price = mid_price - offset
    else:
        limit_price = mid_price + offset

    return (OrderType.LIMIT, round(limit_price, 2))


# ── Core manager ────────────────────────────────────────────────────────────
class OrderManager:
    """Manages order lifecycle, routing, batching, costs, and kill switch."""

    def __init__(self, routing_config: Optional[RoutingConfig] = None) -> None:
        self.config = routing_config or RoutingConfig()
        self._orders: Dict[str, Order] = {}
        self._kill_switch_active: bool = False

    # ── Order creation ──────────────────────────────────────────────────────
    def create_order(
        self,
        experiment_id: str,
        symbol: str,
        side: str,
        quantity: int,
        mid_price: float = 0.0,
        aggression: str = Aggression.MODERATE,
        tags: Optional[Dict[str, str]] = None,
    ) -> Order:
        """Create and register a new order with smart routing."""
        if self._kill_switch_active:
            order = self._make_order(experiment_id, symbol, side,
                                     OrderType.MARKET, quantity, 0.0,
                                     aggression, tags)
            order.status = OrderStatus.REJECTED
            self._add_event(order, "rejected", "Kill switch active")
            self._orders[order.order_id] = order
            return order

        order_type, limit_price = route_order(
            side, quantity, mid_price, aggression, self.config,
        )
        order = self._make_order(experiment_id, symbol, side,
                                 order_type, quantity, limit_price,
                                 aggression, tags)
        self._add_event(order, "created", f"{order_type} @ {limit_price:.2f}")
        self._orders[order.order_id] = order
        return order

    # ── Fill simulation ─────────────────────────────────────────────────────
    def fill_order(
        self,
        order_id: str,
        fill_qty: int,
        fill_price: float,
        slippage: float = 0.0,
    ) -> Order:
        """Record a fill (full or partial) on an order."""
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")
        if order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return order

        order.filled_qty += fill_qty
        # Weighted average fill price
        prev_notional = order.avg_fill_price * (order.filled_qty - fill_qty)
        order.avg_fill_price = (prev_notional + fill_price * fill_qty) / order.filled_qty

        # Costs
        order.cost.commission += fill_qty * self.config.commission_per_contract
        order.cost.exchange_fee += fill_qty * self.config.exchange_fee_per_contract
        order.cost.ecn_rebate += fill_qty * self.config.ecn_rebate_per_contract
        order.cost.slippage += slippage

        if order.filled_qty >= order.quantity:
            order.status = OrderStatus.FILLED
            self._add_event(order, "filled", f"{fill_qty}@{fill_price:.2f}",
                            fill_qty, fill_price)
        else:
            order.status = OrderStatus.PARTIAL
            self._add_event(order, "partial_fill", f"{fill_qty}@{fill_price:.2f}",
                            fill_qty, fill_price)

        order.updated_at = self._now()
        return order

    # ── Cancel ──────────────────────────────────────────────────────────────
    def cancel_order(self, order_id: str, reason: str = "") -> Order:
        """Cancel an open order."""
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")
        if order.status in (OrderStatus.FILLED, OrderStatus.REJECTED):
            return order
        order.status = OrderStatus.CANCELLED
        self._add_event(order, "cancelled", reason)
        order.updated_at = self._now()
        return order

    # ── Kill switch ─────────────────────────────────────────────────────────
    def activate_kill_switch(self) -> int:
        """Cancel ALL open orders immediately. Returns count cancelled."""
        self._kill_switch_active = True
        cancelled = 0
        for order in self._orders.values():
            if order.status in (OrderStatus.PENDING, OrderStatus.PARTIAL):
                order.status = OrderStatus.CANCELLED
                self._add_event(order, "cancelled", "Kill switch activated")
                order.updated_at = self._now()
                cancelled += 1
        logger.warning("Kill switch activated — %d orders cancelled", cancelled)
        return cancelled

    def deactivate_kill_switch(self) -> None:
        """Re-enable order creation."""
        self._kill_switch_active = False

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    # ── Batching / netting ──────────────────────────────────────────────────
    def batch_orders(self) -> List[BatchedOrder]:
        """Net pending orders across experiments by symbol+side."""
        groups: Dict[Tuple[str, str], List[Order]] = {}
        for order in self._orders.values():
            if order.status != OrderStatus.PENDING:
                continue
            key = (order.symbol, order.side)
            groups.setdefault(key, []).append(order)

        batched: List[BatchedOrder] = []
        for (symbol, side), orders in groups.items():
            net_qty = sum(o.quantity - o.filled_qty for o in orders)
            if net_qty <= 0:
                continue
            batched.append(BatchedOrder(
                symbol=symbol,
                side=side,
                net_quantity=net_qty,
                contributing_orders=[o.order_id for o in orders],
                experiments=list({o.experiment_id for o in orders}),
            ))
        return batched

    # ── Replay ──────────────────────────────────────────────────────────────
    def replay_order(self, order_id: str) -> List[OrderEvent]:
        """Reconstruct the decision chain for an order."""
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")
        return list(order.events)

    # ── Query ───────────────────────────────────────────────────────────────
    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        if status is None:
            return list(self._orders.values())
        return [o for o in self._orders.values() if o.status == status]

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values()
                if o.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)]

    # ── Reporting ───────────────────────────────────────────────────────────
    def summary(self) -> OrderManagerResult:
        """Generate full order management summary."""
        orders = list(self._orders.values())
        if not orders:
            return OrderManagerResult(generated_at=self._now())

        fill_stats = self._compute_fill_stats(orders)
        cost_summary = self._compute_cost_summary(orders)
        batched = self.batch_orders()
        type_dist = self._order_type_distribution(orders)

        return OrderManagerResult(
            fill_stats=fill_stats,
            cost_summary=cost_summary,
            batched_orders=batched,
            order_type_dist=type_dist,
            n_orders=len(orders),
            kill_switch_triggered=self._kill_switch_active,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: Optional[OrderManagerResult] = None,
        output_path: str | Path = "reports/order_management.html",
    ) -> Path:
        """Write self-contained HTML report."""
        if result is None:
            result = self.summary()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Order management report written to %s", path)
        return path

    # ── Internal helpers ────────────────────────────────────────────────────
    @staticmethod
    def _make_order(
        experiment_id: str, symbol: str, side: str,
        order_type: str, quantity: int, limit_price: float,
        aggression: str, tags: Optional[Dict[str, str]],
    ) -> Order:
        now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        return Order(
            order_id=uuid.uuid4().hex[:12],
            experiment_id=experiment_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            aggression=aggression,
            created_at=now,
            updated_at=now,
            tags=tags or {},
        )

    @staticmethod
    def _add_event(
        order: Order, event_type: str, detail: str = "",
        fill_qty: int = 0, fill_price: float = 0.0,
    ) -> None:
        order.events.append(OrderEvent(
            timestamp=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            event_type=event_type,
            detail=detail,
            fill_qty=fill_qty,
            fill_price=fill_price,
        ))

    @staticmethod
    def _compute_fill_stats(orders: List[Order]) -> FillStats:
        n = len(orders)
        filled = sum(1 for o in orders if o.status == OrderStatus.FILLED)
        partial = sum(1 for o in orders if o.status == OrderStatus.PARTIAL)
        cancelled = sum(1 for o in orders if o.status == OrderStatus.CANCELLED)
        rejected = sum(1 for o in orders if o.status == OrderStatus.REJECTED)
        pending = sum(1 for o in orders if o.status == OrderStatus.PENDING)
        fill_rate = (filled + partial) / n if n > 0 else 0.0
        return FillStats(
            total_orders=n, filled=filled, partial=partial,
            cancelled=cancelled, rejected=rejected, pending=pending,
            fill_rate=fill_rate,
        )

    @staticmethod
    def _compute_cost_summary(orders: List[Order]) -> CostSummary:
        filled_orders = [o for o in orders if o.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)]
        if not filled_orders:
            return CostSummary(n_orders=len(orders))

        total_comm = sum(o.cost.commission for o in filled_orders)
        total_exch = sum(o.cost.exchange_fee for o in filled_orders)
        total_rebate = sum(o.cost.ecn_rebate for o in filled_orders)
        total_slip = sum(o.cost.slippage for o in filled_orders)
        total = total_comm + total_exch - total_rebate + total_slip

        # Cost by order type
        by_type: Dict[str, float] = {}
        for o in filled_orders:
            by_type[o.order_type] = by_type.get(o.order_type, 0.0) + o.cost.total

        return CostSummary(
            total_commissions=total_comm,
            total_exchange_fees=total_exch,
            total_ecn_rebates=total_rebate,
            total_slippage=total_slip,
            total_cost=total,
            n_orders=len(filled_orders),
            avg_cost_per_order=total / len(filled_orders) if filled_orders else 0.0,
            cost_by_type=by_type,
        )

    @staticmethod
    def _order_type_distribution(orders: List[Order]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for o in orders:
            dist[o.order_type] = dist.get(o.order_type, 0) + 1
        return dist

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: OrderManagerResult) -> str:
        cards = self._html_cards(r)
        fill_tbl = self._html_fill_stats(r.fill_stats)
        cost_tbl = self._html_cost_summary(r.cost_summary)
        type_tbl = self._html_type_dist(r.order_type_dist)
        timeline = self._html_timeline(list(self._orders.values())[:50])
        batch_tbl = self._html_batched(r.batched_orders)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Order Management</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}.warn{{color:#fbbf24}}
.kill{{color:#f87171;font-weight:700}}
</style>
</head>
<body>
<h1>Order Management</h1>
<p class="sub">Generated {r.generated_at or 'N/A'}{' <span class="kill">KILL SWITCH ACTIVE</span>' if r.kill_switch_triggered else ''}</p>

{cards}
{fill_tbl}
{cost_tbl}
{type_tbl}
{batch_tbl}
{timeline}

</body>
</html>"""

    @staticmethod
    def _html_cards(r: OrderManagerResult) -> str:
        fs = r.fill_stats
        cs = r.cost_summary
        fill_r = f"{fs.fill_rate:.0%}" if fs else "N/A"
        total_c = f"${cs.total_cost:,.2f}" if cs else "$0"
        return f"""<div class="grid">
<div class="card"><div class="lbl">Total Orders</div><div class="val">{r.n_orders}</div></div>
<div class="card"><div class="lbl">Fill Rate</div><div class="val">{fill_r}</div></div>
<div class="card"><div class="lbl">Total Cost</div><div class="val">{total_c}</div></div>
<div class="card"><div class="lbl">Batched</div><div class="val">{len(r.batched_orders)}</div></div>
<div class="card"><div class="lbl">Kill Switch</div><div class="val {'kill' if r.kill_switch_triggered else ''}">{'ON' if r.kill_switch_triggered else 'OFF'}</div></div>
</div>"""

    @staticmethod
    def _html_fill_stats(fs: Optional[FillStats]) -> str:
        if not fs:
            return ""
        return f"""<div class="sec">
<h2>Fill Rate Dashboard</h2>
<table>
<thead><tr><th>Status</th><th>Count</th><th>%</th></tr></thead>
<tbody>
<tr><td class="pos">Filled</td><td>{fs.filled}</td><td>{fs.filled/fs.total_orders:.0%}</td></tr>
<tr><td class="warn">Partial</td><td>{fs.partial}</td><td>{fs.partial/fs.total_orders:.0%}</td></tr>
<tr><td>Pending</td><td>{fs.pending}</td><td>{fs.pending/fs.total_orders:.0%}</td></tr>
<tr><td class="neg">Cancelled</td><td>{fs.cancelled}</td><td>{fs.cancelled/fs.total_orders:.0%}</td></tr>
<tr><td class="neg">Rejected</td><td>{fs.rejected}</td><td>{fs.rejected/fs.total_orders:.0%}</td></tr>
</tbody>
</table>
</div>""" if fs.total_orders > 0 else ""

    @staticmethod
    def _html_cost_summary(cs: Optional[CostSummary]) -> str:
        if not cs or cs.n_orders == 0:
            return ""
        return f"""<div class="sec">
<h2>Execution Cost Breakdown</h2>
<table>
<thead><tr><th>Component</th><th>Amount</th></tr></thead>
<tbody>
<tr><td>Commissions</td><td>${cs.total_commissions:,.2f}</td></tr>
<tr><td>Exchange Fees</td><td>${cs.total_exchange_fees:,.2f}</td></tr>
<tr><td>ECN Rebates</td><td class="pos">-${cs.total_ecn_rebates:,.2f}</td></tr>
<tr><td>Slippage</td><td>${cs.total_slippage:,.2f}</td></tr>
<tr><td><strong>Total</strong></td><td><strong>${cs.total_cost:,.2f}</strong></td></tr>
<tr><td>Avg / Order</td><td>${cs.avg_cost_per_order:,.2f}</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_type_dist(dist: Dict[str, int]) -> str:
        if not dist:
            return ""
        rows = "".join(f"<tr><td>{t}</td><td>{c}</td></tr>" for t, c in sorted(dist.items()))
        return f"""<div class="sec">
<h2>Order Type Distribution</h2>
<table><thead><tr><th>Type</th><th>Count</th></tr></thead><tbody>{rows}</tbody></table>
</div>"""

    @staticmethod
    def _html_batched(batched: List[BatchedOrder]) -> str:
        if not batched:
            return ""
        rows = ""
        for b in batched:
            rows += (
                f"<tr><td>{b.symbol}</td><td>{b.side}</td>"
                f"<td>{b.net_quantity}</td>"
                f"<td>{len(b.contributing_orders)}</td>"
                f"<td>{', '.join(b.experiments)}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Batched / Netted Orders</h2>
<table>
<thead><tr><th>Symbol</th><th>Side</th><th>Net Qty</th><th>Orders</th><th>Experiments</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_timeline(orders: List[Order]) -> str:
        if not orders:
            return ""
        rows = ""
        for o in orders:
            st_cls = "pos" if o.status == OrderStatus.FILLED else "neg" if o.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED) else "warn"
            rows += (
                f'<tr><td>{o.order_id[:8]}</td>'
                f"<td>{o.experiment_id}</td>"
                f"<td>{o.symbol}</td>"
                f"<td>{o.side}</td>"
                f"<td>{o.order_type}</td>"
                f"<td>{o.quantity}</td>"
                f"<td>{o.filled_qty}</td>"
                f'<td class="{st_cls}">{o.status}</td>'
                f"<td>${o.cost.total:.2f}</td>"
                f"<td>{o.created_at}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Order Flow Timeline</h2>
<table>
<thead><tr><th>ID</th><th>Experiment</th><th>Symbol</th><th>Side</th><th>Type</th><th>Qty</th><th>Filled</th><th>Status</th><th>Cost</th><th>Created</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
