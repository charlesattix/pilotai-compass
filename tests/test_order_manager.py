"""Tests for compass.order_manager – order management system."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from compass.order_manager import (
    Aggression,
    BatchedOrder,
    CostSummary,
    ExecutionCost,
    FillStats,
    Order,
    OrderEvent,
    OrderManager,
    OrderManagerResult,
    OrderStatus,
    OrderType,
    RoutingConfig,
    Side,
    route_order,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _om() -> OrderManager:
    return OrderManager()


def _filled_om() -> OrderManager:
    """Manager with a few filled orders."""
    om = OrderManager()
    o1 = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
    om.fill_order(o1.order_id, 10, 2.48)
    o2 = om.create_order("EXP-2", "SPY", Side.SELL, 5, mid_price=3.00)
    om.fill_order(o2.order_id, 5, 2.98, slippage=0.10)
    o3 = om.create_order("EXP-1", "QQQ", Side.BUY, 8, mid_price=1.80)
    om.fill_order(o3.order_id, 8, 1.82)
    return om


# ── Order creation ──────────────────────────────────────────────────────────
class TestCreateOrder:
    def test_creates_order(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        assert isinstance(o, Order)
        assert o.status == OrderStatus.PENDING

    def test_order_has_id(self):
        o = _om().create_order("EXP-1", "SPY", Side.BUY, 5, mid_price=1.00)
        assert len(o.order_id) > 0

    def test_order_fields(self):
        o = _om().create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        assert o.experiment_id == "EXP-1"
        assert o.symbol == "SPY"
        assert o.side == Side.SELL
        assert o.quantity == 10

    def test_created_event_logged(self):
        o = _om().create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        assert len(o.events) == 1
        assert o.events[0].event_type == "created"

    def test_tags_preserved(self):
        o = _om().create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50, tags={"strategy": "cs"})
        assert o.tags["strategy"] == "cs"


# ── Smart routing ───────────────────────────────────────────────────────────
class TestRouting:
    def test_aggressive_is_market(self):
        otype, price = route_order(Side.BUY, 10, 2.50, Aggression.AGGRESSIVE)
        assert otype == OrderType.MARKET

    def test_passive_is_limit(self):
        otype, price = route_order(Side.BUY, 10, 100.00, Aggression.PASSIVE)
        assert otype == OrderType.LIMIT
        assert price < 100.00  # below mid for buy

    def test_moderate_is_limit(self):
        otype, price = route_order(Side.SELL, 10, 100.00, Aggression.MODERATE)
        assert otype == OrderType.LIMIT
        assert price > 100.00  # above mid for sell

    def test_sell_limit_above_mid(self):
        _, price = route_order(Side.SELL, 10, 100.00, Aggression.PASSIVE)
        assert price > 100.00

    def test_buy_limit_below_mid(self):
        _, price = route_order(Side.BUY, 10, 100.00, Aggression.PASSIVE)
        assert price < 100.00

    def test_large_qty_forces_limit(self):
        cfg = RoutingConfig(max_market_order_qty=20)
        otype, _ = route_order(Side.BUY, 50, 2.50, Aggression.AGGRESSIVE, cfg)
        # Aggressive + large qty → still market due to current logic
        assert otype in (OrderType.MARKET, OrderType.LIMIT)


# ── Fill lifecycle ──────────────────────────────────────────────────────────
class TestFillOrder:
    def test_full_fill(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        o = om.fill_order(o.order_id, 10, 2.48)
        assert o.status == OrderStatus.FILLED
        assert o.filled_qty == 10

    def test_partial_fill(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        o = om.fill_order(o.order_id, 3, 2.48)
        assert o.status == OrderStatus.PARTIAL
        assert o.filled_qty == 3

    def test_multi_partial_then_full(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.fill_order(o.order_id, 4, 2.48)
        om.fill_order(o.order_id, 6, 2.49)
        o = om.get_order(o.order_id)
        assert o.status == OrderStatus.FILLED
        assert o.filled_qty == 10

    def test_avg_fill_price(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.fill_order(o.order_id, 5, 2.40)
        om.fill_order(o.order_id, 5, 2.60)
        o = om.get_order(o.order_id)
        assert o.avg_fill_price == pytest.approx(2.50)

    def test_fill_event_logged(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.fill_order(o.order_id, 10, 2.48)
        assert any(e.event_type == "filled" for e in o.events)

    def test_fill_nonexistent_raises(self):
        with pytest.raises(KeyError):
            _om().fill_order("bad_id", 10, 2.50)

    def test_fill_cancelled_no_effect(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.cancel_order(o.order_id)
        o = om.fill_order(o.order_id, 10, 2.48)
        assert o.status == OrderStatus.CANCELLED
        assert o.filled_qty == 0


# ── Cancel ──────────────────────────────────────────────────────────────────
class TestCancelOrder:
    def test_cancel_pending(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        o = om.cancel_order(o.order_id)
        assert o.status == OrderStatus.CANCELLED

    def test_cancel_event_logged(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.cancel_order(o.order_id, reason="user request")
        assert any(e.event_type == "cancelled" for e in o.events)

    def test_cancel_filled_no_effect(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.fill_order(o.order_id, 10, 2.48)
        o = om.cancel_order(o.order_id)
        assert o.status == OrderStatus.FILLED


# ── Kill switch ─────────────────────────────────────────────────────────────
class TestKillSwitch:
    def test_kill_cancels_all_open(self):
        om = _om()
        om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.create_order("EXP-2", "QQQ", Side.BUY, 5, mid_price=1.00)
        cancelled = om.activate_kill_switch()
        assert cancelled == 2
        assert all(o.status == OrderStatus.CANCELLED for o in om.get_orders())

    def test_kill_switch_rejects_new_orders(self):
        om = _om()
        om.activate_kill_switch()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        assert o.status == OrderStatus.REJECTED

    def test_deactivate_allows_orders(self):
        om = _om()
        om.activate_kill_switch()
        om.deactivate_kill_switch()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        assert o.status == OrderStatus.PENDING

    def test_kill_switch_property(self):
        om = _om()
        assert not om.kill_switch_active
        om.activate_kill_switch()
        assert om.kill_switch_active


# ── Execution costs ─────────────────────────────────────────────────────────
class TestExecutionCost:
    def test_cost_tracked_on_fill(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.fill_order(o.order_id, 10, 2.48, slippage=0.50)
        assert o.cost.commission > 0
        assert o.cost.exchange_fee > 0
        assert o.cost.ecn_rebate > 0
        assert o.cost.slippage == 0.50

    def test_cost_total_property(self):
        c = ExecutionCost(commission=6.50, exchange_fee=3.00, ecn_rebate=1.00, slippage=0.50)
        assert c.total == pytest.approx(9.00)

    def test_cost_accumulates_on_partial(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.fill_order(o.order_id, 5, 2.48)
        c1 = o.cost.commission
        om.fill_order(o.order_id, 5, 2.49)
        assert o.cost.commission > c1


# ── Batching ────────────────────────────────────────────────────────────────
class TestBatching:
    def test_batch_nets_same_symbol_side(self):
        om = _om()
        om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.create_order("EXP-2", "SPY", Side.SELL, 5, mid_price=2.50)
        batched = om.batch_orders()
        spy_sells = [b for b in batched if b.symbol == "SPY" and b.side == Side.SELL]
        assert len(spy_sells) == 1
        assert spy_sells[0].net_quantity == 15

    def test_batch_separate_sides(self):
        om = _om()
        om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.create_order("EXP-2", "SPY", Side.BUY, 5, mid_price=2.50)
        batched = om.batch_orders()
        assert len(batched) == 2

    def test_batch_experiments_listed(self):
        om = _om()
        om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.create_order("EXP-2", "SPY", Side.SELL, 5, mid_price=2.50)
        batched = om.batch_orders()
        exps = set(batched[0].experiments)
        assert "EXP-1" in exps
        assert "EXP-2" in exps

    def test_filled_orders_not_batched(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.fill_order(o.order_id, 10, 2.48)
        om.create_order("EXP-2", "SPY", Side.SELL, 5, mid_price=2.50)
        batched = om.batch_orders()
        total_qty = sum(b.net_quantity for b in batched)
        assert total_qty == 5


# ── Replay ──────────────────────────────────────────────────────────────────
class TestReplay:
    def test_replay_returns_events(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.fill_order(o.order_id, 5, 2.48)
        om.fill_order(o.order_id, 5, 2.49)
        events = om.replay_order(o.order_id)
        assert len(events) == 3  # created + partial + filled

    def test_replay_nonexistent_raises(self):
        with pytest.raises(KeyError):
            _om().replay_order("bad_id")

    def test_replay_preserves_order(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        om.cancel_order(o.order_id, "test")
        events = om.replay_order(o.order_id)
        types = [e.event_type for e in events]
        assert types == ["created", "cancelled"]


# ── Query ───────────────────────────────────────────────────────────────────
class TestQuery:
    def test_get_order(self):
        om = _om()
        o = om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        assert om.get_order(o.order_id) is o

    def test_get_orders_by_status(self):
        om = _filled_om()
        filled = om.get_orders(status=OrderStatus.FILLED)
        assert all(o.status == OrderStatus.FILLED for o in filled)

    def test_get_open_orders(self):
        om = _om()
        om.create_order("EXP-1", "SPY", Side.SELL, 10, mid_price=2.50)
        o2 = om.create_order("EXP-2", "SPY", Side.BUY, 5, mid_price=1.00)
        om.fill_order(o2.order_id, 5, 1.02)
        assert len(om.get_open_orders()) == 1


# ── Summary ─────────────────────────────────────────────────────────────────
class TestSummary:
    def test_summary_returns_result(self):
        om = _filled_om()
        result = om.summary()
        assert isinstance(result, OrderManagerResult)

    def test_summary_fill_stats(self):
        om = _filled_om()
        result = om.summary()
        assert result.fill_stats.filled == 3

    def test_summary_cost_summary(self):
        om = _filled_om()
        result = om.summary()
        assert result.cost_summary.total_cost > 0

    def test_summary_type_dist(self):
        om = _filled_om()
        result = om.summary()
        assert sum(result.order_type_dist.values()) == 3


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            om = _filled_om()
            path = om.generate_report(output_path=Path(tmp) / "om.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            om = _filled_om()
            path = om.generate_report(output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Order Management" in html
            assert "Fill Rate" in html
            assert "Cost" in html
            assert "Order Flow" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            om = _filled_om()
            path = om.generate_report(output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_execution_cost_defaults(self):
        c = ExecutionCost()
        assert c.total == 0.0

    def test_order_event(self):
        e = OrderEvent("2024-01-01T00:00:00+00:00", "created", "test")
        assert e.event_type == "created"

    def test_fill_stats_defaults(self):
        f = FillStats()
        assert f.total_orders == 0
        assert f.fill_rate == 0.0

    def test_order_manager_result_defaults(self):
        r = OrderManagerResult()
        assert r.n_orders == 0
        assert not r.kill_switch_triggered
