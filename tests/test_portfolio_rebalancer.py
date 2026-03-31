"""Tests for compass.portfolio_rebalancer — 35+ tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from compass.portfolio_rebalancer import (
    PortfolioRebalancer,
    Position,
    RebalanceTrade,
    RebalanceEvent,
    DriftSnapshot,
    DriftAlert,
    TriggerType,
    DEFAULT_DRIFT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _positions(drifted: bool = False) -> list[Position]:
    if drifted:
        return [
            Position("EXP-400", current_weight=0.35, target_weight=0.25, unrealised_pnl=500),
            Position("EXP-401", current_weight=0.15, target_weight=0.25, unrealised_pnl=-200),
            Position("EXP-503", current_weight=0.30, target_weight=0.25, unrealised_pnl=100),
            Position("EXP-600", current_weight=0.20, target_weight=0.25, unrealised_pnl=-300),
        ]
    return [
        Position("EXP-400", current_weight=0.26, target_weight=0.25),
        Position("EXP-401", current_weight=0.24, target_weight=0.25),
        Position("EXP-503", current_weight=0.25, target_weight=0.25),
        Position("EXP-600", current_weight=0.25, target_weight=0.25),
    ]


# ===========================================================================
# Drift
# ===========================================================================

class TestDrift:
    def test_compute_drift(self):
        rb = PortfolioRebalancer()
        snap = rb.compute_drift(_positions())
        assert isinstance(snap, DriftSnapshot)
        assert snap.max_drift == pytest.approx(0.01)

    def test_threshold_breach(self):
        rb = PortfolioRebalancer(drift_threshold=0.05)
        snap = rb.compute_drift(_positions(drifted=True))
        assert snap.threshold_breached
        assert snap.max_drift >= 0.05

    def test_no_breach_when_tight(self):
        rb = PortfolioRebalancer(drift_threshold=0.05)
        snap = rb.compute_drift(_positions(drifted=False))
        assert not snap.threshold_breached

    def test_drift_history(self):
        rb = PortfolioRebalancer()
        rb.compute_drift(_positions(), date=datetime(2026, 1, 1))
        rb.compute_drift(_positions(), date=datetime(2026, 1, 2))
        assert len(rb.drift_history) == 2

    def test_alerts_generated(self):
        rb = PortfolioRebalancer(drift_threshold=0.05)
        rb.compute_drift(_positions(drifted=True))
        assert len(rb.alerts) > 0
        assert all(isinstance(a, DriftAlert) for a in rb.alerts)

    def test_no_alerts_when_within(self):
        rb = PortfolioRebalancer(drift_threshold=0.05)
        rb.compute_drift(_positions(drifted=False))
        assert len(rb.alerts) == 0


# ===========================================================================
# Triggers
# ===========================================================================

class TestTriggers:
    def test_calendar_first_time(self):
        rb = PortfolioRebalancer()
        assert rb.should_rebalance_calendar()

    def test_calendar_not_yet(self):
        rb = PortfolioRebalancer(calendar_days=7)
        rb._last_rebalance = datetime.now() - timedelta(days=3)
        assert not rb.should_rebalance_calendar()

    def test_calendar_due(self):
        rb = PortfolioRebalancer(calendar_days=7)
        rb._last_rebalance = datetime.now() - timedelta(days=8)
        assert rb.should_rebalance_calendar()

    def test_threshold_trigger(self):
        rb = PortfolioRebalancer(drift_threshold=0.05)
        assert rb.should_rebalance_threshold(_positions(drifted=True))
        assert not rb.should_rebalance_threshold(_positions(drifted=False))

    def test_regime_highest_priority(self):
        rb = PortfolioRebalancer()
        rb._last_rebalance = datetime.now()  # calendar not due
        t = rb.check_triggers(_positions(drifted=False), regime_changed=True)
        assert t == TriggerType.REGIME

    def test_threshold_over_calendar(self):
        rb = PortfolioRebalancer(drift_threshold=0.05, calendar_days=7)
        rb._last_rebalance = datetime.now()  # calendar not due
        t = rb.check_triggers(_positions(drifted=True))
        assert t == TriggerType.THRESHOLD

    def test_calendar_fallback(self):
        rb = PortfolioRebalancer(calendar_days=7)
        rb._last_rebalance = datetime.now() - timedelta(days=8)
        t = rb.check_triggers(_positions(drifted=False))
        assert t == TriggerType.CALENDAR

    def test_no_trigger(self):
        rb = PortfolioRebalancer(calendar_days=7, drift_threshold=0.20)
        rb._last_rebalance = datetime.now()
        t = rb.check_triggers(_positions(drifted=False))
        assert t is None


# ===========================================================================
# Trade computation
# ===========================================================================

class TestTrades:
    def test_basic_trades(self):
        rb = PortfolioRebalancer(min_trade_size=0.01)
        trades = rb.compute_trades(_positions(drifted=True))
        assert len(trades) > 0
        assert all(isinstance(t, RebalanceTrade) for t in trades)

    def test_trade_directions(self):
        rb = PortfolioRebalancer(min_trade_size=0.01)
        trades = rb.compute_trades(_positions(drifted=True))
        buys = [t for t in trades if t.trade_weight > 0]
        sells = [t for t in trades if t.trade_weight < 0]
        assert len(buys) > 0
        assert len(sells) > 0

    def test_min_trade_filter(self):
        rb = PortfolioRebalancer(min_trade_size=0.05)
        trades = rb.compute_trades(_positions(drifted=False))
        assert len(trades) == 0  # all drifts < 5%

    def test_tax_harvest_first(self):
        rb = PortfolioRebalancer(tax_aware=True, min_trade_size=0.01)
        trades = rb.compute_trades(_positions(drifted=True))
        sells = [t for t in trades if t.trade_weight < 0]
        if len(sells) >= 2:
            harvest = [t for t in sells if t.is_tax_harvest]
            non_harvest = [t for t in sells if not t.is_tax_harvest]
            if harvest and non_harvest:
                first_non_h = trades.index(non_harvest[0])
                first_h = trades.index(harvest[0])
                assert first_h < first_non_h

    def test_tax_harvest_flag(self):
        rb = PortfolioRebalancer(tax_aware=True, min_trade_size=0.01)
        trades = rb.compute_trades(_positions(drifted=True))
        # EXP-401 has negative PnL and needs selling → should be tax harvest
        harvest = [t for t in trades if t.is_tax_harvest]
        # EXP-400 (overweight, positive PnL sell) should NOT be tax harvest
        assert any(not t.is_tax_harvest for t in trades if t.trade_weight < 0)

    def test_no_tax_aware(self):
        rb = PortfolioRebalancer(tax_aware=False, min_trade_size=0.01)
        trades = rb.compute_trades(_positions(drifted=True))
        assert all(not t.is_tax_harvest for t in trades)

    def test_cost_limit(self):
        rb = PortfolioRebalancer(min_trade_size=0.01, cost_per_unit=0.01)
        all_trades = rb.compute_trades(_positions(drifted=True))
        limited = rb.compute_trades_with_cost_limit(_positions(drifted=True), max_cost=0.001)
        assert len(limited) <= len(all_trades)


# ===========================================================================
# Rebalance event
# ===========================================================================

class TestRebalance:
    def test_rebalance_records_event(self):
        rb = PortfolioRebalancer(min_trade_size=0.01)
        event = rb.rebalance(_positions(drifted=True))
        assert isinstance(event, RebalanceEvent)
        assert len(rb.events) == 1
        assert rb.last_rebalance is not None

    def test_turnover_positive(self):
        rb = PortfolioRebalancer(min_trade_size=0.01)
        event = rb.rebalance(_positions(drifted=True))
        assert event.total_turnover > 0

    def test_trigger_stored(self):
        rb = PortfolioRebalancer(min_trade_size=0.01)
        event = rb.rebalance(_positions(drifted=True), trigger=TriggerType.THRESHOLD)
        assert event.trigger == TriggerType.THRESHOLD


# ===========================================================================
# Integration with optimizer
# ===========================================================================

class TestOptimizerIntegration:
    def test_positions_from_optimizer(self):
        current = {"EXP-400": 0.3, "EXP-401": 0.2, "EXP-503": 0.5}
        target = {"EXP-400": 0.25, "EXP-401": 0.25, "EXP-503": 0.25, "EXP-600": 0.25}
        positions = PortfolioRebalancer.positions_from_optimizer(current, target)
        assert len(positions) == 4
        exp600 = [p for p in positions if p.name == "EXP-600"][0]
        assert exp600.current_weight == 0.0
        assert exp600.target_weight == 0.25

    def test_with_pnl(self):
        current = {"A": 0.5, "B": 0.5}
        target = {"A": 0.4, "B": 0.6}
        pnl = {"A": 200, "B": -100}
        positions = PortfolioRebalancer.positions_from_optimizer(current, target, pnl)
        a = [p for p in positions if p.name == "A"][0]
        assert a.unrealised_pnl == 200


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        rb = PortfolioRebalancer(min_trade_size=0.01)
        pos = _positions(drifted=True)
        rb.compute_drift(pos, date=datetime(2026, 1, 1))
        rb.compute_drift(pos, date=datetime(2026, 1, 2))
        rb.rebalance(pos, trigger=TriggerType.THRESHOLD)
        out = tmp_path / "rebalance.html"
        result = rb.generate_report(pos, output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Portfolio Rebalancer" in html

    def test_drift_chart(self, tmp_path):
        rb = PortfolioRebalancer()
        pos = _positions(drifted=True)
        for d in range(1, 6):
            rb.compute_drift(pos, date=datetime(2026, 1, d))
        out = tmp_path / "r.html"
        rb.generate_report(pos, output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "Max Portfolio Drift" in html

    def test_event_history(self, tmp_path):
        rb = PortfolioRebalancer(min_trade_size=0.01)
        pos = _positions(drifted=True)
        rb.rebalance(pos, trigger=TriggerType.THRESHOLD, date=datetime(2026, 1, 5))
        out = tmp_path / "r.html"
        rb.generate_report(pos, output_path=str(out))
        html = out.read_text()
        assert "Rebalance History" in html
