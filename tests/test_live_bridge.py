"""Tests for compass/live_bridge.py — live trading bridge.

Covers:
  - Dataclass construction
  - Signal ingestion and persistence
  - Position reconciliation
  - Order generation with risk checks
  - Risk check: confidence, max position, drawdown, daily loss, rate limit
  - Order execution (dry-run and mock broker)
  - Full run_cycle
  - P&L tracking and history
  - Health monitoring / heartbeat
  - State persistence (get/set)
  - Alerting protocol
  - from-DB retrieval (signals, orders, pnl)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from compass.live_bridge import (
    AlerterProtocol,
    BrokerProtocol,
    HealthStatus,
    LiveBridge,
    Order,
    PnLSnapshot,
    Position,
    RiskCheckResult,
    RiskLimits,
    Signal,
)


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture
def bridge():
    b = LiveBridge(db_path=":memory:", dry_run=True)
    yield b
    b.close()


@pytest.fixture
def bridge_with_signal(bridge):
    bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.85)
    return bridge


class MockBroker(BrokerProtocol):
    def __init__(self, positions=None, fill=True):
        self._positions = positions or []
        self._fill = fill

    def get_positions(self):
        return self._positions

    def submit_order(self, order):
        order.status = "filled" if self._fill else "rejected"
        return order

    def get_account_value(self):
        return 100_000.0


class MockAlerter(AlerterProtocol):
    def __init__(self):
        self.messages = []

    def send(self, message, level="info"):
        self.messages.append((level, message))
        return True


# ── Dataclass tests ──────────────────────────────────────────────────────


class TestDataclasses:
    def test_signal_fields(self):
        s = Signal("EXP-400", "SPY", 1, 2, 0.85, "2024-01-01", {})
        assert s.direction == 1

    def test_position_fields(self):
        p = Position("SPY", 2, 430.0, 100.0, "EXP-400")
        assert p.contracts == 2

    def test_order_fields(self):
        o = Order("ORD-1", "EXP-400", "SPY", "buy", 2, "market",
                  None, "twap", "pending", "", "2024-01-01")
        assert o.status == "pending"

    def test_risk_check_result(self):
        r = RiskCheckResult(True, {"max_pos": True}, [])
        assert r.passed is True

    def test_health_status(self):
        h = HealthStatus(True, "2024-01-01", 100.0, 5, 3, 0, True)
        assert h.dry_run is True

    def test_pnl_snapshot(self):
        p = PnLSnapshot("2024-01-01", 500, 200, 300, 0.02, 3)
        assert p.total_pnl == pytest.approx(500)

    def test_risk_limits_defaults(self):
        rl = RiskLimits()
        assert rl.max_position_per_asset == 10


# ── Signal ingestion ─────────────────────────────────────────────────────


class TestSignalIngestion:
    def test_ingest_returns_signal(self, bridge):
        sig = bridge.ingest_signal("EXP-400", "SPY", 1, 2)
        assert isinstance(sig, Signal)
        assert sig.strategy == "EXP-400"

    def test_signal_persisted(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 2)
        signals = bridge.get_signals()
        assert len(signals) == 1
        assert signals[0].asset == "SPY"

    def test_multiple_signals(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 2)
        bridge.ingest_signal("EXP-401", "QQQ", -1, 3)
        assert len(bridge.get_signals()) == 2

    def test_signal_counter(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 2)
        bridge.ingest_signal("EXP-400", "SPY", 1, 3)
        assert bridge._signals_processed == 2

    def test_direction_clamped(self, bridge):
        sig = bridge.ingest_signal("EXP-400", "SPY", 5, 2)
        assert sig.direction == 1


# ── Reconciliation ───────────────────────────────────────────────────────


class TestReconciliation:
    def test_reconcile_with_no_positions(self, bridge_with_signal):
        deltas = bridge_with_signal.reconcile()
        assert ("SPY", "EXP-400") in deltas
        assert deltas[("SPY", "EXP-400")] == 2

    def test_reconcile_no_delta_when_matched(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 2)
        # Simulate filled position
        bridge._positions[("SPY", "EXP-400")] = Position("SPY", 2, 430, 0, "EXP-400")
        broker = MockBroker(positions=[Position("SPY", 2, 430, 0, "EXP-400")])
        bridge.broker = broker
        deltas = bridge.reconcile()
        assert len(deltas) == 0

    def test_reconcile_sell_to_reduce(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 1)
        bridge._positions[("SPY", "EXP-400")] = Position("SPY", 3, 430, 0, "EXP-400")
        broker = MockBroker(positions=[Position("SPY", 3, 430, 0, "EXP-400")])
        bridge.broker = broker
        deltas = bridge.reconcile()
        assert deltas[("SPY", "EXP-400")] == -2


# ── Risk checks ──────────────────────────────────────────────────────────


class TestRiskChecks:
    def test_passes_normal(self, bridge):
        result = bridge._risk_check("SPY", "EXP-400", "buy", 2, 0.85)
        assert result.passed is True
        assert len(result.violations) == 0

    def test_fails_low_confidence(self, bridge):
        result = bridge._risk_check("SPY", "EXP-400", "buy", 2, 0.1)
        assert result.passed is False
        assert "Confidence" in result.violations[0]

    def test_fails_max_position(self, bridge):
        bridge.risk_limits.max_position_per_asset = 3
        bridge._positions[("SPY", "EXP-400")] = Position("SPY", 3, 430, 0, "EXP-400")
        result = bridge._risk_check("SPY", "EXP-400", "buy", 2, 0.85)
        assert not result.passed
        assert any("Position" in v for v in result.violations)

    def test_fails_max_total(self, bridge):
        bridge.risk_limits.max_total_position = 5
        for i in range(5):
            bridge._positions[(f"A{i}", "S")] = Position(f"A{i}", 1, 100, 0, "S")
        result = bridge._risk_check("SPY", "EXP-400", "buy", 2, 0.85)
        assert not result.passed
        assert any("Total" in v for v in result.violations)

    def test_fails_drawdown(self, bridge):
        bridge._max_drawdown = 0.10
        bridge.risk_limits.max_drawdown_pct = 0.05
        result = bridge._risk_check("SPY", "EXP-400", "buy", 2, 0.85)
        assert not result.passed
        assert any("Drawdown" in v for v in result.violations)

    def test_fails_daily_loss(self, bridge):
        bridge._realised_pnl = -6000
        bridge.risk_limits.max_daily_loss = 5000
        result = bridge._risk_check("SPY", "EXP-400", "buy", 2, 0.85)
        assert not result.passed
        assert any("Daily loss" in v for v in result.violations)

    def test_rate_limit(self, bridge):
        bridge.risk_limits.max_orders_per_minute = 2
        bridge._risk_check("SPY", "A", "buy", 1, 0.9)
        bridge._risk_check("SPY", "B", "buy", 1, 0.9)
        result = bridge._risk_check("SPY", "C", "buy", 1, 0.9)
        assert not result.passed
        assert any("rate" in v.lower() for v in result.violations)


# ── Order generation ─────────────────────────────────────────────────────


class TestOrderGeneration:
    def test_generates_orders(self, bridge_with_signal):
        deltas = bridge_with_signal.reconcile()
        orders = bridge_with_signal.generate_orders(deltas)
        assert len(orders) == 1
        assert orders[0].side == "buy"
        assert orders[0].contracts == 2

    def test_rejects_low_confidence(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.1)
        deltas = bridge.reconcile()
        orders = bridge.generate_orders(deltas)
        assert len(orders) == 1
        assert orders[0].status == "rejected"

    def test_order_has_algo(self, bridge_with_signal):
        deltas = bridge_with_signal.reconcile()
        orders = bridge_with_signal.generate_orders(deltas)
        assert orders[0].algo == "twap"


# ── Execution ────────────────────────────────────────────────────────────


class TestExecution:
    def test_dry_run_fills(self, bridge_with_signal):
        deltas = bridge_with_signal.reconcile()
        orders = bridge_with_signal.generate_orders(deltas)
        executed = bridge_with_signal.execute_orders(orders)
        assert executed[0].status == "filled"
        assert executed[0].reason == "dry_run_fill"

    def test_position_updated_after_fill(self, bridge_with_signal):
        deltas = bridge_with_signal.reconcile()
        orders = bridge_with_signal.generate_orders(deltas)
        bridge_with_signal.execute_orders(orders)
        pos = bridge_with_signal._positions.get(("SPY", "EXP-400"))
        assert pos is not None
        assert pos.contracts == 2

    def test_mock_broker_fill(self):
        broker = MockBroker(fill=True)
        alerter = MockAlerter()
        bridge = LiveBridge(":memory:", dry_run=False, broker=broker, alerter=alerter)
        bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.85)
        orders = bridge.run_cycle()
        filled = [o for o in orders if o.status == "filled"]
        assert len(filled) == 1
        bridge.close()

    def test_mock_broker_reject(self):
        broker = MockBroker(fill=False)
        bridge = LiveBridge(":memory:", dry_run=False, broker=broker)
        bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.85)
        orders = bridge.run_cycle()
        # broker returns rejected, but bridge still recorded it
        assert len(bridge.get_orders()) >= 1
        bridge.close()

    def test_rejected_orders_not_executed(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.05)
        orders = bridge.run_cycle()
        rejected = [o for o in orders if o.status == "rejected"]
        assert len(rejected) == 1


# ── Full cycle ───────────────────────────────────────────────────────────


class TestRunCycle:
    def test_full_cycle(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 3, confidence=0.9)
        orders = bridge.run_cycle()
        assert len(orders) == 1
        assert orders[0].status == "filled"

    def test_cycle_records_pnl(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.9)
        bridge.run_cycle()
        history = bridge.get_pnl_history()
        assert len(history) >= 1

    def test_cycle_records_heartbeat(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.9)
        bridge.run_cycle()
        row = bridge._conn.execute("SELECT COUNT(*) FROM heartbeats").fetchone()
        assert row[0] >= 1


# ── P&L tracking ─────────────────────────────────────────────────────────


class TestPnL:
    def test_initial_pnl_zero(self, bridge):
        snap = bridge.get_pnl()
        assert snap.total_pnl == pytest.approx(0.0)

    def test_pnl_after_trades(self, bridge):
        bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.9)
        bridge.run_cycle()
        snap = bridge.get_pnl()
        assert snap.positions > 0

    def test_drawdown_tracking(self, bridge):
        bridge._peak_pnl = 1000
        bridge._realised_pnl = -500
        bridge._record_pnl()
        assert bridge._max_drawdown > 0


# ── Health ───────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_alive(self, bridge):
        h = bridge.health()
        assert h.alive is True

    def test_health_dry_run(self, bridge):
        h = bridge.health()
        assert h.dry_run is True

    def test_uptime_positive(self, bridge):
        h = bridge.health()
        assert h.uptime_seconds >= 0

    def test_error_count(self, bridge):
        assert bridge.health().errors == 0


# ── State persistence ────────────────────────────────────────────────────


class TestState:
    def test_set_get_state(self, bridge):
        bridge.set_state("last_run", "2024-06-01")
        assert bridge.get_state("last_run") == "2024-06-01"

    def test_get_default(self, bridge):
        assert bridge.get_state("missing", "default") == "default"

    def test_overwrite_state(self, bridge):
        bridge.set_state("k", "v1")
        bridge.set_state("k", "v2")
        assert bridge.get_state("k") == "v2"


# ── Alerter ──────────────────────────────────────────────────────────────


class TestAlerter:
    def test_default_alerter(self, bridge):
        assert bridge.alerter.send("test") is True

    def test_mock_alerter_captures(self):
        alerter = MockAlerter()
        bridge = LiveBridge(":memory:", alerter=alerter)
        bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.9)
        bridge.run_cycle()
        assert len(alerter.messages) > 0
        bridge.close()

    def test_alert_on_risk_block(self):
        alerter = MockAlerter()
        bridge = LiveBridge(":memory:", alerter=alerter,
                            risk_limits=RiskLimits(min_confidence=0.99))
        bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.5)
        bridge.run_cycle()
        warnings = [m for lvl, m in alerter.messages if lvl == "warning"]
        assert len(warnings) > 0
        bridge.close()


# ── DB retrieval ─────────────────────────────────────────────────────────


class TestDBRetrieval:
    def test_get_signals(self, bridge):
        bridge.ingest_signal("A", "SPY", 1, 1)
        bridge.ingest_signal("B", "QQQ", -1, 2)
        sigs = bridge.get_signals()
        assert len(sigs) == 2

    def test_get_orders(self, bridge):
        bridge.ingest_signal("A", "SPY", 1, 1, confidence=0.9)
        bridge.run_cycle()
        orders = bridge.get_orders()
        assert len(orders) >= 1

    def test_get_pnl_history(self, bridge):
        bridge.ingest_signal("A", "SPY", 1, 1, confidence=0.9)
        bridge.run_cycle()
        history = bridge.get_pnl_history()
        assert len(history) >= 1
