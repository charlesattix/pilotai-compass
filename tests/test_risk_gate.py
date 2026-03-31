"""Tests for alerts.risk_gate — P0 tests: risk gate must never be bypassable.

Every risk rule is tested individually with passing, failing, and boundary
values.  There are no configurable overrides — all limits come from
hard-coded constants.
"""

from datetime import datetime, timedelta, timezone

from alerts.alert_schema import Alert, AlertType, Direction, Leg
from compass.risk_gate import RiskGate
from shared.constants import (
    COOLDOWN_AFTER_STOP,
    DAILY_LOSS_LIMIT,
    MAX_CORRELATED_POSITIONS,
    MAX_RISK_PER_TRADE,
    MAX_TOTAL_EXPOSURE,
    WEEKLY_LOSS_LIMIT,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_legs():
    return [
        Leg(strike=100.0, option_type="put", action="sell", expiration="2025-06-20"),
        Leg(strike=95.0, option_type="put", action="buy", expiration="2025-06-20"),
    ]


def _make_alert(**overrides):
    defaults = dict(
        type=AlertType.credit_spread,
        ticker="SPY",
        direction=Direction.bullish,
        legs=_make_legs(),
        entry_price=1.50,
        stop_loss=3.00,
        profit_target=0.75,
        risk_pct=0.02,
    )
    defaults.update(overrides)
    return Alert(**defaults)


def _clean_state(**overrides):
    """Return a pristine account state with no positions or losses."""
    base = {
        "account_value": 100_000,
        "open_positions": [],
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "recent_stops": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPerTradeRiskCap:
    """Rule 1: alert.risk_pct <= MAX_RISK_PER_TRADE (5%)."""

    def test_within_limit(self):
        gate = RiskGate()
        alert = _make_alert(risk_pct=0.03)
        ok, reason = gate.check(alert, _clean_state())
        assert ok is True
        assert reason == ""

    def test_at_limit(self):
        gate = RiskGate()
        alert = _make_alert(risk_pct=MAX_RISK_PER_TRADE)
        ok, _ = gate.check(alert, _clean_state())
        assert ok is True

    def test_above_limit(self):
        """Cannot construct an Alert with risk_pct > 5% (schema blocks it),
        but we still verify the gate would reject it if reached."""
        gate = RiskGate()
        # Force risk_pct past schema validation
        alert = _make_alert(risk_pct=0.05)
        object.__setattr__(alert, "risk_pct", 0.06)
        ok, reason = gate.check(alert, _clean_state())
        assert ok is False
        assert "exceeds" in reason.lower()


class TestTotalExposure:
    """Rule 2: open_risk + alert.risk_pct <= MAX_TOTAL_EXPOSURE (15%)."""

    def test_room_available(self):
        gate = RiskGate()
        state = _clean_state(open_positions=[
            {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.05},
        ])
        alert = _make_alert(risk_pct=0.05)
        ok, _ = gate.check(alert, state)
        assert ok is True  # 5% + 5% = 10% < 15%

    def test_at_limit(self):
        gate = RiskGate()
        state = _clean_state(open_positions=[
            {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.10},
        ])
        alert = _make_alert(risk_pct=0.05)
        ok, _ = gate.check(alert, state)
        assert ok is True  # 10% + 5% = 15% == limit

    def test_over_limit(self):
        gate = RiskGate()
        state = _clean_state(open_positions=[
            {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.05},
            {"ticker": "IWM", "direction": "bullish", "risk_pct": 0.05},
            {"ticker": "AAPL", "direction": "bearish", "risk_pct": 0.03},
        ])
        alert = _make_alert(risk_pct=0.03)
        ok, reason = gate.check(alert, state)
        assert ok is False  # 13% + 3% = 16% > 15%
        assert "exposure" in reason.lower()


class TestDailyLossLimit:
    """Rule 3: daily_pnl_pct >= -DAILY_LOSS_LIMIT (-8%)."""

    def test_no_loss(self):
        gate = RiskGate()
        ok, _ = gate.check(_make_alert(), _clean_state(daily_pnl_pct=0.0))
        assert ok is True

    def test_at_limit(self):
        gate = RiskGate()
        ok, _ = gate.check(_make_alert(), _clean_state(daily_pnl_pct=-DAILY_LOSS_LIMIT))
        assert ok is True  # exactly at -8% is allowed (>= check)

    def test_breached(self):
        gate = RiskGate()
        ok, reason = gate.check(
            _make_alert(), _clean_state(daily_pnl_pct=-DAILY_LOSS_LIMIT - 0.001)
        )
        assert ok is False
        assert "daily" in reason.lower()


class TestWeeklyLossLimit:
    """Rule 4: weekly loss flags 50% reduction but does NOT block."""

    def test_not_breached(self):
        gate = RiskGate()
        assert gate.weekly_loss_breach(_clean_state(weekly_pnl_pct=-0.10)) is False

    def test_at_limit(self):
        gate = RiskGate()
        # Exactly at -15% is NOT breached (< -0.15 triggers)
        assert gate.weekly_loss_breach(_clean_state(weekly_pnl_pct=-WEEKLY_LOSS_LIMIT)) is False

    def test_breached(self):
        gate = RiskGate()
        assert gate.weekly_loss_breach(_clean_state(weekly_pnl_pct=-0.16)) is True

    def test_does_not_block_alert(self):
        """Weekly loss should not cause check() to reject."""
        gate = RiskGate()
        state = _clean_state(weekly_pnl_pct=-0.20)
        ok, _ = gate.check(_make_alert(), state)
        assert ok is True  # not blocked, only flagged


class TestCorrelatedPositions:
    """Rule 5: max same-direction positions <= MAX_CORRELATED_POSITIONS (3)."""

    def test_below_limit(self):
        gate = RiskGate()
        state = _clean_state(open_positions=[
            {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.02},
            {"ticker": "IWM", "direction": "bullish", "risk_pct": 0.02},
        ])
        ok, _ = gate.check(_make_alert(direction=Direction.bullish), state)
        assert ok is True  # 2 existing + 1 new = 3 but check is count < max

    def test_at_limit(self):
        gate = RiskGate()
        state = _clean_state(open_positions=[
            {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.02},
            {"ticker": "IWM", "direction": "bullish", "risk_pct": 0.02},
            {"ticker": "AAPL", "direction": "bullish", "risk_pct": 0.02},
        ])
        ok, reason = gate.check(_make_alert(direction=Direction.bullish), state)
        assert ok is False  # 3 existing same-direction >= max 3
        assert "positions" in reason.lower()

    def test_different_direction_ok(self):
        gate = RiskGate()
        state = _clean_state(open_positions=[
            {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.02},
            {"ticker": "IWM", "direction": "bullish", "risk_pct": 0.02},
            {"ticker": "AAPL", "direction": "bullish", "risk_pct": 0.02},
        ])
        # Alert is bearish — different direction, should pass
        ok, _ = gate.check(_make_alert(direction=Direction.bearish), state)
        assert ok is True


class TestCooldownAfterStop:
    """Rule 6: no same ticker within COOLDOWN_AFTER_STOP (30 min) of stop."""

    def test_no_recent_stops(self):
        gate = RiskGate()
        ok, _ = gate.check(_make_alert(), _clean_state())
        assert ok is True

    def test_within_cooldown(self):
        gate = RiskGate()
        stopped_at = datetime.now(timezone.utc) - timedelta(seconds=COOLDOWN_AFTER_STOP - 60)
        state = _clean_state(recent_stops=[
            {"ticker": "SPY", "stopped_at": stopped_at},
        ])
        ok, reason = gate.check(_make_alert(ticker="SPY"), state)
        assert ok is False
        assert "cooldown" in reason.lower()

    def test_after_cooldown(self):
        gate = RiskGate()
        stopped_at = datetime.now(timezone.utc) - timedelta(seconds=COOLDOWN_AFTER_STOP + 60)
        state = _clean_state(recent_stops=[
            {"ticker": "SPY", "stopped_at": stopped_at},
        ])
        ok, _ = gate.check(_make_alert(ticker="SPY"), state)
        assert ok is True

    def test_different_ticker_unaffected(self):
        gate = RiskGate()
        stopped_at = datetime.now(timezone.utc) - timedelta(seconds=60)
        state = _clean_state(recent_stops=[
            {"ticker": "QQQ", "stopped_at": stopped_at},
        ])
        ok, _ = gate.check(_make_alert(ticker="SPY"), state)
        assert ok is True

    def test_stopped_at_as_iso_string(self):
        """RiskGate should handle stopped_at as ISO string (from JSON)."""
        gate = RiskGate()
        stopped_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        state = _clean_state(recent_stops=[
            {"ticker": "SPY", "stopped_at": stopped_at},
        ])
        ok, reason = gate.check(_make_alert(ticker="SPY"), state)
        assert ok is False


class TestShortCircuit:
    """Verify rules short-circuit: first failure stops evaluation."""

    def test_per_trade_blocks_before_exposure(self):
        """Even if exposure would also fail, per-trade reason is reported."""
        gate = RiskGate()
        alert = _make_alert(risk_pct=0.05)
        object.__setattr__(alert, "risk_pct", 0.06)  # bypass schema validation
        state = _clean_state(open_positions=[
            {"ticker": "X", "direction": "bullish", "risk_pct": 0.14},
        ])
        ok, reason = gate.check(alert, state)
        assert ok is False
        assert "per-trade" in reason.lower()


class TestNoBypass:
    """Verify there are no configurable overrides or bypass flags."""

    def test_no_constructor_args(self):
        """RiskGate works with no args — config is optional for drawdown CB.
        Core rules (daily/weekly limits, exposure caps) remain hard-coded constants."""
        gate = RiskGate()
        # config attribute exists but is an empty dict when not supplied
        assert hasattr(gate, "config")
        assert gate.config == {}

    def test_constants_match_masterplan(self):
        assert MAX_RISK_PER_TRADE == 0.05
        assert MAX_TOTAL_EXPOSURE == 0.15
        assert DAILY_LOSS_LIMIT == 0.08
        assert WEEKLY_LOSS_LIMIT == 0.15
        assert MAX_CORRELATED_POSITIONS == 3
        assert COOLDOWN_AFTER_STOP == 1800
