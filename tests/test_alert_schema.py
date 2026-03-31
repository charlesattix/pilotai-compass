"""Tests for alerts.alert_schema — Alert creation, validation, conversion."""

from datetime import datetime, timezone

import pytest

from alerts.alert_schema import (
    Alert,
    AlertStatus,
    AlertType,
    Confidence,
    Direction,
    Leg,
    SizeResult,
    TimeSensitivity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_legs(expiration="2025-06-20"):
    """Return a minimal 2-leg bull-put spread."""
    return [
        Leg(strike=100.0, option_type="put", action="sell", expiration=expiration),
        Leg(strike=95.0, option_type="put", action="buy", expiration=expiration),
    ]


def _make_alert(**overrides):
    """Build a valid Alert with sane defaults, overridable by kwargs."""
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


# ---------------------------------------------------------------------------
# Enum value tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_alert_type_values(self):
        assert set(AlertType) == {
            AlertType.credit_spread,
            AlertType.momentum_swing,
            AlertType.iron_condor,
            AlertType.straddle_strangle,
            AlertType.earnings_play,
            AlertType.gamma_lotto,
        }

    def test_confidence_values(self):
        assert set(Confidence) == {Confidence.HIGH, Confidence.MEDIUM, Confidence.SPECULATIVE}

    def test_time_sensitivity_values(self):
        assert set(TimeSensitivity) == {
            TimeSensitivity.IMMEDIATE,
            TimeSensitivity.WITHIN_1HR,
            TimeSensitivity.TODAY,
        }

    def test_direction_values(self):
        assert set(Direction) == {Direction.bullish, Direction.bearish, Direction.neutral}


# ---------------------------------------------------------------------------
# Alert creation & validation
# ---------------------------------------------------------------------------

class TestAlertCreation:
    def test_basic_creation(self):
        alert = _make_alert()
        assert alert.ticker == "SPY"
        assert alert.type == AlertType.credit_spread
        assert alert.status == AlertStatus.pending
        assert len(alert.legs) == 2

    def test_id_auto_generated(self):
        a1 = _make_alert()
        a2 = _make_alert()
        assert a1.id != a2.id  # UUIDs should be unique

    def test_created_at_auto_set(self):
        alert = _make_alert()
        now = datetime.now(timezone.utc)
        assert (now - alert.created_at).total_seconds() < 5

    def test_sizing_default_none(self):
        alert = _make_alert()
        assert alert.sizing is None


class TestAlertValidation:
    def test_empty_legs_raises(self):
        with pytest.raises(ValueError, match="at least one leg"):
            _make_alert(legs=[])

    def test_risk_pct_zero_raises(self):
        with pytest.raises(ValueError, match="risk_pct"):
            _make_alert(risk_pct=0.0)

    def test_risk_pct_negative_raises(self):
        with pytest.raises(ValueError, match="risk_pct"):
            _make_alert(risk_pct=-0.01)

    def test_risk_pct_above_max_raises(self):
        with pytest.raises(ValueError, match="risk_pct"):
            _make_alert(risk_pct=0.06)  # > 5%

    def test_risk_pct_at_max_ok(self):
        alert = _make_alert(risk_pct=0.05)
        assert alert.risk_pct == 0.05

    def test_entry_price_zero_raises(self):
        with pytest.raises(ValueError, match="entry_price"):
            _make_alert(entry_price=0.0)

    def test_entry_price_negative_raises(self):
        with pytest.raises(ValueError, match="entry_price"):
            _make_alert(entry_price=-1.0)


# ---------------------------------------------------------------------------
# from_opportunity conversion
# ---------------------------------------------------------------------------

class TestFromOpportunity:
    def _base_opp(self):
        return {
            "ticker": "AAPL",
            "type": "bull_put_spread",
            "expiration": "2025-07-18",
            "short_strike": 180.0,
            "long_strike": 175.0,
            "credit": 1.20,
            "stop_loss": 2.40,
            "profit_target": 0.60,
            "score": 72,
        }

    def test_bull_put_conversion(self):
        opp = self._base_opp()
        alert = Alert.from_opportunity(opp)

        assert alert.ticker == "AAPL"
        assert alert.type == AlertType.credit_spread
        assert alert.direction == Direction.bullish
        assert len(alert.legs) == 2
        assert alert.entry_price == 1.20

    def test_bear_call_conversion(self):
        opp = self._base_opp()
        opp["type"] = "bear_call_spread"
        alert = Alert.from_opportunity(opp)

        assert alert.direction == Direction.bearish
        assert alert.legs[0].option_type == "call"

    def test_iron_condor_conversion(self):
        opp = self._base_opp()
        opp["type"] = "iron_condor"
        opp["call_short_strike"] = 200.0
        opp["call_long_strike"] = 205.0
        alert = Alert.from_opportunity(opp)

        assert alert.type == AlertType.iron_condor
        assert alert.direction == Direction.neutral
        assert len(alert.legs) == 4

    def test_confidence_from_score(self):
        opp = self._base_opp()

        opp["score"] = 85
        assert Alert.from_opportunity(opp).confidence == Confidence.HIGH

        opp["score"] = 65
        assert Alert.from_opportunity(opp).confidence == Confidence.MEDIUM

        opp["score"] = 40
        assert Alert.from_opportunity(opp).confidence == Confidence.SPECULATIVE

    def test_risk_pct_capped(self):
        opp = self._base_opp()
        opp["risk_pct"] = 0.10  # too high
        alert = Alert.from_opportunity(opp)
        assert alert.risk_pct <= 0.05


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------

class TestToDict:
    def test_round_trip_keys(self):
        alert = _make_alert()
        d = alert.to_dict()

        assert d["type"] == "credit_spread"
        assert d["direction"] == "bullish"
        assert d["confidence"] == "MEDIUM"
        assert d["status"] == "pending"
        assert isinstance(d["created_at"], str)

    def test_legs_serialized(self):
        alert = _make_alert()
        d = alert.to_dict()
        assert len(d["legs"]) == 2
        assert d["legs"][0]["strike"] == 100.0

    def test_sizing_included_when_set(self):
        alert = _make_alert()
        alert.sizing = SizeResult(risk_pct=0.02, contracts=2, dollar_risk=200, max_loss=700)
        d = alert.to_dict()
        assert d["sizing"]["contracts"] == 2
