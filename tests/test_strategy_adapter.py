"""Tests for shared/strategy_adapter.py — conversion between strategy types and paper trader dicts."""

from datetime import datetime, timezone

from shared.strategy_adapter import signal_to_opportunity, trade_dict_to_position
from strategies.base import LegType, Signal, TradeDirection, TradeLeg


def _make_credit_spread_signal():
    """Create a bull put credit spread Signal for testing."""
    exp = datetime(2026, 4, 17, tzinfo=timezone.utc)
    return Signal(
        strategy_name="CreditSpreadStrategy",
        ticker="SPY",
        direction=TradeDirection.SHORT,
        legs=[
            TradeLeg(LegType.SHORT_PUT, 540.0, exp, entry_price=2.50),
            TradeLeg(LegType.LONG_PUT, 530.0, exp, entry_price=1.00),
        ],
        net_credit=1.50,
        max_loss=8.50,
        max_profit=1.50,
        profit_target_pct=0.55,
        stop_loss_pct=1.25,
        score=72.0,
        signal_date=datetime(2026, 3, 6, tzinfo=timezone.utc),
        expiration=exp,
        dte=42,
        metadata={"spread_type": "bull_put", "iv_rank": 45.0},
    )


def _make_iron_condor_signal():
    """Create an iron condor Signal for testing."""
    exp = datetime(2026, 4, 17, tzinfo=timezone.utc)
    return Signal(
        strategy_name="IronCondorStrategy",
        ticker="SPY",
        direction=TradeDirection.NEUTRAL,
        legs=[
            TradeLeg(LegType.SHORT_PUT, 530.0, exp, entry_price=1.80),
            TradeLeg(LegType.LONG_PUT, 518.0, exp, entry_price=0.80),
            TradeLeg(LegType.SHORT_CALL, 570.0, exp, entry_price=1.60),
            TradeLeg(LegType.LONG_CALL, 582.0, exp, entry_price=0.70),
        ],
        net_credit=1.90,
        max_loss=10.10,
        max_profit=1.90,
        profit_target_pct=0.30,
        stop_loss_pct=2.50,
        score=55.0,
        signal_date=datetime(2026, 3, 6, tzinfo=timezone.utc),
        expiration=exp,
        dte=42,
        metadata={
            "put_credit": 1.00,
            "call_credit": 0.90,
            "put_short": 530.0,
            "call_short": 570.0,
            "iv_rank": 50.0,
        },
    )


class TestSignalToOpportunity:
    def test_credit_spread_basic_fields(self):
        sig = _make_credit_spread_signal()
        opp = signal_to_opportunity(sig, current_price=550.0)

        assert opp["ticker"] == "SPY"
        assert opp["type"] == "bull_put_spread"
        assert opp["short_strike"] == 540.0
        assert opp["long_strike"] == 530.0
        assert opp["credit"] == 1.50
        assert opp["max_loss"] == 8.50
        assert opp["dte"] == 42
        assert opp["score"] == 72.0
        assert opp["current_price"] == 550.0
        assert opp["expiration"] == "2026-04-17"

    def test_credit_spread_exit_params(self):
        sig = _make_credit_spread_signal()
        opp = signal_to_opportunity(sig, current_price=550.0)

        assert opp["profit_target_pct"] == 0.55
        assert opp["stop_loss_pct"] == 1.25
        assert opp["strategy_name"] == "CreditSpreadStrategy"

    def test_iron_condor_fields(self):
        sig = _make_iron_condor_signal()
        opp = signal_to_opportunity(sig, current_price=550.0)

        assert opp["type"] == "iron_condor"
        # Primary (put side) strikes
        assert opp["short_strike"] == 530.0
        assert opp["long_strike"] == 518.0
        # Call side
        assert opp["call_short_strike"] == 570.0
        assert opp["call_long_strike"] == 582.0
        assert opp["put_credit"] == 1.00
        assert opp["call_credit"] == 0.90

    def test_iron_condor_exit_params(self):
        sig = _make_iron_condor_signal()
        opp = signal_to_opportunity(sig, current_price=550.0)

        assert opp["profit_target_pct"] == 0.30
        assert opp["stop_loss_pct"] == 2.50

    def test_bear_call_spread(self):
        exp = datetime(2026, 4, 17, tzinfo=timezone.utc)
        sig = Signal(
            strategy_name="CreditSpreadStrategy",
            ticker="QQQ",
            direction=TradeDirection.SHORT,
            legs=[
                TradeLeg(LegType.SHORT_CALL, 490.0, exp),
                TradeLeg(LegType.LONG_CALL, 500.0, exp),
            ],
            net_credit=1.20,
            max_loss=8.80,
            dte=30,
        )
        opp = signal_to_opportunity(sig, current_price=480.0)
        assert opp["type"] == "bear_call_spread"
        assert opp["short_strike"] == 490.0
        assert opp["long_strike"] == 500.0

    def test_iv_rank_carried(self):
        sig = _make_credit_spread_signal()
        opp = signal_to_opportunity(sig, current_price=550.0)
        assert opp["iv_rank"] == 45.0


class TestTradeDictToPosition:
    def test_bull_put_position(self):
        trade = {
            "id": "PT-abc123",
            "ticker": "SPY",
            "type": "bull_put_spread",
            "short_strike": 540.0,
            "long_strike": 530.0,
            "expiration": "2026-04-17",
            "contracts": 2,
            "credit": 1.50,
            "max_loss_per_spread": 8.50,
            "entry_date": "2026-03-06T12:00:00+00:00",
            "profit_target_pct": 0.55,
            "stop_loss_pct": 1.25,
            "strategy_name": "CreditSpreadStrategy",
        }
        pos = trade_dict_to_position(trade)

        assert pos.id == "PT-abc123"
        assert pos.ticker == "SPY"
        assert pos.contracts == 2
        assert pos.net_credit == 1.50
        assert pos.profit_target_pct == 0.55
        assert pos.stop_loss_pct == 1.25
        assert len(pos.legs) == 2
        assert pos.legs[0].leg_type == LegType.SHORT_PUT
        assert pos.legs[0].strike == 540.0
        assert pos.legs[1].leg_type == LegType.LONG_PUT
        assert pos.legs[1].strike == 530.0

    def test_iron_condor_position(self):
        trade = {
            "id": "PT-xyz789",
            "ticker": "SPY",
            "type": "iron_condor",
            "short_strike": 530.0,
            "long_strike": 518.0,
            "call_short_strike": 570.0,
            "call_long_strike": 582.0,
            "expiration": "2026-04-17",
            "contracts": 1,
            "credit": 1.90,
            "profit_target_pct": 0.30,
            "stop_loss_pct": 2.50,
        }
        pos = trade_dict_to_position(trade)

        assert len(pos.legs) == 4
        assert pos.direction == TradeDirection.NEUTRAL
        leg_types = {leg.leg_type for leg in pos.legs}
        assert LegType.SHORT_PUT in leg_types
        assert LegType.LONG_PUT in leg_types
        assert LegType.SHORT_CALL in leg_types
        assert LegType.LONG_CALL in leg_types

    def test_round_trip_profit_target(self):
        """Signal.profit_target_pct survives adapter conversion through trade dict."""
        sig = _make_credit_spread_signal()
        opp = signal_to_opportunity(sig, current_price=550.0)

        # Simulate paper trader storing it in trade dict
        trade = {
            "id": "PT-test",
            "ticker": opp["ticker"],
            "type": opp["type"],
            "short_strike": opp["short_strike"],
            "long_strike": opp["long_strike"],
            "expiration": opp["expiration"],
            "contracts": 1,
            "credit": opp["credit"],
            "profit_target_pct": opp["profit_target_pct"],
            "stop_loss_pct": opp["stop_loss_pct"],
        }

        pos = trade_dict_to_position(trade)
        assert pos.profit_target_pct == 0.55
        assert pos.stop_loss_pct == 1.25

    def test_bear_call_position(self):
        trade = {
            "id": "PT-bear",
            "ticker": "QQQ",
            "type": "bear_call_spread",
            "short_strike": 490.0,
            "long_strike": 500.0,
            "expiration": "2026-04-17",
            "contracts": 1,
            "credit": 1.20,
        }
        pos = trade_dict_to_position(trade)
        assert pos.legs[0].leg_type == LegType.SHORT_CALL
        assert pos.legs[1].leg_type == LegType.LONG_CALL

    def test_long_straddle_position(self):
        trade = {
            "id": "PT-lstrad",
            "ticker": "SPY",
            "type": "long_straddle",
            "call_strike": 550.0,
            "put_strike": 550.0,
            "expiration": "2026-04-17",
            "contracts": 1,
            "credit": -3.80,  # debit paid (negative)
            "strategy_name": "StraddleStrangleStrategy",
        }
        pos = trade_dict_to_position(trade)
        assert pos.direction == TradeDirection.LONG
        assert len(pos.legs) == 2
        assert pos.legs[0].leg_type == LegType.LONG_CALL
        assert pos.legs[0].strike == 550.0
        assert pos.legs[1].leg_type == LegType.LONG_PUT
        assert pos.legs[1].strike == 550.0
        assert pos.net_credit == -3.80
        assert pos.max_loss_per_unit == 3.80  # abs(debit)

    def test_short_strangle_position(self):
        trade = {
            "id": "PT-sstrng",
            "ticker": "SPY",
            "type": "short_strangle",
            "call_strike": 560.0,
            "put_strike": 540.0,
            "expiration": "2026-04-17",
            "contracts": 2,
            "credit": 2.50,
            "strategy_name": "StraddleStrangleStrategy",
        }
        pos = trade_dict_to_position(trade)
        assert pos.direction == TradeDirection.SHORT
        assert len(pos.legs) == 2
        assert pos.legs[0].leg_type == LegType.SHORT_CALL
        assert pos.legs[0].strike == 560.0
        assert pos.legs[1].leg_type == LegType.SHORT_PUT
        assert pos.legs[1].strike == 540.0
        assert pos.net_credit == 2.50
        assert pos.max_loss_per_unit == 7.50  # credit * 3

    def test_straddle_round_trip(self):
        """Signal → opportunity → trade dict → Position round-trip for straddle."""
        exp = datetime(2026, 4, 17, tzinfo=timezone.utc)
        sig = Signal(
            strategy_name="StraddleStrangleStrategy",
            ticker="SPY",
            direction=TradeDirection.SHORT,
            legs=[
                TradeLeg(LegType.SHORT_CALL, 555.0, exp, entry_price=1.30),
                TradeLeg(LegType.SHORT_PUT, 545.0, exp, entry_price=1.20),
            ],
            net_credit=2.50,
            max_loss=7.50,
            max_profit=2.50,
            profit_target_pct=0.55,
            stop_loss_pct=0.45,
            score=60.0,
            signal_date=datetime(2026, 3, 6, tzinfo=timezone.utc),
            expiration=exp,
            dte=42,
            metadata={"spread_type": "short_strangle"},
        )
        opp = signal_to_opportunity(sig, current_price=550.0)
        assert opp["type"] == "short_strangle"
        assert opp["call_strike"] == 555.0
        assert opp["put_strike"] == 545.0
        assert opp["is_debit"] is False

        trade = {
            "id": "PT-rt",
            "ticker": opp["ticker"],
            "type": opp["type"],
            "call_strike": opp["call_strike"],
            "put_strike": opp["put_strike"],
            "expiration": opp["expiration"],
            "contracts": 1,
            "credit": opp["credit"],
            "profit_target_pct": opp["profit_target_pct"],
            "stop_loss_pct": opp["stop_loss_pct"],
        }
        pos = trade_dict_to_position(trade)
        assert pos.profit_target_pct == 0.55
        assert pos.stop_loss_pct == 0.45
        assert pos.legs[0].leg_type == LegType.SHORT_CALL
        assert pos.legs[1].leg_type == LegType.SHORT_PUT


class TestStraddleSignalToOpportunity:
    def test_long_straddle_signal(self):
        exp = datetime(2026, 4, 17, tzinfo=timezone.utc)
        sig = Signal(
            strategy_name="StraddleStrangleStrategy",
            ticker="SPY",
            direction=TradeDirection.LONG,
            legs=[
                TradeLeg(LegType.LONG_CALL, 550.0, exp, entry_price=2.00),
                TradeLeg(LegType.LONG_PUT, 550.0, exp, entry_price=1.80),
            ],
            net_credit=-3.80,  # debit
            max_loss=3.80,
            max_profit=100.0,
            profit_target_pct=0.55,
            stop_loss_pct=0.45,
            score=65.0,
            signal_date=datetime(2026, 3, 6, tzinfo=timezone.utc),
            expiration=exp,
            dte=42,
            metadata={"spread_type": "long_straddle"},
        )
        opp = signal_to_opportunity(sig, current_price=550.0)
        assert opp["type"] == "long_straddle"
        assert opp["call_strike"] == 550.0
        assert opp["put_strike"] == 550.0
        assert opp["is_debit"] is True
        assert opp["credit"] == -3.80  # negative for debit

    def test_short_strangle_signal(self):
        exp = datetime(2026, 4, 17, tzinfo=timezone.utc)
        sig = Signal(
            strategy_name="StraddleStrangleStrategy",
            ticker="SPY",
            direction=TradeDirection.SHORT,
            legs=[
                TradeLeg(LegType.SHORT_CALL, 560.0, exp, entry_price=1.30),
                TradeLeg(LegType.SHORT_PUT, 540.0, exp, entry_price=1.20),
            ],
            net_credit=2.50,
            max_loss=7.50,
            dte=5,
            metadata={"spread_type": "short_strangle"},
        )
        opp = signal_to_opportunity(sig, current_price=550.0)
        assert opp["type"] == "short_strangle"
        assert opp["call_strike"] == 560.0
        assert opp["put_strike"] == 540.0
        assert opp["is_debit"] is False
        assert opp["credit"] == 2.50

    def test_straddle_no_iron_condor_fields(self):
        """Straddle should NOT get iron condor fields like call_short_strike."""
        exp = datetime(2026, 4, 17, tzinfo=timezone.utc)
        sig = Signal(
            strategy_name="StraddleStrangleStrategy",
            ticker="SPY",
            direction=TradeDirection.SHORT,
            legs=[
                TradeLeg(LegType.SHORT_CALL, 555.0, exp),
                TradeLeg(LegType.SHORT_PUT, 545.0, exp),
            ],
            net_credit=2.00,
            max_loss=6.00,
            dte=5,
            metadata={"spread_type": "short_strangle"},
        )
        opp = signal_to_opportunity(sig, current_price=550.0)
        assert "call_short_strike" not in opp
        assert "call_long_strike" not in opp
        assert "call_strike" in opp
        assert "put_strike" in opp
