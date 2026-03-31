"""Tests for shared.signal_scorer — score_signal() returns 0-100, type-specific weighting."""

from datetime import datetime, timezone

import pytest

from shared.signal_scorer import SCORING_WEIGHTS, score_signal
from strategies.base import LegType, Signal, TradeLeg, TradeDirection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_legs(short_strike=450.0, long_strike=445.0, option_type="put"):
    exp = datetime(2025, 6, 20, tzinfo=timezone.utc)
    short_lt = LegType.SHORT_PUT if option_type == "put" else LegType.SHORT_CALL
    long_lt = LegType.LONG_PUT if option_type == "put" else LegType.LONG_CALL
    return [
        TradeLeg(leg_type=short_lt, strike=short_strike, expiration=exp),
        TradeLeg(leg_type=long_lt, strike=long_strike, expiration=exp),
    ]


def _bull_put(net_credit=1.50, spread_width=5.0, score=0.0):
    """A standard bull_put signal with given credit/width."""
    short = 450.0
    long_ = short - spread_width
    legs = _make_legs(short_strike=short, long_strike=long_, option_type="put")
    return Signal(
        strategy_name="CreditSpreadStrategy",
        ticker="SPY",
        direction=TradeDirection.LONG,
        legs=legs,
        net_credit=net_credit,
        max_loss=spread_width - net_credit,
        max_profit=net_credit,
        metadata={"spread_type": "bull_put"},
    )


def _bear_call(net_credit=1.20):
    legs = _make_legs(option_type="call")
    return Signal(
        strategy_name="CreditSpreadStrategy",
        ticker="SPY",
        direction=TradeDirection.SHORT,
        legs=legs,
        net_credit=net_credit,
        max_loss=5.0 - net_credit,
        max_profit=net_credit,
        metadata={"spread_type": "bear_call"},
    )


def _iron_condor(net_credit=2.50):
    exp = datetime(2025, 6, 20, tzinfo=timezone.utc)
    legs = [
        TradeLeg(LegType.SHORT_PUT, 440.0, exp),
        TradeLeg(LegType.LONG_PUT, 435.0, exp),
        TradeLeg(LegType.SHORT_CALL, 460.0, exp),
        TradeLeg(LegType.LONG_CALL, 465.0, exp),
    ]
    return Signal(
        strategy_name="IronCondorStrategy",
        ticker="SPY",
        direction=TradeDirection.NEUTRAL,
        legs=legs,
        net_credit=net_credit,
        max_loss=5.0 - net_credit / 2,
        max_profit=net_credit,
        metadata={"spread_type": "iron_condor"},
    )


def _straddle(net_credit=8.0):
    exp = datetime(2025, 4, 4, tzinfo=timezone.utc)
    legs = [
        TradeLeg(LegType.SHORT_CALL, 450.0, exp),
        TradeLeg(LegType.SHORT_PUT, 450.0, exp),
    ]
    return Signal(
        strategy_name="StraddleStrangleStrategy",
        ticker="SPY",
        direction=TradeDirection.NEUTRAL,
        legs=legs,
        net_credit=net_credit,
        max_loss=50.0,
        max_profit=net_credit,
        metadata={"spread_type": "short_straddle"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoreRange:
    def test_score_within_0_100(self):
        for s in [_bull_put(), _bear_call(), _iron_condor(), _straddle()]:
            sc = score_signal(s)
            assert 0 <= sc <= 100, f"Score {sc} out of range for {s.strategy_name}"

    def test_zero_credit_signal_scores_low(self):
        signal = _bull_put(net_credit=0.0)
        assert score_signal(signal) < 30

    def test_high_credit_scores_higher_than_low_credit(self):
        low = score_signal(_bull_put(net_credit=0.50, spread_width=5.0))
        high = score_signal(_bull_put(net_credit=2.00, spread_width=5.0))
        assert high > low

    def test_returns_float(self):
        assert isinstance(score_signal(_bull_put()), float)


class TestIVRankComponent:
    def test_high_iv_rank_increases_score(self):
        low_iv = score_signal(_bull_put(), iv_rank=10)
        high_iv = score_signal(_bull_put(), iv_rank=80)
        assert high_iv > low_iv

    def test_iv_contribution_capped_at_max(self):
        # Even with IV=1000, score stays ≤ 100
        sc = score_signal(_bull_put(), iv_rank=1000)
        assert sc <= 100

    def test_iv_zero_gives_no_iv_component(self):
        sc_no_iv = score_signal(_bull_put(), iv_rank=0)
        sc_with_iv = score_signal(_bull_put(), iv_rank=50)
        # With IV=0 we expect strictly less score
        assert sc_with_iv > sc_no_iv


class TestTechnicalAlignment:
    def test_bull_put_scores_higher_with_bullish_trend(self):
        bullish = score_signal(_bull_put(), technical_signals={"trend": "bullish"})
        bearish = score_signal(_bull_put(), technical_signals={"trend": "bearish"})
        assert bullish > bearish

    def test_bear_call_scores_higher_with_bearish_trend(self):
        bearish = score_signal(_bear_call(), technical_signals={"trend": "bearish"})
        bullish = score_signal(_bear_call(), technical_signals={"trend": "bullish"})
        assert bearish > bullish

    def test_iron_condor_scores_higher_with_neutral_trend(self):
        neutral = score_signal(_iron_condor(), technical_signals={"trend": "neutral", "rsi": 50})
        bullish = score_signal(_iron_condor(), technical_signals={"trend": "bullish", "rsi": 70})
        assert neutral >= bullish

    def test_iron_condor_rsi_bonus_in_range(self):
        """IC gets RSI bonus when RSI is 40–60."""
        in_range = score_signal(_iron_condor(), technical_signals={"trend": "neutral", "rsi": 50})
        out_range = score_signal(_iron_condor(), technical_signals={"trend": "neutral", "rsi": 75})
        assert in_range > out_range


class TestTypeSpecificWeighting:
    def test_bull_put_uses_spread_width_for_credit_score(self):
        """A tighter spread with same credit pct should still score correctly."""
        wide = _bull_put(net_credit=1.50, spread_width=10.0)   # 15% credit
        narrow = _bull_put(net_credit=0.75, spread_width=5.0)  # 15% credit
        # Same credit % → same credit component
        assert abs(score_signal(wide) - score_signal(narrow)) < 5.0

    def test_iron_condor_recognized_by_strategy_name(self):
        """Iron condor should have non-zero technical score with neutral trend."""
        sc = score_signal(_iron_condor(), technical_signals={"trend": "neutral", "rsi": 50})
        # Minimum expected: condor_tech_neutral(10) + condor_tech_rsi_range(5) = 15
        # But total score must be ≥ these components
        assert sc >= SCORING_WEIGHTS["condor_tech_neutral"]

    def test_straddle_recognized(self):
        sc = score_signal(_straddle())
        assert 0 <= sc <= 100

    def test_no_crash_on_empty_metadata(self):
        signal = _bull_put()
        signal.metadata = {}
        score_signal(signal)  # should not raise


class TestEdgeCases:
    def test_single_leg_signal_does_not_crash(self):
        exp = datetime(2025, 6, 20, tzinfo=timezone.utc)
        sig = Signal(
            strategy_name="CreditSpreadStrategy",
            ticker="SPY",
            direction=TradeDirection.LONG,
            legs=[TradeLeg(LegType.SHORT_PUT, 450.0, exp)],
            net_credit=1.0,
            max_loss=4.0,
            max_profit=1.0,
            metadata={"spread_type": "bull_put"},
        )
        sc = score_signal(sig)
        assert 0 <= sc <= 100

    def test_no_legs_does_not_crash(self):
        sig = Signal(
            strategy_name="CreditSpreadStrategy",
            ticker="SPY",
            direction=TradeDirection.LONG,
            legs=[],
            net_credit=1.5,
            max_loss=3.5,
            max_profit=1.5,
            metadata={"spread_type": "bull_put"},
        )
        sc = score_signal(sig)
        assert 0 <= sc <= 100

    def test_negative_credit_scores_zero_for_credit_component(self):
        """A debit spread with negative net_credit should not produce negative score."""
        signal = _bull_put(net_credit=-2.0)
        sc = score_signal(signal)
        assert sc >= 0
