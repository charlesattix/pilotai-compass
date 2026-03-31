"""
Tests for the unified entry path (Phase 1).

Covers:
  - snapshot_builder: build_live_market_snapshot, reprice_signals_from_chain
  - strategy_factory: build_strategy_list, param extraction
  - signal_scorer: score_signal
  - Alert.from_opportunity() straddle handling
  - AlertRouter dedup key includes alert_type
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from strategies.base import Signal, TradeLeg, LegType, TradeDirection


# ---------------------------------------------------------------------------
# snapshot_builder
# ---------------------------------------------------------------------------

class TestBuildLiveMarketSnapshot:
    def _make_price_data(self, n=100, price=500.0):
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame(
            {"Open": price, "High": price + 5, "Low": price - 5, "Close": price, "Volume": 1e6},
            index=dates,
        )

    def test_basic_snapshot_fields(self):
        from shared.snapshot_builder import build_live_market_snapshot
        price_data = self._make_price_data()
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=price_data,
            current_price=500.0,
            iv_data={"iv_rank": 35.0},
            technical_signals={"trend": "bullish"},
        )
        assert snap.prices["SPY"] == 500.0
        assert snap.iv_rank["SPY"] == 35.0
        assert "SPY" in snap.rsi
        assert "SPY" in snap.realized_vol
        assert snap.regime is None

    def test_regime_normalized(self):
        from shared.snapshot_builder import build_live_market_snapshot
        price_data = self._make_price_data()
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=price_data,
            current_price=500.0,
            iv_data={"iv_rank": 30.0},
            technical_signals={},
            regime="BULL",
        )
        assert snap.regime == "bull"

    def test_neutral_regime_maps_to_low_vol(self):
        from shared.snapshot_builder import build_live_market_snapshot
        price_data = self._make_price_data()
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=price_data,
            current_price=500.0,
            iv_data={},
            technical_signals={},
            regime="NEUTRAL",
        )
        assert snap.regime == "low_vol"

    def test_vix_data_used(self):
        from shared.snapshot_builder import build_live_market_snapshot
        price_data = self._make_price_data()
        vix_data = pd.DataFrame(
            {"Close": [25.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=price_data,
            current_price=500.0,
            iv_data={},
            technical_signals={},
            vix_data=vix_data,
        )
        assert snap.vix == 25.0

    def test_events_passed_through(self):
        from shared.snapshot_builder import build_live_market_snapshot
        price_data = self._make_price_data()
        upcoming = [{"event": "FOMC", "date": "2024-06-12"}]
        recent = [{"event": "CPI", "date": "2024-06-10"}]
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=price_data,
            current_price=500.0,
            iv_data={},
            technical_signals={},
            upcoming_events=upcoming,
            recent_events=recent,
        )
        assert len(snap.upcoming_events) == 1
        assert len(snap.recent_events) == 1


class TestRepriceSignalsFromChain:
    def test_no_chain_returns_unchanged(self):
        from shared.snapshot_builder import reprice_signals_from_chain
        sig = MagicMock()
        result = reprice_signals_from_chain([sig], None)
        assert result == [sig]

    def test_empty_chain_returns_unchanged(self):
        from shared.snapshot_builder import reprice_signals_from_chain
        sig = MagicMock()
        result = reprice_signals_from_chain([sig], pd.DataFrame())
        assert result == [sig]


# ---------------------------------------------------------------------------
# strategy_factory
# ---------------------------------------------------------------------------

class TestStrategyFactory:
    def _base_config(self, **overrides):
        cfg = {
            "strategy": {"direction": "both", "min_dte": 30, "target_dte": 35},
            "risk": {"profit_target": 50, "max_risk_per_trade": 2.0},
        }
        cfg.update(overrides)
        return cfg

    def test_credit_spread_always_enabled(self):
        from shared.strategy_factory import build_strategy_list
        from strategies.credit_spread import CreditSpreadStrategy
        strats = build_strategy_list(self._base_config())
        assert any(isinstance(s, CreditSpreadStrategy) for s in strats)

    def test_iron_condor_disabled_by_default(self):
        from shared.strategy_factory import build_strategy_list
        from strategies.iron_condor import IronCondorStrategy
        strats = build_strategy_list(self._base_config())
        assert not any(isinstance(s, IronCondorStrategy) for s in strats)

    def test_iron_condor_enabled_when_configured(self):
        from shared.strategy_factory import build_strategy_list
        from strategies.iron_condor import IronCondorStrategy
        cfg = self._base_config()
        cfg["strategy"]["iron_condor"] = {"enabled": True}
        strats = build_strategy_list(cfg)
        assert any(isinstance(s, IronCondorStrategy) for s in strats)

    def test_straddle_disabled_by_default(self):
        from shared.strategy_factory import build_strategy_list
        from strategies.straddle_strangle import StraddleStrangleStrategy
        strats = build_strategy_list(self._base_config())
        assert not any(isinstance(s, StraddleStrangleStrategy) for s in strats)

    def test_straddle_enabled_when_configured(self):
        from shared.strategy_factory import build_strategy_list
        from strategies.straddle_strangle import StraddleStrangleStrategy
        cfg = self._base_config()
        cfg["strategy"]["straddle_strangle"] = {"enabled": True}
        strats = build_strategy_list(cfg)
        assert any(isinstance(s, StraddleStrangleStrategy) for s in strats)

    def test_credit_spread_params_extracted(self):
        from shared.strategy_factory import _extract_credit_spread_params
        cfg = self._base_config()
        params = _extract_credit_spread_params(cfg)
        assert params["direction"] == "both"
        assert params["target_dte"] == 35
        assert params["profit_target_pct"] == 0.5
        assert params["scan_weekday"] == "any"


# ---------------------------------------------------------------------------
# signal_scorer
# ---------------------------------------------------------------------------

class TestSignalScorer:
    def _make_signal(self, credit=1.5, max_loss=8.5, max_profit=1.5,
                     short_strike=490.0, long_strike=480.0, spread_type="bull_put"):
        exp = datetime(2024, 7, 19, tzinfo=timezone.utc)
        return Signal(
            strategy_name="CreditSpreadStrategy",
            ticker="SPY",
            direction=TradeDirection.SHORT,
            legs=[
                TradeLeg(leg_type=LegType.SHORT_PUT, strike=short_strike, expiration=exp, contracts=1),
                TradeLeg(leg_type=LegType.LONG_PUT, strike=long_strike, expiration=exp, contracts=1),
            ],
            expiration=exp,
            dte=35,
            net_credit=credit,
            max_loss=max_loss,
            max_profit=max_profit,
            profit_target_pct=0.50,
            stop_loss_pct=2.0,
            score=0,
            metadata={"spread_type": spread_type},
        )

    def test_score_positive(self):
        from shared.signal_scorer import score_signal
        sig = self._make_signal()
        score = score_signal(sig, iv_rank=30.0)
        assert score > 0

    def test_score_max_100(self):
        from shared.signal_scorer import score_signal
        sig = self._make_signal(credit=9.0, max_loss=1.0, max_profit=9.0)
        score = score_signal(sig, iv_rank=100.0, technical_signals={"trend": "bullish"})
        assert score <= 100

    def test_higher_credit_higher_score(self):
        from shared.signal_scorer import score_signal
        low = self._make_signal(credit=0.5, max_loss=9.5, max_profit=0.5)
        high = self._make_signal(credit=3.0, max_loss=7.0, max_profit=3.0)
        assert score_signal(high) > score_signal(low)

    def test_bullish_trend_helps_bull_put(self):
        from shared.signal_scorer import score_signal
        sig = self._make_signal()
        with_trend = score_signal(sig, technical_signals={"trend": "bullish"})
        without_trend = score_signal(sig, technical_signals={"trend": "bearish"})
        assert with_trend > without_trend

    def test_condor_neutral_bonus(self):
        from shared.signal_scorer import score_signal
        sig = self._make_signal(spread_type="iron_condor")
        sig.strategy_name = "IronCondorStrategy"
        neutral_score = score_signal(sig, technical_signals={"trend": "neutral"})
        trend_score = score_signal(sig, technical_signals={"trend": "bullish"})
        assert neutral_score > trend_score


# ---------------------------------------------------------------------------
# Alert straddle handling
# ---------------------------------------------------------------------------

class TestAlertStraddleHandling:
    def _straddle_opp(self, opp_type="short_straddle", is_debit=False):
        opp = {
            "ticker": "SPY",
            "type": opp_type,
            "call_strike": 500.0,
            "put_strike": 500.0,
            "short_strike": 0,
            "long_strike": 0,
            "expiration": "2024-07-19",
            "credit": 8.5,
            "max_loss": 25.0,
            "dte": 7,
            "score": 65,
            "is_debit": is_debit,
        }
        if is_debit:
            opp["debit"] = 8.5
            opp["credit"] = 0.0
        return opp

    def test_short_straddle_type(self):
        from alerts.alert_schema import Alert, AlertType
        alert = Alert.from_opportunity(self._straddle_opp("short_straddle"))
        assert alert.type == AlertType.straddle_strangle

    def test_long_straddle_type(self):
        from alerts.alert_schema import Alert, AlertType
        alert = Alert.from_opportunity(self._straddle_opp("long_straddle", is_debit=True))
        assert alert.type == AlertType.straddle_strangle

    def test_straddle_direction_neutral(self):
        from alerts.alert_schema import Alert, Direction
        alert = Alert.from_opportunity(self._straddle_opp())
        assert alert.direction == Direction.neutral

    def test_straddle_has_two_legs(self):
        from alerts.alert_schema import Alert
        alert = Alert.from_opportunity(self._straddle_opp())
        assert len(alert.legs) == 2
        types = {l.option_type for l in alert.legs}
        assert types == {"call", "put"}

    def test_short_straddle_legs_are_sell(self):
        from alerts.alert_schema import Alert
        alert = Alert.from_opportunity(self._straddle_opp("short_straddle"))
        assert all(l.action == "sell" for l in alert.legs)

    def test_long_straddle_legs_are_buy(self):
        from alerts.alert_schema import Alert
        alert = Alert.from_opportunity(self._straddle_opp("long_straddle", is_debit=True))
        assert all(l.action == "buy" for l in alert.legs)

    def test_strangle_type(self):
        from alerts.alert_schema import Alert, AlertType
        alert = Alert.from_opportunity(self._straddle_opp("short_strangle"))
        assert alert.type == AlertType.straddle_strangle


# ---------------------------------------------------------------------------
# AlertRouter dedup key includes type
# ---------------------------------------------------------------------------

class TestDedupKeyIncludesType:
    def test_ic_and_straddle_dont_collide(self):
        """An iron condor and a straddle for the same ticker should not dedup each other."""
        from alerts.alert_router import AlertRouter
        from alerts.alert_schema import Alert, AlertType, Direction, Leg

        router = AlertRouter(
            risk_gate=MagicMock(),
            position_sizer=MagicMock(),
            telegram_bot=MagicMock(),
            formatter=MagicMock(),
        )

        # Mark an IC as routed
        now = datetime.now(timezone.utc)
        with patch("alerts.alert_router.upsert_dedup_entry"):
            router._mark_dedup("SPY", "neutral", now, "iron_condor")

        # Straddle key should NOT be blocked
        straddle_key = ("SPY", "neutral", "straddle_strangle")
        assert straddle_key not in router._dedup_ledger

        # But IC key IS blocked
        ic_key = ("SPY", "neutral", "iron_condor")
        assert ic_key in router._dedup_ledger
