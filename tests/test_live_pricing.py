"""Tests for shared/live_pricing.py — LivePricing spread valuation."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from shared.live_pricing import LivePricing
from strategies.base import LegType, Position, TradeDirection, TradeLeg


def _make_position(
    ticker="SPY",
    short_strike=450.0,
    long_strike=445.0,
    exp_str="2026-06-20",
    spread_type="bull_put",
):
    """Build a minimal Position for testing."""
    exp = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if spread_type == "bull_put":
        legs = [
            TradeLeg(LegType.SHORT_PUT, short_strike, exp),
            TradeLeg(LegType.LONG_PUT, long_strike, exp),
        ]
    else:
        legs = []

    return Position(
        id="test-1",
        strategy_name="test",
        ticker=ticker,
        direction=TradeDirection.SHORT,
        legs=legs,
        contracts=1,
        net_credit=1.50,
    )


def _make_chain(rows):
    return pd.DataFrame(rows)


class TestLivePricingGetSpreadValue:
    def test_returns_net_spread_value(self):
        lp = LivePricing(MagicMock(), cache_ttl=0)
        chain = _make_chain([
            {"strike": 450.0, "type": "put", "mid": 0.30, "iv": 0.20},
            {"strike": 445.0, "type": "put", "mid": 0.10, "iv": 0.22},
        ])
        lp._get_chain_cached = MagicMock(return_value=chain)

        pos = _make_position(short_strike=450.0, long_strike=445.0)
        result = lp.get_spread_value(pos, underlying_price=460.0)
        assert result == pytest.approx(0.20, abs=0.01)

    def test_returns_none_on_missing_leg(self):
        lp = LivePricing(MagicMock(), cache_ttl=0)
        chain = _make_chain([{"strike": 450.0, "type": "put", "mid": 0.30, "iv": 0.20}])
        lp._get_chain_cached = MagicMock(return_value=chain)

        pos = _make_position(short_strike=450.0, long_strike=445.0)
        assert lp.get_spread_value(pos, 460.0) is None

    def test_returns_none_on_empty_chain(self):
        lp = LivePricing(MagicMock(), cache_ttl=0)
        lp._get_chain_cached = MagicMock(return_value=pd.DataFrame())
        assert lp.get_spread_value(_make_position(), 460.0) is None

    def test_returns_none_on_no_legs(self):
        lp = LivePricing(MagicMock(), cache_ttl=0)
        assert lp.get_spread_value(_make_position(spread_type="none"), 460.0) is None

    def test_strike_snap_finds_nearby(self):
        lp = LivePricing(MagicMock(), cache_ttl=0)
        chain = _make_chain([
            {"strike": 450.5, "type": "put", "mid": 0.25, "iv": 0.20},
            {"strike": 445.0, "type": "put", "mid": 0.10, "iv": 0.22},
        ])
        lp._get_chain_cached = MagicMock(return_value=chain)

        pos = _make_position(short_strike=450.0, long_strike=445.0)
        assert lp.get_spread_value(pos, 460.0) is not None


class TestLivePricingCache:
    def test_cache_hit_avoids_refetch(self):
        analyzer = MagicMock()
        provider = MagicMock()
        analyzer.polygon = provider
        chain = _make_chain([{"strike": 450.0, "type": "put", "mid": 0.30, "iv": 0.20}])
        provider.get_options_chain.return_value = chain

        lp = LivePricing(analyzer, cache_ttl=300)
        lp._get_chain_cached("SPY", "2026-06-20")
        lp._get_chain_cached("SPY", "2026-06-20")
        provider.get_options_chain.assert_called_once()


class TestLivePricingGetContractIV:
    def test_returns_iv(self):
        lp = LivePricing(MagicMock(), cache_ttl=0)
        chain = _make_chain([{"strike": 450.0, "type": "put", "mid": 0.30, "iv": 0.25}])
        lp._get_chain_cached = MagicMock(return_value=chain)
        assert lp.get_contract_iv("SPY", 450.0, "2026-06-20", "put") == pytest.approx(0.25)

    def test_returns_none_for_zero_iv(self):
        lp = LivePricing(MagicMock(), cache_ttl=0)
        chain = _make_chain([{"strike": 450.0, "type": "put", "mid": 0.30, "iv": 0.0}])
        lp._get_chain_cached = MagicMock(return_value=chain)
        assert lp.get_contract_iv("SPY", 450.0, "2026-06-20", "put") is None
