"""Tests for shared.snapshot_builder — build_live_market_snapshot()."""

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from shared.snapshot_builder import build_live_market_snapshot
from strategies.base import MarketSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_df(n=60, start=450.0, seed=42):
    """Build a minimal OHLCV DataFrame with n rows."""
    rng = np.random.default_rng(seed)
    closes = start + np.cumsum(rng.normal(0, 1.0, n))
    closes = np.maximum(closes, 1.0)
    highs = closes + rng.uniform(0.5, 2.0, n)
    lows = closes - rng.uniform(0.5, 2.0, n)
    opens = closes + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume},
        index=idx,
    )


def _vix_df(n=60, mean_vix=18.0):
    closes = np.full(n, mean_vix) + np.random.default_rng(7).normal(0, 1.0, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": closes}, index=idx)


def _iv_data(iv_rank=35.0):
    return {"iv_rank": iv_rank, "iv_percentile": iv_rank, "current_iv": 0.22}


def _tech_signals():
    return {"trend": "bullish", "rsi": 55.0, "near_support": True, "near_resistance": False}


# ---------------------------------------------------------------------------
# TestBuildLiveMarketSnapshot
# ---------------------------------------------------------------------------

class TestBuildLiveMarketSnapshot:
    def test_returns_market_snapshot(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
        )
        assert isinstance(snap, MarketSnapshot)

    def test_ticker_in_prices(self):
        snap = build_live_market_snapshot(
            ticker="QQQ",
            price_data=_price_df(),
            current_price=380.0,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
        )
        assert "QQQ" in snap.prices
        assert snap.prices["QQQ"] == pytest.approx(380.0)

    def test_ticker_in_price_data(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
        )
        assert "SPY" in snap.price_data
        assert isinstance(snap.price_data["SPY"], pd.DataFrame)

    def test_iv_rank_populated(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(iv_rank=42.0),
            technical_signals=_tech_signals(),
        )
        assert snap.iv_rank["SPY"] == pytest.approx(42.0)

    def test_vix_from_vix_df(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
            vix_data=_vix_df(mean_vix=22.0),
        )
        # VIX should be approximately the last value of the vix_df
        assert 18.0 <= snap.vix <= 28.0

    def test_vix_default_when_no_vix_data(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
            vix_data=None,
        )
        assert snap.vix == pytest.approx(20.0)

    def test_realized_vol_in_valid_range(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
        )
        rv = snap.realized_vol.get("SPY", 0)
        assert 0.10 <= rv <= 1.00

    def test_rsi_in_valid_range(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
        )
        rsi = snap.rsi.get("SPY", 0)
        assert 0.0 <= rsi <= 100.0

    def test_regime_normalized_to_lowercase(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
            regime="BULL",
        )
        assert snap.regime == "bull"

    def test_regime_neutral_mapped_to_low_vol(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
            regime="neutral",
        )
        assert snap.regime == "low_vol"

    def test_regime_none_passthrough(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
            regime=None,
        )
        assert snap.regime is None

    def test_upcoming_events_forwarded(self):
        events = [{"event": "FOMC", "date": "2025-03-20"}]
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
            upcoming_events=events,
        )
        assert snap.upcoming_events == events

    def test_multiindex_columns_handled(self):
        """yfinance sometimes returns MultiIndex columns — should not crash."""
        df = _price_df()
        df.columns = pd.MultiIndex.from_tuples(
            [(col, "SPY") for col in df.columns]
        )
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=df,
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
        )
        assert isinstance(snap, MarketSnapshot)

    def test_date_is_utc_aware(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
        )
        assert snap.date.tzinfo is not None

    def test_risk_free_rate_positive(self):
        snap = build_live_market_snapshot(
            ticker="SPY",
            price_data=_price_df(),
            current_price=462.5,
            iv_data=_iv_data(),
            technical_signals=_tech_signals(),
        )
        assert snap.risk_free_rate > 0
