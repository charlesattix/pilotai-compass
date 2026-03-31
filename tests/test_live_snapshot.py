"""Tests for shared/live_snapshot.py — MarketSnapshot builder from live data."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from shared.live_snapshot import build_live_snapshot


def _make_price_df(ticker="SPY", days=60, base_price=550.0):
    """Create a realistic price DataFrame for testing."""
    dates = pd.bdate_range(end="2026-03-28", periods=days)
    n = len(dates)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, n)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.005,
        "Low": prices * 0.995,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)
    return df


def _make_vix_df(days=60, base_vix=18.0):
    """Create a VIX DataFrame for testing."""
    dates = pd.bdate_range(end="2026-03-28", periods=days)
    n = len(dates)
    np.random.seed(99)
    vix_vals = base_vix + np.random.normal(0, 2, n).cumsum()
    vix_vals = np.clip(vix_vals, 10, 80)

    return pd.DataFrame({
        "Open": vix_vals * 0.99,
        "High": vix_vals * 1.02,
        "Low": vix_vals * 0.98,
        "Close": vix_vals,
        "Volume": np.zeros(n),
    }, index=dates)


class TestBuildLiveSnapshot:
    def test_snapshot_structure(self):
        """Verify MarketSnapshot has all required fields."""
        spy_df = _make_price_df("SPY")
        vix_df = _make_vix_df()

        mock_cache = MagicMock()
        def side_effect(ticker, period="1y"):
            if ticker == "^VIX":
                return vix_df
            return spy_df
        mock_cache.get_history.side_effect = side_effect

        snapshot = build_live_snapshot(
            tickers=["SPY"],
            data_cache=mock_cache,
        )

        # Core fields
        assert isinstance(snapshot.date, datetime)
        assert "SPY" in snapshot.prices
        assert "SPY" in snapshot.price_data
        assert "SPY" in snapshot.open_prices
        assert "SPY" in snapshot.iv_rank
        assert "SPY" in snapshot.realized_vol
        assert "SPY" in snapshot.rsi
        assert snapshot.vix > 0
        assert snapshot.risk_free_rate > 0

    def test_iv_rank_calculation(self):
        """IV rank uses same VIX percentile method as backtester."""
        spy_df = _make_price_df("SPY")
        vix_df = _make_vix_df(days=260, base_vix=20.0)

        mock_cache = MagicMock()
        def side_effect(ticker, period="1y"):
            if ticker == "^VIX":
                return vix_df
            return spy_df
        mock_cache.get_history.side_effect = side_effect

        snapshot = build_live_snapshot(
            tickers=["SPY"],
            data_cache=mock_cache,
        )

        # IV rank should be a reasonable value between 0-100
        iv_rank = snapshot.iv_rank.get("SPY", -1)
        assert 0 <= iv_rank <= 100

    def test_realized_vol_calculation(self):
        """Realized vol uses ATR-based formula, same as backtester."""
        spy_df = _make_price_df("SPY", days=60, base_price=550.0)
        vix_df = _make_vix_df()

        mock_cache = MagicMock()
        def side_effect(ticker, period="1y"):
            if ticker == "^VIX":
                return vix_df
            return spy_df
        mock_cache.get_history.side_effect = side_effect

        snapshot = build_live_snapshot(
            tickers=["SPY"],
            data_cache=mock_cache,
        )

        rv = snapshot.realized_vol.get("SPY", -1)
        # Should be between 0.10 and 1.00 (clipped range)
        assert 0.10 <= rv <= 1.00

    def test_rsi_calculation(self):
        """RSI uses 14-period calculation."""
        spy_df = _make_price_df("SPY", days=60)
        vix_df = _make_vix_df()

        mock_cache = MagicMock()
        def side_effect(ticker, period="1y"):
            if ticker == "^VIX":
                return vix_df
            return spy_df
        mock_cache.get_history.side_effect = side_effect

        snapshot = build_live_snapshot(
            tickers=["SPY"],
            data_cache=mock_cache,
        )

        rsi = snapshot.rsi.get("SPY", -1)
        assert 0 <= rsi <= 100

    def test_empty_data_handled(self):
        """Empty DataFrames don't crash the snapshot builder."""
        mock_cache = MagicMock()
        mock_cache.get_history.return_value = pd.DataFrame()

        snapshot = build_live_snapshot(
            tickers=["SPY"],
            data_cache=mock_cache,
        )

        assert snapshot.prices == {}
        assert snapshot.vix == 20.0  # default

    def test_multiple_tickers(self):
        """Snapshot works with multiple tickers."""
        spy_df = _make_price_df("SPY", base_price=550)
        qqq_df = _make_price_df("QQQ", base_price=480)
        vix_df = _make_vix_df()

        mock_cache = MagicMock()
        def side_effect(ticker, period="1y"):
            if ticker == "^VIX":
                return vix_df
            if ticker == "QQQ":
                return qqq_df
            return spy_df
        mock_cache.get_history.side_effect = side_effect

        snapshot = build_live_snapshot(
            tickers=["SPY", "QQQ"],
            data_cache=mock_cache,
        )

        assert "SPY" in snapshot.prices
        assert "QQQ" in snapshot.prices
        assert len(snapshot.price_data) == 2
