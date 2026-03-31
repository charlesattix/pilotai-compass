"""Tests for DataCache."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from shared.data_cache import DataCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(periods=100, seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2025-01-01', periods=periods, freq='B')
    close = 450.0 + np.cumsum(np.random.randn(periods) * 2)
    return pd.DataFrame({
        'Open': close - 0.5,
        'High': close + 1.0,
        'Low': close - 1.0,
        'Close': close,
        'Volume': np.random.randint(1_000_000, 5_000_000, periods),
    }, index=dates)


def _patch_ticker_history(mock_ticker_cls, df=None):
    """Configure mock yf.Ticker so .history() returns the given DataFrame."""
    if df is None:
        df = _make_price_df()
    mock_instance = MagicMock()
    mock_instance.history.return_value = df
    mock_ticker_cls.return_value = mock_instance
    return mock_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataCache:

    @patch('shared.data_cache.yf.Ticker')
    def test_get_history_caches(self, mock_ticker_cls):
        """Second call should use cached data, not download again."""
        mock_inst = _patch_ticker_history(mock_ticker_cls)
        cache = DataCache(ttl_seconds=60)

        result1 = cache.get_history('SPY')
        result2 = cache.get_history('SPY')

        assert mock_ticker_cls.call_count == 1
        assert mock_inst.history.call_count == 1
        assert len(result1) == len(result2)

    @patch('shared.data_cache.yf.Ticker')
    def test_get_history_returns_copy(self, mock_ticker_cls):
        """Each call should return a copy, not a reference to cached data."""
        _patch_ticker_history(mock_ticker_cls)
        cache = DataCache(ttl_seconds=60)

        result1 = cache.get_history('SPY')
        result2 = cache.get_history('SPY')

        # Modifying one should not affect the other
        result1.iloc[0, 0] = -9999
        assert result2.iloc[0, 0] != -9999

    @patch('shared.data_cache.yf.Ticker')
    def test_different_tickers_download_separately(self, mock_ticker_cls):
        """Different tickers should each trigger their own download."""
        _patch_ticker_history(mock_ticker_cls)
        cache = DataCache(ttl_seconds=60)

        cache.get_history('SPY')
        cache.get_history('QQQ')

        assert mock_ticker_cls.call_count == 2

    @patch('shared.data_cache.yf.Ticker')
    def test_clear_resets_cache(self, mock_ticker_cls):
        """After clear(), next call should download again."""
        _patch_ticker_history(mock_ticker_cls)
        cache = DataCache(ttl_seconds=60)

        cache.get_history('SPY')
        cache.clear()
        cache.get_history('SPY')

        assert mock_ticker_cls.call_count == 2

    def test_get_ticker_obj(self):
        """get_ticker_obj should return a yf.Ticker object."""
        cache = DataCache()
        ticker = cache.get_ticker_obj('SPY')
        assert ticker is not None
