"""Tests for OptionsAnalyzer."""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from strategy.options_analyzer import OptionsAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Build a minimal config for OptionsAnalyzer."""
    config = {
        'strategy': {
            'min_dte': 30,
            'max_dte': 45,
        },
        'data': {
            'provider': 'yfinance',
        },
    }
    config.update(overrides)
    return config


def _make_options_df(n_strikes=5, exp_date=None):
    """Create a synthetic options DataFrame."""
    if exp_date is None:
        exp_date = datetime.now() + timedelta(days=35)
    strikes = np.arange(100, 100 + n_strikes * 5, 5, dtype=float)
    rows = []
    for s in strikes:
        for opt_type in ['call', 'put']:
            rows.append({
                'strike': s,
                'bid': 2.0,
                'ask': 2.5,
                'type': opt_type,
                'expiration': exp_date,
                'iv': 0.25,
                'delta': 0.3 if opt_type == 'call' else -0.3,
                'volume': 100,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetOptionsChain:

    @patch('strategy.options_analyzer.yf.Ticker')
    def test_returns_dataframe_on_success(self, mock_ticker_cls):
        """get_options_chain should return a non-empty DataFrame."""
        exp_date = (datetime.now() + timedelta(days=35)).strftime('%Y-%m-%d')
        mock_ticker = MagicMock()
        mock_ticker.options = [exp_date]

        calls_df = pd.DataFrame({
            'strike': [100.0, 105.0],
            'bid': [3.0, 2.0],
            'ask': [3.5, 2.5],
            'impliedVolatility': [0.25, 0.30],
            'volume': [100, 200],
        })
        puts_df = pd.DataFrame({
            'strike': [100.0, 105.0],
            'bid': [2.0, 3.0],
            'ask': [2.5, 3.5],
            'impliedVolatility': [0.25, 0.30],
            'volume': [100, 200],
        })
        chain_mock = MagicMock()
        chain_mock.calls = calls_df
        chain_mock.puts = puts_df
        mock_ticker.option_chain.return_value = chain_mock
        mock_ticker_cls.return_value = mock_ticker

        analyzer = OptionsAnalyzer(_make_config())
        result = analyzer.get_options_chain('SPY')

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @patch('strategy.options_analyzer.yf.Ticker')
    def test_returns_empty_when_no_options(self, mock_ticker_cls):
        """get_options_chain should return empty DataFrame when no options exist."""
        mock_ticker = MagicMock()
        mock_ticker.options = []
        mock_ticker_cls.return_value = mock_ticker

        analyzer = OptionsAnalyzer(_make_config())
        result = analyzer.get_options_chain('NOOPT')

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('strategy.options_analyzer.yf.Ticker')
    def test_returns_empty_on_exception(self, mock_ticker_cls):
        """get_options_chain should return empty DataFrame on error."""
        mock_ticker_cls.side_effect = Exception("network error")

        analyzer = OptionsAnalyzer(_make_config())
        result = analyzer.get_options_chain('ERR')

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestCleanOptionsData:

    def test_renames_columns(self):
        """_clean_options_data should rename yfinance columns to standard names."""
        analyzer = OptionsAnalyzer(_make_config())
        exp_date = datetime.now() + timedelta(days=35)
        df = pd.DataFrame({
            'strike': [100.0],
            'bid': [2.0],
            'ask': [2.5],
            'type': ['call'],
            'expiration': [exp_date],
            'impliedVolatility': [0.25],
        })
        result = analyzer._clean_options_data(df)
        assert 'iv' in result.columns

    def test_removes_zero_bid_ask(self):
        """Rows with zero bid or ask should be filtered out."""
        analyzer = OptionsAnalyzer(_make_config())
        exp_date = datetime.now() + timedelta(days=35)
        df = pd.DataFrame({
            'strike': [100.0, 105.0],
            'bid': [0.0, 2.0],
            'ask': [2.5, 2.5],
            'type': ['call', 'call'],
            'expiration': [exp_date, exp_date],
            'iv': [0.25, 0.30],
        })
        result = analyzer._clean_options_data(df)
        assert len(result) == 1
        assert result.iloc[0]['strike'] == 105.0


class TestDTEFiltering:

    @patch('strategy.options_analyzer.yf.Ticker')
    def test_filters_expirations_by_dte(self, mock_ticker_cls):
        """Only expirations within DTE range (+/- buffer) should be downloaded."""
        now = datetime.now()
        exp_close = (now + timedelta(days=35)).strftime('%Y-%m-%d')  # within range
        exp_far = (now + timedelta(days=120)).strftime('%Y-%m-%d')    # outside range
        exp_near = (now + timedelta(days=5)).strftime('%Y-%m-%d')     # outside range

        mock_ticker = MagicMock()
        mock_ticker.options = [exp_near, exp_close, exp_far]

        calls_df = pd.DataFrame({
            'strike': [100.0],
            'bid': [3.0],
            'ask': [3.5],
            'impliedVolatility': [0.25],
            'volume': [100],
        })
        puts_df = pd.DataFrame({
            'strike': [100.0],
            'bid': [2.0],
            'ask': [2.5],
            'impliedVolatility': [0.25],
            'volume': [100],
        })
        chain_mock = MagicMock()
        chain_mock.calls = calls_df
        chain_mock.puts = puts_df
        mock_ticker.option_chain.return_value = chain_mock
        mock_ticker_cls.return_value = mock_ticker

        analyzer = OptionsAnalyzer(_make_config())
        analyzer._get_chain_yfinance('SPY')

        # option_chain should only be called once (for exp_close)
        assert mock_ticker.option_chain.call_count == 1


class TestCalculateIVRank:

    @patch('strategy.options_analyzer.yf.Ticker')
    def test_returns_valid_iv_rank(self, mock_ticker_cls):
        """calculate_iv_rank should return a dict with iv_rank key."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=252, freq='B')
        prices = 100.0 + np.cumsum(np.random.randn(252) * 0.5)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {'Close': prices}, index=dates
        )
        mock_ticker_cls.return_value = mock_ticker

        analyzer = OptionsAnalyzer(_make_config())
        result = analyzer.calculate_iv_rank('SPY', current_iv=25.0)

        assert 'iv_rank' in result
        assert 'iv_percentile' in result
        assert isinstance(result['iv_rank'], float)


class TestGetCurrentIV:

    def test_returns_median_iv(self):
        """get_current_iv should return the median IV times 100."""
        analyzer = OptionsAnalyzer(_make_config())
        df = _make_options_df()
        result = analyzer.get_current_iv(df)
        assert result == pytest.approx(25.0, abs=0.1)

    def test_returns_zero_for_empty_chain(self):
        """get_current_iv should return 0.0 for empty chain."""
        analyzer = OptionsAnalyzer(_make_config())
        result = analyzer.get_current_iv(pd.DataFrame())
        assert result == 0.0
