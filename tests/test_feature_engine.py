"""Tests for FeatureEngine."""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from compass.features import FeatureEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(periods=130, seed=42):
    """Create a synthetic price DataFrame that mimics data_cache.get_history output."""
    np.random.seed(seed)
    dates = pd.date_range('2025-06-01', periods=periods, freq='B')
    close = 450.0 + np.cumsum(np.random.randn(periods) * 2)
    return pd.DataFrame({
        'Open': close - np.random.rand(periods),
        'High': close + np.abs(np.random.randn(periods)),
        'Low': close - np.abs(np.random.randn(periods)),
        'Close': close,
        'Volume': np.random.randint(1_000_000, 5_000_000, periods),
    }, index=dates)


def _make_options_df():
    exp_date = datetime.now() + timedelta(days=35)
    return pd.DataFrame({
        'strike': [100.0, 105.0],
        'bid': [2.0, 3.0],
        'ask': [2.5, 3.5],
        'type': ['put', 'call'],
        'expiration': [exp_date, exp_date],
        'iv': [0.25, 0.30],
    })


_MOCK_PRICE_DF = _make_price_df()


def _make_data_cache():
    """Create a mock data_cache that returns price data for any ticker."""
    mock_cache = MagicMock()
    mock_cache.get_history.return_value = _MOCK_PRICE_DF.copy()
    mock_cache.get_ticker_obj.return_value = MagicMock(calendar=None)
    return mock_cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildFeatures:

    def test_returns_dict_with_ticker(self):
        """build_features should return a dict containing the ticker key."""
        cache = _make_data_cache()
        engine = FeatureEngine(data_cache=cache)
        result = engine.build_features(
            ticker='SPY',
            current_price=450.0,
            options_chain=_make_options_df(),
        )
        assert result is not None
        assert result['ticker'] == 'SPY'

    def test_contains_technical_features(self):
        """build_features should include technical features like rsi_14."""
        cache = _make_data_cache()
        engine = FeatureEngine(data_cache=cache)
        result = engine.build_features(
            ticker='SPY',
            current_price=450.0,
            options_chain=_make_options_df(),
        )
        assert result is not None
        assert 'rsi_14' in result
        assert 'macd' in result

    def test_returns_none_without_data_cache(self):
        """build_features should return None when no data_cache is provided."""
        engine = FeatureEngine()
        result = engine.build_features(
            ticker='SPY',
            current_price=450.0,
            options_chain=_make_options_df(),
        )
        assert result is None


class TestComputeTechnicalFeatures:

    def test_rsi_in_range(self):
        """RSI should be between 0 and 100."""
        cache = _make_data_cache()
        engine = FeatureEngine(data_cache=cache)
        features = engine._compute_technical_features('SPY', 450.0)
        assert features is not None
        assert 0 <= features['rsi_14'] <= 100

    def test_atr_pct_positive(self):
        """ATR percentage should be positive."""
        cache = _make_data_cache()
        engine = FeatureEngine(data_cache=cache)
        features = engine._compute_technical_features('SPY', 450.0)
        assert features is not None
        assert features['atr_pct'] > 0

    def test_returns_none_on_empty_data(self):
        """Should return None when data_cache returns empty data."""
        cache = MagicMock()
        cache.get_history.return_value = pd.DataFrame()
        engine = FeatureEngine(data_cache=cache)
        features = engine._compute_technical_features('BAD', 100.0)
        assert features is None

    def test_returns_none_without_cache(self):
        """Should return None when no data_cache is provided."""
        engine = FeatureEngine()
        features = engine._compute_technical_features('SPY', 450.0)
        assert features is None


class TestComputeVolatilityFeatures:

    def test_realized_vol_positive(self):
        """Realized volatility values should be positive."""
        cache = _make_data_cache()
        engine = FeatureEngine(data_cache=cache)
        features = engine._compute_volatility_features(
            'SPY', _make_options_df(), None
        )
        assert features is not None
        assert features['realized_vol_10d'] > 0
        assert features['realized_vol_20d'] > 0

    def test_iv_from_analysis(self):
        """When iv_analysis is provided, use its values."""
        cache = _make_data_cache()
        engine = FeatureEngine(data_cache=cache)
        iv_analysis = {
            'iv_rank_percentile': {
                'available': True,
                'iv_rank': 75.0,
                'iv_percentile': 80.0,
                'current_iv': 30.0,
            },
            'skew': {'available': False},
        }
        features = engine._compute_volatility_features(
            'SPY', _make_options_df(), iv_analysis
        )
        assert features is not None
        assert features['iv_rank'] == 75.0

    def test_returns_none_without_cache(self):
        """Should return None when no data_cache is provided."""
        engine = FeatureEngine()
        features = engine._compute_volatility_features(
            'SPY', _make_options_df(), None
        )
        assert features is None


class TestComputeMarketFeatures:

    def test_returns_vix_level(self):
        """Market features should contain vix_level."""
        cache = _make_data_cache()
        engine = FeatureEngine(data_cache=cache)
        features = engine.compute_market_features()
        assert features is not None
        assert 'vix_level' in features

    def test_returns_none_on_empty(self):
        """Should return None when data_cache returns empty data."""
        cache = MagicMock()
        cache.get_history.return_value = pd.DataFrame()
        engine = FeatureEngine(data_cache=cache)
        features = engine.compute_market_features()
        assert features is None

    def test_returns_none_without_cache(self):
        """Should return None when no data_cache is provided."""
        engine = FeatureEngine()
        features = engine.compute_market_features()
        assert features is None


class TestDataCacheBehavior:

    def test_uses_data_cache_when_provided(self):
        """When data_cache is set, _download should call data_cache.get_history."""
        mock_cache = MagicMock()
        mock_cache.get_history.return_value = _MOCK_PRICE_DF.copy()

        engine = FeatureEngine(data_cache=mock_cache)
        result = engine._download('SPY', period='6mo')

        mock_cache.get_history.assert_called_once_with('SPY', '6mo')
        assert result is not None
        assert not result.empty

    def test_returns_none_without_cache(self):
        """Without data_cache, _download should return None."""
        engine = FeatureEngine()
        result = engine._download('SPY', period='6mo')
        assert result is None


class TestGetFeatureNames:

    def test_returns_list(self):
        """get_feature_names should return a non-empty list of strings."""
        engine = FeatureEngine()
        names = engine.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 20
        assert all(isinstance(n, str) for n in names)


class TestDerivedFeatures:

    def test_rsi_oversold_flag(self):
        """rsi_oversold should be 1.0 when RSI < 30."""
        engine = FeatureEngine()
        features = {'rsi_14': 25.0, 'iv_rank': 50.0, 'realized_vol_20d': 20.0,
                     'current_iv': 25.0, 'atr_pct': 2.0, 'return_20d': 5.0}
        derived = engine._compute_derived_features(features)
        assert derived['rsi_oversold'] == 1.0
        assert derived['rsi_overbought'] == 0.0

    def test_vol_premium_calculation(self):
        """vol_premium should be current_iv - realized_vol_20d."""
        engine = FeatureEngine()
        features = {'rsi_14': 50.0, 'iv_rank': 50.0, 'realized_vol_20d': 15.0,
                     'current_iv': 25.0, 'atr_pct': 2.0, 'return_20d': 5.0}
        derived = engine._compute_derived_features(features)
        assert derived['vol_premium'] == pytest.approx(10.0)
