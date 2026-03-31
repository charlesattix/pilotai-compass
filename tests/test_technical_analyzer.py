"""Tests for TechnicalAnalyzer class."""
import numpy as np
import pandas as pd

from strategy.technical_analysis import TechnicalAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    return {
        'strategy': {
            'technical': {
                'use_trend_filter': True,
                'use_rsi_filter': True,
                'use_support_resistance': True,
                'fast_ma': 20,
                'slow_ma': 50,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
            },
        },
    }


def _make_price_data(n=100, seed=42, trend='up'):
    """Create synthetic OHLCV data."""
    np.random.seed(seed)
    dates = pd.date_range('2025-01-01', periods=n, freq='B')
    if trend == 'up':
        close = 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.5
    elif trend == 'down':
        close = 200.0 - np.arange(n) * 0.5 + np.random.randn(n) * 0.5
    else:
        close = 150.0 + np.random.randn(n) * 2
    return pd.DataFrame({
        'Open': close - np.random.rand(n) * 0.5,
        'High': close + np.abs(np.random.randn(n)),
        'Low': close - np.abs(np.random.randn(n)),
        'Close': close,
        'Volume': np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTechnicalAnalyzer:

    def test_analyze_returns_dict(self):
        """analyze() should return a non-empty dict."""
        ta = TechnicalAnalyzer(_make_config())
        signals = ta.analyze('SPY', _make_price_data())
        assert isinstance(signals, dict)
        assert 'ticker' in signals
        assert signals['ticker'] == 'SPY'

    def test_analyze_returns_empty_for_insufficient_data(self):
        """analyze() should return empty dict if data has fewer than 50 rows."""
        ta = TechnicalAnalyzer(_make_config())
        short_data = _make_price_data(n=20)
        signals = ta.analyze('SPY', short_data)
        assert signals == {}

    def test_trend_bullish_in_uptrend(self):
        """In an uptrend, trend should be 'bullish'."""
        ta = TechnicalAnalyzer(_make_config())
        signals = ta.analyze('SPY', _make_price_data(trend='up'))
        assert signals.get('trend') == 'bullish'

    def test_trend_bearish_in_downtrend(self):
        """In a downtrend, trend should be 'bearish'."""
        ta = TechnicalAnalyzer(_make_config())
        signals = ta.analyze('SPY', _make_price_data(trend='down'))
        assert signals.get('trend') == 'bearish'

    def test_rsi_present(self):
        """RSI should be included and within 0-100."""
        ta = TechnicalAnalyzer(_make_config())
        signals = ta.analyze('SPY', _make_price_data())
        assert 'rsi' in signals
        assert 0 <= signals['rsi'] <= 100

    def test_support_resistance_present(self):
        """Support and resistance levels should be in output."""
        ta = TechnicalAnalyzer(_make_config())
        signals = ta.analyze('SPY', _make_price_data())
        assert 'support_levels' in signals
        assert 'resistance_levels' in signals

    def test_consolidate_levels_empty(self):
        """_consolidate_levels with empty list returns empty list."""
        ta = TechnicalAnalyzer(_make_config())
        assert ta._consolidate_levels([]) == []

    def test_consolidate_levels_basic(self):
        """Levels within threshold should be consolidated."""
        ta = TechnicalAnalyzer(_make_config())
        levels = [100.0, 100.5, 105.0, 105.3]
        result = ta._consolidate_levels(levels, threshold=0.01)
        # 100.0 and 100.5 are within 1% -> consolidated
        # 105.0 and 105.3 are within 1% -> consolidated
        assert len(result) == 2

    def test_consolidate_levels_zero_guard(self):
        """_consolidate_levels should handle zero values without dividing by zero."""
        ta = TechnicalAnalyzer(_make_config())
        levels = [0.0, 0.5, 100.0]
        result = ta._consolidate_levels(levels, threshold=0.01)
        # 0.0 is first, then 0.5 should be appended (zero guard), then 100.0
        assert 0.0 in result
        assert len(result) >= 2

    def test_analyze_with_no_trend_filter(self):
        """With use_trend_filter=False, trend should not be in output."""
        config = _make_config()
        config['strategy']['technical']['use_trend_filter'] = False
        ta = TechnicalAnalyzer(config)
        signals = ta.analyze('SPY', _make_price_data())
        assert 'trend' not in signals

    def test_analyze_with_no_rsi_filter(self):
        """With use_rsi_filter=False, rsi should not be in output."""
        config = _make_config()
        config['strategy']['technical']['use_rsi_filter'] = False
        ta = TechnicalAnalyzer(config)
        signals = ta.analyze('SPY', _make_price_data())
        assert 'rsi' not in signals
