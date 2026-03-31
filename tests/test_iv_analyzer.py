"""Tests for ml.iv_analyzer.IVAnalyzer."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from compass.iv_surface import IVAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_options_chain(
    current_price: float = 450.0,
    put_iv_base: float = 0.25,
    call_iv_base: float = 0.22,
    otm_put_iv_premium: float = 0.08,
    otm_call_iv_premium: float = 0.03,
    near_dte: int = 30,
    far_dte: int = 75,
) -> pd.DataFrame:
    """
    Build a synthetic options chain DataFrame around *current_price*.

    Returns a DataFrame with columns:
        strike, bid, ask, volume, iv, type, expiration
    """
    now = datetime.now(timezone.utc)
    near_exp = now + timedelta(days=near_dte)
    far_exp = now + timedelta(days=far_dte)

    rows = []
    strikes = np.arange(
        current_price * 0.85, current_price * 1.15 + 1, 5
    )

    for strike in strikes:
        moneyness = strike / current_price

        # Put IV rises for deeper OTM puts (lower strikes)
        if moneyness <= 0.92:
            p_iv = put_iv_base + otm_put_iv_premium
        elif moneyness <= 0.98:
            p_iv = put_iv_base + otm_put_iv_premium * (0.98 - moneyness) / 0.06
        else:
            p_iv = put_iv_base

        # Call IV rises for deeper OTM calls (higher strikes)
        if moneyness >= 1.08:
            c_iv = call_iv_base + otm_call_iv_premium
        elif moneyness >= 1.02:
            c_iv = call_iv_base + otm_call_iv_premium * (moneyness - 1.02) / 0.06
        else:
            c_iv = call_iv_base

        for exp in (near_exp, far_exp):
            # Puts
            rows.append({
                'strike': float(strike),
                'bid': round(max(0.10, p_iv * 10), 2),
                'ask': round(max(0.15, p_iv * 10 + 0.05), 2),
                'volume': 100,
                'iv': p_iv,
                'type': 'put',
                'expiration': exp,
            })
            # Calls
            rows.append({
                'strike': float(strike),
                'bid': round(max(0.10, c_iv * 10), 2),
                'ask': round(max(0.15, c_iv * 10 + 0.05), 2),
                'volume': 100,
                'iv': c_iv,
                'type': 'call',
                'expiration': exp,
            })

    df = pd.DataFrame(rows)
    df['expiration'] = pd.to_datetime(df['expiration'], utc=True)
    return df


def _make_iv_history(low: float = 12.0, high: float = 35.0, n: int = 252) -> pd.Series:
    """Return a linearly spaced historical volatility Series."""
    return pd.Series(np.linspace(low, high, n))


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestAnalyzeSurface:
    """Tests for IVAnalyzer.analyze_surface."""

    @patch.object(IVAnalyzer, '_get_iv_history', return_value=_make_iv_history())
    def test_returns_expected_keys(self, _mock_hist):
        """analyze_surface should return a dict with all top-level keys."""
        analyzer = IVAnalyzer()
        chain = _make_options_chain()
        result = analyzer.analyze_surface('SPY', chain, current_price=450.0)

        expected_keys = {
            'ticker', 'timestamp', 'skew', 'term_structure',
            'iv_rank_percentile', 'signals',
        }
        assert expected_keys.issubset(result.keys())
        assert result['ticker'] == 'SPY'

    @patch.object(IVAnalyzer, '_get_iv_history', return_value=_make_iv_history())
    def test_signals_contains_required_fields(self, _mock_hist):
        """The signals sub-dict should contain the standard keys."""
        analyzer = IVAnalyzer()
        chain = _make_options_chain()
        result = analyzer.analyze_surface('SPY', chain, current_price=450.0)

        signals = result['signals']
        for key in ('bull_put_favorable', 'bear_call_favorable',
                     'overall_signal', 'reasoning'):
            assert key in signals

    def test_empty_chain_returns_default_analysis(self):
        """An empty options chain should yield the default (neutral) analysis."""
        analyzer = IVAnalyzer()
        empty_chain = pd.DataFrame(
            columns=['strike', 'bid', 'ask', 'volume', 'iv', 'type', 'expiration']
        )
        result = analyzer.analyze_surface('SPY', empty_chain, current_price=450.0)

        assert result['signals']['overall_signal'] == 'neutral'
        assert result['signals']['bull_put_favorable'] is False
        assert result['signals']['bear_call_favorable'] is False
        assert result['skew']['available'] is False


class TestSkewMetrics:
    """Tests for IVAnalyzer._compute_skew_metrics."""

    def test_steep_put_skew_detected(self):
        """
        When OTM put IV is significantly higher than ATM call IV the
        put_call_skew_ratio should exceed 1.15.
        """
        analyzer = IVAnalyzer()
        # Large OTM put premium, small OTM call premium -> steep put skew
        chain = _make_options_chain(
            current_price=450.0,
            put_iv_base=0.25,
            call_iv_base=0.22,
            otm_put_iv_premium=0.14,
            otm_call_iv_premium=0.02,
        )
        metrics = analyzer._compute_skew_metrics(chain, current_price=450.0)

        assert metrics['available'] is True
        assert metrics['put_call_skew_ratio'] > 1.15

    def test_atm_iv_values_positive(self):
        """ATM implied vol values must be positive."""
        analyzer = IVAnalyzer()
        chain = _make_options_chain()
        metrics = analyzer._compute_skew_metrics(chain, current_price=450.0)

        assert metrics['available'] is True
        assert metrics['atm_put_iv'] > 0
        assert metrics['atm_call_iv'] > 0

    def test_empty_chain_returns_unavailable(self):
        """An empty chain should return available=False."""
        analyzer = IVAnalyzer()
        empty = pd.DataFrame(
            columns=['strike', 'bid', 'ask', 'volume', 'iv', 'type', 'expiration']
        )
        metrics = analyzer._compute_skew_metrics(empty, current_price=450.0)
        assert metrics['available'] is False

    def test_chain_without_iv_column_returns_unavailable(self):
        """A chain missing the 'iv' column should return available=False."""
        analyzer = IVAnalyzer()
        chain = _make_options_chain()
        chain = chain.drop(columns=['iv'])
        metrics = analyzer._compute_skew_metrics(chain, current_price=450.0)
        assert metrics['available'] is False


class TestIVRankPercentile:
    """Tests for IVAnalyzer._compute_iv_rank_percentile."""

    @patch.object(IVAnalyzer, '_get_iv_history')
    def test_iv_rank_computed_correctly(self, mock_hist):
        """
        With a linear history from 12 to 35 and a current IV at the midpoint,
        iv_rank should be close to 50.
        """
        mock_hist.return_value = _make_iv_history(low=12.0, high=35.0)

        analyzer = IVAnalyzer()
        # Craft a chain whose median iv maps to the midpoint of [12, 35]
        (12.0 + 35.0) / 2.0  # 23.5
        # median iv * 100 = midpoint  => median iv = 0.235
        chain = _make_options_chain(
            put_iv_base=0.235, call_iv_base=0.235,
            otm_put_iv_premium=0.0, otm_call_iv_premium=0.0,
        )
        result = analyzer._compute_iv_rank_percentile('SPY', chain)

        assert result['available'] is True
        assert result['iv_rank'] == pytest.approx(50.0, abs=1.0)
        assert result['iv_min_52w'] == pytest.approx(12.0, abs=0.1)
        assert result['iv_max_52w'] == pytest.approx(35.0, abs=0.1)

    @patch.object(IVAnalyzer, '_get_iv_history')
    def test_high_iv_rank(self, mock_hist):
        """When current IV is near the historical max, iv_rank should be high."""
        mock_hist.return_value = _make_iv_history(low=12.0, high=35.0)

        analyzer = IVAnalyzer()
        # median iv * 100 ~ 34 => near the top of range
        chain = _make_options_chain(
            put_iv_base=0.34, call_iv_base=0.34,
            otm_put_iv_premium=0.0, otm_call_iv_premium=0.0,
        )
        result = analyzer._compute_iv_rank_percentile('SPY', chain)

        assert result['available'] is True
        assert result['iv_rank'] > 90

    @patch.object(IVAnalyzer, '_get_iv_history', return_value=None)
    def test_no_history_returns_unavailable(self, _mock_hist):
        """Without historical IV data the result should be available=False."""
        analyzer = IVAnalyzer()
        chain = _make_options_chain()
        result = analyzer._compute_iv_rank_percentile('SPY', chain)

        assert result['available'] is False

    @patch.object(IVAnalyzer, '_get_iv_history')
    def test_short_history_returns_unavailable(self, mock_hist):
        """A history shorter than 50 observations is insufficient."""
        mock_hist.return_value = pd.Series(np.linspace(15, 25, 30))

        analyzer = IVAnalyzer()
        chain = _make_options_chain()
        result = analyzer._compute_iv_rank_percentile('SPY', chain)

        assert result['available'] is False

    @patch.object(IVAnalyzer, '_get_iv_history')
    def test_result_includes_statistical_fields(self, mock_hist):
        """Output should contain iv_mean_52w and iv_std_52w."""
        mock_hist.return_value = _make_iv_history()

        analyzer = IVAnalyzer()
        chain = _make_options_chain()
        result = analyzer._compute_iv_rank_percentile('SPY', chain)

        if result['available']:
            assert 'iv_mean_52w' in result
            assert 'iv_std_52w' in result
            assert result['iv_std_52w'] > 0


class TestSignals:
    """Tests for IVAnalyzer._generate_signals."""

    def test_high_iv_rank_makes_both_favorable(self):
        """An IV rank above 70 should flag both bull put and bear call as favorable."""
        analyzer = IVAnalyzer()

        skew = {
            'available': True,
            'put_call_skew_ratio': 1.0,
        }
        term = {'available': True, 'structure_type': 'contango'}
        iv_rp = {
            'available': True,
            'iv_rank': 80.0,
        }

        signals = analyzer._generate_signals(skew, term, iv_rp)

        assert signals['bull_put_favorable'] is True
        assert signals['bear_call_favorable'] is True
        assert signals['overall_signal'] == 'favorable_both'
        assert any('High IV rank' in r for r in signals['reasoning'])

    def test_low_iv_rank_adds_expansion_risk(self):
        """An IV rank below 30 should add an expansion-risk warning."""
        analyzer = IVAnalyzer()

        skew = {'available': True, 'put_call_skew_ratio': 1.0}
        term = {'available': True, 'structure_type': 'contango'}
        iv_rp = {'available': True, 'iv_rank': 20.0}

        signals = analyzer._generate_signals(skew, term, iv_rp)

        assert any('expansion risk' in r.lower() for r in signals['reasoning'])

    def test_backwardation_warning_in_reasoning(self):
        """Backwardation in term structure should add a caution note."""
        analyzer = IVAnalyzer()

        skew = {'available': True, 'put_call_skew_ratio': 1.0}
        term = {'available': True, 'structure_type': 'backwardation'}
        iv_rp = {'available': False}

        signals = analyzer._generate_signals(skew, term, iv_rp)

        assert any('backwardation' in r.lower() for r in signals['reasoning'])

    def test_steep_put_skew_favors_bull_put(self):
        """A put_call_skew_ratio > 1.15 should flag bull_put_favorable."""
        analyzer = IVAnalyzer()

        skew = {'available': True, 'put_call_skew_ratio': 1.30}
        term = {'available': False}
        iv_rp = {'available': False}

        signals = analyzer._generate_signals(skew, term, iv_rp)

        assert signals['bull_put_favorable'] is True
        assert signals['overall_signal'] == 'favorable_bull_put'

    def test_steep_call_skew_favors_bear_call(self):
        """A put_call_skew_ratio < 0.85 should flag bear_call_favorable."""
        analyzer = IVAnalyzer()

        skew = {'available': True, 'put_call_skew_ratio': 0.75}
        term = {'available': False}
        iv_rp = {'available': False}

        signals = analyzer._generate_signals(skew, term, iv_rp)

        assert signals['bear_call_favorable'] is True
        assert signals['overall_signal'] == 'favorable_bear_call'

    def test_insufficient_skew_data_neutral(self):
        """When skew data is unavailable the signal should remain neutral."""
        analyzer = IVAnalyzer()

        skew = {'available': False}
        term = {'available': False}
        iv_rp = {'available': False}

        signals = analyzer._generate_signals(skew, term, iv_rp)

        assert signals['overall_signal'] == 'neutral'
        assert signals['bull_put_favorable'] is False
        assert signals['bear_call_favorable'] is False

    def test_contango_noted_in_reasoning(self):
        """Normal contango term structure should be noted."""
        analyzer = IVAnalyzer()

        skew = {'available': True, 'put_call_skew_ratio': 1.0}
        term = {'available': True, 'structure_type': 'contango'}
        iv_rp = {'available': False}

        signals = analyzer._generate_signals(skew, term, iv_rp)

        assert any('contango' in r.lower() for r in signals['reasoning'])
