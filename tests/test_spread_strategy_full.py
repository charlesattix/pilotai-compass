"""Tests for CreditSpreadStrategy spread-finding methods."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from strategy.spread_strategy import CreditSpreadStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    return {
        'strategy': {
            'min_dte': 30,
            'max_dte': 45,
            'min_delta': 0.10,
            'max_delta': 0.15,
            'spread_width': 5,
            'min_iv_rank': 25,
            'min_iv_percentile': 25,
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
        'risk': {
            'account_size': 100000,
            'max_risk_per_trade': 2.0,
            'max_positions': 7,
            'profit_target': 50,
            'stop_loss_multiplier': 2.5,
            'delta_threshold': 0.30,
            'min_credit_pct': 20,
        },
    }


def _make_option_chain(current_price=450.0, exp_days=35):
    """Create a synthetic options chain with puts and calls."""
    exp_date = datetime.now() + timedelta(days=exp_days)
    rows = []
    for strike in np.arange(430, 471, 5):
        # Puts
        put_delta = -0.05 - (current_price - strike) / 200.0
        put_delta = max(-0.50, min(-0.01, put_delta))
        put_bid = max(0.10, (current_price - strike) * 0.05 + 2.0)
        rows.append({
            'strike': float(strike),
            'bid': round(put_bid, 2),
            'ask': round(put_bid + 0.50, 2),
            'type': 'put',
            'delta': round(put_delta, 4),
            'expiration': exp_date,
            'iv': 0.25,
        })
        # Calls
        call_delta = 0.50 - (strike - current_price) / 200.0
        call_delta = max(0.01, min(0.50, call_delta))
        call_bid = max(0.10, (strike - current_price) * 0.05 + 2.0)
        rows.append({
            'strike': float(strike),
            'bid': round(call_bid, 2),
            'ask': round(call_bid + 0.50, 2),
            'type': 'call',
            'delta': round(call_delta, 4),
            'expiration': exp_date,
            'iv': 0.25,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFilterByDTE:

    def test_filters_within_range(self):
        """Only expirations within min_dte to max_dte should be returned."""
        strategy = CreditSpreadStrategy(_make_config())
        now = datetime.now()
        chain = pd.DataFrame({
            'expiration': [
                now + timedelta(days=10),   # too close
                now + timedelta(days=35),   # good
                now + timedelta(days=100),  # too far
            ],
            'strike': [100, 100, 100],
        })
        valid = strategy._filter_by_dte(chain)
        assert len(valid) == 1

    def test_empty_when_no_valid_expirations(self):
        """No matching expirations should return empty list."""
        strategy = CreditSpreadStrategy(_make_config())
        now = datetime.now()
        chain = pd.DataFrame({
            'expiration': [now + timedelta(days=5)],
            'strike': [100],
        })
        valid = strategy._filter_by_dte(chain)
        assert len(valid) == 0


class TestFindSpreads:

    def test_find_bull_put_spreads(self):
        """_find_bull_put_spreads should return list of dicts."""
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        exp = chain['expiration'].iloc[0]
        spreads = strategy._find_bull_put_spreads(
            'SPY', chain, 450.0, exp
        )
        assert isinstance(spreads, list)
        for s in spreads:
            assert s['type'] == 'bull_put_spread'
            assert s['long_strike'] < s['short_strike']

    def test_find_bear_call_spreads(self):
        """_find_bear_call_spreads should return list of dicts."""
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        exp = chain['expiration'].iloc[0]
        spreads = strategy._find_bear_call_spreads(
            'SPY', chain, 450.0, exp
        )
        assert isinstance(spreads, list)
        for s in spreads:
            assert s['type'] == 'bear_call_spread'
            assert s['long_strike'] > s['short_strike']

    def test_find_spreads_unified_matches_wrappers(self):
        """_find_spreads should produce the same results as the thin wrappers."""
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        exp = chain['expiration'].iloc[0]

        bull_via_wrapper = strategy._find_bull_put_spreads('SPY', chain, 450.0, exp)
        bull_via_unified = strategy._find_spreads('SPY', chain, 450.0, exp, 'bull_put')
        assert len(bull_via_wrapper) == len(bull_via_unified)

    def test_empty_on_no_matching_options(self):
        """Should return empty list when no options match delta criteria."""
        strategy = CreditSpreadStrategy(_make_config())
        # Chain with all deltas outside range
        chain = pd.DataFrame({
            'strike': [100.0],
            'bid': [1.0],
            'ask': [1.5],
            'type': ['put'],
            'delta': [-0.50],  # way too high
            'expiration': [datetime.now() + timedelta(days=35)],
            'iv': [0.25],
        })
        exp = chain['expiration'].iloc[0]
        spreads = strategy._find_bull_put_spreads('SPY', chain, 450.0, exp)
        assert spreads == []


class TestCheckConditions:

    def test_bullish_conditions_pass(self):
        """Bullish conditions should pass with high IV and bullish trend."""
        strategy = CreditSpreadStrategy(_make_config())
        tech = {'trend': 'bullish', 'rsi': 50}
        iv = {'iv_rank': 50, 'iv_percentile': 50}
        assert strategy._check_bullish_conditions(tech, iv) is True

    def test_bullish_conditions_fail_low_iv(self):
        """Bullish conditions should fail with low IV."""
        strategy = CreditSpreadStrategy(_make_config())
        tech = {'trend': 'bullish', 'rsi': 50}
        iv = {'iv_rank': 10, 'iv_percentile': 10}
        assert strategy._check_bullish_conditions(tech, iv) is False

    def test_bearish_conditions_pass(self):
        """Bearish conditions should pass with high IV and bearish trend."""
        strategy = CreditSpreadStrategy(_make_config())
        tech = {'trend': 'bearish', 'rsi': 50}
        iv = {'iv_rank': 50, 'iv_percentile': 50}
        assert strategy._check_bearish_conditions(tech, iv) is True



class TestEvaluateSpreadOpportunity:

    def test_returns_list(self):
        """evaluate_spread_opportunity should return a list."""
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        tech = {'trend': 'bullish', 'rsi': 50}
        iv = {'iv_rank': 50, 'iv_percentile': 50}
        result = strategy.evaluate_spread_opportunity(
            ticker='SPY',
            option_chain=chain,
            technical_signals=tech,
            iv_data=iv,
            current_price=450.0,
        )
        assert isinstance(result, list)
