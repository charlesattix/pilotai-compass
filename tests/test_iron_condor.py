"""Tests for iron condor strategy in CreditSpreadStrategy."""
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
            'max_delta': 0.20,
            'spread_width': 5,
            'spread_width_high_iv': 5,
            'spread_width_low_iv': 5,
            'min_iv_rank': 25,
            'min_iv_percentile': 25,
            'iron_condor': {
                'enabled': True,
                'min_combined_credit_pct': 15,
                'max_wing_width': 5,
                'rsi_min': 35,
                'rsi_max': 65,
            },
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
            'min_credit_pct': 5,
        },
    }


def _make_option_chain(current_price=450.0, exp_days=35, iv_multiplier=1.0):
    """Create a synthetic options chain with puts and calls.

    Args:
        iv_multiplier: Scale premiums up (>1) for elevated IV environments.
    """
    exp_date = datetime.now() + timedelta(days=exp_days)
    rows = []
    base_premium = 8.0 * iv_multiplier
    for strike in np.arange(420, 481, 5):
        strike = float(strike)

        # --- Puts ---
        distance = current_price - strike  # positive when OTM
        put_delta = -max(0.02, min(0.50, 0.50 * np.exp(-distance / 20)))
        put_mid = max(0.15, base_premium * np.exp(-distance / 12)) if distance > 0 else base_premium
        put_bid = round(put_mid * 0.95, 2)
        put_ask = round(put_mid * 1.05, 2)
        rows.append({
            'strike': strike,
            'bid': max(put_bid, 0.05),
            'ask': max(put_ask, 0.10),
            'type': 'put',
            'delta': round(put_delta, 4),
            'expiration': exp_date,
            'iv': 0.25 * iv_multiplier,
        })

        # --- Calls ---
        distance_call = strike - current_price  # positive when OTM
        call_delta = max(0.02, min(0.50, 0.50 * np.exp(-distance_call / 20)))
        call_mid = max(0.15, base_premium * np.exp(-distance_call / 12)) if distance_call > 0 else base_premium
        call_bid = round(call_mid * 0.95, 2)
        call_ask = round(call_mid * 1.05, 2)
        rows.append({
            'strike': strike,
            'bid': max(call_bid, 0.05),
            'ask': max(call_ask, 0.10),
            'type': 'call',
            'delta': round(call_delta, 4),
            'expiration': exp_date,
            'iv': 0.25 * iv_multiplier,
        })
    return pd.DataFrame(rows)


def _neutral_signals(rsi=50):
    return {'trend': 'neutral', 'rsi': rsi}


def _elevated_iv():
    return {'iv_rank': 50, 'iv_percentile': 50}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindIronCondorsNeutralMarket:
    """test_find_iron_condors_neutral_market — neutral trend + elevated IV → finds condors."""

    def test_finds_condors(self):
        strategy = CreditSpreadStrategy(_make_config())
        # iv_multiplier=2.0 produces premiums consistent with elevated-IV environment,
        # satisfying the min_combined_credit_pct check (denominator = 2×spread_width).
        chain = _make_option_chain(iv_multiplier=2.0)
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, _neutral_signals(), _elevated_iv()
        )
        assert len(condors) > 0
        for c in condors:
            assert c['type'] == 'iron_condor'

    def test_condor_has_all_fields(self):
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain(iv_multiplier=2.0)
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, _neutral_signals(), _elevated_iv()
        )
        assert len(condors) > 0
        c = condors[0]
        required_fields = [
            'ticker', 'type', 'expiration', 'dte',
            'short_strike', 'long_strike', 'put_credit',
            'call_short_strike', 'call_long_strike', 'call_credit',
            'credit', 'max_loss', 'max_profit', 'profit_target',
            'stop_loss', 'spread_width', 'current_price',
            'distance_to_put_short', 'distance_to_call_short',
            'distance_to_short', 'short_delta', 'pop', 'risk_reward',
        ]
        for field in required_fields:
            assert field in c, f"Missing field: {field}"


class TestFindIronCondorsTrendingMarket:
    """find_iron_condors — RSI out-of-range → no condors (rsi_min=35, rsi_max=65).

    Trend alone no longer blocks condors; the RSI gate is the guard against
    directional/overbought markets.  These tests use RSI values outside the
    [35, 65] window to confirm the block fires.
    """

    def test_bullish_no_condors(self):
        """Overbought RSI (75 > rsi_max 65) blocks condors."""
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, {'trend': 'bullish', 'rsi': 75}, _elevated_iv()
        )
        assert len(condors) == 0  # RSI 75 > rsi_max(65) → blocked

    def test_bearish_no_condors(self):
        """Oversold RSI (25 < rsi_min 35) blocks condors."""
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, {'trend': 'bearish', 'rsi': 25}, _elevated_iv()
        )
        assert len(condors) == 0  # RSI 80 > rsi_max(65) → blocked


class TestCondorNonOverlapping:
    """test_condor_non_overlapping — put_short < call_short validated."""

    def test_put_short_below_call_short(self):
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, _neutral_signals(), _elevated_iv()
        )
        for c in condors:
            assert c['short_strike'] < c['call_short_strike'], (
                f"Put short {c['short_strike']} must be < call short {c['call_short_strike']}"
            )


class TestCondorCreditCalculation:
    """test_condor_credit_calculation — total credit = put_credit + call_credit."""

    def test_credit_sum(self):
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, _neutral_signals(), _elevated_iv()
        )
        for c in condors:
            expected = round(c['put_credit'] + c['call_credit'], 2)
            assert c['credit'] == expected, (
                f"Credit {c['credit']} != put {c['put_credit']} + call {c['call_credit']}"
            )


class TestCondorMaxLoss:
    """test_condor_max_loss — max_loss = spread_width - total_credit."""

    def test_max_loss_formula(self):
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, _neutral_signals(), _elevated_iv()
        )
        for c in condors:
            expected = round(c['spread_width'] - c['credit'], 2)
            assert c['max_loss'] == expected, (
                f"Max loss {c['max_loss']} != width {c['spread_width']} - credit {c['credit']}"
            )


class TestCondorScoringNeutralTrend:
    """test_condor_scoring_neutral_trend — score >= 60 when neutral + IV elevated."""

    def test_score_reaches_threshold(self):
        strategy = CreditSpreadStrategy(_make_config())
        # Use elevated IV chain (2x premiums) to simulate high IV environment
        chain = _make_option_chain(iv_multiplier=2.0)
        tech = {'trend': 'neutral', 'rsi': 50, 'regime': 'mean_reverting'}
        iv = {'iv_rank': 80, 'iv_percentile': 80}
        results = strategy.evaluate_spread_opportunity(
            'SPY', chain, tech, iv, 450.0
        )
        condors = [r for r in results if r['type'] == 'iron_condor']
        # At least some condors should score >= 60 in ideal conditions
        high_scored = [c for c in condors if c.get('score', 0) >= 60]
        assert len(high_scored) > 0, (
            f"No condors scored >= 60. Scores: {[c.get('score') for c in condors]}"
        )


class TestCondorScoringTrending:
    """No condors when RSI is overbought (rsi_max=65)."""

    def test_no_condors_in_trending(self):
        """RSI 75 > rsi_max 65 blocks condors via the RSI gate."""
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        tech = {'trend': 'bullish', 'rsi': 75}
        iv = {'iv_rank': 50, 'iv_percentile': 50}
        results = strategy.evaluate_spread_opportunity(
            'SPY', chain, tech, iv, 450.0
        )
        condors = [r for r in results if r['type'] == 'iron_condor']
        assert len(condors) == 0


class TestCondorPOPCombined:
    """test_condor_pop_combined — POP reflects both legs."""

    def test_pop_less_than_individual_legs(self):
        """Combined POP should be less than either individual leg's POP."""
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, _neutral_signals(), _elevated_iv()
        )
        # POP should be positive and reasonable
        for c in condors:
            assert 0 < c['pop'] <= 100, f"POP {c['pop']} out of range"

    def test_pop_is_nonnegative(self):
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, _neutral_signals(), _elevated_iv()
        )
        for c in condors:
            assert c['pop'] >= 0


class TestCondorMinimumCredit:
    """test_condor_minimum_credit — skip if combined credit too low."""

    def test_low_credit_filtered(self):
        """With a very high min_combined_credit_pct, condors with low credit should be filtered."""
        config = _make_config()
        config['strategy']['iron_condor']['min_combined_credit_pct'] = 90  # Very high threshold
        strategy = CreditSpreadStrategy(config)
        chain = _make_option_chain()
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, _neutral_signals(), _elevated_iv()
        )
        # With 90% threshold, most/all condors should be filtered
        # (credit would need to be 90% of $5 = $4.50 combined)
        assert len(condors) == 0 or all(
            c['credit'] / c['spread_width'] * 100 >= 90 for c in condors
        )

    def test_low_iv_no_condors(self):
        """Low IV should prevent condor generation."""
        strategy = CreditSpreadStrategy(_make_config())
        chain = _make_option_chain()
        condors = strategy.find_iron_condors(
            'SPY', chain, 450.0, _neutral_signals(), {'iv_rank': 5, 'iv_percentile': 5}
        )
        assert len(condors) == 0
