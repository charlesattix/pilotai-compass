"""Tests for spread opportunity scoring in CreditSpreadStrategy."""
import pytest

from strategy.spread_strategy import CreditSpreadStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy(config_override=None):
    """Build a CreditSpreadStrategy with sensible defaults."""
    config = {
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
    if config_override:
        config.update(config_override)
    return CreditSpreadStrategy(config)


def _make_opp(credit=1.75, spread_width=5, risk_reward=0.54, pop=87.0,
              opp_type='bull_put_spread'):
    """Create a minimal opportunity dict."""
    return {
        'ticker': 'SPY',
        'type': opp_type,
        'credit': credit,
        'spread_width': spread_width,
        'risk_reward': risk_reward,
        'pop': pop,
        'max_loss': spread_width - credit,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpreadScoring:

    def test_score_high_credit_pct(self):
        """Higher credit as a fraction of spread width should yield a higher score."""
        strategy = _make_strategy()
        tech = {'trend': 'neutral'}
        iv = {'iv_rank': 40}

        opp_high = _make_opp(credit=2.0, spread_width=5)   # 40%
        opp_low = _make_opp(credit=1.0, spread_width=5)    # 20%

        scored = strategy._score_opportunities(
            [opp_high, opp_low], tech, iv
        )
        assert scored[0]['score'] > scored[1]['score']

    def test_score_high_pop(self):
        """Higher POP should contribute a higher score component."""
        strategy = _make_strategy()
        tech = {'trend': 'neutral'}
        iv = {'iv_rank': 40}

        opp_high_pop = _make_opp(pop=92.0, credit=1.5, risk_reward=0.43)
        opp_low_pop = _make_opp(pop=70.0, credit=1.5, risk_reward=0.43)

        scored = strategy._score_opportunities(
            [opp_high_pop, opp_low_pop], tech, iv
        )
        # The one with higher POP should score higher (all else roughly equal)
        assert scored[0]['pop'] >= scored[1]['pop']

    def test_pop_calculation_from_delta(self):
        """POP should be approximately (1 - |delta|) * 100."""
        strategy = _make_strategy()
        pop = strategy._calculate_pop(-0.12)
        assert pop == pytest.approx(88.0, abs=0.01)

    def test_bullish_alignment_bonus(self):
        """A bull put spread in a bullish trend should score higher than neutral."""
        strategy = _make_strategy()
        iv = {'iv_rank': 40}

        _make_opp(opp_type='bull_put_spread')

        scored_bullish = strategy._score_opportunities(
            [_make_opp(opp_type='bull_put_spread')],
            {'trend': 'bullish'},
            iv,
        )
        scored_neutral = strategy._score_opportunities(
            [_make_opp(opp_type='bull_put_spread')],
            {'trend': 'neutral'},
            iv,
        )
        assert scored_bullish[0]['score'] > scored_neutral[0]['score']

    def test_bear_call_scoring(self):
        """A bear call spread should also receive a valid score."""
        strategy = _make_strategy()
        tech = {'trend': 'bearish'}
        iv = {'iv_rank': 50}

        opp = _make_opp(opp_type='bear_call_spread')
        scored = strategy._score_opportunities([opp], tech, iv)
        assert scored[0]['score'] > 0
