"""Tests for AlertGenerator."""
from datetime import datetime, timedelta

from alerts.alert_generator import AlertGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_dir=None):
    return {
        'alerts': {
            'min_score': 60,
            'output_json': True,
            'output_text': True,
            'output_csv': True,
            'json_file': 'alerts.json',
            'text_file': 'alerts.txt',
            'csv_file': 'alerts.csv',
            'telegram': {'enabled': False, 'bot_token': '', 'chat_id': ''},
        },
    }


def _make_opportunity(score=75.0, opp_type='bull_put_spread'):
    return {
        'ticker': 'SPY',
        'type': opp_type,
        'expiration': str(datetime.now() + timedelta(days=35)),
        'dte': 35,
        'short_strike': 440.0,
        'long_strike': 435.0,
        'short_delta': 0.12,
        'credit': 1.75,
        'max_loss': 3.25,
        'max_profit': 1.75,
        'profit_target': 0.88,
        'stop_loss': 4.38,
        'spread_width': 5,
        'current_price': 450.0,
        'distance_to_short': -10.0,
        'pop': 88.0,
        'risk_reward': 0.54,
        'score': score,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAlertGenerator:

    def test_generate_alerts_empty(self):
        """generate_alerts with empty list should return empty dict."""
        gen = AlertGenerator(_make_config())
        assert gen.generate_alerts([]) == {}

    def test_generate_alerts_filters_low_score(self):
        """Opportunities below min_alert_score (default 28) should produce no alerts."""
        gen = AlertGenerator(_make_config())
        opps = [_make_opportunity(score=20)]
        assert gen.generate_alerts(opps) == {}

    def test_generate_alerts_produces_outputs(self):
        """Alerts should produce text and csv outputs (JSON is in SQLite only)."""
        gen = AlertGenerator(_make_config())
        opps = [_make_opportunity(score=75)]
        outputs = gen.generate_alerts(opps)
        assert 'text' in outputs
        assert 'csv' in outputs

    def test_format_telegram_message_bull_put(self):
        """format_telegram_message should return a string for bull put spread."""
        gen = AlertGenerator(_make_config())
        opp = _make_opportunity(opp_type='bull_put_spread')
        msg = gen.format_telegram_message(opp)
        assert isinstance(msg, str)
        assert 'SPY' in msg
        assert 'Put' in msg

    def test_format_telegram_message_bear_call(self):
        """format_telegram_message should return a string for bear call spread."""
        gen = AlertGenerator(_make_config())
        opp = _make_opportunity(opp_type='bear_call_spread')
        msg = gen.format_telegram_message(opp)
        assert isinstance(msg, str)
        assert 'Call' in msg
