"""Tests for shared.telegram_alerts — all HTTP mocked, no real calls."""

import importlib
from unittest.mock import MagicMock, patch

# ── Helpers ────────────────────────────────────────────────────────────────


def _reload_module(token="tok123", chat_id="456"):
    """Reimport the module with controlled env vars so module-level reads pick them up."""
    env = {}
    if token is not None:
        env["TELEGRAM_BOT_TOKEN"] = token
    if chat_id is not None:
        env["TELEGRAM_CHAT_ID"] = chat_id

    with patch.dict("os.environ", env, clear=True):
        import shared.telegram_alerts as mod
        importlib.reload(mod)
        # Reset the warning flag after reload
        mod._warned_not_configured = False
    return mod


def _make_trade(**overrides):
    trade = {
        "ticker": "SPY",
        "type": "bull_put_spread",
        "short_strike": 420,
        "long_strike": 415,
        "contracts": 2,
        "total_credit": 150.0,
        "total_max_loss": 850.0,
        "dte_at_entry": 14,
    }
    trade.update(overrides)
    return trade


# ── TestIsConfigured ──────────────────────────────────────────────────────


class TestIsConfigured:
    def test_false_when_missing(self):
        mod = _reload_module(token=None, chat_id=None)
        assert mod.is_configured() is False

    def test_false_when_empty(self):
        mod = _reload_module(token="", chat_id="")
        assert mod.is_configured() is False

    def test_true_when_both_set(self):
        mod = _reload_module(token="abc", chat_id="123")
        assert mod.is_configured() is True


# ── TestSendMessage ───────────────────────────────────────────────────────


class TestSendMessage:
    @patch("shared.telegram_alerts.requests.post")
    def test_success(self, mock_post):
        mod = _reload_module()
        mock_post.return_value = MagicMock(status_code=200)
        assert mod.send_message("hello") is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "hello" in str(call_kwargs)

    @patch("shared.telegram_alerts.requests.post")
    def test_http_error_returns_false(self, mock_post):
        mod = _reload_module()
        mock_post.return_value.raise_for_status.side_effect = Exception("500")
        assert mod.send_message("fail") is False

    @patch("shared.telegram_alerts.requests.post")
    def test_network_exception_returns_false(self, mock_post):
        mod = _reload_module()
        mock_post.side_effect = ConnectionError("timeout")
        assert mod.send_message("fail") is False

    def test_not_configured_returns_false_and_warns(self, caplog):
        mod = _reload_module(token=None, chat_id=None)
        import logging
        with caplog.at_level(logging.WARNING):
            result = mod.send_message("test")
        assert result is False
        assert "not configured" in caplog.text

    def test_warning_only_once(self, caplog):
        mod = _reload_module(token=None, chat_id=None)
        import logging
        with caplog.at_level(logging.WARNING):
            mod.send_message("a")
            caplog.clear()
            mod.send_message("b")
        assert "not configured" not in caplog.text


# ── TestNotifyTradeOpen ───────────────────────────────────────────────────


class TestNotifyTradeOpen:
    @patch("shared.telegram_alerts.requests.post")
    def test_formats_and_sends(self, mock_post):
        mod = _reload_module()
        mock_post.return_value = MagicMock(status_code=200)
        trade = _make_trade()
        assert mod.notify_trade_open(trade) is True

        sent_text = mock_post.call_args[1]["json"]["text"]
        assert "SPY" in sent_text
        assert "420" in sent_text
        assert "415" in sent_text
        assert "150.00" in sent_text
        assert "DTE: 14" in sent_text

    def test_graceful_when_not_configured(self):
        mod = _reload_module(token=None, chat_id=None)
        assert mod.notify_trade_open(_make_trade()) is False


# ── TestNotifyTradeClose ──────────────────────────────────────────────────


class TestNotifyTradeClose:
    @patch("shared.telegram_alerts.requests.post")
    def test_profit_uses_green_emoji(self, mock_post):
        mod = _reload_module()
        mock_post.return_value = MagicMock(status_code=200)
        trade = _make_trade()
        assert mod.notify_trade_close(trade, pnl=75.0, reason="profit_target", balance=100_075) is True

        sent_text = mock_post.call_args[1]["json"]["text"]
        assert "\U0001f4c8" in sent_text  # green chart emoji
        assert "+$75.00" in sent_text
        assert "Profit Target Hit" in sent_text
        assert "$100,075.00" in sent_text

    @patch("shared.telegram_alerts.requests.post")
    def test_loss_uses_red_emoji(self, mock_post):
        mod = _reload_module()
        mock_post.return_value = MagicMock(status_code=200)
        trade = _make_trade()
        assert mod.notify_trade_close(trade, pnl=-200.0, reason="stop_loss", balance=99_800) is True

        sent_text = mock_post.call_args[1]["json"]["text"]
        assert "\U0001f4c9" in sent_text  # red chart emoji
        assert "-$200.00" in sent_text
        assert "Stop Loss Hit" in sent_text


# ── TestNotifyDailySummary ────────────────────────────────────────────────


class TestNotifyDailySummary:
    @patch("shared.telegram_alerts.requests.post")
    @patch("scripts.daily_report.get_daily_summary_metrics")
    def test_calls_metrics_and_sends(self, mock_metrics, mock_post):
        mod = _reload_module()
        mock_post.return_value = MagicMock(status_code=200)
        mock_metrics.return_value = {
            "date": "2026-03-07",
            "alerts_fired": 3,
            "closed_today": 2,
            "wins": 1,
            "losses": 1,
            "day_pnl": 50.0,
            "day_pnl_pct": 0.05,
            "open_positions": 4,
            "total_risk_pct": 3.2,
            "account_balance": 100_050.0,
            "pct_from_start": 0.05,
            "best": "SPY +$75.00",
            "worst": "QQQ -$25.00",
        }
        assert mod.notify_daily_summary("2026-03-07") is True
        mock_metrics.assert_called_once()
        mock_post.assert_called_once()

    @patch("scripts.daily_report.get_daily_summary_metrics")
    def test_handles_metrics_exception(self, mock_metrics):
        mod = _reload_module()
        mock_metrics.side_effect = RuntimeError("DB error")
        assert mod.notify_daily_summary() is False


# ── TestNotifyDeviationAlerts ─────────────────────────────────────────────


class TestNotifyDeviationAlerts:
    @patch("shared.telegram_alerts.requests.post")
    def test_sends_when_alerts_exist(self, mock_post):
        mod = _reload_module()
        mock_post.return_value = MagicMock(status_code=200)

        snapshot = {
            "details": {
                "comparisons": [
                    {"metric": "Win Rate", "status": "WARN", "live_str": "55%", "backtest_str": "65%"},
                    {"metric": "PnL", "status": "PASS", "live_str": "$500", "backtest_str": "$480"},
                ]
            }
        }
        assert mod.notify_deviation_alerts(snapshot) is True
        sent_text = mock_post.call_args[1]["json"]["text"]
        assert "DEVIATION ALERTS" in sent_text
        assert "Win Rate" in sent_text

    @patch("shared.telegram_alerts.requests.post")
    def test_returns_false_when_all_pass(self, mock_post):
        mod = _reload_module()
        snapshot = {
            "details": {
                "comparisons": [
                    {"metric": "Win Rate", "status": "PASS"},
                ]
            }
        }
        assert mod.notify_deviation_alerts(snapshot) is False
        mock_post.assert_not_called()
