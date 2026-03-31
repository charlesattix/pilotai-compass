"""Tests for TelegramBot."""
from unittest.mock import AsyncMock, MagicMock

from alerts.telegram_bot import TelegramBot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(enabled=False, token='FAKE_TOKEN', chat_id='12345'):
    return {
        'alerts': {
            'telegram': {
                'enabled': enabled,
                'bot_token': token,
                'chat_id': chat_id,
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTelegramBot:

    def test_disabled_by_default(self):
        """Bot should be disabled when config says so."""
        bot = TelegramBot(_make_config(enabled=False))
        assert bot.enabled is False

    def test_send_alert_returns_false_when_disabled(self):
        """send_alert should return False when bot is disabled."""
        bot = TelegramBot(_make_config(enabled=False))
        assert bot.send_alert("test") is False

    def test_send_alerts_returns_zero_when_disabled(self):
        """send_alerts should return 0 when bot is disabled."""
        bot = TelegramBot(_make_config(enabled=False))
        assert bot.send_alerts([], MagicMock()) == 0

    def test_send_alert_calls_bot(self):
        """send_alert should call bot.send_message with timeout."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        bot = TelegramBot(_make_config(enabled=False))
        bot.enabled = True
        bot.bot = mock_bot
        bot.chat_id = '12345'

        bot.send_alert("Hello")
        mock_bot.send_message.assert_called_once()
        call_kwargs = mock_bot.send_message.call_args
        assert call_kwargs.kwargs.get('read_timeout') == 10
        assert call_kwargs.kwargs.get('write_timeout') == 10

    def test_send_alert_handles_exception(self):
        """send_alert should return False on exception."""
        mock_bot = MagicMock()
        mock_bot.send_message.side_effect = Exception("network error")

        bot = TelegramBot(_make_config(enabled=False))
        bot.enabled = True
        bot.bot = mock_bot
        bot.chat_id = '12345'

        result = bot.send_alert("Hello")
        assert result is False

    def test_unconfigured_token_disables_bot(self):
        """If token is placeholder, bot should be disabled."""
        bot = TelegramBot(_make_config(enabled=True, token='YOUR_BOT_TOKEN_HERE'))
        assert bot.enabled is False
