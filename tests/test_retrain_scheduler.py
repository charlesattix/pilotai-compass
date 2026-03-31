"""
Tests for compass/retrain_scheduler.py

Coverage:
  - RetrainScheduler delegates to ModelRetrainer.check_and_retrain
  - Telegram alert sent when trigger fires and retrain occurs
  - Telegram alert sent when trigger fires but no promotion (triggered only)
  - No alert when trigger.triggered is False
  - No alert when telegram_bot is None (even if triggered)
  - Graceful failure when Telegram send_message raises
  - feature_pipeline forwarded to ModelRetrainer
  - model_class forwarded to ModelRetrainer
  - Alert message contains expected fields (AUC, filename, timestamp)
  - Alert omits AUC lines when ab_result is None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from compass.retrain_scheduler import RetrainScheduler
from compass.online_retrain import ABResult, RetrainResult, RetrainTrigger


# ── Helpers ────────────────────────────────────────────────────────────────────

def _trigger(triggered: bool = True, reasons: Optional[List[str]] = None) -> RetrainTrigger:
    return RetrainTrigger(triggered=triggered, reasons=reasons or ["model_age"])


def _result(
    triggered: bool = True,
    retrained: bool = True,
    reasons: Optional[List[str]] = None,
    old_auc: float = 0.72,
    new_auc: float = 0.78,
    new_model_path: Optional[str] = "data/models/signal_model_20260328T163000.joblib",
    ab_result: Optional[ABResult] = None,
) -> RetrainResult:
    ab = ab_result
    if ab is None and retrained:
        ab = ABResult(
            old_auc=old_auc, new_auc=new_auc,
            old_accuracy=0.80, new_accuracy=0.83,
            holdout_size=50, promoted=True, reason="new model better",
        )
    return RetrainResult(
        trigger=_trigger(triggered=triggered, reasons=reasons),
        retrained=retrained,
        ab_result=ab,
        new_model_path=new_model_path if retrained else None,
    )


class _FakeBot:
    """Simple Telegram stub that records sent messages."""
    def __init__(self):
        self.messages: List[str] = []

    def send_message(self, text: str) -> None:
        self.messages.append(text)


class _FailBot:
    """Telegram stub whose send_message always raises."""
    def send_message(self, text: str) -> None:
        raise RuntimeError("network error")


def _make_scheduler(bot=None, model_class=None, feature_pipeline=None):
    """Build a RetrainScheduler with a mocked ModelRetrainer."""
    from compass.signal_model import SignalModel
    mc = model_class or SignalModel
    sched = RetrainScheduler(
        model_dir="/tmp/test_models",
        model_class=mc,
        telegram_bot=bot,
        feature_pipeline=feature_pipeline,
    )
    return sched


# ── Delegation to ModelRetrainer ───────────────────────────────────────────────

class TestDelegation:

    def test_run_retrain_check_calls_check_and_retrain(self):
        sched = _make_scheduler()
        expected = _result()
        with patch.object(sched.retrainer, "check_and_retrain", return_value=expected) as mock:
            df = pd.DataFrame({"a": [1, 2]})
            labels = [1, 0]
            result = sched.run_retrain_check(df, labels)
        mock.assert_called_once_with(df, labels, current_model=None)
        assert result is expected

    def test_run_retrain_check_passes_current_model(self):
        sched = _make_scheduler()
        expected = _result()
        current_model = MagicMock()
        with patch.object(sched.retrainer, "check_and_retrain", return_value=expected) as mock:
            sched.run_retrain_check(pd.DataFrame(), [], current_model=current_model)
        assert mock.call_count == 1
        _, kwargs = mock.call_args
        assert kwargs.get("current_model") is current_model

    def test_returns_retrain_result(self):
        sched = _make_scheduler()
        expected = _result()
        with patch.object(sched.retrainer, "check_and_retrain", return_value=expected):
            result = sched.run_retrain_check(pd.DataFrame(), [])
        assert isinstance(result, RetrainResult)

    def test_model_class_forwarded_to_retrainer(self):
        from compass.signal_model import SignalModel
        sched = _make_scheduler(model_class=SignalModel)
        assert sched.retrainer.model_class is SignalModel

    def test_feature_pipeline_forwarded_to_retrainer(self):
        pipeline = MagicMock()
        sched = _make_scheduler(feature_pipeline=pipeline)
        assert sched.retrainer.feature_pipeline is pipeline


# ── Telegram alerting ─────────────────────────────────────────────────────────

class TestTelegramAlert:

    def test_alert_sent_when_triggered_and_retrained(self):
        bot = _FakeBot()
        sched = _make_scheduler(bot=bot)
        with patch.object(sched.retrainer, "check_and_retrain", return_value=_result()):
            sched.run_retrain_check(pd.DataFrame(), [])
        assert len(bot.messages) == 1

    def test_alert_sent_when_triggered_but_not_retrained(self):
        """Trigger fires but no promotion — alert should still fire."""
        bot = _FakeBot()
        sched = _make_scheduler(bot=bot)
        result = _result(triggered=True, retrained=False)
        result.ab_result = None
        with patch.object(sched.retrainer, "check_and_retrain", return_value=result):
            sched.run_retrain_check(pd.DataFrame(), [])
        assert len(bot.messages) == 1

    def test_no_alert_when_not_triggered(self):
        bot = _FakeBot()
        sched = _make_scheduler(bot=bot)
        with patch.object(sched.retrainer, "check_and_retrain",
                          return_value=_result(triggered=False, retrained=False)):
            sched.run_retrain_check(pd.DataFrame(), [])
        assert len(bot.messages) == 0

    def test_no_alert_when_bot_is_none(self):
        """No exception raised, no messages, when bot is None."""
        sched = _make_scheduler(bot=None)
        with patch.object(sched.retrainer, "check_and_retrain", return_value=_result()):
            sched.run_retrain_check(pd.DataFrame(), [])  # should not raise

    def test_graceful_failure_when_bot_raises(self):
        """Telegram send failure must not propagate — result still returned."""
        sched = _make_scheduler(bot=_FailBot())
        expected = _result()
        with patch.object(sched.retrainer, "check_and_retrain", return_value=expected):
            result = sched.run_retrain_check(pd.DataFrame(), [])
        assert result is expected  # no exception raised

    def test_graceful_failure_logs_warning(self, caplog):
        import logging
        sched = _make_scheduler(bot=_FailBot())
        with patch.object(sched.retrainer, "check_and_retrain", return_value=_result()):
            with caplog.at_level(logging.WARNING, logger="compass.retrain_scheduler"):
                sched.run_retrain_check(pd.DataFrame(), [])
        assert any("Telegram alert failed" in r.message for r in caplog.records)


# ── Alert message content ─────────────────────────────────────────────────────

class TestAlertMessageContent:

    def _get_message(self, result: RetrainResult) -> str:
        bot = _FakeBot()
        sched = _make_scheduler(bot=bot)
        with patch.object(sched.retrainer, "check_and_retrain", return_value=result):
            sched.run_retrain_check(pd.DataFrame(), [])
        assert bot.messages, "Expected an alert to be sent"
        return bot.messages[0]

    def test_message_contains_retrained_status(self):
        msg = self._get_message(_result(retrained=True))
        assert "RETRAINED" in msg

    def test_message_contains_triggered_no_promotion_status(self):
        r = _result(triggered=True, retrained=False)
        r.ab_result = None
        msg = self._get_message(r)
        assert "TRIGGERED" in msg or "promotion" in msg.lower()

    def test_message_contains_reasons(self):
        msg = self._get_message(_result(reasons=["model_age", "drift"]))
        assert "model_age" in msg
        assert "drift" in msg

    def test_message_contains_auc_values(self):
        msg = self._get_message(_result(old_auc=0.72, new_auc=0.78))
        assert "0.7200" in msg
        assert "0.7800" in msg

    def test_message_contains_model_filename(self):
        msg = self._get_message(_result(
            new_model_path="data/models/ensemble_model_20260328T163000.joblib"
        ))
        assert "ensemble_model_20260328T163000.joblib" in msg

    def test_message_contains_timestamp(self):
        msg = self._get_message(_result())
        # Timestamp format: "2026-03-28 16:30 UTC" (date portion is enough)
        assert "UTC" in msg

    def test_message_omits_auc_when_no_ab_result(self):
        r = _result(triggered=True, retrained=False)
        r.ab_result = None
        msg = self._get_message(r)
        assert "AUC" not in msg

    def test_message_format_has_newlines(self):
        """Message should be multi-line for readability."""
        msg = self._get_message(_result())
        assert "\n" in msg
