"""
Retrain Scheduler — wires ModelRetrainer into the daily scan scheduler.

Triggered by SLOT_RETRAIN (16:30 ET) via ScanScheduler.  Sends a Telegram
alert whenever a retrain check fires (whether or not a new model is promoted).

Usage::

    from compass.retrain_scheduler import RetrainScheduler
    from compass.ensemble_signal_model import EnsembleSignalModel
    from alerts.telegram_bot import TelegramBot

    scheduler = RetrainScheduler(
        model_dir="data/models",
        model_class=EnsembleSignalModel,
        telegram_bot=bot,          # optional — omit to disable alerts
        feature_pipeline=pipeline, # optional — passed through to ModelRetrainer
    )

    # In the main scan_fn:
    if slot_type == SLOT_RETRAIN:
        result = scheduler.run_retrain_check(trades_df, labels, current_model)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from compass.online_retrain import ModelRetrainer, RetrainResult

logger = logging.getLogger(__name__)


class RetrainScheduler:
    """Integrates :class:`~compass.online_retrain.ModelRetrainer` with the
    scheduler and optional Telegram alerting.

    Args:
        model_dir: Directory for versioned model files (passed to ModelRetrainer).
        model_class: Model class — ``SignalModel`` or ``EnsembleSignalModel``.
        telegram_bot: Optional bot with a ``send_message(text)`` method.
                      If ``None``, alerting is disabled.
        feature_pipeline: Optional feature pipeline passed through to
                          ModelRetrainer (GAP-5 support).
    """

    def __init__(
        self,
        model_dir: str,
        model_class: Type,
        telegram_bot: Optional[Any] = None,
        feature_pipeline: Optional[Any] = None,
    ) -> None:
        self.retrainer = ModelRetrainer(
            model_dir=model_dir,
            model_class=model_class,
            feature_pipeline=feature_pipeline,
        )
        self.telegram_bot = telegram_bot

    def run_retrain_check(self, trades_df, labels, current_model=None) -> RetrainResult:
        """Run a retrain check and send a Telegram alert if the trigger fires.

        Args:
            trades_df: DataFrame of recent trades / features.
            labels: Win/loss labels aligned with ``trades_df``.
            current_model: Currently deployed model instance, or ``None``.

        Returns:
            :class:`~compass.online_retrain.RetrainResult` from
            ``ModelRetrainer.check_and_retrain``.
        """
        result = self.retrainer.check_and_retrain(
            trades_df, labels, current_model=current_model
        )
        self._send_alert(result)
        return result

    def _send_alert(self, result: RetrainResult) -> None:
        """Send a Telegram alert when the retrain trigger fires.

        Only sends when:
        - ``telegram_bot`` is set, **and**
        - ``result.trigger.triggered`` is True

        Logs a warning (does not raise) on Telegram failure.
        """
        if self.telegram_bot is None:
            return
        if not result.trigger.triggered:
            return

        ab = result.ab_result
        old_auc = ab.old_auc if ab is not None else None
        new_auc = ab.new_auc if ab is not None else None

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        status = "RETRAINED" if result.retrained else "TRIGGERED (no promotion)"
        reasons = "; ".join(result.trigger.reasons) if result.trigger.reasons else "unspecified"

        lines = [
            f"[ModelRetrainer] {status}",
            f"Reasons: {reasons}",
        ]
        if old_auc is not None and new_auc is not None:
            lines.append(f"AUC: {old_auc:.4f} → {new_auc:.4f}")
        if result.new_model_path:
            lines.append(f"Model: {result.new_model_path}")
        lines.append(f"At: {timestamp}")

        message = "\n".join(lines)
        try:
            self.telegram_bot.send_message(message)
        except Exception as exc:
            logger.warning("Telegram alert failed: %s", exc)
