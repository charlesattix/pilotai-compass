"""
Unified Telegram alert system for paper trading.

Message types:
  1. Trade execution  (entry/exit with P&L)
  2. Risk alerts      (DD threshold, VIX spike, position limit)
  3. Model alerts     (feature drift, AUC decay, retrain trigger)
  4. Daily summary    (P&L, win rate, open positions, hedge state)
  5. Weekly report    (vs backtest expectations, strategy health)

Priority levels: INFO / WARNING / CRITICAL
Rate-limiting prevents spam (configurable per-priority cooldowns).

Uses shared.telegram_alerts for delivery when available; falls back
to logging when Telegram credentials are not configured.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


PRIORITY_EMOJI = {
    Priority.INFO: "ℹ️",
    Priority.WARNING: "⚠️",
    Priority.CRITICAL: "🚨",
}

# Default cooldowns in seconds per priority
DEFAULT_COOLDOWNS = {
    Priority.INFO: 300,       # 5 min
    Priority.WARNING: 60,     # 1 min
    Priority.CRITICAL: 0,     # no limit
}


@dataclass
class AlertMessage:
    """A single alert to be sent."""
    priority: Priority
    category: str             # trade / risk / model / daily / weekly
    title: str
    body: str
    timestamp: Optional[datetime] = None
    delivered: bool = False
    rate_limited: bool = False


@dataclass
class TradeAlert:
    """Trade execution details."""
    action: str               # "entry" | "exit"
    strategy: str
    symbol: str
    direction: str            # "long" | "short"
    contracts: int
    price: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class RiskAlert:
    """Risk threshold alert."""
    metric: str               # "drawdown" | "vix_spike" | "position_limit" | "daily_loss"
    current_value: float
    threshold: float
    message: str


@dataclass
class ModelAlert:
    """Model health alert."""
    metric: str               # "feature_drift" | "auc_decay" | "retrain_triggered"
    current_value: float
    baseline_value: float
    message: str


@dataclass
class DailySummary:
    """End-of-day summary."""
    date: str
    daily_pnl: float
    total_pnl: float
    equity: float
    win_rate: float
    n_trades_today: int
    n_open_positions: int
    hedge_state: str          # "normal" | "reducing" | "emergency"
    drawdown: float
    vix: float = 0.0


@dataclass
class WeeklyReport:
    """Weekly performance report."""
    week_ending: str
    weekly_pnl: float
    weekly_return: float
    ytd_return: float
    sharpe_rolling: float
    max_dd: float
    win_rate: float
    n_trades: int
    backtest_expected_pnl: float
    drift_pct: float          # actual vs backtest divergence
    strategy_health: str      # "healthy" | "degrading" | "critical"


# ---------------------------------------------------------------------------
# Core alerter
# ---------------------------------------------------------------------------

class TelegramAlerter:
    """Unified Telegram alert system.

    Args:
        experiment_id: Prefix for all messages (e.g. "EXP-880").
        cooldowns: Per-priority cooldown overrides.
        send_fn: Injectable send function (for testing). Signature: (text: str) -> bool.
        enabled: Master switch — if False, all messages are logged only.
    """

    def __init__(
        self,
        experiment_id: str = "EXP-880",
        cooldowns: Optional[Dict[Priority, int]] = None,
        send_fn: Optional[Callable[[str], bool]] = None,
        enabled: bool = True,
    ) -> None:
        self.experiment_id = experiment_id
        self.cooldowns = cooldowns or dict(DEFAULT_COOLDOWNS)
        self.enabled = enabled
        self._send_fn = send_fn or self._default_send
        self._last_sent: Dict[str, float] = {}  # category:priority → timestamp
        self._history: List[AlertMessage] = []
        self._message_count = 0
        self._rate_limited_count = 0

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    @staticmethod
    def _default_send(text: str) -> bool:
        """Try shared.telegram_alerts, fall back to logging."""
        try:
            from shared.telegram_alerts import send_message
            return send_message(text)
        except (ImportError, Exception):
            logger.info("TELEGRAM (not configured): %s", text[:200])
            return False

    def _check_rate_limit(self, category: str, priority: Priority) -> bool:
        """Return True if message should be sent (not rate-limited)."""
        cooldown = self.cooldowns.get(priority, 300)
        if cooldown <= 0:
            return True
        key = f"{category}:{priority.value}"
        now = time.time()
        last = self._last_sent.get(key, 0)
        if now - last < cooldown:
            return False
        self._last_sent[key] = now
        return True

    def send(self, msg: AlertMessage) -> bool:
        """Send an alert message through the pipeline."""
        msg.timestamp = msg.timestamp or datetime.now()

        if not self._check_rate_limit(msg.category, msg.priority):
            msg.rate_limited = True
            self._rate_limited_count += 1
            self._history.append(msg)
            return False

        text = self._format(msg)

        if self.enabled:
            success = self._send_fn(text)
            msg.delivered = success
        else:
            logger.info("ALERT [disabled]: %s", text[:200])
            msg.delivered = False

        self._message_count += 1
        self._history.append(msg)
        return msg.delivered

    def _format(self, msg: AlertMessage) -> str:
        """Format alert as Telegram-friendly text."""
        emoji = PRIORITY_EMOJI.get(msg.priority, "")
        ts = msg.timestamp.strftime("%H:%M") if msg.timestamp else ""
        return f"{emoji} [{self.experiment_id}] {msg.title}\n{msg.body}\n{ts}"

    # ------------------------------------------------------------------
    # 1. Trade execution alerts
    # ------------------------------------------------------------------

    def trade_alert(self, trade: TradeAlert) -> bool:
        """Send trade entry/exit alert."""
        if trade.action == "entry":
            title = f"📈 ENTRY: {trade.direction.upper()} {trade.symbol}"
            body = (f"Strategy: {trade.strategy}\n"
                    f"Contracts: {trade.contracts} @ ${trade.price:.2f}")
            priority = Priority.INFO
        else:
            pnl_emoji = "✅" if trade.pnl >= 0 else "❌"
            title = f"{pnl_emoji} EXIT: {trade.symbol} — ${trade.pnl:+,.0f} ({trade.pnl_pct:+.1%})"
            body = (f"Strategy: {trade.strategy}\n"
                    f"Reason: {trade.exit_reason}\n"
                    f"Contracts: {trade.contracts} @ ${trade.price:.2f}")
            priority = Priority.INFO if trade.pnl >= 0 else Priority.WARNING

        return self.send(AlertMessage(priority, "trade", title, body))

    # ------------------------------------------------------------------
    # 2. Risk alerts
    # ------------------------------------------------------------------

    def risk_alert(self, alert: RiskAlert) -> bool:
        """Send risk threshold alert."""
        if alert.metric == "drawdown":
            if alert.current_value >= alert.threshold:
                priority = Priority.CRITICAL
                title = f"🚨 DRAWDOWN CRITICAL: {alert.current_value:.1%}"
            else:
                priority = Priority.WARNING
                title = f"⚠️ Drawdown Warning: {alert.current_value:.1%}"
        elif alert.metric == "vix_spike":
            priority = Priority.CRITICAL if alert.current_value > 40 else Priority.WARNING
            title = f"📊 VIX Spike: {alert.current_value:.1f}"
        elif alert.metric == "daily_loss":
            priority = Priority.CRITICAL
            title = f"🚨 Daily Loss: {alert.current_value:.1%}"
        else:
            priority = Priority.WARNING
            title = f"⚠️ {alert.metric}: {alert.current_value:.2f}"

        body = f"{alert.message}\nThreshold: {alert.threshold:.2f}"
        return self.send(AlertMessage(priority, "risk", title, body))

    # ------------------------------------------------------------------
    # 3. Model alerts
    # ------------------------------------------------------------------

    def model_alert(self, alert: ModelAlert) -> bool:
        """Send model health alert."""
        if alert.metric == "retrain_triggered":
            priority = Priority.WARNING
            title = "🔄 Model Retrain Triggered"
        elif alert.metric == "auc_decay":
            priority = Priority.WARNING if alert.current_value > 0.60 else Priority.CRITICAL
            title = f"📉 AUC Decay: {alert.current_value:.3f} (baseline {alert.baseline_value:.3f})"
        elif alert.metric == "feature_drift":
            priority = Priority.WARNING
            title = f"📊 Feature Drift Detected: {alert.current_value:.3f}"
        else:
            priority = Priority.INFO
            title = f"🔬 Model: {alert.metric}"

        body = (f"{alert.message}\n"
                f"Current: {alert.current_value:.4f} | Baseline: {alert.baseline_value:.4f}")
        return self.send(AlertMessage(priority, "model", title, body))

    # ------------------------------------------------------------------
    # 4. Daily summary
    # ------------------------------------------------------------------

    def daily_summary(self, summary: DailySummary) -> bool:
        """Send end-of-day summary."""
        pnl_emoji = "🟢" if summary.daily_pnl >= 0 else "🔴"
        hedge_emoji = {"normal": "🛡️", "reducing": "⚠️", "emergency": "🚨"}.get(
            summary.hedge_state, "")

        title = f"📊 Daily Summary — {summary.date}"
        body = (
            f"{pnl_emoji} Daily P&L: ${summary.daily_pnl:+,.0f}\n"
            f"Total P&L: ${summary.total_pnl:+,.0f}\n"
            f"Equity: ${summary.equity:,.0f}\n"
            f"Win Rate: {summary.win_rate:.0%} | Trades Today: {summary.n_trades_today}\n"
            f"Open Positions: {summary.n_open_positions}\n"
            f"Drawdown: {summary.drawdown:.1%}\n"
            f"{hedge_emoji} Hedge: {summary.hedge_state}"
        )
        if summary.vix > 0:
            body += f"\nVIX: {summary.vix:.1f}"

        return self.send(AlertMessage(Priority.INFO, "daily", title, body))

    # ------------------------------------------------------------------
    # 5. Weekly report
    # ------------------------------------------------------------------

    def weekly_report(self, report: WeeklyReport) -> bool:
        """Send weekly performance report."""
        health_emoji = {"healthy": "✅", "degrading": "⚠️", "critical": "🚨"}.get(
            report.strategy_health, "❓")
        drift_emoji = "✅" if abs(report.drift_pct) < 0.30 else "⚠️"

        title = f"📋 Weekly Report — w/e {report.week_ending}"
        body = (
            f"Weekly P&L: ${report.weekly_pnl:+,.0f} ({report.weekly_return:+.1%})\n"
            f"YTD Return: {report.ytd_return:+.1%}\n"
            f"Rolling Sharpe: {report.sharpe_rolling:.2f}\n"
            f"Max DD: {report.max_dd:.1%}\n"
            f"Win Rate: {report.win_rate:.0%} | Trades: {report.n_trades}\n"
            f"\n{drift_emoji} vs Backtest: {report.drift_pct:+.0%} drift\n"
            f"Expected: ${report.backtest_expected_pnl:+,.0f}\n"
            f"{health_emoji} Strategy Health: {report.strategy_health.upper()}"
        )

        priority = Priority.INFO if report.strategy_health == "healthy" else Priority.WARNING
        return self.send(AlertMessage(priority, "weekly", title, body))

    # ------------------------------------------------------------------
    # Kill switch alert (always CRITICAL, never rate-limited)
    # ------------------------------------------------------------------

    def kill_switch_alert(self, reason: str, drawdown: float) -> bool:
        """Emergency kill switch notification."""
        title = "🚨🚨 KILL SWITCH TRIGGERED 🚨🚨"
        body = (
            f"Reason: {reason}\n"
            f"Drawdown: {drawdown:.1%}\n"
            f"ALL POSITIONS LIQUIDATED\n"
            f"Manual review required before restart"
        )
        return self.send(AlertMessage(Priority.CRITICAL, "kill_switch", title, body))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[AlertMessage]:
        return list(self._history)

    @property
    def message_count(self) -> int:
        return self._message_count

    @property
    def rate_limited_count(self) -> int:
        return self._rate_limited_count

    def reset_rate_limits(self) -> None:
        """Clear rate limit state (e.g. at start of new trading day)."""
        self._last_sent.clear()
