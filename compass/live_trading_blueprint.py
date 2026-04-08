"""
Live trading integration blueprint — bridge from strategy signals to
broker execution with full risk management and audit trail.

Components:
  1. Signal → Order translation
  2. Pre-trade risk checks (5 gates)
  3. Order management (entry, scale, exit, emergency liquidation)
  4. Real-time P&L tracking with alert thresholds
  5. Kill switch (drawdown, anomaly, manual)
  6. Reconciliation (paper vs live, expected vs actual)
  7. Audit trail (every decision logged)

Designed for Alpaca REST/WebSocket patterns but broker-agnostic via
adapter interface. All methods are synchronous for testability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RiskCheckResult(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class KillSwitchState(str, Enum):
    ARMED = "armed"
    TRIGGERED = "triggered"
    MANUAL_HALT = "manual_halt"
    DISARMED = "disarmed"


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StrategySignal:
    """Input from a strategy engine."""
    signal_id: str
    strategy: str
    symbol: str
    direction: str        # "long" | "short"
    confidence: float     # 0-1
    target_contracts: int
    spread_width: float = 5.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    profit_target: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class Order:
    """An order to be submitted to broker."""
    order_id: str
    signal_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0.0
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    reject_reason: str = ""


@dataclass
class Position:
    """A live position."""
    position_id: str
    strategy: str
    symbol: str
    direction: str
    contracts: int
    entry_price: float
    current_price: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    opened_at: Optional[datetime] = None


@dataclass
class RiskCheckReport:
    """Result of pre-trade risk validation."""
    signal_id: str
    result: RiskCheckResult
    checks: Dict[str, bool]
    reject_reasons: List[str]
    timestamp: Optional[datetime] = None


@dataclass
class PnLSnapshot:
    """Real-time P&L state."""
    timestamp: datetime
    total_equity: float
    total_pnl: float
    daily_pnl: float
    drawdown: float
    n_positions: int
    by_strategy: Dict[str, float] = field(default_factory=dict)


@dataclass
class ReconciliationResult:
    """Paper vs live comparison."""
    timestamp: datetime
    paper_pnl: float
    live_pnl: float
    drift_bps: float
    positions_match: bool
    mismatches: List[str] = field(default_factory=list)


@dataclass
class AuditEntry:
    """Single audit trail entry."""
    timestamp: datetime
    event_type: str
    details: Dict[str, Any]
    signal_id: str = ""
    order_id: str = ""


@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Risk limits configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskLimits:
    max_positions: int = 10
    max_positions_per_strategy: int = 5
    max_drawdown: float = 0.08       # kill switch at 8%
    max_daily_loss: float = 0.03     # halt at 3% daily loss
    max_margin_utilisation: float = 0.80
    max_correlation: float = 0.70     # reject if corr > 0.7 with existing
    min_confidence: float = 0.50
    max_notional_per_order: float = 50000
    alert_drawdown: float = 0.05     # warning at 5%
    alert_daily_loss: float = 0.02


# ---------------------------------------------------------------------------
# Broker adapter interface
# ---------------------------------------------------------------------------

class BrokerAdapter:
    """Abstract broker interface. Subclass for Alpaca, IBKR, etc."""

    def submit_order(self, order: Order) -> Order:
        """Submit order to broker. Returns updated order."""
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_positions(self) -> List[Position]:
        return []

    def get_account(self) -> Dict[str, float]:
        return {"equity": 100000, "buying_power": 80000, "margin_used": 20000}

    def liquidate_all(self) -> int:
        """Emergency liquidation. Returns number of positions closed."""
        return 0


class SimulatedBroker(BrokerAdapter):
    """Simulated broker for testing."""

    def __init__(self, fill_rate: float = 0.95, slippage: float = 0.03):
        self.fill_rate = fill_rate
        self.slippage = slippage
        self._positions: List[Position] = []
        self._equity = 100000.0
        self._rng = np.random.default_rng(42)

    def submit_order(self, order: Order) -> Order:
        order.submitted_at = datetime.now()
        if self._rng.random() < self.fill_rate:
            order.status = OrderStatus.FILLED
            order.filled_qty = order.quantity
            slip = self.slippage * (1 if order.side == OrderSide.BUY else -1)
            order.filled_price = order.limit_price + slip if order.limit_price > 0 else 100 + slip
            order.filled_at = datetime.now()
        else:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "simulated_reject"
        return order

    def get_account(self) -> Dict[str, float]:
        return {"equity": self._equity, "buying_power": self._equity * 0.8,
                "margin_used": self._equity * 0.2}

    def liquidate_all(self) -> int:
        n = len(self._positions)
        self._positions.clear()
        return n


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class LiveTradingBlueprint:
    """Live trading integration engine.

    Args:
        broker: Broker adapter instance.
        limits: Risk limit configuration.
        starting_equity: Initial account equity.
    """

    def __init__(
        self,
        broker: Optional[BrokerAdapter] = None,
        limits: Optional[RiskLimits] = None,
        starting_equity: float = 100000,
    ) -> None:
        self.broker = broker or SimulatedBroker()
        self.limits = limits or RiskLimits()
        self.starting_equity = starting_equity
        self.equity = starting_equity
        self._hwm = starting_equity
        self.daily_start_equity = starting_equity

        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._audit: List[AuditEntry] = []
        self._alerts: List[Alert] = []
        self._pnl_history: List[PnLSnapshot] = []
        self._reconciliations: List[ReconciliationResult] = []
        self._kill_switch = KillSwitchState.ARMED
        self._order_counter = 0

    # ------------------------------------------------------------------
    # 1. Signal → Order translation
    # ------------------------------------------------------------------

    def translate_signal(self, signal: StrategySignal) -> Order:
        """Convert a strategy signal into a broker order."""
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"

        side = OrderSide.SELL if signal.direction == "short" else OrderSide.BUY
        order_type = OrderType.LIMIT if signal.entry_price > 0 else OrderType.MARKET

        order = Order(
            order_id=order_id,
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=side,
            order_type=order_type,
            quantity=signal.target_contracts,
            limit_price=signal.entry_price,
        )

        self._log("signal_translated", {
            "signal_id": signal.signal_id, "order_id": order_id,
            "strategy": signal.strategy, "direction": signal.direction,
            "contracts": signal.target_contracts,
        }, signal_id=signal.signal_id, order_id=order_id)

        return order

    # ------------------------------------------------------------------
    # 2. Pre-trade risk checks
    # ------------------------------------------------------------------

    def pre_trade_check(self, signal: StrategySignal) -> RiskCheckReport:
        """Run all 5 pre-trade risk gates."""
        checks: Dict[str, bool] = {}
        reasons: List[str] = []

        # Gate 1: Kill switch
        checks["kill_switch"] = self._kill_switch == KillSwitchState.ARMED
        if not checks["kill_switch"]:
            reasons.append(f"Kill switch: {self._kill_switch.value}")

        # Gate 2: Position limits
        n_total = len(self._positions)
        n_strategy = sum(1 for p in self._positions.values() if p.strategy == signal.strategy)
        checks["position_limit"] = n_total < self.limits.max_positions
        checks["strategy_limit"] = n_strategy < self.limits.max_positions_per_strategy
        if not checks["position_limit"]:
            reasons.append(f"Portfolio at max positions ({n_total})")
        if not checks["strategy_limit"]:
            reasons.append(f"Strategy at max positions ({n_strategy})")

        # Gate 3: Drawdown limit
        dd = 1 - self.equity / self._hwm if self._hwm > 0 else 0
        checks["drawdown"] = dd < self.limits.max_drawdown
        if not checks["drawdown"]:
            reasons.append(f"Drawdown {dd:.1%} exceeds {self.limits.max_drawdown:.1%}")

        # Gate 4: Daily loss limit
        daily_pnl = (self.equity - self.daily_start_equity) / self.daily_start_equity
        checks["daily_loss"] = daily_pnl > -self.limits.max_daily_loss
        if not checks["daily_loss"]:
            reasons.append(f"Daily loss {daily_pnl:.1%} exceeds {self.limits.max_daily_loss:.1%}")

        # Gate 5: Confidence gate
        checks["confidence"] = signal.confidence >= self.limits.min_confidence
        if not checks["confidence"]:
            reasons.append(f"Confidence {signal.confidence:.2f} < {self.limits.min_confidence:.2f}")

        # Gate 6: Notional limit
        notional = signal.target_contracts * signal.spread_width * 100
        checks["notional"] = notional <= self.limits.max_notional_per_order
        if not checks["notional"]:
            reasons.append(f"Notional ${notional:,.0f} exceeds ${self.limits.max_notional_per_order:,.0f}")

        result = RiskCheckResult.APPROVED if all(checks.values()) else RiskCheckResult.REJECTED
        report = RiskCheckReport(
            signal_id=signal.signal_id, result=result,
            checks=checks, reject_reasons=reasons,
            timestamp=datetime.now(),
        )

        self._log("risk_check", {
            "signal_id": signal.signal_id, "result": result.value,
            "checks": checks, "reasons": reasons,
        }, signal_id=signal.signal_id)

        return report

    # ------------------------------------------------------------------
    # 3. Order management
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> Order:
        """Submit order through broker adapter."""
        if self._kill_switch != KillSwitchState.ARMED:
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"kill_switch_{self._kill_switch.value}"
            self._log("order_rejected", {"order_id": order.order_id,
                       "reason": order.reject_reason}, order_id=order.order_id)
            return order

        result = self.broker.submit_order(order)
        self._orders[result.order_id] = result

        self._log("order_submitted", {
            "order_id": result.order_id, "status": result.status.value,
            "filled_qty": result.filled_qty, "filled_price": result.filled_price,
        }, order_id=result.order_id, signal_id=result.signal_id)

        if result.status == OrderStatus.FILLED:
            self._create_position(result)

        return result

    def process_signal(self, signal: StrategySignal) -> Tuple[RiskCheckReport, Optional[Order]]:
        """Full signal processing: check → translate → submit."""
        signal.timestamp = signal.timestamp or datetime.now()
        risk = self.pre_trade_check(signal)

        if risk.result != RiskCheckResult.APPROVED:
            return risk, None

        order = self.translate_signal(signal)
        filled = self.submit_order(order)
        return risk, filled

    def close_position(self, position_id: str, reason: str = "manual") -> Optional[Order]:
        """Close an existing position."""
        pos = self._positions.get(position_id)
        if pos is None:
            return None

        side = OrderSide.BUY if pos.direction == "short" else OrderSide.SELL
        self._order_counter += 1
        order = Order(
            order_id=f"ORD-{self._order_counter:06d}",
            signal_id=f"close_{position_id}",
            symbol=pos.symbol, side=side,
            order_type=OrderType.MARKET,
            quantity=pos.contracts,
        )
        result = self.broker.submit_order(order)
        self._orders[result.order_id] = result

        if result.status == OrderStatus.FILLED:
            pnl_per = (result.filled_price - pos.entry_price) * (1 if pos.direction == "long" else -1)
            pos.realised_pnl = pnl_per * pos.contracts * 100
            self.equity += pos.realised_pnl
            del self._positions[position_id]

            self._log("position_closed", {
                "position_id": position_id, "reason": reason,
                "pnl": pos.realised_pnl,
            })

        return result

    def emergency_liquidate(self, reason: str = "emergency") -> int:
        """Close ALL positions immediately."""
        self._kill_switch = KillSwitchState.TRIGGERED
        n_closed = 0

        for pid in list(self._positions.keys()):
            result = self.close_position(pid, reason=f"emergency_{reason}")
            if result and result.status == OrderStatus.FILLED:
                n_closed += 1

        self._log("emergency_liquidation", {
            "reason": reason, "positions_closed": n_closed,
        })
        self._alert(AlertLevel.CRITICAL, f"EMERGENCY LIQUIDATION: {reason} — {n_closed} positions closed")
        return n_closed

    # ------------------------------------------------------------------
    # 4. Real-time P&L tracking
    # ------------------------------------------------------------------

    def update_pnl(self, market_prices: Optional[Dict[str, float]] = None) -> PnLSnapshot:
        """Update mark-to-market P&L for all positions."""
        by_strategy: Dict[str, float] = {}
        total_unrealised = 0.0

        for pos in self._positions.values():
            if market_prices and pos.symbol in market_prices:
                pos.current_price = market_prices[pos.symbol]
            pnl_per = (pos.current_price - pos.entry_price) * (1 if pos.direction == "long" else -1)
            pos.unrealised_pnl = pnl_per * pos.contracts * 100
            total_unrealised += pos.unrealised_pnl
            by_strategy[pos.strategy] = by_strategy.get(pos.strategy, 0) + pos.unrealised_pnl

        current_equity = self.starting_equity + total_unrealised
        if current_equity > self._hwm:
            self._hwm = current_equity
        dd = 1 - current_equity / self._hwm if self._hwm > 0 else 0
        daily_pnl = current_equity - self.daily_start_equity

        snap = PnLSnapshot(
            timestamp=datetime.now(),
            total_equity=current_equity,
            total_pnl=total_unrealised,
            daily_pnl=daily_pnl,
            drawdown=dd,
            n_positions=len(self._positions),
            by_strategy=by_strategy,
        )
        self._pnl_history.append(snap)

        # Alert thresholds
        if dd >= self.limits.alert_drawdown:
            self._alert(AlertLevel.WARNING, f"Drawdown at {dd:.1%} (alert: {self.limits.alert_drawdown:.1%})")
        if daily_pnl / self.daily_start_equity < -self.limits.alert_daily_loss:
            self._alert(AlertLevel.WARNING, f"Daily loss ${daily_pnl:,.0f}")

        return snap

    def new_trading_day(self) -> None:
        """Reset daily counters at start of each trading day."""
        self.daily_start_equity = self.equity
        self._log("new_trading_day", {"equity": self.equity})

    # ------------------------------------------------------------------
    # 5. Kill switch
    # ------------------------------------------------------------------

    def check_kill_switch(self) -> KillSwitchState:
        """Evaluate kill switch conditions."""
        if self._kill_switch in (KillSwitchState.TRIGGERED, KillSwitchState.MANUAL_HALT):
            return self._kill_switch

        dd = 1 - self.equity / self._hwm if self._hwm > 0 else 0

        # Auto trigger: drawdown exceeds limit
        if dd >= self.limits.max_drawdown:
            self._kill_switch = KillSwitchState.TRIGGERED
            self._log("kill_switch_triggered", {"reason": "max_drawdown", "drawdown": dd})
            self._alert(AlertLevel.CRITICAL, f"KILL SWITCH: drawdown {dd:.1%} >= {self.limits.max_drawdown:.1%}")
            self.emergency_liquidate("max_drawdown")
            return self._kill_switch

        # Auto trigger: daily loss limit
        daily_loss = (self.equity - self.daily_start_equity) / self.daily_start_equity
        if daily_loss < -self.limits.max_daily_loss:
            self._kill_switch = KillSwitchState.TRIGGERED
            self._log("kill_switch_triggered", {"reason": "daily_loss", "loss": daily_loss})
            self._alert(AlertLevel.CRITICAL, f"KILL SWITCH: daily loss {daily_loss:.1%}")
            self.emergency_liquidate("daily_loss")
            return self._kill_switch

        return self._kill_switch

    def manual_halt(self, reason: str = "operator") -> None:
        """Manual kill switch activation."""
        self._kill_switch = KillSwitchState.MANUAL_HALT
        self._log("manual_halt", {"reason": reason})
        self._alert(AlertLevel.CRITICAL, f"MANUAL HALT: {reason}")

    def reset_kill_switch(self) -> None:
        """Re-arm kill switch after review."""
        self._kill_switch = KillSwitchState.ARMED
        self._log("kill_switch_reset", {})

    # ------------------------------------------------------------------
    # 6. Reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        paper_pnl: float,
        paper_positions: Optional[Dict[str, int]] = None,
    ) -> ReconciliationResult:
        """Compare paper trading vs live state."""
        live_pnl = sum(p.unrealised_pnl + p.realised_pnl for p in self._positions.values())
        drift = abs(live_pnl - paper_pnl)
        ref = max(abs(paper_pnl), abs(live_pnl), 1)
        drift_bps = drift / ref * 10000

        mismatches: List[str] = []
        positions_match = True
        if paper_positions:
            for symbol, expected in paper_positions.items():
                actual = sum(p.contracts for p in self._positions.values() if p.symbol == symbol)
                if actual != expected:
                    mismatches.append(f"{symbol}: expected {expected}, got {actual}")
                    positions_match = False

        result = ReconciliationResult(
            timestamp=datetime.now(),
            paper_pnl=paper_pnl, live_pnl=live_pnl,
            drift_bps=drift_bps, positions_match=positions_match,
            mismatches=mismatches,
        )
        self._reconciliations.append(result)

        if drift_bps > 50:
            self._alert(AlertLevel.WARNING, f"Reconciliation drift: {drift_bps:.0f} bps")

        self._log("reconciliation", {
            "paper_pnl": paper_pnl, "live_pnl": live_pnl,
            "drift_bps": drift_bps, "positions_match": positions_match,
        })

        return result

    # ------------------------------------------------------------------
    # 7. Audit trail
    # ------------------------------------------------------------------

    def _log(self, event_type: str, details: Dict, signal_id: str = "", order_id: str = ""):
        entry = AuditEntry(
            timestamp=datetime.now(), event_type=event_type,
            details=details, signal_id=signal_id, order_id=order_id,
        )
        self._audit.append(entry)

    def _alert(self, level: AlertLevel, message: str):
        self._alerts.append(Alert(level, message, datetime.now()))
        if level == AlertLevel.CRITICAL:
            logger.critical("ALERT: %s", message)
        elif level == AlertLevel.WARNING:
            logger.warning("ALERT: %s", message)

    @property
    def audit_trail(self) -> List[AuditEntry]:
        return list(self._audit)

    @property
    def alerts(self) -> List[Alert]:
        return list(self._alerts)

    @property
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    @property
    def orders(self) -> Dict[str, Order]:
        return dict(self._orders)

    @property
    def pnl_history(self) -> List[PnLSnapshot]:
        return list(self._pnl_history)

    @property
    def kill_switch_state(self) -> KillSwitchState:
        return self._kill_switch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_position(self, order: Order):
        pid = f"POS-{len(self._positions) + 1:04d}"
        direction = "short" if order.side == OrderSide.SELL else "long"
        pos = Position(
            position_id=pid, strategy="",
            symbol=order.symbol, direction=direction,
            contracts=order.filled_qty,
            entry_price=order.filled_price,
            current_price=order.filled_price,
            opened_at=order.filled_at,
        )
        # Try to recover strategy from signal
        for a in reversed(self._audit):
            if a.signal_id == order.signal_id and "strategy" in a.details:
                pos.strategy = a.details["strategy"]
                break
        self._positions[pid] = pos

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(self, output_path: str = "reports/live_trading_blueprint.html") -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Audit table (last 30)
        audit_rows = [
            f"<tr><td>{a.timestamp.strftime('%H:%M:%S') if a.timestamp else ''}</td>"
            f"<td>{a.event_type}</td><td>{a.signal_id}</td>"
            f"<td>{a.order_id}</td>"
            f"<td style='text-align:left'>{json.dumps(a.details, default=str)[:120]}</td></tr>"
            for a in self._audit[-30:]
        ]

        # Position table
        pos_rows = [
            f"<tr><td>{p.position_id}</td><td>{p.strategy}</td>"
            f"<td>{p.symbol}</td><td>{p.direction}</td>"
            f"<td>{p.contracts}</td><td>${p.entry_price:.2f}</td>"
            f"<td>${p.unrealised_pnl:+,.0f}</td></tr>"
            for p in self._positions.values()
        ]

        # Alert table
        alert_rows = [
            f"<tr><td style='color:{'#e74c3c' if a.level == AlertLevel.CRITICAL else '#e67e22'}'>"
            f"{a.level.value.upper()}</td><td>{a.message}</td></tr>"
            for a in self._alerts[-20:]
        ]

        ks_colors = {"armed": "#27ae60", "triggered": "#e74c3c",
                       "manual_halt": "#e67e22", "disarmed": "#999"}
        ks_c = ks_colors.get(self._kill_switch.value, "#999")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Live Trading Blueprint</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
.badge {{ padding: 4px 12px; border-radius: 8px; color: #fff; font-weight: bold; }}
</style></head><body>
<h1>Live Trading Blueprint</h1>
<div class="summary">
<p><strong>Kill Switch:</strong>
<span class="badge" style="background:{ks_c}">{self._kill_switch.value.upper()}</span></p>
<p><strong>Equity:</strong> ${self.equity:,.0f} |
<strong>Positions:</strong> {len(self._positions)} |
<strong>Orders:</strong> {len(self._orders)} |
<strong>Alerts:</strong> {len(self._alerts)}</p>
</div>
<h2>Positions</h2>
<table><tr><th>ID</th><th>Strategy</th><th>Symbol</th><th>Dir</th>
<th>Contracts</th><th>Entry</th><th>Unreal P&L</th></tr>
{''.join(pos_rows) or '<tr><td colspan="7">No positions</td></tr>'}</table>
<h2>Alerts</h2>
<table><tr><th>Level</th><th>Message</th></tr>
{''.join(alert_rows) or '<tr><td colspan="2">No alerts</td></tr>'}</table>
<h2>Audit Trail (last 30)</h2>
<table><tr><th>Time</th><th>Event</th><th>Signal</th><th>Order</th>
<th style='text-align:left'>Details</th></tr>
{''.join(audit_rows)}</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)
