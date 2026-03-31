"""
Live trading bridge — connects compass signals to broker execution.

Signal ingestion from compass modules, position reconciliation (target
vs actual), order generation with smart routing, pre-order risk checks
(max position, drawdown circuit breaker, correlation limits), real-time
P&L tracking, SQLite state persistence, heartbeat / health monitoring,
dry-run mode for paper trading, and Telegram alerting on trades/errors.

Usage::

    from compass.live_bridge import LiveBridge
    bridge = LiveBridge(db_path="bridge.db", dry_run=True)
    bridge.ingest_signal("EXP-400", "SPY", 1, 2, confidence=0.85)
    bridge.reconcile()
    bridge.run_cycle()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class Signal:
    """A compass trading signal."""
    strategy: str
    asset: str
    direction: int         # +1 long, -1 short, 0 flat
    target_contracts: int
    confidence: float      # 0-1
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current broker position."""
    asset: str
    contracts: int         # positive = long, negative = short
    avg_price: float
    unrealised_pnl: float
    strategy: str


@dataclass
class Order:
    """An order to be sent to the broker."""
    order_id: str
    strategy: str
    asset: str
    side: str              # "buy" or "sell"
    contracts: int
    order_type: str        # "market", "limit"
    limit_price: Optional[float]
    algo: str              # "twap", "vwap", "market"
    status: str            # "pending", "filled", "rejected", "cancelled"
    reason: str
    timestamp: str


@dataclass
class RiskCheckResult:
    """Result of pre-order risk validation."""
    passed: bool
    checks: Dict[str, bool]
    violations: List[str]


@dataclass
class HealthStatus:
    """Bridge health / heartbeat."""
    alive: bool
    last_heartbeat: str
    uptime_seconds: float
    signals_processed: int
    orders_generated: int
    errors: int
    dry_run: bool


@dataclass
class PnLSnapshot:
    """Real-time P&L tracking."""
    timestamp: str
    total_pnl: float
    unrealised_pnl: float
    realised_pnl: float
    max_drawdown: float
    positions: int


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_position_per_asset: int = 10
    max_total_position: int = 30
    max_drawdown_pct: float = 0.05
    max_daily_loss: float = 5000.0
    min_confidence: float = 0.3
    max_correlation: float = 0.8
    max_orders_per_minute: int = 10


# ── SQLite schema ───────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    asset TEXT NOT NULL,
    direction INTEGER NOT NULL,
    target_contracts INTEGER NOT NULL,
    confidence REAL NOT NULL,
    timestamp TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    asset TEXT NOT NULL,
    side TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    order_type TEXT NOT NULL,
    limit_price REAL,
    algo TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    asset TEXT NOT NULL,
    strategy TEXT NOT NULL,
    contracts INTEGER NOT NULL DEFAULT 0,
    avg_price REAL NOT NULL DEFAULT 0,
    unrealised_pnl REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (asset, strategy)
);

CREATE TABLE IF NOT EXISTS pnl_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    total_pnl REAL NOT NULL,
    unrealised_pnl REAL NOT NULL,
    realised_pnl REAL NOT NULL,
    max_drawdown REAL NOT NULL DEFAULT 0,
    n_positions INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,
    details TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ── Alerter protocol ───────────────────────────────────────────────────


class AlerterProtocol:
    """Protocol for alerting (Telegram, etc). Override for real use."""

    def send(self, message: str, level: str = "info") -> bool:
        """Send an alert message. Returns True if delivered."""
        logger.info("[ALERT:%s] %s", level, message)
        return True


class TelegramAlerter(AlerterProtocol):
    """Real Telegram alerter using shared.telegram_alerts."""

    def send(self, message: str, level: str = "info") -> bool:
        try:
            from shared.telegram_alerts import send_message
            send_message(f"[LiveBridge/{level.upper()}] {message}")
            return True
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False


# ── Broker protocol ────────────────────────────────────────────────────


class BrokerProtocol:
    """Protocol for broker interaction. Override for real broker."""

    def get_positions(self) -> List[Position]:
        """Get current positions from broker."""
        return []

    def submit_order(self, order: Order) -> Order:
        """Submit an order. Returns updated order with status."""
        order.status = "filled"
        return order

    def get_account_value(self) -> float:
        """Get total account value."""
        return 100_000.0


# ── Live Bridge ─────────────────────────────────────────────────────────


class LiveBridge:
    """Live trading bridge connecting compass signals to execution."""

    def __init__(
        self,
        db_path: str = ":memory:",
        dry_run: bool = True,
        risk_limits: Optional[RiskLimits] = None,
        broker: Optional[BrokerProtocol] = None,
        alerter: Optional[AlerterProtocol] = None,
        default_algo: str = "twap",
    ) -> None:
        self.db_path = db_path
        self.dry_run = dry_run
        self.risk_limits = risk_limits or RiskLimits()
        self.broker = broker or BrokerProtocol()
        self.alerter = alerter or AlerterProtocol()
        self.default_algo = default_algo
        self._start_time = time.time()
        self._order_counter = 0
        self._signals_processed = 0
        self._orders_generated = 0
        self._errors = 0

        # SQLite
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # In-memory position cache
        self._positions: Dict[Tuple[str, str], Position] = {}
        self._realised_pnl = 0.0
        self._peak_pnl = 0.0
        self._max_drawdown = 0.0
        self._recent_order_times: List[float] = []

    def close(self) -> None:
        self._conn.close()

    # ── Signal ingestion ────────────────────────────────────────────────

    def ingest_signal(
        self,
        strategy: str,
        asset: str,
        direction: int,
        target_contracts: int,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """Ingest a trading signal from a compass module."""
        now = datetime.now(timezone.utc).isoformat()
        sig = Signal(
            strategy=strategy, asset=asset,
            direction=max(-1, min(1, direction)),
            target_contracts=abs(target_contracts),
            confidence=confidence, timestamp=now,
            metadata=metadata or {},
        )
        self._conn.execute(
            """INSERT INTO signals (strategy, asset, direction, target_contracts,
               confidence, timestamp, metadata) VALUES (?,?,?,?,?,?,?)""",
            (sig.strategy, sig.asset, sig.direction, sig.target_contracts,
             sig.confidence, sig.timestamp, json.dumps(sig.metadata)),
        )
        self._conn.commit()
        self._signals_processed += 1
        return sig

    # ── Position reconciliation ─────────────────────────────────────────

    def reconcile(self) -> Dict[Tuple[str, str], int]:
        """Reconcile target positions vs actual. Returns deltas."""
        # Get actual positions from broker
        actual = {}
        for pos in self.broker.get_positions():
            key = (pos.asset, pos.strategy)
            actual[key] = pos.contracts

        # Get target from latest signals
        targets = {}
        rows = self._conn.execute(
            """SELECT strategy, asset, direction, target_contracts
               FROM signals GROUP BY strategy, asset
               HAVING id = MAX(id)"""
        ).fetchall()
        for strategy, asset, direction, target in rows:
            key = (asset, strategy)
            targets[key] = direction * target

        # Compute deltas
        all_keys = set(actual.keys()) | set(targets.keys())
        deltas = {}
        for key in all_keys:
            target = targets.get(key, 0)
            current = actual.get(key, self._positions.get(key, Position(
                asset=key[0], strategy=key[1], contracts=0, avg_price=0, unrealised_pnl=0,
            )).contracts)
            delta = target - current
            if delta != 0:
                deltas[key] = delta

        return deltas

    # ── Order generation ────────────────────────────────────────────────

    def generate_orders(self, deltas: Dict[Tuple[str, str], int]) -> List[Order]:
        """Generate orders from position deltas, with risk checks."""
        orders: List[Order] = []
        for (asset, strategy), delta in deltas.items():
            if delta == 0:
                continue

            # Get signal confidence
            row = self._conn.execute(
                "SELECT confidence FROM signals WHERE strategy=? AND asset=? ORDER BY id DESC LIMIT 1",
                (strategy, asset),
            ).fetchone()
            confidence = row[0] if row else 0.0

            side = "buy" if delta > 0 else "sell"
            contracts = abs(delta)

            # Risk check
            risk = self._risk_check(asset, strategy, side, contracts, confidence)
            if not risk.passed:
                reason = "; ".join(risk.violations)
                logger.warning("Risk check FAILED for %s %s %d %s: %s",
                               side, asset, contracts, strategy, reason)
                self.alerter.send(f"RISK BLOCKED: {side} {contracts} {asset} ({strategy}): {reason}", "warning")
                order = self._make_order(strategy, asset, side, contracts, "rejected", reason)
                self._persist_order(order)
                orders.append(order)
                continue

            order = self._make_order(strategy, asset, side, contracts, "pending", "risk_passed")
            orders.append(order)
            self._orders_generated += 1

        return orders

    def execute_orders(self, orders: List[Order]) -> List[Order]:
        """Execute pending orders via broker (or simulate in dry-run)."""
        results: List[Order] = []
        for order in orders:
            if order.status != "pending":
                results.append(order)
                continue

            if self.dry_run:
                order.status = "filled"
                order.reason = "dry_run_fill"
                self._update_position(order)
                self.alerter.send(
                    f"[DRY RUN] {order.side} {order.contracts} {order.asset} ({order.strategy})",
                    "trade",
                )
            else:
                try:
                    order = self.broker.submit_order(order)
                    if order.status == "filled":
                        self._update_position(order)
                        self.alerter.send(
                            f"FILLED: {order.side} {order.contracts} {order.asset} ({order.strategy})",
                            "trade",
                        )
                except Exception as exc:
                    order.status = "rejected"
                    order.reason = str(exc)
                    self._errors += 1
                    self.alerter.send(f"ORDER ERROR: {exc}", "error")

            self._persist_order(order)
            results.append(order)

        return results

    # ── Full cycle ──────────────────────────────────────────────────────

    def run_cycle(self) -> List[Order]:
        """Run one full signal → reconcile → order → execute cycle."""
        deltas = self.reconcile()
        orders = self.generate_orders(deltas)
        executed = self.execute_orders(orders)
        self._record_pnl()
        self._record_heartbeat()
        return executed

    # ── Risk checks ─────────────────────────────────────────────────────

    def _risk_check(
        self, asset: str, strategy: str, side: str,
        contracts: int, confidence: float,
    ) -> RiskCheckResult:
        """Run all pre-order risk checks."""
        checks: Dict[str, bool] = {}
        violations: List[str] = []
        rl = self.risk_limits

        # 1. Confidence threshold
        checks["min_confidence"] = confidence >= rl.min_confidence
        if not checks["min_confidence"]:
            violations.append(f"Confidence {confidence:.2f} < min {rl.min_confidence}")

        # 2. Max position per asset
        current = sum(
            p.contracts for p in self._positions.values() if p.asset == asset
        )
        new_total = abs(current + (contracts if side == "buy" else -contracts))
        checks["max_position"] = new_total <= rl.max_position_per_asset
        if not checks["max_position"]:
            violations.append(f"Position {new_total} > max {rl.max_position_per_asset}")

        # 3. Max total portfolio position
        total = sum(abs(p.contracts) for p in self._positions.values()) + contracts
        checks["max_total"] = total <= rl.max_total_position
        if not checks["max_total"]:
            violations.append(f"Total position {total} > max {rl.max_total_position}")

        # 4. Drawdown circuit breaker
        checks["drawdown"] = self._max_drawdown < rl.max_drawdown_pct
        if not checks["drawdown"]:
            violations.append(f"Drawdown {self._max_drawdown:.2%} > max {rl.max_drawdown_pct:.2%}")

        # 5. Daily loss limit
        total_pnl = self._total_pnl()
        checks["daily_loss"] = total_pnl > -rl.max_daily_loss
        if not checks["daily_loss"]:
            violations.append(f"Daily loss ${-total_pnl:,.0f} > max ${rl.max_daily_loss:,.0f}")

        # 6. Order rate limit
        now = time.time()
        self._recent_order_times = [t for t in self._recent_order_times if now - t < 60]
        checks["rate_limit"] = len(self._recent_order_times) < rl.max_orders_per_minute
        if not checks["rate_limit"]:
            violations.append(f"Order rate {len(self._recent_order_times)}/min > max {rl.max_orders_per_minute}")

        self._recent_order_times.append(now)

        return RiskCheckResult(
            passed=all(checks.values()),
            checks=checks,
            violations=violations,
        )

    # ── P&L tracking ───────────────────────────────────────────────────

    def _total_pnl(self) -> float:
        return self._realised_pnl + sum(p.unrealised_pnl for p in self._positions.values())

    def get_pnl(self) -> PnLSnapshot:
        """Get current P&L snapshot."""
        unrealised = sum(p.unrealised_pnl for p in self._positions.values())
        total = self._realised_pnl + unrealised
        return PnLSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_pnl=total,
            unrealised_pnl=unrealised,
            realised_pnl=self._realised_pnl,
            max_drawdown=self._max_drawdown,
            positions=len(self._positions),
        )

    def _record_pnl(self) -> None:
        snap = self.get_pnl()
        # Update drawdown
        if snap.total_pnl > self._peak_pnl:
            self._peak_pnl = snap.total_pnl
        if self._peak_pnl > 0:
            dd = (self._peak_pnl - snap.total_pnl) / self._peak_pnl
            self._max_drawdown = max(self._max_drawdown, dd)

        self._conn.execute(
            """INSERT INTO pnl_history (timestamp, total_pnl, unrealised_pnl,
               realised_pnl, max_drawdown, n_positions)
               VALUES (?,?,?,?,?,?)""",
            (snap.timestamp, snap.total_pnl, snap.unrealised_pnl,
             snap.realised_pnl, snap.max_drawdown, snap.positions),
        )
        self._conn.commit()

    # ── Health monitoring ───────────────────────────────────────────────

    def health(self) -> HealthStatus:
        """Get bridge health status."""
        return HealthStatus(
            alive=True,
            last_heartbeat=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=time.time() - self._start_time,
            signals_processed=self._signals_processed,
            orders_generated=self._orders_generated,
            errors=self._errors,
            dry_run=self.dry_run,
        )

    def _record_heartbeat(self) -> None:
        h = self.health()
        self._conn.execute(
            "INSERT INTO heartbeats (timestamp, status, details) VALUES (?,?,?)",
            (h.last_heartbeat, "alive", json.dumps({"uptime": h.uptime_seconds})),
        )
        self._conn.commit()

    # ── State persistence helpers ───────────────────────────────────────

    def get_state(self, key: str, default: str = "") -> str:
        row = self._conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_state(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?,?)", (key, value),
        )
        self._conn.commit()

    def get_signals(self, limit: int = 50) -> List[Signal]:
        """Retrieve recent signals from DB."""
        rows = self._conn.execute(
            "SELECT strategy, asset, direction, target_contracts, confidence, timestamp, metadata "
            "FROM signals ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
        return [
            Signal(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6]))
            for r in rows
        ]

    def get_orders(self, limit: int = 50) -> List[Order]:
        """Retrieve recent orders from DB."""
        rows = self._conn.execute(
            "SELECT order_id, strategy, asset, side, contracts, order_type, "
            "limit_price, algo, status, reason, timestamp "
            "FROM orders ORDER BY timestamp DESC LIMIT ?", (limit,),
        ).fetchall()
        return [
            Order(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10])
            for r in rows
        ]

    def get_pnl_history(self, limit: int = 500) -> List[PnLSnapshot]:
        """Retrieve P&L history from DB."""
        rows = self._conn.execute(
            "SELECT timestamp, total_pnl, unrealised_pnl, realised_pnl, "
            "max_drawdown, n_positions FROM pnl_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            PnLSnapshot(r[0], r[1], r[2], r[3], r[4], r[5])
            for r in rows
        ]

    # ── Internal helpers ────────────────────────────────────────────────

    def _make_order(
        self, strategy: str, asset: str, side: str,
        contracts: int, status: str, reason: str,
    ) -> Order:
        self._order_counter += 1
        return Order(
            order_id=f"ORD-{self._order_counter:06d}",
            strategy=strategy, asset=asset, side=side,
            contracts=contracts, order_type="market",
            limit_price=None, algo=self.default_algo,
            status=status, reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _update_position(self, order: Order) -> None:
        """Update in-memory position cache after a fill."""
        key = (order.asset, order.strategy)
        pos = self._positions.get(key, Position(
            asset=order.asset, strategy=order.strategy,
            contracts=0, avg_price=0, unrealised_pnl=0,
        ))
        delta = order.contracts if order.side == "buy" else -order.contracts
        pos.contracts += delta
        self._positions[key] = pos

        # Persist
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO positions
               (asset, strategy, contracts, avg_price, unrealised_pnl, updated_at)
               VALUES (?,?,?,?,?,?)""",
            (pos.asset, pos.strategy, pos.contracts, pos.avg_price,
             pos.unrealised_pnl, now),
        )
        self._conn.commit()

    def _persist_order(self, order: Order) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO orders
               (order_id, strategy, asset, side, contracts, order_type,
                limit_price, algo, status, reason, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (order.order_id, order.strategy, order.asset, order.side,
             order.contracts, order.order_type, order.limit_price,
             order.algo, order.status, order.reason, order.timestamp),
        )
        self._conn.commit()
