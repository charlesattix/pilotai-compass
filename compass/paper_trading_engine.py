"""
Paper trading engine — forward-testing framework with realistic execution.

Signal ingestion, fill simulation (slippage, partial fills, delay),
position tracking with margin, daily P&L attribution per strategy and
portfolio, risk limit enforcement, and performance dashboard export.

Usage::

    from compass.paper_trading_engine import PaperTradingEngine, EngineConfig
    engine = PaperTradingEngine(EngineConfig(starting_capital=100_000))
    engine.submit_signal(signal)
    engine.step(date, market_prices)
    engine.generate_report("reports/paper_engine.html")
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "paper_engine.html"


# ── Configuration ───────────────────────────────────────────────────────


@dataclass
class EngineConfig:
    starting_capital: float = 100_000.0
    slippage_per_contract: float = 0.04    # $0.03-0.05 range
    fill_rate: float = 0.95                # 95% of orders fill
    fill_delay_bars: int = 0               # bars of delay before fill
    margin_per_spread: float = 500.0       # margin requirement per spread
    max_positions: int = 20
    max_position_per_strategy: int = 10
    max_drawdown_pct: float = 0.12         # 12% DD circuit breaker
    max_daily_loss: float = 5_000.0
    max_portfolio_delta: float = 50.0      # aggregate delta limit
    correlation_limit: float = 0.80        # reject if corr > this
    db_path: str = ":memory:"


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class Signal:
    """Trading signal from a strategy."""
    signal_id: str = ""
    strategy: str = ""
    ticker: str = "SPY"
    direction: str = "short"      # "long" or "short"
    spread_type: str = "bull_put"
    contracts: int = 1
    net_credit: float = 1.0       # per contract
    max_loss: float = 4.0         # per contract
    spread_width: float = 5.0
    dte: int = 30
    stop_loss_pct: float = 3.5    # as multiple of credit
    profit_target_pct: float = 0.50
    confidence: float = 0.5
    regime: str = "neutral"
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = f"SIG-{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class Fill:
    """An executed fill."""
    fill_id: str
    signal_id: str
    strategy: str
    ticker: str
    side: str                     # "open" or "close"
    contracts: int
    price: float                  # fill price (credit or debit)
    slippage: float               # slippage cost
    commission: float
    timestamp: str
    partial: bool = False


@dataclass
class Position:
    """An open position."""
    position_id: str
    strategy: str
    ticker: str
    direction: str
    spread_type: str
    contracts: int
    entry_credit: float           # per contract, net of slippage
    max_loss: float
    spread_width: float
    margin_required: float
    entry_date: str
    expiration_date: str
    stop_loss_level: float
    profit_target_level: float
    current_value: float = 0.0    # estimated current spread value
    unrealised_pnl: float = 0.0
    regime_at_entry: str = "neutral"
    confidence: float = 0.5


@dataclass
class ClosedTrade:
    """A completed trade."""
    position_id: str
    strategy: str
    ticker: str
    direction: str
    spread_type: str
    contracts: int
    entry_credit: float
    exit_debit: float
    pnl: float                    # net of all costs
    return_pct: float
    slippage_total: float
    commission_total: float
    entry_date: str
    exit_date: str
    exit_reason: str
    hold_days: int
    regime_at_entry: str
    confidence: float
    win: bool


@dataclass
class DailyPnL:
    """Daily P&L snapshot."""
    date: str
    total_pnl: float
    realised_pnl: float
    unrealised_pnl: float
    n_positions: int
    margin_used: float
    capital: float
    drawdown: float
    by_strategy: Dict[str, float]


@dataclass
class RiskBreachEvent:
    """A risk limit violation."""
    timestamp: str
    breach_type: str              # "max_dd", "daily_loss", "position_limit", etc.
    limit: float
    actual: float
    action_taken: str             # "reject", "close_all", "reduce"


@dataclass
class PerformanceSummary:
    """Aggregate performance metrics."""
    total_pnl: float
    total_return_pct: float
    win_rate: float
    n_trades: int
    n_wins: int
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    profit_factor: float
    avg_pnl: float
    avg_winner: float
    avg_loser: float
    avg_hold_days: float
    total_slippage: float
    total_commission: float
    by_strategy: Dict[str, Dict[str, float]]
    by_regime: Dict[str, Dict[str, float]]


# ── Fill simulator ──────────────────────────────────────────────────────


class FillSimulator:
    """Simulate realistic fills with slippage and partial fills."""

    def __init__(self, config: EngineConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.RandomState(seed)

    def simulate_fill(self, signal: Signal, side: str) -> Optional[Fill]:
        """Simulate a fill. Returns None if order doesn't fill."""
        # Fill rate check
        if self.rng.random() > self.config.fill_rate:
            return None

        # Slippage: random in [0.5x, 1.5x] of configured amount
        slip_base = self.config.slippage_per_contract
        slip = slip_base * (0.5 + self.rng.random())
        total_slip = slip * signal.contracts

        # Commission: $0.65/contract × 2 legs
        commission = 0.65 * signal.contracts * 2

        if side == "open":
            price = signal.net_credit - slip  # receive less due to slippage
        else:
            price = signal.net_credit * 0.5 + slip  # pay more to close

        # Partial fill: 5% chance of partial (fill 60-90% of contracts)
        partial = False
        contracts = signal.contracts
        if self.rng.random() < 0.05 and contracts > 1:
            partial = True
            contracts = max(1, int(contracts * self.rng.uniform(0.6, 0.9)))

        return Fill(
            fill_id=f"FILL-{uuid.uuid4().hex[:8]}",
            signal_id=signal.signal_id,
            strategy=signal.strategy,
            ticker=signal.ticker,
            side=side,
            contracts=contracts,
            price=price,
            slippage=total_slip,
            commission=commission,
            timestamp=signal.timestamp,
            partial=partial,
        )


# ── Risk monitor ────────────────────────────────────────────────────────


class RiskMonitor:
    """Enforce risk limits."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.breaches: List[RiskBreachEvent] = []
        self.circuit_breaker_active = False

    def check_new_position(
        self,
        signal: Signal,
        positions: List[Position],
        capital: float,
        peak_capital: float,
    ) -> Tuple[bool, str]:
        """Check if a new position is allowed. Returns (allowed, reason)."""

        # Circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker active — no new positions"

        # Drawdown check
        dd = (capital - peak_capital) / peak_capital if peak_capital > 0 else 0
        if dd < -self.config.max_drawdown_pct:
            self.circuit_breaker_active = True
            self._record_breach("max_dd", self.config.max_drawdown_pct, abs(dd), "circuit_breaker")
            return False, f"Drawdown {dd:.1%} exceeds limit {self.config.max_drawdown_pct:.0%}"

        # Position count
        if len(positions) >= self.config.max_positions:
            return False, f"At max positions ({self.config.max_positions})"

        # Per-strategy limit
        strat_count = sum(1 for p in positions if p.strategy == signal.strategy)
        if strat_count >= self.config.max_position_per_strategy:
            return False, f"Strategy {signal.strategy} at limit ({self.config.max_position_per_strategy})"

        # Margin check
        margin_used = sum(p.margin_required for p in positions)
        new_margin = signal.contracts * self.config.margin_per_spread
        if margin_used + new_margin > capital * 0.80:  # 80% margin utilisation cap
            return False, f"Margin {margin_used + new_margin:,.0f} > 80% of capital"

        # Confidence gate
        if signal.confidence < 0.30:
            return False, f"Confidence {signal.confidence:.2f} below minimum 0.30"

        return True, "passed"

    def check_daily_loss(self, daily_pnl: float) -> bool:
        """Check if daily loss limit is breached."""
        if daily_pnl < -self.config.max_daily_loss:
            self._record_breach("daily_loss", self.config.max_daily_loss, abs(daily_pnl), "halt_trading")
            return True
        return False

    def reset_circuit_breaker(self) -> None:
        self.circuit_breaker_active = False

    def _record_breach(self, btype: str, limit: float, actual: float, action: str) -> None:
        self.breaches.append(RiskBreachEvent(
            datetime.now(timezone.utc).isoformat(), btype, limit, actual, action,
        ))


# ── Engine ──────────────────────────────────────────────────────────────


class PaperTradingEngine:
    """Paper trading simulation engine."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self.fill_sim = FillSimulator(self.config)
        self.risk_monitor = RiskMonitor(self.config)

        self.capital = self.config.starting_capital
        self.peak_capital = self.capital
        self.realised_pnl = 0.0

        self.positions: List[Position] = []
        self.closed_trades: List[ClosedTrade] = []
        self.fills: List[Fill] = []
        self.daily_pnl: List[DailyPnL] = []
        self.pending_signals: List[Signal] = []

        # SQLite persistence
        self._conn = sqlite3.connect(self.config.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def close(self) -> None:
        self._conn.close()

    # ── Signal submission ───────────────────────────────────────────────

    def submit_signal(self, signal: Signal) -> Tuple[bool, str]:
        """Submit a trading signal. Returns (accepted, reason)."""
        allowed, reason = self.risk_monitor.check_new_position(
            signal, self.positions, self.capital, self.peak_capital,
        )
        if not allowed:
            logger.info("Signal rejected: %s — %s", signal.signal_id, reason)
            return False, reason

        # Simulate fill
        fill = self.fill_sim.simulate_fill(signal, "open")
        if fill is None:
            return False, "Order did not fill"

        self.fills.append(fill)

        # Create position
        entry_credit = fill.price  # after slippage
        margin = fill.contracts * self.config.margin_per_spread
        exp_date = (datetime.fromisoformat(signal.timestamp.replace("Z", "+00:00"))
                    + timedelta(days=signal.dte)).date().isoformat() if signal.timestamp else ""

        pos = Position(
            position_id=f"POS-{uuid.uuid4().hex[:8]}",
            strategy=signal.strategy,
            ticker=signal.ticker,
            direction=signal.direction,
            spread_type=signal.spread_type,
            contracts=fill.contracts,
            entry_credit=entry_credit,
            max_loss=signal.max_loss,
            spread_width=signal.spread_width,
            margin_required=margin,
            entry_date=signal.timestamp[:10] if signal.timestamp else "",
            expiration_date=exp_date,
            stop_loss_level=entry_credit * signal.stop_loss_pct,
            profit_target_level=entry_credit * signal.profit_target_pct,
            regime_at_entry=signal.regime,
            confidence=signal.confidence,
        )
        self.positions.append(pos)
        self.capital -= fill.commission  # commissions deducted immediately
        self._persist_position(pos)

        logger.info("Opened %s: %s %d @ %.2f (%s)",
                     pos.position_id, signal.ticker, fill.contracts, entry_credit, signal.strategy)
        return True, pos.position_id

    # ── Daily step ──────────────────────────────────────────────────────

    def step(
        self,
        current_date: str,
        market_prices: Optional[Dict[str, float]] = None,
        days_elapsed: int = 1,
    ) -> DailyPnL:
        """Advance one day: revalue positions, check exits, record P&L."""
        market_prices = market_prices or {}
        daily_realised = 0.0
        to_close: List[Tuple[Position, str]] = []

        for pos in self.positions:
            # Estimate current spread value (time decay model)
            days_held = self._days_between(pos.entry_date, current_date)
            total_dte = self._days_between(pos.entry_date, pos.expiration_date)
            remaining_dte = max(total_dte - days_held, 0)

            # Theta decay: credit decays toward zero as expiration approaches
            if total_dte > 0:
                decay_frac = days_held / total_dte
                decay_frac = min(decay_frac, 1.0)
                # Accelerating decay (sqrt model)
                decayed_value = pos.entry_credit * (1 - math.sqrt(decay_frac))
            else:
                decayed_value = 0.0

            pos.current_value = max(decayed_value, 0.0)
            pos.unrealised_pnl = (pos.entry_credit - pos.current_value) * pos.contracts * 100

            # Exit checks
            if remaining_dte <= 0:
                to_close.append((pos, "expiration"))
            elif pos.current_value <= pos.profit_target_level:
                to_close.append((pos, "profit_target"))
            elif pos.unrealised_pnl < -(pos.stop_loss_level * pos.contracts * 100):
                to_close.append((pos, "stop_loss"))

        # Close positions
        for pos, reason in to_close:
            trade = self._close_position(pos, current_date, reason)
            daily_realised += trade.pnl

        # Daily P&L
        unrealised = sum(p.unrealised_pnl for p in self.positions)
        total = self.realised_pnl + unrealised
        margin_used = sum(p.margin_required for p in self.positions)

        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        dd = (self.capital + unrealised - self.peak_capital) / self.peak_capital if self.peak_capital > 0 else 0

        # Per-strategy attribution
        by_strat: Dict[str, float] = {}
        for pos in self.positions:
            by_strat[pos.strategy] = by_strat.get(pos.strategy, 0) + pos.unrealised_pnl
        for trade in self.closed_trades:
            by_strat[trade.strategy] = by_strat.get(trade.strategy, 0) + trade.pnl

        snap = DailyPnL(
            date=current_date, total_pnl=total,
            realised_pnl=self.realised_pnl, unrealised_pnl=unrealised,
            n_positions=len(self.positions), margin_used=margin_used,
            capital=self.capital, drawdown=dd, by_strategy=by_strat,
        )
        self.daily_pnl.append(snap)

        # Daily loss check
        self.risk_monitor.check_daily_loss(daily_realised)

        return snap

    def _close_position(self, pos: Position, exit_date: str, reason: str) -> ClosedTrade:
        """Close a position and record the trade."""
        # Simulate exit fill
        exit_signal = Signal(
            strategy=pos.strategy, ticker=pos.ticker,
            contracts=pos.contracts, net_credit=pos.current_value,
        )
        exit_fill = self.fill_sim.simulate_fill(exit_signal, "close")
        if exit_fill:
            exit_debit = exit_fill.price
            exit_slip = exit_fill.slippage
            exit_comm = exit_fill.commission
            self.fills.append(exit_fill)
        else:
            exit_debit = pos.current_value
            exit_slip = 0
            exit_comm = 0

        # PnL: (credit received - debit paid) × contracts × 100 - costs
        pnl = (pos.entry_credit - exit_debit) * pos.contracts * 100 - exit_slip - exit_comm
        multiplier = pos.entry_credit * pos.contracts * 100
        return_pct = pnl / multiplier if multiplier > 0 else 0

        hold_days = self._days_between(pos.entry_date, exit_date)

        trade = ClosedTrade(
            position_id=pos.position_id, strategy=pos.strategy,
            ticker=pos.ticker, direction=pos.direction,
            spread_type=pos.spread_type, contracts=pos.contracts,
            entry_credit=pos.entry_credit, exit_debit=exit_debit,
            pnl=pnl, return_pct=return_pct,
            slippage_total=exit_slip, commission_total=exit_comm,
            entry_date=pos.entry_date, exit_date=exit_date,
            exit_reason=reason, hold_days=hold_days,
            regime_at_entry=pos.regime_at_entry,
            confidence=pos.confidence, win=pnl > 0,
        )
        self.closed_trades.append(trade)
        self.realised_pnl += pnl
        self.capital += pnl
        self.positions.remove(pos)
        self._persist_trade(trade)

        return trade

    # ── Bulk replay ─────────────────────────────────────────────────────

    def replay(self, trades_df: pd.DataFrame) -> PerformanceSummary:
        """Replay historical trades through the engine.

        DataFrame must have: strategy, entry_date, exit_date, net_credit,
        max_loss, spread_width, contracts, pnl, win, regime, exit_reason.
        """
        trades_df = trades_df.sort_values("entry_date").copy()

        for _, row in trades_df.iterrows():
            sig = Signal(
                strategy=str(row.get("strategy_type", row.get("strategy", "default"))),
                ticker=str(row.get("ticker", "SPY")),
                direction="short",
                spread_type=str(row.get("spread_type", "bull_put")),
                contracts=int(row.get("contracts", 1)),
                net_credit=abs(float(row.get("net_credit", 1.0))),
                max_loss=abs(float(row.get("max_loss_per_unit", row.get("max_loss", 4.0)))),
                spread_width=float(row.get("spread_width", 5.0)),
                dte=int(row.get("dte_at_entry", row.get("dte", 30))),
                confidence=float(row.get("confidence", 0.6)),
                regime=str(row.get("regime", "neutral")),
                timestamp=str(row.get("entry_date", "")),
            )
            self.submit_signal(sig)

            # Step to exit date
            exit_date = str(row.get("exit_date", ""))
            if exit_date and self.positions:
                self.step(exit_date[:10])

        return self.get_performance()

    # ── Performance summary ─────────────────────────────────────────────

    def get_performance(self) -> PerformanceSummary:
        """Compute aggregate performance metrics."""
        trades = self.closed_trades
        if not trades:
            return PerformanceSummary(
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {},
            )

        pnls = np.array([t.pnl for t in trades])
        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]

        total_pnl = float(pnls.sum())
        n = len(trades)
        n_wins = len(wins)
        wr = n_wins / n if n > 0 else 0
        avg_pnl = float(pnls.mean())
        avg_winner = float(np.mean([t.pnl for t in wins])) if wins else 0
        avg_loser = float(np.mean([t.pnl for t in losses])) if losses else 0
        avg_hold = float(np.mean([t.hold_days for t in trades]))
        total_slip = sum(t.slippage_total for t in trades)
        total_comm = sum(t.commission_total for t in trades)

        # Equity curve for Sharpe/DD
        equity = self.config.starting_capital + np.cumsum(pnls)
        equity = np.concatenate([[self.config.starting_capital], equity])
        daily_ret = np.diff(equity) / equity[:-1]
        sh = float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)) if np.std(daily_ret) > 0 else 0

        # Sortino
        downside = daily_ret[daily_ret < 0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else 1
        sortino = float(np.mean(daily_ret) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Max DD
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.where(peak > 0, peak, 1)
        mdd = float(np.min(dd))

        # Calmar
        ann_ret = total_pnl / self.config.starting_capital  # simplified
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0

        # Profit factor
        gross_wins = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))
        pf = gross_wins / gross_losses if gross_losses > 0 else (99.9 if gross_wins > 0 else 0)

        # Per-strategy
        by_strat: Dict[str, Dict[str, float]] = {}
        for t in trades:
            s = t.strategy
            if s not in by_strat:
                by_strat[s] = {"pnl": 0, "n": 0, "wins": 0}
            by_strat[s]["pnl"] += t.pnl
            by_strat[s]["n"] += 1
            by_strat[s]["wins"] += int(t.win)
        for s in by_strat:
            by_strat[s]["win_rate"] = by_strat[s]["wins"] / by_strat[s]["n"] if by_strat[s]["n"] > 0 else 0

        # Per-regime
        by_regime: Dict[str, Dict[str, float]] = {}
        for t in trades:
            r = t.regime_at_entry
            if r not in by_regime:
                by_regime[r] = {"pnl": 0, "n": 0, "wins": 0}
            by_regime[r]["pnl"] += t.pnl
            by_regime[r]["n"] += 1
            by_regime[r]["wins"] += int(t.win)
        for r in by_regime:
            by_regime[r]["win_rate"] = by_regime[r]["wins"] / by_regime[r]["n"] if by_regime[r]["n"] > 0 else 0

        return PerformanceSummary(
            total_pnl=total_pnl,
            total_return_pct=total_pnl / self.config.starting_capital,
            win_rate=wr, n_trades=n, n_wins=n_wins,
            sharpe=sh, sortino=sortino, max_drawdown=mdd,
            calmar=calmar, profit_factor=min(pf, 99.9),
            avg_pnl=avg_pnl, avg_winner=avg_winner, avg_loser=avg_loser,
            avg_hold_days=avg_hold,
            total_slippage=total_slip, total_commission=total_comm,
            by_strategy=by_strat, by_regime=by_regime,
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _days_between(d1: str, d2: str) -> int:
        try:
            a = datetime.fromisoformat(d1[:10])
            b = datetime.fromisoformat(d2[:10])
            return abs((b - a).days)
        except (ValueError, TypeError):
            return 0

    # ── Persistence ─────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            position_id TEXT PRIMARY KEY, strategy TEXT, ticker TEXT,
            contracts INTEGER, entry_credit REAL, entry_date TEXT, status TEXT DEFAULT 'open'
        );
        CREATE TABLE IF NOT EXISTS trades (
            position_id TEXT PRIMARY KEY, strategy TEXT, ticker TEXT,
            contracts INTEGER, pnl REAL, entry_date TEXT, exit_date TEXT,
            exit_reason TEXT, regime TEXT, win INTEGER
        );
        CREATE TABLE IF NOT EXISTS daily_pnl (
            date TEXT PRIMARY KEY, total_pnl REAL, capital REAL,
            n_positions INTEGER, drawdown REAL
        );
        """)
        self._conn.commit()

    def _persist_position(self, pos: Position) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?)",
            (pos.position_id, pos.strategy, pos.ticker, pos.contracts,
             pos.entry_credit, pos.entry_date, "open"),
        )
        self._conn.commit()

    def _persist_trade(self, trade: ClosedTrade) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)",
            (trade.position_id, trade.strategy, trade.ticker, trade.contracts,
             trade.pnl, trade.entry_date, trade.exit_date, trade.exit_reason,
             trade.regime_at_entry, int(trade.win)),
        )
        self._conn.execute(
            "UPDATE positions SET status='closed' WHERE position_id=?",
            (trade.position_id,),
        )
        self._conn.commit()

    # ── JSON export ─────────────────────────────────────────────────────

    def export_json(self, path: Optional[str] = None) -> Dict[str, Any]:
        perf = self.get_performance()
        data = {
            "generated": datetime.now().isoformat(),
            "config": {
                "starting_capital": self.config.starting_capital,
                "slippage": self.config.slippage_per_contract,
                "fill_rate": self.config.fill_rate,
                "max_dd": self.config.max_drawdown_pct,
            },
            "performance": {
                "total_pnl": perf.total_pnl,
                "total_return_pct": perf.total_return_pct,
                "win_rate": perf.win_rate,
                "n_trades": perf.n_trades,
                "sharpe": perf.sharpe,
                "max_drawdown": perf.max_drawdown,
                "profit_factor": perf.profit_factor,
                "total_slippage": perf.total_slippage,
                "total_commission": perf.total_commission,
            },
            "by_strategy": perf.by_strategy,
            "by_regime": perf.by_regime,
            "risk_breaches": len(self.risk_monitor.breaches),
        }
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        return data

    # ── HTML report ─────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        perf = self.get_performance()
        charts = self._render_charts(perf)
        html = self._build_html(perf, charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return str(out.resolve())

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self, perf: PerformanceSummary) -> Dict[str, str]:
        return {
            "equity": self._chart_equity(),
            "strategy": self._chart_strategy(perf),
            "regime": self._chart_regime(perf),
        }

    def _chart_equity(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.closed_trades:
            return ""
        pnls = [t.pnl for t in self.closed_trades]
        equity = self.config.starting_capital + np.cumsum(pnls)
        equity = np.concatenate([[self.config.starting_capital], equity])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(equity)), equity, color="#3b82f6", lw=1.2)
        ax.fill_between(range(len(equity)), equity, self.config.starting_capital, alpha=0.1, color="#3b82f6")
        ax.axhline(self.config.starting_capital, color="#64748b", lw=0.5, ls="--")
        ax.set_xlabel("Trade #"); ax.set_ylabel("Equity ($)")
        ax.set_title("Equity Curve", fontsize=11); ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_strategy(self, perf: PerformanceSummary) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not perf.by_strategy:
            return ""
        names = list(perf.by_strategy.keys())
        pnls = [perf.by_strategy[s]["pnl"] for s in names]
        colors = ["#16a34a" if p > 0 else "#dc2626" for p in pnls]
        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.5)))
        ax.barh(names, pnls, color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("P&L ($)"); ax.set_title("P&L by Strategy", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_regime(self, perf: PerformanceSummary) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not perf.by_regime:
            return ""
        names = list(perf.by_regime.keys())
        pnls = [perf.by_regime[r]["pnl"] for r in names]
        regime_colors = {"bull": "#16a34a", "bear": "#dc2626", "high_vol": "#f59e0b",
                         "crash": "#7f1d1d", "neutral": "#64748b", "low_vol": "#3b82f6"}
        colors = [regime_colors.get(n, "#3b82f6") for n in names]
        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.5)))
        ax.barh(names, pnls, color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("P&L ($)"); ax.set_title("P&L by Regime", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, perf: PerformanceSummary, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        pnl_cls = "good" if perf.total_pnl > 0 else "bad"

        strat_rows = ""
        for s, d in sorted(perf.by_strategy.items(), key=lambda x: -x[1]["pnl"]):
            cls = "good" if d["pnl"] > 0 else "bad"
            strat_rows += (f'<tr><td style="text-align:left">{s}</td><td>{d["n"]:.0f}</td>'
                          f'<td>{d.get("win_rate",0):.0%}</td>'
                          f'<td class="{cls}">${d["pnl"]:+,.0f}</td></tr>\n')

        regime_rows = ""
        for r, d in sorted(perf.by_regime.items(), key=lambda x: -x[1]["pnl"]):
            cls = "good" if d["pnl"] > 0 else "bad"
            regime_rows += (f'<tr><td style="text-align:left">{r}</td><td>{d["n"]:.0f}</td>'
                           f'<td>{d.get("win_rate",0):.0%}</td>'
                           f'<td class="{cls}">${d["pnl"]:+,.0f}</td></tr>\n')

        breach_rows = ""
        for b in self.risk_monitor.breaches[-10:]:
            breach_rows += f'<tr><td>{b.timestamp[:10]}</td><td>{b.breach_type}</td><td>{b.limit}</td><td>{b.actual:.4f}</td><td>{b.action_taken}</td></tr>\n'
        if not breach_rows:
            breach_rows = '<tr><td colspan="5" style="text-align:center;color:#64748b">No breaches</td></tr>'

        recent_rows = ""
        for t in self.closed_trades[-20:]:
            cls = "good" if t.win else "bad"
            recent_rows += (f'<tr><td>{t.entry_date[:10]}</td><td>{t.strategy}</td>'
                           f'<td>{t.contracts}</td><td class="{cls}">${t.pnl:+,.0f}</td>'
                           f'<td>{t.exit_reason}</td><td>{t.hold_days}d</td></tr>\n')

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Paper Trading Engine Report</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }} h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }}
  .kpi-row {{ display:flex; gap:1.2em; flex-wrap:wrap; margin:1.5em 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em 1.5em; min-width:120px; flex:1; text-align:center; }}
  .kpi .value {{ font-size:1.5em; font-weight:700; }} .kpi .label {{ font-size:0.75em; color:#64748b; margin-top:0.2em; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:0.88em; }}
  th {{ background:#f1f5f9; padding:8px 10px; text-align:left; border-bottom:2px solid #cbd5e1; font-weight:600; }}
  td {{ padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:right; }} td:first-child {{ text-align:left; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em; margin:1.5em 0; text-align:center; }}
  .chart img {{ max-width:100%; height:auto; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.8em; color:#94a3b8; }}
</style></head><body>
<h1>Paper Trading Engine Report</h1>
<div class="meta">{perf.n_trades} trades &middot; Slippage ${self.config.slippage_per_contract}/contract &middot; Fill rate {self.config.fill_rate:.0%} &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value {pnl_cls}">${perf.total_pnl:+,.0f}</div><div class="label">Total P&L</div></div>
  <div class="kpi"><div class="value">{perf.win_rate:.0%}</div><div class="label">Win Rate</div></div>
  <div class="kpi"><div class="value">{perf.sharpe:.2f}</div><div class="label">Sharpe</div></div>
  <div class="kpi"><div class="value">{perf.max_drawdown:.1%}</div><div class="label">Max DD</div></div>
  <div class="kpi"><div class="value">{perf.profit_factor:.2f}</div><div class="label">Profit Factor</div></div>
  <div class="kpi"><div class="value bad">${perf.total_slippage:,.0f}</div><div class="label">Total Slippage</div></div>
</div>
<h2>1. Equity Curve</h2>{_img("equity")}
<h2>2. Strategy Attribution</h2>{_img("strategy")}
<table><thead><tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>P&L</th></tr></thead><tbody>{strat_rows}</tbody></table>
<h2>3. Regime Attribution</h2>{_img("regime")}
<table><thead><tr><th>Regime</th><th>Trades</th><th>Win Rate</th><th>P&L</th></tr></thead><tbody>{regime_rows}</tbody></table>
<h2>4. Execution Costs</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
<tr><td style="text-align:left">Total Slippage</td><td>${perf.total_slippage:,.2f}</td></tr>
<tr><td style="text-align:left">Total Commission</td><td>${perf.total_commission:,.2f}</td></tr>
<tr><td style="text-align:left">Cost as % of P&L</td><td>{(perf.total_slippage+perf.total_commission)/abs(perf.total_pnl)*100 if perf.total_pnl else 0:.1f}%</td></tr>
<tr><td style="text-align:left">Avg Hold Days</td><td>{perf.avg_hold_days:.1f}</td></tr>
</tbody></table>
<h2>5. Risk Breaches</h2>
<table><thead><tr><th>Date</th><th>Type</th><th>Limit</th><th>Actual</th><th>Action</th></tr></thead><tbody>{breach_rows}</tbody></table>
<h2>6. Recent Trades</h2>
<table><thead><tr><th>Date</th><th>Strategy</th><th>Contracts</th><th>P&L</th><th>Exit</th><th>Hold</th></tr></thead><tbody>{recent_rows}</tbody></table>
<footer>Generated by <code>compass/paper_trading_engine.py</code></footer>
</body></html>"""
        return html
