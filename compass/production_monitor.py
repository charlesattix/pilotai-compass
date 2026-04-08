"""
Real-time production monitoring dashboard for credit spread strategies.

Tracks:
  1. Open positions — per-strategy breakdown with Greeks
  2. Unrealized P&L — mark-to-market across all positions
  3. Daily P&L — realized + unrealized change since market open
  4. Max drawdown — peak-to-trough equity decline
  5. Strategy-level attribution — P&L, win rate, Sharpe per strategy
  6. Risk budget usage — capital at risk vs allocated budget
  7. VaR utilization — current VaR as fraction of VaR limit

Generates a self-contained auto-refreshing HTML dashboard.
Telegram alerts for: DD breach, daily loss limit, correlation spike,
position limit breach.

Usage::

    from compass.production_monitor import ProductionMonitorDashboard
    monitor = ProductionMonitorDashboard(config=MonitorDashboardConfig())
    monitor.record_position(pos)
    monitor.update_pnl(daily=500, unrealized=1200)
    snap = monitor.snapshot()
    monitor.generate_dashboard(snap, Path("reports/production_dashboard.html"))
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "production_dashboard.html"
DEFAULT_STATE_PATH = ROOT / "reports" / "production_monitor_state.json"


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class MonitorDashboardConfig:
    """Configuration for production monitoring thresholds."""

    # Capital
    initial_capital: float = 100_000.0

    # Drawdown
    dd_warning_pct: float = 0.05
    dd_critical_pct: float = 0.10

    # Daily loss
    daily_loss_warning: float = 2_000.0
    daily_loss_critical: float = 5_000.0

    # Position limits
    max_positions: int = 20
    max_positions_per_strategy: int = 8

    # Risk budget
    max_capital_at_risk_pct: float = 0.40
    risk_budget_warning_pct: float = 0.75  # 75% of budget = warning

    # VaR
    var_limit: float = 10_000.0
    var_warning_pct: float = 0.70
    var_critical_pct: float = 0.90

    # Correlation
    correlation_spike_threshold: float = 0.80  # pairwise correlation

    # Telegram
    telegram_enabled: bool = True
    alert_cooldown_seconds: float = 300.0

    # Dashboard
    auto_refresh_seconds: int = 30


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class PositionRecord:
    """A single open position."""

    position_id: str
    strategy: str
    ticker: str
    direction: str            # bull_put / bear_call
    contracts: int
    entry_credit: float
    max_loss: float
    current_value: float
    unrealized_pnl: float
    entry_date: str
    expiration_date: str
    dte: int
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    margin_required: float = 0.0


@dataclass
class StrategyAttribution:
    """Strategy-level P&L attribution."""

    strategy: str
    n_positions: int = 0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    capital_at_risk: float = 0.0
    sharpe: float = 0.0
    avg_return: float = 0.0
    daily_returns: List[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    @property
    def n_trades(self) -> int:
        return self.win_count + self.loss_count


@dataclass
class RiskBudgetState:
    """Risk budget utilization snapshot."""

    total_capital: float = 0.0
    capital_at_risk: float = 0.0
    max_allowed_risk: float = 0.0
    utilization_pct: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0

    @property
    def headroom(self) -> float:
        return self.max_allowed_risk - self.capital_at_risk


@dataclass
class VaRState:
    """VaR utilization snapshot."""

    current_var_95: float = 0.0
    current_var_99: float = 0.0
    var_limit: float = 0.0
    utilization_pct: float = 0.0


@dataclass
class DrawdownState:
    """Drawdown tracking."""

    current_equity: float = 0.0
    peak_equity: float = 0.0
    drawdown_dollar: float = 0.0
    drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_dollar: float = 0.0
    drawdown_start_date: Optional[str] = None
    days_in_drawdown: int = 0


@dataclass
class CorrelationSnapshot:
    """Strategy correlation matrix snapshot."""

    strategy_pairs: Dict[str, float] = field(default_factory=dict)
    max_correlation: float = 0.0
    max_pair: str = ""
    avg_correlation: float = 0.0


@dataclass
class TelegramAlert:
    """Alert queued for Telegram delivery."""

    timestamp: str
    severity: str  # INFO, WARNING, CRITICAL
    category: str  # dd_breach, daily_loss, correlation_spike, position_limit
    title: str
    message: str
    value: float = 0.0
    threshold: float = 0.0
    delivered: bool = False

    def telegram_format(self) -> str:
        emoji = {"INFO": "\u2139\ufe0f", "WARNING": "\u26a0\ufe0f", "CRITICAL": "\U0001f6a8"}.get(
            self.severity, "\U0001f4ca"
        )
        return (
            f"{emoji} *{self.severity}*: {self.title}\n"
            f"{self.message}\n"
            f"Value: `{self.value:.4f}` | Threshold: `{self.threshold:.4f}`\n"
            f"Time: {self.timestamp}"
        )


@dataclass
class DashboardSnapshot:
    """Complete point-in-time dashboard snapshot."""

    timestamp: str
    positions: List[PositionRecord]
    strategy_attribution: Dict[str, StrategyAttribution]
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    drawdown: DrawdownState
    risk_budget: RiskBudgetState
    var_state: VaRState
    correlation: CorrelationSnapshot
    alerts: List[TelegramAlert]
    alert_history: List[TelegramAlert]
    n_open_positions: int
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_theta: float
    portfolio_vega: float
    health_score: float


# ── Alert engine ─────────────────────────────────────────────────────────


class DashboardAlertEngine:
    """Alert engine with Telegram integration and cooldowns."""

    def __init__(
        self,
        config: MonitorDashboardConfig,
        send_fn: Optional[Callable[[str], bool]] = None,
    ):
        self.config = config
        self._send_fn = send_fn or self._default_send
        self._last_alert: Dict[str, float] = {}
        self._history: List[TelegramAlert] = []

    @staticmethod
    def _default_send(text: str) -> bool:
        try:
            from shared.telegram_alerts import send_message
            return send_message(text)
        except (ImportError, Exception):
            logger.info("TELEGRAM (not configured): %s", text[:200])
            return False

    def _can_send(self, category: str) -> bool:
        now = time.time()
        last = self._last_alert.get(category, 0)
        if now - last < self.config.alert_cooldown_seconds:
            return False
        return True

    def _record(self, alert: TelegramAlert) -> None:
        self._last_alert[alert.category] = time.time()
        self._history.append(alert)

    def send(self, alert: TelegramAlert) -> bool:
        if not self._can_send(alert.category):
            alert.delivered = False
            self._history.append(alert)
            return False
        if self.config.telegram_enabled:
            alert.delivered = self._send_fn(alert.telegram_format())
        else:
            logger.info("ALERT [disabled]: %s", alert.title)
            alert.delivered = False
        self._record(alert)
        return alert.delivered

    def check_drawdown(self, dd: DrawdownState) -> Optional[TelegramAlert]:
        if dd.drawdown_pct >= self.config.dd_critical_pct:
            return TelegramAlert(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                severity="CRITICAL",
                category="dd_breach",
                title=f"DRAWDOWN CRITICAL: {dd.drawdown_pct:.1%}",
                message=(
                    f"Current DD: {dd.drawdown_pct:.1%} (${dd.drawdown_dollar:,.0f})\n"
                    f"Max DD: {dd.max_drawdown_pct:.1%}\n"
                    f"Days in DD: {dd.days_in_drawdown}"
                ),
                value=dd.drawdown_pct,
                threshold=self.config.dd_critical_pct,
            )
        if dd.drawdown_pct >= self.config.dd_warning_pct:
            return TelegramAlert(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                severity="WARNING",
                category="dd_breach",
                title=f"Drawdown Warning: {dd.drawdown_pct:.1%}",
                message=f"Current DD: {dd.drawdown_pct:.1%} (${dd.drawdown_dollar:,.0f})",
                value=dd.drawdown_pct,
                threshold=self.config.dd_warning_pct,
            )
        return None

    def check_daily_loss(self, daily_pnl: float) -> Optional[TelegramAlert]:
        loss = -daily_pnl
        if loss >= self.config.daily_loss_critical:
            return TelegramAlert(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                severity="CRITICAL",
                category="daily_loss",
                title=f"DAILY LOSS LIMIT: ${daily_pnl:+,.0f}",
                message=f"Daily loss ${loss:,.0f} exceeds critical limit ${self.config.daily_loss_critical:,.0f}",
                value=loss,
                threshold=self.config.daily_loss_critical,
            )
        if loss >= self.config.daily_loss_warning:
            return TelegramAlert(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                severity="WARNING",
                category="daily_loss",
                title=f"Daily Loss Warning: ${daily_pnl:+,.0f}",
                message=f"Daily loss ${loss:,.0f} nearing limit ${self.config.daily_loss_critical:,.0f}",
                value=loss,
                threshold=self.config.daily_loss_warning,
            )
        return None

    def check_correlation(self, corr: CorrelationSnapshot) -> Optional[TelegramAlert]:
        if corr.max_correlation >= self.config.correlation_spike_threshold:
            return TelegramAlert(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                severity="WARNING",
                category="correlation_spike",
                title=f"Correlation Spike: {corr.max_correlation:.2f}",
                message=(
                    f"Pair: {corr.max_pair}\n"
                    f"Correlation: {corr.max_correlation:.3f}\n"
                    f"Avg portfolio correlation: {corr.avg_correlation:.3f}"
                ),
                value=corr.max_correlation,
                threshold=self.config.correlation_spike_threshold,
            )
        return None

    def check_position_limit(
        self, n_positions: int, per_strategy: Dict[str, int],
    ) -> Optional[TelegramAlert]:
        if n_positions >= self.config.max_positions:
            return TelegramAlert(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                severity="CRITICAL",
                category="position_limit",
                title=f"POSITION LIMIT REACHED: {n_positions}/{self.config.max_positions}",
                message=f"No new positions can be opened",
                value=float(n_positions),
                threshold=float(self.config.max_positions),
            )
        for strat, count in per_strategy.items():
            if count >= self.config.max_positions_per_strategy:
                return TelegramAlert(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    severity="WARNING",
                    category="position_limit",
                    title=f"Strategy Position Limit: {strat} ({count}/{self.config.max_positions_per_strategy})",
                    message=f"Strategy {strat} at position limit",
                    value=float(count),
                    threshold=float(self.config.max_positions_per_strategy),
                )
        return None

    def run_all_checks(
        self,
        dd: DrawdownState,
        daily_pnl: float,
        corr: CorrelationSnapshot,
        n_positions: int,
        per_strategy: Dict[str, int],
    ) -> List[TelegramAlert]:
        new_alerts: List[TelegramAlert] = []
        for check_fn, args in [
            (self.check_drawdown, (dd,)),
            (self.check_daily_loss, (daily_pnl,)),
            (self.check_correlation, (corr,)),
            (self.check_position_limit, (n_positions, per_strategy)),
        ]:
            alert = check_fn(*args)
            if alert:
                self.send(alert)
                new_alerts.append(alert)
        return new_alerts

    @property
    def history(self) -> List[TelegramAlert]:
        return list(self._history)

    def reset_cooldowns(self) -> None:
        self._last_alert.clear()


# ── Core monitor ─────────────────────────────────────────────────────────


class ProductionMonitorDashboard:
    """Real-time production monitoring dashboard.

    Tracks positions, P&L, drawdown, strategy attribution, risk budget,
    VaR utilization, and fires Telegram alerts on threshold breaches.
    """

    def __init__(
        self,
        config: Optional[MonitorDashboardConfig] = None,
        send_fn: Optional[Callable[[str], bool]] = None,
    ):
        self.config = config or MonitorDashboardConfig()
        self.alert_engine = DashboardAlertEngine(self.config, send_fn)

        # State
        self._positions: Dict[str, PositionRecord] = {}
        self._strategies: Dict[str, StrategyAttribution] = {}
        self._daily_pnl: float = 0.0
        self._realized_pnl: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._peak_equity: float = self.config.initial_capital
        self._max_dd_pct: float = 0.0
        self._max_dd_dollar: float = 0.0
        self._dd_start_date: Optional[str] = None
        self._dd_days: int = 0
        self._var_95: float = 0.0
        self._var_99: float = 0.0
        self._correlation = CorrelationSnapshot()
        self._pnl_history: List[float] = []
        self._equity_history: List[Tuple[str, float]] = []

    # ── Position management ───────────────────────────────────────────

    def record_position(self, pos: PositionRecord) -> None:
        self._positions[pos.position_id] = pos
        if pos.strategy not in self._strategies:
            self._strategies[pos.strategy] = StrategyAttribution(strategy=pos.strategy)
        self._recalc_unrealized()

    def remove_position(
        self,
        position_id: str,
        realized_pnl: float = 0.0,
        won: bool = False,
    ) -> None:
        pos = self._positions.pop(position_id, None)
        if pos is None:
            return
        self._realized_pnl += realized_pnl
        self._daily_pnl += realized_pnl
        strat = self._strategies.get(pos.strategy)
        if strat:
            strat.realized_pnl += realized_pnl
            strat.total_pnl += realized_pnl
            if won:
                strat.win_count += 1
            else:
                strat.loss_count += 1
            if realized_pnl != 0:
                strat.daily_returns.append(realized_pnl)
        self._recalc_unrealized()

    def _recalc_unrealized(self) -> None:
        self._unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        for strat_name, strat in self._strategies.items():
            strat.unrealized_pnl = sum(
                p.unrealized_pnl for p in self._positions.values()
                if p.strategy == strat_name
            )
            strat.n_positions = sum(
                1 for p in self._positions.values() if p.strategy == strat_name
            )
            strat.capital_at_risk = sum(
                p.max_loss for p in self._positions.values()
                if p.strategy == strat_name
            )

    # ── P&L tracking ─────────────────────────────────────────────────

    def update_pnl(self, daily: float, unrealized: float) -> None:
        self._daily_pnl = daily
        self._unrealized_pnl = unrealized

    def record_daily_return(self, pnl: float) -> None:
        self._pnl_history.append(pnl)

    # ── Drawdown tracking ────────────────────────────────────────────

    def _compute_drawdown(self) -> DrawdownState:
        total_pnl = self._realized_pnl + self._unrealized_pnl
        equity = self.config.initial_capital + total_pnl
        self._peak_equity = max(self._peak_equity, equity)

        dd_dollar = self._peak_equity - equity
        dd_pct = dd_dollar / self._peak_equity if self._peak_equity > 0 else 0.0

        if dd_pct > self._max_dd_pct:
            self._max_dd_pct = dd_pct
            self._max_dd_dollar = dd_dollar

        if dd_pct > 0 and self._dd_start_date is None:
            self._dd_start_date = datetime.now().strftime("%Y-%m-%d")
            self._dd_days = 0
        elif dd_pct <= 0:
            self._dd_start_date = None
            self._dd_days = 0

        return DrawdownState(
            current_equity=equity,
            peak_equity=self._peak_equity,
            drawdown_dollar=dd_dollar,
            drawdown_pct=dd_pct,
            max_drawdown_pct=self._max_dd_pct,
            max_drawdown_dollar=self._max_dd_dollar,
            drawdown_start_date=self._dd_start_date,
            days_in_drawdown=self._dd_days,
        )

    # ── Risk budget ──────────────────────────────────────────────────

    def _compute_risk_budget(self) -> RiskBudgetState:
        capital = self.config.initial_capital
        max_risk = capital * self.config.max_capital_at_risk_pct
        car = sum(p.max_loss for p in self._positions.values())
        margin = sum(p.margin_required for p in self._positions.values())
        util = car / max_risk if max_risk > 0 else 0.0

        return RiskBudgetState(
            total_capital=capital,
            capital_at_risk=car,
            max_allowed_risk=max_risk,
            utilization_pct=util,
            margin_used=margin,
            margin_available=capital - margin,
        )

    # ── VaR ──────────────────────────────────────────────────────────

    def update_var(self, var_95: float, var_99: float) -> None:
        self._var_95 = var_95
        self._var_99 = var_99

    def _compute_var_state(self) -> VaRState:
        util = self._var_95 / self.config.var_limit if self.config.var_limit > 0 else 0.0
        return VaRState(
            current_var_95=self._var_95,
            current_var_99=self._var_99,
            var_limit=self.config.var_limit,
            utilization_pct=util,
        )

    # ── Correlation ──────────────────────────────────────────────────

    def update_correlation(self, pairs: Dict[str, float]) -> None:
        if not pairs:
            self._correlation = CorrelationSnapshot()
            return
        max_pair = max(pairs, key=lambda k: abs(pairs[k]))
        avg_corr = sum(abs(v) for v in pairs.values()) / len(pairs) if pairs else 0.0
        self._correlation = CorrelationSnapshot(
            strategy_pairs=dict(pairs),
            max_correlation=abs(pairs[max_pair]),
            max_pair=max_pair,
            avg_correlation=avg_corr,
        )

    # ── Portfolio Greeks ─────────────────────────────────────────────

    def _portfolio_greeks(self) -> Tuple[float, float, float, float]:
        d = sum(p.delta * p.contracts for p in self._positions.values())
        g = sum(p.gamma * p.contracts for p in self._positions.values())
        t = sum(p.theta * p.contracts for p in self._positions.values())
        v = sum(p.vega * p.contracts for p in self._positions.values())
        return d, g, t, v

    # ── Health score ─────────────────────────────────────────────────

    def _health_score(self, dd: DrawdownState, rb: RiskBudgetState, vs: VaRState) -> float:
        """Compute 0-100 health score from key risk metrics."""
        scores: List[float] = []

        # DD (25 pts)
        dd_ratio = min(dd.drawdown_pct / self.config.dd_critical_pct, 1.0) if self.config.dd_critical_pct > 0 else 0
        scores.append(25.0 * (1.0 - dd_ratio))

        # Risk budget (25 pts)
        rb_ratio = min(rb.utilization_pct, 1.0)
        scores.append(25.0 * (1.0 - rb_ratio))

        # VaR (25 pts)
        var_ratio = min(vs.utilization_pct, 1.0)
        scores.append(25.0 * (1.0 - var_ratio))

        # Daily P&L (25 pts)
        if self._daily_pnl >= 0:
            scores.append(25.0)
        else:
            loss_ratio = min(
                abs(self._daily_pnl) / self.config.daily_loss_critical, 1.0
            ) if self.config.daily_loss_critical > 0 else 0
            scores.append(25.0 * (1.0 - loss_ratio))

        return min(100.0, max(0.0, sum(scores)))

    # ── Snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> DashboardSnapshot:
        dd = self._compute_drawdown()
        rb = self._compute_risk_budget()
        vs = self._compute_var_state()
        pd_, pg, pt, pv = self._portfolio_greeks()

        per_strat_counts = {}
        for p in self._positions.values():
            per_strat_counts[p.strategy] = per_strat_counts.get(p.strategy, 0) + 1

        new_alerts = self.alert_engine.run_all_checks(
            dd=dd,
            daily_pnl=self._daily_pnl,
            corr=self._correlation,
            n_positions=len(self._positions),
            per_strategy=per_strat_counts,
        )

        health = self._health_score(dd, rb, vs)
        total_pnl = self._realized_pnl + self._unrealized_pnl

        # Track equity
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._equity_history.append((ts, dd.current_equity))
        if len(self._equity_history) > 500:
            self._equity_history = self._equity_history[-500:]

        return DashboardSnapshot(
            timestamp=ts,
            positions=list(self._positions.values()),
            strategy_attribution=dict(self._strategies),
            daily_pnl=self._daily_pnl,
            unrealized_pnl=self._unrealized_pnl,
            realized_pnl=self._realized_pnl,
            total_pnl=total_pnl,
            drawdown=dd,
            risk_budget=rb,
            var_state=vs,
            correlation=self._correlation,
            alerts=new_alerts,
            alert_history=self.alert_engine.history,
            n_open_positions=len(self._positions),
            portfolio_delta=pd_,
            portfolio_gamma=pg,
            portfolio_theta=pt,
            portfolio_vega=pv,
            health_score=health,
        )

    # ── Persistence ──────────────────────────────────────────────────

    def save_state(self, path: Path = DEFAULT_STATE_PATH) -> None:
        snap = self.snapshot()
        data = {
            "timestamp": snap.timestamp,
            "daily_pnl": snap.daily_pnl,
            "unrealized_pnl": snap.unrealized_pnl,
            "realized_pnl": snap.realized_pnl,
            "total_pnl": snap.total_pnl,
            "drawdown": {
                "current_pct": snap.drawdown.drawdown_pct,
                "max_pct": snap.drawdown.max_drawdown_pct,
                "equity": snap.drawdown.current_equity,
                "peak": snap.drawdown.peak_equity,
            },
            "risk_budget": {
                "capital_at_risk": snap.risk_budget.capital_at_risk,
                "utilization": snap.risk_budget.utilization_pct,
            },
            "var": {
                "var_95": snap.var_state.current_var_95,
                "var_99": snap.var_state.current_var_99,
                "utilization": snap.var_state.utilization_pct,
            },
            "n_positions": snap.n_open_positions,
            "health_score": snap.health_score,
            "strategies": {
                k: {
                    "pnl": v.total_pnl,
                    "positions": v.n_positions,
                    "win_rate": v.win_rate,
                }
                for k, v in snap.strategy_attribution.items()
            },
            "alerts": [
                {
                    "timestamp": a.timestamp,
                    "severity": a.severity,
                    "category": a.category,
                    "title": a.title,
                }
                for a in snap.alert_history[-50:]
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0
        self.alert_engine.reset_cooldowns()

    # ── Dashboard generation ─────────────────────────────────────────

    @staticmethod
    def generate_dashboard(
        snap: DashboardSnapshot,
        output_path: Path = DEFAULT_OUTPUT,
        auto_refresh: int = 30,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_dashboard_html(snap, auto_refresh)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Dashboard written to %s", output_path)
        return output_path


# ── HTML helpers ─────────────────────────────────────────────────────────


def _fd(v: float) -> str:
    return f"${v:,.2f}"


def _fp(v: float) -> str:
    return f"{v:.1%}"


def _fr(v: float) -> str:
    return f"{v:.2f}"


def _pnl_color(v: float) -> str:
    return "#3fb950" if v >= 0 else "#f85149"


def _health_color(score: float) -> str:
    if score >= 80:
        return "#3fb950"
    if score >= 50:
        return "#d29922"
    return "#f85149"


def _sev_color(sev: str) -> str:
    return {"INFO": "#58a6ff", "WARNING": "#d29922", "CRITICAL": "#f85149"}.get(sev, "#8b949e")


def _gauge_svg(value: float, max_val: float, label: str, w: int = 160, h: int = 80) -> str:
    pct = min(value / max_val, 1.0) if max_val > 0 else 0
    color = "#3fb950" if pct < 0.6 else "#d29922" if pct < 0.8 else "#f85149"
    bar_w = w - 20
    fill_w = bar_w * pct
    return (
        f'<svg viewBox="0 0 {w} {h}" class="gauge">'
        f'<text x="{w // 2}" y="16" text-anchor="middle" font-size="10" fill="#8b949e">{label}</text>'
        f'<rect x="10" y="24" width="{bar_w}" height="14" fill="#21262d" rx="7"/>'
        f'<rect x="10" y="24" width="{fill_w:.0f}" height="14" fill="{color}" rx="7"/>'
        f'<text x="{w // 2}" y="55" text-anchor="middle" font-size="14" fill="#f0f6fc" font-weight="bold">{_fp(pct)}</text>'
        f'<text x="{w // 2}" y="70" text-anchor="middle" font-size="9" fill="#8b949e">{_fr(value)} / {_fr(max_val)}</text>'
        f"</svg>"
    )


def _stat_card(label: str, value: str, color: str = "#f0f6fc") -> str:
    return (
        f'<div class="card">'
        f'<div class="card-label">{label}</div>'
        f'<div class="card-value" style="color:{color}">{value}</div>'
        f"</div>"
    )


def _build_dashboard_html(snap: DashboardSnapshot, auto_refresh: int = 30) -> str:
    dd = snap.drawdown
    rb = snap.risk_budget
    vs = snap.var_state
    hc = _health_color(snap.health_score)

    # ── Stat cards ──
    cards = "".join([
        _stat_card("Daily P&L", _fd(snap.daily_pnl), _pnl_color(snap.daily_pnl)),
        _stat_card("Unrealized P&L", _fd(snap.unrealized_pnl), _pnl_color(snap.unrealized_pnl)),
        _stat_card("Realized P&L", _fd(snap.realized_pnl), _pnl_color(snap.realized_pnl)),
        _stat_card("Total P&L", _fd(snap.total_pnl), _pnl_color(snap.total_pnl)),
        _stat_card("Max Drawdown", _fp(dd.max_drawdown_pct), "#f85149" if dd.max_drawdown_pct > 0.05 else "#d29922"),
        _stat_card("Current DD", _fp(dd.drawdown_pct), _health_color(100 * (1 - dd.drawdown_pct / 0.1))),
        _stat_card("Open Positions", str(snap.n_open_positions)),
        _stat_card("Equity", _fd(dd.current_equity)),
    ])

    # ── Greeks cards ──
    greeks_cards = "".join([
        _stat_card("Portfolio Delta", _fr(snap.portfolio_delta)),
        _stat_card("Portfolio Gamma", _fr(snap.portfolio_gamma)),
        _stat_card("Portfolio Theta", _fr(snap.portfolio_theta)),
        _stat_card("Portfolio Vega", _fr(snap.portfolio_vega)),
    ])

    # ── Gauges ──
    gauges = "".join([
        _gauge_svg(rb.capital_at_risk, rb.max_allowed_risk, "Risk Budget"),
        _gauge_svg(vs.current_var_95, vs.var_limit, "VaR (95%)"),
        _gauge_svg(dd.drawdown_pct * 100, 10, "Drawdown %"),
        _gauge_svg(rb.margin_used, rb.total_capital, "Margin Used"),
    ])

    # ── Strategy attribution table ──
    strat_rows = ""
    for name, sa in sorted(snap.strategy_attribution.items()):
        wr = f"{sa.win_rate:.0%}" if sa.n_trades > 0 else "N/A"
        strat_rows += (
            f"<tr>"
            f"<td style='text-align:left'>{name}</td>"
            f"<td>{sa.n_positions}</td>"
            f"<td style='color:{_pnl_color(sa.total_pnl)}'>{_fd(sa.total_pnl)}</td>"
            f"<td style='color:{_pnl_color(sa.realized_pnl)}'>{_fd(sa.realized_pnl)}</td>"
            f"<td style='color:{_pnl_color(sa.unrealized_pnl)}'>{_fd(sa.unrealized_pnl)}</td>"
            f"<td>{wr}</td>"
            f"<td>{_fd(sa.capital_at_risk)}</td>"
            f"</tr>"
        )

    # ── Positions table ──
    pos_rows = ""
    for p in sorted(snap.positions, key=lambda x: x.unrealized_pnl):
        pos_rows += (
            f"<tr>"
            f"<td style='text-align:left'>{p.ticker}</td>"
            f"<td style='text-align:left'>{p.strategy}</td>"
            f"<td>{p.direction}</td>"
            f"<td>{p.contracts}</td>"
            f"<td>{_fd(p.entry_credit)}</td>"
            f"<td style='color:{_pnl_color(p.unrealized_pnl)}'>{_fd(p.unrealized_pnl)}</td>"
            f"<td>{p.dte}d</td>"
            f"<td>{_fr(p.delta)}</td>"
            f"<td>{_fr(p.theta)}</td>"
            f"<td>{_fd(p.max_loss)}</td>"
            f"</tr>"
        )

    # ── Correlation section ──
    corr = snap.correlation
    corr_html = ""
    if corr.strategy_pairs:
        corr_rows = ""
        for pair, val in sorted(corr.strategy_pairs.items(), key=lambda x: -abs(x[1])):
            c = "#f85149" if abs(val) >= 0.8 else "#d29922" if abs(val) >= 0.5 else "#3fb950"
            corr_rows += f"<tr><td style='text-align:left'>{pair}</td><td style='color:{c}'>{_fr(val)}</td></tr>"
        corr_html = (
            f"<h2>Strategy Correlations</h2>"
            f"<p class='meta'>Max: {_fr(corr.max_correlation)} ({corr.max_pair}) | Avg: {_fr(corr.avg_correlation)}</p>"
            f"<table class='dt'><tr><th style='text-align:left'>Pair</th><th>Correlation</th></tr>{corr_rows}</table>"
        )

    # ── Risk budget detail ──
    risk_detail = (
        f"<h2>Risk Budget</h2>"
        f"<div class='cards'>"
        f"{_stat_card('Capital at Risk', _fd(rb.capital_at_risk))}"
        f"{_stat_card('Max Allowed', _fd(rb.max_allowed_risk))}"
        f"{_stat_card('Headroom', _fd(rb.headroom), _pnl_color(rb.headroom))}"
        f"{_stat_card('Utilization', _fp(rb.utilization_pct), _health_color(100 * (1 - rb.utilization_pct)))}"
        f"{_stat_card('Margin Used', _fd(rb.margin_used))}"
        f"{_stat_card('Margin Available', _fd(rb.margin_available), _pnl_color(rb.margin_available))}"
        f"</div>"
    )

    # ── VaR detail ──
    var_detail = (
        f"<h2>VaR Utilization</h2>"
        f"<div class='cards'>"
        f"{_stat_card('VaR (95%)', _fd(vs.current_var_95))}"
        f"{_stat_card('VaR (99%)', _fd(vs.current_var_99))}"
        f"{_stat_card('VaR Limit', _fd(vs.var_limit))}"
        f"{_stat_card('Utilization', _fp(vs.utilization_pct), _health_color(100 * (1 - vs.utilization_pct)))}"
        f"</div>"
    )

    # ── Alert history ──
    alert_rows = ""
    for a in reversed(snap.alert_history[-30:]):
        sc = _sev_color(a.severity)
        alert_rows += (
            f"<tr>"
            f"<td>{a.timestamp}</td>"
            f"<td style='color:{sc}'>{a.severity}</td>"
            f"<td style='text-align:left'>{a.category}</td>"
            f"<td style='text-align:left'>{a.title}</td>"
            f"<td>{'Yes' if a.delivered else 'No'}</td>"
            f"</tr>"
        )
    alert_table = (
        f"<table class='dt'><tr><th>Time</th><th>Severity</th><th style='text-align:left'>Category</th>"
        f"<th style='text-align:left'>Title</th><th>Sent</th></tr>{alert_rows}</table>"
        if alert_rows else "<p class='meta'>No alerts triggered.</p>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta http-equiv="refresh" content="{auto_refresh}"/>
<title>Production Monitor Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; }}
  h2 {{ color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-top: 32px; }}
  .meta {{ color: #8b949e; font-size: 0.9em; }}
  .health {{ background: #161b22; border: 2px solid {hc}; border-radius: 12px;
             padding: 24px; text-align: center; margin: 20px 0; }}
  .health .big {{ font-size: 3.5em; font-weight: 800; color: {hc}; }}
  .health .label {{ color: #8b949e; font-size: 1.1em; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 10px; margin: 16px 0; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 12px; text-align: center; }}
  .card-label {{ color: #8b949e; font-size: 0.78em; margin-bottom: 4px; }}
  .card-value {{ color: #f0f6fc; font-weight: 600; font-size: 1.15em; }}
  .gauges {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
             gap: 8px; margin: 16px 0; }}
  .gauge {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; }}
  table.dt {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.85em; }}
  table.dt th, table.dt td {{ padding: 6px 8px; text-align: right; border-bottom: 1px solid #21262d; }}
  table.dt th {{ color: #8b949e; background: #161b22; position: sticky; top: 0; }}
  .section {{ margin-top: 24px; }}
  .refresh-badge {{ display: inline-block; background: #21262d; color: #8b949e; padding: 2px 8px;
                     border-radius: 10px; font-size: 0.75em; }}
</style>
</head>
<body>
<h1>Production Monitor Dashboard</h1>
<p class="meta">{snap.timestamp} &middot; {snap.n_open_positions} open positions &middot;
   {len(snap.alert_history)} alerts &middot;
   <span class="refresh-badge">Auto-refresh: {auto_refresh}s</span></p>

<div class="health">
  <div class="big">{snap.health_score:.0f}</div>
  <div class="label">System Health Score</div>
</div>

<h2>P&L Summary</h2>
<div class="cards">{cards}</div>

<h2>Portfolio Greeks</h2>
<div class="cards">{greeks_cards}</div>

<h2>Risk Gauges</h2>
<div class="gauges">{gauges}</div>

{risk_detail}

{var_detail}

<h2>Strategy Attribution</h2>
<table class="dt">
  <tr>
    <th style="text-align:left">Strategy</th><th>Positions</th><th>Total P&L</th>
    <th>Realized</th><th>Unrealized</th><th>Win Rate</th><th>Capital at Risk</th>
  </tr>
  {strat_rows}
</table>

<h2>Open Positions</h2>
<table class="dt">
  <tr>
    <th style="text-align:left">Ticker</th><th style="text-align:left">Strategy</th>
    <th>Direction</th><th>Contracts</th><th>Entry Credit</th><th>Unrealized P&L</th>
    <th>DTE</th><th>Delta</th><th>Theta</th><th>Max Loss</th>
  </tr>
  {pos_rows}
</table>

{corr_html}

<h2>Alert History</h2>
{alert_table}

</body></html>"""
