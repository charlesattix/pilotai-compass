"""
compass/paper_trading_monitor.py — Paper Trading Performance Monitor.

Tracks live paper trades, computes real-time metrics, and alerts
when realized performance deviates from backtest predictions.

Usage:
    monitor = PaperTradingMonitor(
        backtest_sharpe=4.10,
        backtest_cagr=0.556,
        backtest_dd=0.072,
        backtest_wr=0.75,
        alert_threshold_sigma=2.0,
    )
    monitor.add_trade(trade)
    alerts = monitor.check_alerts()
    report = monitor.status_report()
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """A single paper trade record."""
    trade_id: str
    entry_date: str
    exit_date: str = ""
    ticker: str = "SPY"
    strategy: str = ""
    direction: str = ""          # "bull_put", "bear_call", "iron_condor"
    entry_credit: float = 0.0
    exit_value: float = 0.0
    pnl: float = 0.0
    contracts: int = 1
    status: str = "open"         # "open", "closed", "expired"
    exit_reason: str = ""


@dataclass
class Alert:
    """Performance deviation alert."""
    timestamp: str
    severity: str                # "INFO", "WARNING", "CRITICAL"
    metric: str                  # what deviated
    expected: float
    actual: float
    deviation_sigma: float
    message: str


@dataclass
class BacktestBenchmark:
    """Backtest predictions we compare against."""
    sharpe: float = 4.10
    cagr: float = 0.556          # 55.6%
    max_dd: float = 0.072        # 7.2%
    win_rate: float = 0.75
    avg_pnl: float = 150.0       # per trade
    avg_hold_days: float = 14.0
    trades_per_month: float = 8.0
    hedge_cost_annual: float = 0.0436  # 4.36% annual SPY put cost


# ═══════════════════════════════════════════════════════════════════════════
# Pre-flight checks
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PreflightResult:
    """Result of a single pre-flight check."""
    name: str
    passed: bool
    message: str
    severity: str = "INFO"       # "INFO", "WARNING", "CRITICAL"


def check_ironvault_freshness(db_path: Optional[str] = None) -> PreflightResult:
    """Check if options_cache.db exists and has recent data."""
    if db_path is None:
        db_path = str(ROOT / "data" / "options_cache.db")
    if not os.path.exists(db_path):
        return PreflightResult("IronVault DB", False,
                               f"Database not found at {db_path}", "CRITICAL")
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    if size_mb < 100:
        return PreflightResult("IronVault DB", False,
                               f"Database too small ({size_mb:.0f}MB, expected >900MB)", "CRITICAL")

    # Check latest data date
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM option_daily WHERE contract_symbol LIKE 'O:SPY%'")
        latest = cur.fetchone()[0]
        conn.close()
        if latest:
            days_old = (datetime.now() - datetime.strptime(latest, "%Y-%m-%d")).days
            if days_old > 7:
                return PreflightResult("IronVault DB", True,
                                       f"DB has data through {latest} ({days_old} days old — consider refresh)",
                                       "WARNING")
            return PreflightResult("IronVault DB", True,
                                   f"DB current through {latest} ({size_mb:.0f}MB)")
        return PreflightResult("IronVault DB", False, "No SPY data found in DB", "CRITICAL")
    except Exception as e:
        return PreflightResult("IronVault DB", False, f"Error reading DB: {e}", "CRITICAL")


def check_alpaca_connectivity(env_file: Optional[str] = None) -> PreflightResult:
    """Check if Alpaca API credentials are configured and reachable."""
    env_path = env_file or str(ROOT / ".env.ultimate_v4")
    if not os.path.exists(env_path):
        # Check if keys are in environment
        key = os.getenv("ALPACA_API_KEY", "")
        if not key:
            return PreflightResult("Alpaca API", False,
                                   f"No env file at {env_path} and ALPACA_API_KEY not set", "CRITICAL")
        return PreflightResult("Alpaca API", True, "API key found in environment")

    # Parse the env file for key presence
    try:
        content = Path(env_path).read_text()
        has_key = "ALPACA_API_KEY=" in content and len(
            [l for l in content.split("\n") if l.startswith("ALPACA_API_KEY=") and len(l) > 20]
        ) > 0
        if has_key:
            return PreflightResult("Alpaca API", True, f"Credentials found in {env_path}")
        return PreflightResult("Alpaca API", False,
                               f"ALPACA_API_KEY not found or empty in {env_path}", "CRITICAL")
    except Exception as e:
        return PreflightResult("Alpaca API", False, f"Error reading {env_path}: {e}", "CRITICAL")


def check_polygon_status() -> PreflightResult:
    """Check if Polygon API key is configured."""
    key = os.getenv("POLYGON_API_KEY", "")
    if not key:
        # Check .env
        env_path = ROOT / ".env"
        if env_path.exists():
            content = env_path.read_text()
            if "POLYGON_API_KEY=" in content:
                lines = [l for l in content.split("\n")
                         if l.startswith("POLYGON_API_KEY=") and len(l) > 20]
                if lines:
                    return PreflightResult("Polygon Data", True,
                                           "API key found in .env")
        return PreflightResult("Polygon Data", False,
                               "POLYGON_API_KEY not set (needed for data refresh)", "WARNING")
    return PreflightResult("Polygon Data", True, "API key set in environment")


def check_config_exists(config_path: Optional[str] = None) -> PreflightResult:
    """Check if paper trading config YAML exists."""
    if config_path is None:
        config_path = str(ROOT / "configs" / "paper_ultimate_v4.yaml")
    if os.path.exists(config_path):
        size = os.path.getsize(config_path)
        return PreflightResult("Config File", True,
                               f"{config_path} exists ({size} bytes)")
    return PreflightResult("Config File", False,
                           f"Config not found at {config_path}", "CRITICAL")


def check_python_deps() -> PreflightResult:
    """Check that required Python packages are importable."""
    missing = []
    for pkg in ["numpy", "pandas", "yfinance"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return PreflightResult("Python Deps", False,
                               f"Missing packages: {', '.join(missing)}", "CRITICAL")
    return PreflightResult("Python Deps", True, "All required packages available")


def run_preflight(env_file: Optional[str] = None,
                   config_path: Optional[str] = None) -> List[PreflightResult]:
    """Run all pre-flight checks."""
    checks = [
        check_python_deps(),
        check_ironvault_freshness(),
        check_alpaca_connectivity(env_file),
        check_polygon_status(),
        check_config_exists(config_path),
    ]
    return checks


# ═══════════════════════════════════════════════════════════════════════════
# Core monitor
# ═══════════════════════════════════════════════════════════════════════════

class PaperTradingMonitor:
    """Monitors paper trading performance vs backtest predictions.

    Alerts when realized metrics deviate by more than alert_threshold_sigma
    standard deviations from expected values.
    """

    def __init__(
        self,
        benchmark: Optional[BacktestBenchmark] = None,
        alert_threshold_sigma: float = 2.0,
        capital: float = 100_000,
    ):
        self.benchmark = benchmark or BacktestBenchmark()
        self.alert_sigma = alert_threshold_sigma
        self.capital = capital
        self.trades: List[Trade] = []
        self.alerts: List[Alert] = []
        self._peak_equity = capital

    @property
    def closed_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.status == "closed"]

    @property
    def open_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.status == "open"]

    def add_trade(self, trade: Trade) -> None:
        """Register a new trade."""
        self.trades.append(trade)
        logger.info("Trade %s added: %s %s %.2f",
                     trade.trade_id, trade.ticker, trade.direction, trade.pnl)

    def close_trade(self, trade_id: str, exit_date: str, pnl: float,
                     exit_reason: str = "closed") -> None:
        """Close an existing trade."""
        for t in self.trades:
            if t.trade_id == trade_id and t.status == "open":
                t.status = "closed"
                t.exit_date = exit_date
                t.pnl = pnl
                t.exit_reason = exit_reason
                return
        logger.warning("Trade %s not found or already closed", trade_id)

    # ── Metrics ────────────────────────────────────────────────────────

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all current performance metrics."""
        closed = self.closed_trades
        if not closed:
            return {
                "n_trades": 0, "n_open": len(self.open_trades),
                "total_pnl": 0, "win_rate": 0, "sharpe": 0,
                "max_dd": 0, "current_dd": 0, "equity": self.capital,
                "annualized_return": 0, "avg_pnl": 0,
                "days_active": 0,
            }

        pnls = np.array([t.pnl for t in closed])
        n = len(pnls)
        total_pnl = float(pnls.sum())
        wins = int((pnls > 0).sum())

        # Equity curve
        equity = np.cumsum(pnls) + self.capital
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        max_dd = float(dd.max())
        current_dd = float(dd[-1]) if len(dd) > 0 else 0

        # Sharpe (annualized from trade returns)
        std = float(pnls.std(ddof=1)) if n > 1 else 1.0
        sharpe = float(pnls.mean() / std * math.sqrt(min(n, 52))) if std > 1e-9 else 0.0

        # Days active
        dates = sorted(set(t.entry_date for t in closed))
        if len(dates) >= 2:
            first = datetime.strptime(dates[0], "%Y-%m-%d")
            last = datetime.strptime(dates[-1], "%Y-%m-%d")
            days_active = (last - first).days
        else:
            days_active = 0

        ann_return = 0
        if days_active > 30:
            years = days_active / 365.25
            ann_return = ((1 + total_pnl / self.capital) ** (1 / years) - 1) if total_pnl > -self.capital else -1

        return {
            "n_trades": n,
            "n_open": len(self.open_trades),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(wins / n, 4) if n > 0 else 0,
            "sharpe": round(sharpe, 3),
            "max_dd": round(max_dd, 4),
            "current_dd": round(current_dd, 4),
            "equity": round(float(equity[-1]), 2),
            "annualized_return": round(ann_return, 4),
            "avg_pnl": round(float(pnls.mean()), 2),
            "days_active": days_active,
            "peak_equity": round(float(peak[-1]), 2),
        }

    # ── Alert checking ─────────────────────────────────────────────────

    def check_alerts(self) -> List[Alert]:
        """Check if realized performance deviates from expectations.

        Returns new alerts generated since last check.
        """
        metrics = self.compute_metrics()
        new_alerts = []
        now = datetime.utcnow().isoformat()

        if metrics["n_trades"] < 5:
            return new_alerts  # too few trades for statistical comparison

        b = self.benchmark

        # Win rate deviation
        expected_wr = b.win_rate
        actual_wr = metrics["win_rate"]
        n = metrics["n_trades"]
        wr_std = math.sqrt(expected_wr * (1 - expected_wr) / max(n, 1))
        if wr_std > 0:
            wr_z = (actual_wr - expected_wr) / wr_std
            if abs(wr_z) > self.alert_sigma:
                sev = "CRITICAL" if abs(wr_z) > 3 else "WARNING"
                new_alerts.append(Alert(
                    timestamp=now, severity=sev, metric="win_rate",
                    expected=expected_wr, actual=actual_wr,
                    deviation_sigma=round(wr_z, 2),
                    message=f"Win rate {actual_wr:.0%} vs expected {expected_wr:.0%} ({wr_z:+.1f}σ)",
                ))

        # Average PnL deviation
        expected_avg = b.avg_pnl
        actual_avg = metrics["avg_pnl"]
        # Use bootstrap estimate of std
        pnl_std = abs(expected_avg) * 0.5  # rough estimate
        if pnl_std > 0:
            pnl_z = (actual_avg - expected_avg) / pnl_std
            if abs(pnl_z) > self.alert_sigma:
                sev = "CRITICAL" if actual_avg < 0 else "WARNING"
                new_alerts.append(Alert(
                    timestamp=now, severity=sev, metric="avg_pnl",
                    expected=expected_avg, actual=actual_avg,
                    deviation_sigma=round(pnl_z, 2),
                    message=f"Avg PnL ${actual_avg:.0f} vs expected ${expected_avg:.0f} ({pnl_z:+.1f}σ)",
                ))

        # Drawdown alert
        if metrics["max_dd"] > b.max_dd * 1.5:
            dd_z = (metrics["max_dd"] - b.max_dd) / (b.max_dd * 0.3)
            new_alerts.append(Alert(
                timestamp=now, severity="CRITICAL", metric="max_drawdown",
                expected=b.max_dd, actual=metrics["max_dd"],
                deviation_sigma=round(dd_z, 2),
                message=f"Max DD {metrics['max_dd']:.1%} exceeds 1.5× backtest DD ({b.max_dd:.1%})",
            ))

        # Trade frequency check
        if metrics["days_active"] > 30:
            months = metrics["days_active"] / 30.0
            actual_rate = metrics["n_trades"] / months
            expected_rate = b.trades_per_month
            rate_std = math.sqrt(expected_rate) / 2  # Poisson-ish
            if rate_std > 0:
                rate_z = (actual_rate - expected_rate) / rate_std
                if abs(rate_z) > self.alert_sigma:
                    sev = "WARNING" if actual_rate < expected_rate else "INFO"
                    new_alerts.append(Alert(
                        timestamp=now, severity=sev, metric="trade_frequency",
                        expected=expected_rate, actual=round(actual_rate, 1),
                        deviation_sigma=round(rate_z, 2),
                        message=f"Trade rate {actual_rate:.1f}/mo vs expected {expected_rate:.0f}/mo ({rate_z:+.1f}σ)",
                    ))

        # Log alerts
        for alert in new_alerts:
            log_fn = {"INFO": logger.info, "WARNING": logger.warning,
                      "CRITICAL": logger.critical}.get(alert.severity, logger.info)
            log_fn("ALERT [%s] %s: %s", alert.severity, alert.metric, alert.message)

        self.alerts.extend(new_alerts)
        return new_alerts

    # ── Status report ──────────────────────────────────────────────────

    def status_report(self) -> str:
        """Generate a human-readable status report."""
        metrics = self.compute_metrics()
        b = self.benchmark
        lines = [
            "=" * 60,
            "  Paper Trading Monitor — Status Report",
            "=" * 60,
            "",
            f"  Trades:        {metrics['n_trades']} closed, {metrics['n_open']} open",
            f"  Total PnL:     ${metrics['total_pnl']:,.2f}",
            f"  Equity:        ${metrics['equity']:,.2f} (peak: ${metrics.get('peak_equity', 0):,.2f})",
            f"  Win Rate:      {metrics['win_rate']:.0%} (backtest: {b.win_rate:.0%})",
            f"  Avg PnL:       ${metrics['avg_pnl']:.2f} (backtest: ${b.avg_pnl:.2f})",
            f"  Sharpe:        {metrics['sharpe']:.2f} (backtest: {b.sharpe:.2f})",
            f"  Max DD:        {metrics['max_dd']:.1%} (backtest: {b.max_dd:.1%})",
            f"  Current DD:    {metrics['current_dd']:.1%}",
            f"  Ann. Return:   {metrics['annualized_return']:.1%} (backtest: {b.cagr:.1%})",
            f"  Days Active:   {metrics['days_active']}",
            "",
            f"  Active Alerts: {len([a for a in self.alerts if a.severity in ('WARNING', 'CRITICAL')])}",
        ]

        if self.alerts:
            lines.append("")
            lines.append("  Recent Alerts:")
            for a in self.alerts[-5:]:
                lines.append(f"    [{a.severity}] {a.message}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────

    def save_state(self, path: str | Path) -> None:
        """Save monitor state to JSON."""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "capital": self.capital,
            "benchmark": {
                "sharpe": self.benchmark.sharpe,
                "cagr": self.benchmark.cagr,
                "max_dd": self.benchmark.max_dd,
                "win_rate": self.benchmark.win_rate,
                "avg_pnl": self.benchmark.avg_pnl,
                "hedge_cost_annual": self.benchmark.hedge_cost_annual,
            },
            "trades": [
                {"trade_id": t.trade_id, "entry_date": t.entry_date,
                 "exit_date": t.exit_date, "ticker": t.ticker,
                 "strategy": t.strategy, "pnl": t.pnl,
                 "status": t.status, "exit_reason": t.exit_reason}
                for t in self.trades
            ],
            "alerts": [
                {"timestamp": a.timestamp, "severity": a.severity,
                 "metric": a.metric, "message": a.message,
                 "deviation_sigma": a.deviation_sigma}
                for a in self.alerts
            ],
            "metrics": self.compute_metrics(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load_state(cls, path: str | Path,
                    benchmark: Optional[BacktestBenchmark] = None) -> "PaperTradingMonitor":
        """Load monitor state from JSON."""
        data = json.loads(Path(path).read_text())
        b = benchmark or BacktestBenchmark(**data.get("benchmark", {}))
        monitor = cls(benchmark=b, capital=data.get("capital", 100_000))
        for td in data.get("trades", []):
            monitor.trades.append(Trade(**td))
        return monitor
