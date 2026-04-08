#!/usr/bin/env python3
"""EXP-880 Paper Trading Monitor — tracks performance, hedge activations,
and deviation from backtest expectations.

Reads from:
  - Alpaca API (read-only) for positions, P&L, orders
  - SQLite trade history database
  - EXP-880 backtest results for comparison

Outputs:
  - HTML dashboard (white background, clean formatting)
  - Telegram alerts for trades, hedges, DD breaches
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── EXP-880 Backtest Expectations ───────────────────────────────────────────
BACKTEST_EXPECTATIONS = {
    "cagr_pct": 76.9,
    "sharpe": 4.97,
    "max_dd_pct": 10.2,
    "annual_hedge_drag_pct": 0.33,
    "win_rate_pct": 75.0,
    "avg_scale": 0.85,
}

# Deviation thresholds (fraction of expected — alert if actual deviates by more)
DEVIATION_WARN = 0.20   # 20% worse than expected → warning
DEVIATION_CRIT = 0.40   # 40% worse → critical

DD_WARN_PCT = 5.0       # warn at 5% DD
DD_CRIT_PCT = 10.0      # critical at 10% DD
DD_HALT_PCT = 13.0      # halt at 13% DD (before 15% ceiling)


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class Position:
    symbol: str
    qty: int
    side: str
    avg_entry: float
    current_price: float
    unrealised_pnl: float
    market_value: float


@dataclass
class TradeRecord:
    trade_id: str
    entry_date: str
    exit_date: str
    symbol: str
    spread_type: str
    contracts: int
    credit: float
    pnl: float
    exit_reason: str
    regime: str = ""
    hedge_scale: float = 1.0


@dataclass
class HedgeEvent:
    timestamp: str
    vix: float
    scale_factor: float
    reason: str
    regime: str = ""
    dd_at_trigger: float = 0.0


@dataclass
class MonitorSnapshot:
    """Complete monitoring snapshot."""
    timestamp: str
    # Portfolio state
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    positions: List[Position] = field(default_factory=list)
    n_open_positions: int = 0
    # Performance
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    current_dd_pct: float = 0.0
    peak_equity: float = 0.0
    sharpe: float = 0.0
    win_rate_pct: float = 0.0
    # Trades
    total_trades: int = 0
    recent_trades: List[TradeRecord] = field(default_factory=list)
    # Hedge
    hedge_events: List[HedgeEvent] = field(default_factory=list)
    current_hedge_scale: float = 1.0
    n_hedge_activations: int = 0
    # Deviation from backtest
    deviations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Alerts
    alerts: List[Dict[str, str]] = field(default_factory=list)


# ── Alpaca API Client (read-only) ──────────────────────────────────────────
class AlpacaReader:
    """Read-only Alpaca paper trading API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url = base_url
        self._connected = False

    def connect(self) -> bool:
        """Test connection. Returns True if API keys are valid."""
        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca API keys not configured — using mock data")
            return False
        self._connected = True
        return True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_account(self) -> Dict[str, Any]:
        """Get account info. Returns mock if not connected."""
        if not self._connected:
            return {
                "equity": 100_000.0,
                "cash": 50_000.0,
                "buying_power": 200_000.0,
                "portfolio_value": 100_000.0,
            }
        # In production, would call: GET /v2/account
        try:
            import requests
            r = requests.get(
                f"{self.base_url}/v2/account",
                headers={"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret},
                timeout=10,
            )
            return r.json()
        except Exception as e:
            logger.error("Alpaca API error: %s", e)
            return {"equity": 0, "cash": 0, "buying_power": 0}

    def get_positions(self) -> List[Position]:
        """Get open positions."""
        if not self._connected:
            return []
        try:
            import requests
            r = requests.get(
                f"{self.base_url}/v2/positions",
                headers={"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret},
                timeout=10,
            )
            positions = []
            for p in r.json():
                positions.append(Position(
                    symbol=p.get("symbol", ""),
                    qty=int(p.get("qty", 0)),
                    side=p.get("side", ""),
                    avg_entry=float(p.get("avg_entry_price", 0)),
                    current_price=float(p.get("current_price", 0)),
                    unrealised_pnl=float(p.get("unrealized_pl", 0)),
                    market_value=float(p.get("market_value", 0)),
                ))
            return positions
        except Exception as e:
            logger.error("Error fetching positions: %s", e)
            return []

    def get_orders(self, status: str = "all", limit: int = 50) -> List[Dict]:
        """Get recent orders."""
        if not self._connected:
            return []
        try:
            import requests
            r = requests.get(
                f"{self.base_url}/v2/orders",
                headers={"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret},
                params={"status": status, "limit": limit},
                timeout=10,
            )
            return r.json()
        except Exception:
            return []


# ── Trade History Database ──────────────────────────────────────────────────
class TradeHistoryDB:
    """Read trade history from SQLite database."""

    def __init__(self, db_path: str = "data/paper_trades.db") -> None:
        self.db_path = db_path

    @property
    def exists(self) -> bool:
        return Path(self.db_path).exists()

    def get_trades(self, limit: int = 500) -> List[TradeRecord]:
        """Load trade records from database."""
        if not self.exists:
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY entry_date DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return [TradeRecord(
                trade_id=str(r.get("trade_id", r.get("id", ""))),
                entry_date=str(r.get("entry_date", "")),
                exit_date=str(r.get("exit_date", "")),
                symbol=str(r.get("symbol", "SPY")),
                spread_type=str(r.get("spread_type", "")),
                contracts=int(r.get("contracts", 0)),
                credit=float(r.get("credit", 0)),
                pnl=float(r.get("pnl", 0)),
                exit_reason=str(r.get("exit_reason", "")),
                regime=str(r.get("regime", "")),
                hedge_scale=float(r.get("hedge_scale", 1.0)),
            ) for r in rows]
        except Exception as e:
            logger.error("Error reading trade DB: %s", e)
            return []

    def get_hedge_events(self, limit: int = 100) -> List[HedgeEvent]:
        """Load hedge events from database."""
        if not self.exists:
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM hedge_events ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return [HedgeEvent(
                timestamp=str(r.get("timestamp", "")),
                vix=float(r.get("vix", 0)),
                scale_factor=float(r.get("scale_factor", 1.0)),
                reason=str(r.get("reason", "")),
                regime=str(r.get("regime", "")),
                dd_at_trigger=float(r.get("dd_at_trigger", 0)),
            ) for r in rows]
        except Exception as e:
            logger.error("Error reading hedge events: %s", e)
            return []

    def get_daily_equity(self) -> pd.DataFrame:
        """Load daily equity curve."""
        if not self.exists:
            return pd.DataFrame()
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql("SELECT date, equity FROM daily_equity ORDER BY date", conn)
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()


# ── Deviation Analysis ──────────────────────────────────────────────────────
def compute_deviations(
    actual: Dict[str, float],
    expected: Dict[str, float] = BACKTEST_EXPECTATIONS,
) -> Dict[str, Dict[str, Any]]:
    """Compare actual vs expected performance."""
    devs = {}
    for metric, exp_val in expected.items():
        act_val = actual.get(metric, 0.0)
        if abs(exp_val) < 1e-9:
            deviation_pct = 0.0
        else:
            deviation_pct = (act_val - exp_val) / abs(exp_val) * 100

        # Determine severity (some metrics: higher is better, some lower)
        higher_better = metric in ("cagr_pct", "sharpe", "win_rate_pct", "avg_scale")
        if higher_better:
            severity = "ok"
            if deviation_pct < -DEVIATION_WARN * 100:
                severity = "warning"
            if deviation_pct < -DEVIATION_CRIT * 100:
                severity = "critical"
        else:
            severity = "ok"
            if deviation_pct > DEVIATION_WARN * 100:
                severity = "warning"
            if deviation_pct > DEVIATION_CRIT * 100:
                severity = "critical"

        devs[metric] = {
            "expected": exp_val,
            "actual": act_val,
            "deviation_pct": round(deviation_pct, 1),
            "severity": severity,
        }
    return devs


# ── Alert Generator ─────────────────────────────────────────────────────────
def generate_alerts(snapshot: MonitorSnapshot) -> List[Dict[str, str]]:
    """Generate alerts based on current state."""
    alerts: List[Dict[str, str]] = []

    # DD alerts
    if snapshot.current_dd_pct >= DD_HALT_PCT:
        alerts.append({"level": "CRITICAL", "message": f"DD {snapshot.current_dd_pct:.1f}% — HALT TRADING"})
    elif snapshot.current_dd_pct >= DD_CRIT_PCT:
        alerts.append({"level": "CRITICAL", "message": f"DD {snapshot.current_dd_pct:.1f}% approaching ceiling"})
    elif snapshot.current_dd_pct >= DD_WARN_PCT:
        alerts.append({"level": "WARNING", "message": f"DD {snapshot.current_dd_pct:.1f}% — monitor closely"})

    # Hedge alerts
    if snapshot.current_hedge_scale < 0.50:
        alerts.append({"level": "WARNING", "message": f"Hedge active: scale {snapshot.current_hedge_scale:.0%}"})

    # Deviation alerts
    for metric, dev in snapshot.deviations.items():
        if dev["severity"] == "critical":
            alerts.append({"level": "CRITICAL", "message": f"{metric} deviation: {dev['deviation_pct']:+.0f}% from expected"})
        elif dev["severity"] == "warning":
            alerts.append({"level": "WARNING", "message": f"{metric} deviation: {dev['deviation_pct']:+.0f}% from expected"})

    return alerts


# ── Telegram Integration ────────────────────────────────────────────────────
def send_telegram_alerts(alerts: List[Dict[str, str]], snapshot: MonitorSnapshot) -> int:
    """Send alerts via Telegram. Returns count sent."""
    try:
        from shared.telegram_alerts import send_message, is_configured, set_experiment_id
    except ImportError:
        logger.warning("Telegram module not available")
        return 0

    if not is_configured():
        logger.info("Telegram not configured — skipping alerts")
        return 0

    set_experiment_id("EXP-880")
    sent = 0

    for alert in alerts:
        icon = "🚨" if alert["level"] == "CRITICAL" else "⚠️"
        text = (
            f"{icon} <b>EXP-880 {alert['level']}</b>\n"
            f"{alert['message']}\n"
            f"Equity: ${snapshot.equity:,.0f} | DD: {snapshot.current_dd_pct:.1f}%"
        )
        if send_message(text):
            sent += 1

    return sent


def send_trade_alert(trade: TradeRecord) -> bool:
    """Send Telegram alert for a new trade."""
    try:
        from shared.telegram_alerts import send_message, is_configured
        if not is_configured():
            return False
        icon = "✅" if trade.pnl > 0 else "❌"
        text = (
            f"{icon} <b>EXP-880 Trade</b>\n"
            f"{trade.spread_type} {trade.symbol} × {trade.contracts}\n"
            f"P&L: ${trade.pnl:+,.0f} | Exit: {trade.exit_reason}\n"
            f"Hedge scale: {trade.hedge_scale:.0%}"
        )
        return send_message(text)
    except Exception:
        return False


def send_hedge_alert(event: HedgeEvent) -> bool:
    """Send Telegram alert for hedge activation."""
    try:
        from shared.telegram_alerts import send_message, is_configured
        if not is_configured():
            return False
        icon = "🛡️" if event.scale_factor < 0.80 else "📊"
        text = (
            f"{icon} <b>EXP-880 Hedge</b>\n"
            f"Scale: {event.scale_factor:.0%} | VIX: {event.vix:.1f}\n"
            f"Reason: {event.reason}\n"
            f"DD at trigger: {event.dd_at_trigger:.1%}"
        )
        return send_message(text)
    except Exception:
        return False


# ── Monitor Core ────────────────────────────────────────────────────────────
class EXP880Monitor:
    """Main monitoring class for EXP-880 paper trading."""

    def __init__(
        self,
        alpaca: Optional[AlpacaReader] = None,
        trade_db: Optional[TradeHistoryDB] = None,
        starting_capital: float = 100_000.0,
    ) -> None:
        self.alpaca = alpaca or AlpacaReader()
        self.trade_db = trade_db or TradeHistoryDB()
        self.starting_capital = starting_capital
        self._snapshots: List[MonitorSnapshot] = []

    def take_snapshot(self) -> MonitorSnapshot:
        """Capture current state from all data sources."""
        now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

        # Account data
        acct = self.alpaca.get_account()
        equity = float(acct.get("equity", acct.get("portfolio_value", self.starting_capital)))
        cash = float(acct.get("cash", 0))
        bp = float(acct.get("buying_power", 0))
        positions = self.alpaca.get_positions()

        # Trades
        trades = self.trade_db.get_trades(limit=200)
        hedge_events = self.trade_db.get_hedge_events(limit=50)

        # Compute performance
        total_pnl = sum(t.pnl for t in trades)
        total_return = (equity - self.starting_capital) / self.starting_capital * 100
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = (wins / len(trades) * 100) if trades else 0

        # Drawdown
        peak = max(equity, self.starting_capital)
        if self._snapshots:
            peak = max(peak, max(s.peak_equity for s in self._snapshots))
        dd = (peak - equity) / peak * 100 if peak > 0 else 0

        # Sharpe (from daily P&L if available)
        eq_df = self.trade_db.get_daily_equity()
        if not eq_df.empty and len(eq_df) > 5:
            daily_ret = eq_df["equity"].pct_change().dropna()
            sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
        else:
            # Estimate from trades
            if trades and len(trades) > 5:
                pnls = [t.pnl for t in trades]
                sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252 / max(len(trades), 1) * 5)) if np.std(pnls) > 0 else 0
            else:
                sharpe = 0

        # Current hedge scale
        current_scale = 1.0
        if hedge_events:
            current_scale = hedge_events[0].scale_factor
        n_activations = sum(1 for h in hedge_events if h.scale_factor < 0.80)

        # Deviations
        actual = {
            "cagr_pct": total_return,  # simplified — not annualised
            "sharpe": sharpe,
            "max_dd_pct": dd,
            "win_rate_pct": win_rate,
            "avg_scale": current_scale,
        }
        devs = compute_deviations(actual)

        snap = MonitorSnapshot(
            timestamp=now,
            equity=equity,
            cash=cash,
            buying_power=bp,
            positions=positions,
            n_open_positions=len(positions),
            total_pnl=round(total_pnl, 2),
            total_return_pct=round(total_return, 2),
            current_dd_pct=round(dd, 2),
            peak_equity=peak,
            sharpe=round(sharpe, 2),
            win_rate_pct=round(win_rate, 1),
            total_trades=len(trades),
            recent_trades=trades[:20],
            hedge_events=hedge_events[:20],
            current_hedge_scale=current_scale,
            n_hedge_activations=n_activations,
            deviations=devs,
        )

        # Generate alerts
        snap.alerts = generate_alerts(snap)
        self._snapshots.append(snap)

        return snap

    def generate_report(
        self,
        snapshot: MonitorSnapshot,
        output_path: str = "reports/exp880_monitor.html",
    ) -> Path:
        """Generate HTML monitoring dashboard."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(snapshot)
        path.write_text(html, encoding="utf-8")
        return path

    def run_alerts(self, snapshot: MonitorSnapshot) -> int:
        """Send all alerts via Telegram. Returns count sent."""
        return send_telegram_alerts(snapshot.alerts, snapshot)

    # ── HTML Report ─────────────────────────────────────────────────────────
    def _build_html(self, s: MonitorSnapshot) -> str:
        cards = self._html_cards(s)
        alerts_sec = self._html_alerts(s.alerts)
        deviation_tbl = self._html_deviations(s.deviations)
        trades_tbl = self._html_trades(s.recent_trades)
        hedge_tbl = self._html_hedge_events(s.hedge_events)
        positions_tbl = self._html_positions(s.positions)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EXP-880 Paper Trading Monitor</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#fff;color:#1e293b;padding:24px;max-width:1100px;margin:0 auto}}
h1{{font-size:1.6rem;margin-bottom:4px;color:#0f172a}}
.sub{{color:#64748b;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:24px}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px}}
.card .lbl{{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.04em}}
.card .val{{font-size:1.3rem;font-weight:700;margin-top:3px;color:#0f172a}}
.sec{{margin-bottom:28px}}
.sec h2{{font-size:1rem;margin-bottom:10px;color:#334155;border-bottom:2px solid #e2e8f0;padding-bottom:4px}}
table{{width:100%;border-collapse:collapse;font-size:.83rem}}
th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #e2e8f0}}
th{{color:#64748b;font-weight:600;background:#f8fafc}}
tr:hover{{background:#f1f5f9}}
.pos{{color:#16a34a}}.neg{{color:#dc2626}}.warn{{color:#d97706}}
.alert{{padding:10px 14px;border-radius:6px;margin-bottom:8px;font-size:.85rem}}
.alert.CRITICAL{{background:#fef2f2;border-left:4px solid #dc2626;color:#991b1b}}
.alert.WARNING{{background:#fffbeb;border-left:4px solid #d97706;color:#92400e}}
.alert.INFO{{background:#f0f9ff;border-left:4px solid #0284c7;color:#075985}}
</style>
</head>
<body>
<h1>EXP-880 Paper Trading Monitor</h1>
<p class="sub">Snapshot: {s.timestamp} &middot; {s.total_trades} trades &middot; Hedge scale: {s.current_hedge_scale:.0%}</p>

{alerts_sec}
{cards}
{deviation_tbl}
{positions_tbl}
{trades_tbl}
{hedge_tbl}

<p style="color:#94a3b8;font-size:.75rem;margin-top:20px">Generated by scripts/monitor_exp880.py</p>
</body>
</html>"""

    @staticmethod
    def _html_cards(s: MonitorSnapshot) -> str:
        eq_cls = "pos" if s.total_return_pct >= 0 else "neg"
        dd_cls = "neg" if s.current_dd_pct > 5 else ""
        return f"""<div class="grid">
<div class="card"><div class="lbl">Equity</div><div class="val">${s.equity:,.0f}</div></div>
<div class="card"><div class="lbl">Total P&L</div><div class="val {eq_cls}">${s.total_pnl:+,.0f}</div></div>
<div class="card"><div class="lbl">Return</div><div class="val {eq_cls}">{s.total_return_pct:+.1f}%</div></div>
<div class="card"><div class="lbl">Drawdown</div><div class="val {dd_cls}">{s.current_dd_pct:.1f}%</div></div>
<div class="card"><div class="lbl">Sharpe</div><div class="val">{s.sharpe:.2f}</div></div>
<div class="card"><div class="lbl">Win Rate</div><div class="val">{s.win_rate_pct:.0f}%</div></div>
<div class="card"><div class="lbl">Trades</div><div class="val">{s.total_trades}</div></div>
<div class="card"><div class="lbl">Hedge Scale</div><div class="val {'warn' if s.current_hedge_scale < 0.8 else ''}">{s.current_hedge_scale:.0%}</div></div>
<div class="card"><div class="lbl">Open Pos</div><div class="val">{s.n_open_positions}</div></div>
<div class="card"><div class="lbl">Hedge Acts</div><div class="val">{s.n_hedge_activations}</div></div>
</div>"""

    @staticmethod
    def _html_alerts(alerts: List[Dict[str, str]]) -> str:
        if not alerts:
            return '<div class="alert INFO">No active alerts — system nominal</div>'
        items = ""
        for a in alerts:
            items += f'<div class="alert {a["level"]}">{a["level"]}: {a["message"]}</div>'
        return f'<div class="sec">{items}</div>'

    @staticmethod
    def _html_deviations(devs: Dict[str, Dict]) -> str:
        if not devs:
            return ""
        rows = ""
        for metric, d in sorted(devs.items()):
            cls = "neg" if d["severity"] == "critical" else "warn" if d["severity"] == "warning" else "pos"
            rows += (f"<tr><td>{metric}</td><td>{d['expected']}</td><td>{d['actual']}</td>"
                     f'<td class="{cls}">{d["deviation_pct"]:+.1f}%</td>'
                     f'<td class="{cls}">{d["severity"].upper()}</td></tr>')
        return f"""<div class="sec"><h2>Backtest Deviation Analysis</h2>
<table><thead><tr><th>Metric</th><th>Expected</th><th>Actual</th><th>Deviation</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_positions(positions: List[Position]) -> str:
        if not positions:
            return '<div class="sec"><h2>Open Positions</h2><p style="color:#94a3b8">No open positions</p></div>'
        rows = ""
        for p in positions:
            cls = "pos" if p.unrealised_pnl >= 0 else "neg"
            rows += (f"<tr><td>{p.symbol}</td><td>{p.qty}</td><td>{p.side}</td>"
                     f"<td>${p.avg_entry:.2f}</td><td>${p.current_price:.2f}</td>"
                     f'<td class="{cls}">${p.unrealised_pnl:+,.0f}</td></tr>')
        return f"""<div class="sec"><h2>Open Positions</h2>
<table><thead><tr><th>Symbol</th><th>Qty</th><th>Side</th><th>Entry</th><th>Current</th><th>P&L</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_trades(trades: List[TradeRecord]) -> str:
        if not trades:
            return ""
        rows = ""
        for t in trades[:20]:
            cls = "pos" if t.pnl > 0 else "neg"
            rows += (f"<tr><td>{t.entry_date}</td><td>{t.exit_date}</td>"
                     f"<td>{t.spread_type}</td><td>{t.contracts}</td>"
                     f'<td class="{cls}">${t.pnl:+,.0f}</td>'
                     f"<td>{t.exit_reason}</td><td>{t.hedge_scale:.0%}</td></tr>")
        return f"""<div class="sec"><h2>Recent Trades</h2>
<table><thead><tr><th>Entry</th><th>Exit</th><th>Type</th><th>Qty</th><th>P&L</th><th>Exit</th><th>Hedge</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_hedge_events(events: List[HedgeEvent]) -> str:
        if not events:
            return ""
        rows = ""
        for h in events[:20]:
            cls = "warn" if h.scale_factor < 0.80 else ""
            rows += (f"<tr><td>{h.timestamp}</td><td>{h.vix:.1f}</td>"
                     f'<td class="{cls}">{h.scale_factor:.0%}</td>'
                     f"<td>{h.reason}</td><td>{h.regime}</td>"
                     f"<td>{h.dd_at_trigger:.1%}</td></tr>")
        return f"""<div class="sec"><h2>Hedge Activity Log</h2>
<table><thead><tr><th>Time</th><th>VIX</th><th>Scale</th><th>Reason</th><th>Regime</th><th>DD</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    monitor = EXP880Monitor()

    # Try to connect to Alpaca
    if monitor.alpaca.connect():
        print("Connected to Alpaca paper trading API")
    else:
        print("Alpaca not configured — using mock data")

    # Take snapshot
    snapshot = monitor.take_snapshot()

    # Generate report
    report_path = monitor.generate_report(snapshot)
    print(f"Report: {report_path}")

    # Send alerts
    n_sent = monitor.run_alerts(snapshot)
    print(f"Alerts: {len(snapshot.alerts)} generated, {n_sent} sent via Telegram")

    # Summary
    print(f"\nEquity: ${snapshot.equity:,.0f}")
    print(f"P&L: ${snapshot.total_pnl:+,.0f} ({snapshot.total_return_pct:+.1f}%)")
    print(f"DD: {snapshot.current_dd_pct:.1f}%")
    print(f"Trades: {snapshot.total_trades}")
    print(f"Hedge: {snapshot.current_hedge_scale:.0%} ({snapshot.n_hedge_activations} activations)")
