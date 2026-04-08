"""
scripts/north_star_monitor.py — EXP-1900 North Star monitoring dashboard.

Polls the Alpaca paper account every N seconds (default 300 = 5 min) and:

  1. Pulls open positions and account equity from Alpaca paper API
  2. Computes intraday + cumulative P&L vs the configured starting capital
  3. Evaluates portfolio against the risk limits in
     configs/north_star_paper.yaml (mirrors compass/risk_overlay.py)
  4. Writes a heartbeat / health JSON to logs/north_star/health.json
  5. Sends Telegram alerts via shared.telegram_alerts on:
        - any new fill since last poll (entry/exit)
        - any risk-limit breach
        - the daily P&L summary (once per session, after configured time)

The monitor is read-only on the broker side: it never submits orders.
The paper engine (compass.paper_engine) is the only writer.

Usage:
    python scripts/north_star_monitor.py --config configs/north_star_paper.yaml
    python scripts/north_star_monitor.py --once         # single poll
    python scripts/north_star_monitor.py --interval 60  # 1-min polling
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("north_star_monitor")


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MonitorConfig:
    config_path: Path
    interval_seconds: int = 300
    once: bool = False
    foreground: bool = False
    starting_capital: float = 100_000.0

    # Risk limits (loaded from YAML)
    max_daily_loss_pct: float = 4.0
    max_weekly_loss_pct: float = 7.0
    max_drawdown_halt_pct: float = 13.0
    drawdown_recovery_pct: float = 6.5
    max_open_positions_total: int = 18
    max_open_positions_per_strategy: int = 8
    max_strategy_weight_pct: float = 65.0
    max_risk_per_trade_pct: float = 2.0
    vix_crisis_block: float = 35.0
    vix_emergency_exit_all: float = 45.0

    # Paths
    log_dir: Path = field(default_factory=lambda: ROOT / "logs" / "north_star")
    health_file: Path = field(default_factory=lambda: ROOT / "logs" / "north_star" / "health.json")
    state_file: Path = field(default_factory=lambda: ROOT / "logs" / "north_star" / "state.json")

    # Daily summary timing
    daily_summary_time: str = "16:05"
    sent_summary_for_date: Optional[str] = None


def load_config(path: Path) -> MonitorConfig:
    cfg_dict = yaml.safe_load(path.read_text())
    risk = cfg_dict.get("risk", {}) or {}
    monitoring = cfg_dict.get("monitoring", {}) or {}
    alerts = cfg_dict.get("alerts", {}) or {}
    account = cfg_dict.get("account", {}) or {}

    log_dir = ROOT / monitoring.get("log_dir",
                                       cfg_dict.get("logging", {}).get("dir",
                                                                         "logs/north_star"))
    health_file = ROOT / monitoring.get("health_file", "logs/north_star/health.json")
    state_file = ROOT / monitoring.get("state_file", "logs/north_star/state.json")

    return MonitorConfig(
        config_path=path,
        interval_seconds=int(monitoring.get("check_interval_minutes", 5)) * 60,
        starting_capital=float(account.get("starting_capital", 100_000)),
        max_daily_loss_pct=float(risk.get("max_daily_loss_pct", 4.0)),
        max_weekly_loss_pct=float(risk.get("max_weekly_loss_pct", 7.0)),
        max_drawdown_halt_pct=float(risk.get("max_drawdown_halt_pct", 13.0)),
        drawdown_recovery_pct=float(risk.get("drawdown_recovery_pct", 6.5)),
        max_open_positions_total=int(risk.get("max_open_positions_total", 18)),
        max_open_positions_per_strategy=int(risk.get("max_open_positions_per_strategy", 8)),
        max_strategy_weight_pct=float(risk.get("max_strategy_weight_pct", 65.0)),
        max_risk_per_trade_pct=float(risk.get("max_risk_per_trade_pct", 2.0)),
        vix_crisis_block=float(risk.get("vix_crisis_block", 35.0)),
        vix_emergency_exit_all=float(risk.get("vix_emergency_exit_all", 45.0)),
        log_dir=log_dir,
        health_file=health_file,
        state_file=state_file,
        daily_summary_time=str(alerts.get("daily_summary_time", "16:05")),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Alpaca paper client (lazy import — degrade gracefully if SDK absent)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    strategy_id: str = "unknown"


@dataclass
class AccountSnapshot:
    timestamp: str
    equity: float
    cash: float
    positions: List[Position]
    fills_since_last: List[Dict[str, Any]]
    raw_error: Optional[str] = None


def fetch_account_snapshot(cfg: MonitorConfig,
                              last_fill_id: Optional[str]) -> AccountSnapshot:
    """Pull account + positions + recent activities from Alpaca paper API.

    Returns an AccountSnapshot with raw_error set if the SDK is unavailable
    or the call fails — the monitor stays running and writes a degraded
    health file rather than crashing.
    """
    ts = datetime.utcnow().isoformat() + "Z"
    api_key = os.environ.get("ALPACA_API_KEY_PAPER", "")
    api_secret = os.environ.get("ALPACA_API_SECRET_PAPER", "")
    base_url = "https://paper-api.alpaca.markets"

    if not api_key or not api_secret:
        return AccountSnapshot(
            timestamp=ts, equity=0.0, cash=0.0, positions=[],
            fills_since_last=[],
            raw_error="ALPACA_API_KEY_PAPER / ALPACA_API_SECRET_PAPER not set",
        )

    # Try alpaca-py first, then alpaca-trade-api
    try:
        try:
            from alpaca.trading.client import TradingClient
            tc = TradingClient(api_key, api_secret, paper=True)
            acct = tc.get_account()
            poss = tc.get_all_positions()
            equity = float(acct.equity)
            cash = float(acct.cash)
            positions = [Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
            ) for p in poss]

            # Recent activities (fills) since last seen id
            try:
                from alpaca.trading.requests import GetAccountActivitiesRequest
                req = GetAccountActivitiesRequest(activity_types=["FILL"])
                acts = tc.get_account_activities(req)
            except Exception:
                acts = []
            fills = []
            for a in acts[:50]:
                aid = getattr(a, "id", None)
                if last_fill_id and aid == last_fill_id:
                    break
                fills.append({
                    "id": aid,
                    "symbol": getattr(a, "symbol", ""),
                    "side": getattr(a, "side", ""),
                    "qty": float(getattr(a, "qty", 0) or 0),
                    "price": float(getattr(a, "price", 0) or 0),
                    "transaction_time": str(getattr(a, "transaction_time", "")),
                })

            return AccountSnapshot(
                timestamp=ts, equity=equity, cash=cash,
                positions=positions, fills_since_last=fills,
            )
        except ImportError:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(api_key, api_secret, base_url=base_url)
            acct = api.get_account()
            poss = api.list_positions()
            equity = float(acct.equity)
            cash = float(acct.cash)
            positions = [Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
            ) for p in poss]
            acts = api.get_activities(activity_types="FILL")
            fills = []
            for a in acts[:50]:
                aid = getattr(a, "id", None)
                if last_fill_id and aid == last_fill_id:
                    break
                fills.append({
                    "id": aid,
                    "symbol": getattr(a, "symbol", ""),
                    "side": getattr(a, "side", ""),
                    "qty": float(getattr(a, "qty", 0) or 0),
                    "price": float(getattr(a, "price", 0) or 0),
                    "transaction_time": str(getattr(a, "transaction_time", "")),
                })

            return AccountSnapshot(
                timestamp=ts, equity=equity, cash=cash,
                positions=positions, fills_since_last=fills,
            )

    except Exception as e:
        return AccountSnapshot(
            timestamp=ts, equity=0.0, cash=0.0, positions=[],
            fills_since_last=[], raw_error=f"alpaca call failed: {e}",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Risk evaluation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RiskBreach:
    code: str
    severity: str   # "warning" | "critical"
    message: str
    value: float
    limit: float


def evaluate_risk(snap: AccountSnapshot,
                    cfg: MonitorConfig,
                    state: Dict[str, Any]) -> List[RiskBreach]:
    breaches: List[RiskBreach] = []
    if snap.raw_error:
        return breaches

    equity = snap.equity
    starting = cfg.starting_capital
    peak = max(state.get("equity_peak", starting), equity)
    state["equity_peak"] = peak

    # Drawdown vs peak
    if peak > 0:
        dd_pct = (peak - equity) / peak * 100.0
        if dd_pct >= cfg.max_drawdown_halt_pct:
            breaches.append(RiskBreach(
                code="dd_halt", severity="critical",
                message=f"Portfolio drawdown {dd_pct:.2f}% ≥ halt {cfg.max_drawdown_halt_pct:.1f}%",
                value=dd_pct, limit=cfg.max_drawdown_halt_pct,
            ))

    # Daily loss vs day-open equity
    today = datetime.utcnow().date().isoformat()
    day_open = state.get("day_open_equity", {}).get(today)
    if day_open is None:
        state.setdefault("day_open_equity", {})[today] = equity
        day_open = equity
    daily_loss_pct = (day_open - equity) / day_open * 100.0 if day_open > 0 else 0.0
    if daily_loss_pct >= cfg.max_daily_loss_pct:
        breaches.append(RiskBreach(
            code="daily_loss", severity="critical",
            message=f"Daily loss {daily_loss_pct:.2f}% ≥ {cfg.max_daily_loss_pct:.1f}%",
            value=daily_loss_pct, limit=cfg.max_daily_loss_pct,
        ))

    # Weekly loss
    week_key = datetime.utcnow().strftime("%G-W%V")
    week_open = state.get("week_open_equity", {}).get(week_key)
    if week_open is None:
        state.setdefault("week_open_equity", {})[week_key] = equity
        week_open = equity
    weekly_loss_pct = (week_open - equity) / week_open * 100.0 if week_open > 0 else 0.0
    if weekly_loss_pct >= cfg.max_weekly_loss_pct:
        breaches.append(RiskBreach(
            code="weekly_loss", severity="critical",
            message=f"Weekly loss {weekly_loss_pct:.2f}% ≥ {cfg.max_weekly_loss_pct:.1f}%",
            value=weekly_loss_pct, limit=cfg.max_weekly_loss_pct,
        ))

    # Position counts
    n_open = len(snap.positions)
    if n_open > cfg.max_open_positions_total:
        breaches.append(RiskBreach(
            code="too_many_positions", severity="warning",
            message=f"{n_open} open positions > limit {cfg.max_open_positions_total}",
            value=float(n_open), limit=float(cfg.max_open_positions_total),
        ))

    # Concentration
    if equity > 0:
        by_strat: Dict[str, float] = {}
        for p in snap.positions:
            by_strat[p.strategy_id] = by_strat.get(p.strategy_id, 0.0) + abs(p.market_value)
        for sid, mv in by_strat.items():
            wt = mv / equity * 100.0
            if wt > cfg.max_strategy_weight_pct:
                breaches.append(RiskBreach(
                    code="concentration", severity="warning",
                    message=f"Strategy {sid} weight {wt:.1f}% > {cfg.max_strategy_weight_pct:.1f}%",
                    value=wt, limit=cfg.max_strategy_weight_pct,
                ))

    return breaches


# ═══════════════════════════════════════════════════════════════════════════
# Telegram alerts
# ═══════════════════════════════════════════════════════════════════════════

def send_alert(message: str, severity: str = "info") -> None:
    """Send a Telegram alert via shared.telegram_alerts; fall back to logging."""
    try:
        from shared.telegram_alerts import send_telegram_alert
        send_telegram_alert(message, severity=severity)
        return
    except Exception:
        pass
    try:
        from compass.telegram_alerter import TelegramAlerter, Priority
        prio = {"info": Priority.INFO,
                "warning": Priority.WARNING,
                "critical": Priority.CRITICAL}.get(severity, Priority.INFO)
        TelegramAlerter().send(message, priority=prio)
        return
    except Exception:
        pass
    LOG.warning("[ALERT/%s] %s", severity.upper(), message)


# ═══════════════════════════════════════════════════════════════════════════
# Health file
# ═══════════════════════════════════════════════════════════════════════════

def write_health(snap: AccountSnapshot,
                   breaches: List[RiskBreach],
                   cfg: MonitorConfig) -> None:
    cfg.health_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "EXP-1900",
        "timestamp": snap.timestamp,
        "alpaca_ok": snap.raw_error is None,
        "alpaca_error": snap.raw_error,
        "equity": snap.equity,
        "cash": snap.cash,
        "starting_capital": cfg.starting_capital,
        "pnl_total": snap.equity - cfg.starting_capital,
        "pnl_total_pct": ((snap.equity / cfg.starting_capital) - 1.0) * 100
            if cfg.starting_capital > 0 else 0.0,
        "open_positions": len(snap.positions),
        "positions": [asdict(p) for p in snap.positions],
        "breaches": [asdict(b) for b in breaches],
        "interval_seconds": cfg.interval_seconds,
    }
    cfg.health_file.write_text(json.dumps(payload, indent=2, default=str))


def load_state(cfg: MonitorConfig) -> Dict[str, Any]:
    if cfg.state_file.exists():
        try:
            return json.loads(cfg.state_file.read_text())
        except Exception:
            pass
    return {}


def save_state(cfg: MonitorConfig, state: Dict[str, Any]) -> None:
    cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.state_file.write_text(json.dumps(state, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════════════════
# Main poll loop
# ═══════════════════════════════════════════════════════════════════════════

def daily_summary_due(cfg: MonitorConfig, state: Dict[str, Any]) -> bool:
    now = datetime.now()
    today = now.date().isoformat()
    if state.get("sent_summary_for_date") == today:
        return False
    try:
        hh, mm = [int(x) for x in cfg.daily_summary_time.split(":")]
    except Exception:
        return False
    return now.hour > hh or (now.hour == hh and now.minute >= mm)


def build_daily_summary(snap: AccountSnapshot, cfg: MonitorConfig,
                          state: Dict[str, Any]) -> str:
    today = datetime.utcnow().date().isoformat()
    day_open = state.get("day_open_equity", {}).get(today, cfg.starting_capital)
    day_pnl = snap.equity - day_open
    day_pct = (day_pnl / day_open * 100.0) if day_open > 0 else 0.0
    total_pnl = snap.equity - cfg.starting_capital
    total_pct = (total_pnl / cfg.starting_capital * 100.0) if cfg.starting_capital > 0 else 0.0
    lines = [
        "📊 *EXP-1900 North Star — Daily Summary*",
        f"Equity: ${snap.equity:,.0f}  (start ${cfg.starting_capital:,.0f})",
        f"Day P&L: {'🟢' if day_pnl >= 0 else '🔴'} ${day_pnl:,.0f} ({day_pct:+.2f}%)",
        f"Total P&L: ${total_pnl:,.0f} ({total_pct:+.2f}%)",
        f"Open positions: {len(snap.positions)}",
    ]
    return "\n".join(lines)


def poll_once(cfg: MonitorConfig, state: Dict[str, Any]) -> int:
    last_fill_id = state.get("last_fill_id")
    snap = fetch_account_snapshot(cfg, last_fill_id)

    if snap.raw_error:
        LOG.warning("snapshot error: %s", snap.raw_error)
        write_health(snap, [], cfg)
        return 1

    # New fills → entry/exit alerts
    if snap.fills_since_last:
        for f in snap.fills_since_last:
            msg = (f"💰 Fill: {f.get('side','?').upper()} "
                   f"{f.get('qty',0)} {f.get('symbol','?')} "
                   f"@ ${f.get('price',0):.2f}")
            send_alert(msg, severity="info")
        # advance pointer
        first = snap.fills_since_last[0].get("id")
        if first:
            state["last_fill_id"] = first

    # Risk evaluation
    breaches = evaluate_risk(snap, cfg, state)
    for b in breaches:
        prefix = "🚨" if b.severity == "critical" else "⚠️"
        send_alert(f"{prefix} {b.code}: {b.message}", severity=b.severity)

    # Daily summary
    if daily_summary_due(cfg, state):
        send_alert(build_daily_summary(snap, cfg, state), severity="info")
        state["sent_summary_for_date"] = datetime.now().date().isoformat()

    write_health(snap, breaches, cfg)
    save_state(cfg, state)

    LOG.info("equity=$%.0f open=%d breaches=%d new_fills=%d",
              snap.equity, len(snap.positions), len(breaches),
              len(snap.fills_since_last))
    return 0


def run_loop(cfg: MonitorConfig) -> int:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(cfg)
    LOG.info("EXP-1900 monitor starting interval=%ds", cfg.interval_seconds)
    if cfg.once:
        return poll_once(cfg, state)
    while True:
        try:
            poll_once(cfg, state)
        except KeyboardInterrupt:
            LOG.info("monitor stopped by user")
            return 0
        except Exception as e:
            LOG.exception("poll loop error: %s", e)
            send_alert(f"🚨 Monitor exception: {e}", severity="critical")
        time.sleep(cfg.interval_seconds)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-1900 North Star paper monitor")
    p.add_argument("--config", default=str(ROOT / "configs" / "north_star_paper.yaml"))
    p.add_argument("--interval", type=int, default=None,
                    help="Override poll interval in seconds")
    p.add_argument("--once", action="store_true", help="Single poll then exit")
    p.add_argument("--foreground", action="store_true",
                    help="Log to stderr in addition to file")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.interval is not None:
        cfg.interval_seconds = args.interval
    cfg.once = args.once
    cfg.foreground = args.foreground

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [
        logging.FileHandler(cfg.log_dir / "monitor.log"),
    ]
    if args.foreground:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )
    return run_loop(cfg)


if __name__ == "__main__":
    sys.exit(main())
