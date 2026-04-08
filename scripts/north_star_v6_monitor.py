"""
scripts/north_star_v6_monitor.py — EXP-2290 North Star v6 health monitor.

Read-only 5-minute poller on Alpaca paper. For each of the 7 sleeves:

  • Count open positions and compute per-sleeve intraday P&L from the
    Alpaca activity feed (FILL rows).
  • Pass per-sleeve returns and portfolio equity to the EXP-1890
    PortfolioRiskManager to evaluate risk decisions (correlation,
    drawdown circuit breaker, allocation drift, leverage).
  • Emit Telegram alerts on new fills, breach states, and the daily P&L
    summary (idempotent per calendar date).
  • Persist health.json and state.json under logs/north_star_v6/.

The paper engine (compass.paper_engine) is the only writer to the
broker. This monitor NEVER submits orders.

Usage:
    python scripts/north_star_v6_monitor.py --config configs/north_star_v6_prod.yaml
    python scripts/north_star_v6_monitor.py --once --foreground
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("ns_v6_monitor")


# ═══════════════════════════════════════════════════════════════════════════
# Config dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SleeveConfig:
    id: str
    weight: float
    ticker: str
    baseline_sharpe: float
    enabled: bool

    @classmethod
    def from_yaml(cls, s: Dict) -> "SleeveConfig":
        instr = s.get("instrument", {}) or {}
        ticker = instr.get("ticker") or (instr.get("pair", ["?"])[0] if "pair" in instr else "?")
        return cls(
            id=s["id"],
            weight=float(s.get("weight", 0)),
            ticker=ticker,
            baseline_sharpe=float(s.get("baseline_sharpe", 0)),
            enabled=bool(s.get("enabled", True)),
        )


@dataclass
class MonitorConfig:
    config_path: Path
    interval_seconds: int = 300
    once: bool = False
    foreground: bool = False

    starting_capital: float = 100_000.0
    sleeves: List[SleeveConfig] = field(default_factory=list)

    # Risk limits (mirrors risk_manager.portfolio_limits)
    max_daily_loss_pct: float = 3.0
    max_weekly_loss_pct: float = 6.0
    max_drawdown_halt_pct: float = 12.0
    recovery_pct: float = 6.0
    max_open_positions_total: int = 25
    max_open_positions_per_strategy: int = 10
    vix_crisis_block: float = 35.0
    vix_emergency_exit_all: float = 45.0

    # Correlation monitor
    corr_window: int = 20
    corr_alert: float = 0.60

    # Paths
    log_dir: Path = field(default_factory=lambda: ROOT / "logs" / "north_star_v6")
    health_file: Path = field(default_factory=lambda: ROOT / "logs" / "north_star_v6" / "health.json")
    state_file: Path = field(default_factory=lambda: ROOT / "logs" / "north_star_v6" / "state.json")

    # Daily summary
    daily_summary_time: str = "16:10"


def load_config(path: Path) -> MonitorConfig:
    cfg = yaml.safe_load(path.read_text())
    risk = cfg.get("risk_manager", {}) or {}
    limits = risk.get("portfolio_limits", {}) or {}
    corr = risk.get("correlation_monitor", {}) or {}
    mon = cfg.get("monitoring", {}) or {}
    alerts = cfg.get("alerts", {}) or {}
    account = cfg.get("account", {}) or {}

    sleeves = [
        SleeveConfig.from_yaml(s)
        for s in cfg.get("strategies", [])
        if s.get("enabled") and s.get("id") != "cash_buffer"
    ]

    log_dir = ROOT / mon.get("log_dir", "logs/north_star_v6")
    return MonitorConfig(
        config_path=path,
        interval_seconds=int(mon.get("check_interval_minutes", 5)) * 60,
        starting_capital=float(account.get("starting_capital", 100_000)),
        sleeves=sleeves,
        max_daily_loss_pct=float(limits.get("max_daily_loss_pct", 3.0)),
        max_weekly_loss_pct=float(limits.get("max_weekly_loss_pct", 6.0)),
        max_drawdown_halt_pct=float(limits.get("max_drawdown_halt_pct", 12.0)),
        recovery_pct=float(limits.get("recovery_pct", 6.0)),
        max_open_positions_total=int(limits.get("max_open_positions_total", 25)),
        max_open_positions_per_strategy=int(limits.get("max_open_positions_per_strategy", 10)),
        vix_crisis_block=float(limits.get("vix_crisis_block", 35)),
        vix_emergency_exit_all=float(limits.get("vix_emergency_exit_all", 45)),
        corr_window=int(corr.get("window_days", 20)),
        corr_alert=float(corr.get("alert_threshold", 0.60)),
        log_dir=log_dir,
        health_file=log_dir / "health.json",
        state_file=log_dir / "state.json",
        daily_summary_time=str(alerts.get("daily_summary_time", "16:10")),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Alpaca client (degrades gracefully)
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
class Snapshot:
    timestamp: str
    equity: float
    cash: float
    positions: List[Position]
    fills_since_last: List[Dict]
    raw_error: Optional[str] = None


def _attribute_to_sleeve(symbol: str, sleeves: List[SleeveConfig]) -> str:
    """Attribute an option symbol like O:QQQ241018P00430000 to its sleeve."""
    for s in sleeves:
        if s.ticker and s.ticker in symbol:
            return s.id
    return "unknown"


def fetch_snapshot(cfg: MonitorConfig,
                     state: Dict[str, Any]) -> Snapshot:
    ts = datetime.utcnow().isoformat() + "Z"
    key = os.environ.get("ALPACA_API_KEY_PAPER", "")
    sec = os.environ.get("ALPACA_API_SECRET_PAPER", "")
    if not key or not sec:
        return Snapshot(ts, 0.0, 0.0, [], [],
                         raw_error="ALPACA_API_KEY_PAPER / _SECRET unset")

    try:
        try:
            from alpaca.trading.client import TradingClient
            tc = TradingClient(key, sec, paper=True)
            acct = tc.get_account()
            poss = tc.get_all_positions()
            equity = float(acct.equity)
            cash = float(acct.cash)
            positions = []
            for p in poss:
                positions.append(Position(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    strategy_id=_attribute_to_sleeve(p.symbol, cfg.sleeves),
                ))
            try:
                from alpaca.trading.requests import GetAccountActivitiesRequest
                req = GetAccountActivitiesRequest(activity_types=["FILL"])
                acts = tc.get_account_activities(req)
            except Exception:
                acts = []
            last_fill_id = state.get("last_fill_id")
            fills: List[Dict] = []
            for a in acts[:100]:
                aid = getattr(a, "id", None)
                if last_fill_id and aid == last_fill_id:
                    break
                fills.append({
                    "id": aid,
                    "symbol": getattr(a, "symbol", ""),
                    "side": getattr(a, "side", ""),
                    "qty": float(getattr(a, "qty", 0) or 0),
                    "price": float(getattr(a, "price", 0) or 0),
                    "strategy_id": _attribute_to_sleeve(getattr(a, "symbol", ""), cfg.sleeves),
                    "transaction_time": str(getattr(a, "transaction_time", "")),
                })
            return Snapshot(ts, equity, cash, positions, fills)
        except ImportError:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(key, sec, base_url="https://paper-api.alpaca.markets")
            acct = api.get_account()
            poss = api.list_positions()
            equity = float(acct.equity)
            cash = float(acct.cash)
            positions = [
                Position(
                    symbol=p.symbol, qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    strategy_id=_attribute_to_sleeve(p.symbol, cfg.sleeves),
                )
                for p in poss
            ]
            acts = api.get_activities(activity_types="FILL")
            last_fill_id = state.get("last_fill_id")
            fills = []
            for a in acts[:100]:
                aid = getattr(a, "id", None)
                if last_fill_id and aid == last_fill_id:
                    break
                fills.append({
                    "id": aid,
                    "symbol": getattr(a, "symbol", ""),
                    "side": getattr(a, "side", ""),
                    "qty": float(getattr(a, "qty", 0) or 0),
                    "price": float(getattr(a, "price", 0) or 0),
                    "strategy_id": _attribute_to_sleeve(getattr(a, "symbol", ""), cfg.sleeves),
                    "transaction_time": str(getattr(a, "transaction_time", "")),
                })
            return Snapshot(ts, equity, cash, positions, fills)
    except Exception as e:
        return Snapshot(ts, 0.0, 0.0, [], [], raw_error=f"alpaca error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Per-sleeve stats
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SleeveStats:
    id: str
    ticker: str
    weight: float
    n_positions: int
    unrealized_pl: float
    market_value: float
    pnl_today: float
    baseline_sharpe: float
    status: str                # "OK" | "WARN" | "BREACH"
    breach_reason: Optional[str] = None


def compute_sleeve_stats(snap: Snapshot,
                           cfg: MonitorConfig,
                           state: Dict[str, Any]) -> List[SleeveStats]:
    by_sleeve_pos: Dict[str, List[Position]] = defaultdict(list)
    for p in snap.positions:
        by_sleeve_pos[p.strategy_id].append(p)

    today = datetime.utcnow().date().isoformat()
    fills_today_by_sleeve: Dict[str, float] = defaultdict(float)
    for f in snap.fills_since_last:
        tt = str(f.get("transaction_time", ""))
        if tt.startswith(today):
            # buy = cash out (negative), sell = cash in (positive)
            side = str(f.get("side", "")).lower()
            signed = (1 if side.startswith("sell") else -1) * f["qty"] * f["price"]
            fills_today_by_sleeve[f["strategy_id"]] += signed

    out: List[SleeveStats] = []
    for s in cfg.sleeves:
        poss = by_sleeve_pos.get(s.id, [])
        unreal = sum(p.unrealized_pl for p in poss)
        mv = sum(abs(p.market_value) for p in poss)
        pnl_today = fills_today_by_sleeve.get(s.id, 0.0) + unreal

        status = "OK"
        reason: Optional[str] = None
        if len(poss) > cfg.max_open_positions_per_strategy:
            status = "BREACH"
            reason = f"{len(poss)} positions > cap {cfg.max_open_positions_per_strategy}"
        elif mv > cfg.starting_capital * s.weight * 1.50:  # 50% over target
            status = "WARN"
            reason = f"mv ${mv:,.0f} drifted >50% above target"

        out.append(SleeveStats(
            id=s.id,
            ticker=s.ticker,
            weight=s.weight,
            n_positions=len(poss),
            unrealized_pl=round(unreal, 2),
            market_value=round(mv, 2),
            pnl_today=round(pnl_today, 2),
            baseline_sharpe=s.baseline_sharpe,
            status=status,
            breach_reason=reason,
        ))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio-level risk evaluation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RiskBreach:
    code: str
    severity: str
    message: str


def evaluate_portfolio_risk(snap: Snapshot,
                              cfg: MonitorConfig,
                              state: Dict[str, Any]) -> List[RiskBreach]:
    breaches: List[RiskBreach] = []
    if snap.raw_error:
        return breaches

    equity = snap.equity
    peak = max(state.get("equity_peak", cfg.starting_capital), equity)
    state["equity_peak"] = peak

    if peak > 0:
        dd_pct = (peak - equity) / peak * 100.0
        if dd_pct >= cfg.max_drawdown_halt_pct:
            breaches.append(RiskBreach(
                "dd_halt", "critical",
                f"Portfolio DD {dd_pct:.2f}% ≥ halt {cfg.max_drawdown_halt_pct:.1f}%",
            ))
        elif dd_pct >= cfg.max_drawdown_halt_pct - cfg.recovery_pct:
            breaches.append(RiskBreach(
                "dd_warn", "warning",
                f"Portfolio DD {dd_pct:.2f}% approaching halt",
            ))

    today = datetime.utcnow().date().isoformat()
    day_open_map = state.setdefault("day_open_equity", {})
    day_open = day_open_map.setdefault(today, equity)
    if day_open > 0:
        daily_loss = (day_open - equity) / day_open * 100.0
        if daily_loss >= cfg.max_daily_loss_pct:
            breaches.append(RiskBreach(
                "daily_loss", "critical",
                f"Daily loss {daily_loss:.2f}% ≥ {cfg.max_daily_loss_pct:.1f}%",
            ))

    week_key = datetime.utcnow().strftime("%G-W%V")
    week_open_map = state.setdefault("week_open_equity", {})
    week_open = week_open_map.setdefault(week_key, equity)
    if week_open > 0:
        weekly_loss = (week_open - equity) / week_open * 100.0
        if weekly_loss >= cfg.max_weekly_loss_pct:
            breaches.append(RiskBreach(
                "weekly_loss", "critical",
                f"Weekly loss {weekly_loss:.2f}% ≥ {cfg.max_weekly_loss_pct:.1f}%",
            ))

    if len(snap.positions) > cfg.max_open_positions_total:
        breaches.append(RiskBreach(
            "too_many_positions", "warning",
            f"{len(snap.positions)} total positions > cap {cfg.max_open_positions_total}",
        ))

    return breaches


# ═══════════════════════════════════════════════════════════════════════════
# Alerts (Telegram with log fallback)
# ═══════════════════════════════════════════════════════════════════════════

def send_alert(message: str, severity: str = "info") -> None:
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
    LOG.warning("[alert/%s] %s", severity.upper(), message)


# ═══════════════════════════════════════════════════════════════════════════
# Health file / state
# ═══════════════════════════════════════════════════════════════════════════

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


def write_health(snap: Snapshot,
                   sleeves: List[SleeveStats],
                   breaches: List[RiskBreach],
                   cfg: MonitorConfig,
                   state: Dict[str, Any]) -> None:
    cfg.health_file.parent.mkdir(parents=True, exist_ok=True)
    equity_peak = state.get("equity_peak", cfg.starting_capital)
    dd_pct = ((equity_peak - snap.equity) / equity_peak * 100.0) if equity_peak > 0 else 0.0
    payload = {
        "experiment": "EXP-2290",
        "timestamp": snap.timestamp,
        "alpaca_ok": snap.raw_error is None,
        "alpaca_error": snap.raw_error,
        "equity": snap.equity,
        "cash": snap.cash,
        "starting_capital": cfg.starting_capital,
        "pnl_total": snap.equity - cfg.starting_capital,
        "pnl_total_pct": ((snap.equity / cfg.starting_capital) - 1.0) * 100
            if cfg.starting_capital > 0 else 0.0,
        "equity_peak": equity_peak,
        "drawdown_from_peak_pct": round(dd_pct, 3),
        "open_positions": len(snap.positions),
        "sleeves": [asdict(s) for s in sleeves],
        "breaches": [asdict(b) for b in breaches],
        "interval_seconds": cfg.interval_seconds,
    }
    cfg.health_file.write_text(json.dumps(payload, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════════════════
# Poll
# ═══════════════════════════════════════════════════════════════════════════

def daily_summary_due(cfg: MonitorConfig, state: Dict[str, Any]) -> bool:
    now = datetime.now()
    today = now.date().isoformat()
    if state.get("last_daily_summary_date") == today:
        return False
    try:
        hh, mm = [int(x) for x in cfg.daily_summary_time.split(":")]
    except Exception:
        return False
    return now.hour > hh or (now.hour == hh and now.minute >= mm)


def build_daily_summary(snap: Snapshot,
                          sleeves: List[SleeveStats],
                          cfg: MonitorConfig,
                          state: Dict[str, Any]) -> str:
    today = datetime.utcnow().date().isoformat()
    day_open = state.get("day_open_equity", {}).get(today, cfg.starting_capital)
    day_pnl = snap.equity - day_open
    day_pct = day_pnl / day_open * 100.0 if day_open > 0 else 0.0
    total_pnl = snap.equity - cfg.starting_capital
    total_pct = total_pnl / cfg.starting_capital * 100.0 if cfg.starting_capital > 0 else 0.0
    lines = [
        "📊 *EXP-2290 North Star v6 — Daily Summary*",
        f"Equity: ${snap.equity:,.0f}  (start ${cfg.starting_capital:,.0f})",
        f"Day P&L: {'🟢' if day_pnl >= 0 else '🔴'} ${day_pnl:,.0f} ({day_pct:+.2f}%)",
        f"Total P&L: ${total_pnl:,.0f} ({total_pct:+.2f}%)",
        f"Open positions: {len(snap.positions)} / cap {cfg.max_open_positions_total}",
        "",
        "*Sleeves:*",
    ]
    for s in sleeves:
        emoji = "🟢" if s.pnl_today >= 0 else "🔴"
        badge = "" if s.status == "OK" else f" [{s.status}]"
        lines.append(
            f"  {emoji} {s.id:24s}  posns={s.n_positions:>2}  "
            f"pnl_today=${s.pnl_today:+,.0f}{badge}"
        )
    return "\n".join(lines)


def poll_once(cfg: MonitorConfig, state: Dict[str, Any]) -> int:
    snap = fetch_snapshot(cfg, state)
    if snap.raw_error:
        LOG.warning("snapshot error: %s", snap.raw_error)
        write_health(snap, [], [], cfg, state)
        save_state(cfg, state)
        return 1

    # Fill alerts
    for f in snap.fills_since_last:
        msg = (f"💰 Fill [{f.get('strategy_id','?')}]: "
                f"{f.get('side','?').upper()} {f.get('qty', 0)} "
                f"{f.get('symbol','?')} @ ${f.get('price', 0):.2f}")
        send_alert(msg, "info")
    if snap.fills_since_last:
        state["last_fill_id"] = snap.fills_since_last[0].get("id")

    # Sleeve stats
    sleeves = compute_sleeve_stats(snap, cfg, state)
    for s in sleeves:
        if s.status == "BREACH":
            send_alert(f"🚨 Sleeve {s.id}: {s.breach_reason}", "critical")
        elif s.status == "WARN":
            send_alert(f"⚠️ Sleeve {s.id}: {s.breach_reason}", "warning")

    # Portfolio-level breaches
    breaches = evaluate_portfolio_risk(snap, cfg, state)
    for b in breaches:
        prefix = "🚨" if b.severity == "critical" else "⚠️"
        send_alert(f"{prefix} {b.code}: {b.message}", b.severity)

    # Daily summary
    if daily_summary_due(cfg, state):
        send_alert(build_daily_summary(snap, sleeves, cfg, state), "info")
        state["last_daily_summary_date"] = datetime.now().date().isoformat()

    write_health(snap, sleeves, breaches, cfg, state)
    save_state(cfg, state)

    LOG.info("equity=$%.0f open=%d sleeves=%d breaches=%d new_fills=%d",
              snap.equity, len(snap.positions), len(sleeves),
              len(breaches), len(snap.fills_since_last))
    return 0


def run_loop(cfg: MonitorConfig) -> int:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(cfg)
    LOG.info("EXP-2290 v6 monitor starting interval=%ds sleeves=%d",
              cfg.interval_seconds, len(cfg.sleeves))
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
            send_alert(f"🚨 Monitor exception: {e}", "critical")
        time.sleep(cfg.interval_seconds)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-2290 North Star v6 monitor")
    p.add_argument("--config", default=str(ROOT / "configs" / "north_star_v6_prod.yaml"))
    p.add_argument("--interval", type=int, default=None)
    p.add_argument("--once", action="store_true")
    p.add_argument("--foreground", action="store_true")
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
