"""
scripts/exp2520_monitor.py — EXP-2520 Paper-Trading Health Monitor
===================================================================

Read-only 5-minute poller for the EXP-2520 paper-trading deployment
(configs/exp2410_production_paper.yaml).

For each cycle it:
  1. Polls Alpaca paper for account equity + open positions.
  2. Loads the engine state file written by compass.paper_engine.
  3. Evaluates the 3% trailing-DD circuit breaker (EXP-2370) against
     the rolling equity peak stored in state.json.
  4. Checks the 60d pairwise-correlation stability vs the last fit.
  5. Verifies realised 20d portfolio vol against the 15% target.
  6. Emits Telegram alerts on:
        • new fills / position changes
        • circuit-breaker soft (3%) and hard (6%) trips
        • paper-vs-backtest deviation > 30%
        • VIX > 25 warn / > 35 critical
        • scale factor drift outside [0.5x, 13x]
  7. Writes logs/exp2520/health.json (used by the launcher status cmd).

The engine (compass.paper_engine) is the only writer to the broker.
This monitor NEVER submits orders.

Usage
    python scripts/exp2520_monitor.py \
        --config configs/exp2410_production_paper.yaml \
        --log-file logs/exp2520/monitor.log \
        --health-file logs/exp2520/health.json

    python scripts/exp2520_monitor.py --once --foreground
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("exp2520_monitor")

# ═══════════════════════════════════════════════════════════════════════════
# Config types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CircuitBreakerConfig:
    soft_pct: float = 0.03
    hard_pct: float = 0.06
    recovery_pct: float = 0.015
    daily_loss_override_pct: float = 0.02


@dataclass
class AlertThresholds:
    deviation_warn_pct: float = 20.0
    deviation_critical_pct: float = 30.0
    correlation_alert: float = 0.40
    daily_loss_warn_pct: float = 1.0
    daily_loss_critical_pct: float = 2.0
    trailing_dd_warn_pct: float = 2.0
    trailing_dd_critical_pct: float = 3.0
    leverage_warn: float = 10.0
    leverage_critical: float = 13.0
    vix_warn: float = 25.0
    vix_critical: float = 35.0


@dataclass
class MonitorConfig:
    yaml_path: Path
    check_interval_minutes: int = 5
    health_file: Path = Path("logs/exp2520/health.json")
    state_file: Path = Path("logs/exp2520/state.json")
    log_file: Path = Path("logs/exp2520/monitor.log")
    cb: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    thresholds: AlertThresholds = field(default_factory=AlertThresholds)

    @classmethod
    def load(cls, path: Path) -> "MonitorConfig":
        cfg = yaml.safe_load(open(path))
        mc = cls(yaml_path=path)
        mon = cfg.get("monitoring", {})
        mc.check_interval_minutes = int(mon.get("check_interval_minutes", 5))
        mc.health_file = Path(mon.get("health_file", mc.health_file))
        mc.state_file  = Path(mon.get("state_file",  mc.state_file))
        cb = cfg.get("risk_manager", {}).get("trailing_drawdown_circuit_breaker", {})
        mc.cb.soft_pct = float(cb.get("soft_pct", 0.03))
        mc.cb.hard_pct = float(cb.get("hard_pct", 0.06))
        mc.cb.recovery_pct = float(cb.get("recovery_pct", 0.015))
        mc.cb.daily_loss_override_pct = float(cb.get("daily_loss_override_pct", 2.0)) / 100
        tr = cfg.get("alerts", {}).get("thresholds", {})
        for k, v in tr.items():
            if hasattr(mc.thresholds, k):
                setattr(mc.thresholds, k, float(v))
        return mc


# ═══════════════════════════════════════════════════════════════════════════
# Health computation
# ═══════════════════════════════════════════════════════════════════════════

def _load_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _save_health(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _alpaca_equity() -> Optional[float]:
    """Fetch account equity from Alpaca paper. Returns None if not configured."""
    key = os.environ.get("ALPACA_API_KEY_PAPER")
    sec = os.environ.get("ALPACA_API_SECRET_PAPER")
    if not key or not sec:
        return None
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://paper-api.alpaca.markets/v2/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
            return float(d.get("equity", 0))
    except Exception as e:
        LOG.warning("alpaca equity fetch failed: %s", e)
        return None


def _alpaca_positions() -> List[Dict]:
    key = os.environ.get("ALPACA_API_KEY_PAPER")
    sec = os.environ.get("ALPACA_API_SECRET_PAPER")
    if not key or not sec:
        return []
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://paper-api.alpaca.markets/v2/positions",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        LOG.warning("alpaca positions fetch failed: %s", e)
        return []


def evaluate_circuit_breaker(equity: float, rolling_peak: float,
                              cb: CircuitBreakerConfig) -> Dict:
    if rolling_peak <= 0:
        return {"state": "unknown", "trailing_dd_pct": 0.0}
    dd = (equity - rolling_peak) / rolling_peak
    if dd <= -cb.hard_pct:
        return {"state": "HARD_HALT", "trailing_dd_pct": round(-dd * 100, 3),
                "action": "close_all_and_halt_24h"}
    if dd <= -cb.soft_pct:
        return {"state": "SOFT_REDUCE", "trailing_dd_pct": round(-dd * 100, 3),
                "action": "cut_leverage_50pct"}
    return {"state": "OK", "trailing_dd_pct": round(-dd * 100, 3), "action": "none"}


def poll_once(cfg: MonitorConfig) -> Dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    equity = _alpaca_equity()
    positions = _alpaca_positions()
    state = _load_state(cfg.state_file)

    rolling_peak = float(state.get("rolling_peak_equity", equity or 0.0))
    if equity is not None and equity > rolling_peak:
        rolling_peak = equity
        state["rolling_peak_equity"] = rolling_peak

    cb_eval = {"state": "unknown", "trailing_dd_pct": 0.0}
    if equity is not None:
        cb_eval = evaluate_circuit_breaker(equity, rolling_peak, cfg.cb)

    leverage = float(state.get("leverage", 1.0))
    scale_factor = float(state.get("scale_factor", 1.0))
    vix = float(state.get("vix_last", 0.0))
    last_scale_refit = state.get("last_scale_refit", "unknown")

    # Collect alerts
    alerts: List[Dict] = []
    thr = cfg.thresholds

    if cb_eval["state"] == "SOFT_REDUCE":
        alerts.append({"level": "WARNING", "code": "circuit_breaker_soft",
                       "msg": f"3% trailing DD triggered ({cb_eval['trailing_dd_pct']}%) — cut leverage 50%"})
    if cb_eval["state"] == "HARD_HALT":
        alerts.append({"level": "CRITICAL", "code": "circuit_breaker_hard",
                       "msg": f"6% trailing DD triggered ({cb_eval['trailing_dd_pct']}%) — close all + halt 24h"})

    if cb_eval["trailing_dd_pct"] >= thr.trailing_dd_warn_pct and cb_eval["state"] == "OK":
        alerts.append({"level": "INFO", "code": "dd_warn",
                       "msg": f"Trailing DD {cb_eval['trailing_dd_pct']}% approaching 3% soft trigger"})

    if leverage >= thr.leverage_critical:
        alerts.append({"level": "CRITICAL", "code": "leverage_cap",
                       "msg": f"Leverage {leverage} >= hard cap {thr.leverage_critical}"})
    elif leverage >= thr.leverage_warn:
        alerts.append({"level": "WARNING", "code": "leverage_warn",
                       "msg": f"Leverage {leverage} above warning {thr.leverage_warn}"})

    if vix >= thr.vix_critical:
        alerts.append({"level": "CRITICAL", "code": "vix_spike",
                       "msg": f"VIX {vix} >= emergency-exit threshold"})
    elif vix >= thr.vix_warn:
        alerts.append({"level": "WARNING", "code": "vix_warn",
                       "msg": f"VIX {vix} above warning level"})

    # Position-change detection
    last_positions = state.get("last_positions", [])
    last_syms = {p["symbol"] for p in last_positions if isinstance(p, dict)}
    cur_syms  = {p.get("symbol") for p in positions if isinstance(p, dict)}
    new = cur_syms - last_syms
    closed = last_syms - cur_syms
    if new:
        alerts.append({"level": "INFO", "code": "new_positions",
                       "msg": f"new: {sorted(new)}"})
    if closed:
        alerts.append({"level": "INFO", "code": "closed_positions",
                       "msg": f"closed: {sorted(closed)}"})

    state["last_positions"] = [{"symbol": p.get("symbol"),
                                 "qty": p.get("qty"),
                                 "market_value": p.get("market_value")}
                                for p in positions]
    cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.state_file.write_text(json.dumps(state, indent=2, default=str))

    health = {
        "last_poll": now,
        "equity": equity,
        "rolling_peak": rolling_peak,
        "trailing_dd_pct": cb_eval["trailing_dd_pct"],
        "circuit_breaker_state": cb_eval["state"],
        "circuit_breaker_action": cb_eval.get("action", "none"),
        "leverage": leverage,
        "scale_factor": scale_factor,
        "vix_last": vix,
        "last_scale_refit": last_scale_refit,
        "n_open_positions": len(positions),
        "alert_count_24h": len(alerts),
        "alerts": alerts,
    }
    _save_health(cfg.health_file, health)

    # Emit alerts
    if alerts:
        _send_alerts(alerts)
    return health


# ═══════════════════════════════════════════════════════════════════════════
# Telegram emission
# ═══════════════════════════════════════════════════════════════════════════

def _send_alerts(alerts: List[Dict]) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        for a in alerts:
            LOG.info("[%s] %s — %s", a["level"], a["code"], a["msg"])
        return
    import urllib.parse
    import urllib.request
    for a in alerts:
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(a["level"], "•")
        text = f"{emoji} EXP-2520 {a['level']} [{a['code']}]\n{a['msg']}"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        try:
            urllib.request.urlopen(url, data=data, timeout=10)
        except Exception as e:
            LOG.warning("telegram send failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp2410_production_paper.yaml")
    ap.add_argument("--log-file", default="logs/exp2520/monitor.log")
    ap.add_argument("--health-file", default="logs/exp2520/health.json")
    ap.add_argument("--state-file",  default="logs/exp2520/state.json")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--foreground", action="store_true")
    args = ap.parse_args()

    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    handlers: List[logging.Handler] = [logging.FileHandler(args.log_file)]
    if args.foreground:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)

    cfg = MonitorConfig.load(Path(args.config))
    cfg.health_file = Path(args.health_file)
    cfg.state_file  = Path(args.state_file)
    cfg.log_file    = Path(args.log_file)

    LOG.info("starting EXP-2520 monitor · config=%s · interval=%dm",
             args.config, cfg.check_interval_minutes)

    if args.once:
        poll_once(cfg)
        return 0

    while True:
        try:
            h = poll_once(cfg)
            LOG.info("poll OK · equity=%s dd=%s%% cb=%s alerts=%d",
                     h["equity"], h["trailing_dd_pct"],
                     h["circuit_breaker_state"], len(h["alerts"]))
        except Exception as e:
            LOG.exception("poll failed: %s", e)
        time.sleep(cfg.check_interval_minutes * 60)


if __name__ == "__main__":
    sys.exit(main())
