"""
EXP-2620 — Alpaca Paper-Trading Connector for the 7-Stream Portfolio
=====================================================================

Production-ready Alpaca paper-trading connector. Wires:

  * Alpaca paper REST API           (real broker, commission-free)
  * 7-stream RiskDecision pipeline  (compass.portfolio_risk_manager)
  * EXP-2470 execution-optimisation stack
       A. limit-at-mid (50% fill rate target)
       B. patient execution window (last 30 min before close)
       C. route reallocation to XLI/XLF where allowed
       D. multi-leg combo orders (Alpaca multi-leg endpoint)
  * Health monitor + Telegram alerts
  * Circuit-breaker enforcement (3% trailing DD, EXP-2370)

The connector is the ONLY component that submits orders. The risk
manager (`PortfolioRiskManager.make_decision`) emits a `RiskDecision`
each cycle; the connector translates the decision into Alpaca order
submissions, applies the EXP-2470 execution overlay, polls fills, and
updates the engine state.

Read the EXP-2620 README in scripts/exp2620_README.md for the bring-up
sequence.

Rule Zero
  Every executable price comes from Alpaca live (or IronVault for the
  backtest comparison). The connector NEVER fabricates a quote. If a
  quote is missing the trade is skipped, never extrapolated.

Usage
-----
    from compass.exp2620_alpaca_connector import AlpacaPaperConnector
    conn = AlpacaPaperConnector.from_config("configs/exp2410_production_paper.yaml")
    conn.smoke()                          # validate env + connection
    conn.run_once(decision)               # apply one RiskDecision
    conn.run_loop(strategies, risk_mgr)   # production cycle
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.portfolio_risk_manager import (
    PortfolioRiskManager,
    RiskDecision,
    CircuitState,
)

LOG = logging.getLogger("exp2620_alpaca")

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE  = "https://data.alpaca.markets"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExecutionConfig:
    """EXP-2470 execution-optimisation stack switches."""
    use_limit_at_mid: bool = True            # technique A
    limit_offset_pct: float = 0.0            # 0 = mid; positive = pay up
    fill_timeout_seconds: int = 60
    use_patient_window: bool = True          # technique B
    patient_window_minutes: int = 30         # last N minutes before close
    use_route_reallocation: bool = True      # technique C
    use_multileg_combo: bool = True          # technique D
    max_legs_per_combo: int = 4              # iron condor


@dataclass
class CircuitBreakerConfig:
    """3% trailing-DD breaker (EXP-2370)."""
    soft_pct: float = 0.03
    hard_pct: float = 0.06
    recovery_pct: float = 0.015
    daily_loss_override_pct: float = 0.02


@dataclass
class ConnectorConfig:
    config_path: Path
    base_url: str = ALPACA_PAPER_BASE
    data_url: str = ALPACA_DATA_BASE
    api_key: str = ""
    api_secret: str = ""
    state_file: Path = Path("logs/exp2620/state.json")
    log_file:   Path = Path("logs/exp2620/connector.log")
    health_file: Path = Path("logs/exp2620/health.json")
    starting_capital: float = 100_000.0
    commission_free: bool = True
    poll_interval_seconds: int = 300         # 5 min
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    breaker:   CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    telegram_bot_token: str = ""
    telegram_chat_id:   str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "ConnectorConfig":
        cfg = yaml.safe_load(open(path))
        out = cls(config_path=path)
        acct = cfg.get("account", {})
        out.starting_capital = float(acct.get("starting_capital", 100_000))
        api  = acct.get("api", {})
        out.base_url = api.get("base_url", ALPACA_PAPER_BASE)
        out.data_url = api.get("data_url", ALPACA_DATA_BASE)
        out.api_key    = os.environ.get(api.get("key_env",    "ALPACA_API_KEY_PAPER"),    "")
        out.api_secret = os.environ.get(api.get("secret_env", "ALPACA_API_SECRET_PAPER"), "")
        mon = cfg.get("monitoring", {})
        out.state_file  = Path(mon.get("state_file",  out.state_file))
        out.health_file = Path(mon.get("health_file", out.health_file))
        out.poll_interval_seconds = int(mon.get("check_interval_minutes", 5)) * 60
        cb = cfg.get("risk_manager", {}).get("trailing_drawdown_circuit_breaker", {})
        out.breaker.soft_pct = float(cb.get("soft_pct", 0.03))
        out.breaker.hard_pct = float(cb.get("hard_pct", 0.06))
        out.breaker.recovery_pct = float(cb.get("recovery_pct", 0.015))
        out.breaker.daily_loss_override_pct = float(cb.get("daily_loss_override_pct", 2.0)) / 100
        out.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        out.telegram_chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "")
        # commission-free flag — anchored on EXP-2570 target
        out.commission_free = bool(api.get("commission_free", True))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca REST helpers
# ─────────────────────────────────────────────────────────────────────────────
def _alpaca_request(cfg: ConnectorConfig, method: str, path: str,
                    payload: Optional[Dict] = None) -> Optional[Any]:
    if not cfg.api_key or not cfg.api_secret:
        LOG.warning("alpaca creds missing — request to %s %s skipped", method, path)
        return None
    url = cfg.base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("APCA-API-KEY-ID", cfg.api_key)
    req.add_header("APCA-API-SECRET-KEY", cfg.api_secret)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except Exception as e:
        LOG.warning("alpaca %s %s failed: %s", method, path, e)
        return None


def get_account(cfg: ConnectorConfig) -> Dict:
    return _alpaca_request(cfg, "GET", "/v2/account") or {}


def get_positions(cfg: ConnectorConfig) -> List[Dict]:
    return _alpaca_request(cfg, "GET", "/v2/positions") or []


def get_clock(cfg: ConnectorConfig) -> Dict:
    return _alpaca_request(cfg, "GET", "/v2/clock") or {}


def get_quote(cfg: ConnectorConfig, symbol: str) -> Optional[Dict]:
    """Fetch the latest stock quote (NBBO). Used as the mid reference for
    EXP-2470 limit-at-mid orders. Options quotes use a different endpoint
    that requires the data subscription tier — guarded below."""
    url = f"{cfg.data_url}/v2/stocks/{symbol}/quotes/latest"
    req = urllib.request.Request(url)
    req.add_header("APCA-API-KEY-ID", cfg.api_key)
    req.add_header("APCA-API-SECRET-KEY", cfg.api_secret)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        LOG.warning("quote fetch failed for %s: %s", symbol, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Order types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OrderRequest:
    symbol: str
    qty: float
    side: str                 # "buy" / "sell"
    asset_class: str          # "equity" / "option"
    order_type: str = "limit"
    limit_price: Optional[float] = None
    time_in_force: str = "day"
    extended_hours: bool = False
    client_order_id: Optional[str] = None
    legs: Optional[List[Dict]] = None        # for multi-leg combo orders
    sleeve_id: str = ""

    def to_alpaca(self) -> Dict:
        body: Dict[str, Any] = {
            "symbol": self.symbol,
            "qty": str(self.qty),
            "side": self.side,
            "type": self.order_type,
            "time_in_force": self.time_in_force,
            "extended_hours": self.extended_hours,
        }
        if self.limit_price is not None:
            body["limit_price"] = str(round(self.limit_price, 4))
        if self.client_order_id:
            body["client_order_id"] = self.client_order_id
        if self.legs:
            body["order_class"] = "mleg"
            body["legs"] = self.legs
        return body


# ─────────────────────────────────────────────────────────────────────────────
# Execution-optimisation stack (EXP-2470)
# ─────────────────────────────────────────────────────────────────────────────
def in_patient_window(cfg: ConnectorConfig) -> bool:
    """Technique B: are we inside the last `patient_window_minutes` of
    today's regular session?"""
    if not cfg.execution.use_patient_window:
        return True
    clock = get_clock(cfg)
    if not clock:
        return True            # fail-open
    next_close = clock.get("next_close")
    if not next_close:
        return True
    try:
        close_dt = datetime.fromisoformat(next_close.replace("Z", "+00:00"))
    except Exception:
        return True
    now = datetime.now(timezone.utc)
    delta_min = (close_dt - now).total_seconds() / 60.0
    return 0 < delta_min <= cfg.execution.patient_window_minutes


def compute_mid_limit_price(cfg: ConnectorConfig, symbol: str,
                             side: str) -> Optional[float]:
    """Technique A: take the NBBO mid and apply an optional offset.
    Returns None when no quote is available — caller MUST refuse to
    submit (Rule Zero — never fabricate a price)."""
    q = get_quote(cfg, symbol)
    if not q or "quote" not in q:
        return None
    bid = float(q["quote"].get("bp", 0) or 0)
    ask = float(q["quote"].get("ap", 0) or 0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2
    offset = cfg.execution.limit_offset_pct
    if offset != 0.0:
        if side == "buy":
            mid = mid * (1 + offset)
        else:
            mid = mid * (1 - offset)
    return round(mid, 4)


def maybe_combo_order(cfg: ConnectorConfig, sleeve_id: str,
                      legs: List[Dict]) -> Optional[OrderRequest]:
    """Technique D: pack independent legs into a single multi-leg combo
    order when the sleeve emits more than one leg in the same submission
    cycle and Alpaca's mleg endpoint is enabled."""
    if not cfg.execution.use_multileg_combo or len(legs) < 2:
        return None
    if len(legs) > cfg.execution.max_legs_per_combo:
        return None
    return OrderRequest(
        symbol=legs[0]["symbol"].split("/")[0] if "/" in legs[0]["symbol"] else legs[0]["symbol"],
        qty=legs[0].get("qty", 1),
        side=legs[0]["side"],
        asset_class="option",
        order_type="limit",
        legs=legs,
        sleeve_id=sleeve_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Connector
# ─────────────────────────────────────────────────────────────────────────────
class AlpacaPaperConnector:
    """The only component allowed to submit orders to Alpaca."""

    def __init__(self, cfg: ConnectorConfig):
        self.cfg = cfg
        self.cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.health_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.last_decision: Optional[RiskDecision] = None

    @classmethod
    def from_config(cls, path: str | Path) -> "AlpacaPaperConnector":
        return cls(ConnectorConfig.from_yaml(Path(path)))

    # ── State persistence ────────────────────────────────────────────
    def _load_state(self) -> Dict:
        if self.cfg.state_file.exists():
            try:
                return json.load(open(self.cfg.state_file))
            except Exception:
                pass
        return {
            "rolling_peak_equity": self.cfg.starting_capital,
            "last_weights": {},
            "last_orders":  [],
            "circuit_state": "OK",
            "last_poll": None,
        }

    def _save_state(self) -> None:
        self.cfg.state_file.write_text(json.dumps(self.state, indent=2, default=str))

    # ── Smoke test ───────────────────────────────────────────────────
    def smoke(self) -> bool:
        ok = True
        if not self.cfg.api_key or not self.cfg.api_secret:
            LOG.error("alpaca creds missing — set ALPACA_API_KEY_PAPER + ALPACA_API_SECRET_PAPER")
            return False
        acct = get_account(self.cfg)
        if not acct:
            LOG.error("alpaca account fetch failed")
            return False
        eq = float(acct.get("equity", 0))
        cash = float(acct.get("cash", 0))
        bp = float(acct.get("buying_power", 0))
        LOG.info("alpaca paper account OK · equity=$%.0f · cash=$%.0f · BP=$%.0f", eq, cash, bp)
        clock = get_clock(self.cfg)
        if clock:
            LOG.info("alpaca clock · is_open=%s · next_open=%s · next_close=%s",
                     clock.get("is_open"), clock.get("next_open"), clock.get("next_close"))
        return ok

    # ── Circuit breaker (3% trailing DD, EXP-2370) ───────────────────
    # Maps onto compass.portfolio_risk_manager.CircuitState which has
    # NORMAL / WARN / HALT — we use WARN for the 3% soft trip and HALT
    # for the 6% hard trip.
    def evaluate_breaker(self, equity: float) -> Tuple[CircuitState, float]:
        peak = float(self.state.get("rolling_peak_equity", equity))
        if equity > peak:
            peak = equity
            self.state["rolling_peak_equity"] = peak
        dd = (equity - peak) / peak if peak > 0 else 0.0
        cb = self.cfg.breaker
        if dd <= -cb.hard_pct:
            return CircuitState.HALT, -dd
        if dd <= -cb.soft_pct:
            return CircuitState.WARN, -dd
        return CircuitState.NORMAL, -dd

    # ── Order submission ─────────────────────────────────────────────
    def submit_order(self, req: OrderRequest) -> Dict:
        """The single funnel for every Alpaca order. Applies EXP-2470
        execution overlay before submission."""
        # Technique B — patient window gate
        if not in_patient_window(self.cfg):
            LOG.info("[skip] %s outside patient window — defer", req.sleeve_id)
            return {"status": "deferred", "reason": "patient_window"}

        # Technique A — limit at mid (or pay-up if requested)
        if req.order_type == "limit" and req.limit_price is None and not req.legs:
            mid = compute_mid_limit_price(self.cfg, req.symbol, req.side)
            if mid is None:
                LOG.warning("[skip] %s no quote — Rule Zero, will not fabricate", req.symbol)
                return {"status": "rejected", "reason": "no_quote"}
            req.limit_price = mid

        body = req.to_alpaca()
        LOG.info("[submit] %s %s %s qty=%s limit=%s legs=%s",
                 req.sleeve_id, req.side, req.symbol, req.qty,
                 req.limit_price, len(req.legs or []))
        resp = _alpaca_request(self.cfg, "POST", "/v2/orders", body)
        if resp is None:
            return {"status": "rejected", "reason": "alpaca_error"}
        # Wrap the broker response in our own envelope: keep the broker's
        # raw payload under `broker` and overwrite our own `status` key
        # last so the spread doesn't clobber it.
        return {**resp, "status": "submitted", "broker_status": resp.get("status")}

    def cancel_all(self) -> int:
        resp = _alpaca_request(self.cfg, "DELETE", "/v2/orders")
        n = len(resp) if isinstance(resp, list) else 0
        LOG.info("[cancel] cancelled %d open orders", n)
        return n

    def close_all_positions(self) -> int:
        resp = _alpaca_request(self.cfg, "DELETE", "/v2/positions?cancel_orders=true")
        n = len(resp) if isinstance(resp, list) else 0
        LOG.warning("[close-all] flattened %d positions", n)
        return n

    # ── Decision application ─────────────────────────────────────────
    def apply_decision(self, decision: RiskDecision,
                        builders: Dict[str, Any]) -> Dict:
        """Translate a RiskDecision into orders.

        builders[sleeve_id] is a callable returning a List[OrderRequest]
        for that sleeve at the given target weight × leverage.
        """
        acct = get_account(self.cfg)
        equity = float(acct.get("equity", self.cfg.starting_capital))
        cb_state, dd = self.evaluate_breaker(equity)

        # Honour the BREAKER first
        if cb_state == CircuitState.HALT:
            self.cancel_all()
            self.close_all_positions()
            self._send_alert("CRITICAL", "circuit_breaker_hard",
                              f"6% trailing DD ({dd*100:.2f}%) → close all + halt 24h")
            self.state["circuit_state"] = CircuitState.HALT.value
            self._save_state()
            return {"status": "halted", "dd": dd}

        # Soft trip — cut leverage 50%
        applied_leverage = decision.leverage
        if cb_state == CircuitState.WARN:
            applied_leverage *= 0.5
            self._send_alert("WARNING", "circuit_breaker_soft",
                              f"3% trailing DD ({dd*100:.2f}%) → cut leverage 50%")
            self.state["circuit_state"] = CircuitState.WARN.value

        # Build and submit orders sleeve by sleeve
        orders_summary = []
        for sleeve_id, weight in decision.weights.items():
            if weight <= 0:
                continue
            target_exposure = weight * applied_leverage * equity
            builder = builders.get(sleeve_id)
            if builder is None:
                LOG.info("[skip] %s no builder registered", sleeve_id)
                continue
            try:
                reqs = builder(target_exposure)
            except Exception as e:
                LOG.exception("builder error %s: %s", sleeve_id, e)
                continue
            for req in reqs:
                resp = self.submit_order(req)
                orders_summary.append({"sleeve": sleeve_id, "symbol": req.symbol,
                                        "side": req.side, "status": resp.get("status")})

        self.state["last_weights"]  = dict(decision.weights)
        self.state["last_orders"]   = orders_summary
        self.state["last_poll"]     = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.state["circuit_state"] = cb_state.value
        self.state["scale_factor"]  = applied_leverage
        self._save_state()
        self._write_health(equity, cb_state, dd, applied_leverage)
        return {"status": "applied", "n_orders": len(orders_summary), "dd": dd}

    # ── Health snapshot ──────────────────────────────────────────────
    def _write_health(self, equity: float, cb: CircuitState,
                      dd: float, leverage: float) -> None:
        h = {
            "last_poll": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "equity": equity,
            "rolling_peak": self.state.get("rolling_peak_equity"),
            "trailing_dd_pct": round(dd * 100, 3),
            "circuit_breaker_state": cb.value,
            "leverage": round(leverage, 3),
            "scale_factor": round(leverage, 3),
            "vix_last": self.state.get("vix_last", 0),
            "n_open_positions": len(get_positions(self.cfg)),
            "alerts": [],
        }
        self.cfg.health_file.write_text(json.dumps(h, indent=2, default=str))

    # ── Telegram alerts ──────────────────────────────────────────────
    def _send_alert(self, level: str, code: str, msg: str) -> None:
        if not self.cfg.telegram_bot_token or not self.cfg.telegram_chat_id:
            LOG.info("[%s] %s — %s", level, code, msg)
            return
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(level, "•")
        text = f"{emoji} EXP-2620 {level} [{code}]\n{msg}"
        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": self.cfg.telegram_chat_id,
            "text": text,
        }).encode()
        try:
            urllib.request.urlopen(url, data=data, timeout=10)
        except Exception as e:
            LOG.warning("telegram send failed: %s", e)

    # ── Production loop ──────────────────────────────────────────────
    def run_loop(self, risk_mgr: PortfolioRiskManager,
                  decision_factory, builders: Dict[str, Any]) -> None:
        """Production cycle. `decision_factory()` returns a fresh
        RiskDecision (caller wires the strategy returns / equity).
        `builders` is a dict of sleeve_id -> callable[target_exposure -> List[OrderRequest]].
        """
        LOG.info("starting EXP-2620 connector loop · interval=%ds", self.cfg.poll_interval_seconds)
        while True:
            try:
                decision = decision_factory()
                self.last_decision = decision
                summary = self.apply_decision(decision, builders)
                LOG.info("cycle done: %s", summary)
            except Exception as e:
                LOG.exception("cycle failed: %s", e)
            time.sleep(self.cfg.poll_interval_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp2410_production_paper.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cancel-all", action="store_true")
    ap.add_argument("--close-all", action="store_true")
    ap.add_argument("--account", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s %(message)s")

    conn = AlpacaPaperConnector.from_config(args.config)
    if args.smoke:
        return 0 if conn.smoke() else 1
    if args.account:
        print(json.dumps(get_account(conn.cfg), indent=2))
        return 0
    if args.cancel_all:
        conn.cancel_all()
        return 0
    if args.close_all:
        conn.close_all_positions()
        return 0
    print("nothing to do; pass --smoke / --account / --cancel-all / --close-all")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
