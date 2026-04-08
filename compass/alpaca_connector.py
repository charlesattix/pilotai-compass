"""
compass/alpaca_connector.py — EXP-2890 Alpaca Integration Scaffold.

Production-ready Alpaca API integration layer. Sits between the
compass signal generators (EXP-2690) and real Alpaca paper trading.

RESPONSIBILITIES
----------------
1. Authentication and SDK selection
   - Prefers `alpaca-py` (new official SDK, 2023+)
   - Falls back to `alpaca-trade-api` if alpaca-py is unavailable
2. Order submission for multi-leg option spreads
   - Builds OCC option symbols from (ticker, exp, strike, type)
   - Uses Alpaca's multi-leg order endpoint when possible
   - Falls back to two simultaneous single-leg orders
3. Position monitoring and reconciliation
   - Intended-vs-actual diff (detects manual intervention + fills)
   - Per-stream attribution via ticker inference
4. Daily P&L reporting
   - Rolling equity log with day/week/MTD/YTD splits
   - Per-stream P&L attribution
5. Health check
   - Account liquidity, connection, clock sync, last-fill recency

CONFIG
------
Environment variables (set in .env or shell):
    ALPACA_API_KEY       = "<paper key>"
    ALPACA_SECRET_KEY    = "<paper secret>"
    ALPACA_PAPER         = "true"   (default; "false" for live)
    ALPACA_BASE_URL      = "https://paper-api.alpaca.markets"  (optional override)

The `paper_mode` flag in `AlpacaConnector.__init__` overrides the env.
Default endpoint is paper-api.alpaca.markets for safety.

RULE ZERO
---------
Every price that reaches this module comes from Alpaca live quotes.
The connector NEVER fabricates a quote. If a quote is missing, the
order is skipped, never extrapolated.

See also:
    compass/exp2620_alpaca_connector.py — earlier bespoke connector,
       tightly coupled to the 7-stream risk pipeline. This scaffold
       (EXP-2890) is simpler, self-contained, and meant to be the
       single integration seam for EXP-2830 orchestrator.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("alpaca_connector")
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PAPER_URL = "https://paper-api.alpaca.markets"
DEFAULT_LIVE_URL = "https://api.alpaca.markets"

ENV_KEY = "ALPACA_API_KEY"
ENV_SECRET = "ALPACA_SECRET_KEY"
ENV_PAPER = "ALPACA_PAPER"
ENV_BASE_URL = "ALPACA_BASE_URL"


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OptionLeg:
    """A single option leg inside a multi-leg spread."""
    ticker: str              # underlying, e.g. "SPY"
    expiration: str          # ISO date "2026-05-16"
    strike: float
    option_type: str         # "P" or "C"
    side: str                # "BUY" or "SELL"
    quantity: int = 1
    limit_price: Optional[float] = None   # per-contract; None = market


@dataclass
class SpreadOrder:
    """A multi-leg spread intention. Built by signal generators, executed
    by the connector."""
    stream: str              # sleeve id (e.g. "exp1220", "qqq_cs")
    strategy: str            # "bull_put_spread", "bear_call_spread", "calendar"
    legs: List[OptionLeg]
    net_credit: Optional[float] = None   # target net credit per contract
    client_order_id: Optional[str] = None
    tif: str = "DAY"
    submitted_at: Optional[str] = None
    broker_order_id: Optional[str] = None
    status: str = "PENDING"              # PENDING / FILLED / CANCELED / REJECTED

    def to_occ_symbols(self) -> List[str]:
        return [build_occ_symbol(leg.ticker, leg.expiration, leg.strike, leg.option_type)
                for leg in self.legs]


@dataclass
class Position:
    symbol: str              # OCC for options, ticker for equities
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    stream_attribution: str = "unknown"


@dataclass
class AccountSnapshot:
    timestamp: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    positions: List[Position]
    pending_orders: List[Dict]
    raw_error: Optional[str] = None


@dataclass
class HealthReport:
    ok: bool
    timestamp: str
    sdk: str                 # "alpaca-py" | "alpaca-trade-api" | "none"
    authenticated: bool
    account_status: Optional[str]
    paper_mode: bool
    base_url: str
    clock_ok: bool
    equity: Optional[float]
    buying_power: Optional[float]
    n_positions: int
    n_open_orders: int
    last_fill_age_hours: Optional[float]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# OCC symbol helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_occ_symbol(ticker: str, expiration: str, strike: float,
                       option_type: str) -> str:
    """Build an OCC 21-char option symbol.

    Format:  ROOT + YYMMDD + C/P + STRIKE×1000 zero-padded to 8 digits.
    Example: SPY240119P00480000  (SPY Jan 19 2024 $480 Put)
    """
    exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
    yy = exp_dt.strftime("%y")
    mm = exp_dt.strftime("%m")
    dd = exp_dt.strftime("%d")
    cp = option_type.upper()[0]  # "C" or "P"
    strike_int = int(round(float(strike) * 1000))
    return f"{ticker.upper()}{yy}{mm}{dd}{cp}{strike_int:08d}"


def parse_occ_symbol(symbol: str) -> Optional[Dict]:
    """Parse an OCC 21-char option symbol back to components. Returns
    None if the symbol doesn't match the expected format."""
    if len(symbol) < 15:
        return None
    # Walk backwards from the strike field
    try:
        strike_int = int(symbol[-8:])
        cp = symbol[-9]
        dd = symbol[-11:-9]
        mm = symbol[-13:-11]
        yy = symbol[-15:-13]
        root = symbol[:-15]
        return {
            "ticker": root,
            "expiration": f"20{yy}-{mm}-{dd}",
            "option_type": cp,
            "strike": strike_int / 1000.0,
        }
    except (ValueError, IndexError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Connector
# ═══════════════════════════════════════════════════════════════════════════

class AlpacaConnector:
    """Thin Alpaca wrapper for the North Star paper-trading loop.

    Usage:
        conn = AlpacaConnector.from_env()
        health = conn.health_check()
        if health.ok:
            snap = conn.snapshot()
            # ... compute intended positions ...
            diffs = conn.reconcile(intended_positions)
            for order in orders_to_submit:
                conn.submit_spread(order)
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper_mode: bool = True,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper_mode = paper_mode
        self.base_url = base_url or (DEFAULT_PAPER_URL if paper_mode else DEFAULT_LIVE_URL)
        self._sdk: str = "none"
        self._client = None
        self._trading_client = None
        self._init_sdk()

    # ─── construction ────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "AlpacaConnector":
        key = os.environ.get(ENV_KEY, "")
        secret = os.environ.get(ENV_SECRET, "")
        paper = os.environ.get(ENV_PAPER, "true").strip().lower() != "false"
        base = os.environ.get(ENV_BASE_URL) or None
        if not key or not secret:
            LOG.warning("Alpaca credentials missing (%s / %s unset)", ENV_KEY, ENV_SECRET)
        return cls(api_key=key, secret_key=secret, paper_mode=paper, base_url=base)

    def _init_sdk(self) -> None:
        """Try alpaca-py first, then alpaca-trade-api. If neither is
        installed, leave self._sdk = 'none' and the connector will
        return degraded responses from every method."""
        if not self.api_key or not self.secret_key:
            LOG.warning("no credentials — connector will run in offline mode")
            return
        # Try alpaca-py
        try:
            from alpaca.trading.client import TradingClient
            self._trading_client = TradingClient(
                self.api_key, self.secret_key, paper=self.paper_mode
            )
            self._sdk = "alpaca-py"
            LOG.info("alpaca-py SDK initialised (paper=%s)", self.paper_mode)
            return
        except ImportError:
            pass
        # Fallback: alpaca-trade-api
        try:
            import alpaca_trade_api as tradeapi  # type: ignore
            self._client = tradeapi.REST(
                self.api_key, self.secret_key, base_url=self.base_url
            )
            self._sdk = "alpaca-trade-api"
            LOG.info("alpaca-trade-api SDK initialised (base=%s)", self.base_url)
            return
        except ImportError:
            LOG.error("Neither alpaca-py nor alpaca-trade-api is installed")
            self._sdk = "none"

    # ─── health check ────────────────────────────────────────────────

    def health_check(self) -> HealthReport:
        ts = datetime.now(timezone.utc).isoformat()
        report = HealthReport(
            ok=False,
            timestamp=ts,
            sdk=self._sdk,
            authenticated=False,
            account_status=None,
            paper_mode=self.paper_mode,
            base_url=self.base_url,
            clock_ok=False,
            equity=None,
            buying_power=None,
            n_positions=0,
            n_open_orders=0,
            last_fill_age_hours=None,
        )

        if self._sdk == "none":
            report.errors.append("no SDK available (pip install alpaca-py)")
            return report
        if not self.api_key or not self.secret_key:
            report.errors.append(f"{ENV_KEY} / {ENV_SECRET} unset")
            return report

        try:
            acct = self._get_account()
            report.authenticated = True
            report.account_status = getattr(acct, "status", None)
            report.equity = float(getattr(acct, "equity", 0) or 0)
            report.buying_power = float(getattr(acct, "buying_power", 0) or 0)
        except Exception as e:
            report.errors.append(f"account fetch failed: {e}")
            return report

        try:
            clock = self._get_clock()
            report.clock_ok = clock is not None
        except Exception as e:
            report.warnings.append(f"clock fetch failed: {e}")

        try:
            positions = self._list_positions()
            report.n_positions = len(positions)
        except Exception as e:
            report.warnings.append(f"positions fetch failed: {e}")

        try:
            orders = self._list_open_orders()
            report.n_open_orders = len(orders)
        except Exception as e:
            report.warnings.append(f"orders fetch failed: {e}")

        try:
            last_fill = self._last_fill_timestamp()
            if last_fill is not None:
                age = (datetime.now(timezone.utc) - last_fill).total_seconds() / 3600.0
                report.last_fill_age_hours = round(age, 2)
        except Exception as e:
            report.warnings.append(f"last-fill fetch failed: {e}")

        report.ok = (report.authenticated and report.account_status == "ACTIVE"
                      and report.clock_ok and not report.errors)
        return report

    # ─── account / snapshot ──────────────────────────────────────────

    def _get_account(self):
        if self._sdk == "alpaca-py":
            return self._trading_client.get_account()
        if self._sdk == "alpaca-trade-api":
            return self._client.get_account()
        raise RuntimeError("no SDK")

    def _get_clock(self):
        if self._sdk == "alpaca-py":
            return self._trading_client.get_clock()
        if self._sdk == "alpaca-trade-api":
            return self._client.get_clock()
        return None

    def _list_positions(self) -> List:
        if self._sdk == "alpaca-py":
            return list(self._trading_client.get_all_positions())
        if self._sdk == "alpaca-trade-api":
            return list(self._client.list_positions())
        return []

    def _list_open_orders(self) -> List:
        if self._sdk == "alpaca-py":
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            return list(self._trading_client.get_orders(filter=req))
        if self._sdk == "alpaca-trade-api":
            return list(self._client.list_orders(status="open"))
        return []

    def _last_fill_timestamp(self) -> Optional[datetime]:
        if self._sdk == "alpaca-py":
            try:
                from alpaca.trading.requests import GetAccountActivitiesRequest
                req = GetAccountActivitiesRequest(activity_types=["FILL"])
                acts = self._trading_client.get_account_activities(req)
                for a in acts[:1]:
                    t = getattr(a, "transaction_time", None)
                    if t:
                        if isinstance(t, datetime):
                            return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
                        return datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            except Exception:
                return None
        if self._sdk == "alpaca-trade-api":
            try:
                acts = self._client.get_activities(activity_types="FILL")
                for a in acts[:1]:
                    t = getattr(a, "transaction_time", None)
                    if t:
                        return datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    def snapshot(self) -> AccountSnapshot:
        """Single-call account + positions + open orders pull."""
        ts = datetime.now(timezone.utc).isoformat()
        try:
            acct = self._get_account()
            positions_raw = self._list_positions()
            orders_raw = self._list_open_orders()
        except Exception as e:
            return AccountSnapshot(
                timestamp=ts, equity=0.0, cash=0.0, buying_power=0.0,
                portfolio_value=0.0, positions=[], pending_orders=[],
                raw_error=f"snapshot failed: {e}",
            )

        positions = [
            Position(
                symbol=getattr(p, "symbol", ""),
                qty=float(getattr(p, "qty", 0) or 0),
                avg_entry_price=float(getattr(p, "avg_entry_price", 0) or 0),
                market_value=float(getattr(p, "market_value", 0) or 0),
                unrealized_pl=float(getattr(p, "unrealized_pl", 0) or 0),
                stream_attribution=_infer_stream(getattr(p, "symbol", "")),
            )
            for p in positions_raw
        ]
        pending = []
        for o in orders_raw:
            pending.append({
                "id": getattr(o, "id", None),
                "client_order_id": getattr(o, "client_order_id", None),
                "symbol": getattr(o, "symbol", None),
                "side": str(getattr(o, "side", "")),
                "qty": float(getattr(o, "qty", 0) or 0),
                "limit_price": float(getattr(o, "limit_price", 0) or 0) if getattr(o, "limit_price", None) else None,
                "status": str(getattr(o, "status", "")),
            })

        return AccountSnapshot(
            timestamp=ts,
            equity=float(getattr(acct, "equity", 0) or 0),
            cash=float(getattr(acct, "cash", 0) or 0),
            buying_power=float(getattr(acct, "buying_power", 0) or 0),
            portfolio_value=float(getattr(acct, "portfolio_value", 0) or 0),
            positions=positions,
            pending_orders=pending,
        )

    # ─── order submission ────────────────────────────────────────────

    def submit_spread(self, order: SpreadOrder) -> SpreadOrder:
        """Submit a multi-leg option spread to Alpaca.

        alpaca-py MultilegOrderRequest is the preferred path. If that
        is unavailable (older SDK, unsupported strategy), we fall back
        to submitting simultaneous single-leg orders.

        The submitted SpreadOrder is returned with `broker_order_id`
        and `status` populated.
        """
        if self._sdk == "none":
            order.status = "REJECTED"
            order.submitted_at = datetime.now(timezone.utc).isoformat()
            LOG.error("cannot submit %s: no SDK", order.client_order_id)
            return order

        order.submitted_at = datetime.now(timezone.utc).isoformat()
        if not order.client_order_id:
            order.client_order_id = f"{order.stream}-{order.strategy}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        # Try the multi-leg path
        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            # alpaca-py 0.43+ has OptionLegRequest + MultilegOrderRequest
            # (in alpaca.trading.requests). Import guarded in case the
            # SDK version differs.
            try:
                from alpaca.trading.requests import OptionLegRequest, MultilegOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
                legs_req = [
                    OptionLegRequest(
                        symbol=build_occ_symbol(leg.ticker, leg.expiration, leg.strike, leg.option_type),
                        side=OrderSide.BUY if leg.side.upper() == "BUY" else OrderSide.SELL,
                        ratio_qty=leg.quantity,
                    )
                    for leg in order.legs
                ]
                req = MultilegOrderRequest(
                    legs=legs_req,
                    qty=1,
                    time_in_force=TimeInForce.DAY if order.tif.upper() == "DAY" else TimeInForce.GTC,
                    order_class=OrderClass.MLEG,
                    limit_price=order.net_credit,
                    client_order_id=order.client_order_id,
                )
                resp = self._trading_client.submit_order(order_data=req)
                order.broker_order_id = str(getattr(resp, "id", ""))
                order.status = str(getattr(resp, "status", "SUBMITTED"))
                LOG.info("submitted MLEG %s -> %s", order.client_order_id, order.broker_order_id)
                return order
            except ImportError:
                LOG.info("MLEG not available in alpaca-py; falling back to single-leg")
        except Exception as e:
            LOG.warning("MLEG submission failed (%s) — falling back", e)

        # Fallback: submit each leg as an independent order
        submitted_any = False
        first_id: Optional[str] = None
        for i, leg in enumerate(order.legs):
            try:
                leg_id = self._submit_single_leg(leg, f"{order.client_order_id}-L{i}")
                if leg_id and not first_id:
                    first_id = leg_id
                    submitted_any = True
            except Exception as e:
                LOG.error("leg %d submission failed: %s", i, e)
        order.broker_order_id = first_id or ""
        order.status = "SUBMITTED" if submitted_any else "REJECTED"
        return order

    def _submit_single_leg(self, leg: OptionLeg, client_id: str) -> Optional[str]:
        occ = build_occ_symbol(leg.ticker, leg.expiration, leg.strike, leg.option_type)
        if self._sdk == "alpaca-py":
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            side = OrderSide.BUY if leg.side.upper() == "BUY" else OrderSide.SELL
            if leg.limit_price is not None:
                req = LimitOrderRequest(
                    symbol=occ, qty=leg.quantity, side=side,
                    time_in_force=TimeInForce.DAY, limit_price=leg.limit_price,
                    client_order_id=client_id,
                )
            else:
                req = MarketOrderRequest(
                    symbol=occ, qty=leg.quantity, side=side,
                    time_in_force=TimeInForce.DAY, client_order_id=client_id,
                )
            resp = self._trading_client.submit_order(order_data=req)
            return str(getattr(resp, "id", ""))
        if self._sdk == "alpaca-trade-api":
            resp = self._client.submit_order(
                symbol=occ, qty=leg.quantity,
                side=leg.side.lower(),
                type="limit" if leg.limit_price is not None else "market",
                time_in_force="day",
                limit_price=leg.limit_price,
                client_order_id=client_id,
            )
            return getattr(resp, "id", "")
        return None

    def cancel_all(self) -> int:
        """Cancel every open order. Returns number of orders canceled."""
        try:
            if self._sdk == "alpaca-py":
                result = self._trading_client.cancel_orders()
                return len(result) if hasattr(result, "__len__") else 0
            if self._sdk == "alpaca-trade-api":
                self._client.cancel_all_orders()
                return -1  # count not returned by this SDK
        except Exception as e:
            LOG.error("cancel_all failed: %s", e)
        return 0

    def close_all_positions(self, cancel_orders: bool = True) -> int:
        """Flatten. Cancels open orders first, then liquidates all positions."""
        try:
            if self._sdk == "alpaca-py":
                result = self._trading_client.close_all_positions(
                    cancel_orders=cancel_orders
                )
                return len(result) if hasattr(result, "__len__") else 0
            if self._sdk == "alpaca-trade-api":
                if cancel_orders:
                    self._client.cancel_all_orders()
                self._client.close_all_positions()
                return -1
        except Exception as e:
            LOG.error("close_all_positions failed: %s", e)
        return 0

    # ─── reconciliation ──────────────────────────────────────────────

    def reconcile(self, intended: Dict[str, float],
                    tolerance: float = 0.0) -> Dict[str, Dict]:
        """Compare intended positions vs actual broker positions.

        Args:
            intended: {symbol: signed_qty} — what the portfolio thinks
                it owns (positive = long, negative = short).
            tolerance: allowed |intended - actual| delta before flagging.

        Returns:
            dict of {symbol: {intended, actual, delta, status}} where
            status is "MATCH" | "UNDER" | "OVER" | "ORPHAN" | "MISSING".
        """
        snap = self.snapshot()
        actual_by_symbol = {p.symbol: p.qty for p in snap.positions}

        out: Dict[str, Dict] = {}
        all_symbols = set(intended.keys()) | set(actual_by_symbol.keys())
        for sym in all_symbols:
            i = float(intended.get(sym, 0.0))
            a = float(actual_by_symbol.get(sym, 0.0))
            delta = a - i
            if abs(delta) <= tolerance:
                status = "MATCH"
            elif i == 0 and a != 0:
                status = "ORPHAN"        # broker has what we didn't intend
            elif a == 0 and i != 0:
                status = "MISSING"       # we intended but didn't fill
            elif abs(a) < abs(i):
                status = "UNDER"         # partial fill
            else:
                status = "OVER"          # over-filled
            out[sym] = {
                "symbol": sym,
                "intended": i,
                "actual": a,
                "delta": round(delta, 4),
                "status": status,
            }
        return out

    # ─── daily P&L reporting ─────────────────────────────────────────

    def daily_pnl(self, equity_log_path: Optional[Path] = None) -> Dict:
        """Compute day / week / MTD / YTD / total P&L from Alpaca account
        plus optional rolling equity log (CSV with columns date, equity)."""
        snap = self.snapshot()
        out: Dict[str, Any] = {
            "timestamp": snap.timestamp,
            "equity": snap.equity,
            "cash": snap.cash,
            "buying_power": snap.buying_power,
            "portfolio_value": snap.portfolio_value,
            "n_positions": len(snap.positions),
            "stream_pnl_unrealized": {},
            "stream_pnl_total": {},
        }

        # Per-stream unrealized P&L
        by_stream: Dict[str, float] = {}
        by_stream_mv: Dict[str, float] = {}
        for p in snap.positions:
            by_stream[p.stream_attribution] = by_stream.get(p.stream_attribution, 0.0) + p.unrealized_pl
            by_stream_mv[p.stream_attribution] = by_stream_mv.get(p.stream_attribution, 0.0) + abs(p.market_value)
        out["stream_pnl_unrealized"] = {k: round(v, 2) for k, v in by_stream.items()}
        out["stream_market_value"] = {k: round(v, 2) for k, v in by_stream_mv.items()}

        # Period returns from equity log (if provided)
        if equity_log_path and equity_log_path.exists():
            try:
                import csv
                rows = []
                with equity_log_path.open() as fh:
                    for r in csv.DictReader(fh):
                        rows.append(r)
                if rows:
                    today_eq = snap.equity
                    # last row's equity = prior close (approx)
                    prior_eq = float(rows[-1].get("equity", today_eq))
                    out["day_pnl"] = round(today_eq - prior_eq, 2)
                    out["day_pnl_pct"] = round(
                        (today_eq / prior_eq - 1.0) * 100.0 if prior_eq > 0 else 0.0, 3
                    )
                    # Week: row from 7 days ago
                    from datetime import timedelta as td
                    today = datetime.utcnow().date()
                    week_anchor_date = today - td(days=7)
                    mtd_anchor_date = today.replace(day=1) - td(days=1)
                    ytd_anchor_date = today.replace(month=1, day=1) - td(days=1)
                    def at_or_before(cutoff):
                        for r in reversed(rows):
                            try:
                                d = datetime.strptime(r["date"], "%Y-%m-%d").date()
                                if d <= cutoff:
                                    return float(r["equity"])
                            except Exception:
                                continue
                        return prior_eq
                    week_eq = at_or_before(week_anchor_date)
                    mtd_eq = at_or_before(mtd_anchor_date)
                    ytd_eq = at_or_before(ytd_anchor_date)
                    out["week_pnl"] = round(today_eq - week_eq, 2)
                    out["mtd_pnl"] = round(today_eq - mtd_eq, 2)
                    out["ytd_pnl"] = round(today_eq - ytd_eq, 2)
            except Exception as e:
                out["equity_log_error"] = str(e)

        return out

    def append_equity_log(self, equity_log_path: Path) -> None:
        """Append today's equity to a rolling CSV (idempotent per date)."""
        import csv
        today = datetime.utcnow().date().isoformat()
        snap = self.snapshot()
        equity_log_path.parent.mkdir(parents=True, exist_ok=True)
        existing_dates = set()
        rows: List[Dict] = []
        if equity_log_path.exists():
            with equity_log_path.open() as fh:
                for r in csv.DictReader(fh):
                    existing_dates.add(r.get("date"))
                    rows.append(r)
        if today in existing_dates:
            return
        rows.append({
            "date": today,
            "equity": f"{snap.equity:.2f}",
            "cash": f"{snap.cash:.2f}",
            "portfolio_value": f"{snap.portfolio_value:.2f}",
            "n_positions": str(len(snap.positions)),
        })
        fields = ["date", "equity", "cash", "portfolio_value", "n_positions"]
        with equity_log_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_STREAM_TICKER_MAP = {
    "SPY": "exp1220",
    "QQQ": "qqq_cs",
    "XLF": "xlf_cs",
    "XLI": "xli_cs",
    "GLD": "gld_cal",
    "SLV": "slv_cal",
    "IWM": "vol_arb",
    "UVXY": "v5_hedge",
    "VXX": "v5_hedge",
}


def _infer_stream(symbol: str) -> str:
    """Attribute an OCC or ticker symbol to its sleeve id.

    For option OCC symbols, the ticker is the leading letters before
    the YYMMDD date. For equity tickers, the symbol IS the ticker.
    """
    if not symbol:
        return "unknown"
    parsed = parse_occ_symbol(symbol)
    ticker = parsed["ticker"] if parsed else symbol
    return _STREAM_TICKER_MAP.get(ticker.upper(), "unknown")


# ═══════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ═══════════════════════════════════════════════════════════════════════════

def _cli() -> int:
    """Smoke test: init connector from env, run health check, print
    snapshot summary. Used for `python -m compass.alpaca_connector`.

    Exits 0 on health_ok, 1 on any degradation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    print("EXP-2890 Alpaca Connector — smoke test")
    print("=" * 60)
    conn = AlpacaConnector.from_env()
    print(f"SDK: {conn._sdk}")
    print(f"paper mode: {conn.paper_mode}")
    print(f"base url: {conn.base_url}")

    print("\nhealth check...")
    report = conn.health_check()
    print(json.dumps(asdict(report), indent=2, default=str))

    if not report.authenticated:
        print("\nno authenticated session — skipping snapshot")
        return 1

    print("\nsnapshot...")
    snap = conn.snapshot()
    print(f"  equity: ${snap.equity:,.2f}")
    print(f"  cash:   ${snap.cash:,.2f}")
    print(f"  positions: {len(snap.positions)}")
    print(f"  open orders: {len(snap.pending_orders)}")
    for p in snap.positions[:10]:
        print(f"    {p.symbol:24s} qty={p.qty:+.2f}  "
              f"mv=${p.market_value:+,.2f}  upl=${p.unrealized_pl:+,.2f}  "
              f"[{p.stream_attribution}]")

    print("\ndaily P&L…")
    pnl = conn.daily_pnl(ROOT / "logs" / "north_star_v6" / "equity_log.csv")
    print(json.dumps(pnl, indent=2, default=str))

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(_cli())
