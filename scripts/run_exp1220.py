#!/usr/bin/env python3
"""
run_exp1220.py — Standalone EXP-1220 paper trading scanner.

Reads configs/deploy_exp1220_1.5x.yaml, scans SPY options via Alpaca,
applies 1.5× position sizing, submits paper orders, tracks positions.

Usage:
    python3 scripts/run_exp1220.py              # live scan + trade
    python3 scripts/run_exp1220.py --dry-run    # show trades without submitting
    python3 scripts/run_exp1220.py --status     # show open positions
    python3 scripts/run_exp1220.py --close-all  # close all open positions
    python3 scripts/run_exp1220.py --smoke-test # validate config + API without trading
    python3 scripts/run_exp1220.py --health     # write health file and exit

Designed to run as a LaunchAgent (once per day at 9:35 AM ET).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import logging.handlers
import math
import os
import sys
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Default paths — overridable per instance via set_instance_id() so multiple
# leverage configs can coexist (e.g. 2x/3x/4x/5x running simultaneously).
_DEFAULT_CONFIG = ROOT / "configs" / "deploy_exp1220_1.5x.yaml"
_CONFIG_PATH = _DEFAULT_CONFIG
_INSTANCE_ID = "exp1220"
JOURNAL_PATH = LOG_DIR / "trade_journal.csv"
STATE_PATH = LOG_DIR / f"{_INSTANCE_ID}_state.json"
HEALTH_PATH = LOG_DIR / f"{_INSTANCE_ID}_health.json"


def set_instance_id(instance_id: str) -> None:
    """Re-route state/health/log/journal paths to an instance-specific namespace.

    Called when --config is provided, so multiple leverage configs can run
    on separate .env/accounts without cross-writing state files.
    """
    global _INSTANCE_ID, STATE_PATH, HEALTH_PATH, JOURNAL_PATH
    _INSTANCE_ID = instance_id
    STATE_PATH = LOG_DIR / f"{instance_id}_state.json"
    HEALTH_PATH = LOG_DIR / f"{instance_id}_health.json"
    JOURNAL_PATH = LOG_DIR / f"{instance_id}_trade_journal.csv"


def setup_logging(log_level: str = "INFO", max_bytes: int = 50 * 1024 * 1024,
                   backup_count: int = 5) -> logging.Logger:
    """Configure logging with rotation. Safe to call multiple times."""
    logger = logging.getLogger(f"exp1220.{_INSTANCE_ID}")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers so we don't duplicate on re-init
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    rotating = logging.handlers.RotatingFileHandler(
        LOG_DIR / f"{_INSTANCE_ID}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    rotating.setFormatter(fmt)
    logger.addHandler(rotating)

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    logger.propagate = False
    return logger


log = setup_logging()


def write_health(status: str, details: Optional[dict] = None,
                  error: Optional[str] = None) -> None:
    """Write a health status file Charles can monitor.

    status: 'ok', 'warning', 'error', 'halted'
    """
    health = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "last_run": datetime.now().isoformat(),
        "error": error,
        "details": details or {},
    }
    try:
        HEALTH_PATH.write_text(json.dumps(health, indent=2, default=str))
    except Exception as e:
        log.error(f"Failed to write health file: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    """Load the YAML config at _CONFIG_PATH (default 1.5x, overridable via --config)."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# State management (persist open positions across runs)
# ═══════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"positions": [], "last_entry_date": None, "total_pnl": 0.0}


def save_state(state: dict):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(entry: dict):
    exists = JOURNAL_PATH.exists()
    with open(JOURNAL_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "action", "ticker", "type", "short_strike", "long_strike",
            "expiration", "contracts", "credit", "order_id", "status",
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(entry)


# ═══════════════════════════════════════════════════════════════════════════
# Market data (yfinance — free, no API key needed) with retry
# ═══════════════════════════════════════════════════════════════════════════

def _yahoo_fetch(symbol: str, max_retries: int = 3,
                  backoff: float = 2.0, timeout: int = 10) -> Optional[float]:
    """Fetch latest price from Yahoo Finance with retry + exponential backoff."""
    import urllib.request
    import urllib.error

    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?range=1d&interval=1d")
    last_err = None

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                results = data.get("chart", {}).get("result")
                if not results:
                    raise ValueError("no results in response")
                meta = results[0].get("meta", {})
                price = meta.get("regularMarketPrice")
                if price is None:
                    raise ValueError("missing regularMarketPrice")
                return float(price)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            log.warning(f"[attempt {attempt+1}/{max_retries}] {symbol} fetch network error: {e}")
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            log.warning(f"[attempt {attempt+1}/{max_retries}] {symbol} fetch parse error: {e}")
        except Exception as e:
            last_err = e
            log.warning(f"[attempt {attempt+1}/{max_retries}] {symbol} fetch error: {e}")

        if attempt < max_retries - 1:
            time.sleep(backoff ** attempt)

    log.error(f"{symbol} fetch failed after {max_retries} attempts: {last_err}")
    return None


def get_vix(default: float = 20.0) -> float:
    """Fetch current VIX. Returns default on failure."""
    price = _yahoo_fetch("%5EVIX")
    if price is None:
        log.warning(f"VIX fetch failed. Using default {default}")
        return default
    return price


def get_spy_price() -> float:
    """Fetch current SPY price. Returns 0.0 on failure."""
    price = _yahoo_fetch("SPY")
    return price if price is not None else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Alpaca client
# ═══════════════════════════════════════════════════════════════════════════

class AlpacaClient:
    """Thin wrapper around alpaca-py for options trading."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.api_key = os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self._client = None
        self._trading_client = None

        if not self.api_key or not self.secret_key:
            if not dry_run:
                log.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
                sys.exit(1)
            else:
                log.info("Dry-run mode — Alpaca credentials not required")

    def _ensure_client(self):
        if self._trading_client is None and not self.dry_run:
            try:
                from alpaca.trading.client import TradingClient
                self._trading_client = TradingClient(
                    self.api_key, self.secret_key, paper=True
                )
            except ImportError:
                log.error("alpaca-py not installed. Run: pip3 install alpaca-py")
                sys.exit(1)

    def get_account(self) -> dict:
        if self.dry_run:
            return {"equity": "100000", "buying_power": "50000", "status": "ACTIVE"}
        self._ensure_client()
        acct = self._trading_client.get_account()
        return {"equity": acct.equity, "buying_power": acct.buying_power, "status": acct.status}

    def get_positions(self) -> list:
        if self.dry_run:
            return []
        self._ensure_client()
        return self._trading_client.get_all_positions()

    def get_option_chain(self, ticker: str, expiration: str) -> list:
        """Fetch option chain with bid/ask quotes for a specific expiration.

        FIX 3: Now fetches real bid/ask via Alpaca option snapshots.
        Returns list of dicts with symbol, strike, type, expiration, bid, ask.
        """
        if self.dry_run:
            spy_price = get_spy_price() or 540.0
            chain = []
            for strike in range(int(spy_price) - 50, int(spy_price) + 10, 1):
                chain.append({
                    "symbol": f"SPY{expiration.replace('-', '')}P{strike:08d}",
                    "strike": float(strike),
                    "type": "put",
                    "expiration": expiration,
                    "bid": round(max(0.10, 3.0 * math.exp(-0.08 * max(0, spy_price - strike))), 2),
                    "ask": round(max(0.15, 3.5 * math.exp(-0.08 * max(0, spy_price - strike))), 2),
                })
            return chain

        self._ensure_client()
        try:
            from alpaca.trading.requests import GetOptionContractsRequest
            req = GetOptionContractsRequest(
                underlying_symbols=["SPY"],
                expiration_date=expiration,
                type="put",
            )
            contracts = self._trading_client.get_option_contracts(req)
            symbols = [c.symbol for c in contracts.option_contracts]

            # FIX 3: Fetch real quotes via option snapshots
            quotes = self._get_option_quotes(symbols)

            chain = []
            for c in contracts.option_contracts:
                q = quotes.get(c.symbol, {})
                chain.append({
                    "symbol": c.symbol,
                    "strike": float(c.strike_price),
                    "type": c.type,
                    "expiration": str(c.expiration_date),
                    "bid": q.get("bid", 0.0),
                    "ask": q.get("ask", 0.0),
                })
            return chain
        except Exception as e:
            log.error(f"Option chain fetch failed: {e}")
            return []

    def _get_option_quotes(self, symbols: list) -> dict:
        """Fetch real bid/ask quotes for option symbols via Alpaca data API.

        FIX 3: Uses Alpaca's option snapshot endpoint for real-time quotes.
        Falls back to empty quotes if API unavailable.
        """
        if not symbols:
            return {}
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.requests import OptionSnapshotRequest
            data_client = OptionHistoricalDataClient(self.api_key, self.secret_key)
            snapshots = data_client.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=symbols[:50])  # batch limit
            )
            result = {}
            for sym, snap in snapshots.items():
                bid = snap.latest_quote.bid_price if snap.latest_quote else 0.0
                ask = snap.latest_quote.ask_price if snap.latest_quote else 0.0
                result[sym] = {"bid": float(bid), "ask": float(ask)}
            return result
        except ImportError:
            log.warning("alpaca.data not available — using REST fallback for quotes")
        except Exception as e:
            log.warning(f"Option snapshot failed: {e} — quotes unavailable")

        # REST API fallback: /v2/options/snapshots
        try:
            import urllib.request
            syms = ",".join(symbols[:20])
            url = f"{self.base_url}/v2/options/snapshots?symbols={syms}"
            req = urllib.request.Request(url, headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                result = {}
                for sym, snap in data.get("snapshots", {}).items():
                    quote = snap.get("latestQuote", {})
                    result[sym] = {
                        "bid": float(quote.get("bp", 0)),
                        "ask": float(quote.get("ap", 0)),
                    }
                return result
        except Exception as e:
            log.warning(f"REST quote fallback failed: {e}")
            return {}

    def submit_spread_order(self, short_symbol: str, long_symbol: str,
                            contracts: int, credit: float,
                            short_bid: float = 0, long_ask: float = 0,
                            ) -> Optional[str]:
        """Submit a credit spread as TWO SEPARATE leg orders.

        Uses ACTUAL bid/ask from option snapshots:
          - Short leg limit = short_bid (sell at bid for immediate fill)
          - Long leg limit = long_ask (buy at ask for immediate fill)
          - Net credit = short_bid - long_ask

        Args:
            short_bid: Real bid price of the short put (from snapshot).
            long_ask:  Real ask price of the long put (from snapshot).
        """
        if self.dry_run:
            ts = datetime.now().strftime('%H%M%S')
            order_id = f"DRY-S-{ts},DRY-L-{ts}"
            log.info(f"[DRY-RUN] Would submit:")
            log.info(f"  Leg 1 (STO): SELL {short_symbol} × {contracts} @ ${short_bid:.2f} (bid)")
            log.info(f"  Leg 2 (BTO): BUY  {long_symbol} × {contracts} @ ${long_ask:.2f} (ask)")
            log.info(f"  Net credit: ${short_bid - long_ask:.2f}")
            return order_id

        self._ensure_client()
        try:
            from alpaca.trading.requests import LimitOrderRequest, OrderSide, TimeInForce

            # Leg 1: STO short put at its bid price
            short_order = self._trading_client.submit_order(
                LimitOrderRequest(
                    symbol=short_symbol,
                    qty=contracts,
                    side=OrderSide.SELL,
                    type="limit",
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(short_bid, 2),
                )
            )
            log.info(f"Short leg submitted: {short_order.id} (STO {short_symbol} @ ${short_bid:.2f})")

            # Leg 2: BTO long put at its ask price
            long_order = self._trading_client.submit_order(
                LimitOrderRequest(
                    symbol=long_symbol,
                    qty=contracts,
                    side=OrderSide.BUY,
                    type="limit",
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(long_ask, 2),
                )
            )
            log.info(f"Long leg submitted: {long_order.id} (BTO {long_symbol} @ ${long_ask:.2f})")

            return f"{short_order.id},{long_order.id}"
        except Exception as e:
            log.error(f"Order submission failed: {e}")
            return None

    def close_spread(self, short_symbol: str, long_symbol: str,
                     contracts: int) -> bool:
        """Close a credit spread by closing BOTH legs.

        FIX 4: Submits BTC (buy-to-close) on short leg and
        STC (sell-to-close) on long leg.
        """
        if self.dry_run:
            log.info(f"[DRY-RUN] Would close spread:")
            log.info(f"  BTC: BUY  {short_symbol} × {contracts}")
            log.info(f"  STC: SELL {long_symbol} × {contracts}")
            return True

        self._ensure_client()
        success = True
        try:
            from alpaca.trading.requests import MarketOrderRequest, OrderSide, TimeInForce

            # BTC: Buy back the short put
            self._trading_client.submit_order(
                MarketOrderRequest(
                    symbol=short_symbol, qty=contracts,
                    side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                )
            )
            log.info(f"BTC submitted: {short_symbol} × {contracts}")
        except Exception as e:
            log.error(f"BTC failed for {short_symbol}: {e}")
            success = False

        try:
            from alpaca.trading.requests import MarketOrderRequest, OrderSide, TimeInForce

            # STC: Sell the long put
            self._trading_client.submit_order(
                MarketOrderRequest(
                    symbol=long_symbol, qty=contracts,
                    side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                )
            )
            log.info(f"STC submitted: {long_symbol} × {contracts}")
        except Exception as e:
            log.error(f"STC failed for {long_symbol}: {e}")
            success = False

        return success


# ═══════════════════════════════════════════════════════════════════════════
# Signal generation
# ═══════════════════════════════════════════════════════════════════════════

def find_target_expiration(target_dte: int, min_dte: int, max_dte: int) -> str:
    """Find the next monthly/weekly expiration within DTE range."""
    today = date.today()
    target_date = today + timedelta(days=target_dte)

    # Find the nearest Friday to target date
    days_to_friday = (4 - target_date.weekday()) % 7
    exp_date = target_date + timedelta(days=days_to_friday)

    # Clamp to min/max DTE
    min_date = today + timedelta(days=min_dte)
    max_date = today + timedelta(days=max_dte)

    if exp_date < min_date:
        exp_date = min_date + timedelta(days=(4 - min_date.weekday()) % 7)
    if exp_date > max_date:
        exp_date = max_date - timedelta(days=(max_date.weekday() - 4) % 7)

    return exp_date.strftime("%Y-%m-%d")


def select_strikes(spy_price: float, otm_pct: float, spread_width: float,
                   direction: str = "bull_put") -> Tuple[float, float]:
    """Select short and long strikes for the spread."""
    if direction == "bull_put":
        short_strike = round(spy_price * (1 - otm_pct))  # below current price
        long_strike = short_strike - spread_width
    else:  # bear_call
        short_strike = round(spy_price * (1 + otm_pct))
        long_strike = short_strike + spread_width
    return float(short_strike), float(long_strike)


def compute_contracts(account_equity: float, config: dict,
                      spread_width: float, estimated_credit: float) -> int:
    """Compute number of contracts based on 1.5× leveraged risk sizing."""
    leverage = config["leverage"]["multiplier"]
    risk_pct = config["sizing"]["base_risk_pct"] / 100 * leverage
    max_risk = account_equity * risk_pct

    max_loss_per_contract = (spread_width - estimated_credit) * 100
    if max_loss_per_contract <= 0:
        return 0

    contracts = int(max_risk / max_loss_per_contract)
    contracts = max(config["sizing"]["contracts_min"],
                    min(contracts, config["sizing"]["contracts_max"]))
    return contracts


# ═══════════════════════════════════════════════════════════════════════════
# Main scanner logic
# ═══════════════════════════════════════════════════════════════════════════

def should_scan(config: dict, state: dict) -> Tuple[bool, str]:
    """Check if we should scan for new entries today."""
    today = date.today()

    # Check day of week
    scan_day = config["cadence"]["scan_day"]
    weekday = today.weekday()
    if scan_day == "monday" and weekday != 0:
        return False, f"Not scan day (today={today.strftime('%A')}, scan={scan_day})"

    # Check spacing from last entry
    last_entry = state.get("last_entry_date")
    if last_entry:
        last_dt = datetime.strptime(last_entry, "%Y-%m-%d").date()
        days_since = (today - last_dt).days
        min_spacing = config["cadence"]["min_spacing_days"]
        if days_since < min_spacing:
            return False, f"Too soon since last entry ({days_since}d < {min_spacing}d min)"

    # Check max concurrent
    open_positions = [p for p in state.get("positions", []) if p.get("status") == "open"]
    max_conc = config["cadence"]["max_concurrent"]
    if len(open_positions) >= max_conc:
        return False, f"Max concurrent reached ({len(open_positions)}/{max_conc})"

    return True, "Ready to scan"


def check_exits(state: dict, config: dict, client: AlpacaClient, spy_price: float):
    """Check open positions for exit conditions.

    FIX 4: Actually submits closing orders to Alpaca (BTC short + STC long).
    """
    today = date.today()
    positions = state.get("positions", [])
    exits = config["exits"]

    for pos in positions:
        if pos.get("status") != "open":
            continue

        exit_reason = None
        exp_date = datetime.strptime(pos["expiration"], "%Y-%m-%d").date()
        dte = (exp_date - today).days
        entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
        hold_days = (today - entry_date).days

        if dte <= exits["dte_exit"]:
            exit_reason = f"dte_exit ({dte}d)"
        elif hold_days >= exits["max_hold_days"]:
            exit_reason = f"max_hold ({hold_days}d)"
        elif today >= exp_date:
            exit_reason = "expiration"

        if exit_reason:
            log.info(f"EXIT ({exit_reason}): {pos['short_strike']}/{pos['long_strike']} exp {pos['expiration']}")

            # FIX 4: Submit actual closing orders for BOTH legs
            short_sym = pos.get("short_symbol", "")
            long_sym = pos.get("long_symbol", "")
            n_contracts = pos.get("contracts", 1)

            if short_sym and long_sym and exit_reason != "expiration":
                # Don't submit close orders for expired contracts — they settle automatically
                closed = client.close_spread(short_sym, long_sym, n_contracts)
                if not closed:
                    log.error(f"Failed to close spread {short_sym}/{long_sym} — marking for retry")
                    pos["exit_retry"] = True
                    continue  # don't mark as closed if orders failed

            pos["status"] = "closed"
            pos["exit_reason"] = exit_reason
            pos["exit_date"] = str(today)
            log_trade({"timestamp": datetime.now().isoformat(), "action": "CLOSE",
                       "ticker": "SPY", "type": exit_reason,
                       "short_strike": pos.get("short_strike", ""),
                       "long_strike": pos.get("long_strike", ""),
                       "expiration": pos.get("expiration", ""),
                       "contracts": n_contracts,
                       "credit": pos.get("credit", ""),
                       "order_id": pos.get("order_id", ""),
                       "status": "closed"})


def run_scan(config: dict, state: dict, client: AlpacaClient, dry_run: bool):
    """Main scan: check VIX, find spread, size position, submit order."""
    # Get market data
    vix = get_vix()
    spy_price = get_spy_price()
    log.info(f"Market data: SPY=${spy_price:.2f}, VIX={vix:.1f}")

    # VIX filter
    vix_max = config["entry_signals"]["vix_max_entry"]
    vix_min = config["entry_signals"]["vix_min_entry"]
    if vix > vix_max:
        log.info(f"SKIP: VIX {vix:.1f} > {vix_max} (too high)")
        return
    if vix < vix_min:
        log.info(f"SKIP: VIX {vix:.1f} < {vix_min} (no premium)")
        return

    if spy_price <= 0:
        log.error("Could not get SPY price. Aborting scan.")
        return

    # Find expiration
    spread_cfg = config["spread"]
    expiration = find_target_expiration(
        spread_cfg["target_dte"], spread_cfg["min_dte"], spread_cfg["max_dte"])
    log.info(f"Target expiration: {expiration}")

    # Select strikes (bull put spread — primary)
    otm_pct = spread_cfg["otm_pct"]
    width = spread_cfg["width"]
    short_strike, long_strike = select_strikes(spy_price, otm_pct, width, "bull_put")
    log.info(f"Spread: SELL {short_strike}P / BUY {long_strike}P (${width} wide)")

    # Estimate credit (rough: in dry-run we use mock, live would use Alpaca quotes)
    chain = client.get_option_chain("SPY", expiration)
    estimated_credit = 0.0
    short_data = next((c for c in chain if abs(c["strike"] - short_strike) < 1.01), None)
    long_data = next((c for c in chain if abs(c["strike"] - long_strike) < 1.01), None)

    if short_data and long_data:
        short_bid = short_data.get("bid", 0.0)
        long_ask = long_data.get("ask", 0.0)
        if short_bid <= 0 or long_ask <= 0:
            log.info("SKIP: No real bid/ask quotes for selected strikes")
            return
        estimated_credit = max(0, short_bid - long_ask)
    else:
        log.info(f"SKIP: Strikes {short_strike}/{long_strike} not found in option chain")
        return

    min_credit = spread_cfg["min_credit"]
    if estimated_credit < min_credit:
        log.info(f"SKIP: Credit ${estimated_credit:.2f} < min ${min_credit:.2f} "
                 f"(short bid=${short_bid:.2f}, long ask=${long_ask:.2f})")
        return

    log.info(f"Credit: ${estimated_credit:.2f} (short bid=${short_bid:.2f}, long ask=${long_ask:.2f})")

    # Size position
    acct = client.get_account()
    equity = float(acct["equity"])
    contracts = compute_contracts(equity, config, width, estimated_credit)
    if contracts < 1:
        log.info("SKIP: Position too small (0 contracts)")
        return

    max_loss = (width - estimated_credit) * 100 * contracts
    log.info(f"Sizing: {contracts} contracts, max loss ${max_loss:.0f} "
             f"({max_loss / equity * 100:.1f}% of ${equity:,.0f})")

    # Check portfolio risk cap
    open_risk = sum(p.get("max_loss", 0) for p in state.get("positions", [])
                    if p.get("status") == "open")
    total_risk_pct = (open_risk + max_loss) / equity * 100
    max_portfolio_risk = config["sizing"]["max_portfolio_risk_pct"]
    if total_risk_pct > max_portfolio_risk:
        log.info(f"SKIP: Total risk {total_risk_pct:.1f}% > max {max_portfolio_risk}%")
        return

    # Build option symbols (Alpaca format: SPY250502P00530000)
    exp_fmt = datetime.strptime(expiration, "%Y-%m-%d").strftime("%y%m%d")
    short_sym = f"SPY{exp_fmt}P{int(short_strike * 1000):08d}"
    long_sym = f"SPY{exp_fmt}P{int(long_strike * 1000):08d}"

    # Submit order
    log.info(f"{'[DRY-RUN] ' if dry_run else ''}Submitting: SELL {short_sym} / BUY {long_sym} "
             f"× {contracts} @ ${estimated_credit:.2f}")

    order_id = client.submit_spread_order(
        short_sym, long_sym, contracts, estimated_credit,
        short_bid=short_bid, long_ask=long_ask,
    )

    if order_id:
        # Record position
        position = {
            "entry_date": str(date.today()),
            "expiration": expiration,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "contracts": contracts,
            "credit": estimated_credit,
            "max_loss": max_loss,
            "order_id": order_id,
            "status": "open",
            "short_symbol": short_sym,
            "long_symbol": long_sym,
            "vix_at_entry": vix,
            "spy_at_entry": spy_price,
        }
        state.setdefault("positions", []).append(position)
        state["last_entry_date"] = str(date.today())

        log_trade({
            "timestamp": datetime.now().isoformat(), "action": "OPEN",
            "ticker": "SPY", "type": "bull_put_spread",
            "short_strike": short_strike, "long_strike": long_strike,
            "expiration": expiration, "contracts": contracts,
            "credit": estimated_credit, "order_id": order_id, "status": "submitted",
        })

        log.info(f"Position opened: {short_strike}/{long_strike} × {contracts} "
                 f"exp {expiration} (order {order_id})")
    else:
        log.error("Order submission failed")


def show_status(state: dict):
    """Display current positions and P&L."""
    positions = state.get("positions", [])
    open_pos = [p for p in positions if p.get("status") == "open"]
    closed_pos = [p for p in positions if p.get("status") == "closed"]

    print(f"\nEXP-1220 Paper Trading Status")
    print(f"{'═' * 60}")
    print(f"Open positions: {len(open_pos)}")
    print(f"Closed positions: {len(closed_pos)}")
    print(f"Last entry: {state.get('last_entry_date', 'never')}")

    if open_pos:
        print(f"\n{'─' * 60}")
        print(f"  {'Entry':12s} {'Strikes':15s} {'Exp':12s} {'Ctrs':>5s} {'Credit':>7s} {'VIX':>5s}")
        for p in open_pos:
            print(f"  {p['entry_date']:12s} {p['short_strike']:.0f}/{p['long_strike']:.0f}"
                  f"{'':5s} {p['expiration']:12s} {p['contracts']:5d} "
                  f"${p['credit']:.2f}  {p.get('vix_at_entry', 0):.1f}")

    if closed_pos:
        print(f"\n  Recent closed:")
        for p in closed_pos[-5:]:
            print(f"  {p['entry_date']} → {p.get('exit_date','?')} "
                  f"{p['short_strike']:.0f}/{p['long_strike']:.0f} "
                  f"({p.get('exit_reason', '?')})")


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test — validates config + API connectivity without placing orders
# ═══════════════════════════════════════════════════════════════════════════

def run_smoke_test() -> int:
    """Validate config, API connectivity, and data fetch without placing orders.

    Returns 0 on success, non-zero on failure.
    """
    print("\n" + "=" * 60)
    print("  EXP-1220 Smoke Test")
    print("=" * 60 + "\n")

    failures = []
    warnings = []

    # 1. Config file loads and has required fields
    print("[1/6] Loading config...")
    try:
        config = load_config()
        required_sections = ["account", "leverage", "cadence", "sizing",
                              "spread", "entry_signals", "exits", "risk"]
        missing = [s for s in required_sections if s not in config]
        if missing:
            failures.append(f"Config missing sections: {missing}")
        else:
            print(f"      OK  Config loaded: {config['name']}")
            print(f"      OK  Leverage: {config['leverage']['multiplier']}x")
    except Exception as e:
        failures.append(f"Config load failed: {e}")
        print(f"      FAIL {e}")

    # 2. State file is readable (or createable)
    print("[2/6] Checking state file...")
    try:
        state = load_state()
        open_count = sum(1 for p in state.get("positions", [])
                         if p.get("status") == "open")
        print(f"      OK  State loaded: {open_count} open positions")
    except Exception as e:
        failures.append(f"State load failed: {e}")
        print(f"      FAIL {e}")

    # 3. Log directory is writable
    print("[3/6] Checking log directory...")
    try:
        test_file = LOG_DIR / ".smoke_test_write"
        test_file.write_text("test")
        test_file.unlink()
        print(f"      OK  {LOG_DIR} writable")
    except Exception as e:
        failures.append(f"Log dir not writable: {e}")
        print(f"      FAIL {e}")

    # 4. VIX data fetch
    print("[4/6] Fetching VIX from Yahoo...")
    vix = _yahoo_fetch("%5EVIX", max_retries=2)
    if vix is None:
        warnings.append("VIX fetch failed — will use default 20.0 at runtime")
        print(f"      WARN VIX unreachable (will use 20.0 default)")
    else:
        print(f"      OK  VIX = {vix:.2f}")

    # 5. SPY data fetch
    print("[5/6] Fetching SPY from Yahoo...")
    spy = _yahoo_fetch("SPY", max_retries=2)
    if spy is None:
        failures.append("SPY fetch failed — scanner cannot operate without SPY price")
        print(f"      FAIL SPY unreachable")
    else:
        print(f"      OK  SPY = ${spy:.2f}")

    # 6. Alpaca connectivity (only if keys are set — don't fail smoke test if unset)
    print("[6/6] Checking Alpaca connectivity...")
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        warnings.append("ALPACA_API_KEY/ALPACA_SECRET_KEY not set — use --dry-run or set env vars")
        print(f"      WARN Alpaca credentials not set (OK for --dry-run only)")
    else:
        try:
            client = AlpacaClient(dry_run=False)
            acct = client.get_account()
            print(f"      OK  Alpaca account status: {acct.get('status', 'unknown')}")
            print(f"      OK  Equity: ${float(acct.get('equity', 0)):,.2f}")
        except SystemExit:
            failures.append("Alpaca client initialization failed")
            print(f"      FAIL Alpaca client init failed")
        except Exception as e:
            warnings.append(f"Alpaca connectivity issue: {e}")
            print(f"      WARN {e}")

    # Summary
    print()
    print("=" * 60)
    if failures:
        print(f"  SMOKE TEST FAILED ({len(failures)} errors, {len(warnings)} warnings)")
        for f in failures:
            print(f"    ERROR: {f}")
        for w in warnings:
            print(f"    WARN:  {w}")
        write_health("error", {"smoke_test": "failed"}, error="; ".join(failures))
        return 1
    elif warnings:
        print(f"  SMOKE TEST PASSED WITH WARNINGS ({len(warnings)} warnings)")
        for w in warnings:
            print(f"    WARN: {w}")
        write_health("warning", {"smoke_test": "passed_with_warnings",
                                   "warnings": warnings})
        return 0
    else:
        print(f"  SMOKE TEST PASSED — All 6 checks OK")
        write_health("ok", {"smoke_test": "passed"})
        return 0


def run_health_check() -> int:
    """Write current health status to file. Returns 0 on ok, 1 on warning, 2 on error."""
    try:
        config = load_config()
        state = load_state()
        open_positions = sum(1 for p in state.get("positions", [])
                              if p.get("status") == "open")
        last_entry = state.get("last_entry_date", "never")

        # Check staleness — if last run was >2 days ago, warn
        if HEALTH_PATH.exists():
            prev = json.loads(HEALTH_PATH.read_text())
            last_run = prev.get("last_run", "")
            if last_run:
                try:
                    last_dt = datetime.fromisoformat(last_run)
                    age_hours = (datetime.now() - last_dt).total_seconds() / 3600
                    if age_hours > 48:
                        write_health("warning",
                                      {"open_positions": open_positions,
                                       "last_entry": str(last_entry),
                                       "hours_since_last_run": round(age_hours, 1)},
                                      error=f"Scanner hasn't run in {age_hours:.0f}h")
                        print(f"WARNING: Scanner hasn't run in {age_hours:.0f}h")
                        return 1
                except Exception:
                    pass

        write_health("ok", {
            "open_positions": open_positions,
            "last_entry": str(last_entry),
            "config_name": config.get("name", ""),
            "leverage": config.get("leverage", {}).get("multiplier", 0),
        })
        print(f"HEALTH: OK (open={open_positions}, last_entry={last_entry})")
        return 0
    except Exception as e:
        write_health("error", error=str(e))
        print(f"HEALTH: ERROR — {e}")
        return 2


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EXP-1220 Paper Trading Scanner")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show trades without submitting to Alpaca")
    parser.add_argument("--status", action="store_true",
                        help="Show current positions")
    parser.add_argument("--close-all", action="store_true",
                        help="Close all open positions")
    parser.add_argument("--force-scan", action="store_true",
                        help="Force scan even if not scan day")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Validate config + API connectivity, no trading")
    parser.add_argument("--health", action="store_true",
                        help="Write health status file and exit")
    parser.add_argument("--config", type=str, default=None,
                        help=("Config YAML path (default: configs/deploy_exp1220_1.5x.yaml). "
                              "Multi-instance safe: state/journal/health files are namespaced "
                              "by the config filename stem."))
    args = parser.parse_args()

    # Route state/log/health files to an instance-specific namespace
    # derived from the config filename.
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = ROOT / cfg_path
        if not cfg_path.exists():
            print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
            sys.exit(2)
        global _CONFIG_PATH
        _CONFIG_PATH = cfg_path
        # Derive instance id from filename stem (e.g. paper_exp1220_3x)
        set_instance_id(cfg_path.stem)
        # Re-initialize logger with new instance paths
        global log
        log = setup_logging()

    if args.smoke_test:
        sys.exit(run_smoke_test())
    if args.health:
        sys.exit(run_health_check())

    config = load_config()
    state = load_state()

    log.info(f"EXP-1220 Scanner — {config['name']}")
    log.info(f"Mode: {'DRY-RUN' if args.dry_run else 'PAPER'}")

    if args.status:
        show_status(state)
        return

    client = AlpacaClient(dry_run=args.dry_run)

    if args.close_all:
        log.info("Closing all positions...")
        for pos in state.get("positions", []):
            if pos.get("status") == "open":
                # FIX 4: Submit actual closing orders for both legs
                short_sym = pos.get("short_symbol", "")
                long_sym = pos.get("long_symbol", "")
                n_contracts = pos.get("contracts", 1)
                if short_sym and long_sym:
                    client.close_spread(short_sym, long_sym, n_contracts)
                pos["status"] = "closed"
                pos["exit_reason"] = "manual_close"
                pos["exit_date"] = str(date.today())
                log.info(f"Closed: {pos['short_strike']}/{pos['long_strike']}")
        save_state(state)
        return

    # Check exits first
    spy_price = get_spy_price()
    check_exits(state, config, client, spy_price)

    # FIX 5: Wrap execution in try/finally to always save state
    try:
        # Check VIX emergency
        vix = get_vix()
        emergency_vix = config["exits"]["vix_emergency_exit"]
        if vix > emergency_vix:
            log.warning(f"VIX EMERGENCY: {vix:.1f} > {emergency_vix}. Closing all positions.")
            for pos in state.get("positions", []):
                if pos.get("status") == "open":
                    # FIX 4: Submit actual closing orders
                    short_sym = pos.get("short_symbol", "")
                    long_sym = pos.get("long_symbol", "")
                    if short_sym and long_sym:
                        client.close_spread(short_sym, long_sym, pos.get("contracts", 1))
                    pos["status"] = "closed"
                    pos["exit_reason"] = f"vix_emergency ({vix:.1f})"
                    pos["exit_date"] = str(date.today())
            return

        # Check if we should scan
        should, reason = should_scan(config, state)
        if not should and not args.force_scan:
            log.info(f"Scan skipped: {reason}")
            return

        if args.force_scan and not should:
            log.info(f"Force scan override (normal reason: {reason})")

        # Run the scan
        run_scan(config, state, client, args.dry_run)
        log.info("Scan complete.")

        # Write health: OK
        open_positions = sum(1 for p in state.get("positions", [])
                              if p.get("status") == "open")
        write_health("ok", {
            "open_positions": open_positions,
            "last_entry": str(state.get("last_entry_date", "")),
            "mode": "dry-run" if args.dry_run else "paper",
        })

    except Exception as e:
        log.error(f"Scanner error: {e}", exc_info=True)
        write_health("error", error=str(e))
    finally:
        # FIX 5: Always save state, even on crash
        save_state(state)
        log.info("State saved.")


if __name__ == "__main__":
    main()
