#!/usr/bin/env python3
"""
Fetch and cache real Deribit BTC historical option data (2020–2024).

Architecture:
  Phase 0 — BTC/USD daily spot prices from CoinGecko → btc_spot table
  Phase 1 — Strike discovery: enumerate monthly expiry dates (Jan 2020–Dec 2024),
             probe candidate strikes per expiry (parallel, threaded),
             confirm which instruments actually exist on Deribit
  Phase 2 — Daily OHLCV bars for every confirmed instrument (parallel)
  Phase 3 — Backfill underlying_price into option rows from btc_spot

All data stored in data/deribit_btc_cache.db.
All prices in BTC (Deribit native) + underlying_price in USD from btc_spot.

Usage:
    python3 scripts/fetch_deribit_btc_options.py
    python3 scripts/fetch_deribit_btc_options.py --workers 12
    python3 scripts/fetch_deribit_btc_options.py --dry-run
    python3 scripts/fetch_deribit_btc_options.py --reset
    python3 scripts/fetch_deribit_btc_options.py --years 2020 2021 2022
"""

import argparse
import calendar
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env.exp400")
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "deribit_btc_cache.db"
CHECKPOINT_PATH = DATA_DIR / "deribit_discovery_progress.json"
BARS_CHECKPOINT_PATH = DATA_DIR / "deribit_bars_progress.json"

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"

MONTH_ABBR = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA cache_size=-32768")
    return conn


def init_db():
    conn = open_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS btc_spot (
            date        TEXT PRIMARY KEY,
            price_usd   REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS btc_contracts (
            instrument_name TEXT PRIMARY KEY,
            expiration_date TEXT NOT NULL,
            strike          REAL NOT NULL,
            option_type     TEXT NOT NULL,
            confirmed_date  TEXT
        );

        CREATE TABLE IF NOT EXISTS btc_option_daily (
            instrument_name  TEXT NOT NULL,
            date             TEXT NOT NULL,
            open             REAL,
            high             REAL,
            low              REAL,
            close            REAL,
            volume           REAL,
            underlying_price REAL,
            PRIMARY KEY (instrument_name, date)
        );

        CREATE INDEX IF NOT EXISTS idx_btc_contracts_expiry
            ON btc_contracts (expiration_date);
        CREATE INDEX IF NOT EXISTS idx_btc_daily_inst
            ON btc_option_daily (instrument_name);
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Per-thread Deribit session
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def get_deribit_session() -> requests.Session:
    if not hasattr(_thread_local, "deribit_session"):
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=1.0,
                      status_forcelist=[500, 502, 503, 504],
                      backoff_jitter=0.2)
        s.mount("https://", HTTPAdapter(max_retries=retry))
        _thread_local.deribit_session = s
        _thread_local.deribit_last = 0.0
    return _thread_local.deribit_session


def deribit_get(method: str, params: dict, min_interval: float = 0.05) -> Optional[dict]:
    """Single Deribit GET with per-thread rate limiting."""
    session = get_deribit_session()
    wait = min_interval - (time.time() - getattr(_thread_local, "deribit_last", 0.0))
    if wait > 0:
        time.sleep(wait)
    url = f"{DERIBIT_BASE}/{method}"
    try:
        resp = session.get(url, params=params, timeout=15)
        _thread_local.deribit_last = time.time()
        if resp.status_code == 429:
            log.warning("Deribit rate limited — sleeping 5s")
            time.sleep(5)
            return deribit_get(method, params, min_interval)
        if resp.status_code == 400:
            return None  # instrument doesn't exist
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {})
    except requests.exceptions.RequestException as e:
        log.debug("Deribit error %s %s: %s", method, params, e)
        _thread_local.deribit_last = time.time()
        return None


# ---------------------------------------------------------------------------
# Expiry date utilities
# ---------------------------------------------------------------------------

def last_friday_of_month(year: int, month: int) -> date:
    """Return the last Friday of the given month (Deribit monthly expiry)."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    days_back = (d.weekday() - 4) % 7  # weekday 4 = Friday
    return d - timedelta(days=days_back)


def expiry_to_instrument_prefix(exp: date) -> str:
    """Convert a date to the Deribit instrument name date component, e.g. '27MAR20'."""
    return f"{exp.day}{MONTH_ABBR[exp.month]}{str(exp.year)[-2:]}"


def all_monthly_expiries(years: List[int]) -> List[date]:
    """Return all last-Friday-of-month dates for the given years."""
    expiries = []
    for year in years:
        for month in range(1, 13):
            expiries.append(last_friday_of_month(year, month))
    return sorted(expiries)


# ---------------------------------------------------------------------------
# Strike candidate generation
# ---------------------------------------------------------------------------

def strike_candidates(spot: float) -> List[int]:
    """
    Generate probe candidates based on BTC spot price.
    Covers the range needed for OTM put credit spreads (~50–110% of spot).
    """
    if spot <= 0:
        return []

    # Deribit standard increment near ATM
    if spot < 5_000:
        inc = 250
    elif spot < 15_000:
        inc = 500
    else:
        inc = 1_000

    candidates: Set[int] = set()

    # Fine grid: 45% to 115% of spot (covers all credit spread entries)
    lo = max(inc, int(spot * 0.45 / inc) * inc)
    hi = int(spot * 1.15 / inc + 1) * inc
    for s in range(lo, hi + inc, inc):
        candidates.add(s)

    # Coarse grid: 20–45% of spot (deep OTM, large increments)
    coarse = max(inc * 5, 5_000)
    for mult in [0.20, 0.25, 0.30, 0.35, 0.40]:
        s = round(spot * mult / coarse) * coarse
        if s > 0:
            candidates.add(s)

    return sorted(candidates)


# ---------------------------------------------------------------------------
# Phase 0: BTC spot prices from CoinGecko
# ---------------------------------------------------------------------------

def fetch_btc_spot_deribit_perpetual(date_from: str, date_to: str) -> List[Tuple[str, float]]:
    """
    Fetch daily BTC/USD prices from Deribit BTC-PERPETUAL futures.
    Price tracks spot extremely closely (funding-rate adjusted).
    No API key required.
    Returns [(date_str, close_usd), ...].
    """
    from_ts = int(datetime.strptime(date_from, "%Y-%m-%d").timestamp() * 1000)
    to_ts   = int(datetime.strptime(date_to,   "%Y-%m-%d").timestamp() * 1000)

    rows: List[Tuple[str, float]] = []
    chunk_size = 90 * 86_400_000  # 90-day chunks in ms (Deribit daily bar limit ~500)
    cursor = from_ts

    while cursor < to_ts:
        end = min(cursor + chunk_size, to_ts)
        result = deribit_get("get_tradingview_chart_data", {
            "instrument_name": "BTC-PERPETUAL",
            "start_timestamp": cursor,
            "end_timestamp":   end,
            "resolution":      "1D",
        }, min_interval=0.25)
        if result:
            ticks  = result.get("ticks",  [])
            closes = result.get("close",  [])
            for i, ts in enumerate(ticks):
                dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                if i < len(closes) and closes[i] is not None:
                    rows.append((dt, float(closes[i])))
        cursor = end + 1
        time.sleep(0.3)

    # Deduplicate
    seen: Set[str] = set()
    deduped = []
    for dt, p in sorted(rows):
        if dt not in seen:
            seen.add(dt)
            deduped.append((dt, p))
    return deduped


def fetch_btc_spot_binance(date_from: str, date_to: str) -> List[Tuple[str, float]]:
    """
    Fallback: Fetch daily BTC/USDT close prices from Binance public API.
    No API key required.
    Returns [(date_str, close_usd), ...].
    """
    from_ts = int(datetime.strptime(date_from, "%Y-%m-%d").timestamp() * 1000)
    to_ts   = int(datetime.strptime(date_to,   "%Y-%m-%d").timestamp() * 1000)

    url = "https://api.binance.com/api/v3/klines"
    rows: List[Tuple[str, float]] = []
    cursor = from_ts

    while cursor < to_ts:
        try:
            resp = requests.get(url, params={
                "symbol":    "BTCUSDT",
                "interval":  "1d",
                "startTime": cursor,
                "endTime":   min(cursor + 1000 * 86_400_000, to_ts),
                "limit":     1000,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for kline in data:
                ts_ms = kline[0]
                close = float(kline[4])
                dt = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
                rows.append((dt, close))
            cursor = data[-1][0] + 86_400_000  # advance past last bar
            time.sleep(0.2)
        except Exception as e:
            log.warning("Binance fetch error: %s", e)
            break

    seen: Set[str] = set()
    deduped = []
    for dt, p in sorted(rows):
        if dt not in seen:
            seen.add(dt)
            deduped.append((dt, p))
    return deduped


def fetch_btc_spot(date_from: str, date_to: str) -> List[Tuple[str, float]]:
    """
    Fetch BTC/USD daily spot prices.
    Primary: Deribit BTC-PERPETUAL (same data source as options).
    Fallback: Binance public API.
    """
    log.info("  Trying Deribit BTC-PERPETUAL as spot source...")
    rows = fetch_btc_spot_deribit_perpetual(date_from, date_to)
    if len(rows) > 100:
        log.info("  Deribit BTC-PERPETUAL: %d rows", len(rows))
        return rows
    log.info("  Deribit returned %d rows — falling back to Binance", len(rows))
    rows = fetch_btc_spot_binance(date_from, date_to)
    log.info("  Binance BTCUSDT: %d rows", len(rows))
    return rows


def store_btc_spot(rows: List[Tuple[str, float]]):
    conn = open_db()
    conn.executemany(
        "INSERT OR IGNORE INTO btc_spot (date, price_usd) VALUES (?, ?)", rows
    )
    conn.commit()
    conn.close()


def load_btc_spot() -> Dict[str, float]:
    """Load all BTC/USD spot prices as {date_str: price}."""
    conn = open_db()
    rows = conn.execute("SELECT date, price_usd FROM btc_spot").fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Phase 1: Strike discovery (parallel probe)
# ---------------------------------------------------------------------------

def probe_instrument(
    instrument_name: str,
    expiry_date: date,
    strike: float,
    option_type: str,
) -> Optional[dict]:
    """
    Check if an instrument exists on Deribit by requesting a narrow date window.
    Returns instrument metadata if confirmed, None if not found.
    """
    # Request 5 days before expiry — if it exists, it should have data
    end_ts   = int(datetime.combine(expiry_date, datetime.min.time()).timestamp() * 1000)
    start_ts = end_ts - 5 * 86_400_000  # 5 days before expiry

    result = deribit_get("get_tradingview_chart_data", {
        "instrument_name": instrument_name,
        "start_timestamp": start_ts,
        "end_timestamp":   end_ts,
        "resolution":      "1D",
    })

    if result is None:
        return None  # 400 or error — instrument doesn't exist

    ticks = result.get("ticks", [])
    if not ticks and result.get("status") != "ok":
        return None

    return {
        "instrument_name": instrument_name,
        "expiration_date":  expiry_date.strftime("%Y-%m-%d"),
        "strike":           strike,
        "option_type":      option_type,
        "confirmed_date":   date.today().strftime("%Y-%m-%d"),
    }


def discover_strikes_for_expiry(
    expiry: date,
    btc_spot: Dict[str, float],
    option_type: str = "P",
) -> List[dict]:
    """
    Probe all strike candidates for one expiry.
    btc_spot must contain price for ~35 days before expiry.
    """
    # Find BTC price ~35 DTE before expiry
    target_entry = expiry - timedelta(days=35)
    spot = None
    for delta in range(0, 10):
        d = (target_entry - timedelta(days=delta)).strftime("%Y-%m-%d")
        if d in btc_spot:
            spot = btc_spot[d]
            break
    if spot is None:
        log.warning("No BTC spot price found near %s — skipping", expiry)
        return []

    prefix = expiry_to_instrument_prefix(expiry)
    candidates = strike_candidates(spot)
    confirmed = []

    for strike in candidates:
        inst_name = f"BTC-{prefix}-{strike}-{option_type}"
        result = probe_instrument(inst_name, expiry, strike, option_type)
        if result:
            confirmed.append(result)

    return confirmed


def store_contracts(contracts: List[dict]):
    if not contracts:
        return
    conn = open_db()
    conn.executemany(
        "INSERT OR IGNORE INTO btc_contracts "
        "(instrument_name, expiration_date, strike, option_type, confirmed_date) "
        "VALUES (:instrument_name, :expiration_date, :strike, :option_type, :confirmed_date)",
        contracts,
    )
    conn.commit()
    conn.close()


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, data: dict):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# Phase 2: Daily bar fetch for confirmed instruments
# ---------------------------------------------------------------------------

def fetch_bars_for_instrument(
    instrument_name: str,
    expiry_date: str,
) -> Tuple[str, Optional[List[tuple]]]:
    """
    Fetch full daily OHLCV history for one instrument.
    Fetches from 90 days before expiry to expiry (covers entry window).
    Returns (instrument_name, rows | [] | None).
    """
    exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    start  = exp_dt - timedelta(days=90)
    start_ts = int(start.timestamp() * 1000)
    end_ts   = int(exp_dt.timestamp() * 1000)

    result = deribit_get("get_tradingview_chart_data", {
        "instrument_name": instrument_name,
        "start_timestamp": start_ts,
        "end_timestamp":   end_ts,
        "resolution":      "1D",
    })

    if result is None:
        return instrument_name, None  # error

    ticks  = result.get("ticks",  [])
    opens  = result.get("open",   [])
    highs  = result.get("high",   [])
    lows   = result.get("low",    [])
    closes = result.get("close",  [])
    vols   = result.get("volume", [])

    if not ticks:
        return instrument_name, []  # sentinel

    rows = []
    for i, ts in enumerate(ticks):
        dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        rows.append((
            instrument_name, dt,
            opens[i]  if i < len(opens)  else None,
            highs[i]  if i < len(highs)  else None,
            lows[i]   if i < len(lows)   else None,
            closes[i] if i < len(closes) else None,
            vols[i]   if i < len(vols)   else None,
            None,  # underlying_price filled in Phase 3
        ))
    return instrument_name, rows


_thread_db: Dict[int, sqlite3.Connection] = {}
_db_lock = threading.Lock()


def get_thread_db() -> sqlite3.Connection:
    tid = threading.get_ident()
    with _db_lock:
        if tid not in _thread_db:
            _thread_db[tid] = open_db()
    return _thread_db[tid]


def write_bars(instrument_name: str, rows: Optional[List[tuple]]):
    conn = get_thread_db()
    if rows is None:
        return
    if len(rows) == 0:
        conn.execute(
            "INSERT OR IGNORE INTO btc_option_daily (instrument_name, date) VALUES (?, '0000-00-00')",
            (instrument_name,),
        )
    else:
        conn.executemany(
            "INSERT OR IGNORE INTO btc_option_daily "
            "(instrument_name, date, open, high, low, close, volume, underlying_price) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Phase 3: Backfill underlying_price
# ---------------------------------------------------------------------------

def backfill_underlying_price():
    """Join btc_spot into btc_option_daily.underlying_price."""
    conn = open_db()
    conn.execute("""
        UPDATE btc_option_daily
        SET underlying_price = (
            SELECT price_usd FROM btc_spot WHERE date = btc_option_daily.date
        )
        WHERE underlying_price IS NULL
          AND date != '0000-00-00'
    """)
    updated = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    log.info("Phase 3: backfilled underlying_price for %d rows", updated)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    conn = open_db()

    n_spot = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM btc_spot").fetchone()
    n_contracts = conn.execute("SELECT COUNT(*) FROM btc_contracts").fetchone()[0]
    by_type = conn.execute(
        "SELECT option_type, COUNT(*), MIN(expiration_date), MAX(expiration_date) "
        "FROM btc_contracts GROUP BY option_type"
    ).fetchall()
    n_bars = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT instrument_name), MIN(date), MAX(date) "
        "FROM btc_option_daily WHERE date != '0000-00-00'"
    ).fetchone()
    n_sentinel = conn.execute(
        "SELECT COUNT(*) FROM btc_option_daily WHERE date = '0000-00-00'"
    ).fetchone()[0]
    n_with_spot = conn.execute(
        "SELECT COUNT(*) FROM btc_option_daily "
        "WHERE date != '0000-00-00' AND underlying_price IS NOT NULL"
    ).fetchone()[0]

    conn.close()

    log.info("=== Deribit BTC Cache Summary ===")
    log.info("  BTC spot rows:       %d  (%s → %s)", n_spot[0], n_spot[1], n_spot[2])
    log.info("  Contracts confirmed: %d", n_contracts)
    for row in by_type:
        log.info("    %s: %d contracts (%s → %s)", *row)
    log.info("  Option bar rows:     %d  (%d instruments)", n_bars[0], n_bars[1])
    log.info("  Sentinel rows:       %d", n_sentinel)
    log.info("  Bar date range:      %s → %s", n_bars[2], n_bars[3])
    log.info("  Rows with spot px:   %d", n_with_spot)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Deribit BTC historical option data")
    parser.add_argument("--years", type=int, nargs="+",
                        default=list(range(2020, 2025)),
                        help="Years to fetch (default: 2020–2024)")
    parser.add_argument("--option-type", default="P", choices=["P", "C", "both"],
                        help="Option type(s) to probe (default: P = puts only)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers for probing and bar fetch (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discovery only, skip bar fetch")
    parser.add_argument("--reset", action="store_true",
                        help="Clear checkpoints and restart")
    parser.add_argument("--skip-discovery", action="store_true",
                        help="Skip Phase 1, use existing btc_contracts table")
    args = parser.parse_args()

    log.info("=== Deribit BTC Options Fetcher ===")
    log.info("Years:     %s", args.years)
    log.info("Type:      %s", args.option_type)
    log.info("Workers:   %d", args.workers)
    log.info("DB:        %s", DB_PATH)

    DATA_DIR.mkdir(exist_ok=True)
    init_db()

    if args.reset:
        for p in (CHECKPOINT_PATH, BARS_CHECKPOINT_PATH):
            if p.exists():
                p.unlink()
                log.info("Cleared checkpoint: %s", p.name)

    option_types = ["P", "C"] if args.option_type == "both" else [args.option_type]

    # -----------------------------------------------------------------------
    # Phase 0: BTC spot prices
    # -----------------------------------------------------------------------
    log.info("")
    log.info("Phase 0: Fetching BTC/USD daily spot from CoinGecko...")
    # Fetch 90 days before start of first year to end of last year + 30 days
    first_year = min(args.years)
    last_year  = max(args.years)
    date_from  = f"{first_year - 1}-10-01"  # 90 days before Jan 1 of first year
    date_to    = f"{last_year + 1}-03-31"

    all_spot_rows = fetch_btc_spot(date_from, date_to)
    store_btc_spot(all_spot_rows)
    btc_spot = load_btc_spot()
    log.info("Phase 0 done: %d BTC spot rows (%s → %s)",
             len(btc_spot),
             min(btc_spot) if btc_spot else "?",
             max(btc_spot) if btc_spot else "?")

    if not btc_spot:
        log.error("No BTC spot data — cannot continue")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase 1: Strike discovery
    # -----------------------------------------------------------------------
    if args.skip_discovery:
        log.info("")
        log.info("Phase 1: Skipped — using existing btc_contracts table")
    else:
        log.info("")
        log.info("Phase 1: Strike discovery for monthly expiries %s–%s...", min(args.years), max(args.years))

        disc_cp = load_checkpoint(CHECKPOINT_PATH)
        already_probed: Set[str] = set(disc_cp.get("probed_expiries", []))

        expiries = all_monthly_expiries(args.years)
        log.info("Phase 1: %d monthly expiries to probe", len(expiries))

        total_confirmed = 0
        probe_lock = threading.Lock()

        def probe_expiry(expiry: date, opt_type: str) -> Tuple[str, List[dict]]:
            key = f"{expiry.isoformat()}_{opt_type}"
            if key in already_probed:
                return key, []
            confirmed = discover_strikes_for_expiry(expiry, btc_spot, opt_type)
            return key, confirmed

        todo = [(exp, ot) for exp in expiries for ot in option_types
                if f"{exp.isoformat()}_{ot}" not in already_probed]

        log.info("Phase 1: %d expiry×type combinations to probe (%d already cached)",
                 len(todo), len(already_probed))

        done_discovery = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(probe_expiry, exp, ot): (exp, ot) for exp, ot in todo}
            for future in as_completed(futures):
                key, confirmed = future.result()
                with probe_lock:
                    done_discovery += 1
                    if confirmed:
                        store_contracts(confirmed)
                        total_confirmed += len(confirmed)
                    already_probed.add(key)
                    if done_discovery % 10 == 0 or done_discovery == len(todo):
                        log.info("  Probed %d/%d expiries | %d instruments confirmed",
                                 done_discovery, len(todo), total_confirmed)
                        save_checkpoint(CHECKPOINT_PATH,
                                        {"probed_expiries": list(already_probed)})

        save_checkpoint(CHECKPOINT_PATH, {"probed_expiries": list(already_probed)})

        # Report by year
        conn = open_db()
        for year in sorted(args.years):
            n = conn.execute(
                "SELECT COUNT(*) FROM btc_contracts WHERE expiration_date LIKE ?",
                (f"{year}%",)
            ).fetchone()[0]
            log.info("  %d: %d confirmed instruments", year, n)
        conn.close()

    if args.dry_run:
        log.info("Dry run — stopping after discovery")
        print_summary()
        return

    # -----------------------------------------------------------------------
    # Phase 2: Fetch daily bars for all confirmed instruments
    # -----------------------------------------------------------------------
    log.info("")
    log.info("Phase 2: Fetching daily OHLCV for confirmed instruments...")

    conn = open_db()
    all_instruments = conn.execute(
        "SELECT instrument_name, expiration_date FROM btc_contracts"
    ).fetchall()

    # Find which already have data
    has_data = {r[0] for r in conn.execute(
        "SELECT DISTINCT instrument_name FROM btc_option_daily"
    ).fetchall()}
    conn.close()

    to_fetch = [(inst, exp) for inst, exp in all_instruments if inst not in has_data]
    total = len(to_fetch)
    log.info("Phase 2: %d instruments to fetch | %d already cached", total, len(has_data))

    bars_cp = load_checkpoint(BARS_CHECKPOINT_PATH)
    already_fetched: Set[str] = set(bars_cp.get("fetched", []))
    to_fetch = [(inst, exp) for inst, exp in to_fetch if inst not in already_fetched]
    log.info("Phase 2: %d after checkpoint filter", len(to_fetch))

    if to_fetch:
        done = errors = sentinels = 0
        start_t = time.time()
        bars_lock = threading.Lock()
        batch: List[str] = []
        SAVE_EVERY = 100

        def process_bars(inst_exp: Tuple[str, str]) -> Tuple[str, str]:
            inst, exp = inst_exp
            name, rows = fetch_bars_for_instrument(inst, exp)
            write_bars(name, rows)
            if rows is None:
                return name, "error"
            elif len(rows) == 0:
                return name, "sentinel"
            return name, f"ok:{len(rows)}"

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_bars, ie): ie for ie in to_fetch}
            for future in as_completed(futures):
                name, status = future.result()
                with bars_lock:
                    done += 1
                    if status == "error":
                        errors += 1
                    else:
                        if status == "sentinel":
                            sentinels += 1
                        already_fetched.add(name)
                        batch.append(name)

                    if len(batch) >= SAVE_EVERY:
                        bars_cp["fetched"] = list(already_fetched)
                        save_checkpoint(BARS_CHECKPOINT_PATH, bars_cp)
                        batch.clear()

                    if done % 100 == 0 or done == len(to_fetch):
                        elapsed = time.time() - start_t
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (len(to_fetch) - done) / rate if rate > 0 else 0
                        log.info("[%d/%d] %.1f%%  %.1f req/s  ETA %.0fm  err=%d sent=%d",
                                 done, len(to_fetch), 100 * done / len(to_fetch),
                                 rate, eta / 60, errors, sentinels)

        bars_cp["fetched"] = list(already_fetched)
        save_checkpoint(BARS_CHECKPOINT_PATH, bars_cp)
        elapsed_min = (time.time() - start_t) / 60
        log.info("Phase 2 done: %d ok | %d sentinel | %d errors in %.1f min",
                 done - errors - sentinels, sentinels, errors, elapsed_min)
    else:
        log.info("Phase 2: nothing to fetch")

    # -----------------------------------------------------------------------
    # Phase 3: Backfill underlying price
    # -----------------------------------------------------------------------
    log.info("")
    log.info("Phase 3: Backfilling BTC spot prices into option rows...")
    backfill_underlying_price()

    log.info("")
    print_summary()


if __name__ == "__main__":
    main()
