#!/usr/bin/env python3
"""
TLT Option Data Backfill — July 2024 to December 2025.

Polygon Starter tier cannot enumerate option contracts, so we construct
OCC symbols ourselves for monthly TLT options and fetch daily bars.

TLT price range (2024-2025): ~$82-$100. We generate put+call contracts
for strikes $70-$110 at $1 intervals for monthly expirations.

Usage:
    python3 scripts/backfill_tlt.py                  # full backfill
    python3 scripts/backfill_tlt.py --dry-run        # report only
    python3 scripts/backfill_tlt.py --workers 8      # more parallelism
"""

import argparse
import logging
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "options_cache.db"
BASE_URL = "https://api.polygon.io"

TICKER = "TLT"
GAP_START = "2024-07-20"       # day after last TLT data
GAP_END = "2025-12-31"         # target end date

# TLT traded ~$82-$100 in 2024-2025. Cover $70-$115 for safety.
STRIKE_MIN = 70
STRIKE_MAX = 115
STRIKE_STEP = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Monthly expirations (3rd Friday of each month) ───────────────────────

def third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given month."""
    d = date(year, month, 1)
    # Find first Friday
    days_until_friday = (4 - d.weekday()) % 7
    first_friday = d + timedelta(days=days_until_friday)
    return first_friday + timedelta(weeks=2)


def generate_monthly_expirations(start: str, end: str) -> List[date]:
    """Generate monthly option expirations between start and end dates."""
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    expirations = []

    y, m = start_dt.year, start_dt.month
    while True:
        exp = third_friday(y, m)
        if exp > end_dt:
            break
        if exp >= start_dt:
            expirations.append(exp)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return expirations


# ── OCC symbol construction ──────────────────────────────────────────────

def occ_symbol(ticker: str, expiration: date, option_type: str, strike: float) -> str:
    """Construct OCC option symbol.

    Format: O:TLT241220P00085000
    Ticker (padded to 6), YYMMDD, P/C, strike×1000 (8 digits zero-padded)
    """
    exp_str = expiration.strftime("%y%m%d")
    strike_int = int(strike * 1000)
    return f"O:{ticker}{exp_str}{option_type}{strike_int:08d}"


def generate_symbols(expirations: List[date]) -> List[Tuple[str, date, str, float]]:
    """Generate all OCC symbols for puts and calls across strikes and expirations."""
    symbols = []
    for exp in expirations:
        for strike in range(STRIKE_MIN, STRIKE_MAX + 1, STRIKE_STEP):
            for opt_type in ["P", "C"]:
                sym = occ_symbol(TICKER, exp, opt_type, float(strike))
                symbols.append((sym, exp, opt_type, float(strike)))
    return symbols


# ── Database ─────────────────────────────────────────────────────────────

def open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def ensure_contract_row(conn: sqlite3.Connection, symbol: str,
                        expiration: date, opt_type: str, strike: float):
    """Insert contract into option_contracts if not exists."""
    exp_str = expiration.strftime("%Y-%m-%d")
    conn.execute(
        "INSERT OR IGNORE INTO option_contracts "
        "(ticker, expiration, strike, option_type, contract_symbol, as_of_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (TICKER, exp_str, strike, opt_type, symbol, exp_str),
    )


def count_existing_bars(conn: sqlite3.Connection, symbol: str,
                        date_from: str, date_to: str) -> int:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM option_daily WHERE contract_symbol=? AND date BETWEEN ? AND ?",
        (symbol, date_from, date_to),
    )
    return cur.fetchone()[0]


# ── Polygon API ──────────────────────────────────────────────────────────

_thread_local = threading.local()


def get_session(api_key: str) -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        retry = Retry(total=4, backoff_factor=1.5,
                      status_forcelist=[429, 500, 502, 503, 504],
                      backoff_jitter=0.3)
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.params = {"apiKey": api_key}
        _thread_local.session = s
        _thread_local.last_call = 0.0
    return _thread_local.session


def fetch_bars(symbol: str, date_from: str, date_to: str,
               api_key: str) -> Optional[List[Tuple]]:
    """Fetch daily OHLCV bars from Polygon."""
    session = get_session(api_key)

    # Rate limit: ~80ms between calls (Starter: 5 calls/min → 12s, but
    # daily bars endpoint is usually unlimited on Starter)
    wait = 0.08 - (time.time() - _thread_local.last_call)
    if wait > 0:
        time.sleep(wait)

    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{date_from}/{date_to}"
    try:
        resp = session.get(url, params={
            "adjusted": "true", "sort": "asc", "limit": 5000,
        }, timeout=30)
        _thread_local.last_call = time.time()

        if resp.status_code == 429:
            log.warning("Rate limited — sleeping 12s")
            time.sleep(12)
            return fetch_bars(symbol, date_from, date_to, api_key)

        if resp.status_code == 404:
            return []  # symbol doesn't exist

        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        log.error("API error %s: %s", symbol, e)
        _thread_local.last_call = time.time()
        return None

    results = data.get("results", [])
    if not results:
        return []

    rows = []
    for bar in results:
        ts = bar.get("t", 0)
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append((
            symbol, dt,
            bar.get("o"), bar.get("h"), bar.get("l"), bar.get("c"),
            bar.get("v", 0), bar.get("oi"),
        ))
    return rows


# ── Main backfill ────────────────────────────────────────────────────────

def run_backfill(workers: int = 4, dry_run: bool = False):
    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key and not dry_run:
        log.error("POLYGON_API_KEY not set. Use --dry-run or set the env var.")
        sys.exit(1)

    # Generate symbols
    expirations = generate_monthly_expirations(GAP_START, GAP_END)
    all_symbols = generate_symbols(expirations)

    print(f"TLT Backfill: {GAP_START} → {GAP_END}")
    print(f"  Expirations: {len(expirations)} monthly ({expirations[0]} to {expirations[-1]})")
    print(f"  Strikes: ${STRIKE_MIN}–${STRIKE_MAX} (step ${STRIKE_STEP})")
    print(f"  Symbols generated: {len(all_symbols)} (puts + calls)")
    print()

    if dry_run:
        print("DRY RUN — no API calls or DB writes.")
        print(f"\nSample symbols:")
        for sym, exp, ot, sk in all_symbols[:8]:
            print(f"  {sym}  exp={exp}  {ot}  strike=${sk:.0f}")
        return

    # Phase 1: Register contracts in DB
    conn = open_db()
    new_contracts = 0
    for sym, exp, opt_type, strike in all_symbols:
        ensure_contract_row(conn, sym, exp, opt_type, strike)
        new_contracts += 1
    conn.commit()
    conn.close()
    print(f"Phase 1: {new_contracts} contracts registered in option_contracts")

    # Phase 2: Fetch bars (skip symbols that already have data)
    conn = open_db()
    to_fetch = []
    for sym, exp, opt_type, strike in all_symbols:
        # Only fetch if we don't already have bars in the gap period
        existing = count_existing_bars(conn, sym, GAP_START, GAP_END)
        if existing == 0:
            to_fetch.append((sym, exp))
    conn.close()

    print(f"Phase 2: {len(to_fetch)} symbols to fetch ({len(all_symbols) - len(to_fetch)} already cached)")

    if not to_fetch:
        print("Nothing to fetch — all data already cached.")
        return

    # Fetch in parallel
    done = 0
    total_bars = 0
    empty = 0
    errors = 0
    lock = threading.Lock()

    # Collect all rows, then batch-write from main thread (avoids DB lock contention)
    all_rows: List[Tuple] = []
    rows_lock = threading.Lock()

    def process(item):
        sym, exp = item
        fetch_to = min(exp.strftime("%Y-%m-%d"), GAP_END)
        rows = fetch_bars(sym, GAP_START, fetch_to, api_key)
        if rows is None:
            return sym, -1, []
        return sym, len(rows), rows

    total = len(to_fetch)
    print(f"\nFetching {total} symbols with {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, item): item for item in to_fetch}
        for future in as_completed(futures):
            sym, n_bars, rows = future.result()
            with lock:
                done += 1
                if n_bars < 0:
                    errors += 1
                elif n_bars == 0:
                    empty += 1
                else:
                    total_bars += n_bars
                    all_rows.extend(rows)

                if done % 100 == 0 or done == total:
                    log.info("[%d/%d] bars=%d empty=%d errors=%d", done, total, total_bars, empty, errors)

    # Batch write all rows from main thread
    if all_rows:
        print(f"\nWriting {len(all_rows):,} bars to database...")
        write_conn = open_db()
        write_conn.executemany(
            "INSERT OR IGNORE INTO option_daily "
            "(contract_symbol, date, open, high, low, close, volume, open_interest) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            all_rows,
        )
        write_conn.commit()
        write_conn.close()
        print(f"  Written successfully.")

    # Final report
    print(f"\n{'='*60}")
    print(f"TLT BACKFILL COMPLETE")
    print(f"  Symbols processed: {done}")
    print(f"  Bars fetched: {total_bars:,}")
    print(f"  Empty (no data): {empty}")
    print(f"  Errors: {errors}")
    print(f"{'='*60}")

    # Verify new coverage
    conn = open_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT MIN(date), MAX(date), COUNT(*)
        FROM option_daily od
        JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
        WHERE oc.ticker = 'TLT' AND od.date >= '2024-07-20'
    """)
    r = cur.fetchone()
    if r[2] > 0:
        print(f"\n  New TLT data: {r[0]} to {r[1]}, {r[2]:,} bars")
    cur.execute("""
        SELECT MIN(date), MAX(date), COUNT(*)
        FROM option_daily od
        JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
        WHERE oc.ticker = 'TLT' AND od.date != '0000-00-00'
    """)
    r2 = cur.fetchone()
    print(f"  Total TLT data: {r2[0]} to {r2[1]}, {r2[2]:,} bars")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TLT option data backfill")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_backfill(workers=args.workers, dry_run=args.dry_run)
