#!/usr/bin/env python3
"""
Targeted backfill: fetch daily bars for the gap period only.

Finds all contracts in options_cache.db whose expiration is AFTER the last
data date for that ticker, and fetches daily bars from Polygon for the
missing window.  Much faster than a full discovery+backfill.

Usage:
    python3 scripts/backfill_gap.py                    # all tickers
    python3 scripts/backfill_gap.py --ticker SPY       # one ticker
    python3 scripts/backfill_gap.py --dry-run          # report only
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
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

from shared.constants import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "options_cache.db")
BASE_URL = "https://api.polygon.io"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def get_ticker_status(conn: sqlite3.Connection) -> List[Dict]:
    """Get last data date and contract count per ticker."""
    cur = conn.cursor()
    cur.execute("""
        SELECT oc.ticker,
               COUNT(DISTINCT oc.contract_symbol) as n_contracts,
               MAX(od.date) as last_date,
               MIN(od.date) as first_date,
               COUNT(DISTINCT CASE WHEN od.date != '0000-00-00' THEN od.date END) as n_dates,
               COUNT(od.rowid) as total_bars
        FROM option_contracts oc
        JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
        WHERE od.date != '0000-00-00'
        GROUP BY oc.ticker
        ORDER BY oc.ticker
    """)
    results = []
    for row in cur.fetchall():
        results.append({
            "ticker": row[0],
            "n_contracts": row[1],
            "last_date": row[2],
            "first_date": row[3],
            "n_dates": row[4],
            "total_bars": row[5],
        })
    return results


def get_contracts_needing_backfill(
    conn: sqlite3.Connection,
    ticker: str,
    last_date: str,
    target_date: str,
) -> List[str]:
    """Find contracts whose expiration is AFTER last_date (still active)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT oc.contract_symbol
        FROM option_contracts oc
        WHERE oc.ticker = ?
          AND oc.expiration > ?
          AND oc.expiration <= ?
    """, (ticker, last_date, target_date))
    return [r[0] for r in cur.fetchall()]


def get_already_fetched_dates(
    conn: sqlite3.Connection,
    symbol: str,
) -> set:
    """Return dates already in option_daily for this symbol."""
    cur = conn.cursor()
    cur.execute(
        "SELECT date FROM option_daily WHERE contract_symbol = ? AND date != '0000-00-00'",
        (symbol,),
    )
    return {r[0] for r in cur.fetchall()}


# ---------------------------------------------------------------------------
# Polygon API
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def get_session(api_key: str) -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        retry = Retry(
            total=4, backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_jitter=0.3,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.params = {"apiKey": api_key}
        _thread_local.session = s
        _thread_local.last_call = 0.0
    return _thread_local.session


def fetch_bars(
    symbol: str,
    date_from: str,
    date_to: str,
    api_key: str,
) -> Optional[List[Tuple]]:
    """Fetch daily OHLCV for one contract in the given date range."""
    session = get_session(api_key)

    # Rate limit: ~60ms between calls per thread
    wait = 0.06 - (time.time() - _thread_local.last_call)
    if wait > 0:
        time.sleep(wait)

    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{date_from}/{date_to}"
    try:
        resp = session.get(url, params={
            "adjusted": "true", "sort": "asc", "limit": 5000
        }, timeout=30)
        _thread_local.last_call = time.time()

        if resp.status_code == 429:
            log.warning("Rate limited — sleeping 10s")
            time.sleep(10)
            return fetch_bars(symbol, date_from, date_to, api_key)

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


# ---------------------------------------------------------------------------
# Main backfill logic
# ---------------------------------------------------------------------------

def backfill_ticker(
    ticker: str,
    last_date: str,
    target_date: str,
    api_key: str,
    workers: int = 4,
    dry_run: bool = False,
) -> Dict:
    """Backfill one ticker from last_date to target_date."""
    conn = open_db()

    # Find contracts that expired AFTER last_date (still tradeable in the gap)
    # Also include contracts expiring up to 60 days after target_date (for new contracts)
    future_cutoff = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=60)).strftime("%Y-%m-%d")
    contracts = get_contracts_needing_backfill(conn, ticker, last_date, future_cutoff)

    # Also add contracts that already have some data but may be missing the gap dates
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT od.contract_symbol
        FROM option_daily od
        JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
        WHERE oc.ticker = ?
          AND od.date = ?
          AND oc.expiration > ?
    """, (ticker, last_date, last_date))
    active_contracts = [r[0] for r in cur.fetchall()]
    contracts = list(set(contracts + active_contracts))

    conn.close()

    log.info("  %s: %d contracts to backfill (%s → %s)", ticker, len(contracts), last_date, target_date)

    if dry_run or not contracts:
        return {
            "ticker": ticker,
            "contracts_to_fetch": len(contracts),
            "bars_fetched": 0,
            "errors": 0,
            "dry_run": dry_run,
        }

    # Fetch date range: day after last_date to target_date
    fetch_from = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_to = target_date

    done = 0
    total_bars = 0
    errors = 0
    progress_lock = threading.Lock()

    _thread_db: Dict[int, sqlite3.Connection] = {}
    _thread_db_lock = threading.Lock()

    def get_thread_db() -> sqlite3.Connection:
        tid = threading.get_ident()
        with _thread_db_lock:
            if tid not in _thread_db:
                _thread_db[tid] = open_db()
        return _thread_db[tid]

    def process(symbol: str) -> Tuple[str, int]:
        rows = fetch_bars(symbol, fetch_from, fetch_to, api_key)
        if rows is None:
            return symbol, -1  # error
        if rows:
            db = get_thread_db()
            db.executemany(
                "INSERT OR IGNORE INTO option_daily "
                "(contract_symbol, date, open, high, low, close, volume, open_interest) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            db.commit()
        return symbol, len(rows)

    total = len(contracts)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, sym): sym for sym in contracts}
        for future in as_completed(futures):
            sym, n_bars = future.result()
            with progress_lock:
                done += 1
                if n_bars < 0:
                    errors += 1
                else:
                    total_bars += n_bars

                if done % 50 == 0 or done == total:
                    log.info("    %s: [%d/%d] bars=%d errors=%d", ticker, done, total, total_bars, errors)

    return {
        "ticker": ticker,
        "contracts_to_fetch": len(contracts),
        "bars_fetched": total_bars,
        "errors": errors,
        "dry_run": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Targeted gap backfill")
    parser.add_argument("--ticker", type=str, default=None, help="Backfill one ticker only")
    parser.add_argument("--target-date", type=str, default="2026-04-03", help="Last trading day to fetch")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no fetching")
    args = parser.parse_args()

    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        log.error("POLYGON_API_KEY not set")
        sys.exit(1)

    conn = open_db()
    status = get_ticker_status(conn)
    conn.close()

    print("=" * 70)
    print("TARGETED GAP BACKFILL")
    print(f"Target date: {args.target_date}")
    print("=" * 70)
    print()

    print(f"{'Ticker':<8} {'Last Date':>12} {'Gap Days':>10} {'Contracts':>10}")
    print("-" * 45)

    results = []
    for s in status:
        gap_days = (datetime.strptime(args.target_date, "%Y-%m-%d") -
                    datetime.strptime(s["last_date"], "%Y-%m-%d")).days
        print(f"{s['ticker']:<8} {s['last_date']:>12} {gap_days:>10} {s['n_contracts']:>10,}")

        if args.ticker and s["ticker"] != args.ticker:
            continue

        if gap_days <= 0:
            log.info("  %s: already up to date", s["ticker"])
            continue

        result = backfill_ticker(
            s["ticker"], s["last_date"], args.target_date,
            api_key, workers=args.workers, dry_run=args.dry_run,
        )
        results.append(result)

    print()
    if results:
        print("BACKFILL RESULTS:")
        print(f"{'Ticker':<8} {'Contracts':>10} {'Bars Fetched':>14} {'Errors':>8}")
        print("-" * 45)
        for r in results:
            print(f"{r['ticker']:<8} {r['contracts_to_fetch']:>10} {r['bars_fetched']:>14,} {r['errors']:>8}")

    return results


if __name__ == "__main__":
    main()
