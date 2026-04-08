#!/usr/bin/env python3
"""
Backfill ALL SPY option daily bars from Polygon into data/options_cache.db.

Phase 1: Discover all SPY option contract symbols (puts + calls, DTE 7-60 range)
         via Polygon /v3/reference/options/contracts (paginated).
         Also pulls from existing option_contracts table.

Phase 2: Fetch daily OHLCV bars for every contract missing from option_daily,
         using /v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}.

Usage:
    python3 scripts/backfill_polygon_cache.py
    python3 scripts/backfill_polygon_cache.py --workers 12
    python3 scripts/backfill_polygon_cache.py --skip-discovery   # skip Phase 1
    python3 scripts/backfill_polygon_cache.py --dry-run          # discovery only
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import threading
import urllib.parse
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
    pass  # dotenv optional; rely on env already set

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "options_cache.db"
CHECKPOINT_PATH = DATA_DIR / "backfill_progress_SPY_daily.json"
SYMBOLS_PATH = DATA_DIR / "backfill_symbols_SPY.json"

BASE_URL = "https://api.polygon.io"
TICKER = "SPY"
DATE_FROM = "2020-01-01"
DATE_TO = datetime.now(timezone.utc).strftime("%Y-%m-%d")
DTE_MIN = 7
DTE_MAX = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def open_db() -> sqlite3.Connection:
    """Open SQLite with WAL mode for safe concurrent writes."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA cache_size=-32768")  # 32 MB page cache
    return conn


def get_db_symbols_and_cached(all_symbols: List[str]) -> Tuple[List[str], int, int]:
    """Return (missing_symbols, n_cached, n_sentinel) by checking option_daily."""
    conn = open_db()
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT contract_symbol FROM option_daily WHERE date != '0000-00-00'")
    has_data = {row[0] for row in cur.fetchall()}

    cur.execute("SELECT DISTINCT contract_symbol FROM option_daily WHERE date = '0000-00-00'")
    sentinel = {row[0] for row in cur.fetchall()}

    conn.close()

    skip = has_data | sentinel
    missing = [s for s in all_symbols if s not in skip]
    return missing, len(has_data), len(sentinel)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    cp = {"discovery_done": False, "fetched_symbols": []}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            cp.update(json.load(f))
    return cp


def save_symbols(symbols: list):
    """Write all_symbols to a separate file (written once, never changes)."""
    tmp = str(SYMBOLS_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(symbols, f)
    os.replace(tmp, str(SYMBOLS_PATH))


def load_symbols() -> list:
    if SYMBOLS_PATH.exists():
        with open(SYMBOLS_PATH) as f:
            return json.load(f)
    return []


def save_checkpoint(cp: dict):
    """Save only fetched_symbols — keeps file small for fast atomic writes."""
    # Never write all_symbols here; that lives in SYMBOLS_PATH
    small = {"discovery_done": cp.get("discovery_done", False),
             "fetched_symbols": cp.get("fetched_symbols", [])}
    tmp = str(CHECKPOINT_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(small, f)
    os.replace(tmp, str(CHECKPOINT_PATH))


# ---------------------------------------------------------------------------
# Polygon API session (one per thread)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def get_session(api_key: str) -> requests.Session:
    """Return a per-thread requests.Session with retry logic."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_jitter=0.3,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.params = {"apiKey": api_key}  # type: ignore[assignment]
        _thread_local.session = s
        _thread_local.last_call = 0.0
    return _thread_local.session


def api_get(
    path: str,
    params: dict,
    api_key: str,
    min_interval: float = 0.06,
) -> Optional[dict]:
    """Single Polygon GET with per-thread rate limiting."""
    session = get_session(api_key)

    wait = min_interval - (time.time() - _thread_local.last_call)
    if wait > 0:
        time.sleep(wait)

    url = f"{BASE_URL}{path}"
    try:
        resp = session.get(url, params=params, timeout=30)
        _thread_local.last_call = time.time()
        if resp.status_code == 429:
            log.warning("Rate limited — sleeping 10s")
            time.sleep(10)
            return api_get(path, params, api_key, min_interval)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        log.error("API error %s: %s", path, e)
        _thread_local.last_call = time.time()
        return None


# ---------------------------------------------------------------------------
# Phase 1: Contract discovery
# ---------------------------------------------------------------------------

def discover_contracts_from_polygon(api_key: str) -> List[str]:
    """Page through /v3/reference/options/contracts for all SPY puts+calls."""
    date_from = datetime.strptime(DATE_FROM, "%Y-%m-%d")
    date_to = datetime.strptime(DATE_TO, "%Y-%m-%d")
    exp_from = (date_from + timedelta(days=DTE_MIN)).strftime("%Y-%m-%d")
    exp_to = (date_to + timedelta(days=DTE_MAX)).strftime("%Y-%m-%d")

    symbols: List[str] = []

    for contract_type in ("put", "call"):
        cursor = None
        page = 0
        while True:
            params: dict = {
                "underlying_ticker": TICKER,
                "contract_type": contract_type,
                "expiration_date.gte": exp_from,
                "expiration_date.lte": exp_to,
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor

            data = api_get("/v3/reference/options/contracts", params, api_key)
            if not data:
                log.warning("No data on page %d for %s — stopping", page, contract_type)
                break

            results = data.get("results", [])
            for c in results:
                sym = c.get("ticker", "")
                if sym:
                    symbols.append(sym)

            page += 1
            if page % 10 == 0:
                log.info("  %s page %d — %d symbols so far", contract_type, page, len(symbols))

            next_url = data.get("next_url")
            if not next_url:
                break
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(next_url).query)
            cursor = qs.get("cursor", [None])[0]
            if not cursor:
                break

        log.info("  %s: done — %d symbols collected", contract_type, len(symbols))

    return list(set(symbols))


def load_symbols_from_db() -> List[str]:
    conn = open_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT contract_symbol FROM option_contracts WHERE ticker = ?", (TICKER,)
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Phase 2: Daily bar fetch
# ---------------------------------------------------------------------------

def fetch_bars_for_symbol(
    symbol: str, api_key: str
) -> Tuple[str, Optional[List[tuple]]]:
    """
    Fetch daily OHLCV bars for one symbol.

    Returns:
        (symbol, rows)  where rows is:
          - List of tuples ready for INSERT — has data
          - []            — Polygon returned no results (insert sentinel)
          - None          — transient error; skip this round
    """
    data = api_get(
        f"/v2/aggs/ticker/{symbol}/range/1/day/{DATE_FROM}/{DATE_TO}",
        params={"adjusted": "true", "sort": "asc", "limit": 5000},
        api_key=api_key,
    )
    if data is None:
        return symbol, None  # timeout/network error

    results = data.get("results", [])
    if not results:
        return symbol, []  # no data from Polygon

    rows = []
    for bar in results:
        ts = bar.get("t", 0)
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append((
            symbol, dt,
            bar.get("o"), bar.get("h"), bar.get("l"), bar.get("c"),
            bar.get("v", 0), bar.get("oi"),
        ))
    return symbol, rows


def write_results(symbol: str, rows: Optional[List[tuple]], conn: sqlite3.Connection):
    if rows is None:
        return  # error — don't write anything
    if len(rows) == 0:
        conn.execute(
            "INSERT OR IGNORE INTO option_daily (contract_symbol, date, close) VALUES (?, ?, ?)",
            (symbol, "0000-00-00", None),
        )
    else:
        conn.executemany(
            "INSERT OR IGNORE INTO option_daily "
            "(contract_symbol, date, open, high, low, close, volume, open_interest) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backfill SPY option daily bars from Polygon")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default 8)")
    parser.add_argument("--dry-run", action="store_true", help="Discovery only, no bar fetching")
    parser.add_argument("--skip-discovery", action="store_true", help="Skip Phase 1, use DB only")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and start fresh")
    args = parser.parse_args()

    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key:
        log.error("POLYGON_API_KEY not set — source .env first")
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)

    if args.reset:
        for p in (CHECKPOINT_PATH, SYMBOLS_PATH):
            if p.exists():
                p.unlink()
        log.info("Checkpoint cleared")

    cp = load_checkpoint()

    # -----------------------------------------------------------------------
    # Phase 1: discover all contract symbols
    # -----------------------------------------------------------------------
    if cp["discovery_done"] and SYMBOLS_PATH.exists():
        all_symbols = load_symbols()
        log.info("Phase 1: loaded %d symbols from checkpoint", len(all_symbols))
    else:
        db_symbols = load_symbols_from_db()
        log.info("Phase 1: %d symbols in option_contracts table", len(db_symbols))

        if args.skip_discovery:
            api_symbols: List[str] = []
        else:
            log.info("Phase 1: fetching from Polygon reference API...")
            api_symbols = discover_contracts_from_polygon(api_key)
            log.info("Phase 1: Polygon returned %d symbols", len(api_symbols))

        all_symbols = list(set(db_symbols + api_symbols))
        log.info("Phase 1: %d unique symbols total", len(all_symbols))

        save_symbols(all_symbols)  # written once to SYMBOLS_PATH
        cp["discovery_done"] = True
        save_checkpoint(cp)

    if args.dry_run:
        log.info("Dry run — stopping after discovery")
        return

    # -----------------------------------------------------------------------
    # Phase 2: fetch missing daily bars
    # -----------------------------------------------------------------------
    already_fetched = set(cp.get("fetched_symbols", []))

    missing, n_cached, n_sentinel = get_db_symbols_and_cached(all_symbols)
    log.info(
        "Phase 2: total=%d | cached=%d | sentinel=%d | missing=%d",
        len(all_symbols), n_cached, n_sentinel, len(missing),
    )

    # Filter out symbols we already handled this session
    to_fetch = [s for s in missing if s not in already_fetched]
    total = len(to_fetch)
    if total == 0:
        log.info("Nothing to fetch — all done!")
        return

    log.info("Fetching %d contracts with %d workers (60ms/req per worker)...", total, args.workers)

    done = 0
    errors = 0
    skipped = 0
    start_time = time.time()
    progress_lock = threading.Lock()

    # Each worker writes to its own DB connection to avoid contention
    # WAL mode allows multiple concurrent writers safely
    _thread_db: Dict[int, sqlite3.Connection] = {}
    _thread_db_lock = threading.Lock()

    def get_thread_db() -> sqlite3.Connection:
        tid = threading.get_ident()
        with _thread_db_lock:
            if tid not in _thread_db:
                _thread_db[tid] = open_db()
        return _thread_db[tid]

    def process(symbol: str) -> Tuple[str, str]:
        """Worker: fetch bars and write to DB. Returns (symbol, status)."""
        sym, rows = fetch_bars_for_symbol(symbol, api_key)
        conn = get_thread_db()
        write_results(sym, rows, conn)
        if rows is None:
            return sym, "error"
        elif len(rows) == 0:
            return sym, "sentinel"
        else:
            return sym, f"ok:{len(rows)}"

    batch: List[str] = []
    SAVE_EVERY = 100

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, sym): sym for sym in to_fetch}

        for future in as_completed(futures):
            sym, status = future.result()

            with progress_lock:
                done += 1
                if status == "error":
                    errors += 1
                    skipped += 1
                else:
                    already_fetched.add(sym)
                    batch.append(sym)

                # Save checkpoint every SAVE_EVERY completions
                if len(batch) >= SAVE_EVERY:
                    cp["fetched_symbols"] = list(already_fetched)
                    save_checkpoint(cp)
                    batch.clear()

                # Progress log every 100
                if done % 100 == 0 or done == total:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_s = (total - done) / rate if rate > 0 else 0
                    log.info(
                        "[%d/%d] %.1f%% | %.1f req/s | ETA %.0fm | errors %d",
                        done, total, 100 * done / total,
                        rate, eta_s / 60, errors,
                    )

    # Final checkpoint save
    cp["fetched_symbols"] = list(already_fetched)
    save_checkpoint(cp)

    # Thread DB connections are closed by GC when worker threads exit
    # (cannot close from main thread — SQLite connections are thread-local)

    elapsed_min = (time.time() - start_time) / 60
    log.info(
        "Done! fetched=%d sentinel=%d errors=%d in %.1f min",
        done - errors, total - (done - errors) - errors, errors, elapsed_min,
    )


if __name__ == "__main__":
    main()
