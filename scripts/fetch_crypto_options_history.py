#!/usr/bin/env python3
"""
Fetch and cache IBIT/ETHA options historical data from Polygon.io.

Schema stored in data/crypto_options_cache.db:
  crypto_option_contracts  — contract metadata (strike, expiry, type)
  crypto_option_daily      — daily bars (OHLCV + greeks where available)
  crypto_underlying_daily  — daily OHLCV for the underlying ETF

Notes on greeks/bid-ask:
  Polygon's daily agg bars (/v2/aggs) do NOT include greeks or bid/ask.
  Those fields are populated as NULL for historical data.
  Only the live /v3/snapshot endpoint returns greeks — which is fine for
  live trading but not for backtesting. The backtester uses `close` as
  the mid-price proxy (same approach as the SPY backtester).

Usage:
    python3 scripts/fetch_crypto_options_history.py --ticker IBIT
    python3 scripts/fetch_crypto_options_history.py --ticker IBIT --date-from 2026-03-13
    python3 scripts/fetch_crypto_options_history.py --ticker ETHA --workers 6
    python3 scripts/fetch_crypto_options_history.py --ticker IBIT --dry-run
    python3 scripts/fetch_crypto_options_history.py --ticker IBIT --reset
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import threading
import time
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
    load_dotenv(ROOT / ".env.exp400")
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

DATA_DIR = ROOT / "data"
BASE_URL = "https://api.polygon.io"

# IBIT launched Nov 2024; ETHA launched Jul 2024 but options started ~Apr 2025
TICKER_DATE_FROM = {
    "IBIT": "2024-11-01",
    "ETHA": "2025-04-01",
    "BITO": "2021-10-01",
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

def open_db(db_path: Path) -> sqlite3.Connection:
    """Open SQLite with WAL mode for safe concurrent writes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA cache_size=-32768")
    return conn


def init_db(db_path: Path):
    """Create tables if they don't exist."""
    conn = open_db(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS crypto_option_contracts (
            ticker          TEXT NOT NULL,
            expiration      TEXT NOT NULL,
            strike          REAL NOT NULL,
            option_type     TEXT NOT NULL,
            contract_symbol TEXT NOT NULL,
            as_of_date      TEXT,
            PRIMARY KEY (ticker, expiration, strike, option_type)
        );

        CREATE TABLE IF NOT EXISTS crypto_option_daily (
            contract_symbol  TEXT NOT NULL,
            date             TEXT NOT NULL,
            -- Daily bar (from /v2/aggs)
            open             REAL,
            high             REAL,
            low              REAL,
            close            REAL,
            volume           INTEGER,
            open_interest    INTEGER,
            -- Bid/ask + greeks: NULL for bar data; populated via snapshot for live dates
            bid              REAL,
            ask              REAL,
            mid              REAL,
            iv               REAL,
            delta            REAL,
            gamma            REAL,
            theta            REAL,
            vega             REAL,
            underlying_price REAL,
            PRIMARY KEY (contract_symbol, date)
        );

        CREATE TABLE IF NOT EXISTS crypto_underlying_daily (
            ticker  TEXT NOT NULL,
            date    TEXT NOT NULL,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  INTEGER,
            PRIMARY KEY (ticker, date)
        );

        CREATE INDEX IF NOT EXISTS idx_cod_contract
            ON crypto_option_daily (contract_symbol);
        CREATE INDEX IF NOT EXISTS idx_coc_ticker_expiry
            ON crypto_option_contracts (ticker, expiration);
    """)
    conn.commit()
    conn.close()
    log.info("DB initialized: %s", db_path)


# ---------------------------------------------------------------------------
# Per-thread session + rate limiter
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def get_session(api_key: str) -> requests.Session:
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
    url_or_path: str,
    params: dict,
    api_key: str,
    min_interval: float = 0.06,
) -> Optional[dict]:
    """Single Polygon GET with per-thread rate limiting and retry."""
    session = get_session(api_key)
    wait = min_interval - (time.time() - getattr(_thread_local, "last_call", 0.0))
    if wait > 0:
        time.sleep(wait)

    url = url_or_path if url_or_path.startswith("http") else f"{BASE_URL}{url_or_path}"
    try:
        resp = session.get(url, params=params, timeout=30)
        _thread_local.last_call = time.time()
        if resp.status_code == 429:
            log.warning("Rate limited — sleeping 10s")
            time.sleep(10)
            return api_get(url, {}, api_key, min_interval)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        log.error("API error %s: %s", url, e)
        _thread_local.last_call = time.time()
        return None


# ---------------------------------------------------------------------------
# Phase 0: Underlying daily bars
# ---------------------------------------------------------------------------

def fetch_underlying_bars(ticker: str, date_from: str, date_to: str, api_key: str) -> List[tuple]:
    """Fetch daily OHLCV for the underlying ETF."""
    data = api_get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{date_from}/{date_to}",
        params={"adjusted": "true", "sort": "asc", "limit": 5000},
        api_key=api_key,
        min_interval=0.25,
    )
    if not data:
        return []
    rows = []
    for bar in data.get("results", []):
        ts = bar.get("t", 0)
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append((ticker, dt, bar.get("o"), bar.get("h"), bar.get("l"), bar.get("c"), bar.get("v")))
    return rows


def store_underlying(rows: List[tuple], db_path: Path):
    conn = open_db(db_path)
    conn.executemany(
        "INSERT OR IGNORE INTO crypto_underlying_daily "
        "(ticker, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Phase 1: Contract discovery
# ---------------------------------------------------------------------------

def discover_contracts(
    ticker: str,
    date_from: str,
    date_to: str,
    api_key: str,
) -> List[dict]:
    """
    Page through /v3/reference/options/contracts for all puts + calls.
    Returns list of {ticker, expiration, strike, option_type, contract_symbol}.
    """
    exp_from = date_from  # fetch contracts that expired after our start date
    # Include contracts expiring up to 2.5 years out from date_to
    exp_to_dt = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=900)
    exp_to = exp_to_dt.strftime("%Y-%m-%d")

    all_contracts = []

    for contract_type in ("put", "call"):
        url = f"{BASE_URL}/v3/reference/options/contracts"
        params: dict = {
            "underlying_ticker": ticker,
            "contract_type": contract_type,
            "expiration_date.gte": exp_from,
            "expiration_date.lte": exp_to,
            "expired": "true",   # include expired so we get historical contracts
            "limit": 1000,
        }
        page = 0
        while url:
            data = api_get(url, params, api_key, min_interval=0.25)
            if not data:
                log.warning("No data on page %d for %s %s", page, ticker, contract_type)
                break

            results = data.get("results", [])
            for c in results:
                sym = c.get("ticker", "")
                if sym:
                    all_contracts.append({
                        "ticker": ticker,
                        "expiration": c.get("expiration_date", ""),
                        "strike": c.get("strike_price"),
                        "option_type": contract_type,
                        "contract_symbol": sym,
                    })

            page += 1
            if page % 5 == 0:
                log.info("  %s %s: page %d → %d contracts so far", ticker, contract_type, page, len(all_contracts))

            next_url = data.get("next_url")
            if next_url:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(next_url).query)
                cursor = qs.get("cursor", [None])[0]
                if cursor:
                    url = f"{BASE_URL}/v3/reference/options/contracts"
                    params = {"cursor": cursor, "limit": 1000}
                else:
                    break
            else:
                break

        log.info("  %s %s: discovery complete — %d contracts", ticker, contract_type, len(all_contracts))

    # Also fetch active (non-expired) contracts
    for contract_type in ("put", "call"):
        url = f"{BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": ticker,
            "contract_type": contract_type,
            "expiration_date.gte": date_to,  # still active after our end date
            "limit": 1000,
        }
        while url:
            data = api_get(url, params, api_key, min_interval=0.25)
            if not data:
                break
            results = data.get("results", [])
            for c in results:
                sym = c.get("ticker", "")
                if sym:
                    all_contracts.append({
                        "ticker": ticker,
                        "expiration": c.get("expiration_date", ""),
                        "strike": c.get("strike_price"),
                        "option_type": contract_type,
                        "contract_symbol": sym,
                    })
            next_url = data.get("next_url")
            if next_url:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(next_url).query)
                cursor = qs.get("cursor", [None])[0]
                if cursor:
                    url = f"{BASE_URL}/v3/reference/options/contracts"
                    params = {"cursor": cursor, "limit": 1000}
                else:
                    break
            else:
                break

    # Deduplicate by contract_symbol
    seen = set()
    deduped = []
    for c in all_contracts:
        if c["contract_symbol"] not in seen:
            seen.add(c["contract_symbol"])
            deduped.append(c)
    return deduped


def store_contracts(contracts: List[dict], db_path: Path):
    """Upsert contracts into crypto_option_contracts table."""
    rows = [
        (c["ticker"], c["expiration"], c["strike"], c["option_type"],
         c["contract_symbol"], datetime.now().strftime("%Y-%m-%d"))
        for c in contracts
    ]
    conn = open_db(db_path)
    conn.executemany(
        "INSERT OR IGNORE INTO crypto_option_contracts "
        "(ticker, expiration, strike, option_type, contract_symbol, as_of_date) "
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Phase 2: Daily bar fetch per contract
# ---------------------------------------------------------------------------

def get_cached_symbols(db_path: Path) -> Tuple[set, set]:
    """Return (symbols_with_data, sentinel_symbols)."""
    conn = open_db(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT contract_symbol FROM crypto_option_daily WHERE date != '0000-00-00'")
    has_data = {r[0] for r in cur.fetchall()}
    cur.execute("SELECT DISTINCT contract_symbol FROM crypto_option_daily WHERE date = '0000-00-00'")
    sentinels = {r[0] for r in cur.fetchall()}
    conn.close()
    return has_data, sentinels


def fetch_daily_bars(
    symbol: str, date_from: str, date_to: str, api_key: str
) -> Tuple[str, Optional[List[tuple]]]:
    """
    Fetch daily OHLCV for one option contract.

    Returns:
        (symbol, rows)  — rows is None on error, [] on no data, list on success
    """
    data = api_get(
        f"/v2/aggs/ticker/{symbol}/range/1/day/{date_from}/{date_to}",
        params={"adjusted": "true", "sort": "asc", "limit": 5000},
        api_key=api_key,
    )
    if data is None:
        return symbol, None  # transient error

    results = data.get("results", [])
    if not results:
        return symbol, []  # no data — write sentinel

    rows = []
    for bar in results:
        ts = bar.get("t", 0)
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        close = bar.get("c")
        rows.append((
            symbol, dt,
            bar.get("o"), bar.get("h"), bar.get("l"), close,
            int(bar.get("v", 0) or 0), bar.get("oi"),
            # bid/ask/mid/greeks: NULL for historical bar data
            None, None, close,  # mid = close as best proxy
            None, None, None, None, None,  # iv, delta, gamma, theta, vega
            None,  # underlying_price (joined later from crypto_underlying_daily)
        ))
    return symbol, rows


def write_bars(symbol: str, rows: Optional[List[tuple]], conn: sqlite3.Connection):
    if rows is None:
        return  # error — don't cache anything
    if len(rows) == 0:
        conn.execute(
            "INSERT OR IGNORE INTO crypto_option_daily (contract_symbol, date) VALUES (?, '0000-00-00')",
            (symbol,),
        )
    else:
        conn.executemany(
            "INSERT OR IGNORE INTO crypto_option_daily "
            "(contract_symbol, date, open, high, low, close, volume, open_interest, "
            " bid, ask, mid, iv, delta, gamma, theta, vega, underlying_price) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Phase 3: Backfill underlying_price into option rows
# ---------------------------------------------------------------------------

def backfill_underlying_price(ticker: str, db_path: Path):
    """
    Join crypto_underlying_daily into crypto_option_daily.underlying_price
    for all rows belonging to this ticker's contracts.
    """
    conn = open_db(db_path)
    conn.execute("""
        UPDATE crypto_option_daily
        SET underlying_price = (
            SELECT cud.close
            FROM crypto_underlying_daily cud
            WHERE cud.ticker = ?
              AND cud.date = crypto_option_daily.date
        )
        WHERE contract_symbol IN (
            SELECT contract_symbol FROM crypto_option_contracts WHERE ticker = ?
        )
        AND underlying_price IS NULL
    """, (ticker, ticker))
    updated = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    log.info("Backfilled underlying_price for %d rows", updated)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"discovery_done": False, "fetched": []}


def save_checkpoint(path: Path, cp: dict):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cp, f)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch crypto ETF options history from Polygon")
    parser.add_argument("--ticker", default="IBIT", help="Underlying ticker (IBIT, ETHA, BITO)")
    parser.add_argument("--date-from", default=None, help="Start date YYYY-MM-DD (default: ticker launch date)")
    parser.add_argument("--date-to", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--days", type=int, default=None, help="Shorthand: fetch last N days (overrides --date-from)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel bar-fetch workers")
    parser.add_argument("--dry-run", action="store_true", help="Discovery only, skip bar fetch")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and re-fetch everything")
    parser.add_argument("--skip-discovery", action="store_true", help="Use cached contracts only")
    args = parser.parse_args()

    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key:
        log.error("POLYGON_API_KEY not set — source .env.exp400 first")
        sys.exit(1)

    ticker = args.ticker.upper()
    date_to = args.date_to or datetime.now().strftime("%Y-%m-%d")
    if args.days:
        date_from = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    else:
        date_from = args.date_from or TICKER_DATE_FROM.get(ticker, "2024-01-01")

    DB_PATH = DATA_DIR / "crypto_options_cache.db"
    CHECKPOINT_PATH = DATA_DIR / f"fetch_progress_{ticker}.json"
    SYMBOLS_PATH = DATA_DIR / f"fetch_symbols_{ticker}.json"

    log.info("=== Crypto Options Fetcher ===")
    log.info("Ticker:    %s", ticker)
    log.info("Date from: %s", date_from)
    log.info("Date to:   %s", date_to)
    log.info("DB:        %s", DB_PATH)
    log.info("Workers:   %d", args.workers)

    DATA_DIR.mkdir(exist_ok=True)
    init_db(DB_PATH)

    if args.reset:
        for p in (CHECKPOINT_PATH, SYMBOLS_PATH):
            if p.exists():
                p.unlink()
                log.info("Cleared: %s", p)

    cp = load_checkpoint(CHECKPOINT_PATH)

    # -----------------------------------------------------------------------
    # Phase 0: Underlying daily price
    # -----------------------------------------------------------------------
    log.info("")
    log.info("Phase 0: Fetching underlying %s daily bars %s → %s", ticker, date_from, date_to)
    underlying_rows = fetch_underlying_bars(ticker, date_from, date_to, api_key)
    if underlying_rows:
        store_underlying(underlying_rows, DB_PATH)
        log.info("Phase 0: Stored %d underlying bars", len(underlying_rows))
    else:
        log.warning("Phase 0: No underlying bars returned")

    # -----------------------------------------------------------------------
    # Phase 1: Contract discovery
    # -----------------------------------------------------------------------
    if not args.skip_discovery and not cp.get("discovery_done"):
        log.info("")
        log.info("Phase 1: Discovering %s option contracts...", ticker)
        contracts = discover_contracts(ticker, date_from, date_to, api_key)
        log.info("Phase 1: Found %d unique contracts", len(contracts))
        store_contracts(contracts, DB_PATH)
        with open(SYMBOLS_PATH, "w") as f:
            json.dump([c["contract_symbol"] for c in contracts], f)
        cp["discovery_done"] = True
        save_checkpoint(CHECKPOINT_PATH, cp)
    elif SYMBOLS_PATH.exists():
        log.info("")
        log.info("Phase 1: Loading cached symbols from %s", SYMBOLS_PATH.name)
    else:
        # Pull from DB
        log.info("")
        log.info("Phase 1: Loading contracts from DB...")
        conn = open_db(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT contract_symbol FROM crypto_option_contracts WHERE ticker=?", (ticker,))
        syms = [r[0] for r in cur.fetchall()]
        conn.close()
        with open(SYMBOLS_PATH, "w") as f:
            json.dump(syms, f)
        cp["discovery_done"] = True
        save_checkpoint(CHECKPOINT_PATH, cp)

    if SYMBOLS_PATH.exists():
        with open(SYMBOLS_PATH) as f:
            all_symbols = json.load(f)
    else:
        log.error("No symbols file — re-run without --skip-discovery")
        sys.exit(1)

    log.info("Phase 1: %d total symbols", len(all_symbols))

    if args.dry_run:
        log.info("Dry run — stopping after discovery")
        _print_discovery_summary(ticker, all_symbols, DB_PATH)
        return

    # -----------------------------------------------------------------------
    # Phase 2: Fetch daily bars
    # -----------------------------------------------------------------------
    log.info("")
    log.info("Phase 2: Fetching daily bars for %d contracts...", len(all_symbols))

    already_fetched = set(cp.get("fetched", []))
    has_data, sentinels = get_cached_symbols(DB_PATH)
    skip = has_data | sentinels | already_fetched

    to_fetch = [s for s in all_symbols if s not in skip]
    total = len(to_fetch)
    log.info("Phase 2: to_fetch=%d | cached=%d | sentinel=%d", total, len(has_data), len(sentinels))

    if total == 0:
        log.info("Phase 2: Nothing to fetch — all cached!")
    else:
        _thread_db: Dict[int, sqlite3.Connection] = {}
        _db_lock = threading.Lock()

        def get_thread_db() -> sqlite3.Connection:
            tid = threading.get_ident()
            with _db_lock:
                if tid not in _thread_db:
                    _thread_db[tid] = open_db(DB_PATH)
            return _thread_db[tid]

        def process(symbol: str) -> Tuple[str, str]:
            sym, rows = fetch_daily_bars(symbol, date_from, date_to, api_key)
            conn = get_thread_db()
            write_bars(sym, rows, conn)
            if rows is None:
                return sym, "error"
            elif len(rows) == 0:
                return sym, "sentinel"
            return sym, f"ok:{len(rows)}"

        done = 0
        errors = 0
        sentinels_written = 0
        start_t = time.time()
        progress_lock = threading.Lock()
        batch: List[str] = []
        SAVE_EVERY = 200

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process, sym): sym for sym in to_fetch}
            for future in as_completed(futures):
                sym, status = future.result()
                with progress_lock:
                    done += 1
                    if status == "error":
                        errors += 1
                    elif status == "sentinel":
                        sentinels_written += 1
                        already_fetched.add(sym)
                        batch.append(sym)
                    else:
                        already_fetched.add(sym)
                        batch.append(sym)

                    if len(batch) >= SAVE_EVERY:
                        cp["fetched"] = list(already_fetched)
                        save_checkpoint(CHECKPOINT_PATH, cp)
                        batch.clear()

                    if done % 200 == 0 or done == total:
                        elapsed = time.time() - start_t
                        rate = done / elapsed if elapsed > 0 else 0
                        eta_s = (total - done) / rate if rate > 0 else 0
                        log.info(
                            "[%d/%d] %.1f%% | %.1f req/s | ETA %.0fm | errors=%d sentinel=%d",
                            done, total, 100 * done / total,
                            rate, eta_s / 60, errors, sentinels_written,
                        )

        cp["fetched"] = list(already_fetched)
        save_checkpoint(CHECKPOINT_PATH, cp)

        elapsed_min = (time.time() - start_t) / 60
        log.info(
            "Phase 2 done: %d bars | %d sentinels | %d errors in %.1f min",
            done - errors - sentinels_written, sentinels_written, errors, elapsed_min,
        )

    # -----------------------------------------------------------------------
    # Phase 3: Backfill underlying_price
    # -----------------------------------------------------------------------
    log.info("")
    log.info("Phase 3: Backfilling underlying_price into option rows...")
    backfill_underlying_price(ticker, DB_PATH)

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    log.info("")
    _print_db_summary(ticker, DB_PATH)


def _print_discovery_summary(ticker: str, symbols: List[str], db_path: Path):
    """Print contract discovery stats."""
    conn = open_db(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT option_type, COUNT(*), MIN(expiration), MAX(expiration) "
        "FROM crypto_option_contracts WHERE ticker=? GROUP BY option_type",
        (ticker,)
    )
    rows = cur.fetchall()
    conn.close()
    log.info("=== Discovery Summary for %s ===", ticker)
    for row in rows:
        log.info("  %s: %d contracts | %s → %s", *row)
    log.info("  Total symbols: %d", len(symbols))


def _print_db_summary(ticker: str, db_path: Path):
    """Print final DB stats."""
    conn = open_db(db_path)
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM crypto_option_contracts WHERE ticker=?", (ticker,)
    )
    n_contracts = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT contract_symbol)
        FROM crypto_option_daily
        WHERE contract_symbol IN (
            SELECT contract_symbol FROM crypto_option_contracts WHERE ticker=?
        ) AND date != '0000-00-00'
    """, (ticker,))
    row = cur.fetchone()
    n_bars, min_date, max_date, n_symbols = row if row else (0, None, None, 0)

    cur.execute("""
        SELECT COUNT(*) FROM crypto_option_daily
        WHERE contract_symbol IN (
            SELECT contract_symbol FROM crypto_option_contracts WHERE ticker=?
        ) AND date = '0000-00-00'
    """, (ticker,))
    n_sentinels = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*), MIN(date), MAX(date) FROM crypto_underlying_daily WHERE ticker=?",
        (ticker,)
    )
    u_row = cur.fetchone()
    u_count, u_min, u_max = u_row if u_row else (0, None, None)

    conn.close()

    log.info("=== DB Summary: %s ===", ticker)
    log.info("  Contracts registered:   %d", n_contracts)
    log.info("  Contracts with bars:    %d / %d (%.1f%%)",
             n_symbols, n_contracts,
             100 * n_symbols / n_contracts if n_contracts else 0)
    log.info("  Total daily bar rows:   %d", n_bars)
    log.info("  Sentinel rows:          %d", n_sentinels)
    log.info("  Bar date range:         %s → %s", min_date, max_date)
    log.info("  Underlying bars:        %d (%s → %s)", u_count, u_min, u_max)


if __name__ == "__main__":
    main()
