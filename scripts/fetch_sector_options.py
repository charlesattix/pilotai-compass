#!/usr/bin/env python3
"""
fetch_sector_options.py — Background ETF options data fetcher

Fetches historical options contracts + daily OHLCV from Polygon for a
given underlying, storing results in the shared options_cache.db.

Strategy:
  1. Enumerate all expirations in the date range via paginated contract listing
  2. For each expiration, filter strikes to the relevant OTM range for credit spreads:
       Puts:  80% to 102% of underlying spot (3–20% OTM range + $5-wide spread buffer)
       Calls: 98% to 120% of underlying spot (matching bear call range)
  3. For each relevant contract, fetch daily OHLCV bars (2020-01-01 to today)
  4. Checkpoint every 200 contracts — resumes seamlessly if interrupted

Estimated throughput: ~900 contracts/hour (1 call/sec × 3 calls/contract avg)

Usage:
    # Start fresh fetch
    python3 scripts/fetch_sector_options.py --ticker QQQ

    # Resume interrupted fetch
    python3 scripts/fetch_sector_options.py --ticker QQQ --resume

    # Custom date range
    python3 scripts/fetch_sector_options.py --ticker XLE --start 2020-01-01 --end 2026-03-07

    # Just count contracts (dry-run discovery, no daily data fetch)
    python3 scripts/fetch_sector_options.py --ticker XLE --discover-only

    # Fetch multiple tickers sequentially (run in terminal, leave overnight)
    python3 scripts/fetch_sector_options.py --ticker QQQ && \\
    python3 scripts/fetch_sector_options.py --ticker XLE && \\
    python3 scripts/fetch_sector_options.py --ticker SOXX

Tier 2 fetch order (highest alpha first):
    QQQ → XLE → SOXX → XLK → XLF → XLI
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import requests

from shared.constants import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_sector")

API_KEY = os.environ.get("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"

# Date range for backtest coverage
DEFAULT_START = "2020-01-01"
DEFAULT_END   = "2026-03-07"

# Strike filter: fetch puts between SPOT * PUT_OTM_LO and SPOT * PUT_OTM_HI
# and calls between SPOT * CALL_OTM_LO and SPOT * CALL_OTM_HI
# The 15% OTM range covers: entry strikes (~3% OTM), spread width ($5), and buffer.
PUT_OTM_LO  = 0.80   # 80% of spot = 20% OTM (far OTM buffer)
PUT_OTM_HI  = 1.02   # 102% of spot = ~2% ITM (slight ITM captures spread body)
CALL_OTM_LO = 0.98   # 98% of spot
CALL_OTM_HI = 1.20   # 120% of spot = 20% OTM

CHECKPOINT_INTERVAL = 200  # checkpoint every N daily-data fetches

# ─────────────────────────────────────────────────────────────────────────────
# Polygon API helpers
# ─────────────────────────────────────────────────────────────────────────────

_last_call = 0.0


def api_get(path: str, params: Optional[Dict] = None, timeout: int = 25) -> Optional[Dict]:
    """Rate-limited Polygon GET (minimum 1s between calls). Returns None on error."""
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    p = dict(params or {})
    p["apiKey"] = API_KEY

    try:
        resp = requests.get(f"{BASE_URL}{path}", params=p, timeout=timeout)
        _last_call = time.time()
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            logger.warning("Rate limited by Polygon, sleeping 10s")
            time.sleep(10.0)
            return api_get(path, params, timeout)  # retry once
        logger.warning("HTTP %d for %s", resp.status_code, path)
        return None
    except requests.exceptions.Timeout:
        logger.warning("Timeout: %s", path)
        _last_call = time.time()
        return None
    except Exception as e:
        logger.warning("Request error for %s: %s", path, e)
        _last_call = time.time()
        return None


def api_get_url(url: str, timeout: int = 25) -> Optional[Dict]:
    """Fetch a pre-formed Polygon next_url (add only the API key)."""
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    try:
        resp = requests.get(url, params={"apiKey": API_KEY}, timeout=timeout)
        _last_call = time.time()
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        logger.warning("Request error for next_url: %s", e)
        _last_call = time.time()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Options cache DB helpers (same DB as HistoricalOptionsData)
# ─────────────────────────────────────────────────────────────────────────────

def get_db_conn(cache_dir: str = DATA_DIR) -> sqlite3.Connection:
    db_path = os.path.join(cache_dir, "options_cache.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # Ensure tables exist (HistoricalOptionsData may not have been run yet for this ticker)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS option_daily (
            contract_symbol TEXT NOT NULL,
            date            TEXT NOT NULL,
            open            REAL,
            high            REAL,
            low             REAL,
            close           REAL,
            volume          INTEGER,
            PRIMARY KEY (contract_symbol, date)
        );
        CREATE TABLE IF NOT EXISTS option_contracts (
            ticker          TEXT NOT NULL,
            expiration      TEXT NOT NULL,
            strike          REAL NOT NULL,
            option_type     TEXT NOT NULL,
            contract_symbol TEXT NOT NULL,
            as_of_date      TEXT NOT NULL,
            PRIMARY KEY (ticker, expiration, strike, option_type)
        );
        CREATE TABLE IF NOT EXISTS option_intraday (
            contract_symbol TEXT NOT NULL,
            date            TEXT NOT NULL,
            bar_time        TEXT NOT NULL,
            open            REAL,
            high            REAL,
            low             REAL,
            close           REAL,
            volume          INTEGER,
            PRIMARY KEY (contract_symbol, date, bar_time)
        );
    """)
    try:
        conn.execute("ALTER TABLE option_daily ADD COLUMN open_interest INTEGER")
    except sqlite3.OperationalError:
        pass  # already exists
    conn.commit()
    return conn


def contract_already_cached(conn: sqlite3.Connection, symbol: str) -> bool:
    """Return True if we already have daily bars for this contract."""
    cur = conn.execute(
        "SELECT 1 FROM option_daily WHERE contract_symbol = ? LIMIT 1",
        (symbol,)
    )
    return cur.fetchone() is not None


def cache_contract_ref(conn: sqlite3.Connection, ticker: str, expiration: str,
                        strike: float, option_type: str, symbol: str, as_of_date: str) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO option_contracts
           (ticker, expiration, strike, option_type, contract_symbol, as_of_date)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (ticker, expiration, strike, option_type, symbol, as_of_date)
    )


def cache_daily_bars(conn: sqlite3.Connection, symbol: str, results: List[Dict]) -> int:
    """Insert daily bars. Returns number of rows inserted."""
    rows = []
    for bar in results:
        ts = bar.get("t", 0)
        dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        rows.append((
            symbol, dt,
            bar.get("o"), bar.get("h"), bar.get("l"), bar.get("c"),
            bar.get("v", 0), bar.get("oi"),
        ))
    if rows:
        conn.executemany(
            """INSERT OR IGNORE INTO option_daily
               (contract_symbol, date, open, high, low, close, volume, open_interest)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint (progress persistence)
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_path(ticker: str) -> Path:
    return ROOT / "data" / f"fetch_progress_{ticker}.json"


def load_checkpoint(ticker: str) -> Dict:
    p = checkpoint_path(ticker)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "ticker": ticker,
        "fetched_symbols": [],   # list of OCC symbols already fetched (daily data done)
        "contracts_cached": [],  # list of (expiration, strike, option_type, symbol) already in DB
        "total_daily_fetched": 0,
        "total_daily_rows": 0,
        "discovery_done": False,
        "started_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
    }


def save_checkpoint(ticker: str, cp: Dict) -> None:
    cp["last_updated"] = datetime.now().isoformat()
    p = checkpoint_path(ticker)
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        json.dump(cp, f, indent=2)
    logger.debug("Checkpoint saved: %d fetched, %d rows", cp["total_daily_fetched"], cp["total_daily_rows"])


# ─────────────────────────────────────────────────────────────────────────────
# Underlying price history (for strike filtering)
# ─────────────────────────────────────────────────────────────────────────────

def get_underlying_prices(ticker: str, start: str, end: str) -> Dict[str, float]:
    """
    Fetch daily closing prices for the underlying from Yahoo Finance via curl.
    Returns dict of {date_str: close_price}.
    Falls back to empty dict (disables strike filtering) on failure.
    """
    import subprocess
    from urllib.parse import quote as url_quote

    try:
        p1 = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        p2 = int(datetime.strptime(end, "%Y-%m-%d").timestamp())
    except Exception:
        return {}

    cookie_file = ROOT / "data" / "yf_cookies.txt"
    cookie_file.parent.mkdir(exist_ok=True)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{url_quote(ticker)}"
        f"?period1={p1}&period2={p2}&interval=1d&includeAdjustedClose=true"
    )
    cmd = [
        "curl", "-s", "--max-time", "30",
        "-c", str(cookie_file), "-b", str(cookie_file),
        "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "-H", "Accept: application/json",
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
        if proc.returncode != 0 or not proc.stdout.strip():
            logger.warning("curl failed for %s price history", ticker)
            return {}
        chart = json.loads(proc.stdout)
        results = chart.get("chart", {}).get("result", [])
        if not results:
            return {}
        r = results[0]
        timestamps = r.get("timestamp", [])
        adj = r.get("indicators", {}).get("adjclose", [{}])
        closes = adj[0].get("adjclose", []) if adj else []
        if not closes:
            closes = r.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        prices = {}
        for ts, close in zip(timestamps, closes):
            if close is not None:
                dt = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                prices[dt] = float(close)
        logger.info("Loaded %d price bars for %s from Yahoo", len(prices), ticker)
        return prices
    except Exception as e:
        logger.warning("Failed to load price history for %s: %s", ticker, e)
        return {}


def get_spot_on_or_before(prices: Dict[str, float], exp_date: str) -> Optional[float]:
    """Return the most recent price at or before exp_date."""
    target = exp_date
    while target >= "2020-01-01":
        if target in prices:
            return prices[target]
        # Step back 1 day (simple string decrement for trading days)
        dt = datetime.strptime(target, "%Y-%m-%d")
        target = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Contract discovery (by Friday enumeration)
# ─────────────────────────────────────────────────────────────────────────────

def get_fridays(start: str, end: str) -> List[str]:
    """Return all Fridays between start and end (inclusive)."""
    dates = []
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    # Advance to the first Friday (weekday 4)
    days_until_friday = (4 - current.weekday()) % 7
    current += timedelta(days=days_until_friday)
    while current <= end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=7)
    return dates


def discover_contracts(ticker: str, start: str, end: str,
                       prices: Dict[str, float],
                       already_cached: set) -> List[Dict]:
    """
    Discover all relevant contracts by querying Polygon for each Friday.

    The Polygon /v3/reference/options/contracts endpoint doesn't support
    date-range filtering for historical data — it only returns current
    contracts unless you specify a specific expiration_date with as_of.
    Solution: enumerate every Friday in the date range and query each one.

    For tickers with weeklies: ~320 Fridays × 2 calls (puts + calls) = ~640 API calls.
    For monthly-only tickers: most Fridays return 0 results quickly.
    """
    fridays = get_fridays(start, end)
    logger.info(
        "Discovering contracts for %s: checking %d Fridays (%s → %s)...",
        ticker, len(fridays), start, end
    )

    all_contracts = []
    seen_symbols = set()
    n_expirations_found = 0
    n_filtered_out = 0

    for i, fri in enumerate(fridays):
        spot = get_spot_on_or_before(prices, fri)

        # Query puts and calls for this expiration date
        for ct, otype_char in [("put", "P"), ("call", "C")]:
            data = api_get("/v3/reference/options/contracts", {
                "underlying_ticker": ticker,
                "expiration_date": fri,
                "contract_type": ct,
                "as_of": fri,
                "limit": 1000,
                "order": "asc",
                "sort": "strike_price",
            })
            if data is None:
                continue

            contracts = data.get("results", [])
            if not contracts:
                continue

            if otype_char == "P":
                n_expirations_found += 1  # count unique expirations by put query

            for c in contracts:
                strike = c.get("strike_price", 0)
                symbol = c.get("ticker", "")
                if not strike or not symbol or symbol in seen_symbols:
                    continue

                # Apply OTM filter if we have price data for this date
                if spot:
                    if otype_char == "P" and not (PUT_OTM_LO * spot <= strike <= PUT_OTM_HI * spot):
                        n_filtered_out += 1
                        continue
                    if otype_char == "C" and not (CALL_OTM_LO * spot <= strike <= CALL_OTM_HI * spot):
                        n_filtered_out += 1
                        continue

                seen_symbols.add(symbol)
                all_contracts.append({
                    "expiration": fri,
                    "strike": strike,
                    "option_type": otype_char,
                    "symbol": symbol,
                })

        # Progress logging every 50 Fridays
        if (i + 1) % 50 == 0:
            logger.info(
                "  Discovery: %d/%d Fridays checked | %d expirations found | %d contracts",
                i + 1, len(fridays), n_expirations_found, len(all_contracts)
            )

    logger.info(
        "Discovery complete: %d Fridays, %d expirations, %d relevant contracts, %d filtered out",
        len(fridays), n_expirations_found, len(all_contracts), n_filtered_out
    )
    return all_contracts


# ─────────────────────────────────────────────────────────────────────────────
# Daily bar fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_daily_bars(symbol: str, fetch_start: str, fetch_end: str) -> Optional[List[Dict]]:
    """Fetch all daily OHLCV bars for one contract."""
    data = api_get(
        f"/v2/aggs/ticker/{symbol}/range/1/day/{fetch_start}/{fetch_end}",
        {"adjusted": "true", "sort": "asc", "limit": 5000},
    )
    if data is None:
        return None  # timeout — don't store sentinel
    return data.get("results", [])  # empty list = no data (store sentinel)


# ─────────────────────────────────────────────────────────────────────────────
# Main fetch loop
# ─────────────────────────────────────────────────────────────────────────────

def run_fetch(ticker: str, start: str, end: str, resume: bool = False,
              discover_only: bool = False):
    if not API_KEY:
        print("ERROR: POLYGON_API_KEY not set. Check .env file.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Fetching options data: {ticker}  [{start} → {end}]")
    if resume:
        print("Mode: RESUME from checkpoint")
    print(f"{'='*70}\n")

    # Load or create checkpoint
    cp = load_checkpoint(ticker)
    if not resume:
        cp = {
            "ticker": ticker,
            "fetched_symbols": [],
            "contracts_cached": [],
            "total_daily_fetched": 0,
            "total_daily_rows": 0,
            "discovery_done": False,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    fetched_set = set(cp.get("fetched_symbols", []))

    # ── Open DB connection ────────────────────────────────────────────────────
    conn = get_db_conn()

    # ── Fetch underlying price history (for OTM filter) ───────────────────────
    logger.info("Loading %s price history from Yahoo Finance...", ticker)
    prices = get_underlying_prices(ticker, start, end)
    if not prices:
        logger.warning("No price data for %s — OTM filter disabled, will fetch ALL strikes", ticker)

    # ── Contract discovery ────────────────────────────────────────────────────
    if not cp.get("discovery_done"):
        contracts = discover_contracts(ticker, start, end, prices, fetched_set)
        cp["all_contracts"] = [
            {"exp": c["expiration"], "strike": c["strike"],
             "otype": c["option_type"], "sym": c["symbol"]}
            for c in contracts
        ]
        cp["discovery_done"] = True
        save_checkpoint(ticker, cp)
        logger.info("Checkpoint saved after discovery: %d contracts", len(contracts))
    else:
        contracts = [
            {"expiration": c["exp"], "strike": c["strike"],
             "option_type": c["otype"], "symbol": c["sym"]}
            for c in cp.get("all_contracts", [])
        ]
        logger.info("Resuming from checkpoint: %d contracts, %d already fetched",
                    len(contracts), len(fetched_set))

    if discover_only:
        print(f"\nDiscovery-only mode: found {len(contracts)} relevant contracts for {ticker}")
        # Print breakdown by year
        by_year: Dict[str, int] = {}
        for c in contracts:
            year = c["expiration"][:4] if "expiration" in c else c.get("exp", "")[:4]
            by_year[year] = by_year.get(year, 0) + 1
        for year in sorted(by_year):
            print(f"  {year}: {by_year[year]} contracts")
        conn.close()
        return

    # ── Daily data fetch ──────────────────────────────────────────────────────
    total = len(contracts)
    remaining = [c for c in contracts if c.get("symbol", c.get("sym", "")) not in fetched_set]
    logger.info("Starting daily data fetch: %d remaining of %d total", len(remaining), total)

    t_start = time.time()
    n_fetched = 0
    n_rows = 0
    n_already_had = total - len(remaining)

    for i, c in enumerate(remaining):
        symbol = c.get("symbol", c.get("sym", ""))
        expiration = c.get("expiration", c.get("exp", ""))

        if not symbol:
            continue

        # Skip if already in DB (e.g., fetched in a previous run without checkpoint)
        if contract_already_cached(conn, symbol):
            fetched_set.add(symbol)
            n_already_had += 1
            continue

        # Cache the contract reference
        cache_contract_ref(
            conn, ticker, expiration,
            c.get("strike", 0.0),
            c.get("option_type", c.get("otype", "P")),
            symbol, expiration
        )

        # Fetch daily bars (full 2020–2026 range)
        bars = fetch_daily_bars(symbol, start, end)

        if bars is None:
            logger.warning("Timeout fetching %s — will retry on next resume", symbol)
            continue  # don't add to fetched_set — will retry

        if not bars:
            # No data — store sentinel so we don't re-fetch
            conn.execute(
                "INSERT OR IGNORE INTO option_daily (contract_symbol, date, close) VALUES (?, ?, ?)",
                (symbol, "0000-00-00", None)
            )

        row_count = cache_daily_bars(conn, symbol, bars)
        n_rows += row_count
        n_fetched += 1
        fetched_set.add(symbol)

        # Checkpoint every N contracts
        if n_fetched % CHECKPOINT_INTERVAL == 0:
            conn.commit()
            cp["fetched_symbols"] = list(fetched_set)
            cp["total_daily_fetched"] = len(fetched_set)
            cp["total_daily_rows"] = cp.get("total_daily_rows", 0) + n_rows
            save_checkpoint(ticker, cp)
            n_rows = 0  # reset local counter

            elapsed = time.time() - t_start
            rate = n_fetched / elapsed * 3600  # contracts/hour
            remaining_count = total - len(fetched_set)
            eta_hours = remaining_count / max(rate, 1)
            print(
                f"  [{len(fetched_set)}/{total}] "
                f"{n_fetched} fetched this session | "
                f"{rate:.0f}/hr | "
                f"ETA: {eta_hours:.1f}h",
                flush=True
            )

        if (i + 1) % 50 == 0:
            # Log progress (without full checkpoint save)
            elapsed = time.time() - t_start
            pct = 100.0 * len(fetched_set) / total
            logger.info(
                "Progress: %d/%d (%.1f%%) | %.0f/hr | session: %d fetched, %d skipped",
                len(fetched_set), total, pct,
                n_fetched / max(elapsed, 1) * 3600,
                n_fetched, n_already_had,
            )

    # Final commit + checkpoint
    conn.commit()
    cp["fetched_symbols"] = list(fetched_set)
    cp["total_daily_fetched"] = len(fetched_set)
    cp["total_daily_rows"] = cp.get("total_daily_rows", 0) + n_rows
    save_checkpoint(ticker, cp)
    conn.close()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"FETCH COMPLETE: {ticker}")
    print(f"  Contracts fetched this session: {n_fetched}")
    print(f"  Total contracts in cache: {len(fetched_set)}/{total}")
    print(f"  Session time: {elapsed/3600:.1f}h")
    print(f"  Checkpoint: {checkpoint_path(ticker)}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch sector ETF options data from Polygon")
    parser.add_argument("--ticker", required=True, help="Underlying ticker (e.g. QQQ, XLE, SOXX)")
    parser.add_argument("--start",  default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=DEFAULT_END,   help="End date YYYY-MM-DD")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--discover-only", action="store_true",
                        help="Only discover contracts, don't fetch daily bars")
    args = parser.parse_args()

    run_fetch(
        ticker=args.ticker.upper(),
        start=args.start,
        end=args.end,
        resume=args.resume,
        discover_only=args.discover_only,
    )


if __name__ == "__main__":
    main()
