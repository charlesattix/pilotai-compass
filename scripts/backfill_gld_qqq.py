#!/usr/bin/env python3
"""
Targeted backfill for GLD and QQQ options.

Since Polygon Starter tier can't enumerate contracts, we build OCC symbols
from known strike grids + expiration dates, then fetch daily bars.

GLD: extend from 2024-10-18 → 2025-12-31
QQQ: extend from 2023-04-21 → 2025-12-31

OCC symbol format: O:{TICKER}{YYMMDD}{C/P}{STRIKE*1000:08d}
Example: O:GLD241220P00230000 = GLD Dec 20 2024 230 Put
"""

import json, logging, os, sqlite3, sys, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

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

DB_PATH = ROOT / "data" / "options_cache.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Load API key
API_KEY = os.environ.get("POLYGON_API_KEY", "")
if not API_KEY:
    for envf in [".env", ".env.exp400"]:
        try:
            with open(ROOT / envf) as f:
                for line in f:
                    if line.startswith("POLYGON_API_KEY"):
                        API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
        except FileNotFoundError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# OCC symbol builder
# ═══════════════════════════════════════════════════════════════════════════

def build_occ(ticker: str, exp_date: str, strike: float, opt_type: str) -> str:
    """Build OCC symbol: O:GLD241220P00230000"""
    dt = datetime.strptime(exp_date, "%Y-%m-%d")
    ymd = dt.strftime("%y%m%d")
    strike_int = int(strike * 1000)
    return f"O:{ticker}{ymd}{opt_type}{strike_int:08d}"


def generate_expirations(start_date: str, end_date: str) -> List[str]:
    """Generate monthly option expiration dates (3rd Friday of each month)."""
    exps = []
    dt = datetime.strptime(start_date, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while dt <= end:
        # Find 3rd Friday
        first = dt.replace(day=1)
        # Weekday of first day (0=Mon, 4=Fri)
        wd = first.weekday()
        # First Friday
        first_fri = first + timedelta(days=(4 - wd) % 7)
        third_fri = first_fri + timedelta(days=14)
        if third_fri <= end:
            exps.append(third_fri.strftime("%Y-%m-%d"))

        # Next month
        if dt.month == 12:
            dt = dt.replace(year=dt.year + 1, month=1)
        else:
            dt = dt.replace(month=dt.month + 1)

    return exps


def generate_strikes(ticker: str, exp_date: str) -> List[Tuple[float, str]]:
    """Generate plausible strikes for a ticker around known price levels."""
    # Approximate price ranges
    if ticker == "GLD":
        # GLD was ~180-240 in 2024-2025
        base_prices = list(range(160, 260, 2))  # $2 increments
    elif ticker == "QQQ":
        # QQQ was ~310-530 in 2023-2025
        base_prices = list(range(280, 560, 5))  # $5 increments
    elif ticker == "TLT":
        # TLT was ~85-105 in 2024-2025
        base_prices = list(range(75, 115, 1))  # $1 increments
    else:
        return []

    # Generate both puts and calls for each strike
    symbols = []
    for strike in base_prices:
        for opt_type in ["P", "C"]:
            symbols.append((float(strike), opt_type))
    return symbols


# ═══════════════════════════════════════════════════════════════════════════
# Polygon API fetch
# ═══════════════════════════════════════════════════════════════════════════

_session = None

def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503])
        _session.mount("https://", HTTPAdapter(max_retries=retries))
    return _session


def fetch_daily_bars(symbol: str, from_date: str, to_date: str) -> List[dict]:
    """Fetch daily OHLCV bars from Polygon."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
    params = {"apiKey": API_KEY, "limit": 5000, "adjusted": "true"}

    try:
        r = get_session().get(url, params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(12)
            r = get_session().get(url, params=params, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("results", [])
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Database operations
# ═══════════════════════════════════════════════════════════════════════════

def open_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def ensure_contract(conn, symbol, ticker, exp_date, strike, opt_type):
    """Insert contract if not exists.

    FIX: as_of_date is NOT NULL in the schema. Previous version omitted it
    and INSERT OR IGNORE silently swallowed the NOT NULL constraint error,
    leaving option_daily bars orphaned (no matching contract metadata).
    """
    as_of = datetime.utcnow().strftime("%Y-%m-%d")
    # Normalize opt_type to P/C (matching rest of DB) not put/call
    ot = opt_type[0].upper() if opt_type else "P"
    conn.execute("""
        INSERT OR IGNORE INTO option_contracts
            (contract_symbol, ticker, expiration, strike, option_type, as_of_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (symbol, ticker, exp_date, strike, ot, as_of))


def insert_bars(conn, symbol, bars):
    """Insert daily bars, skip duplicates."""
    inserted = 0
    for bar in bars:
        ts = bar.get("t", 0)
        date_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d") if ts else "0000-00-00"
        try:
            conn.execute("""
                INSERT OR IGNORE INTO option_daily
                    (contract_symbol, date, open, high, low, close, volume, open_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, date_str,
                bar.get("o", 0), bar.get("h", 0), bar.get("l", 0), bar.get("c", 0),
                bar.get("v", 0), 0,
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    return inserted


# ═══════════════════════════════════════════════════════════════════════════
# Main backfill logic
# ═══════════════════════════════════════════════════════════════════════════

def backfill_ticker(ticker: str, start_date: str, end_date: str, max_contracts: int = 5000):
    """Backfill a ticker by generating symbols and fetching bars."""
    log.info(f"=== Backfilling {ticker}: {start_date} → {end_date} ===")

    expirations = generate_expirations(start_date, end_date)
    log.info(f"  Generated {len(expirations)} monthly expirations")

    conn = open_db()
    total_contracts = 0
    total_bars = 0
    total_fetched = 0
    api_calls = 0
    empty = 0

    for exp in expirations:
        strikes = generate_strikes(ticker, exp)
        log.info(f"  Exp {exp}: {len(strikes)} strike/type combos")

        for strike, opt_type in strikes:
            if total_contracts >= max_contracts:
                break

            symbol = build_occ(ticker, exp, strike, opt_type)

            # Check if we already have this contract with bars
            cur = conn.execute(
                "SELECT COUNT(*) FROM option_daily WHERE contract_symbol=?", (symbol,)
            )
            existing = cur.fetchone()[0]
            if existing > 0:
                continue

            # Fetch from Polygon
            # Fetch bars from 30 days before expiration to expiration
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            fetch_from = (exp_dt - timedelta(days=60)).strftime("%Y-%m-%d")
            fetch_to = exp

            bars = fetch_daily_bars(symbol, fetch_from, fetch_to)
            api_calls += 1

            if bars:
                ensure_contract(conn, symbol, ticker, exp, strike, opt_type)
                inserted = insert_bars(conn, symbol, bars)
                total_bars += inserted
                total_contracts += 1

                if total_contracts % 50 == 0:
                    conn.commit()
                    log.info(f"    Progress: {total_contracts} contracts, {total_bars} bars, "
                            f"{api_calls} API calls, {empty} empty")
            else:
                empty += 1

            # Rate limit: ~5 calls/sec
            if api_calls % 5 == 0:
                time.sleep(1.1)

        if total_contracts >= max_contracts:
            log.info(f"  Reached max_contracts limit ({max_contracts})")
            break

    conn.commit()
    conn.close()

    log.info(f"=== {ticker} complete: {total_contracts} new contracts, {total_bars} bars, "
            f"{api_calls} API calls, {empty} empty ===")

    return {"ticker": ticker, "contracts": total_contracts, "bars": total_bars,
            "api_calls": api_calls, "empty": empty}


def main():
    if not API_KEY:
        log.error("No POLYGON_API_KEY found. Set it in .env or environment.")
        sys.exit(1)

    log.info(f"API key: {API_KEY[:8]}...")
    log.info(f"Database: {DB_PATH}")

    results = []

    # GLD: extend from 2024-10-18 → 2025-12-31
    r = backfill_ticker("GLD", "2024-10-01", "2025-12-31", max_contracts=2000)
    results.append(r)

    # QQQ: extend from 2023-04-21 → 2025-12-31
    r = backfill_ticker("QQQ", "2023-05-01", "2025-12-31", max_contracts=3000)
    results.append(r)

    # Print summary
    log.info("=" * 60)
    log.info("BACKFILL SUMMARY")
    for r in results:
        log.info(f"  {r['ticker']}: {r['contracts']} contracts, {r['bars']} bars")

    # Verify new state
    conn = open_db()
    for ticker in ["GLD", "QQQ"]:
        cur = conn.execute("""
            SELECT COUNT(DISTINCT oc.contract_symbol), MAX(oc.expiration), MAX(od.date)
            FROM option_contracts oc
            JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
            WHERE oc.ticker=? AND od.date != '0000-00-00'
        """, (ticker,))
        n, max_exp, max_date = cur.fetchone()
        log.info(f"  {ticker} DB state: {n} contracts, max exp {max_exp}, last date {max_date}")
    conn.close()


if __name__ == "__main__":
    main()
