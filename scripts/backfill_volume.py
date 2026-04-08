#!/usr/bin/env python3
"""
backfill_volume.py — Pre-populate option_daily volume for every contract ever priced.

Iterates all distinct contract symbols in option_intraday, fetches their full
daily OHLCV history from Polygon, and inserts missing rows into option_daily so
the volume gate in backtester.py has data to work with.

Usage:
    python3 scripts/backfill_volume.py
    python3 scripts/backfill_volume.py --dry-run    # count work items, no API calls
    python3 scripts/backfill_volume.py --limit 500  # process at most N contracts
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill")

CHECKPOINT_PATH = ROOT / "data" / "backfill_checkpoint.json"
DB_PATH         = ROOT / "data" / "options_cache.db"


def _load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_checkpoint(cp: dict):
    CHECKPOINT_PATH.write_text(json.dumps(cp, indent=2))


def _business_days_in_range(from_date: str, to_date: str) -> int:
    """Rough count of Mon-Fri days between two YYYY-MM-DD strings (inclusive)."""
    d0 = datetime.strptime(from_date, "%Y-%m-%d")
    d1 = datetime.strptime(to_date, "%Y-%m-%d")
    days = (d1 - d0).days + 1
    # Simple approximation: 5/7 of calendar days
    return max(1, int(days * 5 / 7))


def _existing_daily_count(conn: sqlite3.Connection, sym: str, from_date: str, to_date: str) -> int:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM option_daily "
        "WHERE contract_symbol = ? AND date >= ? AND date <= ? AND volume IS NOT NULL",
        (sym, from_date, to_date),
    )
    row = cur.fetchone()
    return row[0] if row else 0


def _fetch_daily_bars(hd, sym: str, from_str: str, to_str: str) -> list:
    """Fetch daily bars via HistoricalOptionsData._api_get (rate-limited + retry)."""
    data = hd._api_get(
        f"/v2/aggs/ticker/{sym}/range/1/day/{from_str}/{to_str}",
        params={"adjusted": "true", "sort": "asc", "limit": 5000},
    )
    return data.get("results", [])


def _insert_daily_bars(conn: sqlite3.Connection, sym: str, bars: list) -> int:
    """INSERT OR IGNORE bars into option_daily. Returns number of new rows inserted."""
    if not bars:
        return 0
    rows = []
    for bar in bars:
        ts = bar.get("t", 0)
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append((
            sym,
            dt,
            bar.get("o"),
            bar.get("h"),
            bar.get("l"),
            bar.get("c"),
            bar.get("v", 0),
            bar.get("oi"),  # open_interest — NULL on standard Polygon tier
        ))

    # COUNT before/after is the only reliable way to get true insertion count.
    # SQLite's changes() returns the count from only the *last* row of an
    # executemany batch, not the total, so it always reads 0 or 1.
    dates_in_batch = [r[1] for r in rows]
    min_date, max_date = min(dates_in_batch), max(dates_in_batch)
    before_count = conn.execute(
        "SELECT COUNT(*) FROM option_daily "
        "WHERE contract_symbol = ? AND date BETWEEN ? AND ?",
        (sym, min_date, max_date),
    ).fetchone()[0]

    cur = conn.cursor()
    cur.executemany(
        "INSERT OR IGNORE INTO option_daily "
        "(contract_symbol, date, open, high, low, close, volume, open_interest) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    after_count = conn.execute(
        "SELECT COUNT(*) FROM option_daily "
        "WHERE contract_symbol = ? AND date BETWEEN ? AND ?",
        (sym, min_date, max_date),
    ).fetchone()[0]
    return after_count - before_count


def run(dry_run: bool = False, limit: int = 0):
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key and not dry_run:
        print("ERROR: POLYGON_API_KEY not set. Run: export $(grep -v '^#' .env | xargs)")
        sys.exit(1)

    from backtest.historical_data import HistoricalOptionsData

    # Use HistoricalOptionsData only for session/retry/rate-limit infrastructure.
    # In dry-run mode we still init it (reads from cache only — no API key needed).
    hd = HistoricalOptionsData(api_key or "dummy", cache_dir=str(ROOT / "data"))
    conn = hd._conn

    checkpoint = _load_checkpoint()

    # --- Discover contracts ---
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT contract_symbol
        FROM option_intraday
        WHERE bar_time != 'FETCHED'
        ORDER BY contract_symbol
    """)
    all_symbols = [row[0] for row in cur.fetchall()]

    total = len(all_symbols)
    if limit > 0:
        all_symbols = all_symbols[:limit]

    print(f"Backfill: {len(all_symbols)} contracts to process (total in cache: {total})")
    if dry_run:
        print("DRY-RUN — no API calls will be made")

    skipped_done = 0
    skipped_norange = 0
    processed = 0
    total_new_rows = 0
    t_start = time.time()

    for idx, sym in enumerate(all_symbols, 1):
        # --- Determine date range from intraday rows ---
        cur.execute(
            "SELECT MIN(date), MAX(date) FROM option_intraday "
            "WHERE contract_symbol = ? AND bar_time != 'FETCHED'",
            (sym,),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            skipped_norange += 1
            continue

        min_date_raw, max_date_raw = row
        # Extend min_date back 5 calendar days for prev-day volume lookups
        min_dt = datetime.strptime(min_date_raw, "%Y-%m-%d") - timedelta(days=5)
        from_str = min_dt.strftime("%Y-%m-%d")
        to_str   = max_date_raw

        # --- Skip check: ≥ 80% of expected business days already cached ---
        expected_bdays = _business_days_in_range(from_str, to_str)
        existing = _existing_daily_count(conn, sym, from_str, to_str)
        coverage = existing / expected_bdays if expected_bdays > 0 else 1.0

        if coverage >= 0.80 or checkpoint.get(sym):
            skipped_done += 1
            checkpoint[sym] = True
            if idx % 500 == 0:
                _save_checkpoint(checkpoint)
            continue

        # --- ETA ---
        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 and processed > 0 else 0.5
        remaining = len(all_symbols) - idx
        eta_s = remaining / rate if rate > 0 else 0
        eta_str = f"{int(eta_s // 3600)}h{int((eta_s % 3600) // 60)}m"

        if dry_run:
            print(f"[{idx}/{len(all_symbols)}] {sym}  from={from_str} to={to_str}  "
                  f"coverage={coverage:.0%}  (would fetch)  ETA: {eta_str}")
            processed += 1
            continue

        # --- Fetch from Polygon ---
        bars = _fetch_daily_bars(hd, sym, from_str, to_str)
        new_rows = _insert_daily_bars(conn, sym, bars)

        total_new_rows += new_rows
        checkpoint[sym] = True
        processed += 1

        print(f"[{idx}/{len(all_symbols)}] {sym}  +{new_rows} rows  "
              f"(coverage was {coverage:.0%})  ETA: {eta_str}")

        # Save checkpoint every 100 contracts
        if processed % 100 == 0:
            _save_checkpoint(checkpoint)

    # Final checkpoint save
    _save_checkpoint(checkpoint)

    print(f"\nDone. Processed={processed}, skipped_done={skipped_done}, "
          f"skipped_norange={skipped_norange}, new_rows={total_new_rows}")

    # Final coverage report
    cur.execute("SELECT COUNT(*) FROM option_daily WHERE volume IS NOT NULL")
    total_vol_rows = cur.fetchone()[0]
    print(f"option_daily rows with volume: {total_vol_rows:,}")


def main():
    parser = argparse.ArgumentParser(description="Backfill option_daily volume from Polygon")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count work items and estimate runtime without making API calls")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N contracts (0 = all)")
    args = parser.parse_args()
    run(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
