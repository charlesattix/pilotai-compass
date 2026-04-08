#!/usr/bin/env python3
"""
backfill_macro_events.py — One-time script to populate macro_events table
with all FOMC, CPI, and NFP events for 2020–2025.

This enables the event gate feature for historical backtesting.

Usage:
    python3 scripts/backfill_macro_events.py [--db-path PATH] [--dry-run]

Key design note:
  For historical backfilling, we store the event dates in the macro_events
  table but the backtester must compute days_out dynamically from the
  trading date being evaluated (not from the time the row was inserted).
  Only event_date, event_type, description, and is_emergency are meaningful
  for historical rows. scaling_factor and days_out stored here are informational
  only (they're relative to the time of this script run, not the backtest date).
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from compass.events import (
    ALL_FOMC_DATES,
    FOMC_EMERGENCY_DATES,
    _cpi_release_date,
    _nfp_release_date,
)
from compass.macro_db import get_db, init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_fomc_events(year_start: int = 2020, year_end: int = 2025):
    """Generate FOMC event rows for all dates in range."""
    rows = []
    for fd in ALL_FOMC_DATES:
        if not (year_start <= fd.year <= year_end):
            continue
        is_emergency = 1 if fd in FOMC_EMERGENCY_DATES else 0
        event_type = "FOMC_EMERGENCY" if is_emergency else "FOMC_SCHEDULED"
        desc = (
            f"FOMC Emergency Cut — {fd.strftime('%b %d, %Y')}"
            if is_emergency
            else f"FOMC Rate Decision — {fd.strftime('%b %d, %Y')}"
        )
        rows.append({
            "event_date": fd.strftime("%Y-%m-%d"),
            "event_type": event_type,
            "description": desc,
            "days_out": 0,       # historical: event has passed
            "scaling_factor": 1.0,  # historical: no active restriction
            "is_emergency": is_emergency,
        })
    return rows


def build_cpi_events(year_start: int = 2020, year_end: int = 2025):
    """Generate CPI release event rows for all months in range."""
    rows = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            rd = _cpi_release_date(year, month)
            # Only include dates within our historical range
            if not (date(year_start, 1, 1) <= rd <= date(year_end, 12, 31)):
                continue
            rows.append({
                "event_date": rd.strftime("%Y-%m-%d"),
                "event_type": "CPI",
                "description": f"CPI Release ({year}-{month:02d}) — {rd.strftime('%b %d, %Y')}",
                "days_out": 0,
                "scaling_factor": 1.0,
                "is_emergency": 0,
            })
    return rows


def build_nfp_events(year_start: int = 2020, year_end: int = 2025):
    """Generate NFP (jobs report) event rows for all months in range."""
    rows = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            rd = _nfp_release_date(year, month)
            if not (date(year_start, 1, 1) <= rd <= date(year_end, 12, 31)):
                continue
            rows.append({
                "event_date": rd.strftime("%Y-%m-%d"),
                "event_type": "NFP",
                "description": f"NFP Jobs Report ({year}-{month:02d}) — {rd.strftime('%b %d, %Y')}",
                "days_out": 0,
                "scaling_factor": 1.0,
                "is_emergency": 0,
            })
    return rows


def upsert_events_with_emergency(events, db_path=None):
    """Upsert macro event rows including is_emergency column."""
    conn = get_db(db_path)
    try:
        for ev in events:
            conn.execute(
                """
                INSERT OR REPLACE INTO macro_events
                  (event_date, event_type, description, days_out, scaling_factor, is_emergency)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ev["event_date"],
                    ev["event_type"],
                    ev.get("description", ""),
                    ev.get("days_out", 0),
                    ev.get("scaling_factor", 1.0),
                    ev.get("is_emergency", 0),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill macro_events table with 2020-2025 data")
    parser.add_argument("--db-path", default=None, help="Override macro_state.db path")
    parser.add_argument("--dry-run", action="store_true", help="Print events without writing to DB")
    parser.add_argument("--year-start", type=int, default=2020)
    parser.add_argument("--year-end", type=int, default=2025)
    args = parser.parse_args()

    # Ensure schema is current (runs migrations if needed)
    init_db(args.db_path)

    fomc = build_fomc_events(args.year_start, args.year_end)
    cpi  = build_cpi_events(args.year_start, args.year_end)
    nfp  = build_nfp_events(args.year_start, args.year_end)
    all_events = fomc + cpi + nfp

    logger.info(
        "Generated %d events: %d FOMC (%d emergency), %d CPI, %d NFP",
        len(all_events),
        len(fomc),
        sum(1 for e in fomc if e["is_emergency"]),
        len(cpi),
        len(nfp),
    )

    if args.dry_run:
        logger.info("DRY RUN — first 10 events:")
        for ev in all_events[:10]:
            logger.info("  %s", ev)
        return

    upsert_events_with_emergency(all_events, db_path=args.db_path)
    logger.info("Done — %d rows written to macro_events.", len(all_events))

    # Verify
    conn = get_db(args.db_path)
    try:
        count = conn.execute("SELECT COUNT(*) AS n FROM macro_events").fetchone()["n"]
        emergency = conn.execute(
            "SELECT COUNT(*) AS n FROM macro_events WHERE is_emergency = 1"
        ).fetchone()["n"]
        logger.info("Verification: macro_events now has %d rows (%d emergency FOMC)", count, emergency)

        # Spot-check: FOMC 2022-06-15
        row = conn.execute(
            "SELECT * FROM macro_events WHERE event_date = '2022-06-15'"
        ).fetchone()
        if row:
            logger.info("Spot-check 2022-06-15: %s", dict(row))
        else:
            logger.warning("Spot-check 2022-06-15: NOT FOUND")

        # Spot-check: emergency FOMC 2020-03-15
        row2 = conn.execute(
            "SELECT * FROM macro_events WHERE event_date = '2020-03-15'"
        ).fetchone()
        if row2:
            logger.info("Spot-check 2020-03-15 (emergency): %s", dict(row2))
        else:
            logger.warning("Spot-check 2020-03-15: NOT FOUND")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
