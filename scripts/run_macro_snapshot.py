"""
Macro Snapshot CLI
==================
Entry point for all macro snapshot operations.

Modes:
  --weekly    Full weekly snapshot: refresh data, generate snapshot, store in DB, print report.
              Run every Friday at 5:00 PM ET.

  --daily     Event gate check only: update FOMC/CPI/NFP calendar and scaling factor.
              Run every weekday at 6:00 AM ET. Fast (<1s, no API calls).

  --backfill  Import 323 historical snapshots (2020-2026) from JSON files into macro_state.db.
              Run once after deployment.

  --date DATE Generate/regenerate snapshot for a specific date (YYYY-MM-DD).

Usage:
  python3 scripts/run_macro_snapshot.py --weekly
  python3 scripts/run_macro_snapshot.py --daily
  python3 scripts/run_macro_snapshot.py --backfill
  python3 scripts/run_macro_snapshot.py --date 2024-06-07
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from compass.events import get_upcoming_events, run_daily_event_check
from compass.macro import MacroSnapshotEngine
from compass.macro_db import (
    get_latest_snapshot_date,
    get_snapshot_count,
    init_db,
    save_snapshot,
    set_state,
)

HISTORICAL_SNAPSHOTS_DIR = PROJECT_ROOT / "output" / "historical_snapshots"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def most_recent_friday(as_of: date = None) -> date:
    """Return the most recent Friday on or before as_of (default: today)."""
    d = as_of or date.today()
    days_since_friday = (d.weekday() - 4) % 7  # 4 = Friday
    return d - timedelta(days=days_since_friday)


def build_engine() -> MacroSnapshotEngine:
    polygon_key = os.getenv("POLYGON_API_KEY", "")
    if not polygon_key:
        logger.error("POLYGON_API_KEY not set. Aborting.")
        sys.exit(1)
    return MacroSnapshotEngine(
        polygon_key=polygon_key,
        cache_dir=str(PROJECT_ROOT / "data" / "macro_cache"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode: --daily
# ─────────────────────────────────────────────────────────────────────────────

def run_daily():
    """Event gate check: update FOMC/CPI/NFP calendar and scaling factor."""
    logger.info("Running daily event gate check...")
    init_db()
    scaling, events = run_daily_event_check()

    if events:
        logger.info("Upcoming events (within 5 days):")
        for ev in events:
            logger.info(
                "  %s  %s  days_out=%d  scaling=%.2f",
                ev["event_type"],
                ev["event_date"],
                ev["days_out"],
                ev["scaling_factor"],
            )
        logger.info("Composite scaling factor: %.2f", scaling)
    else:
        logger.info("No events within 5 days. Scaling factor: 1.00")

    print(f"\nEVENT_SCALING_FACTOR={scaling:.2f}")
    if events:
        for ev in events:
            print(f"  {ev['event_type']:6s}  {ev['event_date']}  days_out={ev['days_out']}  scale={ev['scaling_factor']:.2f}")
    else:
        print("  No active events.")

    set_state("last_daily_check", date.today().strftime("%Y-%m-%d"))
    return scaling


# ─────────────────────────────────────────────────────────────────────────────
# Mode: --weekly
# ─────────────────────────────────────────────────────────────────────────────

def run_weekly():
    """Full weekly snapshot: refresh data, generate, store, report."""
    logger.info("Running weekly full macro snapshot...")
    init_db()

    engine = build_engine()
    try:
        snap_date = most_recent_friday()
        logger.info("Snapshot date: %s", snap_date)

        # 1. Refresh price cache (last 20 days)
        logger.info("Refreshing price data...")
        engine.refresh_price_cache(days_back=20)

        # 2. Refresh FRED data
        logger.info("Refreshing FRED macro data...")
        engine.refresh_fred_cache()

        # 3. Generate snapshot
        logger.info("Generating snapshot for %s...", snap_date)
        snap = engine.generate_snapshot(snap_date)

        # 4. Store in macro_state.db
        engine.save_to_db(snap)
        logger.info("Snapshot saved to macro_state.db")

        # 5. Run daily event check
        run_daily_event_check()

        # 6. Print report to stdout
        _print_snapshot_report(snap)

    finally:
        engine.close()


# ─────────────────────────────────────────────────────────────────────────────
# Mode: --date DATE
# ─────────────────────────────────────────────────────────────────────────────

def run_for_date(target_date: date):
    """Generate (or regenerate) snapshot for a specific date."""
    logger.info("Generating snapshot for %s", target_date)
    init_db()

    engine = build_engine()
    try:
        snap = engine.generate_snapshot(target_date)
        engine.save_to_db(snap)
        _print_snapshot_report(snap)
    finally:
        engine.close()


# ─────────────────────────────────────────────────────────────────────────────
# Mode: --backfill
# ─────────────────────────────────────────────────────────────────────────────

def run_backfill(force: bool = False):
    """
    Import all historical JSON snapshots into macro_state.db.
    Skips dates already imported unless --force is passed.
    """
    logger.info("Starting historical backfill into macro_state.db...")
    init_db()

    if not HISTORICAL_SNAPSHOTS_DIR.exists():
        logger.error("Historical snapshots directory not found: %s", HISTORICAL_SNAPSHOTS_DIR)
        logger.error("Run scripts/generate_historical_snapshots.py first.")
        sys.exit(1)

    json_files = sorted(HISTORICAL_SNAPSHOTS_DIR.glob("*/????-??-??.json"))
    logger.info("Found %d snapshot JSON files", len(json_files))

    imported = 0
    skipped = 0
    errors = 0

    for json_path in json_files:
        snap_date = json_path.stem  # e.g. "2022-10-07"
        try:
            with open(json_path) as f:
                snap = json.load(f)

            # Save (INSERT OR REPLACE, so idempotent)
            save_snapshot(snap)
            imported += 1

            if imported % 50 == 0:
                logger.info("  Imported %d / %d...", imported, len(json_files))

        except Exception as exc:
            logger.error("Failed to import %s: %s", snap_date, exc)
            errors += 1

    set_state("last_backfill", date.today().strftime("%Y-%m-%d"))
    logger.info(
        "Backfill complete — imported: %d, skipped: %d, errors: %d",
        imported, skipped, errors,
    )
    logger.info("Total snapshots in DB: %d", get_snapshot_count())
    logger.info("Latest snapshot date: %s", get_latest_snapshot_date())


# ─────────────────────────────────────────────────────────────────────────────
# Console report (brief, used by weekly mode)
# ─────────────────────────────────────────────────────────────────────────────

def _print_snapshot_report(snap: dict):
    ms = snap.get("macro_score") or {}
    ind = ms.get("indicators") or {}
    rankings = snap.get("sector_rankings") or []
    events = get_upcoming_events(horizon_days=14)

    print("\n" + "=" * 60)
    print(f"  MACRO SNAPSHOT — {snap['date']}")
    print("=" * 60)
    print(f"  SPY Close:  ${snap.get('spy_close', 'N/A'):.2f}" if snap.get("spy_close") else "  SPY Close:  N/A")
    print()

    # Macro score
    overall = ms.get("overall")
    if overall is not None:
        regime = "BULL_MACRO" if overall >= 65 else ("BEAR_MACRO" if overall < 45 else "NEUTRAL_MACRO")
        print(f"  MACRO SCORE: {overall:.1f}/100  [{regime}]")
        print(f"    Growth:       {ms.get('growth', 'N/A'):.1f}")
        print(f"    Inflation:    {ms.get('inflation', 'N/A'):.1f}")
        print(f"    Fed Policy:   {ms.get('fed_policy', 'N/A'):.1f}")
        print(f"    Risk Appetite:{ms.get('risk_appetite', 'N/A'):.1f}")
        print()
        print("  KEY INDICATORS:")
        if ind.get("vix"):
            print(f"    VIX:          {ind['vix']:.2f}")
        if ind.get("t10y2y") is not None:
            print(f"    10Y-2Y:       {ind['t10y2y']:.3f}%")
        if ind.get("hy_oas_pct"):
            print(f"    HY OAS:       {ind['hy_oas_pct']:.3f}%")
        if ind.get("cpi_yoy_pct"):
            print(f"    CPI YoY:      {ind['cpi_yoy_pct']:.2f}%")
        if ind.get("fedfunds"):
            print(f"    Fed Funds:    {ind['fedfunds']:.2f}%")
    print()

    # Sector rankings
    print("  SECTOR RANKINGS (3M RS vs SPY):")
    print(f"  {'#':>2}  {'Ticker':<6}  {'RS 3M':>7}  {'RS 12M':>7}  {'Quadrant':<10}")
    print("  " + "-" * 46)
    for item in rankings[:8]:
        rs3 = f"{item['rs_3m']:+.1f}%" if item.get("rs_3m") is not None else "  N/A "
        rs12 = f"{item['rs_12m']:+.1f}%" if item.get("rs_12m") is not None else "  N/A "
        quad = item.get("rrg_quadrant") or "—"
        print(f"  {item['rank_3m']:>2}  {item['ticker']:<6}  {rs3:>7}  {rs12:>7}  {quad:<10}")

    leading = snap.get("leading_sectors") or []
    lagging = snap.get("lagging_sectors") or []
    if leading:
        print(f"\n  LEADING:  {', '.join(leading)}")
    if lagging:
        print(f"  LAGGING:  {', '.join(lagging)}")

    # Events
    if events:
        print("\n  UPCOMING EVENTS:")
        for ev in events:
            print(f"    {ev['event_type']:6s}  {ev['event_date']}  (T-{ev['days_out']}d)  scale={ev['scaling_factor']:.2f}")
        from compass.events import compute_composite_scaling
        composite = compute_composite_scaling(events)
        print(f"\n  EVENT SCALING FACTOR: {composite:.2f}")
    else:
        print("\n  No macro events in next 14 days. Scaling: 1.00")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Macro Snapshot CLI")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--weekly",   action="store_true", help="Full weekly snapshot")
    group.add_argument("--daily",    action="store_true", help="Event gate check only")
    group.add_argument("--backfill", action="store_true", help="Import historical JSON snapshots")
    group.add_argument("--date",     metavar="YYYY-MM-DD", help="Generate snapshot for specific date")
    p.add_argument("--force", action="store_true", help="Re-import even if already in DB")
    return p.parse_args()


def main():
    args = parse_args()

    if args.daily:
        run_daily()

    elif args.weekly:
        run_weekly()

    elif args.backfill:
        run_backfill(force=args.force)

    elif args.date:
        target = date.fromisoformat(args.date)
        run_for_date(target)


if __name__ == "__main__":
    main()
