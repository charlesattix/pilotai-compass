#!/usr/bin/env python3
"""
iron_vault_setup.py — Iron Vault Setup & Validation

Validates that all data prerequisites for backtesting are in place:
  1. POLYGON_API_KEY is set in .env
  2. options_cache.db exists and has data
  3. Reports coverage (tickers, years, contract counts)
  4. Flags any critical gaps

Usage:
    python scripts/iron_vault_setup.py            # check + report
    python scripts/iron_vault_setup.py --verbose  # detailed per-ticker breakdown
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print()
    print("═" * 65)
    print("  IRON VAULT — Data Layer Validation")
    print("═" * 65)

    errors = []
    warnings = []

    # ── 1. POLYGON_API_KEY ────────────────────────────────────────────────────
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        errors.append(
            "POLYGON_API_KEY not set. Add it to .env:\n"
            "  POLYGON_API_KEY=your_key_here\n"
            "  (needed for backfill runs; offline backtests work without it)"
        )
        print("  ✗ POLYGON_API_KEY: NOT SET")
    else:
        masked = api_key[:4] + "****" + api_key[-4:]
        print(f"  ✓ POLYGON_API_KEY: {masked}")

    # ── 2. options_cache.db ───────────────────────────────────────────────────
    from shared.constants import DATA_DIR
    db_path = os.path.join(DATA_DIR, "options_cache.db")
    db_size_mb = os.path.getsize(db_path) / 1024 / 1024 if os.path.exists(db_path) else 0

    if not os.path.exists(db_path):
        errors.append(
            f"options_cache.db not found at {db_path}.\n"
            "  Fetch SPY data: python scripts/fetch_polygon_options.py --ticker SPY\n"
            "  Fetch sectors:  python scripts/fetch_sector_options.py --ticker XLE"
        )
        print(f"  ✗ options_cache.db: NOT FOUND at {db_path}")
    else:
        print(f"  ✓ options_cache.db: {db_size_mb:.0f} MB at {db_path}")

    if errors:
        print()
        print("  CRITICAL ERRORS — Iron Vault cannot start:")
        for e in errors:
            print(f"\n  ✗ {e}")
        print()
        sys.exit(1)

    # ── 3. Coverage report ────────────────────────────────────────────────────
    try:
        from shared.iron_vault import IronVault, IronVaultError
        vault = IronVault(api_key or "offline", cache_dir=DATA_DIR)
        report = vault.coverage_report()
        vault.close()
    except IronVaultError as e:
        print(f"\n  ✗ Iron Vault init failed: {e}")
        sys.exit(1)

    print()
    print("  Coverage Summary:")
    print(f"  {'Metric':<30} {'Count':>12}")
    print("  " + "─" * 44)
    print(f"  {'Option contracts':<30} {report['contracts_total']:>12,}")
    print(f"  {'Daily OHLCV bars':<30} {report['daily_bars_total']:>12,}")
    print(f"  {'Intraday 5-min bars':<30} {report['intraday_bars_total']:>12,}")

    if verbose:
        print()
        print("  Per-Ticker Breakdown:")
        print(f"  {'Ticker':<8} {'Contracts':>10} {'Years'}")
        print("  " + "─" * 50)
        for ticker, info in sorted(report["by_ticker"].items()):
            years_str = ", ".join(str(y) for y in info["years"]) if info["years"] else "—"
            flag = ""
            if info["contracts"] < 100:
                flag = "  ⚠ sparse"
                warnings.append(f"{ticker}: only {info['contracts']} contracts")
            print(f"  {ticker:<8} {info['contracts']:>10,}  {years_str}{flag}")

    # ── 4. Gap analysis ───────────────────────────────────────────────────────
    REQUIRED_TICKERS = ["SPY"]
    REQUIRED_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

    print()
    print("  Required Coverage (SPY 2020–2025):")
    by_ticker = report["by_ticker"]
    all_ok = True

    for ticker in REQUIRED_TICKERS:
        if ticker not in by_ticker:
            warnings.append(f"{ticker}: no data at all")
            print(f"  ✗ {ticker}: NO DATA")
            all_ok = False
            continue
        info = by_ticker[ticker]
        covered = set(info["years"])
        missing = [y for y in REQUIRED_YEARS if y not in covered]
        if missing:
            warnings.append(f"{ticker}: missing years {missing}")
            print(f"  ⚠ {ticker}: missing years {missing}")
            all_ok = False
        else:
            print(f"  ✓ {ticker}: all 6 years covered ({info['contracts']:,} contracts)")

    if warnings and not verbose:
        print()
        print("  Warnings (run --verbose for details):")
        for w in warnings[:5]:
            print(f"  ⚠ {w}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings)-5} more")

    print()
    if all_ok:
        print("  ✅ Iron Vault is READY — all required data present.")
    else:
        print("  ⚠  Iron Vault has gaps — backtests may skip affected dates.")
        print("     Run: python scripts/fetch_polygon_options.py --ticker SPY --resume")
    print("═" * 65)
    print()

    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
