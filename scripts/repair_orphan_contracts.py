#!/usr/bin/env python3
"""
Repair orphaned option bars from the GLD/QQQ backfill.

BUG: scripts/backfill_gld_qqq.py ensure_contract() failed to provide as_of_date
(NOT NULL) so every contract metadata insert was silently rejected by
INSERT OR IGNORE. But bars landed in option_daily anyway → ~486K orphaned bars.

FIX: Parse the OCC symbol (e.g. O:GLD241115P00240000) to reconstruct
ticker + expiration + strike + option_type, then INSERT with as_of_date.

OCC format: O:{TICKER}{YYMMDD}{C|P}{STRIKE*1000:08d}
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "options_cache.db"


def parse_occ(symbol: str):
    """Parse OCC symbol → (ticker, expiration, strike, option_type).

    Example: O:GLD241115P00240000
      O:   prefix
      GLD  ticker (variable length, until digits)
      241115 date YYMMDD
      P    option type
      00240000 strike * 1000 (8 digits)
    """
    if not symbol.startswith("O:"):
        return None
    body = symbol[2:]

    # Find where the date starts (first digit run of 6+)
    i = 0
    while i < len(body) and body[i].isalpha():
        i += 1
    ticker = body[:i]
    if not ticker or len(body) < i + 15:
        return None

    date_str = body[i:i+6]
    try:
        dt = datetime.strptime(date_str, "%y%m%d")
        exp_str = dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

    opt_type = body[i+6]
    if opt_type not in ("P", "C"):
        return None

    strike_str = body[i+7:i+15]
    try:
        strike = int(strike_str) / 1000.0
    except ValueError:
        return None

    return (ticker, exp_str, strike, opt_type)


def main():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    # Find orphaned symbols for GLD and QQQ
    print("Finding orphaned bars...")
    orphaned = conn.execute("""
        SELECT DISTINCT od.contract_symbol
        FROM option_daily od
        LEFT JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
        WHERE oc.contract_symbol IS NULL
          AND (od.contract_symbol LIKE 'O:GLD%' OR od.contract_symbol LIKE 'O:QQQ%')
    """).fetchall()

    print(f"Found {len(orphaned)} orphaned symbols")

    today = datetime.utcnow().strftime("%Y-%m-%d")
    inserted = 0
    failed = 0

    for (sym,) in orphaned:
        parsed = parse_occ(sym)
        if parsed is None:
            failed += 1
            continue
        ticker, exp, strike, opt_type = parsed

        try:
            conn.execute("""
                INSERT OR IGNORE INTO option_contracts
                    (ticker, expiration, strike, option_type, contract_symbol, as_of_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ticker, exp, strike, opt_type, sym, today))
            if conn.total_changes > inserted:
                inserted = conn.total_changes
        except sqlite3.Error as e:
            print(f"  Error inserting {sym}: {e}")
            failed += 1

    conn.commit()
    print(f"Repaired: {inserted} contracts inserted, {failed} failed")

    # Verify
    print("\nPost-repair state:")
    for tk in ["GLD", "QQQ"]:
        r = conn.execute("""
            SELECT COUNT(DISTINCT oc.contract_symbol), MAX(oc.expiration), MAX(od.date)
            FROM option_contracts oc
            JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
            WHERE oc.ticker = ? AND od.close > 0
        """, (tk,)).fetchone()
        print(f"  {tk}: {r[0]:,} contracts, max exp {r[1]}, last bar date {r[2]}")

    conn.close()
    return inserted


if __name__ == "__main__":
    n = main()
    sys.exit(0 if n > 0 else 1)
