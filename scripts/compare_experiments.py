#!/usr/bin/env python3
"""Compare experiment results across isolated SQLite databases.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py data/pilotai_champion.db data/pilotai_exp401.db
"""

import sqlite3
import sys
from pathlib import Path

DEFAULT_DBS = {
    "EXP-400": "data/pilotai_champion.db",
    "EXP-401": "data/pilotai_exp401.db",
}


def query_experiment(label: str, db_path: str) -> dict:
    """Query a single experiment DB and return summary stats."""
    path = Path(db_path)
    if not path.exists():
        return {"label": label, "error": f"DB not found: {db_path}"}

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        # Total trades (closed only)
        rows = conn.execute(
            "SELECT pnl FROM trades WHERE status LIKE 'closed%'"
        ).fetchall()

        if not rows:
            return {"label": label, "trades": 0, "pnl": 0.0, "win_rate": 0.0,
                    "avg_pnl": 0.0, "max_dd": 0.0}

        pnls = [r["pnl"] or 0.0 for r in rows]
        trades = len(pnls)
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = (wins / trades * 100) if trades else 0.0
        avg_pnl = total_pnl / trades if trades else 0.0

        # Max drawdown from cumulative P&L curve
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        # Order by exit_date for proper DD calc
        ordered = conn.execute(
            "SELECT pnl FROM trades WHERE status LIKE 'closed%' ORDER BY exit_date"
        ).fetchall()
        for r in ordered:
            cumulative += r["pnl"] or 0.0
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return {
            "label": label,
            "trades": trades,
            "pnl": total_pnl,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "max_dd": max_dd,
        }
    finally:
        conn.close()


def print_table(results: list) -> None:
    """Print a formatted comparison table."""
    header = f"{'Exp':<10} {'Trades':>7} {'P&L':>10} {'Win Rate':>10} {'Avg P&L':>10} {'Max DD':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['label']:<10} {r['error']}")
            continue
        print(
            f"{r['label']:<10} "
            f"{r['trades']:>7} "
            f"${r['pnl']:>9,.2f} "
            f"{r['win_rate']:>9.1f}% "
            f"${r['avg_pnl']:>9,.2f} "
            f"${r['max_dd']:>9,.2f}"
        )


def main():
    if len(sys.argv) > 1:
        # Custom DB paths from command line
        dbs = {}
        for i, path in enumerate(sys.argv[1:]):
            label = Path(path).stem.replace("pilotai_", "").upper()
            dbs[label] = path
    else:
        dbs = DEFAULT_DBS

    results = [query_experiment(label, path) for label, path in dbs.items()]
    print_table(results)


if __name__ == "__main__":
    main()
