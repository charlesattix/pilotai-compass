#!/usr/bin/env python3
"""List experiments from registry.json.

Usage:
    python scripts/list_experiments.py           # live only (default)
    python scripts/list_experiments.py --live    # live paper trading
    python scripts/list_experiments.py --dev     # in development
    python scripts/list_experiments.py --retired # retired
    python scripts/list_experiments.py --all     # all three tables
"""

import json
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "experiments" / "registry.json"

LIVE_STATUSES    = {"paper_trading", "deployed"}
DEV_STATUSES     = {"in_development", "data_collection", "backtesting", "validated",
                    "pending", "awaiting_deploy"}
RETIRED_STATUSES = {"retired"}


def load_registry() -> dict:
    if not REGISTRY.exists():
        print(f"ERROR: registry not found at {REGISTRY}", file=sys.stderr)
        sys.exit(1)
    with open(REGISTRY) as f:
        return json.load(f)


def sort_key(e: dict) -> int:
    try:
        return int(e["id"].split("-")[1])
    except (IndexError, ValueError):
        return 0


def col_widths(rows: list[list[str]], headers: list[str]) -> list[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"\n{title}")

    if not rows:
        print("  (none)\n")
        return

    widths = col_widths(rows, headers)
    sep    = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt    = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"

    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)
    print(f"  {len(rows)} experiment(s)\n")


def live_table(experiments: list[dict]) -> None:
    headers = ["ID", "Name", "Creator", "Ticker", "Account", "Config", "Live Since"]
    rows = []
    for e in experiments:
        rows.append([
            e["id"],
            e["name"],
            e["created_by"],
            e.get("ticker") or "—",
            e.get("account_id") or "—",
            e.get("paper_config") or "—",
            e.get("live_since") or "—",
        ])
    print_table("Live Paper Trading", headers, rows)


def dev_table(experiments: list[dict]) -> None:
    headers = ["ID", "Name", "Creator", "Phase", "Next Step"]
    rows = []
    for e in experiments:
        next_step = e.get("next_step") or "—"
        # Truncate long next_step for display
        if len(next_step) > 60:
            next_step = next_step[:57] + "..."
        rows.append([
            e["id"],
            e["name"],
            e["created_by"],
            e.get("phase") or "—",
            next_step,
        ])
    print_table("In Development", headers, rows)


def retired_table(experiments: list[dict]) -> None:
    headers = ["ID", "Name", "Creator", "Why Retired"]
    rows = []
    for e in experiments:
        reason = e.get("retired_reason") or e.get("notes") or "—"
        if len(reason) > 70:
            reason = reason[:67] + "..."
        rows.append([
            e["id"],
            e["name"],
            e["created_by"],
            reason,
        ])
    print_table("Retired", headers, rows)


def main() -> None:
    args = sys.argv[1:]
    valid_flags = {"--live", "--dev", "--retired", "--all"}
    if len(args) > 1 or (args and args[0] not in valid_flags):
        print("Usage: list_experiments.py [--live|--dev|--retired|--all]")
        sys.exit(1)

    mode = args[0] if args else "--live"

    registry = load_registry()
    all_exps = sorted(registry["experiments"].values(), key=sort_key)

    live    = [e for e in all_exps if e["status"] in LIVE_STATUSES]
    dev     = [e for e in all_exps if e["status"] in DEV_STATUSES]
    retired = [e for e in all_exps if e["status"] in RETIRED_STATUSES]

    print(f"Experiment Registry  "
          f"(schema v{registry.get('schema_version', '?')}  |  "
          f"last updated: {registry.get('last_updated', '?')})")

    if mode == "--live":
        live_table(live)
    elif mode == "--dev":
        dev_table(dev)
    elif mode == "--retired":
        retired_table(retired)
    else:  # --all
        live_table(live)
        dev_table(dev)
        retired_table(retired)


if __name__ == "__main__":
    main()
