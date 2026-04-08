#!/usr/bin/env python3
"""Register a new experiment in experiments/registry.json.

Creator attribution is MANDATORY — this script cannot complete without it.
All new experiments must go through this script; manual registry edits
will fail validate_registry.py.

Usage (interactive):
    python scripts/register_experiment.py

Usage (non-interactive / agent mode):
    python scripts/register_experiment.py \\
        --id EXP-602 \\
        --creator charles \\
        --name "IBIT Vol Regime" \\
        --ticker IBIT \\
        --status in_development \\
        --notes "VIX-based vol regime switcher for IBIT."

ID allocation rules:
    maximus  →  EXP-000 to EXP-599
    charles  →  EXP-600 and above
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "experiments" / "registry.json"

VALID_CREATORS = ["maximus", "charles"]
VALID_STATUSES = [
    "in_development", "data_collection", "backtesting",
    "validated", "awaiting_deploy", "paper_trading", "deployed",
    "pending", "retired",
]

CREATOR_RANGES = {
    "maximus": (0, 599),
    "charles": (600, 9999),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        print(f"ERROR: registry not found at {REGISTRY_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(registry: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")


def validate_id(exp_id: str, registry: dict) -> str:
    """Validate format, range, and uniqueness. Returns normalized ID."""
    exp_id = exp_id.strip().upper()
    if not re.match(r"^EXP-\d+$", exp_id):
        raise ValueError(f"ID must be EXP-NNN (e.g. EXP-602). Got: '{exp_id}'")
    if exp_id in registry.get("experiments", {}):
        raise ValueError(f"{exp_id} already exists in registry.")
    return exp_id


def validate_creator(creator: str) -> str:
    creator = creator.strip().lower()
    if creator not in VALID_CREATORS:
        raise ValueError(f"creator must be one of {VALID_CREATORS}. Got: '{creator}'")
    return creator


def validate_id_range(exp_id: str, creator: str) -> None:
    num = int(exp_id.split("-")[1])
    lo, hi = CREATOR_RANGES[creator]
    if not (lo <= num <= hi):
        raise ValueError(
            f"ID {exp_id} (number {num}) is out of range for '{creator}'. "
            f"Allowed: EXP-{lo:03d} to EXP-{hi}."
        )


def suggest_next_id(creator: str, registry: dict) -> "str":
    """Suggest the next available ID for a given creator."""
    lo, hi = CREATOR_RANGES[creator]
    used = set()
    for exp_id in registry.get("experiments", {}):
        m = re.match(r"^EXP-(\d+)$", exp_id)
        if m:
            n = int(m.group(1))
            if lo <= n <= hi:
                used.add(n)
    for n in range(lo, hi + 1):
        if n not in used:
            return f"EXP-{n:03d}"
    raise ValueError(f"No available IDs in range {lo}–{hi} for {creator}")


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def prompt(label: str, default: str = "", required: bool = True) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        val = input(f"  {label}{suffix}: ").strip()
        if not val and default:
            return default
        if val:
            return val
        if not required:
            return ""
        print(f"  (required — cannot be blank)")


def prompt_choice(label: str, choices: list[str], default: str = "") -> str:
    options = "/".join(choices)
    suffix = f" [{default}]" if default else ""
    while True:
        val = input(f"  {label} ({options}){suffix}: ").strip().lower()
        if not val and default:
            return default
        if val in choices:
            return val
        print(f"  Must be one of: {choices}")


def interactive_register(registry: dict) -> dict:
    print("\n" + "=" * 55)
    print("  Register New Experiment")
    print("=" * 55)
    print("  Creator attribution is MANDATORY.\n")

    # Creator first — determines valid ID range
    creator = prompt_choice("Creator", VALID_CREATORS)
    try:
        next_id = suggest_next_id(creator, registry)
    except ValueError:
        next_id = ""

    # ID
    while True:
        exp_id_raw = prompt("Experiment ID", default=next_id)
        try:
            exp_id = validate_id(exp_id_raw, registry)
            validate_id_range(exp_id, creator)
            break
        except ValueError as e:
            print(f"  ERROR: {e}")

    name    = prompt("Name (short description)")
    ticker  = prompt("Ticker (e.g. SPY, IBIT)", required=False) or None
    status  = prompt_choice("Initial status", VALID_STATUSES, default="in_development")
    notes   = prompt("Notes (strategy summary)", required=False) or None
    today   = date.today().isoformat()

    return _build_entry(exp_id, creator, name, ticker, status, notes, today)


# ---------------------------------------------------------------------------
# Build the registry entry
# ---------------------------------------------------------------------------

def _build_entry(
    exp_id: str,
    creator: str,
    name: str,
    ticker: Optional[str],
    status: str,
    notes: Optional[str],
    created_date: str,
) -> dict:
    entry: dict = {
        "id":           exp_id,
        "name":         name,
        "created_by":   creator,
        "created_date": created_date,
        "status":       status,
        "ticker":       ticker,
    }
    if status in {"paper_trading", "deployed"}:
        entry["account_id"] = None
        entry["live_since"] = None
        entry["paper_config"] = None
    elif status in {"in_development", "data_collection", "backtesting",
                    "validated", "pending", "awaiting_deploy"}:
        entry["phase"]     = "0 — Not started"
        entry["next_step"] = "Define experiment parameters"
        entry["paper_config"] = None
    entry["backtest_config"] = None
    if notes:
        entry["notes"] = notes
    return entry


# ---------------------------------------------------------------------------
# Confirm + write
# ---------------------------------------------------------------------------

def confirm_and_write(entry: dict, registry: dict) -> None:
    exp_id = entry["id"]
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  New Experiment: {exp_id:<32}│")
    print(f"  │  Name:           {entry['name']:<32}│")
    print(f"  │  Created by:     {entry['created_by']:<32}│")
    print(f"  │  Status:         {entry['status']:<32}│")
    print(f"  │  Ticker:         {str(entry.get('ticker') or '—'):<32}│")
    print(f"  └─────────────────────────────────────────────────┘")

    answer = input("\n  Write to registry.json? [yes/no]: ").strip().lower()
    if answer != "yes":
        print("  Aborted — nothing written.")
        sys.exit(0)

    registry["experiments"][exp_id] = entry
    registry["last_updated"] = date.today().isoformat()
    save_registry(registry)
    print(f"\n  ✅ {exp_id} registered. Next steps:")
    print(f"     1. Fill in backtest_config and paper_config when ready.")
    print(f"     2. Run validate_registry.py to confirm integrity.")
    print(f"     3. Run pre_deploy_check.py before going live.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a new experiment with mandatory creator attribution."
    )
    parser.add_argument("--id",      help="Experiment ID (e.g. EXP-602)")
    parser.add_argument("--creator", choices=VALID_CREATORS,
                        help="Creator: maximus or charles (REQUIRED)")
    parser.add_argument("--name",    help="Short experiment name")
    parser.add_argument("--ticker",  default=None, help="Ticker (e.g. SPY, IBIT)")
    parser.add_argument("--status",  choices=VALID_STATUSES,
                        default="in_development", help="Initial status")
    parser.add_argument("--notes",   default=None, help="Strategy summary")
    parser.add_argument("--date",    default=date.today().isoformat(),
                        help="Created date (YYYY-MM-DD, default: today)")
    args = parser.parse_args()

    registry = load_registry()

    # Non-interactive if all required args supplied
    if args.id and args.creator and args.name:
        try:
            exp_id  = validate_id(args.id, registry)
            creator = validate_creator(args.creator)
            validate_id_range(exp_id, creator)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        entry = _build_entry(
            exp_id, creator, args.name, args.ticker,
            args.status, args.notes, args.date,
        )
        registry["experiments"][exp_id] = entry
        registry["last_updated"] = args.date
        save_registry(registry)
        print(f"✅ {exp_id} registered (created_by: {creator}).")
    else:
        # Interactive mode
        if args.id or args.creator or args.name:
            print("ERROR: provide all of --id, --creator, --name together, or none for interactive mode.",
                  file=sys.stderr)
            sys.exit(1)
        entry = interactive_register(registry)
        confirm_and_write(entry, registry)


if __name__ == "__main__":
    main()
