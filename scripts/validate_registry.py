#!/usr/bin/env python3
"""Validate experiments/registry.json for attribution and schema integrity.

Checks every experiment has required fields, valid created_by, valid status,
and that ID ranges are respected (EXP-000–599 = maximus, EXP-600+ = charles).

Usage:
    python scripts/validate_registry.py           # exit 0 if OK, 1 if errors
    python scripts/validate_registry.py --strict  # also fail on missing optional fields

Run this before any commit that touches registry.json, or in CI.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "experiments" / "registry.json"

VALID_CREATORS = {"maximus", "charles"}
VALID_STATUSES = {
    "data_collection", "backtesting", "validated",
    "paper_trading", "deployed", "in_development",
    "retired", "pending", "awaiting_deploy",
}

REQUIRED_FIELDS = ["id", "name", "created_by", "created_date", "status"]

# ID number ranges per creator
CREATOR_RANGES = {
    "maximus": (0, 599),
    "charles": (600, 9999),
}


def _exp_number(exp_id: str) -> Optional[int]:
    """Extract the numeric part of EXP-NNN. Returns None if non-numeric."""
    m = re.match(r"^EXP-(\d+)$", exp_id, re.IGNORECASE)
    return int(m.group(1)) if m else None


def validate(registry: dict, strict: bool = False) -> "list[str]":
    errors: list[str] = []
    experiments = registry.get("experiments", {})

    if not experiments:
        errors.append("No experiments found in registry.")
        return errors

    seen_ids: set[str] = set()

    for exp_id, exp in experiments.items():
        prefix = f"[{exp_id}]"

        # Duplicate ID check
        if exp_id in seen_ids:
            errors.append(f"{prefix} Duplicate ID.")
        seen_ids.add(exp_id)

        # ID key must match id field
        if exp.get("id") != exp_id:
            errors.append(f"{prefix} id field '{exp.get('id')}' does not match registry key '{exp_id}'.")

        # Required fields
        for field in REQUIRED_FIELDS:
            if not exp.get(field):
                errors.append(f"{prefix} Missing required field: '{field}'.")

        # created_by validation (the core attribution check)
        creator = exp.get("created_by", "")
        if creator not in VALID_CREATORS:
            errors.append(
                f"{prefix} Invalid created_by='{creator}'. "
                f"Must be one of: {sorted(VALID_CREATORS)}."
            )

        # Status validation
        status = exp.get("status", "")
        if status not in VALID_STATUSES:
            errors.append(
                f"{prefix} Invalid status='{status}'. "
                f"Must be one of: {sorted(VALID_STATUSES)}."
            )

        # ID range vs creator (only for standard EXP-NNN IDs)
        num = _exp_number(exp_id)
        if num is not None and creator in CREATOR_RANGES:
            lo, hi = CREATOR_RANGES[creator]
            if not (lo <= num <= hi):
                errors.append(
                    f"{prefix} ID number {num} is out of range for '{creator}' "
                    f"(allowed: {lo}–{hi})."
                )

        # Strict: live experiments must have account_id and live_since
        if strict and status in {"paper_trading", "deployed"}:
            if not exp.get("account_id"):
                errors.append(f"{prefix} Live experiment missing 'account_id'.")
            if not exp.get("live_since"):
                errors.append(f"{prefix} Live experiment missing 'live_since'.")

    # Schema-level checks
    if not registry.get("schema_version"):
        errors.append("Registry missing 'schema_version'.")
    if not registry.get("last_updated"):
        errors.append("Registry missing 'last_updated'.")

    return errors


def main() -> None:
    strict = "--strict" in sys.argv[1:]

    if not REGISTRY_PATH.exists():
        print(f"ERROR: registry not found at {REGISTRY_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    errors = validate(registry, strict=strict)

    exp_count = len(registry.get("experiments", {}))
    creator_counts: dict[str, int] = {}
    for exp in registry.get("experiments", {}).values():
        c = exp.get("created_by", "unknown")
        creator_counts[c] = creator_counts.get(c, 0) + 1

    if errors:
        print(f"registry.json VALIDATION FAILED ({len(errors)} error(s)):\n")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        breakdown = "  |  ".join(f"{c}: {n}" for c, n in sorted(creator_counts.items()))
        print(f"registry.json OK — {exp_count} experiments  ({breakdown})")
        sys.exit(0)


if __name__ == "__main__":
    main()
