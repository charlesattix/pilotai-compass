#!/usr/bin/env python3
"""Pre-deploy gate: registry lookup + preflight validation + Carlos approval.

Usage:
    python scripts/pre_deploy_check.py EXP-503 configs/paper_exp503.yaml
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Allow import from scripts/ directory
sys.path.insert(0, str(Path(__file__).parent))
from preflight_check import validate


REGISTRY_PATH = Path(__file__).parent.parent / "experiments" / "registry.json"
APPROVALS_LOG = Path(__file__).parent.parent / "experiments" / "approvals.log"

CHECKLIST = [
    "Backtest avg return > 20%",
    "Max drawdown < 40% in all years",
    "Walk-forward 3/3 profitable",
    "Monte Carlo P50 passes threshold",
    "paper_mode: true in config",
    "Experiment registered in registry.json",
    "MASTERPLAN.md entry updated",
]


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <EXP-ID> <config.yaml>")
        sys.exit(1)

    exp_id = sys.argv[1]
    config_path = Path(sys.argv[2])

    # 1. Registry lookup
    if not REGISTRY_PATH.exists():
        print(f"ERROR: registry.json not found at {REGISTRY_PATH}")
        sys.exit(1)

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    exp = registry.get("experiments", {}).get(exp_id)
    if exp is None:
        print(f"{exp_id} not found in registry.json. Register it first.")
        sys.exit(1)

    # 2. Attribution display
    width = 47
    border = "─" * width
    print(f"┌{border}┐")
    print(f"│  PRE-DEPLOY CHECKLIST: {exp_id:<{width - 25}}│")
    print(f"│  Name:       {exp['name']:<{width - 15}}│")
    print(f"│  Created by: {exp['created_by']:<{width - 15}}│")
    print(f"│  Status:     {exp['status']:<{width - 15}}│")
    print(f"│  Config:     {str(config_path):<{width - 15}}│")
    print(f"└{border}┘")
    print()

    # 3. Preflight config validation
    if not config_path.exists():
        print(f"FAIL: config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    errors = validate(config)
    if errors:
        print(f"PREFLIGHT FAILED for {config_path}:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"PREFLIGHT OK: {config_path}")
    print()

    # 4. Checklist display
    print("Pre-Deploy Checklist:")
    for item in CHECKLIST:
        print(f"[ ] {item}")
    print()

    # 5. Carlos approval prompt
    answer = input(
        f"Has Carlos (the human) reviewed and approved {exp_id} for paper trading? [yes/no]: "
    ).strip().lower()

    if answer != "yes":
        print("Deploy aborted — Carlos approval required.")
        sys.exit(1)

    # 6. Approval log
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    log_entry = f"{timestamp} | {exp_id} | {exp['created_by']} | approved | {config_path}\n"
    with open(APPROVALS_LOG, "a") as f:
        f.write(log_entry)

    # 7. Success
    print(f"\n✅ Pre-deploy check PASSED for {exp_id}. Safe to run deploy.sh.")


if __name__ == "__main__":
    main()
