#!/usr/bin/env python3
"""Preflight config validator — exits 1 on any missing required field.

Usage:
    python scripts/preflight_check.py configs/paper_champion.yaml
    python scripts/preflight_check.py configs/paper_exp401.yaml

Required fields:
    - db_path
    - logging (section with level and file)
    - strategy (section with min_delta and max_delta)
    - risk (section)
    - paper_mode: true
    - experiment_id
"""

import sys
from pathlib import Path

import yaml


def validate(config: dict) -> list:
    """Return list of error strings. Empty list means all checks passed."""
    errors = []

    # Top-level required fields
    if not config.get("db_path"):
        errors.append("Missing required field: db_path")

    if not config.get("experiment_id"):
        errors.append("Missing required field: experiment_id")

    if config.get("paper_mode") is not True:
        errors.append("paper_mode must be true (safety check)")

    # Logging section
    logging_cfg = config.get("logging")
    if not isinstance(logging_cfg, dict):
        errors.append("Missing required section: logging")
    else:
        if not logging_cfg.get("level"):
            errors.append("logging.level is required")
        if not logging_cfg.get("file"):
            errors.append("logging.file is required")

    # Strategy section
    strategy = config.get("strategy")
    if not isinstance(strategy, dict):
        errors.append("Missing required section: strategy")
    else:
        if "min_delta" not in strategy:
            errors.append("strategy.min_delta is required")
        if "max_delta" not in strategy:
            errors.append("strategy.max_delta is required")

    # Risk section
    risk = config.get("risk")
    if not isinstance(risk, dict):
        errors.append("Missing required section: risk")
    else:
        # Validate regime scales if present (EXP-401 blend)
        if "regime_scale_crash" in risk and risk.get("regime_scale_crash") != 0:
            errors.append(
                "risk.regime_scale_crash should be 0 (no trading during crash regime)"
            )

    # Straddle/strangle config validation (optional section)
    ss_config = strategy.get("straddle_strangle") if isinstance(strategy, dict) else None
    if ss_config and isinstance(ss_config, dict) and ss_config.get("enabled"):
        if "profit_target_pct" not in ss_config:
            errors.append("strategy.straddle_strangle.profit_target_pct is required when enabled")
        if "stop_loss_pct" not in ss_config:
            errors.append("strategy.straddle_strangle.stop_loss_pct is required when enabled")
        if "max_risk_pct" not in ss_config:
            errors.append("strategy.straddle_strangle.max_risk_pct is required when enabled")
        if isinstance(risk, dict) and "straddle_strangle_risk_pct" not in risk:
            errors.append("risk.straddle_strangle_risk_pct is required when straddle_strangle is enabled")

    return errors


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
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


if __name__ == "__main__":
    main()
