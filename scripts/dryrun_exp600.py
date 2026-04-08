#!/usr/bin/env python3
"""
dryrun_exp600.py — EXP-600 deployment dry-run verification.

Verifies:
  1. Config loads and all required fields are present
  2. Preflight validator passes (paper_mode, db_path, strategy, risk, logging)
  3. CreditSpreadStrategy instantiates with EXP-600 params
  4. Key backtest champion params match config (OTM=10%, DTE=14, width=$5,
     PT=30%, SL=2.5x, risk=15%)
  5. Regime mode is MA (not combo) — MA50 direction gate
  6. Iron condor and straddle/strangle are disabled

Does NOT connect to Alpaca or Polygon. Runs entirely offline.

Usage:
    python3 scripts/dryrun_exp600.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PASS = "\033[32m✓ PASS\033[0m"
FAIL = "\033[31m✗ FAIL\033[0m"

_failures = []


def check(label: str, condition: bool, detail: str = ""):
    tag = PASS if condition else FAIL
    print(f"  {tag}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        _failures.append(label)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Config loads and parses
# ─────────────────────────────────────────────────────────────────────────────

def test_config_loads():
    print("\n[1] paper_exp600.yaml — config file loads")
    try:
        from utils import load_config
        cfg = load_config("configs/paper_exp600.yaml")
        check("config loaded", bool(cfg))
        check("experiment_id = EXP-600", cfg.get("experiment_id") == "EXP-600")
        check("paper_mode = true", cfg.get("paper_mode") is True)
        check("db_path contains exp600", "exp600" in cfg.get("db_path", ""))
        check("ticker = IBIT", cfg.get("tickers", []) == ["IBIT"])
        return cfg
    except Exception as exc:
        check("config loads", False, str(exc))
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Preflight validator passes
# ─────────────────────────────────────────────────────────────────────────────

def test_preflight(cfg: dict):
    print("\n[2] Preflight validator")
    try:
        from scripts.preflight_check import validate
        errors = validate(cfg)
        check("no preflight errors", len(errors) == 0, "; ".join(errors) if errors else "")
    except Exception as exc:
        check("preflight_check import", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Champion params match config
# ─────────────────────────────────────────────────────────────────────────────

def test_champion_params(cfg: dict):
    print("\n[3] Champion backtest params — config matches sweep config #14")
    strategy = cfg.get("strategy", {})
    risk = cfg.get("risk", {})

    check("otm_pct = 0.10",          abs(strategy.get("otm_pct", 0) - 0.10) < 0.001,
          f"got {strategy.get('otm_pct')}")
    check("target_dte = 14",         strategy.get("target_dte") == 14,
          f"got {strategy.get('target_dte')}")
    check("spread_width = 5",        strategy.get("spread_width") == 5,
          f"got {strategy.get('spread_width')}")
    check("profit_target = 30",      risk.get("profit_target") == 30,
          f"got {risk.get('profit_target')}")
    check("stop_loss_multiplier = 2.5", abs(risk.get("stop_loss_multiplier", 0) - 2.5) < 0.001,
          f"got {risk.get('stop_loss_multiplier')}")
    check("max_risk_per_trade = 15", abs(risk.get("max_risk_per_trade", 0) - 15.0) < 0.001,
          f"got {risk.get('max_risk_per_trade')}")
    check("max_positions = 5",       risk.get("max_positions") == 5,
          f"got {risk.get('max_positions')}")
    check("sizing_mode = kelly",     risk.get("sizing_mode") == "kelly",
          f"got {risk.get('sizing_mode')}")
    check("kelly_fraction = 1.0",    abs(risk.get("kelly_fraction", 0) - 1.0) < 0.001,
          f"got {risk.get('kelly_fraction')}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Regime config — MA mode, MA50
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_config(cfg: dict):
    print("\n[4] Regime config — MA50 adaptive direction gate")
    strategy = cfg.get("strategy", {})
    regime_config = strategy.get("regime_config", {})
    tech = strategy.get("technical", {})

    check("regime_mode = ma",        strategy.get("regime_mode") == "ma",
          f"got {strategy.get('regime_mode')}")
    check("technical.fast_ma = 50",  tech.get("fast_ma") == 50,
          f"got {tech.get('fast_ma')}")
    check("use_trend_filter = true", tech.get("use_trend_filter") is True)
    check("direction = both",        strategy.get("direction") == "both",
          f"got {strategy.get('direction')}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Iron condor and straddle/strangle disabled
# ─────────────────────────────────────────────────────────────────────────────

def test_overlays_disabled(cfg: dict):
    print("\n[5] Overlays disabled — IC and S/S off for IBIT")
    strategy = cfg.get("strategy", {})
    ic = strategy.get("iron_condor", {})
    ss = strategy.get("straddle_strangle", {})

    check("iron_condor.enabled = false",      ic.get("enabled") is False,
          f"got {ic.get('enabled')}")
    check("straddle_strangle.enabled = false", ss.get("enabled") is False,
          f"got {ss.get('enabled')}")
    check("ml_enhanced absent or disabled",
          not cfg.get("strategy", {}).get("ml_enhanced", {}).get("enabled", False))


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: CreditSpreadStrategy instantiates with EXP-600 params
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy_instantiates():
    print("\n[6] CreditSpreadStrategy — instantiation with EXP-600 params")
    try:
        from strategies.credit_spread import CreditSpreadStrategy
    except ImportError as exc:
        check("CreditSpreadStrategy import", False, str(exc))
        return

    params = {
        "direction": "both",
        "target_dte": 14,
        "min_dte": 10,
        "max_dte": 21,
        "otm_pct": 0.10,
        "spread_width": 5,
        "profit_target_pct": 0.30,
        "stop_loss_multiplier": 2.5,
        "max_risk_pct": 0.15,
        "trend_ma_period": 50,
        "momentum_filter_pct": 2.0,
        "manage_dte": 0,
        "scan_weekday": "any",
        "credit_fraction": 0.35,
    }
    try:
        cs = CreditSpreadStrategy(params)
        check("CreditSpreadStrategy instantiated", True)
        check("strategy.name is set", bool(cs.name))
    except Exception as exc:
        check("CreditSpreadStrategy instantiated", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Log and DB directories are writable
# ─────────────────────────────────────────────────────────────────────────────

def test_paths():
    print("\n[7] Paths — log dir and DB dir writable")
    log_path = Path("/Users/charlesbot/logs")
    db_dir = ROOT / "data" / "exp600"

    check("~/logs dir exists",      log_path.exists(), str(log_path))
    check("~/logs dir writable",    log_path.exists() and log_path.stat().st_mode & 0o200,
          str(log_path))

    db_dir.mkdir(parents=True, exist_ok=True)
    check("data/exp600 dir created", db_dir.exists(), str(db_dir))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EXP-600 IBIT Adaptive — Deployment Dry-Run")
    print("=" * 60)

    cfg = test_config_loads()
    if cfg:
        test_preflight(cfg)
        test_champion_params(cfg)
        test_regime_config(cfg)
        test_overlays_disabled(cfg)
    test_strategy_instantiates()
    test_paths()

    print()
    print("=" * 60)
    if _failures:
        print(f"\033[31m  FAILED: {len(_failures)} check(s)\033[0m")
        for f in _failures:
            print(f"    ✗ {f}")
        sys.exit(1)
    else:
        print("\033[32m  ALL CHECKS PASSED — EXP-600 ready for deployment\033[0m")
    print("=" * 60)


if __name__ == "__main__":
    main()
