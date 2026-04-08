#!/usr/bin/env python3
"""
dryrun_exp503.py — EXP-503 deployment dry-run verification.

Verifies:
  1. ML models load correctly (RegimeModelRouter)
  2. Regime classification produces correct multipliers
  3. MLEnhancedStrategy generates/suppresses signals correctly per regime
  4. Strategy factory wires everything from config

Does NOT connect to Alpaca or Polygon. Runs entirely offline.

Usage:
    python3 scripts/dryrun_exp503.py
"""
from __future__ import annotations

import sys
import logging
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PASS = "\033[32m✓ PASS\033[0m"
FAIL = "\033[31m✗ FAIL\033[0m"
INFO = "\033[34mℹ\033[0m"

_failures = []


def check(label: str, condition: bool, detail: str = ""):
    tag = PASS if condition else FAIL
    print(f"  {tag}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        _failures.append(label)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: RegimeModelRouter loads
# ─────────────────────────────────────────────────────────────────────────────

def test_router_loads():
    print("\n[1] RegimeModelRouter — instantiation and model loading")
    try:
        from ml.regime_model_router import RegimeModelRouter
        router = RegimeModelRouter({
            "min_mult": 0.10,
            "max_mult": 1.50,
            "neutral_mult": 1.00,
            "low_vol_mult": 1.20,
            "crash_mult": 0.00,
            "use_signal_model": True,
        })
        check("RegimeModelRouter instantiated", True)
        return router
    except Exception as exc:
        check("RegimeModelRouter instantiated", False, str(exc))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Regime → multiplier table
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_multipliers(router):
    print("\n[2] Regime multiplier table")
    expected = {
        "bull":     1.50,
        "neutral":  1.00,
        "low_vol":  1.20,
        "high_vol": 0.10,
        "bear":     0.10,
        "crash":    0.00,
        None:       1.00,  # None → neutral
    }
    for regime, exp_mult in expected.items():
        got = router.get_multiplier(regime)
        check(
            f"regime={str(regime):10s} → mult={exp_mult:.2f}",
            abs(got - exp_mult) < 0.001,
            f"got {got:.2f}",
        )

    # Defensive detection
    check("bull  is NOT defensive", not router.is_defensive("bull"))
    check("bear  IS  defensive",         router.is_defensive("bear"))
    check("crash IS  defensive",         router.is_defensive("crash"))
    check("high_vol IS defensive",       router.is_defensive("high_vol"))
    check("neutral is NOT defensive",    not router.is_defensive("neutral"))


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: MLEnhancedStrategy signal gate
# ─────────────────────────────────────────────────────────────────────────────

def test_signal_gate():
    print("\n[3] MLEnhancedStrategy signal gate (no real options data needed)")
    try:
        from strategies.credit_spread import CreditSpreadStrategy
        from strategies.ml_enhanced_strategy import MLEnhancedStrategy
        from strategies.base import (
            Signal, MarketSnapshot, TradeDirection, TradeLeg, LegType
        )
    except ImportError as exc:
        check("imports", False, str(exc))
        return

    def _make_snapshot(regime: Optional[str]) -> MarketSnapshot:
        import pandas as pd
        from datetime import datetime
        return MarketSnapshot(
            date=datetime.today(),
            price_data={"SPY": pd.DataFrame()},
            prices={"SPY": 500.0},
            regime=regime,
            vix=20.0,
            iv_rank={"SPY": 35.0},
            realized_vol={"SPY": 0.15},
        )

    def _make_stub_signal(score: float = 50.0) -> Signal:
        leg = TradeLeg(
            leg_type=LegType.SHORT_PUT,
            strike=490.0,
            expiration=date(date.today().year, date.today().month + 1, 20),
        )
        return Signal(
            strategy_name="credit_spread",
            ticker="SPY",
            direction=TradeDirection.SHORT,
            legs=[leg],
            net_credit=1.50,
            max_loss=10.50,
            max_profit=1.50,
            score=score,
        )

    # Stub wrapped strategy that always returns one signal
    class _StubCS:
        name = "credit_spread"
        def generate_signals(self, snapshot, portfolio_state=None):
            return [_make_stub_signal(50.0)]

    ml_params = {
        "min_mult": 0.10, "max_mult": 1.50, "neutral_mult": 1.00,
        "low_vol_mult": 1.20, "crash_mult": 0.00, "regime_gate": True,
        "use_signal_model": False, "ml_blend_weight": 0.0,
        "min_score_threshold": 30.0,
        # credit spread defaults so super().__init__ doesn't choke
        "direction": "both", "target_dte": 15, "min_dte": 15,
        "otm_pct": 0.02, "spread_width": 12.0, "credit_fraction": 0.35,
        "profit_target_pct": 0.55, "stop_loss_multiplier": 1.25,
        "momentum_filter_pct": 2.0, "trend_ma_period": 80, "max_risk_pct": 0.12,
        "scan_weekday": "any", "manage_dte": 0,
    }

    try:
        ml = MLEnhancedStrategy(ml_params, _StubCS())
        check("MLEnhancedStrategy instantiated", True)
    except Exception as exc:
        check("MLEnhancedStrategy instantiated", False, str(exc))
        return

    test_cases = [
        ("bull",     True,  1.50),   # signals pass, score × 1.50
        ("neutral",  True,  1.00),   # signals pass, score × 1.00
        ("low_vol",  True,  1.20),   # signals pass, score × 1.20
        ("high_vol", False, 0.10),   # regime_gate → suppressed
        ("bear",     False, 0.10),   # regime_gate → suppressed
        ("crash",    False, 0.00),   # crash → suppressed
    ]

    for regime, expect_signals, mult in test_cases:
        snap = _make_snapshot(regime)
        try:
            sigs = ml.generate_signals(snap)
            got_signals = len(sigs) > 0
            check(
                f"regime={regime:10s} signals={'yes' if expect_signals else 'no '}",
                got_signals == expect_signals,
                f"got {len(sigs)} signal(s)",
            )
            if expect_signals and sigs:
                expected_score = round(50.0 * mult, 1)
                got_score = sigs[0].score
                check(
                    f"regime={regime:10s} score={expected_score:.1f}",
                    abs(got_score - expected_score) < 0.5,
                    f"got {got_score:.1f}",
                )
                check(
                    f"regime={regime:10s} metadata has ml_v2_risk_mult",
                    "ml_v2_risk_mult" in (sigs[0].metadata or {}),
                )
        except Exception as exc:
            check(f"regime={regime} signal generation", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Strategy factory wires MLEnhancedStrategy from config
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy_factory():
    print("\n[4] Strategy factory — MLEnhancedStrategy from config")
    try:
        from shared.strategy_factory import build_strategy_list
        from strategies.ml_enhanced_strategy import MLEnhancedStrategy
    except ImportError as exc:
        check("imports", False, str(exc))
        return

    config = {
        "strategy": {
            "direction": "both",
            "min_dte": 15, "max_dte": 25, "target_dte": 15,
            "otm_pct": 0.02, "spread_width": 12,
            "technical": {"slow_ma": 80},
            "iron_condor": {"enabled": False},
            "straddle_strangle": {"enabled": False},
            "ml_enhanced": {
                "enabled": True,
                "min_mult": 0.10,
                "max_mult": 1.50,
                "regime_gate": True,
                "use_signal_model": False,
            },
        },
        "risk": {
            "max_risk_per_trade": 12.0,
            "profit_target": 55,
            "stop_loss_multiplier": 1.25,
        },
    }

    try:
        strategies = build_strategy_list(config)
        check("build_strategy_list returns list", isinstance(strategies, list))
        check("exactly one strategy returned", len(strategies) == 1, f"got {len(strategies)}")
        if strategies:
            s = strategies[0]
            check(
                "strategy is MLEnhancedStrategy",
                isinstance(s, MLEnhancedStrategy),
                type(s).__name__,
            )
            check("strategy name is ml_enhanced_v2", s.name == "ml_enhanced_v2")
    except Exception as exc:
        check("build_strategy_list", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Config file loads and parses
# ─────────────────────────────────────────────────────────────────────────────

def test_config_loads():
    print("\n[5] paper_exp503.yaml — config file loads")
    try:
        from utils import load_config
        cfg = load_config("configs/paper_exp503.yaml")
        check("config loaded", bool(cfg))
        check("experiment_id = EXP-503", cfg.get("experiment_id") == "EXP-503")
        check("paper_mode = true", cfg.get("paper_mode") is True)
        check("db_path contains exp503", "exp503" in cfg.get("db_path", ""))
        ml_cfg = cfg.get("strategy", {}).get("ml_enhanced", {})
        check("ml_enhanced.enabled = true", ml_cfg.get("enabled") is True)
        check("ml_enhanced.min_mult = 0.10", abs(ml_cfg.get("min_mult", 0) - 0.10) < 0.001)
        check("ml_enhanced.max_mult = 1.50", abs(ml_cfg.get("max_mult", 0) - 1.50) < 0.001)
        check("ml_enhanced.regime_gate = true", ml_cfg.get("regime_gate") is True)
    except Exception as exc:
        check("config loads", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EXP-503 ML V2 Aggressive — Deployment Dry-Run")
    print("=" * 60)

    router = test_router_loads()
    if router:
        test_regime_multipliers(router)
    test_signal_gate()
    test_strategy_factory()
    test_config_loads()

    print()
    print("=" * 60)
    if _failures:
        print(f"\033[31m  FAILED: {len(_failures)} check(s)\033[0m")
        for f in _failures:
            print(f"    ✗ {f}")
        sys.exit(1)
    else:
        print("\033[32m  ALL CHECKS PASSED — EXP-503 ready for deployment\033[0m")
    print("=" * 60)


if __name__ == "__main__":
    main()
