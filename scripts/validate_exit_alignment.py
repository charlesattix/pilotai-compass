#!/usr/bin/env python3
"""
Validate exit parameter alignment across the full pipeline.

Verifies that per-trade exit params (profit_target_pct, stop_loss_pct) flow
correctly through:

    Signal → signal_to_opportunity() → upsert_trade() → get_trades()
           → trade_dict_to_position() → _check_exit_conditions()

Expected stop-loss multipliers per strategy type:
    - Credit spread (CS):     1.25× (from paper_champion.yaml risk.stop_loss_multiplier)
    - Iron condor (IC):       2.5×  (from strategy.iron_condor.stop_loss_multiplier)
    - Short straddle (SS):    0.50× (strategy default) + 3× credit hard stop

Usage:
    PYTHONPATH=. python scripts/validate_exit_alignment.py [--config configs/paper_champion.yaml]
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.database import get_trades, init_db, upsert_trade
from shared.strategy_adapter import signal_to_opportunity, trade_dict_to_position
from shared.strategy_factory import build_strategy_list
from strategies.base import (
    LegType, MarketSnapshot, Position, PositionAction, Signal,
    TradeLeg, TradeDirection,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_cs_signal(config: Dict) -> Signal:
    """Build a credit spread Signal using config params."""
    strats = build_strategy_list(config)
    cs_strat = next(s for s in strats if "CreditSpread" in s.__class__.__name__)
    params = cs_strat.params

    exp = datetime.now(timezone.utc) + timedelta(days=30)
    return Signal(
        strategy_name=cs_strat.name,
        ticker="SPY",
        direction=TradeDirection.SHORT,
        legs=[
            TradeLeg(LegType.SHORT_PUT, 490.0, exp, entry_price=2.0),
            TradeLeg(LegType.LONG_PUT, 478.0, exp, entry_price=0.5),
        ],
        net_credit=1.50,
        max_loss=10.50,
        max_profit=1.50,
        profit_target_pct=params.get("profit_target_pct", 0.55),
        stop_loss_pct=params.get("stop_loss_multiplier", 1.25),
        score=60.0,
        signal_date=datetime.now(timezone.utc),
        expiration=exp,
        dte=30,
        metadata={"spread_type": "bull_put"},
    )


def _make_ic_signal(config: Dict) -> Signal:
    """Build an iron condor Signal using config params."""
    strats = build_strategy_list(config)
    ic_strat = next((s for s in strats if "IronCondor" in s.__class__.__name__), None)
    if ic_strat is None:
        # Build with IC enabled
        config_copy = dict(config)
        config_copy.setdefault("strategy", {}).setdefault("iron_condor", {})["enabled"] = True
        strats = build_strategy_list(config_copy)
        ic_strat = next(s for s in strats if "IronCondor" in s.__class__.__name__)

    params = ic_strat.params
    exp = datetime.now(timezone.utc) + timedelta(days=35)

    return Signal(
        strategy_name=ic_strat.name,
        ticker="SPY",
        direction=TradeDirection.NEUTRAL,
        legs=[
            TradeLeg(LegType.SHORT_PUT, 485.0, exp, entry_price=1.5),
            TradeLeg(LegType.LONG_PUT, 475.0, exp, entry_price=0.4),
            TradeLeg(LegType.SHORT_CALL, 515.0, exp, entry_price=1.3),
            TradeLeg(LegType.LONG_CALL, 525.0, exp, entry_price=0.3),
        ],
        net_credit=2.10,
        max_loss=7.90,
        max_profit=2.10,
        profit_target_pct=params.get("profit_target_pct", 0.50),
        stop_loss_pct=params.get("stop_loss_multiplier", 2.0),
        score=55.0,
        signal_date=datetime.now(timezone.utc),
        expiration=exp,
        dte=35,
        metadata={
            "spread_type": "iron_condor",
            "put_credit": 1.10,
            "call_credit": 1.00,
        },
    )


def _make_ss_signal(config: Dict) -> Signal:
    """Build a short straddle Signal using config params."""
    strats = build_strategy_list(config)
    ss_strat = next((s for s in strats if "StraddleStrangle" in s.__class__.__name__), None)
    if ss_strat is None:
        config_copy = dict(config)
        config_copy.setdefault("strategy", {}).setdefault("straddle_strangle", {})["enabled"] = True
        strats = build_strategy_list(config_copy)
        ss_strat = next(s for s in strats if "StraddleStrangle" in s.__class__.__name__)

    params = ss_strat.params
    exp = datetime.now(timezone.utc) + timedelta(days=7)

    return Signal(
        strategy_name=ss_strat.name,
        ticker="SPY",
        direction=TradeDirection.SHORT,
        legs=[
            TradeLeg(LegType.SHORT_CALL, 500.0, exp, entry_price=4.0),
            TradeLeg(LegType.SHORT_PUT, 500.0, exp, entry_price=4.5),
        ],
        net_credit=8.50,
        max_loss=25.50,
        max_profit=8.50,
        profit_target_pct=params.get("profit_target_pct", 0.50),
        stop_loss_pct=params.get("stop_loss_pct", 0.50),
        score=40.0,
        signal_date=datetime.now(timezone.utc),
        expiration=exp,
        dte=7,
        metadata={"trade_type": "short_straddle", "spread_type": "short_straddle"},
    )


# ---------------------------------------------------------------------------
# Pipeline validation
# ---------------------------------------------------------------------------

def validate_signal_to_opportunity(signal: Signal, label: str) -> Dict:
    """Step 1: Signal → opportunity dict."""
    opp = signal_to_opportunity(signal, current_price=500.0)
    print(f"  [{label}] signal_to_opportunity():")
    print(f"    profit_target_pct = {opp.get('profit_target_pct')}")
    print(f"    stop_loss_pct     = {opp.get('stop_loss_pct')}")
    print(f"    strategy_name     = {opp.get('strategy_name')}")
    print(f"    type              = {opp.get('type')}")
    return opp


def validate_db_roundtrip(opp: Dict, label: str, db_path: str) -> Dict:
    """Step 2: upsert_trade → get_trades → verify fields survive DB roundtrip."""
    trade = dict(opp)
    trade["id"] = f"test-{label}-{datetime.now().timestamp()}"
    trade["status"] = "open"
    trade["entry_date"] = datetime.now(timezone.utc).isoformat()
    trade["contracts"] = 1

    upsert_trade(trade, source="execution", path=db_path)
    trades = get_trades(status="open", source="execution", path=db_path)
    db_trade = next((t for t in trades if t["id"] == trade["id"]), None)

    if db_trade is None:
        print(f"  [{label}] DB ROUNDTRIP: FAIL — trade not found in DB!")
        return {}

    # Check that per-trade params survived
    pt = db_trade.get("profit_target_pct")
    sl = db_trade.get("stop_loss_pct")
    print(f"  [{label}] DB roundtrip:")
    print(f"    profit_target_pct = {pt}")
    print(f"    stop_loss_pct     = {sl}")
    print(f"    strategy_name     = {db_trade.get('strategy_name')}")
    return db_trade


def validate_trade_to_position(trade: Dict, label: str) -> Position:
    """Step 3: trade dict → Position (for strategy.manage_position())."""
    pos = trade_dict_to_position(trade)
    print(f"  [{label}] trade_dict_to_position():")
    print(f"    profit_target_pct = {pos.profit_target_pct}")
    print(f"    stop_loss_pct     = {pos.stop_loss_pct}")
    print(f"    strategy_name     = {pos.strategy_name}")
    print(f"    legs              = {len(pos.legs)}")
    return pos


def validate_exit_conditions(trade: Dict, label: str, config: Dict) -> Dict[str, Any]:
    """Step 4: Simulate _check_exit_conditions() with per-trade params."""
    from execution.position_monitor import PositionMonitor

    monitor = object.__new__(PositionMonitor)
    monitor.config = config
    monitor.profit_target_pct = float(config.get("risk", {}).get("profit_target", 50))
    monitor.stop_loss_mult = float(config.get("risk", {}).get("stop_loss_multiplier", 3.5))
    monitor.manage_dte = 0  # disable DTE for this test
    monitor._strategy_registry = {}
    monitor._exit_snapshot_cache = None
    monitor._exit_snapshot_ts = None
    monitor.alpaca = MagicMock()

    credit = float(trade.get("credit", 0))
    sl_pct = float(trade.get("stop_loss_pct", monitor.stop_loss_mult))

    # Loss-based threshold: (1 + sl_mult) × credit
    loss_threshold = (1.0 + sl_pct) * credit

    results = {}
    print(f"  [{label}] _check_exit_conditions():")
    print(f"    credit            = {credit}")
    print(f"    stop_loss_pct     = {sl_pct}")
    print(f"    loss_threshold    = {loss_threshold:.4f}")

    # Test profit target hit
    pt_raw = trade.get("profit_target_pct")
    if pt_raw is not None:
        pt_val = float(pt_raw)
        pt_pct = pt_val * 100 if pt_val < 1.0 else pt_val
    else:
        pt_pct = monitor.profit_target_pct

    # Simulate: value that hits profit target
    profit_value = credit * (1 - pt_pct / 100)
    with patch.object(monitor, '_get_spread_value', return_value=profit_value - 0.01):
        result = monitor._check_exit_conditions(trade, {})
    results["profit_target_fires"] = result == "profit_target"
    print(f"    profit_target fires at value={profit_value:.4f}? {results['profit_target_fires']}")

    # Simulate: value that hits stop loss
    with patch.object(monitor, '_get_spread_value', return_value=loss_threshold + 0.01):
        result = monitor._check_exit_conditions(trade, {})
    results["stop_loss_fires"] = result == "stop_loss"
    print(f"    stop_loss fires at value={loss_threshold:.4f}? {results['stop_loss_fires']}")

    # Simulate: value between PT and SL (should HOLD)
    hold_value = credit * 0.80  # slight profit
    with patch.object(monitor, '_get_spread_value', return_value=hold_value):
        result = monitor._check_exit_conditions(trade, {})
    results["hold_between"] = result is None
    print(f"    holds at value={hold_value:.4f}? {results['hold_between']}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate exit parameter alignment")
    parser.add_argument("--config", type=str, default="configs/paper_champion.yaml",
                        help="Config file path")
    parser.add_argument("--output", type=str, default="output/exit_alignment.json",
                        help="Output JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Enable all strategies for testing
    config.setdefault("strategy", {}).setdefault("iron_condor", {})["enabled"] = True
    config.setdefault("strategy", {}).setdefault("straddle_strangle", {})["enabled"] = True

    print(f"\n{'='*70}")
    print("EXIT PARAMETER ALIGNMENT VALIDATION")
    print(f"{'='*70}")
    print(f"Config: {args.config}")

    # Expected values from config
    cs_sl = float(config.get("risk", {}).get("stop_loss_multiplier", 1.25))
    ic_sl = float(config.get("strategy", {}).get("iron_condor", {}).get("stop_loss_multiplier", 2.5))
    ss_sl = 0.50  # strategy default
    cs_pt = float(config.get("risk", {}).get("profit_target", 55)) / 100.0
    ic_pt = float(config.get("strategy", {}).get("iron_condor", {}).get("profit_target_pct", 0.30))
    ss_pt = 0.50  # strategy default

    print(f"\nExpected exit params:")
    print(f"  CS: profit_target={cs_pt}, stop_loss={cs_sl}x")
    print(f"  IC: profit_target={ic_pt}, stop_loss={ic_sl}x")
    print(f"  SS: profit_target={ss_pt}, stop_loss={ss_sl}x + 3x hard stop")

    # Create temp DB for testing
    db_dir = tempfile.mkdtemp()
    db_path = os.path.join(db_dir, "test_exit_validation.db")
    init_db(db_path)

    all_results = {}
    all_pass = True

    # ---------------------------------------------------------------------------
    # Credit Spread
    # ---------------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("CREDIT SPREAD (CS)")
    print(f"{'─'*70}")

    cs_signal = _make_cs_signal(config)
    cs_opp = validate_signal_to_opportunity(cs_signal, "CS")
    cs_db = validate_db_roundtrip(cs_opp, "CS", db_path)
    if cs_db:
        cs_pos = validate_trade_to_position(cs_db, "CS")
        cs_exit = validate_exit_conditions(cs_db, "CS", config)

        # Verify expected values
        cs_checks = {
            "signal_pt_matches": abs(cs_signal.profit_target_pct - cs_pt) < 0.01,
            "signal_sl_matches": abs(cs_signal.stop_loss_pct - cs_sl) < 0.01,
            "opp_pt_matches": abs(float(cs_opp.get("profit_target_pct", 0)) - cs_pt) < 0.01,
            "opp_sl_matches": abs(float(cs_opp.get("stop_loss_pct", 0)) - cs_sl) < 0.01,
            "position_pt_matches": abs(cs_pos.profit_target_pct - cs_pt) < 0.01,
            "position_sl_matches": abs(cs_pos.stop_loss_pct - cs_sl) < 0.01,
            "profit_target_fires": cs_exit.get("profit_target_fires", False),
            "stop_loss_fires": cs_exit.get("stop_loss_fires", False),
            "hold_between": cs_exit.get("hold_between", False),
        }
        all_results["credit_spread"] = cs_checks
        cs_pass = all(cs_checks.values())
        all_pass = all_pass and cs_pass
        print(f"\n  CS VERDICT: {'PASS' if cs_pass else 'FAIL'}")
        for k, v in cs_checks.items():
            if not v:
                print(f"    FAILED: {k}")

    # ---------------------------------------------------------------------------
    # Iron Condor
    # ---------------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("IRON CONDOR (IC)")
    print(f"{'─'*70}")

    ic_signal = _make_ic_signal(config)
    ic_opp = validate_signal_to_opportunity(ic_signal, "IC")
    ic_db = validate_db_roundtrip(ic_opp, "IC", db_path)
    if ic_db:
        ic_pos = validate_trade_to_position(ic_db, "IC")
        ic_exit = validate_exit_conditions(ic_db, "IC", config)

        ic_checks = {
            "signal_pt_matches": abs(ic_signal.profit_target_pct - ic_pt) < 0.01,
            "signal_sl_matches": abs(ic_signal.stop_loss_pct - ic_sl) < 0.01,
            "opp_pt_matches": abs(float(ic_opp.get("profit_target_pct", 0)) - ic_pt) < 0.01,
            "opp_sl_matches": abs(float(ic_opp.get("stop_loss_pct", 0)) - ic_sl) < 0.01,
            "position_pt_matches": abs(ic_pos.profit_target_pct - ic_pt) < 0.01,
            "position_sl_matches": abs(ic_pos.stop_loss_pct - ic_sl) < 0.01,
            "profit_target_fires": ic_exit.get("profit_target_fires", False),
            "stop_loss_fires": ic_exit.get("stop_loss_fires", False),
            "hold_between": ic_exit.get("hold_between", False),
        }
        all_results["iron_condor"] = ic_checks
        ic_pass = all(ic_checks.values())
        all_pass = all_pass and ic_pass
        print(f"\n  IC VERDICT: {'PASS' if ic_pass else 'FAIL'}")
        for k, v in ic_checks.items():
            if not v:
                print(f"    FAILED: {k}")

    # ---------------------------------------------------------------------------
    # Short Straddle
    # ---------------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("SHORT STRADDLE (SS)")
    print(f"{'─'*70}")

    ss_signal = _make_ss_signal(config)
    ss_opp = validate_signal_to_opportunity(ss_signal, "SS")
    ss_db = validate_db_roundtrip(ss_opp, "SS", db_path)
    if ss_db:
        ss_pos = validate_trade_to_position(ss_db, "SS")
        ss_exit = validate_exit_conditions(ss_db, "SS", config)

        # Verify 3x hard stop for straddles via strategy.manage_position()
        from strategies.straddle_strangle import StraddleStrangleStrategy
        ss_strat = StraddleStrangleStrategy({})
        exp = datetime.now(timezone.utc) + timedelta(days=7)
        hard_stop_pos = Position(
            id="ss-3x-test",
            strategy_name="StraddleStrangleStrategy",
            ticker="SPY",
            direction=TradeDirection.SHORT,
            legs=[
                TradeLeg(LegType.SHORT_CALL, 500.0, exp),
                TradeLeg(LegType.SHORT_PUT, 500.0, exp),
            ],
            contracts=1,
            net_credit=5.0,
            max_loss_per_unit=15.0,
            max_profit_per_unit=5.0,
            profit_target_pct=0.50,
            stop_loss_pct=0.50,
        )
        snap = MarketSnapshot(
            date=datetime.now(timezone.utc),
            price_data={},
            prices={"SPY": 520.0},
            vix=30.0,
            iv_rank={"SPY": 50.0},
            realized_vol={"SPY": 0.40},
            rsi={"SPY": 70.0},
        )
        # Mock spread value so cost_to_close = 16.0 (> 3 × 5.0 = 15.0)
        from strategies.pricing import estimate_spread_value
        with patch("strategies.straddle_strangle.estimate_spread_value", return_value=-16.0):
            hard_stop_result = ss_strat.manage_position(hard_stop_pos, snap)
        hard_stop_fires = hard_stop_result == PositionAction.CLOSE_STOP

        ss_checks = {
            "signal_pt_matches": abs(ss_signal.profit_target_pct - ss_pt) < 0.01,
            "signal_sl_matches": abs(ss_signal.stop_loss_pct - ss_sl) < 0.01,
            "opp_pt_matches": abs(float(ss_opp.get("profit_target_pct", 0)) - ss_pt) < 0.01,
            "opp_sl_matches": abs(float(ss_opp.get("stop_loss_pct", 0)) - ss_sl) < 0.01,
            "position_pt_matches": abs(ss_pos.profit_target_pct - ss_pt) < 0.01,
            "position_sl_matches": abs(ss_pos.stop_loss_pct - ss_sl) < 0.01,
            "profit_target_fires": ss_exit.get("profit_target_fires", False),
            "stop_loss_fires": ss_exit.get("stop_loss_fires", False),
            "hold_between": ss_exit.get("hold_between", False),
            "3x_hard_stop_fires": hard_stop_fires,
        }
        all_results["short_straddle"] = ss_checks
        ss_pass = all(ss_checks.values())
        all_pass = all_pass and ss_pass
        print(f"  3x hard stop fires? {hard_stop_fires}")
        print(f"\n  SS VERDICT: {'PASS' if ss_pass else 'FAIL'}")
        for k, v in ss_checks.items():
            if not v:
                print(f"    FAILED: {k}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total_checks = sum(len(v) for v in all_results.values())
    passed_checks = sum(sum(1 for v in checks.values() if v) for checks in all_results.values())

    print(f"Total checks:  {total_checks}")
    print(f"Passed:        {passed_checks}")
    print(f"Failed:        {total_checks - passed_checks}")

    for strat, checks in all_results.items():
        strat_pass = all(checks.values())
        print(f"  {strat:25s} {'PASS' if strat_pass else 'FAIL'}")

    print(f"\n{'='*70}")
    status = "PASS" if all_pass else "FAIL"
    print(f"VERDICT: {status}")
    print(f"{'='*70}\n")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "all_pass": all_pass,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")

    # Cleanup temp DB
    try:
        os.remove(db_path)
        os.rmdir(db_dir)
    except Exception:
        pass

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
