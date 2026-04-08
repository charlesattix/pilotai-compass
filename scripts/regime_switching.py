#!/usr/bin/env python3
"""
regime_switching.py — Phase 4: Regime Switching (Dynamic Allocation)

Optimizes per-regime risk_budget scaling for Credit Spread + Straddle/Strangle.
Uses staged grid search:
  Phase A: Train on 2020-2022 (optimize CS regime scales, then SS)
  Phase B: Validate top candidates on 2023-2025
  Phase C: Full 6-year confirmation + baseline comparison

Base risk: CS=12%, SS=3% (from Phase 3 best blend).

Usage:
    PYTHONPATH=. python3 scripts/regime_switching.py
"""

import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.portfolio_backtester import PortfolioBacktester
from scripts.portfolio_blend import get_strategy_params
from strategies import STRATEGY_REGISTRY

TICKERS = ["SPY"]
STARTING_CAPITAL = 100_000
TRAIN_YEARS = [2020, 2021, 2022]
VALID_YEARS = [2023, 2024, 2025]
ALL_YEARS = TRAIN_YEARS + VALID_YEARS

CS_BASE_RISK = 0.12
SS_BASE_RISK = 0.03

# Regime scale grids
CS_REGIME_GRID = {
    "regime_scale_bull": [0.8, 1.0, 1.2, 1.5],
    "regime_scale_bear": [0.3, 0.5, 0.8],
    "regime_scale_high_vol": [0.3, 0.5, 0.8],
    "regime_scale_low_vol": [0.8, 1.0, 1.2, 1.5],
    # crash is always 0 for CS (fixed)
}

SS_REGIME_GRID = {
    "regime_scale_bull": [0.5, 1.0, 1.5],
    "regime_scale_bear": [0.5, 1.0, 1.5],
    "regime_scale_high_vol": [1.0, 1.5, 2.0, 2.5],
    "regime_scale_low_vol": [0.5, 1.0, 1.5],
    # crash is fixed at 0.5 for SS
}


def _make_grid(regime_grid: Dict[str, List[float]]) -> List[Dict[str, float]]:
    """Expand a regime grid into a list of param dicts."""
    keys = sorted(regime_grid.keys())
    values = [regime_grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def run_blend_years(
    cs_params: Dict, ss_params: Dict, years: List[int],
    max_pos: int = 10, max_per_strat: int = 5,
) -> Dict:
    """Run CS+SS blend across specified years. Returns summary stats."""
    cs_cls = STRATEGY_REGISTRY["credit_spread"]
    ss_cls = STRATEGY_REGISTRY["straddle_strangle"]

    yearly = {}
    for year in years:
        cs_inst = cs_cls(dict(cs_params))
        ss_inst = ss_cls(dict(ss_params))

        bt = PortfolioBacktester(
            strategies=[("credit_spread", cs_inst), ("straddle_strangle", ss_inst)],
            tickers=TICKERS,
            start_date=datetime(year, 1, 1),
            end_date=datetime(year, 12, 31),
            starting_capital=STARTING_CAPITAL,
            max_positions=max_pos,
            max_positions_per_strategy=max_per_strat,
        )
        raw = bt.run()
        combined = raw.get("combined", raw)
        yearly[str(year)] = {
            "return_pct": combined.get("return_pct", 0),
            "max_drawdown": combined.get("max_drawdown", 0),
            "total_trades": combined.get("total_trades", 0),
            "win_rate": combined.get("win_rate", 0),
            "sharpe_ratio": combined.get("sharpe_ratio", 0),
        }

    rets = [yearly[str(y)]["return_pct"] for y in years]
    dds = [yearly[str(y)]["max_drawdown"] for y in years]
    avg_ret = sum(rets) / len(rets) if rets else 0
    worst_dd = min(dds) if dds else 0

    return {
        "yearly": yearly,
        "avg_return": round(avg_ret, 2),
        "worst_dd": round(worst_dd, 2),
    }


def phase_a_train(cs_base: Dict, ss_base: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Phase A: staged grid search on 2020-2022.

    Stage 1: Optimize CS regime scales (SS at defaults) — ~256 combos
    Stage 2: Fix CS at best, optimize SS regime scales — ~81 combos
    Stage 3: Fine-tune joint top-5 × top-5 — ~25 combos
    """
    print("\n" + "=" * 70)
    print("PHASE A: Training on 2020-2022")
    print("=" * 70)

    # ── Stage 1: CS regime scales ─────────────────────────────────────
    cs_grid = _make_grid(CS_REGIME_GRID)
    # Add fixed crash=0 to all CS configs
    for combo in cs_grid:
        combo["regime_scale_crash"] = 0.0

    print(f"\n[Stage 1] Optimizing CS regime scales ({len(cs_grid)} combos)...")
    t0 = time.time()

    cs_results = []
    for i, cs_regime in enumerate(cs_grid):
        cs_params = dict(cs_base)
        cs_params.update(cs_regime)

        result = run_blend_years(cs_params, ss_base, TRAIN_YEARS)
        entry = {
            "cs_regime": cs_regime,
            "ss_regime": {},
            **result,
        }
        cs_results.append(entry)

        if (i + 1) % 50 == 0 or i == len(cs_grid) - 1:
            valid = [r for r in cs_results if r["worst_dd"] > -15 and r["avg_return"] > 0]
            best_str = f"best={max(r['avg_return'] for r in valid):+.1f}%" if valid else "no valid"
            print(f"  [{i+1}/{len(cs_grid)}] {best_str} ({time.time()-t0:.0f}s)")

    # Filter: positive return, DD > -15%
    cs_valid = [r for r in cs_results if r["avg_return"] > 0 and r["worst_dd"] > -15]
    cs_valid.sort(key=lambda x: x["avg_return"], reverse=True)
    cs_top20 = cs_valid[:20]
    print(f"  Stage 1 done: {len(cs_valid)}/{len(cs_grid)} valid, top={cs_top20[0]['avg_return']:+.1f}%" if cs_top20 else "  No valid CS configs!")

    # ── Stage 2: SS regime scales (using best CS config) ──────────────
    best_cs_regime = cs_top20[0]["cs_regime"] if cs_top20 else {}
    cs_params_best = dict(cs_base)
    cs_params_best.update(best_cs_regime)

    ss_grid = _make_grid(SS_REGIME_GRID)
    # Add fixed crash=0.5 to all SS configs
    for combo in ss_grid:
        combo["regime_scale_crash"] = 0.5

    print(f"\n[Stage 2] Optimizing SS regime scales ({len(ss_grid)} combos, CS fixed at best)...")
    t0 = time.time()

    ss_results = []
    for i, ss_regime in enumerate(ss_grid):
        ss_params = dict(ss_base)
        ss_params.update(ss_regime)

        result = run_blend_years(cs_params_best, ss_params, TRAIN_YEARS)
        entry = {
            "cs_regime": best_cs_regime,
            "ss_regime": ss_regime,
            **result,
        }
        ss_results.append(entry)

        if (i + 1) % 20 == 0 or i == len(ss_grid) - 1:
            valid = [r for r in ss_results if r["worst_dd"] > -15 and r["avg_return"] > 0]
            best_str = f"best={max(r['avg_return'] for r in valid):+.1f}%" if valid else "no valid"
            print(f"  [{i+1}/{len(ss_grid)}] {best_str} ({time.time()-t0:.0f}s)")

    ss_valid = [r for r in ss_results if r["avg_return"] > 0 and r["worst_dd"] > -15]
    ss_valid.sort(key=lambda x: x["avg_return"], reverse=True)
    ss_top20 = ss_valid[:20]
    print(f"  Stage 2 done: {len(ss_valid)}/{len(ss_grid)} valid, top={ss_top20[0]['avg_return']:+.1f}%" if ss_top20 else "  No valid SS configs!")

    # ── Stage 3: Fine-tune joint top-5 × top-5 ───────────────────────
    cs_top5 = cs_top20[:5]
    ss_top5 = ss_top20[:5]

    if not cs_top5 or not ss_top5:
        print("  Skipping Stage 3: not enough valid configs from stages 1-2")
        all_results = cs_results + ss_results
        all_valid = [r for r in all_results if r["avg_return"] > 0 and r["worst_dd"] > -15]
        all_valid.sort(key=lambda x: x["avg_return"], reverse=True)
        return all_valid[:20], all_results

    joint_combos = list(itertools.product(cs_top5, ss_top5))
    print(f"\n[Stage 3] Fine-tuning top-5 × top-5 ({len(joint_combos)} combos)...")
    t0 = time.time()

    joint_results = []
    for i, (cs_entry, ss_entry) in enumerate(joint_combos):
        cs_params = dict(cs_base)
        cs_params.update(cs_entry["cs_regime"])
        ss_params = dict(ss_base)
        ss_params.update(ss_entry["ss_regime"])

        result = run_blend_years(cs_params, ss_params, TRAIN_YEARS)
        entry = {
            "cs_regime": cs_entry["cs_regime"],
            "ss_regime": ss_entry["ss_regime"],
            **result,
        }
        joint_results.append(entry)

    joint_valid = [r for r in joint_results if r["avg_return"] > 0 and r["worst_dd"] > -15]
    joint_valid.sort(key=lambda x: x["avg_return"], reverse=True)
    print(f"  Stage 3 done: {len(joint_valid)}/{len(joint_combos)} valid, "
          f"top={joint_valid[0]['avg_return']:+.1f}%" if joint_valid else "  No valid joint configs!")

    # Merge all valid results and deduplicate by regime config
    all_results = cs_results + ss_results + joint_results
    all_valid = [r for r in all_results if r["avg_return"] > 0 and r["worst_dd"] > -15]
    # Deduplicate by regime key
    seen = set()
    deduped = []
    for r in sorted(all_valid, key=lambda x: x["avg_return"], reverse=True):
        key = (tuple(sorted(r["cs_regime"].items())), tuple(sorted(r["ss_regime"].items())))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    top20 = deduped[:20]
    return top20, all_results


def phase_b_validate(
    top20: List[Dict], cs_base: Dict, ss_base: Dict,
) -> List[Dict]:
    """Phase B: Validate top 20 training candidates on 2023-2025."""
    print("\n" + "=" * 70)
    print("PHASE B: Validation on 2023-2025")
    print("=" * 70)

    validated = []
    for i, candidate in enumerate(top20):
        cs_params = dict(cs_base)
        cs_params.update(candidate["cs_regime"])
        ss_params = dict(ss_base)
        ss_params.update(candidate["ss_regime"])

        result = run_blend_years(cs_params, ss_params, VALID_YEARS)

        entry = {
            "cs_regime": candidate["cs_regime"],
            "ss_regime": candidate["ss_regime"],
            "train_avg_return": candidate["avg_return"],
            "train_worst_dd": candidate["worst_dd"],
            "train_yearly": candidate["yearly"],
            "valid_avg_return": result["avg_return"],
            "valid_worst_dd": result["worst_dd"],
            "valid_yearly": result["yearly"],
        }

        # Score: avg_return * (1 - |max_dd|/30)
        entry["validated_score"] = round(
            result["avg_return"] * (1 - abs(result["worst_dd"]) / 30), 2
        )
        validated.append(entry)

        print(f"  [{i+1}/{len(top20)}] train={candidate['avg_return']:+.1f}% "
              f"valid={result['avg_return']:+.1f}% DD={result['worst_dd']:.1f}% "
              f"score={entry['validated_score']:.1f}")

    # Filter: validation DD > -15%
    valid_filtered = [v for v in validated if v["valid_worst_dd"] > -15]
    valid_filtered.sort(key=lambda x: x["validated_score"], reverse=True)

    print(f"\n  Validated: {len(valid_filtered)}/{len(validated)} pass DD<15% gate")
    if valid_filtered:
        best = valid_filtered[0]
        print(f"  Best: score={best['validated_score']:.1f}, "
              f"train={best['train_avg_return']:+.1f}%, valid={best['valid_avg_return']:+.1f}%")

    return valid_filtered


def phase_c_confirm(
    best: Dict, cs_base: Dict, ss_base: Dict,
) -> Dict:
    """Phase C: Full 6-year confirmation run + baseline comparison."""
    print("\n" + "=" * 70)
    print("PHASE C: Full 6-Year Confirmation")
    print("=" * 70)

    # Run best config across all 6 years
    cs_params = dict(cs_base)
    cs_params.update(best["cs_regime"])
    ss_params = dict(ss_base)
    ss_params.update(best["ss_regime"])

    full_result = run_blend_years(cs_params, ss_params, ALL_YEARS)

    print("\n  OPTIMIZED (Phase 4) — Year-by-year:")
    for year in ALL_YEARS:
        r = full_result["yearly"][str(year)]
        print(f"    {year}: ret={r['return_pct']:+.1f}%  DD={r['max_drawdown']:.1f}%  "
              f"trades={r['total_trades']}  WR={r['win_rate']:.1f}%")
    print(f"    → Avg: {full_result['avg_return']:+.1f}%  Worst DD: {full_result['worst_dd']:.1f}%")

    # Run baseline (no regime scales)
    baseline_result = run_blend_years(cs_base, ss_base, ALL_YEARS)

    print("\n  BASELINE (Phase 3 static) — Year-by-year:")
    for year in ALL_YEARS:
        r = baseline_result["yearly"][str(year)]
        print(f"    {year}: ret={r['return_pct']:+.1f}%  DD={r['max_drawdown']:.1f}%  "
              f"trades={r['total_trades']}  WR={r['win_rate']:.1f}%")
    print(f"    → Avg: {baseline_result['avg_return']:+.1f}%  Worst DD: {baseline_result['worst_dd']:.1f}%")

    # Delta
    delta_ret = full_result["avg_return"] - baseline_result["avg_return"]
    delta_dd = full_result["worst_dd"] - baseline_result["worst_dd"]
    print(f"\n  DELTA: return {delta_ret:+.1f}%  DD {delta_dd:+.1f}%")

    return {
        "optimized": full_result,
        "baseline": baseline_result,
        "cs_regime_scales": best["cs_regime"],
        "ss_regime_scales": best["ss_regime"],
        "delta_return": round(delta_ret, 2),
        "delta_dd": round(delta_dd, 2),
    }


def main():
    print("=" * 70)
    print("PHASE 4: Regime Switching (Dynamic Allocation)")
    print("=" * 70)
    print(f"Base risk: CS={CS_BASE_RISK:.0%}, SS={SS_BASE_RISK:.0%}")
    print(f"Train: {TRAIN_YEARS}  Validate: {VALID_YEARS}")

    t_total = time.time()

    # Load base params
    cs_base = get_strategy_params("credit_spread", risk_override=CS_BASE_RISK)
    ss_base = get_strategy_params("straddle_strangle", risk_override=SS_BASE_RISK)

    # Phase A: Training
    top20, all_training = phase_a_train(cs_base, ss_base)
    if not top20:
        print("\nFATAL: No valid training configs found. Aborting.")
        return

    # Phase B: Validation
    validated = phase_b_validate(top20, cs_base, ss_base)
    if not validated:
        print("\nWARNING: No configs pass validation DD gate. Using best training config.")
        # Fall back to best training config
        validated = [{
            "cs_regime": top20[0]["cs_regime"],
            "ss_regime": top20[0]["ss_regime"],
            "train_avg_return": top20[0]["avg_return"],
            "train_worst_dd": top20[0]["worst_dd"],
            "validated_score": 0,
        }]

    # Phase C: Full confirmation
    confirmation = phase_c_confirm(validated[0], cs_base, ss_base)

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "generated": datetime.now().isoformat(),
        "config": {
            "tickers": TICKERS,
            "starting_capital": STARTING_CAPITAL,
            "cs_base_risk": CS_BASE_RISK,
            "ss_base_risk": SS_BASE_RISK,
            "train_years": TRAIN_YEARS,
            "valid_years": VALID_YEARS,
        },
        "training_top20": top20,
        "validation_results": validated,
        "best_config": {
            "cs_regime_scales": validated[0]["cs_regime"],
            "ss_regime_scales": validated[0]["ss_regime"],
            "validated_score": validated[0].get("validated_score", 0),
        },
        "confirmation": confirmation,
    }

    out_path = ROOT / "output" / "regime_switching_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    print(f"Total time: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
