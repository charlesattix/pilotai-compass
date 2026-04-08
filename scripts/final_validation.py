#!/usr/bin/env python3
"""
final_validation.py — Phase 5: Final Validation & Stress Testing

Validates the Phase 4 regime-optimized blend (CS 12% + SS 3% with regime scales)
through four independent checks:

  Section 1: Walk-Forward Validation (3-fold rolling)
  Section 2: Monte Carlo Simulation (10,000 paths for max DD distribution)
  Section 3: Slippage Modeling (entry/exit cost deduction)
  Section 4: Tail Risk Scenarios (COVID crash, rate hike bear)

Output: output/final_validation_results.json

Usage:
    PYTHONPATH=. python3 scripts/final_validation.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.portfolio_backtester import PortfolioBacktester
from scripts.portfolio_blend import get_strategy_params
from strategies import STRATEGY_REGISTRY

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — Phase 4 best config
# ═══════════════════════════════════════════════════════════════════════════════

TICKERS = ["SPY"]
STARTING_CAPITAL = 100_000
ALL_YEARS = list(range(2020, 2026))

CS_BASE_RISK = 0.12
SS_BASE_RISK = 0.03

CS_REGIME_SCALES = {
    "regime_scale_bull": 1.0,
    "regime_scale_bear": 0.3,
    "regime_scale_high_vol": 0.3,
    "regime_scale_low_vol": 0.8,
    "regime_scale_crash": 0.0,
}

SS_REGIME_SCALES = {
    "regime_scale_bull": 1.5,
    "regime_scale_bear": 1.5,
    "regime_scale_high_vol": 2.5,
    "regime_scale_low_vol": 1.0,
    "regime_scale_crash": 0.5,
}


def build_blend_params() -> Tuple[Dict, Dict]:
    """Build CS and SS params with regime scales applied."""
    cs_params = get_strategy_params("credit_spread", risk_override=CS_BASE_RISK)
    cs_params.update(CS_REGIME_SCALES)

    ss_params = get_strategy_params("straddle_strangle", risk_override=SS_BASE_RISK)
    ss_params.update(SS_REGIME_SCALES)

    return cs_params, ss_params


def run_blend(
    cs_params: Dict, ss_params: Dict,
    start: datetime, end: datetime,
) -> Tuple[PortfolioBacktester, Dict]:
    """Run CS+SS blend for a date range. Returns (backtester, results_dict)."""
    cs_cls = STRATEGY_REGISTRY["credit_spread"]
    ss_cls = STRATEGY_REGISTRY["straddle_strangle"]

    bt = PortfolioBacktester(
        strategies=[("credit_spread", cs_cls(dict(cs_params))),
                    ("straddle_strangle", ss_cls(dict(ss_params)))],
        tickers=TICKERS,
        start_date=start,
        end_date=end,
        starting_capital=STARTING_CAPITAL,
        max_positions=10,
        max_positions_per_strategy=5,
    )
    raw = bt.run()
    combined = raw.get("combined", raw)

    return bt, {
        "return_pct": combined.get("return_pct", 0),
        "max_drawdown": combined.get("max_drawdown", 0),
        "total_trades": combined.get("total_trades", 0),
        "win_rate": combined.get("win_rate", 0),
        "sharpe_ratio": combined.get("sharpe_ratio", 0),
    }


def run_blend_years(
    cs_params: Dict, ss_params: Dict, years: List[int],
) -> Tuple[Dict, List]:
    """Run blend across specified years. Returns (summary, all_closed_trades)."""
    yearly = {}
    all_trades = []

    for year in years:
        bt, result = run_blend(
            cs_params, ss_params,
            datetime(year, 1, 1), datetime(year, 12, 31),
        )
        yearly[str(year)] = result
        all_trades.extend(bt.closed_trades)

    rets = [yearly[str(y)]["return_pct"] for y in years]
    dds = [yearly[str(y)]["max_drawdown"] for y in years]
    avg_ret = sum(rets) / len(rets) if rets else 0
    worst_dd = min(dds) if dds else 0

    summary = {
        "yearly": yearly,
        "avg_return": round(avg_ret, 2),
        "worst_dd": round(worst_dd, 2),
        "total_trades": len(all_trades),
    }
    return summary, all_trades


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════════════════

def section_walk_forward(yearly_results: Dict) -> Dict:
    """3-fold rolling walk-forward validation using cached yearly results."""
    print("\n" + "=" * 70)
    print("SECTION 1: Walk-Forward Validation")
    print("=" * 70)

    folds = [
        {"train": [2020, 2021, 2022], "test": [2023]},
        {"train": [2020, 2021, 2022, 2023], "test": [2024]},
        {"train": [2020, 2021, 2022, 2023, 2024], "test": [2025]},
    ]

    fold_results = []
    for i, fold in enumerate(folds):
        train_rets = [yearly_results[str(y)]["return_pct"] for y in fold["train"]]
        test_ret = yearly_results[str(fold["test"][0])]["return_pct"]
        train_avg = sum(train_rets) / len(train_rets)

        ratio = test_ret / train_avg if train_avg != 0 else 0.0

        entry = {
            "fold": i + 1,
            "train_years": fold["train"],
            "test_year": fold["test"][0],
            "train_avg_return": round(train_avg, 2),
            "test_return": round(test_ret, 2),
            "ratio": round(ratio, 3),
            "test_profitable": test_ret > 0,
        }
        fold_results.append(entry)

        print(f"  Fold {i+1}: train {fold['train']} avg={train_avg:+.1f}% → "
              f"test {fold['test'][0]} ret={test_ret:+.1f}%  ratio={ratio:.3f}")

    ratios = [f["ratio"] for f in fold_results]
    median_ratio = float(np.median(ratios))
    profitable_folds = sum(1 for f in fold_results if f["test_profitable"])

    passed = median_ratio >= 0.50 and profitable_folds >= 2
    print(f"\n  Median ratio: {median_ratio:.3f} (need ≥0.50)")
    print(f"  Profitable folds: {profitable_folds}/3 (need ≥2)")
    print(f"  → {'PASS' if passed else 'FAIL'}")

    return {
        "folds": fold_results,
        "median_ratio": round(median_ratio, 3),
        "profitable_folds": profitable_folds,
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def section_monte_carlo(all_trades: List, n_paths: int = 10_000) -> Dict:
    """Shuffle trade ordering to get max drawdown distribution."""
    print("\n" + "=" * 70)
    print("SECTION 2: Monte Carlo Simulation (10,000 paths)")
    print("=" * 70)

    pnls = np.array([t.realized_pnl for t in all_trades])
    n_trades = len(pnls)
    total_return = float(pnls.sum())

    print(f"  Trades: {n_trades}, Total P&L: ${total_return:,.0f}")

    rng = np.random.default_rng(42)
    max_dds = np.empty(n_paths)

    for i in range(n_paths):
        shuffled = rng.permutation(pnls)
        equity = np.cumsum(shuffled) + STARTING_CAPITAL
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max * 100
        max_dds[i] = float(drawdowns.min())

    percentiles = {
        "p5": round(float(np.percentile(max_dds, 5)), 2),
        "p25": round(float(np.percentile(max_dds, 25)), 2),
        "p50": round(float(np.percentile(max_dds, 50)), 2),
        "p75": round(float(np.percentile(max_dds, 75)), 2),
        "p95": round(float(np.percentile(max_dds, 95)), 2),
    }

    passed = percentiles["p5"] > -25
    print("  Max DD percentiles:")
    for k, v in percentiles.items():
        print(f"    {k.upper()}: {v:.2f}%")
    print(f"\n  P5 DD: {percentiles['p5']:.2f}% (need > -25%)")
    print(f"  → {'PASS' if passed else 'FAIL'}")

    return {
        "n_paths": n_paths,
        "n_trades": n_trades,
        "total_pnl": round(total_return, 2),
        "max_dd_percentiles": percentiles,
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Slippage Modeling
# ═══════════════════════════════════════════════════════════════════════════════

def section_slippage(all_trades: List, yearly_results: Dict) -> Dict:
    """Post-process trades with realistic slippage costs."""
    print("\n" + "=" * 70)
    print("SECTION 3: Slippage Modeling")
    print("=" * 70)

    # Slippage per contract per leg:
    #   Entry: $0.05/share × 100 shares = $5/contract/leg
    #   Exit:  $0.10/share × 100 shares = $10/contract/leg
    ENTRY_SLIPPAGE_PER_CONTRACT_PER_LEG = 5.0   # $0.05 × 100
    EXIT_SLIPPAGE_PER_CONTRACT_PER_LEG = 10.0   # $0.10 × 100

    total_base_pnl = sum(t.realized_pnl for t in all_trades)
    total_slippage = 0.0

    # Per-year tracking
    year_slippage = {}
    for t in all_trades:
        n_legs = len(t.legs)
        contracts = t.contracts
        entry_slip = ENTRY_SLIPPAGE_PER_CONTRACT_PER_LEG * contracts * n_legs
        exit_slip = EXIT_SLIPPAGE_PER_CONTRACT_PER_LEG * contracts * n_legs
        trade_slippage = entry_slip + exit_slip
        total_slippage += trade_slippage

        if t.entry_date:
            year = t.entry_date.year
            if year not in year_slippage:
                year_slippage[year] = {"slippage": 0.0, "trades": 0}
            year_slippage[year]["slippage"] += trade_slippage
            year_slippage[year]["trades"] += 1

    adjusted_pnl = total_base_pnl - total_slippage
    base_return_pct = total_base_pnl / STARTING_CAPITAL * 100
    adjusted_return_pct = adjusted_pnl / STARTING_CAPITAL * 100
    drag_pct = total_slippage / STARTING_CAPITAL * 100

    print(f"  Total trades: {len(all_trades)}")
    print(f"  Total slippage: ${total_slippage:,.0f} ({drag_pct:.2f}% of capital)")
    print(f"  Base P&L:     ${total_base_pnl:,.0f} ({base_return_pct:.1f}%)")
    print(f"  Adjusted P&L: ${adjusted_pnl:,.0f} ({adjusted_return_pct:.1f}%)")

    # Per-year breakdown
    per_year = {}
    print("\n  Per-year breakdown:")
    for year in ALL_YEARS:
        base_ret = yearly_results[str(year)]["return_pct"]
        slip = year_slippage.get(year, {}).get("slippage", 0)
        slip_pct = slip / STARTING_CAPITAL * 100
        adj_ret = base_ret - slip_pct
        per_year[str(year)] = {
            "base_return": round(base_ret, 2),
            "slippage_pct": round(slip_pct, 2),
            "adjusted_return": round(adj_ret, 2),
            "trades": year_slippage.get(year, {}).get("trades", 0),
        }
        print(f"    {year}: base={base_ret:+.1f}%  slip=-{slip_pct:.1f}%  adj={adj_ret:+.1f}%")

    passed = adjusted_pnl > 0
    print(f"\n  Adjusted total return: {adjusted_return_pct:+.1f}% (need > 0%)")
    print(f"  → {'PASS' if passed else 'FAIL'}")

    return {
        "total_trades": len(all_trades),
        "total_slippage_usd": round(total_slippage, 2),
        "slippage_pct_of_capital": round(drag_pct, 2),
        "base_pnl": round(total_base_pnl, 2),
        "adjusted_pnl": round(adjusted_pnl, 2),
        "base_return_pct": round(base_return_pct, 2),
        "adjusted_return_pct": round(adjusted_return_pct, 2),
        "per_year": per_year,
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Tail Risk Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

def section_tail_risk(cs_params: Dict, ss_params: Dict) -> Dict:
    """Run backtests on specific stress periods."""
    print("\n" + "=" * 70)
    print("SECTION 4: Tail Risk Scenarios")
    print("=" * 70)

    scenarios = [
        {
            "name": "COVID Crash",
            "start": datetime(2020, 2, 1),
            "end": datetime(2020, 3, 31),
        },
        {
            "name": "Rate Hike Bear",
            "start": datetime(2022, 1, 1),
            "end": datetime(2022, 6, 30),
        },
    ]

    results = {}
    all_passed = True

    for scenario in scenarios:
        print(f"\n  {scenario['name']} ({scenario['start'].strftime('%Y-%m-%d')} to "
              f"{scenario['end'].strftime('%Y-%m-%d')}):")

        bt, result = run_blend(cs_params, ss_params, scenario["start"], scenario["end"])

        passed = result["max_drawdown"] > -25
        if not passed:
            all_passed = False

        results[scenario["name"]] = {
            "period": f"{scenario['start'].strftime('%Y-%m-%d')} to {scenario['end'].strftime('%Y-%m-%d')}",
            "return_pct": round(result["return_pct"], 2),
            "max_drawdown": round(result["max_drawdown"], 2),
            "total_trades": result["total_trades"],
            "win_rate": round(result["win_rate"], 2),
            "passed": passed,
        }

        print(f"    Return: {result['return_pct']:+.1f}%")
        print(f"    Max DD: {result['max_drawdown']:.1f}%")
        print(f"    Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.1f}%")
        print(f"    → {'PASS' if passed else 'FAIL'} (DD > -25%)")

    print(f"\n  Overall tail risk: {'PASS' if all_passed else 'FAIL'}")

    return {
        "scenarios": results,
        "passed": all_passed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 5: Final Validation & Stress Testing")
    print("=" * 70)
    print(f"Config: CS={CS_BASE_RISK:.0%} + SS={SS_BASE_RISK:.0%} with regime scales")
    print(f"Years: {ALL_YEARS}")

    t_total = time.time()

    # Build params
    cs_params, ss_params = build_blend_params()

    # ── Run all 6 years (shared by sections 1-3) ────────────────────────
    print("\n[0/4] Running 6-year backtest (shared baseline)...")
    t0 = time.time()
    summary, all_trades = run_blend_years(cs_params, ss_params, ALL_YEARS)
    print(f"  Done in {time.time()-t0:.0f}s — {summary['total_trades']} trades, "
          f"avg={summary['avg_return']:+.1f}%, worst DD={summary['worst_dd']:.1f}%")

    for year in ALL_YEARS:
        r = summary["yearly"][str(year)]
        print(f"    {year}: ret={r['return_pct']:+.1f}%  DD={r['max_drawdown']:.1f}%  "
              f"trades={r['total_trades']}  WR={r['win_rate']:.1f}%")

    # ── Section 1: Walk-Forward ──────────────────────────────────────────
    t0 = time.time()
    wf_result = section_walk_forward(summary["yearly"])
    print(f"  (completed in {time.time()-t0:.0f}s)")

    # ── Section 2: Monte Carlo ───────────────────────────────────────────
    t0 = time.time()
    mc_result = section_monte_carlo(all_trades)
    print(f"  (completed in {time.time()-t0:.1f}s)")

    # ── Section 3: Slippage ──────────────────────────────────────────────
    t0 = time.time()
    slip_result = section_slippage(all_trades, summary["yearly"])
    print(f"  (completed in {time.time()-t0:.1f}s)")

    # ── Section 4: Tail Risk ─────────────────────────────────────────────
    t0 = time.time()
    tail_result = section_tail_risk(cs_params, ss_params)
    print(f"  (completed in {time.time()-t0:.0f}s)")

    # ── Overall Verdict ──────────────────────────────────────────────────
    sections = {
        "walk_forward": wf_result["passed"],
        "monte_carlo": mc_result["passed"],
        "slippage": slip_result["passed"],
        "tail_risk": tail_result["passed"],
    }
    n_passed = sum(sections.values())

    if n_passed == 4:
        verdict = "PASS"
    elif n_passed >= 3:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "FAIL"

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    for name, passed in sections.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<20} {status}")
    print(f"\n  Sections passed: {n_passed}/4")
    print(f"  VERDICT: {verdict}")
    print(f"\n  Total time: {time.time()-t_total:.0f}s")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "generated": datetime.now().isoformat(),
        "config": {
            "tickers": TICKERS,
            "starting_capital": STARTING_CAPITAL,
            "cs_base_risk": CS_BASE_RISK,
            "ss_base_risk": SS_BASE_RISK,
            "cs_regime_scales": CS_REGIME_SCALES,
            "ss_regime_scales": SS_REGIME_SCALES,
        },
        "baseline_summary": summary,
        "walk_forward": wf_result,
        "monte_carlo": mc_result,
        "slippage": slip_result,
        "tail_risk": tail_result,
        "sections_passed": sections,
        "verdict": verdict,
    }

    out_path = ROOT / "output" / "final_validation_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
