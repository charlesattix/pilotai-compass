#!/usr/bin/env python3
"""
portfolio_blend.py — Phase 3: Portfolio Blending & Weight Optimization

1. Run each strategy solo through 2020-2025, extract daily P&L series
2. Compute correlation matrix between strategy daily returns
3. Test equal-weight blend combinations (2/3/4-strategy)
4. Optimize per-strategy risk weights for top blends
5. Find max-return blend with drawdown under 30%
6. Output: output/portfolio_blend_results.json

Usage:
    PYTHONPATH=. python3 scripts/portfolio_blend.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.portfolio_backtester import PortfolioBacktester
from strategies import STRATEGY_REGISTRY

TICKERS = ["SPY"]
STARTING_CAPITAL = 100_000
YEARS = list(range(2020, 2026))

CHAMPION_PATH = ROOT / "configs" / "champion.json"


def _load_champion_params() -> Dict:
    with open(CHAMPION_PATH) as f:
        return json.load(f)["strategy_params"]


# Default params for strategies not in champion config
STRATEGY_CONFIGS = {
    "calendar_spread": {
        "front_dte": 13, "back_dte": 35, "strike_selection": "atm",
        "otm_offset_pct": 0.04, "option_type": "put", "max_iv_rank": 30.0,
        "profit_target_pct": 0.15, "stop_loss_pct": 0.6, "max_risk_pct": 0.035,
    },
    "straddle_strangle": {
        "mode": "short_post_event", "days_before_event": 3, "target_dte": 5,
        "otm_pct": 0.04, "event_iv_boost": 0.15, "iv_crush_pct": 0.5,
        "profit_target_pct": 0.55, "stop_loss_pct": 0.45, "max_risk_pct": 0.01,
        "event_types": "fomc_cpi",
    },
}


def get_strategy_params(name: str, risk_override: float = None) -> Dict:
    """Get best-known params for a strategy, optionally overriding max_risk_pct."""
    champ = _load_champion_params()
    if name in champ:
        params = dict(champ[name])
    elif name in STRATEGY_CONFIGS:
        params = dict(STRATEGY_CONFIGS[name])
    else:
        cls = STRATEGY_REGISTRY[name]
        params = cls.get_default_params()
    if risk_override is not None:
        params["max_risk_pct"] = risk_override
    return params


def run_solo_strategy(name: str, year: int) -> Tuple[Dict, List[Tuple[datetime, float]]]:
    """Run a single strategy for one year. Returns (results_dict, equity_curve)."""
    params = get_strategy_params(name)
    cls = STRATEGY_REGISTRY[name]
    strategy_instance = cls(params)

    bt = PortfolioBacktester(
        strategies=[(name, strategy_instance)],
        tickers=TICKERS,
        start_date=datetime(year, 1, 1),
        end_date=datetime(year, 12, 31),
        starting_capital=STARTING_CAPITAL,
        max_positions=10,
        max_positions_per_strategy=5,
    )
    raw = bt.run()
    combined = raw.get("combined", raw)

    return {
        "return_pct": combined.get("return_pct", 0),
        "max_drawdown": combined.get("max_drawdown", 0),
        "total_trades": combined.get("total_trades", 0),
        "win_rate": combined.get("win_rate", 0),
        "sharpe_ratio": combined.get("sharpe_ratio", 0),
    }, bt.equity_curve


def build_daily_returns(equity_curve: List[Tuple[datetime, float]]) -> pd.Series:
    """Convert equity curve to daily returns series."""
    if len(equity_curve) < 2:
        return pd.Series(dtype=float)
    dates = [d for d, _ in equity_curve]
    values = [v for _, v in equity_curve]
    eq = pd.Series(values, index=pd.DatetimeIndex(dates))
    returns = eq.pct_change().fillna(0)
    return returns


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Solo strategies
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_solo() -> Dict:
    """Run each strategy solo for all years, collect results and daily returns."""
    strategies = ["credit_spread", "iron_condor", "calendar_spread", "straddle_strangle"]
    all_results = {}
    all_daily_returns = {}

    for strat_name in strategies:
        print(f"\n  {strat_name}:")
        strat_results = {}
        strat_curves = []

        for year in YEARS:
            try:
                result, curve = run_solo_strategy(strat_name, year)
                strat_results[str(year)] = result
                strat_curves.extend(curve)
                print(f"    {year}: ret={result['return_pct']:+.1f}%  DD={result['max_drawdown']:.1f}%  "
                      f"trades={result['total_trades']}  WR={result['win_rate']:.1f}%")
            except Exception as e:
                print(f"    {year}: ERROR — {e}")
                strat_results[str(year)] = {"return_pct": 0, "error": str(e)}

        all_results[strat_name] = strat_results
        all_daily_returns[strat_name] = build_daily_returns(strat_curves)

    return all_results, all_daily_returns


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Correlation matrix
# ═══════════════════════════════════════════════════════════════════════════════

def compute_correlation(daily_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(daily_returns).fillna(0)
    return df.corr()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Equal-weight blend testing
# ═══════════════════════════════════════════════════════════════════════════════

def run_blend(strategy_names: List[str], year: int,
              risk_overrides: Dict[str, float] = None,
              max_pos: int = 10, max_per_strat: int = 5) -> Dict:
    """Run a blend of strategies for one year."""
    strategies = []
    for name in strategy_names:
        risk = risk_overrides.get(name) if risk_overrides else None
        params = get_strategy_params(name, risk_override=risk)
        cls = STRATEGY_REGISTRY[name]
        strategies.append((name, cls(params)))

    bt = PortfolioBacktester(
        strategies=strategies,
        tickers=TICKERS,
        start_date=datetime(year, 1, 1),
        end_date=datetime(year, 12, 31),
        starting_capital=STARTING_CAPITAL,
        max_positions=max_pos,
        max_positions_per_strategy=max_per_strat,
    )
    raw = bt.run()
    combined = raw.get("combined", raw)
    per_strat = raw.get("per_strategy", {})

    # Max loss streak
    max_streak = 0
    streak = 0
    for t in bt.closed_trades:
        if t.realized_pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "return_pct": combined.get("return_pct", 0),
        "max_drawdown": combined.get("max_drawdown", 0),
        "total_trades": combined.get("total_trades", 0),
        "win_rate": combined.get("win_rate", 0),
        "sharpe_ratio": combined.get("sharpe_ratio", 0),
        "max_loss_streak": max_streak,
        "per_strategy": {k: {"trades": v.get("total_trades", 0),
                             "pnl": v.get("total_pnl", 0)}
                         for k, v in per_strat.items()} if per_strat else {},
    }


def run_blend_all_years(strategy_names: List[str],
                        risk_overrides: Dict[str, float] = None,
                        max_pos: int = 10, max_per_strat: int = 5,
                        label: str = None) -> Dict:
    """Run a blend across all years, compute summary stats."""
    if label is None:
        label = " + ".join(s[:4].upper() for s in strategy_names)

    yearly = {}
    for year in YEARS:
        try:
            r = run_blend(strategy_names, year, risk_overrides, max_pos, max_per_strat)
            yearly[str(year)] = r
        except Exception as e:
            yearly[str(year)] = {"return_pct": 0, "error": str(e)}

    rets = [yearly[str(y)]["return_pct"] for y in YEARS if "error" not in yearly.get(str(y), {})]
    dds = [yearly[str(y)]["max_drawdown"] for y in YEARS if "error" not in yearly.get(str(y), {})]
    sharpes = [yearly[str(y)].get("sharpe_ratio", 0) for y in YEARS if "error" not in yearly.get(str(y), {})]
    trades = [yearly[str(y)]["total_trades"] for y in YEARS if "error" not in yearly.get(str(y), {})]

    avg_ret = sum(rets) / len(rets) if rets else 0
    worst_dd = min(dds) if dds else 0
    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
    total_trades = sum(trades)
    years_positive = sum(1 for r in rets if r > 0)

    return {
        "strategies": strategy_names,
        "label": label,
        "risk_weights": risk_overrides or {},
        "max_positions": max_pos,
        "max_per_strategy": max_per_strat,
        "yearly": yearly,
        "avg_return": round(avg_ret, 2),
        "worst_dd": round(worst_dd, 2),
        "avg_sharpe": round(avg_sharpe, 2),
        "total_trades": total_trades,
        "years_positive": years_positive,
        "dd_under_30": worst_dd > -30,
    }


def test_equal_weight_blends() -> List[Dict]:
    """Test various strategy combinations with default (equal) weights."""
    all_strategies = ["credit_spread", "iron_condor", "calendar_spread", "straddle_strangle"]

    blends_to_test = [
        # 2-strategy combos
        ["credit_spread", "iron_condor"],
        ["credit_spread", "straddle_strangle"],
        ["credit_spread", "calendar_spread"],
        ["iron_condor", "straddle_strangle"],
        ["iron_condor", "calendar_spread"],
        ["straddle_strangle", "calendar_spread"],
        # 3-strategy combos
        ["credit_spread", "iron_condor", "straddle_strangle"],
        ["credit_spread", "iron_condor", "calendar_spread"],
        ["credit_spread", "straddle_strangle", "calendar_spread"],
        ["iron_condor", "straddle_strangle", "calendar_spread"],
        # 4-strategy full blend
        all_strategies,
    ]

    results = []
    for blend in blends_to_test:
        label = " + ".join(s[:4].upper() for s in blend)
        print(f"\n  Blend: {label}")
        entry = run_blend_all_years(blend, label=label)
        results.append(entry)
        print(f"    → Avg={entry['avg_return']:+.1f}%  Worst DD={entry['worst_dd']:.1f}%  "
              f"Trades={entry['total_trades']}  Profitable={entry['years_positive']}/6")
        for y in YEARS:
            r = entry["yearly"][str(y)]
            if "error" in r:
                print(f"    {y}: ERROR — {r['error']}")
            else:
                print(f"    {y}: ret={r['return_pct']:+.1f}%  DD={r['max_drawdown']:.1f}%  trades={r['total_trades']}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Weight optimization for top blends
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_weights(blend_name: str, strategy_names: List[str]) -> List[Dict]:
    """Test multiple risk weight combinations for a given blend."""
    print(f"\n  Optimizing weights for: {blend_name}")

    # Build weight grid based on strategy count
    weight_combos = []

    if set(strategy_names) == {"credit_spread", "straddle_strangle"}:
        # CS is the driver — vary its risk %, keep SS as supplement
        for cs_risk in [0.05, 0.065, 0.085, 0.10, 0.12]:
            for ss_risk in [0.005, 0.01, 0.015, 0.02, 0.03]:
                weight_combos.append({
                    "credit_spread": cs_risk,
                    "straddle_strangle": ss_risk,
                })

    elif set(strategy_names) == {"credit_spread", "iron_condor", "straddle_strangle"}:
        # Core 3-strategy blend
        for cs_risk in [0.05, 0.085, 0.10, 0.12]:
            for ic_risk in [0.02, 0.035, 0.05]:
                for ss_risk in [0.005, 0.01, 0.02, 0.03]:
                    weight_combos.append({
                        "credit_spread": cs_risk,
                        "iron_condor": ic_risk,
                        "straddle_strangle": ss_risk,
                    })

    elif set(strategy_names) == {"credit_spread", "straddle_strangle", "calendar_spread"}:
        for cs_risk in [0.065, 0.085, 0.10, 0.12]:
            for ss_risk in [0.01, 0.02, 0.03]:
                for cal_risk in [0.02, 0.035, 0.05]:
                    weight_combos.append({
                        "credit_spread": cs_risk,
                        "straddle_strangle": ss_risk,
                        "calendar_spread": cal_risk,
                    })

    elif len(strategy_names) == 4:
        # Full 4-strategy — coarser grid
        for cs_risk in [0.065, 0.085, 0.10, 0.12]:
            for ic_risk in [0.02, 0.035]:
                for ss_risk in [0.01, 0.02]:
                    for cal_risk in [0.02, 0.035]:
                        weight_combos.append({
                            "credit_spread": cs_risk,
                            "iron_condor": ic_risk,
                            "straddle_strangle": ss_risk,
                            "calendar_spread": cal_risk,
                        })

    else:
        # Generic 2-strategy
        names = sorted(strategy_names)
        for r0 in [0.03, 0.05, 0.085, 0.10]:
            for r1 in [0.02, 0.035, 0.05]:
                weight_combos.append({names[0]: r0, names[1]: r1})

    # Also vary position limits for the best risk combos
    pos_configs = [
        (10, 5),   # default
        (12, 6),   # looser
        (8, 4),    # tighter
    ]

    results = []
    total = len(weight_combos) * len(pos_configs)
    print(f"    Testing {total} weight × position-limit combinations...")

    for i, weights in enumerate(weight_combos):
        for max_pos, max_per in pos_configs:
            w_str = " | ".join(f"{k[:4]}={v:.1%}" for k, v in weights.items())
            label = f"{blend_name} [{w_str}] pos={max_pos}/{max_per}"

            entry = run_blend_all_years(
                strategy_names, risk_overrides=weights,
                max_pos=max_pos, max_per_strat=max_per, label=label,
            )
            results.append(entry)

        # Progress
        done = (i + 1) * len(pos_configs)
        if (i + 1) % 5 == 0 or i == len(weight_combos) - 1:
            best_so_far = max(
                (r for r in results if r["dd_under_30"]),
                key=lambda x: x["avg_return"],
                default=None,
            )
            best_str = f"best={best_so_far['avg_return']:+.1f}%" if best_so_far else "no valid"
            print(f"    [{done}/{total}] {best_str}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 3: Portfolio Blending & Weight Optimization")
    print("=" * 70)

    # Step 1: Solo strategy performance
    print("\n[1/4] Running solo strategy backtests...")
    t0 = time.time()
    solo_results, daily_returns = run_all_solo()
    print(f"\n  Solo analysis complete in {time.time()-t0:.0f}s")

    # Step 2: Correlation matrix
    print("\n[2/4] Computing correlation matrix...")
    corr = compute_correlation(daily_returns)
    print("\n  Strategy Correlation Matrix:")
    print(corr.to_string(float_format=lambda x: f"{x:+.3f}"))

    # Step 3: Equal-weight blend testing
    print("\n[3/4] Testing equal-weight blends...")
    t0 = time.time()
    equal_results = test_equal_weight_blends()
    print(f"\n  Equal-weight testing complete in {time.time()-t0:.0f}s")

    # Step 4: Weight optimization for top blends
    print("\n[4/4] Optimizing weights for top blends...")
    t0 = time.time()

    # Optimize the top 4 blends from equal-weight results
    top_blends = [
        ("CRED+STRA", ["credit_spread", "straddle_strangle"]),
        ("CRED+IRON+STRA", ["credit_spread", "iron_condor", "straddle_strangle"]),
        ("CRED+STRA+CALE", ["credit_spread", "straddle_strangle", "calendar_spread"]),
        ("FULL_4", ["credit_spread", "iron_condor", "straddle_strangle", "calendar_spread"]),
    ]

    all_optimized = []
    for blend_name, strats in top_blends:
        opt_results = optimize_weights(blend_name, strats)
        all_optimized.extend(opt_results)

    print(f"\n  Weight optimization complete in {time.time()-t0:.0f}s")

    # ── Compile results ──────────────────────────────────────────────────
    # Filter valid blends (DD < 30%)
    all_blends = equal_results + all_optimized
    valid = [b for b in all_blends if b["dd_under_30"]]
    valid.sort(key=lambda x: x["avg_return"], reverse=True)

    print("\n" + "=" * 100)
    print("TOP 15 BLENDS (DD < 30%)")
    print("=" * 100)
    print(f"\n{'Rank':>4}  {'Blend':<55}  {'Avg Ret':>8}  {'Worst DD':>9}  {'Sharpe':>7}  {'Trades':>7}  {'Prof':>5}")
    print("-" * 100)
    for i, b in enumerate(valid[:15], 1):
        print(f"{i:>4}  {b['label'][:55]:<55}  {b['avg_return']:>+7.1f}%  {b['worst_dd']:>8.1f}%  "
              f"{b['avg_sharpe']:>6.2f}  {b['total_trades']:>7}  {b['years_positive']:>3}/6")

    # Best blend details
    if valid:
        best = valid[0]
        print(f"\n  BEST BLEND: {best['label']}")
        print(f"  Risk weights: {best['risk_weights']}")
        print(f"  Positions: max={best['max_positions']}, per_strat={best['max_per_strategy']}")
        print("\n  Year-by-year:")
        for y in YEARS:
            r = best["yearly"][str(y)]
            if "error" in r:
                print(f"    {y}: ERROR")
            else:
                print(f"    {y}: ret={r['return_pct']:+.1f}%  DD={r['max_drawdown']:.1f}%  "
                      f"trades={r['total_trades']}  WR={r['win_rate']:.1f}%  Sharpe={r.get('sharpe_ratio', 0):.2f}")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "generated": datetime.now().isoformat(),
        "tickers": TICKERS,
        "starting_capital": STARTING_CAPITAL,
        "solo_results": solo_results,
        "correlation_matrix": corr.to_dict(),
        "equal_weight_results": equal_results,
        "optimized_results": all_optimized,
        "top_15": valid[:15],
        "best_blend": valid[0] if valid else None,
    }

    out_path = ROOT / "output" / "portfolio_blend_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
