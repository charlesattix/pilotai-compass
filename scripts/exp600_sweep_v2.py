#!/usr/bin/env python3
"""
EXP-600 Phase 2: Targeted sweep with higher risk and focused param ranges.

Quick sweep showed:
- Data works, trades flow (124-468/config across 6 years)
- Returns low with 2% risk — need higher risk to match Charles's results
- 2020 destroys everything with direction="both" (bull puts in COVID crash)
- Best quick: DTE=45, W=$10, OTM=5% → +4% avg

This sweep targets:
- Higher risk (5%, 8.5%, 10%)
- bull_put direction (regime handles bearish)
- DTE 15, 30, 45
- Stop losses 1.25x, 1.5x, 2.0x
- Profit targets 30%, 50%, 75%
- Spread widths 5, 10, 12

Charles reference: 2020:+86.9%, 2021:+216.7%, 2022:+28.3%, 2023:+12.9%, 2024:-5.1%, 2025:-0.8%
"""

import itertools
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

OUTPUT_DIR = ROOT / "results" / "exp600"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def build_config(params, starting_capital=100_000):
    """Build Backtester config from sweep params."""
    target_dte = params["target_dte"]
    min_dte = max(10, target_dte - 10)
    return {
        "strategy": {
            "target_delta":        params.get("target_delta", 0.12),
            "use_delta_selection": params.get("use_delta_selection", True),
            "target_dte":          target_dte,
            "min_dte":             min_dte,
            "spread_width":        params["spread_width"],
            "min_credit_pct":      params.get("min_credit_pct", 10),
            "direction":           params["direction"],
            "trend_ma_period":     params.get("trend_ma_period", 50),
            "regime_mode":         "combo",
            "regime_config":       {},
            "momentum_filter_pct": params.get("momentum_filter_pct", None),
            "iron_condor":         {"enabled": False},
            "iv_rank_min_entry":   0,
            "vix_max_entry":       0,
            "vix_close_all":       0,
            "vix_dynamic_sizing":  {},
            "seasonal_sizing":     {},
            "compass_enabled":     False,
            "compass_rrg_filter":  False,
        },
        "risk": {
            "stop_loss_multiplier": params["stop_loss_multiplier"],
            "profit_target":        params["profit_target"],
            "max_risk_per_trade":   params["max_risk_per_trade"],
            "max_contracts":        params.get("max_contracts", 10),
            "max_positions":        50,
            "drawdown_cb_pct":      20,
        },
        "backtest": {
            "starting_capital":   starting_capital,
            "commission_per_contract": 0.65,
            "slippage":           0.05,
            "exit_slippage":      0.10,
            "compound":           params.get("compound", True),
            "sizing_mode":        "flat",
            "slippage_multiplier": 1.0,
            "max_portfolio_exposure_pct": 100.0,
            "exclude_months":     [],
            "volume_gate":        False,
            "oi_gate":            False,
        },
    }


def run_single(ticker, year, params, starting_capital=100_000):
    """Run a single-year backtest using IronVault real data."""
    from backtest.backtester import Backtester
    from shared.iron_vault import IronVault

    config = build_config(params, starting_capital=starting_capital)
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)

    hd = IronVault.instance()
    otm_pct = params.get("otm_pct", 0.03)
    bt = Backtester(config, historical_data=hd, otm_pct=otm_pct)
    result = bt.run_backtest(ticker, start, end)
    result = result or {}
    result["year"] = year
    return result


def run_sweep(param_grid, years, ticker="SPY"):
    """Run all parameter combinations across all years."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"\n{'='*80}")
    print(f"  EXP-600 Phase 2: Targeted Real Data Sweep")
    print(f"  {total} combos × {len(years)} years = {total * len(years)} backtests")
    print(f"  Ticker: {ticker} | Iron Vault: ACTIVE")
    print(f"{'='*80}\n")

    results = []
    t_start = time.time()

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        combo_id = f"T{i:04d}"
        label = (f"DTE={params['target_dte']} W=${params['spread_width']} "
                 f"OTM={params.get('otm_pct', 0.03)*100:.0f}% "
                 f"PT={params['profit_target']}% SL={params['stop_loss_multiplier']}x "
                 f"risk={params['max_risk_per_trade']}% dir={params['direction']}")
        print(f"  [{i+1}/{total}] {combo_id}: {label}")

        year_results = {}
        for year in sorted(years):
            try:
                r = run_single(ticker, year, params)
                ret = r.get("return_pct", 0)
                trades = r.get("total_trades", 0)
                wr = r.get("win_rate", 0)
                dd = r.get("max_drawdown", 0)
                year_results[str(year)] = {
                    "return_pct": round(ret, 2),
                    "trades": trades,
                    "win_rate": round(wr, 1),
                    "max_drawdown": round(dd, 2),
                    "sharpe": round(r.get("sharpe_ratio", 0), 2),
                }
                flag = "+" if ret > 0 else "-"
                print(f"    {year}: {ret:+6.1f}% ({trades:>3} trades, WR {wr:.0f}%, DD {dd:.1f}%) {flag}")
            except Exception as e:
                year_results[str(year)] = {"return_pct": 0, "trades": 0, "error": str(e)}
                print(f"    {year}: ERROR {e}")

        # Compute summary
        rets = [yr["return_pct"] for yr in year_results.values() if "error" not in yr]
        trades_list = [yr["trades"] for yr in year_results.values() if "error" not in yr]
        dds = [yr["max_drawdown"] for yr in year_results.values() if "error" not in yr]
        wrs = [yr["win_rate"] for yr in year_results.values() if "error" not in yr and yr["trades"] > 0]

        avg_ret = sum(rets) / len(rets) if rets else 0
        total_trades = sum(trades_list)
        avg_trades = sum(trades_list) / len(trades_list) if trades_list else 0
        years_profitable = sum(1 for r in rets if r > 0)
        worst_dd = min(dds) if dds else 0
        avg_wr = sum(wrs) / len(wrs) if wrs else 0

        summary = {
            "combo_id": combo_id,
            "params": params,
            "avg_return": round(avg_ret, 2),
            "total_trades": total_trades,
            "avg_trades_per_year": round(avg_trades, 1),
            "years_profitable": f"{years_profitable}/{len(rets)}",
            "worst_drawdown": round(worst_dd, 2),
            "avg_win_rate": round(avg_wr, 1),
            "by_year": year_results,
        }
        results.append(summary)

        flag = "GOOD" if avg_ret > 10 and years_profitable >= 4 else ("OK" if avg_ret > 0 else "BAD")
        print(f"    → AVG {avg_ret:+.1f}%, {total_trades} trades, "
              f"{years_profitable}/{len(rets)} yrs+, DD {worst_dd:.1f}%  [{flag}]\n")

    elapsed = time.time() - t_start

    # Sort by avg_return descending
    results.sort(key=lambda x: x["avg_return"], reverse=True)

    # Save
    output = {
        "experiment": "EXP-600-phase2",
        "ticker": ticker,
        "years": years,
        "total_combos": total,
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_v2_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print leaderboard
    print(f"\n{'='*80}")
    print(f"  LEADERBOARD — Top 15")
    print(f"{'='*80}")
    print(f"  {'Rk':<3} {'ID':<7} {'Avg':>7} {'Trds':>5} {'Yr+':>4} {'WR':>5} {'DD':>7}  Config")
    print(f"  {'-'*78}")
    for rank, r in enumerate(results[:15], 1):
        p = r["params"]
        cfg = (f"DTE={p['target_dte']} W=${p['spread_width']} "
               f"OTM={p.get('otm_pct',0.03)*100:.0f}% "
               f"PT={p['profit_target']}% SL={p['stop_loss_multiplier']}x "
               f"risk={p['max_risk_per_trade']}%")
        print(f"  {rank:<3} {r['combo_id']:<7} {r['avg_return']:>+6.1f}% {r['total_trades']:>5} "
              f"{r['years_profitable']:>4} {r['avg_win_rate']:>4.0f}% {r['worst_drawdown']:>6.1f}%  {cfg}")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {output_path}")
    print(f"{'='*80}\n")

    return results


def main():
    # Targeted grid: 3×3×2×2×2×2×1 = 144 combos
    param_grid = {
        "target_dte":          [15, 30, 45],
        "spread_width":        [5, 10, 12],
        "otm_pct":             [0.03, 0.05],
        "profit_target":       [50, 75],
        "stop_loss_multiplier":[1.25, 2.0],
        "max_risk_per_trade":  [5.0, 8.5],
        "direction":           ["both"],
        "use_delta_selection":  [False],
    }

    run_sweep(param_grid, YEARS)


if __name__ == "__main__":
    main()
