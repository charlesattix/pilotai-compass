#!/usr/bin/env python3
"""
EXP-600: Real Data Parameter Sweep — Credit Spread on SPY

Uses IronVault.instance() (real Polygon data, cache-only).
NO synthetic data. NO Black-Scholes pricing. Cache miss = skip trade.

Sweeps credit spread parameters:
  - target_dte: 15, 25, 35, 45
  - spread_width: 3, 5, 10
  - otm_pct: 0.02, 0.03, 0.05
  - profit_target: 30, 50, 75 (% of credit)
  - stop_loss_multiplier: 1.5, 2.0, 2.5
  - direction: both, bull_put

Charles reference results (real data):
  2020:+86.9%, 2021:+216.7%, 2022:+28.3%, 2023:+12.9%, 2024:-5.1%, 2025:-0.8%

Usage:
    python3 scripts/exp600_sweep.py                  # full sweep
    python3 scripts/exp600_sweep.py --quick           # reduced grid (fast)
    python3 scripts/exp600_sweep.py --years 2024,2025 # subset
"""

import argparse
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

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

OUTPUT_DIR = ROOT / "results" / "exp600"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("exp600")

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def build_config(params, starting_capital=100_000):
    """Build Backtester-compatible config dict from sweep params."""
    return {
        "strategy": {
            "target_delta":        params.get("target_delta", 0.12),
            "use_delta_selection": params.get("use_delta_selection", True),
            "target_dte":          params["target_dte"],
            "min_dte":             params.get("min_dte", max(10, params["target_dte"] - 10)),
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
            "max_risk_per_trade":   params.get("max_risk_per_trade", 2.0),
            "max_contracts":        params.get("max_contracts", 10),
            "max_positions":        50,
            "drawdown_cb_pct":      20,
        },
        "backtest": {
            "starting_capital":   starting_capital,
            "commission_per_contract": 0.65,
            "slippage":           0.05,
            "exit_slippage":      0.10,
            "compound":           False,
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
    # Generate all combos
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"\n{'='*80}")
    print(f"  EXP-600: Real Data Credit Spread Sweep")
    print(f"  {total} parameter combinations × {len(years)} years = {total * len(years)} backtests")
    print(f"  Ticker: {ticker}")
    print(f"  Iron Vault: ACTIVE (real data only)")
    print(f"{'='*80}\n")

    results = []
    t_start = time.time()

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        combo_id = f"C{i:04d}"
        label = (f"DTE={params['target_dte']} W=${params['spread_width']} "
                 f"OTM={params.get('otm_pct', 0.03)*100:.0f}% "
                 f"PT={params['profit_target']}% SL={params['stop_loss_multiplier']}x "
                 f"dir={params['direction']}")
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
                logger.error("Year %d failed: %s", year, e)
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
        print(f"    → AVG {avg_ret:+.1f}%, {total_trades} trades total, "
              f"{years_profitable}/{len(rets)} yrs profitable, DD {worst_dd:.1f}%  [{flag}]\n")

    elapsed = time.time() - t_start

    # Sort by avg_return descending
    results.sort(key=lambda x: x["avg_return"], reverse=True)

    # Save results
    output = {
        "experiment": "EXP-600",
        "description": "Real Data Credit Spread Parameter Sweep",
        "ticker": ticker,
        "years": years,
        "total_combos": total,
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }

    output_path = OUTPUT_DIR / "sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print leaderboard
    print(f"\n{'='*80}")
    print(f"  LEADERBOARD — Top 10")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'ID':<7} {'AvgRet':>8} {'Trades':>7} {'Yrs+':>5} {'WR':>6} {'DD':>7}  Params")
    print(f"  {'-'*75}")
    for rank, r in enumerate(results[:10], 1):
        p = r["params"]
        param_str = (f"DTE={p['target_dte']} W=${p['spread_width']} "
                     f"OTM={p.get('otm_pct', 0.03)*100:.0f}% "
                     f"PT={p['profit_target']}% SL={p['stop_loss_multiplier']}x "
                     f"dir={p['direction']}")
        print(f"  {rank:<5} {r['combo_id']:<7} {r['avg_return']:>+7.1f}% {r['total_trades']:>6} "
              f" {r['years_profitable']:>4} {r['avg_win_rate']:>5.0f}% {r['worst_drawdown']:>6.1f}%  {param_str}")

    # Print worst 5
    print(f"\n  BOTTOM 5:")
    for rank, r in enumerate(results[-5:], len(results)-4):
        p = r["params"]
        param_str = (f"DTE={p['target_dte']} W=${p['spread_width']} "
                     f"OTM={p.get('otm_pct', 0.03)*100:.0f}% "
                     f"PT={p['profit_target']}% SL={p['stop_loss_multiplier']}x "
                     f"dir={p['direction']}")
        print(f"  {rank:<5} {r['combo_id']:<7} {r['avg_return']:>+7.1f}% {r['total_trades']:>6} "
              f" {r['years_profitable']:>4} {r['avg_win_rate']:>5.0f}% {r['worst_drawdown']:>6.1f}%  {param_str}")

    print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results saved to {output_path}")
    print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="EXP-600 Credit Spread Sweep")
    parser.add_argument("--quick", action="store_true", help="Reduced grid for fast iteration")
    parser.add_argument("--years", help="Comma-separated years, e.g. 2024,2025")
    parser.add_argument("--ticker", default="SPY")
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")] if args.years else YEARS

    if args.quick:
        # Quick grid: 2×2×2×1×1×1 = 8 combos
        param_grid = {
            "target_dte":          [30, 45],
            "spread_width":        [5, 10],
            "otm_pct":             [0.03, 0.05],
            "profit_target":       [50],
            "stop_loss_multiplier":[2.0],
            "direction":           ["both"],
            "use_delta_selection":  [False],
            "max_risk_per_trade":   [2.0],
        }
    else:
        # Full grid: 4×3×3×3×3×2 = 648 combos — too many
        # Practical grid: 3×3×2×2×2×2 = 144 combos × 6 years = 864 backtests
        param_grid = {
            "target_dte":          [15, 30, 45],
            "spread_width":        [3, 5, 10],
            "otm_pct":             [0.02, 0.03, 0.05],
            "profit_target":       [30, 50, 75],
            "stop_loss_multiplier":[1.5, 2.0, 2.5],
            "direction":           ["both", "bull_put"],
            "use_delta_selection":  [False],
            "max_risk_per_trade":   [2.0],
        }

    run_sweep(param_grid, years, ticker=args.ticker)


if __name__ == "__main__":
    main()
