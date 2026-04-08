#!/usr/bin/env python3
"""
EXP-600 Phase 5: Focused sweep for CONSISTENCY.

Key findings from Phase 4 + trade flow debug:
  - W=$10 top results were fake: implicit vol-filter from min_credit threshold
    meant the strategy only traded in 2022/2025 (high vol), sat out 2021/2023
  - W=$5 is more reliable: lower credit threshold, more consistent trade flow
  - DTE=45 OTM=5% was best but daily bars are sparse at high DTE
  - DTE=30 has better bar coverage (median first-bar DTE ~35)

Phase 5 design:
  - W=$5 ONLY (lower min_credit = $0.50, consistent trades)
  - DTE=30 (better bar availability)
  - OTM: 2%, 3%, 4%, 5% (wider search)
  - Risk: 3%, 4%, 5% (higher to boost returns since W=$5 is low absolute risk)
  - PT: 40%, 50%, 65%   SL: 1.5x, 2.0x, 2.5x
  - regime_mode: 'combo' vs 'none' (does regime detection help or hurt?)
  - min_credit_pct: 5% (lowered from 10% — the Phase 4 smoking gun)

Grid: 4 OTM × 3 risk × 3 PT × 3 SL × 2 regime = 216 combos
Too many. Prune: fix SL=2.5x (Phase 4 winner), test PT × OTM × risk × regime.
Grid: 4 OTM × 3 risk × 3 PT × 2 regime = 72 combos × 6 years = 432 backtests

Target: 6/6 profitable years, >30 trades/year, >5% avg return.

Charles reference: 2020:+86.9%, 2021:+216.7%, 2022:+28.3%, 2023:+12.9%, 2024:-5.1%, 2025:-0.8%
"""

import itertools
import json
import logging
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
    target_dte = params["target_dte"]
    min_dte = max(10, target_dte - 10)
    regime_mode = params.get("regime_mode", "combo")
    # When regime_mode='none', direction matters. Use 'both' to allow puts and calls.
    # When regime_mode='combo', direction is overridden by regime detector.
    direction = params.get("direction", "both")
    return {
        "strategy": {
            "target_delta":        0.12,
            "use_delta_selection": False,
            "target_dte":          target_dte,
            "min_dte":             min_dte,
            "spread_width":        params["spread_width"],
            "min_credit_pct":      params.get("min_credit_pct", 5),
            "direction":           direction,
            "trend_ma_period":     params.get("trend_ma_period", 50),
            "regime_mode":         regime_mode,
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
            "max_contracts":        25,
            "max_positions":        50,
            "drawdown_cb_pct":      25,
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
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"\n{'='*80}")
    print(f"  EXP-600 Phase 5: Consistency Sweep (W=$5, DTE=30, min_credit=5%)")
    print(f"  {total} combos × {len(years)} years = {total * len(years)} backtests")
    print(f"  Each year: fresh $100K | compound=False | drawdown_cb=25%")
    print(f"  TARGET: 6/6 yrs profitable, >30 trades/yr, >5% avg return")
    print(f"{'='*80}\n")

    results = []
    t_start = time.time()

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        combo_id = f"C{i:04d}"
        regime = params.get("regime_mode", "combo")
        label = (f"DTE={params['target_dte']} W=${params['spread_width']} "
                 f"OTM={params.get('otm_pct', 0.03)*100:.0f}% "
                 f"PT={params['profit_target']}% SL={params['stop_loss_multiplier']}x "
                 f"risk={params['max_risk_per_trade']}% reg={regime}")
        print(f"  [{i+1}/{total}] {combo_id}: {label}")

        year_results = {}
        for year in sorted(years):
            try:
                r = run_single(ticker, year, params, starting_capital=100_000)
                ret = r.get("return_pct", 0)
                trades = r.get("total_trades", 0)
                wr = r.get("win_rate", 0)
                dd = r.get("max_drawdown", 0)
                ending_cap = r.get("ending_capital", 100_000)
                year_results[str(year)] = {
                    "return_pct": round(ret, 2),
                    "trades": trades,
                    "win_rate": round(wr, 1),
                    "max_drawdown": round(dd, 2),
                    "sharpe": round(r.get("sharpe_ratio", 0), 2),
                    "ending_capital": round(ending_cap),
                }
                flag = "+" if ret > 0 else "-"
                print(f"    {year}: {ret:+6.1f}% ({trades:>3} trades, WR {wr:.0f}%, DD {dd:.1f}%) {flag}")
            except Exception as e:
                year_results[str(year)] = {"return_pct": 0, "trades": 0, "error": str(e)}
                print(f"    {year}: ERROR {e}")

        rets = [yr["return_pct"] for yr in year_results.values() if "error" not in yr]
        trades_list = [yr["trades"] for yr in year_results.values() if "error" not in yr]
        dds = [yr["max_drawdown"] for yr in year_results.values() if "error" not in yr]
        wrs = [yr["win_rate"] for yr in year_results.values() if "error" not in yr and yr["trades"] > 0]

        avg_ret = sum(rets) / len(rets) if rets else 0
        total_trades = sum(trades_list)
        min_trades_yr = min(trades_list) if trades_list else 0
        years_profitable = sum(1 for r in rets if r > 0)
        worst_dd = min(dds) if dds else 0
        avg_wr = sum(wrs) / len(wrs) if wrs else 0

        # Compound final capital
        compound_cap = 100_000
        for yr_key in [str(y) for y in sorted(years)]:
            if yr_key in year_results and "error" not in year_results[yr_key]:
                compound_cap *= (1 + year_results[yr_key]["return_pct"] / 100)
        compound_return = (compound_cap - 100_000) / 100_000 * 100

        # Consistency score: rewards 6/6 profitable + high min trades + positive avg
        consistency = (
            years_profitable / len(rets) * 40  # up to 40 pts for profitability
            + min(1, min_trades_yr / 30) * 30   # up to 30 pts for trade flow
            + min(1, max(0, avg_ret) / 10) * 30 # up to 30 pts for avg return
        ) if rets else 0

        summary = {
            "combo_id": combo_id,
            "params": params,
            "avg_return": round(avg_ret, 2),
            "total_trades": total_trades,
            "min_trades_year": min_trades_yr,
            "years_profitable": f"{years_profitable}/{len(rets)}",
            "worst_drawdown": round(worst_dd, 2),
            "avg_win_rate": round(avg_wr, 1),
            "compound_final_capital": round(compound_cap),
            "compound_return_pct": round(compound_return, 1),
            "consistency_score": round(consistency, 1),
            "by_year": year_results,
        }
        results.append(summary)

        if years_profitable == len(rets) and avg_ret > 5 and min_trades_yr >= 30:
            flag = "TARGET"
        elif years_profitable == len(rets) and avg_ret > 0:
            flag = "GOOD"
        elif avg_ret > 0:
            flag = "OK"
        else:
            flag = "BAD"
        print(f"    → AVG {avg_ret:+.1f}%, {total_trades} trades (min {min_trades_yr}/yr), "
              f"{years_profitable}/{len(rets)} yrs+, DD {worst_dd:.1f}%, "
              f"score {consistency:.0f}  [{flag}]\n")

    elapsed = time.time() - t_start
    # Sort by consistency score (not just avg return)
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    output = {
        "experiment": "EXP-600-phase5",
        "description": "Consistency sweep: W=$5, DTE=30, min_credit=5%, OTM 2-5%",
        "ticker": ticker,
        "years": years,
        "total_combos": total,
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_v5_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Leaderboard
    print(f"\n{'='*90}")
    print(f"  LEADERBOARD — Top 20 by Consistency Score")
    print(f"{'='*90}")
    print(f"  {'Rk':<3} {'ID':<7} {'Score':>5} {'Avg':>7} {'Trds':>5} {'Min/Y':>5} "
          f"{'Yr+':>4} {'WR':>5} {'DD':>7} {'Comp$':>10}  Config")
    print(f"  {'-'*95}")
    for rank, r in enumerate(results[:20], 1):
        p = r["params"]
        reg = p.get("regime_mode", "combo")[:4]
        cfg = (f"OTM={p.get('otm_pct',0.03)*100:.0f}% "
               f"PT={p['profit_target']}% SL={p['stop_loss_multiplier']}x "
               f"risk={p['max_risk_per_trade']}% {reg}")
        flag = " ***" if "TARGET" in str(r.get("consistency_score", 0)) else ""
        # Check if target
        yp = int(r["years_profitable"].split("/")[0])
        yt = int(r["years_profitable"].split("/")[1])
        if yp == yt and r["avg_return"] > 5 and r["min_trades_year"] >= 30:
            flag = " ***TARGET***"
        elif yp == yt:
            flag = " (6/6)"
        print(f"  {rank:<3} {r['combo_id']:<7} {r['consistency_score']:>5.0f} "
              f"{r['avg_return']:>+6.1f}% {r['total_trades']:>5} {r['min_trades_year']:>5} "
              f"{r['years_profitable']:>4} {r['avg_win_rate']:>4.0f}% {r['worst_drawdown']:>6.1f}% "
              f"${r['compound_final_capital']:>9,}  {cfg}{flag}")

    # Count targets
    targets = [r for r in results
               if int(r["years_profitable"].split("/")[0]) == int(r["years_profitable"].split("/")[1])
               and r["avg_return"] > 5
               and r["min_trades_year"] >= 30]
    good = [r for r in results
            if int(r["years_profitable"].split("/")[0]) == int(r["years_profitable"].split("/")[1])
            and r["avg_return"] > 0]

    print(f"\n  TARGET (6/6 + >5% avg + >30 trades/yr): {len(targets)}/{total}")
    print(f"  GOOD (6/6 profitable): {len(good)}/{total}")
    print(f"  Positive avg return: {sum(1 for r in results if r['avg_return'] > 0)}/{total}")
    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {output_path}")
    print(f"{'='*90}\n")

    return results


def main():
    # Phase 5 grid: focused consistency sweep
    # 4 OTM × 3 risk × 3 PT × 2 regime = 72 combos
    param_grid = {
        "target_dte":          [30],
        "spread_width":        [5],
        "otm_pct":             [0.02, 0.03, 0.04, 0.05],
        "profit_target":       [40, 50, 65],
        "stop_loss_multiplier":[2.5],         # Phase 4 winner
        "max_risk_per_trade":  [3.0, 4.0, 5.0],
        "direction":           ["both"],
        "regime_mode":         ["combo", "none"],
        "min_credit_pct":      [5],           # Lowered from 10% (Phase 4 smoking gun)
    }

    run_sweep(param_grid, YEARS)


if __name__ == "__main__":
    main()
