#!/usr/bin/env python3
"""
EXP-600 Phase 7: Direction-Aware Sweep (regime_mode='none', MA trend filter).

Phase 5 (v5) showed regime_mode='none' with direction='both' gets +42% avg but
0/36 configs are 6/6 profitable — wild directional swings (2022 +248%, 2021 -20%).

The backtester already implements MA-based trend filtering when regime_mode='none':
  - bull_put: only enters when price >= MA50 (bullish)
  - bear_call: only enters when price <= MA50 (bearish)
  - both: applies BOTH filters — puts above MA, calls below MA ("trend-following")

This sweep tests each direction separately AND combined, at DTE=30 and DTE=45,
to find which direction mode achieves consistency.

Grid: 3 dir × 2 DTE × 2 OTM × 3 PT × 2 risk = 72 combos × 6 years = 432 backtests

Target: 5/6 or 6/6 profitable years with >20 trades/year.

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

# Direction labels for readability
DIR_LABELS = {
    "bull_put": "puts-only",
    "bear_call": "calls-only",
    "both": "trend-MA",
}


def build_config(params, starting_capital=100_000):
    target_dte = params["target_dte"]
    min_dte = max(10, target_dte - 10)
    return {
        "strategy": {
            "target_delta":        0.12,
            "use_delta_selection": False,
            "target_dte":          target_dte,
            "min_dte":             min_dte,
            "spread_width":        params["spread_width"],
            "min_credit_pct":      params.get("min_credit_pct", 5),
            "direction":           params["direction"],
            "trend_ma_period":     params.get("trend_ma_period", 50),
            "regime_mode":         "none",  # NO combo regime — MA filter only
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

    print(f"\n{'='*90}")
    print(f"  EXP-600 Phase 7: Direction-Aware Sweep (regime_mode=none, MA50 filter)")
    print(f"  {total} combos × {len(years)} years = {total * len(years)} backtests")
    print(f"  Each year: fresh $100K | compound=False | drawdown_cb=25%")
    print(f"  Directions: bull_put (puts above MA), bear_call (calls below MA), both (trend-follow)")
    print(f"  TARGET: 5/6+ yrs profitable, >20 trades/yr")
    print(f"{'='*90}\n")

    results = []
    t_start = time.time()

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        combo_id = f"D{i:04d}"
        dir_label = DIR_LABELS.get(params["direction"], params["direction"])
        label = (f"DTE={params['target_dte']} OTM={params.get('otm_pct', 0.03)*100:.0f}% "
                 f"PT={params['profit_target']}% risk={params['max_risk_per_trade']}% "
                 f"dir={dir_label}")
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

        # Consistency score: heavily rewards profitable years
        consistency = (
            years_profitable / len(rets) * 50  # up to 50 pts for profitability
            + min(1, min_trades_yr / 20) * 25   # up to 25 pts for trade flow (>20/yr)
            + min(1, max(0, avg_ret) / 10) * 25 # up to 25 pts for avg return
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

        yp = years_profitable
        yt = len(rets)
        if yp >= 5 and avg_ret > 5 and min_trades_yr >= 20:
            flag = "TARGET"
        elif yp >= 5 and avg_ret > 0:
            flag = "GOOD"
        elif avg_ret > 0:
            flag = "OK"
        else:
            flag = "BAD"
        print(f"    → AVG {avg_ret:+.1f}%, {total_trades} trades (min {min_trades_yr}/yr), "
              f"{yp}/{yt} yrs+, DD {worst_dd:.1f}%, "
              f"score {consistency:.0f}  [{flag}]\n")

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    output = {
        "experiment": "EXP-600-phase7",
        "description": "Direction-aware sweep: bull_put vs bear_call vs trend-MA, regime_mode=none",
        "ticker": ticker,
        "years": years,
        "total_combos": total,
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_v6_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Leaderboard
    print(f"\n{'='*90}")
    print(f"  LEADERBOARD — Top 25 by Consistency Score")
    print(f"{'='*90}")
    print(f"  {'Rk':<3} {'ID':<7} {'Score':>5} {'Avg':>7} {'Trds':>5} {'Min/Y':>5} "
          f"{'Yr+':>4} {'WR':>5} {'DD':>7} {'Comp$':>10}  Config")
    print(f"  {'-'*100}")
    for rank, r in enumerate(results[:25], 1):
        p = r["params"]
        dir_label = DIR_LABELS.get(p["direction"], p["direction"])
        cfg = (f"DTE={p['target_dte']} OTM={p.get('otm_pct',0.03)*100:.0f}% "
               f"PT={p['profit_target']}% risk={p['max_risk_per_trade']}% "
               f"{dir_label}")
        yp = int(r["years_profitable"].split("/")[0])
        yt = int(r["years_profitable"].split("/")[1])
        flag = ""
        if yp >= 5 and r["avg_return"] > 5 and r["min_trades_year"] >= 20:
            flag = " ***TARGET***"
        elif yp >= 5:
            flag = f" ({yp}/6)"
        print(f"  {rank:<3} {r['combo_id']:<7} {r['consistency_score']:>5.0f} "
              f"{r['avg_return']:>+6.1f}% {r['total_trades']:>5} {r['min_trades_year']:>5} "
              f"{r['years_profitable']:>4} {r['avg_win_rate']:>4.0f}% {r['worst_drawdown']:>6.1f}% "
              f"${r['compound_final_capital']:>9,}  {cfg}{flag}")

    # Summary by direction
    print(f"\n  === Direction Summary ===")
    for d in ["bull_put", "bear_call", "both"]:
        subset = [r for r in results if r["params"]["direction"] == d]
        pos = [r for r in subset if r["avg_return"] > 0]
        five_six = sum(1 for r in subset if int(r["years_profitable"].split("/")[0]) >= 5)
        six_six = sum(1 for r in subset if r["years_profitable"].startswith("6"))
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        print(f"  {DIR_LABELS.get(d, d):12s}: {len(pos):>2}/{len(subset)} positive, "
              f"avg={avg_all:>+6.1f}%, 5/6+={five_six}, 6/6={six_six}")

    # Summary by DTE
    print(f"\n  === DTE Summary ===")
    for dte in [30, 45]:
        subset = [r for r in results if r["params"]["target_dte"] == dte]
        pos = [r for r in subset if r["avg_return"] > 0]
        five_six = sum(1 for r in subset if int(r["years_profitable"].split("/")[0]) >= 5)
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        print(f"  DTE={dte}: {len(pos):>2}/{len(subset)} positive, avg={avg_all:>+6.1f}%, 5/6+={five_six}")

    # Counts
    targets = [r for r in results
               if int(r["years_profitable"].split("/")[0]) >= 5
               and r["avg_return"] > 5
               and r["min_trades_year"] >= 20]
    good = [r for r in results
            if int(r["years_profitable"].split("/")[0]) >= 5
            and r["avg_return"] > 0]

    print(f"\n  TARGET (5/6+ yrs + >5% avg + >20 trades/yr): {len(targets)}/{total}")
    print(f"  GOOD (5/6+ profitable): {len(good)}/{total}")
    print(f"  Positive avg return: {sum(1 for r in results if r['avg_return'] > 0)}/{total}")
    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {output_path}")
    print(f"{'='*90}\n")

    return results


def main():
    # Phase 7 grid: direction-aware sweep
    # 3 dir × 2 DTE × 2 OTM × 3 PT × 2 risk = 72 combos
    param_grid = {
        "direction":           ["bull_put", "bear_call", "both"],
        "target_dte":          [30, 45],
        "spread_width":        [5],
        "otm_pct":             [0.03, 0.05],
        "profit_target":       [40, 50, 60],
        "stop_loss_multiplier":[2.5],
        "max_risk_per_trade":  [3.0, 4.0],
        "min_credit_pct":      [5],
    }

    run_sweep(param_grid, YEARS)


if __name__ == "__main__":
    main()
