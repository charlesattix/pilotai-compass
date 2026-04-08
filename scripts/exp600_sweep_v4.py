#!/usr/bin/env python3
"""
EXP-600 Phase 4: Independent-year sweep with sensible position caps.

Phase 2 bug: max_contracts=10 made 5%/8.5% risk identical (both capped at 10).
Phase 3 disaster: max_contracts=999 → RUIN events in 2020 (capital goes -$400K).

Phase 4 fix:
  - max_contracts=25 (enough headroom to differentiate 2%/5%/8.5% risk)
  - Independent years: reset to $100K each year (no cascade of losses)
  - Also track compound mode separately for final-capital comparison
  - drawdown_cb_pct=25 (circuit breaker to prevent total wipeout)

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
    return {
        "strategy": {
            "target_delta":        0.12,
            "use_delta_selection": params.get("use_delta_selection", False),
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
            "max_contracts":        params.get("max_contracts", 25),
            "max_positions":        50,
            "drawdown_cb_pct":      25,
        },
        "backtest": {
            "starting_capital":   starting_capital,
            "commission_per_contract": 0.65,
            "slippage":           0.05,
            "exit_slippage":      0.10,
            "compound":           False,  # Independent years
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
    print(f"  EXP-600 Phase 4: Independent-Year Sweep (max_contracts=25)")
    print(f"  {total} combos × {len(years)} years = {total * len(years)} backtests")
    print(f"  Each year: fresh $100K | compound=False | drawdown_cb=25%")
    print(f"{'='*80}\n")

    results = []
    t_start = time.time()

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        combo_id = f"P{i:04d}"
        label = (f"DTE={params['target_dte']} W=${params['spread_width']} "
                 f"OTM={params.get('otm_pct', 0.03)*100:.0f}% "
                 f"PT={params['profit_target']}% SL={params['stop_loss_multiplier']}x "
                 f"risk={params['max_risk_per_trade']}% dir={params['direction']}")
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
        years_profitable = sum(1 for r in rets if r > 0)
        worst_dd = min(dds) if dds else 0
        avg_wr = sum(wrs) / len(wrs) if wrs else 0

        # Compute compounded final capital (independent years multiplied)
        compound_cap = 100_000
        for yr_key in [str(y) for y in sorted(years)]:
            if yr_key in year_results and "error" not in year_results[yr_key]:
                compound_cap *= (1 + year_results[yr_key]["return_pct"] / 100)
        compound_return = (compound_cap - 100_000) / 100_000 * 100

        summary = {
            "combo_id": combo_id,
            "params": params,
            "avg_return": round(avg_ret, 2),
            "total_trades": total_trades,
            "years_profitable": f"{years_profitable}/{len(rets)}",
            "worst_drawdown": round(worst_dd, 2),
            "avg_win_rate": round(avg_wr, 1),
            "compound_final_capital": round(compound_cap),
            "compound_return_pct": round(compound_return, 1),
            "by_year": year_results,
        }
        results.append(summary)

        flag = "GOOD" if avg_ret > 10 and years_profitable >= 4 else ("OK" if avg_ret > 0 else "BAD")
        print(f"    → AVG {avg_ret:+.1f}%, {total_trades} trades, "
              f"{years_profitable}/{len(rets)} yrs+, DD {worst_dd:.1f}%, "
              f"comp ${compound_cap:,.0f}  [{flag}]\n")

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["avg_return"], reverse=True)

    output = {
        "experiment": "EXP-600-phase4",
        "description": "Independent-year sweep, max_contracts=25, drawdown_cb=25%",
        "ticker": ticker,
        "years": years,
        "total_combos": total,
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_v4_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Leaderboard
    print(f"\n{'='*80}")
    print(f"  LEADERBOARD — Top 20")
    print(f"{'='*80}")
    print(f"  {'Rk':<3} {'ID':<7} {'Avg':>7} {'Trds':>5} {'Yr+':>4} {'WR':>5} {'DD':>7} {'Comp$':>10}  Config")
    print(f"  {'-'*90}")
    for rank, r in enumerate(results[:20], 1):
        p = r["params"]
        cfg = (f"DTE={p['target_dte']} W=${p['spread_width']} "
               f"OTM={p.get('otm_pct',0.03)*100:.0f}% "
               f"PT={p['profit_target']}% SL={p['stop_loss_multiplier']}x "
               f"risk={p['max_risk_per_trade']}%")
        print(f"  {rank:<3} {r['combo_id']:<7} {r['avg_return']:>+6.1f}% {r['total_trades']:>5} "
              f"{r['years_profitable']:>4} {r['avg_win_rate']:>4.0f}% {r['worst_drawdown']:>6.1f}% "
              f"${r['compound_final_capital']:>9,}  {cfg}")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {output_path}")
    print(f"{'='*80}\n")

    return results


def main():
    # Phase 4 grid: 2×2×2×3×3×3 = 216 combos
    # direction="both" and "bull_put" produce IDENTICAL results (regime_mode=combo
    # routes all trades the same way), so only test "both".
    # max_contracts=25 ensures risk levels differentiate.
    param_grid = {
        "target_dte":          [30, 45],
        "spread_width":        [5, 10],
        "otm_pct":             [0.03, 0.05],
        "profit_target":       [50, 65, 80],
        "stop_loss_multiplier":[1.5, 2.0, 2.5],
        "max_risk_per_trade":  [2.0, 5.0, 8.5],
        "direction":           ["both"],
    }

    run_sweep(param_grid, YEARS)


if __name__ == "__main__":
    main()
