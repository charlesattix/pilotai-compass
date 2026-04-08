#!/usr/bin/env python3
"""
EXP-600 Phase 8: Beat P0141 — Faster Trend Filters + Direction Tuning.

P0141 is the ONLY config across 500+ combos (7 phases) that achieves 6/6 profitable years:
  DTE=45, W=$5, OTM=5%, PT=50%, SL=2.5x, risk=2%, regime_mode=combo
  → +3.7% avg, 6/6 profitable, -13.8% DD, 240 trades

The problem: regime_mode='combo' bypasses MA entirely — ComboRegimeDetector decides
direction with a broken BULL-biased vote (2/3 for BULL vs 3/3 for BEAR).
Phases 6-7 showed regime_mode='none' with MA50 is too slow (lags in trending markets).

Hypothesis: faster MAs (EMA9, EMA20, MA20) may react quickly enough to prevent
wrong-direction entries while keeping trade flow high.

Grid: 4 MA types × 2 dir × 3 risk × 2 DTE = 48 combos + P0141 control = 49
  MA: SMA50 (baseline), SMA20, EMA20, EMA9  (all with regime_mode='none')
  Dir: bull_put (puts-only above MA), both (trend-following: puts above + calls below)
  Risk: 2%, 3%, 4%
  DTE: 30, 45
  Fixed: W=$5, OTM=5%, PT=50%, SL=2.5x, min_credit_pct=5%

Control: P0141 exact config (regime_mode=combo, direction=both, MA50)

Target: beat +3.7% avg while maintaining 5/6 or 6/6 profitable years.

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

# MA type labels for readability
MA_LABELS = {
    ("sma", 50): "SMA50",
    ("sma", 20): "SMA20",
    ("ema", 20): "EMA20",
    ("ema", 9):  "EMA9",
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
            "trend_ma_period":     params["trend_ma_period"],
            "trend_ma_type":       params["trend_ma_type"],
            "regime_mode":         params["regime_mode"],
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
    otm_pct = params.get("otm_pct", 0.05)
    bt = Backtester(config, historical_data=hd, otm_pct=otm_pct)
    result = bt.run_backtest(ticker, start, end)
    result = result or {}
    result["year"] = year
    return result


def make_combo_list():
    """Build the list of param dicts: 48 experimental combos + 1 P0141 control."""
    combos = []

    # P0141 control (regime_mode=combo, direction=both, MA50 bypassed)
    combos.append({
        "label": "P0141-CONTROL",
        "target_dte":          45,
        "spread_width":        5,
        "otm_pct":             0.05,
        "profit_target":       50,
        "stop_loss_multiplier": 2.5,
        "max_risk_per_trade":  2.0,
        "direction":           "both",
        "trend_ma_period":     50,
        "trend_ma_type":       "sma",
        "regime_mode":         "combo",
        "min_credit_pct":      5,
    })

    # Experimental grid: 4 MA × 2 dir × 3 risk × 2 DTE = 48
    ma_configs = [
        ("sma", 50),
        ("sma", 20),
        ("ema", 20),
        ("ema", 9),
    ]
    directions = ["bull_put", "both"]
    risks = [2.0, 3.0, 4.0]
    dtes = [30, 45]

    for (ma_type, ma_period), direction, risk, dte in itertools.product(
        ma_configs, directions, risks, dtes
    ):
        ma_label = MA_LABELS.get((ma_type, ma_period), f"{ma_type.upper()}{ma_period}")
        combos.append({
            "label": f"{ma_label} dir={direction} risk={risk}% DTE={dte}",
            "target_dte":          dte,
            "spread_width":        5,
            "otm_pct":             0.05,
            "profit_target":       50,
            "stop_loss_multiplier": 2.5,
            "max_risk_per_trade":  risk,
            "direction":           direction,
            "trend_ma_period":     ma_period,
            "trend_ma_type":       ma_type,
            "regime_mode":         "none",  # MA filter active (not bypassed)
            "min_credit_pct":      5,
        })

    return combos


def run_sweep(combos, years, ticker="SPY"):
    total = len(combos)
    print(f"\n{'='*90}")
    print(f"  EXP-600 Phase 8: Beat P0141 — Faster Trend Filters")
    print(f"  {total} combos × {len(years)} years = {total * len(years)} backtests")
    print(f"  Each year: fresh $100K | compound=False | drawdown_cb=25%")
    print(f"  Fixed: W=$5, OTM=5%, PT=50%, SL=2.5x, min_credit=5%")
    print(f"  P0141 baseline: +3.7% avg, 6/6 profitable, -13.8% DD")
    print(f"  TARGET: beat +3.7% avg with 5/6 or 6/6 profitable years")
    print(f"{'='*90}\n")

    results = []
    t_start = time.time()

    for i, params in enumerate(combos):
        combo_id = f"F{i:04d}"
        label = params["label"]
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
        wrs = [yr["win_rate"] for yr in year_results.values()
               if "error" not in yr and yr["trades"] > 0]

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
            "label": label,
            "params": {k: v for k, v in params.items() if k != "label"},
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
        if yp >= 5 and avg_ret > 3.7:
            flag = "BEATS P0141"
        elif yp >= 5 and avg_ret > 0:
            flag = "CONSISTENT"
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
        "experiment": "EXP-600-phase8",
        "description": "Beat P0141: faster trend filters (EMA9/EMA20/SMA20) + direction tuning",
        "ticker": ticker,
        "years": years,
        "total_combos": total,
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "p0141_baseline": "+3.7% avg, 6/6 profitable, -13.8% DD, 240 trades",
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_v7_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Find P0141 control result
    control = next((r for r in results if "CONTROL" in r["label"]), None)
    control_avg = control["avg_return"] if control else 3.7

    # Leaderboard
    print(f"\n{'='*90}")
    print(f"  LEADERBOARD — Top 25 by Consistency Score")
    print(f"  P0141 baseline: +3.7% avg, 6/6 profitable")
    print(f"{'='*90}")
    print(f"  {'Rk':<3} {'ID':<7} {'Score':>5} {'Avg':>7} {'Trds':>5} {'Min/Y':>5} "
          f"{'Yr+':>4} {'WR':>5} {'DD':>7} {'Comp$':>10}  Config")
    print(f"  {'-'*105}")
    for rank, r in enumerate(results[:25], 1):
        yp = int(r["years_profitable"].split("/")[0])
        yt = int(r["years_profitable"].split("/")[1])
        flag = ""
        if yp >= 5 and r["avg_return"] > control_avg:
            flag = " ***BEATS P0141***"
        elif yp >= 5:
            flag = f" ({yp}/6)"
        print(f"  {rank:<3} {r['combo_id']:<7} {r['consistency_score']:>5.0f} "
              f"{r['avg_return']:>+6.1f}% {r['total_trades']:>5} {r['min_trades_year']:>5} "
              f"{r['years_profitable']:>4} {r['avg_win_rate']:>4.0f}% {r['worst_drawdown']:>6.1f}% "
              f"${r['compound_final_capital']:>9,}  {r['label']}{flag}")

    # Summary by MA type
    print(f"\n  === MA Type Summary ===")
    for ma_key, ma_label in MA_LABELS.items():
        ma_type, ma_period = ma_key
        subset = [r for r in results
                  if r["params"].get("trend_ma_type") == ma_type
                  and r["params"].get("trend_ma_period") == ma_period
                  and r["params"].get("regime_mode") == "none"]
        if not subset:
            continue
        pos = [r for r in subset if r["avg_return"] > 0]
        five_six = sum(1 for r in subset if int(r["years_profitable"].split("/")[0]) >= 5)
        six_six = sum(1 for r in subset if r["years_profitable"].startswith("6"))
        beats = sum(1 for r in subset
                    if int(r["years_profitable"].split("/")[0]) >= 5
                    and r["avg_return"] > control_avg)
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        print(f"  {ma_label:8s}: {len(pos):>2}/{len(subset)} positive, "
              f"avg={avg_all:>+6.1f}%, 5/6+={five_six}, 6/6={six_six}, beats={beats}")

    # Summary by direction
    print(f"\n  === Direction Summary ===")
    for d in ["bull_put", "both"]:
        subset = [r for r in results
                  if r["params"].get("direction") == d
                  and r["params"].get("regime_mode") == "none"]
        if not subset:
            continue
        pos = [r for r in subset if r["avg_return"] > 0]
        five_six = sum(1 for r in subset if int(r["years_profitable"].split("/")[0]) >= 5)
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        print(f"  {d:12s}: {len(pos):>2}/{len(subset)} positive, avg={avg_all:>+6.1f}%, 5/6+={five_six}")

    # Summary by DTE
    print(f"\n  === DTE Summary ===")
    for dte in [30, 45]:
        subset = [r for r in results
                  if r["params"].get("target_dte") == dte
                  and r["params"].get("regime_mode") == "none"]
        if not subset:
            continue
        pos = [r for r in subset if r["avg_return"] > 0]
        five_six = sum(1 for r in subset if int(r["years_profitable"].split("/")[0]) >= 5)
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        print(f"  DTE={dte}: {len(pos):>2}/{len(subset)} positive, avg={avg_all:>+6.1f}%, 5/6+={five_six}")

    # Configs that beat P0141
    beaters = [r for r in results
               if int(r["years_profitable"].split("/")[0]) >= 5
               and r["avg_return"] > control_avg]

    print(f"\n  {'='*60}")
    print(f"  CONFIGS THAT BEAT P0141 (+{control_avg:.1f}% avg, 5/6+ yrs): {len(beaters)}/{total}")
    print(f"  {'='*60}")
    if beaters:
        for r in beaters[:10]:
            yp = r["years_profitable"]
            print(f"    {r['combo_id']} {r['label']}")
            print(f"      → avg={r['avg_return']:+.1f}%, {r['total_trades']} trades, "
                  f"{yp} yrs+, DD={r['worst_drawdown']:.1f}%")
            for yr in sorted(r["by_year"].keys()):
                yr_data = r["by_year"][yr]
                if "error" not in yr_data:
                    print(f"        {yr}: {yr_data['return_pct']:+.1f}% "
                          f"({yr_data['trades']} trades, WR {yr_data['win_rate']:.0f}%)")
            print()
    else:
        print(f"    None. P0141 remains champion.")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {output_path}")
    print(f"{'='*90}\n")

    return results


def main():
    combos = make_combo_list()
    print(f"  Built {len(combos)} combos ({len(combos) - 1} experimental + 1 control)")
    run_sweep(combos, YEARS)


if __name__ == "__main__":
    main()
