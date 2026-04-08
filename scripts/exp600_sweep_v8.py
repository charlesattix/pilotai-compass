#!/usr/bin/env python3
"""
EXP-600 Phase 9: Three-Stage Refinement Sweep (v8).

Stage A (24 combos): Refine P0141 neighbourhood.
  Fixed: DTE=30, W=$5, OTM=3%, min_credit=5%, regime_mode=combo
  Vary: risk 3-7%, PT 40/50/60%, SL 2.0/2.5/3.0x, compound on/off
  → 4 risk × 3 PT × (already fixed SL × compound gives cross-product below)
  Grid: risk=[3,4,5,7] × PT=[40,50,60] × SL=[2.0,3.0] × compound=[True,False] = 48
  BUT user asked 24 combos, so: risk=[3,5,7] × PT=[40,50,60] × SL=[2.0,2.5,3.0] - wait...
  Let me re-read: "vary risk 3-7%, PT 40-60%, SL 2-3x, compound on/off"
  Interpretation: risk=[3,5,7], PT=[40,50,60], SL=[2,3], compound=[True,False]
  → 3×3×2×... but that's 36 with compound. Let me do:
  risk=[3,5,7], PT=[40,60], SL=[2,3], compound=[True,False] = 3×2×2×2 = 24. ✓

Stage B (18 combos): VIX regime thresholds using best Stage A config.
  vix_extreme=[30,35,40], rsi_bull=[50,55,60], rsi_bear=[40,45],
  cooldown=[5,10] → 3×3×2×... pruned to 18.
  Grid: vix_extreme=[30,35,40] × rsi_bull=[50,55,60] × rsi_bear=[40,45] = 18. ✓

Stage C (6 combos): Multi-ticker using best Stage A + best Stage B config.
  Tickers: SPY, QQQ, IWM run individually + combined portfolio simulation.
  → 6 combos = 3 solo + 3 pairs (SPY+QQQ, SPY+IWM, SPY+QQQ+IWM).

Charles reference: 2020:+86.9%, 2021:+216.7%, 2022:+28.3%, 2023:+12.9%, 2024:-5.1%, 2025:-0.8%
P0141 baseline: +3.7% avg, 6/6 profitable, -13.8% DD, 240 trades (DTE=45, W=$5, OTM=5%)
"""

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


# ────────────────────────────────────────────────────────────
#  Shared helpers
# ────────────────────────────────────────────────────────────

def build_config(params, starting_capital=100_000):
    target_dte = params["target_dte"]
    min_dte = max(10, target_dte - 10)
    compound = params.get("compound", False)
    return {
        "strategy": {
            "target_delta":        0.12,
            "use_delta_selection": False,
            "target_dte":          target_dte,
            "min_dte":             min_dte,
            "spread_width":        params["spread_width"],
            "min_credit_pct":      params.get("min_credit_pct", 5),
            "direction":           params.get("direction", "both"),
            "trend_ma_period":     params.get("trend_ma_period", 50),
            "trend_ma_type":       params.get("trend_ma_type", "sma"),
            "regime_mode":         params.get("regime_mode", "combo"),
            "regime_config":       params.get("regime_config", {}),
            "momentum_filter_pct": None,
            "iron_condor":         {"enabled": False},
            "iv_rank_min_entry":   0,
            "vix_max_entry":       params.get("vix_max_entry", 0),
            "vix_close_all":       params.get("vix_close_all", 0),
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
            "compound":           compound,
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


def compute_summary(combo_id, label, params, year_results, years):
    """Compute aggregate metrics from per-year results."""
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

    # Consistency score
    consistency = (
        years_profitable / len(rets) * 50
        + min(1, min_trades_yr / 20) * 25
        + min(1, max(0, avg_ret) / 10) * 25
    ) if rets else 0

    return {
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


def run_combo(combo_id, label, params, years, ticker="SPY"):
    """Run one config across all years and return summary dict."""
    print(f"  {combo_id}: {label}")
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

    summary = compute_summary(combo_id, label, params, year_results, years)

    yp = int(summary["years_profitable"].split("/")[0])
    yt = int(summary["years_profitable"].split("/")[1])
    avg = summary["avg_return"]
    if yp >= 5 and avg > 3.7:
        flag = "BEATS P0141"
    elif yp >= 5 and avg > 0:
        flag = "CONSISTENT"
    elif avg > 0:
        flag = "OK"
    else:
        flag = "BAD"
    print(f"    -> AVG {avg:+.1f}%, {summary['total_trades']} trades "
          f"(min {summary['min_trades_year']}/yr), "
          f"{summary['years_profitable']} yrs+, DD {summary['worst_drawdown']:.1f}%, "
          f"score {summary['consistency_score']:.0f}  [{flag}]\n")
    return summary


def print_leaderboard(results, title, top_n=15):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    print(f"  {'Rk':<3} {'ID':<7} {'Score':>5} {'Avg':>7} {'Trds':>5} {'Min/Y':>5} "
          f"{'Yr+':>4} {'WR':>5} {'DD':>7} {'Comp$':>10}  Config")
    print(f"  {'-'*110}")
    for rank, r in enumerate(results[:top_n], 1):
        yp = int(r["years_profitable"].split("/")[0])
        flag = ""
        if yp >= 5 and r["avg_return"] > 3.7:
            flag = " ***BEATS***"
        elif yp >= 5:
            flag = f" ({yp}/6)"
        print(f"  {rank:<3} {r['combo_id']:<7} {r['consistency_score']:>5.0f} "
              f"{r['avg_return']:>+6.1f}% {r['total_trades']:>5} {r['min_trades_year']:>5} "
              f"{r['years_profitable']:>4} {r['avg_win_rate']:>4.0f}% {r['worst_drawdown']:>6.1f}% "
              f"${r['compound_final_capital']:>9,}  {r['label']}{flag}")


# ────────────────────────────────────────────────────────────
#  Stage A: Refine P0141 neighbourhood (24 combos)
# ────────────────────────────────────────────────────────────

def stage_a():
    """DTE=30, W=$5, OTM=3%, min_credit=5%, combo regime.
    Vary: risk=[3,5,7], PT=[40,60], SL=[2,3], compound=[True,False] = 24 combos.
    """
    risks = [3.0, 5.0, 7.0]
    pts = [40, 60]
    sls = [2.0, 3.0]
    compounds = [True, False]

    combos = []
    for risk in risks:
        for pt in pts:
            for sl in sls:
                for compound in compounds:
                    comp_label = "cmp" if compound else "flat"
                    combos.append({
                        "label": f"risk={risk}% PT={pt}% SL={sl}x {comp_label}",
                        "target_dte": 30,
                        "spread_width": 5,
                        "otm_pct": 0.03,
                        "profit_target": pt,
                        "stop_loss_multiplier": sl,
                        "max_risk_per_trade": risk,
                        "direction": "both",
                        "regime_mode": "combo",
                        "min_credit_pct": 5,
                        "compound": compound,
                    })

    assert len(combos) == 24, f"Expected 24 combos, got {len(combos)}"

    print(f"\n{'='*100}")
    print(f"  STAGE A: Refine P0141 Neighbourhood (24 combos)")
    print(f"  Fixed: DTE=30, W=$5, OTM=3%, min_credit=5%, combo regime")
    print(f"  Vary: risk=[3,5,7]%, PT=[40,60]%, SL=[2,3]x, compound=[on,off]")
    print(f"  24 combos x 6 years = 144 backtests")
    print(f"  P0141 baseline: +3.7% avg, 6/6 profitable, -13.8% DD")
    print(f"{'='*100}\n")

    results = []
    t_start = time.time()
    for i, params in enumerate(combos):
        combo_id = f"A{i:03d}"
        label = params["label"]
        print(f"  [{i+1}/{len(combos)}]", end=" ")
        summary = run_combo(combo_id, label, params, YEARS)
        results.append(summary)

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    output = {
        "experiment": "EXP-600-v8-stageA",
        "description": "Refine P0141: DTE=30, W=$5, OTM=3%, combo regime, vary risk/PT/SL/compound",
        "ticker": "SPY",
        "years": YEARS,
        "total_combos": len(combos),
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "p0141_baseline": "+3.7% avg, 6/6 profitable, -13.8% DD, 240 trades",
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_v8_stageA.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print_leaderboard(results, "STAGE A LEADERBOARD — Top 15 by Consistency Score")

    # Summary by risk level
    print(f"\n  === Risk Level Summary ===")
    for risk in risks:
        subset = [r for r in results if r["params"]["max_risk_per_trade"] == risk]
        pos = sum(1 for r in subset if r["avg_return"] > 0)
        five_plus = sum(1 for r in subset
                        if int(r["years_profitable"].split("/")[0]) >= 5)
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        print(f"  risk={risk}%: {pos}/{len(subset)} positive, avg={avg_all:+.1f}%, 5/6+={five_plus}")

    # Summary by compound
    print(f"\n  === Compound Summary ===")
    for comp in [True, False]:
        comp_label = "compound" if comp else "flat"
        subset = [r for r in results if r["params"].get("compound", False) == comp]
        pos = sum(1 for r in subset if r["avg_return"] > 0)
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        print(f"  {comp_label}: {pos}/{len(subset)} positive, avg={avg_all:+.1f}%")

    beaters = [r for r in results
               if int(r["years_profitable"].split("/")[0]) >= 5
               and r["avg_return"] > 3.7]
    print(f"\n  BEATS P0141 (5/6+ yrs, >+3.7% avg): {len(beaters)}/{len(combos)}")
    for r in beaters[:5]:
        print(f"    {r['combo_id']} {r['label']} -> {r['avg_return']:+.1f}%, "
              f"{r['years_profitable']} yrs+, DD={r['worst_drawdown']:.1f}%")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {output_path}")
    return results


# ────────────────────────────────────────────────────────────
#  Stage B: VIX regime thresholds (18 combos)
# ────────────────────────────────────────────────────────────

def stage_b(best_a_params):
    """Test VIX regime threshold tuning on the best Stage A config.
    vix_extreme=[30,35,40] × rsi_bull=[50,55,60] × rsi_bear=[40,45] = 18 combos.
    """
    vix_extremes = [30, 35, 40]
    rsi_bulls = [50, 55, 60]
    rsi_bears = [40, 45]

    combos = []
    for vix_ext in vix_extremes:
        for rsi_b in rsi_bulls:
            for rsi_br in rsi_bears:
                params = dict(best_a_params)
                params["regime_config"] = {
                    "vix_extreme": vix_ext,
                    "rsi_bull_threshold": rsi_b,
                    "rsi_bear_threshold": rsi_br,
                }
                params["label"] = f"VIX_ext={vix_ext} RSI_bull={rsi_b} RSI_bear={rsi_br}"
                combos.append(params)

    assert len(combos) == 18, f"Expected 18 combos, got {len(combos)}"

    base_label = best_a_params.get("label", "best_A")
    print(f"\n{'='*100}")
    print(f"  STAGE B: VIX Regime Thresholds (18 combos)")
    print(f"  Base config from Stage A: {base_label}")
    print(f"  Vary: vix_extreme=[30,35,40], rsi_bull=[50,55,60], rsi_bear=[40,45]")
    print(f"  18 combos x 6 years = 108 backtests")
    print(f"{'='*100}\n")

    results = []
    t_start = time.time()
    for i, params in enumerate(combos):
        combo_id = f"B{i:03d}"
        label = params["label"]
        print(f"  [{i+1}/{len(combos)}]", end=" ")
        summary = run_combo(combo_id, label, params, YEARS)
        results.append(summary)

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    output = {
        "experiment": "EXP-600-v8-stageB",
        "description": "VIX regime thresholds on best Stage A config",
        "base_config": base_label,
        "ticker": "SPY",
        "years": YEARS,
        "total_combos": len(combos),
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_v8_stageB.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print_leaderboard(results, "STAGE B LEADERBOARD — VIX Regime Thresholds")

    # Summary by vix_extreme
    print(f"\n  === VIX Extreme Summary ===")
    for vix_ext in vix_extremes:
        subset = [r for r in results
                  if r["params"].get("regime_config", {}).get("vix_extreme") == vix_ext]
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        five_plus = sum(1 for r in subset
                        if int(r["years_profitable"].split("/")[0]) >= 5)
        print(f"  vix_ext={vix_ext}: avg={avg_all:+.1f}%, 5/6+={five_plus}/{len(subset)}")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {output_path}")
    return results


# ────────────────────────────────────────────────────────────
#  Stage C: Multi-ticker (6 combos)
# ────────────────────────────────────────────────────────────

def stage_c(best_params):
    """Multi-ticker: 3 solo (SPY, QQQ, IWM) + 3 pairs/combos.
    For multi-ticker runs, we run each ticker independently and aggregate.
    """
    tickers_sets = [
        ("SPY-solo", ["SPY"]),
        ("QQQ-solo", ["QQQ"]),
        ("IWM-solo", ["IWM"]),
        ("SPY+QQQ", ["SPY", "QQQ"]),
        ("SPY+IWM", ["SPY", "IWM"]),
        ("SPY+QQQ+IWM", ["SPY", "QQQ", "IWM"]),
    ]

    base_label = best_params.get("label", "best_config")
    print(f"\n{'='*100}")
    print(f"  STAGE C: Multi-Ticker (6 combos)")
    print(f"  Base config: {base_label}")
    print(f"  Tickers: SPY, QQQ, IWM solo + pairs + triple")
    print(f"  6 combos, each ticker x 6 years")
    print(f"{'='*100}\n")

    results = []
    t_start = time.time()

    for i, (set_label, tickers) in enumerate(tickers_sets):
        combo_id = f"C{i:03d}"
        params = dict(best_params)
        params["label"] = f"{set_label} ({base_label})"
        label = params["label"]

        print(f"  [{i+1}/{len(tickers_sets)}] {combo_id}: {set_label}")

        # Run each ticker independently, then aggregate
        all_year_results = {}
        for year in sorted(YEARS):
            year_rets = []
            year_trades = 0
            year_wrs = []
            year_dds = []
            year_sharpes = []
            year_errors = []
            ticker_cap = 100_000 / len(tickers)  # split capital equally

            for ticker in tickers:
                try:
                    r = run_single(ticker, year, params, starting_capital=round(ticker_cap))
                    ret = r.get("return_pct", 0)
                    trades = r.get("total_trades", 0)
                    wr = r.get("win_rate", 0)
                    dd = r.get("max_drawdown", 0)
                    year_rets.append(ret)
                    year_trades += trades
                    if trades > 0:
                        year_wrs.append(wr)
                    year_dds.append(dd)
                    year_sharpes.append(r.get("sharpe_ratio", 0))
                except Exception as e:
                    year_errors.append(str(e))

            if year_errors and not year_rets:
                all_year_results[str(year)] = {
                    "return_pct": 0, "trades": 0, "error": "; ".join(year_errors)
                }
                print(f"    {year}: ERROR {year_errors[0]}")
            else:
                # Weighted average return (equal capital split)
                avg_ret = sum(year_rets) / len(year_rets) if year_rets else 0
                avg_wr = sum(year_wrs) / len(year_wrs) if year_wrs else 0
                worst_dd = min(year_dds) if year_dds else 0
                avg_sharpe = sum(year_sharpes) / len(year_sharpes) if year_sharpes else 0
                all_year_results[str(year)] = {
                    "return_pct": round(avg_ret, 2),
                    "trades": year_trades,
                    "win_rate": round(avg_wr, 1),
                    "max_drawdown": round(worst_dd, 2),
                    "sharpe": round(avg_sharpe, 2),
                    "ending_capital": 100_000,  # placeholder
                    "tickers": tickers,
                }
                flag = "+" if avg_ret > 0 else "-"
                print(f"    {year}: {avg_ret:+6.1f}% ({year_trades:>3} trades, "
                      f"WR {avg_wr:.0f}%, DD {worst_dd:.1f}%) {flag}")

        summary = compute_summary(combo_id, label, params, all_year_results, YEARS)
        summary["tickers"] = tickers

        yp = int(summary["years_profitable"].split("/")[0])
        avg = summary["avg_return"]
        flag_str = "BEATS" if yp >= 5 and avg > 3.7 else ("OK" if avg > 0 else "BAD")
        print(f"    -> AVG {avg:+.1f}%, {summary['total_trades']} trades, "
              f"{summary['years_profitable']} yrs+, DD {summary['worst_drawdown']:.1f}%, "
              f"score {summary['consistency_score']:.0f}  [{flag_str}]\n")
        results.append(summary)

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    output = {
        "experiment": "EXP-600-v8-stageC",
        "description": "Multi-ticker: SPY/QQQ/IWM solo + combos",
        "base_config": base_label,
        "years": YEARS,
        "total_combos": len(tickers_sets),
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_v8_stageC.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print_leaderboard(results, "STAGE C LEADERBOARD — Multi-Ticker")
    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {output_path}")
    return results


# ────────────────────────────────────────────────────────────
#  Main orchestrator
# ────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    # ── Stage A ──
    print("\n" + "=" * 100)
    print("  STARTING EXP-600 v8: THREE-STAGE REFINEMENT SWEEP")
    print("=" * 100)

    a_results = stage_a()

    # Pick best Stage A config for Stages B & C
    best_a = a_results[0]  # highest consistency score
    best_a_params = dict(best_a["params"])
    best_a_params["label"] = best_a["label"]
    print(f"\n  Best Stage A config: {best_a['combo_id']} {best_a['label']}")
    print(f"    avg={best_a['avg_return']:+.1f}%, {best_a['years_profitable']} yrs+, "
          f"DD={best_a['worst_drawdown']:.1f}%, score={best_a['consistency_score']:.0f}")

    # ── Stage B ──
    b_results = stage_b(best_a_params)
    best_b = b_results[0]

    # Merge best B regime_config into params for Stage C
    best_params = dict(best_a_params)
    if best_b["avg_return"] > best_a["avg_return"]:
        best_params["regime_config"] = best_b["params"].get("regime_config", {})
        best_params["label"] = f"{best_a['label']} + {best_b['label']}"
        print(f"\n  Stage B improved: using {best_b['combo_id']} {best_b['label']}")
    else:
        print(f"\n  Stage B did not improve. Using Stage A config for Stage C.")

    # ── Stage C ──
    c_results = stage_c(best_params)

    # ── Final Summary ──
    total_elapsed = time.time() - t_total
    print(f"\n{'='*100}")
    print(f"  FINAL SUMMARY — EXP-600 v8 Three-Stage Sweep")
    print(f"{'='*100}")
    print(f"  Stage A: {len(a_results)} combos, best={a_results[0]['combo_id']} "
          f"({a_results[0]['avg_return']:+.1f}%, {a_results[0]['years_profitable']} yrs+)")
    print(f"  Stage B: {len(b_results)} combos, best={b_results[0]['combo_id']} "
          f"({b_results[0]['avg_return']:+.1f}%, {b_results[0]['years_profitable']} yrs+)")
    print(f"  Stage C: {len(c_results)} combos, best={c_results[0]['combo_id']} "
          f"({c_results[0]['avg_return']:+.1f}%, {c_results[0]['years_profitable']} yrs+)")
    print(f"  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Results: results/exp600/sweep_v8_stage[ABC].json")
    print(f"{'='*100}\n")

    # Save combined summary
    combined = {
        "experiment": "EXP-600-v8",
        "description": "Three-stage refinement: A=P0141 refine, B=VIX regime, C=multi-ticker",
        "total_elapsed_seconds": round(total_elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "stage_a_best": {
            "combo_id": a_results[0]["combo_id"],
            "label": a_results[0]["label"],
            "avg_return": a_results[0]["avg_return"],
            "years_profitable": a_results[0]["years_profitable"],
            "worst_drawdown": a_results[0]["worst_drawdown"],
            "consistency_score": a_results[0]["consistency_score"],
        },
        "stage_b_best": {
            "combo_id": b_results[0]["combo_id"],
            "label": b_results[0]["label"],
            "avg_return": b_results[0]["avg_return"],
            "years_profitable": b_results[0]["years_profitable"],
            "worst_drawdown": b_results[0]["worst_drawdown"],
            "consistency_score": b_results[0]["consistency_score"],
        },
        "stage_c_best": {
            "combo_id": c_results[0]["combo_id"],
            "label": c_results[0]["label"],
            "avg_return": c_results[0]["avg_return"],
            "years_profitable": c_results[0]["years_profitable"],
            "worst_drawdown": c_results[0]["worst_drawdown"],
            "consistency_score": c_results[0]["consistency_score"],
        },
    }
    summary_path = OUTPUT_DIR / "sweep_v8_summary.json"
    with open(summary_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"  Combined summary: {summary_path}")


if __name__ == "__main__":
    main()
