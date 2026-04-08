#!/usr/bin/env python3
"""
EXP-601: IC-Enhanced Multi-Regime Sweep.

The untapped lever: iron_condor.enabled=True + neutral_regime_only=True
routes NEUTRAL regime days to iron condors (non-directional) while keeping
directional spreads for BULL/BEAR regimes.  This directly attacks the
directionality problem that limited EXP-600 to a single +3.7% config.

Stage A (24 combos): IC-enhanced credit spreads.
  Fixed: direction=both, regime_mode=combo, min_credit_pct=5,
         profit_target=50%, stop_loss=2.5x, iron_condor.enabled=True,
         neutral_regime_only=True, min_combined_credit_pct=10
  Vary: target_dte=[15,25,35], spread_width=[5,10], otm_pct=[3%,5%],
        max_risk_per_trade=[2%,3%]
  → 3×2×2×2 = 24 combos × 6 years = 144 backtests

Stage B (18 combos): VIX gating + IV rank filtering on best Stage A.
  Vary: iv_rank_min_entry=[0,20,30], vix_max_entry=[0,25,30],
        vix_close_all=[0,35]
  → 3×3×2 = 18 combos × 6 years = 108 backtests

Stage C (6 combos): Multi-ticker using best A+B config.
  SPY/QQQ/IWM solo + SPY+QQQ, SPY+IWM, SPY+QQQ+IWM

P0141 baseline: +3.7% avg, 6/6 profitable, -13.8% DD, 240 trades
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

OUTPUT_DIR = ROOT / "results" / "exp601"
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

    ic_cfg = {
        "enabled":                 params.get("ic_enabled", True),
        "neutral_regime_only":     params.get("ic_neutral_regime_only", True),
        "min_combined_credit_pct": params.get("ic_min_combined_credit_pct", 10),
        "vix_min":                 params.get("ic_vix_min", 0),
        "risk_per_trade":          params.get("ic_risk_per_trade", None),
    }

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
            "iron_condor":         ic_cfg,
            "iv_rank_min_entry":   params.get("iv_rank_min_entry", 0),
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
            "max_contracts":        params.get("max_contracts", 25),
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
    ic_trades_list = [yr.get("ic_trades", 0) for yr in year_results.values() if "error" not in yr]
    dds = [yr["max_drawdown"] for yr in year_results.values() if "error" not in yr]
    wrs = [yr["win_rate"] for yr in year_results.values()
           if "error" not in yr and yr["trades"] > 0]

    avg_ret = sum(rets) / len(rets) if rets else 0
    total_trades = sum(trades_list)
    total_ic_trades = sum(ic_trades_list)
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
        "total_ic_trades": total_ic_trades,
        "ic_pct": round(total_ic_trades / total_trades * 100, 1) if total_trades > 0 else 0,
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
            ic_trades = r.get("iron_condor_trades", 0)
            wr = r.get("win_rate", 0)
            dd = r.get("max_drawdown", 0)
            ending_cap = r.get("ending_capital", 100_000)
            ic_wr = r.get("iron_condor_win_rate", 0)
            year_results[str(year)] = {
                "return_pct": round(ret, 2),
                "trades": trades,
                "ic_trades": ic_trades,
                "ic_win_rate": round(ic_wr, 1),
                "win_rate": round(wr, 1),
                "max_drawdown": round(dd, 2),
                "sharpe": round(r.get("sharpe_ratio", 0), 2),
                "ending_capital": round(ending_cap),
            }
            ic_flag = f" IC={ic_trades}" if ic_trades > 0 else ""
            flag = "+" if ret > 0 else "-"
            print(f"    {year}: {ret:+6.1f}% ({trades:>3} trades{ic_flag}, "
                  f"WR {wr:.0f}%, DD {dd:.1f}%) {flag}")
        except Exception as e:
            year_results[str(year)] = {"return_pct": 0, "trades": 0, "ic_trades": 0, "error": str(e)}
            print(f"    {year}: ERROR {e}")

    summary = compute_summary(combo_id, label, params, year_results, years)

    yp = int(summary["years_profitable"].split("/")[0])
    avg = summary["avg_return"]
    ic_pct = summary["ic_pct"]
    if yp >= 5 and avg > 3.7:
        flag = "BEATS P0141"
    elif yp >= 5 and avg > 0:
        flag = "CONSISTENT"
    elif avg > 0:
        flag = "OK"
    else:
        flag = "BAD"
    print(f"    -> AVG {avg:+.1f}%, {summary['total_trades']} trades "
          f"({summary['total_ic_trades']} IC, {ic_pct:.0f}%), "
          f"min {summary['min_trades_year']}/yr, "
          f"{summary['years_profitable']} yrs+, DD {summary['worst_drawdown']:.1f}%, "
          f"score {summary['consistency_score']:.0f}  [{flag}]\n")
    return summary


def print_leaderboard(results, title, top_n=15):
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    print(f"  {'Rk':<3} {'ID':<7} {'Score':>5} {'Avg':>7} {'Trds':>5} {'IC':>4} "
          f"{'IC%':>4} {'Min/Y':>5} {'Yr+':>4} {'WR':>5} {'DD':>7} {'Comp$':>10}  Config")
    print(f"  {'-'*120}")
    for rank, r in enumerate(results[:top_n], 1):
        yp = int(r["years_profitable"].split("/")[0])
        flag = ""
        if yp >= 5 and r["avg_return"] > 3.7:
            flag = " ***BEATS***"
        elif yp >= 5:
            flag = f" ({yp}/6)"
        print(f"  {rank:<3} {r['combo_id']:<7} {r['consistency_score']:>5.0f} "
              f"{r['avg_return']:>+6.1f}% {r['total_trades']:>5} {r['total_ic_trades']:>4} "
              f"{r['ic_pct']:>3.0f}% {r['min_trades_year']:>5} "
              f"{r['years_profitable']:>4} {r['avg_win_rate']:>4.0f}% {r['worst_drawdown']:>6.1f}% "
              f"${r['compound_final_capital']:>9,}  {r['label']}{flag}")


# ────────────────────────────────────────────────────────────
#  Stage A: IC-Enhanced Credit Spreads (24 combos)
# ────────────────────────────────────────────────────────────

def stage_a():
    """IC-enhanced credit spreads with neutral_regime_only=True.
    Vary: target_dte=[15,25,35], spread_width=[5,10], otm_pct=[3%,5%],
          max_risk_per_trade=[2%,3%] = 24 combos.
    """
    dtes = [15, 25, 35]
    widths = [5, 10]
    otms = [0.03, 0.05]
    risks = [2.0, 3.0]

    combos = []
    for dte in dtes:
        for w in widths:
            for otm in otms:
                for risk in risks:
                    combos.append({
                        "label": f"DTE={dte} W=${w} OTM={otm*100:.0f}% risk={risk}%",
                        "target_dte": dte,
                        "spread_width": w,
                        "otm_pct": otm,
                        "profit_target": 50,
                        "stop_loss_multiplier": 2.5,
                        "max_risk_per_trade": risk,
                        "direction": "both",
                        "regime_mode": "combo",
                        "min_credit_pct": 5,
                        "compound": False,
                        "max_contracts": 25,
                        # IC params — enabled, neutral_regime_only default True in build_config
                        "ic_enabled": True,
                        "ic_neutral_regime_only": True,
                        "ic_min_combined_credit_pct": 10,
                    })

    assert len(combos) == 24, f"Expected 24 combos, got {len(combos)}"

    print(f"\n{'='*110}")
    print(f"  STAGE A: IC-Enhanced Credit Spreads (24 combos)")
    print(f"  Fixed: direction=both, regime_mode=combo, min_credit=5%, PT=50%, SL=2.5x")
    print(f"  IC: enabled=True, neutral_regime_only=True, min_combined_credit=10%")
    print(f"  Vary: DTE=[15,25,35], W=[$5,$10], OTM=[3%,5%], risk=[2%,3%]")
    print(f"  24 combos x 6 years = 144 backtests")
    print(f"  P0141 baseline: +3.7% avg, 6/6 profitable, -13.8% DD")
    print(f"{'='*110}\n")

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
        "experiment": "EXP-601-stageA",
        "description": "IC-enhanced credit spreads: neutral_regime_only=True, vary DTE/W/OTM/risk",
        "ticker": "SPY",
        "years": YEARS,
        "total_combos": len(combos),
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "p0141_baseline": "+3.7% avg, 6/6 profitable, -13.8% DD, 240 trades",
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_stageA.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print_leaderboard(results, "STAGE A LEADERBOARD — IC-Enhanced Credit Spreads")

    # Summary by DTE
    print(f"\n  === DTE Summary ===")
    for dte in dtes:
        subset = [r for r in results if r["params"]["target_dte"] == dte]
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        avg_ic = sum(r["total_ic_trades"] for r in subset) / len(subset) if subset else 0
        five_plus = sum(1 for r in subset if int(r["years_profitable"].split("/")[0]) >= 5)
        print(f"  DTE={dte}: avg={avg_all:+.1f}%, avg_IC_trades={avg_ic:.0f}, 5/6+={five_plus}/{len(subset)}")

    # Summary by spread width
    print(f"\n  === Width Summary ===")
    for w in widths:
        subset = [r for r in results if r["params"]["spread_width"] == w]
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        avg_ic = sum(r["total_ic_trades"] for r in subset) / len(subset) if subset else 0
        print(f"  W=${w}: avg={avg_all:+.1f}%, avg_IC_trades={avg_ic:.0f}")

    # Summary by OTM
    print(f"\n  === OTM Summary ===")
    for otm in otms:
        subset = [r for r in results if r["params"]["otm_pct"] == otm]
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        avg_ic = sum(r["total_ic_trades"] for r in subset) / len(subset) if subset else 0
        print(f"  OTM={otm*100:.0f}%: avg={avg_all:+.1f}%, avg_IC_trades={avg_ic:.0f}")

    # IC trade analysis
    total_ic = sum(r["total_ic_trades"] for r in results)
    total_all = sum(r["total_trades"] for r in results)
    print(f"\n  === IC Trade Analysis ===")
    print(f"  Total IC trades across all combos: {total_ic}")
    print(f"  Total all trades across all combos: {total_all}")
    if total_all > 0:
        print(f"  IC as % of all trades: {total_ic/total_all*100:.1f}%")

    beaters = [r for r in results
               if int(r["years_profitable"].split("/")[0]) >= 5
               and r["avg_return"] > 3.7]
    print(f"\n  BEATS P0141 (5/6+ yrs, >+3.7% avg): {len(beaters)}/{len(combos)}")
    for r in beaters[:5]:
        print(f"    {r['combo_id']} {r['label']} -> {r['avg_return']:+.1f}%, "
              f"{r['years_profitable']} yrs+, IC={r['total_ic_trades']}, "
              f"DD={r['worst_drawdown']:.1f}%")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {output_path}")
    return results


# ────────────────────────────────────────────────────────────
#  Stage B: VIX Gating + IV Rank Filtering (18 combos)
# ────────────────────────────────────────────────────────────

def stage_b(best_a_params):
    """VIX gating + IV rank filtering on best Stage A config.
    iv_rank_min=[0,20,30] × vix_max=[0,25,30] × vix_close_all=[0,35] = 18 combos.
    """
    iv_ranks = [0, 20, 30]
    vix_maxes = [0, 25, 30]
    vix_closes = [0, 35]

    combos = []
    for ivr in iv_ranks:
        for vmax in vix_maxes:
            for vclose in vix_closes:
                params = dict(best_a_params)
                params["iv_rank_min_entry"] = ivr
                params["vix_max_entry"] = vmax
                params["vix_close_all"] = vclose
                params["label"] = f"IVR>={ivr} VIX<={vmax or 'off'} close@{vclose or 'off'}"
                combos.append(params)

    assert len(combos) == 18, f"Expected 18 combos, got {len(combos)}"

    base_label = best_a_params.get("label", "best_A")
    print(f"\n{'='*110}")
    print(f"  STAGE B: VIX Gating + IV Rank Filtering (18 combos)")
    print(f"  Base config from Stage A: {base_label}")
    print(f"  Vary: iv_rank_min=[0,20,30], vix_max=[0,25,30], vix_close_all=[0,35]")
    print(f"  18 combos x 6 years = 108 backtests")
    print(f"{'='*110}\n")

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
        "experiment": "EXP-601-stageB",
        "description": "VIX gating + IV rank filtering on best Stage A config",
        "base_config": base_label,
        "ticker": "SPY",
        "years": YEARS,
        "total_combos": len(combos),
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_stageB.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print_leaderboard(results, "STAGE B LEADERBOARD — VIX Gating + IV Rank")

    # Summary by IV rank
    print(f"\n  === IV Rank Summary ===")
    for ivr in iv_ranks:
        subset = [r for r in results if r["params"].get("iv_rank_min_entry") == ivr]
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        avg_trades = sum(r["total_trades"] for r in subset) / len(subset) if subset else 0
        print(f"  IVR>={ivr}: avg={avg_all:+.1f}%, avg_trades={avg_trades:.0f}")

    # Summary by VIX max
    print(f"\n  === VIX Max Summary ===")
    for vmax in vix_maxes:
        subset = [r for r in results if r["params"].get("vix_max_entry") == vmax]
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        print(f"  VIX<={vmax or 'off'}: avg={avg_all:+.1f}%")

    # Summary by vix_close_all
    print(f"\n  === VIX Close-All Summary ===")
    for vclose in vix_closes:
        subset = [r for r in results if r["params"].get("vix_close_all") == vclose]
        avg_all = sum(r["avg_return"] for r in subset) / len(subset) if subset else 0
        print(f"  close@{vclose or 'off'}: avg={avg_all:+.1f}%")

    beaters = [r for r in results
               if int(r["years_profitable"].split("/")[0]) >= 5
               and r["avg_return"] > 3.7]
    print(f"\n  BEATS P0141 (5/6+ yrs, >+3.7% avg): {len(beaters)}/{len(combos)}")
    for r in beaters[:5]:
        print(f"    {r['combo_id']} {r['label']} -> {r['avg_return']:+.1f}%, "
              f"{r['years_profitable']} yrs+, IC={r['total_ic_trades']}, "
              f"DD={r['worst_drawdown']:.1f}%")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {output_path}")
    return results


# ────────────────────────────────────────────────────────────
#  Stage C: Multi-Ticker (6 combos)
# ────────────────────────────────────────────────────────────

def stage_c(best_params):
    """Multi-ticker: 3 solo (SPY, QQQ, IWM) + 3 pairs/combos."""
    tickers_sets = [
        ("SPY-solo", ["SPY"]),
        ("QQQ-solo", ["QQQ"]),
        ("IWM-solo", ["IWM"]),
        ("SPY+QQQ", ["SPY", "QQQ"]),
        ("SPY+IWM", ["SPY", "IWM"]),
        ("SPY+QQQ+IWM", ["SPY", "QQQ", "IWM"]),
    ]

    base_label = best_params.get("label", "best_config")
    print(f"\n{'='*110}")
    print(f"  STAGE C: Multi-Ticker (6 combos)")
    print(f"  Base config: {base_label}")
    print(f"  Tickers: SPY, QQQ, IWM solo + pairs + triple")
    print(f"  6 combos, each ticker x 6 years")
    print(f"{'='*110}\n")

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
            year_ic_trades = 0
            year_wrs = []
            year_dds = []
            year_sharpes = []
            year_errors = []
            ticker_cap = 100_000 / len(tickers)

            for ticker in tickers:
                try:
                    r = run_single(ticker, year, params, starting_capital=round(ticker_cap))
                    ret = r.get("return_pct", 0)
                    trades = r.get("total_trades", 0)
                    ic_trades = r.get("iron_condor_trades", 0)
                    wr = r.get("win_rate", 0)
                    dd = r.get("max_drawdown", 0)
                    year_rets.append(ret)
                    year_trades += trades
                    year_ic_trades += ic_trades
                    if trades > 0:
                        year_wrs.append(wr)
                    year_dds.append(dd)
                    year_sharpes.append(r.get("sharpe_ratio", 0))
                except Exception as e:
                    year_errors.append(str(e))

            if year_errors and not year_rets:
                all_year_results[str(year)] = {
                    "return_pct": 0, "trades": 0, "ic_trades": 0,
                    "error": "; ".join(year_errors),
                }
                print(f"    {year}: ERROR {year_errors[0]}")
            else:
                avg_ret = sum(year_rets) / len(year_rets) if year_rets else 0
                avg_wr = sum(year_wrs) / len(year_wrs) if year_wrs else 0
                worst_dd = min(year_dds) if year_dds else 0
                avg_sharpe = sum(year_sharpes) / len(year_sharpes) if year_sharpes else 0
                all_year_results[str(year)] = {
                    "return_pct": round(avg_ret, 2),
                    "trades": year_trades,
                    "ic_trades": year_ic_trades,
                    "win_rate": round(avg_wr, 1),
                    "max_drawdown": round(worst_dd, 2),
                    "sharpe": round(avg_sharpe, 2),
                    "ending_capital": 100_000,
                    "tickers": tickers,
                }
                ic_flag = f" IC={year_ic_trades}" if year_ic_trades > 0 else ""
                flag = "+" if avg_ret > 0 else "-"
                print(f"    {year}: {avg_ret:+6.1f}% ({year_trades:>3} trades{ic_flag}, "
                      f"WR {avg_wr:.0f}%, DD {worst_dd:.1f}%) {flag}")

        summary = compute_summary(combo_id, label, params, all_year_results, YEARS)
        summary["tickers"] = tickers

        yp = int(summary["years_profitable"].split("/")[0])
        avg = summary["avg_return"]
        flag_str = "BEATS" if yp >= 5 and avg > 3.7 else ("OK" if avg > 0 else "BAD")
        print(f"    -> AVG {avg:+.1f}%, {summary['total_trades']} trades "
              f"({summary['total_ic_trades']} IC), "
              f"{summary['years_profitable']} yrs+, DD {summary['worst_drawdown']:.1f}%, "
              f"score {summary['consistency_score']:.0f}  [{flag_str}]\n")
        results.append(summary)

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    output = {
        "experiment": "EXP-601-stageC",
        "description": "Multi-ticker: SPY/QQQ/IWM solo + combos",
        "base_config": base_label,
        "years": YEARS,
        "total_combos": len(tickers_sets),
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    output_path = OUTPUT_DIR / "sweep_stageC.json"
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

    print("\n" + "=" * 110)
    print("  STARTING EXP-601: IC-ENHANCED MULTI-REGIME SWEEP")
    print("  Iron condor + neutral_regime_only attacks directionality problem")
    print("=" * 110)

    # ── Stage A ──
    a_results = stage_a()

    # Verify IC trades appeared
    total_ic = sum(r["total_ic_trades"] for r in a_results)
    if total_ic == 0:
        print("\n  *** WARNING: ZERO IC trades across all Stage A combos! ***")
        print("  This means the IC fallback never triggered. Check:")
        print("    1. Are individual credit spreads always meeting min_credit?")
        print("    2. Is the regime detector producing NEUTRAL days?")
        print("    3. Is min_combined_credit_pct too high?")
        print("  Continuing to Stages B & C anyway...\n")

    # Pick best Stage A config
    best_a = a_results[0]
    best_a_params = dict(best_a["params"])
    best_a_params["label"] = best_a["label"]
    print(f"\n  Best Stage A config: {best_a['combo_id']} {best_a['label']}")
    print(f"    avg={best_a['avg_return']:+.1f}%, {best_a['years_profitable']} yrs+, "
          f"IC={best_a['total_ic_trades']}, DD={best_a['worst_drawdown']:.1f}%, "
          f"score={best_a['consistency_score']:.0f}")

    # ── Stage B ──
    b_results = stage_b(best_a_params)
    best_b = b_results[0]

    # Merge best B filters into params for Stage C
    best_params = dict(best_a_params)
    if best_b["avg_return"] > best_a["avg_return"]:
        best_params["iv_rank_min_entry"] = best_b["params"].get("iv_rank_min_entry", 0)
        best_params["vix_max_entry"] = best_b["params"].get("vix_max_entry", 0)
        best_params["vix_close_all"] = best_b["params"].get("vix_close_all", 0)
        best_params["label"] = f"{best_a['label']} + {best_b['label']}"
        print(f"\n  Stage B improved: using {best_b['combo_id']} {best_b['label']}")
    else:
        print(f"\n  Stage B did not improve. Using Stage A config for Stage C.")

    # ── Stage C ──
    c_results = stage_c(best_params)

    # ── Final Summary ──
    total_elapsed = time.time() - t_total
    print(f"\n{'='*110}")
    print(f"  FINAL SUMMARY — EXP-601 IC-Enhanced Multi-Regime Sweep")
    print(f"{'='*110}")
    print(f"  Stage A: {len(a_results)} combos, best={a_results[0]['combo_id']} "
          f"({a_results[0]['avg_return']:+.1f}%, {a_results[0]['years_profitable']} yrs+, "
          f"IC={a_results[0]['total_ic_trades']})")
    print(f"  Stage B: {len(b_results)} combos, best={b_results[0]['combo_id']} "
          f"({b_results[0]['avg_return']:+.1f}%, {b_results[0]['years_profitable']} yrs+, "
          f"IC={b_results[0]['total_ic_trades']})")
    print(f"  Stage C: {len(c_results)} combos, best={c_results[0]['combo_id']} "
          f"({c_results[0]['avg_return']:+.1f}%, {c_results[0]['years_profitable']} yrs+, "
          f"IC={c_results[0]['total_ic_trades']})")
    print(f"  Total IC trades (Stage A): {total_ic}")
    print(f"  P0141 baseline: +3.7% avg, 6/6 profitable, -13.8% DD")
    print(f"  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Results: results/exp601/sweep_stage[ABC].json")
    print(f"{'='*110}\n")

    # Save combined summary
    combined = {
        "experiment": "EXP-601",
        "description": "IC-enhanced multi-regime sweep: iron_condor + neutral_regime_only",
        "total_elapsed_seconds": round(total_elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "total_ic_trades_stage_a": total_ic,
        "stage_a_best": {
            "combo_id": a_results[0]["combo_id"],
            "label": a_results[0]["label"],
            "avg_return": a_results[0]["avg_return"],
            "total_ic_trades": a_results[0]["total_ic_trades"],
            "ic_pct": a_results[0]["ic_pct"],
            "years_profitable": a_results[0]["years_profitable"],
            "worst_drawdown": a_results[0]["worst_drawdown"],
            "consistency_score": a_results[0]["consistency_score"],
        },
        "stage_b_best": {
            "combo_id": b_results[0]["combo_id"],
            "label": b_results[0]["label"],
            "avg_return": b_results[0]["avg_return"],
            "total_ic_trades": b_results[0]["total_ic_trades"],
            "ic_pct": b_results[0]["ic_pct"],
            "years_profitable": b_results[0]["years_profitable"],
            "worst_drawdown": b_results[0]["worst_drawdown"],
            "consistency_score": b_results[0]["consistency_score"],
        },
        "stage_c_best": {
            "combo_id": c_results[0]["combo_id"],
            "label": c_results[0]["label"],
            "avg_return": c_results[0]["avg_return"],
            "total_ic_trades": c_results[0]["total_ic_trades"],
            "ic_pct": c_results[0]["ic_pct"],
            "years_profitable": c_results[0]["years_profitable"],
            "worst_drawdown": c_results[0]["worst_drawdown"],
            "consistency_score": c_results[0]["consistency_score"],
        },
    }
    summary_path = OUTPUT_DIR / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"  Combined summary: {summary_path}")


if __name__ == "__main__":
    main()
