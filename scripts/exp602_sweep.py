#!/usr/bin/env python3
"""
EXP-602: Compounding + Aggressive IC Push.

Builds on EXP-601 breakthrough (+18.2% avg with IC-enhanced configs).
Combines compounding, higher risk, tighter stops, multi-ticker blending,
and VIX-dynamic DTE to close the 3x gap to Carlos's reference (+56.6%).

Stage A (16 combos): Compounding ON with A017 base (DTE=35)
Stage B (24 combos): Higher risk (4-8%) + tighter stops (1.5-2.5x)
Stage C (16 combos): DTE=15 compound push (71% IC rate)
Stage D (8 combos):  Multi-ticker blend (SPY+QQQ)
Stage E (8 combos):  VIX-dynamic DTE switching

Total: 72 combos × 6 years = 432 backtests
Target: ≥+30% avg, 5/6 profitable, DD ≤ -30%
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

OUTPUT_DIR = ROOT / "results" / "exp602"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


# ────────────────────────────────────────────────────────────
#  Shared helpers (from EXP-601, extended for compounding)
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
            "vix_dte_threshold":   params.get("vix_dte_threshold", 0),
            "dte_low_vix":         params.get("dte_low_vix", target_dte),
            "min_dte_low_vix":     params.get("min_dte_low_vix", min_dte),
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
            "sizing_mode":        params.get("sizing_mode", "flat"),
            "slippage_multiplier": 1.0,
            "max_portfolio_exposure_pct": 100.0,
            "exclude_months":     [],
            "volume_gate":        False,
            "oi_gate":            False,
            "risk_cap":           25.0,
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


def run_compound_years(ticker, params, years, starting_capital=100_000):
    """Run years sequentially with compound=True, carrying forward capital."""
    from backtest.backtester import Backtester
    from shared.iron_vault import IronVault

    hd = IronVault.instance()
    otm_pct = params.get("otm_pct", 0.03)
    current_capital = starting_capital
    year_results = {}

    for year in sorted(years):
        try:
            config = build_config(params, starting_capital=round(current_capital))
            start = datetime(year, 1, 1)
            end = datetime(year, 12, 31)
            bt = Backtester(config, historical_data=hd, otm_pct=otm_pct)
            r = bt.run_backtest(ticker, start, end) or {}

            ret = r.get("return_pct", 0)
            trades = r.get("total_trades", 0)
            ic_trades = r.get("iron_condor_trades", 0)
            wr = r.get("win_rate", 0)
            dd = r.get("max_drawdown", 0)
            ending = r.get("ending_capital", current_capital)
            ic_wr = r.get("iron_condor_win_rate", 0)

            year_results[str(year)] = {
                "return_pct": round(ret, 2),
                "trades": trades,
                "ic_trades": ic_trades,
                "ic_win_rate": round(ic_wr, 1),
                "win_rate": round(wr, 1),
                "max_drawdown": round(dd, 2),
                "sharpe": round(r.get("sharpe_ratio", 0), 2),
                "starting_capital": round(current_capital),
                "ending_capital": round(ending),
            }

            # Carry forward for compound mode
            if params.get("compound", False):
                current_capital = ending
            else:
                current_capital = starting_capital

            ic_flag = f" IC={ic_trades}" if ic_trades > 0 else ""
            flag = "+" if ret > 0 else "-"
            cap_str = f"${current_capital:,.0f}" if params.get("compound", False) else ""
            print(f"    {year}: {ret:+6.1f}% ({trades:>3} trades{ic_flag}, "
                  f"WR {wr:.0f}%, DD {dd:.1f}%) {flag} {cap_str}")
        except Exception as e:
            year_results[str(year)] = {
                "return_pct": 0, "trades": 0, "ic_trades": 0,
                "error": str(e),
                "starting_capital": round(current_capital),
                "ending_capital": round(current_capital),
            }
            print(f"    {year}: ERROR {e}")

    return year_results, current_capital


def compute_summary(combo_id, label, params, year_results, years):
    """Compute aggregate metrics from per-year results."""
    valid = {k: v for k, v in year_results.items() if "error" not in v}
    rets = [v["return_pct"] for v in valid.values()]
    trades_list = [v["trades"] for v in valid.values()]
    ic_trades_list = [v.get("ic_trades", 0) for v in valid.values()]
    dds = [v["max_drawdown"] for v in valid.values()]
    wrs = [v["win_rate"] for v in valid.values() if v["trades"] > 0]

    avg_ret = sum(rets) / len(rets) if rets else 0
    total_trades = sum(trades_list)
    total_ic_trades = sum(ic_trades_list)
    min_trades_yr = min(trades_list) if trades_list else 0
    years_profitable = sum(1 for r in rets if r > 0)
    worst_dd = min(dds) if dds else 0
    avg_wr = sum(wrs) / len(wrs) if wrs else 0

    # Final capital (last year ending)
    last_yr = str(max(years))
    final_capital = year_results.get(last_yr, {}).get("ending_capital", 100_000)

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
        "final_capital": round(final_capital),
        "total_return_pct": round((final_capital - 100_000) / 100_000 * 100, 1),
        "consistency_score": round(consistency, 1),
        "by_year": year_results,
    }


def run_combo(combo_id, label, params, years, ticker="SPY"):
    """Run one config across all years. Uses compound carry-forward if compound=True."""
    print(f"  {combo_id}: {label}")
    is_compound = params.get("compound", False)

    if is_compound:
        year_results, final_cap = run_compound_years(ticker, params, years)
    else:
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
                    "starting_capital": 100_000,
                    "ending_capital": round(ending_cap),
                }
                ic_flag = f" IC={ic_trades}" if ic_trades > 0 else ""
                flag = "+" if ret > 0 else "-"
                print(f"    {year}: {ret:+6.1f}% ({trades:>3} trades{ic_flag}, "
                      f"WR {wr:.0f}%, DD {dd:.1f}%) {flag}")
            except Exception as e:
                year_results[str(year)] = {
                    "return_pct": 0, "trades": 0, "ic_trades": 0,
                    "error": str(e), "starting_capital": 100_000,
                    "ending_capital": 100_000,
                }
                print(f"    {year}: ERROR {e}")

    summary = compute_summary(combo_id, label, params, year_results, years)

    yp = int(summary["years_profitable"].split("/")[0])
    avg = summary["avg_return"]
    total_ret = summary["total_return_pct"]
    if avg >= 30 and yp >= 5:
        flag = "TARGET"
    elif avg >= 20 and yp >= 4:
        flag = "STRONG"
    elif yp >= 5 and avg > 3.7:
        flag = "BEATS P0141"
    elif avg > 0:
        flag = "OK"
    else:
        flag = "BAD"
    compound_str = f", final=${summary['final_capital']:,}" if params.get("compound") else ""
    print(f"    -> AVG {avg:+.1f}%, total {total_ret:+.1f}%, {summary['total_trades']} trades "
          f"({summary['total_ic_trades']} IC, {summary['ic_pct']:.0f}%), "
          f"{summary['years_profitable']} yrs+, DD {summary['worst_drawdown']:.1f}%"
          f"{compound_str}  [{flag}]\n")
    return summary


def run_multi_ticker(combo_id, label, params, years, ticker_weights):
    """Run multiple tickers with capital split, optionally compound across years."""
    print(f"  {combo_id}: {label}")
    is_compound = params.get("compound", False)
    total_capital = 100_000
    year_results = {}

    # Per-ticker capital allocation
    ticker_caps = {t: total_capital * w for t, w in ticker_weights.items()}

    for year in sorted(years):
        year_rets_weighted = []
        year_trades = 0
        year_ic_trades = 0
        year_wrs = []
        year_dds = []
        year_errors = []
        new_ticker_caps = {}

        for ticker, cap in ticker_caps.items():
            try:
                from backtest.backtester import Backtester
                from shared.iron_vault import IronVault

                config = build_config(params, starting_capital=round(cap))
                start = datetime(year, 1, 1)
                end = datetime(year, 12, 31)
                hd = IronVault.instance()
                otm_pct = params.get("otm_pct", 0.03)
                bt = Backtester(config, historical_data=hd, otm_pct=otm_pct)
                r = bt.run_backtest(ticker, start, end) or {}

                ret = r.get("return_pct", 0)
                trades = r.get("total_trades", 0)
                ic_trades = r.get("iron_condor_trades", 0)
                wr = r.get("win_rate", 0)
                dd = r.get("max_drawdown", 0)
                ending = r.get("ending_capital", cap)

                weight = ticker_weights[ticker]
                year_rets_weighted.append(ret * weight)
                year_trades += trades
                year_ic_trades += ic_trades
                if trades > 0:
                    year_wrs.append(wr)
                year_dds.append(dd)

                new_ticker_caps[ticker] = ending if is_compound else total_capital * weight
            except Exception as e:
                year_errors.append(f"{ticker}: {e}")
                new_ticker_caps[ticker] = ticker_caps[ticker]

        if year_errors and not year_rets_weighted:
            year_results[str(year)] = {
                "return_pct": 0, "trades": 0, "ic_trades": 0,
                "error": "; ".join(year_errors),
                "starting_capital": round(sum(ticker_caps.values())),
                "ending_capital": round(sum(ticker_caps.values())),
            }
            print(f"    {year}: ERROR {year_errors[0]}")
        else:
            blended_ret = sum(year_rets_weighted)
            avg_wr = sum(year_wrs) / len(year_wrs) if year_wrs else 0
            worst_dd = min(year_dds) if year_dds else 0
            start_cap = sum(ticker_caps.values())
            end_cap = sum(new_ticker_caps.values())
            year_results[str(year)] = {
                "return_pct": round(blended_ret, 2),
                "trades": year_trades,
                "ic_trades": year_ic_trades,
                "win_rate": round(avg_wr, 1),
                "max_drawdown": round(worst_dd, 2),
                "sharpe": 0,
                "starting_capital": round(start_cap),
                "ending_capital": round(end_cap),
            }
            ic_flag = f" IC={year_ic_trades}" if year_ic_trades > 0 else ""
            flag = "+" if blended_ret > 0 else "-"
            cap_str = f"${end_cap:,.0f}" if is_compound else ""
            print(f"    {year}: {blended_ret:+6.1f}% ({year_trades:>3} trades{ic_flag}, "
                  f"WR {avg_wr:.0f}%, DD {worst_dd:.1f}%) {flag} {cap_str}")

        ticker_caps = new_ticker_caps

    summary = compute_summary(combo_id, label, params, year_results, years)
    summary["ticker_weights"] = ticker_weights

    yp = int(summary["years_profitable"].split("/")[0])
    avg = summary["avg_return"]
    total_ret = summary["total_return_pct"]
    if avg >= 30 and yp >= 5:
        flag = "TARGET"
    elif avg >= 20 and yp >= 4:
        flag = "STRONG"
    elif avg > 0:
        flag = "OK"
    else:
        flag = "BAD"
    compound_str = f", final=${summary['final_capital']:,}" if is_compound else ""
    print(f"    -> AVG {avg:+.1f}%, total {total_ret:+.1f}%, {summary['total_trades']} trades "
          f"({summary['total_ic_trades']} IC), "
          f"{summary['years_profitable']} yrs+, DD {summary['worst_drawdown']:.1f}%"
          f"{compound_str}  [{flag}]\n")
    return summary


def print_leaderboard(results, title, top_n=15):
    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")
    print(f"  {'Rk':<3} {'ID':<7} {'Score':>5} {'Avg':>7} {'Total':>8} {'Trds':>5} {'IC':>4} "
          f"{'IC%':>4} {'Yr+':>4} {'WR':>5} {'DD':>7} {'Final$':>12}  Config")
    print(f"  {'-'*130}")
    for rank, r in enumerate(results[:top_n], 1):
        yp = int(r["years_profitable"].split("/")[0])
        flag = ""
        if r["avg_return"] >= 30 and yp >= 5:
            flag = " *** TARGET ***"
        elif r["avg_return"] >= 20 and yp >= 4:
            flag = " ** STRONG **"
        elif yp >= 5 and r["avg_return"] > 3.7:
            flag = " *BEATS*"
        print(f"  {rank:<3} {r['combo_id']:<7} {r['consistency_score']:>5.0f} "
              f"{r['avg_return']:>+6.1f}% {r['total_return_pct']:>+7.1f}% "
              f"{r['total_trades']:>5} {r['total_ic_trades']:>4} "
              f"{r['ic_pct']:>3.0f}% "
              f"{r['years_profitable']:>4} {r['avg_win_rate']:>4.0f}% {r['worst_drawdown']:>6.1f}% "
              f"${r['final_capital']:>11,}  {r['label']}{flag}")


def save_stage(stage_name, description, results, elapsed, extra=None):
    output = {
        "experiment": f"EXP-602-{stage_name}",
        "description": description,
        "years": YEARS,
        "total_combos": len(results),
        "elapsed_seconds": round(elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    if extra:
        output.update(extra)
    path = OUTPUT_DIR / f"sweep_{stage_name}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {path}")
    return path


# ────────────────────────────────────────────────────────────
#  Stage A: Compounding ON (16 combos)
# ────────────────────────────────────────────────────────────

def stage_a():
    """Compounding with A017 base (DTE=35, W=$5, OTM=3%, IC enabled).
    Vary: risk=[2,3,4,5%], stop_loss=[1.5,2.0,2.5,3.0] = 16 combos.
    """
    risks = [2.0, 3.0, 4.0, 5.0]
    stops = [1.5, 2.0, 2.5, 3.0]

    combos = []
    for risk in risks:
        for sl in stops:
            combos.append({
                "label": f"COMP risk={risk}% SL={sl}x DTE=35 W=$5 OTM=3%",
                "target_dte": 35,
                "spread_width": 5,
                "otm_pct": 0.03,
                "profit_target": 50,
                "stop_loss_multiplier": sl,
                "max_risk_per_trade": risk,
                "direction": "both",
                "regime_mode": "combo",
                "min_credit_pct": 5,
                "compound": True,
                "max_contracts": 25,
                "ic_enabled": True,
                "ic_neutral_regime_only": True,
                "ic_min_combined_credit_pct": 10,
            })

    print(f"\n{'='*120}")
    print(f"  STAGE A: Compounding ON — A017 base (DTE=35, W=$5, OTM=3%)")
    print(f"  compound=True, equity carries forward year-to-year")
    print(f"  Vary: risk=[2,3,4,5%], stop_loss=[1.5,2.0,2.5,3.0]")
    print(f"  {len(combos)} combos × 6 years = {len(combos)*6} backtests")
    print(f"  EXP-601 A017 baseline: +18.2% avg (non-compound), compound return +127.4%")
    print(f"{'='*120}\n")

    results = []
    t_start = time.time()
    for i, params in enumerate(combos):
        combo_id = f"A{i:03d}"
        print(f"  [{i+1}/{len(combos)}]", end=" ")
        summary = run_combo(combo_id, params["label"], params, YEARS)
        results.append(summary)

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    print_leaderboard(results, "STAGE A LEADERBOARD — Compounding ON (DTE=35)")
    save_stage("stageA", "Compounding ON with A017 base", results, elapsed)

    # Analysis
    print(f"\n  === Risk Level Analysis ===")
    for risk in risks:
        subset = [r for r in results if r["params"]["max_risk_per_trade"] == risk]
        avg = sum(r["avg_return"] for r in subset) / len(subset)
        best = max(subset, key=lambda x: x["total_return_pct"])
        print(f"  risk={risk}%: avg_avg={avg:+.1f}%, best total={best['total_return_pct']:+.1f}% "
              f"(SL={best['params']['stop_loss_multiplier']}x)")

    print(f"\n  === Stop Loss Analysis ===")
    for sl in stops:
        subset = [r for r in results if r["params"]["stop_loss_multiplier"] == sl]
        avg = sum(r["avg_return"] for r in subset) / len(subset)
        avg_dd = sum(r["worst_drawdown"] for r in subset) / len(subset)
        print(f"  SL={sl}x: avg_avg={avg:+.1f}%, avg_DD={avg_dd:.1f}%")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


# ────────────────────────────────────────────────────────────
#  Stage B: Higher Risk + Tighter Stops (24 combos)
# ────────────────────────────────────────────────────────────

def stage_b():
    """Higher risk (4-8%) with tighter stops (1.5-2.5x). No compounding (isolate effect).
    Vary: risk=[4,5,6,8%], stop_loss=[1.5,2.0,2.5], profit_target=[40,50%] = 24 combos.
    """
    risks = [4.0, 5.0, 6.0, 8.0]
    stops = [1.5, 2.0, 2.5]
    pts = [40, 50]

    combos = []
    for risk in risks:
        for sl in stops:
            for pt in pts:
                combos.append({
                    "label": f"risk={risk}% SL={sl}x PT={pt}% DTE=35 W=$5 OTM=3%",
                    "target_dte": 35,
                    "spread_width": 5,
                    "otm_pct": 0.03,
                    "profit_target": pt,
                    "stop_loss_multiplier": sl,
                    "max_risk_per_trade": risk,
                    "direction": "both",
                    "regime_mode": "combo",
                    "min_credit_pct": 5,
                    "compound": False,
                    "max_contracts": 25,
                    "ic_enabled": True,
                    "ic_neutral_regime_only": True,
                    "ic_min_combined_credit_pct": 10,
                })

    print(f"\n{'='*120}")
    print(f"  STAGE B: Higher Risk + Tighter Stops (no compound)")
    print(f"  Fixed: DTE=35, W=$5, OTM=3%, IC enabled, compound=False")
    print(f"  Vary: risk=[4,5,6,8%], SL=[1.5,2.0,2.5], PT=[40,50%]")
    print(f"  {len(combos)} combos × 6 years = {len(combos)*6} backtests")
    print(f"{'='*120}\n")

    results = []
    t_start = time.time()
    for i, params in enumerate(combos):
        combo_id = f"B{i:03d}"
        print(f"  [{i+1}/{len(combos)}]", end=" ")
        summary = run_combo(combo_id, params["label"], params, YEARS)
        results.append(summary)

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    print_leaderboard(results, "STAGE B LEADERBOARD — Higher Risk + Tighter Stops")
    save_stage("stageB", "Higher risk (4-8%) + tighter stops", results, elapsed)

    # Risk analysis
    print(f"\n  === Risk Level Analysis ===")
    for risk in risks:
        subset = [r for r in results if r["params"]["max_risk_per_trade"] == risk]
        avg = sum(r["avg_return"] for r in subset) / len(subset)
        avg_dd = sum(r["worst_drawdown"] for r in subset) / len(subset)
        print(f"  risk={risk}%: avg={avg:+.1f}%, avg_DD={avg_dd:.1f}%")

    # Stop loss analysis
    print(f"\n  === Stop Loss Analysis ===")
    for sl in stops:
        subset = [r for r in results if r["params"]["stop_loss_multiplier"] == sl]
        avg = sum(r["avg_return"] for r in subset) / len(subset)
        avg_dd = sum(r["worst_drawdown"] for r in subset) / len(subset)
        pos_count = sum(1 for r in subset if r["avg_return"] > 0)
        print(f"  SL={sl}x: avg={avg:+.1f}%, avg_DD={avg_dd:.1f}%, positive={pos_count}/{len(subset)}")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


# ────────────────────────────────────────────────────────────
#  Stage C: DTE=15 Compound Push (16 combos)
# ────────────────────────────────────────────────────────────

def stage_c():
    """DTE=15 with compounding. A003 base (W=$5, OTM=5%, 71% IC rate).
    Vary: risk=[2,3,4,5%], stop_loss=[1.5,2.0,2.5,3.0] = 16 combos.
    """
    risks = [2.0, 3.0, 4.0, 5.0]
    stops = [1.5, 2.0, 2.5, 3.0]

    combos = []
    for risk in risks:
        for sl in stops:
            combos.append({
                "label": f"COMP DTE=15 risk={risk}% SL={sl}x W=$5 OTM=5%",
                "target_dte": 15,
                "spread_width": 5,
                "otm_pct": 0.05,
                "profit_target": 50,
                "stop_loss_multiplier": sl,
                "max_risk_per_trade": risk,
                "direction": "both",
                "regime_mode": "combo",
                "min_credit_pct": 5,
                "compound": True,
                "max_contracts": 25,
                "ic_enabled": True,
                "ic_neutral_regime_only": True,
                "ic_min_combined_credit_pct": 10,
            })

    print(f"\n{'='*120}")
    print(f"  STAGE C: DTE=15 Compound Push — A003 base (W=$5, OTM=5%)")
    print(f"  compound=True, 71% IC rate in EXP-601")
    print(f"  Vary: risk=[2,3,4,5%], SL=[1.5,2.0,2.5,3.0]")
    print(f"  {len(combos)} combos × 6 years = {len(combos)*6} backtests")
    print(f"  EXP-601 A003 baseline: +15.5% avg, 4/6 yrs, 71% IC")
    print(f"{'='*120}\n")

    results = []
    t_start = time.time()
    for i, params in enumerate(combos):
        combo_id = f"C{i:03d}"
        print(f"  [{i+1}/{len(combos)}]", end=" ")
        summary = run_combo(combo_id, params["label"], params, YEARS)
        results.append(summary)

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    print_leaderboard(results, "STAGE C LEADERBOARD — DTE=15 Compound Push")
    save_stage("stageC", "DTE=15 compound push (A003 base)", results, elapsed)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


# ────────────────────────────────────────────────────────────
#  Stage D: Multi-Ticker Blend (8 combos)
# ────────────────────────────────────────────────────────────

def stage_d(best_a_params, best_c_params):
    """Multi-ticker blend using best compound configs from A and C.
    8 combos: SPY/QQQ solo ± compound, blends with different weights.
    """

    combos = [
        ("D000", "SPY-only (best DTE=35 config)", best_a_params, {"SPY": 1.0}),
        ("D001", "QQQ-only (best DTE=35 config)", best_a_params, {"QQQ": 1.0}),
        ("D002", "SPY+QQQ 50/50 no-compound",
         {**best_a_params, "compound": False}, {"SPY": 0.5, "QQQ": 0.5}),
        ("D003", "SPY+QQQ 50/50 compound",
         {**best_a_params, "compound": True}, {"SPY": 0.5, "QQQ": 0.5}),
        ("D004", "SPY+QQQ 70/30 no-compound",
         {**best_a_params, "compound": False}, {"SPY": 0.7, "QQQ": 0.3}),
        ("D005", "SPY+QQQ 70/30 compound",
         {**best_a_params, "compound": True}, {"SPY": 0.7, "QQQ": 0.3}),
        ("D006", "SPY+QQQ+IWM 50/30/20 no-compound",
         {**best_a_params, "compound": False}, {"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2}),
        ("D007", "SPY+QQQ+IWM 50/30/20 compound",
         {**best_a_params, "compound": True}, {"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2}),
    ]

    print(f"\n{'='*120}")
    print(f"  STAGE D: Multi-Ticker Blend")
    print(f"  Using best compound configs from Stages A and C")
    print(f"  8 combos: SPY/QQQ solo, 50/50, 70/30, triple")
    print(f"{'='*120}\n")

    results = []
    t_start = time.time()
    for combo_id, label, params, weights in combos:
        print(f"  [{combo_id}]", end=" ")
        if len(weights) == 1:
            ticker = list(weights.keys())[0]
            summary = run_combo(combo_id, label, params, YEARS, ticker=ticker)
        else:
            summary = run_multi_ticker(combo_id, label, params, YEARS, weights)
        results.append(summary)

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    print_leaderboard(results, "STAGE D LEADERBOARD — Multi-Ticker Blend")
    save_stage("stageD", "Multi-ticker blend", results, elapsed)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


# ────────────────────────────────────────────────────────────
#  Stage E: VIX-Dynamic DTE (8 combos)
# ────────────────────────────────────────────────────────────

def stage_e():
    """VIX-dynamic DTE switching: DTE=15 when VIX high, DTE=35/45 when VIX low.
    The backtester has native support via vix_dte_threshold + dte_low_vix.
    Vary: vix_threshold=[20,25], dte_low_vix=[35,45], risk=[3,5%] = 8 combos.
    """
    vix_thresholds = [20, 25]
    dte_lows = [35, 45]
    risks = [3.0, 5.0]

    combos = []
    for vix_t in vix_thresholds:
        for dte_l in dte_lows:
            for risk in risks:
                min_dte_l = max(10, dte_l - 10)
                combos.append({
                    "label": f"COMP VIX-DTE: DTE=15@VIX>={vix_t} else DTE={dte_l}, risk={risk}%",
                    "target_dte": 15,
                    "spread_width": 5,
                    "otm_pct": 0.03,
                    "profit_target": 50,
                    "stop_loss_multiplier": 2.5,
                    "max_risk_per_trade": risk,
                    "direction": "both",
                    "regime_mode": "combo",
                    "min_credit_pct": 5,
                    "compound": True,
                    "max_contracts": 25,
                    "ic_enabled": True,
                    "ic_neutral_regime_only": True,
                    "ic_min_combined_credit_pct": 10,
                    "vix_dte_threshold": vix_t,
                    "dte_low_vix": dte_l,
                    "min_dte_low_vix": min_dte_l,
                })

    print(f"\n{'='*120}")
    print(f"  STAGE E: VIX-Dynamic DTE Switching")
    print(f"  DTE=15 when VIX >= threshold (high IC rate), DTE=35/45 when VIX low")
    print(f"  compound=True, W=$5, OTM=3%")
    print(f"  Vary: vix_threshold=[20,25], dte_low_vix=[35,45], risk=[3,5%]")
    print(f"  {len(combos)} combos × 6 years = {len(combos)*6} backtests")
    print(f"{'='*120}\n")

    results = []
    t_start = time.time()
    for i, params in enumerate(combos):
        combo_id = f"E{i:03d}"
        print(f"  [{i+1}/{len(combos)}]", end=" ")
        summary = run_combo(combo_id, params["label"], params, YEARS)
        results.append(summary)

    elapsed = time.time() - t_start
    results.sort(key=lambda x: x["consistency_score"], reverse=True)

    print_leaderboard(results, "STAGE E LEADERBOARD — VIX-Dynamic DTE")
    save_stage("stageE", "VIX-dynamic DTE switching", results, elapsed)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


# ────────────────────────────────────────────────────────────
#  Main orchestrator
# ────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    print("\n" + "=" * 120)
    print("  STARTING EXP-602: COMPOUNDING + AGGRESSIVE IC PUSH")
    print("  Building on EXP-601 breakthrough: +18.2% avg with IC-enhanced configs")
    print("  Target: ≥+30% avg, 5/6 yrs profitable, DD ≤ -30%")
    print("  Carlos reference: +56.6% avg")
    print("=" * 120)

    # ── Stage A: Compounding ──
    a_results = stage_a()
    best_a = max(a_results, key=lambda x: x["total_return_pct"])
    best_a_params = dict(best_a["params"])
    best_a_params["compound"] = True
    print(f"\n  Best Stage A: {best_a['combo_id']} {best_a['label']}")
    print(f"    avg={best_a['avg_return']:+.1f}%, total={best_a['total_return_pct']:+.1f}%, "
          f"final=${best_a['final_capital']:,}, {best_a['years_profitable']} yrs+\n")

    # ── Stage B: Higher risk ──
    b_results = stage_b()
    best_b = max(b_results, key=lambda x: x["avg_return"])
    print(f"\n  Best Stage B: {best_b['combo_id']} {best_b['label']}")
    print(f"    avg={best_b['avg_return']:+.1f}%, {best_b['years_profitable']} yrs+, "
          f"DD={best_b['worst_drawdown']:.1f}%\n")

    # ── Stage C: DTE=15 compound ──
    c_results = stage_c()
    best_c = max(c_results, key=lambda x: x["total_return_pct"])
    best_c_params = dict(best_c["params"])
    best_c_params["compound"] = True
    print(f"\n  Best Stage C: {best_c['combo_id']} {best_c['label']}")
    print(f"    avg={best_c['avg_return']:+.1f}%, total={best_c['total_return_pct']:+.1f}%, "
          f"final=${best_c['final_capital']:,}, {best_c['years_profitable']} yrs+\n")

    # ── Stage D: Multi-ticker blend ──
    d_results = stage_d(best_a_params, best_c_params)
    best_d = max(d_results, key=lambda x: x["total_return_pct"])
    print(f"\n  Best Stage D: {best_d['combo_id']} {best_d['label']}")
    print(f"    avg={best_d['avg_return']:+.1f}%, total={best_d['total_return_pct']:+.1f}%, "
          f"final=${best_d['final_capital']:,}, {best_d['years_profitable']} yrs+\n")

    # ── Stage E: VIX-dynamic DTE ──
    e_results = stage_e()
    best_e = max(e_results, key=lambda x: x["total_return_pct"])
    print(f"\n  Best Stage E: {best_e['combo_id']} {best_e['label']}")
    print(f"    avg={best_e['avg_return']:+.1f}%, total={best_e['total_return_pct']:+.1f}%, "
          f"final=${best_e['final_capital']:,}, {best_e['years_profitable']} yrs+\n")

    # ── Grand Leaderboard ──
    all_results = a_results + b_results + c_results + d_results + e_results
    all_results.sort(key=lambda x: x["total_return_pct"], reverse=True)
    print_leaderboard(all_results, "GRAND LEADERBOARD — ALL STAGES (by total return)", top_n=25)

    # Also sort by consistency
    all_by_consistency = sorted(all_results, key=lambda x: x["consistency_score"], reverse=True)
    print_leaderboard(all_by_consistency,
                      "GRAND LEADERBOARD — ALL STAGES (by consistency)", top_n=15)

    # ── Final Summary ──
    total_elapsed = time.time() - t_total
    print(f"\n{'='*120}")
    print(f"  FINAL SUMMARY — EXP-602")
    print(f"{'='*120}")
    print(f"  Stage A (Compound DTE=35): {len(a_results)} combos, "
          f"best={best_a['combo_id']} total={best_a['total_return_pct']:+.1f}% "
          f"final=${best_a['final_capital']:,}")
    print(f"  Stage B (High Risk):       {len(b_results)} combos, "
          f"best={best_b['combo_id']} avg={best_b['avg_return']:+.1f}%")
    print(f"  Stage C (Compound DTE=15): {len(c_results)} combos, "
          f"best={best_c['combo_id']} total={best_c['total_return_pct']:+.1f}% "
          f"final=${best_c['final_capital']:,}")
    print(f"  Stage D (Multi-Ticker):    {len(d_results)} combos, "
          f"best={best_d['combo_id']} total={best_d['total_return_pct']:+.1f}%")
    print(f"  Stage E (VIX-DTE):         {len(e_results)} combos, "
          f"best={best_e['combo_id']} total={best_e['total_return_pct']:+.1f}%")
    print(f"")
    print(f"  TOP 5 OVERALL (by total return):")
    for i, r in enumerate(all_results[:5], 1):
        print(f"    {i}. {r['combo_id']} {r['label']}: "
              f"avg={r['avg_return']:+.1f}%, total={r['total_return_pct']:+.1f}%, "
              f"final=${r['final_capital']:,}, {r['years_profitable']} yrs+, "
              f"DD={r['worst_drawdown']:.1f}%")
    print(f"")
    print(f"  Carlos reference: ~+56.6% avg, ~+339.6% total (compound 6yr)")
    print(f"  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Results: results/exp602/sweep_stage[ABCDE].json")
    print(f"{'='*120}\n")

    # Save combined summary
    combined = {
        "experiment": "EXP-602",
        "description": "Compounding + aggressive IC push",
        "total_elapsed_seconds": round(total_elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "total_combos": len(all_results),
        "target": ">=+30% avg, 5/6 yrs, DD <= -30%",
        "carlos_reference": "+56.6% avg",
        "top_5_by_total_return": [
            {
                "combo_id": r["combo_id"],
                "label": r["label"],
                "avg_return": r["avg_return"],
                "total_return_pct": r["total_return_pct"],
                "final_capital": r["final_capital"],
                "years_profitable": r["years_profitable"],
                "worst_drawdown": r["worst_drawdown"],
                "consistency_score": r["consistency_score"],
            }
            for r in all_results[:5]
        ],
        "top_5_by_consistency": [
            {
                "combo_id": r["combo_id"],
                "label": r["label"],
                "avg_return": r["avg_return"],
                "total_return_pct": r["total_return_pct"],
                "final_capital": r["final_capital"],
                "years_profitable": r["years_profitable"],
                "worst_drawdown": r["worst_drawdown"],
                "consistency_score": r["consistency_score"],
            }
            for r in all_by_consistency[:5]
        ],
        "stage_bests": {
            "A": {"id": best_a["combo_id"], "avg": best_a["avg_return"],
                   "total": best_a["total_return_pct"], "final": best_a["final_capital"]},
            "B": {"id": best_b["combo_id"], "avg": best_b["avg_return"],
                   "total": best_b["total_return_pct"], "final": best_b["final_capital"]},
            "C": {"id": best_c["combo_id"], "avg": best_c["avg_return"],
                   "total": best_c["total_return_pct"], "final": best_c["final_capital"]},
            "D": {"id": best_d["combo_id"], "avg": best_d["avg_return"],
                   "total": best_d["total_return_pct"], "final": best_d["final_capital"]},
            "E": {"id": best_e["combo_id"], "avg": best_e["avg_return"],
                   "total": best_e["total_return_pct"], "final": best_e["final_capital"]},
        },
    }
    summary_path = OUTPUT_DIR / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"  Combined summary: {summary_path}")


if __name__ == "__main__":
    main()
