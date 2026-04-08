#!/usr/bin/env python3
"""
EXP-604: 6/6 Year Consistency — Carlos Directive.

HARD CONSTRAINT: Every single year (2020-2025) must be profitable.

Stage A (24): Conservative IC sweep (DTE=35, low risk, wide OTM)
Stage B (12): P0141 variants with compound (DTE=45)
Stage C (8):  Drawdown-adaptive (month-by-month, cut risk on DD)
Stage D (8):  Monthly circuit breaker (reduce risk after losing months)
Stage E (6):  P0141 + aggressive blend (capital-split)

Total: ~58 combos. 6/6 profitable is the ONLY metric that matters.
"""

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

OUTPUT_DIR = ROOT / "results" / "exp604"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────

def build_config(params, starting_capital=100_000):
    target_dte = params["target_dte"]
    min_dte = max(10, target_dte - 10)

    ic_cfg = {
        "enabled":                 params.get("ic_enabled", False),
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
            "vix_dte_threshold":   0,
            "dte_low_vix":         target_dte,
            "min_dte_low_vix":     min_dte,
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
            "drawdown_cb_pct":      params.get("drawdown_cb_pct", 25),
        },
        "backtest": {
            "starting_capital":   starting_capital,
            "commission_per_contract": 0.65,
            "slippage":           0.05,
            "exit_slippage":      0.10,
            "compound":           params.get("compound", True),
            "sizing_mode":        params.get("sizing_mode", "flat"),
            "slippage_multiplier": 1.0,
            "max_portfolio_exposure_pct": 100.0,
            "exclude_months":     [],
            "volume_gate":        False,
            "oi_gate":            False,
            "risk_cap":           25.0,
        },
    }


def _run_bt(ticker, start, end, params, starting_capital):
    from backtest.backtester import Backtester
    from shared.iron_vault import IronVault
    config = build_config(params, starting_capital=round(starting_capital))
    hd = IronVault.instance()
    otm_pct = params.get("otm_pct", 0.03)
    bt = Backtester(config, historical_data=hd, otm_pct=otm_pct)
    return bt.run_backtest(ticker, start, end) or {}


def run_compound_years(ticker, params, years, starting_capital=100_000):
    """Run years sequentially carrying forward capital."""
    current_capital = starting_capital
    year_results = {}

    for year in sorted(years):
        try:
            r = _run_bt(ticker, datetime(year, 1, 1), datetime(year, 12, 31),
                        params, current_capital)
            ret = r.get("return_pct", 0)
            ending = r.get("ending_capital", current_capital)
            year_results[str(year)] = {
                "return_pct": round(ret, 2),
                "trades": r.get("total_trades", 0),
                "ic_trades": r.get("iron_condor_trades", 0),
                "ic_win_rate": round(r.get("iron_condor_win_rate", 0), 1),
                "win_rate": round(r.get("win_rate", 0), 1),
                "max_drawdown": round(r.get("max_drawdown", 0), 2),
                "sharpe": round(r.get("sharpe_ratio", 0), 2),
                "starting_capital": round(current_capital),
                "ending_capital": round(ending),
            }
            current_capital = ending
            ic_flag = f" IC={r.get('iron_condor_trades',0)}" if r.get("iron_condor_trades", 0) > 0 else ""
            flag = "+" if ret > 0 else "—" if ret == 0 else "✗"
            print(f"    {year}: {ret:+6.1f}% ({r.get('total_trades',0):>3} tr{ic_flag}, "
                  f"WR {r.get('win_rate',0):.0f}%) ${current_capital:,.0f} {flag}")
        except Exception as e:
            year_results[str(year)] = {
                "return_pct": 0, "trades": 0, "ic_trades": 0,
                "error": str(e), "starting_capital": round(current_capital),
                "ending_capital": round(current_capital),
            }
            print(f"    {year}: ERROR {e}")
    return year_results, current_capital


def run_dd_adaptive(ticker, params, years, dd_threshold, risk_factor,
                    starting_capital=100_000):
    """Month-by-month runner with drawdown-adaptive risk reduction.
    When year-to-date return drops below dd_threshold, multiply risk by risk_factor.
    """
    current_capital = starting_capital
    year_results = {}
    base_risk = params["max_risk_per_trade"]

    for year in sorted(years):
        year_start_cap = current_capital
        ytd_return = 0.0
        risk_reduced = False
        year_trades = 0
        year_ic_trades = 0
        year_wrs = []
        year_dds = []
        monthly_rets = []

        for month in range(1, 13):
            m_start = datetime(year, month, 1)
            if month == 12:
                m_end = datetime(year, 12, 31)
            else:
                m_end = datetime(year, month + 1, 1) - timedelta(days=1)

            # Adapt risk based on YTD drawdown
            if ytd_return < dd_threshold and not risk_reduced:
                adj_params = {**params, "max_risk_per_trade": base_risk * risk_factor}
                risk_reduced = True
            elif risk_reduced and ytd_return > 0:
                # Recover risk if back to positive
                adj_params = {**params, "max_risk_per_trade": base_risk}
                risk_reduced = False
            elif risk_reduced:
                adj_params = {**params, "max_risk_per_trade": base_risk * risk_factor}
            else:
                adj_params = params

            try:
                r = _run_bt(ticker, m_start, m_end, adj_params, current_capital)
                ret = r.get("return_pct", 0)
                ending = r.get("ending_capital", current_capital)
                trades = r.get("total_trades", 0)
                ic_trades = r.get("iron_condor_trades", 0)
                wr = r.get("win_rate", 0)
                dd = r.get("max_drawdown", 0)

                ytd_return = (ending - year_start_cap) / year_start_cap * 100
                current_capital = ending
                year_trades += trades
                year_ic_trades += ic_trades
                if trades > 0:
                    year_wrs.append(wr)
                if dd < 0:
                    year_dds.append(dd)
                monthly_rets.append(ret)
            except Exception:
                monthly_rets.append(0)

        # Year summary
        year_ret = (current_capital - year_start_cap) / year_start_cap * 100
        avg_wr = sum(year_wrs) / len(year_wrs) if year_wrs else 0
        worst_dd = min(year_dds) if year_dds else 0

        year_results[str(year)] = {
            "return_pct": round(year_ret, 2),
            "trades": year_trades,
            "ic_trades": year_ic_trades,
            "win_rate": round(avg_wr, 1),
            "max_drawdown": round(worst_dd, 2),
            "starting_capital": round(year_start_cap),
            "ending_capital": round(current_capital),
            "risk_reduced": risk_reduced,
            "monthly_rets": [round(r, 2) for r in monthly_rets],
        }
        flag = "+" if year_ret > 0 else "✗"
        risk_tag = " [RISK CUT]" if risk_reduced else ""
        print(f"    {year}: {year_ret:+6.1f}% ({year_trades:>3} tr, IC={year_ic_trades}, "
              f"WR {avg_wr:.0f}%) ${current_capital:,.0f} {flag}{risk_tag}")

    return year_results, current_capital


def run_monthly_breaker(ticker, params, years, reduction_factor, skip_after,
                        starting_capital=100_000):
    """Month-by-month with circuit breaker: reduce risk after consecutive losing months."""
    current_capital = starting_capital
    year_results = {}
    base_risk = params["max_risk_per_trade"]

    for year in sorted(years):
        year_start_cap = current_capital
        consecutive_losses = 0
        year_trades = 0
        year_ic_trades = 0
        year_wrs = []
        year_dds = []
        monthly_rets = []
        months_skipped = 0

        for month in range(1, 13):
            m_start = datetime(year, month, 1)
            if month == 12:
                m_end = datetime(year, 12, 31)
            else:
                m_end = datetime(year, month + 1, 1) - timedelta(days=1)

            # Circuit breaker logic
            if consecutive_losses >= skip_after:
                # Skip this month entirely
                monthly_rets.append(0)
                months_skipped += 1
                consecutive_losses = 0  # Reset after skip
                continue

            # Reduce risk based on consecutive losses
            factor = reduction_factor ** consecutive_losses
            adj_risk = max(0.5, base_risk * factor)  # Floor at 0.5%
            adj_params = {**params, "max_risk_per_trade": adj_risk}

            try:
                r = _run_bt(ticker, m_start, m_end, adj_params, current_capital)
                ret = r.get("return_pct", 0)
                ending = r.get("ending_capital", current_capital)
                trades = r.get("total_trades", 0)
                ic_trades = r.get("iron_condor_trades", 0)
                wr = r.get("win_rate", 0)
                dd = r.get("max_drawdown", 0)

                current_capital = ending
                year_trades += trades
                year_ic_trades += ic_trades
                if trades > 0:
                    year_wrs.append(wr)
                if dd < 0:
                    year_dds.append(dd)
                monthly_rets.append(ret)

                if ret < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
            except Exception:
                monthly_rets.append(0)
                consecutive_losses += 1

        # Year summary
        year_ret = (current_capital - year_start_cap) / year_start_cap * 100
        avg_wr = sum(year_wrs) / len(year_wrs) if year_wrs else 0
        worst_dd = min(year_dds) if year_dds else 0

        year_results[str(year)] = {
            "return_pct": round(year_ret, 2),
            "trades": year_trades,
            "ic_trades": year_ic_trades,
            "win_rate": round(avg_wr, 1),
            "max_drawdown": round(worst_dd, 2),
            "starting_capital": round(year_start_cap),
            "ending_capital": round(current_capital),
            "months_skipped": months_skipped,
            "monthly_rets": [round(r, 2) for r in monthly_rets],
        }
        flag = "+" if year_ret > 0 else "✗"
        skip_tag = f" [skipped {months_skipped}mo]" if months_skipped else ""
        print(f"    {year}: {year_ret:+6.1f}% ({year_trades:>3} tr, IC={year_ic_trades}, "
              f"WR {avg_wr:.0f}%) ${current_capital:,.0f} {flag}{skip_tag}")

    return year_results, current_capital


def run_blend(params_safe, params_aggr, years, safe_weight,
              starting_capital=100_000):
    """Capital-split blend: safe leg + aggressive leg, compound independently."""
    aggr_weight = 1.0 - safe_weight
    safe_cap = starting_capital * safe_weight
    aggr_cap = starting_capital * aggr_weight
    year_results = {}

    for year in sorted(years):
        try:
            r_safe = _run_bt("SPY", datetime(year, 1, 1), datetime(year, 12, 31),
                             params_safe, safe_cap)
            r_aggr = _run_bt("SPY", datetime(year, 1, 1), datetime(year, 12, 31),
                             params_aggr, aggr_cap)

            safe_end = r_safe.get("ending_capital", safe_cap)
            aggr_end = r_aggr.get("ending_capital", aggr_cap)
            total_start = safe_cap + aggr_cap
            total_end = safe_end + aggr_end
            blended_ret = (total_end - total_start) / total_start * 100

            total_trades = r_safe.get("total_trades", 0) + r_aggr.get("total_trades", 0)
            total_ic = r_safe.get("iron_condor_trades", 0) + r_aggr.get("iron_condor_trades", 0)

            year_results[str(year)] = {
                "return_pct": round(blended_ret, 2),
                "trades": total_trades,
                "ic_trades": total_ic,
                "win_rate": 0,
                "max_drawdown": round(min(r_safe.get("max_drawdown", 0),
                                          r_aggr.get("max_drawdown", 0)), 2),
                "starting_capital": round(total_start),
                "ending_capital": round(total_end),
                "safe_leg": {"ret": round(r_safe.get("return_pct", 0), 2),
                             "start": round(safe_cap), "end": round(safe_end)},
                "aggr_leg": {"ret": round(r_aggr.get("return_pct", 0), 2),
                             "start": round(aggr_cap), "end": round(aggr_end)},
            }
            safe_cap = safe_end
            aggr_cap = aggr_end

            flag = "+" if blended_ret > 0 else "✗"
            print(f"    {year}: {blended_ret:+6.1f}% ({total_trades} tr, IC={total_ic}) "
                  f"${total_end:,.0f}  [safe:{r_safe.get('return_pct',0):+.1f}% / "
                  f"aggr:{r_aggr.get('return_pct',0):+.1f}%] {flag}")
        except Exception as e:
            year_results[str(year)] = {
                "return_pct": 0, "trades": 0, "ic_trades": 0,
                "error": str(e),
                "starting_capital": round(safe_cap + aggr_cap),
                "ending_capital": round(safe_cap + aggr_cap),
            }
            print(f"    {year}: ERROR {e}")

    final_capital = safe_cap + aggr_cap
    return year_results, final_capital


def summarize(combo_id, label, year_results, params=None):
    valid = {k: v for k, v in year_results.items() if "error" not in v}
    rets = [v["return_pct"] for v in valid.values()]
    dds = [v["max_drawdown"] for v in valid.values()]
    wrs = [v["win_rate"] for v in valid.values() if v.get("trades", 0) > 0]
    trades_list = [v.get("trades", 0) for v in valid.values()]
    ic_list = [v.get("ic_trades", 0) for v in valid.values()]

    avg_ret = sum(rets) / len(rets) if rets else 0
    years_profitable = sum(1 for r in rets if r > 0)
    worst_dd = min(dds) if dds else 0
    avg_wr = sum(wrs) / len(wrs) if wrs else 0
    total_trades = sum(trades_list)
    total_ic = sum(ic_list)

    last_yr = str(max(YEARS))
    final_capital = year_results.get(last_yr, {}).get("ending_capital", 100_000)
    total_return = (final_capital - 100_000) / 100_000 * 100

    is_6_6 = years_profitable == len(rets) and len(rets) == 6

    return {
        "combo_id": combo_id,
        "label": label,
        "params": params or {},
        "avg_return": round(avg_ret, 2),
        "total_return_pct": round(total_return, 1),
        "final_capital": round(final_capital),
        "total_trades": total_trades,
        "total_ic_trades": total_ic,
        "ic_pct": round(total_ic / total_trades * 100, 1) if total_trades > 0 else 0,
        "years_profitable": f"{years_profitable}/{len(rets)}",
        "is_6_6": is_6_6,
        "worst_drawdown": round(worst_dd, 2),
        "avg_win_rate": round(avg_wr, 1),
        "by_year": year_results,
    }


def print_summary(s):
    flag = "6/6 ★★★" if s["is_6_6"] else s["years_profitable"]
    compound_str = f" final=${s['final_capital']:,}" if s.get("final_capital") else ""
    print(f"  => AVG {s['avg_return']:+.1f}%, TOTAL {s['total_return_pct']:+.1f}%,"
          f"{compound_str}, {flag}, DD {s['worst_drawdown']:.1f}%\n")


def print_leaderboard(results, title, top_n=20):
    # Partition: 6/6 first, then rest
    six_six = [r for r in results if r["is_6_6"]]
    rest = [r for r in results if not r["is_6_6"]]
    six_six.sort(key=lambda x: x["avg_return"], reverse=True)
    rest.sort(key=lambda x: int(x["years_profitable"].split("/")[0]), reverse=True)

    ordered = six_six + rest

    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"  6/6 configs: {len(six_six)} | Not 6/6: {len(rest)}")
    print(f"{'='*120}")
    print(f"  {'Rk':<3} {'ID':<7} {'6/6':>3} {'Avg':>7} {'Total':>8} {'Final$':>10} "
          f"{'Yr+':>4} {'WR':>5} {'DD':>7} {'Trds':>5} {'IC%':>4}  Config")
    print(f"  {'─'*120}")
    for rank, r in enumerate(ordered[:top_n], 1):
        tag = " ★" if r["is_6_6"] else ""
        print(f"  {rank:<3} {r['combo_id']:<7} {'YES' if r['is_6_6'] else 'NO':>3} "
              f"{r['avg_return']:>+6.1f}% {r['total_return_pct']:>+7.1f}% "
              f"${r['final_capital']:>9,} "
              f"{r['years_profitable']:>4} {r['avg_win_rate']:>4.0f}% "
              f"{r['worst_drawdown']:>6.1f}% "
              f"{r['total_trades']:>5} {r['ic_pct']:>3.0f}%  "
              f"{r['label'][:55]}{tag}")


def save_stage(stage_name, description, results, elapsed, extra=None):
    output = {
        "experiment": f"EXP-604-{stage_name}",
        "description": description,
        "years": YEARS,
        "total_combos": len(results),
        "six_six_count": sum(1 for r in results if r["is_6_6"]),
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
#  Stage A: Conservative IC Sweep (24 combos)
# ────────────────────────────────────────────────────────────

def stage_a():
    ic_modes = [True, False]
    otms = [0.05, 0.07]
    risks = [1.0, 1.5, 2.0]
    pts = [40, 50]

    combos = []
    for ic in ic_modes:
        for otm in otms:
            for risk in risks:
                for pt in pts:
                    ic_tag = "IC" if ic else "noIC"
                    combos.append({
                        "label": f"{ic_tag} OTM={otm*100:.0f}% risk={risk}% PT={pt}% DTE=35",
                        "target_dte": 35, "spread_width": 5, "otm_pct": otm,
                        "profit_target": pt, "stop_loss_multiplier": 2.5,
                        "max_risk_per_trade": risk, "direction": "both",
                        "regime_mode": "combo", "min_credit_pct": 5,
                        "compound": True, "max_contracts": 25,
                        "ic_enabled": ic, "ic_neutral_regime_only": True,
                        "ic_min_combined_credit_pct": 10,
                    })

    print(f"\n{'='*120}")
    print(f"  STAGE A: Conservative IC Sweep (DTE=35)")
    print(f"  ic=[on,off] × OTM=[5%,7%] × risk=[1,1.5,2%] × PT=[40,50%] = {len(combos)} combos")
    print(f"  HARD GATE: 6/6 years profitable")
    print(f"{'='*120}\n")

    results = []
    t0 = time.time()
    for i, p in enumerate(combos):
        cid = f"A{i:03d}"
        print(f"  [{i+1}/{len(combos)}] {cid}: {p['label']}")
        yr, _ = run_compound_years("SPY", p, YEARS)
        s = summarize(cid, p["label"], yr, p)
        print_summary(s)
        results.append(s)

    elapsed = time.time() - t0
    print_leaderboard(results, "STAGE A — Conservative IC Sweep")
    save_stage("stageA", "Conservative IC sweep DTE=35", results, elapsed)

    six_six = [r for r in results if r["is_6_6"]]
    print(f"\n  6/6 configs found: {len(six_six)}/{len(results)}")
    if six_six:
        best = max(six_six, key=lambda x: x["avg_return"])
        print(f"  Best 6/6: {best['combo_id']} {best['label']} -> "
              f"+{best['avg_return']:.1f}% avg, ${best['final_capital']:,}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


# ────────────────────────────────────────────────────────────
#  Stage B: P0141 Variants with Compound (12 combos)
# ────────────────────────────────────────────────────────────

def stage_b():
    ic_modes = [True, False]
    risks = [2.0, 3.0]
    sls = [2.0, 2.5, 3.0]

    combos = []
    for ic in ic_modes:
        for risk in risks:
            for sl in sls:
                ic_tag = "IC" if ic else "noIC"
                combos.append({
                    "label": f"P0141-var {ic_tag} risk={risk}% SL={sl}x DTE=45 OTM=5%",
                    "target_dte": 45, "spread_width": 5, "otm_pct": 0.05,
                    "profit_target": 50, "stop_loss_multiplier": sl,
                    "max_risk_per_trade": risk, "direction": "both",
                    "regime_mode": "combo", "min_credit_pct": 10,
                    "compound": True, "max_contracts": 25,
                    "ic_enabled": ic, "ic_neutral_regime_only": True,
                    "ic_min_combined_credit_pct": 10,
                })

    print(f"\n{'='*120}")
    print(f"  STAGE B: P0141 Variants with Compound (DTE=45)")
    print(f"  P0141 base: DTE=45, W=$5, OTM=5%, min_credit=10% (the ONLY known 6/6 config)")
    print(f"  ic=[on,off] × risk=[2,3%] × SL=[2.0,2.5,3.0] = {len(combos)} combos")
    print(f"{'='*120}\n")

    results = []
    t0 = time.time()
    for i, p in enumerate(combos):
        cid = f"B{i:03d}"
        print(f"  [{i+1}/{len(combos)}] {cid}: {p['label']}")
        yr, _ = run_compound_years("SPY", p, YEARS)
        s = summarize(cid, p["label"], yr, p)
        print_summary(s)
        results.append(s)

    elapsed = time.time() - t0
    print_leaderboard(results, "STAGE B — P0141 Variants")
    save_stage("stageB", "P0141 variants with compound DTE=45", results, elapsed)

    six_six = [r for r in results if r["is_6_6"]]
    print(f"\n  6/6 configs found: {len(six_six)}/{len(results)}")
    if six_six:
        best = max(six_six, key=lambda x: x["avg_return"])
        print(f"  Best 6/6: {best['combo_id']} {best['label']} -> "
              f"+{best['avg_return']:.1f}% avg, ${best['final_capital']:,}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


# ────────────────────────────────────────────────────────────
#  Stage C: Drawdown-Adaptive (8 combos)
# ────────────────────────────────────────────────────────────

def stage_c(base_configs):
    """Drawdown-adaptive: cut risk when YTD DD exceeds threshold."""
    dd_thresholds = [-3, -5]
    risk_factors = [0.5, 0.25]

    print(f"\n{'='*120}")
    print(f"  STAGE C: Drawdown-Adaptive (month-by-month)")
    print(f"  When YTD return drops below threshold → multiply risk by factor")
    print(f"  dd_threshold=[-3%,-5%] × risk_factor=[0.5,0.25] × 2 base configs = 8 combos")
    print(f"{'='*120}\n")

    results = []
    t0 = time.time()
    idx = 0
    for base_label, base_params in base_configs[:2]:
        for dd_t in dd_thresholds:
            for rf in risk_factors:
                cid = f"C{idx:03d}"
                label = f"DD-adapt [{base_label}] dd<{dd_t}%->risk×{rf}"
                print(f"  [{idx+1}/8] {cid}: {label}")
                yr, _ = run_dd_adaptive("SPY", base_params, YEARS, dd_t, rf)
                s = summarize(cid, label, yr, {**base_params, "dd_threshold": dd_t,
                                                "risk_factor": rf})
                print_summary(s)
                results.append(s)
                idx += 1

    elapsed = time.time() - t0
    print_leaderboard(results, "STAGE C — Drawdown-Adaptive")
    save_stage("stageC", "Drawdown-adaptive month-by-month", results, elapsed)

    six_six = [r for r in results if r["is_6_6"]]
    print(f"\n  6/6 configs found: {len(six_six)}/{len(results)}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


# ────────────────────────────────────────────────────────────
#  Stage D: Monthly Circuit Breaker (8 combos)
# ────────────────────────────────────────────────────────────

def stage_d(base_configs):
    """Monthly circuit breaker: reduce risk after consecutive losing months."""
    reduction_factors = [0.5, 0.33]
    skip_afters = [3, 4]

    print(f"\n{'='*120}")
    print(f"  STAGE D: Monthly Circuit Breaker")
    print(f"  After N consecutive losing months → skip. After each loss → risk × factor^N")
    print(f"  factor=[0.5,0.33] × skip_after=[3,4] × 2 base configs = 8 combos")
    print(f"{'='*120}\n")

    results = []
    t0 = time.time()
    idx = 0
    for base_label, base_params in base_configs[:2]:
        for rf in reduction_factors:
            for sa in skip_afters:
                cid = f"D{idx:03d}"
                label = f"Breaker [{base_label}] factor={rf} skip@{sa}"
                print(f"  [{idx+1}/8] {cid}: {label}")
                yr, _ = run_monthly_breaker("SPY", base_params, YEARS, rf, sa)
                s = summarize(cid, label, yr, {**base_params, "reduction_factor": rf,
                                                "skip_after": sa})
                print_summary(s)
                results.append(s)
                idx += 1

    elapsed = time.time() - t0
    print_leaderboard(results, "STAGE D — Monthly Circuit Breaker")
    save_stage("stageD", "Monthly circuit breaker", results, elapsed)

    six_six = [r for r in results if r["is_6_6"]]
    print(f"\n  6/6 configs found: {len(six_six)}/{len(results)}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


# ────────────────────────────────────────────────────────────
#  Stage E: P0141 + Aggressive Blend (6 combos)
# ────────────────────────────────────────────────────────────

def stage_e(best_6_6_params, best_aggressive_params):
    """Blend safe (6/6 floor) + aggressive (upside) configs."""
    # P0141-like safe config (known 6/6)
    p0141_safe = {
        "target_dte": 45, "spread_width": 5, "otm_pct": 0.05,
        "profit_target": 50, "stop_loss_multiplier": 2.5,
        "max_risk_per_trade": 2.0, "direction": "both",
        "regime_mode": "combo", "min_credit_pct": 10,
        "compound": True, "max_contracts": 25,
        "ic_enabled": False,
    }

    # Q1 aggressive config
    q1_aggr = {
        "target_dte": 15, "spread_width": 5, "otm_pct": 0.05,
        "profit_target": 40, "stop_loss_multiplier": 3.0,
        "max_risk_per_trade": 5.0, "direction": "both",
        "regime_mode": "combo", "min_credit_pct": 5,
        "compound": True, "max_contracts": 25,
        "ic_enabled": True, "ic_neutral_regime_only": True,
        "ic_min_combined_credit_pct": 10,
    }

    combos = [
        ("E000", "80% P0141 + 20% Q1-aggr", p0141_safe, q1_aggr, 0.80),
        ("E001", "70% P0141 + 30% Q1-aggr", p0141_safe, q1_aggr, 0.70),
        ("E002", "60% P0141 + 40% Q1-aggr", p0141_safe, q1_aggr, 0.60),
        ("E003", "80% P0141 + 20% A003-config", p0141_safe, best_aggressive_params, 0.80),
        ("E004", f"70% best6/6 + 30% Q1-aggr", best_6_6_params, q1_aggr, 0.70),
        ("E005", f"best6/6 solo (baseline)", best_6_6_params, best_6_6_params, 1.0),
    ]

    print(f"\n{'='*120}")
    print(f"  STAGE E: P0141 + Aggressive Blend")
    print(f"  Capital split: safe leg (6/6 floor) + aggressive leg (upside)")
    print(f"  6 combos with varying weights")
    print(f"{'='*120}\n")

    results = []
    t0 = time.time()
    for cid, label, safe, aggr, safe_w in combos:
        print(f"  {cid}: {label}")
        if safe_w >= 1.0:
            yr, _ = run_compound_years("SPY", safe, YEARS)
        else:
            yr, _ = run_blend(safe, aggr, YEARS, safe_w)
        s = summarize(cid, label, yr, {"safe_weight": safe_w})
        print_summary(s)
        results.append(s)

    elapsed = time.time() - t0
    print_leaderboard(results, "STAGE E — P0141 + Aggressive Blend")
    save_stage("stageE", "P0141 + aggressive blend", results, elapsed)

    six_six = [r for r in results if r["is_6_6"]]
    print(f"\n  6/6 configs found: {len(six_six)}/{len(results)}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    print("\n" + "=" * 120)
    print("  EXP-604: 6/6 YEAR CONSISTENCY — CARLOS DIRECTIVE")
    print("  HARD CONSTRAINT: Every single year 2020-2025 must be profitable.")
    print("  Consistency > raw return. No exceptions.")
    print("=" * 120)

    # ── Stage A ──
    a_results = stage_a()
    a_six = [r for r in a_results if r["is_6_6"]]

    # ── Stage B ──
    b_results = stage_b()
    b_six = [r for r in b_results if r["is_6_6"]]

    # Pick best configs for Stages C/D
    all_ab = a_results + b_results
    all_ab_six = [r for r in all_ab if r["is_6_6"]]

    if all_ab_six:
        best1 = max(all_ab_six, key=lambda x: x["avg_return"])
        # Get second-best (different from first)
        remaining = [r for r in all_ab_six if r["combo_id"] != best1["combo_id"]]
        best2 = max(remaining, key=lambda x: x["avg_return"]) if remaining else best1
        base_configs = [
            (best1["combo_id"], best1["params"]),
            (best2["combo_id"], best2["params"]),
        ]
        print(f"\n  Best 6/6 for Stages C/D: {best1['combo_id']} (+{best1['avg_return']:.1f}%), "
              f"{best2['combo_id']} (+{best2['avg_return']:.1f}%)")
    else:
        # Fallback: use best by years_profitable
        all_ab.sort(key=lambda x: (int(x["years_profitable"].split("/")[0]),
                                    x["avg_return"]), reverse=True)
        best1, best2 = all_ab[0], all_ab[1]
        base_configs = [
            (best1["combo_id"], best1["params"]),
            (best2["combo_id"], best2["params"]),
        ]
        print(f"\n  No 6/6 found in A+B. Using best: {best1['combo_id']} "
              f"({best1['years_profitable']}), {best2['combo_id']} ({best2['years_profitable']})")

    # ── Stage C ──
    c_results = stage_c(base_configs)

    # ── Stage D ──
    d_results = stage_d(base_configs)

    # ── Stage E ──
    best_6_6 = best1["params"] if all_ab_six else base_configs[0][1]
    # A003 config (best balanced from EXP-603)
    a003_params = {
        "target_dte": 35, "spread_width": 5, "otm_pct": 0.03,
        "profit_target": 40, "stop_loss_multiplier": 3.0,
        "max_risk_per_trade": 2.0, "direction": "both",
        "regime_mode": "combo", "min_credit_pct": 5,
        "compound": True, "max_contracts": 25,
        "ic_enabled": True, "ic_neutral_regime_only": True,
        "ic_min_combined_credit_pct": 10,
    }
    e_results = stage_e(best_6_6, a003_params)

    # ── Grand Summary ──
    all_results = a_results + b_results + c_results + d_results + e_results
    all_six = [r for r in all_results if r["is_6_6"]]

    total_elapsed = time.time() - t_total

    print(f"\n{'='*120}")
    print(f"  GRAND SUMMARY — EXP-604: 6/6 Year Consistency")
    print(f"{'='*120}")
    print(f"  Total combos tested: {len(all_results)}")
    print(f"  6/6 configs found: {len(all_six)}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    if all_six:
        all_six.sort(key=lambda x: x["avg_return"], reverse=True)
        print(f"\n  ★★★ ALL 6/6 CONFIGS (ranked by avg return) ★★★")
        for i, r in enumerate(all_six, 1):
            print(f"  {i}. {r['combo_id']} {r['label'][:60]}")
            print(f"     avg={r['avg_return']:+.1f}%, total={r['total_return_pct']:+.1f}%, "
                  f"final=${r['final_capital']:,}, DD={r['worst_drawdown']:.1f}%")
            for yr in sorted(r["by_year"].keys()):
                y = r["by_year"][yr]
                print(f"       {yr}: {y['return_pct']:+.1f}% "
                      f"({y.get('trades',0)} tr, IC={y.get('ic_trades',0)})")
    else:
        print(f"\n  ✗ NO 6/6 configs found. Closest:")
        almost = sorted(all_results,
                        key=lambda x: (int(x["years_profitable"].split("/")[0]),
                                       x["avg_return"]), reverse=True)
        for i, r in enumerate(almost[:5], 1):
            neg_years = [yr for yr, y in r["by_year"].items()
                         if y.get("return_pct", 0) < 0 and "error" not in y]
            print(f"  {i}. {r['combo_id']} {r['label'][:55]}: "
                  f"{r['years_profitable']}, avg={r['avg_return']:+.1f}%, "
                  f"neg years: {', '.join(neg_years)}")

    # Grand leaderboard
    print_leaderboard(all_results,
                      "GRAND LEADERBOARD — ALL STAGES (6/6 first, then by years profitable)",
                      top_n=25)

    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*120}\n")

    # Save combined
    combined = {
        "experiment": "EXP-604",
        "description": "6/6 year consistency — Carlos directive",
        "hard_constraint": "Every year 2020-2025 must be profitable",
        "total_combos": len(all_results),
        "six_six_count": len(all_six),
        "total_elapsed_seconds": round(total_elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "six_six_configs": [
            {k: v for k, v in r.items() if k != "by_year"}
            for r in sorted(all_six, key=lambda x: x["avg_return"], reverse=True)
        ] if all_six else [],
        "best_near_miss": [
            {k: v for k, v in r.items() if k != "by_year"}
            for r in sorted(all_results,
                            key=lambda x: (int(x["years_profitable"].split("/")[0]),
                                           x["avg_return"]), reverse=True)[:5]
        ],
    }
    path = OUTPUT_DIR / "sweep_summary.json"
    with open(path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"  Summary saved: {path}")


if __name__ == "__main__":
    main()
