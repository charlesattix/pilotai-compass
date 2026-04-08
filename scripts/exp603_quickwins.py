#!/usr/bin/env python3
"""
EXP-603: Quick Wins — 5 targeted configs from EXP-602 gap analysis.

Q1: C015 + PT=40%  (DTE=15, compound, risk=5%, SL=3.0x, OTM=5% — change PT 50→40)
Q2: C015 + max_contracts=50  (unthrottle compounding at high equity)
Q3: A003 + PT=40%  (DTE=35, compound, risk=2%, SL=3.0x, OTM=3% — best balanced)
Q4: B010 + compound=True  (risk=5%, SL=2.5x, PT=40%, DTE=35 — Stage B winner)
Q5: Hybrid DTE  (DTE=15 IC-focused 60% capital + DTE=35 directional 40% capital)

All run compound carry-forward across 2020-2025.
Baseline: C015 = +248.3% total, $348K final, +36.2% avg, 3/6 yrs, DD=-52.4%
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

OUTPUT_DIR = ROOT / "results" / "exp603"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


# ────────────────────────────────────────────────────────────
#  Helpers (shared with EXP-602)
# ────────────────────────────────────────────────────────────

def build_config(params, starting_capital=100_000):
    target_dte = params["target_dte"]
    min_dte = max(10, target_dte - 10)

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
            "drawdown_cb_pct":      25,
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


def run_compound_years(ticker, params, years, starting_capital=100_000):
    """Run years sequentially carrying forward capital."""
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
            current_capital = ending

            ic_flag = f" IC={ic_trades}" if ic_trades > 0 else ""
            flag = "+" if ret > 0 else "-"
            print(f"    {year}: {ret:+6.1f}% ({trades:>3} tr{ic_flag}, "
                  f"WR {wr:.0f}%, DD {dd:.1f}%) ${current_capital:,.0f} {flag}")
        except Exception as e:
            year_results[str(year)] = {
                "return_pct": 0, "trades": 0, "ic_trades": 0,
                "error": str(e),
                "starting_capital": round(current_capital),
                "ending_capital": round(current_capital),
            }
            print(f"    {year}: ERROR {e}")

    return year_results, current_capital


def run_hybrid_compound(params_ic, params_dir, years, ic_weight=0.6,
                        starting_capital=100_000):
    """Run two configs (IC-focused + directional) with capital split, compound."""
    from backtest.backtester import Backtester
    from shared.iron_vault import IronVault

    hd = IronVault.instance()
    dir_weight = 1.0 - ic_weight
    ic_cap = starting_capital * ic_weight
    dir_cap = starting_capital * dir_weight
    year_results = {}

    for year in sorted(years):
        try:
            # IC-focused leg (DTE=15)
            cfg_ic = build_config(params_ic, starting_capital=round(ic_cap))
            bt_ic = Backtester(cfg_ic, historical_data=hd,
                               otm_pct=params_ic.get("otm_pct", 0.05))
            r_ic = bt_ic.run_backtest("SPY", datetime(year, 1, 1),
                                       datetime(year, 12, 31)) or {}

            # Directional leg (DTE=35)
            cfg_dir = build_config(params_dir, starting_capital=round(dir_cap))
            bt_dir = Backtester(cfg_dir, historical_data=hd,
                                otm_pct=params_dir.get("otm_pct", 0.03))
            r_dir = bt_dir.run_backtest("SPY", datetime(year, 1, 1),
                                         datetime(year, 12, 31)) or {}

            ic_end = r_ic.get("ending_capital", ic_cap)
            dir_end = r_dir.get("ending_capital", dir_cap)
            total_start = ic_cap + dir_cap
            total_end = ic_end + dir_end
            blended_ret = (total_end - total_start) / total_start * 100

            ic_trades = r_ic.get("total_trades", 0)
            dir_trades = r_dir.get("total_trades", 0)
            ic_ic = r_ic.get("iron_condor_trades", 0)
            dir_ic = r_dir.get("iron_condor_trades", 0)
            total_trades = ic_trades + dir_trades
            total_ic = ic_ic + dir_ic

            ic_wr_list = []
            if ic_trades > 0:
                ic_wr_list.append(r_ic.get("win_rate", 0))
            if dir_trades > 0:
                ic_wr_list.append(r_dir.get("win_rate", 0))
            avg_wr = sum(ic_wr_list) / len(ic_wr_list) if ic_wr_list else 0

            worst_dd = min(r_ic.get("max_drawdown", 0),
                           r_dir.get("max_drawdown", 0))

            year_results[str(year)] = {
                "return_pct": round(blended_ret, 2),
                "trades": total_trades,
                "ic_trades": total_ic,
                "win_rate": round(avg_wr, 1),
                "max_drawdown": round(worst_dd, 2),
                "sharpe": 0,
                "starting_capital": round(total_start),
                "ending_capital": round(total_end),
                "ic_leg": {"start": round(ic_cap), "end": round(ic_end),
                           "ret": round(r_ic.get("return_pct", 0), 2),
                           "trades": ic_trades, "ic": ic_ic},
                "dir_leg": {"start": round(dir_cap), "end": round(dir_end),
                            "ret": round(r_dir.get("return_pct", 0), 2),
                            "trades": dir_trades, "ic": dir_ic},
            }

            # Carry forward compound
            ic_cap = ic_end
            dir_cap = dir_end

            print(f"    {year}: {blended_ret:+6.1f}% ({total_trades} tr, IC={total_ic}, "
                  f"WR {avg_wr:.0f}%, DD {worst_dd:.1f}%) "
                  f"${total_end:,.0f}  "
                  f"[IC-leg: {r_ic.get('return_pct',0):+.1f}% / "
                  f"Dir-leg: {r_dir.get('return_pct',0):+.1f}%]")
        except Exception as e:
            year_results[str(year)] = {
                "return_pct": 0, "trades": 0, "ic_trades": 0,
                "error": str(e),
                "starting_capital": round(ic_cap + dir_cap),
                "ending_capital": round(ic_cap + dir_cap),
            }
            print(f"    {year}: ERROR {e}")

    final_capital = ic_cap + dir_cap
    return year_results, final_capital


def summarize(combo_id, label, year_results, params=None):
    valid = {k: v for k, v in year_results.items() if "error" not in v}
    rets = [v["return_pct"] for v in valid.values()]
    trades_list = [v["trades"] for v in valid.values()]
    ic_trades_list = [v.get("ic_trades", 0) for v in valid.values()]
    dds = [v["max_drawdown"] for v in valid.values()]
    wrs = [v["win_rate"] for v in valid.values() if v["trades"] > 0]

    avg_ret = sum(rets) / len(rets) if rets else 0
    total_trades = sum(trades_list)
    total_ic = sum(ic_trades_list)
    years_profitable = sum(1 for r in rets if r > 0)
    worst_dd = min(dds) if dds else 0
    avg_wr = sum(wrs) / len(wrs) if wrs else 0

    last_yr = str(max(YEARS))
    final_capital = year_results.get(last_yr, {}).get("ending_capital", 100_000)
    total_return = (final_capital - 100_000) / 100_000 * 100

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
        "worst_drawdown": round(worst_dd, 2),
        "avg_win_rate": round(avg_wr, 1),
        "by_year": year_results,
    }


# ────────────────────────────────────────────────────────────
#  Config definitions
# ────────────────────────────────────────────────────────────

# C015 base: DTE=15, compound, risk=5%, SL=3.0x, W=$5, OTM=5%, PT=50%
C015_BASE = {
    "target_dte": 15, "spread_width": 5, "otm_pct": 0.05,
    "profit_target": 50, "stop_loss_multiplier": 3.0,
    "max_risk_per_trade": 5.0, "direction": "both",
    "regime_mode": "combo", "min_credit_pct": 5,
    "compound": True, "max_contracts": 25,
    "ic_enabled": True, "ic_neutral_regime_only": True,
    "ic_min_combined_credit_pct": 10,
}

# A003 base: DTE=35, compound, risk=2%, SL=3.0x, W=$5, OTM=3%, PT=50%
A003_BASE = {
    "target_dte": 35, "spread_width": 5, "otm_pct": 0.03,
    "profit_target": 50, "stop_loss_multiplier": 3.0,
    "max_risk_per_trade": 2.0, "direction": "both",
    "regime_mode": "combo", "min_credit_pct": 5,
    "compound": True, "max_contracts": 25,
    "ic_enabled": True, "ic_neutral_regime_only": True,
    "ic_min_combined_credit_pct": 10,
}

# B010 base: risk=5%, SL=2.5x, PT=40%, DTE=35, W=$5, OTM=3% — was non-compound
B010_BASE = {
    "target_dte": 35, "spread_width": 5, "otm_pct": 0.03,
    "profit_target": 40, "stop_loss_multiplier": 2.5,
    "max_risk_per_trade": 5.0, "direction": "both",
    "regime_mode": "combo", "min_credit_pct": 5,
    "compound": True, "max_contracts": 25,
    "ic_enabled": True, "ic_neutral_regime_only": True,
    "ic_min_combined_credit_pct": 10,
}


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    print("\n" + "=" * 110)
    print("  EXP-603: QUICK WINS (5 configs)")
    print("  Baseline: C015 = +248.3% total, $348K, +36.2% avg, 3/6 yrs, DD=-52.4%")
    print("  Carlos reference: +56.6% avg, ~$440K (6yr compound)")
    print("=" * 110)

    results = []

    # ── Q1: C015 + PT=40% ──
    print(f"\n  Q1: C015 + PT=40% (was 50%)")
    print(f"  {'─'*60}")
    q1_params = {**C015_BASE, "profit_target": 40}
    yr1, cap1 = run_compound_years("SPY", q1_params, YEARS)
    s1 = summarize("Q1", "C015 + PT=40%", yr1, q1_params)
    results.append(s1)
    print(f"  => AVG {s1['avg_return']:+.1f}%, TOTAL {s1['total_return_pct']:+.1f}%, "
          f"FINAL ${s1['final_capital']:,}, {s1['years_profitable']} yrs+, "
          f"DD {s1['worst_drawdown']:.1f}%\n")

    # ── Q2: C015 + max_contracts=50 ──
    print(f"  Q2: C015 + max_contracts=50 (was 25)")
    print(f"  {'─'*60}")
    q2_params = {**C015_BASE, "max_contracts": 50}
    yr2, cap2 = run_compound_years("SPY", q2_params, YEARS)
    s2 = summarize("Q2", "C015 + max_contracts=50", yr2, q2_params)
    results.append(s2)
    print(f"  => AVG {s2['avg_return']:+.1f}%, TOTAL {s2['total_return_pct']:+.1f}%, "
          f"FINAL ${s2['final_capital']:,}, {s2['years_profitable']} yrs+, "
          f"DD {s2['worst_drawdown']:.1f}%\n")

    # ── Q3: A003 + PT=40% ──
    print(f"  Q3: A003 (balanced) + PT=40% (was 50%)")
    print(f"  {'─'*60}")
    q3_params = {**A003_BASE, "profit_target": 40}
    yr3, cap3 = run_compound_years("SPY", q3_params, YEARS)
    s3 = summarize("Q3", "A003 + PT=40%", yr3, q3_params)
    results.append(s3)
    print(f"  => AVG {s3['avg_return']:+.1f}%, TOTAL {s3['total_return_pct']:+.1f}%, "
          f"FINAL ${s3['final_capital']:,}, {s3['years_profitable']} yrs+, "
          f"DD {s3['worst_drawdown']:.1f}%\n")

    # ── Q4: B010 + compound=True ──
    print(f"  Q4: B010 (risk=5% SL=2.5x PT=40%) + compound=True")
    print(f"  {'─'*60}")
    q4_params = {**B010_BASE}  # compound already True
    yr4, cap4 = run_compound_years("SPY", q4_params, YEARS)
    s4 = summarize("Q4", "B010 + compound", yr4, q4_params)
    results.append(s4)
    print(f"  => AVG {s4['avg_return']:+.1f}%, TOTAL {s4['total_return_pct']:+.1f}%, "
          f"FINAL ${s4['final_capital']:,}, {s4['years_profitable']} yrs+, "
          f"DD {s4['worst_drawdown']:.1f}%\n")

    # ── Q5: Hybrid DTE (IC@15 + Directional@35) ──
    print(f"  Q5: Hybrid DTE — IC-focused DTE=15 (60%) + Directional DTE=35 (40%)")
    print(f"  {'─'*60}")
    # IC leg: DTE=15, high IC rate, OTM=5%
    q5_ic_params = {**C015_BASE}
    # Directional leg: DTE=35, OTM=3%, IC disabled for pure directional
    q5_dir_params = {
        "target_dte": 35, "spread_width": 5, "otm_pct": 0.03,
        "profit_target": 50, "stop_loss_multiplier": 3.0,
        "max_risk_per_trade": 5.0, "direction": "both",
        "regime_mode": "combo", "min_credit_pct": 5,
        "compound": True, "max_contracts": 25,
        "ic_enabled": False,  # directional only
        "ic_neutral_regime_only": False,
    }
    yr5, cap5 = run_hybrid_compound(q5_ic_params, q5_dir_params, YEARS,
                                     ic_weight=0.6)
    s5 = summarize("Q5", "Hybrid DTE=15 IC(60%) + DTE=35 Dir(40%)", yr5,
                   {"ic_leg": "C015 DTE=15", "dir_leg": "DTE=35 no-IC",
                    "ic_weight": 0.6, "dir_weight": 0.4})
    results.append(s5)
    print(f"  => AVG {s5['avg_return']:+.1f}%, TOTAL {s5['total_return_pct']:+.1f}%, "
          f"FINAL ${s5['final_capital']:,}, {s5['years_profitable']} yrs+, "
          f"DD {s5['worst_drawdown']:.1f}%\n")

    # ── Comparison Table ──
    total_elapsed = time.time() - t_total
    print(f"\n{'='*110}")
    print(f"  EXP-603 QUICK WINS — COMPARISON TABLE")
    print(f"{'='*110}")
    print(f"  {'ID':<5} {'Config':<50} {'Avg':>7} {'Total':>8} {'Final$':>12} "
          f"{'Yr+':>4} {'DD':>7} {'IC%':>5}  {'vs C015'}")
    print(f"  {'─'*120}")

    # Baseline
    print(f"  {'base':<5} {'C015 baseline (PT=50, max25)':<50} "
          f"{'+36.2%':>7} {'+248.3%':>8} {'$348,337':>12} {'3/6':>4} "
          f"{'-52.4%':>7} {'75.9%':>5}  —")

    for r in results:
        total_delta = r["total_return_pct"] - 248.3
        delta_str = f"{total_delta:+.1f}pp"
        print(f"  {r['combo_id']:<5} {r['label']:<50} "
              f"{r['avg_return']:>+6.1f}% {r['total_return_pct']:>+7.1f}% "
              f"${r['final_capital']:>11,} "
              f"{r['years_profitable']:>4} {r['worst_drawdown']:>6.1f}% "
              f"{r['ic_pct']:>4.0f}%  {delta_str}")

    # Carlos comparison
    print(f"\n  {'ref':<5} {'Carlos reference':<50} "
          f"{'+56.6%':>7} {'~+340%':>8} {'~$440,000':>12} {'4/6':>4} "
          f"{'?':>7} {'?':>5}  target")

    print(f"\n  Elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*110}\n")

    # Save results
    output = {
        "experiment": "EXP-603",
        "description": "Quick wins from EXP-602 gap analysis",
        "baseline": {
            "config": "C015",
            "avg_return": 36.2,
            "total_return_pct": 248.3,
            "final_capital": 348337,
            "years_profitable": "3/6",
            "worst_drawdown": -52.4,
        },
        "carlos_reference": {"avg_return": 56.6, "total_return_approx": 340},
        "total_elapsed_seconds": round(total_elapsed),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    path = OUTPUT_DIR / "quickwins_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {path}")


if __name__ == "__main__":
    main()
