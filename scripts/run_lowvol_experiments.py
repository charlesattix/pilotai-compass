#!/usr/bin/env python3
"""
run_lowvol_experiments.py — Low-Vol Strategy sweep (EXP-501 through EXP-505)

Runs all 5 low-vol complement experiments against EXP-500 baseline and prints
a comparison table showing year-by-year returns + trade counts.

Usage:
    python3 scripts/run_lowvol_experiments.py
    python3 scripts/run_lowvol_experiments.py --years 2023,2024,2025   # quick check
    python3 scripts/run_lowvol_experiments.py --exp 501,504             # subset
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from scripts.run_optimization import run_year, compute_summary

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

EXPERIMENTS = {
    "EXP-500": "configs/exp_500_realdata_champion.json",
    "EXP-501": "configs/exp_501_lowvol_adaptive_credit.json",
    "EXP-502": "configs/exp_502_lowvol_longer_dte.json",
    "EXP-503": "configs/exp_503_lowvol_narrow_spread.json",
    "EXP-504": "configs/exp_504_lowvol_combo.json",
    "EXP-505": "configs/exp_505_lowvol_tighter_otm.json",
    "EXP-506": "configs/exp_506_lowvol_tighter_strike.json",
    "EXP-507": "configs/exp_507_lowvol_wider_otm.json",
    "EXP-508": "configs/exp_508_lowvol_ma80.json",
    "EXP-509": "configs/exp_509_lowvol_ma80_tighter.json",
    "EXP-510": "configs/exp_510_lowvol_ma80_dte30.json",
    "EXP-511": "configs/exp_511_lowvol_ic_fallback.json",
    "EXP-512": "configs/exp_512_lowvol_ic_only.json",
    "EXP-513": "configs/exp_513_lowvol_dte30_protected.json",
    "EXP-514": "configs/exp_514_lowvol_dte30_protected_v2.json",
    "EXP-519": "configs/exp_519_lowvol_dte21.json",
    "EXP-520": "configs/exp_520_lowvol_dte25.json",
}


def load_exp_params(config_path: str) -> dict:
    """Load a flat JSON config as params dict (same format as run_optimization)."""
    with open(ROOT / config_path) as f:
        raw = json.load(f)
    # The JSON files store flat params (same keys as BASELINE_PARAMS / run_optimization).
    # regime_config is nested — pass through as-is.
    return raw


def run_experiment(name: str, config_path: str, years: list) -> dict:
    params = load_exp_params(config_path)
    results = {}
    print(f"\n{'─'*72}")
    print(f"  {name} — {config_path}")
    key_params = (
        f"  otm={params.get('otm_pct', 0.03)*100:.0f}%  "
        f"dte={params.get('target_dte', 15)}  "
        f"width=${params.get('spread_width', 10)}  "
        f"credit≥{params.get('min_credit_pct', 7)}%  "
        f"sl={params.get('stop_loss_multiplier', 1.25)}x"
    )
    if params.get('vix_adaptive_credit'):
        vac = params['vix_adaptive_credit']
        key_params += (
            f"  | adaptive-credit: VIX<{vac.get('threshold',18)}→"
            f"{vac.get('low_vol_pct',4)}% (IC:{vac.get('low_vol_ic_pct',12)}%)"
        )
    if params.get('vix_dte_threshold'):
        key_params += (
            f"  | low-vol DTE: VIX<{params['vix_dte_threshold']}→"
            f"{params.get('dte_low_vix',30)}d"
        )
    print(key_params)
    print(f"{'─'*72}")

    for year in years:
        t0 = time.time()
        print(f"    {year}...", end=" ", flush=True)
        try:
            r = run_year("SPY", year, params)
            elapsed = time.time() - t0
            ret = r.get("return_pct", 0)
            trades = r.get("total_trades", 0)
            dd = r.get("max_drawdown", 0)
            print(f"{ret:+.1f}%  {trades} trades  DD={dd:.1f}%  ({elapsed:.0f}s)")
            results[str(year)] = r
        except Exception as e:
            print(f"ERROR: {e}")
            results[str(year)] = {"return_pct": 0, "total_trades": 0, "max_drawdown": 0, "error": str(e)}

    return results


def print_comparison_table(all_results: dict, years: list):
    exp_names = list(all_results.keys())

    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║  LOW-VOL STRATEGY COMPARISON — EXP-500 vs EXP-501..505" + " "*22 + "║")
    print("╠" + "═"*78 + "╣")

    # Header
    header = f"  {'Experiment':<12}"
    for y in years:
        header += f"  {y:>8}"
    header += f"  {'Avg':>7}  {'Worst':>7}  {'MaxDD':>7}  {'Trades':>7}"
    print("║" + header + "  ║")
    print("╠" + "═"*78 + "╣")

    for name, results in all_results.items():
        rets = []
        trades_total = 0
        max_dd = 0.0
        row = f"  {name:<12}"
        for y in years:
            r = results.get(str(y), {})
            ret = r.get("return_pct", 0.0)
            rets.append(ret)
            trades_total += r.get("total_trades", 0)
            dd = r.get("max_drawdown", 0.0)
            if dd < max_dd:
                max_dd = dd
            marker = "✓" if ret > 0 else "✗"
            row += f"  {ret:>+6.1f}{marker}"
        avg = sum(rets) / len(rets) if rets else 0
        worst = min(rets) if rets else 0
        avg_trades = trades_total // len(years)
        row += f"  {avg:>+6.1f}%  {worst:>+6.1f}%  {max_dd:>6.1f}%  {avg_trades:>6}"
        print("║" + row + "  ║")

    print("╚" + "═"*78 + "╝")

    # 2023-2025 focus
    print("\n  LOW-VOL YEARS (2023-2025) DETAIL:")
    print(f"  {'Experiment':<12}  {'2023':>8}  {'2024':>8}  {'2025':>8}  {'3yr avg':>8}")
    print("  " + "─"*52)
    low_vol_years = [y for y in years if y in (2023, 2024, 2025)]
    for name, results in all_results.items():
        rets_lv = [results.get(str(y), {}).get("return_pct", 0.0) for y in low_vol_years]
        avg_lv = sum(rets_lv) / len(rets_lv) if rets_lv else 0
        vals = "  ".join(f"{r:>+7.1f}%" for r in rets_lv)
        print(f"  {name:<12}  {vals}  {avg_lv:>+7.1f}%")


def save_results(all_results: dict, years: list):
    out_path = ROOT / "output" / "lowvol_experiment_results.json"
    summary = {}
    for name, results in all_results.items():
        rets = [results.get(str(y), {}).get("return_pct", 0.0) for y in years]
        dd = min((results.get(str(y), {}).get("max_drawdown", 0.0) for y in years), default=0.0)
        lv_rets = [results.get(str(y), {}).get("return_pct", 0.0) for y in years if y in (2023,2024,2025)]
        summary[name] = {
            "year_by_year": {str(y): results.get(str(y), {}).get("return_pct", 0.0) for y in years},
            "avg_return": round(sum(rets)/len(rets), 2) if rets else 0,
            "min_return": round(min(rets), 2) if rets else 0,
            "worst_dd": round(dd, 2),
            "lowvol_avg": round(sum(lv_rets)/len(lv_rets), 2) if lv_rets else 0,
            "years_positive": sum(1 for r in rets if r > 0),
            "raw_results": results,
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default=",".join(str(y) for y in YEARS),
                        help="Comma-separated years, e.g. 2023,2024,2025")
    parser.add_argument("--exp", default=None,
                        help="Comma-separated experiment IDs, e.g. 500,501,504")
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")]

    selected_exps = EXPERIMENTS
    if args.exp:
        ids = set(f"EXP-{x.strip()}" for x in args.exp.split(","))
        selected_exps = {k: v for k, v in EXPERIMENTS.items() if k in ids}

    print(f"\n{'═'*72}")
    print(f"  LOW-VOL STRATEGY SWEEP — {len(selected_exps)} experiments × {len(years)} years")
    print(f"  Years: {years}")
    print(f"{'═'*72}")

    all_results = {}
    for name, config_path in selected_exps.items():
        all_results[name] = run_experiment(name, config_path, years)

    print_comparison_table(all_results, years)
    save_results(all_results, years)


if __name__ == "__main__":
    main()
