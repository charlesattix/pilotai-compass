#!/usr/bin/env python3
"""
grid_search_plateau.py — Stability Plateau Grid Search (P4)

Purpose:
    Map the strategy's sensitivity surface by running all combinations of:
      - DTE:    21, 28, 35, 42, 49, 56 (6 values)
      - Width:  $3, $4, $5, $6, $7, $8 (6 values)
      - Credit: 4%, 6%, 8%, 10%        (4 values)
    Total: 6 × 6 × 4 = 144 combinations

    Goal: Find a "basin" — a region where nearby params also perform well.
    If exp_059 sits at the peak of a narrow spike (not a plateau), the strategy
    is fragile.  A wide plateau of consistently positive returns = TRUE EDGE.

Usage:
    python3 scripts/grid_search_plateau.py --years 2022,2023,2024
    python3 scripts/grid_search_plateau.py --years 2020,2021,2022,2023,2024,2025

Output:
    output/plateau_grid_{run_id}.json  — full per-combo results
    Prints ASCII heat map of avg return by DTE vs width (at each credit level)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grid")

# ── Grid parameters ──────────────────────────────────────────────────────────

DTE_VALUES    = [21, 28, 35, 42, 49, 56]
WIDTH_VALUES  = [3, 4, 5, 6, 7, 8]
CREDIT_VALUES = [4, 6, 8, 10]

# Base config derived from exp_059 (champion), with DTE/width/credit varied by grid
BASE_PARAMS = {
    "use_delta_selection": False,
    "otm_pct": 0.03,
    "stop_loss_multiplier": 2.5,
    "profit_target": 50,
    "max_risk_per_trade": 10.0,
    "max_contracts": 25,
    "drawdown_cb_pct": 40,
    "direction": "both",
    "trend_ma_period": 200,
    "compound": True,
    "sizing_mode": "flat",
    "iron_condor_enabled": True,
    "ic_min_combined_credit_pct": 12,
    "iv_rank_min_entry": 0,
}


def run_combo(dte: int, width: int, credit_pct: int, years: list, use_real_data: bool) -> dict:
    """Run a single (DTE, width, credit) combo over all years."""
    from scripts.run_optimization import run_year

    params = {
        **BASE_PARAMS,
        "target_dte":    dte,
        "min_dte":       max(7, dte - 10),
        "spread_width":  width,
        "min_credit_pct": credit_pct,
    }

    per_year = {}
    for year in years:
        result = run_year(
            ticker="SPY",
            year=year,
            params=params,
            use_real_data=use_real_data,
        )
        per_year[year] = result or {}

    returns = [per_year[y].get("total_return_pct", per_year[y].get("return_pct", 0.0)) for y in years]
    avg_return = sum(returns) / len(returns) if returns else 0.0
    profitable_years = sum(1 for r in returns if r > 0)
    max_dds = [per_year[y].get("max_drawdown_pct", per_year[y].get("max_drawdown", 0.0)) for y in years]
    worst_dd = min(max_dds) if max_dds else 0.0
    trade_counts = [per_year[y].get("trade_count", per_year[y].get("total_trades", 0)) for y in years]
    total_trades = sum(trade_counts)

    return {
        "dte": dte,
        "width": width,
        "credit_pct": credit_pct,
        "avg_return_pct": round(avg_return, 2),
        "worst_drawdown_pct": round(worst_dd, 2),
        "profitable_years": profitable_years,
        "total_years": len(years),
        "total_trades": total_trades,
        "per_year": {str(y): {
            "return_pct": round(returns[i], 2),
            "trades": trade_counts[i],
        } for i, y in enumerate(years)},
    }


def _print_heatmap(results: list, credit_pct: int, years: list):
    """Print ASCII heat map of avg_return by DTE (rows) vs width (cols)."""
    print(f"\n  AVG RETURN HEAT MAP — credit>={credit_pct}%  (years: {years})")
    header = "DTE/Width"
    print(f"  {header:>9}  " + "  ".join(f"${w:>3}" for w in WIDTH_VALUES))
    print(f"  {'-' * (8 + len(WIDTH_VALUES) * 7)}")
    for dte in DTE_VALUES:
        row = []
        for w in WIDTH_VALUES:
            combo = next(
                (r for r in results if r["dte"] == dte and r["width"] == w and r["credit_pct"] == credit_pct),
                None,
            )
            if combo is None:
                row.append("  N/A")
            else:
                v = combo["avg_return_pct"]
                tag = f"{v:+5.0f}%" if abs(v) < 1000 else " big%"
                row.append(tag)
        print(f"  DTE={dte:>2}:  " + "  ".join(row))


def main():
    parser = argparse.ArgumentParser(description="Stability plateau grid search (P4)")
    parser.add_argument("--years", default="2022,2023,2024",
                        help="Comma-separated years (default: 2022,2023,2024)")
    parser.add_argument("--heuristic", action="store_true", help="Fast heuristic mode (no Polygon)")
    parser.add_argument("--run-id", default=None, help="Custom run ID")
    parser.add_argument("--dte-values", default=None,
                        help="Override DTE list (comma-separated, e.g. 21,28,35)")
    parser.add_argument("--width-values", default=None,
                        help="Override width list (comma-separated, e.g. 3,5,7)")
    parser.add_argument("--credit-values", default=None,
                        help="Override credit% list (comma-separated, e.g. 4,6,8)")
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")]
    dte_vals    = [int(x) for x in args.dte_values.split(",")]    if args.dte_values    else DTE_VALUES
    width_vals  = [int(x) for x in args.width_values.split(",")]  if args.width_values  else WIDTH_VALUES
    credit_vals = [int(x) for x in args.credit_values.split(",")] if args.credit_values else CREDIT_VALUES

    use_real_data = not args.heuristic
    total_combos = len(dte_vals) * len(width_vals) * len(credit_vals)

    run_id = args.run_id or f"plateau_{datetime.now().strftime('%m%d_%H%M')}"

    print(f"""
════════════════════════════════════════════════════════════════════════
  STABILITY PLATEAU GRID SEARCH — {run_id}
  DTE     : {dte_vals}
  Width   : {width_vals}
  Credit  : {credit_vals}%
  Years   : {years}
  Combos  : {total_combos}
  Mode    : {'heuristic (fast)' if not use_real_data else 'real data (Polygon)'}
════════════════════════════════════════════════════════════════════════
""")

    all_results = []
    t_start = time.time()
    done = 0

    for credit_pct in credit_vals:
        for dte in dte_vals:
            for width in width_vals:
                t0 = time.time()
                result = run_combo(dte, width, credit_pct, years, use_real_data)
                elapsed = time.time() - t0
                done += 1
                print(
                    f"  [{done:3d}/{total_combos}] DTE={dte:>2} W=${width} CR={credit_pct:>2}%: "
                    f"avg={result['avg_return_pct']:+6.1f}%  "
                    f"DD={result['worst_drawdown_pct']:5.1f}%  "
                    f"profitable={result['profitable_years']}/{result['total_years']}  "
                    f"({elapsed:.0f}s)"
                )
                all_results.append(result)

    total_elapsed = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    profitable_combos = [r for r in all_results if r["avg_return_pct"] > 0]
    all_positive = [r for r in all_results if r["profitable_years"] == r["total_years"]]

    sorted_results = sorted(all_results, key=lambda r: r["avg_return_pct"], reverse=True)
    top5 = sorted_results[:5]

    print(f"""
════════════════════════════════════════════════════════════════════════
  GRID SEARCH RESULTS — {run_id}
  Total combos: {total_combos}  |  Profitable avg return: {len(profitable_combos)}/{total_combos}
  All-years profitable: {len(all_positive)}/{total_combos}
────────────────────────────────────────────────────────────────────────
  TOP 5 COMBOS:""")
    for i, r in enumerate(top5, 1):
        print(f"  #{i}: DTE={r['dte']:>2} W=${r['width']} CR={r['credit_pct']:>2}%  "
              f"avg={r['avg_return_pct']:+6.1f}%  DD={r['worst_drawdown_pct']:5.1f}%  "
              f"wins={r['profitable_years']}/{r['total_years']}")

    # Heat maps per credit level
    for credit_pct in credit_vals:
        _print_heatmap(all_results, credit_pct, years)

    print(f"""
────────────────────────────────────────────────────────────────────────
  Elapsed: {total_elapsed/60:.1f} min
════════════════════════════════════════════════════════════════════════
""")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "run_id": run_id,
        "years": years,
        "dte_values": dte_vals,
        "width_values": width_vals,
        "credit_values": credit_vals,
        "total_combos": total_combos,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "profitable_combos": len(profitable_combos),
            "all_years_profitable": len(all_positive),
            "top5": top5,
        },
        "all_results": all_results,
    }

    out_path = OUTPUT / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
