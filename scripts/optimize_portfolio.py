#!/usr/bin/env python3
"""
Portfolio Optimization across Experiments
==========================================
Loads experiment results from output/leaderboard.json, converts per-experiment
monthly P&L into return series, then runs PortfolioOptimizer to produce
allocation weights and combined portfolio metrics.

Supports all four optimization methods (max_sharpe, risk_parity,
equal_risk_contribution, min_variance) and regime-adaptive tilting.

Usage:
    python scripts/optimize_portfolio.py
    python scripts/optimize_portfolio.py --experiments regime_adaptive_20260312 endless_20260309_054247_4ce3
    python scripts/optimize_portfolio.py --method risk_parity --regime BULL_MACRO
    python scripts/optimize_portfolio.py --top 5       # auto-select top 5 by avg return
    python scripts/optimize_portfolio.py --output output/portfolio_allocation.json
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from compass.portfolio_optimizer import (
    EXPERIMENT_PROFILES,
    OptimizationResult,
    PortfolioOptimizer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LEADERBOARD = PROJECT_ROOT / "output" / "leaderboard.json"
METHODS = ["max_sharpe", "risk_parity", "equal_risk_contribution", "min_variance"]
DEFAULT_STARTING_CAPITAL = 100_000


# ── Data loading ────────────────────────────────────────────────────────────

def load_leaderboard(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def select_experiments(
    leaderboard: list,
    run_ids: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    min_years: int = 4,
    max_dd: float = -30.0,
) -> list:
    """Select experiments from leaderboard.

    If run_ids given, pick those.  Otherwise pick top_n by avg_return
    that pass quality filters (min years, max drawdown).
    """
    if run_ids:
        selected = [e for e in leaderboard if e["run_id"] in run_ids]
        missing = set(run_ids) - {e["run_id"] for e in selected}
        if missing:
            logger.warning("Run IDs not found in leaderboard: %s", missing)
        return selected

    # Filter by quality
    candidates = []
    for e in leaderboard:
        s = e.get("summary", {})
        yrs = len(e.get("years_run", []))
        dd = s.get("worst_dd", -100)
        if yrs >= min_years and dd >= max_dd:
            candidates.append(e)

    # Sort by avg_return descending
    candidates.sort(key=lambda x: x.get("summary", {}).get("avg_return", 0), reverse=True)

    n = top_n or 5
    return candidates[:n]


def experiments_to_monthly_returns(
    experiments: list,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Convert experiment results to aligned monthly return arrays.

    Returns:
        (returns_dict, month_labels) where returns_dict maps run_id to
        a numpy array of monthly returns (as fractions, not percentages).
        Only months present in ALL experiments are kept.
    """
    # Collect monthly P&L for each experiment
    exp_monthly: Dict[str, Dict[str, float]] = {}

    for exp in experiments:
        run_id = exp["run_id"]
        monthly_pnl: Dict[str, float] = {}

        for year_str, year_data in exp.get("results", {}).items():
            mp = year_data.get("monthly_pnl", {})
            cap = year_data.get("starting_capital", starting_capital)
            for month_key, month_data in mp.items():
                pnl = month_data["pnl"] if isinstance(month_data, dict) else month_data
                # Convert P&L to return fraction
                monthly_pnl[month_key] = pnl / cap

        if monthly_pnl:
            exp_monthly[run_id] = monthly_pnl

    if not exp_monthly:
        return {}, []

    # Find common months across all experiments
    common_months = None
    for run_id, mp in exp_monthly.items():
        months_set = set(mp.keys())
        if common_months is None:
            common_months = months_set
        else:
            common_months &= months_set

    if not common_months:
        logger.warning("No common months across experiments — using union with 0-fill")
        all_months = set()
        for mp in exp_monthly.values():
            all_months |= set(mp.keys())
        common_months = all_months

    sorted_months = sorted(common_months)

    returns_dict = {}
    for run_id, mp in exp_monthly.items():
        returns_dict[run_id] = np.array([mp.get(m, 0.0) for m in sorted_months])

    return returns_dict, sorted_months


# ── Portfolio metrics ───────────────────────────────────────────────────────

def compute_combined_backtest(
    experiments: list,
    weights: Dict[str, float],
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
) -> Dict:
    """Compute combined portfolio metrics from per-year experiment results.

    Blends annual return_pct, max_drawdown, and trades using allocation
    weights.
    """
    # Collect per-year blended metrics
    all_years = set()
    for exp in experiments:
        all_years |= set(exp.get("results", {}).keys())

    exp_map = {exp["run_id"]: exp for exp in experiments}
    yearly_results = {}

    for year in sorted(all_years):
        blended_return = 0.0
        blended_dd = 0.0
        total_trades = 0
        total_wins = 0
        contributors = 0

        for run_id, w in weights.items():
            exp = exp_map.get(run_id)
            if exp is None:
                continue
            yr_data = exp.get("results", {}).get(year)
            if yr_data is None:
                continue

            ret = yr_data.get("return_pct", 0.0)
            dd = yr_data.get("max_drawdown", 0.0)
            trades = yr_data.get("total_trades", 0)
            wr = yr_data.get("win_rate", 0.0)

            blended_return += w * ret
            blended_dd += w * dd
            total_trades += trades
            total_wins += int(round(trades * wr / 100))
            contributors += 1

        if contributors > 0:
            yearly_results[year] = {
                "return_pct": round(blended_return, 2),
                "max_drawdown": round(blended_dd, 2),
                "total_trades": total_trades,
                "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0.0,
            }

    # Summary
    returns = [yr["return_pct"] for yr in yearly_results.values()]
    drawdowns = [yr["max_drawdown"] for yr in yearly_results.values()]

    return {
        "yearly": yearly_results,
        "avg_return": round(np.mean(returns), 2) if returns else 0.0,
        "worst_dd": round(min(drawdowns), 2) if drawdowns else 0.0,
        "best_year": round(max(returns), 2) if returns else 0.0,
        "worst_year": round(min(returns), 2) if returns else 0.0,
        "years_profitable": sum(1 for r in returns if r > 0),
        "years_total": len(returns),
    }


# ── Printing ────────────────────────────────────────────────────────────────

def _bar(label, width=70):
    print(f"\n{'─' * width}")
    print(f"  {label}")
    print(f"{'─' * width}")


def print_experiment_summary(experiments: list):
    _bar("SELECTED EXPERIMENTS")
    header = f"  {'Run ID':<45s}  {'Ret%':>6s}  {'DD%':>6s}  {'Yrs':>3s}  {'Verdict':>8s}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    for exp in experiments:
        s = exp.get("summary", {})
        v = exp.get("validation", {})
        print(
            f"  {exp['run_id'][:45]:<45s}  {s.get('avg_return', 0):>6.1f}  "
            f"{s.get('worst_dd', 0):>6.1f}  {len(exp.get('years_run', [])):>3d}  "
            f"{v.get('verdict', '?'):>8s}"
        )


def print_optimization_results(results: List[Tuple[str, OptimizationResult]], experiments: list):
    _bar("OPTIMIZATION RESULTS")

    for method, result in results:
        print(f"\n  Method: {method}")
        print(f"  Regime: {result.regime}  |  Event scaling: {result.event_scaling}")
        print(f"  Sharpe: {result.metrics.get('sharpe_ratio', 0):.4f}  |  "
              f"Return: {result.metrics.get('annual_return', 0) * 100:.2f}%  |  "
              f"Vol: {result.metrics.get('annual_volatility', 0) * 100:.2f}%")

        print(f"\n  {'Experiment':<45s}  {'Weight':>7s}  {'Scaled':>7s}")
        print(f"  {'─' * 62}")
        for eid in sorted(result.weights):
            w = result.weights[eid]
            sw = result.scaled_weights[eid]
            bar_len = int(w * 40)
            print(f"  {eid[:45]:<45s}  {w:>6.1%}  {sw:>6.1%}  {'█' * bar_len}")

        print()


def print_combined_metrics(combined: Dict, method: str):
    _bar(f"COMBINED PORTFOLIO ({method})")
    print(f"  Avg annual return:  {combined['avg_return']:>7.2f}%")
    print(f"  Worst drawdown:     {combined['worst_dd']:>7.2f}%")
    print(f"  Best year:          {combined['best_year']:>7.2f}%")
    print(f"  Worst year:         {combined['worst_year']:>7.2f}%")
    print(f"  Years profitable:   {combined['years_profitable']}/{combined['years_total']}")

    yearly = combined.get("yearly", {})
    if yearly:
        print(f"\n  {'Year':<8s}  {'Return%':>8s}  {'MaxDD%':>8s}  {'Trades':>7s}  {'WR%':>5s}")
        print(f"  {'─' * 42}")
        for yr in sorted(yearly):
            yd = yearly[yr]
            print(
                f"  {yr:<8s}  {yd['return_pct']:>8.2f}  {yd['max_drawdown']:>8.2f}  "
                f"{yd['total_trades']:>7d}  {yd['win_rate']:>5.1f}"
            )


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run portfolio optimization across experiments")
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    parser.add_argument("--experiments", nargs="+", help="Specific run_ids to include")
    parser.add_argument("--top", type=int, default=5, help="Auto-select top N experiments")
    parser.add_argument("--method", choices=METHODS, default=None, help="Single method (default: run all)")
    parser.add_argument("--regime", choices=["BULL_MACRO", "NEUTRAL_MACRO", "BEAR_MACRO"], default="NEUTRAL_MACRO")
    parser.add_argument("--output", type=Path, help="Save results to JSON")
    args = parser.parse_args()

    if not args.leaderboard.exists():
        print(f"ERROR: leaderboard not found at {args.leaderboard}")
        sys.exit(1)

    # 1. Load and select experiments
    lb = load_leaderboard(args.leaderboard)
    experiments = select_experiments(lb, run_ids=args.experiments, top_n=args.top)

    if len(experiments) < 2:
        print(f"ERROR: need at least 2 experiments, got {len(experiments)}")
        sys.exit(1)

    print_experiment_summary(experiments)

    # 2. Build monthly returns
    returns_dict, months = experiments_to_monthly_returns(experiments)
    logger.info("Built returns: %d experiments × %d months", len(returns_dict), len(months))

    if not returns_dict:
        print("ERROR: could not extract monthly returns from experiments")
        sys.exit(1)

    # 3. Run optimization (mock event scaling to avoid import of live macro_db)
    methods_to_run = [args.method] if args.method else METHODS

    opt = PortfolioOptimizer(
        returns_dict,
        risk_free_rate=0.045,
        periods_per_year=12,  # monthly returns
    )

    all_results: List[Tuple[str, OptimizationResult]] = []
    for method in methods_to_run:
        with patch.object(PortfolioOptimizer, "get_event_scaling", return_value=1.0):
            result = opt.optimize(method=method, regime=args.regime, as_of=date.today())
        all_results.append((method, result))

    # 4. Print results
    print_optimization_results(all_results, experiments)

    # 5. Combined backtest metrics for best method (highest Sharpe)
    best_method, best_result = max(all_results, key=lambda x: x[1].metrics.get("sharpe_ratio", 0))
    combined = compute_combined_backtest(experiments, best_result.weights)
    print_combined_metrics(combined, best_method)

    # 6. Save if requested
    if args.output:
        output_data = {
            "generated": date.today().isoformat(),
            "regime": args.regime,
            "n_experiments": len(experiments),
            "n_months": len(months),
            "experiments": [e["run_id"] for e in experiments],
            "results": {},
        }
        for method, result in all_results:
            output_data["results"][method] = {
                "weights": result.weights,
                "scaled_weights": result.scaled_weights,
                "metrics": result.metrics,
                "event_scaling": result.event_scaling,
            }
        output_data["combined_backtest"] = combined

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n  Results saved to {args.output}")

    print()


if __name__ == "__main__":
    main()
