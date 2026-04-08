#!/usr/bin/env python3
"""
run_monte_carlo.py — Monte Carlo Robustness Runner

Modes
-----
--mc-mode dte   (legacy, default)
    Randomizes DTE only — U(dte_lo, dte_hi) per trading day.  Fast, backward-compatible.

--mc-mode full
    Simultaneously randomizes DTE + slippage + entry timing + trade skipping + credit
    assumption.  Requires --n-seeds ≥ 1000 for stable percentile estimates.

    Default jitter knobs in full mode (override with --* flags):
      --slippage-jitter 0.50   ±50 % of configured slippage per trade
      --entry-day-jitter 2     ±2 calendar days on expiration reference date
      --trade-skip-pct 0.15    15 % of qualifying trades randomly dropped
      --credit-jitter 0.10     ±0.10 on BACKTEST_CREDIT_FRACTION (heuristic mode)

Carlos's directive: "If median drops from 66% to 20%, then 20% IS THE REAL EDGE."

Usage:
    python3 scripts/run_monte_carlo.py --config configs/exp_213_champion_maxc100.json
    python3 scripts/run_monte_carlo.py --config configs/exp_213_champion_maxc100.json \\
        --mc-mode full --n-seeds 1000 --heuristic

Output:
    output/mc_{run_id}.json  — full per-seed results
    Prints percentile table: P5, P25, P50, P75, P95 for avg return, DD, and Sharpe
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

# Load .env so POLYGON_API_KEY is available when running via Claude or cron
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
OUTPUT.mkdir(exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mc")


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def _build_config(params: dict) -> dict:
    """Mirror of run_optimization.py's _build_config."""
    from scripts.run_optimization import _build_config as _bc
    return _bc(params)


def run_one_seed(seed: int, params: dict, years: list, use_real_data: bool) -> dict:
    """Run the full multi-year backtest with this seed, return per-year results."""
    from scripts.run_optimization import run_year
    per_year = {}
    for year in years:
        result = run_year(
            ticker="SPY",
            year=year,
            params=params,
            use_real_data=use_real_data,
            seed=seed,
        )
        per_year[year] = result
    return per_year


def _percentile(data: list, p: float) -> float:
    """Simple percentile without numpy dependency."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[-1]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo robustness runner")
    parser.add_argument("--config", required=True, help="Config JSON file")
    parser.add_argument("--n-seeds", type=int, default=100, help="Number of seeds (default: 100)")
    parser.add_argument("--years", default="2020,2021,2022,2023,2024,2025",
                        help="Comma-separated years (default: 2020-2025)")
    parser.add_argument("--heuristic", action="store_true", help="Use heuristic mode (fast, no Polygon)")
    parser.add_argument("--run-id", default=None, help="Custom run ID for output file")
    parser.add_argument("--dte-lo", type=int, default=28, help="DTE range lower bound (default: 28)")
    parser.add_argument("--dte-hi", type=int, default=42, help="DTE range upper bound (default: 42)")
    # ── New in BT-3 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--mc-mode", choices=["dte", "full"], default="dte",
        help="'dte' = legacy DTE-only jitter (default); 'full' = all jitters simultaneously",
    )
    parser.add_argument("--slippage-jitter", type=float, default=0.50,
                        help="Full mode: ±fraction of slippage per trade (default: 0.50 = ±50%%)")
    parser.add_argument("--entry-day-jitter", type=int, default=2,
                        help="Full mode: ±calendar days on expiration ref date (default: 2)")
    parser.add_argument("--trade-skip-pct", type=float, default=0.15,
                        help="Full mode: probability of randomly skipping a trade (default: 0.15)")
    parser.add_argument("--credit-jitter", type=float, default=0.10,
                        help="Full mode: ±addition to BACKTEST_CREDIT_FRACTION (default: 0.10)")
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")]
    params = _load_config(args.config)

    # Inject Monte Carlo config
    if "monte_carlo" not in params:
        params["monte_carlo"] = {}
    params["monte_carlo"]["dte_lo"] = args.dte_lo
    params["monte_carlo"]["dte_hi"] = args.dte_hi
    params["monte_carlo"]["mode"] = args.mc_mode

    if args.mc_mode == "full":
        params["monte_carlo"]["slippage_jitter_pct"] = args.slippage_jitter
        params["monte_carlo"]["entry_day_jitter"]    = args.entry_day_jitter
        params["monte_carlo"]["trade_skip_pct"]      = args.trade_skip_pct
        params["monte_carlo"]["credit_jitter"]       = args.credit_jitter

    use_real_data = not args.heuristic

    config_name = Path(args.config).stem
    run_id = args.run_id or f"mc_{args.mc_mode}_{config_name}_seeds{args.n_seeds}_{datetime.now().strftime('%H%M')}"

    _mode_details = (
        f"  DTE     : U({args.dte_lo}, {args.dte_hi}) sampled per trading day\n"
        f"  Slippage jitter : ±{args.slippage_jitter:.0%}\n"
        f"  Entry-day jitter: ±{args.entry_day_jitter} days\n"
        f"  Trade-skip pct  : {args.trade_skip_pct:.0%}\n"
        f"  Credit jitter   : ±{args.credit_jitter}\n"
    ) if args.mc_mode == "full" else (
        f"  DTE     : U({args.dte_lo}, {args.dte_hi}) sampled per trading day\n"
    )

    print(f"""
════════════════════════════════════════════════════════════════════════
  MONTE CARLO ROBUSTNESS — {run_id}
  Config  : {args.config}
  MC mode : {args.mc_mode.upper()}
  Seeds   : 0 to {args.n_seeds - 1} (n={args.n_seeds})
{_mode_details}  Years   : {years}
  Data    : {'heuristic (fast)' if not use_real_data else 'real data (Polygon)'}
════════════════════════════════════════════════════════════════════════
""")

    all_results = []
    t_start = time.time()

    for seed in range(args.n_seeds):
        t_seed = time.time()
        per_year = run_one_seed(seed, params, years, use_real_data)

        # Compute summary stats for this seed
        returns = [per_year[y].get("return_pct", 0.0) for y in years]
        max_dds = [per_year[y].get("max_drawdown", 0.0) for y in years]
        trade_counts = [per_year[y].get("total_trades", 0) for y in years]
        sharpes = [per_year[y].get("sharpe_ratio", 0.0) for y in years]
        avg_return = sum(returns) / len(returns) if returns else 0.0
        worst_dd = min(max_dds) if max_dds else 0.0
        total_trades = sum(trade_counts)
        profitable_years = sum(1 for r in returns if r > 0)
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0

        seed_result = {
            "seed": seed,
            "avg_return_pct": avg_return,
            "worst_drawdown_pct": worst_dd,
            "total_trades": total_trades,
            "profitable_years": profitable_years,
            "avg_sharpe": avg_sharpe,
            "per_year": {str(y): per_year[y] for y in years},
        }
        all_results.append(seed_result)

        elapsed = time.time() - t_seed
        print(f"  Seed {seed:3d}:  avg={avg_return:+.1f}%  trades={total_trades:3d}  "
              f"worst_DD={worst_dd:.1f}%  sharpe={avg_sharpe:.2f}  ({elapsed:.0f}s)")

    # ── Aggregate statistics ─────────────────────────────────────────────────
    avg_returns = [r["avg_return_pct"] for r in all_results]
    worst_dds = [r["worst_drawdown_pct"] for r in all_results]
    total_trades_list = [r["total_trades"] for r in all_results]
    sharpe_list = [r["avg_sharpe"] for r in all_results]

    percentiles = [5, 25, 50, 75, 95]
    ret_pcts    = {p: _percentile(avg_returns, p) for p in percentiles}
    dd_pcts     = {p: _percentile(worst_dds, p) for p in percentiles}
    trade_pcts  = {p: _percentile(total_trades_list, p) for p in percentiles}
    sharpe_pcts = {p: _percentile(sharpe_list, p) for p in percentiles}

    # Per-year percentiles
    per_year_pcts = {}
    for y in years:
        y_returns = [r["per_year"][str(y)].get("return_pct", 0.0) for r in all_results]
        per_year_pcts[y] = {p: _percentile(y_returns, p) for p in percentiles}

    worst_dd_any = min(worst_dds)
    seeds_positive = sum(1 for r in avg_returns if r > 0)
    seeds_dd_gt20 = sum(1 for d in worst_dds if d < -20)
    median_return = ret_pcts[50]

    # Sharpe confidence interval (bootstrap mean ± 1.96 × std / sqrt(n))
    import math
    _n = len(sharpe_list)
    _sharpe_mean = sum(sharpe_list) / _n if _n else 0.0
    _sharpe_std = math.sqrt(sum((s - _sharpe_mean) ** 2 for s in sharpe_list) / max(_n - 1, 1))
    _sharpe_ci_half = 1.96 * _sharpe_std / math.sqrt(max(_n, 1))
    sharpe_ci = (_sharpe_mean - _sharpe_ci_half, _sharpe_mean + _sharpe_ci_half)

    total_elapsed = time.time() - t_start
    _mode_label = (
        f"Mode: FULL (slippage±{args.slippage_jitter:.0%}, entry±{args.entry_day_jitter}d, "
        f"skip={args.trade_skip_pct:.0%}, credit±{args.credit_jitter})"
        if args.mc_mode == "full"
        else f"Mode: DTE-only  U({args.dte_lo}, {args.dte_hi})"
    )
    print(f"""
════════════════════════════════════════════════════════════════════════
  MONTE CARLO RESULTS — {run_id}
  {_mode_label}  |  {args.n_seeds} seeds
────────────────────────────────────────────────────────────────────────
  AVG ANNUAL RETURN DISTRIBUTION:
    P5  (worst realistic):  {ret_pcts[5]:+.1f}%
    P25 (lower quartile):   {ret_pcts[25]:+.1f}%
    P50 (MEDIAN):           {ret_pcts[50]:+.1f}%   ← Real edge
    P75 (upper quartile):   {ret_pcts[75]:+.1f}%
    P95 (best realistic):   {ret_pcts[95]:+.1f}%
────────────────────────────────────────────────────────────────────────
  WORST-YEAR DRAWDOWN DISTRIBUTION:
    P5  (worst 5%):  {dd_pcts[5]:.1f}%
    P25 (lower):     {dd_pcts[25]:.1f}%
    P50 (MEDIAN):    {dd_pcts[50]:.1f}%
    P75 (upper):     {dd_pcts[75]:.1f}%
    P95 (best 5%):   {dd_pcts[95]:.1f}%
  Worst-case DD across ALL {args.n_seeds} seeds:  {worst_dd_any:.1f}%
────────────────────────────────────────────────────────────────────────
  SHARPE RATIO DISTRIBUTION:
    P5={sharpe_pcts[5]:.2f}  P25={sharpe_pcts[25]:.2f}  P50={sharpe_pcts[50]:.2f}  P75={sharpe_pcts[75]:.2f}  P95={sharpe_pcts[95]:.2f}
    95%% CI on mean Sharpe: [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]
────────────────────────────────────────────────────────────────────────
  PER-YEAR MEDIAN RETURNS (P50):""")

    for y in years:
        p = per_year_pcts[y]
        print(f"    {y}:  P5={p[5]:+.1f}%  P25={p[25]:+.1f}%  P50={p[50]:+.1f}%  P75={p[75]:+.1f}%  P95={p[95]:+.1f}%")

    print(f"""────────────────────────────────────────────────────────────────────
  % simulations profitable:     {100*seeds_positive/args.n_seeds:.1f}% ({seeds_positive}/{args.n_seeds})
  % simulations DD > 20%%:       {100*seeds_dd_gt20/args.n_seeds:.1f}% ({seeds_dd_gt20}/{args.n_seeds})
  Elapsed: {total_elapsed/60:.1f} min
════════════════════════════════════════════════════════════════════════

  {'✅ EDGE CONFIRMED' if median_return > 20 else '⚠️  EDGE QUESTIONABLE' if median_return > 0 else '❌ NO EDGE'}: Median return = {median_return:+.1f}%
  {'✅ DRAWDOWN OK' if dd_pcts[50] > -50 else '❌ DRAWDOWN FAILS'}: Median worst-DD = {dd_pcts[50]:.1f}%
  Sharpe 95%% CI: [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]
""")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "run_id": run_id,
        "config": args.config,
        "mc_mode": args.mc_mode,
        "n_seeds": args.n_seeds,
        "dte_lo": args.dte_lo,
        "dte_hi": args.dte_hi,
        "years": years,
        "timestamp": datetime.now().isoformat(),
        "full_mode_params": {
            "slippage_jitter_pct": args.slippage_jitter,
            "entry_day_jitter": args.entry_day_jitter,
            "trade_skip_pct": args.trade_skip_pct,
            "credit_jitter": args.credit_jitter,
        } if args.mc_mode == "full" else {},
        "summary": {
            "avg_return_pct": {f"P{p}": ret_pcts[p] for p in percentiles},
            "worst_drawdown_pct": {f"P{p}": dd_pcts[p] for p in percentiles},
            "total_trades": {f"P{p}": trade_pcts[p] for p in percentiles},
            "sharpe_ratio": {f"P{p}": sharpe_pcts[p] for p in percentiles},
            "worst_dd_any_seed": worst_dd_any,
            "pct_simulations_profitable": round(100 * seeds_positive / args.n_seeds, 1),
            "pct_simulations_dd_gt20": round(100 * seeds_dd_gt20 / args.n_seeds, 1),
            "sharpe_ci_95": [round(sharpe_ci[0], 3), round(sharpe_ci[1], 3)],
            "per_year_p50": {str(y): per_year_pcts[y][50] for y in years},
        },
        "all_seeds": all_results,
    }

    out_path = OUTPUT / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
