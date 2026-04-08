#!/usr/bin/env python3
"""
test_position_sizing.py — Test position sizing variants on the champion config.

Runs credit_spread (regime_adaptive) through 2020-2025 at different risk_pct levels:
  - Fixed fractional: 2%, 5%, 8.5% (current), 10%, 15%, 20%
  - Kelly criterion: computed from baseline win_rate & avg_win/avg_loss

Output: output/position_sizing_results.json

Usage:
    PYTHONPATH=. python3 scripts/test_position_sizing.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.portfolio_backtester import PortfolioBacktester
from strategies import STRATEGY_REGISTRY

TICKERS = ["SPY"]
STARTING_CAPITAL = 100_000
YEARS = list(range(2020, 2026))
CHAMPION_PATH = ROOT / "configs" / "champion.json"

FIXED_RISK_PCTS = [0.02, 0.05, 0.085, 0.10, 0.15, 0.20]


def load_champion_cs_params() -> Dict:
    """Load credit_spread params from champion config."""
    with open(CHAMPION_PATH) as f:
        return dict(json.load(f)["strategy_params"]["credit_spread"])


def run_year(params: Dict, year: int) -> Dict:
    """Run credit_spread for one year, return metrics."""
    cls = STRATEGY_REGISTRY["credit_spread"]
    strategy = cls(params)

    bt = PortfolioBacktester(
        strategies=[("credit_spread", strategy)],
        tickers=TICKERS,
        start_date=datetime(year, 1, 1),
        end_date=datetime(year, 12, 31),
        starting_capital=STARTING_CAPITAL,
        max_positions=10,
        max_positions_per_strategy=5,
    )
    raw = bt.run()
    combined = raw.get("combined", raw)

    # Compute avg_win and avg_loss from closed trades
    wins = [t.realized_pnl for t in bt.closed_trades if t.realized_pnl > 0]
    losses = [t.realized_pnl for t in bt.closed_trades if t.realized_pnl <= 0]

    return {
        "return_pct": combined.get("return_pct", 0),
        "max_drawdown": combined.get("max_drawdown", 0),
        "total_trades": combined.get("total_trades", 0),
        "win_rate": combined.get("win_rate", 0),
        "sharpe_ratio": combined.get("sharpe_ratio", 0),
        "avg_win": sum(wins) / len(wins) if wins else 0,
        "avg_loss": abs(sum(losses) / len(losses)) if losses else 0,
        "wins": len(wins),
        "losses": len(losses),
    }


def run_all_years(params: Dict, label: str) -> Dict:
    """Run across all years, compute summary."""
    print(f"\n  {label} (risk_pct={params['max_risk_pct']:.1%})")
    yearly = {}
    for year in YEARS:
        r = run_year(params, year)
        yearly[str(year)] = r
        print(f"    {year}: ret={r['return_pct']:+.1f}%  DD={r['max_drawdown']:.1f}%  "
              f"trades={r['total_trades']}  WR={r['win_rate']:.1f}%")

    rets = [yearly[str(y)]["return_pct"] for y in YEARS]
    dds = [yearly[str(y)]["max_drawdown"] for y in YEARS]
    sharpes = [yearly[str(y)]["sharpe_ratio"] for y in YEARS]

    avg_ret = sum(rets) / len(rets)
    worst_dd = min(dds)
    avg_sharpe = sum(sharpes) / len(sharpes)

    # Aggregate win/loss stats across all years for Kelly
    total_wins = sum(yearly[str(y)]["wins"] for y in YEARS)
    total_losses = sum(yearly[str(y)]["losses"] for y in YEARS)
    total_trades = total_wins + total_losses
    agg_win_rate = total_wins / total_trades if total_trades else 0
    all_avg_win = sum(yearly[str(y)]["avg_win"] * yearly[str(y)]["wins"] for y in YEARS)
    all_avg_loss = sum(yearly[str(y)]["avg_loss"] * yearly[str(y)]["losses"] for y in YEARS)
    agg_avg_win = all_avg_win / total_wins if total_wins else 0
    agg_avg_loss = all_avg_loss / total_losses if total_losses else 0

    return {
        "label": label,
        "sizing_method": "fixed_fractional",
        "risk_pct": params["max_risk_pct"],
        "avg_annual_return": round(avg_ret, 2),
        "max_drawdown": round(worst_dd, 2),
        "sharpe": round(avg_sharpe, 2),
        "yearly": yearly,
        "agg_win_rate": round(agg_win_rate, 4),
        "agg_avg_win": round(agg_avg_win, 2),
        "agg_avg_loss": round(agg_avg_loss, 2),
    }


def compute_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly criterion: f* = W - (1-W)/R where W=win_rate, R=avg_win/avg_loss."""
    if avg_loss <= 0:
        return 0
    R = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / R
    # Cap at 25% and floor at 1%
    return max(0.01, min(0.25, kelly))


def main():
    print("=" * 70)
    print("POSITION SIZING TEST — Champion Credit Spread (Regime Adaptive)")
    print("=" * 70)

    base_params = load_champion_cs_params()
    t0 = time.time()

    # ── Step 1: Fixed fractional runs ────────────────────────────────────
    print("\n[1/2] Testing fixed fractional sizing...")
    results = []
    baseline_result = None

    for risk_pct in FIXED_RISK_PCTS:
        params = dict(base_params)
        params["max_risk_pct"] = risk_pct
        label = f"fixed_{risk_pct:.1%}"
        r = run_all_years(params, label)
        results.append(r)
        if risk_pct == 0.085:
            baseline_result = r

    # ── Step 2: Kelly criterion ──────────────────────────────────────────
    print("\n[2/2] Computing Kelly criterion from baseline stats...")
    if baseline_result:
        kelly_f = compute_kelly(
            baseline_result["agg_win_rate"],
            baseline_result["agg_avg_win"],
            baseline_result["agg_avg_loss"],
        )
        print(f"  Win rate: {baseline_result['agg_win_rate']:.1%}")
        print(f"  Avg win:  ${baseline_result['agg_avg_win']:.0f}")
        print(f"  Avg loss: ${baseline_result['agg_avg_loss']:.0f}")
        print(f"  Kelly f*: {kelly_f:.1%}")

        # Run Kelly-sized backtest
        params = dict(base_params)
        params["max_risk_pct"] = kelly_f
        kelly_result = run_all_years(params, f"kelly_{kelly_f:.1%}")
        kelly_result["sizing_method"] = "kelly"
        kelly_result["kelly_inputs"] = {
            "win_rate": baseline_result["agg_win_rate"],
            "avg_win": baseline_result["agg_avg_win"],
            "avg_loss": baseline_result["agg_avg_loss"],
            "raw_kelly_f": round(kelly_f, 4),
        }
        results.append(kelly_result)

    elapsed = time.time() - t0

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'=' * 85}")
    print("POSITION SIZING COMPARISON")
    print(f"{'=' * 85}")
    print(f"\n{'Method':<20}  {'Risk %':>7}  {'Avg Return':>10}  {'Max DD':>8}  {'Sharpe':>7}")
    print("-" * 85)
    for r in results:
        method = r["sizing_method"]
        if method == "kelly":
            method = f"kelly (f*={r['risk_pct']:.1%})"
        print(f"{method:<20}  {r['risk_pct']:>6.1%}  {r['avg_annual_return']:>+9.1f}%  "
              f"{r['max_drawdown']:>7.1f}%  {r['sharpe']:>6.2f}")

    # ── Save ─────────────────────────────────────────────────────────────
    # Build clean comparison table
    comparison = []
    for r in results:
        comparison.append({
            "sizing_method": r["sizing_method"],
            "risk_pct": r["risk_pct"],
            "avg_annual_return": r["avg_annual_return"],
            "max_drawdown": r["max_drawdown"],
            "sharpe": r["sharpe"],
        })

    output = {
        "generated": datetime.now().isoformat(),
        "description": "Position sizing variants on champion credit_spread (regime_adaptive)",
        "base_config": "configs/champion.json → credit_spread",
        "tickers": TICKERS,
        "starting_capital": STARTING_CAPITAL,
        "years": YEARS,
        "comparison": comparison,
        "detailed_results": results,
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = ROOT / "output" / "position_sizing_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    print(f"Completed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
