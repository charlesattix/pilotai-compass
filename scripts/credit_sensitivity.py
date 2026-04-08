#!/usr/bin/env python3
"""
credit_sensitivity.py — How fragile are backtest returns to the credit assumption?

Tests BACKTEST_CREDIT_FRACTION (fraction of spread width earned as credit) at:
  0.20, 0.25, 0.30, 0.35, 0.40, 0.45

Uses champion params (exp_213_champion_maxc100.json) in HEURISTIC mode
(no Polygon API) across 2020-2025, which is intentional — this is a parametric
sensitivity sweep, not a replay of real prices.

Outputs:
  - ASCII table to stdout
  - output/credit_sensitivity.json
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import shared.constants as _constants

CHAMPION_CONFIG = ROOT / "configs" / "exp_213_champion_maxc100.json"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
CREDIT_FRACTIONS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def _build_config(params: dict, starting_capital: float = 100_000) -> dict:
    return {
        "strategy": {
            "target_delta":        params.get("target_delta", 0.12),
            "use_delta_selection": params.get("use_delta_selection", False),
            "target_dte":          params.get("target_dte", 35),
            "min_dte":             params.get("min_dte", 25),
            "spread_width":        params.get("spread_width", 5),
            "min_credit_pct":      params.get("min_credit_pct", 8),
            "direction":           params.get("direction", "both"),
            "trend_ma_period":     params.get("trend_ma_period", 200),
            "regime_mode":         params.get("regime_mode", "combo"),
            "regime_config":       params.get("regime_config", {}),
            "momentum_filter_pct": params.get("momentum_filter_pct", None),
            "iron_condor": {
                "enabled":                 params.get("iron_condor_enabled", False),
                "min_combined_credit_pct": params.get("ic_min_combined_credit_pct", 28),
                "neutral_regime_only":     False,
                "vix_min":                 params.get("ic_vix_min", 12),
                "risk_per_trade":          params.get("ic_risk_per_trade", None),
            },
            "iv_rank_min_entry": params.get("iv_rank_min_entry", 0),
        },
        "risk": {
            "stop_loss_multiplier": params.get("stop_loss_multiplier", 2.5),
            "profit_target":        params.get("profit_target", 50),
            "max_risk_per_trade":   params.get("max_risk_per_trade", 23.0),
            "max_contracts":        params.get("max_contracts", 100),
            "max_positions":        50,
            "drawdown_cb_pct":      params.get("drawdown_cb_pct", 55),
        },
        "backtest": {
            "starting_capital":        starting_capital,
            "commission_per_contract": 0.65,
            "slippage":                0.05,
            "exit_slippage":           0.10,
            "compound":                params.get("compound", True),
            "sizing_mode":             params.get("sizing_mode", "flat"),
            "slippage_multiplier":     1.0,
            "max_portfolio_exposure_pct": params.get("max_portfolio_exposure_pct", 100.0),
        },
    }


def run_year(params: dict, year: int) -> dict:
    from backtest.backtester import Backtester
    config = _build_config(params)
    bt = Backtester(config, historical_data=None, otm_pct=params.get("otm_pct", 0.03))
    result = bt.run_backtest("SPY", datetime(year, 1, 1), datetime(year, 12, 31)) or {}
    result["year"] = year
    return result


def run_sweep(fraction: float, params: dict) -> dict:
    """Run all years with BACKTEST_CREDIT_FRACTION patched to `fraction`."""
    _constants.BACKTEST_CREDIT_FRACTION = fraction

    year_results = {}
    for year in YEARS:
        print(f"    {year}...", end=" ", flush=True)
        r = run_year(params, year)
        ret = r.get("return_pct", 0)
        trades = r.get("total_trades", 0)
        print(f"{ret:+.1f}% ({trades}t)", end="  ", flush=True)
        year_results[year] = r
    print()

    returns = [year_results[y].get("return_pct", 0) for y in YEARS]
    drawdowns = [year_results[y].get("max_drawdown", 0) for y in YEARS]
    win_rates = [year_results[y].get("win_rate", 0) for y in YEARS]
    trades_list = [year_results[y].get("total_trades", 0) for y in YEARS]
    sharpes = [year_results[y].get("sharpe_ratio", 0) for y in YEARS]

    avg_return = sum(returns) / len(returns)
    worst_dd = min(drawdowns)
    avg_wr = sum(win_rates) / len(win_rates) if any(win_rates) else 0
    total_trades = sum(trades_list)
    avg_sharpe = sum(sharpes) / len(sharpes) if any(sharpes) else 0

    return {
        "credit_fraction": fraction,
        "avg_annual_return_pct": round(avg_return, 1),
        "worst_drawdown_pct": round(worst_dd, 1),
        "avg_win_rate_pct": round(avg_wr, 1),
        "total_trades": total_trades,
        "avg_sharpe": round(avg_sharpe, 2),
        "profitable_years": sum(1 for r in returns if r > 0),
        "by_year": {
            str(y): {
                "return_pct": round(year_results[y].get("return_pct", 0), 1),
                "max_drawdown": round(year_results[y].get("max_drawdown", 0), 1),
                "win_rate": round(year_results[y].get("win_rate", 0), 1),
                "total_trades": year_results[y].get("total_trades", 0),
                "sharpe_ratio": round(year_results[y].get("sharpe_ratio", 0), 2),
            }
            for y in YEARS
        },
    }


def find_breakeven(rows: list) -> Optional[float]:
    """Find the credit fraction where avg_annual_return crosses zero."""
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if a["avg_annual_return_pct"] > 0 >= b["avg_annual_return_pct"]:
            # Linear interpolation
            span = b["credit_fraction"] - a["credit_fraction"]
            frac = a["avg_annual_return_pct"] / (a["avg_annual_return_pct"] - b["avg_annual_return_pct"])
            return round(a["credit_fraction"] + frac * span, 3)
        if a["avg_annual_return_pct"] < 0 <= b["avg_annual_return_pct"]:
            span = b["credit_fraction"] - a["credit_fraction"]
            frac = abs(a["avg_annual_return_pct"]) / (abs(a["avg_annual_return_pct"]) + abs(b["avg_annual_return_pct"]))
            return round(a["credit_fraction"] + frac * span, 3)
    # Check if all positive or all negative
    if all(r["avg_annual_return_pct"] > 0 for r in rows):
        return None  # Always profitable in this range
    if all(r["avg_annual_return_pct"] <= 0 for r in rows):
        return None  # Always negative in this range
    return None


def print_table(rows: list, baseline_fraction: float = 0.35):
    header = f"{'Credit%':>9}  {'Avg Ret':>8}  {'Worst DD':>9}  {'Win Rate':>9}  {'Trades':>7}  {'Sharpe':>7}  {'ProfYrs':>7}  {'Note':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        frac = row["credit_fraction"]
        pct = f"{frac*100:.0f}%"
        note = " << BASELINE" if abs(frac - baseline_fraction) < 0.001 else ""
        neg = row["avg_annual_return_pct"] < 0
        prefix = "*** " if neg else "    "
        print(
            f"{prefix}{pct:>7}  "
            f"{row['avg_annual_return_pct']:>+7.1f}%  "
            f"{row['worst_drawdown_pct']:>+8.1f}%  "
            f"{row['avg_win_rate_pct']:>8.1f}%  "
            f"{row['total_trades']:>7}  "
            f"{row['avg_sharpe']:>7.2f}  "
            f"{row['profitable_years']:>5}/6    "
            f"{note}"
        )
    print(sep)


def main():
    params = json.loads(CHAMPION_CONFIG.read_text())
    original_fraction = _constants.BACKTEST_CREDIT_FRACTION

    print("=" * 70)
    print("CREDIT SENSITIVITY ANALYSIS — Champion Config (Heuristic Mode)")
    print("=" * 70)
    print(f"Baseline BACKTEST_CREDIT_FRACTION = {original_fraction}")
    print(f"Spread width = {params.get('spread_width', 5)}, Years: {YEARS}")
    print()

    rows = []
    for frac in CREDIT_FRACTIONS:
        print(f"  Credit fraction {frac:.0%}:")
        row = run_sweep(frac, params)
        rows.append(row)

    # Restore original
    _constants.BACKTEST_CREDIT_FRACTION = original_fraction

    print()
    print_table(rows, baseline_fraction=original_fraction)

    breakeven = find_breakeven(rows)
    print()
    if breakeven is not None:
        print(f"  Break-even credit fraction: {breakeven:.0%} of spread width")
        margin = original_fraction - breakeven
        print(f"  Safety margin above break-even: {margin:.0%} ({margin/original_fraction*100:.0f}% of baseline)")
    else:
        all_pos = all(r["avg_annual_return_pct"] > 0 for r in rows)
        if all_pos:
            print("  Strategy is profitable across ALL tested credit fractions (no break-even found)")
        else:
            print("  Strategy is UNPROFITABLE across ALL tested credit fractions")

    # Save JSON
    out = {
        "generated_at": datetime.utcnow().isoformat(),
        "description": "Sensitivity of backtest returns to BACKTEST_CREDIT_FRACTION",
        "baseline_credit_fraction": original_fraction,
        "spread_width": params.get("spread_width", 5),
        "years": YEARS,
        "mode": "heuristic",
        "breakeven_credit_fraction": breakeven,
        "rows": rows,
    }
    out_path = OUTPUT_DIR / "credit_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")
    return out


if __name__ == "__main__":
    main()
