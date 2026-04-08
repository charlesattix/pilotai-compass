#!/usr/bin/env python3
"""
exit_slippage_analysis.py — How fragile are backtest returns to exit slippage?

Tests exit_slippage at 0%, 2%, 5%, 10%, 15%, 20% of spread width.
Champion spread_width=5, so absolute values: 0.00, 0.10, 0.25, 0.50, 0.75, 1.00

Uses champion params (exp_213_champion_maxc100.json) in HEURISTIC mode
across 2020-2025.

Outputs:
  - ASCII table to stdout
  - output/exit_slippage_sensitivity.json
"""

import json
import sys
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

CHAMPION_CONFIG = ROOT / "configs" / "exp_213_champion_maxc100.json"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
# Exit slippage as fraction of spread width
EXIT_SLIPPAGE_FRACTIONS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
SPREAD_WIDTH = 5  # Champion spread width


def _build_config(params: dict, exit_slippage: float, starting_capital: float = 100_000) -> dict:
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
            "exit_slippage":           exit_slippage,
            "compound":                params.get("compound", True),
            "sizing_mode":             params.get("sizing_mode", "flat"),
            "slippage_multiplier":     1.0,
            "max_portfolio_exposure_pct": params.get("max_portfolio_exposure_pct", 100.0),
        },
    }


def run_year(params: dict, year: int, exit_slippage: float) -> dict:
    from backtest.backtester import Backtester
    config = _build_config(params, exit_slippage=exit_slippage)
    bt = Backtester(config, historical_data=None, otm_pct=params.get("otm_pct", 0.03))
    result = bt.run_backtest("SPY", datetime(year, 1, 1), datetime(year, 12, 31)) or {}
    result["year"] = year
    return result


def run_sweep(slip_frac: float, params: dict) -> dict:
    exit_slippage_abs = round(slip_frac * SPREAD_WIDTH, 4)
    year_results = {}
    for year in YEARS:
        print(f"    {year}...", end=" ", flush=True)
        r = run_year(params, year, exit_slippage_abs)
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
        "exit_slippage_fraction": slip_frac,
        "exit_slippage_abs": exit_slippage_abs,
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
    """Find exit slippage fraction where avg_annual_return crosses zero."""
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if a["avg_annual_return_pct"] > 0 >= b["avg_annual_return_pct"]:
            span = b["exit_slippage_fraction"] - a["exit_slippage_fraction"]
            frac = a["avg_annual_return_pct"] / (a["avg_annual_return_pct"] - b["avg_annual_return_pct"])
            return round(a["exit_slippage_fraction"] + frac * span, 3)
        if a["avg_annual_return_pct"] < 0 <= b["avg_annual_return_pct"]:
            span = b["exit_slippage_fraction"] - a["exit_slippage_fraction"]
            frac = abs(a["avg_annual_return_pct"]) / (abs(a["avg_annual_return_pct"]) + abs(b["avg_annual_return_pct"]))
            return round(a["exit_slippage_fraction"] + frac * span, 3)
    if all(r["avg_annual_return_pct"] > 0 for r in rows):
        return None  # Always profitable
    if all(r["avg_annual_return_pct"] <= 0 for r in rows):
        return None  # Always negative
    return None


def print_table(rows: list, baseline_frac: float = 0.02):
    header = (
        f"{'Slip%':>7}  {'Slip$':>6}  {'Avg Ret':>8}  "
        f"{'Worst DD':>9}  {'Win Rate':>9}  {'Trades':>7}  {'Sharpe':>7}  {'ProfYrs':>7}  {'Note':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        frac = row["exit_slippage_fraction"]
        abs_val = row["exit_slippage_abs"]
        pct = f"{frac*100:.0f}%"
        note = " << BASELINE" if abs(frac - baseline_frac) < 0.001 else ""
        neg = row["avg_annual_return_pct"] < 0
        prefix = "*** " if neg else "    "
        print(
            f"{prefix}{pct:>5}  "
            f"{abs_val:>5.2f}  "
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
    baseline_slip = 0.10 / SPREAD_WIDTH  # 0.10 abs / 5 width = 2%

    print("=" * 70)
    print("EXIT SLIPPAGE SENSITIVITY — Champion Config (Heuristic Mode)")
    print("=" * 70)
    print(f"Baseline exit_slippage = $0.10 ({baseline_slip:.0%} of spread_width={SPREAD_WIDTH})")
    print(f"Years: {YEARS}")
    print()

    rows = []
    for frac in EXIT_SLIPPAGE_FRACTIONS:
        abs_val = round(frac * SPREAD_WIDTH, 4)
        print(f"  Exit slippage {frac:.0%} of width (${abs_val:.2f} abs):")
        row = run_sweep(frac, params)
        rows.append(row)

    print()
    print_table(rows, baseline_frac=baseline_slip)

    breakeven = find_breakeven(rows)
    print()
    if breakeven is not None:
        be_abs = round(breakeven * SPREAD_WIDTH, 2)
        print(f"  Break-even exit slippage: {breakeven:.0%} of width (${be_abs:.2f} abs)")
        margin = breakeven - baseline_slip
        print(f"  Safety margin above baseline: +{margin:.0%} ({margin/baseline_slip*100:.0f}% headroom)")
    else:
        all_pos = all(r["avg_annual_return_pct"] > 0 for r in rows)
        if all_pos:
            print("  Strategy is profitable across ALL tested exit slippage levels (no break-even found)")
        else:
            print("  Strategy is UNPROFITABLE across ALL tested exit slippage levels")

    out = {
        "generated_at": datetime.utcnow().isoformat(),
        "description": "Sensitivity of backtest returns to exit slippage",
        "baseline_exit_slippage_abs": 0.10,
        "baseline_exit_slippage_fraction": round(baseline_slip, 3),
        "spread_width": SPREAD_WIDTH,
        "years": YEARS,
        "mode": "heuristic",
        "breakeven_exit_slippage_fraction": breakeven,
        "breakeven_exit_slippage_abs": round(breakeven * SPREAD_WIDTH, 2) if breakeven is not None else None,
        "rows": rows,
    }
    out_path = OUTPUT_DIR / "exit_slippage_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")
    return out


if __name__ == "__main__":
    main()
