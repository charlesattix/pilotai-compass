#!/usr/bin/env python3
"""
champion_monthly_report.py — Per-month and per-year P&L for Gate 2 champion configs.

Champion A: DTE=10, OTM=5%, width=3, PT=65%, SL=1.5x, risk=20%, compound, direction_adaptive
Champion B: DTE=14, OTM=3%, width=3, PT=65%, SL=1.5x, risk=20%, compound, direction_adaptive

Outputs:
  output/champion_a_monthly.json
  output/champion_b_monthly.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# ─── Dynamic import of the backtester from diagnose_ibit_overfit.py ───────────
spec = importlib.util.spec_from_file_location(
    "diagnose_ibit_overfit",
    ROOT / "scripts" / "diagnose_ibit_overfit.py",
)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

IBITBacktester = diag.IBITBacktester
get_conn       = diag.get_conn
get_all_spots  = diag.get_all_spots
STARTING_CAPITAL = diag.STARTING_CAPITAL

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FULL_START = "2024-11-01"
FULL_END   = "2026-03-20"


# ─── Extended backtester that captures full trade detail ─────────────────────

class DetailedBacktester(IBITBacktester):
    """Adds per-trade strike/credit detail and running portfolio value."""

    def _record_close(self, pos: dict, exit_date: str, pnl_usd: float, reason: str):
        """Override to capture extra fields from position dict."""
        self.capital += pnl_usd
        pnl_pct = pnl_usd / (pos["total_max_loss"] * pos["n_contracts"] * diag.MULTIPLIER) * 100.0 \
                  if pos["total_max_loss"] > 0 else 0.0

        self.trades.append({
            "entry_date":            pos["entry_date"],
            "exit_date":             exit_date,
            "expiry_date":           pos["expiry"],
            "dte_at_entry":          pos["dte_at_entry"],
            "direction":             pos["direction"],
            # bull-put leg
            "bp_short_strike":       pos.get("bp_short_strike"),
            "bp_long_strike":        pos.get("bp_long_strike"),
            "bp_credit":             round(pos.get("bp_credit", 0.0), 4),
            # bear-call leg
            "bc_short_strike":       pos.get("bc_short_strike"),
            "bc_long_strike":        pos.get("bc_long_strike"),
            "bc_credit":             round(pos.get("bc_credit", 0.0), 4),
            # combined
            "total_credit":          round(pos["total_credit"], 4),
            "total_max_loss":        round(pos["total_max_loss"], 4),
            "credit_pct_of_width":   round(pos["total_credit"] / pos["total_max_loss"] * 100.0, 2)
                                     if pos["total_max_loss"] > 0 else 0.0,
            "n_contracts":           pos["n_contracts"],
            "spot_at_entry":         round(pos["spot_at_entry"], 4),
            "pnl_usd":               round(pnl_usd, 2),   # keep for parent _build_results
            "pnl_dollar":            round(pnl_usd, 2),
            "pnl_pct_of_risk":       round(pnl_pct, 2),
            "exit_reason":           reason,
            "win":                   pnl_usd > 0,
            "portfolio_value_after": round(self.capital, 2),
        })
        pos["status"] = "closed"


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def _month_key(date_str: str) -> str:
    return date_str[:7]  # "YYYY-MM"

def _year_key(date_str: str) -> str:
    return date_str[:4]  # "YYYY"


def build_monthly(trades: List[Dict], equity_curve: List[Dict]) -> List[Dict]:
    """Bucket closed trades by exit month; compute return% from equity curve."""
    # Build equity by month-end from equity curve
    month_end_equity: Dict[str, float] = {}
    month_start_equity: Dict[str, float] = {}
    prev_month = None
    prev_equity = STARTING_CAPITAL

    for ec in equity_curve:
        m = _month_key(ec["date"])
        if m != prev_month:
            month_start_equity[m] = prev_equity
            prev_month = m
        month_end_equity[m] = ec["equity"]
        prev_equity = ec["equity"]

    # Bucket trades
    month_trades: Dict[str, List[Dict]] = defaultdict(list)
    for t in trades:
        month_trades[_month_key(t["exit_date"])].append(t)

    all_months = sorted(set(list(month_end_equity.keys())))
    rows = []
    cum_return = 0.0

    for m in all_months:
        start_eq = month_start_equity.get(m, STARTING_CAPITAL)
        end_eq   = month_end_equity.get(m, start_eq)
        mret     = (end_eq - start_eq) / start_eq * 100.0 if start_eq > 0 else 0.0
        cum_return = (end_eq - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0

        ts = month_trades.get(m, [])
        wins   = [t for t in ts if t["win"]]
        losses = [t for t in ts if not t["win"]]

        rows.append({
            "month":               m,
            "trades":              len(ts),
            "wins":                len(wins),
            "losses":              len(losses),
            "start_equity":        round(start_eq, 2),
            "end_equity":          round(end_eq, 2),
            "monthly_return_pct":  round(mret, 2),
            "cumulative_return_pct": round(cum_return, 2),
        })

    return rows


def build_yearly(trades: List[Dict], monthly: List[Dict]) -> List[Dict]:
    year_months: Dict[str, List[Dict]] = defaultdict(list)
    for m in monthly:
        year_months[m["month"][:4]].append(m)

    year_trades: Dict[str, List[Dict]] = defaultdict(list)
    for t in trades:
        year_trades[_year_key(t["exit_date"])].append(t)

    rows = []
    for yr in sorted(year_months.keys()):
        ms = year_months[yr]
        ts = year_trades[yr]
        wins   = [t for t in ts if t["win"]]
        losses = [t for t in ts if not t["win"]]

        start_eq = ms[0]["start_equity"]
        end_eq   = ms[-1]["end_equity"]
        ann_ret  = (end_eq - start_eq) / start_eq * 100.0 if start_eq > 0 else 0.0

        worst_month = min(ms, key=lambda x: x["monthly_return_pct"])["monthly_return_pct"]

        rows.append({
            "year":              yr,
            "months_active":     len(ms),
            "trades":            len(ts),
            "wins":              len(wins),
            "losses":            len(losses),
            "win_rate_pct":      round(len(wins) / len(ts) * 100.0, 1) if ts else 0.0,
            "annual_return_pct": round(ann_ret, 2),
            "worst_month_pct":   round(worst_month, 2),
            "start_equity":      round(start_eq, 2),
            "end_equity":        round(end_eq, 2),
        })

    return rows


def build_summary(result: Dict, champion_name: str, cfg: Dict) -> Dict:
    trades = result["trades"]
    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    bull_trades = [t for t in trades if t["direction"] == "bull_put"]
    bear_trades = [t for t in trades if t["direction"] == "bear_call"]

    return {
        "champion":           champion_name,
        "config":             cfg,
        "start_date":         FULL_START,
        "end_date":           FULL_END,
        "starting_capital":   STARTING_CAPITAL,
        "ending_capital":     round(result["ending_capital"], 2),
        "total_return_pct":   round(result["return_pct"], 2),
        "annualized_return_pct": round(result["ann_return"], 2),
        "max_drawdown_pct":   round(result["max_drawdown"], 2),
        "total_trades":       len(trades),
        "wins":               len(wins),
        "losses":             len(losses),
        "win_rate_pct":       round(result["win_rate"], 2),
        "profit_factor":      round(result["profit_factor"], 4),
        "avg_win_dollar":     round(result["avg_win"], 2),
        "avg_loss_dollar":    round(result["avg_loss"], 2),
        "bull_put_trades":    len(bull_trades),
        "bear_call_trades":   len(bear_trades),
    }


def print_report(champion_name: str, summary: Dict, monthly: List[Dict], yearly: List[Dict]):
    print(f"\n{'='*70}")
    print(f"  {champion_name}")
    cfg = summary["config"]
    print(f"  DTE={cfg['dte']} | OTM={cfg['otm_pct']*100:.0f}% | W=${cfg['spread_width']} | "
          f"PT={cfg['profit_target']*100:.0f}% | SL={cfg['stop_loss_mult']}x | "
          f"Risk={cfg['risk_pct']*100:.0f}% | Adaptive={'YES' if cfg.get('direction_adaptive') else 'NO'}")
    print(f"{'='*70}")

    print(f"\n  SUMMARY")
    print(f"  Total return:    {summary['total_return_pct']:+.1f}%")
    print(f"  Annualized:      {summary['annualized_return_pct']:+.1f}%")
    print(f"  Max drawdown:    {summary['max_drawdown_pct']:.1f}%")
    print(f"  Trades:          {summary['total_trades']} "
          f"({summary['bull_put_trades']} bull-put / {summary['bear_call_trades']} bear-call)")
    print(f"  Win rate:        {summary['win_rate_pct']:.1f}%")
    print(f"  Profit factor:   {summary['profit_factor']:.2f}")
    print(f"  Avg win:         ${summary['avg_win_dollar']:,.0f} | Avg loss: ${summary['avg_loss_dollar']:,.0f}")

    print(f"\n  MONTHLY BREAKDOWN")
    print(f"  {'Month':<9} {'Trades':>6} {'W':>4} {'L':>4} {'Mth%':>8} {'Cum%':>9}  Dir")
    for m in monthly:
        bar = "+" * min(int(abs(m["monthly_return_pct"]) / 2), 20)
        sign = "▲" if m["monthly_return_pct"] >= 0 else "▼"
        print(f"  {m['month']:<9} {m['trades']:>6} {m['wins']:>4} {m['losses']:>4} "
              f"{m['monthly_return_pct']:>+7.1f}% {m['cumulative_return_pct']:>+8.1f}%  {sign}{bar}")

    print(f"\n  YEARLY SUMMARY")
    print(f"  {'Year':<6} {'Mo':>4} {'Trd':>5} {'W':>4} {'L':>4} {'WR%':>6} {'Ann%':>9} {'WorstMo':>9}")
    for y in yearly:
        print(f"  {y['year']:<6} {y['months_active']:>4} {y['trades']:>5} "
              f"{y['wins']:>4} {y['losses']:>4} {y['win_rate_pct']:>5.1f}% "
              f"{y['annual_return_pct']:>+8.1f}% {y['worst_month_pct']:>+8.1f}%")

    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

CHAMPIONS = [
    {
        "name": "Champion A",
        "output_file": "champion_a_monthly.json",
        "config": {
            "dte":               10,
            "otm_pct":           0.05,
            "spread_width":      3.0,
            "profit_target":     0.65,
            "stop_loss_mult":    1.5,
            "risk_pct":          0.20,
            "compound":          True,
            "direction_adaptive": True,
            "iron_condor":       False,
            "regime_filter":     "none",
            "min_credit_pct":    3.0,
            "max_contracts":     100,
        },
    },
    {
        "name": "Champion B",
        "output_file": "champion_b_monthly.json",
        "config": {
            "dte":               14,
            "otm_pct":           0.03,
            "spread_width":      3.0,
            "profit_target":     0.65,
            "stop_loss_mult":    1.5,
            "risk_pct":          0.20,
            "compound":          True,
            "direction_adaptive": True,
            "iron_condor":       False,
            "regime_filter":     "none",
            "min_credit_pct":    3.0,
            "max_contracts":     100,
        },
    },
]


def main():
    conn      = get_conn()
    all_spots = get_all_spots(conn)
    print(f"Loaded {len(all_spots)} IBIT spot prices ({min(all_spots)} → {max(all_spots)})")

    for champ in CHAMPIONS:
        name = champ["name"]
        cfg  = champ["config"]
        out_path = OUTPUT_DIR / champ["output_file"]

        print(f"\nRunning {name} ...")
        bt = DetailedBacktester(cfg, conn, all_spots)
        result = bt.run(FULL_START, FULL_END)

        eq_curve = result.get("equity_curve", [])
        trades   = result.get("trades", [])

        monthly = build_monthly(trades, eq_curve)
        yearly  = build_yearly(trades, monthly)
        summary = build_summary(result, name, cfg)

        output = {
            "summary": summary,
            "yearly":  yearly,
            "monthly": monthly,
            "trades":  trades,
        }

        out_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"  Saved → {out_path}")

        print_report(name, summary, monthly, yearly)


if __name__ == "__main__":
    main()
