#!/usr/bin/env python3
"""Daily P&L comparison report for paper trading validation.

Reads both paper trading DBs (EXP-400 champion + EXP-401), computes key
metrics, compares against backtest expectations, and sends a formatted
Telegram message.

Usage:
    python scripts/daily_pnl_report.py
    python scripts/daily_pnl_report.py --date 2026-03-14
    python scripts/daily_pnl_report.py --dry-run   # print only, no Telegram

Can also be called as a module:
    from scripts.daily_pnl_report import build_report, send_report
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from shared.database import get_trades

# ─── Backtest expectations ───────────────────────────────────────────────────
# These are the "fair value" annual returns from the backtest after slippage.
# Used to contextualise paper trading performance at this early stage.

EXPECTATIONS = {
    "EXP-400": {
        "label": "Champion (exp_213)",
        "annual_return_pct": 32.7,   # conservative 2x-slippage estimate
        "win_rate_pct": 78.0,
        "db_path": str(ROOT / "data" / "pilotai_exp400.db"),
        "env_file": str(ROOT / ".env.champion"),
    },
    "EXP-401": {
        "label": "EXP-401",
        "annual_return_pct": 26.9,   # after additional slippage adjustment
        "win_rate_pct": 75.0,
        "db_path": str(ROOT / "data" / "pilotai_exp401.db"),
        "env_file": str(ROOT / ".env.exp401"),
    },
}

TRADING_DAYS_PER_YEAR = 252


# ─── Metrics calculation ──────────────────────────────────────────────────────

def _is_closed(status: str) -> bool:
    return status and status.startswith("closed")


def _is_win(trade: Dict) -> bool:
    pnl = trade.get("pnl") or 0
    return _is_closed(trade.get("status", "")) and pnl > 0


def compute_metrics(trades: List[Dict], account_size: float = 100_000) -> Dict:
    """Compute P&L and trade stats from a list of trade dicts."""
    closed = [t for t in trades if _is_closed(t.get("status", ""))]
    open_pos = [t for t in trades if t.get("status") == "open"]

    total_trades = len(closed)
    wins = [t for t in closed if _is_win(t)]
    losses = [t for t in closed if not _is_win(t)]

    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

    total_pnl = sum(t.get("pnl") or 0 for t in closed)
    gross_profit = sum(t.get("pnl") or 0 for t in wins)
    gross_loss = abs(sum(t.get("pnl") or 0 for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown from cumulative equity curve
    equity = account_size
    peak = equity
    max_dd = 0.0
    running = account_size
    for t in sorted(closed, key=lambda x: x.get("exit_date") or ""):
        pnl = t.get("pnl") or 0
        running += pnl
        if running > peak:
            peak = running
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Annualised return — based on first/last trade dates
    annualised_return = 0.0
    if closed:
        dates = sorted(
            t.get("exit_date") or t.get("entry_date") or ""
            for t in closed
            if (t.get("exit_date") or t.get("entry_date"))
        )
        if len(dates) >= 2:
            try:
                first = date.fromisoformat(dates[0][:10])
                last = date.fromisoformat(dates[-1][:10])
                days = max((last - first).days, 1)
                total_return_pct = total_pnl / account_size * 100
                annualised_return = total_return_pct * (365 / days)
            except (ValueError, TypeError):
                pass

    return {
        "total_trades": total_trades,
        "open_positions": len(open_pos),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": win_rate,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd,
        "annualised_return_pct": annualised_return,
    }


def load_experiment_metrics(exp_id: str, account_size: float = 100_000) -> Dict:
    """Load trades from the experiment DB and compute metrics."""
    cfg = EXPECTATIONS[exp_id]
    db_path = os.environ.get(f"PILOTAI_DB_PATH_{exp_id.replace('-', '_')}", cfg["db_path"])
    try:
        trades = get_trades(source="execution", path=db_path)
        metrics = compute_metrics(trades, account_size=account_size)
    except Exception as e:
        metrics = {
            "total_trades": 0,
            "open_positions": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "total_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "annualised_return_pct": 0.0,
            "error": str(e),
        }
    metrics["exp_id"] = exp_id
    metrics["label"] = cfg["label"]
    metrics["expected_annual_return_pct"] = cfg["annual_return_pct"]
    metrics["expected_win_rate_pct"] = cfg["win_rate_pct"]
    return metrics


# ─── Telegram message formatter ───────────────────────────────────────────────

def _pct_delta_emoji(actual: float, expected: float, higher_is_better: bool = True) -> str:
    """Return ✅/⚠️/❌ based on how actual compares to expected."""
    if higher_is_better:
        if actual >= expected * 0.85:
            return "✅"
        elif actual >= expected * 0.60:
            return "⚠️"
        else:
            return "❌"
    else:
        # Lower is better (e.g. drawdown)
        if actual <= expected * 1.15:
            return "✅"
        elif actual <= expected * 1.50:
            return "⚠️"
        else:
            return "❌"


def format_experiment_block(m: Dict, report_date: str) -> str:
    """Format one experiment's metrics as a Telegram HTML block."""
    exp_id = m["exp_id"]
    label = m["label"]

    if m.get("error"):
        return f"<b>{exp_id} ({label})</b>\n⚠️ DB error: {m['error']}\n"

    total = m["total_trades"]
    open_pos = m["open_positions"]
    wr = m["win_rate_pct"]
    pnl = m["total_pnl"]
    pf = m["profit_factor"]
    dd = m["max_drawdown_pct"]
    ann = m["annualised_return_pct"]
    exp_ann = m["expected_annual_return_pct"]
    exp_wr = m["expected_win_rate_pct"]

    wr_icon = _pct_delta_emoji(wr, exp_wr)
    ann_icon = _pct_delta_emoji(ann, exp_ann) if total >= 5 else "🔄"
    dd_icon = _pct_delta_emoji(dd, 20.0, higher_is_better=False)  # expect DD < 20%

    pnl_sign = "+" if pnl >= 0 else ""
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

    lines = [
        f"<b>📊 {exp_id} — {label}</b>",
        f"  Trades: {total} closed, {open_pos} open",
        f"  P&amp;L: {pnl_sign}${pnl:,.2f}",
        f"  Win rate: {wr:.1f}% {wr_icon}  (bt: {exp_wr:.0f}%)",
        f"  Profit factor: {pf_str}",
        f"  Max drawdown: {dd:.1f}% {dd_icon}",
    ]
    if total >= 5:
        lines.append(f"  Ann. return: {ann:+.1f}% {ann_icon}  (bt: {exp_ann:.1f}%)")
    else:
        lines.append(f"  Ann. return: 🔄 (need ≥5 trades, have {total})")
    return "\n".join(lines)


def build_report(
    report_date: Optional[str] = None,
    account_size: float = 100_000,
) -> str:
    """Build the full comparison report as a Telegram HTML message."""
    if report_date is None:
        report_date = date.today().isoformat()

    lines = [
        "📋 <b>DAILY P&amp;L COMPARISON REPORT</b>",
        f"<i>{report_date}</i>",
        "",
    ]

    for exp_id in EXPECTATIONS:
        m = load_experiment_metrics(exp_id, account_size=account_size)
        lines.append(format_experiment_block(m, report_date))
        lines.append("")

    # Cross-experiment comparison if both have trades
    metrics = {exp: load_experiment_metrics(exp, account_size=account_size)
               for exp in EXPECTATIONS}
    m400 = metrics["EXP-400"]
    m401 = metrics["EXP-401"]

    if m400["total_trades"] > 0 and m401["total_trades"] > 0:
        pnl_diff = m400["total_pnl"] - m401["total_pnl"]
        sign = "+" if pnl_diff >= 0 else ""
        lines += [
            "<b>⚖️ Cross-experiment delta (EXP-400 vs EXP-401)</b>",
            f"  P&amp;L diff: {sign}${pnl_diff:,.2f}",
            f"  Win rate diff: {m400['win_rate_pct'] - m401['win_rate_pct']:+.1f}pp",
            "",
        ]

    lines.append("<i>bt = backtest expectation (after 2x slippage)</i>")
    return "\n".join(lines)


def send_report(report_date: Optional[str] = None, account_size: float = 100_000) -> bool:
    """Build and send the report via Telegram. Returns True on success."""
    msg = build_report(report_date=report_date, account_size=account_size)
    from shared.telegram_alerts import send_message
    return send_message(msg)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Daily P&L comparison report for paper trading")
    parser.add_argument("--date", default=None, help="Report date (YYYY-MM-DD). Default: today")
    parser.add_argument("--account-size", type=float, default=100_000,
                        help="Account size in USD (default: 100000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the report to stdout without sending to Telegram")
    args = parser.parse_args()

    report = build_report(report_date=args.date, account_size=args.account_size)

    if args.dry_run:
        # Strip HTML tags for readable terminal output
        import re
        plain = re.sub(r"<[^>]+>", "", report).replace("&amp;", "&")
        print(plain)
    else:
        ok = send_report(report_date=args.date, account_size=args.account_size)
        if ok:
            print("Report sent to Telegram.")
        else:
            print("Telegram send failed — printing report:\n")
            import re
            plain = re.sub(r"<[^>]+>", "", report).replace("&amp;", "&")
            print(plain)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
