#!/usr/bin/env python3
"""Weekly Friday EOD summary report for paper trading validation.

Reads both paper trading DBs (EXP-400 champion + EXP-401), computes
week-over-week stats, compares running totals to backtest pace, and sends
a formatted Telegram message.

Usage:
    python scripts/weekly_summary_report.py
    python scripts/weekly_summary_report.py --week-end 2026-03-14
    python scripts/weekly_summary_report.py --dry-run

Designed to be called from deploy.sh or a cron at Friday 16:05 ET.
"""

import argparse
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from shared.database import get_trades

# ─── Experiment registry (mirrors daily_pnl_report.py) ────────────────────────

EXPERIMENTS = {
    "EXP-400": {
        "label": "Champion (exp_213)",
        "annual_return_pct": 32.7,
        "win_rate_pct": 78.0,
        "db_path": str(ROOT / "data" / "pilotai_exp400.db"),
    },
    "EXP-401": {
        "label": "EXP-401",
        "annual_return_pct": 26.9,
        "win_rate_pct": 75.0,
        "db_path": str(ROOT / "data" / "pilotai_exp401.db"),
    },
}

TRADING_DAYS_PER_YEAR = 252
TRADING_WEEKS_PER_YEAR = 52


# ─── Metrics helpers ──────────────────────────────────────────────────────────

def _parse_date(s: str) -> Optional[date]:
    """Parse ISO date string (first 10 chars). Returns None on failure."""
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def _is_closed(status: str) -> bool:
    return bool(status) and status.startswith("closed")


def _filter_by_week(trades: List[Dict], week_start: date, week_end: date) -> List[Dict]:
    """Return trades whose exit_date (or entry_date) falls within [week_start, week_end]."""
    result = []
    for t in trades:
        d = _parse_date(t.get("exit_date") or t.get("entry_date"))
        if d and week_start <= d <= week_end:
            result.append(t)
    return result


def compute_week_metrics(
    all_trades: List[Dict],
    week_start: date,
    week_end: date,
) -> Dict:
    """Compute metrics for a single week window."""
    week_trades = _filter_by_week(all_trades, week_start, week_end)

    new_entries = [
        t for t in week_trades
        if _parse_date(t.get("entry_date")) and week_start <= _parse_date(t.get("entry_date")) <= week_end
    ]
    closed = [t for t in week_trades if _is_closed(t.get("status", ""))]
    wins = [t for t in closed if (t.get("pnl") or 0) > 0]

    week_pnl = sum(t.get("pnl") or 0 for t in closed)
    win_rate = len(wins) / len(closed) * 100 if closed else 0.0

    # Max intra-week drawdown from closed trades sorted by exit date
    sorted_closed = sorted(closed, key=lambda x: x.get("exit_date") or "")
    running_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted_closed:
        running_pnl += t.get("pnl") or 0
        if running_pnl > peak:
            peak = running_pnl
        dd = peak - running_pnl
        if dd > max_dd:
            max_dd = dd

    return {
        "new_entries": len(new_entries),
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "win_rate_pct": win_rate,
        "week_pnl": week_pnl,
        "max_intraweek_dd": max_dd,
    }


def compute_running_totals(all_trades: List[Dict], account_size: float = 100_000) -> Dict:
    """Compute all-time cumulative stats."""
    closed = [t for t in all_trades if _is_closed(t.get("status", ""))]
    open_pos = [t for t in all_trades if t.get("status") == "open"]
    wins = [t for t in closed if (t.get("pnl") or 0) > 0]

    total_pnl = sum(t.get("pnl") or 0 for t in closed)
    win_rate = len(wins) / len(closed) * 100 if closed else 0.0

    # Annualised return from first→last exit date
    annualised = 0.0
    dates = sorted(
        _parse_date(t.get("exit_date") or t.get("entry_date"))
        for t in closed
        if _parse_date(t.get("exit_date") or t.get("entry_date"))
    )
    if len(dates) >= 2:
        days_live = max((dates[-1] - dates[0]).days, 1)
        total_return_pct = total_pnl / account_size * 100
        annualised = total_return_pct * (365 / days_live)

    return {
        "total_closed": len(closed),
        "open_positions": len(open_pos),
        "total_pnl": total_pnl,
        "win_rate_pct": win_rate,
        "annualised_return_pct": annualised,
    }


def pace_assessment(
    actual_annual_pct: float,
    expected_annual_pct: float,
    total_closed: int,
) -> str:
    """Return a short pace string with emoji."""
    if total_closed < 5:
        return "🔄 Too early (need ≥5 trades)"
    ratio = actual_annual_pct / expected_annual_pct if expected_annual_pct else 0
    if ratio >= 0.85:
        return f"✅ On pace ({actual_annual_pct:+.1f}% ann. vs {expected_annual_pct:.1f}% target)"
    elif ratio >= 0.60:
        return f"⚠️ Below pace ({actual_annual_pct:+.1f}% ann. vs {expected_annual_pct:.1f}% target)"
    else:
        return f"❌ Off pace ({actual_annual_pct:+.1f}% ann. vs {expected_annual_pct:.1f}% target)"


# ─── Formatter ────────────────────────────────────────────────────────────────

def format_exp_block(
    exp_id: str,
    week_metrics: Dict,
    running: Dict,
    expected: Dict,
    week_start: date,
    week_end: date,
) -> str:
    label = expected["label"]
    wm = week_metrics
    rt = running

    lines = [
        f"<b>📊 {exp_id} — {label}</b>",
        f"  Week {week_start} → {week_end}:",
        f"    New entries: {wm['new_entries']}  |  Closed: {wm['closed_trades']}",
    ]

    if wm["closed_trades"] > 0:
        pnl_sign = "+" if wm["week_pnl"] >= 0 else ""
        lines.append(
            f"    Win rate: {wm['win_rate_pct']:.0f}%  |  "
            f"P&amp;L: {pnl_sign}${wm['week_pnl']:,.2f}"
        )
        if wm["max_intraweek_dd"] > 0:
            lines.append(f"    Intra-week DD: -${wm['max_intraweek_dd']:,.2f}")
    else:
        lines.append("    No closed trades this week")

    lines.append(
        f"  Running totals ({rt['total_closed']} closed, {rt['open_positions']} open):"
    )
    total_sign = "+" if rt["total_pnl"] >= 0 else ""
    lines += [
        f"    Total P&amp;L: {total_sign}${rt['total_pnl']:,.2f}",
        f"    Win rate: {rt['win_rate_pct']:.1f}%  (bt: {expected['win_rate_pct']:.0f}%)",
        f"  Pace: {pace_assessment(rt['annualised_return_pct'], expected['annual_return_pct'], rt['total_closed'])}",
    ]
    return "\n".join(lines)


def build_weekly_report(
    week_end: Optional[date] = None,
    account_size: float = 100_000,
) -> str:
    """Build the full weekly report as a Telegram HTML message string."""
    if week_end is None:
        week_end = date.today()
    week_start = week_end - timedelta(days=6)

    lines = [
        "📅 <b>WEEKLY SUMMARY REPORT</b>",
        f"<i>Week ending {week_end} (Mon {week_start} → {week_end})</i>",
        "",
    ]

    all_metrics = {}
    for exp_id, cfg in EXPERIMENTS.items():
        try:
            trades = get_trades(source="execution", path=cfg["db_path"])
        except Exception as e:
            lines.append(f"<b>{exp_id}</b>: ⚠️ DB error — {e}\n")
            continue

        week_m = compute_week_metrics(trades, week_start, week_end)
        running = compute_running_totals(trades, account_size=account_size)
        all_metrics[exp_id] = {"week": week_m, "running": running}
        lines.append(format_exp_block(exp_id, week_m, running, cfg, week_start, week_end))
        lines.append("")

    # Cross-experiment weekly P&L comparison
    if len(all_metrics) == 2:
        m400 = all_metrics.get("EXP-400", {}).get("week", {})
        m401 = all_metrics.get("EXP-401", {}).get("week", {})
        if m400 and m401:
            diff = m400.get("week_pnl", 0) - m401.get("week_pnl", 0)
            sign = "+" if diff >= 0 else ""
            lines += [
                "<b>⚖️ Week delta (EXP-400 vs EXP-401)</b>",
                f"  P&amp;L diff this week: {sign}${diff:,.2f}",
                "",
            ]

    lines.append("<i>bt = backtest expectation · ann = annualised from live data</i>")
    return "\n".join(lines)


def send_weekly_report(
    week_end: Optional[date] = None,
    account_size: float = 100_000,
) -> bool:
    """Build and send the weekly report via Telegram. Returns True on success."""
    msg = build_weekly_report(week_end=week_end, account_size=account_size)
    from shared.telegram_alerts import send_message
    return send_message(msg)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly paper trading summary report")
    parser.add_argument(
        "--week-end",
        default=None,
        help="Last day of the week to report (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--account-size", type=float, default=100_000,
        help="Account size in USD (default: 100000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print to stdout, do not send to Telegram",
    )
    args = parser.parse_args()

    week_end = date.fromisoformat(args.week_end) if args.week_end else None
    report = build_weekly_report(week_end=week_end, account_size=args.account_size)

    if args.dry_run:
        plain = re.sub(r"<[^>]+>", "", report).replace("&amp;", "&")
        print(plain)
        return

    ok = send_weekly_report(week_end=week_end, account_size=args.account_size)
    if ok:
        print("Weekly report sent to Telegram.")
    else:
        print("Telegram send failed — printing report:\n")
        plain = re.sub(r"<[^>]+>", "", report).replace("&amp;", "&")
        print(plain)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
