#!/usr/bin/env python3
"""Deviation tracker report — compares paper trading signals vs backtest expectations.

Usage:
    python scripts/deviation_report.py
    python scripts/deviation_report.py --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from shared.deviation_tracker import (
    check_deviation_alerts,
    get_deviation_history,
    get_latest_deviation,
    get_rolling_alignment,
)

EXPERIMENTS = {
    "EXP-400": {
        "label": "Champion",
        "db_path": str(ROOT / "data" / "pilotai_champion.db"),
    },
    "EXP-401": {
        "label": "EXP-401",
        "db_path": str(ROOT / "data" / "pilotai_exp401.db"),
    },
}


def _status_icon(status: str) -> str:
    return {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}.get(status, "❓")


def format_exp_block(exp_id: str, cfg: dict) -> str:
    label = cfg["label"]
    db = cfg["db_path"]
    lines = [f"<b>🔍 {exp_id} — {label}</b>"]

    # Rolling per-trade alignment (last 20 trades)
    try:
        rolling = get_rolling_alignment(db_path=db)
        n = rolling["trade_count"]
        if n == 0:
            lines.append("  Per-trade alignment: 🔄 No closed trades yet")
        else:
            score = rolling["alignment_score"] * 100
            credit_dev = rolling["credit_deviation"] * 100
            icon = "✅" if score >= 70 else ("⚠️" if score >= 50 else "❌")
            lines.append(f"  Per-trade alignment ({n} trades): {score:.0f}% {icon}")
            lines.append(f"  Avg credit deviation: {credit_dev:.1f}%")
    except Exception as e:
        lines.append(f"  Per-trade alignment: ⚠️ {e}")

    # Latest daily snapshot
    try:
        snap = get_latest_deviation(db_path=db)
        if snap:
            status = snap.get("overall_status", "INFO")
            icon = _status_icon(status)
            lines.append(
                f"  Latest snapshot ({snap.get('snapshot_date', '?')}): "
                f"{status} {icon}"
            )
            live_wr = snap.get("live_win_rate")
            bt_wr = snap.get("bt_win_rate")
            if live_wr is not None and bt_wr is not None:
                lines.append(f"  Win rate: live {live_wr:.1f}% vs bt {bt_wr:.1f}%")
            live_ret = snap.get("live_return_pct")
            bt_ret = snap.get("bt_return_pct")
            if live_ret is not None and bt_ret is not None:
                lines.append(f"  Return: live {live_ret:+.1f}% vs bt {bt_ret:+.1f}%")
            # Deviation alerts
            alerts = check_deviation_alerts(snap)
            for a in alerts[:3]:  # cap at 3 to keep message short
                lines.append(f"  ⚠️ {a}")
        else:
            lines.append("  Daily snapshot: 🔄 None recorded yet")
    except Exception as e:
        lines.append(f"  Daily snapshot: ⚠️ {e}")

    # Trend: last 7 days of snapshots
    try:
        history = get_deviation_history(days=7, db_path=db)
        if len(history) >= 2:
            statuses = [h.get("overall_status", "INFO") for h in history]
            fail_count = statuses.count("FAIL")
            warn_count = statuses.count("WARN")
            if fail_count > 0:
                lines.append(f"  7-day trend: {fail_count} FAIL days ❌")
            elif warn_count > 0:
                lines.append(f"  7-day trend: {warn_count} WARN days ⚠️")
            else:
                lines.append(f"  7-day trend: stable ✅")
    except Exception:
        pass

    return "\n".join(lines)


def build_report() -> str:
    from datetime import date
    lines = [
        "📐 <b>DEVIATION TRACKER REPORT</b>",
        f"<i>{date.today()}</i>",
        "",
    ]
    for exp_id, cfg in EXPERIMENTS.items():
        lines.append(format_exp_block(exp_id, cfg))
        lines.append("")
    lines.append("<i>Alignment = paper outcomes matching backtest expectations</i>")
    return "\n".join(lines)


def send_report() -> bool:
    from shared.telegram_alerts import send_message
    return send_message(build_report())


def main():
    parser = argparse.ArgumentParser(description="Deviation tracker report")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_report()
    if args.dry_run:
        print(re.sub(r"<[^>]+>", "", report).replace("&amp;", "&"))
        return
    ok = send_report()
    print("Sent." if ok else "Failed — check Telegram config.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
