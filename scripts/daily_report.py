#!/usr/bin/env python3
"""
Daily Report Generator (INF-4)

Generates a comprehensive HTML report with positions, P&L, economic events,
and sends it via Telegram as an HTML document.

Usage:
    python scripts/daily_report.py --config configs/paper_champion.yaml --env-file .env.champion
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy helper (kept for backward compatibility with notify_daily_summary)
# ---------------------------------------------------------------------------

def get_daily_summary_metrics(
    report_date: Optional[str] = None,
    account_size: float = 100_000.0,
) -> dict:
    """Collect today's summary metrics from SQLite and Alpaca.

    Returns a dict matching the kwargs expected by
    TelegramAlertFormatter.format_daily_summary().
    """
    from shared.database import get_trades, get_latest_alerts

    date_str = report_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    db_path = os.environ.get("PILOTAI_DB_PATH")

    closed_all = (
        get_trades(status="closed_profit", path=db_path)
        + get_trades(status="closed_loss", path=db_path)
    )
    closed_today = [
        t for t in closed_all if str(t.get("exit_date", "")).startswith(date_str)
    ]
    wins = [t for t in closed_today if (t.get("pnl") or 0) > 0]
    losses = [t for t in closed_today if (t.get("pnl") or 0) <= 0]
    day_pnl = sum(t.get("pnl", 0) for t in closed_today)
    day_pnl_pct = (day_pnl / account_size * 100) if account_size > 0 else 0.0

    open_trades = get_trades(status="open", path=db_path)
    alerts = get_latest_alerts(limit=100, path=db_path)
    alerts_today = [
        a for a in alerts if str(a.get("created_at", "")).startswith(date_str)
    ]

    best = worst = ""
    if closed_today:
        sorted_by_pnl = sorted(
            closed_today, key=lambda t: t.get("pnl", 0), reverse=True
        )
        b, w = sorted_by_pnl[0], sorted_by_pnl[-1]
        best = f"{b.get('ticker', 'UNK')} {'+' if (b.get('pnl') or 0) >= 0 else ''}${b.get('pnl', 0):.2f}"
        worst = f"{w.get('ticker', 'UNK')} {'+' if (w.get('pnl') or 0) >= 0 else ''}${w.get('pnl', 0):.2f}"

    return {
        "date": date_str,
        "alerts_fired": len(alerts_today),
        "closed_today": len(closed_today),
        "wins": len(wins),
        "losses": len(losses),
        "day_pnl": round(day_pnl, 2),
        "day_pnl_pct": round(day_pnl_pct, 4),
        "open_positions": len(open_trades),
        "total_risk_pct": 0.0,
        "account_balance": account_size + day_pnl,
        "pct_from_start": round(day_pnl_pct, 4),
        "best": best,
        "worst": worst,
    }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_env_file(env_path: str) -> None:
    """Load a .env file into os.environ (simple key=value parser)."""
    p = Path(env_path)
    if not p.exists():
        logger.warning("Env file not found: %s", env_path)
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def get_alpaca_equity(config: dict) -> Optional[float]:
    """Fetch current equity from Alpaca account API."""
    try:
        base_url = config.get("alpaca", {}).get("base_url", "https://paper-api.alpaca.markets")
        api_key = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_API_SECRET", "")
        if not api_key or not api_secret:
            return None
        resp = requests.get(
            f"{base_url}/v2/account",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return float(resp.json().get("equity", 0))
    except Exception as e:
        logger.warning("Could not fetch Alpaca equity: %s", e)
        return None


def collect_report_data(config: dict, report_date: Optional[str] = None) -> dict:
    """Collect all data needed for the daily report."""
    from shared.database import get_trades
    from shared.economic_calendar import EconomicCalendar

    date_str = report_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    db_path = config.get("db_path")
    if db_path:
        os.environ["PILOTAI_DB_PATH"] = db_path

    experiment_id = config.get("experiment_id", "UNKNOWN")
    starting_equity = config.get("risk", {}).get("account_size", 100_000)

    # Current equity
    alpaca_equity = get_alpaca_equity(config)
    current_equity = alpaca_equity or starting_equity

    # Trades
    open_trades = get_trades(status="open", path=db_path)
    closed_profit = get_trades(status="closed_profit", path=db_path)
    closed_loss = get_trades(status="closed_loss", path=db_path)
    closed_manual = get_trades(status="closed_manual", path=db_path)
    closed_expiry = get_trades(status="closed_expiry", path=db_path)

    all_closed = closed_profit + closed_loss + closed_manual + closed_expiry

    # Today's closed
    closed_today = [
        t for t in all_closed if str(t.get("exit_date", "")).startswith(date_str)
    ]

    # Rolling aggregates
    ref_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d7 = (ref_date - timedelta(days=7)).strftime("%Y-%m-%d")
    d30 = (ref_date - timedelta(days=30)).strftime("%Y-%m-%d")

    closed_7d = [
        t for t in all_closed
        if d7 <= str(t.get("exit_date", ""))[:10] <= date_str
    ]
    closed_30d = [
        t for t in all_closed
        if d30 <= str(t.get("exit_date", ""))[:10] <= date_str
    ]

    # Economic calendar
    cal = EconomicCalendar()
    upcoming_events = cal.get_upcoming_events(days_ahead=14, reference_date=ref_date)

    return {
        "date": date_str,
        "experiment_id": experiment_id,
        "starting_equity": starting_equity,
        "current_equity": current_equity,
        "alpaca_equity": alpaca_equity,
        "open_trades": open_trades,
        "all_closed": all_closed,
        "closed_today": closed_today,
        "closed_7d": closed_7d,
        "closed_30d": closed_30d,
        "upcoming_events": upcoming_events,
        "config": config,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #fff; color: #1a1a1a; margin: 0; padding: 20px; font-size: 14px; }
h1 { color: #1a1a1a; border-bottom: 3px solid #2563eb; padding-bottom: 8px; margin-top: 0; }
h2 { color: #374151; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; margin-top: 28px; }
.meta { color: #6b7280; font-size: 13px; margin-bottom: 16px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 20px 0; font-size: 13px; }
th { background: #f3f4f6; color: #374151; text-align: left; padding: 8px 10px;
     border-bottom: 2px solid #d1d5db; font-weight: 600; }
td { padding: 6px 10px; border-bottom: 1px solid #e5e7eb; }
tr:hover { background: #f9fafb; }
.profit { color: #059669; font-weight: 600; }
.loss { color: #dc2626; font-weight: 600; }
.neutral { color: #6b7280; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px; margin: 16px 0; }
.summary-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
                padding: 12px 16px; }
.summary-card .label { font-size: 11px; color: #6b7280; text-transform: uppercase;
                       letter-spacing: 0.5px; }
.summary-card .value { font-size: 20px; font-weight: 700; margin-top: 4px; }
.event-badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
               font-size: 11px; font-weight: 600; text-transform: uppercase; margin-right: 6px; }
.event-fomc { background: #fef3c7; color: #92400e; }
.event-cpi { background: #dbeafe; color: #1e40af; }
.event-ppi { background: #ede9fe; color: #5b21b6; }
.event-jobs { background: #d1fae5; color: #065f46; }
.event-gdp { background: #fce7f3; color: #9d174d; }
.footer { margin-top: 30px; padding-top: 12px; border-top: 1px solid #e5e7eb;
          font-size: 11px; color: #9ca3af; }
"""


def _pnl_class(val: float) -> str:
    if val > 0:
        return "profit"
    elif val < 0:
        return "loss"
    return "neutral"


def _pnl_str(val: float, prefix: str = "$") -> str:
    sign = "+" if val > 0 else ""
    return f'{sign}{prefix}{val:,.2f}'


def _pct_str(val: float) -> str:
    sign = "+" if val > 0 else ""
    return f'{sign}{val:.2f}%'


def _compute_stats(trades: List[dict], starting_equity: float) -> dict:
    """Compute summary stats for a list of closed trades."""
    if not trades:
        return {"count": 0, "wins": 0, "losses": 0, "total_pnl": 0,
                "win_rate": 0, "avg_win": 0, "avg_loss": 0, "pnl_pct": 0}
    wins = [t for t in trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl") or 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    avg_win = (sum(t.get("pnl", 0) for t in wins) / len(wins)) if wins else 0
    avg_loss = (sum(t.get("pnl", 0) for t in losses) / len(losses)) if losses else 0
    win_rate = (len(wins) / len(trades) * 100) if trades else 0
    pnl_pct = (total_pnl / starting_equity * 100) if starting_equity else 0
    return {
        "count": len(trades), "wins": len(wins), "losses": len(losses),
        "total_pnl": total_pnl, "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss, "pnl_pct": pnl_pct,
    }


def _hold_days(trade: dict) -> str:
    """Calculate hold days from entry_date to exit_date."""
    try:
        entry = trade.get("entry_date", "")[:10]
        exit_ = trade.get("exit_date", "")[:10]
        if entry and exit_:
            d1 = datetime.strptime(entry, "%Y-%m-%d")
            d2 = datetime.strptime(exit_, "%Y-%m-%d")
            return str((d2 - d1).days)
    except (ValueError, TypeError):
        pass
    return "—"


def _dte_remaining(trade: dict, report_date: str) -> str:
    """Calculate DTE remaining from report date to expiration."""
    try:
        exp = str(trade.get("expiration", ""))[:10]
        if exp:
            d_exp = datetime.strptime(exp, "%Y-%m-%d")
            d_now = datetime.strptime(report_date, "%Y-%m-%d")
            return str(max(0, (d_exp - d_now).days))
    except (ValueError, TypeError):
        pass
    return "—"


def generate_html(data: dict) -> str:
    """Generate the full HTML daily report."""
    d = data
    starting_equity = d["starting_equity"]
    current_equity = d["current_equity"]
    total_pnl = current_equity - starting_equity
    all_stats = _compute_stats(d["all_closed"], starting_equity)

    parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Daily Report — {d['experiment_id']} — {d['date']}</title>
<style>{_CSS}</style></head><body>
<h1>📊 Daily Report</h1>
<p class="meta"><b>{d['experiment_id']}</b> &bull; {d['date']}
{' &bull; Equity from Alpaca ✓' if d['alpaca_equity'] else ' &bull; Equity from config'}</p>
"""]

    # -- Summary Cards --
    parts.append('<div class="summary-grid">')
    for label, value, cls in [
        ("Current Equity", f"${current_equity:,.2f}", _pnl_class(total_pnl)),
        ("Total P&L", _pnl_str(total_pnl), _pnl_class(total_pnl)),
        ("Total Return", _pct_str(total_pnl / starting_equity * 100 if starting_equity else 0), _pnl_class(total_pnl)),
        ("Win Rate", f"{all_stats['win_rate']:.1f}%", "neutral"),
        ("Open Positions", str(len(d["open_trades"])), "neutral"),
        ("Total Trades", str(all_stats["count"]), "neutral"),
    ]:
        parts.append(f'<div class="summary-card"><div class="label">{label}</div>'
                     f'<div class="value {cls}">{value}</div></div>')
    parts.append('</div>')

    # -- Open Positions --
    parts.append('<h2>📈 Open Positions</h2>')
    if d["open_trades"]:
        parts.append('<table><tr><th>Ticker</th><th>Strategy</th><th>Direction</th>'
                     '<th>Entry Date</th><th>DTE</th><th>Strikes</th><th>Credit</th>'
                     '<th>Contracts</th></tr>')
        for t in d["open_trades"]:
            meta = {}
            if isinstance(t.get("metadata"), str):
                try: meta = json.loads(t["metadata"])
                except: pass
            elif isinstance(t.get("metadata"), dict):
                meta = t["metadata"]
            direction = t.get("direction") or meta.get("direction", "—")
            strategy = (t.get("strategy_type") or t.get("type", "—")).replace("_", " ").title()
            credit = t.get("credit") or t.get("credit_per_spread") or 0
            strikes = f"${t.get('short_strike', '?')}/{t.get('long_strike', '?')}"
            dte = _dte_remaining(t, d["date"])
            parts.append(f'<tr><td><b>{t.get("ticker", "?")}</b></td><td>{strategy}</td>'
                         f'<td>{direction}</td><td>{str(t.get("entry_date", ""))[:10]}</td>'
                         f'<td>{dte}</td><td>{strikes}</td>'
                         f'<td>${credit:.2f}</td><td>{t.get("contracts", 1)}</td></tr>')
        parts.append('</table>')
    else:
        parts.append('<p class="neutral">No open positions.</p>')

    # -- Today's Closed Trades --
    parts.append(f'<h2>🔒 Closed Today ({len(d["closed_today"])})</h2>')
    if d["closed_today"]:
        parts.append('<table><tr><th>Ticker</th><th>Strategy</th><th>Entry</th>'
                     '<th>Exit</th><th>Hold</th><th>P&L $</th><th>P&L %</th>'
                     '<th>Exit Reason</th></tr>')
        for t in d["closed_today"]:
            pnl = t.get("pnl", 0) or 0
            credit = t.get("credit") or 1
            pnl_pct = (pnl / (credit * (t.get("contracts", 1) or 1) * 100)) * 100 if credit else 0
            cls = _pnl_class(pnl)
            strategy = (t.get("strategy_type") or "—").replace("_", " ").title()
            reason = (t.get("exit_reason") or "—").replace("_", " ").title()
            parts.append(f'<tr><td><b>{t.get("ticker", "?")}</b></td><td>{strategy}</td>'
                         f'<td>{str(t.get("entry_date", ""))[:10]}</td>'
                         f'<td>{str(t.get("exit_date", ""))[:10]}</td>'
                         f'<td>{_hold_days(t)}d</td>'
                         f'<td class="{cls}">{_pnl_str(pnl)}</td>'
                         f'<td class="{cls}">{_pct_str(pnl_pct)}</td>'
                         f'<td>{reason}</td></tr>')
        parts.append('</table>')
    else:
        parts.append('<p class="neutral">No trades closed today.</p>')

    # -- Account Summary --
    parts.append('<h2>💰 Account Summary</h2>')
    parts.append('<table><tr><th>Metric</th><th>Value</th></tr>')
    for label, val in [
        ("Starting Equity", f"${starting_equity:,.2f}"),
        ("Current Equity", f"${current_equity:,.2f}"),
        ("Total P&L", f'<span class="{_pnl_class(total_pnl)}">{_pnl_str(total_pnl)}</span>'),
        ("Total Return", f'<span class="{_pnl_class(total_pnl)}">{_pct_str(total_pnl / starting_equity * 100 if starting_equity else 0)}</span>'),
        ("Total Trades", str(all_stats["count"])),
        ("Wins / Losses", f'{all_stats["wins"]} / {all_stats["losses"]}'),
        ("Win Rate", f'{all_stats["win_rate"]:.1f}%'),
        ("Avg Win", f'<span class="profit">{_pnl_str(all_stats["avg_win"])}</span>' if all_stats["avg_win"] else "—"),
        ("Avg Loss", f'<span class="loss">{_pnl_str(all_stats["avg_loss"])}</span>' if all_stats["avg_loss"] else "—"),
    ]:
        parts.append(f'<tr><td>{label}</td><td>{val}</td></tr>')
    parts.append('</table>')

    # -- Strategy Breakdown --
    parts.append('<h2>📋 Strategy Breakdown</h2>')
    strategy_types: Dict[str, List[dict]] = {}
    for t in d["all_closed"]:
        st = (t.get("strategy_type") or t.get("type") or "unknown")
        strategy_types.setdefault(st, []).append(t)

    if strategy_types:
        parts.append('<table><tr><th>Strategy</th><th>Trades</th><th>Wins</th>'
                     '<th>Losses</th><th>Win Rate</th><th>Total P&L</th><th>Avg Win</th>'
                     '<th>Avg Loss</th></tr>')
        for st_name, st_trades in sorted(strategy_types.items()):
            s = _compute_stats(st_trades, starting_equity)
            cls = _pnl_class(s["total_pnl"])
            display = st_name.replace("_", " ").title()
            parts.append(f'<tr><td><b>{display}</b></td><td>{s["count"]}</td>'
                         f'<td>{s["wins"]}</td><td>{s["losses"]}</td>'
                         f'<td>{s["win_rate"]:.1f}%</td>'
                         f'<td class="{cls}">{_pnl_str(s["total_pnl"])}</td>'
                         f'<td class="profit">{_pnl_str(s["avg_win"])}</td>'
                         f'<td class="loss">{_pnl_str(s["avg_loss"])}</td></tr>')
        parts.append('</table>')
    else:
        parts.append('<p class="neutral">No closed trades yet.</p>')

    # -- Rolling Aggregates --
    parts.append('<h2>📅 Rolling Aggregates</h2>')
    s7 = _compute_stats(d["closed_7d"], starting_equity)
    s30 = _compute_stats(d["closed_30d"], starting_equity)
    parts.append('<table><tr><th>Period</th><th>Trades</th><th>Win Rate</th>'
                 '<th>P&L</th><th>Return</th></tr>')
    for label, s in [("Last 7 Days", s7), ("Last 30 Days", s30)]:
        cls = _pnl_class(s["total_pnl"])
        parts.append(f'<tr><td><b>{label}</b></td><td>{s["count"]}</td>'
                     f'<td>{s["win_rate"]:.1f}%</td>'
                     f'<td class="{cls}">{_pnl_str(s["total_pnl"])}</td>'
                     f'<td class="{cls}">{_pct_str(s["pnl_pct"])}</td></tr>')
    parts.append('</table>')

    # -- Upcoming Events --
    parts.append('<h2>🗓️ Upcoming Economic Events</h2>')
    if d["upcoming_events"]:
        parts.append('<table><tr><th>Date</th><th>Event</th><th>Description</th>'
                     '<th>Importance</th></tr>')
        for ev in d["upcoming_events"][:10]:
            ev_type = ev.get("event_type", "")
            badge_cls = f"event-{ev_type}"
            ev_date = ev["date"].strftime("%Y-%m-%d") if hasattr(ev["date"], "strftime") else str(ev["date"])[:10]
            imp_pct = int(ev.get("importance", 0.5) * 100)
            parts.append(f'<tr><td>{ev_date}</td>'
                         f'<td><span class="event-badge {badge_cls}">{ev_type.upper()}</span></td>'
                         f'<td>{ev.get("description", "")}</td>'
                         f'<td>{imp_pct}%</td></tr>')
        parts.append('</table>')
    else:
        parts.append('<p class="neutral">No upcoming events in the next 14 days.</p>')

    # -- Footer --
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f'<div class="footer">Generated {now_str} &bull; PilotAI Credit Spreads</div>')
    parts.append('</body></html>')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Telegram delivery
# ---------------------------------------------------------------------------

def send_html_report_telegram(html: str, report_date: str, experiment_id: str) -> bool:
    """Send the HTML report as a document via Telegram Bot API."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram not configured — skipping send")
        return False

    filename = f"daily_report_{experiment_id}_{report_date}.html"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, prefix="pilotai_report_"
        ) as f:
            f.write(html)
            tmp_path = f.name

        url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        caption = f"📊 Daily Report — {experiment_id} — {report_date}"

        with open(tmp_path, "rb") as doc:
            resp = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                files={"document": (filename, doc, "text/html")},
                timeout=30,
            )
        resp.raise_for_status()
        logger.info("Report sent to Telegram ✓")
        return True
    except Exception as e:
        logger.error("Failed to send report via Telegram: %s", e)
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PilotAI Daily Report Generator")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--env-file", help="Path to .env file")
    parser.add_argument("--date", help="Report date (YYYY-MM-DD), default: today UTC")
    parser.add_argument("--output", help="Write HTML to file instead of sending")
    parser.add_argument("--no-telegram", action="store_true", help="Skip Telegram send")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load env and config
    if args.env_file:
        load_env_file(args.env_file)

    config = load_config(args.config)

    # Set DB path from config
    db_path = config.get("db_path")
    if db_path:
        os.environ["PILOTAI_DB_PATH"] = db_path

    # Set experiment ID
    exp_id = config.get("experiment_id", "UNKNOWN")
    os.environ.setdefault("EXPERIMENT_ID", exp_id)

    report_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("Generating daily report for %s [%s]", exp_id, report_date)

    # Collect data and generate
    data = collect_report_data(config, report_date)
    html = generate_html(data)

    # Output
    if args.output:
        Path(args.output).write_text(html)
        logger.info("Report written to %s", args.output)
    elif not args.no_telegram:
        send_html_report_telegram(html, report_date, exp_id)
    else:
        print(html)


if __name__ == "__main__":
    main()
