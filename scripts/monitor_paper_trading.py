#!/usr/bin/env python3
"""
Live Paper Trading Monitor — EXP-400 / 401 / 503 / 600

Queries real Alpaca paper accounts, computes running Sharpe/DD/returns,
and generates an HTML dashboard. Can run as a daily cron job.

Rule Zero: ZERO synthetic data. Every number is either:
  - Fetched live from Alpaca REST API
  - Read from prior snapshot files (logs/paper_monitor_history.json)
  - Computed arithmetically from the above

No np.random. No fabricated fills. No Black-Scholes prices.

Usage:
    python3 scripts/monitor_paper_trading.py                  # snapshot + HTML
    python3 scripts/monitor_paper_trading.py --html-only      # regenerate HTML from history
    python3 scripts/monitor_paper_trading.py --json           # print JSON to stdout
    python3 scripts/monitor_paper_trading.py --dry-run        # don't write files

Cron example (daily at 16:05 ET):
    5 16 * * 1-5  cd /Users/charles/pilotai && source .env && \\
                  python3 scripts/monitor_paper_trading.py >> logs/monitor.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.check_accounts import (
    BASE_URL, _discover_accounts, _headers, _fetch_account, _fetch_positions,
)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

HISTORY_PATH = LOG_DIR / "paper_monitor_history.json"
REPORT_PATH = ROOT / "reports" / "paper_trading_live.html"

# Paper trading validation window (from MASTERPLAN)
PAPER_START = "2026-03-15"
PAPER_END = "2026-05-11"

# Known paper trading accounts (reference only — real account IDs come from API)
TRACKED_EXPERIMENTS = ["400", "401", "503", "600", "305"]

STARTING_CAPITAL = 100_000.0
TRADING_DAYS_PER_YEAR = 252

logger = logging.getLogger("paper_monitor")


def setup_logging():
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False


# ═══════════════════════════════════════════════════════════════════════════
# Alpaca data fetch
# ═══════════════════════════════════════════════════════════════════════════

def fetch_portfolio_history(key: str, secret: str, period: str = "1M",
                             timeframe: str = "1D") -> Optional[Dict]:
    """Fetch account equity history from Alpaca.

    period: 1D|7D|1M|3M|1A|all
    timeframe: 1Min|5Min|15Min|1H|1D
    """
    try:
        resp = requests.get(
            f"{BASE_URL}/v2/account/portfolio/history",
            headers=_headers(key, secret),
            params={"period": period, "timeframe": timeframe,
                    "extended_hours": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"portfolio/history fetch failed: {e}")
        return None


def fetch_orders(key: str, secret: str, since: str,
                  limit: int = 500) -> List[Dict]:
    """Fetch orders since a given date."""
    try:
        resp = requests.get(
            f"{BASE_URL}/v2/orders",
            headers=_headers(key, secret),
            params={"status": "all", "after": since,
                    "limit": limit, "direction": "desc"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"orders fetch failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Metrics computation (arithmetic, Rule Zero compliant)
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(equity_series: List[float],
                     starting_capital: float = STARTING_CAPITAL) -> Dict:
    """Compute running Sharpe, DD, CAGR, vol from a real equity series.

    Uses arithmetic mean / std * sqrt(252) for Sharpe — not the inflated
    CAGR-based formula we caught in commit 1f0888a.
    """
    if not equity_series or len(equity_series) < 2:
        return {
            "n_days": len(equity_series), "total_return_pct": 0.0,
            "cagr_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0,
            "vol_pct": 0.0, "current_equity": starting_capital,
            "peak_equity": starting_capital,
        }

    eq = [float(e) for e in equity_series]
    current = eq[-1]
    peak = max(eq)

    # Daily returns
    daily_returns = []
    for i in range(1, len(eq)):
        if eq[i - 1] > 0:
            daily_returns.append((eq[i] - eq[i - 1]) / eq[i - 1])
        else:
            daily_returns.append(0.0)

    total_return = (current - starting_capital) / starting_capital
    n_days = len(eq)
    n_years = max(n_days / TRADING_DAYS_PER_YEAR, 1.0 / TRADING_DAYS_PER_YEAR)

    # CAGR
    if current > 0 and starting_capital > 0:
        cagr = (current / starting_capital) ** (1.0 / n_years) - 1.0
    else:
        cagr = -1.0

    # Arithmetic Sharpe (no CAGR inflation)
    if len(daily_returns) > 1:
        mean = sum(daily_returns) / len(daily_returns)
        var = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std = math.sqrt(var)
        sharpe = (mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)) if std > 1e-9 else 0.0
        vol = std * math.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0.0
        vol = 0.0

    # Max drawdown
    max_dd = 0.0
    running_peak = eq[0]
    for v in eq:
        running_peak = max(running_peak, v)
        dd = (running_peak - v) / running_peak if running_peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return {
        "n_days": n_days,
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "current_equity": round(current, 2),
        "peak_equity": round(peak, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Snapshot: query all accounts, compute metrics, return structured data
# ═══════════════════════════════════════════════════════════════════════════

def snapshot_all_accounts() -> Dict:
    """Query every discovered .env.expNNN and build a structured snapshot.

    Returns dict keyed by experiment name with account/positions/metrics/orders.
    """
    accounts = _discover_accounts()
    logger.info(f"Discovered {len(accounts)} credential files")

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "paper_start": PAPER_START,
        "paper_end": PAPER_END,
        "days_since_start": (date.today() - datetime.strptime(PAPER_START, "%Y-%m-%d").date()).days,
        "experiments": {},
    }

    for exp_name, creds in accounts.items():
        key = creds.get("api_key") or creds.get("key")
        secret = creds.get("api_secret") or creds.get("secret")
        if not key or not secret:
            logger.warning(f"{exp_name}: missing credentials")
            snapshot["experiments"][exp_name] = {
                "status": "no_credentials",
                "error": "ALPACA_API_KEY/SECRET missing in .env file",
            }
            continue

        try:
            acct = _fetch_account(key, secret)
            positions = _fetch_positions(key, secret)

            # Portfolio history for Sharpe/DD/vol
            # Use 3M to get a decent sample across the paper window
            hist = fetch_portfolio_history(key, secret, period="3M", timeframe="1D")
            equity_series = []
            if hist and hist.get("equity"):
                equity_series = [float(e) for e in hist["equity"] if e is not None]

            metrics = compute_metrics(equity_series)

            # Recent orders (since paper start)
            orders = fetch_orders(key, secret, since=f"{PAPER_START}T00:00:00Z")

            # Classify orders
            n_filled = sum(1 for o in orders if o.get("status") == "filled")
            n_open = sum(1 for o in orders if o.get("status") == "new")
            n_cancelled = sum(1 for o in orders if o.get("status") == "canceled")

            total_unrealized = sum(float(p.get("unrealized_pl") or 0) for p in positions)

            snapshot["experiments"][exp_name] = {
                "status": "ok",
                "account": {
                    "id": acct.get("account_number") or acct.get("id") or "unknown",
                    "equity": float(acct.get("equity", 0)),
                    "cash": float(acct.get("cash", 0)),
                    "buying_power": float(acct.get("buying_power", 0)),
                    "account_status": acct.get("status", "unknown"),
                },
                "positions": {
                    "count": len(positions),
                    "total_unrealized_pl": round(total_unrealized, 2),
                    "details": [
                        {
                            "symbol": p.get("symbol", ""),
                            "qty": float(p.get("qty", 0)),
                            "avg_entry": float(p.get("avg_entry_price") or 0),
                            "market_value": float(p.get("market_value") or 0),
                            "unrealized_pl": float(p.get("unrealized_pl") or 0),
                            "asset_class": p.get("asset_class", ""),
                        }
                        for p in positions[:20]
                    ],
                },
                "orders_since_start": {
                    "total": len(orders),
                    "filled": n_filled,
                    "open": n_open,
                    "cancelled": n_cancelled,
                },
                "metrics": metrics,
                "equity_series": equity_series,  # raw for charts
            }

            logger.info(
                f"{exp_name}: equity ${float(acct.get('equity', 0)):,.2f}, "
                f"positions {len(positions)}, "
                f"return {metrics['total_return_pct']:+.2f}%, "
                f"Sharpe {metrics['sharpe']:.2f}, "
                f"DD {metrics['max_dd_pct']:.2f}%"
            )

        except Exception as e:
            logger.error(f"{exp_name}: fetch failed: {e}")
            snapshot["experiments"][exp_name] = {
                "status": "error",
                "error": str(e)[:200],
            }

    return snapshot


# ═══════════════════════════════════════════════════════════════════════════
# History persistence (append daily snapshots)
# ═══════════════════════════════════════════════════════════════════════════

def append_history(snapshot: Dict) -> List[Dict]:
    """Persist snapshot to logs/paper_monitor_history.json. Returns full history."""
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text())
    else:
        history = []

    # Keep one snapshot per day (replace same-day entries)
    today_str = date.today().isoformat()
    history = [h for h in history if h.get("date") != today_str]

    # Store lightweight snapshot (drop equity series + position details to keep size down)
    lightweight = {
        "date": today_str,
        "timestamp": snapshot["timestamp"],
        "experiments": {},
    }
    for exp, data in snapshot["experiments"].items():
        if data.get("status") != "ok":
            lightweight["experiments"][exp] = {"status": data.get("status")}
            continue
        lightweight["experiments"][exp] = {
            "status": "ok",
            "equity": data["account"]["equity"],
            "positions": data["positions"]["count"],
            "total_unrealized_pl": data["positions"]["total_unrealized_pl"],
            "orders_filled": data["orders_since_start"]["filled"],
            "metrics": data["metrics"],
        }

    history.append(lightweight)
    HISTORY_PATH.write_text(json.dumps(history, indent=2, default=str))
    return history


# ═══════════════════════════════════════════════════════════════════════════
# HTML dashboard
# ═══════════════════════════════════════════════════════════════════════════

def _color(v: float, good_above: float = 0, warn_below: float = 0) -> str:
    if v > good_above:
        return "#059669"
    if v < warn_below:
        return "#dc2626"
    return "#d97706"


def _build_summary_card(exp_name: str, data: Dict) -> str:
    if data.get("status") != "ok":
        return (
            f'<div class="exp-card error">'
            f'<div class="exp-header"><strong>EXP-{exp_name.replace("exp","")}</strong>'
            f' <span class="badge badge-red">{data.get("status", "error").upper()}</span></div>'
            f'<div class="exp-body"><div class="err-msg">'
            f'{data.get("error", "no data")}</div></div></div>'
        )

    m = data["metrics"]
    acct = data["account"]
    pos = data["positions"]
    ord_sum = data["orders_since_start"]

    ret_c = _color(m["total_return_pct"])
    sharpe_c = _color(m["sharpe"], good_above=1.0, warn_below=0)
    dd_c = "#dc2626" if m["max_dd_pct"] > 5 else ("#d97706" if m["max_dd_pct"] > 2 else "#059669")

    return f"""
    <div class="exp-card">
      <div class="exp-header">
        <strong>EXP-{exp_name.replace("exp","")}</strong>
        <span class="badge badge-green">LIVE</span>
      </div>
      <div class="exp-body">
        <div class="equity">${acct['equity']:,.2f}</div>
        <div class="return" style="color:{ret_c}">{m['total_return_pct']:+.2f}%</div>
        <div class="metrics-row">
          <div><span class="l">Sharpe</span><span class="v" style="color:{sharpe_c}">{m['sharpe']:.2f}</span></div>
          <div><span class="l">Max DD</span><span class="v" style="color:{dd_c}">{m['max_dd_pct']:.2f}%</span></div>
          <div><span class="l">Vol</span><span class="v">{m['vol_pct']:.2f}%</span></div>
        </div>
        <div class="metrics-row">
          <div><span class="l">Positions</span><span class="v">{pos['count']}</span></div>
          <div><span class="l">Unrealized</span><span class="v" style="color:{_color(pos['total_unrealized_pl'])}">${pos['total_unrealized_pl']:+,.0f}</span></div>
          <div><span class="l">Cash</span><span class="v">${acct['cash']:,.0f}</span></div>
        </div>
        <div class="orders-row">
          Orders since {PAPER_START}:
          <strong>{ord_sum['filled']}</strong> filled,
          {ord_sum['open']} open, {ord_sum['cancelled']} cancelled
        </div>
        <div class="account-id">Account: <code>{acct['id']}</code></div>
      </div>
    </div>
    """


def _build_positions_table(exp_name: str, data: Dict) -> str:
    if data.get("status") != "ok":
        return ""
    positions = data["positions"]["details"]
    if not positions:
        return f'<h4>EXP-{exp_name.replace("exp","")}</h4><p class="note">No open positions.</p>'

    rows = ""
    for p in positions:
        upl = p["unrealized_pl"]
        c = "#059669" if upl > 0 else "#dc2626"
        rows += (
            f'<tr><td><code>{p["symbol"]}</code></td>'
            f'<td class="r">{p["qty"]:+g}</td>'
            f'<td class="r">${p["avg_entry"]:.2f}</td>'
            f'<td class="r">${p["market_value"]:,.2f}</td>'
            f'<td class="r" style="color:{c}">${upl:+,.2f}</td>'
            f'<td style="font-size:.72rem;color:#64748b">{p["asset_class"]}</td></tr>\n'
        )

    return (
        f'<h4>EXP-{exp_name.replace("exp","")} Positions</h4>'
        f'<table class="positions"><thead><tr>'
        f'<th>Symbol</th><th class="r">Qty</th><th class="r">Avg Entry</th>'
        f'<th class="r">Market Value</th><th class="r">Unrealized P&L</th><th>Class</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _build_equity_chart_data(data: Dict, max_points: int = 60) -> str:
    """Return SVG polyline for equity curve."""
    if data.get("status") != "ok":
        return ""
    series = data.get("equity_series", [])
    if len(series) < 2:
        return ""

    # Downsample if too many points
    if len(series) > max_points:
        step = len(series) // max_points
        series = series[::step]

    w, h = 280, 60
    min_v = min(series)
    max_v = max(series)
    rng = max_v - min_v if max_v > min_v else 1

    points = []
    for i, v in enumerate(series):
        x = i * w / (len(series) - 1)
        y = h - ((v - min_v) / rng * h)
        points.append(f"{x:.1f},{y:.1f}")

    color = "#059669" if series[-1] > series[0] else "#dc2626"
    return (
        f'<svg viewBox="0 0 {w} {h}" class="sparkline">'
        f'<polyline points="{" ".join(points)}" '
        f'fill="none" stroke="{color}" stroke-width="2"/></svg>'
    )


def build_dashboard(snapshot: Dict, history: List[Dict]) -> str:
    ts = snapshot["timestamp"]
    days = snapshot["days_since_start"]

    ok_exps = {k: v for k, v in snapshot["experiments"].items()
               if v.get("status") == "ok"}
    failed = {k: v for k, v in snapshot["experiments"].items()
              if v.get("status") != "ok"}

    # Portfolio totals
    total_equity = sum(v["account"]["equity"] for v in ok_exps.values())
    total_unrealized = sum(v["positions"]["total_unrealized_pl"] for v in ok_exps.values())
    total_positions = sum(v["positions"]["count"] for v in ok_exps.values())
    total_orders_filled = sum(v["orders_since_start"]["filled"] for v in ok_exps.values())
    total_starting = STARTING_CAPITAL * len(ok_exps)
    total_return_pct = ((total_equity - total_starting) / total_starting * 100) if total_starting > 0 else 0

    # Per-experiment summary cards
    cards_html = "".join(
        _build_summary_card(k, v) for k, v in sorted(snapshot["experiments"].items())
        if k.replace("exp", "") in TRACKED_EXPERIMENTS or v.get("status") == "ok"
    )

    # Per-experiment position tables
    positions_html = "".join(
        _build_positions_table(k, v) for k, v in sorted(ok_exps.items())
    )

    # Summary table
    summary_rows = ""
    for exp_name, v in sorted(ok_exps.items()):
        m = v["metrics"]
        ret_c = _color(m["total_return_pct"])
        sharpe_c = _color(m["sharpe"], good_above=1.0)
        summary_rows += (
            f'<tr><td><strong>EXP-{exp_name.replace("exp","")}</strong></td>'
            f'<td class="r">${v["account"]["equity"]:,.2f}</td>'
            f'<td class="r" style="color:{ret_c}">{m["total_return_pct"]:+.2f}%</td>'
            f'<td class="r">{m["cagr_pct"]:+.2f}%</td>'
            f'<td class="r" style="color:{sharpe_c}">{m["sharpe"]:.2f}</td>'
            f'<td class="r">{m["max_dd_pct"]:.2f}%</td>'
            f'<td class="r">{m["vol_pct"]:.2f}%</td>'
            f'<td class="r">{v["positions"]["count"]}</td>'
            f'<td class="r">{v["orders_since_start"]["filled"]}</td></tr>\n'
        )

    # History trend table (last 7 snapshots)
    history_rows = ""
    if len(history) > 1:
        recent = history[-7:]
        header_exps = sorted({
            exp for h in recent for exp, d in h["experiments"].items()
            if d.get("status") == "ok"
        })
        header_cells = "".join(f'<th class="r">EXP-{e.replace("exp","")}</th>'
                                for e in header_exps)
        history_rows = f'<tr><th>Date</th>{header_cells}</tr>\n'
        for h in recent:
            d = h.get("date", h.get("timestamp", "")[:10])
            cells = f'<td>{d}</td>'
            for exp in header_exps:
                data = h["experiments"].get(exp, {})
                if data.get("status") == "ok":
                    ret = data["metrics"]["total_return_pct"]
                    c = _color(ret)
                    cells += f'<td class="r" style="color:{c}">{ret:+.2f}%</td>'
                else:
                    cells += '<td class="r">—</td>'
            history_rows += f'<tr>{cells}</tr>\n'

    # Validation window progress
    paper_start_dt = datetime.strptime(PAPER_START, "%Y-%m-%d").date()
    paper_end_dt = datetime.strptime(PAPER_END, "%Y-%m-%d").date()
    total_days = (paper_end_dt - paper_start_dt).days
    days_elapsed = (date.today() - paper_start_dt).days
    pct_complete = min(100, max(0, days_elapsed / total_days * 100)) if total_days > 0 else 0
    days_remaining = max(0, (paper_end_dt - date.today()).days)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Paper Trading Live Monitor</title>
<style>
:root{{
  --bg:#fff;--card:#f8f9fa;--border:#e2e8f0;
  --text:#1a1a2e;--muted:#64748b;
  --green:#059669;--red:#dc2626;--blue:#2563eb;--amber:#d97706;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  font-family:'Inter','SF Pro Display',-apple-system,BlinkMacSystemFont,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.55;
  max-width:1200px;margin:0 auto;padding:28px;
}}
h1{{font-size:1.6rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:4px}}
h2{{
  font-size:1.15rem;font-weight:700;margin:36px 0 12px;
  padding-bottom:6px;border-bottom:2px solid var(--border);
}}
h4{{font-size:.95rem;font-weight:600;margin:18px 0 6px;color:#374151}}
.subtitle{{color:var(--muted);font-size:.86rem;margin-bottom:20px}}

.hero{{
  background:linear-gradient(135deg,#eff6ff,#dbeafe);
  border:1px solid var(--blue);border-radius:12px;
  padding:22px 26px;margin:16px 0;
}}
.hero .totals{{display:flex;gap:32px;flex-wrap:wrap}}
.hero .stat .l{{color:#1e40af;font-size:.74rem;text-transform:uppercase;letter-spacing:.04em}}
.hero .stat .v{{font-weight:800;font-size:1.6rem;color:#0c4a6e;letter-spacing:-0.02em}}

.progress{{
  background:#f1f5f9;border-radius:6px;height:10px;margin:12px 0 4px;overflow:hidden;
}}
.progress-bar{{
  height:100%;background:linear-gradient(90deg,#059669,#10b981);
}}

.exp-grid{{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:14px;margin:16px 0;
}}
.exp-card{{
  background:#fff;border:1px solid var(--border);border-radius:10px;
  padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.04);
}}
.exp-card.error{{border-color:#fecaca;background:#fef2f2}}
.exp-header{{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:10px;font-size:1rem;
}}
.badge{{padding:3px 8px;border-radius:4px;font-size:.68rem;font-weight:700}}
.badge-green{{background:#d1fae5;color:#065f46}}
.badge-red{{background:#fee2e2;color:#991b1b}}
.equity{{font-size:1.5rem;font-weight:800;letter-spacing:-0.02em}}
.return{{font-size:1rem;font-weight:600;margin-bottom:8px}}
.metrics-row{{
  display:flex;justify-content:space-between;margin:6px 0;
  font-size:.78rem;
}}
.metrics-row .l{{color:var(--muted);display:block;font-size:.68rem;text-transform:uppercase}}
.metrics-row .v{{font-weight:600;display:block;font-size:.88rem;margin-top:1px}}
.orders-row{{
  font-size:.76rem;color:var(--muted);margin-top:8px;
  padding-top:8px;border-top:1px solid var(--border);
}}
.account-id{{font-size:.68rem;color:var(--muted);margin-top:4px}}
.err-msg{{color:#991b1b;font-size:.8rem;font-family:monospace}}

table{{
  width:100%;border-collapse:collapse;margin:12px 0;font-size:.84rem;
  background:#fff;border:1px solid var(--border);border-radius:6px;overflow:hidden;
}}
th{{
  background:#f1f5f9;color:var(--muted);padding:8px 10px;text-align:left;
  border-bottom:2px solid var(--border);font-size:.7rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.03em;
}}
td{{padding:7px 10px;border-bottom:1px solid #f1f5f9;text-align:left}}
.r{{text-align:right}}
tr:hover td{{background:#fafafa}}
tr:last-child td{{border-bottom:none}}
code{{background:#f1f5f9;padding:2px 5px;border-radius:3px;font-size:.82em}}
.note{{color:var(--muted);font-size:.82rem;font-style:italic;margin:6px 0}}

.footer{{
  text-align:center;color:var(--muted);margin-top:40px;
  padding-top:16px;border-top:1px solid var(--border);font-size:.76rem;
}}
.rule-zero{{
  background:#f0fdf4;border-left:4px solid var(--green);
  padding:12px 16px;margin:20px 0;border-radius:6px;font-size:.82rem;
}}
.rule-zero strong{{color:#065f46}}
</style></head><body>

<h1>Paper Trading Live Monitor</h1>
<p class="subtitle">Real Alpaca paper account data &bull; {ts} &bull; {days} days since launch</p>

<!-- Hero totals -->
<div class="hero">
<div class="totals">
  <div class="stat">
    <div class="l">Total Equity</div>
    <div class="v">${total_equity:,.2f}</div>
  </div>
  <div class="stat">
    <div class="l">Total Return</div>
    <div class="v" style="color:{_color(total_return_pct)}">{total_return_pct:+.2f}%</div>
  </div>
  <div class="stat">
    <div class="l">Unrealized P&amp;L</div>
    <div class="v" style="color:{_color(total_unrealized)}">${total_unrealized:+,.2f}</div>
  </div>
  <div class="stat">
    <div class="l">Open Positions</div>
    <div class="v">{total_positions}</div>
  </div>
  <div class="stat">
    <div class="l">Orders Filled</div>
    <div class="v">{total_orders_filled}</div>
  </div>
  <div class="stat">
    <div class="l">Active Accounts</div>
    <div class="v">{len(ok_exps)}</div>
  </div>
</div>

<!-- Validation progress bar -->
<div style="margin-top:16px">
<div style="display:flex;justify-content:space-between;font-size:.78rem;color:#1e40af">
  <span>Paper Validation Window</span>
  <span>{days_elapsed} / {total_days} days &bull; {days_remaining} remaining</span>
</div>
<div class="progress"><div class="progress-bar" style="width:{pct_complete:.1f}%"></div></div>
<div style="font-size:.72rem;color:#64748b">{PAPER_START} → {PAPER_END}</div>
</div>
</div>

<!-- Experiment cards -->
<h2>Per-Experiment Status</h2>
<div class="exp-grid">
{cards_html}
</div>

<!-- Summary table -->
<h2>Summary Metrics</h2>
<table>
<thead><tr>
  <th>Experiment</th>
  <th class="r">Equity</th>
  <th class="r">Return</th>
  <th class="r">CAGR</th>
  <th class="r">Sharpe</th>
  <th class="r">Max DD</th>
  <th class="r">Vol</th>
  <th class="r">Positions</th>
  <th class="r">Fills</th>
</tr></thead>
<tbody>{summary_rows}</tbody>
</table>

<!-- History trend -->
{'<h2>Recent Daily Trend</h2><table>' + history_rows + '</table>' if history_rows else ''}

<!-- Open positions -->
<h2>Open Positions Detail</h2>
{positions_html if positions_html else '<p class="note">No open positions across any account.</p>'}

<!-- Rule Zero -->
<div class="rule-zero">
<strong>Rule Zero:</strong> Every number in this dashboard is fetched live from
Alpaca REST API or computed arithmetically from real equity history. No synthetic
data. Sharpe uses arithmetic mean / std × √252 (not CAGR-based). Source: live API
{BASE_URL}.
</div>

{'<h2>Failed Accounts</h2><ul>' + ''.join(f"<li><code>{k}</code>: {v.get('error', 'unknown')}</li>" for k, v in failed.items()) + '</ul>' if failed else ''}

<div class="footer">
Live Paper Trading Monitor &bull; scripts/monitor_paper_trading.py &bull;
Refreshed {ts} &bull; Next run: next cron trigger (typically daily 16:05 ET)
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Live paper trading monitor")
    parser.add_argument("--html-only", action="store_true",
                        help="Regenerate HTML from existing history without fetching")
    parser.add_argument("--json", action="store_true",
                        help="Print JSON snapshot to stdout")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write files")
    args = parser.parse_args()

    if args.html_only:
        if not HISTORY_PATH.exists():
            logger.error("No history file — run without --html-only first")
            sys.exit(1)
        history = json.loads(HISTORY_PATH.read_text())
        if not history:
            logger.error("Empty history")
            sys.exit(1)
        latest = history[-1]
        # Reconstruct a snapshot-like dict from history (no equity series)
        snapshot = {
            "timestamp": latest.get("timestamp", datetime.now().isoformat()),
            "paper_start": PAPER_START, "paper_end": PAPER_END,
            "days_since_start": (date.today() - datetime.strptime(PAPER_START, "%Y-%m-%d").date()).days,
            "experiments": {},
        }
        for exp, d in latest["experiments"].items():
            if d.get("status") != "ok":
                snapshot["experiments"][exp] = d
                continue
            snapshot["experiments"][exp] = {
                "status": "ok",
                "account": {
                    "id": "cached",
                    "equity": d["equity"],
                    "cash": 0,
                    "buying_power": 0,
                    "account_status": "cached",
                },
                "positions": {
                    "count": d["positions"],
                    "total_unrealized_pl": d["total_unrealized_pl"],
                    "details": [],
                },
                "orders_since_start": {
                    "total": d["orders_filled"],
                    "filled": d["orders_filled"],
                    "open": 0,
                    "cancelled": 0,
                },
                "metrics": d["metrics"],
                "equity_series": [],
            }
        html = build_dashboard(snapshot, history)
    else:
        snapshot = snapshot_all_accounts()

        if args.json:
            print(json.dumps(snapshot, indent=2, default=str))
            return 0

        if not args.dry_run:
            history = append_history(snapshot)
        else:
            history = []
            if HISTORY_PATH.exists():
                history = json.loads(HISTORY_PATH.read_text())

        html = build_dashboard(snapshot, history)

    if args.dry_run:
        print(f"[dry-run] Would write {REPORT_PATH}")
        logger.info("Dry run — no files written")
        return 0

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    logger.info(f"Report: {REPORT_PATH}")

    # Summary to stdout
    ok_count = sum(1 for v in snapshot["experiments"].values() if v.get("status") == "ok")
    total = len(snapshot["experiments"])
    logger.info(f"Snapshot complete: {ok_count}/{total} accounts OK")
    for exp, v in sorted(snapshot["experiments"].items()):
        if v.get("status") == "ok":
            m = v["metrics"]
            logger.info(
                f"  EXP-{exp.replace('exp','')}: "
                f"equity ${v['account']['equity']:,.2f} "
                f"({m['total_return_pct']:+.2f}%), "
                f"Sharpe {m['sharpe']:.2f}, DD {m['max_dd_pct']:.2f}%, "
                f"{v['positions']['count']} positions"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
