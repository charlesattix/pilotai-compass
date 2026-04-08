#!/usr/bin/env python3
"""
Paper Trading Weekly Report — PilotAI Credit Spreads
=====================================================
Generates a side-by-side HTML comparison of all live paper trading experiments.

Usage:
    python scripts/paper_trading_report.py
    python scripts/paper_trading_report.py --output output/paper_trading_report.html
    python scripts/paper_trading_report.py --date 2026-03-22

Output: output/paper_trading_report.html
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "registry.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "paper_trading_report.html"

# Backtest expectations pulled from MASTERPLAN / registry notes
BACKTEST_EXPECTATIONS = {
    "EXP-400": {"avg_return": 32.7, "max_dd": -12.1, "robust": 0.870, "years": 6},
    "EXP-401": {"avg_return": 40.7, "max_dd": -7.0,  "robust": None,  "years": 6},
    "EXP-503": {"avg_return": None,  "max_dd": None,  "robust": None,  "years": None},
    "EXP-600": {"avg_return": 139.2, "max_dd": -19.4, "robust": 0.950, "years": None},
}

# Victory conditions (per MASTERPLAN)
VICTORY_WIN_RATE    = 70.0   # >70%
VICTORY_MAX_DD      = 20.0   # <20% of account
STARTING_EQUITY     = 100_000.0

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def get_live_experiments(registry: dict) -> list[dict]:
    """Return experiments with status=paper_trading, sorted by ID."""
    exps = [
        exp for exp in registry["experiments"].values()
        if exp.get("status") == "paper_trading"
    ]
    return sorted(exps, key=lambda e: e["id"])


# ---------------------------------------------------------------------------
# DB resolution
# ---------------------------------------------------------------------------

def _resolve_db_path(exp: dict) -> Optional[Path]:
    """
    Try multiple paths to find the live SQLite DB for an experiment.
    Priority:
      1. paper_config yaml → db_path
      2. data/expNNN/pilotai_expNNN.db  (newer convention)
      3. data/pilotai_expNNN.db         (older convention)
    Returns Path if DB exists AND has a trades table, else None.
    """
    candidates = []

    # 1. From yaml config
    paper_cfg = exp.get("paper_config")
    if paper_cfg:
        try:
            import yaml
            cfg_file = PROJECT_ROOT / paper_cfg
            if cfg_file.exists():
                with open(cfg_file) as f:
                    cfg = yaml.safe_load(f)
                db_from_yaml = cfg.get("db_path", "")
                if db_from_yaml:
                    candidates.append(PROJECT_ROOT / db_from_yaml)
        except Exception:
            pass

    # 2/3. Derived from experiment ID
    num = exp["id"].replace("EXP-", "").lower()
    candidates += [
        PROJECT_ROOT / f"data/exp{num}/pilotai_exp{num}.db",
        PROJECT_ROOT / f"data/pilotai_exp{num}.db",
    ]

    first_existing = None
    for p in candidates:
        if not p.exists():
            continue
        if first_existing is None:
            first_existing = p
        try:
            conn = sqlite3.connect(str(p))
            conn.execute("SELECT 1 FROM trades LIMIT 1")
            conn.close()
            return p  # has a trades table — use this one
        except sqlite3.OperationalError:
            conn.close()
            # No trades table — keep searching for a better candidate
            continue
        except Exception:
            pass

    # No candidate has a trades table; return first existing path
    return first_existing

    return None


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _week_start(ref: datetime) -> str:
    monday = ref - timedelta(days=ref.weekday())
    return monday.strftime("%Y-%m-%d")


def query_experiment(exp: dict, report_date: str) -> dict:
    """
    Query a single experiment's SQLite DB and return a stats dict.
    Handles empty DBs and missing tables gracefully.
    """
    exp_id = exp["id"]
    db_path = _resolve_db_path(exp)

    base = {
        "id":           exp_id,
        "name":         exp.get("name", exp_id),
        "ticker":       exp.get("ticker", "SPY"),
        "creator":      exp.get("created_by", "—"),
        "live_since":   exp.get("live_since", "—"),
        "account_id":   exp.get("account_id", "—"),
        "db_path":      str(db_path) if db_path else "NOT FOUND",
        "db_found":     db_path is not None,
        # Trade stats
        "total_closed": 0,
        "wins":         0,
        "losses":       0,
        "win_rate":     0.0,
        "total_pnl":    0.0,
        "max_dd":       0.0,
        "open_count":   0,
        "avg_pnl":      0.0,
        "trades_week":  0,
        "last_trade":   None,
        "strategy_breakdown": {},
        "recent_trades": [],
        "error":        None,
    }

    if not db_path:
        base["error"] = "Database not found"
        return base

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # ── Closed trades ──────────────────────────────────────────────────
        closed_statuses = ("closed_profit", "closed_loss", "closed_manual",
                           "closed_expiry", "closed_external")
        placeholders = ",".join("?" * len(closed_statuses))
        closed_rows = conn.execute(
            f"SELECT pnl, strategy_type, exit_date, entry_date, ticker, "
            f"       short_strike, long_strike, contracts, credit "
            f"FROM trades WHERE status IN ({placeholders}) ORDER BY exit_date",
            closed_statuses,
        ).fetchall()

        pnls = [float(r["pnl"] or 0) for r in closed_rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        win_rate = (len(wins) / len(pnls) * 100) if pnls else 0.0
        avg_pnl = (total_pnl / len(pnls)) if pnls else 0.0

        # Max drawdown (dollar → convert to %)
        cumulative = 0.0
        peak = 0.0
        max_dd_dollars = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd_dollars:
                max_dd_dollars = dd
        max_dd_pct = (max_dd_dollars / STARTING_EQUITY * 100) if max_dd_dollars else 0.0

        # Trades this week
        ref_dt = datetime.strptime(report_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        week_start_str = _week_start(ref_dt)
        trades_week = sum(
            1 for r in closed_rows
            if str(r["exit_date"] or "")[:10] >= week_start_str
        )

        # Last trade date
        last_trade = None
        if closed_rows:
            last_trade = str(closed_rows[-1]["exit_date"] or "")[:10]

        # Strategy breakdown
        strategy_breakdown: dict[str, dict] = {}
        for r in closed_rows:
            st = (r["strategy_type"] or "unknown").replace("_", " ").title()
            p = float(r["pnl"] or 0)
            if st not in strategy_breakdown:
                strategy_breakdown[st] = {"count": 0, "wins": 0, "pnl": 0.0}
            strategy_breakdown[st]["count"] += 1
            if p > 0:
                strategy_breakdown[st]["wins"] += 1
            strategy_breakdown[st]["pnl"] += p

        # ── Open positions ─────────────────────────────────────────────────
        open_rows = conn.execute(
            "SELECT ticker, strategy_type, entry_date, expiration, "
            "       short_strike, long_strike, contracts, credit "
            "FROM trades WHERE status = 'open'"
        ).fetchall()

        # ── Recent closed trades (last 10) ─────────────────────────────────
        recent = conn.execute(
            f"SELECT pnl, strategy_type, exit_date, entry_date, ticker, "
            f"       short_strike, long_strike, contracts, credit, exit_reason "
            f"FROM trades WHERE status IN ({placeholders}) "
            f"ORDER BY exit_date DESC LIMIT 10",
            closed_statuses,
        ).fetchall()

        conn.close()

        base.update({
            "total_closed": len(pnls),
            "wins":         len(wins),
            "losses":       len(losses),
            "win_rate":     win_rate,
            "total_pnl":    total_pnl,
            "max_dd":       max_dd_pct,
            "open_count":   len(open_rows),
            "avg_pnl":      avg_pnl,
            "trades_week":  trades_week,
            "last_trade":   last_trade,
            "strategy_breakdown": strategy_breakdown,
            "recent_trades": [dict(r) for r in recent],
            "open_trades":  [dict(r) for r in open_rows],
        })

    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            base["error"] = "No trades yet (DB initialized, awaiting first trade)"
        else:
            base["error"] = str(e)
    except Exception as e:
        base["error"] = str(e)

    return base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pnl(val: float, prefix: str = "$") -> str:
    sign = "+" if val > 0 else ""
    return f"{sign}{prefix}{abs(val):,.2f}" if val < 0 else f"{sign}{prefix}{val:,.2f}"


def _fmt_pct(val: float) -> str:
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f}%"


def _pnl_cls(val: float) -> str:
    if val > 0: return "up"
    if val < 0: return "down"
    return "neutral"


def _days_live(live_since: str) -> str:
    try:
        d = datetime.strptime(live_since[:10], "%Y-%m-%d")
        delta = (datetime.utcnow() - d).days
        return f"{delta}d"
    except Exception:
        return "—"


def _wr_badge(win_rate: float, trade_count: int) -> str:
    """Color-coded win rate badge."""
    if trade_count == 0:
        return '<span class="badge badge-gray">No trades</span>'
    if win_rate >= VICTORY_WIN_RATE:
        return f'<span class="badge badge-green">{win_rate:.1f}%</span>'
    if win_rate >= 50:
        return f'<span class="badge badge-yellow">{win_rate:.1f}%</span>'
    return f'<span class="badge badge-red">{win_rate:.1f}%</span>'


def _dd_badge(dd_pct: float, trade_count: int) -> str:
    if trade_count == 0:
        return '<span class="badge badge-gray">—</span>'
    if dd_pct <= VICTORY_MAX_DD:
        return f'<span class="badge badge-green">-{dd_pct:.1f}%</span>'
    if dd_pct <= 30:
        return f'<span class="badge badge-yellow">-{dd_pct:.1f}%</span>'
    return f'<span class="badge badge-red">-{dd_pct:.1f}%</span>'


def _victory_status(stats: dict) -> tuple[str, str]:
    """Return (status_text, css_class) for victory condition check."""
    if stats["total_closed"] == 0:
        return "Awaiting Trades", "status-pending"
    # Check conditions: win rate >70%, DD <20%
    wr_ok = stats["win_rate"] >= VICTORY_WIN_RATE
    dd_ok = stats["max_dd"] < VICTORY_MAX_DD
    if wr_ok and dd_ok:
        return "On Track", "status-pass"
    failing = []
    if not wr_ok:
        failing.append(f"WR {stats['win_rate']:.1f}%<{VICTORY_WIN_RATE:.0f}%")
    if not dd_ok:
        failing.append(f"DD -{stats['max_dd']:.1f}%")
    return f"Watch: {', '.join(failing)}", "status-warn"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: #f8fafc;
    color: #1e293b;
    font-size: 14px;
    line-height: 1.5;
}
.page-wrapper { max-width: 1400px; margin: 0 auto; padding: 32px 24px 64px; }

/* Header */
.header { margin-bottom: 32px; }
.header h1 { font-size: 28px; font-weight: 700; color: #0f172a; letter-spacing: -0.5px; }
.header .subtitle { color: #64748b; font-size: 13px; margin-top: 6px; }
.header .subtitle b { color: #334155; }

/* Section */
.section { margin-bottom: 40px; }
.section-title {
    font-size: 13px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.8px; color: #94a3b8; margin-bottom: 16px;
    padding-bottom: 8px; border-bottom: 1px solid #e2e8f0;
}

/* Experiment card grid */
.exp-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 32px;
}
.exp-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 22px 18px;
    position: relative;
    overflow: hidden;
}
.exp-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.exp-card.color-0::before { background: #2563eb; }
.exp-card.color-1::before { background: #7c3aed; }
.exp-card.color-2::before { background: #0891b2; }
.exp-card.color-3::before { background: #d97706; }

.exp-card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }
.exp-id { font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
.exp-name { font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 2px; }
.exp-meta { font-size: 12px; color: #94a3b8; margin-top: 4px; }

.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
.metric-item { }
.metric-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; color: #94a3b8; }
.metric-value { font-size: 22px; font-weight: 700; color: #0f172a; margin-top: 1px; line-height: 1.2; }
.metric-value.up   { color: #059669; }
.metric-value.down { color: #dc2626; }
.metric-value.neutral { color: #64748b; }
.metric-sub { font-size: 11px; color: #94a3b8; margin-top: 1px; }

/* Status badges */
.badge {
    display: inline-block; padding: 2px 8px; border-radius: 20px;
    font-size: 11px; font-weight: 700;
}
.badge-green  { background: #dcfce7; color: #166534; }
.badge-yellow { background: #fef9c3; color: #854d0e; }
.badge-red    { background: #fee2e2; color: #991b1b; }
.badge-gray   { background: #f1f5f9; color: #64748b; }
.badge-blue   { background: #dbeafe; color: #1e40af; }

/* Victory status */
.status-bar {
    padding: 7px 12px; border-radius: 7px; font-size: 12px; font-weight: 600;
    display: flex; align-items: center; gap: 6px; margin-top: 14px;
}
.status-pass    { background: #dcfce7; color: #166534; }
.status-pending { background: #f1f5f9; color: #64748b; }
.status-warn    { background: #fef9c3; color: #854d0e; }

/* Comparison table */
.compare-wrap { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th {
    background: #f8fafc; color: #475569; text-align: left;
    padding: 10px 14px; font-weight: 600; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.5px; border-bottom: 1px solid #e2e8f0;
}
td {
    padding: 10px 14px; border-bottom: 1px solid #f1f5f9;
    vertical-align: middle;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8fafc; }
.td-exp { font-weight: 700; }
.td-exp .exp-tag { font-size: 10px; color: #94a3b8; display: block; }

/* Recent trades table */
.trades-wrap { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; margin-bottom: 20px; }
.trades-header {
    padding: 14px 18px 12px; border-bottom: 1px solid #e2e8f0;
    display: flex; justify-content: space-between; align-items: center;
}
.trades-header-title { font-size: 14px; font-weight: 700; color: #0f172a; }
.trades-header-sub   { font-size: 12px; color: #94a3b8; }
.pnl-up   { color: #059669; font-weight: 600; }
.pnl-down { color: #dc2626; font-weight: 600; }
.strategy-pill {
    display: inline-block; padding: 1px 7px; border-radius: 4px;
    font-size: 10px; font-weight: 600; background: #f1f5f9; color: #475569;
}
.no-data { padding: 24px 18px; color: #94a3b8; font-size: 13px; text-align: center; }

/* Open positions */
.open-pos-wrap { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; margin-bottom: 20px; }

/* Backtest comparison panel */
.bt-row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 8px; }
.bt-item { flex: 1; min-width: 120px; }
.bt-label { font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.bt-val   { font-size: 14px; font-weight: 600; color: #334155; margin-top: 1px; }

/* Progress bar */
.progress-bar-wrap { background: #f1f5f9; border-radius: 4px; height: 5px; margin-top: 6px; overflow: hidden; }
.progress-bar { height: 100%; border-radius: 4px; background: #2563eb; }

/* Footer */
.footer {
    margin-top: 48px; padding-top: 16px; border-top: 1px solid #e2e8f0;
    font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between;
}

/* Ticker badge */
.ticker-badge {
    background: #0f172a; color: #f8fafc; font-size: 11px; font-weight: 700;
    padding: 2px 8px; border-radius: 4px; letter-spacing: 0.5px;
}
.ticker-badge.ibit { background: #d97706; }

/* Divider in card */
.card-divider { border: none; border-top: 1px solid #f1f5f9; margin: 14px 0; }
"""


def _card_color_class(idx: int) -> str:
    return f"color-{idx % 4}"


def _ticker_badge(ticker: str) -> str:
    cls = "ibit" if ticker.upper() == "IBIT" else ""
    return f'<span class="ticker-badge {cls}">{ticker}</span>'


def _backtest_panel(exp_id: str) -> str:
    bt = BACKTEST_EXPECTATIONS.get(exp_id, {})
    if not bt or all(v is None for v in bt.values()):
        return '<p style="color:#94a3b8;font-size:12px;">Backtest expectations: TBD</p>'

    parts = ['<div class="bt-row">']

    avg_ret = bt.get("avg_return")
    if avg_ret is not None:
        parts.append(f'<div class="bt-item"><div class="bt-label">Avg Return / yr</div>'
                     f'<div class="bt-val up">+{avg_ret:.1f}%</div></div>')

    max_dd = bt.get("max_dd")
    if max_dd is not None:
        parts.append(f'<div class="bt-item"><div class="bt-label">Max DD</div>'
                     f'<div class="bt-val">{max_dd:.1f}%</div></div>')

    robust = bt.get("robust")
    if robust is not None:
        parts.append(f'<div class="bt-item"><div class="bt-label">Robust Score</div>'
                     f'<div class="bt-val">{robust:.3f}</div></div>')

    parts.append('</div>')
    return "\n".join(parts)


def _exp_card(stats: dict, idx: int) -> str:
    color = _card_color_class(idx)
    status_txt, status_cls = _victory_status(stats)
    days_live = _days_live(stats["live_since"])

    # P&L display
    pnl_cls = _pnl_cls(stats["total_pnl"])
    pnl_pct = stats["total_pnl"] / STARTING_EQUITY * 100 if stats["total_closed"] > 0 else 0.0
    pnl_display = _fmt_pnl(stats["total_pnl"]) if stats["total_closed"] > 0 else "—"
    pnl_pct_display = _fmt_pct(pnl_pct) if stats["total_closed"] > 0 else ""

    open_display = str(stats["open_count"])

    # Avg P&L
    avg_display = _fmt_pnl(stats["avg_pnl"]) if stats["total_closed"] > 0 else "—"
    avg_cls = _pnl_cls(stats["avg_pnl"])

    error_html = ""
    if stats.get("error"):
        error_html = f'<p style="color:#94a3b8;font-size:11px;margin-bottom:8px;">ℹ {stats["error"]}</p>'

    win_badge  = _wr_badge(stats["win_rate"], stats["total_closed"])
    dd_badge   = _dd_badge(stats["max_dd"], stats["total_closed"])

    # Week trades badge
    wk = stats["trades_week"]
    wk_badge = f'<span class="badge badge-blue">{wk} this week</span>' if wk > 0 else \
               f'<span class="badge badge-gray">0 this week</span>'

    return f"""
<div class="exp-card {color}">
  <div class="exp-card-header">
    <div>
      <div class="exp-id">{stats['id']}</div>
      <div class="exp-name">{stats['name']}</div>
      <div class="exp-meta">
        {_ticker_badge(stats['ticker'])}
        &nbsp; by <b>{stats['creator']}</b>
        &nbsp;&bull;&nbsp; Live {days_live}
        &nbsp;&bull;&nbsp; since {stats['live_since']}
      </div>
    </div>
  </div>

  {error_html}

  <div class="metric-grid">
    <div class="metric-item">
      <div class="metric-label">Total P&L</div>
      <div class="metric-value {pnl_cls}">{pnl_display}</div>
      <div class="metric-sub">{pnl_pct_display}</div>
    </div>
    <div class="metric-item">
      <div class="metric-label">Closed Trades</div>
      <div class="metric-value neutral">{stats['total_closed']}</div>
      <div class="metric-sub">{stats['wins']}W / {stats['losses']}L</div>
    </div>
    <div class="metric-item">
      <div class="metric-label">Win Rate</div>
      <div class="metric-value">{win_badge}</div>
    </div>
    <div class="metric-item">
      <div class="metric-label">Max Drawdown</div>
      <div class="metric-value">{dd_badge}</div>
    </div>
    <div class="metric-item">
      <div class="metric-label">Open Positions</div>
      <div class="metric-value neutral">{open_display}</div>
    </div>
    <div class="metric-item">
      <div class="metric-label">Avg P&L / Trade</div>
      <div class="metric-value {avg_cls}" style="font-size:16px">{avg_display}</div>
    </div>
  </div>

  <div style="margin-bottom:10px">{wk_badge}</div>

  <hr class="card-divider">

  <div style="margin-bottom:10px">
    <div class="metric-label" style="margin-bottom:6px">Backtest Expectations</div>
    {_backtest_panel(stats['id'])}
  </div>

  <div class="status-bar {status_cls}">
    {'✓' if status_cls == 'status-pass' else ('⏳' if status_cls == 'status-pending' else '⚠')}
    &nbsp;{status_txt}
  </div>
</div>
"""


def _comparison_table(all_stats: list[dict]) -> str:
    rows = []
    for s in all_stats:
        if s["total_closed"] == 0:
            pnl_html = '<span style="color:#94a3b8">—</span>'
            wr_html  = '<span style="color:#94a3b8">—</span>'
            dd_html  = '<span style="color:#94a3b8">—</span>'
            avg_html = '<span style="color:#94a3b8">—</span>'
        else:
            pnl_cls = _pnl_cls(s["total_pnl"])
            pnl_html = f'<span class="pnl-{pnl_cls if pnl_cls in ("up","down") else "up"}">{_fmt_pnl(s["total_pnl"])}</span>'
            pnl_pct = s["total_pnl"] / STARTING_EQUITY * 100
            pnl_html += f'<span style="color:#94a3b8;font-size:11px"> ({_fmt_pct(pnl_pct)})</span>'
            wr_html  = _wr_badge(s["win_rate"], s["total_closed"])
            dd_html  = _dd_badge(s["max_dd"], s["total_closed"])
            avg_cls  = _pnl_cls(s["avg_pnl"])
            avg_html = f'<span class="pnl-{avg_cls if avg_cls in ("up","down") else "up"}">{_fmt_pnl(s["avg_pnl"])}</span>'

        bt = BACKTEST_EXPECTATIONS.get(s["id"], {})
        bt_ret = f'+{bt["avg_return"]:.1f}%' if bt.get("avg_return") else "TBD"

        status_txt, status_cls = _victory_status(s)
        open_disp = str(s["open_count"])

        rows.append(f"""
        <tr>
          <td class="td-exp">
            {s['id']}<span class="exp-tag">{s['name']}</span>
          </td>
          <td>{_ticker_badge(s['ticker'])}</td>
          <td>{s['total_closed']}</td>
          <td>{pnl_html}</td>
          <td>{wr_html}</td>
          <td>{dd_html}</td>
          <td>{open_disp}</td>
          <td>{avg_html}</td>
          <td><span style="color:#64748b;font-size:12px">{s['trades_week']}</span></td>
          <td><span style="color:#64748b;font-size:12px">{bt_ret}</span></td>
          <td><span class="status-bar {status_cls}" style="padding:4px 10px;margin:0">{status_txt}</span></td>
        </tr>""")

    return f"""
<div class="compare-wrap">
<table>
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Ticker</th>
      <th>Closed</th>
      <th>Total P&L</th>
      <th>Win Rate</th>
      <th>Max DD</th>
      <th>Open</th>
      <th>Avg / Trade</th>
      <th>This Week</th>
      <th>BT Expect</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    {"".join(rows)}
  </tbody>
</table>
</div>
"""


def _recent_trades_section(all_stats: list[dict]) -> str:
    parts = []
    for s in all_stats:
        trades = s.get("recent_trades", [])
        exp_id = s["id"]
        exp_name = s["name"]

        parts.append(f'<div class="trades-wrap">')
        parts.append(f'<div class="trades-header">'
                     f'<div class="trades-header-title">{exp_id} — {exp_name}</div>'
                     f'<div class="trades-header-sub">{_ticker_badge(s["ticker"])}</div>'
                     f'</div>')

        if not trades:
            parts.append(f'<div class="no-data">No closed trades yet — {s.get("error") or "awaiting first trade"}</div>')
        else:
            parts.append('<table>')
            parts.append('<tr><th>Exit Date</th><th>Ticker</th><th>Strategy</th>'
                         '<th>Strikes</th><th>Contracts</th><th>P&L</th><th>Exit Reason</th></tr>')
            for t in trades:
                pnl = float(t.get("pnl") or 0)
                pnl_cls = "pnl-up" if pnl > 0 else "pnl-down"
                strikes = f"${t.get('short_strike', '?'):.0f}/${t.get('long_strike', '?'):.0f}"
                st = (t.get("strategy_type") or "—").replace("_", " ").title()
                reason = (t.get("exit_reason") or "—").replace("_", " ").title()
                exit_date = str(t.get("exit_date") or "—")[:10]
                contracts = t.get("contracts", 1) or 1
                parts.append(f'<tr>'
                              f'<td>{exit_date}</td>'
                              f'<td><b>{t.get("ticker", "?")}</b></td>'
                              f'<td><span class="strategy-pill">{st}</span></td>'
                              f'<td style="font-size:12px;color:#64748b">{strikes}</td>'
                              f'<td style="text-align:center">{contracts}</td>'
                              f'<td class="{pnl_cls}">{_fmt_pnl(pnl)}</td>'
                              f'<td style="font-size:12px;color:#64748b">{reason}</td>'
                              f'</tr>')
            parts.append('</table>')

        parts.append('</div>')

    return "\n".join(parts)


def _open_positions_section(all_stats: list[dict]) -> str:
    any_open = any(len(s.get("open_trades", [])) > 0 for s in all_stats)
    if not any_open:
        return '<div class="no-data" style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:24px">No open positions across any experiment.</div>'

    parts = []
    for s in all_stats:
        open_trades = s.get("open_trades", [])
        if not open_trades:
            continue
        exp_id = s["id"]
        exp_name = s["name"]

        parts.append(f'<div class="open-pos-wrap">')
        parts.append(f'<div class="trades-header">'
                     f'<div class="trades-header-title">{exp_id} — {exp_name}</div>'
                     f'<div class="trades-header-sub">{len(open_trades)} open &nbsp; {_ticker_badge(s["ticker"])}</div>'
                     f'</div>')
        parts.append('<table>')
        parts.append('<tr><th>Entry Date</th><th>Ticker</th><th>Strategy</th>'
                     '<th>Strikes</th><th>Expiration</th><th>Contracts</th><th>Credit</th></tr>')
        for t in open_trades:
            strikes = f"${t.get('short_strike', '?'):.0f}/${t.get('long_strike', '?'):.0f}"
            st = (t.get("strategy_type") or "—").replace("_", " ").title()
            entry_date = str(t.get("entry_date") or "—")[:10]
            expiration = str(t.get("expiration") or "—")[:10]
            credit = float(t.get("credit") or 0)
            contracts = t.get("contracts", 1) or 1
            parts.append(f'<tr>'
                         f'<td>{entry_date}</td>'
                         f'<td><b>{t.get("ticker", "?")}</b></td>'
                         f'<td><span class="strategy-pill">{st}</span></td>'
                         f'<td style="font-size:12px;color:#64748b">{strikes}</td>'
                         f'<td style="font-size:12px;color:#64748b">{expiration}</td>'
                         f'<td style="text-align:center">{contracts}</td>'
                         f'<td>${credit:.2f}</td>'
                         f'</tr>')
        parts.append('</table></div>')

    return "\n".join(parts) if parts else ""


def _strategy_breakdown_section(all_stats: list[dict]) -> str:
    """Small breakdown table per experiment."""
    parts = ['<div class="compare-wrap"><table>',
             '<tr><th>Experiment</th><th>Strategy</th><th>Trades</th>'
             '<th>Wins</th><th>Win Rate</th><th>Total P&L</th></tr>']

    any_rows = False
    for s in all_stats:
        bd = s.get("strategy_breakdown", {})
        if not bd:
            continue
        for st_name, st_data in sorted(bd.items()):
            wr = st_data["wins"] / st_data["count"] * 100 if st_data["count"] else 0
            pnl = st_data["pnl"]
            pnl_cls = "pnl-up" if pnl >= 0 else "pnl-down"
            parts.append(f'<tr>'
                         f'<td style="color:#64748b;font-size:12px">{s["id"]}</td>'
                         f'<td><span class="strategy-pill">{st_name}</span></td>'
                         f'<td>{st_data["count"]}</td>'
                         f'<td>{st_data["wins"]}</td>'
                         f'<td>{wr:.1f}%</td>'
                         f'<td class="{pnl_cls}">{_fmt_pnl(pnl)}</td>'
                         f'</tr>')
            any_rows = True

    parts.append('</table></div>')

    if not any_rows:
        return '<div class="no-data" style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:24px">No closed trades to break down yet.</div>'

    return "\n".join(parts)


def generate_html(all_stats: list[dict], report_date: str) -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    week_start = _week_start(datetime.strptime(report_date, "%Y-%m-%d").replace(tzinfo=timezone.utc))

    # Summary numbers
    total_exp = len(all_stats)
    exp_with_trades = sum(1 for s in all_stats if s["total_closed"] > 0)
    total_open = sum(s["open_count"] for s in all_stats)

    # Cards
    cards_html = "\n".join(_exp_card(s, i) for i, s in enumerate(all_stats))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Trading Report — {report_date}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page-wrapper">

  <!-- ── Header ── -->
  <div class="header">
    <h1>Paper Trading Report</h1>
    <p class="subtitle">
      <b>PilotAI Credit Spreads</b>
      &bull; Week of {week_start}
      &bull; {total_exp} experiments live
      &bull; {exp_with_trades} with trades
      &bull; {total_open} open positions
      &bull; Generated {now_str}
    </p>
  </div>

  <!-- ── Experiment Cards ── -->
  <div class="section">
    <div class="section-title">Live Experiments</div>
    <div class="exp-grid">
      {cards_html}
    </div>
  </div>

  <!-- ── Side-by-Side Comparison ── -->
  <div class="section">
    <div class="section-title">Side-by-Side Comparison</div>
    {_comparison_table(all_stats)}
  </div>

  <!-- ── Open Positions ── -->
  <div class="section">
    <div class="section-title">Open Positions</div>
    {_open_positions_section(all_stats)}
  </div>

  <!-- ── Strategy Breakdown ── -->
  <div class="section">
    <div class="section-title">Strategy Breakdown</div>
    {_strategy_breakdown_section(all_stats)}
  </div>

  <!-- ── Recent Trades ── -->
  <div class="section">
    <div class="section-title">Recent Closed Trades (last 10 per experiment)</div>
    {_recent_trades_section(all_stats)}
  </div>

  <!-- ── Victory Conditions ── -->
  <div class="section">
    <div class="section-title">Victory Conditions (8-week gate: Mar 16 → May 11, 2026)</div>
    <div class="compare-wrap">
      <table>
        <thead>
          <tr>
            <th>Condition</th>
            <th>Target</th>
            {"".join(f'<th>{s["id"]}</th>' for s in all_stats)}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Win Rate</td>
            <td>&gt; {VICTORY_WIN_RATE:.0f}%</td>
            {"".join(
              f'<td>{_wr_badge(s["win_rate"], s["total_closed"])}</td>'
              for s in all_stats
            )}
          </tr>
          <tr>
            <td>Max Drawdown</td>
            <td>&lt; {VICTORY_MAX_DD:.0f}%</td>
            {"".join(
              f'<td>{_dd_badge(s["max_dd"], s["total_closed"])}</td>'
              for s in all_stats
            )}
          </tr>
          <tr>
            <td>Overall</td>
            <td>On Track</td>
            {"".join(
              f'<td><span class="status-bar {_victory_status(s)[1]}" style="padding:4px 10px;margin:0">'
              f'{_victory_status(s)[0]}</span></td>'
              for s in all_stats
            )}
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── Footer ── -->
  <div class="footer">
    <span>PilotAI Credit Spreads &bull; Paper Trading Dashboard</span>
    <span>Generated {now_str}</span>
  </div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PilotAI Paper Trading Weekly Report")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Output HTML path (default: output/paper_trading_report.html)"
    )
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Report date YYYY-MM-DD (default: today UTC)"
    )
    parser.add_argument(
        "--registry", default=str(REGISTRY_PATH),
        help="Path to experiments/registry.json"
    )
    args = parser.parse_args()

    # Load registry
    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"ERROR: registry not found at {registry_path}", file=sys.stderr)
        sys.exit(1)

    with open(registry_path) as f:
        registry = json.load(f)

    live_exps = get_live_experiments(registry)
    if not live_exps:
        print("No live experiments found in registry.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(live_exps)} live experiments: {[e['id'] for e in live_exps]}")

    # Collect stats for each
    all_stats = []
    for exp in live_exps:
        print(f"  Querying {exp['id']} ({exp['name']})...")
        stats = query_experiment(exp, args.date)
        all_stats.append(stats)
        closed = stats["total_closed"]
        open_c = stats["open_count"]
        err = stats.get("error")
        if err:
            print(f"    ⚠  {err}")
        else:
            print(f"    Closed={closed}  Open={open_c}  PnL={_fmt_pnl(stats['total_pnl'])}  WR={stats['win_rate']:.1f}%")

    # Generate HTML
    html = generate_html(all_stats, args.date)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport written to: {out_path}")
    print(f"Open with: open {out_path}")


if __name__ == "__main__":
    main()
