"""
Paper trading performance tracker — READ-ONLY reporting dashboard.

Reads trade data from experiment SQLite databases and the experiment registry,
then generates a self-contained HTML dashboard comparing all active paper
trading experiments.

This module does NOT connect to any broker or place trades.  It only reads
local SQLite databases and config files.

Data sources (in priority order):
  1. SQLite databases: data/pilotai.db, data/pilotai_exp401.db, etc.
     - ``trades`` table with columns: id, ticker, strategy_type, status,
       credit, contracts, entry_date, exit_date, pnl, metadata, etc.
  2. Experiment registry: experiments/registry.json
     - Experiment metadata: name, status, ticker, live_since, config path
  3. YAML configs: configs/paper_*.yaml
     - db_path, experiment_id, account size, strategy parameters

Usage::

    from compass.paper_tracker import generate_paper_dashboard
    generate_paper_dashboard()                          # default paths
    generate_paper_dashboard(output="my_report.html")   # custom output
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "experiments" / "registry.json"
DEFAULT_OUTPUT = ROOT / "reports" / "paper_trading_dashboard.html"
DEFAULT_CAPITAL = 100_000.0

# Closed-trade status prefixes that indicate a completed trade
_CLOSED_STATUSES = {"closed_profit", "closed_loss", "closed_expiry", "closed_manual", "closed_external"}


# ── Data loading ─────────────────────────────────────────────────────────


def load_registry() -> Dict[str, Any]:
    """Load the experiment registry JSON."""
    if not REGISTRY_PATH.exists():
        logger.warning("Registry not found at %s", REGISTRY_PATH)
        return {}
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    return data.get("experiments", {})


def _resolve_db_path(experiment: Dict[str, Any]) -> Optional[Path]:
    """Resolve the SQLite database path for an experiment.

    Tries in order:
      1. Explicit db_path from YAML config
      2. data/{experiment_id}/pilotai_{id}.db
      3. data/pilotai_{id_lower}.db
      4. data/pilotai.db (default for EXP-400)
    """
    exp_id = experiment.get("id", "")
    config_path = experiment.get("paper_config")

    # Try reading db_path from YAML
    if config_path:
        yaml_path = ROOT / config_path
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path) as f:
                    cfg = yaml.safe_load(f)
                if cfg and cfg.get("db_path"):
                    candidate = ROOT / cfg["db_path"]
                    if candidate.exists():
                        return candidate
            except Exception:
                pass  # YAML parsing optional

    # Convention-based paths
    candidates = [
        ROOT / "data" / exp_id.lower() / f"pilotai_{exp_id.lower()}.db",
        ROOT / "data" / f"pilotai_{exp_id.lower().replace('-', '')}.db",
        ROOT / "data" / f"pilotai_{exp_id.lower().replace('exp-', 'exp')}.db",
    ]

    # Special cases
    if exp_id == "EXP-400":
        candidates.insert(0, ROOT / "data" / "pilotai.db")
    elif exp_id == "EXP-401":
        candidates.insert(0, ROOT / "data" / "pilotai_exp401.db")

    for c in candidates:
        if c.exists():
            return c

    return None


def load_trades(db_path: Path) -> List[Dict[str, Any]]:
    """Load all trades from a SQLite database as list of dicts."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY entry_date"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Failed to read %s: %s", db_path, exc)
        return []


# ── Trade classification ─────────────────────────────────────────────────


def classify_trades(
    trades: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Split trades into open, closed (with PnL), and unmanaged."""
    result: Dict[str, List[Dict[str, Any]]] = {
        "open": [],
        "closed": [],
        "unmanaged": [],
    }
    for t in trades:
        status = (t.get("status") or "").lower()
        if status == "open":
            result["open"].append(t)
        elif status in _CLOSED_STATUSES:
            result["closed"].append(t)
        else:
            result["unmanaged"].append(t)
    return result


# ── Metrics computation ──────────────────────────────────────────────────


def compute_metrics(
    closed_trades: List[Dict[str, Any]],
    open_trades: List[Dict[str, Any]],
    starting_capital: float = DEFAULT_CAPITAL,
) -> Dict[str, Any]:
    """Compute performance metrics from trade lists.

    Returns a dict with all dashboard metrics. Handles sparse data gracefully.
    """
    total_closed = len(closed_trades)
    total_open = len(open_trades)

    pnls = [t["pnl"] for t in closed_trades if t.get("pnl") is not None]
    credits = [t["credit"] for t in closed_trades if t.get("credit") is not None]
    open_credits = [t["credit"] for t in open_trades if t.get("credit") is not None]

    # Win rate
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    win_rate = wins / len(pnls) if pnls else None

    # P&L stats
    total_pnl = sum(pnls) if pnls else 0.0
    avg_pnl = np.mean(pnls) if pnls else None
    avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else None
    avg_loss = np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else None
    best_trade = max(pnls) if pnls else None
    worst_trade = min(pnls) if pnls else None

    # Cumulative return
    cumulative_return_pct = (total_pnl / starting_capital * 100) if pnls else None

    # Daily P&L for Sharpe / drawdown (group by exit_date)
    daily_pnl = _compute_daily_pnl(closed_trades, starting_capital)
    sharpe = _compute_sharpe(daily_pnl)
    max_dd, max_dd_pct = _compute_max_drawdown(daily_pnl, starting_capital)

    # Current drawdown (from equity high-water mark)
    current_equity = starting_capital + total_pnl
    current_dd_pct = None
    if daily_pnl:
        hwm = starting_capital
        eq = starting_capital
        for d in daily_pnl:
            eq += d
            hwm = max(hwm, eq)
        if hwm > 0:
            current_dd_pct = round((eq - hwm) / hwm * 100, 2)

    # Average credit collected
    avg_credit = np.mean(credits) if credits else None

    # Hold duration
    hold_days = []
    for t in closed_trades:
        if t.get("entry_date") and t.get("exit_date"):
            try:
                entry = datetime.fromisoformat(str(t["entry_date"]))
                exit_ = datetime.fromisoformat(str(t["exit_date"]))
                hold_days.append((exit_ - entry).days)
            except (ValueError, TypeError):
                pass
    avg_hold_days = np.mean(hold_days) if hold_days else None

    return {
        "total_trades": total_closed + total_open,
        "closed_trades": total_closed,
        "open_trades": total_open,
        "trades_with_pnl": len(pnls),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2) if avg_pnl is not None else None,
        "avg_win": round(avg_win, 2) if avg_win is not None else None,
        "avg_loss": round(avg_loss, 2) if avg_loss is not None else None,
        "best_trade": round(best_trade, 2) if best_trade is not None else None,
        "worst_trade": round(worst_trade, 2) if worst_trade is not None else None,
        "cumulative_return_pct": round(cumulative_return_pct, 2) if cumulative_return_pct is not None else None,
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd_pct, 2) if max_dd_pct is not None else None,
        "current_drawdown_pct": current_dd_pct,
        "avg_credit": round(avg_credit, 2) if avg_credit is not None else None,
        "avg_hold_days": round(avg_hold_days, 1) if avg_hold_days is not None else None,
        "starting_capital": starting_capital,
    }


def _compute_daily_pnl(
    closed_trades: List[Dict[str, Any]],
    starting_capital: float,
) -> List[float]:
    """Aggregate trade P&L by exit date to get daily return series."""
    by_date: Dict[str, float] = {}
    for t in closed_trades:
        pnl = t.get("pnl")
        exit_date = t.get("exit_date")
        if pnl is None or exit_date is None:
            continue
        date_str = str(exit_date)[:10]  # YYYY-MM-DD
        by_date[date_str] = by_date.get(date_str, 0.0) + pnl

    if not by_date:
        return []

    return [by_date[d] for d in sorted(by_date.keys())]


def _compute_sharpe(daily_pnl: List[float], annual_factor: float = 252.0) -> Optional[float]:
    """Annualized Sharpe ratio from daily P&L series."""
    if len(daily_pnl) < 2:
        return None
    arr = np.array(daily_pnl)
    if np.std(arr) == 0:
        return None
    return float(np.mean(arr) / np.std(arr) * math.sqrt(annual_factor))


def _compute_max_drawdown(
    daily_pnl: List[float],
    starting_capital: float,
) -> Tuple[float, Optional[float]]:
    """Max drawdown from daily P&L series.

    Returns (max_dd_absolute, max_dd_pct).
    """
    if not daily_pnl:
        return 0.0, None

    equity = starting_capital
    hwm = equity
    max_dd = 0.0

    for pnl in daily_pnl:
        equity += pnl
        hwm = max(hwm, equity)
        dd = equity - hwm
        if dd < max_dd:
            max_dd = dd

    max_dd_pct = (max_dd / hwm * 100) if hwm > 0 else None
    return max_dd, max_dd_pct


# ── Per-experiment data assembly ─────────────────────────────────────────


def collect_experiment_data(
    registry: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Collect trade data and metrics for all active paper trading experiments.

    Returns list of experiment dicts, each containing:
        experiment metadata + trades + computed metrics
    """
    if registry is None:
        registry = load_registry()

    experiments = []

    for exp_id, exp_meta in registry.items():
        status = exp_meta.get("status", "")
        if status != "paper_trading":
            continue

        db_path = _resolve_db_path(exp_meta)
        if db_path is None:
            logger.info("No DB found for %s — will show as no data", exp_id)
            trades = []
        else:
            trades = load_trades(db_path)
            logger.info("Loaded %d trades for %s from %s", len(trades), exp_id, db_path)

        classified = classify_trades(trades)
        metrics = compute_metrics(classified["closed"], classified["open"])

        experiments.append({
            "id": exp_id,
            "name": exp_meta.get("name", exp_id),
            "ticker": exp_meta.get("ticker", "?"),
            "live_since": exp_meta.get("live_since", "?"),
            "description": exp_meta.get("description", ""),
            "db_path": str(db_path) if db_path else None,
            "trades_total": len(trades),
            "classified": classified,
            "metrics": metrics,
        })

    # Sort by experiment ID
    experiments.sort(key=lambda e: e["id"])
    return experiments


# ── HTML report ──────────────────────────────────────────────────────────


def _fmt(val: Any, fmt_str: str = ".2f", prefix: str = "", suffix: str = "") -> str:
    """Format a value, returning '—' for None."""
    if val is None:
        return "—"
    try:
        return f"{prefix}{val:{fmt_str}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def _status_badge(metrics: Dict) -> str:
    """Generate a color-coded status badge based on available data."""
    n = metrics.get("trades_with_pnl", 0)
    if n >= 10:
        return '<span class="badge active">Active</span>'
    elif n > 0:
        return '<span class="badge early">Early</span>'
    elif metrics.get("open_trades", 0) > 0:
        return '<span class="badge pending">Pending</span>'
    else:
        return '<span class="badge waiting">Awaiting Data</span>'


def generate_html(experiments: List[Dict[str, Any]]) -> str:
    """Build the self-contained HTML dashboard."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_experiments = len(experiments)
    total_trades = sum(e["trades_total"] for e in experiments)
    total_pnl = sum(e["metrics"]["total_pnl"] for e in experiments)

    # Comparison table rows
    comparison_rows = ""
    for e in experiments:
        m = e["metrics"]
        comparison_rows += (
            f'<tr>'
            f'<td><strong>{e["id"]}</strong><br><small class="muted">{e["name"]}</small></td>'
            f'<td>{e["ticker"]}</td>'
            f'<td>{e["live_since"]}</td>'
            f'<td>{_status_badge(m)}</td>'
            f'<td>{m["total_trades"]}</td>'
            f'<td>{m["trades_with_pnl"]}</td>'
            f'<td>{_fmt(m["win_rate"], ".1%") if m["win_rate"] is not None else "—"}</td>'
            f'<td class="{"good" if m["total_pnl"] > 0 else "bad" if m["total_pnl"] < 0 else ""}">'
            f'{_fmt(m["total_pnl"], ",.2f", prefix="$")}</td>'
            f'<td>{_fmt(m["cumulative_return_pct"], ".2f", suffix="%")}</td>'
            f'<td>{_fmt(m["sharpe"], ".3f")}</td>'
            f'<td>{_fmt(m["max_drawdown_pct"], ".2f", suffix="%")}</td>'
            f'<td>{_fmt(m["current_drawdown_pct"], ".2f", suffix="%")}</td>'
            f'</tr>\n'
        )

    # Per-experiment detail sections
    detail_sections = ""
    for e in experiments:
        m = e["metrics"]
        classified = e["classified"]

        # Trade list table
        trade_rows = ""
        all_trades = classified["closed"] + classified["open"]
        # Sort by entry date
        all_trades.sort(key=lambda t: t.get("entry_date") or "")

        for t in all_trades:
            status = t.get("status", "?")
            pnl = t.get("pnl")
            pnl_cls = "good" if pnl and pnl > 0 else "bad" if pnl and pnl < 0 else ""
            entry = str(t.get("entry_date", ""))[:19]
            exit_ = str(t.get("exit_date", ""))[:19] if t.get("exit_date") else "—"
            trade_rows += (
                f'<tr>'
                f'<td><code>{str(t.get("id", ""))[:20]}</code></td>'
                f'<td>{t.get("ticker", "")}</td>'
                f'<td>{t.get("strategy_type", "—")}</td>'
                f'<td>{status}</td>'
                f'<td>{entry}</td>'
                f'<td>{exit_}</td>'
                f'<td>{_fmt(t.get("credit"), ".2f", prefix="$")}</td>'
                f'<td>{t.get("contracts", "—")}</td>'
                f'<td class="{pnl_cls}">{_fmt(pnl, ".2f", prefix="$")}</td>'
                f'<td>{t.get("exit_reason", "—")}</td>'
                f'</tr>\n'
            )

        if not trade_rows:
            trade_rows = '<tr><td colspan="10" class="empty">No trades recorded yet</td></tr>'

        detail_sections += f"""
        <div class="experiment-detail" id="{e['id']}">
          <h3>{e['id']}: {e['name']} <small class="muted">({e['ticker']})</small></h3>
          <p class="description">{e['description']}</p>
          <div class="kpi-row">
            <div class="kpi"><div class="value">{m['total_trades']}</div><div class="label">Total Trades</div></div>
            <div class="kpi"><div class="value">{m['open_trades']}</div><div class="label">Open</div></div>
            <div class="kpi"><div class="value">{_fmt(m['win_rate'], '.1%') if m['win_rate'] is not None else '—'}</div><div class="label">Win Rate</div></div>
            <div class="kpi"><div class="value {('good' if m['total_pnl'] > 0 else 'bad') if m['total_pnl'] != 0 else ''}">{_fmt(m['total_pnl'], ',.2f', prefix='$')}</div><div class="label">Total P&L</div></div>
            <div class="kpi"><div class="value">{_fmt(m['avg_pnl'], '.2f', prefix='$')}</div><div class="label">Avg P&L</div></div>
            <div class="kpi"><div class="value">{_fmt(m['sharpe'], '.3f')}</div><div class="label">Sharpe</div></div>
            <div class="kpi"><div class="value">{_fmt(m['max_drawdown_pct'], '.2f', suffix='%')}</div><div class="label">Max DD</div></div>
            <div class="kpi"><div class="value">{_fmt(m['avg_hold_days'], '.1f')}</div><div class="label">Avg Hold (d)</div></div>
          </div>
          <table class="trades-table">
            <thead>
              <tr><th>ID</th><th>Ticker</th><th>Strategy</th><th>Status</th>
              <th>Entry</th><th>Exit</th><th>Credit</th><th>Contracts</th>
              <th>P&L</th><th>Exit Reason</th></tr>
            </thead>
            <tbody>
              {trade_rows}
            </tbody>
          </table>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Paper Trading Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; margin-bottom: 0.3em; }}
  h2 {{ color: #334155; margin-top: 2.5em; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3em; }}
  h3 {{ color: #1e293b; margin-bottom: 0.3em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .muted {{ color: #94a3b8; }}
  .description {{ color: #64748b; font-size: 0.88em; margin: 0.3em 0 1em; }}
  .good {{ color: #16a34a; }}
  .bad {{ color: #dc2626; }}
  .empty {{ color: #94a3b8; text-align: center; font-style: italic; padding: 2em; }}
  .kpi-row {{ display: flex; gap: 1em; flex-wrap: wrap; margin: 1em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 0.8em 1.2em; min-width: 100px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.4em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 0.75em; font-weight: 600; }}
  .badge.active {{ background: #dcfce7; color: #166534; }}
  .badge.early {{ background: #fef3c7; color: #92400e; }}
  .badge.pending {{ background: #dbeafe; color: #1e40af; }}
  .badge.waiting {{ background: #f1f5f9; color: #64748b; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; font-size: 0.85em; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  tr:hover {{ background: #f8fafc; }}
  .trades-table {{ font-size: 0.82em; }}
  .trades-table td {{ padding: 4px 8px; }}
  code {{ font-size: 0.85em; color: #64748b; }}
  .experiment-detail {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
                        padding: 1.5em; margin: 1.5em 0; }}
  .summary-kpi {{ display: flex; gap: 1.5em; margin: 1.5em 0; }}
  .summary-kpi .kpi {{ min-width: 140px; }}
  .summary-kpi .kpi .value {{ font-size: 1.6em; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Paper Trading Dashboard</h1>
<div class="meta">
  <strong>{total_experiments}</strong> active experiments &middot;
  <strong>{total_trades}</strong> total trades &middot;
  Generated {now}
</div>

<div class="summary-kpi">
  <div class="kpi"><div class="value">{total_experiments}</div><div class="label">Active Experiments</div></div>
  <div class="kpi"><div class="value">{total_trades}</div><div class="label">Total Trades</div></div>
  <div class="kpi"><div class="value {'good' if total_pnl > 0 else 'bad' if total_pnl < 0 else ''}">${total_pnl:,.2f}</div><div class="label">Combined P&L</div></div>
</div>

<h2>Experiment Comparison</h2>
<table>
<thead>
<tr>
  <th>Experiment</th><th>Ticker</th><th>Live Since</th><th>Status</th>
  <th>Trades</th><th>With P&L</th><th>Win Rate</th><th>Total P&L</th>
  <th>Cumul Return</th><th>Sharpe</th><th>Max DD</th><th>Current DD</th>
</tr>
</thead>
<tbody>
{comparison_rows}
</tbody>
</table>

<h2>Experiment Details</h2>
{detail_sections}

<footer>
  Generated by <code>compass/paper_tracker.py</code> &middot; READ-ONLY reporting — no broker connections
</footer>

</body>
</html>"""

    return html


# ── Public API ───────────────────────────────────────────────────────────


def generate_paper_dashboard(
    output: str = str(DEFAULT_OUTPUT),
    registry_path: str = str(REGISTRY_PATH),
) -> str:
    """Generate the paper trading dashboard HTML report.

    Args:
        output: Path for the HTML report.
        registry_path: Path to experiments/registry.json.

    Returns:
        Absolute path to the generated report.
    """
    logger.info("Loading experiment registry from %s", registry_path)
    registry = load_registry()
    logger.info("Found %d experiments in registry", len(registry))

    experiments = collect_experiment_data(registry)
    logger.info("Collected data for %d active experiments", len(experiments))

    html = generate_html(experiments)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    logger.info("Dashboard written to %s (%d bytes)", out, len(html))

    return str(out.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = generate_paper_dashboard()
    print(f"Report: {path}")
