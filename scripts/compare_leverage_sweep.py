#!/usr/bin/env python3
"""
Leverage Sweep Comparison Dashboard — EXP-1220 at 2x / 3x / 4x / 5x

Reads per-instance state/journal/health files written by run_exp1220.py
(with --config) and generates a side-by-side HTML comparison.

Rule Zero: every number comes from real file state (Alpaca-backed) or
live /v2/account calls. No synthesized fills, no np.random.

Usage:
    python3 scripts/compare_leverage_sweep.py              # default: read all 4
    python3 scripts/compare_leverage_sweep.py --with-alpaca  # also fetch live account state
    python3 scripts/compare_leverage_sweep.py --json        # stdout JSON only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
REPORT_PATH = ROOT / "reports" / "leverage_sweep_comparison.html"

# 4 sweep configs + the existing 1.5x baseline for reference
SWEEP_CONFIGS = [
    {"instance": "deploy_exp1220_1.5x", "label": "1.5x (baseline)",
     "config": "configs/deploy_exp1220_1.5x.yaml"},
    {"instance": "paper_exp1220_2x",    "label": "2x",
     "config": "configs/paper_exp1220_2x.yaml"},
    {"instance": "paper_exp1220_3x",    "label": "3x",
     "config": "configs/paper_exp1220_3x.yaml"},
    {"instance": "paper_exp1220_4x",    "label": "4x",
     "config": "configs/paper_exp1220_4x.yaml"},
    {"instance": "paper_exp1220_5x",    "label": "5x",
     "config": "configs/paper_exp1220_5x.yaml"},
]

logger = logging.getLogger("leverage_sweep")


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
# Per-instance data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_instance_data(instance: str, config_rel_path: str) -> Dict:
    """Load state, health, and trade journal for one instance."""
    out = {
        "instance": instance,
        "config_path": config_rel_path,
        "config": None,
        "state": None,
        "health": None,
        "trades": [],
        "open_positions": 0,
        "closed_positions": 0,
        "last_entry": None,
    }

    # Config
    cfg_path = ROOT / config_rel_path
    if cfg_path.exists():
        try:
            out["config"] = yaml.safe_load(cfg_path.read_text())
        except Exception as e:
            logger.warning(f"{instance}: config load failed: {e}")

    # State
    state_path = LOG_DIR / f"{instance}_state.json"
    if state_path.exists():
        try:
            out["state"] = json.loads(state_path.read_text())
            positions = out["state"].get("positions", [])
            out["open_positions"] = sum(1 for p in positions if p.get("status") == "open")
            out["closed_positions"] = sum(1 for p in positions if p.get("status") == "closed")
            out["last_entry"] = out["state"].get("last_entry_date")
        except Exception as e:
            logger.warning(f"{instance}: state load failed: {e}")

    # Health
    health_path = LOG_DIR / f"{instance}_health.json"
    if health_path.exists():
        try:
            out["health"] = json.loads(health_path.read_text())
        except Exception as e:
            logger.warning(f"{instance}: health load failed: {e}")

    # Trade journal
    journal_path = LOG_DIR / f"{instance}_trade_journal.csv"
    if journal_path.exists():
        try:
            with open(journal_path) as f:
                reader = csv.DictReader(f)
                out["trades"] = list(reader)
        except Exception as e:
            logger.warning(f"{instance}: journal load failed: {e}")

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_instance_metrics(data: Dict,
                               starting_capital: float = 100_000) -> Dict:
    """Compute P&L, win rate, realized return from state + journal."""
    state = data.get("state") or {}
    positions = state.get("positions", [])

    closed = [p for p in positions if p.get("status") == "closed"]
    open_pos = [p for p in positions if p.get("status") == "open"]

    # Realized P&L (closed positions with recorded pnl)
    total_realized = 0.0
    wins = 0
    losses = 0
    for p in closed:
        pnl = p.get("pnl")
        if pnl is not None:
            total_realized += float(pnl)
            if float(pnl) > 0:
                wins += 1
            elif float(pnl) < 0:
                losses += 1

    # Unrealized P&L from open positions if price snapshots available
    total_unrealized = 0.0
    for p in open_pos:
        if p.get("current_pnl") is not None:
            total_unrealized += float(p["current_pnl"])

    total_pnl = total_realized + total_unrealized
    equity = starting_capital + total_pnl
    return_pct = total_pnl / starting_capital * 100 if starting_capital > 0 else 0.0

    win_rate = 0.0
    if wins + losses > 0:
        win_rate = wins / (wins + losses)

    return {
        "starting_capital": starting_capital,
        "equity": equity,
        "total_pnl": total_pnl,
        "realized_pnl": total_realized,
        "unrealized_pnl": total_unrealized,
        "return_pct": return_pct,
        "n_closed": len(closed),
        "n_open": len(open_pos),
        "n_trades": len(positions),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Live Alpaca fetch (optional — requires creds)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_alpaca_equity() -> Optional[Dict]:
    """Fetch current Alpaca paper account equity (if credentials available).

    Returns a single-account snapshot — all leverage configs share the
    same paper account (PA3YFVQCXTD6) in Charles's deployment setup.
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        return None

    try:
        import requests
        resp = requests.get(
            "https://paper-api.alpaca.markets/v2/account",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        a = resp.json()
        return {
            "equity": float(a.get("equity", 0)),
            "cash": float(a.get("cash", 0)),
            "buying_power": float(a.get("buying_power", 0)),
            "account_number": a.get("account_number", "unknown"),
        }
    except Exception as e:
        logger.warning(f"Alpaca fetch failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# HTML builder
# ═══════════════════════════════════════════════════════════════════════════

def _color(v: float, pos: str = "#059669", neg: str = "#dc2626",
            zero: str = "#64748b") -> str:
    if v > 0:
        return pos
    if v < 0:
        return neg
    return zero


def _fmt_money(v: float) -> str:
    sign = "+" if v > 0 else ("-" if v < 0 else "")
    return f"{sign}${abs(v):,.0f}"


def _build_row_card(data: Dict, metrics: Dict) -> str:
    instance = data["instance"]
    cfg = data.get("config") or {}
    name = cfg.get("name", instance)
    lev = cfg.get("leverage", {}).get("multiplier", "?")
    risk_pct = cfg.get("sizing", {}).get("leveraged_risk_pct", "?")
    dd_halt = cfg.get("risk", {}).get("max_drawdown_halt_pct", "?")
    max_conc = cfg.get("cadence", {}).get("max_concurrent", "?")
    vix_max = cfg.get("entry_signals", {}).get("vix_max_entry", "?")

    health = data.get("health") or {}
    status = health.get("status", "unknown")
    status_color = {
        "ok": "#059669", "warning": "#d97706",
        "error": "#dc2626", "halted": "#7f1d1d",
    }.get(status, "#64748b")
    last_run = health.get("last_run", "never")[:19] if health.get("last_run") else "never"

    ret_c = _color(metrics["return_pct"])
    pnl_c = _color(metrics["total_pnl"])

    return f"""
    <div class="instance-card">
      <div class="inst-header">
        <span class="lev-badge">{lev}x</span>
        <span class="badge" style="background:{status_color}">{status.upper()}</span>
      </div>
      <div class="inst-name">{name}</div>

      <div class="metric-rows">
        <div class="metric-row">
          <div><span class="l">Equity</span><span class="v">${metrics['equity']:,.0f}</span></div>
          <div><span class="l">Return</span><span class="v" style="color:{ret_c}">{metrics['return_pct']:+.2f}%</span></div>
        </div>
        <div class="metric-row">
          <div><span class="l">Realized P&amp;L</span><span class="v" style="color:{pnl_c}">{_fmt_money(metrics['realized_pnl'])}</span></div>
          <div><span class="l">Unrealized</span><span class="v">{_fmt_money(metrics['unrealized_pnl'])}</span></div>
        </div>
        <div class="metric-row">
          <div><span class="l">Trades</span><span class="v">{metrics['n_trades']}</span></div>
          <div><span class="l">Open</span><span class="v">{metrics['n_open']}</span></div>
        </div>
        <div class="metric-row">
          <div><span class="l">Wins / Losses</span><span class="v">{metrics['wins']} / {metrics['losses']}</span></div>
          <div><span class="l">Win Rate</span><span class="v">{metrics['win_rate']:.0%}</span></div>
        </div>
      </div>

      <div class="config-params">
        <div><strong>Risk/trade:</strong> {risk_pct}%</div>
        <div><strong>DD halt:</strong> {dd_halt}%</div>
        <div><strong>Max concurrent:</strong> {max_conc}</div>
        <div><strong>VIX max:</strong> {vix_max}</div>
      </div>

      <div class="last-run">Last run: <code>{last_run}</code></div>
    </div>
    """


def build_dashboard(loaded: List[Dict], live_account: Optional[Dict]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Compute all metrics
    all_metrics = []
    for data in loaded:
        m = compute_instance_metrics(data)
        all_metrics.append((data, m))

    cards = "".join(_build_row_card(d, m) for d, m in all_metrics)

    # Summary comparison table
    table_rows = ""
    for data, m in all_metrics:
        cfg = data.get("config") or {}
        lev = cfg.get("leverage", {}).get("multiplier", "?")
        risk = cfg.get("sizing", {}).get("leveraged_risk_pct", "?")
        dd_halt = cfg.get("risk", {}).get("max_drawdown_halt_pct", "?")
        ret_c = _color(m["return_pct"])
        pnl_c = _color(m["total_pnl"])

        table_rows += (
            f'<tr>'
            f'<td><strong>{lev}x</strong></td>'
            f'<td>{risk}%</td>'
            f'<td class="r">${m["equity"]:,.0f}</td>'
            f'<td class="r" style="color:{ret_c}">{m["return_pct"]:+.2f}%</td>'
            f'<td class="r" style="color:{pnl_c}">{_fmt_money(m["total_pnl"])}</td>'
            f'<td class="r">{m["n_trades"]}</td>'
            f'<td class="r">{m["n_open"]}</td>'
            f'<td class="r">{m["win_rate"]:.0%}</td>'
            f'<td class="r">{dd_halt}%</td>'
            f'</tr>\n'
        )

    # Live account section
    live_html = ""
    if live_account:
        live_html = f"""
    <div class="live-box">
      <strong>Live Alpaca account:</strong> {live_account['account_number']}<br/>
      <strong>Equity:</strong> ${live_account['equity']:,.2f} &bull;
      <strong>Cash:</strong> ${live_account['cash']:,.2f} &bull;
      <strong>Buying power:</strong> ${live_account['buying_power']:,.2f}
    </div>
        """

    # Projected vs realized comparison chart data
    projected = {
        "1.5x": 99.2,    # validated
        "2x": 130,       # extrapolated
        "3x": 195,       # extrapolated
        "4x": 260,       # extrapolated
        "5x": 325,       # extrapolated
    }
    proj_rows = ""
    for data, m in all_metrics:
        cfg = data.get("config") or {}
        lev_raw = cfg.get("leverage", {}).get("multiplier", 0)
        lev_key = f"{lev_raw}x" if lev_raw == int(lev_raw) else f"{lev_raw}x"
        if lev_key not in projected:
            lev_key = f"{lev_raw:g}x"
        proj = projected.get(lev_key, None)
        realized = m["return_pct"]
        delta = realized - (proj or 0) if proj else None
        delta_str = f"{delta:+.2f}%" if delta is not None else "—"
        delta_c = _color(delta) if delta is not None else "#64748b"
        proj_str = f"{proj}%" if proj is not None else "—"
        proj_rows += (
            f'<tr>'
            f'<td><strong>{lev_key}</strong></td>'
            f'<td class="r">{proj_str}</td>'
            f'<td class="r">{realized:+.2f}%</td>'
            f'<td class="r" style="color:{delta_c}">{delta_str}</td>'
            f'</tr>\n'
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>EXP-1220 Leverage Sweep Comparison</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e2e8f0;--text:#1a1a2e;--muted:#64748b;
  --green:#059669;--red:#dc2626;--blue:#2563eb;--amber:#d97706;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter','SF Pro Display',-apple-system,BlinkMacSystemFont,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.55;
  max-width:1400px;margin:0 auto;padding:28px;}}
h1{{font-size:1.65rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:4px}}
h2{{font-size:1.15rem;font-weight:700;margin:36px 0 12px;
  padding-bottom:6px;border-bottom:2px solid var(--border);}}
.subtitle{{color:var(--muted);font-size:.86rem;margin-bottom:20px}}

.sweep-grid{{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
  gap:14px;margin:16px 0;
}}
.instance-card{{
  background:#fff;border:1px solid var(--border);border-radius:10px;
  padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.04);
}}
.inst-header{{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:8px;
}}
.lev-badge{{
  font-size:1.6rem;font-weight:800;color:var(--blue);letter-spacing:-0.03em;
}}
.badge{{
  padding:3px 9px;border-radius:4px;font-size:.66rem;font-weight:700;
  color:#fff;text-transform:uppercase;letter-spacing:.04em;
}}
.inst-name{{font-size:.78rem;color:var(--muted);margin-bottom:10px}}
.metric-rows{{margin:10px 0}}
.metric-row{{
  display:flex;justify-content:space-between;margin:4px 0;
}}
.metric-row > div{{flex:1}}
.metric-row .l{{color:var(--muted);font-size:.68rem;text-transform:uppercase;display:block}}
.metric-row .v{{font-weight:600;font-size:.86rem;display:block;margin-top:1px}}

.config-params{{
  margin:10px 0;padding:8px 10px;background:#f1f5f9;border-radius:6px;
  font-size:.72rem;color:#374151;
}}
.config-params > div{{margin:2px 0}}
.last-run{{font-size:.68rem;color:var(--muted);margin-top:6px}}

table{{
  width:100%;border-collapse:collapse;margin:12px 0;font-size:.84rem;
  background:#fff;border:1px solid var(--border);border-radius:6px;overflow:hidden;
}}
th{{background:#f1f5f9;color:var(--muted);padding:8px 10px;text-align:left;
  border-bottom:2px solid var(--border);font-size:.7rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.03em;}}
td{{padding:7px 10px;border-bottom:1px solid #f1f5f9;text-align:left}}
.r{{text-align:right}}
tr:hover td{{background:#fafafa}}

.live-box{{
  background:#eff6ff;border-left:4px solid var(--blue);
  padding:12px 16px;margin:14px 0;border-radius:6px;font-size:.86rem;
}}
.rule-zero{{
  background:#f0fdf4;border-left:4px solid var(--green);
  padding:12px 16px;margin:20px 0;border-radius:6px;font-size:.82rem;
}}
.footer{{
  text-align:center;color:var(--muted);margin-top:40px;
  padding-top:16px;border-top:1px solid var(--border);font-size:.76rem;
}}
code{{background:#f1f5f9;padding:2px 5px;border-radius:3px;font-size:.82em}}
</style></head><body>

<h1>EXP-1220 Leverage Sweep Comparison</h1>
<p class="subtitle">Side-by-side tracking of 1.5x / 2x / 3x / 4x / 5x paper configs &bull;
Generated {ts}</p>

{live_html}

<h2>Per-Instance Cards</h2>
<div class="sweep-grid">{cards}</div>

<h2>Summary Comparison Table</h2>
<table>
<thead><tr>
  <th>Leverage</th><th>Risk/Trade</th>
  <th class="r">Equity</th><th class="r">Return</th><th class="r">Total P&amp;L</th>
  <th class="r">Trades</th><th class="r">Open</th><th class="r">Win Rate</th>
  <th class="r">DD Halt</th>
</tr></thead>
<tbody>{table_rows}</tbody></table>

<h2>Projected vs Realized (Validation Check)</h2>
<p class="subtitle">Projected CAGRs extrapolated linearly from 1.5x validated backtest (99.2%).
Extrapolation is unreliable at high leverage — real paper results are the point of this sweep.</p>
<table>
<thead><tr>
  <th>Leverage</th>
  <th class="r">Projected CAGR (extrap.)</th>
  <th class="r">Realized (paper)</th>
  <th class="r">Delta</th>
</tr></thead>
<tbody>{proj_rows}</tbody></table>

<div class="rule-zero">
<strong>Rule Zero:</strong> All metrics are computed from real state files
written by the runner, which itself uses Alpaca live quotes and Yahoo
Finance VIX/SPY data. No synthetic trades. No np.random. No Black-Scholes
theoretical prices. Source files:
<code>logs/{{instance}}_state.json</code>, <code>logs/{{instance}}_health.json</code>,
<code>logs/{{instance}}_trade_journal.csv</code>.
</div>

<div class="footer">
Leverage Sweep Comparison &bull; scripts/compare_leverage_sweep.py &bull;
{ts}
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Compare EXP-1220 leverage sweep")
    parser.add_argument("--with-alpaca", action="store_true",
                        help="Also fetch live Alpaca account state")
    parser.add_argument("--json", action="store_true",
                        help="Print JSON to stdout instead of writing HTML")
    args = parser.parse_args()

    logger.info("Loading instance data for all leverage configs...")
    loaded = []
    for cfg in SWEEP_CONFIGS:
        data = load_instance_data(cfg["instance"], cfg["config"])
        metrics = compute_instance_metrics(data)
        loaded.append(data)
        logger.info(
            f"  {cfg['label']:<20}"
            f"trades={data['open_positions'] + data['closed_positions']:>3} "
            f"equity=${metrics['equity']:>10,.0f} "
            f"return={metrics['return_pct']:+6.2f}%"
        )

    live_account = fetch_alpaca_equity() if args.with_alpaca else None
    if live_account:
        logger.info(f"Live Alpaca: ${live_account['equity']:,.2f}")

    if args.json:
        payload = {
            "generated": datetime.now().isoformat(),
            "live_account": live_account,
            "instances": [
                {
                    "instance": d["instance"],
                    "config_path": d["config_path"],
                    "metrics": compute_instance_metrics(d),
                    "open_positions": d["open_positions"],
                    "closed_positions": d["closed_positions"],
                    "last_entry": d["last_entry"],
                    "health_status": (d.get("health") or {}).get("status"),
                }
                for d in loaded
            ],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    html = build_dashboard(loaded, live_account)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    logger.info(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
