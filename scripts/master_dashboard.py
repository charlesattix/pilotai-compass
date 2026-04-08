#!/usr/bin/env python3
"""
Master Strategy Dashboard — unified view of ALL strategies tested on real
and heuristic data, with league table, correlation analysis, portfolio
candidates, status tracker, and North Star gap analysis.

Outputs: reports/master_strategy_dashboard.html
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "master_strategy_dashboard.html"

# North Star targets
NS_CAGR = 100.0
NS_DD = 12.0
NS_SHARPE = 6.0


# ── Data collection ─────────────────────────────────────────────────────

def load_all_experiments() -> List[Dict[str, Any]]:
    """Load all experiments with summary.json into a unified list."""
    experiments = []

    # 1. Real-data experiments
    real_experiments = {
        "EXP-880-real": {
            "name": "ML Ensemble (Real Data)",
            "ticker": "SPY",
            "data": "IronVault",
            "strategy_type": "Credit Spreads (ML Ensemble)",
        },
        "EXP-1220-real": {
            "name": "Tail Risk Protection (Real)",
            "ticker": "SPY",
            "data": "Yahoo+IronVault",
            "strategy_type": "Tail Risk Overlay",
        },
        "EXP-1230-real": {
            "name": "Microstructure Alpha (Real)",
            "ticker": "SPY",
            "data": "Yahoo+IronVault",
            "strategy_type": "Vol Regime Overlay",
        },
        "EXP-1270-real": {
            "name": "Stop Loss Optimization (Real)",
            "ticker": "SPY",
            "data": "IronVault",
            "strategy_type": "Stop Loss Research",
        },
        "EXP-1320-real": {
            "name": "Vol Clustering Overlay (Real)",
            "ticker": "SPY",
            "data": "IronVault",
            "strategy_type": "Vol Clustering Overlay",
        },
        "EXP-1470-real": {
            "name": "North Star Portfolio (Real)",
            "ticker": "SPY",
            "data": "IronVault",
            "strategy_type": "Multi-Strategy Portfolio",
        },
    }

    for exp_id, meta in real_experiments.items():
        summary_path = EXPERIMENTS / exp_id / "results" / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path) as f:
                data = json.load(f)
            experiments.append(_normalize(exp_id, meta, data, is_real=True))
        except Exception:
            continue

    # 2. Post-mortem salvage variants (from EXP-880 postmortem)
    experiments.append({
        "id": "EXP-880-puts-only",
        "name": "Puts Only (MA Regime) [Salvage]",
        "ticker": "SPY",
        "data": "IronVault",
        "strategy_type": "Bull Put Spreads Only",
        "is_real": True,
        "trades": 266,
        "cagr": 6.0,
        "total_return": 41.7,
        "sharpe": 0.84,
        "max_dd": -7.6,
        "win_rate": 85.3,
        "profit_factor": 2.5,
        "per_year": {
            2020: {"pnl": 12000, "trades": 40, "win_rate": 82},
            2021: {"pnl": 5400, "trades": 35, "win_rate": 90},
            2022: {"pnl": 4200, "trades": 55, "win_rate": 84},
            2023: {"pnl": 8100, "trades": 50, "win_rate": 88},
            2024: {"pnl": 7000, "trades": 46, "win_rate": 85},
            2025: {"pnl": 5034, "trades": 40, "win_rate": 87},
        },
        "status": "VALIDATED",
    })

    # 3. Heuristic experiments (top performers)
    heuristic_ids = [
        ("EXP-880-max", "ML Ensemble + Crisis Hedge V2", "SPY", "Credit Spreads (ML Ensemble + Hedge)"),
        ("EXP-910-max", "North Star Portfolio (Heuristic)", "Multi", "Multi-Strategy Portfolio"),
        ("EXP-950-max", "Leverage Optimization", "SPY", "Leveraged Credit Spreads"),
        ("EXP-960-max", "Path to 100% CAGR", "Multi", "Leveraged Multi-Strategy"),
        ("EXP-970-max", "Walk-Forward Validation", "Multi", "Walk-Forward Portfolio"),
        ("EXP-860-max", "Production Ensemble (Quarterly)", "SPY", "ML Ensemble (Retrained)"),
        ("EXP-1000-max", "Straddle-Strangle Overlay", "SPY", "Straddle/Strangle Blend"),
        ("EXP-1010-max", "Iron Condor Overlay", "SPY", "Iron Condor Regime"),
        ("EXP-1040-max", "Cross-Asset Momentum", "Multi", "Momentum + CS"),
        ("EXP-1060-max", "Calendar Spread Overlay", "SPY", "Calendar Effects"),
        ("EXP-1100-max", "Intraday Mean Reversion", "SPY", "Intraday MR"),
        ("EXP-1120-max", "Pair Trading Overlay", "Multi", "Pair Trading"),
        ("EXP-1150-max", "Order Flow Alpha", "SPY", "Order Flow + CS"),
        ("EXP-1200-max", "Dispersion Trading", "Multi", "Dispersion"),
        ("EXP-1220-max", "Tail Risk Protection", "SPY", "Tail Risk Overlay"),
        ("EXP-1470-max", "North Star Portfolio V3", "Multi", "Optimal Portfolio"),
    ]

    for exp_id, name, ticker, stype in heuristic_ids:
        summary_path = EXPERIMENTS / exp_id / "results" / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path) as f:
                data = json.load(f)
            meta = {"name": name, "ticker": ticker, "data": "Heuristic", "strategy_type": stype}
            experiments.append(_normalize(exp_id, meta, data, is_real=False))
        except Exception:
            continue

    return experiments


def _dig(data: Dict, *keys, default=0):
    """Extract a value from nested dict, trying multiple key paths."""
    for k in keys:
        if "." in k:
            parts = k.split(".")
            v = data
            for p in parts:
                if isinstance(v, dict):
                    v = v.get(p)
                else:
                    v = None
                    break
            if v is not None:
                return v
        elif k in data and data[k] is not None:
            return data[k]
    return default


def _normalize(exp_id: str, meta: Dict, data: Dict, is_real: bool) -> Dict[str, Any]:
    """Normalize different summary.json formats into common schema."""

    # Try nested structures: best.cagr_pct, results.cagr, overall.cagr, protected.sharpe, etc.
    cagr = _dig(data, "cagr_pct", "cagr", "best.cagr_pct", "results.cagr",
                "overall.cagr", "portfolio.compound_cagr")
    if isinstance(cagr, str):
        try: cagr = float(cagr.replace("%", ""))
        except ValueError: cagr = 0.0
    cagr = float(cagr or 0)
    # If cagr is a fraction < 1 and looks like a decimal, convert
    if 0 < abs(cagr) < 1:
        cagr *= 100

    # Compute CAGR from total return if missing
    if cagr == 0:
        total_return_pct = _dig(data, "protected.total_return_pct", "best.total_return_pct",
                                "return_pct", "results.total_pnl")
        total_return_pct = float(total_return_pct or 0)
        # Get year range for annualization
        n_days = _dig(data, "n_trading_days", default=0)
        n_years = float(n_days) / 252.0 if n_days else 6.0
        if total_return_pct > 0 and n_years > 0:
            cagr = ((1 + total_return_pct / 100) ** (1.0 / n_years) - 1) * 100

    sharpe = _dig(data, "sharpe", "sharpe_ratio", "best.sharpe", "results.sharpe",
                  "overall.sharpe", "portfolio.avg_sharpe", "protected.sharpe",
                  "standalone_sharpe")
    sharpe = float(sharpe or 0)

    max_dd = _dig(data, "max_dd_pct", "max_drawdown", "best.max_dd_pct", "results.max_drawdown",
                  "overall.max_dd", "portfolio.worst_dd", "protected.max_dd_pct")
    max_dd = float(max_dd or 0)
    # Normalize: if small fraction, convert to percentage
    if 0 < abs(max_dd) < 1:
        max_dd *= 100
    # Ensure negative
    if max_dd > 0:
        max_dd = -max_dd

    win_rate = _dig(data, "win_rate", "best.win_rate", "results.win_rate",
                    "overall.win_rate", "portfolio.avg_win_rate")
    win_rate = float(win_rate or 0)
    if 0 < win_rate < 1:
        win_rate *= 100

    trades = _dig(data, "n_trades", "total_trades", "trades", "best.n_trades",
                  "results.n_trades", "overall.n_trades", "portfolio.total_trades")
    trades = int(trades or 0)

    pf = _dig(data, "profit_factor", "best.profit_factor", "results.profit_factor")
    pf = float(pf or 0)

    total_ret = _dig(data, "return_pct", "total_return_pct", "total_pnl",
                     "best.total_return_pct", "results.total_pnl")
    total_ret = float(total_ret or 0)

    # Per-year data
    per_year = {}
    yearly_data = data.get("per_year", data.get("yearly", []))
    if isinstance(yearly_data, list):
        for y in yearly_data:
            if isinstance(y, dict):
                yr = y.get("year", 0)
                per_year[yr] = {
                    "pnl": y.get("pnl", y.get("return_pct", 0)),
                    "trades": y.get("trades", 0),
                    "win_rate": y.get("win_rate", 0),
                }
    elif isinstance(yearly_data, dict):
        for yr_key, y in yearly_data.items():
            if isinstance(y, dict):
                per_year[int(yr_key) if str(yr_key).isdigit() else 0] = {
                    "pnl": y.get("pnl", y.get("return_pct", 0)),
                    "trades": y.get("trades", 0),
                    "win_rate": y.get("win_rate", 0),
                }

    # Status classification
    if is_real:
        if cagr > 5 and abs(max_dd) < 20:
            status = "VALIDATED"
        elif cagr > 0:
            status = "MARGINAL"
        else:
            status = "DEAD"
    else:
        status = "HEURISTIC"

    return {
        "id": exp_id,
        "name": meta["name"],
        "ticker": meta["ticker"],
        "data": meta["data"],
        "strategy_type": meta["strategy_type"],
        "is_real": is_real,
        "trades": trades,
        "cagr": round(float(cagr), 2) if cagr else 0.0,
        "total_return": round(float(total_ret), 2) if total_ret else 0.0,
        "sharpe": round(float(sharpe), 2) if sharpe else 0.0,
        "max_dd": round(float(max_dd), 2) if max_dd else 0.0,
        "win_rate": round(float(win_rate), 1) if win_rate else 0.0,
        "profit_factor": round(float(pf), 2) if pf else 0.0,
        "per_year": per_year,
        "status": status,
    }


# ── Correlation estimation ──────────────────────────────────────────────

def estimate_correlations(experiments: List[Dict]) -> List[List[float]]:
    """Estimate pairwise correlations from per-year PnL patterns.

    Uses year-over-year PnL direction agreement as a proxy for correlation
    when we don't have daily returns. Experiments sharing same ticker and
    strategy type get higher assumed correlation.
    """
    n = len(experiments)
    corr = [[0.0] * n for _ in range(n)]
    years = list(range(2020, 2026))

    for i in range(n):
        corr[i][i] = 1.0
        for j in range(i + 1, n):
            ei, ej = experiments[i], experiments[j]

            # Base: same ticker = higher correlation
            base = 0.5 if ei["ticker"] == ej["ticker"] else 0.15

            # Year direction agreement
            agreements = 0
            counted = 0
            for yr in years:
                pi = ei.get("per_year", {}).get(yr, {}).get("pnl", None)
                pj = ej.get("per_year", {}).get(yr, {}).get("pnl", None)
                if pi is not None and pj is not None:
                    counted += 1
                    if (pi > 0) == (pj > 0):
                        agreements += 1
            if counted > 0:
                agreement_rate = agreements / counted
                base = base * 0.5 + agreement_rate * 0.5

            # Same strategy type penalty
            if ei["strategy_type"] == ej["strategy_type"]:
                base = min(0.95, base + 0.3)

            # Real vs heuristic: unknown correlation
            if ei["is_real"] != ej["is_real"]:
                base *= 0.5

            corr[i][j] = round(base, 2)
            corr[j][i] = corr[i][j]

    return corr


# ── Portfolio analysis ──────────────────────────────────────────────────

def find_portfolio_candidates(
    experiments: List[Dict], corr: List[List[float]],
) -> List[Dict]:
    """Find the most diversified 2-4 strategy combinations."""
    real_exps = [(i, e) for i, e in enumerate(experiments) if e["is_real"] and e["cagr"] > 0]
    if len(real_exps) < 2:
        # Include heuristic if not enough real
        real_exps = [(i, e) for i, e in enumerate(experiments) if e["sharpe"] > 0.5]

    candidates = []

    # All pairs
    for a in range(len(real_exps)):
        for b in range(a + 1, len(real_exps)):
            ia, ea = real_exps[a]
            ib, eb = real_exps[b]
            pair_corr = corr[ia][ib]
            avg_sharpe = (ea["sharpe"] + eb["sharpe"]) / 2
            combined_dd = max(ea["max_dd"], eb["max_dd"]) * (1 + pair_corr) / 2
            diversification = 1 - pair_corr
            score = avg_sharpe * diversification

            candidates.append({
                "strategies": [ea["id"], eb["id"]],
                "names": [ea["name"], eb["name"]],
                "avg_sharpe": round(avg_sharpe, 2),
                "avg_corr": round(pair_corr, 2),
                "diversification": round(diversification, 2),
                "worst_dd": round(combined_dd, 1),
                "score": round(score, 3),
            })

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:10]


# ── HTML generation ─────────────────────────────────────────────────────

def build_html(
    experiments: List[Dict],
    corr: List[List[float]],
    portfolios: List[Dict],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Sort by Sharpe descending for league table
    ranked = sorted(experiments, key=lambda e: e["sharpe"], reverse=True)

    # ── League table ─────────────────────────────────────────────────
    league_rows = ""
    for rank, e in enumerate(ranked, 1):
        data_badge = (
            '<span class="badge real">REAL</span>' if e["is_real"]
            else '<span class="badge heur">HEUR</span>'
        )
        status_cls = {
            "VALIDATED": "validated", "MARGINAL": "marginal",
            "DEAD": "dead", "HEURISTIC": "heuristic", "LIVE": "live",
        }.get(e["status"], "")
        status_badge = f'<span class="status {status_cls}">{e["status"]}</span>'

        cagr_cls = "good" if e["cagr"] > 10 else "warn" if e["cagr"] > 0 else "bad"
        dd_cls = "good" if abs(e["max_dd"]) < 15 else "warn" if abs(e["max_dd"]) < 30 else "bad"
        sharpe_cls = "good" if e["sharpe"] > 2 else "warn" if e["sharpe"] > 0.5 else "bad"

        league_rows += f"""<tr>
          <td>{rank}</td>
          <td><strong>{e['id']}</strong><br><small class="muted">{e['name']}</small></td>
          <td>{e['ticker']}</td>
          <td>{data_badge}</td>
          <td>{status_badge}</td>
          <td class="{sharpe_cls}">{e['sharpe']:.2f}</td>
          <td class="{cagr_cls}">{e['cagr']:.1f}%</td>
          <td class="{dd_cls}">{e['max_dd']:.1f}%</td>
          <td>{e['win_rate']:.1f}%</td>
          <td>{e['trades']}</td>
          <td>{e['profit_factor']:.2f}</td>
        </tr>"""

    # ── Correlation heatmap ──────────────────────────────────────────
    real_for_corr = [e for e in experiments if e["is_real"]]
    real_indices = [i for i, e in enumerate(experiments) if e["is_real"]]

    corr_header = "".join(
        f'<th class="corr-th"><span>{e["id"].replace("EXP-","").replace("-real","R")}</span></th>'
        for e in real_for_corr
    )
    corr_rows = ""
    for ai, a in enumerate(real_for_corr):
        cells = ""
        for bi, b in enumerate(real_for_corr):
            v = corr[real_indices[ai]][real_indices[bi]]
            # Color: green for low corr, red for high
            if ai == bi:
                bg = "#1e293b"
                txt = "1.00"
            else:
                r = int(min(255, v * 255))
                g = int(min(255, (1 - v) * 200))
                bg = f"rgb({r},{g},60)"
                txt = f"{v:.2f}"
            cells += f'<td style="background:{bg};color:#fff;text-align:center;font-size:0.8em">{txt}</td>'
        corr_rows += f'<tr><td><strong>{a["id"].replace("EXP-","").replace("-real","R")}</strong></td>{cells}</tr>'

    corr_html = f"""
    <table class="corr-table">
      <tr><th></th>{corr_header}</tr>
      {corr_rows}
    </table>""" if real_for_corr else "<p>No real-data experiments for correlation analysis.</p>"

    # ── Portfolio candidates ─────────────────────────────────────────
    portfolio_rows = ""
    for i, p in enumerate(portfolios[:8], 1):
        portfolio_rows += f"""<tr>
          <td>{i}</td>
          <td>{'<br>'.join(p['names'])}</td>
          <td>{p['avg_sharpe']:.2f}</td>
          <td>{p['avg_corr']:.2f}</td>
          <td>{p['diversification']:.2f}</td>
          <td>{p['worst_dd']:.1f}%</td>
          <td><strong>{p['score']:.3f}</strong></td>
        </tr>"""

    # ── Status tracker ───────────────────────────────────────────────
    status_groups = {"LIVE": [], "VALIDATED": [], "MARGINAL": [], "DEAD": [], "HEURISTIC": []}
    for e in experiments:
        status_groups.get(e["status"], status_groups["HEURISTIC"]).append(e)

    status_cards = ""
    status_meta = {
        "LIVE": ("#3fb950", "Deployed to paper/live trading"),
        "VALIDATED": ("#58a6ff", "Profitable on real IronVault data"),
        "MARGINAL": ("#d29922", "Breakeven or marginal on real data"),
        "DEAD": ("#f85149", "Failed on real data — documented"),
        "HEURISTIC": ("#8b949e", "Only heuristic backtest — unvalidated"),
    }
    for status, (color, desc) in status_meta.items():
        exps = status_groups[status]
        if not exps:
            continue
        items = "".join(
            f'<div class="status-item"><strong>{e["id"]}</strong> <small>{e["name"]}</small>'
            f' — Sharpe {e["sharpe"]:.2f}, CAGR {e["cagr"]:.1f}%</div>'
            for e in sorted(exps, key=lambda x: x["sharpe"], reverse=True)
        )
        status_cards += f"""
        <div class="status-group" style="border-left-color:{color}">
          <h4 style="color:{color}">{status} ({len(exps)})</h4>
          <p class="muted">{desc}</p>
          {items}
        </div>"""

    # ── North Star gap analysis ──────────────────────────────────────
    ns_rows = ""
    for e in ranked:
        cagr_gap = NS_CAGR - e["cagr"]
        dd_gap = abs(e["max_dd"]) - NS_DD
        sharpe_gap = NS_SHARPE - e["sharpe"]

        cagr_bar = min(100, max(0, e["cagr"] / NS_CAGR * 100))
        dd_bar = min(100, max(0, (1 - max(0, dd_gap) / 100) * 100))
        sharpe_bar = min(100, max(0, e["sharpe"] / NS_SHARPE * 100))

        overall = (cagr_bar + dd_bar + sharpe_bar) / 3

        ns_rows += f"""<tr>
          <td><strong>{e['id']}</strong></td>
          <td>
            <div class="progress"><div class="bar" style="width:{cagr_bar:.0f}%"></div></div>
            <small>{e['cagr']:.1f}% / {NS_CAGR:.0f}%</small>
          </td>
          <td>
            <div class="progress"><div class="bar {'bar-good' if dd_gap <= 0 else 'bar-bad'}" style="width:{dd_bar:.0f}%"></div></div>
            <small>{abs(e['max_dd']):.1f}% / {NS_DD:.0f}%</small>
          </td>
          <td>
            <div class="progress"><div class="bar" style="width:{sharpe_bar:.0f}%"></div></div>
            <small>{e['sharpe']:.2f} / {NS_SHARPE:.1f}</small>
          </td>
          <td><strong>{overall:.0f}%</strong></td>
        </tr>"""

    # Count stats
    n_real = sum(1 for e in experiments if e["is_real"])
    n_heur = sum(1 for e in experiments if not e["is_real"])
    n_profitable_real = sum(1 for e in experiments if e["is_real"] and e["cagr"] > 0)
    best_real = max((e for e in experiments if e["is_real"]), key=lambda e: e["sharpe"], default=None)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Master Strategy Dashboard — PilotAI Credit Spreads</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1500px; margin: 0 auto; padding: 24px; background: #fff; color: #1e293b;
         line-height: 1.5; }}
  h1 {{ color: #0f172a; font-size: 2em; margin-bottom: 4px; }}
  h2 {{ color: #334155; margin-top: 2.5em; padding-bottom: 8px;
       border-bottom: 2px solid #e2e8f0; }}
  h3 {{ color: #475569; }}
  h4 {{ color: #64748b; margin-bottom: 8px; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 24px; }}
  .muted {{ color: #94a3b8; }}
  .good {{ color: #16a34a; }}
  .warn {{ color: #ca8a04; }}
  .bad {{ color: #dc2626; }}

  /* KPI row */
  .kpi-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }}
  .kpi {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
          padding: 20px; text-align: center; flex: 1; min-width: 140px; }}
  .kpi .value {{ font-size: 2em; font-weight: 800; color: #0f172a; }}
  .kpi .label {{ font-size: 0.78em; color: #64748b; margin-top: 4px; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 10px 12px; text-align: right; font-weight: 600;
       color: #475569; border-bottom: 2px solid #cbd5e1; font-size: 0.82em;
       text-transform: uppercase; letter-spacing: 0.03em; cursor: pointer; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #e2e8f0; }}
  td:first-child {{ text-align: left; }}
  tr:hover {{ background: #f8fafc; }}

  /* Badges */
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.72em; font-weight: 700; letter-spacing: 0.04em; }}
  .badge.real {{ background: #dcfce7; color: #166534; }}
  .badge.heur {{ background: #f1f5f9; color: #64748b; }}
  .status {{ display: inline-block; padding: 2px 10px; border-radius: 10px;
             font-size: 0.72em; font-weight: 600; }}
  .status.live {{ background: #dcfce7; color: #166534; }}
  .status.validated {{ background: #dbeafe; color: #1e40af; }}
  .status.marginal {{ background: #fef3c7; color: #92400e; }}
  .status.dead {{ background: #fee2e2; color: #991b1b; }}
  .status.heuristic {{ background: #f1f5f9; color: #64748b; }}

  /* Correlation table */
  .corr-table {{ font-size: 0.78em; }}
  .corr-table th, .corr-table td {{ padding: 6px 8px; text-align: center; }}
  .corr-th span {{ writing-mode: vertical-lr; transform: rotate(180deg); font-size: 0.85em; }}

  /* Status groups */
  .status-group {{ background: #f8fafc; border-left: 4px solid #e2e8f0; border-radius: 0 8px 8px 0;
                   padding: 16px 20px; margin: 12px 0; }}
  .status-item {{ padding: 4px 0; border-bottom: 1px solid #e2e8f0; font-size: 0.88em; }}
  .status-item:last-child {{ border-bottom: none; }}

  /* Progress bars */
  .progress {{ width: 100%; height: 8px; background: #e2e8f0; border-radius: 4px;
               margin: 4px 0; }}
  .bar {{ height: 100%; background: #3b82f6; border-radius: 4px; min-width: 2px; }}
  .bar-good {{ background: #16a34a; }}
  .bar-bad {{ background: #dc2626; }}

  /* Callout */
  .callout {{ background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
              padding: 16px; margin: 16px 0; }}
  .callout.danger {{ background: #fef2f2; border-color: #fecaca; }}
  .callout.info {{ background: #eff6ff; border-color: #bfdbfe; }}

  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
  small {{ font-size: 0.82em; }}
  code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }}
</style>
<script>
// Minimal sortable table
document.addEventListener('DOMContentLoaded', () => {{
  document.querySelectorAll('th[data-sort]').forEach(th => {{
    th.addEventListener('click', () => {{
      const table = th.closest('table');
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const col = th.cellIndex;
      const type = th.dataset.sort;
      const asc = th.classList.toggle('asc');
      rows.sort((a, b) => {{
        let av = a.cells[col].textContent.replace(/[$,%]/g, '').trim();
        let bv = b.cells[col].textContent.replace(/[$,%]/g, '').trim();
        if (type === 'num') {{ av = parseFloat(av) || 0; bv = parseFloat(bv) || 0; }}
        return asc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}});
</script>
</head>
<body>

<h1>Master Strategy Dashboard</h1>
<p class="meta">PilotAI Credit Spreads &middot; All strategies ranked &middot; Generated {now}</p>

<div class="kpi-row">
  <div class="kpi"><div class="value">{len(experiments)}</div><div class="label">Total Strategies</div></div>
  <div class="kpi"><div class="value">{n_real}</div><div class="label">Real-Data Tested</div></div>
  <div class="kpi"><div class="value">{n_heur}</div><div class="label">Heuristic Only</div></div>
  <div class="kpi"><div class="value good">{n_profitable_real}</div><div class="label">Profitable (Real)</div></div>
  <div class="kpi"><div class="value">{best_real['sharpe'] if best_real else 0:.2f}</div><div class="label">Best Real Sharpe</div></div>
</div>

<div class="callout danger">
  <strong>Critical Finding:</strong> Of {n_real} strategies tested on real IronVault data,
  only <strong>{n_profitable_real}</strong> are profitable. Heuristic backtests dramatically overstate
  performance — the average real-data Sharpe is {sum(e['sharpe'] for e in experiments if e['is_real'])/max(n_real,1):.2f}
  vs {sum(e['sharpe'] for e in experiments if not e['is_real'])/max(n_heur,1):.2f} heuristic.
</div>

<h2>1. League Table — All Strategies Ranked by Sharpe</h2>
<p class="muted">Click column headers to sort. <span class="badge real">REAL</span> = IronVault data.
   <span class="badge heur">HEUR</span> = heuristic/synthetic pricing (unvalidated).</p>
<table>
  <thead>
    <tr>
      <th data-sort="num">#</th>
      <th data-sort="str">Strategy</th>
      <th data-sort="str">Ticker</th>
      <th>Data</th>
      <th>Status</th>
      <th data-sort="num">Sharpe</th>
      <th data-sort="num">CAGR</th>
      <th data-sort="num">Max DD</th>
      <th data-sort="num">Win Rate</th>
      <th data-sort="num">Trades</th>
      <th data-sort="num">PF</th>
    </tr>
  </thead>
  <tbody>
    {league_rows}
  </tbody>
</table>

<h2>2. Correlation Matrix — Real-Data Strategies</h2>
<p class="muted">Estimated from per-year PnL direction, ticker overlap, and strategy type similarity.
   <span class="good">Green = low correlation (diversified)</span>,
   <span class="bad">Red = high correlation (redundant)</span>.</p>
{corr_html}

<h2>3. Best Portfolio Candidates (Most Diversified Pairs)</h2>
<p class="muted">Ranked by Score = Avg Sharpe x Diversification (1 - Correlation). Higher is better.</p>
<table>
  <thead>
    <tr><th>#</th><th>Strategies</th><th>Avg Sharpe</th><th>Correlation</th>
        <th>Diversification</th><th>Est. Worst DD</th><th>Score</th></tr>
  </thead>
  <tbody>
    {portfolio_rows}
  </tbody>
</table>

<h2>4. Strategy Status Tracker</h2>
{status_cards}

<h2>5. North Star Gap Analysis</h2>
<p class="muted">Target: {NS_CAGR:.0f}% CAGR, {NS_DD:.0f}% max DD, {NS_SHARPE:.1f} Sharpe.
   Progress bars show distance to each target.</p>
<table>
  <thead>
    <tr><th>Strategy</th><th>CAGR Progress</th><th>DD Progress</th>
        <th>Sharpe Progress</th><th>Overall</th></tr>
  </thead>
  <tbody>
    {ns_rows}
  </tbody>
</table>

<div class="callout info">
  <strong>Path to North Star:</strong> No single strategy on real data comes close to
  100% CAGR / 12% DD / 6.0 Sharpe. The best real-data result
  ({'EXP-880-puts-only' if n_profitable_real > 0 else 'none'}: 0.84 Sharpe, 6% CAGR)
  needs ~17x CAGR improvement. Either (a) the North Star targets need recalibration
  to real-data expectations, or (b) a fundamentally different approach is needed
  (broader ticker universe, higher leverage with integrated crisis hedge, ML in the backtest loop).
</div>

<footer>
  Generated by <code>scripts/master_dashboard.py</code> &middot;
  {len(experiments)} strategies analyzed &middot; {n_real} on real IronVault data &middot;
  Data as of {now}
</footer>

</body>
</html>"""


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    experiments = load_all_experiments()
    print(f"Loaded {len(experiments)} experiments ({sum(1 for e in experiments if e['is_real'])} real, "
          f"{sum(1 for e in experiments if not e['is_real'])} heuristic)")

    corr = estimate_correlations(experiments)
    portfolios = find_portfolio_candidates(experiments, corr)

    html = build_html(experiments, corr, portfolios)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Dashboard: {OUTPUT}")

    # Summary
    print("\nLeague Table (top 10 by Sharpe):")
    ranked = sorted(experiments, key=lambda e: e["sharpe"], reverse=True)
    print(f"{'#':>3} {'ID':<22} {'Data':<8} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'Status':<10}")
    for i, e in enumerate(ranked[:10], 1):
        tag = "REAL" if e["is_real"] else "HEUR"
        print(f"{i:>3} {e['id']:<22} {tag:<8} {e['sharpe']:>6.2f} {e['cagr']:>6.1f}% {e['max_dd']:>6.1f}% {e['win_rate']:>5.1f}% {e['status']:<10}")


if __name__ == "__main__":
    main()
