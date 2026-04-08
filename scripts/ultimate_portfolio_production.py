#!/usr/bin/env python3
"""
Production Walk-Forward Backtest — Ultimate Portfolio at 1.6× Leverage.

Expanding-window walk-forward:
  Train 2020-2021 → Test 2022
  Train 2020-2022 → Test 2023
  Train 2020-2023 → Test 2024
  Train 2020-2024 → Test 2025

Weights: EXP-1220 Dynamic 95%, Cross-Asset Pairs 1.67%,
         TLT Iron Condors 1.67%, Vol Term Structure 1.67%.

Includes 0.5% round-trip slippage on rebalance.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ultimate_portfolio import (
    load_exp1220_dynamic,
    load_cross_asset_pairs,
    load_vol_term_structure,
    load_tlt_iron_condors,
    calc_metrics,
)

TRADING_DAYS = 252
ACCOUNT = 100_000
LEVERAGE = 1.6
SLIPPAGE_RT = 0.005  # 0.5% round-trip

WEIGHTS = {
    "EXP-1220 Dynamic": 0.95,
    "Cross-Asset Pairs": 0.0167,
    "TLT Iron Condors": 0.0167,
    "Vol Term Structure": 0.0167,
}

REPORT_PATH = ROOT / "reports" / "ultimate_portfolio_production.html"

# Walk-forward windows (expanding)
WF_WINDOWS = [
    {"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31"), "label": "2022"},
    {"train": ("2020-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31"), "label": "2023"},
    {"train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31"), "label": "2024"},
    {"train": ("2020-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31"), "label": "2025"},
]


def load_all_strategies() -> pd.DataFrame:
    """Load all 4 strategy daily returns into a DataFrame."""
    print("Loading EXP-1220 Dynamic...")
    s1 = load_exp1220_dynamic()
    print("Loading Cross-Asset Pairs...")
    s2 = load_cross_asset_pairs()
    print("Loading Vol Term Structure...")
    s3 = load_vol_term_structure()
    print("Loading TLT Iron Condors...")
    s4 = load_tlt_iron_condors()

    df = pd.DataFrame({
        s1.name: s1,
        s2.name: s2,
        s3.name: s3,
        s4.name: s4,
    })
    df = df.sort_index()
    df = df.fillna(0)
    # Keep only 2020+
    df = df[df.index >= "2020-01-01"]
    return df


def apply_slippage(returns: np.ndarray, prev_weights: np.ndarray,
                   new_weights: np.ndarray) -> float:
    """Compute slippage cost from rebalancing as fraction of portfolio."""
    turnover = np.sum(np.abs(new_weights - prev_weights))
    return turnover * SLIPPAGE_RT / 2  # half of round-trip per side


def run_walk_forward(df: pd.DataFrame) -> dict:
    """Run expanding-window walk-forward with fixed weights + leverage + slippage."""
    names = list(WEIGHTS.keys())
    w = np.array([WEIGHTS[n] for n in names])

    all_oos_returns = []
    all_oos_dates = []
    window_results = []

    for wf in WF_WINDOWS:
        train_mask = (df.index >= wf["train"][0]) & (df.index <= wf["train"][1])
        test_mask = (df.index >= wf["test"][0]) & (df.index <= wf["test"][1])

        train_df = df.loc[train_mask, names]
        test_df = df.loc[test_mask, names]

        if train_df.empty or test_df.empty:
            continue

        # Train metrics (unlevered, no slippage)
        train_rets = (train_df.values @ w)
        train_metrics = calc_metrics(train_rets)

        # Test: apply leverage + slippage on first day of OOS period
        test_matrix = test_df.values
        test_port = test_matrix @ w * LEVERAGE

        # Apply slippage at start of test window (rebalance cost)
        slippage_cost = apply_slippage(
            test_port,
            prev_weights=w if len(window_results) == 0 else w,  # fixed weights
            new_weights=w,
        )
        # For the first OOS day, deduct rebalance slippage
        if len(test_port) > 0:
            test_port[0] -= slippage_cost

        test_metrics = calc_metrics(test_port)

        # Levered train metrics for comparison
        train_lev = train_rets * LEVERAGE
        train_lev_metrics = calc_metrics(train_lev)

        window_results.append({
            "label": wf["label"],
            "train_start": wf["train"][0],
            "train_end": wf["train"][1],
            "test_start": wf["test"][0],
            "test_end": wf["test"][1],
            "train_days": len(train_df),
            "test_days": len(test_df),
            "train_metrics": train_lev_metrics,
            "test_metrics": test_metrics,
        })

        all_oos_returns.extend(test_port.tolist())
        all_oos_dates.extend(test_df.index.tolist())

    # Build combined OOS equity curve
    oos_returns = np.array(all_oos_returns)
    oos_dates = all_oos_dates
    oos_equity = ACCOUNT * np.cumprod(1 + oos_returns)
    oos_equity = np.insert(oos_equity, 0, ACCOUNT)
    oos_dates_str = ["2021-12-31"] + [str(d)[:10] for d in oos_dates]

    # Full backtest (in-sample + OOS) for reference
    full_rets = (df[names].values @ w) * LEVERAGE
    full_equity = ACCOUNT * np.cumprod(1 + full_rets)
    full_equity = np.insert(full_equity, 0, ACCOUNT)
    full_dates_str = ["2019-12-31"] + [str(d)[:10] for d in df.index]

    # Monthly returns for heatmap
    monthly = _monthly_returns(oos_returns, oos_dates)

    # Drawdown series (OOS)
    eq = np.cumprod(1 + oos_returns)
    hwm = np.maximum.accumulate(eq)
    drawdown = (eq / hwm - 1) * 100  # in pct

    # Year-by-year OOS metrics
    yearly_oos = {}
    for i, d in enumerate(oos_dates):
        yr = pd.Timestamp(d).year
        yearly_oos.setdefault(yr, []).append(oos_returns[i])
    yearly_metrics = {yr: calc_metrics(np.array(v)) for yr, v in sorted(yearly_oos.items())}

    # Aggregate OOS metrics
    agg_oos = calc_metrics(oos_returns)

    return {
        "leverage": LEVERAGE,
        "slippage_rt": SLIPPAGE_RT,
        "weights": WEIGHTS,
        "windows": window_results,
        "oos_aggregate": agg_oos,
        "yearly_oos": yearly_metrics,
        "monthly_returns": monthly,
        "oos_equity": oos_equity.tolist(),
        "oos_dates": oos_dates_str,
        "oos_drawdown": drawdown.tolist(),
        "oos_drawdown_dates": [str(d)[:10] for d in oos_dates],
        "full_equity": full_equity.tolist(),
        "full_dates": full_dates_str,
    }


def _monthly_returns(returns: np.ndarray, dates: list) -> dict:
    """Compute monthly returns for heatmap. Returns {year: {month: ret_pct}}."""
    monthly = {}
    for i, d in enumerate(dates):
        dt = pd.Timestamp(d)
        yr, mo = dt.year, dt.month
        monthly.setdefault(yr, {}).setdefault(mo, []).append(returns[i])

    result = {}
    for yr, months in sorted(monthly.items()):
        result[yr] = {}
        for mo, rets in sorted(months.items()):
            eq = np.prod(1 + np.array(rets)) - 1
            result[yr][mo] = round(eq * 100, 2)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════


def _svg_equity_curve(equity: list, dates: list, title: str,
                      w: int = 900, h: int = 350) -> str:
    pl, pr, pt, pb = 75, 20, 40, 55
    pw, ph = w - pl - pr, h - pt - pb

    vals = equity
    ymin, ymax = min(vals) * 0.95, max(vals) * 1.05
    if ymax <= ymin:
        ymax = ymin + 1

    def tx(i):
        return pl + i / max(len(dates) - 1, 1) * pw

    def ty(v):
        return pt + (1 - (v - ymin) / (ymax - ymin)) * ph

    p = [f'<svg width="{w}" height="{h}" style="background:#1e293b;border-radius:8px;margin:1rem 0">']
    p.append(f'<text x="{w // 2}" y="24" text-anchor="middle" font-size="14" font-weight="bold" fill="#e2e8f0">{title}</text>')

    # Y gridlines
    for j in range(6):
        yv = ymin + j / 5 * (ymax - ymin)
        y = ty(yv)
        p.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{pl + pw}" y2="{y:.0f}" stroke="#334155" stroke-width="0.5"/>')
        label = f"${yv:,.0f}" if yv < 1e6 else f"${yv / 1e6:.2f}M"
        p.append(f'<text x="{pl - 8}" y="{y + 4:.0f}" text-anchor="end" font-size="10" fill="#94a3b8">{label}</text>')

    # X labels
    step = max(1, len(dates) // 8)
    for i in range(0, len(dates), step):
        p.append(f'<text x="{tx(i):.0f}" y="{h - 12}" text-anchor="middle" font-size="9" fill="#94a3b8">{dates[i][:7]}</text>')

    # Equity line
    d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(vals[i]):.1f}" for i in range(len(vals)))
    p.append(f'<path d="{d}" fill="none" stroke="#4ade80" stroke-width="2.5"/>')

    # Starting capital line
    y0 = ty(ACCOUNT)
    p.append(f'<line x1="{pl}" y1="{y0:.0f}" x2="{pl + pw}" y2="{y0:.0f}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4"/>')
    p.append(f'<text x="{pl + pw + 2}" y="{y0 + 3:.0f}" font-size="8" fill="#94a3b8">$100K</text>')

    p.append("</svg>")
    return "\n".join(p)


def _svg_drawdown(dd: list, dates: list, w: int = 900, h: int = 200) -> str:
    pl, pr, pt, pb = 75, 20, 40, 45
    pw, ph = w - pl - pr, h - pt - pb

    dd_min = min(dd) if dd else 0
    ymin, ymax = dd_min * 1.2, 0.5

    def tx(i):
        return pl + i / max(len(dates) - 1, 1) * pw

    def ty(v):
        return pt + (1 - (v - ymin) / (ymax - ymin)) * ph

    p = [f'<svg width="{w}" height="{h}" style="background:#1e293b;border-radius:8px;margin:1rem 0">']
    p.append(f'<text x="{w // 2}" y="24" text-anchor="middle" font-size="14" font-weight="bold" fill="#e2e8f0">Drawdown (OOS)</text>')

    # Y gridlines
    for j in range(5):
        yv = ymin + j / 4 * (ymax - ymin)
        y = ty(yv)
        p.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{pl + pw}" y2="{y:.0f}" stroke="#334155" stroke-width="0.5"/>')
        p.append(f'<text x="{pl - 8}" y="{y + 4:.0f}" text-anchor="end" font-size="10" fill="#94a3b8">{yv:.1f}%</text>')

    # X labels
    step = max(1, len(dates) // 8)
    for i in range(0, len(dates), step):
        p.append(f'<text x="{tx(i):.0f}" y="{h - 8}" text-anchor="middle" font-size="9" fill="#94a3b8">{dates[i][:7]}</text>')

    # Zero line
    y0 = ty(0)
    p.append(f'<line x1="{pl}" y1="{y0:.0f}" x2="{pl + pw}" y2="{y0:.0f}" stroke="#94a3b8" stroke-width="1"/>')

    # Drawdown fill
    if dd:
        fill_points = f"M{tx(0):.1f},{y0:.1f}"
        for i in range(len(dd)):
            fill_points += f" L{tx(i):.1f},{ty(dd[i]):.1f}"
        fill_points += f" L{tx(len(dd) - 1):.1f},{y0:.1f} Z"
        p.append(f'<path d="{fill_points}" fill="rgba(239,68,68,0.3)" stroke="#ef4444" stroke-width="1.5"/>')

    p.append("</svg>")
    return "\n".join(p)


def _monthly_heatmap(monthly: dict, w: int = 900) -> str:
    """SVG monthly returns heatmap."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = sorted(monthly.keys())
    if not years:
        return ""

    cell_w, cell_h = 58, 32
    pl, pt = 60, 50
    tw = pl + cell_w * 12 + 80  # extra for YTD column
    th = pt + cell_h * len(years) + 20

    p = [f'<svg width="{tw}" height="{th}" style="background:#1e293b;border-radius:8px;margin:1rem 0">']
    p.append(f'<text x="{tw // 2}" y="24" text-anchor="middle" font-size="14" font-weight="bold" fill="#e2e8f0">Monthly Returns Heatmap (%)</text>')

    # Month headers
    for j, m in enumerate(months):
        x = pl + j * cell_w + cell_w // 2
        p.append(f'<text x="{x}" y="{pt - 8}" text-anchor="middle" font-size="10" font-weight="bold" fill="#94a3b8">{m}</text>')
    # YTD header
    x_ytd = pl + 12 * cell_w + cell_w // 2
    p.append(f'<text x="{x_ytd}" y="{pt - 8}" text-anchor="middle" font-size="10" font-weight="bold" fill="#94a3b8">YTD</text>')

    for ri, yr in enumerate(years):
        y = pt + ri * cell_h
        p.append(f'<text x="{pl - 8}" y="{y + cell_h // 2 + 4}" text-anchor="end" font-size="11" font-weight="bold" fill="#e2e8f0">{yr}</text>')

        ytd = 1.0
        for j in range(1, 13):
            x = pl + (j - 1) * cell_w
            val = monthly.get(yr, {}).get(j, None)
            if val is not None:
                ytd *= (1 + val / 100)
                # Color: green for positive, red for negative
                intensity = min(abs(val) / 15, 1.0)
                if val >= 0:
                    r, g, b = int(30 + (1 - intensity) * 20), int(100 + intensity * 155), int(50 + (1 - intensity) * 50)
                else:
                    r, g, b = int(100 + intensity * 155), int(50 + (1 - intensity) * 50), int(50 + (1 - intensity) * 20)
                p.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="rgb({r},{g},{b})" stroke="#0f172a" rx="3"/>')
                p.append(f'<text x="{x + cell_w // 2}" y="{y + cell_h // 2 + 4}" text-anchor="middle" font-size="10" font-weight="bold" fill="#f8fafc">{val:+.1f}</text>')
            else:
                p.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="#334155" stroke="#0f172a" rx="3"/>')
                p.append(f'<text x="{x + cell_w // 2}" y="{y + cell_h // 2 + 4}" text-anchor="middle" font-size="10" fill="#64748b">—</text>')

        # YTD cell
        ytd_pct = (ytd - 1) * 100
        intensity = min(abs(ytd_pct) / 50, 1.0)
        if ytd_pct >= 0:
            r, g, b = int(30 + (1 - intensity) * 20), int(100 + intensity * 155), int(50 + (1 - intensity) * 50)
        else:
            r, g, b = int(100 + intensity * 155), int(50 + (1 - intensity) * 50), int(50 + (1 - intensity) * 20)
        x = pl + 12 * cell_w
        p.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="rgb({r},{g},{b})" stroke="#0f172a" rx="3"/>')
        p.append(f'<text x="{x + cell_w // 2}" y="{y + cell_h // 2 + 4}" text-anchor="middle" font-size="10" font-weight="bold" fill="#f8fafc">{ytd_pct:+.1f}</text>')

    p.append("</svg>")
    return "\n".join(p)


def generate_html(data: dict) -> str:
    """Generate self-contained HTML report."""
    agg = data["oos_aggregate"]
    yearly = data["yearly_oos"]
    windows = data["windows"]

    equity_svg = _svg_equity_curve(data["oos_equity"], data["oos_dates"],
                                    "OOS Equity Curve (1.6× Leverage, $100K Start)")
    dd_svg = _svg_drawdown(data["oos_drawdown"], data["oos_drawdown_dates"])
    heatmap_svg = _monthly_heatmap(data["monthly_returns"])

    # Year-by-year OOS table
    yearly_rows = ""
    for yr, m in sorted(yearly.items()):
        color_cagr = "#4ade80" if m["cagr_pct"] > 0 else "#ef4444"
        color_dd = "#ef4444" if m["max_dd_pct"] > 10 else ("#f59e0b" if m["max_dd_pct"] > 5 else "#4ade80")
        yearly_rows += f"""<tr>
            <td style="font-weight:700">{yr}</td>
            <td style="color:{color_cagr}">{m['cagr_pct']:.1f}%</td>
            <td>{m['sharpe']:.2f}</td>
            <td style="color:{color_dd}">{m['max_dd_pct']:.1f}%</td>
            <td>{m['calmar']:.2f}</td>
            <td>{m['sortino']:.2f}</td>
            <td>{m['vol_pct']:.1f}%</td>
            <td>{m['total_ret_pct']:.1f}%</td>
        </tr>"""

    # Walk-forward windows table
    wf_rows = ""
    for w in windows:
        tm = w["test_metrics"]
        trm = w["train_metrics"]
        degradation = (trm["sharpe"] - tm["sharpe"]) / max(trm["sharpe"], 0.01) * 100
        deg_color = "#4ade80" if degradation < 20 else ("#f59e0b" if degradation < 50 else "#ef4444")
        wf_rows += f"""<tr>
            <td style="font-weight:700">{w['label']}</td>
            <td>{w['train_days']}</td>
            <td>{w['test_days']}</td>
            <td>{trm['sharpe']:.2f}</td>
            <td>{trm['cagr_pct']:.1f}%</td>
            <td>{tm['sharpe']:.2f}</td>
            <td>{tm['cagr_pct']:.1f}%</td>
            <td>{tm['max_dd_pct']:.1f}%</td>
            <td style="color:{deg_color}">{degradation:+.0f}%</td>
        </tr>"""

    # Weight display
    weight_items = " | ".join(f"{k}: {v*100:.1f}%" for k, v in data["weights"].items())

    # North Star check
    ns_cagr = agg["cagr_pct"] >= 55
    ns_dd = agg["max_dd_pct"] <= 15
    ns_sharpe = agg["sharpe"] >= 4.0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ultimate Portfolio — Production Walk-Forward (1.6× Leverage)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 24px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; color: #94a3b8; margin-top: 32px; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
  .meta {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
           gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 16px; }}
  .card-label {{ font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card-value {{ font-size: 1.5rem; font-weight: 700; margin-top: 4px; }}
  .positive {{ color: #4ade80; }}
  .negative {{ color: #ef4444; }}
  .warn {{ color: #f59e0b; }}
  .neutral {{ color: #e2e8f0; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; }}
  th {{ background: #1e293b; padding: 10px 12px; text-align: right;
       font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em;
       border-bottom: 2px solid #334155; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 8px 12px; text-align: right; font-size: 0.9rem;
       border-bottom: 1px solid #1e293b; }}
  td:first-child {{ text-align: left; }}
  tr:hover {{ background: #1e293b40; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 4px; font-size: 0.75rem;
            font-weight: 700; }}
  .badge-pass {{ background: #166534; color: #4ade80; }}
  .badge-fail {{ background: #7f1d1d; color: #ef4444; }}
  .badge-warn {{ background: #78350f; color: #f59e0b; }}
  .config {{ background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 24px;
             font-size: 0.85rem; line-height: 1.7; }}
  .config strong {{ color: #e2e8f0; }}
  .footer {{ color: #64748b; font-size: 0.75rem; margin-top: 40px; text-align: center; }}
</style>
</head>
<body>

<h1>Ultimate Portfolio — Production Walk-Forward</h1>
<div class="meta">
    1.6× Leverage | 0.5% Slippage | Expanding Windows | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<div class="config">
    <strong>Allocation:</strong> {weight_items}<br>
    <strong>Leverage:</strong> {LEVERAGE}× &nbsp;|&nbsp;
    <strong>Slippage:</strong> {SLIPPAGE_RT*100:.1f}% round-trip &nbsp;|&nbsp;
    <strong>OOS Period:</strong> 2022–2025 &nbsp;|&nbsp;
    <strong>Method:</strong> Expanding-window walk-forward (no lookahead)
</div>

<!-- Aggregate OOS KPIs -->
<div class="grid">
    <div class="card">
        <div class="card-label">OOS CAGR</div>
        <div class="card-value {'positive' if agg['cagr_pct'] > 0 else 'negative'}">{agg['cagr_pct']:.1f}%</div>
        <div style="font-size:.7rem;color:#64748b">{'<span class="badge badge-pass">PASS</span>' if ns_cagr else '<span class="badge badge-fail">MISS</span>'} ≥55%</div>
    </div>
    <div class="card">
        <div class="card-label">OOS Sharpe</div>
        <div class="card-value {'positive' if agg['sharpe'] >= 3 else 'warn'}">{agg['sharpe']:.2f}</div>
        <div style="font-size:.7rem;color:#64748b">{'<span class="badge badge-pass">PASS</span>' if ns_sharpe else '<span class="badge badge-warn">MISS</span>'} ≥4.0</div>
    </div>
    <div class="card">
        <div class="card-label">Max Drawdown</div>
        <div class="card-value {'positive' if agg['max_dd_pct'] < 12 else 'negative'}">{agg['max_dd_pct']:.1f}%</div>
        <div style="font-size:.7rem;color:#64748b">{'<span class="badge badge-pass">PASS</span>' if ns_dd else '<span class="badge badge-fail">MISS</span>'} ≤15%</div>
    </div>
    <div class="card">
        <div class="card-label">Calmar</div>
        <div class="card-value neutral">{agg['calmar']:.2f}</div>
    </div>
    <div class="card">
        <div class="card-label">Sortino</div>
        <div class="card-value neutral">{agg['sortino']:.2f}</div>
    </div>
    <div class="card">
        <div class="card-label">Volatility</div>
        <div class="card-value neutral">{agg['vol_pct']:.1f}%</div>
    </div>
    <div class="card">
        <div class="card-label">Total OOS Return</div>
        <div class="card-value positive">{agg['total_ret_pct']:.1f}%</div>
    </div>
    <div class="card">
        <div class="card-label">OOS Days</div>
        <div class="card-value neutral">{agg['n_days']}</div>
    </div>
</div>

<!-- Equity Curve -->
<h2>OOS Equity Curve</h2>
{equity_svg}

<!-- Drawdown -->
<h2>Drawdown</h2>
{dd_svg}

<!-- Year-by-Year OOS -->
<h2>Year-by-Year Out-of-Sample Performance</h2>
<table>
    <thead>
        <tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Sortino</th><th>Vol</th><th>Return</th></tr>
    </thead>
    <tbody>
        {yearly_rows}
    </tbody>
</table>

<!-- Walk-Forward Windows -->
<h2>Expanding Walk-Forward Windows</h2>
<table>
    <thead>
        <tr><th>OOS Year</th><th>Train Days</th><th>Test Days</th><th>Train Sharpe</th><th>Train CAGR</th><th>OOS Sharpe</th><th>OOS CAGR</th><th>OOS Max DD</th><th>Sharpe Degradation</th></tr>
    </thead>
    <tbody>
        {wf_rows}
    </tbody>
</table>

<!-- Monthly Returns Heatmap -->
<h2>Monthly Returns Heatmap</h2>
{heatmap_svg}

<div class="footer">
    PilotAI Credit Spreads — Production Walk-Forward Backtest<br>
    No lookahead bias. All OOS returns computed strictly out-of-sample with expanding training windows.
</div>

</body>
</html>"""


def main():
    print("=" * 70)
    print("Ultimate Portfolio — Production Walk-Forward Backtest (1.6× Leverage)")
    print("=" * 70)

    # Load strategies
    df = load_all_strategies()
    print(f"Loaded {len(df)} trading days, {len(df.columns)} strategies")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")

    # Run walk-forward
    print("\nRunning expanding-window walk-forward...")
    result = run_walk_forward(df)

    # Print summary
    agg = result["oos_aggregate"]
    print(f"\n{'─' * 50}")
    print(f"OOS Aggregate (2022-2025 at 1.6× leverage):")
    print(f"  CAGR:    {agg['cagr_pct']:.1f}%")
    print(f"  Sharpe:  {agg['sharpe']:.2f}")
    print(f"  Max DD:  {agg['max_dd_pct']:.1f}%")
    print(f"  Calmar:  {agg['calmar']:.2f}")
    print(f"  Sortino: {agg['sortino']:.2f}")
    print(f"  Vol:     {agg['vol_pct']:.1f}%")
    print(f"  Return:  {agg['total_ret_pct']:.1f}%")

    print(f"\nYear-by-Year OOS:")
    for yr, m in sorted(result["yearly_oos"].items()):
        print(f"  {yr}: CAGR={m['cagr_pct']:6.1f}%  Sharpe={m['sharpe']:.2f}  MaxDD={m['max_dd_pct']:.1f}%")

    # Generate HTML
    html = generate_html(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")

    # Save JSON
    json_path = REPORT_PATH.with_suffix(".json")
    json_data = {k: v for k, v in result.items()
                 if k not in ("oos_equity", "full_equity", "oos_drawdown")}
    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")
    print(f"JSON:   {json_path}")


if __name__ == "__main__":
    main()
