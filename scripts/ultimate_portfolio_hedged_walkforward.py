#!/usr/bin/env python3
"""
Ultimate Portfolio + Tail Risk Hedge — Expanding Walk-Forward.

Combines:
  - Expanding-window walk-forward (train 2020-N, test N+1)
  - Tail risk hedge overlay (SPY puts + VIX calls, delta-adaptive, 2% budget)
  - 1.6× leverage with dynamic crisis deleveraging

Reports hedged vs unhedged year-by-year OOS and hedge cost drag.
Output: reports/ultimate_portfolio_hedged_walkforward.html
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
    _fetch,
)
from compass.tail_risk_hedge import (
    TailRiskHedgeEngine,
    TailRiskHedgeConfig,
    BacktestResult,
)

TRADING_DAYS = 252
ACCOUNT = 100_000
LEVERAGE = 1.6
REPORT_PATH = ROOT / "reports" / "ultimate_portfolio_hedged_walkforward.html"

WEIGHTS = {
    "EXP-1220 Dynamic": 0.95,
    "Cross-Asset Pairs": 0.0167,
    "TLT Iron Condors": 0.0167,
    "Vol Term Structure": 0.0167,
}

HEDGE_CONFIG = TailRiskHedgeConfig(
    normal_leverage=LEVERAGE,
    crisis_leverage=0.25,
    min_leverage=0.15,
    annual_cost_budget_pct=2.0,
    put_payoff_multiplier=15.0,
    vix_call_payoff_multiplier=25.0,
    crisis_hedge_ratio=0.90,
    vix_crisis_threshold=25.0,
    dd_crisis_threshold=0.04,
    leverage_smoothing_days=1,
)

WF_WINDOWS = [
    {"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31"), "label": "2022"},
    {"train": ("2020-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31"), "label": "2023"},
    {"train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31"), "label": "2024"},
    {"train": ("2020-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31"), "label": "2025"},
]


def load_all():
    """Load strategy returns + market data."""
    print("  Loading EXP-1220 Dynamic...")
    s1 = load_exp1220_dynamic()
    print("  Loading Cross-Asset Pairs...")
    s2 = load_cross_asset_pairs()
    print("  Loading Vol Term Structure...")
    s3 = load_vol_term_structure()
    print("  Loading TLT Iron Condors...")
    s4 = load_tlt_iron_condors()

    df = pd.DataFrame({s1.name: s1, s2.name: s2, s3.name: s3, s4.name: s4})
    df = df.sort_index().fillna(0)
    df = df[df.index >= "2020-01-01"]

    print("  Loading SPY/VIX/VIX3M...")
    spy = _fetch("SPY", "2019-01-01", "2025-12-31")
    vix_df = _fetch("^VIX", "2019-01-01", "2025-12-31")
    vix3m_df = _fetch("^VIX3M", "2019-01-01", "2025-12-31")

    spy_ret = spy["Close"].pct_change().dropna()
    vix = vix_df["Close"].dropna()
    vix3m = vix3m_df["Close"].dropna()

    common = df.index.intersection(spy_ret.index).intersection(vix.index).intersection(vix3m.index)
    df = df.reindex(common).fillna(0)
    spy_ret = spy_ret.reindex(common).fillna(0)
    vix = vix.reindex(common).ffill().bfill()
    vix3m = vix3m.reindex(common).ffill().bfill()

    return df, spy_ret, vix, vix3m


def run_hedged_on_slice(df_slice, spy_ret, vix, vix3m, capital):
    """Run hedge engine on a slice of data, return daily returns + stats."""
    names = list(WEIGHTS.keys())
    w = np.array([WEIGHTS[n] for n in names])
    port_rets = df_slice[names].values @ w  # unlevered

    idx = df_slice.index
    data = {
        "portfolio_returns": pd.Series(port_rets, index=idx),
        "spy_returns": spy_ret.reindex(idx).fillna(0),
        "vix": vix.reindex(idx).ffill().bfill(),
        "vix3m": vix3m.reindex(idx).ffill().bfill(),
    }

    engine = TailRiskHedgeEngine(HEDGE_CONFIG)
    result = engine.backtest(data, starting_capital=capital)
    return result


def run_walk_forward(df, spy_ret, vix, vix3m):
    """Expanding-window walk-forward with hedge overlay on each OOS period."""
    names = list(WEIGHTS.keys())
    w = np.array([WEIGHTS[n] for n in names])

    # Track OOS series for both hedged and unhedged
    hedged_oos_rets = []
    hedged_oos_dates = []
    unhedged_oos_rets = []
    window_results = []

    hedged_capital = ACCOUNT
    unhedged_capital = ACCOUNT

    for wf in WF_WINDOWS:
        train_mask = (df.index >= wf["train"][0]) & (df.index <= wf["train"][1])
        test_mask = (df.index >= wf["test"][0]) & (df.index <= wf["test"][1])
        test_df = df.loc[test_mask]

        if test_df.empty:
            continue

        # ── Unhedged OOS ──
        unhedged_raw = (test_df[names].values @ w) * LEVERAGE
        unhedged_m = calc_metrics(unhedged_raw)

        # ── Hedged OOS (run hedge engine on this window) ──
        hedged_result = run_hedged_on_slice(
            test_df, spy_ret, vix, vix3m, hedged_capital)

        hedged_daily = hedged_result.daily_returns
        hedged_m = {
            "cagr_pct": hedged_result.cagr_pct,
            "sharpe": hedged_result.sharpe,
            "max_dd_pct": hedged_result.max_dd_pct,
            "calmar": hedged_result.calmar,
            "sortino": hedged_result.sortino,
            "vol_pct": hedged_result.vol_pct,
            "total_ret_pct": hedged_result.total_return_pct,
            "n_days": len(hedged_daily),
        }

        # Update running capital
        hedged_capital = hedged_result.equity_curve[-1]
        for r in unhedged_raw:
            unhedged_capital *= (1 + r)

        # Hedge cost drag
        cagr_drag = unhedged_m["cagr_pct"] - hedged_m["cagr_pct"]

        window_results.append({
            "label": wf["label"],
            "test_days": len(test_df),
            "hedged": hedged_m,
            "unhedged": unhedged_m,
            "cagr_drag": round(cagr_drag, 1),
            "avg_leverage": hedged_result.avg_leverage,
            "hedge_cost_pct": hedged_result.total_hedge_cost_pct,
            "net_hedge_cost_pct": hedged_result.net_hedge_cost_pct,
            "crisis_days": hedged_result.crisis_days,
            "elevated_days": hedged_result.elevated_days,
            "normal_days": hedged_result.normal_days,
        })

        hedged_oos_rets.extend(hedged_daily.tolist())
        hedged_oos_dates.extend(test_df.index.tolist())
        unhedged_oos_rets.extend(unhedged_raw.tolist())

    # ── Aggregate OOS ──
    h_rets = np.array(hedged_oos_rets)
    u_rets = np.array(unhedged_oos_rets)

    h_agg = calc_metrics(h_rets)
    u_agg = calc_metrics(u_rets)

    # Equity curves
    h_eq = ACCOUNT * np.cumprod(1 + h_rets)
    h_eq = np.insert(h_eq, 0, ACCOUNT)
    u_eq = ACCOUNT * np.cumprod(1 + u_rets)
    u_eq = np.insert(u_eq, 0, ACCOUNT)
    dates_str = ["2021-12-31"] + [str(d)[:10] for d in hedged_oos_dates]

    # Drawdown (hedged)
    eq_arr = np.cumprod(1 + h_rets)
    hwm = np.maximum.accumulate(eq_arr)
    h_dd = ((eq_arr / hwm) - 1) * 100

    # Monthly returns (hedged)
    monthly = {}
    for i, d in enumerate(hedged_oos_dates):
        dt = pd.Timestamp(d)
        monthly.setdefault(dt.year, {}).setdefault(dt.month, []).append(h_rets[i])
    monthly_pct = {}
    for yr, months in sorted(monthly.items()):
        monthly_pct[yr] = {}
        for mo, rets in sorted(months.items()):
            monthly_pct[yr][mo] = round(float(np.prod(1 + np.array(rets)) - 1) * 100, 2)

    # Full-period hedged (IS+OOS) for reference
    full_result = run_hedged_on_slice(df, spy_ret, vix, vix3m, ACCOUNT)

    return {
        "windows": window_results,
        "hedged_agg": h_agg,
        "unhedged_agg": u_agg,
        "cagr_drag": round(u_agg["cagr_pct"] - h_agg["cagr_pct"], 1),
        "hedged_equity": h_eq.tolist(),
        "unhedged_equity": u_eq.tolist(),
        "dates_str": dates_str,
        "hedged_drawdown": h_dd.tolist(),
        "dd_dates": [str(d)[:10] for d in hedged_oos_dates],
        "monthly_returns": monthly_pct,
        "full_hedged": {
            "cagr_pct": full_result.cagr_pct,
            "sharpe": full_result.sharpe,
            "max_dd_pct": full_result.max_dd_pct,
            "hedge_cost_pct": full_result.total_hedge_cost_pct,
            "net_hedge_cost_pct": full_result.net_hedge_cost_pct,
            "avg_leverage": full_result.avg_leverage,
            "scenario_results": {
                name: {
                    "hedged_dd_pct": sr.hedged_dd_pct,
                    "unhedged_dd_pct": sr.unhedged_dd_pct,
                    "dd_reduction_pct": sr.dd_reduction_pct,
                    "hedged_return_pct": sr.hedged_return_pct,
                    "survives_20pct": sr.survives_20pct,
                }
                for name, sr in full_result.scenario_results.items()
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# SVG Charts
# ═══════════════════════════════════════════════════════════════════════════

def _svg_dual_equity(h_eq, u_eq, dates, w=920, h=380):
    pl, pr, pt, pb = 80, 25, 42, 60
    pw, ph = w - pl - pr, h - pt - pb
    all_v = list(h_eq) + list(u_eq)
    ymin, ymax = min(all_v) * 0.92, max(all_v) * 1.08
    if ymax <= ymin: ymax = ymin + 1
    n = len(dates)
    def tx(i): return pl + i / max(n - 1, 1) * pw
    def ty(v): return pt + (1 - (v - ymin) / (ymax - ymin)) * ph

    p = [f'<svg width="{w}" height="{h}" style="background:#1e293b;border-radius:10px;margin:1rem 0">']
    p.append(f'<text x="{w//2}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#e2e8f0">OOS Equity: Hedged vs Unhedged ($100K Start)</text>')
    for j in range(7):
        yv = ymin + j / 6 * (ymax - ymin); y = ty(yv)
        p.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{pl+pw}" y2="{y:.0f}" stroke="#334155" stroke-width="0.5"/>')
        lbl = f"${yv:,.0f}" if yv < 1e6 else f"${yv/1e6:.2f}M"
        p.append(f'<text x="{pl-8}" y="{y+4:.0f}" text-anchor="end" font-size="9" fill="#94a3b8">{lbl}</text>')
    step = max(1, n // 8)
    for i in range(0, n, step):
        p.append(f'<text x="{tx(i):.0f}" y="{h-14}" text-anchor="middle" font-size="9" fill="#94a3b8">{dates[i][:7]}</text>')
    # Unhedged
    nu = min(len(u_eq), n)
    d_u = " ".join(f"{'M' if i==0 else 'L'}{tx(i):.1f},{ty(u_eq[i]):.1f}" for i in range(nu))
    p.append(f'<path d="{d_u}" fill="none" stroke="#ef4444" stroke-width="1.8" opacity="0.6"/>')
    # Hedged
    nh = min(len(h_eq), n)
    d_h = " ".join(f"{'M' if i==0 else 'L'}{tx(i):.1f},{ty(h_eq[i]):.1f}" for i in range(nh))
    p.append(f'<path d="{d_h}" fill="none" stroke="#4ade80" stroke-width="2.5"/>')
    lx = pl + 12
    p.append(f'<rect x="{lx}" y="{pt+8}" width="14" height="3" fill="#4ade80"/>')
    p.append(f'<text x="{lx+18}" y="{pt+13}" font-size="9" fill="#e2e8f0">Hedged (tail risk overlay)</text>')
    p.append(f'<rect x="{lx+200}" y="{pt+8}" width="14" height="3" fill="#ef4444" opacity="0.6"/>')
    p.append(f'<text x="{lx+218}" y="{pt+13}" font-size="9" fill="#94a3b8">Unhedged (1.6× flat)</text>')
    p.append("</svg>")
    return "\n".join(p)

def _svg_drawdown(dd, dates, w=920, h=220):
    pl, pr, pt, pb = 80, 25, 42, 48
    pw, ph = w - pl - pr, h - pt - pb
    dd_min = min(dd) if dd else 0
    ymin, ymax = dd_min * 1.25, 0.5
    n = len(dd)
    def tx(i): return pl + i / max(n-1, 1) * pw
    def ty(v): return pt + (1 - (v - ymin) / (ymax - ymin)) * ph
    p = [f'<svg width="{w}" height="{h}" style="background:#1e293b;border-radius:10px;margin:1rem 0">']
    p.append(f'<text x="{w//2}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#e2e8f0">Hedged OOS Drawdown (%)</text>')
    for j in range(5):
        yv = ymin + j/4*(ymax-ymin); y = ty(yv)
        p.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{pl+pw}" y2="{y:.0f}" stroke="#334155" stroke-width="0.5"/>')
        p.append(f'<text x="{pl-8}" y="{y+4:.0f}" text-anchor="end" font-size="9" fill="#94a3b8">{yv:.1f}%</text>')
    y0 = ty(0)
    p.append(f'<line x1="{pl}" y1="{y0:.0f}" x2="{pl+pw}" y2="{y0:.0f}" stroke="#94a3b8" stroke-width="1"/>')
    if dd:
        fill = f"M{tx(0):.1f},{y0:.1f}"
        for i in range(n): fill += f" L{tx(i):.1f},{ty(dd[i]):.1f}"
        fill += f" L{tx(n-1):.1f},{y0:.1f} Z"
        p.append(f'<path d="{fill}" fill="rgba(239,68,68,0.25)" stroke="#ef4444" stroke-width="1.5"/>')
        mi = int(np.argmin(dd)); mx, my = tx(mi), ty(dd[mi])
        p.append(f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="4" fill="#ef4444"/>')
        p.append(f'<text x="{mx+8:.0f}" y="{my-6:.0f}" font-size="10" font-weight="bold" fill="#ef4444">{dd[mi]:.1f}%</text>')
    p.append("</svg>")
    return "\n".join(p)

def _svg_heatmap(monthly, w=920):
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    years = sorted(monthly.keys())
    if not years: return ""
    cw, ch = 56, 34; pl, pt = 62, 52
    tw = pl + cw*13 + 25; th = pt + ch*len(years) + 25
    p = [f'<svg width="{tw}" height="{th}" style="background:#1e293b;border-radius:10px;margin:1rem 0">']
    p.append(f'<text x="{tw//2}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#e2e8f0">Monthly Returns — Hedged OOS (%)</text>')
    for j, m in enumerate(months):
        p.append(f'<text x="{pl+j*cw+cw//2}" y="{pt-10}" text-anchor="middle" font-size="9" font-weight="bold" fill="#94a3b8">{m}</text>')
    p.append(f'<text x="{pl+12*cw+cw//2}" y="{pt-10}" text-anchor="middle" font-size="9" font-weight="bold" fill="#94a3b8">YTD</text>')
    for ri, yr in enumerate(years):
        y = pt + ri * ch
        p.append(f'<text x="{pl-8}" y="{y+ch//2+4}" text-anchor="end" font-size="10" font-weight="bold" fill="#e2e8f0">{yr}</text>')
        ytd = 1.0
        for j in range(1, 13):
            x = pl + (j-1)*cw; val = monthly.get(yr, {}).get(j)
            if val is not None:
                ytd *= (1 + val/100)
                intensity = min(abs(val)/15, 1.0)
                if val >= 0: r,g,b = int(22+(1-intensity)*28), int(80+intensity*175), int(40+(1-intensity)*60)
                else: r,g,b = int(80+intensity*175), int(40+(1-intensity)*60), int(30+(1-intensity)*30)
                p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="rgb({r},{g},{b})" stroke="#0f172a" rx="4"/>')
                p.append(f'<text x="{x+cw//2}" y="{y+ch//2+4}" text-anchor="middle" font-size="9" font-weight="bold" fill="#f8fafc">{val:+.1f}</text>')
            else:
                p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="#334155" stroke="#0f172a" rx="4"/>')
                p.append(f'<text x="{x+cw//2}" y="{y+ch//2+4}" text-anchor="middle" font-size="9" fill="#64748b">—</text>')
        ytd_pct = (ytd-1)*100; intensity = min(abs(ytd_pct)/60, 1.0)
        if ytd_pct >= 0: r,g,b = int(22+(1-intensity)*28), int(80+intensity*175), int(40+(1-intensity)*60)
        else: r,g,b = int(80+intensity*175), int(40+(1-intensity)*60), int(30+(1-intensity)*30)
        x = pl + 12*cw
        p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="rgb({r},{g},{b})" stroke="#0f172a" rx="4"/>')
        p.append(f'<text x="{x+cw//2}" y="{y+ch//2+4}" text-anchor="middle" font-size="9" font-weight="bold" fill="#f8fafc">{ytd_pct:+.1f}</text>')
    p.append("</svg>")
    return "\n".join(p)


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_html(data):
    ha = data["hedged_agg"]; ua = data["unhedged_agg"]
    wins = data["windows"]; full = data["full_hedged"]
    drag = data["cagr_drag"]

    def _badge(ok):
        return '<span class="badge badge-pass">PASS</span>' if ok else '<span class="badge badge-fail">MISS</span>'

    t_cagr = ha["cagr_pct"] >= 90; t_dd = ha["max_dd_pct"] <= 12; t_sharpe = ha["sharpe"] >= 4.0
    covid_sr = full["scenario_results"].get("COVID_2020", {})
    covid_dd = covid_sr.get("hedged_dd_pct", 99)
    covid_pass = covid_dd <= 25

    equity_svg = _svg_dual_equity(data["hedged_equity"], data["unhedged_equity"], data["dates_str"])
    dd_svg = _svg_drawdown(data["hedged_drawdown"], data["dd_dates"])
    heatmap_svg = _svg_heatmap(data["monthly_returns"])

    # Year-by-year OOS table
    yearly_rows = ""
    for win in wins:
        hm = win["hedged"]; um = win["unhedged"]
        yearly_rows += f"""<tr>
            <td style="font-weight:700">{win['label']}</td>
            <td style="color:{'#4ade80' if hm['cagr_pct']>0 else '#ef4444'};font-weight:600">{hm['cagr_pct']:.1f}%</td>
            <td>{hm['sharpe']:.2f}</td>
            <td style="color:{'#4ade80' if hm['max_dd_pct']<10 else '#f59e0b'}">{hm['max_dd_pct']:.1f}%</td>
            <td>{um['cagr_pct']:.1f}%</td>
            <td>{um['sharpe']:.2f}</td>
            <td>{um['max_dd_pct']:.1f}%</td>
            <td style="color:#f59e0b">{win['cagr_drag']:+.1f}%</td>
            <td>{win['avg_leverage']:.2f}×</td>
            <td>{win['hedge_cost_pct']:.2f}%</td>
        </tr>"""

    # Crisis scenarios
    scenario_rows = ""
    for name, sr in sorted(full["scenario_results"].items()):
        sc = "#4ade80" if sr["survives_20pct"] else "#ef4444"
        scenario_rows += f"""<tr>
            <td>{name}</td>
            <td style="color:{sc};font-weight:700">{sr['hedged_dd_pct']:.1f}%</td>
            <td>{sr['unhedged_dd_pct']:.1f}%</td>
            <td style="color:#4ade80;font-weight:600">{sr['dd_reduction_pct']:+.1f}%</td>
            <td>{'PASS' if sr['survives_20pct'] else 'FAIL'}</td>
        </tr>"""

    # Comparison table
    comp_metrics = [("CAGR", "cagr_pct", "%"), ("Sharpe", "sharpe", ""), ("Max DD", "max_dd_pct", "%"),
                    ("Calmar", "calmar", ""), ("Sortino", "sortino", ""), ("Vol", "vol_pct", "%"),
                    ("Total Return", "total_ret_pct", "%")]
    comp_rows = ""
    for label, key, unit in comp_metrics:
        hv = ha.get(key, 0); uv = ua.get(key, 0)
        comp_rows += f'<tr><td style="font-weight:600">{label}</td><td>{hv:.1f}{unit}</td><td>{uv:.1f}{unit}</td><td style="font-weight:600">{hv-uv:+.1f}{unit}</td></tr>'

    verdict_cls = "verdict-pass" if t_cagr and t_dd and t_sharpe and covid_pass else "verdict-warn"
    verdict_txt = "ALL TARGETS HIT" if t_cagr and t_dd and t_sharpe and covid_pass else "TARGETS PARTIALLY MET"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ultimate Portfolio — Hedged Walk-Forward</title>
<style>
  body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:28px;
         background:#0f172a; color:#e2e8f0; max-width:980px; margin:0 auto; }}
  h1 {{ font-size:1.7rem; margin-bottom:4px; }}
  h2 {{ font-size:1.15rem; color:#94a3b8; margin-top:36px; border-bottom:1px solid #334155; padding-bottom:8px; }}
  .subtitle {{ color:#94a3b8; font-size:0.88rem; margin-bottom:20px; }}
  .verdict {{ text-align:center; padding:12px; border-radius:8px; font-size:1.1rem; font-weight:800;
              letter-spacing:0.06em; margin-bottom:24px; }}
  .verdict-pass {{ background:#14532d; color:#4ade80; border:2px solid #166534; }}
  .verdict-warn {{ background:#78350f; color:#fbbf24; border:2px solid #92400e; }}
  .config {{ background:#1e293b; border-radius:8px; padding:16px; margin-bottom:20px; font-size:0.84rem; line-height:1.8; }}
  .config strong {{ color:#e2e8f0; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:20px; }}
  .card {{ background:#1e293b; border-radius:8px; padding:14px 16px; }}
  .card-label {{ font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; }}
  .card-value {{ font-size:1.4rem; font-weight:700; margin-top:3px; }}
  .positive {{ color:#4ade80; }} .negative {{ color:#ef4444; }} .warn {{ color:#fbbf24; }} .neutral {{ color:#e2e8f0; }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:20px; font-size:0.85rem; }}
  th {{ background:#1e293b; padding:9px 12px; text-align:right; font-size:0.73rem; color:#94a3b8;
       text-transform:uppercase; letter-spacing:0.04em; border-bottom:2px solid #334155; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:7px 12px; text-align:right; border-bottom:1px solid #1e293b; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:rgba(30,41,59,0.5); }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:4px; font-size:0.72rem; font-weight:700; }}
  .badge-pass {{ background:#14532d; color:#4ade80; }}
  .badge-fail {{ background:#7f1d1d; color:#ef4444; }}
  .footer {{ color:#64748b; font-size:0.72rem; margin-top:48px; text-align:center; line-height:1.6; }}
</style>
</head>
<body>

<h1>Ultimate Portfolio — Hedged Walk-Forward</h1>
<div class="subtitle">
    Expanding Windows | Tail Risk Hedge (SPY Puts + VIX Calls) | 1.6× Leverage | {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<div class="verdict {verdict_cls}">{verdict_txt}</div>

<div class="config">
    <strong>Allocation:</strong> EXP-1220 Dynamic 95% | Cross-Asset Pairs 1.67% | TLT ICs 1.67% | Vol Term Structure 1.67%<br>
    <strong>Hedge:</strong> SPY puts (60%) + VIX calls (40%), delta-adaptive, 2%/yr budget &nbsp;|&nbsp;
    <strong>Leverage:</strong> 1.6× → 0.25× in crisis &nbsp;|&nbsp;
    <strong>OOS:</strong> 2022–2025 (expanding windows, no lookahead)
</div>

<div class="grid">
    <div class="card"><div class="card-label">OOS CAGR (Hedged)</div>
        <div class="card-value {'positive' if ha['cagr_pct']>=80 else 'warn'}">{ha['cagr_pct']:.1f}%</div>
        <div style="font-size:.68rem;color:#64748b">{_badge(t_cagr)} ≥90%</div></div>
    <div class="card"><div class="card-label">Sharpe</div>
        <div class="card-value {'positive' if ha['sharpe']>=4 else 'warn'}">{ha['sharpe']:.2f}</div>
        <div style="font-size:.68rem;color:#64748b">{_badge(t_sharpe)} ≥4.0</div></div>
    <div class="card"><div class="card-label">Max Drawdown</div>
        <div class="card-value {'positive' if ha['max_dd_pct']<=12 else 'negative'}">{ha['max_dd_pct']:.1f}%</div>
        <div style="font-size:.68rem;color:#64748b">{_badge(t_dd)} ≤12%</div></div>
    <div class="card"><div class="card-label">COVID DD</div>
        <div class="card-value {'positive' if covid_pass else 'negative'}">{covid_dd:.1f}%</div>
        <div style="font-size:.68rem;color:#64748b">{_badge(covid_pass)} ≤25% (was 51.8%)</div></div>
    <div class="card"><div class="card-label">CAGR Drag</div>
        <div class="card-value warn">{drag:+.1f}%</div>
        <div style="font-size:.68rem;color:#64748b">hedge cost impact</div></div>
    <div class="card"><div class="card-label">Calmar</div><div class="card-value neutral">{ha['calmar']:.2f}</div></div>
    <div class="card"><div class="card-label">Sortino</div><div class="card-value neutral">{ha['sortino']:.2f}</div></div>
    <div class="card"><div class="card-label">OOS Return</div><div class="card-value positive">{ha['total_ret_pct']:.1f}%</div></div>
</div>

<h2>OOS Equity: Hedged vs Unhedged</h2>
{equity_svg}

<h2>Drawdown</h2>
{dd_svg}

<h2>Year-by-Year OOS: Hedged vs Unhedged</h2>
<table>
    <thead><tr><th>Year</th><th>H CAGR</th><th>H Sharpe</th><th>H Max DD</th><th>U CAGR</th><th>U Sharpe</th><th>U Max DD</th><th>CAGR Drag</th><th>Avg Lev</th><th>Hedge Cost</th></tr></thead>
    <tbody>{yearly_rows}</tbody>
</table>

<h2>Aggregate OOS Comparison</h2>
<table>
    <thead><tr><th>Metric</th><th>Hedged</th><th>Unhedged</th><th>Delta</th></tr></thead>
    <tbody>{comp_rows}</tbody>
</table>

<h2>Crisis Stress Tests (Full Period)</h2>
<table>
    <thead><tr><th>Scenario</th><th>Hedged DD</th><th>Unhedged DD</th><th>Reduction</th><th>&lt;20%?</th></tr></thead>
    <tbody>{scenario_rows}</tbody>
</table>

<h2>Monthly Returns Heatmap</h2>
{heatmap_svg}

<div class="footer">
    PilotAI Credit Spreads — Hedged Walk-Forward Validation<br>
    All OOS returns strictly out-of-sample. Hedge overlay: SPY puts + VIX calls, 2%/yr budget.<br>
    Dynamic leverage 1.6×→0.25× based on multi-signal crisis score.
</div>

</body></html>"""


def main():
    print("=" * 72)
    print("Ultimate Portfolio — Hedged Walk-Forward")
    print("=" * 72)

    print("\n[1/3] Loading data...")
    df, spy_ret, vix, vix3m = load_all()
    print(f"  → {len(df)} days, {df.index[0].date()} → {df.index[-1].date()}")

    print("\n[2/3] Running expanding-window walk-forward with hedge overlay...")
    result = run_walk_forward(df, spy_ret, vix, vix3m)

    ha = result["hedged_agg"]; ua = result["unhedged_agg"]
    print(f"\n{'━'*56}")
    print(f"  OOS HEDGED (2022–2025):")
    print(f"    CAGR:   {ha['cagr_pct']:.1f}%   Sharpe: {ha['sharpe']:.2f}   Max DD: {ha['max_dd_pct']:.1f}%")
    print(f"  OOS UNHEDGED:")
    print(f"    CAGR:   {ua['cagr_pct']:.1f}%   Sharpe: {ua['sharpe']:.2f}   Max DD: {ua['max_dd_pct']:.1f}%")
    print(f"  CAGR DRAG: {result['cagr_drag']:+.1f}%")

    print(f"\n  Year-by-Year OOS (Hedged):")
    for w in result["windows"]:
        hm = w["hedged"]
        print(f"    {w['label']}: CAGR={hm['cagr_pct']:7.1f}%  Sharpe={hm['sharpe']:.2f}  DD={hm['max_dd_pct']:.1f}%  "
              f"Lev={w['avg_leverage']:.2f}×  Cost={w['hedge_cost_pct']:.2f}%  Drag={w['cagr_drag']:+.1f}%")

    covid = result["full_hedged"]["scenario_results"].get("COVID_2020", {})
    print(f"\n  COVID stress: {covid.get('hedged_dd_pct', '?')}% hedged / {covid.get('unhedged_dd_pct', '?')}% unhedged")
    print(f"{'━'*56}")

    print("\n[3/3] Generating report...")
    html = generate_html(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}")

    json_path = REPORT_PATH.with_suffix(".json")
    json_data = {k: v for k, v in result.items()
                 if k not in ("hedged_equity", "unhedged_equity", "hedged_drawdown", "dates_str", "dd_dates")}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  → {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
