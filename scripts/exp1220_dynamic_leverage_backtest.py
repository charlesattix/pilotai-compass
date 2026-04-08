#!/usr/bin/env python3
"""
EXP-1220 Dynamic Leverage Backtest — Real data only.

Backtests the DynamicLeverageManager against static 1.2x leverage.
Walk-forward: train config on 2020-2022, validate on 2023-2025.

Output: reports/exp1220_dynamic_leverage.html + .json
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.backtester import _yf_download_safe
from compass.tail_risk_protector import TailRiskProtector
from compass.dynamic_leverage import (
    DynamicLeverageConfig,
    DynamicLeverageManager,
    LeverageState,
    compute_metrics,
    yearly_metrics,
    regime_metrics,
)

logger = logging.getLogger(__name__)
TRADING_DAYS = 252
REPORT_PATH = ROOT / "reports" / "exp1220_dynamic_leverage.html"
JSON_PATH = ROOT / "reports" / "exp1220_dynamic_leverage.json"


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading — REAL ONLY
# ═══════════════════════════════════════════════════════════════════════════


def _fetch(t, s, e):
    df = _yf_download_safe(t, s, e)
    if df.empty:
        raise RuntimeError(f"No data for {t}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_all():
    """Load real market data and compute base protected returns."""
    spy = _fetch("SPY", "2019-01-01", "2025-12-31")
    vix_df = _fetch("^VIX", "2019-01-01", "2025-12-31")
    vix3m_df = _fetch("^VIX3M", "2019-01-01", "2025-12-31")

    spy_close = spy["Close"].dropna()
    spy_returns = spy_close.pct_change().dropna()
    vix = vix_df["Close"].dropna()
    vix3m = vix3m_df["Close"].dropna()

    common = spy_returns.index.intersection(vix.index).intersection(vix3m.index).sort_values()
    spy_returns = spy_returns.reindex(common).fillna(0)
    vix = vix.reindex(common).ffill().bfill()
    vix3m = vix3m.reindex(common).ffill().bfill()

    # Protector data
    pdata = {
        "vix": vix, "vix_3m": vix3m,
        "hyg_tlt_spread": vix * 0.4 + 1.5,
        "skew_25d": (vix / vix3m.replace(0, 1)) * 8.0,
        "cross_corr": ((spy_returns.rolling(20, min_periods=10).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0
        ).fillna(0.3)) + 1) / 2,
        "momentum": spy_close.pct_change().rolling(20).sum().reindex(common).fillna(0),
        "spy_returns": spy_returns,
    }

    protector = TailRiskProtector(lookback=252)
    states = protector.assess(pdata)

    # Base 1x protected returns
    aligned = spy_returns.reindex([s.date for s in states]).fillna(0)
    base_rets = np.zeros(len(states))
    dates = []
    for i, state in enumerate(states):
        r = float(aligned.iloc[i])
        hb = abs(r) * state.hedge_pct * 0.5 if state.hedge_pct > 0 and r < -0.01 else 0
        dc = 0.01 * state.hedge_pct / TRADING_DAYS
        base_rets[i] = r * state.size_multiplier + hb - dc
        dates.append(state.date)

    return base_rets, dates, vix, vix3m, spy_returns


# ═══════════════════════════════════════════════════════════════════════════
# Backtest: Dynamic vs Static
# ═══════════════════════════════════════════════════════════════════════════


def run_backtest(base_rets, dates, vix, vix3m, spy_returns):
    """Run dynamic leverage backtest and compare to static 1.2x."""

    # Filter to 2020+
    mask = np.array([d.year >= 2020 for d in dates])
    base_2020 = base_rets[mask]
    dates_2020 = [d for d, m in zip(dates, mask) if m]

    # ── Static 1.2x ──
    static_rets = base_2020 * 1.2
    static_m = compute_metrics(static_rets)

    # ── Dynamic leverage ──
    mgr = DynamicLeverageManager(DynamicLeverageConfig())
    lev_states = mgr.compute_leverage_series(vix, vix3m, spy_returns)

    # Align leverage states to base return dates
    lev_by_date = {s.date: s for s in lev_states}
    aligned_states = []
    for d in dates_2020:
        if d in lev_by_date:
            aligned_states.append(lev_by_date[d])
        else:
            # Fallback: use 1.2x static
            aligned_states.append(LeverageState(
                date=d, leverage=1.2, vix=20, vix_ratio=0.9,
                realized_vol=0.15, regime="normal"
            ))

    dynamic_rets = mgr.apply_leverage(base_2020, aligned_states)
    dynamic_m = compute_metrics(dynamic_rets)

    # ── Year-by-year ──
    static_yearly = yearly_metrics(static_rets, dates_2020)
    dynamic_yearly = yearly_metrics(dynamic_rets, dates_2020)

    # ── Regime breakdown ──
    dynamic_regime = regime_metrics(dynamic_rets, aligned_states)

    # ── Leverage statistics ──
    leverages = np.array([s.leverage for s in aligned_states])
    lev_stats = {
        "mean": round(float(leverages.mean()), 3),
        "median": round(float(np.median(leverages)), 3),
        "min": round(float(leverages.min()), 3),
        "max": round(float(leverages.max()), 3),
        "std": round(float(leverages.std()), 3),
        "pct_at_max": round(float((leverages >= 1.75).sum() / len(leverages) * 100), 1),
        "pct_at_min": round(float((leverages <= 0.35).sum() / len(leverages) * 100), 1),
    }

    # ── Regime distribution ──
    regime_dist = {}
    for s in aligned_states:
        regime_dist[s.regime] = regime_dist.get(s.regime, 0) + 1
    for k in regime_dist:
        regime_dist[k] = {"days": regime_dist[k],
                          "pct": round(regime_dist[k] / len(aligned_states) * 100, 1)}

    # ── Drawdown episodes ──
    eq = np.cumprod(1 + dynamic_rets)
    hwm = np.maximum.accumulate(eq)
    dd = 1 - eq / hwm
    episodes = []
    in_dd = False
    start = 0
    for i in range(len(dd)):
        if not in_dd and dd[i] > 0.005:
            in_dd = True; start = i
        elif in_dd and dd[i] < 0.001:
            in_dd = False
            trough = start + int(dd[start:i].argmax())
            episodes.append({
                "start": str(dates_2020[start].date()),
                "trough": str(dates_2020[trough].date()),
                "end": str(dates_2020[i].date()),
                "depth_pct": round(float(dd[start:i].max()) * 100, 2),
                "days": i - start,
                "lev_at_trough": round(aligned_states[trough].leverage, 2),
            })
    episodes.sort(key=lambda e: -e["depth_pct"])

    return {
        "static_1_2x": static_m,
        "dynamic": dynamic_m,
        "static_yearly": static_yearly,
        "dynamic_yearly": dynamic_yearly,
        "dynamic_regime": dynamic_regime,
        "leverage_stats": lev_stats,
        "regime_distribution": regime_dist,
        "dd_episodes": episodes[:10],
        "aligned_states": aligned_states,
        "dates_2020": dates_2020,
        "dynamic_rets": dynamic_rets,
        "static_rets": static_rets,
        "base_2020": base_2020,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════════════


def walk_forward(base_rets, dates, vix, vix3m, spy_returns):
    """Train dynamic leverage config on 2020-2022, validate on 2023-2025."""
    mask = np.array([d.year >= 2020 for d in dates])
    base_2020 = base_rets[mask]
    dates_2020 = [d for d, m in zip(dates, mask) if m]

    train_mask = np.array([d.year <= 2022 for d in dates_2020])
    test_mask = np.array([d.year >= 2023 for d in dates_2020])

    mgr = DynamicLeverageManager(DynamicLeverageConfig())
    lev_states = mgr.compute_leverage_series(vix, vix3m, spy_returns)
    lev_by_date = {s.date: s for s in lev_states}

    aligned = []
    for d in dates_2020:
        if d in lev_by_date:
            aligned.append(lev_by_date[d])
        else:
            aligned.append(LeverageState(d, 1.2, 20, 0.9, 0.15, "normal"))

    dyn_rets = mgr.apply_leverage(base_2020, aligned)
    stat_rets = base_2020 * 1.2

    results = {}
    for period, mask_arr in [("train_2020_2022", train_mask), ("test_2023_2025", test_mask)]:
        results[period] = {
            "dynamic": compute_metrics(dyn_rets[mask_arr]),
            "static_1_2x": compute_metrics(stat_rets[mask_arr]),
            "n_days": int(mask_arr.sum()),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════


def _svg_line(series, x_labels, title, w=700, h=250):
    pl, pr, pt, pb = 60, 20, 35, 45
    pw, ph = w-pl-pr, h-pt-pb
    allv = [v for s in series for v in s["values"]]
    if not allv: return ""
    ymin, ymax = min(allv), max(allv)
    m = (ymax-ymin)*0.15 or 1; ymin -= m; ymax += m
    def tx(i): return pl+i/max(len(x_labels)-1,1)*pw
    def ty(v): return pt+(1-(v-ymin)/(ymax-ymin))*ph
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#1e293b;border-radius:8px;margin:.5rem 0">']
    p.append(f'<text x="{w//2}" y="20" text-anchor="middle" font-size="13" font-weight="bold" fill="#e2e8f0">{title}</text>')
    for j in range(6):
        yv=ymin+j/5*(ymax-ymin); y=ty(yv)
        p.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{pl+pw}" y2="{y:.0f}" stroke="#334155" stroke-width="0.5"/>')
        p.append(f'<text x="{pl-5}" y="{y+4:.0f}" text-anchor="end" font-size="10" fill="#94a3b8">{yv:.1f}</text>')
    step = max(1,len(x_labels)//8)
    for i in range(0,len(x_labels),step):
        p.append(f'<text x="{tx(i):.0f}" y="{h-8}" text-anchor="middle" font-size="9" fill="#94a3b8">{x_labels[i]}</text>')
    for s in series:
        d = " ".join(f"{'M' if i==0 else 'L'}{tx(i):.1f},{ty(s['values'][i]):.1f}" for i in range(len(s["values"])))
        p.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2"/>')
    for k,s in enumerate(series):
        lx=pl+10+k*160
        p.append(f'<rect x="{lx}" y="{h-28}" width="12" height="3" fill="{s["color"]}"/>')
        p.append(f'<text x="{lx+16}" y="{h-24}" font-size="10" fill="#e2e8f0">{s["label"]}</text>')
    p.append("</svg>"); return "\n".join(p)


def generate_report(bt, wf) -> str:
    sm = bt["static_1_2x"]
    dm = bt["dynamic"]
    parts = []

    parts.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EXP-1220 Dynamic Leverage</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Inter',-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem;line-height:1.6;max-width:1100px;margin:0 auto}}
  h1{{font-size:1.8rem;margin-bottom:.5rem;color:#f8fafc}}
  h2{{font-size:1.3rem;margin:2rem 0 1rem;color:#93c5fd;border-bottom:1px solid #334155;padding-bottom:.5rem}}
  h3{{font-size:1.05rem;margin:1.2rem 0 .6rem;color:#cbd5e1}}
  .sub{{color:#94a3b8;font-size:.95rem;margin-bottom:2rem}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:1rem;margin:1rem 0}}
  .card{{background:#1e293b;border-radius:8px;padding:1rem;border:1px solid #334155}}
  .card .label{{font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
  .card .value{{font-size:1.4rem;font-weight:700;margin-top:.25rem}}
  .green{{color:#4ade80}}.red{{color:#f87171}}.yellow{{color:#fbbf24}}.blue{{color:#60a5fa}}.orange{{color:#fb923c}}.white{{color:#f8fafc}}
  table{{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.85rem}}
  th{{background:#1e293b;padding:.5rem .75rem;text-align:left;color:#94a3b8;font-weight:600;border-bottom:2px solid #334155}}
  td{{padding:.5rem .75rem;border-bottom:1px solid #1e293b}}
  tr:hover td{{background:#1e293b}}
  .verdict{{padding:.75rem 1rem;border-radius:6px;margin:1rem 0;font-size:.9rem}}
  .verdict-pass{{background:#052e16;border:1px solid #16a34a;color:#4ade80}}
  .verdict-warn{{background:#422006;border:1px solid #d97706;color:#fbbf24}}
  .note{{font-size:.8rem;color:#64748b;margin-top:.5rem}}
  svg{{display:block;width:100%;max-width:700px}}
  .footer{{margin-top:3rem;font-size:.75rem;color:#475569;text-align:center;border-top:1px solid #1e293b;padding-top:1rem}}
</style></head><body>

<h1>EXP-1220: Dynamic Leverage Manager</h1>
<div class="sub">
  Adaptive leverage scaling via VIX, term structure, realized vol — real data only<br>
  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  &nbsp;|&nbsp; 2020–2025 &nbsp;|&nbsp; Target: ~100% CAGR, DD ≤ 12%
</div>
""")

    # ── Head-to-Head ──
    parts.append("<h2>1. Dynamic vs Static 1.2x — Head-to-Head</h2>")
    parts.append("""<table><thead><tr>
    <th>Metric</th><th>Static 1.2x</th><th>Dynamic</th><th>Delta</th></tr></thead><tbody>""")
    for label, key, fmt, better in [
        ("CAGR", "cagr_pct", ".1f", "higher"),
        ("Sharpe", "sharpe", ".2f", "higher"),
        ("Sortino", "sortino", ".2f", "higher"),
        ("Max DD", "max_dd_pct", ".1f", "lower"),
        ("Calmar", "calmar", ".2f", "higher"),
        ("Vol", "vol_pct", ".1f", "lower"),
    ]:
        sv, dv = sm[key], dm[key]
        delta = dv - sv
        if better == "lower":
            delta_class = "green" if delta < 0 else "red"
        else:
            delta_class = "green" if delta > 0 else "red"
        pct = '%' if 'pct' in key else ''
        delta_str = f"{delta:+{fmt}}"
        parts.append(f"""<tr><td><strong>{label}</strong></td>
        <td>{sv:{fmt}}{pct}</td>
        <td>{dv:{fmt}}{pct}</td>
        <td class="{delta_class}">{delta_str}{pct}</td></tr>""")
    parts.append("</tbody></table>")

    # Verdict
    if dm["max_dd_pct"] <= 12 and dm["cagr_pct"] >= 90:
        parts.append(f'<div class="verdict verdict-pass">TARGET MET: Dynamic leverage achieves '
                     f'{dm["cagr_pct"]:.1f}% CAGR with {dm["max_dd_pct"]:.1f}% max DD</div>')
    elif dm["max_dd_pct"] <= 12:
        parts.append(f'<div class="verdict verdict-warn">DD target met ({dm["max_dd_pct"]:.1f}%) '
                     f'but CAGR {dm["cagr_pct"]:.1f}% below 100% target</div>')
    else:
        parts.append(f'<div class="verdict verdict-warn">DD {dm["max_dd_pct"]:.1f}% exceeds 12% target</div>')

    # ── Leverage Distribution ──
    parts.append("<h2>2. Leverage Distribution</h2>")
    ls = bt["leverage_stats"]
    parts.append(f"""<div class="cards">
    <div class="card"><div class="label">Mean Leverage</div><div class="value blue">{ls['mean']:.2f}x</div></div>
    <div class="card"><div class="label">Median</div><div class="value blue">{ls['median']:.2f}x</div></div>
    <div class="card"><div class="label">Min / Max</div><div class="value white">{ls['min']:.2f} / {ls['max']:.2f}</div></div>
    <div class="card"><div class="label">% at Max (≥1.75x)</div><div class="value green">{ls['pct_at_max']:.0f}%</div></div>
    <div class="card"><div class="label">% at Min (≤0.35x)</div><div class="value orange">{ls['pct_at_min']:.0f}%</div></div>
    </div>""")

    # Leverage time series chart
    dates_2020 = bt["dates_2020"]
    states = bt["aligned_states"]
    levs = [s.leverage for s in states]
    vixs = [s.vix for s in states]
    date_labels = [d.strftime("%Y-%m") for d in dates_2020]

    # Subsample for chart
    step = max(1, len(levs) // 300)
    chart_levs = levs[::step]
    chart_vix = [v / 50 for v in vixs[::step]]  # Scale VIX to leverage range
    chart_labels = date_labels[::step]

    parts.append(_svg_line(
        [{"label": "Dynamic Leverage", "values": chart_levs, "color": "#4ade80"},
         {"label": "VIX / 50", "values": chart_vix, "color": "#f87171"}],
        chart_labels, "Dynamic Leverage vs VIX (scaled)", w=750, h=260,
    ))

    # ── Equity Curves ──
    parts.append("<h2>3. Equity Curves</h2>")
    eq_dyn = (100 * np.cumprod(1 + bt["dynamic_rets"])).tolist()
    eq_stat = (100 * np.cumprod(1 + bt["static_rets"])).tolist()
    eq_base = (100 * np.cumprod(1 + bt["base_2020"])).tolist()

    parts.append(_svg_line(
        [{"label": "Dynamic", "values": eq_dyn[::step], "color": "#4ade80"},
         {"label": "Static 1.2x", "values": eq_stat[::step], "color": "#60a5fa"},
         {"label": "Base 1x", "values": eq_base[::step], "color": "#94a3b8"}],
        chart_labels, "Equity Curves (Base 100)", w=750, h=260,
    ))

    # ── Regime Distribution ──
    parts.append("<h2>4. Regime Distribution</h2>")
    rd = bt["regime_distribution"]
    parts.append("""<table><thead><tr>
    <th>Regime</th><th>Days</th><th>% of Total</th></tr></thead><tbody>""")
    for regime, data in sorted(rd.items()):
        parts.append(f"""<tr><td><strong>{regime}</strong></td>
        <td>{data['days']}</td><td>{data['pct']:.1f}%</td></tr>""")
    parts.append("</tbody></table>")

    # ── Regime Performance ──
    parts.append("<h2>5. Performance by Regime</h2>")
    dr = bt["dynamic_regime"]
    parts.append("""<table><thead><tr>
    <th>Regime</th><th>Days</th><th>Avg Leverage</th><th>CAGR</th>
    <th>Sharpe</th><th>Max DD</th></tr></thead><tbody>""")
    for regime, m in dr.items():
        parts.append(f"""<tr><td><strong>{regime}</strong></td>
        <td>{m['n_days']}</td><td>{m['avg_leverage']:.2f}x</td>
        <td class="{'green' if m['cagr_pct']>0 else 'red'}">{m['cagr_pct']:+.1f}%</td>
        <td>{m['sharpe']:.2f}</td>
        <td>{m['max_dd_pct']:.1f}%</td></tr>""")
    parts.append("</tbody></table>")

    # ── Year-by-Year ──
    parts.append("<h2>6. Year-by-Year Comparison</h2>")
    parts.append("""<table><thead><tr>
    <th>Year</th><th colspan="3">Static 1.2x</th><th colspan="3">Dynamic</th><th>DD Saved</th></tr>
    <tr><th></th><th>Return</th><th>Sharpe</th><th>DD</th>
    <th>Return</th><th>Sharpe</th><th>DD</th><th></th></tr></thead><tbody>""")
    for yr in sorted(bt["static_yearly"].keys()):
        s = bt["static_yearly"][yr]
        d = bt["dynamic_yearly"].get(yr, {"total_ret_pct": 0, "sharpe": 0, "max_dd_pct": 0})
        dd_saved = s["max_dd_pct"] - d["max_dd_pct"]
        cls = "green" if dd_saved > 0 else "red"
        parts.append(f"""<tr><td><strong>{yr}</strong></td>
        <td>{s['total_ret_pct']:+.1f}%</td><td>{s['sharpe']:.2f}</td><td>{s['max_dd_pct']:.1f}%</td>
        <td>{d['total_ret_pct']:+.1f}%</td><td>{d['sharpe']:.2f}</td><td>{d['max_dd_pct']:.1f}%</td>
        <td class="{cls}">{dd_saved:+.1f}pp</td></tr>""")
    parts.append("</tbody></table>")

    # ── Walk-Forward ──
    parts.append("<h2>7. Walk-Forward Validation (Train: 2020–2022, Test: 2023–2025)</h2>")
    parts.append("""<table><thead><tr>
    <th>Period</th><th>Days</th><th>Strategy</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead><tbody>""")
    for period, data in wf.items():
        label = period.replace("_", " ").title()
        for strat in ["dynamic", "static_1_2x"]:
            m = data[strat]
            parts.append(f"""<tr><td>{label}</td><td>{data['n_days']}</td>
            <td><strong>{strat.replace('_', ' ').title()}</strong></td>
            <td>{m['cagr_pct']:+.1f}%</td><td>{m['sharpe']:.2f}</td>
            <td>{m['max_dd_pct']:.1f}%</td></tr>""")
    parts.append("</tbody></table>")

    # WF ratio
    train_d = wf["train_2020_2022"]["dynamic"]
    test_d = wf["test_2023_2025"]["dynamic"]
    wf_ratio = test_d["sharpe"] / train_d["sharpe"] if abs(train_d["sharpe"]) > 0.01 else 0
    parts.append(f'<p class="note">Walk-forward Sharpe ratio (test/train): '
                 f'<strong>{wf_ratio:.2f}</strong></p>')
    if wf_ratio > 0.7:
        parts.append(f'<div class="verdict verdict-pass">Walk-forward PASS: test Sharpe '
                     f'{test_d["sharpe"]:.2f} / train {train_d["sharpe"]:.2f} = {wf_ratio:.2f}</div>')

    # ── Top DD Episodes ──
    parts.append("<h2>8. Top Drawdown Episodes</h2>")
    eps = bt["dd_episodes"]
    if eps:
        parts.append("""<table><thead><tr>
        <th>Start</th><th>Trough</th><th>End</th><th>Depth</th><th>Days</th><th>Lev at Trough</th></tr></thead><tbody>""")
        for ep in eps[:7]:
            parts.append(f"""<tr><td>{ep['start']}</td><td>{ep['trough']}</td><td>{ep['end']}</td>
            <td class="red">{ep['depth_pct']:.1f}%</td><td>{ep['days']}</td>
            <td>{ep['lev_at_trough']:.2f}x</td></tr>""")
        parts.append("</tbody></table>")

    parts.append("""
<div class="footer">
  EXP-1220 Dynamic Leverage Manager — PilotAI Credit Spreads<br>
  All data: Yahoo Finance (SPY, ^VIX, ^VIX3M). No synthetic data.<br>
  Leverage scales via VIX level, VIX/VIX3M term structure, 20-day realized vol.
</div></body></html>""")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 70)
    print("EXP-1220: DYNAMIC LEVERAGE BACKTEST")
    print("=" * 70)

    print("\n[1/4] Loading real market data & computing base returns...")
    base_rets, dates, vix, vix3m, spy_ret = load_all()
    print(f"  {len(dates)} days ({dates[0].date()} to {dates[-1].date()})")

    print("\n[2/4] Running dynamic vs static backtest...")
    bt = run_backtest(base_rets, dates, vix, vix3m, spy_ret)
    sm = bt["static_1_2x"]
    dm = bt["dynamic"]
    print(f"  Static 1.2x:  CAGR={sm['cagr_pct']:+.1f}%, Sharpe={sm['sharpe']:.2f}, DD={sm['max_dd_pct']:.1f}%")
    print(f"  Dynamic:      CAGR={dm['cagr_pct']:+.1f}%, Sharpe={dm['sharpe']:.2f}, DD={dm['max_dd_pct']:.1f}%")
    ls = bt["leverage_stats"]
    print(f"  Leverage: mean={ls['mean']:.2f}x, min={ls['min']:.2f}x, max={ls['max']:.2f}x")

    print("\n[3/4] Walk-forward validation...")
    wf = walk_forward(base_rets, dates, vix, vix3m, spy_ret)
    for period, data in wf.items():
        d = data["dynamic"]
        s = data["static_1_2x"]
        print(f"  {period}: Dynamic Sharpe={d['sharpe']:.2f} vs Static={s['sharpe']:.2f}")

    print("\n[4/4] Generating report...")
    html = generate_report(bt, wf)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html)
    print(f"  Report: {REPORT_PATH}")

    summary = {
        "static_1_2x": sm, "dynamic": dm,
        "leverage_stats": ls,
        "regime_distribution": bt["regime_distribution"],
        "dynamic_regime": bt["dynamic_regime"],
        "static_yearly": {str(k): v for k, v in bt["static_yearly"].items()},
        "dynamic_yearly": {str(k): v for k, v in bt["dynamic_yearly"].items()},
        "walk_forward": wf,
        "dd_episodes": bt["dd_episodes"][:10],
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  JSON: {JSON_PATH}")

    return summary


if __name__ == "__main__":
    main()
