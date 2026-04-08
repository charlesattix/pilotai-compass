#!/usr/bin/env python3
"""
Smart Hedge Backtest — 5 Cost-Efficient Variants
==================================================
Real SPY puts cost 4.36%/yr (2.2x the 2% assumption).
Tests 5 alternatives + finds Pareto-optimal hedge.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.smart_hedge import (
    VARIANTS, build_daily_data, backtest_variant, find_pareto_optimal,
)


def pct(v, d=1):
    return f"{v*100:+.{d}f}%"

def clr(v):
    return "#16a34a" if v >= 0 else "#dc2626"


def build_html(results, no_hedge, pareto, old_cost):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    p = pareto

    # Comparison table
    comp_rows = ""
    # No-hedge baseline
    comp_rows += f"""<tr style="background:#fef2f2">
        <td style="text-align:left;font-weight:600">No Hedge (baseline)</td>
        <td style="color:{clr(no_hedge['cagr'])}">{pct(no_hedge['cagr'])}</td>
        <td>{no_hedge['sharpe']:.2f}</td>
        <td style="color:#dc2626;font-weight:600">{pct(no_hedge['max_dd'])}</td>
        <td style="color:#dc2626">{pct(no_hedge['covid_dd'])}</td>
        <td>0.00%</td><td>0.00%</td><td>0.00%</td>
    </tr>"""
    # Old hedge (4.36%/yr)
    comp_rows += f"""<tr style="background:#fef9c3">
        <td style="text-align:left">Old Hedge (4.36%/yr real cost)</td>
        <td style="color:{clr(old_cost['cagr'])}">{pct(old_cost['cagr'])}</td>
        <td>{old_cost['sharpe']:.2f}</td>
        <td style="color:#ca8a04">{pct(old_cost['max_dd'])}</td>
        <td style="color:#ca8a04">{pct(old_cost['covid_dd'])}</td>
        <td>{old_cost['annual_cost_pct']*100:.2f}%</td>
        <td>{old_cost['annual_payoff_pct']*100:.2f}%</td>
        <td>{old_cost['net_cost_pct']*100:.2f}%</td>
    </tr>"""
    # 5 variants
    for r in results:
        is_pareto = r["name"] == p["name"]
        bg = "background:#f0fdf4;" if is_pareto else ""
        label = f"{'** ' if is_pareto else ''}{r['name']}{'  PARETO' if is_pareto else ''}"
        comp_rows += f"""<tr style="{bg}">
            <td style="text-align:left;font-weight:{'700' if is_pareto else '500'}">{label}</td>
            <td style="color:{clr(r['cagr'])}">{pct(r['cagr'])}</td>
            <td>{r['sharpe']:.2f}</td>
            <td style="color:#ca8a04">{pct(r['max_dd'])}</td>
            <td style="color:{'#16a34a' if r['covid_dd'] > -0.15 else '#dc2626'}">{pct(r['covid_dd'])}</td>
            <td>{r['annual_cost_pct']*100:.2f}%</td>
            <td>{r['annual_payoff_pct']*100:.2f}%</td>
            <td>{r['net_cost_pct']*100:.2f}%</td>
        </tr>"""

    # Year-by-year for pareto winner
    yr_rows = ""
    for yr in sorted(p["per_year"].keys()):
        d = p["per_year"][yr]
        nh = no_hedge["per_year"].get(yr, {"return": 0, "dd": 0})
        yr_rows += f"""<tr>
            <td>{yr}</td>
            <td style="color:{clr(d['return'])}">{pct(d['return'])}</td>
            <td style="color:#ca8a04">{pct(d['dd'])}</td>
            <td style="color:{clr(nh['return'])}">{pct(nh['return'])}</td>
            <td style="color:#dc2626">{pct(nh['dd'])}</td>
        </tr>"""

    # Cost breakdown per variant
    cost_rows = ""
    for r in results:
        savings = old_cost["annual_cost_pct"] - r["annual_cost_pct"]
        cost_rows += f"""<tr>
            <td style="text-align:left">{r['name']}</td>
            <td>{r['annual_cost_pct']*100:.2f}%</td>
            <td>{r['annual_payoff_pct']*100:.2f}%</td>
            <td>{r['net_cost_pct']*100:.2f}%</td>
            <td style="color:#16a34a">{savings*100:+.2f}%</td>
        </tr>"""

    pareto_color = "#16a34a" if p["covid_dd"] > -0.15 else "#ca8a04"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Hedge — Cost-Efficient Tail Risk Protection</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:0;padding:24px;background:#fff;color:#1e293b; }}
  h1 {{ font-size:1.4rem;margin-bottom:2px; }}
  h2 {{ font-size:1.05rem;color:#1d4ed8;margin:24px 0 8px;border-bottom:2px solid #e2e8f0;padding-bottom:4px; }}
  .meta {{ color:#64748b;font-size:0.82rem;margin-bottom:18px; }}
  .grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:18px; }}
  .card {{ background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px; }}
  .card-label {{ font-size:0.68rem;color:#64748b;text-transform:uppercase; }}
  .card-value {{ font-size:1.2rem;font-weight:700;margin-top:2px; }}
  table {{ width:100%;border-collapse:collapse;margin-bottom:12px;font-size:0.78rem; }}
  th {{ background:#f1f5f9;padding:5px 8px;text-align:right;font-size:0.7rem;color:#475569;border-bottom:2px solid #e2e8f0; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:4px 8px;text-align:right;border-bottom:1px solid #f1f5f9; }}
  td:first-child {{ text-align:left;font-weight:500; }}
  .verdict {{ border:2px solid {pareto_color};border-radius:10px;padding:14px;margin:16px 0;
              background:{'#f0fdf4' if p['covid_dd'] > -0.15 else '#fef9c3'}; }}
  .verdict h3 {{ color:{pareto_color};margin:0 0 6px;font-size:1rem; }}
  .tag {{ display:inline-block;padding:2px 6px;border-radius:4px;font-size:0.68rem;font-weight:600;margin:1px; }}
  .tg {{ background:#dcfce7;color:#16a34a; }}
  .tb {{ background:#dbeafe;color:#2563eb; }}
  .ty {{ background:#fef9c3;color:#ca8a04; }}
  .tr {{ background:#fef2f2;color:#dc2626; }}
</style></head><body>

<h1>Smart Hedge — Cost-Efficient Tail Risk Protection</h1>
<div class="meta">Generated {ts} | Real SPY put costs from IronVault (4.36%/yr actual) |
5 cost-efficient alternatives backtested 2020-2025</div>

<div class="verdict">
  <h3>Pareto Optimal: {p['name']}</h3>
  <p style="margin:4px 0;font-size:0.85rem">Minimum cost hedge that keeps COVID DD &lt; 15%.</p>
  <span class="tg">CAGR {pct(p['cagr'])}</span>
  <span class="tb">Sharpe {p['sharpe']:.2f}</span>
  <span class="ty">Max DD {pct(p['max_dd'])}</span>
  <span class="{'tg' if p['covid_dd'] > -0.15 else 'ty'}">COVID DD {pct(p['covid_dd'])}</span>
  <span class="tg">Net cost {p['net_cost_pct']*100:.2f}%/yr</span>
  <span class="tg">Saves {(old_cost['annual_cost_pct'] - p['annual_cost_pct'])*100:.1f}%/yr vs old hedge</span>
</div>

<div class="grid">
  <div class="card"><div class="card-label">CAGR</div><div class="card-value" style="color:#16a34a">{pct(p['cagr'])}</div></div>
  <div class="card"><div class="card-label">Sharpe</div><div class="card-value" style="color:#1d4ed8">{p['sharpe']:.2f}</div></div>
  <div class="card"><div class="card-label">Max DD</div><div class="card-value" style="color:#ca8a04">{pct(p['max_dd'])}</div></div>
  <div class="card"><div class="card-label">COVID DD</div><div class="card-value" style="color:{'#16a34a' if p['covid_dd']>-0.15 else '#dc2626'}">{pct(p['covid_dd'])}</div></div>
  <div class="card"><div class="card-label">Annual Cost</div><div class="card-value">{p['annual_cost_pct']*100:.2f}%</div></div>
  <div class="card"><div class="card-label">Cost Savings</div><div class="card-value" style="color:#16a34a">{(old_cost['annual_cost_pct']-p['annual_cost_pct'])*100:.1f}%/yr</div></div>
</div>

<h2>1. The Problem: Real Put Costs Are 2.2x Higher Than Assumed</h2>
<div style="background:#fef2f2;border:1px solid #dc2626;border-radius:8px;padding:12px;margin:8px 0">
  <p style="font-size:0.85rem;margin:0">
    <strong>Assumed cost:</strong> 2.0%/yr flat &nbsp;|&nbsp;
    <strong>Real cost (IronVault):</strong> 4.36%/yr average &nbsp;|&nbsp;
    <strong>Ratio:</strong> 2.18x &nbsp;|&nbsp;
    <strong>Range:</strong> 2.44% (2023) to 7.25% (2025)
  </p>
</div>

<h2>2. Variant Comparison (5 Hedges + Baselines)</h2>
<p style="color:#64748b;font-size:0.78rem">Green row = Pareto optimal. COVID DD &lt; -15% = target.</p>
<table><thead><tr><th>Hedge Variant</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>COVID DD</th><th>Cost/yr</th><th>Payoff/yr</th><th>Net/yr</th></tr></thead>
<tbody>{comp_rows}</tbody></table>

<h2>3. Cost Analysis</h2>
<table><thead><tr><th>Variant</th><th>Annual Cost</th><th>Annual Payoff</th><th>Net Cost</th><th>vs Old Hedge</th></tr></thead>
<tbody>{cost_rows}</tbody></table>

<h2>4. Year-by-Year: Pareto Winner vs No Hedge</h2>
<table><thead><tr><th>Year</th><th>Hedged Return</th><th>Hedged DD</th><th>Unhedged Return</th><th>Unhedged DD</th></tr></thead>
<tbody>{yr_rows}</tbody></table>

<h2>5. Variant Design Rationale</h2>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px">
  <ul style="font-size:0.82rem;margin:0;padding-left:18px">
    <li><strong>A: VIX&lt;15 Puts Only</strong> — Buy only when cheapest (1.8-2.5%/yr). No hedge when VIX elevated (delever instead). Max cost savings.</li>
    <li><strong>B: Put Spreads</strong> — 5% wide put spreads cost 55% of naked puts. Capped payoff but dramatically cheaper. Always on.</li>
    <li><strong>C: Dynamic Budget</strong> — 0.5% in calm, 3% in crisis. Spends more when needed most. Self-adjusting.</li>
    <li><strong>D: Collar</strong> — Sell 3% OTM calls to fund puts. Near-zero net cost but caps upside at +2.5%/day.</li>
    <li><strong>E: Selective Quarterly</strong> — Hedge only Jan-Feb, Aug-Oct (historically worst months). 5 months hedged, 7 unhedged.</li>
  </ul>
</div>

<div style="color:#94a3b8;font-size:0.68rem;margin-top:28px;border-top:1px solid #e2e8f0;padding-top:8px">
  PilotAI — Smart Hedge v1.0 | Real IronVault put costs (4.36%/yr) |
  Pareto-optimal: min cost for COVID DD &lt; 15%
</div></body></html>"""


def main():
    print("=" * 70)
    print("SMART HEDGE — COST-EFFICIENT TAIL RISK PROTECTION")
    print("=" * 70)

    # Build data
    print("\n[1/4] Building daily data...")
    port_ret, spy_ret, vix = build_daily_data()
    n = len(port_ret)
    print(f"      {n} days ({n/252:.0f} years)")

    # No-hedge baseline
    print("\n[2/4] Running baselines...")
    # No hedge = just static 1.6x
    no_hedge_daily = port_ret * 1.6
    no_hedge_cum = np.cumprod(1 + no_hedge_daily)
    n_yr = n / 252
    nh_cagr = no_hedge_cum[-1] ** (1/n_yr) - 1
    nh_vol = np.std(no_hedge_daily) * np.sqrt(252)
    _rf_daily = 0.045 / 252
    nh_sharpe = (float(np.mean(no_hedge_daily)) - _rf_daily) / float(np.std(no_hedge_daily)) * np.sqrt(252) if float(np.std(no_hedge_daily)) > 1e-12 else 0
    nh_pk = np.maximum.accumulate(no_hedge_cum)
    nh_dd = ((no_hedge_cum - nh_pk) / nh_pk).min()
    # COVID DD
    covid_cum = np.cumprod(1 + no_hedge_daily[20:80])
    covid_pk = np.maximum.accumulate(covid_cum)
    nh_covid = ((covid_cum - covid_pk) / covid_pk).min()

    # Per-year
    nh_yearly = {}
    idx = 0
    for yr in range(2020, 2026):
        n_yr_d = 252 if yr != 2025 else 249
        yr_r = no_hedge_daily[idx:idx+n_yr_d]
        yr_cum_v = np.prod(1 + yr_r) - 1
        yr_eq = np.cumprod(1 + yr_r)
        yr_pk = np.maximum.accumulate(yr_eq)
        yr_dd = ((yr_eq - yr_pk) / yr_pk).min()
        nh_yearly[yr] = {"return": float(yr_cum_v), "dd": float(yr_dd)}
        idx += n_yr_d

    no_hedge = {
        "name": "No Hedge", "cagr": float(nh_cagr), "sharpe": float(nh_sharpe),
        "max_dd": float(nh_dd), "covid_dd": float(nh_covid), "vol": float(nh_vol),
        "annual_cost_pct": 0, "annual_payoff_pct": 0, "net_cost_pct": 0,
        "per_year": nh_yearly,
    }
    print(f"      No hedge: CAGR={pct(no_hedge['cagr'])} DD={pct(no_hedge['max_dd'])} COVID={pct(no_hedge['covid_dd'])}")

    # Old hedge (flat 4.36%/yr cost — the reality)
    old_cost = backtest_variant(lambda: type('X', (), {
        'name': 'Old (4.36%/yr flat)',
        'daily': lambda self, pv, sr, v, **kw: type('H', (), {
            'cost': pv * 0.0436 / 252, 'payoff': (
                pv * 0.0436 / 252 * 10 * abs(sr) / 0.01 if sr < -0.005 else 0
            ), 'hedge_active': True, 'leverage_adj': 1.0, 'vix': v, 'regime': 'old'
        })()
    })(), port_ret, spy_ret, vix)
    print(f"      Old hedge: CAGR={pct(old_cost['cagr'])} DD={pct(old_cost['max_dd'])} cost={old_cost['annual_cost_pct']*100:.2f}%/yr")

    # Run 5 variants
    print("\n[3/4] Running 5 hedge variants...")
    results = []
    for key in ["A", "B", "C", "D", "E"]:
        r = backtest_variant(VARIANTS[key], port_ret, spy_ret, vix)
        results.append(r)
        covid_ok = "PASS" if r["covid_dd"] > -0.15 else "FAIL"
        print(f"      {r['name']:40s} CAGR={pct(r['cagr'])} Sharpe={r['sharpe']:.2f} "
              f"DD={pct(r['max_dd'])} COVID={pct(r['covid_dd'])} cost={r['annual_cost_pct']*100:.2f}% {covid_ok}")

    # Find Pareto optimal
    pareto = find_pareto_optimal(results)
    print(f"\n      Pareto optimal: {pareto['name']}")

    # Generate report
    print("\n[4/4] Generating report...")
    html = build_html(results, no_hedge, pareto, old_cost)
    out = ROOT / "reports" / "smart_hedge_analysis.html"
    out.write_text(html, encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Problem: Real puts cost 4.36%/yr (2.2x assumed)")
    print(f"  Pareto winner: {pareto['name']}")
    print(f"    CAGR:     {pct(pareto['cagr'])}")
    print(f"    Sharpe:   {pareto['sharpe']:.2f}")
    print(f"    Max DD:   {pct(pareto['max_dd'])}")
    print(f"    COVID DD: {pct(pareto['covid_dd'])}")
    print(f"    Cost:     {pareto['annual_cost_pct']*100:.2f}%/yr (saves {(old_cost['annual_cost_pct']-pareto['annual_cost_pct'])*100:.1f}%/yr)")
    print(f"  Report: {out}")


if __name__ == "__main__":
    main()
