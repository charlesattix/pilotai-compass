#!/usr/bin/env python3
"""Crisis Alpha Production runner — full backtest + portfolio sizing."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.crisis_alpha_production import (
    PRODUCTION_CONFIG, load_universe, backtest,
    find_optimal_allocation, build_exp1220_daily,
)


def pct(v, d=1): return f"{v:+.{d}f}%"
def clr(v): return "#16a34a" if v >= 0 else "#dc2626"


def build_html(result, optimal_alloc, alloc_sweep):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    cfg = result.config

    # DD period returns
    dd_rows = ""
    for label, dd_info in sorted(result.dd_period_returns.items()):
        delta = dd_info["outperf"]
        dc = clr(delta)
        dd_rows += f"""<tr>
            <td style="text-align:left;font-size:0.75rem">{dd_info['start']}</td>
            <td style="font-size:0.75rem">{dd_info['end']}</td>
            <td>{dd_info['days']}</td>
            <td style="color:{clr(dd_info['spy_return'])}">{pct(dd_info['spy_return'])}</td>
            <td style="color:{clr(dd_info['strategy_return'])};font-weight:600">{pct(dd_info['strategy_return'])}</td>
            <td style="color:{dc};font-weight:700">{pct(delta)}</td>
        </tr>"""

    # Yearly breakdown
    yr_rows = ""
    for yr in sorted(result.yearly.keys()):
        d = result.yearly[yr]
        yr_rows += f"""<tr>
            <td>{yr}</td>
            <td style="color:{clr(d['cagr'])}">{pct(d['cagr'])}</td>
            <td>{d['sharpe']:.2f}</td>
            <td>{d['vol']:.1f}%</td>
            <td style="color:#ca8a04">{pct(d['dd'])}</td>
        </tr>"""

    # Walk-forward folds
    wf_rows = ""
    for f in result.wf_folds:
        wf_rows += f"""<tr>
            <td>{f['test_year']}</td>
            <td>{f['n_train']}</td>
            <td>{f['n_test']}</td>
            <td>{f['is_sharpe']:.2f}</td>
            <td style="color:{clr(f['oos_sharpe'])};font-weight:600">{f['oos_sharpe']:.2f}</td>
            <td style="color:{clr(f['oos_return'])}">{pct(f['oos_return'])}</td>
        </tr>"""

    # Crisis performance
    crisis_rows = ""
    for name in sorted(result.crisis_performance.keys()):
        delta = result.crisis_performance[name]
        crisis_rows += f"""<tr>
            <td style="text-align:left">{name}</td>
            <td style="color:{clr(delta)};font-weight:600">{pct(delta)}</td>
        </tr>"""

    # Allocation sweep
    alloc_rows = ""
    for a in alloc_sweep:
        is_best = a["ca_weight"] == optimal_alloc["ca_weight"]
        bg = "background:#f0fdf4;" if is_best else ""
        alloc_rows += f"""<tr style="{bg}">
            <td style="font-weight:{'700' if is_best else '500'}">{a['ca_weight']*100:.0f}% CA / {a['exp1220_weight']*100:.0f}% EXP-1220</td>
            <td style="color:{clr(a['cagr'])}">{pct(a['cagr'])}</td>
            <td style="font-weight:{'700' if is_best else '400'}">{a['sharpe']:.2f}</td>
            <td style="color:#ca8a04">{pct(a['max_dd'])}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1780 Crisis Alpha — PRODUCTION</title>
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
  .verdict {{ border:2px solid #16a34a;border-radius:10px;padding:14px;margin:16px 0;background:#f0fdf4; }}
  .verdict h3 {{ color:#16a34a;margin:0 0 6px; }}
  .rule {{ background:#fef2f2;border:1px solid #dc2626;border-radius:6px;padding:10px;margin:10px 0;font-size:0.82rem; }}
  pre {{ background:#f1f5f9;padding:10px;border-radius:6px;font-size:0.75rem;overflow-x:auto; }}
</style></head><body>

<h1>EXP-1780 Crisis Alpha — PRODUCTION</h1>
<div class="meta">Generated {ts} | {cfg['name']} v{cfg['version']} |
Real Yahoo Finance data | 13-asset universe</div>

<div class="rule">
  <strong>RULE ZERO:</strong> 100% real Yahoo Finance prices. Zero synthetic data.
  Selected from v3 grid of 40/72 passing configs by highest Sharpe with DD-corr &lt; -0.3.
</div>

<div class="verdict">
  <h3>Production Config: v2_round / vol=0.06 / 1.5x</h3>
  <p style="margin:4px 0;font-size:0.85rem">
    Lookbacks {cfg['lookbacks']} | Vol target {cfg['vol_target']:.2f} |
    Leverage {cfg['leverage']}x | Rebalance {cfg['rebalance_days']}d
  </p>
</div>

<div class="grid">
  <div class="card"><div class="card-label">CAGR</div><div class="card-value" style="color:#16a34a">{pct(result.cagr)}</div></div>
  <div class="card"><div class="card-label">Sharpe (corrected)</div><div class="card-value" style="color:#1d4ed8">{result.sharpe:.2f}</div></div>
  <div class="card"><div class="card-label">Sortino</div><div class="card-value">{result.sortino:.2f}</div></div>
  <div class="card"><div class="card-label">Max DD</div><div class="card-value" style="color:#ca8a04">{pct(result.max_dd)}</div></div>
  <div class="card"><div class="card-label">SPY ρ (overall)</div><div class="card-value">{result.corr_to_spy:+.3f}</div></div>
  <div class="card"><div class="card-label">DD-period ρ</div><div class="card-value" style="color:{clr(-result.corr_during_dd)};font-weight:700">{result.corr_during_dd:+.3f}</div></div>
</div>

<h2>1. Configuration (Frozen)</h2>
<pre>
Universe ({len(cfg['universe'])} assets): {', '.join(cfg['universe'])}
Lookbacks: {cfg['lookbacks']}  (days)
Weights:   {cfg['lookback_weights']}
Vol target: {cfg['vol_target']:.2f}
Leverage: {cfg['leverage']}x
Rebalance: {cfg['rebalance_days']} days
Max asset weight: {cfg['max_asset_weight']:.2f}
Vol lookback: {cfg['vol_lookback_days']} days
</pre>

<h2>2. Returns During EXP-1220 Drawdown Periods (KEY METRIC)</h2>
<p style="color:#64748b;font-size:0.78rem">
  SPY rolling 60-day DD &lt; -3% used as EXP-1220 DD proxy.
  This is WHAT WE CARE ABOUT — outperformance when credit spreads struggle.
</p>
<table><thead><tr><th>Start</th><th>End</th><th>Days</th><th>SPY Return</th>
<th>Strategy Return</th><th>Outperf</th></tr></thead>
<tbody>{dd_rows}</tbody></table>

<h2>3. Year-by-Year Performance</h2>
<table><thead><tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Vol</th><th>Max DD</th></tr></thead>
<tbody>{yr_rows}</tbody></table>

<h2>4. Walk-Forward Validation (Expanding Window)</h2>
<table><thead><tr><th>Test Year</th><th>Train Days</th><th>Test Days</th>
<th>IS Sharpe</th><th>OOS Sharpe</th><th>OOS Return</th></tr></thead>
<tbody>{wf_rows}</tbody></table>

<h2>5. Historical Crisis Performance</h2>
<table><thead><tr><th>Crisis</th><th>vs SPY</th></tr></thead>
<tbody>{crisis_rows}</tbody></table>

<h2>6. Optimal Allocation with EXP-1220</h2>
<p style="color:#64748b;font-size:0.78rem">
  Sweep Crisis Alpha weight from 0% to 50%. Green row = maximum combined Sharpe.
  EXP-1220 returns use real yearly targets (2020-2025) + proxy for 2015-2019.
</p>
<table><thead><tr><th>Allocation</th><th>Combined CAGR</th><th>Combined Sharpe</th><th>Combined Max DD</th></tr></thead>
<tbody>{alloc_rows}</tbody></table>

<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;margin-top:16px">
  <strong>Recommended Allocation:</strong>
  <span style="color:#16a34a;font-weight:700">{optimal_alloc['ca_weight']*100:.0f}% Crisis Alpha + {optimal_alloc['exp1220_weight']*100:.0f}% EXP-1220</span><br>
  Combined CAGR: {pct(optimal_alloc['cagr'])} | Sharpe: {optimal_alloc['sharpe']:.2f} | Max DD: {pct(optimal_alloc['max_dd'])}
</div>

<div style="color:#94a3b8;font-size:0.68rem;margin-top:28px;border-top:1px solid #e2e8f0;padding-top:8px">
  PilotAI — EXP-1780 Crisis Alpha Production v1.0 | 13 assets | Real Yahoo data |
  Rule Zero compliant | Selected from v3 grid (40/72 passing configs)
</div></body></html>"""


def main():
    print("=" * 70)
    print("EXP-1780 CRISIS ALPHA — PRODUCTION VERSION")
    print("=" * 70)

    # Load data
    print("\n[1] Loading 13-asset universe (real Yahoo Finance)...")
    prices = load_universe(start="2014-01-01", end="2026-01-01")
    print(f"  Loaded: {list(prices.columns)} ({len(prices)} days)")

    # Run production backtest
    print("\n[2] Running production backtest (2015-2025)...")
    result = backtest(prices)
    print(f"  CAGR:           {pct(result.cagr)}")
    print(f"  Sharpe:         {result.sharpe:.2f}")
    print(f"  Sortino:        {result.sortino:.2f}")
    print(f"  Max DD:         {pct(result.max_dd)}")
    print(f"  SPY ρ (overall): {result.corr_to_spy:+.3f}")
    print(f"  DD-period ρ:    {result.corr_during_dd:+.3f}")
    print(f"  Crisis outperf: ", end="")
    crisis_avg = np.mean(list(result.crisis_performance.values())) if result.crisis_performance else 0
    print(f"{crisis_avg:+.1f}% (avg)")

    # DD period returns
    print(f"\n[3] Returns during EXP-1220 DD periods:")
    print(f"  Found {len(result.dd_period_returns)} DD periods")
    total_outperf = 0
    for label, dd in sorted(result.dd_period_returns.items()):
        print(f"    {dd['start']} to {dd['end']} ({dd['days']}d): "
              f"SPY {pct(dd['spy_return'])} vs CA {pct(dd['strategy_return'])} "
              f"= {pct(dd['outperf'])} outperf")
        total_outperf += dd['outperf']
    print(f"  Total DD-period outperformance: {total_outperf:+.1f}%")

    # Walk-forward
    print(f"\n[4] Walk-forward validation ({len(result.wf_folds)} folds):")
    for f in result.wf_folds:
        print(f"  {f['test_year']}: IS {f['is_sharpe']:.2f} → OOS {f['oos_sharpe']:.2f} "
              f"({pct(f['oos_return'])} return)")

    # Find optimal allocation
    print("\n[5] Finding optimal allocation with EXP-1220...")
    # Get SPY returns for EXP-1220 proxy
    spy_rets = prices["SPY"].pct_change().dropna()

    # Trim to match result dates
    start_date = result.dates[0]
    spy_rets = spy_rets[spy_rets.index >= start_date]

    optimal, sweep = find_optimal_allocation(
        crisis_alpha_rets=result.daily_returns,
        crisis_alpha_dates=result.dates,
        spy_rets=spy_rets,
    )
    print(f"  Sweep across 0-50% allocation:")
    for a in sweep:
        marker = " ← BEST" if a['sharpe'] == optimal['sharpe'] else ""
        print(f"    {a['ca_weight']*100:3.0f}% CA: CAGR {pct(a['cagr']):>8s} "
              f"Sharpe {a['sharpe']:5.2f} DD {pct(a['max_dd']):>8s}{marker}")

    print(f"\n  OPTIMAL: {optimal['ca_weight']*100:.0f}% Crisis Alpha + "
          f"{optimal['exp1220_weight']*100:.0f}% EXP-1220")
    print(f"    Combined CAGR:   {pct(optimal['cagr'])}")
    print(f"    Combined Sharpe: {optimal['sharpe']:.2f}")
    print(f"    Combined Max DD: {pct(optimal['max_dd'])}")

    # Generate report
    print("\n[6] Generating HTML report...")
    html = build_html(result, optimal, sweep)
    out = ROOT / "reports" / "exp1780_production.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Report: {out}")

    # Summary
    print("\n" + "=" * 70)
    print("PRODUCTION SUMMARY")
    print("=" * 70)
    print(f"  Standalone CA:    CAGR {pct(result.cagr)}  Sharpe {result.sharpe:.2f}  DD {pct(result.max_dd)}")
    print(f"  With {int(optimal['ca_weight']*100)}% allocation: CAGR {pct(optimal['cagr'])}  "
          f"Sharpe {optimal['sharpe']:.2f}  DD {pct(optimal['max_dd'])}")
    print(f"  DD-period correlation: {result.corr_during_dd:+.3f}")


if __name__ == "__main__":
    main()
