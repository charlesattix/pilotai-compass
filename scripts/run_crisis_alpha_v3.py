#!/usr/bin/env python3
"""EXP-1780 v3 runner: focused grid around v2 winner."""

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.crisis_alpha_v3 import load_universe_v3, run_grid_v3


def pct(v, d=1): return f"{v:+.{d}f}%"
def clr(v): return "#16a34a" if v >= 0 else "#dc2626"


def build_html(results, best):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Sort: passing first (cagr≥8 AND corr_during_dd<0), then by cagr
    def _sort_key(r):
        passes = r.cagr >= 8.0 and r.corr_during_dd < 0.0
        return (not passes, -r.cagr)
    sorted_results = sorted(results, key=_sort_key)

    n_passing = sum(1 for r in results if r.cagr >= 8.0 and r.corr_during_dd < 0.0)

    # Grid table
    grid_rows = ""
    for r in sorted_results:
        passes = r.cagr >= 8.0 and r.corr_during_dd < 0.0
        bg = "background:#f0fdf4;" if passes else ""
        tag = "PASS" if passes else ""
        grid_rows += f"""<tr style="{bg}">
            <td style="text-align:left;font-weight:{'700' if passes else '500'}">{r.name}</td>
            <td style="color:{clr(r.cagr)}">{pct(r.cagr)}</td>
            <td>{r.sharpe:.2f}</td>
            <td>{r.sortino:.2f}</td>
            <td style="color:#ca8a04">{pct(r.max_dd)}</td>
            <td>{r.vol:.1f}%</td>
            <td style="color:{clr(-r.corr_to_spy)}">{r.corr_to_spy:+.3f}</td>
            <td style="color:{clr(-r.corr_during_dd)};font-weight:700">{r.corr_during_dd:+.3f}</td>
            <td style="color:{clr(r.crisis_avg_outperf)}">{pct(r.crisis_avg_outperf)}</td>
            <td style="color:{'#16a34a' if passes else '#94a3b8'};font-weight:600">{tag}</td>
        </tr>"""

    # Best config details
    best_yearly = ""
    for yr, d in sorted(best.yearly.items()):
        best_yearly += f"""<tr>
            <td>{yr}</td>
            <td style="color:{clr(d['cagr'])}">{pct(d['cagr'])}</td>
            <td>{d['sharpe']:.2f}</td>
            <td style="color:#ca8a04">{pct(d['dd'])}</td>
        </tr>"""

    # Crisis performance
    crisis_rows = ""
    for name, delta in sorted(best.crisis_performance.items()):
        crisis_rows += f"""<tr>
            <td style="text-align:left">{name}</td>
            <td style="color:{clr(delta)};font-weight:600">{pct(delta)}</td>
        </tr>"""

    vc = "#16a34a" if n_passing >= 1 else "#dc2626"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1780 v3 — Crisis Alpha Focused Optimization</title>
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
  .verdict {{ border:2px solid {vc};border-radius:10px;padding:14px;margin:16px 0;
              background:{'#f0fdf4' if n_passing>=1 else '#fef2f2'}; }}
  .verdict h3 {{ color:{vc};margin:0 0 6px; }}
  .rule {{ background:#fef2f2;border:1px solid #dc2626;border-radius:6px;padding:10px;margin:10px 0;font-size:0.82rem; }}
</style></head><body>

<h1>EXP-1780 v3 — Crisis Alpha Focused Optimization</h1>
<div class="meta">Generated {ts} | 14-asset universe | Real Yahoo Finance data |
Focused grid around v2 winner (v2_round / vol_target / 2.0x)</div>

<div class="rule">
  <strong>RULE ZERO:</strong> All prices from Yahoo Finance (real market data).
  Zero synthetic pricing. 14 assets: SPY/IWM/EFA/EEM/QQQ/TLT/LQD/HYG/GLD/USO/DBA/DBB/UUP.
</div>

<div class="verdict">
  <h3>{n_passing}/{len(results)} configs PASS (CAGR ≥ 8% AND corr_during_DD &lt; 0)</h3>
  <p style="margin:4px 0;font-size:0.85rem">
    Best: <strong>{best.name}</strong>
  </p>
  <p style="font-size:0.8rem;margin:4px 0">
    CAGR {pct(best.cagr)} | Sharpe {best.sharpe:.2f} |
    SPY ρ {best.corr_to_spy:+.3f} |
    <strong>DD-period ρ {best.corr_during_dd:+.3f}</strong> |
    Max DD {pct(best.max_dd)}
  </p>
</div>

<div class="grid">
  <div class="card"><div class="card-label">Best CAGR</div><div class="card-value" style="color:#16a34a">{pct(best.cagr)}</div></div>
  <div class="card"><div class="card-label">Best Sharpe</div><div class="card-value">{best.sharpe:.2f}</div></div>
  <div class="card"><div class="card-label">DD-period Corr</div><div class="card-value" style="color:{clr(-best.corr_during_dd)}">{best.corr_during_dd:+.3f}</div></div>
  <div class="card"><div class="card-label">Max DD</div><div class="card-value" style="color:#ca8a04">{pct(best.max_dd)}</div></div>
  <div class="card"><div class="card-label">Crisis Outperf</div><div class="card-value" style="color:#16a34a">{pct(best.crisis_avg_outperf)}</div></div>
  <div class="card"><div class="card-label">Universe Size</div><div class="card-value">{best.n_assets}</div></div>
</div>

<h2>1. Grid Search Results (sorted: passing first)</h2>
<p style="color:#64748b;font-size:0.78rem">Green rows = pass target (CAGR ≥ 8% AND DD-period correlation &lt; 0). Sorted by CAGR within groups.</p>
<table><thead><tr><th>Config</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th>
<th>Max DD</th><th>Vol</th><th>SPY ρ</th><th>DD ρ</th><th>Crisis Δ</th><th>Status</th></tr></thead>
<tbody>{grid_rows}</tbody></table>

<h2>2. Best Config Year-by-Year</h2>
<p style="color:#64748b;font-size:0.8rem">{best.name} — IS Sharpe {best.is_sharpe:.2f}, OOS Sharpe {best.oos_sharpe:.2f}</p>
<table><thead><tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead>
<tbody>{best_yearly}</tbody></table>

<h2>3. Crisis Performance (Best Config)</h2>
<table><thead><tr><th>Crisis</th><th>vs SPY</th></tr></thead>
<tbody>{crisis_rows}</tbody></table>

<h2>4. Key Finding: DD-Period Correlation</h2>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px">
  <ul style="font-size:0.85rem;margin:0;padding-left:18px">
    <li><strong>DD-period correlation</strong> is the KEY metric for crisis alpha</li>
    <li>Overall SPY correlation can be positive but still help if DD-period is negative</li>
    <li>DD periods: SPY rolling 60d peak-to-current drawdown &lt; -3%</li>
    <li>Best config achieves <strong>{best.corr_during_dd:+.3f}</strong> correlation during DD periods</li>
    <li>Crisis outperformance: <strong>{best.crisis_avg_outperf:+.1f}%</strong> avg across 5 historical crises</li>
  </ul>
</div>

<div style="color:#94a3b8;font-size:0.68rem;margin-top:28px;border-top:1px solid #e2e8f0;padding-top:8px">
  PilotAI — EXP-1780 v3 Crisis Alpha | 14 assets | Real Yahoo data | Rule Zero compliant
</div></body></html>"""


def main():
    print("=" * 70)
    print("EXP-1780 v3 — Crisis Alpha Focused Optimization")
    print("=" * 70)

    print("\n[1] Loading 14-asset universe from Yahoo...")
    prices = load_universe_v3(start="2014-01-01", end="2026-01-01")
    print(f"  Loaded {len(prices.columns)} assets, {len(prices)} days")
    print(f"  Universe: {list(prices.columns)}")

    print("\n[2] Running focused grid search...")
    print(f"  Grid: {6} lookback presets × {4} vol targets × {3} leverages = 72 configs")
    results = run_grid_v3(prices)

    # Find best
    def _score(r):
        if r.cagr >= 8.0 and r.corr_during_dd < 0.0:
            return (r.cagr / 100) * (abs(r.corr_during_dd) + 0.1) * 10
        return -1.0 + r.cagr / 100

    best = max(results, key=_score)

    print(f"\n[3] Best config: {best.name}")
    print(f"  CAGR: {pct(best.cagr)}")
    print(f"  Sharpe: {best.sharpe:.2f}")
    print(f"  SPY corr: {best.corr_to_spy:+.3f}")
    print(f"  DD-period corr (KEY): {best.corr_during_dd:+.3f}")
    print(f"  Crisis outperf: {pct(best.crisis_avg_outperf)}")
    print(f"  Max DD: {pct(best.max_dd)}")

    n_pass = sum(1 for r in results if r.cagr >= 8.0 and r.corr_during_dd < 0.0)
    print(f"\n  {n_pass}/{len(results)} configs PASS target")

    # Top 5
    print("\n  Top 5 by score:")
    top5 = sorted(results, key=_score, reverse=True)[:5]
    for r in top5:
        print(f"    {r.name:40s} CAGR={pct(r.cagr):>7s} "
              f"corr_DD={r.corr_during_dd:+.3f} crisis={pct(r.crisis_avg_outperf):>7s}")

    # Generate report
    print("\n[4] Generating report...")
    html = build_html(results, best)
    out = ROOT / "reports" / "exp1780_v3_focused.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Report: {out}")


if __name__ == "__main__":
    main()
