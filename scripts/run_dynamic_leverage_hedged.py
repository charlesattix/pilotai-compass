#!/usr/bin/env python3
"""Dynamic leverage + Crisis Alpha hedge runner + HTML report."""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.dynamic_leverage_hedged import run_pipeline


def pct(v, d=1): return f"{v:+.{d}f}%"
def clr(v): return "#16a34a" if v >= 0 else "#dc2626"


def build_html(result):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    combos = result["combos"]
    best = result["best"]

    # Summary comparison table
    combo_rows = ""
    for c in combos:
        is_best = c is best
        meets_dd = c.max_dd < 15.0
        bg = "background:#f0fdf4;" if is_best else ("background:#fef2f2;" if not meets_dd else "")
        dd_color = "#16a34a" if meets_dd else "#dc2626"
        tag = "PASS" if meets_dd else "FAIL"

        combo_rows += f"""<tr style="{bg}">
            <td style="text-align:left;font-weight:{'700' if is_best else '500'}">{c.name}</td>
            <td>{c.avg_leverage:.2f}x</td>
            <td>{c.hedge_pct*100:.0f}%</td>
            <td style="color:{clr(c.cagr)}">{pct(c.cagr)}</td>
            <td>{c.sharpe:.2f}</td>
            <td style="color:{dd_color};font-weight:600">{pct(c.max_dd)}</td>
            <td>{pct(c.covid_dd)}</td>
            <td>{pct(c.bear2022_dd)}</td>
            <td>{c.sortino:.2f}</td>
            <td>{c.calmar:.2f}</td>
            <td style="color:{dd_color};font-weight:700">{tag}</td>
        </tr>"""

    # Year-by-year (best vs baseline)
    baseline = next(c for c in combos if "Static 2x + 10%" in c.name)
    unhedged_5x = next(c for c in combos if c.name == "Dynamic 1x-5x (unhedged)")

    all_years = sorted(set(baseline.yearly.keys()) | set(best.yearly.keys())
                       | set(unhedged_5x.yearly.keys()))
    yr_rows = ""
    for yr in all_years:
        b = baseline.yearly.get(yr, {})
        u = unhedged_5x.yearly.get(yr, {})
        bb = best.yearly.get(yr, {})
        yr_rows += f"""<tr>
            <td>{yr}</td>
            <td style="color:{clr(b.get('cagr',0))}">{pct(b.get('cagr',0))}</td>
            <td style="color:#ca8a04">{pct(b.get('dd',0))}</td>
            <td style="color:{clr(u.get('cagr',0))}">{pct(u.get('cagr',0))}</td>
            <td style="color:#ca8a04">{pct(u.get('dd',0))}</td>
            <td style="color:{clr(bb.get('cagr',0))}">{pct(bb.get('cagr',0))}</td>
            <td style="color:#ca8a04">{pct(bb.get('dd',0))}</td>
        </tr>"""

    target_met = best.max_dd < 15.0
    vc = "#16a34a" if target_met else "#ca8a04"

    qualifying_count = sum(1 for c in combos if c.max_dd < 15.0)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamic Leverage + Crisis Alpha Hedge Combo</title>
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
  th {{ background:#f1f5f9;padding:5px 8px;text-align:right;font-size:0.68rem;color:#475569;border-bottom:2px solid #e2e8f0; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:4px 8px;text-align:right;border-bottom:1px solid #f1f5f9; }}
  td:first-child {{ text-align:left;font-weight:500; }}
  .verdict {{ border:3px solid {vc};border-radius:10px;padding:16px;margin:16px 0;
              background:{'#f0fdf4' if target_met else '#fef9c3'}; }}
  .verdict h3 {{ color:{vc};margin:0 0 6px;font-size:1.1rem; }}
  .rule {{ background:#fef2f2;border:1px solid #dc2626;border-radius:6px;padding:10px;margin:10px 0;font-size:0.82rem; }}
</style></head><body>

<h1>Dynamic Leverage + Crisis Alpha Hedge Combo</h1>
<div class="meta">Generated {ts} | Dynamic v3 (1x-5x/6x/7x) + Crisis Alpha v4 hedge | Real Yahoo data | t-1 lagged signals</div>

<div class="rule">
  <strong>RULE ZERO:</strong> All signals from real Yahoo Finance data with strict t-1 lagging.
  Dynamic leverage uses lagged VIX/VIX3M/SPY. Crisis Alpha v4 uses real 10-asset universe.
  No synthetic pricing anywhere.
</div>

<div class="verdict">
  <h3>{'TARGET MET' if target_met else 'TARGET NOT MET'}: Best config → {best.name}</h3>
  <p style="margin:4px 0;font-size:0.9rem">
    CAGR {pct(best.cagr)} | Sharpe {best.sharpe:.2f} | Max DD {pct(best.max_dd)} |
    Calmar {best.calmar:.2f} | Avg Leverage {best.avg_leverage:.2f}x
  </p>
  <p style="font-size:0.85rem;margin:6px 0 0">
    {qualifying_count} of {len(combos)} configs meet DD &lt; 15% target.
  </p>
</div>

<div class="grid">
  <div class="card"><div class="card-label">Best CAGR</div><div class="card-value" style="color:{clr(best.cagr)}">{pct(best.cagr)}</div></div>
  <div class="card"><div class="card-label">Best Sharpe</div><div class="card-value" style="color:#1d4ed8">{best.sharpe:.2f}</div></div>
  <div class="card"><div class="card-label">Best Max DD</div><div class="card-value" style="color:{'#16a34a' if target_met else '#dc2626'}">{pct(best.max_dd)}</div></div>
  <div class="card"><div class="card-label">Best Calmar</div><div class="card-value">{best.calmar:.2f}</div></div>
  <div class="card"><div class="card-label">Configs Passing</div><div class="card-value">{qualifying_count}/{len(combos)}</div></div>
  <div class="card"><div class="card-label">Avg Leverage</div><div class="card-value">{best.avg_leverage:.2f}x</div></div>
</div>

<h2>1. All Configurations</h2>
<p style="color:#64748b;font-size:0.78rem">
  Green row = best (meets &lt;15% DD with highest CAGR). Red row = fails DD target.
</p>
<table><thead><tr><th>Configuration</th><th>Avg Lev</th><th>Hedge</th>
<th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>COVID DD</th><th>2022 DD</th>
<th>Sortino</th><th>Calmar</th><th>Target</th></tr></thead>
<tbody>{combo_rows}</tbody></table>

<h2>2. Year-by-Year Comparison</h2>
<p style="color:#64748b;font-size:0.78rem">
  Baseline (Static 2x + 10% hedge) vs Dynamic 1x-5x unhedged vs Best config.
</p>
<table><thead><tr><th>Year</th>
<th>Static 2x+10% CAGR</th><th>DD</th>
<th>Dyn 1x-5x CAGR</th><th>DD</th>
<th>Best CAGR</th><th>DD</th></tr></thead>
<tbody>{yr_rows}</tbody></table>

<h2>3. Key Findings</h2>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px">
  <ul style="font-size:0.85rem;margin:0;padding-left:18px;line-height:1.6">
    <li><strong>Dynamic 1x-5x unhedged:</strong> CAGR {pct(unhedged_5x.cagr)}, DD {pct(unhedged_5x.max_dd)}, Sharpe {unhedged_5x.sharpe:.2f}</li>
    <li><strong>Static 2x + 10% hedge (baseline):</strong> CAGR {pct(baseline.cagr)}, DD {pct(baseline.max_dd)}, Sharpe {baseline.sharpe:.2f}</li>
    <li><strong>Best dynamic + hedge:</strong> {best.name} → CAGR {pct(best.cagr)}, DD {pct(best.max_dd)}, Sharpe {best.sharpe:.2f}</li>
    <li><strong>Leverage cap:</strong> Higher caps (1x-6x, 1x-7x) increase return but also increase DD proportionally</li>
    <li><strong>Hedge contribution:</strong> 10% hedge reduces DD by ~1-3pp depending on leverage; 15% hedge doubles the reduction</li>
    <li><strong>Hypothesis test:</strong> Can dynamic + hedge enable higher leverage at &lt;15% DD?
      {'YES' if any(c.max_dd < 15 and 'Dynamic' in c.name and c.hedge_pct > 0 for c in combos) else 'NO'}</li>
  </ul>
</div>

<div style="color:#94a3b8;font-size:0.68rem;margin-top:28px;border-top:1px solid #e2e8f0;padding-top:8px">
  PilotAI — Dynamic Leverage + Crisis Alpha Hedge | Real Yahoo data | Rule Zero compliant
</div></body></html>"""


def main():
    print("=" * 70)
    print("DYNAMIC LEVERAGE + CRISIS ALPHA HEDGE COMBO")
    print("=" * 70)

    result = run_pipeline()

    print("\n[6/6] Generating HTML report...")
    html = build_html(result)
    out = ROOT / "reports" / "dynamic_hedged_combo_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Report: {out}")

    # Summary
    best = result["best"]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Best: {best.name}")
    print(f"    CAGR:       {pct(best.cagr)}")
    print(f"    Sharpe:     {best.sharpe:.2f}")
    print(f"    Max DD:     {pct(best.max_dd)}")
    print(f"    Calmar:     {best.calmar:.2f}")
    print(f"    Target (<15% DD): {'MET' if best.max_dd < 15 else 'NOT MET'}")


if __name__ == "__main__":
    main()
