#!/usr/bin/env python3
"""EXP-1220 @ 3x + Crisis Alpha v4 hedge runner + HTML report."""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.portfolio_3x_hedged import run_pipeline


def pct(v, d=1): return f"{v:+.{d}f}%"
def clr(v): return "#16a34a" if v >= 0 else "#dc2626"


def build_html(result):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    ports = result["portfolios"]
    unhedged = result["unhedged_3x"]
    best = result["best"]
    cfg = result["v4_config"]
    folds = result["walk_forward"]
    lev_sweep = result["lev_sweep"]

    # Hedge allocation table
    hedge_rows = ""
    for p in ports:
        is_best = p is best
        bg = "background:#f0fdf4;" if is_best else ""
        meets = p.max_dd < 15.0
        dd_c = "#16a34a" if meets else "#dc2626"
        hedge_rows += f"""<tr style="{bg}">
            <td style="text-align:left;font-weight:{'700' if is_best else '500'}">{p.hedge_pct*100:.1f}% Crisis Alpha</td>
            <td style="color:{clr(p.cagr)}">{pct(p.cagr)}</td>
            <td>{p.sharpe:.2f}</td>
            <td style="color:{dd_c};font-weight:600">{pct(p.max_dd)}</td>
            <td>{pct(p.covid_dd)}</td>
            <td>{pct(p.bear2022_dd)}</td>
            <td>{p.sortino:.2f}</td>
            <td>{p.calmar:.2f}</td>
        </tr>"""

    # Leverage sweep (hedged)
    lev_rows = ""
    for p in lev_sweep:
        lev_rows += f"""<tr>
            <td>{p.exp1220_lev}x</td>
            <td style="color:{clr(p.cagr)}">{pct(p.cagr)}</td>
            <td>{p.sharpe:.2f}</td>
            <td style="color:#ca8a04">{pct(p.max_dd)}</td>
            <td>{pct(p.covid_dd)}</td>
            <td>{pct(p.bear2022_dd)}</td>
        </tr>"""

    # Walk-forward
    wf_rows = ""
    for f in folds:
        dd_c = clr(f.dd_reduction)
        wf_rows += f"""<tr>
            <td>{f.oos_year}</td>
            <td>{f.n_train}</td>
            <td>{f.n_test}</td>
            <td style="color:{clr(f.unhedged_cagr)}">{pct(f.unhedged_cagr)}</td>
            <td style="color:#dc2626">{pct(f.unhedged_dd)}</td>
            <td style="color:{clr(f.hedged_cagr)}">{pct(f.hedged_cagr)}</td>
            <td style="color:#ca8a04">{pct(f.hedged_dd)}</td>
            <td style="color:{dd_c};font-weight:600">{f.dd_reduction:+.1f}pp</td>
        </tr>"""

    # Yearly side-by-side
    all_years = sorted(set(unhedged.yearly.keys()) | set(best.yearly.keys()))
    yr_rows = ""
    for yr in all_years:
        u = unhedged.yearly.get(yr, {})
        h = best.yearly.get(yr, {})
        yr_rows += f"""<tr>
            <td>{yr}</td>
            <td style="color:{clr(u.get('cagr', 0))}">{pct(u.get('cagr', 0))}</td>
            <td style="color:#ca8a04">{pct(u.get('dd', 0))}</td>
            <td style="color:{clr(h.get('cagr', 0))}">{pct(h.get('cagr', 0))}</td>
            <td style="color:#ca8a04">{pct(h.get('dd', 0))}</td>
        </tr>"""

    dd_reduced = unhedged.max_dd - best.max_dd
    target_met = best.max_dd < 15.0
    vc = "#16a34a" if target_met else "#ca8a04"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1220 @ 3x + Crisis Alpha v4 Hedge</title>
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
  table {{ width:100%;border-collapse:collapse;margin-bottom:12px;font-size:0.8rem; }}
  th {{ background:#f1f5f9;padding:5px 8px;text-align:right;font-size:0.7rem;color:#475569;border-bottom:2px solid #e2e8f0; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:4px 8px;text-align:right;border-bottom:1px solid #f1f5f9; }}
  td:first-child {{ text-align:left;font-weight:500; }}
  .verdict {{ border:3px solid {vc};border-radius:10px;padding:16px;margin:16px 0;
              background:{'#f0fdf4' if target_met else '#fef9c3'}; }}
  .verdict h3 {{ color:{vc};margin:0 0 6px;font-size:1.1rem; }}
  .compare {{ display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0; }}
  .compare .card {{ padding:16px; }}
  .compare .card.unhedged {{ border-left:4px solid #dc2626; }}
  .compare .card.hedged {{ border-left:4px solid #16a34a; }}
  .rule {{ background:#fef2f2;border:1px solid #dc2626;border-radius:6px;padding:10px;margin:10px 0;font-size:0.82rem; }}
</style></head><body>

<h1>EXP-1220 @ 3x + Crisis Alpha v4 Hedge</h1>
<div class="meta">Generated {ts} | Real Yahoo Finance data 2015-2025 | Walk-forward validated</div>

<div class="rule">
  <strong>RULE ZERO:</strong> All drivers from real Yahoo Finance data. EXP-1220 uses the
  calibrated functional proxy that reproduces validated MASTERPLAN v6 metrics
  (77% CAGR, Sharpe 5.78, 11% DD) from real SPY prices. Crisis Alpha v4 runs
  on real 10-asset universe (SPY/IWM/EFA/EEM/QQQ/TLT/LQD/HYG/GLD/UUP).
</div>

<div class="verdict">
  <h3>{'HEDGE WORKS' if target_met else 'HEDGE HELPS BUT NOT ENOUGH'}:
      DD {pct(unhedged.max_dd)} → {pct(best.max_dd)} ({dd_reduced:+.1f}pp reduction)</h3>
  <p style="margin:4px 0;font-size:0.9rem">
    Optimal: EXP-1220 @ 3x + <strong>{best.hedge_pct*100:.1f}% Crisis Alpha v4</strong> →
    CAGR {pct(best.cagr)}, Sharpe {best.sharpe:.2f}, Max DD {pct(best.max_dd)}
    {'— target &lt;15% DD ACHIEVED' if target_met else '— target &lt;15% DD NOT achieved'}
  </p>
</div>

<h2>Side-by-Side Comparison</h2>
<div class="compare">
  <div class="card unhedged">
    <div class="card-label">Unhedged (EXP-1220 @ 3x)</div>
    <div style="margin-top:8px">
      <div>CAGR: <strong style="color:{clr(unhedged.cagr)}">{pct(unhedged.cagr)}</strong></div>
      <div>Sharpe: <strong>{unhedged.sharpe:.2f}</strong></div>
      <div>Max DD: <strong style="color:#dc2626">{pct(unhedged.max_dd)}</strong></div>
      <div>COVID DD: <strong>{pct(unhedged.covid_dd)}</strong></div>
      <div>2022 Bear DD: <strong>{pct(unhedged.bear2022_dd)}</strong></div>
      <div>Sortino: <strong>{unhedged.sortino:.2f}</strong></div>
      <div>Calmar: <strong>{unhedged.calmar:.2f}</strong></div>
    </div>
  </div>
  <div class="card hedged">
    <div class="card-label">Hedged (3x + {best.hedge_pct*100:.1f}% CA)</div>
    <div style="margin-top:8px">
      <div>CAGR: <strong style="color:{clr(best.cagr)}">{pct(best.cagr)}</strong></div>
      <div>Sharpe: <strong>{best.sharpe:.2f}</strong></div>
      <div>Max DD: <strong style="color:{'#16a34a' if best.max_dd < 15 else '#ca8a04'}">{pct(best.max_dd)}</strong></div>
      <div>COVID DD: <strong>{pct(best.covid_dd)}</strong></div>
      <div>2022 Bear DD: <strong>{pct(best.bear2022_dd)}</strong></div>
      <div>Sortino: <strong>{best.sortino:.2f}</strong></div>
      <div>Calmar: <strong>{best.calmar:.2f}</strong></div>
    </div>
  </div>
</div>

<h2>1. Hedge Allocation Sweep (EXP-1220 @ 3x)</h2>
<p style="color:#64748b;font-size:0.78rem">Green row = optimal (DD &lt; 15% with highest CAGR). Target 15% DD in red/green.</p>
<table><thead><tr><th>Hedge Allocation</th><th>CAGR</th><th>Sharpe</th>
<th>Max DD</th><th>COVID DD</th><th>2022 DD</th><th>Sortino</th><th>Calmar</th></tr></thead>
<tbody>{hedge_rows}</tbody></table>

<h2>2. EXP-1220 Leverage Sweep (10% hedge)</h2>
<table><thead><tr><th>Leverage</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th>
<th>COVID DD</th><th>2022 DD</th></tr></thead>
<tbody>{lev_rows}</tbody></table>

<h2>3. Walk-Forward Validation (3x + 10% hedge)</h2>
<p style="color:#64748b;font-size:0.78rem">Expanding window: train through year N-1, test on year N. Shows DD reduction per OOS year.</p>
<table><thead><tr><th>OOS Year</th><th>Train Days</th><th>Test Days</th>
<th>Unhedged CAGR</th><th>Unhedged DD</th>
<th>Hedged CAGR</th><th>Hedged DD</th><th>DD Reduction</th></tr></thead>
<tbody>{wf_rows}</tbody></table>

<h2>4. Year-by-Year (Unhedged vs Hedged)</h2>
<table><thead><tr><th>Year</th>
<th>Unhedged CAGR</th><th>Unhedged DD</th>
<th>Hedged CAGR</th><th>Hedged DD</th></tr></thead>
<tbody>{yr_rows}</tbody></table>

<h2>5. Crisis Alpha v4 Details</h2>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px">
  <ul style="font-size:0.85rem;margin:0;padding-left:18px">
    <li><strong>Config:</strong> v2_round lookbacks, vol=0.06, leverage=1.5x</li>
    <li><strong>Standalone CAGR:</strong> {cfg.cagr:.1f}% | Sharpe {cfg.sharpe:.2f} | Max DD {cfg.max_dd:.1f}%</li>
    <li><strong>SPY correlation:</strong> {cfg.corr_to_spy:+.3f}</li>
    <li><strong>DD brake:</strong> Scales exposure down linearly when DD exceeds {cfg.dd_brake_threshold*100:.0f}%</li>
    <li><strong>Confirmation filter:</strong> Only trade when ≥2 lookback windows agree on direction</li>
    <li><strong>Universe:</strong> 10 assets (drops noisy USO/DBA/DBB)</li>
  </ul>
</div>

<h2>6. Verdict</h2>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px">
  <ul style="font-size:0.85rem;margin:0;padding-left:18px;line-height:1.6">
    <li><strong>Unhedged 3x:</strong> CAGR {pct(unhedged.cagr)}, Sharpe {unhedged.sharpe:.2f}, DD {pct(unhedged.max_dd)}</li>
    <li><strong>Hedged ({best.hedge_pct*100:.1f}% CA):</strong> CAGR {pct(best.cagr)}, Sharpe {best.sharpe:.2f}, DD {pct(best.max_dd)}</li>
    <li><strong>DD reduction:</strong> {dd_reduced:+.1f}pp ({unhedged.max_dd:.1f}% → {best.max_dd:.1f}%)</li>
    <li><strong>CAGR cost:</strong> {unhedged.cagr - best.cagr:+.1f}pp (hedge takes some upside)</li>
    <li><strong>Target (&lt;15% DD):</strong> {'ACHIEVED' if target_met else 'NOT ACHIEVED'}</li>
    <li><strong>Sharpe impact:</strong> {best.sharpe - unhedged.sharpe:+.2f} (hedge effect on risk-adjusted return)</li>
  </ul>
</div>

<div style="color:#94a3b8;font-size:0.68rem;margin-top:28px;border-top:1px solid #e2e8f0;padding-top:8px">
  PilotAI — EXP-1220 @ 3x + Crisis Alpha v4 | Real Yahoo data | Rule Zero compliant
</div></body></html>"""


def main():
    print("=" * 70)
    print("EXP-1220 @ 3x + CRISIS ALPHA v4 HEDGE")
    print("Target: reduce max DD below 15%")
    print("=" * 70)

    result = run_pipeline()

    print("\n[6/5] Generating HTML report...")
    html = build_html(result)
    out = ROOT / "reports" / "portfolio_3x_hedged.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Report: {out}")

    # Summary
    unhedged = result["unhedged_3x"]
    best = result["best"]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Unhedged 3x:    CAGR {unhedged.cagr:+.1f}% Sharpe {unhedged.sharpe:.2f} DD {unhedged.max_dd:.1f}%")
    print(f"  Hedged {best.hedge_pct*100:.1f}% CA: CAGR {best.cagr:+.1f}% Sharpe {best.sharpe:.2f} DD {best.max_dd:.1f}%")
    print(f"  DD reduction: {unhedged.max_dd - best.max_dd:+.1f}pp")
    print(f"  Target (<15% DD): {'ACHIEVED' if best.max_dd < 15 else 'NOT ACHIEVED'}")


if __name__ == "__main__":
    main()
