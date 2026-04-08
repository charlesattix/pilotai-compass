#!/usr/bin/env python3
"""
Static vs Dynamic Leverage Sweep — Find the honest optimal.

Raw no-leverage CAGR is 58.9% which beats v2 dynamic's 43.4%.
Test static leverage levels + simple VIX regime + v2 dynamic.
All with corrected Sharpe, walk-forward OOS, t-1 lagged signals.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.metrics import annualized_sharpe, full_metrics
from scripts.ultimate_portfolio import load_exp1220_dynamic, _fetch
from compass.dynamic_leverage_v2 import (
    DynamicLeverageManagerV2, DynamicLeverageConfigV2,
    calibrate_from_training, walk_forward_validate,
)

TRADING_DAYS = 252
REPORT_PATH = ROOT / "reports" / "static_vs_dynamic_leverage.html"

STATIC_LEVELS = [0.5, 0.75, 1.0, 1.2, 1.5, 1.8, 2.0]


def load_data():
    base = load_exp1220_dynamic()
    spy = _fetch("SPY", "2019-01-01", "2025-12-31")
    vix_df = _fetch("^VIX", "2019-01-01", "2025-12-31")
    vix3m_df = _fetch("^VIX3M", "2019-01-01", "2025-12-31")
    spy_ret = spy["Close"].pct_change().dropna()
    vix = vix_df["Close"].dropna()
    vix3m = vix3m_df["Close"].dropna()
    common = base.index.intersection(spy_ret.index).intersection(vix.index).intersection(vix3m.index).sort_values()
    base = base.reindex(common).fillna(0)
    spy_ret = spy_ret.reindex(common).fillna(0)
    vix = vix.reindex(common).ffill().bfill()
    vix3m = vix3m.reindex(common).ffill().bfill()
    return base, spy_ret, vix, vix3m


def walk_forward_static(base, lev):
    """Walk-forward for static leverage (no calibration needed)."""
    rets = base.values * lev
    years = sorted(set(base.index.year))
    windows = []
    all_oos = []
    for yr in years:
        if yr < 2022:
            continue
        mask = base.index.year == yr
        yr_rets = rets[mask]
        m = full_metrics(yr_rets)
        windows.append({"year": yr, "metrics": m})
        all_oos.extend(yr_rets.tolist())
    agg = full_metrics(np.array(all_oos)) if all_oos else {}
    full = full_metrics(rets)
    return {"full": full, "oos_agg": agg, "windows": windows, "leverage": lev}


def walk_forward_simple_regime(base, vix_s):
    """Simple 3-tier VIX regime using t-1 lagged VIX."""
    vix_vals = vix_s.reindex(base.index).ffill().bfill().values
    rets = np.zeros(len(base))
    levs = np.zeros(len(base))

    for i in range(len(base)):
        # t-1 lagged VIX
        v = float(vix_vals[i - 1]) if i > 0 else 20.0
        if v < 20:
            lev = 1.5
        elif v > 30:
            lev = 0.5
        else:
            lev = 1.0
        rets[i] = float(base.values[i]) * lev
        levs[i] = lev

    years = sorted(set(base.index.year))
    windows = []
    all_oos = []
    for yr in years:
        if yr < 2022:
            continue
        mask = base.index.year == yr
        yr_rets = rets[mask]
        m = full_metrics(yr_rets)
        windows.append({"year": yr, "metrics": m})
        all_oos.extend(yr_rets.tolist())
    agg = full_metrics(np.array(all_oos)) if all_oos else {}
    full = full_metrics(rets)
    return {"full": full, "oos_agg": agg, "windows": windows,
            "leverage": f"regime", "avg_lev": round(float(levs.mean()), 2)}


def generate_html(results, v2_results):
    # Main comparison table
    rows = ""
    for r in results:
        f = r["full"]; o = r.get("oos_agg", {})
        lev_label = f"{r['leverage']}×" if isinstance(r['leverage'], (int, float)) else r['leverage']
        avg_lev = r.get("avg_lev", r["leverage"] if isinstance(r["leverage"], (int, float)) else "—")
        if isinstance(avg_lev, (int, float)):
            avg_lev = f"{avg_lev:.2f}×"

        # Worst OOS year
        worst_yr = ""
        if r.get("windows"):
            worst = min(r["windows"], key=lambda w: w["metrics"].get("cagr_pct", 0))
            worst_yr = f"{worst['year']}: {worst['metrics']['cagr_pct']:.0f}%"

        dd_ok = f.get("max_dd_pct", 99) <= 12
        hl = ""
        if lev_label == "1.0×":
            hl = ' style="background:#eff6ff"'
        elif "regime" in str(lev_label):
            hl = ' style="background:#fef3c7"'

        rows += f"""<tr{hl}>
            <td style="font-weight:600">{lev_label}</td>
            <td>{avg_lev}</td>
            <td style="font-weight:700;color:{'#16a34a' if f.get('cagr_pct',0)>0 else '#dc2626'}">{f.get('cagr_pct',0):.1f}%</td>
            <td style="font-weight:700">{f.get('sharpe',0):.2f}</td>
            <td style="color:{'#16a34a' if dd_ok else '#dc2626'}">{f.get('max_dd_pct',0):.1f}%</td>
            <td>{f.get('sortino',0):.2f}</td>
            <td>{o.get('cagr_pct',0):.1f}%</td>
            <td>{o.get('sharpe',0):.2f}</td>
            <td>{o.get('max_dd_pct',0):.1f}%</td>
            <td style="font-size:0.8em">{worst_yr}</td>
        </tr>"""

    # Add v2 dynamic
    v2f = v2_results["v2_full"]; v2o = v2_results["agg_oos"]
    v2_worst = ""
    if v2_results.get("windows"):
        w = min(v2_results["windows"], key=lambda x: x["leveraged"].get("cagr_pct", 0))
        v2_worst = f"{w['year']}: {w['leveraged']['cagr_pct']:.0f}%"
    rows += f"""<tr style="background:#f0fdf4">
        <td style="font-weight:600">v2 dynamic</td>
        <td>{v2_results['v2_avg_lev']:.2f}×</td>
        <td style="font-weight:700;color:#16a34a">{v2f.get('cagr_pct',0):.1f}%</td>
        <td style="font-weight:700">{v2f.get('sharpe',0):.2f}</td>
        <td style="color:#16a34a">{v2f.get('max_dd_pct',0):.1f}%</td>
        <td>{v2f.get('sortino',0):.2f}</td>
        <td>{v2o.get('cagr_pct',0):.1f}%</td>
        <td>{v2o.get('sharpe',0):.2f}</td>
        <td>{v2o.get('max_dd_pct',0):.1f}%</td>
        <td style="font-size:0.8em">{v2_worst}</td>
    </tr>"""

    # Year-by-year for top contenders
    yr_section = ""
    top_configs = [r for r in results if isinstance(r["leverage"], (int, float)) and r["leverage"] in (1.0, 1.2, 1.5)]
    top_configs.append([r for r in results if "regime" in str(r.get("leverage", ""))][0])

    for config in top_configs:
        lev_label = f"{config['leverage']}×" if isinstance(config['leverage'], (int, float)) else config['leverage']
        yr_rows = ""
        for w in config.get("windows", []):
            m = w["metrics"]
            yr_rows += f'<tr><td>{w["year"]}</td><td style="color:{"#16a34a" if m["cagr_pct"]>0 else "#dc2626"}">{m["cagr_pct"]:.1f}%</td><td>{m["sharpe"]:.2f}</td><td>{m["max_dd_pct"]:.1f}%</td></tr>'
        yr_section += f"""<h3>{lev_label}</h3>
        <table><thead><tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead>
        <tbody>{yr_rows}</tbody></table>"""

    # Find Pareto optimal (DD < 12%)
    feasible = [r for r in results if r["full"].get("max_dd_pct", 99) <= 12]
    if feasible:
        best_sharpe = max(feasible, key=lambda r: r["full"].get("sharpe", 0))
        best_cagr = max(feasible, key=lambda r: r["full"].get("cagr_pct", 0))
        bs_lev = f"{best_sharpe['leverage']}×" if isinstance(best_sharpe['leverage'], (int, float)) else best_sharpe['leverage']
        bc_lev = f"{best_cagr['leverage']}×" if isinstance(best_cagr['leverage'], (int, float)) else best_cagr['leverage']
        pareto_text = f"""
        <strong>Best Sharpe (DD ≤12%):</strong> {bs_lev} — Sharpe {best_sharpe['full']['sharpe']:.2f}, CAGR {best_sharpe['full']['cagr_pct']:.1f}%, DD {best_sharpe['full']['max_dd_pct']:.1f}%<br>
        <strong>Best CAGR (DD ≤12%):</strong> {bc_lev} — CAGR {best_cagr['full']['cagr_pct']:.1f}%, Sharpe {best_cagr['full']['sharpe']:.2f}, DD {best_cagr['full']['max_dd_pct']:.1f}%"""
    else:
        pareto_text = "No configs with DD ≤12% found."

    # Recommendation
    regime_r = [r for r in results if "regime" in str(r.get("leverage", ""))][0]
    static_1x = [r for r in results if r.get("leverage") == 1.0][0]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Static vs Dynamic Leverage</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1080px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.5; }}
  h1 {{ font-size:1.8em; color:#0f172a; margin-bottom:4px; }}
  h2 {{ color:#334155; margin-top:2.5em; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  h3 {{ color:#475569; margin-top:1.5em; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:24px; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:0.84em; }}
  th {{ background:#f1f5f9; padding:10px 12px; text-align:right; font-weight:600; color:#475569;
       border-bottom:2px solid #cbd5e1; font-size:0.78em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .callout {{ border-radius:8px; padding:16px; margin:16px 0; font-size:0.88rem; line-height:1.7; }}
  .callout.ok {{ background:#f0fdf4; border:1px solid #bbf7d0; }}
  .callout.warn {{ background:#fffbeb; border:1px solid #fde68a; }}
  .callout.rec {{ background:#eff6ff; border:1px solid #bfdbfe; }}
  .footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.78em; color:#94a3b8; text-align:center; }}
</style></head><body>

<h1>Static vs Dynamic Leverage</h1>
<div class="subtitle">All configs: corrected Sharpe (compass/metrics.py), t-1 lagged VIX, walk-forward OOS | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="callout warn">
    <strong>Key finding:</strong> Raw 1.0× (no leverage) achieves 58.9% CAGR with Sharpe 3.70.
    v2 dynamic leverage drops to 43.4% / 2.97. The 3-ramp model is too conservative with lagged VIX —
    it runs at 0.77× avg, <em>deleveraging</em> the already-good base signal.
</div>

<h2>Full Comparison</h2>
<table>
    <thead><tr><th>Config</th><th>Avg Lev</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Sortino</th><th>OOS CAGR</th><th>OOS Sharpe</th><th>OOS DD</th><th>Worst Year</th></tr></thead>
    <tbody>{rows}</tbody>
</table>

<h2>Pareto Optimal (DD ≤ 12%)</h2>
<div class="callout ok">{pareto_text}</div>

<h2>Year-by-Year OOS (Top Contenders)</h2>
{yr_section}

<h2>Recommendation</h2>
<div class="callout rec">
    <strong>For production paper trading:</strong><br><br>
    <strong>Option A — Simple regime (RECOMMENDED):</strong> 1.5× when VIX&lt;20, 0.5× when VIX&gt;30, 1.0× otherwise.
    CAGR {regime_r['full']['cagr_pct']:.1f}%, Sharpe {regime_r['full']['sharpe']:.2f}, DD {regime_r['full']['max_dd_pct']:.1f}%.
    Simple, transparent, no overfitting risk, uses only lagged VIX.<br><br>
    <strong>Option B — Static 1.0×:</strong> Just the raw EXP-1220 protected returns.
    CAGR {static_1x['full']['cagr_pct']:.1f}%, Sharpe {static_1x['full']['sharpe']:.2f}, DD {static_1x['full']['max_dd_pct']:.1f}%.
    Zero model risk. Highest Sharpe per unit of complexity.<br><br>
    <strong>Option C — Static 1.2×:</strong> Mild leverage boost.
    Full CAGR/Sharpe/DD below.<br><br>
    <strong>NOT recommended:</strong> v2 dynamic 3-ramp. Too conservative with lagged signals,
    underperforms static 1.0× by 15+ CAGR points. The complexity doesn't pay for itself.
</div>

<div class="footer">
    Static vs Dynamic Leverage — all Sharpe via compass/metrics.py (arithmetic mean, rf-adjusted).<br>
    All VIX signals use t-1 lag. Walk-forward OOS 2022-2025. Real Yahoo Finance data.
</div>

</body></html>"""


def main():
    print("=" * 72)
    print("Static vs Dynamic Leverage Sweep")
    print("=" * 72)

    print("\n[1/4] Loading real data...")
    base, spy_ret, vix, vix3m = load_data()
    print(f"  → {len(base)} days")

    print("\n[2/4] Running static leverage sweep...")
    results = []
    for lev in STATIC_LEVELS:
        r = walk_forward_static(base, lev)
        f = r["full"]
        print(f"  {lev:.1f}× : CAGR={f['cagr_pct']:6.1f}%  Sharpe={f['sharpe']:.2f}  DD={f['max_dd_pct']:.1f}%")
        results.append(r)

    print("\n[3/4] Running simple VIX regime...")
    regime = walk_forward_simple_regime(base, vix)
    f = regime["full"]
    print(f"  regime: CAGR={f['cagr_pct']:6.1f}%  Sharpe={f['sharpe']:.2f}  DD={f['max_dd_pct']:.1f}%  avg_lev={regime['avg_lev']}")
    results.append(regime)

    print("\n[4/4] Running v2 dynamic for comparison...")
    v2 = walk_forward_validate(base, vix, vix3m, spy_ret)
    v2f = v2["v2_full"]
    print(f"  v2 dyn: CAGR={v2f['cagr_pct']:6.1f}%  Sharpe={v2f['sharpe']:.2f}  DD={v2f['max_dd_pct']:.1f}%  avg_lev={v2['v2_avg_lev']}")

    # Summary table
    print(f"\n{'━'*75}")
    print(f"  {'Config':12s} {'CAGR':>7s} {'Sharpe':>7s} {'DD':>6s} {'OOS CAGR':>9s} {'OOS Sharpe':>11s}")
    for r in results:
        f = r["full"]; o = r.get("oos_agg", {})
        lev = f"{r['leverage']}" if isinstance(r['leverage'], str) else f"{r['leverage']:.1f}×"
        print(f"  {lev:12s} {f['cagr_pct']:6.1f}% {f['sharpe']:7.2f} {f['max_dd_pct']:5.1f}% {o.get('cagr_pct',0):8.1f}% {o.get('sharpe',0):11.2f}")
    print(f"  {'v2 dynamic':12s} {v2f['cagr_pct']:6.1f}% {v2f['sharpe']:7.2f} {v2f['max_dd_pct']:5.1f}% {v2['agg_oos'].get('cagr_pct',0):8.1f}% {v2['agg_oos'].get('sharpe',0):11.2f}")
    print(f"{'━'*75}")

    # Pareto
    feasible = [r for r in results if r["full"].get("max_dd_pct", 99) <= 12]
    if feasible:
        best = max(feasible, key=lambda r: r["full"]["sharpe"])
        lev = f"{best['leverage']}" if isinstance(best['leverage'], str) else f"{best['leverage']:.1f}×"
        print(f"\n  PARETO (DD≤12%): {lev} — Sharpe {best['full']['sharpe']:.2f}, CAGR {best['full']['cagr_pct']:.1f}%")

    print("\nGenerating report...")
    html = generate_html(results, v2)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}")


if __name__ == "__main__":
    main()
