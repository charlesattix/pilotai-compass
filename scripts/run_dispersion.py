#!/usr/bin/env python3
"""EXP-1820 Dispersion runner — real IronVault data."""

import math, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.dispersion import (
    backtest_dispersion, compute_metrics, corrected_sharpe, CAPITAL,
)


def pct(v, d=1): return f"{v*100:+.{d}f}%"
def clr(v): return "#16a34a" if v >= 0 else "#dc2626"


def compute_exp1220_correlation(trades):
    """Compute yearly return correlation with EXP-1220."""
    if not trades:
        return 0.0
    df = pd.DataFrame([vars(t) for t in trades])
    df["exit_dt"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["exit_dt"].dt.year
    yearly_pnl = df.groupby("year")["pnl"].sum() / CAPITAL

    exp1220_yearly = {
        2020: 0.5297, 2021: 0.4913, 2022: 0.1482,
        2023: 0.4010, 2024: 0.3151, 2025: 0.3724,
    }

    common = set(yearly_pnl.index) & set(exp1220_yearly.keys())
    if len(common) < 2:
        return 0.0
    years = sorted(common)
    x = np.array([yearly_pnl[y] for y in years])
    y = np.array([exp1220_yearly[y] for y in years])
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_html(metrics, exp1220_corr, threshold):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Yearly rows
    yr_rows = ""
    for yr in sorted(metrics.get("yearly", {}).keys()):
        d = metrics["yearly"][yr]
        yr_rows += f"""<tr>
            <td>{yr}</td>
            <td>{d['n']}</td>
            <td style="color:{clr(d['pnl'])}">${d['pnl']:,.0f}</td>
            <td>{d['wr']*100:.0f}%</td>
            <td>{d['sharpe']:.2f}</td>
        </tr>"""

    # Per-ticker breakdown
    ticker_rows = ""
    for tk in sorted(metrics.get("ticker_stats", {}).keys()):
        d = metrics["ticker_stats"][tk]
        ticker_rows += f"""<tr>
            <td style="text-align:left;font-weight:500">{tk}</td>
            <td>{d['n']}</td>
            <td style="color:{clr(d['pnl'])}">${d['pnl']:,.0f}</td>
            <td>{d['wr']*100:.0f}%</td>
            <td>{d['sharpe']:.2f}</td>
            <td>{d['avg_vol_ratio']:.2f}</td>
        </tr>"""

    has_alpha = metrics["n"] >= 20 and metrics["oos_sharpe"] > 1.0
    vc = "#16a34a" if has_alpha else "#ca8a04"

    # Best sector
    ts_stats = metrics.get("ticker_stats", {})
    if ts_stats:
        best_sector = max(ts_stats.items(), key=lambda x: x[1]["sharpe"])[0]
    else:
        best_sector = "N/A"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1820: Dispersion Trading — Real IronVault Data</title>
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
  .verdict {{ border:2px solid {vc};border-radius:10px;padding:14px;margin:16px 0;
              background:{'#f0fdf4' if has_alpha else '#fef9c3'}; }}
  .verdict h3 {{ color:{vc};margin:0 0 6px; }}
  .rule {{ background:#fef2f2;border:1px solid #dc2626;border-radius:6px;padding:10px;margin:10px 0;font-size:0.82rem; }}
</style></head><body>

<h1>EXP-1820: Dispersion Trading (Relative Vol Premium)</h1>
<div class="meta">Generated {ts} | Real IronVault options + Yahoo spot |
Sectors: XLF, XLI, XLK, XLE | Index: SPY</div>

<div class="rule">
  <strong>RULE ZERO:</strong> All option prices from IronVault options_cache.db (Polygon real).
  Spot prices from Yahoo. Zero synthetic data.
</div>

<div class="verdict">
  <h3>{'REAL ALPHA' if has_alpha else 'MARGINAL/NO ALPHA'}: Dispersion Premium {'Found' if has_alpha else 'Not Clearly Present'}</h3>
  <p style="margin:4px 0;font-size:0.85rem">
    Threshold: sector vol / SPY vol &gt; {threshold:.2f}x to trigger trade.
    Correlation to EXP-1220: <strong>{exp1220_corr:+.3f}</strong>
    ({'low — genuine diversifier' if abs(exp1220_corr) < 0.3 else 'moderate'})
  </p>
</div>

<div class="grid">
  <div class="card"><div class="card-label">Total Trades</div><div class="card-value">{metrics['n']}</div></div>
  <div class="card"><div class="card-label">Total PnL</div><div class="card-value" style="color:{clr(metrics['pnl'])}">${metrics['pnl']:,.0f}</div></div>
  <div class="card"><div class="card-label">Win Rate</div><div class="card-value">{metrics['wr']*100:.0f}%</div></div>
  <div class="card"><div class="card-label">Sharpe</div><div class="card-value">{metrics['sharpe']:.2f}</div></div>
  <div class="card"><div class="card-label">IS Sharpe</div><div class="card-value">{metrics['is_sharpe']:.2f}</div></div>
  <div class="card"><div class="card-label">OOS Sharpe</div><div class="card-value" style="color:{clr(metrics['oos_sharpe'])}">{metrics['oos_sharpe']:.2f}</div></div>
  <div class="card"><div class="card-label">CAGR</div><div class="card-value" style="color:{clr(metrics['cagr'])}">{pct(metrics['cagr'])}</div></div>
  <div class="card"><div class="card-label">Max DD</div><div class="card-value" style="color:#ca8a04">{pct(metrics['max_dd'])}</div></div>
</div>

<h2>1. Per-Sector Breakdown</h2>
<p style="color:#64748b;font-size:0.78rem">Avg vol ratio = average (sector credit%/spot) / (SPY credit%/spot) at entry.</p>
<table><thead><tr><th>Sector</th><th>Trades</th><th>PnL</th><th>WR</th><th>Sharpe</th><th>Avg Vol Ratio</th></tr></thead>
<tbody>{ticker_rows}</tbody></table>

<h2>2. Year-by-Year</h2>
<table><thead><tr><th>Year</th><th>Trades</th><th>PnL</th><th>WR</th><th>Sharpe</th></tr></thead>
<tbody>{yr_rows}</tbody></table>

<h2>3. Key Findings</h2>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px">
  <ul style="font-size:0.85rem;margin:0;padding-left:18px">
    <li><strong>Strategy:</strong> Sell sector put spreads when sector vol &gt; SPY vol × {threshold:.2f}</li>
    <li><strong>Dispersion signal fires:</strong> {metrics['n']} times across {len(metrics.get('ticker_stats', {}))} sectors (2020-2025)</li>
    <li><strong>Correlation to EXP-1220:</strong> {exp1220_corr:+.3f} ({'decorrelated ✓' if abs(exp1220_corr) < 0.3 else 'correlated'})</li>
    <li><strong>Is dispersion premium real?</strong> {'YES — OOS Sharpe > 1.0 confirms persistence' if metrics['oos_sharpe'] > 1.0 else 'UNCLEAR — OOS Sharpe below 1.0'}</li>
    <li><strong>Best sector:</strong> {best_sector}</li>
  </ul>
</div>

<div style="color:#94a3b8;font-size:0.68rem;margin-top:28px;border-top:1px solid #e2e8f0;padding-top:8px">
  PilotAI — EXP-1820 Dispersion Trading v1.0 | Real IronVault data | Rule Zero compliant
</div></body></html>"""


def main():
    print("=" * 70)
    print("EXP-1820: DISPERSION TRADING (Real IronVault Data)")
    print("=" * 70)

    print("\nData availability:")
    print("  SPY:  193K contracts (2020-2026) — INDEX")
    print("  XLI:  17K contracts (2020-2026)")
    print("  XLF:  9K contracts  (2020-2026)")
    print("  XLK:  3K contracts  (2020-2026)")
    print("  XLE:  2K contracts  (2020-2026, partial)")
    print("  Note: True dispersion needs individual stock options — not available")
    print("  Implementing RELATIVE VOL PREMIUM variant instead")

    print("\n[1] Running dispersion backtest (threshold=1.15x)...")
    trades = backtest_dispersion(
        start="2020-06-01",
        end="2026-01-01",
        vol_ratio_threshold=1.15,
    )
    print(f"  Total trades: {len(trades)}")

    if not trades:
        print("\n  NO trades generated. Signal threshold may be too strict.")
        print("  Trying lower threshold (1.05x)...")
        trades = backtest_dispersion(
            start="2020-06-01",
            end="2026-01-01",
            vol_ratio_threshold=1.05,
        )
        print(f"  Total trades: {len(trades)}")

    if not trades:
        print("\n  Still no trades. Dispersion signal is not firing.")
        print("  This suggests sector vol is NOT consistently richer than SPY vol.")
        return

    metrics = compute_metrics(trades)
    print(f"\n[2] Overall metrics:")
    print(f"  Trades: {metrics['n']}")
    print(f"  PnL: ${metrics['pnl']:,.0f}")
    print(f"  Win rate: {metrics['wr']*100:.0f}%")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  IS Sharpe: {metrics['is_sharpe']:.2f}")
    print(f"  OOS Sharpe: {metrics['oos_sharpe']:.2f}")
    print(f"  CAGR: {pct(metrics['cagr'])}")
    print(f"  Max DD: {pct(metrics['max_dd'])}")

    print(f"\n[3] Per-sector breakdown:")
    for tk, d in metrics["ticker_stats"].items():
        print(f"  {tk}: n={d['n']:3d} PnL=${d['pnl']:>8,.0f} "
              f"WR={d['wr']*100:.0f}% Sharpe={d['sharpe']:5.2f} "
              f"avg_ratio={d['avg_vol_ratio']:.2f}")

    print(f"\n[4] Correlation with EXP-1220:")
    corr = compute_exp1220_correlation(trades)
    print(f"  Yearly correlation: {corr:+.3f}")

    # Report
    print("\n[5] Generating report...")
    html = build_html(metrics, corr, threshold=1.15)
    out = ROOT / "reports" / "exp1820_dispersion.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Report: {out}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    has_alpha = metrics["n"] >= 20 and metrics["oos_sharpe"] > 1.0
    print(f"  {'REAL ALPHA' if has_alpha else 'NO CLEAR ALPHA'}")
    print(f"  Dispersion premium {'EXISTS' if has_alpha else 'NOT PROVEN'} on sector ETFs")
    print(f"  Correlation to EXP-1220: {corr:+.3f}")


if __name__ == "__main__":
    main()
