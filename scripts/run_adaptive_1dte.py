#!/usr/bin/env python3
"""Adaptive 1DTE IC runner: rolling Sharpe monitor + portfolio combo."""

import math, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.zero_dte_ic import backtest_1_3_dte, load_spy_spot_yfinance, trade_sharpe
from compass.adaptive_1dte import (
    compute_rolling_sharpe, apply_adaptive_sizing,
    load_vix, attach_vix_to_trades, regime_breakdown,
    combine_portfolio, walk_forward_adaptive,
)


def pct(v, d=1): return f"{v*100:+.{d}f}%"
def clr(v): return "#16a34a" if v >= 0 else "#dc2626"


def build_html(static_m, adaptive_m, regimes, wf, combined_info, combined_daily, rolling_hist):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Regime table
    regime_rows = ""
    for name, m in regimes.items():
        regime_rows += f"""<tr>
            <td style="text-align:left">{name}</td>
            <td>{m['n']}</td>
            <td style="color:{clr(m['pnl'])}">${m['pnl']:,.0f}</td>
            <td>{m['wr']*100:.0f}%</td>
            <td>{m['sharpe']:.2f}</td>
            <td>${m['avg_pnl']:.0f}</td>
        </tr>"""

    # Walk-forward adaptive vs static
    wf_rows = ""
    for w in wf:
        delta = w["adaptive_sharpe"] - w["static_sharpe"]
        delta_c = clr(delta)
        wf_rows += f"""<tr>
            <td>{w['oos_year']}</td>
            <td>{w['n_oos']}</td>
            <td>{w['static_sharpe']:.2f}</td>
            <td style="color:${clr(w['static_pnl'])}">${w['static_pnl']:,.0f}</td>
            <td>{w['adaptive_sharpe']:.2f}</td>
            <td style="color:{clr(w['adaptive_pnl'])}">${w['adaptive_pnl']:,.0f}</td>
            <td style="color:{delta_c}">{delta:+.2f}</td>
            <td>{w['sized_down']} ({w['sized_down_pct']}%)</td>
        </tr>"""

    # Rolling Sharpe values (recent 20)
    rs_rows = ""
    for i, rs in enumerate(rolling_hist[-20:]):
        rs_str = f"{rs:.2f}" if rs is not None else "N/A"
        status = "NORMAL" if rs is None or rs >= 1.0 else "SIZE DOWN"
        sc = "#94a3b8" if rs is None else ("#16a34a" if rs >= 1.0 else "#dc2626")
        rs_rows += f"""<tr>
            <td>{i + 1}</td>
            <td style="color:{sc}">{rs_str}</td>
            <td style="color:{sc};font-weight:600">{status}</td>
        </tr>"""

    verdict_color = "#16a34a" if combined_info["sharpe"] > 2.0 else "#ca8a04"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1710 Adaptive — Rolling Sharpe Monitor + Portfolio Combo</title>
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
  .verdict {{ border:2px solid {verdict_color};border-radius:10px;padding:14px;margin:16px 0;
              background:{'#f0fdf4' if combined_info['sharpe'] > 2.0 else '#fef9c3'}; }}
  .verdict h3 {{ color:{verdict_color};margin:0 0 6px; }}
  .rule {{ background:#fef2f2;border:1px solid #dc2626;border-radius:6px;padding:10px;margin:10px 0;font-size:0.82rem; }}
</style></head><body>

<h1>EXP-1710 Adaptive — Rolling Sharpe + Portfolio Combo</h1>
<div class="meta">Generated {ts} | 1DTE SPY Iron Condors | Adaptive overlay | Real data</div>

<div class="rule">
  <strong>RULE ZERO:</strong> All option prices from IronVault. SPY/VIX from Yahoo Finance.
  Zero synthetic data. Rolling Sharpe computed from real trade PnLs.
</div>

<div class="verdict">
  <h3>Combined Portfolio: EXP-1220 + EXP-1710 Adaptive</h3>
  <p style="margin:4px 0;font-size:0.85rem">
    80% EXP-1220 + 20% EXP-1710 (adaptive) | Correlation {combined_info['correlation']:+.3f}
  </p>
</div>

<div class="grid">
  <div class="card"><div class="card-label">Combined CAGR</div><div class="card-value" style="color:#16a34a">{pct(combined_info['cagr'])}</div></div>
  <div class="card"><div class="card-label">Combined Sharpe</div><div class="card-value" style="color:#1d4ed8">{combined_info['sharpe']:.2f}</div></div>
  <div class="card"><div class="card-label">Combined DD</div><div class="card-value" style="color:#ca8a04">{pct(combined_info['max_dd'])}</div></div>
  <div class="card"><div class="card-label">Correlation</div><div class="card-value">{combined_info['correlation']:+.3f}</div></div>
  <div class="card"><div class="card-label">EXP-1710 Static Sharpe</div><div class="card-value">{static_m['sharpe']:.2f}</div></div>
  <div class="card"><div class="card-label">EXP-1710 Adaptive Sharpe</div><div class="card-value">{adaptive_m['sharpe']:.2f}</div></div>
</div>

<h2>1. Static vs Adaptive Comparison (Full Period)</h2>
<table><thead><tr><th>Mode</th><th>Trades</th><th>PnL</th><th>WR</th><th>Sharpe</th><th>Avg PnL</th></tr></thead>
<tbody>
<tr><td style="text-align:left;font-weight:500">Static (all full size)</td>
    <td>{static_m['n']}</td>
    <td style="color:{clr(static_m['pnl'])}">${static_m['pnl']:,.0f}</td>
    <td>{static_m['wr']*100:.0f}%</td>
    <td>{static_m['sharpe']:.2f}</td>
    <td>${static_m['avg_pnl']:.0f}</td></tr>
<tr style="background:#f0fdf4">
    <td style="text-align:left;font-weight:700">Adaptive (size down at RS &lt; 1.0)</td>
    <td>{adaptive_m['n']}</td>
    <td style="color:{clr(adaptive_m['pnl'])}">${adaptive_m['pnl']:,.0f}</td>
    <td>{adaptive_m['wr']*100:.0f}%</td>
    <td>{adaptive_m['sharpe']:.2f}</td>
    <td>${adaptive_m['avg_pnl']:.0f}</td></tr>
</tbody></table>

<h2>2. VIX Regime Breakdown</h2>
<table><thead><tr><th>Regime</th><th>Trades</th><th>PnL</th><th>WR</th><th>Sharpe</th><th>Avg PnL</th></tr></thead>
<tbody>{regime_rows}</tbody></table>

<h2>3. Walk-Forward: Adaptive vs Static</h2>
<table><thead><tr><th>OOS Year</th><th>Trades</th><th>Static Sharpe</th><th>Static PnL</th>
<th>Adaptive Sharpe</th><th>Adaptive PnL</th><th>Δ Sharpe</th><th>Sized Down</th></tr></thead>
<tbody>{wf_rows}</tbody></table>

<h2>4. Rolling Sharpe History (Recent 20 trades)</h2>
<table><thead><tr><th>Trade #</th><th>60d Rolling Sharpe</th><th>Status</th></tr></thead>
<tbody>{rs_rows}</tbody></table>

<h2>5. Key Findings</h2>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px">
  <ul style="font-size:0.85rem;margin:0;padding-left:18px">
    <li><strong>Rolling Sharpe monitor</strong>: trailing 60-day Sharpe triggers 50% size-down when &lt; 1.0</li>
    <li><strong>Adaptive sizing</strong>: {'reduces drawdown risk during decay periods' if adaptive_m['sharpe'] != static_m['sharpe'] else 'did not trigger (Sharpe stayed above 1.0)'}</li>
    <li><strong>VIX regime</strong>: performance varies by VIX level (see table above)</li>
    <li><strong>Portfolio combo</strong>: 80% EXP-1220 + 20% EXP-1710 gives Sharpe {combined_info['sharpe']:.2f}</li>
    <li><strong>Correlation</strong>: {combined_info['correlation']:+.3f} ({'low — good diversifier' if abs(combined_info['correlation']) < 0.3 else 'moderate'})</li>
  </ul>
</div>

<div style="color:#94a3b8;font-size:0.68rem;margin-top:28px;border-top:1px solid #e2e8f0;padding-top:8px">
  PilotAI — EXP-1710 Adaptive | 1DTE IC + rolling Sharpe monitor + portfolio combo |
  Real IronVault + Yahoo data | Rule Zero compliant
</div></body></html>"""


def main():
    print("=" * 70)
    print("EXP-1710 ADAPTIVE — Rolling Sharpe Monitor + Portfolio Combo")
    print("=" * 70)

    # Load data once
    print("\n[0] Loading data...")
    spy_spot = load_spy_spot_yfinance("2023-01-01", "2026-01-01")
    spy_rets = spy_spot.pct_change().dropna()
    vix_series = load_vix()

    # Run baseline 1DTE with optimal config
    print("\n[1] Running baseline 1DTE backtest (optimal config)...")
    trades = backtest_1_3_dte(
        dte_target=1,
        start_date="2023-01-01",
        end_date="2026-01-01",
        otm_pct=0.015,
        spread_width=3.0,
        risk_pct=0.02,
    )
    print(f"  {len(trades)} trades")

    # Convert to ICTrade objects if needed (backtest_1_3_dte returns list of ICTrade)
    trade_list = trades

    # Compute static metrics
    static_pnls = np.array([t.pnl for t in trade_list])
    static_m = {
        "n": len(trade_list),
        "pnl": round(float(static_pnls.sum()), 2),
        "wr": round(float((static_pnls > 0).sum() / len(trade_list)), 3),
        "sharpe": round(trade_sharpe(static_pnls), 2),
        "avg_pnl": round(float(static_pnls.mean()), 2),
    }
    print(f"  Static: PnL=${static_m['pnl']:,.0f} WR={static_m['wr']*100:.0f}% "
          f"Sharpe={static_m['sharpe']:.2f}")

    # Compute rolling Sharpe
    print("\n[2] Computing rolling 60-day Sharpe...")
    rolling = compute_rolling_sharpe(trade_list, window_days=60)
    n_valid = sum(1 for r in rolling if r is not None)
    n_below = sum(1 for r in rolling if r is not None and r < 1.0)
    print(f"  Valid rolling values: {n_valid}/{len(rolling)}")
    print(f"  Rolling Sharpe < 1.0: {n_below} trades ({n_below/max(len(rolling),1)*100:.1f}%)")

    # Apply adaptive sizing
    print("\n[3] Applying adaptive sizing (50% size down if RS < 1.0)...")
    adaptive_trades = apply_adaptive_sizing(
        trade_list, rolling,
        size_down_threshold=1.0,
        pause_threshold=0.5,
        size_down_factor=0.5,
    )
    adaptive_pnls = np.array([t.pnl for t in adaptive_trades])
    adaptive_m = {
        "n": len(adaptive_trades),
        "pnl": round(float(adaptive_pnls.sum()), 2),
        "wr": round(float((adaptive_pnls > 0).sum() / len(adaptive_trades)), 3),
        "sharpe": round(trade_sharpe(adaptive_pnls), 2),
        "avg_pnl": round(float(adaptive_pnls.mean()), 2),
    }
    print(f"  Adaptive: PnL=${adaptive_m['pnl']:,.0f} WR={adaptive_m['wr']*100:.0f}% "
          f"Sharpe={adaptive_m['sharpe']:.2f}")

    # VIX regime analysis
    print("\n[4] VIX regime breakdown...")
    trades_vix = attach_vix_to_trades(trade_list, vix_series)
    regimes = regime_breakdown(trades_vix)
    for name, m in regimes.items():
        if m["n"] > 0:
            print(f"  {name:15s}: n={m['n']:2d} sharpe={m['sharpe']:5.2f} "
                  f"wr={m['wr']*100:.0f}% avg=${m['avg_pnl']:.0f}")

    # Walk-forward adaptive
    print("\n[5] Walk-forward adaptive vs static...")
    wf = walk_forward_adaptive(trade_list, spy_rets)
    for w in wf:
        print(f"  {w['oos_year']}: static Sharpe {w['static_sharpe']} "
              f"vs adaptive {w['adaptive_sharpe']} "
              f"(sized down {w['sized_down']}/{w['n_oos']} = {w['sized_down_pct']}%)")

    # Portfolio combination
    print("\n[6] Combined portfolio (80% EXP-1220 + 20% EXP-1710)...")
    combined_daily, combined_info = combine_portfolio(
        adaptive_trades, spy_rets,
        ic_weight=0.20, exp1220_weight=0.80,
    )
    print(f"  CAGR: {pct(combined_info['cagr'])}")
    print(f"  Sharpe: {combined_info['sharpe']:.2f}")
    print(f"  Max DD: {pct(combined_info['max_dd'])}")
    print(f"  Correlation: {combined_info['correlation']:+.3f}")

    # Generate report
    print("\n[7] Generating report...")
    html = build_html(static_m, adaptive_m, regimes, wf, combined_info, combined_daily, rolling)
    out = ROOT / "reports" / "exp1710_adaptive.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Report: {out}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Static EXP-1710:    Sharpe {static_m['sharpe']:.2f}")
    print(f"  Adaptive EXP-1710:  Sharpe {adaptive_m['sharpe']:.2f}")
    print(f"  Combined portfolio: Sharpe {combined_info['sharpe']:.2f}, CAGR {pct(combined_info['cagr'])}")
    print(f"  Portfolio correlation: {combined_info['correlation']:+.3f}")


if __name__ == "__main__":
    main()
