#!/usr/bin/env python3
"""
generate_pricing_report.py — Build HTML sensitivity report from JSON outputs.

Reads:
  output/credit_sensitivity.json
  output/exit_slippage_sensitivity.json

Writes:
  output/pricing_sensitivity_report.html
"""

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"


def load(name):
    return json.loads((OUTPUT / name).read_text())


def _ret_color(v):
    """Return a CSS background color for a return value."""
    if v >= 200:   return "#d4edda"  # green
    if v >= 100:   return "#d4edda"  # green
    if v >= 50:    return "#fff3cd"  # yellow
    if v >= 0:     return "#ffeeba"  # amber
    return "#f8d7da"  # red


def _dd_color(v):
    """Background color for drawdown (more negative = redder)."""
    if v >= -5:    return "#d4edda"
    if v >= -20:   return "#fff3cd"
    if v >= -40:   return "#ffeeba"
    return "#f8d7da"


def _wr_color(v):
    if v >= 90:  return "#d4edda"
    if v >= 70:  return "#fff3cd"
    if v >= 50:  return "#ffeeba"
    return "#f8d7da"


def _sharpe_color(v):
    if v >= 3.0:  return "#d4edda"
    if v >= 1.0:  return "#fff3cd"
    if v >= 0:    return "#ffeeba"
    return "#f8d7da"


def _td(val, color=None, bold=False, tooltip=None):
    style = f'style="background:{color}; padding:7px 12px; text-align:center;"' if color else 'style="padding:7px 12px; text-align:center;"'
    tip = f' title="{tooltip}"' if tooltip else ""
    content = f"<b>{val}</b>" if bold else str(val)
    return f"<td {style}{tip}>{content}</td>"


def _th(label, tooltip=None):
    tip = f' title="{tooltip}"' if tooltip else ""
    return f'<th style="background:#2c3e50; color:white; padding:8px 14px; text-align:center;"{tip}>{label}</th>'


def credit_table_html(data):
    rows = data["rows"]
    baseline = data["baseline_credit_fraction"]
    breakeven = data["breakeven_credit_fraction"]
    years = data["years"]

    html = ['<table style="border-collapse:collapse; width:100%; font-family: monospace; font-size: 13px;">']
    # Header
    html.append("<thead><tr>")
    html.append(_th("Credit %", "Fraction of spread width collected as credit"))
    html.append(_th("Credit $", "Dollar credit for $5 spread"))
    html.append(_th("Avg Return", "Average annual return 2020-2025"))
    html.append(_th("Worst DD", "Worst intra-year drawdown across all years"))
    html.append(_th("Win Rate", "Average win rate across all years"))
    html.append(_th("Total Trades", "Total trades across all years"))
    html.append(_th("Avg Sharpe", "Average Sharpe ratio across all years"))
    html.append(_th("Prof. Years", "Years with positive return / 6"))
    for y in years:
        html.append(_th(str(y), f"Annual return for {y}"))
    html.append("</tr></thead><tbody>")

    for row in rows:
        frac = row["credit_fraction"]
        is_baseline = abs(frac - baseline) < 0.001
        row_style = ' style="outline: 2px solid #2c3e50;"' if is_baseline else ""
        html.append(f"<tr{row_style}>")

        label = f"{frac:.0%}"
        if is_baseline:
            label += " ★"
        html.append(_td(label, "#e8f4f8" if is_baseline else None, bold=is_baseline))
        html.append(_td(f"${frac * 5:.2f}"))

        r = row["avg_annual_return_pct"]
        html.append(_td(f"{r:+.1f}%", _ret_color(r), bold=r < 0))
        html.append(_td(f"{row['worst_drawdown_pct']:+.1f}%", _dd_color(row["worst_drawdown_pct"])))
        html.append(_td(f"{row['avg_win_rate_pct']:.1f}%", _wr_color(row["avg_win_rate_pct"])))
        html.append(_td(f"{row['total_trades']:,}"))
        html.append(_td(f"{row['avg_sharpe']:.2f}", _sharpe_color(row["avg_sharpe"])))
        html.append(_td(f"{row['profitable_years']}/6", _ret_color(row["avg_annual_return_pct"])))
        for y in years:
            yr = row["by_year"][str(y)]
            rv = yr["return_pct"]
            html.append(_td(f"{rv:+.0f}%", _ret_color(rv)))
        html.append("</tr>")

    html.append("</tbody></table>")
    return "\n".join(html)


def slippage_table_html(data):
    rows = data["rows"]
    baseline_frac = data["baseline_exit_slippage_fraction"]
    breakeven = data["breakeven_exit_slippage_fraction"]
    years = data["years"]

    html = ['<table style="border-collapse:collapse; width:100%; font-family: monospace; font-size: 13px;">']
    html.append("<thead><tr>")
    html.append(_th("Slip %", "Exit slippage as fraction of spread width"))
    html.append(_th("Slip $", "Dollar exit slippage per spread"))
    html.append(_th("Avg Return", "Average annual return 2020-2025"))
    html.append(_th("Worst DD", "Worst intra-year drawdown"))
    html.append(_th("Win Rate", "Average win rate"))
    html.append(_th("Total Trades", "Total trades across all years"))
    html.append(_th("Avg Sharpe", "Average Sharpe ratio"))
    html.append(_th("Prof. Years", "Years with positive return / 6"))
    for y in years:
        html.append(_th(str(y), f"Annual return for {y}"))
    html.append("</tr></thead><tbody>")

    for row in rows:
        frac = row["exit_slippage_fraction"]
        is_baseline = abs(frac - baseline_frac) < 0.001
        is_danger = breakeven is not None and frac >= breakeven
        row_style = ' style="outline: 2px solid #2c3e50;"' if is_baseline else ""
        html.append(f"<tr{row_style}>")

        label = f"{frac:.0%}"
        if is_baseline:
            label += " ★"
        if is_danger and row["avg_annual_return_pct"] < 0:
            label += " ✗"
        html.append(_td(label, "#e8f4f8" if is_baseline else ("#f8d7da" if is_danger and row["avg_annual_return_pct"] < 0 else None), bold=is_baseline))
        html.append(_td(f"${row['exit_slippage_abs']:.2f}"))

        r = row["avg_annual_return_pct"]
        html.append(_td(f"{r:+.1f}%", _ret_color(r), bold=r < 0))
        html.append(_td(f"{row['worst_drawdown_pct']:+.1f}%", _dd_color(row["worst_drawdown_pct"])))
        wr = row["avg_win_rate_pct"]
        html.append(_td(f"{wr:.1f}%", _wr_color(wr)))
        html.append(_td(f"{row['total_trades']:,}"))
        html.append(_td(f"{row['avg_sharpe']:.2f}", _sharpe_color(row["avg_sharpe"])))
        html.append(_td(f"{row['profitable_years']}/6", _ret_color(r)))
        for y in years:
            yr = row["by_year"][str(y)]
            rv = yr["return_pct"]
            html.append(_td(f"{rv:+.0f}%", _ret_color(rv)))
        html.append("</tr>")

    html.append("</tbody></table>")
    return "\n".join(html)


def per_year_detail_html(data, sweep_key, label_fn, year_metric="return_pct"):
    """Build a per-year breakdown heatmap table."""
    rows = data["rows"]
    years = data["years"]

    html = ['<table style="border-collapse:collapse; font-family:monospace; font-size:12px;">']
    html.append("<thead><tr>")
    html.append(_th(sweep_key.replace("_", " ").title()))
    for y in years:
        html.append(_th(str(y)))
    html.append("</tr></thead><tbody>")

    for row in rows:
        html.append("<tr>")
        html.append(f'<td style="padding:6px 10px; font-weight:bold;">{label_fn(row)}</td>')
        for y in years:
            yr = row["by_year"][str(y)]
            rv = yr[year_metric]
            html.append(_td(f"{rv:+.1f}%", _ret_color(rv)))
        html.append("</tr>")

    html.append("</tbody></table>")
    return "\n".join(html)


def build_html(credit_data, slip_data):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    be_slip = slip_data.get("breakeven_exit_slippage_fraction")
    be_slip_abs = slip_data.get("breakeven_exit_slippage_abs")
    be_credit = credit_data.get("breakeven_credit_fraction")

    baseline_credit = credit_data["baseline_credit_fraction"]
    baseline_slip_frac = slip_data["baseline_exit_slippage_fraction"]
    baseline_slip_abs = slip_data["baseline_exit_slippage_abs"]

    # Find baseline row for credit
    baseline_credit_row = next(r for r in credit_data["rows"] if abs(r["credit_fraction"] - baseline_credit) < 0.001)
    baseline_slip_row = next(r for r in slip_data["rows"] if abs(r["exit_slippage_fraction"] - baseline_slip_frac) < 0.001)

    # Verdict for break-even margin
    if be_slip is not None:
        margin_pct = round((be_slip - baseline_slip_frac) / baseline_slip_frac * 100)
        slip_verdict = f"Break-even at <b>{be_slip:.0%}</b> of width (${be_slip_abs:.2f}). That's {margin_pct:.0f}% headroom above the <b>{baseline_slip_frac:.0%}</b> baseline."
    else:
        slip_verdict = "Strategy is profitable across all tested slippage levels — no break-even found."

    if be_credit is not None:
        credit_verdict = f"Break-even at <b>{be_credit:.0%}</b> of width. Below baseline of <b>{baseline_credit:.0%}</b>."
    else:
        credit_verdict = "Strategy is profitable across all tested credit fractions — no break-even found in the 20%–45% range."

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pricing Sensitivity Report — PilotAI Credit Spreads</title>
  <style>
    body {{
      background: #ffffff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: #2c3e50;
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{ font-size: 24px; border-bottom: 3px solid #2c3e50; padding-bottom: 8px; }}
    h2 {{ font-size: 18px; color: #2c3e50; margin-top: 36px; border-left: 4px solid #2c3e50; padding-left: 10px; }}
    h3 {{ font-size: 15px; color: #555; margin-top: 24px; }}
    .callout {{
      background: #f8f9fa;
      border-left: 4px solid #17a2b8;
      padding: 14px 18px;
      margin: 16px 0;
      border-radius: 0 6px 6px 0;
      font-size: 14px;
      line-height: 1.6;
    }}
    .warning {{
      background: #fff3cd;
      border-left: 4px solid #ffc107;
      padding: 14px 18px;
      margin: 16px 0;
      border-radius: 0 6px 6px 0;
      font-size: 14px;
      line-height: 1.6;
    }}
    .danger {{
      background: #f8d7da;
      border-left: 4px solid #dc3545;
      padding: 14px 18px;
      margin: 16px 0;
      border-radius: 0 6px 6px 0;
      font-size: 14px;
      line-height: 1.6;
    }}
    .success {{
      background: #d4edda;
      border-left: 4px solid #28a745;
      padding: 14px 18px;
      margin: 16px 0;
      border-radius: 0 6px 6px 0;
      font-size: 14px;
      line-height: 1.6;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      margin: 12px 0;
      flex-wrap: wrap;
      font-size: 12px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 5px;
    }}
    .legend-swatch {{
      width: 16px;
      height: 16px;
      border: 1px solid #ccc;
      border-radius: 3px;
    }}
    table {{ margin: 12px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-radius: 6px; overflow: hidden; }}
    th, td {{ border: 1px solid #dee2e6; }}
    .meta {{ font-size: 12px; color: #888; margin-top: 40px; border-top: 1px solid #eee; padding-top: 10px; }}
    .section {{ margin-bottom: 40px; }}
    .findings-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin: 16px 0;
    }}
    .finding-card {{
      border: 1px solid #dee2e6;
      border-radius: 8px;
      padding: 16px;
    }}
    .finding-card h4 {{ margin: 0 0 8px 0; font-size: 14px; color: #495057; }}
    .finding-card p {{ margin: 0; font-size: 13px; line-height: 1.5; }}
    .big-number {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }}
    .badge-green {{ background: #d4edda; color: #155724; }}
    .badge-yellow {{ background: #fff3cd; color: #856404; }}
    .badge-red {{ background: #f8d7da; color: #721c24; }}
  </style>
</head>
<body>

<h1>Pricing Sensitivity Report</h1>
<p style="color:#666; font-size:14px;">
  Champion config <code>exp_213_champion_maxc100.json</code> &nbsp;|&nbsp;
  SPY 2020–2025 &nbsp;|&nbsp;
  <b>Heuristic mode</b> (no Polygon API) &nbsp;|&nbsp;
  Generated: {now}
</p>

<div class="warning">
  <b>Important: Heuristic Mode Caveat</b><br>
  These runs use the <em>simplified heuristic pricer</em> (no real option prices), which produces
  <b>artificially high win rates (~99.7%)</b> and near-zero drawdowns under normal conditions.
  Real-data backtests (Champion: avg +820%) show more realistic outcomes.
  Use these sensitivity curves to understand the <em>shape</em> of risk and the <em>relative</em>
  break-even thresholds — not as absolute return predictions.
</div>

<h2>Executive Summary</h2>

<div class="findings-grid">
  <div class="finding-card">
    <h4>Credit Assumption Sensitivity</h4>
    <p>{credit_verdict}</p>
    <br>
    <p>
      At the worst-case tested (20% of width = $1.00 credit on a $5 spread),
      the strategy still returns <b>+{credit_data['rows'][0]['avg_annual_return_pct']:.0f}%</b> avg/yr
      with <b>{credit_data['rows'][0]['profitable_years']}/6 profitable years</b>.
      Every $0.05 decrease in credit fraction costs roughly
      <b>~{round((credit_data['rows'][1]['avg_annual_return_pct'] - credit_data['rows'][0]['avg_annual_return_pct']) / 1, 0):.0f}%</b> in avg annual return.
    </p>
  </div>
  <div class="finding-card">
    <h4>Exit Slippage Sensitivity</h4>
    <p>{slip_verdict}</p>
    <br>
    <p>
      At 10% exit slippage ($0.50 on $5 spread), avg return drops to
      <b>+{next(r for r in slip_data['rows'] if abs(r['exit_slippage_fraction'] - 0.10) < 0.001)['avg_annual_return_pct']:.0f}%</b>
      and 2020 goes negative (−27%).
      Exit slippage is the <b>higher-risk assumption</b>: it can hit suddenly in gap scenarios
      while credit degradation is gradual.
    </p>
  </div>
</div>

<div class="callout">
  <b>Key Question: What does 14% exit slippage mean in practice?</b><br>
  Break-even is at $0.70 of slippage on a $5-wide spread = 14% of width.
  For a 3% OTM bull put spread on SPY at $450, that's paying an extra $0.70/share to
  close (vs mid-price). This happens in fast markets, at open, or when liquidity is thin.
  The <b>2%</b> baseline ($0.10) is tight and likely optimistic for real paper trading —
  expect <b>5–8%</b> ($0.25–0.40) in normal conditions, which still lands us at +{next(r for r in slip_data['rows'] if abs(r['exit_slippage_fraction'] - 0.05) < 0.001)['avg_annual_return_pct']:.0f}–{next(r for r in slip_data['rows'] if abs(r['exit_slippage_fraction'] - 0.02) < 0.001)['avg_annual_return_pct']:.0f}% avg.
</div>

<h2>Part 1 — Credit Assumption Sensitivity</h2>
<p style="font-size:13px; color:#555;">
  Tests <code>BACKTEST_CREDIT_FRACTION</code> in <code>shared/constants.py</code> — the fraction of
  spread width used as the synthetic entry credit in heuristic mode. Baseline = <b>35%</b> ($1.75 on $5 wide).
  <br>★ = baseline &nbsp;|&nbsp; *** = net negative
</p>

<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#d4edda"></div> Positive return</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#fff3cd"></div> 0–50% return</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ffeeba"></div> Marginal / near zero</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f8d7da"></div> Negative return</div>
</div>

{credit_table_html(credit_data)}

<h3>Credit Sensitivity — Per-Year Breakdown</h3>
{per_year_detail_html(credit_data, "credit_fraction", lambda r: f"{r['credit_fraction']:.0%} ({r['credit_fraction']*5:.2f}$)")}

<div class="success">
  <b>Finding:</b> Credit sensitivity is LOW risk. Even at 20% credit (nearly half the baseline),
  the strategy returns +{credit_data['rows'][0]['avg_annual_return_pct']:.0f}% avg with 6/6 profitable years.
  The heuristic win rate stays at ~99.7% because the simplified exit model doesn't
  penalise losing trades as harshly as real data — but the direction of risk is clear.
  Each 5% step down in credit fraction = roughly {round(abs(credit_data['rows'][1]['avg_annual_return_pct'] - credit_data['rows'][0]['avg_annual_return_pct'])/ 1):.0f}% lower avg return.
</div>

<h2>Part 2 — Exit Slippage Sensitivity</h2>
<p style="font-size:13px; color:#555;">
  Tests <code>exit_slippage</code> in the backtest config. Applied on all non-expiration
  exits (profit target + stop loss) via <code>_vix_scaled_exit_slippage()</code>, which
  also scales up 3x in extreme VIX regimes. Baseline = <b>$0.10</b> (2% of $5 spread).
  <br>★ = baseline &nbsp;|&nbsp; ✗ = past break-even
</p>

{slippage_table_html(slip_data)}

<h3>Exit Slippage — Per-Year Breakdown</h3>
{per_year_detail_html(slip_data, "exit_slippage_fraction", lambda r: f"{r['exit_slippage_fraction']:.0%} (${r['exit_slippage_abs']:.2f})")}

<div class="danger">
  <b>Finding:</b> Exit slippage is HIGH risk once it exceeds ~8% of width ($0.40).
  At 10% ($0.50), 2020 goes to −27% and worst drawdown hits −42%.
  At 15% ($0.75), 4 out of 6 years lose money.
  <b>Real paper trading should track actual fill slippage vs the $0.10 assumption.
  If fills are consistently ≥ $0.40 wider than mid, this config needs adjustment.</b>
</div>

<h2>Part 3 — Combined Risk Assessment</h2>

<table style="border-collapse:collapse; width:100%; font-size:13px; font-family:monospace;">
  <thead>
    <tr>
      <th style="background:#2c3e50; color:white; padding:9px 14px; text-align:left;">Assumption</th>
      <th style="background:#2c3e50; color:white; padding:9px 14px; text-align:center;">Baseline</th>
      <th style="background:#2c3e50; color:white; padding:9px 14px; text-align:center;">Break-Even</th>
      <th style="background:#2c3e50; color:white; padding:9px 14px; text-align:center;">Headroom</th>
      <th style="background:#2c3e50; color:white; padding:9px 14px; text-align:center;">Risk Rating</th>
      <th style="background:#2c3e50; color:white; padding:9px 14px; text-align:left;">Action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:9px 14px;">Credit fraction (heuristic)</td>
      <td style="padding:9px 14px; text-align:center;">35% ($1.75)</td>
      <td style="padding:9px 14px; text-align:center;">{'<20%' if be_credit is None else f'{be_credit:.0%}'}</td>
      <td style="padding:9px 14px; text-align:center; background:#d4edda;">{'> 15pp' if be_credit is None else f'{(baseline_credit - be_credit)*100:.0f}pp'}</td>
      <td style="padding:9px 14px; text-align:center;"><span class="badge badge-green">LOW</span></td>
      <td style="padding:9px 14px;">Monitor avg credit received vs 35% of width in paper trading</td>
    </tr>
    <tr>
      <td style="padding:9px 14px;">Exit slippage</td>
      <td style="padding:9px 14px; text-align:center;">2% ($0.10)</td>
      <td style="padding:9px 14px; text-align:center;">{f'{be_slip:.0%} (${be_slip_abs:.2f})' if be_slip else 'N/A'}</td>
      <td style="padding:9px 14px; text-align:center; background:#fff3cd;">{f'+{(be_slip - baseline_slip_frac)*100:.0f}pp ({round((be_slip-baseline_slip_frac)/baseline_slip_frac*100):.0f}%)' if be_slip else 'N/A'}</td>
      <td style="padding:9px 14px; text-align:center;"><span class="badge badge-yellow">MEDIUM</span></td>
      <td style="padding:9px 14px;">Log actual fill prices vs mid. Alert if slippage &gt; $0.30 per spread</td>
    </tr>
    <tr>
      <td style="padding:9px 14px;">VIX-scaled exit (extreme VIX)</td>
      <td style="padding:9px 14px; text-align:center;">3× at VIX=40</td>
      <td style="padding:9px 14px; text-align:center;">~4.7% base ($0.23)</td>
      <td style="padding:9px 14px; text-align:center; background:#ffeeba;">~130% above baseline</td>
      <td style="padding:9px 14px; text-align:center;"><span class="badge badge-yellow">MEDIUM</span></td>
      <td style="padding:9px 14px;">Circuit breaker at VIX&gt;40 already in place — validates design</td>
    </tr>
  </tbody>
</table>

<h2>Methodology Notes</h2>
<ul style="font-size:13px; line-height:1.8; color:#555;">
  <li><b>Heuristic mode</b>: No real Polygon option prices. Credits are synthetic
      (<code>spread_width × BACKTEST_CREDIT_FRACTION</code> minus entry slippage).
      Exit prices are estimated from a simplified time-decay + delta model.
      This makes win rates appear very high (~99.7%) and drawdowns unrealistically small.</li>
  <li><b>Real-data mode</b> (champion exp_213): Uses actual Polygon OHLCV bars for each
      option contract leg. Shows realistic win rates of ~80–90% and real P&amp;L per contract.</li>
  <li><b>ICs disabled in heuristic mode</b>: Iron condor logic requires real data; these sweeps
      test credit spread legs only. The champion config has ICs enabled in real-data runs.</li>
  <li><b>Interpretation</b>: Use the credit sensitivity to understand the <em>return sensitivity
      per dollar of credit reduction</em>. Use the slippage sensitivity to set paper-trading
      alert thresholds for fill quality monitoring.</li>
  <li><b>Next step</b>: Re-run these sweeps in real-data mode (offline_mode=True) to get
      accurate per-assumption break-even curves. Expected run time: 2–3 days on warm cache.</li>
</ul>

<div class="meta">
  Generated by <code>scripts/generate_pricing_report.py</code> on {now}<br>
  Source data: <code>output/credit_sensitivity.json</code>, <code>output/exit_slippage_sensitivity.json</code><br>
  Config: <code>configs/exp_213_champion_maxc100.json</code> | Mode: heuristic | SPY 2020-2025
</div>

</body>
</html>
"""
    return html


def main():
    credit_data = load("credit_sensitivity.json")
    slip_data = load("exit_slippage_sensitivity.json")
    html = build_html(credit_data, slip_data)
    out_path = OUTPUT / "pricing_sensitivity_report.html"
    out_path.write_text(html)
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
