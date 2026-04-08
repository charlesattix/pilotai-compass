#!/usr/bin/env python3
"""
Monte Carlo Forward Simulation — Ultimate Portfolio v4.

Block-bootstrap 10,000 forward paths at 1yr, 3yr, 5yr horizons.
Includes regime-conditional scenarios and Kelly-optimal sizing.
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

from scripts.ultimate_portfolio_v4 import load_all, run_combined_backtest

TRADING_DAYS = 252
N_SIMS = 10_000
BLOCK_SIZE = 5  # 5-day blocks preserve autocorrelation
HORIZONS = {"1yr": 252, "3yr": 756, "5yr": 1260}
ACCOUNT = 100_000
REPORT_PATH = ROOT / "reports" / "monte_carlo_forward_simulation.html"


def get_v4_returns():
    """Run v4 backtest and extract daily returns."""
    print("  Running v4 backtest to get return stream...")
    df, spy_ret, spy_close, vix, vix3m = load_all()
    result = run_combined_backtest(df, spy_ret, spy_close, vix, vix3m)
    rets = result["daily_returns"]
    states = result["states"]
    print(f"  → {len(rets)} daily returns, mean={float(rets.mean())*252*100:.1f}% ann, vol={float(rets.std())*math.sqrt(252)*100:.1f}%")
    return rets, states


def block_bootstrap(rets, n_days, n_sims, block_size, rng):
    """Block bootstrap: sample blocks with replacement to build forward paths."""
    n = len(rets)
    n_blocks = math.ceil(n_days / block_size)
    paths = np.zeros((n_sims, n_days))

    for sim in range(n_sims):
        idx = rng.randint(0, n - block_size, size=n_blocks)
        sampled = np.concatenate([rets[i:i + block_size] for i in idx])
        paths[sim] = sampled[:n_days]

    return paths


def run_simulations(rets):
    """Run Monte Carlo for all horizons."""
    rng = np.random.RandomState(42)
    results = {}

    for label, n_days in HORIZONS.items():
        print(f"  Simulating {label} ({n_days} days, {N_SIMS:,} paths)...")
        paths = block_bootstrap(rets, n_days, N_SIMS, BLOCK_SIZE, rng)

        # Terminal wealth
        terminal = ACCOUNT * np.prod(1 + paths, axis=1)
        total_ret = terminal / ACCOUNT - 1
        n_years = n_days / TRADING_DAYS
        cagr = (terminal / ACCOUNT) ** (1 / n_years) - 1

        # Max drawdown per path
        max_dd = np.zeros(N_SIMS)
        for sim in range(N_SIMS):
            eq = np.cumprod(1 + paths[sim])
            hwm = np.maximum.accumulate(eq)
            dd = 1 - eq / hwm
            max_dd[sim] = float(dd.max())

        # Sharpe per path
        sharpe = np.mean(paths, axis=1) / np.maximum(np.std(paths, axis=1), 1e-10) * math.sqrt(TRADING_DAYS)

        results[label] = {
            "n_days": n_days,
            "terminal": terminal,
            "cagr": cagr,
            "total_ret": total_ret,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "stats": {
                "median_cagr": float(np.median(cagr)) * 100,
                "p5_cagr": float(np.percentile(cagr, 5)) * 100,
                "p25_cagr": float(np.percentile(cagr, 25)) * 100,
                "p75_cagr": float(np.percentile(cagr, 75)) * 100,
                "p95_cagr": float(np.percentile(cagr, 95)) * 100,
                "mean_cagr": float(np.mean(cagr)) * 100,
                "prob_50_return": float(np.mean(cagr > 0.50)) * 100,
                "prob_20_dd": float(np.mean(max_dd > 0.20)) * 100,
                "prob_profit": float(np.mean(cagr > 0)) * 100,
                "median_dd": float(np.median(max_dd)) * 100,
                "p95_dd": float(np.percentile(max_dd, 95)) * 100,
                "p99_dd": float(np.percentile(max_dd, 99)) * 100,
                "worst_dd": float(max_dd.max()) * 100,
                "median_sharpe": float(np.median(sharpe)),
                "median_terminal": float(np.median(terminal)),
                "p5_terminal": float(np.percentile(terminal, 5)),
                "p95_terminal": float(np.percentile(terminal, 95)),
            },
        }

    return results


def run_bear_scenario(rets, states):
    """Regime-conditional: prolonged 2022-style bear for 2 years."""
    # Extract bear/high_vol regime returns
    bear_rets = []
    for i, s in enumerate(states):
        if s["regime"] in ("bear", "high_vol", "neutral") and i < len(rets):
            bear_rets.append(rets[i])

    if len(bear_rets) < 20:
        bear_rets = rets[rets < np.median(rets)]  # fallback: below-median returns

    bear_arr = np.array(bear_rets)
    rng = np.random.RandomState(99)
    n_days = 504  # 2 years
    paths = block_bootstrap(bear_arr, n_days, N_SIMS, BLOCK_SIZE, rng)

    terminal = ACCOUNT * np.prod(1 + paths, axis=1)
    cagr = (terminal / ACCOUNT) ** (1 / 2.0) - 1

    max_dd = np.zeros(N_SIMS)
    for sim in range(N_SIMS):
        eq = np.cumprod(1 + paths[sim])
        hwm = np.maximum.accumulate(eq)
        max_dd[sim] = float((1 - eq / hwm).max())

    return {
        "median_cagr": float(np.median(cagr)) * 100,
        "p5_cagr": float(np.percentile(cagr, 5)) * 100,
        "p95_cagr": float(np.percentile(cagr, 95)) * 100,
        "prob_profit": float(np.mean(cagr > 0)) * 100,
        "median_dd": float(np.median(max_dd)) * 100,
        "p95_dd": float(np.percentile(max_dd, 95)) * 100,
        "worst_dd": float(max_dd.max()) * 100,
        "prob_20_dd": float(np.mean(max_dd > 0.20)) * 100,
        "median_terminal": float(np.median(terminal)),
    }


def compute_kelly(rets):
    """Kelly-optimal fraction for the portfolio."""
    mu = float(np.mean(rets)) * TRADING_DAYS
    sigma2 = float(np.var(rets)) * TRADING_DAYS
    if sigma2 < 1e-10:
        return 0.0, 0.0, 0.0
    full_kelly = mu / sigma2
    half_kelly = full_kelly / 2
    # Edge / odds for discrete approximation
    win_rate = float(np.mean(rets > 0))
    avg_win = float(np.mean(rets[rets > 0])) if np.any(rets > 0) else 0
    avg_loss = abs(float(np.mean(rets[rets < 0]))) if np.any(rets < 0) else 1
    edge = win_rate * avg_win - (1 - win_rate) * avg_loss
    return round(full_kelly, 2), round(half_kelly, 2), round(edge * TRADING_DAYS * 100, 1)


# ═══════════════════════════════════════════════════════════════════════════
# SVG Charts
# ═══════════════════════════════════════════════════════════════════════════

def _svg_histogram(values, title, xlabel, w=450, h=260, color="#16a34a"):
    """SVG histogram with percentile markers."""
    pl, pr, pt, pb = 55, 20, 36, 45; pw, ph = w - pl - pr, h - pt - pb
    vals = np.array(values)
    n_bins = 50
    counts, edges = np.histogram(vals, bins=n_bins)
    max_c = max(counts) if max(counts) > 0 else 1

    p = [f'<svg width="{w}" height="{h}" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px">']
    p.append(f'<text x="{w//2}" y="22" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e293b">{title}</text>')

    bw = pw / n_bins
    for i in range(n_bins):
        bh = counts[i] / max_c * ph
        x = pl + i * bw; y = pt + ph - bh
        p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.7"/>')

    # Percentile lines
    for pct, label, clr in [(5, "P5", "#dc2626"), (50, "P50", "#1e293b"), (95, "P95", "#16a34a")]:
        v = float(np.percentile(vals, pct))
        x = pl + (v - edges[0]) / (edges[-1] - edges[0]) * pw
        if pl <= x <= pl + pw:
            p.append(f'<line x1="{x:.0f}" y1="{pt}" x2="{x:.0f}" y2="{pt+ph}" stroke="{clr}" stroke-width="2" stroke-dasharray="4"/>')
            p.append(f'<text x="{x:.0f}" y="{pt+ph+12}" text-anchor="middle" font-size="8" font-weight="bold" fill="{clr}">{label}: {v:.1f}</text>')

    # X axis labels
    for j in range(5):
        v = edges[0] + j / 4 * (edges[-1] - edges[0])
        x = pl + j / 4 * pw
        p.append(f'<text x="{x:.0f}" y="{h-8}" text-anchor="middle" font-size="8" fill="#64748b">{v:.0f}</text>')

    p.append(f'<text x="{w//2}" y="{h-1}" text-anchor="middle" font-size="9" fill="#94a3b8">{xlabel}</text>')
    p.append("</svg>"); return "\n".join(p)


def _svg_fan_chart(sim_results, w=920, h=380):
    """Fan chart showing P5/P25/P50/P75/P95 wealth paths."""
    pl, pr, pt, pb = 80, 25, 42, 58; pw, ph = w - pl - pr, h - pt - pb

    # Build percentile paths for longest horizon
    longest = max(HORIZONS.values())
    rets_data = sim_results["5yr"]["terminal"]  # just for reference

    # We need to track equity at each time step for the 5yr sim
    # Rerun a small MC just for the fan chart
    rng = np.random.RandomState(42)
    n_fan = 1000  # fewer sims for fan chart
    all_rets = sim_results["_rets"]
    paths = block_bootstrap(all_rets, longest, n_fan, BLOCK_SIZE, rng)

    # Compute equity at each day
    equity = ACCOUNT * np.cumprod(1 + paths, axis=1)
    pcts = [5, 25, 50, 75, 95]
    percentile_paths = {p: np.percentile(equity, p, axis=0) for p in pcts}

    # Downsample for SVG
    step = max(1, longest // 200)
    x_idx = list(range(0, longest, step))
    if x_idx[-1] != longest - 1:
        x_idx.append(longest - 1)

    all_v = [v for pp in percentile_paths.values() for v in pp[x_idx]]
    ymin, ymax = min(all_v) * 0.9, max(all_v) * 1.1
    if ymax <= ymin: ymax = ymin + 1
    n = len(x_idx)

    def tx(i): return pl + i / max(n - 1, 1) * pw
    def ty(v): return pt + (1 - (v - ymin) / (ymax - ymin)) * ph

    p = [f'<svg width="{w}" height="{h}" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;margin:1rem 0">']
    p.append(f'<text x="{w//2}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">5-Year Wealth Fan Chart (1,000 paths)</text>')

    for j in range(7):
        yv = ymin + j / 6 * (ymax - ymin); y = ty(yv)
        p.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{pl+pw}" y2="{y:.0f}" stroke="#e2e8f0" stroke-width="1"/>')
        lbl = f"${yv:,.0f}" if yv < 1e6 else f"${yv / 1e6:.1f}M"
        p.append(f'<text x="{pl-8}" y="{y+4:.0f}" text-anchor="end" font-size="9" fill="#64748b">{lbl}</text>')

    # Year labels
    for yr in range(6):
        day = yr * 252
        if day < longest:
            xi = min(range(len(x_idx)), key=lambda k: abs(x_idx[k] - day))
            p.append(f'<text x="{tx(xi):.0f}" y="{h-14}" text-anchor="middle" font-size="9" fill="#64748b">Yr {yr}</text>')

    # Fan bands (P5-P95, P25-P75)
    colors = [("#dc262640", 5, 95), ("#ca8a0440", 25, 75)]
    for fill, lo, hi in colors:
        lo_path = percentile_paths[lo][x_idx]
        hi_path = percentile_paths[hi][x_idx]
        pts_top = " ".join(f"{tx(i):.1f},{ty(hi_path[i]):.1f}" for i in range(n))
        pts_bot = " ".join(f"{tx(i):.1f},{ty(lo_path[i]):.1f}" for i in range(n - 1, -1, -1))
        p.append(f'<polygon points="{pts_top} {pts_bot}" fill="{fill}"/>')

    # Median line
    med = percentile_paths[50][x_idx]
    d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(med[i]):.1f}" for i in range(n))
    p.append(f'<path d="{d}" fill="none" stroke="#16a34a" stroke-width="2.5"/>')

    # Starting capital line
    y0 = ty(ACCOUNT)
    p.append(f'<line x1="{pl}" y1="{y0:.0f}" x2="{pl+pw}" y2="{y0:.0f}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4"/>')

    # Legend
    lx = pl + 12
    p.append(f'<rect x="{lx}" y="{pt+8}" width="12" height="8" fill="#ca8a0440" stroke="#ca8a04"/>')
    p.append(f'<text x="{lx+16}" y="{pt+16}" font-size="9" fill="#1e293b">P25-P75</text>')
    p.append(f'<rect x="{lx+80}" y="{pt+8}" width="12" height="8" fill="#dc262640" stroke="#dc2626"/>')
    p.append(f'<text x="{lx+96}" y="{pt+16}" font-size="9" fill="#1e293b">P5-P95</text>')
    p.append(f'<rect x="{lx+155}" y="{pt+11}" width="14" height="3" fill="#16a34a"/>')
    p.append(f'<text x="{lx+173}" y="{pt+16}" font-size="9" fill="#1e293b">Median</text>')

    p.append("</svg>"); return "\n".join(p)


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_html(sim_results, bear, kelly):
    full_kelly, half_kelly, edge = kelly

    fan_svg = _svg_fan_chart(sim_results)

    # CAGR histograms
    hist_svgs = ""
    for label in ["1yr", "3yr", "5yr"]:
        s = sim_results[label]["stats"]
        hist_svgs += _svg_histogram(
            sim_results[label]["cagr"] * 100,
            f"{label.upper()} CAGR Distribution ({N_SIMS:,} paths)",
            "CAGR (%)", color="#3b82f6")
        hist_svgs += "\n"

    # DD histogram (5yr)
    dd_hist = _svg_histogram(
        sim_results["5yr"]["max_dd"] * 100,
        "5-Year Max Drawdown Distribution",
        "Max DD (%)", color="#dc2626")

    # Stats table per horizon
    stats_rows = ""
    for label in ["1yr", "3yr", "5yr"]:
        s = sim_results[label]["stats"]
        stats_rows += f"""<tr>
            <td style="font-weight:700">{label.upper()}</td>
            <td style="font-weight:700;color:#16a34a">{s['median_cagr']:.1f}%</td>
            <td style="color:#dc2626">{s['p5_cagr']:.1f}%</td>
            <td>{s['p25_cagr']:.1f}%</td>
            <td>{s['p75_cagr']:.1f}%</td>
            <td style="color:#16a34a">{s['p95_cagr']:.1f}%</td>
            <td style="font-weight:600">{s['prob_50_return']:.0f}%</td>
            <td>{s['prob_20_dd']:.1f}%</td>
            <td>{s['median_dd']:.1f}%</td>
            <td>{s['p95_dd']:.1f}%</td>
            <td>{s['prob_profit']:.0f}%</td>
        </tr>"""

    # Terminal wealth table
    wealth_rows = ""
    for label in ["1yr", "3yr", "5yr"]:
        s = sim_results[label]["stats"]
        wealth_rows += f"""<tr>
            <td style="font-weight:700">{label.upper()}</td>
            <td>${s['p5_terminal']:,.0f}</td>
            <td style="font-weight:700">${s['median_terminal']:,.0f}</td>
            <td>${s['p95_terminal']:,.0f}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monte Carlo Forward Simulation — Ultimate Portfolio v4</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1000px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.5; }}
  h1 {{ font-size:1.8em; color:#0f172a; margin-bottom:4px; }}
  h2 {{ color:#334155; margin-top:2.5em; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  h3 {{ color:#475569; margin-top:1.5em; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:24px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:18px;
          text-align:center; flex:1; min-width:130px; }}
  .kpi .value {{ font-size:1.7em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; }} .warn {{ color:#ca8a04; }} .bad {{ color:#dc2626; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:10px 12px; text-align:right; font-weight:600; color:#475569;
       border-bottom:2px solid #cbd5e1; font-size:0.80em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .hist-row {{ display:flex; flex-wrap:wrap; gap:16px; justify-content:center; margin:16px 0; }}
  .callout {{ background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:16px; margin:16px 0; font-size:0.88rem; }}
  .callout.warn {{ background:#fffbeb; border-color:#fde68a; }}
  .footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.78em; color:#94a3b8; text-align:center; }}
</style></head><body>

<h1>Monte Carlo Forward Simulation</h1>
<div class="subtitle">
    Ultimate Portfolio v4 | {N_SIMS:,} block-bootstrap paths | Block size: {BLOCK_SIZE} days | {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<div class="kpi-row">
    <div class="kpi"><div class="value good">{sim_results['1yr']['stats']['median_cagr']:.0f}%</div><div class="label">1yr Median CAGR</div></div>
    <div class="kpi"><div class="value good">{sim_results['3yr']['stats']['median_cagr']:.0f}%</div><div class="label">3yr Median CAGR</div></div>
    <div class="kpi"><div class="value good">{sim_results['5yr']['stats']['median_cagr']:.0f}%</div><div class="label">5yr Median CAGR</div></div>
    <div class="kpi"><div class="value">{sim_results['5yr']['stats']['prob_profit']:.0f}%</div><div class="label">5yr Prob Profit</div></div>
    <div class="kpi"><div class="value warn">{sim_results['5yr']['stats']['p95_dd']:.0f}%</div><div class="label">5yr P95 Max DD</div></div>
    <div class="kpi"><div class="value">{sim_results['1yr']['stats']['prob_50_return']:.0f}%</div><div class="label">Prob &gt;50% 1yr</div></div>
    <div class="kpi"><div class="value">{half_kelly:.1f}×</div><div class="label">Half-Kelly</div></div>
    <div class="kpi"><div class="value">${sim_results['5yr']['stats']['median_terminal']:,.0f}</div><div class="label">5yr Median Wealth</div></div>
</div>

<h2>Wealth Fan Chart (5-Year)</h2>
{fan_svg}

<h2>Forward CAGR & Risk Statistics</h2>
<table>
    <thead><tr><th>Horizon</th><th>Median CAGR</th><th>P5</th><th>P25</th><th>P75</th><th>P95</th><th>Prob &gt;50%</th><th>Prob &gt;20% DD</th><th>Median DD</th><th>P95 DD</th><th>Prob Profit</th></tr></thead>
    <tbody>{stats_rows}</tbody>
</table>

<h2>Terminal Wealth ($100K Start)</h2>
<table>
    <thead><tr><th>Horizon</th><th>P5 (Worst Case)</th><th>Median</th><th>P95 (Best Case)</th></tr></thead>
    <tbody>{wealth_rows}</tbody>
</table>

<h2>CAGR Distributions</h2>
<div class="hist-row">
{hist_svgs}
</div>

<h2>Max Drawdown Distribution (5yr)</h2>
<div class="hist-row">
{dd_hist}
</div>

<h2>Prolonged Bear Scenario (2-Year, 2022-Style)</h2>
<div class="callout warn">
    <strong>Stress test:</strong> What if the portfolio enters a prolonged bear market (2022-style) for 2 full years?<br>
    Using only bear/high_vol/neutral regime returns from historical data.
</div>
<table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
        <tr><td>Median CAGR</td><td style="color:{'#16a34a' if bear['median_cagr']>0 else '#dc2626'}">{bear['median_cagr']:+.1f}%</td></tr>
        <tr><td>P5 CAGR (worst)</td><td style="color:#dc2626">{bear['p5_cagr']:+.1f}%</td></tr>
        <tr><td>P95 CAGR (best)</td><td>{bear['p95_cagr']:+.1f}%</td></tr>
        <tr><td>Prob Profitable</td><td style="font-weight:700">{bear['prob_profit']:.0f}%</td></tr>
        <tr><td>Median Max DD</td><td>{bear['median_dd']:.1f}%</td></tr>
        <tr><td>P95 Max DD</td><td style="color:{'#dc2626' if bear['p95_dd']>20 else '#ca8a04'}">{bear['p95_dd']:.1f}%</td></tr>
        <tr><td>Worst DD</td><td style="color:#dc2626">{bear['worst_dd']:.1f}%</td></tr>
        <tr><td>Prob &gt;20% DD</td><td>{bear['prob_20_dd']:.1f}%</td></tr>
        <tr><td>Median Terminal ($100K)</td><td>${bear['median_terminal']:,.0f}</td></tr>
    </tbody>
</table>

<h2>Kelly-Optimal Position Sizing</h2>
<div class="callout">
    <strong>Full Kelly:</strong> {full_kelly:.1f}× leverage — mathematically optimal but high variance<br>
    <strong>Half Kelly:</strong> {half_kelly:.1f}× leverage — recommended for practical use (lower variance, ~75% of growth)<br>
    <strong>Annual Edge:</strong> {edge:.1f}% — expected excess return above risk-free<br><br>
    <em>Current portfolio runs at 1.37× avg leverage (v4) — {'within' if half_kelly >= 1.37 else 'above'} half-Kelly bounds.</em>
</div>

<div class="footer">
    PilotAI Credit Spreads — Monte Carlo Forward Simulation<br>
    {N_SIMS:,} block-bootstrap paths (block={BLOCK_SIZE}d) on Ultimate Portfolio v4 daily returns.<br>
    Not a backtest — forward-looking confidence intervals from resampled historical distribution.
</div>

</body></html>"""


def main():
    print("=" * 72)
    print("Monte Carlo Forward Simulation — Ultimate Portfolio v4")
    print("=" * 72)

    print("\n[1/5] Getting v4 return stream...")
    rets, states = get_v4_returns()

    print("\n[2/5] Running Monte Carlo simulations...")
    sim_results = run_simulations(rets)
    sim_results["_rets"] = rets  # for fan chart

    print("\n[3/5] Running bear scenario (2yr prolonged)...")
    bear = run_bear_scenario(rets, states)

    print("\n[4/5] Computing Kelly-optimal sizing...")
    kelly = compute_kelly(rets)

    print(f"\n{'━' * 56}")
    for label in ["1yr", "3yr", "5yr"]:
        s = sim_results[label]["stats"]
        print(f"  {label.upper():4s}  Median CAGR: {s['median_cagr']:6.1f}%  "
              f"P5: {s['p5_cagr']:6.1f}%  P95: {s['p95_cagr']:6.1f}%  "
              f"Prob>50%: {s['prob_50_return']:.0f}%  P95 DD: {s['p95_dd']:.1f}%")

    print(f"\n  BEAR (2yr): Median CAGR={bear['median_cagr']:+.1f}%  "
          f"Prob profit={bear['prob_profit']:.0f}%  P95 DD={bear['p95_dd']:.1f}%")
    print(f"  KELLY: full={kelly[0]:.1f}× half={kelly[1]:.1f}× edge={kelly[2]:.1f}%")
    print(f"{'━' * 56}")

    print("\n[5/5] Generating report...")
    html = generate_html(sim_results, bear, kelly)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}")


if __name__ == "__main__":
    main()
