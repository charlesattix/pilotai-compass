#!/usr/bin/env python3
"""
EXP-1220 Comprehensive Stress Test at 1.2x Leverage.

Runs:
  1. Monte Carlo — 10,000 block-bootstrap paths (block=5)
  2. Crisis scenario replay — COVID, 2022 bear, flash crash, VIX spike
  3. Sensitivity analysis — risk_pct, spread_width, hedge_ratio, leverage 0.5x-3x
  4. Tail risk metrics — CVaR, max consecutive losses, longest DD duration

North Star threshold: 5th-percentile MC DD <= 12% at 1.2x leverage.

Output: reports/exp1220_stress_test.html + JSON summary
"""

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.backtester import _yf_download_safe
from compass.stress_test import (
    StressTester,
    CRISIS_SCENARIOS,
    _returns_to_equity,
    _max_drawdown,
    _sharpe_ratio,
    _cagr,
    _calmar_ratio,
)
from compass.tail_risk_protector import TailRiskProtector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LEVERAGE = 1.2
N_SIMULATIONS = 10_000
BLOCK_SIZE = 5
STARTING_CAPITAL = 100_000
REPORT_PATH = ROOT / "reports" / "exp1220_stress_test.html"


# ---------------------------------------------------------------------------
# Step 0: Generate EXP-1220 protected daily returns from real data
# ---------------------------------------------------------------------------

def generate_protected_returns() -> np.ndarray:
    """Reproduce EXP-1220-real protected returns from real market data."""
    log.info("Generating EXP-1220 protected returns from real market data...")

    def _fetch(ticker, start, end):
        df = _yf_download_safe(ticker, start, end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    spy = _fetch("SPY", "2019-06-01", "2025-12-31")
    vix_df = _fetch("^VIX", "2019-06-01", "2025-12-31")
    vix3m_df = _fetch("^VIX3M", "2019-06-01", "2025-12-31")

    spy_close = spy["Close"].dropna()
    spy_returns = spy_close.pct_change().dropna()
    vix = vix_df["Close"].dropna()
    vix3m = vix3m_df["Close"].dropna()

    common = spy_returns.index.intersection(vix.index).intersection(vix3m.index).sort_values()
    spy_returns = spy_returns.reindex(common).fillna(0)
    vix = vix.reindex(common).ffill().bfill()
    vix3m = vix3m.reindex(common).ffill().bfill()

    hyg_tlt_proxy = vix * 0.4 + 1.5
    skew_proxy = (vix / vix3m.replace(0, 1)) * 8.0
    rolling_corr = spy_returns.rolling(20, min_periods=10).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0
    ).fillna(0.3)
    cross_corr_proxy = (rolling_corr + 1) / 2
    momentum = spy_close.pct_change().rolling(20).sum().reindex(common).fillna(0)

    data = {
        "vix": vix, "vix_3m": vix3m,
        "hyg_tlt_spread": hyg_tlt_proxy, "skew_25d": skew_proxy,
        "cross_corr": cross_corr_proxy, "momentum": momentum,
        "spy_returns": spy_returns,
    }

    protector = TailRiskProtector(lookback=252)
    states = protector.assess(data)
    spy_aligned = spy_returns.reindex([s.date for s in states]).fillna(0)

    # Build protected returns: pure sizing overlay (no hedge profit/cost)
    prot_returns = np.zeros(len(states))
    for i, state in enumerate(states):
        prot_returns[i] = float(spy_aligned.iloc[i]) * state.size_multiplier

    # Filter to 2020+ only (skip warmup)
    dates = [s.date for s in states]
    mask = np.array([d.year >= 2020 for d in dates])
    prot_returns = prot_returns[mask]

    log.info("Protected returns: %d days (2020-2025), mean=%.4f, std=%.4f",
             len(prot_returns), prot_returns.mean(), prot_returns.std())
    return prot_returns


# ---------------------------------------------------------------------------
# Step 4: Tail risk metrics (CVaR, consecutive losses, DD duration)
# ---------------------------------------------------------------------------

def compute_tail_risk_metrics(returns: np.ndarray, capital: float) -> dict:
    """Compute advanced tail risk metrics."""
    equity = _returns_to_equity(returns, capital)

    # CVaR (Expected Shortfall) at 95% and 99%
    sorted_rets = np.sort(returns)
    n = len(sorted_rets)
    cvar_95_idx = max(1, int(n * 0.05))
    cvar_99_idx = max(1, int(n * 0.01))
    cvar_95 = float(sorted_rets[:cvar_95_idx].mean()) * 100
    cvar_99 = float(sorted_rets[:cvar_99_idx].mean()) * 100

    # Maximum consecutive losses
    max_consec = 0
    current = 0
    for r in returns:
        if r < 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    # Longest drawdown duration (days from peak to new peak)
    peak = np.maximum.accumulate(equity)
    in_dd = equity < peak
    max_dd_duration = 0
    current_dd = 0
    for is_dd in in_dd:
        if is_dd:
            current_dd += 1
            max_dd_duration = max(max_dd_duration, current_dd)
        else:
            current_dd = 0

    # Skewness and kurtosis of returns
    skew = float(pd.Series(returns).skew())
    kurt = float(pd.Series(returns).kurtosis())

    # Worst single day
    worst_day = float(returns.min()) * 100

    # VaR at 95% and 99%
    var_95 = float(np.percentile(returns, 5)) * 100
    var_99 = float(np.percentile(returns, 1)) * 100

    return {
        "cvar_95_pct": round(cvar_95, 3),
        "cvar_99_pct": round(cvar_99, 3),
        "var_95_pct": round(var_95, 3),
        "var_99_pct": round(var_99, 3),
        "max_consecutive_losses": max_consec,
        "longest_dd_duration_days": max_dd_duration,
        "skewness": round(skew, 3),
        "excess_kurtosis": round(kurt, 3),
        "worst_day_pct": round(worst_day, 3),
    }


# ---------------------------------------------------------------------------
# Custom sensitivity sweeps for EXP-1220
# ---------------------------------------------------------------------------

CUSTOM_SWEEPS = {
    "position_size_pct": {
        "label": "Risk Per Trade (%)",
        "values": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "baseline": 5.0,
        "description": "Position sizing as pct of account",
    },
    "spread_width": {
        "label": "Spread Width ($)",
        "values": [2.5, 5.0, 7.5, 10.0, 15.0, 20.0],
        "baseline": 5.0,
        "description": "Credit spread width in dollars",
    },
    "profit_target_pct": {
        "label": "Hedge Ratio / Profit Target (%)",
        "values": [25, 40, 50, 60, 75, 90],
        "baseline": 50,
        "description": "Profit take level (proxy for hedge aggressiveness)",
    },
    "stop_loss_multiplier": {
        "label": "Leverage Proxy (Stop Loss Mult)",
        "values": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        "baseline": 3.5,
        "description": "Stop loss multiplier (lower = tighter risk control)",
    },
}


def leverage_sweep_fn(base_returns: np.ndarray):
    """Return a param_sweep_fn that scales returns by leverage."""
    def fn(param_name: str, value: float) -> np.ndarray:
        if param_name == "leverage":
            return base_returns * value / LEVERAGE  # re-scale from current leverage
        return base_returns
    return fn


LEVERAGE_SWEEP = {
    "leverage": {
        "label": "Leverage Multiplier",
        "values": [0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "baseline": 1.2,
        "description": "Portfolio leverage (1.0=unlevered)",
    },
}


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

def generate_html_report(
    mc: dict,
    crisis: list,
    sensitivity: dict,
    summary: dict,
    tail_risk: dict,
    leverage: float,
) -> str:
    """Generate investor-grade HTML stress test report."""

    # MC percentiles
    mc_dd = mc["max_drawdown"]
    mc_tw = mc["terminal_wealth"]
    mc_sh = mc["sharpe_ratio"]
    p5_dd = abs(mc_dd["percentiles_pct"].get("p5", 0))
    p95_dd = abs(mc_dd["percentiles_pct"].get("p95", 0))
    north_star_pass = p5_dd <= 12.0

    # Crisis table rows
    crisis_rows = ""
    for c in crisis:
        hedged = c.get("hedged_portfolio_drawdown_pct")
        hedged_str = f"{hedged:.1f}%" if hedged is not None else "&mdash;"
        recovery = c.get("estimated_recovery_days")
        recovery_str = f"{recovery}d" if recovery else "N/A"
        crisis_rows += (
            f"<tr><td>{c['name']}</td><td>{c['n_days']}</td>"
            f"<td>{c['underlying_drawdown_pct']:.1f}%</td>"
            f"<td>{c['portfolio_drawdown_pct']:.1f}%</td>"
            f"<td>{hedged_str}</td>"
            f"<td>{c['vix_start']:.0f} → {c['vix_peak']:.0f}</td>"
            f"<td>{recovery_str}</td></tr>\n"
        )

    # Sensitivity table rows
    sensitivity_rows = ""
    for param_name, param_data in sensitivity.items():
        for r in param_data["results"]:
            highlight = ' style="background:#f0fdf4;font-weight:600"' if r["is_baseline"] else ""
            sensitivity_rows += (
                f'<tr{highlight}><td>{param_data["label"]}</td><td>{r["value"]}</td>'
                f'<td>{r["sharpe"]:.3f}</td><td>{abs(r["max_dd_pct"]):.2f}%</td>'
                f'<td>{r["cagr_pct"]:.2f}%</td><td>{r["calmar"]:.3f}</td></tr>\n'
            )

    # Sample paths SVG
    paths_svg = ""
    if mc.get("sample_paths"):
        w, h = 800, 300
        pad = 50
        paths = mc["sample_paths"][:100]
        max_len = max(len(p) for p in paths)
        all_vals = [v for p in paths for v in p]
        y_min, y_max = min(all_vals), max(all_vals)
        if y_max == y_min:
            y_max = y_min + 1

        def tx(i): return pad + i / max(max_len - 1, 1) * (w - 2 * pad)
        def ty(v): return pad + (1 - (v - y_min) / (y_max - y_min)) * (h - 2 * pad)

        parts = [f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;margin:1rem 0">']
        # Draw paths
        for p in paths[:50]:
            d = " ".join(f"{'M' if j == 0 else 'L'}{tx(j):.1f},{ty(p[j]):.1f}" for j in range(len(p)))
            parts.append(f'<path d="{d}" fill="none" stroke="#94a3b8" stroke-width="0.5" opacity="0.3"/>')
        # Median path
        median_idx = len(paths) // 2
        if median_idx < len(paths):
            d = " ".join(f"{'M' if j == 0 else 'L'}{tx(j):.1f},{ty(paths[median_idx][j]):.1f}" for j in range(len(paths[median_idx])))
            parts.append(f'<path d="{d}" fill="none" stroke="#059669" stroke-width="2"/>')
        # Starting capital line
        parts.append(f'<line x1="{pad}" y1="{ty(STARTING_CAPITAL):.1f}" x2="{w-pad}" y2="{ty(STARTING_CAPITAL):.1f}" stroke="#ef4444" stroke-width="1" stroke-dasharray="4,4"/>')
        parts.append(f'<text x="{w-pad+5}" y="{ty(STARTING_CAPITAL):.1f}" font-size="10" fill="#ef4444">${STARTING_CAPITAL:,.0f}</text>')
        parts.append("</svg>")
        paths_svg = "\n".join(parts)

    north_star_badge = ('<span style="background:#dcfce7;color:#166534;padding:2px 8px;border-radius:4px;font-weight:600">PASS</span>'
                        if north_star_pass else
                        '<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:4px;font-weight:600">FAIL</span>')

    risk_colors = {"LOW": "#059669", "MODERATE": "#d97706", "HIGH": "#dc2626", "CRITICAL": "#7f1d1d"}
    risk_color = risk_colors.get(summary["risk_rating"], "#64748b")

    hist = summary["historical"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1220 Stress Test — {leverage}x Leverage</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#fff;color:#1e293b;line-height:1.65;font-size:15px}}
.page{{max-width:1060px;margin:0 auto;padding:2.5rem 2rem}}
h1{{font-size:2rem;color:#0f172a;font-weight:800;border-bottom:3px solid #0f172a;padding-bottom:.6rem;margin-bottom:.5rem}}
h2{{font-size:1.3rem;color:#0f172a;margin:2.5rem 0 .8rem;padding-bottom:.3rem;border-bottom:2px solid #e2e8f0;font-weight:700}}
h3{{font-size:1.05rem;color:#334155;margin:1.5rem 0 .5rem}}
p{{margin:.5rem 0}}
.subtitle{{color:#64748b;margin-bottom:2rem;font-size:.95rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.88rem}}
th{{background:#f8fafc;color:#334155;font-weight:600;padding:10px 12px;text-align:right;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}}
td{{padding:9px 12px;text-align:right;border-bottom:1px solid #f1f5f9}}
td:first-child{{text-align:left;font-weight:500}}
tr:hover{{background:#f8fafc}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.5rem 0}}
.m{{text-align:center;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1.2rem .8rem}}
.m .v{{font-size:1.8rem;font-weight:800;color:#0f172a}}.m .l{{font-size:.78rem;color:#64748b;margin-top:2px}}
.m.green .v{{color:#059669}}.m.red .v{{color:#dc2626}}.m.amber .v{{color:#d97706}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1.2rem 1.5rem;margin:1rem 0}}
.highlight{{background:#f0fdf4;border-color:#86efac}}
.warn{{background:#fffbeb;border-color:#fde68a}}
.danger{{background:#fef2f2;border-color:#fecaca}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}}
.footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid #e2e8f0;color:#94a3b8;font-size:.8rem;text-align:center}}
@media(max-width:700px){{.metrics{{grid-template-columns:1fr 1fr}}.grid2{{grid-template-columns:1fr}}}}
@media print{{body{{font-size:11pt}}.page{{max-width:100%;padding:1rem}}}}
</style>
</head>
<body>
<div class="page">

<h1>EXP-1220 Tail Risk Hedging &mdash; Stress Test at {leverage}x Leverage</h1>
<p class="subtitle">10,000 MC paths &middot; Block-bootstrap (block=5) &middot; 4 crisis scenarios &middot; 4 sensitivity sweeps &middot; Generated {datetime.now().strftime('%Y-%m-%d')}</p>

<div class="metrics">
<div class="m {'green' if north_star_pass else 'red'}"><div class="v">{p5_dd:.1f}%</div><div class="l">P5 MC Max Drawdown</div></div>
<div class="m green"><div class="v">{mc['prob_profit']*100:.1f}%</div><div class="l">Prob of Profit</div></div>
<div class="m green"><div class="v">{hist['sharpe']:.2f}</div><div class="l">Historical Sharpe</div></div>
<div class="m" style="border-color:{risk_color}"><div class="v" style="color:{risk_color}">{summary['risk_rating']}</div><div class="l">Risk Rating</div></div>
</div>

<div class="card {'highlight' if north_star_pass else 'danger'}">
<strong>North Star Check (P5 DD &le; 12% at {leverage}x):</strong> {north_star_badge}
&mdash; 5th-percentile Monte Carlo drawdown = <strong>{p5_dd:.1f}%</strong> vs 12% threshold.
Historical CAGR = {hist['cagr_pct']:.1f}%, Sharpe = {hist['sharpe']:.2f}, Max DD = {abs(hist['max_drawdown_pct']):.1f}%.
</div>

<!-- MONTE CARLO -->
<h2>1. Monte Carlo Simulation ({mc['n_simulations']:,} paths)</h2>

{paths_svg}

<div class="grid2">
<div class="card">
<h3>Terminal Wealth Distribution</h3>
<table>
<tr><td>Mean</td><td>${mc_tw['mean']:,.0f}</td></tr>
<tr><td>Median (P50)</td><td>${mc_tw['percentiles'].get('p50', 0):,.0f}</td></tr>
<tr><td>5th Percentile</td><td>${mc_tw['percentiles'].get('p5', 0):,.0f}</td></tr>
<tr><td>95th Percentile</td><td>${mc_tw['percentiles'].get('p95', 0):,.0f}</td></tr>
<tr><td>Worst Case</td><td>${mc_tw['min']:,.0f}</td></tr>
<tr><td>Best Case</td><td>${mc_tw['max']:,.0f}</td></tr>
</table>
</div>
<div class="card">
<h3>Drawdown Distribution</h3>
<table>
<tr><td>Median DD</td><td>{abs(mc_dd['median_pct']):.1f}%</td></tr>
<tr><td>5th Percentile DD</td><td>{p5_dd:.1f}%</td></tr>
<tr><td>95th Percentile DD</td><td>{p95_dd:.1f}%</td></tr>
<tr><td>Worst Case DD</td><td>{abs(mc_dd['worst_pct']):.1f}%</td></tr>
<tr><td>Mean Sharpe</td><td>{mc_sh['mean']:.3f}</td></tr>
<tr><td>Prob of Ruin (&gt;50% loss)</td><td>{mc['prob_ruin_50pct']*100:.2f}%</td></tr>
</table>
</div>
</div>

<!-- CRISIS SCENARIOS -->
<h2>2. Crisis Scenario Replay</h2>

<table>
<tr><th>Scenario</th><th>Days</th><th>Underlying DD</th><th>Portfolio DD</th><th>Hedged DD</th><th>VIX</th><th>Recovery</th></tr>
{crisis_rows}
</table>

<!-- SENSITIVITY -->
<h2>3. Sensitivity Analysis</h2>

<table>
<tr><th>Parameter</th><th>Value</th><th>Sharpe</th><th>Max DD</th><th>CAGR</th><th>Calmar</th></tr>
{sensitivity_rows}
</table>

<!-- TAIL RISK -->
<h2>4. Tail Risk Metrics</h2>

<div class="grid2">
<div class="card">
<h3>Value at Risk / Expected Shortfall</h3>
<table>
<tr><td>VaR (95%)</td><td>{tail_risk['var_95_pct']:.3f}%</td></tr>
<tr><td>VaR (99%)</td><td>{tail_risk['var_99_pct']:.3f}%</td></tr>
<tr><td>CVaR / ES (95%)</td><td>{tail_risk['cvar_95_pct']:.3f}%</td></tr>
<tr><td>CVaR / ES (99%)</td><td>{tail_risk['cvar_99_pct']:.3f}%</td></tr>
<tr><td>Worst Single Day</td><td>{tail_risk['worst_day_pct']:.3f}%</td></tr>
</table>
</div>
<div class="card">
<h3>Drawdown Characteristics</h3>
<table>
<tr><td>Max Consecutive Losses</td><td>{tail_risk['max_consecutive_losses']} days</td></tr>
<tr><td>Longest DD Duration</td><td>{tail_risk['longest_dd_duration_days']} days</td></tr>
<tr><td>Return Skewness</td><td>{tail_risk['skewness']:.3f}</td></tr>
<tr><td>Excess Kurtosis</td><td>{tail_risk['excess_kurtosis']:.3f}</td></tr>
</table>
</div>
</div>

<div class="footer">
EXP-1220 Tail Risk Hedging Stress Test &middot; {leverage}x Leverage &middot; {N_SIMULATIONS:,} MC Paths &middot;
Block Size {BLOCK_SIZE} &middot; Generated {datetime.now().strftime('%Y-%m-%d')}
</div>

</div>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"EXP-1220 STRESS TEST — {LEVERAGE}x LEVERAGE")
    print("=" * 70)

    # Step 0: Generate protected returns
    base_returns = generate_protected_returns()

    # Apply leverage
    levered_returns = base_returns * LEVERAGE
    log.info("Levered returns (%gx): mean=%.5f, std=%.4f, n=%d",
             LEVERAGE, levered_returns.mean(), levered_returns.std(), len(levered_returns))

    # Step 1: Monte Carlo
    print(f"\n[1/4] Running Monte Carlo ({N_SIMULATIONS:,} paths, block={BLOCK_SIZE})...")
    tester = StressTester(
        levered_returns,
        starting_capital=STARTING_CAPITAL,
        n_simulations=N_SIMULATIONS,
        block_size=BLOCK_SIZE,
        seed=42,
    )

    mc = tester.run_monte_carlo()
    p5_dd = abs(mc["max_drawdown"]["percentiles_pct"].get("p5", 0))
    print(f"  P5 DD: {p5_dd:.1f}%  (North Star threshold: <=12%)")
    print(f"  Median terminal: ${mc['terminal_wealth']['median']:,.0f}")
    print(f"  Prob profit: {mc['prob_profit']*100:.1f}%")
    print(f"  Prob ruin: {mc['prob_ruin_50pct']*100:.2f}%")

    # Step 2: Crisis scenarios
    print("\n[2/4] Running crisis scenario replay...")
    crisis = tester.run_crisis_scenarios()
    for c in crisis:
        print(f"  {c['name']}: portfolio DD={c['portfolio_drawdown_pct']:.1f}%, "
              f"recovery={c.get('estimated_recovery_days', 'N/A')}d")

    # Step 3: Sensitivity analysis with custom sweeps
    print("\n[3/4] Running sensitivity analysis...")
    sensitivity = tester.run_sensitivity_analysis(sweeps=CUSTOM_SWEEPS)

    # Also do explicit leverage sweep with actual return scaling
    lev_sweep_fn = leverage_sweep_fn(levered_returns)
    lev_sensitivity = tester.run_sensitivity_analysis(
        param_sweep_fn=lev_sweep_fn, sweeps=LEVERAGE_SWEEP,
    )
    sensitivity.update(lev_sensitivity)

    for param, data in sensitivity.items():
        sharpes = [r["sharpe"] for r in data["results"]]
        print(f"  {data['label']}: Sharpe range [{min(sharpes):.3f}, {max(sharpes):.3f}]")

    # Step 4: Tail risk metrics
    print("\n[4/4] Computing tail risk metrics...")
    tail_risk = compute_tail_risk_metrics(levered_returns, STARTING_CAPITAL)
    print(f"  CVaR (95%): {tail_risk['cvar_95_pct']:.3f}%")
    print(f"  CVaR (99%): {tail_risk['cvar_99_pct']:.3f}%")
    print(f"  Max consecutive losses: {tail_risk['max_consecutive_losses']} days")
    print(f"  Longest DD duration: {tail_risk['longest_dd_duration_days']} days")

    # Build summary
    summary = tester._build_summary(mc, crisis, sensitivity)

    # Generate HTML report
    print("\nGenerating HTML report...")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html_report(mc, crisis, sensitivity, summary, tail_risk, LEVERAGE)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  Report: {REPORT_PATH}")

    # Save JSON summary
    json_path = REPORT_PATH.with_suffix(".json")
    json_summary = {
        "experiment": "EXP-1220",
        "leverage": LEVERAGE,
        "n_simulations": N_SIMULATIONS,
        "north_star_p5_dd_pct": round(p5_dd, 2),
        "north_star_pass": p5_dd <= 12.0,
        "summary": summary,
        "tail_risk": tail_risk,
        "monte_carlo": {k: v for k, v in mc.items() if k != "sample_paths"},
        "crisis_scenarios": [{k: v for k, v in c.items() if k not in ("equity_path", "hedged_equity_path")} for c in crisis],
    }
    json_path.write_text(json.dumps(json_summary, indent=2, default=str))
    print(f"  JSON:   {json_path}")

    # North Star verdict
    print(f"\n{'='*70}")
    if p5_dd <= 12.0:
        print(f"NORTH STAR: PASS — P5 DD = {p5_dd:.1f}% <= 12% at {LEVERAGE}x leverage")
    else:
        print(f"NORTH STAR: FAIL — P5 DD = {p5_dd:.1f}% > 12% at {LEVERAGE}x leverage")
    print(f"{'='*70}")

    return json_summary


if __name__ == "__main__":
    main()
