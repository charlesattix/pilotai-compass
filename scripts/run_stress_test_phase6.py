#!/usr/bin/env python3
"""
Phase 6 Stress Test Runner — real IronVault data, corrected Sharpe.

Runs compass/stress_test.py full suite on combined portfolio:
  - EXP-400 (Champion): 246 trades, $123K PnL
  - EXP-401 (Blend):    415 trades, $30K PnL
  - Combined 661 trades, $154K PnL, 2020-2025

Generates:
  1. Monte Carlo (10K paths, block-bootstrap)
  2. 4 crisis scenarios (COVID, 2022 bear, flash crash, VIX spike)
  3. Sensitivity sweeps on risk_pct, spread_width, stop_loss_mult

Success criteria:
  - 5th-pctile MC DD ≤ 30%
  - All 4 crisis scenarios DD ≤ 40%
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.stress_test import StressTester, CRISIS_SCENARIOS
from compass.metrics import annualized_sharpe, full_metrics

REPORT_PATH = ROOT / "reports" / "stress_test_report.html"
TRADING_DAYS = 252
CAPITAL = 100_000


def build_daily_returns_from_trades() -> pd.Series:
    """Build daily portfolio return stream from real IronVault trades.

    Uses overlapping-positions model: for each day, sum the pro-rated PnL
    of all active positions, converting to a return fraction of capital.
    """
    dfs = []
    for f in ["compass/training_data_exp400.csv", "compass/training_data_exp401.csv"]:
        df = pd.read_csv(ROOT / f)
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True).sort_values("entry_date").reset_index(drop=True)

    first = all_df["entry_date"].min()
    last = all_df["exit_date"].max()
    all_days = pd.bdate_range(first, last)

    # Pre-compute each trade's linear daily PnL
    trade_daily_pnls = []
    for _, row in all_df.iterrows():
        hold = max(1, int(row["hold_days"]))
        daily = float(row["pnl"]) / hold
        trade_daily_pnls.append({
            "entry": pd.Timestamp(row["entry_date"]),
            "exit": pd.Timestamp(row["exit_date"]),
            "daily_pnl": daily,
        })

    # Daily portfolio returns — sum of active positions
    equity = CAPITAL
    daily_rets = []
    for day in all_days:
        day_pnl = sum(
            t["daily_pnl"] for t in trade_daily_pnls
            if t["entry"] <= day <= t["exit"]
        )
        ret = day_pnl / equity if equity > 0 else 0
        equity += day_pnl
        daily_rets.append(ret)

    return pd.Series(daily_rets, index=all_days, name="portfolio_returns")


def run_sensitivity_sweeps(tester: StressTester) -> dict:
    """Custom sensitivity sweeps for risk_pct, spread_width, stop_loss_mult.

    Uses a return-scaling heuristic: scaling risk_pct scales all daily returns
    (and vol). spread_width changes max loss per trade. stop_loss_mult changes
    the skew of the return distribution.
    """
    def scale_returns(returns, factor):
        return returns * factor

    sweeps = {}
    base_returns = tester.returns
    base_metrics = full_metrics(base_returns)

    # Risk pct sweep
    risk_pcts = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    risk_results = []
    for r in risk_pcts:
        scaled = scale_returns(base_returns, r)
        m = full_metrics(scaled)
        risk_results.append({
            "param_value": r,
            "cagr_pct": m["cagr_pct"],
            "sharpe": m["sharpe"],
            "max_dd_pct": m["max_dd_pct"],
            "vol_pct": m["vol_pct"],
        })
    sweeps["risk_pct"] = risk_results

    # Spread width sweep (inverse scaling: wider spreads = lower per-trade return pct)
    widths = [2.0, 3.0, 5.0, 7.0, 10.0]
    width_results = []
    base_width = 5.0
    for w in widths:
        factor = base_width / w
        scaled = scale_returns(base_returns, factor)
        m = full_metrics(scaled)
        width_results.append({
            "param_value": w,
            "cagr_pct": m["cagr_pct"],
            "sharpe": m["sharpe"],
            "max_dd_pct": m["max_dd_pct"],
            "vol_pct": m["vol_pct"],
        })
    sweeps["spread_width"] = width_results

    # Stop loss multiplier sweep — tighter stops = cut tails, lower DD, slight CAGR hit
    stops = [1.0, 1.5, 2.0, 2.5, 3.0]
    stop_results = []
    for s in stops:
        scaled = base_returns.copy()
        # Tight stops cap the worst-day loss; loose stops let them run
        tail_cap = -0.04 / max(s, 0.5)  # at stop=1.0, cap=-4%; stop=3.0, cap=-1.3%
        scaled = np.where(scaled < tail_cap, tail_cap, scaled)
        m = full_metrics(scaled)
        stop_results.append({
            "param_value": s,
            "cagr_pct": m["cagr_pct"],
            "sharpe": m["sharpe"],
            "max_dd_pct": m["max_dd_pct"],
            "vol_pct": m["vol_pct"],
        })
    sweeps["stop_loss_mult"] = stop_results

    sweeps["baseline"] = base_metrics
    return sweeps


def generate_html(mc: dict, crisis: list, sensitivity: dict, base_metrics: dict) -> str:
    # Target checks
    mc_dd_p5 = abs(mc["max_drawdown"]["percentiles_pct"].get("p5", 0))
    mc_pass = mc_dd_p5 <= 30

    crisis_pass = all(abs(c["portfolio_drawdown_pct"]) <= 40 for c in crisis)
    all_pass = mc_pass and crisis_pass

    # Crisis rows
    crisis_rows = ""
    for c in crisis:
        dd = abs(c["portfolio_drawdown_pct"])
        sc = "#16a34a" if dd <= 40 else "#dc2626"
        crisis_rows += f"""<tr>
            <td style="font-weight:600">{c['name']}</td>
            <td>{c['n_days']}</td>
            <td>{c['underlying_drawdown_pct']:.1f}%</td>
            <td style="font-weight:700;color:{sc}">{c['portfolio_drawdown_pct']:.1f}%</td>
            <td>{c.get('vix_start','?')} → {c.get('vix_peak','?')}</td>
            <td>{c.get('estimated_recovery_days','—') or '—'}</td>
            <td style="color:{sc};font-weight:700">{'PASS' if dd <= 40 else 'FAIL'}</td>
        </tr>"""

    # MC percentile rows
    dd_pcts = mc["max_drawdown"]["percentiles_pct"]
    tw_pcts = mc["terminal_wealth"]["percentiles"]
    mc_rows = ""
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        key = f"p{p}"
        mc_rows += f"""<tr>
            <td>P{p}</td>
            <td>${tw_pcts.get(key, 0):,.0f}</td>
            <td style="color:{'#dc2626' if abs(dd_pcts.get(key, 0))>30 else '#16a34a'}">{dd_pcts.get(key, 0):.1f}%</td>
        </tr>"""

    # Sensitivity sweeps
    def _sens_rows(sweep, unit=""):
        r = ""
        for x in sweep:
            r += f"""<tr>
                <td>{x['param_value']}{unit}</td>
                <td>{x['cagr_pct']:.1f}%</td>
                <td>{x['sharpe']:.2f}</td>
                <td>{x['max_dd_pct']:.1f}%</td>
                <td>{x['vol_pct']:.1f}%</td>
            </tr>"""
        return r

    risk_rows = _sens_rows(sensitivity["risk_pct"], "×")
    width_rows = _sens_rows(sensitivity["spread_width"], " pts")
    stop_rows = _sens_rows(sensitivity["stop_loss_mult"], "×")

    vc = "#16a34a" if all_pass else "#ca8a04"
    verdict = "ALL TARGETS MET" if all_pass else "SOME TARGETS MISSED"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 6 Stress Test — EXP-400/401 Combined</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1000px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.5; }}
  h1 {{ font-size:1.8em; color:#0f172a; margin-bottom:4px; }}
  h2 {{ color:#334155; margin-top:2.5em; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  h3 {{ color:#475569; margin-top:1.5em; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:24px; }}
  .verdict {{ text-align:center; padding:14px; border-radius:8px; font-size:1.1rem; font-weight:800;
              background:{vc}10; color:{vc}; border:2px solid {vc}40; margin-bottom:24px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:18px;
          text-align:center; flex:1; min-width:130px; }}
  .kpi .value {{ font-size:1.6em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; }} .warn {{ color:#ca8a04; }} .bad {{ color:#dc2626; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:10px 12px; text-align:right; font-weight:600; color:#475569;
       border-bottom:2px solid #cbd5e1; font-size:0.78em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .callout {{ border-radius:8px; padding:16px; margin:16px 0; font-size:0.88rem; line-height:1.7; }}
  .callout.ok {{ background:#f0fdf4; border:1px solid #bbf7d0; }}
  .callout.warn {{ background:#fffbeb; border:1px solid #fde68a; }}
  .footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.78em; color:#94a3b8; text-align:center; }}
</style></head><body>

<h1>Phase 6 Stress Test Report</h1>
<div class="subtitle">EXP-400/401 combined portfolio (661 real IronVault trades) | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="verdict">{verdict}</div>

<div class="kpi-row">
    <div class="kpi"><div class="value {'good' if base_metrics['cagr_pct']>0 else 'bad'}">{base_metrics['cagr_pct']:.1f}%</div><div class="label">Base CAGR</div></div>
    <div class="kpi"><div class="value">{base_metrics['sharpe']:.2f}</div><div class="label">Sharpe (correct)</div></div>
    <div class="kpi"><div class="value">{base_metrics['max_dd_pct']:.1f}%</div><div class="label">Historical DD</div></div>
    <div class="kpi"><div class="value {'good' if mc_dd_p5<=30 else 'bad'}">{mc_dd_p5:.1f}%</div><div class="label">MC P5 DD</div>
        <div style="font-size:0.68em;color:#64748b">{'PASS' if mc_pass else 'FAIL'} ≤30%</div></div>
    <div class="kpi"><div class="value">{mc['prob_profit']*100:.0f}%</div><div class="label">Prob Profit</div></div>
    <div class="kpi"><div class="value">{mc['prob_ruin_50pct']*100:.1f}%</div><div class="label">Prob Ruin (50%)</div></div>
</div>

<h2>Monte Carlo Simulation ({mc['n_simulations']:,} paths, block={mc['block_size']}d)</h2>
<table>
    <thead><tr><th>Percentile</th><th>Terminal Wealth</th><th>Max Drawdown</th></tr></thead>
    <tbody>{mc_rows}</tbody>
</table>

<div class="callout {'ok' if mc_pass else 'warn'}">
    <strong>Target: 5th-pctile DD ≤ 30%</strong> — Actual: {mc_dd_p5:.1f}% {'✓ PASS' if mc_pass else '✗ FAIL'}<br>
    Median terminal wealth: ${mc['terminal_wealth']['median']:,.0f}<br>
    Mean Sharpe across paths: {mc['sharpe_ratio']['mean']:.2f}
</div>

<h2>Crisis Scenarios (4 historical)</h2>
<table>
    <thead><tr><th>Scenario</th><th>Days</th><th>Underlying DD</th><th>Portfolio DD</th><th>VIX</th><th>Recovery</th><th>≤40%?</th></tr></thead>
    <tbody>{crisis_rows}</tbody>
</table>

<div class="callout {'ok' if crisis_pass else 'warn'}">
    <strong>Target: all crisis DDs ≤ 40%</strong> — {'✓ All 4 PASS' if crisis_pass else '✗ Some FAIL'}<br>
    Portfolio beta (credit spreads): 1.5× underlying (short gamma amplifies losses)
</div>

<h2>Sensitivity Analysis</h2>

<h3>Risk Pct Sweep</h3>
<table>
    <thead><tr><th>Risk Multiplier</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th></tr></thead>
    <tbody>{risk_rows}</tbody>
</table>

<h3>Spread Width Sweep</h3>
<table>
    <thead><tr><th>Spread Width</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th></tr></thead>
    <tbody>{width_rows}</tbody>
</table>

<h3>Stop Loss Multiplier Sweep</h3>
<table>
    <thead><tr><th>Stop Mult</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th></tr></thead>
    <tbody>{stop_rows}</tbody>
</table>

<div class="footer">
    Phase 6 Stress Test — compass/stress_test.py on 661 real IronVault trades<br>
    All Sharpe via compass/metrics.py (correct arithmetic mean). Monte Carlo block-bootstrap preserves autocorrelation.
</div>

</body></html>"""


def main():
    print("=" * 72)
    print("Phase 6 Stress Test — EXP-400/401 Combined")
    print("=" * 72)

    print("\n[1/5] Building daily returns from real IronVault trades...")
    daily_rets = build_daily_returns_from_trades()
    print(f"  → {len(daily_rets)} business days, {daily_rets.index[0].date()} → {daily_rets.index[-1].date()}")

    base_metrics = full_metrics(daily_rets.values)
    print(f"  → Base: CAGR={base_metrics['cagr_pct']:.1f}%  "
          f"Sharpe={base_metrics['sharpe']:.2f}  DD={base_metrics['max_dd_pct']:.1f}%")

    print("\n[2/5] Running Monte Carlo (10K paths, block-bootstrap)...")
    tester = StressTester(
        daily_returns=daily_rets.values,
        starting_capital=CAPITAL,
        n_simulations=10_000,
        block_size=5,
        seed=42,
    )
    mc = tester.run_monte_carlo()
    print(f"  → P5 DD: {mc['max_drawdown']['percentiles_pct']['p5']:.1f}%")
    print(f"  → Median terminal: ${mc['terminal_wealth']['median']:,.0f}")
    print(f"  → Prob profit: {mc['prob_profit']*100:.0f}%")

    print("\n[3/5] Running crisis scenarios...")
    crisis = tester.run_crisis_scenarios()
    for c in crisis:
        status = "PASS" if abs(c["portfolio_drawdown_pct"]) <= 40 else "FAIL"
        print(f"  {c['name']:32s}  Port DD: {c['portfolio_drawdown_pct']:6.1f}%  [{status}]")

    print("\n[4/5] Running sensitivity sweeps...")
    sensitivity = run_sensitivity_sweeps(tester)
    for param, sweep in sensitivity.items():
        if param == "baseline":
            continue
        print(f"\n  {param}:")
        for x in sweep:
            print(f"    {x['param_value']:>6}: CAGR={x['cagr_pct']:6.1f}%  "
                  f"Sharpe={x['sharpe']:.2f}  DD={x['max_dd_pct']:.1f}%")

    print(f"\n{'━'*56}")
    mc_dd_p5 = abs(mc['max_drawdown']['percentiles_pct']['p5'])
    crisis_pass = all(abs(c['portfolio_drawdown_pct']) <= 40 for c in crisis)
    mc_pass = mc_dd_p5 <= 30
    print(f"  VERDICT:")
    print(f"    MC P5 DD ≤30%:       {'PASS' if mc_pass else 'FAIL'} ({mc_dd_p5:.1f}%)")
    print(f"    All crises ≤40%:     {'PASS' if crisis_pass else 'FAIL'}")
    print(f"{'━'*56}")

    print("\n[5/5] Generating HTML report...")
    html = generate_html(mc, crisis, sensitivity, base_metrics)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}")


if __name__ == "__main__":
    main()
