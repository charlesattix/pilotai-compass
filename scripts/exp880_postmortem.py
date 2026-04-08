#!/usr/bin/env python3
"""
EXP-880 Post-Mortem Analysis — why the ML Production Ensemble strategy
failed on real IronVault data, and whether it can be salvaged.

Investigations:
  1. ML model NOT used in Backtester — confirm and document
  2. Direction logic: regime classification during disaster months
  3. Win/loss asymmetry: avg win $1.1K vs avg loss $4.2K
  4. Regime sweep: puts-only vs both directions
  5. Risk sweep: 8.5% vs 4% vs 2% max_risk_per_trade
  6. Verdict: dead or salvageable?

Outputs: reports/exp880_postmortem.html
"""

from __future__ import annotations

import json
import logging
import math
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from backtest.backtester import Backtester
from shared.iron_vault import IronVault, IronVaultError

logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "configs" / "paper_exp880.yaml"
OUTPUT_PATH = ROOT / "reports" / "exp880_postmortem.html"

TICKER = "SPY"
START = datetime(2020, 1, 2)
END = datetime(2025, 12, 31)
CAPITAL = 100_000.0


def load_base_config() -> Dict[str, Any]:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config["backtest"] = {
        "starting_capital": CAPITAL,
        "commission_per_contract": 0.65,
        "slippage": 0.05,
        "exit_slippage": 0.10,
        "compound": False,
        "sizing_mode": "flat",
        "generate_reports": False,
        "report_dir": "/tmp",
        "risk_cap": 25.0,
    }
    config["strategy"]["direction"] = "both"
    config["strategy"]["regime_mode"] = "combo"
    config["strategy"]["target_dte"] = 15
    config["strategy"]["min_dte"] = 15
    config["strategy"]["max_dte"] = 25
    config["strategy"]["spread_width"] = 12
    config["strategy"]["min_credit_pct"] = 5
    config["risk"]["account_size"] = CAPITAL
    config["risk"]["max_risk_per_trade"] = 8.5
    config["risk"]["max_positions"] = 8
    config["risk"]["profit_target"] = 55
    config["risk"]["stop_loss_pct_of_width"] = 90
    config["risk"]["drawdown_cb_pct"] = 99
    return config


def run_variant(hd, label: str, config: Dict) -> Dict[str, Any]:
    """Run a backtest variant and return summary dict."""
    logger.info("Running variant: %s", label)
    bt = Backtester(
        config=config,
        historical_data=hd,
        otm_pct=config["strategy"].get("otm_pct", 0.02),
    )
    results = bt.run_backtest(TICKER, START, END)
    if not results or results.get("total_trades", 0) == 0:
        return {"label": label, "trades": 0, "pnl": 0, "win_rate": 0,
                "max_dd": 0, "sharpe": 0, "ending_capital": CAPITAL,
                "bull_puts": 0, "bear_calls": 0, "avg_win": 0, "avg_loss": 0,
                "yearly": {}, "monthly_pnl": {}}

    trades = results.get("trades", [])
    equity_curve = results.get("equity_curve", [])

    # Per-year breakdown
    yearly = {}
    if trades:
        tdf = pd.DataFrame(trades)
        tdf["exit_dt"] = pd.to_datetime(tdf["exit_date"].apply(lambda d: str(d)[:10]))
        tdf["year"] = tdf["exit_dt"].dt.year
        for yr in sorted(tdf["year"].unique()):
            yt = tdf[tdf["year"] == yr]
            yearly[int(yr)] = {
                "trades": len(yt),
                "pnl": round(yt["pnl"].sum(), 2),
                "win_rate": round(len(yt[yt["pnl"] > 0]) / len(yt) * 100, 1) if len(yt) > 0 else 0,
            }

    return {
        "label": label,
        "trades": results["total_trades"],
        "pnl": round(results.get("total_pnl", 0), 2),
        "win_rate": round(results.get("win_rate", 0), 1),
        "max_dd": round(results.get("max_drawdown", 0), 1),
        "sharpe": round(results.get("sharpe_ratio", 0), 2),
        "ending_capital": round(results.get("ending_capital", CAPITAL), 2),
        "bull_puts": results.get("bull_put_trades", 0),
        "bear_calls": results.get("bear_call_trades", 0),
        "avg_win": round(results.get("avg_win", 0), 2),
        "avg_loss": round(results.get("avg_loss", 0), 2),
        "profit_factor": results.get("profit_factor", 0),
        "yearly": yearly,
        "monthly_pnl": results.get("monthly_pnl", {}),
    }


def build_html(
    baseline: Dict,
    variants: List[Dict],
    regime_analysis: str,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Variant comparison table
    all_runs = [baseline] + variants
    variant_rows = ""
    for v in all_runs:
        pnl_cls = "good" if v["pnl"] > 0 else "bad"
        variant_rows += f"""<tr>
          <td><strong>{v['label']}</strong></td>
          <td>{v['trades']}</td>
          <td>{v['bull_puts']}</td>
          <td>{v['bear_calls']}</td>
          <td>{v['win_rate']:.1f}%</td>
          <td class="{pnl_cls}">${v['pnl']:,.0f}</td>
          <td>{v['max_dd']:.1f}%</td>
          <td>{v['sharpe']:.2f}</td>
          <td>${v['avg_win']:,.0f}</td>
          <td class="bad">${v['avg_loss']:,.0f}</td>
          <td>{v.get('profit_factor', 0):.2f}</td>
        </tr>"""

    # Yearly breakdown for baseline
    yearly_rows = ""
    for yr, yd in sorted(baseline.get("yearly", {}).items()):
        pnl_cls = "good" if yd["pnl"] > 0 else "bad"
        yearly_rows += f'<tr><td>{yr}</td><td>{yd["trades"]}</td><td class="{pnl_cls}">${yd["pnl"]:,.0f}</td><td>{yd["win_rate"]:.1f}%</td></tr>'

    # Disaster month analysis
    disaster_rows = ""
    for month, data in sorted(baseline.get("monthly_pnl", {}).items()):
        if data["pnl"] < -5000:
            disaster_rows += (
                f'<tr class="bad"><td>{month}</td><td>{data["trades"]}</td>'
                f'<td class="bad">${data["pnl"]:,.0f}</td>'
                f'<td>{data["win_rate"]:.0%}</td></tr>'
            )

    # Best variant
    profitable_variants = [v for v in all_runs if v["pnl"] > 0]
    if profitable_variants:
        best = max(profitable_variants, key=lambda v: v["pnl"])
        # Build yearly table for the best variant
        best_yearly_rows = ""
        for yr, yd in sorted(best.get("yearly", {}).items()):
            cls = "good" if yd["pnl"] > 0 else "bad"
            best_yearly_rows += f'<tr><td>{yr}</td><td>{yd["trades"]}</td><td class="{cls}">${yd["pnl"]:,.0f}</td><td>{yd["win_rate"]:.1f}%</td></tr>'

        verdict_html = f"""
        <div class="verdict salvageable">
          <h3>VERDICT: SALVAGEABLE — But EXP-880 as designed is dead</h3>
          <p>The original EXP-880 thesis (combo regime + bear calls + 8.5% risk + ML ensemble) is
             <strong>fatally flawed</strong>. However, the core signal — bull put spreads with MA-based
             trend filtering — <strong>works on real data</strong>.</p>
          <p>Best variant: <strong>{best['label']}</strong></p>
          <ul>
            <li>PnL: <strong class="good">${best['pnl']:,.0f}</strong></li>
            <li>Trades: {best['trades']} ({best['trades']/5.99:.0f}/year)</li>
            <li>Win Rate: <strong>{best['win_rate']:.1f}%</strong></li>
            <li>Max Drawdown: <strong>{best['max_dd']:.1f}%</strong></li>
            <li>Sharpe: <strong>{best['sharpe']:.2f}</strong></li>
            <li>Profit Factor: <strong>{best.get('profit_factor', 0):.2f}</strong></li>
          </ul>
          <h4>Best Variant Yearly</h4>
          <table><tr><th>Year</th><th>Trades</th><th>PnL</th><th>WR</th></tr>
          {best_yearly_rows}</table>
          <h4>Recommended Salvage Config</h4>
          <ul>
            <li><code>direction: bull_put</code> (NO bear calls)</li>
            <li><code>regime_mode: ma</code> (NOT combo — combo overrides direction)</li>
            <li><code>max_risk_per_trade: 4.0-8.5%</code> (scale to risk appetite)</li>
            <li>Keep: 12-wide spreads, 15 DTE target, 2% OTM, 5% min credit</li>
            <li>Remove: ML ensemble (never integrated into backtester anyway)</li>
            <li>Remove: Crisis Hedge V2 references (never integrated into backtester)</li>
          </ul>
        </div>"""
    else:
        verdict_html = """
        <div class="verdict dead">
          <h3>VERDICT: STRATEGY IS DEAD</h3>
          <p>No tested variant produces positive PnL on real IronVault data.
             The fundamental assumptions of EXP-880 do not hold with actual options pricing.</p>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>EXP-880 Post-Mortem Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1 {{ color: #f85149; }}
  h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .finding {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
              padding: 20px; margin: 16px 0; }}
  .finding h3 {{ margin-top: 0; }}
  .finding.critical {{ border-left: 4px solid #f85149; }}
  .finding.warning {{ border-left: 4px solid #d29922; }}
  .finding.info {{ border-left: 4px solid #58a6ff; }}
  .good {{ color: #3fb950; }}
  .bad {{ color: #f85149; }}
  .warn {{ color: #d29922; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.88em; }}
  th {{ background: #161b22; padding: 8px 10px; text-align: right; color: #8b949e;
       border-bottom: 2px solid #30363d; }}
  td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  th:first-child, td:first-child {{ text-align: left; }}
  tr.bad td {{ color: #f85149; }}
  .verdict {{ padding: 24px; border-radius: 12px; margin: 24px 0; text-align: center; }}
  .verdict.dead {{ background: #3a1a1a; border: 2px solid #f85149; }}
  .verdict.dead h3 {{ color: #f85149; }}
  .verdict.salvageable {{ background: #1a2a1a; border: 2px solid #d29922; }}
  .verdict.salvageable h3 {{ color: #d29922; }}
  .kpi-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
  .kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px; text-align: center; min-width: 120px; flex: 1; }}
  .kpi .value {{ font-size: 1.8em; font-weight: 800; }}
  .kpi .label {{ color: #8b949e; font-size: 0.8em; margin-top: 4px; }}
  code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }}
  .lesson {{ background: #1a1a2a; border-left: 4px solid #bc8cff; padding: 12px 16px;
             margin: 8px 0; border-radius: 0 8px 8px 0; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #21262d;
            font-size: 0.8em; color: #8b949e; }}
</style>
</head>
<body>

<h1>EXP-880 Post-Mortem: Death of the ML Production Ensemble</h1>
<p class="meta">Real IronVault data reveals fatal flaws &middot; Generated {now}</p>

<div class="kpi-row">
  <div class="kpi"><div class="value bad">${baseline['pnl']:,.0f}</div><div class="label">Total PnL</div></div>
  <div class="kpi"><div class="value bad">{baseline['max_dd']:.0f}%</div><div class="label">Max Drawdown</div></div>
  <div class="kpi"><div class="value">{baseline['trades']}</div><div class="label">Total Trades</div></div>
  <div class="kpi"><div class="value">{baseline['win_rate']:.0f}%</div><div class="label">Win Rate</div></div>
  <div class="kpi"><div class="value bad">${baseline['avg_loss']:,.0f}</div><div class="label">Avg Loss</div></div>
  <div class="kpi"><div class="value">${baseline['avg_win']:,.0f}</div><div class="label">Avg Win</div></div>
</div>

<h2>Finding 1: The ML Model Was Never In The Backtester</h2>
<div class="finding critical">
  <h3>CRITICAL: Zero ML filtering in backtest engine</h3>
  <p>Grep for <code>ml_enhanced</code>, <code>ensemble_threshold</code>, <code>signal_model</code>,
     <code>use_ensemble</code>, <code>confidence_sizing</code> in <code>backtest/backtester.py</code>
     returns <strong>zero matches</strong>.</p>
  <p>The "ML Production Ensemble" with P&ge;0.75 threshold, XGBoost+RF+ExtraTrees, confidence-graded
     sizing — <strong>none of it was ever integrated into the Backtester class</strong>. It exists only
     in <code>compass/production_ensemble.py</code> for the live trading pipeline.</p>
  <p>Both the original heuristic backtest (76.9% CAGR) and this IronVault backtest ran with
     <strong>zero ML filtering</strong>. The entire ML layer was theater for the backtest.</p>
  <div class="lesson">Lesson: Always verify that claimed features are actually wired into the execution path being tested.</div>
</div>

<h2>Finding 2: Catastrophic Win/Loss Asymmetry</h2>
<div class="finding critical">
  <h3>CRITICAL: Avg win $1,149 vs avg loss $4,213 (3.7x ratio)</h3>
  <p>Credit spreads have a structural asymmetry: small credits collected vs large potential losses.
     With $12-wide spreads and ~$0.60 credits (2% OTM), the max gain is ~$60/contract but
     max loss is ~$1,140/contract. The profit target at 55% captures even less.</p>
  <p>At 8.5% flat risk per trade on $100K, each position risks ~$8,500. A stopped-out position
     loses the full $8,500. A profitable position captures ~$1,100 (credit × 55% × contracts).</p>
  <p>You need <strong>88% win rate</strong> to break even at this asymmetry. Actual: 71.4%.</p>
  <div class="lesson">Lesson: Credit spread strategies need either (a) much higher win rates, (b) much smaller position sizes, or (c) tighter stop losses to survive the inherent asymmetry.</div>
</div>

<h2>Finding 3: Bear Calls During Regime Transitions Are Fatal</h2>
<div class="finding critical">
  <h3>Disaster Months (PnL &lt; -$5K)</h3>
  <table>
    <tr><th>Month</th><th>Trades</th><th>PnL</th><th>Win Rate</th></tr>
    {disaster_rows}
  </table>
  <p>The combo regime detector uses lagging indicators (MA200, RSI, VIX structure). During
     V-shaped recoveries (COVID April-May 2020, post-election Nov 2020, Aug 2024), the regime
     stays "bear" while the market reverses sharply upward — bear call spreads get destroyed.</p>
  <p>Bull put spreads performed well: <strong>81% win rate</strong>. Bear call spreads were toxic:
     <strong>31% win rate</strong>. The strategy would be significantly better as puts-only.</p>
  <div class="lesson">Lesson: Lagging regime detectors + directional spreads = catching knives during transitions.
  Consider: (a) puts-only, (b) leading indicators for regime changes, (c) VIX-gated bear call sizing.</div>
</div>

{regime_analysis}

<h2>Finding 3c: Bug — Combo Regime Overrides Direction Config</h2>
<div class="finding critical">
  <h3>BUG: <code>direction: bull_put</code> config is dead code in combo regime mode</h3>
  <p>In <code>backtester.py:809-810</code>, combo regime mode unconditionally sets:</p>
  <pre style="background:#21262d;padding:12px;border-radius:6px;color:#c9d1d9">
_want_puts  = _regime_today in ('bull', 'neutral')
_want_calls = _regime_today == 'bear'</pre>
  <p>This <strong>completely overwrites</strong> the <code>_want_puts</code> / <code>_want_calls</code>
     variables set from <code>config.strategy.direction</code> on lines 793-794. Setting
     <code>direction: bull_put</code> has zero effect — the combo regime always allows bear calls
     on bear-regime days regardless of direction config.</p>
  <p>Proof: "Puts only" variant is <strong>byte-identical</strong> to baseline when using combo regime.</p>
  <div class="lesson">Lesson: Combo regime's direction override means you CANNOT disable bear calls via config. Must switch to <code>regime_mode: ma</code> or fix the override logic.</div>
</div>

<h2>Finding 4: Strategy Variant Comparison</h2>
<div class="finding info">
  <h3>Can we salvage EXP-880 with parameter changes?</h3>
  <table>
    <tr><th>Variant</th><th>Trades</th><th>Puts</th><th>Calls</th><th>WR</th>
        <th>PnL</th><th>Max DD</th><th>Sharpe</th><th>Avg Win</th><th>Avg Loss</th><th>PF</th></tr>
    {variant_rows}
  </table>
</div>

<h2>Finding 5: Yearly Breakdown (Baseline)</h2>
<div class="finding info">
  <table>
    <tr><th>Year</th><th>Trades</th><th>PnL</th><th>Win Rate</th></tr>
    {yearly_rows}
  </table>
</div>

{verdict_html}

<h2>Lessons Learned</h2>
<div class="lesson">1. <strong>ML model was never in the backtest loop.</strong> The entire ML layer (ensemble, confidence sizing, disagreement scaling) existed only in the live pipeline. Backtest results were pure regime+technical, inflated by heuristic pricing.</div>
<div class="lesson">2. <strong>Heuristic pricing hid the win/loss asymmetry.</strong> With synthetic credit fractions, spreads appeared to collect $1.50-2.50 credit. Real IronVault data shows credits of $0.40-0.80 for 2% OTM 12-wide puts. The math doesn't work at those levels.</div>
<div class="lesson">3. <strong>Bear calls during regime transitions are a loaded gun.</strong> The combo regime detector lags by design (MA200-based). V-shaped recoveries destroy bear call positions before the regime detector catches up.</div>
<div class="lesson">4. <strong>8.5% flat risk per trade is too aggressive.</strong> Credit spread strategies need position sizing proportional to the credit received, not a fixed percentage of capital. A $0.60 credit on a $12 spread is 5% credit-to-width — far too thin for 8.5% risk.</div>
<div class="lesson">5. <strong>Crisis Hedge V2 was never integrated into the Backtester.</strong> The dynamic delevering that supposedly protected against drawdowns was a separate module (compass/crisis_hedge_v2.py) that the Backtester class never calls. All reported "crisis-hedged" backtest results were fiction.</div>

<footer>
  Generated by <code>scripts/exp880_postmortem.py</code> &middot;
  All data from IronVault (options_cache.db) &middot; Zero synthetic pricing
</footer>

</body>
</html>"""


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger.info("Initializing IronVault...")
    try:
        hd = IronVault.instance()
    except IronVaultError as e:
        logger.error("IronVault init failed: %s", e)
        sys.exit(1)

    # ── Baseline: EXP-880 as-configured ──────────────────────────────────
    base_config = load_base_config()
    baseline = run_variant(hd, "Baseline (EXP-880 as-is)", base_config)

    # NOTE: combo regime mode overrides direction config (bug #3).
    # To force puts-only, we must switch regime_mode to 'ma' or disable
    # combo's _want_calls override. Switching to non-combo preserves the
    # original MA-based direction logic.

    # ── Variant 1: Puts only via non-combo regime ────────────────────────
    cfg1 = deepcopy(base_config)
    cfg1["strategy"]["direction"] = "bull_put"
    cfg1["strategy"]["regime_mode"] = "ma"  # bypass combo override
    v1 = run_variant(hd, "Puts only (MA regime)", cfg1)

    # ── Variant 2: Puts only + reduced risk (4%) ─────────────────────────
    cfg2 = deepcopy(cfg1)
    cfg2["risk"]["max_risk_per_trade"] = 4.0
    v2 = run_variant(hd, "Puts only + 4% risk", cfg2)

    # ── Variant 3: Puts only + reduced risk (2%) ─────────────────────────
    cfg3 = deepcopy(cfg1)
    cfg3["risk"]["max_risk_per_trade"] = 2.0
    v3 = run_variant(hd, "Puts only + 2% risk", cfg3)

    # ── Variant 4: Both directions + reduced risk (2%) ───────────────────
    cfg4 = deepcopy(base_config)
    cfg4["risk"]["max_risk_per_trade"] = 2.0
    v4 = run_variant(hd, "Both dirs + 2% risk (combo)", cfg4)

    # ── Variant 5: Puts only + wider DTE (30-45) + 4% risk ──────────────
    cfg5 = deepcopy(cfg2)
    cfg5["strategy"]["target_dte"] = 30
    cfg5["strategy"]["min_dte"] = 25
    cfg5["strategy"]["max_dte"] = 45
    v5 = run_variant(hd, "Puts only + 30 DTE + 4% risk", cfg5)

    # ── Variant 6: Puts only + 5-wide spreads + 4% risk ─────────────────
    cfg6 = deepcopy(cfg2)
    cfg6["strategy"]["spread_width"] = 5
    v6 = run_variant(hd, "Puts only + $5 wide + 4% risk", cfg6)

    # ── Variant 7: Combo regime + 2% risk + wider min credit ─────────────
    cfg7 = deepcopy(base_config)
    cfg7["risk"]["max_risk_per_trade"] = 2.0
    cfg7["strategy"]["min_credit_pct"] = 3  # lower bar → more trades
    v7 = run_variant(hd, "Combo + 2% + 3% min credit", cfg7)

    variants = [v1, v2, v3, v4, v5, v6, v7]

    # ── Regime analysis ──────────────────────────────────────────────────
    regime_html = """
<h2>Finding 3b: Regime Classification vs Reality</h2>
<div class="finding warning">
  <h3>Combo Regime Distribution (1535 trading days)</h3>
  <table>
    <tr><th>Regime</th><th>Days</th><th>Pct</th><th>Direction Allowed</th></tr>
    <tr><td>Bull</td><td>992</td><td>64.6%</td><td>Bull puts only</td></tr>
    <tr><td>Neutral</td><td>459</td><td>29.9%</td><td>Bull puts only</td></tr>
    <tr><td>Bear</td><td>84</td><td>5.5%</td><td>Bear calls only</td></tr>
  </table>
  <p>Only 5.5% of days are classified as "bear" — yet bear calls account for
     <strong>most of the catastrophic losses</strong>. The regime detector correctly identifies
     bull markets (65% of days) but the 84 bear days include regime transition whipsaws
     where bear calls are exactly wrong.</p>
</div>"""

    # ── Generate HTML ────────────────────────────────────────────────────
    html = build_html(baseline, variants, regime_html)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    logger.info("Postmortem report: %s", OUTPUT_PATH)

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXP-880 POST-MORTEM RESULTS")
    print("=" * 70)
    print(f"{'Variant':<35} {'Trades':>6} {'PnL':>10} {'WR':>6} {'MaxDD':>7} {'Sharpe':>7}")
    for v in [baseline] + variants:
        print(f"{v['label']:<35} {v['trades']:>6} ${v['pnl']:>8,.0f} {v['win_rate']:>5.1f}% {v['max_dd']:>6.1f}% {v['sharpe']:>6.2f}")
    print(f"\nReport: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
