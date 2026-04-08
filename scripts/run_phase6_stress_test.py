#!/usr/bin/env python3
"""
Phase 6 Stress Testing — MASTERPLAN

Runs the full StressTester suite on the best portfolio configuration
(EXP-400 + EXP-401 blend, ~81/19 weights from Phase 5 optimization).

Executes:
  1. Monte Carlo 10K paths with block-bootstrap
  2. All 4 crisis scenarios (COVID, 2022 bear, flash crash, VIX spike)
  3. Sensitivity analysis (risk_pct, spread_width, stop_loss_mult)

Thresholds:
  - 5th-percentile MC DD <= 30%
  - All crisis scenarios DD <= 40%

Output: reports/stress_test_full.html
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.stress_test import StressTester

STARTING_CAPITAL = 100_000
# Phase 5 winning blend (Max Sharpe optimizer converged to ~81/19)
PORTFOLIO_WEIGHTS = {"EXP-400": 0.813, "EXP-401": 0.187}

# Thresholds from MASTERPLAN Phase 6 spec
MC_DD_5PCT_THRESHOLD = 30.0      # 5th pctile MC DD <= 30%
CRISIS_DD_THRESHOLD = 40.0       # all crisis scenarios DD <= 40%


def load_exp400_daily_pnl() -> dict:
    """Build daily PnL dict from EXP-400 trade log (560 real trades)."""
    trades = json.load(open(ROOT / "output" / "champion_trade_log.json"))
    daily_pnl = defaultdict(float)
    for t in trades:
        # Attribute the trade PnL to the exit date (realized)
        exit_date = t["exit"][:10]
        net_pnl = t["pnl"] - t.get("comm", 0)
        daily_pnl[exit_date] += net_pnl
    return dict(daily_pnl)


def build_trading_calendar(start: str, end: str) -> list:
    """Build a trading-day calendar (excludes weekends)."""
    days = []
    cur = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while cur <= end_dt:
        if cur.weekday() < 5:  # Mon-Fri
            days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


def build_exp401_daily_returns(calendar: list) -> np.ndarray:
    """Synthesize EXP-401 daily returns from yearly stats.

    EXP-401 yearly stats (from exp401_robust_score.json) are used to scale
    a random-walk return series that matches the observed annual return
    and volatility per year. Uses deterministic seed for reproducibility.
    """
    yearly = json.load(open(ROOT / "output" / "exp401_robust_score.json"))["baseline_yearly"]

    returns = np.zeros(len(calendar))
    rng = np.random.RandomState(401)

    for i, day in enumerate(calendar):
        year = day[:4]
        if year not in yearly:
            continue
        y = yearly[year]
        annual_ret = y["return_pct"] / 100
        annual_sharpe = y["sharpe_ratio"]
        # Imply annual vol from sharpe: sharpe = ret / vol → vol = ret / sharpe
        if annual_sharpe > 0.1:
            annual_vol = annual_ret / annual_sharpe
        else:
            annual_vol = 0.15  # fallback
        daily_vol = annual_vol / np.sqrt(252)
        daily_mean = annual_ret / 252
        # Generate a bar with the right statistical properties
        returns[i] = rng.normal(daily_mean, abs(daily_vol))

    return returns


def build_portfolio_daily_returns():
    """Build the blended EXP-400 + EXP-401 daily return series."""
    # Calendar: 2020-01-02 to 2025-12-31
    calendar = build_trading_calendar("2020-01-02", "2025-12-31")

    # EXP-400: real trade PnL mapped to exit dates
    exp400_pnl = load_exp400_daily_pnl()
    exp400_returns = np.array([
        exp400_pnl.get(day, 0.0) / STARTING_CAPITAL
        for day in calendar
    ])

    # EXP-401: synthesized from yearly stats (vol-matched)
    exp401_returns = build_exp401_daily_returns(calendar)

    # Blended portfolio returns (weighted)
    blended = (
        PORTFOLIO_WEIGHTS["EXP-400"] * exp400_returns +
        PORTFOLIO_WEIGHTS["EXP-401"] * exp401_returns
    )

    return calendar, exp400_returns, exp401_returns, blended


def check_thresholds(results: dict) -> dict:
    """Check results against Phase 6 thresholds."""
    mc = results["monte_carlo"]
    crisis = results["crisis_scenarios"]

    mc_p5_dd = abs(mc["max_drawdown"]["percentiles_pct"].get("p5", 0))
    mc_pass = mc_p5_dd <= MC_DD_5PCT_THRESHOLD

    crisis_pass = True
    crisis_details = []
    for c in crisis:
        # portfolio_drawdown_pct already includes 1.5x credit spread beta
        dd_pct = abs(c.get("portfolio_drawdown_pct", 0))
        passes = dd_pct <= CRISIS_DD_THRESHOLD
        if not passes:
            crisis_pass = False
        crisis_details.append({
            "name": c.get("name", "?"),
            "dd_pct": round(dd_pct, 2),
            "underlying_dd_pct": round(abs(c.get("underlying_drawdown_pct", 0)), 2),
            "passes": passes,
        })

    return {
        "mc_p5_dd_pct": round(mc_p5_dd, 2),
        "mc_threshold_pct": MC_DD_5PCT_THRESHOLD,
        "mc_pass": mc_pass,
        "crisis_pass": crisis_pass,
        "crisis_threshold_pct": CRISIS_DD_THRESHOLD,
        "crisis_details": crisis_details,
        "overall_pass": mc_pass and crisis_pass,
    }


def generate_html(results: dict, threshold_check: dict, meta: dict, output_path: Path):
    """Generate comprehensive stress test HTML report."""
    mc = results["monte_carlo"]
    crisis = results["crisis_scenarios"]
    sens = results["sensitivity"]

    # ── Header & verdict ──
    verdict_color = "#059669" if threshold_check["overall_pass"] else "#dc2626"
    verdict_text = "PASS" if threshold_check["overall_pass"] else "FAIL"

    # ── Monte Carlo percentiles table ──
    dd_pcts = mc["max_drawdown"]["percentiles_pct"]
    tw_pcts = mc["terminal_wealth"]["percentiles"]

    mc_rows = ""
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        key = f"p{p}"
        dd_val = dd_pcts.get(key, 0)
        tw_val = tw_pcts.get(key, 0)
        # Highlight P5 (threshold check)
        hl = ' style="background:#fef3c7"' if p == 5 else ""
        mc_rows += (
            f'<tr{hl}>'
            f'<td><strong>P{p}</strong></td>'
            f'<td class="r">${tw_val:,.0f}</td>'
            f'<td class="r">{(tw_val / STARTING_CAPITAL - 1) * 100:+.1f}%</td>'
            f'<td class="r" style="color:#dc2626">{dd_val:.2f}%</td>'
            f'</tr>\n'
        )

    # ── Crisis scenarios table ──
    crisis_rows = ""
    for c in crisis:
        name = c.get("name", "?")
        desc = c.get("description", "")
        n_days = c.get("n_days", 0)
        underlying_dd = abs(c.get("underlying_drawdown_pct", 0))
        port_dd = abs(c.get("portfolio_drawdown_pct", 0))
        trough = c.get("trough_value", 0)
        recovery = c.get("estimated_recovery_days")
        vix_mult = c.get("vix_multiplier", 1.0)

        passes = port_dd <= CRISIS_DD_THRESHOLD
        status_color = "#059669" if passes else "#dc2626"
        status_icon = "&#10003;" if passes else "&#10007;"

        crisis_rows += (
            f'<tr>'
            f'<td style="text-align:left"><strong>{name}</strong>'
            f'<br/><span style="color:#64748b;font-size:.76rem">{desc}</span></td>'
            f'<td class="r">{n_days}</td>'
            f'<td class="r">{vix_mult:.1f}x</td>'
            f'<td class="r" style="color:#dc2626">{underlying_dd:.2f}%</td>'
            f'<td class="r" style="color:#dc2626;font-weight:700">{port_dd:.2f}%</td>'
            f'<td class="r">${trough:,.0f}</td>'
            f'<td class="r">{recovery if recovery else "&mdash;"} days</td>'
            f'<td class="r" style="color:{status_color};font-weight:700">{status_icon} {"PASS" if passes else "FAIL"}</td>'
            f'</tr>\n'
        )

    # ── Sensitivity tables ──
    sens_tables = ""
    # Only show the 3 params requested (risk_pct = position_size_pct, stop_loss_mult, spread_width)
    param_order = ["position_size_pct", "stop_loss_multiplier", "spread_width"]
    for pname in param_order:
        if pname not in sens:
            continue
        pdata = sens[pname]
        rows = ""
        for r in pdata["results"]:
            hl = ' style="background:#f0fdf4"' if r["is_baseline"] else ""
            sharpe_c = "#059669" if r["sharpe"] > 0 else "#dc2626"
            cagr_c = "#059669" if r["cagr_pct"] > 0 else "#dc2626"
            rows += (
                f'<tr{hl}>'
                f'<td><strong>{r["value"]:.2f}</strong>{" ★" if r["is_baseline"] else ""}</td>'
                f'<td class="r" style="color:{sharpe_c}">{r["sharpe"]:.2f}</td>'
                f'<td class="r" style="color:{cagr_c}">{r["cagr_pct"]:+.1f}%</td>'
                f'<td class="r" style="color:#dc2626">{r["max_dd_pct"]:.2f}%</td>'
                f'<td class="r">{r["calmar"]:.2f}</td>'
                f'<td class="r">${r["terminal_value"]:,.0f}</td>'
                f'</tr>\n'
            )
        sens_tables += (
            f'<h3>{pdata["label"]}</h3>'
            f'<p class="note">{pdata["description"]} &bull; Baseline: {pdata["baseline"]}</p>'
            f'<table>'
            f'<thead><tr><th>Value</th><th class="r">Sharpe</th><th class="r">CAGR</th>'
            f'<th class="r">Max DD</th><th class="r">Calmar</th><th class="r">Terminal</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
        )

    # ── Build HTML ──
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Phase 6 Stress Test — EXP-400/401 Blend</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e2e8f0;--text:#1a1a2e;--muted:#64748b;--green:#059669;--red:#dc2626;--blue:#2563eb}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.55;max-width:1200px;margin:0 auto;padding:28px}}
h1{{font-size:1.55rem;font-weight:800;margin-bottom:4px}}
h2{{font-size:1.15rem;font-weight:700;margin:32px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--border)}}
h3{{font-size:.98rem;font-weight:600;margin:22px 0 8px;color:#374151}}
.sub{{color:var(--muted);font-size:.86rem;margin-bottom:18px}}
.note{{color:var(--muted);font-size:.82rem;font-style:italic;margin:6px 0}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.84rem}}
th{{background:#f1f5f9;color:var(--muted);padding:7px 10px;text-align:left;border-bottom:2px solid var(--border);font-size:.74rem;font-weight:600;text-transform:uppercase}}
td{{padding:6px 10px;border-bottom:1px solid #f1f5f9;text-align:left}}
.r{{text-align:right}}
tr:hover td{{background:#fafafa}}
.hero{{background:linear-gradient(135deg,#f1f5f9,#e2e8f0);border:2px solid {verdict_color};border-radius:12px;padding:24px;text-align:center;margin:18px 0}}
.hero .big{{font-size:1.8rem;font-weight:800;color:{verdict_color}}}
.hero p{{color:var(--muted);font-size:.9rem;margin-top:6px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:16px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:13px;text-align:center}}
.c .l{{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.3px}}
.c .v{{font-weight:700;font-size:1.18rem;margin-top:3px}}
.box{{border:1px solid var(--border);border-radius:8px;padding:16px;margin:12px 0;background:var(--card)}}
.box-green{{border-left:5px solid var(--green)}} .box-red{{border-left:5px solid var(--red)}}
.box-blue{{border-left:5px solid var(--blue)}}
.box h4{{margin:0 0 6px;font-size:.95rem}}
</style></head><body>

<h1>Phase 6 Stress Test — EXP-400/401 Blend</h1>
<p class="sub">Full stress suite on best portfolio configuration from Phase 5 optimization
&bull; {meta['n_returns']} daily returns &bull; {mc['n_simulations']:,} MC paths &bull;
{datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

<div class="hero">
<div class="big">{verdict_text}</div>
<p>Monte Carlo P5 DD: <strong>{threshold_check['mc_p5_dd_pct']:.2f}%</strong> (threshold: ≤{MC_DD_5PCT_THRESHOLD}%)
&bull; Crisis scenarios: {"<strong>all 4 passing</strong>" if threshold_check['crisis_pass'] else "<strong>FAILING</strong>"}
(threshold: ≤{CRISIS_DD_THRESHOLD}%)</p>
</div>

<div class="cards">
<div class="c"><div class="l">Portfolio</div><div class="v" style="font-size:.85rem">EXP-400 {PORTFOLIO_WEIGHTS["EXP-400"]:.0%}<br/>EXP-401 {PORTFOLIO_WEIGHTS["EXP-401"]:.0%}</div></div>
<div class="c"><div class="l">Starting Capital</div><div class="v">${STARTING_CAPITAL:,}</div></div>
<div class="c"><div class="l">Daily Returns</div><div class="v">{meta['n_returns']:,}</div></div>
<div class="c"><div class="l">MC Simulations</div><div class="v">{mc['n_simulations']:,}</div></div>
<div class="c"><div class="l">Block Size</div><div class="v">{mc['block_size']} days</div></div>
<div class="c"><div class="l">Prob. Profit</div><div class="v" style="color:var(--green)">{mc['prob_profit']:.0%}</div></div>
<div class="c"><div class="l">Prob. Ruin (50%)</div><div class="v" style="color:var(--red)">{mc['prob_ruin_50pct']:.1%}</div></div>
</div>

<!-- ══ Monte Carlo ══ -->
<h2>1. Monte Carlo Simulation ({mc['n_simulations']:,} paths, block-bootstrap)</h2>
<p class="note">Block bootstrap preserves volatility clustering and mean reversion.
Block size: {mc['block_size']} trading days. Horizon: {mc['horizon_days']} days.</p>

<table>
<thead><tr><th>Percentile</th><th class="r">Terminal Wealth</th><th class="r">Total Return</th><th class="r">Max Drawdown</th></tr></thead>
<tbody>{mc_rows}</tbody>
</table>

<div class="box box-{'green' if threshold_check['mc_pass'] else 'red'}">
<h4>Monte Carlo Verdict</h4>
<p>P5 max drawdown: <strong>{threshold_check['mc_p5_dd_pct']:.2f}%</strong>
(only 5% of simulated paths exceed this drawdown).
Threshold: ≤{MC_DD_5PCT_THRESHOLD}%. <strong>{"PASS" if threshold_check['mc_pass'] else "FAIL"}</strong>.</p>
<p style="margin-top:6px;font-size:.85rem">
Median terminal: ${mc['terminal_wealth']['median']:,.0f}
&bull; Mean: ${mc['terminal_wealth']['mean']:,.0f}
&bull; Worst: ${mc['terminal_wealth']['min']:,.0f}
&bull; Best: ${mc['terminal_wealth']['max']:,.0f}
</p>
</div>

<!-- ══ Crisis Scenarios ══ -->
<h2>2. Crisis Scenario Analysis (4 historical stress events)</h2>
<p class="note">Each scenario overlays historical crisis daily shocks onto the portfolio,
with a 1.5x beta adjustment for credit spreads (short gamma amplification).</p>

<table>
<thead><tr><th>Scenario</th><th class="r">Days</th><th class="r">VIX Mult</th><th class="r">Underlying DD</th><th class="r">Portfolio DD (1.5x β)</th><th class="r">Trough $</th><th class="r">Recovery</th><th class="r">Status</th></tr></thead>
<tbody>{crisis_rows}</tbody>
</table>

<div class="box box-{'green' if threshold_check['crisis_pass'] else 'red'}">
<h4>Crisis Verdict</h4>
<p>All crisis DDs must be ≤{CRISIS_DD_THRESHOLD}% (with 1.5x credit spread beta).
<strong>{"All 4 scenarios pass" if threshold_check['crisis_pass'] else "One or more scenarios fail"}.</strong></p>
</div>

<!-- ══ Sensitivity Analysis ══ -->
<h2>3. Sensitivity Analysis</h2>
<p class="note">Parameter sweeps measuring Sharpe/drawdown impact. Uses built-in heuristic
scaling model. Baseline row highlighted in green.</p>

{sens_tables}

<!-- ══ Summary ══ -->
<h2>4. Summary</h2>
<div class="box box-blue">
<h4>Phase 6 Verdict: {verdict_text}</h4>
<ul style="padding-left:18px;font-size:.88rem;line-height:1.85;margin-top:6px">
<li><strong>Monte Carlo:</strong> P5 drawdown {threshold_check['mc_p5_dd_pct']:.2f}%
({"within" if threshold_check['mc_pass'] else "exceeds"} 30% threshold).
Median Sharpe {mc['sharpe_ratio']['median']:.2f}. Probability of profit {mc['prob_profit']:.0%}.
Probability of 50% ruin: {mc['prob_ruin_50pct']:.1%}.</li>
<li><strong>Crisis scenarios:</strong>
{sum(1 for d in threshold_check['crisis_details'] if d['passes'])}/{len(threshold_check['crisis_details'])}
passing. Worst case: {max(threshold_check['crisis_details'], key=lambda x: x['dd_pct'])['name']}
at {max(threshold_check['crisis_details'], key=lambda x: x['dd_pct'])['dd_pct']:.2f}% DD.</li>
<li><strong>Sensitivity:</strong> Swept position size (1-15%), stop-loss multiplier (1.5-5.0),
and spread width ($2.5-$20). Results show how parameter changes affect risk/return profile.</li>
</ul>
</div>

<p style="text-align:center;color:var(--muted);margin-top:36px;padding-top:14px;border-top:1px solid var(--border);font-size:.78rem">
Phase 6 Stress Test &bull; compass/stress_test.py &bull;
Portfolio: EXP-400 {PORTFOLIO_WEIGHTS["EXP-400"]:.0%} + EXP-401 {PORTFOLIO_WEIGHTS["EXP-401"]:.0%}
&bull; {datetime.now().strftime("%Y-%m-%d")}
</p>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main():
    print("=" * 70)
    print("Phase 6 Stress Testing — EXP-400/401 Blend")
    print("=" * 70)

    # Build blended portfolio daily returns
    print("\nBuilding blended portfolio returns...")
    calendar, exp400_ret, exp401_ret, blended = build_portfolio_daily_returns()
    print(f"  Calendar: {len(calendar)} trading days ({calendar[0]} to {calendar[-1]})")
    print(f"  EXP-400: sum={exp400_ret.sum():.4f} mean={exp400_ret.mean():.6f} std={exp400_ret.std():.4f}")
    print(f"  EXP-401: sum={exp401_ret.sum():.4f} mean={exp401_ret.mean():.6f} std={exp401_ret.std():.4f}")
    print(f"  Blended: sum={blended.sum():.4f} mean={blended.mean():.6f} std={blended.std():.4f}")

    # Total return check
    total_return = np.prod(1 + blended) - 1
    years = len(blended) / 252
    cagr = (1 + total_return) ** (1 / years) - 1
    print(f"  Blended total return: {total_return * 100:+.2f}% ({cagr * 100:+.2f}% CAGR over {years:.1f} years)")

    # Run stress tester
    print(f"\nRunning StressTester (10,000 MC paths, block=5)...")
    tester = StressTester(
        daily_returns=blended,
        starting_capital=STARTING_CAPITAL,
        n_simulations=10_000,
        block_size=5,
        seed=42,
    )

    results = tester.run_all()

    # Print summary
    mc = results["monte_carlo"]
    print(f"\n--- Monte Carlo Results ---")
    print(f"  Median terminal: ${mc['terminal_wealth']['median']:,.0f}")
    print(f"  Mean terminal:   ${mc['terminal_wealth']['mean']:,.0f}")
    print(f"  P5 drawdown:     {mc['max_drawdown']['percentiles_pct']['p5']:.2f}%")
    print(f"  P1 drawdown:     {mc['max_drawdown']['percentiles_pct']['p1']:.2f}%")
    print(f"  Worst drawdown:  {mc['max_drawdown']['worst_pct']:.2f}%")
    print(f"  Median Sharpe:   {mc['sharpe_ratio']['median']:.2f}")
    print(f"  Prob profit:     {mc['prob_profit']:.1%}")
    print(f"  Prob ruin(50%):  {mc['prob_ruin_50pct']:.3%}")

    print(f"\n--- Crisis Scenarios ---")
    for c in results["crisis_scenarios"]:
        name = c.get("name", "?")
        under_dd = c.get("underlying_drawdown_pct", 0)
        port_dd = c.get("portfolio_drawdown_pct", 0)
        trough = c.get("trough_value", 0)
        print(f"  {name}: underlying {under_dd:+.2f}%, portfolio {port_dd:+.2f}% (1.5x β), trough ${trough:,.0f}")

    print(f"\n--- Sensitivity (baselines) ---")
    for pname in ["position_size_pct", "stop_loss_multiplier", "spread_width"]:
        if pname in results["sensitivity"]:
            pdata = results["sensitivity"][pname]
            baseline = next((r for r in pdata["results"] if r["is_baseline"]), None)
            if baseline:
                print(f"  {pdata['label']} = {baseline['value']}: "
                      f"Sharpe {baseline['sharpe']:.2f}, DD {baseline['max_dd_pct']:.2f}%, "
                      f"CAGR {baseline['cagr_pct']:+.1f}%")

    # Threshold check
    print(f"\n--- Threshold Checks ---")
    checks = check_thresholds(results)
    print(f"  MC P5 DD <= {MC_DD_5PCT_THRESHOLD}%: {checks['mc_p5_dd_pct']:.2f}% - "
          f"{'PASS' if checks['mc_pass'] else 'FAIL'}")
    print(f"  Crisis scenarios DD <= {CRISIS_DD_THRESHOLD}%:")
    for d in checks["crisis_details"]:
        status = "PASS" if d["passes"] else "FAIL"
        print(f"    {d['name']}: {d['dd_pct']:.2f}% - {status}")
    print(f"\n  OVERALL: {'PASS' if checks['overall_pass'] else 'FAIL'}")

    # Generate HTML
    print("\nGenerating HTML report...")
    output_path = ROOT / "reports" / "stress_test_full.html"
    meta = {
        "n_returns": len(blended),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "calendar_start": calendar[0],
        "calendar_end": calendar[-1],
    }
    generate_html(results, checks, meta, output_path)
    print(f"  Report: {output_path}")

    # Save JSON
    import json as _json
    json_path = ROOT / "reports" / "stress_test_full.json"
    json_summary = {
        "generated": datetime.now().isoformat(),
        "portfolio_weights": PORTFOLIO_WEIGHTS,
        "meta": meta,
        "threshold_check": checks,
        "monte_carlo": {
            k: v for k, v in results["monte_carlo"].items()
            if k != "sample_paths"  # too large
        },
        "crisis_scenarios": [
            {k: v for k, v in c.items() if k != "equity_path"}
            for c in results["crisis_scenarios"]
        ],
        "sensitivity": results["sensitivity"],
    }

    def _default(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    json_path.write_text(_json.dumps(json_summary, indent=2, default=_default))
    print(f"  JSON:   {json_path}")

    print(f"\n{'=' * 70}")
    print(f"Phase 6 Complete: {'PASS' if checks['overall_pass'] else 'FAIL'}")
    print(f"{'=' * 70}")

    return checks["overall_pass"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
