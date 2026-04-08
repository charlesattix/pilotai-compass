#!/usr/bin/env python3
"""
Ultimate Portfolio Extreme Stress Test
========================================
Tests the 4-strategy portfolio at 1.6x leverage under synthetic scenarios
WORSE than any historical event. Finds where the portfolio breaks.

Scenarios:
  1. COVID x2: -68% over 46 days (2x real COVID)
  2. Prolonged Bear: -50% over 400 days (2x 2022)
  3. Flash Crash Cascade: -15% in 1 day, -5% day 2, -3% day 3
  4. Stagflation: high vol + slow decline for 252 days (12 months)
  5. Monte Carlo: 10K random paths with fat tails

Strategy-level attribution shows which strategies protect/fail.
"""

import math, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CAPITAL = 100_000
LEVERAGE = 1.6
TRADING_DAYS = 252

# ═══════════════════════════════════════════════════════════════════════════
# Portfolio definition: Ultimate Portfolio at 1.6x
# ═══════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "EXP-1220 Tail Risk": {
        "weight": 0.90,
        "base_vol": 0.10,        # annualized
        "spy_beta": 0.45,
        "vix_sensitivity": -0.8, # reduces exposure as VIX rises
        "crisis_alpha": 0.15,    # earns alpha DURING crises (hedge payoff)
        "normal_alpha": 0.35,    # annual alpha in normal times
        "dd_multiplier": 0.4,    # drawdown attenuated by hedge
    },
    "Cross-Asset Pairs": {
        "weight": 0.033,
        "base_vol": 0.04,
        "spy_beta": 0.02,
        "vix_sensitivity": 0.0,  # market-neutral
        "crisis_alpha": 0.01,
        "normal_alpha": 0.009,
        "dd_multiplier": 0.3,
    },
    "Vol Term Structure": {
        "weight": 0.033,
        "base_vol": 0.03,
        "spy_beta": -0.15,       # counter-cyclical
        "vix_sensitivity": 0.3,  # benefits from vol expansion
        "crisis_alpha": 0.02,    # contango/backwardation signals fire
        "normal_alpha": 0.005,
        "dd_multiplier": 0.5,
    },
    "TLT Iron Condors": {
        "weight": 0.033,
        "base_vol": 0.05,
        "spy_beta": -0.20,       # treasuries hedge equity
        "vix_sensitivity": -0.3, # vol hurts IC
        "crisis_alpha": -0.02,   # ICs lose in crisis
        "normal_alpha": 0.10,
        "dd_multiplier": 1.5,    # ICs amplify DD in crisis
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario generators
# ═══════════════════════════════════════════════════════════════════════════

def gen_covid_x2(n_days=252, crash_days=46, crash_pct=-0.68, recovery_days=120, seed=42):
    """COVID x2: -68% crash in 46 days, then partial recovery."""
    rng = np.random.RandomState(seed)
    spy = np.zeros(n_days)
    vix = np.zeros(n_days)

    # Pre-crash: 20 calm days
    pre = 20
    spy[:pre] = rng.normal(0.0004, 0.008, pre)
    vix[:pre] = 18 + rng.normal(0, 1, pre)

    # Crash phase
    daily_crash = (1 + crash_pct) ** (1 / crash_days) - 1
    for i in range(pre, pre + crash_days):
        spy[i] = daily_crash + rng.normal(0, 0.015)  # high vol noise
        vix[i] = min(80, 18 + (i - pre) / crash_days * 65 + rng.normal(0, 3))

    # Recovery phase
    start_rec = pre + crash_days
    recovery_rate = abs(crash_pct) * 0.6 / recovery_days  # recover 60%
    for i in range(start_rec, min(start_rec + recovery_days, n_days)):
        spy[i] = recovery_rate + rng.normal(0, 0.012)
        vix[i] = max(15, 80 - (i - start_rec) / recovery_days * 55 + rng.normal(0, 2))

    # Post-recovery: normal
    post_start = min(start_rec + recovery_days, n_days)
    for i in range(post_start, n_days):
        spy[i] = rng.normal(0.0003, 0.009)
        vix[i] = max(12, 20 + rng.normal(0, 2))

    return spy, vix


def gen_prolonged_bear(n_days=500, total_decline=-0.50, seed=43):
    """Prolonged bear: -50% over 400 days, then flat."""
    rng = np.random.RandomState(seed)
    spy = np.zeros(n_days)
    vix = np.zeros(n_days)

    bear_days = 400
    daily_decline = (1 + total_decline) ** (1 / bear_days) - 1
    for i in range(bear_days):
        spy[i] = daily_decline + rng.normal(0, 0.010)
        # VIX elevated but not extreme
        vix[i] = 28 + 7 * math.sin(i / 40) + rng.normal(0, 2)

    for i in range(bear_days, n_days):
        spy[i] = rng.normal(0.0001, 0.008)
        vix[i] = max(15, 22 + rng.normal(0, 2))

    return spy, vix


def gen_flash_cascade(n_days=60, seed=44):
    """Flash crash cascade: -15% day 1, -5% day 2, -3% day 3, then recovery."""
    rng = np.random.RandomState(seed)
    spy = np.zeros(n_days)
    vix = np.zeros(n_days)

    spy[0] = -0.15
    vix[0] = 75
    spy[1] = -0.05
    vix[1] = 65
    spy[2] = -0.03
    vix[2] = 55

    # Dead cat bounce
    spy[3] = 0.04
    vix[3] = 48
    spy[4] = 0.02
    vix[4] = 42

    # Aftershock
    spy[5] = -0.04
    vix[5] = 50

    # Gradual normalization
    for i in range(6, n_days):
        spy[i] = rng.normal(0.002, 0.015)
        vix[i] = max(15, 50 - (i - 6) * 0.7 + rng.normal(0, 2))

    return spy, vix


def gen_stagflation(n_days=252, seed=45):
    """Stagflation: high vol + slow decline for 12 months."""
    rng = np.random.RandomState(seed)
    spy = np.zeros(n_days)
    vix = np.zeros(n_days)

    for i in range(n_days):
        # Slow grind down with high volatility
        spy[i] = rng.normal(-0.0006, 0.015)  # ~-15% annual, 24% vol
        # VIX stays 25-35 (persistently elevated)
        vix[i] = 30 + 5 * math.sin(i / 30) + rng.normal(0, 2)

    return spy, vix


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio simulation under stress
# ═══════════════════════════════════════════════════════════════════════════

def simulate_under_stress(spy_returns, vix_levels, label=""):
    """Simulate all 4 strategies under a stress scenario.

    Each strategy's daily return depends on:
      r_i = alpha_i + beta_i * spy_r + vix_adj + noise

    EXP-1220 tail risk hedge is special: it EARNS during crashes via
    dynamic delevering and hedge payoff. Its DD is attenuated.
    """
    n_days = len(spy_returns)
    rng = np.random.RandomState(hash(label) % 2**31)

    strategy_returns = {}
    for name, spec in STRATEGIES.items():
        daily = np.zeros(n_days)
        for i in range(n_days):
            spy_r = spy_returns[i]
            vix_val = vix_levels[i]

            # Base return from alpha
            is_crisis = vix_val > 30
            alpha = spec["crisis_alpha"] / TRADING_DAYS if is_crisis else spec["normal_alpha"] / TRADING_DAYS

            # Beta exposure (attenuated during crisis for hedged strategies)
            beta = spec["spy_beta"]
            if is_crisis and spec["vix_sensitivity"] < 0:
                # Hedge kicks in: reduce beta, flip to hedge payoff
                crisis_intensity = min((vix_val - 30) / 30, 1.0)
                beta *= (1 - crisis_intensity * 0.8)
                alpha += crisis_intensity * abs(spy_r) * 0.3  # hedge profit

            # VIX adjustment
            vix_adj = spec["vix_sensitivity"] * max(0, vix_val - 20) / 10000

            # DD multiplier: how much of SPY DD passes through
            if spy_r < 0:
                market_component = beta * spy_r * spec["dd_multiplier"]
            else:
                market_component = beta * spy_r

            noise = rng.normal(0, spec["base_vol"] / math.sqrt(TRADING_DAYS) * 0.3)
            daily[i] = alpha + market_component + vix_adj + noise

        strategy_returns[name] = daily

    # Combine with weights and leverage
    combined = np.zeros(n_days)
    for name, spec in STRATEGIES.items():
        combined += strategy_returns[name] * spec["weight"]
    combined *= LEVERAGE

    # Portfolio metrics
    cum = np.cumprod(1 + combined)
    peak = np.maximum.accumulate(cum)
    dd_curve = (cum - peak) / peak
    max_dd = dd_curve.min()

    # Recovery time
    recovery_days = 0
    if max_dd < -0.01:
        dd_idx = np.argmin(dd_curve)
        for j in range(dd_idx, n_days):
            if cum[j] >= peak[dd_idx]:
                recovery_days = j - dd_idx
                break
        else:
            recovery_days = n_days - dd_idx  # didn't recover

    # Per-strategy attribution
    attribution = {}
    for name, spec in STRATEGIES.items():
        s_cum = np.cumprod(1 + strategy_returns[name] * spec["weight"] * LEVERAGE)
        s_peak = np.maximum.accumulate(s_cum)
        s_dd = ((s_cum - s_peak) / s_peak).min()
        s_total = s_cum[-1] - 1
        attribution[name] = {
            "total_return": float(s_total),
            "max_dd": float(s_dd),
            "weight": spec["weight"],
            "contributed_return": float(s_total * spec["weight"]),
        }

    return {
        "label": label,
        "n_days": n_days,
        "portfolio_return": float(cum[-1] - 1),
        "max_dd": float(max_dd),
        "recovery_days": recovery_days,
        "final_value": float(cum[-1] * CAPITAL),
        "survived": max_dd > -0.50,  # -50% = effective wipeout
        "attribution": attribution,
        "dd_curve": dd_curve.tolist(),
        "cum_curve": cum.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo stress
# ═══════════════════════════════════════════════════════════════════════════

def monte_carlo_stress(n_paths=10000, n_days=252, seed=100):
    """10K random paths with fat tails (t-distribution, df=4)."""
    rng = np.random.RandomState(seed)
    results = {
        "max_dds": [], "returns": [], "recovery_days": [],
        "survived": 0, "ruin": 0,
    }

    for p in range(n_paths):
        # Fat-tail SPY returns (Student-t with df=4 for heavy tails)
        spy = rng.standard_t(df=4, size=n_days) * 0.012  # ~19% annual vol
        # VIX correlated with negative spy returns
        vix = 20 - spy * 500 + rng.normal(0, 3, n_days)
        vix = np.clip(vix, 10, 80)

        r = simulate_under_stress(spy, vix, label=f"MC_{p}")
        results["max_dds"].append(r["max_dd"])
        results["returns"].append(r["portfolio_return"])
        results["recovery_days"].append(r["recovery_days"])
        if r["survived"]:
            results["survived"] += 1
        if r["max_dd"] < -0.90:
            results["ruin"] += 1

    dds = np.array(results["max_dds"])
    rets = np.array(results["returns"])

    return {
        "n_paths": n_paths,
        "median_dd": float(np.median(dds)),
        "p5_dd": float(np.percentile(dds, 5)),
        "p1_dd": float(np.percentile(dds, 1)),
        "worst_dd": float(dds.min()),
        "median_return": float(np.median(rets)),
        "p5_return": float(np.percentile(rets, 5)),
        "survival_rate": results["survived"] / n_paths,
        "ruin_rate": results["ruin"] / n_paths,
        "prob_positive": float((rets > 0).sum() / n_paths),
        "mean_recovery": float(np.mean(results["recovery_days"])),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def pct(v, d=1):
    return f"{v*100:+.{d}f}%"

def clr(v):
    return "#16a34a" if v >= 0 else "#dc2626"


def build_html(scenarios, mc):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Scenario summary table
    scenario_rows = ""
    for s in scenarios:
        surv = '<span style="color:#16a34a;font-weight:700">SURVIVED</span>' if s["survived"] else '<span style="color:#dc2626;font-weight:700">FAILED</span>'
        bg = "" if s["survived"] else "background:#fef2f2;"
        scenario_rows += f"""<tr style="{bg}">
            <td style="text-align:left;font-weight:600">{s['label']}</td>
            <td style="color:{clr(s['portfolio_return'])};font-weight:600">{pct(s['portfolio_return'])}</td>
            <td style="color:#dc2626;font-weight:600">{pct(s['max_dd'])}</td>
            <td>{s['recovery_days']}d</td>
            <td>${s['final_value']:,.0f}</td>
            <td>{surv}</td>
        </tr>"""

    # Attribution tables per scenario
    attr_html = ""
    for s in scenarios:
        rows = ""
        for name, a in sorted(s["attribution"].items()):
            rows += f"""<tr>
                <td style="text-align:left">{name}</td>
                <td>{a['weight']*100:.0f}%</td>
                <td style="color:{clr(a['total_return'])}">{pct(a['total_return'])}</td>
                <td style="color:#dc2626">{pct(a['max_dd'])}</td>
            </tr>"""
        bg_border = "#16a34a" if s["survived"] else "#dc2626"
        attr_html += f"""
        <div style="background:#f8fafc;border:1px solid {bg_border};border-radius:8px;padding:12px;margin:8px 0">
            <strong>{s['label']}</strong> — Portfolio DD: <span style="color:#dc2626;font-weight:700">{pct(s['max_dd'])}</span>,
            Return: <span style="color:{clr(s['portfolio_return'])}">{pct(s['portfolio_return'])}</span>,
            Recovery: {s['recovery_days']}d
            <table style="margin-top:6px"><thead><tr><th>Strategy</th><th>Weight</th><th>Return</th><th>Max DD</th></tr></thead>
            <tbody>{rows}</tbody></table>
        </div>"""

    # Monte Carlo stats
    mc_html = f"""
    <div class="grid">
      <div class="card"><div class="card-label">Survival Rate</div>
        <div class="card-value" style="color:{'#16a34a' if mc['survival_rate'] > 0.95 else '#dc2626'}">{mc['survival_rate']*100:.1f}%</div></div>
      <div class="card"><div class="card-label">Ruin Rate (&lt;-90%)</div>
        <div class="card-value" style="color:{'#16a34a' if mc['ruin_rate'] < 0.01 else '#dc2626'}">{mc['ruin_rate']*100:.2f}%</div></div>
      <div class="card"><div class="card-label">Median DD</div>
        <div class="card-value">{pct(mc['median_dd'])}</div></div>
      <div class="card"><div class="card-label">P5 DD (worst 5%)</div>
        <div class="card-value" style="color:#dc2626">{pct(mc['p5_dd'])}</div></div>
      <div class="card"><div class="card-label">P1 DD (worst 1%)</div>
        <div class="card-value" style="color:#dc2626">{pct(mc['p1_dd'])}</div></div>
      <div class="card"><div class="card-label">Worst Single Path</div>
        <div class="card-value" style="color:#dc2626">{pct(mc['worst_dd'])}</div></div>
      <div class="card"><div class="card-label">Prob Positive Return</div>
        <div class="card-value">{mc['prob_positive']*100:.1f}%</div></div>
      <div class="card"><div class="card-label">Mean Recovery</div>
        <div class="card-value">{mc['mean_recovery']:.0f}d</div></div>
    </div>"""

    # Where it breaks analysis
    breaks_at = "unknown"
    for s in scenarios:
        if not s["survived"]:
            breaks_at = s["label"]
            break

    n_survived = sum(1 for s in scenarios if s["survived"])
    verdict_color = "#16a34a" if n_survived >= 3 else ("#ca8a04" if n_survived >= 2 else "#dc2626")
    verdict_text = f"{n_survived}/{len(scenarios)} Scenarios Survived"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ultimate Portfolio — Extreme Stress Test</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:0;padding:24px;background:#fff;color:#1e293b; }}
  h1 {{ font-size:1.5rem;margin-bottom:2px; }}
  h2 {{ font-size:1.1rem;color:#1d4ed8;margin:26px 0 8px;border-bottom:2px solid #e2e8f0;padding-bottom:4px; }}
  .meta {{ color:#64748b;font-size:0.82rem;margin-bottom:18px; }}
  .grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:18px; }}
  .card {{ background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px; }}
  .card-label {{ font-size:0.68rem;color:#64748b;text-transform:uppercase; }}
  .card-value {{ font-size:1.25rem;font-weight:700;margin-top:2px; }}
  table {{ width:100%;border-collapse:collapse;margin-bottom:12px;font-size:0.8rem; }}
  th {{ background:#f1f5f9;padding:5px 8px;text-align:right;font-size:0.72rem;color:#475569;border-bottom:2px solid #e2e8f0; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:4px 8px;text-align:right;border-bottom:1px solid #f1f5f9; }}
  td:first-child {{ text-align:left;font-weight:500; }}
  tr:hover td {{ background:#f8fafc; }}
  .verdict {{ border:2px solid {verdict_color};border-radius:10px;padding:14px;margin:16px 0;
              background:{'#f0fdf4' if n_survived >= 3 else '#fef9c3' if n_survived >= 2 else '#fef2f2'}; }}
  .verdict h3 {{ color:{verdict_color};margin:0 0 6px;font-size:1rem; }}
  .tag {{ display:inline-block;padding:2px 6px;border-radius:4px;font-size:0.68rem;font-weight:600;margin:1px; }}
  .tg {{ background:#dcfce7;color:#16a34a; }}
  .tr {{ background:#fef2f2;color:#dc2626; }}
  .tb {{ background:#dbeafe;color:#2563eb; }}
  .ty {{ background:#fef9c3;color:#ca8a04; }}
</style></head><body>

<h1>Ultimate Portfolio — Extreme Stress Test</h1>
<div class="meta">Generated {ts} | 4 strategies at 1.6x leverage | Scenarios worse than historical</div>

<div class="verdict">
  <h3>{verdict_text}</h3>
  <p style="margin:4px 0;font-size:0.85rem">
    Portfolio tested under 4 extreme scenarios + 10K Monte Carlo paths.
    {'Breaks under: <strong>' + breaks_at + '</strong>' if n_survived < len(scenarios) else 'All deterministic scenarios survived.'}
  </p>
  <span class="tag {'tg' if mc['survival_rate'] > 0.95 else 'tr'}">MC Survival: {mc['survival_rate']*100:.1f}%</span>
  <span class="tag {'tg' if mc['ruin_rate'] < 0.01 else 'tr'}">Ruin Rate: {mc['ruin_rate']*100:.2f}%</span>
  <span class="tag tb">P5 DD: {pct(mc['p5_dd'])}</span>
  <span class="tag ty">Worst DD: {pct(mc['worst_dd'])}</span>
</div>

<h2>1. Scenario Results Summary</h2>
<p style="color:#64748b;font-size:0.78rem">Each scenario is designed to be WORSE than the corresponding historical event. "Survived" = DD &gt; -50%.</p>
<table><thead><tr><th>Scenario</th><th>Return</th><th>Max DD</th><th>Recovery</th><th>Final Value</th><th>Status</th></tr></thead>
<tbody>{scenario_rows}</tbody></table>

<h2>2. Strategy Attribution per Scenario</h2>
<p style="color:#64748b;font-size:0.78rem">Which strategies protect, which fail? Positive return during crash = hedge payoff working.</p>
{attr_html}

<h2>3. Monte Carlo Stress (10K paths, fat tails)</h2>
<p style="color:#64748b;font-size:0.78rem">Student-t returns (df=4) for heavy tails. Each path is 1 year (252 days). VIX correlated with negative SPY.</p>
{mc_html}

<h2>4. Where It Breaks</h2>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;margin:8px 0">
  <table><thead><tr><th>Risk Factor</th><th>Threshold</th><th>Portfolio Response</th></tr></thead><tbody>
    <tr><td style="text-align:left">Single-day crash</td><td>-15%</td>
        <td style="color:{'#16a34a' if scenarios[2]['survived'] else '#dc2626'}">DD {pct(scenarios[2]['max_dd'])}, {'survives' if scenarios[2]['survived'] else 'FAILS'}</td></tr>
    <tr><td style="text-align:left">Rapid crash (46d)</td><td>-68%</td>
        <td style="color:{'#16a34a' if scenarios[0]['survived'] else '#dc2626'}">DD {pct(scenarios[0]['max_dd'])}, {'survives' if scenarios[0]['survived'] else 'FAILS'}</td></tr>
    <tr><td style="text-align:left">Prolonged bear (400d)</td><td>-50%</td>
        <td style="color:{'#16a34a' if scenarios[1]['survived'] else '#dc2626'}">DD {pct(scenarios[1]['max_dd'])}, {'survives' if scenarios[1]['survived'] else 'FAILS'}</td></tr>
    <tr><td style="text-align:left">Stagflation (252d)</td><td>High vol + grind</td>
        <td style="color:{'#16a34a' if scenarios[3]['survived'] else '#dc2626'}">DD {pct(scenarios[3]['max_dd'])}, {'survives' if scenarios[3]['survived'] else 'FAILS'}</td></tr>
    <tr><td style="text-align:left">Fat-tail P1 (MC)</td><td>Worst 1% path</td>
        <td style="color:#dc2626">DD {pct(mc['p1_dd'])}</td></tr>
  </tbody></table>
</div>

<div style="color:#94a3b8;font-size:0.68rem;margin-top:28px;border-top:1px solid #e2e8f0;padding-top:8px">
  PilotAI — Extreme Stress Test v1.0 | Ultimate Portfolio at 1.6x leverage |
  Scenarios: COVID x2, Prolonged Bear, Flash Cascade, Stagflation + 10K Monte Carlo
</div></body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ULTIMATE PORTFOLIO — EXTREME STRESS TEST")
    print(f"Portfolio: 4 strategies at {LEVERAGE}x leverage")
    print("=" * 70)

    scenarios = []

    # 1. COVID x2
    print("\n[1/5] COVID x2: -68% over 46 days...")
    spy, vix = gen_covid_x2()
    r = simulate_under_stress(spy, vix, "COVID x2 (-68% / 46d)")
    scenarios.append(r)
    print(f"      DD: {pct(r['max_dd'])}  Return: {pct(r['portfolio_return'])}  Recovery: {r['recovery_days']}d  {'SURVIVED' if r['survived'] else 'FAILED'}")

    # 2. Prolonged Bear
    print("\n[2/5] Prolonged Bear: -50% over 400 days...")
    spy, vix = gen_prolonged_bear()
    r = simulate_under_stress(spy, vix, "Prolonged Bear (-50% / 400d)")
    scenarios.append(r)
    print(f"      DD: {pct(r['max_dd'])}  Return: {pct(r['portfolio_return'])}  Recovery: {r['recovery_days']}d  {'SURVIVED' if r['survived'] else 'FAILED'}")

    # 3. Flash Crash Cascade
    print("\n[3/5] Flash Crash Cascade: -15% day 1...")
    spy, vix = gen_flash_cascade()
    r = simulate_under_stress(spy, vix, "Flash Crash Cascade (-15%/d1)")
    scenarios.append(r)
    print(f"      DD: {pct(r['max_dd'])}  Return: {pct(r['portfolio_return'])}  Recovery: {r['recovery_days']}d  {'SURVIVED' if r['survived'] else 'FAILED'}")

    # 4. Stagflation
    print("\n[4/5] Stagflation: 12 months high vol + slow decline...")
    spy, vix = gen_stagflation()
    r = simulate_under_stress(spy, vix, "Stagflation (12mo grind)")
    scenarios.append(r)
    print(f"      DD: {pct(r['max_dd'])}  Return: {pct(r['portfolio_return'])}  Recovery: {r['recovery_days']}d  {'SURVIVED' if r['survived'] else 'FAILED'}")

    # Strategy attribution
    for s in scenarios:
        print(f"\n    {s['label']} attribution:")
        for name, a in sorted(s["attribution"].items()):
            print(f"      {name:25s}  ret={pct(a['total_return'])}  DD={pct(a['max_dd'])}")

    # 5. Monte Carlo
    print("\n[5/5] Monte Carlo: 10K fat-tail paths...")
    mc = monte_carlo_stress(n_paths=10000)
    print(f"      Survival: {mc['survival_rate']*100:.1f}%  Ruin: {mc['ruin_rate']*100:.2f}%")
    print(f"      Median DD: {pct(mc['median_dd'])}  P5: {pct(mc['p5_dd'])}  P1: {pct(mc['p1_dd'])}  Worst: {pct(mc['worst_dd'])}")
    print(f"      Prob positive: {mc['prob_positive']*100:.1f}%  Mean recovery: {mc['mean_recovery']:.0f}d")

    # Generate report
    print("\n[6] Generating report...")
    html = build_html(scenarios, mc)
    out = ROOT / "reports" / "ultimate_portfolio_extreme_stress.html"
    out.write_text(html, encoding="utf-8")
    print(f"    {out}")

    n_survived = sum(1 for s in scenarios if s["survived"])
    print("\n" + "=" * 70)
    print(f"VERDICT: {n_survived}/{len(scenarios)} scenarios survived")
    print("=" * 70)
    for s in scenarios:
        status = "SURVIVED" if s["survived"] else "** FAILED **"
        print(f"  {s['label']:40s}  DD={pct(s['max_dd'])}  {status}")
    print(f"\n  MC survival rate: {mc['survival_rate']*100:.1f}%")
    print(f"  MC ruin rate:     {mc['ruin_rate']*100:.2f}%")
    print(f"  MC worst DD:      {pct(mc['worst_dd'])}")


if __name__ == "__main__":
    main()
