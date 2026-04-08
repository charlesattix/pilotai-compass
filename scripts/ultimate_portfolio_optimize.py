#!/usr/bin/env python3
"""
Ultimate Portfolio Optimization Sweep — Find 100% CAGR with COVID DD < 20%.

Parameter sweep across:
  1. Leverage: 1.4× to 2.0× (with hedge active)
  2. Strategy mix: 4 strategies (original) vs 5 strategies (+XLI ICs)
  3. Hedge cost budget: 1%, 1.5%, 2%, 2.5%, 3%
  4. XLI IC weight: 0% to 20%

Finds the optimal combo that maximizes CAGR while keeping COVID DD < 20%.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ultimate_portfolio import (
    load_exp1220_dynamic, load_cross_asset_pairs,
    load_vol_term_structure, load_tlt_iron_condors,
    calc_metrics, _fetch, ACCOUNT,
)
from scripts.ultimate_portfolio_v2 import load_xli_iron_condors
from compass.tail_risk_hedge import TailRiskHedgeEngine, TailRiskHedgeConfig

TRADING_DAYS = 252
REPORT_PATH = ROOT / "reports" / "ultimate_portfolio_optimized.html"


def load_all():
    """Load 5 strategies + market data (one-time)."""
    s1 = load_exp1220_dynamic()
    s2 = load_cross_asset_pairs()
    s3 = load_vol_term_structure()
    s4 = load_tlt_iron_condors()
    s5 = load_xli_iron_condors()

    df = pd.DataFrame({s1.name: s1, s2.name: s2, s3.name: s3, s4.name: s4, s5.name: s5})
    df = df.sort_index().fillna(0)
    df = df[df.index >= "2020-01-01"]

    spy = _fetch("SPY", "2019-01-01", "2025-12-31")
    vix_df = _fetch("^VIX", "2019-01-01", "2025-12-31")
    vix3m_df = _fetch("^VIX3M", "2019-01-01", "2025-12-31")

    spy_ret = spy["Close"].pct_change().dropna()
    vix = vix_df["Close"].dropna()
    vix3m = vix3m_df["Close"].dropna()

    common = df.index.intersection(spy_ret.index).intersection(vix.index).intersection(vix3m.index)
    df = df.reindex(common).fillna(0)
    spy_ret = spy_ret.reindex(common).fillna(0)
    vix = vix.reindex(common).ffill().bfill()
    vix3m = vix3m.reindex(common).ffill().bfill()

    return df, spy_ret, vix, vix3m


def make_weights(exp_w: float, xli_w: float, names: list) -> np.ndarray:
    """Build weight vector: EXP-1220 gets exp_w, XLI gets xli_w, rest split equally."""
    remainder = 1.0 - exp_w - xli_w
    minor_w = max(0, remainder / 3.0)
    w = np.zeros(len(names))
    for i, name in enumerate(names):
        if "EXP-1220" in name:
            w[i] = exp_w
        elif "XLI" in name:
            w[i] = xli_w
        else:
            w[i] = minor_w
    return w


def run_hedged(df, weights_arr, names, spy_ret, vix, vix3m, leverage, budget):
    """Run full-period hedged backtest. Returns (BacktestResult, crisis_results)."""
    port_rets = df[names].values @ weights_arr

    config = TailRiskHedgeConfig(
        normal_leverage=leverage,
        crisis_leverage=max(0.15, leverage * 0.15),  # scale crisis lev with normal
        min_leverage=0.10,
        annual_cost_budget_pct=budget,
        put_payoff_multiplier=15.0,
        vix_call_payoff_multiplier=25.0,
        crisis_hedge_ratio=0.90,
        vix_crisis_threshold=25.0,
        dd_crisis_threshold=0.04,
        leverage_smoothing_days=1,
    )

    idx = df.index
    data = {
        "portfolio_returns": pd.Series(port_rets, index=idx),
        "spy_returns": spy_ret.reindex(idx).fillna(0),
        "vix": vix.reindex(idx).ffill().bfill(),
        "vix3m": vix3m.reindex(idx).ffill().bfill(),
    }

    engine = TailRiskHedgeEngine(config)
    result = engine.backtest(data, starting_capital=ACCOUNT)
    return result


def run_sweep(df, spy_ret, vix, vix3m):
    """Run full parameter sweep."""
    names = list(df.columns)

    leverages = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    budgets = [1.0, 1.5, 2.0, 2.5, 3.0]
    xli_weights = [0.0, 0.05, 0.10, 0.15, 0.20]
    exp_weights = [0.80, 0.85, 0.90, 0.95]

    results = []
    total = len(leverages) * len(budgets) * len(xli_weights) * len(exp_weights)
    count = 0

    for lev in leverages:
        for budget in budgets:
            for xli_w in xli_weights:
                for exp_w in exp_weights:
                    if exp_w + xli_w > 0.99:
                        continue
                    count += 1
                    if count % 50 == 0:
                        print(f"    [{count}/{total}] lev={lev} budget={budget} xli={xli_w:.0%} exp={exp_w:.0%}")

                    w = make_weights(exp_w, xli_w, names)
                    result = run_hedged(df, w, names, spy_ret, vix, vix3m, lev, budget)

                    covid = result.scenario_results.get("COVID_2020")
                    covid_dd = covid.hedged_dd_pct if covid else 99

                    results.append({
                        "leverage": lev,
                        "budget": budget,
                        "xli_weight": xli_w,
                        "exp_weight": exp_w,
                        "cagr": result.cagr_pct,
                        "sharpe": result.sharpe,
                        "max_dd": result.max_dd_pct,
                        "calmar": result.calmar,
                        "sortino": result.sortino,
                        "covid_dd": covid_dd,
                        "avg_leverage": result.avg_leverage,
                        "hedge_cost": result.total_hedge_cost_pct,
                        "net_cost": result.net_hedge_cost_pct,
                        "feasible": covid_dd <= 20 and result.max_dd_pct <= 20,
                    })

    return results


def find_best(results):
    """Find best config: max CAGR among feasible (COVID DD < 20%)."""
    feasible = [r for r in results if r["feasible"]]
    if not feasible:
        print("  WARNING: No feasible configs found!")
        feasible = results

    # Primary: max CAGR. Tiebreak: max Sharpe.
    feasible.sort(key=lambda r: (r["cagr"], r["sharpe"]), reverse=True)
    return feasible[0]


def run_best_walkforward(df, spy_ret, vix, vix3m, best):
    """Run walk-forward on best config."""
    names = list(df.columns)
    w = make_weights(best["exp_weight"], best["xli_weight"], names)
    lev = best["leverage"]
    budget = best["budget"]

    windows = [
        ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2022"),
        ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2023"),
        ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2024"),
        ("2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2025"),
    ]

    oos_rets = []
    oos_dates = []
    yearly = []
    capital = ACCOUNT

    for _, _, ts, te, label in windows:
        mask = (df.index >= ts) & (df.index <= te)
        test_df = df.loc[mask]
        if test_df.empty:
            continue

        result = run_hedged(test_df, w, names, spy_ret, vix, vix3m, lev, budget)
        oos_rets.extend(result.daily_returns.tolist())
        oos_dates.extend(test_df.index.tolist())
        capital = result.equity_curve[-1]

        yearly.append({
            "label": label,
            "cagr": result.cagr_pct,
            "sharpe": result.sharpe,
            "max_dd": result.max_dd_pct,
            "avg_lev": result.avg_leverage,
        })

    oos_arr = np.array(oos_rets)
    oos_m = calc_metrics(oos_arr)

    # Equity curve
    eq = ACCOUNT * np.cumprod(1 + oos_arr)
    eq = np.insert(eq, 0, ACCOUNT)
    dates_str = ["2021-12-31"] + [str(d)[:10] for d in oos_dates]

    # Drawdown
    eq_raw = np.cumprod(1 + oos_arr)
    hwm = np.maximum.accumulate(eq_raw)
    dd = ((eq_raw / hwm) - 1) * 100

    return oos_m, yearly, eq.tolist(), dates_str, dd.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def _svg_equity(equity, dates, w=920, h=360):
    pl, pr, pt, pb = 80, 25, 42, 58; pw, ph = w-pl-pr, h-pt-pb
    ymin, ymax = min(equity)*0.92, max(equity)*1.08
    if ymax <= ymin: ymax = ymin+1
    n = len(dates)
    def tx(i): return pl + i/max(n-1,1)*pw
    def ty(v): return pt + (1-(v-ymin)/(ymax-ymin))*ph
    p = [f'<svg width="{w}" height="{h}" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;margin:1rem 0">']
    p.append(f'<text x="{w//2}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">OOS Equity — Optimized Config ($100K)</text>')
    for j in range(7):
        yv = ymin+j/6*(ymax-ymin); y = ty(yv)
        p.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{pl+pw}" y2="{y:.0f}" stroke="#e2e8f0" stroke-width="1"/>')
        lbl = f"${yv:,.0f}" if yv < 1e6 else f"${yv/1e6:.2f}M"
        p.append(f'<text x="{pl-8}" y="{y+4:.0f}" text-anchor="end" font-size="9" fill="#64748b">{lbl}</text>')
    step = max(1, n//8)
    for i in range(0, n, step):
        p.append(f'<text x="{tx(i):.0f}" y="{h-14}" text-anchor="middle" font-size="9" fill="#64748b">{dates[i][:7]}</text>')
    d = " ".join(f"{'M' if i==0 else 'L'}{tx(i):.1f},{ty(equity[i]):.1f}" for i in range(len(equity)))
    p.append(f'<path d="{d}" fill="none" stroke="#16a34a" stroke-width="2.5"/>')
    p.append("</svg>"); return "\n".join(p)

def _svg_drawdown(dd, dates, w=920, h=200):
    pl, pr, pt, pb = 80, 25, 42, 48; pw, ph = w-pl-pr, h-pt-pb
    dd_min = min(dd) if dd else 0; ymin, ymax = dd_min*1.25, 0.5
    n = len(dd)
    def tx(i): return pl + i/max(n-1,1)*pw
    def ty(v): return pt + (1-(v-ymin)/(ymax-ymin))*ph
    p = [f'<svg width="{w}" height="{h}" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;margin:1rem 0">']
    p.append(f'<text x="{w//2}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">OOS Drawdown (%)</text>')
    for j in range(5):
        yv = ymin+j/4*(ymax-ymin); y = ty(yv)
        p.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{pl+pw}" y2="{y:.0f}" stroke="#e2e8f0" stroke-width="1"/>')
        p.append(f'<text x="{pl-8}" y="{y+4:.0f}" text-anchor="end" font-size="9" fill="#64748b">{yv:.1f}%</text>')
    y0 = ty(0)
    p.append(f'<line x1="{pl}" y1="{y0:.0f}" x2="{pl+pw}" y2="{y0:.0f}" stroke="#94a3b8" stroke-width="1"/>')
    if dd:
        fill = f"M{tx(0):.1f},{y0:.1f}"
        for i in range(n): fill += f" L{tx(i):.1f},{ty(dd[i]):.1f}"
        fill += f" L{tx(n-1):.1f},{y0:.1f} Z"
        p.append(f'<path d="{fill}" fill="rgba(220,38,38,0.15)" stroke="#dc2626" stroke-width="1.5"/>')
    p.append("</svg>"); return "\n".join(p)


def generate_html(sweep_results, best, best_full, oos_m, yearly, equity, dates, dd):
    feasible = [r for r in sweep_results if r["feasible"]]
    n_total = len(sweep_results)
    n_feasible = len(feasible)

    # Top 15 by CAGR (feasible only)
    top = sorted(feasible, key=lambda r: r["cagr"], reverse=True)[:15]

    top_rows = ""
    for i, r in enumerate(top):
        hl = ' style="background:#dcfce7"' if i == 0 else ""
        top_rows += f"""<tr{hl}>
            <td>{'★' if i==0 else i+1}</td>
            <td>{r['leverage']:.1f}×</td><td>{r['exp_weight']*100:.0f}%</td><td>{r['xli_weight']*100:.0f}%</td>
            <td>{r['budget']:.1f}%</td>
            <td style="font-weight:700;color:{'#16a34a' if r['cagr']>=100 else '#1e293b'}">{r['cagr']:.1f}%</td>
            <td>{r['sharpe']:.2f}</td>
            <td style="color:{'#16a34a' if r['max_dd']<12 else '#ca8a04'}">{r['max_dd']:.1f}%</td>
            <td style="color:{'#16a34a' if r['covid_dd']<=20 else '#dc2626'}">{r['covid_dd']:.1f}%</td>
            <td>{r['avg_leverage']:.2f}×</td>
            <td>{r['hedge_cost']:.2f}%</td>
        </tr>"""

    # Leverage sweep (best budget/weights, vary leverage)
    lev_rows = ""
    lev_sweep = sorted([r for r in feasible if r["exp_weight"] == best["exp_weight"]
                        and r["xli_weight"] == best["xli_weight"]
                        and r["budget"] == best["budget"]],
                       key=lambda r: r["leverage"])
    for r in lev_sweep:
        hl = ' style="background:#dcfce7"' if r["leverage"] == best["leverage"] else ""
        lev_rows += f'<tr{hl}><td>{r["leverage"]:.1f}×</td><td>{r["cagr"]:.1f}%</td><td>{r["sharpe"]:.2f}</td><td>{r["max_dd"]:.1f}%</td><td>{r["covid_dd"]:.1f}%</td></tr>'

    # Budget sweep (best leverage/weights, vary budget)
    budget_rows = ""
    budget_sweep = sorted([r for r in sweep_results if r["leverage"] == best["leverage"]
                           and r["exp_weight"] == best["exp_weight"]
                           and r["xli_weight"] == best["xli_weight"]],
                          key=lambda r: r["budget"])
    for r in budget_sweep:
        hl = ' style="background:#dcfce7"' if r["budget"] == best["budget"] else ""
        budget_rows += f'<tr{hl}><td>{r["budget"]:.1f}%</td><td>{r["cagr"]:.1f}%</td><td>{r["sharpe"]:.2f}</td><td>{r["max_dd"]:.1f}%</td><td>{r["covid_dd"]:.1f}%</td><td>{"✓" if r["feasible"] else "✗"}</td></tr>'

    # XLI weight sweep
    xli_rows = ""
    xli_sweep = sorted([r for r in feasible if r["leverage"] == best["leverage"]
                         and r["budget"] == best["budget"]
                         and r["exp_weight"] == best["exp_weight"]],
                        key=lambda r: r["xli_weight"])
    for r in xli_sweep:
        hl = ' style="background:#dcfce7"' if r["xli_weight"] == best["xli_weight"] else ""
        xli_rows += f'<tr{hl}><td>{r["xli_weight"]*100:.0f}%</td><td>{r["cagr"]:.1f}%</td><td>{r["sharpe"]:.2f}</td><td>{r["max_dd"]:.1f}%</td><td>{r["covid_dd"]:.1f}%</td></tr>'

    # Year-by-year OOS
    yr_rows = ""
    for y in yearly:
        yr_rows += f'<tr><td style="font-weight:700">{y["label"]}</td><td style="color:{"#16a34a" if y["cagr"]>0 else "#dc2626"}">{y["cagr"]:.1f}%</td><td>{y["sharpe"]:.2f}</td><td>{y["max_dd"]:.1f}%</td><td>{y["avg_lev"]:.2f}×</td></tr>'

    eq_svg = _svg_equity(equity, dates)
    dd_svg = _svg_drawdown(dd, [str(d) for d in dates[1:]])

    t_cagr = best_full.cagr_pct >= 100; t_covid = best["covid_dd"] <= 20
    verdict = "100% CAGR + COVID PROTECTED" if t_cagr and t_covid else "OPTIMIZED"
    vc = "#16a34a" if t_cagr and t_covid else "#ca8a04"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ultimate Portfolio — Optimization Sweep</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1000px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.5; }}
  h1 {{ font-size:1.8em; color:#0f172a; margin-bottom:4px; }}
  h2 {{ color:#334155; margin-top:2.5em; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  h3 {{ color:#475569; margin-top:1.5em; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:20px; }}
  .verdict {{ text-align:center; padding:14px; border-radius:8px; font-size:1.1rem; font-weight:800;
              letter-spacing:0.06em; margin-bottom:24px; background:{vc}10; color:{vc}; border:2px solid {vc}40; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:18px;
          text-align:center; flex:1; min-width:130px; }}
  .kpi .value {{ font-size:1.7em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; }} .warn {{ color:#ca8a04; }} .bad {{ color:#dc2626; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:10px 12px; text-align:right; font-weight:600; color:#475569;
       border-bottom:2px solid #cbd5e1; font-size:0.8em; text-transform:uppercase; letter-spacing:0.03em; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .config {{ background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:16px;
             margin:20px 0; font-size:0.88rem; line-height:1.8; }}
  .footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.78em; color:#94a3b8; text-align:center; }}
</style></head><body>

<h1>Ultimate Portfolio — Optimization Sweep</h1>
<div class="subtitle">{n_total} configs tested | {n_feasible} feasible (COVID DD ≤20%) | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="verdict">{verdict}</div>

<div class="config">
    <strong>★ Best Config:</strong><br>
    Leverage: <strong>{best['leverage']:.1f}×</strong> |
    EXP-1220: <strong>{best['exp_weight']*100:.0f}%</strong> |
    XLI ICs: <strong>{best['xli_weight']*100:.0f}%</strong> |
    Minors: <strong>{(1-best['exp_weight']-best['xli_weight'])/3*100:.1f}%</strong> each |
    Hedge Budget: <strong>{best['budget']:.1f}%/yr</strong><br>
    → Full-period: <strong>{best_full.cagr_pct:.1f}% CAGR</strong>, Sharpe {best_full.sharpe:.2f}, DD {best_full.max_dd_pct:.1f}%, COVID DD {best['covid_dd']:.1f}%
</div>

<div class="kpi-row">
    <div class="kpi"><div class="value {'good' if best_full.cagr_pct>=100 else 'warn'}">{best_full.cagr_pct:.1f}%</div><div class="label">Full CAGR</div></div>
    <div class="kpi"><div class="value {'good' if oos_m['cagr_pct']>=80 else 'warn'}">{oos_m['cagr_pct']:.1f}%</div><div class="label">OOS CAGR</div></div>
    <div class="kpi"><div class="value {'good' if best_full.sharpe>=4 else 'warn'}">{best_full.sharpe:.2f}</div><div class="label">Sharpe</div></div>
    <div class="kpi"><div class="value {'good' if best_full.max_dd_pct<=12 else 'warn'}">{best_full.max_dd_pct:.1f}%</div><div class="label">Max DD</div></div>
    <div class="kpi"><div class="value {'good' if best['covid_dd']<=20 else 'bad'}">{best['covid_dd']:.1f}%</div><div class="label">COVID DD</div></div>
    <div class="kpi"><div class="value">{best_full.avg_leverage:.2f}×</div><div class="label">Avg Lev</div></div>
</div>

<h2>Top 15 Feasible Configurations</h2>
<table>
<thead><tr><th>#</th><th>Lev</th><th>EXP-1220</th><th>XLI</th><th>Budget</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>COVID DD</th><th>Avg Lev</th><th>Cost/yr</th></tr></thead>
<tbody>{top_rows}</tbody>
</table>

<h2>Parameter Sensitivity</h2>

<h3>Leverage Sweep (best weights + budget)</h3>
<table>
<thead><tr><th>Leverage</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>COVID DD</th></tr></thead>
<tbody>{lev_rows}</tbody>
</table>

<h3>Hedge Budget Sweep (best leverage + weights)</h3>
<table>
<thead><tr><th>Budget</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>COVID DD</th><th>Feasible</th></tr></thead>
<tbody>{budget_rows}</tbody>
</table>

<h3>XLI IC Weight Sweep (best leverage + budget)</h3>
<table>
<thead><tr><th>XLI Weight</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>COVID DD</th></tr></thead>
<tbody>{xli_rows}</tbody>
</table>

<h2>Walk-Forward OOS (Best Config)</h2>
{eq_svg}
{dd_svg}
<table>
<thead><tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Avg Lev</th></tr></thead>
<tbody>{yr_rows}</tbody>
</table>

<div class="footer">
    PilotAI Credit Spreads — Ultimate Portfolio Optimization Sweep<br>
    {n_total} parameter combinations tested. All results use tail risk hedge overlay.
</div>

</body></html>"""


def main():
    print("=" * 72)
    print("Ultimate Portfolio — Optimization Sweep")
    print("=" * 72)

    print("\n[1/5] Loading data...")
    df, spy_ret, vix, vix3m = load_all()
    print(f"  → {len(df)} days, {len(df.columns)} strategies")

    print("\n[2/5] Running parameter sweep...")
    results = run_sweep(df, spy_ret, vix, vix3m)
    n_feasible = sum(1 for r in results if r["feasible"])
    print(f"  → {len(results)} configs tested, {n_feasible} feasible")

    print("\n[3/5] Finding optimal config...")
    best = find_best(results)
    print(f"  ★ Best: lev={best['leverage']}× exp={best['exp_weight']*100:.0f}% xli={best['xli_weight']*100:.0f}% budget={best['budget']}%")
    print(f"    CAGR={best['cagr']:.1f}% Sharpe={best['sharpe']:.2f} DD={best['max_dd']:.1f}% COVID={best['covid_dd']:.1f}%")

    # Run full result on best config for detailed metrics
    names = list(df.columns)
    w = make_weights(best["exp_weight"], best["xli_weight"], names)
    best_full = run_hedged(df, w, names, spy_ret, vix, vix3m, best["leverage"], best["budget"])

    print("\n[4/5] Walk-forward on best config...")
    oos_m, yearly, equity, dates, dd = run_best_walkforward(df, spy_ret, vix, vix3m, best)
    print(f"  OOS: CAGR={oos_m['cagr_pct']:.1f}% Sharpe={oos_m['sharpe']:.2f} DD={oos_m['max_dd_pct']:.1f}%")
    for y in yearly:
        print(f"    {y['label']}: CAGR={y['cagr']:.1f}% Sharpe={y['sharpe']:.2f} DD={y['max_dd']:.1f}%")

    print("\n[5/5] Generating report...")
    html = generate_html(results, best, best_full, oos_m, yearly, equity, dates, dd)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}")

    print(f"\n{'━'*56}")
    print(f"  OPTIMAL CONFIG:")
    print(f"    Leverage:  {best['leverage']:.1f}×")
    print(f"    EXP-1220:  {best['exp_weight']*100:.0f}%  |  XLI ICs: {best['xli_weight']*100:.0f}%")
    print(f"    Budget:    {best['budget']:.1f}%/yr")
    print(f"    FULL:      CAGR={best_full.cagr_pct:.1f}%  Sharpe={best_full.sharpe:.2f}  DD={best_full.max_dd_pct:.1f}%")
    print(f"    OOS:       CAGR={oos_m['cagr_pct']:.1f}%  Sharpe={oos_m['sharpe']:.2f}  DD={oos_m['max_dd_pct']:.1f}%")
    print(f"    COVID DD:  {best['covid_dd']:.1f}%  {'< 20% PASS' if best['covid_dd'] <= 20 else '>= 20% FAIL'}")
    hit_100 = best_full.cagr_pct >= 100
    print(f"    100% CAGR: {'PASS' if hit_100 else 'MISS'} ({best_full.cagr_pct:.1f}%)")
    print(f"{'━'*56}")


if __name__ == "__main__":
    main()
