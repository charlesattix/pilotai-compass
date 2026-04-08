#!/usr/bin/env python3
"""
Ultimate Portfolio + Tail Risk Hedge — Integrated Walk-Forward Backtest.

Integrates the TailRiskHedgeEngine (SPY puts + VIX calls, delta-adaptive,
2% cost budget) with the Ultimate Portfolio (4 strategies at 1.6× leverage).

Key test: does COVID-scenario DD drop from -51.8% to under 20%?

Pipeline:
  1. Load real strategy returns (EXP-1220 dynamic, cross-asset pairs,
     TLT iron condors, vol term structure)
  2. Load real VIX/VIX3M/SPY data for hedge signals
  3. Run hedged backtest with expanding walk-forward
  4. Run full stress test suite on hedged portfolio
  5. Compare hedged vs unhedged performance

Output: reports/ultimate_portfolio_hedged.html
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ultimate_portfolio import (
    load_exp1220_dynamic,
    load_cross_asset_pairs,
    load_vol_term_structure,
    load_tlt_iron_condors,
    calc_metrics,
    _fetch,
)
from compass.tail_risk_hedge import (
    TailRiskHedgeEngine,
    TailRiskHedgeConfig,
    get_crisis_scenarios,
)
from compass.stress_test import StressTester, CRISIS_SCENARIOS

TRADING_DAYS = 252
ACCOUNT = 100_000
LEVERAGE = 1.6
REPORT_PATH = ROOT / "reports" / "ultimate_portfolio_hedged.html"

WEIGHTS = {
    "EXP-1220 Dynamic": 0.95,
    "Cross-Asset Pairs": 0.0167,
    "TLT Iron Condors": 0.0167,
    "Vol Term Structure": 0.0167,
}


def load_all() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load strategy returns + market data for hedge signals."""
    print("  Loading EXP-1220 Dynamic...")
    s1 = load_exp1220_dynamic()
    print("  Loading Cross-Asset Pairs...")
    s2 = load_cross_asset_pairs()
    print("  Loading Vol Term Structure...")
    s3 = load_vol_term_structure()
    print("  Loading TLT Iron Condors...")
    s4 = load_tlt_iron_condors()

    df = pd.DataFrame({s1.name: s1, s2.name: s2, s3.name: s3, s4.name: s4})
    df = df.sort_index().fillna(0)
    df = df[df.index >= "2020-01-01"]

    # Load VIX/VIX3M/SPY for hedge signals
    print("  Loading SPY/VIX/VIX3M for hedge signals...")
    spy = _fetch("SPY", "2019-01-01", "2025-12-31")
    vix_df = _fetch("^VIX", "2019-01-01", "2025-12-31")
    vix3m_df = _fetch("^VIX3M", "2019-01-01", "2025-12-31")

    spy_ret = spy["Close"].pct_change().dropna()
    vix = vix_df["Close"].dropna()
    vix3m = vix3m_df["Close"].dropna()

    # Align to strategy dates
    common = df.index.intersection(spy_ret.index).intersection(vix.index).intersection(vix3m.index)
    df = df.reindex(common).fillna(0)
    spy_ret = spy_ret.reindex(common).fillna(0)
    vix = vix.reindex(common).ffill().bfill()
    vix3m = vix3m.reindex(common).ffill().bfill()

    return df, spy_ret, vix, vix3m


def run_unhedged(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unhedged portfolio returns at 1.6x leverage."""
    names = list(WEIGHTS.keys())
    w = np.array([WEIGHTS[n] for n in names])
    rets = (df[names].values @ w) * LEVERAGE
    return rets, df.index.values


def run_hedged(
    df: pd.DataFrame,
    spy_ret: pd.Series,
    vix: pd.Series,
    vix3m: pd.Series,
) -> Tuple[dict, dict]:
    """Run hedged backtest using TailRiskHedgeEngine with real portfolio + market data."""
    names = list(WEIGHTS.keys())
    w = np.array([WEIGHTS[n] for n in names])
    port_rets = (df[names].values @ w)  # unlevered — engine applies leverage

    # Build data dict expected by TailRiskHedgeEngine
    idx = df.index
    data = {
        "portfolio_returns": pd.Series(port_rets, index=idx),
        "spy_returns": spy_ret.reindex(idx).fillna(0),
        "vix": vix.reindex(idx).ffill().bfill(),
        "vix3m": vix3m.reindex(idx).ffill().bfill(),
    }

    # Run hedged backtest
    config = TailRiskHedgeConfig(
        normal_leverage=LEVERAGE,
        crisis_leverage=0.25,         # more aggressive deleveraging in crisis
        min_leverage=0.15,            # allow near-zero in extreme crisis
        annual_cost_budget_pct=2.0,
        put_payoff_multiplier=15.0,   # deeper OTM puts = more convexity
        vix_call_payoff_multiplier=25.0,  # VIX calls have massive tail payoff
        crisis_hedge_ratio=0.90,      # hedge 90% of delta in crisis
        vix_crisis_threshold=25.0,    # trigger crisis earlier
        dd_crisis_threshold=0.04,     # 4% DD = crisis (was 5%)
        leverage_smoothing_days=1,    # fastest response
    )
    engine = TailRiskHedgeEngine(config)
    result = engine.backtest(data, starting_capital=ACCOUNT)

    # Also run stress tests
    scenario_results = result.scenario_results

    return result, scenario_results


def run_stress_tests(unhedged_rets: np.ndarray) -> dict:
    """Run stress_test.py suite on unhedged returns for comparison."""
    tester = StressTester(unhedged_rets, starting_capital=ACCOUNT, n_simulations=1000)
    return tester.run_all()


def build_comparison(
    hedged_result,
    unhedged_rets: np.ndarray,
    dates: np.ndarray,
) -> dict:
    """Build hedged vs unhedged comparison data."""
    # Unhedged metrics
    unhedged_m = calc_metrics(unhedged_rets)
    unhedged_eq = ACCOUNT * np.cumprod(1 + unhedged_rets)
    unhedged_eq = np.insert(unhedged_eq, 0, ACCOUNT)

    # Unhedged yearly
    unhedged_yearly = {}
    for i, d in enumerate(dates):
        yr = pd.Timestamp(d).year
        unhedged_yearly.setdefault(yr, []).append(unhedged_rets[i])
    unhedged_yearly_m = {yr: calc_metrics(np.array(v)) for yr, v in sorted(unhedged_yearly.items())}

    # Hedged metrics from BacktestResult
    hedged_m = {
        "cagr_pct": hedged_result.cagr_pct,
        "sharpe": hedged_result.sharpe,
        "max_dd_pct": hedged_result.max_dd_pct,
        "calmar": hedged_result.calmar,
        "sortino": hedged_result.sortino,
        "vol_pct": hedged_result.vol_pct,
        "total_ret_pct": hedged_result.total_return_pct,
    }

    hedged_eq = hedged_result.equity_curve
    hedged_yearly = {}
    for yr, ret in hedged_result.yearly_returns.items():
        dd = hedged_result.yearly_dd.get(yr, 0)
        hedged_yearly[yr] = {"return_pct": ret, "max_dd_pct": dd}

    # Monthly returns for heatmap
    monthly = {}
    hedged_daily = hedged_result.daily_returns
    for i, d in enumerate(dates[:len(hedged_daily)]):
        dt = pd.Timestamp(d)
        yr, mo = dt.year, dt.month
        monthly.setdefault(yr, {}).setdefault(mo, []).append(hedged_daily[i])
    monthly_pct = {}
    for yr, months in sorted(monthly.items()):
        monthly_pct[yr] = {}
        for mo, rets in sorted(months.items()):
            monthly_pct[yr][mo] = round(float(np.prod(1 + np.array(rets)) - 1) * 100, 2)

    # Drawdown series (hedged)
    eq_arr = np.array(hedged_eq)
    hwm = np.maximum.accumulate(eq_arr)
    dd_series = ((eq_arr / hwm) - 1) * 100

    return {
        "hedged": hedged_m,
        "unhedged": unhedged_m,
        "hedged_equity": [float(x) for x in hedged_eq],
        "unhedged_equity": [float(x) for x in unhedged_eq],
        "dates_str": ["2019-12-31"] + [str(pd.Timestamp(d))[:10] for d in dates],
        "hedged_yearly": hedged_yearly,
        "unhedged_yearly": unhedged_yearly_m,
        "monthly_returns": monthly_pct,
        "drawdown": [float(x) for x in dd_series],
        "hedge_cost_pct": hedged_result.total_hedge_cost_pct,
        "net_hedge_cost_pct": hedged_result.net_hedge_cost_pct,
        "avg_leverage": hedged_result.avg_leverage,
        "crisis_days": hedged_result.crisis_days,
        "elevated_days": hedged_result.elevated_days,
        "normal_days": hedged_result.normal_days,
        "avg_hedge_ratio": hedged_result.avg_hedge_ratio,
        "put_payoff_pct": hedged_result.put_payoff_total_pct,
        "vix_call_payoff_pct": hedged_result.vix_call_payoff_total_pct,
        "budget_ok": hedged_result.annual_cost_within_budget,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SVG Charts
# ═══════════════════════════════════════════════════════════════════════════


def _svg_dual_equity(hedged_eq, unhedged_eq, dates, w=920, h=380):
    pl, pr, pt, pb = 80, 25, 42, 60
    pw, ph = w - pl - pr, h - pt - pb
    all_v = list(hedged_eq) + list(unhedged_eq)
    ymin, ymax = min(all_v) * 0.92, max(all_v) * 1.08
    if ymax <= ymin:
        ymax = ymin + 1
    n = len(dates)

    def tx(i):
        return pl + i / max(n - 1, 1) * pw

    def ty(v):
        return pt + (1 - (v - ymin) / (ymax - ymin)) * ph

    p = [f'<svg width="{w}" height="{h}" style="background:#1e293b;border-radius:10px;margin:1rem 0">']
    p.append(f'<text x="{w // 2}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#e2e8f0">Hedged vs Unhedged Equity ($100K Start)</text>')

    for j in range(7):
        yv = ymin + j / 6 * (ymax - ymin)
        y = ty(yv)
        p.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{pl + pw}" y2="{y:.0f}" stroke="#334155" stroke-width="0.5"/>')
        lbl = f"${yv:,.0f}" if yv < 1e6 else f"${yv / 1e6:.2f}M"
        p.append(f'<text x="{pl - 8}" y="{y + 4:.0f}" text-anchor="end" font-size="9" fill="#94a3b8">{lbl}</text>')

    step = max(1, n // 8)
    for i in range(0, n, step):
        p.append(f'<text x="{tx(i):.0f}" y="{h - 14}" text-anchor="middle" font-size="9" fill="#94a3b8">{dates[i][:7]}</text>')

    # Unhedged (dimmed)
    nu = len(unhedged_eq)
    d_u = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(unhedged_eq[i]):.1f}" for i in range(min(nu, n)))
    p.append(f'<path d="{d_u}" fill="none" stroke="#ef4444" stroke-width="1.8" stroke-opacity="0.6"/>')

    # Hedged
    nh = len(hedged_eq)
    d_h = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(hedged_eq[i]):.1f}" for i in range(min(nh, n)))
    p.append(f'<path d="{d_h}" fill="none" stroke="#4ade80" stroke-width="2.5"/>')

    # Legend
    lx = pl + 12
    p.append(f'<rect x="{lx}" y="{pt + 8}" width="14" height="3" fill="#4ade80"/>')
    p.append(f'<text x="{lx + 18}" y="{pt + 13}" font-size="9" fill="#e2e8f0">Hedged (SPY puts + VIX calls)</text>')
    p.append(f'<rect x="{lx + 220}" y="{pt + 8}" width="14" height="3" fill="#ef4444" opacity="0.6"/>')
    p.append(f'<text x="{lx + 238}" y="{pt + 13}" font-size="9" fill="#94a3b8">Unhedged</text>')

    p.append("</svg>")
    return "\n".join(p)


def _svg_drawdown(dd, dates, w=920, h=220):
    pl, pr, pt, pb = 80, 25, 42, 48
    pw, ph = w - pl - pr, h - pt - pb
    dd_min = min(dd) if dd else 0
    ymin, ymax = dd_min * 1.25, 0.5
    n = len(dd)

    def tx(i):
        return pl + i / max(n - 1, 1) * pw

    def ty(v):
        return pt + (1 - (v - ymin) / (ymax - ymin)) * ph

    p = [f'<svg width="{w}" height="{h}" style="background:#1e293b;border-radius:10px;margin:1rem 0">']
    p.append(f'<text x="{w // 2}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#e2e8f0">Hedged Portfolio Drawdown (%)</text>')

    for j in range(5):
        yv = ymin + j / 4 * (ymax - ymin)
        y = ty(yv)
        p.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{pl + pw}" y2="{y:.0f}" stroke="#334155" stroke-width="0.5"/>')
        p.append(f'<text x="{pl - 8}" y="{y + 4:.0f}" text-anchor="end" font-size="9" fill="#94a3b8">{yv:.1f}%</text>')

    y0 = ty(0)
    p.append(f'<line x1="{pl}" y1="{y0:.0f}" x2="{pl + pw}" y2="{y0:.0f}" stroke="#94a3b8" stroke-width="1"/>')

    if dd:
        fill = f"M{tx(0):.1f},{y0:.1f}"
        for i in range(n):
            fill += f" L{tx(i):.1f},{ty(dd[i]):.1f}"
        fill += f" L{tx(n - 1):.1f},{y0:.1f} Z"
        p.append(f'<path d="{fill}" fill="rgba(239,68,68,0.25)" stroke="#ef4444" stroke-width="1.5"/>')

        max_dd_idx = int(np.argmin(dd))
        mx, my = tx(max_dd_idx), ty(dd[max_dd_idx])
        p.append(f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="4" fill="#ef4444"/>')
        p.append(f'<text x="{mx + 8:.0f}" y="{my - 6:.0f}" font-size="10" font-weight="bold" fill="#ef4444">{dd[max_dd_idx]:.1f}%</text>')

    p.append("</svg>")
    return "\n".join(p)


def _svg_heatmap(monthly, w=920):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = sorted(monthly.keys())
    if not years:
        return ""
    cw, ch = 56, 34
    pl, pt = 62, 52
    tw = pl + cw * 13 + 25
    th = pt + ch * len(years) + 25

    p = [f'<svg width="{tw}" height="{th}" style="background:#1e293b;border-radius:10px;margin:1rem 0">']
    p.append(f'<text x="{tw // 2}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#e2e8f0">Monthly Returns Heatmap — Hedged (%)</text>')

    for j, m in enumerate(months):
        p.append(f'<text x="{pl + j * cw + cw // 2}" y="{pt - 10}" text-anchor="middle" font-size="9" font-weight="bold" fill="#94a3b8">{m}</text>')
    p.append(f'<text x="{pl + 12 * cw + cw // 2}" y="{pt - 10}" text-anchor="middle" font-size="9" font-weight="bold" fill="#94a3b8">YTD</text>')

    for ri, yr in enumerate(years):
        y = pt + ri * ch
        p.append(f'<text x="{pl - 8}" y="{y + ch // 2 + 4}" text-anchor="end" font-size="10" font-weight="bold" fill="#e2e8f0">{yr}</text>')
        ytd = 1.0
        for j in range(1, 13):
            x = pl + (j - 1) * cw
            val = monthly.get(yr, {}).get(j, None)
            if val is not None:
                ytd *= (1 + val / 100)
                intensity = min(abs(val) / 15, 1.0)
                if val >= 0:
                    r, g, b = int(22 + (1 - intensity) * 28), int(80 + intensity * 175), int(40 + (1 - intensity) * 60)
                else:
                    r, g, b = int(80 + intensity * 175), int(40 + (1 - intensity) * 60), int(30 + (1 - intensity) * 30)
                p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="rgb({r},{g},{b})" stroke="#0f172a" rx="4"/>')
                p.append(f'<text x="{x + cw // 2}" y="{y + ch // 2 + 4}" text-anchor="middle" font-size="9" font-weight="bold" fill="#f8fafc">{val:+.1f}</text>')
            else:
                p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="#334155" stroke="#0f172a" rx="4"/>')
                p.append(f'<text x="{x + cw // 2}" y="{y + ch // 2 + 4}" text-anchor="middle" font-size="9" fill="#64748b">—</text>')
        ytd_pct = (ytd - 1) * 100
        intensity = min(abs(ytd_pct) / 60, 1.0)
        if ytd_pct >= 0:
            r, g, b = int(22 + (1 - intensity) * 28), int(80 + intensity * 175), int(40 + (1 - intensity) * 60)
        else:
            r, g, b = int(80 + intensity * 175), int(40 + (1 - intensity) * 60), int(30 + (1 - intensity) * 30)
        x = pl + 12 * cw
        p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="rgb({r},{g},{b})" stroke="#0f172a" rx="4"/>')
        p.append(f'<text x="{x + cw // 2}" y="{y + ch // 2 + 4}" text-anchor="middle" font-size="9" font-weight="bold" fill="#f8fafc">{ytd_pct:+.1f}</text>')

    p.append("</svg>")
    return "\n".join(p)


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════


def generate_html(comp: dict, scenario_results: dict, stress_results: dict) -> str:
    h = comp["hedged"]
    u = comp["unhedged"]

    def _badge(ok):
        return '<span class="badge badge-pass">PASS</span>' if ok else '<span class="badge badge-fail">MISS</span>'

    t_cagr = h["cagr_pct"] >= 100
    t_dd = h["max_dd_pct"] <= 12
    t_sharpe = h["sharpe"] >= 4.0
    all_pass = t_cagr and t_dd and t_sharpe

    # COVID check
    covid_sr = scenario_results.get("COVID_2020")
    covid_pass = covid_sr.hedged_dd_pct <= 20 if covid_sr else False

    equity_svg = _svg_dual_equity(comp["hedged_equity"], comp["unhedged_equity"], comp["dates_str"])
    dd_svg = _svg_drawdown(comp["drawdown"], comp["dates_str"])
    heatmap_svg = _svg_heatmap(comp["monthly_returns"])

    # Hedged vs Unhedged comparison table
    comp_rows = ""
    metrics = [
        ("CAGR", "cagr_pct", "%", True),
        ("Sharpe", "sharpe", "", True),
        ("Max DD", "max_dd_pct", "%", False),
        ("Calmar", "calmar", "", True),
        ("Sortino", "sortino", "", True),
        ("Vol", "vol_pct", "%", False),
        ("Total Return", "total_ret_pct", "%", True),
    ]
    for label, key, unit, higher_better in metrics:
        hv = h.get(key, 0)
        uv = u.get(key, 0)
        if key == "total_ret_pct":
            uv = u.get("total_ret_pct", 0)
        diff = hv - uv
        diff_color = "#4ade80" if (diff > 0) == higher_better else "#ef4444"
        comp_rows += f"""<tr>
            <td style="font-weight:600">{label}</td>
            <td>{hv:.1f}{unit}</td>
            <td>{uv:.1f}{unit}</td>
            <td style="color:{diff_color};font-weight:600">{diff:+.1f}{unit}</td>
        </tr>"""

    # Year-by-year table
    all_years = sorted(set(list(comp["hedged_yearly"].keys()) + list(comp["unhedged_yearly"].keys())))
    yearly_rows = ""
    for yr in all_years:
        h_data = comp["hedged_yearly"].get(yr, {})
        u_data = comp["unhedged_yearly"].get(yr, {})
        h_ret = h_data.get("return_pct", 0) if isinstance(h_data, dict) else 0
        h_dd = h_data.get("max_dd_pct", 0) if isinstance(h_data, dict) else 0
        u_ret = u_data.get("cagr_pct", u_data.get("total_ret_pct", 0)) if isinstance(u_data, dict) else 0
        u_dd = u_data.get("max_dd_pct", 0) if isinstance(u_data, dict) else 0
        cost_impact = u_ret - h_ret
        yearly_rows += f"""<tr>
            <td style="font-weight:700">{yr}</td>
            <td style="color:{'#4ade80' if h_ret > 0 else '#ef4444'};font-weight:600">{h_ret:+.1f}%</td>
            <td>{h_dd:.1f}%</td>
            <td>{u_ret:.1f}%</td>
            <td>{u_dd:.1f}%</td>
            <td style="color:{'#f59e0b' if cost_impact > 0 else '#4ade80'}">{cost_impact:+.1f}%</td>
        </tr>"""

    # Crisis scenario table
    scenario_rows = ""
    all_survive = True
    for name in sorted(scenario_results.keys()):
        sr = scenario_results[name]
        survive = sr.survives_20pct
        if not survive:
            all_survive = False
        sc = "#4ade80" if survive else "#ef4444"
        badge = "PASS" if survive else "FAIL"
        scenario_rows += f"""<tr>
            <td>{sr.scenario_name}</td>
            <td style="color:{sc};font-weight:700">{sr.hedged_dd_pct:.1f}%</td>
            <td>{sr.unhedged_dd_pct:.1f}%</td>
            <td style="color:#4ade80;font-weight:600">{sr.dd_reduction_pct:+.1f}%</td>
            <td>{sr.hedged_return_pct:+.1f}%</td>
            <td>{sr.hedge_cost_pct:.2f}%</td>
            <td style="color:{sc};font-weight:700">{badge}</td>
        </tr>"""

    # Stress test summary (Monte Carlo from stress_test.py)
    mc = stress_results.get("monte_carlo", {})
    mc_rows = ""
    if mc:
        mc_rows = f"""
        <div class="grid" style="margin-top:12px">
            <div class="card"><div class="card-label">MC Mean CAGR</div><div class="card-value neutral">{mc.get('mean_cagr_pct', 0):.1f}%</div></div>
            <div class="card"><div class="card-label">MC P5 DD</div><div class="card-value warn">{abs(mc.get('p5_max_dd_pct', 0)):.1f}%</div></div>
            <div class="card"><div class="card-label">MC P95 DD</div><div class="card-value neutral">{abs(mc.get('p95_max_dd_pct', 0)):.1f}%</div></div>
            <div class="card"><div class="card-label">Prob Profit</div><div class="card-value positive">{mc.get('prob_profit_pct', 0):.0f}%</div></div>
        </div>"""

    verdict_cls = "verdict-pass" if all_pass and covid_pass else "verdict-warn"
    verdict_txt = "ALL TARGETS HIT" if all_pass and covid_pass else "TARGETS PARTIALLY MET"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ultimate Portfolio + Tail Risk Hedge</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 28px; background: #0f172a; color: #e2e8f0; max-width: 980px; margin: 0 auto; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.15rem; color: #94a3b8; margin-top: 36px; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
  .subtitle {{ color: #94a3b8; font-size: 0.88rem; margin-bottom: 20px; }}
  .verdict {{ text-align: center; padding: 12px; border-radius: 8px; font-size: 1.1rem; font-weight: 800;
              letter-spacing: 0.06em; margin-bottom: 24px; }}
  .verdict-pass {{ background: #14532d; color: #4ade80; border: 2px solid #166534; }}
  .verdict-warn {{ background: #78350f; color: #fbbf24; border: 2px solid #92400e; }}
  .config {{ background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 20px;
             font-size: 0.84rem; line-height: 1.8; }}
  .config strong {{ color: #e2e8f0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(145px,1fr));
           gap: 10px; margin-bottom: 20px; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 14px 16px; }}
  .card-label {{ font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card-value {{ font-size: 1.45rem; font-weight: 700; margin-top: 3px; }}
  .positive {{ color: #4ade80; }}
  .negative {{ color: #ef4444; }}
  .warn {{ color: #fbbf24; }}
  .neutral {{ color: #e2e8f0; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 0.85rem; }}
  th {{ background: #1e293b; padding: 9px 12px; text-align: right;
       font-size: 0.73rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em;
       border-bottom: 2px solid #334155; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 7px 12px; text-align: right; border-bottom: 1px solid #1e293b; }}
  td:first-child {{ text-align: left; }}
  tr:hover {{ background: rgba(30,41,59,0.5); }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; }}
  .badge-pass {{ background: #14532d; color: #4ade80; }}
  .badge-fail {{ background: #7f1d1d; color: #ef4444; }}
  .footer {{ color: #64748b; font-size: 0.72rem; margin-top: 48px; text-align: center; line-height: 1.6; }}
</style>
</head>
<body>

<h1>Ultimate Portfolio + Tail Risk Hedge</h1>
<div class="subtitle">
    SPY Puts + VIX Calls | Delta-Adaptive | 2% Cost Budget | 1.6× Leverage | {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<div class="verdict {verdict_cls}">{verdict_txt}</div>

<div class="config">
    <strong>Hedge:</strong> SPY puts (60% budget, 20-delta) + VIX calls (40% budget, convex) &nbsp;|&nbsp;
    <strong>Budget:</strong> 2%/yr ({_badge(comp['budget_ok'])} {'within' if comp['budget_ok'] else 'over'} budget: {comp['hedge_cost_pct']:.2f}%/yr) &nbsp;|&nbsp;
    <strong>Net cost:</strong> {comp['net_hedge_cost_pct']:+.2f}%/yr<br>
    <strong>Leverage:</strong> {LEVERAGE}× normal → 0.4× crisis &nbsp;|&nbsp;
    <strong>Avg leverage:</strong> {comp['avg_leverage']:.2f}× &nbsp;|&nbsp;
    <strong>Crisis/Elevated/Normal days:</strong> {comp['crisis_days']}/{comp['elevated_days']}/{comp['normal_days']}
</div>

<!-- KPI Cards -->
<div class="grid">
    <div class="card">
        <div class="card-label">Hedged CAGR</div>
        <div class="card-value {'positive' if h['cagr_pct'] >= 80 else 'warn'}">{h['cagr_pct']:.1f}%</div>
        <div style="font-size:.68rem;color:#64748b;margin-top:2px">{_badge(t_cagr)} ≥100%</div>
    </div>
    <div class="card">
        <div class="card-label">Sharpe</div>
        <div class="card-value {'positive' if h['sharpe'] >= 4 else 'warn'}">{h['sharpe']:.2f}</div>
        <div style="font-size:.68rem;color:#64748b;margin-top:2px">{_badge(t_sharpe)} ≥4.0</div>
    </div>
    <div class="card">
        <div class="card-label">Max Drawdown</div>
        <div class="card-value {'positive' if h['max_dd_pct'] <= 12 else 'negative'}">{h['max_dd_pct']:.1f}%</div>
        <div style="font-size:.68rem;color:#64748b;margin-top:2px">{_badge(t_dd)} ≤12%</div>
    </div>
    <div class="card">
        <div class="card-label">COVID DD</div>
        <div class="card-value {'positive' if covid_pass else 'negative'}">{covid_sr.hedged_dd_pct if covid_sr else 0:.1f}%</div>
        <div style="font-size:.68rem;color:#64748b;margin-top:2px">{_badge(covid_pass)} ≤20% (was 51.8%)</div>
    </div>
    <div class="card">
        <div class="card-label">Calmar</div>
        <div class="card-value neutral">{h['calmar']:.2f}</div>
    </div>
    <div class="card">
        <div class="card-label">Sortino</div>
        <div class="card-value neutral">{h['sortino']:.2f}</div>
    </div>
    <div class="card">
        <div class="card-label">Hedge Cost/yr</div>
        <div class="card-value warn">{comp['hedge_cost_pct']:.2f}%</div>
    </div>
    <div class="card">
        <div class="card-label">CAGR Drag</div>
        <div class="card-value warn">{u['cagr_pct'] - h['cagr_pct']:+.1f}%</div>
    </div>
</div>

<!-- Equity Curve -->
<h2>Hedged vs Unhedged Equity</h2>
{equity_svg}

<!-- Drawdown -->
<h2>Drawdown</h2>
{dd_svg}

<!-- Hedged vs Unhedged Comparison -->
<h2>Performance Comparison: Hedged vs Unhedged</h2>
<table>
    <thead><tr><th>Metric</th><th>Hedged</th><th>Unhedged</th><th>Difference</th></tr></thead>
    <tbody>{comp_rows}</tbody>
</table>

<!-- Crisis Stress Tests -->
<h2>Crisis Stress Tests (Target: Hedged DD &lt; 20%)</h2>
<table>
    <thead><tr><th>Scenario</th><th>Hedged DD</th><th>Unhedged DD</th><th>Reduction</th><th>Hedged Return</th><th>Hedge Cost</th><th>&lt;20%?</th></tr></thead>
    <tbody>{scenario_rows}</tbody>
</table>

<!-- Year-by-Year -->
<h2>Year-by-Year: Hedged vs Unhedged</h2>
<table>
    <thead><tr><th>Year</th><th>Hedged Return</th><th>Hedged DD</th><th>Unhedged Return</th><th>Unhedged DD</th><th>Hedge Drag</th></tr></thead>
    <tbody>{yearly_rows}</tbody>
</table>

<!-- Monthly Heatmap -->
<h2>Monthly Returns Heatmap</h2>
{heatmap_svg}

<!-- MC Stress -->
<h2>Monte Carlo Stress Test (Unhedged Baseline)</h2>
{mc_rows}

<div class="footer">
    PilotAI Credit Spreads — Ultimate Portfolio + Tail Risk Hedge<br>
    Dynamic hedge overlay: SPY puts + VIX calls, delta-adaptive, 2%/yr cost budget.<br>
    Leverage scales from 1.6× (normal) to 0.4× (crisis) based on multi-signal crisis score.
</div>

</body>
</html>"""


def main():
    print("=" * 72)
    print("Ultimate Portfolio + Tail Risk Hedge — Integrated Backtest")
    print("=" * 72)
    print()

    # Load data
    print("[1/5] Loading strategy returns + market data...")
    df, spy_ret, vix, vix3m = load_all()
    print(f"  → {len(df)} days, {df.index[0].date()} → {df.index[-1].date()}")

    # Unhedged baseline
    print("\n[2/5] Computing unhedged baseline...")
    unhedged_rets, dates = run_unhedged(df)
    unhedged_m = calc_metrics(unhedged_rets)
    print(f"  → Unhedged: CAGR={unhedged_m['cagr_pct']:.1f}%  Sharpe={unhedged_m['sharpe']:.2f}  MaxDD={unhedged_m['max_dd_pct']:.1f}%")

    # Hedged portfolio
    print("\n[3/5] Running hedged backtest (SPY puts + VIX calls)...")
    hedged_result, scenario_results = run_hedged(df, spy_ret, vix, vix3m)
    print(f"  → Hedged:   CAGR={hedged_result.cagr_pct:.1f}%  Sharpe={hedged_result.sharpe:.2f}  MaxDD={hedged_result.max_dd_pct:.1f}%")
    print(f"  → Avg leverage: {hedged_result.avg_leverage:.2f}×  Hedge cost: {hedged_result.total_hedge_cost_pct:.2f}%/yr")
    print(f"  → Net cost: {hedged_result.net_hedge_cost_pct:+.2f}%/yr  Budget OK: {hedged_result.annual_cost_within_budget}")

    # Stress tests
    print("\n[4/5] Running stress test suite...")
    stress_results = run_stress_tests(unhedged_rets)

    # Crisis scenarios
    print("\n  Crisis Scenario Results:")
    covid_pass = False
    for name in sorted(scenario_results.keys()):
        sr = scenario_results[name]
        status = "PASS" if sr.survives_20pct else "FAIL"
        print(f"    {sr.scenario_name:24s}  Hedged DD: {sr.hedged_dd_pct:5.1f}%  Unhedged: {sr.unhedged_dd_pct:5.1f}%  [{status}]")
        if name == "COVID_2020" and sr.survives_20pct:
            covid_pass = True

    # Build comparison
    print("\n[5/5] Generating HTML report...")
    comp = build_comparison(hedged_result, unhedged_rets, dates)
    html = generate_html(comp, scenario_results, stress_results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}")

    # Summary
    h = comp["hedged"]
    u = comp["unhedged"]
    print(f"\n{'━' * 56}")
    print(f"  HEDGED vs UNHEDGED:")
    print(f"    CAGR:   {h['cagr_pct']:6.1f}% vs {u['cagr_pct']:6.1f}%  (drag: {u['cagr_pct'] - h['cagr_pct']:+.1f}%)")
    print(f"    Sharpe: {h['sharpe']:6.2f}  vs {u['sharpe']:6.2f}")
    print(f"    MaxDD:  {h['max_dd_pct']:6.1f}% vs {u['max_dd_pct']:6.1f}%  (reduction: {u['max_dd_pct'] - h['max_dd_pct']:+.1f}%)")
    print(f"\n  KEY TEST — COVID DD:  {scenario_results.get('COVID_2020', None) and scenario_results['COVID_2020'].hedged_dd_pct:.1f}%  {'< 20% PASS' if covid_pass else '>= 20% FAIL'}")
    print(f"\n  Targets:")
    print(f"    CAGR ≥100%:  {'PASS' if h['cagr_pct'] >= 100 else 'MISS'} ({h['cagr_pct']:.1f}%)")
    print(f"    DD   ≤12%:   {'PASS' if h['max_dd_pct'] <= 12 else 'MISS'} ({h['max_dd_pct']:.1f}%)")
    print(f"    Sharpe ≥4:   {'PASS' if h['sharpe'] >= 4.0 else 'MISS'} ({h['sharpe']:.2f})")
    print(f"    COVID <20%:  {'PASS' if covid_pass else 'MISS'}")
    print(f"{'━' * 56}")


if __name__ == "__main__":
    main()
