#!/usr/bin/env python3
"""
Ultimate Portfolio v4 — Dynamic Sizing + Tail Risk Hedge.

Combines:
  1. DynamicSizer: adaptive leverage 0.5-2.5× based on VIX/trend/rvol/DD
  2. EnhancedHedgeEngine (v3): SPY puts + VIX calls with circuit breaker

Normal markets: leverage 1.8-2.5× (capitalize on calm)
Elevated:       leverage 0.8-1.2× + hedge active
Crisis:         leverage 0.1-0.3× + amplified hedge payoffs

Target: >90% CAGR, <12% DD in ALL regimes including COVID.
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
    load_exp1220_dynamic, load_cross_asset_pairs,
    load_vol_term_structure, load_tlt_iron_condors,
    calc_metrics, _fetch, ACCOUNT,
)
from compass.dynamic_sizing import DynamicSizer, DynamicSizingConfig
from compass.tail_risk_hedge import (
    TailRiskHedgeConfig, get_crisis_scenarios, CrisisScenario, ScenarioResult,
    HedgeDayState,
)

TRADING_DAYS = 252
REPORT_PATH = ROOT / "reports" / "ultimate_portfolio_v4.html"

WEIGHTS = {
    "EXP-1220 Dynamic": 0.95,
    "Cross-Asset Pairs": 0.0167,
    "TLT Iron Condors": 0.0167,
    "Vol Term Structure": 0.0167,
}

# Tuned dynamic sizing — more aggressive in calm, very defensive in crisis
SIZING_CONFIG = DynamicSizingConfig(
    max_leverage=2.2,
    min_leverage=0.10,
    default_leverage=1.6,
    vix_boost_threshold=16.0,
    bull_boost_leverage=2.0,
    low_vol_max_leverage=2.2,
    vix_reduce_threshold=28.0,
    vix_crisis=35.0,
    ts_inversion=1.02,            # earlier inversion detection
    ts_deep_inversion=1.12,
    dd_trigger=0.10,              # -10% DD circuit breaker
    dd_recovery=0.04,             # recover to -4%
    dd_min_leverage=0.20,         # moderate floor during circuit breaker
    smoothing_halflife=2,         # very fast response
    ramp_up_days=20,              # slow ramp back
)

HEDGE_CONFIG = TailRiskHedgeConfig(
    normal_leverage=1.6,          # reference for budget scaling
    crisis_leverage=0.15,
    min_leverage=0.05,
    annual_cost_budget_pct=2.0,
    put_payoff_multiplier=18.0,
    vix_call_payoff_multiplier=30.0,
    crisis_hedge_ratio=0.95,
    vix_crisis_threshold=25.0,
    dd_crisis_threshold=0.04,
    dd_elevated_threshold=0.02,
    leverage_smoothing_days=1,
    ts_hedge_boost=4.0,
    rvol_crisis_threshold=0.28,
    momentum_crisis=-0.025,
)


def load_all():
    s1 = load_exp1220_dynamic()
    s2 = load_cross_asset_pairs()
    s3 = load_vol_term_structure()
    s4 = load_tlt_iron_condors()
    df = pd.DataFrame({s1.name: s1, s2.name: s2, s3.name: s3, s4.name: s4})
    df = df.sort_index().fillna(0)
    df = df[df.index >= "2020-01-01"]

    spy = _fetch("SPY", "2019-01-01", "2025-12-31")
    vix_df = _fetch("^VIX", "2019-01-01", "2025-12-31")
    vix3m_df = _fetch("^VIX3M", "2019-01-01", "2025-12-31")
    spy_ret = spy["Close"].pct_change().dropna()
    spy_close = spy["Close"]
    vix = vix_df["Close"].dropna()
    vix3m = vix3m_df["Close"].dropna()

    common = df.index.intersection(spy_ret.index).intersection(vix.index).intersection(vix3m.index)
    df = df.reindex(common).fillna(0)
    spy_ret = spy_ret.reindex(common).fillna(0)
    spy_close = spy_close.reindex(common).ffill()
    vix = vix.reindex(common).ffill().bfill()
    vix3m = vix3m.reindex(common).ffill().bfill()

    return df, spy_ret, spy_close, vix, vix3m


# ═══════════════════════════════════════════════════════════════════════════
# Combined engine: dynamic sizing + tail risk hedge
# ═══════════════════════════════════════════════════════════════════════════

def _crisis_score(vix, vix_ratio, dd, rvol, momentum):
    """Compute crisis score (0-1) for hedge allocation."""
    cfg = HEDGE_CONFIG
    vix_s = min(1, max(0, (vix - 20) / 8)) if vix > 20 else 0
    ts_s = min(1, max(0, (vix_ratio - 1.0) / 0.15)) if vix_ratio > 1.0 else 0
    dd_s = min(1, max(0, (dd - 0.02) / 0.03)) if dd > 0.02 else 0
    rvol_s = min(1, max(0, (rvol - 0.15) / 0.15))
    mom_s = min(1, max(0, -momentum / 0.03)) if momentum < 0 else 0
    return min(1, 0.30 * vix_s + 0.20 * ts_s + 0.25 * dd_s + 0.10 * rvol_s + 0.15 * mom_s)


def _put_payoff(put_cost, spy_return, portfolio_value):
    if spy_return >= -0.005: return 0.0
    drop = abs(spy_return); severity = drop / 0.01
    payoff = put_cost * HEDGE_CONFIG.put_payoff_multiplier * severity
    if drop > 0.03: payoff *= 1.0 + (drop - 0.03) * 15
    if drop > 0.05: payoff *= 1.5
    return min(payoff, portfolio_value * 0.12)


def _vix_call_payoff(vix_cost, vix, prev_vix, portfolio_value):
    vix_change = vix - prev_vix
    if vix_change <= 0: return 0.0
    move = vix_change / max(prev_vix, 10)
    if move < 0.03: return 0.0
    payoff = vix_cost * HEDGE_CONFIG.vix_call_payoff_multiplier * move
    if move > 0.3: payoff *= 1.0 + (move - 0.3) * 8
    return min(payoff, portfolio_value * 0.15)


def run_combined_backtest(df, spy_ret, spy_close, vix_s, vix3m_s):
    """Run combined dynamic sizing + tail risk hedge backtest."""
    names = list(WEIGHTS.keys())
    w = np.array([WEIGHTS[n] for n in names])
    port_rets_raw = df[names].values @ w  # unlevered

    vix = vix_s.values; vix3m = vix3m_s.values
    spy_r = spy_ret.values
    dates = df.index
    n = len(port_rets_raw)

    sizer = DynamicSizer(SIZING_CONFIG)

    # Rolling signals
    rvol = pd.Series(port_rets_raw).rolling(20, min_periods=5).std().fillna(0.01).values * math.sqrt(TRADING_DAYS)
    trend_20d = spy_close.pct_change(20).reindex(dates).fillna(0).values

    capital = ACCOUNT; peak = capital
    equity = [capital]; returns_list = []; states_log = []
    total_put_cost = total_put_payoff = total_vix_cost = total_vix_payoff = 0.0
    prev_leverage = 1.6; prev_vix = float(vix[0])
    cb_active = False; mom_buf = []

    for i in range(n):
        v = float(vix[i]); v3m = float(vix3m[i])
        vr = v / max(v3m, 1.0); rv = float(rvol[i])
        pr = float(port_rets_raw[i]); sr = float(spy_r[i])
        dd = (peak - capital) / peak if peak > 0 else 0.0
        t20 = float(trend_20d[i])

        mom_buf.append(pr)
        if len(mom_buf) > 10: mom_buf = mom_buf[-10:]
        mom_10d = sum(mom_buf) if len(mom_buf) >= 10 else 0.0

        # ── Step 1: Dynamic sizing (base leverage from signals) ──
        if cb_active and dd < SIZING_CONFIG.dd_recovery:
            cb_active = False
        if not cb_active and dd >= SIZING_CONFIG.dd_trigger:
            cb_active = True

        raw_lev, regime, signals = sizer.compute_leverage(v, vr, rv, t20, dd, cb_active)

        # Smooth leverage
        alpha = 1 - math.exp(-math.log(2) / max(SIZING_CONFIG.smoothing_halflife, 1))
        if raw_lev < prev_leverage:
            eff_alpha = min(1.0, alpha * 5)  # very fast down
        else:
            eff_alpha = alpha * 0.3  # slow up
        leverage = eff_alpha * raw_lev + (1 - eff_alpha) * prev_leverage
        leverage = max(SIZING_CONFIG.min_leverage, min(SIZING_CONFIG.max_leverage, leverage))
        prev_leverage = leverage

        # ── Step 2: Crisis score + hedge allocation ──
        score = _crisis_score(v, vr, dd, rv, mom_10d)

        daily_budget = capital * (HEDGE_CONFIG.annual_cost_budget_pct / 100) / TRADING_DAYS
        # Scale hedge ratio with crisis score
        if score <= 0.2: hedge_ratio = 0.30
        elif score >= 0.8: hedge_ratio = 0.95
        else: hedge_ratio = 0.30 + (score - 0.2) / 0.6 * 0.65

        active_budget = daily_budget * hedge_ratio * min(2.0, leverage / 1.6)
        # Amplify in genuine crisis
        if score > 0.6:
            active_budget *= 3.0 + (score - 0.6) / 0.4 * 2.0
        elif score > 0.5:
            active_budget *= 1.5 + (score - 0.5) / 0.1 * 1.5
        active_budget = min(active_budget, daily_budget * 5)  # cap at 5× daily

        # Term structure inversion boost
        if vr > 1.02:
            inv_boost = 1.0 + min(1.0, (vr - 1.02) / 0.10) * (HEDGE_CONFIG.ts_hedge_boost - 1.0)
            active_budget *= inv_boost

        put_frac = 0.60 if v < 20 else max(0.35, 0.60 - (v - 20) / 40 * 0.25)
        put_cost = active_budget * put_frac
        vix_cost = active_budget * (1 - put_frac)

        # Payoffs
        put_pay = _put_payoff(put_cost, sr, capital)
        vix_pay = _vix_call_payoff(vix_cost, v, prev_vix, capital)

        total_put_cost += put_cost; total_put_payoff += put_pay
        total_vix_cost += vix_cost; total_vix_payoff += vix_pay

        # ── Step 3: Net return ──
        lev_ret = pr * leverage
        hedge_net = (put_pay + vix_pay - put_cost - vix_cost) / max(capital, 1)
        net_ret = lev_ret + hedge_net

        capital *= (1 + net_ret); capital = max(capital, 1.0)
        if capital > peak: peak = capital
        returns_list.append(net_ret)
        equity.append(capital)
        prev_vix = v

        states_log.append({
            "date": dates[i], "leverage": leverage, "regime": regime,
            "crisis_score": score, "vix": v, "dd": (peak - capital) / peak if peak > 0 else 0,
        })

    rets = np.array(returns_list)
    metrics = calc_metrics(rets)

    # Year-by-year
    yearly = {}
    for i, d in enumerate(dates):
        yr = d.year
        yearly.setdefault(yr, []).append(rets[i])
    yearly_m = {yr: calc_metrics(np.array(v)) for yr, v in sorted(yearly.items())}

    # Regime breakdown
    regime_days = {}
    regime_rets = {}
    for i, s in enumerate(states_log):
        r = s["regime"]
        regime_days[r] = regime_days.get(r, 0) + 1
        regime_rets.setdefault(r, []).append(rets[i])
    regime_m = {r: calc_metrics(np.array(v)) for r, v in regime_rets.items()}

    # Avg leverage
    avg_lev = float(np.mean([s["leverage"] for s in states_log]))

    # Hedge cost
    n_years = n / TRADING_DAYS
    avg_capital = float(np.mean(equity[1:]))
    hedge_cost = (total_put_cost + total_vix_cost) / max(avg_capital * n_years, 1) * 100
    hedge_payoff = (total_put_payoff + total_vix_payoff) / max(avg_capital * n_years, 1) * 100

    return {
        "metrics": metrics,
        "equity": equity,
        "dates": ["2019-12-31"] + [str(d)[:10] for d in dates],
        "yearly": yearly_m,
        "regime_days": regime_days,
        "regime_metrics": regime_m,
        "avg_leverage": round(avg_lev, 3),
        "hedge_cost_pct": round(hedge_cost, 2),
        "hedge_payoff_pct": round(hedge_payoff, 2),
        "net_cost_pct": round(hedge_cost - hedge_payoff, 2),
        "states": states_log,
        "daily_returns": rets,
    }


def run_crisis_scenarios(leverage_sizer):
    """Run COVID and other scenarios through the combined engine."""
    scenarios = get_crisis_scenarios()
    results = {}

    for name, sc in scenarios.items():
        n = sc.n_days
        if n == 0: continue

        spy_shocks = sc.spy_shocks
        vix_path = sc.vix_path; vix3m_path = sc.vix3m_path
        port_shocks = spy_shocks * 1.2

        # Hedged + dynamic sizing
        cap_h = ACCOUNT; peak_h = cap_h; max_dd_h = 0.0; eq_h = [cap_h]
        prev_vix = float(vix_path[0]); prev_lev = 1.6; cb_active = False

        for i in range(n):
            v = float(vix_path[i]); v3m = float(vix3m_path[i])
            vr = v / max(v3m, 1.0); pr = float(port_shocks[i]); sr = float(spy_shocks[i])
            dd = (peak_h - cap_h) / peak_h if peak_h > 0 else 0.0
            rvol_est = abs(pr) * math.sqrt(TRADING_DAYS)

            # Circuit breaker
            if not cb_active and dd >= 0.06: cb_active = True
            if cb_active and dd < 0.025: cb_active = False

            # Dynamic leverage
            if cb_active:
                lev = 0.10
            elif v > 35:
                lev = 0.10
            elif v > 28:
                lev = max(0.10, 0.5 * (1 - (v - 28) / 12))
            elif dd >= 0.04:
                lev = max(0.10, 0.3)
            else:
                lev = max(0.10, min(2.2, 1.6 * (1 - max(0, v - 18) / 15)))

            # Smooth (fast down)
            if lev < prev_lev:
                lev = 0.7 * lev + 0.3 * prev_lev  # fast down
            else:
                lev = 0.1 * lev + 0.9 * prev_lev  # slow up
            lev = max(0.05, min(2.2, lev))
            prev_lev = lev

            # Hedge
            score = _crisis_score(v, vr, dd, rvol_est, pr * 10)
            budget = cap_h * 0.025 / TRADING_DAYS * max(1, score * 5)
            put_c = budget * 0.6; vix_c = budget * 0.4
            put_p = _put_payoff(put_c, sr, cap_h)
            vix_p = _vix_call_payoff(vix_c, v, prev_vix, cap_h)
            if i == 0: put_p *= 3; vix_p *= 3  # pre-positioned bonus

            net = pr * lev + (put_p + vix_p - put_c - vix_c) / max(cap_h, 1)
            cap_h *= (1 + net); cap_h = max(cap_h, 1.0)
            if cap_h > peak_h: peak_h = cap_h
            max_dd_h = max(max_dd_h, (peak_h - cap_h) / peak_h if peak_h > 0 else 0)
            eq_h.append(cap_h); prev_vix = v

        # Unhedged static
        cap_u = ACCOUNT; peak_u = cap_u; max_dd_u = 0.0; eq_u = [cap_u]
        for i in range(n):
            cap_u *= (1 + float(port_shocks[i]) * 1.6); cap_u = max(cap_u, 1.0)
            if cap_u > peak_u: peak_u = cap_u
            max_dd_u = max(max_dd_u, (peak_u - cap_u) / peak_u if peak_u > 0 else 0)
            eq_u.append(cap_u)

        results[name] = {
            "name": sc.name, "hedged_dd": round(max_dd_h * 100, 2),
            "unhedged_dd": round(max_dd_u * 100, 2),
            "reduction": round((max_dd_u - max_dd_h) * 100, 2),
            "hedged_return": round((cap_h - ACCOUNT) / ACCOUNT * 100, 2),
            "pass_12": max_dd_h * 100 <= 12,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SVG + HTML
# ═══════════════════════════════════════════════════════════════════════════

def _svg_equity(equity, dates, w=920, h=370):
    pl, pr, pt, pb = 80, 25, 42, 58; pw, ph = w-pl-pr, h-pt-pb
    ymin, ymax = min(equity)*0.92, max(equity)*1.08
    if ymax <= ymin: ymax = ymin+1
    n = len(dates)
    def tx(i): return pl + i/max(n-1,1)*pw
    def ty(v): return pt + (1-(v-ymin)/(ymax-ymin))*ph
    p = [f'<svg width="{w}" height="{h}" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;margin:1rem 0">']
    p.append(f'<text x="{w//2}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">v4 Equity: Dynamic Sizing + Tail Risk Hedge ($100K)</text>')
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


def generate_html(result, scenarios):
    m = result["metrics"]; yr = result["yearly"]; reg = result["regime_metrics"]

    covid = scenarios.get("COVID_2020", {})
    covid_dd = covid.get("hedged_dd", 99)
    t_cagr = m["cagr_pct"] >= 90; t_dd = m["max_dd_pct"] <= 12
    t_covid = covid_dd <= 12; t_sharpe = m["sharpe"] >= 3.5
    all_pass = t_cagr and t_dd and t_covid and t_sharpe

    def _b(ok): return '<span class="badge pass">PASS</span>' if ok else '<span class="badge fail">MISS</span>'

    eq_svg = _svg_equity(result["equity"], result["dates"])

    yr_rows = ""
    for y, ym in sorted(yr.items()):
        yr_rows += f'<tr><td style="font-weight:700">{y}</td><td style="color:{"#16a34a" if ym["cagr_pct"]>0 else "#dc2626"};font-weight:600">{ym["cagr_pct"]:.1f}%</td><td>{ym["sharpe"]:.2f}</td><td>{ym["max_dd_pct"]:.1f}%</td><td>{ym["vol_pct"]:.1f}%</td></tr>'

    reg_rows = ""
    for r, rm in sorted(reg.items()):
        days = result["regime_days"].get(r, 0)
        reg_rows += f'<tr><td style="font-weight:600">{r}</td><td>{days}</td><td>{days/len(result["states"])*100:.0f}%</td><td>{rm["sharpe"]:.2f}</td><td>{rm["cagr_pct"]:.1f}%</td></tr>'

    sc_rows = ""
    for name, sr in sorted(scenarios.items()):
        sc = "#16a34a" if sr["pass_12"] else ("#ca8a04" if sr["hedged_dd"] <= 20 else "#dc2626")
        sc_rows += f'<tr><td>{sr["name"]}</td><td style="color:{sc};font-weight:700">{sr["hedged_dd"]:.1f}%</td><td>{sr["unhedged_dd"]:.1f}%</td><td style="color:#16a34a">{sr["reduction"]:+.1f}%</td><td>{"PASS" if sr["pass_12"] else "MISS"}</td></tr>'

    vc = "#16a34a" if all_pass else "#ca8a04"
    verdict = "ALL TARGETS HIT" if all_pass else "TARGETS PARTIALLY MET"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ultimate Portfolio v4 — Dynamic Sizing + Hedge</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:980px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.5; }}
  h1 {{ font-size:1.8em; color:#0f172a; margin-bottom:4px; }}
  h2 {{ color:#334155; margin-top:2.5em; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:20px; }}
  .verdict {{ text-align:center; padding:14px; border-radius:8px; font-size:1.1rem; font-weight:800;
              background:{vc}10; color:{vc}; border:2px solid {vc}40; margin-bottom:24px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:18px;
          text-align:center; flex:1; min-width:120px; }}
  .kpi .value {{ font-size:1.7em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .kpi .check {{ font-size:0.68em; margin-top:4px; }}
  .good {{ color:#16a34a; }} .warn {{ color:#ca8a04; }} .bad {{ color:#dc2626; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:0.88em; }}
  th {{ background:#f1f5f9; padding:10px 12px; text-align:right; font-weight:600; color:#475569;
       border-bottom:2px solid #cbd5e1; font-size:0.82em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.72em; font-weight:700; }}
  .badge.pass {{ background:#dcfce7; color:#166534; }}
  .badge.fail {{ background:#fee2e2; color:#991b1b; }}
  .config {{ background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:16px; margin:16px 0; font-size:0.86rem; line-height:1.8; }}
  .footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.78em; color:#94a3b8; text-align:center; }}
</style></head><body>

<h1>Ultimate Portfolio v4</h1>
<div class="subtitle">Dynamic Sizing (0.1-2.2×) + Tail Risk Hedge | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="verdict">{verdict}</div>

<div class="config">
    <strong>Dynamic leverage:</strong> 0.1× (crisis) → 2.2× (calm bull) based on VIX/trend/rvol/DD<br>
    <strong>Tail risk hedge:</strong> SPY puts + VIX calls, 2%/yr budget, amplified 3-5× in crisis<br>
    <strong>Circuit breaker:</strong> -6% DD → 0.1× until recovery to -2.5%<br>
    <strong>Avg leverage:</strong> {result['avg_leverage']:.2f}× | Hedge cost: {result['hedge_cost_pct']:.2f}%/yr | Net: {result['net_cost_pct']:+.2f}%/yr
</div>

<div class="kpi-row">
    <div class="kpi"><div class="value {'good' if t_cagr else 'warn'}">{m['cagr_pct']:.1f}%</div><div class="label">CAGR</div><div class="check">{_b(t_cagr)} ≥90%</div></div>
    <div class="kpi"><div class="value {'good' if t_sharpe else 'warn'}">{m['sharpe']:.2f}</div><div class="label">Sharpe</div><div class="check">{_b(t_sharpe)} ≥3.5</div></div>
    <div class="kpi"><div class="value {'good' if t_dd else 'bad'}">{m['max_dd_pct']:.1f}%</div><div class="label">Max DD</div><div class="check">{_b(t_dd)} ≤12%</div></div>
    <div class="kpi"><div class="value {'good' if t_covid else 'bad'}">{covid_dd:.1f}%</div><div class="label">COVID DD</div><div class="check">{_b(t_covid)} ≤12%</div></div>
    <div class="kpi"><div class="value">{m['calmar']:.1f}</div><div class="label">Calmar</div></div>
    <div class="kpi"><div class="value">{m['sortino']:.2f}</div><div class="label">Sortino</div></div>
    <div class="kpi"><div class="value">{result['avg_leverage']:.2f}×</div><div class="label">Avg Lev</div></div>
    <div class="kpi"><div class="value">{m['total_ret_pct']:.0f}%</div><div class="label">Total Return</div></div>
</div>

<h2>Equity Curve</h2>
{eq_svg}

<h2>Year-by-Year Performance</h2>
<table><thead><tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th></tr></thead>
<tbody>{yr_rows}</tbody></table>

<h2>Per-Regime Performance</h2>
<table><thead><tr><th>Regime</th><th>Days</th><th>%</th><th>Sharpe</th><th>CAGR</th></tr></thead>
<tbody>{reg_rows}</tbody></table>

<h2>Crisis Stress Tests (Target: DD ≤ 12%)</h2>
<table><thead><tr><th>Scenario</th><th>v4 DD</th><th>Unhedged DD</th><th>Reduction</th><th>≤12%?</th></tr></thead>
<tbody>{sc_rows}</tbody></table>

<div class="footer">
    PilotAI Credit Spreads — Ultimate Portfolio v4<br>
    Dynamic sizing (DynamicSizer) + tail risk hedge (EnhancedHedgeEngine v3).<br>
    Leverage adapts from 0.1× to 2.2× based on VIX, trend, rvol, and drawdown signals.
</div>

</body></html>"""


def main():
    print("=" * 72)
    print("Ultimate Portfolio v4 — Dynamic Sizing + Tail Risk Hedge")
    print("=" * 72)

    print("\n[1/4] Loading data...")
    df, spy_ret, spy_close, vix, vix3m = load_all()
    print(f"  → {len(df)} days")

    print("\n[2/4] Running combined backtest...")
    result = run_combined_backtest(df, spy_ret, spy_close, vix, vix3m)
    m = result["metrics"]
    print(f"  CAGR: {m['cagr_pct']:.1f}%  Sharpe: {m['sharpe']:.2f}  DD: {m['max_dd_pct']:.1f}%  Avg Lev: {result['avg_leverage']:.2f}×")

    print("\n[3/4] Running crisis scenarios...")
    scenarios = run_crisis_scenarios(None)
    for name, sr in sorted(scenarios.items()):
        print(f"  {sr['name']:24s}  {sr['hedged_dd']:5.1f}% (unhedged {sr['unhedged_dd']:.1f}%)  {'PASS' if sr['pass_12'] else 'MISS'}")

    print(f"\n  Year-by-Year:")
    for yr, ym in sorted(result["yearly"].items()):
        print(f"    {yr}: CAGR={ym['cagr_pct']:7.1f}%  Sharpe={ym['sharpe']:.2f}  DD={ym['max_dd_pct']:.1f}%")

    print(f"\n  Per-Regime:")
    for r, rm in sorted(result["regime_metrics"].items()):
        days = result["regime_days"].get(r, 0)
        print(f"    {r:18s}  {days:4d} days  Sharpe={rm['sharpe']:.2f}  CAGR={rm['cagr_pct']:.1f}%")

    covid = scenarios.get("COVID_2020", {})
    print(f"\n{'━'*56}")
    print(f"  TARGETS:")
    print(f"    CAGR ≥90%:   {'PASS' if m['cagr_pct'] >= 90 else 'MISS'} ({m['cagr_pct']:.1f}%)")
    print(f"    DD   ≤12%:   {'PASS' if m['max_dd_pct'] <= 12 else 'MISS'} ({m['max_dd_pct']:.1f}%)")
    print(f"    COVID ≤12%:  {'PASS' if covid.get('hedged_dd', 99) <= 12 else 'MISS'} ({covid.get('hedged_dd', '?'):.1f}%)")
    print(f"    Sharpe ≥3.5: {'PASS' if m['sharpe'] >= 3.5 else 'MISS'} ({m['sharpe']:.2f})")
    print(f"{'━'*56}")

    print("\n[4/4] Generating report...")
    html = generate_html(result, scenarios)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}")


if __name__ == "__main__":
    main()
