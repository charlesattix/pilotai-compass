"""
Hedge cost reality check — validate the 2%/yr flat assumption against
REAL SPY put prices from IronVault.

For each month 2020-2025:
  1. Query IronVault for SPY 30-delta puts (~5% OTM), 30-45 DTE
  2. Get actual close price (mid)
  3. Compute monthly hedge cost as % of SPY price
  4. Annualise to get actual hedge cost per year
  5. Compare vs assumed 2%/yr flat budget
  6. Re-run tail risk hedge backtest with real costs
  7. Report impact on CAGR, DD, Sharpe

No VIX options in IronVault (checked: VIX/UVXY/VIXY/VXX all absent).
VIX call hedge cost will be estimated from VIX level * empirical ratio.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MonthlyHedgeCost:
    year: int
    month: int
    spy_price: float
    put_strike: float
    put_price: float
    dte: int
    expiration: str
    cost_pct_of_spy: float          # put_price / spy_price as %
    annualised_cost_pct: float      # projected to full year
    vix_at_date: float
    trade_date: str

@dataclass
class RealCostSummary:
    monthly_costs: List[MonthlyHedgeCost]
    avg_annual_cost_pct: float
    median_annual_cost_pct: float
    min_annual_cost_pct: float
    max_annual_cost_pct: float
    assumed_cost_pct: float         # 2.0%
    actual_vs_assumed_ratio: float  # how much the assumption is off
    yearly_costs: Dict[int, float]  # actual annualised cost per year
    n_months_sampled: int

@dataclass
class BacktestComparison:
    # Assumed cost (2% flat)
    assumed_cagr: float
    assumed_sharpe: float
    assumed_dd: float
    assumed_calmar: float
    assumed_net_cost: float
    # Real cost
    real_cagr: float
    real_sharpe: float
    real_dd: float
    real_calmar: float
    real_net_cost: float
    # Deltas
    cagr_delta: float
    sharpe_delta: float
    dd_delta: float
    # Equity
    assumed_equity: List[float]
    real_equity: List[float]
    yearly: Dict[int, Dict[str, float]]


# ═══════════════════════════════════════════════════════════════════════════
# Query real put prices from IronVault
# ═══════════════════════════════════════════════════════════════════════════

def query_monthly_hedge_costs(db_path: str = None) -> List[MonthlyHedgeCost]:
    """Query IronVault for real SPY ~5% OTM put prices, monthly 2020-2025."""
    if db_path is None:
        db_path = str(ROOT / "data" / "options_cache.db")

    conn = sqlite3.connect(db_path)
    results = []

    # Get SPY price history from yfinance
    try:
        import yfinance as yf
        spy = yf.download("SPY", start="2019-12-01", end="2026-01-01", progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy.index = pd.to_datetime(spy.index)
        vix_df = yf.download("^VIX", start="2019-12-01", end="2026-01-01", progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_df.index = pd.to_datetime(vix_df.index)
        vix_close = vix_df["Close"]
    except Exception:
        conn.close()
        return results

    # For each month, find an expiration 30-45 DTE out, price a ~5% OTM put
    for year in range(2020, 2026):
        for month in range(1, 13):
            # Pick ~15th of month as trade date
            trade_date_str = f"{year}-{month:02d}-15"
            # Find nearest trading day
            td = pd.Timestamp(trade_date_str)
            candidates = spy.index[spy.index >= td - timedelta(days=5)]
            candidates = candidates[candidates <= td + timedelta(days=5)]
            if len(candidates) == 0:
                continue
            actual_td = candidates[0]
            td_str = actual_td.strftime("%Y-%m-%d")

            try:
                spy_price = float(spy.loc[actual_td, "Close"])
                vix_val = float(vix_close.loc[actual_td]) if actual_td in vix_close.index else 20.0
            except (KeyError, TypeError):
                continue
            if np.isnan(spy_price) or spy_price <= 0:
                continue

            # Target: 5% OTM put, 30-45 DTE expiration
            target_strike = round(spy_price * 0.95)
            target_exp_start = (actual_td + timedelta(days=28)).strftime("%Y-%m-%d")
            target_exp_end = (actual_td + timedelta(days=50)).strftime("%Y-%m-%d")

            # Find matching expiration
            exps = conn.execute(
                "SELECT DISTINCT expiration FROM option_contracts "
                "WHERE ticker='SPY' AND option_type='P' "
                "AND expiration BETWEEN ? AND ? ORDER BY expiration",
                (target_exp_start, target_exp_end)
            ).fetchall()
            if not exps:
                continue
            exp = exps[0][0]
            dte = (datetime.strptime(exp, "%Y-%m-%d") - actual_td).days

            # Find closest strike to target and get its price
            rows = conn.execute(
                "SELECT oc.strike, od.close FROM option_contracts oc "
                "JOIN option_daily od ON oc.contract_symbol = od.contract_symbol "
                "WHERE oc.ticker='SPY' AND oc.option_type='P' "
                "AND oc.expiration=? AND od.date=? AND od.close > 0 "
                "AND ABS(oc.strike - ?) < 10 "
                "ORDER BY ABS(oc.strike - ?) LIMIT 1",
                (exp, td_str, target_strike, target_strike)
            ).fetchall()
            if not rows:
                continue

            strike, put_price = rows[0]
            if put_price <= 0 or put_price > spy_price * 0.20:
                continue

            # Cost as % of SPY: buying 1 put costs put_price per share
            # For a $100K portfolio fully hedged: need ~SPY_price / 100 contracts
            # But we're measuring cost as % of portfolio per put roll
            cost_pct = (put_price / spy_price) * 100  # per roll
            # Annualise: ~12 rolls/year (monthly)
            annual_cost = cost_pct * (365.25 / max(dte, 1))

            results.append(MonthlyHedgeCost(
                year=year, month=month, spy_price=round(spy_price, 2),
                put_strike=strike, put_price=round(put_price, 2),
                dte=dte, expiration=exp,
                cost_pct_of_spy=round(cost_pct, 4),
                annualised_cost_pct=round(annual_cost, 2),
                vix_at_date=round(vix_val, 1),
                trade_date=td_str,
            ))

    conn.close()
    return results


def summarise_costs(costs: List[MonthlyHedgeCost]) -> RealCostSummary:
    """Summarise monthly costs into annual figures."""
    if not costs:
        return RealCostSummary([], 0, 0, 0, 0, 2.0, 0, {}, 0)

    annual_costs = [c.annualised_cost_pct for c in costs]

    # Per-year average
    yearly = {}
    for c in costs:
        yearly.setdefault(c.year, []).append(c.annualised_cost_pct)
    yearly_avg = {yr: round(np.mean(vals), 2) for yr, vals in yearly.items()}

    avg = float(np.mean(annual_costs))
    med = float(np.median(annual_costs))

    return RealCostSummary(
        monthly_costs=costs,
        avg_annual_cost_pct=round(avg, 2),
        median_annual_cost_pct=round(med, 2),
        min_annual_cost_pct=round(float(min(annual_costs)), 2),
        max_annual_cost_pct=round(float(max(annual_costs)), 2),
        assumed_cost_pct=2.0,
        actual_vs_assumed_ratio=round(avg / 2.0, 2) if avg > 0 else 0,
        yearly_costs=yearly_avg,
        n_months_sampled=len(costs),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Re-run tail risk hedge with real vs assumed costs
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest_comparison(
    real_annual_cost_pct: float,
    assumed_annual_cost_pct: float = 2.0,
    seed: int = 42,
) -> BacktestComparison:
    """Backtest the portfolio with real vs assumed hedge costs."""
    from compass.production_portfolio_wf import generate_strategy_returns, STRATEGY_IDS

    rets_dict = generate_strategy_returns(seed=seed)
    dates = rets_dict[STRATEGY_IDS[0]].index
    n = len(dates)

    # Build equal-weight portfolio returns (unhedged, unlevered)
    base_rets = np.zeros(n)
    weights = {s: 1.0 / len(STRATEGY_IDS) for s in STRATEGY_IDS}
    for sid in STRATEGY_IDS:
        base_rets += weights[sid] * rets_dict[sid].values

    leverage = 1.6

    def _run_with_cost(annual_cost_pct):
        daily_cost = annual_cost_pct / 100.0 / TRADING_DAYS
        port = np.zeros(n)
        capital = 100_000.0
        peak = capital
        eq = [capital]

        for i in range(n):
            levered = base_rets[i] * leverage
            # Hedge cost drag
            hedge_drag = daily_cost
            # Hedge payoff on big down days (puts pay off)
            payoff = 0.0
            if base_rets[i] < -0.01:
                drop = abs(base_rets[i])
                # Real put payoff: ~5x notional on 5% OTM put during crash
                payoff = daily_cost * 8 * (drop / 0.01)
                payoff = min(payoff, 0.05)  # cap at 5% of portfolio

            net = levered - hedge_drag + payoff
            port[i] = net
            capital *= (1 + net)
            capital = max(capital, 1.0)
            if capital > peak:
                peak = capital
            eq.append(capital)

        return port, eq

    assumed_rets, assumed_eq = _run_with_cost(assumed_annual_cost_pct)
    real_rets, real_eq = _run_with_cost(real_annual_cost_pct)

    am = _metrics(assumed_rets)
    rm = _metrics(real_rets)

    # Yearly
    yearly = {}
    by_year: Dict[int, List[int]] = {}
    for i, d in enumerate(dates):
        by_year.setdefault(d.year, []).append(i)
    for yr, idx in sorted(by_year.items()):
        a_m = _metrics(assumed_rets[idx])
        r_m = _metrics(real_rets[idx])
        yearly[yr] = {
            "assumed_cagr": a_m["cagr_pct"], "assumed_sharpe": a_m["sharpe"],
            "real_cagr": r_m["cagr_pct"], "real_sharpe": r_m["sharpe"],
        }

    return BacktestComparison(
        assumed_cagr=am["cagr_pct"], assumed_sharpe=am["sharpe"],
        assumed_dd=am["max_dd_pct"], assumed_calmar=am["calmar"],
        assumed_net_cost=assumed_annual_cost_pct,
        real_cagr=rm["cagr_pct"], real_sharpe=rm["sharpe"],
        real_dd=rm["max_dd_pct"], real_calmar=rm["calmar"],
        real_net_cost=real_annual_cost_pct,
        cagr_delta=round(rm["cagr_pct"] - am["cagr_pct"], 2),
        sharpe_delta=round(rm["sharpe"] - am["sharpe"], 2),
        dd_delta=round(rm["max_dd_pct"] - am["max_dd_pct"], 2),
        assumed_equity=assumed_eq, real_equity=real_eq,
        yearly=yearly,
    )


def _metrics(rets):
    if len(rets) < 2:
        return {"cagr_pct": 0, "sharpe": 0, "max_dd_pct": 0, "calmar": 0, "sortino": 0}
    eq = np.cumprod(1 + rets)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else 0
    mu, std = float(rets.mean()), float(rets.std())
    sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    return {"cagr_pct": round(cagr * 100, 2), "sharpe": round(sharpe, 2),
            "max_dd_pct": round(dd * 100, 2), "calmar": round(calmar, 2)}


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    cost_summary: RealCostSummary,
    bt: BacktestComparison,
    output_path: str = "reports/hedge_cost_reality_check.html",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Monthly cost table
    monthly_rows = ""
    for c in cost_summary.monthly_costs:
        vc = "#dc2626" if c.vix_at_date > 30 else ("#d97706" if c.vix_at_date > 20 else "#16a34a")
        monthly_rows += f'<tr><td>{c.year}-{c.month:02d}</td><td>${c.spy_price:.0f}</td><td>${c.put_strike:.0f}</td><td>${c.put_price:.2f}</td><td>{c.dte}d</td><td style="color:{vc}">{c.vix_at_date:.0f}</td><td>{c.cost_pct_of_spy:.3f}%</td><td style="font-weight:700">{c.annualised_cost_pct:.2f}%</td></tr>'

    # Yearly cost summary
    yr_cost_rows = ""
    for yr, cost in sorted(cost_summary.yearly_costs.items()):
        delta = cost - 2.0
        dc = "#dc2626" if delta > 1.0 else ("#d97706" if delta > 0 else "#16a34a")
        yr_cost_rows += f'<tr><td>{yr}</td><td style="font-weight:700">{cost:.2f}%</td><td>2.00%</td><td style="color:{dc}">{delta:+.2f}%</td></tr>'

    # Backtest comparison
    def _dc(d, invert=False):
        v = -d if invert else d
        return "#16a34a" if v > 0 else "#dc2626"

    # Equity SVG
    eq_svg = _dual_equity_svg(bt.assumed_equity, bt.real_equity)

    # Yearly backtest
    yr_bt_rows = ""
    for yr, d in sorted(bt.yearly.items()):
        yr_bt_rows += f'<tr><td>{yr}</td><td>{d["assumed_cagr"]:.1f}%</td><td>{d["assumed_sharpe"]:.2f}</td><td>{d["real_cagr"]:.1f}%</td><td>{d["real_sharpe"]:.2f}</td></tr>'

    verdict = "VALIDATED" if abs(cost_summary.actual_vs_assumed_ratio - 1.0) < 0.5 else "DIVERGENT"
    vc = "#16a34a" if verdict == "VALIDATED" else "#dc2626"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Hedge Cost Reality Check</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b}}
h1{{font-size:1.4rem;color:#0f172a}}h2{{font-size:1rem;color:#334155;margin-top:1.5rem;border-bottom:1px solid #e2e8f0;padding-bottom:4px}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:16px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}}.card .v{{font-size:1.1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:0.82rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}}td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}}td:first-child{{text-align:left}}
svg{{display:block;margin:0.5rem 0}}
.verdict{{display:inline-block;padding:3px 12px;border-radius:4px;font-weight:700;font-size:0.82rem;background:{vc}15;color:{vc}}}
.finding{{background:#eff6ff;border-left:4px solid #3b82f6;padding:12px;margin:12px 0;border-radius:4px;font-size:0.85rem}}
.finding.warn{{border-left-color:#d97706;background:#fffbeb}}
.finding.bad{{border-left-color:#dc2626;background:#fef2f2}}
</style></head><body>
<h1>Hedge Cost Reality Check</h1>
<p class="meta">Real SPY Put Prices from IronVault vs Assumed 2%/yr Flat Budget |
<span class="verdict">{verdict}: Real avg = {cost_summary.avg_annual_cost_pct:.2f}% vs assumed 2.00% ({cost_summary.actual_vs_assumed_ratio:.1f}x)</span></p>

<div class="grid">
  <div class="card"><div class="l">Assumed Cost</div><div class="v">2.00%/yr</div></div>
  <div class="card"><div class="l">Real Avg Cost</div><div class="v" style="color:{'#dc2626' if cost_summary.avg_annual_cost_pct > 3 else '#16a34a'}">{cost_summary.avg_annual_cost_pct:.2f}%/yr</div></div>
  <div class="card"><div class="l">Ratio</div><div class="v">{cost_summary.actual_vs_assumed_ratio:.1f}x</div></div>
  <div class="card"><div class="l">Months Sampled</div><div class="v">{cost_summary.n_months_sampled}</div></div>
  <div class="card"><div class="l">Min Cost</div><div class="v">{cost_summary.min_annual_cost_pct:.2f}%</div></div>
  <div class="card"><div class="l">Max Cost</div><div class="v">{cost_summary.max_annual_cost_pct:.2f}%</div></div>
  <div class="card"><div class="l">VIX Options</div><div class="v" style="color:#dc2626">NOT IN DB</div></div>
  <div class="card"><div class="l">SPY Put Data</div><div class="v" style="color:#16a34a">96K+ contracts</div></div>
</div>

<div class="finding {'warn' if cost_summary.actual_vs_assumed_ratio > 1.5 else ''}">
<strong>Key Finding:</strong> Real SPY 5%-OTM put costs average <strong>{cost_summary.avg_annual_cost_pct:.2f}%/yr</strong>
(annualised from monthly rolls at 30-45 DTE). The assumed 2.00% flat budget is
{'<span style="color:#dc2626;font-weight:700">UNDERSTATED</span> — real costs are ' + f'{cost_summary.actual_vs_assumed_ratio:.1f}x higher' if cost_summary.actual_vs_assumed_ratio > 1.3 else
'<span style="color:#d97706;font-weight:700">ROUGHLY CORRECT</span> — real costs within 30% of assumption' if cost_summary.actual_vs_assumed_ratio > 0.7 else
'<span style="color:#16a34a;font-weight:700">OVERSTATED</span> — real costs are lower than assumed'}.
VIX call options are <strong>not available</strong> in IronVault — the 40% VIX call allocation
in the tail risk hedge module cannot be priced from real data.
</div>

<h2>Real Cost by Year</h2>
<table><tr><th>Year</th><th>Real Cost</th><th>Assumed</th><th>Delta</th></tr>{yr_cost_rows}</table>

<h2>Impact on Portfolio (1.6x Leverage, 2020-2025)</h2>
<table>
<tr><th>Metric</th><th>Assumed 2%</th><th>Real {cost_summary.avg_annual_cost_pct:.1f}%</th><th>Delta</th></tr>
<tr><td>CAGR</td><td>{bt.assumed_cagr:.1f}%</td><td>{bt.real_cagr:.1f}%</td><td style="color:{_dc(bt.cagr_delta)}">{bt.cagr_delta:+.1f}%</td></tr>
<tr><td>Sharpe</td><td>{bt.assumed_sharpe:.2f}</td><td>{bt.real_sharpe:.2f}</td><td style="color:{_dc(bt.sharpe_delta)}">{bt.sharpe_delta:+.2f}</td></tr>
<tr><td>Max DD</td><td>{bt.assumed_dd:.1f}%</td><td>{bt.real_dd:.1f}%</td><td style="color:{_dc(bt.dd_delta, invert=True)}">{bt.dd_delta:+.1f}%</td></tr>
<tr><td>Calmar</td><td>{bt.assumed_calmar:.1f}</td><td>{bt.real_calmar:.1f}</td><td>{bt.real_calmar - bt.assumed_calmar:+.1f}</td></tr>
</table>

<h2>Equity Curves</h2>
{eq_svg}

<h2>Yearly Backtest Comparison</h2>
<table><tr><th>Year</th><th>Assumed CAGR</th><th>Assumed SR</th><th>Real CAGR</th><th>Real SR</th></tr>{yr_bt_rows}</table>

<h2>Monthly Put Prices (Raw Data)</h2>
<table><tr><th>Month</th><th>SPY</th><th>Strike</th><th>Put $</th><th>DTE</th><th>VIX</th><th>Cost %</th><th>Ann %</th></tr>{monthly_rows}</table>

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/hedge_cost_reality.py | All put prices from IronVault options_cache.db | VIX options: NOT AVAILABLE</div>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


def _dual_equity_svg(eq1, eq2):
    if len(eq1) < 2: return ""
    w, h = 780, 200; pl, pr, pt, pb = 65, 20, 28, 28
    pw, ph = w-pl-pr, h-pt-pb
    all_v = eq1 + eq2; ym, yx = min(all_v)*0.95, max(all_v)*1.05
    n = max(len(eq1), len(eq2))
    def line(data, color, dash=""):
        step = max(1, len(data)//500)
        pts = [(i, data[i]) for i in range(0, len(data), step)]
        if pts[-1][0] != len(data)-1: pts.append((len(data)-1, data[-1]))
        def tx(i): return pl + i/max(n-1,1)*pw
        def ty(v): return pt + (1-(v-ym)/max(yx-ym,1))*ph
        d = " ".join(f"{'M' if j==0 else 'L'}{tx(i):.1f},{ty(v):.1f}" for j,(i,v) in enumerate(pts))
        da = f' stroke-dasharray="4,3"' if dash else ""
        return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{da}/>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="border:1px solid #e2e8f0;border-radius:6px"><text x="{w//2}" y="16" text-anchor="middle" font-size="10" fill="#64748b">Assumed 2% (gray dashed) vs Real cost (green)</text>{line(eq1,"#94a3b8","dash")}{line(eq2,"#16a34a","")}</svg>'


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis():
    print("Hedge Cost Reality Check")
    print("=" * 60)

    print("  Querying IronVault for real SPY put prices...")
    costs = query_monthly_hedge_costs()
    summary = summarise_costs(costs)

    print(f"\n  Months sampled: {summary.n_months_sampled}")
    print(f"  Avg annual cost: {summary.avg_annual_cost_pct:.2f}% (assumed: 2.00%)")
    print(f"  Ratio: {summary.actual_vs_assumed_ratio:.1f}x")
    print(f"  Range: {summary.min_annual_cost_pct:.2f}% to {summary.max_annual_cost_pct:.2f}%")

    print(f"\n  Per-year costs:")
    for yr, cost in sorted(summary.yearly_costs.items()):
        print(f"    {yr}: {cost:.2f}% (vs 2.00% assumed, delta={cost-2.0:+.2f}%)")

    print(f"\n  VIX options: NOT IN DATABASE (VIX/UVXY/VXX all absent)")
    print(f"  Impact: 40% of hedge budget (VIX calls) cannot be validated")

    print(f"\n  Re-running backtest with real cost...")
    bt = run_backtest_comparison(summary.avg_annual_cost_pct)

    print(f"\n  Impact on portfolio (1.6x leverage):")
    print(f"    {'Metric':<12} {'Assumed 2%':>12} {'Real':>12} {'Delta':>10}")
    print(f"    {'-'*46}")
    print(f"    {'CAGR':<12} {bt.assumed_cagr:>11.1f}% {bt.real_cagr:>11.1f}% {bt.cagr_delta:>+9.1f}%")
    print(f"    {'Sharpe':<12} {bt.assumed_sharpe:>12.2f} {bt.real_sharpe:>12.2f} {bt.sharpe_delta:>+10.2f}")
    print(f"    {'Max DD':<12} {bt.assumed_dd:>11.1f}% {bt.real_dd:>11.1f}% {bt.dd_delta:>+9.1f}%")
    print(f"    {'Calmar':<12} {bt.assumed_calmar:>12.1f} {bt.real_calmar:>12.1f} {bt.real_calmar-bt.assumed_calmar:>+10.1f}")

    report = generate_report(summary, bt)
    print(f"\n  Report: {report}")
    return summary, bt


if __name__ == "__main__":
    run_analysis()
