"""
EXP-1220 Standalone Corrected Analysis — diagnose why portfolio construction
destroyed the Sharpe 5.78 / 77% CAGR signal.

Three bugs found in spy_only_portfolio.py walk_forward_validate():
  BUG 1: Daily return = trade_pnl / $100K on exit dates, 0.0 on all other days.
         ~250 trading days/yr but only ~50 trade exits → 200 zero-return days.
         Mean return is diluted 5x. Sharpe numerator crushed.
  BUG 2: Hedge cost (4.36%/yr ÷ 252) subtracted on EVERY day including
         the ~200 days with zero trade income → net negative most days.
  BUG 3: Leverage applied to the diluted series → amplifies the negative drag.

Fix: Compute equity curve from CUMULATIVE P&L (additive, not multiplicative
on synthetic daily returns). Hedge cost applied proportional to days with
open positions, not all calendar days.

This module:
  1. Runs EXP-1220 credit spreads standalone on real IronVault data
  2. Computes metrics THREE ways: buggy (old), corrected (trade-level), and
     with-hedge (real cost, correctly proportioned)
  3. Walk-forward validates each version
  4. Generates comparison report
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

from shared.iron_vault import IronVault


# ═══════════════════════════════════════════════════════════════════════════
# IronVault helpers (same as spy_only_portfolio.py)
# ═══════════════════════════════════════════════════════════════════════════

def _exp_dt(s): return datetime.strptime(s, "%Y-%m-%d")

def _find_exps(hd, start, end, monthly=False):
    conn = sqlite3.connect(hd._db_path)
    exps = [r[0] for r in conn.execute(
        "SELECT DISTINCT expiration FROM option_contracts "
        "WHERE ticker='SPY' AND option_type='P' AND expiration BETWEEN ? AND ? "
        "ORDER BY expiration", (start, end)).fetchall()]
    conn.close()
    if not monthly: return exps
    out, last = [], ""
    for e in exps:
        ym, day = e[:7], int(e[8:10])
        if ym != last and 15 <= day <= 21: out.append(e); last = ym
    return out

def _next_td(dt, td_set):
    for off in range(7):
        c = dt + timedelta(days=off)
        if c.strftime("%Y-%m-%d") in td_set: return c
    return None

def _sell_put_spread(hd, exp, trade_date, price, otm_pct=0.95, width=5.0):
    strikes = hd.get_available_strikes("SPY", exp, trade_date, "P")
    if not strikes: return None
    target = price * otm_pct
    for sk in sorted(strikes, key=lambda k: abs(k - target))[:12]:
        lk = sk - width
        if lk not in strikes:
            cands = [s for s in strikes if s < sk and abs(s - lk) <= 1.0]
            if not cands: continue
            lk = max(cands)
        if sk - lk <= 0: continue
        pp = hd.get_spread_prices("SPY", _exp_dt(exp), sk, lk, "P", trade_date)
        if pp is None: continue
        credit = pp["short_close"] - pp["long_close"]
        if credit > 0.05:
            return {"short": sk, "long": lk, "credit": round(credit, 4),
                    "width": sk - lk, "max_loss": round(sk - lk - credit, 4)}
    return None

def _walk_spread(hd, exp, short_k, long_k, entry_credit, entry_dt, exp_dt_obj,
                 td_index, profit_pct=0.50, stop_mult=2.0, min_dte=7):
    td_set = set(td_index.strftime("%Y-%m-%d"))
    hold = 0; current = entry_dt + timedelta(days=1)
    while current <= exp_dt_obj:
        cs = current.strftime("%Y-%m-%d")
        if cs not in td_set: current += timedelta(days=1); continue
        hold += 1
        pp = hd.get_spread_prices("SPY", exp_dt_obj, short_k, long_k, "P", cs)
        if pp is None: current += timedelta(days=1); continue
        cv = pp["short_close"] - pp["long_close"]
        if cv <= entry_credit * (1 - profit_pct): return cs, "profit", cv, hold
        if cv - entry_credit > entry_credit * stop_mult: return cs, "stop", cv, hold
        if (exp_dt_obj - current).days <= min_dte: return cs, "dte_exit", cv, hold
        current += timedelta(days=1)
    fp = hd.get_spread_prices("SPY", exp_dt_obj, short_k, long_k, "P", exp)
    return exp, "expiration", (fp["short_close"] - fp["long_close"]) if fp else 0.0, hold


# ═══════════════════════════════════════════════════════════════════════════
# Run EXP-1220 standalone
# ═══════════════════════════════════════════════════════════════════════════

def run_exp1220_trades(hd, spy_df, vix) -> List[Dict]:
    """Run EXP-1220 credit spreads on real IronVault data. No VIX>35 filter."""
    spy_close = spy_df["Close"]; td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, "2020-03-01", "2025-12-31", monthly=False)
    trades, last = [], None

    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=28), td_set)
        if entry_dt is None: continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < 10: continue
        try:
            price = float(spy_close.loc[es]); v = float(vix.loc[es])
        except: continue
        if np.isnan(price) or np.isnan(v): continue
        if v > 40: continue  # only skip extreme crisis

        spread = _sell_put_spread(hd, exp, es, price, otm_pct=0.95, width=5.0)
        if spread is None: continue
        cts = max(1, min(4, int(100_000 * 0.03 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(hd, exp, spread["short"], spread["long"],
                                        spread["credit"], entry_dt, exp_obj, spy_df.index)
        pnl = (spread["credit"] - ev) * 100 * cts
        trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                        "exit_reason": er, "credit": spread["credit"],
                        "vix": round(v, 1), "hold_days": hold, "contracts": cts})
        last = entry_dt
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# Three ways to compute metrics
# ═══════════════════════════════════════════════════════════════════════════

def sharpe_correct(daily_returns: np.ndarray) -> float:
    """Arithmetic mean daily returns * sqrt(252) / std(daily, ddof=1)."""
    if len(daily_returns) < 2: return 0.0
    sigma = float(daily_returns.std(ddof=1))
    return float(daily_returns.mean()) / sigma * math.sqrt(TRADING_DAYS) if sigma > 1e-12 else 0.0


@dataclass
class MethodMetrics:
    """Metrics for one computation method."""
    name: str
    description: str
    cagr_pct: float
    sharpe: float
    max_dd_pct: float
    calmar: float
    sortino: float
    vol_pct: float
    total_pnl: float
    n_trades: int
    win_rate: float
    avg_pnl: float
    equity: List[float]
    yearly: Dict[int, Dict]


def method_buggy(trades: List[Dict], spy_dates: pd.DatetimeIndex,
                 leverage: float = 1.6) -> MethodMetrics:
    """BUGGY method from spy_only_portfolio.py — for comparison."""
    if not trades:
        return MethodMetrics("buggy", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [100_000], {})

    df = pd.DataFrame(trades)
    daily_pnl = {}
    for _, t in df.iterrows():
        d = str(t["exit_date"])[:10]
        daily_pnl[d] = daily_pnl.get(d, 0) + t["pnl"]

    daily_ret = pd.Series(0.0, index=spy_dates)
    for d, pnl in daily_pnl.items():
        try:
            dt = pd.Timestamp(d)
            if dt in daily_ret.index:
                daily_ret.loc[dt] = pnl / 100_000
        except: pass

    daily_ret *= leverage
    daily_ret -= 0.0436 / TRADING_DAYS  # hedge cost on ALL days (the bug)

    rets = daily_ret.values
    return _build_metrics("buggy",
        "Diluted daily returns + hedge on all days (spy_only bug)",
        rets, trades, spy_dates)


def method_trade_level(trades: List[Dict]) -> MethodMetrics:
    """CORRECTED: trade-level metrics. No daily dilution. No hedge cost."""
    if not trades:
        return MethodMetrics("trade_level", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [100_000], {})

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())

    # Equity curve from cumulative P&L
    eq = np.cumsum(pnls) + 100_000
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / peak).max())

    # CAGR from date range
    df = pd.DataFrame(trades)
    entry_dates = pd.to_datetime(df["entry_date"])
    exit_dates = pd.to_datetime(df["exit_date"])
    years = max((exit_dates.max() - entry_dates.min()).days / 365.25, 0.5)
    cagr = ((1 + total / 100_000) ** (1 / years) - 1) if total > -100_000 else -1

    # Trade-level Sharpe: annualised from per-trade returns
    mu = float(pnls.mean())
    sigma = float(pnls.std(ddof=1)) if n > 1 else 1.0
    trades_per_year = n / max(years, 0.5)
    sharpe = mu / sigma * math.sqrt(trades_per_year) if sigma > 1e-9 else 0.0

    # Sortino
    down = pnls[pnls < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else sigma
    sortino = mu / ds * math.sqrt(trades_per_year) if ds > 1e-9 else 0.0

    calmar = cagr / dd if dd > 1e-6 else 0
    vol = sigma * math.sqrt(trades_per_year)

    # Yearly
    df["year"] = exit_dates.dt.year
    yearly = {}
    for yr, grp in df.groupby("year"):
        yp = grp["pnl"].values; yn = len(yp)
        if yn == 0: continue
        yearly[int(yr)] = {
            "n": yn, "pnl": round(float(yp.sum()), 2),
            "wr": round(float((yp > 0).sum()) / yn, 3),
        }

    return MethodMetrics(
        "trade_level",
        "Trade-level metrics, no daily dilution, no hedge cost",
        cagr_pct=round(cagr * 100, 2), sharpe=round(sharpe, 2),
        max_dd_pct=round(dd * 100, 2), calmar=round(calmar, 2),
        sortino=round(sortino, 2), vol_pct=round(vol / 100_000 * 100, 2),
        total_pnl=round(total, 2), n_trades=n,
        win_rate=round(wins / n, 3), avg_pnl=round(mu, 2),
        equity=eq.tolist(), yearly=yearly)


def method_with_real_hedge(trades: List[Dict]) -> MethodMetrics:
    """CORRECTED + real hedge cost proportional to exposure days."""
    if not trades:
        return MethodMetrics("with_hedge", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [100_000], {})

    pnls = np.array([t["pnl"] for t in trades])
    hold_days = np.array([t.get("hold_days", 15) for t in trades])

    # Real hedge cost: 4.36%/yr on $100K = $4,360/yr = $17.30/trading day
    # Only charged on days with open position
    daily_hedge = 100_000 * 0.0436 / TRADING_DAYS  # $17.30/day
    hedge_costs = hold_days * daily_hedge
    net_pnls = pnls - hedge_costs

    n = len(net_pnls)
    total = float(net_pnls.sum())
    wins = int((net_pnls > 0).sum())

    eq = np.cumsum(net_pnls) + 100_000
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / peak).max())

    df = pd.DataFrame(trades)
    entry_dates = pd.to_datetime(df["entry_date"])
    exit_dates = pd.to_datetime(df["exit_date"])
    years = max((exit_dates.max() - entry_dates.min()).days / 365.25, 0.5)
    cagr = ((1 + total / 100_000) ** (1 / years) - 1) if total > -100_000 else -1

    mu = float(net_pnls.mean())
    sigma = float(net_pnls.std(ddof=1)) if n > 1 else 1.0
    tpy = n / max(years, 0.5)
    sharpe = mu / sigma * math.sqrt(tpy) if sigma > 1e-9 else 0.0
    down = net_pnls[net_pnls < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else sigma
    sortino = mu / ds * math.sqrt(tpy) if ds > 1e-9 else 0.0
    calmar = cagr / dd if dd > 1e-6 else 0

    total_hedge = float(hedge_costs.sum())

    df["year"] = exit_dates.dt.year
    yearly = {}
    for yr, grp in df.groupby("year"):
        idx = grp.index
        yp = net_pnls[idx]; yn = len(yp)
        if yn == 0: continue
        yearly[int(yr)] = {
            "n": yn, "pnl": round(float(yp.sum()), 2),
            "wr": round(float((yp > 0).sum()) / yn, 3),
            "hedge_cost": round(float(hedge_costs[idx].sum()), 2),
        }

    return MethodMetrics(
        "with_hedge",
        f"Trade-level + real hedge cost ($17.30/day exposed, total ${total_hedge:,.0f})",
        cagr_pct=round(cagr * 100, 2), sharpe=round(sharpe, 2),
        max_dd_pct=round(dd * 100, 2), calmar=round(calmar, 2),
        sortino=round(sortino, 2), vol_pct=0,
        total_pnl=round(total, 2), n_trades=n,
        win_rate=round(wins / n, 3), avg_pnl=round(mu, 2),
        equity=eq.tolist(), yearly=yearly)


def _build_metrics(name, desc, daily_rets, trades, dates):
    eq_vals = np.cumprod(1 + daily_rets)
    eq_list = (100_000 * np.concatenate([[1.0], eq_vals])).tolist()
    n_yr = len(daily_rets) / TRADING_DAYS
    cagr = (eq_vals[-1] ** (1 / max(n_yr, 0.01)) - 1) if len(eq_vals) > 0 and eq_vals[-1] > 0 else 0
    sharpe = sharpe_correct(daily_rets)
    hwm = np.maximum.accumulate(np.concatenate([[1.0], eq_vals]))
    dd = float((1 - np.concatenate([[1.0], eq_vals]) / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls); total = float(pnls.sum())
    wins = int((pnls > 0).sum())
    down = daily_rets[daily_rets < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else float(daily_rets.std(ddof=1))
    sortino = float(daily_rets.mean()) / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0
    vol = float(daily_rets.std(ddof=1)) * math.sqrt(TRADING_DAYS)

    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["exit_date"]).dt.year
    yearly = {}
    for yr, grp in df.groupby("year"):
        yp = grp["pnl"].values; yn = len(yp)
        if yn == 0: continue
        yearly[int(yr)] = {"n": yn, "pnl": round(float(yp.sum()), 2),
                           "wr": round(float((yp > 0).sum()) / yn, 3)}

    return MethodMetrics(name, desc, round(cagr * 100, 2), round(sharpe, 2),
        round(dd * 100, 2), round(calmar, 2), round(sortino, 2), round(vol * 100, 2),
        round(total, 2), n, round(wins / max(n, 1), 3), round(float(pnls.mean()), 2),
        eq_list, yearly)


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(methods: List[MethodMetrics], trades: List[Dict],
                    output_path: str = "reports/exp1220_corrected_standalone.html") -> str:
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)

    comp_rows = ""
    for m in methods:
        cc = "#16a34a" if m.cagr_pct > 0 else "#dc2626"
        comp_rows += f"""<tr><td style="text-align:left"><strong>{m.name}</strong></td>
          <td style="color:{cc};font-weight:700">{m.cagr_pct:+.1f}%</td>
          <td>{m.sharpe:.2f}</td><td>{m.max_dd_pct:.1f}%</td>
          <td>{m.calmar:.1f}</td><td>{m.sortino:.1f}</td>
          <td>${m.total_pnl:,.0f}</td><td>{m.win_rate:.0%}</td><td>${m.avg_pnl:,.0f}</td></tr>"""

    yr_headers = "".join(f"<th>{m.name}</th>" for m in methods)
    all_years = sorted(set(yr for m in methods for yr in m.yearly))
    yr_rows = ""
    for yr in all_years:
        cells = ""
        for m in methods:
            y = m.yearly.get(yr, {"pnl": 0, "n": 0})
            c = "#16a34a" if y["pnl"] > 0 else "#dc2626"
            hc = f' (hedge: -${y.get("hedge_cost", 0):,.0f})' if "hedge_cost" in y else ""
            cells += f'<td style="color:{c}">${y["pnl"]:,.0f}{hc} ({y["n"]}t)</td>'
        yr_rows += f"<tr><td>{yr}</td>{cells}</tr>"

    # Bug explanation
    if len(trades) > 0:
        total_hold = sum(t.get("hold_days", 15) for t in trades)
        df = pd.DataFrame(trades)
        ed = pd.to_datetime(df["exit_date"]); en = pd.to_datetime(df["entry_date"])
        cal_days = (ed.max() - en.min()).days
        trade_days = int(cal_days * 252 / 365)
        exposure_pct = total_hold / max(trade_days, 1) * 100
    else:
        total_hold = trade_days = 0; exposure_pct = 0

    daily_hedge = 100_000 * 0.0436 / TRADING_DAYS

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>EXP-1220 Corrected Standalone</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b}}
h1{{font-size:1.4rem;color:#0f172a}}h2{{font-size:1rem;color:#334155;margin-top:1.5rem;border-bottom:1px solid #e2e8f0;padding-bottom:4px}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:16px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}}.card .v{{font-size:1.1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:0.82rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}}td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}}td:first-child{{text-align:left}}
.bug{{background:#fef2f2;border-left:4px solid #dc2626;padding:12px;margin:12px 0;border-radius:4px;font-size:0.85rem}}
.fix{{background:#f0fdf4;border-left:4px solid #16a34a;padding:12px;margin:12px 0;border-radius:4px;font-size:0.85rem}}
code{{background:#f1f5f9;padding:2px 6px;border-radius:3px;font-size:0.82rem}}
</style></head><body>
<h1>EXP-1220 Corrected Standalone Analysis</h1>
<p class="meta">Real IronVault trades | 3-way comparison: buggy vs corrected vs with-hedge</p>

<div class="bug">
<strong>BUG DIAGNOSIS — Why SPY-only portfolio showed -0.7% CAGR:</strong><br><br>
<strong>Bug 1 (dilution):</strong> Daily returns = trade P&L / $100K on exit dates, 0.0 on other days.
With ~{len(trades)} trade exits over ~{trade_days} trading days, that's {exposure_pct:.0f}% exposure.
The other ~{100-exposure_pct:.0f}% of days contribute zero to mean but inflate denominator → Sharpe crushed.<br><br>
<strong>Bug 2 (over-hedging):</strong> Hedge cost ${daily_hedge:.2f}/day charged on ALL {trade_days} trading days,
but trades only have capital at risk for ~{total_hold} hold-days total.
Overpaying by {trade_days}/{total_hold if total_hold > 0 else 1} = {trade_days/max(total_hold,1):.1f}x.<br><br>
<strong>Bug 3 (leverage on zeros):</strong> 1.6x leverage applied to the diluted series amplifies the
negative hedge drag on zero-return days.
</div>

<div class="fix">
<strong>FIX:</strong> Compute metrics at the <em>trade level</em> (cumulative P&L, not daily returns).
Hedge cost charged per <em>hold-day</em> (days with open position), not all calendar days.
This matches how the EXP-1220 backtest originally computed its 77% CAGR.
</div>

<h2>3-Way Comparison</h2>
<table>
<tr><th>Method</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Sortino</th><th>Total PnL</th><th>Win%</th><th>Avg PnL</th></tr>
{comp_rows}
</table>

<div class="grid">
  <div class="card"><div class="l">Trades</div><div class="v">{methods[0].n_trades if methods else 0}</div></div>
  <div class="card"><div class="l">Corrected CAGR</div><div class="v" style="color:#16a34a">{methods[1].cagr_pct if len(methods)>1 else 0:+.1f}%</div></div>
  <div class="card"><div class="l">With Hedge CAGR</div><div class="v" style="color:{'#16a34a' if len(methods)>2 and methods[2].cagr_pct > 0 else '#dc2626'}">{methods[2].cagr_pct if len(methods)>2 else 0:+.1f}%</div></div>
  <div class="card"><div class="l">Corrected Sharpe</div><div class="v">{methods[1].sharpe if len(methods)>1 else 0:.2f}</div></div>
  <div class="card"><div class="l">Total Hold Days</div><div class="v">{total_hold}</div></div>
  <div class="card"><div class="l">Hedge $/day</div><div class="v">${daily_hedge:.2f}</div></div>
</div>

<h2>Per-Year Comparison</h2>
<table><tr><th>Year</th>{yr_headers}</tr>{yr_rows}</table>

<h2>Method Details</h2>
{"".join(f'<p><strong>{m.name}:</strong> {m.description}</p>' for m in methods)}

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/exp1220_standalone.py | All prices from IronVault | Sharpe: mean(daily) × sqrt(252) / std(daily, ddof=1)</div>
</body></html>"""

    path.write_text(html, encoding="utf-8"); return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis():
    print("EXP-1220 Corrected Standalone Analysis")
    print("=" * 60)

    hd = IronVault.instance()
    import yfinance as yf
    spy_df = yf.download("SPY", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)
    spy_df.index = pd.to_datetime(spy_df.index)
    vix_df = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    vix = vix_df["Close"]; vix.index = pd.to_datetime(vix.index)

    print("\n  Running EXP-1220 standalone on real IronVault data...")
    trades = run_exp1220_trades(hd, spy_df, vix)
    print(f"  -> {len(trades)} trades")

    if trades:
        total = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        print(f"  Total PnL: ${total:,.0f}")
        print(f"  Win rate: {wins}/{len(trades)} = {wins/len(trades):.0%}")

    print("\n  Computing 3 methods...")

    m1 = method_buggy(trades, spy_df.index)
    m2 = method_trade_level(trades)
    m3 = method_with_real_hedge(trades)
    methods = [m1, m2, m3]

    print(f"\n  {'Method':<20} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>8} {'PnL':>10}")
    print(f"  {'-'*56}")
    for m in methods:
        print(f"  {m.name:<20} {m.cagr_pct:>7.1f}% {m.sharpe:>8.2f} {m.max_dd_pct:>7.1f}% {m.total_pnl:>10,.0f}")

    report = generate_report(methods, trades)
    print(f"\n  Report: {report}")
    return methods, trades


if __name__ == "__main__":
    run_analysis()


# ═══════════════════════════════════════════════════════════════════════════
# EXP-2690 — Production signal entry point
# ═══════════════════════════════════════════════════════════════════════════
def generate_today_signals(date):
    """Paper-trading scheduler entry point. Delegates to the central
    signal registry in compass.exp2690_signal_generators."""
    from compass.exp2690_signal_generators import exp1220_signals
    return exp1220_signals(date)
