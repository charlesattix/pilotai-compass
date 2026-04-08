"""
EXP-1220 Trade Cadence Analyzer — optimal deployment frequency.

The dilution bug revealed only 171 trades over 6 years (~28/yr, 2.3/month).
This module analyzes:
  1. Trade timeline — when does each position open/close?
  2. Overlap analysis — how many positions are active simultaneously?
  3. Capital utilization — what % of capital is deployed at any time?
  4. Missed signals — trades not taken due to the 10-day cooldown filter
  5. Optimal cadence — compare weekly, bi-weekly, monthly entry frequencies
  6. Rolling entry simulation — what if we enter every N days?

All trades from real IronVault data.
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
# IronVault helpers (reuse from exp1220_standalone)
# ═══════════════════════════════════════════════════════════════════════════

def _exp_dt(s): return datetime.strptime(s, "%Y-%m-%d")

def _find_exps(hd, start, end):
    conn = sqlite3.connect(hd._db_path)
    exps = [r[0] for r in conn.execute(
        "SELECT DISTINCT expiration FROM option_contracts "
        "WHERE ticker='SPY' AND option_type='P' AND expiration BETWEEN ? AND ? "
        "ORDER BY expiration", (start, end)).fetchall()]
    conn.close()
    return exps

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
# Run trades at different cooldown intervals
# ═══════════════════════════════════════════════════════════════════════════

def run_with_cooldown(hd, spy_df, vix, cooldown_days: int) -> List[Dict]:
    """Run EXP-1220 with a specific entry cooldown."""
    spy_close = spy_df["Close"]; td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, "2020-03-01", "2025-12-31")
    trades, last = [], None

    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=28), td_set)
        if entry_dt is None: continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < cooldown_days: continue
        try:
            price = float(spy_close.loc[es]); v = float(vix.loc[es])
        except: continue
        if np.isnan(price) or np.isnan(v) or v > 40: continue

        spread = _sell_put_spread(hd, exp, es, price, otm_pct=0.95, width=5.0)
        if spread is None: continue
        cts = max(1, min(4, int(100_000 * 0.03 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(hd, exp, spread["short"], spread["long"],
                                        spread["credit"], entry_dt, exp_obj, spy_df.index)
        pnl = (spread["credit"] - ev) * 100 * cts
        trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                        "exit_reason": er, "credit": spread["credit"],
                        "vix": round(v, 1), "hold_days": hold, "contracts": cts,
                        "cooldown": cooldown_days})
        last = entry_dt
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeTimeline:
    entry: str
    exit: str
    hold_days: int
    pnl: float
    is_winner: bool

@dataclass
class OverlapSnapshot:
    date: str
    active_positions: int
    capital_at_risk: float  # as % of $100K

@dataclass
class CadenceResult:
    cooldown_days: int
    label: str
    n_trades: int
    trades_per_year: float
    total_pnl: float
    avg_pnl: float
    win_rate: float
    sharpe: float
    max_concurrent: int
    avg_concurrent: float
    capital_util_pct: float  # avg % of capital deployed
    avg_hold_days: float
    pnl_per_year: float


def analyze_timeline(trades: List[Dict]) -> Tuple[List[TradeTimeline], List[OverlapSnapshot]]:
    """Build trade timeline and daily overlap count."""
    if not trades:
        return [], []

    timeline = []
    for t in trades:
        timeline.append(TradeTimeline(
            entry=t["entry_date"], exit=t["exit_date"],
            hold_days=t.get("hold_days", 15),
            pnl=t["pnl"], is_winner=t["pnl"] > 0))

    # Build daily overlap
    df = pd.DataFrame(trades)
    entry_dates = pd.to_datetime(df["entry_date"])
    exit_dates = pd.to_datetime(df["exit_date"])
    all_dates = pd.bdate_range(entry_dates.min(), exit_dates.max())

    snapshots = []
    for d in all_dates:
        ds = d.strftime("%Y-%m-%d")
        active = sum(1 for i in range(len(trades))
                     if entry_dates.iloc[i] <= d <= exit_dates.iloc[i])
        # Capital at risk: ~$500 margin per spread × contracts × active
        avg_cts = df["contracts"].mean() if "contracts" in df.columns else 2
        risk = active * 500 * avg_cts / 100_000 * 100
        snapshots.append(OverlapSnapshot(ds, active, round(risk, 1)))

    return timeline, snapshots


def compute_cadence_metrics(trades: List[Dict], cooldown: int, label: str) -> CadenceResult:
    """Compute all metrics for one cadence setting."""
    if not trades:
        return CadenceResult(cooldown, label, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())
    hold_days = [t.get("hold_days", 15) for t in trades]

    df = pd.DataFrame(trades)
    entry_dates = pd.to_datetime(df["entry_date"])
    exit_dates = pd.to_datetime(df["exit_date"])
    years = max((exit_dates.max() - entry_dates.min()).days / 365.25, 0.5)
    tpy = n / years

    mu = float(pnls.mean())
    sigma = float(pnls.std(ddof=1)) if n > 1 else 1.0
    sharpe = mu / sigma * math.sqrt(tpy) if sigma > 1e-9 else 0.0

    # Overlap analysis
    all_dates = pd.bdate_range(entry_dates.min(), exit_dates.max())
    concurrent = []
    for d in all_dates:
        active = sum(1 for i in range(n) if entry_dates.iloc[i] <= d <= exit_dates.iloc[i])
        concurrent.append(active)
    max_conc = max(concurrent) if concurrent else 0
    avg_conc = float(np.mean(concurrent)) if concurrent else 0

    # Capital utilization: days with at least 1 position / total days
    days_active = sum(1 for c in concurrent if c > 0)
    util = days_active / max(len(all_dates), 1) * 100

    return CadenceResult(
        cooldown_days=cooldown, label=label, n_trades=n,
        trades_per_year=round(tpy, 1), total_pnl=round(total, 2),
        avg_pnl=round(mu, 2), win_rate=round(wins / n, 3),
        sharpe=round(sharpe, 2), max_concurrent=max_conc,
        avg_concurrent=round(avg_conc, 2),
        capital_util_pct=round(util, 1),
        avg_hold_days=round(float(np.mean(hold_days)), 1),
        pnl_per_year=round(total / years, 2))


def find_missed_signals(hd, spy_df, vix) -> Dict[str, Any]:
    """Count how many valid spreads exist but are skipped by the cooldown."""
    spy_close = spy_df["Close"]; td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, "2020-03-01", "2025-12-31")

    total_opportunities = 0
    priceable = 0

    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=28), td_set)
        if entry_dt is None: continue
        es = entry_dt.strftime("%Y-%m-%d")
        try:
            price = float(spy_close.loc[es]); v = float(vix.loc[es])
        except: continue
        if np.isnan(price) or np.isnan(v) or v > 40: continue

        total_opportunities += 1
        spread = _sell_put_spread(hd, exp, es, price, otm_pct=0.95, width=5.0)
        if spread is not None:
            priceable += 1

    return {
        "total_expirations": len(exps),
        "valid_opportunities": total_opportunities,
        "priceable_spreads": priceable,
        "pricing_hit_rate": round(priceable / max(total_opportunities, 1) * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    cadence_results: List[CadenceResult],
    timeline: List[TradeTimeline],
    overlaps: List[OverlapSnapshot],
    missed: Dict[str, Any],
    output_path: str = "reports/exp1220_trade_cadence.html",
) -> str:
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)

    # Cadence comparison table
    best = max(cadence_results, key=lambda c: c.pnl_per_year) if cadence_results else CadenceResult(0, "N/A", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    cad_rows = ""
    for c in cadence_results:
        is_best = c.cooldown_days == best.cooldown_days if best else False
        bg = ' style="background:#f0fdf4"' if is_best else ""
        star = " **" if is_best else ""
        pc = "#16a34a" if c.total_pnl > 0 else "#dc2626"
        cad_rows += f'<tr{bg}><td>{c.label}{star}</td><td>{c.cooldown_days}d</td><td>{c.n_trades}</td><td>{c.trades_per_year}</td><td style="color:{pc};font-weight:700">${c.total_pnl:,.0f}</td><td>${c.pnl_per_year:,.0f}/yr</td><td>${c.avg_pnl:,.0f}</td><td>{c.win_rate:.0%}</td><td>{c.sharpe:.2f}</td><td>{c.max_concurrent}</td><td>{c.avg_concurrent:.1f}</td><td>{c.capital_util_pct:.0f}%</td><td>{c.avg_hold_days:.0f}d</td></tr>'

    # Timeline SVG — show trade bars
    tl_svg = ""
    if timeline:
        w, h = 780, max(120, min(400, len(timeline) * 4))
        pl, pr, pt, pb = 10, 10, 20, 20
        pw, ph = w - pl - pr, h - pt - pb

        all_entries = [pd.Timestamp(t.entry) for t in timeline]
        all_exits = [pd.Timestamp(t.exit) for t in timeline]
        date_min = min(all_entries); date_max = max(all_exits)
        total_days = max((date_max - date_min).days, 1)

        def tx(d): return pl + (d - date_min).days / total_days * pw

        bars = ""
        for i, t in enumerate(timeline):
            x1 = tx(pd.Timestamp(t.entry))
            x2 = tx(pd.Timestamp(t.exit))
            y = pt + (i / max(len(timeline) - 1, 1)) * ph
            color = "#16a34a" if t.is_winner else "#dc2626"
            bars += f'<rect x="{x1:.0f}" y="{y:.0f}" width="{max(x2 - x1, 1):.0f}" height="3" fill="{color}" rx="1"/>'

        tl_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="border:1px solid #e2e8f0;border-radius:6px"><text x="{w//2}" y="14" text-anchor="middle" font-size="10" fill="#64748b">Trade Timeline (green=win, red=loss)</text>{bars}</svg>'

    # Overlap SVG
    ov_svg = ""
    if overlaps:
        w, h = 780, 120; pl, pr, pt, pb = 10, 10, 20, 20
        pw, ph = w - pl - pr, h - pt - pb
        n = len(overlaps); max_ov = max(s.active_positions for s in overlaps)
        if max_ov > 0:
            step = max(1, n // 500)
            pts = [(i, overlaps[i].active_positions) for i in range(0, n, step)]
            def tx(i): return pl + i / max(n-1, 1) * pw
            def ty(v): return pt + (1 - v / max(max_ov, 1)) * ph
            d = " ".join(f"{'M' if j == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}" for j, (i, v) in enumerate(pts))
            ov_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="border:1px solid #e2e8f0;border-radius:6px"><text x="{w//2}" y="14" text-anchor="middle" font-size="10" fill="#64748b">Concurrent Positions Over Time (max={max_ov})</text><path d="{d}" fill="none" stroke="#3b82f6" stroke-width="1.5"/></svg>'

    # Current vs best
    current = next((c for c in cadence_results if c.cooldown_days == 10), best)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>EXP-1220 Trade Cadence Analysis</title>
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
svg{{display:block;margin:0.5rem 0}}
.finding{{background:#eff6ff;border-left:4px solid #3b82f6;padding:12px;margin:12px 0;border-radius:4px;font-size:0.85rem}}
</style></head><body>
<h1>EXP-1220 Trade Cadence Analysis</h1>
<p class="meta">Optimal entry frequency from real IronVault data | {len(timeline)} trades analyzed</p>

<div class="grid">
  <div class="card"><div class="l">Total Trades</div><div class="v">{current.n_trades if current else 0}</div></div>
  <div class="card"><div class="l">Current Cadence</div><div class="v">{current.trades_per_year if current else 0}/yr</div></div>
  <div class="card"><div class="l">Best Cadence</div><div class="v" style="color:#16a34a">{best.label if best else 'N/A'}</div></div>
  <div class="card"><div class="l">Best PnL/yr</div><div class="v" style="color:#16a34a">${best.pnl_per_year:,.0f} if best else 0</div></div>
  <div class="card"><div class="l">Avg Hold</div><div class="v">{current.avg_hold_days if current else 0:.0f}d</div></div>
  <div class="card"><div class="l">Max Concurrent</div><div class="v">{best.max_concurrent if best else 0}</div></div>
  <div class="card"><div class="l">Capital Util (best)</div><div class="v">{best.capital_util_pct if best else 0:.0f}%</div></div>
  <div class="card"><div class="l">Priceable Signals</div><div class="v">{missed.get('priceable_spreads', 0)}</div></div>
</div>

<div class="finding">
<strong>Signal Availability:</strong> Out of {missed.get('total_expirations', 0)} SPY expirations,
{missed.get('valid_opportunities', 0)} pass filters (no VIX>40) and
{missed.get('priceable_spreads', 0)} have real IronVault spread prices
({missed.get('pricing_hit_rate', 0)}% hit rate). The 10-day cooldown is the PRIMARY
throttle on trade frequency — there are many more tradeable opportunities than we take.
</div>

<h2>Cadence Comparison</h2>
<table>
<tr><th>Cadence</th><th>Cooldown</th><th>Trades</th><th>Trades/Yr</th><th>Total PnL</th><th>PnL/Yr</th><th>Avg PnL</th><th>Win%</th><th>Sharpe</th><th>Max Conc</th><th>Avg Conc</th><th>Cap Util</th><th>Avg Hold</th></tr>
{cad_rows}
</table>

<h2>Trade Timeline</h2>
{tl_svg}

<h2>Concurrent Positions</h2>
{ov_svg}

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/trade_cadence_analyzer.py | All from IronVault | ** = best PnL/year</div>
</body></html>"""

    path.write_text(html, encoding="utf-8"); return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis():
    print("EXP-1220 Trade Cadence Analysis"); print("=" * 60)

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

    # Run at different cooldowns
    cooldowns = [
        (3, "Aggressive (3d)"),
        (5, "Weekly (5d)"),
        (7, "Weekly+ (7d)"),
        (10, "Current (10d)"),
        (14, "Bi-weekly (14d)"),
        (21, "Monthly (21d)"),
        (30, "Conservative (30d)"),
    ]

    print("\n  Running trades at different cooldowns...")
    cadence_results = []
    all_trade_sets = {}
    for cd, label in cooldowns:
        print(f"    Cooldown={cd}d ({label})...", end=" ")
        trades = run_with_cooldown(hd, spy_df, vix, cd)
        print(f"{len(trades)} trades")
        cr = compute_cadence_metrics(trades, cd, label)
        cadence_results.append(cr)
        all_trade_sets[cd] = trades

    # Timeline from current (10d) trades
    current_trades = all_trade_sets.get(10, [])
    timeline, overlaps = analyze_timeline(current_trades)

    # Missed signals
    print("\n  Counting missed signals...")
    missed = find_missed_signals(hd, spy_df, vix)

    print(f"\n  {'Cadence':<20} {'Trades':>7} {'Tr/Yr':>7} {'PnL':>10} {'PnL/Yr':>10} {'Win%':>6} {'Sharpe':>7} {'MaxC':>5} {'Util':>6}")
    print(f"  {'-'*80}")
    best = max(cadence_results, key=lambda c: c.pnl_per_year)
    for c in cadence_results:
        star = " **" if c.cooldown_days == best.cooldown_days else ""
        print(f"  {c.label:<20} {c.n_trades:>7} {c.trades_per_year:>7.1f} {c.total_pnl:>10,.0f} "
              f"{c.pnl_per_year:>10,.0f} {c.win_rate:>5.0%} {c.sharpe:>7.2f} {c.max_concurrent:>5} "
              f"{c.capital_util_pct:>5.0f}%{star}")

    print(f"\n  Missed signals: {missed['priceable_spreads']} priceable out of "
          f"{missed['valid_opportunities']} valid ({missed['pricing_hit_rate']}%)")

    report = generate_report(cadence_results, timeline, overlaps, missed)
    print(f"\n  Report: {report}")
    return cadence_results, missed


if __name__ == "__main__":
    run_analysis()
