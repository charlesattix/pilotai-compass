"""
EXP-2020 — Cross-Sectional Vol Arbitrage
========================================

Hypothesis
----------
Across an ETF universe, the IV−RV spread (volatility risk premium) varies
cross-sectionally. At any point in time, the name with the widest IV−RV
spread is the "richest" vol to sell, the narrowest is the "cheapest" to
buy. A long/short pair capturing that dispersion should earn the VRP
without directional equity beta.

Universe (REAL IronVault option coverage)
-----------------------------------------
Requested: SPY, QQQ, IWM, XLF, XLI, IBIT
Available: SPY (193K), QQQ (23K), XLF (9K), XLI (17K)
Dropped  : IWM (0 contracts), IBIT (0 contracts)   ← Rule Zero

With 4 names, the long/short pair is top-1 vs bottom-1. That is still a
market-neutral vol trade but obviously less diversified than a 6-name
version; flagged in the report.

Signal
------
Weekly (Mondays):
  1. For each ticker, locate the closest-to-30-DTE expiry from IronVault.
  2. At ATM (strike nearest underlying close), pull the call AND put
     close prices. Invert Black-Scholes on each to get σ_call and σ_put;
     IV = mean of the two (put-call average — robust to quoting noise).
  3. RV_20d = trailing 20d annualised stdev of log-returns on Yahoo close.
  4. spread_i = IV_i − RV_i.
  5. Rank spreads across the 4 tickers. Long the minimum, short the
     maximum, equal vega-notional ($1 of vega per leg).

P&L model
---------
Closed-form variance-swap-style payoff over the 21-trading-day holding
window:

     PnL_short_leg = +vega_notional × (IV_entry − RV_forward_21d)
     PnL_long_leg  = +vega_notional × (RV_forward_21d − IV_entry)
     position_pnl  = PnL_short_leg + PnL_long_leg

This is the standard linear-vega approximation of a delta-hedged
straddle held through τ days of realisation. It is analytically clean,
avoids path dependence from imperfect delta hedging, and only uses
REAL entry IV (from IronVault chains) and REAL forward RV (from Yahoo
closes). NO synthetic prices.

Positions overlap weekly (rebalance each Monday) so the book holds
~4 concurrent pairs at steady state.

Outputs
-------
  compass/reports/exp2020_cross_vol_arb.json
  compass/reports/exp2020_cross_vol_arb.html

Rule Zero clean.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.pricing import bs_price
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2020_cross_vol_arb.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2020_cross_vol_arb.html"

UNIVERSE_REQUESTED = ["SPY", "QQQ", "IWM", "XLF", "XLI", "IBIT"]
UNIVERSE           = ["SPY", "QQQ", "XLF", "XLI"]   # IronVault-verified
DROPPED            = ["IWM", "IBIT"]

START = "2020-01-01"
END   = "2026-01-01"
TRADING_DAYS = 252
RISK_FREE = 0.045
HOLDING_DAYS = 21
VEGA_NOTIONAL = 10_000.0   # $ per 1.00 vol-point per leg  → $100/vol-pt


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_prices(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for t in tickers:
        d = yf.download(t, start="2019-06-01", end=END, progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).normalize()
        d["logret"] = np.log(d["Close"] / d["Close"].shift(1))
        d["rv_20"]  = d["logret"].rolling(20).std(ddof=1) * math.sqrt(TRADING_DAYS)
        out[t] = d
    return out


def invert_iv(price: float, S: float, K: float, T: float, option_type: str,
              r: float = RISK_FREE) -> Optional[float]:
    """Brent's method on BS price − target. Returns None if no bracket."""
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    lo, hi = 0.05, 3.0
    try:
        f = lambda sig: bs_price(S, K, T, r, sig, option_type) - price
        if f(lo) > 0:
            return lo          # price below min → floor
        if f(hi) < 0:
            return hi          # price above max → ceiling
        return float(brentq(f, lo, hi, xtol=1e-4, maxiter=60))
    except Exception:
        return None


def atm_iv(con: sqlite3.Connection, ticker: str, as_of: str, S: float,
           target_dte: int = 30) -> Optional[Tuple[float, float, float]]:
    """Return (IV_mean, expiry_yyyy_mm_dd, strike) for ATM ~30DTE on as_of.

    IV_mean averages call-IV and put-IV when both exist, else uses whichever.
    """
    # pick the expiration closest to +target_dte from as_of
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    lo = (as_of_dt + timedelta(days=target_dte - 10)).strftime("%Y-%m-%d")
    hi = (as_of_dt + timedelta(days=target_dte + 21)).strftime("%Y-%m-%d")
    exps = [r[0] for r in con.execute(
        "SELECT DISTINCT expiration FROM option_contracts "
        "WHERE ticker=? AND expiration BETWEEN ? AND ? ORDER BY expiration",
        (ticker, lo, hi),
    ).fetchall()]
    if not exps:
        return None
    best_exp = min(exps, key=lambda e: abs(
        (datetime.strptime(e, "%Y-%m-%d") - as_of_dt).days - target_dte
    ))
    T = max((datetime.strptime(best_exp, "%Y-%m-%d") - as_of_dt).days / 365.0, 1 / 365)
    # strikes for that expiry
    strikes = [float(r[0]) for r in con.execute(
        "SELECT DISTINCT strike FROM option_contracts "
        "WHERE ticker=? AND expiration=?",
        (ticker, best_exp),
    ).fetchall()]
    if not strikes:
        return None
    K = min(strikes, key=lambda k: abs(k - S))
    # pull call + put closes
    iv_vals = []
    for opt_type in ("C", "P"):
        row = con.execute(
            "SELECT od.close FROM option_daily od "
            "JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol "
            "WHERE oc.ticker=? AND oc.expiration=? AND oc.strike=? "
            "  AND oc.option_type=? AND od.date=?",
            (ticker, best_exp, K, opt_type, as_of),
        ).fetchone()
        if row is None or row[0] is None:
            continue
        iv = invert_iv(float(row[0]), S, K, T, opt_type)
        if iv is not None:
            iv_vals.append(iv)
    if not iv_vals:
        return None
    return float(np.mean(iv_vals)), best_exp, K


# ─────────────────────────────────────────────────────────────────────────────
# Weekly signal construction
# ─────────────────────────────────────────────────────────────────────────────
def weekly_signal_panel(prices: Dict[str, pd.DataFrame],
                         hd: IronVault) -> pd.DataFrame:
    con = sqlite3.connect(hd._db_path)
    mondays = pd.date_range(START, END, freq="W-MON")
    records = []
    for d in mondays:
        ds = d.strftime("%Y-%m-%d")
        row: Dict[str, float] = {"date": d}
        valid_count = 0
        for t in UNIVERSE:
            df = prices[t]
            if d not in df.index:
                continue
            S = float(df.loc[d, "Close"])
            rv = float(df.loc[d, "rv_20"]) if not pd.isna(df.loc[d, "rv_20"]) else None
            if rv is None or S != S:
                continue
            iv_tuple = atm_iv(con, t, ds, S)
            if iv_tuple is None:
                continue
            iv, exp, K = iv_tuple
            row[f"{t}_iv"]     = iv
            row[f"{t}_rv"]     = rv
            row[f"{t}_spread"] = iv - rv
            row[f"{t}_exp"]    = exp
            row[f"{t}_strike"] = K
            row[f"{t}_S"]      = S
            valid_count += 1
        if valid_count >= 2:
            records.append(row)
    con.close()
    panel = pd.DataFrame.from_records(records)
    if not panel.empty:
        panel = panel.set_index("date")
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Trade construction: long narrowest, short widest
# ─────────────────────────────────────────────────────────────────────────────
def build_trades(panel: pd.DataFrame, prices: Dict[str, pd.DataFrame]) -> List[Dict]:
    trades = []
    for dt, row in panel.iterrows():
        spreads = {}
        for t in UNIVERSE:
            v = row.get(f"{t}_spread")
            if v is not None and not pd.isna(v):
                spreads[t] = float(v)
        if len(spreads) < 2:
            continue
        ranked = sorted(spreads.items(), key=lambda kv: kv[1])
        long_t, long_spread   = ranked[0]    # narrowest
        short_t, short_spread = ranked[-1]   # widest
        if long_t == short_t:
            continue
        iv_long  = float(row[f"{long_t}_iv"])
        iv_short = float(row[f"{short_t}_iv"])

        # forward 21td realised vol for each leg, real Yahoo
        forward_rv = {}
        for t in (long_t, short_t):
            df = prices[t]
            if dt not in df.index:
                forward_rv[t] = None
                continue
            idx = df.index.get_loc(dt)
            if idx + HOLDING_DAYS >= len(df):
                forward_rv[t] = None
                continue
            window = df["logret"].iloc[idx + 1: idx + 1 + HOLDING_DAYS]
            if window.isna().all():
                forward_rv[t] = None
                continue
            fr = float(window.std(ddof=1) * math.sqrt(TRADING_DAYS))
            forward_rv[t] = fr
        if forward_rv[long_t] is None or forward_rv[short_t] is None:
            continue

        pnl_long  = VEGA_NOTIONAL * (forward_rv[long_t]  - iv_long)
        pnl_short = VEGA_NOTIONAL * (iv_short - forward_rv[short_t])
        pnl = pnl_long + pnl_short

        exit_idx = idx + HOLDING_DAYS  # computed above for short; use it
        exit_date = df.index[exit_idx]

        trades.append({
            "entry_date": dt.strftime("%Y-%m-%d"),
            "exit_date":  exit_date.strftime("%Y-%m-%d"),
            "long":       long_t,  "long_iv":  round(iv_long, 4),
            "long_rv_fwd":  round(forward_rv[long_t], 4),
            "long_spread": round(long_spread, 4),
            "short":      short_t, "short_iv": round(iv_short, 4),
            "short_rv_fwd": round(forward_rv[short_t], 4),
            "short_spread": round(short_spread, 4),
            "pnl_long":   round(pnl_long, 2),
            "pnl_short":  round(pnl_short, 2),
            "pnl":        round(pnl, 2),
        })
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(trades: List[Dict], label: str, starting_capital: float = 100_000) -> Dict:
    if not trades:
        return {"label": label, "n": 0, "pnl": 0.0, "wr": 0.0,
                "sharpe": 0.0, "cagr_pct": 0.0, "max_dd_pct": 0.0, "avg_pnl": 0.0}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    eq = starting_capital + pnl.cumsum()
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    yrs = max(1.0, (
        datetime.strptime(trades[-1]["exit_date"], "%Y-%m-%d") -
        datetime.strptime(trades[0]["entry_date"], "%Y-%m-%d")
    ).days / 365.25)
    tpy = len(pnl) / yrs
    rets = pnl / starting_capital
    mu, sd = rets.mean(), (rets.std(ddof=1) if len(rets) > 1 else 0.0)
    sharpe = (mu / sd) * math.sqrt(tpy) if sd > 1e-12 else 0.0
    return {
        "label": label, "n": int(len(pnl)), "pnl": float(pnl.sum()),
        "wr": float((pnl > 0).mean()),
        "sharpe": round(float(sharpe), 3),
        "cagr_pct": round(float((eq[-1] / starting_capital) ** (1 / yrs) * 100 - 100), 3),
        "max_dd_pct": round(float(-dd.min() * 100), 3),
        "avg_pnl": round(float(pnl.mean()), 2),
        "trades_per_yr": round(float(tpy), 2),
    }


def walk_forward(trades: List[Dict]) -> List[Dict]:
    by_year: Dict[int, List[Dict]] = {}
    for t in trades:
        by_year.setdefault(int(t["entry_date"][:4]), []).append(t)
    return [dict(year=y, **metrics(ts, str(y))) for y, ts in sorted(by_year.items())]


def corr_to_exp1220(trades: List[Dict]) -> Dict:
    from compass.exp1220_standalone import run_exp1220_trades
    import yfinance as yf
    hd = IronVault.instance()
    spy = yf.download("SPY", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).normalize()
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).normalize()
    e1220 = run_exp1220_trades(hd, spy, vix)
    a = pd.Series(0.0, index=spy.index)
    for t in e1220:
        ed = pd.Timestamp(t["exit_date"]).normalize()
        if ed in a.index:
            a.loc[ed] += t["pnl"] / 100_000
    b = pd.Series(0.0, index=spy.index)
    for t in trades:
        ed = pd.Timestamp(t["exit_date"]).normalize()
        if ed in b.index:
            b.loc[ed] += t["pnl"] / 100_000
    am = (1 + a).resample("ME").apply(lambda x: x.prod() - 1)
    bm = (1 + b).resample("ME").apply(lambda x: x.prod() - 1)
    common = am.index.intersection(bm.index)
    if len(common) < 6:
        return {"n_months": int(len(common)), "pearson": None, "spearman": None}
    return {
        "n_months": int(len(common)),
        "pearson":  round(float(am.loc[common].corr(bm.loc[common], method="pearson")), 3),
        "spearman": round(float(am.loc[common].corr(bm.loc[common], method="spearman")), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"[1/5] universe: {UNIVERSE}  (dropped: {DROPPED})")
    print("[2/5] loading prices (real Yahoo) …")
    prices = load_prices(UNIVERSE)

    print("[3/5] building weekly IV/RV panel (real IronVault chains) …")
    hd = IronVault.instance()
    panel = weekly_signal_panel(prices, hd)
    print(f"      {len(panel)} weekly rows")

    print("[4/5] building long/short trades …")
    trades = build_trades(panel, prices)
    print(f"      {len(trades)} trades")

    m = metrics(trades, "cross_vol_arb")
    wf = walk_forward(trades)

    print("[5/5] correlation to EXP-1220 …")
    try:
        corr = corr_to_exp1220(trades)
    except Exception as e:
        print("      corr failed:", e)
        corr = {"n_months": 0, "pearson": None, "spearman": None}

    # Per-pair breakdown for transparency
    pair_stats = {}
    for t in trades:
        key = f"{t['long']}↑ / {t['short']}↓"
        pair_stats.setdefault(key, []).append(t["pnl"])
    pair_summary = {
        k: {
            "n": len(v),
            "total_pnl": round(float(np.sum(v)), 2),
            "avg_pnl": round(float(np.mean(v)), 2),
            "win_rate": round(float((np.array(v) > 0).mean()), 3),
        } for k, v in pair_stats.items()
    }

    payload = {
        "experiment": "EXP-2020",
        "name": "Cross-Sectional Vol Arbitrage (IV−RV long/short)",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "data_sources": {
            "implied_vol": "IronVault options_cache.db (real Polygon ATM ~30 DTE)",
            "realized_vol": "Yahoo Finance (trailing 20d annualised, forward 21d for settlement)",
            "options_iv_method": "Black-Scholes inversion via scipy.optimize.brentq on ATM C and P",
        },
        "universe_requested": UNIVERSE_REQUESTED,
        "universe_used": UNIVERSE,
        "universe_dropped": DROPPED,
        "drop_reason": "Zero contracts in IronVault options_cache.db (Rule Zero)",
        "params": {
            "target_dte": 30,
            "holding_days": HOLDING_DAYS,
            "rv_lookback": 20,
            "vega_notional_per_leg": VEGA_NOTIONAL,
            "risk_free": RISK_FREE,
        },
        "headline": m,
        "walk_forward": wf,
        "correlation_to_exp1220": corr,
        "pair_breakdown": pair_summary,
        "target_sharpe": 1.5,
        "target_sharpe_met": m["sharpe"] >= 1.5,
        "target_corr_lt": 0.3,
        "target_corr_met": (corr.get("pearson") is not None
                             and abs(corr["pearson"]) < 0.3),
        "first_trades_sample": trades[:5],
        "last_trades_sample":  trades[-5:],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    m = p["headline"]; corr = p["correlation_to_exp1220"]
    rows_wf = "".join(
        f"<tr><td>{r['year']}</td><td>{r['n']}</td><td>{r['wr']*100:.1f}%</td>"
        f"<td>{r['sharpe']:.2f}</td><td>{r['cagr_pct']:.2f}%</td>"
        f"<td>{r['max_dd_pct']:.2f}%</td><td>${r['pnl']:.0f}</td></tr>"
        for r in p["walk_forward"]
    )
    rows_pair = "".join(
        f"<tr><td>{k}</td><td>{v['n']}</td><td>{v['win_rate']*100:.1f}%</td>"
        f"<td>${v['avg_pnl']:.0f}</td><td>${v['total_pnl']:.0f}</td></tr>"
        for k, v in sorted(p["pair_breakdown"].items(), key=lambda kv: -kv[1]['total_pnl'])
    )
    sh_cls = "ok" if p["target_sharpe_met"] else "warn"
    co_cls = "ok" if p["target_corr_met"] else "warn"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2020 — Cross-Sectional Vol Arbitrage</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .warn{{color:#b86b00;font-weight:600}} .bad{{color:#b80000;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-2020 — Cross-Sectional Vol Arbitrage</h1>
<p class='small'>Generated {p['generated']} · IV from real IronVault ATM~30DTE ·
  RV from real Yahoo · Universe {p['universe_used']} (dropped {p['universe_dropped']} — zero IronVault coverage).</p>

<h2>Headline</h2>
<table>
<tr><th>Trades</th><th>Win rate</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>PnL</th><th>Avg PnL</th></tr>
<tr><td>{m['n']}</td><td>{m['wr']*100:.1f}%</td><td>{m['sharpe']:.2f}</td>
 <td>{m['cagr_pct']:.2f}%</td><td>{m['max_dd_pct']:.2f}%</td>
 <td>${m['pnl']:.0f}</td><td>${m['avg_pnl']:.0f}</td></tr>
</table>

<p>Target Sharpe &gt; 1.5: <span class='{sh_cls}'>{'MET' if p['target_sharpe_met'] else 'NOT MET'}</span>
 &nbsp;·&nbsp;
 Target |corr(EXP-1220)| &lt; 0.3: <span class='{co_cls}'>{'MET' if p['target_corr_met'] else 'NOT MET'}</span>
 (monthly pearson = {corr['pearson']}, spearman = {corr['spearman']}, n={corr['n_months']})</p>

<h2>Walk-forward by year</h2>
<table>
<tr><th>Year</th><th>n</th><th>WR</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>PnL</th></tr>
{rows_wf}
</table>

<h2>Pair breakdown (long↑ / short↓)</h2>
<table>
<tr><th>Pair</th><th>n</th><th>Win rate</th><th>Avg PnL</th><th>Total PnL</th></tr>
{rows_pair}
</table>

<h2>Method notes</h2>
<ul>
<li>Weekly Monday rebalance; 21-trading-day holding window per position.</li>
<li>IV = Black-Scholes inversion (scipy brentq) of ATM call AND put close
    prices on the closest-to-30DTE IronVault expiry, then averaged.</li>
<li>RV = trailing 20d annualised stdev of Yahoo log-returns at entry;
    forward RV = realised over the next 21 trading days (settlement).</li>
<li>P&amp;L = linear-vega variance-swap proxy:
    short = +V · (IV − RV_fwd), long = +V · (RV_fwd − IV),
    with V = ${VEGA_NOTIONAL:,.0f} notional per leg.</li>
<li>Universe shrunk from 6 to 4 names because IWM and IBIT have zero
    contracts in IronVault — Rule Zero forbids using synthetic chains.</li>
</ul>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# EXP-2690 — Production signal entry point
# ═══════════════════════════════════════════════════════════════════════════
def generate_today_signals(date):
    """Paper-trading scheduler entry point. Delegates to the central
    signal registry in compass.exp2690_signal_generators."""
    from compass.exp2690_signal_generators import vol_arb_signals
    return vol_arb_signals(date)
