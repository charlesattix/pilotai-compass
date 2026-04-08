"""
EXP-2010 — Tail Risk Convexity (Long OTM Puts as Alpha).

HYPOTHESIS (Carlos): instead of *selling* vol like the rest of the
compass book, BUY cheap OTM puts on SPY (and QQQ where data permits)
that bleed slowly in calm markets but pay off massively in crashes. Even
with negative standalone CAGR, the *negative correlation* to EXP-1220
should improve the combined-portfolio Sharpe enough to be worth the
premium drag — the classic Spitznagel/Universa "convexity" trade.

REAL DATA — Rule Zero respected:
  * Underlying: real Yahoo SPY/QQQ daily close.
  * Option prices: REAL closes from `data/options_cache.db` (IronVault),
    via `option_contracts` JOIN `option_daily` (ticker = SPY or QQQ).
  * Implied vol is *derived* from those real prices via Brent's-method
    BS inversion, used only to identify the ~10-delta strike. The trade
    P&L itself is the difference of two literal `option_daily` closes —
    no model anywhere in the cash flows.

UNIVERSE NOTE — QQQ is BLOCKED for 2024-2025: IronVault only carries
99 sparse QQQ snapshot dates ending 2023, so the strategy is run on
SPY for the full window and on QQQ wherever data exists, with the
QQQ gap documented in the report. This is the same gap MASTERPLAN
flags ("QQQ stale 32 months"). Including a synthetic QQQ leg would
violate Rule Zero.

PIPELINE
  Each month, on the first available IronVault snapshot date:
    1. Find the SPY (and QQQ if available) put expiration closest to
       30 DTE.
    2. Pull every put strike on that expiry that has a real
       `option_daily` close on the snapshot date.
    3. For each strike, invert the BS put formula → σ_K, then compute
       Δ_K from that σ_K.
    4. Buy the strike whose Δ is closest to −0.10 ("10-delta put").
    5. Hold to expiration. Exit at the real `option_daily` close on
       (or the most recent close before) the expiration date. If the
       contract has no recorded close after entry, mark exit price = 0
       (the put expired worthless — the most common outcome).
    6. P&L per contract = exit_price − entry_price (in dollars).
       Position size = 0.5% of capital allocated to premium per leg
       per month, so the calm-market bleed is bounded around
       12 × 0.5% = 6%/yr per leg before any payoffs.

Walk-forward by year (2020 → 2025) for OOS reporting.

Outputs:
  compass/exp2010_tail_convexity.py            (this file)
  compass/reports/exp2010_tail_convexity.json
  compass/reports/exp2010_tail_convexity.html

Tag: EXP-2010
Run: python3 -m compass.exp2010_tail_convexity
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.exp1960_skew_alpha import (
    bs_put_delta,
    fetch_contract_close,
    fetch_put_chain,
    find_target_expiration,
    implied_vol_put,
    list_snapshot_dates,
    load_exp1220_yearly,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT, "data", "options_cache.db")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")

START = "2020-01-01"
END = "2025-12-31"

TICKERS = ["SPY", "QQQ"]
TARGET_DTE = 35              # midpoint of 30-45 DTE band
TARGET_DELTA = -0.10         # 10-delta put
PREMIUM_PCT_PER_LEG = 0.005  # 0.5% of capital allocated as premium per leg per month
RISK_FREE = 0.045


# ── Snapshot enumeration ───────────────────────────────────────────────


def fetch_underlying_close(symbol: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(symbol, start=START, end=END, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"Yahoo {symbol} empty")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = symbol
    return s.dropna()


def first_snapshot_per_month(con: sqlite3.Connection, ticker: str) -> List[str]:
    """Pick one snapshot per (year, month) — the earliest available."""
    rows = con.execute("""
        SELECT MIN(as_of_date)
        FROM option_contracts
        WHERE ticker=? AND as_of_date BETWEEN ? AND ?
        GROUP BY substr(as_of_date,1,7)
        ORDER BY 1
    """, (ticker, START, END)).fetchall()
    return [r[0] for r in rows if r[0] is not None]


def list_snapshot_dates_for(con: sqlite3.Connection, ticker: str) -> List[str]:
    rows = con.execute("""
        SELECT DISTINCT as_of_date FROM option_contracts
        WHERE ticker=? ORDER BY as_of_date
    """, (ticker,)).fetchall()
    return [r[0] for r in rows]


def find_target_expiration_for(con: sqlite3.Connection, ticker: str,
                               snapshot: str, target_dte: int) -> Optional[str]:
    rows = con.execute("""
        SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker=? AND as_of_date=? AND option_type='P'
        ORDER BY expiration
    """, (ticker, snapshot)).fetchall()
    if not rows:
        return None
    snap_dt = datetime.strptime(snapshot, "%Y-%m-%d")
    target = snap_dt + timedelta(days=target_dte)
    best = min((datetime.strptime(r[0], "%Y-%m-%d") for r in rows),
               key=lambda e: abs((e - target).days))
    if (best - snap_dt).days < 14:
        return None
    return best.strftime("%Y-%m-%d")


def fetch_put_chain_for(con: sqlite3.Connection, ticker: str,
                        snapshot: str, expiration: str
                        ) -> List[Tuple[float, float, str]]:
    rows = con.execute("""
        SELECT c.strike, d.close, c.contract_symbol
        FROM option_contracts c
        JOIN option_daily d ON c.contract_symbol = d.contract_symbol
        WHERE c.ticker=? AND c.option_type='P'
          AND c.expiration=? AND d.date=? AND d.close > 0
        ORDER BY c.strike
    """, (ticker, expiration, snapshot)).fetchall()
    return [(float(s), float(p), sym) for s, p, sym in rows]


# ── Trade structures ───────────────────────────────────────────────────


@dataclass
class TailTrade:
    ticker: str
    entry_date: str
    expiration: str
    strike: float
    entry_premium: float
    exit_date: str
    exit_premium: float
    pnl_per_contract: float
    pnl_pct_capital: float
    delta_at_entry: float
    iv_at_entry: float
    spot_at_entry: float


def select_ten_delta_put(spot: float, dte_days: int, chain: List[Tuple[float, float, str]],
                         target_delta: float = TARGET_DELTA
                         ) -> Optional[Tuple[float, float, str, float, float]]:
    """Return (strike, premium, contract_symbol, delta, iv) for the strike
    whose computed Δ is closest to target_delta."""
    T = dte_days / 365.0
    if T <= 0:
        return None
    best = None
    best_dist = math.inf
    for K, px, sym in chain:
        sigma = implied_vol_put(px, spot, K, T, RISK_FREE)
        if sigma is None or sigma <= 0:
            continue
        delta = bs_put_delta(spot, K, T, sigma, RISK_FREE)
        dist = abs(delta - target_delta)
        if dist < best_dist:
            best_dist = dist
            best = (K, px, sym, delta, sigma)
    return best


def run_one_trade(con: sqlite3.Connection, ticker: str, snapshot: str,
                  spot: float) -> Optional[TailTrade]:
    expiration = find_target_expiration_for(con, ticker, snapshot, TARGET_DTE)
    if expiration is None:
        return None
    chain = fetch_put_chain_for(con, ticker, snapshot, expiration)
    if len(chain) < 6:
        return None
    snap_dt = datetime.strptime(snapshot, "%Y-%m-%d")
    exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
    dte = (exp_dt - snap_dt).days
    pick = select_ten_delta_put(spot, dte, chain)
    if pick is None:
        return None
    strike, entry_premium, sym, delta, iv = pick

    # Exit at (or just before) expiration. If no close exists in that
    # window, treat the put as having expired worthless (the typical
    # outcome for a 10-delta put).
    exit_window_start = (snap_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    exit_info = fetch_contract_close(con, sym, exit_window_start, expiration)
    if exit_info is None:
        # Try a 5-day window past expiration in case the last bar was
        # recorded slightly late
        late = (exp_dt + timedelta(days=5)).strftime("%Y-%m-%d")
        exit_info = fetch_contract_close(con, sym, exit_window_start, late)
    if exit_info is None:
        exit_date = expiration
        exit_premium = 0.0  # expired worthless
    else:
        exit_date, exit_premium = exit_info

    pnl_per_contract = exit_premium - entry_premium

    # Position sizing: spend PREMIUM_PCT_PER_LEG of capital on entry
    # premium. n_contracts = (cap × pct) / (entry_premium × 100). The
    # P&L as % of capital is then:
    #   pnl_pct = n_contracts × pnl_per_contract × 100 / cap
    #           = (cap × pct / (entry_premium × 100)) × pnl_per_contract × 100 / cap
    #           = pct × pnl_per_contract / entry_premium
    if entry_premium <= 0:
        return None
    pnl_pct_capital = PREMIUM_PCT_PER_LEG * pnl_per_contract / entry_premium

    return TailTrade(
        ticker=ticker,
        entry_date=snapshot,
        expiration=expiration,
        strike=float(strike),
        entry_premium=float(entry_premium),
        exit_date=str(exit_date),
        exit_premium=float(exit_premium),
        pnl_per_contract=float(pnl_per_contract),
        pnl_pct_capital=float(pnl_pct_capital),
        delta_at_entry=float(delta),
        iv_at_entry=float(iv),
        spot_at_entry=float(spot),
    )


# ── Backtest driver ────────────────────────────────────────────────────


@dataclass
class TickerResult:
    ticker: str
    n_trades: int
    n_wins: int
    win_rate: float
    total_pnl_pct: float
    cagr: float
    sharpe: float
    max_dd: float
    yearly_pnl_pct: Dict[int, float]
    trades: List[TailTrade]
    daily_pnl: pd.Series


def backtest_ticker(con: sqlite3.Connection, ticker: str) -> Optional[TickerResult]:
    print(f"[exp2010] backtest {ticker}…", flush=True)
    spot = fetch_underlying_close(ticker)

    snap_dates = first_snapshot_per_month(con, ticker)
    if not snap_dates:
        print(f"[exp2010] {ticker}: no snapshots in window")
        return None
    print(f"[exp2010] {ticker}: {len(snap_dates)} monthly entry candidates")

    trades: List[TailTrade] = []
    daily_pnl = pd.Series(0.0, index=pd.date_range(START, END, freq="D"))

    for s in snap_dates:
        try:
            spot_val = float(spot.loc[:s].iloc[-1])
        except (KeyError, IndexError):
            continue
        trade = run_one_trade(con, ticker, s, spot_val)
        if trade is None:
            continue
        trades.append(trade)
        # Spread P&L over the holding period for the daily series
        try:
            entry_dt = pd.Timestamp(trade.entry_date)
            exit_dt = pd.Timestamp(trade.exit_date)
            window = pd.date_range(entry_dt, exit_dt, freq="D")
            if len(window) > 0:
                per_day = trade.pnl_pct_capital / len(window)
                idx = daily_pnl.index.intersection(window)
                daily_pnl.loc[idx] += per_day
        except Exception:
            pass

    if not trades:
        print(f"[exp2010] {ticker}: no trades produced")
        return None

    n_wins = sum(1 for t in trades if t.pnl_per_contract > 0)
    total_pct = sum(t.pnl_pct_capital for t in trades)
    yearly: Dict[int, float] = {}
    for t in trades:
        y = int(t.entry_date[:4])
        yearly.setdefault(y, 0.0)
        yearly[y] += t.pnl_pct_capital

    eq = (1.0 + daily_pnl).cumprod()
    years = (daily_pnl.index[-1] - daily_pnl.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    pk = eq.cummax()
    max_dd = float(((eq - pk) / pk).min())
    nz_std = float(daily_pnl.std())
    sharpe = float(daily_pnl.mean() / nz_std * math.sqrt(252)) if nz_std > 0 else 0.0

    print(f"[exp2010] {ticker}: {len(trades)} trades, {n_wins} wins, "
          f"total {total_pct*100:+.2f}%, CAGR {cagr*100:+.2f}%, "
          f"DD {max_dd*100:+.2f}%, Sharpe {sharpe:+.2f}")

    return TickerResult(
        ticker=ticker,
        n_trades=len(trades),
        n_wins=n_wins,
        win_rate=n_wins / len(trades),
        total_pnl_pct=total_pct,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=max_dd,
        yearly_pnl_pct=yearly,
        trades=trades,
        daily_pnl=daily_pnl,
    )


# ── Correlation helpers ────────────────────────────────────────────────


def correlate_yearly(yearly_strat: Dict[int, float],
                     yearly_ref: Dict[int, float]) -> Optional[float]:
    common = sorted(set(yearly_strat) & set(yearly_ref))
    if len(common) < 3:
        return None
    a = np.array([yearly_strat[y] for y in common], dtype=float)
    b = np.array([yearly_ref[y] for y in common], dtype=float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ── Report ─────────────────────────────────────────────────────────────


def _fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x*100:.{dp}f}%" if np.isfinite(x) else "—"


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(results: Dict[str, TickerResult],
                exp1220: Dict[int, float],
                correlations: Dict[str, Optional[float]]) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1200px;color:#111}
    h1{border-bottom:3px solid #0e3b59}
    h2{margin-top:2em;color:#0e3b59}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#0e3b59;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#0e3b59}
    """
    h: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2010 Tail Risk Convexity</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2010 — Tail Risk Convexity (Long OTM Puts)</h1>",
        "<p class='muted'>Systematic monthly long ~10-delta SPY/QQQ puts, "
        "~30-45 DTE, held to expiration. Real IronVault prices, no synthetic "
        "fills.</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault option prices</span></p>",
    ]

    # Per-ticker
    h.append("<h2>Per-ticker results</h2>")
    h.append("<table><tr><th>Ticker</th><th>Trades</th><th>Wins</th>"
             "<th>Win rate</th><th>Total P&L (%cap)</th><th>CAGR</th>"
             "<th>Sharpe</th><th>Max DD</th><th>Corr vs EXP-1220</th></tr>")
    for tk, r in results.items():
        c = correlations.get(tk)
        c_str = f"{c:+.2f}" if c is not None else "n/a"
        h.append(
            f"<tr><td class='l'><b>{tk}</b></td>"
            f"<td>{r.n_trades}</td><td>{r.n_wins}</td>"
            f"<td>{_fmt_pct(r.win_rate, 1)}</td>"
            f"<td class='{ 'pos' if r.total_pnl_pct>0 else 'neg' }'>{_fmt_pct(r.total_pnl_pct)}</td>"
            f"<td class='{ 'pos' if r.cagr>0 else 'neg' }'>{_fmt_pct(r.cagr)}</td>"
            f"<td>{_fmt(r.sharpe)}</td>"
            f"<td class='neg'>{_fmt_pct(r.max_dd)}</td>"
            f"<td>{c_str}</td></tr>"
        )
    h.append("</table>")

    # Yearly grid
    h.append("<h2>Yearly P&L by ticker (% of capital)</h2>")
    years = sorted({y for r in results.values() for y in r.yearly_pnl_pct})
    if years:
        h.append("<table><tr><th>Ticker</th>" +
                 "".join(f"<th>{y}</th>" for y in years) + "</tr>")
        for tk, r in results.items():
            h.append(f"<tr><td class='l'><b>{tk}</b></td>")
            for y in years:
                v = r.yearly_pnl_pct.get(y, 0.0)
                cls = "pos" if v > 0 else ("neg" if v < 0 else "")
                h.append(f"<td class='{cls}'>{_fmt_pct(v, 2)}</td>")
            h.append("</tr>")
        h.append("</table>")

    if exp1220:
        h.append("<h3>EXP-1220 reference (yearly protected return)</h3>")
        h.append("<table><tr>" + "".join(f"<th>{y}</th>" for y in sorted(exp1220)) + "</tr><tr>")
        for y in sorted(exp1220):
            v = exp1220[y]
            cls = "pos" if v > 0 else "neg"
            h.append(f"<td class='{cls}'>{_fmt_pct(v)}</td>")
        h.append("</tr></table>")

    # Trade tape (first 15 + last 15 + biggest payoffs)
    for tk, r in results.items():
        h.append(f"<h2>{tk} trade tape ({len(r.trades)} trades)</h2>")
        biggest = sorted(r.trades, key=lambda t: -t.pnl_pct_capital)[:5]
        h.append("<h3>Top 5 payoffs</h3>")
        h.append("<table><tr><th>Entry</th><th>Exit</th><th>K</th>"
                 "<th>Δ</th><th>IV</th><th>Entry $</th><th>Exit $</th>"
                 "<th>P&L %cap</th></tr>")
        for t in biggest:
            h.append(
                f"<tr><td class='l'>{t.entry_date}</td>"
                f"<td class='l'>{t.exit_date}</td>"
                f"<td>{t.strike:.0f}</td>"
                f"<td>{t.delta_at_entry:+.3f}</td>"
                f"<td>{t.iv_at_entry*100:.1f}%</td>"
                f"<td>{t.entry_premium:.2f}</td>"
                f"<td>{t.exit_premium:.2f}</td>"
                f"<td class='pos'>{_fmt_pct(t.pnl_pct_capital, 3)}</td></tr>"
            )
        h.append("</table>")

        h.append("<h3>Recent 15 trades</h3>")
        h.append("<table><tr><th>Entry</th><th>Exit</th><th>K</th>"
                 "<th>Δ</th><th>IV</th><th>Entry $</th><th>Exit $</th>"
                 "<th>P&L %cap</th></tr>")
        for t in r.trades[-15:]:
            cls = "pos" if t.pnl_pct_capital > 0 else "neg"
            h.append(
                f"<tr><td class='l'>{t.entry_date}</td>"
                f"<td class='l'>{t.exit_date}</td>"
                f"<td>{t.strike:.0f}</td>"
                f"<td>{t.delta_at_entry:+.3f}</td>"
                f"<td>{t.iv_at_entry*100:.1f}%</td>"
                f"<td>{t.entry_premium:.2f}</td>"
                f"<td>{t.exit_premium:.2f}</td>"
                f"<td class='{cls}'>{_fmt_pct(t.pnl_pct_capital, 3)}</td></tr>"
            )
        h.append("</table>")

    # Methodology
    h.append("<h2>Methodology &amp; honest caveats</h2>")
    h.append("<ul>")
    h.append("<li><b>Real options:</b> entry premium and exit premium are "
             "both literal closes from <code>option_daily</code>. Strike "
             "selection uses BS Δ inverted from each strike's real close, "
             "which is allowed under Rule Zero (model used to "
             "<i>identify</i> the strike, not to fabricate the price).</li>")
    h.append("<li><b>Position size:</b> 0.5%/cap of premium per leg per "
             "month. Calm-market upper bound bleed ≈ 12 × 0.5% = "
             "6%/yr/leg if every put expires worthless.</li>")
    h.append("<li><b>Exit:</b> last `option_daily` close at or before the "
             "expiration date; if none recorded, the put is marked "
             "expired worthless (exit price = 0). This matches the "
             "real life of an OTM put — the typical outcome IS exit at 0.</li>")
    h.append("<li><b>QQQ data gap:</b> IronVault carries 99 sparse QQQ "
             "snapshots ending 2023; nothing in 2024-2025. The QQQ leg is "
             "honest only for the dates with real data and is not "
             "extrapolated. Production deployment of the convexity sleeve "
             "will need a backfill of QQQ contracts (same OCC-construction "
             "method as the Dec 2025 TLT backfill). Until then SPY is the "
             "only continuously-funded leg.</li>")
    h.append("<li><b>Walk-forward:</b> the rule has zero free parameters "
             "(monthly entry, fixed Δ target, fixed DTE target, hold to "
             "expiration), so there is nothing to overfit and no train/test "
             "split is meaningful. The OOS report IS the full-period "
             "report, broken down per year.</li>")
    h.append("<li><b>Why the headline CAGR is the wrong number:</b> a "
             "convexity sleeve is meant to lose money in calm regimes and "
             "make multiples in tail events. The standalone CAGR is the "
             "*premium burn rate*, not the value-add. The real measurement "
             "lives in the portfolio-combination test (run this stream "
             "through compass/north_star_stress_test.py as a 5% sleeve and "
             "look at the combined Sharpe/DD).</li>")
    h.append("<li><b>What this is NOT:</b> a vol-of-vol or VIX-call hedge. "
             "Both of those need OPRA-level VIX option data which IronVault "
             "does not carry. The SPY 10-Δ put is the cleanest tail-hedge "
             "instrument we can run on existing real data.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("[exp2010] opening IronVault DB…", flush=True)
    con = sqlite3.connect(DB_PATH)
    try:
        results: Dict[str, TickerResult] = {}
        for tk in TICKERS:
            r = backtest_ticker(con, tk)
            if r is not None:
                results[tk] = r

        if not results:
            print("[exp2010] no tickers produced results — aborting")
            return 1

        exp1220 = load_exp1220_yearly()
        correlations = {
            tk: correlate_yearly(r.yearly_pnl_pct, exp1220) for tk, r in results.items()
        }

        html = render_html(results, exp1220, correlations)
        out_html = os.path.join(REPORT_DIR, "exp2010_tail_convexity.html")
        with open(out_html, "w") as f:
            f.write(html)
        print(f"[exp2010] wrote {out_html}")

        out_json = os.path.join(REPORT_DIR, "exp2010_tail_convexity.json")
        summary = {
            "experiment": "EXP-2010",
            "tag": "EXP-2010",
            "description": "Long ~10Δ OTM puts, monthly roll, real IronVault data",
            "data_sources": {
                "spot": "Yahoo Finance daily close",
                "options": "IronVault data/options_cache.db (real put closes)",
                "iv_method": "Black-Scholes inversion via Brent's method (strike selection only)",
            },
            "config": {
                "tickers": TICKERS,
                "target_dte": TARGET_DTE,
                "target_delta": TARGET_DELTA,
                "premium_pct_per_leg": PREMIUM_PCT_PER_LEG,
                "risk_free": RISK_FREE,
                "window": {"start": START, "end": END},
            },
            "results": {
                tk: {
                    "n_trades": r.n_trades,
                    "n_wins": r.n_wins,
                    "win_rate": r.win_rate,
                    "total_pnl_pct": r.total_pnl_pct,
                    "cagr": r.cagr,
                    "sharpe": r.sharpe,
                    "max_dd": r.max_dd,
                    "yearly_pnl_pct": r.yearly_pnl_pct,
                    "corr_vs_exp1220_yearly": correlations.get(tk),
                    "trades": [
                        {
                            "entry": t.entry_date,
                            "exit": t.exit_date,
                            "expiration": t.expiration,
                            "strike": t.strike,
                            "entry_premium": t.entry_premium,
                            "exit_premium": t.exit_premium,
                            "pnl_per_contract": t.pnl_per_contract,
                            "pnl_pct_capital": t.pnl_pct_capital,
                            "delta_at_entry": t.delta_at_entry,
                            "iv_at_entry": t.iv_at_entry,
                            "spot_at_entry": t.spot_at_entry,
                        }
                        for t in r.trades
                    ],
                }
                for tk, r in results.items()
            },
            "exp1220_yearly_protected_return": exp1220,
        }
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[exp2010] wrote {out_json}")
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
