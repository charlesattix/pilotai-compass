"""
EXP-1960 — SPY Put-Skew Alpha (smile-shape mean reversion).

HYPOTHESIS (Carlos): SPY put skew (25-delta-put IV minus ATM-put IV)
mean-reverts. When skew is extremely steep (fear premium high) the
market is over-paying for downside protection — sell the OTM put. When
skew is flat the protection is cheap and there's nothing to harvest;
stay flat. This is *different* from a VRP trade — VRP harvests the
*level* of IV vs realised; skew alpha trades the *shape* of the smile.

REAL DATA — Rule Zero respected:
  * Underlying spot: real Yahoo SPY daily close.
  * Option prices: REAL SPY put closes from
    `data/options_cache.db` (IronVault), via `option_contracts` joined
    to `option_daily`.
  * Implied vol is *derived* from those real prices by Black-Scholes
    inversion (Brent's method on `compass.greeks_sensitivity.bs_put_price`).
    Using a model to *invert* a real market price into IV is the
    standard textbook procedure and is allowed under Rule Zero — no
    fills are fabricated, only the σ that reproduces the real close.

PIPELINE
  Each weekly observation date:
    1. Pick the SPY snapshot date in `option_contracts` closest to today.
    2. Find the put expiration closest to ~30 calendar DTE.
    3. Pull every put strike on that expiry that has a real `option_daily`
       close on the snapshot date.
    4. For each strike, invert the BS put formula to get σ_K.
    5. ATM IV  = σ at the strike closest to spot.
       25-Δ IV = σ at the strike closest to a 0.25-Δ put (computed from
       the σ_K curve via the BS Δ formula — no parametric fitting,
       just a one-pass scan).
    6. skew_t = IV(25Δ put) − IV(ATM put).
  Then z-score skew over a 60-observation rolling window.

TRADING RULE
  * z > +1.5  → SHORT one 25-delta put for the chosen expiry. Hold to
                exit (14 calendar days OR expiration, whichever comes
                first), close at the real `option_daily` close on the
                exit date. P&L = entry premium − exit premium (per
                contract, no margin model — strategy reports P&L in
                vol-points-equivalent percent of capital).
  * z < −0.5  → close any open position immediately.
  * Walk-forward: 52 weekly observations train / 13 OOS / step 13.

OUTPUTS
  compass/exp1960_skew_alpha.py            (this file)
  compass/reports/exp1960_skew_alpha.json
  compass/reports/exp1960_skew_alpha.html

Tag: EXP-1960
Run: python3 -m compass.exp1960_skew_alpha
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.greeks_sensitivity import bs_put_price

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT, "data", "options_cache.db")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
EXP1220_SUMMARY = os.path.join(
    ROOT, "experiments", "EXP-1220-real", "results", "summary.json"
)

START = "2020-01-01"
END = "2025-12-31"

TARGET_DTE = 30          # ~30 calendar-day expiration
HOLD_DAYS = 14           # exit after 14 calendar days max
OBS_FREQ_DAYS = 7        # weekly observation cadence
RISK_FREE = 0.045        # used by BS inversion (FRED 3M T-bill ~average)

ZSCORE_WINDOW = 60       # rolling window for skew z-score (observations)
Z_ENTRY = 1.5            # short put when z above this
Z_EXIT = -0.5            # exit when z below this

TRAIN_OBS = 52           # ~1y of weekly observations
TEST_OBS = 13            # ~3m OOS
STEP_OBS = 13

# Capital model: each trade risks 1% of capital, P&L scaled accordingly
RISK_PER_TRADE = 0.01


# ── Data layer ─────────────────────────────────────────────────────────


def fetch_spy_close() -> pd.Series:
    import yfinance as yf
    df = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise RuntimeError("Yahoo SPY empty")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "spy"
    return s.dropna()


def list_snapshot_dates(con: sqlite3.Connection) -> List[str]:
    return [r[0] for r in con.execute("""
        SELECT DISTINCT as_of_date FROM option_contracts
        WHERE ticker='SPY'
        ORDER BY as_of_date
    """).fetchall()]


def find_target_expiration(con: sqlite3.Connection, snapshot: str,
                           target_dte: int) -> Optional[str]:
    rows = con.execute("""
        SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker='SPY' AND as_of_date=? AND option_type='P'
        ORDER BY expiration
    """, (snapshot,)).fetchall()
    if not rows:
        return None
    snap_dt = datetime.strptime(snapshot, "%Y-%m-%d")
    target = snap_dt + timedelta(days=target_dte)
    best = min((datetime.strptime(r[0], "%Y-%m-%d") for r in rows),
               key=lambda e: abs((e - target).days))
    if (best - snap_dt).days < 7:
        return None
    return best.strftime("%Y-%m-%d")


def fetch_put_chain(con: sqlite3.Connection, snapshot: str,
                    expiration: str) -> List[Tuple[float, float, str]]:
    rows = con.execute("""
        SELECT c.strike, d.close, c.contract_symbol
        FROM option_contracts c
        JOIN option_daily d ON c.contract_symbol = d.contract_symbol
        WHERE c.ticker='SPY' AND c.option_type='P'
          AND c.expiration=? AND d.date=?
          AND d.close > 0
        ORDER BY c.strike
    """, (expiration, snapshot)).fetchall()
    return [(float(s), float(p), sym) for s, p, sym in rows]


def fetch_contract_close(con: sqlite3.Connection, contract_symbol: str,
                         on_or_after: str, on_or_before: str) -> Optional[Tuple[str, float]]:
    """Latest available close for `contract_symbol` in [on_or_after, on_or_before]."""
    row = con.execute("""
        SELECT date, close FROM option_daily
        WHERE contract_symbol=? AND date BETWEEN ? AND ? AND close > 0
        ORDER BY date DESC LIMIT 1
    """, (contract_symbol, on_or_after, on_or_before)).fetchone()
    if not row:
        return None
    return row[0], float(row[1])


# ── BS inversion ───────────────────────────────────────────────────────


def implied_vol_put(price: float, S: float, K: float, T: float,
                    r: float = RISK_FREE) -> Optional[float]:
    """Brent's method on bs_put_price(σ) = market_price."""
    if T <= 0 or S <= 0 or K <= 0 or price <= 0:
        return None
    intrinsic = max(K * math.exp(-r * T) - S, 0.0)
    if price < intrinsic - 1e-6:
        return None
    lo, hi = 1e-4, 5.0
    f_lo = bs_put_price(S, K, T, lo, r) - price
    f_hi = bs_put_price(S, K, T, hi, r) - price
    if f_lo * f_hi > 0:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = bs_put_price(S, K, T, mid, r) - price
        if abs(f_mid) < 1e-6 or (hi - lo) < 1e-6:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def bs_put_delta(S: float, K: float, T: float, sigma: float,
                 r: float = RISK_FREE) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0))) - 1.0  # N(d1) − 1


# ── Skew snapshot ──────────────────────────────────────────────────────


@dataclass
class SkewSnapshot:
    snapshot: str
    expiration: str
    dte: int
    spot: float
    iv_atm: float
    iv_25d: float
    skew: float
    atm_strike: float
    twentyfive_strike: float
    twentyfive_symbol: str
    twentyfive_premium: float


def build_snapshot(con: sqlite3.Connection, snapshot: str,
                   spot: float) -> Optional[SkewSnapshot]:
    expiration = find_target_expiration(con, snapshot, TARGET_DTE)
    if expiration is None:
        return None
    chain = fetch_put_chain(con, snapshot, expiration)
    if len(chain) < 6:
        return None

    snap_dt = datetime.strptime(snapshot, "%Y-%m-%d")
    exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
    T = (exp_dt - snap_dt).days / 365.0
    if T <= 0:
        return None

    # Compute σ_K and Δ_K for every strike in the chain
    table: List[Tuple[float, float, float, str, float]] = []  # (K, σ, Δ, sym, px)
    for K, px, sym in chain:
        sigma = implied_vol_put(px, spot, K, T)
        if sigma is None or sigma <= 0:
            continue
        delta = bs_put_delta(spot, K, T, sigma)
        table.append((K, sigma, delta, sym, px))
    if len(table) < 6:
        return None

    # ATM σ: strike closest to spot
    atm = min(table, key=lambda r: abs(r[0] - spot))
    iv_atm = atm[1]
    atm_strike = atm[0]

    # 25-delta put: strike whose computed Δ is closest to −0.25
    target_delta = -0.25
    twentyfive = min(table, key=lambda r: abs(r[2] - target_delta))
    iv_25d = twentyfive[1]
    twentyfive_strike = twentyfive[0]
    twentyfive_symbol = twentyfive[3]
    twentyfive_premium = twentyfive[4]

    return SkewSnapshot(
        snapshot=snapshot,
        expiration=expiration,
        dte=(exp_dt - snap_dt).days,
        spot=float(spot),
        iv_atm=float(iv_atm),
        iv_25d=float(iv_25d),
        skew=float(iv_25d - iv_atm),
        atm_strike=float(atm_strike),
        twentyfive_strike=float(twentyfive_strike),
        twentyfive_symbol=twentyfive_symbol,
        twentyfive_premium=float(twentyfive_premium),
    )


def build_skew_history(con: sqlite3.Connection, spy: pd.Series) -> pd.DataFrame:
    """Walk weekly through every snapshot date, attach skew & metadata."""
    snapshots = list_snapshot_dates(con)
    snapshots = [s for s in snapshots if START <= s <= END]
    print(f"[exp1960] {len(snapshots)} candidate snapshot dates", flush=True)

    rows: List[SkewSnapshot] = []
    last_obs: Optional[datetime] = None
    for s in snapshots:
        sd = datetime.strptime(s, "%Y-%m-%d")
        if last_obs is not None and (sd - last_obs).days < OBS_FREQ_DAYS:
            continue
        # SPY spot for that date (forward-fill if exact day missing)
        if s in spy.index.strftime("%Y-%m-%d"):
            spot = float(spy.loc[s])
        else:
            slc = spy.loc[:s]
            if len(slc) == 0:
                continue
            spot = float(slc.iloc[-1])
        snap = build_snapshot(con, s, spot)
        if snap is None:
            continue
        rows.append(snap)
        last_obs = sd

    print(f"[exp1960] {len(rows)} usable weekly skew observations", flush=True)
    return pd.DataFrame([r.__dict__ for r in rows]).set_index(
        pd.DatetimeIndex([datetime.strptime(r.snapshot, "%Y-%m-%d") for r in rows])
    )


# ── Walk-forward backtest ──────────────────────────────────────────────


@dataclass
class FoldResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_obs_train: int
    n_obs_test: int
    n_trades: int
    pnl: float
    win_rate: float


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    expiration: str
    strike: float
    entry_premium: float
    exit_premium: float
    skew: float
    skew_z: float
    pnl_pct: float        # as fraction of capital, after risk_per_trade scaling
    win: bool


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=max(10, window // 3)).mean()
    sd = series.rolling(window, min_periods=max(10, window // 3)).std(ddof=0)
    return (series - mu) / sd.replace(0, np.nan)


def backtest_skew(con: sqlite3.Connection, snapshots: pd.DataFrame) -> Tuple[
        List[TradeRecord], pd.Series, List[FoldResult]]:
    """Walk-forward sell-put rule on the skew z-score."""
    snapshots = snapshots.sort_index().copy()
    snapshots["z"] = _zscore(snapshots["skew"], ZSCORE_WINDOW)

    n = len(snapshots)
    folds: List[FoldResult] = []
    trades: List[TradeRecord] = []
    daily_pnl_index = pd.date_range(START, END, freq="D")
    daily_pnl = pd.Series(0.0, index=daily_pnl_index)

    start = TRAIN_OBS
    while start + TEST_OBS <= n:
        train = snapshots.iloc[start - TRAIN_OBS:start]
        test = snapshots.iloc[start:start + TEST_OBS]

        # No per-fold tuning of z thresholds beyond the fixed Carlos rule —
        # we still use the train window to compute the rolling z, and only
        # apply trades that occur within the OOS window.
        n_test_trades = 0
        wins = 0
        fold_pnl = 0.0

        position_open = False
        open_trade: Optional[Dict] = None

        for ts, row in test.iterrows():
            z = row["z"]
            if not np.isfinite(z):
                continue

            # Manage existing position
            if position_open:
                # Exit if hold-period passed OR z below exit threshold
                hold_days = (ts - datetime.strptime(open_trade["entry_date"], "%Y-%m-%d")).days
                if z < Z_EXIT or hold_days >= HOLD_DAYS:
                    exit_info = fetch_contract_close(
                        con, open_trade["symbol"],
                        open_trade["entry_date"],
                        ts.strftime("%Y-%m-%d"),
                    )
                    if exit_info is not None:
                        exit_date, exit_px = exit_info
                        # Short put: gain when premium decays
                        gross_pnl = open_trade["entry_premium"] - exit_px
                        # Scale to %-of-capital using max-loss = strike − premium
                        max_loss = max(open_trade["strike"] - open_trade["entry_premium"], 1.0)
                        pnl_pct = RISK_PER_TRADE * gross_pnl / max_loss
                        win = gross_pnl > 0
                        trades.append(TradeRecord(
                            entry_date=open_trade["entry_date"],
                            exit_date=exit_date,
                            expiration=open_trade["expiration"],
                            strike=open_trade["strike"],
                            entry_premium=open_trade["entry_premium"],
                            exit_premium=exit_px,
                            skew=open_trade["skew"],
                            skew_z=open_trade["skew_z"],
                            pnl_pct=float(pnl_pct),
                            win=bool(win),
                        ))
                        # Spread the OOS P&L over the holding period for daily metrics
                        try:
                            entry_dt = pd.Timestamp(open_trade["entry_date"])
                            exit_dt = pd.Timestamp(exit_date)
                            window = pd.date_range(entry_dt, exit_dt, freq="D")
                            if len(window) > 0:
                                per_day = pnl_pct / len(window)
                                in_idx = daily_pnl.index.intersection(window)
                                daily_pnl.loc[in_idx] += per_day
                        except Exception:
                            pass
                        fold_pnl += pnl_pct
                        if win:
                            wins += 1
                        n_test_trades += 1
                    position_open = False
                    open_trade = None

            # Open a new position
            if not position_open and z > Z_ENTRY:
                position_open = True
                open_trade = {
                    "entry_date": row["snapshot"],
                    "expiration": row["expiration"],
                    "strike": row["twentyfive_strike"],
                    "entry_premium": row["twentyfive_premium"],
                    "symbol": row["twentyfive_symbol"],
                    "skew": row["skew"],
                    "skew_z": float(z),
                }

        # Close any open position at end of fold using last known close
        if position_open and open_trade is not None:
            exit_info = fetch_contract_close(
                con, open_trade["symbol"],
                open_trade["entry_date"],
                test.index[-1].strftime("%Y-%m-%d"),
            )
            if exit_info is not None:
                exit_date, exit_px = exit_info
                gross_pnl = open_trade["entry_premium"] - exit_px
                max_loss = max(open_trade["strike"] - open_trade["entry_premium"], 1.0)
                pnl_pct = RISK_PER_TRADE * gross_pnl / max_loss
                win = gross_pnl > 0
                trades.append(TradeRecord(
                    entry_date=open_trade["entry_date"],
                    exit_date=exit_date,
                    expiration=open_trade["expiration"],
                    strike=open_trade["strike"],
                    entry_premium=open_trade["entry_premium"],
                    exit_premium=exit_px,
                    skew=open_trade["skew"],
                    skew_z=open_trade["skew_z"],
                    pnl_pct=float(pnl_pct),
                    win=bool(win),
                ))
                fold_pnl += pnl_pct
                if win:
                    wins += 1
                n_test_trades += 1
                try:
                    entry_dt = pd.Timestamp(open_trade["entry_date"])
                    exit_dt = pd.Timestamp(exit_date)
                    window = pd.date_range(entry_dt, exit_dt, freq="D")
                    if len(window) > 0:
                        per_day = pnl_pct / len(window)
                        in_idx = daily_pnl.index.intersection(window)
                        daily_pnl.loc[in_idx] += per_day
                except Exception:
                    pass

        folds.append(FoldResult(
            train_start=str(train.index[0].date()),
            train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()),
            test_end=str(test.index[-1].date()),
            n_obs_train=len(train),
            n_obs_test=len(test),
            n_trades=n_test_trades,
            pnl=float(fold_pnl),
            win_rate=float(wins / n_test_trades) if n_test_trades > 0 else 0.0,
        ))
        start += STEP_OBS

    return trades, daily_pnl, folds


# ── Metrics ────────────────────────────────────────────────────────────


def aggregate_metrics(daily_pnl: pd.Series, trades: List[TradeRecord]) -> Dict[str, float]:
    r = daily_pnl.dropna()
    nz = r[r != 0]
    eq = (1.0 + r).cumprod()
    n_days = int(len(r))
    if len(nz) < 2:
        return dict(n_days=n_days, n_trades=len(trades), n_active_days=int(len(nz)),
                    cagr=0.0, sharpe=0.0, max_dd=0.0, vol=0.0,
                    win_rate=0.0, total_return=0.0)
    years = n_days / 365
    cagr = float(eq.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    pk = eq.cummax()
    max_dd = float(((eq - pk) / pk).min())
    vol = float(r.std() * math.sqrt(252))
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0
    wr = float(np.mean([t.win for t in trades])) if trades else 0.0
    return dict(
        n_days=n_days,
        n_trades=len(trades),
        n_active_days=int(len(nz)),
        cagr=cagr,
        sharpe=sharpe,
        max_dd=max_dd,
        vol=vol,
        win_rate=wr,
        total_return=float(eq.iloc[-1] - 1.0),
    )


def load_exp1220_yearly() -> Dict[int, float]:
    if not os.path.exists(EXP1220_SUMMARY):
        return {}
    with open(EXP1220_SUMMARY) as f:
        data = json.load(f)
    out: Dict[int, float] = {}
    for y, blob in data.get("yearly", {}).items():
        try:
            out[int(y)] = float(blob["protected"]["return_pct"]) / 100.0
        except (KeyError, TypeError, ValueError):
            continue
    return out


def correlate_yearly(daily: pd.Series, exp1220: Dict[int, float]) -> Optional[float]:
    if not exp1220:
        return None
    yearly = daily.groupby(daily.index.year).apply(
        lambda r: float((1.0 + r).prod() - 1.0)
    ).to_dict()
    common = sorted(set(yearly) & set(exp1220))
    if len(common) < 3:
        return None
    a = np.array([yearly[y] for y in common], dtype=float)
    b = np.array([exp1220[y] for y in common], dtype=float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ── Report ─────────────────────────────────────────────────────────────


def _fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x*100:.{dp}f}%" if np.isfinite(x) else "—"


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(metrics: Dict[str, float], folds: List[FoldResult],
                trades: List[TradeRecord], correlation: Optional[float],
                exp1220: Dict[int, float], snap_summary: Dict[str, float]) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1200px;color:#111}
    h1{border-bottom:3px solid #5b1c4a}
    h2{margin-top:2em;color:#5b1c4a}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#5b1c4a;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#5b1c4a}
    """
    h: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-1960 SPY Put-Skew Alpha</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-1960 — SPY Put-Skew Alpha</h1>",
        "<p class='muted'>25-delta-put IV vs ATM-put IV mean reversion. "
        "Real IronVault SPY put closes inverted to IV via Black-Scholes. "
        "Walk-forward 2020-2025, weekly observation cadence.</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault option prices</span></p>",
    ]

    # Snapshot summary
    h.append("<h2>Skew snapshot universe</h2>")
    h.append("<table><tr><th>Field</th><th>Value</th></tr>"
             f"<tr><td class='l'>Observations used</td><td>{snap_summary.get('n_obs', 0):,}</td></tr>"
             f"<tr><td class='l'>Date range</td><td class='l'>{snap_summary.get('first', '—')} → {snap_summary.get('last', '—')}</td></tr>"
             f"<tr><td class='l'>Mean DTE</td><td>{snap_summary.get('mean_dte', 0):.1f}</td></tr>"
             f"<tr><td class='l'>Mean ATM IV</td><td>{snap_summary.get('mean_iv_atm', 0)*100:.2f}%</td></tr>"
             f"<tr><td class='l'>Mean 25Δ IV</td><td>{snap_summary.get('mean_iv_25d', 0)*100:.2f}%</td></tr>"
             f"<tr><td class='l'>Mean skew (25Δ−ATM)</td><td>{snap_summary.get('mean_skew', 0)*100:.2f}%</td></tr>"
             f"<tr><td class='l'>Skew σ</td><td>{snap_summary.get('std_skew', 0)*100:.2f}%</td></tr>"
             "</table>")

    # Headline metrics
    m = metrics
    h.append("<h2>Walk-forward strategy results</h2>")
    h.append("<table><tr><th>n trades</th><th>Win rate</th><th>CAGR</th>"
             "<th>Sharpe</th><th>Vol</th><th>Max DD</th><th>Total return</th>"
             "<th>Active days</th><th>Corr vs EXP-1220</th></tr>"
             f"<tr><td>{m['n_trades']}</td>"
             f"<td>{_fmt_pct(m['win_rate'], 1)}</td>"
             f"<td class='{ 'pos' if m['cagr']>0 else 'neg' }'>{_fmt_pct(m['cagr'])}</td>"
             f"<td>{_fmt(m['sharpe'])}</td>"
             f"<td>{_fmt_pct(m['vol'])}</td>"
             f"<td class='neg'>{_fmt_pct(m['max_dd'])}</td>"
             f"<td class='{ 'pos' if m['total_return']>0 else 'neg' }'>{_fmt_pct(m['total_return'])}</td>"
             f"<td>{m['n_active_days']}</td>"
             f"<td>{(f'{correlation:+.2f}' if correlation is not None else 'n/a')}</td></tr></table>")

    # Targets
    target_sharpe = m["sharpe"] >= 2.0
    target_corr = (correlation is not None and abs(correlation) < 0.30)
    h.append("<h3>Targets</h3>"
             "<table><tr><th>Target</th><th>Required</th><th>Actual</th><th>Pass</th></tr>"
             f"<tr><td class='l'>Sharpe</td><td>≥ 2.0</td><td>{_fmt(m['sharpe'])}</td>"
             f"<td class='{ 'pos' if target_sharpe else 'neg' }'>{ 'YES' if target_sharpe else 'NO' }</td></tr>"
             f"<tr><td class='l'>|Corr vs EXP-1220|</td><td>&lt; 0.30</td>"
             f"<td>{(f'{correlation:+.2f}' if correlation is not None else 'n/a')}</td>"
             f"<td class='{ 'pos' if target_corr else 'neg' }'>{ 'YES' if target_corr else 'NO' }</td></tr>"
             "</table>")

    # Folds
    h.append("<h2>Walk-forward folds</h2>")
    h.append("<table><tr><th>Train</th><th>Test</th><th># trades</th>"
             "<th>Win rate</th><th>P&L</th></tr>")
    for f in folds:
        cls = "pos" if f.pnl > 0 else ("neg" if f.pnl < 0 else "")
        h.append(
            f"<tr><td class='l'>{f.train_start} → {f.train_end}</td>"
            f"<td class='l'>{f.test_start} → {f.test_end}</td>"
            f"<td>{f.n_trades}</td>"
            f"<td>{_fmt_pct(f.win_rate, 1)}</td>"
            f"<td class='{cls}'>{_fmt_pct(f.pnl)}</td></tr>"
        )
    h.append("</table>")

    # EXP-1220 reference
    if exp1220:
        h.append("<h2>EXP-1220 yearly protected returns (reference)</h2>")
        h.append("<table><tr>" + "".join(f"<th>{y}</th>" for y in sorted(exp1220)) + "</tr><tr>")
        for y in sorted(exp1220):
            v = exp1220[y]
            cls = "pos" if v > 0 else "neg"
            h.append(f"<td class='{cls}'>{_fmt_pct(v)}</td>")
        h.append("</tr></table>")

    # Trade tape
    h.append(f"<h2>Trade tape (first/last 20 of {len(trades)})</h2>")
    h.append("<table><tr><th>Entry</th><th>Exit</th><th>Expiry</th><th>K</th>"
             "<th>Entry $</th><th>Exit $</th><th>Skew</th><th>z</th>"
             "<th>P&L %</th><th>Win</th></tr>")
    show = trades[:20] + ([None] if len(trades) > 40 else []) + trades[-20:] if len(trades) > 40 else trades
    for t in show:
        if t is None:
            h.append("<tr><td class='l' colspan='10'>…</td></tr>")
            continue
        cls = "pos" if t.pnl_pct > 0 else "neg"
        h.append(
            f"<tr><td class='l'>{t.entry_date}</td><td class='l'>{t.exit_date}</td>"
            f"<td class='l'>{t.expiration}</td><td>{t.strike:.0f}</td>"
            f"<td>{t.entry_premium:.2f}</td><td>{t.exit_premium:.2f}</td>"
            f"<td>{t.skew*100:+.2f}%</td><td>{t.skew_z:+.2f}</td>"
            f"<td class='{cls}'>{_fmt_pct(t.pnl_pct, 3)}</td>"
            f"<td>{'Y' if t.win else 'N'}</td></tr>"
        )
    h.append("</table>")

    # Methodology
    h.append("<h2>Methodology & honest caveats</h2>")
    h.append("<ul>")
    h.append("<li><b>Data:</b> SPY put closes from "
             "<code>data/options_cache.db</code> (IronVault). Spot from "
             "Yahoo Finance. No synthetic prices.</li>")
    h.append("<li><b>IV inversion:</b> Brent's method on the BS put price. "
             "Inverting a real market price into σ is standard textbook "
             "practice and is allowed under Rule Zero — no fills are "
             "fabricated, only the σ that reproduces the real close.</li>")
    h.append("<li><b>Skew measure:</b> "
             "skew = σ(strike with Δ closest to −0.25) − σ(strike closest to spot). "
             "Δ is computed from the same σ that was inverted from the real "
             "price, so this is a self-consistent vol-surface read.</li>")
    h.append("<li><b>Trading rule:</b> short the 25-Δ put when skew z &gt; +1.5, "
             "exit at min(14d, expiration) OR z &lt; −0.5. Position size = "
             f"{RISK_PER_TRADE*100:.1f}% of capital per trade, computed against "
             "the strike − premium max-loss of a naked-put structure.</li>")
    h.append("<li><b>Walk-forward:</b> 52 weekly obs train / 13 obs OOS / step 13. "
             "The z-score thresholds are FIXED (Carlos's spec), not grid-searched, "
             "so there is no per-fold parameter overfitting. The train window only "
             "controls when the rolling z-score is allowed to start firing.</li>")
    h.append("<li><b>What this is:</b> a real-options test of a real-options "
             "hypothesis. Entry premium and exit premium are both literal closes "
             "from <code>option_daily</code>; P&amp;L is the difference, scaled to a "
             "risk-percent of capital. The Sharpe and CAGR you see are from real "
             "fills, not from a proxy.</li>")
    h.append("<li><b>What this is NOT:</b> a fully-hedged structure. A naked "
             "short put has unlimited downside in a crash; a production version "
             "would convert to a put credit spread, which would tighten the max "
             "loss but also reduce the premium harvested. The Sharpe number "
             "above should therefore be read as the *raw skew alpha*, before "
             "any structural overlay is bolted on.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("[exp1960] loading SPY spot…", flush=True)
    spy = fetch_spy_close()
    print(f"[exp1960] SPY: {len(spy)} days {spy.index[0].date()} → {spy.index[-1].date()}")

    print("[exp1960] opening IronVault DB…", flush=True)
    con = sqlite3.connect(DB_PATH)
    try:
        snapshots_df = build_skew_history(con, spy)
        if len(snapshots_df) < TRAIN_OBS + TEST_OBS:
            print(f"[exp1960] insufficient snapshots ({len(snapshots_df)})")
            return 1

        snap_summary = {
            "n_obs": int(len(snapshots_df)),
            "first": str(snapshots_df.index[0].date()),
            "last": str(snapshots_df.index[-1].date()),
            "mean_dte": float(snapshots_df["dte"].mean()),
            "mean_iv_atm": float(snapshots_df["iv_atm"].mean()),
            "mean_iv_25d": float(snapshots_df["iv_25d"].mean()),
            "mean_skew": float(snapshots_df["skew"].mean()),
            "std_skew": float(snapshots_df["skew"].std(ddof=0)),
        }
        print(f"[exp1960] mean ATM IV={snap_summary['mean_iv_atm']*100:.2f}%  "
              f"mean 25ΔIV={snap_summary['mean_iv_25d']*100:.2f}%  "
              f"mean skew={snap_summary['mean_skew']*100:.2f}%")

        print("[exp1960] running walk-forward backtest…", flush=True)
        trades, daily_pnl, folds = backtest_skew(con, snapshots_df)
        print(f"[exp1960] {len(trades)} trades, "
              f"{int(np.sum([t.win for t in trades]))} wins")

        metrics = aggregate_metrics(daily_pnl, trades)
        print(f"[exp1960] CAGR={metrics['cagr']*100:.2f}%  "
              f"Sharpe={metrics['sharpe']:.2f}  "
              f"DD={metrics['max_dd']*100:.2f}%  "
              f"WR={metrics['win_rate']*100:.1f}%")

        exp1220 = load_exp1220_yearly()
        correlation = correlate_yearly(daily_pnl, exp1220)
        print(f"[exp1960] corr vs EXP-1220: {correlation}")

        html = render_html(metrics, folds, trades, correlation, exp1220, snap_summary)
        out_html = os.path.join(REPORT_DIR, "exp1960_skew_alpha.html")
        with open(out_html, "w") as f:
            f.write(html)
        print(f"[exp1960] wrote {out_html}")

        out_json = os.path.join(REPORT_DIR, "exp1960_skew_alpha.json")
        summary = {
            "experiment": "EXP-1960",
            "tag": "EXP-1960",
            "description": "SPY put-skew (25Δ − ATM IV) mean reversion — real IronVault data",
            "data_sources": {
                "spot": "Yahoo Finance SPY daily close",
                "options": "IronVault data/options_cache.db (real SPY put closes)",
                "iv_method": "Black-Scholes inversion via Brent's method",
            },
            "config": {
                "target_dte": TARGET_DTE,
                "hold_days": HOLD_DAYS,
                "obs_freq_days": OBS_FREQ_DAYS,
                "risk_free": RISK_FREE,
                "z_window": ZSCORE_WINDOW,
                "z_entry": Z_ENTRY,
                "z_exit": Z_EXIT,
                "train_obs": TRAIN_OBS,
                "test_obs": TEST_OBS,
                "step_obs": STEP_OBS,
                "risk_per_trade": RISK_PER_TRADE,
            },
            "snapshot_summary": snap_summary,
            "metrics": metrics,
            "corr_vs_exp1220": correlation,
            "exp1220_yearly_protected_return": exp1220,
            "targets": {
                "sharpe_min": 2.0,
                "abs_corr_max": 0.30,
                "sharpe_pass": metrics["sharpe"] >= 2.0,
                "corr_pass": (correlation is not None and abs(correlation) < 0.30),
            },
            "n_folds": len(folds),
            "fold_pnl": [f.pnl for f in folds],
            "fold_trades": [f.n_trades for f in folds],
            "fold_win_rate": [f.win_rate for f in folds],
            "n_trades": len(trades),
            "n_wins": int(np.sum([t.win for t in trades])) if trades else 0,
        }
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[exp1960] wrote {out_json}")
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
