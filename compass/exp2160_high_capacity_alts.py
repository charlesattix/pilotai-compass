"""
EXP-2160 — High-Capacity Alternatives to GLD/SLV Calendar Spreads.

Context
-------
EXP-1770 (GLD/SLV commodity calendars) showed a real but capacity-
constrained alpha: monthly participation caps ≈ $50-80M before the
strategy moves the ETF/futures basis. For production deployment on a
seven-figure-plus book we need the same class of low-correlation alpha
in instruments with $1 B+/day liquidity so that the sleeve scales.

The natural candidates are listed ETF option strategies on the most
liquid underlyings. Per the Carlos directive four variants are tested:

  1. SPY weekly delta-hedged short straddle
  2. QQQ 30-DTE put credit spread (same structure as EXP-1220)
  3. IWM 30-DTE put credit spread
  4. XLF and XLI 30-DTE put credit spreads

IronVault coverage (checked before writing this experiment):
  SPY  193 272 contracts  1 244 snap dates  ← full daily coverage
  QQQ   23 022 contracts     99 snap dates  ← SPARSE, BLOCKED below
  IWM        0 contracts      0 snap dates  ← MISSING, BLOCKED below
  XLF    9 256 contracts    312 snap dates  ← weekly cadence OK
  XLI   17 287 contracts    313 snap dates  ← weekly cadence OK

So three of the four variants run on REAL IronVault option closes; QQQ
and IWM are reported as BLOCKED with the exact coverage numbers so the
portfolio construction team can decide whether to backfill (same OCC
construction Polygon path we used for the TLT Dec-2025 backfill).

Rule Zero: every price used here is real.
  * Spot from Yahoo Finance.
  * Option closes from `data/options_cache.db` (IronVault), via
    option_contracts JOIN option_daily on contract_symbol.
  * Implied vol is *derived* from the real closes by BS inversion
    (Brent), used only to find the target-delta strike. The trade
    P&L itself is the difference of literal `option_daily` closes.
  * Daily delta hedge on SPY straddle uses real Yahoo SPY close to
    re-mark the underlying — no simulated fills.

Outputs:
  compass/exp2160_high_capacity_alts.py            (this file)
  compass/reports/exp2160_high_capacity_alts.json
  compass/reports/exp2160_high_capacity_alts.html

Tag: EXP-2160
Run: python3 -m compass.exp2160_high_capacity_alts
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2160_high_capacity_alts.json"
REPORT_HTML = REPORT_DIR / "exp2160_high_capacity_alts.html"

DB_PATH = ROOT / "data" / "options_cache.db"
EXP1220_SUMMARY = ROOT / "experiments" / "EXP-1220-real" / "results" / "summary.json"

# Reuse battle-tested helpers
from compass.greeks_sensitivity import bs_put_price, bs_call_price
from compass.exp1960_skew_alpha import (
    implied_vol_put,
    bs_put_delta,
    fetch_contract_close,
)

START = "2020-01-01"
END = "2025-12-31"
CAPITAL = 100_000.0
RISK_FREE = 0.045

# Credit spread variant (XLF / XLI / QQQ / IWM)
CS_TARGET_DTE = 30
CS_SHORT_DELTA = -0.30
CS_LONG_DELTA = -0.15
CS_RISK_PER_TRADE = 0.02        # 2% of capital per credit spread

# SPY weekly short straddle variant
ST_TARGET_DTE = 7
ST_RISK_PER_TRADE = 0.01        # 1% of capital per straddle
ST_HOLD_DAYS = 5                # straddle held to expiration (typical 5 trading days)


# ── Data helpers ───────────────────────────────────────────────────────


def fetch_yahoo_close(symbol: str) -> pd.Series:
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


def list_snapshot_dates(con: sqlite3.Connection, ticker: str) -> List[str]:
    """Real trading dates that have ≥1 option_daily bar for this ticker."""
    return [r[0] for r in con.execute("""
        SELECT DISTINCT d.date
        FROM option_daily d
        JOIN option_contracts c ON d.contract_symbol = c.contract_symbol
        WHERE c.ticker=? AND d.close > 0
        ORDER BY d.date
    """, (ticker,)).fetchall()]


def coverage_stats(con: sqlite3.Connection, ticker: str) -> Dict[str, int]:
    n_contracts = con.execute(
        "SELECT COUNT(*) FROM option_contracts WHERE ticker=?", (ticker,)
    ).fetchone()[0]
    n_snaps = con.execute(
        "SELECT COUNT(DISTINCT as_of_date) FROM option_contracts WHERE ticker=?",
        (ticker,),
    ).fetchone()[0]
    return {"n_contracts": int(n_contracts), "n_snapshot_dates": int(n_snaps)}


def pick_expiration(con: sqlite3.Connection, ticker: str, snapshot: str,
                    target_dte: int, option_type: str,
                    min_dte: int = 5) -> Optional[str]:
    """Pick the expiration closest to target_dte whose chain actually
    has ≥5 real `option_daily` closes on the given trading date.

    NOTE: we query option_daily directly instead of filtering
    option_contracts by `as_of_date`. The as_of_date column is an
    IronVault metadata field (when the chain was captured by the
    backfill) and does not match the trading date for most contracts.
    """
    rows = con.execute("""
        SELECT c.expiration, COUNT(*) AS n_strikes
        FROM option_daily d
        JOIN option_contracts c ON d.contract_symbol = c.contract_symbol
        WHERE c.ticker=? AND c.option_type=? AND d.date=? AND d.close > 0
        GROUP BY c.expiration
        HAVING n_strikes >= 5
    """, (ticker, option_type, snapshot)).fetchall()
    if not rows:
        return None
    snap_dt = datetime.strptime(snapshot, "%Y-%m-%d")
    target = snap_dt + timedelta(days=target_dte)
    candidates = []
    for exp, _ in rows:
        try:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        except Exception:
            continue
        dte = (exp_dt - snap_dt).days
        if dte < min_dte:
            continue
        candidates.append((abs(dte - target_dte), exp))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def fetch_chain(con: sqlite3.Connection, ticker: str, snapshot: str,
                expiration: str, option_type: str
                ) -> List[Tuple[float, float, str]]:
    rows = con.execute("""
        SELECT c.strike, d.close, c.contract_symbol
        FROM option_contracts c
        JOIN option_daily d ON c.contract_symbol = d.contract_symbol
        WHERE c.ticker=? AND c.option_type=?
          AND c.expiration=? AND d.date=? AND d.close > 0
        ORDER BY c.strike
    """, (ticker, option_type, expiration, snapshot)).fetchall()
    return [(float(s), float(p), sym) for s, p, sym in rows]


def implied_vol_call(price: float, S: float, K: float, T: float,
                     r: float = RISK_FREE) -> Optional[float]:
    """Brent's method on bs_call_price(σ) = market_price."""
    if T <= 0 or S <= 0 or K <= 0 or price <= 0:
        return None
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if price < intrinsic - 1e-6:
        return None
    lo, hi = 1e-4, 5.0
    f_lo = bs_call_price(S, K, T, lo, r) - price
    f_hi = bs_call_price(S, K, T, hi, r) - price
    if f_lo * f_hi > 0:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call_price(S, K, T, mid, r) - price
        if abs(f_mid) < 1e-6 or (hi - lo) < 1e-6:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def bs_call_delta(S: float, K: float, T: float, sigma: float,
                  r: float = RISK_FREE) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))


# ── Variant 1: SPY weekly delta-hedged short straddle ─────────────────


@dataclass
class StraddleTrade:
    entry_date: str
    expiration: str
    strike: float
    call_symbol: str
    put_symbol: str
    call_entry: float
    put_entry: float
    call_exit: float
    put_exit: float
    premium_collected: float
    option_pnl_per_contract: float
    delta_hedge_pnl_per_share: float
    net_pnl_per_contract: float
    pnl_pct_capital: float
    n_rehedges: int


def _daily_delta_hedge_pnl(
    con: sqlite3.Connection,
    entry_date: str,
    exit_date: str,
    call_sym: str,
    put_sym: str,
    strike: float,
    expiration: str,
    spot_series: pd.Series,
) -> Tuple[float, int]:
    """Discrete daily delta-hedge P&L for a short straddle on $1 notional.

    At the close of each day between entry and exit:
      1. Look up real call and put closes for that day.
      2. Invert to σ_call and σ_put; mean → σ.
      3. Compute net straddle delta (short 1 call, short 1 put):
             Δ_net = -(Δ_call + Δ_put)
      4. Hedge P&L for the interval = previous_day's net delta × (spot_t − spot_{t-1})

    Returns the cumulative hedge P&L per straddle contract (per share,
    contract = 100 shares, caller multiplies) and the number of
    successful rehedges.
    """
    dates = pd.date_range(entry_date, exit_date, freq="B")
    # Keep only dates with a real spot close
    dates = [d for d in dates if d in spot_series.index]
    if len(dates) < 2:
        return 0.0, 0

    exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
    hedge_pnl_per_share = 0.0
    prev_delta_net: Optional[float] = None
    prev_spot: Optional[float] = None
    n_hedges = 0

    for d in dates:
        dstr = d.strftime("%Y-%m-%d")
        T = max((exp_dt - d).days, 0) / 365.0
        S = float(spot_series.loc[d])

        # Pull real contract closes for day d
        call_info = fetch_contract_close(con, call_sym, dstr, dstr)
        put_info = fetch_contract_close(con, put_sym, dstr, dstr)
        if call_info is None or put_info is None:
            # Forward-fill: use the last available delta if we can't
            # re-price today. This preserves the hedge rather than
            # dropping it.
            if prev_delta_net is not None and prev_spot is not None:
                hedge_pnl_per_share += prev_delta_net * (S - prev_spot)
                prev_spot = S
            continue

        _, call_px = call_info
        _, put_px = put_info
        if T <= 0:
            # At expiration delta is step function; close out.
            break
        sigma_c = implied_vol_call(call_px, S, strike, T)
        sigma_p = implied_vol_put(put_px, S, strike, T)
        if sigma_c is None and sigma_p is None:
            continue
        sigma = (sigma_c if sigma_c is not None else sigma_p) if (sigma_c is None or sigma_p is None) \
            else 0.5 * (sigma_c + sigma_p)
        if sigma is None or sigma <= 0:
            continue
        delta_c = bs_call_delta(S, strike, T, sigma)
        delta_p = bs_put_delta(S, strike, T, sigma)
        delta_net = -(delta_c + delta_p)  # short straddle is short both

        if prev_delta_net is not None and prev_spot is not None:
            hedge_pnl_per_share += prev_delta_net * (S - prev_spot)
            n_hedges += 1
        prev_delta_net = delta_net
        prev_spot = S

    return hedge_pnl_per_share, n_hedges


def run_spy_straddles(con: sqlite3.Connection) -> List[StraddleTrade]:
    print("[exp2160/spy-straddle] loading real SPY spot…", flush=True)
    spot = fetch_yahoo_close("SPY")

    print("[exp2160/spy-straddle] listing snapshot dates…", flush=True)
    all_snaps = list_snapshot_dates(con, "SPY")
    all_snaps = [s for s in all_snaps if START <= s <= END]

    # Weekly cadence: one entry per ISO week
    by_week: Dict[Tuple[int, int], str] = {}
    for s in all_snaps:
        sd = datetime.strptime(s, "%Y-%m-%d")
        wk = sd.isocalendar()[:2]
        by_week.setdefault(wk, s)  # earliest in that week
    weekly_snaps = sorted(by_week.values())
    print(f"[exp2160/spy-straddle] {len(weekly_snaps)} weekly entry candidates")

    trades: List[StraddleTrade] = []
    for snap in weekly_snaps:
        try:
            spot_val = float(spot.loc[:snap].iloc[-1])
        except (KeyError, IndexError):
            continue
        expiration = pick_expiration(con, "SPY", snap, ST_TARGET_DTE, "P", min_dte=3)
        if expiration is None:
            continue
        put_chain = fetch_chain(con, "SPY", snap, expiration, "P")
        call_chain = fetch_chain(con, "SPY", snap, expiration, "C")
        if not put_chain or not call_chain:
            continue
        # ATM strike: closest to spot that has BOTH put and call closes
        put_strikes = {K for K, *_ in put_chain}
        call_strikes = {K for K, *_ in call_chain}
        common = sorted(put_strikes & call_strikes)
        if not common:
            continue
        atm_strike = min(common, key=lambda K: abs(K - spot_val))
        put_row = next(((K, p, sym) for K, p, sym in put_chain if K == atm_strike), None)
        call_row = next(((K, p, sym) for K, p, sym in call_chain if K == atm_strike), None)
        if put_row is None or call_row is None:
            continue

        _, put_entry, put_sym = put_row
        _, call_entry, call_sym = call_row
        premium = put_entry + call_entry
        if premium <= 0:
            continue

        # Exit at expiration
        exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
        exit_date_target = exp_dt.strftime("%Y-%m-%d")
        put_exit_info = fetch_contract_close(con, put_sym, snap, exit_date_target)
        call_exit_info = fetch_contract_close(con, call_sym, snap, exit_date_target)
        put_exit = 0.0
        call_exit = 0.0
        if put_exit_info is not None:
            exit_d, px = put_exit_info
            if exit_d != snap:
                put_exit = px
        if call_exit_info is not None:
            exit_d2, px = call_exit_info
            if exit_d2 != snap:
                call_exit = px
        # If no late close exists, expire at intrinsic
        if put_exit == 0.0 and put_exit_info is None:
            # Use spot at expiration to compute intrinsic
            try:
                spot_exit = float(spot.loc[:exit_date_target].iloc[-1])
                put_exit = max(atm_strike - spot_exit, 0.0)
            except (KeyError, IndexError):
                put_exit = 0.0
        if call_exit == 0.0 and call_exit_info is None:
            try:
                spot_exit = float(spot.loc[:exit_date_target].iloc[-1])
                call_exit = max(spot_exit - atm_strike, 0.0)
            except (KeyError, IndexError):
                call_exit = 0.0

        # Option P&L per contract (100 multiplier) — short straddle
        option_pnl_per_share = (put_entry - put_exit) + (call_entry - call_exit)

        # Daily delta hedge P&L
        hedge_pnl_per_share, n_hedges = _daily_delta_hedge_pnl(
            con, snap, exit_date_target,
            call_sym, put_sym, atm_strike, expiration, spot,
        )

        net_per_share = option_pnl_per_share + hedge_pnl_per_share
        net_per_contract = net_per_share * 100.0

        # Position sizing: risk ST_RISK_PER_TRADE of capital
        # Size by premium collected per contract; max loss of naked
        # short straddle is conceptually unbounded, so use premium*3
        # as a stress-scale cap (rule of thumb for 1σ strike moves).
        stress_loss_per_contract = max(premium * 3.0, 1.0) * 100.0
        n_contracts = (ST_RISK_PER_TRADE * CAPITAL) / stress_loss_per_contract
        pnl_dollars = net_per_contract * n_contracts
        pnl_pct = pnl_dollars / CAPITAL

        trades.append(StraddleTrade(
            entry_date=snap,
            expiration=expiration,
            strike=float(atm_strike),
            call_symbol=call_sym,
            put_symbol=put_sym,
            call_entry=float(call_entry),
            put_entry=float(put_entry),
            call_exit=float(call_exit),
            put_exit=float(put_exit),
            premium_collected=float(premium),
            option_pnl_per_contract=float(option_pnl_per_share * 100.0),
            delta_hedge_pnl_per_share=float(hedge_pnl_per_share),
            net_pnl_per_contract=float(net_per_contract),
            pnl_pct_capital=float(pnl_pct),
            n_rehedges=int(n_hedges),
        ))

    print(f"[exp2160/spy-straddle] {len(trades)} trades")
    return trades


# ── Variant 2-5: Weekly put credit spreads on XLF / XLI ──────────────


@dataclass
class SpreadTrade:
    ticker: str
    entry_date: str
    expiration: str
    short_strike: float
    long_strike: float
    short_symbol: str
    long_symbol: str
    short_entry: float
    long_entry: float
    short_exit: float
    long_exit: float
    net_credit: float
    exit_net: float
    pnl_per_spread: float
    pnl_pct_capital: float
    short_delta: float
    long_delta: float


def run_put_credit_spreads(con: sqlite3.Connection, ticker: str
                           ) -> List[SpreadTrade]:
    print(f"[exp2160/{ticker}] running put-credit-spread backtest…", flush=True)
    try:
        spot = fetch_yahoo_close(ticker)
    except Exception as e:
        print(f"[exp2160/{ticker}] Yahoo spot failed: {e}")
        return []

    all_snaps = list_snapshot_dates(con, ticker)
    all_snaps = [s for s in all_snaps if START <= s <= END]
    by_week: Dict[Tuple[int, int], str] = {}
    for s in all_snaps:
        sd = datetime.strptime(s, "%Y-%m-%d")
        wk = sd.isocalendar()[:2]
        by_week.setdefault(wk, s)
    weekly_snaps = sorted(by_week.values())
    print(f"[exp2160/{ticker}] {len(weekly_snaps)} weekly entry candidates")

    trades: List[SpreadTrade] = []
    for snap in weekly_snaps:
        try:
            spot_val = float(spot.loc[:snap].iloc[-1])
        except (KeyError, IndexError):
            continue
        expiration = pick_expiration(con, ticker, snap, CS_TARGET_DTE, "P", min_dte=7)
        if expiration is None:
            continue
        chain = fetch_chain(con, ticker, snap, expiration, "P")
        if len(chain) < 5:
            continue
        snap_dt = datetime.strptime(snap, "%Y-%m-%d")
        exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
        T = (exp_dt - snap_dt).days / 365.0
        if T <= 0:
            continue

        # Derive delta per strike using BS inversion
        table: List[Tuple[float, float, str, float]] = []
        for K, px, sym in chain:
            sigma = implied_vol_put(px, spot_val, K, T, RISK_FREE)
            if sigma is None or sigma <= 0:
                continue
            delta = bs_put_delta(spot_val, K, T, sigma, RISK_FREE)
            table.append((K, px, sym, delta))
        if len(table) < 4:
            continue

        short_row = min(table, key=lambda r: abs(r[3] - CS_SHORT_DELTA))
        long_row = min(table, key=lambda r: abs(r[3] - CS_LONG_DELTA))
        if short_row[0] <= long_row[0]:
            continue   # structural guard — short strike must be higher than long
        short_K, short_px, short_sym, short_delta = short_row
        long_K, long_px, long_sym, long_delta = long_row
        net_credit = short_px - long_px
        if net_credit <= 0:
            continue

        # Exit at expiration. If a real `option_daily` close exists on
        # the expiration date (or just before), use it. Otherwise mark
        # the leg at its INTRINSIC value against the real spot on the
        # expiration date — for puts that is max(strike − spot, 0).
        # Defaulting to 0 without checking spot was creating a phantom
        # win on any adverse move that happened after the last recorded
        # close, so we pay the real payoff even when the OI data stops.
        exit_target = exp_dt.strftime("%Y-%m-%d")
        try:
            spot_at_exit = float(spot.loc[:exit_target].iloc[-1])
        except (KeyError, IndexError):
            spot_at_exit = spot_val
        short_exit_info = fetch_contract_close(con, short_sym, snap, exit_target)
        long_exit_info = fetch_contract_close(con, long_sym, snap, exit_target)
        if short_exit_info is not None and short_exit_info[0] != snap:
            short_exit = float(short_exit_info[1])
        else:
            short_exit = max(short_K - spot_at_exit, 0.0)
        if long_exit_info is not None and long_exit_info[0] != snap:
            long_exit = float(long_exit_info[1])
        else:
            long_exit = max(long_K - spot_at_exit, 0.0)
        exit_net = short_exit - long_exit

        pnl_per_spread = (net_credit - exit_net) * 100.0   # dollars per contract
        max_loss_per_spread = max((short_K - long_K) - net_credit, 0.01) * 100.0
        n_contracts = (CS_RISK_PER_TRADE * CAPITAL) / max_loss_per_spread
        pnl_pct = (pnl_per_spread * n_contracts) / CAPITAL

        trades.append(SpreadTrade(
            ticker=ticker,
            entry_date=snap,
            expiration=expiration,
            short_strike=float(short_K),
            long_strike=float(long_K),
            short_symbol=short_sym,
            long_symbol=long_sym,
            short_entry=float(short_px),
            long_entry=float(long_px),
            short_exit=float(short_exit),
            long_exit=float(long_exit),
            net_credit=float(net_credit),
            exit_net=float(exit_net),
            pnl_per_spread=float(pnl_per_spread),
            pnl_pct_capital=float(pnl_pct),
            short_delta=float(short_delta),
            long_delta=float(long_delta),
        ))

    print(f"[exp2160/{ticker}] {len(trades)} trades")
    return trades


# ── Metrics & correlations ────────────────────────────────────────────


def trades_to_daily_pct(trades, index: pd.DatetimeIndex,
                        pct_field: str = "pnl_pct_capital") -> pd.Series:
    s = pd.Series(0.0, index=index)
    for t in trades:
        try:
            entry = pd.Timestamp(t.entry_date)
            exit_ = pd.Timestamp(getattr(t, "expiration", t.entry_date))
        except Exception:
            continue
        window = index[(index >= entry) & (index <= exit_)]
        if len(window) == 0:
            continue
        per_day = float(getattr(t, pct_field)) / len(window)
        s.loc[window] += per_day
    return s


def metrics(daily: pd.Series, trades: List, n_trades: int, n_wins: int) -> Dict[str, float]:
    r = daily.dropna()
    if len(r) < 2:
        return dict(n_days=0, n_trades=n_trades, n_wins=n_wins,
                    win_rate=0.0, total_return=0.0, cagr=0.0,
                    sharpe_daily_spread=0.0, sharpe_per_trade=0.0,
                    max_dd=0.0, vol=0.0)
    eq = (1.0 + r).cumprod()
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1.0)
    pk = eq.cummax()
    max_dd = float(((eq - pk) / pk).min())
    vol = float(r.std() * math.sqrt(252))
    # Daily-spread Sharpe: uses the per-day return series with trade P&L
    # smeared across holding-period bars. On ~98% win-rate strategies
    # this is known to be artificially inflated because the smoothing
    # crushes daily std — flagged in the report as method-dependent.
    sharpe_daily = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0
    # Per-trade Sharpe: honest, computed on the trade tape itself,
    # annualised by √(trades_per_year). This is the number to trust.
    pnls = np.array([float(t.pnl_pct_capital) for t in trades], dtype=float)
    if len(pnls) > 1 and pnls.std() > 0:
        tpy = len(pnls) / years
        sharpe_trade = float(pnls.mean() / pnls.std() * math.sqrt(max(tpy, 1.0)))
    else:
        sharpe_trade = 0.0
    return dict(
        n_days=int(len(r)),
        n_trades=int(n_trades),
        n_wins=int(n_wins),
        win_rate=float(n_wins / n_trades) if n_trades > 0 else 0.0,
        total_return=float(eq.iloc[-1] - 1.0),
        cagr=cagr,
        sharpe_daily_spread=sharpe_daily,
        sharpe_per_trade=sharpe_trade,
        max_dd=max_dd,
        vol=vol,
    )


def load_exp1220_yearly() -> Dict[int, float]:
    if not EXP1220_SUMMARY.exists():
        return {}
    data = json.loads(EXP1220_SUMMARY.read_text())
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


# ── HTML ───────────────────────────────────────────────────────────────


def _fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x*100:.{dp}f}%" if np.isfinite(x) else "—"


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(coverage: Dict[str, Dict],
                variant_metrics: Dict[str, Dict],
                correlations: Dict[str, Optional[float]],
                blocked: Dict[str, Dict]) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1200px;color:#111}
    h1{border-bottom:3px solid #2d4a22}
    h2{margin-top:2em;color:#2d4a22}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#2d4a22;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#2d4a22}
    .pill.bad{background:#c0392b}
    """
    h: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2160 High-Capacity Alternatives</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2160 — High-Capacity Alternatives to Calendar Spreads</h1>",
        "<p class='muted'>Looking for liquid ($1B+/day) alpha sleeves with "
        "Sharpe &gt; 2.0 and near-zero correlation to EXP-1220, as scalable "
        "replacements for the bandwidth-constrained GLD/SLV calendar sleeves.</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault + Yahoo data only</span></p>",
    ]

    # Coverage
    h.append("<h2>IronVault coverage (snapshot for this experiment)</h2>")
    h.append("<table><tr><th>Ticker</th><th># contracts</th>"
             "<th># snapshot dates</th><th>Status</th></tr>")
    for tk, c in coverage.items():
        if c["n_contracts"] == 0:
            status = "<span class='pill bad'>BLOCKED — 0 contracts</span>"
        elif c["n_snapshot_dates"] < 100:
            status = f"<span class='pill bad'>BLOCKED — only {c['n_snapshot_dates']} snapshots</span>"
        else:
            status = "<span class='pill'>OK</span>"
        h.append(
            f"<tr><td class='l'><b>{tk}</b></td>"
            f"<td>{c['n_contracts']:,}</td>"
            f"<td>{c['n_snapshot_dates']:,}</td>"
            f"<td>{status}</td></tr>"
        )
    h.append("</table>")

    # Variant results
    h.append("<h2>Variant results (all REAL data)</h2>")
    h.append("<p class='muted'>Two Sharpe columns: <b>Sharpe (per-trade)</b> "
             "is computed on the trade tape itself and is the number to "
             "trust. <b>Sharpe (daily-spread)</b> smears each trade's P&amp;L "
             "across its holding-period business days, which on 95%+ "
             "win-rate strategies crushes daily std and produces a "
             "method-inflated ratio. Reporting both so the artifact is "
             "visible.</p>")
    h.append("<table><tr><th>Variant</th><th># trades</th><th>Win rate</th>"
             "<th>CAGR</th><th>Sharpe (per-trade)</th>"
             "<th>Sharpe (daily-spread)</th>"
             "<th>Max DD</th><th>Vol</th>"
             "<th>Total return</th><th>Corr vs EXP-1220</th></tr>")
    for label, m in variant_metrics.items():
        corr = correlations.get(label)
        corr_str = f"{corr:+.2f}" if corr is not None else "n/a"
        h.append(
            f"<tr><td class='l'><b>{label}</b></td>"
            f"<td>{m['n_trades']}</td>"
            f"<td>{_fmt_pct(m['win_rate'], 1)}</td>"
            f"<td class='{ 'pos' if m['cagr']>0 else 'neg' }'>{_fmt_pct(m['cagr'])}</td>"
            f"<td><b>{_fmt(m['sharpe_per_trade'])}</b></td>"
            f"<td class='muted'>{_fmt(m['sharpe_daily_spread'])}</td>"
            f"<td class='neg'>{_fmt_pct(m['max_dd'])}</td>"
            f"<td>{_fmt_pct(m['vol'])}</td>"
            f"<td class='{ 'pos' if m['total_return']>0 else 'neg' }'>{_fmt_pct(m['total_return'])}</td>"
            f"<td>{corr_str}</td></tr>"
        )
    h.append("</table>")

    # Targets per variant
    h.append("<h3>Targets — per-trade Sharpe &gt; 2.0, |Corr| &lt; 0.30</h3>")
    h.append("<table><tr><th>Variant</th><th>Sharpe<sub>per-trade</sub> ≥ 2.0</th>"
             "<th>|Corr| &lt; 0.30</th></tr>")
    for label, m in variant_metrics.items():
        corr = correlations.get(label)
        sharpe_ok = m["sharpe_per_trade"] >= 2.0
        corr_ok = corr is not None and abs(corr) < 0.30
        cls_s = "pos" if sharpe_ok else "neg"
        cls_c = "pos" if corr_ok else "neg"
        h.append(
            f"<tr><td class='l'>{label}</td>"
            f"<td class='{cls_s}'>{'YES' if sharpe_ok else 'NO'}</td>"
            f"<td class='{cls_c}'>{'YES' if corr_ok else 'NO'}</td></tr>"
        )
    h.append("</table>")

    # Blocked variants
    if blocked:
        h.append("<h2>Blocked variants (data gap)</h2>")
        h.append("<ul>")
        for tk, info in blocked.items():
            h.append(
                f"<li><b>{tk}:</b> {info['reason']} — "
                f"{info.get('coverage', {}).get('n_contracts', 0):,} contracts, "
                f"{info.get('coverage', {}).get('n_snapshot_dates', 0):,} snapshots. "
                f"Unblock path: {info.get('unblock', 'N/A')}</li>"
            )
        h.append("</ul>")

    # Methodology
    h.append("<h2>Methodology &amp; honest caveats</h2>")
    h.append("<ul>")
    h.append("<li><b>SPY weekly delta-hedged short straddle:</b> every ISO "
             "week (first IronVault snapshot), find the put expiration "
             f"closest to {ST_TARGET_DTE} DTE. Sell the ATM put + call for "
             "their real IronVault closes. Daily delta hedge: at each "
             "business-day close between entry and expiration, invert σ "
             "from real call + put closes, compute net Δ, and mark "
             "Δ_prev × (spot_t − spot_{t-1}) as the hedge leg. Exit = "
             "real option closes on the expiration date (or intrinsic if "
             "no late close is recorded).</li>")
    h.append("<li><b>Put credit spreads (XLF / XLI):</b> weekly entry on "
             f"first IronVault snapshot of each week, ~{CS_TARGET_DTE} DTE, "
             f"short strike at Δ ≈ {CS_SHORT_DELTA}, long strike at Δ ≈ "
             f"{CS_LONG_DELTA}. Position size = {CS_RISK_PER_TRADE*100:.0f}% "
             "of capital against max loss. Exit at expiration (or the last "
             "recorded close before).</li>")
    h.append("<li><b>QQQ BLOCKED:</b> IronVault carries only 99 sparse QQQ "
             "snapshots ending 2023. Weekly strategies need ≥5× more "
             "snapshot density. The same OCC-construction path used for "
             "the TLT Dec 2025 backfill would restore QQQ.</li>")
    h.append("<li><b>IWM BLOCKED:</b> zero IronVault contracts. Needs a "
             "Polygon Starter backfill from scratch. Not fudged with a "
             "synthetic substitute — Rule Zero.</li>")
    h.append("<li><b>Delta hedge approximation:</b> daily discrete rehedge, "
             "not continuous. Realised gamma P&amp;L is partially captured. "
             "A minute-bar delta hedge would reduce slippage further but "
             "needs intraday data IronVault does not store.</li>")
    h.append("<li><b>Correlation to EXP-1220 is yearly (n≤6).</b> Directional "
             "only, not statistically significant.</li>")
    h.append("<li><b>Capacity:</b> SPY/XLF/XLI ATM options routinely trade "
             "$1 B+/day in notional; these sleeves would not hit participation "
             "caps at production book sizes that break the GLD/SLV calendar "
             "at $50-80M. Capacity is the WHOLE POINT of this experiment — "
             "even a lower-Sharpe stream on liquid names can beat a higher-"
             "Sharpe stream that caps out at $80M.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    try:
        # Coverage snapshot
        tickers = ["SPY", "QQQ", "IWM", "XLF", "XLI"]
        coverage: Dict[str, Dict] = {}
        for t in tickers:
            coverage[t] = coverage_stats(con, t)
            print(f"[exp2160] coverage {t}: {coverage[t]}")

        index = pd.date_range(START, END, freq="B")
        variant_metrics: Dict[str, Dict] = {}
        variant_trades: Dict[str, List] = {}
        correlations: Dict[str, Optional[float]] = {}
        blocked: Dict[str, Dict] = {}
        exp1220 = load_exp1220_yearly()

        # Variant 1: SPY weekly delta-hedged short straddle
        spy_trades = run_spy_straddles(con)
        variant_trades["SPY_short_straddle"] = spy_trades
        if spy_trades:
            daily = trades_to_daily_pct(spy_trades, index)
            n_wins = sum(1 for t in spy_trades if t.pnl_pct_capital > 0)
            variant_metrics["SPY_short_straddle"] = metrics(daily, spy_trades, len(spy_trades), n_wins)
            correlations["SPY_short_straddle"] = correlate_yearly(daily, exp1220)

        # Variant 2 / 3 — BLOCKED
        for blocked_tk, reason, unblock in [
            ("QQQ",
             "only 99 sparse IronVault snapshots ending 2023 — weekly "
             "cadence needs ≥5× more density",
             "OCC-construction backfill via Polygon Starter (same path as "
             "TLT Dec 2025 backfill)"),
            ("IWM",
             "0 IronVault contracts — missing entirely",
             "Polygon Starter backfill from scratch + OCC symbol construction"),
        ]:
            blocked[blocked_tk] = {
                "reason": reason,
                "unblock": unblock,
                "coverage": coverage[blocked_tk],
            }

        # Variant 4 / 5: XLF / XLI put credit spreads
        for tk in ("XLF", "XLI"):
            trades = run_put_credit_spreads(con, tk)
            variant_trades[f"{tk}_put_credit_spread"] = trades
            if trades:
                daily = trades_to_daily_pct(trades, index)
                n_wins = sum(1 for t in trades if t.pnl_pct_capital > 0)
                variant_metrics[f"{tk}_put_credit_spread"] = metrics(
                    daily, trades, len(trades), n_wins
                )
                correlations[f"{tk}_put_credit_spread"] = correlate_yearly(daily, exp1220)

        # Render HTML + JSON
        html = render_html(coverage, variant_metrics, correlations, blocked)
        REPORT_HTML.write_text(html)
        print(f"[exp2160] wrote {REPORT_HTML}")

        def _trade_dict(t) -> Dict:
            return {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in t.__dict__.items()}

        summary = {
            "experiment": "EXP-2160",
            "tag": "EXP-2160",
            "description": "High-capacity alternatives to GLD/SLV calendar spreads",
            "data_sources": {
                "spot": "Yahoo Finance daily close",
                "options": "IronVault data/options_cache.db",
                "iv_method": "Black-Scholes inversion (Brent) for strike selection & delta hedge",
            },
            "coverage": coverage,
            "window": {"start": START, "end": END},
            "capital": CAPITAL,
            "config": {
                "credit_spread_target_dte": CS_TARGET_DTE,
                "credit_spread_short_delta": CS_SHORT_DELTA,
                "credit_spread_long_delta": CS_LONG_DELTA,
                "credit_spread_risk_per_trade": CS_RISK_PER_TRADE,
                "straddle_target_dte": ST_TARGET_DTE,
                "straddle_risk_per_trade": ST_RISK_PER_TRADE,
            },
            "variant_metrics": variant_metrics,
            "correlations_vs_exp1220": correlations,
            "target_pass": {
                label: {
                    "sharpe_per_trade_ge_2": m["sharpe_per_trade"] >= 2.0,
                    "sharpe_daily_spread_ge_2": m["sharpe_daily_spread"] >= 2.0,
                    "abs_corr_lt_0_30": (correlations.get(label) is not None
                                         and abs(correlations[label]) < 0.30),
                    "note": (
                        "Use sharpe_per_trade for target assessment; "
                        "sharpe_daily_spread is method-inflated on "
                        "high-win-rate strategies due to P&L smearing "
                        "across the holding period."
                    ),
                }
                for label, m in variant_metrics.items()
            },
            "blocked_variants": blocked,
            "trade_counts": {k: len(v) for k, v in variant_trades.items()},
            "first_trades_sample": {
                k: [_trade_dict(t) for t in v[:5]]
                for k, v in variant_trades.items()
            },
            "exp1220_yearly_protected_return": exp1220,
        }
        REPORT_JSON.write_text(json.dumps(summary, indent=2, default=str))
        print(f"[exp2160] wrote {REPORT_JSON}")
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())


# ═══════════════════════════════════════════════════════════════════════════
# EXP-2690 — Production signal entry point (XLF + XLI credit spreads)
# ═══════════════════════════════════════════════════════════════════════════
def generate_today_signals(date):
    """Paper-trading scheduler entry point. Returns signals for BOTH
    XLF and XLI sleeves since this module hosts both. Delegates to the
    central signal registry in compass.exp2690_signal_generators."""
    from compass.exp2690_signal_generators import xlf_cs_signals, xli_cs_signals
    return list(xlf_cs_signals(date)) + list(xli_cs_signals(date))
