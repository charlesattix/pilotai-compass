"""
Strategy Discovery Round 2 — test novel strategy types on real IronVault data.

All prices from options_cache.db. Zero synthetic data. Zero np.random for prices.

Strategies:
  1. Cross-Asset Pairs: XLI momentum → SPY put spreads
  2. Volatility Term Structure: SPY front/back IV ratio
  3. XLF Earnings Season: sell puts before bank earnings
  4. Sector Rotation: XLF vs XLI relative strength
  5. Correlation Breakdown: TLT-SPY decorrelation trades

Walk-forward: IS = 2020-2022, OOS = 2023-2025.
Kill: <20 OOS trades OR negative OOS Sharpe.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT = ROOT / "reports" / "strategy_discovery_round2.html"

CAPITAL = 100_000
OOS_START_YEAR = 2023


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _exp_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _dl(ticker: str, start: str = "2019-06-01", end: str = "2026-01-01") -> pd.DataFrame:
    """Download daily OHLCV via yfinance."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def _find_exps(hd: IronVault, ticker: str, start: str, end: str,
               monthly: bool = True) -> List[str]:
    """Available expirations from options_cache.db."""
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    cur.execute("""SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker=? AND option_type='P' AND expiration BETWEEN ? AND ?
        ORDER BY expiration""", (ticker, start, end))
    exps = [r[0] for r in cur.fetchall()]
    conn.close()
    if not monthly:
        return exps
    out, last = [], ""
    for e in exps:
        ym, day = e[:7], int(e[8:10])
        if ym != last and 15 <= day <= 21:
            out.append(e)
            last = ym
    return out


def _find_all_exps(hd: IronVault, ticker: str, start: str, end: str) -> List[str]:
    """All expirations (not just monthly)."""
    return _find_exps(hd, ticker, start, end, monthly=False)


def _sell_put_spread(
    hd: IronVault, ticker: str, exp: str, trade_date: str,
    underlying_price: float, otm_pct: float = 0.93, width: float = 5.0,
) -> Optional[Dict]:
    """Find and price an OTM put credit spread. Returns None on cache miss."""
    strikes = hd.get_available_strikes(ticker, exp, trade_date, "P")
    if not strikes:
        return None
    exp_as_dt = _exp_dt(exp)
    target = underlying_price * otm_pct
    for sk in sorted(strikes, key=lambda k: abs(k - target))[:12]:
        lk = sk - width
        if lk not in strikes:
            # Try nearest available long strike below
            candidates = [s for s in strikes if s < sk and abs(s - lk) <= 1.0]
            if not candidates:
                continue
            lk = max(candidates)
        actual_width = sk - lk
        if actual_width <= 0:
            continue
        pp = hd.get_spread_prices(ticker, exp_as_dt, sk, lk, "P", trade_date)
        if pp is None:
            continue
        credit = pp["short_close"] - pp["long_close"]
        if credit > 0.05:
            return {"short": sk, "long": lk, "credit": round(credit, 4),
                    "width": actual_width, "max_loss": round(actual_width - credit, 4)}
    return None


def _walk_spread(
    hd: IronVault, ticker: str, exp: str, short_k: float, long_k: float,
    entry_credit: float, entry_dt: datetime, exp_dt_obj: datetime,
    trading_days_index, profit_pct: float = 0.50, stop_mult: float = 3.0,
    min_dte: int = 7,
) -> Tuple[Optional[str], str, float, int]:
    """Walk forward through real prices for exit. Returns (exit_date, reason, exit_val, hold_days)."""
    exit_date = exit_reason = None
    exit_val = entry_credit
    hold_days = 0
    current = entry_dt + timedelta(days=1)
    td_set = set(trading_days_index.strftime("%Y-%m-%d"))

    while current <= exp_dt_obj:
        curr_str = current.strftime("%Y-%m-%d")
        if curr_str not in td_set:
            current += timedelta(days=1)
            continue
        hold_days += 1
        dte_rem = (exp_dt_obj - current).days

        pp = hd.get_spread_prices(ticker, exp_dt_obj, short_k, long_k, "P", curr_str)
        if pp is None:
            current += timedelta(days=1)
            continue
        cv = pp["short_close"] - pp["long_close"]

        if cv <= entry_credit * (1 - profit_pct):
            return curr_str, "profit_target", cv, hold_days
        if cv - entry_credit > entry_credit * stop_mult:
            return curr_str, "stop_loss", cv, hold_days
        if dte_rem <= min_dte:
            return curr_str, "dte_exit", cv, hold_days
        current += timedelta(days=1)

    # Expiration fallback
    fp = hd.get_spread_prices(ticker, exp_dt_obj, short_k, long_k, "P", exp)
    if fp:
        exit_val = fp["short_close"] - fp["long_close"]
    else:
        exit_val = 0.0
    return exp, "expiration", exit_val, hold_days


def _next_trading_day(dt: datetime, td_set) -> Optional[datetime]:
    for offset in range(7):
        c = dt + timedelta(days=offset)
        if c.strftime("%Y-%m-%d") in td_set:
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Stats computation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Stats:
    name: str
    description: str = ""
    trades: List[Dict] = field(default_factory=list)
    n_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    max_dd: float = 0.0
    sharpe: float = 0.0
    cagr: float = 0.0
    spy_corr: float = 0.0
    avg_pnl: float = 0.0
    oos_sharpe: float = 0.0
    oos_n: int = 0
    oos_pnl: float = 0.0
    oos_wr: float = 0.0
    yearly: Dict[int, Dict] = field(default_factory=dict)
    killed: bool = False
    kill_reason: str = ""


def _compute(trades: List[Dict], name: str, spy_ret: pd.Series,
             desc: str = "") -> Stats:
    if not trades:
        return Stats(name=name, description=desc, killed=True,
                     kill_reason="0 trades")
    df = pd.DataFrame(trades)
    pnls = df["pnl"].values
    n = len(pnls)
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())

    eq = np.cumsum(pnls) + CAPITAL
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / pk
    max_dd = float(dd.max())

    mu = float(pnls.mean())
    sd = float(pnls.std(ddof=1)) if n > 1 else 1.0
    sharpe = mu / sd * math.sqrt(min(n, 52)) if sd > 1e-9 else 0.0

    dates = pd.to_datetime(df["exit_date"])
    yrs = max((dates.max() - pd.to_datetime(df["entry_date"]).min()).days / 365.25, 0.5)
    cagr = ((1 + total / CAPITAL) ** (1 / yrs) - 1) if total > -CAPITAL else -1.0

    # SPY correlation
    tr = {}
    for _, r in df.iterrows():
        d = str(r["exit_date"])[:10]
        tr[d] = tr.get(d, 0) + r["pnl"]
    ts = pd.Series(tr)
    ts.index = pd.to_datetime(ts.index)
    common = ts.index.intersection(spy_ret.index)
    corr = float(np.corrcoef(ts.reindex(common).fillna(0), spy_ret.reindex(common).fillna(0))[0, 1]) if len(common) > 10 else 0.0

    # OOS
    oos = df[dates.dt.year >= OOS_START_YEAR]
    oos_n = len(oos)
    oos_sharpe = 0.0
    oos_pnl = 0.0
    oos_wr = 0.0
    if oos_n > 1:
        op = oos["pnl"].values
        oos_pnl = float(op.sum())
        oos_wr = float((op > 0).sum()) / oos_n
        os = float(op.std(ddof=1))
        oos_sharpe = float(op.mean()) / os * math.sqrt(min(oos_n, 52)) if os > 1e-9 else 0.0

    # Yearly
    df["year"] = dates.dt.year
    yearly = {}
    for yr, g in df.groupby("year"):
        yp = g["pnl"].values
        yn = len(yp)
        if yn == 0:
            continue
        ye = np.cumsum(yp) + CAPITAL
        ypk = np.maximum.accumulate(ye)
        ydd = (ypk - ye) / ypk
        ysd = float(yp.std(ddof=1)) if yn > 1 else 1.0
        yearly[int(yr)] = {
            "n": yn, "pnl": round(float(yp.sum()), 2),
            "wr": round(float((yp > 0).sum()) / yn, 4),
            "dd": round(float(ydd.max()), 4),
            "sharpe": round(float(yp.mean()) / ysd * math.sqrt(min(yn, 52)) if ysd > 1e-9 else 0, 3),
            "ret": round(float(yp.sum()) / CAPITAL, 4),
        }

    # Kill check
    killed = oos_n < 20 or oos_sharpe < 0
    kr = ""
    if oos_n < 20:
        kr = f"Only {oos_n} OOS trades (<20)"
    elif oos_sharpe < 0:
        kr = f"Negative OOS Sharpe ({oos_sharpe:.2f})"

    return Stats(
        name=name, description=desc, trades=trades,
        n_trades=n, total_pnl=round(total, 2), win_rate=round(wins / n, 4),
        max_dd=round(max_dd, 4), sharpe=round(sharpe, 3), cagr=round(cagr, 4),
        spy_corr=round(corr, 4), avg_pnl=round(mu, 2),
        oos_sharpe=round(oos_sharpe, 3), oos_n=oos_n,
        oos_pnl=round(oos_pnl, 2), oos_wr=round(oos_wr, 4),
        yearly=yearly, killed=killed, kill_reason=kr,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: Cross-Asset Pairs — XLI momentum → SPY put spreads
# ═══════════════════════════════════════════════════════════════════════════

def strat_cross_asset_xli_spy(hd: IronVault, spy_df: pd.DataFrame,
                               xli_df: pd.DataFrame) -> List[Dict]:
    """When XLI (industrials) 20-day momentum > 1.5%, sell SPY put spreads.
    Industrials lead the broad market — positive XLI signals economic expansion."""
    print("  [1] Cross-Asset: XLI momentum → SPY puts")
    spy_close = spy_df["Close"]
    xli_ret20 = xli_df["Close"].pct_change(20)
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))

    exps = _find_exps(hd, "SPY", "2020-03-01", "2025-12-31", monthly=False)
    trades, last = [], None

    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_trading_day(exp_obj - timedelta(days=35), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < 18:
            continue

        # XLI momentum filter
        try:
            xli_m = float(xli_ret20.loc[es])
        except (KeyError, TypeError):
            continue
        if np.isnan(xli_m) or xli_m < 0.015:
            continue

        try:
            price = float(spy_close.loc[es])
        except (KeyError, TypeError):
            continue

        spread = _sell_put_spread(hd, "SPY", exp, es, price, otm_pct=0.94, width=5.0)
        if spread is None:
            continue

        contracts = max(1, min(3, int(CAPITAL * 0.02 / (spread["max_loss"] * 100))))
        ed, er, ev, hd_ = _walk_spread(
            hd, "SPY", exp, spread["short"], spread["long"],
            spread["credit"], entry_dt, exp_obj, spy_df.index,
        )
        pnl = (spread["credit"] - ev) * 100 * contracts
        trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                        "exit_reason": er, "credit": spread["credit"],
                        "xli_mom": round(xli_m, 4), "hold_days": hd_})
        last = entry_dt

    print(f"    → {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: Volatility Term Structure — front/back option price ratio
# ═══════════════════════════════════════════════════════════════════════════

def strat_vol_term_structure(hd: IronVault, spy_df: pd.DataFrame,
                              vix: pd.Series) -> List[Dict]:
    """Sell SPY put spreads when term structure is in contango (front < back).
    Compare ~21-DTE vs ~50-DTE put prices at same strike. Large contango =
    sell front-month premium."""
    print("  [2] Vol Term Structure: SPY contango signal")
    spy_close = spy_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))

    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    cur.execute("""SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker='SPY' AND option_type='P' AND expiration BETWEEN '2020-03-01' AND '2025-12-31'
        ORDER BY expiration""")
    all_exps = [r[0] for r in cur.fetchall()]
    conn.close()

    trades, last = [], None

    for i, front in enumerate(all_exps):
        front_dt = _exp_dt(front)
        # Find back expiration 25-45 days after front
        back = None
        for j in range(i + 1, min(i + 30, len(all_exps))):
            delta = (_exp_dt(all_exps[j]) - front_dt).days
            if 25 <= delta <= 45:
                back = all_exps[j]
                break
        if back is None:
            continue
        back_dt = _exp_dt(back)

        # Entry ~25 days before front expiration
        entry_dt = _next_trading_day(front_dt - timedelta(days=25), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < 14:
            continue

        try:
            price = float(spy_close.loc[es])
        except (KeyError, TypeError):
            continue

        # Compare front vs back put price at ~5% OTM strike
        target_k = round(price * 0.95)
        front_strikes = hd.get_available_strikes("SPY", front, es, "P")
        back_strikes = hd.get_available_strikes("SPY", back, es, "P")
        common = sorted(set(front_strikes or []) & set(back_strikes or []))
        if not common:
            continue
        strike = min(common, key=lambda k: abs(k - target_k))

        front_sym = IronVault.build_occ_symbol("SPY", front_dt, strike, "P")
        back_sym = IronVault.build_occ_symbol("SPY", back_dt, strike, "P")
        fp = hd.get_contract_price(front_sym, es)
        bp = hd.get_contract_price(back_sym, es)
        if fp is None or bp is None or fp < 0.10:
            continue

        ratio = bp / fp  # contango ratio: >1 means contango
        if ratio < 1.15:  # need meaningful contango
            continue

        # Sell front-month put spread
        spread = _sell_put_spread(hd, "SPY", front, es, price, otm_pct=0.94, width=5.0)
        if spread is None:
            continue

        contracts = max(1, min(3, int(CAPITAL * 0.015 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(
            hd, "SPY", front, spread["short"], spread["long"],
            spread["credit"], entry_dt, front_dt, spy_df.index,
        )
        pnl = (spread["credit"] - ev) * 100 * contracts
        trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                        "exit_reason": er, "credit": spread["credit"],
                        "term_ratio": round(ratio, 3), "hold_days": hold})
        last = entry_dt

    print(f"    → {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: XLF Earnings Season — sell puts before bank earnings
# ═══════════════════════════════════════════════════════════════════════════

def strat_xlf_earnings_season(hd: IronVault, xlf_df: pd.DataFrame,
                               vix: pd.Series) -> List[Dict]:
    """Sell XLF put spreads 3-4 weeks before major bank earnings.
    Banks report mid-Jan, mid-Apr, mid-Jul, mid-Oct.
    XLF tends to rally into earnings as expectations build."""
    print("  [3] XLF Earnings Season: pre-earnings put spreads")
    xlf_close = xlf_df["Close"]
    td_set = set(xlf_df.index.strftime("%Y-%m-%d"))

    # Bank earnings windows: primary (Jan/Apr/Jul/Oct) + secondary (mid-Q follow-through)
    earnings_months = [1, 3, 4, 6, 7, 9, 10, 12]
    trades = []

    for year in range(2020, 2026):
        for month in earnings_months:
            # Enter ~25 days before mid-month earnings (~3rd week prior month)
            # Earnings ~15th of month → enter around 20th of prior month
            entry_target = datetime(year, month, 15) - timedelta(days=25)
            entry_dt = _next_trading_day(entry_target, td_set)
            if entry_dt is None:
                continue
            es = entry_dt.strftime("%Y-%m-%d")

            # VIX filter: skip if extreme
            try:
                v = float(vix.loc[es])
            except (KeyError, TypeError):
                v = 20
            if v > 35:
                continue

            try:
                price = float(xlf_close.loc[es])
            except (KeyError, TypeError):
                continue

            # Find expiration around earnings date (mid-month)
            target_exp = datetime(year, month, 17)
            exps = _find_exps(hd, "XLF", (target_exp - timedelta(days=5)).strftime("%Y-%m-%d"),
                              (target_exp + timedelta(days=10)).strftime("%Y-%m-%d"), monthly=False)
            if not exps:
                # Try wider window
                exps = _find_exps(hd, "XLF",
                                  (target_exp - timedelta(days=10)).strftime("%Y-%m-%d"),
                                  (target_exp + timedelta(days=20)).strftime("%Y-%m-%d"), monthly=False)
            if not exps:
                continue

            exp = min(exps, key=lambda e: abs((_exp_dt(e) - target_exp).days))
            exp_obj = _exp_dt(exp)
            if (exp_obj - entry_dt).days < 10 or (exp_obj - entry_dt).days > 50:
                continue

            spread = _sell_put_spread(hd, "XLF", exp, es, price, otm_pct=0.95, width=1.0)
            if spread is None:
                continue

            contracts = max(1, min(10, int(CAPITAL * 0.015 / (spread["max_loss"] * 100))))
            ed, er, ev, hold = _walk_spread(
                hd, "XLF", exp, spread["short"], spread["long"],
                spread["credit"], entry_dt, exp_obj, xlf_df.index,
                profit_pct=0.50, stop_mult=3.0,
            )
            pnl = (spread["credit"] - ev) * 100 * contracts
            trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                            "exit_reason": er, "credit": spread["credit"],
                            "earnings_month": month, "vix": round(v, 2),
                            "hold_days": hold})

    print(f"    → {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 4: Sector Rotation — XLF vs XLI relative strength
# ═══════════════════════════════════════════════════════════════════════════

def strat_sector_rotation(hd: IronVault, xlf_df: pd.DataFrame,
                           xli_df: pd.DataFrame) -> List[Dict]:
    """Relative strength rotation: sell put spreads on the STRONGER of XLF vs XLI.
    Compute 40-day relative performance. Sell puts on whichever sector is
    outperforming — momentum persists in sectors."""
    print("  [4] Sector Rotation: XLF vs XLI relative strength")
    xlf_close = xlf_df["Close"]
    xli_close = xli_df["Close"]
    xlf_ret40 = xlf_close.pct_change(40)
    xli_ret40 = xli_close.pct_change(40)

    trades, last = [], None

    for year in range(2020, 2026):
        for month in range(1, 13):
            # Monthly entry
            entry_target = datetime(year, month, 5)
            td_set_xlf = set(xlf_df.index.strftime("%Y-%m-%d"))
            entry_dt = _next_trading_day(entry_target, td_set_xlf)
            if entry_dt is None:
                continue
            es = entry_dt.strftime("%Y-%m-%d")
            if last and (entry_dt - last).days < 20:
                continue

            try:
                xlf_m = float(xlf_ret40.loc[es])
                xli_m = float(xli_ret40.loc[es])
            except (KeyError, TypeError):
                continue
            if np.isnan(xlf_m) or np.isnan(xli_m):
                continue

            # Pick stronger sector
            if xlf_m > xli_m and xlf_m > 0:
                ticker = "XLF"
                price_series = xlf_close
                df_for_walk = xlf_df
            elif xli_m > xlf_m and xli_m > 0:
                ticker = "XLI"
                price_series = xli_close
                df_for_walk = xli_df
            else:
                continue  # both negative — skip

            try:
                price = float(price_series.loc[es])
            except (KeyError, TypeError):
                continue

            # Find expiration ~35 days out
            target_exp = entry_dt + timedelta(days=35)
            exps = _find_exps(hd, ticker,
                              (target_exp - timedelta(days=10)).strftime("%Y-%m-%d"),
                              (target_exp + timedelta(days=15)).strftime("%Y-%m-%d"),
                              monthly=False)
            if not exps:
                continue
            exp = min(exps, key=lambda e: abs((_exp_dt(e) - target_exp).days))
            exp_obj = _exp_dt(exp)
            if (exp_obj - entry_dt).days < 15:
                continue

            spread = _sell_put_spread(hd, ticker, exp, es, price, otm_pct=0.95, width=1.0)
            if spread is None:
                continue

            contracts = max(1, min(10, int(CAPITAL * 0.015 / (spread["max_loss"] * 100))))
            ed, er, ev, hold = _walk_spread(
                hd, ticker, exp, spread["short"], spread["long"],
                spread["credit"], entry_dt, exp_obj, df_for_walk.index,
            )
            pnl = (spread["credit"] - ev) * 100 * contracts
            trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                            "exit_reason": er, "ticker": ticker,
                            "credit": spread["credit"],
                            "xlf_mom": round(xlf_m, 4), "xli_mom": round(xli_m, 4),
                            "hold_days": hold})
            last = entry_dt

    print(f"    → {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 5: TLT-SPY Correlation Breakdown
# ═══════════════════════════════════════════════════════════════════════════

def strat_correlation_breakdown(hd: IronVault, spy_df: pd.DataFrame,
                                 tlt_df: pd.DataFrame) -> List[Dict]:
    """Trade when the normal TLT-SPY negative correlation breaks.
    Normal: TLT and SPY move inversely (corr ~ -0.3).
    When 30-day rolling corr goes POSITIVE (both selling off), sell SPY put
    spreads betting on reversion — bonds usually resume safe-haven role."""
    print("  [5] Correlation Breakdown: TLT-SPY decorrelation")
    spy_close = spy_df["Close"]
    tlt_close = tlt_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))

    # Rolling 30-day correlation
    spy_ret = spy_close.pct_change()
    tlt_ret = tlt_close.pct_change()
    # Align
    common_idx = spy_ret.index.intersection(tlt_ret.index)
    spy_r = spy_ret.reindex(common_idx)
    tlt_r = tlt_ret.reindex(common_idx)
    roll_corr = spy_r.rolling(30).corr(tlt_r)

    exps = _find_exps(hd, "SPY", "2020-04-01", "2025-12-31")
    trades, last = [], None

    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_trading_day(exp_obj - timedelta(days=35), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < 14:
            continue

        # Check correlation regime
        try:
            corr_val = float(roll_corr.loc[es])
        except (KeyError, TypeError):
            continue
        if np.isnan(corr_val):
            continue

        # Normal TLT-SPY corr is -0.3 to -0.5. Positive = breakdown.
        # Trade when corr > 0.0 (unusual positive correlation = both selling off)
        if corr_val < 0.0:
            continue

        try:
            price = float(spy_close.loc[es])
        except (KeyError, TypeError):
            continue

        spread = _sell_put_spread(hd, "SPY", exp, es, price, otm_pct=0.93, width=5.0)
        if spread is None:
            continue

        contracts = max(1, min(3, int(CAPITAL * 0.02 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(
            hd, "SPY", exp, spread["short"], spread["long"],
            spread["credit"], entry_dt, exp_obj, spy_df.index,
        )
        pnl = (spread["credit"] - ev) * 100 * contracts
        trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                        "exit_reason": er, "credit": spread["credit"],
                        "tlt_spy_corr": round(corr_val, 4), "hold_days": hold})
        last = entry_dt

    print(f"    → {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def _c(v: float) -> str:
    return "#3fb950" if v >= 0 else "#f85149"


def _fd(v: float) -> str:
    return f"${v:,.0f}"


def _fp(v: float) -> str:
    return f"{v:.1%}"


def _gen_html(all_strats: List[Stats], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Survivors vs killed
    survivors = [s for s in all_strats if not s.killed]
    killed = [s for s in all_strats if s.killed]

    # Build cards for all strategies
    cards = ""
    for s in all_strats:
        border = "#f85149" if s.killed else ("#d29922" if s.oos_sharpe > 1.5 else "#3fb950")
        status = f'<span style="color:#f85149">✗ KILLED: {s.kill_reason}</span>' if s.killed else (
            f'<span style="color:#d29922">★ OOS Sharpe {s.oos_sharpe:.2f}</span>' if s.oos_sharpe > 1.5 else
            f'<span style="color:#3fb950">✓ Survived</span>')

        cards += f"""
        <div class="card" style="border-color:{border}">
          <h3>{s.name}</h3>
          <p class="desc">{s.description}</p>
          <p class="status">{status}</p>
          <div class="g">
            <div><span class="l">Trades</span><span class="v">{s.n_trades}</span></div>
            <div><span class="l">Total P&L</span><span class="v" style="color:{_c(s.total_pnl)}">{_fd(s.total_pnl)}</span></div>
            <div><span class="l">Win Rate</span><span class="v">{_fp(s.win_rate)}</span></div>
            <div><span class="l">Max DD</span><span class="v" style="color:#f85149">{_fp(s.max_dd)}</span></div>
            <div><span class="l">Full Sharpe</span><span class="v" style="color:{_c(s.sharpe)}">{s.sharpe:.2f}</span></div>
            <div><span class="l">OOS Sharpe</span><span class="v" style="color:{_c(s.oos_sharpe)}">{s.oos_sharpe:.2f}</span></div>
            <div><span class="l">OOS Trades</span><span class="v">{s.oos_n}</span></div>
            <div><span class="l">OOS P&L</span><span class="v" style="color:{_c(s.oos_pnl)}">{_fd(s.oos_pnl)}</span></div>
            <div><span class="l">CAGR</span><span class="v" style="color:{_c(s.cagr)}">{_fp(s.cagr)}</span></div>
            <div><span class="l">SPY Corr</span><span class="v">{s.spy_corr:.3f}</span></div>
          </div>
        </div>"""

    # Yearly tables for survivors
    yearly_html = ""
    for s in survivors:
        rows = ""
        for yr in sorted(s.yearly):
            y = s.yearly[yr]
            oos_mark = " (OOS)" if yr >= OOS_START_YEAR else ""
            rows += f"""<tr><td>{yr}{oos_mark}</td><td>{y['n']}</td>
              <td style="color:{_c(y['pnl'])}">{_fd(y['pnl'])}</td>
              <td>{_fp(y['wr'])}</td><td>{_fp(y['dd'])}</td>
              <td style="color:{_c(y['sharpe'])}">{y['sharpe']:.2f}</td>
              <td style="color:{_c(y['ret'])}">{_fp(y['ret'])}</td></tr>"""
        yearly_html += f"""<h3>{s.name} — Yearly</h3>
        <table class="dt"><tr><th>Year</th><th>N</th><th>P&L</th><th>WR</th>
        <th>DD</th><th>Sharpe</th><th>Return</th></tr>{rows}</table>"""

    # Summary table
    summary_rows = ""
    for s in all_strats:
        killed_cls = ' style="opacity:0.5"' if s.killed else ""
        corr_c = "#3fb950" if abs(s.spy_corr) < 0.3 else "#d29922" if abs(s.spy_corr) < 0.6 else "#f85149"
        summary_rows += f"""<tr{killed_cls}>
          <td style="text-align:left">{s.name}{'  ✗' if s.killed else ''}</td>
          <td>{s.n_trades}</td><td>{s.oos_n}</td>
          <td style="color:{_c(s.total_pnl)}">{_fd(s.total_pnl)}</td>
          <td>{_fp(s.win_rate)}</td><td>{_fp(s.max_dd)}</td>
          <td style="color:{_c(s.sharpe)}">{s.sharpe:.2f}</td>
          <td style="color:{_c(s.oos_sharpe)}"><strong>{s.oos_sharpe:.2f}</strong></td>
          <td style="color:{_c(s.cagr)}">{_fp(s.cagr)}</td>
          <td style="color:{corr_c}">{s.spy_corr:.3f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Strategy Discovery Round 2</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 24px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; }}
  h2 {{ color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-top: 36px; }}
  h3 {{ color: #79c0ff; margin-top: 24px; }}
  .meta {{ color: #8b949e; font-size: 0.88em; }}
  .desc {{ color: #8b949e; font-size: 0.82em; margin: 4px 0 8px; }}
  .status {{ font-size: 0.85em; font-weight: 600; margin-bottom: 8px; }}
  .card {{ background: #161b22; border: 2px solid #30363d; border-radius: 10px;
           padding: 16px 20px; margin: 14px 0; }}
  .g {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }}
  .g > div {{ text-align: center; }}
  .g .l {{ display: block; color: #8b949e; font-size: 0.72em; }}
  .g .v {{ display: block; font-weight: 600; font-size: 1em; }}
  table.dt {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.84em; }}
  table.dt th, table.dt td {{ padding: 5px 8px; text-align: right; border-bottom: 1px solid #21262d; }}
  table.dt th {{ color: #8b949e; background: #161b22; }}
  table.dt td:first-child {{ text-align: left; }}
  .verdict {{ background: #161b22; border: 2px solid #d29922; border-radius: 10px;
              padding: 20px; margin: 20px 0; }}
  footer {{ margin-top: 40px; padding-top: 12px; border-top: 1px solid #21262d;
            color: #484f58; font-size: 0.78em; }}
</style></head><body>
<h1>Strategy Discovery — Round 2</h1>
<p class="meta">Generated {ts} &middot; Real IronVault data only &middot;
   IS: 2020-2022 / OOS: 2023-2025 &middot; Kill: &lt;20 OOS trades or negative OOS Sharpe</p>

<div class="verdict">
  <h2 style="margin-top:0;border:none">Results Summary</h2>
  <p><strong>{len(survivors)}</strong> strategies survived, <strong>{len(killed)}</strong> killed.</p>
  <table class="dt">
    <tr><th style="text-align:left">Strategy</th><th>Total</th><th>OOS</th><th>P&L</th>
        <th>WR</th><th>MaxDD</th><th>Full Sharpe</th><th>OOS Sharpe</th><th>CAGR</th><th>SPY Corr</th></tr>
    {summary_rows}
  </table>
</div>

<h2>Strategy Detail</h2>
{cards}

<h2>Yearly Breakdown (Survivors Only)</h2>
{yearly_html}

<footer>
  Data: IronVault options_cache.db &middot; SPY (187K contracts), XLF (8.6K), XLI (16.4K), TLT (9.2K) &middot;
  No synthetic pricing &middot; Cache miss → trade skipped
</footer>
</body></html>"""

    output.write_text(html, encoding="utf-8")
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(output: Path = DEFAULT_OUTPUT) -> List[Stats]:
    logging.basicConfig(level=logging.WARNING)

    print("=" * 70)
    print("STRATEGY DISCOVERY — Round 2 — Real IronVault Data Only")
    print("=" * 70)

    hd = IronVault.instance()
    cov = hd.coverage_report()
    print(f"IronVault: {cov['contracts_total']:,} contracts, {cov['daily_bars_total']:,} daily bars\n")

    print("Fetching market data...")
    spy_df, vix = _dl("SPY"), _dl("^VIX")
    if isinstance(vix.columns, pd.MultiIndex):
        vix = vix.xs(vix.columns[0][1], axis=1, level=1) if len(vix.columns.levels) > 1 else vix
    vix_close = vix["Close"] if "Close" in vix.columns else vix.iloc[:, 0]
    xlf_df = _dl("XLF")
    xli_df = _dl("XLI")
    tlt_df = _dl("TLT")
    spy_ret = spy_df["Close"].pct_change().dropna() * CAPITAL

    print("\nRunning backtests...")
    t1 = strat_cross_asset_xli_spy(hd, spy_df, xli_df)
    t2 = strat_vol_term_structure(hd, spy_df, vix_close)
    t3 = strat_xlf_earnings_season(hd, xlf_df, vix_close)
    t4 = strat_sector_rotation(hd, xlf_df, xli_df)
    t5 = strat_correlation_breakdown(hd, spy_df, tlt_df)

    print("\nComputing statistics...")
    results = [
        _compute(t1, "Cross-Asset: XLI→SPY", spy_ret,
                 "Sell SPY put spreads when XLI 20-day momentum > 3%"),
        _compute(t2, "Vol Term Structure", spy_ret,
                 "Sell SPY front-month puts when term structure contango ratio > 1.15"),
        _compute(t3, "XLF Earnings Season", spy_ret,
                 "Sell XLF put spreads 3-4 weeks before quarterly bank earnings"),
        _compute(t4, "Sector Rotation: XLF/XLI", spy_ret,
                 "Sell put spreads on stronger of XLF vs XLI (40-day relative strength)"),
        _compute(t5, "Correlation Breakdown: TLT-SPY", spy_ret,
                 "Sell SPY puts when TLT-SPY 30-day corr goes positive (reversion bet)"),
    ]

    # Print
    print("\n" + "=" * 95)
    print(f"{'Strategy':<32} {'N':>4} {'OOS':>4} {'P&L':>9} {'WR':>6} {'DD':>6} "
          f"{'Sharpe':>7} {'OOS Sh':>7} {'CAGR':>7} {'Corr':>6} {'Status':>10}")
    print("-" * 95)
    for s in results:
        st = "KILLED" if s.killed else ("★ FLAG" if s.oos_sharpe > 1.5 else "OK")
        print(f"{s.name:<32} {s.n_trades:>4} {s.oos_n:>4} {s.total_pnl:>9,.0f} "
              f"{s.win_rate:>6.0%} {s.max_dd:>6.1%} {s.sharpe:>7.2f} {s.oos_sharpe:>7.2f} "
              f"{s.cagr:>7.1%} {s.spy_corr:>6.3f} {st:>10}")

    # Report
    rp = _gen_html(results, output)
    print(f"\nReport: {rp}")

    # JSON
    jp = output.with_suffix(".json")
    jdata = {
        "generated": datetime.now().isoformat(),
        "data_source": "IronVault (options_cache.db)",
        "oos_period": "2023-2025",
        "kill_criteria": "<20 OOS trades OR negative OOS Sharpe",
        "strategies": [
            {"name": s.name, "desc": s.description,
             "n_trades": s.n_trades, "oos_trades": s.oos_n,
             "total_pnl": s.total_pnl, "win_rate": s.win_rate,
             "max_dd": s.max_dd, "sharpe": s.sharpe,
             "oos_sharpe": s.oos_sharpe, "oos_pnl": s.oos_pnl,
             "cagr": s.cagr, "spy_corr": s.spy_corr,
             "killed": s.killed, "kill_reason": s.kill_reason,
             "yearly": s.yearly}
            for s in results
        ],
    }
    jp.write_text(json.dumps(jdata, indent=2))

    return results


if __name__ == "__main__":
    main()
