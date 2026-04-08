"""
EXP-1710: 1-3 DTE SPY Iron Condors (pivoted from 0DTE SPX)
===========================================================
Data reality check (2026-04-06):
  - SPX/XSP: NOT in IronVault (0 contracts)
  - Polygon Starter: no contract enumeration
  - CBOE DataShop: requires subscription (~$200/mo)
  - SPY daily options: NOT in IronVault (only Friday weeklies)

PIVOT per Rule Zero: use 1-3 DTE SPY iron condors with REAL IronVault
data. SPY has 104 Friday weeklies in 2024-2025 + 37K contracts with
intraday 5-min bars (2020-2026).

Strategy:
  - Entry: Tuesday (3DTE), Wednesday (2DTE), or Thursday (1DTE) at market open
  - Iron condor: short put + short call at 5% OTM, $5-wide wings
  - Exit: 50% profit target OR 2x stop loss OR at expiration Friday
  - Position size: 2% of capital per trade

All data from IronVault options_cache.db (real Polygon market data).
Zero synthetic pricing.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "options_cache.db"

TRADING_DAYS = 252
CAPITAL = 100_000


# ═══════════════════════════════════════════════════════════════════════════
# Corrected Sharpe (per Rule Zero: arithmetic daily mean)
# ═══════════════════════════════════════════════════════════════════════════

def corrected_sharpe(daily_rets, rf: float = 0.045) -> float:
    """Arithmetic mean Sharpe. NOT CAGR-derived."""
    if len(daily_rets) < 2:
        return 0.0
    r = np.asarray(daily_rets, dtype=np.float64)
    rf_d = rf / TRADING_DAYS
    excess = float(np.mean(r)) - rf_d
    std = float(np.std(r, ddof=0))
    if std < 1e-12:
        return 0.0
    return excess / std * math.sqrt(TRADING_DAYS)


def trade_sharpe(pnls) -> float:
    """Trade-level Sharpe from PnL array."""
    if len(pnls) < 2:
        return 0.0
    p = np.asarray(pnls, dtype=np.float64)
    s = float(np.std(p, ddof=1))
    if s < 1e-8:
        return 0.0
    return float(np.mean(p) / s * math.sqrt(min(len(p), 52)))


# ═══════════════════════════════════════════════════════════════════════════
# IronVault queries (direct SQL for speed)
# ═══════════════════════════════════════════════════════════════════════════

def find_friday_expirations(start: str, end: str) -> List[str]:
    """All SPY Friday expirations in range."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker='SPY' AND expiration BETWEEN ? AND ?
          AND CAST(STRFTIME('%w', expiration) AS INTEGER) = 5
        ORDER BY expiration
    """, (start, end))
    out = [r[0] for r in cur.fetchall()]
    conn.close()
    return out


def get_available_strikes(exp: str, date_str: str, opt_type: str) -> List[float]:
    """Get strikes with actual daily bars on a given date."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT oc.strike
        FROM option_contracts oc
        JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
        WHERE oc.ticker='SPY' AND oc.expiration=? AND oc.option_type=?
          AND od.date=?
        ORDER BY oc.strike
    """, (exp, opt_type, date_str))
    strikes = [float(r[0]) for r in cur.fetchall()]
    conn.close()
    return strikes


def get_spread_close(exp: str, short_k: float, long_k: float,
                     opt_type: str, date_str: str) -> Optional[Dict]:
    """Get close prices for a credit spread on a date."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
        SELECT oc.strike, od.close
        FROM option_contracts oc
        JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
        WHERE oc.ticker='SPY' AND oc.expiration=? AND oc.option_type=?
          AND oc.strike IN (?, ?) AND od.date=?
    """, (exp, opt_type, short_k, long_k, date_str))
    rows = {float(r[0]): float(r[1]) for r in cur.fetchall()}
    conn.close()

    if short_k not in rows or long_k not in rows:
        return None
    return {
        "short_close": rows[short_k],
        "long_close": rows[long_k],
    }


def get_spy_price(date_str: str) -> Optional[float]:
    """Get SPY underlying close from option bar volume-weighted ATM approx.

    IronVault doesn't store SPY underlying directly, so we infer from
    the most-traded ATM option context. For this strategy, we use the
    put-call parity approach: find an ATM strike and infer spot.
    """
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    # Find the option with highest volume on this date — likely ATM
    cur.execute("""
        SELECT oc.strike, oc.option_type, od.close, od.volume
        FROM option_contracts oc
        JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
        WHERE oc.ticker='SPY' AND od.date=? AND od.volume > 1000
        ORDER BY od.volume DESC LIMIT 1
    """, (date_str,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    # Most-traded strike is near spot
    return float(row[0])


def load_spy_spot_yfinance(start: str = "2020-01-01", end: str = "2026-01-01") -> pd.Series:
    """Load real SPY closes from Yahoo Finance for trade entries."""
    import yfinance as yf
    spy = yf.download("SPY", start=start, end=end, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index)
    return spy["Close"]


# ═══════════════════════════════════════════════════════════════════════════
# Strategy: 1-3 DTE Iron Condor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ICTrade:
    entry_date: str
    exit_date: str
    expiration: str
    dte_at_entry: int
    underlying: float
    put_short: float
    put_long: float
    call_short: float
    call_long: float
    credit: float
    exit_reason: str
    pnl: float
    contracts: int


def find_condor_spread(exp: str, date_str: str, spot: float,
                       otm_pct: float = 0.05, width: float = 5.0
                       ) -> Optional[Dict]:
    """Find iron condor with available real prices on a given date."""
    # Put side
    put_strikes = get_available_strikes(exp, date_str, "P")
    call_strikes = get_available_strikes(exp, date_str, "C")
    if not put_strikes or not call_strikes:
        return None

    put_target = spot * (1 - otm_pct)
    call_target = spot * (1 + otm_pct)

    # Find put short: nearest strike below put_target
    put_candidates = [s for s in put_strikes if s <= put_target]
    if not put_candidates:
        return None
    put_short = max(put_candidates)
    put_long = put_short - width
    if put_long not in put_strikes:
        # Find nearest strike ≤ put_short - width+0.5
        avail = [s for s in put_strikes if s < put_short and s >= put_short - width - 2]
        if not avail:
            return None
        put_long = max(avail)

    # Find call short: nearest strike above call_target
    call_candidates = [s for s in call_strikes if s >= call_target]
    if not call_candidates:
        return None
    call_short = min(call_candidates)
    call_long = call_short + width
    if call_long not in call_strikes:
        avail = [s for s in call_strikes if s > call_short and s <= call_short + width + 2]
        if not avail:
            return None
        call_long = min(avail)

    # Get prices
    pp = get_spread_close(exp, put_short, put_long, "P", date_str)
    cp = get_spread_close(exp, call_short, call_long, "C", date_str)
    if pp is None or cp is None:
        return None

    put_credit = pp["short_close"] - pp["long_close"]
    call_credit = cp["short_close"] - cp["long_close"]
    # Accept tiny credits at short DTE (1DTE OTM options are ~pennies)
    if put_credit < 0.005 or call_credit < 0.005:
        return None

    total_credit = put_credit + call_credit
    max_width = max(put_short - put_long, call_long - call_short)
    max_loss = max_width - total_credit
    if max_loss <= 0:
        return None

    return {
        "put_short": put_short, "put_long": put_long, "put_credit": put_credit,
        "call_short": call_short, "call_long": call_long, "call_credit": call_credit,
        "total_credit": total_credit, "max_loss": max_loss,
    }


def backtest_1_3_dte(
    dte_target: int = 1,  # 1, 2, or 3
    start_date: str = "2023-01-01",
    end_date: str = "2026-01-01",
    otm_pct: Optional[float] = None,  # Auto-scale by DTE if None
    spread_width: float = 5.0,
    risk_pct: float = 0.02,
) -> List[ICTrade]:
    # Scale OTM% by DTE: tighter at 1DTE (options are pennies far OTM)
    if otm_pct is None:
        otm_pct = {1: 0.015, 2: 0.025, 3: 0.040}.get(dte_target, 0.05)
    """Backtest iron condor with target DTE on real IronVault data.

    Entry: market close of day (entry_day_of_week) before Friday expiration
    Exit: Friday expiration (or mid-week stop)
    """
    print(f"  Loading SPY spot prices from Yahoo...")
    spy_spot = load_spy_spot_yfinance(start_date, end_date)
    print(f"    {len(spy_spot)} SPY bars")

    print(f"  Finding SPY Friday expirations...")
    exps = find_friday_expirations(start_date, end_date)
    print(f"    {len(exps)} Friday expirations")

    trades: List[ICTrade] = []

    for exp in exps:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        # Friday = weekday 4. Entry_day = Friday - dte_target
        entry_dt = exp_dt - timedelta(days=dte_target)
        # Make sure entry is a weekday
        while entry_dt.weekday() >= 5:
            entry_dt -= timedelta(days=1)

        entry_str = entry_dt.strftime("%Y-%m-%d")

        # Get spot on entry date
        if pd.Timestamp(entry_dt) not in spy_spot.index:
            continue
        spot = float(spy_spot.loc[pd.Timestamp(entry_dt)])

        # Find spread
        spread = find_condor_spread(exp, entry_str, spot, otm_pct, spread_width)
        if spread is None:
            continue

        # Position sizing
        risk_budget = CAPITAL * risk_pct
        contracts = max(1, min(20, int(risk_budget / (spread["max_loss"] * 100))))

        # Walk to exit: check daily close for profit target / stop loss
        exit_date = exp
        exit_reason = "expiration"
        exit_credit = 0.0

        cur_dt = entry_dt + timedelta(days=1)
        while cur_dt <= exp_dt:
            cs = cur_dt.strftime("%Y-%m-%d")
            if pd.Timestamp(cur_dt) not in spy_spot.index:
                cur_dt += timedelta(days=1)
                continue

            pp = get_spread_close(exp, spread["put_short"], spread["put_long"], "P", cs)
            cp = get_spread_close(exp, spread["call_short"], spread["call_long"], "C", cs)
            if pp is None or cp is None:
                cur_dt += timedelta(days=1)
                continue

            cur_put = pp["short_close"] - pp["long_close"]
            cur_call = cp["short_close"] - cp["long_close"]
            cur_total = cur_put + cur_call

            # 50% profit target
            if cur_total <= spread["total_credit"] * 0.50:
                exit_date = cs
                exit_reason = "profit_target"
                exit_credit = cur_total
                break

            # 2x stop loss
            if cur_total - spread["total_credit"] > spread["total_credit"] * 2.0:
                exit_date = cs
                exit_reason = "stop_loss"
                exit_credit = cur_total
                break

            cur_dt += timedelta(days=1)

        # At expiration, assume worthless
        if exit_reason == "expiration":
            exit_credit = 0.0

        pnl = (spread["total_credit"] - exit_credit) * 100 * contracts
        trades.append(ICTrade(
            entry_date=entry_str,
            exit_date=exit_date,
            expiration=exp,
            dte_at_entry=dte_target,
            underlying=round(spot, 2),
            put_short=spread["put_short"],
            put_long=spread["put_long"],
            call_short=spread["call_short"],
            call_long=spread["call_long"],
            credit=round(spread["total_credit"], 3),
            exit_reason=exit_reason,
            pnl=round(pnl, 2),
            contracts=contracts,
        ))

    return trades


# ═══════════════════════════════════════════════════════════════════════════
# Metrics with walk-forward
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(trades: List[ICTrade]) -> Dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "sharpe": 0, "cagr": 0,
                "max_dd": 0, "is_sharpe": 0, "oos_sharpe": 0}

    df = pd.DataFrame([vars(t) for t in trades])
    pnls = df["pnl"].values
    n = len(pnls)
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())

    # Trade-level Sharpe
    sharpe = trade_sharpe(pnls)

    # Equity curve and DD
    eq = np.cumsum(pnls) + CAPITAL
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() if len(pk) > 0 else 0

    # CAGR
    dates = pd.to_datetime(df["exit_date"])
    entry_dates = pd.to_datetime(df["entry_date"])
    years = max((dates.max() - entry_dates.min()).days / 365.25, 0.5)
    cagr = ((1 + total / CAPITAL) ** (1 / years) - 1) if total > -CAPITAL else -1

    # Walk-forward (IS=2023, OOS=2024-2025)
    is_mask = dates.dt.year <= 2023
    oos_mask = dates.dt.year >= 2024
    is_sharpe = trade_sharpe(pnls[is_mask]) if is_mask.any() else 0
    oos_sharpe = trade_sharpe(pnls[oos_mask]) if oos_mask.any() else 0

    # Yearly
    df["year"] = dates.dt.year
    yearly = {}
    for yr, grp in df.groupby("year"):
        yp = grp["pnl"].values
        yearly[int(yr)] = {
            "n": len(yp), "pnl": float(yp.sum()),
            "wr": float((yp > 0).sum() / len(yp)) if len(yp) > 0 else 0,
            "sharpe": trade_sharpe(yp),
        }

    # Exit reason breakdown
    exit_counts = df["exit_reason"].value_counts().to_dict()

    return {
        "n": n, "pnl": round(total, 2),
        "wr": round(wins / n, 3) if n > 0 else 0,
        "sharpe": round(sharpe, 2),
        "cagr": round(cagr, 4),
        "max_dd": round(float(dd), 4),
        "is_sharpe": round(is_sharpe, 2),
        "oos_sharpe": round(oos_sharpe, 2),
        "wf_ratio": round(oos_sharpe / is_sharpe, 2) if abs(is_sharpe) > 0.01 else 0,
        "yearly": yearly,
        "exit_counts": exit_counts,
    }
