"""
EXP-1820: Dispersion-Inspired Relative Vol Premium
====================================================
Theoretical edge: Sector ETFs (XLF, XLK, XLI, XLE) have higher idiosyncratic
risk than the index (SPY), so their options carry richer premiums. When the
sector-to-index vol ratio widens beyond normal, selling the richer leg
captures "dispersion premium" — similar to classic dispersion but adapted
to the instruments we have real data for.

True dispersion trading requires individual stock options. With only sector
ETFs available, we approximate by comparing vol richness ratios between
SPY and its sector constituents. When sector vol > SPY vol by >15%, sell
the sector put spreads (richer vol); otherwise skip.

Data availability (IronVault):
  SPY: 193K contracts, full coverage 2020-2026
  XLF:  9K contracts, 2020-2026
  XLI: 17K contracts, 2020-2026
  XLK:  3K contracts, 2020-2026
  XLE:  2K contracts, 2020-2026 (partial)
  QQQ: 23K contracts, 2020-2025

All data from IronVault options_cache.db (Polygon real prices) + Yahoo
Finance for spot prices. Rule Zero compliant: zero synthetic.
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
COMMISSION = 0.65  # $/contract/leg

# Sector tickers we have data for (components that roughly compose SPY)
SECTORS = ["XLF", "XLI", "XLK", "XLE"]
INDEX = "SPY"


# ═══════════════════════════════════════════════════════════════════════════
# Corrected Sharpe
# ═══════════════════════════════════════════════════════════════════════════

def corrected_sharpe(rets: np.ndarray, rf: float = 0.045) -> float:
    if len(rets) < 2:
        return 0.0
    r = np.asarray(rets, dtype=np.float64)
    rf_d = rf / TRADING_DAYS
    excess = float(np.mean(r)) - rf_d
    std = float(np.std(r, ddof=0))
    if std < 1e-12:
        return 0.0
    return excess / std * math.sqrt(TRADING_DAYS)


def trade_sharpe(pnls) -> float:
    if len(pnls) < 2:
        return 0.0
    p = np.asarray(pnls, dtype=np.float64)
    s = float(np.std(p, ddof=1))
    if s < 1e-8:
        return 0.0
    return float(np.mean(p) / s * math.sqrt(min(len(p), 52)))


# ═══════════════════════════════════════════════════════════════════════════
# IronVault queries
# ═══════════════════════════════════════════════════════════════════════════

def load_spot(ticker: str, start: str = "2020-01-01", end: str = "2026-01-01") -> pd.Series:
    """Load real spot prices from Yahoo Finance."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df["Close"]


def find_friday_exps(ticker: str, start: str, end: str) -> List[str]:
    """Find all Friday expirations for a ticker."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker=? AND expiration BETWEEN ? AND ?
          AND CAST(STRFTIME('%w', expiration) AS INTEGER) = 5
        ORDER BY expiration
    """, (ticker, start, end))
    out = [r[0] for r in cur.fetchall()]
    conn.close()
    return out


def get_available_strikes(ticker: str, exp: str, date_str: str, opt_type: str) -> List[float]:
    """Get put strikes with actual bars on a given date."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT oc.strike
        FROM option_contracts oc
        JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
        WHERE oc.ticker=? AND oc.expiration=? AND oc.option_type=? AND od.date=?
        ORDER BY oc.strike
    """, (ticker, exp, opt_type, date_str))
    strikes = [float(r[0]) for r in cur.fetchall()]
    conn.close()
    return strikes


def get_spread_close(ticker: str, exp: str, short_k: float, long_k: float,
                     opt_type: str, date_str: str) -> Optional[Dict]:
    """Get close prices for a credit spread."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
        SELECT oc.strike, od.close
        FROM option_contracts oc
        JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
        WHERE oc.ticker=? AND oc.expiration=? AND oc.option_type=?
          AND oc.strike IN (?, ?) AND od.date=?
    """, (ticker, exp, opt_type, short_k, long_k, date_str))
    rows = {float(r[0]): float(r[1]) for r in cur.fetchall()}
    conn.close()
    if short_k not in rows or long_k not in rows:
        return None
    return {"short_close": rows[short_k], "long_close": rows[long_k]}


# ═══════════════════════════════════════════════════════════════════════════
# Build a 5% OTM put spread and return its credit/width ratio (vol proxy)
# ═══════════════════════════════════════════════════════════════════════════

def find_put_spread(
    ticker: str, exp: str, date_str: str, spot: float,
    otm_pct: float = 0.05, width: float = 5.0,
) -> Optional[Dict]:
    """Find an available 5% OTM put spread and return credit + structure."""
    strikes = get_available_strikes(ticker, exp, date_str, "P")
    if not strikes:
        return None

    put_target = spot * (1 - otm_pct)

    # Find short strike: highest available below target
    candidates = [s for s in strikes if s <= put_target]
    if not candidates:
        return None
    put_short = max(candidates)

    put_long = put_short - width
    if put_long not in strikes:
        # Find nearest available
        nearest = [s for s in strikes if s < put_short and s >= put_short - width - 2]
        if not nearest:
            return None
        put_long = max(nearest)

    pp = get_spread_close(ticker, exp, put_short, put_long, "P", date_str)
    if pp is None:
        return None

    credit = pp["short_close"] - pp["long_close"]
    actual_width = put_short - put_long
    if credit <= 0.02 or actual_width <= 0:
        return None

    return {
        "ticker": ticker,
        "put_short": put_short,
        "put_long": put_long,
        "width": actual_width,
        "credit": round(credit, 3),
        "credit_pct_of_spot": round(credit / spot * 100, 3),  # vol proxy
        "credit_pct_of_width": round(credit / actual_width, 3),
        "max_loss": round(actual_width - credit, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Dispersion signal: compute vol premium ratio
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DispersionTrade:
    entry_date: str
    exit_date: str
    exp: str
    ticker: str          # which one we sold
    spot: float
    put_short: float
    put_long: float
    credit: float
    contracts: int
    exit_reason: str
    pnl: float
    vol_ratio: float     # sector_vol / spy_vol at entry
    dispersion_signal: str


def backtest_dispersion(
    start: str = "2020-06-01",
    end: str = "2026-01-01",
    vol_ratio_threshold: float = 1.15,  # sector must be >15% richer than SPY
    risk_pct: float = 0.02,
) -> List[DispersionTrade]:
    """Run dispersion backtest.

    On each weekly entry (35 DTE from Friday expiration):
      1. Compute SPY 5% OTM put spread credit/spot (vol proxy for index)
      2. Compute each sector's 5% OTM put spread credit/spot (vol proxy for component)
      3. If any sector vol > SPY vol × threshold → sell THAT sector's put spread
      4. Exit at 50% profit, 2x stop, or 7 DTE
    """
    print("  Loading spot prices...")
    spots = {}
    for t in [INDEX] + SECTORS:
        try:
            spots[t] = load_spot(t, start=start, end=end)
        except Exception as e:
            print(f"    {t}: {e}")
    print(f"    Loaded {len(spots)} spot series")

    print("  Finding SPY Friday expirations...")
    exps = find_friday_exps(INDEX, start, end)
    print(f"    {len(exps)} expirations")

    trades = []
    last_entry_by_ticker = {t: None for t in SECTORS}

    for exp in exps:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        # Entry ~35 days before expiration
        entry_dt = exp_dt - timedelta(days=35)
        # Find valid trading day
        for off in range(7):
            cand = entry_dt + timedelta(days=off)
            if pd.Timestamp(cand) in spots[INDEX].index:
                entry_dt = cand
                break
        else:
            continue

        entry_str = entry_dt.strftime("%Y-%m-%d")

        # Get SPY vol proxy
        if pd.Timestamp(entry_dt) not in spots[INDEX].index:
            continue
        spy_spot = float(spots[INDEX].loc[pd.Timestamp(entry_dt)])
        spy_spread = find_put_spread(INDEX, exp, entry_str, spy_spot)
        if spy_spread is None:
            continue
        spy_vol_proxy = spy_spread["credit_pct_of_spot"]

        # Evaluate each sector
        for sector in SECTORS:
            if sector not in spots:
                continue

            # Enforce spacing per ticker (don't over-trade same sector)
            if last_entry_by_ticker[sector] is not None:
                if (entry_dt - last_entry_by_ticker[sector]).days < 20:
                    continue

            if pd.Timestamp(entry_dt) not in spots[sector].index:
                continue
            sec_spot = float(spots[sector].loc[pd.Timestamp(entry_dt)])

            # Check if sector has this expiration
            sec_strikes = get_available_strikes(sector, exp, entry_str, "P")
            if not sec_strikes:
                continue

            # Adjust width for lower-priced sectors (use $1 spreads for <$50 tickers)
            sec_width = 1.0 if sec_spot < 80 else 2.0
            sec_spread = find_put_spread(sector, exp, entry_str, sec_spot,
                                         otm_pct=0.05, width=sec_width)
            if sec_spread is None:
                continue

            sec_vol_proxy = sec_spread["credit_pct_of_spot"]

            # Dispersion signal: is sector significantly richer?
            if spy_vol_proxy <= 0:
                continue
            ratio = sec_vol_proxy / spy_vol_proxy

            if ratio < vol_ratio_threshold:
                continue  # Not enough dispersion premium

            # Trade: sell sector put spread (capture the richer vol)
            # Position size
            max_loss = sec_spread["max_loss"]
            risk_budget = CAPITAL * risk_pct
            contracts = max(1, min(15, int(risk_budget / (max_loss * 100))))

            # Walk to exit
            exit_date = exp
            exit_reason = "expiration"
            exit_credit = 0.0

            cur_dt = entry_dt + timedelta(days=1)
            sec_spot_series = spots[sector]
            while cur_dt <= exp_dt:
                cs = cur_dt.strftime("%Y-%m-%d")
                if pd.Timestamp(cur_dt) not in sec_spot_series.index:
                    cur_dt += timedelta(days=1)
                    continue

                pp = get_spread_close(sector, exp, sec_spread["put_short"],
                                      sec_spread["put_long"], "P", cs)
                if pp is None:
                    cur_dt += timedelta(days=1)
                    continue

                cur_val = pp["short_close"] - pp["long_close"]

                # 50% profit target
                if cur_val <= sec_spread["credit"] * 0.50:
                    exit_date = cs
                    exit_reason = "profit_target"
                    exit_credit = cur_val
                    break

                # 2x stop loss
                if cur_val - sec_spread["credit"] > sec_spread["credit"] * 2.0:
                    exit_date = cs
                    exit_reason = "stop_loss"
                    exit_credit = cur_val
                    break

                # DTE exit
                if (exp_dt - cur_dt).days <= 7:
                    exit_date = cs
                    exit_reason = "dte_exit"
                    exit_credit = cur_val
                    break

                cur_dt += timedelta(days=1)

            # At expiration, assume worthless (5% OTM usually)
            if exit_reason == "expiration":
                exit_credit = 0.0

            # Commissions: 2 legs × 2 sides × $0.65 = $2.60/contract round trip
            commission = 2 * 2 * COMMISSION * contracts
            pnl = (sec_spread["credit"] - exit_credit) * 100 * contracts - commission

            trades.append(DispersionTrade(
                entry_date=entry_str,
                exit_date=exit_date,
                exp=exp,
                ticker=sector,
                spot=round(sec_spot, 2),
                put_short=sec_spread["put_short"],
                put_long=sec_spread["put_long"],
                credit=round(sec_spread["credit"], 3),
                contracts=contracts,
                exit_reason=exit_reason,
                pnl=round(pnl, 2),
                vol_ratio=round(ratio, 3),
                dispersion_signal=f"{sector} vol/SPY vol = {ratio:.2f}",
            ))
            last_entry_by_ticker[sector] = entry_dt

    return trades


# ═══════════════════════════════════════════════════════════════════════════
# Metrics + walk-forward
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(trades: List[DispersionTrade]) -> Dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "sharpe": 0, "cagr": 0,
                "max_dd": 0, "is_sharpe": 0, "oos_sharpe": 0}

    df = pd.DataFrame([vars(t) for t in trades])
    pnls = df["pnl"].values
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())

    sharpe = trade_sharpe(pnls)
    eq = np.cumsum(pnls) + CAPITAL
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max()

    dates = pd.to_datetime(df["exit_date"])
    entry_dates = pd.to_datetime(df["entry_date"])
    years = max((dates.max() - entry_dates.min()).days / 365.25, 0.5)
    cagr = ((1 + total / CAPITAL) ** (1 / years) - 1) if total > -CAPITAL else -1

    # Walk-forward: IS 2020-2022, OOS 2023-2025
    is_mask = dates.dt.year <= 2022
    oos_mask = dates.dt.year >= 2023
    is_sharpe = trade_sharpe(pnls[is_mask]) if is_mask.any() else 0
    oos_sharpe = trade_sharpe(pnls[oos_mask]) if oos_mask.any() else 0

    # Yearly
    df["year"] = dates.dt.year
    yearly = {}
    for yr, grp in df.groupby("year"):
        yp = grp["pnl"].values
        yearly[int(yr)] = {
            "n": len(yp),
            "pnl": float(yp.sum()),
            "wr": float((yp > 0).sum() / len(yp)) if len(yp) > 0 else 0,
            "sharpe": trade_sharpe(yp),
        }

    # Per-ticker breakdown
    ticker_stats = {}
    for tk, grp in df.groupby("ticker"):
        tp = grp["pnl"].values
        ticker_stats[str(tk)] = {
            "n": len(tp),
            "pnl": round(float(tp.sum()), 2),
            "wr": round(float((tp > 0).sum() / len(tp)), 3),
            "sharpe": round(trade_sharpe(tp), 2),
            "avg_vol_ratio": round(float(grp["vol_ratio"].mean()), 3),
        }

    return {
        "n": len(pnls), "pnl": round(total, 2),
        "wr": round(wins / len(pnls), 3),
        "sharpe": round(sharpe, 2),
        "cagr": round(float(cagr), 4),
        "max_dd": round(float(dd), 4),
        "is_sharpe": round(is_sharpe, 2),
        "oos_sharpe": round(oos_sharpe, 2),
        "yearly": yearly,
        "ticker_stats": ticker_stats,
    }
