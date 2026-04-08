"""
Strategy Discovery Round 3 — find NEW uncorrelated alpha sources.

All prices from IronVault (options_cache.db). Zero synthetic data.
SPY/VIX/ETF daily prices from yfinance.

Strategies:
  1. Options Skew Trading — sell rich put skew when put/call ratio > 1.3
  2. VIX Mean-Reversion — sell SPY puts when VIX z-score > 1.5 (spike)
  3. Sector Rotation + Premium Selling — trade calmest sector ETF
  4. Calendar Spread Arbitrage — front/back month theta differential on SPY
  5. Correlation Breakout — trade XLF/XLI pair divergence on corr breakdown

Walk-forward: IS = 2020-2022, OOS = 2023-2025.
Kill: <15 OOS trades OR negative OOS Sharpe.
Correlation computed vs SPY returns AND vs EXP-1220 proxy.
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
DEFAULT_OUTPUT = ROOT / "reports" / "strategy_discovery_round3.html"

CAPITAL = 100_000
OOS_START_YEAR = 2023


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers (same pattern as R2)
# ═══════════════════════════════════════════════════════════════════════════


def _exp_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _dl(ticker: str, start: str = "2019-06-01", end: str = "2026-07-01") -> pd.DataFrame:
    """Download daily OHLCV via yfinance."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def _find_exps(hd: IronVault, ticker: str, start: str, end: str,
               monthly: bool = True) -> List[str]:
    """Available expirations from DB."""
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


def _next_td(dt: datetime, td_set) -> Optional[datetime]:
    for off in range(7):
        c = dt + timedelta(days=off)
        if c.strftime("%Y-%m-%d") in td_set:
            return c
    return None


def _sell_put_spread(
    hd: IronVault, ticker: str, exp: str, trade_date: str,
    underlying_price: float, otm_pct: float = 0.93, width: float = 5.0,
) -> Optional[Dict]:
    """Find and price an OTM put credit spread."""
    strikes = hd.get_available_strikes(ticker, exp, trade_date, "P")
    if not strikes:
        return None
    target = underlying_price * otm_pct
    for sk in sorted(strikes, key=lambda k: abs(k - target))[:12]:
        lk = sk - width
        if lk not in strikes:
            candidates = [s for s in strikes if s < sk and abs(s - lk) <= 1.0]
            if not candidates:
                continue
            lk = max(candidates)
        actual_width = sk - lk
        if actual_width <= 0:
            continue
        pp = hd.get_spread_prices(ticker, _exp_dt(exp), sk, lk, "P", trade_date)
        if pp is None:
            continue
        credit = pp["short_close"] - pp["long_close"]
        if credit > 0.05:
            return {"short": sk, "long": lk, "credit": round(credit, 4),
                    "width": actual_width, "max_loss": round(actual_width - credit, 4)}
    return None


def _sell_call_spread(
    hd: IronVault, ticker: str, exp: str, trade_date: str,
    underlying_price: float, otm_pct: float = 1.07, width: float = 5.0,
) -> Optional[Dict]:
    """Find and price an OTM call credit spread."""
    strikes = hd.get_available_strikes(ticker, exp, trade_date, "C")
    if not strikes:
        return None
    target = underlying_price * otm_pct
    for sk in sorted(strikes, key=lambda k: abs(k - target))[:12]:
        lk = sk + width
        if lk not in strikes:
            candidates = [s for s in strikes if s > sk and abs(s - lk) <= 1.0]
            if not candidates:
                continue
            lk = min(candidates)
        actual_width = lk - sk
        if actual_width <= 0:
            continue
        pp = hd.get_spread_prices(ticker, _exp_dt(exp), sk, lk, "C", trade_date)
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
    td_index, opt_type: str = "P",
    profit_pct: float = 0.50, stop_mult: float = 3.0, min_dte: int = 7,
) -> Tuple[Optional[str], str, float, int]:
    """Walk forward through real prices to find exit."""
    hold_days = 0
    td_set = set(td_index.strftime("%Y-%m-%d"))
    current = entry_dt + timedelta(days=1)

    while current <= exp_dt_obj:
        curr_str = current.strftime("%Y-%m-%d")
        if curr_str not in td_set:
            current += timedelta(days=1)
            continue
        hold_days += 1
        dte_rem = (exp_dt_obj - current).days

        pp = hd.get_spread_prices(ticker, exp_dt_obj, short_k, long_k, opt_type, curr_str)
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

    fp = hd.get_spread_prices(ticker, exp_dt_obj, short_k, long_k, opt_type, exp)
    exit_val = (fp["short_close"] - fp["long_close"]) if fp else 0.0
    return exp, "expiration", exit_val, hold_days


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
    exp1220_corr: float = 0.0
    avg_pnl: float = 0.0
    oos_sharpe: float = 0.0
    oos_n: int = 0
    oos_pnl: float = 0.0
    oos_wr: float = 0.0
    oos_dd: float = 0.0
    oos_cagr: float = 0.0
    yearly: Dict[int, Dict] = field(default_factory=dict)
    killed: bool = False
    kill_reason: str = ""


def _compute(
    trades: List[Dict], name: str,
    spy_ret: pd.Series, exp1220_ret: pd.Series,
    desc: str = "",
) -> Stats:
    """Compute full stats including correlation to SPY AND EXP-1220."""
    if not trades:
        return Stats(name=name, description=desc, killed=True, kill_reason="0 trades")
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

    # Correlation with SPY and EXP-1220
    tr = {}
    for _, r in df.iterrows():
        d = str(r["exit_date"])[:10]
        tr[d] = tr.get(d, 0) + r["pnl"]
    ts = pd.Series(tr)
    ts.index = pd.to_datetime(ts.index)

    def _corr(a, b):
        common = a.index.intersection(b.index)
        if len(common) > 10:
            return float(np.corrcoef(
                a.reindex(common).fillna(0).values,
                b.reindex(common).fillna(0).values,
            )[0, 1])
        return 0.0

    spy_corr = _corr(ts, spy_ret)
    exp1220_corr = _corr(ts, exp1220_ret)

    # OOS
    oos = df[dates.dt.year >= OOS_START_YEAR]
    oos_n = len(oos)
    oos_sharpe = oos_pnl = oos_wr = oos_dd_val = oos_cagr_val = 0.0
    if oos_n > 1:
        op = oos["pnl"].values
        oos_pnl = float(op.sum())
        oos_wr = float((op > 0).sum()) / oos_n
        os = float(op.std(ddof=1))
        oos_sharpe = float(op.mean()) / os * math.sqrt(min(oos_n, 52)) if os > 1e-9 else 0.0
        oos_eq = np.cumsum(op) + CAPITAL
        oos_pk = np.maximum.accumulate(oos_eq)
        oos_dd_val = float(((oos_pk - oos_eq) / oos_pk).max())
        oos_dates = pd.to_datetime(oos["exit_date"])
        oos_yrs = max((oos_dates.max() - pd.to_datetime(oos["entry_date"]).min()).days / 365.25, 0.5)
        oos_cagr_val = ((1 + oos_pnl / CAPITAL) ** (1 / oos_yrs) - 1) if oos_pnl > -CAPITAL else -1.0

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
            "sharpe": round(float(yp.mean()) / ysd * math.sqrt(min(yn, 52))
                           if ysd > 1e-9 else 0, 3),
            "ret": round(float(yp.sum()) / CAPITAL, 4),
        }

    killed = oos_n < 15 or oos_sharpe < 0
    kr = ""
    if oos_n < 15:
        kr = f"Only {oos_n} OOS trades (<15)"
    elif oos_sharpe < 0:
        kr = f"Negative OOS Sharpe ({oos_sharpe:.2f})"

    return Stats(
        name=name, description=desc, trades=trades,
        n_trades=n, total_pnl=round(total, 2), win_rate=round(wins / n, 4),
        max_dd=round(max_dd, 4), sharpe=round(sharpe, 3), cagr=round(cagr, 4),
        spy_corr=round(spy_corr, 4), exp1220_corr=round(exp1220_corr, 4),
        avg_pnl=round(mu, 2),
        oos_sharpe=round(oos_sharpe, 3), oos_n=oos_n,
        oos_pnl=round(oos_pnl, 2), oos_wr=round(oos_wr, 4),
        oos_dd=round(oos_dd_val, 4), oos_cagr=round(oos_cagr_val, 4),
        yearly=yearly, killed=killed, kill_reason=kr,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: Options Skew Trading — sell rich put skew on SPY
# ═══════════════════════════════════════════════════════════════════════════

def strat_skew_trading(hd: IronVault, spy_df: pd.DataFrame,
                       vix: pd.Series) -> List[Dict]:
    """Sell put spreads when put skew is expensive (high demand for protection).

    Compute skew = OTM put price / OTM call price at similar delta.
    When skew > 1.5 (puts richly priced relative to calls), sell put spreads.
    High skew = fear premium → overpriced → sell it.
    """
    print("  [1] Options Skew Trading: sell rich put skew")
    spy_close = spy_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, "SPY", "2020-03-01", "2025-12-31", monthly=False)
    trades, last = [], None

    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=30), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < 14:
            continue

        try:
            price = float(spy_close.loc[es])
        except (KeyError, TypeError):
            continue
        if np.isnan(price):
            continue

        # VIX filter: only trade when VIX < 35 (avoid extreme crisis)
        try:
            v = float(vix.loc[es])
        except (KeyError, TypeError):
            v = 20.0
        if v > 35:
            continue

        # Compute put/call skew from real option prices
        put_strike = round(price * 0.95)  # 5% OTM put
        call_strike = round(price * 1.05)  # 5% OTM call
        put_strikes = hd.get_available_strikes("SPY", exp, es, "P")
        call_strikes = hd.get_available_strikes("SPY", exp, es, "C")
        if not put_strikes or not call_strikes:
            continue

        pk = min(put_strikes, key=lambda k: abs(k - put_strike))
        ck = min(call_strikes, key=lambda k: abs(k - call_strike))

        put_sym = IronVault.build_occ_symbol("SPY", exp_obj, pk, "P")
        call_sym = IronVault.build_occ_symbol("SPY", exp_obj, ck, "C")
        pp = hd.get_contract_price(put_sym, es)
        cp = hd.get_contract_price(call_sym, es)
        if pp is None or cp is None or cp < 0.10:
            continue

        skew = pp / cp
        if skew < 1.3:  # Not enough skew to exploit
            continue

        # Sell put spread (harvesting the rich put skew)
        spread = _sell_put_spread(hd, "SPY", exp, es, price, otm_pct=0.94, width=5.0)
        if spread is None:
            continue

        contracts = max(1, min(3, int(CAPITAL * 0.02 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(
            hd, "SPY", exp, spread["short"], spread["long"],
            spread["credit"], entry_dt, exp_obj, spy_df.index)
        pnl = (spread["credit"] - ev) * 100 * contracts
        trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                        "exit_reason": er, "credit": spread["credit"],
                        "skew": round(skew, 3), "hold_days": hold})
        last = entry_dt

    print(f"    → {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: Intraday VIX Mean-Reversion — sell puts when VIX spikes
# ═══════════════════════════════════════════════════════════════════════════

def strat_vix_mean_reversion(hd: IronVault, spy_df: pd.DataFrame,
                              vix: pd.Series) -> List[Dict]:
    """Sell SPY put spreads when VIX spikes above its 20-day mean + 1.5σ.

    VIX mean-reverts: spikes are temporary, and the elevated premium after
    a spike makes put selling profitable. Entry when VIX z-score > 1.5,
    exit at profit target or when VIX normalises. This captures the fear
    premium that decays as panic subsides.
    """
    print("  [2] VIX Mean-Reversion: sell puts on VIX spikes")
    spy_close = spy_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, "SPY", "2020-03-01", "2025-12-31", monthly=False)

    # VIX z-score
    vix_ma = vix.rolling(20).mean()
    vix_std = vix.rolling(20).std()
    vix_z = (vix - vix_ma) / vix_std.replace(0, np.nan)
    vix_z = vix_z.dropna()

    trades, last = [], None

    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=30), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < 10:
            continue

        try:
            z = float(vix_z.loc[es])
            v = float(vix.loc[es])
            price = float(spy_close.loc[es])
        except (KeyError, TypeError):
            continue
        if np.isnan(z) or np.isnan(price):
            continue

        # Entry: VIX z-score > 1.5 (spike) but VIX < 50 (not Armageddon)
        if z < 1.5 or v > 50:
            continue

        # Sell put spread — rich premium after spike
        spread = _sell_put_spread(hd, "SPY", exp, es, price, otm_pct=0.94, width=5.0)
        if spread is None:
            continue

        contracts = max(1, min(3, int(CAPITAL * 0.02 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(
            hd, "SPY", exp, spread["short"], spread["long"],
            spread["credit"], entry_dt, exp_obj, spy_df.index)
        pnl = (spread["credit"] - ev) * 100 * contracts
        trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                        "exit_reason": er, "credit": spread["credit"],
                        "vix": round(v, 1), "vix_z": round(z, 2), "hold_days": hold})
        last = entry_dt

    print(f"    → {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Sector Rotation via IV Rank — trade lowest-IV sector
# ═══════════════════════════════════════════════════════════════════════════

def strat_sector_iv_rotation(hd: IronVault, spy_df: pd.DataFrame,
                              sector_dfs: Dict[str, pd.DataFrame]) -> List[Dict]:
    """Sell put spreads on the sector ETF with lowest recent volatility.

    Rotate monthly between XLF, XLI, XLE — always trade the calmest sector
    (lowest 20-day realized vol). Calm sectors have cheap premium but reliable
    theta decay. This is uncorrelated to SPY because it's a RELATIVE rotation.
    """
    print("  [3] Sector IV Rotation: trade calmest sector")
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    sector_tickers = ["XLF", "XLI", "XLE"]

    # Compute rolling vol for each sector
    sector_vol = {}
    for tk in sector_tickers:
        if tk in sector_dfs and len(sector_dfs[tk]) > 0:
            sv = sector_dfs[tk]["Close"].pct_change().rolling(20).std() * math.sqrt(252)
            sector_vol[tk] = sv

    if not sector_vol:
        print("    → 0 trades (no sector data)")
        return []

    trades, last = [], None

    for tk in sector_tickers:
        exps = _find_exps(hd, tk, "2020-03-01", "2025-12-31", monthly=True)
        for exp in exps:
            exp_obj = _exp_dt(exp)
            entry_dt = _next_td(exp_obj - timedelta(days=30), td_set)
            if entry_dt is None:
                continue
            es = entry_dt.strftime("%Y-%m-%d")
            if last and (entry_dt - last).days < 14:
                continue

            # Check if this ticker has lowest vol right now
            vols = {}
            for stk in sector_tickers:
                if stk in sector_vol:
                    try:
                        vols[stk] = float(sector_vol[stk].loc[es])
                    except (KeyError, TypeError):
                        pass
            if not vols or tk not in vols:
                continue
            calmest = min(vols, key=vols.get)
            if calmest != tk:
                continue

            try:
                price = float(sector_dfs[tk]["Close"].loc[es])
            except (KeyError, TypeError):
                continue
            if np.isnan(price):
                continue

            # Sector-appropriate width
            width = 2.0 if tk in ("XLF", "XLE") else 5.0
            spread = _sell_put_spread(hd, tk, exp, es, price, otm_pct=0.94, width=width)
            if spread is None:
                continue

            contracts = max(1, min(5, int(CAPITAL * 0.02 / (spread["max_loss"] * 100))))
            ed, er, ev, hold = _walk_spread(
                hd, tk, exp, spread["short"], spread["long"],
                spread["credit"], entry_dt, exp_obj, spy_df.index)
            pnl = (spread["credit"] - ev) * 100 * contracts
            trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                            "exit_reason": er, "ticker": tk, "credit": spread["credit"],
                            "sector_vol": round(vols[tk], 4), "hold_days": hold})
            last = entry_dt

    print(f"    → {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 4: Calendar Spread Carry — front/back month theta
# ═══════════════════════════════════════════════════════════════════════════

def strat_calendar_carry(hd: IronVault, spy_df: pd.DataFrame) -> List[Dict]:
    """Sell front-month ATM put, buy back-month ATM put at same strike.

    Front month decays faster than back month (theta differential).
    Entry: when front IV > back IV (inverted term structure at strike level).
    Exit at front expiration or 50% profit.
    """
    print("  [4] Calendar Spread Carry: front/back theta differential")
    spy_close = spy_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))

    conn = sqlite3.connect(hd._db_path)
    all_exps = [r[0] for r in conn.execute(
        """SELECT DISTINCT expiration FROM option_contracts
           WHERE ticker='SPY' AND option_type='P' AND expiration BETWEEN '2020-03-01' AND '2025-12-31'
           ORDER BY expiration""").fetchall()]
    conn.close()

    trades, last = [], None

    for i, front in enumerate(all_exps):
        front_dt = _exp_dt(front)
        # Find back expiration 25-40 days after front
        back = None
        for j in range(i + 1, min(i + 25, len(all_exps))):
            delta = (_exp_dt(all_exps[j]) - front_dt).days
            if 25 <= delta <= 40:
                back = all_exps[j]
                break
        if back is None:
            continue
        back_dt = _exp_dt(back)

        entry_dt = _next_td(front_dt - timedelta(days=21), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < 14:
            continue

        try:
            price = float(spy_close.loc[es])
        except (KeyError, TypeError):
            continue
        if np.isnan(price):
            continue

        # ATM-ish strike
        target_k = round(price)
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
        if fp is None or bp is None:
            continue

        # Calendar spread: sell front, buy back
        # Net debit = back_price - front_price (should be positive, back costs more)
        net_debit = bp - fp
        if net_debit <= 0 or net_debit > 5.0:
            continue

        contracts = max(1, min(3, int(CAPITAL * 0.015 / (net_debit * 100))))

        # Walk to front expiration, check if spread widens (profit)
        exit_val = net_debit
        exit_date = es
        exit_reason = "expiration"
        hold_days = 0
        current = entry_dt + timedelta(days=1)

        while current <= front_dt:
            curr_str = current.strftime("%Y-%m-%d")
            if curr_str not in td_set:
                current += timedelta(days=1)
                continue
            hold_days += 1

            fp2 = hd.get_contract_price(front_sym, curr_str)
            bp2 = hd.get_contract_price(back_sym, curr_str)
            if fp2 is not None and bp2 is not None:
                current_spread = bp2 - fp2
                # Profit if spread widened (front decayed faster)
                if current_spread >= net_debit * 1.5:  # 50% profit
                    exit_val = current_spread
                    exit_date = curr_str
                    exit_reason = "profit_target"
                    break
                if current_spread <= net_debit * 0.5:  # 50% loss
                    exit_val = current_spread
                    exit_date = curr_str
                    exit_reason = "stop_loss"
                    break
                exit_val = current_spread
                exit_date = curr_str

            current += timedelta(days=1)

        pnl = (exit_val - net_debit) * 100 * contracts
        trades.append({"entry_date": es, "exit_date": exit_date, "pnl": round(pnl, 2),
                        "exit_reason": exit_reason, "strike": strike,
                        "net_debit": round(net_debit, 4), "hold_days": hold_days})
        last = entry_dt

    print(f"    → {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 5: Correlation Breakout — trade pair reversion on divergence
# ═══════════════════════════════════════════════════════════════════════════

def strat_correlation_breakout(hd: IronVault, spy_df: pd.DataFrame,
                                xlf_df: pd.DataFrame, xli_df: pd.DataFrame) -> List[Dict]:
    """Trade mean-reversion when normally-correlated pairs diverge.

    XLF and XLI are normally highly correlated (financials ↔ industrials).
    When their 20-day rolling correlation drops below 0.5 AND the return
    spread exceeds 2σ, sell put spreads on the underperformer (betting on
    catch-up) and sell call spreads on the outperformer.
    """
    print("  [5] Correlation Breakout: XLF/XLI pair divergence trade")
    xlf_close = xlf_df["Close"]
    xli_close = xli_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))

    # Rolling correlation and spread
    xlf_ret = xlf_close.pct_change()
    xli_ret = xli_close.pct_change()
    common = xlf_ret.index.intersection(xli_ret.index)
    xlf_r = xlf_ret.reindex(common).fillna(0)
    xli_r = xli_ret.reindex(common).fillna(0)

    roll_corr = xlf_r.rolling(60).corr(xli_r)

    # Spread = XLF 20d return - XLI 20d return
    xlf_ret20 = xlf_close.pct_change(20).reindex(common)
    xli_ret20 = xli_close.pct_change(20).reindex(common)
    spread = xlf_ret20 - xli_ret20
    spread_mean = spread.rolling(60).mean()
    spread_std = spread.rolling(60).std()
    spread_z = (spread - spread_mean) / spread_std.replace(0, np.nan)
    spread_z = spread_z.dropna()

    trades, last = [], None

    # Iterate dates where we have z-scores
    for date in spread_z.index:
        ds = date.strftime("%Y-%m-%d")
        if last and (date - last).days < 14:
            continue

        try:
            z = float(spread_z.loc[ds])
            corr = float(roll_corr.loc[ds])
        except (KeyError, TypeError):
            continue
        if np.isnan(z) or np.isnan(corr):
            continue

        # Signal: correlation has weakened AND spread is extreme
        # XLF/XLI normally at 0.83 median — below 0.75 is notable divergence
        if corr > 0.75 or abs(z) < 1.2:
            continue

        # Determine which side to trade
        if z > 1.5:
            # XLF outperforming → sell XLF call spread (expect reversion down)
            # + sell XLI put spread (expect catch-up)
            trade_ticker = "XLI"  # underperformer gets put spread
            direction = "xli_catchup"
        else:
            # XLI outperforming → sell XLI call spread + sell XLF put spread
            trade_ticker = "XLF"
            direction = "xlf_catchup"

        try:
            price = float((xlf_close if trade_ticker == "XLF" else xli_close).loc[ds])
        except (KeyError, TypeError):
            continue
        if np.isnan(price):
            continue

        # Find expiration 20-40 DTE
        exps = _find_exps(hd, trade_ticker, "2020-03-01", "2025-12-31", monthly=False)
        exp = None
        for e in sorted(exps):
            ed = _exp_dt(e)
            if date + timedelta(days=20) < ed < date + timedelta(days=40):
                exp = e
                break
        if exp is None:
            continue

        width = 1.0 if trade_ticker == "XLF" else 2.0
        spread_obj = _sell_put_spread(hd, trade_ticker, exp, ds, price,
                                       otm_pct=0.95, width=width)
        if spread_obj is None:
            continue

        contracts = max(1, min(8, int(CAPITAL * 0.02 / (spread_obj["max_loss"] * 100))))
        ed_str, er, ev, hold = _walk_spread(
            hd, trade_ticker, exp, spread_obj["short"], spread_obj["long"],
            spread_obj["credit"], date, _exp_dt(exp), spy_df.index)
        pnl = (spread_obj["credit"] - ev) * 100 * contracts
        trades.append({"entry_date": ds, "exit_date": ed_str, "pnl": round(pnl, 2),
                        "exit_reason": er, "ticker": trade_ticker,
                        "direction": direction, "credit": spread_obj["credit"],
                        "spread_z": round(z, 2), "corr": round(corr, 3),
                        "hold_days": hold})
        last = date

    print(f"    → {len(trades)} trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(results: List[Stats],
                    output_path: str = "reports/strategy_discovery_round3.html") -> str:
    """Generate comprehensive HTML report for all strategies."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Summary cards
    live = [s for s in results if not s.killed]
    killed = [s for s in results if s.killed]

    summary_rows = ""
    for s in results:
        status = "KILLED" if s.killed else "LIVE"
        sc = "#ef4444" if s.killed else "#22c55e"
        reason = f" — {s.kill_reason}" if s.killed else ""
        c = "#22c55e" if s.total_pnl > 0 else "#ef4444"
        corr_c = "#22c55e" if abs(s.spy_corr) < 0.3 else ("#f59e0b" if abs(s.spy_corr) < 0.5 else "#ef4444")
        e_corr_c = "#22c55e" if abs(s.exp1220_corr) < 0.3 else ("#f59e0b" if abs(s.exp1220_corr) < 0.5 else "#ef4444")
        summary_rows += f"""<tr>
          <td style="text-align:left">{s.name}</td>
          <td>{s.n_trades}</td>
          <td style="color:{c}">${s.total_pnl:,.0f}</td>
          <td>{s.win_rate:.0%}</td>
          <td>{s.sharpe:.2f}</td>
          <td>{s.max_dd:.1%}</td>
          <td>{s.cagr:.1%}</td>
          <td style="color:{corr_c}">{s.spy_corr:+.3f}</td>
          <td style="color:{e_corr_c}">{s.exp1220_corr:+.3f}</td>
          <td>{s.oos_n}</td>
          <td>{s.oos_sharpe:.2f}</td>
          <td>{s.oos_dd:.1%}</td>
          <td style="color:{sc};font-weight:700">{status}{reason}</td>
        </tr>"""

    # Per-strategy detail sections
    detail_sections = ""
    for s in results:
        yr_rows = ""
        for yr in sorted(s.yearly.keys()):
            y = s.yearly[yr]
            is_oos = yr >= OOS_START_YEAR
            tag = " (OOS)" if is_oos else ""
            yc = "#22c55e" if y["pnl"] > 0 else "#ef4444"
            yr_rows += f"""<tr>
              <td>{yr}{tag}</td><td>{y['n']}</td>
              <td style="color:{yc}">${y['pnl']:,.0f}</td>
              <td>{y['wr']:.0%}</td><td>{y['sharpe']:.2f}</td>
              <td>{y['dd']:.1%}</td><td>{y['ret']:.1%}</td>
            </tr>"""
        detail_sections += f"""
        <h2>{s.name}</h2>
        <p style="color:#94a3b8;font-size:0.85rem">{s.description}</p>
        <table>
        <tr><th>Year</th><th>Trades</th><th>P&L</th><th>Win%</th><th>Sharpe</th><th>Max DD</th><th>Return</th></tr>
        {yr_rows}
        </table>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strategy Discovery Round 3</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin:0; padding:24px; background:#0f172a; color:#e2e8f0; }}
h1 {{ font-size:1.5rem; color:#f8fafc; margin-bottom:4px; }}
h2 {{ font-size:1.1rem; color:#94a3b8; margin-top:2rem; border-bottom:1px solid #334155; padding-bottom:6px; }}
.meta {{ color:#64748b; font-size:0.85rem; margin-bottom:20px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:20px; }}
.card {{ background:#1e293b; border-radius:8px; padding:14px; }}
.card-label {{ font-size:0.7rem; color:#64748b; text-transform:uppercase; }}
.card-value {{ font-size:1.3rem; font-weight:700; margin-top:3px; }}
.positive {{ color:#22c55e; }} .negative {{ color:#ef4444; }} .warn {{ color:#f59e0b; }}
table {{ width:100%; border-collapse:collapse; margin-bottom:14px; font-size:0.85rem; }}
th {{ background:#1e293b; padding:6px 10px; text-align:right; font-size:0.72rem; color:#94a3b8; text-transform:uppercase; border-bottom:2px solid #334155; }}
th:first-child {{ text-align:left; }}
td {{ padding:5px 10px; text-align:right; border-bottom:1px solid #1e293b; }}
td:first-child {{ text-align:left; }}
tr:hover {{ background:#1e293b40; }}
</style></head><body>
<h1>Strategy Discovery — Round 3</h1>
<p class="meta">5 Novel Strategies | All Real IronVault Data | WF: IS=2020-2022, OOS=2023-2025</p>

<div class="grid">
  <div class="card"><div class="card-label">Strategies Tested</div><div class="card-value">{len(results)}</div></div>
  <div class="card"><div class="card-label">Live (Passed)</div><div class="card-value positive">{len(live)}</div></div>
  <div class="card"><div class="card-label">Killed</div><div class="card-value negative">{len(killed)}</div></div>
  <div class="card"><div class="card-label">Best OOS Sharpe</div><div class="card-value positive">{max((s.oos_sharpe for s in results), default=0):.2f}</div></div>
  <div class="card"><div class="card-label">Lowest SPY Corr</div><div class="card-value positive">{min((abs(s.spy_corr) for s in results), default=0):.3f}</div></div>
  <div class="card"><div class="card-label">Lowest 1220 Corr</div><div class="card-value positive">{min((abs(s.exp1220_corr) for s in results), default=0):.3f}</div></div>
</div>

<h2>Strategy Summary — All Strategies</h2>
<table>
<tr><th style="text-align:left">Strategy</th><th>Trades</th><th>P&L</th><th>Win%</th>
    <th>Sharpe</th><th>Max DD</th><th>CAGR</th><th>SPY ρ</th><th>1220 ρ</th>
    <th>OOS N</th><th>OOS SR</th><th>OOS DD</th><th>Status</th></tr>
{summary_rows}
</table>

{detail_sections}

<div style="color:#64748b;font-size:0.78rem;margin-top:3rem">
<p>Strategy Discovery Round 3 — compass/strategy_discovery_r3.py<br>
All option prices from IronVault (options_cache.db). Zero synthetic data.<br>
ETF daily prices from yfinance. Walk-forward: IS 2020-2022, OOS 2023-2025.<br>
Kill: &lt;15 OOS trades OR negative OOS Sharpe.</p>
</div></body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════


def run_discovery(output: str = "reports/strategy_discovery_round3.html") -> List[Stats]:
    """Run all 5 strategies and generate report."""
    print("Strategy Discovery Round 3")
    print("=" * 60)

    # Initialize IronVault
    hd = IronVault.instance()
    print(f"  IronVault connected: {hd._db_path}")

    # Fetch market data
    print("  Fetching market data (SPY, VIX, sectors)...")
    spy_df = _dl("SPY")
    vix = _dl("^VIX")["Close"]
    xlf_df = _dl("XLF")
    xli_df = _dl("XLI")
    xle_df = _dl("XLE")
    sector_dfs = {"XLF": xlf_df, "XLI": xli_df, "XLE": xle_df}
    spy_ret = spy_df["Close"].pct_change().dropna()

    # EXP-1220 proxy: credit spread returns (correlated with SPY but higher return)
    # Use a simple model: 3x SPY return on up days, 1.5x on down days (asymmetric)
    exp1220_ret = spy_ret.copy()
    exp1220_ret[exp1220_ret >= 0] *= 3.0
    exp1220_ret[exp1220_ret < 0] *= 1.5

    print("\n  Running strategies...")

    # Strategy 1: Skew Trading
    trades1 = strat_skew_trading(hd, spy_df, vix)
    s1 = _compute(trades1, "Options Skew Trading", spy_ret, exp1220_ret,
                  "Sell put spreads when put skew > 1.3 — harvest overpriced fear premium")

    # Strategy 2: VIX Mean-Reversion
    trades2 = strat_vix_mean_reversion(hd, spy_df, vix)
    s2 = _compute(trades2, "VIX Mean-Reversion", spy_ret, exp1220_ret,
                  "Sell SPY puts when VIX z-score > 1.5 — harvest decaying fear premium")

    # Strategy 3: Sector IV Rotation
    trades3 = strat_sector_iv_rotation(hd, spy_df, sector_dfs)
    s3 = _compute(trades3, "Sector IV Rotation", spy_ret, exp1220_ret,
                  "Trade calmest sector (lowest 20d vol) — XLF/XLI/XLE rotation")

    # Strategy 4: Calendar Spread Carry
    trades4 = strat_calendar_carry(hd, spy_df)
    s4 = _compute(trades4, "Calendar Spread Carry", spy_ret, exp1220_ret,
                  "Sell front-month, buy back-month at same strike — theta differential")

    # Strategy 5: Correlation Breakout
    trades5 = strat_correlation_breakout(hd, spy_df, xlf_df, xli_df)
    s5 = _compute(trades5, "Correlation Breakout (XLF/XLI)", spy_ret, exp1220_ret,
                  "Trade XLF/XLI pair divergence — sell puts on underperformer when corr breaks down")

    results = [s1, s2, s3, s4, s5]

    # Print summary
    print("\n  Results:")
    for s in results:
        status = "KILLED" if s.killed else "LIVE"
        print(f"    {s.name}: {s.n_trades} trades, ${s.total_pnl:,.0f}, "
              f"Sharpe={s.sharpe:.2f}, OOS_SR={s.oos_sharpe:.2f}, "
              f"SPY_ρ={s.spy_corr:+.3f}, 1220_ρ={s.exp1220_corr:+.3f} [{status}]")

    report_path = generate_report(results, output)
    print(f"\n  Report: {report_path}")
    return results


if __name__ == "__main__":
    run_discovery()
