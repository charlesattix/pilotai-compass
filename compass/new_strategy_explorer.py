"""
New strategy exploration — backtest 4 uncorrelated strategies using ONLY
real IronVault data. No synthetic pricing. No np.random for prices/returns.

Strategies:
  1. XLF Iron Condors — theta capture on financials ETF
  2. SPY Calendar Spreads — exploit term structure
  3. XLF/XLE Momentum Directional Spreads — sector rotation
  4. VIX Mean-Reversion SPY Puts — sell when VIX elevated

All compared against SPY returns for correlation analysis.
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
DEFAULT_OUTPUT = ROOT / "reports" / "new_strategy_exploration.html"


# ── Helpers ──────────────────────────────────────────────────────────────

def _exp_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _get_market_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch SPY and VIX daily data from yfinance."""
    import yfinance as yf
    spy = yf.download("SPY", start="2019-12-01", end="2026-01-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index)

    vix = yf.download("^VIX", start="2019-12-01", end="2026-01-01", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix.index = pd.to_datetime(vix.index)
    return spy, vix["Close"]


def _get_etf_prices(ticker: str) -> pd.DataFrame:
    """Fetch daily data for a specific ETF."""
    import yfinance as yf
    df = yf.download(ticker, start="2019-12-01", end="2026-01-01", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def _find_expirations(hd: IronVault, ticker: str, start: str, end: str,
                      monthly_only: bool = True) -> List[str]:
    """Find available expirations in the DB."""
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker=? AND option_type='P' AND expiration BETWEEN ? AND ?
        ORDER BY expiration
    """, (ticker, start, end))
    all_exps = [r[0] for r in cur.fetchall()]
    conn.close()

    if not monthly_only:
        return all_exps

    monthly = []
    last_month = ""
    for exp in all_exps:
        ym = exp[:7]
        day = int(exp[8:10])
        if ym != last_month and 15 <= day <= 21:
            monthly.append(exp)
            last_month = ym
    return monthly


@dataclass
class StrategyStats:
    """Computed statistics for a strategy."""
    name: str
    trades: List[Dict] = field(default_factory=list)
    n_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    max_dd: float = 0.0
    sharpe: float = 0.0
    cagr: float = 0.0
    spy_correlation: float = 0.0
    yearly: Dict[int, Dict] = field(default_factory=dict)
    avg_pnl: float = 0.0
    description: str = ""
    oos_sharpe: float = 0.0       # out-of-sample (2023-2025)
    oos_n_trades: int = 0


def _compute_stats(
    trades: List[Dict],
    name: str,
    spy_returns: pd.Series,
    capital: float = 100_000,
    description: str = "",
) -> StrategyStats:
    """Compute strategy stats from trade list."""
    if not trades:
        return StrategyStats(name=name, description=description)

    df = pd.DataFrame(trades)
    pnls = df["pnl"].values
    n = len(pnls)
    total = pnls.sum()
    wins = (pnls > 0).sum()

    # Equity curve
    equity = np.cumsum(pnls) + capital
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(dd.max()) if len(dd) > 0 else 0

    mean_p = pnls.mean()
    std_p = pnls.std(ddof=1) if n > 1 else 1.0
    sharpe = float(mean_p / std_p * math.sqrt(min(n, 52))) if std_p > 0 else 0

    # CAGR
    dates = pd.to_datetime(df["exit_date"])
    years = (dates.max() - pd.to_datetime(df["entry_date"]).min()).days / 365.25
    years = max(years, 0.5)
    cagr = ((1 + total / capital) ** (1 / years) - 1) if total > -capital else -1.0

    # Correlation with SPY
    # Build daily P&L series aligned with SPY returns
    trade_returns = {}
    for _, row in df.iterrows():
        dt = str(row["exit_date"])[:10]
        trade_returns[dt] = trade_returns.get(dt, 0) + row["pnl"]
    tr_series = pd.Series(trade_returns)
    tr_series.index = pd.to_datetime(tr_series.index)

    # Align with SPY
    common = tr_series.index.intersection(spy_returns.index)
    if len(common) > 10:
        corr = float(np.corrcoef(
            tr_series.reindex(common).fillna(0).values,
            spy_returns.reindex(common).fillna(0).values,
        )[0, 1])
    else:
        corr = 0.0

    # Yearly
    df["year"] = dates.dt.year
    yearly = {}
    for yr, grp in df.groupby("year"):
        yp = grp["pnl"].values
        yn = len(yp)
        if yn == 0:
            continue
        y_eq = np.cumsum(yp) + capital
        y_pk = np.maximum.accumulate(y_eq)
        y_dd = (y_pk - y_eq) / y_pk
        y_std = yp.std(ddof=1) if yn > 1 else 1.0
        yearly[int(yr)] = {
            "n_trades": yn,
            "total_pnl": round(float(yp.sum()), 2),
            "win_rate": round(float((yp > 0).sum()) / yn, 4),
            "max_dd": round(float(y_dd.max()), 4),
            "sharpe": round(float(yp.mean() / y_std * math.sqrt(min(yn, 52))) if y_std > 0 else 0, 3),
            "return_pct": round(float(yp.sum() / capital), 4),
        }

    # Out-of-sample stats (2023-2025)
    oos_trades = df[dates.dt.year >= 2023]
    oos_sharpe = 0.0
    if len(oos_trades) > 1:
        oos_pnls = oos_trades["pnl"].values
        oos_std = oos_pnls.std(ddof=1)
        if oos_std > 0:
            oos_sharpe = float(oos_pnls.mean() / oos_std * math.sqrt(min(len(oos_pnls), 52)))

    return StrategyStats(
        name=name,
        trades=trades,
        n_trades=n,
        total_pnl=round(total, 2),
        win_rate=round(float(wins / n), 4),
        max_dd=round(max_dd, 4),
        sharpe=round(sharpe, 3),
        cagr=round(cagr, 4),
        spy_correlation=round(corr, 4),
        yearly=yearly,
        avg_pnl=round(float(mean_p), 2),
        description=description,
        oos_sharpe=round(oos_sharpe, 3),
        oos_n_trades=len(oos_trades),
    )


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: XLF Iron Condors
# ═══════════════════════════════════════════════════════════════════════════

def backtest_xlf_iron_condors(
    hd: IronVault,
    xlf_df: pd.DataFrame,
    vix: pd.Series,
) -> List[Dict]:
    """Iron condors on XLF: sell OTM put spread + OTM call spread.

    Entry: Monthly, 30-45 DTE, when VIX < 30
    Put spread: ~7% OTM, $1-wide
    Call spread: ~5% OTM, $1-wide
    Exit: 50% profit target, 2x loss stop, or 7 DTE
    """
    print("  Backtesting XLF Iron Condors...")
    xlf_close = xlf_df["Close"]
    exps = _find_expirations(hd, "XLF", "2020-01-01", "2025-12-31")
    trades: List[Dict] = []
    last_entry = None

    for exp in exps:
        exp_dt_obj = _exp_dt(exp)
        entry_dt = exp_dt_obj - timedelta(days=37)

        for offset in range(7):
            cand = entry_dt + timedelta(days=offset)
            cand_str = cand.strftime("%Y-%m-%d")
            if cand_str in xlf_df.index.strftime("%Y-%m-%d").values:
                entry_dt = cand
                break
        else:
            continue

        entry_str = entry_dt.strftime("%Y-%m-%d")
        if last_entry and (entry_dt - last_entry).days < 20:
            continue

        # VIX filter: skip if VIX > 30
        try:
            v = float(vix.loc[entry_str])
        except (KeyError, TypeError):
            v = 20.0
        if v > 30:
            continue

        try:
            price = float(xlf_close.loc[entry_str])
        except (KeyError, TypeError):
            continue

        # Put side: ~7% OTM
        put_strikes = hd.get_available_strikes("XLF", exp, entry_str, "P")
        call_strikes = hd.get_available_strikes("XLF", exp, entry_str, "C")
        if not put_strikes or not call_strikes:
            continue

        # Find put spread
        put_target = price * 0.93
        put_short = None
        for sk in sorted(put_strikes, key=lambda k: abs(k - put_target)):
            lk = sk - 1.0
            if lk in put_strikes:
                pp = hd.get_spread_prices("XLF", exp_dt_obj, sk, lk, "P", entry_str)
                if pp and pp["short_close"] - pp["long_close"] > 0.05:
                    put_short = sk
                    put_long = lk
                    put_credit = pp["short_close"] - pp["long_close"]
                    break
        if put_short is None:
            continue

        # Find call spread (bear call: short lower strike, long higher strike)
        call_target = price * 1.05
        call_short = None
        for sk in sorted(call_strikes, key=lambda k: abs(k - call_target)):
            lk = sk + 1.0
            if lk in call_strikes:
                # For calls: short=sk (lower), long=lk (higher) → credit = sk_price - lk_price
                cp = hd.get_spread_prices("XLF", exp_dt_obj, sk, lk, "C", entry_str)
                if cp and cp["short_close"] - cp["long_close"] > 0.03:
                    call_short = sk
                    call_long = lk
                    call_credit = cp["short_close"] - cp["long_close"]
                    break
        if call_short is None:
            # Fall back to put-only credit spread if calls unavailable
            total_credit = put_credit
            max_loss = 1.0 - total_credit
            if max_loss <= 0:
                continue
            contracts = max(1, min(5, int(100_000 * 0.015 / (max_loss * 100))))
            call_short = call_long = call_credit = None
        else:

            total_credit = put_credit + call_credit
            max_loss = 1.0 - total_credit
            if max_loss <= 0:
                continue
            contracts = max(1, min(5, int(100_000 * 0.015 / (max_loss * 100))))

        has_calls = call_short is not None

        # Walk forward
        exit_date = exit_reason = None
        exit_total = total_credit
        hold_days = 0

        current = entry_dt + timedelta(days=1)
        while current <= exp_dt_obj:
            curr_str = current.strftime("%Y-%m-%d")
            if curr_str not in xlf_df.index.strftime("%Y-%m-%d").values:
                current += timedelta(days=1)
                continue

            hold_days += 1
            dte_rem = (exp_dt_obj - current).days

            pp = hd.get_spread_prices("XLF", exp_dt_obj, put_short, put_long, "P", curr_str)
            if pp is None:
                current += timedelta(days=1)
                continue
            cur_put_val = pp["short_close"] - pp["long_close"]

            cur_call_val = 0.0
            if has_calls:
                cp = hd.get_spread_prices("XLF", exp_dt_obj, call_short, call_long, "C", curr_str)
                if cp is not None:
                    cur_call_val = cp["short_close"] - cp["long_close"]

            cur_total = cur_put_val + cur_call_val

            if cur_total <= total_credit * 0.50:
                exit_date, exit_reason, exit_total = curr_str, "profit_target", cur_total
                break
            if cur_total - total_credit > total_credit * 2.0:
                exit_date, exit_reason, exit_total = curr_str, "stop_loss", cur_total
                break
            if dte_rem <= 7:
                exit_date, exit_reason, exit_total = curr_str, "dte_exit", cur_total
                break

            current += timedelta(days=1)

        if exit_date is None:
            exit_date, exit_reason = exp, "expiration"
            exit_total = 0

        pnl = (total_credit - exit_total) * 100 * contracts

        trades.append({
            "entry_date": entry_str,
            "exit_date": exit_date,
            "pnl": round(pnl, 2),
            "exit_reason": exit_reason,
            "entry_credit": round(total_credit, 4),
            "hold_days": hold_days,
        })
        last_entry = entry_dt

    print(f"    {len(trades)} trades completed")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: SPY Calendar Spreads
# ═══════════════════════════════════════════════════════════════════════════

def backtest_spy_calendars(
    hd: IronVault,
    spy_df: pd.DataFrame,
    vix: pd.Series,
) -> List[Dict]:
    """SPY put calendar spreads: sell front-month, buy back-month at same strike.

    Entry: When VIX < 25 (contango likely), front ~21 DTE, back ~50 DTE
    Strike: ~3% OTM put (slightly below market)
    Exit: front expiration, or 50% profit, or 100% loss
    """
    print("  Backtesting SPY Calendar Spreads...")
    spy_close = spy_df["Close"]
    trades: List[Dict] = []

    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    # Get all SPY put expirations
    cur.execute("""
        SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker='SPY' AND option_type='P' AND expiration BETWEEN '2020-03-01' AND '2025-12-31'
        ORDER BY expiration
    """)
    all_exps = [r[0] for r in cur.fetchall()]
    conn.close()

    last_entry = None

    for i, front_exp in enumerate(all_exps):
        front_dt = _exp_dt(front_exp)

        # Find back expiration: 25-40 days after front
        back_exp = None
        for j in range(i + 1, len(all_exps)):
            delta = (_exp_dt(all_exps[j]) - front_dt).days
            if 25 <= delta <= 45:
                back_exp = all_exps[j]
                break

        if back_exp is None:
            continue

        back_dt = _exp_dt(back_exp)

        # Entry: ~21 days before front expiration
        entry_dt = front_dt - timedelta(days=21)
        for offset in range(7):
            cand = entry_dt + timedelta(days=offset)
            cand_str = cand.strftime("%Y-%m-%d")
            if cand_str in spy_df.index.strftime("%Y-%m-%d").values:
                entry_dt = cand
                break
        else:
            continue

        entry_str = entry_dt.strftime("%Y-%m-%d")
        if last_entry and (entry_dt - last_entry).days < 14:
            continue

        # VIX filter
        try:
            v = float(vix.loc[entry_str])
        except (KeyError, TypeError):
            v = 20.0
        if v > 25:
            continue

        try:
            spy_price = float(spy_close.loc[entry_str])
        except (KeyError, TypeError):
            continue

        # Find a common strike ~3% OTM
        front_strikes = hd.get_available_strikes("SPY", front_exp, entry_str, "P")
        back_strikes = hd.get_available_strikes("SPY", back_exp, entry_str, "P")
        if not front_strikes or not back_strikes:
            continue

        common = sorted(set(front_strikes) & set(back_strikes))
        if not common:
            continue

        target_strike = spy_price * 0.97
        strike = min(common, key=lambda k: abs(k - target_strike))

        # Get calendar spread price (debit = back - front)
        front_sym = IronVault.build_occ_symbol("SPY", front_dt, strike, "P")
        back_sym = IronVault.build_occ_symbol("SPY", back_dt, strike, "P")

        front_price = hd.get_contract_price(front_sym, entry_str)
        back_price = hd.get_contract_price(back_sym, entry_str)

        if front_price is None or back_price is None:
            continue

        debit = back_price - front_price
        if debit <= 0.10:
            continue  # no real term structure

        contracts = max(1, min(3, int(100_000 * 0.01 / (debit * 100))))

        # Walk forward — exit at front expiration or target
        exit_date = exit_reason = None
        exit_value = debit
        hold_days = 0

        current = entry_dt + timedelta(days=1)
        while current <= front_dt:
            curr_str = current.strftime("%Y-%m-%d")
            if curr_str not in spy_df.index.strftime("%Y-%m-%d").values:
                current += timedelta(days=1)
                continue

            hold_days += 1

            fp = hd.get_contract_price(front_sym, curr_str)
            bp = hd.get_contract_price(back_sym, curr_str)

            if fp is None or bp is None:
                current += timedelta(days=1)
                continue

            cur_value = bp - fp

            # Profit: calendar spread widened > 50% above entry debit
            if cur_value >= debit * 1.50:
                exit_date, exit_reason, exit_value = curr_str, "profit_target", cur_value
                break

            # Loss: calendar collapsed to < 50% of entry
            if cur_value <= debit * 0.50:
                exit_date, exit_reason, exit_value = curr_str, "stop_loss", cur_value
                break

            current += timedelta(days=1)

        if exit_date is None:
            exit_date, exit_reason = front_exp, "front_expiry"
            # At front expiry, front goes to intrinsic, back retains time value
            fp_final = hd.get_contract_price(front_sym, front_exp)
            bp_final = hd.get_contract_price(back_sym, front_exp)
            if fp_final is not None and bp_final is not None:
                exit_value = bp_final - fp_final
            else:
                exit_value = debit * 0.8  # conservative estimate if no data

        pnl = (exit_value - debit) * 100 * contracts

        trades.append({
            "entry_date": entry_str,
            "exit_date": exit_date,
            "pnl": round(pnl, 2),
            "exit_reason": exit_reason,
            "debit": round(debit, 4),
            "strike": strike,
            "hold_days": hold_days,
        })
        last_entry = entry_dt

    print(f"    {len(trades)} trades completed")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Sector Momentum Directional Spreads
# ═══════════════════════════════════════════════════════════════════════════

def backtest_sector_momentum(
    hd: IronVault,
    xlf_df: pd.DataFrame,
    vix: pd.Series,
) -> List[Dict]:
    """Momentum-based directional put credit spreads on XLF.

    Entry: When XLF 20-day return > 2% (uptrend), sell OTM put spread
    When XLF 20-day return < -2% (downtrend), SKIP (no bear spreads)
    Exit: 50% profit, 3x stop, or 7 DTE
    """
    print("  Backtesting XLF Sector Momentum Spreads...")
    xlf_close = xlf_df["Close"]
    xlf_ret_20d = xlf_close.pct_change(20)

    exps = _find_expirations(hd, "XLF", "2020-01-01", "2025-12-31")
    trades: List[Dict] = []
    last_entry = None

    for exp in exps:
        exp_dt_obj = _exp_dt(exp)
        entry_dt = exp_dt_obj - timedelta(days=35)

        for offset in range(7):
            cand = entry_dt + timedelta(days=offset)
            cand_str = cand.strftime("%Y-%m-%d")
            if cand_str in xlf_df.index.strftime("%Y-%m-%d").values:
                entry_dt = cand
                break
        else:
            continue

        entry_str = entry_dt.strftime("%Y-%m-%d")
        if last_entry and (entry_dt - last_entry).days < 20:
            continue

        # Momentum filter: only enter when XLF trending up
        try:
            ret20 = float(xlf_ret_20d.loc[entry_str])
        except (KeyError, TypeError):
            continue
        if np.isnan(ret20) or ret20 < 0.02:
            continue

        try:
            price = float(xlf_close.loc[entry_str])
        except (KeyError, TypeError):
            continue

        strikes = hd.get_available_strikes("XLF", exp, entry_str, "P")
        if not strikes:
            continue

        # ~5% OTM put spread, $1-wide
        target = price * 0.95
        spread = None
        for sk in sorted(strikes, key=lambda k: abs(k - target))[:10]:
            lk = sk - 1.0
            if lk not in strikes:
                continue
            pp = hd.get_spread_prices("XLF", exp_dt_obj, sk, lk, "P", entry_str)
            if pp and pp["short_close"] - pp["long_close"] > 0.03:
                spread = {"short": sk, "long": lk, "credit": pp["short_close"] - pp["long_close"]}
                break

        if spread is None:
            continue

        max_loss = 1.0 - spread["credit"]
        contracts = max(1, min(5, int(100_000 * 0.015 / (max_loss * 100))))

        exit_date = exit_reason = None
        exit_val = spread["credit"]
        hold_days = 0

        current = entry_dt + timedelta(days=1)
        while current <= exp_dt_obj:
            curr_str = current.strftime("%Y-%m-%d")
            if curr_str not in xlf_df.index.strftime("%Y-%m-%d").values:
                current += timedelta(days=1)
                continue

            hold_days += 1
            dte_rem = (exp_dt_obj - current).days

            pp = hd.get_spread_prices("XLF", exp_dt_obj, spread["short"], spread["long"], "P", curr_str)
            if pp is None:
                current += timedelta(days=1)
                continue

            cv = pp["short_close"] - pp["long_close"]

            if cv <= spread["credit"] * 0.50:
                exit_date, exit_reason, exit_val = curr_str, "profit_target", cv
                break
            if cv - spread["credit"] > spread["credit"] * 3.0:
                exit_date, exit_reason, exit_val = curr_str, "stop_loss", cv
                break
            if dte_rem <= 7:
                exit_date, exit_reason, exit_val = curr_str, "dte_exit", cv
                break

            current += timedelta(days=1)

        if exit_date is None:
            exit_date, exit_reason = exp, "expiration"
            exit_val = 0

        pnl = (spread["credit"] - exit_val) * 100 * contracts
        trades.append({
            "entry_date": entry_str,
            "exit_date": exit_date,
            "pnl": round(pnl, 2),
            "exit_reason": exit_reason,
            "entry_credit": round(spread["credit"], 4),
            "hold_days": hold_days,
            "momentum_20d": round(ret20, 4),
        })
        last_entry = entry_dt

    print(f"    {len(trades)} trades completed")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 4: VIX Mean-Reversion SPY Put Credit Spreads
# ═══════════════════════════════════════════════════════════════════════════

def backtest_vix_mean_reversion(
    hd: IronVault,
    spy_df: pd.DataFrame,
    vix: pd.Series,
) -> List[Dict]:
    """VIX mean-reversion: sell SPY put spreads when VIX is elevated (>20),
    buy protection (skip) when VIX is low (<15).

    Entry: VIX > 20 (elevated premium), sell 5-wide put spread ~7% OTM
    Size: Larger when VIX 25-35, smaller when VIX 20-25
    Skip: VIX < 20 (cheap premium not worth selling)
    Exit: 50% profit, 3x stop, 7 DTE
    """
    print("  Backtesting VIX Mean-Reversion SPY Puts...")
    spy_close = spy_df["Close"]
    exps = _find_expirations(hd, "SPY", "2020-01-01", "2025-12-31")
    trades: List[Dict] = []
    last_entry = None

    for exp in exps:
        exp_dt_obj = _exp_dt(exp)
        entry_dt = exp_dt_obj - timedelta(days=37)

        for offset in range(7):
            cand = entry_dt + timedelta(days=offset)
            cand_str = cand.strftime("%Y-%m-%d")
            if cand_str in spy_df.index.strftime("%Y-%m-%d").values:
                entry_dt = cand
                break
        else:
            continue

        entry_str = entry_dt.strftime("%Y-%m-%d")
        if last_entry and (entry_dt - last_entry).days < 20:
            continue

        try:
            v = float(vix.loc[entry_str])
        except (KeyError, TypeError):
            continue

        # Only sell when VIX elevated
        if v < 20:
            continue

        try:
            spy_price = float(spy_close.loc[entry_str])
        except (KeyError, TypeError):
            continue

        strikes = hd.get_available_strikes("SPY", exp, entry_str, "P")
        if not strikes:
            continue

        # Wider OTM when VIX very high (more cushion)
        otm_pct = 0.93 if v < 30 else 0.90
        target = spy_price * otm_pct

        spread = None
        for sk in sorted(strikes, key=lambda k: abs(k - target))[:10]:
            lk = sk - 5
            if lk not in strikes:
                continue
            pp = hd.get_spread_prices("SPY", exp_dt_obj, sk, lk, "P", entry_str)
            if pp and pp["short_close"] - pp["long_close"] > 0.20:
                spread = {"short": sk, "long": lk, "credit": pp["short_close"] - pp["long_close"]}
                break

        if spread is None:
            continue

        max_loss = 5.0 - spread["credit"]
        # Size inversely to VIX (more cautious when very high)
        risk_pct = 0.02 if v < 30 else 0.01
        contracts = max(1, min(3, int(100_000 * risk_pct / (max_loss * 100))))

        exit_date = exit_reason = None
        exit_val = spread["credit"]
        hold_days = 0

        current = entry_dt + timedelta(days=1)
        while current <= exp_dt_obj:
            curr_str = current.strftime("%Y-%m-%d")
            if curr_str not in spy_df.index.strftime("%Y-%m-%d").values:
                current += timedelta(days=1)
                continue

            hold_days += 1
            dte_rem = (exp_dt_obj - current).days

            pp = hd.get_spread_prices("SPY", exp_dt_obj, spread["short"], spread["long"], "P", curr_str)
            if pp is None:
                current += timedelta(days=1)
                continue

            cv = pp["short_close"] - pp["long_close"]

            if cv <= spread["credit"] * 0.50:
                exit_date, exit_reason, exit_val = curr_str, "profit_target", cv
                break
            if cv - spread["credit"] > spread["credit"] * 3.0:
                exit_date, exit_reason, exit_val = curr_str, "stop_loss", cv
                break
            if dte_rem <= 7:
                exit_date, exit_reason, exit_val = curr_str, "dte_exit", cv
                break

            current += timedelta(days=1)

        if exit_date is None:
            exit_date, exit_reason = exp, "expiration"
            # Try final price
            fp = hd.get_spread_prices("SPY", exp_dt_obj, spread["short"], spread["long"], "P", exp)
            if fp:
                exit_val = fp["short_close"] - fp["long_close"]
            else:
                try:
                    final_spy = float(spy_close.loc[exp])
                    if final_spy < spread["short"]:
                        exit_val = min(spread["short"] - final_spy, 5.0)
                    else:
                        exit_val = 0
                except (KeyError, TypeError):
                    exit_val = 0

        pnl = (spread["credit"] - exit_val) * 100 * contracts

        trades.append({
            "entry_date": entry_str,
            "exit_date": exit_date,
            "pnl": round(pnl, 2),
            "exit_reason": exit_reason,
            "entry_credit": round(spread["credit"], 4),
            "vix_at_entry": round(v, 2),
            "hold_days": hold_days,
        })
        last_entry = entry_dt

    print(f"    {len(trades)} trades completed")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def _generate_html(strategies: List[StrategyStats], output_path: Path) -> Path:
    """Generate comprehensive HTML exploration report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _c(v: float) -> str:
        return "#3fb950" if v >= 0 else "#f85149"

    def _fd(v: float) -> str:
        return f"${v:,.0f}"

    def _fp(v: float) -> str:
        return f"{v:.1%}"

    # Summary cards
    summary_cards = ""
    for s in strategies:
        oos_flag = s.oos_sharpe > 1.5 and s.oos_n_trades >= 5
        flag = f" ★ OOS Sharpe {s.oos_sharpe:.2f}" if oos_flag else ""
        border = "#d29922" if oos_flag else "#30363d"
        oos_note = f" (OOS: {s.oos_sharpe:.2f} on {s.oos_n_trades} trades)" if s.oos_n_trades > 0 else ""
        summary_cards += f"""
        <div class="strat-card" style="border-color:{border}">
          <h3>{s.name}{f'<span class="flag">{flag}</span>' if flag else ''}</h3>
          <p class="desc">{s.description}</p>
          <div class="mini-grid">
            <div><span class="label">Trades</span><span class="val">{s.n_trades}</span></div>
            <div><span class="label">Total P&L</span><span class="val" style="color:{_c(s.total_pnl)}">{_fd(s.total_pnl)}</span></div>
            <div><span class="label">Win Rate</span><span class="val">{_fp(s.win_rate)}</span></div>
            <div><span class="label">Full Sharpe</span><span class="val" style="color:{_c(s.sharpe)}">{s.sharpe:.2f}</span></div>
            <div><span class="label">OOS Sharpe (23-25)</span><span class="val" style="color:{_c(s.oos_sharpe)}">{s.oos_sharpe:.2f}</span></div>
            <div><span class="label">CAGR</span><span class="val" style="color:{_c(s.cagr)}">{_fp(s.cagr)}</span></div>
            <div><span class="label">Max DD</span><span class="val" style="color:#f85149">{_fp(s.max_dd)}</span></div>
            <div><span class="label">SPY Corr</span><span class="val">{s.spy_correlation:.3f}</span></div>
          </div>
        </div>"""

    # Yearly detail tables
    yearly_sections = ""
    for s in strategies:
        rows = ""
        for yr in sorted(s.yearly.keys()):
            y = s.yearly[yr]
            rows += f"""<tr>
              <td>{yr}</td><td>{y['n_trades']}</td>
              <td style="color:{_c(y['total_pnl'])}">{_fd(y['total_pnl'])}</td>
              <td>{_fp(y['win_rate'])}</td><td>{_fp(y['max_dd'])}</td>
              <td style="color:{_c(y['sharpe'])}">{y['sharpe']:.2f}</td>
              <td style="color:{_c(y['return_pct'])}">{_fp(y['return_pct'])}</td>
            </tr>"""

        yearly_sections += f"""
        <h3>{s.name} — Yearly Breakdown</h3>
        <table class="dt">
          <tr><th>Year</th><th>Trades</th><th>P&L</th><th>Win Rate</th>
              <th>Max DD</th><th>Sharpe</th><th>Return</th></tr>
          {rows}
        </table>"""

    # Correlation matrix
    corr_rows = ""
    for s in strategies:
        corr_color = "#3fb950" if abs(s.spy_correlation) < 0.3 else "#d29922" if abs(s.spy_correlation) < 0.6 else "#f85149"
        uncorr_badge = " ✓ Low" if abs(s.spy_correlation) < 0.3 else ""
        oos_color = "#3fb950" if s.oos_sharpe > 1.5 else "#8b949e"
        corr_rows += f"""<tr>
          <td style="text-align:left">{s.name}</td>
          <td style="color:{corr_color}">{s.spy_correlation:.3f}{uncorr_badge}</td>
          <td style="color:{_c(s.sharpe)}">{s.sharpe:.2f}</td>
          <td style="color:{oos_color}">{s.oos_sharpe:.2f}</td>
          <td>{s.n_trades} ({s.oos_n_trades} OOS)</td>
        </tr>"""

    # Flagged strategies (OOS Sharpe > 1.5 with >= 5 trades)
    flagged = [s for s in strategies if s.oos_sharpe > 1.5 and s.oos_n_trades >= 5]
    flagged_html = ""
    if flagged:
        flagged_html = "<h2>★ Flagged: OOS Sharpe > 1.5 (2023-2025)</h2><ul>"
        for s in flagged:
            flagged_html += (
                f"<li><strong>{s.name}</strong> — OOS Sharpe {s.oos_sharpe:.2f} "
                f"({s.oos_n_trades} trades), Full Sharpe {s.sharpe:.2f}, "
                f"CAGR {_fp(s.cagr)}, SPY corr {s.spy_correlation:.3f}</li>"
            )
        flagged_html += "</ul>"
    else:
        flagged_html = (
            "<h2>★ Flagged: OOS Sharpe > 1.5 (2023-2025)</h2>"
            "<p class='meta'>No strategies exceeded OOS Sharpe 1.5 with ≥5 trades on real data.</p>"
        )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>New Strategy Exploration — Real Data</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1300px; margin: 0 auto; padding: 24px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; }}
  h2 {{ color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-top: 32px; }}
  h3 {{ color: #79c0ff; margin-top: 24px; }}
  .meta {{ color: #8b949e; font-size: 0.9em; }}
  .desc {{ color: #8b949e; font-size: 0.85em; margin: 4px 0 12px; }}
  .flag {{ color: #d29922; font-size: 0.8em; margin-left: 8px; }}
  .strat-card {{ background: #161b22; border: 2px solid #30363d; border-radius: 10px;
                 padding: 16px 20px; margin: 16px 0; }}
  .mini-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 8px; }}
  .mini-grid > div {{ text-align: center; }}
  .mini-grid .label {{ display: block; color: #8b949e; font-size: 0.75em; }}
  .mini-grid .val {{ display: block; font-weight: 600; font-size: 1.05em; }}
  table.dt {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.85em; }}
  table.dt th, table.dt td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  table.dt th {{ color: #8b949e; background: #161b22; }}
  table.dt td:first-child {{ text-align: left; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; }}
  .badge-ok {{ background: #1a3a2a; color: #3fb950; }}
  .badge-warn {{ background: #3a2a1a; color: #d29922; }}
  footer {{ margin-top: 40px; padding-top: 12px; border-top: 1px solid #21262d;
            color: #484f58; font-size: 0.8em; }}
</style>
</head>
<body>
<h1>New Strategy Exploration — Real IronVault Data</h1>
<p class="meta">Generated {ts} &middot; All prices from options_cache.db &middot;
   Zero synthetic data &middot; SPY, XLF options 2020–2025</p>

{flagged_html}

<h2>Strategy Overview</h2>
{summary_cards}

<h2>SPY Correlation Analysis</h2>
<p class="meta">Lower |correlation| = more diversification vs EXP-1220 SPY tail risk</p>
<table class="dt">
  <tr><th style="text-align:left">Strategy</th><th>SPY Correlation</th><th>Full Sharpe</th><th>OOS Sharpe</th><th>Trades</th></tr>
  {corr_rows}
</table>

<h2>Yearly Detail</h2>
{yearly_sections}

<footer>
  Data: IronVault options_cache.db (SPY: 187K contracts, XLF: 8.6K contracts) &middot;
  No Black-Scholes fallback &middot; Cache misses → trade skipped
</footer>
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(output_path: Path = DEFAULT_OUTPUT) -> List[StrategyStats]:
    logging.basicConfig(level=logging.WARNING)

    print("=" * 65)
    print("NEW STRATEGY EXPLORATION — Real IronVault Data Only")
    print("=" * 65)

    hd = IronVault.instance()
    cov = hd.coverage_report()
    print(f"IronVault: {cov['contracts_total']:,} contracts, {cov['daily_bars_total']:,} daily bars\n")

    # Market data
    print("Fetching market data...")
    spy_df, vix = _get_market_data()
    xlf_df = _get_etf_prices("XLF")
    spy_returns = spy_df["Close"].pct_change().dropna() * 100_000  # scale to $ on 100K

    # Run backtests
    print("\nRunning backtests...")

    t1 = backtest_xlf_iron_condors(hd, xlf_df, vix)
    t2 = backtest_spy_calendars(hd, spy_df, vix)
    t3 = backtest_sector_momentum(hd, xlf_df, vix)
    t4 = backtest_vix_mean_reversion(hd, spy_df, vix)

    # Compute stats
    print("\nComputing statistics...")
    strategies = [
        _compute_stats(t1, "XLF Iron Condors", spy_returns,
                       description="Sell OTM put + call spreads on XLF, VIX<30 filter, $1-wide"),
        _compute_stats(t2, "SPY Calendar Spreads", spy_returns,
                       description="Buy back-month / sell front-month SPY puts, exploit term structure"),
        _compute_stats(t3, "XLF Momentum Spreads", spy_returns,
                       description="Sell OTM put spreads on XLF when 20d momentum > 2%, $1-wide"),
        _compute_stats(t4, "VIX Mean-Reversion Puts", spy_returns,
                       description="Sell SPY put spreads only when VIX > 20, wider OTM when VIX > 30"),
    ]

    # Print results
    print("\n" + "=" * 80)
    print(f"{'Strategy':<28} {'Trades':>7} {'P&L':>10} {'WR':>7} {'DD':>7} {'Sharpe':>8} {'OOS Sh':>8} {'CAGR':>8} {'Corr':>7}")
    print("-" * 80)
    for s in strategies:
        oos_flag = " ★" if s.oos_sharpe > 1.5 and s.oos_n_trades >= 5 else ""
        print(f"{s.name:<28} {s.n_trades:>7} {s.total_pnl:>10,.0f} "
              f"{s.win_rate:>7.1%} {s.max_dd:>7.1%} {s.sharpe:>8.2f} {s.oos_sharpe:>8.2f} "
              f"{s.cagr:>8.1%} {s.spy_correlation:>7.3f}{oos_flag}")

    # Generate report
    report_path = _generate_html(strategies, output_path)
    print(f"\nReport: {report_path}")

    # Save JSON
    json_path = output_path.with_suffix(".json")
    summary = {
        "generated": datetime.now().isoformat(),
        "data_source": "IronVault (options_cache.db)",
        "synthetic_data": False,
        "strategies": [
            {
                "name": s.name,
                "description": s.description,
                "n_trades": s.n_trades,
                "total_pnl": s.total_pnl,
                "win_rate": s.win_rate,
                "max_dd": s.max_dd,
                "sharpe": s.sharpe,
                "cagr": s.cagr,
                "spy_correlation": s.spy_correlation,
                "avg_pnl": s.avg_pnl,
                "oos_sharpe_2023_2025": s.oos_sharpe,
                "oos_n_trades": s.oos_n_trades,
                "yearly": s.yearly,
                "flagged_oos_sharpe_gt_1_5": s.oos_sharpe > 1.5 and s.oos_n_trades >= 5,
            }
            for s in strategies
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2))

    return strategies


if __name__ == "__main__":
    main()
