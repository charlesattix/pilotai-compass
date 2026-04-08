"""
EXP-1820 Dispersion Strategy — Production Version
====================================================
Clean, deployable implementation of the relative-vol premium strategy.
Same interface pattern as other production strategies (EXP-1710, EXP-1780).

Selected from hardening sweep — config frozen:
  vol_ratio_threshold: 1.15 (sector must be 15%+ richer than SPY)
  otm_pct:             0.05 (5% OTM puts)
  sector_width:        $2 (or $1 for <$80 tickers)
  spy_width:           $5
  profit_target:       50% of credit
  stop_loss:           2x credit
  dte_exit:            7 days
  risk_per_trade:      2% of capital
  min_spacing_days:    20 per sector

Rule Zero: 100% real IronVault option prices + Yahoo spot. Zero synthetic.
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
DB_PATH = ROOT / "data" / "options_cache.db"
TRADING_DAYS = 252
CAPITAL = 100_000
COMMISSION = 0.65  # $/contract/leg (4 total per spread round-trip)

# Production config (frozen)
PRODUCTION_CONFIG = {
    "name": "EXP-1820 Dispersion Relative Vol Premium",
    "version": "1.0",
    "index": "SPY",
    "sectors": ["XLF", "XLI", "XLK", "XLE"],
    "vol_ratio_threshold": 1.15,
    "otm_pct": 0.05,
    "spy_width": 5.0,
    "sector_width_large": 2.0,   # ≥$80 tickers
    "sector_width_small": 1.0,   # <$80 tickers
    "profit_target": 0.50,
    "stop_loss_multiplier": 2.0,
    "dte_exit_days": 7,
    "risk_per_trade": 0.02,
    "min_spacing_days": 20,
    "min_credit": 0.02,
    "target_dte": 35,
}


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def corrected_sharpe(rets: np.ndarray, rf: float = 0.045) -> float:
    """Arithmetic daily mean Sharpe."""
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
    """Trade-level Sharpe."""
    if len(pnls) < 2:
        return 0.0
    p = np.asarray(pnls, dtype=np.float64)
    s = float(np.std(p, ddof=1))
    if s < 1e-8:
        return 0.0
    return float(np.mean(p) / s * math.sqrt(min(len(p), 52)))


# ═══════════════════════════════════════════════════════════════════════════
# Data queries (direct SQL on IronVault)
# ═══════════════════════════════════════════════════════════════════════════

def load_spot(ticker: str, start: str = "2020-01-01", end: str = "2026-01-01") -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df["Close"]


def _friday_exps(ticker: str, start: str, end: str) -> List[str]:
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


def _strikes(ticker: str, exp: str, date_str: str, opt_type: str) -> List[float]:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT oc.strike
        FROM option_contracts oc
        JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
        WHERE oc.ticker=? AND oc.expiration=? AND oc.option_type=? AND od.date=?
        ORDER BY oc.strike
    """, (ticker, exp, opt_type, date_str))
    s = [float(r[0]) for r in cur.fetchall()]
    conn.close()
    return s


def _spread_close(ticker: str, exp: str, short_k: float, long_k: float,
                  opt_type: str, date_str: str) -> Optional[Dict]:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
        SELECT oc.strike, od.close
        FROM option_contracts oc
        JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
        WHERE oc.ticker=? AND oc.expiration=? AND oc.option_type=?
          AND oc.strike IN (?, ?) AND od.date=? AND od.close IS NOT NULL
    """, (ticker, exp, opt_type, short_k, long_k, date_str))
    rows = {float(r[0]): float(r[1]) for r in cur.fetchall() if r[1] is not None}
    conn.close()
    if short_k not in rows or long_k not in rows:
        return None
    return {"short_close": rows[short_k], "long_close": rows[long_k]}


# ═══════════════════════════════════════════════════════════════════════════
# Spread finder
# ═══════════════════════════════════════════════════════════════════════════

def find_put_spread(ticker: str, exp: str, date_str: str, spot: float,
                    otm_pct: float, width: float, min_credit: float) -> Optional[Dict]:
    """Find 5% OTM put spread with adequate credit."""
    strikes = _strikes(ticker, exp, date_str, "P")
    if not strikes:
        return None

    put_target = spot * (1 - otm_pct)
    candidates = [s for s in strikes if s <= put_target]
    if not candidates:
        return None
    put_short = max(candidates)
    put_long = put_short - width

    if put_long not in strikes:
        nearest = [s for s in strikes if s < put_short and s >= put_short - width - 2]
        if not nearest:
            return None
        put_long = max(nearest)

    pp = _spread_close(ticker, exp, put_short, put_long, "P", date_str)
    if pp is None:
        return None

    credit = pp["short_close"] - pp["long_close"]
    actual_width = put_short - put_long
    if credit <= min_credit or actual_width <= 0:
        return None

    return {
        "ticker": ticker,
        "put_short": put_short,
        "put_long": put_long,
        "width": actual_width,
        "credit": round(credit, 3),
        "credit_pct_of_spot": round(credit / spot * 100, 4),
        "max_loss": round(actual_width - credit, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Strategy class
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_date: str
    exit_date: str
    exp: str
    ticker: str
    spot: float
    put_short: float
    put_long: float
    credit: float
    contracts: int
    exit_reason: str
    pnl: float
    vol_ratio: float
    avg_contracts_at_entry: int  # for capacity calc


class DispersionStrategy:
    """Production dispersion (relative vol premium) strategy.

    Usage:
        strat = DispersionStrategy()
        trades = strat.backtest(start="2020-06-01", end="2026-01-01")
        metrics = strat.metrics(trades)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**PRODUCTION_CONFIG, **(config or {})}

    def backtest(
        self,
        start: str = "2020-06-01",
        end: str = "2026-01-01",
    ) -> List[Trade]:
        cfg = self.config

        # Load spots
        spots = {}
        for t in [cfg["index"]] + cfg["sectors"]:
            try:
                spots[t] = load_spot(t, start=start, end=end)
            except Exception:
                pass

        if cfg["index"] not in spots:
            return []

        # Get SPY expirations
        exps = _friday_exps(cfg["index"], start, end)
        trades: List[Trade] = []
        last_entry_by_ticker = {t: None for t in cfg["sectors"]}

        for exp in exps:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            entry_dt = exp_dt - timedelta(days=cfg["target_dte"])

            # Find valid trading day
            for off in range(7):
                cand = entry_dt + timedelta(days=off)
                if pd.Timestamp(cand) in spots[cfg["index"]].index:
                    entry_dt = cand
                    break
            else:
                continue

            entry_str = entry_dt.strftime("%Y-%m-%d")

            # SPY vol proxy
            if pd.Timestamp(entry_dt) not in spots[cfg["index"]].index:
                continue
            spy_spot = float(spots[cfg["index"]].loc[pd.Timestamp(entry_dt)])
            spy_spread = find_put_spread(
                cfg["index"], exp, entry_str, spy_spot,
                otm_pct=cfg["otm_pct"],
                width=cfg["spy_width"],
                min_credit=cfg["min_credit"],
            )
            if spy_spread is None:
                continue
            spy_vol_proxy = spy_spread["credit_pct_of_spot"]
            if spy_vol_proxy <= 0:
                continue

            # Scan sectors
            for sector in cfg["sectors"]:
                if sector not in spots:
                    continue

                # Spacing check
                if last_entry_by_ticker[sector] is not None:
                    if (entry_dt - last_entry_by_ticker[sector]).days < cfg["min_spacing_days"]:
                        continue

                if pd.Timestamp(entry_dt) not in spots[sector].index:
                    continue
                sec_spot = float(spots[sector].loc[pd.Timestamp(entry_dt)])

                sec_width = (cfg["sector_width_small"] if sec_spot < 80
                             else cfg["sector_width_large"])

                sec_spread = find_put_spread(
                    sector, exp, entry_str, sec_spot,
                    otm_pct=cfg["otm_pct"],
                    width=sec_width,
                    min_credit=cfg["min_credit"],
                )
                if sec_spread is None:
                    continue

                sec_vol_proxy = sec_spread["credit_pct_of_spot"]
                ratio = sec_vol_proxy / spy_vol_proxy

                if ratio < cfg["vol_ratio_threshold"]:
                    continue

                # Position sizing
                max_loss = sec_spread["max_loss"]
                risk_budget = CAPITAL * cfg["risk_per_trade"]
                contracts = max(1, min(15, int(risk_budget / (max_loss * 100))))

                # Walk to exit
                exit_date = exp
                exit_reason = "expiration"
                exit_credit = 0.0

                cur_dt = entry_dt + timedelta(days=1)
                sec_spot_idx = spots[sector].index
                while cur_dt <= exp_dt:
                    cs = cur_dt.strftime("%Y-%m-%d")
                    if pd.Timestamp(cur_dt) not in sec_spot_idx:
                        cur_dt += timedelta(days=1)
                        continue

                    pp = _spread_close(sector, exp, sec_spread["put_short"],
                                       sec_spread["put_long"], "P", cs)
                    if pp is None:
                        cur_dt += timedelta(days=1)
                        continue

                    cur_val = pp["short_close"] - pp["long_close"]

                    if cur_val <= sec_spread["credit"] * (1 - cfg["profit_target"]):
                        exit_date, exit_reason, exit_credit = cs, "profit_target", cur_val
                        break
                    if cur_val - sec_spread["credit"] > sec_spread["credit"] * cfg["stop_loss_multiplier"]:
                        exit_date, exit_reason, exit_credit = cs, "stop_loss", cur_val
                        break
                    if (exp_dt - cur_dt).days <= cfg["dte_exit_days"]:
                        exit_date, exit_reason, exit_credit = cs, "dte_exit", cur_val
                        break

                    cur_dt += timedelta(days=1)

                if exit_reason == "expiration":
                    exit_credit = 0.0

                # Commissions
                commission = 2 * 2 * COMMISSION * contracts
                pnl = (sec_spread["credit"] - exit_credit) * 100 * contracts - commission

                trades.append(Trade(
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
                    avg_contracts_at_entry=contracts,
                ))
                last_entry_by_ticker[sector] = entry_dt

        return trades

    def metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        if not trades:
            return {"n": 0, "pnl": 0, "wr": 0, "sharpe": 0}

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

        is_mask = dates.dt.year <= 2022
        oos_mask = dates.dt.year >= 2023
        is_sharpe = trade_sharpe(pnls[is_mask]) if is_mask.any() else 0
        oos_sharpe = trade_sharpe(pnls[oos_mask]) if oos_mask.any() else 0

        return {
            "n": len(trades),
            "pnl": round(total, 2),
            "wr": round(wins / len(trades), 3),
            "sharpe": round(sharpe, 2),
            "cagr": round(cagr, 4),
            "max_dd": round(float(dd), 4),
            "is_sharpe": round(is_sharpe, 2),
            "oos_sharpe": round(oos_sharpe, 2),
        }
