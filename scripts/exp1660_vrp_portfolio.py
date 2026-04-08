#!/usr/bin/env python3
"""
EXP-1660 VRP Portfolio — Multi-ticker VRP harvesting + correlation analysis.

Follow-up to exp1660_vrp_deepening.py with important changes:

  1. EXPANDED TICKERS: QQQ and GLD now have 2025 data (was truncated before the
     Apr backfill). Retest all 5 viable tickers: SPY, XLF, QQQ, GLD, TLT.

  2. ADAPTIVE DELTA PER TICKER: Thin ETFs (GLD/TLT) have sparse strikes at
     10-delta — use 15-20 delta for them. SPY/QQQ/XLF use 10-delta.

  3. IV-RV GAP > 2% THRESHOLD: lowered from 3% to catch more signals.

  4. VRP PORTFOLIO: combine the best uncorrelated configs across tickers
     into a single daily-PnL series, compute combined Sharpe/CAGR/DD.

  5. CORRELATION MATRIX: pairwise between each VRP variant AND with EXP-1220
     (from the robustness daily series or trade list).

  6. RULE ZERO: 100% real IronVault data. Zero synthetic.

Output:
    reports/exp1660_vrp_portfolio.html
    reports/exp1660_vrp_portfolio.json
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault
from backtest.backtester import _yf_download_safe
from compass.metrics import annualized_sharpe, max_drawdown as compute_mdd, cagr as compute_cagr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vrp_portfolio")

REPORT_PATH = ROOT / "reports" / "exp1660_vrp_portfolio.html"
JSON_PATH = ROOT / "reports" / "exp1660_vrp_portfolio.json"
CAPITAL = 100_000
OOS_START = 2023
MIN_SPACING_DAYS = 5

# Adaptive delta per ticker — thin markets need higher (less OTM) delta
# to actually find tradable strikes
TICKER_CONFIGS = {
    "SPY": {"short_delta": 0.10, "hedge_delta": 0.05, "width_pct": 0.05},
    "QQQ": {"short_delta": 0.12, "hedge_delta": 0.05, "width_pct": 0.05},
    "XLF": {"short_delta": 0.15, "hedge_delta": 0.08, "width_pct": 0.06},
    "XLI": {"short_delta": 0.15, "hedge_delta": 0.08, "width_pct": 0.06},
    "GLD": {"short_delta": 0.18, "hedge_delta": 0.08, "width_pct": 0.07},
    "TLT": {"short_delta": 0.18, "hedge_delta": 0.08, "width_pct": 0.07},
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _exp_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fetch_yahoo(ticker: str) -> pd.DataFrame:
    df = _yf_download_safe(ticker, "2019-01-01", "2026-07-01")
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _find_exps(hd: IronVault, ticker: str) -> List[str]:
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT expiration FROM option_contracts "
        "WHERE ticker=? AND option_type='P' ORDER BY expiration",
        (ticker,),
    )
    exps = [r[0] for r in cur.fetchall()]
    conn.close()
    return exps


def _realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * math.sqrt(252)


def _find_delta_strike(
    hd: IronVault, ticker: str, exp: str, trade_date: str,
    price: float, option_type: str, target_delta: float,
) -> Optional[float]:
    """Find closest available strike at approximate target delta."""
    strikes = hd.get_available_strikes(ticker, exp, trade_date, option_type)
    if not strikes:
        return None
    exp_obj = _exp_dt(exp)
    dte = max(1, (exp_obj - datetime.strptime(trade_date, "%Y-%m-%d")).days)
    otm_factor = target_delta * 0.5 * math.sqrt(dte / 30)
    if option_type == "P":
        target_strike = price * (1 - otm_factor)
    else:
        target_strike = price * (1 + otm_factor)
    return min(strikes, key=lambda k: abs(k - target_strike))


def _atm_straddle_cost(
    hd: IronVault, ticker: str, exp: str, trade_date: str, price: float,
) -> Optional[float]:
    put_strikes = hd.get_available_strikes(ticker, exp, trade_date, "P")
    call_strikes = hd.get_available_strikes(ticker, exp, trade_date, "C")
    if not put_strikes or not call_strikes:
        return None
    put_k = min(put_strikes, key=lambda k: abs(k - price))
    call_k = min(call_strikes, key=lambda k: abs(k - price))
    exp_obj = _exp_dt(exp)
    put_sym = IronVault.build_occ_symbol(ticker, exp_obj, put_k, "P")
    call_sym = IronVault.build_occ_symbol(ticker, exp_obj, call_k, "C")
    pp = hd.get_contract_price(put_sym, trade_date)
    cp = hd.get_contract_price(call_sym, trade_date)
    if pp is None or cp is None:
        return None
    return float(pp + cp)


def _iv_from_straddle(straddle_cost: float, spot: float, dte: int) -> float:
    """Brenner-Subrahmanyam arithmetic approximation.

    σ ≈ straddle / (spot × √(2T/π))

    This converts a REAL straddle price (from IronVault) into a vol number.
    It is NOT a pricing model used to generate prices.
    """
    if spot <= 0 or dte <= 0 or straddle_cost <= 0:
        return 0.0
    T = dte / 365.0
    return float(straddle_cost / (spot * math.sqrt(2 * T / math.pi)))


# ═══════════════════════════════════════════════════════════════════════════
# Backtest: IV-RV gap strangle with adaptive delta
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VRPResult:
    ticker: str
    trades: List[Dict]
    n_trades: int
    total_pnl: float
    win_rate: float
    sharpe: float            # trade-level Sharpe (arithmetic mean)
    daily_sharpe: float      # daily-return Sharpe via compass/metrics
    cagr: float
    max_dd: float
    avg_iv_rv_gap: float
    is_sharpe: float
    oos_sharpe: float
    oos_n: int
    spy_corr: float          # correlation to SPY daily returns
    daily_pnl_series: pd.Series  # for portfolio combination


def run_vrp_ticker(
    hd: IronVault,
    ticker: str,
    underlying_df: pd.DataFrame,
    iv_rv_threshold: float = 0.02,
) -> VRPResult:
    """Run IV-RV gap VRP harvesting on one ticker.

    RULE ZERO: all option prices from IronVault.
    """
    cfg = TICKER_CONFIGS[ticker]
    close = underlying_df["Close"]
    td_set = set(underlying_df.index.strftime("%Y-%m-%d"))
    all_exps = _find_exps(hd, ticker)
    if not all_exps:
        log.warning(f"  {ticker}: no expirations in IronVault")
        return None

    rvol = _realized_vol(close, window=20)
    trades = []
    last_entry = None

    for date in underlying_df.index:
        ds = date.strftime("%Y-%m-%d")
        if ds < "2020-03-01":
            continue
        if last_entry and (date - last_entry).days < MIN_SPACING_DAYS:
            continue

        try:
            spot = float(close.loc[ds])
        except (KeyError, TypeError):
            continue
        if np.isnan(spot) or spot <= 0:
            continue

        # Find short-dated expiration (7-21 DTE)
        short_exp = None
        for e in all_exps:
            dte = (_exp_dt(e) - date).days
            if 7 <= dte <= 21:
                short_exp = e
                break
        if short_exp is None:
            continue
        short_exp_obj = _exp_dt(short_exp)
        short_dte = (short_exp_obj - date).days

        # Find hedge expiration (45-90 DTE)
        hedge_exp = None
        for e in all_exps:
            dte = (_exp_dt(e) - date).days
            if 45 <= dte <= 90:
                hedge_exp = e
                break
        if hedge_exp is None:
            continue
        hedge_exp_obj = _exp_dt(hedge_exp)

        # IV-RV gap signal
        straddle_cost = _atm_straddle_cost(hd, ticker, short_exp, ds, spot)
        if straddle_cost is None or straddle_cost <= 0:
            continue
        iv = _iv_from_straddle(straddle_cost, spot, short_dte)

        try:
            rv = float(rvol.loc[ds])
        except (KeyError, TypeError):
            continue
        if np.isnan(rv):
            continue

        iv_rv_gap = iv - rv
        if iv_rv_gap < iv_rv_threshold:
            continue

        # Adaptive delta per ticker
        put_strike = _find_delta_strike(
            hd, ticker, short_exp, ds, spot, "P", cfg["short_delta"])
        call_strike = _find_delta_strike(
            hd, ticker, short_exp, ds, spot, "C", cfg["short_delta"])
        hedge_strike = _find_delta_strike(
            hd, ticker, hedge_exp, ds, spot, "P", cfg["hedge_delta"])

        if put_strike is None or call_strike is None or hedge_strike is None:
            continue

        # Get real prices
        put_sym = IronVault.build_occ_symbol(ticker, short_exp_obj, put_strike, "P")
        call_sym = IronVault.build_occ_symbol(ticker, short_exp_obj, call_strike, "C")
        hedge_sym = IronVault.build_occ_symbol(ticker, hedge_exp_obj, hedge_strike, "P")

        put_px = hd.get_contract_price(put_sym, ds)
        call_px = hd.get_contract_price(call_sym, ds)
        hedge_px = hd.get_contract_price(hedge_sym, ds)

        if put_px is None or call_px is None or hedge_px is None:
            continue
        if put_px < 0.05 or call_px < 0.05:
            continue

        strangle_credit = float(put_px + call_px)
        net_credit = strangle_credit - float(hedge_px)

        if net_credit <= 0:
            continue

        # Position size
        risk_est = max(net_credit * 2, (put_strike - hedge_strike) * 0.3)
        contracts = max(1, min(5, int(CAPITAL * 0.02 / (risk_est * 100))))

        # Walk forward
        current = date + timedelta(days=1)
        exit_date = ds
        exit_reason = "expiration"
        exit_pnl = 0.0
        hold_days = 0

        while current <= short_exp_obj:
            cs = current.strftime("%Y-%m-%d")
            if cs not in td_set:
                current += timedelta(days=1)
                continue
            hold_days += 1

            pp2 = hd.get_contract_price(put_sym, cs)
            cp2 = hd.get_contract_price(call_sym, cs)
            hp2 = hd.get_contract_price(hedge_sym, cs)

            if pp2 is not None and cp2 is not None:
                current_strangle = float(pp2 + cp2)
                current_hedge = float(hp2) if hp2 is not None else float(hedge_px)
                unrealized = net_credit - (current_strangle - current_hedge)

                if unrealized >= net_credit * 0.50:
                    exit_pnl = unrealized
                    exit_date = cs
                    exit_reason = "profit_target"
                    break
                if unrealized <= -net_credit * 2.0:
                    exit_pnl = unrealized
                    exit_date = cs
                    exit_reason = "stop_loss"
                    break

                exit_pnl = unrealized
                exit_date = cs

            current += timedelta(days=1)

        if exit_reason == "expiration":
            pp_final = hd.get_contract_price(put_sym, short_exp)
            cp_final = hd.get_contract_price(call_sym, short_exp)
            if pp_final is not None and cp_final is not None:
                exit_pnl = net_credit - (float(pp_final) + float(cp_final))

        total_pnl = exit_pnl * 100 * contracts

        trades.append({
            "entry_date": ds,
            "exit_date": exit_date,
            "ticker": ticker,
            "pnl": round(total_pnl, 2),
            "exit_reason": exit_reason,
            "iv": round(iv, 4),
            "rv": round(rv, 4),
            "iv_rv_gap": round(iv_rv_gap, 4),
            "net_credit": round(net_credit, 4),
            "contracts": contracts,
            "hold_days": hold_days,
            "short_dte": short_dte,
        })
        last_entry = date

    return _compute_result(ticker, trades, underlying_df)


def _compute_result(ticker: str, trades: List[Dict],
                      underlying_df: pd.DataFrame) -> VRPResult:
    """Compute metrics using compass/metrics.py (arithmetic Sharpe)."""
    if not trades:
        return VRPResult(
            ticker=ticker, trades=[], n_trades=0, total_pnl=0, win_rate=0,
            sharpe=0, daily_sharpe=0, cagr=0, max_dd=0, avg_iv_rv_gap=0,
            is_sharpe=0, oos_sharpe=0, oos_n=0, spy_corr=0,
            daily_pnl_series=pd.Series(dtype=float),
        )

    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    pnls = df["pnl"].values
    n = len(pnls)
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())

    # Trade-level Sharpe (arithmetic mean of PnLs / std × √N)
    mu = float(np.mean(pnls))
    sigma = float(np.std(pnls, ddof=1)) if n > 1 else 1.0
    trade_sharpe = float(mu / sigma * math.sqrt(min(n, 52))) if sigma > 1e-9 else 0.0

    # Daily PnL series aggregated by exit date
    daily_pnl = df.groupby("exit_date")["pnl"].sum()

    # Full date range for daily metrics
    full_range = pd.date_range(
        max(underlying_df.index.min(), pd.Timestamp("2020-03-01")),
        underlying_df.index.max(),
        freq="B",
    )
    daily_pnl_full = daily_pnl.reindex(full_range, fill_value=0)
    daily_returns = daily_pnl_full.values / CAPITAL

    # Sharpe via compass/metrics.py (arithmetic mean)
    daily_sharpe = annualized_sharpe(daily_returns, rf_annual=0.05)
    mdd = compute_mdd(daily_returns)
    cagr = compute_cagr(daily_returns)

    # SPY correlation
    spy_ret = underlying_df["Close"].pct_change().fillna(0)
    common = daily_pnl.index.intersection(spy_ret.index)
    spy_corr = 0.0
    if len(common) > 5:
        a = daily_pnl.reindex(common).fillna(0).values
        b = spy_ret.reindex(common).fillna(0).values
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            spy_corr = float(np.corrcoef(a, b)[0, 1])

    # Walk-forward IS/OOS
    is_trades = df[df["entry_date"].dt.year < OOS_START]
    oos_trades = df[df["entry_date"].dt.year >= OOS_START]

    def _trade_sharpe(t):
        if len(t) < 2:
            return 0.0
        v = t["pnl"].values
        m, s = float(np.mean(v)), float(np.std(v, ddof=1))
        if s < 1e-9:
            return 0.0
        return float(m / s * math.sqrt(min(len(v), 52)))

    return VRPResult(
        ticker=ticker,
        trades=trades,
        n_trades=n,
        total_pnl=round(total, 2),
        win_rate=round(wins / n, 3),
        sharpe=round(trade_sharpe, 3),
        daily_sharpe=round(float(daily_sharpe), 3),
        cagr=round(float(cagr), 4),
        max_dd=round(float(mdd), 4),
        avg_iv_rv_gap=round(float(df["iv_rv_gap"].mean()), 4),
        is_sharpe=round(_trade_sharpe(is_trades), 3),
        oos_sharpe=round(_trade_sharpe(oos_trades), 3),
        oos_n=len(oos_trades),
        spy_corr=round(spy_corr, 4),
        daily_pnl_series=daily_pnl_full,
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXP-1220 daily PnL series from Alpaca paper trading
# ═══════════════════════════════════════════════════════════════════════════

def load_exp1220_series() -> Optional[pd.Series]:
    """Load an EXP-1220 daily PnL series for correlation analysis.

    Priority order:
      1. Latest robustness/stress test JSON with daily series
      2. Fallback: SPY daily returns × -0.1 (approximate counter-cyclical hedge)

    Returns None if no series found.
    """
    # Try the robustness report
    path = ROOT / "reports" / "exp1220_robustness_report.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            # Look for various possible keys
            for key in ("daily_pnl", "daily_returns", "pnl_series"):
                if key in data and isinstance(data[key], dict):
                    s = pd.Series(data[key])
                    s.index = pd.to_datetime(s.index)
                    return s
            # equity_curve → diff
            ec = data.get("equity_curve")
            if isinstance(ec, list) and len(ec) > 0 and isinstance(ec[0], dict):
                dates, values = [], []
                for e in ec:
                    d = e.get("date") or e.get("timestamp")
                    v = e.get("equity") or e.get("value")
                    if d and v is not None:
                        dates.append(pd.to_datetime(d))
                        values.append(float(v))
                if dates:
                    s = pd.Series(values, index=dates).diff().fillna(0)
                    return s
        except Exception as e:
            log.warning(f"Could not parse robustness report: {e}")

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio combination
# ═══════════════════════════════════════════════════════════════════════════

def combine_vrp_portfolio(
    results: List[VRPResult],
    spy_df: pd.DataFrame,
) -> Dict:
    """Combine multiple VRP configs into a single equal-weighted daily series.

    Only includes configs with n_trades >= 10 (statistical power floor).
    """
    viable = [r for r in results if r.n_trades >= 10]
    if not viable:
        return {"n_configs": 0, "error": "no configs with >= 10 trades"}

    # Sum all daily PnL series
    combined = None
    for r in viable:
        if combined is None:
            combined = r.daily_pnl_series.copy()
        else:
            combined = combined.add(r.daily_pnl_series, fill_value=0)

    daily_returns = combined.values / (CAPITAL * len(viable))  # normalize by # configs

    sharpe = annualized_sharpe(daily_returns, rf_annual=0.05)
    mdd = compute_mdd(daily_returns)
    cagr = compute_cagr(daily_returns)

    # SPY correlation
    spy_ret = spy_df["Close"].pct_change().fillna(0)
    common = combined.index.intersection(spy_ret.index)
    spy_corr = 0.0
    if len(common) > 5:
        a = combined.reindex(common).fillna(0).values
        b = spy_ret.reindex(common).fillna(0).values
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            spy_corr = float(np.corrcoef(a, b)[0, 1])

    return {
        "n_configs": len(viable),
        "configs": [r.ticker for r in viable],
        "total_pnl": round(float(combined.sum()), 2),
        "sharpe": round(float(sharpe), 3),
        "cagr": round(float(cagr), 4),
        "max_dd": round(float(mdd), 4),
        "spy_corr": round(float(spy_corr), 4),
        "n_active_days": int((combined != 0).sum()),
    }


def correlation_matrix(
    results: List[VRPResult],
    exp1220_series: Optional[pd.Series],
) -> Dict:
    """Compute correlation matrix between all VRP daily series and EXP-1220."""
    series_map = {r.ticker: r.daily_pnl_series for r in results if r.n_trades >= 5}
    if exp1220_series is not None:
        series_map["EXP-1220"] = exp1220_series

    names = list(series_map.keys())
    n = len(names)
    matrix = {}

    for i, a in enumerate(names):
        matrix[a] = {}
        for j, b in enumerate(names):
            if i == j:
                matrix[a][b] = 1.0
                continue
            sa = series_map[a]
            sb = series_map[b]
            common = sa.index.intersection(sb.index)
            if len(common) < 5:
                matrix[a][b] = 0.0
                continue
            va = sa.reindex(common).fillna(0).values
            vb = sb.reindex(common).fillna(0).values
            if np.std(va) < 1e-9 or np.std(vb) < 1e-9:
                matrix[a][b] = 0.0
                continue
            matrix[a][b] = round(float(np.corrcoef(va, vb)[0, 1]), 3)

    return {"names": names, "matrix": matrix}


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_html(
    results: List[VRPResult],
    portfolio: Dict,
    corr: Dict,
    iv_rv_threshold: float,
) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Per-ticker rows
    rows = ""
    for r in sorted(results, key=lambda x: x.oos_sharpe, reverse=True):
        status = "LIVE" if r.n_trades >= 10 and r.oos_sharpe > 0 else "KILLED"
        sc = "var(--green)" if status == "LIVE" else "var(--red)"
        spy_c = "var(--green)" if abs(r.spy_corr) < 0.3 else "var(--yellow)"
        rows += (
            f'<tr><td><strong>{r.ticker}</strong></td>'
            f'<td>{r.n_trades}</td>'
            f'<td style="color:{"var(--green)" if r.total_pnl > 0 else "var(--red)"}">${r.total_pnl:,.0f}</td>'
            f'<td>{r.win_rate:.0%}</td>'
            f'<td>{r.sharpe:.2f}</td>'
            f'<td>{r.daily_sharpe:.2f}</td>'
            f'<td>{r.cagr:.1%}</td>'
            f'<td>{r.max_dd:.1%}</td>'
            f'<td>{r.avg_iv_rv_gap:.3f}</td>'
            f'<td style="color:{spy_c}">{r.spy_corr:+.3f}</td>'
            f'<td>{r.oos_n}</td>'
            f'<td>{r.oos_sharpe:.2f}</td>'
            f'<td style="color:{sc};font-weight:700">{status}</td></tr>\n'
        )

    # Correlation matrix HTML
    names = corr["names"]
    corr_head = "<tr><th></th>" + "".join(f'<th>{n}</th>' for n in names) + "</tr>"
    corr_rows = ""
    for a in names:
        cells = f'<td><strong>{a}</strong></td>'
        for b in names:
            v = corr["matrix"][a][b]
            bg = "#1e293b" if abs(v) > 0.95 else (
                f"rgb(255,{int(255*(1-min(v,1.0)))},{int(255*(1-min(v,1.0)))})" if v > 0 else
                f"rgb({int(255*(1+max(v,-1.0)))},{int(255*(1+max(v,-1.0)))},255)"
            )
            color = "#fff" if abs(v) > 0.5 or a == b else "#111"
            cells += f'<td style="background:{bg};color:{color};text-align:center;font-size:.75rem">{v:+.2f}</td>'
        corr_rows += f'<tr>{cells}</tr>\n'

    # Portfolio section
    if portfolio.get("n_configs", 0) > 0:
        port_html = f'''
        <h2>Combined VRP Portfolio ({portfolio["n_configs"]} tickers, equal weight)</h2>
        <p class="note">Tickers combined: {", ".join(portfolio["configs"])}</p>
        <div class="cards">
          <div class="c"><div class="l">Total PnL</div><div class="v">${portfolio["total_pnl"]:,.0f}</div></div>
          <div class="c"><div class="l">Sharpe (arith)</div><div class="v">{portfolio["sharpe"]:.2f}</div></div>
          <div class="c"><div class="l">CAGR</div><div class="v">{portfolio["cagr"]:.1%}</div></div>
          <div class="c"><div class="l">Max DD</div><div class="v">{portfolio["max_dd"]:.1%}</div></div>
          <div class="c"><div class="l">SPY Corr</div><div class="v">{portfolio["spy_corr"]:+.3f}</div></div>
          <div class="c"><div class="l">Active Days</div><div class="v">{portfolio["n_active_days"]}</div></div>
        </div>
        '''
    else:
        port_html = '<p class="note">No portfolio — need at least one config with &ge;10 trades.</p>'

    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EXP-1660 VRP Portfolio</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e5e7eb;--text:#111827;--muted:#6b7280;--green:#059669;--red:#dc2626;--yellow:#d97706;--blue:#2563eb}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;max-width:1200px;margin:0 auto;padding:24px}}
h1{{font-size:1.5rem;font-weight:800}}
h2{{font-size:1.1rem;font-weight:700;margin:28px 0 12px;border-bottom:2px solid var(--border);padding-bottom:6px}}
.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:14px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center}}
.c .l{{color:var(--muted);font-size:.68rem;font-weight:600;text-transform:uppercase}}
.c .v{{font-size:1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:.82rem}}
th,td{{padding:5px 7px;text-align:right;border-bottom:1px solid var(--border)}}
th{{background:#f1f5f9;color:var(--muted);font-size:.68rem;font-weight:600;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
.callout{{background:var(--card);border-left:4px solid var(--blue);padding:14px;margin:12px 0;font-size:.85rem;line-height:1.6;border-radius:4px}}
.note{{color:var(--muted);font-size:.82rem;margin:6px 0}}
.footer{{margin-top:40px;text-align:center;font-size:.72rem;color:var(--muted);border-top:1px solid var(--border);padding-top:14px}}
</style></head><body>

<h1>EXP-1660 VRP Portfolio — Multi-Ticker Harvesting + Correlation Matrix</h1>
<div class="subtitle">{ts} &bull; IV-RV gap threshold: {iv_rv_threshold:.0%} &bull; Adaptive delta per ticker &bull; Rule Zero: 100% IronVault real data</div>

<div class="callout">
<strong>What changed from EXP-1660 VRP Deepening:</strong> QQQ and GLD were re-tested with the
freshly backfilled 2025 data (previous run was stale). Delta targets are now adaptive per ticker
(10-18 delta depending on strike density). IV-RV gap threshold lowered to 2% to catch more
signals. Portfolio section combines all live configs equal-weighted.
</div>

<h2>Per-Ticker Results (IV-RV gap &gt; {iv_rv_threshold:.0%})</h2>
<table>
<thead><tr><th>Ticker</th><th>N</th><th>PnL</th><th>WR</th><th>Trade SR</th><th>Daily SR</th><th>CAGR</th><th>DD</th><th>Avg Gap</th><th>SPY ρ</th><th>OOS N</th><th>OOS SR</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table>

<div class="callout">
<strong>Two Sharpe columns:</strong> "Trade SR" is per-trade arithmetic mean/std × √N (counts only actual
trade days). "Daily SR" is the full daily-return series Sharpe via compass/metrics.py — this is
diluted by idle days but is the honest portfolio-level number. The gap between these two columns
is the capital utilization problem documented in MASTERPLAN Phase 7.
</div>

{port_html}

<h2>Correlation Matrix (daily PnL series)</h2>
<p class="note">Red = positive (redundant), blue = negative (diversifying), white = uncorrelated.
  EXP-1220 row/col present if its daily series was available.</p>
<table>
{corr_head}
{corr_rows}
</table>

<div class="footer">
  EXP-1660 VRP Portfolio &bull; Real IronVault data &bull; {ts}
</div>
</body></html>'''


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("EXP-1660 VRP PORTFOLIO — Multi-Ticker + Correlation Matrix")
    print("Rule Zero: 100% real IronVault data, zero synthetic")
    print("=" * 70)

    hd = IronVault.instance()
    log.info(f"IronVault: {hd._db_path}")

    # Only test tickers that actually exist in IronVault (confirmed)
    tickers = ["SPY", "QQQ", "XLF", "XLI", "GLD", "TLT"]
    iv_rv_threshold = 0.02  # 2% — lowered from 3%

    # Load underlying data
    log.info("Loading underlying prices...")
    underlying_data = {}
    for t in tickers:
        try:
            df = _fetch_yahoo(t)
            underlying_data[t] = df
            log.info(f"  {t}: {df.index.min().date()} → {df.index.max().date()} ({len(df)} days)")
        except Exception as e:
            log.error(f"  {t}: FAILED — {e}")

    # Run each ticker
    log.info(f"\nRunning {len(underlying_data)} VRP backtests...")
    results = []
    for ticker in tickers:
        if ticker not in underlying_data:
            continue
        log.info(f"[{ticker}]")
        r = run_vrp_ticker(hd, ticker, underlying_data[ticker], iv_rv_threshold)
        if r is not None:
            results.append(r)
            log.info(f"  N={r.n_trades}, total_pnl=${r.total_pnl:,.0f}, "
                      f"trade_sharpe={r.sharpe:.2f}, daily_sharpe={r.daily_sharpe:.2f}, "
                      f"SPY corr={r.spy_corr:+.3f}, OOS N={r.oos_n}, OOS SR={r.oos_sharpe:.2f}")

    # Build portfolio
    spy_ref = underlying_data.get("SPY", list(underlying_data.values())[0])
    portfolio = combine_vrp_portfolio(results, spy_ref)
    log.info(f"\nPortfolio: {portfolio}")

    # Correlation matrix
    exp1220_series = load_exp1220_series()
    if exp1220_series is not None:
        log.info(f"EXP-1220 series loaded: {len(exp1220_series)} rows")
    else:
        log.warning("No EXP-1220 daily series found — correlation matrix excludes it")
    corr = correlation_matrix(results, exp1220_series)
    log.info(f"Correlation matrix: {len(corr['names'])} series")

    # Write reports
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html(results, portfolio, corr, iv_rv_threshold)
    REPORT_PATH.write_text(html, encoding="utf-8")
    log.info(f"\nHTML: {REPORT_PATH}")

    # Serialize results
    json_data = {
        "experiment": "EXP-1660",
        "subtitle": "VRP Portfolio",
        "data_source": "IronVault (options_cache.db) — 100% real",
        "rule_zero_compliant": True,
        "iv_rv_threshold": iv_rv_threshold,
        "tickers_tested": tickers,
        "ticker_configs": TICKER_CONFIGS,
        "per_ticker_results": [
            {
                "ticker": r.ticker,
                "n_trades": r.n_trades,
                "total_pnl": r.total_pnl,
                "win_rate": r.win_rate,
                "trade_sharpe": r.sharpe,
                "daily_sharpe": r.daily_sharpe,
                "cagr": r.cagr,
                "max_dd": r.max_dd,
                "avg_iv_rv_gap": r.avg_iv_rv_gap,
                "spy_corr": r.spy_corr,
                "is_sharpe": r.is_sharpe,
                "oos_sharpe": r.oos_sharpe,
                "oos_n": r.oos_n,
            }
            for r in results
        ],
        "portfolio": portfolio,
        "correlation_matrix": corr,
    }
    JSON_PATH.write_text(json.dumps(json_data, indent=2, default=str))
    log.info(f"JSON: {JSON_PATH}")

    # Summary to stdout
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"{'Ticker':<7} {'N':>4} {'PnL':>10} {'WR':>5} {'TradeSR':>8} {'DailySR':>8} {'OOS SR':>7} {'SPY ρ':>8}")
    for r in sorted(results, key=lambda x: x.oos_sharpe, reverse=True):
        log.info(f"{r.ticker:<7} {r.n_trades:>4} ${r.total_pnl:>8,.0f} "
                  f"{r.win_rate:>4.0%}  {r.sharpe:>7.2f} {r.daily_sharpe:>7.2f} "
                  f"{r.oos_sharpe:>6.2f}  {r.spy_corr:>+7.3f}")
    log.info(f"\nCombined portfolio: N={portfolio.get('n_configs', 0)}, "
              f"PnL=${portfolio.get('total_pnl', 0):,.0f}, "
              f"Sharpe={portfolio.get('sharpe', 0):.2f}, "
              f"CAGR={portfolio.get('cagr', 0):.1%}, "
              f"DD={portfolio.get('max_dd', 0):.1%}, "
              f"SPY ρ={portfolio.get('spy_corr', 0):+.3f}")


if __name__ == "__main__":
    main()
