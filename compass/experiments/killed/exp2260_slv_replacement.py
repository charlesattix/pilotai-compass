"""
EXP-2260 — SLV Replacement (capacity solution).

The 5-stream static portfolio (EXP-2080) holds a 20% SLV-calendar sleeve
that saturates around $82M of AUM. For production deployment at book
sizes >$500M that SLV sleeve is the binding constraint. This experiment
tests four candidate replacements on REAL IronVault / Yahoo data and
reports their Sharpe, CAGR, max DD, correlation to EXP-1220, and a
capacity heuristic.

Candidates:

  1. TLT put credit spreads       (real IronVault options)
  2. GDX put credit spreads       (real IronVault options)  ← BLOCKED
  3. Larger GLD allocation        (reweight EXP-1770 GLD calendar stream)
  4. SPY weekly UNHEDGED short straddle  (real IronVault options)

The SPY delta-HEDGED weekly short straddle was already tested in EXP-2160
and came in as a Sharpe-0.16 null result (delta hedging removes the
directional component so theta ≈ gamma on average). The "unhedged"
variant is a different animal — a pure theta-collection / short-vol trade
that runs naked directional risk. It is included here because Carlos
explicitly named it as a high-capacity candidate.

REAL DATA — Rule Zero:
  * Option chains from `data/options_cache.db` (IronVault) via
    option_contracts JOIN option_daily on contract_symbol.
  * Spot from Yahoo.
  * BS inversion (Brent) for strike-delta selection only.
  * GLD calendar stream from compass.exp1770_commodity_calendars
    (real Yahoo GLD − GC=F spread returns, walk-forward).
  * EXP-1220 yearly returns from experiments/EXP-1220-real/results.

Outputs:
  compass/exp2260_slv_replacement.py            (this file)
  compass/reports/exp2260_slv_replacement.json
  compass/reports/exp2260_slv_replacement.html

Tag: EXP-2260
Run: python3 -m compass.exp2260_slv_replacement
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
REPORT_JSON = REPORT_DIR / "exp2260_slv_replacement.json"
REPORT_HTML = REPORT_DIR / "exp2260_slv_replacement.html"
DB_PATH = ROOT / "data" / "options_cache.db"
EXP1220_SUMMARY = ROOT / "experiments" / "EXP-1220-real" / "results" / "summary.json"

from compass.exp1960_skew_alpha import (
    implied_vol_put, bs_put_delta, fetch_contract_close,
)
from compass.exp2160_high_capacity_alts import (
    fetch_yahoo_close, list_snapshot_dates, pick_expiration,
    fetch_chain, coverage_stats, implied_vol_call, bs_call_delta,
)

START = "2020-01-01"
END = "2025-12-31"
CAPITAL = 100_000.0
RISK_FREE = 0.045

# Credit spread config (TLT / GDX)
CS_SHORT_DELTA = -0.30
CS_LONG_DELTA = -0.15
CS_TARGET_DTE = 30
CS_RISK_PER_TRADE = 0.02
CS_SLIPPAGE = 10.0  # $/spread round-trip — applied based on EXP-2210 findings

# SPY unhedged weekly short straddle
ST_TARGET_DTE = 7
ST_RISK_PER_TRADE = 0.01
ST_SLIPPAGE = 4.0   # $/contract round-trip (SPY is liquid)

# Capacity heuristic
PARTICIPATION_RATE = 0.10        # cap at 10% of daily volume


# ── Credit spread backtest (TLT / GDX) ─────────────────────────────────


@dataclass
class CSTrade:
    ticker: str
    entry_date: str
    expiration: str
    short_strike: float
    long_strike: float
    short_symbol: str
    long_symbol: str
    net_credit: float
    exit_net: float
    pnl_per_spread: float       # after slippage, per contract (×100)
    pnl_pct_capital: float
    short_delta: float
    win: bool


def backtest_credit_spread(con: sqlite3.Connection, ticker: str
                           ) -> Tuple[List[CSTrade], Dict[str, int]]:
    spot = fetch_yahoo_close(ticker)
    all_dates = list_snapshot_dates(con, ticker)
    all_dates = [d for d in all_dates if START <= d <= END]

    # Weekly cadence
    by_week: Dict[Tuple[int, int], str] = {}
    for s in all_dates:
        wk = datetime.strptime(s, "%Y-%m-%d").isocalendar()[:2]
        by_week.setdefault(wk, s)
    weekly_snaps = sorted(by_week.values())

    trades: List[CSTrade] = []
    diag = {"n_attempted": 0, "n_dropped_no_exp": 0,
            "n_dropped_thin_chain": 0, "n_real_exits": 0,
            "n_intrinsic_exits": 0}

    for snap in weekly_snaps:
        diag["n_attempted"] += 1
        try:
            spot_val = float(spot.loc[:snap].iloc[-1])
        except (KeyError, IndexError):
            continue
        expiration = pick_expiration(
            con, ticker, snap, CS_TARGET_DTE, "P", min_dte=7,
        )
        if expiration is None:
            diag["n_dropped_no_exp"] += 1
            continue
        chain = fetch_chain(con, ticker, snap, expiration, "P")
        if len(chain) < 5:
            diag["n_dropped_thin_chain"] += 1
            continue

        snap_dt = datetime.strptime(snap, "%Y-%m-%d")
        exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
        T = (exp_dt - snap_dt).days / 365.0
        if T <= 0:
            continue

        table: List[Tuple[float, float, str, float]] = []
        for K, px, sym in chain:
            sigma = implied_vol_put(px, spot_val, K, T, RISK_FREE)
            if sigma is None or sigma <= 0:
                continue
            delta = bs_put_delta(spot_val, K, T, sigma, RISK_FREE)
            table.append((K, px, sym, delta))
        if len(table) < 4:
            diag["n_dropped_thin_chain"] += 1
            continue

        short_row = min(table, key=lambda r: abs(r[3] - CS_SHORT_DELTA))
        long_row = min(table, key=lambda r: abs(r[3] - CS_LONG_DELTA))
        if short_row[0] <= long_row[0]:
            continue
        short_K, short_px, short_sym, short_d = short_row
        long_K, long_px, long_sym, _ = long_row
        net_credit = short_px - long_px
        if net_credit <= 0:
            continue

        exit_target = expiration
        try:
            spot_exit = float(spot.loc[:exit_target].iloc[-1])
        except (KeyError, IndexError):
            spot_exit = spot_val
        short_exit_info = fetch_contract_close(con, short_sym, snap, exit_target)
        long_exit_info = fetch_contract_close(con, long_sym, snap, exit_target)
        real_short = short_exit_info is not None and short_exit_info[0] != snap
        real_long = long_exit_info is not None and long_exit_info[0] != snap
        if real_short:
            short_exit = float(short_exit_info[1])
        else:
            short_exit = max(short_K - spot_exit, 0.0)
        if real_long:
            long_exit = float(long_exit_info[1])
        else:
            long_exit = max(long_K - spot_exit, 0.0)
        if real_short and real_long:
            diag["n_real_exits"] += 1
        else:
            diag["n_intrinsic_exits"] += 1

        exit_net = short_exit - long_exit
        pnl_gross = (net_credit - exit_net) * 100.0
        pnl_net = pnl_gross - CS_SLIPPAGE

        max_loss = max((short_K - long_K) - net_credit, 0.01) * 100.0
        n_contracts = (CS_RISK_PER_TRADE * CAPITAL) / max_loss
        pnl_pct = (pnl_net * n_contracts) / CAPITAL

        trades.append(CSTrade(
            ticker=ticker,
            entry_date=snap,
            expiration=expiration,
            short_strike=float(short_K),
            long_strike=float(long_K),
            short_symbol=short_sym,
            long_symbol=long_sym,
            net_credit=float(net_credit),
            exit_net=float(exit_net),
            pnl_per_spread=float(pnl_net),
            pnl_pct_capital=float(pnl_pct),
            short_delta=float(short_d),
            win=bool(pnl_net > 0),
        ))
    return trades, diag


# ── SPY unhedged weekly short straddle ────────────────────────────────


@dataclass
class StraddleTrade:
    entry_date: str
    expiration: str
    strike: float
    call_entry: float
    put_entry: float
    call_exit: float
    put_exit: float
    premium_collected: float
    pnl_per_contract: float     # after slippage
    pnl_pct_capital: float
    spot_entry: float
    spot_exit: float
    win: bool


def backtest_spy_unhedged_straddle(con: sqlite3.Connection
                                   ) -> Tuple[List[StraddleTrade], Dict[str, int]]:
    spot = fetch_yahoo_close("SPY")
    all_dates = list_snapshot_dates(con, "SPY")
    all_dates = [d for d in all_dates if START <= d <= END]

    by_week: Dict[Tuple[int, int], str] = {}
    for s in all_dates:
        wk = datetime.strptime(s, "%Y-%m-%d").isocalendar()[:2]
        by_week.setdefault(wk, s)
    weekly_snaps = sorted(by_week.values())

    trades: List[StraddleTrade] = []
    diag = {"n_attempted": 0, "n_dropped_no_exp": 0, "n_dropped_thin": 0}

    for snap in weekly_snaps:
        diag["n_attempted"] += 1
        try:
            spot_val = float(spot.loc[:snap].iloc[-1])
        except (KeyError, IndexError):
            continue
        expiration = pick_expiration(con, "SPY", snap, ST_TARGET_DTE, "P", min_dte=3)
        if expiration is None:
            diag["n_dropped_no_exp"] += 1
            continue
        put_chain = fetch_chain(con, "SPY", snap, expiration, "P")
        call_chain = fetch_chain(con, "SPY", snap, expiration, "C")
        if not put_chain or not call_chain:
            diag["n_dropped_thin"] += 1
            continue
        put_strikes = {K for K, *_ in put_chain}
        call_strikes = {K for K, *_ in call_chain}
        common = sorted(put_strikes & call_strikes)
        if not common:
            diag["n_dropped_thin"] += 1
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

        exit_target = expiration
        try:
            spot_exit = float(spot.loc[:exit_target].iloc[-1])
        except (KeyError, IndexError):
            spot_exit = spot_val

        put_exit_info = fetch_contract_close(con, put_sym, snap, exit_target)
        call_exit_info = fetch_contract_close(con, call_sym, snap, exit_target)
        if put_exit_info is not None and put_exit_info[0] != snap:
            put_exit = float(put_exit_info[1])
        else:
            put_exit = max(atm_strike - spot_exit, 0.0)
        if call_exit_info is not None and call_exit_info[0] != snap:
            call_exit = float(call_exit_info[1])
        else:
            call_exit = max(spot_exit - atm_strike, 0.0)

        # Short straddle P&L: collect premium, buy back both legs
        pnl_per_share = (put_entry - put_exit) + (call_entry - call_exit)
        pnl_per_contract = pnl_per_share * 100.0 - ST_SLIPPAGE

        # Size: risk ST_RISK_PER_TRADE% of capital against a ~3σ move
        # (premium × 3 ≈ loss on a 3σ move on a 1-week ATM straddle)
        stress_loss = max(premium * 3.0, 0.01) * 100.0
        n_contracts = (ST_RISK_PER_TRADE * CAPITAL) / stress_loss
        pnl_pct = pnl_per_contract * n_contracts / CAPITAL

        trades.append(StraddleTrade(
            entry_date=snap,
            expiration=expiration,
            strike=float(atm_strike),
            call_entry=float(call_entry),
            put_entry=float(put_entry),
            call_exit=float(call_exit),
            put_exit=float(put_exit),
            premium_collected=float(premium),
            pnl_per_contract=float(pnl_per_contract),
            pnl_pct_capital=float(pnl_pct),
            spot_entry=float(spot_val),
            spot_exit=float(spot_exit),
            win=bool(pnl_per_contract > 0),
        ))
    return trades, diag


# ── GLD calendar reweight ─────────────────────────────────────────────


def gld_calendar_daily() -> pd.Series:
    from compass.exp1770_commodity_calendars import PAIRS, load_pair, walk_forward
    etf, fut, _ = PAIRS["GLD"]
    bt = walk_forward("GLD", load_pair(etf, fut))
    s = bt.daily_returns.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s.reindex(pd.date_range(START, END, freq="B")).fillna(0.0)
    return s


# ── Capacity heuristic ─────────────────────────────────────────────────


def capacity_cs(con: sqlite3.Connection, trades: List[CSTrade]) -> Dict:
    if not trades:
        return {"n": 0}
    vols: List[int] = []
    for t in trades:
        row = con.execute("""
            SELECT d.volume FROM option_contracts c
            JOIN option_daily d ON c.contract_symbol = d.contract_symbol
            WHERE c.ticker=? AND c.expiration=? AND c.strike=? AND c.option_type='P'
              AND d.date=?
            LIMIT 1
        """, (t.ticker, t.expiration, t.short_strike, t.entry_date)).fetchone()
        if row and row[0]:
            vols.append(int(row[0]))
    if not vols:
        return {"n": 0}
    v = np.array(vols, dtype=float)
    # Dollar-vega capacity per month: 10% of median daily volume × weeks-in-month
    dollar_per_contract = 100.0  # rough notional per vega-pt ATM credit spread
    median_cap = float(np.median(v) * PARTICIPATION_RATE)
    monthly_cap = float(median_cap * 4)   # 4 weekly entries/month
    return {
        "n_observations": int(len(vols)),
        "participation_rate": PARTICIPATION_RATE,
        "median_daily_volume": float(np.median(v)),
        "p5_daily_volume": float(np.percentile(v, 5)),
        "median_max_contracts_per_entry": int(median_cap),
        "monthly_cap_contracts": int(monthly_cap),
        "monthly_dollar_vega": float(monthly_cap * dollar_per_contract),
    }


def capacity_spy_straddle(con: sqlite3.Connection,
                          trades: List[StraddleTrade]) -> Dict:
    if not trades:
        return {"n": 0}
    vols: List[int] = []
    for t in trades:
        row = con.execute("""
            SELECT MIN(d.volume) FROM option_contracts c
            JOIN option_daily d ON c.contract_symbol = d.contract_symbol
            WHERE c.ticker='SPY' AND c.expiration=? AND c.strike=?
              AND d.date=?
        """, (t.expiration, t.strike, t.entry_date)).fetchone()
        if row and row[0]:
            vols.append(int(row[0]))
    if not vols:
        return {"n": 0}
    v = np.array(vols, dtype=float)
    dollar_per_contract = 200.0  # ATM SPY straddle ~$2/pt vega per 1% move
    median_cap = float(np.median(v) * PARTICIPATION_RATE)
    monthly_cap = float(median_cap * 4)
    return {
        "n_observations": int(len(vols)),
        "participation_rate": PARTICIPATION_RATE,
        "median_daily_volume": float(np.median(v)),
        "p5_daily_volume": float(np.percentile(v, 5)),
        "median_max_contracts_per_entry": int(median_cap),
        "monthly_cap_contracts": int(monthly_cap),
        "monthly_dollar_notional": float(monthly_cap * dollar_per_contract),
    }


def capacity_gld_calendar_doubled() -> Dict:
    # GLD ETF 20d median ~ $1.5B / day, GC=F futures ~ $50B notional / day.
    # At 10% participation: $150M/day ETF + ~$5B/day futures = effective
    # binding leg is GLD ETF.
    return {
        "note": (
            "GLD ETF 20d median ADV ≈ $1.5B/day, GC=F futures notional "
            "≈ $50B/day. Binding leg is the GLD ETF at 10% participation "
            "= $150M/day of ETF flow, ≈ $3B/month total capacity. "
            "Doubling the existing 20% SLV allocation into GLD (to 40%) "
            "is well inside that ceiling. Source: public Yahoo daily "
            "volumes × median 2024 close, rule-of-thumb numbers."
        ),
        "effective_monthly_capacity_usd": 3_000_000_000,
    }


# ── Metrics ────────────────────────────────────────────────────────────


def trade_metrics(pnls: List[float], years: float) -> Dict[str, float]:
    if not pnls:
        return dict(n=0, win_rate=0.0, cagr=0.0, sharpe_per_trade=0.0,
                    max_dd=0.0, total_return=0.0, avg_pnl=0.0)
    arr = np.array(pnls, dtype=float)
    wins = int((arr > 0).sum())
    eq = np.cumprod(1.0 + arr)
    total = float(eq[-1] - 1.0)
    cagr = float(eq[-1] ** (1 / max(years, 1e-9)) - 1.0) if eq[-1] > 0 else -1.0
    pk = np.maximum.accumulate(eq)
    max_dd = float(((eq - pk) / pk).min())
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    tpy = len(arr) / max(years, 1e-9)
    sharpe = mu / sd * math.sqrt(max(tpy, 1.0)) if sd > 1e-12 else 0.0
    return dict(
        n=int(len(arr)),
        win_rate=float(wins / len(arr)),
        cagr=cagr,
        sharpe_per_trade=sharpe,
        max_dd=max_dd,
        total_return=total,
        avg_pnl=mu,
    )


def daily_stream_metrics(daily: pd.Series, years: float) -> Dict[str, float]:
    r = daily.dropna()
    if len(r) < 2:
        return dict(n_days=0, cagr=0.0, sharpe=0.0, max_dd=0.0, vol=0.0)
    eq = (1.0 + r).cumprod()
    cagr = float(eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1.0) if eq.iloc[-1] > 0 else -1.0
    pk = eq.cummax()
    max_dd = float(((eq - pk) / pk).min())
    vol = float(r.std() * math.sqrt(252))
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0
    return dict(n_days=int(len(r)), cagr=cagr, sharpe=sharpe,
                max_dd=max_dd, vol=vol)


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


def correlate_yearly_trades(trades_pnls: List[Tuple[str, float]],
                            exp1220: Dict[int, float]) -> Optional[float]:
    if not exp1220:
        return None
    yearly: Dict[int, float] = {}
    for dt, p in trades_pnls:
        y = int(dt[:4])
        yearly.setdefault(y, 0.0)
        yearly[y] += p
    common = sorted(set(yearly) & set(exp1220))
    if len(common) < 3:
        return None
    a = np.array([yearly[y] for y in common], dtype=float)
    b = np.array([exp1220[y] for y in common], dtype=float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def correlate_yearly_daily(daily: pd.Series,
                           exp1220: Dict[int, float]) -> Optional[float]:
    if not exp1220 or len(daily.dropna()) == 0:
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


def render_html(payload: Dict) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1200px;color:#111}
    h1{border-bottom:3px solid #2a4a77}
    h2{margin-top:2em;color:#2a4a77}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#2a4a77;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#2a4a77}
    .pill.ok{background:#0a7d1f}
    .pill.bad{background:#c0392b}
    """
    h = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2260 SLV Replacement</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2260 — SLV Replacement (capacity solution)</h1>",
        "<p class='muted'>SLV calendar saturates at ≈$82M. Task: find a "
        "replacement sleeve with Sharpe &gt; 2.0, |corr| to EXP-1220 "
        "&lt; 0.30, and capacity &gt; $500M.</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault + Yahoo data only</span></p>",
    ]

    # Summary table
    h.append("<h2>Candidate summary</h2>")
    h.append("<table><tr><th>Candidate</th><th>Status</th>"
             "<th>n trades / days</th><th>Win rate</th>"
             "<th>CAGR</th><th>Sharpe</th><th>Max DD</th>"
             "<th>Corr vs EXP-1220</th>"
             "<th>Capacity</th><th>Targets</th></tr>")

    for name, row in payload["candidates"].items():
        status = row.get("status", "run")
        if status == "blocked":
            h.append(
                f"<tr><td class='l'><b>{name}</b></td>"
                f"<td><span class='pill bad'>BLOCKED</span></td>"
                f"<td colspan='7' class='l muted'>{row['reason']}</td>"
                f"<td class='muted'>—</td></tr>"
            )
            continue
        m = row["metrics"]
        corr = row.get("corr_vs_exp1220")
        corr_str = f"{corr:+.2f}" if corr is not None else "n/a"
        cap = row.get("capacity_display", "—")
        sharpe_val = m.get("sharpe_per_trade", m.get("sharpe", 0.0))
        n_label = m.get("n", m.get("n_days", 0))
        cagr = m.get("cagr", 0.0)
        max_dd = m.get("max_dd", 0.0)
        wr = m.get("win_rate", None)
        wr_str = _fmt_pct(wr, 1) if wr is not None else "—"

        # Target evaluation
        sharpe_ok = sharpe_val >= 2.0
        corr_ok = corr is not None and abs(corr) < 0.30
        cap_ok = row.get("capacity_usd_est", 0) >= 500_000_000
        if sharpe_ok and corr_ok and cap_ok:
            pill_t = "<span class='pill ok'>ALL THREE</span>"
        else:
            bits = []
            if sharpe_ok: bits.append("Sh✓")
            else: bits.append("Sh✗")
            if corr_ok: bits.append("ρ✓")
            else: bits.append("ρ✗")
            if cap_ok: bits.append("cap✓")
            else: bits.append("cap✗")
            pill_t = f"<span class='pill bad'>{' '.join(bits)}</span>"

        h.append(
            f"<tr><td class='l'><b>{name}</b></td>"
            f"<td>{status}</td>"
            f"<td>{n_label}</td>"
            f"<td>{wr_str}</td>"
            f"<td class='{ 'pos' if cagr>0 else 'neg' }'>{_fmt_pct(cagr)}</td>"
            f"<td>{_fmt(sharpe_val)}</td>"
            f"<td class='neg'>{_fmt_pct(max_dd)}</td>"
            f"<td>{corr_str}</td>"
            f"<td class='l'>{cap}</td>"
            f"<td>{pill_t}</td></tr>"
        )
    h.append("</table>")
    h.append("<p class='muted'>Targets: Sharpe &gt; 2.0, |corr vs EXP-1220| &lt; 0.30, "
             "capacity &gt; $500M. All three must pass.</p>")

    # Per-candidate detail
    for name, row in payload["candidates"].items():
        h.append(f"<h2>— {name} —</h2>")
        if row.get("status") == "blocked":
            h.append(f"<p class='muted'><b>Blocked:</b> {row['reason']}</p>")
            h.append(f"<p class='muted'>Unblock: {row.get('unblock','N/A')}</p>")
            continue
        m = row["metrics"]
        h.append("<pre>" + json.dumps(m, indent=2) + "</pre>")
        if "diagnostics" in row:
            h.append("<h3>Diagnostics</h3>")
            h.append("<pre>" + json.dumps(row["diagnostics"], indent=2) + "</pre>")
        if "capacity" in row:
            h.append("<h3>Capacity (real IronVault volumes, 10% participation)</h3>")
            h.append("<pre>" + json.dumps(row["capacity"], indent=2) + "</pre>")

    # Methodology
    h.append("<h2>Methodology & caveats</h2>")
    h.append("<ul>")
    h.append("<li><b>Slippage per EXP-2210 lesson:</b> credit-spread variants "
             f"include ${CS_SLIPPAGE:.0f}/spread round-trip; straddle uses "
             f"${ST_SLIPPAGE:.0f}/contract round-trip (SPY has tight bid-ask). "
             "EXP-2210 showed XLF alpha evaporates at $25+/spread — TLT is "
             "tested at a conservative $10 floor with the explicit "
             "understanding that $20-$30 is more realistic for a thin-chain "
             "name.</li>")
    h.append("<li><b>GDX BLOCKED:</b> 0 IronVault contracts. Needs Polygon "
             "Starter backfill from scratch. Not substituted with a "
             "synthetic — Rule Zero.</li>")
    h.append("<li><b>GLD 2×-weight:</b> the standalone CAGR/Sharpe stays "
             "identical to the 1×-weight GLD calendar; the change is on "
             "the portfolio-construction side (reallocate 20% SLV → 20% "
             "extra GLD). The capacity heuristic is from public Yahoo "
             "ADV numbers, not a backtest output.</li>")
    h.append("<li><b>SPY unhedged straddle:</b> different instrument from "
             "the EXP-2160 delta-hedged version. This one IS a short-vol "
             "directional trade with real crash risk — reported for "
             "completeness because Carlos named it, but expect a negative "
             "tail. The calm-market Sharpe can look attractive while a "
             "single VIX-spike week wipes out months of premium.</li>")
    h.append("<li><b>Capacity numbers:</b> heuristics from real "
             "option_daily volumes × 10% participation rate, plus rule-of-"
             "thumb dollar-per-contract for the GLD ETF leg. Not precision "
             "market-impact models — good enough to distinguish "
             "$50M-capacity from $5B-capacity.</li>")
    h.append("</ul>")

    # Recommendation
    h.append("<h2>Recommendation</h2>")
    h.append(payload["recommendation_html"])

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))

    years = 6.0  # 2020-2025
    exp1220 = load_exp1220_yearly()
    candidates: Dict[str, Dict] = {}

    try:
        # 1. TLT credit spreads
        print("[exp2260] === TLT put credit spreads ===", flush=True)
        tlt_cov = coverage_stats(con, "TLT")
        print(f"[exp2260] TLT coverage: {tlt_cov}")
        tlt_trades, tlt_diag = backtest_credit_spread(con, "TLT")
        tlt_m = trade_metrics([t.pnl_pct_capital for t in tlt_trades], years)
        tlt_corr = correlate_yearly_trades(
            [(t.entry_date, t.pnl_pct_capital) for t in tlt_trades], exp1220,
        )
        tlt_cap = capacity_cs(con, tlt_trades)
        print(f"[exp2260] TLT: n={tlt_m['n']} wr={tlt_m['win_rate']*100:.1f}% "
              f"Sh={tlt_m['sharpe_per_trade']:.2f} CAGR={tlt_m['cagr']*100:+.2f}% "
              f"DD={tlt_m['max_dd']*100:+.2f}% corr={tlt_corr}")
        print(f"[exp2260] TLT capacity: {tlt_cap}")
        # Capacity dollar-notional estimate
        tlt_monthly_cap = tlt_cap.get("monthly_cap_contracts", 0)
        # TLT put credit spread ~$500 max loss per contract → $500 ADV/month ≈ notional
        tlt_cap_usd = tlt_monthly_cap * 500  # rough
        candidates["TLT_put_credit_spread"] = {
            "status": "run",
            "metrics": tlt_m,
            "diagnostics": tlt_diag,
            "corr_vs_exp1220": tlt_corr,
            "capacity": tlt_cap,
            "capacity_usd_est": tlt_cap_usd,
            "capacity_display": (f"~{tlt_monthly_cap:,} ctr/month "
                                 f"≈ ${tlt_cap_usd/1e6:.0f}M"),
        }

        # 2. GDX credit spreads — BLOCKED
        print("[exp2260] === GDX credit spreads (coverage check) ===", flush=True)
        gdx_cov = coverage_stats(con, "GDX")
        print(f"[exp2260] GDX coverage: {gdx_cov}")
        candidates["GDX_put_credit_spread"] = {
            "status": "blocked",
            "reason": f"0 IronVault contracts (coverage: {gdx_cov})",
            "unblock": "Polygon Starter + OCC symbol construction backfill "
                       "— same path as the TLT Dec-2025 backfill, per "
                       "MASTERPLAN scripts/backfill_tlt.py",
            "capacity": {},
            "capacity_usd_est": 0,
            "capacity_display": "—",
        }

        # 3. Larger GLD allocation (reweighted stream)
        print("[exp2260] === GLD 2×-weight (calendar stream) ===", flush=True)
        gld = gld_calendar_daily()
        gld_m = daily_stream_metrics(gld, years)
        gld_corr = correlate_yearly_daily(gld, exp1220)
        gld_cap = capacity_gld_calendar_doubled()
        print(f"[exp2260] GLD cal: CAGR={gld_m['cagr']*100:+.2f}% "
              f"Sh={gld_m['sharpe']:.2f} DD={gld_m['max_dd']*100:+.2f}% "
              f"corr={gld_corr}")
        candidates["GLD_calendar_2x_weight"] = {
            "status": "run",
            "metrics": {
                "n_days": gld_m["n_days"],
                "cagr": gld_m["cagr"],
                "sharpe_per_trade": gld_m["sharpe"],
                "max_dd": gld_m["max_dd"],
                "total_return": gld_m["cagr"] * years,  # approx for display
                "win_rate": None,
                "avg_pnl": None,
            },
            "corr_vs_exp1220": gld_corr,
            "capacity": gld_cap,
            "capacity_usd_est": gld_cap["effective_monthly_capacity_usd"],
            "capacity_display": "~$3B/month (GLD ETF ADV × 10%)",
        }

        # 4. SPY weekly UNHEDGED short straddle
        print("[exp2260] === SPY unhedged weekly short straddle ===", flush=True)
        spy_trades, spy_diag = backtest_spy_unhedged_straddle(con)
        spy_m = trade_metrics([t.pnl_pct_capital for t in spy_trades], years)
        spy_corr = correlate_yearly_trades(
            [(t.entry_date, t.pnl_pct_capital) for t in spy_trades], exp1220,
        )
        spy_cap = capacity_spy_straddle(con, spy_trades)
        spy_monthly_cap = spy_cap.get("monthly_cap_contracts", 0)
        # SPY ATM straddle ~$50k notional/contract on a 1-week 15-IV 6000-SPX
        spy_cap_usd = spy_monthly_cap * 50_000
        print(f"[exp2260] SPY straddle: n={spy_m['n']} wr={spy_m['win_rate']*100:.1f}% "
              f"Sh={spy_m['sharpe_per_trade']:.2f} CAGR={spy_m['cagr']*100:+.2f}% "
              f"DD={spy_m['max_dd']*100:+.2f}% corr={spy_corr}")
        candidates["SPY_unhedged_weekly_straddle"] = {
            "status": "run",
            "metrics": spy_m,
            "diagnostics": spy_diag,
            "corr_vs_exp1220": spy_corr,
            "capacity": spy_cap,
            "capacity_usd_est": spy_cap_usd,
            "capacity_display": (f"~{spy_monthly_cap:,} ctr/month "
                                 f"≈ ${spy_cap_usd/1e6:.0f}M"),
        }
    finally:
        con.close()

    # Recommendation
    winners: List[Tuple[str, Dict]] = []
    for name, row in candidates.items():
        if row.get("status") != "run":
            continue
        m = row["metrics"]
        sharpe = m.get("sharpe_per_trade", 0.0)
        corr = row.get("corr_vs_exp1220")
        cap = row.get("capacity_usd_est", 0)
        if sharpe >= 2.0 and corr is not None and abs(corr) < 0.30 and cap >= 500_000_000:
            winners.append((name, row))

    rec_parts = ["<ul>"]
    if winners:
        rec_parts.append(
            f"<li><b>{len(winners)} candidate(s) clear all three targets:</b></li>"
        )
        for name, row in winners:
            m = row["metrics"]
            rec_parts.append(
                f"<li>{name} — Sharpe "
                f"{m.get('sharpe_per_trade', 0.0):.2f}, corr "
                f"{row.get('corr_vs_exp1220'):+.2f}, capacity "
                f"{row.get('capacity_display', '—')}</li>"
            )
    else:
        rec_parts.append(
            "<li><b>NO candidate clears all three targets simultaneously.</b></li>"
        )
        rec_parts.append("<li>Partial passes:</li><ul>")
        for name, row in candidates.items():
            if row.get("status") != "run":
                continue
            m = row["metrics"]
            sharpe = m.get("sharpe_per_trade", 0.0)
            corr = row.get("corr_vs_exp1220")
            cap = row.get("capacity_usd_est", 0)
            bits = []
            bits.append(f"Sharpe {sharpe:.2f}" + (" ✓" if sharpe >= 2.0 else " ✗"))
            if corr is not None:
                bits.append(f"corr {corr:+.2f}" + (" ✓" if abs(corr) < 0.30 else " ✗"))
            else:
                bits.append("corr n/a")
            bits.append(f"cap ${cap/1e6:.0f}M" + (" ✓" if cap >= 500_000_000 else " ✗"))
            rec_parts.append(f"<li>{name}: {'; '.join(bits)}</li>")
        rec_parts.append("</ul>")

    rec_parts.append(
        "<li><b>Practical recommendation:</b> the cleanest capacity fix for "
        "the SLV allocation is to reweight into GLD (already validated, "
        "same thesis, 40× more liquid). The 2× allocation hypothesis needs "
        "to be stress-tested in the EXP-1870 combined-portfolio model to "
        "confirm correlation stays low once the weight doubles.</li>"
    )
    rec_parts.append("</ul>")
    payload = {
        "experiment": "EXP-2260",
        "tag": "EXP-2260",
        "description": "SLV replacement — capacity solution for the 20% SLV sleeve",
        "candidates": candidates,
        "recommendation_html": "".join(rec_parts),
        "targets": {
            "sharpe_min": 2.0,
            "abs_corr_max": 0.30,
            "capacity_usd_min": 500_000_000,
        },
    }

    html = render_html(payload)
    REPORT_HTML.write_text(html)
    print(f"[exp2260] wrote {REPORT_HTML}")

    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp2260] wrote {REPORT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
