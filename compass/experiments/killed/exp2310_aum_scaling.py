"""
EXP-2310 — AUM Scaling Research: Replace GLD/SLV Calendar Spreads.

The 5-stream static portfolio holds 20% GLD-calendar + 20% SLV-calendar
sleeves. Both saturate between $50-80M. For production deployment at
book sizes above $500M these calendar sleeves are the binding capacity
constraint.

This experiment asks: what LIQUID credit-spread / strangle sleeves on
real IronVault option coverage can replace one or both calendar legs
at ≥$500M of effective capacity, Sharpe > 1.5, and low correlation to
EXP-1220?

Carlos's requested candidates, with real IronVault coverage audit:

  1. TLT credit spreads or strangles  — 10 749 contracts, 1501 trading
                                        dates                      RUN
  2. GDX credit spreads               — 0 contracts             BLOCKED
  3. EEM credit spreads               — 0 contracts             BLOCKED
  4. DIA credit spreads               — 0 contracts             BLOCKED
  5. SPX / ^SPX / XSP credit spreads  — 0 contracts             BLOCKED

Four of the five requested candidates are not in IronVault. Rather
than substitute with synthetic chains (Rule Zero) or call the
experiment "mostly blocked and done", this file ALSO tests three
high-liquidity ETF substitutes that ARE in IronVault and that cover
the same economic exposures Carlos wanted:

  SUBSTITUTE                COVERS CARLOS'S                REASONING
  QQQ put credit spread     DIA (broad-index analog)       Nasdaq-100 ATM
                                                           is ~$2B/day ADV
  XLK put credit spread     SPX / tech-beta analog         Tech sector
                                                           options are deep
  XLE put credit spread     GDX (commodity-sector analog)  Energy sector
                                                           shares GDX-class
                                                           mean-reversion

Plus TLT short strangle as the second TLT variant Carlos explicitly
named.

REAL DATA — Rule Zero:
  * Option chains from IronVault data/options_cache.db
  * Spot from Yahoo
  * IV inverted via Brent only for strike-Δ selection, not for fills
  * Slippage applied per EXP-2210 lesson:
      - put credit spreads: $10 / spread round-trip
      - strangles         : $20 / contract round-trip (2 legs)

Outputs:
  compass/exp2310_aum_scaling.py            (this file)
  compass/reports/exp2310_aum_scaling.json
  compass/reports/exp2310_aum_scaling.html

Tag: EXP-2310
Run: python3 -m compass.exp2310_aum_scaling
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
REPORT_JSON = REPORT_DIR / "exp2310_aum_scaling.json"
REPORT_HTML = REPORT_DIR / "exp2310_aum_scaling.html"
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

CS_SHORT_DELTA = -0.30
CS_LONG_DELTA = -0.15
CS_TARGET_DTE = 30
CS_RISK_PER_TRADE = 0.02
CS_SLIPPAGE = 10.0

STG_PUT_DELTA = -0.20      # short strangle legs at ±0.20 Δ
STG_CALL_DELTA = 0.20
STG_TARGET_DTE = 30
STG_RISK_PER_TRADE = 0.02
STG_SLIPPAGE = 20.0

PARTICIPATION_RATE = 0.10

# Carlos targets for EXP-2310
TGT_SHARPE = 1.5
TGT_CORR = 0.30
TGT_CAPACITY = 500_000_000   # $500M


# ── Helpers ───────────────────────────────────────────────────────────


@dataclass
class CSTrade:
    ticker: str
    entry_date: str
    expiration: str
    short_strike: float
    long_strike: float
    short_delta: float
    short_symbol: str
    long_symbol: str
    net_credit: float
    exit_net: float
    pnl_per_spread: float
    pnl_pct_capital: float
    real_exit: bool


@dataclass
class StrangleTrade:
    ticker: str
    entry_date: str
    expiration: str
    put_strike: float
    call_strike: float
    put_symbol: str
    call_symbol: str
    premium_collected: float
    pnl_per_contract: float
    pnl_pct_capital: float
    spot_entry: float
    spot_exit: float


def _weekly_snaps(all_dates: List[str]) -> List[str]:
    by_week: Dict[Tuple[int, int], str] = {}
    for s in all_dates:
        wk = datetime.strptime(s, "%Y-%m-%d").isocalendar()[:2]
        by_week.setdefault(wk, s)
    return sorted(by_week.values())


def backtest_put_credit_spread(con: sqlite3.Connection, ticker: str
                               ) -> Tuple[List[CSTrade], Dict[str, int]]:
    spot = fetch_yahoo_close(ticker)
    all_dates = list_snapshot_dates(con, ticker)
    all_dates = [d for d in all_dates if START <= d <= END]
    weekly = _weekly_snaps(all_dates)
    diag = {"n_attempted": 0, "n_dropped_no_exp": 0,
            "n_dropped_thin_chain": 0, "n_real_exits": 0, "n_intrinsic_exits": 0}
    trades: List[CSTrade] = []
    for snap in weekly:
        diag["n_attempted"] += 1
        try:
            spot_val = float(spot.loc[:snap].iloc[-1])
        except (KeyError, IndexError):
            continue
        expiration = pick_expiration(con, ticker, snap, CS_TARGET_DTE, "P", min_dte=7)
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
        table = []
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
        try:
            spot_exit = float(spot.loc[:expiration].iloc[-1])
        except (KeyError, IndexError):
            spot_exit = spot_val
        short_exit_info = fetch_contract_close(con, short_sym, snap, expiration)
        long_exit_info = fetch_contract_close(con, long_sym, snap, expiration)
        real_short = short_exit_info is not None and short_exit_info[0] != snap
        real_long = long_exit_info is not None and long_exit_info[0] != snap
        short_exit = float(short_exit_info[1]) if real_short else max(short_K - spot_exit, 0.0)
        long_exit = float(long_exit_info[1]) if real_long else max(long_K - spot_exit, 0.0)
        real_exit = real_short and real_long
        if real_exit:
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
            ticker=ticker, entry_date=snap, expiration=expiration,
            short_strike=float(short_K), long_strike=float(long_K),
            short_delta=float(short_d),
            short_symbol=short_sym, long_symbol=long_sym,
            net_credit=float(net_credit), exit_net=float(exit_net),
            pnl_per_spread=float(pnl_net), pnl_pct_capital=float(pnl_pct),
            real_exit=bool(real_exit),
        ))
    return trades, diag


def backtest_strangle(con: sqlite3.Connection, ticker: str
                      ) -> Tuple[List[StrangleTrade], Dict[str, int]]:
    spot = fetch_yahoo_close(ticker)
    all_dates = list_snapshot_dates(con, ticker)
    all_dates = [d for d in all_dates if START <= d <= END]
    weekly = _weekly_snaps(all_dates)
    diag = {"n_attempted": 0, "n_dropped_no_exp": 0, "n_dropped_thin": 0}
    trades: List[StrangleTrade] = []
    for snap in weekly:
        diag["n_attempted"] += 1
        try:
            spot_val = float(spot.loc[:snap].iloc[-1])
        except (KeyError, IndexError):
            continue
        expiration = pick_expiration(con, ticker, snap, STG_TARGET_DTE, "P", min_dte=7)
        if expiration is None:
            diag["n_dropped_no_exp"] += 1
            continue
        put_chain = fetch_chain(con, ticker, snap, expiration, "P")
        call_chain = fetch_chain(con, ticker, snap, expiration, "C")
        if len(put_chain) < 4 or len(call_chain) < 4:
            diag["n_dropped_thin"] += 1
            continue
        snap_dt = datetime.strptime(snap, "%Y-%m-%d")
        exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
        T = (exp_dt - snap_dt).days / 365.0
        if T <= 0:
            continue
        # Put leg
        put_table = []
        for K, px, sym in put_chain:
            sigma = implied_vol_put(px, spot_val, K, T, RISK_FREE)
            if sigma is None or sigma <= 0: continue
            delta = bs_put_delta(spot_val, K, T, sigma, RISK_FREE)
            put_table.append((K, px, sym, delta))
        call_table = []
        for K, px, sym in call_chain:
            sigma = implied_vol_call(px, spot_val, K, T, RISK_FREE)
            if sigma is None or sigma <= 0: continue
            delta = bs_call_delta(spot_val, K, T, sigma, RISK_FREE)
            call_table.append((K, px, sym, delta))
        if len(put_table) < 2 or len(call_table) < 2:
            diag["n_dropped_thin"] += 1
            continue
        put_row = min(put_table, key=lambda r: abs(r[3] - STG_PUT_DELTA))
        call_row = min(call_table, key=lambda r: abs(r[3] - STG_CALL_DELTA))
        p_K, p_px, p_sym, _ = put_row
        c_K, c_px, c_sym, _ = call_row
        premium = p_px + c_px
        if premium <= 0:
            continue
        try:
            spot_exit = float(spot.loc[:expiration].iloc[-1])
        except (KeyError, IndexError):
            spot_exit = spot_val
        p_exit_info = fetch_contract_close(con, p_sym, snap, expiration)
        c_exit_info = fetch_contract_close(con, c_sym, snap, expiration)
        if p_exit_info is not None and p_exit_info[0] != snap:
            p_exit = float(p_exit_info[1])
        else:
            p_exit = max(p_K - spot_exit, 0.0)
        if c_exit_info is not None and c_exit_info[0] != snap:
            c_exit = float(c_exit_info[1])
        else:
            c_exit = max(spot_exit - c_K, 0.0)
        pnl_per_share = (p_px - p_exit) + (c_px - c_exit)
        pnl_per_contract = pnl_per_share * 100.0 - STG_SLIPPAGE
        # Stress sizing: premium × 3 ≈ max-loss proxy
        stress_loss = max(premium * 3.0, 0.01) * 100.0
        n_contracts = (STG_RISK_PER_TRADE * CAPITAL) / stress_loss
        pnl_pct = pnl_per_contract * n_contracts / CAPITAL
        trades.append(StrangleTrade(
            ticker=ticker, entry_date=snap, expiration=expiration,
            put_strike=float(p_K), call_strike=float(c_K),
            put_symbol=p_sym, call_symbol=c_sym,
            premium_collected=float(premium),
            pnl_per_contract=float(pnl_per_contract),
            pnl_pct_capital=float(pnl_pct),
            spot_entry=float(spot_val), spot_exit=float(spot_exit),
        ))
    return trades, diag


# ── Metrics ────────────────────────────────────────────────────────────


def trade_metrics(pnls: List[float], years: float) -> Dict[str, float]:
    if not pnls:
        return dict(n=0, win_rate=0.0, cagr=0.0, sharpe_per_trade=0.0,
                    max_dd=0.0, total_return=0.0, avg_pnl=0.0,
                    worst_trade=0.0)
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
        cagr=cagr, sharpe_per_trade=sharpe,
        max_dd=max_dd, total_return=total, avg_pnl=mu,
        worst_trade=float(arr.min()),
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


def correlate_yearly(entry_pnls: List[Tuple[str, float]],
                     exp1220: Dict[int, float]) -> Optional[float]:
    if not exp1220 or not entry_pnls:
        return None
    yearly: Dict[int, float] = {}
    for dt, p in entry_pnls:
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


# ── Capacity ───────────────────────────────────────────────────────────


def capacity_cs(con: sqlite3.Connection, trades: List[CSTrade]) -> Dict:
    if not trades:
        return {"n": 0}
    vols: List[int] = []
    for t in trades:
        row = con.execute("""
            SELECT d.volume FROM option_contracts c
            JOIN option_daily d ON c.contract_symbol = d.contract_symbol
            WHERE c.ticker=? AND c.expiration=? AND c.strike=? AND c.option_type='P'
              AND d.date=? LIMIT 1
        """, (t.ticker, t.expiration, t.short_strike, t.entry_date)).fetchone()
        if row and row[0]:
            vols.append(int(row[0]))
    if not vols:
        return {"n_observations": 0}
    v = np.array(vols, dtype=float)
    median_cap = float(np.median(v) * PARTICIPATION_RATE)
    monthly_cap = median_cap * 4
    # Dollar capacity rough-estimate: $500 max-loss per spread × contracts
    usd_month = monthly_cap * 500
    return {
        "n_observations": int(len(vols)),
        "median_daily_volume": float(np.median(v)),
        "p5_daily_volume": float(np.percentile(v, 5)),
        "median_max_contracts_per_entry": int(median_cap),
        "monthly_cap_contracts": int(monthly_cap),
        "monthly_usd_est": float(usd_month),
    }


def capacity_stg(con: sqlite3.Connection, trades: List[StrangleTrade]) -> Dict:
    if not trades:
        return {"n": 0}
    vols: List[int] = []
    for t in trades:
        # Use min of put and call volumes
        row = con.execute("""
            SELECT MIN(d.volume) FROM option_contracts c
            JOIN option_daily d ON c.contract_symbol = d.contract_symbol
            WHERE c.ticker=? AND c.expiration=? AND d.date=?
              AND ((c.option_type='P' AND c.strike=?) OR (c.option_type='C' AND c.strike=?))
        """, (t.ticker, t.expiration, t.entry_date, t.put_strike, t.call_strike)).fetchone()
        if row and row[0]:
            vols.append(int(row[0]))
    if not vols:
        return {"n_observations": 0}
    v = np.array(vols, dtype=float)
    median_cap = float(np.median(v) * PARTICIPATION_RATE)
    monthly_cap = median_cap * 4
    # Strangle notional: premium × 3 × 100 rough max-loss, assume ~$800/ctr
    usd_month = monthly_cap * 800
    return {
        "n_observations": int(len(vols)),
        "median_daily_volume": float(np.median(v)),
        "p5_daily_volume": float(np.percentile(v, 5)),
        "median_max_contracts_per_entry": int(median_cap),
        "monthly_cap_contracts": int(monthly_cap),
        "monthly_usd_est": float(usd_month),
    }


# ── HTML ───────────────────────────────────────────────────────────────


def _fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x*100:.{dp}f}%" if np.isfinite(x) else "—"


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def _dollar(x: float) -> str:
    if x >= 1e9: return f"${x/1e9:.2f}B"
    if x >= 1e6: return f"${x/1e6:.1f}M"
    if x >= 1e3: return f"${x/1e3:.0f}k"
    return f"${x:.0f}"


def render_html(payload: Dict) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1280px;color:#111}
    h1{border-bottom:3px solid #1c3a5e}
    h2{margin-top:2em;color:#1c3a5e}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#1c3a5e;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#1c3a5e}
    .pill.ok{background:#0a7d1f}
    .pill.bad{background:#c0392b}
    .pill.warn{background:#c07a1f}
    """
    h = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2310 AUM Scaling</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2310 — AUM Scaling: Replace GLD/SLV Calendar Spreads</h1>",
        "<p class='muted'>Looking for credit-spread / strangle sleeves with "
        "Sharpe &gt; 1.5, |corr vs EXP-1220| &lt; 0.30, and capacity "
        "&gt; $500M as scalable replacements for the bandwidth-"
        "constrained GLD/SLV calendar sleeves.</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault + Yahoo data only</span></p>",
    ]

    # Coverage audit
    h.append("<h2>IronVault coverage audit (requested + substitutes)</h2>")
    h.append("<table><tr><th>Ticker</th><th>Role</th><th>Contracts</th>"
             "<th>Trading dates with bars</th><th>Status</th></tr>")
    for tk, row in payload["coverage"].items():
        status = row["status"]
        cls = {"RUN": "ok", "BLOCKED": "bad", "SUBSTITUTE": "warn"}.get(status, "")
        h.append(
            f"<tr><td class='l'><b>{tk}</b></td>"
            f"<td class='l'>{row['role']}</td>"
            f"<td>{row['n_contracts']:,}</td>"
            f"<td>{row['n_trading_dates']:,}</td>"
            f"<td><span class='pill {cls}'>{status}</span></td></tr>"
        )
    h.append("</table>")

    # Results summary
    h.append("<h2>Results — Sharpe &gt; 1.5, |corr| &lt; 0.30, capacity &gt; $500M</h2>")
    h.append("<table><tr><th>Variant</th><th>n</th><th>WR</th>"
             "<th>CAGR</th><th>Sharpe/trade</th><th>Max DD</th>"
             "<th>Corr vs EXP-1220</th><th>Capacity</th><th>Targets</th></tr>")
    for name, r in payload["variants"].items():
        if r.get("status") == "blocked":
            h.append(
                f"<tr><td class='l'><b>{name}</b></td>"
                f"<td colspan='7' class='l muted'>{r['reason']}</td>"
                f"<td><span class='pill bad'>blocked</span></td></tr>"
            )
            continue
        m = r["metrics"]
        corr = r.get("corr_vs_exp1220")
        corr_str = f"{corr:+.2f}" if corr is not None else "n/a"
        cap_usd = r.get("capacity", {}).get("monthly_usd_est", 0)
        cap_str = _dollar(cap_usd) + "/mo"
        sharpe_ok = m["sharpe_per_trade"] >= TGT_SHARPE
        corr_ok = corr is not None and abs(corr) < TGT_CORR
        cap_ok = cap_usd >= TGT_CAPACITY
        if sharpe_ok and corr_ok and cap_ok:
            pill = "<span class='pill ok'>ALL 3</span>"
        else:
            bits = [f"Sh{'✓' if sharpe_ok else '✗'}",
                    f"ρ{'✓' if corr_ok else '✗'}",
                    f"cap{'✓' if cap_ok else '✗'}"]
            pill = f"<span class='pill bad'>{' '.join(bits)}</span>"
        h.append(
            f"<tr><td class='l'><b>{name}</b></td>"
            f"<td>{m['n']}</td>"
            f"<td>{_fmt_pct(m['win_rate'], 1)}</td>"
            f"<td class='{ 'pos' if m['cagr']>0 else 'neg' }'>{_fmt_pct(m['cagr'])}</td>"
            f"<td>{_fmt(m['sharpe_per_trade'])}</td>"
            f"<td class='neg'>{_fmt_pct(m['max_dd'])}</td>"
            f"<td>{corr_str}</td>"
            f"<td class='l'>{cap_str}</td>"
            f"<td>{pill}</td></tr>"
        )
    h.append("</table>")

    # Per-variant detail
    for name, r in payload["variants"].items():
        if r.get("status") == "blocked":
            continue
        h.append(f"<h3>{name} — detail</h3>")
        h.append("<pre>" + json.dumps({
            "metrics": r["metrics"],
            "diagnostics": r.get("diagnostics", {}),
            "capacity": r.get("capacity", {}),
            "corr_vs_exp1220": r.get("corr_vs_exp1220"),
        }, indent=2) + "</pre>")

    # Recommendation
    h.append("<h2>Recommendation</h2>")
    h.append(payload["recommendation_html"])

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    years = 6.0
    exp1220 = load_exp1220_yearly()

    try:
        # Coverage audit
        coverage_rows: Dict[str, Dict] = {}
        for tk, role in (
            ("TLT", "Carlos #1 (requested)"),
            ("GDX", "Carlos #2 (requested)"),
            ("EEM", "Carlos #3 (requested)"),
            ("DIA", "Carlos #4 (requested)"),
            ("SPX", "Carlos #5 (requested)"),
            ("QQQ", "substitute for DIA (broad-index)"),
            ("XLK", "substitute for SPX (tech-beta)"),
            ("XLE", "substitute for GDX (commodity-sector)"),
        ):
            cov = coverage_stats(con, tk)
            n_td = con.execute("""
                SELECT COUNT(DISTINCT d.date) FROM option_daily d
                JOIN option_contracts c USING(contract_symbol)
                WHERE c.ticker=? AND d.close > 0
            """, (tk,)).fetchone()[0]
            if cov["n_contracts"] == 0:
                status = "BLOCKED"
            elif "substitute" in role:
                status = "SUBSTITUTE"
            else:
                status = "RUN"
            coverage_rows[tk] = {
                "role": role,
                "n_contracts": cov["n_contracts"],
                "n_trading_dates": int(n_td),
                "status": status,
            }
            print(f"[exp2310] coverage {tk}: {coverage_rows[tk]}")

        # Variants to run (those not blocked)
        variants: Dict[str, Dict] = {}

        # TLT credit spread
        print("\n[exp2310] === TLT put credit spread ===", flush=True)
        tlt_trades, tlt_diag = backtest_put_credit_spread(con, "TLT")
        tlt_m = trade_metrics([t.pnl_pct_capital for t in tlt_trades], years)
        tlt_corr = correlate_yearly(
            [(t.entry_date, t.pnl_pct_capital) for t in tlt_trades], exp1220,
        )
        tlt_cap = capacity_cs(con, tlt_trades)
        print(f"[exp2310] TLT CS: n={tlt_m['n']} Sh={tlt_m['sharpe_per_trade']:.2f} "
              f"CAGR={tlt_m['cagr']*100:+.2f}% DD={tlt_m['max_dd']*100:+.2f}% "
              f"corr={tlt_corr}")
        variants["TLT_put_credit_spread"] = {
            "metrics": tlt_m, "diagnostics": tlt_diag,
            "corr_vs_exp1220": tlt_corr, "capacity": tlt_cap,
        }

        # TLT strangle
        print("[exp2310] === TLT short strangle ===", flush=True)
        tlt_stg, tlt_stg_diag = backtest_strangle(con, "TLT")
        tlt_stg_m = trade_metrics([t.pnl_pct_capital for t in tlt_stg], years)
        tlt_stg_corr = correlate_yearly(
            [(t.entry_date, t.pnl_pct_capital) for t in tlt_stg], exp1220,
        )
        tlt_stg_cap = capacity_stg(con, tlt_stg)
        print(f"[exp2310] TLT stg: n={tlt_stg_m['n']} Sh={tlt_stg_m['sharpe_per_trade']:.2f} "
              f"CAGR={tlt_stg_m['cagr']*100:+.2f}% DD={tlt_stg_m['max_dd']*100:+.2f}% "
              f"corr={tlt_stg_corr}")
        variants["TLT_short_strangle"] = {
            "metrics": tlt_stg_m, "diagnostics": tlt_stg_diag,
            "corr_vs_exp1220": tlt_stg_corr, "capacity": tlt_stg_cap,
        }

        # Blocked
        for tk, reason, unblock in [
            ("GDX", "0 IronVault contracts",
             "Polygon Starter + OCC construction (same path as TLT Dec-2025 backfill)"),
            ("EEM", "0 IronVault contracts",
             "Polygon Starter backfill"),
            ("DIA", "0 IronVault contracts",
             "Polygon Starter backfill — DIA is a broad-index ETF, deep chains"),
            ("SPX", "0 IronVault contracts — the SPX cash index (and mini-SPX XSP) are not in IronVault",
             "Requires a CBOE / OPRA feed; not available via Polygon Starter"),
        ]:
            variants[f"{tk}_put_credit_spread"] = {
                "status": "blocked",
                "reason": f"{reason}. Unblock: {unblock}",
            }

        # Substitutes
        for tk in ("QQQ", "XLK", "XLE"):
            print(f"[exp2310] === {tk} put credit spread (substitute) ===", flush=True)
            trades, diag = backtest_put_credit_spread(con, tk)
            m = trade_metrics([t.pnl_pct_capital for t in trades], years)
            corr = correlate_yearly(
                [(t.entry_date, t.pnl_pct_capital) for t in trades], exp1220,
            )
            cap = capacity_cs(con, trades)
            print(f"[exp2310] {tk}: n={m['n']} Sh={m['sharpe_per_trade']:.2f} "
                  f"CAGR={m['cagr']*100:+.2f}% DD={m['max_dd']*100:+.2f}% corr={corr}")
            variants[f"{tk}_put_credit_spread"] = {
                "metrics": m, "diagnostics": diag,
                "corr_vs_exp1220": corr, "capacity": cap,
            }
    finally:
        con.close()

    # Recommendation
    winners = []
    partial = []
    for name, r in variants.items():
        if r.get("status") == "blocked":
            continue
        m = r["metrics"]
        sharpe = m["sharpe_per_trade"]
        corr = r.get("corr_vs_exp1220")
        cap = r.get("capacity", {}).get("monthly_usd_est", 0)
        sharpe_ok = sharpe >= TGT_SHARPE
        corr_ok = corr is not None and abs(corr) < TGT_CORR
        cap_ok = cap >= TGT_CAPACITY
        if sharpe_ok and corr_ok and cap_ok:
            winners.append((name, sharpe, corr, cap))
        else:
            partial.append((name, sharpe, corr, cap, sharpe_ok, corr_ok, cap_ok))

    rec = ["<ul>"]
    if winners:
        rec.append("<li><b>Candidates clearing ALL three targets:</b></li>")
        rec.append("<ul>")
        for name, s, c, cap in sorted(winners, key=lambda x: -x[1]):
            rec.append(
                f"<li><b>{name}</b> — Sharpe {s:.2f}, corr {c:+.2f}, "
                f"capacity {_dollar(cap)}/mo</li>"
            )
        rec.append("</ul>")
    else:
        rec.append(
            "<li><b>No candidate clears all three targets simultaneously</b> "
            "(Sharpe ≥ 1.5, |ρ| &lt; 0.30, capacity ≥ $500M).</li>"
        )
    rec.append("<li>Per-candidate partial passes:<ul>")
    for name, s, c, cap, sok, cok, capok in partial:
        c_str = f"{c:+.2f}" if c is not None else "n/a"
        rec.append(
            f"<li><b>{name}</b>: Sh {s:.2f} {'✓' if sok else '✗'}, "
            f"ρ {c_str} {'✓' if cok else '✗'}, "
            f"cap {_dollar(cap)} {'✓' if capok else '✗'}</li>"
        )
    rec.append("</ul></li>")
    rec.append(
        "<li><b>Carlos's requested candidates 2-5 (GDX, EEM, DIA, SPX) are "
        "BLOCKED in IronVault.</b> Unblock requires a Polygon Starter backfill "
        "via OCC symbol construction — same path as the TLT Dec-2025 backfill "
        "(<code>scripts/backfill_tlt.py</code>). SPX is harder because the "
        "cash index needs a CBOE / OPRA feed, not Polygon Starter. Until "
        "these are backfilled, only TLT is a genuine answer to the "
        "original question.</li>"
    )
    rec.append("</ul>")
    payload = {
        "experiment": "EXP-2310",
        "tag": "EXP-2310",
        "description": "AUM scaling research — replace GLD/SLV calendar spreads",
        "targets": {
            "sharpe_min": TGT_SHARPE,
            "abs_corr_max": TGT_CORR,
            "capacity_usd_min": TGT_CAPACITY,
        },
        "coverage": coverage_rows,
        "variants": variants,
        "recommendation_html": "".join(rec),
    }

    html = render_html(payload)
    REPORT_HTML.write_text(html)
    print(f"\n[exp2310] wrote {REPORT_HTML}")

    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp2310] wrote {REPORT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
