"""
EXP-2210 — XLF/XLI Credit Spread Deep Validation.

EXP-2160 printed a per-trade Sharpe of 11.3 for XLF and 6.0 for XLI with
98% / 97% win rates on real IronVault data. Those numbers are extraordinary
and the honest EXP-2160 commit flagged them as artifact-suspect. This
experiment runs six formal validation passes to determine whether the
result is real alpha, a data artifact, or a methodology bug:

  1. Walk-forward with expanding window (train 252d, test 63d, 20 folds)
  2. T+1 entry pricing (shift every entry to the next available trading
     day's real close — removes any implicit close-at-signal lookahead)
  3. Survivorship / data completeness audit (how many entries had a
     REAL exit price vs intrinsic-against-spot fallback, how many
     candidate snapshots got dropped for missing chains)
  4. Parameter sensitivity — grid over short Δ ∈ {-0.20, -0.25, -0.30,
     -0.35, -0.40} × DTE target ∈ {14, 21, 30, 45}
  5. Regime analysis — yearly breakdown 2020-2025 with focus on 2022
     bear market (SPY -18%, XLF -12%, XLI -8%)
  6. Slippage/cost analysis — subtract $0, $10, $25, $50 per spread
     (round-trip) and re-measure Sharpe / CAGR

Pipeline: every variant calls a single `run_cs_config` function that is
a hardened version of EXP-2160's put-credit-spread loop. REAL IronVault
option closes, REAL Yahoo spot, BS inversion for strike-delta selection
only (not for fills). JSON + HTML output.

Rule Zero respected throughout — no synthetic fills, no random draws.

Outputs:
  compass/exp2210_xlf_xli_validation.py            (this file)
  compass/reports/exp2210_xlf_xli_validation.json
  compass/reports/exp2210_xlf_xli_validation.html

Tag: EXP-2210
Run: python3 -m compass.exp2210_xlf_xli_validation
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
REPORT_JSON = REPORT_DIR / "exp2210_xlf_xli_validation.json"
REPORT_HTML = REPORT_DIR / "exp2210_xlf_xli_validation.html"
DB_PATH = ROOT / "data" / "options_cache.db"

from compass.exp1960_skew_alpha import (
    implied_vol_put, bs_put_delta, fetch_contract_close,
)
from compass.exp2160_high_capacity_alts import (
    fetch_yahoo_close, list_snapshot_dates, pick_expiration,
    fetch_chain, coverage_stats,
)

START = "2020-01-01"
END = "2025-12-31"
CAPITAL = 100_000.0
RISK_FREE = 0.045

TICKERS = ("XLF", "XLI")


# ── Config ─────────────────────────────────────────────────────────────


@dataclass
class CSConfig:
    ticker: str
    short_delta: float = -0.30
    long_delta: float = -0.15
    target_dte: int = 30
    min_dte: int = 7
    risk_per_trade: float = 0.02
    entry_offset_days: int = 0       # 0 = enter at snap close, 1 = T+1 close
    slippage_per_spread: float = 0.0 # dollars per round-trip
    stop_loss_multiplier: Optional[float] = None  # exit if mark > N × credit
    label: str = "baseline"


@dataclass
class TradeRow:
    entry_date: str
    effective_entry_date: str
    expiration: str
    short_strike: float
    long_strike: float
    short_entry: float
    long_entry: float
    short_exit: float
    long_exit: float
    short_exit_source: str          # "option_daily" | "intrinsic"
    long_exit_source: str
    net_credit: float
    exit_net: float
    pnl_before_slippage: float      # per contract (×100)
    slippage: float
    pnl_per_spread: float
    pnl_pct_capital: float
    short_delta_estimate: float
    long_delta_estimate: float
    spot_entry: float
    spot_exit: float
    win: bool


@dataclass
class BacktestResult:
    config: CSConfig
    trades: List[TradeRow]
    n_attempted: int
    n_dropped_no_expiration: int
    n_dropped_thin_chain: int
    n_dropped_no_credit: int
    n_exits_real: int
    n_exits_intrinsic: int


# ── Core backtest loop ────────────────────────────────────────────────


def _next_trading_date(all_dates: List[str], current: str, offset: int) -> Optional[str]:
    """Return the date that is `offset` trading days after `current`."""
    try:
        idx = all_dates.index(current)
    except ValueError:
        # Fall back: first date > current
        for i, d in enumerate(all_dates):
            if d > current:
                idx = i - offset
                break
        else:
            return None
    target = idx + offset
    if target < 0 or target >= len(all_dates):
        return None
    return all_dates[target]


def run_cs_config(con: sqlite3.Connection, spot: pd.Series,
                  all_dates: List[str], cfg: CSConfig) -> BacktestResult:
    """Run a single config. Returns all trades + diagnostics."""
    # Weekly entries: first trading day of each ISO week that has data
    by_week: Dict[Tuple[int, int], str] = {}
    for s in all_dates:
        sd = datetime.strptime(s, "%Y-%m-%d")
        wk = sd.isocalendar()[:2]
        by_week.setdefault(wk, s)
    weekly_snaps = sorted(by_week.values())

    trades: List[TradeRow] = []
    n_attempted = 0
    n_dropped_no_expiration = 0
    n_dropped_thin_chain = 0
    n_dropped_no_credit = 0
    n_exits_real = 0
    n_exits_intrinsic = 0

    for snap in weekly_snaps:
        n_attempted += 1

        # Effective entry date (T+0 or T+1)
        effective_entry = _next_trading_date(all_dates, snap, cfg.entry_offset_days)
        if effective_entry is None:
            continue

        try:
            spot_entry = float(spot.loc[:effective_entry].iloc[-1])
        except (KeyError, IndexError):
            continue

        expiration = pick_expiration(
            con, cfg.ticker, effective_entry, cfg.target_dte, "P", min_dte=cfg.min_dte,
        )
        if expiration is None:
            n_dropped_no_expiration += 1
            continue
        chain = fetch_chain(con, cfg.ticker, effective_entry, expiration, "P")
        if len(chain) < 5:
            n_dropped_thin_chain += 1
            continue

        exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
        eff_dt = datetime.strptime(effective_entry, "%Y-%m-%d")
        T = (exp_dt - eff_dt).days / 365.0
        if T <= 0:
            continue

        # Compute σ and Δ per strike
        table: List[Tuple[float, float, str, float]] = []
        for K, px, sym in chain:
            sigma = implied_vol_put(px, spot_entry, K, T, RISK_FREE)
            if sigma is None or sigma <= 0:
                continue
            delta = bs_put_delta(spot_entry, K, T, sigma, RISK_FREE)
            table.append((K, px, sym, delta))
        if len(table) < 4:
            n_dropped_thin_chain += 1
            continue

        short_row = min(table, key=lambda r: abs(r[3] - cfg.short_delta))
        long_row = min(table, key=lambda r: abs(r[3] - cfg.long_delta))
        if short_row[0] <= long_row[0]:
            continue
        short_K, short_px, short_sym, short_d = short_row
        long_K, long_px, long_sym, long_d = long_row
        net_credit = short_px - long_px
        if net_credit <= 0:
            n_dropped_no_credit += 1
            continue

        # Exit at expiration — REAL close if present, intrinsic if not
        exit_target = expiration
        short_exit_info = fetch_contract_close(con, short_sym, effective_entry, exit_target)
        long_exit_info = fetch_contract_close(con, long_sym, effective_entry, exit_target)
        try:
            spot_exit = float(spot.loc[:exit_target].iloc[-1])
        except (KeyError, IndexError):
            spot_exit = spot_entry

        if short_exit_info is not None and short_exit_info[0] != effective_entry:
            short_exit = float(short_exit_info[1])
            short_src = "option_daily"
        else:
            short_exit = max(short_K - spot_exit, 0.0)
            short_src = "intrinsic"
        if long_exit_info is not None and long_exit_info[0] != effective_entry:
            long_exit = float(long_exit_info[1])
            long_src = "option_daily"
        else:
            long_exit = max(long_K - spot_exit, 0.0)
            long_src = "intrinsic"
        if short_src == "option_daily" and long_src == "option_daily":
            n_exits_real += 1
        else:
            n_exits_intrinsic += 1

        exit_net = short_exit - long_exit

        pnl_per_spread_raw = (net_credit - exit_net) * 100.0  # dollars/contract
        pnl_per_spread = pnl_per_spread_raw - cfg.slippage_per_spread

        max_loss_per_spread = max((short_K - long_K) - net_credit, 0.01) * 100.0
        n_contracts = (cfg.risk_per_trade * CAPITAL) / max_loss_per_spread
        pnl_pct = (pnl_per_spread * n_contracts) / CAPITAL

        trades.append(TradeRow(
            entry_date=snap,
            effective_entry_date=effective_entry,
            expiration=expiration,
            short_strike=float(short_K),
            long_strike=float(long_K),
            short_entry=float(short_px),
            long_entry=float(long_px),
            short_exit=float(short_exit),
            long_exit=float(long_exit),
            short_exit_source=short_src,
            long_exit_source=long_src,
            net_credit=float(net_credit),
            exit_net=float(exit_net),
            pnl_before_slippage=float(pnl_per_spread_raw),
            slippage=float(cfg.slippage_per_spread),
            pnl_per_spread=float(pnl_per_spread),
            pnl_pct_capital=float(pnl_pct),
            short_delta_estimate=float(short_d),
            long_delta_estimate=float(long_d),
            spot_entry=float(spot_entry),
            spot_exit=float(spot_exit),
            win=bool(pnl_per_spread > 0),
        ))

    return BacktestResult(
        config=cfg, trades=trades,
        n_attempted=n_attempted,
        n_dropped_no_expiration=n_dropped_no_expiration,
        n_dropped_thin_chain=n_dropped_thin_chain,
        n_dropped_no_credit=n_dropped_no_credit,
        n_exits_real=n_exits_real,
        n_exits_intrinsic=n_exits_intrinsic,
    )


# ── Metrics ────────────────────────────────────────────────────────────


def trade_metrics(trades: List[TradeRow], years: float) -> Dict[str, float]:
    if not trades:
        return dict(n_trades=0, win_rate=0.0, total_return=0.0, cagr=0.0,
                    sharpe_per_trade=0.0, avg_pnl_pct=0.0,
                    median_pnl_pct=0.0, worst_pnl_pct=0.0, max_dd_pct=0.0)
    pnls = np.array([t.pnl_pct_capital for t in trades], dtype=float)
    wins = int((pnls > 0).sum())
    eq = np.cumsum(pnls) + 1.0
    eq_factor = np.cumprod(1.0 + pnls)
    total_return = float(eq_factor[-1] - 1.0)
    cagr = float(eq_factor[-1] ** (1 / max(years, 1e-9)) - 1.0) if eq_factor[-1] > 0 else -1.0
    pk = np.maximum.accumulate(eq_factor)
    max_dd = float(((eq_factor - pk) / pk).min())
    mu = float(pnls.mean())
    sd = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    tpy = len(pnls) / max(years, 1e-9)
    sharpe = (mu / sd) * math.sqrt(max(tpy, 1.0)) if sd > 1e-12 else 0.0
    return dict(
        n_trades=int(len(pnls)),
        win_rate=float(wins / len(pnls)),
        total_return=total_return,
        cagr=cagr,
        sharpe_per_trade=sharpe,
        avg_pnl_pct=mu,
        median_pnl_pct=float(np.median(pnls)),
        worst_pnl_pct=float(pnls.min()),
        max_dd_pct=max_dd,
    )


def regime_breakdown(trades: List[TradeRow]) -> Dict[int, Dict[str, float]]:
    by_year: Dict[int, List[TradeRow]] = {}
    for t in trades:
        y = int(t.effective_entry_date[:4])
        by_year.setdefault(y, []).append(t)
    out: Dict[int, Dict[str, float]] = {}
    for y, ts in sorted(by_year.items()):
        out[y] = trade_metrics(ts, years=1.0)
    return out


# ── Walk-forward (expanding train) ────────────────────────────────────


def walk_forward_expanding(baseline_trades: List[TradeRow]) -> List[Dict]:
    """20 folds of 63-day OOS windows, expanding train start from 2020-01."""
    if not baseline_trades:
        return []
    df = pd.DataFrame([{
        "date": pd.Timestamp(t.effective_entry_date),
        "pnl_pct": t.pnl_pct_capital,
    } for t in baseline_trades]).sort_values("date")
    if df.empty:
        return []

    # Bucket trades by 63-day test windows starting from first_date + 252d.
    first = df["date"].min()
    results: List[Dict] = []
    for i in range(20):
        train_end = first + pd.Timedelta(days=252 + i * 63)
        test_end = train_end + pd.Timedelta(days=63)
        train_mask = df["date"] <= train_end
        test_mask = (df["date"] > train_end) & (df["date"] <= test_end)
        train_pnls = df.loc[train_mask, "pnl_pct"].values
        test_pnls = df.loc[test_mask, "pnl_pct"].values
        if len(test_pnls) == 0:
            results.append({
                "fold": i + 1,
                "train_end": str(train_end.date()),
                "test_end": str(test_end.date()),
                "n_train": int(len(train_pnls)),
                "n_test": 0,
                "test_pnl": 0.0,
                "test_win_rate": 0.0,
            })
            continue
        wins = int((test_pnls > 0).sum())
        results.append({
            "fold": i + 1,
            "train_end": str(train_end.date()),
            "test_end": str(test_end.date()),
            "n_train": int(len(train_pnls)),
            "n_test": int(len(test_pnls)),
            "test_pnl": float(test_pnls.sum()),
            "test_win_rate": float(wins / len(test_pnls)),
        })
    return results


# ── Report rendering ───────────────────────────────────────────────────


def _fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x*100:.{dp}f}%" if np.isfinite(x) else "—"


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(payload: Dict) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1280px;color:#111}
    h1{border-bottom:3px solid #5c2e00}
    h2{margin-top:2em;color:#5c2e00}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#5c2e00;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#5c2e00}
    .pill.bad{background:#c0392b}
    .pill.ok{background:#0a7d1f}
    """
    h = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2210 XLF/XLI Deep Validation</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2210 — XLF/XLI Credit Spread Deep Validation</h1>",
        "<p class='muted'>Six formal validation passes on the EXP-2160 "
        "XLF/XLI credit-spread result (per-trade Sharpe 11.3 / 6.0, 98% / "
        "97% WR) to determine whether the number is real alpha, a data "
        "artifact, or a methodology bug.</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault + Yahoo data only</span></p>",
    ]

    for tk in TICKERS:
        tk_data = payload["tickers"][tk]
        h.append(f"<h1>— {tk} —</h1>")

        # Coverage
        cov = tk_data["coverage"]
        h.append("<h2>Data coverage (IronVault)</h2>")
        h.append("<table><tr><th>Total contracts</th><th>Snapshot dates</th>"
                 "<th>Option_daily rows</th></tr>"
                 f"<tr><td>{cov['n_contracts']:,}</td>"
                 f"<td>{cov['n_snapshot_dates']:,}</td>"
                 f"<td>{cov.get('n_bars', '—')}</td></tr></table>")

        # Baseline
        base = tk_data["baseline"]
        h.append("<h2>1. Baseline reproduction (EXP-2160 parity)</h2>")
        h.append("<table><tr><th>n trades</th><th>Win rate</th><th>CAGR</th>"
                 "<th>Sharpe (per-trade)</th><th>Max DD</th><th>Avg P&L</th>"
                 "<th>Worst P&L</th></tr>"
                 f"<tr><td>{base['metrics']['n_trades']}</td>"
                 f"<td>{_fmt_pct(base['metrics']['win_rate'], 1)}</td>"
                 f"<td>{_fmt_pct(base['metrics']['cagr'])}</td>"
                 f"<td>{_fmt(base['metrics']['sharpe_per_trade'])}</td>"
                 f"<td class='neg'>{_fmt_pct(base['metrics']['max_dd_pct'])}</td>"
                 f"<td>{_fmt_pct(base['metrics']['avg_pnl_pct'], 3)}</td>"
                 f"<td class='neg'>{_fmt_pct(base['metrics']['worst_pnl_pct'], 3)}</td></tr></table>")
        diag = base["diagnostics"]
        h.append(f"<p class='muted'>Attempted {diag['n_attempted']} weekly snapshots · "
                 f"dropped {diag['n_dropped_no_expiration']} no-expiration · "
                 f"dropped {diag['n_dropped_thin_chain']} thin-chain · "
                 f"dropped {diag['n_dropped_no_credit']} no-credit · "
                 f"{diag['n_exits_real']} real exits / "
                 f"{diag['n_exits_intrinsic']} intrinsic exits</p>")

        # T+1 comparison
        h.append("<h2>2. T+1 entry pricing (look-ahead check)</h2>")
        t0 = base["metrics"]
        t1 = tk_data["t_plus_1"]["metrics"]
        h.append("<table><tr><th>Variant</th><th>n</th><th>WR</th><th>Sharpe</th>"
                 "<th>CAGR</th><th>Max DD</th></tr>"
                 f"<tr><td class='l'>T+0 baseline</td>"
                 f"<td>{t0['n_trades']}</td>"
                 f"<td>{_fmt_pct(t0['win_rate'],1)}</td>"
                 f"<td>{_fmt(t0['sharpe_per_trade'])}</td>"
                 f"<td>{_fmt_pct(t0['cagr'])}</td>"
                 f"<td class='neg'>{_fmt_pct(t0['max_dd_pct'])}</td></tr>"
                 f"<tr><td class='l'>T+1 entry</td>"
                 f"<td>{t1['n_trades']}</td>"
                 f"<td>{_fmt_pct(t1['win_rate'],1)}</td>"
                 f"<td>{_fmt(t1['sharpe_per_trade'])}</td>"
                 f"<td>{_fmt_pct(t1['cagr'])}</td>"
                 f"<td class='neg'>{_fmt_pct(t1['max_dd_pct'])}</td></tr></table>")
        delta_sharpe = t1["sharpe_per_trade"] - t0["sharpe_per_trade"]
        verdict = ("<span class='pill ok'>no look-ahead — delta &lt; 0.5</span>"
                   if abs(delta_sharpe) < 0.5
                   else f"<span class='pill bad'>look-ahead suspected — Δ Sharpe {delta_sharpe:+.2f}</span>")
        h.append(f"<p>{verdict}</p>")

        # Survivorship
        h.append("<h2>3. Survivorship / data-completeness audit</h2>")
        surv = tk_data["survivorship"]
        real_pct = surv["n_exits_real"] / max(surv["n_attempted"], 1)
        h.append("<table><tr><th>Attempted weekly snaps</th>"
                 "<th>Dropped no-exp</th><th>Dropped thin-chain</th>"
                 "<th>Dropped no-credit</th>"
                 "<th>Real exits (option_daily close)</th>"
                 "<th>Intrinsic exits (fallback)</th></tr>"
                 f"<tr><td>{surv['n_attempted']}</td>"
                 f"<td>{surv['n_dropped_no_expiration']}</td>"
                 f"<td>{surv['n_dropped_thin_chain']}</td>"
                 f"<td>{surv['n_dropped_no_credit']}</td>"
                 f"<td>{surv['n_exits_real']}</td>"
                 f"<td class='neg'>{surv['n_exits_intrinsic']}</td></tr></table>")
        pill_class = "ok" if real_pct > 0.5 else "bad"
        h.append(f"<p><span class='pill {pill_class}'>"
                 f"{real_pct*100:.0f}% of trades exit on a REAL option_daily close</span></p>")

        # Parameter sensitivity
        h.append("<h2>4. Parameter sensitivity (short Δ × target DTE)</h2>")
        sens = tk_data["sensitivity"]
        # Short delta rows, DTE cols
        deltas = sorted({r["short_delta"] for r in sens}, reverse=True)
        dtes = sorted({r["target_dte"] for r in sens})
        h.append("<h3>Sharpe (per-trade)</h3>")
        h.append("<table><tr><th>Δ \\ DTE</th>" + "".join(f"<th>{d}d</th>" for d in dtes) + "</tr>")
        for d in deltas:
            h.append(f"<tr><td class='l'><b>{d:+.2f}</b></td>")
            for dte in dtes:
                row = next((r for r in sens if r["short_delta"] == d and r["target_dte"] == dte), None)
                if row and row["n_trades"] > 0:
                    s = row["sharpe_per_trade"]
                    h.append(f"<td>{_fmt(s)}</td>")
                else:
                    h.append("<td class='muted'>—</td>")
            h.append("</tr>")
        h.append("</table>")
        h.append("<h3>Win rate</h3>")
        h.append("<table><tr><th>Δ \\ DTE</th>" + "".join(f"<th>{d}d</th>" for d in dtes) + "</tr>")
        for d in deltas:
            h.append(f"<tr><td class='l'><b>{d:+.2f}</b></td>")
            for dte in dtes:
                row = next((r for r in sens if r["short_delta"] == d and r["target_dte"] == dte), None)
                if row and row["n_trades"] > 0:
                    h.append(f"<td>{_fmt_pct(row['win_rate'], 1)}</td>")
                else:
                    h.append("<td class='muted'>—</td>")
            h.append("</tr>")
        h.append("</table>")

        # Regime
        h.append("<h2>5. Regime analysis (yearly)</h2>")
        regime = tk_data["regime"]
        years = sorted(int(y) for y in regime.keys())
        h.append("<table><tr><th>Year</th><th>n trades</th>"
                 "<th>Win rate</th><th>Total return</th><th>Avg P&L</th>"
                 "<th>Worst trade</th></tr>")
        for y in years:
            r = regime[y]
            h.append(
                f"<tr><td class='l'><b>{y}</b></td>"
                f"<td>{r['n_trades']}</td>"
                f"<td>{_fmt_pct(r['win_rate'], 1)}</td>"
                f"<td class='{ 'pos' if r['total_return']>0 else 'neg' }'>{_fmt_pct(r['total_return'])}</td>"
                f"<td>{_fmt_pct(r['avg_pnl_pct'], 3)}</td>"
                f"<td class='neg'>{_fmt_pct(r['worst_pnl_pct'], 3)}</td></tr>"
            )
        h.append("</table>")
        if 2022 in regime and regime[2022]["n_trades"] > 0:
            r22 = regime[2022]
            survives = r22["total_return"] > 0
            pill_class = "ok" if survives else "bad"
            h.append(f"<p><span class='pill {pill_class}'>"
                     f"2022 bear market: {_fmt_pct(r22['total_return'])} "
                     f"({r22['n_trades']} trades, WR {_fmt_pct(r22['win_rate'], 1)})</span></p>")

        # Slippage
        h.append("<h2>6. Slippage / cost sensitivity</h2>")
        slip = tk_data["slippage"]
        h.append("<table><tr><th>$ per spread (round-trip)</th>"
                 "<th>n trades</th><th>WR</th><th>CAGR</th>"
                 "<th>Sharpe (per-trade)</th></tr>")
        for s_cost in sorted(slip.keys(), key=float):
            r = slip[s_cost]
            h.append(
                f"<tr><td class='l'>${s_cost}</td>"
                f"<td>{r['n_trades']}</td>"
                f"<td>{_fmt_pct(r['win_rate'], 1)}</td>"
                f"<td class='{ 'pos' if r['cagr']>0 else 'neg' }'>{_fmt_pct(r['cagr'])}</td>"
                f"<td>{_fmt(r['sharpe_per_trade'])}</td></tr>"
            )
        h.append("</table>")

        # Walk-forward
        h.append("<h2>Walk-forward (expanding window, 63d test)</h2>")
        wf = tk_data["walk_forward"]
        h.append("<table><tr><th>Fold</th><th>Train end</th><th>Test end</th>"
                 "<th>n train</th><th>n test</th><th>Test P&L</th>"
                 "<th>Test WR</th></tr>")
        for f in wf:
            cls = "pos" if f["test_pnl"] > 0 else ("neg" if f["test_pnl"] < 0 else "")
            h.append(
                f"<tr><td>{f['fold']}</td>"
                f"<td class='l'>{f['train_end']}</td>"
                f"<td class='l'>{f['test_end']}</td>"
                f"<td>{f['n_train']}</td>"
                f"<td>{f['n_test']}</td>"
                f"<td class='{cls}'>{_fmt_pct(f['test_pnl'])}</td>"
                f"<td>{_fmt_pct(f['test_win_rate'], 1)}</td></tr>"
            )
        h.append("</table>")

    # Verdict
    h.append("<h1>Verdict</h1>")
    h.append(payload["verdict_html"])

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))

    try:
        payload: Dict = {"tickers": {}}

        for tk in TICKERS:
            print(f"\n[exp2210] ===== {tk} =====", flush=True)
            cov = coverage_stats(con, tk)
            n_bars = con.execute("""
                SELECT COUNT(*) FROM option_daily d
                JOIN option_contracts c USING(contract_symbol)
                WHERE c.ticker=?
            """, (tk,)).fetchone()[0]
            cov["n_bars"] = int(n_bars)
            print(f"[exp2210] coverage: {cov}")

            spot = fetch_yahoo_close(tk)
            all_dates = list_snapshot_dates(con, tk)
            all_dates = [d for d in all_dates if START <= d <= END]
            years = (datetime.strptime(END, "%Y-%m-%d")
                     - datetime.strptime(START, "%Y-%m-%d")).days / 365.25

            # 1. Baseline
            print(f"[exp2210/{tk}] baseline…", flush=True)
            base_cfg = CSConfig(ticker=tk, label="baseline")
            base_res = run_cs_config(con, spot, all_dates, base_cfg)
            base_m = trade_metrics(base_res.trades, years)
            print(f"[exp2210/{tk}]   baseline: n={base_m['n_trades']} "
                  f"wr={base_m['win_rate']*100:.1f}% Sh={base_m['sharpe_per_trade']:.2f} "
                  f"CAGR={base_m['cagr']*100:.2f}%")

            # 2. T+1 entry
            print(f"[exp2210/{tk}] T+1 entry…", flush=True)
            t1_cfg = CSConfig(ticker=tk, label="T+1", entry_offset_days=1)
            t1_res = run_cs_config(con, spot, all_dates, t1_cfg)
            t1_m = trade_metrics(t1_res.trades, years)
            print(f"[exp2210/{tk}]   T+1:      n={t1_m['n_trades']} "
                  f"wr={t1_m['win_rate']*100:.1f}% Sh={t1_m['sharpe_per_trade']:.2f} "
                  f"CAGR={t1_m['cagr']*100:.2f}%")

            # 3. Survivorship: already in base_res.diagnostics
            survivorship = {
                "n_attempted": base_res.n_attempted,
                "n_dropped_no_expiration": base_res.n_dropped_no_expiration,
                "n_dropped_thin_chain": base_res.n_dropped_thin_chain,
                "n_dropped_no_credit": base_res.n_dropped_no_credit,
                "n_exits_real": base_res.n_exits_real,
                "n_exits_intrinsic": base_res.n_exits_intrinsic,
            }

            # 4. Parameter sensitivity
            print(f"[exp2210/{tk}] parameter sensitivity…", flush=True)
            sens_rows: List[Dict] = []
            for d in (-0.20, -0.25, -0.30, -0.35, -0.40):
                for dte in (14, 21, 30, 45):
                    cfg = CSConfig(
                        ticker=tk,
                        short_delta=d,
                        long_delta=d + 0.15,  # keep width-of-Δ = 0.15
                        target_dte=dte,
                        min_dte=min(dte - 3, 7),
                        label=f"Δ{d}_DTE{dte}",
                    )
                    res = run_cs_config(con, spot, all_dates, cfg)
                    m = trade_metrics(res.trades, years)
                    sens_rows.append({
                        "short_delta": d,
                        "long_delta": round(d + 0.15, 2),
                        "target_dte": dte,
                        "n_trades": m["n_trades"],
                        "win_rate": m["win_rate"],
                        "cagr": m["cagr"],
                        "sharpe_per_trade": m["sharpe_per_trade"],
                    })
            print(f"[exp2210/{tk}]   {len(sens_rows)} sensitivity runs")

            # 5. Regime (yearly) — reuse baseline trades
            regime = regime_breakdown(base_res.trades)

            # 6. Slippage
            print(f"[exp2210/{tk}] slippage sweep…", flush=True)
            slip: Dict[str, Dict[str, float]] = {}
            for s_cost in (0.0, 10.0, 25.0, 50.0):
                cfg = CSConfig(ticker=tk, slippage_per_spread=s_cost,
                               label=f"slip_{s_cost}")
                res = run_cs_config(con, spot, all_dates, cfg)
                m = trade_metrics(res.trades, years)
                slip[str(int(s_cost))] = m
                print(f"[exp2210/{tk}]   slip=${s_cost}: CAGR={m['cagr']*100:.2f}%  "
                      f"Sh={m['sharpe_per_trade']:.2f}  wr={m['win_rate']*100:.1f}%")

            # Walk-forward on baseline trades
            wf = walk_forward_expanding(base_res.trades)

            payload["tickers"][tk] = {
                "coverage": cov,
                "baseline": {
                    "metrics": base_m,
                    "diagnostics": {
                        "n_attempted": base_res.n_attempted,
                        "n_dropped_no_expiration": base_res.n_dropped_no_expiration,
                        "n_dropped_thin_chain": base_res.n_dropped_thin_chain,
                        "n_dropped_no_credit": base_res.n_dropped_no_credit,
                        "n_exits_real": base_res.n_exits_real,
                        "n_exits_intrinsic": base_res.n_exits_intrinsic,
                    },
                },
                "t_plus_1": {"metrics": t1_m},
                "survivorship": survivorship,
                "sensitivity": sens_rows,
                "regime": regime,
                "slippage": slip,
                "walk_forward": wf,
            }
    finally:
        con.close()

    # Verdict — use data to form an honest summary
    verdict_parts = ["<ul>"]
    for tk in TICKERS:
        d = payload["tickers"][tk]
        base_m = d["baseline"]["metrics"]
        t1_m = d["t_plus_1"]["metrics"]
        surv = d["survivorship"]
        real_pct = surv["n_exits_real"] / max(surv["n_attempted"], 1)
        lookahead_delta = t1_m["sharpe_per_trade"] - base_m["sharpe_per_trade"]
        slip_50 = d["slippage"]["50"]
        r2022 = d["regime"].get(2022)

        issues = []
        if real_pct < 0.50:
            issues.append(
                f"<b>Survivorship flag:</b> only {real_pct*100:.0f}% of trades "
                "exit on a real option_daily close — the rest fall back to "
                "intrinsic-vs-spot, which systematically inflates the win "
                "rate because real bid-ask noise at expiration is erased."
            )
        if abs(lookahead_delta) > 1.0:
            issues.append(
                f"<b>T+1 look-ahead sensitivity:</b> Sharpe drops by "
                f"{abs(lookahead_delta):.2f} when entries move from T+0 to T+1 "
                "— the baseline may have been benefiting from close-at-signal "
                "pricing that a production strategy cannot actually achieve."
            )
        if slip_50["sharpe_per_trade"] < 2.0:
            issues.append(
                f"<b>Slippage fragility:</b> at $50/spread round-trip "
                f"(realistic for thin XLF/XLI chains), Sharpe drops to "
                f"{slip_50['sharpe_per_trade']:.2f} — below the 2.0 target."
            )
        if r2022 and r2022["total_return"] < 0:
            issues.append(
                f"<b>2022 bear-market failure:</b> total return "
                f"{r2022['total_return']*100:+.2f}% across {r2022['n_trades']} "
                "trades — the strategy did NOT survive the only real stress "
                "year in the 2020-2025 window."
            )
        if not issues:
            issues.append("No obvious methodology or data flags detected.")

        verdict_parts.append(f"<li><h3>{tk}</h3><ul>")
        for iss in issues:
            verdict_parts.append(f"<li>{iss}</li>")
        verdict_parts.append("</ul></li>")
    verdict_parts.append("</ul>")
    payload["verdict_html"] = "".join(verdict_parts)

    html = render_html(payload)
    REPORT_HTML.write_text(html)
    print(f"\n[exp2210] wrote {REPORT_HTML}")

    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp2210] wrote {REPORT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
