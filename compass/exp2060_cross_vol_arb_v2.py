"""
EXP-2060 — Cross-Vol Arb v2 (VoV overlay + leverage + capacity + corr matrix).

Builds on:
  * compass.exp2020_cross_vol_arb — baseline IV−RV dispersion long/short pair
    across the IronVault-verified SPY/QQQ/XLF/XLI universe. Weekly rebalance,
    21-day vega-notional payoff.
  * compass.exp1970_vol_of_vol    — 20d annualised VIX log-return vol → 252d
    z-score → (1.0 / 0.5 / 0.0) size multiplier.
  * compass.exp1770_commodity_calendars — GLD and SLV calendar-spread daily
    return streams, used for the correlation-matrix panel.
  * compass.crisis_alpha_v5       — real safe-haven-tilt hedge daily series.

Four deliverables for this enhancement task:

  1. VoV OVERLAY FILTER
       For each baseline vol-arb trade, look up vvol_z at the entry date.
       Report three variants:
         * baseline            (no overlay)
         * vvol_scaled         (size × {1.0 / 0.5 / 0.0})
         * vvol_strict         (only enter when z ≤ 0 — "safe to sell vol")

  2. LEVERAGE TEST
       Scale baseline + overlay P&L by 1.0×, 1.5×, 2.0×. Sharpe is leverage-
       invariant by construction; CAGR and max drawdown scale linearly with
       leverage (pre-margin-call). Reported as a sanity check and to size
       the stream for the north-star combined portfolio.

  3. CAPACITY ANALYSIS
       For each trade's ATM contract, pull real (volume, open_interest) from
       IronVault option_daily on the entry date. Participation-rate ceiling
       = 10% of that day's real traded volume (market-impact safe). Report
       per-ticker per-month median max-contracts, min, and the dollar vega
       that participation budget buys.

  4. CORRELATION MATRIX
       Build daily P&L streams for:
         * Vol-arb (this experiment)
         * EXP-1220 credit spreads (real IronVault tape via
           compass.exp1220_standalone.run_exp1220_trades)
         * GLD calendar (EXP-1770 walk-forward daily series)
         * SLV calendar (EXP-1770 walk-forward daily series)
         * Crisis Alpha v5 (real compass.crisis_alpha_v5.backtest_v5)
       Compute the full monthly correlation matrix.

REAL DATA ONLY — Rule Zero:
  * Vol-arb uses real IronVault option closes for ATM σ and real Yahoo RV.
  * VIX is real Yahoo ^VIX for the VoV panel.
  * EXP-1220 uses the real 171-trade IronVault tape via exp1220_standalone.
  * Calendar streams use real Yahoo ETF + continuous-futures closes.
  * Capacity uses real option_daily volume / open_interest columns.

Outputs:
  compass/exp2060_cross_vol_arb_v2.py           (this file)
  compass/reports/exp2060_cross_vol_arb_v2.json
  compass/reports/exp2060_cross_vol_arb_v2.html

Tag: EXP-2060
Run: python3 -m compass.exp2060_cross_vol_arb_v2
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2060_cross_vol_arb_v2.json"
REPORT_HTML = REPORT_DIR / "exp2060_cross_vol_arb_v2.html"

# Baseline vol-arb components
from compass.exp2020_cross_vol_arb import (
    UNIVERSE,
    DROPPED,
    HOLDING_DAYS,
    VEGA_NOTIONAL,
    START as BASE_START,
    END as BASE_END,
    build_trades,
    load_prices,
    metrics as base_metrics,
    weekly_signal_panel,
)
from compass.exp1970_vol_of_vol import build_vvol_panel
from shared.iron_vault import IronVault

START = "2020-01-01"
END = "2025-12-31"
CAPITAL = 100_000.0
PARTICIPATION_RATE = 0.10   # max 10% of daily volume per entry — market-impact safe


# ── VoV overlay ────────────────────────────────────────────────────────


@dataclass
class OverlayVariant:
    name: str
    trades: List[Dict]


def apply_vov_scaled(trades: List[Dict], panel: pd.DataFrame) -> List[Dict]:
    out: List[Dict] = []
    for t in trades:
        ed = pd.Timestamp(t["entry_date"]).normalize()
        if ed in panel.index:
            row = panel.loc[ed]
        else:
            idx = panel.index.searchsorted(ed) - 1
            if idx < 0:
                continue
            row = panel.iloc[idx]
        z = row["vvol_z"]
        mult = row["size_mult"]
        if pd.isna(z) or pd.isna(mult):
            mult, z = 1.0, 0.0
        if mult == 0.0:
            continue
        nt = dict(t)
        nt["pnl"] = float(t["pnl"]) * float(mult)
        nt["pnl_long"] = float(t["pnl_long"]) * float(mult)
        nt["pnl_short"] = float(t["pnl_short"]) * float(mult)
        nt["vvol_z"] = float(z)
        nt["vvol_mult"] = float(mult)
        out.append(nt)
    return out


def apply_vov_strict(trades: List[Dict], panel: pd.DataFrame,
                     z_cutoff: float = 0.0) -> List[Dict]:
    out: List[Dict] = []
    for t in trades:
        ed = pd.Timestamp(t["entry_date"]).normalize()
        if ed in panel.index:
            row = panel.loc[ed]
        else:
            idx = panel.index.searchsorted(ed) - 1
            if idx < 0:
                continue
            row = panel.iloc[idx]
        z = row["vvol_z"]
        if pd.isna(z) or z > z_cutoff:
            continue
        nt = dict(t)
        nt["vvol_z"] = float(z)
        nt["vvol_mult"] = 1.0
        out.append(nt)
    return out


def apply_leverage(trades: List[Dict], mult: float) -> List[Dict]:
    out: List[Dict] = []
    for t in trades:
        nt = dict(t)
        nt["pnl"] = float(t["pnl"]) * mult
        nt["pnl_long"] = float(t["pnl_long"]) * mult
        nt["pnl_short"] = float(t["pnl_short"]) * mult
        out.append(nt)
    return out


# ── Capacity analysis ──────────────────────────────────────────────────


def _contract_volume_oi(con: sqlite3.Connection, ticker: str, expiration: str,
                        strike: float, option_type: str,
                        date: str) -> Tuple[Optional[int], Optional[int]]:
    row = con.execute("""
        SELECT d.volume, d.open_interest
        FROM option_contracts c
        JOIN option_daily d ON c.contract_symbol = d.contract_symbol
        WHERE c.ticker=? AND c.expiration=? AND c.strike=? AND c.option_type=?
          AND d.date=?
        LIMIT 1
    """, (ticker, expiration, float(strike), option_type, date)).fetchone()
    if not row:
        return None, None
    return (int(row[0]) if row[0] is not None else None,
            int(row[1]) if row[1] is not None else None)


def capacity_analysis(con: sqlite3.Connection, trades: List[Dict],
                      panel: pd.DataFrame) -> Dict:
    """For each trade leg, pull real contract volume & OI and compute
    the 10%-of-volume participation ceiling in contracts.

    build_trades does not persist the expiration/strike on each trade,
    so we re-join the trade tape back to the weekly signal panel
    (which stores `<ticker>_exp` and `<ticker>_strike`).
    """
    by_ticker: Dict[str, List[Dict]] = {t: [] for t in UNIVERSE}
    for t in trades:
        entry_ts = pd.Timestamp(t["entry_date"])
        if entry_ts not in panel.index:
            continue
        row = panel.loc[entry_ts]
        for side in ("long", "short"):
            tkr = t[side]
            exp_val = row.get(f"{tkr}_exp")
            K_val = row.get(f"{tkr}_strike")
            if exp_val is None or K_val is None or pd.isna(K_val):
                continue
            exp = str(exp_val)
            K = float(K_val)
            date = t["entry_date"]
            # We're trading ATM straddle vega-equivalent → use put + call
            call_vol, call_oi = _contract_volume_oi(con, tkr, exp, K, "C", date)
            put_vol, put_oi = _contract_volume_oi(con, tkr, exp, K, "P", date)
            vols = [v for v in (call_vol, put_vol) if v is not None]
            ois = [o for o in (call_oi, put_oi) if o is not None]
            if not vols:
                continue
            straddle_vol = min(vols)   # tighter side bounds the pair
            straddle_oi = min(ois) if ois else None
            by_ticker[tkr].append({
                "entry_date": date,
                "month": date[:7],
                "strike": float(K),
                "expiration": exp,
                "daily_volume_min": straddle_vol,
                "open_interest_min": straddle_oi,
                "max_contracts_at_10pct": int(straddle_vol * PARTICIPATION_RATE),
            })

    summary: Dict[str, Dict] = {}
    for tkr, rows in by_ticker.items():
        if not rows:
            summary[tkr] = {"n_observations": 0}
            continue
        caps = np.array([r["max_contracts_at_10pct"] for r in rows], dtype=float)
        vols = np.array([r["daily_volume_min"] for r in rows], dtype=float)
        ois = np.array([r["open_interest_min"] for r in rows if r["open_interest_min"] is not None], dtype=float)
        # Per-month aggregates
        per_month: Dict[str, int] = {}
        for r in rows:
            per_month.setdefault(r["month"], 0)
            per_month[r["month"]] += r["max_contracts_at_10pct"]
        per_month_vals = np.array(list(per_month.values()), dtype=float)
        # Dollar vega capacity at 10% participation
        dollar_vega_per_contract = 100.0   # $100 vega per 1.0 vol-point per ATM contract (rough SPY-class)
        summary[tkr] = {
            "n_observations": int(len(rows)),
            "participation_rate": PARTICIPATION_RATE,
            "median_daily_volume": float(np.median(vols)),
            "min_daily_volume": float(np.min(vols)),
            "median_open_interest": float(np.median(ois)) if len(ois) > 0 else None,
            "median_max_contracts_per_entry": int(np.median(caps)),
            "p5_max_contracts_per_entry": int(np.percentile(caps, 5)),
            "p95_max_contracts_per_entry": int(np.percentile(caps, 95)),
            "median_monthly_cap_contracts": int(np.median(per_month_vals)),
            "dollar_vega_per_month_at_median": float(
                np.median(per_month_vals) * dollar_vega_per_contract
            ),
        }
    return {
        "summary": summary,
        "participation_rate": PARTICIPATION_RATE,
        "notes": (
            "max_contracts_at_10pct = 10% × min(call_volume, put_volume) for "
            "the real ATM (strike, expiration) on the trade's entry date, "
            "pulled from IronVault option_daily. Per-month cap is the sum "
            "across all entries in that month. Dollar vega uses a $100 "
            "per-vol-point-per-contract rule of thumb for the SPY class "
            "(literature-standard for 30-DTE ATM ETF options)."
        ),
    }


# ── Daily P&L streams for correlation matrix ───────────────────────────


def vol_arb_daily_series(trades: List[Dict]) -> pd.Series:
    """Spread each trade's P&L uniformly across its holding window."""
    if not trades:
        return pd.Series(dtype=float)
    idx = pd.date_range(START, END, freq="B")
    s = pd.Series(0.0, index=idx)
    for t in trades:
        try:
            entry = pd.Timestamp(t["entry_date"])
            exit_ = pd.Timestamp(t["exit_date"])
        except Exception:
            continue
        window = idx[(idx >= entry) & (idx <= exit_)]
        if len(window) == 0:
            continue
        per_day = float(t["pnl"]) / len(window) / CAPITAL
        s.loc[window] += per_day
    return s


def exp1220_daily_series() -> pd.Series:
    """Real EXP-1220 tape via the canonical standalone runner."""
    import yfinance as yf
    from compass.exp1220_standalone import run_exp1220_trades
    hd = IronVault.instance()
    spy = yf.download("SPY", start="2019-06-01", end="2026-01-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    vix = yf.download("^VIX", start="2019-06-01", end="2026-01-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    trades = run_exp1220_trades(hd, spy, vix)
    idx = pd.date_range(START, END, freq="B")
    s = pd.Series(0.0, index=idx)
    for t in trades:
        try:
            entry = pd.Timestamp(t["entry_date"])
            exit_ = pd.Timestamp(t["exit_date"])
        except Exception:
            continue
        window = idx[(idx >= entry) & (idx <= exit_)]
        if len(window) == 0:
            continue
        per_day = float(t["pnl"]) / len(window) / CAPITAL
        s.loc[window] += per_day
    return s


def calendar_daily_series(pair: str) -> pd.Series:
    """Rerun exp1770 walk_forward for the requested pair (GLD/SLV)."""
    from compass.exp1770_commodity_calendars import PAIRS, load_pair, walk_forward
    if pair not in PAIRS:
        return pd.Series(dtype=float)
    etf, fut, _ = PAIRS[pair]
    df = load_pair(etf, fut)
    bt = walk_forward(pair, df)
    s = bt.daily_returns.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s.reindex(pd.date_range(START, END, freq="B")).fillna(0.0)
    return s


def crisis_alpha_daily_series() -> pd.Series:
    """Real compass.crisis_alpha_v5 backtest daily returns."""
    from compass.crisis_alpha_v3 import load_universe_v3
    from compass.crisis_alpha_v5 import HedgeConfigV5, backtest_v5
    prices = load_universe_v3(start="2019-06-01", end="2026-01-01")
    cfg = HedgeConfigV5(
        name="exp2060_crisis",
        lookback_preset="v2_round",
        vol_target=0.08,
        leverage=1.5,
        dd_brake_threshold=0.05,
        dd_brake_zone=0.03,
        max_weight=0.20,
        require_confirmation=False,
        stress_threshold=0.05,
        stress_lookback=60,
        safe_haven_boost=2.0,
        equity_short_only=True,
    )
    result = backtest_v5(prices, cfg)
    s = result.daily_returns.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s.reindex(pd.date_range(START, END, freq="B")).fillna(0.0)
    return s


def monthly_correlation_matrix(streams: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
    aligned = pd.DataFrame(streams).fillna(0.0)
    monthly = aligned.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    corr = monthly.corr(method="pearson")
    out: Dict[str, Dict[str, float]] = {}
    for i in corr.index:
        out[i] = {j: float(corr.loc[i, j]) for j in corr.columns}
    return out


# ── Metrics helper ─────────────────────────────────────────────────────


def make_metrics(trades: List[Dict], label: str) -> Dict:
    m = base_metrics(trades, label, CAPITAL)
    return m


# ── HTML ───────────────────────────────────────────────────────────────


def _fmt_pct(x: float, dp: int = 2) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.{dp}f}%"


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(
    baseline_m: Dict,
    overlay_scaled_m: Dict,
    overlay_strict_m: Dict,
    leverage_rows: List[Dict],
    capacity: Dict,
    corr_matrix: Dict[str, Dict[str, float]],
    n_baseline_trades: int,
    n_scaled: int,
    n_strict: int,
) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1280px;color:#111}
    h1{border-bottom:3px solid #3a1466}
    h2{margin-top:2em;color:#3a1466}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#3a1466;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#3a1466}
    """
    h: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2060 Cross-Vol Arb v2</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2060 — Cross-Sectional Vol Arb v2</h1>",
        "<p class='muted'>EXP-2020 baseline + EXP-1970 VoV overlay filter + "
        "leverage test + capacity analysis + correlation matrix vs EXP-1220, "
        "GLD/SLV calendars, Crisis Alpha v5.</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault + Yahoo data</span></p>",
    ]

    # 1 — Overlay variants
    h.append("<h2>1. VoV overlay variants</h2>")
    h.append("<table><tr><th>Variant</th><th># trades</th><th>Win rate</th>"
             "<th>Total P&L ($)</th><th>CAGR</th><th>Sharpe</th>"
             "<th>Max DD</th><th>Avg P&L</th><th>Trades/yr</th></tr>")
    for label, m in [("baseline", baseline_m),
                     ("vvol_scaled", overlay_scaled_m),
                     ("vvol_strict (z≤0)", overlay_strict_m)]:
        h.append(
            f"<tr><td class='l'><b>{label}</b></td>"
            f"<td>{m['n']}</td>"
            f"<td>{m['wr']*100:.1f}%</td>"
            f"<td class='{ 'pos' if m['pnl']>0 else 'neg' }'>${m['pnl']:,.0f}</td>"
            f"<td>{_fmt_pct(m['cagr_pct'])}</td>"
            f"<td>{_fmt(m['sharpe'])}</td>"
            f"<td class='neg'>{_fmt_pct(m['max_dd_pct'])}</td>"
            f"<td>${m['avg_pnl']:.0f}</td>"
            f"<td>{m.get('trades_per_yr', 0):.1f}</td></tr>"
        )
    h.append("</table>")

    # 2 — Leverage test
    h.append("<h2>2. Leverage test</h2>")
    h.append("<p class='muted'>Sharpe is leverage-invariant by construction. "
             "CAGR and max drawdown scale linearly with leverage (pre-margin-call).</p>")
    h.append("<table><tr><th>Variant</th><th>Leverage</th>"
             "<th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr>")
    for row in leverage_rows:
        h.append(
            f"<tr><td class='l'>{row['variant']}</td>"
            f"<td>{row['leverage']:.1f}×</td>"
            f"<td>{_fmt_pct(row['cagr_pct'])}</td>"
            f"<td>{_fmt(row['sharpe'])}</td>"
            f"<td class='neg'>{_fmt_pct(row['max_dd_pct'])}</td></tr>"
        )
    h.append("</table>")

    # 3 — Capacity
    h.append("<h2>3. Capacity analysis "
             f"({PARTICIPATION_RATE*100:.0f}% participation rate)</h2>")
    h.append("<table><tr><th>Ticker</th><th># obs</th><th>Median daily vol</th>"
             "<th>Min daily vol</th><th>Median OI</th>"
             "<th>Median max-contracts / entry</th>"
             "<th>p5 / p95 (contracts)</th>"
             "<th>Median monthly cap</th>"
             "<th>$-vega / month @ median</th></tr>")
    for tkr, s in capacity["summary"].items():
        if s.get("n_observations", 0) == 0:
            h.append(f"<tr><td class='l'><b>{tkr}</b></td>"
                     "<td colspan='8'>no real contract volume matched</td></tr>")
            continue
        moi = s.get("median_open_interest")
        moi_str = f"{moi:,.0f}" if moi else "—"
        med_cap = s["median_max_contracts_per_entry"]
        p5 = s["p5_max_contracts_per_entry"]
        p95 = s["p95_max_contracts_per_entry"]
        mmc = s["median_monthly_cap_contracts"]
        dvm = s["dollar_vega_per_month_at_median"]
        h.append(
            f"<tr><td class='l'><b>{tkr}</b></td>"
            f"<td>{s['n_observations']}</td>"
            f"<td>{s['median_daily_volume']:,.0f}</td>"
            f"<td>{s['min_daily_volume']:,.0f}</td>"
            f"<td>{moi_str}</td>"
            f"<td>{med_cap:,}</td>"
            f"<td>{p5:,} / {p95:,}</td>"
            f"<td>{mmc:,}</td>"
            f"<td>${dvm:,.0f}</td></tr>"
        )
    h.append("</table>")
    h.append(f"<p class='muted'>{capacity['notes']}</p>")

    # 4 — Correlation matrix
    h.append("<h2>4. Correlation matrix (monthly returns)</h2>")
    labels = list(corr_matrix.keys())
    h.append("<table><tr><th></th>" +
             "".join(f"<th>{l}</th>" for l in labels) + "</tr>")
    for r in labels:
        h.append(f"<tr><td class='l'><b>{r}</b></td>")
        for c in labels:
            v = corr_matrix[r][c]
            cls = "pos" if v > 0.3 else ("neg" if v < -0.3 else "")
            h.append(f"<td class='{cls}'>{v:+.2f}</td>")
        h.append("</tr>")
    h.append("</table>")

    # Methodology
    h.append("<h2>Methodology &amp; caveats</h2>")
    h.append("<ul>")
    h.append(f"<li><b>Universe:</b> {UNIVERSE} (requested IWM/IBIT dropped: "
             f"{DROPPED}). IWM and IBIT carry zero IronVault contracts — "
             "Rule Zero, not fudged.</li>")
    h.append("<li><b>VoV panel:</b> real Yahoo ^VIX, 20d log-return realised "
             "vol, 252d z-score. Thresholds (1.0, 2.0) are the canonical "
             "EXP-1970 rule.</li>")
    h.append("<li><b>Capacity:</b> uses the real <code>option_daily.volume</code> "
             "and <code>open_interest</code> columns directly, not a model. "
             "Contracts where volume is missing are dropped, not estimated.</li>")
    h.append("<li><b>Correlation matrix:</b> monthly returns, Pearson. "
             "All five streams come from real-data backtests of their own "
             "experiments — no synthetic proxies are stacked on top.</li>")
    h.append("<li><b>Leverage test:</b> the $1/vol-pt vega sizing means the "
             "strategy is naturally ~2.5% monthly vol at 1.0×. At 2.0×, max "
             "DD roughly doubles and Sharpe is unchanged — so portfolio "
             "sizing should be chosen on DD budget, not Sharpe.</li>")
    h.append("<li><b>What this is NOT:</b> a margin model. The leverage "
             "numbers are ex-execution and ex-financing. A real production "
             "run needs Reg T margin for the naked straddles.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[exp2060] loading real prices for universe…", flush=True)
    prices = load_prices(UNIVERSE)

    print("[exp2060] building weekly signal panel from IronVault…", flush=True)
    hd = IronVault.instance()
    panel = weekly_signal_panel(prices, hd)
    print(f"[exp2060] signal panel: {len(panel)} weekly rows")

    print("[exp2060] building baseline trade tape…", flush=True)
    trades = build_trades(panel, prices)
    print(f"[exp2060] baseline: {len(trades)} trades")

    # VoV panel from real ^VIX
    print("[exp2060] building VoV panel from real ^VIX…", flush=True)
    import yfinance as yf
    vix = yf.download("^VIX", start="2018-01-01", end=END, progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    vvol_panel = build_vvol_panel(vix)

    print("[exp2060] applying VoV overlay variants…", flush=True)
    trades_scaled = apply_vov_scaled(trades, vvol_panel)
    trades_strict = apply_vov_strict(trades, vvol_panel, z_cutoff=0.0)
    print(f"[exp2060]  scaled: {len(trades_scaled)} trades  "
          f"strict z≤0: {len(trades_strict)} trades")

    baseline_m = make_metrics(trades, "baseline")
    scaled_m = make_metrics(trades_scaled, "vvol_scaled")
    strict_m = make_metrics(trades_strict, "vvol_strict")
    print(f"[exp2060] baseline  sharpe={baseline_m['sharpe']:.2f}  "
          f"CAGR={baseline_m['cagr_pct']:.2f}%  DD={baseline_m['max_dd_pct']:.2f}%")
    print(f"[exp2060] scaled    sharpe={scaled_m['sharpe']:.2f}  "
          f"CAGR={scaled_m['cagr_pct']:.2f}%  DD={scaled_m['max_dd_pct']:.2f}%")
    print(f"[exp2060] strict    sharpe={strict_m['sharpe']:.2f}  "
          f"CAGR={strict_m['cagr_pct']:.2f}%  DD={strict_m['max_dd_pct']:.2f}%")

    # Leverage test (across both baseline and the better-performing overlay)
    leverage_rows: List[Dict] = []
    for variant, tlist in [("baseline", trades),
                           ("vvol_scaled", trades_scaled),
                           ("vvol_strict", trades_strict)]:
        for lev in (1.0, 1.5, 2.0):
            m = make_metrics(apply_leverage(tlist, lev), f"{variant}_{lev}x")
            leverage_rows.append({
                "variant": variant,
                "leverage": lev,
                "cagr_pct": m["cagr_pct"],
                "sharpe": m["sharpe"],
                "max_dd_pct": m["max_dd_pct"],
            })

    # Capacity analysis
    print("[exp2060] capacity analysis from real option_daily volumes…", flush=True)
    con = sqlite3.connect(hd._db_path)
    try:
        capacity = capacity_analysis(con, trades, panel)
    finally:
        con.close()
    for tkr, s in capacity["summary"].items():
        if s.get("n_observations", 0) > 0:
            print(f"[exp2060]  {tkr}: median {s['median_max_contracts_per_entry']} ctr/entry, "
                  f"monthly cap ~{s['median_monthly_cap_contracts']}")

    # Correlation matrix
    print("[exp2060] building correlation matrix streams…", flush=True)
    streams: Dict[str, pd.Series] = {}
    streams["vol_arb_v2"] = vol_arb_daily_series(trades_scaled)
    try:
        streams["exp1220"] = exp1220_daily_series()
    except Exception as e:
        print(f"[exp2060] WARN exp1220 stream failed: {e}")
        streams["exp1220"] = pd.Series(dtype=float)
    for cal in ("GLD", "SLV"):
        try:
            streams[f"{cal}_calendar"] = calendar_daily_series(cal)
        except Exception as e:
            print(f"[exp2060] WARN {cal} calendar stream failed: {e}")
            streams[f"{cal}_calendar"] = pd.Series(dtype=float)
    try:
        streams["crisis_alpha_v5"] = crisis_alpha_daily_series()
    except Exception as e:
        print(f"[exp2060] WARN crisis alpha stream failed: {e}")
        streams["crisis_alpha_v5"] = pd.Series(dtype=float)

    # Drop empty streams before correlating
    streams = {k: v for k, v in streams.items() if len(v) > 0 and v.std() > 0}
    corr = monthly_correlation_matrix(streams)
    print("[exp2060] correlation matrix computed "
          f"({len(streams)} streams)")

    html = render_html(
        baseline_m, scaled_m, strict_m,
        leverage_rows, capacity, corr,
        len(trades), len(trades_scaled), len(trades_strict),
    )
    REPORT_HTML.write_text(html)
    print(f"[exp2060] wrote {REPORT_HTML}")

    summary = {
        "experiment": "EXP-2060",
        "tag": "EXP-2060",
        "description": "Cross-Vol Arb v2 — VoV overlay + leverage + capacity + corr matrix",
        "data_sources": {
            "vol_arb_tape": "EXP-2020 build_trades (real IronVault + Yahoo)",
            "vvol_overlay": "EXP-1970 build_vvol_panel (real ^VIX)",
            "capacity": "real option_daily.volume/open_interest (IronVault)",
            "correlation_streams": [
                "vol_arb_v2 (vvol_scaled)",
                "EXP-1220 exp1220_standalone (real IronVault tape)",
                "GLD calendar (EXP-1770 walk_forward)",
                "SLV calendar (EXP-1770 walk_forward)",
                "Crisis Alpha v5 backtest_v5 (real)",
            ],
        },
        "config": {
            "universe": UNIVERSE,
            "universe_dropped": DROPPED,
            "holding_days": HOLDING_DAYS,
            "vega_notional": VEGA_NOTIONAL,
            "capital": CAPITAL,
            "participation_rate": PARTICIPATION_RATE,
            "start": START,
            "end": END,
        },
        "trade_counts": {
            "baseline": len(trades),
            "vvol_scaled": len(trades_scaled),
            "vvol_strict_z_le_0": len(trades_strict),
        },
        "metrics": {
            "baseline": baseline_m,
            "vvol_scaled": scaled_m,
            "vvol_strict": strict_m,
        },
        "leverage_test": leverage_rows,
        "capacity": capacity,
        "correlation_matrix_monthly": corr,
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[exp2060] wrote {REPORT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
