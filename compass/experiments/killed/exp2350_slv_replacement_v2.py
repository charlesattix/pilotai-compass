"""
EXP-2350 — Replace SLV Calendar with a High-Capacity Stream.

The EXP-2080 static 5-stream portfolio holds a 20% SLV-calendar sleeve
that saturates between $16M (soft) and $82M (hard) of AUM. To scale the
book past ~$100M the SLV slice has to be swapped for something with
real capacity while preserving the pooled walk-forward Sharpe > 5.0 that
EXP-2080 established as the north-star target.

Earlier experiments — honest reconciliation
-------------------------------------------
* EXP-2240 ran QQQ put credit spreads through the canonical EXP-1220
  loop (28 DTE, 5% OTM, 50% profit target, 2× stop, VIX<40) and
  reported Sharpe 2.26, win rate 90%, corr-to-SPY 0.11, n=84 trades.
* EXP-2310 (yesterday) ran a DIFFERENT QQQ variant — weekly entries,
  Δ=-0.30, no profit target, no stop, no VIX filter — and got Sharpe
  -0.14, DD -98%. That was a methodology-stripped comparison, not a
  contradiction of EXP-2240.
* EXP-2350 uses the EXP-2240 risk-managed version because it is the
  real candidate. The naked version is not a serious proposal.
* EXP-2260 (SLV replacement v1) tested TLT credit spreads and got
  Sharpe -0.74 / DD -41% / corr +0.68 using a simplistic weekly loop.
  This experiment re-tests TLT via the same EXP-1220 risk-managed
  framework used by EXP-2240 for QQQ — the 2× stop, profit target
  and VIX filter might rescue the rate-shock losers that killed the
  naive version.

Four candidates

  1. QQQ put credit spreads   RUN  (reuse cached EXP-2250 trades)
  2. TLT put credit spreads   RUN  (fresh via exp2240 framework)
  3. EFA credit spreads    BLOCKED (0 IronVault contracts)
  4. EEM credit spreads    BLOCKED (0 IronVault contracts)

Real data only — Rule Zero:
  * QQQ / TLT option closes from IronVault data/options_cache.db via
    the canonical EXP-1220 put-credit-spread loop in
    compass.exp2240_qqq_iwm_credit_spreads.run_credit_spread_trades.
  * Spot from Yahoo.
  * Stream cache pickle from EXP-2080 (exp1220/v5_hedge/gld_cal/slv_cal/
    cross_vol) for the swap test.

Swap test
---------
Build the 5-stream static-weight daily portfolio three ways:

  baseline        exp1220 40 / gld 20 / slv 20 / cross 15 / v5 5
  swap_to_qqq     exp1220 40 / gld 20 / QQQ 20 / cross 15 / v5 5
  swap_to_tlt     exp1220 40 / gld 20 / TLT 20 / cross 15 / v5 5

Run the same 20-fold walk-forward (train 252 / test 63) used by
EXP-2080. Report pooled OOS Sharpe / CAGR / Max DD and the delta
from baseline.

Success criterion
-----------------
  * Standalone stream Sharpe > 1.5
  * Standalone stream |corr to EXP-1220| < 0.20
  * Standalone stream capacity > $500M/month
  * Combined portfolio pooled-OOS Sharpe > 5.0 after swap

Outputs:
  compass/exp2350_slv_replacement_v2.py            (this file)
  compass/reports/exp2350_slv_replacement_v2.json
  compass/reports/exp2350_slv_replacement_v2.html

Tag: EXP-2350
Run: python3 -m compass.exp2350_slv_replacement_v2
"""

from __future__ import annotations

import json
import math
import os
import pickle
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
REPORT_JSON = REPORT_DIR / "exp2350_slv_replacement_v2.json"
REPORT_HTML = REPORT_DIR / "exp2350_slv_replacement_v2.html"
CACHE_DIR = ROOT / "compass" / "cache"
CACHE_QQQ = CACHE_DIR / "exp2250_qqq_trades.pkl"
CACHE_TLT = CACHE_DIR / "exp2350_tlt_trades.pkl"
CACHE_STREAMS = CACHE_DIR / "exp2080_streams.pkl"
DB_PATH = ROOT / "data" / "options_cache.db"

from compass.exp2240_qqq_iwm_credit_spreads import (
    run_credit_spread_trades, per_trade_metrics, daily_return_stream,
)
from compass.exp2080_corr_regime import (
    metrics as pool_metrics, STATIC_WEIGHTS,
)
from shared.iron_vault import IronVault

START = "2020-03-01"
END = "2025-12-31"
CAPITAL = 100_000.0

TRAIN_DAYS = 252
TEST_DAYS = 63


# ── Data loaders ──────────────────────────────────────────────────────


def load_exp2080_streams() -> pd.DataFrame:
    df = pickle.load(open(CACHE_STREAMS, "rb"))
    return df


def load_qqq_trades() -> List[Dict]:
    if CACHE_QQQ.exists():
        return pickle.load(open(CACHE_QQQ, "rb"))
    raise FileNotFoundError(f"{CACHE_QQQ} missing; run EXP-2250 first")


def load_or_run_tlt_trades() -> List[Dict]:
    if CACHE_TLT.exists():
        print(f"[exp2350] using cached {CACHE_TLT.name}", flush=True)
        return pickle.load(open(CACHE_TLT, "rb"))
    print("[exp2350] running fresh TLT credit spread backtest via exp2240 framework…", flush=True)
    import yfinance as yf
    tlt = yf.download("TLT", start="2019-06-01", end="2026-01-01", progress=False)
    if isinstance(tlt.columns, pd.MultiIndex):
        tlt.columns = tlt.columns.get_level_values(0)
    tlt.index = pd.to_datetime(tlt.index).tz_localize(None).normalize()
    vix = yf.download("^VIX", start="2019-06-01", end="2026-01-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    hd = IronVault.instance()
    trades = run_credit_spread_trades(
        hd, "TLT", tlt, vix,
        width=2.0,           # TLT strikes are $1 apart — use $2 width
        otm_pct=0.97,        # 3% OTM (TLT has lower vol than QQQ)
        dte_target=28,
        min_spacing=10,
        risk_pct=0.03,
        max_contracts=4,
        vix_block=40.0,
        start=START, end=END,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(trades, open(CACHE_TLT, "wb"))
    return trades


# ── Metrics helpers ───────────────────────────────────────────────────


def trades_to_daily(trades: List[Dict], index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(0.0, index=index)
    for t in trades:
        try:
            d = pd.Timestamp(t["exit_date"])
            if d in s.index:
                s.loc[d] += float(t["pnl"]) / CAPITAL
        except Exception:
            pass
    return s


def correlate_streams(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(both) < 30 or both.iloc[:, 0].std() == 0 or both.iloc[:, 1].std() == 0:
        return float("nan")
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def walk_forward_pooled(daily: pd.Series, train_days: int = TRAIN_DAYS,
                         test_days: int = TEST_DAYS) -> Dict[str, float]:
    n = len(daily)
    pooled: List[pd.Series] = []
    i = train_days
    while i + test_days <= n:
        pooled.append(daily.iloc[i:i + test_days])
        i += test_days
    if not pooled:
        return pool_metrics(pd.Series(dtype=float))
    pooled_series = pd.concat(pooled).sort_index()
    return pool_metrics(pooled_series)


def build_portfolio(streams: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    out = pd.Series(0.0, index=streams.index)
    for k, w in weights.items():
        if k in streams.columns:
            out = out + w * streams[k]
    return out


# ── Capacity estimation from real option_daily volumes ───────────────


def capacity_from_trades(trades: List[Dict], ticker: str,
                         contract_dollar_risk: float) -> Dict:
    if not trades:
        return {"n_observations": 0}
    con = sqlite3.connect(str(DB_PATH))
    try:
        vols: List[int] = []
        for t in trades:
            exp = t.get("expiration") or t.get("exit_date")
            sk = t.get("short_strike")
            date = t.get("entry_date")
            if sk is None or date is None:
                continue
            # Try with provided expiration
            row = con.execute("""
                SELECT d.volume FROM option_contracts c
                JOIN option_daily d ON c.contract_symbol = d.contract_symbol
                WHERE c.ticker=? AND c.option_type='P' AND c.strike=? AND d.date=?
                ORDER BY d.volume DESC LIMIT 1
            """, (ticker, float(sk), date)).fetchone()
            if row and row[0]:
                vols.append(int(row[0]))
        if not vols:
            return {"n_observations": 0}
        v = np.array(vols, dtype=float)
        median_entry_cap = float(np.median(v) * 0.10)
        trades_per_year = len(trades) / 6.0
        # Monthly dollar capacity = (median max contracts per entry) × (trades per month) × $risk per contract
        monthly_trades = trades_per_year / 12.0
        monthly_usd = median_entry_cap * monthly_trades * contract_dollar_risk
        return {
            "n_observations": int(len(vols)),
            "median_daily_volume": float(np.median(v)),
            "p5_daily_volume": float(np.percentile(v, 5)),
            "median_max_contracts_per_entry": int(median_entry_cap),
            "monthly_usd_est": float(monthly_usd),
            "trades_per_year": float(trades_per_year),
            "contract_dollar_risk_assumed": float(contract_dollar_risk),
        }
    finally:
        con.close()


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
    h1{border-bottom:3px solid #08486a}
    h2{margin-top:2em;color:#08486a}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#08486a;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#08486a}
    .pill.ok{background:#0a7d1f}
    .pill.bad{background:#c0392b}
    """
    h = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2350 SLV Replacement v2</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2350 — SLV Replacement (v2, swap test)</h1>",
        "<p class='muted'>Swap the 20% SLV-calendar sleeve with a "
        "high-capacity credit-spread stream. Targets: standalone Sharpe "
        "&gt; 1.5, |corr to EXP-1220| &lt; 0.20, capacity &gt; $500M/mo, "
        "AND combined portfolio Sharpe &gt; 5.0 after swap.</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault + Yahoo data only</span></p>",
    ]

    # Candidate standalone metrics
    h.append("<h2>Standalone candidate metrics (EXP-1220 framework: "
             "28 DTE, 5% OTM, 50% profit, 2× stop, VIX&lt;40)</h2>")
    h.append("<table><tr><th>Candidate</th><th>Status</th>"
             "<th>n trades</th><th>Win rate</th><th>Sharpe</th>"
             "<th>CAGR</th><th>Max DD</th><th>Corr vs EXP-1220</th>"
             "<th>Capacity /mo</th><th>Pass</th></tr>")
    for name, row in payload["candidates"].items():
        if row.get("status") == "blocked":
            h.append(
                f"<tr><td class='l'><b>{name}</b></td>"
                f"<td><span class='pill bad'>BLOCKED</span></td>"
                f"<td colspan='7' class='l muted'>{row['reason']}</td>"
                f"<td><span class='pill bad'>n/a</span></td></tr>"
            )
            continue
        m = row["metrics"]
        corr = row.get("corr_vs_exp1220")
        corr_str = f"{corr:+.2f}" if corr is not None and np.isfinite(corr) else "n/a"
        cap_usd = row.get("capacity", {}).get("monthly_usd_est", 0)
        cap_str = _dollar(cap_usd) if cap_usd else "—"
        sharpe = m.get("sharpe", 0.0)
        sharpe_ok = sharpe >= 1.5
        corr_ok = corr is not None and abs(corr) < 0.20
        cap_ok = cap_usd >= 500_000_000
        if sharpe_ok and corr_ok and cap_ok:
            pill = "<span class='pill ok'>ALL 3</span>"
        else:
            bits = [f"Sh{'✓' if sharpe_ok else '✗'}",
                    f"ρ{'✓' if corr_ok else '✗'}",
                    f"cap{'✓' if cap_ok else '✗'}"]
            pill = f"<span class='pill bad'>{' '.join(bits)}</span>"
        h.append(
            f"<tr><td class='l'><b>{name}</b></td>"
            f"<td>RUN</td>"
            f"<td>{m.get('n_trades', 0)}</td>"
            f"<td>{_fmt_pct(m.get('win_rate', 0), 1)}</td>"
            f"<td>{_fmt(sharpe)}</td>"
            f"<td>{_fmt_pct(m.get('cagr_pct', 0) / 100)}</td>"
            f"<td class='neg'>{_fmt_pct(m.get('max_dd_pct', 0) / 100)}</td>"
            f"<td>{corr_str}</td>"
            f"<td class='l'>{cap_str}</td>"
            f"<td>{pill}</td></tr>"
        )
    h.append("</table>")

    # Swap test
    h.append("<h2>Portfolio swap test (walk-forward 20 folds, pooled OOS)</h2>")
    h.append("<p class='muted'>Static weights: exp1220 40 / gld_cal 20 / "
             "slv_cal 20 / cross_vol 15 / v5_hedge 5 → swap the 20% SLV slot.</p>")
    h.append("<table><tr><th>Portfolio</th><th>n days</th>"
             "<th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th>"
             "<th>Sharpe Δ vs baseline</th><th>Sharpe &gt; 5</th></tr>")
    base_sharpe = payload["swap_test"]["baseline"]["sharpe"]
    for label, m in payload["swap_test"].items():
        delta = m["sharpe"] - base_sharpe
        cls = "pos" if m["sharpe"] >= 5.0 else "neg"
        pass_pill = "<span class='pill ok'>YES</span>" if m["sharpe"] >= 5.0 \
            else "<span class='pill bad'>NO</span>"
        h.append(
            f"<tr><td class='l'><b>{label}</b></td>"
            f"<td>{m['n']}</td>"
            f"<td class='{ 'pos' if m['cagr_pct']>0 else 'neg' }'>{m['cagr_pct']:.2f}%</td>"
            f"<td class='{cls}'>{_fmt(m['sharpe'])}</td>"
            f"<td class='neg'>{_fmt_pct(m['max_dd_pct'] / 100)}</td>"
            f"<td>{_fmt_pct(m['vol_pct'] / 100)}</td>"
            f"<td>{delta:+.2f}</td>"
            f"<td>{pass_pill}</td></tr>"
        )
    h.append("</table>")

    # Recommendation
    h.append("<h2>Recommendation</h2>")
    h.append(payload["recommendation_html"])

    # Reconciliation with prior experiments
    h.append("<h2>Reconciliation with prior experiments</h2>")
    h.append("<ul>")
    h.append("<li><b>EXP-2240</b> established QQQ credit spread Sharpe 2.26, "
             "corr-to-SPY 0.11, 84 trades via the canonical EXP-1220 loop "
             "(28 DTE, 5% OTM, 50% profit target, 2× stop, VIX&lt;40). That "
             "result is the basis for this experiment.</li>")
    h.append("<li><b>EXP-2310</b> (yesterday) ran a stripped-down QQQ variant — "
             "weekly entries, Δ=-0.30, no stops, no VIX filter — and got "
             "Sharpe -0.14 / DD -98%. That is NOT a contradiction of "
             "EXP-2240; it is a demonstration that the EXP-1220 risk controls "
             "(profit target, stop loss, VIX filter) are load-bearing. The "
             "naked variant should not be deployed under any circumstance.</li>")
    h.append("<li><b>EXP-2260</b> tested TLT credit spreads via a simplistic "
             "weekly loop and got Sharpe -0.74 / DD -41% / corr +0.68. This "
             "experiment retests TLT via the same risk-managed framework. If "
             "TLT still fails after the EXP-1220 controls, the failure is "
             "structural (rate-shock exposure) rather than a sizing-or-exit "
             "bug.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[exp2350] loading EXP-2080 stream cache…", flush=True)
    streams = load_exp2080_streams()
    print(f"[exp2350] streams: {streams.shape[0]} days × {streams.shape[1]} cols")
    idx = streams.index

    exp1220_daily = streams["exp1220"]

    # ── Candidate 1: QQQ ────────────────────────────────────────────
    print("\n[exp2350] === QQQ credit spreads (reuse EXP-2250 cache) ===", flush=True)
    qqq_trades = load_qqq_trades()
    qqq_m = per_trade_metrics(qqq_trades, "QQQ")
    qqq_daily = daily_return_stream(qqq_trades, idx)
    qqq_corr = correlate_streams(qqq_daily, exp1220_daily)
    qqq_cap = capacity_from_trades(qqq_trades, "QQQ", contract_dollar_risk=500.0)
    print(f"[exp2350] QQQ: n={qqq_m['n_trades']} WR={qqq_m['win_rate']*100:.1f}% "
          f"Sh={qqq_m['sharpe']:.2f} CAGR={qqq_m['cagr_pct']:.2f}% "
          f"DD={qqq_m['max_dd_pct']:.2f}% corr={qqq_corr:.2f}")
    print(f"[exp2350] QQQ capacity: {qqq_cap}")

    # ── Candidate 2: TLT ────────────────────────────────────────────
    print("\n[exp2350] === TLT credit spreads (fresh via exp2240 framework) ===", flush=True)
    tlt_trades = load_or_run_tlt_trades()
    tlt_m = per_trade_metrics(tlt_trades, "TLT")
    tlt_daily = daily_return_stream(tlt_trades, idx)
    tlt_corr = correlate_streams(tlt_daily, exp1220_daily)
    tlt_cap = capacity_from_trades(tlt_trades, "TLT", contract_dollar_risk=200.0)
    print(f"[exp2350] TLT: n={tlt_m['n_trades']} WR={tlt_m['win_rate']*100:.1f}% "
          f"Sh={tlt_m['sharpe']:.2f} CAGR={tlt_m['cagr_pct']:.2f}% "
          f"DD={tlt_m['max_dd_pct']:.2f}% corr={tlt_corr:.2f}")
    print(f"[exp2350] TLT capacity: {tlt_cap}")

    candidates: Dict[str, Dict] = {
        "QQQ_put_credit_spread": {
            "status": "run",
            "metrics": qqq_m,
            "corr_vs_exp1220": qqq_corr,
            "capacity": qqq_cap,
        },
        "TLT_put_credit_spread": {
            "status": "run",
            "metrics": tlt_m,
            "corr_vs_exp1220": tlt_corr,
            "capacity": tlt_cap,
        },
        "EFA_credit_spread": {
            "status": "blocked",
            "reason": ("0 IronVault contracts. Unblock: Polygon Starter "
                       "backfill via OCC symbol construction (same path "
                       "as TLT Dec-2025 backfill)."),
        },
        "EEM_credit_spread": {
            "status": "blocked",
            "reason": ("0 IronVault contracts. Unblock: Polygon Starter "
                       "backfill via OCC symbol construction."),
        },
    }

    # ── Swap test ────────────────────────────────────────────────────
    print("\n[exp2350] === portfolio swap test ===", flush=True)
    streams_qqq = streams.copy()
    streams_qqq["slv_cal"] = qqq_daily.reindex(streams.index).fillna(0.0)
    streams_tlt = streams.copy()
    streams_tlt["slv_cal"] = tlt_daily.reindex(streams.index).fillna(0.0)

    baseline_daily = build_portfolio(streams, STATIC_WEIGHTS)
    swap_qqq_daily = build_portfolio(streams_qqq, STATIC_WEIGHTS)
    swap_tlt_daily = build_portfolio(streams_tlt, STATIC_WEIGHTS)

    baseline_pool = walk_forward_pooled(baseline_daily)
    qqq_pool = walk_forward_pooled(swap_qqq_daily)
    tlt_pool = walk_forward_pooled(swap_tlt_daily)

    print(f"[exp2350] baseline (with SLV)   : Sh={baseline_pool['sharpe']:.2f} "
          f"CAGR={baseline_pool['cagr_pct']:.2f}% DD={baseline_pool['max_dd_pct']:.2f}%")
    print(f"[exp2350] swap SLV → QQQ        : Sh={qqq_pool['sharpe']:.2f} "
          f"CAGR={qqq_pool['cagr_pct']:.2f}% DD={qqq_pool['max_dd_pct']:.2f}%")
    print(f"[exp2350] swap SLV → TLT        : Sh={tlt_pool['sharpe']:.2f} "
          f"CAGR={tlt_pool['cagr_pct']:.2f}% DD={tlt_pool['max_dd_pct']:.2f}%")

    swap_test = {
        "baseline": baseline_pool,
        "swap_to_qqq": qqq_pool,
        "swap_to_tlt": tlt_pool,
    }

    # Recommendation
    rec_parts = ["<ul>"]
    winners = []
    for name, row in candidates.items():
        if row.get("status") != "run":
            continue
        m = row["metrics"]
        corr = row.get("corr_vs_exp1220")
        cap = row.get("capacity", {}).get("monthly_usd_est", 0)
        sharpe_ok = m.get("sharpe", 0.0) >= 1.5
        corr_ok = corr is not None and np.isfinite(corr) and abs(corr) < 0.20
        cap_ok = cap >= 500_000_000
        if sharpe_ok and corr_ok and cap_ok:
            winners.append(name)
    qqq_swap_ok = swap_test["swap_to_qqq"]["sharpe"] >= 5.0
    tlt_swap_ok = swap_test["swap_to_tlt"]["sharpe"] >= 5.0

    if qqq_swap_ok and "QQQ_put_credit_spread" in winners:
        rec_parts.append(
            "<li><b>QQQ passes all four criteria.</b> It clears standalone "
            f"Sharpe ≥ 1.5 ({qqq_m['sharpe']:.2f}), "
            f"|corr| &lt; 0.20 ({qqq_corr:+.2f}), capacity &gt; $500M, AND "
            f"the swapped portfolio holds Sharpe "
            f"{swap_test['swap_to_qqq']['sharpe']:.2f} &gt; 5.0. "
            "<b>Recommendation: swap the 20% SLV sleeve into QQQ credit "
            "spreads using the EXP-1220 risk-managed framework "
            "(28 DTE, 5% OTM, 50% profit target, 2× stop, VIX&lt;40).</b></li>"
        )
    else:
        bits = []
        if "QQQ_put_credit_spread" in winners:
            bits.append("QQQ passes standalone criteria")
        else:
            bits.append("QQQ fails at least one standalone criterion")
        if qqq_swap_ok:
            bits.append("QQQ swap holds combined Sharpe &gt; 5.0")
        else:
            bits.append(f"QQQ swap combined Sharpe {swap_test['swap_to_qqq']['sharpe']:.2f}")
        rec_parts.append(f"<li><b>QQQ verdict:</b> {'; '.join(bits)}.</li>")

    if tlt_swap_ok and "TLT_put_credit_spread" in winners:
        rec_parts.append(
            "<li><b>TLT also passes</b> — standalone Sharpe "
            f"{tlt_m['sharpe']:.2f}, corr {tlt_corr:+.2f}, combined swap "
            f"Sharpe {swap_test['swap_to_tlt']['sharpe']:.2f}.</li>"
        )
    else:
        bits = []
        if "TLT_put_credit_spread" in winners:
            bits.append("TLT passes standalone criteria")
        else:
            bits.append("TLT fails at least one standalone criterion")
        if tlt_swap_ok:
            bits.append("TLT swap holds combined Sharpe &gt; 5.0")
        else:
            bits.append(f"TLT swap combined Sharpe {swap_test['swap_to_tlt']['sharpe']:.2f}")
        rec_parts.append(f"<li><b>TLT verdict:</b> {'; '.join(bits)}.</li>")

    rec_parts.append(
        "<li><b>EFA and EEM</b> both blocked at 0 IronVault contracts. "
        "Unblock via Polygon Starter backfill before they can be "
        "evaluated for a second leg.</li>"
    )
    rec_parts.append("</ul>")

    payload = {
        "experiment": "EXP-2350",
        "tag": "EXP-2350",
        "description": "SLV replacement v2 — QQQ/TLT/EFA/EEM swap test",
        "targets": {
            "standalone_sharpe_min": 1.5,
            "standalone_abs_corr_max": 0.20,
            "standalone_capacity_usd_min": 500_000_000,
            "combined_sharpe_min": 5.0,
        },
        "candidates": candidates,
        "swap_test": swap_test,
        "recommendation_html": "".join(rec_parts),
    }

    html = render_html(payload)
    REPORT_HTML.write_text(html)
    print(f"\n[exp2350] wrote {REPORT_HTML}")

    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp2350] wrote {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
