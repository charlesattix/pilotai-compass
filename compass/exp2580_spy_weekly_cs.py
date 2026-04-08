"""
EXP-2580 — SPY Weekly Credit Spreads (separate stream from EXP-1220)

Hypothesis
----------
EXP-1220 runs a single SPY put-credit-spread cadence: enter 28 days before
a monthly expiration, exit at 50% profit / stop / DTE ≤ 7. It produces
171 trades over 2020-2025 and is the anchor of the portfolio.

The capacity bottleneck at $50M is NOT SPY options (SPY is the most
liquid options market in the world) — it is the SLV calendar and the
VIX-call proxy. Adding *another* SPY-based stream with a different
cadence could therefore raise portfolio capacity without re-introducing
the same bottleneck, PROVIDED the new stream is not perfectly correlated
to EXP-1220.

Weekly Cadence (this experiment)
--------------------------------
- Enter a SPY put credit spread every Monday for the FRIDAY of the
  FOLLOWING week (~10-12 calendar-day DTE).
- Strike selection: 3% OTM short strike (vs EXP-1220's 5% OTM), width 5.
  Tighter OTM → more premium per trade, shorter holding period.
- Exit: 50% profit, 2× stop, or DTE ≤ 2 (tighter because holding window
  is shorter).

The hypothesis is that shorter DTE + closer-to-ATM strikes + weekly
cadence will make the P&L profile moderately decorrelated from EXP-1220
(target correlation < 0.5).

Walk-forward 2020-2025 on real IronVault SPY options. Report:
  - Sharpe, CAGR, DD on trade-level and daily-return series
  - Per-year breakdown
  - Correlation to EXP-1220 (from cached sparse 7-stream frame)
  - Capacity estimate using EXP-2140 square-root impact model

REAL DATA ONLY. Uses the same IronVault primitives as EXP-1220 standalone.

Outputs
-------
  compass/exp2580_spy_weekly_cs.py
  compass/reports/exp2580_spy_weekly_cs.json
  compass/reports/exp2580_spy_weekly_cs.html
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2580_spy_weekly_cs.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2580_spy_weekly_cs.html"
CACHE_V3    = ROOT / "compass" / "cache" / "exp2280_v6_sparse.pkl"

TRADING_DAYS = 252
CAPITAL      = 100_000.0
START = "2020-01-01"
END   = "2025-12-31"

OTM_PCT_SHORT = 0.97    # 3% OTM short strike (tighter than EXP-1220's 5%)
WIDTH         = 5.0
PROFIT_PCT    = 0.50
STOP_MULT     = 2.0
MIN_DTE_EXIT  = 2
TARGET_DTE_DAYS = 10    # target ~10 calendar-day DTE (≈ next-Friday)
MAX_DTE_DAYS    = 14


# ───────────────────────────────────────────────────────────────────────────
# IronVault primitives (adapted from exp1220_standalone)
# ───────────────────────────────────────────────────────────────────────────

def _exp_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _find_weekly_exps(hd, start: str, end: str) -> List[str]:
    """Every SPY put expiration in range."""
    conn = sqlite3.connect(hd._db_path)
    exps = [r[0] for r in conn.execute(
        "SELECT DISTINCT expiration FROM option_contracts "
        "WHERE ticker='SPY' AND option_type='P' AND expiration BETWEEN ? AND ? "
        "ORDER BY expiration", (start, end)).fetchall()]
    conn.close()
    return exps


def _next_td(dt: datetime, td_set: set) -> Optional[datetime]:
    for off in range(7):
        c = dt + timedelta(days=off)
        if c.strftime("%Y-%m-%d") in td_set:
            return c
    return None


def _sell_put_spread(hd, exp: str, trade_date: str, price: float,
                     otm_pct: float, width: float) -> Optional[Dict]:
    strikes = hd.get_available_strikes("SPY", exp, trade_date, "P")
    if not strikes:
        return None
    target = price * otm_pct
    for sk in sorted(strikes, key=lambda k: abs(k - target))[:12]:
        lk = sk - width
        if lk not in strikes:
            cands = [s for s in strikes if s < sk and abs(s - lk) <= 1.0]
            if not cands:
                continue
            lk = max(cands)
        if sk - lk <= 0:
            continue
        pp = hd.get_spread_prices("SPY", _exp_dt(exp), sk, lk, "P", trade_date)
        if pp is None:
            continue
        credit = pp["short_close"] - pp["long_close"]
        if credit > 0.05:
            return {"short": sk, "long": lk, "credit": round(credit, 4),
                    "width": sk - lk, "max_loss": round(sk - lk - credit, 4)}
    return None


def _walk_spread(hd, exp: str, short_k: float, long_k: float,
                 entry_credit: float, entry_dt: datetime, exp_dt_obj: datetime,
                 td_index: pd.DatetimeIndex) -> Tuple[str, str, float, int]:
    td_set = set(td_index.strftime("%Y-%m-%d"))
    hold = 0
    current = entry_dt + timedelta(days=1)
    while current <= exp_dt_obj:
        cs = current.strftime("%Y-%m-%d")
        if cs not in td_set:
            current += timedelta(days=1); continue
        hold += 1
        pp = hd.get_spread_prices("SPY", exp_dt_obj, short_k, long_k, "P", cs)
        if pp is None:
            current += timedelta(days=1); continue
        cv = pp["short_close"] - pp["long_close"]
        if cv <= entry_credit * (1 - PROFIT_PCT):
            return cs, "profit", cv, hold
        if cv - entry_credit > entry_credit * STOP_MULT:
            return cs, "stop", cv, hold
        if (exp_dt_obj - current).days <= MIN_DTE_EXIT:
            return cs, "dte_exit", cv, hold
        current += timedelta(days=1)
    fp = hd.get_spread_prices("SPY", exp_dt_obj, short_k, long_k, "P", exp)
    return exp, "expiration", (fp["short_close"] - fp["long_close"]) if fp else 0.0, hold


# ───────────────────────────────────────────────────────────────────────────
# Weekly-cadence trade driver
# ───────────────────────────────────────────────────────────────────────────

def run_weekly_trades(hd, spy_df: pd.DataFrame, vix: pd.Series) -> List[Dict]:
    """Generate weekly SPY put-credit-spread trades."""
    spy_close = spy_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_weekly_exps(hd, START, END)
    exp_dts = [_exp_dt(e) for e in exps]

    # Iterate every Monday from START to END
    current = pd.Timestamp(START)
    # Move to first Monday
    while current.weekday() != 0:
        current += timedelta(days=1)
    end_ts = pd.Timestamp(END)

    trades = []
    while current <= end_ts:
        entry_dt = _next_td(current.to_pydatetime(), td_set)
        if entry_dt is None:
            current += timedelta(days=7); continue
        es = entry_dt.strftime("%Y-%m-%d")

        # Pick the soonest expiration with TARGET_DTE_DAYS <= dte <= MAX_DTE_DAYS
        target_dte = [e for e in exp_dts
                      if TARGET_DTE_DAYS - 4 <= (e - entry_dt).days <= MAX_DTE_DAYS]
        if not target_dte:
            current += timedelta(days=7); continue
        target_dte.sort(key=lambda e: abs((e - entry_dt).days - TARGET_DTE_DAYS))
        exp_obj = target_dte[0]
        exp = exp_obj.strftime("%Y-%m-%d")

        try:
            price = float(spy_close.loc[es]); v = float(vix.loc[es])
        except Exception:
            current += timedelta(days=7); continue
        if np.isnan(price) or np.isnan(v) or v > 40:
            current += timedelta(days=7); continue

        spread = _sell_put_spread(hd, exp, es, price,
                                  otm_pct=OTM_PCT_SHORT, width=WIDTH)
        if spread is None:
            current += timedelta(days=7); continue

        cts = max(1, min(4, int(CAPITAL * 0.03 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(hd, exp, spread["short"], spread["long"],
                                        spread["credit"], entry_dt, exp_obj,
                                        spy_df.index)
        pnl = (spread["credit"] - ev) * 100 * cts
        trades.append({
            "entry_date": es, "exit_date": ed,
            "dte_days":  (exp_obj - entry_dt).days,
            "pnl": round(pnl, 2),
            "exit_reason": er, "credit": spread["credit"],
            "vix": round(v, 1), "hold_days": hold,
            "contracts": cts,
            "short_strike": spread["short"],
            "long_strike":  spread["long"],
        })
        current += timedelta(days=7)

    return trades


# ───────────────────────────────────────────────────────────────────────────
# Metrics
# ───────────────────────────────────────────────────────────────────────────

def trade_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {"n": 0, "total_pnl": 0.0, "win_rate": 0.0, "avg_pnl": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "max_dd_pct": 0.0,
                "cagr_pct": 0.0, "calmar": 0.0}
    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls); total = float(pnls.sum()); wins = int((pnls > 0).sum())
    eq = np.cumsum(pnls) + CAPITAL
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / peak).max())
    df = pd.DataFrame(trades)
    en = pd.to_datetime(df["entry_date"]); ex = pd.to_datetime(df["exit_date"])
    years = max((ex.max() - en.min()).days / 365.25, 0.5)
    cagr = ((1 + total / CAPITAL) ** (1 / years) - 1) if total > -CAPITAL else -1.0
    mu = float(pnls.mean()); sigma = float(pnls.std(ddof=1)) if n > 1 else 1.0
    tpy = n / max(years, 0.5)
    sharpe = mu / sigma * math.sqrt(tpy) if sigma > 1e-9 else 0.0
    down = pnls[pnls < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else sigma
    sortino = mu / ds * math.sqrt(tpy) if ds > 1e-9 else 0.0
    return {
        "n": n, "total_pnl": round(total, 2),
        "win_rate": round(wins / n, 4),
        "avg_pnl": round(mu, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(dd * 100, 3),
        "cagr_pct": round(cagr * 100, 3),
        "calmar": round(cagr / dd, 3) if dd > 1e-9 else 0.0,
        "avg_hold_days": round(float(df["hold_days"].mean()), 2),
        "avg_dte_days":  round(float(df["dte_days"].mean()), 2),
        "trades_per_year": round(tpy, 2),
    }


def yearly_metrics(trades: List[Dict]) -> Dict[int, Dict]:
    if not trades: return {}
    df = pd.DataFrame(trades)
    df["exit_year"] = pd.to_datetime(df["exit_date"]).dt.year
    return {int(y): trade_metrics(g.to_dict("records"))
            for y, g in df.groupby("exit_year")}


# ───────────────────────────────────────────────────────────────────────────
# Correlation to EXP-1220 (sparse daily series)
# ───────────────────────────────────────────────────────────────────────────

def correlation_to_exp1220(trades: List[Dict]) -> Dict:
    """Build a daily return series from weekly trades, correlate against
    the EXP-1220 series from the cached sparse 7-stream frame."""
    if not CACHE_V3.exists() or not trades:
        return {"status": "cache_missing_or_empty"}
    sparse = pickle.load(open(CACHE_V3, "rb"))
    if "exp1220" not in sparse.columns:
        return {"status": "exp1220_col_missing"}

    # Build weekly CS daily series on the same index
    daily = pd.Series(0.0, index=sparse.index)
    for t in trades:
        try:
            d = pd.Timestamp(t["exit_date"])
            if d in daily.index:
                daily.loc[d] += t["pnl"] / CAPITAL
        except Exception:
            pass

    # Full-sample pearson
    combined = pd.concat([daily.rename("weekly"),
                          sparse["exp1220"].rename("exp1220")], axis=1).dropna()
    if len(combined) < 10:
        return {"status": "insufficient_overlap"}
    overall_corr = float(combined["weekly"].corr(combined["exp1220"]))

    # Only-on-exit-days correlation (non-zero return days)
    non_zero = combined[(combined["weekly"] != 0) | (combined["exp1220"] != 0)]
    active_corr = float(non_zero["weekly"].corr(non_zero["exp1220"])) if len(non_zero) >= 10 else None

    # Per-year correlation
    yearly_corr = {}
    for yr in sorted({d.year for d in combined.index}):
        sub = combined[combined.index.year == yr]
        if len(sub) < 20:
            continue
        yearly_corr[str(yr)] = round(float(sub["weekly"].corr(sub["exp1220"])), 4)

    return {
        "status": "ok",
        "full_sample_pearson": round(overall_corr, 4),
        "active_days_pearson": round(active_corr, 4) if active_corr is not None else None,
        "yearly_pearson":      yearly_corr,
        "weekly_daily_vol":    round(float(daily.std(ddof=1)), 6),
        "exp1220_daily_vol":   round(float(sparse["exp1220"].std(ddof=1)), 6),
        "n_weekly_nonzero":    int((daily != 0).sum()),
    }


# ───────────────────────────────────────────────────────────────────────────
# Capacity estimate
# ───────────────────────────────────────────────────────────────────────────

def capacity_estimate(trades: List[Dict]) -> Dict:
    """Square-root impact model using SPY option ADV + per-trade notional."""
    if not trades:
        return {}
    # SPY median daily contract volume from IronVault (cached in EXP-2230)
    spy_adv_contracts = 2_314_201
    spy_price_now = 655.84
    spy_adv_notional = spy_adv_contracts * 100 * spy_price_now  # ~$151.9B/d

    df = pd.DataFrame(trades)
    avg_contracts = float(df["contracts"].mean())
    short_strike_avg = float(df["short_strike"].mean())
    notional_per_leg = avg_contracts * 100 * short_strike_avg
    notional_per_trade = notional_per_leg * 2  # 2 legs

    # At 1% ADV soft cap:
    # stream_notional(cap) = 0.01 × spy_adv_notional
    # AUM ceiling = stream_notional / portfolio_weight (assume 20%)
    assumed_weight = 0.20
    soft_cap_notional = 0.01 * spy_adv_notional
    hard_cap_notional = 0.05 * spy_adv_notional
    soft_cap_aum = soft_cap_notional / assumed_weight
    hard_cap_aum = hard_cap_notional / assumed_weight

    # Current per-trade participation
    participation_at_100k = notional_per_trade / spy_adv_notional

    return {
        "spy_adv_contracts": spy_adv_contracts,
        "spy_adv_notional_usd": round(spy_adv_notional, 0),
        "avg_contracts_per_trade": round(avg_contracts, 2),
        "avg_short_strike": round(short_strike_avg, 2),
        "notional_per_trade_usd": round(notional_per_trade, 0),
        "participation_at_100k_pct": round(participation_at_100k * 100, 6),
        "assumed_portfolio_weight": assumed_weight,
        "soft_cap_stream_notional_usd": round(soft_cap_notional, 0),
        "hard_cap_stream_notional_usd": round(hard_cap_notional, 0),
        "soft_cap_portfolio_aum_usd":   round(soft_cap_aum, 0),
        "hard_cap_portfolio_aum_usd":   round(hard_cap_aum, 0),
    }


# ───────────────────────────────────────────────────────────────────────────
# HTML
# ───────────────────────────────────────────────────────────────────────────

def write_html(payload: Dict, path: Path) -> None:
    m = payload["metrics"]
    corr = payload["correlation_to_exp1220"]
    cap = payload["capacity"]

    corr_val = corr.get("full_sample_pearson") if isinstance(corr, dict) else None
    corr_ok = corr_val is not None and abs(corr_val) < 0.5
    color = "#16a34a" if corr_ok and m["sharpe"] > 0 else "#ca8a04"

    yr_rows = ""
    for y, ym in sorted(payload["yearly"].items()):
        yr_rows += (f"<tr><td>{y}</td><td>{ym['n']}</td>"
                    f"<td>{ym['sharpe']:+.2f}</td>"
                    f"<td>${ym['total_pnl']:,.0f}</td>"
                    f"<td>{ym['win_rate']:.0%}</td>"
                    f"<td>{ym['max_dd_pct']:.2f}%</td></tr>")

    corr_rows = ""
    if isinstance(corr, dict) and "yearly_pearson" in corr:
        for y, c in sorted(corr["yearly_pearson"].items()):
            corr_rows += f"<tr><td>{y}</td><td>{c:+.3f}</td></tr>"

    full_pearson_str = (f"{corr['full_sample_pearson']:+.3f}"
                        if isinstance(corr.get("full_sample_pearson"), (int, float))
                        else "—")
    active_pearson_str = (f"{corr['active_days_pearson']:+.3f}"
                          if isinstance(corr.get("active_days_pearson"), (int, float))
                          else "—")

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>EXP-2580 SPY Weekly Credit Spreads</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b;max-width:1100px}}
h1{{font-size:1.5rem;color:#0f172a}} h2{{font-size:1.05rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:1.6rem}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:18px}}
.headline{{background:#f0fdf4;border-left:5px solid {color};padding:14px 18px;border-radius:6px;margin:14px 0;font-size:0.95rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:14px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}} .card .v{{font-size:1.15rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:0.83rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}} td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}} td:first-child{{text-align:left}}
</style></head><body>
<h1>EXP-2580 — SPY Weekly Credit Spreads (separate from EXP-1220)</h1>
<p class="meta">Weekly-cadence SPY put credit spreads, 3% OTM short, width 5, target ~10-day DTE.
REAL IronVault SPY options 2020-2025.</p>

<div class="headline">
<strong>Headline:</strong> {m['n']} trades · Sharpe <strong>{m['sharpe']:+.2f}</strong>
· CAGR <strong>{m['cagr_pct']:+.1f}%</strong>
· DD <strong>{m['max_dd_pct']:.2f}%</strong>
· corr to EXP-1220 <strong>{corr_val:+.2f}</strong> ({'MODERATE <0.5 ✓' if corr_ok else 'HIGH ≥0.5 — limited diversification'})
· Capacity (20% weight assumption) soft <strong>${cap['soft_cap_portfolio_aum_usd']/1e9:.2f}B</strong> / hard <strong>${cap['hard_cap_portfolio_aum_usd']/1e9:.2f}B</strong>
</div>

<div class="grid">
  <div class="card"><div class="l">Trades</div><div class="v">{m['n']}</div></div>
  <div class="card"><div class="l">Win rate</div><div class="v">{m['win_rate']:.0%}</div></div>
  <div class="card"><div class="l">Sharpe</div><div class="v">{m['sharpe']:+.2f}</div></div>
  <div class="card"><div class="l">CAGR</div><div class="v">{m['cagr_pct']:+.1f}%</div></div>
  <div class="card"><div class="l">Max DD</div><div class="v">{m['max_dd_pct']:.2f}%</div></div>
  <div class="card"><div class="l">Avg DTE</div><div class="v">{m['avg_dte_days']:.1f}d</div></div>
  <div class="card"><div class="l">Avg hold</div><div class="v">{m['avg_hold_days']:.1f}d</div></div>
  <div class="card"><div class="l">Trades/yr</div><div class="v">{m['trades_per_year']:.0f}</div></div>
</div>

<h2>Per-year breakdown</h2>
<table><tr><th>Year</th><th>n</th><th>Sharpe</th><th>Total PnL</th><th>Win%</th><th>Max DD</th></tr>
{yr_rows}</table>

<h2>Correlation to EXP-1220</h2>
<p class="meta">Lower correlation = better diversification value. Target &lt; 0.5.</p>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Full-sample pearson</td><td>{full_pearson_str}</td></tr>
<tr><td>Active-days pearson</td><td>{active_pearson_str}</td></tr>
<tr><td>Non-zero weekly days</td><td>{corr.get('n_weekly_nonzero','—')}</td></tr>
</table>

<h3>Per-year correlation to EXP-1220</h3>
<table><tr><th>Year</th><th>Pearson ρ</th></tr>{corr_rows}</table>

<h2>Capacity estimate (EXP-2140 impact model)</h2>
<p class="meta">SPY options are the most liquid options market in the world; capacity is gated by
portfolio weight and participation cap only.</p>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>SPY ADV contracts</td><td>{cap['spy_adv_contracts']:,}</td></tr>
<tr><td>SPY ADV notional</td><td>${cap['spy_adv_notional_usd']/1e9:.1f}B/d</td></tr>
<tr><td>Avg contracts/trade</td><td>{cap['avg_contracts_per_trade']:.2f}</td></tr>
<tr><td>Avg short strike</td><td>${cap['avg_short_strike']:,.0f}</td></tr>
<tr><td>Notional per trade</td><td>${cap['notional_per_trade_usd']:,.0f}</td></tr>
<tr><td>Participation @ $100K</td><td>{cap['participation_at_100k_pct']:.6f}%</td></tr>
<tr><td>Assumed weight</td><td>{cap['assumed_portfolio_weight']*100:.0f}%</td></tr>
<tr><td>Stream soft-cap notional</td><td>${cap['soft_cap_stream_notional_usd']/1e9:.2f}B</td></tr>
<tr><td>Stream hard-cap notional</td><td>${cap['hard_cap_stream_notional_usd']/1e9:.2f}B</td></tr>
<tr><td><strong>Soft-cap AUM</strong></td><td><strong>${cap['soft_cap_portfolio_aum_usd']/1e9:.2f}B</strong></td></tr>
<tr><td><strong>Hard-cap AUM</strong></td><td><strong>${cap['hard_cap_portfolio_aum_usd']/1e9:.2f}B</strong></td></tr>
</table>

<h2>Method</h2>
<ul>
<li>Entry: every Monday (or next trading day), pick SPY put expiration with
   {TARGET_DTE_DAYS-4}-{MAX_DTE_DAYS} day DTE. Enter 3% OTM short put, 5-point width.</li>
<li>Exit: 50% profit, 2× stop, or DTE ≤ 2.</li>
<li>Sizing: 3% of $100K capital as max loss → 1-4 contracts.</li>
<li>Correlation: convert weekly trades to exit-date daily return series on the
   cached EXP-2280 sparse index, correlate against exp1220 column.</li>
<li>Capacity: 1% ADV soft / 5% ADV hard × 20% assumed portfolio weight.</li>
</ul>
<div style="color:#94a3b8;font-size:.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/exp2580_spy_weekly_cs.py · REAL IronVault SPY options + Yahoo</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-2580 — SPY Weekly Credit Spreads")
    print("=" * 60)

    hd = IronVault.instance()

    print("[1/4] Loading SPY + VIX (Yahoo)...")
    import yfinance as yf
    spy_df = yf.download("SPY", start="2019-06-01", end="2026-04-02",
                         progress=False, auto_adjust=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)
    spy_df.index = pd.to_datetime(spy_df.index)
    vix_df = yf.download("^VIX", start="2019-06-01", end="2026-04-02",
                         progress=False, auto_adjust=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    vix = vix_df["Close"]; vix.index = pd.to_datetime(vix.index)

    print("[2/4] Running weekly-cadence SPY put-credit-spread trades...")
    trades = run_weekly_trades(hd, spy_df, vix)
    print(f"      {len(trades)} trades")

    if not trades:
        print("NO TRADES — abort")
        return

    m = trade_metrics(trades)
    print(f"      Sharpe {m['sharpe']:+.2f}  CAGR {m['cagr_pct']:+.2f}%  "
          f"DD {m['max_dd_pct']:.2f}%  WR {m['win_rate']:.0%}  "
          f"avg DTE {m['avg_dte_days']:.1f}d")

    print("[3/4] Correlation to EXP-1220 (sparse 7-stream frame)...")
    corr = correlation_to_exp1220(trades)
    if corr.get("status") == "ok":
        print(f"      full-sample ρ = {corr['full_sample_pearson']:+.3f}")
        print(f"      active-days ρ = {corr['active_days_pearson']}")
        for y, c in sorted(corr.get("yearly_pearson", {}).items()):
            print(f"        {y}: {c:+.3f}")
    else:
        print(f"      status: {corr.get('status')}")

    print("[4/4] Capacity estimate (EXP-2140 impact model)...")
    cap = capacity_estimate(trades)
    print(f"      avg notional/trade ${cap['notional_per_trade_usd']:,.0f}")
    print(f"      soft-cap AUM ${cap['soft_cap_portfolio_aum_usd']/1e9:.2f}B @ 20% weight")
    print(f"      hard-cap AUM ${cap['hard_cap_portfolio_aum_usd']/1e9:.2f}B @ 20% weight")

    yr = yearly_metrics(trades)

    payload = {
        "experiment": "EXP-2580",
        "title": "SPY Weekly Credit Spreads — high-capacity separate stream",
        "date_range": {"start": START, "end": END},
        "params": {
            "otm_pct_short": OTM_PCT_SHORT,
            "width": WIDTH,
            "profit_pct": PROFIT_PCT,
            "stop_mult": STOP_MULT,
            "target_dte_days": TARGET_DTE_DAYS,
            "max_dte_days": MAX_DTE_DAYS,
            "min_dte_exit": MIN_DTE_EXIT,
        },
        "data_sources": {
            "spy_options": "IronVault options_cache.db (REAL)",
            "spy": "Yahoo Finance SPY (REAL)",
            "vix": "Yahoo Finance ^VIX (REAL)",
            "exp1220_series": "compass/cache/exp2280_v6_sparse.pkl (REAL)",
        },
        "metrics": m,
        "yearly": {str(k): v for k, v in yr.items()},
        "correlation_to_exp1220": corr,
        "capacity": cap,
        "n_trades": len(trades),
        "first_trades_sample": trades[:5],
        "last_trades_sample":  trades[-5:],
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    write_html(payload, REPORT_HTML)
    print(f"\nReports → {REPORT_JSON.name} + {REPORT_HTML.name}")
    return payload


if __name__ == "__main__":
    main()
