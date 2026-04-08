"""
EXP-2660 — AUM Capacity: Multi-Underlying Scaling Audit
========================================================

Question
--------
Can we scale the existing credit-spread strategy to MORE underlyings to
linearly grow capacity? Test 8 candidate names against the EXP-1220
engine and measure (a) data feasibility, (b) standalone Sharpe, (c)
correlation to the existing 7-stream cube, (d) capacity proxies.

Candidates
----------
  IWM, EEM, DIA, XLE, XLV, AAPL, MSFT, AMZN

Rule Zero data check (FIRST step — no synthesised options chains)
-----------------------------------------------------------------
IronVault has options data for:
  XLE   1,757 contracts   2020-04-17 → 2026-04-04   ← TRADEABLE
Zero contracts for:
  IWM, EEM, DIA, XLV, AAPL, MSFT, AMZN              ← BLOCKED

For the seven blocked names we cannot run a credit-spread backtest
under Rule Zero. We DO report Yahoo-derived underlier ADV (daily
dollar volume) as a capacity-feasibility proxy so the next data-buy
decision is informed by liquidity, not by guesswork. Adding their
options chains requires a paid Polygon Options Advanced (~$199/mo) or
CBOE DataShop subscription.

For XLE we run the EXP-2160 put-credit-spread engine end-to-end on the
real IronVault chain and report Sharpe / DD / correlation to the
existing 7-stream portfolio.

Outputs
  compass/reports/exp2660_aum_capacity_scaling.json
  compass/reports/exp2660_aum_capacity_scaling.html
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp2080_corr_regime import load_streams
from compass.exp2160_high_capacity_alts import (
    run_put_credit_spreads,
    trades_to_daily_pct,
)
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2660_aum_capacity_scaling.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2660_aum_capacity_scaling.html"

CANDIDATES = ["IWM", "EEM", "DIA", "XLE", "XLV", "AAPL", "MSFT", "AMZN"]

CAPITAL = 100_000
TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: IronVault coverage check
# ─────────────────────────────────────────────────────────────────────────────
def check_ironvault_coverage(tickers: List[str]) -> Dict[str, Dict]:
    hd = IronVault.instance()
    con = sqlite3.connect(hd._db_path)
    out: Dict[str, Dict] = {}
    for tk in tickers:
        n = con.execute(
            "SELECT COUNT(*) FROM option_contracts WHERE ticker=?", (tk,)
        ).fetchone()[0]
        if n > 0:
            dr = con.execute(
                "SELECT MIN(as_of_date), MAX(as_of_date) FROM option_contracts WHERE ticker=?",
                (tk,),
            ).fetchone()
            out[tk] = {
                "n_contracts": int(n),
                "date_range": [str(dr[0]), str(dr[1])],
                "tradeable": True,
            }
        else:
            out[tk] = {"n_contracts": 0, "date_range": None, "tradeable": False}
    con.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Yahoo capacity proxies for ALL candidates (tradeable or not)
# ─────────────────────────────────────────────────────────────────────────────
def yahoo_capacity_proxies(tickers: List[str],
                            start: str = "2024-01-01",
                            end: str = "2026-01-01") -> Dict[str, Dict]:
    import yfinance as yf
    out: Dict[str, Dict] = {}
    for tk in tickers:
        try:
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                out[tk] = {"error": "no data"}
                continue
            close = df["Close"].dropna()
            vol   = df["Volume"].dropna()
            adv_shares = float(vol.mean())
            adv_notional = float((vol * close.reindex(vol.index)).mean())
            out[tk] = {
                "n_days": int(len(df)),
                "last_close": round(float(close.iloc[-1]), 2),
                "median_close": round(float(close.median()), 2),
                "adv_shares": round(adv_shares, 0),
                "adv_notional_usd": round(adv_notional, 0),
                "median_volume": round(float(vol.median()), 0),
            }
        except Exception as e:
            out[tk] = {"error": str(e)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Run EXP-2160 engine on tradeable candidates (XLE)
# ─────────────────────────────────────────────────────────────────────────────
def run_credit_spread_backtest(ticker: str) -> Dict:
    hd = IronVault.instance()
    con = sqlite3.connect(hd._db_path)
    try:
        trades = run_put_credit_spreads(con, ticker)
    finally:
        con.close()
    return {
        "ticker": ticker,
        "n_trades": len(trades),
        "trades": trades,
    }


def trade_metrics(trades: List, ticker: str) -> Dict:
    if not trades:
        return {"ticker": ticker, "n": 0, "sharpe": 0.0, "cagr_pct": 0.0,
                "max_dd_pct": 0.0, "win_rate": 0.0, "total_pnl_pct": 0.0}
    pnl_pcts = np.array([t.pnl_pct_capital for t in trades], dtype=float)
    eq = np.cumprod(1 + pnl_pcts)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    yrs = max(1.0, (
        datetime.strptime(trades[-1].expiration, "%Y-%m-%d") -
        datetime.strptime(trades[0].entry_date, "%Y-%m-%d")
    ).days / 365.25)
    tpy = len(pnl_pcts) / yrs
    mu, sd = pnl_pcts.mean(), (pnl_pcts.std(ddof=1) if len(pnl_pcts) > 1 else 0.0)
    sharpe = (mu / sd) * math.sqrt(tpy) if sd > 1e-12 else 0.0
    return {
        "ticker": ticker,
        "n": int(len(pnl_pcts)),
        "sharpe": round(float(sharpe), 3),
        "cagr_pct": round(float((eq[-1]) ** (1 / yrs) * 100 - 100), 3),
        "max_dd_pct": round(float(-dd.min() * 100), 3),
        "win_rate": round(float((pnl_pcts > 0).mean()), 3),
        "total_pnl_pct": round(float((eq[-1] - 1) * 100), 3),
        "trades_per_yr": round(float(tpy), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Correlation to existing 7-stream cube
# ─────────────────────────────────────────────────────────────────────────────
def correlation_to_existing(daily_new: pd.Series) -> Dict:
    base = load_streams()
    common = base.index.intersection(daily_new.index)
    if len(common) < 60:
        return {"n_days": int(len(common)), "by_stream": {}}
    new = daily_new.reindex(common).fillna(0.0)
    out = {}
    for col in base.columns:
        sub = base[col].reindex(common).fillna(0.0)
        try:
            r = float(new.corr(sub))
        except Exception:
            r = float("nan")
        out[col] = round(r, 4)
    # also vs the equal-weighted portfolio
    pw = base.reindex(common).mean(axis=1)
    out["equal_weight_portfolio"] = round(float(new.corr(pw)), 4)
    return {"n_days": int(len(common)), "by_stream": out}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[1/5] checking IronVault coverage for 8 candidates …")
    coverage = check_ironvault_coverage(CANDIDATES)
    tradeable = [t for t, v in coverage.items() if v["tradeable"]]
    blocked   = [t for t, v in coverage.items() if not v["tradeable"]]
    print(f"      tradeable: {tradeable}")
    print(f"      blocked  : {blocked}")

    print("[2/5] pulling Yahoo capacity proxies for ALL candidates …")
    yahoo = yahoo_capacity_proxies(CANDIDATES)
    for tk, v in yahoo.items():
        if "error" in v:
            print(f"      {tk:5s}  ERR {v['error']}")
        else:
            print(f"      {tk:5s}  ADV ${v['adv_notional_usd']/1e9:6.2f}B  median vol {v['median_volume']:>12,.0f}")

    print("[3/5] running EXP-2160 credit-spread engine on tradeable candidates …")
    backtests: Dict[str, Dict] = {}
    daily_series: Dict[str, pd.Series] = {}
    base_index = load_streams().index
    for tk in tradeable:
        r = run_credit_spread_backtest(tk)
        m = trade_metrics(r["trades"], tk)
        backtests[tk] = m
        if r["trades"]:
            daily_series[tk] = trades_to_daily_pct(r["trades"], base_index).rename(f"{tk.lower()}_cs")
        print(f"      {tk:5s}  n={m['n']:3}  WR={m['win_rate']*100:5.1f}%  Sharpe={m['sharpe']:5.2f}  CAGR={m['cagr_pct']:6.2f}%  DD={m['max_dd_pct']:5.2f}%")

    print("[4/5] correlation of new streams to existing 7-stream cube …")
    correlations: Dict[str, Dict] = {}
    for tk, s in daily_series.items():
        correlations[tk] = correlation_to_existing(s)
        ew = correlations[tk]["by_stream"].get("equal_weight_portfolio")
        print(f"      {tk:5s}  vs EW portfolio: pearson {ew}")

    print("[5/5] writing report …")
    payload = {
        "experiment": "EXP-2660",
        "name": "AUM Capacity — Multi-Underlying Scaling Audit",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "candidates": CANDIDATES,
        "ironvault_coverage": coverage,
        "yahoo_capacity_proxies": yahoo,
        "backtests": backtests,
        "correlations_to_existing_7stream": correlations,
        "summary": {
            "n_candidates": len(CANDIDATES),
            "n_tradeable_under_rule_zero": len(tradeable),
            "n_blocked_by_data": len(blocked),
            "tradeable_tickers": tradeable,
            "blocked_tickers": blocked,
            "data_unblocking_required": (
                "7 of 8 candidates have ZERO option contracts in IronVault. "
                "Rule Zero forbids backtesting them on synthesised chains. "
                "Unblocking them requires a paid Polygon Options Advanced "
                "(~$199/mo) or CBOE DataShop subscription."
            ),
        },
        "rule_zero": {
            "enforced": True,
            "synthetic_chains_used": False,
            "real_data_only": True,
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    rows_cov = ""
    for tk, v in p["ironvault_coverage"].items():
        ya = p["yahoo_capacity_proxies"].get(tk, {})
        adv = ya.get("adv_notional_usd")
        adv_str = f"${adv/1e9:.2f}B" if adv else "—"
        last = ya.get("last_close")
        status = "TRADEABLE" if v["tradeable"] else "BLOCKED"
        cls = "ok" if v["tradeable"] else "warn"
        rows_cov += (
            f"<tr><td>{tk}</td><td>{v['n_contracts']:,}</td>"
            f"<td>{(v['date_range'] or ['—','—'])[0]}</td>"
            f"<td>{(v['date_range'] or ['—','—'])[1]}</td>"
            f"<td>{adv_str}</td><td>${last or '—'}</td>"
            f"<td class='{cls}'>{status}</td></tr>"
        )

    rows_bt = ""
    for tk, m in p["backtests"].items():
        rows_bt += (
            f"<tr><td>{tk}</td><td>{m['n']}</td><td>{m['win_rate']*100:.1f}%</td>"
            f"<td>{m['sharpe']:.2f}</td><td>{m['cagr_pct']:.2f}%</td>"
            f"<td>{m['max_dd_pct']:.2f}%</td><td>{m['total_pnl_pct']:.2f}%</td></tr>"
        )
    if not rows_bt:
        rows_bt = "<tr><td colspan='7' class='small'>no tradeable candidates</td></tr>"

    rows_corr = ""
    for tk, c in p["correlations_to_existing_7stream"].items():
        cells = "".join(f"<td>{v:+.3f}</td>" for k, v in c["by_stream"].items() if k != "equal_weight_portfolio")
        ew = c["by_stream"].get("equal_weight_portfolio", "—")
        rows_corr += f"<tr><td>{tk}</td>{cells}<td><b>{ew:+.3f}</b></td></tr>"
    headers_corr = ""
    if p["correlations_to_existing_7stream"]:
        first = next(iter(p["correlations_to_existing_7stream"].values()))
        headers_corr = "".join(
            f"<th>{k}</th>" for k in first["by_stream"].keys() if k != "equal_weight_portfolio"
        ) + "<th>EW portfolio</th>"

    s = p["summary"]
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2660 — Multi-Underlying Scaling Audit</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;background:#fff;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .warn{{color:#b86b00;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
 .callout{{background:#fff8e1;border-left:4px solid #e0a500;padding:.9em 1.1em;margin:1em 0}}
</style></head><body>
<h1>EXP-2660 — Multi-Underlying Scaling Audit</h1>
<p class='small'>Generated {p['generated']} · 8 candidates tested · Rule Zero clean.</p>

<div class='callout'>
<b>Headline:</b> {s['n_tradeable_under_rule_zero']} of {s['n_candidates']} candidates
have IronVault option chains and can be backtested today. The remaining
{s['n_blocked_by_data']} ({', '.join(s['blocked_tickers'])}) are <b>blocked by
Rule Zero</b> — zero option contracts in <code>data/options_cache.db</code>.
Unblocking them requires a paid market-data subscription.
</div>

<h2>1. IronVault coverage + Yahoo capacity proxies</h2>
<table>
<tr><th>Ticker</th><th>IronVault contracts</th><th>From</th><th>To</th>
 <th>ADV notional (Yahoo)</th><th>Last close</th><th>Status</th></tr>
{rows_cov}
</table>

<h2>2. Credit-spread backtest (tradeable candidates only)</h2>
<table>
<tr><th>Ticker</th><th>n</th><th>WR</th><th>Sharpe</th>
 <th>CAGR</th><th>Max DD</th><th>Total PnL</th></tr>
{rows_bt}
</table>

<h2>3. Correlation to existing 7-stream cube</h2>
<table>
<tr><th>New stream</th>{headers_corr}</tr>
{rows_corr}
</table>

<h2>4. Recommendation</h2>
<ul>
<li><b>Add XLE</b> as the 8th sleeve if its standalone Sharpe / correlation
    pass the production bar (typically Sharpe ≥ 1.0, |corr| &lt; 0.4).</li>
<li><b>Buy Polygon Options Advanced</b> (~$199/mo) to unblock IWM, EEM,
    DIA, XLV, AAPL, MSFT, AMZN — all 7 candidates have multi-billion-dollar
    underlier ADV (Yahoo capacity proxy is informative). At $199/mo the
    payback on a single ~$50M AUM uplift is &lt; 1 day.</li>
<li><b>Do NOT extrapolate fills</b> for the blocked names. Rule Zero
    forbids running the engine on synthesised option chains.</li>
</ul>

<h2>Honest scope note</h2>
<p class='small'>This experiment is a feasibility audit, not a portfolio
construction step. The single XLE backtest (if Sharpe is positive)
should be promoted via a separate experiment that integrates it into
the 7-stream cube and re-runs the risk-parity walk-forward.</p>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
