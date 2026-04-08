"""
EXP-2270 — XLF / XLI Slippage Impact Analysis
==============================================

Question
--------
EXP-2210 flagged that XLF/XLI alpha may not survive realistic execution
costs. Quantify exactly how much slippage XLF/XLI credit spreads can
absorb before they stop being viable, and measure the knock-on impact
on the 7-stream portfolio from EXP-2220.

Critical data limitation (Rule Zero)
------------------------------------
IronVault holds NO bid/ask quotes for XLF/XLI options — only daily
OHLC, volume, and open-interest in `option_daily`. There is also
zero intraday data for these tickers (`option_intraday` rows = 0).
We therefore cannot *measure* the bid-ask directly. Instead we:

  1. Report empirical liquidity proxies from real IronVault rows for
     the contracts actually traded in EXP-2160's XLF/XLI tape:
        • per-contract daily volume distribution
        • per-contract open-interest distribution
        • intraday range (high − low) / close on days with non-zero volume
  2. Sweep slippage as an *exogenous parameter* across the industry-
     standard range for ETF options ($0.01–$0.10 per leg per side) and
     report Sharpe break-evens.

This is the honest framing — no fabricated quotes, just real OHLC
descriptive stats next to a transparent sensitivity sweep.

P&L cost model
--------------
A put credit spread has 4 legs of execution per round trip
(open short, open long, close short, close long). At one-way per-leg
slippage S (in dollars):

    Δpnl_per_spread = -4 · S · 100   (per contract, both legs round trip)

Multiplied by contract count and expressed as a fraction of capital,
that is exactly what we subtract from the EXP-2160 trade tape's
``pnl_pct_capital`` field.

Outputs
-------
  compass/reports/exp2270_xlf_xli_slippage.json
  compass/reports/exp2270_xlf_xli_slippage.html
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp2080_corr_regime import load_streams
from compass.exp2160_high_capacity_alts import (
    run_put_credit_spreads,
    trades_to_daily_pct,
    SpreadTrade,
)
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2270_xlf_xli_slippage.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2270_xlf_xli_slippage.html"

TRADING_DAYS = 252
CAPITAL = 100_000

# One-way per-leg slippage in dollars (i.e. 0.01 = 1 cent per leg per side).
SLIPPAGE_GRID_CENTS = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]


# ─────────────────────────────────────────────────────────────────────────────
# Empirical liquidity proxies for the actual contracts traded
# ─────────────────────────────────────────────────────────────────────────────
def liquidity_stats_for_trades(con: sqlite3.Connection,
                                trades: List[SpreadTrade]) -> Dict:
    """Distribution of volume / OI / (high-low)/close for the contracts that
    EXP-2160 actually used. Only counts rows with volume > 0."""
    used = set()
    for t in trades:
        used.add(t.short_symbol)
        used.add(t.long_symbol)
    if not used:
        return {"n_contracts": 0}
    placeholders = ",".join(["?"] * len(used))
    rows = con.execute(
        f"SELECT contract_symbol, date, high, low, close, volume, open_interest "
        f"FROM option_daily "
        f"WHERE contract_symbol IN ({placeholders}) AND volume > 0",
        list(used),
    ).fetchall()
    if not rows:
        return {"n_contracts": int(len(used)), "n_active_days": 0}

    closes = np.array([r[4] for r in rows if r[4] is not None and r[4] > 0], dtype=float)
    vols   = np.array([r[5] for r in rows if r[5] is not None], dtype=float)
    ois    = np.array([r[6] for r in rows if r[6] is not None], dtype=float)
    hl_pct = np.array(
        [(r[2] - r[3]) / r[4]
         for r in rows
         if r[2] is not None and r[3] is not None and r[4] is not None and r[4] > 0],
        dtype=float,
    )

    def _stats(arr: np.ndarray) -> Dict:
        if len(arr) == 0:
            return {"n": 0}
        return {
            "n": int(len(arr)),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "p10": round(float(np.quantile(arr, 0.10)), 4),
            "p25": round(float(np.quantile(arr, 0.25)), 4),
            "p75": round(float(np.quantile(arr, 0.75)), 4),
            "p90": round(float(np.quantile(arr, 0.90)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }

    return {
        "n_contracts": int(len(used)),
        "n_active_rows": int(len(rows)),
        "volume_per_day":     _stats(vols),
        "open_interest":      _stats(ois),
        "high_low_over_close": _stats(hl_pct),
        "close_premium":      _stats(closes),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Slippage application
# ─────────────────────────────────────────────────────────────────────────────
def apply_slippage(trades: List[SpreadTrade],
                   slip_per_leg_dollars: float) -> List[Dict]:
    """Return a list of trade dicts with slippage-adjusted pnl_pct.

    Each spread costs 4 leg executions per round trip; with S$ per leg
    one-way slippage, we lose 4·S·100 dollars per contract.
    """
    out = []
    cost_per_contract = 4.0 * slip_per_leg_dollars * 100.0
    for t in trades:
        # Re-derive contract count from the original pnl path
        max_loss_per_spread = max((t.short_strike - t.long_strike) - t.net_credit, 0.01) * 100.0
        n_contracts = (0.02 * CAPITAL) / max_loss_per_spread   # CS_RISK_PER_TRADE = 0.02
        slip_cost = cost_per_contract * n_contracts
        adjusted_pnl_dollars = (t.pnl_per_spread * n_contracts) - slip_cost
        out.append({
            "ticker": t.ticker,
            "entry_date": t.entry_date,
            "expiration": t.expiration,
            "weekday": datetime.strptime(t.entry_date, "%Y-%m-%d").weekday(),
            "n_contracts": n_contracts,
            "pnl_pct_capital_clean": float(t.pnl_pct_capital),
            "pnl_pct_capital_adj":   float(adjusted_pnl_dollars / CAPITAL),
            "slip_cost_dollars":     float(slip_cost),
        })
    return out


def trade_metrics(trades: List[Dict], pct_field: str, label: str) -> Dict:
    if not trades:
        return {"label": label, "n": 0, "wr": 0.0, "sharpe": 0.0,
                "cagr_pct": 0.0, "max_dd_pct": 0.0,
                "total_return_pct": 0.0}
    rets = np.array([t[pct_field] for t in trades], dtype=float)
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    yrs = max(1.0, (
        datetime.strptime(trades[-1]["expiration"], "%Y-%m-%d") -
        datetime.strptime(trades[0]["entry_date"],  "%Y-%m-%d")
    ).days / 365.25)
    tpy = len(rets) / yrs
    mu, sd = rets.mean(), (rets.std(ddof=1) if len(rets) > 1 else 0.0)
    sharpe = (mu / sd) * math.sqrt(tpy) if sd > 1e-12 else 0.0
    return {
        "label": label, "n": int(len(rets)),
        "wr": float((rets > 0).mean()),
        "sharpe": round(float(sharpe), 3),
        "total_return_pct": round(float(eq[-1] - 1) * 100, 3),
        "cagr_pct": round(float(eq[-1] ** (1 / yrs) - 1) * 100, 3),
        "max_dd_pct": round(float(-dd.min() * 100), 3),
        "trades_per_yr": round(float(tpy), 2),
    }


def slippage_sweep(raw_trades: List[SpreadTrade],
                   ticker: str) -> Dict:
    rows = []
    break_even_15 = None
    break_even_10 = None
    for cents in SLIPPAGE_GRID_CENTS:
        s = cents / 100.0
        adj = apply_slippage(raw_trades, s)
        m = trade_metrics(adj, "pnl_pct_capital_adj", f"{ticker} slip {cents:g}c")
        rows.append({"slip_cents_per_leg": cents,
                     **{k: v for k, v in m.items() if k != "label"}})
        if m["sharpe"] < 1.5 and break_even_15 is None:
            break_even_15 = cents
        if m["sharpe"] < 1.0 and break_even_10 is None:
            break_even_10 = cents
    return {
        "ticker": ticker,
        "n_raw_trades": len(raw_trades),
        "sweep": rows,
        "break_even_sharpe_15_cents": break_even_15,
        "break_even_sharpe_10_cents": break_even_10,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trade-timing experiment
# ─────────────────────────────────────────────────────────────────────────────
def by_weekday(trades: List[Dict], pct_field: str) -> Dict:
    out: Dict[int, List[Dict]] = {}
    for t in trades:
        out.setdefault(t["weekday"], []).append(t)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    return {
        days[d]: trade_metrics(out.get(d, []), pct_field, days[d])
        for d in range(5)
        if out.get(d)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio impact (use the 7-stream cube from EXP-2220)
# ─────────────────────────────────────────────────────────────────────────────
def build_seven_stream_cube_with_slippage(slip_cents: float,
                                           min_var_weights: bool = True) -> Dict:
    """Reconstruct the 7-stream cube with the chosen slippage applied to
    XLF and XLI streams only. Compute pooled metrics under both equal-weight
    and the EXP-2170 min-variance allocator."""
    base = load_streams()  # 5 cached streams
    cols = list(base.columns) + ["xlf_cs", "xli_cs"]

    # rebuild XLF / XLI raw trades and convert to daily pct with slippage
    hd = IronVault.instance()
    con = sqlite3.connect(hd._db_path)
    streams = {}
    for tk in ("XLF", "XLI"):
        raw = run_put_credit_spreads(con, tk)
        # Apply slippage by editing pnl_pct_capital in-place on copies
        adjusted = []
        cost_per_contract = 4.0 * (slip_cents / 100.0) * 100.0
        for t in raw:
            max_loss_per_spread = max((t.short_strike - t.long_strike) - t.net_credit, 0.01) * 100.0
            n_contracts = (0.02 * CAPITAL) / max_loss_per_spread
            adj_pnl_dollars = (t.pnl_per_spread * n_contracts) - cost_per_contract * n_contracts
            # build a lightweight stand-in object that quacks like SpreadTrade
            class _T: pass
            tt = _T()
            tt.entry_date = t.entry_date
            tt.expiration = t.expiration
            tt.pnl_pct_capital = adj_pnl_dollars / CAPITAL
            adjusted.append(tt)
        s = trades_to_daily_pct(adjusted, base.index)
        streams[f"{tk.lower()}_cs"] = s
    con.close()

    df = base.copy()
    for k, v in streams.items():
        df[k] = v.reindex(df.index).fillna(0.0)
    df = df[cols]

    # Equal weight metrics
    eq = df.mean(axis=1)
    eq_metrics = _series_metrics(eq, "equal_weight_7stream")
    # Min-var on a single training slice (full sample) — coarse approximation,
    # honest but not look-ahead-clean. Used here only to gauge sensitivity.
    cov = df.cov().values
    inv = np.linalg.pinv(cov)
    one = np.ones(len(cols))
    w = inv @ one / (one @ inv @ one)
    w = np.clip(w, 0, 1)
    if w.sum() > 0:
        w = w / w.sum()
    mv = (df.values @ w)
    mv = pd.Series(mv, index=df.index)
    mv_metrics = _series_metrics(mv, "min_var_7stream")
    return {
        "slip_cents": slip_cents,
        "equal_weight": eq_metrics,
        "min_var": mv_metrics,
        "min_var_weights": {cols[i]: round(float(w[i]), 4) for i in range(len(cols))},
    }


def _series_metrics(daily: pd.Series, label: str) -> Dict:
    daily = daily.dropna()
    if len(daily) < 30:
        return {"label": label, "n_days": 0}
    eq = (1 + daily).cumprod()
    yrs = len(daily) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    mu, sd = daily.mean(), daily.std(ddof=1)
    sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return {
        "label": label,
        "n_days": int(len(daily)),
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(float(sharpe), 3),
        "max_dd_pct": round(float(-dd.min() * 100), 3),
        "vol_pct": round(float(sd) * math.sqrt(TRADING_DAYS) * 100, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[1/5] re-running EXP-2160 XLF & XLI engines (real IronVault) …")
    hd = IronVault.instance()
    con = sqlite3.connect(hd._db_path)
    xlf_trades = run_put_credit_spreads(con, "XLF")
    xli_trades = run_put_credit_spreads(con, "XLI")

    print("[2/5] empirical liquidity proxies (real OHLC of used contracts) …")
    xlf_liq = liquidity_stats_for_trades(con, xlf_trades)
    xli_liq = liquidity_stats_for_trades(con, xli_trades)
    con.close()

    print("[3/5] slippage sweep …")
    xlf_sweep = slippage_sweep(xlf_trades, "XLF")
    xli_sweep = slippage_sweep(xli_trades, "XLI")
    for r in xlf_sweep["sweep"]:
        print(f"      XLF {r['slip_cents_per_leg']:>4.1f}c  Sharpe {r['sharpe']:5.2f}  CAGR {r['cagr_pct']:6.2f}%")
    for r in xli_sweep["sweep"]:
        print(f"      XLI {r['slip_cents_per_leg']:>4.1f}c  Sharpe {r['sharpe']:5.2f}  CAGR {r['cagr_pct']:6.2f}%")

    print("[4/5] trade-timing breakdown by entry weekday (clean fills) …")
    xlf_clean = apply_slippage(xlf_trades, 0.0)
    xli_clean = apply_slippage(xli_trades, 0.0)
    xlf_by_dow = by_weekday(xlf_clean, "pnl_pct_capital_clean")
    xli_by_dow = by_weekday(xli_clean, "pnl_pct_capital_clean")

    # Same with realistic slippage to see if any weekday survives
    realistic_cents = 3.0
    xlf_real = apply_slippage(xlf_trades, realistic_cents / 100.0)
    xli_real = apply_slippage(xli_trades, realistic_cents / 100.0)
    xlf_by_dow_real = by_weekday(xlf_real, "pnl_pct_capital_adj")
    xli_by_dow_real = by_weekday(xli_real, "pnl_pct_capital_adj")

    print("[5/5] 7-stream portfolio sensitivity to XLF/XLI slippage …")
    portfolio_sweep = []
    for cents in (0.0, 1.0, 2.0, 3.0, 5.0):
        port = build_seven_stream_cube_with_slippage(cents)
        portfolio_sweep.append(port)
        print(f"      slip {cents:>4.1f}c  EW Sharpe {port['equal_weight']['sharpe']:5.2f}  "
              f"MV Sharpe {port['min_var']['sharpe']:5.2f}")

    payload = {
        "experiment": "EXP-2270",
        "name": "XLF / XLI slippage impact analysis",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "data_limitation": (
            "IronVault has no bid/ask quotes and no intraday rows for "
            "XLF/XLI options — only daily OHLC, volume, and open interest. "
            "Slippage is therefore swept as an exogenous parameter rather "
            "than measured directly. Empirical liquidity proxies from the "
            "real OHLC of every contract actually traded are reported "
            "alongside, so the reader can pick a defensible slippage "
            "assumption from observable data."
        ),
        "rule_zero": "All trades and OHLC stats from real IronVault Polygon data.",
        "empirical_liquidity": {
            "XLF": xlf_liq,
            "XLI": xli_liq,
        },
        "slippage_sweep": {
            "XLF": xlf_sweep,
            "XLI": xli_sweep,
        },
        "weekday_breakdown_clean": {
            "XLF": xlf_by_dow,
            "XLI": xli_by_dow,
        },
        "weekday_breakdown_with_3c_slip": {
            "XLF": xlf_by_dow_real,
            "XLI": xli_by_dow_real,
        },
        "portfolio_sensitivity_7stream": portfolio_sweep,
        "answers": {
            "q1_realistic_bid_ask": (
                "Cannot be measured directly — IronVault has no quotes. "
                "Industry baseline for XLF/XLI ATM-ish options at the "
                "$0.10-$2.00 premium range is roughly $0.02-$0.05 wide, "
                "i.e. one-way per-leg slippage of 1-3 cents."
            ),
            "q2_xlf_break_even_sharpe_15": xlf_sweep["break_even_sharpe_15_cents"],
            "q2_xlf_break_even_sharpe_10": xlf_sweep["break_even_sharpe_10_cents"],
            "q2_xli_break_even_sharpe_15": xli_sweep["break_even_sharpe_15_cents"],
            "q2_xli_break_even_sharpe_10": xli_sweep["break_even_sharpe_10_cents"],
            "q3_portfolio_sharpe_at_3c_slip":
                portfolio_sweep[3]["min_var"]["sharpe"]
                if len(portfolio_sweep) > 3 else None,
            "q3_portfolio_sharpe_at_0c_slip":
                portfolio_sweep[0]["min_var"]["sharpe"]
                if portfolio_sweep else None,
            "q4_timing_finding": (
                "Per-weekday Sharpe (clean fills) is reported; if any single "
                "weekday clears 1.5 under realistic 3c slippage we have a "
                "tradable carve-out."
            ),
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    def _row(label: str, s: Dict) -> str:
        if not s or s.get("n", 0) == 0:
            return f"<tr><td>{label}</td><td colspan='4'>no data</td></tr>"
        return (f"<tr><td>{label}</td><td>{s['median']}</td><td>{s['p25']}</td>"
                f"<td>{s['p75']}</td><td>{s['mean']}</td></tr>")

    def _liq_block(name: str, liq: Dict) -> str:
        if not liq.get("n_active_rows"):
            return f"<h3>{name}</h3><p>no active rows</p>"
        return (f"<h3>{name}</h3>"
                f"<p class='small'>{liq['n_contracts']} contracts, "
                f"{liq['n_active_rows']} active days</p>"
                f"<table>"
                f"<tr><th>Stat</th><th>median</th><th>p25</th><th>p75</th><th>mean</th></tr>"
                f"{_row('volume / day', liq.get('volume_per_day', {}))}"
                f"{_row('open interest', liq.get('open_interest', {}))}"
                f"{_row('(high-low)/close', liq.get('high_low_over_close', {}))}"
                f"{_row('close premium $', liq.get('close_premium', {}))}"
                f"</table>")

    def _sweep_table(sweep: Dict) -> str:
        rows = "".join(
            f"<tr><td>{r['slip_cents_per_leg']:.1f}c</td><td>{r['n']}</td>"
            f"<td>{r['wr']*100:.1f}%</td><td>{r['sharpe']:.2f}</td>"
            f"<td>{r['cagr_pct']:.2f}%</td><td>{r['max_dd_pct']:.2f}%</td></tr>"
            for r in sweep["sweep"]
        )
        return (f"<h3>{sweep['ticker']} ({sweep['n_raw_trades']} trades)</h3>"
                f"<p>Sharpe break-even: 1.5 at {sweep['break_even_sharpe_15_cents']}c · "
                f"1.0 at {sweep['break_even_sharpe_10_cents']}c</p>"
                f"<table><tr><th>Slip</th><th>n</th><th>WR</th><th>Sharpe</th>"
                f"<th>CAGR</th><th>Max DD</th></tr>{rows}</table>")

    def _wd_table(name: str, wd: Dict) -> str:
        rows = "".join(
            f"<tr><td>{d}</td><td>{m['n']}</td><td>{m['wr']*100:.1f}%</td>"
            f"<td>{m['sharpe']:.2f}</td><td>{m['cagr_pct']:.2f}%</td>"
            f"<td>{m['max_dd_pct']:.2f}%</td></tr>"
            for d, m in wd.items()
        )
        return (f"<h4>{name}</h4>"
                f"<table><tr><th>Day</th><th>n</th><th>WR</th><th>Sharpe</th>"
                f"<th>CAGR</th><th>DD</th></tr>{rows}</table>")

    rows_port = "".join(
        f"<tr><td>{r['slip_cents']:.1f}c</td>"
        f"<td>{r['equal_weight']['sharpe']:.2f}</td>"
        f"<td>{r['equal_weight']['cagr_pct']:.2f}%</td>"
        f"<td>{r['equal_weight']['max_dd_pct']:.2f}%</td>"
        f"<td>{r['min_var']['sharpe']:.2f}</td>"
        f"<td>{r['min_var']['cagr_pct']:.2f}%</td>"
        f"<td>{r['min_var']['max_dd_pct']:.2f}%</td></tr>"
        for r in p["portfolio_sensitivity_7stream"]
    )
    a = p["answers"]

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2270 — XLF/XLI Slippage Impact</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .small{{color:#555;font-size:.88em}}
 .callout{{background:#fff8e1;border-left:4px solid #e0a500;padding:.8em 1em;margin:1em 0}}
</style></head><body>
<h1>EXP-2270 — XLF/XLI Slippage Impact Analysis</h1>
<p class='small'>Generated {p['generated']} · Rule-Zero clean
 (real IronVault OHLC, no fabricated quotes).</p>

<div class='callout'>
<b>Data limitation.</b> {p['data_limitation']}
</div>

<h2>1. Empirical liquidity (real OHLC of contracts actually traded)</h2>
{_liq_block('XLF', p['empirical_liquidity']['XLF'])}
{_liq_block('XLI', p['empirical_liquidity']['XLI'])}

<h2>2. Slippage sweep — when does Sharpe break?</h2>
{_sweep_table(p['slippage_sweep']['XLF'])}
{_sweep_table(p['slippage_sweep']['XLI'])}

<h2>3. 7-stream portfolio sensitivity</h2>
<table>
<tr><th>XLF/XLI slip</th><th colspan='3'>Equal-weight</th><th colspan='3'>Min-variance</th></tr>
<tr><th></th><th>Sharpe</th><th>CAGR</th><th>DD</th><th>Sharpe</th><th>CAGR</th><th>DD</th></tr>
{rows_port}
</table>

<h2>4. Trade-timing carve-outs</h2>
<h3>XLF</h3>
{_wd_table('clean fills', p['weekday_breakdown_clean']['XLF'])}
{_wd_table('with 3c slip', p['weekday_breakdown_with_3c_slip']['XLF'])}
<h3>XLI</h3>
{_wd_table('clean fills', p['weekday_breakdown_clean']['XLI'])}
{_wd_table('with 3c slip', p['weekday_breakdown_with_3c_slip']['XLI'])}

<h2>Answers</h2>
<ol>
<li><b>Realistic bid-ask:</b> {a['q1_realistic_bid_ask']}</li>
<li><b>XLF break-even:</b> Sharpe 1.5 at {a['q2_xlf_break_even_sharpe_15']}c, Sharpe 1.0 at {a['q2_xlf_break_even_sharpe_10']}c.<br>
    <b>XLI break-even:</b> Sharpe 1.5 at {a['q2_xli_break_even_sharpe_15']}c, Sharpe 1.0 at {a['q2_xli_break_even_sharpe_10']}c.</li>
<li><b>7-stream Sharpe @ 0c slip:</b> {a['q3_portfolio_sharpe_at_0c_slip']} → @ 3c slip: {a['q3_portfolio_sharpe_at_3c_slip']}.</li>
<li><b>Trade-timing:</b> {a['q4_timing_finding']}</li>
</ol>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
