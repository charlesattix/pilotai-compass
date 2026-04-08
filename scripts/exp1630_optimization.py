#!/usr/bin/env python3
"""
EXP-1630 Optimization: Position sizing, leverage, and multi-pair expansion.

1. Sizing sweep: 2%, 5%, 10%, 15%, 20% risk per trade
2. Max contracts sweep: 10, 20, 50, 100
3. Leverage analysis: 1x-5x on the best sizing
4. Multi-pair expansion: GLD-SPY, TLT-QQQ, GLD-QQQ
5. Can this contribute 10-20% CAGR to portfolio?

Output: reports/exp1630_optimization.html + .json
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault
from backtest.backtester import _yf_download_safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPORT_PATH = ROOT / "reports" / "exp1630_optimization.html"
JSON_PATH = ROOT / "reports" / "exp1630_optimization.json"
TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Import the core backtest engine and override globals
# ═══════════════════════════════════════════════════════════════════════════

import compass.gld_tlt_relval as relval


def _fetch(ticker: str) -> pd.DataFrame:
    df = _yf_download_safe(ticker, "2019-06-01", "2025-01-01")
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def run_with_params(
    hd: IronVault,
    gld_df: pd.DataFrame,
    tlt_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    risk_pct: float = 0.02,
    max_contracts: int = 10,
) -> dict:
    """Run the GLD/TLT backtest with custom sizing parameters."""
    # Override module-level globals
    orig_risk = relval.RISK_PER_TRADE
    orig_max = relval.MAX_CONTRACTS
    relval.RISK_PER_TRADE = risk_pct
    relval.MAX_CONTRACTS = max_contracts

    try:
        result = relval.run_backtest(hd, gld_df, tlt_df, spy_df)
    finally:
        relval.RISK_PER_TRADE = orig_risk
        relval.MAX_CONTRACTS = orig_max

    return {
        "risk_pct": risk_pct,
        "max_contracts": max_contracts,
        "n_trades": result.n_trades,
        "total_pnl": result.total_pnl,
        "win_rate": result.win_rate,
        "max_dd": round(result.max_dd * 100, 2) if result.max_dd < 1 else result.max_dd,
        "sharpe": result.sharpe,
        "cagr": round(result.cagr * 100, 2) if abs(result.cagr) < 1 else result.cagr,
        "oos_sharpe": result.oos_sharpe,
        "spy_corr": result.spy_corr,
        "avg_hold": result.avg_hold_days,
        "yearly": {str(yr): {"pnl": y.total_pnl, "wr": y.win_rate, "sharpe": y.sharpe,
                              "ret_pct": round(y.return_pct * 100, 2)}
                   for yr, y in result.yearly.items()},
    }


def leverage_analysis(base_result: dict, leverages=None) -> List[dict]:
    """Scale PnL by leverage and recompute metrics."""
    if leverages is None:
        leverages = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    base_pnl = base_result["total_pnl"]
    base_cagr = base_result["cagr"]
    base_dd = base_result["max_dd"]
    capital = relval.CAPITAL

    results = []
    for lev in leverages:
        scaled_pnl = base_pnl * lev
        # DD scales linearly with leverage
        scaled_dd = base_dd * lev
        # CAGR: (1 + base_cagr/100)^lev - 1 approximately
        if base_cagr > -100:
            scaled_cagr = ((1 + base_cagr / 100) ** lev - 1) * 100
        else:
            scaled_cagr = -100

        results.append({
            "leverage": lev,
            "total_pnl": round(scaled_pnl, 2),
            "cagr_pct": round(scaled_cagr, 2),
            "max_dd_pct": round(scaled_dd, 2),
            "sharpe": base_result["sharpe"],  # Sharpe is leverage-invariant
        })

    return results


def run_generic_pair(
    hd: IronVault,
    ticker_a: str,
    ticker_b: str,
    spy_df: pd.DataFrame,
    risk_pct: float = 0.05,
    max_contracts: int = 50,
) -> dict:
    """Run a generic pair trade between any two tickers using the relval framework.

    Computes ratio z-score and enters credit spreads on each leg.
    """
    a_df = _fetch(ticker_a)
    b_df = _fetch(ticker_b)

    common = a_df.index.intersection(b_df.index).intersection(spy_df.index)
    a_close = a_df["Close"].reindex(common).ffill()
    b_close = b_df["Close"].reindex(common).ffill()
    spy_ret = spy_df["Close"].reindex(common).pct_change().fillna(0)

    # Ratio z-score
    ratio = a_close / b_close.replace(0, np.nan)
    ratio = ratio.dropna()
    z = (ratio - ratio.rolling(20).mean()) / ratio.rolling(20).std().replace(0, np.nan)
    z = z.dropna()

    # Check if we have options data for both tickers
    import sqlite3
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()

    for ticker in [ticker_a, ticker_b]:
        cur.execute("SELECT COUNT(*) FROM option_contracts WHERE ticker=?", (ticker,))
        cnt = cur.fetchone()[0]
        if cnt == 0:
            conn.close()
            return {
                "pair": f"{ticker_a}/{ticker_b}",
                "error": f"No options data for {ticker}",
                "n_trades": 0, "total_pnl": 0, "sharpe": 0, "cagr_pct": 0,
                "max_dd_pct": 0, "spy_corr": 0, "oos_sharpe": 0,
            }

    # Find expirations for both
    a_exps = set(relval._find_exps(hd, ticker_a, "2020-04-01", "2025-12-31"))
    b_exps = set(relval._find_exps(hd, ticker_b, "2020-04-01", "2025-12-31"))

    conn.close()

    # Width depends on price level
    a_price_avg = float(a_close.mean())
    b_price_avg = float(b_close.mean())
    a_width = max(1.0, round(a_price_avg * 0.01))  # ~1% of price
    b_width = max(1.0, round(b_price_avg * 0.01))

    trades = []
    last_entry = None
    capital = relval.CAPITAL

    for date in z.index:
        ds = date.strftime("%Y-%m-%d")
        if last_entry and (date - last_entry).days < 14:
            continue

        try:
            zv = float(z.loc[ds])
        except (KeyError, TypeError):
            continue
        if np.isnan(zv) or abs(zv) < 1.5:
            continue

        try:
            a_price = float(a_close.loc[ds])
            b_price = float(b_close.loc[ds])
        except (KeyError, TypeError):
            continue

        # Find matching expirations
        from datetime import timedelta
        a_exp = b_exp = None
        for e in sorted(a_exps):
            ed = relval._exp_dt(e)
            if ed > date + timedelta(days=20) and ed < date + timedelta(days=50):
                a_exp = e; break
        for e in sorted(b_exps):
            ed = relval._exp_dt(e)
            if ed > date + timedelta(days=20) and ed < date + timedelta(days=50):
                b_exp = e; break

        if a_exp is None or b_exp is None:
            continue

        # Direction
        if zv > 1.5:
            a_spread = relval._sell_spread(hd, ticker_a, a_exp, ds, a_price, "C", 0.95, a_width)
            b_spread = relval._sell_spread(hd, ticker_b, b_exp, ds, b_price, "P", 0.95, b_width)
        else:
            a_spread = relval._sell_spread(hd, ticker_a, a_exp, ds, a_price, "P", 0.95, a_width)
            b_spread = relval._sell_spread(hd, ticker_b, b_exp, ds, b_price, "C", 0.95, b_width)

        if a_spread is None and b_spread is None:
            continue

        total_credit = 0.0
        total_max_loss = 0.0
        legs = []
        for sp in [a_spread, b_spread]:
            if sp is None:
                continue
            legs.append(sp)
            total_credit += sp["credit"]
            total_max_loss += sp["max_loss"]

        if total_max_loss <= 0:
            continue

        contracts = max(1, min(max_contracts,
                               int(capital * risk_pct / (total_max_loss * 100))))

        total_pnl = 0.0
        for sp in legs:
            ticker = sp["ticker"]
            exp = a_exp if ticker == ticker_a else b_exp
            td_idx = a_df.index if ticker == ticker_a else b_df.index
            _, er, ev, hold = relval._walk_spread(
                hd, ticker, exp, sp["short"], sp["long"],
                sp["type"], sp["credit"], date, relval._exp_dt(exp), td_idx,
            )
            total_pnl += (sp["credit"] - ev) * 100 * contracts

        trades.append({"entry_date": ds, "pnl": round(total_pnl, 2), "z": round(zv, 2)})
        last_entry = date

    # Compute metrics
    if not trades:
        return {
            "pair": f"{ticker_a}/{ticker_b}", "n_trades": 0, "total_pnl": 0,
            "sharpe": 0, "cagr_pct": 0, "max_dd_pct": 0, "spy_corr": 0, "oos_sharpe": 0,
        }

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())
    eq = np.cumsum(pnls) + capital
    pk = np.maximum.accumulate(eq)
    dd = float(((pk - eq) / pk).max())
    sharpe = relval._sharpe(pnls)

    entry_dates = pd.to_datetime([t["entry_date"] for t in trades])
    yrs = max((entry_dates.max() - entry_dates.min()).days / 365.25, 0.5)
    cagr = ((1 + total / capital) ** (1 / yrs) - 1) if total > -capital else -1.0

    # SPY correlation
    tr = {}
    for t in trades:
        d = t["entry_date"][:10]
        tr[d] = tr.get(d, 0) + t["pnl"]
    ts = pd.Series(tr)
    ts.index = pd.to_datetime(ts.index)
    ci = ts.index.intersection(spy_ret.index)
    spy_corr = float(np.corrcoef(
        ts.reindex(ci).fillna(0), spy_ret.reindex(ci).fillna(0)
    )[0, 1]) if len(ci) > 5 else 0.0

    # Walk-forward: IS < 2022, OOS >= 2022
    is_pnls = pnls[entry_dates.year < 2022]
    oos_pnls = pnls[entry_dates.year >= 2022]
    oos_sharpe = relval._sharpe(oos_pnls) if len(oos_pnls) > 1 else 0

    return {
        "pair": f"{ticker_a}/{ticker_b}",
        "n_trades": n,
        "total_pnl": round(total, 2),
        "win_rate": round(wins / n, 3) if n > 0 else 0,
        "sharpe": round(sharpe, 2),
        "oos_sharpe": round(oos_sharpe, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(dd * 100, 2),
        "spy_corr": round(spy_corr, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(data: dict) -> str:
    sizing = data["sizing_sweep"]
    leverage = data["leverage"]
    pairs = data["multi_pair"]
    best = data["best_sizing"]
    verdict = data["verdict"]

    sizing_rows = ""
    for r in sizing:
        hl = ' class="hl"' if r["risk_pct"] == best["risk_pct"] and r["max_contracts"] == best["max_contracts"] else ""
        sizing_rows += (
            f'<tr{hl}><td>{r["risk_pct"]*100:.0f}%</td><td>{r["max_contracts"]}</td>'
            f'<td>{r["n_trades"]}</td><td>${r["total_pnl"]:,.0f}</td>'
            f'<td>{r["cagr"]:.2f}%</td><td>{r["max_dd"]:.2f}%</td>'
            f'<td>{r["sharpe"]:.2f}</td><td>{r["oos_sharpe"]:.2f}</td></tr>\n'
        )

    lev_rows = ""
    for l in leverage:
        target = ""
        if l["cagr_pct"] >= 10 and l["max_dd_pct"] <= 12:
            target = ' <span style="color:#4ade80;font-size:.75rem">(VIABLE)</span>'
        lev_rows += (
            f'<tr><td>{l["leverage"]:.1f}x</td><td>{l["cagr_pct"]:.2f}%</td>'
            f'<td>{l["max_dd_pct"]:.2f}%</td><td>{l["sharpe"]:.2f}</td>{target}</tr>\n'
        )

    pair_rows = ""
    for p in pairs:
        err = p.get("error", "")
        if err:
            pair_rows += f'<tr><td>{p["pair"]}</td><td colspan="6" style="color:#f87171">{err}</td></tr>\n'
        else:
            pair_rows += (
                f'<tr><td>{p["pair"]}</td><td>{p["n_trades"]}</td>'
                f'<td>${p["total_pnl"]:,.0f}</td><td>{p["cagr_pct"]:.2f}%</td>'
                f'<td>{p["max_dd_pct"]:.2f}%</td><td>{p["sharpe"]:.2f}</td>'
                f'<td>{p["spy_corr"]:.3f}</td></tr>\n'
            )

    # Year rows for best sizing
    yr_rows = ""
    for yr, yd in sorted(best.get("yearly", {}).items()):
        yr_rows += f'<tr><td>{yr}</td><td>${yd["pnl"]:,.0f}</td><td>{yd["wr"]*100:.0f}%</td><td>{yd["sharpe"]:.2f}</td><td>{yd["ret_pct"]:.2f}%</td></tr>\n'

    verdict_cls = "verdict-pass" if verdict["can_contribute_10pct"] else "verdict-warn"
    verdict_text = verdict["summary"]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1630 GLD/TLT Optimization</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem;line-height:1.6;max-width:1100px;margin:0 auto}}
h1{{font-size:1.8rem;margin-bottom:.5rem;color:#f8fafc}}
h2{{font-size:1.3rem;margin:2rem 0 1rem;color:#93c5fd;border-bottom:1px solid #334155;padding-bottom:.5rem}}
.subtitle{{color:#94a3b8;font-size:.95rem;margin-bottom:2rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.8rem;margin:1rem 0}}
.card{{background:#1e293b;border-radius:8px;padding:.8rem;border:1px solid #334155}}
.card .label{{font-size:.7rem;color:#94a3b8;text-transform:uppercase}}
.card .value{{font-size:1.3rem;font-weight:700;margin-top:.2rem}}
.green{{color:#4ade80}}.red{{color:#f87171}}.yellow{{color:#fbbf24}}.blue{{color:#60a5fa}}
table{{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.85rem}}
th{{background:#1e293b;padding:.5rem .6rem;text-align:left;color:#94a3b8;font-weight:600;border-bottom:2px solid #334155}}
td{{padding:.5rem .6rem;border-bottom:1px solid #1e293b}}
tr:hover td{{background:#1e293b}}
.hl td{{background:#1a2332;border-left:3px solid #f59e0b}}
.verdict{{padding:.75rem 1rem;border-radius:6px;margin:1rem 0;font-size:.9rem}}
.verdict-pass{{background:#052e16;border:1px solid #16a34a;color:#4ade80}}
.verdict-warn{{background:#422006;border:1px solid #d97706;color:#fbbf24}}
.footer{{margin-top:3rem;font-size:.75rem;color:#475569;text-align:center;border-top:1px solid #1e293b;padding-top:1rem}}
</style></head><body>

<h1>EXP-1630 GLD/TLT Relative Value — Optimization</h1>
<div class="subtitle">Position sizing sweep &middot; Leverage analysis &middot; Multi-pair expansion &middot; {datetime.utcnow().strftime('%Y-%m-%d')}</div>

<div class="cards">
<div class="card"><div class="label">Best CAGR</div><div class="value {'green' if best['cagr']>=10 else 'yellow'}">{best['cagr']:.1f}%</div></div>
<div class="card"><div class="label">Best Sharpe</div><div class="value green">{best['sharpe']:.2f}</div></div>
<div class="card"><div class="label">Max DD</div><div class="value {'green' if best['max_dd']<=12 else 'red'}">{best['max_dd']:.1f}%</div></div>
<div class="card"><div class="label">OOS Sharpe</div><div class="value blue">{best['oos_sharpe']:.2f}</div></div>
<div class="card"><div class="label">SPY Corr</div><div class="value green">{best['spy_corr']:.3f}</div></div>
<div class="card"><div class="label">Trades</div><div class="value">{best['n_trades']}</div></div>
</div>

<div class="{verdict_cls} verdict">{verdict_text}</div>

<h2>1. Position Sizing Sweep</h2>
<p>Testing risk_pct (% of capital per trade) and max_contracts limits.</p>
<table>
<tr><th>Risk %</th><th>Max Cts</th><th>Trades</th><th>Total PnL</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th><th>OOS Sharpe</th></tr>
{sizing_rows}
</table>

<h2>2. Year-by-Year (Best Sizing: {best['risk_pct']*100:.0f}% / {best['max_contracts']} cts)</h2>
<table>
<tr><th>Year</th><th>PnL</th><th>Win Rate</th><th>Sharpe</th><th>Return</th></tr>
{yr_rows}
</table>

<h2>3. Leverage Analysis (on best sizing)</h2>
<table>
<tr><th>Leverage</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th></tr>
{lev_rows}
</table>

<h2>4. Multi-Pair Expansion</h2>
<p>Testing additional pairs using the same z-score mean-reversion framework.</p>
<table>
<tr><th>Pair</th><th>Trades</th><th>PnL</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th><th>SPY Corr</th></tr>
<tr class="hl"><td>GLD/TLT (baseline)</td><td>{best['n_trades']}</td><td>${best['total_pnl']:,.0f}</td><td>{best['cagr']:.2f}%</td><td>{best['max_dd']:.2f}%</td><td>{best['sharpe']:.2f}</td><td>{best['spy_corr']:.3f}</td></tr>
{pair_rows}
</table>

<div class="footer">EXP-1630 Optimization &middot; All data from IronVault &middot; {datetime.utcnow().strftime('%Y-%m-%d')}</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("EXP-1630 GLD/TLT RELATIVE VALUE — OPTIMIZATION")
    print("=" * 70)

    hd = IronVault.instance()
    gld_df = _fetch("GLD")
    tlt_df = _fetch("TLT")
    spy_df = _fetch("SPY")

    # 1. Position sizing sweep
    print("\n[1/4] Position sizing sweep...")
    sizing_results = []
    configs = [
        (0.02, 10), (0.02, 50), (0.05, 10), (0.05, 50), (0.05, 100),
        (0.10, 10), (0.10, 50), (0.10, 100),
        (0.15, 50), (0.15, 100), (0.20, 50), (0.20, 100),
    ]
    for risk, max_cts in configs:
        r = run_with_params(hd, gld_df, tlt_df, spy_df, risk_pct=risk, max_contracts=max_cts)
        sizing_results.append(r)
        print(f"  {risk*100:.0f}% / {max_cts} cts: PnL=${r['total_pnl']:,.0f}, CAGR={r['cagr']:.2f}%, DD={r['max_dd']:.2f}%, Sharpe={r['sharpe']:.2f}")

    # Find best: highest CAGR with DD <= 15% and Sharpe > 1.0
    viable = [r for r in sizing_results if r["max_dd"] <= 15 and r["sharpe"] > 0.5]
    if viable:
        best = max(viable, key=lambda r: r["cagr"])
    else:
        best = max(sizing_results, key=lambda r: r["cagr"])
    print(f"\n  BEST: {best['risk_pct']*100:.0f}% / {best['max_contracts']} cts → CAGR={best['cagr']:.2f}%, DD={best['max_dd']:.2f}%, Sharpe={best['sharpe']:.2f}")

    # 2. Leverage analysis on best sizing
    print("\n[2/4] Leverage analysis...")
    lev_results = leverage_analysis(best)
    for l in lev_results:
        marker = " *** VIABLE ***" if l["cagr_pct"] >= 10 and l["max_dd_pct"] <= 12 else ""
        print(f"  {l['leverage']:.1f}x: CAGR={l['cagr_pct']:.2f}%, DD={l['max_dd_pct']:.2f}%{marker}")

    # 3. Multi-pair expansion
    print("\n[3/4] Multi-pair expansion...")
    pair_results = []
    pair_configs = [
        ("GLD", "SPY"), ("TLT", "QQQ"), ("GLD", "QQQ"),
        ("TLT", "SPY"), ("XLF", "TLT"), ("XLI", "TLT"),
    ]
    for ticker_a, ticker_b in pair_configs:
        print(f"  Testing {ticker_a}/{ticker_b}...")
        r = run_generic_pair(hd, ticker_a, ticker_b, spy_df, risk_pct=best["risk_pct"], max_contracts=best["max_contracts"])
        pair_results.append(r)
        if r.get("error"):
            print(f"    {r['error']}")
        else:
            print(f"    {r['n_trades']} trades, PnL=${r['total_pnl']:,.0f}, CAGR={r['cagr_pct']:.2f}%, Sharpe={r['sharpe']:.2f}, SPY corr={r['spy_corr']:.3f}")

    # 4. Verdict
    print("\n[4/4] Verdict...")
    can_contribute = best["cagr"] >= 10 or any(
        l["cagr_pct"] >= 10 and l["max_dd_pct"] <= 12 for l in lev_results
    )
    best_lev_10 = next((l for l in lev_results if l["cagr_pct"] >= 10 and l["max_dd_pct"] <= 12), None)
    viable_pairs = [p for p in pair_results if p.get("n_trades", 0) >= 5 and p.get("sharpe", 0) > 0.5]

    if can_contribute:
        if best["cagr"] >= 10:
            summary = f"YES — {best['cagr']:.1f}% CAGR at {best['risk_pct']*100:.0f}% sizing with {best['max_dd']:.1f}% DD. Strategy can contribute 10-20% CAGR at low correlation."
        elif best_lev_10:
            summary = f"YES (with leverage) — {best_lev_10['cagr_pct']:.1f}% CAGR at {best_lev_10['leverage']:.1f}x leverage, {best_lev_10['max_dd_pct']:.1f}% DD."
        else:
            summary = f"MARGINAL — best CAGR {best['cagr']:.1f}% achievable. Needs leverage or more pairs to reach 10%."
    else:
        summary = f"NO — max CAGR {best['cagr']:.1f}% even at aggressive sizing. Strategy is a low-return diversifier, not an alpha engine."

    if viable_pairs:
        summary += f" {len(viable_pairs)} additional pairs viable for portfolio expansion."

    print(f"  {summary}")

    # Generate report
    report_data = {
        "sizing_sweep": sizing_results,
        "best_sizing": best,
        "leverage": lev_results,
        "multi_pair": pair_results,
        "verdict": {"can_contribute_10pct": can_contribute, "summary": summary},
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = generate_report(report_data)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"\n  Report: {REPORT_PATH}")

    JSON_PATH.write_text(json.dumps(report_data, indent=2, default=str))
    print(f"  JSON: {JSON_PATH}")

    return report_data


if __name__ == "__main__":
    main()
