"""
compass/earnings_crush.py — EXP-1800 Systematic Earnings Vol Crush.

Per task: sell straddles/iron condors on AAPL, MSFT, AMZN, GOOGL, TSLA, NVDA,
META 1-2 days before earnings, close after earnings. Real earnings dates via
yfinance (lxml now installed). Real option data from IronVault or Yahoo.

HONEST DATA REALITY CHECK (Rule Zero):

  yfinance.Ticker.earnings_dates was verified working after installing lxml.
  However, it only returns ~25 dates per ticker (most recent 4 quarters plus
  the next scheduled date), NOT the full 2020-2025 history this backtest
  requires. Source: verified 2026-04-06 with
      >>> yf.Ticker('AAPL').earnings_dates
      DataFrame with 25 rows, earliest ~2023-11
  For the 2020-2022 portion we fall back to SEC 8-K filing dates that are
  already hardcoded in compass/earnings_vol_crush.py. These are public
  historical facts, not synthetic data.

  IronVault contains options only for 9 ETFs/indices (SPY, QQQ, GLD, TLT,
  XLE, XLF, XLI, XLK, SOXX). The classic single-name strategy (sell AAPL
  earnings straddles) CANNOT be tested on our data — 0 contracts for all
  7 target names. This is documented in the Wave 1 post-mortem as the
  blocker for Gao-Xing 2020 style tests.

  THIS MODULE DOES TWO THINGS:
    1. Harvests real earnings dates via yfinance (+ hardcoded SEC fallback)
    2. Runs the SPY/QQQ index-proxy backtest (since the constituents move
       the index materially — Magnificent 7 ≈ 30% of SPY, 45% of QQQ)

  A proper single-name backtest is a Phase 8 project blocked on Polygon
  Options tier data ($29-199/mo) for single-name option history.

DATA SOURCES (all REAL, cited):
  - Earnings dates: yfinance.Ticker.earnings_dates (recent) + hardcoded SEC
    8-K filing dates (historical). Both are public record.
  - Option prices: IronVault options_cache.db (real Polygon data).
  - Underlying prices: Yahoo Finance chart API.

NO SYNTHETIC DATA. All prices real. Sharpe via compass/metrics.py.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault
from backtest.backtester import _yf_download_safe

# Reuse the index-proxy backtest engine built in earnings_vol_crush.py
from compass.earnings_vol_crush import (
    EARNINGS_DATES as HARDCODED_EARNINGS,
    TICKER_TO_INDEX,
    run_earnings_vol_crush,
    compute_metrics,
    generate_html,
    REPORT_PATH as VCR_REPORT,
    JSON_PATH as VCR_JSON,
    CAPITAL,
    OOS_START,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("earnings_crush")

REPORT_PATH = ROOT / "reports" / "exp1800_earnings_crush.html"
JSON_PATH = ROOT / "reports" / "exp1800_earnings_crush.json"

TARGET_NAMES = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "META"]


# ═══════════════════════════════════════════════════════════════════════════
# Earnings date harvesting — real yfinance + SEC hardcoded fallback
# ═══════════════════════════════════════════════════════════════════════════

def fetch_yfinance_earnings(ticker: str) -> List[str]:
    """Get real earnings dates from yfinance.Ticker.earnings_dates.

    Returns ISO date strings. Returns empty list if lxml missing or API fails.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or len(ed) == 0:
            return []
        dates = []
        for idx in ed.index:
            try:
                # Index is a pandas Timestamp with timezone
                d = idx.date() if hasattr(idx, 'date') else datetime.strptime(str(idx)[:10], "%Y-%m-%d").date()
                dates.append(d.strftime("%Y-%m-%d"))
            except Exception:
                continue
        return sorted(set(dates))
    except ImportError as e:
        log.warning(f"yfinance/lxml import failed for {ticker}: {e}")
        return []
    except Exception as e:
        log.warning(f"yfinance earnings_dates failed for {ticker}: {e}")
        return []


def build_earnings_universe() -> Dict[str, List[str]]:
    """Build the full earnings date universe by merging yfinance + SEC hardcoded.

    yfinance provides recent data (last ~4 quarters). SEC hardcoded provides
    historical. Union of both gives the longest possible history.
    """
    universe: Dict[str, List[str]] = {}

    for ticker in TARGET_NAMES:
        # Start with hardcoded SEC data (historical 2020-2025)
        sec_dates = set(HARDCODED_EARNINGS.get(ticker, []))

        # Merge with yfinance recent data
        yf_dates = set(fetch_yfinance_earnings(ticker))

        all_dates = sorted(sec_dates | yf_dates)
        # Filter to 2020-01-01 through today
        all_dates = [d for d in all_dates
                     if "2020-01-01" <= d <= datetime.now().strftime("%Y-%m-%d")]

        universe[ticker] = all_dates
        log.info(f"  {ticker}: {len(sec_dates)} SEC + {len(yf_dates)} yfinance "
                  f"= {len(all_dates)} unique dates")

    return universe


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("EXP-1800 Earnings Vol Crush — yfinance + SEC dates, IronVault options")
    log.info("Rule Zero: 100% real data")
    log.info("=" * 70)

    # Step 1: Harvest real earnings dates
    log.info("\nStep 1: Harvesting earnings dates (yfinance + SEC hardcoded)...")
    earnings_universe = build_earnings_universe()

    n_events = sum(len(dates) for dates in earnings_universe.values())
    log.info(f"\nTotal earnings events: {n_events}")

    # Step 2: Data reality check for single-name options
    log.info("\nStep 2: Data reality check — single-name options in IronVault...")
    hd = IronVault.instance()
    import sqlite3
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    single_name_count = {}
    for ticker in TARGET_NAMES:
        cur.execute("SELECT COUNT(*) FROM option_contracts WHERE ticker=?", (ticker,))
        single_name_count[ticker] = cur.fetchone()[0]
    conn.close()

    total_single_name_contracts = sum(single_name_count.values())
    log.info(f"Single-name option contracts in IronVault: {total_single_name_contracts}")
    for t, n in single_name_count.items():
        log.info(f"  {t}: {n}")

    if total_single_name_contracts == 0:
        log.warning("=" * 70)
        log.warning("DATA GAP: IronVault has ZERO single-name options for target tickers.")
        log.warning("Cannot test the classic single-name earnings straddle strategy.")
        log.warning("Falling back to SPY/QQQ index proxy (constituents are ~30% of SPY,")
        log.warning("~45% of QQQ by weight — earnings DO move the index).")
        log.warning("=" * 70)

    # Step 3: Monkey-patch the earnings universe into the existing backtest engine
    # The run_earnings_vol_crush function iterates globals EARNINGS_DATES by default.
    # We pass the merged universe in via module state.
    import compass.earnings_vol_crush as evc
    original_earnings = evc.EARNINGS_DATES
    evc.EARNINGS_DATES = earnings_universe

    try:
        log.info("\nStep 3: Loading SPY and QQQ price data from Yahoo...")
        spy_df = _yf_download_safe("SPY", "2019-06-01", "2026-07-01")
        qqq_df = _yf_download_safe("QQQ", "2019-06-01", "2026-07-01")
        for df in (spy_df, qqq_df):
            df.index = pd.to_datetime(df.index)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        log.info(f"  SPY: {spy_df.index.min().date()} → {spy_df.index.max().date()}")
        log.info(f"  QQQ: {qqq_df.index.min().date()} → {qqq_df.index.max().date()}")

        log.info("\nStep 4: Running SPY/QQQ earnings vol crush backtest...")
        trades, skip = run_earnings_vol_crush(hd, spy_df, qqq_df)

        log.info("\nStep 5: Computing metrics...")
        metrics = compute_metrics(trades, spy_df)

    finally:
        evc.EARNINGS_DATES = original_earnings

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("EXP-1800 RESULTS")
    log.info("=" * 70)
    log.info(f"N events:          {n_events}")
    log.info(f"Successful trades: {metrics['n_trades']}")
    log.info(f"Total PnL:         ${metrics['total_pnl']:,.0f}")
    log.info(f"Win rate:          {metrics['win_rate']:.0%}")
    log.info(f"Trade Sharpe:      {metrics['trade_sharpe']:.2f}")
    log.info(f"Daily Sharpe:      {metrics['sharpe_arith']:.2f}")
    log.info(f"CAGR (daily):      {metrics['cagr']:.2%}")
    log.info(f"Max DD:            {metrics['max_dd']:.2%}")
    log.info(f"OOS Sharpe:        {metrics['oos_sharpe']:.2f} ({metrics['oos_n']} trades)")
    log.info(f"SPY correlation:   {metrics['spy_corr']:+.3f}")
    log.info(f"EXP-1220 corr:     {metrics['exp1220_corr']:+.3f}")
    log.info("")
    log.info("Per-ticker PnL:")
    for ticker, stats in sorted(metrics.get("by_ticker", {}).items()):
        log.info(f"  {ticker:6s} ({TICKER_TO_INDEX.get(ticker, '?')}) "
                  f"N={stats['n']:3d} PnL=${stats['pnl']:>7,.0f} WR={stats['wr']:.0%}")

    # Write reports
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = _generate_exp1800_html(metrics, skip, n_events, single_name_count,
                                     earnings_universe)
    REPORT_PATH.write_text(html, encoding="utf-8")
    log.info(f"\nHTML: {REPORT_PATH}")

    json_data = {
        "experiment": "EXP-1800",
        "name": "Systematic Earnings Vol Crush",
        "data_source": "yfinance earnings_dates + SEC 8-K dates + IronVault SPY/QQQ options",
        "rule_zero_compliant": True,
        "single_name_options_available": False,
        "single_name_contracts_in_ironvault": single_name_count,
        "earnings_universe_size": n_events,
        "earnings_per_ticker": {k: len(v) for k, v in earnings_universe.items()},
        "target_tickers": TARGET_NAMES,
        "ticker_to_index": TICKER_TO_INDEX,
        "skip_reasons": skip,
        "metrics": metrics,
        "n_trades": metrics["n_trades"],
        "total_pnl": metrics["total_pnl"],
        "trade_sharpe": metrics["trade_sharpe"],
        "oos_sharpe": metrics["oos_sharpe"],
        "spy_correlation": metrics["spy_corr"],
        "exp1220_correlation": metrics["exp1220_corr"],
    }
    JSON_PATH.write_text(json.dumps(json_data, indent=2, default=str))
    log.info(f"JSON: {JSON_PATH}")


def _generate_exp1800_html(metrics: Dict, skip: Dict, n_events: int,
                             single_name_count: Dict, universe: Dict) -> str:
    """EXP-1800 specific HTML with data gap disclosure."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    rows = ""
    for ticker, stats in sorted(metrics.get("by_ticker", {}).items()):
        c = "var(--green)" if stats["pnl"] > 0 else "var(--red)"
        rows += (
            f'<tr><td><strong>{ticker}</strong></td>'
            f'<td>{TICKER_TO_INDEX.get(ticker, "?")}</td>'
            f'<td>{len(universe.get(ticker, []))}</td>'
            f'<td>{single_name_count.get(ticker, 0)}</td>'
            f'<td>{stats["n"]}</td>'
            f'<td style="color:{c}">${stats["pnl"]:,.0f}</td>'
            f'<td>{stats["wr"]:.0%}</td>'
            f'<td>${stats["avg_pnl"]:,.0f}</td></tr>\n'
        )

    yr_rows = ""
    for yr, stats in sorted(metrics.get("yearly", {}).items()):
        tag = "OOS" if yr >= OOS_START else "IS"
        c = "var(--green)" if stats["pnl"] > 0 else "var(--red)"
        yr_rows += (
            f'<tr><td>{yr} ({tag})</td><td>{stats["n"]}</td>'
            f'<td style="color:{c}">${stats["pnl"]:,.0f}</td>'
            f'<td>{stats["wr"]:.0%}</td></tr>\n'
        )

    verdict_class = "callout-red"
    verdict = "KILL — OOS Sharpe below threshold"
    if metrics["oos_sharpe"] >= 1.0:
        verdict_class = "callout-green"
        verdict = "PROMISING"
    elif metrics["oos_sharpe"] >= 0.5:
        verdict_class = "callout-yellow"
        verdict = "MARGINAL"
    elif metrics["n_trades"] < 20:
        verdict_class = "callout-yellow"
        verdict = "INSUFFICIENT DATA"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EXP-1800 Earnings Crush</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e5e7eb;--text:#111827;--muted:#6b7280;--green:#059669;--red:#dc2626;--yellow:#d97706;--blue:#2563eb}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;max-width:1100px;margin:0 auto;padding:24px}}
h1{{font-size:1.5rem;font-weight:800}}
h2{{font-size:1.1rem;font-weight:700;margin:28px 0 12px;border-bottom:2px solid var(--border);padding-bottom:6px}}
.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:14px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center}}
.c .l{{color:var(--muted);font-size:.68rem;font-weight:600;text-transform:uppercase}}
.c .v{{font-size:1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:.82rem}}
th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid var(--border)}}
th{{background:#f1f5f9;color:var(--muted);font-size:.68rem;font-weight:600;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
.callout{{padding:14px;margin:14px 0;border-radius:8px;font-size:.88rem;line-height:1.7}}
.callout-red{{background:#fef2f2;border-left:4px solid var(--red)}}
.callout-yellow{{background:#fffbeb;border-left:4px solid var(--yellow)}}
.callout-green{{background:#ecfdf5;border-left:4px solid var(--green)}}
.callout-blue{{background:#eff6ff;border-left:4px solid var(--blue)}}
.footer{{margin-top:36px;text-align:center;font-size:.72rem;color:var(--muted);border-top:1px solid var(--border);padding-top:14px}}
</style></head><body>

<h1>EXP-1800: Systematic Earnings Volatility Crush</h1>
<div class="subtitle">{ts} &bull; Rule Zero: 100% real data (yfinance + SEC + IronVault) &bull; Zero synthetic</div>

<div class="callout callout-blue">
<strong>Honest data gap disclosure (Rule Zero):</strong>
<br><br>
The task asked for single-name straddles on AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA. I installed
lxml (successfully) and verified yfinance.Ticker.earnings_dates works, but IronVault contains
<strong>ZERO single-name option contracts</strong> for all 7 target tickers. The database only has
9 ETFs/indices: SPY, QQQ, GLD, TLT, XLE, XLF, XLI, XLK, SOXX.
<br><br>
<strong>What was done instead:</strong> Harvested real earnings dates (yfinance recent + SEC 8-K
historical) and backtested SPY/QQQ iron condors around those dates. The Magnificent 7 are ~30% of
SPY and ~45% of QQQ by weight, so their earnings materially move index IV. This is a proxy but
uses 100% real data.
<br><br>
<strong>What's still needed for the classic strategy:</strong> Polygon Options tier ($29-199/mo)
for single-name option history. Documented in the Wave 1 post-mortem as a Phase 8 requirement.
</div>

<div class="callout {verdict_class}">
<strong>Verdict: {verdict}</strong><br>
{metrics['n_trades']} successful trades from {n_events} earnings events.
Trade Sharpe {metrics['trade_sharpe']:.2f}, OOS Sharpe {metrics['oos_sharpe']:.2f}.
SPY correlation {metrics['spy_corr']:+.3f}, EXP-1220 correlation {metrics['exp1220_corr']:+.3f}.
</div>

<h2>Summary Metrics</h2>
<div class="cards">
  <div class="c"><div class="l">N Trades</div><div class="v">{metrics['n_trades']}</div></div>
  <div class="c"><div class="l">Events Tested</div><div class="v">{n_events}</div></div>
  <div class="c"><div class="l">Total PnL</div><div class="v">${metrics['total_pnl']:,.0f}</div></div>
  <div class="c"><div class="l">Win Rate</div><div class="v">{metrics['win_rate']:.0%}</div></div>
  <div class="c"><div class="l">Trade Sharpe</div><div class="v">{metrics['trade_sharpe']:.2f}</div></div>
  <div class="c"><div class="l">Daily Sharpe</div><div class="v">{metrics['sharpe_arith']:.2f}</div></div>
  <div class="c"><div class="l">CAGR (daily)</div><div class="v">{metrics['cagr']:.1%}</div></div>
  <div class="c"><div class="l">Max DD</div><div class="v">{metrics['max_dd']:.1%}</div></div>
  <div class="c"><div class="l">OOS Sharpe</div><div class="v">{metrics['oos_sharpe']:.2f}</div></div>
  <div class="c"><div class="l">OOS N</div><div class="v">{metrics['oos_n']}</div></div>
  <div class="c"><div class="l">SPY Corr</div><div class="v">{metrics['spy_corr']:+.3f}</div></div>
  <div class="c"><div class="l">EXP-1220 Corr</div><div class="v">{metrics['exp1220_corr']:+.3f}</div></div>
</div>

<h2>Per-Ticker Breakdown</h2>
<table>
<thead><tr><th>Ticker</th><th>Index</th><th>Events</th><th>IronVault Contracts</th><th>N Trades</th><th>PnL</th><th>WR</th><th>Avg</th></tr></thead>
<tbody>{rows}</tbody></table>

<h2>Year-by-Year</h2>
<table>
<thead><tr><th>Year</th><th>N</th><th>PnL</th><th>WR</th></tr></thead>
<tbody>{yr_rows}</tbody></table>

<h2>Data Sources</h2>
<ul style="padding-left:20px;line-height:1.7">
<li><strong>Earnings dates (primary):</strong> yfinance.Ticker.earnings_dates — works after
<code>pip install --user --break-system-packages lxml</code>. Returns ~25 most recent dates per ticker.</li>
<li><strong>Earnings dates (historical):</strong> SEC 8-K filings hardcoded in
compass/earnings_vol_crush.py for 2020-2023 coverage gap.</li>
<li><strong>Option prices:</strong> IronVault options_cache.db (real Polygon data, SPY/QQQ only).</li>
<li><strong>Underlying prices:</strong> Yahoo Finance chart API.</li>
<li><strong>Sharpe:</strong> compass/metrics.py annualized_sharpe (arithmetic mean formula).</li>
</ul>

<div class="footer">
  EXP-1800 Earnings Crush &bull; 100% real data &bull; {ts}
</div>
</body></html>"""


if __name__ == "__main__":
    main()
