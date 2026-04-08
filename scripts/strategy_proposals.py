#!/usr/bin/env python3
"""
Next Strategy Proposals — 5 new uncorrelated alpha sources to complement EXP-1220.
Based on real IronVault data availability and lessons learned from dead strategies.

Outputs: reports/next_strategy_proposals.html
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reports" / "next_strategy_proposals.html"
DB_PATH = ROOT / "data" / "options_cache.db"


def get_data_coverage() -> Dict[str, Any]:
    """Query IronVault DB for actual coverage per ticker."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    tickers = {}
    cur.execute("""
        SELECT ticker, COUNT(*) as contracts,
               MIN(expiration) as first_exp, MAX(expiration) as last_exp,
               MIN(as_of_date) as first_date, MAX(as_of_date) as last_date
        FROM option_contracts
        GROUP BY ticker ORDER BY COUNT(*) DESC
    """)
    for row in cur.fetchall():
        ticker, cnt, first_exp, last_exp, first_date, last_date = row
        tickers[ticker] = {
            "contracts": cnt,
            "first_exp": first_exp,
            "last_exp": last_exp,
            "first_date": first_date,
            "last_date": last_date,
        }

    # Daily bars per ticker
    cur.execute("""
        SELECT oc.ticker, COUNT(od.contract_symbol) as bars
        FROM option_daily od
        JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
        WHERE od.date != '0000-00-00'
        GROUP BY oc.ticker
    """)
    for row in cur.fetchall():
        ticker, bars = row
        if ticker in tickers:
            tickers[ticker]["daily_bars"] = bars

    # Intraday bars per ticker
    cur.execute("""
        SELECT oc.ticker, COUNT(oi.contract_symbol) as bars
        FROM option_intraday oi
        JOIN option_contracts oc ON oi.contract_symbol = oc.contract_symbol
        WHERE oi.bar_time != 'FETCHED'
        GROUP BY oc.ticker
    """)
    for row in cur.fetchall():
        ticker, bars = row
        if ticker in tickers:
            tickers[ticker]["intraday_bars"] = bars

    conn.close()
    return tickers


def build_html(coverage: Dict[str, Any]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Coverage table
    cov_rows = ""
    for ticker in sorted(coverage.keys(), key=lambda t: coverage[t]["contracts"], reverse=True):
        c = coverage[ticker]
        intra = c.get("intraday_bars", 0)
        cov_rows += f"""<tr>
          <td><strong>{ticker}</strong></td>
          <td>{c['contracts']:,}</td>
          <td>{c.get('daily_bars', 0):,}</td>
          <td>{intra:,}</td>
          <td>{c['first_date'][:7]} &rarr; {c['last_date'][:7]}</td>
          <td>{'Yes' if intra > 1000 else 'Limited' if intra > 0 else 'No'}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Next Strategy Proposals — PilotAI Credit Spreads</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1300px; margin: 0 auto; padding: 24px 32px; background: #fff;
         color: #1e293b; line-height: 1.6; }}
  h1 {{ color: #0f172a; font-size: 1.9em; border-bottom: 3px solid #3b82f6;
       padding-bottom: 8px; }}
  h2 {{ color: #1e40af; margin-top: 2.5em; }}
  h3 {{ color: #334155; margin-top: 1.5em; }}
  h4 {{ color: #475569; margin: 16px 0 8px; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 24px; }}
  .muted {{ color: #94a3b8; }}

  /* Proposal cards */
  .proposal {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
               padding: 24px 28px; margin: 20px 0; }}
  .proposal.top-pick {{ border-left: 5px solid #3b82f6; }}
  .proposal-header {{ display: flex; justify-content: space-between; align-items: flex-start;
                      margin-bottom: 12px; }}
  .proposal-num {{ background: #3b82f6; color: #fff; border-radius: 50%; width: 32px;
                   height: 32px; display: flex; align-items: center; justify-content: center;
                   font-weight: 700; font-size: 0.9em; flex-shrink: 0; }}
  .proposal-title {{ font-size: 1.2em; font-weight: 700; color: #0f172a; margin-left: 12px;
                     flex: 1; }}
  .tag {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.72em;
          font-weight: 600; letter-spacing: 0.03em; margin-left: 6px; }}
  .tag.asset {{ background: #dbeafe; color: #1e40af; }}
  .tag.timeframe {{ background: #dcfce7; color: #166534; }}
  .tag.corr {{ background: #fef3c7; color: #92400e; }}
  .tag.data-yes {{ background: #dcfce7; color: #166534; }}
  .tag.data-partial {{ background: #fef3c7; color: #92400e; }}
  .tag.data-no {{ background: #fee2e2; color: #991b1b; }}

  /* KPI mini cards */
  .kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 10px 14px; text-align: center; min-width: 100px; }}
  .kpi .value {{ font-size: 1.3em; font-weight: 800; color: #0f172a; }}
  .kpi .label {{ font-size: 0.72em; color: #64748b; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: right; font-weight: 600;
       color: #475569; border-bottom: 2px solid #cbd5e1; font-size: 0.82em; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #e2e8f0; }}
  td:first-child {{ text-align: left; }}

  .good {{ color: #16a34a; }}
  .warn {{ color: #ca8a04; }}
  .bad {{ color: #dc2626; }}
  code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }}
  .callout {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
              padding: 14px 18px; margin: 16px 0; font-size: 0.92em; }}
  .callout.warn {{ background: #fffbeb; border-color: #fde68a; }}
  .risk {{ background: #fef2f2; border-left: 4px solid #f87171; border-radius: 0 8px 8px 0;
           padding: 10px 16px; margin: 8px 0; font-size: 0.88em; }}
  .hypothesis {{ background: #f0fdf4; border-left: 4px solid #4ade80; border-radius: 0 8px 8px 0;
                 padding: 10px 16px; margin: 8px 0; font-size: 0.92em; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Next Strategy Proposals</h1>
<p class="meta">5 uncorrelated alpha sources to complement EXP-1220 (Tail Risk, Sharpe 5.78) &middot;
   Generated {now}</p>

<div class="callout">
  <strong>Design Principles:</strong> Every proposal must (1) use real IronVault/Polygon data,
  (2) target low correlation to EXP-1220's VIX-based tail risk signals, (3) survive the
  lessons from dead strategies (no calendar effects, no weak cross-asset leads, no synthetic pricing).
  <br><br>
  <strong>Anchor Strategy — EXP-1220-real:</strong> VIX term structure crash detector, Sharpe 5.78,
  77% CAGR, -6.6% max DD. Signals are driven by VIX level, VIX/VIX3M ratio, and vol regime.
  New strategies should be driven by <em>different</em> factors.
</div>

<h2>Available Data (IronVault options_cache.db)</h2>
<table>
  <tr><th>Ticker</th><th>Contracts</th><th>Daily Bars</th><th>Intraday Bars</th>
      <th>Date Range</th><th>Intraday Support</th></tr>
  {cov_rows}
</table>

<!-- ════════════════════════════════════════════════════════════════════ -->

<h2>Proposal 1: GLD-TLT Relative Value Spread</h2>
<div class="proposal top-pick">
  <div class="proposal-header">
    <div class="proposal-num">1</div>
    <div class="proposal-title">Gold vs Bonds Relative Value
      <span class="tag asset">Fixed Income / Commodity</span>
      <span class="tag timeframe">Weekly</span>
      <span class="tag corr">Low Corr to EXP-1220</span>
    </div>
  </div>

  <div class="hypothesis">
    <strong>Hypothesis:</strong> The GLD/TLT ratio mean-reverts over 2-6 week periods.
    When gold outperforms bonds by &gt;1.5 standard deviations (z-score of 20-day rolling ratio),
    sell put spreads on GLD (expecting reversion lower) and sell call spreads on TLT (expecting
    reversion higher), or vice versa. The spread captures the reversion premium while hedging
    macro direction.
  </div>

  <div class="kpi-row">
    <div class="kpi"><div class="value">1.0-1.5</div><div class="label">Expected Sharpe</div></div>
    <div class="kpi"><div class="value">&lt;10%</div><div class="label">Target Max DD</div></div>
    <div class="kpi"><div class="value">~60</div><div class="label">Trades/Year</div></div>
    <div class="kpi"><div class="value">0.1-0.2</div><div class="label">Est. Corr to 1220</div></div>
  </div>

  <h4>Why Low Correlation to EXP-1220?</h4>
  <p>EXP-1220 trades VIX regime (equity vol). GLD/TLT relative value is driven by real rates,
     inflation expectations, and flight-to-quality dynamics — different factors. During a crash,
     both GLD and TLT rally (flight to safety), but their <em>ratio</em> may not change much.
     Conversely, in calm markets, the ratio trends on macro fundamentals while EXP-1220 is dormant.</p>

  <h4>Data Requirements</h4>
  <table>
    <tr><th>Data</th><th>Source</th><th>Available?</th></tr>
    <tr><td>GLD options (daily)</td><td>IronVault</td><td class="good">Yes — {coverage.get('GLD',{}).get('contracts',0):,} contracts</td></tr>
    <tr><td>TLT options (daily)</td><td>IronVault</td><td class="good">Yes — {coverage.get('TLT',{}).get('contracts',0):,} contracts</td></tr>
    <tr><td>GLD/TLT underlying prices</td><td>Yahoo Finance</td><td class="good">Yes</td></tr>
    <tr><td>Real yields (TIPS spread)</td><td>FRED API</td><td class="warn">Optional enhancement</td></tr>
  </table>

  <div class="risk">
    <strong>Key Risk:</strong> GLD and TLT options are less liquid than SPY — wider bid-ask spreads
    eat into the thin credit spread premium. Must verify min credit exceeds realistic slippage.
    Also, GLD coverage ends 2024 — need to check if 2025 data exists.
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════ -->

<h2>Proposal 2: Sector Rotation Momentum + Put Selling</h2>
<div class="proposal top-pick">
  <div class="proposal-header">
    <div class="proposal-num">2</div>
    <div class="proposal-title">Sell Puts on Strongest Sector ETF
      <span class="tag asset">Equity Sectors</span>
      <span class="tag timeframe">Bi-weekly</span>
      <span class="tag corr">Medium-Low Corr</span>
    </div>
  </div>

  <div class="hypothesis">
    <strong>Hypothesis:</strong> Relative sector momentum persists over 2-4 week horizons.
    Rank XLF, XLI, XLK, XLE by trailing 20-day return. Sell OTM put spreads on the
    top-ranked sector (momentum winner) and avoid the bottom-ranked (momentum loser).
    The winner has structural demand support — institutional rebalancing flows push it
    higher, making put spreads safer. Unlike EXP-880's SPY-only approach, this
    diversifies across sectors and avoids directional SPY bets.
  </div>

  <div class="kpi-row">
    <div class="kpi"><div class="value">0.8-1.2</div><div class="label">Expected Sharpe</div></div>
    <div class="kpi"><div class="value">&lt;12%</div><div class="label">Target Max DD</div></div>
    <div class="kpi"><div class="value">~100</div><div class="label">Trades/Year</div></div>
    <div class="kpi"><div class="value">0.2-0.35</div><div class="label">Est. Corr to 1220</div></div>
  </div>

  <h4>Why Low Correlation to EXP-1220?</h4>
  <p>Sector rotation is driven by earnings cycles, fiscal policy, and industry fundamentals —
     not VIX regime. During a VIX spike (when EXP-1220 goes defensive), sector rotation may
     actually accelerate as money flows from cyclicals to defensives. The alpha source is
     <em>relative performance</em>, not absolute market direction.</p>

  <h4>Data Requirements</h4>
  <table>
    <tr><th>Data</th><th>Source</th><th>Available?</th></tr>
    <tr><td>XLF options</td><td>IronVault</td><td class="good">Yes — {coverage.get('XLF',{}).get('contracts',0):,} contracts</td></tr>
    <tr><td>XLI options</td><td>IronVault</td><td class="good">Yes — {coverage.get('XLI',{}).get('contracts',0):,} contracts</td></tr>
    <tr><td>XLK options</td><td>IronVault</td><td class="good">Yes — {coverage.get('XLK',{}).get('contracts',0):,} contracts</td></tr>
    <tr><td>XLE options</td><td>IronVault</td><td class="good">Yes — {coverage.get('XLE',{}).get('contracts',0):,} contracts</td></tr>
    <tr><td>Sector ETF prices</td><td>Yahoo Finance</td><td class="good">Yes</td></tr>
  </table>

  <div class="risk">
    <strong>Key Risk:</strong> Sector ETF options have lower liquidity and wider spreads than SPY.
    XLE has only {coverage.get('XLE',{}).get('contracts',0):,} contracts — may not have sufficient
    strike granularity. Need to verify min credit viability per sector.
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════ -->

<h2>Proposal 3: VIX Contango Carry Harvester</h2>
<div class="proposal">
  <div class="proposal-header">
    <div class="proposal-num">3</div>
    <div class="proposal-title">Systematic VIX Term Structure Carry
      <span class="tag asset">Volatility</span>
      <span class="tag timeframe">Daily</span>
      <span class="tag corr">Negative Corr Possible</span>
    </div>
  </div>

  <div class="hypothesis">
    <strong>Hypothesis:</strong> VIX futures are in contango ~80% of the time (VIX &lt; VIX3M).
    This creates a systematic "roll yield" that can be harvested by selling short-dated SPY
    strangles when contango is strong (&gt;5% term premium) and the VIX level is moderate (15-25).
    When the term structure inverts (backwardation), go flat or buy protection.
    This is the <em>opposite trade</em> to EXP-1220: EXP-1220 cuts exposure in backwardation,
    while this strategy harvests the carry in contango.
  </div>

  <div class="kpi-row">
    <div class="kpi"><div class="value">1.5-2.5</div><div class="label">Expected Sharpe</div></div>
    <div class="kpi"><div class="value">&lt;15%</div><div class="label">Target Max DD</div></div>
    <div class="kpi"><div class="value">~150</div><div class="label">Trades/Year</div></div>
    <div class="kpi"><div class="value">-0.3 to 0.0</div><div class="label">Est. Corr to 1220</div></div>
  </div>

  <h4>Why Potentially Negative Correlation to EXP-1220?</h4>
  <p>EXP-1220 is <em>long vol</em> in spirit — it profits when crashes happen and VIX spikes.
     This strategy is <em>short vol</em> — it profits from the contango roll yield in calm markets.
     When EXP-1220 fires (VIX spike → backwardation), this strategy goes flat or hedges.
     The two strategies are natural complements: one profits in crashes, the other in the calm
     periods between them.</p>

  <h4>Data Requirements</h4>
  <table>
    <tr><th>Data</th><th>Source</th><th>Available?</th></tr>
    <tr><td>SPY options (daily + intraday)</td><td>IronVault</td><td class="good">Yes — {coverage.get('SPY',{}).get('contracts',0):,} contracts</td></tr>
    <tr><td>VIX daily closes</td><td>Yahoo Finance (^VIX)</td><td class="good">Yes</td></tr>
    <tr><td>VIX3M daily closes</td><td>Yahoo Finance (^VIX3M)</td><td class="good">Yes</td></tr>
    <tr><td>VIX futures curve</td><td>CBOE/Polygon</td><td class="warn">Not cached — use spot proxy</td></tr>
  </table>

  <div class="callout warn">
    <strong>Key Insight:</strong> Combined with EXP-1220, this creates a "barbell" portfolio:
    long vol protection (1220) + short vol carry (this). The combination may achieve higher
    risk-adjusted returns than either alone, because one is always earning while the other
    is hedging.
  </div>

  <div class="risk">
    <strong>Key Risk:</strong> Short vol strategies have fat-tailed losses. The contango carry
    can be wiped out by a single VIX spike event. Position sizing must be conservative
    (2-3% max risk per trade). The VIX-gated entry rule (no selling when VIX &gt; 25) is critical.
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════ -->

<h2>Proposal 4: SPY Intraday Gamma Scalping</h2>
<div class="proposal">
  <div class="proposal-header">
    <div class="proposal-num">4</div>
    <div class="proposal-title">Long Gamma with Delta-Hedge Scalping
      <span class="tag asset">Equity Vol</span>
      <span class="tag timeframe">Intraday</span>
      <span class="tag corr">Negative Corr Expected</span>
    </div>
  </div>

  <div class="hypothesis">
    <strong>Hypothesis:</strong> Buy ATM SPY straddles when IV rank is below the 25th percentile
    (vol is cheap relative to history). Then delta-hedge every 30-60 minutes using the intraday
    5-min bars in IronVault. Profits come when realized vol exceeds implied vol — the gamma
    scalping P&L is (realized vol² - implied vol²) × vega. This is a <em>long vol</em> strategy
    that profits from underpriced options, in contrast to credit spread strategies that sell vol.
  </div>

  <div class="kpi-row">
    <div class="kpi"><div class="value">0.5-1.0</div><div class="label">Expected Sharpe</div></div>
    <div class="kpi"><div class="value">&lt;8%</div><div class="label">Target Max DD</div></div>
    <div class="kpi"><div class="value">~200</div><div class="label">Trades/Year</div></div>
    <div class="kpi"><div class="value">-0.2 to -0.5</div><div class="label">Est. Corr to 1220</div></div>
  </div>

  <h4>Why Negative Correlation to EXP-1220?</h4>
  <p>EXP-1220 profits from <em>detecting</em> vol spikes and cutting exposure. Gamma scalping
     profits from <em>being long</em> during vol spikes (the straddle gains value). In calm
     markets, gamma scalping bleeds theta (loses money) while EXP-1220 is neutral.
     During crashes, gamma scalping profits from the vol explosion while EXP-1220 profits
     from avoiding losses. Different mechanisms, similar crisis-period payoffs.</p>

  <h4>Data Requirements</h4>
  <table>
    <tr><th>Data</th><th>Source</th><th>Available?</th></tr>
    <tr><td>SPY ATM options (daily)</td><td>IronVault</td><td class="good">Yes</td></tr>
    <tr><td>SPY intraday 5-min bars</td><td>IronVault</td><td class="warn">Partial — {coverage.get('SPY',{}).get('intraday_bars',0):,} bars</td></tr>
    <tr><td>SPY underlying intraday</td><td>Yahoo Finance / cache</td><td class="warn">Available via backtester</td></tr>
    <tr><td>IV surface / skew data</td><td>Compute from options</td><td class="good">Derivable from strike prices</td></tr>
  </table>

  <div class="risk">
    <strong>Key Risk:</strong> Gamma scalping has negative expected value if IV is fairly priced
    (which it usually is). The strategy only works in periods where realized vol significantly
    exceeds implied vol. IV rank &lt; 25th percentile filter is the edge — but it may produce
    very few trades. Intraday data coverage is partial (1.4M bars ≠ full 6-year coverage).
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════ -->

<h2>Proposal 5: QQQ-SPY Tech Premium Pairs Trade</h2>
<div class="proposal">
  <div class="proposal-header">
    <div class="proposal-num">5</div>
    <div class="proposal-title">QQQ/SPY Ratio Mean Reversion with Options
      <span class="tag asset">Equity Pairs</span>
      <span class="tag timeframe">2-4 Weeks</span>
      <span class="tag corr">Low Corr to EXP-1220</span>
    </div>
  </div>

  <div class="hypothesis">
    <strong>Hypothesis:</strong> The QQQ/SPY ratio (tech premium) mean-reverts over 2-4 week
    horizons around a slow-moving trend. When the z-score of the 20-day log ratio exceeds
    &plusmn;1.5 standard deviations, sell options on the extended side: if QQQ outperforms
    (ratio high), sell QQQ call spreads + SPY put spreads; if QQQ underperforms, reverse.
    This captures the reversion premium while being market-neutral (long one, short the other).
  </div>

  <div class="kpi-row">
    <div class="kpi"><div class="value">0.8-1.5</div><div class="label">Expected Sharpe</div></div>
    <div class="kpi"><div class="value">&lt;10%</div><div class="label">Target Max DD</div></div>
    <div class="kpi"><div class="value">~50</div><div class="label">Trades/Year</div></div>
    <div class="kpi"><div class="value">0.1-0.25</div><div class="label">Est. Corr to 1220</div></div>
  </div>

  <h4>Why Low Correlation to EXP-1220?</h4>
  <p>The QQQ/SPY ratio is driven by tech sector earnings, AI narrative, and growth/value rotation —
     factors orthogonal to VIX term structure. During a crash, both QQQ and SPY drop together
     (ratio may not move much), so the pairs trade is relatively protected. EXP-1220 fires on
     <em>absolute</em> vol; this trades <em>relative</em> performance.</p>

  <h4>Data Requirements</h4>
  <table>
    <tr><th>Data</th><th>Source</th><th>Available?</th></tr>
    <tr><td>QQQ options (daily)</td><td>IronVault</td><td class="warn">Partial — {coverage.get('QQQ',{}).get('contracts',0):,} contracts, ends ~2023</td></tr>
    <tr><td>SPY options (daily)</td><td>IronVault</td><td class="good">Yes — full coverage</td></tr>
    <tr><td>QQQ/SPY underlying prices</td><td>Yahoo Finance</td><td class="good">Yes</td></tr>
    <tr><td>Cointegration test</td><td>Computed</td><td class="good">statsmodels available</td></tr>
  </table>

  <div class="risk">
    <strong>Key Risk:</strong> QQQ IronVault coverage ends ~2023 — only 3 years of backtest data.
    May need Polygon API backfill for 2024-2025. Also, QQQ options are less liquid than SPY,
    and the pair trade requires simultaneous execution on both legs — slippage on both sides.
    The tech premium can trend persistently (2023 AI rally), breaking the mean-reversion assumption.
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════ -->

<h2>Priority Ranking & Implementation Plan</h2>

<table>
  <tr><th>#</th><th>Strategy</th><th>Data Ready?</th><th>Expected Sharpe</th>
      <th>Corr to 1220</th><th>Complexity</th><th>Priority</th></tr>
  <tr>
    <td>3</td><td><strong>VIX Contango Carry</strong></td>
    <td class="good">Full</td><td>1.5-2.5</td><td>-0.3 to 0.0</td><td>Medium</td>
    <td><strong class="good">P0 — Start here</strong></td>
  </tr>
  <tr>
    <td>1</td><td><strong>GLD-TLT Relative Value</strong></td>
    <td class="good">Full</td><td>1.0-1.5</td><td>0.1-0.2</td><td>Medium</td>
    <td><strong class="good">P0 — Different asset class</strong></td>
  </tr>
  <tr>
    <td>2</td><td><strong>Sector Rotation Momentum</strong></td>
    <td class="warn">Partial (XLE thin)</td><td>0.8-1.2</td><td>0.2-0.35</td><td>Low</td>
    <td>P1 — Simple to implement</td>
  </tr>
  <tr>
    <td>4</td><td><strong>Gamma Scalping</strong></td>
    <td class="warn">Partial (intraday)</td><td>0.5-1.0</td><td>-0.2 to -0.5</td><td>High</td>
    <td>P2 — Best diversifier but hard</td>
  </tr>
  <tr>
    <td>5</td><td><strong>QQQ-SPY Tech Premium</strong></td>
    <td class="bad">Incomplete (QQQ ends 2023)</td><td>0.8-1.5</td><td>0.1-0.25</td><td>Medium</td>
    <td>P2 — Needs data backfill</td>
  </tr>
</table>

<div class="callout">
  <strong>Recommended First Steps:</strong>
  <ol>
    <li><strong>VIX Contango Carry (P0):</strong> Highest expected Sharpe, negative correlation to EXP-1220,
        full data available. Build as EXP-1620. Natural "barbell" complement to the tail-risk strategy.</li>
    <li><strong>GLD-TLT Relative Value (P0):</strong> Different asset class entirely — real diversification.
        Build as EXP-1630. Both tickers have sufficient IronVault coverage.</li>
    <li><strong>Sector Rotation (P1):</strong> Simple implementation, leverage existing sector ETF data.
        Build as EXP-1640. Start with XLF+XLK (most contracts), add XLI+XLE if liquid enough.</li>
  </ol>
</div>

<h2>What We Explicitly Ruled Out (And Why)</h2>
<table>
  <tr><th>Strategy</th><th>Reason Rejected</th><th>Prior Evidence</th></tr>
  <tr><td>Dispersion (Index vs Components)</td><td>No individual stock options in IronVault</td>
      <td>EXP-1200: -0.36% return, dispersion premium not captured even with component data</td></tr>
  <tr><td>Calendar Effects</td><td>All 8 anomalies non-significant on SPY</td>
      <td>EXP-1150: Sharpe -0.54, all p-values &gt; 0.35</td></tr>
  <tr><td>Cross-Asset Lead-Lag</td><td>Composite Sharpe 0.38, overlay hurts performance</td>
      <td>EXP-1110: overlay reduced WR by 18.6pp</td></tr>
  <tr><td>Earnings Vol Crush (individual stocks)</td><td>No individual stock options data</td>
      <td>Sector ETFs don't have concentrated earnings like individual names</td></tr>
  <tr><td>0-DTE / Ultra-Short Expiry</td><td>Too sparse: 0.82 trades/month</td>
      <td>EXP-1020: signal scarcity killed the strategy</td></tr>
</table>

<footer>
  Generated by <code>scripts/strategy_proposals.py</code> &middot;
  Based on IronVault data coverage and lessons from 19 strategies tested &middot;
  {now}
</footer>

</body>
</html>"""


def main():
    coverage = get_data_coverage()
    print(f"IronVault coverage: {len(coverage)} tickers")
    for t in sorted(coverage.keys(), key=lambda k: coverage[k]["contracts"], reverse=True):
        c = coverage[t]
        print(f"  {t}: {c['contracts']:>7,} contracts, {c.get('daily_bars',0):>9,} daily bars, "
              f"{c.get('intraday_bars',0):>8,} intraday, {c['first_date'][:7]}→{c['last_date'][:7]}")

    html = build_html(coverage)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"\nReport: {OUTPUT}")


if __name__ == "__main__":
    main()
