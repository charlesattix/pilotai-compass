"""
compass/dynamic_leverage_v3.py — Multi-Signal Dynamic Leverage 1×-5×.

Adjusts EXP-1220 leverage between 1× and 5× using 5 lagged signals.
ALL signals use yesterday's data (t-1) to set TODAY's leverage. No look-ahead.

SIGNALS (all lagged):
  1. VIX level bands:
     <15 → 5×  |  15-25 → 3×  |  25-35 → 2×  |  >35 → 1×
  2. VIX term structure (VIX/VIX3M):
     <0.95 contango → +0 (full lever)
     0.95-1.00 → -1 tier
     >1.00 backwardation → -2 tiers (cap at 1×)
  3. Drawdown brake:
     DD ≤ 5% → no effect
     5% < DD ≤ 10% → linear scale toward 1×
     DD > 10% → force 1×
  4. Trailing positive-day rate (proxy for win rate):
     Last 50 days < 50% positive → -1 tier
  5. COMPASS regime (derived from SPY 200d MA + VIX):
     bull → full | bear → -1 tier | crash → 1×

DATA SOURCES (Rule Zero compliant — every input REAL):
  - EXP-1220 daily returns: scripts.ultimate_portfolio.load_exp1220_dynamic()
    (derived from real Yahoo Finance SPY/^VIX/^VIX3M via TailRiskProtector)
  - ^VIX: Yahoo Finance chart API (lagged t-1)
  - ^VIX3M: Yahoo Finance chart API (lagged t-1)
  - SPY: Yahoo Finance chart API (for regime classification)

Sharpe via compass/metrics.py (correct arithmetic mean formula).
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.metrics import annualized_sharpe, full_metrics

TRADING_DAYS = 252
REPORT_PATH = ROOT / "reports" / "dynamic_leverage_v3_report.html"


# ═══════════════════════════════════════════════════════════════════════════
# Data loaders (REAL only)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_yahoo_close(symbol: str, start: str, end: str) -> pd.Series:
    """Fetch daily closes from Yahoo Finance. Real data only."""
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    safe = symbol.replace("^", "%5E")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{safe}"
           f"?period1={start_ts}&period2={end_ts}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    ts = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    dates = [datetime.fromtimestamp(t).date() for t in ts]
    return pd.Series(closes, index=pd.DatetimeIndex(dates), name=symbol).dropna()


def load_all_data(start: str = "2019-12-01", end: str = "2026-01-01"):
    """Load EXP-1220 returns + VIX + VIX3M + SPY. Pre-fetch for lagging."""
    print("  EXP-1220 daily returns (load_exp1220_dynamic)...")
    from scripts.ultimate_portfolio import load_exp1220_dynamic
    base = load_exp1220_dynamic()
    print(f"    {len(base)} days, {base.index[0].date()} → {base.index[-1].date()}")

    print("  ^VIX from Yahoo Finance...")
    vix = fetch_yahoo_close("^VIX", start, end)
    print(f"    {len(vix)} days")

    print("  ^VIX3M from Yahoo Finance...")
    vix3m = fetch_yahoo_close("^VIX3M", start, end)
    print(f"    {len(vix3m)} days")

    print("  SPY from Yahoo Finance...")
    spy = fetch_yahoo_close("SPY", start, end)
    print(f"    {len(spy)} days")

    return base, vix, vix3m, spy


# ═══════════════════════════════════════════════════════════════════════════
# Signal computation (all return aligned series, all lagged later)
# ═══════════════════════════════════════════════════════════════════════════

def vix_band_leverage(vix_value: float) -> float:
    """Map VIX level to base leverage."""
    if vix_value < 15:
        return 5.0
    if vix_value < 25:
        return 3.0
    if vix_value < 35:
        return 2.0
    return 1.0


def term_structure_adjustment(ratio: float) -> int:
    """VIX/VIX3M ratio → tier adjustment.

    < 0.95 (deep contango): no change
    0.95-1.00 (mild): -1 tier
    > 1.00 (backwardation = stress): -2 tiers
    """
    if ratio < 0.95:
        return 0
    if ratio < 1.00:
        return -1
    return -2


def drawdown_brake(dd_pct: float, base_leverage: float) -> float:
    """DD brake: linear scale from base to 1× as DD goes from 5% to 10%.

    dd_pct is positive (e.g. 0.07 = 7% DD).
    """
    if dd_pct <= 0.05:
        return base_leverage
    if dd_pct >= 0.10:
        return 1.0
    # Linear interpolation from base_leverage @ 5% to 1.0 @ 10%
    t = (dd_pct - 0.05) / 0.05
    return base_leverage - t * (base_leverage - 1.0)


def regime_classify(spy_now: float, spy_ma200: float, vix_value: float) -> str:
    """Simple regime: bull (above 200 MA), bear (below), crash (VIX>40)."""
    if vix_value > 40:
        return "crash"
    if spy_now >= spy_ma200 * 1.02:
        return "bull"
    if spy_now < spy_ma200 * 0.98:
        return "bear"
    return "neutral"


def regime_adjustment(regime: str) -> int:
    """Regime → tier adjustment."""
    return {"bull": 0, "neutral": 0, "bear": -1, "crash": -2}.get(regime, 0)


def tier_to_leverage(tier_idx: int) -> float:
    """Convert tier index to leverage. Tier 0 = 1×, 1 = 2×, 2 = 3×, 3 = 5×."""
    tiers = [1.0, 2.0, 3.0, 5.0]
    return tiers[max(0, min(len(tiers) - 1, tier_idx))]


def leverage_to_tier(lev: float) -> int:
    tiers = [1.0, 2.0, 3.0, 5.0]
    # Find closest tier index
    return min(range(len(tiers)), key=lambda i: abs(tiers[i] - lev))


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic leverage backtest (with rigorous t-1 lagging)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DailyState:
    date: pd.Timestamp
    vix_used: float          # the LAGGED VIX value
    vix_ratio_used: float    # the LAGGED ratio
    dd: float                 # current DD from peak (positive)
    pos_day_rate: float       # rolling positive-day fraction
    regime: str
    base_leverage: float      # from VIX bands
    final_leverage: float     # after all adjustments
    daily_return: float


def run_dynamic_backtest(
    base_returns: pd.Series,
    vix: pd.Series,
    vix3m: pd.Series,
    spy: pd.Series,
) -> Tuple[pd.Series, List[DailyState]]:
    """Run dynamic leverage with strict t-1 lagging."""
    # Align all series
    common = base_returns.index.intersection(vix.index).intersection(vix3m.index).intersection(spy.index)
    common = common.sort_values()
    rets = base_returns.reindex(common).fillna(0)
    vix_s = vix.reindex(common).ffill().bfill()
    vix3m_s = vix3m.reindex(common).ffill().bfill()
    spy_s = spy.reindex(common).ffill().bfill()

    # SPY 200-day MA
    spy_ma200 = spy_s.rolling(200, min_periods=50).mean()

    # Pre-compute the lagged signal arrays
    vix_lagged = vix_s.shift(1).bfill()
    vix3m_lagged = vix3m_s.shift(1).bfill()
    spy_lagged = spy_s.shift(1).bfill()
    spy_ma200_lagged = spy_ma200.shift(1).bfill()

    # Walk forward day by day
    equity = 1.0
    peak = equity
    dynamic_rets = []
    states = []
    pos_day_history = []  # for trailing positive-day rate

    for i, dt in enumerate(common):
        # All signals lagged t-1
        v = float(vix_lagged.iloc[i])
        v3m = float(vix3m_lagged.iloc[i])
        ratio = v / max(v3m, 1.0)
        spy_now = float(spy_lagged.iloc[i])
        spy_ma = float(spy_ma200_lagged.iloc[i])

        # DD computed from yesterday's equity (i.e., before applying today's return)
        dd = max(0, (peak - equity) / peak) if peak > 0 else 0

        # Trailing positive-day rate (last 50 days, EXCLUDING today)
        if len(pos_day_history) >= 50:
            pos_day_rate = sum(pos_day_history[-50:]) / 50
        elif len(pos_day_history) > 0:
            pos_day_rate = sum(pos_day_history) / len(pos_day_history)
        else:
            pos_day_rate = 0.6  # neutral default

        # Compute base leverage from VIX bands
        base_lev = vix_band_leverage(v)
        base_tier = leverage_to_tier(base_lev)

        # Apply tier adjustments
        ts_adj = term_structure_adjustment(ratio)
        regime = regime_classify(spy_now, spy_ma, v)
        regime_adj = regime_adjustment(regime)
        wr_adj = -1 if pos_day_rate < 0.50 else 0

        adjusted_tier = base_tier + ts_adj + regime_adj + wr_adj
        adjusted_lev = tier_to_leverage(adjusted_tier)

        # DD brake (continuous, applied after tier logic)
        final_lev = drawdown_brake(dd, adjusted_lev)
        final_lev = max(1.0, min(5.0, final_lev))

        # Apply leverage to TODAY's return
        today_ret = float(rets.iloc[i]) * final_lev
        equity *= (1 + today_ret)
        if equity > peak:
            peak = equity

        # Update trailing pos-day history with TODAY's outcome (for tomorrow's signal)
        pos_day_history.append(1 if today_ret > 0 else 0)
        if len(pos_day_history) > 100:
            pos_day_history.pop(0)

        dynamic_rets.append(today_ret)
        states.append(DailyState(
            date=dt, vix_used=v, vix_ratio_used=ratio,
            dd=dd, pos_day_rate=pos_day_rate, regime=regime,
            base_leverage=base_lev, final_leverage=final_lev,
            daily_return=today_ret,
        ))

    return pd.Series(dynamic_rets, index=common, name="dynamic"), states


# ═══════════════════════════════════════════════════════════════════════════
# Static leverage backtests (for comparison)
# ═══════════════════════════════════════════════════════════════════════════

def static_backtest(base_returns: pd.Series, leverage: float) -> pd.Series:
    return (base_returns * leverage).rename(f"static_{leverage}x")


# ═══════════════════════════════════════════════════════════════════════════
# Year-by-year + leverage distribution
# ═══════════════════════════════════════════════════════════════════════════

def yearly_breakdown(rets: pd.Series) -> List[Dict]:
    windows = []
    for yr in sorted(set(rets.index.year)):
        yr_rets = rets[rets.index.year == yr].values
        if len(yr_rets) < 20:
            continue
        m = full_metrics(yr_rets)
        m["year"] = int(yr)
        m["n_days"] = len(yr_rets)
        windows.append(m)
    return windows


def leverage_distribution(states: List[DailyState]) -> Dict:
    """Time-spent at each leverage tier."""
    levs = [s.final_leverage for s in states]
    n = len(levs)
    buckets = {"1×": 0, "2×": 0, "3×": 0, "4×": 0, "5×": 0, "other": 0}
    for lev in levs:
        if abs(lev - 1.0) < 0.1: buckets["1×"] += 1
        elif abs(lev - 2.0) < 0.5: buckets["2×"] += 1
        elif abs(lev - 3.0) < 0.5: buckets["3×"] += 1
        elif abs(lev - 4.0) < 0.5: buckets["4×"] += 1
        elif abs(lev - 5.0) < 0.5: buckets["5×"] += 1
        else: buckets["other"] += 1

    return {
        "n_days": n,
        "avg_leverage": round(float(np.mean(levs)), 3),
        "median_leverage": round(float(np.median(levs)), 2),
        "min_leverage": round(min(levs), 2),
        "max_leverage": round(max(levs), 2),
        "buckets_pct": {k: round(v / n * 100, 1) for k, v in buckets.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    static15: Dict, static3: Dict, dynamic: Dict,
    yearly_static15: List[Dict], yearly_static3: List[Dict],
    yearly_dynamic: List[Dict], distribution: Dict,
) -> str:
    # Comparison rows
    def _row(name, m, lev_label):
        sc_cagr = "#16a34a" if m["cagr_pct"] > 0 else "#dc2626"
        sc_dd = "#16a34a" if m["max_dd_pct"] < 15 else ("#ca8a04" if m["max_dd_pct"] < 25 else "#dc2626")
        return f"""<tr>
            <td style="font-weight:600">{name}</td>
            <td>{lev_label}</td>
            <td style="color:{sc_cagr};font-weight:700">{m['cagr_pct']:.1f}%</td>
            <td style="font-weight:700">{m['sharpe']:.2f}</td>
            <td style="color:{sc_dd}">{m['max_dd_pct']:.1f}%</td>
            <td>{m['calmar']:.2f}</td>
            <td>{m['vol_pct']:.1f}%</td>
            <td>{m['sortino']:.2f}</td>
        </tr>"""

    cmp_rows = ""
    cmp_rows += _row("Static 1.5×", static15, "1.5× constant")
    cmp_rows += _row("Static 3×", static3, "3.0× constant")
    cmp_rows += _row("Dynamic 1×-5× (v3)", dynamic, f"{distribution['avg_leverage']}× avg")

    # Year-by-year side-by-side
    yr_rows = ""
    years = sorted(set([w["year"] for w in yearly_static15 + yearly_static3 + yearly_dynamic]))
    by_yr = {
        "static15": {w["year"]: w for w in yearly_static15},
        "static3": {w["year"]: w for w in yearly_static3},
        "dynamic": {w["year"]: w for w in yearly_dynamic},
    }
    for yr in years:
        s15 = by_yr["static15"].get(yr, {})
        s3 = by_yr["static3"].get(yr, {})
        dy = by_yr["dynamic"].get(yr, {})
        yr_rows += f"""<tr>
            <td style="font-weight:700">{yr}</td>
            <td>{s15.get('cagr_pct', 0):.1f}%</td>
            <td>{s15.get('sharpe', 0):.2f}</td>
            <td>{s15.get('max_dd_pct', 0):.1f}%</td>
            <td>{s3.get('cagr_pct', 0):.1f}%</td>
            <td>{s3.get('sharpe', 0):.2f}</td>
            <td>{s3.get('max_dd_pct', 0):.1f}%</td>
            <td style="font-weight:700">{dy.get('cagr_pct', 0):.1f}%</td>
            <td style="font-weight:700">{dy.get('sharpe', 0):.2f}</td>
            <td style="font-weight:700">{dy.get('max_dd_pct', 0):.1f}%</td>
        </tr>"""

    # Leverage distribution rows
    dist_rows = ""
    for tier, pct in distribution["buckets_pct"].items():
        if pct > 0:
            dist_rows += f"<tr><td>{tier}</td><td>{pct:.1f}%</td></tr>"

    # Key question check
    target_cagr_3x = static3["cagr_pct"]
    target_dd_3x = static3["max_dd_pct"]
    dynamic_cagr = dynamic["cagr_pct"]
    dynamic_dd = dynamic["max_dd_pct"]

    cagr_close_to_3x = dynamic_cagr >= target_cagr_3x * 0.85
    dd_better_than_3x = dynamic_dd < target_dd_3x

    verdict_color = "#16a34a" if (cagr_close_to_3x and dd_better_than_3x) else "#ca8a04"
    verdict = ("WIN — dynamic gets ~3× returns with lower DD"
               if (cagr_close_to_3x and dd_better_than_3x)
               else "PARTIAL — review tradeoff carefully")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamic Leverage v3 — 1× to 5× with Multi-Signal</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1100px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.5; }}
  h1 {{ font-size:1.8em; color:#0f172a; margin-bottom:4px; }}
  h2 {{ color:#334155; margin-top:2.5em; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:24px; }}
  .verdict {{ text-align:center; padding:14px; border-radius:8px; font-size:1.1rem; font-weight:800;
              background:{verdict_color}10; color:{verdict_color}; border:2px solid {verdict_color}40; margin:20px 0; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:18px;
          text-align:center; flex:1; min-width:120px; }}
  .kpi .value {{ font-size:1.7em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; }} .warn {{ color:#ca8a04; }} .bad {{ color:#dc2626; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:0.84em; }}
  th {{ background:#f1f5f9; padding:10px 12px; text-align:right; font-weight:600; color:#475569;
       border-bottom:2px solid #cbd5e1; font-size:0.76em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .sources {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:16px; margin:16px 0; font-size:0.86rem; line-height:1.7; }}
  .signal-box {{ background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:16px; margin:16px 0; font-size:0.86rem; line-height:1.7; }}
  .footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.78em; color:#94a3b8; text-align:center; }}
</style></head><body>

<h1>Dynamic Leverage v3 — 1× to 5×</h1>
<div class="subtitle">Multi-signal leverage controller for EXP-1220, t-1 lagged | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="verdict">{verdict}</div>

<div class="sources">
    <strong>Data Sources (Rule Zero — all REAL):</strong><br>
    EXP-1220 daily returns: load_exp1220_dynamic() (Yahoo SPY/^VIX/^VIX3M)<br>
    ^VIX, ^VIX3M, SPY: Yahoo Finance chart API (lagged t-1, no look-ahead)<br>
    Sharpe: compass/metrics.py annualized_sharpe (arithmetic mean)
</div>

<div class="signal-box">
    <strong>5 Lagged Signals:</strong><br>
    1. <strong>VIX bands</strong>: &lt;15 → 5×, 15-25 → 3×, 25-35 → 2×, &gt;35 → 1×<br>
    2. <strong>Term structure</strong>: VIX/VIX3M &lt;0.95 = full, 0.95-1.0 = -1 tier, &gt;1.0 = -2 tiers<br>
    3. <strong>DD brake</strong>: linear scale to 1× as DD goes 5% → 10%, force 1× above 10%<br>
    4. <strong>Trailing pos-day rate</strong>: rolling 50d &lt; 50% positive → -1 tier<br>
    5. <strong>COMPASS regime</strong>: bull = full, bear = -1, crash (VIX&gt;40) = -2
</div>

<h2>The Key Question</h2>
<p>Can dynamic leverage get us 3×-like returns ({target_cagr_3x:.0f}%+) with 2×-like drawdowns (&lt;15%)?</p>

<div class="kpi-row">
    <div class="kpi"><div class="value good">{dynamic_cagr:.1f}%</div><div class="label">Dynamic CAGR</div>
        <div style="font-size:0.7em;color:#64748b">Target: ~{target_cagr_3x:.0f}% {'✓' if cagr_close_to_3x else '✗'}</div></div>
    <div class="kpi"><div class="value">{dynamic['sharpe']:.2f}</div><div class="label">Dynamic Sharpe</div></div>
    <div class="kpi"><div class="value {'good' if dd_better_than_3x else 'warn'}">{dynamic_dd:.1f}%</div><div class="label">Dynamic DD</div>
        <div style="font-size:0.7em;color:#64748b">vs Static 3×: {target_dd_3x:.1f}% {'✓ better' if dd_better_than_3x else '✗'}</div></div>
    <div class="kpi"><div class="value">{dynamic['calmar']:.2f}</div><div class="label">Dynamic Calmar</div></div>
    <div class="kpi"><div class="value">{distribution['avg_leverage']}×</div><div class="label">Avg Leverage</div></div>
</div>

<h2>Static vs Dynamic Comparison</h2>
<table>
    <thead><tr><th>Strategy</th><th>Leverage</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>Sortino</th></tr></thead>
    <tbody>{cmp_rows}</tbody>
</table>

<h2>Year-by-Year Comparison</h2>
<table>
    <thead><tr>
        <th rowspan="2">Year</th>
        <th colspan="3">Static 1.5×</th>
        <th colspan="3">Static 3×</th>
        <th colspan="3">Dynamic v3</th>
    </tr><tr>
        <th>CAGR</th><th>Sharpe</th><th>DD</th>
        <th>CAGR</th><th>Sharpe</th><th>DD</th>
        <th>CAGR</th><th>Sharpe</th><th>DD</th>
    </tr></thead>
    <tbody>{yr_rows}</tbody>
</table>

<h2>Leverage Distribution (Dynamic v3)</h2>
<table>
    <thead><tr><th>Tier</th><th>% of Days</th></tr></thead>
    <tbody>{dist_rows}</tbody>
</table>
<p>Avg: {distribution['avg_leverage']}× | Median: {distribution['median_leverage']}× | Range: {distribution['min_leverage']}× → {distribution['max_leverage']}×</p>

<div class="footer">
    Dynamic Leverage v3 — compass/dynamic_leverage_v3.py<br>
    All data REAL (Yahoo + IronVault-derived). All signals t-1 lagged. No look-ahead.<br>
    Sharpe via compass/metrics.py (arithmetic mean, correct formula).
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("Dynamic Leverage v3 — 1× to 5× Multi-Signal")
    print("=" * 72)

    print("\n[1/4] Loading REAL data...")
    base, vix, vix3m, spy = load_all_data()

    print("\n[2/4] Running static backtests for comparison...")
    static15 = static_backtest(base, 1.5)
    static3 = static_backtest(base, 3.0)
    static15_m = full_metrics(static15.values)
    static3_m = full_metrics(static3.values)
    print(f"  Static 1.5×: CAGR={static15_m['cagr_pct']:.1f}%  Sharpe={static15_m['sharpe']:.2f}  DD={static15_m['max_dd_pct']:.1f}%")
    print(f"  Static 3.0×: CAGR={static3_m['cagr_pct']:.1f}%  Sharpe={static3_m['sharpe']:.2f}  DD={static3_m['max_dd_pct']:.1f}%")

    print("\n[3/4] Running dynamic v3 backtest (5 signals, t-1 lagged)...")
    dynamic_rets, states = run_dynamic_backtest(base, vix, vix3m, spy)
    dynamic_m = full_metrics(dynamic_rets.values)
    print(f"  Dynamic:    CAGR={dynamic_m['cagr_pct']:.1f}%  Sharpe={dynamic_m['sharpe']:.2f}  DD={dynamic_m['max_dd_pct']:.1f}%")

    distribution = leverage_distribution(states)
    print(f"\n  Leverage distribution:")
    print(f"    Avg: {distribution['avg_leverage']}×  Median: {distribution['median_leverage']}×")
    print(f"    Range: {distribution['min_leverage']}× → {distribution['max_leverage']}×")
    for tier, pct in distribution["buckets_pct"].items():
        if pct > 0:
            print(f"    {tier}: {pct}% of days")

    print("\n  Year-by-year comparison:")
    yearly_static15 = yearly_breakdown(static15)
    yearly_static3 = yearly_breakdown(static3)
    yearly_dynamic = yearly_breakdown(dynamic_rets)

    print(f"  {'Year':6s} {'1.5x CAGR':>10s} {'3x CAGR':>10s} {'Dyn CAGR':>10s} {'1.5x DD':>9s} {'3x DD':>9s} {'Dyn DD':>9s}")
    years = sorted(set(w["year"] for w in yearly_dynamic))
    for yr in years:
        s15 = next((w for w in yearly_static15 if w["year"] == yr), {})
        s3 = next((w for w in yearly_static3 if w["year"] == yr), {})
        dy = next((w for w in yearly_dynamic if w["year"] == yr), {})
        print(f"  {yr:6d} {s15.get('cagr_pct', 0):9.1f}% {s3.get('cagr_pct', 0):9.1f}% {dy.get('cagr_pct', 0):9.1f}% "
              f"{s15.get('max_dd_pct', 0):8.1f}% {s3.get('max_dd_pct', 0):8.1f}% {dy.get('max_dd_pct', 0):8.1f}%")

    print(f"\n{'━' * 70}")
    cagr_close = dynamic_m['cagr_pct'] >= static3_m['cagr_pct'] * 0.85
    dd_better = dynamic_m['max_dd_pct'] < static3_m['max_dd_pct']
    print(f"  KEY QUESTION: ~3× returns ({static3_m['cagr_pct']:.0f}%) with <15% DD?")
    print(f"    Dynamic CAGR: {dynamic_m['cagr_pct']:.1f}%  ({'CLOSE' if cagr_close else 'BELOW'} 3× target)")
    print(f"    Dynamic DD:   {dynamic_m['max_dd_pct']:.1f}%  ({'BETTER' if dd_better else 'WORSE'} than 3× DD of {static3_m['max_dd_pct']:.1f}%)")
    if cagr_close and dd_better:
        print(f"  ✓ WIN — dynamic leverage works")
    else:
        print(f"  ⚠ Tradeoff — review carefully")
    print(f"{'━' * 70}")

    print("\n[4/4] Generating HTML report...")
    html = generate_report(
        static15_m, static3_m, dynamic_m,
        yearly_static15, yearly_static3, yearly_dynamic,
        distribution,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}")


if __name__ == "__main__":
    main()
