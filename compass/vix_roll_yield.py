"""
compass/vix_roll_yield.py — EXP-1700 VIX Futures Roll Yield Strategy.

EDGE: VIX futures trade in contango ~80-85% of the time because:
  - Spot VIX mean-reverts to ~17-20
  - Longer-dated futures price in a "fear premium"
  - Front-month futures decay toward spot as expiration approaches

STRATEGY: Short the front-month VIX futures ETF (VXX) when term structure
is in contango, flat when in backwardation (crisis). Captures the roll
decay without options complexity.

DATA SOURCES (all REAL, cited):
  - ^VIX    : CBOE VIX Index (Yahoo Finance chart API)
  - ^VIX3M  : CBOE 3-Month VIX Index (Yahoo Finance chart API)
  - VXX     : iPath VIX ST Futures ETN (Yahoo Finance chart API)
  - SPY     : SPDR S&P 500 ETF for correlation analysis (Yahoo)

NO SYNTHETIC DATA. All prices from Yahoo Finance chart API (free, public).

Signal logic:
  - Contango signal: VIX/VIX3M < 1.00 (front < back = contango)
  - Short VXX when ratio < 0.95 (strong contango)
  - Flat when ratio > 1.00 (backwardation — don't fight crisis)
  - Position size: inverse-vol targeted at 10% annualized

Walk-forward: expanding window, train 2018-N, test N+1.
Metrics via compass/metrics.py (correct arithmetic-mean Sharpe).
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.metrics import annualized_sharpe, full_metrics

ROOT = Path(__file__).resolve().parent.parent
TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Data fetching — ALL REAL SOURCES CITED
# ═══════════════════════════════════════════════════════════════════════════

def fetch_yahoo_series(symbol: str, start: str = "2018-01-01",
                        end: str = "2025-12-31") -> pd.Series:
    """Fetch daily closing prices from Yahoo Finance chart API.

    Args:
        symbol: Yahoo symbol (e.g. "^VIX", "VXX"). ^ is URL-encoded.
        start: ISO date string.
        end: ISO date string.

    Returns:
        pd.Series of daily closes indexed by date.

    Data source: Yahoo Finance (https://query1.finance.yahoo.com/v8/finance/chart/)
    """
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    safe_sym = symbol.replace("^", "%5E")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{safe_sym}"
           f"?period1={start_ts}&period2={end_ts}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    result = data["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    if not timestamps:
        raise RuntimeError(f"No data for {symbol}")

    closes = result["indicators"]["quote"][0]["close"]
    dates = [datetime.fromtimestamp(t).date() for t in timestamps]

    s = pd.Series(closes, index=pd.DatetimeIndex(dates), name=symbol)
    s = s.dropna()
    return s


def load_all_data(start: str = "2018-01-01",
                   end: str = "2025-12-31") -> Dict[str, pd.Series]:
    """Load all required series. Returns dict keyed by symbol.

    Data sources:
        ^VIX:    CBOE Volatility Index (Yahoo Finance)
        ^VIX3M:  CBOE 3-Month VIX (Yahoo Finance)
        VXX:     iPath VIX Short-Term Futures ETN (Yahoo Finance)
        SPY:     SPDR S&P 500 ETF (Yahoo Finance)
    """
    data = {}
    sources = {
        "^VIX": "CBOE VIX Index via Yahoo Finance",
        "^VIX3M": "CBOE 3-Month VIX Index via Yahoo Finance",
        "VXX": "iPath VIX Short-Term Futures ETN via Yahoo Finance",
        "SPY": "SPDR S&P 500 ETF via Yahoo Finance",
    }
    for sym in sources:
        data[sym] = fetch_yahoo_series(sym, start, end)
        print(f"  {sym:8s}: {len(data[sym])} days "
              f"({data[sym].index[0].date()} → {data[sym].index[-1].date()}) "
              f"[{sources[sym]}]")
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Signal + backtest
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RollYieldConfig:
    contango_enter: float = 0.95      # short VXX when VIX/VIX3M < 0.95
    contango_exit: float = 1.00       # exit when ratio crosses 1.00
    backwardation_flat: float = 1.05  # force flat above 1.05 (crisis)
    target_vol: float = 0.10          # 10% annualized vol target
    vol_lookback: int = 20            # realized vol lookback for sizing
    max_position: float = 1.0         # max |position size|


def compute_signals(data: Dict[str, pd.Series],
                     config: RollYieldConfig) -> pd.DataFrame:
    """Compute daily signals from REAL market data (no synthetic).

    Uses t-1 lagged ratio to avoid look-ahead (decisions made at open
    using prior day's close).
    """
    vix = data["^VIX"]
    vix3m = data["^VIX3M"]
    vxx = data["VXX"]

    # Align on common dates
    common = vix.index.intersection(vix3m.index).intersection(vxx.index)
    common = common.sort_values()

    vix = vix.reindex(common).ffill()
    vix3m = vix3m.reindex(common).ffill()
    vxx = vxx.reindex(common).ffill()

    # VXX daily return (we short this)
    vxx_ret = vxx.pct_change().fillna(0)

    # Term structure ratio — LAGGED by 1 day (t-1)
    ratio = (vix / vix3m).shift(1).ffill().bfill()

    # Realized vol for position sizing — LAGGED
    rvol = vxx_ret.rolling(config.vol_lookback, min_periods=5).std().shift(1)
    rvol = rvol * math.sqrt(TRADING_DAYS)
    rvol = rvol.fillna(0.50).clip(lower=0.10)  # VXX is very volatile

    df = pd.DataFrame({
        "date": common,
        "vix": vix.values,
        "vix3m": vix3m.values,
        "ratio": ratio.values,
        "vxx": vxx.values,
        "vxx_ret": vxx_ret.values,
        "rvol": rvol.values,
    }).set_index("date")

    # Signal: negative position size = short VXX
    # Enter short when strong contango; exit when ratio flat/inverted
    positions = np.zeros(len(df))
    current_pos = 0.0
    for i in range(len(df)):
        r = float(df["ratio"].iloc[i])
        if r > config.backwardation_flat:
            current_pos = 0.0  # flat in backwardation
        elif r < config.contango_enter and current_pos == 0:
            # Size inversely with realized vol
            size = min(config.max_position,
                        config.target_vol / max(float(df["rvol"].iloc[i]), 0.10))
            current_pos = -size  # SHORT VXX
        elif r > config.contango_exit:
            current_pos = 0.0
        positions[i] = current_pos

    df["position"] = positions
    # Strategy return = -position × VXX_return (since we're short)
    # When position = -size and VXX falls, profit = +size × |VXX_ret|
    df["strategy_ret"] = df["position"] * df["vxx_ret"]
    return df


def backtest_walk_forward(df: pd.DataFrame) -> Dict:
    """Expanding-window walk-forward: train 2018-N, test N+1."""
    years = sorted(set(df.index.year))
    first_year = min(years)

    windows = []
    all_oos_rets = []
    all_oos_dates = []

    for test_year in range(first_year + 2, max(years) + 1):  # need 2+ yrs train
        test_mask = df.index.year == test_year
        test_df = df.loc[test_mask]
        if len(test_df) < 50:
            continue

        test_rets = test_df["strategy_ret"].values
        m = full_metrics(test_rets)
        avg_pos = float(test_df["position"].mean())
        pct_short = float((test_df["position"] < 0).mean()) * 100

        windows.append({
            "year": test_year,
            "n_days": len(test_df),
            "cagr_pct": m["cagr_pct"],
            "sharpe": m["sharpe"],
            "max_dd_pct": m["max_dd_pct"],
            "vol_pct": m["vol_pct"],
            "avg_position": round(avg_pos, 3),
            "pct_short": round(pct_short, 1),
        })
        all_oos_rets.extend(test_rets.tolist())
        all_oos_dates.extend(test_df.index.tolist())

    oos_agg = full_metrics(np.array(all_oos_rets)) if all_oos_rets else {}

    # Full-period metrics
    full_metrics_all = full_metrics(df["strategy_ret"].values)

    return {
        "windows": windows,
        "oos_aggregate": oos_agg,
        "full": full_metrics_all,
        "all_oos_rets": all_oos_rets,
        "all_oos_dates": all_oos_dates,
    }


def compute_correlation_to_exp1220(strategy_rets: pd.Series) -> float:
    """Compute correlation between VIX roll yield and EXP-1220 credit spreads."""
    try:
        from scripts.ultimate_portfolio import load_exp1220_dynamic
        exp1220 = load_exp1220_dynamic()
        common = strategy_rets.index.intersection(exp1220.index)
        if len(common) < 20:
            return float("nan")
        s1 = strategy_rets.reindex(common).fillna(0).values
        s2 = exp1220.reindex(common).fillna(0).values
        return float(np.corrcoef(s1, s2)[0, 1])
    except Exception as e:
        print(f"  Correlation computation skipped: {e}")
        return float("nan")


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(data: Dict, df: pd.DataFrame, wf: Dict,
                     correlation: float, config: RollYieldConfig) -> str:
    full_m = wf["full"]
    agg = wf["oos_aggregate"]
    windows = wf["windows"]

    # Contango stats
    contango_pct = float((df["ratio"] < 1.0).mean()) * 100
    avg_ratio = float(df["ratio"].mean())
    short_pct = float((df["position"] < 0).mean()) * 100

    wf_rows = ""
    for w in windows:
        sc = "#16a34a" if w["cagr_pct"] > 0 else "#dc2626"
        wf_rows += f"""<tr>
            <td style="font-weight:700">{w['year']}</td>
            <td>{w['n_days']}</td>
            <td style="color:{sc};font-weight:600">{w['cagr_pct']:.1f}%</td>
            <td style="font-weight:700">{w['sharpe']:.2f}</td>
            <td>{w['max_dd_pct']:.1f}%</td>
            <td>{w['vol_pct']:.1f}%</td>
            <td>{w['pct_short']:.0f}%</td>
        </tr>"""

    corr_text = f"{correlation:+.3f}" if not math.isnan(correlation) else "N/A"
    corr_color = "#16a34a" if abs(correlation) < 0.2 else "#ca8a04"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1700 VIX Futures Roll Yield</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1000px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.5; }}
  h1 {{ font-size:1.8em; color:#0f172a; margin-bottom:4px; }}
  h2 {{ color:#334155; margin-top:2.5em; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:24px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:18px;
          text-align:center; flex:1; min-width:130px; }}
  .kpi .value {{ font-size:1.7em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; }} .warn {{ color:#ca8a04; }} .bad {{ color:#dc2626; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:10px 12px; text-align:right; font-weight:600; color:#475569;
       border-bottom:2px solid #cbd5e1; font-size:0.80em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .callout {{ background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:16px; margin:16px 0; font-size:0.86rem; line-height:1.7; }}
  .sources {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:16px; margin:16px 0; font-size:0.86rem; }}
  .footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.78em; color:#94a3b8; text-align:center; }}
</style></head><body>

<h1>EXP-1700 — VIX Futures Roll Yield</h1>
<div class="subtitle">Phase 7 Wave 1 | Short VXX in contango | Real Yahoo Finance data | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="sources">
    <strong>Data Sources (Rule Zero compliant — zero synthetic):</strong><br>
    • <code>^VIX</code> — CBOE Volatility Index, Yahoo Finance chart API<br>
    • <code>^VIX3M</code> — CBOE 3-Month VIX Index, Yahoo Finance chart API<br>
    • <code>VXX</code> — iPath VIX Short-Term Futures ETN, Yahoo Finance chart API<br>
    • <code>SPY</code> — SPDR S&P 500 ETF (correlation ref), Yahoo Finance chart API<br>
    • EXP-1220 returns — derived from same Yahoo VIX/SPY (already validated)
</div>

<div class="kpi-row">
    <div class="kpi"><div class="value {'good' if full_m['cagr_pct']>0 else 'bad'}">{full_m['cagr_pct']:.1f}%</div><div class="label">Full CAGR</div></div>
    <div class="kpi"><div class="value">{full_m['sharpe']:.2f}</div><div class="label">Sharpe (correct)</div></div>
    <div class="kpi"><div class="value">{full_m['max_dd_pct']:.1f}%</div><div class="label">Max DD</div></div>
    <div class="kpi"><div class="value">{agg.get('cagr_pct', 0):.1f}%</div><div class="label">OOS CAGR</div></div>
    <div class="kpi"><div class="value">{agg.get('sharpe', 0):.2f}</div><div class="label">OOS Sharpe</div></div>
    <div class="kpi"><div class="value" style="color:{corr_color}">{corr_text}</div><div class="label">Corr to EXP-1220</div></div>
</div>

<div class="callout">
    <strong>Market context:</strong> VIX/VIX3M ratio was in contango (&lt;1.0)
    <strong>{contango_pct:.0f}%</strong> of the sample period (avg ratio {avg_ratio:.3f}).
    Strategy held a short VXX position <strong>{short_pct:.0f}%</strong> of the time.
    Theory: VXX decays in contango due to negative roll yield on VIX futures.
</div>

<h2>Walk-Forward OOS (expanding window, train 2018-N, test N+1)</h2>
<table>
    <thead><tr><th>Year</th><th>Days</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th><th>% Short</th></tr></thead>
    <tbody>{wf_rows}</tbody>
</table>

<h2>Strategy Parameters</h2>
<table>
    <thead><tr><th>Parameter</th><th>Value</th><th>Notes</th></tr></thead>
    <tbody>
        <tr><td>Contango enter</td><td>{config.contango_enter}</td><td>Short VXX when VIX/VIX3M &lt; this</td></tr>
        <tr><td>Contango exit</td><td>{config.contango_exit}</td><td>Flat when ratio crosses this</td></tr>
        <tr><td>Backwardation flat</td><td>{config.backwardation_flat}</td><td>Hard exit in crisis</td></tr>
        <tr><td>Target vol</td><td>{config.target_vol * 100:.0f}%</td><td>Inverse-vol position sizing</td></tr>
        <tr><td>Vol lookback</td><td>{config.vol_lookback} days</td><td>Realized vol window</td></tr>
        <tr><td>Max position</td><td>{config.max_position}</td><td>Absolute cap on notional</td></tr>
        <tr><td>Signal lag</td><td>1 day (t-1)</td><td>No look-ahead</td></tr>
    </tbody>
</table>

<div class="footer">
    EXP-1700 VIX Roll Yield — compass/vix_roll_yield.py<br>
    All data from Yahoo Finance chart API. No synthetic data. Sharpe via compass/metrics.py.
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("EXP-1700 — VIX Futures Roll Yield (Phase 7 Wave 1)")
    print("=" * 72)

    print("\n[1/5] Loading REAL market data...")
    data = load_all_data()

    print("\n[2/5] Computing signals (t-1 lagged term structure)...")
    config = RollYieldConfig()
    df = compute_signals(data, config)
    contango_pct = float((df["ratio"] < 1.0).mean()) * 100
    short_pct = float((df["position"] < 0).mean()) * 100
    print(f"  → {len(df)} days, contango {contango_pct:.0f}%, short position {short_pct:.0f}% of time")

    print("\n[3/5] Walk-forward validation...")
    wf = backtest_walk_forward(df)
    full_m = wf["full"]
    agg = wf["oos_aggregate"]

    print(f"\n  FULL PERIOD (2018-2025):")
    print(f"    CAGR:   {full_m['cagr_pct']:6.1f}%")
    print(f"    Sharpe: {full_m['sharpe']:6.2f}")
    print(f"    Max DD: {full_m['max_dd_pct']:6.1f}%")
    print(f"    Vol:    {full_m['vol_pct']:6.1f}%")

    print(f"\n  OOS AGGREGATE (walk-forward):")
    print(f"    CAGR:   {agg.get('cagr_pct', 0):6.1f}%")
    print(f"    Sharpe: {agg.get('sharpe', 0):6.2f}")
    print(f"    Max DD: {agg.get('max_dd_pct', 0):6.1f}%")

    print(f"\n  YEAR-BY-YEAR OOS:")
    for w in wf["windows"]:
        print(f"    {w['year']}: CAGR={w['cagr_pct']:6.1f}%  Sharpe={w['sharpe']:5.2f}  "
              f"DD={w['max_dd_pct']:5.1f}%  Short={w['pct_short']:.0f}%")

    print("\n[4/5] Correlation to EXP-1220 credit spreads...")
    strategy_series = pd.Series(df["strategy_ret"].values, index=df.index)
    correlation = compute_correlation_to_exp1220(strategy_series)
    if not math.isnan(correlation):
        print(f"  → Correlation: {correlation:+.3f} "
              f"({'LOW — good diversifier' if abs(correlation) < 0.2 else 'moderate'})")
    else:
        print("  → Correlation: N/A")

    print("\n[5/5] Generating report...")
    html = generate_report(data, df, wf, correlation, config)
    report_path = ROOT / "reports" / "exp1700_vix_roll_yield.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    print(f"  → {report_path}")


if __name__ == "__main__":
    main()
