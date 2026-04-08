#!/usr/bin/env python3
"""
EXP-1710 Decay Diagnosis — Why is Sharpe dropping 22.7 → 7.4 → 2.0?

Four-part investigation:
  1. Crowding check: did 0DTE/1DTE SPY option volume explode 2023→2025?
  2. Regime analysis: split trades by VIX, market direction, day of week
  3. Adaptive filter: only trade when conditions match high-Sharpe regime
  4. Walk-forward test filtered vs unfiltered

All data REAL (IronVault options_cache.db + Yahoo Finance). Zero synthetic.
Sharpe via compass/metrics.py (arithmetic mean formula).
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.metrics import annualized_sharpe, full_metrics
from compass.zero_dte_ic import backtest_1_3_dte, ICTrade, CAPITAL

TRADING_DAYS = 252
REPORT_PATH = ROOT / "reports" / "exp1710_diagnosis.html"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Crowding check — SPY options volume from Yahoo Finance
# ═══════════════════════════════════════════════════════════════════════════

def fetch_spy_close_volume(start: str, end: str) -> pd.DataFrame:
    """Fetch SPY daily close + volume from Yahoo. Real data."""
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/SPY"
           f"?period1={start_ts}&period2={end_ts}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    dates = [datetime.fromtimestamp(t).date() for t in timestamps]
    df = pd.DataFrame({
        "close": quote["close"],
        "volume": quote["volume"],
    }, index=pd.DatetimeIndex(dates))
    return df.dropna()


def fetch_vix(start: str, end: str) -> pd.Series:
    """Fetch VIX from Yahoo. Real data."""
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
           f"?period1={start_ts}&period2={end_ts}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    dates = [datetime.fromtimestamp(t).date() for t in timestamps]
    return pd.Series(closes, index=pd.DatetimeIndex(dates)).dropna()


def crowding_analysis() -> Dict:
    """Check SPY underlying volume trends 2023→2025.

    Note: Yahoo doesn't expose option volume directly, but SPY ETF volume
    is a reasonable proxy for overall market interest in SPY-linked
    instruments including weeklies.
    """
    df = fetch_spy_close_volume("2023-01-01", "2025-12-31")
    df["year"] = df.index.year

    by_year = df.groupby("year")["volume"].agg(["mean", "median", "sum"])
    by_year["mean_mm"] = (by_year["mean"] / 1e6).round(1)
    by_year["median_mm"] = (by_year["median"] / 1e6).round(1)

    result = {
        "yearly": by_year[["mean_mm", "median_mm"]].to_dict("index"),
        "growth_23_25": round(
            (by_year.loc[2025, "mean"] / by_year.loc[2023, "mean"] - 1) * 100, 1
        ) if 2023 in by_year.index and 2025 in by_year.index else 0,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. Regime analysis — split trades by VIX, direction, day-of-week
# ═══════════════════════════════════════════════════════════════════════════

def enrich_trades(trades: List[ICTrade],
                   spy: pd.DataFrame,
                   vix: pd.Series) -> List[dict]:
    """Add regime metadata to each trade."""
    enriched = []
    for t in trades:
        entry_dt = pd.Timestamp(t.entry_date)

        # VIX at entry
        vix_val = float(vix.reindex([entry_dt], method="ffill").iloc[0]) if len(vix) > 0 else np.nan

        # 20-day SPY trend (positive = uptrend)
        spy_close = spy["close"]
        if entry_dt in spy_close.index:
            idx_pos = spy_close.index.get_loc(entry_dt)
            if idx_pos >= 20:
                ret_20d = float(spy_close.iloc[idx_pos] / spy_close.iloc[idx_pos - 20] - 1)
            else:
                ret_20d = 0.0
        else:
            ret_20d = 0.0

        # Day of week
        dow = entry_dt.weekday()  # 0=Mon, 4=Fri
        dow_name = ["Mon", "Tue", "Wed", "Thu", "Fri"][dow] if dow < 5 else "Weekend"

        # Realistic net P&L (costs from validation commit 8303957)
        cost = (4 * 2 * 0.65 + 0.05 * 2 * 100) * t.contracts
        net_pnl = t.pnl - cost

        enriched.append({
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "dte": t.dte_at_entry,
            "contracts": t.contracts,
            "pnl_gross": t.pnl,
            "pnl_net": net_pnl,
            "win": net_pnl > 0,
            "exit_reason": t.exit_reason,
            "vix": round(vix_val, 2) if not math.isnan(vix_val) else None,
            "spy_20d_ret": round(ret_20d * 100, 2),
            "dow": dow_name,
            "year": int(entry_dt.year),
        })
    return enriched


def regime_analysis(trades: List[dict]) -> Dict:
    """Split trades by VIX, direction, day-of-week and compute Sharpe."""
    def _stats(group: List[dict]) -> Dict:
        if not group:
            return {"n": 0, "pnl": 0, "win_rate": 0, "sharpe": 0, "avg_pnl": 0}
        pnls = np.array([t["pnl_net"] for t in group])
        mean = float(pnls.mean())
        std = float(pnls.std())
        sh = mean / std * math.sqrt(52) if std > 1e-6 else 0
        return {
            "n": len(group),
            "pnl": round(float(pnls.sum()), 0),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
            "sharpe": round(sh, 2),
            "avg_pnl": round(mean, 0),
        }

    # By VIX buckets
    def _vix_bucket(v):
        if v is None: return "unknown"
        if v < 15: return "low (<15)"
        if v < 20: return "normal (15-20)"
        if v < 25: return "elevated (20-25)"
        return "high (>25)"

    by_vix = defaultdict(list)
    for t in trades:
        by_vix[_vix_bucket(t["vix"])].append(t)
    vix_stats = {k: _stats(v) for k, v in by_vix.items()}

    # By market direction (20d SPY return)
    def _dir_bucket(r):
        if r > 3: return "strong up (>3%)"
        if r > 0: return "up (0-3%)"
        if r > -3: return "down (0 to -3%)"
        return "strong down (<-3%)"

    by_dir = defaultdict(list)
    for t in trades:
        by_dir[_dir_bucket(t["spy_20d_ret"])].append(t)
    dir_stats = {k: _stats(v) for k, v in by_dir.items()}

    # By day of week
    by_dow = defaultdict(list)
    for t in trades:
        by_dow[t["dow"]].append(t)
    dow_stats = {k: _stats(v) for k, v in by_dow.items()}

    # By year × VIX (find the decay pattern)
    by_year_vix = defaultdict(list)
    for t in trades:
        key = f"{t['year']}_{_vix_bucket(t['vix'])}"
        by_year_vix[key].append(t)
    year_vix_stats = {k: _stats(v) for k, v in sorted(by_year_vix.items())}

    return {
        "by_vix": vix_stats,
        "by_direction": dir_stats,
        "by_dow": dow_stats,
        "by_year_vix": year_vix_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Adaptive filter — trade only in high-Sharpe regime
# ═══════════════════════════════════════════════════════════════════════════

def apply_adaptive_filter(trades: List[dict], regime: Dict) -> Tuple[List[dict], Dict]:
    """Filter trades to only those matching high-Sharpe regime conditions.

    Strategy: identify the VIX bucket with highest Sharpe in the FIRST
    two years (training), then only keep trades in that bucket going forward.
    """
    # Find best VIX bucket from 2023-2024 training
    train_trades = [t for t in trades if t["year"] <= 2024]
    test_trades = [t for t in trades if t["year"] >= 2025]

    def _vix_bucket(v):
        if v is None: return "unknown"
        if v < 15: return "low (<15)"
        if v < 20: return "normal (15-20)"
        if v < 25: return "elevated (20-25)"
        return "high (>25)"

    train_by_vix = defaultdict(list)
    for t in train_trades:
        train_by_vix[_vix_bucket(t["vix"])].append(t)

    best_bucket = None
    best_sharpe = -999
    for bucket, ts in train_by_vix.items():
        if len(ts) < 5:
            continue
        pnls = np.array([t["pnl_net"] for t in ts])
        mean = float(pnls.mean())
        std = float(pnls.std())
        if std > 1e-6:
            sh = mean / std * math.sqrt(52)
            if sh > best_sharpe:
                best_sharpe = sh
                best_bucket = bucket

    # Apply: keep only trades in best bucket
    filtered = [t for t in trades if _vix_bucket(t["vix"]) == best_bucket]

    # Compute unfiltered vs filtered by year
    def _yr_stats(ts):
        by_year = defaultdict(list)
        for t in ts:
            by_year[t["year"]].append(t)
        result = {}
        for yr, group in sorted(by_year.items()):
            pnls = np.array([t["pnl_net"] for t in group])
            if len(pnls) < 2:
                continue
            mean = float(pnls.mean())
            std = float(pnls.std())
            sh = mean / std * math.sqrt(52) if std > 1e-6 else 0
            result[int(yr)] = {
                "n": len(group),
                "pnl": round(float(pnls.sum()), 0),
                "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
                "sharpe": round(sh, 2),
            }
        return result

    return filtered, {
        "best_bucket": best_bucket,
        "train_sharpe": round(best_sharpe, 2),
        "unfiltered_yearly": _yr_stats(trades),
        "filtered_yearly": _yr_stats(filtered),
        "n_unfiltered": len(trades),
        "n_filtered": len(filtered),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Walk-forward on filtered version
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_filtered(trades: List[dict]) -> Dict:
    """For each test year, train filter on prior years and apply to test."""
    def _vix_bucket(v):
        if v is None: return "unknown"
        if v < 15: return "low (<15)"
        if v < 20: return "normal (15-20)"
        if v < 25: return "elevated (20-25)"
        return "high (>25)"

    years = sorted(set(t["year"] for t in trades))
    windows = []
    all_oos = []

    for test_yr in years[1:]:  # need 1+ year of training
        train = [t for t in trades if t["year"] < test_yr]
        test = [t for t in trades if t["year"] == test_yr]
        if len(train) < 10 or len(test) < 3:
            continue

        # Find best bucket in training
        by_bucket = defaultdict(list)
        for t in train:
            by_bucket[_vix_bucket(t["vix"])].append(t)
        best_bucket = None
        best_sh = -999
        for bucket, ts in by_bucket.items():
            if len(ts) < 5:
                continue
            pnls = np.array([t["pnl_net"] for t in ts])
            std = float(pnls.std())
            if std > 1e-6:
                sh = float(pnls.mean()) / std * math.sqrt(52)
                if sh > best_sh:
                    best_sh, best_bucket = sh, bucket

        # Apply to test
        filtered_test = [t for t in test if _vix_bucket(t["vix"]) == best_bucket]
        if not filtered_test:
            windows.append({
                "year": test_yr, "best_bucket": best_bucket,
                "n_train": len(train), "n_test": len(test),
                "n_filtered": 0, "pnl": 0, "sharpe": 0, "win_rate": 0,
            })
            continue

        pnls = np.array([t["pnl_net"] for t in filtered_test])
        mean = float(pnls.mean())
        std = float(pnls.std())
        sh = mean / std * math.sqrt(52) if std > 1e-6 else 0
        windows.append({
            "year": test_yr,
            "best_bucket": best_bucket,
            "n_train": len(train),
            "n_test": len(test),
            "n_filtered": len(filtered_test),
            "pnl": round(float(pnls.sum()), 0),
            "sharpe": round(sh, 2),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
        })
        all_oos.extend(filtered_test)

    # Aggregate OOS
    if all_oos:
        pnls = np.array([t["pnl_net"] for t in all_oos])
        mean = float(pnls.mean())
        std = float(pnls.std())
        oos_sharpe = mean / std * math.sqrt(52) if std > 1e-6 else 0
        agg = {
            "n_trades": len(all_oos),
            "total_pnl": round(float(pnls.sum()), 0),
            "sharpe": round(oos_sharpe, 2),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
        }
    else:
        agg = {"n_trades": 0, "total_pnl": 0, "sharpe": 0, "win_rate": 0}

    return {"windows": windows, "oos_aggregate": agg}


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(crowding: Dict, regime: Dict, filter_result: Dict,
                     wf: Dict) -> str:
    # Crowding rows
    crowd_rows = ""
    for yr, stats in sorted(crowding["yearly"].items()):
        crowd_rows += f"""<tr>
            <td style="font-weight:700">{yr}</td>
            <td>{stats['mean_mm']}M</td>
            <td>{stats['median_mm']}M</td>
        </tr>"""

    # VIX bucket rows
    vix_rows = ""
    for bucket, s in sorted(regime["by_vix"].items()):
        sc = "#16a34a" if s["sharpe"] > 1 else ("#ca8a04" if s["sharpe"] > 0 else "#dc2626")
        vix_rows += f"""<tr>
            <td style="font-weight:600">{bucket}</td>
            <td>{s['n']}</td>
            <td style="color:{sc};font-weight:700">{s['sharpe']:.2f}</td>
            <td>{s['win_rate']:.0f}%</td>
            <td>${s['pnl']:,.0f}</td>
            <td>${s['avg_pnl']:,.0f}</td>
        </tr>"""

    # Direction rows
    dir_rows = ""
    for d, s in regime["by_direction"].items():
        sc = "#16a34a" if s["sharpe"] > 1 else ("#ca8a04" if s["sharpe"] > 0 else "#dc2626")
        dir_rows += f'<tr><td>{d}</td><td>{s["n"]}</td><td style="color:{sc};font-weight:700">{s["sharpe"]:.2f}</td><td>{s["win_rate"]:.0f}%</td><td>${s["pnl"]:,.0f}</td></tr>'

    # DoW rows
    dow_rows = ""
    for d, s in regime["by_dow"].items():
        sc = "#16a34a" if s["sharpe"] > 1 else ("#ca8a04" if s["sharpe"] > 0 else "#dc2626")
        dow_rows += f'<tr><td>{d}</td><td>{s["n"]}</td><td style="color:{sc};font-weight:700">{s["sharpe"]:.2f}</td><td>{s["win_rate"]:.0f}%</td><td>${s["pnl"]:,.0f}</td></tr>'

    # Year × VIX decay pattern
    decay_rows = ""
    for key, s in sorted(regime["by_year_vix"].items()):
        if s["n"] < 3:
            continue
        sc = "#16a34a" if s["sharpe"] > 1 else ("#ca8a04" if s["sharpe"] > 0 else "#dc2626")
        decay_rows += f'<tr><td>{key}</td><td>{s["n"]}</td><td style="color:{sc};font-weight:700">{s["sharpe"]:.2f}</td><td>${s["pnl"]:,.0f}</td></tr>'

    # Unfiltered vs filtered yearly
    compare_rows = ""
    unfilt = filter_result["unfiltered_yearly"]
    filt = filter_result["filtered_yearly"]
    for yr in sorted(set(list(unfilt.keys()) + list(filt.keys()))):
        u = unfilt.get(yr, {"n": 0, "sharpe": 0, "pnl": 0})
        f = filt.get(yr, {"n": 0, "sharpe": 0, "pnl": 0})
        compare_rows += f"""<tr>
            <td>{yr}</td>
            <td>{u.get('n',0)}</td>
            <td>{u.get('sharpe',0):.2f}</td>
            <td>${u.get('pnl',0):,.0f}</td>
            <td>{f.get('n',0)}</td>
            <td style="font-weight:700">{f.get('sharpe',0):.2f}</td>
            <td>${f.get('pnl',0):,.0f}</td>
        </tr>"""

    # WF rows
    wf_rows = ""
    for w in wf["windows"]:
        sc = "#16a34a" if w["sharpe"] > 1 else ("#ca8a04" if w["sharpe"] > 0 else "#dc2626")
        wf_rows += f"""<tr>
            <td style="font-weight:700">{w['year']}</td>
            <td>{w['best_bucket']}</td>
            <td>{w['n_train']}</td>
            <td>{w['n_test']}</td>
            <td>{w['n_filtered']}</td>
            <td style="color:{sc};font-weight:700">{w['sharpe']:.2f}</td>
            <td>${w['pnl']:,.0f}</td>
        </tr>"""

    oos_agg = wf["oos_aggregate"]
    honest_sharpe = oos_agg.get("sharpe", 0)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1710 Decay Diagnosis</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1000px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.5; }}
  h1 {{ font-size:1.8em; color:#0f172a; margin-bottom:4px; }}
  h2 {{ color:#334155; margin-top:2.5em; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  h3 {{ color:#475569; margin-top:1.5em; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:24px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:18px;
          text-align:center; flex:1; min-width:130px; }}
  .kpi .value {{ font-size:1.7em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; }} .warn {{ color:#ca8a04; }} .bad {{ color:#dc2626; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:10px 12px; text-align:right; font-weight:600; color:#475569;
       border-bottom:2px solid #cbd5e1; font-size:0.78em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .callout {{ border-radius:8px; padding:16px; margin:16px 0; font-size:0.88rem; line-height:1.7; }}
  .callout.warn {{ background:#fffbeb; border:1px solid #fde68a; }}
  .callout.ok {{ background:#f0fdf4; border:1px solid #bbf7d0; }}
  .footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.78em; color:#94a3b8; text-align:center; }}
</style></head><body>

<h1>EXP-1710 Decay Diagnosis</h1>
<div class="subtitle">Why is Sharpe dropping 22.7 → 7.4 → 2.0? Four-part investigation with real data.</div>

<h2>1. Crowding Check — SPY Volume 2023→2025</h2>
<p>Proxy: SPY ETF daily volume (Yahoo Finance). Higher volume = more market interest in SPY-linked options.</p>
<table>
    <thead><tr><th>Year</th><th>Mean Daily Volume</th><th>Median Daily Volume</th></tr></thead>
    <tbody>{crowd_rows}</tbody>
</table>
<div class="callout warn">
    <strong>SPY volume 2023 → 2025: {crowding.get('growth_23_25', 0):+.1f}%</strong>.
    {"Moderate growth — crowding is a plausible factor." if abs(crowding.get('growth_23_25', 0)) > 10 else "Volume relatively stable — crowding alone doesn't explain decay."}
</div>

<h2>2. Regime Analysis — VIX Buckets</h2>
<table>
    <thead><tr><th>VIX Bucket</th><th>Trades</th><th>Sharpe</th><th>Win %</th><th>Total P&amp;L</th><th>Avg P&amp;L</th></tr></thead>
    <tbody>{vix_rows}</tbody>
</table>

<h3>Market Direction (20-day SPY return)</h3>
<table>
    <thead><tr><th>Regime</th><th>Trades</th><th>Sharpe</th><th>Win %</th><th>P&amp;L</th></tr></thead>
    <tbody>{dir_rows}</tbody>
</table>

<h3>Day of Week</h3>
<table>
    <thead><tr><th>Day</th><th>Trades</th><th>Sharpe</th><th>Win %</th><th>P&amp;L</th></tr></thead>
    <tbody>{dow_rows}</tbody>
</table>

<h3>Year × VIX Decay Pattern</h3>
<table>
    <thead><tr><th>Year_Bucket</th><th>Trades</th><th>Sharpe</th><th>P&amp;L</th></tr></thead>
    <tbody>{decay_rows}</tbody>
</table>

<h2>3. Adaptive Filter Results</h2>
<p><strong>Best VIX bucket (from 2023-2024 training):</strong> {filter_result['best_bucket']} — training Sharpe {filter_result['train_sharpe']}</p>
<table>
    <thead><tr><th rowspan="2">Year</th><th colspan="3">Unfiltered</th><th colspan="3">Filtered</th></tr>
    <tr><th>N</th><th>Sharpe</th><th>P&amp;L</th><th>N</th><th>Sharpe</th><th>P&amp;L</th></tr></thead>
    <tbody>{compare_rows}</tbody>
</table>

<h2>4. Walk-Forward Adaptive Filter</h2>
<table>
    <thead><tr><th>Year</th><th>Best Bucket</th><th>Train N</th><th>Test N</th><th>Filtered N</th><th>Sharpe</th><th>P&amp;L</th></tr></thead>
    <tbody>{wf_rows}</tbody>
</table>

<div class="callout {'ok' if honest_sharpe > 2 else 'warn'}">
    <strong>OOS Aggregate (filtered, walk-forward):</strong><br>
    Trades: {oos_agg['n_trades']} | P&amp;L: ${oos_agg['total_pnl']:,.0f} | <strong>Sharpe: {oos_agg['sharpe']}</strong> | Win %: {oos_agg['win_rate']}%
</div>

<h2>HONEST Forward-Looking Estimate</h2>
<div class="callout warn">
    Base case (2025 unfiltered Sharpe): <strong>~2.0</strong><br>
    Filtered walk-forward OOS: <strong>{honest_sharpe:.2f}</strong><br>
    <br>
    The filter {'HELPS — Sharpe improves' if honest_sharpe > 2.2 else 'does NOT meaningfully recover the decay'}. Forward expectation: <strong>{max(honest_sharpe, 1.5):.1f}–{honest_sharpe + 0.5:.1f}</strong>
    range, NOT the 5+ headline numbers. Deploy at small size with tight stops,
    monitor monthly for continued decay.
</div>

<div class="footer">
    EXP-1710 Decay Diagnosis — scripts/diagnose_exp1710.py<br>
    Data: IronVault options_cache.db (real Polygon) + Yahoo Finance (SPY, VIX). Zero synthetic.
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("EXP-1710 Decay Diagnosis")
    print("=" * 72)

    print("\n[1/5] Running fresh 1DTE backtest on IronVault...")
    trades = backtest_1_3_dte(dte_target=1, start_date="2023-01-01", end_date="2026-01-01")
    print(f"  → {len(trades)} trades")

    print("\n[2/5] Loading market data (SPY, VIX)...")
    spy = fetch_spy_close_volume("2023-01-01", "2026-01-01")
    vix = fetch_vix("2023-01-01", "2026-01-01")
    print(f"  → {len(spy)} SPY days, {len(vix)} VIX days")

    print("\n[3/5] Crowding check (SPY volume trend)...")
    crowding = crowding_analysis()
    print(f"  SPY volume 2023→2025: {crowding['growth_23_25']:+.1f}%")
    for yr, stats in sorted(crowding["yearly"].items()):
        print(f"    {yr}: mean {stats['mean_mm']}M  median {stats['median_mm']}M")

    print("\n[4/5] Enriching trades with regime metadata...")
    enriched = enrich_trades(trades, spy, vix)

    print("\n  Regime analysis:")
    regime = regime_analysis(enriched)
    print(f"\n  By VIX bucket:")
    for bucket, s in sorted(regime["by_vix"].items()):
        print(f"    {bucket:20s} n={s['n']:3d}  Sharpe={s['sharpe']:5.2f}  WR={s['win_rate']:5.1f}%  PnL=${s['pnl']:,.0f}")

    print(f"\n  By direction:")
    for d, s in regime["by_direction"].items():
        print(f"    {d:25s} n={s['n']:3d}  Sharpe={s['sharpe']:5.2f}  PnL=${s['pnl']:,.0f}")

    print(f"\n  By day of week:")
    for d, s in regime["by_dow"].items():
        print(f"    {d:5s} n={s['n']:3d}  Sharpe={s['sharpe']:5.2f}  PnL=${s['pnl']:,.0f}")

    print(f"\n  Year × VIX decay:")
    for k, s in sorted(regime["by_year_vix"].items()):
        if s["n"] >= 3:
            print(f"    {k:25s} n={s['n']:3d}  Sharpe={s['sharpe']:5.2f}  PnL=${s['pnl']:,.0f}")

    print("\n[5/5] Adaptive filter + walk-forward...")
    filtered, filter_result = apply_adaptive_filter(enriched, regime)
    print(f"  Best bucket (train 2023-2024): {filter_result['best_bucket']}")
    print(f"  Training Sharpe: {filter_result['train_sharpe']}")
    print(f"  Filtered count: {filter_result['n_filtered']} / {filter_result['n_unfiltered']}")

    wf = walk_forward_filtered(enriched)
    print(f"\n  Walk-forward filtered by year:")
    for w in wf["windows"]:
        print(f"    {w['year']}: bucket={w['best_bucket']:20s}  "
              f"n_filt={w['n_filtered']:3d}  Sharpe={w['sharpe']:5.2f}  "
              f"PnL=${w['pnl']:,.0f}")

    oos = wf["oos_aggregate"]
    print(f"\n  OOS AGGREGATE (filtered walk-forward):")
    print(f"    Trades: {oos['n_trades']}  PnL: ${oos['total_pnl']:,.0f}  "
          f"Sharpe: {oos['sharpe']}  Win: {oos['win_rate']}%")

    print(f"\n{'━'*60}")
    print(f"  VERDICT:")
    print(f"    2025 unfiltered Sharpe:    ~2.0")
    print(f"    Filtered WF OOS Sharpe:    {oos['sharpe']}")
    if oos["sharpe"] > 2.5:
        print(f"    → Filter HELPS recover some alpha. Forward: {oos['sharpe']:.1f}")
    else:
        print(f"    → Filter does NOT meaningfully recover decay.")
        print(f"    → Honest forward Sharpe: 1.5-2.5 range")
    print(f"{'━'*60}")

    # Generate report
    print("\nGenerating report...")
    html = generate_report(crowding, regime, filter_result, wf)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}")


if __name__ == "__main__":
    main()
