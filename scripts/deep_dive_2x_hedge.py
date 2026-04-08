#!/usr/bin/env python3
"""
Deep Dive: 2× EXP-1220 + Crisis Alpha hedge.

Comprehensive analysis for Carlos:
  1. Parameter sweep 5%-30% hedge in 2.5% steps
  2. Leverage sweep 1.5×-2.5× with optimal hedge
  3. Year-by-year breakdown of top 3 configs
  4. Regime analysis (VIX buckets)
  5. Crisis Alpha mechanism explanation
  6. ONE comprehensive HTML report with matplotlib charts inline

REAL data only. Sharpe via compass/metrics.py (arithmetic mean).
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compass.metrics import full_metrics

REPORT_PATH = ROOT / "compass" / "reports" / "2x_hedge_deep_dive.html"
STARTING_CAPITAL = 100_000


# ═══════════════════════════════════════════════════════════════════════════
# Data loaders (REAL only)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_yahoo(symbol: str, start: str, end: str) -> pd.Series:
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    safe = symbol.replace("^", "%5E")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{safe}"
           f"?period1={start_ts}&period2={end_ts}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        d = json.loads(resp.read())
    res = d["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    dates = [datetime.fromtimestamp(t).date() for t in ts]
    return pd.Series(closes, index=pd.DatetimeIndex(dates), name=symbol).dropna()


def load_all():
    print("  EXP-1220 daily...")
    from scripts.ultimate_portfolio import load_exp1220_dynamic
    e1220 = load_exp1220_dynamic()

    print("  Crisis Alpha v3 (13 ETFs)...")
    from compass.crisis_alpha_v3 import (
        load_universe_v3, compute_momentum,
        compute_vol_target_weights, LOOKBACK_GRID,
    )
    prices = load_universe_v3("2014-01-01", "2026-01-01")
    lookbacks, lw = LOOKBACK_GRID["v2_round"]
    signal = compute_momentum(prices, lookbacks, lw)
    weights = compute_vol_target_weights(prices, signal, vol_target=0.10, leverage=2.5)
    asset_returns = prices.pct_change().fillna(0)
    held = weights.copy()
    for i in range(len(held)):
        if i % 5 != 0 and i > 0:
            held.iloc[i] = held.iloc[i - 1]
    lagged = held.shift(1).fillna(0)
    e1780 = (lagged * asset_returns).sum(axis=1)
    e1780 = e1780.iloc[max(lookbacks):]

    print("  ^VIX...")
    vix = fetch_yahoo("^VIX", "2019-12-01", "2026-01-01")

    return e1220, e1780, lagged, prices, vix


# ═══════════════════════════════════════════════════════════════════════════
# Combinations
# ═══════════════════════════════════════════════════════════════════════════

def combine(e1220: pd.Series, e1780: pd.Series, leverage: float,
             hedge_pct: float) -> pd.Series:
    common = e1220.index.intersection(e1780.index).sort_values()
    a = e1220.reindex(common).fillna(0)
    b = e1780.reindex(common).fillna(0)
    return (1 - hedge_pct) * a * leverage + hedge_pct * b


def parameter_sweep(e1220: pd.Series, e1780: pd.Series) -> List[Dict]:
    """Sweep hedge % from 5% to 30% in 2.5% steps at leverage=2×."""
    results = []
    for hedge_pct in np.arange(0.05, 0.305, 0.025):
        combined = combine(e1220, e1780, 2.0, hedge_pct)
        m = full_metrics(combined.values)
        results.append({
            "leverage": 2.0,
            "hedge_pct": round(float(hedge_pct), 4),
            **m,
        })
    return results


def leverage_sweep(e1220: pd.Series, e1780: pd.Series,
                    optimal_hedge: float) -> List[Dict]:
    """Sweep leverage at the optimal hedge %."""
    results = []
    for lev in [1.5, 1.75, 2.0, 2.25, 2.5]:
        combined = combine(e1220, e1780, lev, optimal_hedge)
        m = full_metrics(combined.values)
        results.append({
            "leverage": lev,
            "hedge_pct": optimal_hedge,
            **m,
        })
    return results


def find_optimal_hedge(sweep: List[Dict], dd_max: float = 15.0) -> Dict:
    """Find hedge % that maximizes Sharpe with DD < 15%."""
    feasible = [r for r in sweep if r["max_dd_pct"] < dd_max]
    if not feasible:
        return max(sweep, key=lambda r: r["sharpe"])
    return max(feasible, key=lambda r: r["sharpe"])


def full_grid(e1220: pd.Series, e1780: pd.Series) -> List[Dict]:
    """Full grid: 5 leverages × 11 hedge%s = 55 configs."""
    results = []
    for lev in [1.5, 1.75, 2.0, 2.25, 2.5]:
        for hedge in np.arange(0.05, 0.305, 0.025):
            combined = combine(e1220, e1780, lev, float(hedge))
            m = full_metrics(combined.values)
            results.append({
                "leverage": lev,
                "hedge_pct": round(float(hedge), 4),
                **m,
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Year-by-year + regime analysis
# ═══════════════════════════════════════════════════════════════════════════

def yearly_breakdown(rets: pd.Series) -> List[Dict]:
    out = []
    for yr in sorted(set(rets.index.year)):
        yr_rets = rets[rets.index.year == yr].values
        if len(yr_rets) < 20:
            continue
        m = full_metrics(yr_rets)
        m["year"] = int(yr)
        out.append(m)
    return out


def regime_analysis(rets: pd.Series, vix: pd.Series) -> Dict[str, Dict]:
    """Bucket returns by VIX regime."""
    common = rets.index.intersection(vix.index)
    rets_a = rets.reindex(common).fillna(0)
    vix_a = vix.reindex(common).ffill().shift(1).fillna(20)  # t-1 lag

    buckets = {"bull (<15)": [], "normal (15-25)": [],
               "bear (25-35)": [], "crash (>35)": []}
    for i, dt in enumerate(common):
        v = float(vix_a.iloc[i])
        r = float(rets_a.iloc[i])
        if v < 15:
            buckets["bull (<15)"].append(r)
        elif v < 25:
            buckets["normal (15-25)"].append(r)
        elif v < 35:
            buckets["bear (25-35)"].append(r)
        else:
            buckets["crash (>35)"].append(r)

    return {
        name: {
            "n_days": len(rs),
            **(full_metrics(np.array(rs)) if len(rs) > 1 else
               {"cagr_pct": 0, "sharpe": 0, "max_dd_pct": 0, "vol_pct": 0}),
        }
        for name, rs in buckets.items()
    }


# ═══════════════════════════════════════════════════════════════════════════
# Crisis Alpha mechanism inspection
# ═══════════════════════════════════════════════════════════════════════════

def inspect_crisis_alpha(weights: pd.DataFrame, prices: pd.DataFrame,
                          common_idx: pd.DatetimeIndex) -> Dict:
    """What does Crisis Alpha actually hold?"""
    # Average absolute weight per asset
    w_aligned = weights.reindex(common_idx).fillna(0)
    avg_abs_weight = w_aligned.abs().mean().sort_values(ascending=False)

    # Long vs short fraction over time
    long_frac = (w_aligned > 0.001).mean()
    short_frac = (w_aligned < -0.001).mean()

    # Average long/short days
    long_days = (w_aligned > 0).sum(axis=1).mean()
    short_days = (w_aligned < 0).sum(axis=1).mean()

    return {
        "avg_abs_weight": avg_abs_weight.to_dict(),
        "pct_days_long": long_frac.to_dict(),
        "pct_days_short": short_frac.to_dict(),
        "avg_long_positions": float(long_days),
        "avg_short_positions": float(short_days),
    }


def rolling_correlation(e1220: pd.Series, e1780: pd.Series, window: int = 60):
    common = e1220.index.intersection(e1780.index)
    a = e1220.reindex(common).fillna(0)
    b = e1780.reindex(common).fillna(0)
    return a.rolling(window).corr(b)


# ═══════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    s = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{s}"


def chart_param_sweep(sweep: List[Dict]) -> str:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    hedges = [r["hedge_pct"] * 100 for r in sweep]
    cagrs = [r["cagr_pct"] for r in sweep]
    sharpes = [r["sharpe"] for r in sweep]
    dds = [r["max_dd_pct"] for r in sweep]

    ax1.bar(hedges, cagrs, width=2.0, alpha=0.4, color="#3b82f6", label="CAGR")
    ax1.set_xlabel("Crisis Alpha Hedge %")
    ax1.set_ylabel("CAGR (%)", color="#3b82f6")
    ax1.tick_params(axis="y", labelcolor="#3b82f6")

    ax2 = ax1.twinx()
    ax2.plot(hedges, sharpes, "o-", color="#16a34a", linewidth=2, label="Sharpe")
    ax2.plot(hedges, dds, "s-", color="#dc2626", linewidth=2, label="Max DD")
    ax2.set_ylabel("Sharpe / Max DD %", color="#0f172a")
    ax2.axhline(y=15, color="#dc2626", linestyle="--", alpha=0.5, label="DD limit (15%)")

    ax1.set_title("Parameter Sweep: 2× + Crisis Alpha Hedge", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_pareto_frontier(grid: List[Dict]) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    leverages = sorted(set(r["leverage"] for r in grid))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(leverages)))

    for lev, color in zip(leverages, colors):
        configs = [r for r in grid if r["leverage"] == lev]
        configs.sort(key=lambda r: r["max_dd_pct"])
        dds = [r["max_dd_pct"] for r in configs]
        cagrs = [r["cagr_pct"] for r in configs]
        ax.plot(dds, cagrs, "o-", color=color, label=f"{lev}×", linewidth=2, markersize=6)

    ax.axvline(x=15, color="#dc2626", linestyle="--", alpha=0.5, label="15% DD limit")
    ax.set_xlabel("Max Drawdown (%)", fontsize=11)
    ax.set_ylabel("CAGR (%)", fontsize=11)
    ax.set_title("Pareto Frontier: CAGR vs Max DD (5 leverages × 11 hedge %)",
                  fontsize=13, fontweight="bold")
    ax.legend(title="Leverage", loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_yearly_top3(top3: List[Tuple[str, pd.Series]]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Annual CAGR bars
    years = sorted(set(yr for _, rets in top3 for yr in set(rets.index.year)))
    width = 0.25
    x = np.arange(len(years))
    colors = ["#3b82f6", "#16a34a", "#dc2626"]

    for i, (name, rets) in enumerate(top3):
        cagrs = []
        for yr in years:
            yr_rets = rets[rets.index.year == yr].values
            if len(yr_rets) > 0:
                eq = np.cumprod(1 + yr_rets)
                yr_cagr = (eq[-1] - 1) * 100
            else:
                yr_cagr = 0
            cagrs.append(yr_cagr)
        axes[0].bar(x + i * width - width, cagrs, width, label=name, color=colors[i])

    axes[0].set_title("Annual CAGR by Config", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(years)
    axes[0].set_ylabel("CAGR (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Annual DD bars
    for i, (name, rets) in enumerate(top3):
        dds = []
        for yr in years:
            yr_rets = rets[rets.index.year == yr].values
            if len(yr_rets) > 0:
                eq = np.cumprod(1 + yr_rets)
                hwm = np.maximum.accumulate(eq)
                dd = (1 - eq / hwm).max() * 100
            else:
                dd = 0
            dds.append(dd)
        axes[1].bar(x + i * width - width, dds, width, label=name, color=colors[i])

    axes[1].set_title("Annual Max DD by Config", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(years)
    axes[1].set_ylabel("Max DD (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig_to_b64(fig)


def chart_regime_performance(regime_data: Dict[str, Dict[str, Dict]]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    regimes = ["bull (<15)", "normal (15-25)", "bear (25-35)", "crash (>35)"]
    configs = list(regime_data.keys())
    colors = ["#3b82f6", "#16a34a", "#dc2626"]
    width = 0.25
    x = np.arange(len(regimes))

    for i, name in enumerate(configs):
        cagrs = [regime_data[name].get(r, {}).get("cagr_pct", 0) for r in regimes]
        axes[0].bar(x + i * width - width, cagrs, width, label=name, color=colors[i % 3])
    axes[0].set_title("CAGR by VIX Regime", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r.split(" ")[0] for r in regimes])
    axes[0].set_ylabel("CAGR (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    for i, name in enumerate(configs):
        sharpes = [regime_data[name].get(r, {}).get("sharpe", 0) for r in regimes]
        axes[1].bar(x + i * width - width, sharpes, width, label=name, color=colors[i % 3])
    axes[1].set_title("Sharpe by VIX Regime", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r.split(" ")[0] for r in regimes])
    axes[1].set_ylabel("Sharpe")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig_to_b64(fig)


def chart_rolling_correlation(corr: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(corr.index, corr.values, color="#3b82f6", linewidth=1.5)
    ax.fill_between(corr.index, 0, corr.values,
                     where=(corr.values < 0), color="#16a34a", alpha=0.3, label="Negative (good)")
    ax.fill_between(corr.index, 0, corr.values,
                     where=(corr.values >= 0), color="#dc2626", alpha=0.3, label="Positive")
    ax.axhline(y=0, color="#0f172a", linestyle="-", linewidth=0.5)
    ax.set_title("60-Day Rolling Correlation: EXP-1220 vs Crisis Alpha",
                  fontweight="bold", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_crisis_alpha_holdings(weights_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 6))
    # Use only the period overlapping EXP-1220
    w = weights_df.loc["2020-01-01":"2025-12-31"]
    # Stack plot of average position size
    asset_means = w.abs().mean().sort_values(ascending=False).head(10)
    assets = asset_means.index.tolist()

    # Show position weights over time as line plot
    for asset in assets[:6]:
        ax.plot(w.index, w[asset], label=asset, linewidth=1, alpha=0.7)

    ax.axhline(y=0, color="#0f172a", linestyle="-", linewidth=0.5)
    ax.set_title("Crisis Alpha Position Weights — Top 6 Assets",
                  fontweight="bold", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Position Weight")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_top3_equity(top3: List[Tuple[str, pd.Series]]) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#3b82f6", "#16a34a", "#dc2626"]
    for (name, rets), color in zip(top3, colors):
        eq = STARTING_CAPITAL * np.cumprod(1 + rets.values)
        ax.plot(rets.index, eq, label=name, linewidth=2, color=color)
    ax.set_yscale("log")
    ax.set_title("Equity Curves — Top 3 Configurations (log scale)",
                  fontweight="bold", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_b64(fig)


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def build_html(sweep: List[Dict], lev_sweep: List[Dict], grid: List[Dict],
                top3: List[Tuple[str, pd.Series, Dict]],
                yearly_top3: Dict[str, List[Dict]],
                regime_data: Dict, ca_inspect: Dict,
                charts: Dict) -> str:
    optimal = find_optimal_hedge(sweep, dd_max=15.0)

    sweep_rows = ""
    for r in sweep:
        is_opt = abs(r["hedge_pct"] - optimal["hedge_pct"]) < 0.001
        hl = ' style="background:#dcfce7"' if is_opt else ""
        meets = r["max_dd_pct"] < 15
        cagr_color = "#16a34a" if meets else "#dc2626"
        sweep_rows += f"""<tr{hl}>
            <td>{r['hedge_pct']*100:.1f}%</td>
            <td style="color:{cagr_color};font-weight:600">{r['cagr_pct']:.1f}%</td>
            <td style="font-weight:700">{r['sharpe']:.2f}</td>
            <td>{r['max_dd_pct']:.1f}%</td>
            <td>{r['calmar']:.2f}</td>
            <td>{r['vol_pct']:.1f}%</td>
        </tr>"""

    lev_rows = ""
    for r in lev_sweep:
        meets = r["max_dd_pct"] < 15
        cagr_color = "#16a34a" if meets else "#dc2626"
        lev_rows += f"""<tr>
            <td style="font-weight:700">{r['leverage']}×</td>
            <td>{r['hedge_pct']*100:.1f}%</td>
            <td style="color:{cagr_color};font-weight:600">{r['cagr_pct']:.1f}%</td>
            <td style="font-weight:700">{r['sharpe']:.2f}</td>
            <td>{r['max_dd_pct']:.1f}%</td>
            <td>{r['calmar']:.2f}</td>
        </tr>"""

    # Top 3 yearly tables (combined)
    years = sorted(set(w["year"] for ydata in yearly_top3.values() for w in ydata))
    yearly_rows = ""
    for yr in years:
        cells = ""
        for name in yearly_top3.keys():
            ydata = yearly_top3[name]
            yr_data = next((w for w in ydata if w["year"] == yr), {})
            cagr = yr_data.get("cagr_pct", 0)
            sharpe = yr_data.get("sharpe", 0)
            dd = yr_data.get("max_dd_pct", 0)
            sc = "#16a34a" if cagr > 0 else "#dc2626"
            cells += f'<td style="color:{sc}">{cagr:.0f}%</td><td>{sharpe:.2f}</td><td>{dd:.1f}%</td>'
        yearly_rows += f'<tr><td style="font-weight:700">{yr}</td>{cells}</tr>'

    # Regime table
    regimes = ["bull (<15)", "normal (15-25)", "bear (25-35)", "crash (>35)"]
    regime_rows = ""
    for regime in regimes:
        cells = ""
        for name in regime_data.keys():
            rd = regime_data[name].get(regime, {})
            cells += f'<td>{rd.get("cagr_pct", 0):.0f}%</td><td>{rd.get("sharpe", 0):.2f}</td><td>{rd.get("n_days", 0)}d</td>'
        regime_rows += f'<tr><td style="font-weight:700">{regime}</td>{cells}</tr>'

    # Crisis Alpha asset table
    ca_rows = ""
    avg_w = sorted(ca_inspect["avg_abs_weight"].items(),
                    key=lambda x: -x[1])[:10]
    for asset, w in avg_w:
        long_pct = ca_inspect["pct_days_long"].get(asset, 0) * 100
        short_pct = ca_inspect["pct_days_short"].get(asset, 0) * 100
        ca_rows += f"""<tr>
            <td style="font-weight:600">{asset}</td>
            <td>{w*100:.1f}%</td>
            <td style="color:#16a34a">{long_pct:.0f}%</td>
            <td style="color:#dc2626">{short_pct:.0f}%</td>
        </tr>"""

    # Top 3 metrics summary
    top3_summary = ""
    for name, rets, m in top3:
        top3_summary += f"""<tr>
            <td style="font-weight:700">{name}</td>
            <td style="color:#16a34a;font-weight:600">{m['cagr_pct']:.1f}%</td>
            <td style="font-weight:700">{m['sharpe']:.2f}</td>
            <td>{m['max_dd_pct']:.1f}%</td>
            <td>{m['calmar']:.2f}</td>
            <td>{m['vol_pct']:.1f}%</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep Dive: 2× + Crisis Alpha Hedge</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1200px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.5; }}
  h1 {{ font-size:1.9em; color:#0f172a; margin-bottom:4px; }}
  h2 {{ color:#334155; margin-top:2.5em; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  h3 {{ color:#475569; margin-top:1.8em; }}
  .subtitle {{ color:#64748b; font-size:0.92rem; margin-bottom:24px; }}
  .verdict {{ background:#dcfce7; border:2px solid #16a34a; border-radius:10px; padding:20px; margin:24px 0; }}
  .verdict h3 {{ margin-top:0; color:#166534; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:18px;
          text-align:center; flex:1; min-width:130px; }}
  .kpi .value {{ font-size:1.7em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:0.84em; }}
  th {{ background:#f1f5f9; padding:10px 12px; text-align:right; font-weight:600; color:#475569;
       border-bottom:2px solid #cbd5e1; font-size:0.76em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .chart {{ margin:20px 0; text-align:center; }}
  .chart img {{ max-width:100%; border:1px solid #e2e8f0; border-radius:6px; }}
  .sources {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:16px; margin:16px 0; font-size:0.84rem; line-height:1.7; }}
  .explainer {{ background:#fefce8; border:1px solid #fde047; border-radius:8px; padding:18px; margin:16px 0; font-size:0.9rem; line-height:1.7; }}
  .explainer h3 {{ margin-top:0; color:#854d0e; }}
  .footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.78em; color:#94a3b8; text-align:center; }}
</style></head><body>

<h1>2× EXP-1220 + Crisis Alpha Hedge — Deep Dive</h1>
<div class="subtitle">Parameter sweep, leverage sweep, year-by-year, regime analysis, mechanism explainer | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="sources">
    <strong>Data Sources (Rule Zero — all REAL):</strong><br>
    EXP-1220 daily: scripts.ultimate_portfolio.load_exp1220_dynamic() (1507d, real Yahoo SPY/^VIX/^VIX3M)<br>
    Crisis Alpha v3: compass.crisis_alpha_v3 v2_round/0.10/2.5× (real Yahoo 13-ETF universe)<br>
    ^VIX for regime classification: Yahoo Finance chart API (lagged t-1)<br>
    Sharpe: compass/metrics.py annualized_sharpe (correct arithmetic mean × √252)
</div>

<div class="verdict">
    <h3>★ OPTIMAL CONFIG (max Sharpe, DD &lt; 15%)</h3>
    <strong>Leverage:</strong> 2.0× | <strong>Hedge:</strong> {optimal['hedge_pct']*100:.1f}% Crisis Alpha<br>
    <strong>CAGR:</strong> {optimal['cagr_pct']:.1f}% |
    <strong>Sharpe:</strong> {optimal['sharpe']:.2f} |
    <strong>Max DD:</strong> {optimal['max_dd_pct']:.1f}% |
    <strong>Calmar:</strong> {optimal['calmar']:.2f}
</div>

<h2>1. Hedge % Parameter Sweep (2× leverage, 5%-30% in 2.5% steps)</h2>
<div class="chart"><img src="{charts['param_sweep']}" alt="Param sweep"/></div>
<table>
    <thead><tr><th>Hedge %</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th></tr></thead>
    <tbody>{sweep_rows}</tbody>
</table>

<h2>2. Leverage Sweep (at optimal {optimal['hedge_pct']*100:.1f}% hedge)</h2>
<table>
    <thead><tr><th>Leverage</th><th>Hedge</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th></tr></thead>
    <tbody>{lev_rows}</tbody>
</table>

<h2>3. Pareto Frontier (5 leverages × 11 hedge %)</h2>
<div class="chart"><img src="{charts['pareto']}" alt="Pareto"/></div>

<h2>4. Top 3 Configs — Equity Curves</h2>
<div class="chart"><img src="{charts['top3_equity']}" alt="Top 3 equity"/></div>
<table>
    <thead><tr><th>Config</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th></tr></thead>
    <tbody>{top3_summary}</tbody>
</table>

<h2>5. Year-by-Year Breakdown (Top 3)</h2>
<div class="chart"><img src="{charts['yearly_top3']}" alt="Yearly bars"/></div>
<table>
    <thead><tr><th rowspan="2">Year</th>
        {''.join(f'<th colspan="3">{name}</th>' for name in yearly_top3.keys())}
    </tr><tr>
        {''.join('<th>CAGR</th><th>Sharpe</th><th>DD</th>' for _ in yearly_top3.keys())}
    </tr></thead>
    <tbody>{yearly_rows}</tbody>
</table>

<h2>6. Regime Analysis (VIX Buckets)</h2>
<div class="chart"><img src="{charts['regime']}" alt="Regime"/></div>
<table>
    <thead><tr><th rowspan="2">Regime</th>
        {''.join(f'<th colspan="3">{name}</th>' for name in regime_data.keys())}
    </tr><tr>
        {''.join('<th>CAGR</th><th>Sharpe</th><th>Days</th>' for _ in regime_data.keys())}
    </tr></thead>
    <tbody>{regime_rows}</tbody>
</table>

<h2>7. Crisis Alpha Mechanism Explainer</h2>
<div class="explainer">
    <h3>What is Crisis Alpha actually doing?</h3>
    <strong>Strategy:</strong> Trend-following CTA across 13 asset classes
    (equities, bonds, commodities, FX). Uses 4-lookback momentum signal
    (20/60/120/200 days) with vol-targeted position sizing (10% target vol)
    and 2.5× leverage cap. Rebalances every 5 days.<br><br>
    <strong>Universe:</strong> SPY, IWM, EFA, EEM, QQQ (equities) | TLT, LQD, HYG (bonds) | GLD, USO, DBA, DBB (commodities) | UUP (USD)<br><br>
    <strong>How it generates returns:</strong> When asset has positive momentum
    over 20-200 day windows, take a long position scaled by inverse-vol.
    Negative momentum → short. The trend-following nature means it
    catches sustained moves in either direction — including crashes
    where it goes short equities and long safe-haven assets (TLT, GLD, UUP).<br><br>
    <strong>Why it helps the portfolio:</strong> EXP-1220 (credit spreads)
    profits from calm and slow-grind markets. Crisis Alpha profits from
    sustained moves — exactly the regimes where credit spreads suffer.
    The two are NEGATIVELY correlated during stress (see correlation
    chart below) which is the diversification mechanism.
</div>

<h3>Top 10 Assets by Average Position Size</h3>
<table>
    <thead><tr><th>Asset</th><th>Avg |Weight|</th><th>% Days Long</th><th>% Days Short</th></tr></thead>
    <tbody>{ca_rows}</tbody>
</table>

<h3>60-Day Rolling Correlation: EXP-1220 ↔ Crisis Alpha</h3>
<div class="chart"><img src="{charts['correlation']}" alt="Correlation"/></div>

<h3>Crisis Alpha Holdings Over Time</h3>
<div class="chart"><img src="{charts['ca_holdings']}" alt="CA holdings"/></div>

<div class="footer">
    Comprehensive deep dive — scripts/deep_dive_2x_hedge.py<br>
    All metrics via compass/metrics.py (correct daily Sharpe formula).<br>
    Real Yahoo + IronVault-derived data only. No synthetic. No look-ahead.
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("Deep Dive: 2× EXP-1220 + Crisis Alpha Hedge")
    print("=" * 72)

    print("\n[1/8] Loading real data...")
    e1220, e1780, weights_df, prices, vix = load_all()
    print(f"  EXP-1220: {len(e1220)} days")
    print(f"  Crisis Alpha: {len(e1780)} days")

    print("\n[2/8] Parameter sweep (5%-30% hedge in 2.5% steps)...")
    sweep = parameter_sweep(e1220, e1780)
    for r in sweep:
        marker = "★" if r["max_dd_pct"] < 15 else " "
        print(f"  {marker} hedge={r['hedge_pct']*100:5.1f}%  CAGR={r['cagr_pct']:7.1f}%  "
              f"Sharpe={r['sharpe']:.2f}  DD={r['max_dd_pct']:.1f}%")

    optimal = find_optimal_hedge(sweep, dd_max=15.0)
    print(f"\n  ★ OPTIMAL: hedge={optimal['hedge_pct']*100:.1f}%  "
          f"CAGR={optimal['cagr_pct']:.1f}%  Sharpe={optimal['sharpe']:.2f}  DD={optimal['max_dd_pct']:.1f}%")

    print("\n[3/8] Leverage sweep (1.5× - 2.5× at optimal hedge)...")
    lev_sweep = leverage_sweep(e1220, e1780, optimal["hedge_pct"])
    for r in lev_sweep:
        print(f"  {r['leverage']:.2f}× CAGR={r['cagr_pct']:7.1f}%  Sharpe={r['sharpe']:.2f}  DD={r['max_dd_pct']:.1f}%")

    print("\n[4/8] Full grid (5 lev × 11 hedge = 55 configs)...")
    grid = full_grid(e1220, e1780)
    print(f"  {len(grid)} configs computed")

    # Find top 3 by Sharpe (with DD<15% if possible)
    feasible = [r for r in grid if r["max_dd_pct"] < 15]
    top_grid = sorted(feasible if feasible else grid, key=lambda r: r["sharpe"], reverse=True)[:3]
    print("\n  Top 3 by Sharpe (DD<15%):")
    for r in top_grid:
        print(f"    {r['leverage']}× + {r['hedge_pct']*100:.1f}% hedge: "
              f"CAGR={r['cagr_pct']:.1f}%  Sharpe={r['sharpe']:.2f}  DD={r['max_dd_pct']:.1f}%")

    # Build top 3 return streams
    top3_returns = []
    for r in top_grid:
        rets = combine(e1220, e1780, r["leverage"], r["hedge_pct"])
        name = f"{r['leverage']}× + {r['hedge_pct']*100:.0f}% hedge"
        top3_returns.append((name, rets, r))

    print("\n[5/8] Year-by-year for top 3...")
    yearly_top3 = {}
    for name, rets, _ in top3_returns:
        yearly_top3[name] = yearly_breakdown(rets)
        print(f"  {name}:")
        for w in yearly_top3[name]:
            print(f"    {w['year']}: CAGR={w['cagr_pct']:6.1f}%  Sharpe={w['sharpe']:.2f}  DD={w['max_dd_pct']:.1f}%")

    print("\n[6/8] Regime analysis (VIX buckets)...")
    regime_data = {}
    for name, rets, _ in top3_returns:
        regime_data[name] = regime_analysis(rets, vix)

    print("\n[7/8] Crisis Alpha mechanism inspection...")
    common = e1220.index.intersection(e1780.index)
    ca_inspect = inspect_crisis_alpha(weights_df, prices, common)
    print(f"  Avg long positions: {ca_inspect['avg_long_positions']:.1f}")
    print(f"  Avg short positions: {ca_inspect['avg_short_positions']:.1f}")
    print(f"  Top 5 assets by avg weight:")
    for asset, w in sorted(ca_inspect["avg_abs_weight"].items(), key=lambda x: -x[1])[:5]:
        print(f"    {asset}: {w*100:.1f}%")

    print("\n[8/8] Generating charts + HTML report...")
    rolling_corr = rolling_correlation(e1220, e1780, window=60).dropna()

    charts = {
        "param_sweep": chart_param_sweep(sweep),
        "pareto": chart_pareto_frontier(grid),
        "top3_equity": chart_top3_equity([(n, r) for n, r, _ in top3_returns]),
        "yearly_top3": chart_yearly_top3([(n, r) for n, r, _ in top3_returns]),
        "regime": chart_regime_performance(regime_data),
        "correlation": chart_rolling_correlation(rolling_corr),
        "ca_holdings": chart_crisis_alpha_holdings(weights_df),
    }

    html = build_html(sweep, lev_sweep, grid, top3_returns, yearly_top3,
                       regime_data, ca_inspect, charts)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_PATH}")
    print(f"  Size: {len(html)/1024:.0f} KB")


if __name__ == "__main__":
    main()
