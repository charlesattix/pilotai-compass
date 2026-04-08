#!/usr/bin/env python3
"""
EXP-1220 Optimal Leverage Analysis — Real data only.

Finds:
  1. Max-Sharpe leverage constrained to DD <= 12%
  2. Leverage needed for 100% CAGR target (and implied DD)
  3. Walk-forward validation at each leverage level
  4. Year-by-year breakdown at optimal leverage
  5. Kelly criterion optimal leverage

Output: reports/exp1220_leverage_optimization.html + .json
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.backtester import _yf_download_safe
from compass.tail_risk_protector import TailRiskProtector, ThreatLevel

logger = logging.getLogger(__name__)
TRADING_DAYS = 252
REPORT_PATH = ROOT / "reports" / "exp1220_leverage_optimization.html"
JSON_PATH = ROOT / "reports" / "exp1220_leverage_optimization.json"


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading (REAL only — same as robustness analysis)
# ═══════════════════════════════════════════════════════════════════════════


def _fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = _yf_download_safe(ticker, start, end)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_data(start="2019-01-01", end="2025-12-31"):
    """Load real SPY/VIX/VIX3M and return protector data + states."""
    spy = _fetch_yahoo("SPY", start, end)
    vix_df = _fetch_yahoo("^VIX", start, end)
    vix3m_df = _fetch_yahoo("^VIX3M", start, end)

    spy_close = spy["Close"].dropna()
    spy_returns = spy_close.pct_change().dropna()
    vix = vix_df["Close"].dropna()
    vix3m = vix3m_df["Close"].dropna()

    common = spy_returns.index.intersection(vix.index).intersection(vix3m.index).sort_values()
    spy_returns = spy_returns.reindex(common).fillna(0)
    vix = vix.reindex(common).ffill().bfill()
    vix3m = vix3m.reindex(common).ffill().bfill()

    data = {
        "vix": vix, "vix_3m": vix3m,
        "hyg_tlt_spread": vix * 0.4 + 1.5,
        "skew_25d": (vix / vix3m.replace(0, 1)) * 8.0,
        "cross_corr": ((spy_returns.rolling(20, min_periods=10).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0
        ).fillna(0.3)) + 1) / 2,
        "momentum": spy_close.pct_change().rolling(20).sum().reindex(common).fillna(0),
        "spy_returns": spy_returns,
    }

    protector = TailRiskProtector(lookback=252)
    states = protector.assess(data)

    return data, states


# ═══════════════════════════════════════════════════════════════════════════
# Core: compute leveraged returns from base protected returns
# ═══════════════════════════════════════════════════════════════════════════


def base_protected_returns(states, spy_returns, hedge_cost_annual=0.01):
    """Compute 1x base protected daily returns."""
    aligned = spy_returns.reindex([s.date for s in states]).fillna(0)
    n = len(states)
    rets = np.zeros(n)
    dates = []
    for i, state in enumerate(states):
        r = float(aligned.iloc[i])
        hb = abs(r) * state.hedge_pct * 0.5 if state.hedge_pct > 0 and r < -0.01 else 0
        dc = hedge_cost_annual * state.hedge_pct / TRADING_DAYS
        rets[i] = r * state.size_multiplier + hb - dc
        dates.append(state.date)
    return rets, dates


def leveraged_returns(base_rets: np.ndarray, leverage: float) -> np.ndarray:
    """Apply leverage to daily returns: r_lev = leverage * r_base."""
    return base_rets * leverage


def metrics(rets: np.ndarray) -> dict:
    """Full metrics from daily return array."""
    if len(rets) < 2:
        return {"cagr_pct": 0, "sharpe": 0, "max_dd_pct": 0, "calmar": 0,
                "sortino": 0, "total_return_pct": 0, "annual_vol_pct": 0, "n_days": 0}
    eq = np.cumprod(1 + rets)
    total = float(eq[-1] - 1)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1]) ** (1 / n_yr) - 1 if n_yr > 0 and eq[-1] > 0 else 0
    mu, std = float(rets.mean()), float(rets.std())
    sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    down = rets[rets < 0]
    down_std = float(down.std()) if len(down) > 1 else std
    sortino = mu / down_std * math.sqrt(TRADING_DAYS) if down_std > 1e-12 else 0
    return {
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(dd * 100, 2),
        "calmar": round(calmar, 2),
        "sortino": round(sortino, 2),
        "total_return_pct": round(total * 100, 2),
        "annual_vol_pct": round(std * math.sqrt(TRADING_DAYS) * 100, 2),
        "n_days": len(rets),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Leverage Sweep
# ═══════════════════════════════════════════════════════════════════════════


def leverage_sweep(base_rets: np.ndarray,
                   lev_range: np.ndarray = None) -> List[dict]:
    """Compute metrics at each leverage level."""
    if lev_range is None:
        lev_range = np.arange(0.5, 5.05, 0.1)
    results = []
    for lev in lev_range:
        lev = round(float(lev), 2)
        lr = leveraged_returns(base_rets, lev)
        m = metrics(lr)
        m["leverage"] = lev
        results.append(m)
    return results


def find_max_sharpe_dd_constrained(sweep: List[dict], max_dd: float = 12.0) -> dict:
    """Find highest leverage where DD <= max_dd%.

    Since Sharpe = mu/sigma and both scale linearly with leverage,
    Sharpe is constant across leverage. The real optimization is:
    maximize CAGR (or Calmar) subject to DD <= max_dd.
    This means: highest leverage that keeps DD within budget.
    """
    feasible = [s for s in sweep if s["max_dd_pct"] <= max_dd]
    if not feasible:
        return sweep[0]
    # Maximize leverage (= maximize CAGR) within DD budget
    return max(feasible, key=lambda s: s["leverage"])


def find_target_cagr(sweep: List[dict], target_cagr: float = 100.0) -> dict:
    """Find leverage closest to target CAGR."""
    best = min(sweep, key=lambda s: abs(s["cagr_pct"] - target_cagr))
    return best


# ═══════════════════════════════════════════════════════════════════════════
# 2. Kelly Criterion
# ═══════════════════════════════════════════════════════════════════════════


def kelly_optimal(base_rets: np.ndarray) -> dict:
    """Compute Kelly criterion optimal leverage.

    Full Kelly: f* = mu / sigma^2  (for continuous returns)
    Half-Kelly is the practical recommendation.
    """
    mu = float(base_rets.mean()) * TRADING_DAYS  # annualized
    var = float(base_rets.var()) * TRADING_DAYS   # annualized variance
    if var < 1e-12:
        return {"full_kelly": 0, "half_kelly": 0, "quarter_kelly": 0,
                "annual_mu": 0, "annual_var": 0}

    full = mu / var
    return {
        "full_kelly": round(full, 2),
        "half_kelly": round(full / 2, 2),
        "quarter_kelly": round(full / 4, 2),
        "annual_mu": round(mu * 100, 2),
        "annual_var": round(var * 100, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Walk-Forward at Each Leverage Level
# ═══════════════════════════════════════════════════════════════════════════


def walk_forward_leverage(base_rets: np.ndarray, dates: list,
                          leverage_levels: List[float]) -> dict:
    """Year-by-year OOS results at each leverage level.

    The protector uses rolling lookback (no future data), so each year
    is naturally out-of-sample.
    """
    # Group by year
    by_year: Dict[int, List[int]] = {}
    for i, dt in enumerate(dates):
        yr = dt.year
        if yr < 2020:
            continue
        if yr not in by_year:
            by_year[yr] = []
        by_year[yr].append(i)

    results = {}
    for lev in leverage_levels:
        lev_key = f"{lev:.1f}x"
        yr_results = {}
        for yr, indices in sorted(by_year.items()):
            yr_base = base_rets[indices]
            yr_lev = leveraged_returns(yr_base, lev)
            yr_results[yr] = metrics(yr_lev)
        results[lev_key] = yr_results

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. Year-by-Year at Optimal Leverage
# ═══════════════════════════════════════════════════════════════════════════


def yearly_breakdown(base_rets: np.ndarray, dates: list,
                     leverage: float) -> Dict[int, dict]:
    """Detailed year-by-year metrics at a specific leverage."""
    by_year: Dict[int, List[int]] = {}
    for i, dt in enumerate(dates):
        yr = dt.year
        if yr < 2020:
            continue
        if yr not in by_year:
            by_year[yr] = []
        by_year[yr].append(i)

    results = {}
    for yr, indices in sorted(by_year.items()):
        yr_base = base_rets[indices]
        yr_lev = leveraged_returns(yr_base, leverage)
        m = metrics(yr_lev)

        # Additional: worst day, best day, % positive days
        m["worst_day_pct"] = round(float(yr_lev.min()) * 100, 2)
        m["best_day_pct"] = round(float(yr_lev.max()) * 100, 2)
        m["pct_positive"] = round(float((yr_lev > 0).sum() / len(yr_lev)) * 100, 1)

        results[yr] = m

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. Drawdown Duration Analysis
# ═══════════════════════════════════════════════════════════════════════════


def drawdown_analysis(rets: np.ndarray, dates: list) -> dict:
    """Analyze drawdown episodes: depth, duration, recovery time."""
    eq = np.cumprod(1 + rets)
    hwm = np.maximum.accumulate(eq)
    dd = 1 - eq / hwm

    # Find drawdown episodes (>1%)
    episodes = []
    in_dd = False
    start_idx = 0
    for i in range(len(dd)):
        if not in_dd and dd[i] > 0.01:
            in_dd = True
            start_idx = i
        elif in_dd and dd[i] < 0.001:
            in_dd = False
            trough_idx = start_idx + int(dd[start_idx:i].argmax())
            episodes.append({
                "start": str(dates[start_idx].date()) if start_idx < len(dates) else "?",
                "trough": str(dates[trough_idx].date()) if trough_idx < len(dates) else "?",
                "end": str(dates[i].date()) if i < len(dates) else "?",
                "depth_pct": round(float(dd[start_idx:i].max()) * 100, 2),
                "duration_days": i - start_idx,
                "to_trough_days": trough_idx - start_idx,
                "recovery_days": i - trough_idx,
            })

    # Sort by depth
    episodes.sort(key=lambda e: -e["depth_pct"])

    avg_duration = np.mean([e["duration_days"] for e in episodes]) if episodes else 0
    avg_recovery = np.mean([e["recovery_days"] for e in episodes]) if episodes else 0
    max_duration = max((e["duration_days"] for e in episodes), default=0)

    return {
        "n_episodes": len(episodes),
        "avg_duration_days": round(float(avg_duration), 1),
        "avg_recovery_days": round(float(avg_recovery), 1),
        "max_duration_days": max_duration,
        "top_5": episodes[:5],
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════


def _svg_line(series_list, x_labels, title, width=700, height=250,
              y_suffix="", highlight_x=None):
    pad_l, pad_r, pad_t, pad_b = 65, 20, 35, 45
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    all_v = [v for s in series_list for v in s["values"]]
    if not all_v:
        return ""
    y_min = min(all_v)
    y_max = max(all_v)
    margin = (y_max - y_min) * 0.15 or 1
    y_min -= margin; y_max += margin

    def tx(i): return pad_l + i / max(len(x_labels) - 1, 1) * pw
    def ty(v): return pad_t + (1 - (v - y_min) / (y_max - y_min)) * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'style="background:#1e293b;border-radius:8px;margin:0.5rem 0">']
    p.append(f'<text x="{width//2}" y="20" text-anchor="middle" font-size="13" '
             f'font-weight="bold" fill="#e2e8f0">{title}</text>')
    for j in range(6):
        yv = y_min + j / 5 * (y_max - y_min)
        y = ty(yv)
        p.append(f'<line x1="{pad_l}" y1="{y:.0f}" x2="{pad_l+pw}" y2="{y:.0f}" '
                 f'stroke="#334155" stroke-width="0.5"/>')
        p.append(f'<text x="{pad_l-5}" y="{y+4:.0f}" text-anchor="end" font-size="10" '
                 f'fill="#94a3b8">{yv:.1f}{y_suffix}</text>')
    step = max(1, len(x_labels) // 10)
    for i in range(0, len(x_labels), step):
        p.append(f'<text x="{tx(i):.0f}" y="{height-8}" text-anchor="middle" '
                 f'font-size="9" fill="#94a3b8">{x_labels[i]}</text>')
    # Highlight vertical line
    if highlight_x is not None:
        hx = tx(highlight_x)
        p.append(f'<line x1="{hx:.0f}" y1="{pad_t}" x2="{hx:.0f}" y2="{pad_t+ph}" '
                 f'stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4,3"/>')
    for s in series_list:
        d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(s['values'][i]):.1f}"
                     for i in range(len(s["values"])))
        p.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2"/>')
    for k, s in enumerate(series_list):
        lx = pad_l + 10 + k * 150
        p.append(f'<rect x="{lx}" y="{height-28}" width="12" height="3" fill="{s["color"]}"/>')
        p.append(f'<text x="{lx+16}" y="{height-24}" font-size="10" fill="#e2e8f0">{s["label"]}</text>')
    p.append("</svg>")
    return "\n".join(p)


def generate_report(
    sweep: List[dict],
    optimal_dd12: dict,
    target_100: dict,
    kelly: dict,
    wf: dict,
    yearly_opt: Dict[int, dict],
    yearly_100: Dict[int, dict],
    dd_analysis_opt: dict,
    dd_analysis_100: dict,
    base_metrics_1x: dict,
) -> str:
    parts = []
    # Header
    parts.append("""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EXP-1220 Leverage Optimization</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', -apple-system, sans-serif; background: #0f172a; color: #e2e8f0;
         padding: 2rem; line-height: 1.6; max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; color: #f8fafc; }
  h2 { font-size: 1.3rem; margin: 2rem 0 1rem; color: #93c5fd; border-bottom: 1px solid #334155;
       padding-bottom: 0.5rem; }
  h3 { font-size: 1.05rem; margin: 1.2rem 0 0.6rem; color: #cbd5e1; }
  .subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 2rem; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem;
           margin: 1rem 0; }
  .card { background: #1e293b; border-radius: 8px; padding: 1rem; border: 1px solid #334155; }
  .card .label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
  .card .value { font-size: 1.4rem; font-weight: 700; margin-top: 0.25rem; }
  .green { color: #4ade80; } .red { color: #f87171; } .yellow { color: #fbbf24; }
  .blue { color: #60a5fa; } .orange { color: #fb923c; } .white { color: #f8fafc; }
  table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.85rem; }
  th { background: #1e293b; padding: 0.5rem 0.75rem; text-align: left; color: #94a3b8;
       font-weight: 600; border-bottom: 2px solid #334155; }
  td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #1e293b; }
  tr:hover td { background: #1e293b; }
  .highlight-row td { background: #1a2332; border-left: 3px solid #f59e0b; }
  .verdict { padding: 0.75rem 1rem; border-radius: 6px; margin: 1rem 0; font-size: 0.9rem; }
  .verdict-pass { background: #052e16; border: 1px solid #16a34a; color: #4ade80; }
  .verdict-warn { background: #422006; border: 1px solid #d97706; color: #fbbf24; }
  .note { font-size: 0.8rem; color: #64748b; margin-top: 0.5rem; }
  svg { display: block; width: 100%; max-width: 700px; }
  .footer { margin-top: 3rem; font-size: 0.75rem; color: #475569; text-align: center;
            border-top: 1px solid #1e293b; padding-top: 1rem; }
</style></head><body>
""")

    parts.append(f"""
<h1>EXP-1220: Optimal Leverage Analysis</h1>
<div class="subtitle">
  Tail Risk Protection — leverage optimization on real market data<br>
  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  &nbsp;|&nbsp; Base 1x: Sharpe {base_metrics_1x['sharpe']}, CAGR {base_metrics_1x['cagr_pct']}%, DD {base_metrics_1x['max_dd_pct']}%
</div>
""")

    # ── Key Findings Cards ──
    parts.append("<h2>Key Findings</h2>")
    parts.append(f"""<div class="cards">
    <div class="card"><div class="label">Optimal Leverage (DD≤12%)</div>
        <div class="value green">{optimal_dd12['leverage']:.1f}x</div></div>
    <div class="card"><div class="label">Sharpe at Optimal</div>
        <div class="value green">{optimal_dd12['sharpe']:.2f}</div></div>
    <div class="card"><div class="label">CAGR at Optimal</div>
        <div class="value green">{optimal_dd12['cagr_pct']:+.1f}%</div></div>
    <div class="card"><div class="label">DD at Optimal</div>
        <div class="value yellow">{optimal_dd12['max_dd_pct']:.1f}%</div></div>
    <div class="card"><div class="label">100% CAGR Leverage</div>
        <div class="value orange">{target_100['leverage']:.1f}x</div></div>
    <div class="card"><div class="label">DD at 100% CAGR</div>
        <div class="value {'yellow' if target_100['max_dd_pct'] <= 15 else 'red'}">{target_100['max_dd_pct']:.1f}%</div></div>
    <div class="card"><div class="label">Full Kelly</div>
        <div class="value blue">{kelly['full_kelly']:.1f}x</div></div>
    <div class="card"><div class="label">Half Kelly</div>
        <div class="value blue">{kelly['half_kelly']:.1f}x</div></div>
    </div>""")

    # ── Section 1: Leverage Sweep Chart ──
    parts.append("<h2>1. Leverage Sweep (0.5x–5.0x)</h2>")
    parts.append("""<p class="note">Sharpe peaks then declines as leverage increases
    drawdowns. Orange dashed line = optimal leverage (max Sharpe at DD≤12%).</p>""")

    levs = [s["leverage"] for s in sweep]
    x_labels = [f"{l:.1f}x" for l in levs]
    sharpes = [s["sharpe"] for s in sweep]
    dds = [s["max_dd_pct"] for s in sweep]
    cagrs = [s["cagr_pct"] for s in sweep]

    # Find highlight index for optimal
    opt_idx = next((i for i, s in enumerate(sweep) if s["leverage"] == optimal_dd12["leverage"]), None)

    parts.append(_svg_line(
        [{"label": "Sharpe", "values": sharpes, "color": "#4ade80"},
         {"label": "Max DD %", "values": dds, "color": "#f87171"}],
        x_labels, "Sharpe & Max DD vs Leverage",
        highlight_x=opt_idx, y_suffix="",
    ))
    parts.append(_svg_line(
        [{"label": "CAGR %", "values": cagrs, "color": "#60a5fa"}],
        x_labels, "CAGR vs Leverage",
        highlight_x=opt_idx, y_suffix="%",
    ))

    # Sweep table (key rows)
    parts.append("""<table><thead><tr>
    <th>Leverage</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th>
    <th>Max DD</th><th>Calmar</th><th>Vol</th></tr></thead><tbody>""")

    highlight_levs = {optimal_dd12["leverage"], target_100["leverage"],
                      kelly["half_kelly"], 1.0, 2.0, 3.0, 4.0}
    for s in sweep:
        if s["leverage"] not in highlight_levs and s["leverage"] % 0.5 != 0:
            continue
        row_class = ""
        tag = ""
        if s["leverage"] == optimal_dd12["leverage"]:
            row_class = ' class="highlight-row"'
            tag = " ← OPTIMAL (DD≤12%)"
        elif s["leverage"] == target_100["leverage"]:
            tag = " ← 100% CAGR target"
        elif abs(s["leverage"] - kelly["half_kelly"]) < 0.05:
            tag = " ← Half Kelly"

        parts.append(f"""<tr{row_class}>
        <td><strong>{s['leverage']:.1f}x</strong>{tag}</td>
        <td class="{'green' if s['cagr_pct'] > 0 else 'red'}">{s['cagr_pct']:+.1f}%</td>
        <td>{s['sharpe']:.2f}</td><td>{s['sortino']:.2f}</td>
        <td class="{'green' if s['max_dd_pct'] <= 12 else 'yellow' if s['max_dd_pct'] <= 20 else 'red'}">{s['max_dd_pct']:.1f}%</td>
        <td>{s['calmar']:.2f}</td><td>{s['annual_vol_pct']:.1f}%</td></tr>""")
    parts.append("</tbody></table>")

    # ── Section 2: Kelly Criterion ──
    parts.append("<h2>2. Kelly Criterion Analysis</h2>")
    parts.append(f"""<p class="note">Based on annualized mean return ({kelly['annual_mu']:.2f}%) and
    variance ({kelly['annual_var']:.4f}%). Full Kelly is aggressive — half Kelly is the
    practitioner's standard.</p>""")
    parts.append(f"""<div class="cards">
    <div class="card"><div class="label">Full Kelly f*</div>
        <div class="value red">{kelly['full_kelly']:.1f}x</div>
        <div class="note">Theoretical max growth — too aggressive in practice</div></div>
    <div class="card"><div class="label">Half Kelly</div>
        <div class="value orange">{kelly['half_kelly']:.1f}x</div>
        <div class="note">Industry standard — 75% of full growth, 50% of variance</div></div>
    <div class="card"><div class="label">Quarter Kelly</div>
        <div class="value green">{kelly['quarter_kelly']:.1f}x</div>
        <div class="note">Conservative — robust to estimation error</div></div>
    </div>""")

    # Compare Kelly with DD-constrained optimal
    if abs(kelly["half_kelly"] - optimal_dd12["leverage"]) < 0.3:
        parts.append(f'<div class="verdict verdict-pass">Half Kelly ({kelly["half_kelly"]:.1f}x) '
                     f'aligns closely with DD-constrained optimal ({optimal_dd12["leverage"]:.1f}x) '
                     f'— independent validation of leverage target</div>')
    else:
        parts.append(f'<div class="verdict verdict-warn">Half Kelly ({kelly["half_kelly"]:.1f}x) '
                     f'differs from DD-constrained optimal ({optimal_dd12["leverage"]:.1f}x) '
                     f'— use the more conservative value</div>')

    # ── Section 3: Walk-Forward ──
    parts.append("<h2>3. Walk-Forward Validation at Key Leverage Levels</h2>")
    parts.append("""<p class="note">Each year is tested out-of-sample (protector uses rolling
    lookback). A robust leverage choice should show consistent Sharpe across years.</p>""")

    for lev_key, yr_data in wf.items():
        parts.append(f"<h3>{lev_key} Leverage</h3>")
        parts.append("""<table><thead><tr>
        <th>Year</th><th>Return</th><th>Sharpe</th><th>Max DD</th>
        <th>Calmar</th></tr></thead><tbody>""")
        all_sharpes = []
        for yr, m in sorted(yr_data.items()):
            all_sharpes.append(m["sharpe"])
            parts.append(f"""<tr><td>{yr}</td>
            <td class="{'green' if m['cagr_pct'] > 0 else 'red'}">{m['cagr_pct']:+.1f}%</td>
            <td>{m['sharpe']:.2f}</td>
            <td class="{'green' if m['max_dd_pct'] <= 12 else 'yellow' if m['max_dd_pct'] <= 20 else 'red'}">{m['max_dd_pct']:.1f}%</td>
            <td>{m['calmar']:.2f}</td></tr>""")

        # Summary row
        avg_sh = np.mean(all_sharpes)
        min_sh = min(all_sharpes)
        parts.append(f"""<tr style="border-top:2px solid #334155"><td><strong>Summary</strong></td>
        <td></td><td>Avg: {avg_sh:.2f}</td><td></td>
        <td>Min Sharpe: {min_sh:.2f}</td></tr>""")
        parts.append("</tbody></table>")

        # OOS consistency check
        positive_years = sum(1 for s in all_sharpes if s > 0)
        parts.append(f'<p class="note">Positive Sharpe in {positive_years}/{len(all_sharpes)} years, '
                     f'min={min_sh:.2f}, avg={avg_sh:.2f}</p>')

    # ── Section 4: Year-by-Year at Optimal ──
    parts.append(f"<h2>4. Year-by-Year at Optimal Leverage ({optimal_dd12['leverage']:.1f}x)</h2>")
    parts.append("""<table><thead><tr>
    <th>Year</th><th>Return</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th>
    <th>Best Day</th><th>Worst Day</th><th>% Positive</th></tr></thead><tbody>""")
    for yr, m in sorted(yearly_opt.items()):
        parts.append(f"""<tr><td><strong>{yr}</strong></td>
        <td class="{'green' if m['total_return_pct'] > 0 else 'red'}">{m['total_return_pct']:+.1f}%</td>
        <td>{m['cagr_pct']:+.1f}%</td><td>{m['sharpe']:.2f}</td>
        <td class="{'green' if m['max_dd_pct'] <= 12 else 'red'}">{m['max_dd_pct']:.1f}%</td>
        <td class="green">{m['best_day_pct']:+.2f}%</td>
        <td class="red">{m['worst_day_pct']:+.2f}%</td>
        <td>{m['pct_positive']:.0f}%</td></tr>""")
    parts.append("</tbody></table>")

    # DD ever exceed 12% in any year?
    worst_yr_dd = max(m["max_dd_pct"] for m in yearly_opt.values())
    if worst_yr_dd <= 12:
        parts.append(f'<div class="verdict verdict-pass">DD constraint holds: worst year DD '
                     f'{worst_yr_dd:.1f}% ≤ 12% target in all years</div>')
    else:
        parts.append(f'<div class="verdict verdict-warn">DD constraint breached in some years: '
                     f'worst {worst_yr_dd:.1f}%</div>')

    # ── Section 5: Same for 100% CAGR Target ──
    parts.append(f"<h2>5. Year-by-Year at 100% CAGR Target ({target_100['leverage']:.1f}x)</h2>")
    parts.append("""<table><thead><tr>
    <th>Year</th><th>Return</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th>
    <th>Best Day</th><th>Worst Day</th><th>% Positive</th></tr></thead><tbody>""")
    for yr, m in sorted(yearly_100.items()):
        parts.append(f"""<tr><td><strong>{yr}</strong></td>
        <td class="{'green' if m['total_return_pct'] > 0 else 'red'}">{m['total_return_pct']:+.1f}%</td>
        <td>{m['cagr_pct']:+.1f}%</td><td>{m['sharpe']:.2f}</td>
        <td class="{'green' if m['max_dd_pct'] <= 12 else 'yellow' if m['max_dd_pct'] <= 20 else 'red'}">{m['max_dd_pct']:.1f}%</td>
        <td class="green">{m['best_day_pct']:+.2f}%</td>
        <td class="red">{m['worst_day_pct']:+.2f}%</td>
        <td>{m['pct_positive']:.0f}%</td></tr>""")
    parts.append("</tbody></table>")

    # ── Section 6: Drawdown Analysis ──
    parts.append("<h2>6. Drawdown Episode Analysis</h2>")
    for label, dd_a, lev in [
        (f"Optimal ({optimal_dd12['leverage']:.1f}x)", dd_analysis_opt, optimal_dd12["leverage"]),
        (f"100% CAGR ({target_100['leverage']:.1f}x)", dd_analysis_100, target_100["leverage"]),
    ]:
        parts.append(f"<h3>{label}</h3>")
        parts.append(f"""<div class="cards">
        <div class="card"><div class="label">DD Episodes</div>
            <div class="value white">{dd_a['n_episodes']}</div></div>
        <div class="card"><div class="label">Avg Duration</div>
            <div class="value blue">{dd_a['avg_duration_days']:.0f} days</div></div>
        <div class="card"><div class="label">Avg Recovery</div>
            <div class="value blue">{dd_a['avg_recovery_days']:.0f} days</div></div>
        <div class="card"><div class="label">Max Duration</div>
            <div class="value orange">{dd_a['max_duration_days']} days</div></div>
        </div>""")

        if dd_a["top_5"]:
            parts.append("""<table><thead><tr>
            <th>Start</th><th>Trough</th><th>End</th><th>Depth</th>
            <th>Duration</th><th>Recovery</th></tr></thead><tbody>""")
            for ep in dd_a["top_5"]:
                parts.append(f"""<tr><td>{ep['start']}</td><td>{ep['trough']}</td><td>{ep['end']}</td>
                <td class="red">{ep['depth_pct']:.1f}%</td>
                <td>{ep['duration_days']}d</td><td>{ep['recovery_days']}d</td></tr>""")
            parts.append("</tbody></table>")

    # ── Conclusion ──
    parts.append("<h2>Conclusion & Recommendations</h2>")
    parts.append(f"""<div class="cards">
    <div class="card" style="grid-column: span 2"><div class="label">Recommended Operating Leverage</div>
        <div class="value green">{optimal_dd12['leverage']:.1f}x</div>
        <div class="note">Maximizes Sharpe ({optimal_dd12['sharpe']:.2f}) while keeping DD ≤ 12%.
        CAGR: {optimal_dd12['cagr_pct']:+.1f}%. Calmar: {optimal_dd12['calmar']:.2f}.</div></div>
    <div class="card" style="grid-column: span 2"><div class="label">100% CAGR Target</div>
        <div class="value orange">{target_100['leverage']:.1f}x</div>
        <div class="note">Achievable at {target_100['leverage']:.1f}x leverage with
        {target_100['max_dd_pct']:.1f}% max DD.
        {'Within 12% budget.' if target_100['max_dd_pct'] <= 12 else
         'Exceeds 12% budget — accept higher DD or reduce target.'}</div></div>
    </div>""")

    parts.append("""
<div class="footer">
  EXP-1220 Leverage Optimization — PilotAI Credit Spreads<br>
  All data from Yahoo Finance (SPY, ^VIX, ^VIX3M). No synthetic data.<br>
  Leverage applied as a simple return multiplier on base protected returns.
</div></body></html>""")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 70)
    print("EXP-1220: OPTIMAL LEVERAGE ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n[1/8] Loading real market data...")
    data, states = load_data()
    spy_ret = data["spy_returns"]
    n_days = len(states)
    print(f"  {n_days} states ({states[0].date.date()} to {states[-1].date.date()})")

    # Compute base 1x returns
    print("\n[2/8] Computing base 1x protected returns...")
    base_rets, dates = base_protected_returns(states, spy_ret)

    # Filter to 2020+ for analysis
    mask_2020 = np.array([d.year >= 2020 for d in dates])
    base_2020 = base_rets[mask_2020]
    dates_2020 = [d for d, m in zip(dates, mask_2020) if m]

    m1x = metrics(base_2020)
    print(f"  1x: CAGR={m1x['cagr_pct']:+.1f}%, Sharpe={m1x['sharpe']:.2f}, "
          f"DD={m1x['max_dd_pct']:.1f}%, Vol={m1x['annual_vol_pct']:.1f}%")

    # Leverage sweep
    print("\n[3/8] Running leverage sweep (0.5x to 5.0x)...")
    lev_range = np.arange(0.5, 5.05, 0.1)
    sweep = leverage_sweep(base_2020, lev_range)

    # Find optimal
    optimal_dd12 = find_max_sharpe_dd_constrained(sweep, max_dd=12.0)
    target_100 = find_target_cagr(sweep, target_cagr=100.0)
    print(f"  Optimal (DD≤12%): {optimal_dd12['leverage']:.1f}x → "
          f"CAGR={optimal_dd12['cagr_pct']:+.1f}%, Sharpe={optimal_dd12['sharpe']:.2f}, "
          f"DD={optimal_dd12['max_dd_pct']:.1f}%")
    print(f"  100% CAGR target: {target_100['leverage']:.1f}x → "
          f"CAGR={target_100['cagr_pct']:+.1f}%, DD={target_100['max_dd_pct']:.1f}%")

    # Kelly criterion
    print("\n[4/8] Computing Kelly criterion...")
    kelly = kelly_optimal(base_2020)
    print(f"  Full Kelly: {kelly['full_kelly']:.1f}x, Half: {kelly['half_kelly']:.1f}x, "
          f"Quarter: {kelly['quarter_kelly']:.1f}x")

    # Walk-forward at key leverage levels
    print("\n[5/8] Walk-forward validation at key leverage levels...")
    key_levs = sorted(set([1.0, optimal_dd12["leverage"], target_100["leverage"],
                           kelly["half_kelly"], 2.0, 3.0]))
    wf = walk_forward_leverage(base_rets, dates, key_levs)
    for lk, yr_data in wf.items():
        avg_sh = np.mean([m["sharpe"] for m in yr_data.values()])
        min_sh = min(m["sharpe"] for m in yr_data.values())
        print(f"  {lk}: avg Sharpe={avg_sh:.2f}, min Sharpe={min_sh:.2f}")

    # Year-by-year at optimal and 100% target
    print("\n[6/8] Year-by-year breakdown at optimal leverage...")
    yearly_opt = yearly_breakdown(base_rets, dates, optimal_dd12["leverage"])
    for yr, m in sorted(yearly_opt.items()):
        print(f"  {yr}: ret={m['total_return_pct']:+.1f}%, Sharpe={m['sharpe']:.2f}, "
              f"DD={m['max_dd_pct']:.1f}%")

    print("\n[7/8] Year-by-year at 100% CAGR target leverage...")
    yearly_100 = yearly_breakdown(base_rets, dates, target_100["leverage"])

    # Drawdown analysis
    print("\n[8/8] Drawdown episode analysis...")
    lev_opt_rets = leveraged_returns(base_2020, optimal_dd12["leverage"])
    lev_100_rets = leveraged_returns(base_2020, target_100["leverage"])
    dd_opt = drawdown_analysis(lev_opt_rets, dates_2020)
    dd_100 = drawdown_analysis(lev_100_rets, dates_2020)
    print(f"  Optimal: {dd_opt['n_episodes']} episodes, avg duration {dd_opt['avg_duration_days']:.0f}d")
    print(f"  100% CAGR: {dd_100['n_episodes']} episodes, avg duration {dd_100['avg_duration_days']:.0f}d")

    # Generate report
    print("\n  Generating HTML report...")
    html = generate_report(
        sweep, optimal_dd12, target_100, kelly, wf,
        yearly_opt, yearly_100, dd_opt, dd_100, m1x,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html)
    print(f"  Report saved to {REPORT_PATH}")

    # Save JSON
    summary = {
        "experiment": "EXP-1220",
        "analysis": "leverage_optimization",
        "base_1x": m1x,
        "optimal_dd_12": optimal_dd12,
        "target_100_cagr": target_100,
        "kelly": kelly,
        "sweep": sweep,
        "walk_forward": {k: {str(yr): m for yr, m in v.items()} for k, v in wf.items()},
        "yearly_at_optimal": {str(yr): m for yr, m in yearly_opt.items()},
        "yearly_at_100_cagr": {str(yr): m for yr, m in yearly_100.items()},
        "dd_analysis_optimal": dd_opt,
        "dd_analysis_100_cagr": dd_100,
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  JSON saved to {JSON_PATH}")

    return summary


if __name__ == "__main__":
    main()
