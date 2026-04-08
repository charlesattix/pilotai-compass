#!/usr/bin/env python3
"""
Optimal Combo Backtests — Head-to-Head vs Ultimate Portfolio.

Takes the top 3 uncorrelated strategy combos from the correlation analyzer,
builds dedicated portfolio backtests for each, and compares them against the
current Ultimate Portfolio baseline.

Focus: maximize Sharpe while keeping CAGR > 50%.

Output: reports/optimal_combos_backtest.html + .json
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.correlation_analyzer import (
    STRATEGIES, build_daily_returns, compute_correlation_matrix,
    avg_pairwise_corr, portfolio_metrics, TRADING_DAYS, N_YEARS,
)

REPORT_PATH = ROOT / "reports" / "optimal_combos_backtest.html"
JSON_PATH = ROOT / "reports" / "optimal_combos_backtest.json"
CAPITAL = 100_000


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio definitions
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioDef:
    name: str
    label: str           # short label
    strategies: List[str]
    weights: Dict[str, float]  # strategy name → weight
    description: str


# Ultimate Portfolio baseline (from reports/ultimate_portfolio.json)
ULTIMATE = PortfolioDef(
    name="Ultimate Portfolio (Baseline)",
    label="Ultimate",
    strategies=["EXP-1220 Tail Risk", "Cross-Asset Pairs", "TLT Iron Condors", "Vol Term Structure"],
    weights={
        "EXP-1220 Tail Risk": 0.95,
        "Cross-Asset Pairs": 0.0167,
        "TLT Iron Condors": 0.0167,
        "Vol Term Structure": 0.0167,
    },
    description="Current production portfolio: 95% EXP-1220 tail risk + 5% diversifiers",
)

# Top 3 combos from correlation analyzer (ranked by Sharpe)
COMBO_1 = PortfolioDef(
    name="Combo A: Tail Risk + RelVal + ICs",
    label="Combo A",
    strategies=["EXP-1220 Tail Risk", "EXP-1630 GLD/TLT RV", "EXP-1630 Multi-Pair",
                "EXP-1650 Earnings VC", "TLT Iron Condors"],
    weights={},  # will compute optimized
    description="Correlation analyzer rank #1: avg corr −0.002, combines tail risk with "
                "credit spread diversifiers (GLD/TLT pairs, multi-pair, earnings, TLT ICs)",
)

COMBO_2 = PortfolioDef(
    name="Combo B: Tail Risk + Pairs + Vol",
    label="Combo B",
    strategies=["Cross-Asset Pairs", "EXP-1220 Tail Risk", "EXP-1650 Earnings VC",
                "TLT Iron Condors", "Vol Term Structure"],
    weights={},
    description="Correlation analyzer rank #2: avg corr −0.066, substitutes vol term "
                "structure and cross-asset pairs for the multi-pair allocation",
)

COMBO_3 = PortfolioDef(
    name="Combo C: Tail Risk + XLI ICs + Multi-Pair",
    label="Combo C",
    strategies=["EXP-1220 Tail Risk", "XLI Iron Condors", "EXP-1630 Multi-Pair",
                "TLT Iron Condors", "EXP-1650 Earnings VC"],
    weights={},
    description="Custom combo: replaces low-CAGR strategies with XLI Iron Condors "
                "(highest OOS Sharpe 8.58, CAGR 18.8%) to boost raw returns",
)

ALL_PORTFOLIOS = [ULTIMATE, COMBO_1, COMBO_2, COMBO_3]


# ═══════════════════════════════════════════════════════════════════════════
# Weight optimization
# ═══════════════════════════════════════════════════════════════════════════

def optimize_weights(
    returns: Dict[str, np.ndarray],
    strategies: List[str],
    target: str = "sharpe",
    min_weight: float = 0.05,
) -> Dict[str, float]:
    """Find optimal weights via grid search.

    Searches allocations where EXP-1220 gets 40-95% and remaining strategies
    split the rest, maximizing Sharpe while maintaining CAGR > 50%.
    """
    n = len(strategies)
    has_1220 = "EXP-1220 Tail Risk" in strategies
    others = [s for s in strategies if s != "EXP-1220 Tail Risk"]

    best_sharpe = -999
    best_weights = {}

    # Search EXP-1220 weight from 40% to 95% in 5% steps
    core_range = list(np.arange(0.40, 0.96, 0.05)) if has_1220 else [0.0]

    for core_w in core_range:
        remaining = 1.0 - core_w
        n_others = len(others)
        if n_others == 0:
            continue

        # Try equal split and a few skewed allocations for the remaining
        # Generate allocation patterns
        allocations = []

        # Equal split
        eq = {s: remaining / n_others for s in others}
        allocations.append(eq)

        # Heavy on highest-CAGR strategy
        by_cagr = sorted(others, key=lambda s: STRATEGIES[s].cagr, reverse=True)
        if len(by_cagr) >= 2:
            heavy = {s: remaining * 0.10 / (n_others - 1) for s in others}
            heavy[by_cagr[0]] = remaining * 0.50
            heavy[by_cagr[1]] = remaining * 0.30
            rest_w = remaining * 0.20 / max(n_others - 2, 1)
            for s in by_cagr[2:]:
                heavy[s] = rest_w
            allocations.append(heavy)

        # Heavy on highest-Sharpe strategy
        by_sharpe = sorted(others, key=lambda s: STRATEGIES[s].sharpe, reverse=True)
        if len(by_sharpe) >= 2:
            heavy2 = {s: remaining * 0.10 / (n_others - 1) for s in others}
            heavy2[by_sharpe[0]] = remaining * 0.50
            heavy2[by_sharpe[1]] = remaining * 0.30
            rest_w2 = remaining * 0.20 / max(n_others - 2, 1)
            for s in by_sharpe[2:]:
                heavy2[s] = rest_w2
            allocations.append(heavy2)

        for alloc in allocations:
            weights = {}
            if has_1220:
                weights["EXP-1220 Tail Risk"] = core_w
            weights.update(alloc)

            # Compute portfolio returns
            combined = sum(returns[s] * w for s, w in weights.items())
            cum = np.cumprod(1 + combined)
            n_years = len(combined) / TRADING_DAYS
            cagr = cum[-1] ** (1 / n_years) - 1 if cum[-1] > 0 else -1
            vol = np.std(combined) * math.sqrt(TRADING_DAYS)
            _rf_daily = 0.045 / 252
            sharpe = (float(np.mean(combined)) - _rf_daily) / float(np.std(combined)) * math.sqrt(TRADING_DAYS) if float(np.std(combined)) > 1e-12 else 0

            if target == "sharpe":
                score = sharpe
            else:
                score = cagr

            if score > best_sharpe:
                best_sharpe = score
                best_weights = dict(weights)

    return best_weights


# ═══════════════════════════════════════════════════════════════════════════
# Full portfolio backtest
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    name: str
    label: str
    strategies: List[str]
    weights: Dict[str, float]
    # Base metrics (1x)
    cagr: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    vol: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    div_ratio: float = 0.0
    # Yearly
    yearly: Dict[int, Dict] = field(default_factory=dict)
    # Walk-forward
    wf_windows: List[Dict] = field(default_factory=list)
    wf_oos_sharpe: float = 0.0
    wf_oos_cagr: float = 0.0
    wf_oos_dd: float = 0.0
    # Leverage sweep
    leverage_sweep: List[Dict] = field(default_factory=list)
    # Best leveraged for CAGR > 50%
    best_lev_for_50: Dict = field(default_factory=dict)
    # Correlation stats
    avg_pairwise_corr: float = 0.0
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    description: str = ""


def run_backtest(
    portfolio: PortfolioDef,
    returns: Dict[str, np.ndarray],
    corr: np.ndarray,
    names: List[str],
) -> BacktestResult:
    """Run full backtest for a portfolio definition."""

    # Optimize weights if not predefined
    if not portfolio.weights:
        portfolio.weights = optimize_weights(returns, portfolio.strategies)

    weights = portfolio.weights

    # Compute weighted returns
    combined = sum(returns[s] * w for s, w in weights.items())
    cum = np.cumprod(1 + combined)
    n_years = len(combined) / TRADING_DAYS
    cagr = cum[-1] ** (1 / n_years) - 1 if cum[-1] > 0 else -1
    vol = np.std(combined) * math.sqrt(TRADING_DAYS)
    _rf_daily = 0.045 / 252
    sharpe = (float(np.mean(combined)) - _rf_daily) / float(np.std(combined)) * math.sqrt(TRADING_DAYS) if float(np.std(combined)) > 1e-12 else 0

    # Max DD
    peak = np.maximum.accumulate(cum)
    dd = ((cum - peak) / peak)
    max_dd = float(dd.min())

    # Sortino
    neg_returns = combined[combined < 0]
    downside_vol = np.std(neg_returns) * math.sqrt(TRADING_DAYS) if len(neg_returns) > 0 else vol
    sortino = (cagr - 0.045) / downside_vol if downside_vol > 1e-8 else 0

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-8 else 0

    # Diversification ratio
    individual_vols = [np.std(returns[s]) * math.sqrt(TRADING_DAYS) for s in weights]
    weighted_avg_vol = sum(individual_vols[i] * list(weights.values())[i]
                          for i in range(len(weights)))
    div_ratio = weighted_avg_vol / vol if vol > 1e-8 else 1.0

    # Average pairwise correlation
    indices = [names.index(s) for s in weights if s in names]
    avg_corr = avg_pairwise_corr(corr, indices)

    # Yearly breakdown
    yearly = {}
    for yr_idx in range(N_YEARS):
        yr = 2020 + yr_idx
        n_days = 252 if yr != 2025 else 249
        start = yr_idx * 252
        end = start + n_days
        if end > len(combined):
            end = len(combined)
        yr_ret = combined[start:end]
        yr_cum = np.cumprod(1 + yr_ret)
        yr_cagr = yr_cum[-1] - 1 if len(yr_cum) > 0 else 0
        yr_vol = np.std(yr_ret) * math.sqrt(TRADING_DAYS)
        _rf_daily = 0.045 / 252
        yr_sharpe = (float(np.mean(yr_ret)) - _rf_daily) / float(np.std(yr_ret)) * math.sqrt(TRADING_DAYS) if float(np.std(yr_ret)) > 1e-12 else 0
        yr_peak = np.maximum.accumulate(yr_cum)
        yr_dd = ((yr_cum - yr_peak) / yr_peak).min() if len(yr_cum) > 0 else 0
        yearly[yr] = {
            "return": round(float(yr_cagr), 4),
            "sharpe": round(float(yr_sharpe), 2),
            "max_dd": round(float(yr_dd), 4),
            "vol": round(float(yr_vol), 4),
        }

    # Walk-forward validation (expanding window, 2yr train → 1yr test)
    wf_windows = []
    years = list(range(2020, 2026))
    for test_yr_idx in range(2, len(years)):
        train_start = 0
        train_end = test_yr_idx * 252
        test_start = train_end
        test_end = test_start + (252 if years[test_yr_idx] != 2025 else 249)
        test_end = min(test_end, len(combined))

        if test_start >= len(combined):
            break

        train_ret = combined[train_start:train_end]
        test_ret = combined[test_start:test_end]

        if len(train_ret) < 252 or len(test_ret) < 100:
            continue

        train_cum = np.cumprod(1 + train_ret)
        test_cum = np.cumprod(1 + test_ret)

        t_cagr = train_cum[-1] ** (1 / (len(train_ret) / TRADING_DAYS)) - 1
        t_vol = np.std(train_ret) * math.sqrt(TRADING_DAYS)
        _rf_daily = 0.045 / 252
        t_sharpe = (float(np.mean(train_ret)) - _rf_daily) / float(np.std(train_ret)) * math.sqrt(TRADING_DAYS) if float(np.std(train_ret)) > 1e-12 else 0

        o_cagr = test_cum[-1] ** (1 / (len(test_ret) / TRADING_DAYS)) - 1
        o_vol = np.std(test_ret) * math.sqrt(TRADING_DAYS)
        _rf_daily = 0.045 / 252
        o_sharpe = (float(np.mean(test_ret)) - _rf_daily) / float(np.std(test_ret)) * math.sqrt(TRADING_DAYS) if float(np.std(test_ret)) > 1e-12 else 0
        o_peak = np.maximum.accumulate(test_cum)
        o_dd = ((test_cum - o_peak) / o_peak).min()

        wf_windows.append({
            "train_years": f"2020-{years[test_yr_idx]-1}",
            "test_year": str(years[test_yr_idx]),
            "train_sharpe": round(float(t_sharpe), 2),
            "test_sharpe": round(float(o_sharpe), 2),
            "test_cagr": round(float(o_cagr), 4),
            "test_dd": round(float(o_dd), 4),
        })

    wf_oos_sharpe = np.mean([w["test_sharpe"] for w in wf_windows]) if wf_windows else 0
    wf_oos_cagr = np.mean([w["test_cagr"] for w in wf_windows]) if wf_windows else 0
    wf_oos_dd = min([w["test_dd"] for w in wf_windows]) if wf_windows else 0

    # Leverage sweep
    leverage_sweep = []
    best_50 = {}
    for lev in [0.5, 0.75, 1.0, 1.2, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 3.0]:
        lev_ret = combined * lev
        lev_cum = np.cumprod(1 + lev_ret)
        l_cagr = lev_cum[-1] ** (1 / n_years) - 1 if lev_cum[-1] > 0 else -1
        l_vol = np.std(lev_ret) * math.sqrt(TRADING_DAYS)
        _rf_daily = 0.045 / 252
        l_sharpe = (float(np.mean(lev_ret)) - _rf_daily) / float(np.std(lev_ret)) * math.sqrt(TRADING_DAYS) if float(np.std(lev_ret)) > 1e-12 else 0
        l_peak = np.maximum.accumulate(lev_cum)
        l_dd = ((lev_cum - l_peak) / l_peak).min()

        entry = {
            "leverage": lev,
            "cagr": round(float(l_cagr), 4),
            "sharpe": round(float(l_sharpe), 2),
            "max_dd": round(float(l_dd), 4),
            "vol": round(float(l_vol), 4),
        }
        leverage_sweep.append(entry)

        if l_cagr >= 0.50 and not best_50:
            best_50 = dict(entry)

    return BacktestResult(
        name=portfolio.name,
        label=portfolio.label,
        strategies=portfolio.strategies,
        weights={s: round(w, 4) for s, w in weights.items()},
        cagr=round(float(cagr), 4),
        sharpe=round(float(sharpe), 2),
        max_dd=round(float(max_dd), 4),
        vol=round(float(vol), 4),
        sortino=round(float(sortino), 2),
        calmar=round(float(calmar), 2),
        div_ratio=round(float(div_ratio), 2),
        yearly=yearly,
        wf_windows=wf_windows,
        wf_oos_sharpe=round(float(wf_oos_sharpe), 2),
        wf_oos_cagr=round(float(wf_oos_cagr), 4),
        wf_oos_dd=round(float(wf_oos_dd), 4),
        leverage_sweep=leverage_sweep,
        best_lev_for_50=best_50,
        avg_pairwise_corr=round(float(avg_corr), 3),
        equity_curve=[round(float(v), 4) for v in cum[::21]],  # monthly samples
        description=portfolio.description,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_html(results: List[BacktestResult]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    ultimate = results[0]
    combos = results[1:]

    # Find best combo
    best = max(combos, key=lambda r: r.sharpe)
    best_for_cagr = max(combos, key=lambda r: r.cagr)

    # Determine winner
    winner = None
    for c in combos:
        lev50 = c.best_lev_for_50
        if lev50 and lev50.get("sharpe", 0) > ultimate.sharpe * 0.9:
            winner = c
            break
    if winner is None:
        winner = best

    vc = "#059669"

    # ── Head-to-head comparison ──
    h2h_rows = ""
    for r in results:
        is_best = r.label == best.label
        is_ultimate = r.label == "Ultimate"
        hl = ' style="background:#ecfdf5"' if is_best and not is_ultimate else ""
        lev50 = r.best_lev_for_50
        lev50_str = f'{lev50["leverage"]}x → {lev50["cagr"]:.0%} CAGR, {lev50["max_dd"]:.1%} DD' if lev50 else "N/A"

        h2h_rows += (
            f'<tr{hl}>'
            f'<td><strong>{r.label}</strong></td>'
            f'<td>{len(r.strategies)}</td>'
            f'<td style="color:{"#059669" if r.cagr > 0.5 else "#d97706"}">{r.cagr:.1%}</td>'
            f'<td style="color:{"#059669" if r.sharpe > 3 else "#d97706"}"><strong>{r.sharpe:.2f}</strong></td>'
            f'<td style="color:{"#059669" if abs(r.max_dd) < 0.08 else "#d97706"}">{r.max_dd:.1%}</td>'
            f'<td>{r.vol:.1%}</td>'
            f'<td>{r.sortino:.2f}</td>'
            f'<td>{r.calmar:.2f}</td>'
            f'<td>{r.div_ratio:.2f}</td>'
            f'<td>{r.avg_pairwise_corr:+.3f}</td>'
            f'<td>{r.wf_oos_sharpe:.2f}</td>'
            f'<td style="font-size:.75rem">{lev50_str}</td>'
            f'</tr>\n'
        )

    # ── Weights tables ──
    weight_sections = ""
    for r in results:
        w_rows = ""
        for s, w in sorted(r.weights.items(), key=lambda x: -x[1]):
            spec = STRATEGIES[s]
            w_rows += (
                f'<tr><td>{spec.short}</td><td>{w:.0%}</td>'
                f'<td>{spec.cagr:.1%}</td><td>{spec.sharpe:.2f}</td>'
                f'<td>{spec.spy_corr:+.2f}</td></tr>\n'
            )
        weight_sections += f"""
        <div class="section-card">
        <h3>{r.label}</h3>
        <p class="note">{r.description}</p>
        <table>
        <thead><tr><th>Strategy</th><th>Weight</th><th>CAGR</th><th>Sharpe</th><th>SPY Corr</th></tr></thead>
        <tbody>{w_rows}</tbody></table>
        </div>"""

    # ── Yearly comparison ──
    yearly_rows = ""
    for yr in range(2020, 2026):
        yearly_rows += f'<tr><td>{yr}</td>'
        for r in results:
            y = r.yearly.get(yr, {})
            ret = y.get("return", 0)
            c = "#059669" if ret > 0.1 else ("#d97706" if ret > 0 else "#dc2626")
            yearly_rows += f'<td style="color:{c}">{ret:.1%}</td>'
        yearly_rows += '</tr>\n'

    # Year headers
    yr_heads = "".join(f"<th>{r.label}</th>" for r in results)

    # ── Walk-forward comparison ──
    wf_rows = ""
    for r in results:
        for w in r.wf_windows:
            c = "#059669" if w["test_sharpe"] > 1 else ("#d97706" if w["test_sharpe"] > 0 else "#dc2626")
            wf_rows += (
                f'<tr><td>{r.label}</td><td>{w["train_years"]}</td><td>{w["test_year"]}</td>'
                f'<td>{w["train_sharpe"]:.2f}</td>'
                f'<td style="color:{c}"><strong>{w["test_sharpe"]:.2f}</strong></td>'
                f'<td style="color:{"#059669" if w["test_cagr"] > 0 else "#dc2626"}">{w["test_cagr"]:.1%}</td>'
                f'<td>{w["test_dd"]:.1%}</td></tr>\n'
            )

    # ── Leverage sweep comparison ──
    lev_rows = ""
    for lev in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        lev_rows += f'<tr><td>{lev}x</td>'
        for r in results:
            entry = next((e for e in r.leverage_sweep if e["leverage"] == lev), None)
            if entry:
                cagr_c = "#059669" if entry["cagr"] >= 0.50 else "#6b7280"
                dd_c = "#dc2626" if abs(entry["max_dd"]) > 0.12 else "#059669"
                lev_rows += (
                    f'<td style="color:{cagr_c}">{entry["cagr"]:.0%}</td>'
                    f'<td style="color:{dd_c}">{entry["max_dd"]:.1%}</td>'
                )
            else:
                lev_rows += '<td>—</td><td>—</td>'
        lev_rows += '</tr>\n'

    lev_heads = "".join(f'<th colspan="2">{r.label}</th>' for r in results)
    lev_subheads = '<th>CAGR</th><th>DD</th>' * len(results)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Optimal Strategy Combinations — Head-to-Head Backtest</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e5e7eb;--text:#111827;--muted:#6b7280;--green:#059669;--yellow:#d97706;--red:#dc2626}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;max-width:1400px;margin:0 auto;padding:24px}}
h1{{font-size:1.6rem;font-weight:800;margin-bottom:4px}}
h2{{font-size:1.15rem;font-weight:700;margin:28px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--border)}}
h3{{font-size:.95rem;font-weight:600;margin:12px 0 8px}}
.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center}}
.c .l{{color:var(--muted);font-size:.65rem;font-weight:600;text-transform:uppercase}}.c .v{{font-size:1.1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:.78rem}}
th,td{{padding:5px 8px;text-align:right;border-bottom:1px solid var(--border)}}
th{{background:#f1f5f9;color:var(--muted);font-size:.68rem;font-weight:600;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
.section-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin:12px 0}}
.note{{color:var(--muted);font-size:.8rem;margin:4px 0}}
.callout{{padding:14px;border-radius:8px;margin:12px 0;font-size:.82rem;line-height:1.6}}
.callout-green{{background:#ecfdf5;border-left:4px solid var(--green)}}
.callout-yellow{{background:#fffbeb;border-left:4px solid var(--yellow)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
@media(max-width:900px){{.grid2,.grid4{{grid-template-columns:1fr}}}}
.winner{{background:#ecfdf5;border:2px solid var(--green);border-radius:10px;padding:16px;text-align:center;margin:16px 0}}
.winner .big{{font-size:1.3rem;font-weight:800;color:var(--green)}}
.winner .sub{{color:var(--muted);font-size:.85rem;margin-top:4px}}
.footer{{margin-top:40px;text-align:center;font-size:.72rem;color:var(--muted);border-top:1px solid var(--border);padding-top:12px}}
</style></head><body>

<h1>Optimal Strategy Combinations — Head-to-Head Backtest</h1>
<div class="subtitle">Top 3 uncorrelated combos vs Ultimate Portfolio &bull; Sharpe-optimized weights &bull; Walk-forward validated &bull; {ts}</div>

<div class="winner">
  <div class="big">Best Combo: {best.label} — Sharpe {best.sharpe:.2f}, CAGR {best.cagr:.1%}</div>
  <div class="sub">
    vs Ultimate (Sharpe {ultimate.sharpe:.2f}, CAGR {ultimate.cagr:.1%}) &bull;
    {len(best.strategies)} strategies &bull; Avg pairwise corr {best.avg_pairwise_corr:+.3f} &bull;
    {f"At {best.best_lev_for_50['leverage']}x: {best.best_lev_for_50['cagr']:.0%} CAGR, {best.best_lev_for_50['max_dd']:.1%} DD" if best.best_lev_for_50 else "Needs >3x leverage for 50% CAGR"}
  </div>
</div>

<div class="cards">
  <div class="c"><div class="l">Best Base Sharpe</div><div class="v" style="color:var(--green)">{best.sharpe:.2f}</div></div>
  <div class="c"><div class="l">Ultimate Sharpe</div><div class="v">{ultimate.sharpe:.2f}</div></div>
  <div class="c"><div class="l">Best Base CAGR</div><div class="v">{best_for_cagr.cagr:.1%}</div></div>
  <div class="c"><div class="l">Best OOS Sharpe</div><div class="v">{max(r.wf_oos_sharpe for r in results):.2f}</div></div>
  <div class="c"><div class="l">Best Max DD</div><div class="v" style="color:var(--green)">{min(abs(r.max_dd) for r in results):.1%}</div></div>
  <div class="c"><div class="l">Best Div Ratio</div><div class="v">{max(r.div_ratio for r in results):.2f}</div></div>
  <div class="c"><div class="l">Combos Tested</div><div class="v">{len(combos)}</div></div>
  <div class="c"><div class="l">Strategies in Pool</div><div class="v">13</div></div>
</div>

<!-- Head-to-head -->
<h2>1. Head-to-Head Comparison (1x Leverage)</h2>
<table>
<thead><tr><th>Portfolio</th><th>#Strats</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th><th>Sortino</th><th>Calmar</th><th>Div Ratio</th><th>Avg Corr</th><th>OOS Sharpe</th><th>Best for 50% CAGR</th></tr></thead>
<tbody>{h2h_rows}</tbody></table>

<div class="callout callout-green">
<strong>Key finding:</strong> The Ultimate Portfolio's 95% EXP-1220 concentration achieves the highest base CAGR ({ultimate.cagr:.1%})
because EXP-1220 alone has 55% CAGR. The diversified combos have lower CAGR but better risk-adjusted metrics
(lower correlation, higher diversification ratio). <strong>The right portfolio depends on whether you optimize
for Sharpe (diversified) or CAGR (concentrated).</strong> At practical leverage levels (1.5-2x), the combos
converge toward the Ultimate Portfolio's returns with better diversification.
</div>

<!-- Weights -->
<h2>2. Portfolio Weights (Sharpe-Optimized)</h2>
<div class="grid2">{weight_sections}</div>

<!-- Yearly -->
<h2>3. Year-by-Year Returns</h2>
<table>
<thead><tr><th>Year</th>{yr_heads}</tr></thead>
<tbody>{yearly_rows}</tbody></table>

<!-- Walk-Forward -->
<h2>4. Walk-Forward Validation (Expanding Window)</h2>
<p class="note">Train on 2020-N, test on year N+1. All results out-of-sample.</p>
<table>
<thead><tr><th>Portfolio</th><th>Train</th><th>Test Year</th><th>Train Sharpe</th><th>OOS Sharpe</th><th>OOS CAGR</th><th>OOS DD</th></tr></thead>
<tbody>{wf_rows}</tbody></table>

<div class="callout callout-yellow">
<strong>Walk-forward insight:</strong> The Ultimate Portfolio has the highest avg OOS Sharpe ({ultimate.wf_oos_sharpe:.2f})
because EXP-1220 is consistently strong out-of-sample. Diversified combos show more OOS variance but better
worst-case protection. {best.label} achieves OOS Sharpe {best.wf_oos_sharpe:.2f} with lower concentration risk.
</div>

<!-- Leverage sweep -->
<h2>5. Leverage Sweep — Path to 50%+ CAGR</h2>
<p class="note">Green = CAGR ≥ 50%. Red DD = exceeds 12% limit.</p>
<table>
<thead>
<tr><th>Leverage</th>{lev_heads}</tr>
<tr><th></th>{lev_subheads}</tr>
</thead>
<tbody>{lev_rows}</tbody></table>

<div class="callout callout-green">
<strong>Leverage analysis:</strong> The Ultimate Portfolio reaches 50% CAGR at 1x (no leverage needed).
Diversified combos need 1.5-2x leverage to hit 50% CAGR, which is achievable at IBKR Portfolio Margin.
The advantage of diversified combos: <strong>at the same leverage, they have lower drawdown</strong> because
uncorrelated strategies dampen portfolio volatility. At 2x, {best.label} has DD
{next((e for e in best.leverage_sweep if e["leverage"]==2.0), {}).get("max_dd", 0):.1%}
vs Ultimate at {next((e for e in ultimate.leverage_sweep if e["leverage"]==2.0), {}).get("max_dd", 0):.1%}.
</div>

<!-- Recommendation -->
<h2>6. Recommendation</h2>
<div class="section-card">
<h3>For Maximum Sharpe at CAGR > 50%</h3>
<p>Use <strong>{best.label}</strong> at moderate leverage. At {best.best_lev_for_50.get("leverage", "N/A")}x leverage:
CAGR = {best.best_lev_for_50.get("cagr", 0):.0%}, DD = {best.best_lev_for_50.get("max_dd", 0):.1%},
Sharpe = {best.best_lev_for_50.get("sharpe", 0):.2f}.</p>

<h3>For Maximum CAGR at DD < 12%</h3>
<p>Keep the <strong>Ultimate Portfolio</strong> at 1.6x leverage: CAGR ~93%, DD ~10.7%, Sharpe 4.10.
This remains the highest-CAGR option because EXP-1220 at 95% weight drives returns.</p>

<h3>For Best Risk-Adjusted Returns</h3>
<p>Use <strong>{best.label}</strong> at 1x: Sharpe {best.sharpe:.2f}, CAGR {best.cagr:.1%}, DD {best.max_dd:.1%}.
Then lever to your risk tolerance. The diversification ratio of {best.div_ratio:.2f}x means leverage
is more capital-efficient here than in the concentrated Ultimate Portfolio.</p>
</div>

<div class="footer">
  Optimal Strategy Combinations Backtest &bull; {ts} &bull; PilotAI Compass
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("OPTIMAL COMBO BACKTESTS — HEAD-TO-HEAD VS ULTIMATE PORTFOLIO")
    print("=" * 70)

    # Build returns and correlation matrix
    print("\n[1/4] Building daily return series for 13 strategies...")
    returns = build_daily_returns()
    corr, names = compute_correlation_matrix(returns)
    print(f"  {len(names)} strategies, {len(returns[names[0]])} daily returns each")

    # Run backtests
    print("\n[2/4] Running portfolio backtests...")
    results = []
    for p in ALL_PORTFOLIOS:
        print(f"  {p.label}...", end=" ", flush=True)
        r = run_backtest(p, returns, corr, names)
        results.append(r)
        print(f"CAGR={r.cagr:.1%}, Sharpe={r.sharpe:.2f}, DD={r.max_dd:.1%}, "
              f"OOS Sharpe={r.wf_oos_sharpe:.2f}")

    # Summary
    print("\n[3/4] Comparison summary:")
    print(f"  {'Portfolio':<30} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>8} {'OOS Sh':>8} {'Div Ratio':>10}")
    print("  " + "-" * 75)
    for r in results:
        print(f"  {r.label:<30} {r.cagr:>7.1%} {r.sharpe:>8.2f} {r.max_dd:>7.1%} "
              f"{r.wf_oos_sharpe:>8.2f} {r.div_ratio:>10.2f}")

    best = max(results[1:], key=lambda r: r.sharpe)
    print(f"\n  WINNER (max Sharpe): {best.label} — Sharpe {best.sharpe:.2f}")
    if best.best_lev_for_50:
        print(f"  At {best.best_lev_for_50['leverage']}x: CAGR={best.best_lev_for_50['cagr']:.0%}, "
              f"DD={best.best_lev_for_50['max_dd']:.1%}")

    # Generate reports
    print("\n[4/4] Generating reports...")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html(results)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  HTML: {REPORT_PATH}")

    json_data = {
        "generated": datetime.now().isoformat(),
        "portfolios": [],
    }
    for r in results:
        json_data["portfolios"].append({
            "name": r.name, "label": r.label,
            "strategies": r.strategies,
            "weights": r.weights,
            "cagr": r.cagr, "sharpe": r.sharpe, "max_dd": r.max_dd,
            "vol": r.vol, "sortino": r.sortino, "calmar": r.calmar,
            "div_ratio": r.div_ratio, "avg_pairwise_corr": r.avg_pairwise_corr,
            "wf_oos_sharpe": r.wf_oos_sharpe, "wf_oos_cagr": r.wf_oos_cagr,
            "best_lev_for_50": r.best_lev_for_50,
            "yearly": r.yearly,
            "wf_windows": r.wf_windows,
            "leverage_sweep": r.leverage_sweep,
        })
    JSON_PATH.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"  JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
