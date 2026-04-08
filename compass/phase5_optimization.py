"""
Phase 5 — Portfolio Optimization Across EXP-400/401/503/600

Runs the full PortfolioOptimizer pipeline (4 methods + regime tilts + event scaling)
on 2020-2025 return streams for the 4 live paper trading experiments.

Since paper trading only started March 15-22 (too recent for meaningful stats),
we generate calibrated return streams from documented profiles in
compass/portfolio_optimizer.py EXPERIMENT_PROFILES and backtest snapshots.

Comparison baselines:
  - Single-experiment buy-and-hold (EXP-400 only, EXP-401 only, etc.)
  - Equal weight (25% each)
  - Optimized: max_sharpe, risk_parity, ERC, min_variance
  - Each optimized variant also run with regime tilts (BULL/BEAR/NEUTRAL)

Uses corrected Sharpe formula: arithmetic mean × sqrt(252) / std(daily, ddof=1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRADING_DAYS = 252

from compass.portfolio_optimizer import (
    PortfolioOptimizer, EXPERIMENT_PROFILES, EXPERIMENT_IDS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Return stream generation (calibrated to documented profiles)
# ═══════════════════════════════════════════════════════════════════════════

# Documented profiles from EXPERIMENT_PROFILES and paper trading registry.
# Each tuple: (annual_return, annual_vol, crisis_beta)
# Sourced from:
#   - EXP-400: balanced CS+IC (~30% CAGR, 12% vol per backtest)
#   - EXP-401: CS+Straddle/Strangle blend (~22% CAGR, 14% vol)
#   - EXP-503: ML V2 Aggressive (~45% CAGR, 20% vol)
#   - EXP-600: Real Data Optimized (~18% CAGR, 8% vol)
EXPERIMENT_RETURNS_PROFILE = {
    "EXP-400": {"mu": 0.30, "sigma": 0.12, "crisis_beta": 0.9},
    "EXP-401": {"mu": 0.22, "sigma": 0.14, "crisis_beta": 0.6},
    "EXP-503": {"mu": 0.45, "sigma": 0.20, "crisis_beta": 1.2},
    "EXP-600": {"mu": 0.18, "sigma": 0.08, "crisis_beta": 0.4},
}

# Pairwise correlations (calibrated from overlap in strategy components)
EXPERIMENT_CORRELATIONS = {
    ("EXP-400", "EXP-401"): 0.45,
    ("EXP-400", "EXP-503"): 0.55,
    ("EXP-400", "EXP-600"): 0.30,
    ("EXP-401", "EXP-503"): 0.25,
    ("EXP-401", "EXP-600"): 0.15,
    ("EXP-503", "EXP-600"): 0.20,
}


def generate_returns(n_years: float = 6.0, seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate correlated daily return streams for the 4 experiments.

    Uses Cholesky decomposition for correlated normals, then injects crisis
    periods (COVID Mar 2020, 2022 bear) via crisis_beta multiplier.
    """
    rng = np.random.RandomState(seed)
    n = int(n_years * TRADING_DAYS)
    ids = list(EXPERIMENT_RETURNS_PROFILE.keys())
    n_exp = len(ids)

    # Build correlation matrix
    corr = np.eye(n_exp)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i >= j:
                continue
            key = (a, b) if (a, b) in EXPERIMENT_CORRELATIONS else (b, a)
            if key in EXPERIMENT_CORRELATIONS:
                corr[i, j] = corr[j, i] = EXPERIMENT_CORRELATIONS[key]

    # Cholesky with fallback to nearest PSD
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-6)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(corr, 1.0)
        L = np.linalg.cholesky(corr)

    Z = rng.randn(n, n_exp) @ L.T
    results = {}

    for i, eid in enumerate(ids):
        prof = EXPERIMENT_RETURNS_PROFILE[eid]
        daily_mu = prof["mu"] / TRADING_DAYS
        daily_sigma = prof["sigma"] / math.sqrt(TRADING_DAYS)
        rets = daily_mu + daily_sigma * Z[:, i]

        # Crisis injection: COVID days 40-63
        cb = prof["crisis_beta"]
        covid_shock = np.linspace(-0.04, -0.01, 23) * cb + rng.normal(0, 0.005, 23)
        rets[40:min(63, n)] = covid_shock[:min(23, n - 40)]

        # 2022 bear days 500-690
        if n > 690:
            bear_daily = -0.12 / 190 * cb
            rets[500:690] = rng.normal(bear_daily, abs(bear_daily) * 0.8, 190)

        results[eid] = rets.copy()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Corrected Sharpe metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(daily_rets: np.ndarray) -> dict:
    """All metrics with corrected Sharpe formula."""
    if len(daily_rets) < 2:
        return {"cagr_pct": 0, "sharpe": 0, "max_dd_pct": 0, "calmar": 0,
                "sortino": 0, "vol_pct": 0}
    eq = np.cumprod(1 + daily_rets)
    n_yr = len(daily_rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else 0

    # CORRECTED: arithmetic mean × sqrt(252) / std(daily, ddof=1)
    mu = float(daily_rets.mean())
    sigma = float(daily_rets.std(ddof=1))
    sharpe = mu / sigma * math.sqrt(TRADING_DAYS) if sigma > 1e-12 else 0.0

    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    down = daily_rets[daily_rets < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else sigma
    sortino = mu / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0
    vol = sigma * math.sqrt(TRADING_DAYS)

    return {
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(dd * 100, 2),
        "calmar": round(calmar, 2),
        "sortino": round(sortino, 2),
        "vol_pct": round(vol * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Optimization runner
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioRun:
    name: str
    method: str
    regime: str
    weights: Dict[str, float]
    event_scaling: float
    scaled_weights: Dict[str, float]
    metrics: dict
    daily_returns: np.ndarray
    equity: List[float]


def apply_weights(returns: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Compute portfolio daily returns from weights."""
    n = len(next(iter(returns.values())))
    port = np.zeros(n)
    for eid, ret in returns.items():
        w = weights.get(eid, 0)
        port += w * ret
    return port


def run_single_experiment(returns: Dict[str, np.ndarray], eid: str) -> PortfolioRun:
    """Single-experiment buy-and-hold."""
    weights = {e: (1.0 if e == eid else 0.0) for e in returns}
    daily = returns[eid]
    eq = [100_000.0]
    for r in daily:
        eq.append(eq[-1] * (1 + r))
    return PortfolioRun(
        name=f"{eid} only", method="single", regime="N/A",
        weights=weights, event_scaling=1.0, scaled_weights=weights,
        metrics=compute_metrics(daily), daily_returns=daily, equity=eq)


def run_equal_weight(returns: Dict[str, np.ndarray]) -> PortfolioRun:
    """Equal-weight 25% across 4 experiments."""
    n_exp = len(returns)
    weights = {eid: 1.0 / n_exp for eid in returns}
    daily = apply_weights(returns, weights)
    eq = [100_000.0]
    for r in daily:
        eq.append(eq[-1] * (1 + r))
    return PortfolioRun(
        name="Equal Weight (25% each)", method="equal", regime="N/A",
        weights=weights, event_scaling=1.0, scaled_weights=weights,
        metrics=compute_metrics(daily), daily_returns=daily, equity=eq)


def run_optimized(returns: Dict[str, np.ndarray], method: str,
                  regime: str = "NEUTRAL_MACRO",
                  event_scaling: float = 1.0) -> PortfolioRun:
    """Run PortfolioOptimizer with explicit regime and event scaling."""
    optimizer = PortfolioOptimizer(
        returns=returns, risk_free_rate=0.045,
        regime_blend=0.30, min_weight=0.05, periods_per_year=TRADING_DAYS,
    )

    # Call the raw method to avoid macro_db dependency
    method_fns = {
        "max_sharpe": optimizer.max_sharpe,
        "risk_parity": optimizer.risk_parity,
        "equal_risk_contribution": optimizer.equal_risk_contribution,
        "min_variance": optimizer.min_variance,
    }
    raw_weights = method_fns[method]()
    tilted = optimizer.apply_regime_tilt(raw_weights, regime)

    weights = {eid: round(float(w), 6) for eid, w in zip(optimizer.experiment_ids, tilted)}
    scaled = {eid: round(w * event_scaling, 6) for eid, w in weights.items()}

    # Compute portfolio daily returns (use scaled weights for realistic P&L)
    daily = apply_weights(returns, scaled)
    eq = [100_000.0]
    for r in daily:
        eq.append(eq[-1] * (1 + r))

    return PortfolioRun(
        name=f"{method} ({regime}, event={event_scaling:.2f})",
        method=method, regime=regime, weights=weights,
        event_scaling=event_scaling, scaled_weights=scaled,
        metrics=compute_metrics(daily), daily_returns=daily, equity=eq)


# ═══════════════════════════════════════════════════════════════════════════
# Full comparison pipeline
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    single_runs: List[PortfolioRun]
    equal_weight: PortfolioRun
    optimized_runs: List[PortfolioRun]
    best_run: PortfolioRun


def run_full_comparison(seed: int = 42) -> ComparisonResult:
    print("Phase 5 — Portfolio Optimization Across EXP-400/401/503/600")
    print("=" * 65)

    # 1. Generate calibrated return streams
    print("\n  [1/4] Generating calibrated return streams (2020-2025)...")
    returns = generate_returns(n_years=6.0, seed=seed)
    for eid, rets in returns.items():
        m = compute_metrics(rets)
        print(f"    {eid}: CAGR={m['cagr_pct']:+.1f}%, "
              f"Sharpe={m['sharpe']:.2f}, DD={m['max_dd_pct']:.1f}%")

    # 2. Single-experiment baselines
    print("\n  [2/4] Running single-experiment baselines...")
    single_runs = [run_single_experiment(returns, eid) for eid in returns]

    # 3. Equal weight
    print("  [3/4] Running equal-weight baseline...")
    eq_run = run_equal_weight(returns)

    # 4. All 4 optimization methods × 3 regimes × event scaling
    print("  [4/4] Running 4 methods × 3 regimes × event scaling (0.85, 1.0)...")
    optimized = []
    methods = ["max_sharpe", "risk_parity", "equal_risk_contribution", "min_variance"]
    regimes = ["BULL_MACRO", "NEUTRAL_MACRO", "BEAR_MACRO"]

    for method in methods:
        for regime in regimes:
            # Default: no event scaling
            run = run_optimized(returns, method, regime, event_scaling=1.0)
            optimized.append(run)

    # Add event-scaled variants for NEUTRAL (simulating FOMC/CPI window)
    for method in methods:
        run = run_optimized(returns, method, "NEUTRAL_MACRO", event_scaling=0.85)
        optimized.append(run)

    # Find best by Sharpe
    all_runs = single_runs + [eq_run] + optimized
    best = max(all_runs, key=lambda r: r.metrics["sharpe"])

    print(f"\n  Best run: {best.name}")
    print(f"    CAGR={best.metrics['cagr_pct']:.1f}%, "
          f"Sharpe={best.metrics['sharpe']:.2f}, DD={best.metrics['max_dd_pct']:.1f}%")
    print(f"    Weights: {best.weights}")

    return ComparisonResult(
        single_runs=single_runs, equal_weight=eq_run,
        optimized_runs=optimized, best_run=best,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    result: ComparisonResult,
    output_path: str = "reports/portfolio_optimization_backtest.html",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _row(r: PortfolioRun, is_best: bool = False):
        bg = ' style="background:#f0fdf4"' if is_best else ""
        star = " ★" if is_best else ""
        m = r.metrics
        cc = "#16a34a" if m["cagr_pct"] > 0 else "#dc2626"
        dc = "#16a34a" if m["max_dd_pct"] < 15 else "#dc2626"
        w_str = ", ".join(f"{k.split('-')[1]}:{v:.0%}"
                          for k, v in sorted(r.weights.items()) if v > 0.01)
        return f"""<tr{bg}>
          <td>{r.name}{star}</td>
          <td>{r.method}</td>
          <td>{r.regime}</td>
          <td style="color:{cc};font-weight:700">{m['cagr_pct']:+.1f}%</td>
          <td style="font-weight:700">{m['sharpe']:.2f}</td>
          <td style="color:{dc}">{m['max_dd_pct']:.1f}%</td>
          <td>{m['calmar']:.1f}</td>
          <td>{m['sortino']:.1f}</td>
          <td>{m['vol_pct']:.1f}%</td>
          <td style="font-size:0.72rem;color:#64748b">{w_str}</td>
        </tr>"""

    # Single-experiment table
    single_rows = "".join(_row(r, r == result.best_run) for r in result.single_runs)

    # Equal-weight row
    eq_row = _row(result.equal_weight, result.equal_weight == result.best_run)

    # Optimized table, grouped by method
    opt_rows = ""
    for r in result.optimized_runs:
        opt_rows += _row(r, r == result.best_run)

    # Experiment profile table
    prof_rows = ""
    for eid, prof in sorted(EXPERIMENT_PROFILES.items()):
        rp = EXPERIMENT_RETURNS_PROFILE[eid]
        prof_rows += f"""<tr>
          <td>{eid}</td><td>{prof['name']}</td>
          <td>{prof['profile']}</td>
          <td>{rp['mu']:.0%}</td>
          <td>{rp['sigma']:.0%}</td>
          <td>{rp['crisis_beta']:.1f}x</td>
          <td>{prof['momentum_affinity']:.1f}</td>
          <td>{prof['defensive_affinity']:.1f}</td>
        </tr>"""

    # Correlation matrix
    corr_rows = ""
    ids = list(EXPERIMENT_RETURNS_PROFILE.keys())
    corr_header = "<th></th>" + "".join(f"<th>{e}</th>" for e in ids)
    for i, ea in enumerate(ids):
        cells = f"<td>{ea}</td>"
        for j, eb in enumerate(ids):
            if i == j:
                v = 1.0
            else:
                key = (ea, eb) if (ea, eb) in EXPERIMENT_CORRELATIONS else (eb, ea)
                v = EXPERIMENT_CORRELATIONS.get(key, 0)
            cells += f"<td>{v:.2f}</td>"
        corr_rows += f"<tr>{cells}</tr>"

    # Equity SVG for best
    best = result.best_run
    eq_svg = ""
    if len(best.equity) > 2:
        w, h = 780, 220
        pl, pr, pt, pb = 65, 20, 28, 28
        pw, ph = w - pl - pr, h - pt - pb
        n = len(best.equity)
        ym, yx = min(best.equity) * 0.95, max(best.equity) * 1.05
        step = max(1, n // 500)
        pts = [(i, best.equity[i]) for i in range(0, n, step)]
        if pts[-1][0] != n - 1:
            pts.append((n - 1, best.equity[-1]))

        def tx(i): return pl + i / max(n - 1, 1) * pw
        def ty(v): return pt + (1 - (v - ym) / max(yx - ym, 1)) * ph
        d = " ".join(f"{'M' if j == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
                     for j, (i, v) in enumerate(pts))

        # Also overlay equal-weight for comparison
        ew_eq = result.equal_weight.equity
        step_ew = max(1, len(ew_eq) // 500)
        pts_ew = [(i, ew_eq[i]) for i in range(0, len(ew_eq), step_ew)]
        if pts_ew[-1][0] != len(ew_eq) - 1:
            pts_ew.append((len(ew_eq) - 1, ew_eq[-1]))
        d_ew = " ".join(f"{'M' if j == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
                        for j, (i, v) in enumerate(pts_ew))

        eq_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"
  style="border:1px solid #e2e8f0;border-radius:6px">
  <text x="{w//2}" y="16" text-anchor="middle" font-size="11" fill="#64748b">
    Best ({best.method}, {best.regime}) vs Equal Weight (gray)
  </text>
  <path d="{d_ew}" fill="none" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>
  <path d="{d}" fill="none" stroke="#16a34a" stroke-width="1.5"/>
</svg>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 5 — Portfolio Optimization EXP-400/401/503/600</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b}}
h1{{font-size:1.4rem;color:#0f172a}}
h2{{font-size:1.05rem;color:#334155;margin-top:1.5rem;border-bottom:1px solid #e2e8f0;padding-bottom:4px}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:8px;margin:16px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}}
.card .v{{font-size:1.1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:0.8rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.68rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}}
td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}}
td:first-child{{text-align:left}}
svg{{display:block;margin:0.5rem 0}}
.callout{{background:#eff6ff;border-left:4px solid #3b82f6;padding:12px;margin:12px 0;border-radius:4px;font-size:0.85rem}}
</style></head><body>
<h1>Phase 5 — Portfolio Optimization Backtest</h1>
<p class="meta">EXP-400 / EXP-401 / EXP-503 / EXP-600 | 2020-2025 calibrated returns | Corrected Sharpe formula</p>

<div class="callout">
<strong>Methodology:</strong> Paper trading started March 15-22 (too recent for meaningful stats).
Returns are calibrated from documented experiment profiles in compass/portfolio_optimizer.py
and historical backtest snapshots. COVID (days 40-63) and 2022 bear (days 500-690) are
injected via crisis_beta multipliers. All metrics use the corrected Sharpe formula
(arithmetic mean × √252 / std(daily, ddof=1)).
</div>

<div class="grid">
  <div class="card"><div class="l">Best Sharpe</div><div class="v" style="color:#16a34a">{best.metrics['sharpe']:.2f}</div></div>
  <div class="card"><div class="l">Best CAGR</div><div class="v">{best.metrics['cagr_pct']:.1f}%</div></div>
  <div class="card"><div class="l">Best Max DD</div><div class="v">{best.metrics['max_dd_pct']:.1f}%</div></div>
  <div class="card"><div class="l">Best Calmar</div><div class="v">{best.metrics['calmar']:.1f}</div></div>
  <div class="card"><div class="l">Best Method</div><div class="v" style="font-size:0.85rem">{best.method}</div></div>
  <div class="card"><div class="l">Best Regime</div><div class="v" style="font-size:0.85rem">{best.regime}</div></div>
  <div class="card"><div class="l">Experiments</div><div class="v">4</div></div>
  <div class="card"><div class="l">Methods</div><div class="v">4</div></div>
</div>

<h2>Experiment Profiles</h2>
<table>
<tr><th>ID</th><th>Name</th><th>Profile</th><th>CAGR</th><th>Vol</th><th>Crisis Beta</th><th>Mom Affinity</th><th>Def Affinity</th></tr>
{prof_rows}
</table>

<h2>Correlation Matrix (Calibrated)</h2>
<table style="width:auto"><tr>{corr_header}</tr>{corr_rows}</table>

<h2>Single-Experiment Baselines</h2>
<table>
<tr><th>Run</th><th>Method</th><th>Regime</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Sortino</th><th>Vol</th><th>Weights</th></tr>
{single_rows}
</table>

<h2>Equal-Weight Baseline</h2>
<table>
<tr><th>Run</th><th>Method</th><th>Regime</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Sortino</th><th>Vol</th><th>Weights</th></tr>
{eq_row}
</table>

<h2>Optimized Portfolios (4 Methods × 3 Regimes + Event Scaling)</h2>
<table>
<tr><th>Run</th><th>Method</th><th>Regime</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Sortino</th><th>Vol</th><th>Weights</th></tr>
{opt_rows}
</table>

<h2>Equity Curve — Best vs Equal Weight</h2>
{eq_svg}

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/phase5_optimization.py | Calibrated returns (paper trading too recent for stats) |
Sharpe: arithmetic mean × √252 / std(daily, ddof=1) | ★ = best Sharpe
</div>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis():
    result = run_full_comparison(seed=42)
    report = generate_report(result)
    print(f"\n  Report: {report}")
    return result


if __name__ == "__main__":
    run_analysis()
