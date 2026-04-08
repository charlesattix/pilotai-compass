"""
Combined Portfolio V2 — multi-strategy portfolio combining uncorrelated
alpha streams with optimised allocation.

Proven streams:
  - EXP-880: ML-filtered credit spreads (76.9% CAGR, 10.2% DD, Sharpe 4.97)
  - EXP-1000: Intraday mean reversion (10.6% CAGR, 1.2% DD, Sharpe 9.92)

Optional streams:
  - Earnings volatility capture
  - Overnight gap fade

Tests allocations from 50/50 through 80/20, finds optimal blend,
quantifies diversification benefit.

Usage::

    from compass.combined_portfolio_v2 import CombinedPortfolioV2
    pf = CombinedPortfolioV2()
    result = pf.optimize(streams)
    CombinedPortfolioV2.generate_report(result)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "combined_portfolio_v2.html"
TRADING_DAYS = 252


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class AlphaStream:
    """A single return stream with its characteristics."""

    name: str
    cagr_pct: float
    max_dd_pct: float
    sharpe: float
    win_rate: float = 0.0
    annual_pnl: float = 0.0  # on $100K
    n_trades_year: float = 0.0
    source_experiment: str = ""


@dataclass
class AllocationResult:
    """Result for one specific allocation mix."""

    weights: Dict[str, float]
    combined_cagr: float
    combined_dd: float
    combined_sharpe: float
    combined_sortino: float
    diversification_ratio: float  # weighted DD / combined DD
    per_stream_contribution: Dict[str, float]


@dataclass
class PortfolioResult:
    """Full optimisation result."""

    streams: List[AlphaStream]
    correlation_matrix: Dict[Tuple[str, str], float]
    allocations_tested: List[AllocationResult]
    optimal_allocation: AllocationResult
    max_sharpe_allocation: AllocationResult
    max_cagr_dd_constrained: AllocationResult  # max CAGR at DD < 12%
    # Diversification
    standalone_best_cagr: float
    standalone_best_sharpe: float
    cagr_improvement_pct: float
    dd_reduction_pct: float
    # Leverage analysis
    leverage_results: List[Dict[str, Any]]
    can_hit_100_cagr: bool
    leverage_for_100: float


# ── Portfolio math ───────────────────────────────────────────────────────


def portfolio_metrics(
    streams: List[AlphaStream],
    weights: Dict[str, float],
    correlations: Dict[Tuple[str, str], float],
) -> Tuple[float, float, float, float]:
    """Compute portfolio CAGR, DD, Sharpe, Sortino from weighted streams.

    Returns: (cagr, max_dd, sharpe, sortino)
    """
    names = [s.name for s in streams]
    n = len(names)
    w = np.array([weights.get(name, 0.0) for name in names])
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0, 0.0, 0.0, 0.0
    w = w / w_sum

    # Weighted return
    cagrs = np.array([s.cagr_pct for s in streams])
    port_cagr = float(w @ cagrs)

    # Portfolio DD via correlation-adjusted combination
    dds = np.array([s.max_dd_pct for s in streams])
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            key = (names[i], names[j])
            rkey = (names[j], names[i])
            rho = correlations.get(key, correlations.get(rkey, 0.3))
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    cov = np.outer(dds, dds) * corr_matrix
    port_var = float(w @ cov @ w)
    port_dd = math.sqrt(max(port_var, 0))

    # Sharpe: CAGR / DD (annualised proxy)
    port_sharpe = port_cagr / port_dd if port_dd > 0.01 else 0.0

    # Sortino: use Sharpe * 1.2 as estimate (downside vol < total vol)
    port_sortino = port_sharpe * 1.2

    return port_cagr, port_dd, port_sharpe, port_sortino


def diversification_ratio(
    streams: List[AlphaStream],
    weights: Dict[str, float],
    combined_dd: float,
) -> float:
    """Weighted sum of individual DDs / combined DD. >1 means diversification helps."""
    names = [s.name for s in streams]
    weighted_dd = sum(weights.get(s.name, 0) * s.max_dd_pct for s in streams)
    w_sum = sum(weights.get(n, 0) for n in names)
    if w_sum > 0:
        weighted_dd /= w_sum
    if combined_dd <= 0.01:
        return 1.0
    return weighted_dd / combined_dd


# ── Allocation sweep ─────────────────────────────────────────────────────


def sweep_allocations(
    streams: List[AlphaStream],
    correlations: Dict[Tuple[str, str], float],
    step: float = 0.05,
) -> List[AllocationResult]:
    """Sweep all allocation combinations at given step size."""
    names = [s.name for s in streams]
    n = len(names)
    results: List[AllocationResult] = []

    if n == 2:
        for w0 in np.arange(0.10, 0.95, step):
            w1 = 1.0 - w0
            weights = {names[0]: w0, names[1]: w1}
            cagr, dd, sharpe, sortino = portfolio_metrics(streams, weights, correlations)
            dr = diversification_ratio(streams, weights, dd)
            contribs = {s.name: weights[s.name] * s.cagr_pct for s in streams}
            results.append(AllocationResult(
                weights=weights, combined_cagr=cagr, combined_dd=dd,
                combined_sharpe=sharpe, combined_sortino=sortino,
                diversification_ratio=dr, per_stream_contribution=contribs,
            ))
    elif n >= 3:
        rng = np.random.RandomState(42)
        for _ in range(20000):
            raw = rng.dirichlet(np.ones(n))
            weights = {names[i]: float(raw[i]) for i in range(n)}
            cagr, dd, sharpe, sortino = portfolio_metrics(streams, weights, correlations)
            dr = diversification_ratio(streams, weights, dd)
            contribs = {s.name: weights[s.name] * s.cagr_pct for s in streams}
            results.append(AllocationResult(
                weights=weights, combined_cagr=cagr, combined_dd=dd,
                combined_sharpe=sharpe, combined_sortino=sortino,
                diversification_ratio=dr, per_stream_contribution=contribs,
            ))

    return results


def find_optimal(
    allocations: List[AllocationResult],
    dd_constraint: float = 12.0,
) -> Tuple[AllocationResult, AllocationResult, AllocationResult]:
    """Find optimal, max-Sharpe, and max-CAGR-at-DD-constraint allocations."""
    if not allocations:
        empty = AllocationResult({}, 0, 0, 0, 0, 1, {})
        return empty, empty, empty

    # Max Sharpe (unconstrained)
    max_sharpe = max(allocations, key=lambda a: a.combined_sharpe)

    # Max CAGR at DD < constraint
    constrained = [a for a in allocations if a.combined_dd <= dd_constraint]
    if constrained:
        max_cagr_dd = max(constrained, key=lambda a: a.combined_cagr)
    else:
        max_cagr_dd = min(allocations, key=lambda a: a.combined_dd)

    # Optimal: highest Calmar (CAGR / DD)
    optimal = max(allocations, key=lambda a: a.combined_cagr / max(a.combined_dd, 0.01))

    return optimal, max_sharpe, max_cagr_dd


# ── Leverage analysis ────────────────────────────────────────────────────


def leverage_sweep(
    base_cagr: float,
    base_dd: float,
    max_leverage: float = 5.0,
    step: float = 0.25,
) -> List[Dict[str, Any]]:
    """Sweep leverage on the optimal allocation."""
    results = []
    for lev in np.arange(1.0, max_leverage + step / 2, step):
        cagr = base_cagr * lev
        dd = base_dd * lev
        sharpe = base_cagr / base_dd if base_dd > 0 else 0  # constant
        results.append({
            "leverage": float(lev),
            "cagr": cagr,
            "dd": dd,
            "sharpe": sharpe,
            "within_12_dd": bool(dd <= 12.0),
        })
    return results


# ── Core engine ──────────────────────────────────────────────────────────


# Default proven streams
DEFAULT_STREAMS = [
    AlphaStream("CS-880", cagr_pct=76.9, max_dd_pct=10.2, sharpe=4.97,
                win_rate=0.87, annual_pnl=76900, n_trades_year=31.8,
                source_experiment="EXP-880"),
    AlphaStream("Intraday-1000", cagr_pct=10.6, max_dd_pct=1.2, sharpe=9.92,
                win_rate=0.859, annual_pnl=10600, n_trades_year=67,
                source_experiment="EXP-1000"),
]

DEFAULT_CORRELATIONS: Dict[Tuple[str, str], float] = {
    ("CS-880", "Intraday-1000"): 0.033,
    ("CS-880", "Earnings"): 0.15,
    ("CS-880", "OvernightGap"): 0.20,
    ("Intraday-1000", "Earnings"): 0.10,
    ("Intraday-1000", "OvernightGap"): 0.05,
    ("Earnings", "OvernightGap"): 0.08,
}

OPTIONAL_STREAMS = [
    AlphaStream("Earnings", cagr_pct=8.0, max_dd_pct=3.0, sharpe=4.0,
                win_rate=0.72, annual_pnl=8000, n_trades_year=16,
                source_experiment="EXP-1060 (hypothetical)"),
    AlphaStream("OvernightGap", cagr_pct=6.0, max_dd_pct=2.5, sharpe=3.5,
                win_rate=0.65, annual_pnl=6000, n_trades_year=60,
                source_experiment="EXP-1070 (hypothetical)"),
]


class CombinedPortfolioV2:
    """Multi-stream portfolio optimiser."""

    def __init__(
        self,
        streams: Optional[List[AlphaStream]] = None,
        correlations: Optional[Dict[Tuple[str, str], float]] = None,
        include_optional: bool = False,
    ):
        base = streams or list(DEFAULT_STREAMS)
        if include_optional:
            base = base + list(OPTIONAL_STREAMS)
        self.streams = base
        self.correlations = correlations or dict(DEFAULT_CORRELATIONS)

    def optimize(self, dd_constraint: float = 12.0) -> PortfolioResult:
        """Run full allocation optimisation."""
        allocations = sweep_allocations(self.streams, self.correlations)
        optimal, max_sh, max_cagr_dd = find_optimal(allocations, dd_constraint)

        # Standalone bests
        best_cagr_solo = max(self.streams, key=lambda s: s.cagr_pct)
        best_sharpe_solo = max(self.streams, key=lambda s: s.sharpe)

        # Diversification metrics
        cagr_imp = (optimal.combined_cagr - best_cagr_solo.cagr_pct) / best_cagr_solo.cagr_pct * 100
        dd_red = (best_cagr_solo.max_dd_pct - optimal.combined_dd) / best_cagr_solo.max_dd_pct * 100

        # Leverage on optimal
        lev_results = leverage_sweep(optimal.combined_cagr, optimal.combined_dd)
        can_100 = any(r["cagr"] >= 100 and r["within_12_dd"] for r in lev_results)
        lev_100_list = [r["leverage"] for r in lev_results if r["cagr"] >= 100 and r["within_12_dd"]]
        lev_100 = min(lev_100_list) if lev_100_list else 0.0

        return PortfolioResult(
            streams=self.streams,
            correlation_matrix=self.correlations,
            allocations_tested=allocations,
            optimal_allocation=optimal,
            max_sharpe_allocation=max_sh,
            max_cagr_dd_constrained=max_cagr_dd,
            standalone_best_cagr=best_cagr_solo.cagr_pct,
            standalone_best_sharpe=best_sharpe_solo.sharpe,
            cagr_improvement_pct=cagr_imp,
            dd_reduction_pct=dd_red,
            leverage_results=lev_results,
            can_hit_100_cagr=can_100,
            leverage_for_100=lev_100,
        )

    @staticmethod
    def generate_report(result: PortfolioResult, output_path: Path = DEFAULT_OUTPUT) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(result), encoding="utf-8")
        return output_path


# ── HTML ─────────────────────────────────────────────────────────────────


def _fr(v): return f"{v:.2f}"
def _fp(v): return f"{v:.1f}%"
def _fd(v): return f"${v:,.0f}"
def _ti(m): return '<span style="color:#3fb950">&#10003;</span>' if m else '<span style="color:#f85149">&#10007;</span>'


def _build_html(r: PortfolioResult) -> str:
    opt = r.optimal_allocation
    msh = r.max_sharpe_allocation
    mcd = r.max_cagr_dd_constrained
    oc = "#3fb950" if r.can_hit_100_cagr else "#d29922"

    stream_rows = "".join(
        f"<tr><td style='text-align:left'>{s.name}</td><td>{_fp(s.cagr_pct)}</td>"
        f"<td>{_fp(s.max_dd_pct)}</td><td>{_fr(s.sharpe)}</td>"
        f"<td>{_fp(s.win_rate*100)}</td><td>{s.source_experiment}</td></tr>"
        for s in r.streams
    )

    def _alloc_row(name, a):
        wstr = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(a.weights.items()))
        return (f"<tr><td style='text-align:left'>{name}</td><td>{wstr}</td>"
                f"<td>{_fp(a.combined_cagr)}</td><td>{_fp(a.combined_dd)}</td>"
                f"<td>{_fr(a.combined_sharpe)}</td><td>{_fr(a.diversification_ratio)}</td></tr>")

    alloc_rows = _alloc_row("Optimal (Calmar)", opt) + _alloc_row("Max Sharpe", msh) + _alloc_row("Max CAGR@DD<12%", mcd)

    def _lev_row(l):
        hl = " style='color:#3fb950;font-weight:700'" if l["cagr"] >= 100 and l["within_12_dd"] else ""
        return f"<tr{hl}><td>{l['leverage']:.2f}x</td><td>{_fp(l['cagr'])}</td><td>{_fp(l['dd'])}</td><td>{_ti(l['within_12_dd'])}</td></tr>"
    key_levs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    lev_rows = "".join(_lev_row(l) for l in r.leverage_results if l["leverage"] in key_levs)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Combined Portfolio V2</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2{{color:#58a6ff}}.meta{{color:#8b949e}}
.hero{{background:#161b22;border:2px solid {oc};border-radius:12px;padding:24px;text-align:center;margin:20px 0}}
.hero .big{{font-size:2em;font-weight:800;color:{oc}}}.hero .sub{{color:#8b949e}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}th{{color:#8b949e;background:#161b22}}
</style></head><body>
<h1>Combined Portfolio V2</h1>
<div class="hero">
<div class="big">{"100% CAGR ACHIEVABLE at " + _fr(r.leverage_for_100) + "x" if r.can_hit_100_cagr else "Optimal: " + _fp(opt.combined_cagr) + " CAGR"}</div>
<div class="sub">{len(r.streams)} streams &middot; {len(r.allocations_tested)} allocations tested &middot;
   Div ratio: {_fr(opt.diversification_ratio)}x</div>
</div>
<div class="cards">
<div class="c"><div class="l">Optimal CAGR</div><div class="v">{_fp(opt.combined_cagr)}</div></div>
<div class="c"><div class="l">Optimal DD</div><div class="v">{_fp(opt.combined_dd)}</div></div>
<div class="c"><div class="l">Optimal Sharpe</div><div class="v">{_fr(opt.combined_sharpe)}</div></div>
<div class="c"><div class="l">Div Ratio</div><div class="v">{_fr(opt.diversification_ratio)}x</div></div>
<div class="c"><div class="l">Best Solo CAGR</div><div class="v">{_fp(r.standalone_best_cagr)}</div></div>
<div class="c"><div class="l">CAGR Improvement</div><div class="v">{_fp(r.cagr_improvement_pct)}</div></div>
<div class="c"><div class="l">DD Reduction</div><div class="v">{_fp(r.dd_reduction_pct)}</div></div>
<div class="c"><div class="l">100% CAGR?</div><div class="v" style="color:{oc}">{'YES at '+_fr(r.leverage_for_100)+'x' if r.can_hit_100_cagr else 'NO'}</div></div>
</div>
<h2>Alpha Streams</h2>
<table><tr><th style="text-align:left">Stream</th><th>CAGR</th><th>DD</th><th>Sharpe</th><th>WR</th><th>Source</th></tr>{stream_rows}</table>
<h2>Optimal Allocations</h2>
<table><tr><th style="text-align:left">Method</th><th>Weights</th><th>CAGR</th><th>DD</th><th>Sharpe</th><th>Div Ratio</th></tr>{alloc_rows}</table>
<h2>Leverage on Optimal</h2>
<table><tr><th>Leverage</th><th>CAGR</th><th>DD</th><th>DD&lt;12%</th></tr>{lev_rows}</table>
</body></html>"""
