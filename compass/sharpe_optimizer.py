"""
Sharpe ratio attribution, decomposition, and optimization.

Current portfolio Sharpe ~4.1, North Star target 6.0.  This module:

  1. Decomposes Sharpe by strategy — which drag, which contribute
  2. Tests 3 specific optimizations:
     (a) Volatility targeting — scale exposure to constant 12% annual vol
     (b) Regime filtering — zero-weight strategies in their failing regimes
     (c) Conviction weighting — overweight high-Sharpe strategies dynamically
  3. Implements the best optimization and backtests it

Uses calibrated return streams from production_portfolio_wf.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.production_portfolio_wf import (
    STRATEGY_IDS, STRATEGY_PROFILES, STRATEGY_CORRELATIONS,
    generate_strategy_returns, TRADING_DAYS,
)
from compass.regime_performance import (
    Regime, ALL_REGIMES, generate_regime_returns,
    STRATEGY_PROFILES as REGIME_PROFILES,
)

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SharpeAttribution:
    """Per-strategy contribution to portfolio Sharpe."""
    strategy_id: str
    strategy_name: str
    weight: float
    standalone_sharpe: float
    standalone_cagr: float
    standalone_vol: float
    standalone_dd: float
    marginal_sharpe_contribution: float  # how much this strategy adds
    sharpe_drag: float                   # negative = dragging down
    corr_to_portfolio: float
    is_drag: bool


@dataclass
class OptimizationResult:
    """Result of one optimization approach."""
    name: str
    description: str
    baseline_sharpe: float
    optimized_sharpe: float
    sharpe_improvement: float
    baseline_cagr: float
    optimized_cagr: float
    baseline_dd: float
    optimized_dd: float
    optimized_sortino: float
    optimized_calmar: float
    mechanism: str
    # Yearly detail
    yearly_sharpe: Dict[int, float]
    # Return series
    daily_returns: np.ndarray
    equity_curve: List[float]


@dataclass
class SharpeAnalysisResult:
    """Complete Sharpe optimization analysis."""
    # Attribution
    attributions: List[SharpeAttribution]
    portfolio_sharpe: float
    portfolio_cagr: float
    portfolio_dd: float
    target_sharpe: float
    sharpe_gap: float
    # Top drags
    top_drags: List[str]
    top_contributors: List[str]
    # Optimizations tested
    optimizations: List[OptimizationResult]
    best_optimization: OptimizationResult
    # Final optimized metrics
    final_sharpe: float
    final_cagr: float
    final_dd: float


# ═══════════════════════════════════════════════════════════════════════════
# Sharpe decomposition
# ═══════════════════════════════════════════════════════════════════════════


def _compute_metrics(rets: np.ndarray) -> dict:
    if len(rets) < 2:
        return {"cagr_pct": 0, "sharpe": 0, "max_dd_pct": 0, "calmar": 0,
                "sortino": 0, "vol_pct": 0}
    eq = np.cumprod(1 + rets)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else 0
    mu, std = float(rets.mean()), float(rets.std())
    sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    down = rets[rets < 0]
    down_std = float(down.std()) if len(down) > 1 else std
    sortino = mu / down_std * math.sqrt(TRADING_DAYS) if down_std > 1e-12 else 0
    return {
        "cagr_pct": round(cagr * 100, 2), "sharpe": round(sharpe, 2),
        "max_dd_pct": round(dd * 100, 2), "calmar": round(calmar, 2),
        "sortino": round(sortino, 2), "vol_pct": round(std * math.sqrt(TRADING_DAYS) * 100, 2),
    }


def _yearly_sharpe(rets: np.ndarray, dates: pd.DatetimeIndex) -> Dict[int, float]:
    by_year: Dict[int, List[float]] = {}
    for i, d in enumerate(dates):
        by_year.setdefault(d.year, []).append(rets[i])
    result = {}
    for yr, vals in sorted(by_year.items()):
        arr = np.array(vals)
        mu, std = arr.mean(), arr.std()
        result[yr] = round(mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0, 2)
    return result


class SharpeAnalyzer:
    """Sharpe attribution and optimization engine."""

    def __init__(self, leverage: float = 1.6, target_sharpe: float = 6.0, seed: int = 42):
        self.leverage = leverage
        self.target = target_sharpe
        self.seed = seed

        # Generate return streams
        self.returns = generate_strategy_returns(seed=seed)
        _, self.regimes = generate_regime_returns(seed=seed)
        self.strategy_ids = sorted(self.returns.keys())
        self.dates = self.returns[self.strategy_ids[0]].index
        self.n = len(self.dates)

    def analyze(self) -> SharpeAnalysisResult:
        """Run full Sharpe attribution and optimization."""
        # Step 1: Baseline portfolio (equal weight × leverage)
        baseline_weights = self._baseline_weights()
        baseline_rets = self._portfolio_returns(baseline_weights)
        baseline_m = _compute_metrics(baseline_rets)

        # Step 2: Attribution
        attributions = self._attribute_sharpe(baseline_weights, baseline_rets)

        # Identify drags and contributors
        drags = sorted([a for a in attributions if a.is_drag],
                       key=lambda a: a.sharpe_drag)
        contributors = sorted([a for a in attributions if not a.is_drag],
                              key=lambda a: -a.marginal_sharpe_contribution)

        # Step 3: Test 3 optimizations
        opt1 = self._opt_vol_targeting(baseline_m["sharpe"])
        opt2 = self._opt_regime_filtering(baseline_m["sharpe"])
        opt3 = self._opt_conviction_weighting(baseline_m["sharpe"])

        optimizations = [opt1, opt2, opt3]
        best = max(optimizations, key=lambda o: o.optimized_sharpe)

        return SharpeAnalysisResult(
            attributions=attributions,
            portfolio_sharpe=baseline_m["sharpe"],
            portfolio_cagr=baseline_m["cagr_pct"],
            portfolio_dd=baseline_m["max_dd_pct"],
            target_sharpe=self.target,
            sharpe_gap=round(self.target - baseline_m["sharpe"], 2),
            top_drags=[d.strategy_id for d in drags[:3]],
            top_contributors=[c.strategy_id for c in contributors[:3]],
            optimizations=optimizations,
            best_optimization=best,
            final_sharpe=best.optimized_sharpe,
            final_cagr=best.optimized_cagr,
            final_dd=best.optimized_dd,
        )

    # ── Baseline weights ──────────────────────────────────────────────────

    def _baseline_weights(self) -> Dict[str, float]:
        """Current production weights (from STRATEGY_PROFILES hints, normalized)."""
        hints = {sid: STRATEGY_PROFILES[sid]["weight_hint"] for sid in self.strategy_ids}
        total = sum(hints.values())
        return {sid: h / total for sid, h in hints.items()}

    def _portfolio_returns(self, weights: Dict[str, float]) -> np.ndarray:
        port = np.zeros(self.n)
        for sid in self.strategy_ids:
            port += weights.get(sid, 0) * self.returns[sid].values
        return port * self.leverage

    # ── Sharpe attribution ────────────────────────────────────────────────

    def _attribute_sharpe(
        self, weights: Dict[str, float], port_rets: np.ndarray,
    ) -> List[SharpeAttribution]:
        """Decompose portfolio Sharpe by strategy contribution.

        Marginal contribution = change in portfolio Sharpe when removing one strategy.
        """
        port_m = _compute_metrics(port_rets)
        port_sharpe = port_m["sharpe"]
        attributions = []

        for sid in self.strategy_ids:
            strat_rets = self.returns[sid].values * self.leverage
            strat_m = _compute_metrics(strat_rets)

            # Leave-one-out: portfolio without this strategy
            loo_weights = {k: v for k, v in weights.items() if k != sid}
            loo_total = sum(loo_weights.values())
            if loo_total > 0:
                loo_weights = {k: v / loo_total for k, v in loo_weights.items()}
            loo_rets = self._portfolio_returns(loo_weights)
            loo_m = _compute_metrics(loo_rets)

            # Marginal contribution = port_sharpe - loo_sharpe
            marginal = port_sharpe - loo_m["sharpe"]

            # Correlation to portfolio
            if len(strat_rets) == len(port_rets) and np.std(strat_rets) > 1e-12:
                corr = float(np.corrcoef(strat_rets, port_rets)[0, 1])
            else:
                corr = 0.0

            # Is this strategy a drag?
            is_drag = marginal < 0 or strat_m["sharpe"] < 1.0

            attributions.append(SharpeAttribution(
                strategy_id=sid,
                strategy_name=STRATEGY_PROFILES[sid]["name"],
                weight=round(weights.get(sid, 0), 4),
                standalone_sharpe=strat_m["sharpe"],
                standalone_cagr=strat_m["cagr_pct"],
                standalone_vol=strat_m["vol_pct"],
                standalone_dd=strat_m["max_dd_pct"],
                marginal_sharpe_contribution=round(marginal, 3),
                sharpe_drag=round(marginal, 3) if marginal < 0 else 0.0,
                corr_to_portfolio=round(corr, 3),
                is_drag=is_drag,
            ))

        return sorted(attributions, key=lambda a: -a.marginal_sharpe_contribution)

    # ── Optimization 1: Volatility Targeting ──────────────────────────────

    def _opt_vol_targeting(self, baseline_sharpe: float) -> OptimizationResult:
        """Scale daily portfolio exposure to maintain constant vol.

        Insight: Sharpe = μ/σ. If we keep σ constant at target_vol, then
        high-vol periods get scaled down (reducing drawdowns) and low-vol
        periods get scaled up (capturing more return). Net effect: same μ
        on average but lower σ peaks → higher Sharpe.
        """
        weights = self._baseline_weights()
        raw_port = np.zeros(self.n)
        for sid in self.strategy_ids:
            raw_port += weights.get(sid, 0) * self.returns[sid].values

        target_vol = 0.12 / math.sqrt(TRADING_DAYS)  # 12% annualized target
        lookback = 20

        scaled_rets = np.zeros(self.n)
        for i in range(self.n):
            if i < lookback:
                # Not enough data yet — use leverage as-is
                scaled_rets[i] = raw_port[i] * self.leverage
            else:
                # Realized vol over lookback window
                recent = raw_port[max(0, i - lookback):i]
                realized_vol = recent.std()
                if realized_vol > 1e-12:
                    scale = target_vol / realized_vol
                    scale = max(0.3, min(3.0, scale))  # clamp
                else:
                    scale = self.leverage
                scaled_rets[i] = raw_port[i] * scale

        m = _compute_metrics(scaled_rets)
        yr_sharpe = _yearly_sharpe(scaled_rets, self.dates)
        eq = [100_000.0]
        for r in scaled_rets:
            eq.append(eq[-1] * (1 + r))

        baseline_m = _compute_metrics(self._portfolio_returns(weights))

        return OptimizationResult(
            name="Volatility Targeting",
            description="Scale exposure to maintain constant 12% annual vol. "
                        "Reduces vol clustering → smoother returns → higher Sharpe.",
            baseline_sharpe=baseline_sharpe,
            optimized_sharpe=m["sharpe"],
            sharpe_improvement=round(m["sharpe"] - baseline_sharpe, 2),
            baseline_cagr=baseline_m["cagr_pct"],
            optimized_cagr=m["cagr_pct"],
            baseline_dd=baseline_m["max_dd_pct"],
            optimized_dd=m["max_dd_pct"],
            optimized_sortino=m["sortino"],
            optimized_calmar=m["calmar"],
            mechanism="σ_target=12%, lookback=20d, scale=clamp(σ_target/σ_realized, 0.3, 3.0)",
            yearly_sharpe=yr_sharpe,
            daily_returns=scaled_rets,
            equity_curve=eq,
        )

    # ── Optimization 2: Regime Filtering ──────────────────────────────────

    def _opt_regime_filtering(self, baseline_sharpe: float) -> OptimizationResult:
        """Zero-weight strategies in their failing regimes.

        From regime_performance.py we know:
          - EXP-1220 FAILS in Crisis (negative CAGR)
          - SectorMom FAILS in HighVol + Crisis
          - TLT_IC FAILS in Bear + HighVol
          - VolTermStruct FAILS in HighVol + Recovery

        Replace failing strategy weight with CrossAsset (all-weather).
        """
        # Map regime_performance IDs to production_portfolio IDs
        failure_map = {
            "EXP-1220_DynLev": [Regime.CRISIS],
            "XLI_IronCondors": [Regime.HIGH_VOL, Regime.CRISIS],  # SectorMom proxy
            "TLT_IronCondors": [Regime.BEAR, Regime.HIGH_VOL],
            "VolTermStructure": [Regime.HIGH_VOL, Regime.RECOVERY],
            "CrossAsset_Pairs": [],  # all-weather
        }

        base_weights = self._baseline_weights()
        regime_labels = self.regimes.values

        filtered_rets = np.zeros(self.n)
        for i in range(self.n):
            regime = regime_labels[i] if i < len(regime_labels) else Regime.BULL
            day_weights = dict(base_weights)

            # Zero out failing strategies, redistribute to CrossAsset
            removed = 0.0
            for sid, fails in failure_map.items():
                if regime in fails and sid in day_weights:
                    removed += day_weights[sid]
                    day_weights[sid] = 0.0

            if removed > 0 and "CrossAsset_Pairs" in day_weights:
                day_weights["CrossAsset_Pairs"] += removed

            for sid in self.strategy_ids:
                filtered_rets[i] += day_weights.get(sid, 0) * self.returns[sid].values[i]

        filtered_rets *= self.leverage
        m = _compute_metrics(filtered_rets)
        yr_sharpe = _yearly_sharpe(filtered_rets, self.dates)
        eq = [100_000.0]
        for r in filtered_rets:
            eq.append(eq[-1] * (1 + r))

        baseline_m = _compute_metrics(self._portfolio_returns(base_weights))

        return OptimizationResult(
            name="Regime Filtering",
            description="Zero-weight strategies in their failing regimes. "
                        "EXP-1220→0 in Crisis, TLT→0 in Bear/HighVol, etc. "
                        "Redistribute to CrossAsset Pairs (all-weather).",
            baseline_sharpe=baseline_sharpe,
            optimized_sharpe=m["sharpe"],
            sharpe_improvement=round(m["sharpe"] - baseline_sharpe, 2),
            baseline_cagr=baseline_m["cagr_pct"],
            optimized_cagr=m["cagr_pct"],
            baseline_dd=baseline_m["max_dd_pct"],
            optimized_dd=m["max_dd_pct"],
            optimized_sortino=m["sortino"],
            optimized_calmar=m["calmar"],
            mechanism="If regime ∈ strategy.failing_regimes: weight=0, redirect to CrossAsset",
            yearly_sharpe=yr_sharpe,
            daily_returns=filtered_rets,
            equity_curve=eq,
        )

    # ── Optimization 3: Conviction Weighting ──────────────────────────────

    def _opt_conviction_weighting(self, baseline_sharpe: float) -> OptimizationResult:
        """Dynamically overweight strategies with high rolling Sharpe.

        Compute 60-day rolling Sharpe for each strategy.  Weight proportional
        to rolling Sharpe (Sharpe^2 for convexity).  Strategies with recent
        negative Sharpe get minimum weight.  This is a momentum-of-Sharpe signal.
        """
        lookback = 60
        min_weight = 0.05

        conv_rets = np.zeros(self.n)
        for i in range(self.n):
            if i < lookback:
                # Equal weight during warmup
                for sid in self.strategy_ids:
                    conv_rets[i] += (1.0 / len(self.strategy_ids)) * self.returns[sid].values[i]
            else:
                # Compute rolling Sharpe for each strategy
                rolling_sharpes = {}
                for sid in self.strategy_ids:
                    window = self.returns[sid].values[i - lookback:i]
                    mu = window.mean()
                    std = window.std()
                    rs = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
                    rolling_sharpes[sid] = max(rs, 0.01)  # floor

                # Weight by Sharpe^2 (convex: heavily favor high-Sharpe)
                sq = {sid: s ** 2 for sid, s in rolling_sharpes.items()}
                total = sum(sq.values())
                weights = {}
                for sid in self.strategy_ids:
                    w = sq[sid] / total if total > 0 else 1.0 / len(self.strategy_ids)
                    weights[sid] = max(min_weight, min(0.60, w))

                # Renormalize
                wsum = sum(weights.values())
                weights = {sid: w / wsum for sid, w in weights.items()}

                for sid in self.strategy_ids:
                    conv_rets[i] += weights[sid] * self.returns[sid].values[i]

        conv_rets *= self.leverage
        m = _compute_metrics(conv_rets)
        yr_sharpe = _yearly_sharpe(conv_rets, self.dates)
        eq = [100_000.0]
        for r in conv_rets:
            eq.append(eq[-1] * (1 + r))

        base_weights = self._baseline_weights()
        baseline_m = _compute_metrics(self._portfolio_returns(base_weights))

        return OptimizationResult(
            name="Conviction Weighting",
            description="Weight strategies by 60-day rolling Sharpe² (convex). "
                        "Momentum-of-quality: overweight what's working, fade what isn't.",
            baseline_sharpe=baseline_sharpe,
            optimized_sharpe=m["sharpe"],
            sharpe_improvement=round(m["sharpe"] - baseline_sharpe, 2),
            baseline_cagr=baseline_m["cagr_pct"],
            optimized_cagr=m["cagr_pct"],
            baseline_dd=baseline_m["max_dd_pct"],
            optimized_dd=m["max_dd_pct"],
            optimized_sortino=m["sortino"],
            optimized_calmar=m["calmar"],
            mechanism="w_i ∝ Sharpe_60d² (min 5%, max 60%, renormalized)",
            yearly_sharpe=yr_sharpe,
            daily_returns=conv_rets,
            equity_curve=eq,
        )


# ═══════════════════════════════════════════════════════════════════════════
# HTML helpers
# ═══════════════════════════════════════════════════════════════════════════


def _build_opt_details(optimizations: List[OptimizationResult], best_name: str) -> str:
    parts = []
    for o in optimizations:
        tag = " ★ BEST" if o.name == best_name else ""
        parts.append(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:10px 0">'
            f'<h3 style="margin:0 0 6px;font-size:0.95rem;color:#334155">{o.name}{tag}</h3>'
            f'<p style="color:#64748b;font-size:0.82rem;margin:0 0 8px">{o.description}</p>'
            f'<p style="font-size:0.85rem">Sharpe: {o.baseline_sharpe:.2f} &#8594; '
            f'<strong>{o.optimized_sharpe:.2f}</strong> ({o.sharpe_improvement:+.2f}) | '
            f'CAGR: {o.optimized_cagr:.1f}% | DD: {o.optimized_dd:.1f}% | '
            f'Sortino: {o.optimized_sortino:.1f}</p></div>'
        )
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════


def generate_report(
    result: SharpeAnalysisResult,
    output_path: str = "reports/sharpe_optimization.html",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Attribution table
    attr_rows = ""
    for a in result.attributions:
        mc = "#16a34a" if a.marginal_sharpe_contribution > 0 else "#dc2626"
        dc = "#dc2626" if a.is_drag else "#16a34a"
        attr_rows += f"""<tr>
          <td>{a.strategy_name}</td>
          <td>{a.weight:.0%}</td>
          <td>{a.standalone_sharpe:.2f}</td>
          <td>{a.standalone_cagr:.1f}%</td>
          <td>{a.standalone_vol:.1f}%</td>
          <td>{a.standalone_dd:.1f}%</td>
          <td style="color:{mc};font-weight:700">{a.marginal_sharpe_contribution:+.3f}</td>
          <td>{a.corr_to_portfolio:+.3f}</td>
          <td style="color:{dc};font-weight:700">{'DRAG' if a.is_drag else 'OK'}</td>
        </tr>"""

    # Optimization comparison table
    opt_rows = ""
    for o in result.optimizations:
        sc = "#16a34a" if o.sharpe_improvement > 0 else "#dc2626"
        is_best = o.name == result.best_optimization.name
        best_tag = ' style="background:#f0fdf4"' if is_best else ""
        star = " ★" if is_best else ""
        opt_rows += f"""<tr{best_tag}>
          <td>{o.name}{star}</td>
          <td>{o.baseline_sharpe:.2f}</td>
          <td style="font-weight:700">{o.optimized_sharpe:.2f}</td>
          <td style="color:{sc};font-weight:700">{o.sharpe_improvement:+.2f}</td>
          <td>{o.optimized_cagr:.1f}%</td>
          <td>{o.optimized_dd:.1f}%</td>
          <td>{o.optimized_sortino:.1f}</td>
          <td>{o.optimized_calmar:.1f}</td>
        </tr>"""

    # Per-year Sharpe for best optimization
    best = result.best_optimization
    yr_rows = ""
    for yr, s in sorted(best.yearly_sharpe.items()):
        sc = "#16a34a" if s >= 4 else ("#f59e0b" if s >= 2 else "#dc2626")
        yr_rows += f'<tr><td>{yr}</td><td style="color:{sc};font-weight:700">{s:.2f}</td></tr>'

    # Equity SVG for best optimization
    eq = best.equity_curve
    eq_svg = ""
    if len(eq) > 2:
        w, h = 780, 200
        pl, pr, pt, pb = 60, 20, 28, 25
        pw, ph = w - pl - pr, h - pt - pb
        n = len(eq)
        ym, yx = min(eq) * 0.95, max(eq) * 1.05
        step = max(1, n // 500)
        pts = [(i, eq[i]) for i in range(0, n, step)]
        if pts[-1][0] != n - 1:
            pts.append((n - 1, eq[-1]))

        def tx(i): return pl + i / max(n - 1, 1) * pw
        def ty(v): return pt + (1 - (v - ym) / max(yx - ym, 1)) * ph

        d = " ".join(f"{'M' if j == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
                     for j, (i, v) in enumerate(pts))
        eq_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="border:1px solid #e2e8f0;border-radius:6px;margin:0.5rem 0"><text x="{w//2}" y="16" text-anchor="middle" font-size="11" fill="#64748b">Optimized Equity ({best.name})</text><path d="{d}" fill="none" stroke="#16a34a" stroke-width="1.5"/></svg>'

    gap_pct = result.sharpe_gap / result.target_sharpe * 100
    closed_pct = (result.final_sharpe - result.portfolio_sharpe) / result.sharpe_gap * 100 if result.sharpe_gap > 0 else 0

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sharpe Optimization Analysis</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin:0; padding:24px; background:#fff; color:#1e293b; }}
h1 {{ font-size:1.5rem; color:#0f172a; margin-bottom:4px; }}
h2 {{ font-size:1.1rem; color:#334155; margin-top:2rem; border-bottom:1px solid #e2e8f0; padding-bottom:6px; }}
.meta {{ color:#64748b; font-size:0.85rem; margin-bottom:20px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:20px; }}
.card {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:14px; }}
.card-label {{ font-size:0.7rem; color:#64748b; text-transform:uppercase; }}
.card-value {{ font-size:1.3rem; font-weight:700; margin-top:3px; }}
table {{ width:100%; border-collapse:collapse; margin-bottom:14px; font-size:0.85rem; }}
th {{ background:#f1f5f9; padding:7px 10px; text-align:right; font-size:0.72rem; color:#64748b; text-transform:uppercase; border-bottom:2px solid #e2e8f0; }}
th:first-child {{ text-align:left; }}
td {{ padding:6px 10px; text-align:right; border-bottom:1px solid #f1f5f9; }}
td:first-child {{ text-align:left; }}
tr:hover {{ background:#f8fafc; }}
.positive {{ color:#16a34a; }} .negative {{ color:#dc2626; }} .warn {{ color:#d97706; }}
.gap-bar {{ background:#f1f5f9; border-radius:6px; height:32px; position:relative; margin:10px 0; }}
.gap-fill {{ background:linear-gradient(90deg,#dc2626,#f59e0b,#16a34a); border-radius:6px; height:100%; }}
.gap-label {{ position:absolute; top:7px; left:10px; font-size:0.8rem; font-weight:700; color:#fff; }}
</style></head><body>
<h1>Sharpe Ratio Optimization</h1>
<p class="meta">Current: {result.portfolio_sharpe:.2f} | Target: {result.target_sharpe:.1f} | Gap: {result.sharpe_gap:+.2f} | Best: {result.best_optimization.name} → {result.final_sharpe:.2f}</p>

<div class="grid">
  <div class="card"><div class="card-label">Current Sharpe</div><div class="card-value warn">{result.portfolio_sharpe:.2f}</div></div>
  <div class="card"><div class="card-label">Target</div><div class="card-value">{result.target_sharpe:.1f}</div></div>
  <div class="card"><div class="card-label">Gap</div><div class="card-value negative">{result.sharpe_gap:+.2f}</div></div>
  <div class="card"><div class="card-label">Optimized Sharpe</div><div class="card-value positive">{result.final_sharpe:.2f}</div></div>
  <div class="card"><div class="card-label">Gap Closed</div><div class="card-value positive">{closed_pct:.0f}%</div></div>
  <div class="card"><div class="card-label">Optimized CAGR</div><div class="card-value positive">{result.final_cagr:.1f}%</div></div>
  <div class="card"><div class="card-label">Optimized DD</div><div class="card-value {'positive' if result.final_dd < 12 else 'negative'}">{result.final_dd:.1f}%</div></div>
  <div class="card"><div class="card-label">Best Method</div><div class="card-value" style="font-size:1rem">{result.best_optimization.name}</div></div>
</div>

<div class="gap-bar">
  <div class="gap-fill" style="width:{min(100, result.final_sharpe / result.target_sharpe * 100):.0f}%"></div>
  <div class="gap-label">{result.portfolio_sharpe:.2f} → {result.final_sharpe:.2f} / {result.target_sharpe:.1f}</div>
</div>

<h2>Sharpe Attribution by Strategy</h2>
<p style="color:#64748b;font-size:0.82rem">Marginal contribution = change in portfolio Sharpe when removing strategy (leave-one-out).</p>
<table>
<tr><th>Strategy</th><th>Weight</th><th>Sharpe</th><th>CAGR</th><th>Vol</th><th>DD</th><th>Marginal ΔSharpe</th><th>ρ to Port</th><th>Status</th></tr>
{attr_rows}
</table>

<h2>Optimization Comparison</h2>
<table>
<tr><th>Optimization</th><th>Baseline</th><th>Optimized</th><th>ΔSharpe</th><th>CAGR</th><th>DD</th><th>Sortino</th><th>Calmar</th></tr>
{opt_rows}
</table>

<h2>Best Optimization: {best.name}</h2>
<p style="color:#64748b;font-size:0.82rem">{best.description}</p>
<p style="font-size:0.82rem"><strong>Mechanism:</strong> <code>{best.mechanism}</code></p>

{eq_svg}

<h2>Year-by-Year Sharpe ({best.name})</h2>
<table style="width:300px"><tr><th>Year</th><th>Sharpe</th></tr>{yr_rows}</table>

<h2>Optimization Details</h2>
{_build_opt_details(result.optimizations, best.name)}

<div style="color:#94a3b8;font-size:0.78rem;margin-top:3rem;border-top:1px solid #e2e8f0;padding-top:1rem">
<p>Sharpe Optimization — compass/sharpe_optimizer.py<br>
Attribution: leave-one-out marginal contribution.<br>
3 optimizations: vol targeting (12%), regime filtering, conviction weighting (Sharpe² 60d).</p>
</div></body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def run_analysis(seed: int = 42) -> SharpeAnalysisResult:
    print("Sharpe Optimization Analysis")
    print("=" * 60)

    analyzer = SharpeAnalyzer(seed=seed)
    result = analyzer.analyze()

    print(f"\n  Current Sharpe: {result.portfolio_sharpe:.2f}")
    print(f"  Target Sharpe:  {result.target_sharpe:.1f}")
    print(f"  Gap:            {result.sharpe_gap:+.2f}")

    print(f"\n  Attribution (marginal contribution):")
    for a in result.attributions:
        tag = "DRAG" if a.is_drag else "OK"
        print(f"    {a.strategy_name:35s}: Sharpe={a.standalone_sharpe:.2f}, "
              f"Marginal={a.marginal_sharpe_contribution:+.3f}, "
              f"ρ={a.corr_to_portfolio:+.3f} [{tag}]")

    print(f"\n  Optimizations:")
    for o in result.optimizations:
        star = " ★ BEST" if o.name == result.best_optimization.name else ""
        print(f"    {o.name:25s}: {o.baseline_sharpe:.2f} → {o.optimized_sharpe:.2f} "
              f"(Δ={o.sharpe_improvement:+.2f}, CAGR={o.optimized_cagr:.1f}%, "
              f"DD={o.optimized_dd:.1f}%){star}")

    print(f"\n  Final: Sharpe={result.final_sharpe:.2f}, "
          f"CAGR={result.final_cagr:.1f}%, DD={result.final_dd:.1f}%")

    report = generate_report(result)
    print(f"  Report: {report}")
    return result


if __name__ == "__main__":
    run_analysis()
