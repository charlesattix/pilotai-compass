"""
compass/risk_decomposition.py — Portfolio risk decomposition engine.

Decomposes portfolio risk into factor contributions and provides risk budgeting
with VaR/CVaR estimation via historical simulation.

Provides:
  1. Factor risk decomposition: market (SPY beta), volatility (VIX exposure),
     regime, and idiosyncratic components per experiment
  2. Marginal risk contribution per experiment
  3. Risk budgeting framework (target allocations, breach detection)
  4. VaR and CVaR at 95% and 99% confidence via historical simulation
  5. HTML report generation with charts and tables

Usage:
    from compass.risk_decomposition import RiskDecomposer

    decomposer = RiskDecomposer(
        returns={"EXP-400": r400, "EXP-503": r503},
        spy_returns=spy_ret,
        vix_levels=vix,
        weights={"EXP-400": 0.6, "EXP-503": 0.4},
    )
    results = decomposer.run_all()
    html = decomposer.generate_html(results)
"""

import base64
import io
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PERIODS_PER_YEAR = 252
DEFAULT_CONFIDENCE_LEVELS = (0.95, 0.99)
REGIME_LABELS = ("BULL", "BEAR", "HIGH_VOL", "LOW_VOL", "CRASH")

# VIX thresholds for regime classification
VIX_HIGH_VOL_THRESHOLD = 25.0
VIX_CRASH_THRESHOLD = 35.0
VIX_LOW_VOL_THRESHOLD = 15.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FactorContribution:
    """Risk contribution from a single factor for one experiment."""
    experiment_id: str
    market: float  # fraction of variance explained by SPY beta
    volatility: float  # fraction explained by VIX exposure
    regime: float  # fraction explained by regime indicator
    idiosyncratic: float  # residual (unexplained)

    @property
    def total(self) -> float:
        return self.market + self.volatility + self.regime + self.idiosyncratic


@dataclass
class MarginalRisk:
    """Marginal risk contribution of one experiment to portfolio."""
    experiment_id: str
    marginal_vol: float  # dσ_p / dw_i (annualized)
    risk_contribution: float  # w_i * marginal_vol
    pct_contribution: float  # risk_contribution / σ_p


@dataclass
class RiskBudget:
    """Risk budgeting state for the portfolio."""
    experiment_id: str
    target_pct: float  # target risk allocation (0-1)
    actual_pct: float  # realized risk allocation
    deviation: float  # actual - target
    breach: bool  # |deviation| > tolerance


@dataclass
class VaRResult:
    """Value-at-Risk and Conditional VaR estimates."""
    confidence: float  # e.g. 0.95
    var: float  # VaR (loss, positive number = loss)
    cvar: float  # CVaR / Expected Shortfall
    horizon_days: int


@dataclass
class DecompositionResult:
    """Complete output of a risk decomposition run."""
    factor_contributions: List[FactorContribution]
    marginal_risks: List[MarginalRisk]
    risk_budgets: List[RiskBudget]
    var_results: List[VaRResult]
    portfolio_vol: float  # annualized portfolio volatility
    portfolio_vol_daily: float  # daily portfolio volatility
    summary: Dict[str, Any] = field(default_factory=dict)


# ── Core engine ───────────────────────────────────────────────────────────────

class RiskDecomposer:
    """Portfolio risk decomposition engine.

    Decomposes portfolio risk into factor contributions (market, volatility,
    regime, idiosyncratic), computes marginal risk per experiment, enforces
    risk budgets, and estimates VaR/CVaR via historical simulation.

    Args:
        returns: Dict mapping experiment ID to numpy array of daily returns.
        spy_returns: numpy array of SPY daily returns (same length as experiment returns).
        vix_levels: numpy array of VIX closing levels (same length).
        weights: Dict mapping experiment ID to portfolio weight (must sum to ~1.0).
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Trading days per year.
    """

    def __init__(
        self,
        returns: Dict[str, np.ndarray],
        spy_returns: np.ndarray,
        vix_levels: np.ndarray,
        weights: Dict[str, float],
        risk_free_rate: float = 0.045,
        periods_per_year: int = PERIODS_PER_YEAR,
    ):
        if not returns:
            raise ValueError("returns dict must contain at least one experiment")

        self.experiment_ids = sorted(returns.keys())
        self.n_experiments = len(self.experiment_ids)

        # Validate lengths
        arrays = [returns[eid] for eid in self.experiment_ids]
        lengths = {eid: len(returns[eid]) for eid in self.experiment_ids}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(f"All return arrays must have same length, got {lengths}")

        self.n_periods = list(unique_lengths)[0]
        if self.n_periods < 2:
            raise ValueError("Need at least 2 return periods for decomposition")

        if len(spy_returns) != self.n_periods:
            raise ValueError(
                f"spy_returns length ({len(spy_returns)}) != experiment returns length ({self.n_periods})"
            )
        if len(vix_levels) != self.n_periods:
            raise ValueError(
                f"vix_levels length ({len(vix_levels)}) != experiment returns length ({self.n_periods})"
            )

        # Store core data
        self.returns_matrix = np.column_stack(arrays)  # (T, N)
        self.spy_returns = np.asarray(spy_returns, dtype=float)
        self.vix_levels = np.asarray(vix_levels, dtype=float)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Weights
        self.weights = np.array([weights.get(eid, 0.0) for eid in self.experiment_ids])
        weight_sum = self.weights.sum()
        if abs(weight_sum - 1.0) > 0.05:
            logger.warning(
                "Weights sum to %.4f (expected ~1.0) — normalizing", weight_sum
            )
            self.weights = self.weights / weight_sum

        self.weights_dict = {
            eid: float(w) for eid, w in zip(self.experiment_ids, self.weights)
        }

        # Pre-compute portfolio returns
        self.portfolio_returns = self.returns_matrix @ self.weights  # (T,)

        # Pre-compute covariance
        self.cov_matrix = np.cov(self.returns_matrix, rowvar=False)
        if self.n_experiments == 1:
            self.cov_matrix = np.array([[float(self.cov_matrix)]])

        logger.info(
            "RiskDecomposer: %d experiments, %d periods, portfolio vol=%.4f",
            self.n_experiments, self.n_periods,
            np.std(self.portfolio_returns) * np.sqrt(self.periods_per_year),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Factor decomposition
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_regimes(self) -> np.ndarray:
        """Classify each period into a regime based on VIX and SPY trend.

        Returns integer array: 0=BULL, 1=BEAR, 2=HIGH_VOL, 3=LOW_VOL, 4=CRASH.
        """
        regimes = np.zeros(self.n_periods, dtype=int)

        for t in range(self.n_periods):
            vix = self.vix_levels[t]
            if vix >= VIX_CRASH_THRESHOLD:
                regimes[t] = 4  # CRASH
            elif vix >= VIX_HIGH_VOL_THRESHOLD:
                regimes[t] = 2  # HIGH_VOL
            elif vix <= VIX_LOW_VOL_THRESHOLD:
                regimes[t] = 3  # LOW_VOL
            else:
                # Use SPY trend: look-back 20d cumulative return
                lookback = min(t, 20)
                if lookback > 0:
                    cum_ret = np.prod(1 + self.spy_returns[t - lookback:t]) - 1
                    regimes[t] = 0 if cum_ret >= 0 else 1  # BULL / BEAR
                else:
                    regimes[t] = 0  # default BULL on first period

        return regimes

    def decompose_factors(self) -> List[FactorContribution]:
        """Decompose each experiment's risk into factor contributions.

        Uses OLS regression against:
          - Market factor: SPY returns
          - Volatility factor: VIX daily changes (normalized)
          - Regime factor: dummy variable encoding

        The R² of each factor (incremental) gives the fraction of variance
        explained. Residual = idiosyncratic risk.
        """
        vix_changes = np.diff(self.vix_levels, prepend=self.vix_levels[0])
        vix_std = np.std(vix_changes)
        if vix_std > 0:
            vix_norm = vix_changes / vix_std
        else:
            vix_norm = np.zeros_like(vix_changes)

        regimes = self._classify_regimes()

        # Build regime dummy (deviation from mean for each regime)
        regime_dummies = np.zeros((self.n_periods, len(REGIME_LABELS)))
        for i, _ in enumerate(REGIME_LABELS):
            regime_dummies[:, i] = (regimes == i).astype(float)
        # Drop one column to avoid multicollinearity (drop last)
        regime_dummies = regime_dummies[:, :-1]

        contributions = []
        for j, eid in enumerate(self.experiment_ids):
            y = self.returns_matrix[:, j]
            total_var = np.var(y)

            if total_var < 1e-16:
                contributions.append(FactorContribution(
                    experiment_id=eid,
                    market=0.0, volatility=0.0, regime=0.0, idiosyncratic=0.0,
                ))
                continue

            # 1. Market factor: regress y ~ SPY
            market_r2 = self._ols_r_squared(y, self.spy_returns.reshape(-1, 1))

            # 2. Volatility factor: regress residual ~ VIX changes
            market_resid = self._ols_residual(y, self.spy_returns.reshape(-1, 1))
            vol_var = np.var(market_resid)
            if vol_var > 1e-16:
                vol_r2_of_resid = self._ols_r_squared(
                    market_resid, vix_norm.reshape(-1, 1)
                )
                vol_r2 = vol_r2_of_resid * (1 - market_r2)
            else:
                vol_r2 = 0.0

            # 3. Regime factor: regress remaining residual ~ regime dummies
            market_vol_resid = self._ols_residual(
                market_resid, vix_norm.reshape(-1, 1)
            )
            mv_var = np.var(market_vol_resid)
            if mv_var > 1e-16 and regime_dummies.shape[1] > 0:
                regime_r2_of_resid = self._ols_r_squared(
                    market_vol_resid, regime_dummies
                )
                regime_r2 = regime_r2_of_resid * (1 - market_r2 - vol_r2)
            else:
                regime_r2 = 0.0

            # 4. Idiosyncratic = remainder
            idio = max(0.0, 1.0 - market_r2 - vol_r2 - regime_r2)

            # Clamp to valid range
            market_r2 = max(0.0, min(1.0, market_r2))
            vol_r2 = max(0.0, min(1.0, vol_r2))
            regime_r2 = max(0.0, min(1.0, regime_r2))

            contributions.append(FactorContribution(
                experiment_id=eid,
                market=round(market_r2, 6),
                volatility=round(vol_r2, 6),
                regime=round(regime_r2, 6),
                idiosyncratic=round(idio, 6),
            ))

        return contributions

    @staticmethod
    def _ols_r_squared(y: np.ndarray, X: np.ndarray) -> float:
        """Compute R² from OLS regression y ~ X (with intercept)."""
        n = len(y)
        X_aug = np.column_stack([np.ones(n), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        y_hat = X_aug @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot < 1e-16:
            return 0.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    @staticmethod
    def _ols_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute OLS residuals from y ~ X (with intercept)."""
        n = len(y)
        X_aug = np.column_stack([np.ones(n), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        except np.linalg.LinAlgError:
            return y.copy()
        return y - X_aug @ beta

    # ─────────────────────────────────────────────────────────────────────────
    # Marginal risk contribution
    # ─────────────────────────────────────────────────────────────────────────

    def compute_marginal_risk(self) -> List[MarginalRisk]:
        """Compute marginal risk contribution per experiment.

        Marginal volatility: ∂σ_p / ∂w_i = (Σw)_i / σ_p
        Risk contribution: RC_i = w_i × marginal_vol_i
        Sum of RC_i = σ_p (Euler decomposition).
        """
        cov_ann = self.cov_matrix * self.periods_per_year
        sigma_w = cov_ann @ self.weights  # (N,)
        port_vol = np.sqrt(self.weights @ sigma_w)

        if port_vol < 1e-12:
            return [
                MarginalRisk(
                    experiment_id=eid,
                    marginal_vol=0.0,
                    risk_contribution=0.0,
                    pct_contribution=1.0 / self.n_experiments,
                )
                for eid in self.experiment_ids
            ]

        marginal_vols = sigma_w / port_vol  # (N,)
        risk_contributions = self.weights * marginal_vols  # (N,)

        results = []
        for i, eid in enumerate(self.experiment_ids):
            results.append(MarginalRisk(
                experiment_id=eid,
                marginal_vol=round(float(marginal_vols[i]), 6),
                risk_contribution=round(float(risk_contributions[i]), 6),
                pct_contribution=round(float(risk_contributions[i] / port_vol), 6),
            ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Risk budgeting
    # ─────────────────────────────────────────────────────────────────────────

    def compute_risk_budgets(
        self,
        targets: Optional[Dict[str, float]] = None,
        tolerance: float = 0.10,
    ) -> List[RiskBudget]:
        """Compare actual risk allocation to target budget.

        Args:
            targets: Dict mapping experiment ID to target risk share (0-1).
                     If None, uses equal risk budget (1/N each).
            tolerance: Absolute deviation threshold for breach flagging.

        Returns:
            List of RiskBudget dataclasses.
        """
        if targets is None:
            targets = {eid: 1.0 / self.n_experiments for eid in self.experiment_ids}

        # Normalize targets
        target_sum = sum(targets.values())
        if abs(target_sum - 1.0) > 1e-6:
            targets = {k: v / target_sum for k, v in targets.items()}

        marginals = self.compute_marginal_risk()
        marginal_map = {m.experiment_id: m for m in marginals}

        budgets = []
        for eid in self.experiment_ids:
            target_pct = targets.get(eid, 1.0 / self.n_experiments)
            actual_pct = marginal_map[eid].pct_contribution
            deviation = actual_pct - target_pct
            budgets.append(RiskBudget(
                experiment_id=eid,
                target_pct=round(target_pct, 6),
                actual_pct=round(actual_pct, 6),
                deviation=round(deviation, 6),
                breach=abs(deviation) > tolerance,
            ))

        return budgets

    # ─────────────────────────────────────────────────────────────────────────
    # VaR and CVaR (historical simulation)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_var_cvar(
        self,
        confidence_levels: Tuple[float, ...] = DEFAULT_CONFIDENCE_LEVELS,
        horizon_days: int = 1,
    ) -> List[VaRResult]:
        """Compute VaR and CVaR using historical simulation.

        For multi-day horizons, uses overlapping windows of portfolio returns.
        VaR is reported as a positive loss number.

        Args:
            confidence_levels: Tuple of confidence levels (e.g., (0.95, 0.99)).
            horizon_days: Holding period in days.

        Returns:
            List of VaRResult for each confidence level.
        """
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")

        if horizon_days == 1:
            returns = self.portfolio_returns
        else:
            # Overlapping multi-day returns
            n = len(self.portfolio_returns)
            if n < horizon_days:
                raise ValueError(
                    f"Not enough data ({n} periods) for {horizon_days}-day horizon"
                )
            returns = np.array([
                np.prod(1 + self.portfolio_returns[i:i + horizon_days]) - 1
                for i in range(n - horizon_days + 1)
            ])

        results = []
        for cl in confidence_levels:
            alpha = 1.0 - cl
            # VaR: quantile of losses (negative returns → positive loss)
            var_value = -float(np.percentile(returns, alpha * 100))

            # CVaR: mean of returns below -VaR threshold
            tail = returns[returns <= -var_value]
            if len(tail) > 0:
                cvar_value = -float(np.mean(tail))
            else:
                cvar_value = var_value

            results.append(VaRResult(
                confidence=cl,
                var=round(var_value, 6),
                cvar=round(cvar_value, 6),
                horizon_days=horizon_days,
            ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Full decomposition pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def run_all(
        self,
        risk_targets: Optional[Dict[str, float]] = None,
        budget_tolerance: float = 0.10,
        var_horizons: Tuple[int, ...] = (1, 5),
    ) -> DecompositionResult:
        """Run the full risk decomposition pipeline.

        Args:
            risk_targets: Optional risk budget targets per experiment.
            budget_tolerance: Tolerance for risk budget breaches.
            var_horizons: Holding periods for VaR/CVaR computation.

        Returns:
            DecompositionResult with all decomposition outputs.
        """
        factors = self.decompose_factors()
        marginals = self.compute_marginal_risk()
        budgets = self.compute_risk_budgets(
            targets=risk_targets, tolerance=budget_tolerance
        )

        var_results = []
        for h in var_horizons:
            if h <= self.n_periods:
                var_results.extend(self.compute_var_cvar(horizon_days=h))

        # Portfolio-level volatility
        cov_ann = self.cov_matrix * self.periods_per_year
        port_vol_ann = float(np.sqrt(self.weights @ cov_ann @ self.weights))
        port_vol_daily = float(np.std(self.portfolio_returns))

        # Summary
        total_market = sum(f.market * f.total for f in factors) / max(len(factors), 1)
        total_vol = sum(f.volatility * f.total for f in factors) / max(len(factors), 1)
        total_regime = sum(f.regime * f.total for f in factors) / max(len(factors), 1)
        total_idio = sum(f.idiosyncratic * f.total for f in factors) / max(len(factors), 1)
        n_breaches = sum(1 for b in budgets if b.breach)

        summary = {
            "n_experiments": self.n_experiments,
            "n_periods": self.n_periods,
            "portfolio_vol_ann_pct": round(port_vol_ann * 100, 2),
            "portfolio_vol_daily_pct": round(port_vol_daily * 100, 4),
            "avg_factor_market_pct": round(total_market * 100, 2),
            "avg_factor_volatility_pct": round(total_vol * 100, 2),
            "avg_factor_regime_pct": round(total_regime * 100, 2),
            "avg_factor_idiosyncratic_pct": round(total_idio * 100, 2),
            "risk_budget_breaches": n_breaches,
            "dominant_factor": max(
                ["market", "volatility", "regime", "idiosyncratic"],
                key=lambda f: {"market": total_market, "volatility": total_vol,
                               "regime": total_regime, "idiosyncratic": total_idio}[f],
            ),
        }

        result = DecompositionResult(
            factor_contributions=factors,
            marginal_risks=marginals,
            risk_budgets=budgets,
            var_results=var_results,
            portfolio_vol=port_vol_ann,
            portfolio_vol_daily=port_vol_daily,
            summary=summary,
        )

        logger.info(
            "Risk decomposition complete: vol=%.2f%%, dominant=%s, breaches=%d",
            port_vol_ann * 100, summary["dominant_factor"], n_breaches,
        )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # HTML report generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_html(self, result: DecompositionResult) -> str:
        """Generate a self-contained HTML report with charts and tables.

        Includes:
          - Factor contribution stacked bar chart (SVG)
          - Marginal risk contribution bar chart (SVG)
          - VaR/CVaR summary table
          - Risk budget status table
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        s = result.summary

        # ── Factor contribution chart (SVG stacked bars) ─────────────────
        factor_chart = self._render_factor_chart(result.factor_contributions)

        # ── Marginal risk bar chart (SVG) ────────────────────────────────
        marginal_chart = self._render_marginal_chart(result.marginal_risks)

        # ── VaR/CVaR table ───────────────────────────────────────────────
        var_rows = ""
        for vr in result.var_results:
            var_rows += (
                f"<tr><td>{vr.confidence:.0%}</td>"
                f"<td>{vr.horizon_days}d</td>"
                f"<td class='bad'>{vr.var:.4%}</td>"
                f"<td class='bad'>{vr.cvar:.4%}</td></tr>\n"
            )

        # ── Risk budget table ────────────────────────────────────────────
        budget_rows = ""
        for rb in result.risk_budgets:
            status_cls = "bad" if rb.breach else "good"
            status_label = "BREACH" if rb.breach else "OK"
            budget_rows += (
                f"<tr><td>{rb.experiment_id}</td>"
                f"<td>{rb.target_pct:.1%}</td>"
                f"<td>{rb.actual_pct:.1%}</td>"
                f"<td>{rb.deviation:+.1%}</td>"
                f"<td class='{status_cls}'>{status_label}</td></tr>\n"
            )

        # ── Factor contribution table ────────────────────────────────────
        factor_rows = ""
        for fc in result.factor_contributions:
            factor_rows += (
                f"<tr><td>{fc.experiment_id}</td>"
                f"<td>{fc.market:.1%}</td>"
                f"<td>{fc.volatility:.1%}</td>"
                f"<td>{fc.regime:.1%}</td>"
                f"<td>{fc.idiosyncratic:.1%}</td></tr>\n"
            )

        # ── Marginal risk table ──────────────────────────────────────────
        marginal_rows = ""
        for mr in result.marginal_risks:
            marginal_rows += (
                f"<tr><td>{mr.experiment_id}</td>"
                f"<td>{mr.marginal_vol:.4f}</td>"
                f"<td>{mr.risk_contribution:.4f}</td>"
                f"<td>{mr.pct_contribution:.1%}</td></tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Portfolio Risk Decomposition</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 130px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  .good {{ color: #16a34a; }}
  .bad {{ color: #dc2626; }}
  .warn {{ color: #d97706; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  .chart-container {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
                      padding: 1.5em; margin: 1.5em 0; }}
  .section {{ margin-bottom: 2.5em; }}
  .legend {{ display: flex; gap: 1.5em; margin-top: 0.5em; font-size: 0.85em; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.3em; }}
  .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Portfolio Risk Decomposition</h1>
<div class="meta">Generated: {now} | {s['n_experiments']} experiments | {s['n_periods']} periods</div>

<div class="kpi-row">
  <div class="kpi">
    <div class="value">{s['portfolio_vol_ann_pct']:.1f}%</div>
    <div class="label">Annualized Vol</div>
  </div>
  <div class="kpi">
    <div class="value">{s['dominant_factor'].title()}</div>
    <div class="label">Dominant Factor</div>
  </div>
  <div class="kpi">
    <div class="value {'bad' if s['risk_budget_breaches'] > 0 else 'good'}">{s['risk_budget_breaches']}</div>
    <div class="label">Budget Breaches</div>
  </div>
  <div class="kpi">
    <div class="value">{s['avg_factor_market_pct']:.1f}%</div>
    <div class="label">Avg Market Factor</div>
  </div>
</div>

<div class="section">
<h2>Factor Risk Contributions</h2>
<div class="chart-container">
{factor_chart}
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#3b82f6"></div>Market</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f59e0b"></div>Volatility</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#8b5cf6"></div>Regime</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#94a3b8"></div>Idiosyncratic</div>
</div>
</div>
<table>
<thead><tr><th>Experiment</th><th>Market</th><th>Volatility</th><th>Regime</th><th>Idiosyncratic</th></tr></thead>
<tbody>{factor_rows}</tbody>
</table>
</div>

<div class="section">
<h2>Marginal Risk Contributions</h2>
<div class="chart-container">
{marginal_chart}
</div>
<table>
<thead><tr><th>Experiment</th><th>Marginal Vol</th><th>Risk Contribution</th><th>% of Portfolio</th></tr></thead>
<tbody>{marginal_rows}</tbody>
</table>
</div>

<div class="section">
<h2>Value-at-Risk &amp; CVaR</h2>
<table>
<thead><tr><th>Confidence</th><th>Horizon</th><th>VaR</th><th>CVaR (ES)</th></tr></thead>
<tbody>{var_rows}</tbody>
</table>
</div>

<div class="section">
<h2>Risk Budget Status</h2>
<table>
<thead><tr><th>Experiment</th><th>Target</th><th>Actual</th><th>Deviation</th><th>Status</th></tr></thead>
<tbody>{budget_rows}</tbody>
</table>
</div>

</body>
</html>"""
        return html

    # ─────────────────────────────────────────────────────────────────────────
    # SVG chart helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _render_factor_chart(contributions: List[FactorContribution]) -> str:
        """Render an SVG stacked horizontal bar chart of factor contributions."""
        if not contributions:
            return "<p>No factor data</p>"

        n = len(contributions)
        bar_h = 30
        gap = 10
        label_w = 100
        chart_w = 500
        h = n * (bar_h + gap) + gap
        colors = {"market": "#3b82f6", "volatility": "#f59e0b",
                  "regime": "#8b5cf6", "idiosyncratic": "#94a3b8"}

        bars = ""
        for i, fc in enumerate(contributions):
            y = gap + i * (bar_h + gap)
            # Label
            bars += (
                f'<text x="{label_w - 8}" y="{y + bar_h / 2 + 4}" '
                f'text-anchor="end" font-size="12" fill="#334155">{fc.experiment_id}</text>\n'
            )
            # Stacked segments
            x = label_w
            for factor, color in colors.items():
                val = getattr(fc, factor)
                w = val * chart_w
                if w > 0.5:
                    bars += (
                        f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" '
                        f'height="{bar_h}" fill="{color}" rx="2"/>\n'
                    )
                x += w

        total_w = label_w + chart_w + 20
        return (
            f'<svg width="{total_w}" height="{h}" '
            f'xmlns="http://www.w3.org/2000/svg">\n{bars}</svg>'
        )

    @staticmethod
    def _render_marginal_chart(marginals: List[MarginalRisk]) -> str:
        """Render an SVG bar chart of marginal risk contributions."""
        if not marginals:
            return "<p>No marginal risk data</p>"

        n = len(marginals)
        bar_w = 60
        gap = 20
        label_h = 50
        chart_h = 200
        w = n * (bar_w + gap) + gap + 60
        max_pct = max(m.pct_contribution for m in marginals) if marginals else 1.0
        max_pct = max(max_pct, 0.01)  # avoid division by zero

        bars = ""
        for i, mr in enumerate(marginals):
            x = 40 + gap + i * (bar_w + gap)
            h = (mr.pct_contribution / max_pct) * (chart_h - 20)
            y = chart_h - h
            bars += (
                f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" '
                f'height="{h:.1f}" fill="#3b82f6" rx="3"/>\n'
            )
            # Value label
            bars += (
                f'<text x="{x + bar_w / 2}" y="{y - 5}" text-anchor="middle" '
                f'font-size="11" fill="#334155">{mr.pct_contribution:.0%}</text>\n'
            )
            # X-axis label
            bars += (
                f'<text x="{x + bar_w / 2}" y="{chart_h + 18}" text-anchor="middle" '
                f'font-size="10" fill="#64748b">{mr.experiment_id}</text>\n'
            )

        return (
            f'<svg width="{w}" height="{chart_h + label_h}" '
            f'xmlns="http://www.w3.org/2000/svg">\n'
            f'<line x1="40" y1="0" x2="40" y2="{chart_h}" '
            f'stroke="#cbd5e1" stroke-width="1"/>\n'
            f'<line x1="40" y1="{chart_h}" x2="{w}" y2="{chart_h}" '
            f'stroke="#cbd5e1" stroke-width="1"/>\n'
            f'{bars}</svg>'
        )


# ── Convenience: generate and save report ─────────────────────────────────────

def generate_report(
    returns: Dict[str, np.ndarray],
    spy_returns: np.ndarray,
    vix_levels: np.ndarray,
    weights: Dict[str, float],
    output_path: str = "reports/risk_decomposition.html",
    risk_targets: Optional[Dict[str, float]] = None,
) -> DecompositionResult:
    """One-call convenience: decompose risk and write HTML report.

    Args:
        returns: Experiment returns dict.
        spy_returns: SPY daily returns.
        vix_levels: VIX closing levels.
        weights: Portfolio weights.
        output_path: Where to write the HTML report.
        risk_targets: Optional risk budget targets.

    Returns:
        DecompositionResult (report is also written to disk).
    """
    decomposer = RiskDecomposer(
        returns=returns,
        spy_returns=spy_returns,
        vix_levels=vix_levels,
        weights=weights,
    )
    result = decomposer.run_all(risk_targets=risk_targets)
    html = decomposer.generate_html(result)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info("Risk decomposition report written to %s", output_path)
    return result
