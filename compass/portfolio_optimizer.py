"""
Cross-Experiment Portfolio Optimizer
=====================================
Allocates capital across live experiments (EXP-400, 401, 503, 600) using
mean-variance optimization with regime-adaptive tilts and event-driven scaling.

Optimization methods:
  max_sharpe            — maximize Sharpe ratio (tangency portfolio)
  risk_parity           — inverse-volatility weighting
  equal_risk_contribution — each experiment contributes equal portfolio risk
  min_variance          — minimize total portfolio variance

Regime tilts (from COMPASS macro regime via macro_db):
  BULL_MACRO   — tilt toward momentum experiments (EXP-503, EXP-400)
  NEUTRAL_MACRO — no tilt, use optimizer output directly
  BEAR_MACRO   — tilt toward defensive/hedged experiments (EXP-401, EXP-600)

Event scaling (from compass.events):
  Pre-FOMC/CPI/NFP windows reduce total allocation (not relative weights)
  using the composite scaling factor from the event gate.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Experiment metadata ─────────────────────────────────────────────────────
# Profile tags drive regime tilts. "momentum" experiments get upweighted in
# bull regimes; "defensive"/"hedged" experiments get upweighted in bear.

EXPERIMENT_PROFILES: Dict[str, Dict] = {
    "EXP-400": {
        "name": "Champion — Regime-adaptive CS+IC",
        "profile": "balanced",
        "momentum_affinity": 0.6,   # moderate momentum lean (regime-adaptive)
        "defensive_affinity": 0.4,
    },
    "EXP-401": {
        "name": "CS+Straddle/Strangle blend",
        "profile": "hedged",
        "momentum_affinity": 0.3,
        "defensive_affinity": 0.7,  # event strangles provide hedging
    },
    "EXP-503": {
        "name": "ML V2 Aggressive",
        "profile": "momentum",
        "momentum_affinity": 0.9,   # 1.5× bull sizing, 0.1× bear
        "defensive_affinity": 0.1,
    },
    "EXP-600": {
        "name": "Real Data Optimized (conservative)",
        "profile": "defensive",
        "momentum_affinity": 0.1,
        "defensive_affinity": 0.9,  # 2% risk, DTE=45, conservative
    },
}

EXPERIMENT_IDS = list(EXPERIMENT_PROFILES.keys())

# ── Regime tilt parameters ──────────────────────────────────────────────────
# How much to blend optimizer weights with regime-tilted weights.
# 0.0 = pure optimizer, 1.0 = pure regime tilt.
DEFAULT_REGIME_BLEND = 0.30

# Minimum weight any experiment can have (prevents zero allocation).
MIN_WEIGHT = 0.05

# ── Rebalance schedule ──────────────────────────────────────────────────────
REBALANCE_FREQUENCY_DAYS = 7  # weekly rebalance by default


@dataclass
class OptimizationResult:
    """Output of a portfolio optimization run."""
    weights: Dict[str, float]
    method: str
    regime: str
    event_scaling: float
    scaled_weights: Dict[str, float]  # weights after event scaling
    metrics: Dict[str, float] = field(default_factory=dict)
    next_rebalance: Optional[date] = None


class PortfolioOptimizer:
    """Cross-experiment portfolio optimizer.

    Args:
        returns: Dict mapping experiment ID to numpy array of periodic returns.
                 All arrays must have the same length.
        risk_free_rate: Annualized risk-free rate (default 4.5% for current env).
        regime_blend: How much to blend regime tilts vs optimizer output (0-1).
        min_weight: Floor weight per experiment.
        periods_per_year: Number of return periods per year (252 for daily, 52 weekly).
    """

    def __init__(
        self,
        returns: Dict[str, np.ndarray],
        risk_free_rate: float = 0.045,
        regime_blend: float = DEFAULT_REGIME_BLEND,
        min_weight: float = MIN_WEIGHT,
        periods_per_year: int = 252,
    ):
        self.experiment_ids = sorted(returns.keys())
        n = len(self.experiment_ids)
        if n == 0:
            raise ValueError("returns dict must contain at least one experiment")

        # Stack returns into (T, N) matrix
        arrays = [returns[eid] for eid in self.experiment_ids]
        lengths = [len(a) for a in arrays]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All return arrays must have same length, got {dict(zip(self.experiment_ids, lengths))}"
            )

        self.returns_matrix = np.column_stack(arrays)  # (T, N)
        self.n_assets = n
        self.risk_free_rate = risk_free_rate
        self.regime_blend = regime_blend
        self.min_weight = min_weight
        self.periods_per_year = periods_per_year

        # Pre-compute statistics
        self.mean_returns = self.returns_matrix.mean(axis=0)  # (N,)
        self.cov_matrix = np.cov(self.returns_matrix, rowvar=False)  # (N, N)

        # Handle single-asset edge case (cov returns scalar)
        if self.n_assets == 1:
            self.cov_matrix = np.array([[self.cov_matrix]])

        self.std_returns = np.sqrt(np.diag(self.cov_matrix))  # (N,)

        logger.info(
            "PortfolioOptimizer: %d experiments, %d periods, rf=%.3f",
            n, lengths[0], risk_free_rate,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Optimization methods
    # ─────────────────────────────────────────────────────────────────────────

    def max_sharpe(self) -> np.ndarray:
        """Maximize Sharpe ratio (analytical solution for long-only).

        Uses the closed-form tangency portfolio:
          w* = Σ⁻¹(μ - rf) / 1ᵀΣ⁻¹(μ - rf)
        then clips to [min_weight, 1] and renormalizes.
        """
        excess = self.mean_returns * self.periods_per_year - self.risk_free_rate
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix — falling back to equal weight")
            return self._equal_weight()

        raw = inv_cov @ excess
        # If all excess returns are negative, equal-weight is safer
        if raw.sum() <= 0:
            logger.warning("No positive excess returns — falling back to equal weight")
            return self._equal_weight()

        weights = raw / raw.sum()
        return self._enforce_constraints(weights)

    def risk_parity(self) -> np.ndarray:
        """Inverse-volatility weighting (simple risk parity)."""
        vols = self.std_returns * np.sqrt(self.periods_per_year)
        # Guard against zero vol
        vols = np.maximum(vols, 1e-8)
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return self._enforce_constraints(weights)

    def equal_risk_contribution(self, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
        """Equal risk contribution via damped fixed-point iteration.

        Each experiment contributes the same marginal risk to the portfolio:
          w_i * (Σw)_i = constant for all i.

        Uses a damped update (α=0.5) for stable convergence.
        """
        n = self.n_assets
        cov = self.cov_matrix * self.periods_per_year
        damping = 0.5

        # Initialize with risk parity (good starting point)
        w = self.risk_parity()

        for _ in range(max_iter):
            sigma_w = cov @ w
            port_vol = np.sqrt(w @ sigma_w)
            if port_vol < 1e-12:
                return self._equal_weight()

            # Risk contribution per asset
            rc = w * (sigma_w / port_vol)
            target_rc = port_vol / n

            # Damped multiplicative update
            ratio = target_rc / np.maximum(rc, 1e-12)
            w_new = w * (damping * ratio + (1.0 - damping))
            w_new = np.maximum(w_new, 1e-12)
            w_new = w_new / w_new.sum()
            w_new = self._enforce_constraints(w_new)

            if np.max(np.abs(w_new - w)) < tol:
                w = w_new
                break
            w = w_new

        return w

    def min_variance(self) -> np.ndarray:
        """Global minimum variance portfolio (analytical).

        w* = Σ⁻¹1 / 1ᵀΣ⁻¹1
        """
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix — falling back to equal weight")
            return self._equal_weight()

        ones = np.ones(self.n_assets)
        raw = inv_cov @ ones
        weights = raw / raw.sum()
        return self._enforce_constraints(weights)

    # ─────────────────────────────────────────────────────────────────────────
    # Regime-adaptive tilting
    # ─────────────────────────────────────────────────────────────────────────

    def apply_regime_tilt(
        self,
        weights: np.ndarray,
        regime: str,
    ) -> np.ndarray:
        """Blend optimizer weights with regime-tilted weights.

        Args:
            weights: Raw optimizer output (N,).
            regime: One of "BULL_MACRO", "NEUTRAL_MACRO", "BEAR_MACRO".

        Returns:
            Tilted weights (N,), summing to 1.0.
        """
        if regime == "NEUTRAL_MACRO" or self.regime_blend == 0.0:
            return weights

        # Build affinity vector based on regime
        affinity = np.zeros(self.n_assets)
        for i, eid in enumerate(self.experiment_ids):
            profile = EXPERIMENT_PROFILES.get(eid, {})
            if regime == "BULL_MACRO":
                affinity[i] = profile.get("momentum_affinity", 0.5)
            elif regime == "BEAR_MACRO":
                affinity[i] = profile.get("defensive_affinity", 0.5)
            else:
                affinity[i] = 0.5  # unknown regime → neutral

        # Normalize affinity to a weight vector
        affinity_weights = affinity / affinity.sum()

        # Blend: (1 - α) * optimizer + α * regime_tilt
        blended = (1.0 - self.regime_blend) * weights + self.regime_blend * affinity_weights
        return self._enforce_constraints(blended)

    # ─────────────────────────────────────────────────────────────────────────
    # Event-driven scaling
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_event_scaling(as_of: Optional[date] = None) -> float:
        """Get composite event scaling factor from COMPASS event gate.

        Returns a float in (0, 1] — multiply total allocation by this.
        1.0 = no event risk, 0.50 = day-of FOMC (maximum reduction).
        """
        from compass.events import get_upcoming_events, compute_composite_scaling

        events = get_upcoming_events(
            as_of=as_of,
            horizon_days=5,
        )
        return compute_composite_scaling(events)

    # ─────────────────────────────────────────────────────────────────────────
    # Full optimization pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def optimize(
        self,
        method: str = "max_sharpe",
        regime: Optional[str] = None,
        as_of: Optional[date] = None,
    ) -> OptimizationResult:
        """Run full optimization: method → regime tilt → event scaling.

        Args:
            method: One of "max_sharpe", "risk_parity", "equal_risk_contribution",
                    "min_variance".
            regime: Macro regime override. If None, fetches from macro_db.
            as_of: Date for event scaling. If None, uses today.

        Returns:
            OptimizationResult with weights, scaled weights, and metadata.
        """
        # 1. Run optimizer
        method_map = {
            "max_sharpe": self.max_sharpe,
            "risk_parity": self.risk_parity,
            "equal_risk_contribution": self.equal_risk_contribution,
            "min_variance": self.min_variance,
        }
        if method not in method_map:
            raise ValueError(f"Unknown method '{method}', choose from {list(method_map)}")

        raw_weights = method_map[method]()

        # 2. Get macro regime
        if regime is None:
            regime = self._fetch_macro_regime()

        # 3. Apply regime tilt
        tilted_weights = self.apply_regime_tilt(raw_weights, regime)

        # 4. Event scaling (reduces total allocation, not relative weights)
        event_scaling = self.get_event_scaling(as_of=as_of)
        scaled_weights = {
            eid: round(w * event_scaling, 6)
            for eid, w in zip(self.experiment_ids, tilted_weights)
        }

        # 5. Compute portfolio metrics
        w = tilted_weights
        ann_return = float(w @ self.mean_returns * self.periods_per_year)
        ann_vol = float(np.sqrt(w @ (self.cov_matrix * self.periods_per_year) @ w))
        sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

        weights_dict = {
            eid: round(float(w_i), 6)
            for eid, w_i in zip(self.experiment_ids, tilted_weights)
        }

        # 6. Rebalance schedule
        today = as_of or date.today()
        from datetime import timedelta
        next_rebalance = today + timedelta(days=REBALANCE_FREQUENCY_DAYS)
        # Skip weekends
        while next_rebalance.weekday() >= 5:
            next_rebalance += timedelta(days=1)

        result = OptimizationResult(
            weights=weights_dict,
            method=method,
            regime=regime,
            event_scaling=round(event_scaling, 4),
            scaled_weights=scaled_weights,
            metrics={
                "annual_return": round(ann_return, 6),
                "annual_volatility": round(ann_vol, 6),
                "sharpe_ratio": round(sharpe, 4),
                "max_weight": round(float(tilted_weights.max()), 4),
                "min_weight": round(float(tilted_weights.min()), 4),
            },
            next_rebalance=next_rebalance,
        )

        logger.info(
            "Optimized (%s): regime=%s event_scale=%.2f sharpe=%.3f weights=%s",
            method, regime, event_scaling, sharpe,
            {k: f"{v:.1%}" for k, v in weights_dict.items()},
        )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _equal_weight(self) -> np.ndarray:
        return np.full(self.n_assets, 1.0 / self.n_assets)

    def _enforce_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Clip to [min_weight, 1], enforce long-only, renormalize to sum=1.

        Uses iterative redistribution: pin assets at min_weight, redistribute
        remaining budget among free assets, repeat until stable.
        """
        n = len(weights)
        w = np.clip(weights, 0.0, None)  # long-only first
        if w.sum() == 0:
            return np.full(n, 1.0 / n)
        w = w / w.sum()

        # Iteratively fix floor violations (max n iterations)
        for _ in range(n):
            pinned = w < self.min_weight
            if not pinned.any():
                break
            # Pin violators at floor, redistribute rest proportionally
            budget = 1.0 - self.min_weight * pinned.sum()
            if budget <= 0:
                # All assets pinned — equal weight
                return np.full(n, 1.0 / n)
            free_sum = w[~pinned].sum()
            if free_sum <= 0:
                return np.full(n, 1.0 / n)
            w[pinned] = self.min_weight
            w[~pinned] = w[~pinned] / free_sum * budget

        return w

    @staticmethod
    def _fetch_macro_regime() -> str:
        """Read current macro regime from macro_db."""
        try:
            from compass.macro_db import (
                get_current_macro_score,
                MACRO_BULL_THRESHOLD,
                MACRO_BEAR_THRESHOLD,
            )
            score = get_current_macro_score()
            if score is None:
                return "NEUTRAL_MACRO"
            if score >= MACRO_BULL_THRESHOLD:
                return "BULL_MACRO"
            if score < MACRO_BEAR_THRESHOLD:
                return "BEAR_MACRO"
            return "NEUTRAL_MACRO"
        except Exception as e:
            logger.warning("Could not fetch macro regime: %s — defaulting to NEUTRAL", e)
            return "NEUTRAL_MACRO"
