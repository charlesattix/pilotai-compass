"""
EXP-1890 — Portfolio Risk Manager

Production-grade risk engine for the multi-strategy portfolio. Wraps the
EXP-1850 risk_parity_regime_tilt allocation methodology with explicit safety
controls before any sizing decision is allowed to flow downstream to execution.

Five components (one class each):

  1. CrossStrategySizer
        Risk-parity (1/vol) sizing per regime. Optional Kelly cap per
        strategy from in-sample mean/variance. Always returns non-negative
        weights summing to ≤ 1.0.

  2. CorrelationMonitor
        Rolling pairwise correlation matrix on strategy returns. Raises
        CorrelationAlert when any off-diagonal pair exceeds the configured
        threshold (default 0.50) AND market is in a stress regime
        (HIGH_VOL / BEAR / CRASH).

  3. DrawdownCircuitBreaker
        Tracks portfolio equity high-water-mark. Three states:
            - NORMAL  (DD < soft_pct)
            - WARN    (soft_pct ≤ DD < hard_pct) → emergency deleverage
            - HALT    (DD ≥ hard_pct)            → flatten + lock
        Defaults: soft_pct=0.10, hard_pct=0.12 (per task spec).

  4. AllocationLimiter
        Caps each strategy at a max weight, enforces a min weight (or
        zeroes out below it), and triggers rebalance when drift from
        target exceeds rebalance_threshold.

  5. LeverageGovernor
        Maps the current Regime to a leverage multiplier in [1×, 3×],
        scaled from the EXP-1850 risk_parity_regime_tilt result and
        clamped by recent realized portfolio vol.

The classes are intentionally orthogonal — each is independently testable —
and `PortfolioRiskManager` composes them. The composition method
`make_decision()` returns a `RiskDecision` dataclass that the execution layer
treats as the single source of truth for sizing on a given day.

REAL DATA ONLY. No randomness. No hidden look-ahead — all decisions are
made strictly from the history available up to (and including) the
decision date.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from compass.regime import Regime


# ───────────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────────

TRADING_DAYS = 252

# Per-regime base leverage (EXP-1850 risk_parity_regime_tilt result), then
# rescaled to span [1×, 3×] as required by the task spec.
_RAW_REGIME_LEV = {
    Regime.BULL:     1.8,
    Regime.LOW_VOL:  1.6,
    Regime.HIGH_VOL: 1.0,
    Regime.BEAR:     0.7,
    Regime.CRASH:    0.5,
}


def _scale_to_range(values: Mapping[Regime, float],
                    lo: float, hi: float) -> Dict[Regime, float]:
    """Affine map mapping min(values)→lo, max(values)→hi."""
    vmin, vmax = min(values.values()), max(values.values())
    span = (vmax - vmin) or 1.0
    return {k: lo + (v - vmin) / span * (hi - lo) for k, v in values.items()}


REGIME_LEVERAGE: Dict[Regime, float] = _scale_to_range(_RAW_REGIME_LEV, 1.0, 3.0)
# bull→3.0, low_vol→2.54, high_vol→1.69, bear→1.31, crash→1.0


# Stress regimes — used by CorrelationMonitor + DrawdownCircuitBreaker
STRESS_REGIMES = frozenset({Regime.BEAR, Regime.HIGH_VOL, Regime.CRASH})


# ───────────────────────────────────────────────────────────────────────────
# Exceptions / state enums
# ───────────────────────────────────────────────────────────────────────────


class CorrelationAlert(RuntimeError):
    """Raised when pairwise correlations cross the stress threshold."""

    def __init__(self, pair: Tuple[str, str], rho: float, regime: Regime):
        self.pair = pair
        self.rho = rho
        self.regime = regime
        super().__init__(
            f"Correlation alert: {pair[0]}-{pair[1]} ρ={rho:.2f} "
            f"during {regime.value}"
        )


class CircuitState(str, Enum):
    NORMAL = "normal"
    WARN = "warn"      # soft trip — emergency deleverage
    HALT = "halt"      # hard trip — flatten + lock


# ───────────────────────────────────────────────────────────────────────────
# 1. CrossStrategySizer
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class CrossStrategySizer:
    """Risk-parity per-strategy sizing with optional Kelly clamp.

    Args:
        vol_lookback:    days of history used to estimate per-strategy vol.
        use_kelly:       if True, apply per-strategy Kelly cap (μ/σ²).
        kelly_fraction:  fraction of full Kelly to use (0 < f ≤ 1).
        kelly_cap:       absolute cap on any single Kelly weight.
    """

    vol_lookback: int = 60
    use_kelly: bool = False
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.50

    def size(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Return weight per strategy column. Always non-negative, sums ≤ 1.

        Args:
            returns: DataFrame of daily returns (columns=strategies).
        """
        if returns.empty or returns.shape[1] == 0:
            return {}

        recent = returns.tail(self.vol_lookback)
        if len(recent) < 5:
            # Insufficient history → equal-weight
            n = returns.shape[1]
            return {c: 1.0 / n for c in returns.columns}

        vols = recent.std(ddof=1).replace(0, np.nan)
        inv_vol = 1.0 / vols
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if inv_vol.sum() == 0:
            n = returns.shape[1]
            return {c: 1.0 / n for c in returns.columns}

        weights = inv_vol / inv_vol.sum()

        if self.use_kelly:
            mu = recent.mean()
            var = recent.var(ddof=1).replace(0, np.nan)
            kelly = (mu / var).clip(lower=0).fillna(0) * self.kelly_fraction
            kelly = kelly.clip(upper=self.kelly_cap)
            # Element-wise minimum: never above the Kelly cap
            weights = pd.concat([weights, kelly], axis=1).min(axis=1)
            total = weights.sum()
            if total > 0:
                weights = weights / max(total, 1.0)  # keep ≤ 1

        return {str(k): float(v) for k, v in weights.items()}


# ───────────────────────────────────────────────────────────────────────────
# 2. CorrelationMonitor
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class CorrelationMonitor:
    """Rolling pairwise correlation monitor with stress-regime gating.

    Args:
        window:        rolling window length in trading days.
        threshold:     pairwise |ρ| above this triggers an alert in stress.
        always_alert:  if True, alert regardless of regime (default False).
    """

    window: int = 30
    threshold: float = 0.50
    always_alert: bool = False

    def check(self,
              returns: pd.DataFrame,
              regime: Regime) -> List[Tuple[Tuple[str, str], float]]:
        """Return list of (pair, rho) violations. Empty list = clean."""
        if returns.shape[1] < 2 or len(returns) < max(5, self.window // 2):
            return []
        recent = returns.tail(self.window)
        corr = recent.corr().fillna(0.0)
        violations: List[Tuple[Tuple[str, str], float]] = []
        cols = list(corr.columns)
        gating = self.always_alert or (regime in STRESS_REGIMES)
        if not gating:
            return []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                rho = float(corr.iloc[i, j])
                if abs(rho) >= self.threshold:
                    violations.append(((cols[i], cols[j]), rho))
        return violations

    def check_or_raise(self, returns: pd.DataFrame, regime: Regime) -> None:
        v = self.check(returns, regime)
        if v:
            pair, rho = v[0]
            raise CorrelationAlert(pair, rho, regime)


# ───────────────────────────────────────────────────────────────────────────
# 3. DrawdownCircuitBreaker
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class DrawdownCircuitBreaker:
    """Equity-curve drawdown circuit with two trip points.

    Args:
        soft_pct: emergency-deleverage trip (default 0.10).
        hard_pct: flatten + lock trip       (default 0.12).
        deleverage_factor: leverage multiplier applied in WARN state.
    """

    soft_pct: float = 0.10
    hard_pct: float = 0.12
    deleverage_factor: float = 0.5

    _hwm: float = field(default=-np.inf, init=False)
    _state: CircuitState = field(default=CircuitState.NORMAL, init=False)

    def reset(self) -> None:
        self._hwm = -np.inf
        self._state = CircuitState.NORMAL

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def hwm(self) -> float:
        return self._hwm

    def update(self, equity: float) -> CircuitState:
        """Feed today's equity, return current state."""
        if equity > self._hwm:
            self._hwm = equity
        if self._hwm <= 0:
            return self._state

        dd = (self._hwm - equity) / self._hwm

        # HALT is sticky: once tripped it must be reset() to clear.
        if self._state == CircuitState.HALT:
            return self._state

        if dd >= self.hard_pct:
            self._state = CircuitState.HALT
        elif dd >= self.soft_pct:
            self._state = CircuitState.WARN
        else:
            self._state = CircuitState.NORMAL
        return self._state

    def leverage_multiplier(self) -> float:
        if self._state == CircuitState.HALT:
            return 0.0
        if self._state == CircuitState.WARN:
            return self.deleverage_factor
        return 1.0

    def current_dd(self, equity: float) -> float:
        if self._hwm <= 0 or equity <= 0:
            return 0.0
        return max(0.0, (self._hwm - equity) / self._hwm)


# ───────────────────────────────────────────────────────────────────────────
# 4. AllocationLimiter
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class AllocationLimiter:
    """Per-strategy weight caps + drift-based rebalance trigger.

    Args:
        max_per_strategy:    upper bound on any single weight.
        min_per_strategy:    lower bound; weights below it are zeroed.
        rebalance_threshold: |w_actual - w_target| above which we rebalance.
    """

    max_per_strategy: float = 0.40
    min_per_strategy: float = 0.05
    rebalance_threshold: float = 0.10

    def apply(self, weights: Mapping[str, float]) -> Dict[str, float]:
        """Clamp + zero-out + renormalise to sum ≤ 1."""
        if not weights:
            return {}
        out = {k: float(v) for k, v in weights.items()}
        # Drop tiny weights
        for k in list(out):
            if out[k] < self.min_per_strategy:
                out[k] = 0.0
        # Cap large weights
        for k in out:
            if out[k] > self.max_per_strategy:
                out[k] = self.max_per_strategy
        # Renormalise (only if total > 1; never inflate above 1)
        total = sum(out.values())
        if total > 1.0:
            out = {k: v / total for k, v in out.items()}
        return out

    def needs_rebalance(self,
                        target: Mapping[str, float],
                        actual: Mapping[str, float]) -> bool:
        keys = set(target) | set(actual)
        for k in keys:
            if abs(target.get(k, 0.0) - actual.get(k, 0.0)) >= self.rebalance_threshold:
                return True
        return False


# ───────────────────────────────────────────────────────────────────────────
# 5. LeverageGovernor
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class LeverageGovernor:
    """Regime-aware leverage in [1×, 3×] with realised-vol clamp.

    Args:
        regime_table:  Mapping[Regime, base leverage]. Defaults to the
                       EXP-1850 risk_parity_regime_tilt schedule rescaled
                       to [1×, 3×].
        target_vol:    annualised vol target used for the realised-vol clamp.
        min_lev:       absolute floor (default 1.0).
        max_lev:       absolute ceiling (default 3.0).
    """

    regime_table: Dict[Regime, float] = field(default_factory=lambda: dict(REGIME_LEVERAGE))
    target_vol: float = 0.10
    min_lev: float = 1.0
    max_lev: float = 3.0

    def base_leverage(self, regime: Regime) -> float:
        return float(self.regime_table.get(regime, 1.0))

    def leverage(self,
                 regime: Regime,
                 recent_returns: Optional[pd.Series] = None) -> float:
        lev = self.base_leverage(regime)
        if recent_returns is not None and len(recent_returns) >= 10:
            rv = float(recent_returns.std(ddof=1)) * math.sqrt(TRADING_DAYS)
            if rv > 1e-9:
                clamp = self.target_vol / rv
                lev = min(lev, max(self.min_lev, clamp * self.base_leverage(regime)))
        return float(np.clip(lev, self.min_lev, self.max_lev))


# ───────────────────────────────────────────────────────────────────────────
# Composition: PortfolioRiskManager
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class RiskDecision:
    """Single source of truth handed to the execution layer each day."""
    weights: Dict[str, float]
    leverage: float
    state: CircuitState
    regime: Regime
    correlation_alerts: List[Tuple[Tuple[str, str], float]] = field(default_factory=list)
    rebalance: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def total_exposure(self) -> float:
        return sum(self.weights.values()) * self.leverage

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "leverage": round(self.leverage, 4),
            "state": self.state.value,
            "regime": self.regime.value,
            "correlation_alerts": [
                {"pair": list(p), "rho": round(r, 4)} for p, r in self.correlation_alerts
            ],
            "rebalance": self.rebalance,
            "notes": self.notes,
            "total_exposure": round(self.total_exposure, 4),
        }


@dataclass
class PortfolioRiskManager:
    """Composes the five components and produces a daily RiskDecision."""

    sizer: CrossStrategySizer = field(default_factory=CrossStrategySizer)
    monitor: CorrelationMonitor = field(default_factory=CorrelationMonitor)
    breaker: DrawdownCircuitBreaker = field(default_factory=DrawdownCircuitBreaker)
    limiter: AllocationLimiter = field(default_factory=AllocationLimiter)
    governor: LeverageGovernor = field(default_factory=LeverageGovernor)

    def make_decision(self,
                      strategy_returns: pd.DataFrame,
                      regime: Regime,
                      equity: float,
                      current_weights: Optional[Mapping[str, float]] = None,
                      portfolio_returns: Optional[pd.Series] = None,
                      ) -> RiskDecision:
        """Run the full pipeline and return a RiskDecision."""
        notes: List[str] = []

        # Step 1: drawdown circuit
        state = self.breaker.update(equity)

        # Step 2: target weights from sizer + caps from limiter
        raw = self.sizer.size(strategy_returns)
        weights = self.limiter.apply(raw)

        # Step 3: correlation monitor (advisory — does not zero weights,
        # but if HALT is tripped we already flatten below).
        alerts = self.monitor.check(strategy_returns, regime)
        if alerts:
            notes.append(
                f"correlation alert: {len(alerts)} pair(s) ≥ "
                f"{self.monitor.threshold}"
            )

        # Step 4: leverage from governor + circuit deleverage
        portfolio_for_clamp = portfolio_returns
        if portfolio_for_clamp is None and not strategy_returns.empty:
            portfolio_for_clamp = strategy_returns.mean(axis=1).tail(60)
        lev = self.governor.leverage(regime, portfolio_for_clamp)
        lev *= self.breaker.leverage_multiplier()
        if state != CircuitState.NORMAL:
            notes.append(f"circuit {state.value}: leverage→{lev:.2f}")

        # Step 5: rebalance flag
        rebalance = False
        if current_weights is not None:
            rebalance = self.limiter.needs_rebalance(weights, current_weights)

        # If HALT, flatten everything
        if state == CircuitState.HALT:
            weights = {k: 0.0 for k in weights}
            lev = 0.0
            notes.append("HALT: portfolio flattened")

        return RiskDecision(
            weights=weights,
            leverage=lev,
            state=state,
            regime=regime,
            correlation_alerts=alerts,
            rebalance=rebalance,
            notes=notes,
        )


__all__ = [
    "CircuitState",
    "CorrelationAlert",
    "CorrelationMonitor",
    "CrossStrategySizer",
    "DrawdownCircuitBreaker",
    "AllocationLimiter",
    "LeverageGovernor",
    "PortfolioRiskManager",
    "RiskDecision",
    "REGIME_LEVERAGE",
    "STRESS_REGIMES",
]
