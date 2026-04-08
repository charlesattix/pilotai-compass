"""North Star portfolio deployment engine — orchestrates the 4-strategy
blend for paper trading with risk budgeting, signal combination,
rebalancing, and circuit breakers.

Strategies:
  1. ML-CS-860: ML-enhanced credit spreads (EXP-860)
  2. Regime-Lev: Regime leverage optimizer (EXP-840)
  3. Intraday-MR: Mean reversion z-score (EXP-1300)
  4. Combined-750: CS + vol blend (EXP-750)

Provides:
  1. Config generator for paper trading deployment
  2. Signal orchestrator combining 4 strategy signals
  3. Risk budget allocator (12% total DD → per-strategy budgets)
  4. Daily rebalancing with transaction cost optimization
  5. Circuit breakers (per-strategy DD, correlation spike, kill switch)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Strategy definitions ────────────────────────────────────────────────────
STRATEGIES = {
    "ML-CS-860": {"weight": 0.30, "dd_budget": 0.04, "style": "ml_credit_spread"},
    "Regime-Lev": {"weight": 0.25, "dd_budget": 0.03, "style": "regime_leverage"},
    "Intraday-MR": {"weight": 0.20, "dd_budget": 0.03, "style": "mean_reversion"},
    "Combined-750": {"weight": 0.25, "dd_budget": 0.02, "style": "cs_vol_blend"},
}

TOTAL_DD_BUDGET = 0.12
CORRELATION_HALT_THRESHOLD = 0.85


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class DeployConfig:
    """Paper trading deployment configuration."""
    strategies: Dict[str, Dict[str, Any]]
    total_dd_budget: float
    leverage: float
    rebalance_freq: str
    cost_bps: float
    kill_switch_dd: float
    correlation_halt: float
    starting_capital: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategies": self.strategies,
            "total_dd_budget": self.total_dd_budget,
            "leverage": self.leverage,
            "rebalance_freq": self.rebalance_freq,
            "cost_bps": self.cost_bps,
            "kill_switch_dd": self.kill_switch_dd,
            "correlation_halt": self.correlation_halt,
            "starting_capital": self.starting_capital,
        }


@dataclass
class StrategySignal:
    """Signal from one strategy."""
    strategy: str
    direction: str           # "bullish", "bearish", "neutral"
    confidence: float        # 0-1
    raw_signal: float        # -1..+1
    regime: str = ""


@dataclass
class CombinedSignal:
    """Orchestrated signal from all strategies."""
    direction: str
    confidence: float
    weighted_signal: float   # weight-adjusted -1..+1
    agreement: float         # fraction of strategies agreeing
    component_signals: Dict[str, float]
    recommended_action: str  # "enter_put", "enter_call", "hold", "reduce"


@dataclass
class RiskBudgetState:
    """Current risk budget allocation and utilisation."""
    strategy_budgets: Dict[str, float]      # strategy → DD budget
    strategy_dd: Dict[str, float]           # strategy → current DD
    strategy_utilisation: Dict[str, float]   # strategy → budget used %
    total_dd: float
    total_utilisation: float
    budget_remaining: float


@dataclass
class CircuitBreakerState:
    """Circuit breaker status."""
    is_halted: bool
    halted_strategies: List[str]
    reasons: List[str]
    correlation_alert: bool
    kill_switch: bool


@dataclass
class RebalanceAction:
    """Rebalancing recommendation."""
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    turnover: float
    cost_estimate: float
    should_rebalance: bool
    reason: str


@dataclass
class DeployerResult:
    """Complete deployer output for one cycle."""
    config: Optional[DeployConfig] = None
    combined_signal: Optional[CombinedSignal] = None
    risk_budget: Optional[RiskBudgetState] = None
    circuit_breakers: Optional[CircuitBreakerState] = None
    rebalance: Optional[RebalanceAction] = None
    timestamp: str = ""


# ── Config generator ────────────────────────────────────────────────────────
class ConfigGenerator:
    """Generates paper trading deployment configuration."""

    def generate(
        self,
        strategies: Optional[Dict[str, Dict]] = None,
        leverage: float = 2.0,
        starting_capital: float = 100_000.0,
        rebalance_freq: str = "daily",
        cost_bps: float = 10.0,
    ) -> DeployConfig:
        strats = strategies or dict(STRATEGIES)

        # Validate weights sum to 1
        total_w = sum(s.get("weight", 0) for s in strats.values())
        if abs(total_w - 1.0) > 0.01:
            # Renormalise
            for s in strats.values():
                s["weight"] = s.get("weight", 0) / total_w

        # Validate DD budgets sum ≤ total
        total_db = sum(s.get("dd_budget", 0) for s in strats.values())
        if total_db > TOTAL_DD_BUDGET:
            scale = TOTAL_DD_BUDGET / total_db
            for s in strats.values():
                s["dd_budget"] = s.get("dd_budget", 0) * scale

        return DeployConfig(
            strategies=strats,
            total_dd_budget=TOTAL_DD_BUDGET,
            leverage=leverage,
            rebalance_freq=rebalance_freq,
            cost_bps=cost_bps,
            kill_switch_dd=0.15,
            correlation_halt=CORRELATION_HALT_THRESHOLD,
            starting_capital=starting_capital,
        )

    def save_yaml(self, config: DeployConfig, path: str) -> Path:
        """Save config as JSON (YAML-like, no pyyaml dependency)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(config.to_dict(), indent=2))
        return p


# ── Signal orchestrator ────────────────────────────────────────────────────
class SignalOrchestrator:
    """Combines signals from multiple strategies into unified signal."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.50,
    ) -> None:
        self.weights = weights or {s: d["weight"] for s, d in STRATEGIES.items()}
        self.min_confidence = min_confidence

    def combine(self, signals: List[StrategySignal]) -> CombinedSignal:
        """Weight-average strategy signals into combined signal."""
        if not signals:
            return CombinedSignal("neutral", 0, 0, 0, {}, "hold")

        component = {}
        weighted_sum = 0.0
        conf_sum = 0.0
        for sig in signals:
            w = self.weights.get(sig.strategy, 1.0 / len(signals))
            component[sig.strategy] = sig.raw_signal
            weighted_sum += w * sig.raw_signal
            conf_sum += w * sig.confidence

        weighted_sum = float(np.clip(weighted_sum, -1, 1))

        # Agreement: fraction with same sign as combined
        combined_sign = 1 if weighted_sum > 0.05 else (-1 if weighted_sum < -0.05 else 0)
        if combined_sign != 0:
            agreeing = sum(1 for s in signals
                          if (s.raw_signal > 0.05 and combined_sign > 0) or
                             (s.raw_signal < -0.05 and combined_sign < 0))
            agreement = agreeing / len(signals)
        else:
            agreement = 0.0

        # Direction
        if weighted_sum > 0.10:
            direction = "bullish"
        elif weighted_sum < -0.10:
            direction = "bearish"
        else:
            direction = "neutral"

        # Action
        if direction == "bullish" and conf_sum >= self.min_confidence:
            action = "enter_put"
        elif direction == "bearish" and conf_sum >= self.min_confidence:
            action = "enter_call"
        elif abs(weighted_sum) < 0.05:
            action = "hold"
        else:
            action = "reduce"

        return CombinedSignal(
            direction=direction,
            confidence=round(conf_sum, 4),
            weighted_signal=round(weighted_sum, 4),
            agreement=round(agreement, 2),
            component_signals=component,
            recommended_action=action,
        )


# ── Risk budget allocator ──────────────────────────────────────────────────
class RiskBudgetAllocator:
    """Distributes and monitors the 12% DD budget across strategies."""

    def __init__(
        self,
        budgets: Optional[Dict[str, float]] = None,
        total_budget: float = TOTAL_DD_BUDGET,
    ) -> None:
        self.budgets = budgets or {s: d["dd_budget"] for s, d in STRATEGIES.items()}
        self.total_budget = total_budget

    def compute_state(
        self, strategy_dd: Dict[str, float],
    ) -> RiskBudgetState:
        """Compute current risk budget utilisation."""
        utilisation = {}
        for strat, budget in self.budgets.items():
            dd = strategy_dd.get(strat, 0.0)
            utilisation[strat] = dd / budget * 100 if budget > 0 else 0

        total_dd = sum(strategy_dd.get(s, 0) for s in self.budgets)
        total_util = total_dd / self.total_budget * 100 if self.total_budget > 0 else 0
        remaining = max(0, self.total_budget - total_dd)

        return RiskBudgetState(
            strategy_budgets=dict(self.budgets),
            strategy_dd=dict(strategy_dd),
            strategy_utilisation={k: round(v, 1) for k, v in utilisation.items()},
            total_dd=round(total_dd, 4),
            total_utilisation=round(total_util, 1),
            budget_remaining=round(remaining, 4),
        )

    def adjust_weights(
        self,
        base_weights: Dict[str, float],
        budget_state: RiskBudgetState,
    ) -> Dict[str, float]:
        """Reduce weight for strategies near their DD budget."""
        adjusted = {}
        for strat, weight in base_weights.items():
            util = budget_state.strategy_utilisation.get(strat, 0)
            if util >= 100:
                adjusted[strat] = weight * 0.10  # near-halt
            elif util >= 80:
                adjusted[strat] = weight * 0.50
            elif util >= 60:
                adjusted[strat] = weight * 0.75
            else:
                adjusted[strat] = weight

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted


# ── Rebalancing engine ──────────────────────────────────────────────────────
class RebalancingEngine:
    """Daily rebalancing with transaction cost optimization."""

    def __init__(
        self,
        tolerance: float = 0.03,
        cost_bps: float = 10.0,
        min_rebalance_interval: int = 1,
    ) -> None:
        self.tolerance = tolerance
        self.cost_bps = cost_bps
        self.min_interval = min_rebalance_interval
        self._last_rebalance: int = -999

    def check(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        day: int = 0,
    ) -> RebalanceAction:
        """Decide whether to rebalance."""
        all_keys = sorted(set(current_weights) | set(target_weights))
        old_arr = np.array([current_weights.get(k, 0) for k in all_keys])
        new_arr = np.array([target_weights.get(k, 0) for k in all_keys])

        turnover = float(np.sum(np.abs(new_arr - old_arr)) / 2)
        max_drift = float(np.max(np.abs(new_arr - old_arr)))
        cost = turnover * portfolio_value * self.cost_bps / 10_000

        should = max_drift > self.tolerance and (day - self._last_rebalance) >= self.min_interval

        if should:
            reason = f"max drift {max_drift:.1%} > {self.tolerance:.1%}"
            self._last_rebalance = day
        else:
            reason = "within tolerance" if max_drift <= self.tolerance else "too soon"

        return RebalanceAction(
            old_weights=current_weights,
            new_weights=target_weights if should else current_weights,
            turnover=round(turnover, 4),
            cost_estimate=round(cost, 2),
            should_rebalance=should,
            reason=reason,
        )


# ── Circuit breakers ────────────────────────────────────────────────────────
class CircuitBreakers:
    """Halt trading when risk limits are breached."""

    def __init__(
        self,
        kill_switch_dd: float = 0.15,
        correlation_threshold: float = CORRELATION_HALT_THRESHOLD,
    ) -> None:
        self.kill_dd = kill_switch_dd
        self.corr_threshold = correlation_threshold

    def check(
        self,
        budget_state: RiskBudgetState,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> CircuitBreakerState:
        """Check all circuit breakers."""
        halted: List[str] = []
        reasons: List[str] = []

        # Per-strategy DD breach
        for strat, util in budget_state.strategy_utilisation.items():
            if util >= 100:
                halted.append(strat)
                reasons.append(f"{strat}: DD budget exhausted ({util:.0f}%)")

        # Total DD kill switch
        kill = budget_state.total_dd >= self.kill_dd
        if kill:
            reasons.append(f"Kill switch: total DD {budget_state.total_dd:.1%} >= {self.kill_dd:.1%}")

        # Correlation spike
        corr_alert = False
        if correlation_matrix is not None and correlation_matrix.shape[0] > 1:
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            off_diag = correlation_matrix[mask]
            max_corr = float(off_diag.max()) if len(off_diag) > 0 else 0
            if max_corr > self.corr_threshold:
                corr_alert = True
                reasons.append(f"Correlation spike: max={max_corr:.2f} > {self.corr_threshold:.2f}")

        is_halted = kill or len(halted) > 0 or corr_alert

        return CircuitBreakerState(
            is_halted=is_halted,
            halted_strategies=halted,
            reasons=reasons,
            correlation_alert=corr_alert,
            kill_switch=kill,
        )


# ── Main deployer ───────────────────────────────────────────────────────────
class NorthStarDeployer:
    """Orchestrates the complete North Star portfolio deployment."""

    def __init__(
        self,
        leverage: float = 2.0,
        starting_capital: float = 100_000.0,
    ) -> None:
        self.config_gen = ConfigGenerator()
        self.orchestrator = SignalOrchestrator()
        self.risk_budget = RiskBudgetAllocator()
        self.rebalancer = RebalancingEngine()
        self.breakers = CircuitBreakers()
        self.leverage = leverage
        self.capital = starting_capital
        self._config: Optional[DeployConfig] = None

    def initialize(self) -> DeployConfig:
        """Generate and store deployment config."""
        self._config = self.config_gen.generate(
            leverage=self.leverage,
            starting_capital=self.capital,
        )
        return self._config

    def run_cycle(
        self,
        signals: List[StrategySignal],
        strategy_dd: Dict[str, float],
        current_weights: Dict[str, float],
        portfolio_value: float,
        correlation_matrix: Optional[np.ndarray] = None,
        day: int = 0,
    ) -> DeployerResult:
        """Run one deployment cycle (called daily)."""
        if self._config is None:
            self.initialize()

        # 1. Combine signals
        combined = self.orchestrator.combine(signals)

        # 2. Risk budget
        budget_state = self.risk_budget.compute_state(strategy_dd)

        # 3. Circuit breakers
        cb = self.breakers.check(budget_state, correlation_matrix)

        # 4. Adjust weights based on risk budget
        base_weights = {s: d["weight"] for s, d in STRATEGIES.items()}
        if cb.is_halted:
            # Reduce all to minimum
            target_weights = {s: 0.05 for s in base_weights}
            total = sum(target_weights.values())
            target_weights = {k: v / total for k, v in target_weights.items()}
        else:
            target_weights = self.risk_budget.adjust_weights(base_weights, budget_state)

        # 5. Rebalance decision
        rebalance = self.rebalancer.check(
            current_weights, target_weights, portfolio_value, day,
        )

        return DeployerResult(
            config=self._config,
            combined_signal=combined,
            risk_budget=budget_state,
            circuit_breakers=cb,
            rebalance=rebalance,
            timestamp=_now(),
        )


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
