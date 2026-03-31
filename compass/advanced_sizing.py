"""
compass/advanced_sizing.py — Regime-adaptive fractional Kelly with drawdown
and correlation scaling.

Extends compass/sizing.py with:
  1. Regime-adaptive Kelly fraction (bull/low_vol → 0.75, bear → 0.50,
     high_vol/crash → 0.25)
  2. Drawdown-reactive scaling: linearly reduces size when current DD
     exceeds 50% of max allowed DD, exits fully at 90%
  3. Correlation-aware sizing: reduces position size when portfolio
     pairwise correlation exceeds a threshold
  4. Per-experiment config overrides

Usage::

    from compass.advanced_sizing import AdvancedPositionSizer

    sizer = AdvancedPositionSizer(max_dd_pct=30.0)
    result = sizer.compute(
        win_prob=0.65, win_return=0.30, loss_return=1.00,
        regime="bull", current_dd_pct=8.0, portfolio_correlation=0.35,
    )
    print(result.position_fraction)  # fraction of portfolio to risk
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Regime Kelly fractions ───────────────────────────────────────────────

DEFAULT_REGIME_FRACTIONS: Dict[str, float] = {
    "bull": 0.75,
    "low_vol": 0.75,
    "bear": 0.50,
    "high_vol": 0.25,
    "crash": 0.25,
}


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class SizingConfig:
    """Per-experiment sizing configuration.

    Attributes:
        max_position_pct: Maximum single-position size as % of portfolio.
        max_dd_pct: Maximum tolerated drawdown (%). Scaling begins at 50%
            of this value, full exit at 90%.
        dd_scale_start: Fraction of max_dd_pct where scaling begins (0.50).
        dd_scale_exit: Fraction of max_dd_pct where size goes to zero (0.90).
        correlation_threshold: Portfolio correlation above this reduces size.
        correlation_penalty: Size multiplier applied per 0.1 correlation
            above the threshold (e.g. 0.15 means lose 15% per 0.1 excess).
        regime_fractions: Override regime → Kelly fraction mapping.
        base_kelly_fraction: Fallback Kelly fraction for unknown regimes.
    """
    max_position_pct: float = 10.0
    max_dd_pct: float = 30.0
    dd_scale_start: float = 0.50
    dd_scale_exit: float = 0.90
    correlation_threshold: float = 0.60
    correlation_penalty: float = 0.15
    regime_fractions: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REGIME_FRACTIONS))
    base_kelly_fraction: float = 0.50


@dataclass
class SizingResult:
    """Output of AdvancedPositionSizer.compute()."""
    position_fraction: float   # final fraction of portfolio to risk
    kelly_raw: float           # raw Kelly criterion value
    kelly_fraction: float      # regime-adapted Kelly fraction used
    kelly_adjusted: float      # kelly_raw * kelly_fraction
    dd_scale: float            # drawdown scaling factor (0-1)
    corr_scale: float          # correlation scaling factor (0-1)
    capped: bool               # True if max_position_pct was binding
    regime: str
    constraints: List[str] = field(default_factory=list)


# ── Kelly calculation ────────────────────────────────────────────────────


def kelly_criterion(win_prob: float, win_return: float, loss_return: float) -> float:
    """Compute the Kelly criterion fraction.

    Kelly% = (p * b - q) / b
    where p = win probability, q = 1-p, b = win/loss ratio.

    Args:
        win_prob: Probability of winning (0, 1).
        win_return: Expected gain per dollar risked if win (e.g. 0.30).
        loss_return: Expected loss per dollar risked if loss (positive, e.g. 1.00).

    Returns:
        Optimal fraction of bankroll to risk. Clamped to [0, 1].
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0
    if win_return <= 0 or loss_return <= 0:
        return 0.0

    p = win_prob
    q = 1.0 - p
    b = win_return / loss_return

    kelly = (p * b - q) / b
    return max(0.0, min(1.0, kelly))


# ── Drawdown scaling ─────────────────────────────────────────────────────


def drawdown_scale(
    current_dd_pct: float,
    max_dd_pct: float,
    scale_start: float = 0.50,
    scale_exit: float = 0.90,
) -> float:
    """Compute a drawdown-reactive scaling factor in [0, 1].

    - DD below scale_start * max_dd: scale = 1.0 (full size)
    - DD between start and exit: linearly decreases from 1.0 to 0.0
    - DD above scale_exit * max_dd: scale = 0.0 (no new positions)

    Args:
        current_dd_pct: Current drawdown as positive percentage (e.g. 8.0 for -8%).
        max_dd_pct: Maximum tolerated drawdown percentage.
        scale_start: Fraction of max_dd where scaling begins.
        scale_exit: Fraction of max_dd where scaling reaches zero.

    Returns:
        Float in [0, 1].
    """
    if max_dd_pct <= 0:
        return 1.0

    dd = abs(current_dd_pct)
    start_dd = max_dd_pct * scale_start
    exit_dd = max_dd_pct * scale_exit

    if dd <= start_dd:
        return 1.0
    if dd >= exit_dd:
        return 0.0

    # Linear interpolation
    return (exit_dd - dd) / (exit_dd - start_dd)


# ── Correlation scaling ──────────────────────────────────────────────────


def correlation_scale(
    portfolio_correlation: float,
    threshold: float = 0.60,
    penalty_per_0_1: float = 0.15,
) -> float:
    """Compute a correlation-aware scaling factor in [0, 1].

    When portfolio average pairwise correlation exceeds `threshold`,
    size is reduced by `penalty_per_0_1` for each 0.1 unit of excess.

    Args:
        portfolio_correlation: Average pairwise correlation (0-1).
        threshold: Correlation level above which penalty kicks in.
        penalty_per_0_1: Fraction of size lost per 0.1 excess correlation.

    Returns:
        Float in [0, 1].
    """
    if portfolio_correlation <= threshold:
        return 1.0

    excess = portfolio_correlation - threshold
    penalty = (excess / 0.1) * penalty_per_0_1
    return max(0.0, 1.0 - penalty)


# ── Advanced sizer ───────────────────────────────────────────────────────


class AdvancedPositionSizer:
    """Regime-adaptive fractional Kelly with drawdown and correlation scaling.

    Wraps the existing PositionSizer logic with three additional layers:
      1. Regime-adaptive Kelly fraction
      2. Drawdown-reactive scaling
      3. Correlation-aware sizing

    Args:
        config: SizingConfig instance, or None for defaults.
        experiment_overrides: Dict mapping experiment_id → SizingConfig
            for per-experiment tuning.
    """

    def __init__(
        self,
        config: Optional[SizingConfig] = None,
        experiment_overrides: Optional[Dict[str, SizingConfig]] = None,
    ) -> None:
        self.config = config or SizingConfig()
        self.experiment_overrides = experiment_overrides or {}

    def get_config(self, experiment_id: Optional[str] = None) -> SizingConfig:
        """Get the effective config for an experiment."""
        if experiment_id and experiment_id in self.experiment_overrides:
            return self.experiment_overrides[experiment_id]
        return self.config

    def compute(
        self,
        win_prob: float,
        win_return: float,
        loss_return: float,
        regime: str = "bull",
        current_dd_pct: float = 0.0,
        portfolio_correlation: float = 0.0,
        experiment_id: Optional[str] = None,
    ) -> SizingResult:
        """Compute the position size fraction.

        Args:
            win_prob: Probability of winning (0-1).
            win_return: Expected gain per dollar risked if win.
            loss_return: Expected loss per dollar risked if loss (positive).
            regime: Current market regime (bull/bear/high_vol/low_vol/crash).
            current_dd_pct: Current portfolio drawdown (positive, e.g. 8.0).
            portfolio_correlation: Average pairwise correlation of open positions.
            experiment_id: Optional experiment identifier for config lookup.

        Returns:
            SizingResult with the computed position fraction and diagnostics.
        """
        cfg = self.get_config(experiment_id)
        constraints: List[str] = []

        # 1. Raw Kelly
        kelly_raw = kelly_criterion(win_prob, win_return, loss_return)

        # 2. Regime-adaptive Kelly fraction
        regime_key = regime.lower().strip() if regime else "bull"
        kf = cfg.regime_fractions.get(regime_key, cfg.base_kelly_fraction)
        kelly_adj = kelly_raw * kf

        # 3. Drawdown-reactive scaling
        dd_sc = drawdown_scale(
            current_dd_pct, cfg.max_dd_pct,
            cfg.dd_scale_start, cfg.dd_scale_exit,
        )
        if dd_sc < 1.0:
            constraints.append(
                f"DD scale {dd_sc:.2f} (DD {current_dd_pct:.1f}% vs max {cfg.max_dd_pct:.0f}%)"
            )

        # 4. Correlation scaling
        corr_sc = correlation_scale(
            portfolio_correlation, cfg.correlation_threshold, cfg.correlation_penalty,
        )
        if corr_sc < 1.0:
            constraints.append(
                f"Corr scale {corr_sc:.2f} (corr {portfolio_correlation:.2f} > {cfg.correlation_threshold:.2f})"
            )

        # 5. Combine
        position_frac = kelly_adj * dd_sc * corr_sc

        # 6. Cap at max position size
        max_frac = cfg.max_position_pct / 100.0
        capped = position_frac > max_frac
        if capped:
            position_frac = max_frac
            constraints.append(f"Capped at {cfg.max_position_pct:.1f}%")

        position_frac = round(max(0.0, position_frac), 6)

        return SizingResult(
            position_fraction=position_frac,
            kelly_raw=round(kelly_raw, 6),
            kelly_fraction=kf,
            kelly_adjusted=round(kelly_adj, 6),
            dd_scale=round(dd_sc, 4),
            corr_scale=round(corr_sc, 4),
            capped=capped,
            regime=regime_key,
            constraints=constraints,
        )

    def compute_contracts(
        self,
        account_value: float,
        spread_width: float,
        credit_received: float,
        max_contracts: int = 10,
        **kwargs,
    ) -> int:
        """Compute the number of contracts to trade.

        Calls compute() with the provided kwargs, then converts the
        position fraction to a contract count.

        Args:
            account_value: Portfolio value in dollars.
            spread_width: Spread width in dollars (e.g. 5.0).
            credit_received: Net credit per spread in dollars.
            max_contracts: Hard cap on contract count.
            **kwargs: Passed to compute() (win_prob, regime, etc.)

        Returns:
            Integer contract count.
        """
        result = self.compute(**kwargs)
        dollar_risk = account_value * result.position_fraction

        max_loss_per = (spread_width - credit_received) * 100
        if max_loss_per <= 0:
            return 0

        contracts = int(dollar_risk // max_loss_per)
        return min(max(0, contracts), max_contracts)
