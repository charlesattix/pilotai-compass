"""
Regime Gate — CS entry filter based on market regime.

Prevents credit-spread entries during bear / high-volatility regimes.
Validated by walk-forward analysis (compass/benchmark_cs_only.md):

  Regime    Historical CS WR    Gate decision
  --------  ------------------  -----------------
  bull       85.6%  (n=208)     ALLOW  (full size)
  low_vol   100.0%  (n=6)       ALLOW  (full size)
  bear       64.7%  (n=17)      HALT   (below threshold)
  high_vol   50.0%  (n=2)       HALT   (below threshold)
  crash      <50%   (assumed)   HALT   (extreme risk)
  neutral    ~84%   (estimated) ALLOW  (reduced size)

Usage::

    from compass.regime_gate import RegimeGate
    from compass.regime import Regime

    gate = RegimeGate()
    if gate.should_trade(regime):
        scale = gate.position_scale(regime)
        contracts = base_contracts * scale

    # Or get a full decision with reasoning:
    decision = gate.evaluate(regime)
    print(decision.reason)          # "Bear regime: CS win rate 64.7% below 75.0% threshold"
    print(decision.position_scale)  # 0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Union

logger = logging.getLogger(__name__)


# ── Regime label constants (mirrors compass.regime.Regime enum values) ────────
# Stored as strings so this module works without importing compass/__init__.py
# (which pulls in requests, SQLAlchemy, etc).

BULL = "bull"
BEAR = "bear"
HIGH_VOL = "high_vol"
LOW_VOL = "low_vol"
CRASH = "crash"
NEUTRAL = "neutral"  # produced by ComboRegimeDetector

_ALL_KNOWN = frozenset({BULL, BEAR, HIGH_VOL, LOW_VOL, CRASH, NEUTRAL})

# Historical CS win rates from 2020-2025 walk-forward analysis
# Source: compass/benchmark_cs_only.md
HISTORICAL_WIN_RATES: Dict[str, Optional[float]] = {
    BULL:     0.856,
    LOW_VOL:  1.000,   # small sample (n=6); treat with caution
    NEUTRAL:  0.840,   # estimated from overall CS WR
    BEAR:     0.647,
    HIGH_VOL: 0.500,
    CRASH:    None,    # no historical data; assume worst-case
}

# Default thresholds
DEFAULT_MIN_WIN_RATE = 0.75   # halt if historical WR < this
DEFAULT_HALT_REGIMES = frozenset({BEAR, HIGH_VOL, CRASH})
DEFAULT_ALLOW_REGIMES = frozenset({BULL, LOW_VOL, NEUTRAL})

# Position scale by regime (1.0 = full size, 0.0 = no trade)
DEFAULT_POSITION_SCALES: Dict[str, float] = {
    BULL:     1.0,
    LOW_VOL:  1.0,
    NEUTRAL:  0.75,   # slightly reduced — lower confidence in neutral markets
    BEAR:     0.0,
    HIGH_VOL: 0.0,
    CRASH:    0.0,
}


# ── Gate decision ─────────────────────────────────────────────────────────────

@dataclass
class GateDecision:
    """Result of a regime gate evaluation.

    Attributes:
        should_trade:    True if the gate allows a new CS entry.
        regime:          Normalized regime string (lower-case).
        reason:          Human-readable explanation for the decision.
        position_scale:  Recommended contract scale 0.0–1.0.
                         0.0 means no trade; 1.0 means full size.
        historical_wr:   Historical CS win rate for this regime, or None
                         if no data is available.
        confidence:      Gate confidence 0.0–1.0.  Derived from sample
                         size and distance of historical WR from threshold.
                         Low confidence flags regimes with thin data.
    """

    should_trade: bool
    regime: str
    reason: str
    position_scale: float
    historical_wr: Optional[float] = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "should_trade": self.should_trade,
            "regime": self.regime,
            "reason": self.reason,
            "position_scale": round(self.position_scale, 2),
            "historical_wr": (
                round(self.historical_wr, 4) if self.historical_wr is not None else None
            ),
            "confidence": round(self.confidence, 3),
        }


# ── Main gate class ───────────────────────────────────────────────────────────

class RegimeGate:
    """Credit-spread entry gate based on market regime.

    Blocks new CS positions when the regime has historically poor win rates.
    All thresholds are configurable via the ``config`` dict.

    Args:
        config: Optional configuration overrides.  Supported keys:

            ``min_win_rate`` (float, default 0.75)
                Halt trading when historical CS win rate for the regime is
                below this value.  Used as a secondary check for regimes not
                explicitly listed in halt/allow sets.

            ``halt_regimes`` (list[str], default [bear, high_vol, crash])
                Regime names that always block CS entries regardless of
                win-rate threshold.

            ``allow_regimes`` (list[str], default [bull, low_vol, neutral])
                Regime names that always allow CS entries regardless of
                win-rate threshold.

            ``position_scales`` (dict[str, float])
                Override default position scales for specific regimes.
                Values must be between 0.0 and 1.0.

            ``unknown_regime_action`` (str, default "allow")
                What to do when the regime is unrecognised: "allow" or "halt".

    Example::

        gate = RegimeGate()
        decision = gate.evaluate("bear")
        # GateDecision(should_trade=False, regime='bear', reason='...', ...)

        # Custom — also halt neutral markets
        gate = RegimeGate({"halt_regimes": ["bear", "high_vol", "crash", "neutral"]})
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}

        self.min_win_rate: float = float(cfg.get("min_win_rate", DEFAULT_MIN_WIN_RATE))

        halt_list = cfg.get("halt_regimes", None)
        self.halt_regimes: Set[str] = (
            frozenset(str(r).lower() for r in halt_list)
            if halt_list is not None
            else DEFAULT_HALT_REGIMES
        )

        allow_list = cfg.get("allow_regimes", None)
        self.allow_regimes: Set[str] = (
            frozenset(str(r).lower() for r in allow_list)
            if allow_list is not None
            else DEFAULT_ALLOW_REGIMES
        )

        # Merge default scales with any user overrides
        user_scales = cfg.get("position_scales", {})
        self.position_scales: Dict[str, float] = {
            **DEFAULT_POSITION_SCALES,
            **{str(k).lower(): float(v) for k, v in user_scales.items()},
        }

        unknown_action = str(cfg.get("unknown_regime_action", "allow")).lower()
        if unknown_action not in ("allow", "halt"):
            raise ValueError(
                f"unknown_regime_action must be 'allow' or 'halt', got '{unknown_action}'"
            )
        self.unknown_regime_action: str = unknown_action

        logger.debug(
            "RegimeGate initialized: halt=%s allow=%s min_wr=%.2f",
            self.halt_regimes,
            self.allow_regimes,
            self.min_win_rate,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def should_trade(self, regime: Union[str, object]) -> bool:
        """Return True if a new CS entry is allowed in this regime.

        Args:
            regime: Regime string or ``compass.regime.Regime`` enum value.

        Returns:
            True  → proceed with CS entry
            False → halt, skip this signal
        """
        return self.evaluate(regime).should_trade

    def evaluate(self, regime: Union[str, object]) -> GateDecision:
        """Full gate evaluation with reasoning.

        Args:
            regime: Regime string or ``compass.regime.Regime`` enum value.

        Returns:
            :class:`GateDecision` with should_trade, reason, and sizing.
        """
        regime_str = self._normalize(regime)
        hist_wr = HISTORICAL_WIN_RATES.get(regime_str)
        scale = self.position_scales.get(regime_str, 0.0 if regime_str in self.halt_regimes else 1.0)
        confidence = self._confidence(regime_str, hist_wr)

        # 1. Explicit halt list (highest priority)
        if regime_str in self.halt_regimes:
            wr_str = f"{hist_wr:.1%}" if hist_wr is not None else "unknown"
            return GateDecision(
                should_trade=False,
                regime=regime_str,
                reason=(
                    f"{regime_str.replace('_', ' ').title()} regime: "
                    f"CS win rate {wr_str} — halting entries"
                ),
                position_scale=0.0,
                historical_wr=hist_wr,
                confidence=confidence,
            )

        # 2. Explicit allow list
        if regime_str in self.allow_regimes:
            wr_str = f"{hist_wr:.1%}" if hist_wr is not None else "estimated"
            return GateDecision(
                should_trade=True,
                regime=regime_str,
                reason=(
                    f"{regime_str.replace('_', ' ').title()} regime: "
                    f"CS win rate {wr_str} — entry allowed"
                ),
                position_scale=scale,
                historical_wr=hist_wr,
                confidence=confidence,
            )

        # 3. Unknown regime — apply win-rate threshold if data available
        if hist_wr is not None:
            if hist_wr < self.min_win_rate:
                return GateDecision(
                    should_trade=False,
                    regime=regime_str,
                    reason=(
                        f"Regime '{regime_str}': CS win rate {hist_wr:.1%} "
                        f"below {self.min_win_rate:.1%} threshold — halting"
                    ),
                    position_scale=0.0,
                    historical_wr=hist_wr,
                    confidence=confidence,
                )
            return GateDecision(
                should_trade=True,
                regime=regime_str,
                reason=(
                    f"Regime '{regime_str}': CS win rate {hist_wr:.1%} "
                    f"above {self.min_win_rate:.1%} threshold — entry allowed"
                ),
                position_scale=scale,
                historical_wr=hist_wr,
                confidence=confidence,
            )

        # 4. Unknown regime, no historical data
        if self.unknown_regime_action == "halt":
            return GateDecision(
                should_trade=False,
                regime=regime_str,
                reason=f"Unknown regime '{regime_str}': no historical data — halting (safe default)",
                position_scale=0.0,
                historical_wr=None,
                confidence=0.0,
            )
        return GateDecision(
            should_trade=True,
            regime=regime_str,
            reason=f"Unknown regime '{regime_str}': no historical data — allowing (configured default)",
            position_scale=scale,
            historical_wr=None,
            confidence=0.0,
        )

    def position_scale(self, regime: Union[str, object]) -> float:
        """Return the recommended position scale (0.0–1.0) for this regime.

        0.0 → do not trade
        0.75 → 75% of normal contract size
        1.0 → full size
        """
        return self.evaluate(regime).position_scale

    def regime_summary(self) -> dict:
        """Return a summary of gate decisions for all known regimes."""
        return {
            r: self.evaluate(r).to_dict()
            for r in sorted(_ALL_KNOWN)
        }

    # ── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(regime: Union[str, object]) -> str:
        """Coerce Regime enum or string to a lower-case string."""
        # Support compass.regime.Regime enum (its .value is already lower-case)
        val = getattr(regime, "value", regime)
        return str(val).lower().strip()

    def _confidence(self, regime_str: str, hist_wr: Optional[float]) -> float:
        """Compute gate confidence for a regime.

        Based on:
        - Sample size tier (bear/high_vol have thin data → lower confidence)
        - Distance of hist_wr from the min_win_rate threshold
        """
        # Sample-size confidence (from 2020-2025 CS trade counts)
        _sample_confidence = {
            BULL:     1.0,   # n=208
            LOW_VOL:  0.5,   # n=6  (thin)
            BEAR:     0.8,   # n=17
            HIGH_VOL: 0.3,   # n=2  (very thin)
            CRASH:    0.2,   # no CS data
            NEUTRAL:  0.6,   # estimated
        }
        sample_conf = _sample_confidence.get(regime_str, 0.0)

        if hist_wr is None:
            return round(sample_conf * 0.5, 3)  # penalize when no data

        # Distance confidence: how far from threshold?
        distance = abs(hist_wr - self.min_win_rate)
        distance_conf = min(1.0, distance / 0.20)  # saturates at 20pp distance

        # Sample size dominates (80%) — prevents thin data from appearing
        # more confident just because WR happens to be far from threshold.
        return round(0.8 * sample_conf + 0.2 * distance_conf, 3)
