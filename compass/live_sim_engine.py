"""Live trading simulation engine — realistic fill simulation with slippage,
partial fills, queue priority, market impact, and latency modeling.

Provides far more accurate performance estimates than standard backtests
by modeling the execution realities of live options trading.

Components:
  1. Bid-ask spread dynamics (VIX-scaled, time-of-day varying)
  2. Queue position model (depth-based fill probability)
  3. Random latency simulation (10-500ms)
  4. Market impact (Kyle lambda: price moves against large orders)
  5. Partial fill simulation (fill rate × order size)
  6. Strategy re-simulation comparing realistic vs ideal fills
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class SpreadModel:
    """Bid-ask spread state."""
    mid_price: float
    bid: float
    ask: float
    spread_bps: float
    vix_component: float      # spread widening from vol
    time_component: float     # spread widening at open/close


@dataclass
class FillResult:
    """Result of a simulated order fill."""
    order_qty: int
    filled_qty: int
    fill_price: float
    slippage_bps: float
    market_impact_bps: float
    latency_ms: float
    is_partial: bool
    queue_position: int
    fill_rate: float           # filled / ordered


@dataclass
class SimulatedTrade:
    """One trade through the realistic simulation."""
    trade_id: int
    entry_fill: FillResult
    exit_fill: FillResult
    ideal_pnl: float          # what standard backtest says
    realistic_pnl: float      # after slippage, impact, partials
    degradation_pct: float    # (ideal - realistic) / |ideal|
    entry_spread: SpreadModel
    exit_spread: SpreadModel


@dataclass
class StrategySimResult:
    """Result of simulating one strategy through realistic engine."""
    strategy: str
    ideal_return_pct: float
    realistic_return_pct: float
    degradation_pct: float
    ideal_sharpe: float
    realistic_sharpe: float
    avg_slippage_bps: float
    avg_impact_bps: float
    avg_fill_rate: float
    partial_fill_pct: float   # % of orders that were partial
    avg_latency_ms: float
    n_trades: int


@dataclass
class LiveSimResult:
    """Complete simulation output."""
    strategy_results: List[StrategySimResult] = field(default_factory=list)
    trades: List[SimulatedTrade] = field(default_factory=list)
    avg_degradation_pct: float = 0.0
    worst_degradation: str = ""
    generated_at: str = ""


# ── Bid-ask spread model ───────────────────────────────────────────────────
class SpreadDynamics:
    """Models realistic bid-ask spread behavior."""

    def __init__(
        self,
        base_spread_bps: float = 8.0,
        vix_sensitivity: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.base = base_spread_bps
        self.vix_sens = vix_sensitivity
        self.rng = np.random.RandomState(seed)

    def compute(
        self, mid_price: float, vix: float = 18.0, hour: int = 12,
    ) -> SpreadModel:
        """Compute bid-ask spread for current conditions."""
        # VIX component: spread widens with volatility
        vix_mult = 1.0 + self.vix_sens * max(0, (vix - 15) / 15)

        # Time-of-day: wider at open (9:30-10) and close (3:30-4)
        if hour <= 10:
            time_mult = 1.4
        elif hour >= 15:
            time_mult = 1.3
        else:
            time_mult = 1.0

        # Random noise (±20%)
        noise = 1.0 + self.rng.uniform(-0.2, 0.2)

        spread_bps = self.base * vix_mult * time_mult * noise
        half_spread = mid_price * spread_bps / 10_000 / 2

        return SpreadModel(
            mid_price=round(mid_price, 4),
            bid=round(mid_price - half_spread, 4),
            ask=round(mid_price + half_spread, 4),
            spread_bps=round(spread_bps, 2),
            vix_component=round((vix_mult - 1) * self.base, 2),
            time_component=round((time_mult - 1) * self.base, 2),
        )


# ── Queue position model ───────────────────────────────────────────────────
class QueueModel:
    """Models queue position and fill probability for limit orders."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def estimate_position(self, depth: int = 100) -> int:
        """Estimate queue position (1 = front)."""
        return max(1, int(self.rng.uniform(1, depth * 0.7)))

    def fill_probability(
        self, queue_pos: int, depth: int = 100, time_in_queue_s: float = 30.0,
    ) -> float:
        """Probability of getting filled at this queue position.

        Front of queue → high probability. Back → low.
        Longer time in queue → higher probability.
        """
        pos_factor = max(0.1, 1.0 - queue_pos / max(depth, 1))
        time_factor = min(1.0, time_in_queue_s / 60.0)
        base_prob = pos_factor * 0.7 + time_factor * 0.3
        return float(np.clip(base_prob + self.rng.uniform(-0.1, 0.1), 0.05, 0.99))


# ── Latency model ──────────────────────────────────────────────────────────
class LatencyModel:
    """Simulates random network/processing latency."""

    def __init__(
        self, min_ms: float = 10.0, max_ms: float = 500.0, seed: int = 42,
    ) -> None:
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.rng = np.random.RandomState(seed)

    def sample(self) -> float:
        """Sample latency in milliseconds (log-normal distribution)."""
        # Log-normal: most fills are fast, occasional slow ones
        mu = math.log(50)  # median ~50ms
        sigma = 0.8
        latency = float(self.rng.lognormal(mu, sigma))
        return max(self.min_ms, min(self.max_ms, latency))


# ── Market impact model (Kyle lambda) ──────────────────────────────────────
class MarketImpactModel:
    """Kyle's lambda model: price impact proportional to sqrt(order_size / ADV)."""

    def __init__(
        self, kyle_lambda: float = 0.10, adv: int = 5000, seed: int = 42,
    ) -> None:
        self.lam = kyle_lambda
        self.adv = adv
        self.rng = np.random.RandomState(seed)

    def compute(self, order_qty: int, mid_price: float) -> float:
        """Compute market impact in bps.

        impact = λ × √(Q / ADV) × 10000
        """
        if order_qty <= 0 or self.adv <= 0:
            return 0.0
        participation = order_qty / self.adv
        impact = self.lam * math.sqrt(participation) * 10_000
        # Add noise (±30%)
        impact *= (1.0 + self.rng.uniform(-0.3, 0.3))
        return max(0.0, round(impact, 2))


# ── Partial fill model ─────────────────────────────────────────────────────
class PartialFillModel:
    """Models partial fills based on order size and liquidity."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def simulate(
        self, order_qty: int, fill_prob: float,
    ) -> Tuple[int, float]:
        """Simulate fill. Returns (filled_qty, fill_rate)."""
        if order_qty <= 0:
            return 0, 0.0

        # Each contract has independent fill probability
        if fill_prob >= 0.95:
            return order_qty, 1.0

        filled = sum(1 for _ in range(order_qty) if self.rng.rand() < fill_prob)
        filled = max(0, filled)
        rate = filled / order_qty if order_qty > 0 else 0.0
        return filled, round(rate, 4)


# ── Live simulation engine ─────────────────────────────────────────────────
class LiveSimEngine:
    """Realistic trade execution simulator."""

    def __init__(
        self,
        base_spread_bps: float = 8.0,
        kyle_lambda: float = 0.10,
        adv: int = 5000,
        seed: int = 42,
    ) -> None:
        self.spread = SpreadDynamics(base_spread_bps, seed=seed)
        self.queue = QueueModel(seed=seed + 1)
        self.latency = LatencyModel(seed=seed + 2)
        self.impact = MarketImpactModel(kyle_lambda, adv, seed=seed + 3)
        self.partial = PartialFillModel(seed=seed + 4)

    def simulate_fill(
        self,
        mid_price: float,
        order_qty: int,
        side: str = "sell",
        vix: float = 18.0,
        hour: int = 12,
        order_type: str = "limit",
    ) -> FillResult:
        """Simulate a single order fill with all friction components."""
        # Spread
        sm = self.spread.compute(mid_price, vix, hour)

        # Queue position
        queue_pos = self.queue.estimate_position()
        fill_prob = self.queue.fill_probability(queue_pos)

        # Market order → guaranteed fill but at ask/bid
        if order_type == "market":
            fill_prob = 1.0
            queue_pos = 1

        # Partial fill
        filled, fill_rate = self.partial.simulate(order_qty, fill_prob)
        is_partial = 0 < filled < order_qty

        # Slippage: half-spread for limit, full spread for market
        if order_type == "market":
            slippage_bps = sm.spread_bps
        else:
            slippage_bps = sm.spread_bps / 2

        # Market impact
        impact_bps = self.impact.compute(filled, mid_price)

        # Fill price
        total_friction = (slippage_bps + impact_bps) / 10_000
        if side == "sell":
            fill_price = mid_price * (1 - total_friction)
        else:
            fill_price = mid_price * (1 + total_friction)

        # Latency
        lat = self.latency.sample()

        return FillResult(
            order_qty=order_qty,
            filled_qty=filled,
            fill_price=round(fill_price, 4),
            slippage_bps=round(slippage_bps, 2),
            market_impact_bps=round(impact_bps, 2),
            latency_ms=round(lat, 1),
            is_partial=is_partial,
            queue_position=queue_pos,
            fill_rate=fill_rate,
        )

    def simulate_strategy(
        self,
        strategy_name: str,
        trade_returns: np.ndarray,
        trade_sizes: np.ndarray,
        mid_prices: np.ndarray,
        vix_series: np.ndarray,
        capital: float = 100_000.0,
    ) -> Tuple[StrategySimResult, List[SimulatedTrade]]:
        """Run a strategy through the realistic simulation.

        Parameters
        ----------
        trade_returns : per-trade ideal returns (fraction)
        trade_sizes : contracts per trade
        mid_prices : option mid price at entry/exit
        vix_series : VIX at each trade
        """
        n = min(len(trade_returns), len(trade_sizes), len(mid_prices), len(vix_series))
        if n == 0:
            return StrategySimResult(strategy_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), []

        ideal_pnls: List[float] = []
        real_pnls: List[float] = []
        trades: List[SimulatedTrade] = []
        slippages: List[float] = []
        impacts: List[float] = []
        fill_rates: List[float] = []
        latencies: List[float] = []
        n_partial = 0

        for i in range(n):
            qty = int(trade_sizes[i])
            mid = float(mid_prices[i])
            vix = float(vix_series[i])
            ideal_ret = float(trade_returns[i])

            # Entry fill
            entry = self.simulate_fill(mid, qty, "sell", vix, 10)
            # Exit fill
            exit_mid = mid * (1 + ideal_ret)
            exit_fill = self.simulate_fill(exit_mid, entry.filled_qty, "buy", vix, 14)

            # Ideal P&L
            ideal_pnl = ideal_ret * qty * mid * 100
            ideal_pnls.append(ideal_pnl)

            # Realistic P&L: account for slippage, impact, partials
            if entry.filled_qty > 0 and exit_fill.filled_qty > 0:
                actual_qty = min(entry.filled_qty, exit_fill.filled_qty)
                entry_cost = entry.fill_price * actual_qty * 100
                exit_proceeds = exit_fill.fill_price * actual_qty * 100
                real_pnl = exit_proceeds - entry_cost
                if ideal_ret > 0:  # selling premium → credit upfront
                    real_pnl = ideal_pnl * (actual_qty / qty) - (
                        (entry.slippage_bps + entry.market_impact_bps +
                         exit_fill.slippage_bps + exit_fill.market_impact_bps) / 10_000
                        * abs(ideal_pnl)
                    )
                else:
                    real_pnl = ideal_pnl * (actual_qty / qty) * (
                        1 - (entry.slippage_bps + exit_fill.slippage_bps) / 10_000
                    )
            else:
                real_pnl = 0.0

            real_pnls.append(real_pnl)

            deg = ((ideal_pnl - real_pnl) / abs(ideal_pnl) * 100) if abs(ideal_pnl) > 0.01 else 0

            entry_spread = self.spread.compute(mid, vix, 10)
            exit_spread = self.spread.compute(exit_mid, vix, 14)

            trades.append(SimulatedTrade(
                trade_id=i,
                entry_fill=entry, exit_fill=exit_fill,
                ideal_pnl=round(ideal_pnl, 2),
                realistic_pnl=round(real_pnl, 2),
                degradation_pct=round(deg, 2),
                entry_spread=entry_spread, exit_spread=exit_spread,
            ))

            slippages.append(entry.slippage_bps)
            impacts.append(entry.market_impact_bps)
            fill_rates.append(entry.fill_rate)
            latencies.append(entry.latency_ms)
            if entry.is_partial:
                n_partial += 1

        # Aggregate metrics
        ideal_total = sum(ideal_pnls)
        real_total = sum(real_pnls)
        ideal_ret_pct = ideal_total / capital * 100
        real_ret_pct = real_total / capital * 100
        deg_pct = ((ideal_total - real_total) / abs(ideal_total) * 100) if abs(ideal_total) > 0.01 else 0

        ideal_dr = np.array(ideal_pnls) / capital
        real_dr = np.array(real_pnls) / capital
        ideal_sh = float(ideal_dr.mean() / ideal_dr.std() * np.sqrt(n)) if ideal_dr.std() > 0 else 0
        real_sh = float(real_dr.mean() / real_dr.std() * np.sqrt(n)) if real_dr.std() > 0 else 0

        return StrategySimResult(
            strategy=strategy_name,
            ideal_return_pct=round(ideal_ret_pct, 2),
            realistic_return_pct=round(real_ret_pct, 2),
            degradation_pct=round(deg_pct, 2),
            ideal_sharpe=round(ideal_sh, 2),
            realistic_sharpe=round(real_sh, 2),
            avg_slippage_bps=round(float(np.mean(slippages)), 2),
            avg_impact_bps=round(float(np.mean(impacts)), 2),
            avg_fill_rate=round(float(np.mean(fill_rates)), 4),
            partial_fill_pct=round(n_partial / n * 100, 1),
            avg_latency_ms=round(float(np.mean(latencies)), 1),
            n_trades=n,
        ), trades

    def run_comparison(
        self,
        strategies: Dict[str, Dict[str, np.ndarray]],
        capital: float = 100_000.0,
    ) -> LiveSimResult:
        """Run multiple strategies and compare ideal vs realistic.

        strategies: name → {"returns", "sizes", "prices", "vix"}
        """
        all_results: List[StrategySimResult] = []
        all_trades: List[SimulatedTrade] = []

        for name, data in strategies.items():
            result, trades = self.simulate_strategy(
                name,
                data["returns"], data["sizes"],
                data["prices"], data["vix"],
                capital,
            )
            all_results.append(result)
            all_trades.extend(trades)

        avg_deg = float(np.mean([r.degradation_pct for r in all_results])) if all_results else 0
        worst = max(all_results, key=lambda r: r.degradation_pct).strategy if all_results else ""

        return LiveSimResult(
            strategy_results=all_results,
            trades=all_trades,
            avg_degradation_pct=round(avg_deg, 2),
            worst_degradation=worst,
            generated_at=_now(),
        )


# ── Synthetic data ──────────────────────────────────────────────────────────
def generate_strategy_data(
    n_trades: int = 100, seed: int = 42,
) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {
        "returns": rng.randn(n_trades) * 0.03 + 0.005,
        "sizes": np.maximum(rng.randint(1, 15, n_trades), 1),
        "prices": rng.uniform(1.5, 5.0, n_trades),
        "vix": np.maximum(15 + rng.randn(n_trades) * 5, 10),
    }


def generate_multi_strategy(
    n_strategies: int = 5, n_trades: int = 80, seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        f"EXP-{880 + i * 30}": generate_strategy_data(n_trades, seed + i)
        for i in range(n_strategies)
    }


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
