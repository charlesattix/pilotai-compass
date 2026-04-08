"""
Smart execution engine — TWAP, VWAP, and adaptive execution algorithms.

Features:
  - TWAP: equal time-weighted slices
  - VWAP: volume-profile-weighted slices
  - Adaptive: mid-price start with urgency-driven walk
  - Market impact model (Almgren-Chriss square-root)
  - Implementation shortfall tracking (vs arrival price)
  - Fill rate simulation with latency
  - Execution quality analytics

Usage::

    from compass.smart_execution import SmartExecutionEngine
    engine = SmartExecutionEngine()
    result = engine.execute(order, market_state)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "smart_execution.html"


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class Order:
    """Parent order to execute."""

    order_id: str
    side: str               # "buy" or "sell"
    total_qty: int
    limit_price: float      # max buy / min sell
    urgency: float = 0.5    # 0=patient, 1=aggressive
    max_slices: int = 10


@dataclass
class MarketState:
    """Current market snapshot."""

    bid: float
    ask: float
    mid: float = 0.0
    spread_bps: float = 0.0
    daily_volume: int = 1_000_000
    volatility: float = 0.01     # daily vol
    time_of_day_frac: float = 0.5  # 0=open, 1=close

    def __post_init__(self):
        if self.mid == 0:
            self.mid = (self.bid + self.ask) / 2
        if self.spread_bps == 0 and self.mid > 0:
            self.spread_bps = (self.ask - self.bid) / self.mid * 10_000


@dataclass
class ChildOrder:
    """One slice of the parent order."""

    slice_id: int
    qty: int
    price: float
    time_frac: float       # when in the execution window
    filled: bool = False
    fill_price: float = 0.0
    fill_qty: int = 0
    slippage_bps: float = 0.0


@dataclass
class ExecutionResult:
    """Result of executing one parent order."""

    order_id: str
    algorithm: str
    arrival_price: float       # mid at decision time
    avg_fill_price: float
    total_filled: int
    total_requested: int
    fill_rate: float
    implementation_shortfall_bps: float  # vs arrival price
    market_impact_bps: float
    total_cost_bps: float      # shortfall + spread
    n_slices: int
    child_orders: List[ChildOrder]
    latency_ms: float = 0.0


@dataclass
class BacktestResult:
    """Comparison across algorithms and order sizes."""

    scenarios: List[Dict[str, Any]]
    avg_cost_by_algo: Dict[str, float]
    avg_fill_rate_by_algo: Dict[str, float]
    best_algorithm: str
    savings_vs_naive_bps: float


# ── Volume profile ───────────────────────────────────────────────────────


def intraday_volume_profile(n_slices: int) -> np.ndarray:
    """Typical U-shaped intraday volume profile.

    High at open, dip mid-day, high at close.
    """
    x = np.linspace(0, 1, n_slices)
    # U-shape: high at 0 and 1, low at 0.5
    profile = 1.0 + 0.8 * (4 * (x - 0.5) ** 2)
    return profile / profile.sum()


# ── Market impact model ──────────────────────────────────────────────────


def temporary_impact(
    qty: int,
    daily_volume: int,
    volatility: float,
    eta: float = 0.1,
) -> float:
    """Almgren-Chriss temporary impact in bps.

    impact = eta * sigma * sqrt(qty / volume)
    """
    if daily_volume <= 0:
        return volatility * eta * 10_000
    participation = qty / daily_volume
    return eta * volatility * math.sqrt(participation) * 10_000


def permanent_impact(
    qty: int,
    daily_volume: int,
    volatility: float,
    gamma: float = 0.05,
) -> float:
    """Permanent price impact in bps.

    impact = gamma * sigma * (qty / volume)
    """
    if daily_volume <= 0:
        return 0.0
    participation = qty / daily_volume
    return gamma * volatility * participation * 10_000


# ── TWAP algorithm ───────────────────────────────────────────────────────


def twap_slices(order: Order, market: MarketState) -> List[ChildOrder]:
    """Time-weighted average price: equal qty per slice."""
    n = min(order.max_slices, max(order.total_qty, 1))
    base_qty = order.total_qty // n
    remainder = order.total_qty % n

    children = []
    for i in range(n):
        qty = base_qty + (1 if i < remainder else 0)
        if qty <= 0:
            continue
        time_frac = (i + 0.5) / n
        # Price: mid with small edge toward limit
        edge = market.spread_bps / 10_000 * market.mid * 0.3 * order.urgency
        if order.side == "buy":
            price = market.mid + edge * (i / n)  # walk up over time
        else:
            price = market.mid - edge * (i / n)

        children.append(ChildOrder(
            slice_id=i, qty=qty, price=round(price, 2), time_frac=time_frac,
        ))
    return children


# ── VWAP algorithm ───────────────────────────────────────────────────────


def vwap_slices(order: Order, market: MarketState) -> List[ChildOrder]:
    """Volume-weighted: more qty in high-volume periods."""
    n = min(order.max_slices, max(order.total_qty, 1))
    profile = intraday_volume_profile(n)
    raw_qtys = profile * order.total_qty
    qtys = np.round(raw_qtys).astype(int)

    # Fix rounding
    diff = order.total_qty - qtys.sum()
    for i in range(abs(diff)):
        idx = i % n
        qtys[idx] += 1 if diff > 0 else -1

    children = []
    for i in range(n):
        if qtys[i] <= 0:
            continue
        time_frac = (i + 0.5) / n
        edge = market.spread_bps / 10_000 * market.mid * 0.2 * order.urgency
        if order.side == "buy":
            price = market.mid + edge * profile[i]
        else:
            price = market.mid - edge * profile[i]

        children.append(ChildOrder(
            slice_id=i, qty=int(qtys[i]), price=round(price, 2), time_frac=time_frac,
        ))
    return children


# ── Adaptive algorithm ───────────────────────────────────────────────────


def adaptive_slices(order: Order, market: MarketState) -> List[ChildOrder]:
    """Start passive (mid), walk toward aggressive based on urgency + time."""
    n = min(order.max_slices, max(order.total_qty, 1))
    base_qty = order.total_qty // n
    remainder = order.total_qty % n

    half_spread = (market.ask - market.bid) / 2

    children = []
    for i in range(n):
        qty = base_qty + (1 if i < remainder else 0)
        if qty <= 0:
            continue
        time_frac = (i + 0.5) / n
        # Urgency ramp: start at mid, walk toward marketable as time passes
        progress = i / max(n - 1, 1)
        aggression = order.urgency * progress  # 0 → urgency over time

        if order.side == "buy":
            price = market.mid + half_spread * aggression
        else:
            price = market.mid - half_spread * aggression

        children.append(ChildOrder(
            slice_id=i, qty=qty, price=round(price, 2), time_frac=time_frac,
        ))
    return children


# ── Naive market order (baseline) ────────────────────────────────────────


def naive_slices(order: Order, market: MarketState) -> List[ChildOrder]:
    """Single market order — crosses full spread immediately."""
    price = market.ask if order.side == "buy" else market.bid
    return [ChildOrder(slice_id=0, qty=order.total_qty, price=price, time_frac=0.0)]


# ── Fill simulation ──────────────────────────────────────────────────────


def simulate_fills(
    children: List[ChildOrder],
    order: Order,
    market: MarketState,
    seed: int = 42,
) -> List[ChildOrder]:
    """Simulate fills with realistic noise, latency, and partial fills."""
    rng = np.random.RandomState(seed)
    half_spread = (market.ask - market.bid) / 2

    for child in children:
        # Fill probability: closer to market → higher fill rate
        if order.side == "buy":
            distance = (market.ask - child.price) / half_spread if half_spread > 0 else 0
        else:
            distance = (child.price - market.bid) / half_spread if half_spread > 0 else 0

        fill_prob = max(0.1, min(1.0, 1.0 - distance * 0.3))

        # Market impact on fill price
        impact = temporary_impact(child.qty, market.daily_volume, market.volatility)
        noise = rng.normal(0, market.volatility * market.mid * 0.1)

        if rng.random() < fill_prob:
            child.filled = True
            child.fill_qty = child.qty
            if order.side == "buy":
                child.fill_price = child.price + impact / 10_000 * market.mid + noise
            else:
                child.fill_price = child.price - impact / 10_000 * market.mid + noise
            child.fill_price = round(max(child.fill_price, 0.01), 4)
        else:
            # Partial fill
            child.filled = True
            child.fill_qty = max(1, int(child.qty * rng.uniform(0.3, 0.8)))
            child.fill_price = round(child.price + rng.normal(0, 0.01), 4)

        # Slippage
        if child.fill_qty > 0:
            child.slippage_bps = abs(child.fill_price - market.mid) / market.mid * 10_000

    return children


# ── Execution quality ────────────────────────────────────────────────────


def compute_execution_quality(
    children: List[ChildOrder],
    arrival_price: float,
    order: Order,
) -> ExecutionResult:
    """Compute execution quality metrics."""
    total_filled = sum(c.fill_qty for c in children)
    total_cost = sum(c.fill_price * c.fill_qty for c in children if c.fill_qty > 0)
    avg_price = total_cost / total_filled if total_filled > 0 else arrival_price
    fill_rate = total_filled / order.total_qty if order.total_qty > 0 else 0.0

    # Implementation shortfall
    if order.side == "buy":
        shortfall_bps = (avg_price - arrival_price) / arrival_price * 10_000
    else:
        shortfall_bps = (arrival_price - avg_price) / arrival_price * 10_000

    # Market impact estimate
    impact = sum(c.slippage_bps * c.fill_qty for c in children) / max(total_filled, 1)

    return ExecutionResult(
        order_id=order.order_id,
        algorithm="",
        arrival_price=arrival_price,
        avg_fill_price=avg_price,
        total_filled=total_filled,
        total_requested=order.total_qty,
        fill_rate=fill_rate,
        implementation_shortfall_bps=shortfall_bps,
        market_impact_bps=impact,
        total_cost_bps=shortfall_bps + impact * 0.5,
        n_slices=len(children),
        child_orders=children,
    )


# ── Core engine ──────────────────────────────────────────────────────────


ALGORITHMS = {
    "twap": twap_slices,
    "vwap": vwap_slices,
    "adaptive": adaptive_slices,
    "naive": naive_slices,
}


class SmartExecutionEngine:
    """Smart order execution engine."""

    def __init__(self, default_algo: str = "adaptive"):
        if default_algo not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {default_algo}")
        self.default_algo = default_algo

    def execute(
        self,
        order: Order,
        market: MarketState,
        algorithm: Optional[str] = None,
        seed: int = 42,
    ) -> ExecutionResult:
        """Execute an order using the specified algorithm."""
        algo = algorithm or self.default_algo
        slicer = ALGORITHMS[algo]
        children = slicer(order, market)
        children = simulate_fills(children, order, market, seed)
        result = compute_execution_quality(children, market.mid, order)
        result.algorithm = algo
        return result

    def compare_algorithms(
        self,
        order: Order,
        market: MarketState,
        seed: int = 42,
    ) -> Dict[str, ExecutionResult]:
        """Run all algorithms on the same order for comparison."""
        results = {}
        for name in ALGORITHMS:
            results[name] = self.execute(order, market, name, seed)
        return results

    def backtest(
        self,
        order_sizes: List[int],
        volatility_regimes: List[float],
        n_trials: int = 50,
        seed: int = 42,
    ) -> BacktestResult:
        """Backtest across order sizes and vol regimes."""
        rng = np.random.RandomState(seed)
        scenarios: List[Dict[str, Any]] = []

        for qty in order_sizes:
            for vol in volatility_regimes:
                for trial in range(n_trials):
                    mid = 5.0 + rng.uniform(-0.5, 0.5)
                    spread = mid * vol * 0.5
                    market = MarketState(
                        bid=mid - spread / 2,
                        ask=mid + spread / 2,
                        daily_volume=rng.randint(500_000, 3_000_000),
                        volatility=vol,
                    )
                    order = Order(
                        order_id=f"BT-{qty}-{trial}",
                        side=rng.choice(["buy", "sell"]),
                        total_qty=qty,
                        limit_price=market.ask * 1.01 if rng.random() > 0.5 else market.bid * 0.99,
                        urgency=rng.uniform(0.2, 0.8),
                    )

                    for algo in ALGORITHMS:
                        result = self.execute(order, market, algo, seed + trial)
                        scenarios.append({
                            "qty": qty, "vol": vol, "trial": trial,
                            "algo": algo,
                            "shortfall_bps": result.implementation_shortfall_bps,
                            "impact_bps": result.market_impact_bps,
                            "total_cost_bps": result.total_cost_bps,
                            "fill_rate": result.fill_rate,
                        })

        df = pd.DataFrame(scenarios)
        avg_cost = df.groupby("algo")["total_cost_bps"].mean().to_dict()
        avg_fill = df.groupby("algo")["fill_rate"].mean().to_dict()

        best = min(avg_cost, key=lambda k: avg_cost[k] if k != "naive" else float("inf"))
        naive_cost = avg_cost.get("naive", 0)
        savings = naive_cost - avg_cost.get(best, 0)

        return BacktestResult(
            scenarios=scenarios,
            avg_cost_by_algo=avg_cost,
            avg_fill_rate_by_algo=avg_fill,
            best_algorithm=best,
            savings_vs_naive_bps=savings,
        )

    @staticmethod
    def generate_report(result: BacktestResult, output_path: Path = DEFAULT_OUTPUT) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(result), encoding="utf-8")
        return output_path


# ── HTML ─────────────────────────────────────────────────────────────────


def _fr(v): return f"{v:.2f}"
def _fp(v): return f"{v:.1f}%"


def _build_html(r: BacktestResult) -> str:
    algo_rows = "".join(
        f"<tr><td style='text-align:left'>{a}</td><td>{_fr(r.avg_cost_by_algo[a])} bps</td>"
        f"<td>{_fp(r.avg_fill_rate_by_algo[a]*100)}</td></tr>"
        for a in ["naive", "twap", "vwap", "adaptive"]
        if a in r.avg_cost_by_algo
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Smart Execution</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2{{color:#58a6ff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;text-align:center}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.2em}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}th{{color:#8b949e;background:#161b22}}
</style></head><body>
<h1>Smart Execution Engine</h1>
<div class="cards">
<div class="c"><div class="l">Best Algorithm</div><div class="v">{r.best_algorithm}</div></div>
<div class="c"><div class="l">Savings vs Naive</div><div class="v">{_fr(r.savings_vs_naive_bps)} bps</div></div>
<div class="c"><div class="l">Scenarios Tested</div><div class="v">{len(r.scenarios)}</div></div>
</div>
<h2>Algorithm Comparison</h2>
<table><tr><th style="text-align:left">Algorithm</th><th>Avg Cost</th><th>Fill Rate</th></tr>{algo_rows}</table>
</body></html>"""
