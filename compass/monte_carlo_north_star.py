"""Monte Carlo stress testing for the North Star 4-strategy portfolio.

50K-path simulation under crisis scenarios, correlation breakdown,
liquidity stress, regime shift, and black swan events.

Pure-Python — no external dependencies.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs: List[float]) -> float:
    if len(xs) < 2: return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _percentile(xs: List[float], pct: float) -> float:
    if not xs: return 0.0
    s = sorted(xs); idx = pct / 100 * (len(s) - 1)
    lo = int(idx); hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)


# ---------------------------------------------------------------------------
# Portfolio specification (from EXP-1470 4-strategy blend)
# ---------------------------------------------------------------------------

@dataclass
class StrategySpec:
    name: str
    annual_return: float
    annual_vol: float
    weight: float


DEFAULT_PORTFOLIO = [
    StrategySpec("credit_spreads", 0.42, 0.12, 0.40),
    StrategySpec("vol_harvesting", 0.28, 0.15, 0.25),
    StrategySpec("momentum_overlay", 0.18, 0.10, 0.20),
    StrategySpec("tail_hedge", -0.03, 0.20, 0.15),
]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """One stress scenario specification."""
    name: str
    description: str
    n_days: int
    equity_shock: float       # total equity move
    shock_duration: int       # days for the shock
    correlation_override: Optional[float]  # override inter-strategy corr
    cost_multiplier: float    # transaction cost multiplier
    fill_rate: float          # fill rate (1.0 = normal)
    drift_override: Optional[float]  # override annual drift


DEFAULT_SCENARIOS: List[Scenario] = [
    Scenario("base", "Normal conditions (1 year)", 252, 0.0, 0, None, 1.0, 1.0, None),
    Scenario("covid_crash", "COVID-like: -34% in 23 days", 252, -0.34, 23, 0.85, 2.0, 0.8, None),
    Scenario("bear_2022", "2022 bear: -25% over 200 days", 252, -0.25, 200, 0.6, 1.5, 0.9, None),
    Scenario("flash_crash", "Flash crash: -7% in 1 day", 252, -0.07, 1, 0.95, 3.0, 0.5, None),
    Scenario("corr_05", "Correlation breakdown: all at 0.5", 252, 0.0, 0, 0.5, 1.0, 1.0, None),
    Scenario("corr_07", "Correlation breakdown: all at 0.7", 252, 0.0, 0, 0.7, 1.0, 1.0, None),
    Scenario("corr_09", "Correlation breakdown: all at 0.9", 252, 0.0, 0, 0.9, 1.0, 1.0, None),
    Scenario("liquidity", "Liquidity crisis: 3x costs, 50% fills", 252, 0.0, 0, None, 3.0, 0.5, None),
    Scenario("permanent_bear", "Permanent bear: negative drift", 252, 0.0, 0, 0.6, 1.5, 0.9, -0.10),
    Scenario("black_swan", "Black swan: 5-sigma daily move", 252, -0.15, 1, 0.95, 3.0, 0.3, None),
]


# ---------------------------------------------------------------------------
# Monte Carlo engine
# ---------------------------------------------------------------------------

@dataclass
class PathResult:
    final_equity: float
    max_drawdown: float
    total_return: float
    survived: bool         # DD never exceeded 50%


@dataclass
class ScenarioResult:
    """Results for one scenario across all paths."""
    scenario: str
    description: str
    n_paths: int
    survival_rate: float
    mean_return: float
    median_return: float
    p5_return: float       # 5th percentile (worst 5%)
    p1_return: float       # 1st percentile
    mean_dd: float
    p95_dd: float
    p99_dd: float
    worst_dd: float
    mean_cagr: float
    worst_cagr: float
    prob_loss: float       # P(return < 0)
    prob_dd_gt_20: float
    prob_dd_gt_30: float
    prob_dd_gt_50: float


def simulate_path(
    rng: random.Random,
    portfolio: List[StrategySpec],
    scenario: Scenario,
    initial_capital: float = 1_000_000.0,
) -> PathResult:
    """Simulate one MC path for the portfolio under a scenario."""
    n = scenario.n_days
    n_strats = len(portfolio)

    # Per-strategy daily params
    daily_mus = []
    daily_vols = []
    for s in portfolio:
        mu = (scenario.drift_override if scenario.drift_override is not None else s.annual_return) / 252
        vol = s.annual_vol / math.sqrt(252)
        daily_mus.append(mu)
        daily_vols.append(vol)

    weights = [s.weight for s in portfolio]

    # Correlation (Cholesky-like: common + idio factor)
    corr = scenario.correlation_override if scenario.correlation_override is not None else 0.3
    common_weight = math.sqrt(max(0, corr))
    idio_weight = math.sqrt(max(0, 1 - corr))

    # Cost drag per day
    base_cost = 0.0001  # ~1bp/day base
    cost_per_day = base_cost * scenario.cost_multiplier

    equity = initial_capital
    peak = equity
    worst_dd = 0.0
    survived = True

    # Pre-generate shock days
    shock_start = rng.randint(20, max(21, n - scenario.shock_duration - 1)) if scenario.shock_duration > 0 and scenario.equity_shock != 0 else -1
    shock_per_day = scenario.equity_shock / max(scenario.shock_duration, 1) if scenario.shock_duration > 0 else 0

    for day in range(n):
        common_z = rng.gauss(0, 1)
        port_return = 0.0

        for i in range(n_strats):
            idio_z = rng.gauss(0, 1)
            z = common_weight * common_z + idio_weight * idio_z
            ret = daily_mus[i] + daily_vols[i] * z

            # Apply shock during shock period
            if shock_start <= day < shock_start + scenario.shock_duration:
                ret += shock_per_day / n_strats

            # Fill rate: randomly skip some returns (simulating unfilled orders)
            if rng.random() > scenario.fill_rate:
                ret *= 0.3  # partial execution

            port_return += weights[i] * ret

        # Cost drag
        port_return -= cost_per_day

        equity *= (1 + port_return)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > worst_dd:
            worst_dd = dd
        if dd > 0.50:
            survived = False

    return PathResult(
        final_equity=equity,
        max_drawdown=worst_dd,
        total_return=(equity / initial_capital) - 1,
        survived=survived,
    )


def run_scenario(
    scenario: Scenario,
    portfolio: Optional[List[StrategySpec]] = None,
    n_paths: int = 50_000,
    seed: int = 2024,
) -> ScenarioResult:
    """Run MC simulation for one scenario."""
    rng = random.Random(seed)
    port = portfolio or DEFAULT_PORTFOLIO

    paths: List[PathResult] = []
    for _ in range(n_paths):
        paths.append(simulate_path(rng, port, scenario))

    returns = [p.total_return for p in paths]
    dds = [p.max_drawdown for p in paths]
    survived = sum(1 for p in paths if p.survived)

    cagrs = [(1 + r) ** (252 / scenario.n_days) - 1 if r > -1 else -1 for r in returns]

    return ScenarioResult(
        scenario=scenario.name, description=scenario.description,
        n_paths=n_paths,
        survival_rate=round(survived / n_paths, 4),
        mean_return=round(_mean(returns), 4),
        median_return=round(_percentile(returns, 50), 4),
        p5_return=round(_percentile(returns, 5), 4),
        p1_return=round(_percentile(returns, 1), 4),
        mean_dd=round(_mean(dds), 4),
        p95_dd=round(_percentile(dds, 95), 4),
        p99_dd=round(_percentile(dds, 99), 4),
        worst_dd=round(max(dds), 4),
        mean_cagr=round(_mean(cagrs), 4),
        worst_cagr=round(min(cagrs), 4),
        prob_loss=round(sum(1 for r in returns if r < 0) / n_paths, 4),
        prob_dd_gt_20=round(sum(1 for d in dds if d > 0.20) / n_paths, 4),
        prob_dd_gt_30=round(sum(1 for d in dds if d > 0.30) / n_paths, 4),
        prob_dd_gt_50=round(sum(1 for d in dds if d > 0.50) / n_paths, 4),
    )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

@dataclass
class MCNorthStarResult:
    """Complete MC stress test output."""
    n_scenarios: int
    n_paths_per_scenario: int
    portfolio: List[StrategySpec]
    results: List[ScenarioResult]
    worst_scenario: str
    safest_scenario: str
    runtime_seconds: float


def run_full_stress_test(
    portfolio: Optional[List[StrategySpec]] = None,
    scenarios: Optional[List[Scenario]] = None,
    n_paths: int = 50_000,
    seed: int = 2024,
) -> MCNorthStarResult:
    """Run all scenarios and compile results."""
    t0 = time.monotonic()
    port = portfolio or DEFAULT_PORTFOLIO
    scens = scenarios or DEFAULT_SCENARIOS

    results: List[ScenarioResult] = []
    for sc in scens:
        results.append(run_scenario(sc, port, n_paths, seed))

    worst = min(results, key=lambda r: r.p1_return).scenario
    safest = max(results, key=lambda r: r.survival_rate).scenario

    return MCNorthStarResult(
        len(scens), n_paths, port, results,
        worst, safest, round(time.monotonic() - t0, 1),
    )


# ---------------------------------------------------------------------------
# Synthetic test helper
# ---------------------------------------------------------------------------

def run_quick_test(n_paths: int = 1000, seed: int = 42) -> MCNorthStarResult:
    """Quick version for testing (fewer paths)."""
    return run_full_stress_test(n_paths=n_paths, seed=seed)
