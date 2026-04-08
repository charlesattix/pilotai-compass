"""Universal Portfolio — Cover's log-optimal meta-allocation via the
Exponential Gradient (EG) algorithm.

EG is an online convex optimisation approach that achieves the same
asymptotic growth rate as the best constant-rebalanced portfolio in
hindsight, with O(√T log N) regret.

Provides:
  1. Exponential Gradient algorithm for online portfolio selection
  2. Handles 10+ strategies efficiently (O(N) per step)
  3. Regret vs best constant-rebalanced portfolio (CRP)
  4. Comparison vs Thompson Sampling, equal weight, risk parity
  5. Cumulative wealth, turnover, max DD tracking
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
class EGState:
    """Exponential Gradient state at one time step."""
    t: int
    weights: Dict[str, float]
    portfolio_return: float
    cumulative_wealth: float
    turnover: float


@dataclass
class RegretAnalysis:
    """Regret vs best constant-rebalanced portfolio."""
    eg_log_wealth: float
    crp_log_wealth: float      # best CRP in hindsight
    regret: float              # crp - eg (lower = better)
    crp_weights: Dict[str, float]
    regret_per_round: float    # regret / T


@dataclass
class MethodResult:
    """Performance of one allocation method."""
    method: str
    final_wealth: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_dd_pct: float
    total_turnover: float
    log_wealth: float


@dataclass
class UniversalPortfolioResult:
    """Complete experiment output."""
    eg_history: List[EGState] = field(default_factory=list)
    regret: Optional[RegretAnalysis] = None
    comparisons: List[MethodResult] = field(default_factory=list)
    best_method: str = ""
    n_strategies: int = 0
    n_periods: int = 0
    generated_at: str = ""


# ── Exponential Gradient algorithm ──────────────────────────────────────────
class ExponentialGradient:
    """Online portfolio selection via Exponential Gradient (Helmbold et al. 1998).

    Update rule: w_{t+1,i} = w_{t,i} × exp(η × r_{t,i} / (w_t · r_t)) / Z
    where Z normalises to sum=1, η is the learning rate.
    """

    def __init__(
        self,
        n_assets: int,
        learning_rate: Optional[float] = None,
        min_weight: float = 0.01,
    ) -> None:
        self.n = n_assets
        # Default η from theory: √(8 ln(N) / T) — set for T≈1000
        self.eta = learning_rate or math.sqrt(8 * math.log(max(n_assets, 2)) / 1000)
        self.min_weight = min_weight
        self.weights = np.ones(n_assets) / n_assets
        self._t = 0

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def update(self, returns: np.ndarray) -> float:
        """Update weights after observing period returns.

        Parameters
        ----------
        returns : (N,) array of strategy returns for this period

        Returns
        -------
        Portfolio return for this period (using pre-update weights).
        """
        port_return = float(self.weights @ returns)

        # Multiplicative update
        if abs(port_return) > 1e-10:
            # Use log-wealth gradient: ∂ ln(w·r) / ∂w_i = r_i / (w·r)
            # But for returns (not prices), use: r_i / (1 + w·r)
            denom = 1.0 + port_return
            if denom > 0:
                gradient = returns / denom
                self.weights = self.weights * np.exp(self.eta * gradient)
        else:
            # Near-zero return: small gradient update
            self.weights = self.weights * np.exp(self.eta * returns)

        # Project onto simplex with minimum weight
        self.weights = np.maximum(self.weights, self.min_weight)
        self.weights = self.weights / self.weights.sum()

        self._t += 1
        return port_return

    @property
    def time_step(self) -> int:
        return self._t

    def reset(self) -> None:
        self.weights = np.ones(self.n) / self.n
        self._t = 0


# ── Best Constant-Rebalanced Portfolio (CRP) ───────────────────────────────
def best_crp(returns: np.ndarray, n_grid: int = 50) -> Tuple[np.ndarray, float]:
    """Find the best constant-rebalanced portfolio in hindsight.

    Uses grid search over the simplex for N≤4, random search for N>4.

    Returns (weights, log_wealth).
    """
    T, N = returns.shape
    if N == 1:
        log_w = np.sum(np.log(1 + returns[:, 0]))
        return np.array([1.0]), float(log_w)

    best_w = np.ones(N) / N
    best_lw = -np.inf

    if N <= 4:
        # Grid search
        from itertools import product
        steps = max(3, n_grid // N)
        grid = np.linspace(0.02, 0.96, steps)
        for combo in product(grid, repeat=N - 1):
            remainder = 1.0 - sum(combo)
            if remainder < 0.01:
                continue
            w = np.array(list(combo) + [remainder])
            w = w / w.sum()
            port_ret = returns @ w
            lw = float(np.sum(np.log(np.maximum(1 + port_ret, 1e-10))))
            if lw > best_lw:
                best_lw = lw
                best_w = w.copy()
    else:
        # Random search for high dimensions
        rng = np.random.RandomState(42)
        for _ in range(n_grid * 20):
            raw = rng.dirichlet(np.ones(N))
            w = np.maximum(raw, 0.02)
            w = w / w.sum()
            port_ret = returns @ w
            lw = float(np.sum(np.log(np.maximum(1 + port_ret, 1e-10))))
            if lw > best_lw:
                best_lw = lw
                best_w = w.copy()

    return best_w, best_lw


# ── Baseline allocators ────────────────────────────────────────────────────
def _equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def _risk_parity(returns: np.ndarray) -> np.ndarray:
    vols = returns.std(axis=0) * np.sqrt(TRADING_DAYS)
    vols = np.maximum(vols, 1e-8)
    inv = 1.0 / vols
    return inv / inv.sum()


def _thompson_sampling(returns: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mus = returns.mean(axis=0)
    stds = returns.std(axis=0) / np.sqrt(max(len(returns), 1))
    samples = mus + rng.randn(returns.shape[1]) * stds
    shifted = np.exp((samples - samples.min()) * 100)
    return shifted / shifted.sum()


# ── Full backtest ───────────────────────────────────────────────────────────
class UniversalPortfolioBacktest:
    """Backtest EG vs baselines on multi-strategy returns."""

    def __init__(
        self,
        starting_wealth: float = 1.0,
        learning_rate: Optional[float] = None,
        cost_bps: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.starting_wealth = starting_wealth
        self.lr = learning_rate
        self.cost_bps = cost_bps
        self.seed = seed

    def run(self, returns: pd.DataFrame) -> UniversalPortfolioResult:
        """Run full comparison backtest.

        returns: DataFrame, columns = strategy IDs, index = dates.
        """
        T, N = returns.shape
        strategies = list(returns.columns)
        if T < 30 or N < 2:
            return UniversalPortfolioResult(generated_at=_now())

        ret_arr = returns.values

        # Run EG
        eg = ExponentialGradient(N, self.lr)
        eg_history: List[EGState] = []
        eg_wealth = self.starting_wealth
        eg_daily: List[float] = []
        prev_w = eg.get_weights()

        for t in range(T):
            w = eg.get_weights()
            turnover = float(np.sum(np.abs(w - prev_w)) / 2)
            cost = turnover * self.cost_bps / 10_000
            port_ret = eg.update(ret_arr[t])
            eg_wealth *= (1 + port_ret - cost)
            eg_daily.append(port_ret - cost)

            eg_history.append(EGState(
                t=t,
                weights={strategies[j]: round(float(w[j]), 4) for j in range(N)},
                portfolio_return=round(port_ret, 6),
                cumulative_wealth=round(eg_wealth, 6),
                turnover=round(turnover, 6),
            ))
            prev_w = w.copy()

        # Best CRP
        crp_w, crp_lw = best_crp(ret_arr)
        eg_lw = float(np.sum(np.log(np.maximum(1 + np.array(eg_daily), 1e-10))))
        regret_val = crp_lw - eg_lw

        regret = RegretAnalysis(
            eg_log_wealth=round(eg_lw, 6),
            crp_log_wealth=round(crp_lw, 6),
            regret=round(regret_val, 6),
            crp_weights={strategies[j]: round(float(crp_w[j]), 4) for j in range(N)},
            regret_per_round=round(regret_val / T, 8),
        )

        # Run baselines
        comparisons: List[MethodResult] = []
        comparisons.append(self._evaluate("exponential_gradient", eg_daily))

        # Equal weight
        ew_daily = self._simulate_static(ret_arr, _equal_weight(N))
        comparisons.append(self._evaluate("equal_weight", ew_daily))

        # Risk parity (quarterly rebalance)
        rp_daily = self._simulate_adaptive(ret_arr, lambda r: _risk_parity(r), 60)
        comparisons.append(self._evaluate("risk_parity", rp_daily))

        # Thompson sampling
        ts_daily = self._simulate_adaptive(ret_arr, lambda r: _thompson_sampling(r, self.seed), 20)
        comparisons.append(self._evaluate("thompson_sampling", ts_daily))

        best = max(comparisons, key=lambda c: c.sharpe)

        return UniversalPortfolioResult(
            eg_history=eg_history,
            regret=regret,
            comparisons=comparisons,
            best_method=best.method,
            n_strategies=N,
            n_periods=T,
            generated_at=_now(),
        )

    def _simulate_static(self, returns: np.ndarray, weights: np.ndarray) -> List[float]:
        daily = []
        for t in range(len(returns)):
            daily.append(float(returns[t] @ weights))
        return daily

    def _simulate_adaptive(
        self, returns: np.ndarray, weight_fn, rebal_freq: int,
    ) -> List[float]:
        T, N = returns.shape
        daily = []
        w = _equal_weight(N)
        for t in range(T):
            if t > 0 and t % rebal_freq == 0 and t >= 60:
                w = weight_fn(returns[:t])
            daily.append(float(returns[t] @ w))
        return daily

    def _evaluate(self, method: str, daily_returns: List[float]) -> MethodResult:
        dr = np.array(daily_returns)
        wealth = self.starting_wealth * np.cumprod(1 + dr)
        final = float(wealth[-1]) if len(wealth) > 0 else self.starting_wealth
        total_ret = (final / self.starting_wealth - 1) * 100
        T = len(dr)
        years = T / TRADING_DAYS
        cagr = ((final / self.starting_wealth) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else 0
        sharpe = float(dr.mean() / dr.std() * np.sqrt(TRADING_DAYS)) if dr.std() > 0 else 0

        peak = np.maximum.accumulate(wealth)
        dd = (peak - wealth) / peak
        max_dd = float(dd.max()) * 100 if len(dd) > 0 else 0

        # Turnover (for static methods, 0; EG tracked separately)
        total_to = 0.0  # simplified

        log_w = float(np.sum(np.log(np.maximum(1 + dr, 1e-10))))

        return MethodResult(
            method=method,
            final_wealth=round(final, 6),
            total_return_pct=round(total_ret, 2),
            cagr_pct=round(cagr, 2),
            sharpe=round(sharpe, 2),
            max_dd_pct=round(max_dd, 2),
            total_turnover=round(total_to, 4),
            log_wealth=round(log_w, 6),
        )


# ── Synthetic data ──────────────────────────────────────────────────────────
def generate_strategy_returns(
    n: int = 1000, n_strategies: int = 5, seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    common = rng.randn(n) * 0.006
    data = {}
    for i in range(n_strategies):
        mu = 0.0002 + i * 0.00004
        vol = 0.007 + i * 0.001
        beta = 0.4 + i * 0.12
        data[f"S{i}"] = common * beta + rng.randn(n) * vol + mu
    return pd.DataFrame(data, index=idx)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
