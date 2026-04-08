"""Walk-forward ensemble optimizer — expanding-window optimization of strategy
weights that maximize Sharpe while capping drawdown at a threshold.

Uses gradient-based optimization (projected gradient descent) to find optimal
weights each period, then evaluates out-of-sample. Tracks weight stability,
turnover, and OOS degradation.

Provides:
  1. Expanding-window weight optimization (max Sharpe s.t. DD ≤ 12%)
  2. Comparison vs equal weight, risk parity, Bayesian selector
  3. Weight stability and turnover tracking
  4. Out-of-sample degradation analysis
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
class OptimizationWindow:
    """Result of one walk-forward optimization window."""
    window_end: str
    train_days: int
    test_days: int
    weights: Dict[str, float]
    train_sharpe: float
    train_dd: float
    test_sharpe: float
    test_dd: float
    degradation_pct: float    # (train_sharpe - test_sharpe) / train_sharpe


@dataclass
class TurnoverPoint:
    """Turnover at one rebalance."""
    date: str
    turnover: float           # sum of abs weight changes / 2
    cost: float               # turnover × cost_bps


@dataclass
class MethodComparison:
    """Performance of one allocation method."""
    method: str
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_dd_pct: float
    avg_turnover: float
    total_cost: float


@dataclass
class WFEnsembleResult:
    """Complete walk-forward optimizer output."""
    windows: List[OptimizationWindow] = field(default_factory=list)
    comparisons: List[MethodComparison] = field(default_factory=list)
    turnover_history: List[TurnoverPoint] = field(default_factory=list)
    best_method: str = ""
    avg_oos_degradation: float = 0.0
    weight_stability: float = 0.0   # 1 - avg turnover (higher = more stable)
    generated_at: str = ""


# ── Constrained optimizer (projected gradient descent) ──────────────────────
def optimize_weights(
    returns: np.ndarray,
    max_dd: float = 0.12,
    rf: float = 0.045,
    n_iter: int = 200,
    lr: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Find weights that maximize Sharpe subject to DD ≤ max_dd.

    Uses projected gradient ascent on the Sharpe ratio with a DD penalty.

    Parameters
    ----------
    returns : (T, N) array of daily strategy returns
    max_dd : maximum allowed drawdown fraction
    rf : annualized risk-free rate
    n_iter : optimization iterations
    lr : learning rate

    Returns
    -------
    Optimal weight vector (N,), sums to 1, all ≥ 0.
    """
    rng = np.random.RandomState(seed)
    T, N = returns.shape
    if N < 1 or T < 20:
        return np.ones(N) / max(N, 1)

    # Initialize near equal weight with small perturbation
    w = np.ones(N) / N + rng.randn(N) * 0.01
    w = _project_simplex(w)

    best_w = w.copy()
    best_obj = -np.inf

    for it in range(n_iter):
        port_ret = returns @ w
        mu = port_ret.mean() * TRADING_DAYS
        vol = port_ret.std() * np.sqrt(TRADING_DAYS)
        sharpe = (mu - rf) / max(vol, 1e-8)

        # Compute drawdown
        dd = _max_drawdown(port_ret)

        # Objective: Sharpe - penalty for DD > threshold
        penalty = max(0, (dd - max_dd)) * 50  # heavy penalty
        obj = sharpe - penalty

        if obj > best_obj:
            best_obj = obj
            best_w = w.copy()

        # Gradient of Sharpe w.r.t. weights (approximate)
        grad = np.zeros(N)
        eps = 1e-5
        for j in range(N):
            w_plus = w.copy()
            w_plus[j] += eps
            w_plus = w_plus / w_plus.sum()  # renormalize
            pr = returns @ w_plus
            mu_p = pr.mean() * TRADING_DAYS
            vol_p = pr.std() * np.sqrt(TRADING_DAYS)
            s_p = (mu_p - rf) / max(vol_p, 1e-8)
            dd_p = _max_drawdown(pr)
            pen_p = max(0, (dd_p - max_dd)) * 50
            grad[j] = (s_p - pen_p - obj) / eps

        # Gradient ascent step
        w = w + lr * grad
        w = _project_simplex(w)

        # Decay learning rate
        lr *= 0.998

    return best_w


def _project_simplex(w: np.ndarray, min_w: float = 0.02) -> np.ndarray:
    """Project onto simplex: all weights ≥ min_w, sum to 1."""
    w = np.maximum(w, min_w)
    return w / w.sum()


def _max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown from daily return array."""
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(dd.max()) if len(dd) > 0 else 0.0


def _sharpe(returns: np.ndarray, rf: float = 0.045) -> float:
    if len(returns) < 5 or returns.std() < 1e-10:
        return 0.0
    mu = returns.mean() * TRADING_DAYS
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    return float((mu - rf) / vol)


# ── Baseline allocators ────────────────────────────────────────────────────
def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def risk_parity(returns: np.ndarray) -> np.ndarray:
    vols = returns.std(axis=0) * np.sqrt(TRADING_DAYS)
    vols = np.maximum(vols, 1e-8)
    inv = 1.0 / vols
    return inv / inv.sum()


def bayesian_weights(returns: np.ndarray, seed: int = 42) -> np.ndarray:
    """Simple Thompson-sampling-style: sample from posterior, softmax."""
    rng = np.random.RandomState(seed)
    N = returns.shape[1]
    mus = returns.mean(axis=0)
    stds = returns.std(axis=0) / np.sqrt(max(len(returns), 1))
    samples = mus + rng.randn(N) * stds
    shifted = np.exp((samples - samples.min()) * 100)
    return shifted / shifted.sum()


# ── Walk-forward engine ────────────────────────────────────────────────────
class WFEnsembleOptimizer:
    """Walk-forward expanding-window ensemble optimizer."""

    def __init__(
        self,
        min_train: int = 120,
        test_size: int = 60,
        step_size: int = 60,
        max_dd: float = 0.12,
        cost_bps: float = 10.0,
        seed: int = 42,
    ) -> None:
        self.min_train = min_train
        self.test_size = test_size
        self.step_size = step_size
        self.max_dd = max_dd
        self.cost_bps = cost_bps
        self.seed = seed

    def run(self, returns: pd.DataFrame) -> WFEnsembleResult:
        """Run walk-forward optimization.

        returns: DataFrame with columns = strategy IDs, index = dates.
        """
        n, N = returns.shape
        strategies = list(returns.columns)
        if n < self.min_train + self.test_size + 10 or N < 2:
            return WFEnsembleResult(generated_at=_now())

        ret_arr = returns.values

        # Walk-forward windows
        windows: List[OptimizationWindow] = []
        wf_weights: List[Tuple[int, np.ndarray]] = []  # (start_of_test, weights)

        train_end = self.min_train
        while train_end + self.test_size <= n:
            train = ret_arr[:train_end]
            test = ret_arr[train_end:train_end + self.test_size]

            # Optimize on training data
            w = optimize_weights(train, self.max_dd, seed=self.seed + train_end)

            # Evaluate
            train_port = train @ w
            test_port = test @ w

            train_sh = _sharpe(train_port)
            test_sh = _sharpe(test_port)
            train_dd = _max_drawdown(train_port) * 100
            test_dd = _max_drawdown(test_port) * 100
            deg = (train_sh - test_sh) / max(abs(train_sh), 0.01) * 100

            d = str(returns.index[min(train_end, n - 1)])
            w_dict = {strategies[j]: round(float(w[j]), 4) for j in range(N)}

            windows.append(OptimizationWindow(
                window_end=d, train_days=train_end,
                test_days=self.test_size, weights=w_dict,
                train_sharpe=round(train_sh, 2),
                train_dd=round(train_dd, 2),
                test_sharpe=round(test_sh, 2),
                test_dd=round(test_dd, 2),
                degradation_pct=round(deg, 1),
            ))
            wf_weights.append((train_end, w))
            train_end += self.step_size

        if not windows:
            return WFEnsembleResult(generated_at=_now())

        # Build full WF weight trajectory
        wf_daily_weights = np.zeros((n, N))
        wf_daily_weights[:self.min_train] = equal_weight(N)
        for idx, (start, w) in enumerate(wf_weights):
            end = wf_weights[idx + 1][0] if idx + 1 < len(wf_weights) else n
            wf_daily_weights[start:end] = w

        # Simulate all methods
        methods_weights: Dict[str, np.ndarray] = {
            "wf_optimizer": wf_daily_weights,
            "equal_weight": np.tile(equal_weight(N), (n, 1)),
        }

        # Risk parity (rolling quarterly)
        rp_w = np.tile(equal_weight(N), (n, 1))
        for i in range(self.min_train, n, 60):
            rp_w[i:] = risk_parity(ret_arr[:i])
        methods_weights["risk_parity"] = rp_w

        # Bayesian
        bay_w = np.tile(equal_weight(N), (n, 1))
        for i in range(self.min_train, n, 20):
            bay_w[i:] = bayesian_weights(ret_arr[:i], seed=self.seed + i)
        methods_weights["bayesian"] = bay_w

        # Evaluate each
        comparisons: List[MethodComparison] = []
        eval_start = self.min_train
        for method, daily_w in methods_weights.items():
            comp, turnover = self._evaluate(
                method, ret_arr[eval_start:], daily_w[eval_start:],
                [str(returns.index[i]) for i in range(eval_start, n)],
            )
            comparisons.append(comp)

        # Turnover for WF optimizer
        turnover_hist: List[TurnoverPoint] = []
        prev_w = equal_weight(N)
        for start, w in wf_weights:
            to = float(np.sum(np.abs(w - prev_w)) / 2)
            cost = to * self.cost_bps / 10_000
            d = str(returns.index[min(start, n - 1)])
            turnover_hist.append(TurnoverPoint(d, round(to, 4), round(cost, 6)))
            prev_w = w

        # Summary stats
        avg_deg = float(np.mean([w.degradation_pct for w in windows]))
        avg_to = float(np.mean([t.turnover for t in turnover_hist])) if turnover_hist else 0
        stability = 1.0 - min(avg_to, 1.0)
        best = max(comparisons, key=lambda c: c.sharpe)

        return WFEnsembleResult(
            windows=windows,
            comparisons=comparisons,
            turnover_history=turnover_hist,
            best_method=best.method,
            avg_oos_degradation=round(avg_deg, 1),
            weight_stability=round(stability, 4),
            generated_at=_now(),
        )

    def _evaluate(
        self, method: str, returns: np.ndarray, weights: np.ndarray,
        dates: List[str],
    ) -> Tuple[MethodComparison, List[TurnoverPoint]]:
        n = min(len(returns), len(weights))
        capital = 100_000.0
        peak = capital
        max_dd = 0.0
        pnls: List[float] = []
        total_cost = 0.0
        turnover_pts: List[TurnoverPoint] = []
        prev_w = weights[0] if n > 0 else np.array([])

        for i in range(n):
            w = weights[i]
            to = float(np.sum(np.abs(w - prev_w)) / 2) if len(prev_w) == len(w) else 0
            cost = to * capital * self.cost_bps / 10_000
            total_cost += cost
            capital -= cost

            port_ret = float(returns[i] @ w)
            pnl = port_ret * capital
            capital += pnl
            pnls.append(pnl)
            prev_w = w

            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        total_ret = (capital - 100_000) / 100_000 * 100
        years = n / TRADING_DAYS
        cagr = ((capital / 100_000) ** (1 / years) - 1) * 100 if years > 0 and capital > 0 else 0

        dr = np.array(pnls) if pnls else np.array([0.0])
        sharpe = float(dr.mean() / dr.std() * np.sqrt(TRADING_DAYS)) if dr.std() > 0 else 0

        avg_to = total_cost / max(n, 1)

        return MethodComparison(
            method=method,
            total_return_pct=round(total_ret, 2),
            cagr_pct=round(cagr, 2),
            sharpe=round(sharpe, 2),
            max_dd_pct=round(max_dd * 100, 2),
            avg_turnover=round(avg_to, 4),
            total_cost=round(total_cost, 2),
        ), turnover_pts


# ── Synthetic data ──────────────────────────────────────────────────────────
def generate_strategy_returns(
    n: int = 1000, n_strategies: int = 4, seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    common = rng.randn(n) * 0.007
    data = {}
    for i in range(n_strategies):
        mu = 0.0003 + i * 0.00005
        vol = 0.008 + i * 0.001
        beta = 0.5 + i * 0.15
        data[f"S{i}"] = common * beta + rng.randn(n) * vol + mu
    return pd.DataFrame(data, index=idx)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
