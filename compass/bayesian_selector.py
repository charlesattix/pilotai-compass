"""Bayesian strategy selection via Thompson Sampling — models each strategy
as a bandit arm with Normal-Inverse-Gamma posterior on mean return and
variance, then allocates capital proportional to probability of being best.

Provides:
  1. Normal-Inverse-Gamma posterior per strategy (conjugate prior for unknown μ,σ²)
  2. Thompson Sampling: sample from posteriors, allocate to highest sample
  3. Comparison vs equal-weight, risk-parity, Markowitz
  4. Cumulative regret tracking
  5. Posterior evolution visualisation
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── Normal-Inverse-Gamma posterior ──────────────────────────────────────────
@dataclass
class NIGPosterior:
    """Normal-Inverse-Gamma posterior for (μ, σ²).

    Parameterisation: μ|σ² ~ N(mu, σ²/kappa), σ² ~ IG(alpha, beta)
    """
    mu: float = 0.0       # posterior mean of μ
    kappa: float = 1.0    # precision scaling (higher = more certain about μ)
    alpha: float = 2.0    # shape of IG (>1 for defined variance)
    beta: float = 0.001   # scale of IG

    def update(self, x: float) -> None:
        """Bayesian update with single observation x."""
        kappa_new = self.kappa + 1
        mu_new = (self.kappa * self.mu + x) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + 0.5 * self.kappa / kappa_new * (x - self.mu) ** 2
        self.mu = mu_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = beta_new

    def update_batch(self, xs: np.ndarray) -> None:
        """Update with multiple observations."""
        for x in xs:
            self.update(float(x))

    def sample_mean(self, rng: np.random.RandomState) -> float:
        """Sample μ from the marginal posterior (Student-t)."""
        # σ² ~ IG(alpha, beta) → sample variance first
        if self.alpha <= 0 or self.beta <= 0:
            return self.mu
        var = self.beta / max(self.alpha, 1e-6)  # posterior mean of σ²
        # μ | σ² ~ N(mu, σ²/kappa)
        std = math.sqrt(var / max(self.kappa, 1e-6))
        return float(self.mu + rng.randn() * std)

    def mean_estimate(self) -> float:
        return self.mu

    def variance_estimate(self) -> float:
        """Posterior mean of σ²."""
        if self.alpha <= 1:
            return self.beta
        return self.beta / (self.alpha - 1)

    def sharpe_estimate(self) -> float:
        """Estimated annualised Sharpe from posterior."""
        vol = math.sqrt(max(self.variance_estimate(), 1e-12))
        return self.mu / vol * math.sqrt(TRADING_DAYS) if vol > 1e-12 else 0.0

    @property
    def n_observations(self) -> int:
        """Approximate number of observations seen."""
        return max(0, int(self.kappa - 1))


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class StrategyArm:
    """One bandit arm = one strategy."""
    strategy_id: str
    posterior: NIGPosterior = field(default_factory=NIGPosterior)
    cumulative_return: float = 0.0
    n_selected: int = 0


@dataclass
class AllocationResult:
    """Single-period allocation decision."""
    weights: Dict[str, float]
    method: str
    sampled_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegretPoint:
    """Cumulative regret at one time step."""
    t: int
    thompson_cumret: float
    oracle_cumret: float       # best single strategy in hindsight
    regret: float              # oracle - thompson


@dataclass
class ComparisonResult:
    """Performance of one allocation method."""
    method: str
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_dd_pct: float


@dataclass
class BayesianSelectorResult:
    """Complete experiment output."""
    arms: List[StrategyArm] = field(default_factory=list)
    comparisons: List[ComparisonResult] = field(default_factory=list)
    regret_curve: List[RegretPoint] = field(default_factory=list)
    best_method: str = ""
    thompson_sharpe: float = 0.0
    total_regret: float = 0.0
    generated_at: str = ""


# ── Thompson Sampling selector ──────────────────────────────────────────────
class BayesianSelector:
    """Thompson Sampling for dynamic strategy allocation."""

    def __init__(
        self,
        strategy_ids: List[str],
        prior_mu: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 2.0,
        prior_beta: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.arms: Dict[str, StrategyArm] = {}
        for sid in strategy_ids:
            self.arms[sid] = StrategyArm(
                strategy_id=sid,
                posterior=NIGPosterior(prior_mu, prior_kappa, prior_alpha, prior_beta),
            )

    def select(self) -> AllocationResult:
        """Thompson Sampling: sample from each posterior, allocate proportionally."""
        samples: Dict[str, float] = {}
        for sid, arm in self.arms.items():
            samples[sid] = arm.posterior.sample_mean(self.rng)

        # Softmax-style allocation (shift so all positive, then normalise)
        min_s = min(samples.values())
        shifted = {k: math.exp((v - min_s) * 100) for k, v in samples.items()}
        total = sum(shifted.values())
        weights = {k: v / total for k, v in shifted.items()}

        return AllocationResult(
            weights=weights,
            method="thompson",
            sampled_values=samples,
        )

    def update(self, returns: Dict[str, float]) -> None:
        """Update posteriors with observed daily returns."""
        for sid, ret in returns.items():
            if sid in self.arms:
                self.arms[sid].posterior.update(ret)
                self.arms[sid].cumulative_return += ret

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Rank strategies by posterior Sharpe estimate."""
        ranked = [(sid, arm.posterior.sharpe_estimate()) for sid, arm in self.arms.items()]
        return sorted(ranked, key=lambda x: -x[1])

    def get_posteriors(self) -> Dict[str, Dict[str, float]]:
        """Current posterior summaries."""
        return {
            sid: {
                "mu": arm.posterior.mu,
                "variance": arm.posterior.variance_estimate(),
                "sharpe": arm.posterior.sharpe_estimate(),
                "n_obs": arm.posterior.n_observations,
            }
            for sid, arm in self.arms.items()
        }


# ── Baseline allocators ────────────────────────────────────────────────────
def equal_weight(strategies: List[str]) -> Dict[str, float]:
    n = len(strategies)
    return {s: 1.0 / n for s in strategies} if n > 0 else {}


def risk_parity(returns: pd.DataFrame) -> Dict[str, float]:
    """Inverse-volatility weights."""
    vols = returns.std() * np.sqrt(TRADING_DAYS)
    vols = vols.clip(lower=1e-8)
    inv = 1.0 / vols
    w = inv / inv.sum()
    return w.to_dict()


def markowitz(returns: pd.DataFrame, rf: float = 0.045) -> Dict[str, float]:
    """Max-Sharpe tangency portfolio."""
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    excess = mu - rf
    try:
        inv_cov = np.linalg.inv(cov.values + np.eye(len(mu)) * 1e-8)
        raw = inv_cov @ excess.values
        if raw.sum() <= 0:
            return equal_weight(list(returns.columns))
        w = raw / raw.sum()
        w = np.clip(w, 0.05, 0.60)
        w = w / w.sum()
        return {c: float(w[i]) for i, c in enumerate(returns.columns)}
    except Exception:
        return equal_weight(list(returns.columns))


# ── Backtest comparison ─────────────────────────────────────────────────────
class BayesianBacktest:
    """Backtest Thompson Sampling vs baselines."""

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        rebalance_freq: int = 1,
        warmup: int = 30,
        seed: int = 42,
    ) -> None:
        self.starting_capital = starting_capital
        self.rebalance_freq = rebalance_freq
        self.warmup = warmup
        self.seed = seed

    def run(self, returns: pd.DataFrame) -> BayesianSelectorResult:
        """Run comparison backtest.

        returns: DataFrame with columns = strategy IDs, index = dates.
        """
        n = len(returns)
        strategies = list(returns.columns)
        if n < self.warmup + 10 or len(strategies) < 2:
            return BayesianSelectorResult(generated_at=_now())

        # Run each method
        methods = {
            "thompson": self._run_thompson(returns),
            "equal_weight": self._run_static(returns, equal_weight(strategies)),
            "risk_parity": self._run_risk_parity(returns),
            "markowitz": self._run_markowitz(returns),
        }

        comparisons: List[ComparisonResult] = []
        for name, (cap, pnls) in methods.items():
            total_ret = (cap - self.starting_capital) / self.starting_capital * 100
            years = (n - self.warmup) / TRADING_DAYS
            cagr = ((cap / self.starting_capital) ** (1 / years) - 1) * 100 if years > 0 and cap > 0 else 0
            dr = np.array(pnls)
            sharpe = float(dr.mean() / dr.std() * np.sqrt(TRADING_DAYS)) if len(dr) > 1 and dr.std() > 0 else 0
            peak = self.starting_capital
            max_dd = 0.0
            equity = self.starting_capital
            for p in pnls:
                equity += p
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            comparisons.append(ComparisonResult(
                method=name,
                total_return_pct=round(total_ret, 2),
                cagr_pct=round(cagr, 2),
                sharpe=round(sharpe, 2),
                max_dd_pct=round(max_dd * 100, 2),
            ))

        # Regret curve
        thompson_pnls = methods["thompson"][1]
        # Oracle: best single strategy in hindsight
        strategy_cumrets = {s: float(returns[s].iloc[self.warmup:].sum()) for s in strategies}
        oracle_strategy = max(strategy_cumrets, key=strategy_cumrets.get)
        oracle_daily = returns[oracle_strategy].iloc[self.warmup:].values

        regret_curve: List[RegretPoint] = []
        t_cum = 0.0
        o_cum = 0.0
        for t in range(len(thompson_pnls)):
            t_cum += thompson_pnls[t] / self.starting_capital
            if t < len(oracle_daily):
                o_cum += oracle_daily[t]
            regret_curve.append(RegretPoint(
                t=t, thompson_cumret=round(t_cum, 6),
                oracle_cumret=round(o_cum, 6),
                regret=round(o_cum - t_cum, 6),
            ))

        best = max(comparisons, key=lambda c: c.sharpe)
        thompson_comp = next(c for c in comparisons if c.method == "thompson")
        total_regret = regret_curve[-1].regret if regret_curve else 0

        # Get final arms state
        selector = BayesianSelector(strategies, seed=self.seed)
        for i in range(self.warmup, n):
            day_ret = {s: float(returns[s].iloc[i]) for s in strategies}
            selector.update(day_ret)
        arms = list(selector.arms.values())

        return BayesianSelectorResult(
            arms=arms,
            comparisons=comparisons,
            regret_curve=regret_curve,
            best_method=best.method,
            thompson_sharpe=thompson_comp.sharpe,
            total_regret=round(total_regret, 4),
            generated_at=_now(),
        )

    def _run_thompson(self, returns: pd.DataFrame) -> Tuple[float, List[float]]:
        strategies = list(returns.columns)
        selector = BayesianSelector(strategies, seed=self.seed)
        capital = self.starting_capital
        pnls: List[float] = []

        # Warmup: update posteriors only
        for i in range(self.warmup):
            day_ret = {s: float(returns[s].iloc[i]) for s in strategies}
            selector.update(day_ret)

        for i in range(self.warmup, len(returns)):
            if i % self.rebalance_freq == 0:
                alloc = selector.select()
                weights = alloc.weights

            day_ret = {s: float(returns[s].iloc[i]) for s in strategies}
            port_ret = sum(weights.get(s, 0) * day_ret[s] for s in strategies)
            pnl = port_ret * capital
            capital += pnl
            pnls.append(pnl)
            selector.update(day_ret)

        return capital, pnls

    def _run_static(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Tuple[float, List[float]]:
        capital = self.starting_capital
        pnls: List[float] = []
        strategies = list(returns.columns)
        for i in range(self.warmup, len(returns)):
            day_ret = {s: float(returns[s].iloc[i]) for s in strategies}
            port_ret = sum(weights.get(s, 0) * day_ret[s] for s in strategies)
            pnl = port_ret * capital
            capital += pnl
            pnls.append(pnl)
        return capital, pnls

    def _run_risk_parity(self, returns: pd.DataFrame) -> Tuple[float, List[float]]:
        capital = self.starting_capital
        pnls: List[float] = []
        strategies = list(returns.columns)
        weights = equal_weight(strategies)
        for i in range(self.warmup, len(returns)):
            if i % 20 == 0 and i >= 60:  # monthly rebalance
                weights = risk_parity(returns.iloc[:i])
            day_ret = {s: float(returns[s].iloc[i]) for s in strategies}
            port_ret = sum(weights.get(s, 0) * day_ret[s] for s in strategies)
            pnl = port_ret * capital
            capital += pnl
            pnls.append(pnl)
        return capital, pnls

    def _run_markowitz(self, returns: pd.DataFrame) -> Tuple[float, List[float]]:
        capital = self.starting_capital
        pnls: List[float] = []
        strategies = list(returns.columns)
        weights = equal_weight(strategies)
        for i in range(self.warmup, len(returns)):
            if i % 20 == 0 and i >= 60:
                weights = markowitz(returns.iloc[:i])
            day_ret = {s: float(returns[s].iloc[i]) for s in strategies}
            port_ret = sum(weights.get(s, 0) * day_ret[s] for s in strategies)
            pnl = port_ret * capital
            capital += pnl
            pnls.append(pnl)
        return capital, pnls


# ── Synthetic data ──────────────────────────────────────────────────────────
def generate_strategy_returns(
    n: int = 1000, n_strategies: int = 4, seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    common = rng.randn(n) * 0.008
    data = {}
    names = [f"EXP-{880 + i * 30}" for i in range(n_strategies)]
    for i, name in enumerate(names):
        mu = 0.0003 + i * 0.0001  # slightly different expected returns
        vol = 0.010 + i * 0.002
        data[name] = common * (0.5 + i * 0.15) + rng.randn(n) * vol + mu
    return pd.DataFrame(data, index=idx)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
