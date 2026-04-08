"""
Reinforcement learning portfolio manager — lightweight PPO agent.

Numpy-only PPO for dynamic allocation across strategies.  State =
portfolio metrics + regime + market features.  Action = allocation
weights.  Reward = risk-adjusted return with DD penalty.

Usage::

    from compass.rl_portfolio_manager import RLPortfolioManager, RLConfig
    mgr = RLPortfolioManager(strategy_returns, RLConfig())
    mgr.train()
    bt = mgr.backtest()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Configuration ───────────────────────────────────────────────────────


@dataclass
class RLConfig:
    state_dim: int = 10
    n_strategies: int = 5
    # PPO hyperparameters
    lr_policy: float = 0.003
    lr_value: float = 0.01
    gamma: float = 0.99                 # discount
    lam: float = 0.95                   # GAE lambda
    clip_eps: float = 0.20              # PPO clip
    n_epochs: int = 4                   # PPO epochs per update
    batch_size: int = 32
    # Training
    n_episodes: int = 200
    episode_length: int = 60            # trading days per episode
    train_fraction: float = 0.75        # IS/OOS split
    # Reward shaping
    dd_penalty: float = 2.0             # penalty multiplier for drawdown
    sharpe_bonus: float = 0.5           # bonus for Sharpe-like smoothness
    # Exploration
    entropy_coef: float = 0.01
    seed: int = 42


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class State:
    """Portfolio state observation."""
    portfolio_return: float    # trailing 20d return
    portfolio_vol: float       # trailing 20d vol
    drawdown: float            # current drawdown from peak
    regime_code: float         # 0=bull, 0.33=neutral, 0.67=bear, 1=crash
    vix_norm: float            # VIX normalised to [0,1]
    momentum: float            # 10d momentum
    mean_corr: float           # avg pairwise strategy correlation
    best_strat_ret: float      # best strategy trailing return
    worst_strat_ret: float     # worst strategy trailing return
    time_frac: float           # fraction of episode elapsed


@dataclass
class Action:
    """Portfolio allocation action."""
    weights: np.ndarray        # allocation weights summing to 1
    log_prob: float            # log probability of this action
    value: float               # value function estimate


@dataclass
class Transition:
    """One step of experience."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


@dataclass
class TrainResult:
    """Training result."""
    n_episodes: int
    final_avg_reward: float
    reward_history: List[float]
    policy_loss_history: List[float]
    value_loss_history: List[float]
    best_episode_reward: float


@dataclass
class AllocationSnapshot:
    """One step of allocation during backtest."""
    step: int
    weights: np.ndarray
    portfolio_return: float
    regime: str
    drawdown: float


@dataclass
class BacktestResult:
    """RL vs baseline comparison."""
    # RL agent
    rl_pnl: float
    rl_sharpe: float
    rl_dd: float
    rl_win_rate: float
    # Equal weight baseline
    ew_pnl: float
    ew_sharpe: float
    ew_dd: float
    # Risk parity baseline
    rp_pnl: float
    rp_sharpe: float
    rp_dd: float
    # Inverse-vol baseline (HRP proxy)
    iv_pnl: float
    iv_sharpe: float
    iv_dd: float
    # Comparisons
    sharpe_improvement_vs_ew: float
    sharpe_improvement_vs_rp: float
    dd_improvement_vs_ew: float
    # Allocations
    allocations: List[AllocationSnapshot]
    oos_fraction: float


# ── Neural network (numpy-only linear policy + value) ───────────────────


class LinearPolicy:
    """Linear policy: state → action logits + value."""

    def __init__(self, state_dim: int, action_dim: int, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        scale = 0.1
        self.W_policy = rng.randn(state_dim, action_dim) * scale
        self.b_policy = np.zeros(action_dim)
        self.W_value = rng.randn(state_dim, 1) * scale
        self.b_value = np.zeros(1)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns (action_probs, value)."""
        logits = state @ self.W_policy + self.b_policy
        probs = _softmax(logits)
        value = float((state @ self.W_value + self.b_value)[0])
        return probs, value

    def sample_action(self, state: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, float, float]:
        """Sample action from policy. Returns (weights, log_prob, value)."""
        probs, value = self.forward(state)
        # Dirichlet-like sampling: use probs as concentration
        alpha = np.maximum(probs * 10, 0.1)
        weights = rng.dirichlet(alpha)
        log_prob = float(np.sum(np.log(np.maximum(probs, 1e-8)) * weights))
        return weights, log_prob, value

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        lr_policy: float,
        lr_value: float,
        clip_eps: float,
        entropy_coef: float,
    ) -> Tuple[float, float]:
        """PPO update step. Returns (policy_loss, value_loss)."""
        n = len(states)
        if n == 0:
            return 0.0, 0.0

        # Forward pass
        logits = states @ self.W_policy + self.b_policy
        probs = np.array([_softmax(l) for l in logits])
        values = (states @ self.W_value + self.b_value).flatten()

        # Log probs of taken actions
        new_log_probs = np.sum(np.log(np.maximum(probs, 1e-8)) * actions, axis=1)

        # PPO ratio
        ratio = np.exp(new_log_probs - old_log_probs)
        clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -float(np.mean(np.minimum(ratio * advantages, clipped * advantages)))

        # Entropy bonus
        entropy = -float(np.mean(np.sum(probs * np.log(np.maximum(probs, 1e-8)), axis=1)))
        policy_loss -= entropy_coef * entropy

        # Value loss
        value_loss = float(np.mean((values - returns) ** 2))

        # Gradient approximation via finite differences (simplified)
        # Policy gradient: ∂L/∂W ≈ -advantages × state × (action - probs)
        for i in range(n):
            grad = np.outer(states[i], (actions[i] - probs[i]) * advantages[i])
            self.W_policy += lr_policy * grad / n
            self.b_policy += lr_policy * (actions[i] - probs[i]) * advantages[i] / n

        # Value gradient
        for i in range(n):
            v_err = returns[i] - values[i]
            self.W_value += lr_value * v_err * states[i].reshape(-1, 1) / n
            self.b_value += lr_value * v_err / n

        return policy_loss, value_loss


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ── Environment ─────────────────────────────────────────────────────────


class PortfolioEnv:
    """Portfolio allocation environment."""

    def __init__(
        self,
        strategy_returns: np.ndarray,
        regimes: np.ndarray,
        vix: np.ndarray,
        episode_length: int = 60,
        dd_penalty: float = 2.0,
        sharpe_bonus: float = 0.5,
    ) -> None:
        self.returns = strategy_returns  # (T, N)
        self.regimes = regimes
        self.vix = vix
        self.T, self.N = strategy_returns.shape
        self.episode_length = min(episode_length, self.T)
        self.dd_penalty = dd_penalty
        self.sharpe_bonus = sharpe_bonus
        self._step = 0
        self._start = 0
        self._equity = 1.0
        self._peak = 1.0
        self._returns_history: List[float] = []

    def reset(self, rng: np.random.RandomState) -> np.ndarray:
        max_start = self.T - self.episode_length
        self._start = rng.randint(0, max(1, max_start))
        self._step = 0
        self._equity = 1.0
        self._peak = 1.0
        self._returns_history = []
        return self._get_state()

    def step(self, weights: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take one step. Returns (next_state, reward, done)."""
        t = self._start + self._step
        if t >= self.T:
            return self._get_state(), 0.0, True

        period_returns = self.returns[t]
        portfolio_ret = float(np.dot(weights, period_returns))
        self._equity *= (1 + portfolio_ret)
        self._peak = max(self._peak, self._equity)
        self._returns_history.append(portfolio_ret)

        # Reward: return - DD penalty + Sharpe bonus
        dd = (self._equity - self._peak) / self._peak
        reward = portfolio_ret
        reward += dd * self.dd_penalty  # dd is negative → penalty
        if len(self._returns_history) > 5:
            rets = np.array(self._returns_history[-20:])
            if np.std(rets) > 0:
                reward += self.sharpe_bonus * np.mean(rets) / np.std(rets) * 0.01

        self._step += 1
        done = self._step >= self.episode_length
        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:
        t = self._start + self._step
        t = min(t, self.T - 1)
        lookback = 20

        # Strategy returns trailing
        start_lb = max(0, t - lookback)
        trail = self.returns[start_lb:t + 1]

        port_ret = float(np.mean(trail.sum(axis=1))) if len(trail) > 0 else 0
        port_vol = float(np.std(trail.sum(axis=1))) if len(trail) > 1 else 0.01
        dd = (self._equity - self._peak) / max(self._peak, 1e-10)

        # Regime encoding
        regime = str(self.regimes[t]).lower() if t < len(self.regimes) else "neutral"
        regime_map = {"bull": 0, "neutral": 0.33, "bear": 0.67, "high_vol": 0.83, "crash": 1.0}
        regime_code = regime_map.get(regime, 0.33)

        vix_norm = float(self.vix[t] / 50) if t < len(self.vix) else 0.36

        momentum = float(np.mean(trail[-5:].sum(axis=1))) if len(trail) >= 5 else 0

        if trail.shape[1] >= 2 and len(trail) > 5:
            corr = np.corrcoef(trail.T)
            np.fill_diagonal(corr, np.nan)
            mean_corr = float(np.nanmean(corr))
        else:
            mean_corr = 0.0

        strat_rets = trail.mean(axis=0) if len(trail) > 0 else np.zeros(self.N)
        best = float(np.max(strat_rets))
        worst = float(np.min(strat_rets))
        time_frac = self._step / max(self.episode_length, 1)

        return np.array([port_ret, port_vol, dd, regime_code, vix_norm,
                         momentum, mean_corr, best, worst, time_frac])


# ── RL Portfolio Manager ────────────────────────────────────────────────


class RLPortfolioManager:
    """RL-based dynamic portfolio allocator."""

    def __init__(
        self,
        strategy_returns: pd.DataFrame,
        config: Optional[RLConfig] = None,
        regimes: Optional[np.ndarray] = None,
        vix: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config or RLConfig()
        self.returns_df = strategy_returns
        self.returns = strategy_returns.values.astype(float)
        self.strategies = list(strategy_returns.columns)
        self.T, self.N = self.returns.shape

        self.config.n_strategies = self.N

        self.regimes = regimes if regimes is not None else np.full(self.T, "neutral")
        self.vix = vix if vix is not None else np.full(self.T, 18.0)

        self.rng = np.random.RandomState(self.config.seed)
        self.policy = LinearPolicy(self.config.state_dim, self.N, self.config.seed)
        self.train_result: Optional[TrainResult] = None
        self.backtest_result: Optional[BacktestResult] = None

    def train(self) -> TrainResult:
        """Train the PPO agent on in-sample data."""
        cfg = self.config
        split = int(self.T * cfg.train_fraction)
        train_returns = self.returns[:split]
        train_regimes = self.regimes[:split]
        train_vix = self.vix[:split]

        env = PortfolioEnv(
            train_returns, train_regimes, train_vix,
            cfg.episode_length, cfg.dd_penalty, cfg.sharpe_bonus,
        )

        reward_history: List[float] = []
        policy_losses: List[float] = []
        value_losses: List[float] = []
        best_reward = -float("inf")

        for ep in range(cfg.n_episodes):
            # Collect episode
            transitions: List[Transition] = []
            state = env.reset(self.rng)
            ep_reward = 0.0

            for step in range(cfg.episode_length):
                weights, log_prob, value = self.policy.sample_action(state, self.rng)
                next_state, reward, done = env.step(weights)
                transitions.append(Transition(
                    state, weights, reward, next_state, done, log_prob, value,
                ))
                ep_reward += reward
                state = next_state
                if done:
                    break

            reward_history.append(ep_reward)
            best_reward = max(best_reward, ep_reward)

            # Compute GAE advantages
            advantages, returns = self._compute_gae(transitions)

            # PPO update
            states = np.array([t.state for t in transitions])
            actions = np.array([t.action for t in transitions])
            old_lps = np.array([t.log_prob for t in transitions])

            for _ in range(cfg.n_epochs):
                pl, vl = self.policy.update(
                    states, actions, advantages, old_lps, returns,
                    cfg.lr_policy, cfg.lr_value, cfg.clip_eps, cfg.entropy_coef,
                )
                policy_losses.append(pl)
                value_losses.append(vl)

        self.train_result = TrainResult(
            cfg.n_episodes,
            float(np.mean(reward_history[-20:])),
            reward_history, policy_losses, value_losses, best_reward,
        )
        return self.train_result

    def _compute_gae(self, transitions: List[Transition]) -> Tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation."""
        cfg = self.config
        n = len(transitions)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = transitions[t + 1].value
            delta = transitions[t].reward + cfg.gamma * next_value - transitions[t].value
            gae = delta + cfg.gamma * cfg.lam * gae
            advantages[t] = gae
            returns[t] = advantages[t] + transitions[t].value

        # Normalise advantages
        if np.std(advantages) > 0:
            advantages = (advantages - np.mean(advantages)) / np.std(advantages)
        return advantages, returns

    def backtest(self, capital: float = 100_000) -> BacktestResult:
        """Backtest RL vs baselines on OOS data."""
        if self.train_result is None:
            self.train()

        cfg = self.config
        split = int(self.T * cfg.train_fraction)
        oos_returns = self.returns[split:]
        oos_regimes = self.regimes[split:]
        oos_vix = self.vix[split:]
        n_oos = len(oos_returns)

        if n_oos < 10:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], 0)

        # RL allocation
        env = PortfolioEnv(oos_returns, oos_regimes, oos_vix, n_oos)
        state = env.reset(self.rng)
        rl_rets = []
        allocations = []

        for t in range(n_oos):
            probs, _ = self.policy.forward(state)
            weights = probs / probs.sum()
            ret = float(np.dot(weights, oos_returns[t]))
            rl_rets.append(ret)
            regime = str(oos_regimes[t]) if t < len(oos_regimes) else "neutral"
            dd = (env._equity - env._peak) / max(env._peak, 1e-10)
            allocations.append(AllocationSnapshot(t, weights, ret, regime, dd))
            state, _, _ = env.step(weights)

        rl_rets = np.array(rl_rets)

        # Equal weight
        ew_weights = np.ones(self.N) / self.N
        ew_rets = oos_returns @ ew_weights

        # Risk parity (inverse vol)
        trailing = self.returns[max(0, split - 60):split]
        vols = np.std(trailing, axis=0)
        vols = np.maximum(vols, 1e-8)
        rp_weights = (1 / vols) / (1 / vols).sum()
        rp_rets = oos_returns @ rp_weights

        # Inverse vol (HRP proxy)
        iv_weights = rp_weights  # same as risk parity for this implementation
        iv_rets = oos_returns @ iv_weights

        def _metrics(rets):
            eq = capital * np.cumprod(1 + rets)
            eq_f = np.concatenate([[capital], eq])
            pk = np.maximum.accumulate(eq_f)
            dd = float(np.min((eq_f - pk) / np.where(pk > 0, pk, 1)))
            sh = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0
            pnl = float(eq_f[-1] - capital)
            wr = float((rets > 0).mean())
            return pnl, sh, dd, wr

        rl_pnl, rl_sh, rl_dd, rl_wr = _metrics(rl_rets)
        ew_pnl, ew_sh, ew_dd, _ = _metrics(ew_rets)
        rp_pnl, rp_sh, rp_dd, _ = _metrics(rp_rets)
        iv_pnl, iv_sh, iv_dd, _ = _metrics(iv_rets)

        sh_imp_ew = (rl_sh - ew_sh) / abs(ew_sh) * 100 if abs(ew_sh) > 0 else 0
        sh_imp_rp = (rl_sh - rp_sh) / abs(rp_sh) * 100 if abs(rp_sh) > 0 else 0
        dd_imp = (abs(ew_dd) - abs(rl_dd)) / abs(ew_dd) * 100 if abs(ew_dd) > 0 else 0

        self.backtest_result = BacktestResult(
            rl_pnl, rl_sh, rl_dd, rl_wr,
            ew_pnl, ew_sh, ew_dd,
            rp_pnl, rp_sh, rp_dd,
            iv_pnl, iv_sh, iv_dd,
            sh_imp_ew, sh_imp_rp, dd_imp,
            allocations, 1 - cfg.train_fraction,
        )
        return self.backtest_result
