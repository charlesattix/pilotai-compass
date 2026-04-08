"""Reinforcement learning position sizer.

Tabular Q-learning agent that learns optimal position sizing from
historical PnL sequences. State = (drawdown_bucket, vol_regime,
signal_bucket, heat_bucket). Action = position size in 10% steps.
Reward = risk-adjusted PnL (Sharpe-like).

Includes Kelly, half-Kelly, and fixed-fractional baselines for
comparison, plus a full train/validate split framework.

Pure-Python — no external dependencies.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Environment: trading episode
# ---------------------------------------------------------------------------

@dataclass
class TradeStep:
    """One step in the trading environment."""
    date: str
    trade_pnl_pct: float     # raw trade P&L
    signal_strength: float   # ML signal 0-1
    vol_regime: int           # 0=low, 1=normal, 2=high
    portfolio_heat: float     # current allocation 0-1


@dataclass
class EnvState:
    """Discretised state for Q-learning."""
    dd_bucket: int        # 0-4: current drawdown level
    vol_bucket: int       # 0-2: low/normal/high vol
    signal_bucket: int    # 0-3: signal strength quartile
    heat_bucket: int      # 0-2: portfolio utilisation

    def key(self) -> Tuple[int, int, int, int]:
        return (self.dd_bucket, self.vol_bucket, self.signal_bucket, self.heat_bucket)


ACTIONS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
N_ACTIONS = len(ACTIONS)


def discretise_state(
    current_dd: float,
    vol_regime: int,
    signal_strength: float,
    portfolio_heat: float,
) -> EnvState:
    """Convert continuous state to discrete buckets."""
    dd_bucket = min(4, int(current_dd / 0.03))  # 0-3%=0, 3-6%=1, ..., 12%+=4
    signal_bucket = min(3, int(signal_strength * 4))  # quartiles
    heat_bucket = min(2, int(portfolio_heat / 0.35))  # 0-35%=0, 35-70%=1, 70%+=2
    return EnvState(dd_bucket, vol_regime, signal_bucket, heat_bucket)


def compute_reward(
    pnl: float,
    position_size: float,
    current_dd: float,
    dd_penalty: float = 2.0,
) -> float:
    """Risk-adjusted reward: PnL minus drawdown penalty."""
    scaled_pnl = pnl * position_size
    # Penalty for large DD
    dd_cost = dd_penalty * max(0, current_dd - 0.05) * position_size
    return scaled_pnl - dd_cost


# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """Tabular Q-learning position sizer."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_table: Dict[Tuple, List[float]] = {}
        self._training = True

    def _get_q(self, state_key: Tuple) -> List[float]:
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * N_ACTIONS
        return self.q_table[state_key]

    def select_action(self, state: EnvState) -> int:
        """Epsilon-greedy action selection. Returns action index."""
        q = self._get_q(state.key())
        if self._training and self.rng.random() < self.epsilon:
            return self.rng.randint(0, N_ACTIONS - 1)
        return max(range(N_ACTIONS), key=lambda i: q[i])

    def update(
        self,
        state: EnvState,
        action: int,
        reward: float,
        next_state: EnvState,
    ) -> None:
        """Q-learning update."""
        q = self._get_q(state.key())
        next_q = self._get_q(next_state.key())
        best_next = max(next_q)
        q[action] += self.lr * (reward + self.gamma * best_next - q[action])

    def get_size(self, state: EnvState) -> float:
        """Get position size (for inference)."""
        action_idx = self.select_action(state)
        return ACTIONS[action_idx]

    def set_training(self, training: bool) -> None:
        self._training = training

    @property
    def n_states_visited(self) -> int:
        return len(self.q_table)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    n_episodes: int
    n_steps: int
    states_visited: int
    final_epsilon: float
    avg_reward_first_quarter: float
    avg_reward_last_quarter: float
    reward_improvement: float


def train_agent(
    agent: QLearningAgent,
    episodes: List[List[TradeStep]],
    n_epochs: int = 10,
) -> TrainResult:
    """Train the Q-learning agent on multiple episodes."""
    agent.set_training(True)
    all_rewards: List[float] = []
    total_steps = 0

    for epoch in range(n_epochs):
        # Decay epsilon
        agent.epsilon = max(0.05, 0.15 * (1 - epoch / n_epochs))

        for episode in episodes:
            current_dd = 0.0
            peak_equity = 1.0
            equity = 1.0
            heat = 0.0

            for i, step in enumerate(episode):
                state = discretise_state(current_dd, step.vol_regime, step.signal_strength, heat)
                action_idx = agent.select_action(state)
                size = ACTIONS[action_idx]

                # Execute
                pnl = step.trade_pnl_pct * size
                equity *= (1 + pnl)
                if equity > peak_equity:
                    peak_equity = equity
                current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                heat = size

                reward = compute_reward(step.trade_pnl_pct, size, current_dd)
                all_rewards.append(reward)

                # Next state
                next_dd = current_dd
                next_vol = episode[i+1].vol_regime if i+1 < len(episode) else step.vol_regime
                next_sig = episode[i+1].signal_strength if i+1 < len(episode) else 0.5
                next_state = discretise_state(next_dd, next_vol, next_sig, size)

                agent.update(state, action_idx, reward, next_state)
                total_steps += 1

    n = len(all_rewards)
    q1 = _mean(all_rewards[:n//4]) if n >= 4 else 0
    q4 = _mean(all_rewards[-n//4:]) if n >= 4 else 0

    return TrainResult(
        n_episodes=len(episodes) * n_epochs,
        n_steps=total_steps,
        states_visited=agent.n_states_visited,
        final_epsilon=round(agent.epsilon, 3),
        avg_reward_first_quarter=round(q1, 6),
        avg_reward_last_quarter=round(q4, 6),
        reward_improvement=round(q4 - q1, 6),
    )


# ---------------------------------------------------------------------------
# Baseline sizers
# ---------------------------------------------------------------------------

def kelly_size(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Full Kelly fraction."""
    if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0:
        return 0.0
    edge = win_rate * avg_win - (1 - win_rate) * avg_loss
    return max(0, min(1, edge / avg_win))


def half_kelly_size(win_rate: float, avg_win: float, avg_loss: float) -> float:
    return kelly_size(win_rate, avg_win, avg_loss) * 0.5


def fixed_size(fraction: float = 0.10) -> float:
    return fraction


def regime_size(vol_regime: int, signal_strength: float) -> float:
    """Regime-based sizing from EXP-720."""
    base = {0: 0.20, 1: 0.10, 2: 0.05}.get(vol_regime, 0.10)
    return min(1.0, base * (1 + signal_strength))


# ---------------------------------------------------------------------------
# Backtest comparison
# ---------------------------------------------------------------------------

@dataclass
class SizerResult:
    name: str
    total_return: float
    max_dd: float
    sharpe: float
    avg_size: float
    n_trades: int


def backtest_sizer(
    name: str,
    episodes: List[List[TradeStep]],
    size_fn,  # callable(step, dd, equity) -> float
) -> SizerResult:
    """Backtest one sizing method on episodes."""
    all_pnls: List[float] = []
    sizes: List[float] = []

    for episode in episodes:
        equity = 1.0
        peak = 1.0
        dd = 0.0
        for step in episode:
            size = size_fn(step, dd, equity)
            pnl = step.trade_pnl_pct * size
            equity *= (1 + pnl)
            if equity > peak: peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            all_pnls.append(pnl)
            sizes.append(size)

    total = 1.0
    peak_t = 1.0
    worst_dd = 0.0
    for p in all_pnls:
        total *= (1 + p)
        if total > peak_t: peak_t = total
        d = (peak_t - total) / peak_t if peak_t > 0 else 0
        if d > worst_dd: worst_dd = d

    m = _mean(all_pnls)
    s = _std(all_pnls)
    sharpe = m / s * math.sqrt(252) if s > 0 else 0

    return SizerResult(name, round(total - 1, 4), round(worst_dd, 4),
                       round(sharpe, 2), round(_mean(sizes), 3), len(all_pnls))


def compare_all_sizers(
    train_episodes: List[List[TradeStep]],
    test_episodes: List[List[TradeStep]],
    agent: QLearningAgent,
    win_rate: float = 0.65,
    avg_win: float = 0.02,
    avg_loss: float = 0.015,
) -> List[SizerResult]:
    """Compare RL agent vs all baselines on test data."""
    agent.set_training(False)

    def rl_fn(step, dd, eq):
        state = discretise_state(dd, step.vol_regime, step.signal_strength, 0.5)
        return agent.get_size(state)

    def kelly_fn(step, dd, eq):
        return kelly_size(win_rate, avg_win, avg_loss)

    def half_kelly_fn(step, dd, eq):
        return half_kelly_size(win_rate, avg_win, avg_loss)

    def fixed_fn(step, dd, eq):
        return 0.10

    def regime_fn(step, dd, eq):
        return regime_size(step.vol_regime, step.signal_strength)

    results = []
    for name, fn in [("RL Q-Learning", rl_fn), ("Kelly", kelly_fn),
                      ("Half-Kelly", half_kelly_fn), ("Fixed 10%", fixed_fn),
                      ("Regime-Based", regime_fn)]:
        results.append(backtest_sizer(name, test_episodes, fn))

    results.sort(key=lambda r: r.sharpe, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

@dataclass
class RLSizerResult:
    train_result: TrainResult
    comparison: List[SizerResult]
    rl_rank: int           # 1-based rank of RL among all sizers
    rl_beats_kelly: bool
    rl_beats_fixed: bool
    best_sizer: str


def run_full_analysis(
    episodes: List[List[TradeStep]],
    train_frac: float = 0.6,
    n_epochs: int = 15,
    seed: int = 1290,
) -> RLSizerResult:
    """Train RL agent and compare vs baselines."""
    split = int(len(episodes) * train_frac)
    train_ep = episodes[:split]
    test_ep = episodes[split:]

    agent = QLearningAgent(learning_rate=0.1, discount=0.95, epsilon=0.15, seed=seed)
    tr = train_agent(agent, train_ep, n_epochs=n_epochs)

    comparison = compare_all_sizers(train_ep, test_ep, agent)

    rl_result = next((r for r in comparison if r.name == "RL Q-Learning"), None)
    kelly_result = next((r for r in comparison if r.name == "Kelly"), None)
    fixed_result = next((r for r in comparison if r.name == "Fixed 10%"), None)

    rl_rank = next((i+1 for i, r in enumerate(comparison) if r.name == "RL Q-Learning"), len(comparison))

    return RLSizerResult(
        train_result=tr,
        comparison=comparison,
        rl_rank=rl_rank,
        rl_beats_kelly=rl_result.sharpe > kelly_result.sharpe if rl_result and kelly_result else False,
        rl_beats_fixed=rl_result.sharpe > fixed_result.sharpe if rl_result and fixed_result else False,
        best_sizer=comparison[0].name if comparison else "unknown",
    )


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_episodes(
    n_episodes: int = 50,
    steps_per_episode: int = 60,
    seed: int = 1290,
) -> List[List[TradeStep]]:
    """Generate synthetic trading episodes with regime structure."""
    rng = random.Random(seed)
    episodes: List[List[TradeStep]] = []

    for ep in range(n_episodes):
        steps: List[TradeStep] = []
        # Random regime for this episode
        vol_regime = rng.choice([0, 0, 1, 1, 1, 2])  # mostly normal
        edge = {0: 0.012, 1: 0.008, 2: 0.002}[vol_regime]
        vol = {0: 0.008, 1: 0.015, 2: 0.030}[vol_regime]

        for s in range(steps_per_episode):
            pnl = rng.gauss(edge, vol)
            signal = max(0, min(1, 0.5 + pnl / 0.02 + rng.gauss(0, 0.2)))
            heat = rng.uniform(0, 0.5)
            yr = 2020 + ep // 10
            steps.append(TradeStep(f"{yr}-{(s%12)+1:02d}-01", pnl, signal, vol_regime, heat))

        episodes.append(steps)

    return episodes
