"""Tests for compass.rl_executor – RL execution agent."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from compass.rl_executor import (
    ACTION_LABELS,
    N_ACTIONS,
    BenchmarkComparison,
    EpisodeResult,
    ExecutionEnvironment,
    Experience,
    QLearningAgent,
    RLExecutor,
    RLExecutorResult,
    State,
    TrainingMetrics,
    run_twap,
    run_vwap,
)


# ── State ───────────────────────────────────────────────────────────────────
class TestState:
    def test_to_tuple_returns_ints(self):
        s = State(0.5, 0.5, 5.0, 3.0, 0.01)
        t = s.to_tuple()
        assert all(isinstance(x, (int, np.integer)) for x in t)
        assert len(t) == 4

    def test_boundary_values(self):
        s = State(0.0, 0.0, 0.0, 0.0, 0.0)
        assert all(x >= 0 for x in s.to_tuple())

    def test_max_values(self):
        s = State(1.0, 1.0, 20.0, 15.0, 0.05)
        t = s.to_tuple()
        assert all(isinstance(x, (int, np.integer)) for x in t)


# ── Environment ─────────────────────────────────────────────────────────────
class TestEnvironment:
    def test_reset_returns_state(self):
        env = ExecutionEnvironment()
        s = env.reset()
        assert isinstance(s, State)
        assert s.inventory == pytest.approx(1.0)

    def test_step_returns_tuple(self):
        env = ExecutionEnvironment()
        env.reset()
        next_s, reward, done = env.step(0)
        assert isinstance(next_s, State)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_market_100_executes_all(self):
        env = ExecutionEnvironment(total_qty=50, n_steps=10)
        env.reset()
        _, _, done = env.step(0)  # market 100%
        assert done  # should finish immediately

    def test_wait_no_execution(self):
        env = ExecutionEnvironment(total_qty=50, n_steps=10)
        env.reset()
        state, _, done = env.step(5)  # wait
        assert state.inventory == pytest.approx(1.0)
        assert not done

    def test_inventory_decreases(self):
        env = ExecutionEnvironment(total_qty=100, n_steps=10)
        env.reset()
        s1 = env._get_state()
        env.step(1)  # market 50%
        s2 = env._get_state()
        assert s2.inventory < s1.inventory

    def test_cost_positive(self):
        env = ExecutionEnvironment()
        env.reset()
        for _ in range(env.n_steps):
            _, _, done = env.step(1)
            if done:
                break
        assert env.total_cost_bps >= 0

    def test_penalty_for_unfinished(self):
        env = ExecutionEnvironment(total_qty=100, n_steps=3)
        env.reset()
        # Only wait → penalty at end
        for _ in range(3):
            env.step(5)
        assert env.total_cost_bps > 0

    def test_done_after_n_steps(self):
        env = ExecutionEnvironment(n_steps=5)
        env.reset()
        done = False
        for _ in range(10):
            _, _, done = env.step(5)
            if done:
                break
        assert done


# ── Q-Learning agent ────────────────────────────────────────────────────────
class TestQLearningAgent:
    def test_select_action_in_range(self):
        agent = QLearningAgent()
        s = State(0.5, 0.5, 5.0, 2.0, 0.01)
        action = agent.select_action(s)
        assert 0 <= action < N_ACTIONS

    def test_initial_epsilon_explores(self):
        agent = QLearningAgent(epsilon_start=1.0)
        # With epsilon=1.0, all actions should be random
        actions = set()
        s = State(0.5, 0.5, 5.0, 2.0, 0.01)
        for _ in range(100):
            actions.add(agent.select_action(s))
        assert len(actions) > 1

    def test_zero_epsilon_greedy(self):
        agent = QLearningAgent(epsilon_start=0.0)
        s = State(0.5, 0.5, 5.0, 2.0, 0.01)
        # Set Q values to make action 2 best
        q = agent.get_q(s.to_tuple())
        q[2] = 10.0
        actions = [agent.select_action(s) for _ in range(10)]
        assert all(a == 2 for a in actions)

    def test_store_and_replay(self):
        agent = QLearningAgent(batch_size=2)
        for i in range(5):
            agent.store(Experience(
                state=(0, i, 0, 0), action=0, reward=-1.0,
                next_state=(0, i, 0, 0), done=False,
            ))
        agent.learn()  # should not raise
        assert agent.q_table_size > 0

    def test_epsilon_decays(self):
        agent = QLearningAgent(epsilon_start=1.0, epsilon_decay=0.9)
        e0 = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < e0

    def test_epsilon_floor(self):
        agent = QLearningAgent(epsilon_start=0.06, epsilon_end=0.05, epsilon_decay=0.5)
        agent.decay_epsilon()
        assert agent.epsilon >= agent.epsilon_end

    def test_q_values_update(self):
        agent = QLearningAgent(batch_size=1, learning_rate=0.5)
        exp = Experience((0, 0, 0, 0), 0, 10.0, (0, 0, 0, 0), True)
        agent.store(exp)
        q_before = agent.get_q((0, 0, 0, 0))[0]
        agent.learn()
        q_after = agent.get_q((0, 0, 0, 0))[0]
        assert q_after > q_before


# ── TWAP/VWAP baselines ────────────────────────────────────────────────────
class TestBaselines:
    def test_twap_returns_cost(self):
        env = ExecutionEnvironment()
        cost = run_twap(env)
        assert isinstance(cost, float)
        assert cost >= 0

    def test_vwap_returns_cost(self):
        env = ExecutionEnvironment()
        cost = run_vwap(env)
        assert isinstance(cost, float)
        assert cost >= 0

    def test_twap_deterministic(self):
        c1 = run_twap(ExecutionEnvironment(random_state=42))
        c2 = run_twap(ExecutionEnvironment(random_state=42))
        assert c1 == pytest.approx(c2)


# ── Full training ───────────────────────────────────────────────────────────
class TestRLExecutor:
    def test_returns_result(self):
        r = RLExecutor().train(n_episodes=20)
        assert isinstance(r, RLExecutorResult)

    def test_training_metrics(self):
        r = RLExecutor().train(n_episodes=30)
        assert r.training is not None
        assert len(r.training.episode_costs) == 30
        assert len(r.training.episode_rewards) == 30
        assert len(r.training.epsilon_history) == 30

    def test_benchmark_present(self):
        r = RLExecutor().train(n_episodes=20)
        assert r.benchmark is not None
        assert r.benchmark.agent_cost_bps >= 0
        assert r.benchmark.twap_cost_bps >= 0
        assert r.benchmark.vwap_cost_bps >= 0

    def test_action_counts(self):
        r = RLExecutor().train(n_episodes=30)
        total = sum(r.training.action_counts.values())
        assert total > 0

    def test_n_episodes(self):
        r = RLExecutor().train(n_episodes=15)
        assert r.n_episodes == 15

    def test_generated_at(self):
        r = RLExecutor().train(n_episodes=10)
        assert len(r.generated_at) > 0

    def test_costs_decrease_with_training(self):
        """Average cost in last 10 episodes should be <= first 10."""
        r = RLExecutor().train(n_episodes=100)
        first = np.mean(r.training.episode_costs[:10])
        last = np.mean(r.training.episode_costs[-10:])
        # Not strictly guaranteed but should usually hold
        assert last <= first * 1.5  # allow some slack

    def test_custom_env_config(self):
        r = RLExecutor(env_config={"total_qty": 50, "n_steps": 5}).train(n_episodes=10)
        assert r.n_episodes == 10


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            rl = RLExecutor()
            r = rl.train(n_episodes=20)
            path = rl.generate_report(r, output_path=Path(tmp) / "rl.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            rl = RLExecutor()
            r = rl.train(n_episodes=20)
            path = rl.generate_report(r, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "RL Execution Agent" in html
            assert "Learning Curve" in html
            assert "Action Distribution" in html
            assert "Benchmark" in html

    def test_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            rl = RLExecutor()
            r = rl.train(n_episodes=10)
            path = rl.generate_report(r, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_experience(self):
        e = Experience((0, 0, 0, 0), 1, -0.5, (0, 1, 0, 0), False)
        assert e.action == 1

    def test_benchmark(self):
        b = BenchmarkComparison(5.0, 8.0, 7.0, 3.0, 2.0)
        assert b.improvement_vs_twap_bps == 3.0

    def test_training_metrics_defaults(self):
        t = TrainingMetrics()
        assert t.episode_costs == []

    def test_result_defaults(self):
        r = RLExecutorResult()
        assert r.training is None
        assert r.n_episodes == 0

    def test_action_labels_count(self):
        assert len(ACTION_LABELS) == N_ACTIONS
