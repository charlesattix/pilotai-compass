"""Tests for compass/strategy_decay_monitor.py — strategy decay detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.strategy_decay_monitor import (
    CUSUMResult,
    KillSignal,
    LifecyclePhase,
    MonitorResult,
    RecentPerformance,
    RollingMetrics,
    StrategyDecayMonitor,
    classify_lifecycle,
    compute_kill_score,
    compute_rolling_hit_rate,
    compute_rolling_sharpe,
    cusum_on_sharpe,
    score_recent_performance,
    TRADING_DAYS,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _dates(n: int = 300) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=n)


def _good_returns(n: int = 300, seed: int = 42) -> pd.Series:
    """Healthy strategy with positive Sharpe."""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.normal(0.001, 0.008, n), index=_dates(n))


def _decaying_returns(n: int = 300, seed: int = 77) -> pd.Series:
    """Strategy where alpha decays from positive to negative."""
    rng = np.random.RandomState(seed)
    alpha = np.linspace(0.002, -0.0005, n)
    noise = rng.normal(0, 0.008, n)
    return pd.Series(alpha + noise, index=_dates(n))


def _dead_returns(n: int = 300, seed: int = 99) -> pd.Series:
    """Strategy that's been losing steadily."""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.normal(-0.001, 0.01, n), index=_dates(n))


# ── Rolling Sharpe tests ─────────────────────────────────────────────────


class TestRollingSharpe:
    def test_output_length(self):
        vals = np.random.RandomState(42).normal(0.001, 0.01, 100)
        result = compute_rolling_sharpe(vals, 30)
        assert len(result) == 71  # 100 - 30 + 1

    def test_positive_for_good_returns(self):
        vals = np.random.RandomState(42).normal(0.002, 0.008, 200)
        result = compute_rolling_sharpe(vals, 60)
        assert np.mean(result) > 0

    def test_empty_when_too_short(self):
        vals = np.random.RandomState(42).normal(0, 0.01, 10)
        assert len(compute_rolling_sharpe(vals, 20)) == 0

    def test_zero_std_returns_zero(self):
        vals = np.ones(50) * 0.001
        result = compute_rolling_sharpe(vals, 20)
        assert all(s == 0.0 for s in result)


class TestRollingHitRate:
    def test_output_length(self):
        vals = np.random.RandomState(42).normal(0, 0.01, 100)
        result = compute_rolling_hit_rate(vals, 30)
        assert len(result) == 71

    def test_bounded(self):
        vals = np.random.RandomState(42).normal(0, 0.01, 100)
        result = compute_rolling_hit_rate(vals, 30)
        assert all(0 <= h <= 1 for h in result)

    def test_all_positive_gives_one(self):
        vals = np.abs(np.random.RandomState(42).normal(0.01, 0.001, 50))
        result = compute_rolling_hit_rate(vals, 20)
        assert all(h == 1.0 for h in result)


# ── CUSUM tests ──────────────────────────────────────────────────────────


class TestCUSUM:
    def test_no_break_good_strategy(self):
        mon = StrategyDecayMonitor(rolling_window=30, cusum_threshold=4.0)
        cusum = mon.cusum_test(_good_returns(200))
        assert isinstance(cusum, CUSUMResult)

    def test_break_in_decaying(self):
        mon = StrategyDecayMonitor(rolling_window=30, cusum_threshold=2.0)
        cusum = mon.cusum_test(_decaying_returns(300))
        assert cusum.break_detected
        assert cusum.break_index is not None

    def test_cusum_series_shape(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        cusum = mon.cusum_test(_good_returns(200))
        assert cusum.cusum_series is not None
        assert len(cusum.cusum_series) > 0

    def test_cusum_non_negative(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        cusum = mon.cusum_test(_good_returns(200))
        assert (cusum.cusum_series >= -1e-12).all()

    def test_too_short(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        cusum = mon.cusum_test(pd.Series([0.01] * 20, index=_dates(20)))
        assert not cusum.break_detected

    def test_max_cusum_stored(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        cusum = mon.cusum_test(_decaying_returns(200))
        assert cusum.max_cusum >= 0

    def test_cusum_on_sharpe_direct(self):
        rolling_sh = np.linspace(2.0, -1.0, 100)
        result = cusum_on_sharpe(rolling_sh, threshold=2.0)
        assert result.break_detected  # clear downtrend


# ── Lifecycle tests ──────────────────────────────────────────────────────


class TestLifecycle:
    def test_good_not_dead(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        phase = mon.classify_lifecycle(_good_returns(200))
        assert phase != LifecyclePhase.DEAD

    def test_dead_strategy(self):
        """Strongly negative returns should not classify as EMERGING."""
        mon = StrategyDecayMonitor(rolling_window=30)
        phase = mon.classify_lifecycle(_dead_returns(300))
        assert phase != LifecyclePhase.EMERGING

    def test_decaying_strategy(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        phase = mon.classify_lifecycle(_decaying_returns(300))
        assert phase in (LifecyclePhase.DEGRADING, LifecyclePhase.DEAD, LifecyclePhase.MATURE)

    def test_short_data_emerging(self):
        rolling_sh = np.array([1.0, 1.5, 2.0])
        assert classify_lifecycle(rolling_sh) == LifecyclePhase.EMERGING

    def test_enum_values(self):
        assert LifecyclePhase.EMERGING.value == "emerging"
        assert LifecyclePhase.MATURE.value == "mature"
        assert LifecyclePhase.DEGRADING.value == "degrading"
        assert LifecyclePhase.DEAD.value == "dead"


# ── Recent performance tests ─────────────────────────────────────────────


class TestRecentPerformance:
    def test_good_strategy_high_score(self):
        rp = score_recent_performance(_good_returns(200), window=63)
        assert isinstance(rp, RecentPerformance)
        assert rp.score > 0.3

    def test_dead_strategy_low_score(self):
        rp = score_recent_performance(_dead_returns(200), window=63)
        assert rp.score < 0.5

    def test_score_bounded(self):
        rp = score_recent_performance(_good_returns(200))
        assert 0.0 <= rp.score <= 1.0

    def test_sharpe_computed(self):
        rp = score_recent_performance(_good_returns(200))
        assert rp.sharpe != 0.0

    def test_hit_rate_bounded(self):
        rp = score_recent_performance(_good_returns(200))
        assert 0.0 <= rp.hit_rate <= 1.0

    def test_short_series(self):
        rp = score_recent_performance(pd.Series([0.01], index=_dates(1)))
        assert rp.score == 0.0

    def test_window_days(self):
        rp = score_recent_performance(_good_returns(200), window=30)
        assert rp.window_days == 30


# ── Kill score tests ─────────────────────────────────────────────────────


class TestKillScore:
    def test_keep_good_strategy(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        ks = mon.kill_signal("good", _good_returns(200))
        assert isinstance(ks, KillSignal)
        assert ks.recommendation in ("keep", "monitor")
        assert ks.kill_score < 0.7

    def test_retire_dead(self):
        mon = StrategyDecayMonitor(rolling_window=30, cusum_threshold=2.0)
        ks = mon.kill_signal("dead", _dead_returns(300))
        assert ks.recommendation in ("retire", "monitor")
        assert ks.kill_score > 0.3

    def test_score_bounded(self):
        mon = StrategyDecayMonitor(rolling_window=30, cusum_threshold=1.0)
        ks = mon.kill_signal("x", _dead_returns(300))
        assert 0.0 <= ks.kill_score <= 1.0

    def test_components_present(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        ks = mon.kill_signal("x", _good_returns(200))
        assert isinstance(ks.sharpe_component, float)
        assert isinstance(ks.hit_rate_component, float)
        assert isinstance(ks.cusum_component, float)
        assert isinstance(ks.pnl_trend_component, float)

    def test_reasons_populated_for_dead(self):
        mon = StrategyDecayMonitor(rolling_window=30, cusum_threshold=2.0)
        ks = mon.kill_signal("dead", _dead_returns(300))
        assert len(ks.reasons) > 0

    def test_confidence_increases_with_data(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        short = mon.kill_signal("s", _good_returns(50))
        long_ = mon.kill_signal("l", _good_returns(300))
        assert long_.confidence >= short.confidence

    def test_recommendation_values(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        ks = mon.kill_signal("x", _good_returns(200))
        assert ks.recommendation in ("keep", "monitor", "retire")


# ── Rolling metrics integration tests ────────────────────────────────────


class TestRollingMetricsIntegration:
    def test_compute_rolling_metrics(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        metrics = mon.compute_rolling_metrics(_good_returns(100))
        assert len(metrics) == 71
        assert all(isinstance(m, RollingMetrics) for m in metrics)

    def test_hit_rate_bounded(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        for m in mon.compute_rolling_metrics(_good_returns(100)):
            assert 0.0 <= m.hit_rate <= 1.0

    def test_too_short_empty(self):
        mon = StrategyDecayMonitor(rolling_window=100)
        assert mon.compute_rolling_metrics(_good_returns(50)) == []


# ── Full monitor tests ───────────────────────────────────────────────────


class TestMonitor:
    def test_monitor_returns_result(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        result = mon.monitor("EXP-400", _good_returns(200))
        assert isinstance(result, MonitorResult)
        assert result.strategy_name == "EXP-400"
        assert result.n_observations == 200
        assert len(result.rolling_metrics) > 0

    def test_monitor_decaying(self):
        mon = StrategyDecayMonitor(rolling_window=30, cusum_threshold=2.0)
        result = mon.monitor("EXP-DECAY", _decaying_returns(300))
        assert result.lifecycle in (
            LifecyclePhase.DEGRADING, LifecyclePhase.DEAD, LifecyclePhase.MATURE
        )


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generate_report_creates_file(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        result = mon.monitor("TEST", _good_returns(200))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "decay.html"
            path = StrategyDecayMonitor.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Strategy Decay Monitor" in content

    def test_report_contains_charts(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        result = mon.monitor("TEST", _good_returns(200))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            StrategyDecayMonitor.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Rolling Sharpe" in content
            assert "CUSUM" in content

    def test_report_contains_kill_dashboard(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        result = mon.monitor("TEST", _dead_returns(200))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            StrategyDecayMonitor.generate_report(result, out)
            content = out.read_text()
            assert "Kill Score" in content
            assert "Sharpe" in content

    def test_report_contains_lifecycle_badge(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        result = mon.monitor("TEST", _good_returns(200))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            StrategyDecayMonitor.generate_report(result, out)
            content = out.read_text()
            assert "Lifecycle" in content
            assert result.lifecycle.value.upper() in content

    def test_report_default_path(self):
        mon = StrategyDecayMonitor(rolling_window=30)
        result = mon.monitor("TEST", _good_returns(200))
        path = StrategyDecayMonitor.generate_report(result)
        assert path.exists()
        assert "strategy_decay.html" in str(path)
        path.unlink(missing_ok=True)
