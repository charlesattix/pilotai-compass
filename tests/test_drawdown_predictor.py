"""Tests for compass.drawdown_predictor — 38 tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.drawdown_predictor import (
    DrawdownPredictor,
    DrawdownSeverity,
    DrawdownEpisode,
    RecoveryEstimate,
    RegimeRecoveryProfile,
    MonteCarloRecovery,
    EarlyWarning,
    DrawdownPrediction,
    SEVERITY_THRESHOLDS,
    SEVERITY_SIZE_MULT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(n: int = 300) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=n)


def _equity_stable(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.003, n)
    return pd.Series(100000 * np.cumprod(1 + r), index=_dates(n))


def _equity_with_dd(n: int = 300, seed: int = 77) -> pd.Series:
    """Equity with a ~12% drawdown mid-series then recovery."""
    rng = np.random.default_rng(seed)
    a, b = int(n * 0.35), int(n * 0.25)
    c = n - a - b
    r = np.concatenate([
        rng.normal(0.0005, 0.003, a),
        rng.normal(-0.004, 0.006, b),
        rng.normal(0.001, 0.003, c),
    ])
    return pd.Series(100000 * np.cumprod(1 + r), index=_dates(n))


def _regimes(n: int = 300) -> pd.Series:
    labels = []
    for i in range(n):
        if i < n * 0.35:
            labels.append("bull")
        elif i < n * 0.6:
            labels.append("high_vol")
        else:
            labels.append("bull")
    return pd.Series(labels, index=_dates(n))


# ===========================================================================
# Severity classification
# ===========================================================================

class TestSeverity:
    def test_shallow(self):
        assert DrawdownPredictor.classify_severity(0.03) == DrawdownSeverity.SHALLOW

    def test_moderate(self):
        assert DrawdownPredictor.classify_severity(0.07) == DrawdownSeverity.MODERATE

    def test_deep(self):
        assert DrawdownPredictor.classify_severity(0.15) == DrawdownSeverity.DEEP

    def test_catastrophic(self):
        assert DrawdownPredictor.classify_severity(0.25) == DrawdownSeverity.CATASTROPHIC

    def test_boundary(self):
        assert DrawdownPredictor.classify_severity(0.05) == DrawdownSeverity.MODERATE


# ===========================================================================
# Episode detection
# ===========================================================================

class TestEpisodes:
    def test_detects_episodes(self):
        eq = _equity_with_dd(300)
        eps = DrawdownPredictor.detect_episodes(eq, min_drawdown=0.02)
        assert len(eps) >= 1
        assert all(isinstance(e, DrawdownEpisode) for e in eps)

    def test_no_episodes_in_stable(self):
        eq = _equity_stable(100)
        eps = DrawdownPredictor.detect_episodes(eq, min_drawdown=0.10)
        assert len(eps) == 0

    def test_severity_assigned(self):
        eq = _equity_with_dd(300)
        eps = DrawdownPredictor.detect_episodes(eq, min_drawdown=0.02)
        for e in eps:
            assert isinstance(e.severity, DrawdownSeverity)

    def test_with_regimes(self):
        eq = _equity_with_dd(300)
        reg = _regimes(300)
        eps = DrawdownPredictor.detect_episodes(eq, min_drawdown=0.02, regimes=reg)
        regime_eps = [e for e in eps if e.regime_at_trough is not None]
        assert len(regime_eps) >= 1

    def test_empty_equity(self):
        assert DrawdownPredictor.detect_episodes(pd.Series(dtype=float)) == []

    def test_ongoing_drawdown(self):
        rng = np.random.default_rng(99)
        r = np.concatenate([
            rng.normal(0.001, 0.003, 50),
            rng.normal(-0.005, 0.005, 50),  # still falling
        ])
        eq = pd.Series(100000 * np.cumprod(1 + r), index=_dates(100))
        eps = DrawdownPredictor.detect_episodes(eq, min_drawdown=0.02)
        ongoing = [e for e in eps if e.recovery_date is None]
        assert len(ongoing) >= 1


# ===========================================================================
# Recovery estimation
# ===========================================================================

class TestRecovery:
    def test_basic(self):
        eq = _equity_with_dd(300)
        eps = DrawdownPredictor.detect_episodes(eq, min_drawdown=0.02)
        rec = DrawdownPredictor.estimate_recovery(eps, 0.08)
        assert isinstance(rec, RecoveryEstimate)
        assert rec.expected_days > 0

    def test_no_analogues(self):
        rec = DrawdownPredictor.estimate_recovery([], 0.10)
        assert rec.n_analogues == 0
        assert rec.expected_days > 0  # rough fallback

    def test_size_multiplier(self):
        rec = DrawdownPredictor.estimate_recovery([], 0.15)
        assert rec.size_multiplier == SEVERITY_SIZE_MULT[DrawdownSeverity.DEEP]


# ===========================================================================
# Regime recovery profiles
# ===========================================================================

class TestRegimeProfiles:
    def test_basic(self):
        eq = _equity_with_dd(300)
        reg = _regimes(300)
        eps = DrawdownPredictor.detect_episodes(eq, min_drawdown=0.02, regimes=reg)
        profiles = DrawdownPredictor.regime_recovery_profiles(eps)
        assert isinstance(profiles, list)

    def test_empty(self):
        assert DrawdownPredictor.regime_recovery_profiles([]) == []


# ===========================================================================
# Monte Carlo
# ===========================================================================

class TestMonteCarlo:
    def test_basic(self):
        dp = DrawdownPredictor(mc_simulations=500, mc_horizon=120)
        mc = dp.monte_carlo_recovery(0.08, 0.0005, 0.01)
        assert isinstance(mc, MonteCarloRecovery)
        assert mc.n_simulations == 500
        assert mc.expected_days > 0

    def test_probabilities(self):
        dp = DrawdownPredictor(mc_simulations=1000, mc_horizon=120)
        mc = dp.monte_carlo_recovery(0.05, 0.001, 0.01)
        assert 0 <= mc.prob_recover_30d <= 1
        assert mc.prob_recover_30d <= mc.prob_recover_60d <= mc.prob_recover_90d

    def test_confidence_bands(self):
        dp = DrawdownPredictor(mc_simulations=500, mc_horizon=60)
        mc = dp.monte_carlo_recovery(0.05, 0.001, 0.01)
        assert "p50" in mc.confidence_bands
        assert len(mc.confidence_bands["p50"]) == 60

    def test_small_dd_fast_recovery(self):
        dp = DrawdownPredictor(mc_simulations=1000, mc_horizon=120)
        small = dp.monte_carlo_recovery(0.01, 0.001, 0.005)
        large = dp.monte_carlo_recovery(0.15, 0.001, 0.005)
        assert small.expected_days < large.expected_days


# ===========================================================================
# Early warnings
# ===========================================================================

class TestWarnings:
    def test_no_warnings_stable(self):
        dp = DrawdownPredictor()
        eq = _equity_stable(100)
        warns = dp.early_warnings(eq)
        triggered = [w for w in warns if w.triggered]
        assert len(triggered) == 0

    def test_velocity_warning(self):
        dp = DrawdownPredictor(velocity_window=5)
        rng = np.random.default_rng(99)
        r = np.concatenate([
            rng.normal(0.001, 0.003, 30),
            rng.normal(-0.01, 0.005, 20),  # sharp drop
        ])
        eq = pd.Series(100000 * np.cumprod(1 + r), index=_dates(50))
        warns = dp.early_warnings(eq)
        vel_warns = [w for w in warns if w.signal_type == "velocity"]
        assert len(vel_warns) >= 1

    def test_breadth_warning(self):
        dp = DrawdownPredictor(breadth_threshold=0.5)
        eq = _equity_stable(50)
        rng = np.random.default_rng(88)
        strat_eq = {
            "A": pd.Series(100000 * np.cumprod(1 + rng.normal(-0.003, 0.005, 50)), index=_dates(50)),
            "B": pd.Series(100000 * np.cumprod(1 + rng.normal(-0.003, 0.005, 50)), index=_dates(50)),
            "C": pd.Series(100000 * np.cumprod(1 + rng.normal(0.001, 0.003, 50)), index=_dates(50)),
        }
        warns = dp.early_warnings(eq, strategy_equities=strat_eq)
        breadth = [w for w in warns if w.signal_type == "breadth"]
        assert len(breadth) == 1

    def test_empty(self):
        dp = DrawdownPredictor()
        assert dp.early_warnings(pd.Series(dtype=float)) == []


# ===========================================================================
# Size multiplier
# ===========================================================================

class TestSizing:
    def test_shallow(self):
        assert DrawdownPredictor.size_multiplier(0.03) == 1.0

    def test_catastrophic(self):
        assert DrawdownPredictor.size_multiplier(0.25) == 0.10


# ===========================================================================
# Full prediction
# ===========================================================================

class TestPredict:
    def test_basic(self):
        dp = DrawdownPredictor(mc_simulations=200, mc_horizon=60)
        eq = _equity_with_dd(200)
        pred = dp.predict(eq)
        assert isinstance(pred, DrawdownPrediction)
        assert pred.severity is not None
        assert pred.recovery is not None

    def test_empty(self):
        dp = DrawdownPredictor()
        pred = dp.predict(pd.Series(dtype=float), run_mc=False)
        assert pred.current_drawdown == 0.0


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        dp = DrawdownPredictor(mc_simulations=200, mc_horizon=60)
        eq = _equity_with_dd(200)
        pred = dp.predict(eq)
        out = tmp_path / "dd_pred.html"
        result = dp.generate_report(pred, equity=eq, output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Drawdown Recovery Predictor" in html

    def test_contains_charts(self, tmp_path):
        dp = DrawdownPredictor(mc_simulations=200, mc_horizon=60)
        eq = _equity_with_dd(200)
        pred = dp.predict(eq)
        out = tmp_path / "dd.html"
        dp.generate_report(pred, equity=eq, output_path=str(out))
        html = out.read_text()
        assert "<svg" in html

    def test_contains_warnings(self, tmp_path):
        dp = DrawdownPredictor(mc_simulations=100, mc_horizon=30)
        eq = _equity_with_dd(200)
        pred = dp.predict(eq)
        out = tmp_path / "dd.html"
        dp.generate_report(pred, equity=eq, output_path=str(out))
        html = out.read_text()
        assert "Early Warning" in html
