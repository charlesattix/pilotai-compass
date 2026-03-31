"""Tests for compass.signal_decay – signal decay analyzer."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.signal_decay import (
    DecayAnalysis,
    HalfLifeResult,
    ICResult,
    RegimeDecayResult,
    SignalDecayAnalyzer,
    TurnoverResult,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_signals(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(rng.randn(n), index=idx, name="signal")


def _make_returns(signals: pd.Series, noise: float = 0.5, seed: int = 99) -> pd.Series:
    rng = np.random.RandomState(seed)
    return signals * 0.02 + rng.randn(len(signals)) * noise * 0.02


def _make_regimes(signals: pd.Series) -> pd.Series:
    n = len(signals)
    labels = ["bull", "bear", "high_vol"]
    regime_arr = [labels[i % len(labels)] for i in range(n)]
    return pd.Series(regime_arr, index=signals.index, name="regime")


# ── Constructor ─────────────────────────────────────────────────────────────
class TestSignalDecayAnalyzerInit:
    def test_default_periods(self):
        a = SignalDecayAnalyzer()
        assert "1h" in a.holding_periods
        assert "5d" in a.holding_periods
        assert len(a.holding_periods) == 5

    def test_custom_periods(self):
        a = SignalDecayAnalyzer(holding_periods={"2h": 2, "8h": 8})
        assert len(a.holding_periods) == 2
        assert a.holding_periods["2h"] == 2

    def test_default_cost(self):
        a = SignalDecayAnalyzer()
        assert a.cost_per_flip_bps == 2.0

    def test_custom_cost(self):
        a = SignalDecayAnalyzer(cost_per_flip_bps=5.0)
        assert a.cost_per_flip_bps == 5.0


# ── IC computation ──────────────────────────────────────────────────────────
class TestICComputation:
    def test_ic_results_count(self):
        analyzer = SignalDecayAnalyzer()
        signals = _make_signals()
        returns = _make_returns(signals)
        result = analyzer.analyze(signals, returns)
        assert len(result.ic_results) == 5

    def test_ic_values_bounded(self):
        analyzer = SignalDecayAnalyzer()
        signals = _make_signals()
        returns = _make_returns(signals)
        result = analyzer.analyze(signals, returns)
        for ic_r in result.ic_results:
            assert -1.0 <= ic_r.ic <= 1.0

    def test_ic_with_correlated_signal(self):
        """Strongly correlated signal should yield positive IC at some horizon."""
        signals = _make_signals(n=2000)
        returns = _make_returns(signals, noise=0.05)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        max_ic = max(r.ic for r in result.ic_results)
        assert max_ic > 0.0

    def test_ic_result_fields(self):
        analyzer = SignalDecayAnalyzer()
        signals = _make_signals()
        returns = _make_returns(signals)
        result = analyzer.analyze(signals, returns)
        ic_r = result.ic_results[0]
        assert isinstance(ic_r.period_label, str)
        assert isinstance(ic_r.period_hours, int)
        assert isinstance(ic_r.n_obs, int)
        assert ic_r.n_obs > 0

    def test_ic_ir_sign_matches_ic(self):
        signals = _make_signals(n=1000)
        returns = _make_returns(signals, noise=0.1)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        for ic_r in result.ic_results:
            if abs(ic_r.ic_std) > 1e-9 and abs(ic_r.ic) > 0.01:
                assert (ic_r.ic_ir >= 0) == (ic_r.ic >= 0)


# ── SNR ─────────────────────────────────────────────────────────────────────
class TestSNR:
    def test_snr_nonnegative(self):
        signals = _make_signals()
        returns = _make_returns(signals)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert result.snr >= 0.0

    def test_snr_zero_for_zero_mean(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="h")
        signals = pd.Series([1, -1] * 50, index=idx)
        returns = pd.Series(np.ones(100) * 0.01, index=idx)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert result.snr == pytest.approx(0.0, abs=1e-9)

    def test_snr_constant_signal(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="h")
        signals = pd.Series(np.ones(100) * 5.0, index=idx)
        returns = pd.Series(np.ones(100) * 0.01, index=idx)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        # std ≈ 0 → snr = 0 (guarded)
        assert result.snr == 0.0


# ── Optimal period ──────────────────────────────────────────────────────────
class TestOptimalPeriod:
    def test_optimal_period_in_labels(self):
        signals = _make_signals()
        returns = _make_returns(signals)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert result.optimal_period in ["1h", "4h", "1d", "2d", "5d"]

    def test_optimal_ic_matches(self):
        signals = _make_signals()
        returns = _make_returns(signals)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        best = max(result.ic_results, key=lambda r: abs(r.ic))
        assert result.optimal_period == best.period_label
        assert result.optimal_ic == pytest.approx(best.ic)


# ── Turnover ────────────────────────────────────────────────────────────────
class TestTurnover:
    def test_flip_rate_bounded(self):
        signals = _make_signals()
        returns = _make_returns(signals)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert 0.0 <= result.turnover.flip_rate <= 1.0

    def test_cost_scales_with_flip_rate(self):
        signals = _make_signals()
        returns = _make_returns(signals)
        low_cost = SignalDecayAnalyzer(cost_per_flip_bps=1.0).analyze(signals, returns)
        high_cost = SignalDecayAnalyzer(cost_per_flip_bps=10.0).analyze(signals, returns)
        assert high_cost.turnover.estimated_cost_bps > low_cost.turnover.estimated_cost_bps

    def test_all_same_sign_no_flips(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="h")
        signals = pd.Series(np.ones(200), index=idx)
        returns = pd.Series(np.ones(200) * 0.01, index=idx)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert result.turnover.flip_rate == 0.0

    def test_alternating_signal_max_flips(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="h")
        signals = pd.Series([1, -1] * 100, index=idx)
        returns = pd.Series(np.ones(200) * 0.01, index=idx)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert result.turnover.flip_rate == pytest.approx(1.0, abs=0.01)


# ── Half-life ───────────────────────────────────────────────────────────────
class TestHalfLife:
    def test_half_life_positive(self):
        signals = _make_signals(n=1000)
        returns = _make_returns(signals, noise=0.3)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert result.half_life is not None
        assert result.half_life.half_life_hours > 0

    def test_half_life_r_squared_bounded(self):
        signals = _make_signals(n=1000)
        returns = _make_returns(signals, noise=0.3)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert 0.0 <= result.half_life.r_squared <= 1.0

    def test_few_periods_returns_inf(self):
        analyzer = SignalDecayAnalyzer(holding_periods={"1h": 1})
        signals = _make_signals(n=100)
        returns = _make_returns(signals)
        result = analyzer.analyze(signals, returns)
        # only one valid point → can't fit
        assert result.half_life.half_life_hours == float("inf")


# ── Per-regime ──────────────────────────────────────────────────────────────
class TestRegimeBreakdown:
    def test_regime_results_present(self):
        signals = _make_signals(n=600)
        returns = _make_returns(signals)
        regimes = _make_regimes(signals)
        result = SignalDecayAnalyzer().analyze(signals, returns, regimes=regimes)
        assert len(result.regime_results) > 0

    def test_regime_fields(self):
        signals = _make_signals(n=600)
        returns = _make_returns(signals)
        regimes = _make_regimes(signals)
        result = SignalDecayAnalyzer().analyze(signals, returns, regimes=regimes)
        rr = result.regime_results[0]
        assert isinstance(rr.regime, str)
        assert isinstance(rr.ic_by_period, dict)
        assert rr.n_obs > 0

    def test_no_regimes_empty_list(self):
        signals = _make_signals()
        returns = _make_returns(signals)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert result.regime_results == []


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            signals = _make_signals()
            returns = _make_returns(signals)
            analyzer = SignalDecayAnalyzer()
            analysis = analyzer.analyze(signals, returns)
            path = analyzer.generate_report(analysis, output_path=Path(tmp) / "test.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_contains_key_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            signals = _make_signals(n=600)
            returns = _make_returns(signals)
            regimes = _make_regimes(signals)
            analyzer = SignalDecayAnalyzer()
            analysis = analyzer.analyze(signals, returns, regimes=regimes)
            path = analyzer.generate_report(analysis, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Signal Decay Analysis" in html
            assert "IC Decay" in html
            assert "SNR" in html
            assert "Half-Life" in html
            assert "Per-Regime" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            signals = _make_signals()
            returns = _make_returns(signals)
            analyzer = SignalDecayAnalyzer()
            analysis = analyzer.analyze(signals, returns)
            path = analyzer.generate_report(analysis, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Edge cases ──────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_too_few_observations(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        signals = pd.Series([1, 2, 3, 4, 5], index=idx)
        returns = pd.Series([0.01] * 5, index=idx)
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert result.ic_results == []
        assert result.optimal_period == ""

    def test_misaligned_index(self):
        idx_a = pd.date_range("2024-01-01", periods=300, freq="h")
        idx_b = pd.date_range("2024-01-05", periods=300, freq="h")
        signals = pd.Series(np.random.randn(300), index=idx_a)
        returns = pd.Series(np.random.randn(300) * 0.01, index=idx_b)
        # overlapping portion should be used
        result = SignalDecayAnalyzer().analyze(signals, returns)
        assert isinstance(result, DecayAnalysis)


# ── Dataclass construction ──────────────────────────────────────────────────
class TestDataclasses:
    def test_ic_result_dataclass(self):
        r = ICResult("1h", 1, 0.05, 0.02, 2.5, 100)
        assert r.period_label == "1h"
        assert r.ic == 0.05

    def test_turnover_result_dataclass(self):
        t = TurnoverResult(flip_rate=0.3, avg_holding_bars=3.3, estimated_cost_bps=0.6)
        assert t.flip_rate == 0.3

    def test_half_life_result_dataclass(self):
        h = HalfLifeResult(half_life_hours=12.0, decay_rate=0.058, r_squared=0.95)
        assert h.half_life_hours == 12.0

    def test_decay_analysis_defaults(self):
        d = DecayAnalysis()
        assert d.ic_results == []
        assert d.snr == 0.0
        assert d.regime_results == []
