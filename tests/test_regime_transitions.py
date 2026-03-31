"""Tests for compass/regime_transitions.py — regime transition detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.regime import Regime
from compass.regime_transitions import (
    EarlyWarning,
    RegimeSummary,
    RegimeTransitionDetector,
    TransitionEvent,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_regime_series(
    pattern: list[str], days_per: int = 20,
) -> pd.Series:
    """Build a regime series from a pattern like ["bull","bear","bull"]."""
    labels = []
    for regime in pattern:
        labels.extend([regime] * days_per)
    dates = pd.bdate_range("2020-01-02", periods=len(labels))
    return pd.Series(labels, index=dates, name="regime")


@pytest.fixture
def simple_series() -> pd.Series:
    return _make_regime_series(["bull", "bear", "bull", "high_vol", "bull"])


@pytest.fixture
def fitted_detector(simple_series) -> RegimeTransitionDetector:
    d = RegimeTransitionDetector()
    d.fit(simple_series)
    return d


@pytest.fixture
def long_series() -> pd.Series:
    """A more realistic multi-regime series."""
    pattern = [
        "bull", "bull", "high_vol", "crash", "bear",
        "bear", "bull", "low_vol", "bull", "high_vol",
    ]
    return _make_regime_series(pattern, days_per=30)


# ── TransitionEvent dataclass ────────────────────────────────────────────


class TestTransitionEvent:

    def test_creation(self):
        t = TransitionEvent(
            date=pd.Timestamp("2023-06-01"),
            from_regime="bull",
            to_regime="bear",
            duration_days=45,
        )
        assert t.from_regime == "bull"
        assert t.to_regime == "bear"
        assert t.duration_days == 45


# ── RegimeSummary ────────────────────────────────────────────────────────


class TestRegimeSummary:

    def test_creation(self):
        s = RegimeSummary(
            regime="bull", total_days=100, n_episodes=3,
            avg_duration=33.3, median_duration=30.0,
            min_duration=20, max_duration=50, pct_of_total=40.0,
        )
        assert s.regime == "bull"
        assert s.pct_of_total == 40.0


# ── fit() ────────────────────────────────────────────────────────────────


class TestFit:

    def test_fit_produces_transitions(self, simple_series):
        d = RegimeTransitionDetector()
        d.fit(simple_series)
        assert len(d.transitions) == 4  # bull→bear, bear→bull, bull→high_vol, high_vol→bull

    def test_transition_matrix_shape(self, fitted_detector):
        m = fitted_detector.transition_matrix
        assert m.shape == (5, 5)  # 5 regime types
        assert list(m.index) == [r.value for r in [Regime.BULL, Regime.BEAR, Regime.HIGH_VOL, Regime.LOW_VOL, Regime.CRASH]]

    def test_transition_matrix_rows_sum_to_one(self, fitted_detector):
        m = fitted_detector.transition_matrix
        for regime in m.index:
            row_sum = m.loc[regime].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 0.01, f"{regime} row sums to {row_sum}"

    def test_transition_counts_match_events(self, fitted_detector):
        total_counts = fitted_detector.transition_counts.values.sum()
        assert total_counts == len(fitted_detector.transitions)

    def test_bull_to_bear_probability(self, fitted_detector):
        # In simple_series: bull→bear (1), bull→high_vol (1) = 2 transitions from bull
        p = fitted_detector.get_transition_for("bull", "bear")
        assert p == pytest.approx(0.5, abs=0.01)

    def test_regime_summaries_populated(self, fitted_detector):
        assert len(fitted_detector.regime_summaries) == 5
        bull = next(s for s in fitted_detector.regime_summaries if s.regime == "bull")
        assert bull.n_episodes >= 2
        assert bull.total_days > 0

    def test_pct_of_total_sums_to_100(self, fitted_detector):
        total = sum(s.pct_of_total for s in fitted_detector.regime_summaries)
        assert abs(total - 100.0) < 1.0

    def test_fit_with_enum_values(self):
        dates = pd.bdate_range("2023-01-02", periods=40)
        labels = [Regime.BULL] * 20 + [Regime.BEAR] * 20
        s = pd.Series(labels, index=dates, name="regime")
        d = RegimeTransitionDetector()
        d.fit(s)
        assert len(d.transitions) == 1
        assert d.transitions[0].from_regime == "bull"
        assert d.transitions[0].to_regime == "bear"

    def test_fit_single_regime_no_transitions(self):
        s = _make_regime_series(["bull"], days_per=100)
        d = RegimeTransitionDetector()
        d.fit(s)
        assert len(d.transitions) == 0
        assert d.transition_counts.values.sum() == 0

    def test_fit_empty_series(self):
        s = pd.Series(dtype=str)
        d = RegimeTransitionDetector()
        d.fit(s)
        assert d._fitted
        assert len(d.transitions) == 0

    def test_fit_returns_self(self, simple_series):
        d = RegimeTransitionDetector()
        result = d.fit(simple_series)
        assert result is d

    def test_transition_dates_chronological(self, fitted_detector):
        dates = [t.date for t in fitted_detector.transitions]
        assert dates == sorted(dates)

    def test_duration_positive(self, fitted_detector):
        for t in fitted_detector.transitions:
            assert t.duration_days > 0

    def test_long_series_many_transitions(self, long_series):
        d = RegimeTransitionDetector()
        d.fit(long_series)
        # Pattern has 10 segments but consecutive same-regimes merge,
        # so expect ~7 transitions (bull-bull and bear-bear don't count)
        assert len(d.transitions) >= 5


# ── get_transition_for() ─────────────────────────────────────────────────


class TestGetTransitionFor:

    def test_known_transition(self, fitted_detector):
        p = fitted_detector.get_transition_for("bull", "bear")
        assert 0.0 <= p <= 1.0

    def test_unknown_regime_returns_zero(self, fitted_detector):
        assert fitted_detector.get_transition_for("nonexistent", "bull") == 0.0

    def test_unfitted_returns_zero(self):
        d = RegimeTransitionDetector()
        assert d.get_transition_for("bull", "bear") == 0.0


# ── early_warning() ──────────────────────────────────────────────────────


class TestEarlyWarning:

    def test_high_vix_higher_severity_than_low_vix(self, fitted_detector):
        """Higher VIX in bull regime should produce higher warning severity."""
        w_calm = fitted_detector.early_warning(
            current_regime="bull", current_vix=15.0, vix_5d_change=0.0,
        )
        w_stressed = fitted_detector.early_warning(
            current_regime="bull", current_vix=35.0, vix_5d_change=+8.0,
        )
        # Stressed conditions should produce warning with higher probability
        assert w_stressed is not None
        calm_prob = w_calm.probability if w_calm else 0.0
        assert w_stressed.probability >= calm_prob

    def test_warning_on_vix_spike(self, fitted_detector):
        w = fitted_detector.early_warning(
            current_regime="bull", current_vix=30.0, vix_5d_change=+8.0,
        )
        assert w is not None
        assert w.severity in ("medium", "high")
        assert w.probability > 0.15

    def test_warning_contains_trigger_reason(self, fitted_detector):
        w = fitted_detector.early_warning(
            current_regime="bull", current_vix=28.0, vix_5d_change=+6.0,
        )
        assert w is not None
        assert len(w.trigger_reason) > 0

    def test_warning_probability_capped_at_one(self, fitted_detector):
        w = fitted_detector.early_warning(
            current_regime="bull", current_vix=50.0, vix_5d_change=+15.0,
            regime_duration_days=500,
        )
        if w is not None:
            assert w.probability <= 1.0

    def test_bear_to_bull_recovery_signal(self, fitted_detector):
        w = fitted_detector.early_warning(
            current_regime="bear", current_vix=18.0, vix_5d_change=-5.0,
        )
        if w is not None:
            assert w.likely_next != "bear"

    def test_unfitted_returns_none(self):
        d = RegimeTransitionDetector()
        assert d.early_warning(current_regime="bull") is None

    def test_duration_exhaustion(self, fitted_detector):
        w = fitted_detector.early_warning(
            current_regime="bull", current_vix=18.0, vix_5d_change=0.0,
            regime_duration_days=200,
        )
        if w is not None:
            assert "Duration" in w.trigger_reason or "exceeds" in w.trigger_reason


# ── generate_report() ────────────────────────────────────────────────────


class TestGenerateReport:

    def test_returns_html(self, fitted_detector, simple_series):
        html = fitted_detector.generate_report(simple_series)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_contains_title(self, fitted_detector, simple_series):
        html = fitted_detector.generate_report(simple_series)
        assert "Regime Transition Analysis" in html

    def test_contains_timeline(self, fitted_detector, simple_series):
        html = fitted_detector.generate_report(simple_series)
        assert "timeline" in html

    def test_contains_transition_matrix(self, fitted_detector, simple_series):
        html = fitted_detector.generate_report(simple_series)
        assert "Transition Probability Matrix" in html

    def test_contains_duration_summary(self, fitted_detector, simple_series):
        html = fitted_detector.generate_report(simple_series)
        assert "Duration Summary" in html

    def test_contains_recent_transitions(self, fitted_detector, simple_series):
        html = fitted_detector.generate_report(simple_series)
        assert "Recent Transitions" in html

    def test_regime_colors_in_timeline(self, fitted_detector, simple_series):
        html = fitted_detector.generate_report(simple_series)
        assert "#22c55e" in html  # bull green
        assert "#ef4444" in html  # bear red

    def test_auto_fits_if_needed(self, simple_series):
        d = RegimeTransitionDetector()
        html = d.generate_report(simple_series)
        assert d._fitted
        assert "Transition" in html

    def test_writes_to_file(self, fitted_detector, simple_series):
        html = fitted_detector.generate_report(simple_series)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            path.write_text(html)
            assert path.exists()
            assert path.stat().st_size > 2000

    def test_handles_spy_and_vix_overlay(self, fitted_detector, simple_series):
        n = len(simple_series)
        spy = pd.Series(np.linspace(400, 450, n), index=simple_series.index)
        vix = pd.Series(np.linspace(15, 30, n), index=simple_series.index)
        html = fitted_detector.generate_report(simple_series, spy_close=spy, vix_series=vix)
        assert len(html) > 2000


# ── Integration with real training data ──────────────────────────────────


class TestRealDataIntegration:

    @pytest.fixture
    def real_regime_series(self):
        csv = Path(__file__).resolve().parent.parent / "compass" / "training_data_combined.csv"
        if not csv.exists():
            pytest.skip("training data not available")
        df = pd.read_csv(csv, parse_dates=["entry_date"])
        if "regime" not in df.columns:
            pytest.skip("no regime column")
        # Build a daily regime series from trade-level data
        regime_by_date = df.set_index("entry_date")["regime"].sort_index()
        # Deduplicate: one regime per date (take first)
        regime_by_date = regime_by_date[~regime_by_date.index.duplicated(keep="first")]
        return regime_by_date

    def test_fit_on_real_data(self, real_regime_series):
        d = RegimeTransitionDetector()
        d.fit(real_regime_series)
        assert len(d.transitions) >= 5
        assert d.transition_matrix.shape == (5, 5)

    def test_bull_is_dominant_regime(self, real_regime_series):
        d = RegimeTransitionDetector()
        d.fit(real_regime_series)
        bull = next(s for s in d.regime_summaries if s.regime == "bull")
        assert bull.pct_of_total > 50  # bull should dominate

    def test_report_on_real_data(self, real_regime_series):
        d = RegimeTransitionDetector()
        d.fit(real_regime_series)
        html = d.generate_report(real_regime_series)
        assert "bull" in html
        assert "bear" in html
        assert len(html) > 5000
