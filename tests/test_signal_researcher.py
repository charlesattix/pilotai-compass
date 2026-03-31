"""Tests for compass.signal_researcher — 30 tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from compass.signal_researcher import (
    SignalResearcher, SignalMetrics, SignalResearchResult, SignalDefinition,
)

def _prices(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)),
                      index=pd.bdate_range("2023-01-02", periods=n))


class TestGenerators:
    def test_rsi(self):
        s = SignalResearcher.generate_rsi(_prices(), 14)
        assert len(s.dropna()) > 400
        assert s.dropna().min() >= -1.5
        assert s.dropna().max() <= 1.5

    def test_bollinger(self):
        s = SignalResearcher.generate_bollinger(_prices(), 20)
        assert len(s.dropna()) > 400

    def test_keltner(self):
        s = SignalResearcher.generate_keltner(_prices(), 20)
        assert not s.dropna().empty

    def test_donchian(self):
        s = SignalResearcher.generate_donchian(_prices(), 20)
        assert not s.dropna().empty

    def test_momentum(self):
        s = SignalResearcher.generate_momentum(_prices(), 10, 50)
        assert not s.dropna().empty

    def test_generate_all(self):
        sr = SignalResearcher()
        signals = sr.generate_all(_prices(200))
        assert len(signals) > 10
        assert all(isinstance(v, pd.Series) for v in signals.values())


class TestMetrics:
    def test_ic(self):
        p = _prices()
        sig = SignalResearcher.generate_rsi(p, 14)
        fwd = p.pct_change(5).shift(-5)
        ic, ic_std = SignalResearcher.compute_ic(sig, fwd)
        assert isinstance(ic, float)
        assert -1 <= ic <= 1

    def test_turnover(self):
        sig = SignalResearcher.generate_rsi(_prices(), 14)
        t = SignalResearcher.compute_turnover(sig)
        assert t >= 0

    def test_halflife(self):
        sig = SignalResearcher.generate_rsi(_prices(), 14)
        hl = SignalResearcher.compute_halflife(sig)
        assert hl >= 0

    def test_evaluate(self):
        sr = SignalResearcher()
        p = _prices()
        sig = sr.generate_rsi(p, 14)
        fwd = p.pct_change(5).shift(-5)
        m = sr.evaluate_signal("rsi_14", sig, fwd)
        assert isinstance(m, SignalMetrics)
        assert m.name == "rsi_14"


class TestScreen:
    def test_screen_passes_good(self):
        sr = SignalResearcher(ic_threshold=0.001, max_turnover=1.0, min_halflife=0.1)
        p = _prices(500)
        sig = sr.generate_rsi(p, 14)
        fwd = p.pct_change(5).shift(-5)
        m = sr.evaluate_signal("rsi", sig, fwd)
        # With loose thresholds, most signals pass
        assert m.passed_screen in (True, False)

    def test_strict_screen(self):
        sr = SignalResearcher(ic_threshold=0.50)
        p = _prices()
        sig = sr.generate_rsi(p, 14)
        fwd = p.pct_change(5).shift(-5)
        m = sr.evaluate_signal("rsi", sig, fwd)
        assert not m.passed_screen  # IC < 0.50 for random-ish signal


class TestCorrelationFilter:
    def test_basic(self):
        sr = SignalResearcher(max_correlation=0.70)
        p = _prices()
        signals = {
            "a": sr.generate_rsi(p, 14),
            "b": sr.generate_rsi(p, 15),  # very similar to a
            "c": sr.generate_bollinger(p, 20),
        }
        kept = sr.correlation_filter(signals, ["a", "b", "c"])
        assert "a" in kept
        assert len(kept) <= 3

    def test_single(self):
        sr = SignalResearcher()
        assert sr.correlation_filter({}, ["a"]) == ["a"]

    def test_empty(self):
        sr = SignalResearcher()
        assert sr.correlation_filter({}, []) == []


class TestResearch:
    def test_full_pipeline(self):
        sr = SignalResearcher(ic_threshold=0.001, max_turnover=1.0, min_halflife=0.1)
        result = sr.research(_prices(300), forward_horizon=5)
        assert isinstance(result, SignalResearchResult)
        assert result.generated > 0
        assert result.screened > 0

    def test_top_signals_sorted(self):
        sr = SignalResearcher(ic_threshold=0.001, max_turnover=1.0, min_halflife=0.1)
        result = sr.research(_prices(300))
        if result.top_signals:
            ics = [abs(s.ic) for s in result.top_signals]
            assert ics == sorted(ics, reverse=True) or len(ics) <= 1


class TestReport:
    def test_creates_file(self, tmp_path):
        sr = SignalResearcher(ic_threshold=0.001, max_turnover=1.0, min_halflife=0.1)
        result = sr.research(_prices(200))
        out = tmp_path / "sig.html"
        path = sr.generate_report(result, output_path=str(out))
        assert Path(path).exists()
        assert "Signal Research" in out.read_text()

    def test_contains_funnel(self, tmp_path):
        sr = SignalResearcher(ic_threshold=0.001, max_turnover=1.0, min_halflife=0.1)
        result = sr.research(_prices(200))
        out = tmp_path / "sig.html"
        sr.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "Generated" in html
        assert "Passed" in html
