"""Tests for compass/cross_asset_signals.py — cross-asset signal generator.

Covers:
  - Dataclass construction
  - ADF test implementation
  - Inter-market correlations
  - Lead-lag detection
  - Cointegration testing (Engle-Granger)
  - Spread trading signals (z-score)
  - Cross-asset momentum
  - Macro regime signals
  - Signal dashboard
  - from_csv constructor
  - HTML report generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.cross_asset_signals import (
    CointegrationResult,
    CrossAssetSignalGenerator,
    InterMarketCorrelation,
    LeadLagResult,
    MacroRegimeSignal,
    MomentumSignal,
    SignalDashboard,
    SpreadSignal,
    _adf_test,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_prices(n=300, assets=5, seed=42):
    """Generate synthetic multi-asset price series with known structure."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    names = ["SPY", "TLT", "GLD", "UUP", "HYG"][:assets]

    # Random walk prices with some cross-correlation
    common = rng.normal(0, 0.005, n).cumsum()
    data = {}
    for i, name in enumerate(names):
        noise = rng.normal(0, 0.008, n).cumsum()
        corr_factor = 0.3 if i % 2 == 0 else -0.2  # some positive, some negative
        level = 100 + i * 50
        data[name] = level + common * corr_factor * level + noise * level * 0.1

    return pd.DataFrame(data, index=dates)


def _make_cointegrated_prices(n=300, seed=42):
    """Generate two cointegrated price series."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    # Common stochastic trend
    trend = rng.normal(0, 0.5, n).cumsum() + 100
    # Asset A follows trend
    a = trend + rng.normal(0, 0.3, n)
    # Asset B = 1.5 * trend + stationary noise
    b = 1.5 * trend + rng.normal(0, 0.5, n)
    # Add a non-cointegrated asset
    c = rng.normal(0, 0.5, n).cumsum() + 200
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)


def _make_macro(index):
    """Generate macro indicators."""
    rng = np.random.RandomState(42)
    n = len(index)
    return pd.DataFrame({
        "vix_term": rng.normal(0.05, 0.02, n),
        "yield_curve": rng.normal(0.5, 0.3, n),
        "credit_spread": rng.normal(1.5, 0.5, n),
    }, index=index)


def _make_generator(n=300, assets=5, seed=42, **kwargs):
    prices = _make_prices(n=n, assets=assets, seed=seed)
    return CrossAssetSignalGenerator(prices, **kwargs)


# ── Dataclass tests ──────────────────────────────────────────────────────


class TestDataclasses:
    def test_inter_market_correlation_fields(self):
        c = InterMarketCorrelation(
            asset_a="SPY", asset_b="TLT", correlation=0.3,
            rolling_corr_mean=0.25, rolling_corr_std=0.1,
            current_corr=0.4, z_score=1.5,
        )
        assert c.z_score == pytest.approx(1.5)

    def test_lead_lag_result_fields(self):
        ll = LeadLagResult(
            leader="SPY", lagger="TLT", optimal_lag=3,
            correlation_at_lag=0.3, p_value=0.01, direction="positive",
        )
        assert ll.optimal_lag == 3

    def test_cointegration_result_fields(self):
        ci = CointegrationResult(
            asset_a="A", asset_b="B", cointegrated=True,
            adf_stat=-3.5, p_value=0.01, hedge_ratio=1.5, half_life=15.0,
        )
        assert ci.cointegrated is True

    def test_spread_signal_fields(self):
        s = SpreadSignal(
            asset_a="A", asset_b="B", current_spread=5.0,
            z_score=2.5, signal="short_spread",
            entry_threshold=2.0, exit_threshold=0.5, hedge_ratio=1.5,
        )
        assert s.signal == "short_spread"

    def test_momentum_signal_fields(self):
        m = MomentumSignal(
            asset="SPY", momentum_1m=0.05, momentum_3m=0.12,
            momentum_6m=0.20, rank=1, signal="overweight",
        )
        assert m.rank == 1

    def test_macro_regime_signal_fields(self):
        ms = MacroRegimeSignal(
            indicator="vix_term", value=0.08, z_score=1.5,
            regime="risk_off", description="test",
        )
        assert ms.regime == "risk_off"

    def test_signal_dashboard_fields(self):
        db = SignalDashboard(
            overall_regime="risk_on", confidence=0.7,
            n_risk_on=3, n_risk_off=1, n_neutral=1,
            top_signals=["sig1"],
        )
        assert db.n_risk_on == 3


# ── ADF test ─────────────────────────────────────────────────────────────


class TestADFTest:
    def test_stationary_series_rejects(self):
        rng = np.random.RandomState(42)
        y = rng.normal(0, 1, 200)
        stat, p = _adf_test(y)
        assert p < 0.10

    def test_random_walk_fails_to_reject(self):
        rng = np.random.RandomState(42)
        y = rng.normal(0, 1, 200).cumsum()
        stat, p = _adf_test(y)
        assert p > 0.05

    def test_short_series(self):
        stat, p = _adf_test(np.array([1.0, 2.0, 3.0]))
        assert p == 1.0  # too short

    def test_returns_tuple(self):
        result = _adf_test(np.random.randn(100))
        assert len(result) == 2


# ── Inter-market correlations ────────────────────────────────────────────


class TestCorrelations:
    def test_produces_pairs(self):
        gen = _make_generator(assets=4)
        gen.analyze()
        expected = 4 * 3 // 2
        assert len(gen.correlations) == expected

    def test_correlation_range(self):
        gen = _make_generator()
        gen.analyze()
        for c in gen.correlations:
            assert -1.0 <= c.correlation <= 1.0

    def test_z_score_computed(self):
        gen = _make_generator()
        gen.analyze()
        for c in gen.correlations:
            assert isinstance(c.z_score, float)


# ── Lead-lag tests ───────────────────────────────────────────────────────


class TestLeadLag:
    def test_produces_results(self):
        gen = _make_generator()
        gen.analyze()
        # May or may not find significant lead-lag
        assert isinstance(gen.lead_lags, list)

    def test_sorted_by_correlation(self):
        gen = _make_generator()
        gen.analyze()
        if len(gen.lead_lags) >= 2:
            corrs = [abs(ll.correlation_at_lag) for ll in gen.lead_lags]
            assert corrs == sorted(corrs, reverse=True)

    def test_lag_within_bounds(self):
        gen = _make_generator(lead_lag_max=5)
        gen.analyze()
        for ll in gen.lead_lags:
            assert 0 <= ll.optimal_lag <= 5

    def test_direction_field(self):
        gen = _make_generator()
        gen.analyze()
        for ll in gen.lead_lags:
            assert ll.direction in ("positive", "negative")


# ── Cointegration tests ──────────────────────────────────────────────────


class TestCointegration:
    def test_detects_cointegrated_pair(self):
        prices = _make_cointegrated_prices()
        gen = CrossAssetSignalGenerator(prices, cointegration_pvalue=0.10)
        gen.analyze()
        ab = [c for c in gen.cointegrations if
              {c.asset_a, c.asset_b} == {"A", "B"}]
        assert len(ab) == 1
        assert ab[0].cointegrated is True

    def test_cointegrated_pair_has_lower_pvalue(self):
        """The truly cointegrated pair (A,B) should have a lower p-value than (A,C)."""
        prices = _make_cointegrated_prices()
        gen = CrossAssetSignalGenerator(prices, cointegration_pvalue=0.10)
        gen.analyze()
        ab = [c for c in gen.cointegrations if {c.asset_a, c.asset_b} == {"A", "B"}]
        ac = [c for c in gen.cointegrations if {c.asset_a, c.asset_b} == {"A", "C"}]
        if ab and ac:
            assert ab[0].p_value < ac[0].p_value

    def test_hedge_ratio_nonzero(self):
        prices = _make_cointegrated_prices()
        gen = CrossAssetSignalGenerator(prices)
        gen.analyze()
        for c in gen.cointegrations:
            assert c.hedge_ratio != 0.0

    def test_half_life_positive(self):
        prices = _make_cointegrated_prices()
        gen = CrossAssetSignalGenerator(prices, cointegration_pvalue=0.10)
        gen.analyze()
        coint = [c for c in gen.cointegrations if c.cointegrated]
        for c in coint:
            assert c.half_life > 0

    def test_sorted_by_pvalue(self):
        gen = _make_generator()
        gen.analyze()
        pvals = [c.p_value for c in gen.cointegrations]
        assert pvals == sorted(pvals)


# ── Spread signals tests ────────────────────────────────────────────────


class TestSpreadSignals:
    def test_only_cointegrated_pairs(self):
        prices = _make_cointegrated_prices()
        gen = CrossAssetSignalGenerator(prices, cointegration_pvalue=0.10)
        gen.analyze()
        for s in gen.spread_signals:
            pairs = [(c.asset_a, c.asset_b) for c in gen.cointegrations if c.cointegrated]
            assert (s.asset_a, s.asset_b) in pairs

    def test_signal_values(self):
        prices = _make_cointegrated_prices()
        gen = CrossAssetSignalGenerator(prices, cointegration_pvalue=0.10)
        gen.analyze()
        for s in gen.spread_signals:
            assert s.signal in ("long_spread", "short_spread", "neutral")

    def test_sorted_by_abs_zscore(self):
        prices = _make_cointegrated_prices()
        gen = CrossAssetSignalGenerator(prices, cointegration_pvalue=0.10)
        gen.analyze()
        if len(gen.spread_signals) >= 2:
            zs = [abs(s.z_score) for s in gen.spread_signals]
            assert zs == sorted(zs, reverse=True)


# ── Momentum tests ───────────────────────────────────────────────────────


class TestMomentum:
    def test_all_assets_ranked(self):
        gen = _make_generator()
        gen.analyze()
        assert len(gen.momentum_signals) == len(gen.assets)

    def test_ranks_unique(self):
        gen = _make_generator()
        gen.analyze()
        ranks = [m.rank for m in gen.momentum_signals]
        assert len(ranks) == len(set(ranks))

    def test_signal_assignment(self):
        gen = _make_generator()
        gen.analyze()
        signals = {m.signal for m in gen.momentum_signals}
        assert signals.issubset({"overweight", "underweight", "neutral"})

    def test_top_ranked_overweight(self):
        gen = _make_generator()
        gen.analyze()
        top = gen.momentum_signals[0]
        assert top.signal == "overweight"
        assert top.rank == 1


# ── Macro regime tests ───────────────────────────────────────────────────


class TestMacroRegime:
    def test_macro_signals_from_indicators(self):
        prices = _make_prices()
        macro = _make_macro(prices.index)
        gen = CrossAssetSignalGenerator(prices, macro_indicators=macro)
        gen.analyze()
        assert len(gen.macro_signals) == 3

    def test_regime_values(self):
        prices = _make_prices()
        macro = _make_macro(prices.index)
        gen = CrossAssetSignalGenerator(prices, macro_indicators=macro)
        gen.analyze()
        for ms in gen.macro_signals:
            assert ms.regime in ("risk_on", "risk_off", "neutral")

    def test_no_macro_no_signals(self):
        gen = _make_generator()
        gen.analyze()
        assert len(gen.macro_signals) == 0

    def test_description_populated(self):
        prices = _make_prices()
        macro = _make_macro(prices.index)
        gen = CrossAssetSignalGenerator(prices, macro_indicators=macro)
        gen.analyze()
        for ms in gen.macro_signals:
            assert len(ms.description) > 0


# ── Dashboard tests ──────────────────────────────────────────────────────


class TestDashboard:
    def test_dashboard_populated(self):
        gen = _make_generator()
        gen.analyze()
        assert gen.dashboard is not None

    def test_regime_valid(self):
        gen = _make_generator()
        gen.analyze()
        assert gen.dashboard.overall_regime in ("risk_on", "risk_off", "neutral")

    def test_confidence_range(self):
        gen = _make_generator()
        gen.analyze()
        assert 0 <= gen.dashboard.confidence <= 1


# ── Full pipeline tests ─────────────────────────────────────────────────


class TestPipeline:
    def test_analyze_returns_all_keys(self):
        gen = _make_generator()
        result = gen.analyze()
        expected = {
            "correlations", "lead_lags", "cointegrations",
            "spread_signals", "momentum_signals", "macro_signals", "dashboard",
        }
        assert set(result.keys()) == expected

    def test_from_csv(self, tmp_path):
        prices = _make_prices()
        csv = tmp_path / "prices.csv"
        prices.to_csv(csv)
        gen = CrossAssetSignalGenerator.from_csv(str(csv))
        gen.analyze()
        assert gen.dashboard is not None

    def test_from_csv_with_macro(self, tmp_path):
        prices = _make_prices()
        macro = _make_macro(prices.index)
        p_csv = tmp_path / "prices.csv"
        m_csv = tmp_path / "macro.csv"
        prices.to_csv(p_csv)
        macro.to_csv(m_csv)
        gen = CrossAssetSignalGenerator.from_csv(str(p_csv), str(m_csv))
        gen.analyze()
        assert len(gen.macro_signals) > 0


# ── Report tests ─────────────────────────────────────────────────────────


class TestReport:
    def test_generates_html(self, tmp_path):
        gen = _make_generator()
        path = gen.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Cross-Asset" in content

    def test_report_contains_sections(self, tmp_path):
        gen = _make_generator()
        path = gen.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "Correlation" in content
        assert "Lead-Lag" in content
        assert "Cointegration" in content
        assert "Spread" in content
        assert "Momentum" in content

    def test_report_embeds_charts(self, tmp_path):
        gen = _make_generator()
        path = gen.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "data:image/png;base64," in content

    def test_report_auto_analyzes(self, tmp_path):
        gen = _make_generator()
        assert gen.dashboard is None
        gen.generate_report(str(tmp_path / "report.html"))
        assert gen.dashboard is not None

    def test_report_at_default_path(self):
        gen = _make_generator()
        path = gen.generate_report()
        assert "cross_asset_signals.html" in path
        assert open(path).read().startswith("<!DOCTYPE html>")
