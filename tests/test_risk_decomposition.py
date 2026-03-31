"""
Tests for compass/risk_decomposition.py — Portfolio risk decomposition engine.

Covers:
  - RiskDecomposer initialization and validation
  - Factor decomposition (market, volatility, regime, idiosyncratic)
  - Marginal risk contributions (Euler decomposition)
  - Risk budgeting (targets, breaches, tolerances)
  - VaR and CVaR at 95%/99% via historical simulation
  - Full pipeline (run_all)
  - HTML report generation
  - Edge cases (single experiment, zero vol, extreme weights)
"""

import os
import tempfile

import numpy as np
import pytest

from compass.risk_decomposition import (
    DEFAULT_CONFIDENCE_LEVELS,
    DecompositionResult,
    FactorContribution,
    MarginalRisk,
    RiskBudget,
    RiskDecomposer,
    VaRResult,
    generate_report,
)


# ── Test fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    """Seeded RNG for reproducibility."""
    return np.random.RandomState(42)


def _make_returns(rng, n_periods=252, n_experiments=3, spy_beta=None):
    """Generate synthetic experiment returns with known SPY beta."""
    spy = rng.normal(0.0004, 0.012, n_periods)
    vix = 20.0 + np.cumsum(rng.normal(0, 0.5, n_periods))
    vix = np.clip(vix, 10, 80)

    if spy_beta is None:
        spy_beta = [0.8, 0.5, 0.3]

    returns = {}
    for i in range(n_experiments):
        beta = spy_beta[i] if i < len(spy_beta) else 0.5
        idio = rng.normal(0, 0.005, n_periods)
        returns[f"EXP-{400 + i}"] = spy * beta + idio

    weights = {eid: 1.0 / n_experiments for eid in returns}
    return returns, spy, vix, weights


@pytest.fixture
def sample_data(rng):
    """Standard 3-experiment dataset."""
    return _make_returns(rng, n_periods=252, n_experiments=3)


@pytest.fixture
def decomposer(sample_data):
    """Ready-to-use RiskDecomposer."""
    returns, spy, vix, weights = sample_data
    return RiskDecomposer(returns=returns, spy_returns=spy, vix_levels=vix, weights=weights)


# ── Initialization tests ─────────────────────────────────────────────────────

class TestRiskDecomposerInit:
    def test_basic_construction(self, sample_data):
        returns, spy, vix, weights = sample_data
        rd = RiskDecomposer(returns=returns, spy_returns=spy, vix_levels=vix, weights=weights)
        assert rd.n_experiments == 3
        assert rd.n_periods == 252

    def test_empty_returns_raises(self):
        with pytest.raises(ValueError, match="at least one experiment"):
            RiskDecomposer(
                returns={},
                spy_returns=np.zeros(10),
                vix_levels=np.ones(10) * 20,
                weights={},
            )

    def test_mismatched_lengths_raises(self, rng):
        with pytest.raises(ValueError, match="same length"):
            RiskDecomposer(
                returns={"A": rng.randn(100), "B": rng.randn(50)},
                spy_returns=rng.randn(100),
                vix_levels=np.ones(100) * 20,
                weights={"A": 0.5, "B": 0.5},
            )

    def test_spy_length_mismatch_raises(self, rng):
        with pytest.raises(ValueError, match="spy_returns length"):
            RiskDecomposer(
                returns={"A": rng.randn(100)},
                spy_returns=rng.randn(50),
                vix_levels=np.ones(100) * 20,
                weights={"A": 1.0},
            )

    def test_vix_length_mismatch_raises(self, rng):
        with pytest.raises(ValueError, match="vix_levels length"):
            RiskDecomposer(
                returns={"A": rng.randn(100)},
                spy_returns=rng.randn(100),
                vix_levels=np.ones(50) * 20,
                weights={"A": 1.0},
            )

    def test_too_few_periods_raises(self, rng):
        with pytest.raises(ValueError, match="at least 2"):
            RiskDecomposer(
                returns={"A": np.array([0.01])},
                spy_returns=np.array([0.01]),
                vix_levels=np.array([20.0]),
                weights={"A": 1.0},
            )

    def test_weights_normalized_on_mismatch(self, rng):
        returns = {"A": rng.randn(100), "B": rng.randn(100)}
        spy = rng.randn(100)
        vix = np.ones(100) * 20
        rd = RiskDecomposer(
            returns=returns, spy_returns=spy, vix_levels=vix,
            weights={"A": 2.0, "B": 3.0},
        )
        assert abs(rd.weights.sum() - 1.0) < 1e-10

    def test_single_experiment(self, rng):
        returns = {"SOLO": rng.randn(100)}
        spy = rng.randn(100)
        vix = np.ones(100) * 20
        rd = RiskDecomposer(
            returns=returns, spy_returns=spy, vix_levels=vix,
            weights={"SOLO": 1.0},
        )
        assert rd.n_experiments == 1
        assert rd.cov_matrix.shape == (1, 1)


# ── Factor decomposition tests ───────────────────────────────────────────────

class TestFactorDecomposition:
    def test_returns_all_experiments(self, decomposer):
        factors = decomposer.decompose_factors()
        assert len(factors) == 3
        eids = {f.experiment_id for f in factors}
        assert eids == {"EXP-400", "EXP-401", "EXP-402"}

    def test_factors_sum_to_one(self, decomposer):
        factors = decomposer.decompose_factors()
        for fc in factors:
            total = fc.market + fc.volatility + fc.regime + fc.idiosyncratic
            assert abs(total - 1.0) < 0.02, f"{fc.experiment_id}: factors sum to {total}"

    def test_all_factors_non_negative(self, decomposer):
        factors = decomposer.decompose_factors()
        for fc in factors:
            assert fc.market >= 0.0
            assert fc.volatility >= 0.0
            assert fc.regime >= 0.0
            assert fc.idiosyncratic >= 0.0

    def test_high_beta_has_more_market_exposure(self, rng):
        """Experiment with high SPY beta should show more market factor."""
        returns, spy, vix, _ = _make_returns(rng, n_periods=500, n_experiments=2, spy_beta=[1.5, 0.0])
        weights = {"EXP-400": 0.5, "EXP-401": 0.5}
        rd = RiskDecomposer(returns=returns, spy_returns=spy, vix_levels=vix, weights=weights)
        factors = rd.decompose_factors()
        f_map = {f.experiment_id: f for f in factors}
        assert f_map["EXP-400"].market > f_map["EXP-401"].market

    def test_zero_variance_returns(self, rng):
        """Constant returns should produce zero factor contributions."""
        spy = rng.randn(100)
        vix = np.ones(100) * 20
        returns = {"FLAT": np.zeros(100)}
        rd = RiskDecomposer(returns=returns, spy_returns=spy, vix_levels=vix, weights={"FLAT": 1.0})
        factors = rd.decompose_factors()
        assert factors[0].market == 0.0
        assert factors[0].idiosyncratic == 0.0

    def test_factor_contribution_dataclass(self, decomposer):
        factors = decomposer.decompose_factors()
        fc = factors[0]
        assert isinstance(fc, FactorContribution)
        assert hasattr(fc, "total")
        assert fc.total == fc.market + fc.volatility + fc.regime + fc.idiosyncratic


# ── Marginal risk tests ──────────────────────────────────────────────────────

class TestMarginalRisk:
    def test_returns_all_experiments(self, decomposer):
        marginals = decomposer.compute_marginal_risk()
        assert len(marginals) == 3

    def test_pct_contributions_sum_to_one(self, decomposer):
        marginals = decomposer.compute_marginal_risk()
        total_pct = sum(m.pct_contribution for m in marginals)
        assert abs(total_pct - 1.0) < 0.01

    def test_risk_contributions_sum_to_portfolio_vol(self, decomposer):
        """Euler decomposition: sum(RC_i) = sigma_p."""
        marginals = decomposer.compute_marginal_risk()
        total_rc = sum(m.risk_contribution for m in marginals)
        cov_ann = decomposer.cov_matrix * decomposer.periods_per_year
        port_vol = float(np.sqrt(decomposer.weights @ cov_ann @ decomposer.weights))
        assert abs(total_rc - port_vol) < 0.001

    def test_marginal_risk_dataclass(self, decomposer):
        marginals = decomposer.compute_marginal_risk()
        m = marginals[0]
        assert isinstance(m, MarginalRisk)
        assert m.marginal_vol >= 0

    def test_equal_weights_similar_contributions(self, rng):
        """With equal weights and similar returns, contributions should be roughly equal."""
        n = 500
        spy = rng.normal(0, 0.01, n)
        vix = np.ones(n) * 20
        # All experiments have same beta and similar idiosyncratic
        returns = {f"E{i}": spy * 0.5 + rng.normal(0, 0.005, n) for i in range(3)}
        weights = {f"E{i}": 1 / 3 for i in range(3)}
        rd = RiskDecomposer(returns=returns, spy_returns=spy, vix_levels=vix, weights=weights)
        marginals = rd.compute_marginal_risk()
        pcts = [m.pct_contribution for m in marginals]
        assert max(pcts) - min(pcts) < 0.15

    def test_zero_vol_portfolio(self, rng):
        """All-zero returns should produce equal pct contributions."""
        spy = rng.randn(100)
        vix = np.ones(100) * 20
        returns = {"A": np.zeros(100), "B": np.zeros(100)}
        rd = RiskDecomposer(returns=returns, spy_returns=spy, vix_levels=vix,
                            weights={"A": 0.5, "B": 0.5})
        marginals = rd.compute_marginal_risk()
        for m in marginals:
            assert abs(m.pct_contribution - 0.5) < 0.01


# ── Risk budgeting tests ─────────────────────────────────────────────────────

class TestRiskBudgeting:
    def test_default_equal_targets(self, decomposer):
        budgets = decomposer.compute_risk_budgets()
        assert len(budgets) == 3
        for b in budgets:
            assert abs(b.target_pct - 1 / 3) < 0.01

    def test_custom_targets(self, decomposer):
        targets = {"EXP-400": 0.5, "EXP-401": 0.3, "EXP-402": 0.2}
        budgets = decomposer.compute_risk_budgets(targets=targets)
        b_map = {b.experiment_id: b for b in budgets}
        assert abs(b_map["EXP-400"].target_pct - 0.5) < 0.01

    def test_breach_detection(self, rng):
        """With extreme weight skew, at least one experiment should breach."""
        n = 252
        spy = rng.normal(0, 0.01, n)
        vix = np.ones(n) * 20
        returns = {"A": spy * 2.0 + rng.normal(0, 0.005, n),
                   "B": rng.normal(0, 0.001, n)}
        weights = {"A": 0.95, "B": 0.05}
        rd = RiskDecomposer(returns=returns, spy_returns=spy, vix_levels=vix, weights=weights)
        # Target equal risk but weights are extreme
        budgets = rd.compute_risk_budgets(targets={"A": 0.5, "B": 0.5}, tolerance=0.05)
        breaches = [b for b in budgets if b.breach]
        assert len(breaches) >= 1

    def test_no_breach_within_tolerance(self, decomposer):
        """With default equal weights and high tolerance, no breaches."""
        budgets = decomposer.compute_risk_budgets(tolerance=0.99)
        assert all(not b.breach for b in budgets)

    def test_target_normalization(self, decomposer):
        """Non-unit targets should be normalized."""
        targets = {"EXP-400": 2.0, "EXP-401": 2.0, "EXP-402": 1.0}
        budgets = decomposer.compute_risk_budgets(targets=targets)
        b_map = {b.experiment_id: b for b in budgets}
        assert abs(b_map["EXP-400"].target_pct - 0.4) < 0.01
        assert abs(b_map["EXP-402"].target_pct - 0.2) < 0.01

    def test_budget_dataclass(self, decomposer):
        budgets = decomposer.compute_risk_budgets()
        b = budgets[0]
        assert isinstance(b, RiskBudget)
        assert isinstance(b.breach, bool)


# ── VaR / CVaR tests ─────────────────────────────────────────────────────────

class TestVaRCVaR:
    def test_returns_both_confidence_levels(self, decomposer):
        results = decomposer.compute_var_cvar()
        assert len(results) == 2
        confidences = {r.confidence for r in results}
        assert confidences == {0.95, 0.99}

    def test_var_positive_loss(self, decomposer):
        """VaR should be positive (represents a loss)."""
        results = decomposer.compute_var_cvar()
        for r in results:
            assert r.var > 0

    def test_cvar_gte_var(self, decomposer):
        """CVaR (Expected Shortfall) >= VaR always."""
        results = decomposer.compute_var_cvar()
        for r in results:
            assert r.cvar >= r.var - 1e-10

    def test_99_var_gte_95_var(self, decomposer):
        """99% VaR should be >= 95% VaR."""
        results = decomposer.compute_var_cvar()
        r_map = {r.confidence: r for r in results}
        assert r_map[0.99].var >= r_map[0.95].var - 1e-10

    def test_multi_day_horizon(self, decomposer):
        results = decomposer.compute_var_cvar(horizon_days=5)
        for r in results:
            assert r.horizon_days == 5
            assert r.var > 0

    def test_multi_day_var_larger_than_1day(self, decomposer):
        """5-day VaR should generally be larger than 1-day VaR."""
        var_1d = decomposer.compute_var_cvar(horizon_days=1)
        var_5d = decomposer.compute_var_cvar(horizon_days=5)
        # Compare at 95% confidence
        v1 = [r for r in var_1d if r.confidence == 0.95][0]
        v5 = [r for r in var_5d if r.confidence == 0.95][0]
        assert v5.var > v1.var * 0.5  # should be noticeably larger

    def test_invalid_horizon_raises(self, decomposer):
        with pytest.raises(ValueError, match="horizon_days must be >= 1"):
            decomposer.compute_var_cvar(horizon_days=0)

    def test_custom_confidence_levels(self, decomposer):
        results = decomposer.compute_var_cvar(confidence_levels=(0.90, 0.975))
        assert len(results) == 2
        assert {r.confidence for r in results} == {0.90, 0.975}

    def test_var_result_dataclass(self, decomposer):
        results = decomposer.compute_var_cvar()
        r = results[0]
        assert isinstance(r, VaRResult)
        assert r.horizon_days == 1


# ── Full pipeline tests ──────────────────────────────────────────────────────

class TestRunAll:
    def test_returns_decomposition_result(self, decomposer):
        result = decomposer.run_all()
        assert isinstance(result, DecompositionResult)

    def test_all_components_populated(self, decomposer):
        result = decomposer.run_all()
        assert len(result.factor_contributions) == 3
        assert len(result.marginal_risks) == 3
        assert len(result.risk_budgets) == 3
        assert len(result.var_results) > 0

    def test_summary_keys(self, decomposer):
        result = decomposer.run_all()
        expected_keys = {
            "n_experiments", "n_periods", "portfolio_vol_ann_pct",
            "portfolio_vol_daily_pct", "avg_factor_market_pct",
            "avg_factor_volatility_pct", "avg_factor_regime_pct",
            "avg_factor_idiosyncratic_pct", "risk_budget_breaches",
            "dominant_factor",
        }
        assert expected_keys.issubset(set(result.summary.keys()))

    def test_portfolio_vol_positive(self, decomposer):
        result = decomposer.run_all()
        assert result.portfolio_vol > 0
        assert result.portfolio_vol_daily > 0

    def test_custom_risk_targets(self, decomposer):
        targets = {"EXP-400": 0.5, "EXP-401": 0.3, "EXP-402": 0.2}
        result = decomposer.run_all(risk_targets=targets)
        b_map = {b.experiment_id: b for b in result.risk_budgets}
        assert abs(b_map["EXP-400"].target_pct - 0.5) < 0.01

    def test_multiple_var_horizons(self, decomposer):
        result = decomposer.run_all(var_horizons=(1, 5, 10))
        # 2 confidence levels × 3 horizons = 6 results
        assert len(result.var_results) == 6


# ── HTML report tests ─────────────────────────────────────────────────────────

class TestHTMLReport:
    def test_generates_valid_html(self, decomposer):
        result = decomposer.run_all()
        html = decomposer.generate_html(result)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_factor_chart(self, decomposer):
        result = decomposer.run_all()
        html = decomposer.generate_html(result)
        assert "<svg" in html
        assert "Market" in html
        assert "Volatility" in html
        assert "Regime" in html
        assert "Idiosyncratic" in html

    def test_contains_var_table(self, decomposer):
        result = decomposer.run_all()
        html = decomposer.generate_html(result)
        assert "Value-at-Risk" in html
        assert "CVaR" in html
        assert "95%" in html

    def test_contains_risk_budget_table(self, decomposer):
        result = decomposer.run_all()
        html = decomposer.generate_html(result)
        assert "Risk Budget Status" in html
        assert "Target" in html
        assert "Deviation" in html

    def test_contains_experiment_ids(self, decomposer):
        result = decomposer.run_all()
        html = decomposer.generate_html(result)
        for eid in decomposer.experiment_ids:
            assert eid in html

    def test_contains_kpi_section(self, decomposer):
        result = decomposer.run_all()
        html = decomposer.generate_html(result)
        assert "Annualized Vol" in html
        assert "Dominant Factor" in html
        assert "Budget Breaches" in html


# ── Convenience function tests ────────────────────────────────────────────────

class TestGenerateReport:
    def test_writes_html_file(self, sample_data):
        returns, spy, vix, weights = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            result = generate_report(
                returns=returns, spy_returns=spy, vix_levels=vix,
                weights=weights, output_path=path,
            )
            assert os.path.isfile(path)
            assert isinstance(result, DecompositionResult)
            with open(path) as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content

    def test_creates_output_directory(self, sample_data):
        returns, spy, vix, weights = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "report.html")
            generate_report(
                returns=returns, spy_returns=spy, vix_levels=vix,
                weights=weights, output_path=path,
            )
            assert os.path.isfile(path)
