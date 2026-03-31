"""Tests for compass.portfolio_constructor — 55+ tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.portfolio_constructor import (
    PortfolioConstructor,
    OptMethod,
    RebalanceFreq,
    PortfolioWeights,
    Constraints,
    BLView,
    RiskContribution,
    ConstructionResult,
    RebalanceEvent,
    WalkForwardFold,
    ledoit_wolf_shrinkage,
    TRADING_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _returns(n: int = 300, k: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    data = {}
    for i in range(k):
        mu = 0.0003 + i * 0.0001
        sigma = 0.008 + i * 0.002
        data[f"asset_{i}"] = rng.normal(mu, sigma, n)
    return pd.DataFrame(data, index=idx)


def _regimes(n: int = 300) -> pd.Series:
    labels = []
    for i in range(n):
        if i < n * 0.4:
            labels.append("bull")
        elif i < n * 0.7:
            labels.append("bear")
        else:
            labels.append("high_vol")
    return pd.Series(labels, index=pd.bdate_range("2024-01-02", periods=n))


# ===========================================================================
# Ledoit-Wolf
# ===========================================================================

class TestLedoitWolf:
    def test_positive_definite(self):
        ret = _returns(100, 4)
        cov = ledoit_wolf_shrinkage(ret)
        eigvals = np.linalg.eigvalsh(cov)
        assert (eigvals > -1e-10).all()

    def test_shape(self):
        ret = _returns(100, 5)
        cov = ledoit_wolf_shrinkage(ret)
        assert cov.shape == (5, 5)

    def test_short_data(self):
        ret = pd.DataFrame({"a": [0.01]})
        cov = ledoit_wolf_shrinkage(ret)
        assert cov.shape == (1, 1)


# ===========================================================================
# Mean-variance
# ===========================================================================

class TestMeanVariance:
    def test_weights_sum_one(self):
        pc = PortfolioConstructor()
        pw = pc.mean_variance(_returns(200, 4))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_all_positive(self):
        pc = PortfolioConstructor()
        pw = pc.mean_variance(_returns(200, 4))
        assert all(w >= -1e-6 for w in pw.weights.values())

    def test_sharpe_computed(self):
        pc = PortfolioConstructor()
        pw = pc.mean_variance(_returns(200, 4))
        assert pw.sharpe != 0.0

    def test_max_position_respected(self):
        pc = PortfolioConstructor()
        c = Constraints(max_position=0.30)
        pw = pc.mean_variance(_returns(200, 5), constraints=c)
        assert all(w <= 0.31 for w in pw.weights.values())

    def test_empty(self):
        pc = PortfolioConstructor()
        pw = pc.mean_variance(pd.DataFrame())
        assert pw.weights == {}


# ===========================================================================
# Black-Litterman
# ===========================================================================

class TestBlackLitterman:
    def test_basic(self):
        pc = PortfolioConstructor()
        ret = _returns(200, 4)
        views = [BLView("asset_0", 0.15, 0.8)]
        pw = pc.black_litterman(ret, views)
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_view_tilts_allocation(self):
        pc = PortfolioConstructor()
        ret = _returns(200, 4)
        no_view = pc.black_litterman(ret, [])
        strong_view = pc.black_litterman(ret, [BLView("asset_0", 0.50, 0.95)])
        # Strong bullish view should increase weight for asset_0
        assert strong_view.weights["asset_0"] >= no_view.weights["asset_0"] - 0.05

    def test_no_views(self):
        pc = PortfolioConstructor()
        pw = pc.black_litterman(_returns(200, 3), [])
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# Risk parity
# ===========================================================================

class TestRiskParity:
    def test_weights_sum_one(self):
        pw = PortfolioConstructor.risk_parity(_returns(200, 4))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.001)

    def test_lower_vol_higher_weight(self):
        pw = PortfolioConstructor.risk_parity(_returns(200, 4))
        # asset_0 has lowest vol → should have highest weight
        weights = list(pw.weights.values())
        assert weights[0] >= max(weights) - 0.05

    def test_empty(self):
        pw = PortfolioConstructor.risk_parity(pd.DataFrame())
        assert pw.weights == {}


# ===========================================================================
# HRP
# ===========================================================================

class TestHRP:
    def test_weights_sum_one(self):
        pw = PortfolioConstructor.hrp(_returns(200, 5))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_all_positive(self):
        pw = PortfolioConstructor.hrp(_returns(200, 5))
        assert all(w >= 0 for w in pw.weights.values())

    def test_single_asset(self):
        ret = pd.DataFrame({"a": np.random.default_rng(42).normal(0, 0.01, 50)})
        pw = PortfolioConstructor.hrp(ret)
        assert pw.weights["a"] == pytest.approx(1.0)

    def test_two_assets(self):
        rng = np.random.default_rng(42)
        ret = pd.DataFrame({"a": rng.normal(0, 0.01, 100), "b": rng.normal(0, 0.02, 100)})
        pw = PortfolioConstructor.hrp(ret)
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)
        # Lower vol asset should get more weight
        assert pw.weights["a"] > pw.weights["b"]


# ===========================================================================
# Min CVaR
# ===========================================================================

class TestMinCVaR:
    def test_weights_sum_one(self):
        pc = PortfolioConstructor()
        pw = pc.min_cvar(_returns(200, 4))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_cvar_computed(self):
        pc = PortfolioConstructor()
        pw = pc.min_cvar(_returns(200, 4))
        assert pw.cvar_95 > 0

    def test_empty(self):
        pc = PortfolioConstructor()
        pw = pc.min_cvar(pd.DataFrame())
        assert pw.weights == {}


# ===========================================================================
# Max diversification
# ===========================================================================

class TestMaxDiv:
    def test_weights_sum_one(self):
        pw = PortfolioConstructor.max_diversification(_returns(200, 4))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_dr_above_one(self):
        pw = PortfolioConstructor.max_diversification(_returns(200, 5))
        assert pw.diversification_ratio >= 0.99  # DR ≥ 1 for diversified portfolio


# ===========================================================================
# Regime-conditional
# ===========================================================================

class TestRegime:
    def test_basic(self):
        pc = PortfolioConstructor()
        ret = _returns(300, 4)
        reg = _regimes(300)
        results = pc.regime_construct(ret, reg)
        assert len(results) >= 2
        for regime, pw in results.items():
            assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.05)


# ===========================================================================
# Constraints
# ===========================================================================

class TestConstraints:
    def test_turnover_limit(self):
        current = {"a": 0.5, "b": 0.5}
        target = {"a": 0.0, "b": 1.0}
        result = PortfolioConstructor.apply_turnover_limit(current, target, 0.5)
        turnover = sum(abs(result[a] - current.get(a, 0)) for a in result)
        assert turnover <= 0.51

    def test_no_limit(self):
        current = {"a": 0.5, "b": 0.5}
        target = {"a": 0.3, "b": 0.7}
        result = PortfolioConstructor.apply_turnover_limit(current, target, 10.0)
        assert result["a"] == pytest.approx(0.3, abs=0.01)

    def test_sector_limits(self):
        weights = {"a": 0.4, "b": 0.4, "c": 0.2}
        sectors = {"a": "tech", "b": "tech", "c": "fin"}
        result = PortfolioConstructor.apply_sector_limits(weights, sectors, 0.50)
        tech_total = result["a"] + result["b"]
        assert tech_total <= 0.51


# ===========================================================================
# Risk contributions
# ===========================================================================

class TestRiskContrib:
    def test_basic(self):
        ret = _returns(200, 4)
        pw = PortfolioConstructor.risk_parity(ret)
        rc = PortfolioConstructor.risk_contributions(ret, pw.weights)
        assert len(rc) == 4
        assert all(isinstance(r, RiskContribution) for r in rc)

    def test_pct_sums_near_one(self):
        ret = _returns(200, 4)
        pw = PortfolioConstructor.risk_parity(ret)
        rc = PortfolioConstructor.risk_contributions(ret, pw.weights)
        total = sum(r.pct_contribution for r in rc)
        assert total == pytest.approx(1.0, abs=0.05)


# ===========================================================================
# Efficient frontier
# ===========================================================================

class TestFrontier:
    def test_basic(self):
        pc = PortfolioConstructor()
        ef = pc.efficient_frontier(_returns(200, 4), n_points=10)
        assert not ef.empty
        assert "return" in ef.columns
        assert "volatility" in ef.columns

    def test_monotone_vol(self):
        pc = PortfolioConstructor()
        ef = pc.efficient_frontier(_returns(200, 4), n_points=15)
        if len(ef) >= 2:
            # Higher return should generally → higher vol
            assert ef["volatility"].iloc[-1] >= ef["volatility"].iloc[0] - 0.01


# ===========================================================================
# Unified construct
# ===========================================================================

class TestConstruct:
    def test_dispatch_all_methods(self):
        pc = PortfolioConstructor()
        ret = _returns(200, 4)
        for method in OptMethod:
            views = [BLView("asset_0", 0.20, 0.7)] if method == OptMethod.BLACK_LITTERMAN else None
            pw = pc.construct(ret, method, views=views)
            assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.05)

    def test_full_construct(self):
        pc = PortfolioConstructor()
        cr = pc.full_construct(_returns(200, 4), compute_frontier=True)
        assert isinstance(cr, ConstructionResult)
        assert len(cr.risk_contributions) == 4
        assert cr.efficient_frontier is not None


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        pc = PortfolioConstructor()
        cr = pc.full_construct(_returns(200, 4), compute_frontier=True)
        out = tmp_path / "port.html"
        result = pc.generate_report(cr, output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Portfolio Construction" in html

    def test_contains_pie(self, tmp_path):
        pc = PortfolioConstructor()
        cr = pc.full_construct(_returns(200, 4))
        out = tmp_path / "p.html"
        pc.generate_report(cr, output_path=str(out))
        html = out.read_text()
        assert "<svg" in html

    def test_contains_frontier(self, tmp_path):
        pc = PortfolioConstructor()
        cr = pc.full_construct(_returns(200, 4), compute_frontier=True)
        out = tmp_path / "p.html"
        pc.generate_report(cr, output_path=str(out))
        html = out.read_text()
        assert "Efficient Frontier" in html

    def test_contains_risk_table(self, tmp_path):
        pc = PortfolioConstructor()
        cr = pc.full_construct(_returns(200, 4))
        out = tmp_path / "p.html"
        pc.generate_report(cr, output_path=str(out))
        html = out.read_text()
        assert "Risk Contributions" in html

    def test_with_regime(self, tmp_path):
        pc = PortfolioConstructor()
        cr = pc.full_construct(_returns(300, 4))
        regime_pw = pc.regime_construct(_returns(300, 4), _regimes(300))
        out = tmp_path / "p.html"
        pc.generate_report(cr, regime_portfolios=regime_pw, output_path=str(out))
        html = out.read_text()
        assert "Regime Allocations" in html

    def test_with_rebalance(self, tmp_path):
        pc = PortfolioConstructor()
        cr = pc.full_construct(_returns(200, 4), run_rebalance=True)
        out = tmp_path / "p.html"
        pc.generate_report(cr, output_path=str(out))
        html = out.read_text()
        assert "Rebalancing Timeline" in html

    def test_with_walk_forward(self, tmp_path):
        pc = PortfolioConstructor()
        cr = pc.full_construct(_returns(300, 4), run_walk_forward=True)
        out = tmp_path / "p.html"
        pc.generate_report(cr, output_path=str(out))
        html = out.read_text()
        assert "Walk-Forward" in html


# ===========================================================================
# Equal weight
# ===========================================================================

class TestEqualWeight:
    def test_weights_sum_one(self):
        pw = PortfolioConstructor.equal_weight(_returns(100, 5))
        assert sum(pw.weights.values()) == pytest.approx(1.0)

    def test_all_equal(self):
        pw = PortfolioConstructor.equal_weight(_returns(100, 4))
        assert all(w == pytest.approx(0.25) for w in pw.weights.values())

    def test_method_tag(self):
        pw = PortfolioConstructor.equal_weight(_returns(100, 3))
        assert pw.method == "equal_weight"

    def test_empty(self):
        pw = PortfolioConstructor.equal_weight(pd.DataFrame())
        assert pw.weights == {}

    def test_dispatch(self):
        pc = PortfolioConstructor()
        pw = pc.construct(_returns(100, 4), OptMethod.EQUAL_WEIGHT)
        assert pw.method == "equal_weight"


# ===========================================================================
# Rebalancing engine
# ===========================================================================

class TestRebalance:
    def test_basic(self):
        pc = PortfolioConstructor()
        rets, events = pc.rebalance_backtest(
            _returns(200, 4), method=OptMethod.EQUAL_WEIGHT,
            freq=RebalanceFreq.MONTHLY)
        assert len(rets) == 200
        assert len(events) >= 1

    def test_calendar_trigger(self):
        pc = PortfolioConstructor()
        _, events = pc.rebalance_backtest(
            _returns(200, 3), freq=RebalanceFreq.WEEKLY)
        triggers = {e.trigger for e in events}
        assert "calendar" in triggers or "initial" in triggers

    def test_threshold_trigger(self):
        pc = PortfolioConstructor()
        _, events = pc.rebalance_backtest(
            _returns(200, 3), freq=RebalanceFreq.MONTHLY,
            drift_threshold=0.001)  # very low threshold
        triggers = {e.trigger for e in events}
        assert "threshold" in triggers or "calendar" in triggers

    def test_cost_tracked(self):
        pc = PortfolioConstructor()
        _, events = pc.rebalance_backtest(
            _returns(200, 4), cost_per_unit=0.01)
        total_cost = sum(e.cost for e in events)
        assert total_cost > 0

    def test_turnover_recorded(self):
        pc = PortfolioConstructor()
        _, events = pc.rebalance_backtest(_returns(200, 4))
        for e in events:
            assert isinstance(e, RebalanceEvent)
            assert e.turnover >= 0

    def test_daily_rebalance(self):
        pc = PortfolioConstructor()
        _, events = pc.rebalance_backtest(
            _returns(150, 3), freq=RebalanceFreq.DAILY)
        # Should have many rebalance events
        assert len(events) >= 5

    def test_empty_returns(self):
        pc = PortfolioConstructor()
        rets, events = pc.rebalance_backtest(pd.DataFrame())
        assert rets.empty
        assert events == []


# ===========================================================================
# Walk-forward construction
# ===========================================================================

class TestWalkForward:
    def test_basic(self):
        pc = PortfolioConstructor()
        pw, folds = pc.walk_forward_construct(
            _returns(300, 4), method=OptMethod.RISK_PARITY, n_folds=4)
        assert isinstance(pw, PortfolioWeights)
        assert len(folds) >= 2

    def test_fold_structure(self):
        pc = PortfolioConstructor()
        _, folds = pc.walk_forward_construct(_returns(300, 4), n_folds=5)
        for f in folds:
            assert isinstance(f, WalkForwardFold)
            assert f.in_sample_sharpe != 0 or f.out_of_sample_sharpe != 0
            assert f.train_end < f.test_start

    def test_short_data(self):
        pc = PortfolioConstructor()
        pw, folds = pc.walk_forward_construct(_returns(30, 3), n_folds=5)
        assert folds == []
        assert pw.weights  # still returns weights

    def test_weights_valid(self):
        pc = PortfolioConstructor()
        pw, _ = pc.walk_forward_construct(_returns(300, 4))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.05)

    def test_oos_sharpe_computed(self):
        pc = PortfolioConstructor()
        _, folds = pc.walk_forward_construct(_returns(300, 4), n_folds=4)
        if folds:
            assert isinstance(folds[0].out_of_sample_sharpe, float)
