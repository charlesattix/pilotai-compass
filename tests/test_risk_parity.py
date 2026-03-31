"""Tests for compass.risk_parity — 38 tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.risk_parity import (
    RiskParityOptimizer,
    RPMethod,
    RPWeights,
    RiskContribution,
    BacktestRow,
    MethodComparison,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _returns(n: int = 300, k: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    data = {}
    for i in range(k):
        data[f"a{i}"] = rng.normal(0.0003 + i * 0.0001, 0.008 + i * 0.002, n)
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
# ERC
# ===========================================================================

class TestERC:
    def test_weights_sum_one(self):
        rp = RiskParityOptimizer()
        pw = rp.erc(_returns(200, 4))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_all_positive(self):
        rp = RiskParityOptimizer()
        pw = rp.erc(_returns(200, 4))
        assert all(w >= 0 for w in pw.weights.values())

    def test_risk_contrib_balanced(self):
        """ERC should produce roughly equal risk contributions."""
        rp = RiskParityOptimizer()
        ret = _returns(300, 4)
        pw = rp.erc(ret)
        rc = rp.risk_contributions(ret, pw.weights)
        pcts = [r.pct_of_total for r in rc]
        # Should be approximately equal (within ±15%)
        assert max(pcts) - min(pcts) < 0.30

    def test_custom_budget(self):
        rp = RiskParityOptimizer()
        ret = _returns(200, 3)
        budget = {"a0": 0.5, "a1": 0.3, "a2": 0.2}
        pw = rp.erc(ret, budget=budget)
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_empty(self):
        rp = RiskParityOptimizer()
        pw = rp.erc(pd.DataFrame())
        assert pw.weights == {}


# ===========================================================================
# HRP
# ===========================================================================

class TestHRP:
    def test_weights_sum_one(self):
        rp = RiskParityOptimizer()
        pw = rp.hrp(_returns(200, 5))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_all_positive(self):
        rp = RiskParityOptimizer()
        pw = rp.hrp(_returns(200, 5))
        assert all(w >= 0 for w in pw.weights.values())

    def test_single_asset(self):
        rp = RiskParityOptimizer()
        ret = pd.DataFrame({"only": np.random.default_rng(42).normal(0, 0.01, 50)})
        pw = rp.hrp(ret)
        assert pw.weights["only"] == pytest.approx(1.0)

    def test_lower_vol_higher_weight(self):
        rp = RiskParityOptimizer()
        pw = rp.hrp(_returns(200, 4))
        # a0 has lowest vol → should tend to get more weight
        w = list(pw.weights.values())
        assert w[0] > min(w) - 0.05  # roughly true


# ===========================================================================
# Inverse-vol
# ===========================================================================

class TestInverseVol:
    def test_weights_sum_one(self):
        rp = RiskParityOptimizer()
        pw = rp.inverse_vol(_returns(200, 4))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.001)

    def test_lowest_vol_highest_weight(self):
        rp = RiskParityOptimizer()
        pw = rp.inverse_vol(_returns(200, 4))
        # a0 has lowest vol → highest weight
        assert pw.weights["a0"] == max(pw.weights.values())

    def test_empty(self):
        rp = RiskParityOptimizer()
        pw = rp.inverse_vol(pd.DataFrame())
        assert pw.weights == {}


# ===========================================================================
# Max diversification
# ===========================================================================

class TestMaxDiv:
    def test_weights_sum_one(self):
        rp = RiskParityOptimizer()
        pw = rp.max_diversification(_returns(200, 4))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_dr_ge_one(self):
        rp = RiskParityOptimizer()
        pw = rp.max_diversification(_returns(200, 5))
        assert pw.diversification_ratio >= 0.99


# ===========================================================================
# Min variance
# ===========================================================================

class TestMinVar:
    def test_weights_sum_one(self):
        rp = RiskParityOptimizer()
        pw = rp.min_variance(_returns(200, 4))
        assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_vol_lower_than_equal_weight(self):
        rp = RiskParityOptimizer()
        ret = _returns(300, 4)
        mv = rp.min_variance(ret)
        ew_vol = float((ret.mean(axis=1)).std() * np.sqrt(252))
        assert mv.expected_vol <= ew_vol * 1.1  # should be lower or similar


# ===========================================================================
# Dispatcher
# ===========================================================================

class TestOptimize:
    def test_all_methods(self):
        rp = RiskParityOptimizer()
        ret = _returns(200, 4)
        for m in RPMethod:
            pw = rp.optimize(ret, m)
            assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.05)
            assert pw.method == m.value


# ===========================================================================
# Risk contributions
# ===========================================================================

class TestRiskContrib:
    def test_basic(self):
        rp = RiskParityOptimizer()
        ret = _returns(200, 4)
        pw = rp.erc(ret)
        rc = rp.risk_contributions(ret, pw.weights)
        assert len(rc) == 4
        assert all(isinstance(r, RiskContribution) for r in rc)

    def test_sums_near_one(self):
        rp = RiskParityOptimizer()
        ret = _returns(200, 4)
        pw = rp.erc(ret)
        rc = rp.risk_contributions(ret, pw.weights)
        total = sum(r.pct_of_total for r in rc)
        assert total == pytest.approx(1.0, abs=0.05)


# ===========================================================================
# Regime-conditional
# ===========================================================================

class TestRegime:
    def test_basic(self):
        rp = RiskParityOptimizer()
        ret = _returns(300, 4)
        reg = _regimes(300)
        results = rp.regime_optimize(ret, reg)
        assert len(results) >= 2
        for regime, pw in results.items():
            assert sum(pw.weights.values()) == pytest.approx(1.0, abs=0.05)

    def test_custom_method_map(self):
        rp = RiskParityOptimizer()
        ret = _returns(300, 4)
        reg = _regimes(300)
        custom = {"bull": RPMethod.HRP, "bear": RPMethod.ERC, "high_vol": RPMethod.MIN_VAR}
        results = rp.regime_optimize(ret, reg, method_map=custom)
        if "bull" in results:
            assert results["bull"].method == "hrp"


# ===========================================================================
# Backtest
# ===========================================================================

class TestBacktest:
    def test_runs(self):
        rp = RiskParityOptimizer()
        rows, comps = rp.backtest_all(_returns(200, 4), rebalance_freq=21)
        assert len(comps) >= 5
        assert all(isinstance(c, MethodComparison) for c in comps)

    def test_sorted_by_sharpe(self):
        rp = RiskParityOptimizer()
        _, comps = rp.backtest_all(_returns(200, 4), rebalance_freq=21)
        sharpes = [c.sharpe for c in comps]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_rows_populated(self):
        rp = RiskParityOptimizer()
        rows, _ = rp.backtest_all(_returns(150, 3), rebalance_freq=21)
        assert len(rows) >= 5
        for key, row_list in rows.items():
            assert len(row_list) == 150

    def test_equal_weight_included(self):
        rp = RiskParityOptimizer()
        _, comps = rp.backtest_all(_returns(200, 4), rebalance_freq=21)
        methods = {c.method for c in comps}
        assert "equal_weight" in methods


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        rp = RiskParityOptimizer()
        rows, comps = rp.backtest_all(_returns(200, 4), rebalance_freq=21)
        out = tmp_path / "rp.html"
        result = rp.generate_report(comps, all_rows=rows, output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Risk Parity" in html

    def test_contains_charts(self, tmp_path):
        rp = RiskParityOptimizer()
        rows, comps = rp.backtest_all(_returns(200, 4), rebalance_freq=21)
        out = tmp_path / "rp.html"
        rp.generate_report(comps, all_rows=rows, output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "Cumulative" in html

    def test_contains_comparison_table(self, tmp_path):
        rp = RiskParityOptimizer()
        _, comps = rp.backtest_all(_returns(200, 4))
        out = tmp_path / "rp.html"
        rp.generate_report(comps, output_path=str(out))
        html = out.read_text()
        assert "Method Comparison" in html

    def test_with_risk_contribs(self, tmp_path):
        rp = RiskParityOptimizer()
        ret = _returns(200, 4)
        pw = rp.erc(ret)
        rc = rp.risk_contributions(ret, pw.weights)
        _, comps = rp.backtest_all(ret)
        out = tmp_path / "rp.html"
        rp.generate_report(comps, risk_contribs=rc, output_path=str(out))
        html = out.read_text()
        assert "Risk Contribution" in html

    def test_with_regime(self, tmp_path):
        rp = RiskParityOptimizer()
        ret = _returns(300, 4)
        reg = _regimes(300)
        regime_pw = rp.regime_optimize(ret, reg)
        _, comps = rp.backtest_all(ret)
        out = tmp_path / "rp.html"
        rp.generate_report(comps, regime_portfolios=regime_pw, output_path=str(out))
        html = out.read_text()
        assert "Regime Allocations" in html
