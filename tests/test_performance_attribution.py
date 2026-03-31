"""Tests for compass.performance_attribution — 40+ tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.performance_attribution import (
    PerformanceAttribution,
    BrinsonAttribution,
    FactorDecomposition,
    ExperimentContribution,
    PeriodAttribution,
    SkillTestResult,
    StrategyDecomposition,
    TRADING_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(n: int = 252) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=n)


def _returns(n: int = 252, mu: float = 0.0003, sigma: float = 0.01,
             seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n), index=_dates(n), name="ret")


def _market(n: int = 252, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.012, n), index=_dates(n), name="mkt")


# ===========================================================================
# Brinson attribution
# ===========================================================================

class TestBrinson:
    def test_zero_active(self):
        """Identical weights & returns → zero active return."""
        w = np.array([0.5, 0.5])
        r = np.array([0.01, 0.02])
        ba = PerformanceAttribution.brinson_attribution(w, w, r, r)
        assert ba.allocation_effect == pytest.approx(0.0)
        assert ba.selection_effect == pytest.approx(0.0)
        assert ba.total_active == pytest.approx(0.0)

    def test_allocation_effect(self):
        """Overweight segment that outperforms → positive allocation."""
        pw = np.array([0.7, 0.3])
        bw = np.array([0.5, 0.5])
        br = np.array([0.05, 0.01])  # seg 0 outperforms
        ba = PerformanceAttribution.brinson_attribution(pw, bw, br, br)
        assert ba.allocation_effect > 0

    def test_selection_effect(self):
        """Portfolio outperforms benchmark in same segment → positive selection."""
        w = np.array([0.5, 0.5])
        pr = np.array([0.06, 0.02])
        br = np.array([0.03, 0.02])
        ba = PerformanceAttribution.brinson_attribution(w, w, pr, br)
        assert ba.selection_effect > 0

    def test_total_active_decomposition(self):
        """Alloc + Selection + Interaction = Total active."""
        pw = np.array([0.6, 0.4])
        bw = np.array([0.5, 0.5])
        pr = np.array([0.04, 0.01])
        br = np.array([0.03, 0.02])
        ba = PerformanceAttribution.brinson_attribution(pw, bw, pr, br)
        assert ba.total_active == pytest.approx(
            ba.allocation_effect + ba.selection_effect + ba.interaction_effect)

    def test_date_passthrough(self):
        dt = datetime(2026, 3, 15)
        ba = PerformanceAttribution.brinson_attribution(
            np.array([1.0]), np.array([1.0]),
            np.array([0.01]), np.array([0.01]), date=dt)
        assert ba.date == dt

    def test_brinson_series(self):
        dates = pd.bdate_range("2026-01-01", periods=5)
        pw = pd.DataFrame([[0.6, 0.4]] * 5, index=dates)
        bw = pd.DataFrame([[0.5, 0.5]] * 5, index=dates)
        pr = pd.DataFrame(np.random.default_rng(1).normal(0.01, 0.005, (5, 2)), index=dates)
        br = pd.DataFrame(np.random.default_rng(2).normal(0.01, 0.005, (5, 2)), index=dates)
        pa = PerformanceAttribution()
        results = pa.brinson_series(pw, bw, pr, br)
        assert len(results) == 5
        assert all(isinstance(r, BrinsonAttribution) for r in results)


# ===========================================================================
# Factor attribution
# ===========================================================================

class TestFactorAttribution:
    def test_market_only(self):
        port = _returns(200, mu=0.0005, seed=10)
        mkt = _market(200, seed=10)
        # Make portfolio track market closely
        port = mkt * 1.1 + 0.0001
        pa = PerformanceAttribution()
        fd = pa.factor_attribution(port, mkt)
        assert isinstance(fd, FactorDecomposition)
        assert fd.market != 0.0

    def test_total_return_set(self):
        port = _returns(200)
        mkt = _market(200)
        pa = PerformanceAttribution()
        fd = pa.factor_attribution(port, mkt)
        assert fd.total_return != 0.0

    def test_with_vol_factor(self):
        port = _returns(200)
        mkt = _market(200)
        vol = pd.Series(np.random.default_rng(99).normal(0, 0.005, 200),
                         index=_dates(200))
        pa = PerformanceAttribution()
        fd = pa.factor_attribution(port, mkt, vol_factor=vol)
        # volatility contribution should be populated
        assert isinstance(fd.volatility, float)

    def test_with_all_factors(self):
        n = 200
        idx = _dates(n)
        rng = np.random.default_rng(55)
        port = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
        mkt = pd.Series(rng.normal(0.0002, 0.012, n), index=idx)
        vol = pd.Series(rng.normal(0, 0.005, n), index=idx)
        reg = pd.Series(rng.normal(0, 0.003, n), index=idx)
        tim = pd.Series(rng.normal(0, 0.002, n), index=idx)
        pa = PerformanceAttribution()
        fd = pa.factor_attribution(port, mkt, vol, reg, tim)
        # All factors present
        assert isinstance(fd.regime, float)
        assert isinstance(fd.timing, float)

    def test_too_few_returns(self):
        port = pd.Series([0.01, 0.02], index=pd.bdate_range("2026-01-01", periods=2))
        mkt = pd.Series([0.005, 0.01], index=port.index)
        pa = PerformanceAttribution()
        fd = pa.factor_attribution(port, mkt)
        assert isinstance(fd, FactorDecomposition)

    def test_selection_is_alpha(self):
        """When portfolio has positive alpha, selection should be positive."""
        n = 300
        rng = np.random.default_rng(77)
        mkt = pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
        port = mkt + 0.001  # constant positive alpha
        pa = PerformanceAttribution()
        fd = pa.factor_attribution(port, mkt)
        assert fd.selection > 0


# ===========================================================================
# Rolling factor attribution
# ===========================================================================

class TestRollingAttribution:
    def test_shape(self):
        port = _returns(150)
        mkt = _market(150)
        pa = PerformanceAttribution()
        df = pa.rolling_factor_attribution(port, mkt, window=60)
        assert len(df) > 0
        assert "market" in df.columns
        assert "selection" in df.columns

    def test_window_effect(self):
        port = _returns(200)
        mkt = _market(200)
        pa = PerformanceAttribution()
        short = pa.rolling_factor_attribution(port, mkt, window=30)
        long_ = pa.rolling_factor_attribution(port, mkt, window=100)
        assert len(short) > len(long_)

    def test_empty_when_too_short(self):
        port = _returns(10)
        mkt = _market(10)
        pa = PerformanceAttribution()
        df = pa.rolling_factor_attribution(port, mkt, window=20)
        assert df.empty

    def test_all_factor_columns(self):
        n = 150
        idx = _dates(n)
        rng = np.random.default_rng(88)
        port = pd.Series(rng.normal(0, 0.01, n), index=idx)
        mkt = pd.Series(rng.normal(0, 0.01, n), index=idx)
        vol = pd.Series(rng.normal(0, 0.005, n), index=idx)
        pa = PerformanceAttribution()
        df = pa.rolling_factor_attribution(port, mkt, window=30, vol_factor=vol)
        assert "volatility" in df.columns


# ===========================================================================
# Experiment contributions
# ===========================================================================

class TestExperimentContributions:
    def test_basic(self):
        w = {"A": 0.5, "B": 0.3, "C": 0.2}
        r = {"A": 0.02, "B": -0.01, "C": 0.04}
        contribs = PerformanceAttribution.experiment_contributions(w, r)
        assert len(contribs) == 3
        total = sum(c.contribution for c in contribs)
        assert total == pytest.approx(0.5 * 0.02 + 0.3 * -0.01 + 0.2 * 0.04)

    def test_pct_of_total_sums_to_one(self):
        w = {"A": 0.6, "B": 0.4}
        r = {"A": 0.03, "B": 0.01}
        contribs = PerformanceAttribution.experiment_contributions(w, r)
        total_pct = sum(c.pct_of_total for c in contribs)
        assert total_pct == pytest.approx(1.0)

    def test_zero_total(self):
        w = {"A": 0.5, "B": 0.5}
        r = {"A": 0.01, "B": -0.01}
        contribs = PerformanceAttribution.experiment_contributions(w, r)
        assert all(c.pct_of_total == 0.0 for c in contribs)

    def test_single_experiment(self):
        contribs = PerformanceAttribution.experiment_contributions(
            {"solo": 1.0}, {"solo": 0.05})
        assert len(contribs) == 1
        assert contribs[0].contribution == pytest.approx(0.05)
        assert contribs[0].pct_of_total == pytest.approx(1.0)

    def test_missing_return(self):
        contribs = PerformanceAttribution.experiment_contributions(
            {"A": 0.5, "B": 0.5}, {"A": 0.02})
        b = [c for c in contribs if c.name == "B"][0]
        assert b.contribution == 0.0

    def test_ts_shape(self):
        idx = pd.bdate_range("2026-01-01", periods=10)
        w = pd.DataFrame({"A": [0.6] * 10, "B": [0.4] * 10}, index=idx)
        r = pd.DataFrame(np.random.default_rng(1).normal(0.001, 0.01, (10, 2)),
                          index=idx, columns=["A", "B"])
        ts = PerformanceAttribution.experiment_contributions_ts(w, r)
        assert ts.shape == (10, 2)


# ===========================================================================
# Skill vs luck
# ===========================================================================

class TestSkillTest:
    def test_significant_alpha(self):
        """Large persistent alpha should be significant."""
        n = 500
        rng = np.random.default_rng(42)
        mkt = pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
        port = mkt + 0.002  # 2bps daily alpha
        pa = PerformanceAttribution(bootstrap_n=2000, confidence=0.95)
        res = pa.skill_test(port, mkt)
        assert isinstance(res, SkillTestResult)
        assert res.is_significant
        assert res.p_value < 0.05

    def test_no_alpha(self):
        """No excess return → not significant."""
        n = 300
        rng = np.random.default_rng(42)
        mkt = pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
        pa = PerformanceAttribution(bootstrap_n=2000)
        res = pa.skill_test(mkt, mkt)  # port == benchmark
        assert not res.is_significant
        assert res.observed_alpha == pytest.approx(0.0)

    def test_too_few_returns(self):
        port = pd.Series([0.01], index=pd.bdate_range("2026-01-01", periods=1))
        mkt = pd.Series([0.005], index=port.index)
        pa = PerformanceAttribution()
        res = pa.skill_test(port, mkt)
        assert res.p_value == 1.0
        assert not res.is_significant
        assert res.n_bootstrap == 0

    def test_bootstrap_count(self):
        pa = PerformanceAttribution(bootstrap_n=1000)
        port = _returns(100)
        mkt = _market(100)
        res = pa.skill_test(port, mkt)
        assert res.n_bootstrap == 1000

    def test_confidence_level_stored(self):
        pa = PerformanceAttribution(confidence=0.99)
        res = pa.skill_test(_returns(100), _market(100))
        assert res.confidence_level == 0.99


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        pa = PerformanceAttribution()
        fd = FactorDecomposition(total_return=0.05, market=0.03,
                                  selection=0.015, residual=0.005)
        out = tmp_path / "attr.html"
        result = pa.generate_report(fd, output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Performance Attribution" in html
        assert "Factor Decomposition" in html

    def test_waterfall_svg(self, tmp_path):
        pa = PerformanceAttribution()
        fd = FactorDecomposition(total_return=0.08, market=0.04,
                                  volatility=0.01, regime=-0.005,
                                  timing=0.015, selection=0.02)
        out = tmp_path / "attr.html"
        pa.generate_report(fd, output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "Factor Waterfall" in html

    def test_rolling_section(self, tmp_path):
        pa = PerformanceAttribution()
        port = _returns(150)
        mkt = _market(150)
        rolling = pa.rolling_factor_attribution(port, mkt, window=30)
        fd = pa.factor_attribution(port, mkt)
        out = tmp_path / "attr.html"
        pa.generate_report(fd, rolling_df=rolling, output_path=str(out))
        html = out.read_text()
        assert "Rolling Factor Exposures" in html

    def test_skill_section(self, tmp_path):
        pa = PerformanceAttribution(bootstrap_n=500)
        port = _returns(200)
        mkt = _market(200)
        fd = pa.factor_attribution(port, mkt)
        skill = pa.skill_test(port, mkt)
        out = tmp_path / "attr.html"
        pa.generate_report(fd, skill_result=skill, output_path=str(out))
        html = out.read_text()
        assert "Skill vs Luck" in html
        assert "p-value" in html

    def test_experiment_section(self, tmp_path):
        pa = PerformanceAttribution()
        fd = FactorDecomposition(total_return=0.05, market=0.03)
        contribs = PerformanceAttribution.experiment_contributions(
            {"A": 0.6, "B": 0.4}, {"A": 0.04, "B": 0.01})
        out = tmp_path / "attr.html"
        pa.generate_report(fd, experiment_contribs=contribs, output_path=str(out))
        html = out.read_text()
        assert "Experiment Contributions" in html

    def test_brinson_section(self, tmp_path):
        pa = PerformanceAttribution()
        fd = FactorDecomposition(total_return=0.05)
        ba = BrinsonAttribution(allocation_effect=0.01, selection_effect=0.02,
                                 interaction_effect=0.005, total_active=0.035)
        out = tmp_path / "attr.html"
        pa.generate_report(fd, brinson=ba, output_path=str(out))
        html = out.read_text()
        assert "Brinson Attribution" in html
        assert "Allocation" in html

    def test_full_report(self, tmp_path):
        pa = PerformanceAttribution(bootstrap_n=500)
        port = _returns(200)
        mkt = _market(200)
        fd = pa.factor_attribution(port, mkt)
        rolling = pa.rolling_factor_attribution(port, mkt, window=30)
        skill = pa.skill_test(port, mkt)
        contribs = PerformanceAttribution.experiment_contributions(
            {"X": 0.5, "Y": 0.5}, {"X": 0.03, "Y": -0.01})
        ba = pa.brinson_attribution(
            np.array([0.6, 0.4]), np.array([0.5, 0.5]),
            np.array([0.03, -0.01]), np.array([0.02, -0.005]))
        out = tmp_path / "full.html"
        result = pa.generate_report(
            fd, rolling_df=rolling, skill_result=skill,
            experiment_contribs=contribs, brinson=ba,
            output_path=str(out))
        html = Path(result).read_text()
        for section in ["Factor Waterfall", "Rolling Factor", "Skill vs Luck",
                         "Experiment Contributions", "Brinson"]:
            assert section in html

    def test_strategy_section(self, tmp_path):
        pa = PerformanceAttribution()
        fd = FactorDecomposition(total_return=0.05, market=0.03)
        trades = _trades_df(60)
        strat = pa.strategy_decomposition(trades)
        out = tmp_path / "strat.html"
        pa.generate_report(fd, strategy_decomp=strat, output_path=str(out))
        html = out.read_text()
        assert "Strategy Decomposition" in html

    def test_period_section(self, tmp_path):
        pa = PerformanceAttribution()
        port = _returns(200)
        mkt = _market(200)
        fd = pa.factor_attribution(port, mkt)
        periods = pa.period_attribution(port, mkt, freq="M")
        out = tmp_path / "period.html"
        pa.generate_report(fd, period_attrs=periods, output_path=str(out))
        html = out.read_text()
        assert "Period Attribution" in html


# ===========================================================================
# Strategy decomposition
# ===========================================================================

def _trades_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic trades with strategy types."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "strategy_type": rng.choice(["credit_spread", "iron_condor", "straddle"], n),
        "pnl": rng.normal(50, 200, n),
        "entry_date": pd.bdate_range("2024-01-02", periods=n),
    })


class TestStrategyDecomposition:
    def test_basic(self):
        trades = _trades_df(60)
        result = PerformanceAttribution.strategy_decomposition(trades)
        assert len(result) > 0
        assert all(isinstance(s, StrategyDecomposition) for s in result)

    def test_sorted_by_pnl(self):
        trades = _trades_df(100)
        result = PerformanceAttribution.strategy_decomposition(trades)
        pnls = [s.total_pnl for s in result]
        assert pnls == sorted(pnls, reverse=True)

    def test_contribution_sums_to_one(self):
        trades = _trades_df(100)
        result = PerformanceAttribution.strategy_decomposition(trades)
        total = sum(s.contribution_pct for s in result)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_win_rate_bounded(self):
        trades = _trades_df(100)
        result = PerformanceAttribution.strategy_decomposition(trades)
        for s in result:
            assert 0.0 <= s.win_rate <= 1.0

    def test_trade_counts_sum(self):
        trades = _trades_df(80)
        result = PerformanceAttribution.strategy_decomposition(trades)
        assert sum(s.n_trades for s in result) == 80

    def test_empty_trades(self):
        result = PerformanceAttribution.strategy_decomposition(pd.DataFrame())
        assert result == []

    def test_custom_columns(self):
        df = pd.DataFrame({
            "strat": ["CS", "IC", "CS"],
            "profit": [100, -50, 200],
        })
        result = PerformanceAttribution.strategy_decomposition(
            df, strategy_col="strat", pnl_col="profit"
        )
        assert len(result) == 2
        cs = [s for s in result if s.strategy == "CS"][0]
        assert cs.total_pnl == 300.0

    def test_single_strategy(self):
        df = pd.DataFrame({
            "strategy_type": ["credit_spread"] * 10,
            "pnl": [100.0] * 10,
        })
        result = PerformanceAttribution.strategy_decomposition(df)
        assert len(result) == 1
        assert result[0].contribution_pct == pytest.approx(1.0)


# ===========================================================================
# Period attribution
# ===========================================================================

class TestPeriodAttribution:
    def test_monthly(self):
        port = _returns(200)
        mkt = _market(200)
        pa = PerformanceAttribution()
        periods = pa.period_attribution(port, mkt, freq="M")
        assert len(periods) > 0
        assert all(isinstance(p, PeriodAttribution) for p in periods)

    def test_quarterly(self):
        port = _returns(300)
        mkt = _market(300)
        pa = PerformanceAttribution()
        periods = pa.period_attribution(port, mkt, freq="Q")
        assert len(periods) > 0
        # Quarterly should have fewer periods than monthly
        monthly = pa.period_attribution(port, mkt, freq="M")
        assert len(periods) < len(monthly)

    def test_period_labels(self):
        port = _returns(200)
        mkt = _market(200)
        pa = PerformanceAttribution()
        periods = pa.period_attribution(port, mkt, freq="M")
        for p in periods:
            assert isinstance(p.period_label, str)
            assert len(p.period_label) > 0

    def test_dates_set(self):
        port = _returns(200)
        mkt = _market(200)
        pa = PerformanceAttribution()
        periods = pa.period_attribution(port, mkt, freq="M")
        for p in periods:
            assert p.start_date <= p.end_date

    def test_empty_returns(self):
        port = pd.Series(dtype=float)
        mkt = pd.Series(dtype=float)
        pa = PerformanceAttribution()
        assert pa.period_attribution(port, mkt) == []

    def test_n_trades_positive(self):
        port = _returns(200)
        mkt = _market(200)
        pa = PerformanceAttribution()
        periods = pa.period_attribution(port, mkt, freq="M")
        for p in periods:
            assert p.n_trades >= 2
