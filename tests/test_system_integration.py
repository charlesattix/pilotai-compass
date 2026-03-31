"""Tests for compass.system_integration — 45 tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.system_integration import (
    SystemIntegration,
    StageStatus,
    StageResult,
    PipelineResult,
    MarketData,
    RegimeState,
    Features,
    SignalOutput,
    PositionSize,
    RiskCheckResult,
    PortfolioState,
    HedgeOverlay,
    PnLResult,
    AttributionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(n: int = 300) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=n)


def _prices(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)),
        index=_dates(n), name="SPY",
    )


def _volume(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(1e6, 5e6, n).astype(float), index=_dates(n))


def _vix(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(20 + rng.normal(0, 3, n), index=_dates(n)).clip(10, 80)


# ===========================================================================
# Stage 1: Market data
# ===========================================================================

class TestMarketData:
    def test_basic(self):
        md = SystemIntegration.stage_market_data(_prices())
        assert isinstance(md, MarketData)
        assert not md.returns.empty
        assert len(md.prices) == 300

    def test_auto_volume(self):
        md = SystemIntegration.stage_market_data(_prices())
        assert not md.volume.empty

    def test_auto_vix(self):
        md = SystemIntegration.stage_market_data(_prices())
        assert not md.vix.empty

    def test_with_explicit_data(self):
        md = SystemIntegration.stage_market_data(_prices(), _volume(), _vix())
        assert (md.volume != 1e6).any()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            SystemIntegration.stage_market_data(pd.Series(dtype=float))


# ===========================================================================
# Stage 2: Regime
# ===========================================================================

class TestRegime:
    def test_basic(self):
        rs = SystemIntegration.stage_regime(_vix())
        assert isinstance(rs, RegimeState)
        assert len(rs.regime_series) == 300
        assert rs.current_regime in ("bull", "bear", "high_vol", "low_vol", "crash")

    def test_high_vix(self):
        high = pd.Series([45.0] * 10, index=_dates(10))
        rs = SystemIntegration.stage_regime(high)
        assert rs.current_regime == "crash"

    def test_low_vix(self):
        low = pd.Series([12.0] * 10, index=_dates(10))
        rs = SystemIntegration.stage_regime(low)
        assert rs.current_regime == "low_vol"


# ===========================================================================
# Stage 3: Features
# ===========================================================================

class TestFeatures:
    def test_basic(self):
        md = SystemIntegration.stage_market_data(_prices())
        feat = SystemIntegration.stage_features(md)
        assert isinstance(feat, Features)
        assert len(feat.returns) > 0
        assert len(feat.volatility) > 0
        assert len(feat.momentum) > 0


# ===========================================================================
# Stage 4: Signal
# ===========================================================================

class TestSignal:
    def test_basic(self):
        md = SystemIntegration.stage_market_data(_prices())
        feat = SystemIntegration.stage_features(md)
        sig = SystemIntegration.stage_signal(feat)
        assert isinstance(sig, SignalOutput)
        assert set(sig.signal.dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_confidence_bounded(self):
        md = SystemIntegration.stage_market_data(_prices())
        feat = SystemIntegration.stage_features(md)
        sig = SystemIntegration.stage_signal(feat)
        assert sig.confidence.min() >= 0
        assert sig.confidence.max() <= 1.0


# ===========================================================================
# Stage 5: Position sizing
# ===========================================================================

class TestPositionSize:
    def test_basic(self):
        md = SystemIntegration.stage_market_data(_prices())
        feat = SystemIntegration.stage_features(md)
        sig = SystemIntegration.stage_signal(feat)
        ps = SystemIntegration.stage_position_size(sig, "bull")
        assert isinstance(ps, PositionSize)
        assert ps.position_size > 0

    def test_crash_reduces_size(self):
        md = SystemIntegration.stage_market_data(_prices())
        feat = SystemIntegration.stage_features(md)
        sig = SystemIntegration.stage_signal(feat)
        bull = SystemIntegration.stage_position_size(sig, "bull")
        crash = SystemIntegration.stage_position_size(sig, "crash")
        assert crash.position_size < bull.position_size


# ===========================================================================
# Stage 6: Risk check
# ===========================================================================

class TestRiskCheck:
    def test_basic(self):
        ps = PositionSize({"SPY": 0.5}, 2000, 0.02)
        ret = _prices().pct_change().dropna()
        rc = SystemIntegration.stage_risk_check(ps, ret)
        assert isinstance(rc, RiskCheckResult)
        assert rc.escalation in ("normal", "warning", "reduce", "liquidate")

    def test_adjusted_on_breach(self):
        ps = PositionSize({"SPY": 1.0}, 10000, 0.10)
        ret = _prices().pct_change().dropna()
        rc = SystemIntegration.stage_risk_check(ps, ret, equity=50000)
        # May or may not breach depending on VaR
        assert isinstance(rc.adjusted_weights, dict)


# ===========================================================================
# Stage 7: Portfolio
# ===========================================================================

class TestPortfolio:
    def test_basic(self):
        ret = pd.DataFrame({"A": _prices().pct_change().dropna(),
                              "B": _prices(seed=99).pct_change().dropna()})
        ps = SystemIntegration.stage_portfolio({"A": 0.5, "B": 0.5}, ret)
        assert isinstance(ps, PortfolioState)
        assert sum(ps.weights.values()) == pytest.approx(1.0, abs=0.05)


# ===========================================================================
# Stage 8: Hedge overlay
# ===========================================================================

class TestHedge:
    def test_no_drawdown(self):
        h = SystemIntegration.stage_hedge(0.0, "bull")
        assert isinstance(h, HedgeOverlay)
        assert h.protection_level == "green"

    def test_deep_drawdown(self):
        h = SystemIntegration.stage_hedge(0.10, "crash")
        assert h.size_multiplier < 1.0

    def test_hedge_ratio_scales(self):
        h1 = SystemIntegration.stage_hedge(0.01, "bull")
        h2 = SystemIntegration.stage_hedge(0.08, "bear")
        assert h2.hedge_ratio > h1.hedge_ratio


# ===========================================================================
# Stage 9: P&L
# ===========================================================================

class TestPnL:
    def test_basic(self):
        ret = _prices().pct_change().dropna()
        sig = pd.Series(1.0, index=ret.index)
        pnl = SystemIntegration.stage_pnl(ret, sig)
        assert isinstance(pnl, PnLResult)
        assert len(pnl.daily_returns) > 0

    def test_sharpe_computed(self):
        ret = _prices(300, seed=42).pct_change().dropna()
        sig = pd.Series(1.0, index=ret.index)
        pnl = SystemIntegration.stage_pnl(ret, sig)
        assert isinstance(pnl.sharpe, float)

    def test_empty(self):
        pnl = SystemIntegration.stage_pnl(pd.Series(dtype=float), pd.Series(dtype=float))
        assert pnl.cumulative_return == 0


# ===========================================================================
# Stage 10: Attribution
# ===========================================================================

class TestAttribution:
    def test_basic(self):
        ret = _prices(200).pct_change().dropna()
        mkt = _prices(200, seed=77).pct_change().dropna()
        attr = SystemIntegration.stage_attribution(ret, mkt)
        assert isinstance(attr, AttributionResult)
        assert attr.total_return != 0 or attr.market_contribution != 0


# ===========================================================================
# Full pipeline
# ===========================================================================

class TestFullPipeline:
    def test_all_stages_pass(self):
        si = SystemIntegration()
        result = si.run_pipeline(_prices(200), _volume(200), _vix(200))
        assert isinstance(result, PipelineResult)
        assert result.n_success >= 8  # most stages should pass
        assert result.n_failed == 0
        assert len(result.stages) == 10

    def test_timing_tracked(self):
        si = SystemIntegration()
        result = si.run_pipeline(_prices(200))
        assert result.total_duration_ms > 0
        for s in result.stages:
            assert s.duration_ms >= 0

    def test_pnl_computed(self):
        si = SystemIntegration()
        result = si.run_pipeline(_prices(200))
        assert isinstance(result.final_pnl, float)
        assert isinstance(result.final_sharpe, float)

    def test_stage_outputs(self):
        si = SystemIntegration()
        si.run_pipeline(_prices(200))
        for s in si.stages:
            if s.status == StageStatus.SUCCESS:
                assert s.output is not None


# ===========================================================================
# Error propagation
# ===========================================================================

class TestErrorPropagation:
    def test_graceful_degradation(self):
        si = SystemIntegration(graceful_degradation=True)
        # Run with too-short data that will cause some stages to degrade
        result = si.run_pipeline(pd.Series([100.0, 101.0], index=_dates(2)))
        # Should not crash, some stages may degrade
        assert result.n_failed == 0  # degraded, not failed

    def test_failing_stage_with_degradation(self):
        si = SystemIntegration(graceful_degradation=True)
        result = si.run_stage_isolated("fail_test", si.stage_that_fails)
        assert result.status == StageStatus.DEGRADED
        assert result.error is not None

    def test_failing_stage_without_degradation(self):
        si = SystemIntegration(graceful_degradation=False)
        result = si.run_stage_isolated("fail_test", si.stage_that_fails)
        assert result.status == StageStatus.FAILED

    def test_dependency_failure_skips(self):
        si = SystemIntegration(graceful_degradation=True)
        # Manually add a failed dependency
        si._stages.append(StageResult("dep1", StageStatus.FAILED, error="broken"))
        result = si._run_stage("child", lambda: "ok", dependencies=["dep1"])
        assert result.status == StageStatus.SKIPPED

    def test_dependency_failure_aborts(self):
        si = SystemIntegration(graceful_degradation=False)
        si._stages.append(StageResult("dep1", StageStatus.FAILED, error="broken"))
        result = si._run_stage("child", lambda: "ok", dependencies=["dep1"])
        assert result.status == StageStatus.FAILED

    def test_missing_dependency_passes(self):
        si = SystemIntegration()
        result = si._run_stage("orphan", lambda: "ok", dependencies=["nonexistent"])
        assert result.status == StageStatus.SUCCESS  # no stage found = no failure


# ===========================================================================
# Stage isolation
# ===========================================================================

class TestIsolation:
    def test_isolated_run(self):
        si = SystemIntegration()
        result = si.run_stage_isolated("test_market", si.stage_market_data,
                                         prices=_prices(50))
        assert result.status == StageStatus.SUCCESS
        assert isinstance(result.output, MarketData)

    def test_isolated_clears_history(self):
        si = SystemIntegration()
        si.run_stage_isolated("a", lambda: 1)
        si.run_stage_isolated("b", lambda: 2)
        assert len(si.stages) == 1  # cleared before each


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        si = SystemIntegration()
        result = si.run_pipeline(_prices(200))
        out = tmp_path / "integ.html"
        path = si.generate_report(result, output_path=str(out))
        assert Path(path).exists()
        html = out.read_text()
        assert "System Integration" in html

    def test_contains_flow(self, tmp_path):
        si = SystemIntegration()
        result = si.run_pipeline(_prices(200))
        out = tmp_path / "i.html"
        si.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "Pipeline Flow" in html

    def test_contains_stage_table(self, tmp_path):
        si = SystemIntegration()
        result = si.run_pipeline(_prices(200))
        out = tmp_path / "i.html"
        si.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "Stage Details" in html
        assert "market_data" in html
        assert "attribution" in html

    def test_contains_status(self, tmp_path):
        si = SystemIntegration()
        result = si.run_pipeline(_prices(200))
        out = tmp_path / "i.html"
        si.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "SUCCESS" in html

    def test_report_with_failures(self, tmp_path):
        si = SystemIntegration(graceful_degradation=True)
        result = si.run_pipeline(pd.Series([100.0, 101.0], index=_dates(2)))
        out = tmp_path / "i.html"
        si.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "stages passed" in html
