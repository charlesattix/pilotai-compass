"""Tests for compass.master_runner — 36 tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from compass.master_runner import (
    MasterRunner, RunConfig, StageStatus, StageResult, MasterRunResult, STAGE_NAMES,
)

def _prices(n=252, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=idx)


class TestConfig:
    def test_default(self):
        c = RunConfig()
        assert len(c.enabled_stages) == 12
        assert c.dry_run is False
    def test_custom(self):
        c = RunConfig(experiments=["EXP-1"], dry_run=True)
        assert c.dry_run
        assert "EXP-1" in c.experiments


class TestStages:
    def test_validate(self):
        mr = MasterRunner()
        result = mr._run_stage("validate")
        assert result.status == StageStatus.SUCCESS
    def test_readiness(self):
        mr = MasterRunner()
        result = mr._run_stage("readiness")
        assert result.status == StageStatus.SUCCESS
    def test_data(self):
        mr = MasterRunner()
        result = mr._run_stage("data")
        assert result.status == StageStatus.SUCCESS
        assert "n_days" in result.output
    def test_features(self):
        mr = MasterRunner()
        mr._run_stage("data")
        result = mr._run_stage("features")
        assert result.status == StageStatus.SUCCESS
    def test_signals(self):
        mr = MasterRunner()
        mr._run_stage("data")
        mr._run_stage("features")
        result = mr._run_stage("signals")
        assert result.status == StageStatus.SUCCESS
    def test_model(self):
        mr = MasterRunner()
        mr._run_stage("data"); mr._run_stage("features"); mr._run_stage("signals")
        result = mr._run_stage("model")
        assert result.status == StageStatus.SUCCESS
    def test_sizing(self):
        mr = MasterRunner()
        mr._run_stage("data"); mr._run_stage("features"); mr._run_stage("signals")
        result = mr._run_stage("sizing")
        assert result.status == StageStatus.SUCCESS
    def test_risk(self):
        mr = MasterRunner()
        mr._run_stage("data"); mr._run_stage("features"); mr._run_stage("signals"); mr._run_stage("sizing")
        result = mr._run_stage("risk")
        assert result.status == StageStatus.SUCCESS
    def test_orders(self):
        mr = MasterRunner()
        for s in ["data", "features", "signals", "sizing", "risk"]:
            mr._run_stage(s)
        result = mr._run_stage("orders")
        assert result.status == StageStatus.SUCCESS
    def test_hedge(self):
        mr = MasterRunner()
        mr._run_stage("data")
        result = mr._run_stage("hedge")
        assert result.status == StageStatus.SUCCESS
    def test_pnl(self):
        mr = MasterRunner()
        for s in ["data", "features", "signals"]:
            mr._run_stage(s)
        result = mr._run_stage("pnl")
        assert result.status == StageStatus.SUCCESS
    def test_reports(self):
        mr = MasterRunner()
        result = mr._run_stage("reports")
        assert result.status == StageStatus.SUCCESS


class TestFullRun:
    def test_all_pass(self):
        mr = MasterRunner()
        result = mr.run()
        assert isinstance(result, MasterRunResult)
        assert result.n_success == 12
        assert result.n_failed == 0
    def test_12_stages(self):
        mr = MasterRunner()
        result = mr.run()
        assert len(result.stages) == 12
    def test_timing(self):
        mr = MasterRunner()
        result = mr.run()
        assert result.total_duration_ms > 0
        for s in result.stages:
            assert s.duration_ms >= 0
    def test_context_populated(self):
        mr = MasterRunner()
        mr.run()
        assert "data" in mr.context
        assert "signal" in mr.context
    def test_with_custom_data(self):
        prices = _prices(100)
        config = RunConfig(data={"prices": prices, "returns": prices.pct_change().dropna(),
                                   "volume": pd.Series(1e6, index=prices.index)})
        mr = MasterRunner(config)
        result = mr.run()
        assert result.n_success >= 10


class TestDryRun:
    def test_all_dry(self):
        config = RunConfig(dry_run=True)
        mr = MasterRunner(config)
        result = mr.run()
        assert all(s.status == StageStatus.DRY_RUN for s in result.stages)
        assert result.n_skipped == 12


class TestSkipStages:
    def test_skip_disabled(self):
        config = RunConfig(enabled_stages=["data", "features"])
        mr = MasterRunner(config)
        result = mr.run()
        enabled = [s for s in result.stages if s.status == StageStatus.SUCCESS]
        skipped = [s for s in result.stages if s.status == StageStatus.SKIPPED]
        assert len(enabled) == 2
        assert len(skipped) == 10


class TestErrorHandling:
    def test_bad_readiness(self):
        config = RunConfig(experiments=[], account_size=0)
        mr = MasterRunner(config)
        result = mr.run()
        readiness = [s for s in result.stages if s.name == "readiness"][0]
        assert readiness.status == StageStatus.FAILED
    def test_continues_after_failure(self):
        config = RunConfig(experiments=[], account_size=0)
        mr = MasterRunner(config)
        result = mr.run()
        # Later stages should still attempt
        assert len(result.stages) == 12


class TestReport:
    def test_creates_file(self, tmp_path):
        mr = MasterRunner()
        result = mr.run()
        out = tmp_path / "master.html"
        path = mr.generate_report(result, output_path=str(out))
        assert Path(path).exists()
        assert "Master Pipeline" in out.read_text()
    def test_contains_waterfall(self, tmp_path):
        mr = MasterRunner()
        result = mr.run()
        out = tmp_path / "m.html"
        mr.generate_report(result, output_path=str(out))
        assert "<svg" in out.read_text()
    def test_contains_stages(self, tmp_path):
        mr = MasterRunner()
        result = mr.run()
        out = tmp_path / "m.html"
        mr.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "Stage Details" in html
        assert "validate" in html
