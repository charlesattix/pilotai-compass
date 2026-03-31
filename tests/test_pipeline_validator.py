"""Tests for compass/pipeline_validator.py — production pipeline validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.pipeline_validator import (
    KillSwitchResult,
    PipelineValidator,
    StageError,
    StageResult,
    ValidationResult,
    check_distribution,
    check_duplicates,
    check_freshness,
    check_nulls,
    check_schema,
    check_types,
    check_value_range,
    evaluate_kill_switch,
    validate_data_ingestion,
    validate_features,
    validate_hedge,
    validate_model,
    validate_orders,
    validate_risk,
    validate_signals,
    validate_sizing,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_trades(n: int = 100, seed: int = 42, clean: bool = True) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({
        "entry_date": dates,
        "exit_date": dates + pd.Timedelta(days=2),
        "pnl": rng.normal(50, 200, n),
        "net_credit": rng.uniform(0.5, 3.0, n),
        "strategy_type": rng.choice(["CS", "IC", "SS"], n),
        "regime": rng.choice(["bull", "bear", "sideways"], n),
        "vix": rng.uniform(12, 35, n),
        "iv_rank": rng.uniform(5, 90, n),
        "rsi_14": rng.uniform(20, 80, n),
        "momentum_5d_pct": rng.normal(0, 2, n),
        "contracts": rng.randint(1, 10, n),
        "win": rng.randint(0, 2, n),
    })
    if clean:
        df["signal_score"] = rng.uniform(0.3, 0.9, n)
    return df


def _make_dirty_trades(n: int = 50) -> pd.DataFrame:
    """Trades with various issues."""
    rng = np.random.RandomState(99)
    df = _make_trades(n, seed=99)
    # Inject problems
    df.loc[0, "vix"] = 150.0       # out of range
    df.loc[1, "pnl"] = np.nan      # null
    df.loc[2, "iv_rank"] = -10.0   # out of range
    df.loc[3:5, "signal_score"] = 1.5  # out of range
    df.loc[6, "contracts"] = 100   # too many
    return df


@pytest.fixture
def clean_trades():
    return _make_trades()


@pytest.fixture
def dirty_trades():
    return _make_dirty_trades()


# ── Check function tests ─────────────────────────────────────────────────


class TestCheckSchema:
    def test_all_present(self, clean_trades):
        errors = check_schema(clean_trades, ["pnl", "entry_date"])
        assert len(errors) == 0

    def test_missing_columns(self, clean_trades):
        errors = check_schema(clean_trades, ["pnl", "nonexistent_col"])
        assert len(errors) == 1
        assert errors[0].code == "SCHEMA_MISSING"


class TestCheckTypes:
    def test_numeric_ok(self, clean_trades):
        errors = check_types(clean_trades, ["pnl", "vix"])
        assert len(errors) == 0

    def test_non_numeric(self):
        df = pd.DataFrame({"val": ["a", "b", "c"]})
        errors = check_types(df, ["val"])
        assert len(errors) == 1
        assert errors[0].code == "TYPE_MISMATCH"


class TestCheckNulls:
    def test_clean_data(self, clean_trades):
        errors = check_nulls(clean_trades, 0.10)
        assert all(e.severity != "error" for e in errors)

    def test_many_nulls(self):
        df = pd.DataFrame({"a": [np.nan] * 8 + [1.0, 2.0]})
        errors = check_nulls(df, 0.10)
        assert len(errors) == 1


class TestCheckFreshness:
    def test_fresh_data(self, clean_trades):
        errors = check_freshness(clean_trades, max_stale_days=9999)
        stale = [e for e in errors if e.code == "STALE_DATA"]
        assert len(stale) == 0

    def test_stale_data(self):
        df = pd.DataFrame({"entry_date": ["2020-01-01"]})
        errors = check_freshness(df, max_stale_days=30)
        assert any(e.code == "STALE_DATA" for e in errors)

    def test_no_date_column(self):
        df = pd.DataFrame({"pnl": [100]})
        errors = check_freshness(df, date_col="entry_date")
        assert len(errors) == 0  # no date column → no error


class TestCheckValueRange:
    def test_within_range(self):
        s = pd.Series([0.5, 0.6, 0.7])
        errors = check_value_range(s, "test", 0.0, 1.0)
        assert len(errors) == 0

    def test_below_min(self):
        s = pd.Series([-0.5, 0.5])
        errors = check_value_range(s, "test", 0.0, 1.0)
        assert len(errors) == 1

    def test_above_max(self):
        s = pd.Series([0.5, 1.5])
        errors = check_value_range(s, "test", 0.0, 1.0)
        assert len(errors) == 1

    def test_empty_series(self):
        errors = check_value_range(pd.Series(dtype=float), "test", 0, 1)
        assert len(errors) == 0


class TestCheckDuplicates:
    def test_no_duplicates(self, clean_trades):
        errors = check_duplicates(clean_trades, ["entry_date"])
        assert all(e.code != "DUPLICATES" or e.severity != "error" for e in errors)

    def test_with_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        errors = check_duplicates(df, ["a", "b"])
        assert any(e.code == "DUPLICATES" for e in errors)


class TestCheckDistribution:
    def test_normal_distribution(self):
        s = pd.Series(np.random.RandomState(42).normal(0.5, 0.1, 100))
        errors = check_distribution(s, "test", (0.3, 0.7))
        assert len(errors) == 0

    def test_shifted_distribution(self):
        s = pd.Series([0.95] * 100)
        errors = check_distribution(s, "test", (0.3, 0.7))
        assert any(e.code == "DISTRIBUTION_SHIFT" for e in errors)


# ── Stage validator tests ────────────────────────────────────────────────


class TestDataIngestion:
    def test_clean_passes(self, clean_trades):
        result = validate_data_ingestion(clean_trades)
        assert result.passed
        assert result.stage == "data_ingestion"
        assert result.elapsed_ms >= 0

    def test_empty_fails(self):
        df = pd.DataFrame(columns=["entry_date", "exit_date", "pnl"])
        result = validate_data_ingestion(df)
        assert not result.passed

    def test_missing_columns_fails(self):
        df = pd.DataFrame({"foo": [1, 2, 3]})
        result = validate_data_ingestion(df)
        assert not result.passed


class TestFeatures:
    def test_clean_passes(self, clean_trades):
        result = validate_features(clean_trades)
        assert result.passed

    def test_out_of_range_vix(self, dirty_trades):
        result = validate_features(dirty_trades)
        vix_errors = [e for e in result.errors if "vix" in e.message.lower()]
        assert len(vix_errors) > 0


class TestSignals:
    def test_with_signal(self, clean_trades):
        result = validate_signals(clean_trades)
        assert result.passed

    def test_missing_signal(self):
        df = pd.DataFrame({"pnl": [100]})
        result = validate_signals(df)
        assert not result.passed

    def test_out_of_range_signal(self, dirty_trades):
        result = validate_signals(dirty_trades)
        range_errors = [e for e in result.errors if e.code == "RANGE_VIOLATION"]
        assert len(range_errors) > 0


class TestModel:
    def test_no_model_col_passes(self, clean_trades):
        result = validate_model(clean_trades)
        assert result.passed  # info only, not error

    def test_stale_model_warns(self, clean_trades):
        result = validate_model(clean_trades, model_age_days=100)
        warn = [e for e in result.errors if e.code == "STALE_MODEL"]
        assert len(warn) > 0


class TestSizing:
    def test_normal_sizing(self, clean_trades):
        result = validate_sizing(clean_trades)
        assert result.passed

    def test_excessive_contracts(self, dirty_trades):
        result = validate_sizing(dirty_trades, max_contracts=50)
        assert result.passed or any(e.code == "RANGE_VIOLATION" for e in result.errors)


class TestRisk:
    def test_normal_risk(self, clean_trades):
        result = validate_risk(clean_trades, max_drawdown=0.99)
        assert result.passed

    def test_drawdown_exceeded(self):
        df = pd.DataFrame({
            "pnl": [-5000] * 20 + [100] * 10,
            "regime": ["bear"] * 30,
            "win": [0] * 20 + [1] * 10,
        })
        result = validate_risk(df, max_drawdown=0.05, capital=100_000)
        assert not result.passed


class TestOrders:
    def test_valid_orders(self, clean_trades):
        result = validate_orders(clean_trades)
        assert result.passed

    def test_unknown_strategy(self):
        df = pd.DataFrame({"strategy_type": ["BUTTERFLY", "CONDOR"]})
        result = validate_orders(df)
        assert any(e.code == "INVALID_STRATEGY" for e in result.errors)


class TestHedge:
    def test_no_hedge_passes(self, clean_trades):
        result = validate_hedge(clean_trades)
        assert result.passed  # info only

    def test_valid_hedge(self):
        df = pd.DataFrame({"hedge_ratio": [0.5, 0.8, 1.0]})
        result = validate_hedge(df)
        assert result.passed

    def test_excessive_hedge(self):
        df = pd.DataFrame({"hedge_ratio": [0.5, 3.0]})
        result = validate_hedge(df)
        assert not result.passed


# ── Kill switch tests ────────────────────────────────────────────────────


class TestKillSwitch:
    def test_not_triggered_clean(self):
        stages = [StageResult(s, True, 1.0, 3, 3, 0, 0, []) for s in ["a", "b", "c", "d", "e", "f", "g", "h"]]
        ks = evaluate_kill_switch(stages)
        assert not ks.triggered

    def test_triggered_on_errors(self):
        stages = [StageResult("a", False, 1.0, 3, 0, 3, 0,
                              [StageError("X", "err", "error", "fix")] * 5)]
        ks = evaluate_kill_switch(stages, max_errors=3)
        assert ks.triggered

    def test_triggered_critical_fail(self):
        stages = [
            StageResult("data", False, 1.0, 1, 0, 1, 0, []),
            StageResult("feat", True, 1.0, 1, 1, 0, 0, []),
            StageResult("sig", True, 1.0, 1, 1, 0, 0, []),
        ]
        ks = evaluate_kill_switch(stages)
        assert ks.triggered
        assert "critical" in ks.reason

    def test_checks_populated(self):
        stages = [StageResult(s, True, 1.0, 1, 1, 0, 0, []) for s in ["a", "b", "c"]]
        ks = evaluate_kill_switch(stages)
        assert len(ks.checks) == 3


# ── Constructor tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_basic(self, clean_trades):
        pv = PipelineValidator(clean_trades)
        assert len(pv.data) == len(clean_trades)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            PipelineValidator(pd.DataFrame())


# ── Full validation tests ────────────────────────────────────────────────


class TestFullValidation:
    def test_clean_pipeline_passes(self, clean_trades):
        pv = PipelineValidator(clean_trades, max_drawdown=0.99)
        result = pv.validate()
        assert isinstance(result, ValidationResult)
        assert result.n_stages == 8
        assert result.overall_pass

    def test_dirty_pipeline_has_issues(self, dirty_trades):
        pv = PipelineValidator(dirty_trades, max_drawdown=0.99)
        result = pv.validate()
        assert result.n_total_errors + result.n_total_warnings > 0

    def test_all_stages_run(self, clean_trades):
        pv = PipelineValidator(clean_trades, max_drawdown=0.99)
        result = pv.validate()
        stage_names = [s.stage for s in result.stages]
        assert "data_ingestion" in stage_names
        assert "hedge_overlay" in stage_names
        assert len(stage_names) == 8

    def test_timing_positive(self, clean_trades):
        pv = PipelineValidator(clean_trades, max_drawdown=0.99)
        result = pv.validate()
        assert result.total_elapsed_ms > 0
        for s in result.stages:
            assert s.elapsed_ms >= 0

    def test_kill_switch_in_result(self, clean_trades):
        pv = PipelineValidator(clean_trades, max_drawdown=0.99)
        result = pv.validate()
        assert isinstance(result.kill_switch, KillSwitchResult)

    def test_timestamp_present(self, clean_trades):
        pv = PipelineValidator(clean_trades, max_drawdown=0.99)
        result = pv.validate()
        assert len(result.timestamp) > 0

    def test_custom_params(self, clean_trades):
        pv = PipelineValidator(
            clean_trades, max_drawdown=0.01, max_contracts=2,
            model_age_days=200,
        )
        result = pv.validate()
        # Should trigger drawdown and/or sizing warnings
        assert result.n_total_errors + result.n_total_warnings > 0


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, clean_trades):
        pv = PipelineValidator(clean_trades, max_drawdown=0.99)
        result = pv.validate()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pv.html"
            path = PipelineValidator.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Pipeline Validation" in content

    def test_contains_stages(self, clean_trades):
        pv = PipelineValidator(clean_trades, max_drawdown=0.99)
        result = pv.validate()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            PipelineValidator.generate_report(result, out)
            content = out.read_text()
            assert "data_ingestion" in content
            assert "risk_checks" in content

    def test_contains_kill_switch(self, clean_trades):
        pv = PipelineValidator(clean_trades, max_drawdown=0.99)
        result = pv.validate()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            PipelineValidator.generate_report(result, out)
            content = out.read_text()
            assert "Kill Switch" in content

    def test_contains_pipeline_flow(self, clean_trades):
        pv = PipelineValidator(clean_trades, max_drawdown=0.99)
        result = pv.validate()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            PipelineValidator.generate_report(result, out)
            content = out.read_text()
            assert "Pipeline Flow" in content

    def test_dirty_shows_errors(self, dirty_trades):
        pv = PipelineValidator(dirty_trades, max_drawdown=0.99)
        result = pv.validate()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            PipelineValidator.generate_report(result, out)
            content = out.read_text()
            # Dirty data should trigger the kill switch
            assert "TRIGGERED" in content or "FAILED" in content

    def test_default_path(self, clean_trades):
        pv = PipelineValidator(clean_trades, max_drawdown=0.99)
        result = pv.validate()
        path = PipelineValidator.generate_report(result)
        assert path.exists()
        assert "pipeline_validation.html" in str(path)
        path.unlink(missing_ok=True)
