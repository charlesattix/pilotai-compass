"""
Tests for compass/experiment_launcher.py — Experiment launcher with config
validation, pre-flight checks, dry-run trade simulation, and rollback.

Covers:
  - Schema validation (missing fields, type errors, ranges, cross-field)
  - Pre-flight checks (training data, models, features, compatibility,
    risk gates, hedge params)
  - Dry-run simulation (pipeline stages, 10 simulated trades)
  - Lifecycle (start, stop, rollback, update_config, state transitions)
  - PreflightReport properties
  - HTML launch report (all sections, simulated trades table)
  - Edge cases (empty config, already running, no rollback snapshot)
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import pytest

from compass.experiment_launcher import (
    CheckResult,
    DryRunResult,
    ExperimentLauncher,
    PreflightReport,
    REQUIRED_FEATURES,
    SimulatedTrade,
    ValidationError,
    _check_feature_compatibility,
    _check_feature_pipeline,
    _check_hedge_params,
    _check_risk_gates,
    _check_training_data,
    _simulate_trades,
    simulate_dry_run,
    validate_config,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_config():
    """A fully valid experiment config."""
    return {
        "strategy": {
            "min_dte": 30,
            "max_dte": 45,
            "min_delta": 0.20,
            "max_delta": 0.30,
            "spread_width": 5,
        },
        "risk": {
            "max_risk_per_trade": 5.0,
            "max_positions": 6,
            "profit_target": 50,
            "stop_loss_pct_of_width": 75,
            "max_portfolio_risk_pct": 40,
        },
        "backtest": {
            "starting_capital": 100_000,
            "commission_per_contract": 0.65,
        },
    }


@pytest.fixture
def base_dir(tmp_path):
    """Temp dir mimicking project structure with training data and features."""
    compass_dir = tmp_path / "compass"
    compass_dir.mkdir()
    # Training data with required feature columns
    header = ",".join(sorted(REQUIRED_FEATURES)) + ",extra_col"
    (compass_dir / "training_data_combined.csv").write_text(header + "\n1,2,3,4,5,6,7,8,9\n")
    # Feature pipeline
    (compass_dir / "feature_pipeline.py").write_text("# stub\n")
    # Model file
    (compass_dir / "signal_model.joblib").write_text("stub")
    return tmp_path


@pytest.fixture
def launcher(valid_config, base_dir):
    """Ready-to-use ExperimentLauncher."""
    return ExperimentLauncher(
        config=valid_config,
        experiment_id="EXP-TEST",
        base_dir=base_dir,
        backtest_metrics={"sharpe": 1.5, "max_dd_pct": 12.0, "win_rate": 65.0},
    )


# ── Validation tests ─────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_config_no_errors(self, valid_config):
        errors = validate_config(valid_config)
        assert len(errors) == 0

    def test_missing_top_level_section(self, valid_config):
        del valid_config["risk"]
        errors = validate_config(valid_config)
        assert any(e.field == "risk" for e in errors)

    def test_missing_all_sections(self):
        errors = validate_config({})
        assert len(errors) >= 3

    def test_missing_strategy_field(self, valid_config):
        del valid_config["strategy"]["min_dte"]
        errors = validate_config(valid_config)
        assert any(e.field == "min_dte" for e in errors)

    def test_wrong_type(self, valid_config):
        valid_config["strategy"]["min_dte"] = "thirty"
        errors = validate_config(valid_config)
        assert any("type" in e.message.lower() for e in errors)

    def test_below_minimum(self, valid_config):
        valid_config["strategy"]["min_delta"] = -0.5
        errors = validate_config(valid_config)
        assert any("below minimum" in e.message for e in errors)

    def test_above_maximum(self, valid_config):
        valid_config["strategy"]["max_delta"] = 5.0
        errors = validate_config(valid_config)
        assert any("above maximum" in e.message for e in errors)

    def test_cross_field_min_dte_gte_max_dte(self, valid_config):
        valid_config["strategy"]["min_dte"] = 50
        valid_config["strategy"]["max_dte"] = 30
        errors = validate_config(valid_config)
        assert any("min_dte" in e.field and "max_dte" in e.field for e in errors)

    def test_cross_field_min_delta_gte_max_delta(self, valid_config):
        valid_config["strategy"]["min_delta"] = 0.40
        valid_config["strategy"]["max_delta"] = 0.20
        errors = validate_config(valid_config)
        assert any("min_delta" in e.field for e in errors)

    def test_risk_field_missing(self, valid_config):
        del valid_config["risk"]["max_positions"]
        errors = validate_config(valid_config)
        assert any(e.section == "risk" and e.field == "max_positions" for e in errors)

    def test_backtest_capital_too_low(self, valid_config):
        valid_config["backtest"]["starting_capital"] = 100
        errors = validate_config(valid_config)
        assert any("below minimum" in e.message for e in errors)

    def test_validation_error_dataclass(self):
        ve = ValidationError(section="test", field="f", message="bad")
        assert ve.severity == "error"


# ── Pre-flight check tests ────────────────────────────────────────────────────

class TestPreflightChecks:
    def test_training_data_found(self, valid_config, base_dir):
        result = _check_training_data(valid_config, base_dir)
        assert result.passed

    def test_training_data_missing(self, valid_config, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "compass").mkdir()
        result = _check_training_data(valid_config, empty)
        assert not result.passed

    def test_feature_pipeline_found(self, valid_config, base_dir):
        result = _check_feature_pipeline(valid_config, base_dir)
        assert result.passed

    def test_feature_pipeline_missing(self, valid_config, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "compass").mkdir()
        result = _check_feature_pipeline(valid_config, empty)
        assert not result.passed

    def test_feature_compatibility_pass(self, valid_config, base_dir):
        result = _check_feature_compatibility(valid_config, base_dir)
        assert result.passed

    def test_feature_compatibility_missing_columns(self, valid_config, tmp_path):
        compass = tmp_path / "compat" / "compass"
        compass.mkdir(parents=True)
        (compass / "training_data_test.csv").write_text("col_a,col_b\n1,2\n")
        result = _check_feature_compatibility(valid_config, tmp_path / "compat")
        assert not result.passed
        assert "Missing" in result.message

    def test_feature_compatibility_no_csv(self, valid_config, tmp_path):
        compass = tmp_path / "nocsv" / "compass"
        compass.mkdir(parents=True)
        result = _check_feature_compatibility(valid_config, tmp_path / "nocsv")
        assert not result.passed

    def test_risk_gates_configured(self, valid_config, base_dir):
        result = _check_risk_gates(valid_config, base_dir)
        assert result.passed

    def test_risk_gates_missing_config(self, base_dir):
        result = _check_risk_gates({}, base_dir)
        assert not result.passed

    def test_hedge_params_configured(self, valid_config, base_dir):
        result = _check_hedge_params(valid_config, base_dir)
        assert result.passed

    def test_hedge_params_missing_stop(self, base_dir):
        config = {"risk": {"profit_target": 50}}
        result = _check_hedge_params(config, base_dir)
        assert not result.passed


# ── Simulated trades tests ────────────────────────────────────────────────────

class TestSimulatedTrades:
    def test_returns_correct_count(self, valid_config):
        trades = _simulate_trades(valid_config, n_trades=10)
        assert len(trades) == 10

    def test_trade_has_correct_fields(self, valid_config):
        trades = _simulate_trades(valid_config, n_trades=1)
        t = trades[0]
        assert isinstance(t, SimulatedTrade)
        assert t.trade_id == 1
        assert t.direction in ("bull_put", "bear_call")
        assert t.contracts >= 1
        assert t.credit > 0
        assert t.max_loss > 0
        assert t.outcome in ("win", "loss")

    def test_deterministic_with_seed(self, valid_config):
        t1 = _simulate_trades(valid_config, n_trades=5, seed=99)
        t2 = _simulate_trades(valid_config, n_trades=5, seed=99)
        assert [t.pnl for t in t1] == [t.pnl for t in t2]

    def test_risk_gate_blocks_excess_positions(self, valid_config):
        valid_config["risk"]["max_positions"] = 3
        trades = _simulate_trades(valid_config, n_trades=10)
        blocked = [t for t in trades if not t.risk_gate_passed]
        assert len(blocked) > 0  # trades beyond position 3 should be blocked

    def test_win_loss_pnl_signs(self, valid_config):
        trades = _simulate_trades(valid_config, n_trades=20, seed=42)
        for t in trades:
            if t.outcome == "win":
                assert t.pnl > 0
            else:
                assert t.pnl < 0


# ── Dry-run tests ─────────────────────────────────────────────────────────────

class TestDryRun:
    def test_returns_dry_run_result(self, valid_config, base_dir):
        result = simulate_dry_run(valid_config, base_dir)
        assert isinstance(result, DryRunResult)

    def test_all_stages_present(self, valid_config, base_dir):
        result = simulate_dry_run(valid_config, base_dir)
        stages = {s.stage for s in result.steps}
        assert "load_config" in stages
        assert "build_features" in stages
        assert "run_model" in stages
        assert "check_sizing" in stages
        assert "risk_gates" in stages
        assert "simulate_trades" in stages
        assert "summary" in stages

    def test_has_simulated_trades(self, valid_config, base_dir):
        result = simulate_dry_run(valid_config, base_dir, n_trades=10)
        assert len(result.simulated_trades) == 10

    def test_would_trade_with_valid_config(self, valid_config, base_dir):
        result = simulate_dry_run(valid_config, base_dir)
        assert result.would_trade

    def test_blocked_by_excessive_risk(self, valid_config, base_dir):
        valid_config["risk"]["max_risk_per_trade"] = 10.0
        result = simulate_dry_run(valid_config, base_dir)
        assert not result.would_trade
        assert result.risk_gate_result == "block"

    def test_estimated_contracts_positive(self, valid_config, base_dir):
        result = simulate_dry_run(valid_config, base_dir)
        assert result.estimated_contracts >= 1

    def test_summary_has_simulated_metrics(self, valid_config, base_dir):
        result = simulate_dry_run(valid_config, base_dir)
        assert "simulated_pnl" in result.summary
        assert "simulated_win_rate" in result.summary


# ── Lifecycle tests ───────────────────────────────────────────────────────────

class TestLifecycle:
    def test_initial_state_is_idle(self, launcher):
        assert launcher.state == "idle"
        assert not launcher.is_running

    def test_start_transitions_to_running(self, launcher):
        launcher.start()
        assert launcher.state == "running"
        assert launcher.is_running

    def test_start_saves_previous_config(self, launcher):
        original = copy.deepcopy(launcher.config)
        launcher.start()
        assert launcher.previous_config == original

    def test_start_raises_if_already_running(self, launcher):
        launcher.start()
        with pytest.raises(RuntimeError, match="already running"):
            launcher.start()

    def test_start_raises_on_validation_errors(self, base_dir):
        bad_config = {"strategy": {"min_dte": "bad"}}
        launcher = ExperimentLauncher(bad_config, base_dir=base_dir)
        with pytest.raises(RuntimeError, match="validation error"):
            launcher.start()

    def test_stop_transitions_to_stopped(self, launcher):
        launcher.start()
        launcher.stop(reason="test done")
        assert launcher.state == "stopped"
        assert not launcher.is_running

    def test_stop_raises_if_not_running(self, launcher):
        with pytest.raises(RuntimeError, match="not running"):
            launcher.stop()

    def test_rollback_restores_config(self, launcher):
        original = copy.deepcopy(launcher.config)
        launcher.start()
        launcher.config["strategy"]["spread_width"] = 99
        restored = launcher.rollback()
        assert restored == original
        assert launcher.config == original

    def test_rollback_stops_running_experiment(self, launcher):
        launcher.start()
        launcher.rollback()
        assert launcher.state == "stopped"

    def test_rollback_raises_with_no_snapshot(self, valid_config, base_dir):
        launcher = ExperimentLauncher(valid_config, base_dir=base_dir)
        with pytest.raises(RuntimeError, match="No previous config"):
            launcher.rollback()

    def test_update_config_validates_first(self, launcher):
        bad = {"strategy": {"min_dte": "bad"}}
        errors = launcher.update_config(bad)
        assert len(errors) > 0
        # Config should NOT have changed
        assert launcher.config != bad

    def test_update_config_saves_rollback(self, launcher):
        original = copy.deepcopy(launcher.config)
        new_config = copy.deepcopy(launcher.config)
        new_config["strategy"]["spread_width"] = 10
        errors = launcher.update_config(new_config)
        assert len([e for e in errors if e.severity == "error"]) == 0
        assert launcher.config == new_config
        assert launcher.previous_config == original

    def test_update_config_raises_while_running(self, launcher):
        launcher.start()
        with pytest.raises(RuntimeError, match="running"):
            launcher.update_config(launcher.config)

    def test_start_with_force_bypasses_check_failures(self, valid_config, tmp_path):
        # No training data / model = failed checks, but no validation errors
        empty = tmp_path / "force"
        empty.mkdir()
        (empty / "compass").mkdir()
        (empty / "compass" / "feature_pipeline.py").write_text("# stub")
        launcher = ExperimentLauncher(valid_config, base_dir=empty)
        # Without force: should raise
        with pytest.raises(RuntimeError, match="pre-flight check"):
            launcher.start(force=False)
        # Reset state
        launcher._state = "idle"
        # With force: should succeed
        launcher.start(force=True)
        assert launcher.is_running


# ── PreflightReport tests ────────────────────────────────────────────────────

class TestPreflightReport:
    def test_all_passed_with_valid(self, launcher):
        report = launcher.preflight()
        assert isinstance(report, PreflightReport)
        assert report.n_errors == 0

    def test_n_checks_total(self, launcher):
        report = launcher.preflight()
        assert report.n_checks_total == 6  # 6 default checks now

    def test_dry_run_included(self, launcher):
        report = launcher.preflight(include_dry_run=True)
        assert report.dry_run is not None
        assert len(report.dry_run.simulated_trades) == 10

    def test_dry_run_excluded(self, launcher):
        report = launcher.preflight(include_dry_run=False)
        assert report.dry_run is None


# ── HTML report tests ─────────────────────────────────────────────────────────

class TestHTMLReport:
    def test_generates_valid_html(self, launcher):
        report = launcher.preflight()
        html = launcher.generate_html(report)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_experiment_id(self, launcher):
        report = launcher.preflight()
        html = launcher.generate_html(report)
        assert "EXP-TEST" in html

    def test_contains_simulated_trades_table(self, launcher):
        report = launcher.preflight()
        html = launcher.generate_html(report)
        assert "Simulated Trades" in html
        assert "bull_put" in html or "bear_call" in html

    def test_contains_status_badge(self, launcher):
        report = launcher.preflight()
        html = launcher.generate_html(report)
        assert "READY" in html

    def test_contains_all_sections(self, launcher):
        report = launcher.preflight()
        html = launcher.generate_html(report)
        assert "Validation" in html
        assert "Pre-Flight Checks" in html
        assert "Dry Run" in html
        assert "Config Summary" in html
        assert "Estimated Performance" in html

    def test_blocked_status_badge(self, base_dir):
        launcher = ExperimentLauncher({}, base_dir=base_dir)
        report = launcher.preflight()
        html = launcher.generate_html(report)
        assert "NOT READY" in html

    def test_writes_file(self, launcher):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "launch.html")
            report = launcher.generate_report_file(output_path=path)
            assert os.path.isfile(path)
            with open(path) as f:
                assert "<!DOCTYPE html>" in f.read()


# ── Custom checks tests ──────────────────────────────────────────────────────

class TestCustomChecks:
    def test_custom_check_function(self, valid_config, base_dir):
        def _custom_check(config, bd):
            return CheckResult(name="custom", passed=True, message="Custom OK")

        launcher = ExperimentLauncher(
            valid_config, base_dir=base_dir, checks=[_custom_check]
        )
        report = launcher.preflight(include_dry_run=False)
        assert report.n_checks_total == 1
        assert report.checks[0].name == "custom"

    def test_empty_checks_list(self, valid_config, base_dir):
        launcher = ExperimentLauncher(
            valid_config, base_dir=base_dir, checks=[]
        )
        report = launcher.preflight(include_dry_run=False)
        assert report.n_checks_total == 0
