"""
Tests for compass/experiment_pipeline.py — automated batch experiment runner.

Coverage:
  - ExperimentConfig dataclass creation, validation, defaults
  - ExperimentResult serialisation
  - extract_metrics() from backtest results
  - ExperimentPipeline init, validation, duplicate detection
  - run_single() with mocked backtester (success and failure)
  - run_all() with multiple configs
  - compare() side-by-side metrics DataFrame
  - to_json() and to_html() output
  - Walk-forward integration
  - Edge cases (empty results, missing fields, etc.)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from compass.experiment_pipeline import (
    ExperimentConfig,
    ExperimentPipeline,
    ExperimentResult,
    extract_metrics,
    run_walk_forward,
    _compute_cagr_from_yearly,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backtest_results(
    sharpe: float = 1.5,
    return_pct: float = 25.0,
    max_dd: float = -0.15,
    total_trades: int = 100,
    winning_trades: int = 65,
    total_pnl: float = 25_000.0,
) -> Dict[str, Any]:
    """Build a realistic PortfolioBacktester result dict for testing."""
    return {
        "timestamp": "2025-01-01T00:00:00",
        "config": {"strategies": [], "tickers": ["SPY"]},
        "combined": {
            "starting_capital": 100_000,
            "ending_capital": 125_000,
            "return_pct": return_pct,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
            "total_pnl": total_pnl,
            "avg_win": 500.0,
            "avg_loss": -300.0,
            "profit_factor": 1.8,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "equity_curve": [],
            "monthly_pnl": {},
        },
        "per_strategy": {},
        "trades": [],
        "yearly": {
            "2020": {"return_pct": 30.0, "max_drawdown": -0.10, "sharpe_ratio": 1.8, "total_trades": 20},
            "2021": {"return_pct": 20.0, "max_drawdown": -0.12, "sharpe_ratio": 1.2, "total_trades": 25},
            "2022": {"return_pct": -5.0, "max_drawdown": -0.20, "sharpe_ratio": -0.3, "total_trades": 22},
            "2023": {"return_pct": 15.0, "max_drawdown": -0.08, "sharpe_ratio": 1.5, "total_trades": 18},
            "2024": {"return_pct": 22.0, "max_drawdown": -0.15, "sharpe_ratio": 1.4, "total_trades": 15},
        },
    }


def _make_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        experiment_id="EXP-001",
        name="Test Experiment",
        strategy_class="CreditSpreadStrategy",
        ticker="SPY",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _dummy_strategy_factory(config: ExperimentConfig) -> Tuple[str, Any]:
    """Strategy factory that returns a MagicMock strategy."""
    return (config.name, MagicMock())


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


class TestExperimentConfig:

    def test_basic_creation(self):
        cfg = _make_config()
        assert cfg.experiment_id == "EXP-001"
        assert cfg.name == "Test Experiment"
        assert cfg.strategy_class == "CreditSpreadStrategy"
        assert cfg.ticker == "SPY"

    def test_default_values(self):
        cfg = _make_config()
        assert cfg.config_overrides == {}
        assert cfg.description == ""
        assert cfg.start_date == "2020-01-01"
        assert cfg.end_date == "2025-12-31"
        assert cfg.starting_capital == 100_000.0

    def test_config_overrides(self):
        cfg = _make_config(config_overrides={"spread_width": 10, "dte": 45})
        assert cfg.config_overrides["spread_width"] == 10
        assert cfg.config_overrides["dte"] == 45

    def test_empty_experiment_id_raises(self):
        with pytest.raises(ValueError, match="experiment_id"):
            _make_config(experiment_id="")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            _make_config(name="")

    def test_empty_strategy_class_raises(self):
        with pytest.raises(ValueError, match="strategy_class"):
            _make_config(strategy_class="")

    def test_empty_ticker_raises(self):
        with pytest.raises(ValueError, match="ticker"):
            _make_config(ticker="")

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError, match="starting_capital"):
            _make_config(starting_capital=-1000)

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError, match="starting_capital"):
            _make_config(starting_capital=0)

    def test_custom_dates(self):
        cfg = _make_config(start_date="2022-06-01", end_date="2024-12-31")
        assert cfg.start_date == "2022-06-01"
        assert cfg.end_date == "2024-12-31"


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentResult:

    def test_to_dict_has_required_keys(self):
        cfg = _make_config()
        result = ExperimentResult(
            config=cfg,
            metrics={"sharpe": 1.5},
            timestamp="2025-01-01T00:00:00",
            status="completed",
        )
        d = result.to_dict()
        assert "config" in d
        assert "metrics" in d
        assert "timestamp" in d
        assert "status" in d

    def test_to_dict_excludes_backtest_results(self):
        """Raw backtest results are large; to_dict should not include them."""
        result = ExperimentResult(
            config=_make_config(),
            backtest_results=_mock_backtest_results(),
            metrics={"sharpe": 1.5},
            timestamp="2025-01-01T00:00:00",
        )
        d = result.to_dict()
        assert "backtest_results" not in d

    def test_to_dict_includes_walk_forward(self):
        result = ExperimentResult(
            config=_make_config(),
            walk_forward={"consistent": True, "pass_rate": 0.67},
            metrics={},
            timestamp="2025-01-01T00:00:00",
        )
        d = result.to_dict()
        assert d["walk_forward"]["consistent"] is True

    def test_failed_result(self):
        result = ExperimentResult(
            config=_make_config(),
            status="failed",
            error="Strategy not found",
            timestamp="2025-01-01T00:00:00",
        )
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Strategy not found"


# ---------------------------------------------------------------------------
# extract_metrics
# ---------------------------------------------------------------------------


class TestExtractMetrics:

    def test_extracts_all_keys(self):
        results = _mock_backtest_results()
        m = extract_metrics(results)
        expected_keys = {
            "sharpe", "cagr_pct", "max_dd_pct", "win_rate", "total_trades",
            "profit_factor", "total_pnl", "avg_win", "avg_loss", "return_pct",
        }
        assert expected_keys == set(m.keys())

    def test_sharpe_value(self):
        m = extract_metrics(_mock_backtest_results(sharpe=2.1))
        assert m["sharpe"] == 2.1

    def test_win_rate_calculation(self):
        m = extract_metrics(_mock_backtest_results(total_trades=200, winning_trades=140))
        assert m["win_rate"] == 70.0

    def test_win_rate_zero_trades(self):
        m = extract_metrics(_mock_backtest_results(total_trades=0, winning_trades=0))
        assert m["win_rate"] == 0.0

    def test_max_dd_as_percentage(self):
        m = extract_metrics(_mock_backtest_results(max_dd=-0.25))
        assert m["max_dd_pct"] == -25.0

    def test_empty_combined(self):
        m = extract_metrics({"combined": {}})
        assert m["sharpe"] == 0.0
        assert m["total_trades"] == 0

    def test_missing_combined(self):
        m = extract_metrics({})
        assert m["sharpe"] == 0.0


# ---------------------------------------------------------------------------
# _compute_cagr_from_yearly
# ---------------------------------------------------------------------------


class TestComputeCAGR:

    def test_positive_returns(self):
        yearly = {
            "2020": {"return_pct": 20.0},
            "2021": {"return_pct": 10.0},
        }
        cagr = _compute_cagr_from_yearly(yearly, {})
        # (1.2 * 1.1)^(1/2) - 1 = (1.32)^0.5 - 1 ≈ 14.89%
        assert 14.0 < cagr < 16.0

    def test_mixed_returns(self):
        yearly = {
            "2020": {"return_pct": 30.0},
            "2021": {"return_pct": -10.0},
        }
        cagr = _compute_cagr_from_yearly(yearly, {})
        # (1.3 * 0.9)^0.5 - 1 = 1.17^0.5 - 1 ≈ 8.17%
        assert 7.0 < cagr < 10.0

    def test_empty_yearly_uses_combined(self):
        cagr = _compute_cagr_from_yearly({}, {"return_pct": 15.0})
        assert cagr == 15.0

    def test_no_data(self):
        cagr = _compute_cagr_from_yearly({}, {})
        assert cagr == 0.0


# ---------------------------------------------------------------------------
# ExperimentPipeline — init
# ---------------------------------------------------------------------------


class TestPipelineInit:

    def test_basic_init(self):
        configs = [_make_config()]
        pipeline = ExperimentPipeline(configs, strategy_factory=_dummy_strategy_factory)
        assert len(pipeline.configs) == 1
        assert pipeline.results == []

    def test_empty_configs_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ExperimentPipeline([])

    def test_duplicate_ids_raises(self):
        c1 = _make_config(experiment_id="EXP-001")
        c2 = _make_config(experiment_id="EXP-001", name="Duplicate")
        with pytest.raises(ValueError, match="Duplicate"):
            ExperimentPipeline([c1, c2])

    def test_multiple_configs(self):
        c1 = _make_config(experiment_id="EXP-001")
        c2 = _make_config(experiment_id="EXP-002", name="Second")
        pipeline = ExperimentPipeline([c1, c2], strategy_factory=_dummy_strategy_factory)
        assert len(pipeline.configs) == 2

    def test_output_dir_stored(self):
        pipeline = ExperimentPipeline(
            [_make_config()], output_dir="/tmp/test_output",
            strategy_factory=_dummy_strategy_factory,
        )
        assert pipeline.output_dir == Path("/tmp/test_output")


# ---------------------------------------------------------------------------
# ExperimentPipeline — run_single (mocked backtester)
# ---------------------------------------------------------------------------


class TestRunSingle:

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_successful_run(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        result = pipeline.run_single("EXP-001")

        assert result.status == "completed"
        assert result.metrics["sharpe"] == 1.5
        assert result.metrics["total_trades"] == 100
        assert result.timestamp != ""

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_failed_run(self, mock_bt):
        mock_bt.side_effect = RuntimeError("Data not found")
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        result = pipeline.run_single("EXP-001")

        assert result.status == "failed"
        assert "Data not found" in result.error
        assert result.metrics == {}

    def test_unknown_id_raises(self):
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        with pytest.raises(KeyError):
            pipeline.run_single("EXP-999")

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_result_stored_by_id(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_single("EXP-001")

        assert pipeline.get_result("EXP-001") is not None
        assert pipeline.get_result("EXP-001").status == "completed"

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_backtest_results_preserved(self, mock_bt):
        bt_results = _mock_backtest_results()
        mock_bt.return_value = bt_results
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        result = pipeline.run_single("EXP-001")

        assert result.backtest_results is bt_results


# ---------------------------------------------------------------------------
# ExperimentPipeline — run_all
# ---------------------------------------------------------------------------


class TestRunAll:

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_runs_all_configs(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        c1 = _make_config(experiment_id="EXP-001")
        c2 = _make_config(experiment_id="EXP-002", name="Second")
        pipeline = ExperimentPipeline([c1, c2], strategy_factory=_dummy_strategy_factory)

        results = pipeline.run_all()
        assert len(results) == 2
        assert all(r.status == "completed" for r in results)

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_completed_and_failed_properties(self, mock_bt):
        mock_bt.side_effect = [
            _mock_backtest_results(),
            RuntimeError("boom"),
        ]
        c1 = _make_config(experiment_id="EXP-001")
        c2 = _make_config(experiment_id="EXP-002", name="Fails")
        pipeline = ExperimentPipeline([c1, c2], strategy_factory=_dummy_strategy_factory)
        pipeline.run_all()

        assert len(pipeline.completed) == 1
        assert len(pipeline.failed) == 1
        assert pipeline.completed[0].config.experiment_id == "EXP-001"
        assert pipeline.failed[0].config.experiment_id == "EXP-002"

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_run_all_clears_previous_results(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()
        pipeline.run_all()
        assert len(pipeline.results) == 1


# ---------------------------------------------------------------------------
# ExperimentPipeline — compare
# ---------------------------------------------------------------------------


class TestCompare:

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_compare_returns_dataframe(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        c1 = _make_config(experiment_id="EXP-001")
        c2 = _make_config(experiment_id="EXP-002", name="Second")
        pipeline = ExperimentPipeline([c1, c2], strategy_factory=_dummy_strategy_factory)
        pipeline.run_all()

        df = pipeline.compare()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_compare_columns(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()

        df = pipeline.compare()
        assert "experiment_id" in df.columns
        assert "sharpe" in df.columns
        assert "cagr_pct" in df.columns
        assert "max_dd_pct" in df.columns
        assert "win_rate" in df.columns
        assert "wf_consistent" in df.columns

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_compare_sorted_by_sharpe_desc(self, mock_bt):
        mock_bt.side_effect = [
            _mock_backtest_results(sharpe=0.5),
            _mock_backtest_results(sharpe=2.0),
        ]
        c1 = _make_config(experiment_id="EXP-001", name="Low Sharpe")
        c2 = _make_config(experiment_id="EXP-002", name="High Sharpe")
        pipeline = ExperimentPipeline([c1, c2], strategy_factory=_dummy_strategy_factory)
        pipeline.run_all()

        df = pipeline.compare()
        assert df.iloc[0]["sharpe"] == 2.0
        assert df.iloc[1]["sharpe"] == 0.5

    def test_compare_empty_results(self):
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        df = pipeline.compare()
        assert df.empty

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_compare_excludes_failed(self, mock_bt):
        mock_bt.side_effect = [
            _mock_backtest_results(),
            RuntimeError("fail"),
        ]
        c1 = _make_config(experiment_id="EXP-001")
        c2 = _make_config(experiment_id="EXP-002", name="Fails")
        pipeline = ExperimentPipeline([c1, c2], strategy_factory=_dummy_strategy_factory)
        pipeline.run_all()

        df = pipeline.compare()
        assert len(df) == 1
        assert df.iloc[0]["experiment_id"] == "EXP-001"


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


class TestToJson:

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_to_json_string(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()

        json_str = pipeline.to_json()
        data = json.loads(json_str)
        assert "pipeline_run" in data
        assert "experiments" in data
        assert data["pipeline_run"]["n_completed"] == 1

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_to_json_file(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            result_path = pipeline.to_json(path)
            assert os.path.exists(result_path)
            data = json.loads(Path(result_path).read_text())
            assert len(data["experiments"]) == 1

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_to_json_includes_comparison(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()

        data = json.loads(pipeline.to_json())
        assert "comparison" in data
        assert len(data["comparison"]) == 1


# ---------------------------------------------------------------------------
# to_html
# ---------------------------------------------------------------------------


class TestToHtml:

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_to_html_string(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()

        html = pipeline.to_html()
        assert "<!DOCTYPE html>" in html
        assert "EXP-001" in html
        assert "Experiment Pipeline Report" in html

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_to_html_file(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            result_path = pipeline.to_html(path)
            assert os.path.exists(result_path)
            content = Path(result_path).read_text()
            assert "Side-by-Side Comparison" in content

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_html_shows_failed_experiments(self, mock_bt):
        mock_bt.side_effect = RuntimeError("kaboom")
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()

        html = pipeline.to_html()
        assert "Failed Experiments" in html
        assert "kaboom" in html

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_html_contains_metric_values(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results(sharpe=1.5)
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()

        html = pipeline.to_html()
        assert "1.500" in html  # Sharpe formatted to 3dp


# ---------------------------------------------------------------------------
# Walk-forward integration
# ---------------------------------------------------------------------------


class TestWalkForward:

    def test_insufficient_years_returns_none(self):
        results = {"yearly": {"2020": {}, "2021": {}}}
        assert run_walk_forward(results) is None

    def test_no_yearly_returns_none(self):
        assert run_walk_forward({}) is None

    @patch("compass.experiment_pipeline.run_walk_forward")
    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_walk_forward_result_attached(self, mock_bt, mock_wf):
        mock_bt.return_value = _mock_backtest_results()
        mock_wf.return_value = {"consistent": True, "pass_rate": 0.67}

        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()

        # The result's walk_forward field should be populated
        result = pipeline.get_result("EXP-001")
        assert result is not None


# ---------------------------------------------------------------------------
# get_result
# ---------------------------------------------------------------------------


class TestGetResult:

    def test_returns_none_before_run(self):
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        assert pipeline.get_result("EXP-001") is None

    @patch("compass.experiment_pipeline.ExperimentPipeline._run_backtest")
    def test_returns_result_after_run(self, mock_bt):
        mock_bt.return_value = _mock_backtest_results()
        pipeline = ExperimentPipeline(
            [_make_config()], strategy_factory=_dummy_strategy_factory,
        )
        pipeline.run_all()
        assert pipeline.get_result("EXP-001") is not None
        assert pipeline.get_result("NONEXISTENT") is None
