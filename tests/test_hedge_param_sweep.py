"""Tests for compass/hedge_param_sweep.py — parameter grid search."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from compass.crisis_hedge import CrisisHedgeConfig, CrisisHedgeController
from compass.hedge_param_sweep import (
    CSV_FIELDS,
    SweepResult,
    build_grid,
    evaluate_config,
    load_daily_returns,
    load_daily_returns_hedged,
    run_sweep,
    write_csv,
    write_summary_md,
)

ROOT = Path(__file__).resolve().parent.parent
EXP400_CSV = ROOT / "compass" / "training_data_exp400.csv"
EXP401_CSV = ROOT / "compass" / "training_data_exp401.csv"


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_returns() -> np.ndarray:
    rng = np.random.RandomState(42)
    return rng.normal(0.0004, 0.01, 252)


@pytest.fixture
def default_config() -> CrisisHedgeConfig:
    return CrisisHedgeConfig(log_decisions=False)


@pytest.fixture
def sample_results() -> List[SweepResult]:
    return [
        SweepResult("EXP-400", 20.0, 50.0, 3.5, 0.25, -21.5, 1.5, 18.2, True),
        SweepResult("EXP-400", 16.0, 42.0, 2.5, 0.10, -28.3, 1.2, 15.1, True),
        SweepResult("EXP-400", 12.0, 35.0, 1.5, 0.05, -35.1, 0.8, 10.5, False),
        SweepResult("EXP-401", 14.0, 38.0, 2.0, 0.10, -29.4, 0.9, 12.3, True),
        SweepResult("EXP-401", 20.0, 50.0, 3.5, 0.25, -42.4, 0.6, 8.1, False),
    ]


# ── SweepResult dataclass ────────────────────────────────────────────────


class TestSweepResult:

    def test_creation(self):
        r = SweepResult("EXP-400", 20.0, 50.0, 3.5, 0.25, -21.5, 1.5, 18.2, True)
        assert r.experiment == "EXP-400"
        assert r.mc_p5_dd == -21.5
        assert r.passes_30pct is True

    def test_failing_result(self):
        r = SweepResult("EXP-401", 20.0, 50.0, 3.5, 0.25, -42.0, 0.5, 8.0, False)
        assert r.passes_30pct is False
        assert abs(r.mc_p5_dd) > 30.0

    def test_fields_match_csv(self):
        r = SweepResult("EXP-400", 20.0, 50.0, 3.5, 0.25, -21.5, 1.5, 18.2, True)
        # All CSV fields should be accessible as attributes
        for field in CSV_FIELDS:
            assert hasattr(r, field), f"SweepResult missing field: {field}"


# ── build_grid ────────────────────────────────────────────────────────────


class TestBuildGrid:

    def test_default_grid_not_empty(self):
        configs = build_grid()
        assert len(configs) > 0

    def test_filters_invalid_floor_ge_ceiling(self):
        configs = build_grid(
            vix_floors=[30.0, 40.0],
            vix_ceilings=[35.0],
            base_stops=[2.0],
            hv_scales=[0.10],
        )
        # floor=40 >= ceiling=35 should be filtered
        assert all(c.vix_scale_floor < c.vix_scale_ceiling for c in configs)
        assert len(configs) == 1  # only floor=30, ceiling=35

    def test_all_configs_have_log_decisions_false(self):
        configs = build_grid(
            vix_floors=[18.0], vix_ceilings=[42.0],
            base_stops=[2.5], hv_scales=[0.10],
        )
        for c in configs:
            assert c.log_decisions is False

    def test_stop_ceiling_derived_from_scale_thresholds(self):
        configs = build_grid(
            vix_floors=[20.0], vix_ceilings=[50.0],
            base_stops=[3.0], hv_scales=[0.10],
        )
        c = configs[0]
        expected = 20.0 + 0.6 * (50.0 - 20.0)  # 38.0
        assert c.vix_stop_ceiling == expected

    def test_min_stop_derived_from_base(self):
        configs = build_grid(
            vix_floors=[20.0], vix_ceilings=[50.0],
            base_stops=[2.0], hv_scales=[0.10],
        )
        c = configs[0]
        assert c.min_stop_multiplier == max(1.0, 2.0 - 1.5)  # 1.0

    def test_min_stop_clamped_at_one(self):
        configs = build_grid(
            vix_floors=[20.0], vix_ceilings=[50.0],
            base_stops=[1.5], hv_scales=[0.10],
        )
        c = configs[0]
        assert c.min_stop_multiplier == 1.0  # max(1.0, 1.5 - 1.5) = max(1.0, 0) = 1.0

    def test_grid_size_is_product_of_valid_combos(self):
        floors = [15.0, 20.0]
        ceilings = [40.0, 50.0]
        stops = [2.0, 3.0]
        hv = [0.10]
        configs = build_grid(floors, ceilings, stops, hv)
        # All floor < ceiling combos: 2 floors × 2 ceilings × 2 stops × 1 hv = 8
        assert len(configs) == 8

    def test_custom_single_point_grid(self):
        configs = build_grid(
            vix_floors=[18.0], vix_ceilings=[45.0],
            base_stops=[3.0], hv_scales=[0.20],
        )
        assert len(configs) == 1
        c = configs[0]
        assert c.vix_scale_floor == 18.0
        assert c.vix_scale_ceiling == 45.0
        assert c.base_stop_multiplier == 3.0
        assert c.high_vol_regime_scale == 0.20


# ── load_daily_returns ────────────────────────────────────────────────────


class TestLoadDailyReturns:

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="training data not available")
    def test_load_exp400(self):
        returns = load_daily_returns(EXP400_CSV, 100_000)
        assert isinstance(returns, pd.Series)
        assert len(returns) > 100

    @pytest.mark.skipif(not EXP401_CSV.exists(), reason="training data not available")
    def test_load_exp401(self):
        returns = load_daily_returns(EXP401_CSV, 100_000)
        assert isinstance(returns, pd.Series)
        assert len(returns) > 100


# ── load_daily_returns_hedged ─────────────────────────────────────────────


class TestLoadDailyReturnsHedged:

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="training data not available")
    def test_hedged_returns_differ_from_unhedged(self):
        cfg = CrisisHedgeConfig(vix_scale_floor=15.0, log_decisions=False)
        ctrl = CrisisHedgeController(cfg)
        unhedged = load_daily_returns(EXP400_CSV, 100_000)
        hedged = load_daily_returns_hedged(EXP400_CSV, 100_000, ctrl)
        # Hedged should differ (some trades are VIX-scaled)
        assert not np.allclose(unhedged.values, hedged.values)

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="training data not available")
    def test_noop_hedge_matches_unhedged(self):
        """With all thresholds disabled, hedged ≈ unhedged."""
        cfg = CrisisHedgeConfig(
            vix_scale_floor=100.0,
            vix_scale_ceiling=200.0,
            vix_stop_floor=100.0,
            vix_stop_ceiling=200.0,
            high_vol_regime_scale=1.0,   # don't cap high_vol regime
            crash_regime_scale=1.0,      # don't gate crash regime
            use_vix_term_structure=False,
            log_decisions=False,
        )
        ctrl = CrisisHedgeController(cfg)
        unhedged = load_daily_returns(EXP400_CSV, 100_000)
        hedged = load_daily_returns_hedged(EXP400_CSV, 100_000, ctrl)
        assert np.allclose(unhedged.values, hedged.values, atol=1e-10)


# ── evaluate_config ───────────────────────────────────────────────────────


class TestEvaluateConfig:

    def test_returns_sweep_result(self, sample_returns, default_config):
        result = evaluate_config(
            sample_returns, "TEST", default_config, n_simulations=100,
        )
        assert isinstance(result, SweepResult)
        assert result.experiment == "TEST"

    def test_mc_p5_dd_is_negative(self, sample_returns, default_config):
        result = evaluate_config(
            sample_returns, "TEST", default_config, n_simulations=100,
        )
        assert result.mc_p5_dd < 0

    def test_sharpe_is_finite(self, sample_returns, default_config):
        result = evaluate_config(
            sample_returns, "TEST", default_config, n_simulations=100,
        )
        assert np.isfinite(result.hedged_sharpe)

    def test_annual_return_reasonable(self, sample_returns, default_config):
        result = evaluate_config(
            sample_returns, "TEST", default_config, n_simulations=100,
        )
        assert -100 < result.annual_return_pct < 200

    def test_passes_flag_consistent(self, sample_returns, default_config):
        result = evaluate_config(
            sample_returns, "TEST", default_config, n_simulations=100,
        )
        assert result.passes_30pct == (abs(result.mc_p5_dd) <= 30.0)

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="training data not available")
    def test_with_real_exp400_data(self):
        cfg = CrisisHedgeConfig(log_decisions=False)
        ctrl = CrisisHedgeController(cfg)
        hedged = load_daily_returns_hedged(EXP400_CSV, 100_000, ctrl)
        result = evaluate_config(hedged.values, "EXP-400", cfg, n_simulations=200)
        assert result.mc_p5_dd < 0
        assert result.hedged_sharpe > 0


# ── run_sweep ─────────────────────────────────────────────────────────────


class TestRunSweep:

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="training data not available")
    def test_small_sweep(self):
        configs = build_grid(
            vix_floors=[18.0], vix_ceilings=[45.0],
            base_stops=[3.0], hv_scales=[0.20],
        )
        results = run_sweep(EXP400_CSV, "EXP-400", configs, n_simulations=100)
        assert len(results) == 1
        assert results[0].experiment == "EXP-400"

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="training data not available")
    def test_results_sorted_by_p5_dd_desc(self):
        configs = build_grid(
            vix_floors=[16.0, 20.0], vix_ceilings=[42.0],
            base_stops=[2.5], hv_scales=[0.10],
        )
        results = run_sweep(EXP400_CSV, "EXP-400", configs, n_simulations=100)
        for i in range(1, len(results)):
            assert results[i].mc_p5_dd <= results[i - 1].mc_p5_dd


# ── write_csv ─────────────────────────────────────────────────────────────


class TestWriteCsv:

    def test_creates_file(self, sample_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            write_csv(sample_results, path)
            assert path.exists()

    def test_correct_row_count(self, sample_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            write_csv(sample_results, path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == len(sample_results)

    def test_header_matches_fields(self, sample_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            write_csv(sample_results, path)
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
            assert header == CSV_FIELDS

    def test_creates_parent_dirs(self, sample_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "results.csv"
            write_csv(sample_results, path)
            assert path.exists()

    def test_empty_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.csv"
            write_csv([], path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1  # header only


# ── write_summary_md ──────────────────────────────────────────────────────


class TestWriteSummaryMd:

    def test_creates_file(self, sample_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.md"
            write_summary_md(sample_results, path)
            assert path.exists()

    def test_contains_experiment_headers(self, sample_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.md"
            write_summary_md(sample_results, path)
            content = path.read_text()
            assert "## EXP-400" in content
            assert "## EXP-401" in content

    def test_contains_top_10_table(self, sample_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.md"
            write_summary_md(sample_results, path)
            content = path.read_text()
            assert "### Top 10 Configs" in content
            assert "VIX Floor" in content

    def test_contains_pass_fail_counts(self, sample_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.md"
            write_summary_md(sample_results, path)
            content = path.read_text()
            assert "Passing" in content

    def test_contains_sensitivity_analysis(self, sample_results):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.md"
            write_summary_md(sample_results, path)
            content = path.read_text()
            assert "Parameter Sensitivity" in content

    def test_empty_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.md"
            write_summary_md([], path)
            content = path.read_text()
            assert "# Hedge Parameter Sweep Results" in content
            assert "Total combos evaluated: 0" in content

    def test_single_experiment(self):
        results = [
            SweepResult("EXP-400", 20.0, 50.0, 3.5, 0.25, -21.5, 1.5, 18.2, True),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "single.md"
            write_summary_md(results, path)
            content = path.read_text()
            assert "## EXP-400" in content
            assert "EXP-401" not in content
