"""Tests for compass/master_dashboard.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from compass.master_dashboard import (
    generate_html,
    load_git_log,
    load_hedge_configs,
    load_model_files,
    load_pruned_benchmark,
    load_pruned_validation,
    load_registry,
    load_stress_results,
)

ROOT = Path(__file__).resolve().parent.parent


# ── Data loaders ─────────────────────────────────────────────────────────


class TestDataLoaders:

    def test_load_registry_returns_dict(self):
        data = load_registry()
        assert isinstance(data, dict)

    def test_registry_has_experiments(self):
        data = load_registry()
        assert "experiments" in data
        assert len(data["experiments"]) > 0

    def test_load_stress_results(self):
        data = load_stress_results()
        assert isinstance(data, dict)

    def test_stress_has_hedge_impact(self):
        data = load_stress_results()
        assert "crisis_hedge_impact" in data
        assert len(data["crisis_hedge_impact"]) >= 2

    def test_load_pruned_validation(self):
        data = load_pruned_validation()
        assert isinstance(data, dict)
        assert data.get("feature_count") == 21

    def test_pruned_validation_has_walk_forward(self):
        data = load_pruned_validation()
        wf = data.get("xgboost_walk_forward", {})
        assert "aggregate" in wf
        assert "folds" in wf

    def test_load_pruned_benchmark(self):
        data = load_pruned_benchmark()
        assert isinstance(data, dict)

    def test_load_model_files(self):
        models = load_model_files()
        assert isinstance(models, list)
        assert len(models) >= 2
        for m in models:
            assert "name" in m
            assert "size_kb" in m
            assert "modified" in m

    def test_load_git_log(self):
        log = load_git_log(5)
        assert isinstance(log, list)
        assert len(log) > 0
        assert "hash" in log[0]
        assert "message" in log[0]
        assert "date" in log[0]

    def test_load_hedge_configs(self):
        cfgs = load_hedge_configs()
        assert "EXP-400" in cfgs
        assert "EXP-401" in cfgs
        assert cfgs["EXP-400"]["vix_floor"] == 12.0
        assert cfgs["EXP-401"]["vix_floor"] == 14.0


# ── HTML generation ──────────────────────────────────────────────────────


class TestGenerateHtml:

    @pytest.fixture(scope="class")
    def html(self):
        return generate_html()

    def test_is_valid_html(self, html):
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_title(self, html):
        assert "COMPASS Master Dashboard" in html

    def test_contains_experiment_registry(self, html):
        assert "Experiment Registry" in html
        assert "EXP-400" in html
        assert "EXP-401" in html

    def test_contains_stress_test_section(self, html):
        assert "Stress Test" in html
        assert "Hedge Impact" in html

    def test_contains_hedge_config_section(self, html):
        assert "Hedge Configuration" in html
        assert "VIX Floor" in html

    def test_contains_model_diagnostics(self, html):
        assert "Model Diagnostics" in html
        assert "Production Models" in html

    def test_contains_pruned_benchmark(self, html):
        assert "Pruned vs Full" in html

    def test_contains_walk_forward_folds(self, html):
        assert "Walk-Forward Folds" in html
        assert "Fold 0" in html

    def test_contains_feature_set(self, html):
        assert "Feature Set" in html
        assert "credit_to_width" in html
        assert "contracts_log" in html  # in removed list

    def test_contains_phase_milestones(self, html):
        assert "Phase Milestones" in html

    def test_contains_pass_fail_badges(self, html):
        assert "PASS" in html

    def test_contains_status_badges(self, html):
        assert "Paper Trading" in html

    def test_model_files_listed(self, html):
        assert ".joblib" in html

    def test_hedge_config_values(self, html):
        assert "12.0" in html or "12" in html  # EXP-400 floor
        assert "14.0" in html or "14" in html  # EXP-401 floor

    def test_writes_to_file(self):
        html = generate_html()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dashboard.html"
            path.write_text(html)
            assert path.exists()
            assert path.stat().st_size > 5000
