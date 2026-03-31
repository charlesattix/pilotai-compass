"""Tests for compass/benchmark_pruned_features.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from compass.benchmark_pruned_features import (
    ALREADY_PRUNED_BY_PIPELINE,
    HARMFUL_PIPELINE_FEATURES,
    NOISE_PIPELINE_FEATURES,
    PRUNE_LIST,
    build_pipeline_df,
    generate_markdown,
    run_xgboost,
)
from compass.feature_pipeline import FeaturePipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "compass" / "training_data_combined.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    pytest.skip("training_data_combined.csv not available")


@pytest.fixture
def mock_results() -> Dict:
    """A realistic aggregate results dict."""
    return {
        "aggregate": {
            "auc_mean": 0.82, "auc_std": 0.05,
            "accuracy_mean": 0.72, "accuracy_std": 0.04,
            "precision_mean": 0.75, "precision_std": 0.05,
            "recall_mean": 0.68, "recall_std": 0.06,
            "brier_score_mean": 0.21, "brier_score_std": 0.03,
            "signal_sharpe_mean": 1.5, "signal_sharpe_std": 0.4,
            "total_oos_samples": 368,
        },
        "folds": [
            {"fold": i, "auc": 0.80 + i * 0.01, "test_period": f"202{i+1}",
             "n_train": 60 + i * 40, "n_test": 70}
            for i in range(5)
        ],
        "n_folds": 5,
    }


# ── Prune list consistency ───────────────────────────────────────────────


class TestPruneList:

    def test_prune_list_is_union(self):
        assert set(PRUNE_LIST) == set(HARMFUL_PIPELINE_FEATURES + NOISE_PIPELINE_FEATURES)

    def test_no_duplicates(self):
        assert len(PRUNE_LIST) == len(set(PRUNE_LIST))

    def test_harmful_count(self):
        assert len(HARMFUL_PIPELINE_FEATURES) == 1  # contracts_log

    def test_noise_count(self):
        assert len(NOISE_PIPELINE_FEATURES) == 9

    def test_total_prune_count(self):
        assert len(PRUNE_LIST) == 10

    def test_already_pruned_count(self):
        assert len(ALREADY_PRUNED_BY_PIPELINE) == 5

    def test_contracts_log_in_harmful(self):
        assert "contracts_log" in HARMFUL_PIPELINE_FEATURES

    def test_regime_dummies_in_noise(self):
        regime_dummies = [f for f in NOISE_PIPELINE_FEATURES if f.startswith("regime_")]
        assert len(regime_dummies) == 5

    def test_no_overlap_with_already_pruned(self):
        assert not set(PRUNE_LIST) & set(ALREADY_PRUNED_BY_PIPELINE)


# ── build_pipeline_df ────────────────────────────────────────────────────


class TestBuildPipelineDf:

    def test_full_returns_31_features(self, raw_df):
        df_out, features = build_pipeline_df(raw_df, prune=None)
        assert len(features) == 31

    def test_pruned_returns_21_features(self, raw_df):
        df_out, features = build_pipeline_df(raw_df, prune=PRUNE_LIST)
        assert len(features) == 21

    def test_pruned_excludes_prune_list(self, raw_df):
        _, features = build_pipeline_df(raw_df, prune=PRUNE_LIST)
        for f in PRUNE_LIST:
            assert f not in features, f"{f} should have been pruned"

    def test_pruned_keeps_signal_features(self, raw_df):
        _, features = build_pipeline_df(raw_df, prune=PRUNE_LIST)
        signal = ["credit_to_width", "loss_to_width", "iv_rank", "rsi_14",
                  "strategy_type_CS", "spread_type_bull_put"]
        for f in signal:
            assert f in features, f"Signal feature {f} should be kept"

    def test_metadata_columns_preserved(self, raw_df):
        df_out, _ = build_pipeline_df(raw_df, prune=PRUNE_LIST)
        assert "entry_date" in df_out.columns
        assert "win" in df_out.columns
        assert "return_pct" in df_out.columns

    def test_row_count_unchanged(self, raw_df):
        df_out, _ = build_pipeline_df(raw_df, prune=PRUNE_LIST)
        assert len(df_out) == len(raw_df)

    def test_no_nans_in_features(self, raw_df):
        df_out, features = build_pipeline_df(raw_df, prune=PRUNE_LIST)
        assert df_out[features].isna().sum().sum() == 0

    def test_empty_prune_matches_full(self, raw_df):
        _, full = build_pipeline_df(raw_df, prune=None)
        _, empty_prune = build_pipeline_df(raw_df, prune=[])
        assert full == empty_prune

    def test_nonexistent_prune_ignored(self, raw_df):
        _, features = build_pipeline_df(raw_df, prune=["does_not_exist"])
        assert len(features) == 31  # nothing removed


# ── XGBoost benchmark ────────────────────────────────────────────────────


class TestRunXgboost:

    @pytest.mark.skipif(not DATA_PATH.exists(), reason="training data not available")
    def test_returns_results_dict(self):
        df = pd.read_csv(DATA_PATH)
        df_out, features = build_pipeline_df(df, prune=None)
        results = run_xgboost(df_out, features, "XGBoost Test")
        assert "aggregate" in results
        assert "folds" in results
        assert results["n_features"] == 31

    @pytest.mark.skipif(not DATA_PATH.exists(), reason="training data not available")
    def test_pruned_vs_full_runs(self):
        df = pd.read_csv(DATA_PATH)
        df_full, full_feat = build_pipeline_df(df, prune=None)
        df_pruned, pruned_feat = build_pipeline_df(df, prune=PRUNE_LIST)

        full_res = run_xgboost(df_full, full_feat, "Full")
        pruned_res = run_xgboost(df_pruned, pruned_feat, "Pruned")

        assert full_res["n_features"] == 31
        assert pruned_res["n_features"] == 21
        # Both should produce valid AUC
        assert 0.5 < full_res["aggregate"]["auc_mean"] < 1.0
        assert 0.5 < pruned_res["aggregate"]["auc_mean"] < 1.0


# ── Markdown generation ──────────────────────────────────────────────────


class TestGenerateMarkdown:

    def test_contains_header(self, mock_results):
        md = generate_markdown(
            mock_results, mock_results, None, None,
            [f"f{i}" for i in range(31)],
            [f"f{i}" for i in range(21)],
        )
        assert "# Pruned Features Benchmark" in md

    def test_contains_comparison_table(self, mock_results):
        md = generate_markdown(
            mock_results, mock_results, None, None,
            [f"f{i}" for i in range(31)],
            [f"f{i}" for i in range(21)],
        )
        assert "| AUC |" in md
        assert "| Accuracy |" in md

    def test_contains_verdict(self, mock_results):
        md = generate_markdown(
            mock_results, mock_results, None, None,
            [f"f{i}" for i in range(31)],
            [f"f{i}" for i in range(21)],
        )
        assert "## Verdict" in md

    def test_contains_prune_list(self, mock_results):
        md = generate_markdown(
            mock_results, mock_results, None, None,
            [f"f{i}" for i in range(31)],
            [f"f{i}" for i in range(21)],
        )
        assert "contracts_log" in md
        assert "regime_bear" in md

    def test_ensemble_section_when_provided(self, mock_results):
        md = generate_markdown(
            mock_results, mock_results, mock_results, mock_results,
            [f"f{i}" for i in range(31)],
            [f"f{i}" for i in range(21)],
        )
        assert "## Ensemble: Full vs Pruned" in md

    def test_no_ensemble_section_when_none(self, mock_results):
        md = generate_markdown(
            mock_results, mock_results, None, None,
            [f"f{i}" for i in range(31)],
            [f"f{i}" for i in range(21)],
        )
        assert "Ensemble" not in md.split("## Verdict")[0].split("## XGBoost")[1]

    def test_per_fold_table(self, mock_results):
        md = generate_markdown(
            mock_results, mock_results, None, None,
            [f"f{i}" for i in range(31)],
            [f"f{i}" for i in range(21)],
        )
        assert "### Per-Fold AUC" in md

    def test_already_pruned_documented(self, mock_results):
        md = generate_markdown(
            mock_results, mock_results, None, None,
            [f"f{i}" for i in range(31)],
            [f"f{i}" for i in range(21)],
        )
        assert "vix_percentile_20d" in md
        assert "already absent from pipeline" in md.lower()
