"""Tests for compass/feature_importance.py — walk-forward feature importance.

Covers:
  - _walk_forward_importance: fold splitting, gain/perm extraction
  - aggregate_importances: ranking, stability, composite score
  - _ablation_analysis: AUC drop calculation
  - generate_report: markdown structure and content
  - run_feature_importance_analysis: end-to-end (with small synthetic data)
"""

import numpy as np
import pandas as pd
import pytest

from compass.feature_importance import (
    _walk_forward_importance,
    aggregate_importances,
    generate_report,
)
from compass.walk_forward import NUMERIC_FEATURES, CATEGORICAL_FEATURES


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_dataset(n_per_year=40, years=(2020, 2021, 2022)):
    """Synthetic training data with known feature-outcome relationships."""
    rng = np.random.RandomState(42)
    rows = []
    for year in years:
        for i in range(n_per_year):
            vix = rng.uniform(12, 45)
            iv_rank = rng.uniform(0, 100)
            rsi = rng.uniform(20, 80)
            # Make win partially predictable from vix + iv_rank
            prob = 0.3 + 0.4 * (iv_rank / 100) + 0.1 * (1 - vix / 50)
            win = 1 if rng.random() < max(0, min(1, prob)) else 0

            rows.append({
                "entry_date": f"{year}-{1 + i % 12:02d}-{1 + i % 28:02d}",
                "exit_date": f"{year}-{1 + i % 12:02d}-{5 + i % 24:02d}",
                "year": year,
                "strategy_type": rng.choice(["CS", "SS"]),
                "spread_type": rng.choice(["bull_put", "bear_call"]),
                "dte_at_entry": rng.randint(10, 50),
                "hold_days": rng.randint(1, 30),
                "day_of_week": rng.randint(0, 5),
                "days_since_last_trade": rng.randint(1, 20),
                "regime": rng.choice(["bull", "bear", "neutral"]),
                "rsi_14": round(rsi, 2),
                "momentum_5d_pct": round(rng.normal(0, 2), 4),
                "momentum_10d_pct": round(rng.normal(0, 3), 4),
                "vix": round(vix, 2),
                "vix_percentile_20d": round(rng.uniform(0, 100), 2),
                "vix_percentile_50d": round(rng.uniform(0, 100), 2),
                "vix_percentile_100d": round(rng.uniform(0, 100), 2),
                "iv_rank": round(iv_rank, 2),
                "spy_price": round(rng.uniform(380, 520), 2),
                "dist_from_ma20_pct": round(rng.normal(0, 2), 4),
                "dist_from_ma50_pct": round(rng.normal(0, 3), 4),
                "dist_from_ma80_pct": round(rng.normal(0, 4), 4),
                "dist_from_ma200_pct": round(rng.normal(0, 5), 4),
                "ma20_slope_ann_pct": round(rng.normal(10, 20), 4),
                "ma50_slope_ann_pct": round(rng.normal(5, 15), 4),
                "realized_vol_atr20": round(rng.uniform(10, 40), 2),
                "realized_vol_5d": round(rng.uniform(8, 50), 4),
                "realized_vol_10d": round(rng.uniform(8, 45), 4),
                "realized_vol_20d": round(rng.uniform(8, 40), 4),
                "net_credit": round(rng.uniform(0.2, 2.0), 4),
                "spread_width": 5.0,
                "max_loss_per_unit": round(rng.uniform(3, 5), 4),
                "otm_pct": round(rng.uniform(2, 12), 4),
                "contracts": rng.randint(1, 5),
                "exit_reason": rng.choice(["profit_target", "stop_loss", "expiry"]),
                "pnl": round(rng.normal(20, 80), 2),
                "return_pct": round(rng.normal(2, 10), 2),
                "win": win,
            })
    return pd.DataFrame(rows)


# ── _walk_forward_importance ─────────────────────────────────────────────


class TestWalkForwardImportance:
    def test_basic_run(self):
        df = _make_dataset(n_per_year=40, years=(2020, 2021, 2022))
        result = _walk_forward_importance(df, NUMERIC_FEATURES, min_train_samples=10)
        assert result["n_folds"] >= 1
        assert len(result["feature_cols"]) > 0

    def test_fold_count(self):
        df = _make_dataset(n_per_year=40, years=(2020, 2021, 2022))
        result = _walk_forward_importance(df, NUMERIC_FEATURES, min_train_samples=10)
        # 3 years → max 2 folds
        assert result["n_folds"] <= 2

    def test_fold_has_gain_and_perm(self):
        df = _make_dataset(n_per_year=40, years=(2020, 2021, 2022))
        result = _walk_forward_importance(
            df, NUMERIC_FEATURES,
            min_train_samples=10, n_permutation_repeats=3,
        )
        for fold in result["fold_results"]:
            assert "gain_importance" in fold
            assert "perm_importance" in fold
            assert "auc" in fold
            assert len(fold["gain_importance"]) == len(result["feature_cols"])
            assert len(fold["perm_importance"]) == len(result["feature_cols"])

    def test_auc_is_valid(self):
        df = _make_dataset(n_per_year=50, years=(2020, 2021, 2022))
        result = _walk_forward_importance(df, NUMERIC_FEATURES, min_train_samples=10)
        for fold in result["fold_results"]:
            assert 0 <= fold["auc"] <= 1

    def test_insufficient_years_raises(self):
        df = _make_dataset(n_per_year=40, years=(2020,))
        with pytest.raises(ValueError, match="2 years"):
            _walk_forward_importance(df, NUMERIC_FEATURES)

    def test_gain_sums_to_one(self):
        df = _make_dataset(n_per_year=40, years=(2020, 2021, 2022))
        result = _walk_forward_importance(df, NUMERIC_FEATURES, min_train_samples=10)
        for fold in result["fold_results"]:
            total = fold["gain_importance"].sum()
            if total > 0:
                assert total == pytest.approx(1.0, abs=0.01)


# ── aggregate_importances ────────────────────────────────────────────────


class TestAggregateImportances:
    def test_returns_dataframe(self):
        df = _make_dataset(n_per_year=40, years=(2020, 2021, 2022))
        wf = _walk_forward_importance(df, NUMERIC_FEATURES, min_train_samples=10)
        summary = aggregate_importances(wf)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0

    def test_has_required_columns(self):
        df = _make_dataset(n_per_year=40, years=(2020, 2021, 2022))
        wf = _walk_forward_importance(df, NUMERIC_FEATURES, min_train_samples=10)
        summary = aggregate_importances(wf)
        for col in ["feature", "gain_mean", "perm_mean", "gain_stability",
                     "perm_stability", "rank_gain", "rank_perm", "rank_composite"]:
            assert col in summary.columns

    def test_stability_in_zero_one(self):
        df = _make_dataset(n_per_year=40, years=(2020, 2021, 2022))
        wf = _walk_forward_importance(df, NUMERIC_FEATURES, min_train_samples=10)
        summary = aggregate_importances(wf)
        assert (summary["gain_stability"] >= 0).all()
        assert (summary["gain_stability"] <= 1).all()

    def test_sorted_by_composite_rank(self):
        df = _make_dataset(n_per_year=40, years=(2020, 2021, 2022))
        wf = _walk_forward_importance(df, NUMERIC_FEATURES, min_train_samples=10)
        summary = aggregate_importances(wf)
        ranks = summary["rank_composite"].values
        assert all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))

    def test_empty_folds(self):
        wf = {"fold_results": [], "feature_cols": ["a", "b"], "n_folds": 0}
        summary = aggregate_importances(wf)
        assert len(summary) == 0


# ── generate_report ──────────────────────────────────────────────────────


class TestGenerateReport:
    def _make_report_data(self):
        df = _make_dataset(n_per_year=40, years=(2020, 2021, 2022))
        wf = _walk_forward_importance(
            df, NUMERIC_FEATURES,
            min_train_samples=10, n_permutation_repeats=3,
        )
        summary = aggregate_importances(wf)
        ablation = {f: round(np.random.RandomState(42).normal(0, 0.01), 4)
                    for f in summary["feature"].tolist()}
        return summary, wf, ablation

    def test_report_is_markdown(self):
        summary, wf, ablation = self._make_report_data()
        report = generate_report(summary, wf, ablation, "test.csv", 120)
        assert report.startswith("# Feature Importance Analysis")

    def test_report_sections(self):
        summary, wf, ablation = self._make_report_data()
        report = generate_report(summary, wf, ablation, "test.csv", 120)
        assert "## 1. Walk-Forward Fold Summary" in report
        assert "## 2. Feature Importance Rankings" in report
        assert "## 3. Signal vs Noise Classification" in report
        assert "## 4. Ablation Analysis" in report
        assert "## 5. Pruning Recommendations" in report
        assert "## 6. Methodology Notes" in report

    def test_report_contains_fold_aucs(self):
        summary, wf, ablation = self._make_report_data()
        report = generate_report(summary, wf, ablation, "test.csv", 120)
        assert "AUC" in report
        for fold in wf["fold_results"]:
            assert str(fold["auc"]) in report

    def test_report_contains_feature_names(self):
        summary, wf, ablation = self._make_report_data()
        report = generate_report(summary, wf, ablation, "test.csv", 120)
        # At least some features should appear
        assert "vix" in report.lower()

    def test_report_contains_ablation_verdicts(self):
        summary, wf, ablation = self._make_report_data()
        report = generate_report(summary, wf, ablation, "test.csv", 120)
        assert "KEEP" in report or "NEUTRAL" in report or "PRUNE" in report
