"""Tests for compass/model_diagnostics.py — HTML diagnostics dashboard.

Covers:
  - _collect_fold_data: fold splitting, metric extraction, gain importance
  - _fig_to_base64: produces valid base64 PNG
  - _render_*: each chart renderer returns non-empty base64
  - _build_html: HTML structure and content
  - generate_diagnostics: end-to-end on synthetic data
"""

import base64
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from compass.model_diagnostics import (
    _build_html,
    _collect_fold_data,
    _fig_to_base64,
    _render_calibration_curve,
    _render_confusion_matrices,
    _render_feature_importance,
    _render_prediction_distributions,
    _render_roc_curves,
    generate_diagnostics,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_dataset(n_per_year=40, years=(2020, 2021, 2022)):
    """Synthetic training data with partially predictable outcome."""
    rng = np.random.RandomState(42)
    rows = []
    for year in years:
        for i in range(n_per_year):
            vix = rng.uniform(12, 45)
            iv_rank = rng.uniform(0, 100)
            prob = 0.3 + 0.4 * (iv_rank / 100)
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
                "rsi_14": round(rng.uniform(20, 80), 2),
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
                "exit_reason": "profit_target",
                "pnl": round(rng.normal(20, 80), 2),
                "return_pct": round(rng.normal(2, 10), 2),
                "win": win,
            })
    return pd.DataFrame(rows)


def _make_folds():
    """Collect folds from synthetic data."""
    df = _make_dataset(n_per_year=50, years=(2020, 2021, 2022))
    return _collect_fold_data(df, min_train_samples=10)


# ── _collect_fold_data ───────────────────────────────────────────────────


class TestCollectFoldData:
    def test_returns_list_of_folds(self):
        folds = _make_folds()
        assert isinstance(folds, list)
        assert len(folds) >= 1

    def test_fold_has_required_keys(self):
        folds = _make_folds()
        required = {
            "fold", "train_years", "test_year", "n_train", "n_test",
            "y_test", "y_proba", "y_pred", "auc", "accuracy", "precision",
            "recall", "brier", "fpr", "tpr", "optimal_threshold", "gain_importance",
        }
        for f in folds:
            assert required.issubset(f.keys())

    def test_auc_in_valid_range(self):
        folds = _make_folds()
        for f in folds:
            assert 0 <= f["auc"] <= 1

    def test_y_proba_in_zero_one(self):
        folds = _make_folds()
        for f in folds:
            assert f["y_proba"].min() >= 0
            assert f["y_proba"].max() <= 1

    def test_optimal_threshold_in_range(self):
        folds = _make_folds()
        for f in folds:
            assert 0 <= f["optimal_threshold"] <= 1

    def test_gain_importance_is_dict(self):
        folds = _make_folds()
        for f in folds:
            assert isinstance(f["gain_importance"], dict)
            if f["gain_importance"]:
                total = sum(f["gain_importance"].values())
                assert total == pytest.approx(1.0, abs=0.02)

    def test_insufficient_years_raises(self):
        df = _make_dataset(n_per_year=40, years=(2020,))
        with pytest.raises(ValueError, match="2 years"):
            _collect_fold_data(df)


# ── _fig_to_base64 ──────────────────────────────────────────────────────


class TestFigToBase64:
    def test_produces_valid_base64(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        b64 = _fig_to_base64(fig)
        plt.close(fig)

        assert isinstance(b64, str)
        assert len(b64) > 100
        # Verify it's valid base64 by decoding
        raw = base64.b64decode(b64)
        assert raw[:4] == b"\x89PNG"


# ── Chart renderers ──────────────────────────────────────────────────────


class TestChartRenderers:
    @pytest.fixture(scope="class")
    def folds(self):
        return _make_folds()

    def test_roc_curves(self, folds):
        b64 = _render_roc_curves(folds)
        assert len(b64) > 1000
        assert base64.b64decode(b64)[:4] == b"\x89PNG"

    def test_calibration_curve(self, folds):
        b64 = _render_calibration_curve(folds)
        assert len(b64) > 1000

    def test_confusion_matrices(self, folds):
        b64 = _render_confusion_matrices(folds)
        assert len(b64) > 1000

    def test_feature_importance(self, folds):
        b64 = _render_feature_importance(folds)
        assert len(b64) > 1000

    def test_prediction_distributions(self, folds):
        b64 = _render_prediction_distributions(folds)
        assert len(b64) > 1000


# ── _build_html ──────────────────────────────────────────────────────────


class TestBuildHTML:
    def test_valid_html_structure(self):
        folds = _make_folds()
        roc = _render_roc_curves(folds)
        cal = _render_calibration_curve(folds)
        cm = _render_confusion_matrices(folds)
        fi = _render_feature_importance(folds)
        dist = _render_prediction_distributions(folds)
        html = _build_html(folds, roc, cal, cm, fi, dist, "test.csv", 120)

        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_kpi_values(self):
        folds = _make_folds()
        roc = _render_roc_curves(folds)
        cal = _render_calibration_curve(folds)
        cm = _render_confusion_matrices(folds)
        fi = _render_feature_importance(folds)
        dist = _render_prediction_distributions(folds)
        html = _build_html(folds, roc, cal, cm, fi, dist, "test.csv", 120)

        assert "Mean AUC" in html
        assert "Mean Accuracy" in html
        assert "Mean Precision" in html
        assert "Brier Score" in html

    def test_contains_embedded_images(self):
        folds = _make_folds()
        roc = _render_roc_curves(folds)
        cal = _render_calibration_curve(folds)
        cm = _render_confusion_matrices(folds)
        fi = _render_feature_importance(folds)
        dist = _render_prediction_distributions(folds)
        html = _build_html(folds, roc, cal, cm, fi, dist, "test.csv", 120)

        assert html.count("data:image/png;base64,") == 5

    def test_contains_all_sections(self):
        folds = _make_folds()
        roc = _render_roc_curves(folds)
        cal = _render_calibration_curve(folds)
        cm = _render_confusion_matrices(folds)
        fi = _render_feature_importance(folds)
        dist = _render_prediction_distributions(folds)
        html = _build_html(folds, roc, cal, cm, fi, dist, "test.csv", 120)

        assert "Per-Fold Summary" in html
        assert "ROC Curves" in html
        assert "Calibration" in html
        assert "Confusion Matrices" in html
        assert "Feature Importance" in html
        assert "Prediction Distributions" in html


# ── generate_diagnostics (end-to-end) ────────────────────────────────────


class TestGenerateDiagnostics:
    def test_end_to_end(self, tmp_path):
        df = _make_dataset(n_per_year=50, years=(2020, 2021, 2022))
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        out_path = tmp_path / "report.html"
        result = generate_diagnostics(str(csv_path), str(out_path), min_train_samples=10)

        assert os.path.exists(result)
        content = open(result).read()
        assert "<!DOCTYPE html>" in content
        assert "data:image/png;base64," in content
        assert len(content) > 10000  # should be substantial

    def test_output_is_self_contained(self, tmp_path):
        df = _make_dataset(n_per_year=50, years=(2020, 2021, 2022))
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        out_path = tmp_path / "report.html"
        generate_diagnostics(str(csv_path), str(out_path), min_train_samples=10)

        content = open(out_path).read()
        # No external stylesheet/script/image references
        assert "http://" not in content
        assert "https://" not in content
        assert '<link rel="stylesheet"' not in content
        assert "<script src=" not in content
