"""Tests for compass/trade_outcome_predictor.py — trade outcome predictor.

Covers:
  - Dataclass construction
  - Model fitting and feature importance
  - Single prediction with confidence intervals
  - Similar-trade lookup
  - Risk/reward scoring
  - Calibration analysis
  - Batch prediction
  - from_csv constructor
  - HTML report generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.trade_outcome_predictor import (
    CalibrationBucket,
    FeatureImportance,
    FitResult,
    Prediction,
    SimilarTrade,
    TradeOutcomePredictor,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_trades(n=200, seed=42):
    """Generate synthetic historical trades with realistic features."""
    rng = np.random.RandomState(seed)
    regimes = rng.choice(["bull", "bear", "high_vol", "neutral"], n)
    vix = rng.uniform(12, 40, n)
    dte = rng.randint(7, 50, n).astype(float)
    credit = rng.uniform(0.3, 3.0, n)
    width = rng.choice([2.5, 5.0, 10.0], n)
    iv_pct = rng.uniform(10, 95, n)
    delta_short = rng.uniform(-0.4, -0.05, n)
    rsi = rng.uniform(20, 80, n)
    momentum = rng.normal(0, 2, n)
    dow = rng.randint(0, 5, n).astype(float)
    hour = rng.randint(9, 16, n).astype(float)

    # P&L: loosely driven by features + noise
    pnl = (
        credit * 100
        - (vix - 20) * 3
        + (iv_pct - 50) * 0.5
        + rng.normal(0, 80, n)
    )
    return_pct = pnl / (width * 100)

    return pd.DataFrame({
        "vix": vix, "iv_percentile": iv_pct, "dte_at_entry": dte,
        "net_credit": credit, "spread_width": width,
        "delta_short": delta_short, "rsi": rsi, "momentum": momentum,
        "day_of_week": dow, "hour_of_day": hour,
        "regime": regimes, "pnl": pnl, "return_pct": return_pct,
    })


def _make_predictor(n=200, seed=42, **kwargs):
    return TradeOutcomePredictor(_make_trades(n=n, seed=seed), **kwargs)


def _sample_features():
    return {
        "vix": 22.0, "iv_percentile": 55.0, "dte_at_entry": 30.0,
        "net_credit": 1.5, "spread_width": 5.0, "delta_short": -0.15,
        "rsi": 45.0, "momentum": 0.5, "day_of_week": 2.0, "hour_of_day": 11.0,
    }


# ── Dataclass tests ──────────────────────────────────────────────────────


class TestDataclasses:
    def test_prediction_fields(self):
        p = Prediction(
            expected_pnl=50.0, ci_lower=-100, ci_upper=200,
            win_probability=0.65, risk_reward_score=72.0,
            confidence=0.7, regime="bull",
        )
        assert p.win_probability == pytest.approx(0.65)

    def test_similar_trade_fields(self):
        s = SimilarTrade(
            index=5, distance=1.2, pnl=80.0, return_pct=0.08,
            regime="bull", features={"vix": 20.0},
        )
        assert s.distance == pytest.approx(1.2)

    def test_feature_importance_fields(self):
        fi = FeatureImportance(feature="vix", importance=0.25, rank=1)
        assert fi.rank == 1

    def test_calibration_bucket_fields(self):
        cb = CalibrationBucket(
            predicted_mean=50.0, actual_mean=55.0, n_trades=20,
            bucket_label="Q1",
        )
        assert cb.n_trades == 20

    def test_fit_result_fields(self):
        fr = FitResult(
            n_train=100, n_features=10, train_r2=0.85,
            cv_mae=25.0, feature_importances=[],
        )
        assert fr.train_r2 == pytest.approx(0.85)


# ── Model fitting tests ─────────────────────────────────────────────────


class TestFitting:
    def test_fit_returns_result(self):
        pred = _make_predictor()
        result = pred.fit()
        assert isinstance(result, FitResult)
        assert result.n_train > 0

    def test_fit_populates_importances(self):
        pred = _make_predictor()
        pred.fit()
        assert len(pred.fit_result.feature_importances) > 0

    def test_importances_sum_to_one(self):
        pred = _make_predictor()
        pred.fit()
        total = sum(fi.importance for fi in pred.fit_result.feature_importances)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_importances_ranked(self):
        pred = _make_predictor()
        pred.fit()
        ranks = [fi.rank for fi in pred.fit_result.feature_importances]
        assert ranks == sorted(ranks)

    def test_train_r2_positive(self):
        pred = _make_predictor()
        pred.fit()
        assert pred.fit_result.train_r2 > 0

    def test_features_detected(self):
        pred = _make_predictor()
        assert len(pred.features) > 0

    def test_no_features_raises(self):
        df = pd.DataFrame({"pnl": [1, 2, 3], "regime": ["bull"] * 3})
        with pytest.raises(ValueError, match="No usable features"):
            TradeOutcomePredictor(df, features=["nonexistent_col"])


# ── Prediction tests ─────────────────────────────────────────────────────


class TestPrediction:
    def test_predict_returns_prediction(self):
        pred = _make_predictor()
        pred.fit()
        result = pred.predict(_sample_features())
        assert isinstance(result, Prediction)

    def test_ci_lower_lt_upper(self):
        pred = _make_predictor()
        pred.fit()
        result = pred.predict(_sample_features())
        assert result.ci_lower <= result.ci_upper

    def test_win_probability_range(self):
        pred = _make_predictor()
        pred.fit()
        result = pred.predict(_sample_features())
        assert 0 <= result.win_probability <= 1

    def test_risk_reward_score_range(self):
        pred = _make_predictor()
        pred.fit()
        result = pred.predict(_sample_features())
        assert 0 <= result.risk_reward_score <= 100

    def test_confidence_range(self):
        pred = _make_predictor()
        pred.fit()
        result = pred.predict(_sample_features())
        assert 0 <= result.confidence <= 1

    def test_predict_auto_fits(self):
        pred = _make_predictor()
        assert pred.fit_result is None
        pred.predict(_sample_features())
        assert pred.fit_result is not None

    def test_predictions_accumulated(self):
        pred = _make_predictor()
        pred.fit()
        pred.predict(_sample_features())
        pred.predict(_sample_features())
        assert len(pred.predictions) == 2


# ── Similar trade tests ──────────────────────────────────────────────────


class TestSimilarTrades:
    def test_returns_n_neighbors(self):
        pred = _make_predictor(n_neighbors=5)
        pred.fit()
        similar = pred.find_similar_trades(_sample_features())
        assert len(similar) == 5

    def test_custom_n(self):
        pred = _make_predictor(n_neighbors=10)
        pred.fit()
        similar = pred.find_similar_trades(_sample_features(), n=3)
        assert len(similar) == 3

    def test_sorted_by_distance(self):
        pred = _make_predictor()
        pred.fit()
        similar = pred.find_similar_trades(_sample_features())
        dists = [s.distance for s in similar]
        assert dists == sorted(dists)

    def test_similar_trades_have_features(self):
        pred = _make_predictor()
        pred.fit()
        similar = pred.find_similar_trades(_sample_features())
        for s in similar:
            assert len(s.features) > 0

    def test_similar_trades_have_pnl(self):
        pred = _make_predictor()
        pred.fit()
        similar = pred.find_similar_trades(_sample_features())
        for s in similar:
            assert isinstance(s.pnl, float)


# ── Risk/reward score tests ──────────────────────────────────────────────


class TestRiskReward:
    def test_positive_expected_scores_higher(self):
        s1 = TradeOutcomePredictor._risk_reward_score(100, -50, 200, 0.7)
        s2 = TradeOutcomePredictor._risk_reward_score(-100, -200, 50, 0.3)
        assert s1 > s2

    def test_high_win_prob_scores_higher(self):
        s1 = TradeOutcomePredictor._risk_reward_score(50, -50, 150, 0.9)
        s2 = TradeOutcomePredictor._risk_reward_score(50, -50, 150, 0.2)
        assert s1 > s2

    def test_score_in_range(self):
        s = TradeOutcomePredictor._risk_reward_score(50, -100, 200, 0.6)
        assert 0 <= s <= 100


# ── Calibration tests ────────────────────────────────────────────────────


class TestCalibration:
    def test_calibration_populated_after_fit(self):
        pred = _make_predictor()
        pred.fit()
        assert len(pred.calibration) > 0

    def test_calibration_buckets_have_trades(self):
        pred = _make_predictor()
        pred.fit()
        for b in pred.calibration:
            assert b.n_trades > 0

    def test_calibration_bucket_labels(self):
        pred = _make_predictor()
        pred.fit()
        labels = [b.bucket_label for b in pred.calibration]
        assert labels == ["Q1", "Q2", "Q3", "Q4", "Q5"]


# ── Batch prediction tests ──────────────────────────────────────────────


class TestBatchPrediction:
    def test_batch_returns_list(self):
        pred = _make_predictor()
        pred.fit()
        df = _make_trades(n=5, seed=99)
        results = pred.predict_batch(df)
        assert len(results) == 5

    def test_batch_elements_are_predictions(self):
        pred = _make_predictor()
        pred.fit()
        df = _make_trades(n=3, seed=99)
        results = pred.predict_batch(df)
        for r in results:
            assert isinstance(r, Prediction)


# ── from_csv tests ───────────────────────────────────────────────────────


class TestFromCSV:
    def test_from_csv_constructs(self, tmp_path):
        df = _make_trades()
        csv = tmp_path / "trades.csv"
        df.to_csv(csv, index=False)
        pred = TradeOutcomePredictor.from_csv(str(csv))
        assert len(pred.trades) == len(df)

    def test_from_csv_fits(self, tmp_path):
        df = _make_trades()
        csv = tmp_path / "trades.csv"
        df.to_csv(csv, index=False)
        pred = TradeOutcomePredictor.from_csv(str(csv))
        pred.fit()
        assert pred.fit_result.n_train > 0


# ── Report generation tests ──────────────────────────────────────────────


class TestReport:
    def test_generates_html(self, tmp_path):
        pred = _make_predictor()
        pred.fit()
        pred.predict(_sample_features())
        path = pred.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Trade Outcome" in content

    def test_report_contains_sections(self, tmp_path):
        pred = _make_predictor()
        pred.fit()
        path = pred.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "Feature Importance" in content
        assert "Calibration" in content
        assert "P&L Distribution" in content

    def test_report_embeds_charts(self, tmp_path):
        pred = _make_predictor()
        pred.fit()
        path = pred.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "data:image/png;base64," in content

    def test_report_auto_fits(self, tmp_path):
        pred = _make_predictor()
        assert pred.fit_result is None
        pred.generate_report(str(tmp_path / "report.html"))
        assert pred.fit_result is not None

    def test_report_at_default_path(self):
        pred = _make_predictor()
        pred.fit()
        path = pred.generate_report()
        assert "trade_outcome.html" in path
        assert open(path).read().startswith("<!DOCTYPE html>")
