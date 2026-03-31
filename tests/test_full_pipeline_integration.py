"""
Full-pipeline integration tests for the COMPASS system.

Tests the entire chain from feature engineering through ML prediction
through position sizing through risk management, verifying that all
modules interoperate correctly with both happy paths and failure modes.

Coverage:
  A. Feature → SignalModel → Sizing → RiskGate pipeline
  B. EnsembleSignalModel → ShadowEnsemble → RegimeModelRouter
  C. CrisisHedge → AdvancedSizing → PortfolioOptimizer
  D. OnlineRetrain trigger → WalkForward validation
  E. Failure modes: missing data, NaN features, model not trained, etc.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Module imports ───────────────────────────────────────────────────────

from compass.advanced_sizing import AdvancedPositionSizer, SizingConfig, SizingResult
from compass.crisis_hedge import (
    CrisisHedgeConfig,
    CrisisHedgeController,
    get_hedge_config,
)
from compass.ensemble_signal_model import EnsembleSignalModel
from compass.feature_pipeline import FeaturePipeline
from compass.features import PRUNED_FEATURES, PRUNED_REMOVED
from compass.online_retrain import ModelRetrainer, RetrainResult, RetrainTrigger
from compass.regime import Regime, RegimeClassifier
from compass.retrain_scheduler import RetrainScheduler
from compass.signal_model import SignalModel
from compass.sizing import PositionSizer, calculate_dynamic_risk, get_contract_size

ROOT = Path(__file__).resolve().parent.parent
TRAINING_CSV = ROOT / "compass" / "training_data_combined.csv"


# ── Shared fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def raw_training_df() -> pd.DataFrame:
    if not TRAINING_CSV.exists():
        pytest.skip("training_data_combined.csv not available")
    return pd.read_csv(TRAINING_CSV)


@pytest.fixture(scope="module")
def pipeline_features(raw_training_df) -> pd.DataFrame:
    return FeaturePipeline(pruned=True).transform(raw_training_df)


@pytest.fixture(scope="module")
def labels(raw_training_df) -> np.ndarray:
    return raw_training_df["win"].values.astype(int)


def _make_synthetic_features(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic DataFrame with PRUNED_FEATURES columns."""
    rng = np.random.RandomState(seed)
    data = {f: rng.randn(n) for f in PRUNED_FEATURES}
    # Binary columns should be 0/1
    for c in ["strategy_type_CS", "spread_type_bull_put"]:
        data[c] = rng.choice([0, 1], n)
    return pd.DataFrame(data)


def _make_synthetic_labels(n: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice([0, 1], n, p=[0.35, 0.65])


# ══════════════════════════════════════════════════════════════════════════
# A. Feature → SignalModel → Sizing → RiskGate pipeline
# ══════════════════════════════════════════════════════════════════════════


class TestFeatureToSignalPipeline:
    """FeaturePipeline → SignalModel → sizing decisions."""

    def test_pipeline_produces_21_features(self, pipeline_features):
        assert pipeline_features.shape[1] == 21
        assert list(pipeline_features.columns) == PRUNED_FEATURES

    def test_pipeline_no_nans(self, pipeline_features):
        assert pipeline_features.isna().sum().sum() == 0

    def test_signal_model_trains_on_pipeline_output(self, pipeline_features, labels):
        model = SignalModel(model_dir=tempfile.mkdtemp())
        model.train(pipeline_features, labels, calibrate=True, save_model=False)
        assert model.trained
        assert model.feature_names == PRUNED_FEATURES

    def test_signal_model_predicts_after_train(self, pipeline_features, labels):
        model = SignalModel(model_dir=tempfile.mkdtemp())
        model.train(pipeline_features, labels, save_model=False)
        probs = model.predict_batch(pipeline_features)
        assert len(probs) == len(labels)
        assert all(0 <= p <= 1 for p in probs)

    def test_predictions_feed_into_sizing(self, pipeline_features, labels):
        model = SignalModel(model_dir=tempfile.mkdtemp())
        model.train(pipeline_features, labels, save_model=False)
        probs = model.predict_batch(pipeline_features[:5])

        sizer = PositionSizer()
        for prob in probs:
            result = sizer.calculate_position_size(
                win_probability=float(prob),
                expected_return=0.30,
                expected_loss=-1.00,
                ml_confidence=abs(float(prob) - 0.5) * 2,
            )
            assert 0 <= result["recommended_size"] <= 1.0

    def test_sizing_feeds_into_contract_count(self, pipeline_features, labels):
        model = SignalModel(model_dir=tempfile.mkdtemp())
        model.train(pipeline_features, labels, save_model=False)
        prob = float(model.predict_batch(pipeline_features[:1])[0])

        risk = calculate_dynamic_risk(
            account_value=100_000,
            iv_rank=50.0,
            current_portfolio_risk=5_000,
            ml_confidence_multiplier=abs(prob - 0.5) * 2,
        )
        contracts = get_contract_size(risk, spread_width=5.0, credit_received=0.65)
        assert isinstance(contracts, int)
        assert 0 <= contracts <= 5

    def test_risk_gate_evaluates_alert(self):
        """RiskGate.check accepts an alert and account state."""
        from alerts.alert_schema import Alert, AlertType, Direction, Leg
        from compass.risk_gate import RiskGate

        rg = RiskGate()
        leg = Leg(strike=540.0, option_type="put", action="sell", expiration="2026-06-20")
        alert = Alert(
            type=AlertType.credit_spread,
            ticker="SPY",
            direction=Direction.bullish,
            legs=[leg],
            entry_price=0.65,
            stop_loss=1.95,
            profit_target=0.33,
            risk_pct=0.05,
        )
        account = {
            "account_value": 100_000,
            "open_positions": [],
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
        }
        approved, reason = rg.check(alert, account)
        assert isinstance(approved, bool)
        assert isinstance(reason, str)


# ══════════════════════════════════════════════════════════════════════════
# B. EnsembleSignalModel → ShadowEnsemble → RegimeModelRouter
# ══════════════════════════════════════════════════════════════════════════


class TestEnsembleShadowRouter:
    """Ensemble model with shadow logging and regime-based routing."""

    def test_ensemble_trains_on_pruned_features(self, pipeline_features, labels):
        model = EnsembleSignalModel(model_dir=tempfile.mkdtemp())
        model.train(pipeline_features, labels, save_model=False)
        assert model.trained
        assert len(model.feature_names) == 21

    def test_ensemble_predicts_probabilities(self, pipeline_features, labels):
        model = EnsembleSignalModel(model_dir=tempfile.mkdtemp())
        model.train(pipeline_features, labels, save_model=False)
        probs = model.predict_batch(pipeline_features[:10])
        assert len(probs) == 10
        assert all(0 <= p <= 1 for p in probs)

    def test_shadow_ensemble_wraps_primary(self, pipeline_features, labels):
        from compass.shadow_ensemble import ShadowEnsemble

        primary = EnsembleSignalModel(model_dir=tempfile.mkdtemp())
        primary.train(pipeline_features, labels, save_model=False)

        shadow = EnsembleSignalModel(model_dir=tempfile.mkdtemp())
        shadow.train(pipeline_features, labels, save_model=False)

        with tempfile.TemporaryDirectory() as d:
            se = ShadowEnsemble(primary, shadow, log_path=Path(d) / "shadow.csv")
            probs = se.predict_batch(pipeline_features[:5])
            assert len(probs) == 5
            # predict_batch may or may not increment stats depending on impl;
            # verify it produces valid probabilities
            assert all(0 <= p <= 1 for p in probs)

    def test_regime_model_router_multiplier(self):
        from ml.regime_model_router import RegimeModelRouter

        router = RegimeModelRouter(config={"use_signal_model": False})
        for regime in ["bull", "bear", "high_vol", "low_vol", "crash", None]:
            mult = router.get_multiplier(regime)
            assert 0.0 <= mult <= 2.0

    def test_router_with_model_config(self):
        from ml.regime_model_router import RegimeModelRouter

        router = RegimeModelRouter(config={
            "use_signal_model": False,
            "shadow_ensemble": False,
            "min_mult": 0.10,
            "max_mult": 1.50,
            "crash_mult": 0.00,
        })
        assert router.get_multiplier("crash") == 0.0
        assert router.get_multiplier("bull") > 0

    def test_ensemble_to_shadow_to_sizing(self, pipeline_features, labels):
        """Full chain: ensemble → shadow → extract prob → size position."""
        from compass.shadow_ensemble import ShadowEnsemble

        model = EnsembleSignalModel(model_dir=tempfile.mkdtemp())
        model.train(pipeline_features, labels, save_model=False)

        with tempfile.TemporaryDirectory() as d:
            se = ShadowEnsemble(model, model, log_path=Path(d) / "s.csv")
            prob = float(se.predict_batch(pipeline_features[:1])[0])

        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=prob, win_return=0.30, loss_return=1.0, regime="bull",
        )
        assert isinstance(result, SizingResult)
        assert result.position_fraction >= 0


# ══════════════════════════════════════════════════════════════════════════
# C. CrisisHedge → AdvancedSizing → PortfolioOptimizer
# ══════════════════════════════════════════════════════════════════════════


class TestHedgeSizingOptimizer:
    """Crisis hedge interacts with sizing and portfolio optimisation."""

    def test_hedge_config_per_experiment(self):
        cfg400 = get_hedge_config("EXP-400")
        cfg401 = get_hedge_config("EXP-401")
        assert cfg400.vix_scale_floor == 12.0
        assert cfg401.vix_scale_floor == 14.0

    def test_hedge_controller_scales_position(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        # Below floor
        assert ctrl.position_scale_factor(vix=10.0) == 1.0
        # Above ceiling
        assert ctrl.position_scale_factor(vix=40.0) == 0.0

    def test_hedge_scale_feeds_into_advanced_sizing(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        scale = ctrl.position_scale_factor(vix=25.0)

        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.0,
            regime="bull", current_dd_pct=10.0,
        )
        adjusted = result.position_fraction * scale
        assert 0 <= adjusted <= result.position_fraction

    def test_sizing_respects_regime_fractions(self):
        sizer = AdvancedPositionSizer()
        bull = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.0, regime="bull")
        crash = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.0, regime="crash")
        assert crash.position_fraction <= bull.position_fraction

    def test_drawdown_kills_sizing(self):
        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.0,
            regime="bull", current_dd_pct=28.0,
        )
        assert result.position_fraction == 0.0
        assert result.dd_scale == 0.0

    def test_portfolio_optimizer_produces_weights(self):
        from compass.portfolio_optimizer import PortfolioOptimizer

        rng = np.random.RandomState(42)
        returns = {
            "EXP-400": rng.normal(0.001, 0.01, 252),
            "EXP-401": rng.normal(0.0005, 0.015, 252),
        }
        opt = PortfolioOptimizer(returns=returns)
        result = opt.optimize(method="max_sharpe")
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        assert all(w >= 0 for w in result.weights.values())

    def test_optimizer_with_regime(self):
        from compass.portfolio_optimizer import PortfolioOptimizer

        rng = np.random.RandomState(42)
        returns = {
            "A": rng.normal(0.001, 0.01, 252),
            "B": rng.normal(0.0005, 0.015, 252),
        }
        opt = PortfolioOptimizer(returns=returns)
        bull = opt.optimize(regime="bull")
        bear = opt.optimize(regime="bear")
        # Both should produce valid weights
        assert abs(sum(bull.weights.values()) - 1.0) < 0.01
        assert abs(sum(bear.weights.values()) - 1.0) < 0.01


# ══════════════════════════════════════════════════════════════════════════
# D. OnlineRetrain → WalkForward validation
# ══════════════════════════════════════════════════════════════════════════


class TestRetrainWalkForward:
    """Retrain triggers and walk-forward validation integration."""

    def test_retrain_trigger_on_stale_model(self):
        retrainer = ModelRetrainer(
            model_dir=tempfile.mkdtemp(),
            max_age_days=0,  # force age trigger
            model_class=EnsembleSignalModel,
        )
        features = _make_synthetic_features(200)
        labels = _make_synthetic_labels(200)
        result = retrainer.check_and_retrain(features, labels)
        assert isinstance(result, RetrainResult)
        assert result.trigger.triggered

    def test_retrain_result_has_trigger(self):
        retrainer = ModelRetrainer(
            model_dir=tempfile.mkdtemp(),
            model_class=EnsembleSignalModel,
        )
        features = _make_synthetic_features(200)
        labels = _make_synthetic_labels(200)
        result = retrainer.check_and_retrain(features, labels, force=True)
        assert isinstance(result.trigger, RetrainTrigger)
        assert result.retrained

    def test_retrain_scheduler_delegates(self):
        scheduler = RetrainScheduler(
            model_dir=tempfile.mkdtemp(),
            model_class=EnsembleSignalModel,
        )
        features = _make_synthetic_features(200)
        labels = _make_synthetic_labels(200)
        result = scheduler.run_retrain_check(features, labels)
        assert isinstance(result, RetrainResult)

    def test_retrain_scheduler_sends_alert_on_trigger(self):
        mock_bot = MagicMock()
        scheduler = RetrainScheduler(
            model_dir=tempfile.mkdtemp(),
            model_class=EnsembleSignalModel,
            telegram_bot=mock_bot,
        )
        features = _make_synthetic_features(200)
        labels = _make_synthetic_labels(200)
        result = scheduler.run_retrain_check(features, labels)
        if result.trigger.triggered:
            mock_bot.send_message.assert_called_once()

    def test_walk_forward_validates_model(self):
        from compass.walk_forward import WalkForwardValidator
        import xgboost as xgb

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)

        # Build synthetic walk-forward data with dates
        rng = np.random.RandomState(42)
        n = 300
        features = _make_synthetic_features(n)
        dates = pd.bdate_range("2020-01-02", periods=n)
        df = features.copy()
        df["entry_date"] = dates
        df["win"] = _make_synthetic_labels(n)
        df["return_pct"] = rng.normal(5, 25, n)

        validator = WalkForwardValidator(
            model=model,
            numeric_features=PRUNED_FEATURES,
            categorical_features=[],
            min_train_samples=30,
        )
        result = validator.run(df)
        assert result["n_folds"] >= 1
        assert "aggregate" in result

    def test_retrain_produces_new_model_on_force(self):
        retrainer = ModelRetrainer(
            model_dir=tempfile.mkdtemp(),
            model_class=EnsembleSignalModel,
        )
        features = _make_synthetic_features(200)
        labels = _make_synthetic_labels(200)
        result = retrainer.check_and_retrain(features, labels, force=True)
        assert result.retrained
        assert result.new_model_path is not None


# ══════════════════════════════════════════════════════════════════════════
# E. Failure modes and edge cases
# ══════════════════════════════════════════════════════════════════════════


class TestFailureModes:
    """Verify graceful handling of bad inputs and edge conditions."""

    def test_signal_model_predict_before_train(self):
        model = SignalModel(model_dir=tempfile.mkdtemp())
        features = _make_synthetic_features(5)
        probs = model.predict_batch(features)
        # Should return fallback (zeros or uniform)
        assert len(probs) == 5

    def test_ensemble_predict_before_train(self):
        model = EnsembleSignalModel(model_dir=tempfile.mkdtemp())
        features = _make_synthetic_features(5)
        probs = model.predict_batch(features)
        assert len(probs) == 5

    def test_pipeline_handles_missing_columns(self):
        df = pd.DataFrame({"spy_price": [450, 460], "vix": [20, 25]})
        pipeline = FeaturePipeline(pruned=True)
        result = pipeline.transform(df)
        # Should produce a DataFrame (with default-filled missing cols)
        assert isinstance(result, pd.DataFrame)
        assert result.isna().sum().sum() == 0

    def test_sizing_with_zero_probability(self):
        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=0.0, win_return=0.30, loss_return=1.0, regime="bull",
        )
        assert result.position_fraction == 0.0

    def test_sizing_with_one_probability(self):
        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=1.0, win_return=0.30, loss_return=1.0, regime="bull",
        )
        assert result.position_fraction == 0.0  # Kelly undefined at p=1

    def test_hedge_controller_at_vix_zero(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        scale = ctrl.position_scale_factor(vix=0.0)
        assert scale == 1.0

    def test_hedge_controller_at_vix_extreme(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        scale = ctrl.position_scale_factor(vix=200.0)
        assert scale == 0.0

    def test_retrain_with_insufficient_data(self):
        retrainer = ModelRetrainer(
            model_dir=tempfile.mkdtemp(),
            model_class=EnsembleSignalModel,
            min_samples=500,
        )
        features = _make_synthetic_features(50)  # way below min_samples
        labels = _make_synthetic_labels(50)
        result = retrainer.check_and_retrain(features, labels)
        assert isinstance(result, RetrainResult)
        # Should not retrain with insufficient data
        assert not result.retrained

    def test_feature_pipeline_pruned_vs_full(self):
        df = pd.DataFrame({
            "spy_price": [450], "vix": [20], "contracts": [3],
            "net_credit": [0.65], "spread_width": [5.0],
            "max_loss_per_unit": [4.35], "regime": ["bull"],
            "strategy_type": ["CS"], "spread_type": ["bull_put"],
        })
        pruned = FeaturePipeline(pruned=True).transform(df)
        full = FeaturePipeline(pruned=False).transform(df)
        assert pruned.shape[1] < full.shape[1]
        assert "contracts_log" not in pruned.columns
        assert "contracts_log" in full.columns

    def test_get_hedge_config_unknown_returns_defaults(self):
        cfg = get_hedge_config("UNKNOWN-999")
        assert cfg.vix_scale_floor == 12.0  # default

    def test_portfolio_sizer_result_is_non_negative(self):
        sizer = PositionSizer()
        result = sizer.calculate_position_size(
            win_probability=0.1,  # low probability
            expected_return=0.05,
            expected_loss=-2.0,
            ml_confidence=0.3,
        )
        assert result["recommended_size"] >= 0


# ══════════════════════════════════════════════════════════════════════════
# F. Cross-cutting integration (multi-step chains)
# ══════════════════════════════════════════════════════════════════════════


class TestCrossCuttingIntegration:
    """Multi-step chains that exercise several modules together."""

    def test_full_chain_features_to_contracts(self, raw_training_df, pipeline_features, labels):
        """features → model → probability → kelly → contracts."""
        model = EnsembleSignalModel(model_dir=tempfile.mkdtemp())
        model.train(pipeline_features, labels, save_model=False)
        prob = float(model.predict_batch(pipeline_features[:1])[0])

        sizer = AdvancedPositionSizer()
        sizing = sizer.compute(
            win_prob=prob, win_return=0.40, loss_return=1.0,
            regime="bull", current_dd_pct=5.0,
        )

        contracts = sizer.compute_contracts(
            account_value=100_000, spread_width=5.0, credit_received=0.65,
            win_prob=prob, win_return=0.40, loss_return=1.0,
            regime="bull",
        )
        assert isinstance(contracts, int)
        assert contracts >= 0

    def test_pipeline_pruning_removes_expected_features(self):
        """Verify that PRUNED_REMOVED features are absent from pipeline output."""
        df = pd.DataFrame({
            "spy_price": [450], "vix": [20], "contracts": [3],
            "net_credit": [0.65], "spread_width": [5.0],
            "max_loss_per_unit": [4.35], "regime": ["bull"],
            "strategy_type": ["CS"], "spread_type": ["bull_put"],
        })
        result = FeaturePipeline(pruned=True).transform(df)
        for removed in PRUNED_REMOVED:
            assert removed not in result.columns

    def test_hedge_and_sizer_both_respond_to_same_regime(self):
        """Verify hedge and sizing both use the same regime taxonomy."""
        for regime in ["bull", "bear", "high_vol", "low_vol", "crash"]:
            ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
            scale = ctrl.position_scale_factor(vix=20.0, regime=regime)
            sizer = AdvancedPositionSizer()
            result = sizer.compute(
                win_prob=0.75, win_return=0.40, loss_return=1.0, regime=regime,
            )
            assert result.regime == regime
            assert 0 <= scale <= 1.0

    def test_regime_classifier_feeds_hedge_and_sizing(self):
        """RegimeClassifier → hedge scale → sizing regime fraction."""
        # Classify a regime
        classifier = RegimeClassifier()
        regime = classifier.classify(vix=35.0, spy_prices=pd.Series([450, 440, 430]), date=pd.Timestamp.now())
        regime_str = regime.value

        # Hedge controller uses the regime
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        scale = ctrl.position_scale_factor(vix=35.0, regime=regime_str)

        # Sizing uses the regime
        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.0,
            regime=regime_str,
        )
        assert result.regime == regime_str
        assert 0 <= scale <= 1.0
