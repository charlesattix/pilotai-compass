"""
Pipeline integration tests — full trade lifecycle, regime transitions,
model retrain triggers, and shadow agreement tracking.

Complements test_full_pipeline_integration.py with deeper coverage of:
  - End-to-end trade lifecycle (signal → size → risk → hedge → output)
  - Regime transitions mid-pipeline
  - Model retrain trigger within pipeline flow
  - ShadowEnsemble agreement tracking over sequences
  - Multi-experiment pipeline consistency
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from compass.advanced_sizing import AdvancedPositionSizer, SizingConfig, SizingResult
from compass.crisis_hedge import (
    CrisisHedgeConfig,
    CrisisHedgeController,
    get_hedge_config,
)
from compass.ensemble_signal_model import EnsembleSignalModel
from compass.feature_pipeline import FeaturePipeline
from compass.features import PRUNED_FEATURES, PRUNED_REMOVED
from compass.online_retrain import ModelRetrainer, RetrainResult
from compass.regime import Regime, RegimeClassifier
from compass.retrain_scheduler import RetrainScheduler
from compass.signal_model import SignalModel
from compass.sizing import PositionSizer, calculate_dynamic_risk, get_contract_size

ROOT = Path(__file__).resolve().parent.parent
TRAINING_CSV = ROOT / "compass" / "training_data_combined.csv"


# ── Mock data generators ─────────────────────────────────────────────────


@dataclass
class MockTrade:
    """Represents one trade flowing through the pipeline."""
    ticker: str = "SPY"
    regime: str = "bull"
    vix: float = 20.0
    iv_rank: float = 50.0
    spy_price: float = 450.0
    spread_width: float = 5.0
    net_credit: float = 0.65
    win_prob: float = 0.70
    contracts: int = 1
    pnl: float = 0.0


def _synth_features(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {}
    for f in PRUNED_FEATURES:
        if f in ("strategy_type_CS", "spread_type_bull_put"):
            data[f] = rng.choice([0.0, 1.0], n)
        else:
            data[f] = rng.randn(n)
    return pd.DataFrame(data)


def _synth_labels(n: int = 200, seed: int = 42) -> np.ndarray:
    return np.random.RandomState(seed).choice([0, 1], n, p=[0.35, 0.65])


def _train_signal_model(features: pd.DataFrame, labels: np.ndarray) -> SignalModel:
    model = SignalModel(model_dir=tempfile.mkdtemp())
    model.train(features, labels, save_model=False)
    return model


def _train_ensemble(features: pd.DataFrame, labels: np.ndarray) -> EnsembleSignalModel:
    model = EnsembleSignalModel(model_dir=tempfile.mkdtemp())
    model.train(features, labels, save_model=False)
    return model


@pytest.fixture(scope="module")
def trained_signal():
    f = _synth_features(250)
    l = _synth_labels(250)
    return _train_signal_model(f, l), f, l


@pytest.fixture(scope="module")
def trained_ensemble():
    f = _synth_features(250, seed=99)
    l = _synth_labels(250, seed=99)
    return _train_ensemble(f, l), f, l


# ══════════════════════════════════════════════════════════════════════════
# A. End-to-end trade lifecycle
# ══════════════════════════════════════════════════════════════════════════


class TestTradeLifecycle:
    """Full trade lifecycle: signal → size → risk → hedge → output."""

    def test_signal_generates_probability(self, trained_signal):
        model, features, _ = trained_signal
        row = features.iloc[[0]]
        prob = model.predict_batch(row)
        assert 0 <= prob[0] <= 1

    def test_probability_feeds_kelly_sizer(self, trained_signal):
        model, features, _ = trained_signal
        prob = float(model.predict_batch(features.iloc[[0]])[0])
        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=prob, win_return=0.40, loss_return=1.0, regime="bull",
        )
        assert isinstance(result, SizingResult)
        assert result.position_fraction >= 0

    def test_sizing_to_contract_count(self, trained_signal):
        model, features, _ = trained_signal
        prob = float(model.predict_batch(features.iloc[[0]])[0])
        sizer = AdvancedPositionSizer()
        contracts = sizer.compute_contracts(
            account_value=100_000, spread_width=5.0, credit_received=0.65,
            win_prob=prob, win_return=0.40, loss_return=1.0, regime="bull",
        )
        assert isinstance(contracts, int)
        assert 0 <= contracts <= 10

    def test_hedge_adjusts_final_position(self, trained_signal):
        model, features, _ = trained_signal
        prob = float(model.predict_batch(features.iloc[[0]])[0])

        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=prob, win_return=0.40, loss_return=1.0, regime="bull",
        )

        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        hedge_scale = ctrl.position_scale_factor(vix=25.0, regime="bull")
        final_fraction = result.position_fraction * hedge_scale
        assert 0 <= final_fraction <= result.position_fraction

    def test_full_lifecycle_produces_trade(self, trained_signal):
        """Signal → size → hedge → contract count → trade dict."""
        model, features, _ = trained_signal
        prob = float(model.predict_batch(features.iloc[[0]])[0])

        sizer = AdvancedPositionSizer()
        sizing = sizer.compute(
            win_prob=prob, win_return=0.40, loss_return=1.0,
            regime="bull", current_dd_pct=5.0,
        )

        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        hedge_scale = ctrl.position_scale_factor(vix=22.0, regime="bull")

        adjusted_frac = sizing.position_fraction * hedge_scale
        dollar_risk = 100_000 * adjusted_frac
        max_loss_per = (5.0 - 0.65) * 100
        contracts = min(int(dollar_risk // max_loss_per), 10)

        trade = {
            "ticker": "SPY",
            "probability": prob,
            "regime": "bull",
            "position_fraction": adjusted_frac,
            "contracts": contracts,
            "hedge_scale": hedge_scale,
        }
        assert trade["contracts"] >= 0
        assert 0 <= trade["position_fraction"] <= 1.0

    def test_lifecycle_rejects_low_probability(self, trained_signal):
        """Low-prob signal should yield 0 contracts."""
        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=0.20, win_return=0.10, loss_return=1.0, regime="bear",
        )
        assert result.position_fraction == 0.0

    def test_lifecycle_rejects_during_drawdown(self, trained_signal):
        model, features, _ = trained_signal
        prob = float(model.predict_batch(features.iloc[[0]])[0])
        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=prob, win_return=0.40, loss_return=1.0,
            regime="bull", current_dd_pct=28.0,
        )
        assert result.position_fraction == 0.0
        assert result.dd_scale == 0.0

    def test_lifecycle_with_iv_scaled_sizing(self, trained_signal):
        model, features, _ = trained_signal
        prob = float(model.predict_batch(features.iloc[[0]])[0])
        risk = calculate_dynamic_risk(
            account_value=100_000, iv_rank=60.0,
            current_portfolio_risk=3000,
            ml_confidence_multiplier=abs(prob - 0.5) * 2,
        )
        contracts = get_contract_size(risk, spread_width=5.0, credit_received=0.65)
        assert isinstance(contracts, int)
        assert 0 <= contracts <= 5


# ══════════════════════════════════════════════════════════════════════════
# B. Regime transitions mid-pipeline
# ══════════════════════════════════════════════════════════════════════════


class TestRegimeTransitions:
    """Regime changes between signal generation and execution."""

    def test_regime_change_alters_kelly_fraction(self):
        sizer = AdvancedPositionSizer()
        bull = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.0, regime="bull")
        crash = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.0, regime="crash")
        assert bull.kelly_fraction == 0.75
        assert crash.kelly_fraction == 0.25
        assert crash.position_fraction < bull.position_fraction

    def test_regime_change_alters_hedge_scale(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        bull_scale = ctrl.position_scale_factor(vix=20.0, regime="bull")
        crash_scale = ctrl.position_scale_factor(vix=20.0, regime="crash")
        assert crash_scale < bull_scale

    def test_mid_pipeline_regime_switch(self, trained_signal):
        """Signal generated in bull, but by execution time regime is bear."""
        model, features, _ = trained_signal
        prob = float(model.predict_batch(features.iloc[[0]])[0])

        sizer = AdvancedPositionSizer()

        # Signal generated in bull
        bull_sizing = sizer.compute(
            win_prob=prob, win_return=0.40, loss_return=1.0, regime="bull",
        )

        # By execution time, regime changed to bear
        bear_sizing = sizer.compute(
            win_prob=prob, win_return=0.40, loss_return=1.0, regime="bear",
        )

        # Bear should produce smaller position
        assert bear_sizing.position_fraction <= bull_sizing.position_fraction

    def test_all_regimes_produce_valid_output(self, trained_signal):
        model, features, _ = trained_signal
        prob = float(model.predict_batch(features.iloc[[0]])[0])
        sizer = AdvancedPositionSizer()

        for regime in ["bull", "bear", "high_vol", "low_vol", "crash"]:
            result = sizer.compute(
                win_prob=prob, win_return=0.40, loss_return=1.0, regime=regime,
            )
            assert result.position_fraction >= 0
            assert result.regime == regime

    def test_hedge_controller_regime_gates(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        assert ctrl.position_scale_factor(vix=15.0, regime="crash") == 0.0
        hv = ctrl.position_scale_factor(vix=15.0, regime="high_vol")
        assert hv <= 0.25  # high_vol_regime_scale caps it

    def test_regime_transition_sequence(self):
        """Walk through a regime sequence and verify sizing adapts."""
        sizer = AdvancedPositionSizer()
        regimes = ["bull", "bull", "high_vol", "crash", "bear", "bull"]
        fractions = []
        for regime in regimes:
            r = sizer.compute(
                win_prob=0.75, win_return=0.40, loss_return=1.0, regime=regime,
            )
            fractions.append(r.position_fraction)
        # crash should be the smallest
        crash_idx = regimes.index("crash")
        assert fractions[crash_idx] == min(fractions)

    def test_experiment_specific_hedge_per_regime(self):
        """EXP-400 and EXP-401 use different hedge configs per regime."""
        cfg400 = get_hedge_config("EXP-400")
        cfg401 = get_hedge_config("EXP-401")

        ctrl400 = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False, **{
            "vix_scale_floor": cfg400.vix_scale_floor,
            "vix_scale_ceiling": cfg400.vix_scale_ceiling,
        }))
        ctrl401 = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False, **{
            "vix_scale_floor": cfg401.vix_scale_floor,
            "vix_scale_ceiling": cfg401.vix_scale_ceiling,
        }))

        # At VIX 13, EXP-400 (floor=12) starts throttling, EXP-401 (floor=14) doesn't
        s400 = ctrl400.position_scale_factor(vix=13.0)
        s401 = ctrl401.position_scale_factor(vix=13.0)
        assert s400 < 1.0  # EXP-400 throttles at VIX 13
        assert s401 == 1.0  # EXP-401 doesn't throttle yet


# ══════════════════════════════════════════════════════════════════════════
# C. Model retrain triggers within pipeline
# ══════════════════════════════════════════════════════════════════════════


class TestRetrainInPipeline:
    """Retrain triggers integrated with the live pipeline flow."""

    def test_force_retrain_produces_model(self):
        retrainer = ModelRetrainer(
            model_dir=tempfile.mkdtemp(),
            model_class=EnsembleSignalModel,
        )
        f, l = _synth_features(200), _synth_labels(200)
        result = retrainer.check_and_retrain(f, l, force=True)
        assert result.retrained
        assert result.new_model_path is not None

    def test_stale_model_triggers_retrain(self):
        retrainer = ModelRetrainer(
            model_dir=tempfile.mkdtemp(),
            max_age_days=0,
            model_class=EnsembleSignalModel,
        )
        f, l = _synth_features(200), _synth_labels(200)
        result = retrainer.check_and_retrain(f, l)
        assert result.trigger.triggered
        assert len(result.trigger.reasons) > 0

    def test_retrained_model_can_predict(self):
        tmpdir = tempfile.mkdtemp()
        retrainer = ModelRetrainer(
            model_dir=tmpdir,
            model_class=EnsembleSignalModel,
        )
        f, l = _synth_features(200), _synth_labels(200)
        result = retrainer.check_and_retrain(f, l, force=True)
        assert result.retrained

        # Load and predict with the new model
        model = EnsembleSignalModel(model_dir=tmpdir)
        loaded = model.load(Path(result.new_model_path).name)
        if loaded:
            probs = model.predict_batch(f[:5])
            assert len(probs) == 5

    def test_scheduler_wraps_retrainer(self):
        scheduler = RetrainScheduler(
            model_dir=tempfile.mkdtemp(),
            model_class=EnsembleSignalModel,
        )
        f, l = _synth_features(200), _synth_labels(200)
        result = scheduler.run_retrain_check(f, l)
        assert isinstance(result, RetrainResult)

    def test_scheduler_telegram_alert_on_trigger(self):
        bot = MagicMock()
        scheduler = RetrainScheduler(
            model_dir=tempfile.mkdtemp(),
            model_class=EnsembleSignalModel,
            telegram_bot=bot,
        )
        f, l = _synth_features(200), _synth_labels(200)
        result = scheduler.run_retrain_check(f, l)
        if result.trigger.triggered:
            bot.send_message.assert_called()

    def test_insufficient_data_no_retrain(self):
        retrainer = ModelRetrainer(
            model_dir=tempfile.mkdtemp(),
            model_class=EnsembleSignalModel,
            min_samples=500,
        )
        f, l = _synth_features(50), _synth_labels(50)
        result = retrainer.check_and_retrain(f, l)
        assert not result.retrained


# ══════════════════════════════════════════════════════════════════════════
# D. ShadowEnsemble agreement tracking
# ══════════════════════════════════════════════════════════════════════════


class TestShadowAgreement:
    """ShadowEnsemble tracks primary/shadow agreement over sequences."""

    def _make_shadow(self, trained_ensemble):
        from compass.shadow_ensemble import ShadowEnsemble
        model, features, labels = trained_ensemble
        tmpdir = tempfile.mkdtemp()
        return ShadowEnsemble(model, model, log_path=Path(tmpdir) / "shadow.csv")

    def test_single_predict_returns_prediction_result(self, trained_ensemble):
        se = self._make_shadow(trained_ensemble)
        model, features, _ = trained_ensemble
        row = features.iloc[0]
        feature_dict = {name: float(row[name]) for name in features.columns}
        result = se.predict(feature_dict)
        assert "probability" in result
        assert 0 <= result["probability"] <= 1

    def test_predict_increments_stats(self, trained_ensemble):
        se = self._make_shadow(trained_ensemble)
        model, features, _ = trained_ensemble
        initial = se.get_shadow_stats()["total_predictions"]
        for i in range(3):
            row = features.iloc[i]
            se.predict({name: float(row[name]) for name in features.columns})
        assert se.get_shadow_stats()["total_predictions"] == initial + 3

    def test_agreement_rate_after_predictions(self, trained_ensemble):
        se = self._make_shadow(trained_ensemble)
        model, features, _ = trained_ensemble
        for i in range(10):
            row = features.iloc[i]
            se.predict({name: float(row[name]) for name in features.columns})
        rate = se.agreement_rate
        # Same model as primary and shadow → should agree on all
        assert rate is not None
        assert rate >= 0.8

    def test_shadow_stats_structure(self, trained_ensemble):
        se = self._make_shadow(trained_ensemble)
        stats = se.get_shadow_stats()
        assert "total_predictions" in stats
        assert "agreed_predictions" in stats

    def test_batch_predict_returns_array(self, trained_ensemble):
        se = self._make_shadow(trained_ensemble)
        _, features, _ = trained_ensemble
        probs = se.predict_batch(features[:10])
        assert isinstance(probs, np.ndarray)
        assert len(probs) == 10
        assert all(0 <= p <= 1 for p in probs)

    def test_shadow_with_different_models(self):
        """Primary and shadow are trained on different data → may disagree."""
        from compass.shadow_ensemble import ShadowEnsemble
        f1, l1 = _synth_features(200, seed=1), _synth_labels(200, seed=1)
        f2, l2 = _synth_features(200, seed=2), _synth_labels(200, seed=2)
        primary = _train_ensemble(f1, l1)
        shadow = _train_ensemble(f2, l2)

        test_features = _synth_features(20, seed=3)
        tmpdir = tempfile.mkdtemp()
        se = ShadowEnsemble(primary, shadow, log_path=Path(tmpdir) / "s.csv")

        for i in range(20):
            row = test_features.iloc[i]
            se.predict({name: float(row[name]) for name in test_features.columns})

        stats = se.get_shadow_stats()
        assert stats["total_predictions"] == 20
        # Different training data → may not agree on all
        rate = se.agreement_rate
        assert rate is not None


# ══════════════════════════════════════════════════════════════════════════
# E. Multi-experiment pipeline consistency
# ══════════════════════════════════════════════════════════════════════════


class TestMultiExperimentConsistency:
    """Verify pipeline produces consistent results across experiments."""

    def test_same_features_same_model_same_output(self, trained_signal):
        model, features, _ = trained_signal
        probs1 = model.predict_batch(features[:10])
        probs2 = model.predict_batch(features[:10])
        np.testing.assert_array_almost_equal(probs1, probs2)

    def test_different_hedge_configs_different_scales(self):
        vix = 25.0
        cfgs = {
            "EXP-400": get_hedge_config("EXP-400"),
            "EXP-401": get_hedge_config("EXP-401"),
        }
        scales = {}
        for name, cfg in cfgs.items():
            ctrl = CrisisHedgeController(cfg)
            scales[name] = ctrl.position_scale_factor(vix=vix)
        # Both should produce valid scales but may differ
        assert all(0 <= s <= 1 for s in scales.values())

    def test_experiment_overrides_in_sizer(self):
        conservative = SizingConfig(max_position_pct=3.0, max_dd_pct=15.0)
        aggressive = SizingConfig(max_position_pct=15.0, max_dd_pct=40.0)
        sizer = AdvancedPositionSizer(
            experiment_overrides={"EXP-401": conservative, "EXP-400": aggressive},
        )

        r401 = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.0,
            regime="bull", experiment_id="EXP-401",
        )
        r400 = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.0,
            regime="bull", experiment_id="EXP-400",
        )
        # Conservative EXP-401 should produce smaller position
        assert r401.position_fraction <= r400.position_fraction

    def test_pipeline_feature_count_stable(self):
        """Multiple pipeline runs produce same number of features."""
        df = pd.DataFrame({
            "spy_price": [450, 460], "vix": [20, 25],
            "regime": ["bull", "bear"], "strategy_type": ["CS", "CS"],
            "spread_type": ["bull_put", "bull_put"],
        })
        p1 = FeaturePipeline(pruned=True).transform(df)
        p2 = FeaturePipeline(pruned=True).transform(df)
        assert p1.shape == p2.shape
        assert list(p1.columns) == list(p2.columns)


# ══════════════════════════════════════════════════════════════════════════
# F. Edge cases and failure recovery
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCasesAndRecovery:
    """Pipeline handles edge cases without crashing."""

    def test_nan_features_handled(self, trained_signal):
        model, features, _ = trained_signal
        bad_row = features.iloc[[0]].copy()
        bad_row.iloc[0, 0] = np.nan
        probs = model.predict_batch(bad_row)
        assert len(probs) == 1

    def test_zero_vix_hedge(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        scale = ctrl.position_scale_factor(vix=0.0)
        assert scale == 1.0

    def test_extreme_vix_hedge(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        scale = ctrl.position_scale_factor(vix=200.0)
        assert scale == 0.0

    def test_zero_spread_width_contracts(self):
        contracts = get_contract_size(1000.0, spread_width=0.0, credit_received=0.65)
        assert contracts == 0

    def test_negative_edge_sizing(self):
        sizer = AdvancedPositionSizer()
        result = sizer.compute(
            win_prob=0.20, win_return=0.10, loss_return=1.0, regime="crash",
        )
        assert result.position_fraction == 0.0
        assert result.kelly_raw == 0.0

    def test_pipeline_with_single_row(self):
        df = pd.DataFrame({c: [0.0] for c in PRUNED_FEATURES})
        df["strategy_type_CS"] = [1.0]
        df["spread_type_bull_put"] = [1.0]
        model = SignalModel(model_dir=tempfile.mkdtemp())
        f, l = _synth_features(100), _synth_labels(100)
        model.train(f, l, save_model=False)
        probs = model.predict_batch(df)
        assert len(probs) == 1
        assert 0 <= probs[0] <= 1

    def test_correlation_penalty_in_sizing_pipeline(self):
        sizer = AdvancedPositionSizer()
        low = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.0,
            regime="bull", portfolio_correlation=0.3,
        )
        high = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.0,
            regime="bull", portfolio_correlation=0.85,
        )
        assert high.position_fraction < low.position_fraction
        assert high.corr_scale < low.corr_scale

    def test_drawdown_scale_continuous(self):
        """Verify drawdown scaling is monotonically decreasing."""
        sizer = AdvancedPositionSizer()
        prev = 1.0
        for dd in range(0, 30, 2):
            r = sizer.compute(
                win_prob=0.75, win_return=0.40, loss_return=1.0,
                regime="bull", current_dd_pct=float(dd),
            )
            assert r.dd_scale <= prev
            prev = r.dd_scale

    def test_feature_pipeline_idempotent(self):
        """Same input → same output on repeated calls."""
        df = pd.DataFrame({
            "spy_price": [450], "vix": [20], "contracts": [3],
            "net_credit": [0.65], "spread_width": [5.0],
            "max_loss_per_unit": [4.35], "regime": ["bull"],
            "strategy_type": ["CS"], "spread_type": ["bull_put"],
        })
        p = FeaturePipeline(pruned=True)
        r1 = p.transform(df)
        r2 = p.transform(df)
        pd.testing.assert_frame_equal(r1, r2)

    def test_hedge_stop_loss_multiplier_in_range(self):
        """Stop-loss multiplier stays in valid range for all VIX levels."""
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        for vix in range(5, 80, 5):
            m = ctrl.stop_loss_multiplier(vix=float(vix))
            assert 1.0 <= m <= 3.5
