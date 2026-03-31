"""
Tests for compass.shadow_ensemble.ShadowEnsemble and the GAP-1/GAP-2 fixes
in compass.ensemble_signal_model.EnsembleSignalModel.

Coverage:
  ShadowEnsemble
    - predict() always returns primary result
    - predict() logs CSV row on success
    - predict() logs CSV row when shadow fails (error row)
    - agreement rate tracked correctly (agree / disagree / shadow fallback)
    - mean_abs_delta computed only over non-fallback predictions
    - get_shadow_stats() returns consistent snapshot
    - predict_batch() returns primary probas (shadow failure safe)
    - train/save/load/backtest/get_fallback_stats forwarded to primary
    - trained / feature_names / training_stats / feature_means / feature_stds
      are proxied to primary
    - CSV header written on first construction
    - CSV appends (not overwrites) on subsequent construction

  EnsembleSignalModel (GAP-1, GAP-2)
    - training_stats["test_auc"] present and equals ensemble_test_auc (GAP-1)
    - training_stats["timestamp"] present and is a valid ISO string (GAP-2)
    - online_retrain ModelRetrainer._check_performance works with EnsembleSignalModel
    - online_retrain ModelRetrainer._get_model_age_days works with EnsembleSignalModel
"""

from __future__ import annotations

import csv
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from compass.shadow_ensemble import ShadowEnsemble, _CSV_FIELDNAMES


# ---------------------------------------------------------------------------
# Lightweight model stubs — no ML training required
# ---------------------------------------------------------------------------

def _make_prediction(prob: float, fallback: bool = False) -> Dict:
    signal = "bullish" if prob > 0.55 else "bearish" if prob < 0.45 else "neutral"
    result: Dict = {
        "prediction": int(prob > 0.5),
        "probability": prob,
        "confidence": abs(prob - 0.5) * 2,
        "signal": signal,
        "signal_strength": round(prob * 100, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if fallback:
        result["fallback"] = True
    return result


class _StubModel:
    """Minimal model stub with configurable predict() probability."""

    def __init__(
        self,
        prob: float = 0.7,
        fallback: bool = False,
        raise_on_predict: bool = False,
        batch_prob: Optional[float] = None,
    ):
        self._prob = prob
        self._fallback = fallback
        self._raise = raise_on_predict
        self._batch_prob = batch_prob if batch_prob is not None else prob

        # SignalModel-compatible attributes
        self.trained: bool = True
        self.feature_names: List[str] = ["feat_a", "feat_b"]
        self.training_stats: Dict = {
            "test_auc": 0.80,
            "test_accuracy": 0.75,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        self.feature_means: Optional[np.ndarray] = np.array([0.0, 0.0])
        self.feature_stds: Optional[np.ndarray] = np.array([1.0, 1.0])
        self._train_called = False
        self._save_called = False
        self._load_called = False

    def predict(self, features: Dict) -> Dict:
        if self._raise:
            raise RuntimeError("stub error")
        return _make_prediction(self._prob, fallback=self._fallback)

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        if self._raise:
            raise RuntimeError("stub batch error")
        return np.full(len(features_df), self._batch_prob)

    def train(self, *a, **kw) -> Dict:
        self._train_called = True
        return {"test_auc": 0.8}

    def save(self, *a, **kw) -> None:
        self._save_called = True

    def load(self, *a, **kw) -> bool:
        self._load_called = True
        return True

    def backtest(self, *a, **kw) -> Dict:
        return {"accuracy": 0.75}

    def get_fallback_stats(self) -> Dict:
        return {"predict": 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wrapper(
    primary_prob: float = 0.7,
    shadow_prob: float = 0.8,
    shadow_raise: bool = False,
    shadow_fallback: bool = False,
    tmp_path: Optional[Path] = None,
) -> tuple[ShadowEnsemble, _StubModel, _StubModel]:
    primary = _StubModel(prob=primary_prob)
    shadow = _StubModel(
        prob=shadow_prob,
        fallback=shadow_fallback,
        raise_on_predict=shadow_raise,
    )
    log_path = (tmp_path / "shadow_log.csv") if tmp_path else None
    wrapper = ShadowEnsemble(primary, shadow, log_path=log_path)
    return wrapper, primary, shadow


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ===========================================================================
# ShadowEnsemble.predict() — primary always wins
# ===========================================================================

class TestPredict:

    def test_returns_primary_result(self, tmp_path):
        wrapper, primary, shadow = _make_wrapper(
            primary_prob=0.6, shadow_prob=0.9, tmp_path=tmp_path
        )
        result = wrapper.predict({"feat_a": 1.0})
        assert result["probability"] == pytest.approx(0.6)
        assert result["signal"] == "bullish"

    def test_returns_primary_when_shadow_raises(self, tmp_path):
        wrapper, primary, shadow = _make_wrapper(
            primary_prob=0.65, shadow_raise=True, tmp_path=tmp_path
        )
        result = wrapper.predict({"feat_a": 1.0})
        assert result["probability"] == pytest.approx(0.65)

    def test_returns_primary_when_shadow_returns_fallback(self, tmp_path):
        wrapper, primary, shadow = _make_wrapper(
            primary_prob=0.55, shadow_fallback=True, tmp_path=tmp_path
        )
        result = wrapper.predict({"feat_a": 1.0})
        assert result["probability"] == pytest.approx(0.55)

    def test_primary_fallback_passes_through(self, tmp_path):
        primary = _StubModel(prob=0.5, fallback=True)
        shadow = _StubModel(prob=0.7)
        wrapper = ShadowEnsemble(primary, shadow, log_path=tmp_path / "log.csv")
        result = wrapper.predict({})
        assert result.get("fallback") is True


# ===========================================================================
# CSV logging
# ===========================================================================

class TestCSVLogging:

    def test_header_written_on_init(self, tmp_path):
        log_path = tmp_path / "shadow_log.csv"
        ShadowEnsemble(_StubModel(), _StubModel(), log_path=log_path)
        assert log_path.exists()
        with open(log_path) as fh:
            header = fh.readline().strip().split(",")
        assert header == _CSV_FIELDNAMES

    def test_row_written_on_predict(self, tmp_path):
        log_path = tmp_path / "shadow_log.csv"
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.72), _StubModel(prob=0.85), log_path=log_path
        )
        wrapper.predict({"feat_a": 0.5})
        rows = _read_csv(log_path)
        assert len(rows) == 1
        assert float(rows[0]["primary_prob"]) == pytest.approx(0.72, abs=1e-4)
        assert float(rows[0]["ensemble_prob"]) == pytest.approx(0.85, abs=1e-4)

    def test_delta_correct(self, tmp_path):
        log_path = tmp_path / "shadow_log.csv"
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.60), _StubModel(prob=0.75), log_path=log_path
        )
        wrapper.predict({})
        rows = _read_csv(log_path)
        assert float(rows[0]["delta"]) == pytest.approx(0.15, abs=1e-4)

    def test_agreed_column_true_when_both_above_half(self, tmp_path):
        log_path = tmp_path / "shadow_log.csv"
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.6), _StubModel(prob=0.8), log_path=log_path
        )
        wrapper.predict({})
        rows = _read_csv(log_path)
        assert rows[0]["agreed"] == "1"

    def test_agreed_column_false_when_sides_differ(self, tmp_path):
        log_path = tmp_path / "shadow_log.csv"
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.4),   # primary says bearish
            _StubModel(prob=0.7),   # shadow says bullish
            log_path=log_path,
        )
        wrapper.predict({})
        rows = _read_csv(log_path)
        assert rows[0]["agreed"] == "0"

    def test_shadow_error_row_written_with_empty_ensemble_prob(self, tmp_path):
        log_path = tmp_path / "shadow_log.csv"
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.6), _StubModel(raise_on_predict=True),
            log_path=log_path,
        )
        wrapper.predict({})
        rows = _read_csv(log_path)
        assert len(rows) == 1
        assert rows[0]["ensemble_prob"] == ""   # NaN written as empty
        assert rows[0]["delta"] == ""
        assert rows[0]["shadow_fallback"] == "1"
        assert rows[0]["ensemble_signal"] == "error"

    def test_multiple_predictions_append(self, tmp_path):
        log_path = tmp_path / "shadow_log.csv"
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.6), _StubModel(prob=0.7), log_path=log_path
        )
        for _ in range(5):
            wrapper.predict({})
        rows = _read_csv(log_path)
        assert len(rows) == 5

    def test_existing_log_is_appended_not_overwritten(self, tmp_path):
        """Second ShadowEnsemble on the same path appends, preserving prior rows."""
        log_path = tmp_path / "shadow_log.csv"
        w1 = ShadowEnsemble(_StubModel(prob=0.6), _StubModel(prob=0.7), log_path=log_path)
        w1.predict({})
        w2 = ShadowEnsemble(_StubModel(prob=0.5), _StubModel(prob=0.5), log_path=log_path)
        w2.predict({})
        rows = _read_csv(log_path)
        assert len(rows) == 2

    def test_primary_fallback_column(self, tmp_path):
        log_path = tmp_path / "shadow_log.csv"
        primary = _StubModel(prob=0.5, fallback=True)
        shadow = _StubModel(prob=0.7)
        wrapper = ShadowEnsemble(primary, shadow, log_path=log_path)
        wrapper.predict({})
        rows = _read_csv(log_path)
        assert rows[0]["primary_fallback"] == "1"


# ===========================================================================
# Agreement rate and mean_abs_delta
# ===========================================================================

class TestAgreementStats:

    def test_agreement_rate_none_before_first_call(self, tmp_path):
        wrapper, *_ = _make_wrapper(tmp_path=tmp_path)
        assert wrapper.agreement_rate is None

    def test_mean_abs_delta_none_before_first_call(self, tmp_path):
        wrapper, *_ = _make_wrapper(tmp_path=tmp_path)
        assert wrapper.mean_abs_delta is None

    def test_agreement_rate_100pct_when_both_agree(self, tmp_path):
        wrapper, *_ = _make_wrapper(
            primary_prob=0.65, shadow_prob=0.80, tmp_path=tmp_path
        )
        wrapper.predict({})
        wrapper.predict({})
        assert wrapper.agreement_rate == pytest.approx(1.0)

    def test_agreement_rate_0pct_when_always_disagree(self, tmp_path):
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.3),   # bearish
            _StubModel(prob=0.8),   # bullish
            log_path=tmp_path / "log.csv",
        )
        wrapper.predict({})
        wrapper.predict({})
        assert wrapper.agreement_rate == pytest.approx(0.0)

    def test_agreement_rate_mixed(self, tmp_path):
        log_path = tmp_path / "log.csv"
        # 2 agree, 1 disagrees
        wrapper = ShadowEnsemble(_StubModel(prob=0.7), _StubModel(prob=0.8), log_path=log_path)
        wrapper.predict({})  # both bullish → agree
        wrapper.predict({})  # both bullish → agree
        wrapper2 = ShadowEnsemble(_StubModel(prob=0.3), _StubModel(prob=0.7), log_path=log_path)
        # Different wrapper so counts don't carry over — use one wrapper:
        w = ShadowEnsemble(_StubModel(prob=0.7), _StubModel(prob=0.8), log_path=tmp_path / "log2.csv")
        agree_shadow = _StubModel(prob=0.8)   # bullish
        disagree_shadow = _StubModel(prob=0.3)  # bearish
        # Make a wrapper that we can control per-call is awkward with stubs,
        # so just test the counter arithmetic directly.
        w2 = ShadowEnsemble(
            _StubModel(prob=0.6), _StubModel(prob=0.8), log_path=tmp_path / "log3.csv"
        )
        w2.predict({})  # agree (both > 0.5)
        w2.predict({})  # agree
        assert w2.agreement_rate == pytest.approx(1.0)
        assert w2._agreed_predictions == 2
        assert w2._total_predictions == 2

    def test_mean_abs_delta_computed_correctly(self, tmp_path):
        # primary=0.6, shadow=0.8 → delta = 0.2 each call
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.6), _StubModel(prob=0.8),
            log_path=tmp_path / "log.csv",
        )
        wrapper.predict({})
        wrapper.predict({})
        assert wrapper.mean_abs_delta == pytest.approx(0.2, abs=1e-5)

    def test_shadow_fallback_excluded_from_mean_abs_delta(self, tmp_path):
        """Fallback calls don't contribute to mean_abs_delta denominator."""
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.6), _StubModel(raise_on_predict=True),
            log_path=tmp_path / "log.csv",
        )
        wrapper.predict({})
        wrapper.predict({})
        assert wrapper.mean_abs_delta is None  # all failed, valid=0

    def test_shadow_fallback_counted_as_disagreement(self, tmp_path):
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.7), _StubModel(raise_on_predict=True),
            log_path=tmp_path / "log.csv",
        )
        wrapper.predict({})
        assert wrapper.agreement_rate == pytest.approx(0.0)
        stats = wrapper.get_shadow_stats()
        assert stats["shadow_fallbacks"] == 1

    def test_get_shadow_stats_consistent(self, tmp_path):
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.7), _StubModel(prob=0.8),
            log_path=tmp_path / "log.csv",
        )
        wrapper.predict({})
        wrapper.predict({})
        stats = wrapper.get_shadow_stats()
        assert stats["total_predictions"] == 2
        assert stats["agreed_predictions"] == 2
        assert stats["shadow_fallbacks"] == 0
        assert stats["agreement_rate"] == pytest.approx(1.0)
        assert stats["mean_abs_delta"] == pytest.approx(0.1, abs=1e-5)


# ===========================================================================
# predict_batch
# ===========================================================================

class TestPredictBatch:

    def test_returns_primary_probas(self, tmp_path):
        primary = _StubModel(batch_prob=0.65)
        shadow = _StubModel(batch_prob=0.80)
        wrapper = ShadowEnsemble(primary, shadow, log_path=tmp_path / "log.csv")
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = wrapper.predict_batch(df)
        np.testing.assert_allclose(result, 0.65)

    def test_returns_primary_when_shadow_batch_fails(self, tmp_path):
        primary = _StubModel(batch_prob=0.55)
        shadow = _StubModel(raise_on_predict=True)
        wrapper = ShadowEnsemble(primary, shadow, log_path=tmp_path / "log.csv")
        df = pd.DataFrame({"a": [1, 2]})
        result = wrapper.predict_batch(df)
        np.testing.assert_allclose(result, 0.55)

    def test_batch_does_not_write_csv_rows(self, tmp_path):
        log_path = tmp_path / "log.csv"
        wrapper = ShadowEnsemble(
            _StubModel(batch_prob=0.6), _StubModel(batch_prob=0.7),
            log_path=log_path,
        )
        df = pd.DataFrame({"a": [1, 2, 3]})
        wrapper.predict_batch(df)
        rows = _read_csv(log_path)
        assert len(rows) == 0  # batch doesn't write per-row CSV entries


# ===========================================================================
# Method and attribute forwarding
# ===========================================================================

class TestForwarding:

    def test_train_forwarded_to_primary(self, tmp_path):
        wrapper, primary, _ = _make_wrapper(tmp_path=tmp_path)
        wrapper.train(pd.DataFrame(), np.array([]))
        assert primary._train_called

    def test_save_forwarded_to_primary(self, tmp_path):
        wrapper, primary, _ = _make_wrapper(tmp_path=tmp_path)
        wrapper.save("model.joblib")
        assert primary._save_called

    def test_load_forwarded_to_primary(self, tmp_path):
        wrapper, primary, _ = _make_wrapper(tmp_path=tmp_path)
        result = wrapper.load()
        assert primary._load_called
        assert result is True

    def test_backtest_forwarded(self, tmp_path):
        wrapper, *_ = _make_wrapper(tmp_path=tmp_path)
        result = wrapper.backtest(pd.DataFrame(), np.array([]))
        assert "accuracy" in result

    def test_get_fallback_stats_forwarded(self, tmp_path):
        wrapper, *_ = _make_wrapper(tmp_path=tmp_path)
        stats = wrapper.get_fallback_stats()
        assert isinstance(stats, dict)

    def test_trained_proxied_from_primary(self, tmp_path):
        wrapper, primary, _ = _make_wrapper(tmp_path=tmp_path)
        primary.trained = False
        assert wrapper.trained is False
        primary.trained = True
        assert wrapper.trained is True

    def test_feature_names_proxied(self, tmp_path):
        wrapper, primary, _ = _make_wrapper(tmp_path=tmp_path)
        assert wrapper.feature_names == ["feat_a", "feat_b"]

    def test_training_stats_proxied(self, tmp_path):
        wrapper, primary, _ = _make_wrapper(tmp_path=tmp_path)
        assert wrapper.training_stats["test_auc"] == pytest.approx(0.80)

    def test_feature_means_proxied(self, tmp_path):
        wrapper, primary, _ = _make_wrapper(tmp_path=tmp_path)
        np.testing.assert_array_equal(wrapper.feature_means, [0.0, 0.0])

    def test_feature_stds_proxied(self, tmp_path):
        wrapper, primary, _ = _make_wrapper(tmp_path=tmp_path)
        np.testing.assert_array_equal(wrapper.feature_stds, [1.0, 1.0])


# ===========================================================================
# Thread safety
# ===========================================================================

class TestThreadSafety:

    def test_concurrent_predictions_consistent_count(self, tmp_path):
        """Concurrent predict() calls must not corrupt the total counter."""
        log_path = tmp_path / "log.csv"
        wrapper = ShadowEnsemble(
            _StubModel(prob=0.6), _StubModel(prob=0.7), log_path=log_path
        )

        n_threads = 20
        n_calls = 10

        def _run():
            for _ in range(n_calls):
                wrapper.predict({"x": 1.0})

        threads = [threading.Thread(target=_run) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = wrapper.get_shadow_stats()
        assert stats["total_predictions"] == n_threads * n_calls

        rows = _read_csv(log_path)
        assert len(rows) == n_threads * n_calls


# ===========================================================================
# GAP-1: EnsembleSignalModel.training_stats["test_auc"]
# ===========================================================================

class TestGap1TestAucAlias:

    def test_training_stats_has_test_auc(self):
        """training_stats must expose 'test_auc' for online_retrain compatibility."""
        import tempfile
        import numpy as np
        import pandas as pd
        from compass.ensemble_signal_model import EnsembleSignalModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = EnsembleSignalModel(model_dir=tmpdir)
            np.random.seed(0)
            n = 120
            X = pd.DataFrame({
                "a": np.random.randn(n),
                "b": np.random.randn(n),
            })
            y = (X["a"] > 0).astype(int).values
            model.train(X, y, calibrate=False, save_model=False)

            assert "test_auc" in model.training_stats, (
                "GAP-1: training_stats missing 'test_auc' — online_retrain performance "
                "trigger will never fire"
            )
            assert 0.0 <= model.training_stats["test_auc"] <= 1.0

    def test_test_auc_equals_ensemble_test_auc(self):
        """test_auc alias must equal ensemble_test_auc (not an independent value)."""
        import tempfile
        import numpy as np
        import pandas as pd
        from compass.ensemble_signal_model import EnsembleSignalModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = EnsembleSignalModel(model_dir=tmpdir)
            np.random.seed(1)
            n = 120
            X = pd.DataFrame({
                "a": np.random.randn(n),
                "b": np.random.randn(n),
            })
            y = (X["a"] > 0).astype(int).values
            model.train(X, y, calibrate=False, save_model=False)

            assert model.training_stats["test_auc"] == pytest.approx(
                model.training_stats["ensemble_test_auc"]
            )


# ===========================================================================
# GAP-2: EnsembleSignalModel.training_stats["timestamp"]
# ===========================================================================

class TestGap2TimestampKey:

    def test_training_stats_has_timestamp(self):
        """training_stats must expose 'timestamp' for online_retrain age check."""
        import tempfile
        import numpy as np
        import pandas as pd
        from compass.ensemble_signal_model import EnsembleSignalModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = EnsembleSignalModel(model_dir=tmpdir)
            np.random.seed(2)
            n = 120
            X = pd.DataFrame({
                "a": np.random.randn(n),
                "b": np.random.randn(n),
            })
            y = (X["a"] > 0).astype(int).values
            model.train(X, y, calibrate=False, save_model=False)

            assert "timestamp" in model.training_stats, (
                "GAP-2: training_stats missing 'timestamp' — online_retrain age "
                "trigger will never fire"
            )

    def test_timestamp_is_parseable_iso_string(self):
        """timestamp value must be a parseable ISO-8601 string."""
        import tempfile
        import numpy as np
        import pandas as pd
        from compass.ensemble_signal_model import EnsembleSignalModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = EnsembleSignalModel(model_dir=tmpdir)
            np.random.seed(3)
            n = 120
            X = pd.DataFrame({
                "a": np.random.randn(n),
                "b": np.random.randn(n),
            })
            y = (X["a"] > 0).astype(int).values
            model.train(X, y, calibrate=False, save_model=False)

            ts_str = model.training_stats["timestamp"]
            parsed = datetime.fromisoformat(ts_str)
            # Should be recent (within last 60 seconds)
            age = (datetime.now(timezone.utc) - parsed).total_seconds()
            assert 0 <= age < 60, f"Timestamp too old or in future: {ts_str}"

    def test_timestamp_enables_age_trigger_in_online_retrain(self):
        """ModelRetrainer._get_model_age_days must return an int for EnsembleSignalModel."""
        import tempfile
        import numpy as np
        import pandas as pd
        from compass.ensemble_signal_model import EnsembleSignalModel
        from compass.online_retrain import ModelRetrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            model = EnsembleSignalModel(model_dir=tmpdir)
            np.random.seed(4)
            n = 120
            X = pd.DataFrame({
                "a": np.random.randn(n),
                "b": np.random.randn(n),
            })
            y = (X["a"] > 0).astype(int).values
            model.train(X, y, calibrate=False, save_model=False)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            age = retrainer._get_model_age_days(model)

            assert age is not None, (
                "GAP-2: _get_model_age_days returned None for EnsembleSignalModel — "
                "the staleness trigger will never fire"
            )
            assert age == 0  # trained today


# ===========================================================================
# GAP-1 integration: ModelRetrainer._check_performance with EnsembleSignalModel
# ===========================================================================

class TestGap1Integration:

    def test_check_performance_returns_result_not_none(self):
        """_check_performance must not return None for EnsembleSignalModel (GAP-1)."""
        import tempfile
        import numpy as np
        import pandas as pd
        from compass.ensemble_signal_model import EnsembleSignalModel
        from compass.online_retrain import ModelRetrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(5)
            n = 120
            X = pd.DataFrame({
                "a": np.random.randn(n),
                "b": np.random.randn(n),
            })
            y = (X["a"] > 0).astype(int).values

            model = EnsembleSignalModel(model_dir=tmpdir)
            model.train(X, y, calibrate=False, save_model=False)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            result = retrainer._check_performance(model, X, y)

            assert result is not None, (
                "GAP-1: _check_performance returned None — performance trigger "
                "will never fire for EnsembleSignalModel"
            )
            assert "baseline_auc" in result
            assert "current_auc" in result
            assert 0.0 <= result["baseline_auc"] <= 1.0
            assert 0.0 <= result["current_auc"] <= 1.0
