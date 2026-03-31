# Phase 4 Integration Plan: EnsembleSignalModel → Production

**Date:** 2026-03-27
**Author:** Maximus / Claude
**Scope:** Swap `SignalModel` (standalone XGBoost) for `EnsembleSignalModel` (XGB + RF + ET, walk-forward weighted) across all production paths.

---

## 1. Interface Compatibility Check

### 1.1 Public Method Comparison

| Method | SignalModel | EnsembleSignalModel | Compatible? |
|---|---|---|---|
| `train(features_df, labels, calibrate, save_model)` | ✓ | ✓ + `n_wf_folds` param | **Yes** — extra optional param is additive |
| `predict(features: Dict) → PredictionResult` | ✓ | ✓ | **Yes** — identical signature and return shape |
| `predict_batch(features_df) → np.ndarray` | ✓ | ✓ | **Yes** — identical |
| `backtest(features_df, labels) → Dict` | ✓ | ✓ | **Yes** — identical return shape |
| `save(filename: str)` | ✓ | ✓ | **Yes** |
| `load(filename=None) → bool` | ✓ | ✓ | **Partial** — different glob pattern (see §1.3) |
| `get_fallback_stats() → Dict[str, int]` | ✓ | ✓ | **Yes** |

### 1.2 Attribute Compatibility

| Attribute | SignalModel | EnsembleSignalModel | Compatible? |
|---|---|---|---|
| `trained: bool` | ✓ | ✓ | **Yes** |
| `feature_names: List[str] \| None` | ✓ | ✓ | **Yes** |
| `feature_means: np.ndarray \| None` | ✓ | ✓ | **Yes** |
| `feature_stds: np.ndarray \| None` | ✓ | ✓ | **Yes** |
| `fallback_counter: Counter` | ✓ | ✓ | **Yes** |
| `training_stats: Dict` | ✓ | ✓ | **Partial** — different keys (see §1.3) |
| `model` (raw XGBClassifier) | ✓ | ✗ (uses `calibrated_models` dict) | **N/A** — never accessed externally |

### 1.3 Interface Gaps (Breaking Differences)

#### GAP-1: `training_stats["test_auc"]` key is missing

`online_retrain.py:ModelRetrainer._check_performance()` reads:
```python
baseline_auc = model.training_stats.get("test_auc")  # line 353
```

- `SignalModel.training_stats` → has `"test_auc"` key
- `EnsembleSignalModel.training_stats` → has `"ensemble_test_auc"` key, **no `"test_auc"`**

**Effect:** `_check_performance()` returns `None` → the performance-degradation retrain trigger never fires. Silent breakage — no error, just dead monitoring.

---

#### GAP-2: `training_stats["timestamp"]` missing in EnsembleSignalModel

`online_retrain.py:ModelRetrainer._get_model_age_days()` reads:
```python
ts = model.training_stats.get("timestamp")  # line 284
```

- `SignalModel.training_stats` → does **not** include `timestamp` either (it's only in the saved joblib file)
- `EnsembleSignalModel.training_stats` → also doesn't include it

Both fall through to the mtime-based fallback. **However**, the fallback glob (`signal_model_*.joblib`) doesn't match ensemble files:
```python
model_files = sorted(self.model_dir.glob("signal_model_*.joblib"), ...)  # line 288
```
**Effect:** Staleness check uses `None` age → age trigger also never fires for the ensemble.

---

#### GAP-3: Hardcoded `SignalModel()` instantiations in `online_retrain.py`

Two hard-coded construction sites:
```python
# line 156
current_model = SignalModel(model_dir=str(self.model_dir))

# line 198
new_model = SignalModel(model_dir=str(self.model_dir))
```
**Effect:** `ModelRetrainer` always retrains a `SignalModel`, not an `EnsembleSignalModel`. The promoted model on disk is a different class than what's running inference.

---

#### GAP-4: `ModelRetrainer._prune_old_versions()` prunes wrong files

```python
model_files = sorted(self.model_dir.glob("signal_model_*.joblib"), ...)  # line 474
```
**Effect:** Old ensemble model files (`ensemble_model_*.joblib`) are never pruned. Disk fills up indefinitely.

---

#### GAP-5: `ModelRetrainer._save_versioned()` uses wrong filename prefix

```python
filename = f"signal_model_{ts}.joblib"  # line 463
model.save(filename)
```
When `model` is an `EnsembleSignalModel`, it saves a file named `signal_model_YYYYMMDD_HHMMSS.joblib`, but `EnsembleSignalModel.load()` searches for `ensemble_model_*.joblib`:
```python
model_files = list(self.model_dir.glob('ensemble_model_*.joblib'))
```
**Effect:** `EnsembleSignalModel.load()` won't find models saved by `ModelRetrainer`. The promoted model is orphaned.

---

#### GAP-6: `compass/ml_strategy.py:RegimeModelRouter` instantiates `SignalModel` for regime-specific models

```python
# lines 78-79
model = SignalModel(model_dir=os.path.dirname(path))
if model.load(os.path.basename(path)):
```
**Effect:** Regime-specific models loaded via `RegimeModelRouter` are always `SignalModel`. If ensemble files are pointed to, loading fails (wrong `.load()` glob). Duck-typing safe once loaded — the router only calls `.predict()`.

---

#### GAP-7: `ml/regime_model_router.py` loads raw joblib, bypasses SignalModel/EnsembleSignalModel entirely

```python
# ml/regime_model_router.py:84,176-188
model_path = Path(cfg.get("model_path", str(_DEFAULT_MODEL_PATH)))
model = joblib.load(str(path))
```
The default path is `ml/models/signal_model_20260217.joblib`. This loads the raw joblib dict (not a class instance), then calls:
```python
proba = self._model.predict_proba(vec)[0]
```
**Effect:** This bypasses both SignalModel and EnsembleSignalModel entirely. It would need a raw XGBClassifier or sklearn-compatible object at that path — it won't work with an EnsembleSignalModel joblib payload (which is a dict, not a classifier).

---

#### GAP-8: `compass/__init__.py` exports `SignalModel` but not `EnsembleSignalModel`

```python
# line 24
from compass.signal_model import SignalModel
# EnsembleSignalModel is not imported or re-exported
```
**Effect:** Code using `from compass import EnsembleSignalModel` fails at import. Any calling code that does `from compass import SignalModel` and passes it to `MLEnhancedStrategy` will receive the old class.

---

### 1.4 Compatibility Summary

| Call site | File | Compatible? | Gap# |
|---|---|---|---|
| `MLEnhancedStrategy(signal_model=...)` | `compass/ml_strategy.py:124` | **Yes** — only calls `.predict()` | — |
| `RegimeModelRouter.regime_models[r].predict()` | `compass/ml_strategy.py:94,97` | **Yes** — duck-typing | — |
| `RegimeModelRouter` instantiates `SignalModel` | `compass/ml_strategy.py:78` | **Fixed (2026-03-28)** | GAP-6 |
| `ModelRetrainer.check_and_retrain(current_model)` | `compass/online_retrain.py` | **Fixed (prior)** | GAP-1,2,3,4,5 |
| `compass/__init__.py` export | `compass/__init__.py:24` | **Fixed (2026-03-28)** | GAP-8 |
| `ml/regime_model_router.py` raw joblib load | `ml/regime_model_router.py:84` | **Fixed (2026-03-28)** | GAP-7 |

---

## 2. Config Changes Needed

### 2.1 No YAML config changes required for core swap

The `configs/paper_exp503.yaml` config doesn't name the model class — it references `use_signal_model: true` and `model_path: ...` which point to file paths, not Python class names. **No YAML changes needed.**

### 2.2 Model file path update for `ml/regime_model_router.py`

The default model path is hardcoded in `ml/regime_model_router.py:28`:
```python
_DEFAULT_MODEL_PATH = ROOT / "ml" / "models" / "signal_model_20260217.joblib"
```

After training an EnsembleSignalModel, update the config path to point to the new ensemble joblib file:
```yaml
# configs/paper_exp503.yaml
strategy:
  ml_enhanced:
    model_path: "ml/models/ensemble_model_YYYYMMDD.joblib"  # new
```

**But note:** `ml/regime_model_router.py._score_features()` calls `predict_proba()` directly on the raw loaded joblib payload — this only works if the payload is a sklearn-compatible classifier, not the EnsembleSignalModel dict format. See §3.3 for how to fix.

### 2.3 Optional: add `use_ensemble: true` config flag for shadow mode

For the shadow mode migration (§3.1), add a flag to control which model class is used without code changes:
```yaml
strategy:
  ml_enhanced:
    model_class: "ensemble"   # "xgboost" | "ensemble"
```
This is read in the factory to decide which class to instantiate.

---

## 3. Migration Path

### Phase 4A — Shadow Mode (Observe, Don't Act)

**Goal:** Run both models in parallel for 2–4 weeks. Log both predictions. Confirm ensemble predictions are reasonable before flipping the switch. Zero production risk.

**How:**

Add a `ShadowEnsemble` wrapper in a new file `compass/shadow_model.py`:
```python
class ShadowEnsemble:
    """Runs both SignalModel and EnsembleSignalModel; returns SignalModel result."""

    def __init__(self, primary: SignalModel, shadow: EnsembleSignalModel):
        self.primary = primary   # controls actual trades
        self.shadow  = shadow    # logged only, never acts

    def predict(self, features: Dict) -> PredictionResult:
        primary_result = self.primary.predict(features)
        try:
            shadow_result = self.shadow.predict(features)
            log.info(
                "SHADOW | primary_prob=%.3f ensemble_prob=%.3f delta=%.3f",
                primary_result["probability"],
                shadow_result["probability"],
                shadow_result["probability"] - primary_result["probability"],
            )
        except Exception as exc:
            log.warning("SHADOW prediction failed: %s", exc)
        return primary_result  # always returns primary

    # Forward all other methods to primary unchanged
    def predict_batch(self, df): return self.primary.predict_batch(df)
    def train(self, *a, **kw): return self.primary.train(*a, **kw)
    def save(self, *a, **kw): return self.primary.save(*a, **kw)
    def load(self, *a, **kw): return self.primary.load(*a, **kw)
    def get_fallback_stats(self): return self.primary.get_fallback_stats()
    @property
    def trained(self): return self.primary.trained
    @property
    def feature_names(self): return self.primary.feature_names
    @property
    def training_stats(self): return self.primary.training_stats
    @property
    def feature_means(self): return self.primary.feature_means
    @property
    def feature_stds(self): return self.primary.feature_stds
```

`ShadowEnsemble` is fully compatible with `MLEnhancedStrategy` (same interface as `SignalModel`) and introduces **zero production risk** — all signals still come from `SignalModel`.

**Shadow mode activation** — one wiring point in the scheduler/main:
```python
signal_model = SignalModel(...)
signal_model.load()
ensemble = EnsembleSignalModel(...)
ensemble.train(features_df, labels)
signal_model = ShadowEnsemble(signal_model, ensemble)  # drop-in
```

**Observe for 2–4 weeks.** Monitor log lines with `SHADOW |` for:
- Mean absolute probability delta between primary and shadow
- Fold-alignment: does ensemble flip signals more often during bear/high_vol regimes?
- Fallback rate for ensemble (should be 0 if training succeeded)

---

### Phase 4B — Hard Swap (Ensemble Takes Control)

**Pre-conditions:**
- Shadow mode ran for ≥ 2 weeks with no ensemble fallbacks
- Mean |Δ probability| < 0.15 (no wildly divergent predictions)
- Ensemble Brier Score in live shadow data is within 0.02 of SignalModel

**Steps:**

#### Step 1: Fix `compass/online_retrain.py` (5 lines, 4 gaps)

```python
# Change line 23: import
from compass.signal_model import SignalModel
# → add:
from compass.ensemble_signal_model import EnsembleSignalModel

# Change line 156 (instantiation):
current_model = SignalModel(model_dir=str(self.model_dir))
# →
current_model = EnsembleSignalModel(model_dir=str(self.model_dir))

# Change line 198 (instantiation):
new_model = SignalModel(model_dir=str(self.model_dir))
# →
new_model = EnsembleSignalModel(model_dir=str(self.model_dir))

# Change line 353 (training_stats key):
baseline_auc = model.training_stats.get("test_auc")
# →
baseline_auc = model.training_stats.get("ensemble_test_auc") \
    or model.training_stats.get("test_auc")  # backward-compat during transition

# Change line 288 (age fallback glob):
model_files = sorted(self.model_dir.glob("signal_model_*.joblib"), ...)
# →
model_files = sorted(
    list(self.model_dir.glob("ensemble_model_*.joblib")) or
    list(self.model_dir.glob("signal_model_*.joblib")), ...
)

# Change line 463 (_save_versioned filename):
filename = f"signal_model_{ts}.joblib"
# →
filename = f"ensemble_model_{ts}.joblib"

# Change line 474 (_prune_old_versions glob):
model_files = sorted(self.model_dir.glob("signal_model_*.joblib"), ...)
# →
model_files = sorted(self.model_dir.glob("ensemble_model_*.joblib"), ...)
```

#### Step 2: Fix `compass/ml_strategy.py:RegimeModelRouter` (1 line)

```python
# line 78: change SignalModel → EnsembleSignalModel
model = SignalModel(model_dir=os.path.dirname(path))
# →
model = EnsembleSignalModel(model_dir=os.path.dirname(path))
```
Also update the import at the top of `ml_strategy.py`:
```python
from compass.signal_model import SignalModel
# → add:
from compass.ensemble_signal_model import EnsembleSignalModel
```

#### Step 3: Fix `compass/__init__.py` (2 lines)

```python
# After line 24:
from compass.signal_model import SignalModel
# Add:
from compass.ensemble_signal_model import EnsembleSignalModel

# In __all__ list, add:
"EnsembleSignalModel",
```

#### Step 4: Fix `ml/regime_model_router.py` (requires architectural decision)

`_score_features()` uses raw `predict_proba()` on a loaded joblib payload. This assumes the payload **is** a sklearn classifier. But `EnsembleSignalModel.save()` writes a dict:
```python
model_data = {
    'calibrated_models': ...,   # dict of name → CalibratedClassifierCV
    'ensemble_weights': ...,
    ...
}
```

Two options:

**Option A (minimal):** Keep `ml/regime_model_router.py` using the existing `signal_model_*.joblib` file. Don't change it. The regime router uses a raw XGBoost model for blending, which still works. The full ensemble only runs through `compass/ml_strategy.py`.

**Option B (full):** Refactor `_score_features()` to accept an `EnsembleSignalModel` instance:
```python
# In __init__:
from compass.ensemble_signal_model import EnsembleSignalModel
self._model = EnsembleSignalModel(model_dir=...)
self._model.load()

# In _score_features():
if isinstance(self._model, EnsembleSignalModel):
    result = self._model.predict(features)
    return result["probability"]
```

**Recommendation: Option A for Phase 4B** — keep the regime router on the old XGBoost model. It's a blending signal (25% weight) and is separate from the primary ensemble path. Upgrade it in a Phase 4C if needed.

#### Step 5: Train and deploy the ensemble model

```python
# One-time training (run as a script before cutover):
import pandas as pd
from compass.feature_pipeline import FeaturePipeline
from compass.ensemble_signal_model import EnsembleSignalModel

df = pd.read_csv("compass/training_data_combined.csv")
pipeline = FeaturePipeline()
features = pipeline.transform(df)
labels = df["win"].values

model = EnsembleSignalModel(model_dir="ml/models")
model.train(features, labels, calibrate=True, save_model=True)
# → writes ml/models/ensemble_model_YYYYMMDD.joblib
```

#### Step 6: Update call sites in live production entry points

Grep for all locations that construct `SignalModel(...)` and replace with `EnsembleSignalModel(...)`:
```
scripts/dryrun_exp503.py
scripts/run_x003_combined.py
scripts/exp601_ml_signal_filter.py
scripts/ml_walkforward_train.py
scripts/backtest_ml_filter.py
```
These are scripts, not live production paths. Audit each one; most can be migrated on the next rerun.

---

## 4. Risk Assessment

### 4.1 What Could Break

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| `online_retrain.py` silently uses `SignalModel` for retraining while serving `EnsembleSignalModel` inference | **HIGH** | High if GAP-3 not fixed | Fix §3.2 Step 1 before swap |
| Performance retrain trigger never fires (GAP-1, silent) | **MEDIUM** | Certain | Fix `training_stats` key; add integration test |
| Staleness trigger never fires (GAP-2, silent) | **MEDIUM** | Certain | Fix glob pattern in `_get_model_age_days` |
| Old ensemble model files accumulate on disk (GAP-4) | **LOW** | Certain | Fix `_prune_old_versions` glob |
| `RegimeModelRouter` in `ml_strategy.py` fails to load ensemble regime models | **MEDIUM** | Only if regime-specific models configured | Fix §3.2 Step 2 |
| `ml/regime_model_router.py` raw joblib fails on ensemble payload | **LOW** | Only if model_path updated | Use Option A (keep old XGB model) |
| `compass.SignalModel` imported by external scripts breaks | **LOW** | Low — most scripts import directly | Fix `__init__.py` export; keep `SignalModel` exported too |
| Prediction latency increase | **LOW** | Certain — ensemble is 27× slower per trade | Negligible at weekly retraining; ~6ms per batch vs 0.24ms |
| First-fold calibration instability (EnsembleSignalModel warns "too few samples") | **MEDIUM** | On small datasets | Already observed in benchmark; not a blocking risk |
| Feature set mismatch if FeaturePipeline not applied at inference | **HIGH** | If `_build_features_for_signal` returns raw features | Ensure FeaturePipeline runs at inference (see §4.3) |

### 4.2 Critical: Feature Set Must Match Between Training and Inference

The biggest latent risk is a training/inference feature mismatch.

**Training** (benchmark): `FeaturePipeline.transform()` → 31 pipeline features (`vix_zscore`, `credit_to_width`, etc.)

**Inference** (`compass/ml_strategy.py:_build_features_for_signal`): Builds a **raw** feature dict from `MarketSnapshot` + `FeatureEngine.build_features()`:
```python
features['vix_level'] = market_data.vix        # raw VIX, not z-scored!
features['iv_rank'] = market_data.iv_rank       # OK
features['rsi_14'] = market_data.rsi.get(...)   # OK
```

If the model is trained on `vix_zscore` but served `vix_level`, predictions are garbage. The model silently fills missing features with 0.0 via `_features_to_array()`.

**Mitigation:** Either:
- **Option A:** Keep training on raw features (drop `FeaturePipeline`), matching existing inference feature set
- **Option B (preferred):** Apply `FeaturePipeline` at inference — rewrite `_build_features_for_signal` to return pipeline-normalized features, or wrap `FeatureEngine` output with `FeaturePipeline.transform()`

This decision must be made **before** training the production ensemble. The benchmark already validated Option B (pipeline features → better AUC). Option B requires `_build_features_for_signal` to build a single-row DataFrame and run it through `FeaturePipeline.transform()`.

### 4.3 Rollback Plan

1. Shadow mode (Phase 4A) allows immediate rollback by removing the `ShadowEnsemble` wrapper — zero signal impact.
2. After Phase 4B swap: rollback by reinstating `SignalModel` import and loading the old `.joblib` file. Keep the last 3 `signal_model_*.joblib` files (existing `ModelRetrainer.keep_versions=3` policy).
3. If `ModelRetrainer` is already retrained with ensemble, the last `signal_model_*.joblib` (pre-swap) is still on disk from the old retention policy.

---

## 5. File Change Summary

| File | Change | Scope |
|---|---|---|
| `compass/online_retrain.py` | Replace 2 `SignalModel()` instantiations; fix 3 globs; fix `test_auc` key | ~10 lines |
| `compass/ml_strategy.py` | Replace `SignalModel(...)` in `RegimeModelRouter.__init__`; add import | ~3 lines |
| `compass/__init__.py` | Add `EnsembleSignalModel` import + `__all__` entry | 2 lines |
| `compass/shadow_model.py` | **New file** — `ShadowEnsemble` wrapper (Phase 4A only) | ~40 lines |
| `ml/regime_model_router.py` | Option A: no change. Option B: refactor `_score_features` | 0–15 lines |
| Scripts in `scripts/` | Replace `SignalModel(...)` construction | 1 line each |
| `configs/paper_exp503.yaml` | Update `model_path` if using ensemble in regime router | 1 line |

**No changes needed to:**
- `compass/walk_forward.py` — works with any sklearn-compatible model
- `compass/feature_pipeline.py` — stateless, no model dependency
- `strategies/ml_enhanced_strategy.py` — uses `ml/regime_model_router.py`, not `SignalModel` directly
- `shared/types.py` — `PredictionResult` shape matches both models
- Test files — tests already isolated per model class

---

## 6. Recommended Sequence

```
Week 1:    Deploy ShadowEnsemble (Phase 4A)
           → Train ensemble on combined dataset
           → Log SHADOW predictions for 2 weeks

Week 2-3:  Monitor shadow logs
           → Confirm fallback_rate = 0
           → Confirm mean |Δprob| < 0.15
           → Confirm no inference-time KeyErrors (feature mismatch)

Week 3-4:  Implement Phase 4B fixes [COMPLETED 2026-03-28]
           → Fixed online_retrain.py (GAP-1..5) — _model_file_prefix, model_class param
           → Fixed ml_strategy.py:RegimeModelRouter (GAP-6) — _load_model_from_path helper
           → Fixed compass/__init__.py (GAP-8) — EnsembleSignalModel now exported
           → Fixed ml/regime_model_router.py (GAP-7) — delegates to EnsembleSignalModel
           → Fixed compass/signal_model.py cv='prefit' sklearn 1.6+ compat
           → Tests: tests/test_ensemble_integration.py (19 tests, all pass)

Week 4:    Hard swap
           → Remove ShadowEnsemble
           → EnsembleSignalModel is primary
           → Run for 1 week with heightened monitoring

Week 5+:   Normal operations
           → ModelRetrainer now retrains ensemble
           → Confirm retrain trigger fires correctly (check logs for "auc_drop")
```
