# Ensemble vs XGBoost Benchmark — Combined Dataset

**Date:** 2026-03-26
**Dataset:** `compass/training_data_combined.csv`
**Trades:** 428 (2020–2025, mixed CS/IC/SS strategy types)
**Feature pipeline:** `FeaturePipeline.transform()` — stationary, normalized features
**Validation:** Walk-forward (expanding-window, year-by-year chronological splits, 5 folds)

---

## Setup

### Models

| Model | Description |
|---|---|
| **XGBoost** | Standalone `XGBClassifier` with same hyperparameters as `SignalModel.train()` (depth=6, lr=0.05, 200 estimators, L1/L2 regularization) |
| **EnsembleSignalModel** | Weighted average of XGBoost + RandomForest + ExtraTrees; per-model weights from internal walk-forward AUC; each base learner individually calibrated via `CalibratedClassifierCV(sigmoid)` |

### Feature Pipeline

`FeaturePipeline.transform()` produced **31 features** (versus 28 raw numeric + 3 one-hot in the baseline `WalkForwardValidator`):

| Category | Features |
|---|---|
| Calendar | `days_since_last_trade` |
| Momentum | `rsi_14`, `momentum_5d_pct`, `momentum_10d_pct` |
| Volatility | `vix_zscore`, `vix_change_5d_pct`, `vix_percentile_50d`, `vix_percentile_100d`, `iv_rank` |
| Price structure | `spy_price_zscore`, `dist_from_ma20/50/80/200_pct`, `ma50_slope_ann_pct` |
| Realized vol | `realized_vol_atr20`, `realized_vol_20d` |
| Trade structure | `credit_to_width`, `loss_to_width`, `contracts_log` |
| Regime (one-hot) | `regime_bear`, `regime_bull`, `regime_crash`, `regime_high_vol`, `regime_low_vol` |
| Strategy (one-hot) | `strategy_type_CS`, `strategy_type_IC`, `strategy_type_SS` |
| Spread type (one-hot) | `spread_type_bear_call`, `spread_type_bull_put`, `spread_type_unknown` |

Key normalizations vs raw features: `spy_price` → z-score, `vix` → z-score, `contracts` → log(1+x), `net_credit`/`max_loss_per_unit` → ratios w.r.t. `spread_width`, `short_strike` dropped.

### Validation Protocol

```
Fold 0: train [2020]       → test [2021]   (60 train, 101 test)
Fold 1: train [2020–2021]  → test [2022]  (161 train,  59 test)
Fold 2: train [2020–2022]  → test [2023]  (220 train,  69 test)
Fold 3: train [2020–2023]  → test [2024]  (289 train,  69 test)
Fold 4: train [2020–2024]  → test [2025]  (358 train,  70 test)
Total OOS: 368 trades
```

---

## Results

### Aggregate Metrics (mean ± std across 5 folds)

| Metric | XGBoost | EnsembleSignalModel | Δ (Ens − XGB) | Winner |
|---|---|---|---|---|
| **Accuracy** | 0.7645 ± 0.0781 | 0.7924 ± 0.0575 | **+0.0279** | **Ensemble** |
| **Precision** | 0.8024 ± 0.1215 | 0.8146 ± 0.1203 | +0.0122 | **Ensemble** |
| **Recall** | 0.6966 ± 0.2473 | 0.8072 ± 0.0539 | **+0.1106** | **Ensemble** |
| **Brier Score** ↓ | 0.1682 ± 0.0374 | 0.1753 ± 0.0395 | +0.0071 | **XGBoost** |
| **AUC** | 0.8025 ± 0.0751 | 0.8277 ± 0.0708 | **+0.0252** | **Ensemble** |
| **Signal Sharpe** | 3.3211 ± 3.6684 | 2.8862 ± 4.1876 | −0.4349 | **XGBoost** |

*Brier Score: lower is better. All other metrics: higher is better.*

### Per-Fold AUC Detail

| Fold | Test Period | N-train | N-test | XGB AUC | Ens AUC | Δ AUC |
|---|---|---|---|---|---|---|
| 0 | 2021-01-11 → 2021-12-27 | 60 | 101 | 0.8149 | 0.8659 | **+0.0510** |
| 1 | 2022-01-05 → 2022-12-21 | 161 | 59 | 0.6733 | 0.7053 | **+0.0320** |
| 2 | 2023-01-04 → 2023-12-27 | 220 | 69 | 0.8519 | 0.8650 | +0.0131 |
| 3 | 2024-01-03 → 2024-12-23 | 289 | 69 | 0.8140 | 0.8274 | +0.0134 |
| 4 | 2025-01-02 → 2025-12-26 | 358 | 70 | 0.8583 | 0.8750 | +0.0167 |

**Ensemble beats XGBoost on AUC in every single fold** (Δ range: +0.013 to +0.051).

---

## Verdict

| Criterion | Winner | Margin |
|---|---|---|
| **AUC** | **Ensemble** | +0.0252 (avg), consistent across all 5 folds |
| **Brier Score** (calibration) | **XGBoost** | +0.0071 (XGB lower = better) — marginal |
| **Signal Sharpe** | **XGBoost** | −0.4349 — but high variance (std > mean for both) |

### Primary metric (AUC): **Ensemble wins**

Ensemble achieves mean OOS AUC of **0.8277 vs 0.8025** for XGBoost — a +0.025 improvement that is consistent across all five test years, including the challenging 2022 bear market fold (0.705 vs 0.673).

### Secondary metrics: Mixed

- **Accuracy (+2.8%) and Recall (+11.1%)**: Ensemble clearly better. The recall improvement is especially important for credit spread strategies — the ensemble is more willing to commit to predicted winners.
- **Brier Score**: XGBoost wins narrowly (0.1682 vs 0.1753). The ensemble's probability calibration is slightly worse, which is notable given it explicitly uses `CalibratedClassifierCV`. This may reflect the ensemble averaging across calibrated models trained on different data subsets within each fold, introducing slight miscalibration.
- **Signal Sharpe**: XGBoost wins, but the metric is noisy — std >100% of mean for both models — due to a very large Fold 0 spike (8.6 vs 8.6) followed by near-zero and negative values. Not a reliable discriminator at this sample size.

---

## Analysis

### Why Ensemble Beats XGBoost on AUC

The AUC improvement comes from **variance reduction via averaging**. The ensemble's lower standard deviation across folds (±0.071 vs ±0.075 for XGB) reflects more stable OOS performance. The RandomForest and ExtraTrees base learners complement XGBoost by:

1. **Capturing different signal patterns**: RF/ET use random feature subsets (`max_features='sqrt'`), reducing correlation between learners.
2. **Bear market robustness**: The largest AUC gain is in Fold 1 (2022 bear market, +0.032). RF and ET are more robust to distribution shifts because they don't rely on gradient boosting's sequential residual fitting.
3. **Recall recovery**: XGBoost's Recall in 2022 was 0.261 (very conservative); Ensemble maintained 0.810 recall across all folds (std ±0.054 vs XGB's ±0.247).

### Why XGBoost Wins on Brier Score

The ensemble's Brier Score disadvantage (+0.007) is small but real. Two likely causes:

1. **Nested calibration limitation**: Within each outer fold, `EnsembleSignalModel.train()` performs an *internal* train/test split to calibrate base models. With fold sizes of 60–358 samples, this leaves ~40–285 samples for fitting base models — small for reliable calibration. XGBoost's direct probability output is better calibrated at these sample sizes.
2. **Equal weights forced for small folds**: Folds 0 and 1 triggered `"Too few samples per fold ... using equal weights"` — the ensemble can't derive per-model AUC weights when fold sizes are <30. This prevents optimal weighting in early folds.

### Data Distribution Notes

The `EnsembleSignalModel`'s feature drift detector flagged expected regime/strategy OOS distribution shifts (e.g., `regime_bear` 8.9σ from 2020-bull training mean in 2022 test folds, `strategy_type_IC` 7.3σ). These are structural shifts — the features are working correctly by encoding regime changes.

### Elapsed Time

| Model | Time |
|---|---|
| XGBoost (5 folds) | 0.24s |
| Ensemble (5 folds) | 6.56s |

The ensemble is ~27× slower due to training 3 base models + calibration per fold. Still fast enough for weekly retraining cycles.

---

## Recommendations

1. **Use Ensemble for AUC-sensitive decisions** (e.g., regime-aware signal filtering, trade selection): +2.5% AUC with consistent improvement across all years.

2. **Prefer XGBoost when probability calibration matters** (e.g., position sizing based on win probability, Kelly criterion): marginally better Brier Score.

3. **Address recall instability in XGBoost**: XGBoost's 2022 recall collapsed to 0.261 (vs Ensemble's 0.810). For a credit spread strategy, this means XGBoost would have passed on ~74% of potentially profitable 2022 trades. Consider adding a calibrated threshold tuned per-regime.

4. **More data needed for reliable Signal Sharpe**: The std/mean ratio >100% makes Sharpe comparison unreliable at 59–101 test samples per fold. At 200+ samples per fold, Signal Sharpe would be a more meaningful discriminator.

5. **Investigate Brier Score gap**: Run a targeted ablation with a larger calibration set (if data allows) to close the 0.007 Brier gap while preserving the AUC gain.

---

## Files

| File | Description |
|---|---|
| `compass/benchmark_ensemble_vs_xgboost.py` | Benchmark script |
| `compass/benchmark_results_combined.json` | Full per-fold and aggregate results (JSON) |
| `compass/benchmark_ensemble_vs_xgboost.md` | This report |
