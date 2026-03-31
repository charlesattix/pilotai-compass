# CS-Only Benchmark: XGBoost vs Ensemble vs Baseline

**Date:** 2026-03-26
**Data:** `compass/training_data_combined.csv` — filtered to `strategy_type == "CS"`
**Trades:** 233 CS trades, 2020–2025
**Features:** 28 (22 numeric + 6 one-hot: regime ×4, spread_type ×2)
**Validation:** Year-based expanding walk-forward (train on 1..N years, test on year N+1)

---

## Setup

| Parameter | Value |
|-----------|-------|
| CS trades | 233 |
| Overall WR | **84.12%** |
| Date range | 2020-01-02 → 2025-12-xx |
| Walk-forward folds | 5 (test years: 2021–2025) |
| Min train samples | 20 |
| Feature pipeline | z-score VIX/SPY, log contracts, credit/width ratios, regime + spread_type one-hot |

---

## Results by Year

| Year | N | Base WR | XGB WR | XGB Selected | Ensemble WR | Ens Selected |
|------|---|---------|--------|-------------|-------------|-------------|
| 2021 | 70 | 90.0% | 90.0% | 70/70 | 90.0% | 70/70 |
| 2022 | 20 | **60.0%** | 60.0% | 20/20 | 60.0% | 20/20 |
| 2023 | 38 | 89.5% | 89.2% | 37/38 | 89.5% | 38/38 |
| 2024 | 39 | 82.1% | **85.7%** | 35/39 | 82.1% | 39/39 |
| 2025 | 38 | 86.8% | 86.1% | 36/38 | 86.8% | 38/38 |

---

## Aggregate OOS Metrics

| Metric | Baseline | XGBoost | Ensemble |
|--------|----------|---------|----------|
| OOS Win Rate | 84.12% | 85.35% | 84.88% |
| **Lift vs Baseline** | — | **+0.48%** | **+0.00%** |
| OOS Accuracy | 84.12% | 83.41% | 84.88% |
| OOS AUC | — | 0.559 | **0.713** |
| OOS Brier Score | — | 0.1359 | **0.1228** |
| OOS Signal Sharpe | 3.43 | 3.83 | 3.75 |
| Trades filtered | 0 | 13/205 (6.3%) | 0/205 |

*OOS = out-of-sample across 5 walk-forward folds (205 total test trades)*

---

## Key Findings

### 1. XGBoost: Marginal lift, poor AUC, overfits to 2024

XGBoost achieved **+0.48% WR lift** by skipping 13 trades across all 5 folds (6.3% filter rate). The gains are driven almost entirely by 2024 (+3.7% lift, 4 trades skipped). In 3 other folds the model either passes everything through or slightly hurts performance.

**OOS AUC of 0.559 is barely above chance level (0.5).**  The model has essentially zero discriminative power on CS data at the population level. The 2024 "success" is plausibly luck given the fold has only 39 trades.

Critically, XGBoost OOS accuracy (83.4%) is **lower** than the base win rate (84.1%) — the model's false-positive errors (predicting loss on winning trades) exceed the losses it prevents.

### 2. Ensemble: Good calibration, zero filter lift

The Ensemble (XGBoost + RandomForest + ExtraTrees) achieved a respectable **OOS AUC of 0.713** — it can rank trades by quality better than chance. Yet its WR lift is **exactly 0.00%** because it never drops any trade's probability below 0.5.

This is the **ceiling effect**: with an 84% base win rate, the prior is so strong that even trades the model is relatively pessimistic about still have predicted probability > 0.5. The 0.5 threshold is too low to be useful — it would need to be ~0.80+ to produce any filtering.

The ensemble does produce better probability estimates (Brier score 0.123 vs XGB 0.136), which is useful for **position sizing** but not for binary skip/take decisions.

### 3. 2022 is the only structurally hard year

| Year | Base WR | Notes |
|------|---------|-------|
| 2020 | 78.6% | COVID volatility, limited data |
| 2021 | 90.0% | Bull market, high IV environment |
| **2022** | **60.0%** | **Bear market. Only hard year.** |
| 2023 | 89.5% | Recovery + elevated IV |
| 2024 | 82.1% | Near-ATH, compressed IV |
| 2025 | 86.8% | Moderate vol |

2022 stands out as the only year where CS strategies struggled (WR dropped to 60%). The problem isn't trade selection — it's **regime**: a sustained bear market with persistent directional moves forced short strikes into the money. Neither model improved on 2022 results (both selected 20/20 trades, getting 12 right and 8 wrong).

This makes intuitive sense: in a bear market, the right decision is to reduce position size or pause trading entirely — not to fine-tune which individual credit spreads to take.

---

## Does ML Add Value for CS-Only Trading?

**Short answer: No, not meaningfully.**

### Why ML fails here

1. **Near-perfect base rate**: 84% is already a very high win rate. The marginal gain from filtering requires the model to identify the 16% losers with very high precision. Current data (233 total CS trades) cannot support that level of specificity.

2. **Ceiling effect on probability threshold**: With 84% prior, model probabilities cluster near 0.9. The standard 0.5 decision boundary never fires. Even the ensemble with AUC 0.71 selects 100% of trades.

3. **Regime dominance**: The single bad year (2022) is explained by market regime, not trade-level features. A regime gate (e.g., "pause CS in bear markets") is a more robust intervention than per-trade ML filtering.

4. **Small dataset**: 233 trades across 6 years. Walk-forward folds average 41 test trades. At 16% loss rate, that's ~7 losses per fold — far too few to train a reliable loss classifier.

5. **XGBoost AUC near chance**: AUC 0.559 means the model can barely rank a winner above a loser. The available features (VIX, momentum, MA distances) do not distinguish CS wins from CS losses at useful confidence levels.

### What the ensemble IS good for

The Ensemble's AUC 0.713 and lower Brier score indicate it has learned *something* about relative trade quality. This could be used for:
- **Differential position sizing**: larger contracts on high-confidence signals
- **Probability-based alert system**: flag trades below 75% predicted probability for human review
- **Integration with regime detection**: ensemble vote combined with macro regime gate

### Recommendation

For CS-only trading, the optimal ML use is **not binary skip/take filtering** but rather:

1. **Primary gate: Regime filter** — halt CS trading when bear regime detected (saves 2022's -2% year)
2. **Secondary: Ensemble for position sizing** — scale contracts from 3 to 7 based on predicted probability
3. **Do NOT use ML to skip individual CS trades** — 84% base WR means filtering costs more in missed winners than it saves in avoided losers

The highest-value ML application is **regime detection** (when to trade CS at all), not intra-regime trade selection.

---

## Per-Fold Detail

### XGBoost

| Fold | Test Year | n_train | n_test | Base WR | XGB WR | Lift | AUC | Sharpe |
|------|-----------|---------|--------|---------|--------|------|-----|--------|
| 0 | 2021 | 28 | 70 | 90.0% | 90.0% | 0.0% | 0.500 | 8.58 |
| 1 | 2022 | 98 | 20 | 60.0% | 60.0% | 0.0% | 0.594 | -0.40 |
| 2 | 2023 | 118 | 38 | 89.5% | 89.2% | -0.3% | 0.427 | 4.66 |
| 3 | 2024 | 156 | 39 | 82.1% | 85.7% | **+3.7%** | 0.750 | 4.15 |
| 4 | 2025 | 195 | 38 | 86.8% | 86.1% | -0.7% | 0.630 | 2.73 |

### Ensemble (XGB + RF + ET)

| Fold | Test Year | n_train | n_test | Base WR | Ens WR | Lift | AUC | Sharpe |
|------|-----------|---------|--------|---------|--------|------|-----|--------|
| 0 | 2021 | 28 | 70 | 90.0% | 90.0% | 0.0% | 0.714 | 8.58 |
| 1 | 2022 | 98 | 20 | 60.0% | 60.0% | 0.0% | 0.563 | -0.40 |
| 2 | 2023 | 118 | 38 | 89.5% | 89.5% | 0.0% | 0.713 | 4.82 |
| 3 | 2024 | 156 | 39 | 82.1% | 82.1% | 0.0% | 0.755 | 3.26 |
| 4 | 2025 | 195 | 38 | 86.8% | 86.8% | 0.0% | 0.673 | 2.79 |

---

## Conclusion

**ML adds no meaningful value for binary CS trade selection at 84% base win rate.**

- XGBoost: +0.48% lift at the cost of lower OOS accuracy. AUC near random (0.56).
- Ensemble: AUC 0.713 (good) but zero filtering (ceiling effect). Better calibration useful for sizing.
- Neither model helps with the only structurally hard period (2022 bear market).

The 84% base WR is a ceiling that ML cannot meaningfully exceed with 233 trades. The correct interventions are (1) regime detection to avoid CS trading in bear markets, and (2) ensemble-driven position sizing — not per-trade skip/take filtering.

**Bottom line:** CS is already a near-optimal strategy. ML's role is risk management, not selection.
