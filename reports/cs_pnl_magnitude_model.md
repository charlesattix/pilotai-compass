# CS Trade Magnitude Prediction — XGBoost Regressor Analysis

**Date:** 2026-03-25
**Approach:** Predict continuous trade P&L as % of max risk (not binary win/loss)
**Model:** XGBoost Regressor, walk-forward validation (expanding window by year)
**Dataset:** 233 CS trades from EXP-400+401 combined, 2020-2025

---

## Motivation

The binary classifier (win/loss) achieves AUC ~0.50-0.65 on CS trades because the base win rate is 84% — there simply aren't enough losses for the model to learn meaningful loss patterns. A magnitude model predicting *how much* a trade returns (continuous target) could be more useful for position sizing: bet more on trades the model predicts will return 30%, less on those predicted at 5%.

**Target variable:** `return_pct / 100` = actual PnL / max risk at entry. Range: -1.0 (full loss) to +0.55 (best win). Mean: +10.1%.

---

## Walk-Forward Results

| Fold | Train | Test | n_train | n_test | R² | MAE | Corr | Top-Q Return | Bot-Q Return | Spread |
|------|-------|------|---------|--------|-----|-----|------|-------------|-------------|--------|
| 1 | 2020 | 2021 | 28 | 70 | -0.93 | 19.0% | N/A | +18.0% | +18.0% | 0.0pp |
| 2 | 2020-21 | 2022 | 98 | 20 | -0.19 | 30.7% | 0.07 | +0.9% | -4.0% | +4.9pp |
| 3 | 2020-22 | 2023 | 118 | 38 | +0.05 | 7.7% | 0.56 | +9.3% | +6.3% | +3.0pp |
| 4 | 2020-23 | 2024 | 156 | 39 | -0.05 | 9.1% | 0.06 | +6.1% | +5.9% | +0.2pp |
| 5 | 2020-24 | 2025 | 195 | 38 | -0.00 | 12.6% | 0.06 | +9.6% | +7.0% | +2.6pp |

### Aggregate Out-of-Sample (205 trades)

| Metric | Value |
|--------|-------|
| R² | **-0.14** (worse than predicting the mean) |
| MAE | 14.97% of max risk |
| Correlation | -0.20 |
| Top-quartile actual return | +6.90% |
| Bottom-quartile actual return | **+18.40%** |
| Quartile spread | **-11.50pp** (inverted!) |

---

## Key Finding: The Model Fails — And the Failure Is Informative

**The quartile spread is inverted.** Trades the model predicts will be the *worst* actually return +18.4% on average (90.5% win rate), while trades predicted to be the *best* return only +6.9% (80.6% win rate). The model's confidence is anti-correlated with actual outcomes.

### Quintile Breakdown (OOS)

| Quintile | N | Predicted | Actual | Win Rate |
|----------|---|-----------|--------|----------|
| Q1 (predicted worst) | 74 | +3.4% | **+18.4%** | **90.5%** |
| Q2 | 31 | +9.0% | +5.1% | 83.9% |
| Q3 | 37 | +9.5% | +5.5% | 81.1% |
| Q4 | 34 | +12.0% | +6.9% | 88.2% |
| Q5 (predicted best) | 29 | +14.8% | +7.6% | 72.4% |

The Q1-labeled trades (what the model thinks will underperform) are actually the best performers. This happens because fold 1 (train on 28 trades from 2020, test on 70 trades in 2021) has a severe distribution shift — 2021 was a historically strong year (90% WR, +18% avg return) that looks nothing like 2020's mix of COVID volatility and recovery. The model learned "high VIX = bad" from 2020, but 2021's calmer, trending market rewarded exactly the trades that looked risky by 2020 standards.

### Position Sizing Simulation

| Strategy | Total Return |
|----------|-------------|
| Uniform sizing | +2,175.9% |
| ML-sized (1.5x top half, 0.5x bottom half) | +1,946.8% |
| **Difference** | **-229.1pp (worse)** |

ML-based sizing actively destroys value because the magnitude predictions are anti-correlated with outcomes.

---

## Why This Model Cannot Work (With 233 Trades)

### 1. Insufficient data for regression

Binary classification on 233 trades with 84% positive rate is already marginal. Continuous regression requires learning the *magnitude* of returns — a strictly harder problem. With 28 training samples in fold 1 and 20 test samples in fold 2, the model has no statistical power to learn return magnitudes.

### 2. Non-stationary target distribution

CS return distributions shift dramatically between years:

- 2020: mean +6.0%, many moderate wins (post-COVID recovery)
- 2021: mean +18.1%, very few losses (strong bull market)
- 2022: mean -2.0%, many losses (bear market)
- 2023-2025: mean +6-9%, stable but different from 2020-2021

A model trained on 2020-2021 learns a bimodal distribution that doesn't generalize to 2022's bear market or 2023-2025's moderate returns.

### 3. Feature importance reveals leakage concern

The top 4 features by importance (all other features have 0.0 importance):

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | max_loss_per_unit | 0.2998 |
| 2 | net_credit | 0.2432 |
| 3 | hold_days | 0.2290 |
| 4 | ma50_slope_ann_pct | 0.2281 |

`hold_days` is partially forward-looking — you don't know how long a trade will be held at entry time. It's computed from `exit_date - entry_date` in the training data. **This feature leaks information and should be removed from any production model.** The fact that only 4 features have non-zero importance (with strong regularization) confirms the model is starved for signal.

### 4. High base rate renders sorting useless

When 84% of trades are winners averaging +13% and only 16% are losers averaging -25%, the variance within the "win" category is small (~5-20% returns). The model would need to distinguish between a +8% win and a +18% win — a signal that's buried in market noise at this sample size.

---

## Comparison: Magnitude vs Binary Classification

| Metric | Binary Classifier (AUC) | Magnitude Regressor (R²) |
|--------|------------------------|-------------------------|
| OOS performance | 0.82 (useful) | -0.14 (useless) |
| Quartile spread | +1.6pp (correct direction) | -11.5pp (inverted) |
| Sizing improvement | +1-2% (modest) | -229pp (destructive) |

**The binary classifier is the better approach** despite its modest AUC. It correctly identifies the small number of likely losers (16% of trades), which is what matters for a strategy with an 84% base win rate. The magnitude model tries to solve a harder problem without enough data.

---

## Recommendations

1. **Do not deploy magnitude-based sizing for CS trades.** The inverted quartile spread means it would actively reduce returns.

2. **Stick with the binary classifier for signal filtering.** Use the existing ensemble model (AUC 0.84) to filter out likely losers rather than trying to predict exact return magnitudes.

3. **If magnitude prediction is revisited**, it requires: (a) 1000+ CS trades minimum, (b) `hold_days` removed from features, (c) quantile regression instead of MSE (to handle the skewed target distribution), and (d) separate models per regime to handle non-stationarity.

4. **For position sizing, use simpler heuristics** that the data actually supports: size by credit_to_width_ratio (higher credit = more edge = bigger position) and reduce size near events (event_risk_score). These don't need ML.
