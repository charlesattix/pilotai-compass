# Training Data Quality Report

**Dataset:** `compass/training_data_combined.csv`
**Generated:** 2026-03-21
**Source:** EXP-400 (CS+IC champion) + EXP-401 (CS+SS regime-adaptive), deduplicated

---

## 1. Overview

| Metric | Value |
|--------|-------|
| Total rows | 430 |
| Total columns | 39 |
| Date range | 2020-01-02 to 2025-12-26 |
| Win rate | 58.4% (251 W / 179 L) |
| Strategies | CS=236, SS=181, IC=13 |

## 2. Rows Per Year

| Year | Trades | % of Total |
|------|--------|-----------|
| 2020 | 60 | 14.0% |
| 2021 | 100 | 23.3% |
| 2022 | 59 | 13.7% |
| 2023 | 73 | 17.0% |
| 2024 | 69 | 16.0% |
| 2025 | 69 | 16.0% |

All 6 years represented. 2021 has the most trades (bull market, high signal frequency).
2022 (bear) has fewest — regime scaling reduces CS sizing in bear/high_vol.

## 3. Feature Completeness

| Feature | Non-Null | % Complete | Notes |
|---------|----------|-----------|-------|
| entry_date | 430 | 100.0% | |
| exit_date | 430 | 100.0% | |
| year | 430 | 100.0% | |
| strategy_type | 430 | 100.0% | |
| spread_type | 430 | 100.0% | |
| dte_at_entry | 430 | 100.0% | |
| hold_days | 430 | 100.0% | |
| day_of_week | 430 | 100.0% | |
| days_since_last_trade | 422 | 98.1% | First trade per year = null (expected) |
| regime | 430 | 100.0% | |
| rsi_14 | 430 | 100.0% | |
| momentum_5d_pct | 430 | 100.0% | |
| momentum_10d_pct | 430 | 100.0% | |
| vix | 430 | 100.0% | |
| vix_percentile_20d | 430 | 100.0% | |
| vix_percentile_50d | 430 | 100.0% | |
| vix_percentile_100d | 430 | 100.0% | |
| iv_rank | 430 | 100.0% | |
| spy_price | 430 | 100.0% | |
| dist_from_ma20_pct | 430 | 100.0% | |
| dist_from_ma50_pct | 430 | 100.0% | |
| dist_from_ma80_pct | 430 | 100.0% | |
| dist_from_ma200_pct | 430 | 100.0% | |
| ma20_slope_ann_pct | 430 | 100.0% | |
| ma50_slope_ann_pct | 430 | 100.0% | |
| realized_vol_atr20 | 430 | 100.0% | |
| realized_vol_5d | 430 | 100.0% | |
| realized_vol_10d | 430 | 100.0% | |
| realized_vol_20d | 430 | 100.0% | |
| net_credit | 430 | 100.0% | |
| spread_width | 430 | 100.0% | |
| max_loss_per_unit | 430 | 100.0% | |
| short_strike | 249 | 57.9% | Null for SS trades (no single short strike) |
| otm_pct | 249 | 57.9% | Null for SS trades (derived from short_strike) |
| contracts | 430 | 100.0% | |
| exit_reason | 430 | 100.0% | |
| pnl | 430 | 100.0% | |
| return_pct | 430 | 100.0% | |
| win | 430 | 100.0% | |

**Summary:** 37/39 features are 100% complete. The 2 partial columns (`short_strike`, `otm_pct`)
are structurally null for SS trades which use ATM straddle/strangle positions with no single
short strike. These should be treated as strategy-conditional features in ML models.

## 4. Class Balance

| Class | Count | % |
|-------|-------|---|
| Win (1) | 251 | 58.4% |
| Loss (0) | 179 | 41.6% |

Mild class imbalance (1.4:1 ratio). No resampling needed — standard classification thresholds
or class_weight="balanced" are sufficient.

### Win Rate by Strategy

| Strategy | Trades | Win Rate |
|----------|--------|----------|
| CS | 236 | 67.8% |
| SS | 181 | 46.4% |
| IC | 13 | 46.2% |

CS has the highest win rate. SS trades are closer to 50/50 (expected for straddle strategies).
IC has very few samples (13) — treat with caution in ML training.

## 5. Regime Distribution

| Regime | Trades | % | Win Rate |
|--------|--------|---|----------|
| bull | 355 | 82.6% | 60.0% |
| bear | 41 | 9.5% | 48.8% |
| high_vol | 19 | 4.4% | 52.6% |
| low_vol | 11 | 2.6% | 63.6% |
| crash | 4 | 0.9% | 50.0% |

**Note:** Heavy bull skew reflects 2020-2025 market reality. The regime classifier correctly
identifies bear (2022), high_vol (VIX spikes), and crash (COVID, rate shocks) periods, but
sample sizes for non-bull regimes are small. Consider regime-aware stratification or
oversampling for minority regimes in train/test splits.

## 6. Data Quality Assessment

### Strengths
- Near-perfect feature completeness (37/39 at 100%)
- Full 6-year coverage with no gaps
- Reasonable class balance
- All 3 strategy types represented
- MA200 warmup handled correctly (2018+ data used for 2020 trades)

### Known Limitations
- **IC trades (13)**: Too few for reliable IC-specific ML. Consider CS+SS only models.
- **Regime imbalance**: crash (4) and low_vol (11) regimes have very few samples.
  Use stratified splits or regime-aware cross-validation.
- **short_strike/otm_pct nulls**: Structurally null for SS — handle as conditional features.
- **Backtester pricing**: Trades use yfinance+BS pricing (PortfolioBacktester), not IronVault
  real option prices. This is shared-scope and acceptable for training data generation.

### Recommendations
1. For binary classification (win/loss): use all 430 rows, drop short_strike/otm_pct for SS
2. For return regression: use return_pct as target, consider winsorizing outliers
3. Train/test split: use temporal split (e.g., 2020-2023 train, 2024-2025 test) to prevent leakage
4. Feature selection: start with VIX, iv_rank, rsi_14, regime, dist_from_ma200_pct, realized_vol_20d

---

*Generated by CC-3 (ML & Data) — Phase 3 data quality check*
