# CS-Only Clean vs Legacy Feature Pipeline Analysis

**Generated:** 2026-03-25T05:24:13.771652
**Data:** 233 credit spread trades, 2020-2025
**Baseline win rate:** 84.1% (196/233)
**Total P&L:** $197,599

## Why CS-Only?

The all-strategy model's top feature was `strategy_type_CS` — it learned "CS wins, SS loses"
rather than finding signal within CS trades. This analysis removes that crutch by training
exclusively on CS trades, forcing the model to find real predictive features.

## Head-to-Head Results

| Metric | Legacy Pipeline | Clean Pipeline | Delta |
|--------|:--------------:|:--------------:|:-----:|
| Avg AUC | 0.5582 | 0.5448 | -0.0134 |
| Avg Accuracy | 0.7762 | 0.7762 | +0.0000 |
| Avg Brier | 0.1812 | 0.1762 | -0.0050 |
| Features | 28 | 26 | -2 |

## Per-Fold Detail

### Legacy Pipeline (28 features)
| Year | Train | Test | AUC | Accuracy | Brier | Gated WR |
|------|------:|-----:|----:|---------:|------:|---------:|
| 2022 | 98 | 20 | 0.5000 | 0.6000 | 0.3109 | 60.0% |
| 2023 | 118 | 38 | 0.5000 | 0.8947 | 0.0995 | 89.5% |
| 2024 | 156 | 39 | 0.6540 | 0.8205 | 0.1538 | 81.6% |
| 2025 | 195 | 38 | 0.5788 | 0.7895 | 0.1605 | 87.5% |

### Clean Pipeline (26 features)
| Year | Train | Test | AUC | Accuracy | Brier | Gated WR |
|------|------:|-----:|----:|---------:|------:|---------:|
| 2022 | 98 | 20 | 0.5000 | 0.6000 | 0.3109 | 60.0% |
| 2023 | 118 | 38 | 0.5000 | 0.8947 | 0.0995 | 89.5% |
| 2024 | 156 | 39 | 0.6763 | 0.8205 | 0.1400 | 80.0% |
| 2025 | 195 | 38 | 0.5030 | 0.7895 | 0.1546 | 86.5% |

## Top Features (CS-Only)

### Legacy Pipeline (best fold: 2024)
1. `vix` — 0.1702 #################
2. `realized_vol_5d` — 0.1600 ################
3. `rsi_14` — 0.1389 #############
4. `dist_from_ma200_pct` — 0.1332 #############
5. `dte_at_entry` — 0.1115 ###########

### Clean Pipeline (best fold: 2024)
1. `dist_from_ma200_pct` — 0.1988 ###################
2. `vix_zscore` — 0.1761 #################
3. `rsi_14` — 0.1389 #############
4. `iv_rank` — 0.1116 ###########
5. `spy_price_zscore` — 0.1008 ##########

## Calibration (Pooled OOS)

### Legacy
| Bin | Predicted | Actual | Count |
|-----|:---------:|:------:|------:|
| 0.5-0.6 | 0.530 | 1.000 | 3 |
| 0.6-0.7 | 0.646 | 0.667 | 3 |
| 0.7-0.8 | 0.764 | 0.833 | 6 |
| 0.8-0.9 | 0.841 | 0.779 | 68 |
| 0.9-1.0 | 0.938 | 0.865 | 52 |

### Clean
| Bin | Predicted | Actual | Count |
|-----|:---------:|:------:|------:|
| 0.6-0.7 | 0.664 | 0.600 | 10 |
| 0.7-0.8 | 0.724 | 0.857 | 7 |
| 0.8-0.9 | 0.840 | 0.795 | 83 |
| 0.9-1.0 | 0.935 | 0.933 | 30 |

## ML Gating Impact (30% confidence threshold)

| Year | Pipeline | Trades In | Gated Out | Kept WR | Base WR |
|------|----------|:---------:|:---------:|:-------:|:-------:|
| 2022 | Legacy | 20 | 0 | 60.0% | 60.0% |
| 2023 | Legacy | 38 | 0 | 89.5% | 89.5% |
| 2024 | Legacy | 39 | 1 | 81.6% | 82.1% |
| 2025 | Legacy | 38 | 6 | 87.5% | 86.8% |
| 2022 | Clean | 20 | 0 | 60.0% | 60.0% |
| 2023 | Clean | 38 | 0 | 89.5% | 89.5% |
| 2024 | Clean | 39 | 4 | 80.0% | 82.1% |
| 2025 | Clean | 38 | 1 | 86.5% | 86.8% |

## Verdict

**Legacy pipeline wins on CS-only trades (AUC -0.0134)**

AUC delta (clean - legacy): -0.0134
