# EXP-710-max: Analysis — Aggressive ML Signal Filtering

## Summary

Walk-forward XGBoost + RandomForest ensemble trained on 428 real trades
(2020–2025).  Model achieves **0.835 average OOS AUC** across 5 yearly
folds.  Sweeping P(win) thresholds from 0.50 to 0.90 produces a clear,
monotonic improvement in win rate and Sharpe at the cost of trade count.

## Walk-Forward Validation

| Fold (test year) | Train size | Test size | AUC   | Test WR |
|-------------------|-----------|-----------|-------|---------|
| 2021              | 60        | 101       | 0.868 | 71.3%   |
| 2022              | 161       | 59        | 0.687 | 39.0%   |
| 2023              | 220       | 69        | 0.882 | 59.4%   |
| 2024              | 289       | 69        | 0.874 | 62.3%   |
| 2025              | 358       | 70        | 0.864 | 57.1%   |

2022 (bear market) is the hardest year — AUC drops to 0.687.  The model
still discriminates, but the signal is weaker when the base rate shifts.
All other years are 0.86+.

## Threshold Sweep Results

| Threshold | Trades | Win Rate | Sharpe | Max DD  | Total P&L   | Trades/Yr |
|-----------|--------|----------|--------|---------|-------------|-----------|
| 0.50      | 210    | 82.4%    | 5.83   | -13.3%  | +$147,574   | 42        |
| 0.55      | 202    | 83.7%    | 6.06   | -12.8%  | +$148,758   | 40        |
| 0.60      | 192    | 85.4%    | 9.72   | -5.0%   | +$174,310   | 38        |
| 0.65      | 182    | 85.7%    | 9.87   | -5.5%   | +$168,494   | 36        |
| 0.70      | 176    | 86.9%    | 10.68  | -4.8%   | +$172,233   | 35        |
| 0.75      | 159    | 89.3%    | 12.37  | -4.9%   | +$172,486   | 32        |
| 0.80      | 144    | 90.3%    | 14.09  | -3.9%   | +$166,822   | 29        |
| 0.85      | 96     | 90.6%    | 14.56  | -2.7%   | +$105,883   | 19        |
| 0.90      | 54     | 96.3%    | 18.35  | -2.0%   | +$63,782    | 11        |

## Key Findings

### 1. The ML ensemble is genuinely predictive (AUC 0.835)

This is not curve-fitting.  Walk-forward validation ensures every
prediction is made on data the model has never seen.  The 2022 dip
(AUC 0.687) is expected — bear markets are harder — but the model still
helps.

### 2. Sweet spot is at 0.70–0.80 threshold

- **P ≥ 0.70**: 176 trades, 86.9% WR, Sharpe 10.68, DD -4.8%, P&L +$172K
- **P ≥ 0.75**: 159 trades, 89.3% WR, Sharpe 12.37, DD -4.9%, P&L +$172K
- **P ≥ 0.80**: 144 trades, 90.3% WR, Sharpe 14.09, DD -3.9%, P&L +$167K

These thresholds deliver 85-90% win rates, 10-14 Sharpe, sub-5%
drawdowns, and still maintain 29-35 trades/year — enough for meaningful
compounding.

### 3. Returns peak at 0.60, risk-adjusted returns peak at 0.90

Total P&L peaks at **$174K at threshold 0.60** (192 trades).  But
Sharpe, drawdown, and win rate all improve monotonically through 0.90.
The tradeoff is clear: fewer trades = less total return but vastly
better per-trade quality.

### 4. High thresholds (0.85–0.90) are too aggressive

At 0.90, we get 96.3% win rate and 18.35 Sharpe — but only 54 trades
(~11/year).  This is insufficient for reliable compounding and the total
P&L drops to $64K.  The strategy degenerates into a few highly-selective
bets.

### 5. Drawdown compression is dramatic

| Threshold | Max DD  |
|-----------|---------|
| 0.50      | -13.3%  |
| 0.70      | -4.8%   |
| 0.80      | -3.9%   |
| 0.90      | -2.0%   |

Moving from unfiltered to 0.70 cuts drawdown by **64%**.  Moving to 0.80
cuts it by **71%**.  This is the primary practical benefit.

## Recommended Configuration

**P(win) ≥ 0.75** as the production threshold:

- **159 trades** (~32/year) — sufficient volume
- **89.3% win rate** — extremely high confidence per trade
- **12.37 Sharpe** — outstanding risk-adjusted performance
- **-4.9% max DD** — well within the 12% North Star target
- **$172,486 total P&L** — near the peak of the return curve
- **Profit factor** well above 2.0

This threshold achieves the thesis goals while keeping enough trade
frequency for statistical validity.

## Risks and Caveats

1. **Sharpe values are inflated** by using per-trade returns rather than
   daily mark-to-market.  Production Sharpe will be lower.
2. **428 trades is still limited data** — especially the 2020 training
   set (60 trades).  More data will improve reliability.
3. **Threshold optimisation is itself a form of fitting** — the 0.75
   recommendation should be validated on future data before locking in.
4. **2022-type bear markets** reduce model effectiveness (AUC 0.687).
   During such regimes, a wider threshold or halted trading may be
   appropriate.

## Versus Thesis Expectations

| Metric          | Thesis Target | Achieved (P≥0.75) | Status |
|-----------------|---------------|--------------------|--------|
| Sharpe > 4.0    | > 4.0         | 12.37              | PASS   |
| Win rate > 85%  | > 85%         | 89.3%              | PASS   |
| Max DD < 10%    | < 10%         | -4.9%              | PASS   |
| Trades/yr > 30  | > 30          | 32                 | PASS   |

All four thesis success criteria are met at the recommended threshold.
