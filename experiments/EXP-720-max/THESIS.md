# EXP-720-max: Dynamic Regime Sizing

## Hypothesis

Our regime detector identifies bull, bear, high-vol, low-vol, and crash environments. Instead of fixed position sizing across all regimes, we can dramatically improve returns by sizing UP (2-3x) in favorable regimes and sizing DOWN (0.25-0.5x) in hostile ones.

## Rationale

- Credit spreads perform very differently across regimes (bull + low vol = best)
- EXP-400 already shows 100%+ returns in 2021 (bull regime) but -1.9% in 2022 (bear)
- If we sized 3x in 2021-type regimes and 0.25x in 2022-type, overall returns skyrocket
- The regime classifier is already built and validated
- Key insight: it's not about finding more edges — it's about betting MORE on the edges we already have

## Strategy

- Use existing EXP-400/401 signals and entries
- Classify current regime using regime.py
- Apply regime-dependent position multiplier:
  - Bull + Low Vol: 2.5-3.0x base size
  - Bull + Normal Vol: 2.0x base size
  - Sideways: 1.0x base size (default)
  - Bear + Low Vol: 0.5x base size
  - Bear + High Vol: 0.25x base size
  - Crash: 0x (no new positions)
- Sweep multiplier grid to find optimal sizing per regime

## Entry Rules

- Same as base strategy
- Position size = base_size × regime_multiplier
- Max position cap: never exceed 5% of portfolio per trade regardless of multiplier
- Aggregate exposure cap: 30% of portfolio

## Exit Rules

- Same as base strategy
- Add regime transition exit: if regime shifts from bull to bear mid-trade, tighten stops

## Expected Outcome

- Annual returns: 50-80% (from leveraging favorable regimes)
- Max DD: 10-15% (from reducing in hostile regimes)
- Sharpe: 3.0-5.0 (dramatically better risk-adjusted)
- Should turn 2022's -1.9% into a small positive or flat year

## Success Criteria

- Returns > 50% annualized across full 2020-2025 period
- Max DD < 15%
- No individual year worse than -5%
- Regime multiplier impact is monotonic (better regimes = better results at higher sizing)
- Walk-forward validation within 20% of in-sample

## Data Requirements

- Existing trade data with regime labels (training_data_exp400/401.csv)
- Regime classifier (compass/regime.py)
- Per-regime performance breakdown (already computed)

## Risks

- Regime detection has lag — by the time we detect "bull", the best part may be over
- Over-sizing in "bull" that suddenly becomes "crash" = large losses before regime re-classifies
- Regime transitions are the dangerous periods — need careful handling
- Could lead to higher drawdown if regime detection is wrong
