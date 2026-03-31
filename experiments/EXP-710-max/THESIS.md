# EXP-710-max: Aggressive ML Signal Filtering

## Hypothesis

Our ML ensemble (AUC 0.83) predicts trade outcomes well, but we currently use it as a soft filter. By ONLY taking trades where the model predicts >70-80% win probability, we sacrifice trade frequency but dramatically improve win rate, reduce drawdown, and boost risk-adjusted returns.

## Rationale

- Current win rate is ~75-80%. ML model can identify the top-decile trades at 90%+ predicted probability
- Fewer trades = less exposure = lower drawdown
- Higher win rate = smoother equity curve = higher Sharpe
- The ML model already exists (trained on 660 real trades, walk-forward validated)
- This is the lowest-risk experiment — we're just being more selective with an existing edge

## Strategy

- Use existing EXP-400/401 trade signals as base
- Apply ensemble ML model (XGBoost + RF + ExtraTrees) as hard filter
- Only enter trades with predicted win probability > threshold (sweep 0.6-0.9)
- Backtest each threshold to find optimal selectivity vs. returns tradeoff

## Entry Rules

- Base signal fires (existing EXP-400 or EXP-401 logic)
- ML model predicts P(win) > threshold
- Regime check passes (existing regime gate)
- Risk limits satisfied

## Exit Rules

- Same as base strategy (stop-loss, profit target, expiration)
- No changes to exit logic — filtering is entry-only

## Expected Outcome

- At P>0.7 threshold: fewer trades, ~85-90% win rate, moderate returns
- At P>0.8 threshold: much fewer trades, ~92%+ win rate, lower but very consistent returns  
- Sharpe improvement: 2.0x-3.0x over unfiltered
- Max DD reduction: 30-50% lower than unfiltered

## Success Criteria

- Sharpe > 4.0 at optimal threshold
- Win rate > 85%
- Max DD < 10%
- Still generates >30 trades/year (enough to be meaningful)
- Out-of-sample performance holds within 15% of in-sample

## Data Requirements

- Existing training data (training_data_exp400.csv, training_data_exp401.csv)
- Pre-trained ensemble model artifacts
- Walk-forward fold definitions

## Risks

- Too aggressive filtering = too few trades = returns can't compound
- Model may be overconfident on certain trade types
- Threshold optimization could itself be a form of overfitting
- Need sufficient out-of-sample data to validate
