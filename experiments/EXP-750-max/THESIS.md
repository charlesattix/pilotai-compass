# EXP-750-max: Combined ML-Filtered CS + Vol Harvesting

## Hypothesis

Combining two uncorrelated strategies — ML-filtered credit spreads (EXP-710, Sharpe 12.37) and volatility harvesting (EXP-740, Sharpe 2.55, correlation 0.12 with CS) — with a 60/40 allocation will produce a portfolio with:
- Higher risk-adjusted returns than either strategy alone
- Lower max drawdown through decorrelation
- Consistently positive annual returns across all regimes

## Source Strategies

### Leg 1: ML-Filtered Credit Spreads (EXP-710, P≥0.75)
- Sharpe: 12.37
- Annual return: 15.9%
- Max drawdown: -4.9%
- Win rate: 89.3%
- 159 trades over 5 years (31.8/year)

### Leg 2: Volatility Harvesting (EXP-740)
- Sharpe: 2.55
- Annual return: 15.2%
- Max drawdown: -6.8%
- Win rate: 81.8%
- 44 trades over 6 years (~7/year)
- Cross-strategy correlation with CS: ~0.12

## Allocation
- 60% capital → ML-filtered CS (higher Sharpe, higher trade frequency)
- 40% capital → Vol harvesting (decorrelation benefit, all-weather returns)

## Expected Outcome
- Combined Sharpe: 5.0-8.0 (portfolio diversification effect)
- Annual return: 15-20%+ (both legs contribute ~15% independently)
- Max drawdown: 3-6% (decorrelation reduces combined DD)
- All 6 years profitable (vol harvesting covers CS weak periods)

## Success Criteria
- Combined Sharpe > max(single leg Sharpe) * 0.5
- Max DD < 8%
- All 6 years profitable
- Correlation between legs < 0.3
