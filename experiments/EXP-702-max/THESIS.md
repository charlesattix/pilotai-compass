# EXP-702-max: Multi-Asset Portfolio

## Hypothesis

A single underlying (SPY) caps returns at ~25-30% annually. By combining credit spread strategies across multiple uncorrelated underlyings (SPY, QQQ, IWM, IBIT), we can achieve portfolio-level returns of 60-100%+ while maintaining drawdown under 12% through diversification.

## Rationale

- SPY-only strategies are limited by trade frequency and per-trade sizing constraints
- Cross-asset diversification reduces portfolio drawdown (correlation < 1.0)
- Each asset has different volatility regimes — when SPY is quiet, IBIT may be active
- Portfolio-level Sharpe should exceed any single-asset Sharpe
- This is how institutional funds scale — diversification across uncorrelated return streams

## Strategy

- Run existing credit spread strategies (CS, IC, straddles) on SPY, QQQ, IWM, IBIT
- Regime-adaptive allocation: tilt toward assets in favorable regimes
- Risk parity weighting: allocate risk budget inversely to each asset's realized vol
- Cross-asset correlation monitoring: reduce allocation when correlations spike
- Position limits per underlying and aggregate

## Entry Rules

- Per-asset: use existing EXP-400/401 entry logic adapted to each underlying
- Portfolio-level: only enter if aggregate portfolio risk < budget
- Correlation gate: reduce sizing if cross-asset correlation > 0.7

## Exit Rules

- Per-trade: existing stop-loss and profit-target logic
- Portfolio-level: reduce all positions if portfolio DD > 8%
- Kill switch: flatten everything if DD > 10%

## Expected Outcome

- Annual returns: 60-100% (from diversification multiplier)
- Max drawdown: 8-12% (from decorrelation benefit)
- Sharpe: 3.0-5.0+ (from risk-adjusted diversification)
- Trade frequency: 3-5x higher than SPY-only

## Success Criteria

- Portfolio Sharpe > single-asset Sharpe by at least 1.0
- Max DD < 12%
- All 6 years (2020-2025) profitable
- Positive returns in at least 3 of 4 underlyings

## Data Requirements

- SPY options pricing (IronVault — available)
- QQQ, IWM, IBIT options pricing — need to verify availability
- Cross-asset correlation data
- Per-asset regime classification

## Risks

- IBIT history only goes back to ~2024 (limited data)
- QQQ/IWM may be highly correlated with SPY (reducing diversification benefit)
- More underlyings = more execution complexity
- Liquidity varies across underlyings
