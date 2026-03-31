# EXP-740-max: Volatility Harvesting

## Hypothesis

Implied volatility consistently overprices realized volatility (the "volatility risk premium"). By systematically selling straddles/strangles and dynamically hedging, we can harvest this premium with controlled risk, generating consistent returns uncorrelated with market direction.

## Rationale

- IV > RV historically ~83% of the time for SPY (academic consensus)
- The volatility risk premium (VRP) is one of the most persistent anomalies in finance
- Selling vol is direction-neutral — profits from time decay regardless of market direction
- Dynamic delta hedging reduces directional exposure while preserving theta capture
- VIX mean-reversion: sell when VIX is high (>20), hedge when it spikes further
- This strategy is fundamentally different from credit spreads — uncorrelated alpha source

## Strategy

- Sell ATM or near-ATM straddles/strangles on SPY (30-45 DTE)
- Delta hedge daily (or intraday if delta exceeds threshold)
- Use VIX term structure as timing signal (contango = sell vol, backwardation = reduce)
- IV rank > 50th percentile to enter (sell expensive vol)
- Gamma scalping: profit from realized vol if it exceeds implied
- Vega hedging with VIX futures/options in extreme environments

## Entry Rules

- IV rank > 50th percentile (vol is relatively expensive)
- VIX term structure in contango (front month < back month)
- No entry within 2 days of FOMC/CPI/NFP (event vol is real, not overpriced)
- Max portfolio vega exposure limit
- Position size: target 2-3% of portfolio theta per day

## Exit Rules

- Profit target: 40-60% of initial credit
- Stop loss: underlying moves > 1.5 standard deviations
- Time exit: roll or close at 14 DTE (avoid gamma risk acceleration)
- Hedge adjustment: re-hedge when delta exceeds ±15 deltas per contract
- VIX spike exit: close if VIX jumps >30% in a single day (regime change)

## Expected Outcome

- Annual returns: 20-40% from vol premium alone
- Max DD: 8-12% (with proper hedging)
- Sharpe: 2.0-3.0 (very consistent, low variance)
- Win rate: 65-75% (after hedging costs)
- KEY VALUE: uncorrelated with credit spread strategies — portfolio diversifier

## Success Criteria

- Positive returns in 5 of 6 years (2020-2025)
- Correlation with EXP-400/401 < 0.3 (true diversification)
- Sharpe > 2.0 standalone
- Max DD < 12%
- Hedging cost < 30% of gross premium collected
- Survives 2020 COVID crash and 2022 bear market

## Data Requirements

- SPY ATM options chain (straddle/strangle pricing) — IronVault
- VIX index + VIX futures term structure
- Realized volatility calculations (various windows)
- Intraday data for hedge simulation (or daily approximation)

## Risks

- Tail risk: selling vol = short gamma = unlimited theoretical loss
- 2020 COVID crash would test this strategy severely
- Hedging costs can eat premium in choppy markets (whipsaw)
- VIX backwardation periods reduce entry opportunities
- Requires more active management than credit spreads
- Gap risk overnight — can't hedge when market is closed
