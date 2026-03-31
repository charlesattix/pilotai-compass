# EXP-800-max: Master Portfolio — Three-Stream Risk Parity

## Hypothesis

By combining three uncorrelated return streams via risk parity allocation, we can achieve portfolio-level returns and risk metrics that surpass any individual strategy. The low cross-stream correlations create a diversification benefit that should push Sharpe above 4.0 while keeping max drawdown below 12%.

## The Three Streams

### Stream 1: ML-Filtered Credit Spreads (EXP-400 lineage)
- Bull put spreads on SPY, filtered by XGBoost model at P>=0.75
- Expected: 25-35% annual return, Sharpe ~2.0, max DD ~15%
- Regime-adaptive: reduces size in high-vol, pauses in crash
- Best in: bull and low-vol regimes
- Worst in: sharp crashes before model can react

### Stream 2: Volatility Harvesting (EXP-740-max)
- Systematic short straddles with delta hedging
- Proven: 15.2% annual, Sharpe 2.55, max DD 6.8%
- Correlation with credit spreads: 0.12 (near zero)
- Best in: high-vol regimes (rich premium)
- Worst in: sustained low-vol (fewer entry signals)

### Stream 3: Short DTE (0-7 DTE) High Frequency
- Rapid-cycle credit spreads at 0-7 DTE
- Expected: 20-30% annual return, higher win rate, faster capital turnover
- Lower per-trade risk, higher trade count
- Best in: range-bound markets
- Worst in: gap moves (overnight risk with short DTE)

## Allocation Strategy

**Risk parity**: allocate inversely proportional to each stream's volatility, so each stream contributes equally to portfolio risk.

Target allocation (approximate):
- Stream 1 (credit spreads): ~40% (moderate vol)
- Stream 2 (vol harvesting): ~35% (low vol, Sharpe 2.55)
- Stream 3 (short DTE): ~25% (higher vol, higher return)

Dynamic rebalancing: monthly, or when any stream's weight drifts >5%.

## Expected Outcome

- **Annual returns: 50%+** (from combining three 15-30% streams with low correlation)
- **Max DD: <12%** (diversification reduces portfolio DD below worst single stream)
- **Sharpe: >4.0** (diversification benefit on Sharpe is multiplicative with uncorrelated streams)
- **All 6 years positive** (each stream covers the others' weak regimes)

## Success Criteria

- Combined annual return > 50%
- Sharpe > 4.0
- Max DD < 12%
- No single year negative
- Each stream contributes positively in aggregate
- Portfolio Sharpe > best individual stream Sharpe

## Risk Model

- Cross-stream correlation increases during market stress (correlation breakdown)
- All three streams have options exposure → correlated tail risk
- Rebalancing costs with monthly frequency
- Model risk: ML filter could degrade over time

## Data Requirements

- SPY price + VIX history (2020-2025)
- Option chain data via IronVault for credit spread pricing
- Synthetic stream returns calibrated to backtested performance
