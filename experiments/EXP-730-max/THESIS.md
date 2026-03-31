# EXP-730-max: Short DTE Theta Decay

## Hypothesis

0-7 DTE credit spreads capture accelerated theta decay (time value evaporates fastest in the last week). More trades per year + faster premium collection = higher annualized returns, provided we manage the increased gamma risk.

## Rationale

- Theta decay is non-linear — the last 7 days capture ~50% of total time decay
- Current strategies use 20-45 DTE, generating ~60-80 trades/year
- Short DTE could generate 150-250+ trades/year
- More trades = more compounding opportunities = higher annualized returns
- 0DTE options market has exploded (massive liquidity in SPY weeklies)
- Risk is manageable with tight stops and small position sizes

## Strategy

- Sell credit spreads on SPY with 0-7 DTE
- Focus on $1-2 wide spreads (defined risk)
- Target 15-30% credit-to-width ratio
- Sell at market open, manage throughout the day
- Use VIX/IV rank as entry filter (only sell when IV is favorable)
- Regime filter: reduce/skip in high-vol and crash regimes

## Entry Rules

- DTE: 0-7 days
- IV rank > 30th percentile
- Spread OTM distance: 5-10% from current price
- Max position: 1-2% of portfolio per trade
- Not during first/last 15 min of trading session (volatile)
- No entry on FOMC/CPI/NFP announcement days

## Exit Rules

- Profit target: 50-70% of max profit (close early)
- Stop loss: 2x credit received
- Time stop: close by 3:30 PM on expiration day
- Emergency: close if underlying moves within 1% of short strike

## Expected Outcome

- Annual returns: 40-80% (from high frequency + theta capture)
- Max DD: 8-15% (small position sizes + defined risk)
- Win rate: 70-80% (OTM credit spreads have statistical edge)
- Trade count: 150-250 per year
- Sharpe: 2.5-4.0

## Success Criteria

- Annualized returns > 40%
- Max DD < 15%
- Win rate > 70%
- Average trade duration < 5 days
- No single trade loss > 2% of portfolio
- Walk-forward validation holds

## Data Requirements

- Intraday or daily SPY options chain data (0-7 DTE)
- IronVault options_cache.db — need to verify short DTE coverage
- VIX/IV data for filtering
- FOMC/CPI/NFP calendar for event avoidance

## Risks

- Gamma risk: short DTE options have extreme gamma — small moves = big P&L swings
- Pin risk: near expiration, delta can flip rapidly
- Assignment risk on ITM shorts near expiration
- May not have sufficient 0DTE historical data for robust backtesting
- Execution is critical — slippage matters much more on narrow spreads
- Black swan intraday moves (flash crash, news events) before stops trigger
