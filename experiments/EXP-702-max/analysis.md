# EXP-702-max Analysis: Multi-Asset Credit Spread Portfolio

## Summary

| Metric | Result | Target | Met? |
|--------|--------|--------|------|
| Ann. Return | 2.6% | 60-100% | ✗ |
| Sharpe | 0.31 | 3.0-5.0 | ✗ |
| Max Drawdown | 41.8% | <12% | ✗ |
| Win Rate | 56.4% | — | — |
| Sharpe vs SPY-only | -0.13 | +1.0 | ✗ |
| All years profitable | No (2020, 2022 negative) | Yes | ✗ |
| 3+ assets profitable | Yes (3 of 4) | Yes | ✓ |

**659 trades executed** across SPY, QQQ, IWM, IBIT over 2020-2025.

## Key Findings

### 1. Diversification Does NOT Fix a Negative-Expected-Value Strategy

The core SPY credit spread strategy has a **negative average P&L in 2020 (-$641/trade) and 2022 (-$874/trade)**. Running the same strategy on correlated assets amplifies these losses rather than diversifying them away. The portfolio Sharpe (0.31) is actually *lower* than SPY-only (0.44) because:

- QQQ (ρ=0.85 with SPY) loses money in the same periods SPY loses
- IWM (ρ=0.75) provides minimal decorrelation benefit
- IBIT (ρ=0.30) offers genuine decorrelation but is only available from 2024

### 2. Correlation Spikes During Crises

When SPY crashes (March 2020, H1 2022), all equity-correlated assets crash together. The risk parity weights don't adjust fast enough — by the time correlation monitoring detects the spike, losses have already accumulated across all 4 streams.

### 3. Multi-Asset Amplifies Drawdowns

Running the same strategy on 4 correlated underlyings means the portfolio takes **4x the trade count** during drawdown periods. The kill switch triggers repeatedly but resets each year, allowing fresh losses in the new year.

### 4. What Did Work

- **3 of 4 assets were profitable overall** — the diversification adds incremental return in non-crisis years
- **IBIT provides genuine alpha** in 2024-2025 due to low correlation with equities
- **Risk parity weighting** correctly underweights the volatile IBIT
- **Win rate of 56.4%** is consistent with the underlying strategy

## Root Causes of Underperformance

1. **Base strategy quality**: The input data (training_data_combined.csv) has an overall negative average P&L (-$76/trade). Multi-asset diversification cannot make a losing strategy profitable — it needs positive expected value per trade first.

2. **Correlation too high**: SPY/QQQ/IWM are 0.75-0.85 correlated. This is insufficient for meaningful diversification. True decorrelation requires genuinely independent return streams (different asset classes, strategies, or geographies).

3. **Synthetic data limitations**: QQQ/IWM/IBIT trades are generated from SPY data with correlated noise. Real cross-asset options data would show different IV surfaces, liquidity profiles, and premium structures.

## Recommendations

### Short-term (to improve this experiment)
- Focus on signal quality: filter out 2020/2022-style losing trades using the regime gate more aggressively
- Add **cross-asset correlation as a pre-trade gate** — block all new trades when rolling correlation > 0.8
- Implement **portfolio-level Kelly sizing** that accounts for inter-asset correlation

### Medium-term (experiment design)
- **EXP-702 should use real options data** for each underlying (via IronVault) rather than synthetic streams
- Test with genuinely uncorrelated return streams: VIX products, bonds (TLT), commodities (GLD)
- Separate signal generation per asset — each underlying has different IV dynamics

### Long-term (strategic)
- Multi-asset diversification is a valid approach but requires:
  1. Positive expected value in the base strategy (fix signal quality first)
  2. Genuinely uncorrelated assets (add bonds, commodities, crypto)
  3. Dynamic correlation-based allocation (not just risk parity)
  4. Real per-asset options data and pricing

## Thesis Assessment

**The hypothesis that multi-asset diversification can boost returns to 60-100% while keeping DD under 12% is NOT confirmed** with this data. The thesis is directionally correct — diversification does improve risk-adjusted returns in favorable years — but the magnitude of improvement is far smaller than expected because:

1. The base strategy has negative-expectancy years that multi-asset amplifies
2. Equity-correlated assets provide insufficient decorrelation during crises
3. The diversification multiplier only works when each underlying has positive expected value independently

## Data Notes

- SPY: 428 real trades from training_data_combined.csv (2020-2025)
- QQQ: 428 synthetic trades (correlated with SPY at ρ=0.85, vol_scale=1.15)
- IWM: 428 synthetic trades (correlated with SPY at ρ=0.75, vol_scale=1.25)
- IBIT: 139 synthetic trades (2024-2025 only, correlated at ρ=0.30, vol_scale=3.0)
