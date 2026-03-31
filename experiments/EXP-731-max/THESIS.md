# EXP-731-max: Short DTE Theta Decay — High Frequency Iteration

## Parent Experiment
EXP-730-max (Short DTE Theta Decay)

## Problem Statement
EXP-730 demonstrated excellent risk-adjusted performance (Sharpe 3.69, max DD 3.12%) but generated only 98 trades over 6 years (~16/yr), resulting in just 1.27% CAGR. The IV rank filter, event-day blocking, and regime gating eliminated the vast majority of eligible trading days.

## Hypothesis
By removing the IV rank filter entirely (always trade when regime allows), increasing target frequency to 3 trades/week on a fixed Mon/Wed/Fri schedule, and only blocking FOMC days (not CPI/NFP), we can achieve 150+ trades/year while preserving the theta decay edge. More trades = more compounding = target 15-30% CAGR.

## Changes from EXP-730

| Parameter | EXP-730 | EXP-731 |
|---|---|---|
| IV rank filter | >20th percentile | **None (always trade)** |
| Entry schedule | Random eligible days | **Fixed Mon/Wed/Fri** |
| Event blocking | FOMC + CPI + NFP | **FOMC only** |
| Trade frequency target | 3-5/week | **3/week (Mon/Wed/Fri)** |
| All other params | Same | Same |

## Unchanged Parameters
- DTE: 0-7 days
- Spread width: $1-2
- Credit ratio: 15-30%
- OTM distance: 5-10%
- Max position: 2% of portfolio
- Profit target: 50%
- Stop loss: 2x credit
- Slippage: $0.03-0.05/leg
- Regime filter: skip crash, reduce high_vol
- Starting capital: $100K

## Expected Outcome
- Trade count: 150-200/year (900-1200 over 6 years)
- Win rate: 65-75% (slightly lower than EXP-730 due to less selective entry)
- CAGR: 15-30%
- Max DD: 5-12% (higher than EXP-730 due to more exposure)
- Sharpe: 2.0-3.5

## Success Criteria
- Trades > 150/year
- CAGR > 15%
- Max DD < 15%
- Win rate > 65%
- Sharpe > 2.0
- Improvement over EXP-730 in absolute returns

## Risks
- Without IV filter, we may sell cheap premium in low-vol → lower win rate
- More trades = more slippage drag (still $1-2 spreads)
- Fixed schedule may force trades on suboptimal days
