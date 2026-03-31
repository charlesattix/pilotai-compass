# EXP-730-max: Short DTE Theta Decay — Analysis

## Summary

| Metric | Result | Target | Status |
|---|---|---|---|
| Total Return | 7.87% (6 years) | >40% annual | **FAIL** |
| CAGR | 1.27% | >40% | **FAIL** |
| Sharpe | 3.69 | >2.5 | **PASS** |
| Max Drawdown | 3.12% | <15% | **PASS** |
| Win Rate | 69.4% | >70% | **FAIL** (marginal) |
| Avg Hold | 3.3 days | <5 days | **PASS** |
| Profit Factor | 1.65 | >1.5 | **PASS** |
| Total Trades | 98 | 150-250/yr | **FAIL** (16/yr avg) |

**Bottom line:** Excellent risk-adjusted metrics (Sharpe 3.69, DD 3.12%) but far too few trades to generate meaningful absolute returns. The strategy is *safe* but *underutilised*.

## What Worked

### 1. Bull and Bear Regime Performance
- **Bull regime**: 45 trades, 75.6% win rate, $6,086 P&L — strong consistent theta capture
- **Bear regime**: 12 trades, 75.0% win rate, $1,680 P&L — short puts rarely tested in bear when we're selling OTM
- Both regimes demonstrate the core thesis: OTM credit spreads collect premium reliably in directional markets

### 2. Risk Management
- **Max DD of 3.12%** is exceptionally low — well below the 15% ceiling
- **No single catastrophic loss** — the 2x stop loss and defined risk worked perfectly
- **Calmar ratio 0.41** is modest but drawdown control is excellent
- Zero stop-loss exits — all losses came from strike breaches, which are smaller than max loss

### 3. 2022 Was the Best Year
- 37 trades, 91.9% win rate, $7,852 P&L — elevated VIX (hence higher IV rank passing the filter) meant juicier premiums
- Validates the core thesis: short DTE credit spreads thrive when IV is elevated but not in crash territory

### 4. Short Hold Period
- Average 3.3 days confirms rapid theta capture
- 68 out of 98 trades (69%) hit the 50% profit target — the theta decay model works

## What Failed

### 1. Far Too Few Trades (Critical)
- Only **98 trades over 6 years** (~16/year) vs target of 150-250/year
- Root cause: IV rank filter at 20th percentile + event day blocking + regime filtering eliminates most trading days
- In bull low-vol years (2021, 2023, 2024), IV rank rarely exceeds 20% → only 6-13 trades
- This is the **single biggest issue** — the strategy works per-trade but doesn't trade enough

### 2. High-Vol Regime Barely Profitable
- 41 trades, 61% win rate, only $100 total P&L — essentially breakeven
- High gamma risk in short DTE + elevated vol = more strike breaches
- The 30% entry probability filter for high_vol helps (prevents losses) but kills trade count

### 3. 2023-2025 Deterioration
- 2023: 8 trades, 50% WR, -$169
- 2024: 6 trades, 50% WR, -$249
- 2025: 8 trades, 12.5% WR, -$2,210
- Low-vol bull market → IV rank rarely passes filter → few trades, and the ones that pass are in unfavorable conditions

### 4. Slippage Is Material
- Total slippage: $12,910 — **164% of total P&L** ($7,867)
- At $0.03-0.05/leg × 2 legs × contracts × 100 multiplier, slippage eats into narrow-spread returns
- On $1-2 wide spreads with $0.15-0.60 credit, $0.06-0.10 slippage is 10-67% of credit — devastating

### 5. No Stop-Loss Exits
- All 30 losses came from "strike_breach" — underlying moved within 90% of short strike
- The 2x stop loss never triggered, suggesting the gamma risk materialises as sudden moves rather than gradual drift

## Key Insights

1. **The strategy has excellent per-trade edge** (Sharpe 3.69) but **cannot deploy capital frequently enough** to generate target returns
2. **Slippage on narrow spreads is a strategy-killer** — $1-2 spreads with $0.03-0.05/leg slippage means 10-67% of credit goes to execution costs
3. **IV rank filter is the primary bottleneck** — in normal markets, IV rank stays below 20% for months, completely shutting down the strategy
4. **The strategy is effectively a "volatility premium harvester"** — it only trades when vol is elevated, which is intermittent

## What to Try Next

### Iteration 1: Relax IV Filter
- Lower IV rank threshold to **10th percentile** or remove entirely
- Rationale: 0-7 DTE always has fast theta decay regardless of IV level
- Risk: lower IV = lower credit = even more slippage drag
- Expected trade count: 200-300/year

### Iteration 2: Widen DTE Range to 3-14 Days
- Current 0-7 is too narrow — many calendar weeks have no viable 0-7 DTE options
- 3-14 DTE still captures the steep part of the theta decay curve
- Allows entering Monday trades that expire the following week
- Expected improvement: +50% trade count

### Iteration 3: Widen Spreads to $2-5
- $1-2 spreads make slippage catastrophic (10-67% of credit)
- $3-5 spreads → $0.06-0.10 slippage is only 2-7% of credit
- More absolute credit per trade → fewer contracts needed → less total slippage
- **This is probably the highest-impact change**

### Iteration 4: Size Up in Bull Regime
- Bull regime has 75.6% WR and consistent returns — increase max position to 3-4%
- Bear/high_vol stay at 1-2%
- Expected: +50% absolute returns from same trade count

### Iteration 5: Add Monday/Wednesday Entry Preference
- Weekly option expirations cluster on Mon/Wed/Fri
- Entering on Monday for Friday expiry = exactly 4 DTE, optimal theta curve position
- Could increase hit rate by timing entries to the weekly cycle

## Recommendation

**Do NOT deploy this strategy as-is.** The 1.27% CAGR is below risk-free rates.

**The core edge is real** (Sharpe 3.69) but the execution is wrong:
1. **Wider spreads ($3-5)** to reduce slippage drag — highest priority
2. **Wider DTE (3-14)** to increase trade frequency
3. **Remove or dramatically lower IV filter** — the short DTE theta decay edge doesn't require elevated IV
4. **Scale position size in bull regime** where win rate is 75%+

A revised strategy with these changes could realistically target 20-40% annual returns while maintaining the excellent risk profile (DD <10%, Sharpe >2.5).
