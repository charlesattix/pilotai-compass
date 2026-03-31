# EXP-731-max: Short DTE High Frequency — Analysis

## Comparison vs EXP-730

| Metric | EXP-730 | EXP-731 | Change |
|---|---|---|---|
| Total Return | +7.87% | **-22.67%** | -30.5pp |
| CAGR | +1.27% | -4.20% | -5.5pp |
| Sharpe | 3.69 | **-3.11** | -6.8 |
| Max DD | 3.12% | **24.31%** | +21.2pp |
| Win Rate | 69.4% | **46.1%** | -23.3pp |
| Total Trades | 98 | **347** | +249 (+253%) |
| Profit Factor | 1.65 | **0.66** | -0.99 |
| Total Slippage | $12,910 | **$43,566** | +$30,656 |
| Trades/Year | 16 | **58** | +42 |

## Verdict

**The frequency increase succeeded (347 vs 98 trades) but the strategy is now unprofitable.** Removing the IV rank filter destroyed the edge completely. The result is definitive: **the IV filter was not just a filter — it WAS the edge.**

## What This Proves

### 1. IV Rank Is the Alpha Source
- EXP-730 with IV > 20th percentile: 69.4% win rate, Sharpe 3.69
- EXP-731 without IV filter: 46.1% win rate, Sharpe -3.11
- The entire profit came from selective entry when IV was elevated
- Without this filter, short DTE credit spreads are a **net loser** after slippage

### 2. Low-Vol Regime Is Toxic
- **153 trades in low_vol regime: 37.9% win rate, -$16,000**
- This is 44% of all trades and 71% of all losses
- In low-vol environments, credit received is tiny ($0.15-0.30 on $1 spread) but slippage is fixed ($0.06-0.10) — slippage consumes 20-67% of credit before any market risk
- Bull regime also lost money (-$2,355) — even "good" regimes don't help without IV edge

### 3. Slippage on $1-2 Spreads Is Catastrophic at Scale
- $43,566 in slippage on 347 trades = **$126/trade average**
- Average winning trade: $275; average losing trade: -$356
- Slippage is **46% of average win** — unviable for a scalable strategy
- This confirms EXP-730's analysis: wider spreads are necessary

### 4. The Strategy Degrades Over Time
- 2020: -3.5%, 2021: -3.2%, 2022: **+5.5%** (high IV year), 2023: -5.2%, 2024: -7.0%, 2025: -9.2%
- Only 2022 was profitable — the year with highest average VIX
- The strategy gets worse as markets normalise, because credit received shrinks

### 5. Strike Breaches Dominate Losses
- 185 strike breaches (53% of trades) vs 162 profit targets (47%)
- More strike breaches than profit targets = negative expectancy
- Short DTE + low IV = strikes are closer to ATM relative to premium received

## Key Learning

The original EXP-730 thesis was **partially correct**: short DTE theta decay works, but **only when IV is elevated**. The strategy is really a **"sell elevated short-term premium"** strategy, not a "trade frequently" strategy. Forcing frequency by removing the IV filter turns a profitable niche strategy into a consistent loser.

## What to Try Next (EXP-732+)

### Option A: Keep IV Filter, Widen Spreads
- Restore IV rank > 20th percentile
- Widen spreads to $3-5 to reduce slippage impact
- Expected: fewer trades (~100/yr) but each trade is more profitable
- This addresses the slippage problem without losing the edge

### Option B: IV-Adaptive Sizing
- Trade on every Mon/Wed/Fri (high frequency)
- BUT size positions by IV rank: high IV = 3-4% risk, low IV = 0.5% risk
- This gets frequency AND preserves the IV edge
- Expected: 150+ trades/yr with variable size

### Option C: Hybrid DTE (3-14 days)
- Extend DTE to 3-14 to access more expiration cycles
- Keep IV filter at 20th percentile
- Expected: 2-3x more trading opportunities while maintaining edge

### Option D: Abandon Short DTE, Return to 20-45 DTE
- The data says short DTE only works with IV edge + selective entry
- 20-45 DTE naturally has wider spreads and lower slippage impact
- EXP-400 (parent strategy at 20-45 DTE) already achieves better absolute returns

## Recommendation

**EXP-731 conclusively disproves the high-frequency hypothesis.** The alpha is in IV-based entry selection, not in trade frequency. Next iteration should focus on **Option B (IV-adaptive sizing)** which preserves both frequency and edge, or **Option A (wider spreads)** which is the lowest-risk improvement.
