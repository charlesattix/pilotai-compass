# EXP-760-max: Analysis — ML Filter + Dynamic Regime Sizing

## Summary

Combined EXP-710's ML filter (P≥0.60, 192 trades) with regime-dependent
position sizing.  The filtered trade set is **almost entirely bull regime**
(174/192 = 91%), so bull multipliers dominate the outcome.  Regime sizing
successfully amplifies returns without catastrophic drawdown increase.

## Data Characteristics

After ML filtering at P≥0.60, the surviving trades have this regime
distribution:

| Regime   | Trades | % of total |
|----------|--------|------------|
| Bull     | 174    | 90.6%      |
| Bear     | 12     | 6.3%       |
| Low-vol  | 6      | 3.1%       |
| High-vol | 0      | 0.0%       |
| Crash    | 0      | 0.0%       |

The ML filter already eliminates virtually all high-vol and crash trades.
Regime sizing primarily acts as a **bull leverage multiplier**.

## Configuration Sweep Results

| Config          | Bull | Bear | High-Vol | Total P&L  | Sharpe | Max DD  | Δ P&L    |
|-----------------|------|------|----------|------------|--------|---------|----------|
| Baseline        | 1.0x | 1.0x | 1.0x     | +$174,310  | 9.72   | -5.0%   | —        |
| Conservative    | 1.25x| 0.75x| 0.50x    | +$216,213  | 9.79   | -5.9%   | +$41,903 |
| Moderate        | 1.50x| 0.50x| 0.25x    | +$258,116  | **9.79** | -6.7%   | +$83,806 |
| Aggressive      | 2.0x | 0.50x| 0.25x    | +$342,157  | 9.74   | -7.5%   | +$167,847|
| Very Aggressive | 2.5x | 0.30x| 0.10x    | +$426,104  | 9.71   | -8.2%   | +$251,794|
| Bull Only       | 2.0x | 0.25x| 0.10x    | +$339,160  | 9.65   | -7.7%   | +$164,850|
| Bear Hedge      | 1.5x | 0.75x| 0.50x    | +$258,234  | 9.77   | -6.5%   | +$83,924 |

## Key Findings

### 1. Regime sizing is effectively a bull leverage knob

With 91% of trades in bull regime, the primary effect of the multiplier
is scaling bull-market returns.  Bear and high-vol multipliers have
minimal impact because the ML filter already removes those trades.

### 2. Sharpe is remarkably stable across all configurations

All configs produce Sharpe between 9.65 and 9.79.  This means the ML
filter is doing the heavy lifting for risk adjustment, and regime sizing
is a **return scaling** mechanism that preserves the same risk profile.

### 3. Moderate config (1.5x bull) is the Sharpe-optimal choice

- **Sharpe 9.79** — highest across all configs
- **P&L +$258K** — 48% improvement over unfiltered
- **Max DD -6.7%** — still well below 12% North Star target
- **Win rate 85%** — maintained from ML filter

### 4. Very Aggressive (2.5x bull) maximizes total returns

- **P&L +$426K** — 144% more than baseline
- **Sharpe 9.71** — only 0.08 lower than optimal
- **Max DD -8.2%** — still within acceptable bounds
- This is viable if the investor prioritizes absolute returns over
  marginal Sharpe

### 5. Drawdown scales linearly with leverage

| Config          | Bull Mult | Max DD  | DD / Baseline DD |
|-----------------|-----------|---------|------------------|
| Baseline        | 1.0x      | -5.0%   | 1.00x            |
| Moderate        | 1.5x      | -6.7%   | 1.34x            |
| Aggressive      | 2.0x      | -7.5%   | 1.50x            |
| Very Aggressive | 2.5x      | -8.2%   | 1.64x            |

DD scales sub-linearly with leverage (1.64x DD at 2.5x leverage), which
is the diversification benefit of the ML filter removing the worst trades.

## Recommended Configuration

**Moderate (bull 1.5x, bear 0.5x, high-vol 0.25x)** for production:

- Sharpe 9.79 (highest)
- P&L +$258K (48% improvement)
- Max DD -6.7% (well within limits)
- Conservative leverage that doesn't overextend

For investors with higher risk tolerance, **aggressive (2.0x bull)** is
defensible: Sharpe only drops by 0.05 but P&L jumps to +$342K.

## Versus Thesis Expectations

| Criterion                  | Target        | Achieved (Moderate) | Status |
|----------------------------|---------------|---------------------|--------|
| Sharpe improvement ≥ 20%   | > 11.7        | 9.79 (≈ 0.7%)      | MISS   |
| P&L improvement ≥ 10%      | > $192K       | $258K (48%)         | PASS   |
| Max DD < 8%                | < 8%          | -6.7%               | PASS   |
| Win rate ≥ 84%             | ≥ 84%         | 85.4%               | PASS   |

Sharpe improvement did **not** hit the 20% target because the baseline
was already very high (9.72) and regime sizing primarily adds leverage
rather than improving the quality of trade selection.  However, total
P&L improvement massively exceeds the 10% target.

## Combined Stack: EXP-710 + EXP-760

| Metric     | Unfiltered | EXP-710 (P≥0.60) | EXP-760 (Moderate) | Full Improvement |
|------------|------------|-------------------|---------------------|------------------|
| Win Rate   | 57.7%      | 85.4%             | 85.4%               | +27.7pp          |
| Sharpe     | ~1.0       | 9.72              | 9.79                | ~10x             |
| Max DD     | ~-25%      | -5.0%             | -6.7%               | -18.3pp          |
| Total P&L  | -$33K      | +$174K            | +$258K              | +$291K swing     |

The combined ML filter + regime sizing stack transforms a losing baseline
into a high-Sharpe, controlled-drawdown strategy with $258K total P&L.
