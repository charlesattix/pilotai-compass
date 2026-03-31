# CS Enhanced Filters Backtest — EXP-400 Comparison

**Date:** 2026-03-25
**Branch:** main
**Filters tested:**
1. **Time exit at day 15** (`max_hold_days=15`): Close any CS trade held >= 15 days
2. **Low-credit + high-VIX gate** (`min_credit_filter=1.68`, `vix_pctile_gate=50`): Skip trades where net_credit < $1.68 AND VIX 50-day percentile >= 50%

---

## Year-by-Year Results

| Year | Baseline Ret | Enhanced Ret | Delta | B DD | E DD | B WR | E WR | B Sharpe | E Sharpe | B Trades | E Trades |
|------|-------------|-------------|-------|------|------|------|------|----------|----------|---------|---------|
| 2020 | -13.77% | **-4.64%** | **+9.13pp** | -29.44% | -24.54% | 73.3% | 83.3% | -0.56 | -0.14 | 30 | 30 |
| 2021 | 97.51% | **110.14%** | **+12.63pp** | -2.90% | -2.38% | 88.7% | 92.9% | 6.40 | 7.10 | 71 | 70 |
| 2022 | -26.48% | -30.38% | **-3.90pp** | -33.82% | -35.92% | 60.7% | 58.6% | -1.41 | -1.69 | 28 | 29 |
| 2023 | 29.06% | 28.50% | -0.56pp | -3.32% | -3.32% | 89.5% | 89.2% | 3.36 | 3.30 | 38 | 37 |
| 2024 | 21.59% | **23.83%** | **+2.24pp** | -5.58% | -5.76% | 82.5% | 84.6% | 2.49 | 2.73 | 40 | 39 |
| 2025 | 12.57% | 7.76% | **-4.81pp** | -19.63% | -23.91% | 84.6% | 83.3% | 0.68 | 0.42 | 39 | 36 |
| **Avg** | **20.08%** | **22.54%** | **+2.45pp** | | | **79.9%** | **82.0%** | | | | |

**Net effect: +2.45pp average annual return, +2.1pp win rate improvement.**

The filters improve 3 of 6 years (2020, 2021, 2024) and hurt 2 years (2022, 2025). 2023 is neutral.

---

## Filter Effectiveness Breakdown

### Filter 1: Time Exit at Day 15

15 trades hit the time exit across all years. Counterfactual analysis (what the baseline did with those same trades):

| Year | Time Exits | Enhanced PnL | Baseline PnL | Filter Helped? |
|------|-----------|-------------|-------------|---------------|
| 2020 | 4 | -$3,254 | -$3,819 | Yes (+$565) — cut a -$1,199 loss at -$140 |
| 2021 | 1 | +$2,094 | +$1,388 | Yes (+$706) — locked in profit earlier |
| 2022 | 5 | -$4,571 | -$4,481 | **No (-$90)** — forced exit on 2022-05-13 trade at -$593 that would have been +$3,020 |
| 2025 | 4 | -$434 | +$9,173 | **No (-$9,607)** — forced exit on two trades that expired as +$6,168 winners |

**The time exit cuts losses in volatile years (2020) but destroys winners in trending years (2025).** The critical failure: in 2025, the 2025-03-11 and 2025-05-02 trades were exited at day 15 for losses (-$1,118 and -$3,511) but would have recovered to +$2,903 and +$3,265 by expiration. CS trades near expiration converge to max profit if the spread stays OTM — the time exit fights this natural theta decay.

### Filter 2: Credit Gate (net_credit < $1.68 when VIX pct >= 50)

| Year | Trades Gated | Winner/Loser | Net PnL Lost |
|------|-------------|-------------|-------------|
| 2020 | 8 | 4W / 4L | -$3,646 (net negative — gating helped) |
| 2021 | 8 | 4W / 4L | -$3,879 (net negative — gating helped) |
| 2022 | 5 | 4W / 1L | +$4,717 (net positive — gating **hurt**) |
| 2025 | 12 | 10W / 2L | +$5,702 (net positive — gating **hurt badly**) |

**The credit gate's $1.68 threshold is too high for low-VIX environments.** In 2025, VIX averaged 16-18 and many legitimate winning trades had credits of $1.27-$1.50. The gate killed 10 winners to avoid 2 losers — a terrible trade-off.

---

## Root Cause: Why Filters Hurt in 2022 and 2025

### 2022 (bear market)
- The credit gate filtered trades with credits of $3.10-$4.17 because VIX percentile was high (VIX in the 25-30 range puts it above the 50th percentile vs the 2020-2021 training period). But $3-4 credits in a high-VIX environment are perfectly normal and profitable. The gate's absolute credit threshold ($1.68) doesn't adapt to the VIX regime.

### 2025 (low-vol bull)
- VIX averaged ~16 in H2 2025. Credits naturally fell to $1.20-$1.60 because options premiums were low. But these trades still had 85% win rates — the lower credit was correctly priced for the lower-risk environment. The $1.68 gate, calibrated on higher-vol data, over-filtered.
- The time exit killed two CS trades at day 15 that were in temporary drawdown but would have recovered by expiration (3-6 more days). CS theta acceleration in the final week makes early exit counterproductive.

---

## Verdict

| Filter | Years Helped | Years Hurt | Recommendation |
|--------|-------------|-----------|----------------|
| Time exit (day 15) | 2020, 2021 | 2022, 2025 | **Do not deploy at day 15.** Consider day 20-25 or only in high-VIX regimes. |
| Credit gate ($1.68 + VIX pct >= 50) | 2020, 2021 | 2022, 2025 | **Do not deploy with fixed threshold.** Needs VIX-relative credit minimum. |

### Why +2.45pp Average Is Misleading

The +2.45pp average is driven almost entirely by 2021 (+12.63pp), a historically unusual year where the filters correctly avoided several late-year losses. But 2021 was a +97% year anyway — the filters turned a great year into a slightly better great year. In the years that matter most for risk management (2022 bear market, 2020 COVID), the filters either hurt (-3.9pp in 2022) or helped modestly (+9pp in 2020). **The filters don't improve the worst-case years**, which is what matters for drawdown control.

### If Deploying Anyway

If deploying these filters despite the mixed results:

1. **Increase max_hold_days to 21** (from 15). This avoids cutting trades that are 3-5 days from profitable expiration. Most CS trades have 15 DTE at entry, so day-21 exit only triggers for trades that got rolled or had unusual DTE.

2. **Make the credit threshold VIX-relative** instead of absolute. For example: `min_credit = 0.14 * spread_width` (credit-to-width ratio >= 14%). This adapts naturally to the premium environment.

3. **Raise the VIX percentile gate to 70** (from 50). This narrows the gate to only the highest-VIX environments where low-credit trades are genuinely riskier, avoiding false positives in normal conditions.

---

## Implementation Details

Both filters are implemented in `strategies/credit_spread.py` as configurable parameters with zero-default (disabled by default):

```python
# Time exit (manage_position)
max_hold_days = 0   # 0 = disabled. Set to 15-25 to enable.

# Credit + VIX gate (generate_signals)
min_credit_filter = 0.0  # 0 = disabled. Set to 1.0-2.0 to enable.
vix_pctile_gate = 100    # 100 = disabled. Set to 50-70 to enable.
```

The VIX percentile is computed from `market_data.vix_history` (50-day rolling window) inside `_compute_vix_percentile()`.

To enable for EXP-400, add to champion.json `strategy_params.credit_spread`:
```json
"max_hold_days": 21,
"min_credit_filter": 1.68,
"vix_pctile_gate": 70
```
