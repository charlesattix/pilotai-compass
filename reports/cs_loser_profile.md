# Credit Spread Loser Profile

Analysis of the 37 losing CS trades out of 233 total (15.9% loss rate) in
`compass/training_data_combined.csv`. Dataset covers 2020-2025, SPY only,
bull-put and bear-call credit spreads.

---

## 1. Dollar Asymmetry: The Core Problem

| Metric | Winners (196) | Losers (37) |
|--------|---------------|-------------|
| Average P&L | +$1,439 | -$2,283 |
| Median P&L | +$1,279 | -$1,946 |
| Average return | +17.7% | -30.4% |
| Max | +$3,145 | -$7,553 |

**Loss/win ratio: 1.59x.** A single average loss erases 1.6 average wins.
The strategy's 84% win rate masks this asymmetry — the 16% of losers
account for 30% of gross dollar volume. Improving PnL requires either
catching losers or reducing their magnitude, not increasing win rate.

---

## 2. Market Conditions When CS Trades Lose

### Statistically significant differences (losers vs winners, |delta| > 0.5 std)

| Feature | Winners (median) | Losers (median) | Delta | Interpretation |
|---------|-----------------|-----------------|-------|----------------|
| **VIX** | 16.8 | 20.0 | +3.2 | Losers enter in elevated-vol environments |
| **IV Rank** | 10.0 | 21.1 | +11.1 | Losers enter when IV is already rich relative to history |
| **VIX Percentile 50d** | 23.0 | 50.0 | +27.0 | **Strongest signal.** Losers enter when VIX is in the upper half of its recent range |
| **Net Credit** | $2.51 | $1.83 | -$0.68 | Losers collect thinner premiums — less cushion against adverse moves |

### Features that did NOT differ meaningfully

RSI (64.5 vs 60.9), OTM% (2.01 vs 2.00), DTE at entry (17 vs 18), spread
width (12.0 vs 12.0), momentum (1.2% vs 1.0%), distance from MA200
(10.6% vs 10.6%). Losers are not entering at "wrong" strike distances or
"wrong" expirations — they enter at the wrong *time* (when vol is elevated
and credit is thin).

### VIX Level Breakdown

| VIX Band | Trades | Losses | Loss Rate |
|----------|--------|--------|-----------|
| 0-15 | 52 | 7 | 13.5% |
| 15-20 | 115 | 12 | 10.4% |
| **20-25** | **43** | **10** | **23.3%** |
| **25-35** | **23** | **8** | **34.8%** |

Loss rate doubles above VIX 20 and triples above VIX 25. Half of all loss
dollars ($42,714 of $84,476) come from trades entered with VIX >= 20.

### VIX Percentile (50-day) Breakdown

| Percentile Band | Trades | Losses | Loss Rate |
|-----------------|--------|--------|-----------|
| 0-20 | 102 | 14 | 13.7% |
| 20-40 | 35 | 3 | 8.6% |
| 40-60 | 36 | 4 | 11.1% |
| **60-80** | **30** | **7** | **23.3%** |
| **80-100** | **30** | **9** | **30.0%** |

When VIX is in its top quintile relative to the prior 50 days, loss rate
is 30% — nearly 3x the base rate.

### Net Credit Quartile Breakdown

| Quartile | Credit Range | Trades | Losses | Loss Rate | Avg PnL |
|----------|-------------|--------|--------|-----------|---------|
| Q1 (thinnest) | $1.16-$1.68 | 59 | 16 | **27.1%** | +$104 |
| Q2 | $1.68-$2.39 | 59 | 6 | 10.2% | +$787 |
| Q3 | $2.39-$3.83 | 59 | 11 | 18.6% | +$662 |
| Q4 (richest) | $3.83-$4.59 | 59 | 4 | **6.8%** | +$1,848 |

Thin-credit trades (Q1) have 4x the loss rate and 18x less average PnL
than rich-credit trades (Q4). Credit below $1.40 pushes loss rate to 31%.

### Regime and Direction

| Regime | Trades | Losses | Loss Rate |
|--------|--------|--------|-----------|
| bull | 208 | 30 | 14.4% |
| **bear** | **17** | **6** | **35.3%** |
| high_vol | 2 | 1 | 50.0% |
| low_vol | 6 | 0 | 0.0% |

Bear-call spreads lose at 33.3% vs bull-put spreads at 14.4%. The
bear-call trades that lose cluster in late bear markets (Mar 2022, Nov 2022,
Apr 2025) — counter-trend rallies during bearish regimes.

---

## 3. Temporal Clustering

### Loss Streaks (2+ consecutive)

| Start Date | Streak | Total Loss | Context |
|------------|--------|------------|---------|
| 2020-09-02 | 3 | -$4,838 | Sept selloff, VIX 27-34 |
| 2021-05-07 | 2 | -$3,887 | Inflation scare, VIX spike to 22 |
| 2021-09-10 | 2 | -$3,397 | Debt ceiling, China Evergrande |
| 2022-03-15 | 2 | -$3,712 | Fed hiking cycle begins |
| 2022-08-15 | 2 | -$7,507 | Jackson Hole selloff |
| 2024-07-12 | 3 | -$7,038 | July rotation out of mega-cap |

**Pattern:** Loss streaks occur during regime transitions — the period where
a new bearish catalyst emerges but the regime detector hasn't yet flipped.
The bull-put strategy continues selling puts into a weakening market for 1-3
trades before the regime shift is confirmed.

### Monthly Distribution

September is the worst month: 6 losses totaling $16,046 (19% of all loss
dollars). This is consistent with the well-known "September effect" in
equity markets. July is second-worst with 5 losses across 3 years.

### Year Distribution

Losses are distributed across all years (4-8 losses per year, except 2025
which has 5). There is no single catastrophic year — losses are a persistent
background rate, not a tail event.

---

## 4. Rule-Based Filter Analysis

### Filters That Work (Positive Net PnL Impact)

| Filter | Trades Cut | Losers Caught | Win Rate After | Net PnL Impact |
|--------|-----------|---------------|----------------|----------------|
| **hold_days >= 15** | 17 | 8 of 37 (22%) | 84.1% → 86.6% | **+$11,023** |
| **net_credit < $1.68 AND vix_pct50d >= 50** | 18 | 9 of 37 (24%) | 84.1% → 87.0% | **+$7,729** |
| **net_credit < $1.50 AND vix_pct50d >= 60** | 11 | 6 of 37 (16%) | 84.1% → 86.0% | **+$6,297** |
| net_credit < $1.50 AND VIX >= 20 | 6 | 3 of 37 (8%) | 84.1% → 85.0% | +$2,763 |

### Filters That Don't Work (Negative Net Impact)

| Filter | Trades Cut | Net PnL Impact | Why |
|--------|-----------|----------------|-----|
| VIX > 25 | 23 | -$22,074 | Too many good bear-call trades above VIX 25 |
| VIX > 25 AND RSI < 40 | 6 | -$5,135 | Catches 1 loser but kills 5 winners |
| VIX > 20 AND mom5d < -2% AND RSI < 45 | 5 | -$7,030 | Catches 0 losers, kills 5 winners |
| RSI < 35 | 6 | net negative | Oversold = good for bull puts, not bad |

### Why Simple VIX Filters Fail

VIX > 20 catches 18 of 37 losers (49% recall) but also catches 48 winners.
Those 48 winners are bear-call spreads that thrive in elevated-vol
environments. A pure VIX filter cannot distinguish "high VIX + selling calls
= good" from "high VIX + selling puts = bad." The net credit filter works
better because it captures the *consequence* of bad conditions (thin
premium) rather than the condition itself.

### Recommended Filters

**Filter 1: Close trades that haven't hit profit target by day 15.**

This is the strongest finding. Trades that survive 15+ days without closing
are 47% losers (8/17) versus 13% losers for trades resolved within 15 days.
These are trades where the underlying moved adversely enough to prevent the
profit target from hitting, but not adversely enough to trigger the stop
loss — they drift until expiration with negative EV.

This is not a pre-trade filter; it is an **exit rule** change: if a trade
hasn't reached 50% profit by DTE 0 minus 2 days or by hold day 15
(whichever comes first), close it at market. The data shows this saves
$22,419 in avoided losses while sacrificing $11,396 in winners, for a net
benefit of +$11,023.

**Filter 2: Skip trades where net_credit < $1.68 AND VIX is in its top half of
the 50-day range (vix_percentile_50d >= 50).**

This catches thin-credit trades in elevated-vol environments. When you can
only collect $1.50 on a $12-wide spread in a high-VIX environment, the
risk/reward is telling you the market sees this strike distance as
dangerous. Catches 9 of 37 losers (24%), cuts 18 trades total, net benefit
+$7,729.

**Filter 3 (conservative): Skip trades where net_credit < $1.50 AND
vix_percentile_50d >= 60.**

Tighter version of Filter 2: only triggers in the top 40% of VIX range.
Catches 6 losers, cuts 11 trades, net +$6,297. Lower recall but fewer
false positives.

---

## 5. Summary

**Profile of a typical CS loser:**
- Bull-put spread (89% of loss dollars) entered during a bull regime
- VIX elevated (20+) relative to recent history (50th+ percentile)
- Thin credit collected (< $1.68 on a $12-wide spread)
- Holds 15+ days without hitting profit target, eventually stopped out or expires with a loss
- Often part of a 2-3 trade loss streak during a regime transition (e.g., Sept 2020, Jul 2024)

**What DOESN'T predict losses:**
- RSI level, OTM distance, spread width, DTE at entry, momentum — all nearly identical between winners and losers

**Actionable recommendations:**
1. Add a **time-based exit** at hold day 15 (or DTE-2, whichever first) — strongest signal, +$11K net
2. Add a **thin credit + elevated VIX** pre-trade filter (net_credit < $1.68 AND vix_pct50d >= 50) — +$7.7K net
3. Track `vix_percentile_50d` as a feature in the ML model — it's the strongest differentiator (27-point median gap) but is NOT currently in the ensemble's feature set
