# Research Synthesis — 2026-03-25

**For:** Carlos
**From:** Maximus research session
**Read time:** 5 minutes

---

## Bottom Line

The CS strategy works. The SS strategy doesn't. ML adds marginal value to CS because the base win rate (84%) leaves too few losers to learn from. The biggest PnL improvement available today is a simple exit rule — close stale trades at day 15 — not a fancier model.

---

## 1. Feature Pipeline: Clean > Legacy, But the Gain Is Small

We compared two feature pipelines on all 428 trades (CS+SS) using walk-forward validation:

| Pipeline | Features | Avg AUC | Top Feature |
|----------|----------|---------|-------------|
| Legacy (raw prices, 0-fill) | 21 | 0.831 | `strategy_type_CS` |
| Clean (z-scored, ratios, domain imputation) | 31 | **0.849** | `strategy_type_CS` |

**+1.8% AUC improvement.** The clean pipeline's z-scored VIX and price features generalize better across years than raw absolute values.

But both pipelines have the same problem: the #1 feature is `strategy_type_CS`. The model's primary "insight" is that CS trades win and SS trades lose. This is a data composition artifact, not real signal.

When we strip the model down to **CS-only** trades (233 trades), both pipelines collapse to AUC 0.50-0.55 — barely better than a coin flip. The clean pipeline's advantage disappears because there aren't enough CS losers (37 out of 233) to learn meaningful patterns.

**Takeaway:** The clean pipeline is technically better and should be the default going forward. But don't expect it to meaningfully change CS trade outcomes.

---

## 2. EXP-401 SS Is a Systematic Loser — Kill It

The straddle/strangle leg of EXP-401 has lost money in **every single year** from 2020 to 2025:

| Year | SS PnL | SS Win Rate |
|------|--------|-------------|
| 2020 | -$29,953 | 20.0% |
| 2021 | -$33,154 | 30.0% |
| 2022 | -$23,983 | 19.4% |
| 2023 | -$31,763 | 22.6% |
| 2024 | -$19,241 | 34.5% |
| 2025 | -$28,850 | 22.6% |
| **Total** | **-$166,944** | **24.7%** |

Meanwhile, the CS leg made +$197,599 over the same period. The blended EXP-401 return (+$30,655 total, +5.1% over 6 years) is almost entirely the CS leg dragging the SS leg out of the hole.

**Root cause:** The backtester runs SS in `long_pre_event` mode (buy straddles before FOMC/CPI), which is a structurally negative-edge trade — you pay elevated pre-event IV and then suffer IV crush after the event. The paper trading config specifies `short_post_event` (sell after events, capture crush), but this mode was never backtested.

**Additional finding:** The MASTERPLAN's +40.7% average annual return for EXP-401 is based on a stale results file from 2026-03-12 that predates multiple code changes. Re-running the same backtest today produces +0.86% for 2025, not +35%.

**Action: Disable SS in EXP-401 immediately. CS-only would have returned +$197K instead of +$31K.**

---

## 3. ML Adds Marginal Value for CS-Only

### Binary classification (win/loss)

On CS-only trades, the ensemble achieves AUC 0.55 in walk-forward validation — functionally useless. The V1 confidence gate (threshold 0.30) filters 0-6 trades per year, nudging win rate from 84.1% to 86-87%. This is a real but tiny effect.

### Magnitude prediction (how much will the trade return?)

We tried XGBoost regression predicting continuous PnL. Result: **R² = -0.14** (worse than predicting the mean). The model's quartile spread is **inverted** — trades it labels "worst" actually return +18.4%, while trades labeled "best" return +6.9%. ML-based sizing would have destroyed -229pp of return.

### Why ML struggles here

- 233 CS trades total, only 37 losers — far below the ~1,000+ sample threshold where gradient boosting reliably outperforms simple rules.
- Winners and losers have nearly identical RSI, OTM%, DTE, spread width, and momentum. The distinguishing features (VIX percentile, net credit) are already captured by the strategy's existing filters.
- The top model feature (`hold_days`) is partially forward-looking and should be removed from any production model.

**Takeaway:** ML gating is a nice-to-have, not a must-have for CS. The ensemble earns its keep on the mixed CS+SS dataset (AUC 0.86) because it correctly gates out SS losers — but if SS is killed, the ML's value drops sharply.

---

## 4. Rule-Based Filters: Real Promise, But Thresholds Are Fragile

### What distinguishes CS losers from CS winners

| Feature | Winners (median) | Losers (median) | Gap |
|---------|-----------------|-----------------|-----|
| **VIX Percentile 50d** | 23 | 50 | +27 pts |
| **Net Credit** | $2.51 | $1.83 | -$0.68 |
| **VIX Level** | 16.8 | 20.0 | +3.2 |
| **IV Rank** | 10 | 21 | +11 |

RSI, OTM%, DTE, spread width, momentum — **no difference**. Losers enter at the wrong *time*, not the wrong *price*.

### Filters with positive net PnL impact

| Filter | Trades Cut | Net PnL Impact | Mechanism |
|--------|-----------|----------------|-----------|
| **Close at hold day 15** | 17 | **+$11,023** | Exit rule: stale trades are 47% losers |
| net_credit < $1.68 AND vix_pct50d >= 50 | 18 | +$7,729 | Entry filter: thin credit in hot VIX |
| net_credit < $1.50 AND vix_pct50d >= 60 | 11 | +$6,297 | Tighter version of above |

### Filters that DON'T work

Simple VIX thresholds (VIX > 25) kill profitable bear-call trades that thrive in high vol. RSI filters catch no losers. Multi-condition momentum filters (VIX > 20 AND mom5d < -2% AND RSI < 45) caught 0 losers and 5 winners.

### The threshold problem

These filters were optimized on the full 2020-2025 dataset. The $1.68 credit threshold and 50th percentile VIX cutoff are fit to this specific data. On a $5-wide spread they'd be different numbers. On a $12-wide spread in a different VIX regime, the optimal thresholds shift.

The day-15 exit rule is more robust because it's structural, not threshold-dependent: a trade that hasn't worked after 15 days in a 15-20 DTE strategy is in trouble regardless of the specific market environment. But even this needs validation on out-of-sample years before deployment.

---

## 5. Recommended Next Steps

### Do Now (This Week)

1. **Kill SS in EXP-401.** Set `straddle_strangle.enabled: false` in `configs/paper_exp401.yaml`. The CS leg alone returns 6x more. There is no scenario where long-pre-event straddles make money at this parameterization.

2. **Implement the day-15 exit rule as a backtest experiment.** Add `max_hold_days: 15` to the backtester and re-run 2020-2025. If it confirms the +$11K net benefit out-of-sample (especially in 2022 and 2024 which have the worst loss streaks), ship it.

3. **Flag stale results.** The MASTERPLAN's EXP-401 numbers are wrong. Re-run all validation and update the MASTERPLAN with current numbers.

### Do Next (Next 2 Weeks)

4. **Add `vix_percentile_50d` to the ML feature set.** It's the strongest loser signal (27-point median gap) and is NOT in the current ensemble's 39 features. This is a one-line change in `compass/walk_forward.py` NUMERIC_FEATURES.

5. **Test the thin-credit entry filter in paper trading.** Run a shadow filter that logs "would have skipped" for trades where net_credit < $1.68 AND vix_pct50d >= 50, without actually blocking them. After 30 flagged trades, measure whether the flagged group underperforms.

6. **Switch the clean feature pipeline as default.** The z-scored features are +1.8% AUC better on the mixed dataset and can't be worse on CS-only. No downside risk.

### Don't Do

- **Don't deploy ML-based magnitude sizing.** The regression model is inverted and would destroy value.
- **Don't use complex multi-condition VIX+RSI+momentum filters.** They overfit to historical loss clusters and catch 0 losers in the forward test.
- **Don't expect ML to solve the CS loser problem at current sample size.** 37 losers across 6 years is not enough for any model to learn robust patterns. The rule-based filters are more honest about what the data can support.

---

*Reports referenced: `exp401_2025_investigation.md`, `retroactive_clean_vs_legacy.json`, `cs_only_clean_analysis.md`, `cs_pnl_magnitude_model.md`, `cs_loser_profile.md`*
