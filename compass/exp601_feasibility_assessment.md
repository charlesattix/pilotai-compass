# EXP-601 IBIT ML Signal Filter — Feasibility Assessment
**Date:** 2026-03-27
**Author:** Research review of existing code & outputs
**Sources:** `scripts/exp601_ml_signal_filter.py`, `ml/ibit_model_report.md`,
`ml/ibit_feature_importance.md`, `results/exp601/gap_analysis.md`,
`backtest/ibit_backtester.py`, `compass/crypto/`, `experiments/registry.json`

---

## Executive Summary

**The honest verdict: EXP-601 is premature.** The XGBoost classifier already built
has CV AUC = 0.501 — indistinguishable from a coin flip. The 249-trade training set
is too small, too class-imbalanced (84.3% wins), and covers too narrow a market cycle
(bull-only Nov 2024 – Mar 2026) to support reliable ML. The model code and feature
engineering are solid; the data is not there yet.

**Recommendation:** Run EXP-600 paper trading for 9–12 months, accumulate
400–600 trades including at least one drawdown period, then revisit EXP-601.

---

## 1. How Much IBIT Historical Trade Data Do We Have?

### The options market is very young

IBIT (BlackRock's spot Bitcoin ETF) launched in January 2024. Listed options
became available in ~July 2024. The `crypto_options_cache.db` holds data starting
**2024-11-19** — the earliest date with reliable pricing in the Polygon feed.
That gives a maximum possible backtest window of ~16 months (Nov 2024 – Mar 2026).

### What we actually have

| Source | Trades | Period | Quality |
|--------|--------|--------|---------|
| EXP-600 champion backtest | **249** | Nov 2024 – Mar 2026 | Backtested, not live |
| EXP-600 paper trading (live) | **~0** | Mar 22, 2026 – now (5 days) | Live, but too early |
| `pilotai.db` (live DB) | **1** | 2026-03-24 (test trade, SPY) | Not IBIT |

The 249 training trades are all from a **single backtest run** of the champion config.
They are not walk-forward validated. The crypto options cache DB itself is not present
in the current development environment (only available on the deployed node), which
means the training script requires production access to regenerate data.

### Critical data gaps

1. **No bear-market IBIT data.** Nov 2024 – Mar 2026 was predominantly bullish for
   BTC (Trump election rally Nov 2024, continued momentum through Q1 2025). The model
   has almost no exposure to sustained IBIT downtrends.

2. **Class imbalance: 84.3% win rate.** With 249 trades, there are only ~39 losses
   to learn from. XGBoost (even with `scale_pos_weight=0.186`) struggles to extract
   meaningful loss-predictors from 39 examples.

3. **Single-strategy, single-config data.** All 249 trades come from one parameter
   set (DTE=14, OTM=10%, Width=$5). There's no variation in trade structure to help
   the model learn what separates winners from losers across different market conditions.

---

## 2. What Features Are Relevant for a Crypto-Underlying Options Strategy?

### Current feature set (12 features)

| Feature | In-Sample Importance | Notes |
|---------|---------------------|-------|
| `dte` | 0.168 | Time to expiry — key for theta decay |
| `vix` | 0.143 | SPX volatility — cross-asset risk proxy |
| `credit_received` | 0.142 | Dollar credit — implicit IV/size signal |
| `ma50_distance_pct` | 0.122 | Trend strength |
| `btc_corr_30d` | 0.116 | IBIT-ETHA 30d correlation (BTC proxy) |
| `rsi_14` | 0.108 | Momentum |
| `volume_ratio` | 0.105 | Unusual activity indicator |
| `credit_pct` | 0.097 | Credit / width — proper IV proxy |
| `otm_pct` | **0.000** | No signal found |
| `spread_width` | **0.000** | No signal found |
| `realized_vol_20d` | **0.000** | No signal found |
| `direction_bull` | **0.000** | Direction not predictive? Or too sparse |

**Caveat:** In-sample importance with 249 samples and AUC=0.50 out-of-sample means
these importances are noise-fitted, not discovered signal. The zero-importance features
are correctly excluded but the non-zero ones cannot be trusted yet.

### Missing crypto-specific features

The `compass/crypto/` directory already has code for several signals that are not
being used in EXP-601. These are the most valuable additions for a crypto-options
strategy:

**High priority (crypto-native):**

| Feature | Source | Why it matters |
|---------|--------|----------------|
| **Fear & Greed Index** | `compass/crypto/fear_greed.py` | Direct sentiment; crypto ≠ SPX — greed drives IV mispricing |
| **Funding rate** | `compass/crypto/funding_rates.py` | Perpetuals funding = leveraged positioning signal; elevated funding → crowded long → crash risk |
| **BTC dominance** | `compass/crypto/composite_score.py` | When BTC dominance falls, altcoin risk-on; IBIT options pricing skews |
| **Crypto composite score** | `compass/crypto/composite_score.py` | Already integrates 7 signals into 0-100 sentiment band — should be a feature, not just a routing gate |
| **IBIT IV rank** (vs own history) | Derivable from cache | IBIT implied vol reverts to its OWN mean, not SPX's — need IBIT-specific IV percentile |
| **IBIT realized vol (7d)** | Polygon cache | Short-window crypto vol changes faster than SPX; 7d more actionable than 20d for crypto |

**Medium priority (macro cross-asset):**

| Feature | Why it matters |
|---------|---------------|
| `vix_percentile` vs raw VIX | Raw VIX of 20 means different things in 2020 vs 2021 — percentile is more stable |
| `vix_change_5d_pct` | Rate of VIX change matters for IBIT: rapid VIX spikes kill short-put P&L |
| `spy_return_5d` | Cross-asset momentum: SPX down 5% usually means IBIT down 10-15% |
| `weekend gap` (binary) | Crypto trades 24/7; Monday opens can gap hard after weekend news |

**Lower priority (already implicit in data):**

- Raw `otm_pct` and `spread_width` had zero importance — this makes sense because
  the champion config has fixed values for these. They'll only be informative when
  we vary the strategy structure.

### Features to drop or reconsider

- `btc_corr_30d` uses ETHA as a BTC proxy. ETHA (Ethereum ETF) is directionally
  correlated to IBIT but not the same asset class. A 30d window is too slow for
  crypto correlation. Consider: `btc_corr_7d` using Deribit BTC spot or a rolling
  beta calculation.

- `direction_bull` (zero importance) is a derived feature of MA50 position — which
  is already captured in `ma50_distance_pct`. Redundant.

---

## 3. Is the Training Data Sufficient for ML?

### Short answer: No. Not yet.

#### The cross-validation tells the whole story

```
Fold 1:  Train=63   Val=62   AUC=0.500   Acc=0.032
Fold 2:  Train=125  Val=62   AUC=0.500   Acc=0.774
Fold 3:  Train=187  Val=62   AUC=0.502   Acc=0.710
Mean:                         AUC=0.501   Acc=0.505
```

Fold 1 accuracy of 3.2% reveals the model is predicting nearly all wins (matching
the 84.3% base rate) while ignoring actual signal. AUC=0.50 in all three folds means
the model cannot rank winners above losers on unseen data. **This is random.**

The in-sample results (89.4% win rate, +40.99% return when filtered) are pure overfitting
on 249 examples with 50 trees — the model has memorized the training set.

#### Why 249 trades is insufficient

| Requirement | Current | Needed |
|-------------|---------|--------|
| **Minimum for XGBoost signal** (rule of thumb: 10× features) | 249 (12 features) | ~500+ |
| **Balanced classes** (enough negative examples) | ~39 losses | ~150–200 losses |
| **Regime diversity** (bull + bear + high_vol) | Bull-only | 2+ distinct regimes |
| **Walk-forward OOS window** | 0 live trades | 6+ months of live data |
| **Reliable AUC threshold for deployment** | 0.501 | ≥0.55 consistently |

#### Structural problem: the strategy is too good

The 84.3% base win rate is a double-edged sword for ML. Because the strategy already
filters most bad trades through rule-based logic (MA50 adaptive, 10% OTM, DTE
constraints), there isn't much signal left for the model to find. The remaining ~16%
losers may be genuinely unforeseeable from entry-time features — they may be
stop-losses triggered by sudden crypto volatility that no feature at entry predicts.

This is consistent with the zero-importance for `direction_bull` and `realized_vol_20d`:
these features would matter for a naive strategy, but after the rule-based filter
has already done its job, they carry no residual signal.

#### What the gap analysis confirms

The `results/exp601/gap_analysis.md` identifies a deeper structural problem: the
backtest period (Nov 2024 – Mar 2026) was predominantly a bull-and-recovery cycle
for BTC. There is no 2022-style crypto bear market in the training data. When that
regime arrives, the model trained on this data will be extrapolating beyond its
training distribution — exactly the scenario where ML fails most badly.

---

## 4. Path Forward: What Needs to Happen Before EXP-601 Can Work

### Phase 1: Data accumulation (months 1–9, happening now)

EXP-600 is live as of 2026-03-22. Let it run. Every paper trade is a labeled
data point with real entry conditions (not backtested).

| Milestone | Target | Estimated timing |
|-----------|--------|-----------------|
| 50 live trades | Minimum sanity check | ~2–3 months |
| 200 live trades | Begin feature engineering | ~6–7 months |
| 400+ live trades with regime diversity | Retrain EXP-601 | ~9–12 months |
| First crypto drawdown event in paper data | Critical — can't train without it | Unknown |

### Phase 2: Feature improvements (can start now)

Integrate the existing `compass/crypto/` signals as ML features:

1. Add `fear_greed_index` as a daily feature (already scraped by `composite_score.py`)
2. Add `funding_rate` (perpetuals) as entry-time feature
3. Add `crypto_composite_score` (0-100 band) as a single aggregated sentiment feature
4. Replace `btc_corr_30d` with `btc_corr_7d` for faster crypto signal
5. Add `vix_percentile_50d` instead of raw VIX
6. Add `ibit_iv_rank` computed against IBIT's own vol history (not SPX IV rank)

These can be added to the feature capture in `IBITAdaptiveBacktester` without
waiting for live data — they can be back-filled from `crypto_options_cache.db`
and the composite score historical data.

### Phase 3: Model redesign (when data is ready)

When 400+ trades exist:

- Use the `EnsembleSignalModel` pattern (XGBoost + RF + ET) already validated on
  SPY in EXP-503 — the multi-learner approach was specifically better in high-vol
  regimes, which matters enormously for IBIT
- Replace 3-fold TimeSeriesSplit with proper year-based walk-forward (as in
  `compass/walk_forward.py`) — fold sizes of 63 are too small for reliable AUC
- Target AUC ≥ 0.55 OOS before activating the filter in live trading
- Consider a **regime-conditional model**: separate models for bull/bear/high_vol
  regimes (exactly what EXP-503 did with the SPY ensemble)

---

## 5. Summary Scorecard

| Question | Answer |
|---------|--------|
| How much IBIT trade data? | **249 backtested trades, ~0 live trades** |
| Date range of data | **Nov 2024 – Mar 2026 (16 months, bull-only)** |
| CV AUC (honest) | **0.501 — no signal** |
| Is training data sufficient? | **No. Need 400+, bear regime included** |
| Are features right? | **8/12 are reasonable; 4 have zero importance; missing crypto-native signals** |
| Biggest gap | **No bear/crash data; class imbalance (84% wins)** |
| Best next action | **Run EXP-600 for 9–12 months, add Fear&Greed/funding features, retrain** |
| When might EXP-601 be viable? | **~Q1 2027 if BTC has at least one significant drawdown before then** |

---

## 6. What the Code Gets Right

Despite the data limitations, the EXP-601 implementation is well-built:

- **No data leakage:** TimeSeriesSplit only (no random shuffle on time-series data)
- **No synthetic pricing:** `credit_pct = credit / spread_width` — a direct IV proxy
  from real fills, consistent with the project's no-Black-Scholes constraint
- **Proper class weighting:** `scale_pos_weight=0.186` handles the 84% imbalance
- **Conservative hyperparameters:** `max_depth=3`, `min_child_weight=5` — appropriate
  regularization for a small dataset
- **Honest documentation:** The model report explicitly flags "CV AUC ≈ 0.501" and
  warns against relying on the filter yet

The infrastructure is production-ready. The bottleneck is purely data volume and
regime diversity.
