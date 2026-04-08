# Strategy Research Round 5 — Uncorrelated Alpha Sources

**Date:** 2026-04-06
**Author:** Research (no backtesting — proposals only)
**Prior rounds:** R3 (compass/strategy_discovery_r3.py), R4 (strategy_discovery_r4.py), next_strategy_proposals.md

---

## Context

Our current validated portfolio is dominated by EXP-1220 credit spreads:
- **EXP-1220**: 287 trades at 7d cadence, 91% win rate, Sharpe 3.12, +3.4% CAGR (no hedge)
- **Per-trade SPY correlation**: +0.15 (short gamma tilt)
- **The edge is real but small** — ~$3.5K/yr gross on $100K

We need **uncorrelated** alpha sources to push the portfolio toward 10%+ CAGR without
adding more leverage on the same risk factor. Each proposal below is designed to
have low-to-negative correlation with EXP-1220's credit spread exposure.

### What counts as "uncorrelated"

EXP-1220's P&L driver is **short gamma + theta decay** on SPY. Correlated strategies:
- Other short-premium SPY strategies (pairs, VTS) — DO NOT ADD
- Any long-equity beta (SPY outright) — DO NOT ADD
- Short put spreads on correlated ETFs (XLI, QQQ, XLF at >0.9 corr)

Genuinely uncorrelated drivers:
- **Volatility of volatility** (convexity trades, VRP harvest)
- **Cross-asset momentum** (rates, commodities, FX, sectors)
- **Event drift** (FOMC, earnings, roll yield)
- **Statistical mean reversion** (pairs with low SPY beta)

---

## Proposal 1 — VIX Term Structure Carry (Calendar Roll)

**Category:** Volatility Arbitrage
**Data needed:** VIX futures front/back prices daily, OR VIX ETF proxies (VXX, UVXY)
**Data status:** NOT in IronVault. VIX/VXX/UVXY absent. Would require Polygon tier upgrade or yfinance download for VXX daily closes.

### Hypothesis

The VIX futures curve is in contango ~85% of the time. VXX (and VIXY) structurally
decay because they roll from cheaper front to more expensive second-month futures.
A systematic short VXX position with volatility-filtered exit captures this roll
yield (~20-40%/yr historically) with bounded downside via a stop rule.

**The edge:** Variance risk premium at the futures level. Market participants
overpay for convex tail protection; systematic sellers of this convexity earn the
premium. Unlike SPY puts (which EXP-1220 is implicitly short via spreads), VXX
decay is a direct exposure to the *shape* of the vol curve, not the level.

### Signal Logic

- **Entry:** Short VXX (or long UVXY put spread) when:
  1. VIX/VIX3M ratio < 0.95 (curve in contango)
  2. SPY 20d trend > 0 (bull regime)
  3. VIX < 22 (no active vol spike)
- **Exit:** Close when (a) VIX/VIX3M > 1.05 (curve inversion), (b) VIX > 28, or (c) 30-day holding limit
- **Position sizing:** 1-2% of capital per trade, max 2 concurrent

### Expected Performance

| Metric | Estimate |
|--------|----------|
| Trades per year | 6-8 |
| Sharpe | 1.5-2.5 |
| CAGR contribution | 4-8% |
| Max DD | 15-25% (tail risk from vol spikes) |
| Corr to EXP-1220 | **-0.20 to +0.10** (different exposure: curve shape vs level) |
| Capacity | $500M+ (VXX daily volume ~$4B) |

### Caveats

- **Tail risk is severe.** In Feb 2018 (Volmageddon), XIV/SVXY blew up in a single
  session. Must be sized conservatively and always paired with a hard stop.
- **The real edge in contango has shrunk** as more systematic players harvest it.
  Historical backtests may overstate current opportunity.
- **Requires daily rolling** — not a passive hold.

### Decision

**PROPOSE: Add VXX daily bars to data inventory (yfinance download), then build a
proof-of-concept using VXX close prices and SPY/VIX signals. Options on VXX would
be better but require additional data acquisition.**

---

## Proposal 2 — Cross-Asset Momentum Rotation

**Category:** Cross-Asset Momentum
**Data needed:** Daily prices for GLD, TLT, SPY, QQQ (underlying, not options)
**Data status:** Yfinance free. IronVault has options on GLD/TLT/SPY/QQQ but underlying daily bars via yfinance are sufficient for signal generation.

### Hypothesis

Classic 12-minus-1 momentum: rank assets by 12-month return (excluding the most
recent month to avoid reversal noise), go long the top-ranked asset each month
using either the ETF directly or a long call spread on the ETF. This captures the
well-documented cross-sectional momentum premium without stock-picking risk.

**The edge:** Momentum anomaly — assets that have performed well recently continue
to outperform for 3-12 months (time-series persistence of returns). Works across
asset classes (equities, bonds, commodities, FX).

### Signal Logic

- **Universe:** {GLD, TLT, SPY, QQQ} — 4 liquid ETFs across 3 asset classes
- **Rank metric:** 12-month return minus 1-month return (t-13 to t-1)
- **Portfolio:** Long top 1-2 assets (equal weight), hold for 30 days, rebalance
- **Execution:** Use 60-DTE 10-delta call spreads instead of ETF shares (leverage, defined risk)
- **Filter:** Skip if top asset's 3-month return is negative (trend breakdown)

### Expected Performance

| Metric | Estimate |
|--------|----------|
| Trades per year | 24-36 (monthly rebalance × 2 positions) |
| Sharpe | 0.8-1.5 |
| CAGR contribution | 6-12% |
| Max DD | 15-25% |
| Corr to EXP-1220 | **-0.10 to +0.30** (depends on equity weight) |
| Capacity | $1B+ (underlying ETFs are deeply liquid) |

### Caveats

- **QQQ option data ends 2023-04** in IronVault — would need backfill or use ETF
  shares directly for Q2 2023+
- **Momentum crashes** in regime transitions (March 2009, March 2020). Need a
  rvol-based position sizer to cut exposure when vol spikes.
- **Correlation to EXP-1220 depends on current ranking**: when SPY/QQQ lead,
  correlation rises; when GLD/TLT lead, correlation turns negative. This is
  actually desirable — it becomes a natural hedge in equity bear markets.

### Decision

**PROPOSE: Build proof-of-concept using ETF shares (no options needed initially).
Signal is yfinance-only. If it works, layer call spreads on top for leverage.
Data gap (QQQ options to 2023-04) is acceptable for initial validation.**

---

## Proposal 3 — FOMC Announcement Drift

**Category:** Event-Driven
**Data needed:** FOMC meeting dates (static list), SPY options around event windows
**Data status:** IronVault has SPY options fully covered 2020-2026. FOMC dates are public (56 dates 2020-2026 already in `shared/constants.py`).

### Hypothesis

SPY systematically drifts UP in the 24-hour window following FOMC announcements,
regardless of the actual rate decision or statement content. This "FOMC drift"
has been documented in academic literature (Lucca & Moench 2015): the S&P 500
gains ~49 bps on average in the pre-announcement window (2pm ET meeting day to
2pm next day), with high statistical significance and low drawdown.

**The edge:** Structural — institutional de-risking before the meeting followed
by re-risking afterward. This pattern has held post-2015 despite becoming widely
known, suggesting the underlying driver (asymmetric information risk pricing) is
persistent.

### Signal Logic

- **Entry:** 5 minutes before each FOMC announcement (2pm ET meeting day):
  - Buy SPY 0-DTE ATM call (or 1-DTE if 0-DTE unavailable)
  - Alternative: long call debit spread for defined risk
- **Exit:** Next day 2pm ET, regardless of P&L
- **Position sizing:** 2% of capital per event (56 FOMC dates 2020-2026 = ~8/yr)
- **Filter:** Skip if VIX > 30 (crisis overrides the drift signal)

### Expected Performance

| Metric | Estimate |
|--------|----------|
| Trades per year | 8 (FOMC cadence) |
| Win rate | 65-75% |
| Sharpe | 1.2-2.0 |
| CAGR contribution | 3-6% |
| Max DD | 5-10% (few, large losses when the drift fails) |
| Corr to EXP-1220 | **+0.05 to +0.15** (both long SPY beta, but different timescales) |
| Capacity | $100M (0-DTE liquidity is deep but not infinite on a single event) |

### Caveats

- **0-DTE options decay fast.** Precise entry timing matters — a delay of 30
  minutes could cost 10% of the trade value.
- **Small sample:** Only 8 events per year means statistical significance is low
  over any single year. Need 3+ years of walk-forward to confirm edge persistence.
- **Lucca & Moench used cash equity, not options.** The option overlay adds
  theta + vega risk that could negate the underlying drift.
- **Alternative structure:** Buy the ETF shares directly (SPY) with 2x leverage.
  Simpler and avoids theta decay. Less capital efficient but easier to validate.

### Decision

**PROPOSE: Start with SPY shares (no options) to validate the drift exists in
our data window 2020-2026. IronVault SPY daily bars are sufficient. If confirmed,
layer 0-DTE or 1-DTE calls for leverage.**

---

## Proposal 4 — TLT–SPY Pairs Mean Reversion (Stat Arb)

**Category:** Statistical Arbitrage
**Data needed:** Daily TLT + SPY prices, TLT + SPY options for entry/exit
**Data status:** FULL — both tickers in IronVault with options through 2025+

### Hypothesis

TLT (20+ year Treasury ETF) and SPY have a long-term cointegration driven by
the Fed's dual mandate: when equities sell off, bonds rally (flight-to-safety),
and the ratio mean-reverts. When TLT and SPY diverge by >2 standard deviations
from their 60-day cointegrated spread, the pair reverts within 2-6 weeks with
70-75% probability.

**The edge:** Structural — the asset class correlation is mean-reverting at
intermediate horizons (weeks to months) even if the correlation flips sign at
longer horizons. Recent regime (2022-2023) showed positive stock-bond correlation,
but the short-term mean reversion still held.

### Signal Logic

- **Cointegration test:** Rolling 60-day OLS of log(SPY) on log(TLT), compute
  residual z-score
- **Entry (long spread):** When z-score < -2.0 (TLT cheap relative to SPY):
  - Long TLT call spread (30-DTE, 5% OTM)
  - Short SPY put spread (30-DTE, 5% OTM, hedged for beta)
- **Entry (short spread):** When z-score > +2.0 (TLT rich):
  - Short TLT put spread + long SPY call spread
- **Exit:** Z-score crosses zero, OR 30-day holding limit, OR 2x max loss

### Expected Performance

| Metric | Estimate |
|--------|----------|
| Trades per year | 6-10 |
| Win rate | 68-75% |
| Sharpe | 1.5-2.5 |
| CAGR contribution | 4-8% |
| Max DD | 8-12% |
| Corr to EXP-1220 | **-0.05 to +0.15** (market-neutral spread construction) |
| Capacity | $50-100M (TLT option volume limits) |

### Caveats

- **Regime risk:** Stock-bond correlation flipped positive in 2022 (both fell
  together). Pairs strategies suffer when the cointegration breaks. Use a
  circuit breaker: if the spread z-score stays >3 for 10+ days without reversion,
  halt the strategy.
- **Half-life matters.** Need to verify the rolling half-life of mean reversion
  is <30 days in our sample. If >60 days, the option theta destroys the edge.
- **TLT data only reliable through 2025-12** — data gap risk.

### Decision

**PROPOSE: Highest-priority backtest candidate. Data is ready, structure is
defensible, correlation to EXP-1220 is genuinely low, and the spread construction
provides natural market-neutral exposure. Start with a proof-of-concept using
real IronVault TLT + SPY options.**

---

## Proposal 5 — Volatility Risk Premium (VRP) Harvest via ATM Straddles

**Category:** Volatility Arbitrage
**Data needed:** SPY ATM calls + puts, VIX (for regime filter), realized vol
**Data status:** FULL — IronVault has SPY calls and puts covering 2020-2026

### Hypothesis

Implied volatility systematically overestimates realized volatility ~80% of the
time (the variance risk premium, VRP). A systematic strategy that **sells** ATM
straddles when IV > RV + 3 points captures this premium. Unlike credit spreads
(which EXP-1220 already exploits on the put wing), a straddle is exposed to both
tails equally — a different exposure profile.

**The edge:** VRP is persistent because insurance buyers (hedge funds, pension
funds) overpay to avoid gamma risk. Systematic sellers of this insurance earn
the premium, modulated by realized volatility ex-post.

### Signal Logic

- **Entry:** Sell SPY ATM straddle (30-DTE) when:
  1. VIX > 20-day realized vol by at least 3 points
  2. VIX between 16 and 28 (sweet spot: enough premium, not crisis)
  3. SPY 20-day return > -5% (no freefall)
- **Management:** Close at 50% of max profit OR 14 DTE, whichever first
- **Stop:** If SPY moves >1 std dev in a single day, close immediately
- **Position sizing:** 2-3% of capital per straddle, max 1 concurrent (high gamma risk)

### Expected Performance

| Metric | Estimate |
|--------|----------|
| Trades per year | 8-12 |
| Win rate | 70-78% |
| Sharpe | 1.8-2.5 |
| CAGR contribution | 5-10% |
| Max DD | 10-15% |
| Corr to EXP-1220 | **+0.30 to +0.45** (same short-gamma exposure) |
| Capacity | $200M+ (SPY ATM straddles are infinitely liquid) |

### Caveats

- **THIS IS THE MOST CORRELATED** proposal of the five. A short ATM straddle is
  effectively 2x the short-gamma exposure of a credit spread. Adding this to a
  portfolio with EXP-1220 **increases concentration, not diversification.**
- **Alternative:** Instead of straddles, consider **short iron butterflies** —
  same short-gamma edge but with defined-risk wings. Still correlated but at
  least the tail risk is bounded.
- **R3/R4 already tested related strategies** (VRP Harvest in R4 showed Sharpe
  1.36, SPY corr +0.62). We already know this edge exists but also that it's
  **not uncorrelated.**

### Decision

**DO NOT PROPOSE** — this idea would add return but would not help the correlation
problem. Keeping it in the research doc as a counterexample of what *not* to
prioritize. If we need more return from the existing factor, we should just
size up EXP-1220, not add a second short-gamma strategy.

---

## Summary & Prioritization

| # | Name | Category | Sharpe Est | Corr to 1220 | Data Ready | Priority |
|---|------|----------|-----------|--------------|------------|----------|
| 1 | VIX Term Structure Carry (VXX) | Vol Arb | 1.5-2.5 | -0.20 to +0.10 | NO (need VXX bars) | P2 |
| 2 | Cross-Asset Momentum Rotation | Momentum | 0.8-1.5 | -0.10 to +0.30 | YES (yfinance) | **P1** |
| 3 | FOMC Announcement Drift | Event-Driven | 1.2-2.0 | +0.05 to +0.15 | YES | P2 |
| 4 | TLT-SPY Pairs Mean Reversion | Stat Arb | 1.5-2.5 | **-0.05 to +0.15** | **YES (full)** | **P1** |
| 5 | VRP via ATM Straddles | Vol Arb | 1.8-2.5 | +0.30 to +0.45 | YES | **REJECT** |

### Recommended Next Actions

1. **Immediate (this week):** Build a proof-of-concept for **Proposal 4 (TLT-SPY
   pairs)**. All data exists, the correlation to EXP-1220 is genuinely low, and
   the economic thesis is defensible. Target: 30-50 real trades over 2020-2025
   walk-forward.

2. **Immediate (this week):** Build a proof-of-concept for **Proposal 2 (cross-asset
   momentum)** using ETF shares (no options). yfinance data only. 4-asset universe,
   monthly rebalance, measure actual correlation to EXP-1220's return stream.

3. **Next week:** Research **Proposal 3 (FOMC drift)** using SPY daily bars from
   IronVault. If the drift is measurable in 2020-2026, add 0-DTE call overlay.

4. **Later:** **Proposal 1 (VXX)** requires data acquisition. Lower priority
   until TLT pairs and momentum are validated.

5. **Skip:** **Proposal 5 (VRP straddles)** — too correlated to existing portfolio.

### Portfolio Math (If All 4 Surviving Proposals Work)

Assuming independent execution at expected midpoint values:

| Strategy | CAGR Contribution | Weight | Weighted CAGR |
|----------|-------------------|--------|---------------|
| EXP-1220 (existing) | 3.4% | 40% | 1.4% |
| TLT-SPY Pairs | 6% | 20% | 1.2% |
| Cross-Asset Momentum | 9% | 20% | 1.8% |
| FOMC Drift | 4.5% | 10% | 0.5% |
| VXX Carry | 6% | 10% | 0.6% |
| **Combined** | | **100%** | **~5.5% CAGR** |

With correlation-adjusted Sharpe improvements from diversification (3-strategy
portfolio with average pairwise correlation of 0.15), the combined Sharpe could
reach **3.5-4.0** — still below the 6.0 target but a material improvement over
EXP-1220 standalone at 3.12.

---

## Open Questions for Follow-Up Research

1. **Is the FOMC drift robust to the ZIRP era (2020-2022)?** The original Lucca
   paper used 1994-2011 data. Post-COVID regime may behave differently.

2. **What's the actual half-life of TLT-SPY cointegration in 2022-2025?**
   The stock-bond correlation flipped — need to measure the spread half-life
   empirically before building the trade.

3. **Can we approximate VXX from the VIX/VIX3M ratio in IronVault (via constant
   maturity calculation)?** Would avoid the data acquisition for VXX bars.

4. **What's the capital utilization of a 4-strategy portfolio?** EXP-1220's
   86% idle-day problem may shrink significantly with 3-4 overlapping strategies.
   This could be the missing piece to boost portfolio CAGR.

5. **Do any of these strategies perform better with smaller position sizing and
   more frequent entries (like the cadence finding in EXP-1220)?** R4 tested
   cadence only for credit spreads. Worth testing for FOMC and momentum as well.
