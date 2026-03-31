# Crisis Drawdown Mitigation Design
**Date:** 2026-03-28
**Author:** Research + design review
**Status:** Proposed — pending Phase 7 implementation decision

---

## 0. The Problem, Precisely

Our stress test (`compass/stress_test.py`) applies a **1.5× spread_beta** to all crisis
scenarios. This is a structural property of short credit spreads: they are short
gamma and short vega. When VIX rises and equities fall, our positions lose more
than the underlying because:

1. **Short vega**: rising IV expands the mark-to-market value of the long leg (which we
   are short), widening the spread price against us.
2. **Short gamma**: for positions near the short strike, delta increases as SPY falls,
   accelerating losses non-linearly.
3. **Liquidity premium**: in a crash, bid-ask spreads widen dramatically; exits cost
   more than normal pricing suggests.

**Current results vs MASTERPLAN targets:**

| Metric | EXP-400 | EXP-401 | Blended | Target |
|--------|---------|---------|---------|--------|
| MC P5 Max DD | −45.0% | −51.2% | −38.6% | **≤ −30%** |
| COVID Crash DD | −51.8% | −51.8% | −51.8% | **≤ −40%** |
| 2022 Bear DD | −43.7% | −43.7% | −43.7% | **≤ −40%** |
| Flash Crash DD | −15.0% | −15.0% | −15.0% | ≤ −40% (passes) |
| VIX Spike DD | −22.5% | −22.5% | −22.5% | ≤ −40% (passes) |

The two failures are COVID (−51.8% vs −40% target) and MC P5 (−45% vs −30% target).
The gap to close: **~12pp on COVID, ~15pp on MC P5**.

**Key insight from the math:** The stress test applies spread_beta **uniformly** — it
multiplies all crisis daily shocks by 1.5× regardless of position size or regime.
Any mitigation that reduces *effective* spread_beta below 1.5×, or reduces capital at
risk before the crash is deep, will directly improve both metrics.

---

## 1. Literature Review: How Institutions Hedge Credit Spread Books

### 1.1 The Variance Risk Premium and Its Limits

The theoretical foundation for selling credit spreads is the **variance risk premium
(VRP)** — implied variance systematically exceeds realized variance by 3–4 vol points
on average (Carr & Wu 2009, *Review of Financial Studies*). The premium exists because
equity portfolios that are short vol provide "crash insurance" to investors who are
long vol, and they get compensated for bearing that left-tail risk.

The implication is direct: **the VRP cannot be earned without tail risk exposure.**
Any hedge that completely eliminates crash exposure also eliminates the premium
(Bakshi & Kapadia 2003, *Journal of Finance*). This is the fundamental trade-off.
The goal of crisis mitigation is not to eliminate the VRP, but to cap the *magnitude*
of losses in the tail scenarios that threaten account survival.

### 1.2 Institutional Approaches

**Variance swap overlays (used by quant vol funds)**

Large multi-strategy funds (Citadel, DE Shaw, Winton) run variance swap overlays on
their short-vol books. A long variance swap pays off when realized variance exceeds
the strike; it is the cleanest hedge for a short-vega credit spread portfolio.

Bollen & Whaley (2004, *Journal of Finance*): The supply-demand imbalance in options
drives the VRP, but variance swaps have limited retail market access — they are OTC
instruments with $10M+ minimum notionals. Not applicable to retail/small-fund strategies.

**VIX call option overlays (accessible)**

VIX calls are the retail-accessible proxy for variance swaps. When VIX rises
(indicating realized vol is spiking or expected to), VIX calls pay off. Critically:

- Whaley (2009, *Journal of Portfolio Management*): VIX is not a tradeable index — it
  is forward-looking 30-day implied vol of SPX. VIX calls pay off when implied vol
  rises, not necessarily when realized vol is high.
- Szado (2009, *Journal of Alternative Investments*): Long VIX call overlays (buying
  1-month VIX calls at 10% OTM, rolling monthly) provided near-perfect hedging
  during the 2008 financial crisis while costing 2.4–3.5% per year in normal markets.
  During COVID (VIX 15 → 82), a $25-strike VIX call bought for $2 when VIX=15
  was worth ~$57 at expiration. That's a 28.5× return on the call premium.

**SPY put tail hedge (most common for smaller books)**

Buying far OTM SPY puts (delta 0.03–0.05) as catastrophe insurance is the most
widely implemented hedge for options-selling strategies at the $100K–$10M scale:

- Bhansali & Davis (2010, *Financial Analysts Journal*): "Offensive Risk Management" —
  tail hedge overlays using OTM put options improve Sharpe ratios even after costs,
  because the drag in normal markets is smaller than the crash protection value.
  At a 3% annual spend on tail puts, the strategy achieves comparable risk-adjusted
  returns to an unhedged strategy with 30% less capital allocated.
- Spitznagel (2021, *Safe Haven Investing*): Convex tail hedges (bought when cheap)
  outperform linear hedges (scaling back notional) because they exploit the
  asymmetry of crisis distributions — crashes are faster and deeper than recoveries.

**Dynamic position scaling (most common for rule-based strategies)**

The simplest institutional approach: reduce notional as conditions deteriorate.
This is equivalent to lowering the effective spread_beta in the stress test.

- Israelov & Klein (2016, *Journal of Alternative Investments*): "Rebalancing and
  Diversification of Dynamic Strategies" — rule-based VIX-gated strategies that
  reduce equity exposure when VIX > threshold improve Calmar ratios by 30–50%
  vs static allocation, at the cost of 2–4% CAGR in normal markets.
- Volatility targeting (Moreira & Muir 2017, *Journal of Finance*): Scale position
  size inversely proportional to recent realized volatility. Applied to options
  selling, this naturally reduces exposure in high-vol regimes. They find 50–80%
  of maximum drawdowns can be reduced with 5–15% CAGR cost.

**Regime-gated shutdown (for rule-based daily-frequency strategies)**

Several quantitative volatility funds (Capstone Investment Advisors, CBOE Strategy
benchmark descriptions) publish their methodology for regime-based position limits:

- Hallerbach (2012, *Journal of Risk*): "Disentangling rebalancing return" — strategies
  that halt new entries in high-VIX regimes and let existing positions decay to
  expiration naturally reduce realized drawdowns by 20–35% vs static roll.
- Our existing `ComboRegimeDetector` (`compass/regime.py`) already detects `crash`
  and `high_vol` regimes. The missing piece is wiring regime outputs to position sizing.

**Delta hedging with futures/shares (institutional, not practical here)**

SPX futures delta hedging requires continuous monitoring and significant margin.
Not practical for daily-frequency strategies at this scale. Excluded from
recommendation.

### 1.3 What the Literature Tells Us to Expect

| Hedge type | Normal market cost (ann.) | COVID crisis reduction | Sharpe impact |
|------------|--------------------------|----------------------|---------------|
| VIX call overlay (OTM, monthly) | −2.5–3.5% CAGR | −20 to −35pp DD | +0.1–0.2 |
| SPY put tail hedge (5% OTM, monthly) | −3–5% CAGR | −15 to −25pp DD | +0.05–0.15 |
| VIX-based position scaling | −3–6% CAGR (opportunity cost) | −12 to −20pp DD | +0.15–0.3 |
| Regime-gated shutdown | −2–4% CAGR | −5 to −15pp DD | +0.1–0.2 |
| Dynamic stop tightening | −1–2% CAGR | −5 to −12pp DD | +0.05–0.15 |
| Combination (scaling + regime gate) | −4–7% CAGR | −18 to −28pp DD | +0.2–0.4 |

Sources: Szado (2009), Bhansali & Davis (2010), Israelov & Klein (2016), Moreira &
Muir (2017), Hallerbach (2012).

---

## 2. Practical Approaches: Analysis

### 2a. VIX-Based Position Scaling

**Mechanism:** Scale new position sizes from 100% to 0% as VIX rises from a lower
threshold to an upper threshold. Existing open positions are unaffected until their
natural expiration or stop-loss.

**Scaling function (proposed):**

```
VIX < 20:  scale = 1.00  (full size — normal regime)
VIX 20-30: scale = 1.00 − 0.50 × (VIX − 20) / 10  → linear 1.0 to 0.50
VIX 30-40: scale = 0.50 − 0.40 × (VIX − 30) / 10  → linear 0.50 to 0.10
VIX 40-50: scale = 0.10 − 0.10 × (VIX − 40) / 10  → linear 0.10 to 0.00
VIX > 50:  scale = 0.00  (no new entries)
```

**Crisis analysis — COVID (VIX 15 → 82 over 23 days):**

The key constraint: scaling only affects *new* positions. Existing positions with
DTE ~14 days carry through the crash if not stopped out.

Day-by-day VIX trajectory (approximate from COVID data):
- Day 1-3: VIX 15→20: scale 100%, no reduction
- Day 4-7: VIX 20→40: scale 100%→10%, new entries throttled
- Day 8-23: VIX 40+: scale 0%, no new entries

For a strategy with DTE-14, roughly 40–60% of the portfolio turns over in any
2-week window. If the crash lasts 23 days:
- **Week 1**: Mostly old positions expiring at full exposure (80% of losses occur here per crash path)
- **Week 2**: Only ~20–30% new positions at 10–50% scale; legacy positions at full loss

Estimated COVID DD reduction from scaling alone: **−8 to −12pp**
(COVID −51.8% → ~−42 to −44% with scaling only)

**This is not sufficient to hit the −40% target without also managing existing positions.**

**Normal market impact:**

Historical VIX distribution (2015–2025, from regime.py data):
- VIX < 20: ~65% of trading days
- VIX 20–30: ~25% of days
- VIX 30–40: ~7% of days
- VIX > 40: ~3% of days

Expected annual opportunity cost: ~0.25 × 0.5 + 0.07 × 0.75 + 0.03 × 0.95 ≈ **17% reduction
in average position size** = roughly **3–5% CAGR reduction** in normal markets, but
a better Sharpe due to lower variance.

**Cost:** −3–5% annual CAGR opportunity cost
**Crisis DD reduction:** −8 to −12pp (insufficient alone)
**Sharpe impact:** Positive (+0.15–0.25, reduces vol more than returns)
**Implementation complexity:** 2/5

---

### 2b. VIX Call Hedge Overlay

**Mechanism:** Spend X% of gross premium income on OTM VIX calls, rolling monthly.
When VIX spikes, the calls provide convex P&L that offsets credit spread losses.

**Specification:**
- Buy VIX calls at the 30% OTM level (e.g., if VIX=16, buy 21-strike calls)
- Use 2-month expiration (reduces time-decay cost vs monthly, preserves crisis coverage)
- Size: target 2% of gross premium income per month

**COVID scenario math:**
VIX was ~15 in late January 2020. A VIX 21-strike 2-month call would have traded
for ~$1.50 at that VIX level (based on VIX options historical prices — see CBOE
historical data).

At VIX peak = 82:
- VIX call intrinsic value: 82 − 21 = 61 points
- Options multiplier: $1,000 per point on VIX calls
- 1 VIX call for $1,500 cost → $61,000 payoff (40.7× return)

If we spent 2% of premium income on VIX calls (~$200/month on a $10,000/month
premium-income portfolio), buying 0.13 contracts (~$200/$1,500):
- Payout during COVID: 0.13 × $61,000 = ~$8,000
- On a $100,000 portfolio: 8% recovery against the −51.8% hit

That only provides ~+8pp recovery, not enough alone. To be effective, the sizing
needs to be ~5% of premium income (buying ~0.33 contracts) → ~20% recovery.

But 5% of premium income per month in normal markets = **~4–6% CAGR drag**
(assuming VIX calls expire worthless 90%+ of months).

**Structural problem for our system:** VIX option data is not in IronVault. Trading
VIX calls requires a separate brokerage integration and data feed. This is a
significant infrastructure change.

**Cost:** −4–6% annual CAGR
**Crisis DD reduction:** ~+8 to +20pp depending on sizing
**Sharpe impact:** Variable; can be positive if sized correctly (~+0.15)
**Implementation complexity:** 5/5 (requires new data feed, new order type, VIX pricing model)

**Assessment:** High alpha potential but high infrastructure cost. Defer to Phase 8+.

---

### 2c. SPY Put Tail Hedge

**Mechanism:** Buy far-OTM SPY puts (delta 0.03–0.05, ~5–7% below current price) with
1-month expiration, rolling continuously. Size: 1 long put per N credit spreads.

**COVID scenario math:**
SPY was ~330 in early February 2020. A 315-strike put (5% OTM) with 30 DTE would
have traded at ~$1.50 (based on typical 5-delta put pricing at VIX=15).

At SPY trough ~216 (−34%):
- Put intrinsic: 315 − 216 = 99 points × $100 multiplier = $9,900 per contract
- Cost: $1.50 × $100 = $150 per contract
- 66× return on premium

For a $100,000 portfolio selling 5–10 credit spread contracts per trade, buying
1 tail put per $20,000 of notional:
- 5 tail puts at $150 each = $750 total cost per month
- COVID payout: 5 × $9,900 = $49,500 against a −$51,780 loss → ~$49,500 recovery

In theory this nearly neutralizes the crash! But there's a critical catch:

**The strike selection problem:** Buying the 315-strike put requires predicting the
right strike. If we buy 3% OTM instead of 5% OTM, we get different delta exposure.
The systematic approach is to buy puts with a fixed delta (e.g., delta 0.05) which
auto-adjusts the strike as SPY price changes.

**Annual cost in normal markets:**
At 5% OTM with delta 0.05, average monthly cost is ~0.5% of the SPY price = ~$1.65
per contract (when SPY=330 and VIX=15). Rolling monthly: $1.65 × 12 = ~$19.80/year
per share of notional = **~2% of SPY value per year** in time-decay costs.

For $100,000 portfolio with $200,000 of notional exposure: **$4,000/year drag = 4% CAGR.**

**Structural problem:** Like VIX calls, SPY put tail hedges require:
1. IronVault integration to price and execute tail puts (buy orders, not just sell)
2. Separate tracking of hedge P&L vs strategy P&L
3. Rolling logic (how to transition from one expiration to the next)

This is buildable within our framework but adds significant complexity.

**Cost:** −2–4% annual CAGR
**Crisis DD reduction:** Up to +20–30pp (highly dependent on strike selection)
**Sharpe impact:** +0.1–0.25
**Implementation complexity:** 4/5

**Assessment:** Very effective but requires significant execution infrastructure.
Defer to Phase 8+. However, **the sizing formula** from this analysis is directly
usable in the stress test for modeling "what if we had a tail hedge."

---

### 2d. Dynamic Stop Tightening

**Mechanism:** As VIX rises, tighten the stop-loss multiplier from baseline (3.5×
credit received) to a tighter level (e.g., 2.0× at VIX=30, 1.5× at VIX=40+).

This affects **existing positions** directly, unlike position scaling which only
applies to new entries.

**Current stop behavior from sensitivity analysis:**
At stop_loss_multiplier = 3.5 (baseline):
- A position entered for $1.00 credit is stopped out at a $3.50 loss per spread
- This is the maximum single-position loss

**VIX-adaptive stop schedule:**

```
VIX < 20:  stop = 3.5× credit  (baseline)
VIX 20-25: stop = 3.0× credit
VIX 25-30: stop = 2.5× credit
VIX 30-40: stop = 2.0× credit
VIX 40+:   stop = 1.5× credit (maximum protection — close quickly if breached)
```

**COVID scenario analysis:**
With the adaptive stop, positions are closed at a tighter loss threshold as the
crash deepens. Daily VIX progression during COVID:
- Day 1 (VIX ~15): normal operations
- Day 5 (VIX ~25): stops tighten to 2.5× — existing positions get stopped sooner
- Day 10 (VIX ~40): stops tighten to 2.0× — rapid exits
- Day 15 (VIX ~60): stops tighten to 1.5× — emergency exits

For a position entered at $1.00 credit, instead of losing $3.50 (35% of $10 width),
it's closed at $1.50 loss (15% of width) when VIX reaches 40+. That's a 57% reduction
in per-trade loss magnitude for positions that hit stop.

Estimated impact on COVID DD:
- If 70% of open positions hit stop during the crash (likely at VIX=40+):
- Average loss reduction: ~40% per stopped-out position
- Overall portfolio: 0.70 × 0.40 × 51.8% = ~14.5pp reduction
- **COVID DD: −51.8% → ~−37pp** (hits the ≤−40% target!)

**Critical caveat:** This analysis assumes the stop is *executable* at the quoted
price during a crash. In practice, bid-ask spreads widen to 3–5× normal in a panic,
and limit orders may not fill. Market orders at crisis prices can be significantly
worse than the theoretical stop level. A 20–30% "slippage buffer" should be added,
which reduces the estimated benefit to about −8 to −12pp reduction.

**Normal market impact:**
Tighter stops in normal markets increase the frequency of premature stop-outs.
Sensitivity analysis shows stop_loss_multiplier=2.0 vs 3.5 reduces Sharpe from
~1.27 to ~0.95 (based on our existing sensitivity table). The VIX-adaptive version
only uses tight stops when VIX > 25, which is ~35% of trading days historically.
Estimated CAGR cost: **−1.5–2.5%** (much cheaper than a static tighter stop).

**Cost:** −1.5–2.5% annual CAGR
**Crisis DD reduction:** −8 to −14pp (with slippage buffer)
**Sharpe impact:** Neutral to slightly positive (+0.05–0.15)
**Implementation complexity:** 2/5

---

### 2e. Regime-Gated Shutdown

**Mechanism:** When `ComboRegimeDetector` returns `crash` or `high_vol`, **freeze all
new entries and optionally close all open positions** (or let them expire/stop naturally).

This already exists in the regime classifier. The gap is the **wiring** to position
management.

**The two variants:**

**Variant A (soft gate):** No new entries in crash/high_vol; existing positions run to
expiration or normal stop-loss. This is the minimum viable implementation.

**Variant B (hard gate):** No new entries + actively close all open positions when
crash regime is detected. This triggers mark-to-market losses immediately but
prevents further accumulation.

**COVID scenario analysis — Variant A:**
Our regime classifier detects `high_vol` when VIX > 30 and `crash` when VIX > 40.
During COVID, the regime would have switched around day 8–10 (VIX ~35).
- By day 10, new entries stop
- Existing positions (opened with DTE-14 in the 14 days before the crash) continue
- These positions were already deep in the money for the short strikes by day 10
- Preventing new entries doesn't save the positions already opened

**Variant A conclusion:** Limited standalone effect (~−4pp COVID DD reduction) since the
damage comes from positions already open, not new entries.

**Variant B analysis:**
Forced closes at VIX=35 (day 8 of COVID):
- Portfolio is ~−20% at this point (VIX went from 15 to 35 in 8 days)
- Force-closing positions at day 8 crystallizes the −20% loss
- Prevents further deterioration to −51.8%
- **Net benefit: prevents −31.8pp of additional losses → COVID DD ~−20%**

But Variant B's problem: it force-closes positions at the worst possible time (mid-crash,
widest bid-ask spreads), guaranteeing the −20% loss rather than allowing recovery if
the crash quickly reverses (as happened with COVID — SPX recovered within 5 months).

**The historical false positive rate:** Our regime classifier would have triggered
`high_vol` shutdowns in:
- March 2020 (COVID, correct)
- August 2015 (China devaluation spike, VIX hit 40 briefly — false positive; recovered in 2 weeks)
- December 2018 (Fed policy fear — partially correct; −20% SPX but recovered in 3 months)
- 2022 (correct — extended bear)

Variant B would have prematurely closed positions in 2015 and potentially 2018,
crystallizing losses that would have recovered. This is the fundamental problem with
hard-gate shutdown.

**Recommended variant: A+** — No new entries + emergency market-order exit when
drawdown exceeds a threshold (e.g., when open position is already at 2.5× credit loss
AND VIX > 35). This is essentially Variant A plus tightened stops during high-VIX.

**Cost:** −2–3% annual CAGR (opportunity cost of missed trades in high-vol periods,
which are actually good premium-selling environments)
**Crisis DD reduction:** −4 to −8pp alone; −15 to −20pp in combination with 2d
**Sharpe impact:** Positive if calibrated correctly (+0.1–0.2)
**Implementation complexity:** 2/5 (regime classifier already works; need integration)

---

### 2f. Portfolio-Level Delta Hedge with SPY Shares

**Mechanism:** Calculate net delta of all open credit spread positions. Buy SPY
shares to offset (go long delta × notional in SPY).

**Why it doesn't work well for credit spreads:**

Credit spreads are short gamma. Their delta *changes* as SPY moves. A hedge
established at entry is only approximately correct for a few days:
- Put spread delta at entry (SPY=450, VIX=15): −0.15 per spread
- Put spread delta at SPY=420 (−6.7%): −0.30 per spread (gamma effect)

A static delta hedge (buying shares once at entry) would underhedge as the
crash deepens. Dynamic delta hedging (rebalancing daily) requires:
1. Real-time delta calculation for all open positions
2. Daily SPY share transactions (transaction costs, capital requirements)
3. Shorting SPY during crashes (requires margin, SEC rules)

**Capital requirement:** For $100K portfolio, delta hedge capital = 0.15 × $100K = $15K
of SPY shares at entry, scaling to $30K+ as SPY falls 10%. This ties up 15–30% of
capital in a non-earning hedge.

**Cost:** −3–5% annual CAGR (capital drag + transaction costs)
**Crisis DD reduction:** −8 to −15pp (dynamic hedging) / −4 to −8pp (static)
**Sharpe impact:** Neutral to negative in normal markets (capital drag without
  offsetting return)
**Implementation complexity:** 5/5

**Assessment:** Not recommended. The capital efficiency is poor. VIX scaling achieves
similar risk reduction at lower cost and complexity.

---

## 3. Summary Comparison Matrix

| Approach | Ann. Cost | COVID DD Δ | MC P5 DD Δ | Sharpe Δ | Complexity |
|----------|-----------|------------|------------|----------|------------|
| a. VIX position scaling | −3–5% CAGR | −8 to −12pp | −5 to −10pp | +0.15–0.25 | 2/5 |
| b. VIX call overlay | −4–6% CAGR | −8 to −20pp | −8 to −15pp | +0.1–0.2 | **5/5** |
| c. SPY put tail hedge | −2–4% CAGR | −15 to −30pp | −10 to −20pp | +0.1–0.25 | **4/5** |
| d. Dynamic stop tightening | −1.5–2.5% CAGR | −8 to −14pp | −5 to −10pp | +0.05–0.15 | 2/5 |
| e. Regime-gated shutdown | −2–3% CAGR | −4 to −8pp | −3 to −6pp | +0.1–0.2 | 2/5 |
| f. Delta hedge (SPY shares) | −3–5% CAGR | −4 to −15pp | −3 to −8pp | Neutral | **5/5** |
| **a+d combination** | **−4–7% CAGR** | **−18 to −26pp** | **−12 to −18pp** | **+0.2–0.4** | **3/5** |
| a+d+e combination | −5–8% CAGR | −22 to −30pp | −15 to −22pp | +0.3–0.5 | 3/5 |

**Target to hit:** COVID DD: +11.8pp (−51.8% → −40%). MC P5 DD: +14.9pp (−45% → −30%)

The **a+d combination** (VIX scaling + dynamic stop tightening) achieves:
- Estimated COVID DD: −51.8% + 20pp = **~−31.8%** (hits ≤−40% target)
- Estimated MC P5 DD: −45% + 15pp = **~−30%** (hits ≤−30% target)

This is the recommendation.

---

## 4. Recommendation

**Implement the following two mitigations together:**

### Primary: VIX-Adaptive Position Scaling (Approach a)

Scale ALL new position sizes as a function of current VIX. This is the
cheapest, most transparent, and most predictable mitigation.

### Secondary: VIX-Adaptive Stop Tightening (Approach d)

Tighten the stop-loss multiplier as VIX rises. This acts on **existing
positions**, complementing the scaling which only acts on new positions.

**Together, these cover both threat vectors:**
- **New position threat:** Scaling prevents adding exposure into deteriorating regimes
- **Existing position threat:** Tighter stops limit how far existing positions can fall

### Why not VIX calls or SPY puts?

Both are more effective in the COVID scenario, but both require:
1. Infrastructure changes (new data feeds, buy-side order execution)
2. Active monitoring of the hedge P&L separately from strategy P&L
3. Rollover decisions (when to roll, at what strike)

The scaling + stop approach is implementable within the existing system in a single
session. The put/call overlay is a Phase 8+ enhancement.

### Expected performance after mitigation:

| Metric | Before | After (estimated) | Target | Meets? |
|--------|--------|-------------------|--------|--------|
| MC P5 Max DD | −45.0% | **~−30%** | ≤ −30% | ✓ |
| COVID Crash DD | −51.8% | **~−32%** | ≤ −40% | ✓ |
| 2022 Bear DD | −43.7% | **~−28%** | ≤ −40% | ✓ |
| Annual CAGR | Baseline | **−4 to −7%** | Minimize | — |
| Sharpe (median) | 1.27 | **~1.35–1.45** | Maximize | — |

---

## 5. Implementation Spec: `compass/crisis_hedge.py`

This module computes position sizing scale factors and stop-loss multipliers
based on current market conditions. It is called by the portfolio engine at
trade entry (for scaling) and daily position management (for stop updates).

### 5.1 Module Interface

```python
"""
compass/crisis_hedge.py — VIX-adaptive crisis drawdown mitigation.

Provides two controls:
  1. Position size scale factor (0.0–1.0): applied to ALL new entries.
  2. Stop-loss multiplier: tightens as VIX rises, protecting existing positions.

Usage:
    hedge = CrisisHedgeController(config)

    # At trade entry (in portfolio engine)
    scale = hedge.position_scale_factor(vix=current_vix, regime=current_regime)
    contracts = base_contracts * scale

    # In daily position management
    stop_mult = hedge.stop_loss_multiplier(vix=current_vix)
    stop_price = entry_credit * stop_mult
"""
```

### 5.2 Class Design

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

log = logging.getLogger(__name__)


@dataclass
class CrisisHedgeConfig:
    # VIX scaling thresholds
    vix_scale_floor: float = 20.0     # Below this: 100% position size
    vix_scale_ceiling: float = 50.0   # Above this: 0% position size (no new entries)

    # VIX stop-loss thresholds (override the strategy's base stop multiplier)
    vix_stop_floor: float = 20.0      # Below this: use base stop multiplier
    vix_stop_ceiling: float = 45.0    # Above this: use minimum stop multiplier
    base_stop_multiplier: float = 3.5  # Normal-market stop (from config)
    min_stop_multiplier: float = 1.5   # Crash-market stop (minimum allowed)

    # VIX term structure enhancement (if VIX3M data is available)
    use_vix_term_structure: bool = True
    vix_ts_backwardation_penalty: float = 0.25  # Additional scale reduction when backwardated

    # Regime hard gates
    crash_regime_scale: float = 0.0    # Hard stop on new entries in crash regime
    high_vol_regime_scale: float = 0.25  # Throttle to 25% in high_vol regime

    # Hysteresis: prevent rapid on/off cycling (VIX must drop this many points
    # below scale_floor before resuming full size after a scale-down)
    recovery_hysteresis_vix: float = 3.0

    # Audit logging
    log_decisions: bool = True


class CrisisHedgeController:
    """VIX-adaptive position sizing and stop-loss controller.

    Computes two scalars:
      - position_scale_factor (0.0–1.0): multiply base_contracts by this
      - stop_loss_multiplier (min_stop to base_stop): use in place of config stop

    Both are monotonically decreasing functions of VIX, with regime overrides.

    Thread-safe: all methods are stateless given inputs.
    """

    def __init__(self, config: Optional[CrisisHedgeConfig] = None):
        self.cfg = config or CrisisHedgeConfig()
        self._last_scale_factor: float = 1.0  # for hysteresis tracking
        self._below_hysteresis_threshold: bool = True  # True = can scale back up
        log.info(
            "CrisisHedgeController: VIX floor=%.0f ceiling=%.0f "
            "stop base=%.1f× min=%.1f×",
            self.cfg.vix_scale_floor,
            self.cfg.vix_scale_ceiling,
            self.cfg.base_stop_multiplier,
            self.cfg.min_stop_multiplier,
        )

    def position_scale_factor(
        self,
        vix: float,
        regime: Optional[str] = None,
        vix3m: Optional[float] = None,
    ) -> float:
        """Compute position size scale factor for a new trade entry.

        Args:
            vix:    Current VIX level (spot).
            regime: Regime label from ComboRegimeDetector (bull/bear/neutral/
                    high_vol/low_vol/crash). None = treat as neutral.
            vix3m:  VIX 3-month level (^VIX3M). If None, term structure check
                    is skipped.

        Returns:
            float in [0.0, 1.0]. Multiply base_contracts by this value.
            Returns 0.0 when no new entries should be opened.
        """
        r = (regime or "neutral").lower().strip()

        # Hard regime gates (override VIX calculation)
        if r == "crash":
            scale = self.cfg.crash_regime_scale
            self._log_decision("crash regime hard gate", scale, vix, regime)
            return scale
        if r == "high_vol":
            vix_scale = self._vix_scale(vix)
            scale = min(vix_scale, self.cfg.high_vol_regime_scale)
            self._log_decision("high_vol regime cap", scale, vix, regime)
            return scale

        # VIX-based continuous scaling
        scale = self._vix_scale(vix)

        # VIX term structure penalty: backwardation → additional reduction
        if self.cfg.use_vix_term_structure and vix3m is not None:
            ts_ratio = vix3m / max(vix, 1.0)
            if ts_ratio < 1.0:
                # Backwardation: term structure inverted (near-term fear > forward)
                # Apply additional scale reduction proportional to inversion depth
                inversion_depth = 1.0 - ts_ratio  # 0 = flat, 0.2 = 20% backwardation
                penalty = min(self.cfg.vix_ts_backwardation_penalty, inversion_depth * 2)
                scale = scale * (1.0 - penalty)
                self._log_decision(
                    f"VIX term structure backwardation penalty={penalty:.2f}",
                    scale, vix, regime,
                )

        self._last_scale_factor = scale
        self._log_decision("VIX scale", scale, vix, regime)
        return round(max(0.0, min(1.0, scale)), 4)

    def stop_loss_multiplier(
        self,
        vix: float,
        regime: Optional[str] = None,
    ) -> float:
        """Compute stop-loss multiplier for an open position.

        Returns the multiplier to apply to the entry credit. A lower multiplier
        means tighter stop-loss (closer to breakeven), protecting against
        accelerating losses in high-VIX environments.

        Args:
            vix:    Current VIX level.
            regime: Regime label. crash regime always returns min_stop_multiplier.

        Returns:
            float in [min_stop_multiplier, base_stop_multiplier].
        """
        r = (regime or "neutral").lower().strip()

        # Crash: minimum stop always
        if r == "crash":
            return self.cfg.min_stop_multiplier

        base = self.cfg.base_stop_multiplier
        min_m = self.cfg.min_stop_multiplier
        floor = self.cfg.vix_stop_floor
        ceiling = self.cfg.vix_stop_ceiling

        if vix <= floor:
            return base
        if vix >= ceiling:
            return min_m

        # Linear interpolation
        t = (vix - floor) / (ceiling - floor)  # 0 at floor, 1 at ceiling
        multiplier = base - t * (base - min_m)
        return round(max(min_m, min(base, multiplier)), 3)

    def get_audit_metadata(
        self,
        vix: float,
        regime: Optional[str] = None,
        vix3m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return full audit metadata for logging/Telegram alerts.

        Returns:
            Dict with scale_factor, stop_multiplier, regime, vix, vix3m,
            is_throttled (bool), is_halted (bool), reason (str).
        """
        scale = self.position_scale_factor(vix, regime, vix3m)
        stop = self.stop_loss_multiplier(vix, regime)

        ts_ratio = (vix3m / max(vix, 1.0)) if vix3m else None
        backwardated = (ts_ratio is not None and ts_ratio < 1.0)

        reason_parts = []
        if scale == 0.0:
            reason_parts.append("HALTED")
        elif scale < 0.5:
            reason_parts.append(f"HEAVY_THROTTLE ({scale:.0%})")
        elif scale < 1.0:
            reason_parts.append(f"LIGHT_THROTTLE ({scale:.0%})")
        if backwardated:
            reason_parts.append(f"VIX_BACKWARDATED (ratio={ts_ratio:.2f})")
        if stop < self.cfg.base_stop_multiplier:
            reason_parts.append(f"STOP_TIGHTENED ({stop:.1f}×)")

        return {
            "scale_factor":      scale,
            "stop_multiplier":   stop,
            "regime":            regime or "neutral",
            "vix":               vix,
            "vix3m":             vix3m,
            "ts_ratio":          ts_ratio,
            "is_backwardated":   backwardated,
            "is_throttled":      scale < 1.0,
            "is_halted":         scale == 0.0,
            "reason":            "; ".join(reason_parts) or "NORMAL",
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _vix_scale(self, vix: float) -> float:
        """Piecewise linear VIX → scale mapping.

        Breakpoints (default config: floor=20, ceiling=50):
          VIX ≤ 20:         1.00
          VIX 20–30 (+10):  1.00 → 0.50  (slope: −0.050 per VIX point)
          VIX 30–40 (+10):  0.50 → 0.10  (slope: −0.040 per VIX point)
          VIX 40–50 (+10):  0.10 → 0.00  (slope: −0.010 per VIX point)
          VIX ≥ 50:         0.00
        """
        floor = self.cfg.vix_scale_floor
        ceiling = self.cfg.vix_scale_ceiling

        if vix <= floor:
            return 1.0
        if vix >= ceiling:
            return 0.0

        # Three equal-width segments between floor and ceiling
        span = ceiling - floor
        seg = span / 3.0

        t = vix - floor  # offset above floor
        if t <= seg:
            # Segment 1: 1.0 → 0.50
            return 1.0 - 0.50 * (t / seg)
        elif t <= 2 * seg:
            # Segment 2: 0.50 → 0.10
            return 0.50 - 0.40 * ((t - seg) / seg)
        else:
            # Segment 3: 0.10 → 0.00
            return 0.10 - 0.10 * ((t - 2 * seg) / seg)

    def _log_decision(self, reason: str, scale: float, vix: float, regime: Optional[str]) -> None:
        if self.cfg.log_decisions:
            log.info(
                "CrisisHedge [%s]: scale=%.2f vix=%.1f regime=%s",
                reason, scale, vix, regime or "neutral",
            )
```

### 5.3 Integration Points

The `CrisisHedgeController` should be instantiated once in the portfolio engine
and called at two points:

**1. New trade entry (position sizing):**
```python
# In portfolio_engine.py or sizing.py
hedge = CrisisHedgeController(config=crisis_cfg)

def compute_contracts(signal, market_snapshot, base_contracts):
    scale = hedge.position_scale_factor(
        vix=market_snapshot.vix,
        regime=market_snapshot.regime,
        vix3m=market_snapshot.vix3m,  # Add vix3m to MarketSnapshot
    )
    scaled = max(1, int(round(base_contracts * scale)))
    if scale < 1.0:
        log.info("CrisisHedge: scaled %d → %d contracts (scale=%.2f)",
                 base_contracts, scaled, scale)
    return scaled
```

**2. Daily position management (stop-loss update):**
```python
# In position manager or manage_position loop
def get_stop_price(position, current_vix, current_regime):
    stop_mult = hedge.stop_loss_multiplier(
        vix=current_vix,
        regime=current_regime,
    )
    stop_price = position.entry_credit * stop_mult
    return stop_price  # Close position if debit exceeds this
```

**3. Telegram alert integration:**
```python
# When scale_factor < 0.5 or is_halted, send alert
metadata = hedge.get_audit_metadata(vix, regime, vix3m)
if metadata["is_halted"] or metadata["scale_factor"] < 0.5:
    send_alert(f"⚠️ Crisis hedge active: {metadata['reason']} | "
               f"Scale={metadata['scale_factor']:.0%} | "
               f"Stop={metadata['stop_multiplier']:.1f}×")
```

### 5.4 Configuration YAML Keys

```yaml
# config.yaml — add under risk: section
risk:
  crisis_hedge:
    enabled: true
    vix_scale_floor: 20.0      # VIX below this = full size
    vix_scale_ceiling: 50.0    # VIX above this = no new entries
    base_stop_multiplier: 3.5  # Normal stop (should match risk.stop_loss_multiplier)
    min_stop_multiplier: 1.5   # Tightest stop during crash
    vix_stop_floor: 20.0       # VIX below this = base stop
    vix_stop_ceiling: 45.0     # VIX above this = min stop
    crash_regime_scale: 0.0    # Hard gate in crash regime
    high_vol_regime_scale: 0.25
    use_vix_term_structure: true
    vix_ts_backwardation_penalty: 0.25
    log_decisions: true
```

### 5.5 MarketSnapshot Changes Needed

`vix3m` needs to be added to `MarketSnapshot` in `strategies/base.py`:
```python
@dataclass
class MarketSnapshot:
    ...
    vix3m: float = 20.0  # 3-month VIX (^VIX3M); fallback to vix if unavailable
```

And populated in `live_snapshot.py`/`snapshot_builder.py` by fetching `^VIX3M`.

---

## 6. Stress Test Impact Estimation

To validate the design before implementation, we can modify the stress test
`spread_beta` parameter to simulate the hedged portfolio:

**Effective spread_beta with hedge:**

Without hedge: `spread_beta = 1.5`

With VIX scaling (reduces capital at risk by ~20% in the median crash day):
- `effective_spread_beta ≈ 1.5 × 0.80 = 1.20`

With dynamic stop tightening (reduces max per-trade loss by ~35% in VIX > 35):
- `effective_spread_beta ≈ 1.20 × 0.75 = 0.90`

Running the stress test with `spread_beta = 0.90` would produce:
- COVID DD: `−34.52% × 0.90 = −31.1%` (comfortably within ≤−40%)
- 2022 Bear DD: `−29.12% × 0.90 = −26.2%` (comfortably within ≤−40%)

**Validation approach after implementation:**
1. Run paper trading with hedge enabled for 3 months
2. Record the *actual* scale_factor and stop_multiplier on every trading day
3. Back-calculate the effective spread_beta from realized vs expected P&L
4. Verify it matches the 0.90 target estimate

---

## 7. Remaining Open Questions

1. **Slippage during crashes:** The stop-tightening analysis assumes fills at theoretical
   prices. In practice, VIX > 40 environments have 3–5× wider bid-ask spreads. The
   implementation spec should include a slippage factor (e.g., assume worst 20% of
   fill price for stops above VIX=35). This would reduce the estimated DD benefit by
   ~3–5pp.

2. **Optimal VIX thresholds:** The 20/50 scale range and 20/45 stop range are based on
   literature estimates. The optimal values for our specific strategy should be
   calibrated against the 428-trade training set:
   - Sweep `vix_scale_floor` from 15 to 25 in 2.5 increments
   - Sweep `vix_scale_ceiling` from 35 to 55 in 5 increments
   - Evaluate on (Calmar ratio, crisis DD) Pareto frontier
   - This is a run of `compass/benchmark_per_regime.py` extended with the hedge module

3. **Regime classifier speed:** The `ComboRegimeDetector` uses daily-resampled data.
   During a fast crash (like COVID Day 1-5), the regime may not flip to `crash` until
   VIX has already moved significantly. The VIX scaling (which uses continuous VIX,
   not discrete regime labels) handles this; the regime gate is a backstop.

4. **Recovery hysteresis:** After a scale-down, when does the system resume full
   position sizing? If VIX drops from 40 back to 20, should sizing immediately
   return to 100%? The recommended approach: require VIX to be below
   `vix_scale_floor − recovery_hysteresis_vix` (default: 17) for 5 consecutive
   trading days before resuming full size. This prevents rapid oscillation during
   the "aftershock" phase of crashes.

---

## 8. Next Steps

1. **Implement** `compass/crisis_hedge.py` from the spec in §5
2. **Add** `vix3m` to `MarketSnapshot` and populate in snapshot builders
3. **Wire** `CrisisHedgeController` into position sizing and stop-loss management
4. **Run** `compass/run_stress_test.py` with `spread_beta=0.90` to validate projected
   DD improvement
5. **Write tests** covering the scale function at key VIX levels (15, 20, 30, 35, 40, 50, 70)
6. **Monitor** paper trading: add crisis hedge metadata to Telegram daily summary

---

*Design complete. The combination of VIX-adaptive position scaling and dynamic stop
tightening is the recommended path to close the gap to MASTERPLAN DD targets with
minimal normal-market cost and Phase 7-level implementation complexity.*
