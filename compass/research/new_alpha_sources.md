# New Alpha Sources for Credit Spread Ensemble Model
**Date:** 2026-03-28
**Author:** Research review
**Scope:** Feature candidates for `compass/features.py` to improve ensemble model AUC
**Baseline model:** EnsembleSignalModel (XGBoost + RF + ET), walk-forward, 428-trade dataset

---

## Context: What We Already Have

The current `FeatureEngine` (`compass/features.py`) covers:

| Category | Features |
|----------|----------|
| Technical | RSI-14, MACD, Bollinger %B, ATR, volume ratio, returns (5/10/20d), SMA distances |
| Volatility | RV (10/20/60d), IV rank, IV percentile, RV−IV spread, put/call skew ratio, skew steepness |
| Market | VIX level, VIX change (1d/5d), **put/call ratio (placeholder — always 1.0)**, SPY returns |
| Trade structure | credit_to_width_ratio |
| Event risk | Days to earnings/FOMC/CPI, event risk score |
| Seasonal | day_of_week, OPEX week flag, Monday flag, month-end flag |
| Regime | Regime ID, confidence, duration, one-hot encoded (4 regimes) |

**Critical gap:** `put_call_ratio` is hardcoded to `1.0` — this is the single biggest low-hanging fruit.

---

## 1. Order Flow Signals

### 1.1 CBOE Total Put/Call Ratio (the real one)

**Signal description:**
The CBOE publishes the aggregate equity put/call ratio and index put/call ratio daily. The equity PCR measures retail/hedging demand for puts vs calls; the index PCR (SPX, SPY) captures institutional hedging. High PCR (>1.3) signals fear/hedging; low PCR (<0.7) signals complacency. For credit spread sellers, the PCR is a two-sided signal: high PCR inflates put premiums (good for put spread sellers) but also signals elevated crash risk.

**Academic evidence:**
- Pan & Poteshman (2006, *Review of Financial Studies*): "The information in option volume for future stock prices" — equity put volume predicts next-day returns with a 2-day lead. Cross-sectional alpha from signed put flow is significant at 0.5% weekly.
- Cremers & Weinbaum (2010, *Journal of Financial and Quantitative Analysis*): Deviations of implied volatility between calls and puts (put-call parity violations) predict stock returns; spreads persist 5–10 days.
- Johnson & So (2012, *Journal of Financial Economics*): Options-to-stock volume ratio predicts negative earnings surprises; elevated put volume 3 days before earnings is predictive.

**Implementation:**
CBOE publishes daily data at `https://www.cboe.com/us/options/market_statistics/daily/`. The free CSV is updated end-of-day and goes back to 2006. Three series matter:
1. `EQUITY_PC_RATIO` (retail sentiment)
2. `INDEX_PC_RATIO` (institutional hedging)
3. `TOTAL_PC_RATIO` (aggregate)

Features to engineer:
```python
# Raw level
put_call_ratio_equity       # CBOE equity PCR
put_call_ratio_index        # CBOE index PCR

# Normalized against own history
put_call_ratio_5d_zscore    # (pcr - mean_20d) / std_20d — regime-invariant
put_call_ratio_percentile_63d  # Percentile rank over 63 trading days

# Signal direction
put_call_ratio_trend        # (pcr_today - pcr_5d_ago)  — rising fear vs falling fear
```

**Implementation difficulty:** 2/5
**Alpha potential:** High — this is the most glaring gap in the current feature set
**Data availability:** Free (CBOE daily CSV), no API key needed; ~1MB/year

---

### 1.2 Options Volume Skew — Strike-Level Put/Call Imbalance

**Signal description:**
Beyond the aggregate PCR, strike-level volume imbalance reveals where participants are positioned. For SPY, tracking the volume ratio of OTM puts (delta 0.10–0.25) to OTM calls at the same delta reveals tail-hedge demand independently of directional bets. Elevated OTM put volume at specific strikes sometimes clusters around technical support levels that participants are defending.

**Academic evidence:**
- Easley, O'Hara & Srinivas (1998, *Journal of Finance*): "Option volume and stock prices: evidence on where informed traders trade." Informed traders prefer options before large moves; directional volume imbalance (puts vs calls) leads price by 1–2 days.
- Cao, Chen & Griffin (2005, *Journal of Finance*): Options volume in advance of large price moves (pre-event leakage study) — heavy OTM put volume 1–5 days before major index drops.

**Implementation:**
Requires options chain data (IronVault or Polygon). If we have the chain at entry, compute:
```python
# Delta-bucketed volume ratio
atm_put_volume   = sum(volume where delta in [-0.55, -0.45])
atm_call_volume  = sum(volume where delta in [+0.45, +0.55])
otm_put_volume   = sum(volume where delta in [-0.25, -0.10])
otm_call_volume  = sum(volume where delta in [+0.10, +0.25])

put_call_vol_atm   = atm_put_volume / (atm_call_volume + 1)
put_call_vol_otm   = otm_put_volume / (otm_call_volume + 1)
tail_demand_ratio  = otm_put_volume / (atm_put_volume + 1)  # "wing buying"
```

**Implementation difficulty:** 3/5 (need to query chain at entry; Polygon gives this for $29/mo)
**Alpha potential:** Medium — useful for sizing, not directional prediction
**Data availability:** IronVault (existing), Polygon options data ($29/mo tier)

---

### 1.3 Unusual Options Activity (UOA) Score

**Signal description:**
UOA detects when a single option contract trades at anomalously high volume relative to open interest. Institutional block trades often appear as 10x+ volume/OI ratio at a specific strike. For index options (SPY/SPX), large block put buying at below-market strikes is a leading indicator of institutional hedging or informed trading. The signal is more about timing (don't sell premium when whales are buying protection) than direction.

**Academic evidence:**
- Chakravarty, Gulen & Mayhew (2004, *Journal of Finance*): "Informed trading in stock and option markets" — options order flow has significant price discovery content for individual stocks; effect is weaker for index options but still detectable.
- Muravyev, Pearson & Broussard (2013, *Journal of Financial Economics*): Options market makers' quotes reflect information from order flow within 15 minutes; large trades move the term structure.

**Implementation:**
```python
# UOA score for SPY options (compute at trade entry)
for each strike in options_chain:
    volume_oi_ratio = volume / (open_interest + 1)
    if volume_oi_ratio > 5.0:  # anomalous threshold
        uoa_count += 1
        if option_type == 'put':
            uoa_put_notional += volume * contract_multiplier * premium

uoa_score = uoa_count / len(options_chain)          # breadth
uoa_put_dominance = uoa_put_notional / total_notional  # directionality
```

**Implementation difficulty:** 3/5
**Alpha potential:** Medium
**Data availability:** IronVault / Polygon; Unusual Whales API offers pre-computed UOA ($30/mo)

---

## 2. Cross-Asset Signals

### 2.1 VIX Term Structure (Contango/Backwardation) — HIGH PRIORITY

**Signal description:**
The relationship between spot VIX and VIX futures (VX front-month, VX second-month) is one of the most powerful regime signals available. In contango (VX2 > VX1 > spot), the market is calm and premium sellers have edge — realized vol consistently underdelivers. In backwardation (spot VIX > VX1 > VX2), the market is fearful about the near term, and selling options carries elevated realized loss risk.

The ratio `VIX / VIX3M` (spot vs 3-month forward vol) captures this in a single scalar and is closely related to the "variance risk premium" (VRP), which is the most academically validated signal for options selling strategies.

**Academic evidence:**
- Carr & Wu (2009, *Review of Financial Studies*): "Variance risk premiums" — sellers of variance earn a consistent premium because implied variance exceeds realized variance; the premium is time-varying and predictable from VIX futures slope.
- Simon (2003, *Journal of Futures Markets*): VIX futures term structure slope predicts SPX returns 1–4 weeks forward; backwardation precedes 2–3x the loss rate of contango periods.
- Koijen, Moskowitz, Pedersen & Vrugt (2018, *Journal of Financial Economics*): "Carry" in volatility is positive on average; VIX term structure slope is the options-seller analog of bond carry.
- Avellaneda & Cont (2021, *SSRN*): VIX3M/VIX ratio above 1.05 identifies the best risk-adjusted windows for variance selling; below 0.95 signals regime change risk.

**Implementation:**
```python
# VIX term structure — both CBOE products are free
vix_spot   = fetch('^VIX')      # spot 30-day implied vol
vix3m      = fetch('^VIX3M')    # 93-day implied vol
vxv        = fetch('^VXV')      # alias for VIX3M in some providers

vix_contango_ratio   = vix3m / vix_spot        # >1.0 = contango = good for sellers
vix_ts_slope         = vix3m - vix_spot        # absolute point spread
vix_ts_percentile    = percentile_rank(vix_contango_ratio, 252)  # 1-year window

# Regime-level interaction: contango + low VIX = maximum edge window
contango_and_low_vix = 1.0 if (vix_contango_ratio > 1.05 and vix_spot < 20) else 0.0
```

**VIX3M is available free via yfinance ticker `^VIX3M`.** No extra API cost.

**Implementation difficulty:** 1/5 — trivially add to `compute_market_features()`
**Alpha potential:** High — arguably the most important single feature we're missing
**Data availability:** Free (Yahoo Finance, `^VIX3M` / `^VXV`)

---

### 2.2 High-Yield Credit Spread (HYG/LQD Ratio)

**Signal description:**
Credit markets price default risk with a 2–5 day lead over equity options. When the HY−IG spread widens (HYG underperforms LQD), credit stress is building even if equities haven't moved. This is a leading indicator for volatility expansion that doesn't show up in VIX until 2–3 days later. For SPY put spread sellers, a widening credit spread is a stop-sign even when VIX looks benign.

The signal is particularly effective as a regime filter: when the HYG/LQD spread is at a 52-week high and widening, credit spread selling underperforms by a large margin.

**Academic evidence:**
- Gilchrist & Zakrajšek (2012, *American Economic Review*): "Credit spreads and business cycle fluctuations" — excess bond premium (EBP) predicts equity vol 2–6 weeks forward; GZ spread component attributable to financial distress has been a consistent leading indicator of equity market stress since 1973.
- Bao, Pan & Wang (2011, *Journal of Finance*): Corporate bond illiquidity and equity volatility are co-integrated; credit market stress Granger-causes equity vol in the short run (2–10 day horizon).
- Mueller, Vedolin & Yen (2019, *Review of Asset Pricing Studies*): Bond vol premium forecasts stock market returns at 1–12 month horizons; short-run (1–4 week) signal is strongest after credit spread spikes.

**Implementation:**
```python
# Both HYG and LQD prices are free via yfinance
hyg_price = fetch('HYG')['Close'].iloc[-1]  # iShares HY Bond ETF
lqd_price = fetch('LQD')['Close'].iloc[-1]  # iShares IG Bond ETF

# Ratio (normalized; level matters less than change)
hyg_lqd_ratio = hyg_price / lqd_price
hyg_lqd_ratio_change_5d = (hyg_lqd_ratio / hyg_lqd_ratio_5d_ago) - 1
hyg_lqd_percentile_63d  = percentile_rank(hyg_lqd_ratio, 63)  # 3-month window

# Binary stress signal
credit_stress = 1.0 if hyg_lqd_ratio_change_5d < -0.02 else 0.0  # >2% credit widening in 5d
```

**Implementation difficulty:** 1/5
**Alpha potential:** Medium-High — most valuable as a regime filter, not a directional predictor
**Data availability:** Free (Yahoo Finance, `HYG`, `LQD`)

---

### 2.3 Treasury Yield Curve Slope (2s10s)

**Signal description:**
The 2-year vs 10-year Treasury yield spread has two distinct effects on options sellers:
1. **Inverted curve** (2Y > 10Y): Historically precedes recessions; elevated equity risk premium; realized vol tends to rise 6–18 months later. Poor environment for naked short premium.
2. **Steepening from inversion**: Historically the most dangerous regime — inversion to re-steepening transitions have occurred near every major equity bear market (2000, 2007, 2020).

At the daily frequency, the *rate of change* of the yield curve matters more than its level. A sudden 20bp+ steepening in a week signals bond market stress that often precedes equity vol spikes.

**Academic evidence:**
- Harvey (1989, *Journal of Financial Economics*): Yield curve inversion predicts recessions at 4-8 quarter horizon; the first paper to formally establish this relationship.
- Estrella & Mishkin (1998, *Review of Economics and Statistics*): Yield spread is the single best predictor of US recession probability in 2–6 quarter window; outperforms all other macro indicators.
- Berge & Jorda (2011, *FRBSF Working Paper*): Updating Harvey — yield curve slope predicts equity vol spikes with 8–16 week lead; the signal is strongest at the inflection point from inversion to normalization.

**Implementation:**
```python
# FRED provides 2Y and 10Y yields free via pandas_datareader
# Or approximate from ETF prices: SHY (2Y proxy), TLT (20Y proxy)
# Better: quandl FRED/GS2, FRED/GS10 (free, no API key for recent data)

yield_2y  = fetch('^IRX')['Close'].iloc[-1]   # yfinance 13-week (proxy; imperfect)
# Better approach: use FRED API (free key required, 20-sec setup)
# yield_2y = fred.get_series('GS2').iloc[-1]
# yield_10y = fred.get_series('GS10').iloc[-1]

yield_curve_slope    = yield_10y - yield_2y     # positive = normal, negative = inverted
yield_curve_change_5d = slope_today - slope_5d_ago  # rate of change
yield_curve_inverted  = 1.0 if yield_curve_slope < 0 else 0.0
```

**Implementation difficulty:** 2/5 (FRED API setup is trivial; free key)
**Alpha potential:** Medium — better as a regime signal than a daily trade filter
**Data availability:** Free via FRED API (`https://fred.stlouisfed.org/docs/api/`); yfinance proxies available

---

### 2.4 USD Strength Index (DXY)

**Signal description:**
USD strength correlates inversely with risk assets and especially with S&P 500 when both equities and currency are in trend. For credit spread sellers on SPY, a rapidly strengthening dollar is a negative signal — it tends to coincide with risk-off flows. The more actionable signal is the *change* in DXY relative to recent volatility: a 1% DXY move in a week is unremarkable; a 3%+ move in a week is a regime signal.

**Academic evidence:**
- Lustig, Roussanov & Verdelhan (2011, *Review of Financial Studies*): Currency carry factor "dollar" risk — global equity volatility and USD appreciation are negatively correlated at the country level; the correlation strengthens during periods of financial stress.
- Menkhoff, Sarno, Schmeling & Schrimpf (2012, *Journal of Finance*): "Carry trades and global foreign exchange volatility" — FX volatility risk premium has systematic component correlated with equity vol; DXY is a clean proxy.

**Implementation:**
```python
dxy = fetch('DX-Y.NYB')['Close']  # yfinance DXY futures proxy
dxy_change_5d    = (dxy.iloc[-1] / dxy.iloc[-6] - 1) * 100
dxy_zscore_20d   = (dxy.iloc[-1] - dxy.tail(20).mean()) / dxy.tail(20).std()
dxy_trending_up  = 1.0 if dxy_change_5d > 1.5 else 0.0  # >1.5% USD strengthening in 5d
```

**Implementation difficulty:** 1/5
**Alpha potential:** Low-Medium — useful as a regime interaction term
**Data availability:** Free (yfinance `DX-Y.NYB`)

---

### 2.5 SPY/TLT Correlation (Equity-Bond Correlation Regime)

**Signal description:**
The sign of the equity-bond correlation is one of the most important macro regime signals. In the "normal" post-2000 environment, bonds rally when equities fall (negative correlation), making equities easier to hedge. When the correlation turns positive (both sell off together, as in 2022), traditional hedging breaks down and volatility becomes harder to predict. For credit spread sellers, a positive equity-bond correlation regime is historically associated with worse outcomes for short premium strategies.

**Academic evidence:**
- Baele, Bekaert & Inghelbrecht (2010, *Journal of Finance*): "The determinants of stock and bond return comovements" — inflation regime is the primary driver of equity-bond correlation sign; high inflation expectations → positive correlation.
- Campbell, Sunderam & Viceira (2017, *Review of Financial Studies*): Time-varying bond betas; correlation sign shift predicts macro regime changes 3–6 months forward.

**Implementation:**
```python
spy_returns = fetch('SPY')['Close'].pct_change()
tlt_returns = fetch('TLT')['Close'].pct_change()  # 20Y Treasury ETF

# Rolling 20-day equity-bond correlation
eb_corr_20d = spy_returns.tail(20).corr(tlt_returns.tail(20))
eb_corr_regime = 1.0 if eb_corr_20d > 0.2 else 0.0  # positive = unusual
eb_corr_trend  = eb_corr_20d - rolling_corr_60d_value  # is correlation changing?
```

**Implementation difficulty:** 1/5
**Alpha potential:** Medium — best used as a regime multiplier, not standalone
**Data availability:** Free (yfinance `TLT`)

---

## 3. Intraday Patterns

**Note on applicability:** The current system runs at daily frequency — entry decisions are made once per day. True intraday features (tick data, VWAP, order book) are therefore out of scope. However, several "end-of-day summary" signals derived from intraday data are cheaply available and relevant.

### 3.1 Opening Range and Gap Characteristics

**Signal description:**
The first 30 minutes of trading set the tone for daily volatility. A large opening gap (>0.5% SPY) that holds direction all day is a trending signal; a gap that reverses within 30 minutes is a mean-reversion signal. For credit spread sellers, trend days (characterized by large opening ranges that hold) expand realized vol; mean-reversion days keep realized vol low.

The "opening range" as a feature can be captured end-of-day using open, high, and close prices — no intraday data required:

```python
gap_pct            = (open_today - close_yesterday) / close_yesterday * 100
open_to_close_move = (close_today - open_today) / open_today * 100
gap_filled_same_day = 1.0 if gap_pct > 0 and open_to_close_move < 0 else 0.0

# Trend day identification (gap in direction, close near extreme)
daily_range_pct = (high - low) / open * 100
close_position  = (close - low) / (high - low)  # 0=closed at low, 1=at high
trend_day       = 1.0 if abs(gap_pct) > 0.3 and close_position > 0.8 else 0.0
```

**Academic evidence:**
- Garvey & Murphy (2004, *Journal of Financial Research*): Opening price revision and intraday returns — gap direction predicts first-hour returns; gaps > 1% have different mean-reversion profile.
- Berkman, Koch, Tuttle & Zhang (2012, *Journal of Financial and Quantitative Analysis*): Overnight order flow and opening prices — institutional imbalances drive opening gaps that partially reverse.

**Implementation difficulty:** 1/5 (daily OHLC data already in cache)
**Alpha potential:** Medium — particularly useful for regime detection
**Data availability:** Free (already have OHLC data)

---

### 3.2 Daily Close-to-High/Low Ratio (Intraday Range Utilization)

**Signal description:**
How much of the daily range gets "used" by the market provides a measure of conviction. A day where SPY moves 1% point-to-point but the total high-low range was 2.5% (50% range utilization) looks very different than a day where a 1% move used 95% of the range. High range utilization with directional close = trending regime. Low range utilization = choppy/mean-reverting regime.

This is directly relevant to realized vol estimation: choppy days overstate realized vol relative to terminal moves.

```python
# Average over rolling window to smooth noise
range_utilization_5d_avg = mean(abs(close-to-close_pct) / (high-low)/open for last 5 days)
directional_efficiency   = abs(sum_of_close_to_close_5d) / sum_of_ranges_5d  # 0→1
```

**Implementation difficulty:** 1/5
**Alpha potential:** Low-Medium (useful as a derived feature)
**Data availability:** Free

---

### 3.3 Volume at Price Distribution (VPOC as Resistance/Support)

**Signal description:**
The Volume Point of Control (VPOC) — the price level with the most volume traded in a recent window — acts as a gravity point for mean-reversion strategies. Credit put spreads that are positioned near the VPOC are likely to see the underlying oscillate around that level, supporting the short-put position. Spreads positioned far above the VPOC in a downtrend are more exposed.

This is a **moderate** implementation effort since we need intraday volume-at-price data, but it can be approximated from daily data using Gaussian-weighted volume distributions.

**Academic evidence:**
- Steidlmayer (1986, original Market Profile work): Volume at price as natural support/resistance; VPOC migration as a trend signal.
- Kavajecz & Odders-White (2004, *Review of Financial Studies*): Price clustering at high-volume levels provides support/resistance that persists 3–5 days; effect is stronger for ETFs than individual stocks.

**Implementation difficulty:** 3/5 (need intraday data or approximation)
**Alpha potential:** Low-Medium
**Data availability:** Polygon.io intraday ($29/mo); yfinance has 60-day intraday history free

---

## 4. Sentiment Signals

### 4.1 CNN Fear & Greed Index — HIGH PRIORITY

**Signal description:**
The CNN Fear & Greed Index aggregates 7 sub-signals into a 0–100 sentiment score:
1. Stock Price Momentum (SPY vs 125d MA)
2. Stock Price Strength (52-week highs vs lows on NYSE)
3. Stock Price Breadth (McClellan Volume Summation)
4. Put and Call Options (PCR)
5. Market Volatility (VIX)
6. Safe Haven Demand (stock vs bond returns)
7. Junk Bond Demand (HY vs IG spread)

For credit spread sellers, **Extreme Greed (>75)** has historically been followed by elevated realized vol and mean reversion — counterintuitively, it's a *worse* environment to sell premium because everyone is already short vol and the margin for error is smaller. **Extreme Fear (<25)** is the best environment: IV is elevated, everyone is buying protection, and realized vol tends to mean-revert.

**Academic evidence:**
- Baker & Wurgler (2006, *Journal of Finance*): "Investor sentiment and the cross-section of stock returns" — composite sentiment indexes predict cross-sectional returns; high sentiment predicts underperformance of volatile stocks (exactly the environment where short vol is dangerous).
- Han (2008, *Review of Financial Studies*): "Investor sentiment and the implied volatility smile" — retail sentiment (proxied by PCR and flows) shifts the entire IV smile; high optimism flattens the smile, compressing put premiums.
- Stambaugh, Yu & Yuan (2012, *Journal of Financial Economics*): Mispricing and investor sentiment — sentiment-based anomalies are stronger on the short side; Extreme Greed readings predict elevated future volatility at 1–4 week horizons.

**Implementation:**
CNN does not provide an official API, but the index is scraped by several free community tools. Most reliable approach:
```python
# Option A: Use the unofficial CNN API endpoint (public, no key)
# https://production.dataviz.cnn.io/index/fearandgreed/graphdata/YYYY-MM-DD
import requests
resp = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2026-03-28")
fear_greed_score = resp.json()["fear_and_greed"]["score"]  # 0-100
fear_greed_rating = resp.json()["fear_and_greed"]["rating"]  # "Extreme Fear" etc.

# Features
fg_score = fear_greed_score  # raw 0-100
fg_extreme_fear  = 1.0 if fg_score < 25 else 0.0
fg_extreme_greed = 1.0 if fg_score > 75 else 0.0
fg_change_7d     = fg_score_today - fg_score_7d_ago  # momentum of sentiment
```

**Implementation difficulty:** 2/5 (scrape risk; CNN may change endpoint)
**Alpha potential:** Medium-High — particularly strong for regime classification
**Data availability:** Free (unofficial CNN API endpoint); `feargreed` Python package available

---

### 4.2 AAII Investor Sentiment Survey

**Signal description:**
The American Association of Individual Investors publishes weekly bull/bear/neutral percentages for retail investors. The bull-bear spread has a strong contrarian signal at extremes: when bulls exceed 50% or bears exceed 50%, the subsequent 8-week market return tends to reverse. For options sellers, high bearishness is a positive signal (IV is elevated relative to likely realized moves); high bullishness is a warning (realized vol tends to spike when retail is maximally complacent).

**Academic evidence:**
- Brown & Cliff (2004, *Journal of Financial and Quantitative Analysis*): "Investor sentiment and the near-term stock market" — AAII bull-bear spread has contrarian predictive power at 1–2 year horizons; weekly data shows weaker but significant short-run signal.
- Fisher & Statman (2000, *Financial Analysts Journal*): "Investor Sentiment and Stock Returns" — three sentiment indicators (AAII, Investors Intelligence, mutual fund flows) all show contrarian value; AAII weekly data has the strongest signal for subsequent 6-month returns.
- Antoniou, Doukas & Subrahmanyam (2013, *Review of Financial Studies*): Momentum strategies perform significantly worse after periods of high retail optimism (high AAII bullishness); directly relevant for trend-following components of our feature set.

**Implementation:**
```python
# AAII publishes free weekly CSV at https://www.aaii.com/sentimentsurvey/sent_results
# Data released every Thursday after market close

aaii_bullish_pct   # percentage bullish (0-100)
aaii_bearish_pct   # percentage bearish (0-100)
aaii_bull_bear_spread = aaii_bullish_pct - aaii_bearish_pct  # -100 to +100

# Contrarian signal
aaii_extreme_bullish = 1.0 if aaii_bullish_pct > 50 else 0.0   # danger for option sellers
aaii_extreme_bearish = 1.0 if aaii_bearish_pct > 50 else 0.0   # opportunity for sellers
aaii_spread_4w_change = spread_today - spread_4w_ago  # trend of sentiment
```

**Caution:** Weekly frequency means this feature is stale for 4 out of 5 trading days. Carry forward the last published value. This is appropriate for daily models (the signal is slow-moving by nature).

**Implementation difficulty:** 2/5
**Alpha potential:** Medium — better as a regime filter than a daily signal
**Data availability:** Free (AAII CSV download), published weekly

---

### 4.3 Social Media Sentiment (StockTwits / Reddit)

**Signal description:**
For SPY specifically (broad market ETF, not a single stock), social media sentiment is noisy and heavily post-hoc. The academic evidence for SPY-level social sentiment is weaker than for individual stocks. However, two specific applications have stronger evidence:
1. **Extreme spike in negative SPY/SPX mentions** (e.g., 5 SD above baseline in a 24-hour window) predicts elevated realized vol in the next 2–5 days.
2. **Reddit r/wallstreetbets options flow** — WSB is large enough to move SPY options open interest at specific strikes; this is detectable via Unusual Whales data.

**Academic evidence:**
- Bollen, Mao & Zeng (2011, *Journal of Computational Science*): "Twitter mood predicts the stock market" — Twitter valence (calm/anxiety/happy) predicts DJIA direction with 87.6% accuracy in a 3-4 day window; effect is weaker for broad indices vs stocks.
- Chen, De, Hu & Hwang (2014, *Review of Financial Studies*): Seeking Alpha articles predict stock returns; negative articles have stronger predictive content than positive. Less applicable to index.
- Subrahmanyam (2018, *Pacific-Basin Finance Journal*): Review of social media effects on markets — aggregate sentiment has a stronger effect on high-retail-ownership stocks (small caps) than large-cap ETFs.

**Honest assessment:** Social media sentiment for SPY specifically has limited incremental value beyond the CNN Fear & Greed index (which already incorporates market-derived sentiment signals). Not recommended as a core feature.

**Implementation difficulty:** 4/5 (API rate limits, NLP preprocessing)
**Alpha potential:** Low for SPY (Medium for individual single-stock underlyings)
**Data availability:** StockTwits API (free tier: 200 req/hr); Reddit API (free); Unusual Whales ($30/mo)

---

## 5. Seasonal / Calendar Effects

### 5.1 Monthly Return Pattern ("Calendar Anomalies")

**Signal description:**
January, April, October, and November historically produce the strongest SPX returns. February, May, June, and September are the weakest. For credit spread sellers, month-of-year affects both the base rate of wins and the distribution of returns. The "Sell in May and go away" effect is real but not tradeable directly — it's a 6-month forward signal.

More actionable at the monthly frequency:
- **January effect**: High upside momentum → puts expire safely; calls at risk
- **September effect**: Consistent underperformance (average −0.5% SPX since 1950); elevated put spread risk
- **October volatility**: VIX tends to peak in October (the "October effect"); historically higher realized vol

**Academic evidence:**
- Bouman & Jacobsen (2002, *American Economic Review*): "The Halloween Indicator, 'Sell in May and Go Away'" — formal test of the seasonal pattern in 37 countries; the effect persists out-of-sample.
- Moller & Zilca (2008, *Financial Management*): The "January effect" and its decay; small stocks > large stocks; SPX January premium has diminished since 2000 but remains detectable.
- Cadsby & Ratner (1992, *Journal of Banking & Finance*): International holiday and turn-of-month effects — month-end to month-start provides systematic positive returns; consistent across 10 markets.

**Current coverage:** We already have `month` and `is_month_end`. What we're missing:
```python
# Add to seasonal features
is_january      = 1.0 if month == 1 else 0.0   # strong equity month
is_september    = 1.0 if month == 9 else 0.0   # weak equity month
is_october      = 1.0 if month == 10 else 0.0  # elevated vol month
quarter_end     = 1.0 if month in (3, 6, 9, 12) else 0.0  # rebalancing flows
month_start_3d  = 1.0 if day <= 3 else 0.0     # turn-of-month effect
```

**Implementation difficulty:** 1/5 (already have `datetime` in feature computation)
**Alpha potential:** Low (second-order effect, partially captured by existing features)
**Data availability:** Free (derived from date)

---

### 5.2 OPEX Week Dynamics — IMPROVE EXISTING FEATURE

**Signal description:**
Our current `is_opex_week` flag (days 15–21) is an approximation. The actual OPEX is the **third Friday** of each month, which ranges from the 15th to the 21st — but the gamma pinning and dealer hedging effects are most pronounced on the specific Thursday (last trading day before OPEX) and OPEX Friday itself, not the whole week.

The specific day-of-week within OPEX week matters:
- **Tuesday/Wednesday pre-OPEX**: Maximum gamma pressure; underlying pinned near max-pain strike
- **OPEX Thursday**: Pin breaks if news catalysts arrive; accelerated time decay
- **OPEX Friday itself**: Time decay cliff; short options typically worth < 10% of mid-week value

For our strategy (DTE ~28 days), OPEX week of the *next* month's expiration is when our positions have the highest gamma. This is when position management is most critical.

**Academic evidence:**
- Ni, Pearson & Poteshman (2005, *Journal of Finance*): "Stock price clustering on option expiration dates" — SPX price clustering at round numbers near major strikes on OPEX day is statistically significant; dealers hedge dynamically, creating price magnetism.
- Golez & Jackwerth (2012, *Review of Financial Studies*): "Pinning in the S&P 500 futures" — futures pin to nearby strikes on OPEX; magnitude varies with dealer net gamma exposure.
- Bartram, Conrad, Lee & Subrahmanyam (2021, *Management Science*): OPEX week options open interest changes predict next-week returns; OI buildup at high strikes signals upside resistance.

**Implementation:**
```python
from datetime import date
import calendar

def get_opex_friday(year: int, month: int) -> date:
    """Third Friday of given month."""
    cal = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] > 0]
    return date(year, month, fridays[2])  # third Friday

now = date.today()
opex = get_opex_friday(now.year, now.month)
days_to_opex = (opex - now).days

is_opex_friday    = 1.0 if days_to_opex == 0 else 0.0
is_opex_thursday  = 1.0 if days_to_opex == 1 else 0.0
days_to_opex_feature = days_to_opex  # continuous (replace binary flag)
```

This replaces the current `is_opex_week` approximation.

**Implementation difficulty:** 1/5
**Alpha potential:** Medium (especially for gamma-sensitive position management)
**Data availability:** Free (computed from date)

---

### 5.3 Quadruple Witching (Quad Witching)

**Signal description:**
Quad witching occurs on the third Friday of March, June, September, and December — when stock index futures, stock index options, stock options, and single stock futures all expire simultaneously. Volume typically surges 2–3x on quad witching Friday and the preceding Thursday. Realized vol is elevated, but IV also spikes (so the RV−IV spread may not be systematically affected). The effect on credit spread returns is mixed: entry near quad witching captures elevated premium, but exit in the following week benefits from rapid vol deflation.

**Academic evidence:**
- Stoll & Whaley (1986, *Journal of Business*): First documentation of "expiration day effects" — price volatility and volume spikes on index futures expiration; persistent since inception.
- Chamberlain, Cheung & Kwan (1993, *Journal of Derivatives*): Quad witching effects on TSX; volume and volatility effects generalize internationally; price reversal post-expiration supports the "mechanical pressure" interpretation.

**Implementation:**
```python
QUAD_WITCHING_MONTHS = {3, 6, 9, 12}

def is_quad_witching_week(dt: date) -> bool:
    """True if within 5 days of quad witching Friday."""
    if dt.month not in QUAD_WITCHING_MONTHS:
        return False
    opex = get_opex_friday(dt.year, dt.month)
    return abs((opex - dt).days) <= 4

is_quad_witching     = 1.0 if is_quad_witching_week(now) else 0.0
is_quad_witching_day = 1.0 if days_to_opex == 0 and month in QUAD_WITCHING_MONTHS else 0.0
```

**Implementation difficulty:** 1/5
**Alpha potential:** Low (partially captured by existing OPEX and month features)
**Data availability:** Free (derived from date)

---

### 5.4 Earnings Season Concentration Effect

**Signal description:**
SPY implied vol responds not just to individual ticker earnings but to the *concentration* of earnings in a given week. When 20+ S&P 500 large-caps report in the same week (typical in Jan/Apr/Jul/Oct weeks 2–4), realized vol of the index is systematically elevated due to idiosyncratic moves partially adding rather than canceling. For SPY put spreads, the expected move implied by options is frequently insufficient during high-concentration earnings weeks — which actually makes our credit spreads *less* attractive (we're getting paid fairly, not over-paid).

**Academic evidence:**
- Frazzini & Lamont (2007, *Journal of Financial Economics*): "The earnings announcement premium and trading volume" — aggregate earnings announcement effects are not fully diversified away at the index level during concentration weeks; realized vol is higher than the same weeks in off-season.
- Savor & Wilson (2016, *Management Science*): "Earnings announcements and systematic risk" — systematic risk (beta) is elevated on earnings days; SPY as the underlying amplifies this during concentration periods.

**Implementation:**
This requires a forward-looking earnings calendar (e.g., from Nasdaq, Earnings Whispers, or the `yfinance` calendar endpoint). A free approximation: flag the known heavy-earnings weeks by calendar position:
```python
# Weeks 2-5 in January, April, July, October = earnings season peaks
# Approximation: flag based on month and week of month
week_of_month = (day - 1) // 7 + 1  # 1-5
is_earnings_season_peak = 1.0 if (
    month in (1, 4, 7, 10) and week_of_month in (2, 3, 4)
) else 0.0
```

**Implementation difficulty:** 2/5 (approximation is trivial; exact count requires earnings calendar API)
**Alpha potential:** Medium (particularly useful for sizing decisions)
**Data availability:** Approximation is free; exact count from Nasdaq calendar (free scrape) or Benzinga API ($30/mo)

---

## 6. Priority Matrix and Implementation Roadmap

### Tier 1: Implement Now (high alpha, low effort)

| Feature | Section | Est. Lift in AUC | Effort | Data |
|---------|---------|-----------------|--------|------|
| **VIX term structure (^VIX3M/^VIX contango ratio)** | §2.1 | +0.02–0.04 | 1/5 | Free |
| **Real CBOE put/call ratio** (replace placeholder 1.0) | §1.1 | +0.02–0.03 | 2/5 | Free |
| **PCR 5-day z-score** | §1.1 | +0.01–0.02 | 2/5 | Free |
| **HYG/LQD ratio + 5d change** | §2.2 | +0.01–0.02 | 1/5 | Free |
| **SPY/TLT rolling correlation** | §2.5 | +0.01 | 1/5 | Free |
| **Accurate OPEX days_to_opex** (replace is_opex_week) | §5.2 | +0.01 | 1/5 | Free |
| **Opening gap pct** | §3.1 | +0.01 | 1/5 | Free (have OHLC) |

These can all be added to `compass/features.py` without new data providers.

### Tier 2: Implement Soon (medium alpha, low-medium effort)

| Feature | Section | Est. Lift in AUC | Effort | Data |
|---------|---------|-----------------|--------|------|
| **CNN Fear & Greed index** | §4.1 | +0.01–0.02 | 2/5 | Free (scrape) |
| **AAII bull-bear spread** | §4.2 | +0.01 | 2/5 | Free (weekly CSV) |
| **Treasury yield curve slope** | §2.3 | +0.01 | 2/5 | Free (FRED API) |
| **Earnings season concentration flag** | §5.4 | +0.01 | 1/5 | Free (approx) |
| **Monthly/quarter seasonals** | §5.1 | +0.005 | 1/5 | Free |
| **Quad witching flag** | §5.3 | +0.005 | 1/5 | Free |

### Tier 3: Consider Later (medium-high alpha, higher effort/cost)

| Feature | Section | Blocker |
|---------|---------|---------|
| Strike-level put/call volume imbalance | §1.2 | Need options chain query at entry |
| VPOC / volume profile | §3.3 | Need intraday data (Polygon) |
| UOA score | §1.3 | Need options chain volume/OI per strike |
| DXY z-score | §2.4 | Already partially captured by SPY returns |
| Social media sentiment | §4.3 | Noise > signal for SPY |

### Not Recommended

| Feature | Reason |
|---------|--------|
| Raw social media sentiment (SPY) | Too noisy; CNN FGI already aggregates this |
| High-frequency microstructure | Daily system; out of scope |
| Alternative data (satellite, credit card) | High cost; diminishing returns on small dataset |

---

## 7. Interaction Features Worth Engineering

Beyond individual signals, certain combinations have stronger predictive power than their components:

```python
# VIX term structure × PCR interaction
# Best environment: contango (calm forward vol) + elevated PCR (fear in near term)
# → premium is elevated; future vol likely to mean-revert
vix_ts_good = 1.0 if vix_contango_ratio > 1.05 else 0.0
pcr_elevated = 1.0 if put_call_ratio_equity > 1.1 else 0.0
premium_selling_sweet_spot = vix_ts_good * pcr_elevated  # conjunction

# Credit stress × regime interaction
# Worst environment: widening credit spreads (HYG/LQD falling) + bear regime
credit_stress_in_bear = credit_stress * regime_bear_flag  # multiplicative

# Sentiment reversal: high fear + improving trend = best entry timing
fg_turning_up = 1.0 if (fg_score < 40 and fg_change_7d > 5) else 0.0
```

These interaction terms are particularly valuable for tree-based models (XGBoost, RF) since trees discover interactions naturally — but explicitly engineering them reduces the amount of training data needed to find them.

---

## 8. Notes on Feature Validation

Before adding any new feature, run this checklist against the 428-trade training set:

1. **Individual IV**: Does each new feature have AUC > 0.52 alone? If not, it will likely dilute the ensemble.
2. **Correlation audit**: Correlation > 0.7 with an existing feature means it's redundant. Drop it.
3. **Lag consistency**: Features derived from EOD data must use `close[-1]` (today's), not `close[-0]` (today's intraday) — avoid look-ahead by 1 day.
4. **Missing rate**: If a feature is missing in > 10% of training records, it will harm tree splitting. Either fix the data gap or drop it.
5. **Walk-forward re-run**: After adding a batch of features, re-run `compass/benchmark_per_regime.py` and compare the new OOS AUC against the baseline of 0.806 (ensemble, current).

The VIX term structure feature alone (§2.1) is expected to increase AUC the most, given how directly it captures the variance risk premium — the economic foundation of why short premium strategies earn returns at all.

---

*Research complete. Next action: implement Tier 1 features in `compass/features.py`, then retrain and validate.*
