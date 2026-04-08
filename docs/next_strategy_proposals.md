# Next Strategy Proposals — 5 Uncorrelated Alpha Sources

**Objective:** Complement EXP-1220 (Tail Risk, Sharpe 5.78) with uncorrelated strategies testable on real IronVault data.

---

## IronVault Data Availability

| Ticker | Contracts | Daily Bars | Date Range | Intraday |
|--------|-----------|-----------|------------|----------|
| SPY | 193,272 | 4,378,094 | 2020-01 → 2026-04 | 1.4M bars |
| XLI | 17,287 | 200,761 | 2020-01 → 2026-04 | No |
| GLD | 12,515 | 154,290 | 2019-12 → 2024-02 | No |
| XLF | 9,256 | 243,583 | 2020-01 → 2026-04 | No |
| QQQ | 9,194 | 304,080 | 2020-01 → 2023-04 | No |
| TLT | 9,185 | 185,357 | 2019-12 → 2024-06 | No |
| SOXX | 3,460 | 37,229 | 2020-07 → 2026-04 | No |
| XLK | 2,680 | 18,702 | 2020-01 → 2026-04 | No |
| XLE | 1,757 | 20,542 | 2020-04 → 2026-04 | No |

**No individual stock options.** No IWM, IEF, SHY, HYG.

---

## (A) Dispersion Trading — Sector ETFs as Component Proxy

### Hypothesis
Classical dispersion trades sell index vol and buy single-stock vol. We don't have individual stock options, but we can approximate using sector ETFs as "components." Sell SPY straddles and buy XLF+XLK+XLI+XLE straddles when implied correlation (SPY IV² vs weighted sector IV²) exceeds realized correlation by >10pp. The premium comes from the correlation risk premium — index IV systematically overprices correlated moves.

### Expected Performance
- **Sharpe:** 0.5-1.0 (lower than classical dispersion due to sector-level approximation)
- **CAGR:** 3-8%
- **Max DD:** <10%
- **Correlation to EXP-1220:** 0.1-0.3 (dispersion is correlation-driven, not vol-level-driven)

### Data Requirements
- SPY options: **Full coverage** (IronVault)
- XLF, XLI, XLK, XLE options: **Available** but XLE/XLK are thin (1.7K-2.7K contracts)
- Sector ETF underlying prices: **Yahoo Finance**
- Realized correlation: Computed from underlying returns

### Feasibility: MEDIUM
Prior experiment EXP-1200-max (dispersion) returned -0.36% with Sharpe -0.22. However, that used a different implementation. The sector-ETF approach hasn't been tested. Main risk: sector ETF options lack the strike granularity needed for precise vega-neutral positioning.

---

## (B) Gamma Scalping — Long Gamma with Delta-Hedge

### Hypothesis
Buy ATM SPY straddles when IV rank < 25th percentile (cheap vol). Delta-hedge every 30-60 minutes using intraday 5-min bars. P&L = (realized vol² - implied vol²) × vega. This is a long-vol strategy that profits when options are underpriced — the opposite payoff profile to credit spreads.

### Expected Performance
- **Sharpe:** 0.5-1.0
- **CAGR:** 2-5%
- **Max DD:** <8%
- **Correlation to EXP-1220:** -0.2 to -0.5 (both profit in high-vol regimes, but different mechanisms)

### Data Requirements
- SPY ATM options (daily): **Full coverage** (IronVault)
- SPY intraday 5-min bars: **Partial** (1.4M bars — not comprehensive)
- IV surface/skew: **Derivable** from option strike prices
- VIX for IV rank: **Yahoo Finance**

### Feasibility: LOW-MEDIUM
Intraday data coverage is partial (subset of contracts/dates). Gamma scalping requires frequent delta adjustments — if intraday data has gaps, the backtest won't be realistic. Also, gamma scalping has negative expected value in fairly-priced markets; the IV rank filter is the entire edge, and it produces very few trades.

---

## (C) Earnings Volatility — Sector ETF Earnings Season

### Hypothesis
Sector ETFs exhibit systematic IV elevation during their sectors' concentrated earnings periods (e.g., XLF in Jan/Apr/Jul/Oct when banks report). Sell strangles on sector ETFs 2 weeks before earnings season starts, buy back after the IV crush. The edge: sector ETF IV overstates the actual earnings move because it prices in idiosyncratic risk that gets diversified away at the ETF level.

### Expected Performance
- **Sharpe:** 0.8-1.5 (if the IV overstatement exists at ETF level)
- **CAGR:** 3-6%
- **Max DD:** <12%
- **Correlation to EXP-1220:** 0.1-0.2 (earnings IV is calendar-driven, not crash-driven)

### Data Requirements
- XLF, XLK, XLE options: **Available** in IronVault
- Earnings calendar: **Algorithmic** (shared/economic_calendar.py has quarter dates)
- Sector earnings concentration dates: **Hard-coded** (banks report mid-Jan, tech mid-Apr, etc.)
- IV history per sector ETF: **Derivable** from option prices

### Feasibility: MEDIUM
The key question is whether sector ETF IV shows the same earnings crush pattern as individual stocks. ETFs are diversified — the crush may be muted. XLE only has 1,757 contracts (potentially too thin for reliable pricing). XLF (9,256) and XLK (2,680) are the best candidates. Needs empirical validation before building a full strategy.

---

## (D) Sector Momentum with Options Overlay

### Hypothesis
Rank XLF, XLI, XLK, XLE by trailing 20-day return. Sell OTM put spreads on the top-ranked sector (momentum winner) — the winner has institutional demand support making put spreads safer. Avoid the bottom-ranked sector entirely. Rebalance bi-weekly. This captures momentum premium via options rather than directional equity exposure.

### Expected Performance
- **Sharpe:** 0.8-1.2
- **CAGR:** 5-10%
- **Max DD:** <12%
- **Correlation to EXP-1220:** 0.2-0.35 (sector rotation is structural, not vol-driven)

### Data Requirements
- XLF, XLI, XLK, XLE options: **Available** in IronVault
- Sector ETF underlying prices: **Yahoo Finance**
- Sector relative strength ranking: **Computed** from returns

### Feasibility: HIGH
Simplest to implement. All data is available. The main risk is that sector ETF options are less liquid than SPY — need to verify credit viability per sector (min credit > slippage). Start with XLF+XLK (most contracts), add XLI+XLE if liquid.

---

## (E) Fixed Income Relative Value — GLD/TLT Spread

### Hypothesis
The GLD/TLT ratio mean-reverts over 2-6 week periods around a trend driven by real interest rates. When gold outperforms bonds by >1.5 standard deviations (20-day z-score), sell put spreads on GLD and call spreads on TLT (expecting reversion). Vice versa when bonds outperform. The spread captures the reversion premium while hedging macro direction.

Note: We don't have IEF or SHY options in IronVault — only TLT. So traditional curve trades (2s10s, 5s30s) are not feasible with options. The GLD/TLT spread is the best available fixed income relative value trade.

### Expected Performance
- **Sharpe:** 1.0-1.5
- **CAGR:** 5-10%
- **Max DD:** <10%
- **Correlation to EXP-1220:** 0.1-0.2 (gold/bond dynamics are macro-driven, not VIX-driven)

### Data Requirements
- GLD options: **Available** (12,515 contracts) but **ends Feb 2024** — only ~4 years
- TLT options: **Available** (9,185 contracts) but **ends Jun 2024** — only ~4.5 years
- GLD/TLT underlying prices: **Yahoo Finance** (full history)
- Real yields proxy: **Optional** — TIPS spread from FRED

### Feasibility: MEDIUM-HIGH
Both GLD and TLT have sufficient options data for 2020-2024. The 4-year backtest window is shorter than ideal but covers COVID, rate hikes, and recovery — all interesting regimes. Main risk: GLD/TLT options liquidity is much lower than SPY — wider spreads eat into credits. Must verify min credit exceeds slippage.

---

## Priority Ranking

| Priority | Strategy | Data Ready? | Expected Sharpe | Corr to 1220 | Next Step |
|----------|----------|-------------|-----------------|--------------|-----------|
| **P0** | (D) Sector Momentum | Full | 0.8-1.2 | 0.2-0.35 | Build EXP-1640 |
| **P0** | (E) GLD-TLT RelVal | Mostly (ends 2024) | 1.0-1.5 | 0.1-0.2 | Build EXP-1630 |
| **P1** | (C) Earnings Vol Crush | Available | 0.8-1.5 | 0.1-0.2 | Verify IV crush exists on ETFs |
| **P2** | (B) Gamma Scalping | Partial (intraday) | 0.5-1.0 | -0.2 to -0.5 | Assess intraday coverage |
| **P2** | (A) Dispersion (Sector) | Available | 0.5-1.0 | 0.1-0.3 | Prior attempt failed — needs new approach |

## Strategies Explicitly Ruled Out

| Strategy | Reason | Evidence |
|----------|--------|----------|
| Calendar effects | All 8 anomalies non-significant | EXP-1150: Sharpe -0.54 |
| Cross-asset lead-lag | Weak signals, overlay hurts | EXP-1110: Sharpe 0.38, -18.6pp WR |
| 0-DTE ultra-short | Too few signals | EXP-1020: 0.82 trades/month |
| Individual stock earnings | No stock options in IronVault | Data constraint |
| Yield curve (IEF/SHY) | No IEF/SHY options in IronVault | Data constraint |
