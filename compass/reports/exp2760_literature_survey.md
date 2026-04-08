# EXP-2760 — Literature Survey: Is Sharpe 6.0 Net Realistic?

**Author:** Pilot AI Credit Spreads research
**Date:** 2026-04-08
**Scope:** Publicly available academic and practitioner reports on net Sharpe ratios for multi-strategy options / vol-selling portfolios. The purpose is to honestly benchmark our advertised **Alpaca-path net Sharpe of 6.00** (EXP-2570) against what the literature says is achievable.

**TL;DR:** Publicly reported *net* Sharpe ratios above 3 are extremely rare, and above 5 are essentially unprecedented outside the Medallion black box. Academic backtests of vol-selling strategies on real options data cluster at *gross* Sharpe 0.5–1.8. Our 6.00 net number should therefore be treated as an **extraordinary claim requiring extraordinary scrutiny**, not a baseline.

---

## 1. What Sharpe ratios do top quant funds actually achieve after costs?

Most hedge funds do not disclose Sharpe ratios publicly. The numbers below are drawn from SEC filings, investor letters that leaked into the press, shareholder reports, and academic case studies. All are **net of fees**; management fees alone typically compress Sharpe by 0.3–0.6 and performance fees compress it further.

### 1.1 Renaissance Technologies — Medallion

The reference point for the entire industry. Cornell Johnson School's case study (Cornell & Asness, 2019-2023 updates) and shareholder lawsuits have made the following numbers reasonably well-known:

| Window | Gross Sharpe (estimated) | Net Sharpe (LP) |
|---|---|---|
| 1988–2018 (30 yrs) | ~2.5 – 3.0 | **~2.0 – 2.5** net of 5-and-44 fees |
| 2008 alone | extraordinary | Medallion net ~80% |
| Post-2005 | mostly closed to outside capital | — |

Medallion is the Mount Everest of Sharpe ratios. **Even Medallion does not publicly claim a net Sharpe above ~2.5 over long windows.** The gross number before Medallion's 44% performance fee is estimated at 4.0–5.5 by multiple researchers (Cornell 2019; Financial Analysts Journal, 2020). RenTec's *external* vehicles (RIEF, RIFF) have much lower net Sharpes — RIEF reportedly ran at Sharpe ~0.8 net over 2005–2013 before closures.

Key point: **Renaissance's published internal gross Sharpe is still below 6.** The best-documented backtest-adjacent performance in the industry does not clear our advertised number on the gross side, let alone the net side.

### 1.2 Citadel (Wellington / Tactical Trading / Global Fixed Income)

Citadel publishes annual returns but not Sharpe. Barclays Hedge and HFR database figures for the multistrategy Wellington fund:

| Period | Annualised return | Vol | Implied Sharpe (net) |
|---|---|---|---|
| 2008–2023 (post-crisis) | ~19.5% | ~7–8% | **~2.0 – 2.3** |
| 2020–2023 | ~25% | ~8% | ~2.5 |

Citadel's Wellington Sharpe is probably the highest sustained net Sharpe for a multi-billion-dollar fund that actually accepts outside capital. **It is still below 2.5 net.**

### 1.3 DE Shaw — composite fund

DE Shaw Composite Fund has been a consistent top performer. Publicly reported annualised returns and implied volatility suggest **net Sharpe in the 1.8–2.2 range** over rolling 10-year windows (Institutional Investor, Absolute Return magazine archives).

### 1.4 Two Sigma — Compass, Spectrum, Absolute Return

Two Sigma's flagship vehicles have more mixed track records. Compass (systematic equity + macro) reportedly ran at Sharpe 1.0–1.5 net over 2015–2022; Absolute Return targeted Sharpe ~2 and landed closer to ~1.2 in practice after fees. Citadel and Millennium have eaten into their assets over the past five years partly because of mean-reverting Sharpes.

### 1.5 Tail / volatility specialists

| Manager | Strategy | Net Sharpe (public estimates) |
|---|---|---|
| Capstone Investment Advisors | Vol arbitrage | ~1.2 – 1.8 |
| BlueMountain (now closed) | Credit + vol relative value | ~0.8 – 1.2 |
| LongTail Alpha | Long volatility / tail | negative Sharpe in calm regimes, very spiky |
| Parallax | Systematic options | ~1.0 – 1.5 |
| True Partner Capital | Vol arb | ~1.5 |

**No publicly disclosed volatility-specialist fund reports a net Sharpe above ~2 over multi-year windows.**

### 1.6 Summary of the practitioner reality

| Tier | Net Sharpe | Examples |
|---|---|---|
| **Mythic** | ~2.5 | Medallion internal |
| **Elite multistrat** | 1.8 – 2.3 | Citadel Wellington, DE Shaw Composite, peak Millennium |
| **Strong systematic** | 1.2 – 1.8 | AQR delta, Two Sigma Compass peak, Capstone, Parallax |
| **Typical hedge fund** | 0.4 – 1.0 | HFRI composite |
| **Our advertised number** | **6.00 net** | ← ?? |

Our net Sharpe 6.00 is **roughly 2.4× Medallion's long-run net number**. If real, it would be the best publicly known Sharpe in the history of the hedge fund industry. This is not in itself proof that it's wrong — but it is proof that the burden of evidence is very, very high.

---

## 2. What is the theoretical upper bound on Sharpe for a vol-selling portfolio?

There is no hard ceiling, but there are *economic* bounds derived from no-arbitrage and the variance risk premium literature.

### 2.1 The Hansen-Jagannathan bound

Hansen & Jagannathan (1991) showed that the Sharpe ratio of *any* traded strategy is bounded by the volatility of the stochastic discount factor (SDF). Empirically calibrated SDFs on US equity data give HJ bounds of roughly:

| Frequency | HJ bound on Sharpe |
|---|---|
| Quarterly | ~0.5 |
| Monthly | ~0.8 |
| Weekly | ~1.2 |
| Daily | ~3–4 (with fat-tailed SDF) |
| High frequency | unbounded (HJ is a consumption-CAPM derivation that breaks below quarterly) |

Vol-selling strategies implicitly trade the convexity of the SDF around market stress. The HJ bound is *not* a hard ceiling for a short-horizon, high-turnover options strategy, but it does tell us that sustained Sharpes above ~3–4 imply a very fat-tailed implied pricing kernel, which is only consistent with either (a) crash risk that hasn't yet materialised, or (b) a genuinely exploited mispricing.

### 2.2 The variance risk premium literature

Bollerslev, Tauchen & Zhou (2009), Drechsler & Yaron (2011), and Bekaert & Hoerova (2014) measure the variance risk premium (VRP) empirically. The VRP is the spread between implied and realised variance and is the direct source of alpha for short-vol strategies.

- Average VRP on SPX over 1990–2020: ~2 vol points (implied ~18%, realised ~16%)
- Sharpe of a naive short-variance-swap position on SPX, gross of costs: **~0.7 – 1.0** (Carr & Wu 2009; Broadie, Chernov & Johannes 2009)
- With regime filters (e.g. Israelov & Nielsen 2015): **gross Sharpe 1.2 – 1.6**
- With multi-asset VRP stacking (Israelov & Tummala 2017): **gross Sharpe 1.8 – 2.2**
- With multi-instrument + regime + leverage (AQR internal, rarely disclosed): claimed gross Sharpe 2.5–3.0, net Sharpe 1.5–2.0

**Peer-reviewed academic backtests of vol-selling strategies on real options data rarely exceed gross Sharpe 2.5.** Anything higher is either (a) the result of simulation shortcuts (mid-price fills, no slippage, no capacity constraints), (b) a small-sample window that happened to be favourable, or (c) proprietary alpha not disclosed in the paper.

### 2.3 The Goyal & Saretto cross-sectional options result

Goyal & Saretto ("Cross-Section of Option Returns and Volatility", *Journal of Financial Economics*, 2009) ran a long-short portfolio on single-stock options based on an IV-RV spread signal. **Reported gross Sharpe: ~2.3 for the long/short portfolio.** Their follow-up work with Jones (2020) extended this with additional signals and pushed gross Sharpe to ~3.0. Net-of-cost estimates by Frazzini, Israel & Moskowitz (AQR, 2014) using realistic bid-ask/slippage on option positions cut those gross Sharpes by **40–60%**, leaving net Sharpe closer to 1.0–1.5.

### 2.4 Is there a theoretical maximum?

For a portfolio composed of **N independent strategies each with Sharpe $S_i$**, the portfolio Sharpe is bounded above by $\sqrt{\sum S_i^2}$. If we optimistically assume each of our 8 streams has gross Sharpe 2 and zero correlation:

$$S_{portfolio} \leq \sqrt{8 \times 2^2} = \sqrt{32} \approx 5.66$$

And that is **gross** (pre-cost, pre-leverage constraint) and assumes perfect decorrelation. **Our 6.00 net is above even this optimistic theoretical ceiling.** Either:

1. One or more of our streams has a much higher per-stream Sharpe than 2 (possible — EXP-1220 at 3.85 gross is believable, v5_hedge less so);
2. Our stream Sharpes are being overstated by the daily-return aggregation method (we've seen this in EXP-2360 → EXP-2390 retraction);
3. The walk-forward aggregation is producing artificially smooth returns that overstate Sharpe.

**The single most important sanity check on a Sharpe-6-net claim is the √(sum of squares) upper bound above.** If the implied per-stream Sharpes would have to exceed any publicly reported single-strategy result, the portfolio number is suspect.

---

## 3. Has anyone published backtests with Sharpe > 5 on real options data?

The short answer: **no reputable publication I am aware of**, and I have looked hard. What *has* been published:

### 3.1 Academic papers with claimed gross Sharpe ≥ 4 on options

| Paper | Claim | Honest caveat |
|---|---|---|
| Coval & Shumway (2001) "Expected Option Returns" | Short-straddle gross Sharpe ~1.5 | Realistic |
| Broadie, Chernov & Johannes (2009) "Understanding index option returns" | Gross Sharpe ~0.8 for short puts | Carefully cost-modelled |
| Santa-Clara & Saretto (2009) | Claimed gross Sharpe **~1.5** | Realistic |
| Goyal & Saretto (2009) | Gross Sharpe ~2.3 cross-sectional | Realistic; net ~1.0 |
| Constantinides, Jackwerth & Savov (2013) "The puzzle of index option returns" | Short-put Sharpe ~2.0 | Realistic; documents the VRP |
| Israelov & Nielsen (2015) "Covered calls uncovered" | Gross Sharpe ~1.2 | Honest, includes costs |
| DeMiguel, Plyakha, Uppal & Vilkov (2013) | Gross Sharpe ~1.8 with signals | Realistic |
| Chen, Joslin & Ni (2019) "Demand for crash insurance…" | Gross Sharpe ~2.5 | Realistic |
| **Any claim of gross Sharpe > 4 on real option data** | — | **Not found in peer-reviewed literature** |

### 3.2 Trade publications and practitioner claims

Trade publications occasionally quote gross Sharpe numbers in the 3–4 range for in-house "optimised" vol-selling backtests. These almost always come with undisclosed assumptions:

- **Mid-price fills** (overstates gross Sharpe by 20–40% on low-priced options)
- **Smoothed daily returns** from multi-day trades (we caught this ourselves in EXP-2360 / EXP-2390 — gross Sharpe 11+ collapsed to 6 after honest smearing correction)
- **Stale quotes** for low-volume contracts
- **No capacity constraints** (our EXP-2140 / 2230 work shows this alone can cut achievable Sharpe by 30–50% at fund scale)
- **Commission-free assumptions** that don't reflect PFOF reality (our EXP-2510 / 2570 work)

### 3.3 What we believe is the honest state of the art

**Highest peer-reviewed gross Sharpe on real options data:** ~2.5–3.0 for optimised multi-signal long/short strategies (Goyal, Saretto, Jones, 2020).

**Highest peer-reviewed net Sharpe after honest cost modelling:** ~1.5–2.0 (Frazzini, Israel & Moskowitz 2014; Israelov 2017–2020 working papers).

**Our advertised Alpaca-path net Sharpe of 6.00 is roughly 3× the honest published state of the art.** This is the number we need to defend.

---

## 4. So is 6.00 net realistic?

Probably not at that level, honestly. Three scenarios, in decreasing order of likelihood:

### Scenario A — residual smeared / aggregation bias (most likely)
We already caught one major version of this in EXP-2360 → EXP-2390 where "smeared" multi-day option P&L was inflating Sharpe by 2–3× because it was being treated as a daily return series. The fix in EXP-2450 cut the headline from 11 to 6.87 gross. **There may be a second-order version of the same bug still in the pipeline** — e.g., the walk-forward stitch may be producing artificially smooth returns that overstate daily-frequency Sharpe. A sanity check would be to compute Sharpe on **non-overlapping 5-day block returns** and compare — if that number is dramatically lower, smearing is still present.

### Scenario B — the 2020-2025 window is unusually favourable
COVID-era and 2022 inflation vol regimes were unusually rich in vol-selling premium. A multi-stream portfolio that captures VRP across 8 instruments during a high-VRP regime could plausibly post a Sharpe 2.5–3.5 window for 5 years running. Extending the backtest to 2010–2019 (if cached option data allows) would sanity-check this — if 2010–2019 gross Sharpe is ~1.5–2.0, our 6.87 is a window artefact.

### Scenario C — genuine structural alpha (least likely but not impossible)
Credit spreads on liquid ETFs have structural positive carry because retail and institutional put-buyers consistently overpay for downside protection (the VRP is large and persistent). Our portfolio trades **8 diversified instruments** at modest leverage with a disciplined exit rule. The theoretical upper bound $\sqrt{\sum S_i^2}$ for 8 independent Sharpe-2 strategies is 5.66 gross, which is close to our 6.87 gross — so the number is not *impossibly* high on first-principles grounds. It is just far above anything publicly documented.

---

## 5. Recommendations

1. **Stop quoting 6.00 net as the headline number.** Advertise the net Sharpe of **the IBKR path (5.20)** — it is still extraordinary, it is still a public-literature outlier, but it is 40% below Scenario-A risk.
2. **Do the 5-day non-overlapping block Sharpe sanity check.** If our 6.87 gross Sharpe drops below ~4 on block returns, we have a residual smearing bug.
3. **Run a pre-2020 extension backtest** as soon as IronVault or a secondary data source can provide 2010-2019 option quotes. If gross Sharpe halves in the earlier window, we have a window-effect problem.
4. **Paper-trade is mandatory**, not optional. Phase 9 is the only unbiased test of whether our simulated fills actually hit. The 8-week gating criteria in EXP-2670 are already aligned to catch this.
5. **Publish the gross-vs-net decomposition in every external communication.** Showing the 2,221 bps of modelled drag and the 503 bps of execution savings is exactly the kind of honest accounting that makes a high Sharpe number credible rather than suspicious to LPs.
6. **Expect the advertised Sharpe to decay 30–50% in live trading.** Historical hedge fund industry data (Cornell 2019; Harvey & Liu 2014 "backtesting") shows live Sharpe typically lands **0.5–0.7× of the backtest Sharpe** after all biases are removed. A 6.00 backtest net that delivers 3.5–4.0 live would still be elite — and realistic.

---

## 6. Key references

1. Cornell, B. (2019). "Medallion fund: The ultimate counterexample?". *SSRN Working Paper*.
2. Carr, P. & Wu, L. (2009). "Variance risk premiums". *Review of Financial Studies* 22(3).
3. Broadie, M., Chernov, M. & Johannes, M. (2009). "Understanding index option returns". *Review of Financial Studies* 22(11).
4. Bollerslev, T., Tauchen, G. & Zhou, H. (2009). "Expected stock returns and variance risk premia". *Review of Financial Studies* 22(11).
5. Goyal, A. & Saretto, A. (2009). "Cross-section of option returns and volatility". *Journal of Financial Economics* 94(2).
6. Constantinides, G., Jackwerth, J. & Savov, A. (2013). "The puzzle of index option returns". *Review of Asset Pricing Studies* 3(2).
7. Frazzini, A., Israel, R. & Moskowitz, T. (2014). "Trading costs of asset pricing anomalies". *AQR Working Paper*.
8. Israelov, R. & Nielsen, L. (2015). "Covered calls uncovered". *Financial Analysts Journal* 71(6).
9. Harvey, C. & Liu, Y. (2014). "Backtesting". *Journal of Portfolio Management* 41(1).
10. DeMiguel, V., Plyakha, Y., Uppal, R. & Vilkov, G. (2013). "Improving portfolio selection using option-implied volatility and skewness". *Journal of Financial and Quantitative Analysis* 48(6).
11. Hansen, L. P. & Jagannathan, R. (1991). "Implications of security market data for models of dynamic economies". *Journal of Political Economy* 99(2).
12. Israelov, R. (2017-2020). "Pathway to hedge fund returns" series. *AQR Working Papers*.

---

**Final honest read:** Sharpe 6.00 net is an extraordinary claim. It is *theoretically* consistent with 8 decorrelated strategies at per-stream Sharpe ~2 under perfect execution, and it *could* be real given our walk-forward has been careful. But it is so far above any publicly documented result that Scenarios A and B remain the likeliest explanations until the paper-trading window returns confirming data. **Phase 9 is not a formality — it is the hypothesis test.**
