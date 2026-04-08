# MASTERPLAN.md v11 — Final State · Ready for Paper Trading

**Updated:** 2026-04-08 (night, end of the 3-day sprint)
**Status:** Paper trading on Alpaca starts 2026-04-09. 95+ experiments run across 3 days. North Star v8a (8 streams, Ledoit-Wolf weighted, EXP-2370 DD circuit, EXP-2470 execution stack, commission-free broker) hits Sharpe 6.00 net on backtest. Honest expected live number is 3.5–4.5. Phase 9 is the hypothesis test.
**Policy:** Gross, net, AND expected-live reported side-by-side. Every number traces to a committed experiment. No inflated claims, no smeared inputs, no synthetic data.

---

## 1. Mission

Build a validated multi-strategy options-trading system. Data-driven: kill losers, optimise winners, paper trade, scale to capacity, go live.

---

## 2. North Star Dashboard — Gross · Net · Expected Live

| Target | Goal | **Gross** (backtest) | **Net IBKR** | **Net Alpaca** | **Expected Live** | Status |
|---|---|---|---|---|---|---|
| **Sharpe** | ≥ 6.0 | **6.87** (EXP-2570 LW) | 5.20 | **6.00** | **3.5 – 4.5** (EXP-2760 decay) | ✅ gross · ⚠ live TBC |
| **CAGR** | ≥ 100% | ~97% | ~80% | **~93%** | ~60 – 75% | ✅ backtest · ⚠ live TBC |
| **Max DD** | ≤ 12% | 5.5% | 5.5% | **4.2%** (circuit ON) | < 10% expected | ✅ |
| **Recovery from 3% DD** | fast | 5.5d mean, 11d max (EXP-2720) | same | same | same | ✅ |
| **6/6 years positive** | yes | yes (EXP-2280 WF) | yes | yes | TBC | ✅ backtest |
| **AUM capacity** | ≥ $500M | ~$50M (EXP-2230 SLV-gated) | same | same | **~$1–10M seed** | ❌ structural |
| **Win rate** | — | 88% (171 real IronVault trades, EXP-1220) | same | same | TBC | ✅ |
| **Multi-strategy** | ≥ 5 | **8 streams live** | same | same | same | ✅ |
| **Rule Zero (real data)** | 100% | IronVault + Yahoo + Fed calendar | same | same | same | ✅ HELD |

### The three-column reality

| Context | Sharpe | CAGR | DD | When to quote |
|---|---:|---:|---:|---|
| **Gross** (ideal execution, zero costs) | **6.87** | ~97% | 5.5% | Theoretical upper bound. *Never the headline.* |
| **Net (IBKR Pro $0.65/ctr)** | 5.20 | ~80% | 5.5% | Conservative · fallback broker · what to quote to skeptical LPs |
| **Net (Alpaca commission-free)** | **6.00** | ~93% | **4.2%** | **Production config headline** · assumes no worst-case PFOF tax |
| **Expected live (Sharpe decay 0.5-0.7×)** | **3.5 – 4.5** | 60–75% | < 10% | **What Carlos should actually underwrite** |

**Why the expected-live range matters.** Historical hedge fund industry data (Cornell 2019, Harvey-Liu 2014, reviewed in EXP-2760 literature survey) shows live Sharpe lands at **0.5-0.7× of backtest Sharpe** after all biases are removed. A 6.00 backtest delivering 3.5-4.5 live would still be elite — Medallion's long-run net is ~2.5 — and is the number to underwrite capital deployment against.

---

## 3. Architecture — North Star v8a (8 streams)

```
┌─────────────────────────── INPUTS (all REAL) ───────────────────────────┐
│                                                                          │
│   IronVault options_cache.db          Yahoo Finance               Fed    │
│   • SPY/QQQ/XLF/XLI/GLD/SLV           • SPY/QQQ/XLF/XLI/GLD/SLV    • FOMC│
│     option_daily + contracts          • ^VIX / ^VIX3M / ^VVIX      cal.  │
│   • 276K contracts, 6.3M option-days  • 90d ADV + last close             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────── 8 ALPHA STREAMS ──────────────────────────────┐
│                                                                         │
│  1. exp1220      SPY put credit spread 28 DTE 5% OTM     weight 35.0%   │
│  2. qqq_cs       QQQ put credit spread 28 DTE 5% OTM     weight 15.0%   │
│  3. xlf_cs       XLF put credit spread 28 DTE            weight 10.0%   │
│  4. xli_cs       XLI put credit spread 28 DTE            weight 10.0%   │
│  5. gld_cal      GLD calendar (front-back 30/60 DTE)     weight 10.0%   │
│  6. slv_cal      SLV calendar (front-back 30/60 DTE)     weight  5.0%   │
│  7. cross_vol    SPY/ETF IV-RV relative value            weight 10.0%   │
│  8. v5_hedge     SPY tail puts + VIX calls (hedge)       weight  5.0%   │
│                                                                         │
│  Mean pairwise ρ:  +0.016   (essentially zero — genuine diversification)│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────── PORTFOLIO OVERLAYS (causal) ───────────────────────┐
│                                                                          │
│  Ledoit-Wolf covariance   + EXP-1970 VoV gate     + EXP-1880 FOMC filter │
│  Risk-parity weights      + EXP-2070 VIX term str + EXP-2540 regime TC   │
│  Vol target 12% ann       + EXP-2370 3% DD circuit (flatten, causal)    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────── RISK BINDING ─────────────────────────────────┐
│                                                                          │
│   EXP-1890 Portfolio Risk Manager (30/30 tests):                         │
│     • CrossStrategySizer (risk parity + Kelly clamp)                     │
│     • CorrelationMonitor  (alerts ≥ 0.50 in stress regimes)              │
│     • DrawdownCircuitBreaker (soft 10% / hard 12%)                       │
│     • AllocationLimiter   (per-stream caps + rebalance trigger)          │
│     • LeverageGovernor    (1× → 3× regime-scaled)                        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────── EXECUTION STACK (EXP-2470) ────────────────────────┐
│                                                                          │
│   A. Limit orders at mid       bid-ask ×0.50   (50% fill rate)          │
│   B. Patient pre-close window  slippage ×0.75  (EoD ADV 2× open)        │
│   C. Route to cheapest $/ntl   bid-ask ×0.78   (XLI → XLF → SPY)        │
│   D. Multi-leg combo orders    bid-ask ×0.75   (combo price improvement)│
│                                                                          │
│   Stacked savings: 503 bps/yr at 3× leverage                             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────── BROKER & PAPER TRADING ──────────────────────────┐
│                                                                          │
│   Primary:  Alpaca commission-free options (since 2023)                  │
│   Fallback: IBKR Pro fixed ($0.65/ctr) — used if Alpaca PFOF drift       │
│                                                                          │
│   Daily cron: compass.exp2830_paper_signal_generator                     │
│               → compass/reports/paper_signals/signals_YYYY-MM-DD.json    │
│               → compass/logs/paper_signals_audit.jsonl                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase Plan

### Phase 7 — Capital Utilisation ✅ COMPLETE (2026-04-07)
Multi-stream portfolio + vol targeting + equal_risk_15%. Wave 6 close.

### Phase 8 — AUM Capacity ⏳ MID-FLIGHT

**Key lesson** (learned the hard way via 4 rejections on April 8): capacity is *not* a weight-shuffling problem. The only two wins came from *adding high-liquidity streams* with different cadence:

| Approach | Result |
|---|---|
| EXP-2580 SPY-weekly credit spreads | ρ=+0.13 · $7.6B sleeve capacity |
| EXP-2590 QQQ deep-dive + v8a integration | +0.40 portfolio Sharpe · 1.31× capacity |
| EXP-2350 SLV → QQQ/TLT replacement | ❌ REJECTED (combined Sharpe+capacity bar missed) |
| EXP-2380 Futures calendars as sleeve | ❌ REJECTED (futures ≈ ETF spreads on real data) |
| EXP-2430 Capacity-optimised re-weight | ❌ REJECTED (XLI becomes next bottleneck) |
| EXP-2480 3-sleeve high-capacity collapse | ❌ REJECTED (−0.33 Sharpe, only 1.3× capacity) |

**Remaining Phase 8 work:** integrate EXP-2580 SPY-weekly as stream 9; aggressively cut SLV (→ 2%) and XLI (→ 3%) weights; expected ceiling lift to ≥ $200M. Continues in parallel with Phase 9 paper trading.

### Phase 9 — Paper Trading ⭐ STARTS 2026-04-09 (TOMORROW)

**Config:** 8-stream v8a Ledoit-Wolf on Alpaca commission-free · 3% trailing-DD circuit ON · EXP-2470 execution stack A+B+C+D · EXP-2540 regime-TC skip filter when VIX ≥ 25.

**Harness:**
- `compass/exp2830_paper_signal_generator.py` — daily 09:00 ET cron
- `compass/reports/paper_signals/signals_YYYY-MM-DD.json` — machine-readable signals
- `compass/logs/paper_signals_audit.jsonl` — append-only audit trail
- `compass/paper_monitor_dashboard.py` — 16:30 ET EoD reconciliation
- `compass/portfolio_risk_manager.py` (EXP-1890) — runtime risk binding
- Alpaca integration layer (separate, consumes the JSON)

**Hard go/no-go gates (EXP-2670 + EXP-2670 + EXP-2830):**

| # | Criterion | Source |
|---|---|---|
| G1 | ≥ 4 consecutive weeks paper (20 trading days) | EXP-2670 |
| G2 | 20-day rolling Sharpe within ±15% of EXP-2570 forecast 6.00 (floor 5.10, ceiling 6.90) | EXP-2570 |
| G3 | 20-day annualised CAGR within ±20% of 93% (floor 74.4%, ceiling 111.6%) | EXP-2570 |
| G4 | Max DD ≤ 8% (expected ≤ 4.2% with circuit) | EXP-2370 |
| G5 | Circuit breaker ≤ 1 spurious HALT in 20 days | EXP-2370 + EXP-2630 |
| G6 | Limit-at-mid fill rate ≥ 50% | EXP-2470 technique A |
| G7 | Slippage ≥ 25% lower vs open-of-day baseline | EXP-2470 technique B |
| G8 | Alpaca fills within ±3 cents/ctr of IBKR NBBO (no PFOF tax) | EXP-2510 |
| G9 | Rolling pairwise correlation < 0.50 in stress | EXP-1890 CorrelationMonitor |
| G10 | Zero manual trade overrides | ops discipline |

**Abort triggers** (any one flattens immediately):
- Live DD hits 12% hard circuit
- Rolling 4-week Sharpe < 2.0 for 5 consecutive days
- Alpaca fills deviate from IBKR NBBO > 5 cents on > 20% of orders
- Any Rule Zero violation (synthetic fill, extrapolated quote)

**Expected paper numbers (8-week window, ~40 trading days):**

| Metric | Target | Acceptable | Hard reject |
|---|---|---|---|
| Sharpe | 6.00 (or ~4.0 expected-live) | 3.5 – 6.5 | < 3.0 |
| Annualised CAGR | 93% (or ~65% live) | 50% – 120% | < 30% |
| Max DD | < 5% | < 10% | ≥ 12% |
| Total trades | 30–50 | 20–70 | — |
| Fill rate | ≥ 50% | 40% – 80% | < 30% |

### Phase 10 — Live Deployment (after Phase 9 passes)

**Hard prerequisites before first live $:**

1. ≥ 4 consecutive weeks of paper P&L within ±15% of forecast (Gate G1+G2+G3)
2. Zero Gate-G* red flags
3. Zero abort-trigger events in paper window
4. EXP-2670 checklist re-run returns **OVERALL=GO**
5. Secondary broker (IBKR Pro) account funded and wired as fallback
6. On-call rotation in place for 24/7 circuit-breaker monitoring
7. Dollar-notional sizing patched in (current integer-contract sizing is a sub-$1M accuracy issue)
8. Polygon Options secondary data feed added (single-provider dependency is unacceptable at T3+)
9. Carlos sign-off on the advertised Sharpe number (Alpaca 6.00 vs IBKR 5.20 vs expected-live 4.0 — we cannot send three different numbers to three different LPs)

**Scaling schedule:**

| Tranche | Capital | Leverage | Gate | Duration |
|---|---|---|---|---|
| T0 | Paper $100K | 1× | Phase 9 passes all 6 hard gates | 4 weeks min |
| T1 | **$25K live** | 1× | T0 ±15% hold | 4 weeks |
| T2 | $100K | 2× | T1 ±15% hold | 4 weeks |
| T3 | $1M | 2× | T2 ±15% hold + no live DD > 8% + Polygon added | 8 weeks |
| T4 | $10M | 3× | T3 pass + Phase 8 SLV cut to ≤ 3% | 8 weeks |
| T5 | **$50M** | 3× | T4 pass + EXP-2580 SPY-weekly integrated | 12 weeks |
| T6 | $100M | 3× | T5 pass + new XLI replacement | 12 weeks |
| T7 | $500M | 3× | T6 pass + SPY-weekly at full weight | TBD |

**Headline AUM cap today:** ~$50M (SLV-binding). **After Phase 8 integration:** projected ~$200M.

---

## 5. Wave Registry — ~95 experiments, April 6-8

| Wave | Range | Count | Headline winners ★ | Killed | Retractions |
|---|---|---|---|---|---|
| **1** — Alpha discovery (Apr 6) | EXP-1660 – 1840 | ~16 | 1750, 1770, 1780 | 3 | — |
| **2** — Portfolio construction | EXP-1850 – 1880 | 4 | 1850, 1880 | — | — |
| **3** — Risk infra | EXP-1890 – 1900 | 2 | **1890** ★ | — | — |
| **4** — Alpha hunt | EXP-1910 – 1990 | 9 | **1970** ★ | 5 | — |
| **5** — Overlay integration | EXP-2000 – 2030 | 4 | **2000** ★, 2020 | 1 | — |
| **6** — First Sharpe 6 hit | EXP-2050 – 2090 | 5 | **2050** ★★, 2070, 2080 | — | — |
| **7** — Capacity round 1 + Carlos report | EXP-2100 – 2180 | ~9 | 2130, 2180 | 2 | — |
| **8** — 7-stream + robustness audit | EXP-2200 – 2280 | ~9 | **2200** ★★★, 2230, **2280** ★★ | — | — |
| **9** — Cost + broker reality | EXP-2340 – 2480 | ~15 | **2370** ★★, **2420** ★★★, **2470** ★★ | 5 | **2360→2390, 2400→2450** |
| **10** — Commission-free + Phase 8 | EXP-2500 – 2630 | ~16 | **2510, 2540, 2560, 2570 ★★★, 2580 ★★, 2590 ★**, 2600, 2630 | 2 | — |
| **11** — Paper-trading prep | EXP-2640 – 2840 | ~20 | **2670** ★, **2720** ★, **2830** ★ | — | — |
| **Total** | **EXP-1660 → 2880** | **~95** | **~25 ★** | **~18** | **4** |

### Top 15 experiments by production value

1. **EXP-1220** — 171 real credit-spread trades, 88% WR — *the foundation*
2. **EXP-2570** ★★★ — Net Sharpe 6.00 on Alpaca commission-free — *the headline*
3. **EXP-2200** ★★★ — First 7-stream equal_risk_15% (Sh 5.96 gross)
4. **EXP-2420** ★★★ — Real transaction cost model (baseline net 4.49)
5. **EXP-2370** ★★ — DD circuit breaker (24% → 6.77% DD, Sharpe UP)
6. **EXP-2470** ★★ — Execution optimization stack (+0.33 Sharpe)
7. **EXP-2280** ★★ — 20-fold WF robustness (no losing fold)
8. **EXP-1890** ★★ — Portfolio Risk Manager (30/30 tests)
9. **EXP-2580** ★★ — SPY weekly credit spreads (ρ +0.13, $7.6B cap)
10. **EXP-2590** ★ — QQQ deep-dive + v8a integration (+0.40 Sharpe)
11. **EXP-2050** ★★ — First Sharpe 6+ hit
12. **EXP-2540** ★ — Regime-conditional TC model (+0.83 Sharpe skip)
13. **EXP-2720** ★ — DD recovery analysis (max 11 days, the marketable number)
14. **EXP-2670** ★ — Go/No-Go checklist + 10 gating criteria
15. **EXP-2830** ★ — Production-ready daily signal generator

### Notable retractions & honest negatives

- **EXP-2360 → EXP-2390**: "robust covariance" inflated Sharpe via input smearing; retracted
- **EXP-2400 → EXP-2450**: sparse combined "best-of" numbers retracted after smearing audit
- **EXP-2480**: 3-sleeve collapse rejected (−0.33 Sharpe, only 1.3× capacity)
- **EXP-2430**: Capacity-optimised 7-stream rejected (XLI becomes next bottleneck)
- **EXP-2350**: SLV → QQQ/TLT replacement rejected (combined bar missed)
- **EXP-2380**: Futures calendars rejected (futures ≈ ETF option spreads on real data)
- **EXP-2460**: Zero-cost T+V overlay rejected (NEGATIVE on diversified portfolio)
- **EXP-2090**: GLD/SLV seasonality filter rejected (pre-pandemic patterns didn't persist)
- **EXP-1990**: Meta-learner overfits with 10 features on 141-trade OOS
- **EXP-1930**: VVIX signal overlay killed (+0.05 pooled OOS, parameter sweep artifact)
- **EXP-2030**: Intraweek seasonality killed (pooled OOS −0.13)

---

## 6. Honest Corrections Log — Bugs We Fixed

| # | Bug | Impact | Fixed by | Rule added |
|---|---|---|---|---|
| 1 | Sharpe formula: used `CAGR/(vol×√252)` instead of `mean/std×√252` | Every pre-`ff9dd15` Sharpe inflated 1.07–2.4× | Fixed pre-Wave 1 | "Use arithmetic mean, not geometric CAGR" |
| 2 | Synthetic data contamination (adaptive+hedge Sharpe 9.09 used `np.random.normal()`) | Whole portfolio variant invalidated | Operation Real Data, Rule Zero from EXP-1220 forward | **Rule 1: NO SYNTHETIC DATA** |
| 3 | Capital dilution (86% zero-return days crushed daily Sharpe) | Per-trade Sharpe 1.26 → daily Sharpe near zero | Wave 6 multi-stream + vol targeting (EXP-2050, 2200) | "Capital utilisation must be solved" |
| 4 | Hedge cost underestimation (academic 2%/yr vs real 4.36%/yr) | 2× over-estimated alpha net | v5 hedge redesign (EXP-1780) | "Hedge costs from real IronVault quotes only" |
| 5 | VIX call hedge unvalidated (VIX options not in IronVault) | 40% of hedge budget was modelled, not measured | UVXY/VXX proxy documented as proxy only (EXP-2230) | "Proxy data must be flagged as such" |
| 6 | Per-fold parameter-sweep artifacts (EXP-1930 IS +0.39 → OOS +0.05) | 8 candidate experiments rejected on OOS | "Pool test trades, not fold metrics" | "Parameter sweeps require OOS validation" |
| 7 | Pooled vs stitched Sharpe divergence | Full-sample 5.96 vs stitched fold 4.43 — both correct, different metrics | Advertise both explicitly | "Report pooled AND stitched" |
| 8 | **Smeared-input Sharpe inflation** (EXP-2360 claimed 11.73, real 6.87) | Multi-day option P&L treated as daily returns | EXP-2390 audit + EXP-2450 retraction | **Rule 10: smeared inputs are synthetic inputs** |
| 9 | Zero-cost overlays that help single-strategy can hurt diversified portfolio (EXP-2460 T+V overlay) | Overlay flagged as winner at stream level, negative at portfolio level | EXP-2460 killed at portfolio level | "Re-test every overlay at portfolio level" |
| 10 | Capacity is not a weight-shuffling problem (4 rejections in a row) | Wasted 4 experiments on wrong hypothesis | Pivoted to adding high-liquidity streams (EXP-2580, 2590) | "Capacity is added, not reallocated" |
| 11 | Broker commissions are the single largest controllable cost (IBKR → Alpaca = +0.80 Sharpe on same portfolio) | Every pre-April-8 net Sharpe quote needed a broker qualifier | EXP-2510 broker analysis + EXP-2570 commission-free headline | **Rule 12: every net Sharpe names the broker** |
| 12 | Walk-forward DD looked 24% pooled vs 5.5% full-sample (fold-stitching artifact + 2022 inflation shock) | DD headline confusing | EXP-2370 causal 3% trailing-DD circuit breaker solved it definitively | "DD is a control problem, not a reporting problem" |
| 13 | Sharpe 6.00 is ~3× the published academic state of the art (EXP-2760 literature survey) | Extraordinary claim requires extraordinary scrutiny | Published survey, expected-live column added to dashboard, paper trading is the hypothesis test | **Rule 13: expected live = 0.5-0.7× backtest; underwrite against that** |

---

## 7. Pending Decisions for Carlos

**Decision 1 — Advertised Sharpe number.** Which of these goes on the marketing deck?

- **(A) 6.00 Sharpe / 93% CAGR** — the Alpaca commission-free backtest headline. Maximum impact but highest Scenario-A risk (residual smearing we haven't caught).
- **(B) 5.20 Sharpe / 80% CAGR** — the IBKR Pro backtest. More conservative. Still historically elite.
- **(C) ~4.0 Sharpe / ~65% CAGR** — the expected-live number after 0.5–0.7× decay. Safest promise, still top-quartile.
- **(D) Triple-column honest disclosure:** "6.00 gross headline on Alpaca, 4.0 expected live, 2.5 Medallion's long-run net — we target between 4 and 6."

**Recommendation:** **(D)**. LPs who matter will do the decay math anyway; getting ahead of it builds credibility.

**Decision 2 — Paper trading duration.** EXP-2670 gates require 4 weeks. EXP-2720 shows deep-DD recovery is 11 days max, so 4 weeks should capture ~2 drawdown cycles.

- **(A) 4 weeks** — minimum gating, fastest to live
- **(B) 6 weeks** — one more DD cycle, reduced variance
- **(C) 8 weeks** — full MASTERPLAN spec, most conservative

**Recommendation:** **(B) 6 weeks.** Good balance: enough signal to be statistically meaningful, not so long that the live opportunity decays.

**Decision 3 — First live tranche size.** MASTERPLAN says $25K at 1×. EXP-2420/2470 show costs are proportional to notional, so $25K is noisy on fills.

- **(A) $25K** — as specified, maximum caution
- **(B) $100K** — better fill quality, same leverage, still well below any capacity concern
- **(C) $250K** — meaningful P&L signal, still 1/200th of soft cap

**Recommendation:** **(B) $100K.** $25K sizing produces 1-contract trades where slippage is dominated by rounding; $100K lets us cleanly validate EXP-2470 execution claims.

**Decision 4 — Phase 8 SLV decision.** SLV calendar is the $16M bottleneck. Options:

- **(A) Keep SLV at 5% weight** — maintain diversification, live at $50M cap forever
- **(B) Cut SLV to 2% weight** — lift cap to ~$200M, lose some correlation benefit
- **(C) Drop SLV, add SPY-weekly (EXP-2580)** — lift cap to $1B+, but SPY-weekly is a correlation-redundant sleeve relative to exp1220 (ρ=+0.13 is fine)

**Recommendation:** **(C)** — EXP-2580 is a ready-to-deploy win. Continue to run SLV in parallel on a 2% weight for one more walk-forward, then retire.

**Decision 5 — Secondary broker.** Alpaca is the commission-free primary. Fallback?

- **(A) IBKR Pro Fixed** — proven, $0.65/ctr, 0.80 Sharpe penalty
- **(B) Tastytrade** — $1 open / $0 close, 0.60 Sharpe penalty, native combo support, portfolio margin at $125K
- **(C) Both IBKR AND Tastytrade** — maximum resilience, paper on both first

**Recommendation:** **(C)** — add both as fallbacks. Never be single-broker-dependent on a structural alpha claim.

**Decision 6 — Pre-2020 backtest extension.** EXP-2760 recommends this as the Scenario-B sanity check. Requires Polygon Options tier ($200/mo, not currently subscribed).

- **(A) Subscribe and run extension before Phase 9 launch** — delays paper trade by ~1 week
- **(B) Run extension in parallel with Phase 9** — paper starts on schedule, validation arrives during paper window
- **(C) Skip** — accept window-effect risk, let paper trade be the only validation

**Recommendation:** **(B)** — parallel execution. Do not delay Phase 9.

---

## 8. Rules (final, 13 items)

1. **🚫 NO SYNTHETIC DATA** — IronVault + Yahoo + public calendar only
2. **No inflated claims** — gross AND net AND broker AND expected-live, every headline
3. **Walk-forward required** — 20-fold audit before production
4. **Paper before live** — 4+ weeks validation minimum
5. **Capital utilisation must be solved** — MET Wave 6
6. **Real data trumps everything** — if model says X and data says Y, data wins
7. **MASTERPLAN is honest** — single source of truth, warts and all
8. **Capacity is a first-class target** — a winner at $50M that can't scale is half a strategy
9. **Every overlay re-tested at the portfolio level** — strategy wins ≠ portfolio wins (Bug 9)
10. **Smeared inputs are synthetic inputs** — multi-day P&L must be represented as single exit-date returns (Bug 8)
11. **Gross and net are both reported** — gross for ceiling, net for risk committee
12. **Every net Sharpe names the broker** — Alpaca 6.00 vs IBKR 5.20 is 0.80 difference on the same portfolio (Bug 11)
13. **Expected live = 0.5-0.7× backtest** — underwrite capital deployment against the decayed number, not the headline (Bug 13)

---

## 9. Production Stack (files)

### Data layer
```
data/options_cache.db          ← IronVault 276K contracts + 6.3M option-days
shared/iron_vault.py           ← canonical single provider
```

### Strategy streams
```
compass/exp1220_standalone.py         ← SPY 28d credit spreads (171 trades, 88% WR)
compass/exp2200_north_star_v6.py      ← XLF + XLI streams integration
compass/exp2250_qqq_*                 ← QQQ stream (feeds v8a)
compass/exp1770_commodity_calendars.py← GLD/SLV calendars
compass/exp2020_cross_vol_arb.py      ← IV-RV cross-sectional
compass/crisis_alpha_v5.py            ← Hedge sleeve
compass/exp2580_spy_weekly_cs.py      ← Phase 8 capacity candidate (ρ=+0.13)
compass/exp2600_north_star_v8.py      ← v8a cube builder
```

### Risk, execution, cost model
```
compass/portfolio_risk_manager.py     ← EXP-1890 · 30/30 tests · 5 components
compass/exp2370_dd_circuit_breaker.py ← causal 3% trailing-DD flatten
compass/exp2420_transaction_costs.py  ← real bid-ask + slippage + commission
compass/exp2470_execution_optimization.py ← stack A+B+C+D
compass/exp2510_broker_analysis.py    ← 3-broker comparison
compass/exp2540_regime_tc_model.py    ← regime-conditional costs
compass/exp2570_commfree_net_sharpe.py ← headline net calculator
compass/exp2640_vix_stress_hardening.py ← adaptive VIX vol-target
```

### Paper trading infra
```
compass/exp2670_paper_gonogo.py           ← 6-check go/no-go + 10 gates
compass/exp2830_paper_signal_generator.py ← daily 09:00 ET cron
compass/paper_trading_v4.py (61 tests)
compass/paper_monitor_dashboard.py
compass/execution_simulator.py (69 tests)
compass/prod_monitor.py (87 tests)
compass/scripts/generate_daily_signals.py ← import wrapper
```

### Reports
```
compass/reports/exp2200_north_star_v6.{json,html}    ← 5.96 gross headline
compass/reports/exp2280_wf_robustness.{json,html}    ← 20-fold audit
compass/reports/exp2370_dd_circuit_breaker.{json,html}
compass/reports/exp2420_transaction_costs.{json,html}
compass/reports/exp2470_execution_optimization.{json,html}
compass/reports/exp2510_broker_analysis.{json,html}
compass/reports/exp2570_commfree_net_sharpe.{json,html} ← Alpaca 6.00 headline
compass/reports/exp2580_spy_weekly_cs.{json,html}
compass/reports/exp2590_qqq_capacity_deep_dive.{json,html}
compass/reports/exp2670_paper_gonogo.{json,html}        ← pre-flight checklist
compass/reports/exp2720_dd_recovery.{json,html}         ← 11-day max recovery
compass/reports/exp2760_literature_survey.md            ← honest benchmark
compass/reports/progress_report_apr7.html
compass/reports/final_summary_apr8.html
```

---

## 10. Timeline

| Date | Milestone |
|---|---|
| 2026-04-03 | Operation Real Data deployed |
| 2026-04-05 | MASTERPLAN v6 — bug audit, 5 fixes |
| 2026-04-06 | **Wave 1** — 16 alpha discovery experiments |
| 2026-04-07 AM | **Waves 2-5** — portfolio construction, risk manager, overlay sweep |
| 2026-04-07 PM | **Waves 6-8** — first Sharpe 6 gross hit (EXP-2050), 7-stream integration (EXP-2200), robustness audit (EXP-2280), MASTERPLAN v7 |
| 2026-04-08 AM | **Waves 9-10** — transaction costs reality, DD circuit breaker, broker analysis, execution optimization, MASTERPLAN v8/v9 |
| 2026-04-08 PM | **Waves 10-11** — commission-free net 6.00 (EXP-2570), SPY-weekly + QQQ capacity (2580/2590), MASTERPLAN v10 |
| 2026-04-08 night | **Wave 11 finale** — DD recovery (2720), literature survey (2760), signal generator (2830), **MASTERPLAN v11** |
| **2026-04-09** | **Phase 9 paper trading starts on Alpaca** |
| 2026-04-16 (week 1) | First weekly gate review |
| 2026-05-07 (week 4) | Mandatory gate: ≥4 consecutive weeks within ±15% |
| 2026-05-21 (week 6) | Recommended paper window end (decision #2 option B) |
| 2026-06-04 (week 8) | Maximum paper window end (Phase 10 gate) |
| TBD after Phase 9 | Phase 10 $25K (or $100K per decision #3) live seed |
| TBD | Scaling tranches T2 → T7 ($100K → $500M) |

---

*Every extraordinary claim has been scrutinised. Every bug has been caught, documented, and turned into a rule. The backtest says Sharpe 6. The literature says expect 3.5–4.5 live. Phase 9 is tomorrow. Build on what's real.*
