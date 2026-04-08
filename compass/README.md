# compass/ — Strategy Research and Production Code

North Star v8a (2026-04-08). This directory holds the research experiments,
production strategy engines, risk management, and reporting code for the
8-stream Ledoit-Wolf North Star portfolio.

> Canonical single source of truth: `../MASTERPLAN.md` v10.
> Go/no-go dashboard: `../compass/reports/master_dashboard_apr8.html`.

---

## North Star v8a Architecture — 8 Streams

| # | Stream | Module | Weight | Baseline Sh | ρ → EXP-1220 |
|---|---|---|---:|---:|---:|
| 1 | SPY put credit spreads | `exp1220_standalone.py` | 35% | 3.85 | 1.00 |
| 2 | QQQ put credit spreads | `exp2240_qqq_iwm_credit_spreads.py` | 15% | 2.26 | +0.24 |
| 3 | XLF put credit spreads | `exp2210_xlf_xli_validation.py` | 10% | 2.06 | +0.12 |
| 4 | XLI put credit spreads | `exp2210_xlf_xli_validation.py` | 10% | 2.25 | −0.01 |
| 5 | GLD calendar (GC=F) | `exp1770_commodity_calendars.py` | 10% | 2.70 | +0.03 |
| 6 | SLV calendar (SI=F) | `exp1770_commodity_calendars.py` | 5% | 2.27 | −0.03 |
| 7 | Cross-vol arb | `exp2020_cross_vol_arb.py` | 10% | 1.80 | +0.00 |
| 8 | Crisis Alpha v5 | `crisis_alpha_v5.py` | 5% | 1.20 | −0.08 |
| — | Cash buffer (T-bill) | — | 10% | — | — |

Blended 8-stream Ledoit-Wolf gross Sharpe: **6.87**. Net on Alpaca
commission-free: **6.00**. Max DD with EXP-2370 circuit breaker: **4.2%**.

---

## Signal Flow

```
┌───────────────────────────────────────────────────────────────────┐
│ compass.paper_engine (production loop, Alpaca paper API)           │
└───────────────┬───────────────────────────────────────────────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌───────────────┐   ┌────────────────┐
│ Data layer    │   │ Signal layer    │
│ IronVault +   │   │ (per stream)    │
│ Yahoo + Fed   │   └───────┬────────┘
└───────────────┘           │
                            ▼
                ┌───────────────────────────┐
                │ Entry overlays            │
                │ 1. FOMC gate (1740)       │
                │ 2. P/C overlay (1750)     │
                │ 3. Regime-TC skip (2540)  │
                │ 4. VoV overlay (1970)     │
                │ 5. Term structure (2070)  │
                └──────────────┬────────────┘
                               │
                               ▼
                ┌───────────────────────────┐
                │ Sizing / Risk             │
                │ 1. Ledoit-Wolf covariance │
                │    (2400, kept for matrix)│
                │ 2. Risk parity weights    │
                │    (1890 PortfolioRiskMgr)│
                │ 3. Vol targeting (2180)   │
                │ 4. Circuit breaker (2370) │
                └──────────────┬────────────┘
                               │
                               ▼
                ┌───────────────────────────┐
                │ Execution stack (2470)    │
                │ A. Limit-at-mid patient   │
                │ B. End-of-day window      │
                │ C. Route bias             │
                │ D. Combo orders           │
                └──────────────┬────────────┘
                               │
                               ▼
                ┌───────────────────────────┐
                │ Alpaca paper API          │
                │ (commission-free, EXP-2510)│
                └──────────────┬────────────┘
                               │
                               ▼
                ┌───────────────────────────┐
                │ Monitoring + Alerts       │
                │ - 5-min health poller     │
                │ - Telegram alerts         │
                │ - Daily P&L report        │
                └───────────────────────────┘
```

---

## Cost Model (EXP-2420 / 2470 / 2510 / 2540)

Total cost stack (bps/year on leveraged notional):

| Component | Source | Bps/yr | Notes |
|---|---|---:|---|
| Commission (IBKR Pro baseline) | EXP-2510 | 827 | $0.65/contract/leg |
| Commission (Alpaca) | EXP-2510 | **0** | Commission-free options tier |
| Base bid-ask slippage | EXP-2420 | ~5 | Per active day, LOW regime |
| Execution stack credit | EXP-2470 | −503 | A+B+C+D techniques |
| Regime-TC CRISIS premium | EXP-2540 | +7.3 | 2.5× LOW on VIX ≥ 35 |
| **Net drag (Alpaca path)** | — | **~230** | Production config |
| **Net drag (IBKR path)** | — | **~1057** | Fallback config |

Net Sharpe implication:
- Gross 6.87 − IBKR drag = **5.20**
- Gross 6.87 − Alpaca drag = **6.00** ★ headline

---

## Directory Layout (v8a, post EXP-2770 cleanup)

```
compass/
├── README.md                             ← you are here
├── archive/                              ← 23 killed/superseded experiments
│   ├── exp1760_crypto_vol.py             ← crypto, small sample
│   ├── exp1910_…1990_*.py                ← Wave 4 alpha hunt kills
│   ├── exp2030_seasonality_overlay.py
│   ├── exp2090_calendar_seasonality.py
│   ├── exp2100_vf_true_integration.py    ← retracted
│   ├── exp2190_tail_risk_parity.py
│   ├── exp2260, 2350                     ← SLV replacement kills
│   ├── exp2310_aum_scaling.py
│   ├── exp2380_futures_calendar_capacity.py
│   ├── exp2430_capacity_optimized.py
│   ├── exp2460_zero_cost_overlay.py
│   ├── exp2480_three_sleeve_hicap.py
│   ├── exp2050_north_star_v5.py          ← superseded by v6/v7/v8
│   ├── exp2250_north_star_v7.py          ← superseded by v8
│   ├── exp2320_final_report.py           ← superseded by exp2680
│   └── tests/test_exp1760_crypto_vol.py
│
├── Foundation strategies (LIVE)
│   ├── exp1220_standalone.py             ← 171 real trades, 88% WR
│   ├── exp1770_commodity_calendars.py    ← GLD/SLV calendars
│   ├── exp1770_commodity_spreads.py
│   ├── exp2020_cross_vol_arb.py          ← Cross-vol arb sleeve
│   ├── exp2040_leveraged_calendars.py    ← Leveraged GLD/SLV
│   ├── exp2210_xlf_xli_validation.py     ← XLF/XLI CS
│   ├── exp2240_qqq_iwm_credit_spreads.py ← QQQ CS (+ signal entry)
│   ├── exp2580_spy_weekly_cs.py          ← SPY weekly (Phase 8)
│   ├── exp2600_north_star_v8.py          ← 8-stream headline
│   ├── crisis_alpha_v5.py                ← Tail hedge sleeve
│   └── tail_risk_hedge.py
│
├── Entry overlays (LIVE)
│   ├── exp1740_sentiment_filter.py       ← FOMC NLP
│   ├── exp1750_putcall_overlay.py        ← P/C ratio
│   ├── exp1780_exp1220_integration.py
│   ├── exp1880_integrated_overlays.py
│   ├── exp1960_skew_alpha.py
│   ├── exp1970_vol_of_vol.py             ← VoV overlay (+0.86)
│   ├── exp2000_triple_overlay.py
│   ├── exp2010_tail_convexity.py
│   ├── exp2070_term_structure.py         ← VIX term (+1.42)
│   ├── exp2080_corr_regime.py            ← Corr regime switching
│   └── exp2120_triple_overlay.py         ← T+V+F winner
│
├── Portfolio construction (LIVE)
│   ├── exp1850_regime_portfolio.py       ← Regime-adaptive
│   ├── exp1980_dynamic_hedge.py
│   ├── exp2060_cross_vol_arb_v2.py
│   ├── exp2110_leveraged_diversified.py
│   ├── exp2160_high_capacity_alts.py
│   ├── exp2180_vol_targeting.py
│   ├── exp2200_north_star_v6.py          ← Original 7-stream
│   ├── exp2220_seven_stream_corr.py
│   ├── exp2230_capacity_xlf_xli.py
│   ├── exp2280_wf_robustness.py          ← 20-fold walk-forward
│   ├── exp2300_portfolio_runner.py
│   └── exp2400_combined_best_of.py       ← Ledoit-Wolf covariance ref
│
├── Capacity + risk (LIVE)
│   ├── exp1890 → portfolio_risk_manager.py
│   ├── exp2140_portfolio_capacity.py
│   ├── exp2270_xlf_xli_slippage.py
│   ├── exp2330_mc_stress_test.py         ← MC 6/6 gates
│   ├── exp2340_dd_deep_dive.py
│   ├── exp2360_robust_cov.py             ← Retracted Sh but infra kept
│   ├── exp2370_dd_circuit_breaker.py     ← ★★ Winner
│   ├── exp2390_robust_cov_audit.py
│   ├── exp2440_cost_aware_optimization.py
│   ├── exp2450_sparse_combined_honest.py ← Retraction doc
│   ├── exp2630_regime_stress_oos.py      ← OOS stress test
│   ├── exp2640_vix_stress_hardening.py
│   ├── exp2650_multi_expiry_capacity.py
│   ├── exp2660_aum_capacity_scaling.py
│   ├── exp2720_dd_recovery.py
│   └── risk_overlay.py
│
├── Cost / execution / broker (LIVE)
│   ├── exp2420_transaction_costs.py      ← Real cost baseline
│   ├── exp2470_execution_optimization.py ← +0.33 stack
│   ├── exp2500_true_net_backtest.py
│   ├── exp2510_broker_analysis.py        ← 3-broker comparison
│   ├── exp2540_regime_tc_model.py        ← +0.83 regime skip
│   ├── exp2550_net_sharpe_recovery.py
│   ├── exp2560_trade_frequency_compression.py
│   └── exp2570_commfree_net_sharpe.py    ← ★★★ Net Sh 6.00 headline
│
├── Deployment infra (LIVE)
│   ├── exp1660_vrp_deepening.py
│   ├── exp2290 → (configs/north_star_v6_prod.yaml)
│   ├── exp2590_qqq_capacity_deep_dive.py
│   ├── exp2600_north_star_v8.py
│   ├── exp2610_spy_weekly_integration.py
│   ├── exp2620_alpaca_connector.py
│   ├── exp2670_paper_gonogo.py
│   ├── exp2680 → (MASTERPLAN.md v10 + final report)
│   ├── exp2690_signal_generators.py      ← Production signal entry
│   ├── exp2700_reproducibility_audit.py
│   ├── exp2710_xle_integration.py
│   ├── portfolio_risk_manager.py         ← EXP-1890
│   ├── paper_engine.py
│   ├── telegram_alerter.py
│   ├── metrics.py                        ← Canonical Sharpe
│   ├── gld_tlt_relval.py
│   └── capacity_analyzer.py
│
└── reports/                              ← Per-experiment JSON+HTML
    ├── master_dashboard_apr8.html        ← Carlos single-file dashboard
    ├── north_star_v8_final.html          ← Final go/no-go package
    └── exp*.json, exp*.html              ← Per-experiment outputs
```

---

## How to Run

### Single experiment
```bash
python3 -m compass.exp2590_qqq_capacity_deep_dive
```

### Paper trading engine (v8 production)
```bash
./scripts/launch_north_star_v6.sh smoke      # validate config
./scripts/launch_north_star_v6.sh dry        # scan, no orders
./scripts/launch_north_star_v6.sh daemon     # start engine + monitor
./scripts/launch_north_star_v6.sh status     # health snapshot
./scripts/launch_north_star_v6.sh report     # daily P&L report
```

### Master dashboard
```bash
open compass/reports/master_dashboard_apr8.html
```

---

## Rule Zero

Every price, every fill, every signal traces to real data:
- **IronVault** `data/options_cache.db` (276K contracts, 6.3M option-days)
- **Yahoo Finance** for underlying prices, VIX term structure
- **federalreserve.gov** for FOMC minutes (NLP sentiment)

Zero synthetic data. Zero `np.random.normal()` in any P&L path. If a quote
is missing the trade is skipped, never extrapolated.

---

## Archived Experiments (post-EXP-2770 cleanup)

See `compass/archive/` for killed and superseded experiments. Each archived
file is preserved intact — `git mv` retained history. Full verdicts and
retraction notes are in `experiments/registry.json`.

Categories of archived code:
- **Honest kills (19):** strategies that failed OOS validation
  (EXP-1760, 1910, 1920, 1930, 1940, 1950, 1990, 2030, 2090, 2150, 2170,
  2190, 2260, 2310, 2350, 2380, 2430, 2460, 2480)
- **Retractions / superseded headlines (4):**
  (EXP-2050 north_star_v5 → v8, EXP-2100 retracted, EXP-2250 north_star_v7
  → v8, EXP-2320 → EXP-2680)

Note on retracted infrastructure: EXP-2360 / 2390 / 2400 / 2450 had their
headline Sharpe numbers retracted after smeared-input audits, but the
covariance and sparsity math in those files is still imported by downstream
live experiments. They remain in `compass/` root (not archived) as
infrastructure, with their retraction documented in the registry.
