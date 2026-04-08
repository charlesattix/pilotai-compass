# compass/archive/ — Archived Experiments

Killed, retracted, and superseded experiments moved out of the
`compass/` root during the EXP-2770 code cleanup (2026-04-08). Every
file here is preserved intact — `git mv` retained history, and verdicts
are recorded in `experiments/registry.json`.

**None of these files are imported by the live 8-stream North Star v8
portfolio.** If you need one, either (a) import from
`compass.archive.<name>` (the folder is a package via inherited
namespace; see note below) or (b) move it back to `compass/` root
explicitly.

> Note: `compass/archive/` does NOT have an `__init__.py`. This is
> deliberate — it prevents accidental imports from production code.
> If you want to test an archived module interactively, use
> `sys.path.insert(0, "compass/archive")` in a throwaway script.

## Contents (23 files)

### Honest kills — strategies that failed OOS validation

| File | Source exp | Verdict reason |
|---|---|---|
| `exp1760_crypto_vol.py` | EXP-1760 | Crypto vol small-sample, Sharpe 1.04 |
| `exp1910_intraday_breakout.py` | EXP-1910 | Sharpe 0.31, no edge |
| `exp1920_carry_trade.py` | EXP-1920 | Sharpe 0.72, regime-dependent |
| `exp1930_vvix_signal.py` | EXP-1930 | +0.12 Δ Sharpe, under threshold |
| `exp1940_multi_tf_momentum.py` | EXP-1940 | Sharpe 0.14 long/short, 0.76 long-only |
| `exp1950_adaptive_kelly.py` | EXP-1950 | +0.03 Δ Sharpe, noise |
| `exp1990_meta_learner.py` | EXP-1990 | Overfit 10 features on 141-trade OOS |
| `exp2030_seasonality_overlay.py` | EXP-2030 | Sharpe 0.42, patterns didn't persist |
| `exp2090_calendar_seasonality.py` | EXP-2090 | GLD −0.42 / SLV −0.29 Δ Sharpe |
| `exp2150_higher_frequency.py` | EXP-2150 | Weekly + T+V filters hurt portfolio |
| `exp2170_weight_optimization.py` | EXP-2170 | Sharpe 5.47, target 6.0 not met |
| `exp2190_tail_risk_parity.py` | EXP-2190 | Reactive triggers can't predict DD |
| `exp2260_slv_replacement.py` | EXP-2260 | No clean SLV replacement found |
| `exp2310_aum_scaling.py` | EXP-2310 | IronVault universe cannot answer |
| `exp2350_slv_replacement_v2.py` | EXP-2350 | QQQ/TLT fail combined Sharpe + capacity |
| `exp2380_futures_calendar_capacity.py` | EXP-2380 | Futures ADV ≈ ETF option ADV |
| `exp2430_capacity_optimized.py` | EXP-2430 | Dropping SLV reveals XLI as bottleneck |
| `exp2460_zero_cost_overlay.py` | EXP-2460 | Negative on diversified portfolio |
| `exp2480_three_sleeve_hicap.py` | EXP-2480 | −0.33 Sharpe, only 1.31× capacity |

### Superseded — replaced by a later experiment in the same family

| File | Source exp | Superseded by |
|---|---|---|
| `exp2050_north_star_v5.py` | EXP-2050 | EXP-2200 (v6) / EXP-2250 (v7) / EXP-2600 (v8) |
| `exp2100_vf_true_integration.py` | EXP-2100 | Retracted |
| `exp2250_north_star_v7.py` | EXP-2250 | EXP-2600 (v8) |
| `exp2320_final_report.py` | EXP-2320 | EXP-2680 (MASTERPLAN v10) |

### Tests

| File | Notes |
|---|---|
| `tests/test_exp1760_crypto_vol.py` | Test suite for the archived crypto-vol experiment |

## Why these are archived (not deleted)

1. **Honest-negative provenance.** The registry claim "3 of 4 rails MET"
   is only trustworthy because the kills are documented and replicable.
   Deleting them would erase the evidence trail.
2. **Git history is easier to search with files present.** `git log --follow
   compass/archive/exp1910_intraday_breakout.py` immediately surfaces
   every change and the original rejection commit.
3. **Future resurrection.** A strategy that failed in 2020-2025 data
   might work in a different regime. The files are cheap to keep
   and easy to un-archive.

## Not archived (despite being "retracted")

EXP-2360, 2390, 2400, and 2450 had their headline Sharpe numbers retracted
after smeared-input audits, but the covariance and sparsity math in those
files is still imported by downstream LIVE experiments (EXP-2500, 2550,
2560, 2600, 2610, 2710). Those files remain in `compass/` root with their
retraction noted in the registry.

---

See `../README.md` for the full North Star v8a architecture and the live
production layout.
