# compass/experiments — Experiment archive layout

This directory holds experiments that have been retired from the active
production codebase (which lives at `compass/` root). Active v8a
pipeline files remain at `compass/` — only experiments with a clear
terminal verdict are moved here.

## Structure

```
compass/experiments/
├── killed/     experiments with a NEGATIVE verdict (rejected or retracted)
├── archived/   experiments with a neutral or informational verdict that
│               are no longer in the active pipeline but were not killed
└── active/     (currently unused — live experiments stay at compass/ root)
```

## Placement rules

- **`killed/`** — An experiment goes here when it was explicitly
  rejected in a MASTERPLAN retraction, failed walk-forward OOS, was
  superseded by a cleaner implementation, or has a verdict of KILLED
  / REJECTED / NEGATIVE in its commit message. It stays importable
  so git-archaeology still works but is out of the active namespace.

- **`archived/`** — An experiment goes here when it was a one-shot
  report generator, a historical snapshot, or an infrastructure
  scaffold that was useful at the time but is no longer referenced
  by the production pipeline. Neutral verdict.

- **`active/`** — Reserved. Active production experiments live at
  `compass/` root, not here.

## Invariant

No file in `compass/experiments/` is imported by any production-path
module at `compass/` root. If you need to re-activate a killed or
archived experiment, move it back to `compass/` and add an import
entry to the v8a production stack documented in `MASTERPLAN.md`.

## Current contents

### killed/ (25 entries)

Experiments rejected on honest OOS or retracted after an audit:

- **EXP-1760** crypto_vol — marginal, low Sharpe, no capacity
- **EXP-1910** intraday_breakout — killed (daily-OHLC proxy)
- **EXP-1920** carry_trade — killed (Sharpe below target)
- **EXP-1930** vvix_signal — killed (OOS +0.05, parameter-sweep artifact)
- **EXP-1940** multi_tf_momentum — marginal
- **EXP-1950** adaptive_kelly — killed (+0.03 Sharpe, no edge)
- **EXP-1990** meta_learner — killed (OOS 1.73 vs baseline 1.78)
- **EXP-2030** seasonality_overlay — killed (pooled OOS −0.13)
- **EXP-2050** north_star_v5 — superseded by v6/v8a
- **EXP-2090** calendar_seasonality — rejected (pre-pandemic patterns didn't persist)
- **EXP-2100** vf_true_integration — superseded
- **EXP-2150** higher_frequency — superseded
- **EXP-2170** weight_optimization — superseded by Ledoit-Wolf in EXP-2360
- **EXP-2190** tail_risk_parity — rejected (reactive triggers don't predict DD)
- **EXP-2250** north_star_v7 — superseded by v8a
- **EXP-2260** slv_replacement — rejected
- **EXP-2310** aum_scaling — superseded
- **EXP-2320** final_report — superseded by v8 / v11 reports
- **EXP-2350** slv_replacement_v2 — rejected (combined Sharpe + capacity bar missed)
- **EXP-2380** futures_calendar_capacity — rejected (futures ≈ ETF option spreads)
- **EXP-2430** capacity_optimized — rejected (XLI becomes next bottleneck)
- **EXP-2460** zero_cost_overlay — killed (NEGATIVE on diversified portfolio)
- **EXP-2480** three_sleeve_hicap — rejected (−0.33 Sharpe, only 1.3× capacity)
- plus `tests/` — moved along with the experiments they tested

### archived/ (empty at creation)

Populated over time as neutral-verdict experiments are cleared out of
compass/ root.

### active/ (empty, reserved)
