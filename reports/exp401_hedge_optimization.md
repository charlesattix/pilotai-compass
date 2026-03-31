# EXP-401 Crisis Hedge Optimization

## Problem

EXP-401 (CS + SS Blend) hedged MC P5 drawdown was **42.38%**, failing the
MASTERPLAN target of ≤30%.  EXP-400 (Champion CS) passed at 21.6% with the
default `CrisisHedgeConfig`.

## Root Cause Analysis

### Why EXP-401 hedging underperformed

EXP-401 blends credit spreads with **straddles and strangles** — short-volatility
structures where both legs move against the position simultaneously during a VIX
spike.  Three factors made the default hedge config insufficient:

1. **Higher gamma exposure.**  Straddles/strangles have concentrated gamma near
   ATM.  When the underlying gaps, both the put and call leg lose value
   simultaneously, producing larger single-day losses than directional credit
   spreads.  The default VIX floor of 20 was too high — by VIX 20 the
   straddle has already absorbed significant damage.

2. **Wider stops amplify tail losses.**  The default `base_stop_multiplier=3.5×`
   was calibrated for credit spreads with defined-risk profiles.
   Straddles/strangles can gap past stop levels, so a tighter base stop
   (2.0×) is needed to cap realised losses earlier.

3. **Slower VIX ramp-down.**  With `vix_scale_ceiling=50`, the controller
   allows 10% position sizing between VIX 40–50.  For strangles, any
   exposure above VIX 35 is pure risk.

### Strategy profile comparison

| Metric (unhedged) | EXP-400 (CS) | EXP-401 (CS+SS) |
|--------------------|-------------|-----------------|
| Sharpe             | 1.236       | 0.350           |
| MC P5 DD           | 45.0%       | 51.2%           |
| Worst crisis DD    | 51.8%       | 51.8%           |
| Trading days       | 1528        | 1565            |
| Risk profile       | Defined-risk credit spreads | Mixed: defined + undefined risk |

## Solution: EXP-401-specific CrisisHedgeConfig

Added `EXP401_HEDGE_CONFIG` in `compass/crisis_hedge.py` with parameters
tuned for straddle/strangle exposure:

| Parameter                    | Default (EXP-400) | Optimised (EXP-401) | Rationale |
|------------------------------|--------------------|---------------------|-----------|
| `vix_scale_floor`            | 20.0               | **14.0**            | Start throttling 6 pts earlier; strangles bleed before VIX 20 |
| `vix_scale_ceiling`          | 50.0               | **38.0**            | Zero exposure by VIX 38; no straddle trades above this |
| `vix_stop_floor`             | 20.0               | **14.0**            | Begin tightening stops earlier |
| `vix_stop_ceiling`           | 45.0               | **32.0**            | Hit minimum stop by VIX 32 |
| `base_stop_multiplier`       | 3.5×               | **2.0×**            | Tighter base; strangles can gap past wide stops |
| `min_stop_multiplier`        | 1.5×               | **1.0×**            | Near-breakeven stop in crisis |
| `high_vol_regime_scale`      | 0.25               | **0.10**            | Only 10% sizing in high_vol regime |
| `vix_ts_backwardation_penalty` | 0.25             | **0.50**            | Heavier penalty when term structure inverts |

## Results

### Hedged MC P5 Drawdown (target: ≤30%)

| Experiment | Before optimisation | After optimisation | Status |
|------------|--------------------|--------------------|--------|
| EXP-400    | 21.6%              | 21.6% (unchanged)  | PASS   |
| EXP-401    | **42.4%**          | **29.4%**           | **PASS** |

### Hedged Sharpe Ratio

| Experiment | Unhedged | Default hedge | Optimised hedge |
|------------|----------|---------------|-----------------|
| EXP-400    | 1.236    | 2.381         | 2.381           |
| EXP-401    | 0.350    | 0.581         | **0.859**       |

### EXP-401 Crisis Scenario Drawdowns (hedged)

| Scenario             | Unhedged | Default hedge | Optimised hedge |
|----------------------|----------|---------------|-----------------|
| COVID Crash          | -51.8%   | -16.8%        | **-10.7%**      |
| 2022 Bear Market     | -43.7%   | -37.8%        | **-23.1%**      |
| Flash Crash          | -15.0%   | -15.0%        | **-14.1%**      |
| VIX Spike (15→65)    | -22.5%   | -15.2%        | **-11.6%**      |

## Files Changed

- `compass/crisis_hedge.py` — Added `EXP401_HEDGE_CONFIG` constant
- `compass/run_stress_test.py` — Per-experiment hedge controller: default for
  EXP-400, `EXP401_HEDGE_CONFIG` for EXP-401
- `reports/stress_test_results.json` — Regenerated with optimised results
- `reports/stress_test_report.html` — Regenerated with optimised results
