# EXP-401 2025 Loss Investigation

**Date:** 2026-03-25
**Investigator:** Maximus
**Scope:** Why EXP-401 underperformed in 2025 and whether the MASTERPLAN's +40.7% claim is reliable

---

## Executive Summary

EXP-401's straddle/strangle (SS) leg is a **systematic money loser** — not just in 2025, but in every single year from 2020 to 2025. The credit spread (CS) leg is consistently profitable. The MASTERPLAN's reported +40.7% average return is based on a **stale confirmation backtest** that does not match current re-runs. When re-run today, EXP-401 returns +0.86% in 2025 (not +35%), with the SS leg losing $28,850 against CS gains of $30,257.

The "70.5% loss" claim cannot be confirmed from the available data. The worst the training CSV shows is a +0.86% return (essentially flat). However, the SS component's persistent losses are the critical finding regardless of the exact magnitude.

---

## Finding 1: Stale Results — The +40.7% Number Is Not Reproducible

### Evidence

The MASTERPLAN claims EXP-401 averages +40.7% annually, citing `output/regime_switching_results.json`. That file was generated on **2026-03-12 20:13 UTC** and reports:

| Year | Confirmation Return | Confirmation WR | Confirmation Trades |
|------|-------------------|-----------------|-------------------|
| 2020 | +24.07% | 78.4% | 51 |
| 2021 | +107.40% | 89.5% | 86 |
| 2022 | +8.11% | 74.4% | 39 |
| 2023 | +43.18% | 86.7% | 60 |
| 2024 | +26.44% | 79.0% | 62 |
| 2025 | +34.97% | 90.9% | 55 |

When the **exact same backtest code** is re-run today via `collect_training_data.py --config exp401`, the results are dramatically different:

| Year | Re-run Return | Re-run WR | Re-run Trades |
|------|--------------|-----------|---------------|
| 2020 | -14.26% | 48.3% | 58 |
| 2021 | +68.45% | 72.0% | 100 |
| 2022 | -28.25% | 35.3% | 51 |
| 2023 | -0.52% | 59.4% | 69 |
| 2024 | +3.83% | 61.8% | 68 |
| 2025 | +0.86% | 58.0% | 69 |

The trade counts differ (55 vs 69 in 2025), and the win rates diverge enormously (90.9% vs 58.0%). The stale results file pre-dates several code changes to the strategy and backtester (commits `ff283a0`, `5b5fa46`, `c176209`).

### Root Cause

The `regime_switching_results.json` was generated once and never re-validated after code changes. Multiple commits between 2026-03-12 and 2026-03-15 modified straddle/strangle exit logic, DTE management, and the portfolio backtester itself.

### Impact

**All MASTERPLAN performance claims for EXP-401 are unreliable.** The validation results (robust score 0.951, walk-forward 3/3, Monte Carlo pass) were computed against the stale backtest, not the current code.

---

## Finding 2: Straddle/Strangle Leg Loses Money Every Year

### Evidence

Full breakdown from `compass/training_data_exp401.csv` (415 trades, 2020-2025):

**Straddle/Strangle (SS) — 182 trades total:**

| Year | Trades | Win Rate | Total PnL | Avg PnL/Trade | Avg VIX | Avg DTE |
|------|--------|----------|-----------|---------------|---------|---------|
| 2020 | 30 | 20.0% | -$29,953 | -$999 | 30.2 | 7.6 |
| 2021 | 30 | 30.0% | -$33,154 | -$1,105 | 19.7 | 7.3 |
| 2022 | 31 | 19.4% | -$23,983 | -$774 | 25.5 | 7.6 |
| 2023 | 31 | 22.6% | -$31,763 | -$1,025 | 17.0 | 7.9 |
| 2024 | 29 | 34.5% | -$19,241 | -$663 | 16.2 | 8.3 |
| 2025 | 31 | 22.6% | -$28,850 | -$931 | 20.0 | 7.6 |
| **TOTAL** | **182** | **24.7%** | **-$166,944** | **-$917** | | |

**Credit Spread (CS) — 233 trades total:**

| Year | Trades | Win Rate | Total PnL | Avg PnL/Trade |
|------|--------|----------|-----------|---------------|
| 2020 | 28 | 78.6% | +$15,695 | +$561 |
| 2021 | 70 | 90.0% | +$101,600 | +$1,451 |
| 2022 | 20 | 60.0% | -$4,264 | -$213 |
| 2023 | 38 | 89.5% | +$31,238 | +$822 |
| 2024 | 39 | 82.1% | +$23,072 | +$592 |
| 2025 | 38 | 86.8% | +$30,257 | +$796 |
| **TOTAL** | **233** | **84.1%** | **+$197,599** | **+$848** |

The SS leg lost -$166,944 across 6 years. The CS leg made +$197,599. Net: +$30,655 — a paltry 5.1% total over 6 years on $100K capital.

### Root Cause

The SS strategy runs in **`long_pre_event` mode** (the default). This means it **buys** straddles before FOMC/CPI events, paying premium (debit). The strategy bets on a large move around the event. With a 50% stop loss, it needs the underlying to move enough to overcome:

1. The premium paid (negative `net_credit` averaging -$14.08 per unit)
2. Time decay (7-day average DTE means rapid theta burn)
3. IV crush after the event (the premium paid included elevated pre-event IV)

This is a structurally negative-edge trade in most environments. The 24.7% win rate across 182 trades over 6 years confirms this is not a 2025-specific issue — it's a fundamental strategy flaw.

### Configuration Mismatch

The paper trading config (`configs/paper_exp401.yaml`) specifies `mode: short_post_event` (sell straddles after events to capture IV crush), which is a completely different strategy with positive theta. But the backtester uses `mode: long_pre_event` (the default from `StraddleStrangleStrategy.get_default_params()`). **The paper trader and backtester are running different strategies.**

---

## Finding 3: 2025-Specific Regime Analysis

### Regime Performance (2025 SS trades only)

| Regime | Trades | Win Rate | Avg Return | Total PnL |
|--------|--------|----------|------------|-----------|
| bull | 22 | 22.7% | -30.6% | -$18,218 |
| bear | 3 | 0.0% | -47.7% | -$5,347 |
| high_vol | 2 | 0.0% | -22.2% | -$2,124 |
| low_vol | 2 | 0.0% | -79.9% | -$3,411 |
| crash | 2 | 100.0% | +50.0% | +$2,656 |

The SS strategy only profits in **crash** regimes (extreme vol spikes). In every other regime — including bull (22 trades!) — it loses consistently. The regime scales (bull=1.5x, high_vol=2.5x) actually **amplify** the losses by increasing position size in losing environments.

### Market Conditions in 2025

The 2025 market was characterized by:
- Moderate VIX (avg 20.0 for SS trades)
- Extended bull regime (22/31 SS trades in bull)
- Relatively calm FOMC/CPI events without large surprises
- Low realized volatility (avg RV20d ~15-20%)

This is the worst environment for long straddles: low vol means the options are relatively cheap but the moves are too small to overcome the debit. The only profitable SS trades came during the April 2025 tariff crash (VIX spike to 45.3).

---

## Finding 4: Exit Pattern Analysis (2025 SS)

| Exit Reason | Count | Win Rate | Avg PnL |
|------------|-------|----------|---------|
| close_expiration | 16 | 25.0% | -$656 |
| close_stop_loss | 12 | 0.0% | -$1,703 |
| close_profit_target | 3 | 100.0% | +$1,834 |

51.6% of SS trades expire worthless or near-worthless (close_expiration). 38.7% hit the 50% stop loss. Only 9.7% reach the 50% profit target. This profile is consistent with buying cheap options that decay to zero — a classic long-gamma trap in a low-vol environment.

---

## Finding 5: Parameter Decay Assessment

The CS strategy does NOT show parameter decay:

| Year | CS Win Rate | CS Avg Return |
|------|-----------|--------------|
| 2020 | 78.6% | +6.03% |
| 2021 | 90.0% | +18.05% |
| 2022 | 60.0% | -2.03% |
| 2023 | 89.5% | +9.27% |
| 2024 | 82.1% | +6.44% |
| 2025 | 86.8% | +9.19% |

CS parameters are stable. The 2022 dip is expected (bear market). 2023-2025 performance is consistent and strong.

The SS strategy shows no improvement or degradation trend — it's consistently bad:

| Year | SS Win Rate |
|------|-----------|
| 2020 | 20.0% |
| 2021 | 30.0% |
| 2022 | 19.4% |
| 2023 | 22.6% |
| 2024 | 34.5% |
| 2025 | 22.6% |

This is not parameter decay. This is a strategy that never worked in the first place.

---

## Recommendations

### Immediate (Critical)

1. **Disable the SS leg in EXP-401 paper trading.** The straddle/strangle component is a net negative in every year tested. CS-only would have returned +$197,599 over 6 years vs the blended +$30,655.

2. **Re-run all EXP-401 validation** (robust score, walk-forward, Monte Carlo) with the current codebase. The stale results in `output/` do not reflect the current system.

3. **Fix the mode mismatch.** The paper config uses `short_post_event` but the backtester uses `long_pre_event`. Either align them (test `short_post_event` in backtests) or remove the SS component.

### Short-Term

4. **If keeping SS, switch to `short_post_event` mode** (sell premium after events) and re-backtest. This captures IV crush rather than betting on moves. The paper config already specifies this mode — it's the backtester that's wrong.

5. **Increase minimum DTE for SS from 7 to 14+ days.** The current 7-day average DTE means extreme theta burn. Longer DTE gives more time for the trade thesis to play out.

6. **Invert the SS regime scales.** The current scales (bull=1.5x, high_vol=2.5x) amplify a losing strategy. If SS is kept, crash should be the only regime where it's active (the only regime where it profits).

### Long-Term

7. **Add a MASTERPLAN rule:** All performance claims must be re-validated against the current codebase before deployment. Stale results files should be timestamped and flagged if older than the most recent relevant code change.

8. **Add per-strategy P&L tracking** to the backtester summary output. The current combined metrics mask that one leg is systematically losing.

---

## Methodology

- Training data: `compass/training_data_exp401.csv` (415 trades, 2020-2025), generated by `compass/collect_training_data.py --config exp401`
- Confirmation backtest: `output/regime_switching_results.json` (generated 2026-03-12)
- Live re-run: `compass/collect_training_data.py` via `run_year_backtest_exp401(2025)` (executed 2026-03-25)
- Config: `configs/paper_exp401.yaml`, `strategies/straddle_strangle.py` defaults
- Git history: `git log --since="2026-03-10"` for code change timeline
