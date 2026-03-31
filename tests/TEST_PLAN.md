# CC-2 Test Plan — COMPASS Phase 3

**Author:** CC-2 (Testing & Quality)
**Date:** 2026-03-20
**Branch:** compass-v2
**Status:** RECON COMPLETE — ready for Phase 3 execution

---

## 1. Current Test Inventory

### 1.1 Summary Statistics

| Metric | Value |
|--------|-------|
| Total test files | 66 (excluding `run_phase*.py`) |
| Total test functions | ~1,289 |
| Total lines (test code) | ~29,709 |
| Shared fixtures (conftest.py) | 2 (`sample_config`, `sample_price_data`) |
| JSON fixtures | 3 (Tradier chain, yfinance history, Telegram response) |
| Largest test file | `test_backtester.py` (2,228 lines, ~100 tests) |

### 1.2 File-by-File Inventory

| Test File | Lines | Tests | Module Under Test | Coverage Quality |
|-----------|------:|------:|-------------------|-----------------|
| **Alerts System** | | | | |
| `test_alert_generator.py` | 88 | 5 | `alerts.alert_generator` | PARTIAL |
| `test_alert_position_sizer.py` | 194 | 8 | `alerts.alert_position_sizer` | GOOD |
| `test_alert_router.py` | 448 | 26 | `alerts.alert_router` (full 6-stage pipeline) | EXCELLENT |
| `test_alert_schema.py` | 223 | 23 | `alerts.alert_schema` (Alert, enums, validation) | EXCELLENT |
| `test_api_failure_alerts.py` | 171 | 8 | `shared.telegram_alerts`, `execution.position_monitor` | GOOD |
| **Backtesting** | | | | |
| `test_backtester.py` | 2,228 | ~100 | `backtest.backtester.Backtester` | EXCELLENT |
| `test_backtest_integrity.py` | 367 | 25 | `backtest.backtester` (commission, capital, leverage) | GOOD |
| `test_monte_carlo.py` | 534 | 31 | `backtest.backtester` (MC randomization) | GOOD |
| **Bug Fixes (cross-cutting)** | | | | |
| `test_bug_fixes.py` | 648 | 44 | Multi-module regression tests (Bugs #7-#24) | EXCELLENT |
| **COMPASS Integration** | | | | |
| `test_compass_scanner.py` | 595 | 25 | `main.CreditSpreadSystem` COMPASS universe/state | GOOD |
| **Configuration** | | | | |
| `test_config.py` | 80 | 8 | `utils.load_config`, `validate_config` | GOOD |
| `test_contracts.py` | 189 | 12 | Fixture file contract validation | GOOD |
| **Data Layer** | | | | |
| `test_data_cache.py` | 95 | 5 | `shared.data_cache.DataCache` | PARTIAL |
| `test_database.py` | 298 | 15 | `shared.database` (trades, alerts, regime snapshots) | GOOD |
| **Execution Engine** | | | | |
| `test_execution_fixes.py` | 1,218 | 58 | `execution.execution_engine`, `position_monitor`, `risk_gate` | EXCELLENT |
| `test_execution_straddle.py` | 645 | 22 | `execution.execution_engine` (straddle paths) | GOOD |
| `test_edge_case_fixes.py` | 301 | 15 | `execution` (partial fill, stale order retry) | GOOD |
| `test_hardening.py` | 760 | 33 | `execution` (non-retryable, market clock, reconciliation) | EXCELLENT |
| `test_hardening2.py` | 821 | 47 | `execution` (rate limit, early close, orphan, commission) | EXCELLENT |
| `test_hardening3.py` | 641 | 27 | `execution` (stop-loss, WAL recovery, startup reconcile) | EXCELLENT |
| `test_position_monitor.py` | 1,038 | 40 | `execution.position_monitor` (IC, straddle, P&L) | EXCELLENT |
| **Reports & Monitoring** | | | | |
| `test_daily_report.py` | 265 | 15 | `scripts.daily_report` | GOOD |
| `test_deviation_tracker.py` | 501 | 30 | `shared.deviation_tracker`, `scripts.live_vs_backtest` | EXCELLENT |
| **Live Trading** | | | | |
| `test_live_pricing.py` | 112 | 8 | `shared.live_pricing.LivePricing` | GOOD |
| `test_live_snapshot.py` | 171 | 6 | `shared.live_snapshot.build_live_snapshot` | GOOD |
| **Macro & API** | | | | |
| `test_macro_api.py` | 662 | 53 | `api.macro_api` (FastAPI endpoints) | EXCELLENT |
| `test_macro_state_db.py` | 277 | 20 | `shared.macro_state_db` | GOOD |
| **ML / Feature Engineering** | | | | |
| `test_feature_engine.py` | 210 | 14 | `ml.feature_engine.FeatureEngine` | GOOD |
| `test_feature_logger.py` | 155 | 10 | `shared.feature_logger` | GOOD |
| `test_iv_analyzer.py` | 364 | 19 | `ml.iv_analyzer.IVAnalyzer` | EXCELLENT |
| `test_ml_pipeline.py` | 1,011 | 29 | `ml.ml_pipeline` (**BEING DELETED Phase 1**) | N/A |
| `test_signal_model.py` | 154 | 10 | `ml.signal_model.SignalModel` | PARTIAL |
| `test_combo_regime_detector.py` | 308 | 8 | `ml.combo_regime_detector` (being merged into regime.py) | GOOD |
| `test_sentiment_scanner.py` | 248 | 16 | `ml.sentiment_scanner` (**BEING DELETED Phase 1**) | N/A |
| `test_position_sizer.py` | 128 | 7 | `ml.position_sizer.PositionSizer` (class, not utilities) | PARTIAL |
| **Risk Management** | | | | |
| `test_risk_gate.py` | 287 | 24 | `alerts.risk_gate.RiskGate` (rules 0-7) | EXCELLENT |
| `test_risk_gate_macro.py` | 352 | 25 | `alerts.risk_gate.RiskGate` (COMPASS rules 8-10) | EXCELLENT |
| `test_portfolio_sizer.py` | 785 | 52 | `alerts.alert_position_sizer`, `portfolio_heat_tracker` | EXCELLENT |
| **Scheduling & Sizing** | | | | |
| `test_scheduler.py` | 164 | 17 | `shared.scheduler.ScanScheduler` | GOOD |
| `test_position_sizer.py` | 128 | 7 | `ml.position_sizer.PositionSizer` (Kelly class) | GOOD |
| **Signal & Snapshot** | | | | |
| `test_signal_scorer.py` | 217 | 18 | `shared.signal_scorer` | GOOD |
| `test_snapshot_builder.py` | 218 | 15 | `shared.snapshot_builder` | GOOD |
| **Strategy** | | | | |
| `test_iron_condor.py` | 308 | 13 | `strategy.spread_strategy` (IC path) | GOOD |
| `test_iron_condor_scanner.py` | 444 | 38 | `alerts.iron_condor_scanner/config/exit_monitor` | EXCELLENT |
| `test_spread_scoring.py` | 129 | 5 | `strategy.spread_strategy.score_opportunity` | PARTIAL |
| `test_spread_strategy_full.py` | 209 | 10 | `strategy.spread_strategy` (find spreads) | GOOD |
| `test_strategy_adapter.py` | 391 | 16 | `shared.strategy_adapter` | GOOD |
| `test_strategy_factory.py` | 226 | 22 | `shared.strategy_factory` | GOOD |
| **Straddle/Strangle** | | | | |
| `test_straddle_alerts.py` | 528 | 26 | `alerts.alert_schema`, `formatters.telegram` (straddle) | GOOD |
| **Tail Hedge** | | | | |
| `test_tail_hedge.py` | 394 | 30 | `shared.tail_hedge` | EXCELLENT |
| **Technical Analysis** | | | | |
| `test_technical_analysis.py` | 34 | 4 | `shared.indicators.calculate_rsi` | MINIMAL |
| `test_technical_analyzer.py` | 131 | 11 | `strategy.technical_analysis.TechnicalAnalyzer` | GOOD |
| **Telegram** | | | | |
| `test_telegram_alerts.py` | 221 | 16 | `shared.telegram_alerts` | GOOD |
| `test_telegram_bot.py` | 74 | 6 | `alerts.telegram_bot` | GOOD |
| `test_telegram_formatter.py` | 230 | 22 | `alerts.formatters.telegram` | GOOD |
| **Trade Tracking** | | | | |
| `test_trade_tracker.py` | 231 | 16 | `tracker.trade_tracker` | GOOD |
| **Unified Entry/Exit** | | | | |
| `test_unified_entry.py` | 333 | 26 | snapshot_builder, strategy_factory, signal_scorer, AlertRouter | GOOD |
| `test_unified_exit.py` | 352 | 14 | `execution.position_monitor` (per-trade exit params) | GOOD |
| **VIX & Regime** | | | | |
| `test_iv_rank.py` | 33 | 3 | `shared.indicators.calculate_iv_rank` | MINIMAL |
| `test_property_based.py` | 193 | 10 | Hypothesis-based property tests (IV rank, sizing, features) | GOOD |
| **Watchdog & 0-DTE** | | | | |
| `test_watchdog.py` | 249 | 22 | `scripts.watchdog` | EXCELLENT |
| `test_zero_dte_scanner.py` | 572 | 61 | `alerts.zero_dte_*` (config, scanner, exit, backtest) | EXCELLENT |

### 1.3 Existing Test Infrastructure

**conftest.py** (74 lines):
- `sample_config` — full config dict with strategy, risk, alerts, backtest sections
- `sample_price_data` — 100-row SPY-like OHLCV DataFrame (numpy seed 42, ~$450 base)

**tests/fixtures/** (3 JSON files):
- `tradier_chain_response.json` — frozen Tradier options chain with full Greeks
- `yfinance_spy_history.json` — 20-row frozen SPY OHLCV
- `telegram_send_message.json` — frozen Telegram API response

**Common test patterns across the suite:**
- `_make_config()` / `_make_position()` local factory helpers (most files)
- `tmp_path` for isolated SQLite databases per test
- `monkeypatch` for module-level env vars and DB path redirection
- `pytest.skip` at module level for optional deps (e.g., `alpaca.trading.requests`)
- `_make_price_data(values, start)` pattern in `test_combo_regime_detector.py`

---

## 2. Gap Analysis — ZERO Coverage

### GAP 1: `engine/regime.py` — RegimeClassifier (CRITICAL, P0)

**Source:** `engine/regime.py` (236 lines)
**Current tests:** ZERO. No test file exists. No imports of `engine.regime` found in tests/.
**Impact:** This module drives ALL regime-adaptive sizing in the champion strategy (+40.7% avg, ROBUST 0.951). Every trade's position size depends on this classifier. Zero tests is a critical gap.

**What needs testing:**
- `Regime` enum (5 values: BULL, BEAR, HIGH_VOL, LOW_VOL, CRASH)
- `REGIME_INFO` dict (metadata per regime)
- `RegimeClassifier.classify()` — single-day classification with VIX thresholds + trend
- `RegimeClassifier.classify_series()` — tags entire DataFrame
- `RegimeClassifier._trend_direction()` — slope-based trend detection
- `RegimeClassifier._is_declining()` — sharp decline check (>5% in 10 days)
- `RegimeClassifier.summarize()` — regime distribution stats

**Post-enhancement additions (from blueprint):**
After Phase 2 moves this to `compass/regime.py`, CC-1 will enhance with:
- Configurable thresholds via config dict
- 10-day hysteresis (from ComboRegimeDetector)
- RSI momentum signal
- VIX/VIX3M term structure signal
- Explicit shift-by-1 lookahead protection

Tests must cover both current and enhanced behaviors.

### GAP 2: `shared/macro_event_gate.py` — Event Calendar (HIGH, P1)

**Source:** `shared/macro_event_gate.py` (338 lines)
**Current tests:** ZERO. No imports of `macro_event_gate` found in tests/.
**Impact:** Event scaling modifies live position sizes (0.50-1.00x) for every trade near FOMC/CPI/NFP. Completely untested despite affecting production sizing.

**What needs testing:**
- `ALL_FOMC_DATES` — correctness and completeness (2020-2026)
- `_first_friday_of_month()` — always returns a Friday
- `_cpi_release_date()` — falls on weekday, correct month M+1 logic
- `_nfp_release_date()` — correct first-Friday-of-next-month
- `_iter_months()` — year boundary wrap (Dec→Jan)
- `get_upcoming_events()` — pre-event scaling, post-event buffers (G5), horizon window
- `compute_composite_scaling()` — per-type minimums (G4), empty list → 1.0
- `run_daily_event_check()` — integration with macro_state_db persistence

### GAP 3: `ml/position_sizer.py` utility functions — Sizing (MEDIUM, P2)

**Source:** `ml/position_sizer.py` lines 25-114 (two standalone functions)
**Current tests:** INDIRECT ONLY. `test_backtester.py` imports `get_contract_size` in 1 test. `test_alert_position_sizer.py` references `calculate_dynamic_risk` in comments but doesn't import it directly. No dedicated test file for the utility functions.
**Impact:** These functions will become `compass/sizing.py` — the core sizing API for all strategies.

**What needs testing:**
- `calculate_dynamic_risk()` — IV-rank tiers (IVR<20, 20-50, >50), flat_risk_pct bypass, max_risk_pct cap, 40% heat cap, edge cases (0 account, max heat)
- `get_contract_size()` — standard case, max_contracts cap, credit >= spread_width → 0

### GAP 4 (bonus): `compass/collect_training_data.py` — Training Pipeline

**Source:** `ml/collect_training_data.py` (687 lines)
**Current tests:** ZERO (blueprint mentions 8+ tests needed)
**Impact:** Lower priority — offline script, not production path.

---

## 3. Phase 3 Test Plan — 38+ Tests Across 4 Files

### 3.1 `tests/test_regime_classifier.py` — 15+ tests

Tests for `compass/regime.py` (or `engine/regime.py` pre-move).

| # | Test Name | Category | What It Verifies |
|---|-----------|----------|-----------------|
| 1 | `test_crash_vix_above_40_with_decline` | classify | VIX=45, >5% 10-day decline → CRASH |
| 2 | `test_high_vol_vix_above_40_no_decline` | classify | VIX=45, no decline → HIGH_VOL (not CRASH) |
| 3 | `test_high_vol_vix_above_30` | classify | VIX=32, any trend → HIGH_VOL |
| 4 | `test_bear_vix_above_25_downtrend` | classify | VIX=27, downtrend → BEAR |
| 5 | `test_bull_vix_below_20_uptrend` | classify | VIX=18, uptrend → BULL |
| 6 | `test_low_vol_vix_below_15_flat` | classify | VIX=12, flat trend → LOW_VOL |
| 7 | `test_ambiguous_uptrend_defaults_bull` | classify | VIX=22, trend>0 → BULL |
| 8 | `test_ambiguous_downtrend_high_vix_bear` | classify | VIX=23, trend<0 → BEAR |
| 9 | `test_ambiguous_downtrend_low_vix_bull` | classify | VIX=18, trend<0 → BULL (mild pullback) |
| 10 | `test_no_trend_low_vix_low_vol` | classify | VIX=16, trend=0 → LOW_VOL |
| 11 | `test_no_trend_moderate_vix_bull` | classify | VIX=19, trend=0 → BULL (neutral default) |
| 12 | `test_classify_series_multi_regime` | series | Tags multi-month DataFrame with varying VIX/prices |
| 13 | `test_classify_series_vix_default_on_missing` | series | Missing VIX dates default to 20.0 |
| 14 | `test_summarize_distribution` | summarize | Counts, percentages, transitions, avg duration |
| 15 | `test_trend_direction_short_data` | helpers | <50 days falls back to shorter window (min 10) |
| 16 | `test_is_declining_sharp_drop` | helpers | >5% drop in 10 days → True |
| 17 | `test_is_declining_gentle_drop` | helpers | 3% drop in 10 days → False |
| 18 | `test_is_declining_insufficient_data` | helpers | <10 days → False |
| 19 | `test_custom_trend_window` | config | `trend_window=20` changes classification boundary |
| 20 | `test_custom_trend_threshold` | config | `trend_threshold=10.0` raises bar for trend detection |

**Post-enhancement additions (if CC-1 adds hysteresis/RSI/VIX3M):**

| # | Test Name | Category |
|---|-----------|----------|
| 21 | `test_hysteresis_prevents_rapid_flip` | hysteresis |
| 22 | `test_rsi_momentum_boosts_bull` | rsi signal |
| 23 | `test_vix3m_term_structure_signal` | vix3m signal |
| 24 | `test_configurable_thresholds_via_dict` | config |
| 25 | `test_lookahead_protection_shift_by_1` | correctness |

### 3.2 `tests/test_event_gate.py` — 12+ tests

Tests for `compass/events.py` (or `shared/macro_event_gate.py` pre-move).

| # | Test Name | Category | What It Verifies |
|---|-----------|----------|-----------------|
| 1 | `test_fomc_dates_2026_present` | data | Known 2026 FOMC dates in `ALL_FOMC_DATES` |
| 2 | `test_fomc_emergency_dates_included` | data | 2020-03-03 and 2020-03-15 in set |
| 3 | `test_fomc_scaling_day_of` | scaling | FOMC day → 0.50x scaling |
| 4 | `test_fomc_scaling_1_day_before` | scaling | 1 day before FOMC → 0.60x |
| 5 | `test_fomc_scaling_5_plus_days` | scaling | ≥6 days out → no FOMC event returned |
| 6 | `test_post_fomc_buffer` | G5 | 1 day after FOMC → FOMC_POST at 0.70x |
| 7 | `test_cpi_date_is_weekday` | helpers | CPI release never falls on Sat/Sun |
| 8 | `test_nfp_date_is_friday` | helpers | NFP is always first Friday of next month |
| 9 | `test_iter_months_year_boundary` | helpers | Dec→Jan wraps year correctly |
| 10 | `test_composite_scaling_empty` | composite | No events → 1.0 |
| 11 | `test_composite_scaling_fomc_only` | composite | Single FOMC event → its scaling factor |
| 12 | `test_composite_scaling_fomc_plus_cpi` | composite | Concurrent events → min(fomc_floor, data_floor) |
| 13 | `test_post_cpi_buffer` | G5 | 1 day after CPI → CPI_POST at 0.80x |
| 14 | `test_post_nfp_buffer` | G5 | 1 day after NFP → NFP_POST at 0.80x |
| 15 | `test_run_daily_event_check_persists` | integration | `run_daily_event_check()` writes to DB via `set_state` |

### 3.3 `tests/test_sizing.py` — 8+ tests

Tests for `compass/sizing.py` (or `ml/position_sizer.py` utility functions pre-move).

| # | Test Name | Category | What It Verifies |
|---|-----------|----------|-----------------|
| 1 | `test_dynamic_risk_low_ivr` | IV tiers | IVR=10 → 1% of account ($1,000 on $100K) |
| 2 | `test_dynamic_risk_standard_ivr` | IV tiers | IVR=35 → 2% of account ($2,000 on $100K) |
| 3 | `test_dynamic_risk_high_ivr` | IV tiers | IVR=75 → between 2% and 3% |
| 4 | `test_dynamic_risk_flat_override` | flat mode | `flat_risk_pct=5.0` → 5% of account directly |
| 5 | `test_dynamic_risk_heat_cap` | heat cap | 38% heat used → budget reduced to fit under 40% |
| 6 | `test_dynamic_risk_heat_full` | heat cap | 40%+ heat → returns 0.0 |
| 7 | `test_dynamic_risk_max_risk_pct` | cap | `max_risk_pct=1.5` caps IV-rank result |
| 8 | `test_contract_size_standard` | contracts | $1000 risk, $5 spread, $0.65 credit → 2 contracts |
| 9 | `test_contract_size_max_cap` | contracts | Large budget → capped at `max_contracts` |
| 10 | `test_contract_size_degenerate` | contracts | credit >= spread_width → 0 |

### 3.4 `tests/test_collect_training_data.py` — 8+ tests (if time permits)

Tests for `compass/collect_training_data.py` (lower priority).

| # | Test Name | Category |
|---|-----------|----------|
| 1 | `test_feature_enrichment_all_columns` | features |
| 2 | `test_year_splitting_correct_boundaries` | splitting |
| 3 | `test_vix_percentile_computation` | features |
| 4 | `test_no_lookahead_in_features` | correctness |
| 5 | `test_handles_missing_vix_data` | edge case |
| 6 | `test_output_csv_format` | output |
| 7 | `test_dedup_across_exp400_exp401` | merging |
| 8 | `test_min_feature_count_per_row` | quality |

---

## 4. Test Fixtures & Utilities Needed

### 4.1 Shared Helper: `tests/compass_helpers.py`

Reusable fixtures for COMPASS test files. Modeled after patterns in `test_combo_regime_detector.py`.

```python
# Price data builder
def make_spy_prices(base=450.0, trend_pct=0.0, days=100, start="2024-01-02"):
    """Build SPY price Series with controllable trend.

    Args:
        base: Starting price
        trend_pct: Annualized trend percentage (+20 = 20% annual uptrend)
        days: Number of business days
        start: Start date string

    Returns:
        pd.Series with DatetimeIndex (business days)
    """

# VIX series builder
def make_vix_series(value=20.0, dates=None):
    """Build constant-VIX Series matching a price index.

    Args:
        value: Constant VIX level (or list for varying VIX)
        dates: DatetimeIndex to match

    Returns:
        pd.Series with same index
    """

# SPY DataFrame builder (for classify_series)
def make_spy_dataframe(prices_series):
    """Wrap a price Series into a DataFrame with 'Close' column."""

# Regime scenario presets
SCENARIOS = {
    "crash_2020": {"vix": 82.0, "trend": -40.0, "expected": Regime.CRASH},
    "bull_2021": {"vix": 16.0, "trend": 25.0, "expected": Regime.BULL},
    "bear_2022": {"vix": 28.0, "trend": -20.0, "expected": Regime.BEAR},
    "low_vol_2017": {"vix": 10.0, "trend": 0.0, "expected": Regime.LOW_VOL},
    "high_vol_spike": {"vix": 35.0, "trend": 5.0, "expected": Regime.HIGH_VOL},
}
```

### 4.2 Event Gate Fixtures

```python
# Known FOMC dates for deterministic testing (no date.today() dependency)
TEST_DATE_FOMC_DAY = date(2026, 1, 29)       # FOMC decision day
TEST_DATE_FOMC_MINUS_1 = date(2026, 1, 28)   # 1 day before
TEST_DATE_FOMC_PLUS_1 = date(2026, 1, 30)    # 1 day after (post-buffer)
TEST_DATE_NO_EVENTS = date(2026, 2, 15)      # Far from any event

# For CPI/NFP: specific months where we know the computed date
# CPI for Jan 2026 releases ~Feb 12, 2026 (Thursday — weekday, no adjustment)
# NFP for Jan 2026 releases first Friday of Feb 2026 = Feb 6, 2026
```

### 4.3 Sizing Fixtures

```python
# Standard test account
ACCOUNT_100K = 100_000.0
ACCOUNT_10K = 10_000.0

# Standard spread parameters
SPREAD_5_WIDE = 5.0      # $5-wide spread
CREDIT_065 = 0.65        # $0.65 credit received
# max_loss = (5.0 - 0.65) * 100 = $435 per contract
```

### 4.4 Shared conftest.py Additions

The existing `conftest.py` fixtures (`sample_config`, `sample_price_data`) are sufficient for most existing tests but not for COMPASS tests. Rather than modifying conftest.py, COMPASS-specific fixtures will live in `tests/compass_helpers.py` as importable utilities. This avoids coupling and keeps conftest lean.

---

## 5. Execution Notes

### 5.1 Import Path Strategy

Tests should import from the **final** `compass/` path when it exists (after Phase 2 move), with a fallback comment noting the pre-move path:

```python
# After Phase 2: from compass.regime import Regime, RegimeClassifier
# Pre-Phase 2 fallback:
from engine.regime import Regime, RegimeClassifier
```

### 5.2 Test Ordering

1. **`test_regime_classifier.py`** — first, since regime is the highest-priority gap
2. **`test_event_gate.py`** — second, event scaling is production-critical
3. **`test_sizing.py`** — third, utility functions with clear boundaries
4. **`test_collect_training_data.py`** — last, lower priority offline script

### 5.3 No Source Code Modifications

All tests must work against the existing source code without modifications. Tests for post-enhancement features (hysteresis, RSI, VIX3M) will be written as `@pytest.mark.skip(reason="pending CC-1 enhancement")` stubs until CC-1 delivers the enhanced `compass/regime.py`.

### 5.4 Test Runner

```bash
PYTHONPATH=. python3 -m pytest tests/test_regime_classifier.py tests/test_event_gate.py tests/test_sizing.py -v -o 'addopts='
```

### 5.5 Exit Criteria (from Blueprint)

- [ ] `tests/test_regime_classifier.py`: 15+ tests, all green
- [ ] `tests/test_event_gate.py`: 10+ tests, all green
- [ ] `tests/test_sizing.py`: 5+ tests, all green
- [ ] Total new tests: 30+ covering previously-untested critical paths
- [ ] All existing tests still pass (no regressions)
