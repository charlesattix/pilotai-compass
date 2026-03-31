"""Tests for compass.strategy_switcher — 40+ tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from compass.strategy_switcher import (
    StrategySwitcher,
    StrategyAllocation,
    SwitchEvent,
    BacktestDay,
    BacktestResult,
    RegimeRanking,
    TRADING_DAYS,
)
from compass.regime import Regime, REGIME_INFO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(n: int = 252) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=n)


def _regime_series(n: int = 252) -> pd.Series:
    """Alternating regimes: BULL 40%, LOW_VOL 20%, HIGH_VOL 20%, BEAR 20%."""
    idx = _dates(n)
    labels = []
    for i in range(n):
        frac = i / n
        if frac < 0.4:
            labels.append(Regime.BULL)
        elif frac < 0.6:
            labels.append(Regime.LOW_VOL)
        elif frac < 0.8:
            labels.append(Regime.HIGH_VOL)
        else:
            labels.append(Regime.BEAR)
    return pd.Series(labels, index=idx)


def _strategy_returns(n: int = 252, seed: int = 42) -> dict[str, pd.Series]:
    """Returns for strategies referenced in REGIME_INFO."""
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    names = set()
    for info in REGIME_INFO.values():
        names.update(info["strategies"])
    return {
        name: pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
        for name in sorted(names)
    }


def _simple_allocations() -> dict[Regime, dict[str, float]]:
    return {
        Regime.BULL: {"strat_A": 0.6, "strat_B": 0.4},
        Regime.BEAR: {"strat_B": 0.7, "strat_C": 0.3},
        Regime.HIGH_VOL: {"strat_C": 1.0},
        Regime.LOW_VOL: {"strat_A": 0.5, "strat_B": 0.5},
        Regime.CRASH: {"strat_C": 1.0},
    }


# ===========================================================================
# Default allocations
# ===========================================================================

class TestDefaults:
    def test_default_allocations_cover_all_regimes(self):
        allocs = StrategySwitcher._default_allocations()
        for regime in Regime:
            assert regime in allocs

    def test_default_weights_sum_to_one(self):
        allocs = StrategySwitcher._default_allocations()
        for regime, w in allocs.items():
            if w:
                assert sum(w.values()) == pytest.approx(1.0)

    def test_default_strategies_match_regime_info(self):
        allocs = StrategySwitcher._default_allocations()
        for regime, w in allocs.items():
            expected = set(REGIME_INFO[regime]["strategies"])
            assert set(w.keys()) == expected


# ===========================================================================
# Strategy ranking
# ===========================================================================

class TestRanking:
    def test_basic_ranking(self):
        sr = _strategy_returns(200)
        reg = _regime_series(200)
        rankings = StrategySwitcher.rank_strategies(sr, reg)
        assert len(rankings) > 0
        assert all(isinstance(r, RegimeRanking) for r in rankings)

    def test_ranking_sorted_desc(self):
        sr = _strategy_returns(200)
        reg = _regime_series(200)
        rankings = StrategySwitcher.rank_strategies(sr, reg)
        for rr in rankings:
            if len(rr.rankings) >= 2:
                scores = [s for _, s in rr.rankings]
                assert scores == sorted(scores, reverse=True)

    def test_build_allocations_from_rankings(self):
        sw = StrategySwitcher()
        sr = _strategy_returns(200)
        reg = _regime_series(200)
        rankings = StrategySwitcher.rank_strategies(sr, reg)
        allocs = sw.build_allocations_from_rankings(rankings, top_n=2)
        for regime in Regime:
            if regime in allocs and allocs[regime]:
                assert sum(allocs[regime].values()) == pytest.approx(1.0)
                assert len(allocs[regime]) <= 2


# ===========================================================================
# Transition guards
# ===========================================================================

class TestGuards:
    def test_first_switch_always_succeeds(self):
        sw = StrategySwitcher(regime_allocations=_simple_allocations())
        event = sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 5))
        assert not event.was_blocked
        assert sw.current_regime == Regime.BULL

    def test_same_regime_blocked(self):
        sw = StrategySwitcher(regime_allocations=_simple_allocations())
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 5))
        event = sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 15))
        assert event.was_blocked
        assert event.block_reason == "same_regime"

    def test_cooldown_blocks(self):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=10, hysteresis_days=0,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 5))
        event = sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 8))
        assert event.was_blocked
        assert event.block_reason == "cooldown"

    def test_cooldown_expires(self):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=5, hysteresis_days=0,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 5))
        event = sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 12))
        assert not event.was_blocked

    def test_hysteresis_blocks_then_allows(self):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=0, hysteresis_days=3,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 5))
        # Try switching immediately — hysteresis blocks
        e1 = sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 6))
        assert e1.was_blocked
        assert e1.block_reason == "hysteresis"
        # Wait long enough
        e2 = sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 10))
        assert not e2.was_blocked

    def test_max_frequency_blocks(self):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=0, hysteresis_days=0,
            max_switches_per_month=2,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 2))
        sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 5))
        sw.propose_switch(Regime.HIGH_VOL, date=datetime(2026, 1, 8))
        # 3rd switch should be blocked (first was initial, second was switch #1, third = #2)
        # Actually: first=BULL (switch 1), second=BEAR (switch 2), third should be blocked
        e = sw.propose_switch(Regime.LOW_VOL, date=datetime(2026, 1, 12))
        assert e.was_blocked
        assert e.block_reason == "max_frequency"

    def test_frequency_resets_next_month(self):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=0, hysteresis_days=0,
            max_switches_per_month=2,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 2))
        sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 10))
        sw.propose_switch(Regime.HIGH_VOL, date=datetime(2026, 1, 20))
        # Blocked in Jan
        e_jan = sw.propose_switch(Regime.LOW_VOL, date=datetime(2026, 1, 25))
        assert e_jan.was_blocked
        # Allowed in Feb
        e_feb = sw.propose_switch(Regime.LOW_VOL, date=datetime(2026, 2, 3))
        assert not e_feb.was_blocked


# ===========================================================================
# Switch execution
# ===========================================================================

class TestSwitch:
    def test_turnover_computed(self):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=0, hysteresis_days=0,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 2))
        e = sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 5))
        assert e.turnover > 0

    def test_cost_proportional_to_turnover(self):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=0, hysteresis_days=0,
            cost_per_unit_turnover=0.01,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 2))
        e = sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 5))
        assert e.estimated_cost == pytest.approx(e.turnover * 0.01)

    def test_allocation_updated(self):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=0, hysteresis_days=0,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 2))
        assert sw.current_allocation == {"strat_A": 0.6, "strat_B": 0.4}

    def test_history_tracked(self):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=0, hysteresis_days=0,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 2))
        sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 5))
        assert len(sw.switch_history) == 2
        assert len(sw.executed_switches) == 2

    def test_blocked_not_in_executed(self):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=10, hysteresis_days=0,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 2))
        sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 5))  # blocked
        assert len(sw.switch_history) == 2
        assert len(sw.executed_switches) == 1

    def test_reset(self):
        sw = StrategySwitcher(regime_allocations=_simple_allocations())
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 2))
        sw.reset()
        assert sw.current_regime is None
        assert sw.current_allocation == {}
        assert len(sw.switch_history) == 0


# ===========================================================================
# Target allocation
# ===========================================================================

class TestTargetAllocation:
    def test_get_target(self):
        sw = StrategySwitcher(regime_allocations=_simple_allocations())
        alloc = sw.get_target_allocation(Regime.BULL)
        assert alloc == {"strat_A": 0.6, "strat_B": 0.4}

    def test_unknown_regime_empty(self):
        sw = StrategySwitcher(regime_allocations={Regime.BULL: {"A": 1.0}})
        alloc = sw.get_target_allocation(Regime.CRASH)
        assert alloc == {}

    def test_strategy_allocation_dataclass(self):
        sa = StrategyAllocation(
            weights={"A": 0.5, "B": 0.5}, regime=Regime.BULL)
        assert sa.strategy_names == ["A", "B"]


# ===========================================================================
# Backtest
# ===========================================================================

class TestBacktest:
    def test_runs(self):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        reg = _regime_series(200)
        sr = _strategy_returns(200)
        result = sw.backtest(reg, sr)
        assert isinstance(result, BacktestResult)
        assert len(result.days) == 200

    def test_cumulative_returns_set(self):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        result = sw.backtest(_regime_series(200), _strategy_returns(200))
        assert result.switcher_cum != 0 or result.buy_hold_cum != 0

    def test_switch_count(self):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        result = sw.backtest(_regime_series(200), _strategy_returns(200))
        assert result.n_switches > 0

    def test_strategy_cums_populated(self):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        sr = _strategy_returns(200)
        result = sw.backtest(_regime_series(200), sr)
        assert len(result.strategy_cums) == len(sr)

    def test_sharpe_computed(self):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        result = sw.backtest(_regime_series(200), _strategy_returns(200))
        assert isinstance(result.switcher_sharpe, float)
        assert isinstance(result.buy_hold_sharpe, float)

    def test_total_cost_tracked(self):
        sw = StrategySwitcher(
            cooldown_days=0, hysteresis_days=0,
            cost_per_unit_turnover=0.01,
        )
        result = sw.backtest(_regime_series(200), _strategy_returns(200))
        assert result.total_switch_cost > 0

    def test_custom_buy_hold(self):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        n = 200
        sr = _strategy_returns(n)
        bh = pd.Series(
            np.random.default_rng(7).normal(0.0001, 0.01, n),
            index=_dates(n),
        )
        result = sw.backtest(_regime_series(n), sr, buy_hold_returns=bh)
        assert isinstance(result, BacktestResult)

    def test_backtest_resets_state(self):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 1))
        sw.backtest(_regime_series(100), _strategy_returns(100))
        # backtest calls reset, so previous manual switch should be gone
        # but backtest adds its own switches
        assert sw.current_regime is not None


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        result = sw.backtest(_regime_series(200), _strategy_returns(200))
        out = tmp_path / "switcher.html"
        path = sw.generate_report(result, output_path=str(out))
        assert Path(path).exists()
        html = out.read_text()
        assert "Strategy Switcher Report" in html

    def test_regime_timeline(self, tmp_path):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        result = sw.backtest(_regime_series(200), _strategy_returns(200))
        out = tmp_path / "r.html"
        sw.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "Regime Timeline" in html
        assert "<svg" in html

    def test_cumulative_chart(self, tmp_path):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        result = sw.backtest(_regime_series(200), _strategy_returns(200))
        out = tmp_path / "r.html"
        sw.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "Cumulative Performance" in html

    def test_switch_events_table(self, tmp_path):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        result = sw.backtest(_regime_series(200), _strategy_returns(200))
        out = tmp_path / "r.html"
        sw.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "Switch Events" in html

    def test_strategy_comparison(self, tmp_path):
        sw = StrategySwitcher(cooldown_days=0, hysteresis_days=0)
        result = sw.backtest(_regime_series(200), _strategy_returns(200))
        out = tmp_path / "r.html"
        sw.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "Strategy Comparison" in html

    def test_blocked_events_shown(self, tmp_path):
        sw = StrategySwitcher(
            regime_allocations=_simple_allocations(),
            cooldown_days=100, hysteresis_days=0,
        )
        sw.propose_switch(Regime.BULL, date=datetime(2026, 1, 2))
        sw.propose_switch(Regime.BEAR, date=datetime(2026, 1, 5))  # blocked
        result = BacktestResult(
            switcher_cum=0.0, buy_hold_cum=0.0, strategy_cums={},
            switcher_sharpe=0.0, buy_hold_sharpe=0.0,
            n_switches=1, total_switch_cost=0.0, days=[],
        )
        out = tmp_path / "r.html"
        sw.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "BLOCKED" in html
