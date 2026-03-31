"""Tests for compass.options_strategy — 32 tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from compass.options_strategy import (
    OptionsStrategyEngine, StrategyType, Direction, Strategy, OptionLeg,
    RollSignal, ScenarioResult, StrategyRecommendation,
)


class TestVertical:
    def test_bull_put(self):
        eng = OptionsStrategyEngine()
        s = eng.build_vertical(450, 440, 435, 45, 0.20, 0.22, is_put=True)
        assert s.strategy_type == StrategyType.VERTICAL
        assert s.direction == Direction.BULL
        assert s.max_loss > 0
        assert s.net_premium >= 0

    def test_bear_call(self):
        eng = OptionsStrategyEngine()
        s = eng.build_vertical(450, 460, 465, 45, 0.20, 0.22, is_put=False)
        assert s.direction == Direction.BEAR

    def test_max_loss_bounded(self):
        eng = OptionsStrategyEngine()
        s = eng.build_vertical(100, 95, 90, 30, 0.25, 0.25, True)
        assert s.max_loss <= 5.5  # width + small premium

    def test_greeks_populated(self):
        eng = OptionsStrategyEngine()
        s = eng.build_vertical(100, 95, 90, 45, 0.20, 0.22, True)
        assert s.net_delta != 0
        assert len(s.legs) == 2


class TestIronCondor:
    def test_basic(self):
        eng = OptionsStrategyEngine()
        ic = eng.build_iron_condor(450, 430, 425, 470, 475, 45, 0.20)
        assert ic.strategy_type == StrategyType.IRON_CONDOR
        assert ic.direction == Direction.NEUTRAL
        assert len(ic.legs) == 4

    def test_max_profit_positive(self):
        eng = OptionsStrategyEngine()
        ic = eng.build_iron_condor(100, 90, 85, 110, 115, 45, 0.25)
        assert ic.max_profit > 0

    def test_four_legs(self):
        eng = OptionsStrategyEngine()
        ic = eng.build_iron_condor(100, 90, 85, 110, 115, 45, 0.20)
        assert len(ic.legs) == 4


class TestButterfly:
    def test_basic(self):
        eng = OptionsStrategyEngine()
        b = eng.build_butterfly(100, 95, 100, 105, 45, 0.20)
        assert b.strategy_type == StrategyType.BUTTERFLY
        assert len(b.legs) == 4

    def test_max_loss_bounded(self):
        eng = OptionsStrategyEngine()
        b = eng.build_butterfly(100, 95, 100, 105, 45, 0.20)
        assert b.max_loss >= 0


class TestCalendar:
    def test_basic(self):
        s = OptionsStrategyEngine.build_calendar(100, 100, 30, 60, 0.20, 0.22)
        assert s.strategy_type == StrategyType.CALENDAR
        assert len(s.legs) == 2


class TestSelection:
    def test_crash_regime(self):
        stype, direction = OptionsStrategyEngine.select_strategy("crash", 80, 0.10)
        assert stype == StrategyType.VERTICAL
        assert direction == Direction.BEAR

    def test_high_vol(self):
        stype, _ = OptionsStrategyEngine.select_strategy("high_vol", 60, 0.05)
        assert stype == StrategyType.IRON_CONDOR

    def test_low_vol(self):
        stype, _ = OptionsStrategyEngine.select_strategy("low_vol", 20, 0.02)
        assert stype == StrategyType.CALENDAR

    def test_bull(self):
        stype, _ = OptionsStrategyEngine.select_strategy("bull", 40, 0.08)
        assert stype in (StrategyType.VERTICAL, StrategyType.IRON_CONDOR)


class TestRecommend:
    def test_basic(self):
        eng = OptionsStrategyEngine()
        rec = eng.recommend(450, "bull", 60, 0.05)
        assert isinstance(rec, StrategyRecommendation)
        assert rec.strategy is not None
        assert rec.score != 0


class TestSizing:
    def test_size_by_max_loss(self):
        eng = OptionsStrategyEngine(max_loss_pct=0.02)
        s = Strategy("test", StrategyType.VERTICAL, Direction.BULL, [],
                      max_loss=2.0, max_profit=1.0, margin_required=2.0)
        n = eng.size_by_max_loss(s, 100000)
        assert n >= 1
        assert n * s.max_loss * 100 <= 100000 * 0.02

    def test_zero_max_loss(self):
        eng = OptionsStrategyEngine()
        s = Strategy("test", StrategyType.VERTICAL, Direction.BULL, [], max_loss=0)
        assert eng.size_by_max_loss(s, 100000) == 0


class TestRoll:
    def test_time_trigger(self):
        r = OptionsStrategyEngine.check_roll(10, 0.15, dte_trigger=14)
        assert r.should_roll
        assert r.trigger_type == "time"

    def test_delta_trigger(self):
        r = OptionsStrategyEngine.check_roll(30, 0.35, delta_trigger=0.30)
        assert r.should_roll
        assert r.trigger_type == "delta"

    def test_no_trigger(self):
        r = OptionsStrategyEngine.check_roll(30, 0.15)
        assert not r.should_roll


class TestMarginEfficiency:
    def test_basic(self):
        s = Strategy("t", StrategyType.VERTICAL, Direction.BULL, [],
                      max_profit=1.5, margin_required=5.0)
        assert OptionsStrategyEngine.margin_efficiency(s) == pytest.approx(0.30)

    def test_zero_margin(self):
        s = Strategy("t", StrategyType.VERTICAL, Direction.BULL, [], margin_required=0)
        assert OptionsStrategyEngine.margin_efficiency(s) == 0.0


class TestScenario:
    def test_basic(self):
        eng = OptionsStrategyEngine()
        s = eng.build_vertical(100, 95, 90, 45, 0.20, 0.22, True)
        results = eng.scenario_analysis(s, 100, n_prices=5, horizons=[0, 30])
        assert len(results) == 10
        assert all(isinstance(r, ScenarioResult) for r in results)


class TestReport:
    def test_creates_file(self, tmp_path):
        eng = OptionsStrategyEngine()
        s = eng.build_vertical(100, 95, 90, 45, 0.20, 0.22, True)
        out = tmp_path / "opt.html"
        result = eng.generate_report([s], output_path=str(out))
        assert Path(result).exists()
        assert "Options Strategy" in out.read_text()
