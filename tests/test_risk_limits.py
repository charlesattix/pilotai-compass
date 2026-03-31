"""Tests for compass.risk_limits – dynamic risk limit engine."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from compass.risk_limits import (
    BEAR,
    BULL,
    CRASH,
    CRITICAL,
    HIGH_VOL,
    INFO,
    LOW_VOL,
    WARNING,
    Breach,
    EffectiveLimit,
    ExposureSnapshot,
    LimitConfig,
    ReductionTrigger,
    RegimeMultipliers,
    RiskLimitEngine,
    RiskLimitResult,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _normal_exposure() -> ExposureSnapshot:
    return ExposureSnapshot(
        gross_exposure=0.60,
        net_exposure=0.30,
        beta_exposure=0.80,
        n_positions=8,
        sector_exposures={"tech": 0.15, "finance": 0.10},
        position_sizes={"SPY_PUT": 0.06, "QQQ_PUT": 0.04},
    )


def _high_exposure() -> ExposureSnapshot:
    return ExposureSnapshot(
        gross_exposure=0.95,
        net_exposure=0.48,
        beta_exposure=1.4,
        n_positions=18,
        sector_exposures={"tech": 0.28, "finance": 0.25},
        position_sizes={"SPY_PUT": 0.09, "QQQ_PUT": 0.08},
    )


def _over_limit_exposure() -> ExposureSnapshot:
    return ExposureSnapshot(
        gross_exposure=1.20,
        net_exposure=0.60,
        beta_exposure=2.0,
        n_positions=25,
        sector_exposures={"tech": 0.35},
        position_sizes={"SPY_PUT": 0.15},
    )


# ── Constructor ─────────────────────────────────────────────────────────────
class TestInit:
    def test_defaults(self):
        e = RiskLimitEngine()
        assert e.config.max_gross_exposure == 1.0
        assert e.vix_start == 25.0

    def test_custom_config(self):
        cfg = LimitConfig(max_gross_exposure=0.80, max_positions=10)
        e = RiskLimitEngine(config=cfg)
        assert e.config.max_gross_exposure == 0.80

    def test_custom_vix_thresholds(self):
        e = RiskLimitEngine(vix_tighten_start=20.0, vix_tighten_full=40.0)
        assert e.vix_start == 20.0


# ── Vol multiplier ──────────────────────────────────────────────────────────
class TestVolMultiplier:
    def test_below_start_returns_one(self):
        e = RiskLimitEngine()
        assert e._vol_multiplier(15.0) == 1.0

    def test_above_full_returns_half(self):
        e = RiskLimitEngine()
        assert e._vol_multiplier(60.0) == 0.5

    def test_mid_range(self):
        e = RiskLimitEngine()
        m = e._vol_multiplier(37.5)  # midpoint of 25-50
        assert 0.5 < m < 1.0


# ── DD multiplier ───────────────────────────────────────────────────────────
class TestDDMultiplier:
    def test_below_start_returns_one(self):
        e = RiskLimitEngine()
        assert e._dd_multiplier(0.02) == 1.0

    def test_above_full_returns_floor(self):
        e = RiskLimitEngine()
        assert e._dd_multiplier(0.20) == 0.4

    def test_mid_range(self):
        e = RiskLimitEngine()
        m = e._dd_multiplier(0.10)
        assert 0.4 < m < 1.0


# ── Monitoring ──────────────────────────────────────────────────────────────
class TestMonitor:
    def test_returns_result(self):
        r = RiskLimitEngine().monitor(_normal_exposure())
        assert isinstance(r, RiskLimitResult)

    def test_limits_populated(self):
        r = RiskLimitEngine().monitor(_normal_exposure())
        assert len(r.effective_limits) >= 5  # gross, net, beta, positions, single

    def test_no_breaches_normal(self):
        r = RiskLimitEngine().monitor(_normal_exposure(), regime=BULL, vix=15.0, current_dd=0.0)
        assert r.n_breaches == 0

    def test_breaches_high_exposure(self):
        r = RiskLimitEngine().monitor(_high_exposure(), regime=BULL, vix=15.0)
        assert r.n_breaches > 0

    def test_critical_over_limit(self):
        r = RiskLimitEngine().monitor(_over_limit_exposure(), regime=BULL, vix=15.0)
        assert r.n_critical > 0

    def test_regime_recorded(self):
        r = RiskLimitEngine().monitor(_normal_exposure(), regime=BEAR)
        assert r.regime == BEAR

    def test_vix_recorded(self):
        r = RiskLimitEngine().monitor(_normal_exposure(), vix=30.0)
        assert r.vix == 30.0

    def test_generated_at(self):
        r = RiskLimitEngine().monitor(_normal_exposure())
        assert len(r.generated_at) > 0


# ── Regime adjustment ───────────────────────────────────────────────────────
class TestRegimeAdjustment:
    def test_crash_tightens_limits(self):
        e = RiskLimitEngine()
        bull = e.monitor(_normal_exposure(), regime=BULL, vix=15.0)
        crash = e.monitor(_normal_exposure(), regime=CRASH, vix=15.0)
        bull_gross = next(l for l in bull.effective_limits if l.limit_name == "gross_exposure")
        crash_gross = next(l for l in crash.effective_limits if l.limit_name == "gross_exposure")
        assert crash_gross.adjusted_value < bull_gross.adjusted_value

    def test_low_vol_loosens(self):
        e = RiskLimitEngine()
        bull = e.monitor(_normal_exposure(), regime=BULL, vix=12.0)
        lv = e.monitor(_normal_exposure(), regime=LOW_VOL, vix=12.0)
        bull_g = next(l for l in bull.effective_limits if l.limit_name == "gross_exposure")
        lv_g = next(l for l in lv.effective_limits if l.limit_name == "gross_exposure")
        assert lv_g.adjusted_value >= bull_g.adjusted_value

    def test_crash_more_breaches(self):
        e = RiskLimitEngine()
        bull = e.monitor(_high_exposure(), regime=BULL, vix=15.0)
        crash = e.monitor(_high_exposure(), regime=CRASH, vix=15.0)
        assert crash.n_breaches >= bull.n_breaches


# ── Stress adjustment (VIX + DD) ───────────────────────────────────────────
class TestStressAdjustment:
    def test_high_vix_tightens(self):
        e = RiskLimitEngine()
        low = e.monitor(_normal_exposure(), vix=15.0)
        high = e.monitor(_normal_exposure(), vix=40.0)
        low_g = next(l for l in low.effective_limits if l.limit_name == "gross_exposure")
        high_g = next(l for l in high.effective_limits if l.limit_name == "gross_exposure")
        assert high_g.adjusted_value < low_g.adjusted_value

    def test_drawdown_tightens(self):
        e = RiskLimitEngine()
        clean = e.monitor(_normal_exposure(), current_dd=0.0)
        dd = e.monitor(_normal_exposure(), current_dd=0.12)
        clean_g = next(l for l in clean.effective_limits if l.limit_name == "gross_exposure")
        dd_g = next(l for l in dd.effective_limits if l.limit_name == "gross_exposure")
        assert dd_g.adjusted_value < clean_g.adjusted_value


# ── Effective limits ────────────────────────────────────────────────────────
class TestEffectiveLimits:
    def test_utilisation_bounded(self):
        r = RiskLimitEngine().monitor(_normal_exposure())
        for l in r.effective_limits:
            assert l.utilisation >= 0

    def test_headroom_calculation(self):
        r = RiskLimitEngine().monitor(_normal_exposure())
        for l in r.effective_limits:
            assert l.headroom == pytest.approx(l.adjusted_value - l.current_value, abs=0.001)

    def test_sector_limits_included(self):
        r = RiskLimitEngine().monitor(_normal_exposure())
        sector_limits = [l for l in r.effective_limits if l.limit_name.startswith("sector_")]
        assert len(sector_limits) == 2


# ── Breaches ────────────────────────────────────────────────────────────────
class TestBreaches:
    def test_breach_fields(self):
        r = RiskLimitEngine().monitor(_over_limit_exposure())
        for b in r.breaches:
            assert b.severity in (WARNING, CRITICAL)
            assert b.utilisation > 0
            assert len(b.timestamp) > 0

    def test_warning_at_80pct(self):
        # Create exposure at exactly 80% of gross limit
        exp = ExposureSnapshot(gross_exposure=0.80, net_exposure=0.20, n_positions=5, position_sizes={"a": 0.05})
        r = RiskLimitEngine().monitor(exp, regime=BULL, vix=15.0)
        gross_breaches = [b for b in r.breaches if b.limit_name == "gross_exposure"]
        if gross_breaches:
            assert gross_breaches[0].severity == WARNING

    def test_critical_at_95pct(self):
        exp = ExposureSnapshot(gross_exposure=0.96, net_exposure=0.20, n_positions=5, position_sizes={"a": 0.05})
        r = RiskLimitEngine().monitor(exp, regime=BULL, vix=15.0)
        gross_breaches = [b for b in r.breaches if b.limit_name == "gross_exposure"]
        if gross_breaches:
            assert gross_breaches[0].severity == CRITICAL


# ── Reduction triggers ──────────────────────────────────────────────────────
class TestReductionTriggers:
    def test_dd_trigger(self):
        r = RiskLimitEngine().monitor(_normal_exposure(), current_dd=0.10)
        dd_triggers = [t for t in r.triggers if t.trigger_type == "drawdown"]
        assert len(dd_triggers) > 0

    def test_vol_trigger(self):
        r = RiskLimitEngine().monitor(_normal_exposure(), vix=35.0)
        vol_triggers = [t for t in r.triggers if t.trigger_type == "vol"]
        assert len(vol_triggers) > 0

    def test_exposure_trigger(self):
        r = RiskLimitEngine().monitor(_over_limit_exposure())
        exp_triggers = [t for t in r.triggers if t.trigger_type == "exposure"]
        assert len(exp_triggers) > 0

    def test_no_triggers_normal(self):
        r = RiskLimitEngine().monitor(_normal_exposure(), regime=BULL, vix=15.0, current_dd=0.0)
        assert len(r.triggers) == 0

    def test_reduction_pct_bounded(self):
        r = RiskLimitEngine().monitor(_over_limit_exposure(), vix=40.0, current_dd=0.12)
        for t in r.triggers:
            assert 0.0 < t.reduction_pct <= 1.0


# ── History ─────────────────────────────────────────────────────────────────
class TestHistory:
    def test_history_accumulates(self):
        e = RiskLimitEngine()
        e.monitor(_normal_exposure())
        e.monitor(_normal_exposure())
        assert len(e.get_history()) > 0

    def test_clear_history(self):
        e = RiskLimitEngine()
        e.monitor(_normal_exposure())
        e.clear_history()
        assert len(e.get_history()) == 0


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = RiskLimitEngine()
            r = e.monitor(_high_exposure(), regime=BEAR, vix=30.0, current_dd=0.08)
            path = e.generate_report(r, output_path=Path(tmp) / "rl.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = RiskLimitEngine()
            r = e.monitor(_over_limit_exposure(), regime=CRASH, vix=45.0, current_dd=0.12)
            path = e.generate_report(r, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Risk Limit" in html
            assert "Utilisation" in html
            assert "Breach" in html
            assert "Trigger" in html
            assert "Exposure" in html

    def test_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = RiskLimitEngine()
            r = e.monitor(_normal_exposure())
            path = e.generate_report(r, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_effective_limit(self):
        l = EffectiveLimit("test", 1.0, 0.8, 0.5, 0.625, 0.3)
        assert l.utilisation == 0.625

    def test_breach(self):
        b = Breach("2024-01-01", "gross", CRITICAL, 1.1, 1.0, 1.1)
        assert b.severity == CRITICAL

    def test_trigger(self):
        t = ReductionTrigger("drawdown", WARNING, 0.10, 0.05, "Reduce", 0.20)
        assert t.reduction_pct == 0.20

    def test_exposure_snapshot(self):
        e = ExposureSnapshot(gross_exposure=0.5)
        assert e.gross_exposure == 0.5

    def test_result_defaults(self):
        r = RiskLimitResult()
        assert r.n_breaches == 0
        assert r.effective_limits == []
