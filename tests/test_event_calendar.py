"""Tests for compass/event_calendar.py — event calendar engine."""

from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest
from compass.event_calendar import (
    Event, EventCalendarEngine, EventCluster, EventPnL, EventTypeStats,
    PostEventSignal, PreEventRule, UpcomingEvent,
    generate_cpi_dates, generate_nfp_dates, generate_opex_dates,
    generate_quad_witching, generate_vix_expiry, _third_friday,
)

# ── Helpers ──────────────────────────────────────────────────────────────

def _make_trades(n=200, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2025-01-02", periods=n)
    return pd.DataFrame({
        "entry_date": dates,
        "pnl": rng.normal(30, 150, n),
        "iv_change": rng.normal(0, 0.05, n),
        "regime": rng.choice(["bull", "bear", "neutral"], n),
    })

def _make_events():
    return [
        Event(date(2025, 3, 19), "fomc", "FOMC Mar 2025"),
        Event(date(2025, 3, 21), "opex", "OpEx Mar 2025"),
        Event(date(2025, 4, 10), "cpi", "CPI Apr 2025"),
        Event(date(2025, 5, 2), "nfp", "NFP May 2025"),
        Event(date(2025, 6, 20), "quad_witching", "Quad Jun 2025"),
    ]

def _make_engine(with_trades=True, ref=None, **kwargs):
    trades = _make_trades() if with_trades else None
    events = _make_events()
    return EventCalendarEngine(
        trades=trades, events=events,
        reference_date=ref or date(2025, 3, 10),
        **kwargs,
    )

# ── Date generator tests ─────────────────────────────────────────────────

class TestDateGenerators:
    def test_third_friday(self):
        f = _third_friday(2025, 1)
        assert f.weekday() == 4  # Friday
        assert 15 <= f.day <= 21

    def test_opex_12_dates(self):
        dates = generate_opex_dates(2025)
        assert len(dates) == 12
        for d in dates:
            assert d.weekday() == 4

    def test_quad_witching_4_dates(self):
        dates = generate_quad_witching(2025)
        assert len(dates) == 4
        months = {d.month for d in dates}
        assert months == {3, 6, 9, 12}

    def test_vix_expiry_12_dates(self):
        dates = generate_vix_expiry(2025)
        assert len(dates) == 12
        for d in dates:
            assert d.weekday() == 2  # Wednesday

    def test_nfp_first_friday(self):
        dates = generate_nfp_dates(2025)
        assert len(dates) == 12
        for d in dates:
            assert d.weekday() == 4
            assert d.day <= 7

    def test_cpi_weekday(self):
        dates = generate_cpi_dates(2025)
        assert len(dates) == 12
        for d in dates:
            assert d.weekday() < 5

# ── Dataclass tests ──────────────────────────────────────────────────────

class TestDataclasses:
    def test_event(self):
        e = Event(date(2025, 3, 19), "fomc", "FOMC")
        assert e.event_type == "fomc"

    def test_event_pnl(self):
        e = Event(date(2025, 3, 19), "fomc", "FOMC")
        ep = EventPnL(e, 50, 30, 80, 0.05, 5, 3, "win")
        assert ep.total_pnl == pytest.approx(80)

    def test_pre_event_rule(self):
        r = PreEventRule("fomc", 5, 0.6, 0.08, True, "desc")
        assert r.gamma_scalp_window is True

    def test_event_cluster(self):
        e1 = Event(date(2025, 3, 19), "fomc", "A")
        e2 = Event(date(2025, 3, 21), "opex", "B")
        cl = EventCluster(date(2025, 3, 19), date(2025, 3, 21), [e1, e2], 2, "medium", 0.6)
        assert cl.n_events == 2

    def test_upcoming(self):
        e = Event(date(2025, 3, 19), "fomc", "FOMC")
        r = PreEventRule("fomc", 5, 0.6, 0.08, True, "desc")
        u = UpcomingEvent(e, 9, r, "low")
        assert u.days_until == 9

    def test_post_event_signal(self):
        e = Event(date(2025, 3, 19), "fomc", "FOMC")
        ps = PostEventSignal(e, 2.0, 0.6, -0.8, "fade", 0.6)
        assert ps.signal == "fade"

# ── Pre-event rules ──────────────────────────────────────────────────────

class TestPreEventRules:
    def test_rules_for_all_types(self):
        engine = _make_engine()
        engine.analyze()
        for etype in ("fomc", "cpi", "nfp", "opex", "quad_witching", "earnings", "vix_expiry"):
            assert etype in engine.pre_event_rules

    def test_fomc_scaling_below_one(self):
        engine = _make_engine()
        engine.analyze()
        assert engine.pre_event_rules["fomc"].position_scaling < 1.0

    def test_quad_witching_most_conservative(self):
        engine = _make_engine()
        engine.analyze()
        qw = engine.pre_event_rules["quad_witching"].position_scaling
        fomc = engine.pre_event_rules["fomc"].position_scaling
        assert qw <= fomc

    def test_gamma_scalp_flags(self):
        engine = _make_engine()
        engine.analyze()
        assert engine.pre_event_rules["opex"].gamma_scalp_window is True
        assert engine.pre_event_rules["cpi"].gamma_scalp_window is False

# ── Event P&L ────────────────────────────────────────────────────────────

class TestEventPnL:
    def test_pnl_computed_with_trades(self):
        engine = _make_engine(with_trades=True)
        engine.analyze()
        assert len(engine.event_pnl) > 0

    def test_no_pnl_without_trades(self):
        engine = _make_engine(with_trades=False)
        engine.analyze()
        assert len(engine.event_pnl) == 0

    def test_outcome_field(self):
        engine = _make_engine()
        engine.analyze()
        for ep in engine.event_pnl:
            assert ep.outcome in ("win", "loss")

    def test_total_is_pre_plus_post(self):
        engine = _make_engine()
        engine.analyze()
        for ep in engine.event_pnl:
            assert ep.total_pnl == pytest.approx(ep.pre_pnl + ep.post_pnl, abs=0.01)

# ── Type stats ───────────────────────────────────────────────────────────

class TestTypeStats:
    def test_stats_computed(self):
        engine = _make_engine()
        engine.analyze()
        assert len(engine.type_stats) > 0

    def test_win_rate_range(self):
        engine = _make_engine()
        engine.analyze()
        for s in engine.type_stats:
            assert 0 <= s.win_rate <= 1

    def test_sorted_by_pnl(self):
        engine = _make_engine()
        engine.analyze()
        pnls = [s.avg_pnl for s in engine.type_stats]
        assert pnls == sorted(pnls, reverse=True)

# ── Clustering ───────────────────────────────────────────────────────────

class TestClustering:
    def test_detects_cluster(self):
        """FOMC Mar 19 + OpEx Mar 21 should cluster within 5d window."""
        engine = _make_engine(cluster_window=5)
        engine.analyze()
        march_clusters = [c for c in engine.clusters
                          if any(e.date.month == 3 and e.date.year == 2025 for e in c.events)]
        assert len(march_clusters) > 0

    def test_cluster_risk_levels(self):
        engine = _make_engine()
        engine.analyze()
        for c in engine.clusters:
            assert c.risk_level in ("low", "medium", "high")

    def test_sizing_adjustment_range(self):
        engine = _make_engine()
        engine.analyze()
        for c in engine.clusters:
            assert 0 < c.sizing_adjustment <= 1.0

    def test_no_single_event_clusters(self):
        engine = _make_engine()
        engine.analyze()
        for c in engine.clusters:
            assert c.n_events >= 2

# ── Post-event signals ───────────────────────────────────────────────────

class TestPostEvent:
    def test_signals_computed(self):
        engine = _make_engine()
        engine.analyze()
        # May have 0 if no event_pnl, but should be a list
        assert isinstance(engine.post_event_signals, list)

    def test_signal_values(self):
        engine = _make_engine()
        engine.analyze()
        for ps in engine.post_event_signals:
            assert ps.signal in ("fade", "follow", "neutral")

    def test_confidence_range(self):
        engine = _make_engine()
        engine.analyze()
        for ps in engine.post_event_signals:
            assert 0 <= ps.confidence <= 1

# ── Upcoming events ──────────────────────────────────────────────────────

class TestUpcoming:
    def test_upcoming_within_horizon(self):
        engine = _make_engine(ref=date(2025, 3, 10))
        engine.analyze()
        for u in engine.upcoming:
            assert 0 <= u.days_until <= 30

    def test_sorted_by_days(self):
        engine = _make_engine(ref=date(2025, 3, 10))
        engine.analyze()
        days = [u.days_until for u in engine.upcoming]
        assert days == sorted(days)

    def test_has_rule(self):
        engine = _make_engine(ref=date(2025, 3, 10))
        engine.analyze()
        for u in engine.upcoming:
            if u.event.event_type in ("fomc", "cpi", "nfp", "opex"):
                assert u.rule is not None

# ── Auto-generated calendar ──────────────────────────────────────────────

class TestAutoCalendar:
    def test_generates_events(self):
        engine = EventCalendarEngine(reference_date=date(2025, 6, 1))
        assert len(engine.events) > 50

    def test_multiple_types(self):
        engine = EventCalendarEngine(reference_date=date(2025, 6, 1))
        types = {e.event_type for e in engine.events}
        assert "fomc" in types
        assert "opex" in types
        assert "nfp" in types

    def test_sorted_by_date(self):
        engine = EventCalendarEngine(reference_date=date(2025, 6, 1))
        dates = [e.date for e in engine.events]
        assert dates == sorted(dates)

# ── Pipeline ─────────────────────────────────────────────────────────────

class TestPipeline:
    def test_analyze_keys(self):
        engine = _make_engine()
        result = engine.analyze()
        expected = {"event_pnl", "pre_event_rules", "post_event_signals",
                    "clusters", "type_stats", "upcoming"}
        assert set(result.keys()) == expected

    def test_from_csv(self, tmp_path):
        df = _make_trades()
        csv = tmp_path / "trades.csv"
        df.to_csv(csv, index=False)
        engine = EventCalendarEngine.from_csv(str(csv), reference_date=date(2025, 3, 10))
        engine.analyze()
        assert len(engine.pre_event_rules) > 0

# ── Report ───────────────────────────────────────────────────────────────

class TestReport:
    def test_generates_html(self, tmp_path):
        engine = _make_engine()
        path = engine.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "Event Calendar" in c

    def test_report_sections(self, tmp_path):
        engine = _make_engine()
        path = engine.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Upcoming" in c and "Positioning Rules" in c
        assert "Historical" in c and "Cluster" in c

    def test_report_charts(self, tmp_path):
        engine = _make_engine()
        path = engine.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()

    def test_report_auto_analyzes(self, tmp_path):
        engine = _make_engine()
        assert not engine.pre_event_rules
        engine.generate_report(str(tmp_path / "r.html"))
        assert len(engine.pre_event_rules) > 0

    def test_report_default_path(self):
        engine = _make_engine()
        path = engine.generate_report()
        assert "event_calendar.html" in path
