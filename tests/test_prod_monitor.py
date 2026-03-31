"""Tests for compass/prod_monitor.py — production monitoring system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from compass.prod_monitor import (
    Alert,
    AlertEngine,
    AlertThreshold,
    FillMetrics,
    GreeksExposure,
    ModelMetrics,
    MonitorConfig,
    MonitorSnapshot,
    PnLState,
    ProductionMonitor,
    Severity,
    SystemMetrics,
    compute_health_score,
    load_state,
    save_state,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    return MonitorConfig()


@pytest.fixture
def monitor(config):
    return ProductionMonitor(config, initial_capital=100_000)


# ── Severity tests ───────────────────────────────────────────────────────


class TestSeverity:
    def test_rank_order(self):
        assert Severity.rank("INFO") < Severity.rank("WARNING") < Severity.rank("CRITICAL")

    def test_unknown(self):
        assert Severity.rank("UNKNOWN") == -1


# ── GreeksExposure tests ─────────────────────────────────────────────────


class TestGreeks:
    def test_default(self):
        g = GreeksExposure()
        assert g.delta == 0.0 and g.vega == 0.0

    def test_to_dict(self):
        g = GreeksExposure(delta=100, gamma=5, vega=200, theta=-50)
        d = g.to_dict()
        assert d["delta"] == 100
        assert "theta" in d


# ── Alert tests ──────────────────────────────────────────────────────────


class TestAlert:
    def test_to_dict(self):
        a = Alert("2024-01-01 10:00:00", "WARNING", "vix", "high", 30.0, 25.0)
        d = a.to_dict()
        assert d["severity"] == "WARNING"
        assert d["metric"] == "vix"

    def test_telegram_format(self):
        a = Alert("2024-01-01 10:00:00", "CRITICAL", "drawdown", "bad", 0.15, 0.10)
        msg = a.telegram_format()
        assert "CRITICAL" in msg
        assert "drawdown" in msg
        assert "🚨" in msg

    def test_telegram_info(self):
        a = Alert("2024-01-01", "INFO", "test", "ok", 1.0, 2.0)
        assert "ℹ️" in a.telegram_format()

    def test_telegram_warning(self):
        a = Alert("2024-01-01", "WARNING", "test", "warn", 1.0, 2.0)
        assert "⚠️" in a.telegram_format()


# ── AlertEngine tests ────────────────────────────────────────────────────


class TestAlertEngine:
    def test_no_alert_normal(self):
        engine = AlertEngine([AlertThreshold("pnl", -3000, -5000, "below")])
        assert engine.check("pnl", 100.0) is None

    def test_warning_triggered(self):
        engine = AlertEngine([AlertThreshold("pnl", -3000, -5000, "below", cooldown_seconds=0)])
        alert = engine.check("pnl", -3500)
        assert alert is not None
        assert alert.severity == "WARNING"

    def test_critical_triggered(self):
        engine = AlertEngine([AlertThreshold("pnl", -3000, -5000, "below", cooldown_seconds=0)])
        alert = engine.check("pnl", -6000)
        assert alert is not None
        assert alert.severity == "CRITICAL"

    def test_above_direction(self):
        engine = AlertEngine([AlertThreshold("vix", 25, 35, "above", cooldown_seconds=0)])
        assert engine.check("vix", 30) is not None
        engine.reset_cooldowns()
        assert engine.check("vix", 20) is None

    def test_cooldown_prevents_spam(self):
        engine = AlertEngine([AlertThreshold("x", 10, 20, "above", cooldown_seconds=9999)])
        a1 = engine.check("x", 15)
        a2 = engine.check("x", 15)
        assert a1 is not None
        assert a2 is None  # cooldown active

    def test_reset_cooldowns(self):
        engine = AlertEngine([AlertThreshold("x", 10, 20, "above", cooldown_seconds=9999)])
        engine.check("x", 15)
        engine.reset_cooldowns()
        assert engine.check("x", 15) is not None

    def test_check_all(self):
        engine = AlertEngine([
            AlertThreshold("a", 10, 20, "above", cooldown_seconds=0),
            AlertThreshold("b", 5, 3, "below", cooldown_seconds=0),
        ])
        alerts = engine.check_all({"a": 15, "b": 2})
        assert len(alerts) == 2

    def test_unknown_metric_no_alert(self):
        engine = AlertEngine([AlertThreshold("x", 10, 20, "above")])
        assert engine.check("y", 100) is None


# ── Health score tests ───────────────────────────────────────────────────


class TestHealthScore:
    def test_perfect_health(self):
        score = compute_health_score(
            PnLState(), GreeksExposure(), 0.0, FillMetrics(fill_rate=1.0),
            ModelMetrics(model_accuracy=0.7, n_predictions=10), SystemMetrics(),
            MonitorConfig(),
        )
        assert score > 80

    def test_bad_drawdown_lowers_score(self):
        good = compute_health_score(
            PnLState(drawdown_pct=0.01), GreeksExposure(), 0.0, FillMetrics(fill_rate=1.0),
            ModelMetrics(model_accuracy=0.7, n_predictions=10), SystemMetrics(),
            MonitorConfig(),
        )
        bad = compute_health_score(
            PnLState(drawdown_pct=0.10), GreeksExposure(), 0.0, FillMetrics(fill_rate=1.0),
            ModelMetrics(model_accuracy=0.7, n_predictions=10), SystemMetrics(),
            MonitorConfig(),
        )
        assert good > bad

    def test_high_latency_lowers_score(self):
        good = compute_health_score(
            PnLState(), GreeksExposure(), 0.0, FillMetrics(fill_rate=1.0),
            ModelMetrics(model_accuracy=0.7, n_predictions=10), SystemMetrics(latency_ms=10),
            MonitorConfig(),
        )
        bad = compute_health_score(
            PnLState(), GreeksExposure(), 0.0, FillMetrics(fill_rate=1.0),
            ModelMetrics(model_accuracy=0.7, n_predictions=10), SystemMetrics(latency_ms=400),
            MonitorConfig(),
        )
        assert good > bad

    def test_bounded(self):
        score = compute_health_score(
            PnLState(drawdown_pct=1.0), GreeksExposure(delta=9999), 1.0,
            FillMetrics(fill_rate=0.0), ModelMetrics(), SystemMetrics(latency_ms=9999),
            MonitorConfig(),
        )
        assert 0 <= score <= 100


# ── Recording method tests ───────────────────────────────────────────────


class TestRecording:
    def test_record_trade(self, monitor):
        monitor.record_trade(500.0, "EXP-400")
        snap = monitor.snapshot()
        assert snap.pnl.daily_pnl == 500.0
        assert snap.pnl.per_experiment["EXP-400"] == 500.0

    def test_record_multiple_trades(self, monitor):
        monitor.record_trade(300.0, "A")
        monitor.record_trade(-100.0, "B")
        snap = monitor.snapshot()
        assert snap.pnl.daily_pnl == 200.0
        assert snap.pnl.n_trades_today == 2

    def test_drawdown_tracking(self, monitor):
        monitor.record_trade(1000.0)
        monitor.record_trade(-500.0)
        snap = monitor.snapshot()
        assert snap.pnl.drawdown_pct > 0

    def test_record_greeks(self, monitor):
        monitor.record_greeks(100, 5, 200, -50)
        snap = monitor.snapshot()
        assert snap.greeks.delta == 100
        assert snap.greeks.vega == 200

    def test_record_margin(self, monitor):
        monitor.record_margin(0.75)
        snap = monitor.snapshot()
        assert snap.margin_utilization == 0.75

    def test_record_fill(self, monitor):
        monitor.record_fill(True, 3.0)
        monitor.record_fill(False, 0.0)
        snap = monitor.snapshot()
        assert snap.fills.total_fills == 1
        assert snap.fills.total_orders == 2
        assert snap.fills.fill_rate == 0.5

    def test_record_signal(self, monitor):
        monitor.record_signal(15.0)
        snap = monitor.snapshot()
        assert snap.model.signal_age_minutes == 15.0

    def test_record_prediction(self, monitor):
        monitor.record_prediction(1, 1)
        monitor.record_prediction(1, 0)
        snap = monitor.snapshot()
        assert snap.model.model_accuracy == 0.5
        assert snap.model.n_predictions == 2

    def test_record_latency(self, monitor):
        monitor.record_latency(42.5)
        snap = monitor.snapshot()
        assert snap.system.latency_ms == 42.5

    def test_record_error(self, monitor):
        monitor.record_error()
        monitor.record_error()
        snap = monitor.snapshot()
        assert snap.system.errors_last_hour == 2

    def test_set_active_positions(self, monitor):
        monitor.set_active_positions(7)
        snap = monitor.snapshot()
        assert snap.n_active_positions == 7

    def test_reset_daily(self, monitor):
        monitor.record_trade(500.0)
        monitor.record_fill(True, 5.0)
        monitor.record_error()
        monitor.reset_daily()
        snap = monitor.snapshot()
        assert snap.pnl.daily_pnl == 0.0
        assert snap.fills.total_fills == 0
        assert snap.system.errors_last_hour == 0
        # Cumulative should persist
        assert snap.pnl.cumulative_pnl == 500.0


# ── Alert integration tests ──────────────────────────────────────────────


class TestAlertIntegration:
    def test_trade_loss_triggers_alert(self):
        thresholds = [AlertThreshold("daily_pnl", -3000, -5000, "below", cooldown_seconds=0)]
        mon = ProductionMonitor(thresholds=thresholds)
        alerts = mon.record_trade(-6000.0)
        assert len(alerts) > 0
        assert alerts[0].severity == "CRITICAL"

    def test_high_margin_triggers(self):
        thresholds = [AlertThreshold("margin_util", 0.6, 0.8, "above", cooldown_seconds=0)]
        mon = ProductionMonitor(thresholds=thresholds)
        alerts = mon.record_margin(0.85)
        assert any(a.metric == "margin_util" for a in alerts)

    def test_alert_history_accumulates(self):
        thresholds = [AlertThreshold("daily_pnl", -100, -200, "below", cooldown_seconds=0)]
        mon = ProductionMonitor(thresholds=thresholds)
        mon.record_trade(-150)
        mon.alert_engine.reset_cooldowns()
        mon.record_trade(-100)
        snap = mon.snapshot()
        assert len(snap.alert_history) >= 1


# ── State persistence tests ──────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self, monitor):
        monitor.record_trade(500.0, "EXP-400")
        snap = monitor.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            save_state(snap, path)
            assert path.exists()
            data = load_state(path)
            assert data is not None
            assert data["pnl"]["daily"] == 500.0

    def test_load_nonexistent(self):
        assert load_state(Path("/nonexistent/path.json")) is None

    def test_save_creates_dirs(self, monitor):
        snap = monitor.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "state.json"
            save_state(snap, path)
            assert path.exists()

    def test_json_valid(self, monitor):
        monitor.record_trade(100.0)
        snap = monitor.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            save_state(snap, path)
            data = json.loads(path.read_text())
            assert "health_score" in data


# ── Snapshot tests ───────────────────────────────────────────────────────


class TestSnapshot:
    def test_timestamp(self, monitor):
        snap = monitor.snapshot()
        assert len(snap.timestamp) > 0

    def test_health_score_present(self, monitor):
        snap = monitor.snapshot()
        assert 0 <= snap.health_score <= 100

    def test_uptime_positive(self, monitor):
        snap = monitor.snapshot()
        assert snap.system.uptime_seconds >= 0


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, monitor):
        monitor.record_trade(500.0, "EXP-400")
        monitor.record_greeks(50, 3, 100, -20)
        snap = monitor.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "mon.html"
            path = ProductionMonitor.generate_report(snap, out)
            assert path.exists()
            content = path.read_text()
            assert "Production Monitor" in content

    def test_contains_metrics(self, monitor):
        monitor.record_trade(500.0)
        monitor.record_margin(0.5)
        snap = monitor.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            ProductionMonitor.generate_report(snap, out)
            content = out.read_text()
            assert "Daily P&amp;L" in content or "Daily P&L" in content
            assert "Margin" in content

    def test_contains_health(self, monitor):
        snap = monitor.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            ProductionMonitor.generate_report(snap, out)
            content = out.read_text()
            assert "Health" in content

    def test_contains_experiment_table(self, monitor):
        monitor.record_trade(300.0, "EXP-A")
        monitor.record_trade(-100.0, "EXP-B")
        snap = monitor.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            ProductionMonitor.generate_report(snap, out)
            content = out.read_text()
            assert "EXP-A" in content
            assert "EXP-B" in content

    def test_default_path(self, monitor):
        snap = monitor.snapshot()
        path = ProductionMonitor.generate_report(snap)
        assert path.exists()
        assert "prod_monitor.html" in str(path)
        path.unlink(missing_ok=True)
