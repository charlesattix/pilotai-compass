"""Tests for compass.portfolio_dashboard — 25+ tests covering all sections."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.portfolio_dashboard import (
    GREEN,
    YELLOW,
    RED,
    SEVERITY_ORDER,
    AlertEntry,
    DashboardResult,
    ExecutiveSummary,
    ExperimentCard,
    PortfolioDashboard,
    RegimePanel,
    RiskBudgetRow,
    SignalQualityRow,
    TradeLogEntry,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def dashboard():
    return PortfolioDashboard()


@pytest.fixture
def sample_experiment_metrics():
    return {
        "exp400": {
            "sharpe": 2.1,
            "return_pct": 18.5,
            "max_dd": -12.3,
            "win_rate": 72.0,
            "profit_factor": 2.0,
            "total_trades": 150,
            "capacity": 500_000,
        },
        "exp401": {
            "sharpe": 1.0,
            "return_pct": 8.2,
            "max_dd": -20.1,
            "win_rate": 55.0,
            "profit_factor": 1.2,
            "total_trades": 80,
            "capacity": 300_000,
        },
        "exp500": {
            "sharpe": 0.3,
            "return_pct": 2.0,
            "max_dd": -35.0,
            "win_rate": 45.0,
            "profit_factor": 0.8,
            "total_trades": 40,
            "capacity": 100_000,
        },
    }


@pytest.fixture
def sample_risk_budget():
    return {
        "exp400": {"var": 0.02, "cvar": 0.035, "contribution_pct": 45.0},
        "exp401": {"var": 0.03, "cvar": 0.05, "contribution_pct": 35.0},
        "exp500": {"var": 0.04, "cvar": 0.07, "contribution_pct": 20.0},
    }


@pytest.fixture
def sample_signal_quality():
    return {
        "exp400": {"ic": 0.08, "decay_hours": 24.0, "snr": 1.5},
        "exp401": {"ic": 0.04, "decay_hours": 12.0, "snr": 0.8},
    }


@pytest.fixture
def sample_recent_trades():
    return [
        {"date": "2026-03-25", "pnl": 120.50, "experiment": "exp400", "type": "bull_put"},
        {"date": "2026-03-24", "pnl": -45.00, "experiment": "exp401", "type": "bear_call"},
        {"date": "2026-03-23", "pnl": 200.00, "experiment": "exp400", "type": "bull_put"},
        {"date": "2026-03-22", "pnl": 0.0, "experiment": "exp500", "type": "iron_condor"},
    ]


@pytest.fixture
def sample_alerts():
    return [
        {"severity": "low", "message": "Routine check passed", "timestamp": "2026-03-25T10:00:00Z"},
        {"severity": "critical", "message": "VaR breach on exp500", "timestamp": "2026-03-25T09:00:00Z"},
        {"severity": "medium", "message": "Signal decay detected", "timestamp": "2026-03-25T08:00:00Z"},
        {"severity": "high", "message": "Regime transition imminent", "timestamp": "2026-03-25T07:00:00Z"},
    ]


@pytest.fixture
def sample_regime_info():
    return {"current": "low_vol", "forecast": "high_vol", "confidence": 0.82}


@pytest.fixture
def full_result(
    dashboard,
    sample_experiment_metrics,
    sample_risk_budget,
    sample_signal_quality,
    sample_recent_trades,
    sample_alerts,
    sample_regime_info,
):
    return dashboard.build(
        experiment_metrics=sample_experiment_metrics,
        risk_budget=sample_risk_budget,
        signal_quality=sample_signal_quality,
        recent_trades=sample_recent_trades,
        alerts=sample_alerts,
        regime_info=sample_regime_info,
    )


# ── Executive summary tests ────────────────────────────────────────────────

class TestExecutiveSummary:

    def test_portfolio_sharpe_is_mean(self, full_result):
        s = full_result.executive_summary
        expected = np.mean([2.1, 1.0, 0.3])
        assert abs(s.portfolio_sharpe - expected) < 1e-6

    def test_total_return_is_mean(self, full_result):
        s = full_result.executive_summary
        expected = np.mean([18.5, 8.2, 2.0])
        assert abs(s.total_return_pct - expected) < 1e-6

    def test_worst_drawdown(self, full_result):
        s = full_result.executive_summary
        assert s.worst_drawdown_pct == -35.0

    def test_estimated_capacity_sum(self, full_result):
        s = full_result.executive_summary
        assert s.estimated_capacity == 900_000

    def test_experiment_count(self, full_result):
        s = full_result.executive_summary
        assert s.n_experiments == 3

    def test_traffic_light_counts(self, full_result):
        s = full_result.executive_summary
        assert s.n_green == 1  # exp400 sharpe=2.1
        assert s.n_yellow == 1  # exp401 sharpe=1.0
        assert s.n_red == 1  # exp500 sharpe=0.3

    def test_overall_status_red_when_any_red(self, full_result):
        s = full_result.executive_summary
        assert s.overall_status == RED

    def test_overall_status_green_all_green(self, dashboard):
        metrics = {
            "a": {"sharpe": 2.0},
            "b": {"sharpe": 1.8},
        }
        result = dashboard.build(experiment_metrics=metrics)
        assert result.executive_summary.overall_status == GREEN

    def test_overall_status_yellow_no_red(self, dashboard):
        metrics = {
            "a": {"sharpe": 2.0},
            "b": {"sharpe": 1.0},  # YELLOW
        }
        result = dashboard.build(experiment_metrics=metrics)
        assert result.executive_summary.overall_status == YELLOW

    def test_empty_metrics(self, dashboard):
        result = dashboard.build()
        s = result.executive_summary
        assert s.portfolio_sharpe == 0.0
        assert s.n_experiments == 0


# ── Experiment cards tests ──────────────────────────────────────────────────

class TestExperimentCards:

    def test_card_count(self, full_result):
        assert len(full_result.experiment_cards) == 3

    def test_card_sorted_by_id(self, full_result):
        ids = [c.experiment_id for c in full_result.experiment_cards]
        assert ids == sorted(ids)

    def test_green_card(self, full_result):
        card = next(c for c in full_result.experiment_cards if c.experiment_id == "exp400")
        assert card.status == GREEN
        assert card.sharpe == 2.1
        assert card.return_pct == 18.5

    def test_yellow_card(self, full_result):
        card = next(c for c in full_result.experiment_cards if c.experiment_id == "exp401")
        assert card.status == YELLOW

    def test_red_card(self, full_result):
        card = next(c for c in full_result.experiment_cards if c.experiment_id == "exp500")
        assert card.status == RED
        assert card.win_rate == 45.0

    def test_custom_thresholds(self):
        dash = PortfolioDashboard(sharpe_green=3.0, sharpe_yellow=2.0)
        result = dash.build(experiment_metrics={"x": {"sharpe": 2.5}})
        card = result.experiment_cards[0]
        assert card.status == YELLOW


# ── Regime panel tests ──────────────────────────────────────────────────────

class TestRegimePanel:

    def test_regime_current(self, full_result):
        assert full_result.regime_panel.current == "low_vol"

    def test_regime_forecast(self, full_result):
        assert full_result.regime_panel.forecast == "high_vol"

    def test_regime_confidence(self, full_result):
        assert abs(full_result.regime_panel.confidence - 0.82) < 1e-6

    def test_regime_defaults(self, dashboard):
        result = dashboard.build()
        rp = result.regime_panel
        assert rp.current == "unknown"
        assert rp.forecast == "unknown"
        assert rp.confidence == 0.0


# ── Risk budget tests ──────────────────────────────────────────────────────

class TestRiskBudget:

    def test_risk_budget_count(self, full_result):
        assert len(full_result.risk_budget) == 3

    def test_risk_budget_values(self, full_result):
        row = next(r for r in full_result.risk_budget if r.experiment_id == "exp400")
        assert row.var == 0.02
        assert row.cvar == 0.035
        assert row.contribution_pct == 45.0

    def test_risk_budget_empty(self, dashboard):
        result = dashboard.build()
        assert result.risk_budget == []


# ── Signal quality tests ────────────────────────────────────────────────────

class TestSignalQuality:

    def test_signal_quality_count(self, full_result):
        assert len(full_result.signal_quality) == 2

    def test_signal_quality_values(self, full_result):
        row = next(r for r in full_result.signal_quality if r.experiment_id == "exp400")
        assert row.ic == 0.08
        assert row.decay_hours == 24.0
        assert row.snr == 1.5

    def test_signal_quality_empty(self, dashboard):
        result = dashboard.build()
        assert result.signal_quality == []


# ── Trade log tests ─────────────────────────────────────────────────────────

class TestTradeLog:

    def test_trade_log_count(self, full_result):
        assert len(full_result.trade_log) == 4

    def test_trade_log_fields(self, full_result):
        t = full_result.trade_log[0]
        assert t.date == "2026-03-25"
        assert t.pnl == 120.50
        assert t.experiment == "exp400"
        assert t.trade_type == "bull_put"

    def test_trade_log_truncated_to_max(self, dashboard):
        trades = [{"date": f"2026-01-{i:02d}", "pnl": i * 10.0, "experiment": "x", "type": "t"}
                  for i in range(1, 31)]
        result = dashboard.build(recent_trades=trades)
        assert len(result.trade_log) == 20  # default max

    def test_trade_log_custom_max(self):
        dash = PortfolioDashboard(max_recent_trades=5)
        trades = [{"date": "2026-01-01", "pnl": 10.0, "experiment": "x", "type": "t"}
                  for _ in range(10)]
        result = dash.build(recent_trades=trades)
        assert len(result.trade_log) == 5

    def test_trade_log_empty(self, dashboard):
        result = dashboard.build()
        assert result.trade_log == []


# ── Alerts tests ────────────────────────────────────────────────────────────

class TestAlerts:

    def test_alerts_sorted_by_severity(self, full_result):
        severities = [a.severity for a in full_result.alerts]
        assert severities == ["critical", "high", "medium", "low"]

    def test_alert_fields(self, full_result):
        critical = full_result.alerts[0]
        assert critical.severity == "critical"
        assert "VaR breach" in critical.message
        assert critical.timestamp == "2026-03-25T09:00:00Z"

    def test_alerts_empty(self, dashboard):
        result = dashboard.build()
        assert result.alerts == []

    def test_alerts_case_insensitive(self, dashboard):
        alerts = [{"severity": "HIGH", "message": "test", "timestamp": "now"}]
        result = dashboard.build(alerts=alerts)
        assert result.alerts[0].severity == "high"


# ── HTML report tests ───────────────────────────────────────────────────────

class TestHTMLReport:

    def test_generate_report_creates_file(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            result_path = dashboard.generate_report(full_result, path)
            assert result_path.exists()
            assert result_path.stat().st_size > 0

    def test_html_contains_title(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            dashboard.generate_report(full_result, path)
            html = path.read_text()
            assert "<title>Portfolio Dashboard</title>" in html

    def test_html_contains_executive_summary(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            dashboard.generate_report(full_result, path)
            html = path.read_text()
            assert "Portfolio Sharpe" in html
            assert "Worst Drawdown" in html

    def test_html_contains_experiment_cards(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            dashboard.generate_report(full_result, path)
            html = path.read_text()
            assert "exp400" in html
            assert "exp-card" in html

    def test_html_contains_regime_panel(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            dashboard.generate_report(full_result, path)
            html = path.read_text()
            assert "low_vol" in html
            assert "high_vol" in html

    def test_html_contains_svg_charts(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            dashboard.generate_report(full_result, path)
            html = path.read_text()
            assert "<svg" in html
            assert "viewBox" in html

    def test_html_contains_alerts(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            dashboard.generate_report(full_result, path)
            html = path.read_text()
            assert "VaR breach" in html
            assert "alert-critical" in html

    def test_html_contains_risk_table(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            dashboard.generate_report(full_result, path)
            html = path.read_text()
            assert "Risk Budget" in html
            assert "CVaR" in html

    def test_html_contains_trade_log(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            dashboard.generate_report(full_result, path)
            html = path.read_text()
            assert "Recent Trades" in html

    def test_html_dark_theme(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            dashboard.generate_report(full_result, path)
            html = path.read_text()
            assert "#0f172a" in html  # dark background

    def test_html_self_contained_no_external_deps(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash.html"
            dashboard.generate_report(full_result, path)
            html = path.read_text()
            # No CDN links or script src
            assert "cdn" not in html.lower()
            assert '<script src=' not in html

    def test_html_creates_parent_dirs(self, dashboard, full_result):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub" / "dir" / "dash.html"
            dashboard.generate_report(full_result, path)
            assert path.exists()


# ── Dataclass tests ─────────────────────────────────────────────────────────

class TestDataclasses:

    def test_dashboard_result_defaults(self):
        r = DashboardResult()
        assert r.executive_summary is None
        assert r.experiment_cards == []
        assert r.regime_panel is None
        assert r.risk_budget == []
        assert r.signal_quality == []
        assert r.trade_log == []
        assert r.alerts == []
        assert r.generated_at == ""

    def test_executive_summary_defaults(self):
        s = ExecutiveSummary()
        assert s.portfolio_sharpe == 0.0
        assert s.overall_status == GREEN

    def test_experiment_card_fields(self):
        c = ExperimentCard(experiment_id="x", status=RED, sharpe=-0.5)
        assert c.experiment_id == "x"
        assert c.status == RED
        assert c.sharpe == -0.5

    def test_regime_panel_defaults(self):
        rp = RegimePanel()
        assert rp.current == "unknown"

    def test_risk_budget_row_fields(self):
        r = RiskBudgetRow(experiment_id="a", var=0.01, cvar=0.02, contribution_pct=50.0)
        assert r.contribution_pct == 50.0

    def test_signal_quality_row_fields(self):
        sq = SignalQualityRow(experiment_id="b", ic=0.1, decay_hours=48.0, snr=2.0)
        assert sq.snr == 2.0

    def test_trade_log_entry_fields(self):
        t = TradeLogEntry(date="2026-01-01", pnl=-50.0, experiment="exp", trade_type="put")
        assert t.pnl == -50.0

    def test_alert_entry_defaults(self):
        a = AlertEntry()
        assert a.severity == "info"
        assert a.message == ""


# ── Edge case tests ─────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_experiment(self, dashboard):
        result = dashboard.build(experiment_metrics={"solo": {"sharpe": 1.8, "return_pct": 10.0}})
        assert result.executive_summary.n_experiments == 1
        assert len(result.experiment_cards) == 1

    def test_missing_metric_fields(self, dashboard):
        result = dashboard.build(experiment_metrics={"x": {}})
        card = result.experiment_cards[0]
        assert card.sharpe == 0.0
        assert card.return_pct == 0.0
        assert card.status == RED  # sharpe 0 < 0.8

    def test_negative_sharpe(self, dashboard):
        result = dashboard.build(experiment_metrics={"neg": {"sharpe": -1.5}})
        assert result.experiment_cards[0].status == RED

    def test_zero_pnl_trade(self, dashboard):
        trades = [{"date": "2026-01-01", "pnl": 0.0, "experiment": "x", "type": "flat"}]
        result = dashboard.build(recent_trades=trades)
        assert result.trade_log[0].pnl == 0.0

    def test_generated_at_populated(self, dashboard):
        result = dashboard.build()
        assert "UTC" in result.generated_at

    def test_html_empty_dashboard(self, dashboard):
        """Generate HTML even with no data at all."""
        result = dashboard.build()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.html"
            dashboard.generate_report(result, path)
            html = path.read_text()
            assert "Portfolio Dashboard" in html

    def test_severity_order_constant(self):
        assert SEVERITY_ORDER["critical"] < SEVERITY_ORDER["high"]
        assert SEVERITY_ORDER["high"] < SEVERITY_ORDER["medium"]
        assert SEVERITY_ORDER["medium"] < SEVERITY_ORDER["low"]
        assert SEVERITY_ORDER["low"] < SEVERITY_ORDER["info"]

    def test_many_alerts_sorted(self, dashboard):
        alerts = [
            {"severity": "info", "message": "a", "timestamp": "t1"},
            {"severity": "critical", "message": "b", "timestamp": "t2"},
            {"severity": "low", "message": "c", "timestamp": "t3"},
            {"severity": "high", "message": "d", "timestamp": "t4"},
            {"severity": "medium", "message": "e", "timestamp": "t5"},
        ]
        result = dashboard.build(alerts=alerts)
        sevs = [a.severity for a in result.alerts]
        assert sevs == ["critical", "high", "medium", "low", "info"]

    def test_risk_budget_contribution_preserved(self, dashboard):
        rb = {"x": {"var": 0.01, "cvar": 0.02, "contribution_pct": 100.0}}
        result = dashboard.build(risk_budget=rb)
        assert result.risk_budget[0].contribution_pct == 100.0

    def test_regime_confidence_gt_one_treated_as_pct(self, dashboard):
        """If confidence > 1 it is already a percentage."""
        result = dashboard.build(regime_info={"current": "x", "forecast": "y", "confidence": 82})
        rp = result.regime_panel
        assert rp.confidence == 82.0
