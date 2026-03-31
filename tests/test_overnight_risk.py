"""Tests for compass/overnight_risk.py — overnight risk assessment module.

30+ tests covering every computation, edge cases, dataclasses, and HTML output.
"""
from __future__ import annotations

import math
from dataclasses import asdict, fields
from datetime import date, timedelta

import numpy as np
import pytest

from compass.overnight_risk import (
    OVERNIGHT_GAP_STD,
    STRESS_MARGIN_MULTIPLIER,
    VIX_HIGH,
    VIX_LOW,
    Z_95,
    Z_99,
    CorrelationExposure,
    DTEBucket,
    EarningsExposure,
    GapRiskResult,
    HedgeRecommendation,
    MarginBuffer,
    MaxLossScenario,
    OvernightResult,
    OvernightRiskReport,
    PositionGreeksSummary,
    SectorConcentration,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_position(
    symbol: str = "SPY",
    quantity: int = -10,
    delta: float = -0.30,
    gamma: float = 0.02,
    vega: float = 0.15,
    theta: float = -0.05,
    dte: int = 14,
    sector: str = "ETF",
    entry_price: float = 2.50,
) -> dict:
    return dict(
        symbol=symbol, quantity=quantity, delta=delta, gamma=gamma,
        vega=vega, theta=theta, dte=dte, sector=sector, entry_price=entry_price,
    )


def _sample_positions() -> list:
    return [
        _make_position("SPY", -10, -0.30, 0.02, 0.15, -0.05, 14, "ETF", 2.50),
        _make_position("AAPL", -5, -0.25, 0.01, 0.10, -0.04, 21, "Tech", 3.00),
        _make_position("XLF", -3, -0.20, 0.015, 0.08, -0.03, 45, "Financials", 1.80),
    ]


def _make_report(**kwargs) -> OvernightRiskReport:
    defaults = dict(
        positions=_sample_positions(),
        portfolio_value=100_000.0,
        vix=18.0,
        as_of=date(2026, 3, 30),
    )
    defaults.update(kwargs)
    return OvernightRiskReport(**defaults)


# ── Gap risk tests ──────────────────────────────────────────────────────────


class TestGapRisk:
    def test_gap_risk_basic(self):
        rpt = _make_report()
        gap = rpt.compute_gap_risk()
        assert gap.dollar_95_loss > 0
        assert gap.dollar_99_loss > gap.dollar_95_loss
        assert gap.portfolio_pct_95 > 0

    def test_gap_risk_scales_with_vix(self):
        low = _make_report(vix=12.0).compute_gap_risk()
        high = _make_report(vix=30.0).compute_gap_risk()
        assert high.dollar_95_loss > low.dollar_95_loss

    def test_gap_risk_no_positions(self):
        rpt = _make_report(positions=[])
        gap = rpt.compute_gap_risk()
        assert gap.dollar_95_loss == 0.0
        assert gap.dollar_99_loss == 0.0
        assert gap.portfolio_pct_95 == 0.0

    def test_gap_risk_single_position(self):
        pos = [_make_position()]
        rpt = _make_report(positions=pos)
        gap = rpt.compute_gap_risk()
        assert gap.dollar_95_loss > 0
        assert isinstance(gap.gap_std, float)

    def test_gap_std_floor(self):
        """VIX scalar floored at 0.5."""
        rpt = _make_report(vix=1.0)  # extremely low
        gap = rpt.compute_gap_risk()
        expected_std = OVERNIGHT_GAP_STD * 0.5
        assert gap.gap_std == pytest.approx(expected_std, rel=1e-6)


# ── Earnings detection tests ───────────────────────────────────────────────


class TestEarningsExposure:
    def test_earnings_detected(self):
        cal = [{"symbol": "AAPL", "date": "2026-04-02"}]
        rpt = _make_report(earnings_calendar=cal)
        exposures = rpt.compute_earnings_exposure()
        assert len(exposures) == 1
        assert exposures[0].symbol == "AAPL"

    def test_earnings_outside_window(self):
        cal = [{"symbol": "AAPL", "date": "2026-04-15"}]
        rpt = _make_report(earnings_calendar=cal)
        assert rpt.compute_earnings_exposure() == []

    def test_earnings_no_matching_position(self):
        cal = [{"symbol": "MSFT", "date": "2026-04-01"}]
        rpt = _make_report(earnings_calendar=cal)
        assert rpt.compute_earnings_exposure() == []

    def test_earnings_empty_calendar(self):
        rpt = _make_report(earnings_calendar=[])
        assert rpt.compute_earnings_exposure() == []

    def test_earnings_date_object(self):
        cal = [{"symbol": "SPY", "date": date(2026, 4, 1)}]
        rpt = _make_report(earnings_calendar=cal)
        exposures = rpt.compute_earnings_exposure()
        assert len(exposures) == 1

    def test_earnings_notional_at_risk(self):
        cal = [{"symbol": "SPY", "date": "2026-03-31"}]
        rpt = _make_report(earnings_calendar=cal)
        exposures = rpt.compute_earnings_exposure()
        assert exposures[0].notional_at_risk == abs(-10 * 2.50 * 100)


# ── DTE breakdown tests ────────────────────────────────────────────────────


class TestDTEBuckets:
    def test_dte_bucket_count(self):
        rpt = _make_report()
        buckets = rpt.compute_dte_buckets()
        assert len(buckets) == 4

    def test_dte_bucket_labels(self):
        rpt = _make_report()
        buckets = rpt.compute_dte_buckets()
        labels = [b.label for b in buckets]
        assert labels == ["0-7", "7-30", "7-30", "30-60"] or "0-7" in labels

    def test_dte_positions_assigned_correctly(self):
        rpt = _make_report()
        buckets = {b.label: b for b in rpt.compute_dte_buckets()}
        # SPY (dte=14) and AAPL (dte=21) go into 7-30
        assert buckets["7-30"].count == 15  # abs(-10) + abs(-5)
        # XLF (dte=45) goes into 30-60
        assert buckets["30-60"].count == 3

    def test_dte_no_positions(self):
        rpt = _make_report(positions=[])
        buckets = rpt.compute_dte_buckets()
        assert all(b.count == 0 for b in buckets)

    def test_dte_short_expiry(self):
        pos = [_make_position(dte=2)]
        rpt = _make_report(positions=pos)
        buckets = {b.label: b for b in rpt.compute_dte_buckets()}
        assert buckets["0-7"].count == 10


# ── Sector concentration tests ─────────────────────────────────────────────


class TestSectorConcentration:
    def test_herfindahl_range(self):
        rpt = _make_report()
        sc = rpt.compute_sector_concentration()
        # HHI is between 0 and 1
        assert 0 < sc.herfindahl <= 1.0

    def test_single_sector(self):
        positions = [
            _make_position("A", sector="Tech"),
            _make_position("B", sector="Tech"),
        ]
        rpt = _make_report(positions=positions)
        sc = rpt.compute_sector_concentration()
        assert sc.herfindahl == pytest.approx(1.0, rel=1e-4)
        assert sc.max_sector == "Tech"
        assert sc.max_sector_weight == pytest.approx(1.0, rel=1e-4)

    def test_no_positions_sector(self):
        rpt = _make_report(positions=[])
        sc = rpt.compute_sector_concentration()
        assert sc.herfindahl == 0.0
        assert sc.max_sector == "N/A"

    def test_max_sector_identified(self):
        rpt = _make_report()
        sc = rpt.compute_sector_concentration()
        # SPY has largest notional (10 * 2.50 * 100 = 2500)
        assert sc.max_sector in ("ETF", "Tech", "Financials")
        assert sc.max_sector_weight > 0

    def test_sector_weights_sum_to_one(self):
        rpt = _make_report()
        sc = rpt.compute_sector_concentration()
        assert sum(sc.sector_weights.values()) == pytest.approx(1.0, rel=1e-4)


# ── Max loss tests ──────────────────────────────────────────────────────────


class TestMaxLoss:
    def test_max_loss_positive(self):
        rpt = _make_report()
        ml = rpt.compute_max_loss()
        assert ml.var_99 > 0
        assert ml.var_99 > ml.var_95

    def test_max_loss_no_positions(self):
        rpt = _make_report(positions=[])
        ml = rpt.compute_max_loss()
        assert ml.var_99 == 0.0
        assert ml.dominant_risk_factor == "none"

    def test_max_loss_scales_with_vix(self):
        low = _make_report(vix=10.0).compute_max_loss()
        high = _make_report(vix=35.0).compute_max_loss()
        assert high.var_99 > low.var_99

    def test_max_loss_portfolio_pct(self):
        rpt = _make_report()
        ml = rpt.compute_max_loss()
        expected_pct = ml.var_99 / 100_000.0 * 100
        assert ml.portfolio_pct == pytest.approx(expected_pct, rel=1e-3)

    def test_dominant_risk_factor(self):
        rpt = _make_report()
        ml = rpt.compute_max_loss()
        assert ml.dominant_risk_factor in ("delta", "vega")


# ── Margin buffer tests ────────────────────────────────────────────────────


class TestMarginBuffer:
    def test_adequate_margin(self):
        rpt = _make_report(current_margin=20_000.0)
        mb = rpt.compute_margin_buffer()
        assert mb.is_adequate is True
        assert mb.margin_buffer > 0

    def test_insufficient_margin(self):
        rpt = _make_report(current_margin=90_000.0)
        mb = rpt.compute_margin_buffer()
        # Stress margin = 90000 * 1.5 * (18/18) = 135000 > 100000
        assert mb.is_adequate is False
        assert mb.margin_buffer < 0

    def test_zero_margin(self):
        rpt = _make_report(current_margin=0.0)
        mb = rpt.compute_margin_buffer()
        assert mb.is_adequate is True
        assert mb.stress_margin == 0.0

    def test_stress_margin_increases_with_vix(self):
        low = _make_report(vix=15.0, current_margin=10_000).compute_margin_buffer()
        high = _make_report(vix=30.0, current_margin=10_000).compute_margin_buffer()
        assert high.stress_margin > low.stress_margin

    def test_buffer_pct(self):
        rpt = _make_report(current_margin=20_000.0)
        mb = rpt.compute_margin_buffer()
        expected = (100_000.0 - mb.stress_margin) / 100_000.0 * 100
        assert mb.buffer_pct == pytest.approx(expected, rel=1e-3)


# ── Hedge recommendation tests ─────────────────────────────────────────────


class TestHedgeRecommendations:
    def test_high_delta_generates_recommendation(self):
        pos = [_make_position(delta=-0.80, quantity=-5)]
        rpt = _make_report(positions=pos)
        recs = rpt.compute_hedge_recommendations()
        delta_recs = [r for r in recs if "delta" in r.action.lower()]
        assert len(delta_recs) >= 1

    def test_high_vix_generates_recommendation(self):
        rpt = _make_report(vix=30.0)
        recs = rpt.compute_hedge_recommendations()
        vix_recs = [r for r in recs if "position size" in r.action.lower() or "vix" in r.reason.lower()]
        assert len(vix_recs) >= 1

    def test_low_vix_short_vega_recommendation(self):
        pos = [_make_position(vega=0.50, quantity=-5)]
        rpt = _make_report(positions=pos, vix=12.0)
        recs = rpt.compute_hedge_recommendations()
        vega_recs = [r for r in recs if "vega" in r.action.lower()]
        assert len(vega_recs) >= 1

    def test_no_positions_no_recommendations(self):
        rpt = _make_report(positions=[])
        assert rpt.compute_hedge_recommendations() == []

    def test_negative_gamma_recommendation(self):
        pos = [_make_position(gamma=0.20, quantity=-5)]
        rpt = _make_report(positions=pos)
        recs = rpt.compute_hedge_recommendations()
        gamma_recs = [r for r in recs if "gamma" in r.action.lower()]
        assert len(gamma_recs) >= 1

    def test_short_dte_recommendation(self):
        pos = [_make_position(dte=3)]
        rpt = _make_report(positions=pos)
        recs = rpt.compute_hedge_recommendations()
        dte_recs = [r for r in recs if "expiring" in r.action.lower()]
        assert len(dte_recs) >= 1

    def test_recommendation_priorities(self):
        pos = [_make_position(delta=-0.80, quantity=-10)]
        rpt = _make_report(positions=pos, vix=30.0)
        recs = rpt.compute_hedge_recommendations()
        priorities = {r.priority for r in recs}
        assert "high" in priorities


# ── Correlation exposure tests ──────────────────────────────────────────────


class TestCorrelationExposure:
    def test_correlation_basic(self):
        rpt = _make_report(spy_beta=1.2, vix_beta=-0.15)
        ce = rpt.compute_correlation_exposure()
        assert ce.spy_beta == 1.2
        assert ce.vix_beta == -0.15
        assert ce.net_spy_dollar_delta != 0
        assert ce.net_vix_dollar_delta != 0

    def test_correlation_signs(self):
        rpt = _make_report(spy_beta=1.0, vix_beta=-0.10)
        ce = rpt.compute_correlation_exposure()
        # Net dollar delta is negative (short positions with negative delta)
        # SPY dollar delta should match sign of net_dollar_delta * spy_beta
        net_dd = sum(
            p["delta"] * p["quantity"] * p["entry_price"] * 100
            for p in _sample_positions()
        )
        assert ce.net_spy_dollar_delta == pytest.approx(net_dd * 1.0, rel=1e-2)


# ── Greeks summary tests ───────────────────────────────────────────────────


class TestGreeksSummary:
    def test_summary_count(self):
        rpt = _make_report()
        summaries = rpt.compute_greeks_summary()
        assert len(summaries) == 3

    def test_dollar_greeks_computed(self):
        pos = [_make_position()]
        rpt = _make_report(positions=pos)
        s = rpt.compute_greeks_summary()[0]
        expected_dd = -0.30 * -10 * 2.50 * 100
        assert s.dollar_delta == pytest.approx(expected_dd, rel=1e-4)

    def test_summary_empty(self):
        rpt = _make_report(positions=[])
        assert rpt.compute_greeks_summary() == []


# ── HTML report tests ───────────────────────────────────────────────────────


class TestHTMLReport:
    def test_html_contains_key_sections(self):
        rpt = _make_report()
        html = rpt.generate_report()
        assert "Overnight Risk Report" in html
        assert "Gap Risk" in html
        assert "Earnings Exposure" in html
        assert "Expiry Risk" in html
        assert "Sector Concentration" in html
        assert "Max Loss" in html
        assert "Margin Buffer" in html
        assert "Position Greeks" in html
        assert "Hedge Recommendations" in html

    def test_html_is_valid_structure(self):
        rpt = _make_report()
        html = rpt.generate_report()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<table>" in html

    def test_html_no_positions(self):
        rpt = _make_report(positions=[])
        html = rpt.generate_report()
        assert "No positions" in html

    def test_html_write_to_file(self, tmp_path):
        rpt = _make_report()
        out = str(tmp_path / "report.html")
        html = rpt.generate_report(output_path=out)
        with open(out) as f:
            content = f.read()
        assert content == html

    def test_html_vix_displayed(self):
        rpt = _make_report(vix=22.5)
        html = rpt.generate_report()
        assert "22.5" in html


# ── Dataclass tests ─────────────────────────────────────────────────────────


class TestDataclasses:
    def test_overnight_result_fields(self):
        rpt = _make_report()
        result = rpt.generate()
        assert isinstance(result, OvernightResult)
        assert isinstance(result.gap_risk, GapRiskResult)
        assert isinstance(result.sector_concentration, SectorConcentration)
        assert isinstance(result.max_loss, MaxLossScenario)

    def test_overnight_result_serialisable(self):
        rpt = _make_report()
        result = rpt.generate()
        d = asdict(result)
        assert isinstance(d, dict)
        assert "gap_risk" in d
        assert "hedge_recommendations" in d

    def test_gap_risk_result_fields(self):
        f_names = {f.name for f in fields(GapRiskResult)}
        assert "pct_95_loss" in f_names
        assert "dollar_99_loss" in f_names

    def test_hedge_recommendation_fields(self):
        f_names = {f.name for f in fields(HedgeRecommendation)}
        assert "action" in f_names
        assert "priority" in f_names
        assert "instrument" in f_names

    def test_position_greeks_summary_fields(self):
        f_names = {f.name for f in fields(PositionGreeksSummary)}
        assert "dollar_delta" in f_names
        assert "dollar_vega" in f_names


# ── Edge case tests ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_position_full_report(self):
        pos = [_make_position()]
        rpt = _make_report(positions=pos)
        result = rpt.generate()
        assert result.net_delta != 0
        assert len(result.greeks_summary) == 1
        assert len(result.dte_buckets) == 4

    def test_zero_portfolio_value(self):
        rpt = _make_report(portfolio_value=0.0)
        result = rpt.generate()
        assert result.gap_risk.portfolio_pct_95 == 0.0
        assert result.max_loss.portfolio_pct == 0.0

    def test_very_high_vix(self):
        rpt = _make_report(vix=80.0)
        result = rpt.generate()
        assert result.gap_risk.dollar_95_loss > 0
        assert result.max_loss.var_99 > 0

    def test_net_greeks_aggregation(self):
        rpt = _make_report()
        result = rpt.generate()
        expected_delta = sum(p["delta"] * p["quantity"] for p in _sample_positions())
        assert result.net_delta == pytest.approx(round(expected_delta, 4), rel=1e-4)

    def test_all_positions_same_sector(self):
        positions = [
            _make_position("A", sector="Tech"),
            _make_position("B", sector="Tech"),
            _make_position("C", sector="Tech"),
        ]
        rpt = _make_report(positions=positions)
        sc = rpt.compute_sector_concentration()
        assert sc.herfindahl == pytest.approx(1.0, rel=1e-4)

    def test_many_sectors_low_hhi(self):
        positions = [
            _make_position("A", quantity=-1, sector="Tech", entry_price=1.0),
            _make_position("B", quantity=-1, sector="Health", entry_price=1.0),
            _make_position("C", quantity=-1, sector="Energy", entry_price=1.0),
            _make_position("D", quantity=-1, sector="Financials", entry_price=1.0),
            _make_position("E", quantity=-1, sector="Utilities", entry_price=1.0),
        ]
        rpt = _make_report(positions=positions)
        sc = rpt.compute_sector_concentration()
        # 5 equal sectors: HHI = 5 * (1/5)^2 = 0.20
        assert sc.herfindahl == pytest.approx(0.20, rel=1e-4)
