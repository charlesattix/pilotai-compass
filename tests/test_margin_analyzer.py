"""Tests for compass/margin_analyzer.py — margin efficiency analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.margin_analyzer import (
    AnalysisResult,
    ExperimentEfficiency,
    MarginAnalyzer,
    MarginRequirement,
    StressScenario,
    UtilizationSnapshot,
    compute_margin,
    compute_stressed_margin,
    _fmt_pct,
    _fmt_dollar,
    _build_html,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_trades(
    n: int = 30, seed: int = 42, experiments: bool = True
) -> pd.DataFrame:
    """Generate synthetic trade data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    types = rng.choice(
        ["credit_spread", "iron_condor", "straddle", "strangle"], size=n
    )
    widths = rng.choice([2.0, 5.0, 10.0], size=n)
    contracts = rng.randint(1, 5, size=n)
    premiums = rng.uniform(50, 500, size=n)
    pnls = rng.normal(100, 300, size=n)
    prices = rng.uniform(400, 500, size=n)
    vixs = rng.uniform(15, 35, size=n)

    df = pd.DataFrame({
        "date": dates,
        "spread_type": types,
        "spread_width": widths,
        "contracts": contracts,
        "premium": premiums,
        "pnl": pnls,
        "underlying_price": prices,
        "vix": vixs,
    })
    if experiments:
        df["experiment"] = rng.choice(["EXP-400", "EXP-401", "EXP-402"], size=n)
    return df


@pytest.fixture
def trades():
    return _make_trades()


@pytest.fixture
def analyzer(trades):
    a = MarginAnalyzer(account_capital=100_000)
    a.add_trades(trades)
    return a


# ── Constructor tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_default_capital(self):
        a = MarginAnalyzer()
        assert a.account_capital == 100_000.0

    def test_custom_capital(self):
        a = MarginAnalyzer(account_capital=50_000)
        assert a.account_capital == 50_000.0

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError, match="positive"):
            MarginAnalyzer(account_capital=0)

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError, match="positive"):
            MarginAnalyzer(account_capital=-10_000)

    def test_add_trades_missing_columns(self):
        a = MarginAnalyzer()
        with pytest.raises(ValueError, match="Missing required"):
            a.add_trades(pd.DataFrame({"foo": [1]}))


# ── Margin computation tests ─────────────────────────────────────────────


class TestComputeMargin:
    def test_credit_spread(self):
        m = compute_margin("credit_spread", 5.0, 1, 100, 450)
        assert m == 500.0  # 5 * 1 * 100

    def test_iron_condor(self):
        m = compute_margin("iron_condor", 10.0, 2, 200, 450)
        assert m == 2000.0  # 10 * 2 * 100

    def test_straddle_base_vix(self):
        m = compute_margin("straddle", 0, 1, 100, 450, vix=20)
        expected = (0.20 * 450 * 100 + 100) * 1.0
        assert abs(m - expected) < 0.01

    def test_straddle_high_vix(self):
        m_low = compute_margin("straddle", 0, 1, 100, 450, vix=20)
        m_high = compute_margin("straddle", 0, 1, 100, 450, vix=40)
        assert m_high > m_low  # VIX doubles → margin increases

    def test_strangle(self):
        m = compute_margin("strangle", 0, 1, 100, 450, vix=20)
        expected = (0.15 * 450 * 100 + 100) * 1.0
        assert abs(m - expected) < 0.01

    def test_strangle_high_vix(self):
        m_low = compute_margin("strangle", 0, 1, 100, 450, vix=20)
        m_high = compute_margin("strangle", 0, 1, 100, 450, vix=40)
        assert m_high > m_low

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown spread_type"):
            compute_margin("butterfly", 5, 1, 100, 450)

    def test_multiple_contracts(self):
        m1 = compute_margin("credit_spread", 5.0, 1, 100, 450)
        m3 = compute_margin("credit_spread", 5.0, 3, 300, 450)
        assert m3 == m1 * 3


# ── MarginRequirement dataclass tests ────────────────────────────────────


class TestMarginRequirement:
    def test_return_on_margin(self):
        mr = MarginRequirement(
            spread_type="credit_spread", spread_width=5, contracts=1,
            premium_received=100, margin_required=500,
            underlying_price=450,
        )
        assert mr.return_on_margin == pytest.approx(0.2)

    def test_return_on_margin_zero_margin(self):
        mr = MarginRequirement(
            spread_type="credit_spread", spread_width=0, contracts=0,
            premium_received=100, margin_required=0,
            underlying_price=450,
        )
        assert mr.return_on_margin == 0.0

    def test_to_dict(self):
        mr = MarginRequirement(
            spread_type="iron_condor", spread_width=10, contracts=2,
            premium_received=200, margin_required=2000,
            underlying_price=450, vix=25,
        )
        d = mr.to_dict()
        assert d["spread_type"] == "iron_condor"
        assert "return_on_margin" in d


# ── UtilizationSnapshot tests ────────────────────────────────────────────


class TestUtilizationSnapshot:
    def test_utilization_pct(self):
        s = UtilizationSnapshot(
            date=pd.Timestamp("2024-01-02"),
            total_margin_used=60_000,
            account_capital=100_000,
            n_positions=5,
        )
        assert s.utilization_pct == pytest.approx(0.6)

    def test_buying_power_remaining(self):
        s = UtilizationSnapshot(
            date=pd.Timestamp("2024-01-02"),
            total_margin_used=80_000,
            account_capital=100_000,
            n_positions=5,
        )
        assert s.buying_power_remaining == pytest.approx(20_000)

    def test_buying_power_floor_zero(self):
        s = UtilizationSnapshot(
            date=pd.Timestamp("2024-01-02"),
            total_margin_used=120_000,
            account_capital=100_000,
            n_positions=5,
        )
        assert s.buying_power_remaining == 0.0


# ── Analyzer integration tests ───────────────────────────────────────────


class TestAnalyzerIntegration:
    def test_compute_all_margins(self, analyzer):
        margins = analyzer.compute_all_margins()
        assert len(margins) == 30
        assert all(isinstance(m, MarginRequirement) for m in margins)
        assert all(m.margin_required > 0 for m in margins)

    def test_utilization_history(self, analyzer):
        margins = analyzer.compute_all_margins()
        history = analyzer.compute_utilization_history(margins)
        assert len(history) > 0
        assert all(isinstance(s, UtilizationSnapshot) for s in history)
        # Sorted by date
        dates = [s.date for s in history]
        assert dates == sorted(dates)

    def test_experiment_efficiency(self, analyzer):
        margins = analyzer.compute_all_margins()
        eff = analyzer.compute_experiment_efficiency(margins)
        assert len(eff) > 0
        assert all(isinstance(e, ExperimentEfficiency) for e in eff)
        # Sorted by ROM descending
        roms = [e.avg_return_on_margin for e in eff]
        assert roms == sorted(roms, reverse=True)

    def test_no_experiment_column(self):
        trades = _make_trades(experiments=False)
        a = MarginAnalyzer()
        a.add_trades(trades)
        margins = a.compute_all_margins()
        eff = a.compute_experiment_efficiency(margins)
        assert eff == []


# ── Stress scenario tests ───────────────────────────────────────────────


class TestStressScenarios:
    def test_default_scenarios(self, analyzer):
        margins = analyzer.compute_all_margins()
        stress = analyzer.run_stress_scenarios(margins)
        assert len(stress) == 5

    def test_custom_scenarios(self, analyzer):
        margins = analyzer.compute_all_margins()
        custom = [("Mild", 1.1), ("Extreme", 5.0)]
        stress = analyzer.run_stress_scenarios(margins, scenarios=custom)
        assert len(stress) == 2
        assert stress[0].scenario_name == "Mild"

    def test_stress_margin_increases(self, analyzer):
        margins = analyzer.compute_all_margins()
        stress = analyzer.run_stress_scenarios(margins)
        for s in stress:
            assert s.stressed_margin >= s.baseline_margin

    def test_stressed_margin_computation(self):
        mr = MarginRequirement(
            spread_type="straddle", spread_width=0, contracts=1,
            premium_received=100, margin_required=9100,
            underlying_price=450, vix=20,
        )
        stressed = compute_stressed_margin(mr, 2.0)
        assert stressed > mr.margin_required

    def test_credit_spread_stress_unchanged(self):
        """Credit spreads have fixed margin regardless of VIX."""
        mr = MarginRequirement(
            spread_type="credit_spread", spread_width=5, contracts=1,
            premium_received=100, margin_required=500,
            underlying_price=450, vix=20,
        )
        stressed = compute_stressed_margin(mr, 3.0)
        assert stressed == mr.margin_required

    def test_empty_margins_no_stress(self, analyzer):
        stress = analyzer.run_stress_scenarios([])
        assert stress == []


# ── Buying power tests ───────────────────────────────────────────────────


class TestBuyingPower:
    def test_buying_power_impact(self, analyzer):
        margins = analyzer.compute_all_margins()
        bp = MarginAnalyzer.buying_power_impact(100_000, margins)
        assert bp["account_capital"] == 100_000
        assert bp["total_margin_used"] > 0
        assert bp["buying_power_remaining"] >= 0
        assert 0 <= bp["utilization_pct"] <= 10  # reasonable range
        assert bp["n_positions"] == 30

    def test_buying_power_empty(self):
        bp = MarginAnalyzer.buying_power_impact(100_000, [])
        assert bp["total_margin_used"] == 0
        assert bp["buying_power_remaining"] == 100_000
        assert bp["avg_margin_per_position"] == 0

    def test_buying_power_by_type(self, analyzer):
        margins = analyzer.compute_all_margins()
        bp = MarginAnalyzer.buying_power_impact(100_000, margins)
        by_type = bp["margin_by_spread_type"]
        assert isinstance(by_type, dict)
        assert sum(by_type.values()) == pytest.approx(bp["total_margin_used"])


# ── Full analysis tests ─────────────────────────────────────────────────


class TestFullAnalysis:
    def test_analyze_returns_result(self, analyzer):
        result = analyzer.analyze()
        assert isinstance(result, AnalysisResult)
        assert result.account_capital == 100_000
        assert len(result.margin_requirements) == 30
        assert len(result.stress_scenarios) == 5
        assert "total_margin" in result.summary

    def test_analyze_empty_trades(self):
        a = MarginAnalyzer()
        trades = pd.DataFrame({
            "spread_type": pd.Series(dtype=str),
            "spread_width": pd.Series(dtype=float),
            "contracts": pd.Series(dtype=int),
            "premium": pd.Series(dtype=float),
        })
        a.add_trades(trades)
        result = a.analyze()
        assert len(result.margin_requirements) == 0


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generate_report_creates_file(self, analyzer):
        result = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_margin.html"
            path = MarginAnalyzer.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Margin Efficiency Analysis" in content

    def test_report_contains_sections(self, analyzer):
        result = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MarginAnalyzer.generate_report(result, out)
            content = out.read_text()
            assert "Buying Power" in content
            assert "Stress Scenarios" in content
            assert "Recommendations" in content

    def test_report_contains_svg(self, analyzer):
        result = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MarginAnalyzer.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content

    def test_fmt_pct(self):
        assert _fmt_pct(0.1234) == "12.34%"
        assert _fmt_pct(-0.05) == "-5.00%"

    def test_fmt_dollar(self):
        assert _fmt_dollar(1234.56) == "$1,234.56"
        assert _fmt_dollar(0) == "$0.00"

    def test_report_default_path(self, analyzer):
        result = analyzer.analyze()
        path = MarginAnalyzer.generate_report(result)
        assert path.exists()
        assert "margin_analysis.html" in str(path)
        path.unlink(missing_ok=True)
