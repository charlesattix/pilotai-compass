"""Tests for compass/trade_journal.py — trade journal and analytics.

Covers:
  - attribute_pnl: per-trade decomposition, edge cases, component sign logic
  - attribute_all: DataFrame enrichment
  - compute_streaks: win/loss runs, single trades, empty data
  - streak_summary: aggregation
  - day_of_week_analysis: grouping, ordering
  - regime_analysis: grouping, sorting
  - monthly_rollup: date grouping, aggregation
  - quarterly_rollup: period grouping
  - TradeJournal: from_csv, properties, report generation
"""

import numpy as np
import pandas as pd
import pytest

from compass.trade_journal import (
    TradeJournal,
    attribute_all,
    attribute_pnl,
    compute_streaks,
    day_of_week_analysis,
    monthly_rollup,
    quarterly_rollup,
    regime_analysis,
    streak_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _trade(
    pnl=100.0, net_credit=1.5, dte_at_entry=30, hold_days=10,
    contracts=2, spread_width=5.0, vix=20, realized_vol_20d=18,
    momentum_10d_pct=1.0, spread_type="bull_put", win=1,
    entry_date="2024-03-20", exit_date="2024-03-30",
    day_of_week=2, regime="bull", strategy_type="CS",
    exit_reason="close_profit_target", return_pct=5.0,
    **kwargs,
):
    d = {
        "pnl": pnl, "net_credit": net_credit, "dte_at_entry": dte_at_entry,
        "hold_days": hold_days, "contracts": contracts, "spread_width": spread_width,
        "vix": vix, "realized_vol_20d": realized_vol_20d,
        "momentum_10d_pct": momentum_10d_pct, "spread_type": spread_type,
        "win": win, "entry_date": entry_date, "exit_date": exit_date,
        "day_of_week": day_of_week, "regime": regime, "strategy_type": strategy_type,
        "exit_reason": exit_reason, "return_pct": return_pct,
    }
    d.update(kwargs)
    return pd.Series(d)


def _make_trades(n=20, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=n * 2)
    for i in range(n):
        pnl = rng.normal(50, 200)
        rows.append({
            "entry_date": str(dates[i * 2].date()),
            "exit_date": str(dates[i * 2 + 1].date()),
            "year": 2024,
            "strategy_type": rng.choice(["CS", "SS"]),
            "spread_type": rng.choice(["bull_put", "bear_call"]),
            "dte_at_entry": rng.randint(10, 50),
            "hold_days": rng.randint(1, 20),
            "day_of_week": rng.randint(0, 5),
            "regime": rng.choice(["bull", "bear", "neutral", "high_vol"]),
            "vix": round(rng.uniform(12, 40), 2),
            "realized_vol_20d": round(rng.uniform(10, 35), 2),
            "momentum_10d_pct": round(rng.normal(0, 3), 2),
            "net_credit": round(rng.uniform(0.5, 3.0), 4),
            "spread_width": 5.0,
            "contracts": rng.randint(1, 5),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / 500, 2),
            "win": 1 if pnl > 0 else 0,
            "exit_reason": rng.choice(["close_profit_target", "close_stop_loss", "close_expiration"]),
            "max_loss_per_unit": round(rng.uniform(3, 5), 4),
        })
    return pd.DataFrame(rows)


# ── attribute_pnl ────────────────────────────────────────────────────────


class TestAttributePnl:
    def test_components_sum_to_total(self):
        trade = _trade(pnl=200.0)
        attr = attribute_pnl(trade)
        assert attr["theta"] + attr["delta"] + attr["vega"] + attr["residual"] == pytest.approx(attr["total"], abs=0.1)

    def test_total_matches_trade_pnl(self):
        trade = _trade(pnl=150.0)
        attr = attribute_pnl(trade)
        assert attr["total"] == 150.0

    def test_theta_positive_for_credit_spread(self):
        trade = _trade(pnl=200.0, net_credit=1.5, hold_days=15, dte_at_entry=30)
        attr = attribute_pnl(trade)
        assert attr["theta"] > 0

    def test_theta_capped_at_pnl_for_winners(self):
        trade = _trade(pnl=50.0, net_credit=5.0, hold_days=28, dte_at_entry=30, contracts=3)
        attr = attribute_pnl(trade)
        assert attr["theta"] <= 50.0

    def test_nan_pnl_returns_zeros(self):
        trade = _trade(pnl=np.nan)
        attr = attribute_pnl(trade)
        assert attr["total"] == 0
        assert attr["theta"] == 0

    def test_losing_trade_has_positive_theta(self):
        """Even losing trades collect some theta before the loss."""
        trade = _trade(pnl=-500.0, net_credit=1.0, hold_days=5, dte_at_entry=30)
        attr = attribute_pnl(trade)
        assert attr["theta"] > 0  # still collected some time decay

    def test_bear_call_delta_sign(self):
        """Bear call: positive momentum should hurt (negative delta)."""
        trade = _trade(pnl=-100.0, spread_type="bear_call", momentum_10d_pct=5.0)
        attr = attribute_pnl(trade)
        # With bear_call, delta is negated, so positive momentum → negative delta
        assert attr["delta"] < 0

    def test_zero_credit_trade(self):
        trade = _trade(pnl=0, net_credit=0)
        attr = attribute_pnl(trade)
        assert attr["total"] == 0


# ── attribute_all ────────────────────────────────────────────────────────


class TestAttributeAll:
    def test_adds_attr_columns(self):
        trades = _make_trades(10)
        result = attribute_all(trades)
        for col in ["attr_theta", "attr_delta", "attr_vega", "attr_residual", "attr_total"]:
            assert col in result.columns

    def test_row_count_preserved(self):
        trades = _make_trades(15)
        result = attribute_all(trades)
        assert len(result) == 15

    def test_attr_total_matches_pnl(self):
        trades = _make_trades(10)
        result = attribute_all(trades)
        np.testing.assert_allclose(result["attr_total"], result["pnl"], atol=0.1)


# ── compute_streaks ──────────────────────────────────────────────────────


class TestComputeStreaks:
    def test_empty_trades(self):
        assert compute_streaks(pd.DataFrame()) == []

    def test_all_wins(self):
        trades = pd.DataFrame({
            "win": [1, 1, 1],
            "pnl": [100, 200, 150],
            "entry_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "exit_date": ["2024-01-05", "2024-01-06", "2024-01-07"],
            "regime": ["bull", "bull", "bull"],
        })
        streaks = compute_streaks(trades)
        assert len(streaks) == 1
        assert streaks[0].streak_type == "win"
        assert streaks[0].length == 3

    def test_alternating(self):
        trades = pd.DataFrame({
            "win": [1, 0, 1, 0],
            "pnl": [100, -50, 80, -30],
            "entry_date": ["2024-01-01"] * 4,
            "exit_date": ["2024-01-02"] * 4,
            "regime": ["bull"] * 4,
        })
        streaks = compute_streaks(trades)
        assert len(streaks) == 4
        assert all(s.length == 1 for s in streaks)

    def test_mixed_streaks(self):
        trades = pd.DataFrame({
            "win": [1, 1, 1, 0, 0, 1],
            "pnl": [100, 200, 150, -50, -80, 120],
            "entry_date": ["2024-01-01"] * 6,
            "exit_date": ["2024-01-02"] * 6,
            "regime": ["bull", "bull", "bull", "bear", "bear", "bull"],
        })
        streaks = compute_streaks(trades)
        assert len(streaks) == 3
        assert streaks[0].length == 3  # 3 wins
        assert streaks[1].length == 2  # 2 losses

    def test_streak_pnl_summed(self):
        trades = pd.DataFrame({
            "win": [1, 1],
            "pnl": [100, 200],
            "entry_date": ["2024-01-01", "2024-01-02"],
            "exit_date": ["2024-01-03", "2024-01-04"],
            "regime": ["bull", "bull"],
        })
        streaks = compute_streaks(trades)
        assert streaks[0].total_pnl == 300

    def test_streak_regimes_tracked(self):
        trades = pd.DataFrame({
            "win": [0, 0],
            "pnl": [-100, -200],
            "entry_date": ["2024-01-01", "2024-01-02"],
            "exit_date": ["2024-01-03", "2024-01-04"],
            "regime": ["bear", "high_vol"],
        })
        streaks = compute_streaks(trades)
        assert streaks[0].regimes == ["bear", "high_vol"]


# ── streak_summary ───────────────────────────────────────────────────────


class TestStreakSummary:
    def test_empty(self):
        s = streak_summary([])
        assert s["max_win_streak"] == 0
        assert s["max_loss_streak"] == 0

    def test_with_streaks(self):
        trades = pd.DataFrame({
            "win": [1, 1, 1, 0, 0, 1, 1],
            "pnl": [100] * 7,
            "entry_date": ["2024-01-01"] * 7,
            "exit_date": ["2024-01-02"] * 7,
            "regime": ["bull"] * 7,
        })
        streaks = compute_streaks(trades)
        s = streak_summary(streaks)
        assert s["max_win_streak"] == 3
        assert s["max_loss_streak"] == 2


# ── day_of_week_analysis ─────────────────────────────────────────────────


class TestDayOfWeekAnalysis:
    def test_returns_five_days(self):
        trades = _make_trades(50)
        result = day_of_week_analysis(trades)
        assert len(result) <= 5
        assert "day" in result.columns

    def test_ordered_mon_to_fri(self):
        trades = _make_trades(50)
        result = day_of_week_analysis(trades)
        days = result["day"].tolist()
        expected_order = [d for d in ["Mon", "Tue", "Wed", "Thu", "Fri"] if d in days]
        assert days == expected_order

    def test_empty_trades(self):
        result = day_of_week_analysis(pd.DataFrame())
        assert result.empty

    def test_has_win_rate(self):
        trades = _make_trades(30)
        result = day_of_week_analysis(trades)
        assert "win_rate" in result.columns
        assert result["win_rate"].between(0, 1).all()


# ── regime_analysis ──────────────────────────────────────────────────────


class TestRegimeAnalysis:
    def test_groups_by_regime(self):
        trades = _make_trades(30)
        result = regime_analysis(trades)
        assert "regime" in result.columns
        assert len(result) > 0

    def test_sorted_by_count(self):
        trades = _make_trades(50)
        result = regime_analysis(trades)
        counts = result["count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_empty_trades(self):
        result = regime_analysis(pd.DataFrame())
        assert result.empty


# ── monthly_rollup ───────────────────────────────────────────────────────


class TestMonthlyRollup:
    def test_returns_months(self):
        trades = _make_trades(30)
        result = monthly_rollup(trades)
        assert "month" in result.columns
        assert len(result) > 0

    def test_pnl_aggregated(self):
        trades = _make_trades(20)
        result = monthly_rollup(trades)
        assert "total_pnl" in result.columns
        # Total across months should approximate total trades PnL
        assert abs(result["total_pnl"].sum() - trades["pnl"].sum()) < 0.1

    def test_empty_trades(self):
        result = monthly_rollup(pd.DataFrame())
        assert result.empty


# ── quarterly_rollup ─────────────────────────────────────────────────────


class TestQuarterlyRollup:
    def test_returns_quarters(self):
        trades = _make_trades(30)
        result = quarterly_rollup(trades)
        assert "quarter" in result.columns
        assert len(result) > 0

    def test_empty_trades(self):
        result = quarterly_rollup(pd.DataFrame())
        assert result.empty


# ── TradeJournal ─────────────────────────────────────────────────────────


class TestTradeJournal:
    def test_from_dataframe(self):
        trades = _make_trades(30)
        journal = TradeJournal(trades)
        assert journal.n_trades == 30

    def test_from_csv(self, tmp_path):
        csv = tmp_path / "trades.csv"
        _make_trades(25).to_csv(csv, index=False)
        journal = TradeJournal.from_csv(str(csv))
        assert journal.n_trades == 25

    def test_total_pnl(self):
        trades = _make_trades(20)
        journal = TradeJournal(trades)
        assert journal.total_pnl == pytest.approx(trades["pnl"].sum(), abs=0.1)

    def test_win_rate(self):
        trades = _make_trades(20)
        journal = TradeJournal(trades)
        assert 0 <= journal.win_rate <= 1

    def test_attribution_summary(self):
        trades = _make_trades(20)
        journal = TradeJournal(trades)
        attr = journal.attribution_summary()
        assert "theta" in attr
        assert "delta" in attr
        assert "vega" in attr
        assert "residual" in attr
        assert "total" in attr

    def test_streaks_computed(self):
        trades = _make_trades(20)
        journal = TradeJournal(trades)
        assert isinstance(journal.streaks, list)

    def test_generate_report(self, tmp_path):
        trades = _make_trades(30)
        journal = TradeJournal(trades)
        out = str(tmp_path / "report.html")
        path = journal.generate_report(out)
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Trade Journal" in content
        assert "data:image/png;base64," in content
        assert "P&L Attribution" in content
        assert "Monthly P&L" in content
        assert "Regime Breakdown" in content
        assert "Top Streaks" in content
        assert len(content) > 5000

    def test_report_no_external_resources(self, tmp_path):
        trades = _make_trades(20)
        journal = TradeJournal(trades)
        path = journal.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "http://" not in content
        assert "https://" not in content
