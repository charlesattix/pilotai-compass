"""Tests for shared/deviation_tracker.py — persistent deviation tracking.

Covers both:
  - Daily snapshot tracking (original)
  - Per-trade deviation tracking (INF-5)
"""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from shared.database import init_db
from shared.deviation_tracker import (
    _ensure_trade_deviations_table,
    _upsert_snapshot,
    check_deviation_alerts,
    get_deviation_history,
    get_latest_deviation,
    get_rolling_alignment,
    print_report,
    record_deviation,
    record_deviation_snapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Create a temp DB with schema initialized."""
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def _make_live_metrics(total=10, win_rate=65.0, pnl=500.0, pf=1.8, dd=-3.5, ret=0.5):
    """Build a realistic live metrics dict."""
    return {
        "total_trades": total,
        "winning_trades": int(total * win_rate / 100),
        "losing_trades": total - int(total * win_rate / 100),
        "win_rate": win_rate,
        "total_pnl": pnl,
        "avg_win": 100.0,
        "avg_loss": 50.0,
        "profit_factor": pf,
        "return_pct": ret,
        "max_drawdown": dd,
        "per_strategy": {"credit_spread": {"total_trades": total, "win_rate": win_rate, "total_pnl": pnl}},
        "start_date": "2026-03-01",
        "end_date": "2026-03-07",
    }


def _make_backtest_results(total=12, win_rate=70.0, pnl=600.0, pf=2.0, dd=-4.0, ret=0.6):
    """Build a realistic backtest results dict."""
    return {
        "combined": {
            "total_trades": total,
            "win_rate": win_rate,
            "total_pnl": pnl,
            "avg_win": 110.0,
            "avg_loss": 55.0,
            "profit_factor": pf,
            "return_pct": ret,
            "max_drawdown": dd,
        },
        "per_strategy": {"credit_spread": {"total_trades": total, "win_rate": win_rate, "total_pnl": pnl}},
    }


def _make_trades(n=10, pnl_each=50.0):
    """Build a list of mock trade dicts."""
    trades = []
    for i in range(n):
        trades.append({
            "id": f"trade-{i}",
            "ticker": "SPY",
            "strategy_type": "credit_spread",
            "status": "closed_profit" if pnl_each > 0 else "closed_loss",
            "pnl": pnl_each,
            "entry_date": f"2026-03-0{min(i+1, 9)}",
            "exit_date": f"2026-03-0{min(i+2, 9)}",
        })
    return trades


def _make_position(trade_id="test-001", credit=0.38, short_strike=450, long_strike=445,
                   contracts=1, entry_date=None):
    """Build a mock position dict for per-trade deviation testing."""
    if entry_date is None:
        entry_date = (datetime.now() - timedelta(days=15)).isoformat()
    return {
        "id": trade_id,
        "credit": credit,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "contracts": contracts,
        "entry_date": entry_date,
        "strategy_type": "credit_spread",
    }


# ---------------------------------------------------------------------------
# TestRecordDeviation (Daily Snapshots)
# ---------------------------------------------------------------------------

class TestRecordDeviationSnapshot:
    """Tests for record_deviation_snapshot()."""

    @patch("scripts.live_vs_backtest.load_live_trades")
    @patch("scripts.live_vs_backtest.compute_live_metrics")
    @patch("scripts.live_vs_backtest.run_backtest_for_range")
    @patch("scripts.live_vs_backtest.compare_metrics")
    def test_snapshot_persisted(self, mock_compare, mock_bt, mock_metrics, mock_trades, db_path):
        """Full snapshot with backtest is persisted correctly."""
        mock_trades.return_value = _make_trades(10)
        mock_metrics.return_value = _make_live_metrics()
        mock_bt.return_value = _make_backtest_results()
        mock_compare.return_value = [
            {"metric": "Win Rate", "status": "PASS", "live_str": "65.0%", "backtest_str": "70.0%",
             "live_value": 65.0, "backtest_value": 70.0, "deviation_str": "-5.0pp", "deviation_pct": -5.0},
        ]

        result = record_deviation_snapshot(db_path=db_path)

        assert result is not None
        assert result["live_trades"] == 10
        assert result["overall_status"] == "PASS"

        stored = get_latest_deviation(db_path)
        assert stored is not None
        assert stored["live_trades"] == 10
        assert stored["overall_status"] == "PASS"

    @patch("scripts.live_vs_backtest.load_live_trades")
    @patch("scripts.live_vs_backtest.compute_live_metrics")
    def test_empty_trades_returns_none(self, mock_metrics, mock_trades, db_path):
        """No closed trades → returns None."""
        mock_trades.return_value = []
        mock_metrics.return_value = {"total_trades": 0}

        result = record_deviation_snapshot(db_path=db_path)
        assert result is None

    @patch("scripts.live_vs_backtest.load_live_trades")
    @patch("scripts.live_vs_backtest.compute_live_metrics")
    def test_few_trades_live_only(self, mock_metrics, mock_trades, db_path):
        """< 5 trades → live-only snapshot, bt fields null."""
        mock_trades.return_value = _make_trades(3)
        mock_metrics.return_value = _make_live_metrics(total=3)

        result = record_deviation_snapshot(db_path=db_path)

        assert result is not None
        assert result["live_trades"] == 3
        assert result["bt_trades"] is None
        assert result["overall_status"] == "INFO"

    @patch("scripts.live_vs_backtest.load_live_trades")
    @patch("scripts.live_vs_backtest.compute_live_metrics")
    @patch("scripts.live_vs_backtest.run_backtest_for_range")
    @patch("scripts.live_vs_backtest.compare_metrics")
    def test_overall_status_worst(self, mock_compare, mock_bt, mock_metrics, mock_trades, db_path):
        """overall_status reflects the worst metric status."""
        mock_trades.return_value = _make_trades(10)
        mock_metrics.return_value = _make_live_metrics()
        mock_bt.return_value = _make_backtest_results()
        mock_compare.return_value = [
            {"metric": "Win Rate", "status": "PASS", "live_str": "65%", "backtest_str": "70%",
             "live_value": 65, "backtest_value": 70, "deviation_str": "-5pp", "deviation_pct": -5},
            {"metric": "Profit Factor", "status": "FAIL", "live_str": "0.8", "backtest_str": "2.0",
             "live_value": 0.8, "backtest_value": 2.0, "deviation_str": "-60%", "deviation_pct": -60},
        ]

        result = record_deviation_snapshot(db_path=db_path)
        assert result["overall_status"] == "FAIL"


# ---------------------------------------------------------------------------
# TestPerTradeDeviation (INF-5)
# ---------------------------------------------------------------------------

class TestPerTradeDeviation:
    """Tests for record_deviation() — per-trade paper vs backtest."""

    def test_basic_winning_trade(self, db_path):
        """Winning trade records correct deviation."""
        pos = _make_position(credit=0.38, short_strike=450, long_strike=445)
        # P&L: (0.38 - 0.19) * 1 * 100 = $19 (win at ~50% profit target)
        result = record_deviation(pos, pnl=19.0, fill_price=0.19, db_path=db_path)

        assert result is not None
        assert result["trade_id"] == "test-001"
        assert result["paper_credit"] == 0.38
        assert result["expected_credit"] == 1.75  # 5 * 0.35
        assert result["paper_outcome"] == "win"
        assert result["expected_outcome"] == "win"
        assert result["deviation_score"] >= 0

    def test_losing_trade(self, db_path):
        """Losing trade has outcome mismatch with expected win."""
        pos = _make_position(trade_id="loss-001", credit=0.30, short_strike=450, long_strike=445)
        result = record_deviation(pos, pnl=-200.0, fill_price=2.30, db_path=db_path)

        assert result is not None
        assert result["paper_outcome"] == "loss"
        assert result["expected_outcome"] == "win"
        assert result["deviation_score"] > 0  # mismatch adds to score

    def test_no_spread_width_returns_none(self, db_path):
        """Missing strikes → returns None."""
        pos = {"id": "no-strikes", "credit": 0.5, "contracts": 1}
        result = record_deviation(pos, pnl=10.0, fill_price=0.25, db_path=db_path)
        assert result is None

    def test_no_trade_id_returns_none(self, db_path):
        """Missing trade ID → returns None."""
        pos = {"credit": 0.5, "short_strike": 450, "long_strike": 445}
        result = record_deviation(pos, pnl=10.0, fill_price=0.25, db_path=db_path)
        assert result is None

    def test_upsert_on_duplicate(self, db_path):
        """Recording same trade_id twice updates rather than duplicating."""
        pos = _make_position(trade_id="dup-001")
        record_deviation(pos, pnl=19.0, fill_price=0.19, db_path=db_path)
        record_deviation(pos, pnl=25.0, fill_price=0.13, db_path=db_path)

        metrics = get_rolling_alignment(db_path)
        assert metrics["trade_count"] == 1  # not 2

    def test_scratch_trade(self, db_path):
        """Near-zero P&L → scratch outcome."""
        pos = _make_position(trade_id="scratch-001")
        result = record_deviation(pos, pnl=0.0, fill_price=0.38, db_path=db_path)
        assert result["paper_outcome"] == "scratch"

    def test_custom_config(self, db_path):
        """Custom config overrides default backtest assumptions."""
        pos = _make_position(credit=0.50, short_strike=450, long_strike=445)
        config = {
            "backtest": {"credit_ratio": 0.40, "expected_hold_days": 14},
            "risk": {"profit_target": 60},
        }
        result = record_deviation(pos, pnl=30.0, fill_price=0.20, db_path=db_path, config=config)

        assert result["expected_credit"] == 2.0  # 5 * 0.40
        assert result["expected_pnl_pct"] == 60.0
        assert result["expected_hold_days"] == 14.0


# ---------------------------------------------------------------------------
# TestRollingAlignment
# ---------------------------------------------------------------------------

class TestRollingAlignment:
    """Tests for get_rolling_alignment()."""

    def test_empty_db(self, db_path):
        """No data → perfect defaults."""
        metrics = get_rolling_alignment(db_path)
        assert metrics["alignment_score"] == 1.0
        assert metrics["credit_deviation"] == 0.0
        assert metrics["trade_count"] == 0

    def test_all_matching(self, db_path):
        """All wins matching expected → 100% alignment."""
        for i in range(5):
            pos = _make_position(trade_id=f"win-{i}", credit=1.75, short_strike=450, long_strike=445)
            record_deviation(pos, pnl=87.5, fill_price=0.875, db_path=db_path)

        metrics = get_rolling_alignment(db_path)
        assert metrics["alignment_score"] == 1.0
        assert metrics["trade_count"] == 5

    def test_mixed_outcomes(self, db_path):
        """Mix of wins and losses → partial alignment."""
        # 3 wins
        for i in range(3):
            pos = _make_position(trade_id=f"w-{i}", short_strike=450, long_strike=445)
            record_deviation(pos, pnl=50.0, fill_price=0.10, db_path=db_path)
        # 2 losses (mismatch with expected "win")
        for i in range(2):
            pos = _make_position(trade_id=f"l-{i}", short_strike=450, long_strike=445)
            record_deviation(pos, pnl=-200.0, fill_price=2.30, db_path=db_path)

        metrics = get_rolling_alignment(db_path)
        assert metrics["alignment_score"] == 0.6  # 3/5
        assert metrics["trade_count"] == 5

    def test_credit_deviation_calculation(self, db_path):
        """Credit deviation computed correctly."""
        # expected_credit = 5 * 0.35 = 1.75
        # paper_credit = 0.875 → deviation = |0.875-1.75|/1.75 = 0.5
        pos = _make_position(trade_id="cd-1", credit=0.875, short_strike=450, long_strike=445)
        record_deviation(pos, pnl=40.0, fill_price=0.10, db_path=db_path)

        metrics = get_rolling_alignment(db_path)
        assert abs(metrics["credit_deviation"] - 0.5) < 0.01

    def test_respects_window(self, db_path):
        """Only returns up to window trades."""
        for i in range(25):
            pos = _make_position(trade_id=f"t-{i}", short_strike=450, long_strike=445)
            record_deviation(pos, pnl=50.0, fill_price=0.10, db_path=db_path)

        metrics = get_rolling_alignment(db_path, window=10)
        assert metrics["trade_count"] == 10


# ---------------------------------------------------------------------------
# TestRollingAlerts
# ---------------------------------------------------------------------------

class TestRollingAlerts:
    """Tests for _check_rolling_alerts() via record_deviation()."""

    @patch("shared.deviation_tracker.get_rolling_alignment")
    @patch("shared.telegram_alerts.send_message")
    def test_alignment_alert_sent(self, mock_send, mock_metrics, db_path):
        """Low alignment triggers Telegram alert."""
        mock_metrics.return_value = {
            "alignment_score": 0.55,
            "credit_deviation": 0.10,
            "trade_count": 20,
            "recent_deviations": [],
        }
        mock_send.return_value = True

        pos = _make_position(short_strike=450, long_strike=445)
        record_deviation(pos, pnl=50.0, fill_price=0.10, db_path=db_path)

        mock_send.assert_called_once()
        call_text = mock_send.call_args[0][0]
        assert "alignment score" in call_text.lower() or "alignment" in call_text.lower()

    @patch("shared.deviation_tracker.get_rolling_alignment")
    @patch("shared.telegram_alerts.send_message")
    def test_credit_deviation_alert_sent(self, mock_send, mock_metrics, db_path):
        """High credit deviation triggers Telegram alert."""
        mock_metrics.return_value = {
            "alignment_score": 0.85,
            "credit_deviation": 0.35,
            "trade_count": 20,
            "recent_deviations": [],
        }
        mock_send.return_value = True

        pos = _make_position(trade_id="alert-cd", short_strike=450, long_strike=445)
        record_deviation(pos, pnl=50.0, fill_price=0.10, db_path=db_path)

        mock_send.assert_called_once()
        call_text = mock_send.call_args[0][0]
        assert "credit" in call_text.lower()

    @patch("shared.deviation_tracker.get_rolling_alignment")
    @patch("shared.telegram_alerts.send_message")
    def test_no_alert_when_ok(self, mock_send, mock_metrics, db_path):
        """Good metrics → no alert."""
        mock_metrics.return_value = {
            "alignment_score": 0.85,
            "credit_deviation": 0.10,
            "trade_count": 20,
            "recent_deviations": [],
        }

        pos = _make_position(trade_id="ok-001", short_strike=450, long_strike=445)
        record_deviation(pos, pnl=50.0, fill_price=0.10, db_path=db_path)

        mock_send.assert_not_called()

    @patch("shared.deviation_tracker.get_rolling_alignment")
    def test_no_alert_below_window(self, mock_metrics, db_path):
        """< 20 trades → no alert check."""
        mock_metrics.return_value = {
            "alignment_score": 0.30,
            "credit_deviation": 0.50,
            "trade_count": 10,
            "recent_deviations": [],
        }
        # Should not raise or send anything
        pos = _make_position(trade_id="few-001", short_strike=450, long_strike=445)
        record_deviation(pos, pnl=50.0, fill_price=0.10, db_path=db_path)


# ---------------------------------------------------------------------------
# TestPrintReport
# ---------------------------------------------------------------------------

class TestPrintReport:
    """Tests for print_report() CLI output."""

    def test_empty_report(self, db_path, capsys):
        """Empty DB prints placeholder."""
        print_report(db_path)
        captured = capsys.readouterr()
        assert "No trade deviations" in captured.out

    def test_report_with_data(self, db_path, capsys):
        """Report with data shows alignment score."""
        for i in range(3):
            pos = _make_position(trade_id=f"rpt-{i}", short_strike=450, long_strike=445)
            record_deviation(pos, pnl=50.0, fill_price=0.10, db_path=db_path)

        print_report(db_path)
        captured = capsys.readouterr()
        assert "Alignment score" in captured.out
        assert "Credit deviation" in captured.out
        assert "rpt-" in captured.out


# ---------------------------------------------------------------------------
# TestGetHistory (daily snapshots)
# ---------------------------------------------------------------------------

class TestGetHistory:
    """Tests for get_deviation_history()."""

    def test_ordered_newest_first(self, db_path):
        for i in range(3):
            _upsert_snapshot({
                "snapshot_date": f"2026-03-0{i+1}",
                "live_trades": 10 + i,
                "overall_status": "PASS",
                "details": {},
            }, db_path)

        history = get_deviation_history(days=30, db_path=db_path)
        assert len(history) == 3
        assert history[0]["snapshot_date"] == "2026-03-03"

    def test_respects_days_param(self, db_path):
        today = datetime.now()
        _upsert_snapshot({
            "snapshot_date": today.strftime("%Y-%m-%d"),
            "live_trades": 10,
            "overall_status": "PASS",
            "details": {},
        }, db_path)
        _upsert_snapshot({
            "snapshot_date": (today - timedelta(days=60)).strftime("%Y-%m-%d"),
            "live_trades": 5,
            "overall_status": "WARN",
            "details": {},
        }, db_path)

        history = get_deviation_history(days=30, db_path=db_path)
        assert len(history) == 1

    def test_empty_db(self, db_path):
        assert get_deviation_history(db_path=db_path) == []


# ---------------------------------------------------------------------------
# TestGetLatest
# ---------------------------------------------------------------------------

class TestGetLatest:

    def test_returns_most_recent(self, db_path):
        _upsert_snapshot({"snapshot_date": "2026-03-01", "live_trades": 5, "overall_status": "PASS", "details": {}}, db_path)
        _upsert_snapshot({"snapshot_date": "2026-03-05", "live_trades": 15, "overall_status": "WARN", "details": {}}, db_path)

        latest = get_latest_deviation(db_path)
        assert latest["snapshot_date"] == "2026-03-05"

    def test_empty_db_returns_none(self, db_path):
        assert get_latest_deviation(db_path) is None


# ---------------------------------------------------------------------------
# TestCheckAlerts (daily snapshots)
# ---------------------------------------------------------------------------

class TestCheckAlerts:

    def test_all_pass_empty(self):
        snapshot = {"details": {"comparisons": [
            {"metric": "Win Rate", "status": "PASS", "live_str": "65%", "backtest_str": "70%"},
        ]}}
        assert check_deviation_alerts(snapshot) == []

    def test_warn_metrics(self):
        snapshot = {"details": {"comparisons": [
            {"metric": "Win Rate", "status": "WARN", "live_str": "55.0%", "backtest_str": "70.0%"},
        ]}}
        alerts = check_deviation_alerts(snapshot)
        assert len(alerts) == 1
        assert "WARN" in alerts[0]

    def test_mixed_statuses(self):
        snapshot = {"details": {"comparisons": [
            {"metric": "Win Rate", "status": "PASS", "live_str": "65%", "backtest_str": "70%"},
            {"metric": "Profit Factor", "status": "WARN", "live_str": "1.2", "backtest_str": "2.0"},
            {"metric": "Max Drawdown", "status": "FAIL", "live_str": "-15%", "backtest_str": "-5%"},
            {"metric": "Total Trades", "status": "INFO", "live_str": "10", "backtest_str": "12"},
        ]}}
        alerts = check_deviation_alerts(snapshot)
        assert len(alerts) == 2
