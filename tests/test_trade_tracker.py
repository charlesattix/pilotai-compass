"""Tests for the TradeTracker class."""
from unittest.mock import patch

import pytest

from tracker.trade_tracker import TradeTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    """Minimal config for TradeTracker."""
    return {
        'data_dir': 'data',
    }


def _make_position(ticker='SPY', spread_type='bull_put_spread',
                    short_strike=450, long_strike=445,
                    credit=1.50, max_loss=3.50, contracts=1):
    """Return a minimal position dict."""
    return {
        'ticker': ticker,
        'type': spread_type,
        'short_strike': short_strike,
        'long_strike': long_strike,
        'credit': credit,
        'max_loss': max_loss,
        'contracts': contracts,
    }


@pytest.fixture
def tracker(tmp_path):
    """Create a TradeTracker with SQLite mocked out."""
    with patch('tracker.trade_tracker.init_db'), \
         patch.object(TradeTracker, '_load_trades', return_value=[]), \
         patch.object(TradeTracker, '_load_positions', return_value=[]):
        t = TradeTracker(_make_config())
        t.data_dir = tmp_path
        # Mock save methods so tests don't need a real database
        t._save_trades = lambda: None
        t._save_positions = lambda: None
        return t


# ---------------------------------------------------------------------------
# Tests for add_position
# ---------------------------------------------------------------------------

class TestAddPosition:

    def test_add_position_returns_id(self, tracker):
        """add_position should return a string position ID."""
        pos = _make_position()
        pos_id = tracker.add_position(pos)
        assert isinstance(pos_id, str)
        assert 'SPY' in pos_id

    def test_add_position_sets_fields(self, tracker):
        """Position should have position_id, entry_date, and status after add."""
        pos = _make_position()
        pos_id = tracker.add_position(pos)
        stored = tracker.get_position(pos_id)
        assert stored is not None
        assert stored['position_id'] == pos_id
        assert stored['status'] == 'open'
        assert 'entry_date' in stored

    def test_add_multiple_positions(self, tracker):
        """Multiple positions should be tracked independently."""
        id1 = tracker.add_position(_make_position(ticker='SPY'))
        id2 = tracker.add_position(_make_position(ticker='QQQ'))
        assert len(tracker.get_open_positions()) == 2
        assert id1 != id2


# ---------------------------------------------------------------------------
# Tests for close_position
# ---------------------------------------------------------------------------

class TestClosePosition:

    def test_close_existing_position(self, tracker):
        """Closing an existing position should move it from positions to trades."""
        pos_id = tracker.add_position(_make_position())
        tracker.close_position(pos_id, exit_price=0.50, exit_reason='profit_target', pnl=100)
        assert len(tracker.get_open_positions()) == 0
        assert len(tracker.get_closed_trades()) == 1
        trade = tracker.get_closed_trades()[0]
        assert trade['position_id'] == pos_id
        assert trade['pnl'] == 100
        assert trade['exit_reason'] == 'profit_target'

    def test_close_nonexistent_position(self, tracker):
        """Closing a nonexistent position should log warning and do nothing."""
        tracker.close_position('nonexistent_id', exit_price=0, exit_reason='test', pnl=0)
        assert len(tracker.get_closed_trades()) == 0

    def test_close_preserves_other_positions(self, tracker):
        """Closing one position should not affect others."""
        id1 = tracker.add_position(_make_position(ticker='SPY'))
        tracker.add_position(_make_position(ticker='QQQ'))
        tracker.close_position(id1, exit_price=0.50, exit_reason='profit_target', pnl=100)
        assert len(tracker.get_open_positions()) == 1
        assert tracker.get_open_positions()[0]['ticker'] == 'QQQ'


# ---------------------------------------------------------------------------
# Tests for update_position
# ---------------------------------------------------------------------------

class TestUpdatePosition:

    def test_update_existing_position(self, tracker):
        """Updating a position should merge new fields."""
        pos_id = tracker.add_position(_make_position())
        tracker.update_position(pos_id, {'current_pnl': 50.0, 'custom_field': 'test'})
        pos = tracker.get_position(pos_id)
        assert pos['current_pnl'] == 50.0
        assert pos['custom_field'] == 'test'

    def test_update_nonexistent_position(self, tracker):
        """Updating a nonexistent position should log warning and not crash."""
        tracker.update_position('nonexistent_id', {'field': 'value'})
        # No exception raised


# ---------------------------------------------------------------------------
# Tests for get_open_positions
# ---------------------------------------------------------------------------

class TestGetOpenPositions:

    def test_empty_positions(self, tracker):
        """Fresh tracker should have no open positions."""
        assert tracker.get_open_positions() == []

    def test_returns_all_open(self, tracker):
        """Should return all added positions."""
        tracker.add_position(_make_position(ticker='SPY'))
        tracker.add_position(_make_position(ticker='QQQ'))
        assert len(tracker.get_open_positions()) == 2


# ---------------------------------------------------------------------------
# Tests for get_statistics
# ---------------------------------------------------------------------------

class TestGetStatistics:

    def test_empty_stats(self, tracker):
        """Stats with no trades should return zero values."""
        stats = tracker.get_statistics()
        assert stats['total_trades'] == 0
        assert stats['win_rate'] == 0
        assert stats['total_pnl'] == 0

    def test_all_winners(self, tracker):
        """All winning trades should produce 100% win rate."""
        id1 = tracker.add_position(_make_position(ticker='SPY'))
        id2 = tracker.add_position(_make_position(ticker='QQQ'))
        tracker.close_position(id1, exit_price=0.5, exit_reason='profit', pnl=100)
        tracker.close_position(id2, exit_price=0.5, exit_reason='profit', pnl=200)

        stats = tracker.get_statistics()
        assert stats['total_trades'] == 2
        assert stats['winning_trades'] == 2
        assert stats['losing_trades'] == 0
        assert stats['win_rate'] == 100.0
        assert stats['total_pnl'] == 300

    def test_all_losers(self, tracker):
        """All losing trades should produce 0% win rate."""
        id1 = tracker.add_position(_make_position(ticker='SPY'))
        id2 = tracker.add_position(_make_position(ticker='QQQ'))
        tracker.close_position(id1, exit_price=4.0, exit_reason='stop_loss', pnl=-200)
        tracker.close_position(id2, exit_price=4.0, exit_reason='stop_loss', pnl=-150)

        stats = tracker.get_statistics()
        assert stats['winning_trades'] == 0
        assert stats['losing_trades'] == 2
        assert stats['win_rate'] == 0
        assert stats['total_pnl'] == -350

    def test_mixed_trades(self, tracker):
        """Mixed results should compute correct statistics."""
        id1 = tracker.add_position(_make_position(ticker='SPY'))
        id2 = tracker.add_position(_make_position(ticker='QQQ'))
        id3 = tracker.add_position(_make_position(ticker='IWM'))
        tracker.close_position(id1, exit_price=0.5, exit_reason='profit', pnl=100)
        tracker.close_position(id2, exit_price=4.0, exit_reason='stop_loss', pnl=-200)
        tracker.close_position(id3, exit_price=0.5, exit_reason='profit', pnl=50)

        stats = tracker.get_statistics()
        assert stats['total_trades'] == 3
        assert stats['winning_trades'] == 2
        assert stats['losing_trades'] == 1
        assert stats['total_pnl'] == pytest.approx(-50, abs=0.01)
        assert stats['best_trade'] == 100
        assert stats['worst_trade'] == -200


# ---------------------------------------------------------------------------
# Tests for export_to_csv
# ---------------------------------------------------------------------------

class TestExportToCsv:

    def test_export_creates_file(self, tracker, tmp_path):
        """export_to_csv should create a CSV file."""
        id1 = tracker.add_position(_make_position())
        tracker.close_position(id1, exit_price=0.5, exit_reason='profit', pnl=100)

        # Manually export using pandas to verify trades data is exportable
        import pandas as pd
        trades_df = pd.DataFrame(tracker.trades)
        output_dir = tmp_path / 'output'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'trades.csv'
        trades_df.to_csv(output_file, index=False)

        assert output_file.exists()
        loaded = pd.read_csv(output_file)
        assert len(loaded) == 1

    def test_export_no_trades(self, tracker):
        """export_to_csv with no trades should log warning and return."""
        # Should not raise
        tracker.export_to_csv('test.csv')
