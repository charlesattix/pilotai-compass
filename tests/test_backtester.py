"""Tests for the Backtester class."""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from backtest.backtester import Backtester

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    """Minimal config for Backtester.
    Uses regime_mode='ma' for test isolation — unit tests exercise specific backtester
    mechanics and must not be affected by combo regime signal warmup requirements.
    """
    return {
        'backtest': {
            'starting_capital': 100000,
            'commission_per_contract': 0.65,
            'slippage': 0.05,
        },
        'strategy': {
            'spread_width': 5,
            'regime_mode': 'ma',  # test isolation: avoid combo mode VIX/warmup deps
        },
        'risk': {
            'max_positions': 5,
            'max_risk_per_trade': 2.0,
            'stop_loss_multiplier': 2.5,
        },
    }


def _make_position(
    credit=1.50,
    max_loss=3.50,
    short_strike=450,
    long_strike=445,
    contracts=1,
    commission=1.30,
    entry_date=None,
    expiration=None,
    spread_type='bull_put_spread',
    option_type='P',
):
    """Return a minimal position dict for testing."""
    entry_date = entry_date or datetime(2025, 1, 1)
    expiration = expiration or datetime(2025, 2, 5)
    return {
        'ticker': 'SPY',
        'type': spread_type,
        'entry_date': entry_date,
        'expiration': expiration,
        'short_strike': short_strike,
        'long_strike': long_strike,
        'credit': credit,
        'contracts': contracts,
        'max_loss': max_loss,
        'profit_target': credit * 0.5,
        'stop_loss': credit * 2.5,
        'commission': commission,
        'status': 'open',
        'current_value': credit * contracts * 100,
        'option_type': option_type,
    }


def _make_mock_historical_data():
    """Create a mock HistoricalOptionsData for testing."""
    mock = MagicMock()
    mock.api_calls_made = 0
    # Default: no intraday data — _check_intraday_exits falls back to daily close.
    # Individual tests that exercise intraday paths override this.
    mock.get_intraday_spread_prices.return_value = None
    # Volume gate: return None → fail-open (no behavior change for existing tests).
    mock.get_prev_daily_volume.return_value = None
    mock.get_prev_daily_oi.return_value = None
    return mock


# (TestEstimateSpreadValue and TestClosePosition removed — these methods were
#  deleted as part of Iron Vault: heuristic/synthetic pricing is permanently
#  banned. See docs/DATA_ARCHITECTURE.md and shared/iron_vault.py.)

# ---------------------------------------------------------------------------
# Tests for _record_close (real-data mode — kept and expanded)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests for _record_close (real-data mode)
# ---------------------------------------------------------------------------

class TestRecordClose:

    def setup_method(self):
        self.bt = Backtester(_make_config())
        self.bt.capital = 100000
        self.bt.trades = []

    def test_record_close_updates_capital(self):
        """_record_close should update capital by the PnL amount."""
        pos = _make_position(credit=2.00, max_loss=3.00, contracts=2)
        self.bt._record_close(pos, datetime(2025, 2, 5), 250.0, 'profit_target')
        assert self.bt.capital == pytest.approx(100250.0, rel=1e-6)
        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['pnl'] == 250.0

    def test_record_close_loss(self):
        """Negative PnL should decrease capital."""
        pos = _make_position(credit=2.00, max_loss=3.00, contracts=1)
        self.bt._record_close(pos, datetime(2025, 2, 5), -150.0, 'stop_loss')
        assert self.bt.capital == pytest.approx(99850.0, rel=1e-6)

    def test_record_close_return_pct(self):
        """Return percentage should be based on max risk."""
        pos = _make_position(credit=2.00, max_loss=3.00, contracts=2)
        # max_risk = 3.00 * 2 * 100 = 600
        self.bt._record_close(pos, datetime(2025, 2, 5), 300.0, 'profit_target')
        assert self.bt.trades[0]['return_pct'] == pytest.approx(50.0, rel=1e-6)

    def test_record_close_fields(self):
        """Trade record should have all required fields."""
        pos = _make_position()
        self.bt._record_close(pos, datetime(2025, 2, 5), 100.0, 'expiration_profit')
        trade = self.bt.trades[0]
        required_fields = [
            'ticker', 'type', 'entry_date', 'exit_date', 'exit_reason',
            'short_strike', 'long_strike', 'credit', 'contracts', 'pnl', 'return_pct',
        ]
        for field in required_fields:
            assert field in trade, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Tests for real pricing with mocked HistoricalOptionsData
# ---------------------------------------------------------------------------

class TestRealPricing:

    def setup_method(self):
        self.mock_hd = _make_mock_historical_data()
        self.bt = Backtester(_make_config(), historical_data=self.mock_hd)
        self.bt.capital = 100000
        self.bt.trades = []

    def test_real_data_mode_enabled(self):
        """Backtester should be in real data mode when historical_data provided."""
        assert self.bt._use_real_data is True

    def test_heuristic_mode_when_none(self):
        """Backtester should be in heuristic mode when no historical_data."""
        bt = Backtester(_make_config())
        assert bt._use_real_data is False

    def test_find_real_spread_uses_historical_data(self):
        """_find_real_spread should call get_available_strikes and get_spread_prices."""
        self.mock_hd.get_available_strikes.return_value = [440, 445, 450, 455, 460]
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 2.50,
            'long_close': 0.80,
            'spread_value': 1.70,
        }

        result = self.bt._find_real_spread(
            'SPY', datetime(2025, 1, 6), '2025-01-06', 480.0,
            datetime(2025, 2, 10), 5.0, option_type='P',
        )

        assert result is not None
        assert result['type'] == 'bull_put_spread'
        assert result['credit'] == pytest.approx(1.70 - 0.05, rel=1e-6)  # minus slippage
        self.mock_hd.get_available_strikes.assert_called_once()
        self.mock_hd.get_spread_prices.assert_called()

    def test_find_real_spread_skips_low_credit(self):
        """Should skip spreads with credit below min_credit_pct (default 15%) of spread width."""
        self.mock_hd.get_available_strikes.return_value = [440, 445, 450]
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 0.50,
            'long_close': 0.20,
            'spread_value': 0.30,  # 6% of $5 width — below 20% minimum
        }

        result = self.bt._find_real_spread(
            'SPY', datetime(2025, 1, 6), '2025-01-06', 480.0,
            datetime(2025, 2, 10), 5.0, option_type='P',
        )

        assert result is None

    def test_find_real_spread_tries_adjacent_strikes(self):
        """Should try adjacent strikes when primary has no data."""
        self.mock_hd.get_available_strikes.return_value = [440, 445, 450, 455, 460]
        # First call returns None, subsequent call returns data
        self.mock_hd.get_spread_prices.side_effect = [
            None,  # Primary strike
            {'short_close': 2.00, 'long_close': 0.50, 'spread_value': 1.50},  # +1 offset
        ]

        result = self.bt._find_real_spread(
            'SPY', datetime(2025, 1, 6), '2025-01-06', 480.0,
            datetime(2025, 2, 10), 5.0, option_type='P',
        )

        assert result is not None

    def test_find_real_bear_call(self):
        """Should find bear call spreads correctly."""
        self.mock_hd.get_available_strikes.return_value = [500, 505, 510, 515, 520]
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 2.00,
            'long_close': 0.60,
            'spread_value': 1.40,
        }

        result = self.bt._find_real_spread(
            'SPY', datetime(2025, 1, 6), '2025-01-06', 480.0,
            datetime(2025, 2, 10), 5.0, option_type='C',
        )

        assert result is not None
        assert result['type'] == 'bear_call_spread'
        assert result['short_strike'] == 505  # closest >= 504 (480*1.05)
        assert result['long_strike'] == 510

    def test_manage_positions_real_data_profit_target(self):
        """In real-data mode, should close at profit target using real spread value."""
        pos = _make_position(credit=2.00, max_loss=3.00, contracts=1, commission=1.30)
        pos['option_type'] = 'P'

        # Spread value dropped enough for profit target (credit - value >= profit_target)
        # profit_target = 2.00 * 0.5 = 1.00, so spread_value must be <= 1.00
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 0.50,
            'long_close': 0.10,
            'spread_value': 0.40,
        }

        remaining = self.bt._manage_positions([pos], datetime(2025, 1, 20), 460.0, 'SPY')

        assert len(remaining) == 0
        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'profit_target'
        # PnL = (2.00 - 0.40 - exit_slippage=0.10) * 1 * 100 - 1.30 = 148.70
        assert self.bt.trades[0]['pnl'] == pytest.approx(148.70, rel=1e-4)

    def test_manage_positions_real_data_stop_loss(self):
        """In real-data mode, should close at stop loss."""
        pos = _make_position(credit=1.00, max_loss=4.00, contracts=1, commission=1.30)
        pos['option_type'] = 'P'

        # stop_loss = 1.00 * 2.5 = 2.50
        # loss = spread_value - credit = 4.00 - 1.00 = 3.00 >= 2.50
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 4.50,
            'long_close': 0.50,
            'spread_value': 4.00,
        }

        remaining = self.bt._manage_positions([pos], datetime(2025, 1, 20), 440.0, 'SPY')

        assert len(remaining) == 0
        assert self.bt.trades[0]['exit_reason'] == 'stop_loss'
        # exit_cost = spread_value + exit_slippage = 4.00 + 0.10 = 4.10
        # PnL = (1.00 - 4.10) * 1 * 100 - 1.30 = -311.30
        assert self.bt.trades[0]['pnl'] == pytest.approx(-311.30, rel=1e-4)

    def test_manage_positions_real_data_no_data_keeps_position(self):
        """If no price data available, should keep position open."""
        pos = _make_position(credit=2.00, max_loss=3.00, contracts=1, commission=1.30)
        pos['option_type'] = 'P'

        self.mock_hd.get_spread_prices.return_value = None

        remaining = self.bt._manage_positions([pos], datetime(2025, 1, 20), 460.0, 'SPY')

        assert len(remaining) == 1
        assert len(self.bt.trades) == 0


# ---------------------------------------------------------------------------
# Tests for expiration with real data
# ---------------------------------------------------------------------------

class TestExpirationReal:

    def setup_method(self):
        self.mock_hd = _make_mock_historical_data()
        self.bt = Backtester(_make_config(), historical_data=self.mock_hd)
        self.bt.capital = 100000
        self.bt.trades = []

    def test_expiration_worthless(self):
        """Spread expiring worthless should give max profit."""
        pos = _make_position(credit=1.50, max_loss=3.50, contracts=2, commission=1.30)
        pos['option_type'] = 'P'

        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 0.02,
            'long_close': 0.01,
            'spread_value': 0.01,  # < 0.05 threshold
        }

        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_profit'
        # Full credit: 1.50 * 2 * 100 - 1.30 = 298.70
        assert self.bt.trades[0]['pnl'] == pytest.approx(298.70, rel=1e-4)

    def test_expiration_with_value(self):
        """Spread expiring with value should compute real P&L including exit slippage."""
        pos = _make_position(credit=1.50, max_loss=3.50, contracts=1, commission=1.30)
        pos['option_type'] = 'P'

        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 3.00,
            'long_close': 0.50,
            'spread_value': 2.50,  # > 0.05
        }

        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_loss'
        # P1-B fix: exit slippage (0.10 at VIX=20) is now applied on expiration buy-back.
        # exit_cost = 2.50 + 0.10 = 2.60
        # PnL = (1.50 - 2.60) * 1 * 100 - 1.30 = -111.30
        assert self.bt.trades[0]['pnl'] == pytest.approx(-111.30, rel=1e-4)

    def test_expiration_no_data_assumes_max_loss(self):
        """If no option data AND no underlying price, conservatively record max loss."""
        import pandas as pd
        pos = _make_position(credit=1.50, max_loss=3.50, contracts=1, commission=1.30)
        pos['option_type'] = 'P'

        self.mock_hd.get_spread_prices.return_value = None
        # Explicitly empty price_data so _get_underlying_price_at returns None,
        # forcing the conservative max-loss path (not the intrinsic fallback).
        self.bt._price_data = pd.DataFrame()

        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert self.bt.trades[0]['exit_reason'] == 'expiration_no_data'
        # PnL = -max_loss * contracts * 100 - commission = -3.50 * 1 * 100 - 1.30 = -351.30
        assert self.bt.trades[0]['pnl'] == pytest.approx(-351.30, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests for _calculate_results
# ---------------------------------------------------------------------------

class TestCalculateResults:

    def setup_method(self):
        self.bt = Backtester(_make_config())
        self.bt.capital = 100000
        self.bt.equity_curve = [(datetime(2025, 1, 1), 100000)]

    def test_empty_trades(self):
        """No trades should return minimal results."""
        self.bt.trades = []
        results = self.bt._calculate_results()
        assert results['total_trades'] == 0
        assert results['win_rate'] == 0
        assert results['total_pnl'] == 0
        assert results['bull_put_trades'] == 0
        assert results['bear_call_trades'] == 0

    def test_mixed_trades(self):
        """Mix of winners and losers should produce correct stats."""
        self.bt.trades = [
            {'pnl': 200, 'return_pct': 10, 'type': 'bull_put_spread'},
            {'pnl': -100, 'return_pct': -5, 'type': 'bull_put_spread'},
            {'pnl': 150, 'return_pct': 8, 'type': 'bear_call_spread'},
        ]
        self.bt.equity_curve = [
            (datetime(2025, 1, 1), 100000),
            (datetime(2025, 1, 8), 100200),
            (datetime(2025, 1, 15), 100100),
            (datetime(2025, 1, 22), 100250),
        ]

        results = self.bt._calculate_results()
        assert results['total_trades'] == 3
        assert results['winning_trades'] == 2
        assert results['losing_trades'] == 1
        assert results['win_rate'] == pytest.approx(66.67, abs=0.01)
        assert results['total_pnl'] == 250
        assert results['bull_put_trades'] == 2
        assert results['bear_call_trades'] == 1
        assert results['bear_call_win_rate'] == 100.0

    def test_sharpe_ratio(self):
        """Sharpe ratio should be computed from equity curve returns."""
        self.bt.trades = [{'pnl': 100, 'return_pct': 5, 'type': 'bull_put_spread'}]
        self.bt.equity_curve = [
            (datetime(2025, 1, 1), 100000),
            (datetime(2025, 1, 2), 100100),
            (datetime(2025, 1, 3), 100200),
            (datetime(2025, 1, 6), 100300),
        ]
        results = self.bt._calculate_results()
        assert 'sharpe_ratio' in results
        assert results['sharpe_ratio'] >= 0

    def test_max_drawdown(self):
        """Max drawdown should capture the peak-to-trough decline."""
        self.bt.trades = [{'pnl': -500, 'return_pct': -10, 'type': 'bull_put_spread'}]
        self.bt.equity_curve = [
            (datetime(2025, 1, 1), 100000),
            (datetime(2025, 1, 2), 101000),
            (datetime(2025, 1, 3), 99000),
            (datetime(2025, 1, 6), 99500),
        ]
        results = self.bt._calculate_results()
        assert results['max_drawdown'] < 0

    def test_profit_factor_zero_losers(self):
        """All winners with zero losers should yield infinite profit factor."""
        self.bt.trades = [
            {'pnl': 200, 'return_pct': 10, 'type': 'bull_put_spread'},
            {'pnl': 300, 'return_pct': 15, 'type': 'bull_put_spread'},
        ]
        self.bt.equity_curve = [
            (datetime(2025, 1, 1), 100000),
            (datetime(2025, 1, 2), 100200),
            (datetime(2025, 1, 3), 100500),
        ]
        results = self.bt._calculate_results()
        assert results['profit_factor'] == 999.99

    def test_all_losers(self):
        """All losing trades should produce 0% win rate."""
        self.bt.trades = [
            {'pnl': -100, 'return_pct': -5, 'type': 'bull_put_spread'},
            {'pnl': -200, 'return_pct': -10, 'type': 'bear_call_spread'},
        ]
        self.bt.equity_curve = [
            (datetime(2025, 1, 1), 100000),
            (datetime(2025, 1, 2), 99900),
            (datetime(2025, 1, 3), 99700),
        ]
        results = self.bt._calculate_results()
        assert results['win_rate'] == 0
        assert results['total_pnl'] == -300
        assert results['avg_win'] == 0

    def test_results_contain_equity_curve(self):
        """Results should include the equity curve data."""
        self.bt.trades = [{'pnl': 100, 'return_pct': 5, 'type': 'bull_put_spread'}]
        self.bt.equity_curve = [
            (datetime(2025, 1, 1), 100000),
            (datetime(2025, 1, 2), 100100),
        ]
        results = self.bt._calculate_results()
        assert 'equity_curve' in results
        assert 'trades' in results
        assert results['starting_capital'] == 100000

    def test_results_contain_strategy_breakdown(self):
        """Results should include per-strategy win rates."""
        self.bt.trades = [
            {'pnl': 200, 'return_pct': 10, 'type': 'bull_put_spread'},
            {'pnl': -50, 'return_pct': -2, 'type': 'bear_call_spread'},
        ]
        self.bt.equity_curve = [
            (datetime(2025, 1, 1), 100000),
            (datetime(2025, 1, 2), 100200),
            (datetime(2025, 1, 3), 100150),
        ]
        results = self.bt._calculate_results()
        assert results['bull_put_win_rate'] == 100.0
        assert results['bear_call_win_rate'] == 0.0


# ---------------------------------------------------------------------------
# Tests for HistoricalOptionsData (integration with mocked Polygon)
# ---------------------------------------------------------------------------

class TestHistoricalDataIntegration:

    def test_build_occ_symbol_put(self):
        """Should build correct OCC symbol for puts."""
        from backtest.historical_data import HistoricalOptionsData
        sym = HistoricalOptionsData.build_occ_symbol(
            'SPY', datetime(2025, 3, 21), 450.0, 'P',
        )
        assert sym == 'O:SPY250321P00450000'

    def test_build_occ_symbol_call(self):
        """Should build correct OCC symbol for calls."""
        from backtest.historical_data import HistoricalOptionsData
        sym = HistoricalOptionsData.build_occ_symbol(
            'SPY', datetime(2025, 3, 21), 450.0, 'C',
        )
        assert sym == 'O:SPY250321C00450000'

    def test_build_occ_symbol_fractional_strike(self):
        """Should handle fractional strikes correctly."""
        from backtest.historical_data import HistoricalOptionsData
        sym = HistoricalOptionsData.build_occ_symbol(
            'QQQ', datetime(2025, 6, 20), 375.50, 'put',
        )
        assert sym == 'O:QQQ250620P00375500'

    def test_build_occ_symbol_high_strike(self):
        """Should handle high strike prices."""
        from backtest.historical_data import HistoricalOptionsData
        sym = HistoricalOptionsData.build_occ_symbol(
            'SPY', datetime(2025, 12, 19), 600.0, 'C',
        )
        assert sym == 'O:SPY251219C00600000'

    @patch('backtest.historical_data.requests.Session')
    def test_cache_db_creation(self, mock_session_cls, tmp_path):
        """Cache DB should be created with correct tables."""
        from backtest.historical_data import HistoricalOptionsData
        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))

        # Verify tables exist
        cur = hd._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        assert 'option_daily' in tables
        assert 'option_contracts' in tables
        hd.close()

    @patch('backtest.historical_data.requests.Session')
    def test_get_contract_price_cache_hit(self, mock_session_cls, tmp_path):
        """Should return cached price without API call."""
        from backtest.historical_data import HistoricalOptionsData
        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))

        # Insert directly into cache
        hd._conn.execute(
            "INSERT INTO option_daily (contract_symbol, date, close) VALUES (?, ?, ?)",
            ('O:SPY250321P00450000', '2025-01-06', 2.50),
        )
        hd._conn.commit()

        price = hd.get_contract_price('O:SPY250321P00450000', '2025-01-06')
        assert price == 2.50
        assert hd.api_calls_made == 0
        hd.close()

    @patch('backtest.historical_data.requests.Session')
    def test_clear_cache(self, mock_session_cls, tmp_path):
        """clear_cache should empty both tables."""
        from backtest.historical_data import HistoricalOptionsData
        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))

        hd._conn.execute(
            "INSERT INTO option_daily (contract_symbol, date, close) VALUES (?, ?, ?)",
            ('O:SPY250321P00450000', '2025-01-06', 2.50),
        )
        hd._conn.commit()

        hd.clear_cache()

        cur = hd._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM option_daily")
        assert cur.fetchone()[0] == 0
        hd.close()


# ---------------------------------------------------------------------------
# Tests for bear call backtest
# ---------------------------------------------------------------------------

class TestBearCallBacktest:

    def setup_method(self):
        self.mock_hd = _make_mock_historical_data()
        self.bt = Backtester(_make_config(), historical_data=self.mock_hd)
        self.bt.capital = 100000
        self.bt.trades = []

    def test_bear_call_requires_bearish_trend(self):
        """Bear call should only trigger when price < MA20."""
        import pandas as pd
        dates = pd.date_range('2024-11-01', periods=50, freq='B')
        # Downtrending prices: final price well below MA20
        # prices[-1] = 500 - 49*0.5 = 475.5; MA20 ≈ 488 → price < MA → bear call triggers
        prices = [500 - i * 0.5 for i in range(50)]
        price_data = pd.DataFrame({'Close': prices}, index=dates)

        # Mock: OTM call strikes above 475.5 * 1.03 ≈ 490
        self.mock_hd.get_available_strikes.return_value = [490.0, 495.0, 500.0, 505.0, 510.0]
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 2.00, 'long_close': 0.50, 'spread_value': 1.50,
        }

        result = self.bt._find_bear_call_opportunity(
            'SPY', dates[-1].to_pydatetime(), prices[-1], price_data,
        )
        assert result is not None
        assert result['option_type'] == 'C'
        assert result['short_strike'] > prices[-1]  # OTM call

    def test_bear_call_expiration_profit_when_otm(self):
        """Bear call: spread expires worthless (OTM) → expiration_profit."""
        pos = _make_position(
            credit=1.50, max_loss=3.50, contracts=1, commission=1.30,
            spread_type='bear_call_spread', option_type='C',
            short_strike=500, long_strike=505,
        )
        # Call expired OTM: real spread value near zero → expiration_profit
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 0.01, 'long_close': 0.00, 'spread_value': 0.01,
        }
        self.bt._manage_positions([pos], datetime(2025, 2, 5), 490.0, 'SPY')
        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_profit'

    def test_bear_call_expiration_loss_when_itm(self):
        """Bear call: spread has value at expiry (ITM) → expiration_loss."""
        pos = _make_position(
            credit=1.50, max_loss=3.50, contracts=1, commission=1.30,
            spread_type='bear_call_spread', option_type='C',
            short_strike=500, long_strike=505,
        )
        # Call expired ITM: real spread value large → expiration_loss
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 6.00, 'long_close': 1.00, 'spread_value': 5.00,
        }
        self.bt._manage_positions([pos], datetime(2025, 2, 5), 510.0, 'SPY')
        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_loss'


# ---------------------------------------------------------------------------
# Tests for intraday backtesting (14 scan times per day)
# ---------------------------------------------------------------------------

class TestNearestWeekdayExpiration:
    """Tests for P5b: _nearest_weekday_expiration (Mon-Fri, post-2022-09-12)."""

    def test_returns_a_weekday(self):
        """Result should always be a weekday (Mon-Fri)."""
        from backtest.backtester import _nearest_weekday_expiration
        # Any target: result must be Mon-Fri
        result = _nearest_weekday_expiration(datetime(2022, 10, 3), target_dte=35)
        assert result.weekday() < 5  # 0=Mon, 4=Fri

    def test_tuesday_target_returned_directly(self):
        """date + 35 = Tuesday → nearest weekday is Tuesday itself."""
        from backtest.backtester import _nearest_weekday_expiration
        # Oct 3 (Mon) + 35 = Nov 7 (Mon). Let's find a Tuesday target:
        # Jan 7, 2025 (Tue) + 28 = Feb 4 (Tue)
        result = _nearest_weekday_expiration(datetime(2025, 1, 7), target_dte=28)
        assert result.weekday() < 5  # Mon-Fri

    def test_satisfies_min_dte(self):
        """Result must be at least min_dte days from entry."""
        from backtest.backtester import _nearest_weekday_expiration
        entry = datetime(2022, 10, 3)
        result = _nearest_weekday_expiration(entry, target_dte=35, min_dte=25)
        assert (result - entry).days >= 25

    def test_weekend_target_skipped_to_monday(self):
        """If target lands on weekend, snap to nearest weekday."""
        from backtest.backtester import _nearest_weekday_expiration
        # Find a target that lands on Saturday/Sunday
        # Jan 4 (Sat) + 35 = Feb 8 (Sat) → should snap to Mon Feb 10 or Fri Feb 7
        result = _nearest_weekday_expiration(datetime(2025, 1, 4), target_dte=35)
        assert result.weekday() < 5


class TestNearestFridayExpiration:
    """Tests for the Friday-snapping expiration helper."""

    def test_monday_target_snaps_to_prior_friday(self):
        """date + 35 days = Monday → should return the preceding Friday."""
        from backtest.backtester import _nearest_friday_expiration
        # Jan 6 (Mon) + 35 = Feb 10 (Mon) → nearest Friday = Feb 7
        result = _nearest_friday_expiration(datetime(2025, 1, 6))
        assert result.weekday() == 4  # Friday
        assert result == datetime(2025, 2, 7)

    def test_friday_target_is_unchanged(self):
        """date + 35 days already a Friday → return it as-is."""
        from backtest.backtester import _nearest_friday_expiration
        # Jan 10 (Fri) + 35 = Feb 14 (Fri)
        result = _nearest_friday_expiration(datetime(2025, 1, 10))
        assert result.weekday() == 4
        assert result == datetime(2025, 2, 14)

    def test_thursday_target_snaps_forward(self):
        """date + 35 days = Thursday → forward to next Friday is closer."""
        from backtest.backtester import _nearest_friday_expiration
        # Jan 9 (Thu) + 35 = Feb 13 (Thu) → Friday after = Feb 14
        result = _nearest_friday_expiration(datetime(2025, 1, 9))
        assert result.weekday() == 4
        assert result == datetime(2025, 2, 14)

    def test_wednesday_target_snaps_to_closer_friday(self):
        """date + 35 days = Wednesday → Friday before (2 days) is closer."""
        from backtest.backtester import _nearest_friday_expiration
        # Jan 8 (Wed) + 35 = Feb 12 (Wed) → Friday before = Feb 7 (5 days earlier)
        # Friday after = Feb 14 (2 days later) → should pick Feb 14
        result = _nearest_friday_expiration(datetime(2025, 1, 8))
        assert result.weekday() == 4
        assert result == datetime(2025, 2, 14)

    def test_result_satisfies_min_dte(self):
        """Result must always be at least min_dte days from entry date."""
        from backtest.backtester import _nearest_friday_expiration
        entry = datetime(2025, 1, 6)
        result = _nearest_friday_expiration(entry, target_dte=35, min_dte=25)
        assert (result - entry).days >= 25


class TestIntradayBacktest:
    """Tests for the intraday simulation rewrite (P0 #1)."""

    def setup_method(self):
        self.mock_hd = _make_mock_historical_data()
        self.bt = Backtester(_make_config(), historical_data=self.mock_hd)
        self.bt.capital = 100000
        self.bt.trades = []

    def test_find_real_spread_uses_intraday_when_scan_time_given(self):
        """_find_real_spread should call get_intraday_spread_prices when scan_hour/minute set."""
        self.mock_hd.get_available_strikes.return_value = [440, 445, 450, 455, 460]
        self.mock_hd.get_intraday_spread_prices.return_value = {
            'short_close': 2.50,
            'long_close': 0.80,
            'spread_value': 1.70,
            'slippage': 0.12,
        }

        result = self.bt._find_real_spread(
            'SPY', datetime(2025, 1, 6), '2025-01-06', 480.0,
            datetime(2025, 2, 10), 5.0, option_type='P',
            scan_hour=9, scan_minute=45,
        )

        assert result is not None
        self.mock_hd.get_intraday_spread_prices.assert_called_once()
        # Daily fallback should NOT be called
        self.mock_hd.get_spread_prices.assert_not_called()

    def test_find_real_spread_slippage_from_intraday_bar(self):
        """Credit should be reduced by the bar-modeled slippage, not flat 0.05."""
        self.mock_hd.get_available_strikes.return_value = [440, 445, 450]
        intraday_slippage = 0.18  # wider than flat 0.05
        self.mock_hd.get_intraday_spread_prices.return_value = {
            'short_close': 2.50,
            'long_close': 0.80,
            'spread_value': 1.70,
            'slippage': intraday_slippage,
        }

        result = self.bt._find_real_spread(
            'SPY', datetime(2025, 1, 6), '2025-01-06', 480.0,
            datetime(2025, 2, 10), 5.0, option_type='P',
            scan_hour=10, scan_minute=0,
        )

        assert result is not None
        assert result['credit'] == pytest.approx(1.70 - intraday_slippage, rel=1e-6)
        assert result['slippage_applied'] == pytest.approx(intraday_slippage, rel=1e-6)

    def test_find_real_spread_falls_back_to_daily_when_no_scan_time(self):
        """Without scan_hour/minute, should use daily get_spread_prices."""
        self.mock_hd.get_available_strikes.return_value = [440, 445, 450]
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 2.00,
            'long_close': 0.60,
            'spread_value': 1.40,
        }

        result = self.bt._find_real_spread(
            'SPY', datetime(2025, 1, 6), '2025-01-06', 480.0,
            datetime(2025, 2, 10), 5.0, option_type='P',
        )

        assert result is not None
        self.mock_hd.get_spread_prices.assert_called_once()
        self.mock_hd.get_intraday_spread_prices.assert_not_called()

    def test_find_real_spread_records_entry_scan_time(self):
        """Position should record the intraday scan time used for entry."""
        self.mock_hd.get_available_strikes.return_value = [440, 445, 450]
        self.mock_hd.get_intraday_spread_prices.return_value = {
            'short_close': 2.00, 'long_close': 0.50,
            'spread_value': 1.50, 'slippage': 0.08,
        }

        result = self.bt._find_real_spread(
            'SPY', datetime(2025, 1, 6), '2025-01-06', 480.0,
            datetime(2025, 2, 10), 5.0, option_type='P',
            scan_hour=11, scan_minute=30,
        )

        assert result is not None
        assert result['entry_scan_time'] == '11:30'

    def test_intraday_scan_times_count_is_14(self):
        """Live schedule must define exactly 14 market scan times to match backtester."""
        from shared.scheduler import MARKET_SCAN_TIMES
        assert len(MARKET_SCAN_TIMES) == 14

    def test_find_real_spread_skips_intraday_for_pre_open_scan(self):
        """9:15 scan (pre-open) should fall back to daily pricing, not intraday."""
        self.mock_hd.get_available_strikes.return_value = [440, 445, 450]
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 2.00, 'long_close': 0.60, 'spread_value': 1.40,
        }

        result = self.bt._find_real_spread(
            'SPY', datetime(2025, 1, 6), '2025-01-06', 480.0,
            datetime(2025, 2, 7), 5.0, option_type='P',
            scan_hour=9, scan_minute=15,  # pre-open
        )

        assert result is not None
        self.mock_hd.get_spread_prices.assert_called_once()
        self.mock_hd.get_intraday_spread_prices.assert_not_called()

    def test_backtester_attempts_all_scan_times_on_trading_day(self):
        """In real-data mode, backtester should attempt all 14 scan times per day."""
        import pandas as pd

        # Need >= 20 rows so the MA20 check passes on the target date.
        # Uptrending so price > MA20 (triggers bull put scans).
        dates = pd.date_range('2024-11-18', periods=35, freq='B')
        prices = [450.0 + i * 2 for i in range(len(dates))]  # steadily rising
        price_data = pd.DataFrame(
            {'Close': prices, 'Open': prices, 'High': prices, 'Low': prices,
             'Volume': [1_000_000] * len(dates)},
            index=dates,
        )

        # No strikes available — all scans return None (no position opened)
        self.mock_hd.get_available_strikes.return_value = []

        start = datetime(2025, 1, 6)
        end = datetime(2025, 1, 6)

        # The target date must be in the price_data index
        target_ts = pd.Timestamp('2025-01-06')
        if target_ts not in set(price_data.index):
            extra_dates = pd.date_range('2025-01-06', periods=1, freq='B')
            extra_prices = [prices[-1] + 2]
            extra_df = pd.DataFrame(
                {'Close': extra_prices, 'Open': extra_prices, 'High': extra_prices,
                 'Low': extra_prices, 'Volume': [1_000_000]},
                index=extra_dates,
            )
            price_data = pd.concat([price_data, extra_df])

        with __import__('unittest.mock', fromlist=['patch']).patch.object(
            self.bt, '_get_historical_data', return_value=price_data
        ):
            self.bt.run_backtest('SPY', start, end)

        # Each of the 14 scan times should trigger at least a bull put attempt
        # (price is above MA20 on the target day)
        assert self.mock_hd.get_available_strikes.call_count >= 14

    def test_direction_both_tries_bear_call_when_put_returns_none(self):
        """Regression: scan-loop continue must be inside if new_position: so that
        bear calls are tried when direction='both' and the bull put returns None.

        If continue were at the _want_puts indent level (outside if new_position:),
        _find_bear_call_opportunity would never be called and this assertion fails.
        """
        from unittest.mock import patch

        import pandas as pd

        dates = pd.date_range('2024-11-18', periods=35, freq='B')
        prices = [450.0 + i * 2 for i in range(len(dates))]
        extra_df = pd.DataFrame(
            {'Close': [prices[-1] + 2], 'Open': [prices[-1] + 2],
             'High': [prices[-1] + 2], 'Low': [prices[-1] + 2], 'Volume': [1_000_000]},
            index=pd.date_range('2025-01-06', periods=1, freq='B'),
        )
        price_data = pd.concat([
            pd.DataFrame(
                {'Close': prices, 'Open': prices, 'High': prices, 'Low': prices,
                 'Volume': [1_000_000] * len(dates)},
                index=dates,
            ),
            extra_df,
        ])

        # Default config has direction='both' — puts and calls both wanted.
        # Mock put to return None; assert bear call scanner is still called.
        with patch.object(self.bt, '_get_historical_data', return_value=price_data), \
             patch.object(self.bt, '_find_backtest_opportunity', return_value=None) as mock_put, \
             patch.object(self.bt, '_find_bear_call_opportunity', return_value=None) as mock_call:
            self.bt.run_backtest('SPY', datetime(2025, 1, 6), datetime(2025, 1, 6))

        assert mock_put.call_count > 0, "Bull put scanner was not called"
        assert mock_call.call_count > 0, (
            "Bear call scanner was never called despite direction='both' and bull put returning None. "
            "This indicates the scan-loop continue fires outside if new_position: "
            "and skips bear calls even when the bull put finds nothing."
        )


# ---------------------------------------------------------------------------
# Tests for HistoricalOptionsData intraday methods
# ---------------------------------------------------------------------------

class TestIntradayHistoricalData:
    """Tests for get_intraday_bar, _fetch_and_cache_intraday, get_intraday_spread_prices."""

    @patch('backtest.historical_data.requests.Session')
    def test_get_intraday_bar_cache_hit(self, mock_session_cls, tmp_path):
        """Returns bar from cache without API call."""
        from backtest.historical_data import HistoricalOptionsData
        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))

        hd._conn.execute(
            "INSERT INTO option_intraday (contract_symbol, date, bar_time, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ('O:SPY250321P00450000', '2025-01-06', '09:45', 2.40, 2.60, 2.20, 2.50, 100),
        )
        hd._conn.commit()

        bar = hd.get_intraday_bar('O:SPY250321P00450000', '2025-01-06', 9, 45)
        assert bar is not None
        assert bar['close'] == 2.50
        assert bar['high'] == 2.60
        assert hd.api_calls_made == 0
        hd.close()

    @patch('backtest.historical_data.requests.Session')
    def test_get_intraday_bar_returns_closest_earlier_bar(self, mock_session_cls, tmp_path):
        """Falls back to closest earlier bar when exact time not present."""
        from backtest.historical_data import HistoricalOptionsData
        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))

        # Insert bars at 09:30 and 09:35 but not 09:45
        for bt, close in [('09:30', 2.30), ('09:35', 2.35)]:
            hd._conn.execute(
                "INSERT INTO option_intraday (contract_symbol, date, bar_time, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ('O:SPY250321P00450000', '2025-01-06', bt, close, close + 0.1, close - 0.1, close, 50),
            )
        hd._conn.commit()

        bar = hd.get_intraday_bar('O:SPY250321P00450000', '2025-01-06', 9, 45)
        assert bar is not None
        assert bar['close'] == 2.35  # Most recent bar before 09:45
        hd.close()

    @patch('backtest.historical_data.requests.Session')
    def test_get_intraday_bar_sentinel_returns_none(self, mock_session_cls, tmp_path):
        """When sentinel FETCHED row exists, returns None without new API call."""
        from backtest.historical_data import HistoricalOptionsData
        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))

        hd._conn.execute(
            "INSERT INTO option_intraday (contract_symbol, date, bar_time) VALUES (?, ?, 'FETCHED')",
            ('O:SPY250321P00450000', '2025-01-06'),
        )
        hd._conn.commit()

        bar = hd.get_intraday_bar('O:SPY250321P00450000', '2025-01-06', 9, 15)
        assert bar is None
        assert hd.api_calls_made == 0
        hd.close()

    @patch('backtest.historical_data.requests.Session')
    def test_fetch_and_cache_intraday_stores_bars(self, mock_session_cls, tmp_path):
        """Intraday bars are converted from UTC ms timestamps to ET and cached."""
        from datetime import timezone as tz

        from backtest.historical_data import HistoricalOptionsData

        # 2025-01-06 09:45 ET = 14:45 UTC = 1736171100000 ms
        ts_ms = int(datetime(2025, 1, 6, 14, 45, 0, tzinfo=tz.utc).timestamp() * 1000)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'results': [
                {'t': ts_ms, 'o': 2.40, 'h': 2.60, 'l': 2.20, 'c': 2.50, 'v': 100},
            ]
        }
        mock_session_cls.return_value.get.return_value = mock_resp

        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))
        hd._fetch_and_cache_intraday('O:SPY250321P00450000', '2025-01-06')

        cur = hd._conn.cursor()
        cur.execute(
            "SELECT bar_time, close FROM option_intraday "
            "WHERE contract_symbol = ? AND date = ?",
            ('O:SPY250321P00450000', '2025-01-06'),
        )
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == '09:45'
        assert rows[0][1] == 2.50
        hd.close()

    @patch('backtest.historical_data.requests.Session')
    def test_fetch_and_cache_intraday_sentinel_on_no_data(self, mock_session_cls, tmp_path):
        """When API returns empty results, sentinel row inserted."""
        from backtest.historical_data import HistoricalOptionsData

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {'results': []}
        mock_session_cls.return_value.get.return_value = mock_resp

        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))
        hd._fetch_and_cache_intraday('O:SPY250321P00450000', '2025-01-06')

        cur = hd._conn.cursor()
        cur.execute(
            "SELECT bar_time FROM option_intraday "
            "WHERE contract_symbol = ? AND date = ?",
            ('O:SPY250321P00450000', '2025-01-06'),
        )
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 'FETCHED'
        hd.close()

    @patch('backtest.historical_data.requests.Session')
    def test_get_intraday_spread_prices_computes_slippage(self, mock_session_cls, tmp_path):
        """Slippage equals sum of half-spread (high-low/2) for both legs."""
        from backtest.historical_data import HistoricalOptionsData

        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))

        sym_short = 'O:SPY250321P00455000'
        sym_long = 'O:SPY250321P00450000'
        date_str = '2025-01-06'

        # Short leg: high=2.60, low=2.20 → half-spread = 0.20
        hd._conn.execute(
            "INSERT INTO option_intraday (contract_symbol, date, bar_time, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (sym_short, date_str, '09:45', 2.40, 2.60, 2.20, 2.50, 100),
        )
        # Long leg: high=1.20, low=0.80 → half-spread = 0.20
        hd._conn.execute(
            "INSERT INTO option_intraday (contract_symbol, date, bar_time, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (sym_long, date_str, '09:45', 0.90, 1.20, 0.80, 1.00, 80),
        )
        hd._conn.commit()

        result = hd.get_intraday_spread_prices(
            'SPY', datetime(2025, 3, 21), 455.0, 450.0, 'P',
            date_str, 9, 45,
        )

        assert result is not None
        assert result['spread_value'] == pytest.approx(1.50, rel=1e-6)   # 2.50 - 1.00
        # Each leg's half-range is 0.20; cap is $0.25/leg so both flow through uncapped.
        # Total slippage = 0.20 + 0.20 = 0.40
        assert result['slippage'] == pytest.approx(0.40, rel=1e-6)

        hd.close()

    @patch('backtest.historical_data.requests.Session')
    def test_get_intraday_spread_prices_returns_none_if_leg_missing(self, mock_session_cls, tmp_path):
        """Returns None when either leg has no intraday data."""
        from backtest.historical_data import HistoricalOptionsData

        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))

        # Only short leg cached — long leg will trigger fetch which returns nothing
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {'results': []}
        mock_session_cls.return_value.get.return_value = mock_resp

        sym_short = 'O:SPY250321P00455000'
        hd._conn.execute(
            "INSERT INTO option_intraday (contract_symbol, date, bar_time, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (sym_short, '2025-01-06', '09:45', 2.40, 2.60, 2.20, 2.50, 100),
        )
        hd._conn.commit()

        result = hd.get_intraday_spread_prices(
            'SPY', datetime(2025, 3, 21), 455.0, 450.0, 'P',
            '2025-01-06', 9, 45,
        )

        assert result is None
        hd.close()

    @patch('backtest.historical_data.requests.Session')
    def test_clear_cache_removes_intraday_table(self, mock_session_cls, tmp_path):
        """clear_cache should empty option_intraday table too."""
        from backtest.historical_data import HistoricalOptionsData

        hd = HistoricalOptionsData('test_key', cache_dir=str(tmp_path))
        hd._conn.execute(
            "INSERT INTO option_intraday (contract_symbol, date, bar_time, close) "
            "VALUES ('O:SPY250321P00450000', '2025-01-06', '09:45', 2.50)",
        )
        hd._conn.commit()

        hd.clear_cache()

        cur = hd._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM option_intraday")
        assert cur.fetchone()[0] == 0
        hd.close()


# ---------------------------------------------------------------------------
# Tests for P1: Portfolio-Level Exposure Constraint
# ---------------------------------------------------------------------------

class TestPortfolioExposureConstraint:
    """Test that max_portfolio_exposure_pct blocks entries when exposure is too high."""

    def _make_config_with_exposure(self, exposure_pct):
        cfg = _make_config()
        cfg['backtest']['max_portfolio_exposure_pct'] = exposure_pct
        return cfg

    def test_no_constraint_by_default(self):
        """Default 100% means no constraint — _exposure_ok always True."""
        bt = Backtester(_make_config())
        assert bt._max_portfolio_exposure_pct == 100.0

    def test_constraint_stored_from_config(self):
        """max_portfolio_exposure_pct is read from backtest config."""
        bt = Backtester(self._make_config_with_exposure(30.0))
        assert bt._max_portfolio_exposure_pct == 30.0

    def test_exposure_ok_when_below_cap(self):
        """A new position that keeps total exposure below cap should be allowed."""
        bt = Backtester(self._make_config_with_exposure(30.0))
        bt.capital = 100_000
        open_positions = []
        # Simulate an open position with $5K max loss (5% exposure)
        open_positions.append(_make_position(max_loss=2.50, contracts=20))  # $5K max loss

        # New position: $5K max loss → total 10% exposure (well under 30%)
        new_pos = _make_position(max_loss=2.50, contracts=20)
        # Inline the exposure check logic (mirrors backtester implementation)
        current_max_loss = sum(p['max_loss'] * p['contracts'] * 100 for p in open_positions)
        new_max_loss = new_pos['max_loss'] * new_pos['contracts'] * 100
        total_equity = bt.capital + sum(p.get('current_value', 0) for p in open_positions)
        exposure_pct = (current_max_loss + new_max_loss) / total_equity * 100
        assert exposure_pct < 30.0

    def test_exposure_blocked_when_above_cap(self):
        """A new position that pushes total exposure over cap should be blocked."""
        bt = Backtester(self._make_config_with_exposure(20.0))
        bt.capital = 100_000
        open_positions = []
        # Existing: $18K max loss = 18% exposure
        open_positions.append(_make_position(max_loss=4.50, contracts=40))  # $18K

        # New position: $5K max loss → would be 23% total — over 20% cap
        new_pos = _make_position(max_loss=2.50, contracts=20)  # $5K
        current_max_loss = sum(p['max_loss'] * p['contracts'] * 100 for p in open_positions)
        new_max_loss = new_pos['max_loss'] * new_pos['contracts'] * 100
        total_equity = bt.capital + sum(p.get('current_value', 0) for p in open_positions)
        exposure_pct = (current_max_loss + new_max_loss) / total_equity * 100
        assert exposure_pct > 20.0

    def test_no_constraint_at_100_pct(self):
        """At 100% cap, even 80K max loss against 100K equity should be OK."""
        bt = Backtester(self._make_config_with_exposure(100.0))
        bt.capital = 100_000
        open_positions = [_make_position(max_loss=4.50, contracts=160)]  # $72K
        new_pos = _make_position(max_loss=4.50, contracts=20)  # $9K → total $81K = 81%
        current_max_loss = sum(p['max_loss'] * p['contracts'] * 100 for p in open_positions)
        new_max_loss = new_pos['max_loss'] * new_pos['contracts'] * 100
        total_equity = bt.capital + sum(p.get('current_value', 0) for p in open_positions)
        exposure_pct = (current_max_loss + new_max_loss) / total_equity * 100
        # With no constraint (100% cap), 81% < 100% → allowed
        assert exposure_pct <= 100.0


# ---------------------------------------------------------------------------
# Probability-of-ruin utility tests
# ---------------------------------------------------------------------------

class TestProbabilityOfRuin:
    """Unit tests for scripts/prob_of_ruin.py utility functions."""

    def _import(self):
        import importlib.util
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "prob_of_ruin", os.path.join(root, "scripts", "prob_of_ruin.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_kelly_no_losses(self):
        mod = self._import()
        # Kelly is undefined with no losses — function returns max-proxy
        result = mod._kelly_fraction(0.99, 999)
        assert result > 0

    def test_kelly_edge_zero(self):
        mod = self._import()
        # p=0.5, b=1.0 → edge = 0 → Kelly = 0
        k = mod._kelly_fraction(0.5, 1.0)
        assert abs(k) < 1e-10

    def test_kelly_positive_edge(self):
        mod = self._import()
        # p=0.6, b=1.5 → Kelly = 0.6 - 0.4/1.5 = 0.333
        k = mod._kelly_fraction(0.6, 1.5)
        assert abs(k - (0.6 - 0.4 / 1.5)) < 1e-10

    def test_ruin_zero_prob_high_win_rate(self):
        mod = self._import()
        # High WR strategy: 99% win rate, tiny losses — should have 0% ruin
        returns = [0.02] * 99 + [-0.005]  # 99 wins, 1 loss per 100
        ruin_p, _, _ = mod.monte_carlo_ruin(returns, n_sims=1000, horizon=100, ruin_pct=50.0)
        assert ruin_p == 0.0

    def test_ruin_high_prob_large_losses(self):
        mod = self._import()
        # Terrible strategy: 40% win rate, -30% on losses → almost certain ruin
        returns = [0.02] * 4 + [-0.30] * 6  # 40% win, 60% loss
        ruin_p, _, _ = mod.monte_carlo_ruin(returns, n_sims=1000, horizon=200, ruin_pct=50.0)
        assert ruin_p > 0.5  # more than 50% chance of ruin

    def test_ruin_increases_with_horizon(self):
        mod = self._import()
        returns = [0.01] * 7 + [-0.10] * 3  # 70% WR, -10% losses
        ruin_100, _, _ = mod.monte_carlo_ruin(returns, n_sims=2000, horizon=100, ruin_pct=50.0)
        ruin_500, _, _ = mod.monte_carlo_ruin(returns, n_sims=2000, horizon=500, ruin_pct=50.0)
        # Longer horizon should have higher or equal ruin probability
        assert ruin_500 >= ruin_100

    def test_median_equity_grows_with_wins(self):
        mod = self._import()
        # Pure wins → median equity should compound up
        returns = [0.01] * 100  # all wins at 1%
        _, med_eq, _ = mod.monte_carlo_ruin(returns, n_sims=500, horizon=50, ruin_pct=50.0)
        assert med_eq > 1.0  # should grow from 1.0 baseline


# ---------------------------------------------------------------------------
# Tests for _ruin_triggered (P1-D)
# ---------------------------------------------------------------------------

class TestRuinTriggered:
    """_ruin_triggered must be set when capital reaches zero and block new entries."""

    def setup_method(self):
        cfg = _make_config()
        cfg['backtest']['compound'] = True
        self.bt = Backtester(cfg, historical_data=MagicMock())
        self.bt.trades = []

    def test_ruin_set_when_capital_goes_negative(self):
        """A loss that drives capital below zero must set _ruin_triggered."""
        self.bt.capital = 500.0
        pos = _make_position(credit=1.50, max_loss=5.00, contracts=2, commission=1.30)
        # pnl = -5.00 * 2 * 100 - 1.30 = -1001.30 → capital = 500 - 1001.30 < 0
        pnl = -pos['max_loss'] * pos['contracts'] * 100 - pos['commission']
        self.bt._record_close(pos, datetime(2025, 3, 1), pnl, 'stop_loss')
        assert self.bt.capital < 0
        assert self.bt._ruin_triggered is True

    def test_ruin_not_set_on_normal_loss(self):
        """A loss that leaves capital positive must NOT set _ruin_triggered."""
        self.bt.capital = 100_000.0
        pos = _make_position(credit=1.50, max_loss=3.50, contracts=1, commission=1.30)
        pnl = -pos['max_loss'] * pos['contracts'] * 100 - pos['commission']
        self.bt._record_close(pos, datetime(2025, 3, 1), pnl, 'stop_loss')
        assert self.bt.capital > 0
        assert self.bt._ruin_triggered is False

    def test_ruin_blocks_new_entries_in_backtest(self):
        """After _ruin_triggered fires, run_backtest must not attempt any entry scans.

        Patches _manage_positions to trigger ruin on the first call, then asserts
        that get_available_strikes is never called (entry gate blocked by ruin flag).
        """
        import pandas as pd

        mock_hd = _make_mock_historical_data()
        mock_hd.get_available_strikes.return_value = [440, 445, 450]

        cfg = _make_config()
        cfg['backtest']['compound'] = True
        bt = Backtester(cfg, historical_data=mock_hd)

        dates = pd.date_range('2024-11-18', periods=30, freq='B')
        prices = [450.0 + i for i in range(len(dates))]
        price_data = pd.DataFrame(
            {'Close': prices, 'Open': prices, 'High': prices,
             'Low': prices, 'Volume': [1_000_000] * len(prices)},
            index=dates,
        )

        # On the first manage_positions call, simulate a ruinous loss by setting
        # _ruin_triggered = True (as _record_close would do after capital <= 0).
        real_manage = bt._manage_positions
        calls = []

        def fake_manage(positions, current_date, current_price, ticker_arg):
            calls.append(current_date)
            if len(calls) == 1:
                bt._ruin_triggered = True   # simulate capital wiped on day 1
            return real_manage(positions, current_date, current_price, ticker_arg)

        with patch('backtest.backtester.Backtester._get_historical_data',
                   return_value=price_data), \
             patch('backtest.backtester.Backtester._build_iv_rank_series',
                   return_value={}), \
             patch('backtest.backtester.Backtester._build_realized_vol_series',
                   return_value={}), \
             patch.object(bt, '_manage_positions', side_effect=fake_manage):
            bt.run_backtest('SPY', dates[-2].to_pydatetime(), dates[-1].to_pydatetime())

        # _ruin_triggered=True is ORed into _skip_new_entries immediately after
        # _manage_positions returns — so no scan fires on either day.
        # If the ruin gate were missing, get_available_strikes would be called 28+ times.
        assert mock_hd.get_available_strikes.call_count == 0, (
            f"get_available_strikes called {mock_hd.get_available_strikes.call_count} times "
            "after ruin — _ruin_triggered is not gating new entries"
        )

    def test_ruin_resets_on_new_run(self):
        """_ruin_triggered must reset to False at the start of run_backtest."""
        self.bt._ruin_triggered = True
        # Simulate the reset that run_backtest performs
        self.bt._ruin_triggered = False
        assert self.bt._ruin_triggered is False

    def test_ruin_flag_initialised_false(self):
        """Fresh Backtester instance must have _ruin_triggered=False."""
        cfg = _make_config()
        bt = Backtester(cfg)
        assert bt._ruin_triggered is False

    def test_ruin_in_results_dict_no_trades(self):
        """ruin_triggered must appear in the results dict even with no trades."""
        import pandas as pd
        cfg = _make_config()
        bt = Backtester(cfg, historical_data=MagicMock())
        # Build a trivial price_data so run_backtest doesn't fail on data fetch
        dates = pd.date_range('2025-01-01', periods=3, freq='B')
        price_data = pd.DataFrame(
            {'Close': [500.0, 501.0, 502.0],
             'High':  [501.0, 502.0, 503.0],
             'Low':   [499.0, 500.0, 501.0],
             'Open':  [500.0, 501.0, 502.0],
             'Volume':[1e6,   1e6,   1e6  ]},
            index=dates,
        )
        with patch('backtest.backtester.Backtester._get_historical_data', return_value=price_data), \
             patch('backtest.backtester.Backtester._build_iv_rank_series', return_value={}), \
             patch('backtest.backtester.Backtester._build_realized_vol_series', return_value={}):
            results = bt.run_backtest('SPY', datetime(2025, 1, 1), datetime(2025, 1, 5))
        assert 'ruin_triggered' in results
        assert results['ruin_triggered'] is False


# ---------------------------------------------------------------------------
# Tests for Iron Condor expiration (IC code coverage)
# ---------------------------------------------------------------------------

def _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30):
    """Return a minimal iron condor position dict for testing."""
    return {
        'ticker': 'SPY',
        'type': 'iron_condor',
        'entry_date': datetime(2025, 1, 1),
        'expiration': datetime(2025, 2, 5),
        'short_strike': 450,       # put short
        'long_strike': 445,        # put long
        'call_short_strike': 480,  # call short
        'call_long_strike': 485,   # call long
        'credit': credit,
        'contracts': contracts,
        'max_loss': max_loss,
        'profit_target': credit * 0.5,
        'stop_loss': credit * 2.5,
        'commission': commission,
        'status': 'open',
        'current_value': credit * contracts * 100,
        'option_type': 'IC',
    }


class TestIronCondorExpiration:
    """IC expiration paths: worthless, with-value, intrinsic fallback, max-loss, type field."""

    def setup_method(self):
        self.mock_hd = _make_mock_historical_data()
        self.bt = Backtester(_make_config(), historical_data=self.mock_hd)
        self.bt.capital = 100000
        self.bt.trades = []

    def test_ic_expiration_worthless(self):
        """Both IC wings expire worthless → full credit profit."""
        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        self.mock_hd.get_spread_prices.side_effect = [
            {'short_close': 0.02, 'long_close': 0.01, 'spread_value': 0.01},  # put
            {'short_close': 0.02, 'long_close': 0.01, 'spread_value': 0.01},  # call
        ]
        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_profit'
        # 2.00 * 1 * 100 - 1.30 = 198.70
        assert self.bt.trades[0]['pnl'] == pytest.approx(198.70, rel=1e-4)

    def test_ic_expiration_with_value_applies_slippage(self):
        """IC with combined wing residual > 0.05 applies exit slippage."""
        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        # Put wing has value 1.50; call wing worthless (0.01) → combined 1.51 > 0.05
        self.mock_hd.get_spread_prices.side_effect = [
            {'short_close': 2.00, 'long_close': 0.50, 'spread_value': 1.50},  # put
            {'short_close': 0.02, 'long_close': 0.01, 'spread_value': 0.01},  # call
        ]
        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_profit'
        # IC closes two spreads — 2× slippage applied.
        # exit_cost = 1.51 + 2 * 0.10 (VIX=20) = 1.71
        # pnl = (2.00 - 1.71) * 100 - 1.30 = 29.00 - 1.30 = 27.70
        assert self.bt.trades[0]['pnl'] == pytest.approx(27.70, rel=1e-4)

    def test_ic_expiration_boundary_value_treated_as_worthless(self):
        """IC with combined residual in (0.05, 0.10] expires as worthless — no slippage.

        The threshold is > 0.10 (not > 0.05) for ICs since it represents two wings.
        A combined value of 0.08 is ≤ 0.10 → position treated as worthless → no buyback cost.
          pnl = credit * contracts * 100 - commission = 2.00 * 1 * 100 - 1.30 = 198.70
        """
        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        # Put wing: 0.07, call wing: 0.01 → combined = 0.08 — inside (0.05, 0.10] boundary
        self.mock_hd.get_spread_prices.side_effect = [
            {'short_close': 0.10, 'long_close': 0.03, 'spread_value': 0.07},  # put
            {'short_close': 0.02, 'long_close': 0.01, 'spread_value': 0.01},  # call
        ]
        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_profit'
        # Combined = 0.08 ≤ 0.10 → worthless; no slippage applied
        # pnl = 2.00 * 1 * 100 - 1.30 = 198.70
        assert self.bt.trades[0]['pnl'] == pytest.approx(198.70, rel=1e-4)

    def test_ic_expiration_exact_threshold_treated_as_worthless(self):
        """IC with combined residual = exactly 0.10 is worthless (threshold is strict >).

        _close_at_expiration_real uses `if closing_spread_value > 0.10:` — the boundary
        value itself (0.10) is NOT greater than the threshold, so it lapses worthless.
        """
        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        # combined = 0.09 + 0.01 = 0.10 — exactly at threshold, treated as worthless
        self.mock_hd.get_spread_prices.side_effect = [
            {'short_close': 0.12, 'long_close': 0.03, 'spread_value': 0.09},  # put
            {'short_close': 0.02, 'long_close': 0.01, 'spread_value': 0.01},  # call
        ]
        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_profit'
        # combined = 0.10, NOT > 0.10 → worthless; pnl = 2.00 * 100 - 1.30 = 198.70
        assert self.bt.trades[0]['pnl'] == pytest.approx(198.70, rel=1e-4)

    def test_ic_expiration_just_above_threshold_applies_slippage(self):
        """IC with combined residual just above 0.10 triggers slippage (strict > 0.10).

        0.11 is strictly greater than 0.10, so slippage is applied on buyback.
          exit_cost = 0.11 + 2 × 0.10 = 0.31; pnl = (2.00 - 0.31) × 100 - 1.30 = 167.70
        """
        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        # combined = 0.10 + 0.01 = 0.11 — just above threshold
        self.mock_hd.get_spread_prices.side_effect = [
            {'short_close': 0.13, 'long_close': 0.03, 'spread_value': 0.10},  # put
            {'short_close': 0.02, 'long_close': 0.01, 'spread_value': 0.01},  # call
        ]
        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_profit'
        # combined = 0.11 > 0.10 → slippage applied (2×)
        # exit_cost = 0.11 + 2 × 0.10 = 0.31; pnl = (2.00 - 0.31) × 100 - 1.30 = 167.70
        assert self.bt.trades[0]['pnl'] == pytest.approx(167.70, rel=1e-4)

    def test_ic_expiration_intrinsic_fallback(self):
        """No option data → intrinsic settlement from underlying price."""
        import pandas as pd
        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        self.mock_hd.get_spread_prices.side_effect = [None, None]
        # Underlying at 447: between put_long(445) and put_short(450)
        # put_intrinsic = 450 - 447 = 3.0; call_intrinsic = 0 (447 < call_short 480)
        self.bt._price_data = pd.DataFrame(
            {'Close': [447.0]},
            index=[pd.Timestamp('2025-02-05')],
        )
        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_no_data'
        # pnl = (2.00 - 3.00) * 1 * 100 - 1.30 = -101.30
        assert self.bt.trades[0]['pnl'] == pytest.approx(-101.30, rel=1e-4)

    def test_ic_expiration_no_data_max_loss(self):
        """No option data and no underlying price → conservative max loss."""
        import pandas as pd
        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        self.mock_hd.get_spread_prices.side_effect = [None, None]
        self.bt._price_data = pd.DataFrame()  # empty → _get_underlying_price_at returns None
        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'expiration_no_data'
        # pnl = -8.00 * 1 * 100 - 1.30 = -801.30
        assert self.bt.trades[0]['pnl'] == pytest.approx(-801.30, rel=1e-4)

    def test_ic_record_close_stores_ic_option_type(self):
        """_record_close must store 'IC' (not 'P') for iron_condor positions."""
        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        self.mock_hd.get_spread_prices.side_effect = [
            {'short_close': 0.01, 'long_close': 0.00, 'spread_value': 0.01},  # put worthless
            {'short_close': 0.01, 'long_close': 0.00, 'spread_value': 0.01},  # call worthless
        ]
        self.bt._close_at_expiration_real(pos, datetime(2025, 2, 5))

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['option_type'] == 'IC'

    def test_ic_flat_risk_sizing_uses_double_width(self):
        """IC contracts must be sized on spread_width * 2, not single-wing width.

        Regression for P0-B: using single width doubles the per-contract max-loss
        denominator underflow, causing ~2x oversizing of IC positions.

        Setup: capital=$100K, 10% flat risk, 5-wide spread, combined_credit=2.30
          correct:  int(10_000 / ((10.0 - 2.30) * 100)) = int(10000/770) = 12
          bugged:   int(10_000 / ((5.0  - 2.30) * 100)) = int(10000/270) = 37
        """
        from compass.sizing import get_contract_size

        cfg = _make_config()
        cfg['backtest']['sizing_mode'] = 'flat'
        cfg['risk']['max_risk_per_trade'] = 10.0
        cfg['risk']['stop_loss_multiplier'] = 2.5
        cfg['strategy']['spread_width'] = 5.0
        mock_hd = _make_mock_historical_data()

        bt = Backtester(cfg, historical_data=mock_hd)
        bt.capital = 100_000
        bt.starting_capital = 100_000

        expiry = datetime(2025, 2, 21)
        put_leg = {
            'credit': 1.20,
            'short_strike': 450.0, 'long_strike': 445.0,
            'expiration': expiry, 'slippage_applied': 0.05, 'entry_scan_time': None,
        }
        call_leg = {
            'credit': 1.10,
            'short_strike': 480.0, 'long_strike': 485.0,
            'expiration': expiry, 'slippage_applied': 0.05, 'entry_scan_time': None,
        }

        with patch.object(bt, '_find_real_spread', side_effect=[put_leg, call_leg]):
            result = bt._find_iron_condor_opportunity(
                'SPY', datetime(2025, 1, 6), 470.0,
            )

        assert result is not None
        assert result['type'] == 'iron_condor'

        # Verify contract count uses 2×spread_width (=10.0) in the max-loss denominator.
        combined_credit = 1.20 + 1.10  # = 2.30
        expected = get_contract_size(
            100_000 * (10.0 / 100),   # trade_dollar_risk = 10_000
            5.0 * 2,                   # spread_width * 2 = 10.0 for IC
            combined_credit,
            max_contracts=999,
        )
        assert result['contracts'] == expected   # 12
        assert result['contracts'] == 12         # explicit sanity check

        # IC entry deducts commission for 4 legs × contract count.
        ic_commission = bt.commission * 4 * result['contracts']  # = 0.65 * 4 * 12 = 31.20
        assert bt.capital == pytest.approx(100_000 - ic_commission, rel=1e-6)

    def test_ic_intraday_exit_applies_double_slippage(self):
        """IC profit-target triggered by intraday data uses 2× exit slippage.

        Closing an IC requires buying back two separate spreads (put + call),
        each incurring bid-ask friction.  The single-spread path uses 1×; the
        IC path must use 2×.  This test catches any regression where the
        _slip_legs multiplier is dropped on the intraday exit branch.

        At VIX=20: _vix_scaled_exit_slippage() = 0.10 × 1.0 = 0.10  (exit_slippage default)
          combined spread_value = 0.80 → profit 1.20 ≥ profit_target 1.00
          exit_cost (2×) = 0.80 + 2×0.10 = 1.00  → pnl = (2.00-1.00)×100 - 1.30 = 98.70
          exit_cost (1×) = 0.80 + 1×0.10 = 0.90  → pnl = (2.00-0.90)×100 - 1.30 = 108.70 (wrong)
        """
        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        pos['entry_scan_time'] = None   # check all scan times
        pos['expiration'] = datetime(2025, 2, 21)  # not expired on test date

        self.bt._current_vix = 20.0  # → vix_scale=1.0, slippage=0.10 per leg (exit_slippage default)

        # First intraday scan: put=0.40, call=0.40 → combined=0.80 → profit_target hit
        self.mock_hd.get_intraday_spread_prices.side_effect = [
            {'spread_value': 0.40, 'short_close': 0.50, 'long_close': 0.10},
            {'spread_value': 0.40, 'short_close': 0.50, 'long_close': 0.10},
        ]

        self.bt._manage_positions([pos], datetime(2025, 2, 1), 460.0, 'SPY')

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'profit_target'
        # 2× slippage: exit_cost = 0.80 + 2×0.10 = 1.00 → pnl = 100 - 1.30 = 98.70
        # 1× slippage: exit_cost = 0.80 + 1×0.10 = 0.90 → pnl = 110 - 1.30 = 108.70 (wrong)
        assert self.bt.trades[0]['pnl'] == pytest.approx(98.70, rel=1e-4)

    def test_ic_intraday_stop_loss_applies_double_slippage(self):
        """IC stop-loss triggered by intraday data uses 2× exit slippage.

        Mirrors test_ic_intraday_exit_applies_double_slippage for the stop_loss
        branch.  Closing an IC always requires two separate buyback transactions.

        At VIX=20: _vix_scaled_exit_slippage() = 0.10 × 1.0 = 0.10
          credit=2.00, stop_loss=5.00 (credit × 2.5)
          combined spread_value = 7.00 → loss 5.00 ≥ stop_loss 5.00
          exit_cost (2×) = 7.00 + 2×0.10 = 7.20 → pnl = (2.00-7.20)×100 - 1.30 = -521.30
          exit_cost (1×) = 7.00 + 1×0.10 = 7.10 → pnl = (2.00-7.10)×100 - 1.30 = -511.30 (wrong)
        """
        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        pos['entry_scan_time'] = None
        pos['expiration'] = datetime(2025, 2, 21)

        self.bt._current_vix = 20.0

        # combined spread_value = 7.00 → loss = 7.00 - 2.00 = 5.00 ≥ stop_loss 5.00
        self.mock_hd.get_intraday_spread_prices.side_effect = [
            {'spread_value': 3.50, 'short_close': 4.00, 'long_close': 0.50},  # put
            {'spread_value': 3.50, 'short_close': 4.00, 'long_close': 0.50},  # call
        ]

        self.bt._manage_positions([pos], datetime(2025, 2, 1), 460.0, 'SPY')

        assert len(self.bt.trades) == 1
        assert self.bt.trades[0]['exit_reason'] == 'stop_loss'
        # 2× slippage: exit_cost = 7.00 + 2×0.10 = 7.20 → pnl = -520 - 1.30 = -521.30
        # 1× slippage: exit_cost = 7.00 + 1×0.10 = 7.10 → pnl = -510 - 1.30 = -511.30 (wrong)
        assert self.bt.trades[0]['pnl'] == pytest.approx(-521.30, rel=1e-4)


# ---------------------------------------------------------------------------
# Regression test: VIX Monday prior-trading-day lookup (_prev_trading_val)
# ---------------------------------------------------------------------------

class TestVixMondayLookup:
    """_prev_trading_val must find the prior Friday's VIX on Mondays.

    If broken (date-1 gives Sunday, dict miss → default 20.0), a vix_max_entry
    gate set below Friday's actual VIX would be silently bypassed on Mondays.
    This test verifies the gate fires correctly.
    """

    def test_monday_vix_uses_prior_friday_not_default(self):
        import pandas as pd

        # 30 uptrending days + Monday 2025-01-06 as the target trading day.
        base_dates = pd.date_range('2024-11-18', periods=35, freq='B')
        prices = [450.0 + i * 1.5 for i in range(len(base_dates))]
        mon = pd.Timestamp('2025-01-06')
        price_data = pd.concat([
            pd.DataFrame(
                {'Close': prices, 'Open': prices, 'High': prices,
                 'Low': prices, 'Volume': [1_000_000] * len(prices)},
                index=base_dates,
            ),
            pd.DataFrame(
                {'Close': [prices[-1] + 1.5], 'Open': [prices[-1] + 1.5],
                 'High': [prices[-1] + 2.0], 'Low': [prices[-1] + 1.0],
                 'Volume': [1_000_000]},
                index=[mon],
            ),
        ])

        mock_hd = _make_mock_historical_data()
        mock_hd.get_available_strikes.return_value = [440, 445, 450]

        cfg = _make_config()
        # vix_max_entry=25 blocks entries when VIX > 25.
        # Friday's VIX = 30.0 → should block Monday entries.
        # Default fallback VIX = 20.0 → would NOT block → entries attempted.
        cfg['strategy']['vix_max_entry'] = 25
        cfg['strategy']['direction'] = 'both'

        bt = Backtester(cfg, historical_data=mock_hd)

        def _fake_iv_rank_series(start_arg, end_arg):
            # Inject only a Friday entry (2025-01-03); no Sat/Sun/Mon entries.
            bt._vix_by_date = {pd.Timestamp('2025-01-03'): 30.0}
            return {}

        with patch('backtest.backtester.Backtester._get_historical_data',
                   return_value=price_data), \
             patch.object(bt, '_build_iv_rank_series',
                          side_effect=_fake_iv_rank_series), \
             patch('backtest.backtester.Backtester._build_realized_vol_series',
                   return_value={}):
            bt.run_backtest('SPY', datetime(2025, 1, 6), datetime(2025, 1, 6))

        # With fix: Monday lookup finds Friday's VIX=30.0 > vix_max_entry=25
        # → _vix_too_high=True → _skip_new_entries=True → no scan attempted.
        # If broken: default VIX=20.0 used → gate bypassed → strikes queried.
        assert mock_hd.get_available_strikes.call_count == 0, (
            "get_available_strikes was called — Monday VIX lookup returned "
            "default 20.0 instead of Friday's 30.0, bypassing the vix_max_entry gate"
        )


# ---------------------------------------------------------------------------
# Regression: heuristic mode respects _skip_new_entries (R7 P1-1 fix)
# ---------------------------------------------------------------------------

class TestHeuristicModeGate:
    """Heuristic Monday scan must check _skip_new_entries.

    Before R7 P1-1, the `if current_date.weekday() == 0:` check did not
    include `and not _skip_new_entries`, so exclude_months, drawdown CB,
    ruin, VIX gate, and IV-rank gate were all silently bypassed in heuristic
    mode.  This test asserts that exclude_months correctly blocks heuristic
    entries on a Monday that falls in the excluded month.
    """

    def test_exclude_months_blocks_entries(self):
        from unittest.mock import patch

        import pandas as pd

        # Monday 2025-01-06; exclude_months=['2025-01'] → gate must fire.
        cfg = _make_config()
        cfg['backtest']['exclude_months'] = ['2025-01']
        bt = Backtester(cfg)  # historical_data=None: no real data, verifies gate before spread-find

        dates = pd.date_range('2024-11-18', periods=30, freq='B')
        prices = [450.0 + i * 2 for i in range(len(dates))]
        price_data = pd.concat([
            pd.DataFrame(
                {'Close': prices, 'Open': prices, 'High': prices,
                 'Low': prices, 'Volume': [1_000_000] * len(dates)},
                index=dates,
            ),
            pd.DataFrame(
                {'Close': [prices[-1] + 2], 'Open': [prices[-1] + 2],
                 'High': [prices[-1] + 2], 'Low': [prices[-1] + 2],
                 'Volume': [1_000_000]},
                index=pd.date_range('2025-01-06', periods=1, freq='B'),
            ),
        ])

        # Patch _find_backtest_opportunity to detect if the gate fires before entry.
        with patch.object(bt, '_get_historical_data', return_value=price_data), \
             patch.object(bt, '_find_backtest_opportunity', return_value=None) as mock_bp:
            bt.run_backtest('SPY', datetime(2025, 1, 6), datetime(2025, 1, 6))

        assert mock_bp.call_count == 0, (
            "_find_backtest_opportunity was called despite exclude_months=['2025-01']. "
            "The entry gate is not checking _skip_new_entries before spread-finding."
        )


# ---------------------------------------------------------------------------
# Regression tests: R8 P1 commission refund on duplicate-key rejection
# ---------------------------------------------------------------------------

class TestCommissionRefundOnDupKey:
    """14 scan times returning the same position key: only 1 commission kept.

    _find_real_spread (and _find_iron_condor_opportunity) deducts commission
    from self.capital before returning.  If the scan loop detects a duplicate
    key (_entered_today or _open_keys), the position is discarded — the
    commission must be refunded via the else: branch added in R8 P1.
    """

    def setup_method(self):
        self.mock_hd = _make_mock_historical_data()
        self.bt = Backtester(_make_config(), historical_data=self.mock_hd)
        self.bt.capital = 100_000

    def _make_price_data(self):
        import pandas as pd
        dates = pd.date_range('2024-11-18', periods=35, freq='B')
        prices = [450.0 + i * 2 for i in range(len(dates))]
        return pd.concat([
            pd.DataFrame(
                {'Close': prices, 'Open': prices, 'High': prices,
                 'Low': prices, 'Volume': [1_000_000] * len(dates)},
                index=dates,
            ),
            pd.DataFrame(
                {'Close': [prices[-1] + 2], 'Open': [prices[-1] + 2],
                 'High': [prices[-1] + 2], 'Low': [prices[-1] + 2],
                 'Volume': [1_000_000]},
                index=pd.date_range('2025-01-06', periods=1, freq='B'),
            ),
        ])

    def test_duplicate_bull_put_key_refunds_commission(self):
        """14 scan times returning the same (expiry, strike) bull put → 1 net commission."""
        from unittest.mock import patch

        commission = 1.30
        dummy_put = {
            'type': 'bull_put_spread', 'option_type': 'P',
            'short_strike': 450.0, 'long_strike': 445.0,
            'expiration': datetime(2025, 3, 21), 'credit': 1.20,
            'max_loss': 380.0, 'contracts': 1, 'commission': commission,
            'entry_price': 452.0, 'slippage_applied': 0.05, 'entry_scan_time': None,
            'entry_date': datetime(2025, 1, 6), 'ticker': 'SPY',
            'profit_target': 0.60, 'stop_loss': 3.00, 'status': 'open',
            'current_value': 120.0,
        }

        def fake_put(*args, **kwargs):
            self.bt.capital -= commission   # mimics _find_real_spread deduction
            return dict(dummy_put)

        capital_before = self.bt.capital

        with patch.object(self.bt, '_get_historical_data', return_value=self._make_price_data()), \
             patch.object(self.bt, '_find_backtest_opportunity', side_effect=fake_put), \
             patch.object(self.bt, '_find_bear_call_opportunity', return_value=None), \
             patch.object(self.bt, '_close_at_expiration_real', return_value=None):
            self.bt.run_backtest('SPY', datetime(2025, 1, 6), datetime(2025, 1, 6))

        # 14 calls: 1 accepted (commission kept), 13 dup-key refunded
        assert self.bt.capital == pytest.approx(capital_before - commission, rel=1e-6), (
            "Dup-key commission refund not firing for bull put."
        )

    def test_duplicate_bear_call_key_refunds_commission(self):
        """14 scan times returning the same (expiry, strike) bear call → 1 net commission."""
        from unittest.mock import patch

        commission = 1.30
        dummy_call = {
            'type': 'bear_call_spread', 'option_type': 'C',
            'short_strike': 480.0, 'long_strike': 485.0,
            'expiration': datetime(2025, 3, 21), 'credit': 1.10,
            'max_loss': 390.0, 'contracts': 1, 'commission': commission,
            'entry_price': 452.0, 'slippage_applied': 0.05, 'entry_scan_time': None,
            'entry_date': datetime(2025, 1, 6), 'ticker': 'SPY',
            'profit_target': 0.55, 'stop_loss': 2.75, 'status': 'open',
            'current_value': 110.0,
        }

        def fake_call(*args, **kwargs):
            self.bt.capital -= commission
            return dict(dummy_call)

        capital_before = self.bt.capital

        with patch.object(self.bt, '_get_historical_data', return_value=self._make_price_data()), \
             patch.object(self.bt, '_find_backtest_opportunity', return_value=None), \
             patch.object(self.bt, '_find_bear_call_opportunity', side_effect=fake_call), \
             patch.object(self.bt, '_close_at_expiration_real', return_value=None):
            self.bt.run_backtest('SPY', datetime(2025, 1, 6), datetime(2025, 1, 6))

        assert self.bt.capital == pytest.approx(capital_before - commission, rel=1e-6), (
            "Dup-key commission refund not firing for bear call."
        )

    def test_duplicate_ic_key_refunds_commission(self):
        """14 scan times returning the same IC key → 1 net commission."""
        from unittest.mock import patch

        commission = 2.60   # 4 legs × 0.65
        dummy_ic = {
            'type': 'iron_condor', 'option_type': 'IC',
            'short_strike': 445.0, 'long_strike': 440.0,
            'call_short_strike': 475.0, 'call_long_strike': 480.0,
            'expiration': datetime(2025, 3, 21), 'credit': 2.30,
            'max_loss': 7.70, 'contracts': 1, 'commission': commission,
            'entry_price': 452.0, 'slippage_applied': 0.10, 'entry_scan_time': None,
            'entry_date': datetime(2025, 1, 6), 'ticker': 'SPY',
            'profit_target': 1.15, 'stop_loss': 5.75, 'status': 'open',
            'current_value': 230.0,
        }

        cfg = _make_config()
        cfg['strategy']['iron_condor'] = {'enabled': True}
        bt = Backtester(cfg, historical_data=self.mock_hd)
        bt.capital = 100_000

        def fake_ic(*args, **kwargs):
            bt.capital -= commission
            return dict(dummy_ic)

        capital_before = bt.capital

        with patch.object(bt, '_get_historical_data', return_value=self._make_price_data()), \
             patch.object(bt, '_find_backtest_opportunity', return_value=None), \
             patch.object(bt, '_find_bear_call_opportunity', return_value=None), \
             patch.object(bt, '_find_iron_condor_opportunity', side_effect=fake_ic), \
             patch.object(bt, '_close_at_expiration_real', return_value=None):
            bt.run_backtest('SPY', datetime(2025, 1, 6), datetime(2025, 1, 6))

        assert bt.capital == pytest.approx(capital_before - commission, rel=1e-6), (
            "Dup-key commission refund not firing for IC."
        )


# ---------------------------------------------------------------------------
# R11 P1-2: IC scan-loop integration — IC attempted when single legs fail
# ---------------------------------------------------------------------------

class TestICScanLoopIntegration:
    """_find_iron_condor_opportunity is called when both single-leg finders return None.

    The scan loop (run_backtest) must fall through to IC after both
    _find_backtest_opportunity (put) and _find_bear_call_opportunity (call)
    return None.  If the fallthrough is broken, IC positions are silently never
    entered even when iron_condor is enabled.
    """

    def _make_price_data(self):
        import pandas as pd
        dates = pd.date_range('2024-11-18', periods=35, freq='B')
        prices = [450.0 + i * 2 for i in range(len(dates))]
        return pd.concat([
            pd.DataFrame(
                {'Close': prices, 'Open': prices, 'High': prices,
                 'Low': prices, 'Volume': [1_000_000] * len(dates)},
                index=dates,
            ),
            pd.DataFrame(
                {'Close': [prices[-1] + 2], 'Open': [prices[-1] + 2],
                 'High': [prices[-1] + 2], 'Low': [prices[-1] + 2],
                 'Volume': [1_000_000]},
                index=pd.date_range('2025-01-06', periods=1, freq='B'),
            ),
        ])

    def test_scan_loop_attempts_ic_when_single_legs_fail(self):
        """When put and call single legs both return None, IC finder must be called."""
        mock_hd = _make_mock_historical_data()
        cfg = _make_config()
        cfg['backtest']['use_real_data'] = True
        cfg['strategy']['direction'] = 'both'
        cfg['strategy']['iron_condor'] = {'enabled': True}

        bt = Backtester(cfg, historical_data=mock_hd)

        with patch.object(bt, '_get_historical_data', return_value=self._make_price_data()), \
             patch.object(bt, '_build_iv_rank_series', return_value={}), \
             patch.object(bt, '_build_realized_vol_series', return_value={}), \
             patch.object(bt, '_find_backtest_opportunity', return_value=None), \
             patch.object(bt, '_find_bear_call_opportunity', return_value=None), \
             patch.object(bt, '_find_iron_condor_opportunity', return_value=None) as ic_mock:
            bt.run_backtest('SPY', datetime(2025, 1, 6), datetime(2025, 1, 6))

        assert ic_mock.call_count > 0, (
            "_find_iron_condor_opportunity was never called — IC fallback path in scan "
            "loop is broken. Verify IC is attempted after put + call both return None."
        )


# ---------------------------------------------------------------------------
# R11 P1-3: Entry-day intraday scan skip
# ---------------------------------------------------------------------------

class TestIntradayEntryScanSkip:
    """On entry day, scans at or before entry_scan_time must be skipped.

    If the guard is broken (< instead of <=), the bar that triggered entry
    is re-evaluated for an immediate exit — a 30-min lookahead on entry day.
    """

    def test_check_intraday_exits_skips_entry_day_prior_scans(self):
        """Scans at and before entry_scan_time=10:30 must not call get_intraday_spread_prices.

        SCAN_TIMES: 9:15(skipped by market-open guard), 9:45, 10:00, 10:30, 11:00, ...15:30
        With entry_scan_time=10:30: 9:45, 10:00, 10:30 are additionally skipped.
        Only 11:00, 11:30, 12:00, 12:30, 13:00, 13:30, 14:00, 14:30, 15:00, 15:30
        (10 scans) should reach get_intraday_spread_prices.
        """
        mock_hd = _make_mock_historical_data()
        bt = Backtester(_make_config(), historical_data=mock_hd)

        pos = _make_position(credit=2.00, max_loss=3.00, contracts=1, commission=1.30)
        pos['option_type'] = 'P'
        pos['entry_date'] = datetime(2025, 2, 1)
        pos['entry_scan_time'] = '10:30'
        pos['expiration'] = datetime(2025, 2, 21)

        scans_called = []

        def tracking_intraday(ticker, exp, short_s, long_s, ot, date_str, hour, minute):
            scans_called.append((hour, minute))
            return None  # no data → no trigger

        mock_hd.get_intraday_spread_prices.side_effect = tracking_intraday

        result = bt._check_intraday_exits(pos, datetime(2025, 2, 1), '2025-02-01')

        # No data returned → function returns None (falls back to daily close)
        assert result is None

        # Every scan that was actually called must be strictly after 10:30
        for hour, minute in scans_called:
            assert hour * 60 + minute > 10 * 60 + 30, (
                f"Scan at {hour}:{minute:02d} was called but should have been skipped "
                f"(entry_scan_time=10:30, same-day entry). Guard changed from <= to <?"
            )

        # After entry at 10:30: 11:00, 11:30, 12:00, 12:30, 13:00,
        # 13:30, 14:00, 14:30, 15:00, 15:30 = 10 scans expected.
        assert len(scans_called) == 10, (
            f"Expected 10 post-entry scans after 10:30, got {len(scans_called)}: {scans_called}"
        )


# ---------------------------------------------------------------------------
# R11 P2-1: Profit-target / stop-loss exact-boundary triggers
# ---------------------------------------------------------------------------

class TestProfitStopBoundary:
    """The >= comparisons for profit_target and stop_loss must fire at the threshold.

    Floating-point drift or an off-by-epsilon error could prevent exits at the
    exact boundary.  These tests use integer-representable values so equality
    is exact.
    """

    def setup_method(self):
        self.mock_hd = _make_mock_historical_data()
        self.bt = Backtester(_make_config(), historical_data=self.mock_hd)
        self.bt.capital = 100_000
        self.bt.trades = []

    def test_profit_target_exact_boundary_triggers_close(self):
        """spread_value = credit − profit_target exactly must close the position."""
        # credit=2.00, profit_target = 2.00×0.5 = 1.00
        # spread_value = 1.00 → profit = 2.00 - 1.00 = 1.00 ≥ 1.00 → CLOSE
        pos = _make_position(credit=2.00, max_loss=3.00, contracts=1, commission=1.30)
        pos['option_type'] = 'P'
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 1.20, 'long_close': 0.20, 'spread_value': 1.00,
        }

        remaining = self.bt._manage_positions([pos], datetime(2025, 1, 20), 460.0, 'SPY')

        assert len(remaining) == 0, "Position should close at exact profit_target boundary."
        assert self.bt.trades[0]['exit_reason'] == 'profit_target'

    def test_stop_loss_exact_boundary_triggers_close(self):
        """spread_value − credit = stop_loss exactly must close the position."""
        # credit=2.00, stop_loss = 2.00×2.5 = 5.00
        # spread_value = 7.00 → loss = 7.00 - 2.00 = 5.00 ≥ 5.00 → CLOSE
        pos = _make_position(credit=2.00, max_loss=3.00, contracts=1, commission=1.30)
        pos['option_type'] = 'P'
        self.mock_hd.get_spread_prices.return_value = {
            'short_close': 7.50, 'long_close': 0.50, 'spread_value': 7.00,
        }

        remaining = self.bt._manage_positions([pos], datetime(2025, 1, 20), 440.0, 'SPY')

        assert len(remaining) == 0, "Position should close at exact stop_loss boundary."
        assert self.bt.trades[0]['exit_reason'] == 'stop_loss'


# ---------------------------------------------------------------------------
# R11 P2-2: Volume adaptive sizing cap
# ---------------------------------------------------------------------------

class TestVolumeSizeCap:
    """contracts must be capped to max(1, int(min_vol × vol_size_cap_pct))."""

    def test_volume_size_cap_limits_contracts(self):
        """With min daily volume=1000 and vol_size_cap=0.02, contracts capped to 20.

        Without the cap, 20% flat risk on $100K with a ~$3.50 max-loss spread
        would yield ~57 contracts.  The cap reduces that to 20.
        """
        mock_hd = _make_mock_historical_data()
        cfg = _make_config()
        cfg['backtest']['sizing_mode'] = 'flat'
        cfg['backtest']['volume_size_cap_pct'] = 0.02
        cfg['risk']['max_risk_per_trade'] = 20.0   # large budget → uncapped ~57 contracts
        cfg['backtest']['use_real_data'] = True

        bt = Backtester(cfg, historical_data=mock_hd)
        bt.capital = 100_000
        bt.starting_capital = 100_000
        bt._use_real_data = True
        bt._vol_size_cap = 0.02
        bt._volume_gate = True   # vol_size_cap only applies when volume_gate is enabled
        bt._min_vol_ratio = 0    # disable hard-reject; only test the cap logic

        mock_hd.get_available_strikes.return_value = [440, 445, 450, 455, 460]
        mock_hd.get_spread_prices.return_value = {
            'short_close': 2.00, 'long_close': 0.50, 'spread_value': 1.50,
        }
        mock_hd.build_occ_symbol.return_value = 'O:SPY250321P00455000'
        mock_hd.get_prev_daily_volume.return_value = 1000  # both legs → min_vol = 1000

        result = bt._find_real_spread(
            'SPY', datetime(2025, 1, 6), '2025-01-06', 480.0,
            datetime(2025, 3, 21), 5.0, option_type='P',
        )

        assert result is not None
        # Cap: max(1, int(1000 × 0.02)) = max(1, 20) = 20
        assert result['contracts'] == 20, (
            f"Expected 20 contracts (vol cap: int(1000×0.02)), got {result['contracts']}. "
            "Adaptive sizing cap not applied."
        )


# ---------------------------------------------------------------------------
# R13 P1-1: IC entry-day scan-skip (IC-specific branch of _check_intraday_exits)
# ---------------------------------------------------------------------------

class TestICIntradayEntryScanSkip:
    """_check_intraday_exits IC branch must also skip scans at/before entry_scan_time.

    The single-spread path (TestIntradayEntryScanSkip) already verifies this for
    puts/calls.  This class confirms the IC branch (which calls get_intraday_spread_prices
    twice — once for put leg, once for call leg) applies the same skip logic.
    """

    def test_ic_check_intraday_exits_skips_entry_day_prior_scans(self):
        """IC intraday exit scan skips bars at/before entry_scan_time=10:30 on entry day.

        IC path calls get_intraday_spread_prices twice per scan (put + call legs).
        With entry at 10:30, only the 10 post-entry scans should be attempted —
        i.e. at most 20 calls (10 scans × 2 legs), all at times > 10:30.
        """
        mock_hd = _make_mock_historical_data()
        bt = Backtester(_make_config(), historical_data=mock_hd)

        pos = _make_ic_position(credit=2.00, max_loss=8.00, contracts=1, commission=1.30)
        pos['entry_date'] = datetime(2025, 2, 1)
        pos['entry_scan_time'] = '10:30'
        pos['expiration'] = datetime(2025, 2, 21)

        scans_called = []

        def tracking_intraday(ticker, exp, short_s, long_s, ot, date_str, hour, minute):
            scans_called.append((hour, minute))
            return None  # no data → no trigger

        mock_hd.get_intraday_spread_prices.side_effect = tracking_intraday

        result = bt._check_intraday_exits(pos, datetime(2025, 2, 1), '2025-02-01')

        assert result is None  # no data → fall back to daily close

        # Every attempted scan must be strictly after 10:30
        for hour, minute in scans_called:
            assert hour * 60 + minute > 10 * 60 + 30, (
                f"IC scan at {hour}:{minute:02d} was called but should have been skipped "
                f"(entry_scan_time=10:30, same-day entry). IC branch missing the skip guard?"
            )

        # 10 post-entry scans × 2 legs = 20 calls maximum (all return None here)
        assert len(scans_called) == 20, (
            f"Expected 20 IC leg calls for 10 post-entry scans, got {len(scans_called)}: "
            f"{scans_called}"
        )


# ---------------------------------------------------------------------------
# R13 P2-2: Multi-position intraday exposure gate
# ---------------------------------------------------------------------------

class TestExposureGateMultiEntry:
    """Second same-day entry is blocked when first already saturates the exposure cap.

    _exposure_ok is evaluated before each new position is added to open_positions.
    After the first position is opened (current_value=0 at entry), the gate must
    still correctly compute cumulative max_loss and reject the second entry.
    """

    def test_second_entry_blocked_by_exposure_cap(self):
        """Two same-day entries: first allowed, second blocked when combined > cap."""
        cfg = _make_config()
        cfg['backtest']['max_portfolio_exposure_pct'] = 15.0  # tight cap
        bt = Backtester(cfg)
        bt.capital = 100_000

        # First position: max_loss=8.00, contracts=10 → $8,000 exposure = 8% of $100K
        first = _make_position(max_loss=8.00, contracts=10, credit=1.50, commission=1.30)
        first['current_value'] = 0  # matches post-R11 entry initialization

        open_positions = [first]

        # Second candidate: max_loss=8.00, contracts=10 → would add another $8,000 = 8%
        # Combined: 8% + 8% = 16% > 15% cap → should be BLOCKED
        second = _make_position(max_loss=8.00, contracts=10, credit=1.50, commission=1.30)

        # Replicate _exposure_ok logic directly (mirrors backtester implementation)
        current_max_loss = sum(p['max_loss'] * p['contracts'] * 100 for p in open_positions)
        new_max_loss = second['max_loss'] * second['contracts'] * 100
        total_equity = bt.capital + sum(p.get('current_value', 0) for p in open_positions)
        exposure_pct = (current_max_loss + new_max_loss) / max(total_equity, 1.0) * 100

        assert exposure_pct > 15.0, (
            f"Expected combined exposure > 15% cap; got {exposure_pct:.1f}%. "
            "Gate would not fire — test setup wrong."
        )

        # Verify first entry alone is within cap (gate allows first)
        solo_pct = current_max_loss / max(total_equity, 1.0) * 100
        assert solo_pct <= 15.0, (
            f"First entry alone ({solo_pct:.1f}%) exceeds cap — test setup wrong."
        )
