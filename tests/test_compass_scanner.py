"""
Tests for COMPASS multi-underlying support in CreditSpreadSystem.

Covers:
  - _get_compass_universe(): dynamic universe selection
  - _augment_with_compass_state(): macro state injection into account_state
  - scan_opportunities(): dynamic vs static universe dispatch
  - _analyze_ticker(): COMPASS direction_override / rrg_quadrant injection
  - Backward compatibility: compass_universe_enabled=false behaves like before
"""

import pytest

try:
    from alpaca.trading.requests import OptionLegRequest  # noqa: F401
except ImportError:
    pytest.skip("OptionLegRequest not available in this alpaca-py version", allow_module_level=True)

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers — minimal config builders
# ---------------------------------------------------------------------------

def _base_config(**overrides) -> dict:
    """Minimal config matching CreditSpreadSystem's required shape."""
    cfg = {
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'risk': {'account_size': 100_000},
        'strategy': {'regime_mode': 'combo', 'ml_score_weight': 0.6},
        'alpaca': {'enabled': False},
        'compass': {},
    }
    cfg.update(overrides)
    return cfg


def _compass_config(**overrides) -> dict:
    """Config with COMPASS universe enabled."""
    cfg = _base_config()
    cfg['compass'] = {
        'universe_enabled': True,
        'rrg_filter': True,
        'min_leading_pct': 0.65,
    }
    cfg.update(overrides)
    return cfg


def _sector_rankings(items: list) -> list:
    """Build get_sector_rankings() response from shorthand list of (ticker, quadrant)."""
    return [
        {
            'ticker': ticker,
            'name': ticker,
            'category': 'sector',
            'rs_3m': 3.0,
            'rs_12m': 8.0,
            'rank_3m': i + 1,
            'rank_12m': i + 1,
            'rrg_quadrant': quadrant,
        }
        for i, (ticker, quadrant) in enumerate(items)
    ]


# ---------------------------------------------------------------------------
# _get_compass_universe() — unit tests
# ---------------------------------------------------------------------------

class TestGetCompassUniverse:
    """Tests for the dynamic universe selection method."""

    def _make_system(self, config: dict):
        """Build a CreditSpreadSystem with all heavy components mocked out."""
        from main import CreditSpreadSystem
        sys = object.__new__(CreditSpreadSystem)
        sys.config = config
        sys._peak_equity = float(config.get('risk', {}).get('account_size', 100_000))
        return sys

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_spy_always_included(self, mock_score, mock_rankings):
        """SPY must always be the first entry regardless of sector state."""
        mock_score.return_value = 60.0
        mock_rankings.return_value = _sector_rankings([('XLK', 'Leading'), ('XLF', 'Lagging')])

        sys = self._make_system(_compass_config())
        universe = sys._get_compass_universe()

        tickers = [t for t, _, _ in universe]
        assert tickers[0] == 'SPY'

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_strict_mode_only_leading(self, mock_score, mock_rankings):
        """min_leading_pct=0.65 → only 'Leading' quadrant added for bull_put."""
        mock_score.return_value = 60.0
        mock_rankings.return_value = _sector_rankings([
            ('XLK', 'Leading'),
            ('XLF', 'Improving'),   # excluded in strict mode
            ('XLE', 'Weakening'),
        ])

        sys = self._make_system(_compass_config())
        universe = sys._get_compass_universe()

        directions = {t: d for t, d, _ in universe}
        assert directions.get('XLK') == 'bull_put'
        assert 'XLF' not in directions  # Improving excluded at 0.65 threshold
        assert directions.get('XLE') == 'bear_call'

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_relaxed_mode_includes_improving(self, mock_score, mock_rankings):
        """min_leading_pct=0.50 → 'Leading' and 'Improving' both qualify."""
        mock_score.return_value = 60.0
        mock_rankings.return_value = _sector_rankings([
            ('XLK', 'Leading'),
            ('XLF', 'Improving'),
            ('XLE', 'Lagging'),
        ])

        cfg = _compass_config()
        cfg['compass']['min_leading_pct'] = 0.50
        sys = self._make_system(cfg)
        universe = sys._get_compass_universe()

        directions = {t: d for t, d, _ in universe}
        assert directions.get('XLK') == 'bull_put'
        assert directions.get('XLF') == 'bull_put'   # Improving included at 0.50
        assert directions.get('XLE') == 'bear_call'

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_bear_macro_veto_returns_spy_only(self, mock_score, mock_rankings):
        """macro_score < 45 → bear-macro veto; only SPY returned."""
        mock_score.return_value = 40.0
        mock_rankings.return_value = _sector_rankings([('XLK', 'Leading'), ('XLF', 'Leading')])

        sys = self._make_system(_compass_config())
        universe = sys._get_compass_universe()

        assert universe == [('SPY', None, None)]
        mock_rankings.assert_not_called()  # Should short-circuit before rankings

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_non_liquid_sector_excluded(self, mock_score, mock_rankings):
        """Tickers not in LIQUID_SECTOR_ETFS are always excluded."""
        mock_score.return_value = 60.0
        mock_rankings.return_value = _sector_rankings([
            ('XLK', 'Leading'),
            ('GLD', 'Leading'),    # not a LIQUID_SECTOR_ETF
            ('QQQ', 'Leading'),    # BASE_UNIVERSE, not sector ETF
        ])

        sys = self._make_system(_compass_config())
        universe = sys._get_compass_universe()

        tickers = [t for t, _, _ in universe]
        assert 'XLK' in tickers
        assert 'GLD' not in tickers
        assert 'QQQ' not in tickers  # excluded from sector universe

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_rrg_quadrant_preserved_in_tuple(self, mock_score, mock_rankings):
        """The raw rrg_quadrant string must be carried through for downstream use."""
        mock_score.return_value = 60.0
        mock_rankings.return_value = _sector_rankings([('XLK', 'Leading')])

        sys = self._make_system(_compass_config())
        universe = sys._get_compass_universe()

        xlk_entry = next((u for u in universe if u[0] == 'XLK'), None)
        assert xlk_entry is not None
        ticker, direction, quadrant = xlk_entry
        assert direction == 'bull_put'
        assert quadrant == 'Leading'

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_db_failure_falls_back_to_config_tickers(self, mock_score, mock_rankings):
        """Any DB exception falls back to config.tickers with no overrides."""
        mock_score.side_effect = RuntimeError("DB connection failed")

        sys = self._make_system(_compass_config())
        universe = sys._get_compass_universe()

        # Should fall back to config tickers with (ticker, None, None)
        assert ('SPY', None, None) in universe
        assert ('QQQ', None, None) in universe
        assert ('IWM', None, None) in universe
        # All overrides should be None
        assert all(d is None for _, d, _ in universe)


# ---------------------------------------------------------------------------
# _augment_with_compass_state() — unit tests
# ---------------------------------------------------------------------------

class TestAugmentWithCompassState:
    """Tests for macro state injection into account_state."""

    def _make_system(self, config: dict):
        from main import CreditSpreadSystem
        sys = object.__new__(CreditSpreadSystem)
        sys.config = config
        sys._peak_equity = 100_000.0
        return sys

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_adds_macro_score_and_quadrants(self, mock_score, mock_rankings):
        mock_score.return_value = 58.0
        mock_rankings.return_value = _sector_rankings([('XLK', 'Leading'), ('XLF', 'Lagging')])

        sys = self._make_system(_compass_config())
        state = {}
        sys._augment_with_compass_state(state)

        assert state['macro_score'] == 58.0
        assert state['rrg_quadrants']['XLK'] == 'Leading'
        assert state['rrg_quadrants']['XLF'] == 'Lagging'

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_fear_flag_when_score_below_45(self, mock_score, mock_rankings):
        mock_score.return_value = 40.0
        mock_rankings.return_value = []

        sys = self._make_system(_compass_config())
        state = {}
        sys._augment_with_compass_state(state)

        assert state['macro_sizing_flag'] == 'boost'

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_greed_flag_when_score_above_75(self, mock_score, mock_rankings):
        mock_score.return_value = 80.0
        mock_rankings.return_value = []

        sys = self._make_system(_compass_config())
        state = {}
        sys._augment_with_compass_state(state)

        assert state['macro_sizing_flag'] == 'reduce'

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_neutral_flag_in_normal_range(self, mock_score, mock_rankings):
        mock_score.return_value = 60.0
        mock_rankings.return_value = []

        sys = self._make_system(_compass_config())
        state = {}
        sys._augment_with_compass_state(state)

        assert state['macro_sizing_flag'] == 'neutral'

    def test_no_op_when_compass_disabled(self):
        """When compass is fully off, state must not be modified."""
        sys = self._make_system(_base_config())  # no compass keys
        state = {'account_value': 100_000}
        sys._augment_with_compass_state(state)

        assert 'macro_score' not in state
        assert 'rrg_quadrants' not in state
        assert 'macro_sizing_flag' not in state

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_db_failure_is_non_fatal(self, mock_score, mock_rankings):
        """DB errors during augmentation must not propagate — state unchanged."""
        mock_score.side_effect = RuntimeError("DB error")

        sys = self._make_system(_compass_config())
        state = {'account_value': 100_000}
        # Should not raise
        sys._augment_with_compass_state(state)
        # State should remain unchanged
        assert 'macro_score' not in state


# ---------------------------------------------------------------------------
# scan_opportunities() — universe dispatch tests
# ---------------------------------------------------------------------------

class TestScanOpportunitiesDispatch:
    """Verify scan_opportunities builds the correct universe for each mode."""

    def _make_system(self, config: dict):
        """Build a minimal CreditSpreadSystem suitable for dispatch tests."""
        from main import CreditSpreadSystem
        sys = object.__new__(CreditSpreadSystem)
        sys.config = config
        sys._peak_equity = 100_000.0
        sys.alert_generator = MagicMock()
        sys.alert_generator.generate_alerts.return_value = {}
        sys.telegram_bot = MagicMock()
        sys.telegram_bot.enabled = False
        sys.alert_router = MagicMock()
        sys.alert_router.route_opportunities.return_value = []
        sys.alpaca_provider = None
        # _analyze_ticker is mocked per test
        return sys

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_compass_enabled_uses_dynamic_universe(self, mock_score, mock_rankings):
        """With compass.universe_enabled=true, _analyze_ticker is called for each
        dynamic universe member, not the static config.tickers list."""
        mock_score.return_value = 60.0
        mock_rankings.return_value = _sector_rankings([('XLK', 'Leading')])

        sys = self._make_system(_compass_config())
        sys._analyze_ticker = MagicMock(return_value=[])
        sys._augment_with_compass_state = MagicMock()

        sys.scan_opportunities()

        called_tickers = {call_args[0][0] for call_args in sys._analyze_ticker.call_args_list}
        assert 'SPY' in called_tickers
        assert 'XLK' in called_tickers
        # Static tickers QQQ / IWM must NOT be called via config.tickers path
        assert 'QQQ' not in called_tickers
        assert 'IWM' not in called_tickers

    def test_compass_disabled_uses_static_tickers(self):
        """With compass.universe_enabled=false, _analyze_ticker is called exactly
        for the static config.tickers list with no overrides."""
        cfg = _base_config(tickers=['SPY', 'QQQ'])
        sys = self._make_system(cfg)
        sys._analyze_ticker = MagicMock(return_value=[])
        sys._augment_with_compass_state = MagicMock()

        sys.scan_opportunities()

        call_args_list = sys._analyze_ticker.call_args_list
        [(args[0], args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None)
                  for args, kwargs in call_args_list]
        # Extract positional args
        positional = [args for args, _ in sys._analyze_ticker.call_args_list]
        called_tickers = {a[0] for a in positional}
        called_overrides = {a[1] if len(a) > 1 else None for a in positional}

        assert called_tickers == {'SPY', 'QQQ'}
        assert called_overrides == {None}  # no direction overrides

    @patch('main.get_sector_rankings')
    @patch('main.get_current_macro_score')
    def test_sector_etf_gets_direction_override(self, mock_score, mock_rankings):
        """COMPASS mode: sector ETFs receive direction_override, SPY gets None."""
        mock_score.return_value = 60.0
        mock_rankings.return_value = _sector_rankings([('XLK', 'Leading')])

        sys = self._make_system(_compass_config())
        calls_received = []

        def capture_analyze(ticker, direction_override=None, rrg_quadrant=None):
            calls_received.append((ticker, direction_override, rrg_quadrant))
            return []

        sys._analyze_ticker = capture_analyze
        sys._augment_with_compass_state = MagicMock()

        sys.scan_opportunities()

        by_ticker = {t: (d, q) for t, d, q in calls_received}
        # SPY has no override
        assert by_ticker['SPY'] == (None, None)
        # XLK is Leading → bull_put
        assert by_ticker['XLK'][0] == 'bull_put'
        assert by_ticker['XLK'][1] == 'Leading'


# ---------------------------------------------------------------------------
# _analyze_ticker() — regime injection tests
# ---------------------------------------------------------------------------

class TestAnalyzeTickerRegimeInjection:
    """Verify that direction_override causes the correct combo_regime injection
    and that ComboRegimeDetector is NOT invoked for sector ETFs."""

    def _make_system(self, config: dict):
        from main import CreditSpreadSystem
        sys = object.__new__(CreditSpreadSystem)
        sys.config = config
        sys._peak_equity = 100_000.0

        # Mock strategy with combo regime active
        strategy = MagicMock()
        strategy.regime_mode = 'combo'
        strategy._combo_regime_detector = MagicMock()
        sys.strategy = strategy

        # Mock data_cache
        import pandas as pd
        dates = pd.date_range('2023-01-01', periods=300, freq='B')
        price_df = pd.DataFrame(
            {'Open': 400.0, 'High': 405.0, 'Low': 395.0, 'Close': 400.0, 'Volume': 1_000_000},
            index=dates,
        )
        data_cache = MagicMock()
        data_cache.get_history.return_value = price_df
        sys.data_cache = data_cache

        # Mock technical_analyzer
        technical_analyzer = MagicMock()
        technical_analyzer.analyze.return_value = {}
        sys.technical_analyzer = technical_analyzer

        # Mock options_analyzer
        options_chain = pd.DataFrame({'strike': [395, 400, 405]})
        options_analyzer = MagicMock()
        options_analyzer.get_options_chain.return_value = options_chain
        options_analyzer.get_current_iv.return_value = 0.20
        options_analyzer.calculate_iv_rank.return_value = {'iv_rank': 30}
        sys.options_analyzer = options_analyzer

        # Mock strategy.evaluate_spread_opportunity (legacy path)
        strategy.evaluate_spread_opportunity.return_value = []

        # Unified strategy path — empty list so generate_signals loop is no-op
        sys.unified_strategies = []

        sys.ml_pipeline = None

        return sys

    @patch('main.build_live_market_snapshot')
    def test_bull_put_override_injects_bull_regime(self, mock_snapshot):
        """direction_override='bull_put' → combo_regime='bull' in technical_signals."""
        from unittest.mock import MagicMock as MM
        mock_snapshot.return_value = MM()

        sys = self._make_system(_compass_config())
        captured_signals = {}

        orig_build = mock_snapshot
        def capture_snapshot(**kwargs):
            captured_signals.update(kwargs.get('technical_signals', {}))
            return MM()
        mock_snapshot.side_effect = capture_snapshot

        sys._analyze_ticker('XLK', direction_override='bull_put', rrg_quadrant='Leading')

        assert captured_signals.get('combo_regime') == 'bull'
        assert captured_signals.get('compass_rrg_quadrant') == 'Leading'

    @patch('main.build_live_market_snapshot')
    def test_bear_call_override_injects_bear_regime(self, mock_snapshot):
        """direction_override='bear_call' → combo_regime='bear' in technical_signals."""
        from unittest.mock import MagicMock as MM
        mock_snapshot.return_value = MM()

        sys = self._make_system(_compass_config())
        captured_signals = {}

        def capture_snapshot(**kwargs):
            captured_signals.update(kwargs.get('technical_signals', {}))
            return MM()
        mock_snapshot.side_effect = capture_snapshot

        sys._analyze_ticker('XLE', direction_override='bear_call', rrg_quadrant='Lagging')

        assert captured_signals.get('combo_regime') == 'bear'
        assert captured_signals.get('compass_rrg_quadrant') == 'Lagging'

    @patch('main.build_live_market_snapshot')
    def test_compass_override_skips_combo_regime_detector(self, mock_snapshot):
        """When direction_override is set, ComboRegimeDetector must NOT be called."""
        from unittest.mock import MagicMock as MM
        mock_snapshot.return_value = MM()

        sys = self._make_system(_compass_config())

        sys._analyze_ticker('XLK', direction_override='bull_put', rrg_quadrant='Leading')

        sys.strategy._combo_regime_detector.compute_regime_series.assert_not_called()

    @patch('main.build_live_market_snapshot')
    def test_no_override_runs_combo_regime_detector(self, mock_snapshot):
        """Without direction_override (SPY), ComboRegimeDetector IS invoked."""
        from unittest.mock import MagicMock as MM
        mock_snapshot.return_value = MM()

        sys = self._make_system(_compass_config())

        # Provide a mock regime series result
        from datetime import date
        sys.strategy._combo_regime_detector.compute_regime_series.return_value = {
            date(2024, 1, 5): 'bull'
        }

        sys._analyze_ticker('SPY', direction_override=None, rrg_quadrant=None)

        sys.strategy._combo_regime_detector.compute_regime_series.assert_called_once()

    def test_no_price_data_returns_empty(self):
        """Empty price data must return [] without raising."""
        import pandas as pd
        sys = self._make_system(_compass_config())
        sys.data_cache.get_history.return_value = pd.DataFrame()

        result = sys._analyze_ticker('XLK', direction_override='bull_put', rrg_quadrant='Leading')
        assert result == []

    def test_no_options_data_returns_empty(self):
        """Empty options chain must return [] without raising."""
        import pandas as pd
        sys = self._make_system(_compass_config())
        sys.options_analyzer.get_options_chain.return_value = pd.DataFrame()

        result = sys._analyze_ticker('XLK', direction_override='bull_put', rrg_quadrant='Leading')
        assert result == []


# ---------------------------------------------------------------------------
# Backward compatibility — compass disabled
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """When compass is entirely off, behaviour must be identical to pre-COMPASS."""

    def _make_minimal_system(self, config: dict):
        from main import CreditSpreadSystem
        sys = object.__new__(CreditSpreadSystem)
        sys.config = config
        sys._peak_equity = 100_000.0
        sys.alert_generator = MagicMock()
        sys.alert_generator.generate_alerts.return_value = {}
        sys.telegram_bot = MagicMock()
        sys.telegram_bot.enabled = False
        sys.alert_router = MagicMock()
        sys.alert_router.route_opportunities.return_value = []
        sys.alpaca_provider = None
        sys._analyze_ticker = MagicMock(return_value=[])
        sys._augment_with_compass_state = MagicMock()
        return sys

    def test_static_tickers_called_exactly(self):
        """All config.tickers are analyzed, no extras, no COMPASS DB calls."""
        cfg = _base_config(tickers=['SPY', 'QQQ', 'IWM'])
        # Ensure compass disabled
        cfg.pop('compass', None)

        sys = self._make_minimal_system(cfg)

        with patch('main.get_sector_rankings') as mock_rankings, \
             patch('main.get_current_macro_score') as mock_score:

            sys.scan_opportunities()

            # macro DB should NOT be called for universe selection
            mock_rankings.assert_not_called()
            mock_score.assert_not_called()

        called_tickers = {args[0] for args, _ in sys._analyze_ticker.call_args_list}
        assert called_tickers == {'SPY', 'QQQ', 'IWM'}

    def test_analyze_ticker_no_override(self):
        """_analyze_ticker is called with direction_override=None, rrg_quadrant=None."""
        cfg = _base_config(tickers=['SPY'])
        cfg.pop('compass', None)

        sys = self._make_minimal_system(cfg)
        sys.scan_opportunities()

        call_args = sys._analyze_ticker.call_args_list[0]
        args, kwargs = call_args
        # direction_override is second positional arg
        direction_override = args[1] if len(args) > 1 else kwargs.get('direction_override')
        rrg_quadrant = args[2] if len(args) > 2 else kwargs.get('rrg_quadrant')
        assert direction_override is None
        assert rrg_quadrant is None

    def test_augment_skipped_when_compass_off(self):
        """_augment_with_compass_state is called but no-ops when compass is off."""
        from main import CreditSpreadSystem
        sys = object.__new__(CreditSpreadSystem)
        sys.config = _base_config()  # no compass keys → both flags False
        sys._peak_equity = 100_000.0

        state = {'account_value': 100_000}

        with patch('main.get_current_macro_score') as mock_score:
            sys._augment_with_compass_state(state)
            mock_score.assert_not_called()

        assert 'macro_score' not in state
        assert 'macro_sizing_flag' not in state
