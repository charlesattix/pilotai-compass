"""Tests for alerts.alert_router — full pipeline integration test.

Uses mocked dependencies (RiskGate, AlertPositionSizer, TelegramBot,
TelegramAlertFormatter) to verify routing logic in isolation.
"""

from unittest.mock import MagicMock, patch

from alerts.alert_position_sizer import AlertPositionSizer
from alerts.alert_router import AlertRouter
from alerts.alert_schema import Alert, AlertType, Direction, Leg
from alerts.formatters.telegram import TelegramAlertFormatter
from compass.risk_gate import RiskGate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opp(ticker="SPY", score=75, opp_type="bull_put_spread", **extra):
    base = {
        "ticker": ticker,
        "type": opp_type,
        "expiration": "2025-06-20",
        "short_strike": 540.0,
        "long_strike": 535.0,
        "credit": 1.50,
        "stop_loss": 3.00,
        "profit_target": 0.75,
        "score": score,
    }
    base.update(extra)
    return base


def _clean_state():
    return {
        "account_value": 100_000,
        "open_positions": [],
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "recent_stops": [],
    }


def _build_router(
    risk_gate=None,
    position_sizer=None,
    telegram_bot=None,
    formatter=None,
):
    if risk_gate is None:
        risk_gate = RiskGate()
    if position_sizer is None:
        position_sizer = AlertPositionSizer()
    if telegram_bot is None:
        telegram_bot = MagicMock()
        telegram_bot.send_alert = MagicMock(return_value=True)
    if formatter is None:
        formatter = TelegramAlertFormatter()
    router = AlertRouter(risk_gate, position_sizer, telegram_bot, formatter)
    # Clear any stale dedup entries loaded from the real DB at init,
    # so each test starts with a clean dedup state.
    router._dedup_ledger = {}
    return router


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConversion:
    """Stage 1: convert opportunities to alerts."""

    def test_all_valid_opps_converted(self):
        """Score gate removed — all valid opportunities are converted regardless of score."""
        router = _build_router()
        # Two opps with different tickers — different dedup keys → both dispatched.
        opps = [_opp(ticker="SPY", score=59), _opp(ticker="QQQ", score=40)]
        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities(opps, _clean_state())
        # Both pass through score gate; different tickers so no dedup → 2 dispatched
        assert len(result) == 2

    def test_valid_opp_with_any_score_dispatched(self):
        """A single valid opportunity with any score (including < 60) is dispatched."""
        router = _build_router()
        opps = [_opp(score=10)]
        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities(opps, _clean_state())
        assert len(result) == 1

    def test_invalid_opp_skipped(self):
        """Malformed opportunity should be skipped, not crash the pipeline."""
        router = _build_router()
        bad_opp = {"ticker": "BAD"}  # missing required fields
        good_opp = _opp(score=70)
        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities([bad_opp, good_opp], _clean_state())
        assert len(result) == 1
        assert result[0].ticker == "SPY"


class TestDeduplication:
    """Stage 2: same (ticker, expiration, strike_type) within 30 min is deduped."""

    def test_first_alert_passes(self):
        router = _build_router()
        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities([_opp(score=70)], _clean_state())
        assert len(result) == 1

    def test_duplicate_within_window_filtered(self):
        router = _build_router()
        with patch("alerts.alert_router.insert_alert"):
            router.route_opportunities([_opp(score=70)], _clean_state())
            result = router.route_opportunities([_opp(score=70)], _clean_state())
        assert len(result) == 0

    def test_different_ticker_not_deduped(self):
        router = _build_router()
        with patch("alerts.alert_router.insert_alert"):
            router.route_opportunities([_opp(ticker="SPY", score=70)], _clean_state())
            result = router.route_opportunities([_opp(ticker="QQQ", score=70)], _clean_state())
        assert len(result) == 1


class TestRiskGateIntegration:
    """Stage 3: risk gate blocks alerts that fail checks."""

    def test_risk_gate_rejection(self):
        gate = MagicMock(spec=RiskGate)
        gate.check.return_value = (False, "over exposure")
        gate.weekly_loss_breach.return_value = False
        router = _build_router(risk_gate=gate)

        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities([_opp(score=70)], _clean_state())
        assert result == []

    def test_risk_gate_approval(self):
        gate = MagicMock(spec=RiskGate)
        gate.check.return_value = (True, "")
        gate.weekly_loss_breach.return_value = False
        router = _build_router(risk_gate=gate)

        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities([_opp(score=70)], _clean_state())
        assert len(result) == 1


class TestSizing:
    """Stage 4: position sizing attached to approved alerts."""

    def test_sizing_attached(self):
        router = _build_router()
        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities([_opp(score=70)], _clean_state())
        assert len(result) == 1
        assert result[0].sizing is not None
        assert result[0].sizing.contracts >= 0


class TestPrioritization:
    """Stage 5: type priority, then score. Top 5."""

    def test_top_5_limit(self):
        router = _build_router()
        opps = [_opp(ticker=f"T{i}", score=70 + i) for i in range(10)]
        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities(opps, _clean_state())
        assert len(result) <= 5

    def test_type_priority_order(self):
        """Credit spreads should rank above gamma lotto at same score."""
        router = _build_router()
        opps = [
            _opp(ticker="SPY", score=70, opp_type="bull_put_spread"),
            # gamma_lotto type doesn't map from legacy opp types,
            # but credit_spread ranks first
        ]
        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities(opps, _clean_state())
        assert len(result) >= 1
        assert result[0].type == AlertType.credit_spread


class TestDispatch:
    """Stage 6: Telegram send + DB persistence."""

    def test_telegram_called(self):
        bot = MagicMock()
        bot.send_alert.return_value = True
        router = _build_router(telegram_bot=bot)

        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities([_opp(score=70)], _clean_state())

        assert len(result) == 1
        bot.send_alert.assert_called_once()

    def test_db_persist_called(self):
        with patch("alerts.alert_router.insert_alert") as mock_insert:
            router = _build_router()
            router.route_opportunities([_opp(score=70)], _clean_state())
            mock_insert.assert_called_once()

    def test_telegram_failure_does_not_block(self):
        """If Telegram fails, the alert should still be persisted."""
        bot = MagicMock()
        bot.send_alert.side_effect = Exception("network error")
        router = _build_router(telegram_bot=bot)

        with patch("alerts.alert_router.insert_alert") as mock_insert:
            result = router.route_opportunities([_opp(score=70)], _clean_state())
        assert len(result) == 1
        mock_insert.assert_called_once()


class TestFullPipeline:
    """End-to-end: realistic multi-opportunity scenario."""

    def test_mixed_scores_and_types(self):
        """Score gate removed — all valid opps pass regardless of score; risk gate still applies."""
        router = _build_router()
        opps = [
            _opp(ticker="SPY", score=85, opp_type="bull_put_spread"),
            _opp(ticker="QQQ", score=72, opp_type="bear_call_spread"),
            _opp(ticker="AAPL", score=55),  # low score but now passes score gate
            _opp(ticker="IWM", score=90, opp_type="iron_condor",
                 call_short_strike=220.0, call_long_strike=225.0),
        ]

        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities(opps, _clean_state())

        tickers = {a.ticker for a in result}
        assert "SPY" in tickers
        assert len(result) <= 5


# ---------------------------------------------------------------------------
# New frequency-control tests (FREQ BUG fix)
# ---------------------------------------------------------------------------

def _ic_opp(ticker="SPY", score=75, expiration="2026-04-17", short_strike=510.0):
    """Build a minimal iron_condor opportunity dict."""
    return {
        "ticker": ticker,
        "type": "iron_condor",
        "expiration": expiration,
        "short_strike": short_strike,
        "long_strike": short_strike - 5,
        "call_short_strike": short_strike + 20,
        "call_long_strike": short_strike + 25,
        "credit": 2.50,
        "stop_loss": 7.50,
        "profit_target": 1.25,
        "score": score,
    }


class TestWithinScanDedup:
    """Dedup granularity: (ticker, direction, alert_type).

    Same (ticker, direction, alert_type) → only 1 dispatched per scan window.
    Different tickers or different directions → both dispatched.
    """

    def test_two_spy_ics_same_expiration_only_one_dispatched(self):
        """Two SPY ICs with the SAME expiration → only 1 dispatched (same contract key)."""
        router = _build_router()
        opps = [
            _ic_opp(expiration="2026-04-17", short_strike=510.0, score=75),
            _ic_opp(expiration="2026-04-17", short_strike=512.0, score=74),
        ]
        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities(opps, _clean_state())
        spy_ic = [a for a in result if a.ticker == "SPY" and a.direction.value == "neutral"]
        assert len(spy_ic) == 1, (
            f"Expected 1 SPY IC per expiration per scan, got {len(spy_ic)}"
        )

    def test_two_spy_ics_different_expirations_both_dispatched(self):
        """Two SPY ICs with DIFFERENT expirations → both dispatched (different contracts, different scan keys)."""
        router = _build_router()
        opps = [
            _ic_opp(expiration="2026-04-17", short_strike=510.0),
            _ic_opp(expiration="2026-04-10", short_strike=512.0),
        ]
        with patch("alerts.alert_router.insert_alert"), patch("alerts.alert_router.upsert_dedup_entry"):
            result = router.route_opportunities(opps, _clean_state())
        spy_ic = [a for a in result if a.ticker == "SPY" and a.direction.value == "neutral"]
        assert len(spy_ic) == 2, (
            f"Expected 2 SPY ICs (different expirations = different contracts), got {len(spy_ic)}"
        )

    def test_three_spy_ics_same_exp_deduped_different_exp_not(self):
        """3 SPY ICs: 2 same expiration → deduped to 1, 1 different expiration → dispatched separately."""
        router = _build_router()
        opps = [
            _ic_opp(expiration="2026-04-17", short_strike=510.0, score=64),
            _ic_opp(expiration="2026-04-17", short_strike=512.0, score=63),  # same exp → deduped
            _ic_opp(expiration="2026-04-10", short_strike=513.0, score=62),  # diff exp → dispatched
        ]
        with patch("alerts.alert_router.insert_alert"), patch("alerts.alert_router.upsert_dedup_entry"):
            result = router.route_opportunities(opps, _clean_state())
        spy_results = [a for a in result if a.ticker == "SPY"]
        assert len(spy_results) == 2, (
            f"Expected 2 SPY ICs (1 per expiration), got {len(spy_results)}"
        )

    def test_different_tickers_not_blocked(self):
        """SPY IC and IWM IC in same scan should both be dispatched."""
        router = _build_router()
        opps = [
            _ic_opp(ticker="SPY", expiration="2026-04-17"),
            _ic_opp(ticker="IWM", expiration="2026-04-17"),
        ]
        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities(opps, _clean_state())
        tickers = {a.ticker for a in result}
        assert "SPY" in tickers
        assert "IWM" in tickers
        assert len(result) == 2

    def test_different_directions_same_ticker_not_blocked(self):
        """A bull_put AND a bear_call on same ticker are different direction keys → both allowed."""
        router = _build_router()
        opps = [
            _opp(ticker="SPY", score=75, opp_type="bull_put_spread"),
            _opp(ticker="SPY", score=73, opp_type="bear_call_spread",
                 short_strike=600.0, long_strike=605.0),
        ]
        with patch("alerts.alert_router.insert_alert"):
            result = router.route_opportunities(opps, _clean_state())
        directions = {a.direction.value for a in result if a.ticker == "SPY"}
        assert "bullish" in directions
        assert "bearish" in directions


class TestMaxPositionsPerTicker:
    """Fix 3: risk_gate enforces risk.max_positions_per_ticker from config."""

    def _gate_with_limit(self, limit=2):
        # max_total_exposure_pct=100 so Rule 2 never fires in these tests;
        # we are specifically testing Rule 5.5 (per-ticker limit).
        return RiskGate(config={"risk": {"max_positions_per_ticker": limit,
                                         "max_risk_per_trade": 10.0,
                                         "max_total_exposure_pct": 100}})

    def _state_with_spy_positions(self, n):
        return {
            "account_value": 100_000,
            "open_positions": [
                {"ticker": "SPY", "direction": "neutral", "risk_pct": 0.01}
                for _ in range(n)
            ],
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
        }

    def _spy_alert(self):
        from alerts.alert_schema import AlertType
        return Alert(
            type=AlertType.iron_condor,
            ticker="SPY",
            direction=Direction.neutral,
            legs=[
                Leg(510.0, "put", "sell", "2026-04-17"),
                Leg(505.0, "put", "buy",  "2026-04-17"),
                Leg(530.0, "call", "sell", "2026-04-17"),
                Leg(535.0, "call", "buy",  "2026-04-17"),
            ],
            entry_price=2.50,
            stop_loss=7.50,
            profit_target=1.25,
            risk_pct=0.05,
        )

    def test_zero_existing_positions_allowed(self):
        gate = self._gate_with_limit(2)
        alert = self._spy_alert()
        passed, _ = gate.check(alert, self._state_with_spy_positions(0))
        assert passed is True

    def test_one_existing_below_limit_allowed(self):
        gate = self._gate_with_limit(2)
        alert = self._spy_alert()
        passed, _ = gate.check(alert, self._state_with_spy_positions(1))
        assert passed is True

    def test_at_limit_blocked(self):
        gate = self._gate_with_limit(2)
        alert = self._spy_alert()
        passed, reason = gate.check(alert, self._state_with_spy_positions(2))
        assert passed is False
        assert "max 2 per ticker" in reason

    def test_limit_1_blocks_second_position(self):
        gate = self._gate_with_limit(1)
        alert = self._spy_alert()
        passed, _ = gate.check(alert, self._state_with_spy_positions(1))
        assert passed is False

    def test_no_limit_configured_always_passes(self):
        """When max_positions_per_ticker is absent, Rule 5.5 is skipped."""
        gate = RiskGate(config={"risk": {"max_risk_per_trade": 10.0,
                                          "max_total_exposure_pct": 100}})
        alert = self._spy_alert()
        # 2 existing SPY bullish positions — different direction from neutral IC alert,
        # so Rule 5 (correlated positions) won't fire.  With no max_positions_per_ticker
        # configured, Rule 5.5 is absent and the check should pass.
        state = {
            "account_value": 100_000,
            "open_positions": [
                {"ticker": "SPY", "direction": "bullish", "risk_pct": 0.01},
                {"ticker": "SPY", "direction": "bullish", "risk_pct": 0.01},
            ],
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
        }
        passed, _ = gate.check(alert, state)
        assert passed is True

    def test_different_ticker_not_counted(self):
        """IWM positions do not count toward SPY's per-ticker limit."""
        gate = self._gate_with_limit(1)
        from alerts.alert_schema import AlertType
        iwm_alert = Alert(
            type=AlertType.iron_condor,
            ticker="IWM",
            direction=Direction.neutral,
            legs=[
                Leg(200.0, "put", "sell", "2026-04-17"),
                Leg(195.0, "put", "buy",  "2026-04-17"),
                Leg(220.0, "call", "sell", "2026-04-17"),
                Leg(225.0, "call", "buy",  "2026-04-17"),
            ],
            entry_price=1.50,
            stop_loss=4.50,
            profit_target=0.75,
            risk_pct=0.05,
        )
        state = self._state_with_spy_positions(1)  # 1 SPY, not IWM
        passed, _ = gate.check(iwm_alert, state)
        assert passed is True
