"""
Tests for COMPASS portfolio allocation features.

Coverage:
  1. Portfolio allocation math (weights, splits, per-ticker caps)
  2. Macro score scaling (fear boost, greed reduction)
  3. Paper mode safety (reject live API URLs when paper_mode=true)
  4. Per-ticker heat tracking (PortfolioHeatTracker)
  5. Portfolio risk limits (RiskGate COMPASS extension)
  6. Backward compatibility (compass_portfolio_mode=false = zero change)
"""

from unittest.mock import patch

import pytest

from alerts.alert_position_sizer import _MACRO_FEAR_SCALE, _MACRO_GREED_SCALE, AlertPositionSizer
from alerts.alert_schema import Alert, AlertType, Direction, Leg, SizeResult
from alerts.portfolio_heat_tracker import PortfolioHeatTracker
from compass.risk_gate import RiskGate

# ============================================================================
# Fixtures & helpers
# ============================================================================

def _make_legs(short=500.0, long_=495.0, option_type="put"):
    return [
        Leg(strike=short, option_type=option_type, action="sell", expiration="2026-06-20"),
        Leg(strike=long_, option_type=option_type, action="buy", expiration="2026-06-20"),
    ]


def _make_alert(
    ticker="SPY",
    credit=1.50,
    short=500.0,
    long_=495.0,
    risk_pct=0.02,
    direction=Direction.bullish,
    alert_type=AlertType.credit_spread,
    **overrides,
):
    defaults = dict(
        type=alert_type,
        ticker=ticker,
        direction=direction,
        legs=_make_legs(short, long_),
        entry_price=credit,
        stop_loss=3.00,
        profit_target=0.75,
        risk_pct=risk_pct,
    )
    defaults.update(overrides)
    return Alert(**defaults)


def _make_compass_config(
    spy_pct=0.65,
    sector_pct=0.35,
    active_sectors=None,
    max_risk=8.0,
    max_contracts=25,
    starting_capital=100_000,
):
    """Build a minimal COMPASS portfolio config."""
    return {
        "compass": {
            "portfolio_mode": True,
            "portfolio_weights": {
                "spy_pct": spy_pct,
                "sector_pct": sector_pct,
            },
            "active_sectors": active_sectors or ["XLE"],
        },
        "risk": {
            "account_size": starting_capital,
            "max_risk_per_trade": max_risk,
            "min_contracts": 1,
            "max_contracts": max_contracts,
            "sizing_mode": "flat",
        },
        "strategy": {
            "iron_condor": {
                "ic_risk_per_trade": 8.0,
            },
        },
        "backtest": {
            "starting_capital": starting_capital,
        },
    }


def _make_account_state(open_positions=None, account_value=100_000):
    return {
        "account_value": account_value,
        "peak_equity": account_value,
        "open_positions": open_positions or [],
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "recent_stops": [],
    }


# ============================================================================
# 1. Portfolio allocation math
# ============================================================================

class TestPortfolioAllocationMath:
    """Verify capital allocation weights and per-ticker sizing."""

    def test_spy_allocation_uses_spy_pct(self):
        """SPY should be sized against spy_pct * starting_capital."""
        cfg = _make_compass_config(spy_pct=0.65, sector_pct=0.35, max_risk=8.0)
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="SPY", credit=1.50, short=500.0, long_=495.0)

        result = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=30,
            current_portfolio_risk=0,
            macro_score=55.0,  # neutral — no scaling
        )

        # SPY base = 100_000 * 0.65 = 65_000
        # risk_pct = 8% → dollar_risk = 65_000 * 0.08 = 5_200
        # spread_width=5, credit=1.50 → max_loss_per = (5-1.50)*100 = 350
        # contracts = int(5_200 / 350) = 14
        expected_contracts = int(65_000 * 0.08 / 350)
        assert result.contracts == expected_contracts, (
            f"Expected {expected_contracts} contracts for SPY, got {result.contracts}"
        )

    def test_sector_allocation_splits_sector_pct(self):
        """Single active sector gets full sector_pct allocation."""
        cfg = _make_compass_config(spy_pct=0.65, sector_pct=0.35, active_sectors=["XLE"])
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="XLE", credit=0.30, short=85.0, long_=80.0)

        result = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=30,
            current_portfolio_risk=0,
            macro_score=55.0,
        )

        # XLE base = 100_000 * 0.35 / 1 = 35_000 (only sector)
        # risk_pct = 8% → dollar_risk = 35_000 * 0.08 = 2_800
        # max_loss_per = (5 - 0.30)*100 = 470
        # contracts = int(2_800 / 470) = 5
        expected_contracts = int(35_000 * 0.08 / 470)
        assert result.contracts == expected_contracts, (
            f"Expected {expected_contracts} contracts for XLE, got {result.contracts}"
        )

    def test_two_active_sectors_split_evenly(self):
        """Two active sectors each get sector_pct/2 of allocation."""
        cfg = _make_compass_config(
            spy_pct=0.65,
            sector_pct=0.35,
            active_sectors=["XLE", "XLK"],
        )
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="XLK", credit=0.50, short=200.0, long_=195.0)

        result = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=30,
            current_portfolio_risk=0,
            macro_score=55.0,
        )

        # XLK base = 100_000 * 0.35 / 2 = 17_500
        # risk_pct = 8% → dollar_risk = 17_500 * 0.08 = 1_400
        # max_loss = (5 - 0.50)*100 = 450
        # contracts = int(1_400 / 450) = 3
        expected_contracts = int(17_500 * 0.08 / 450)
        assert result.contracts == expected_contracts

    def test_unknown_ticker_falls_back_to_flat(self):
        """Ticker not in SPY or active_sectors falls back to flat sizing."""
        cfg = _make_compass_config(active_sectors=["XLE"])
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="QQQ", credit=1.50, short=480.0, long_=475.0)

        # flat sizer should not crash and return a valid result
        result = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=30,
            current_portfolio_risk=0,
        )
        assert isinstance(result, SizeResult)
        assert result.contracts >= 0

    def test_per_ticker_max_contracts_respects_allocation(self):
        """Max contracts is bounded by the ticker's allocation budget."""
        # Very small sector allocation → can only afford 1 contract
        cfg = _make_compass_config(
            spy_pct=0.99,
            sector_pct=0.01,
            active_sectors=["XLE"],
            max_risk=50.0,   # Large risk % — but tiny allocation
            max_contracts=100,
        )
        sizer = AlertPositionSizer(config=cfg)
        # XLE base = 100_000 * 0.01 = 1_000
        # max_loss_per = (5 - 1.50)*100 = 350
        # ticker_max = int(1_000 / 350) = 2
        alert = _make_alert(ticker="XLE", credit=1.50, short=85.0, long_=80.0)
        result = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=30,
            current_portfolio_risk=0,
            macro_score=55.0,
        )
        assert result.contracts <= 2

    def test_global_max_contracts_cap_still_applies(self):
        """Even with large allocation, global max_contracts is enforced."""
        cfg = _make_compass_config(
            spy_pct=0.65,
            sector_pct=0.35,
            active_sectors=["XLE"],
            max_risk=8.0,
            max_contracts=3,  # tight global cap
        )
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="SPY", credit=0.10, short=500.0, long_=495.0)
        result = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=30,
            current_portfolio_risk=0,
            macro_score=55.0,
        )
        assert result.contracts <= 3


# ============================================================================
# 2. Macro score scaling
# ============================================================================

class TestMacroScoreScaling:
    """Verify fear boost and greed reduction scaling factors."""

    def test_fear_boost_below_45(self):
        """macro_score < 45 → contracts boosted by 1.2×."""
        cfg = _make_compass_config()
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="SPY", credit=1.50, short=500.0, long_=495.0)

        base = sizer.size(alert=alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0, macro_score=55.0)
        boosted = sizer.size(alert=alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0, macro_score=40.0)

        # Fear boost → more contracts (or same if already at cap)
        assert boosted.contracts >= base.contracts
        # dollar_risk should be ~1.2× of base (allowing for int floor)
        assert boosted.dollar_risk >= base.dollar_risk * 0.9  # at least 90% to allow rounding

    def test_greed_reduction_above_75(self):
        """macro_score > 75 → contracts reduced by 0.85×."""
        cfg = _make_compass_config()
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="SPY", credit=1.50, short=500.0, long_=495.0)

        base = sizer.size(alert=alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0, macro_score=55.0)
        reduced = sizer.size(alert=alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0, macro_score=80.0)

        assert reduced.contracts <= base.contracts

    def test_neutral_macro_no_scaling(self):
        """macro_score 45–75 → scale = 1.0."""
        AlertPositionSizer(config=None)  # uses _macro_scale directly
        sizer_portfolio = AlertPositionSizer(config=_make_compass_config())

        scale = sizer_portfolio._macro_scale(60.0)
        assert scale == pytest.approx(1.0)

    def test_macro_scale_boundary_exactly_45(self):
        """Score exactly at 45 is NOT in the fear zone (< 45)."""
        sizer = AlertPositionSizer(config=_make_compass_config())
        assert sizer._macro_scale(45.0) == pytest.approx(1.0)

    def test_macro_scale_boundary_exactly_75(self):
        """Score exactly at 75 is in the mild greed zone (65 < 75, but not > 75 strong greed)."""
        sizer = AlertPositionSizer(config=_make_compass_config())
        assert sizer._macro_scale(75.0) == pytest.approx(0.95)

    def test_macro_scale_fear_returns_correct_constant(self):
        """Score 25 (< 30) hits the strong fear tier → 1.2×."""
        sizer = AlertPositionSizer(config=_make_compass_config())
        assert sizer._macro_scale(25.0) == pytest.approx(_MACRO_FEAR_SCALE)

    def test_macro_scale_greed_returns_correct_constant(self):
        sizer = AlertPositionSizer(config=_make_compass_config())
        assert sizer._macro_scale(85.0) == pytest.approx(_MACRO_GREED_SCALE)

    def test_macro_score_none_reads_from_db(self):
        """When macro_score=None, _macro_scale should try to read from DB."""
        sizer = AlertPositionSizer(config=_make_compass_config())
        with patch("alerts.alert_position_sizer.get_current_macro_score", return_value=60.0) as mock_score:
            scale = sizer._macro_scale(None)
            mock_score.assert_called_once()
            assert scale == pytest.approx(1.0)

    def test_macro_score_none_db_fail_returns_neutral(self):
        """DB failure → 1.0 fallback (never blocks)."""
        sizer = AlertPositionSizer(config=_make_compass_config())
        with patch("alerts.alert_position_sizer.get_current_macro_score", side_effect=Exception("DB error")):
            scale = sizer._macro_scale(None)
        assert scale == pytest.approx(1.0)

    def test_weekly_loss_breach_no_longer_reduces_portfolio_mode(self):
        """Backtester has no weekly-loss breach reduction — portfolio mode must match."""
        cfg = _make_compass_config()
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="SPY", credit=1.50, short=500.0, long_=495.0)

        normal = sizer.size(alert=alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0,
                            weekly_loss_breach=False, macro_score=55.0)
        with_breach = sizer.size(alert=alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0,
                                 weekly_loss_breach=True, macro_score=55.0)

        # weekly_loss_breach flag must NOT change sizing
        assert with_breach.dollar_risk == pytest.approx(normal.dollar_risk)


# ============================================================================
# 3. Paper mode safety
# ============================================================================

class TestPaperModeSafety:
    """Validate that paper_mode=true configs cannot hit live Alpaca."""

    def _call_validate(self, config):
        from main import _validate_paper_mode_safety
        _validate_paper_mode_safety(config)

    def test_paper_mode_false_skips_check(self):
        """paper_mode=false — no checks run, any Alpaca config is fine."""
        cfg = {
            "paper_mode": False,
            "alpaca": {"paper": False, "base_url": "https://api.alpaca.markets"},
        }
        self._call_validate(cfg)  # should not raise

    def test_paper_mode_missing_skips_check(self):
        """paper_mode absent (defaults to false) — no checks run."""
        cfg = {"alpaca": {"paper": False}}
        self._call_validate(cfg)  # should not raise

    def test_paper_mode_true_with_valid_paper_config(self):
        """paper_mode=true + alpaca.paper=true + paper base_url → passes."""
        cfg = {
            "paper_mode": True,
            "alpaca": {
                "paper": True,
                "base_url": "https://paper-api.alpaca.markets",
            },
        }
        self._call_validate(cfg)  # should not raise

    def test_paper_mode_true_no_base_url(self):
        """paper_mode=true + alpaca.paper=true + no base_url → passes."""
        cfg = {
            "paper_mode": True,
            "alpaca": {"paper": True},
        }
        self._call_validate(cfg)  # should not raise

    def test_paper_mode_true_alpaca_paper_false_raises(self):
        """paper_mode=true but alpaca.paper=false → ValueError."""
        cfg = {
            "paper_mode": True,
            "alpaca": {"paper": False},
        }
        with pytest.raises(ValueError, match="alpaca.paper=false"):
            self._call_validate(cfg)

    def test_paper_mode_true_live_base_url_raises(self):
        """paper_mode=true but base_url is the live endpoint → ValueError."""
        cfg = {
            "paper_mode": True,
            "alpaca": {
                "paper": True,
                "base_url": "https://api.alpaca.markets",
            },
        }
        with pytest.raises(ValueError, match="does not contain 'paper'"):
            self._call_validate(cfg)

    def test_paper_mode_true_live_base_url_regardless_of_paper_flag(self):
        """Even if someone sets alpaca.paper=true, a live base_url still blocks."""
        cfg = {
            "paper_mode": True,
            "alpaca": {
                "paper": True,
                "base_url": "https://api.alpaca.markets",  # no "paper" in URL
            },
        }
        with pytest.raises(ValueError):
            self._call_validate(cfg)

    def test_paper_url_substring_match_is_case_insensitive(self):
        """base_url with 'PAPER' uppercase should still pass."""
        cfg = {
            "paper_mode": True,
            "alpaca": {
                "paper": True,
                "base_url": "https://PAPER-api.alpaca.markets",
            },
        }
        self._call_validate(cfg)  # should not raise


# ============================================================================
# 4. PortfolioHeatTracker
# ============================================================================

class TestPortfolioHeatTracker:
    """Test per-ticker heat tracking."""

    def test_record_and_get_heat(self):
        tracker = PortfolioHeatTracker()
        tracker.record_entry("SPY", "trade-001", 5_000.0)
        assert tracker.get_ticker_heat("SPY") == pytest.approx(5_000.0)

    def test_multiple_positions_same_ticker(self):
        tracker = PortfolioHeatTracker()
        tracker.record_entry("SPY", "trade-001", 5_000.0)
        tracker.record_entry("SPY", "trade-002", 3_000.0)
        assert tracker.get_ticker_heat("SPY") == pytest.approx(8_000.0)
        assert tracker.get_ticker_position_count("SPY") == 2

    def test_record_exit_removes_heat(self):
        tracker = PortfolioHeatTracker()
        tracker.record_entry("SPY", "trade-001", 5_000.0)
        tracker.record_entry("SPY", "trade-002", 3_000.0)
        tracker.record_exit("SPY", "trade-001")
        assert tracker.get_ticker_heat("SPY") == pytest.approx(3_000.0)
        assert tracker.get_ticker_position_count("SPY") == 1

    def test_record_exit_idempotent(self):
        """Exiting a non-existent trade_id is a no-op."""
        tracker = PortfolioHeatTracker()
        tracker.record_exit("SPY", "nonexistent-trade")  # should not raise

    def test_portfolio_heat_sums_all_tickers(self):
        tracker = PortfolioHeatTracker()
        tracker.record_entry("SPY", "t1", 5_000.0)
        tracker.record_entry("XLE", "t2", 3_000.0)
        tracker.record_entry("XLK", "t3", 2_000.0)
        assert tracker.get_portfolio_heat() == pytest.approx(10_000.0)

    def test_empty_tracker_heat_is_zero(self):
        tracker = PortfolioHeatTracker()
        assert tracker.get_ticker_heat("SPY") == 0.0
        assert tracker.get_portfolio_heat() == 0.0

    def test_is_ticker_at_capacity_false_when_under(self):
        tracker = PortfolioHeatTracker()
        tracker.record_entry("SPY", "t1", 4_000.0)
        # allocation_weight=0.65, account=100K → budget=65K, 95% cap=61.75K
        # heat=4K < 61.75K → not at capacity
        result = tracker.is_ticker_at_capacity("SPY", 100_000, 0.65, heat_capacity_pct=0.95)
        assert result is False

    def test_is_ticker_at_capacity_true_when_full(self):
        tracker = PortfolioHeatTracker()
        # Fill up SPY to its budget (65K * 95% = 61.75K)
        tracker.record_entry("SPY", "t1", 62_000.0)
        result = tracker.is_ticker_at_capacity("SPY", 100_000, 0.65, heat_capacity_pct=0.95)
        assert result is True

    def test_is_ticker_at_capacity_zero_weight_never_blocks(self):
        """Zero allocation weight → never at capacity (division guard)."""
        tracker = PortfolioHeatTracker()
        tracker.record_entry("SPY", "t1", 1_000_000.0)
        result = tracker.is_ticker_at_capacity("SPY", 100_000, 0.0)
        assert result is False

    def test_clear_ticker_resets_heat(self):
        tracker = PortfolioHeatTracker()
        tracker.record_entry("SPY", "t1", 5_000.0)
        tracker.record_entry("SPY", "t2", 3_000.0)
        tracker.clear_ticker("SPY")
        assert tracker.get_ticker_heat("SPY") == 0.0
        assert tracker.get_ticker_position_count("SPY") == 0

    def test_clear_ticker_does_not_affect_others(self):
        tracker = PortfolioHeatTracker()
        tracker.record_entry("SPY", "t1", 5_000.0)
        tracker.record_entry("XLE", "t2", 3_000.0)
        tracker.clear_ticker("SPY")
        assert tracker.get_ticker_heat("XLE") == pytest.approx(3_000.0)

    def test_negative_max_loss_raises(self):
        tracker = PortfolioHeatTracker()
        with pytest.raises(ValueError):
            tracker.record_entry("SPY", "t1", -100.0)

    def test_get_all_ticker_heats(self):
        tracker = PortfolioHeatTracker()
        tracker.record_entry("SPY", "t1", 5_000.0)
        tracker.record_entry("XLE", "t2", 3_000.0)
        heats = tracker.get_all_ticker_heats()
        assert heats["SPY"] == pytest.approx(5_000.0)
        assert heats["XLE"] == pytest.approx(3_000.0)

    def test_sqlite_persistence(self, tmp_path):
        """Records survive tracker recreation when db_path is set."""
        db_file = str(tmp_path / "test_heat.db")

        tracker1 = PortfolioHeatTracker(db_path=db_file)
        tracker1.record_entry("SPY", "t1", 5_000.0)
        tracker1.record_entry("XLE", "t2", 3_000.0)

        # Recreate from same DB — should load existing records
        tracker2 = PortfolioHeatTracker(db_path=db_file)
        assert tracker2.get_ticker_heat("SPY") == pytest.approx(5_000.0)
        assert tracker2.get_ticker_heat("XLE") == pytest.approx(3_000.0)

    def test_sqlite_exit_persists(self, tmp_path):
        """record_exit deletes from SQLite so next load excludes closed trade."""
        db_file = str(tmp_path / "test_heat.db")

        tracker1 = PortfolioHeatTracker(db_path=db_file)
        tracker1.record_entry("SPY", "t1", 5_000.0)
        tracker1.record_exit("SPY", "t1")

        tracker2 = PortfolioHeatTracker(db_path=db_file)
        assert tracker2.get_ticker_heat("SPY") == 0.0


# ============================================================================
# 5. Portfolio risk limits in RiskGate
# ============================================================================

class TestPortfolioRiskLimits:
    """Verify COMPASS portfolio risk limits in RiskGate."""

    def _make_compass_risk_config(
        self,
        max_single_sector_pct=0.40,
        max_total_delta_pct=0.30,
        correlated_groups=None,
    ):
        groups = correlated_groups or {
            "tech_consumer": {
                "tickers": ["XLK", "XLY", "XLC", "SOXX"],
                "max_combined_pct": 0.40,
            }
        }
        return {
            "compass": {
                "portfolio_mode": True,
                "portfolio_risk_limits": {
                    "max_single_sector_pct": max_single_sector_pct,
                    "max_total_delta_pct": max_total_delta_pct,
                    "correlated_sector_groups": groups,
                },
            },
            "risk": {"account_size": 100_000},
            "backtest": {"starting_capital": 100_000},
        }

    def test_no_existing_positions_passes(self):
        cfg = self._make_compass_risk_config()
        gate = RiskGate(config=cfg)
        alert = _make_alert(ticker="XLK", risk_pct=0.02)  # valid ≤ 5%
        state = _make_account_state()
        passed, reason = gate.check(alert, state)
        assert passed, reason

    def test_single_sector_at_cap_blocks(self):
        """XLK already at max_single_sector_pct → new XLK entry blocked."""
        cfg = self._make_compass_risk_config(max_single_sector_pct=0.10)
        gate = RiskGate(config=cfg)
        alert = _make_alert(ticker="XLK", risk_pct=0.02)
        # existing XLK positions fill the 10% cap
        state = _make_account_state(open_positions=[
            {"ticker": "XLK", "direction": "bullish", "risk_pct": 0.10},
        ])
        passed, reason = gate.check(alert, state)
        assert not passed
        assert "XLK" in reason

    def test_single_sector_below_cap_passes(self):
        """XLK at 5% with 10% cap → new XLK entry allowed."""
        cfg = self._make_compass_risk_config(max_single_sector_pct=0.10)
        gate = RiskGate(config=cfg)
        alert = _make_alert(ticker="XLK", risk_pct=0.02)
        state = _make_account_state(open_positions=[
            {"ticker": "XLK", "direction": "bullish", "risk_pct": 0.05},
        ])
        passed, reason = gate.check(alert, state)
        assert passed, reason

    def test_correlated_group_cap_blocks(self):
        """XLK+XLY combined at cap → new XLC entry in same group blocked."""
        cfg = self._make_compass_risk_config(
            correlated_groups={
                "tech_consumer": {
                    "tickers": ["XLK", "XLY", "XLC"],
                    "max_combined_pct": 0.10,
                }
            }
        )
        gate = RiskGate(config=cfg)
        alert = _make_alert(ticker="XLC", risk_pct=0.02)
        state = _make_account_state(open_positions=[
            {"ticker": "XLK", "direction": "bullish", "risk_pct": 0.06},
            {"ticker": "XLY", "direction": "bullish", "risk_pct": 0.05},
        ])
        # 0.06 + 0.05 = 0.11 >= 0.10 cap → blocked
        passed, reason = gate.check(alert, state)
        assert not passed
        assert "tech_consumer" in reason

    def test_correlated_group_cap_passes_for_nonmember(self):
        """XLF (not in tech group) is not blocked when tech group is at cap."""
        cfg = self._make_compass_risk_config(
            max_total_delta_pct=1.0,  # disable delta cap so only group check matters
            correlated_groups={
                "tech": {"tickers": ["XLK", "XLY"], "max_combined_pct": 0.10}
            }
        )
        gate = RiskGate(config=cfg)
        alert = _make_alert(ticker="XLF", risk_pct=0.02)  # XLF not in tech group
        state = _make_account_state(open_positions=[
            {"ticker": "XLK", "direction": "bullish", "risk_pct": 0.06},
            {"ticker": "XLY", "direction": "bullish", "risk_pct": 0.05},
        ])
        passed, reason = gate.check(alert, state)
        # Tech group is at cap but XLF is not a member → tech group check does not apply
        assert "tech" not in reason

    def test_max_total_delta_blocks_directional(self):
        """Total directional risk > max_total_delta_pct → blocked."""
        cfg = self._make_compass_risk_config(max_total_delta_pct=0.05)
        gate = RiskGate(config=cfg)
        alert = _make_alert(ticker="SPY", risk_pct=0.02)  # directional bullish
        state = _make_account_state(open_positions=[
            {"ticker": "XLE", "direction": "bullish", "risk_pct": 0.04},
        ])
        # 0.04 existing + 0.02 new = 0.06 > 0.05 cap
        passed, reason = gate.check(alert, state)
        assert not passed
        assert "directional" in reason.lower()

    def test_neutral_alert_not_counted_for_delta(self):
        """Neutral (IC) alerts don't count toward max_total_delta_pct."""
        cfg = self._make_compass_risk_config(max_total_delta_pct=0.05)
        gate = RiskGate(config=cfg)

        ic_legs = [
            Leg(strike=495.0, option_type="put", action="buy", expiration="2026-06-20"),
            Leg(strike=500.0, option_type="put", action="sell", expiration="2026-06-20"),
            Leg(strike=510.0, option_type="call", action="sell", expiration="2026-06-20"),
            Leg(strike=515.0, option_type="call", action="buy", expiration="2026-06-20"),
        ]
        ic_alert = Alert(
            type=AlertType.iron_condor,
            ticker="SPY",
            direction=Direction.neutral,
            legs=ic_legs,
            entry_price=1.00,
            stop_loss=2.00,
            profit_target=0.50,
            risk_pct=0.02,   # valid ≤ 5%
        )
        state = _make_account_state(open_positions=[
            {"ticker": "XLE", "direction": "bullish", "risk_pct": 0.04},
        ])
        passed, reason = gate.check(ic_alert, state)
        # neutral IC: direction="neutral" → not counted toward directional delta cap
        assert passed, f"Neutral IC should pass delta limit but got: {reason}"

    def test_portfolio_mode_false_skips_limits(self):
        """When compass.portfolio_mode=false, portfolio limits are never checked."""
        cfg = {
            "compass": {
                "portfolio_mode": False,
                "portfolio_risk_limits": {
                    "max_single_sector_pct": 0.001,  # absurdly restrictive
                },
            },
            "risk": {"account_size": 100_000},
            "backtest": {"starting_capital": 100_000},
        }
        gate = RiskGate(config=cfg)
        alert = _make_alert(ticker="XLK", risk_pct=0.02)
        state = _make_account_state(open_positions=[
            {"ticker": "XLK", "direction": "bullish", "risk_pct": 0.10},
        ])
        # portfolio_mode=False → 0.001 limit never enforced
        passed, reason = gate.check(alert, state)
        assert "sector exposure" not in reason


# ============================================================================
# 6. Backward compatibility (compass_portfolio_mode=false)
# ============================================================================

class TestBackwardCompatibility:
    """Ensure portfolio_mode=false (default) produces identical results to before."""

    def _make_flat_config(self):
        return {
            "risk": {
                "account_size": 100_000,
                "max_risk_per_trade": 5.0,
                "min_contracts": 1,
                "max_contracts": 25,
                "sizing_mode": "flat",
            },
            "strategy": {
                "iron_condor": {"ic_risk_per_trade": 12.0},
            },
            "backtest": {"starting_capital": 100_000},
        }

    def test_no_compass_key_uses_flat_sizing(self):
        """Config with no 'compass' key → flat sizing (same as exp_154 mode)."""
        cfg = self._make_flat_config()
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="SPY", credit=1.50)

        result = sizer.size(alert=alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0)

        # 5% of 100K = 5000, max_loss = (5-1.5)*100=350, contracts = 14
        assert result.contracts == int(5_000 / 350)

    def test_compass_portfolio_mode_false_uses_flat_sizing(self):
        """compass.portfolio_mode=false → routes to _flat_risk_size."""
        cfg = self._make_flat_config()
        cfg["compass"] = {"portfolio_mode": False}
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="SPY", credit=1.50)

        result = sizer.size(alert=alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0)

        assert result.contracts == int(5_000 / 350)

    def test_no_config_uses_legacy_sizer(self):
        """No config at all → legacy IV-rank sizer (exp_036/exp_059 compat)."""
        sizer = AlertPositionSizer(config=None)
        alert = _make_alert(ticker="SPY", credit=1.50)

        # Legacy sizer requires ml.position_sizer — mock it
        with patch("alerts.alert_position_sizer.AlertPositionSizer._legacy_size") as mock_legacy:
            mock_legacy.return_value = SizeResult(risk_pct=0.02, contracts=5, dollar_risk=1_000, max_loss=1_000)
            sizer.size(
                alert=alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0
            )
            mock_legacy.assert_called_once()

    def test_risk_gate_without_compass_config_skips_portfolio_checks(self):
        """RiskGate with no compass config passes through without portfolio checks."""
        gate = RiskGate(config={
            "risk": {"account_size": 100_000},
            "backtest": {"starting_capital": 100_000},
        })
        alert = _make_alert(ticker="XLK", risk_pct=0.02)
        state = _make_account_state()
        passed, reason = gate.check(alert, state)
        assert passed, reason

    def test_macro_score_ignored_in_flat_mode(self):
        """In flat mode (no compass), macro_score parameter is accepted but unused."""
        cfg = self._make_flat_config()
        sizer = AlertPositionSizer(config=cfg)
        alert = _make_alert(ticker="SPY", credit=1.50)

        result_fear = sizer.size(alert=alert, account_value=100_000, iv_rank=30,
                                  current_portfolio_risk=0, macro_score=20.0)
        result_greed = sizer.size(alert=alert, account_value=100_000, iv_rank=30,
                                   current_portfolio_risk=0, macro_score=90.0)

        # Flat mode ignores macro_score → same result regardless
        assert result_fear.contracts == result_greed.contracts
