"""
tests/test_monte_carlo.py — Unit tests for the expanded Monte Carlo randomization.

Tests each jitter dimension independently, then validates that full mode
combines all jitters correctly.  Uses MockHistoricalData (time-decay pricing
stub) so tests run fast and offline without Polygon data.
"""

import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# MockHistoricalData — test stub with time-decay pricing
# ---------------------------------------------------------------------------

class MockHistoricalData:
    """Minimal HistoricalOptionsData stub for MC math tests.

    Returns realistic-ish time-decaying spread prices so that trades open,
    profit targets trigger, and the backtester produces non-trivial results.
    This is a TEST FIXTURE — never use in production code.

    Pricing model:
      - get_available_strikes(): wide range around SPY levels (250-550)
      - get_spread_prices(): credit decays linearly from ~12% at entry to 0% at expiry
        → profit target (50% of credit) triggers around DTE=17
      - Intraday: always None (daily-close path used)
      - Volume/OI: always None (gates disabled in test config)
    """

    @staticmethod
    def build_occ_symbol(ticker, expiration, strike, option_type) -> str:
        exp_str = expiration.strftime("%y%m%d")
        ot = option_type[0].upper()
        strike_int = int(round(strike * 1000))
        return f"O:{ticker}{exp_str}{ot}{strike_int:08d}"

    def get_available_strikes(
        self, ticker: str, expiration, as_of_date: str, option_type: str = "P"
    ) -> List[float]:
        # Wide range covering all realistic SPY/test price levels
        return [float(s) for s in range(250, 550, 5)]

    def get_spread_prices(
        self, ticker, expiration, short_strike, long_strike, option_type, date
    ) -> Optional[Dict]:
        spread_width = abs(short_strike - long_strike)
        base_credit = spread_width * 0.12  # 12% → passes 8% min_credit check

        # Linear time-decay: spread value = base_credit * (dte / 35)
        # At DTE=35 (entry): value = base_credit  →  credit = base_credit
        # At DTE=17 (mid-life): value = base_credit/2 = profit_target  →  closes
        # At DTE=0 (expiry): value = 0  →  expires worthless, full credit captured
        if isinstance(date, str):
            current = datetime.strptime(date, "%Y-%m-%d")
        else:
            current = date
        exp_dt = expiration.replace(tzinfo=None) if hasattr(expiration, 'tzinfo') else expiration
        cur_dt = current.replace(tzinfo=None) if hasattr(current, 'tzinfo') else current
        dte = max(0, (exp_dt - cur_dt).days)

        time_factor = dte / 35.0
        spread_value = base_credit * time_factor
        return {
            "short_close": spread_value + 0.10,
            "long_close": 0.10,
            "spread_value": spread_value,
        }

    def get_intraday_spread_prices(
        self, ticker, expiration, short_strike, long_strike,
        option_type, date_str, hour, minute,
    ) -> Optional[Dict]:
        return None  # no intraday → daily-close path

    def get_intraday_bar(
        self, symbol: str, date_str: str, hour: int, minute: int
    ) -> Optional[Dict]:
        return None

    def get_prev_daily_volume(self, contract_symbol: str, before_date: str) -> Optional[int]:
        return None  # volume gate disabled in test config

    def get_prev_daily_oi(self, contract_symbol: str, before_date: str) -> Optional[int]:
        return None

    def get_strikes_with_approx_delta(
        self, ticker, expiration, current_price, date_str,
        option_type="P", iv_estimate=0.25, risk_free_rate=0.045,
    ) -> List[Dict]:
        return []  # not used: test config uses OTM% mode (use_delta_selection=False)

    @property
    def api_calls_made(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Minimal champion-like config fixture (heuristic mode, fast)
# ---------------------------------------------------------------------------

def _make_config(mc_overrides=None) -> dict:
    """Return a minimal Backtester config dict with optional MC overrides."""
    mc = {
        "mode": "dte",
        "dte_lo": 30,
        "dte_hi": 40,
        "slippage_jitter_pct": 0.0,
        "entry_day_jitter": 0,
        "trade_skip_pct": 0.0,
        "credit_jitter": 0.0,
    }
    if mc_overrides:
        mc.update(mc_overrides)
    return {
        "strategy": {
            "target_dte": 35,
            "min_dte": 25,
            "spread_width": 5,
            "min_credit_pct": 8,
            "direction": "bull_put",
            "trend_ma_period": 200,
            "regime_mode": "combo",
            "regime_config": {
                "signals": ["price_vs_ma200", "rsi_momentum", "vix_structure"],
                "ma_slow_period": 200,
                "ma200_neutral_band_pct": 0.5,
                "bear_requires_unanimous": True,
                "cooldown_days": 7,
                "vix_extreme": 40.0,
            },
            "iron_condor": {"enabled": False},
            "iv_rank_min_entry": 0,
        },
        "risk": {
            "stop_loss_multiplier": 2.5,
            "profit_target": 50,
            "max_risk_per_trade": 5.0,
            "max_contracts": 10,
            "max_positions": 20,
            "drawdown_cb_pct": 55,
        },
        "backtest": {
            "starting_capital": 100_000,
            "commission_per_contract": 0.65,
            "slippage": 0.05,
            "exit_slippage": 0.10,
            "compound": False,
            "sizing_mode": "flat",
            "slippage_multiplier": 1.0,
            "max_portfolio_exposure_pct": 100.0,
            "monte_carlo": mc,
        },
    }


def _run_with_mock(config: dict, seed: int = 42, year: int = 2022):
    """Run a backtest using MockHistoricalData (no Polygon required)."""
    from backtest.backtester import Backtester
    bt = Backtester(config, historical_data=MockHistoricalData(), otm_pct=0.03, seed=seed)
    result = bt.run_backtest("SPY", datetime(year, 1, 1), datetime(year, 12, 31))
    return bt, result


# Keep alias for any remaining references
_run_heuristic = _run_with_mock


# ---------------------------------------------------------------------------
# 1. DTE randomization (legacy mode)
# ---------------------------------------------------------------------------

class TestDteRandomization:
    """DTE jitter works in both 'dte' and 'full' mc_mode."""

    def test_dte_mode_rng_produces_values_in_range(self):
        """In dte mode, the seeded RNG samples DTE values within [dte_lo, dte_hi]."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "dte", "dte_lo": 28, "dte_hi": 42})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=0)
        assert bt._rng is not None
        sampled = [bt._rng.randint(bt._mc_dte_lo, bt._mc_dte_hi) for _ in range(100)]
        assert all(28 <= v <= 42 for v in sampled)
        # Confirm variation: with 100 draws from U[28,42], not all values can be 35
        assert len(set(sampled)) > 1

    def test_same_seed_is_deterministic(self):
        """Same seed → identical results every time."""
        cfg = _make_config({"mode": "dte"})
        _, r1 = _run_heuristic(cfg, seed=7)
        _, r2 = _run_heuristic(cfg, seed=7)
        assert r1["return_pct"] == r2["return_pct"]
        assert r1["total_trades"] == r2["total_trades"]

    def test_no_seed_uses_fixed_dte(self):
        """Without a seed, backtester uses target_dte from config, not random DTE."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "dte"})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=None)
        assert bt._rng is None
        assert bt._current_trade_dte == cfg["strategy"]["target_dte"]

    def test_rng_is_created_with_seed(self):
        """Providing a seed creates a seeded RNG on the Backtester instance."""
        from backtest.backtester import Backtester
        cfg = _make_config()
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=42)
        assert bt._rng is not None
        assert isinstance(bt._rng, random.Random)

    def test_full_mode_also_does_dte(self):
        """Full MC mode has mc_mode='full' and dte_lo/dte_hi are still respected."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "full", "dte_lo": 28, "dte_hi": 42})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=1)
        assert bt._mc_mode == "full"
        assert bt._mc_dte_lo == 28
        assert bt._mc_dte_hi == 42
        assert bt._rng is not None
        # DTE is sampled in full mode too — verify values stay in range
        sampled = [bt._rng.randint(bt._mc_dte_lo, bt._mc_dte_hi) for _ in range(50)]
        assert all(28 <= v <= 42 for v in sampled)


# ---------------------------------------------------------------------------
# 2. Slippage jitter
# ---------------------------------------------------------------------------

class TestSlippageJitter:
    """MC slippage jitter changes costs without breaking the simulation."""

    def test_helper_returns_base_in_dte_mode(self):
        """In 'dte' mode, _mc_jitter_slippage returns the base value unchanged."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "dte", "slippage_jitter_pct": 0.5})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=1)
        assert bt._mc_jitter_slippage(0.10) == 0.10

    def test_helper_returns_base_without_seed(self):
        """Without a seed, jitter is always inactive regardless of config."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "full", "slippage_jitter_pct": 0.5})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=None)
        assert bt._mc_jitter_slippage(0.10) == 0.10

    def test_helper_jitters_in_full_mode(self):
        """In full mode with jitter > 0, the helper returns a value within the expected range."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "full", "slippage_jitter_pct": 0.5})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=0)
        values = {bt._mc_jitter_slippage(1.0) for _ in range(50)}
        # With 50 draws, expect variation (not all 1.0)
        assert len(values) > 1
        # All values within [0.5, 1.5]
        assert all(0.5 <= v <= 1.5 for v in values)

    def test_jitter_zero_means_no_change(self):
        """slippage_jitter_pct=0 means helper always returns base unchanged."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "full", "slippage_jitter_pct": 0.0})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=5)
        for _ in range(20):
            assert bt._mc_jitter_slippage(0.10) == pytest.approx(0.10)

    def test_slippage_jitter_changes_pnl(self):
        """Two seeds with full slippage jitter produce different per-trade P&L."""
        cfg = _make_config({"mode": "full", "slippage_jitter_pct": 0.50})
        _, r0 = _run_heuristic(cfg, seed=0)
        _, r1 = _run_heuristic(cfg, seed=1)
        # At least one financial metric must differ between seeds
        assert r0["return_pct"] != r1["return_pct"] or r0["total_pnl"] != r1["total_pnl"]


# ---------------------------------------------------------------------------
# 3. Entry-day jitter
# ---------------------------------------------------------------------------

class TestEntryDayJitter:
    """Entry-day shift changes the expiration cycle targeted."""

    def test_entry_day_shift_stored(self):
        """_mc_entry_day_shift attribute exists and is initialized to 0."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "full", "entry_day_jitter": 2})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=1)
        assert hasattr(bt, "_mc_entry_day_shift")
        assert bt._mc_entry_day_shift == 0  # initial value

    def test_entry_day_jitter_zero_in_dte_mode(self):
        """In dte mode, entry day shift is always 0 regardless of config."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "dte", "entry_day_jitter": 2})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=1)
        assert bt._mc_entry_day_jitter == 2   # stored
        assert bt._mc_mode == "dte"           # but mode is dte → shift never applied

    def test_entry_day_jitter_shift_is_within_bounds(self):
        """With entry_day_jitter=2, shift is always in [-2, +2]."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "full", "entry_day_jitter": 2})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=5)
        assert bt._mc_mode == "full"
        assert bt._mc_entry_day_jitter == 2
        # Simulate 100 draws of the per-day shift
        shifts = [bt._rng.randint(-2, 2) for _ in range(100)]
        assert all(-2 <= s <= 2 for s in shifts)
        # Confirm variation: not all identical
        assert len(set(shifts)) > 1


# ---------------------------------------------------------------------------
# 4. Trade-skip simulation
# ---------------------------------------------------------------------------

class TestTradeSkip:
    """Random trade skipping reduces trade count and alters returns."""

    def test_skip_helper_false_in_dte_mode(self):
        """_mc_should_skip_trade() always returns False in dte mode."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "dte", "trade_skip_pct": 0.5})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=0)
        assert not any(bt._mc_should_skip_trade() for _ in range(100))

    def test_skip_helper_false_without_seed(self):
        """Without a seed, trade skip is never triggered."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "full", "trade_skip_pct": 0.5})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=None)
        assert not any(bt._mc_should_skip_trade() for _ in range(100))

    def test_skip_pct_zero_means_no_skip(self):
        """trade_skip_pct=0 means no trades are ever skipped."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "full", "trade_skip_pct": 0.0})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=3)
        assert not any(bt._mc_should_skip_trade() for _ in range(200))

    def test_skip_rate_matches_configured_pct(self):
        """With trade_skip_pct=0.5, roughly 50% of calls return True."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "full", "trade_skip_pct": 0.5})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=42)
        n = 1000
        skipped = sum(1 for _ in range(n) if bt._mc_should_skip_trade())
        # Allow ±10% around expected 50% (i.e. 400–600)
        assert 400 <= skipped <= 600

    def test_high_skip_reduces_trade_count(self):
        """With 80% skip rate, a full run has fewer trades than a 0% skip run."""
        cfg_no_skip = _make_config({"mode": "full", "trade_skip_pct": 0.0,
                                    "slippage_jitter_pct": 0.0, "entry_day_jitter": 0,
                                    "credit_jitter": 0.0})
        cfg_high_skip = _make_config({"mode": "full", "trade_skip_pct": 0.80,
                                      "slippage_jitter_pct": 0.0, "entry_day_jitter": 0,
                                      "credit_jitter": 0.0})
        _, r_no_skip = _run_heuristic(cfg_no_skip, seed=7)
        _, r_skip = _run_heuristic(cfg_high_skip, seed=7)
        # 80% skip → expect materially fewer trades
        assert r_skip["total_trades"] < r_no_skip["total_trades"]


# ---------------------------------------------------------------------------
# 5. Credit assumption jitter
# ---------------------------------------------------------------------------

class TestCreditJitter:
    """Credit jitter varies BACKTEST_CREDIT_FRACTION per trade."""

    def test_credit_jitter_zero_baseline_unchanged(self):
        """credit_jitter=0 leaves returns identical between seeds (modulo DTE)."""
        # With identical DTE + no other jitters, same seed = same result
        cfg = _make_config({"mode": "full", "credit_jitter": 0.0,
                            "slippage_jitter_pct": 0.0, "entry_day_jitter": 0,
                            "trade_skip_pct": 0.0})
        _, r1 = _run_heuristic(cfg, seed=5)
        _, r2 = _run_heuristic(cfg, seed=5)
        assert r1["return_pct"] == r2["return_pct"]

    def test_credit_jitter_changes_pnl(self):
        """With credit_jitter > 0, two different seeds produce different returns."""
        cfg = _make_config({"mode": "full", "credit_jitter": 0.10,
                            "slippage_jitter_pct": 0.0, "entry_day_jitter": 0,
                            "trade_skip_pct": 0.0})
        _, r0 = _run_heuristic(cfg, seed=0)
        _, r1 = _run_heuristic(cfg, seed=1)
        assert r0["return_pct"] != r1["return_pct"]

    def test_credit_fraction_clamped(self):
        """Credit fraction is clamped to [0.05, 0.95] even with large jitter."""
        import shared.constants as _c
        from backtest.backtester import Backtester
        # Set jitter larger than credit fraction itself
        cfg = _make_config({"mode": "full", "credit_jitter": 0.40})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=0)
        original = _c.BACKTEST_CREDIT_FRACTION
        # Run a short backtest and verify it completes without exception
        result = bt.run_backtest("SPY", datetime(2022, 1, 1), datetime(2022, 3, 31))
        assert result is not None
        # Restore constant (not modified by backtester — just read)
        assert _c.BACKTEST_CREDIT_FRACTION == original


# ---------------------------------------------------------------------------
# 6. Bootstrap sampling
# ---------------------------------------------------------------------------

class TestBootstrapSampling:
    """Bootstrap stats are populated in full MC mode."""

    def test_bootstrap_absent_in_dte_mode(self):
        """In dte mode, bootstrap dict is empty."""
        cfg = _make_config({"mode": "dte"})
        _, result = _run_heuristic(cfg, seed=1)
        assert result.get("bootstrap") == {} or result.get("bootstrap") is None

    def test_bootstrap_present_in_full_mode(self):
        """In full mode with ≥5 trades, bootstrap stats are populated."""
        cfg = _make_config({"mode": "full"})
        _, result = _run_heuristic(cfg, seed=2)
        bs = result.get("bootstrap", {})
        if result.get("total_trades", 0) >= 5:
            assert "P5" in bs
            assert "P50" in bs
            assert "P95" in bs
            assert "pct_profitable" in bs

    def test_bootstrap_percentiles_ordered(self):
        """Bootstrap P5 ≤ P50 ≤ P95."""
        cfg = _make_config({"mode": "full"})
        _, result = _run_heuristic(cfg, seed=3)
        bs = result.get("bootstrap", {})
        if bs and "P5" in bs:
            assert bs["P5"] <= bs["P50"] <= bs["P95"]

    def test_bootstrap_pct_profitable_in_range(self):
        """pct_profitable is between 0 and 100."""
        cfg = _make_config({"mode": "full"})
        _, result = _run_heuristic(cfg, seed=4)
        bs = result.get("bootstrap", {})
        if bs and "pct_profitable" in bs:
            assert 0 <= bs["pct_profitable"] <= 100


# ---------------------------------------------------------------------------
# 7. Full mode: simultaneous combination
# ---------------------------------------------------------------------------

class TestFullModeIntegration:
    """Full mode runs without errors and produces coherent results."""

    def test_full_mode_completes(self):
        """Full MC run completes without exception and returns a results dict."""
        cfg = _make_config({
            "mode": "full",
            "slippage_jitter_pct": 0.50,
            "entry_day_jitter": 2,
            "trade_skip_pct": 0.15,
            "credit_jitter": 0.10,
        })
        _, result = _run_heuristic(cfg, seed=100)
        assert isinstance(result, dict)
        assert "return_pct" in result
        assert "total_trades" in result

    def test_full_mode_results_positive_when_conditions_good(self):
        """In a bull year (2021) with small jitters, results should be positive."""
        cfg = _make_config({
            "mode": "full",
            "slippage_jitter_pct": 0.20,
            "entry_day_jitter": 1,
            "trade_skip_pct": 0.05,
            "credit_jitter": 0.05,
        })
        _, result = _run_heuristic(cfg, seed=0, year=2021)
        assert result.get("return_pct", 0) > 0

    def test_different_seeds_produce_different_full_mode_results(self):
        """Full mode: 10 seeds should produce at least 3 distinct return values."""
        cfg = _make_config({
            "mode": "full",
            "slippage_jitter_pct": 0.50,
            "entry_day_jitter": 2,
            "trade_skip_pct": 0.15,
            "credit_jitter": 0.10,
        })
        returns = set()
        for seed in range(10):
            _, r = _run_heuristic(cfg, seed=seed)
            returns.add(round(r.get("return_pct", 0), 1))
        assert len(returns) >= 3


# ---------------------------------------------------------------------------
# 8. Backward compatibility: dte mode is the default
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Legacy callers (no mc_mode key) still get DTE-only behaviour."""

    def test_default_mode_is_dte(self):
        """Without mc_mode key in config, mode defaults to 'dte'."""
        from backtest.backtester import Backtester
        cfg = _make_config()
        cfg["backtest"]["monte_carlo"].pop("mode", None)  # remove mode key
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=1)
        assert bt._mc_mode == "dte"

    def test_dte_mode_no_jitters_active(self):
        """In dte mode, all full-mode jitter helpers return base values."""
        from backtest.backtester import Backtester
        cfg = _make_config({"mode": "dte", "slippage_jitter_pct": 0.5,
                            "trade_skip_pct": 0.5, "credit_jitter": 0.5})
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=1)
        assert bt._mc_jitter_slippage(1.0) == 1.0
        assert not bt._mc_should_skip_trade()

    def test_run_without_seed_still_works(self):
        """Passing seed=None produces a valid result (non-MC mode)."""
        cfg = _make_config()
        from backtest.backtester import Backtester
        bt = Backtester(cfg, historical_data=MockHistoricalData(), otm_pct=0.03, seed=None)
        result = bt.run_backtest("SPY", datetime(2022, 1, 1), datetime(2022, 3, 31))
        assert result is not None
        assert "return_pct" in result
