"""Tests for shared.strategy_factory — build_strategy_list() and param extraction."""

import pytest

from shared.strategy_factory import (
    _extract_credit_spread_params,
    _extract_iron_condor_params,
    _extract_straddle_strangle_params,
    build_strategy_list,
)
from strategies.credit_spread import CreditSpreadStrategy
from strategies.iron_condor import IronCondorStrategy
from strategies.straddle_strangle import StraddleStrangleStrategy


# ---------------------------------------------------------------------------
# Minimal configs
# ---------------------------------------------------------------------------

def _base_config():
    return {
        "strategy": {
            "direction": "both",
            "target_dte": 35,
            "min_dte": 25,
            "otm_pct": 0.03,
            "spread_width": 10.0,
        },
        "risk": {
            "profit_target": 50,
            "stop_loss_multiplier": 2.5,
            "max_risk_per_trade": 2.0,
        },
    }


def _config_with_ic():
    cfg = _base_config()
    cfg["strategy"]["iron_condor"] = {
        "enabled": True,
        "target_dte": 35,
        "min_dte": 25,
    }
    return cfg


def _config_with_ss():
    cfg = _base_config()
    cfg["strategy"]["straddle_strangle"] = {
        "enabled": True,
        "mode": "short_post_event",
        "target_dte": 7,
    }
    return cfg


def _config_all_enabled():
    cfg = _base_config()
    cfg["strategy"]["iron_condor"] = {"enabled": True}
    cfg["strategy"]["straddle_strangle"] = {"enabled": True, "mode": "short_post_event"}
    return cfg


# ---------------------------------------------------------------------------
# build_strategy_list
# ---------------------------------------------------------------------------

class TestBuildStrategyList:
    def test_default_config_returns_credit_spread_only(self):
        """With no IC or SS sections, only CreditSpreadStrategy is returned."""
        strategies = build_strategy_list(_base_config())
        assert len(strategies) == 1
        assert isinstance(strategies[0], CreditSpreadStrategy)

    def test_ic_enabled_returns_two_strategies(self):
        strategies = build_strategy_list(_config_with_ic())
        types = [type(s) for s in strategies]
        assert CreditSpreadStrategy in types
        assert IronCondorStrategy in types
        assert len(strategies) == 2

    def test_ss_enabled_returns_two_strategies(self):
        strategies = build_strategy_list(_config_with_ss())
        types = [type(s) for s in strategies]
        assert CreditSpreadStrategy in types
        assert StraddleStrangleStrategy in types
        assert len(strategies) == 2

    def test_all_enabled_returns_three_strategies(self):
        strategies = build_strategy_list(_config_all_enabled())
        assert len(strategies) == 3
        types = [type(s) for s in strategies]
        assert CreditSpreadStrategy in types
        assert IronCondorStrategy in types
        assert StraddleStrangleStrategy in types

    def test_credit_spread_first(self):
        """CreditSpreadStrategy is always first in the list."""
        strategies = build_strategy_list(_config_all_enabled())
        assert isinstance(strategies[0], CreditSpreadStrategy)

    def test_ic_disabled_by_default(self):
        """IC section present but enabled=False (or absent) → not included."""
        cfg = _base_config()
        cfg["strategy"]["iron_condor"] = {"enabled": False, "target_dte": 35}
        strategies = build_strategy_list(cfg)
        types = [type(s) for s in strategies]
        assert IronCondorStrategy not in types

    def test_empty_config_does_not_crash(self):
        """build_strategy_list({}) should return at least a CreditSpreadStrategy."""
        strategies = build_strategy_list({})
        assert len(strategies) >= 1
        assert isinstance(strategies[0], CreditSpreadStrategy)


# ---------------------------------------------------------------------------
# _extract_credit_spread_params
# ---------------------------------------------------------------------------

class TestExtractCreditSpreadParams:
    def test_reads_target_dte_from_config(self):
        cfg = _base_config()
        cfg["strategy"]["target_dte"] = 42
        params = _extract_credit_spread_params(cfg)
        assert params["target_dte"] == 42

    def test_falls_back_to_min_dte_when_target_dte_missing(self):
        cfg = _base_config()
        cfg["strategy"].pop("target_dte", None)
        cfg["strategy"]["min_dte"] = 28
        params = _extract_credit_spread_params(cfg)
        assert params["target_dte"] == 28

    def test_profit_target_pct_is_fraction(self):
        """profit_target_pct should be 0-1 not 0-100."""
        params = _extract_credit_spread_params(_base_config())
        assert 0 < params["profit_target_pct"] <= 1.0
        assert params["profit_target_pct"] == pytest.approx(0.50)

    def test_max_risk_pct_is_fraction(self):
        params = _extract_credit_spread_params(_base_config())
        assert params["max_risk_pct"] == pytest.approx(0.02)

    def test_direction_passthrough(self):
        cfg = _base_config()
        cfg["strategy"]["direction"] = "bull"
        params = _extract_credit_spread_params(cfg)
        assert params["direction"] == "bull"

    def test_defaults_when_keys_missing(self):
        params = _extract_credit_spread_params({})
        assert params["direction"] == "both"
        assert params["target_dte"] == 35
        assert params["otm_pct"] == 0.05


# ---------------------------------------------------------------------------
# _extract_iron_condor_params
# ---------------------------------------------------------------------------

class TestExtractIronCondorParams:
    def test_reads_ic_section(self):
        cfg = _config_with_ic()
        cfg["strategy"]["iron_condor"]["rsi_min"] = 40
        params = _extract_iron_condor_params(cfg)
        assert params["rsi_min"] == 40

    def test_max_risk_pct_is_fraction(self):
        cfg = _config_with_ic()
        cfg["strategy"]["iron_condor"]["max_risk_pct"] = 4.0
        params = _extract_iron_condor_params(cfg)
        assert params["max_risk_pct"] == pytest.approx(0.04)

    def test_defaults(self):
        params = _extract_iron_condor_params({})
        assert params["rsi_min"] == 35
        assert params["rsi_max"] == 60
        assert params["target_dte"] == 35


# ---------------------------------------------------------------------------
# _extract_straddle_strangle_params
# ---------------------------------------------------------------------------

class TestExtractStraddleStrangleParams:
    def test_reads_mode(self):
        cfg = _config_with_ss()
        cfg["strategy"]["straddle_strangle"]["mode"] = "long_pre_event"
        params = _extract_straddle_strangle_params(cfg)
        assert params["mode"] == "long_pre_event"

    def test_max_risk_pct_is_fraction(self):
        cfg = _config_with_ss()
        cfg["strategy"]["straddle_strangle"]["max_risk_pct"] = 2.0
        params = _extract_straddle_strangle_params(cfg)
        assert params["max_risk_pct"] == pytest.approx(0.02)

    def test_defaults(self):
        params = _extract_straddle_strangle_params({})
        assert params["mode"] == "short_post_event"
        assert params["target_dte"] == 7


# ---------------------------------------------------------------------------
# Integration: build_strategy_list produces strategies that can generate_signals
# ---------------------------------------------------------------------------

class TestBuildStrategyListIntegration:
    def test_credit_spread_strategy_has_generate_signals(self):
        from strategies.base import MarketSnapshot
        strategies = build_strategy_list(_base_config())
        for s in strategies:
            assert hasattr(s, "generate_signals"), f"{type(s).__name__} missing generate_signals"

    def test_all_strategies_have_params_method(self):
        strategies = build_strategy_list(_config_all_enabled())
        for s in strategies:
            # All strategy instances should expose _p() for param lookup
            assert hasattr(s, "_p"), f"{type(s).__name__} missing _p"

    def test_regime_scale_params_forwarded(self):
        cfg = _base_config()
        cfg["risk"]["regime_scale_bull"] = 1.2
        params = _extract_credit_spread_params(cfg)
        assert params.get("regime_scale_bull") == 1.2
