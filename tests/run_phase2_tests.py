#!/usr/bin/env python3
"""Self-contained Phase 2 test runner.

Mocks third-party dependencies (same pattern as Phase 1) and exercises:
- 0DTE config overlay
- Timing window logic
- Exit monitor alert logic
- from_opportunity 0DTE-aware behavior
- Window name mapping
"""

import sys
import types

# ---------------------------------------------------------------------------
# Mock third-party modules before any project imports
# ---------------------------------------------------------------------------

def _mock_module(name, attrs=None):
    """Register a fake module in sys.modules."""
    if name in sys.modules and not isinstance(sys.modules[name], types.ModuleType):
        return
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# pandas
pd = _mock_module("pandas", {
    "DataFrame": type("DataFrame", (), {"empty": True}),
    "Series": type("Series", (), {}),
    "read_csv": lambda *a, **k: None,
    "to_datetime": lambda *a, **k: None,
    "Timestamp": type("Timestamp", (), {}),
})

# numpy
np = _mock_module("numpy", {
    "array": lambda *a, **k: [],
    "ndarray": type("ndarray", (), {}),
    "float64": float,
    "int64": int,
    "mean": lambda x: sum(x) / len(x) if x else 0,
    "std": lambda x: 0.0,
    "nan": float("nan"),
    "isnan": lambda x: x != x,
    "log": lambda x: 0,
    "sqrt": lambda x: x ** 0.5,
    "exp": lambda x: 2.718 ** x,
    "zeros": lambda *a: [],
    "ones": lambda *a: [],
    "inf": float("inf"),
})

# sklearn
for mod_name in [
    "sklearn", "sklearn.ensemble", "sklearn.model_selection",
    "sklearn.preprocessing", "sklearn.metrics", "sklearn.pipeline",
    "sklearn.base",
]:
    _mock_module(mod_name, {
        "RandomForestClassifier": type("RandomForestClassifier", (), {}),
        "GradientBoostingClassifier": type("GBC", (), {}),
        "train_test_split": lambda *a, **k: ([], [], [], []),
        "StandardScaler": type("SS", (), {}),
        "accuracy_score": lambda *a: 0.0,
        "Pipeline": type("Pipeline", (), {}),
        "BaseEstimator": type("BE", (), {}),
    })

# scipy
for mod_name in ["scipy", "scipy.stats"]:
    _mock_module(mod_name, {
        "norm": type("norm", (), {"cdf": staticmethod(lambda x: 0.5), "pdf": staticmethod(lambda x: 0.5)}),
    })

# hmmlearn
for mod_name in ["hmmlearn", "hmmlearn.hmm"]:
    _mock_module(mod_name, {
        "GaussianHMM": type("GaussianHMM", (), {"__init__": lambda self, **k: None}),
    })

# joblib
_mock_module("joblib", {
    "dump": lambda *a, **k: None,
    "load": lambda *a, **k: None,
})

# yaml
_mock_module("yaml", {
    "safe_load": lambda s: {},
    "dump": lambda d, **k: "",
})

# telegram
_mock_module("telegram", {"Bot": type("Bot", (), {})})

# yfinance, requests, sentry_sdk
for mod_name in ["yfinance", "requests", "sentry_sdk"]:
    _mock_module(mod_name, {"init": lambda **k: None})

# pytz (needed by shared.scheduler → backtest chain)
_mock_module("pytz", {
    "timezone": lambda tz: None,
    "utc": None,
})

# shared.indicators
_mock_module("shared.indicators", {
    "calculate_rsi": lambda *a, **k: None,
    "calculate_iv_rank": lambda *a, **k: None,
    "sanitize_features": lambda *a, **k: {},
})

# shared.metrics
_mock_module("shared.metrics", {
    "metrics": type("Metrics", (), {"inc": lambda self, *a: None, "set": lambda self, *a: None})(),
})

# shared.data_cache
_mock_module("shared.data_cache", {
    "DataCache": type("DataCache", (), {"__init__": lambda self, *a, **k: None}),
})

sys.path.insert(0, ".")

# Pre-load ml.position_sizer (same trick as Phase 1)
import importlib.util

ml_pkg = types.ModuleType("ml")
ml_pkg.__path__ = ["."]
sys.modules["ml"] = ml_pkg

_ps_spec = importlib.util.spec_from_file_location(
    "ml.position_sizer",
    "./ml/position_sizer.py",
)
_ps_mod = importlib.util.module_from_spec(_ps_spec)
sys.modules["ml.position_sizer"] = _ps_mod
_ps_spec.loader.exec_module(_ps_mod)

# ----- Import modules under test -----
from datetime import datetime, timedelta, timezone

from alerts.alert_schema import (
    Alert,
    TimeSensitivity,
)
from alerts.zero_dte_config import SPX_PROPERTIES, build_zero_dte_config
from alerts.zero_dte_exit_monitor import ZeroDTEExitMonitor
from alerts.zero_dte_scanner import ZeroDTEScanner

print("All Phase 2 modules imported successfully\n")

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_pass = 0
_fail = 0
_errors = []


def test(name):
    """Decorator to register and run a test."""
    def wrapper(fn):
        global _pass, _fail
        try:
            fn()
            _pass += 1
            print(f"  PASS  {name}")
        except Exception as e:
            _fail += 1
            _errors.append((name, e))
            print(f"  FAIL  {name}: {e}")
    return wrapper


def _base_config():
    """Minimal base config for testing."""
    return {
        "strategy": {
            "min_dte": 30,
            "max_dte": 45,
            "min_delta": 0.20,
            "max_delta": 0.30,
            "spread_width": 10,
            "spread_width_high_iv": 15,
            "spread_width_low_iv": 10,
            "min_iv_rank": 20,
            "min_iv_percentile": 20,
            "iron_condor": {"enabled": True},
        },
        "risk": {
            "stop_loss_multiplier": 2.5,
            "min_credit_pct": 20,
            "account_size": 100000,
        },
        "tickers": ["SPY", "QQQ", "IWM"],
        "data": {},
    }


# ===== CONFIG OVERLAY TESTS =====
print("--- 0DTE Config Overlay ---")


@test("config: min_dte=0, max_dte=1")
def _():
    cfg = build_zero_dte_config(_base_config())
    assert cfg["strategy"]["min_dte"] == 0
    assert cfg["strategy"]["max_dte"] == 1


@test("config: correct delta range")
def _():
    cfg = build_zero_dte_config(_base_config())
    assert cfg["strategy"]["min_delta"] == 0.08
    assert cfg["strategy"]["max_delta"] == 0.16


@test("config: spread width = 5")
def _():
    cfg = build_zero_dte_config(_base_config())
    assert cfg["strategy"]["spread_width"] == 5
    assert cfg["strategy"]["spread_width_high_iv"] == 5
    assert cfg["strategy"]["spread_width_low_iv"] == 3


@test("config: iron_condor disabled")
def _():
    cfg = build_zero_dte_config(_base_config())
    assert cfg["strategy"]["iron_condor"]["enabled"] is False


@test("config: risk overrides")
def _():
    cfg = build_zero_dte_config(_base_config())
    assert cfg["risk"]["stop_loss_multiplier"] == 2.0
    assert cfg["risk"]["min_credit_pct"] == 10


@test("config: tickers = SPY + SPX")
def _():
    cfg = build_zero_dte_config(_base_config())
    assert cfg["tickers"] == ["SPY", "SPX"]


@test("config: base config NOT mutated")
def _():
    base = _base_config()
    original_min_dte = base["strategy"]["min_dte"]
    original_tickers = list(base["tickers"])
    build_zero_dte_config(base)
    assert base["strategy"]["min_dte"] == original_min_dte
    assert base["tickers"] == original_tickers


@test("config: SPX_PROPERTIES has required keys")
def _():
    for key in ("settlement", "exercise_style", "tax_treatment", "price_ticker"):
        assert key in SPX_PROPERTIES, f"Missing key: {key}"
    assert SPX_PROPERTIES["settlement"] == "cash"
    assert SPX_PROPERTIES["price_ticker"] == "^GSPC"


# ===== TIMING WINDOW TESTS =====
print("\n--- Timing Windows ---")


def _make_et_time(hour, minute):
    """Create a naive datetime at a specific ET time (treated as ET by scanner)."""
    return datetime(2026, 2, 26, hour, minute, 0)


@test("window: in post_open (9:40)")
def _():
    assert ZeroDTEScanner.is_in_entry_window(_make_et_time(9, 40)) is True


@test("window: in midday (11:30)")
def _():
    assert ZeroDTEScanner.is_in_entry_window(_make_et_time(11, 30)) is True


@test("window: in afternoon (14:15)")
def _():
    assert ZeroDTEScanner.is_in_entry_window(_make_et_time(14, 15)) is True


@test("window: outside at 8:00")
def _():
    assert ZeroDTEScanner.is_in_entry_window(_make_et_time(8, 0)) is False


@test("window: outside at 10:30")
def _():
    assert ZeroDTEScanner.is_in_entry_window(_make_et_time(10, 30)) is False


@test("window: outside at 13:00")
def _():
    assert ZeroDTEScanner.is_in_entry_window(_make_et_time(13, 0)) is False


@test("window: outside at 15:00")
def _():
    assert ZeroDTEScanner.is_in_entry_window(_make_et_time(15, 0)) is False


@test("window: boundary 9:35 exact → in window")
def _():
    assert ZeroDTEScanner.is_in_entry_window(_make_et_time(9, 35)) is True


@test("window: boundary 10:00 exact → out of window")
def _():
    assert ZeroDTEScanner.is_in_entry_window(_make_et_time(10, 0)) is False


# ===== WINDOW NAME TESTS =====
print("\n--- Window Names ---")


@test("name: post_open at 9:40")
def _():
    assert ZeroDTEScanner.active_window_name(_make_et_time(9, 40)) == "post_open"


@test("name: midday at 11:30")
def _():
    assert ZeroDTEScanner.active_window_name(_make_et_time(11, 30)) == "midday"


@test("name: afternoon at 14:15")
def _():
    assert ZeroDTEScanner.active_window_name(_make_et_time(14, 15)) == "afternoon"


@test("name: none at 13:00")
def _():
    assert ZeroDTEScanner.active_window_name(_make_et_time(13, 0)) == "none"


# ===== EXIT MONITOR TESTS =====
print("\n--- Exit Monitor ---")


class _MockPaperTrader:
    """Minimal paper trader mock."""
    def __init__(self, trades):
        self.open_trades = trades

    def _evaluate_position(self, trade, price, dte):
        # Return pre-set pnl and no close reason
        return trade.get("_mock_pnl", 0), None


class _MockTelegramBot:
    """Captures sent alerts."""
    def __init__(self):
        self.sent = []

    def send_alert(self, msg):
        self.sent.append(msg)


class _MockFormatter:
    """Returns raw string from format_exit_alert."""
    def format_exit_alert(self, **kwargs):
        return f"EXIT: {kwargs.get('ticker')} {kwargs.get('reason')} pnl={kwargs.get('current_pnl')}"


@test("exit: 50% profit triggers alert")
def _():
    trade = {
        "id": "t1", "ticker": "SPY", "dte_at_entry": 0,
        "total_credit": 100, "_mock_pnl": 55,
    }
    bot = _MockTelegramBot()
    monitor = ZeroDTEExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "profit_target"
    assert len(bot.sent) == 1


@test("exit: 2x stop triggers alert")
def _():
    trade = {
        "id": "t2", "ticker": "SPY", "dte_at_entry": 1,
        "total_credit": 100, "_mock_pnl": -210,
    }
    bot = _MockTelegramBot()
    monitor = ZeroDTEExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "stop_loss"


@test("exit: below threshold → no alert")
def _():
    trade = {
        "id": "t3", "ticker": "SPY", "dte_at_entry": 0,
        "total_credit": 100, "_mock_pnl": 30,  # < 50%
    }
    bot = _MockTelegramBot()
    monitor = ZeroDTEExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    assert len(triggered) == 0
    assert len(bot.sent) == 0


@test("exit: duplicate suppression")
def _():
    trade = {
        "id": "t4", "ticker": "SPY", "dte_at_entry": 0,
        "total_credit": 100, "_mock_pnl": 60,
    }
    bot = _MockTelegramBot()
    monitor = ZeroDTEExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    monitor.check_and_alert({"SPY": 550.0})
    # Second call should NOT fire again
    triggered2 = monitor.check_and_alert({"SPY": 550.0})
    assert len(triggered2) == 0
    assert len(bot.sent) == 1  # only 1 alert total


@test("exit: non-0DTE positions skipped")
def _():
    trade = {
        "id": "t5", "ticker": "SPY", "dte_at_entry": 30,
        "total_credit": 100, "_mock_pnl": 60,
    }
    bot = _MockTelegramBot()
    monitor = ZeroDTEExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    assert len(triggered) == 0


# ===== from_opportunity 0DTE TESTS =====
print("\n--- from_opportunity 0DTE ---")


def _make_opp(**overrides):
    base = {
        "ticker": "SPY", "type": "bull_put_spread", "expiration": "2026-02-26",
        "short_strike": 540.0, "long_strike": 535.0, "credit": 1.20,
        "stop_loss": 2.40, "profit_target": 0.60, "score": 72,
    }
    base.update(overrides)
    return base


@test("from_opp: DTE=0 → IMMEDIATE sensitivity")
def _():
    opp = _make_opp(dte=0)
    alert = Alert.from_opportunity(opp)
    assert alert.time_sensitivity == TimeSensitivity.IMMEDIATE


@test("from_opp: DTE=1 → IMMEDIATE sensitivity")
def _():
    opp = _make_opp(dte=1)
    alert = Alert.from_opportunity(opp)
    assert alert.time_sensitivity == TimeSensitivity.IMMEDIATE


@test("from_opp: DTE=30 → TODAY sensitivity")
def _():
    opp = _make_opp(dte=30)
    alert = Alert.from_opportunity(opp)
    assert alert.time_sensitivity == TimeSensitivity.TODAY


@test("from_opp: SPX cash-settled thesis")
def _():
    opp = _make_opp(ticker="SPX", settlement="cash")
    alert = Alert.from_opportunity(opp)
    assert "cash-settled" in alert.thesis
    assert "Section 1256" in alert.thesis


@test("from_opp: 0DTE expires in ~1hr")
def _():
    opp = _make_opp(dte=0)
    alert = Alert.from_opportunity(opp)
    # Should expire about 1 hour from now (within a few seconds)
    diff = alert.expires_at - datetime.now(timezone.utc)
    assert timedelta(minutes=59) <= diff <= timedelta(hours=1, seconds=5)


@test("from_opp: regular DTE expires in ~4hr")
def _():
    opp = _make_opp(dte=30)
    alert = Alert.from_opportunity(opp)
    diff = alert.expires_at - datetime.now(timezone.utc)
    assert timedelta(hours=3, minutes=59) <= diff <= timedelta(hours=4, seconds=5)


@test("from_opp: zero_dte source uses custom management_instructions")
def _():
    opp = _make_opp(
        alert_source="zero_dte",
        management_instructions="Custom 0DTE instructions here.",
    )
    alert = Alert.from_opportunity(opp)
    assert alert.management_instructions == "Custom 0DTE instructions here."


# ===== BACKTEST VALIDATOR TESTS =====
print("\n--- Backtest Validator ---")

from alerts.zero_dte_backtest import ZeroDTEBacktestValidator


@test("backtest: high win rate passes validation")
def _():
    results = {"total_trades": 100, "win_rate": 82.0, "profit_factor": 2.5}
    v = ZeroDTEBacktestValidator._validate(results)
    assert v["passed"] is True


@test("backtest: low win rate fails validation")
def _():
    results = {"total_trades": 100, "win_rate": 65.0, "profit_factor": 2.0}
    v = ZeroDTEBacktestValidator._validate(results)
    assert v["passed"] is False
    assert "Win rate" in v["reason"]


@test("backtest: low profit factor fails validation")
def _():
    results = {"total_trades": 100, "win_rate": 80.0, "profit_factor": 1.0}
    v = ZeroDTEBacktestValidator._validate(results)
    assert v["passed"] is False
    assert "Profit factor" in v["reason"]


@test("backtest: insufficient trades fails validation")
def _():
    results = {"total_trades": 5, "win_rate": 100.0, "profit_factor": 5.0}
    v = ZeroDTEBacktestValidator._validate(results)
    assert v["passed"] is False
    assert "Insufficient" in v["reason"]


@test("backtest: boundary 78% win rate passes")
def _():
    results = {"total_trades": 50, "win_rate": 78.0, "profit_factor": 1.5}
    v = ZeroDTEBacktestValidator._validate(results)
    assert v["passed"] is True


@test("backtest: config uses 0DTE overlay")
def _():
    validator = ZeroDTEBacktestValidator(_base_config())
    assert validator._zero_dte_config["strategy"]["min_dte"] == 0
    assert validator._zero_dte_config["strategy"]["max_dte"] == 1
    assert validator._zero_dte_config["tickers"] == ["SPY", "SPX"]


# ===== RESULTS =====
print(f"\n{'='*60}")
print(f"Phase 2 tests: {_pass} passed, {_fail} failed")
if _errors:
    print("\nFailed tests:")
    for name, err in _errors:
        print(f"  {name}: {err}")
print(f"{'='*60}")

sys.exit(1 if _fail else 0)
