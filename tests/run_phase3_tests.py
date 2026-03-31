#!/usr/bin/env python3
"""Self-contained Phase 3 test runner.

Mocks third-party dependencies (same pattern as Phase 1/2) and exercises:
- Iron condor config overlay
- Day-of-week entry gates
- Scanner scan() pipeline with mocks
- Exit monitor: profit/stop/dedup/weekly-close alerts
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

# talib
_mock_module("talib", {
    "SMA": lambda *a, **k: [],
    "RSI": lambda *a, **k: [],
    "BBANDS": lambda *a, **k: ([], [], []),
    "ATR": lambda *a, **k: [],
})

# yfinance, requests, sentry_sdk
for mod_name in ["yfinance", "requests", "sentry_sdk"]:
    _mock_module(mod_name, {"init": lambda **k: None})

# pytz
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

# Pre-load ml.position_sizer (same trick as Phase 1/2)
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
from datetime import datetime

from alerts.iron_condor_config import (
    CLOSE_DAYS,
    ENTRY_DAYS,
    build_iron_condor_config,
)
from alerts.iron_condor_exit_monitor import IronCondorExitMonitor
from alerts.iron_condor_scanner import IronCondorScanner

print("All Phase 3 modules imported successfully\n")

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
            "technical": {
                "sma_fast": 20,
                "sma_slow": 50,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "bb_period": 20,
                "bb_std_dev": 2.0,
            },
        },
        "risk": {
            "stop_loss_multiplier": 2.5,
            "min_credit_pct": 20,
            "account_size": 100000,
        },
        "tickers": ["SPY", "QQQ", "IWM"],
        "data": {},
    }


# Date helpers (2026-02-23 is Monday)
def _make_monday(h=10, m=0):
    return datetime(2026, 2, 23, h, m, 0)

def _make_tuesday(h=10, m=0):
    return datetime(2026, 2, 24, h, m, 0)

def _make_wednesday(h=10, m=0):
    return datetime(2026, 2, 25, h, m, 0)

def _make_thursday(h=10, m=0):
    return datetime(2026, 2, 26, h, m, 0)

def _make_friday(h=10, m=0):
    return datetime(2026, 2, 27, h, m, 0)

def _make_saturday(h=10, m=0):
    return datetime(2026, 2, 28, h, m, 0)


# ===== CONFIG OVERLAY TESTS =====
print("--- Iron Condor Config Overlay ---")


@test("config: min_dte=4, max_dte=10")
def _():
    cfg = build_iron_condor_config(_base_config())
    assert cfg["strategy"]["min_dte"] == 4
    assert cfg["strategy"]["max_dte"] == 10


@test("config: delta range 0.12-0.20")
def _():
    cfg = build_iron_condor_config(_base_config())
    assert cfg["strategy"]["min_delta"] == 0.12
    assert cfg["strategy"]["max_delta"] == 0.20


@test("config: spread width uniform 5")
def _():
    cfg = build_iron_condor_config(_base_config())
    assert cfg["strategy"]["spread_width"] == 5
    assert cfg["strategy"]["spread_width_high_iv"] == 5
    assert cfg["strategy"]["spread_width_low_iv"] == 5


@test("config: iron_condor enabled")
def _():
    cfg = build_iron_condor_config(_base_config())
    assert cfg["strategy"]["iron_condor"]["enabled"] is True


@test("config: min_combined_credit_pct=34")
def _():
    cfg = build_iron_condor_config(_base_config())
    assert cfg["strategy"]["iron_condor"]["min_combined_credit_pct"] == 34


@test("config: IV rank/percentile thresholds = 50")
def _():
    cfg = build_iron_condor_config(_base_config())
    assert cfg["strategy"]["min_iv_rank"] == 50
    assert cfg["strategy"]["min_iv_percentile"] == 50


@test("config: risk overrides")
def _():
    cfg = build_iron_condor_config(_base_config())
    assert cfg["risk"]["profit_target"] == 50
    assert cfg["risk"]["stop_loss_multiplier"] == 2.0


@test("config: expanded tickers")
def _():
    cfg = build_iron_condor_config(_base_config())
    assert cfg["tickers"] == ["SPY", "QQQ", "TSLA", "AMZN", "META", "GOOGL"]


@test("config: base config NOT mutated")
def _():
    base = _base_config()
    original_min_dte = base["strategy"]["min_dte"]
    original_tickers = list(base["tickers"])
    build_iron_condor_config(base)
    assert base["strategy"]["min_dte"] == original_min_dte
    assert base["tickers"] == original_tickers


@test("config: preserves other base keys")
def _():
    base = _base_config()
    base["custom_setting"] = "keep_me"
    cfg = build_iron_condor_config(base)
    assert cfg["custom_setting"] == "keep_me"


# ===== DAY-OF-WEEK GATE TESTS =====
print("\n--- Day-of-Week Gates ---")


@test("gate: Monday is entry day")
def _():
    assert IronCondorScanner.is_entry_day(_make_monday()) is True


@test("gate: Tuesday is entry day")
def _():
    assert IronCondorScanner.is_entry_day(_make_tuesday()) is True


@test("gate: Wednesday NOT entry day")
def _():
    assert IronCondorScanner.is_entry_day(_make_wednesday()) is False


@test("gate: Thursday NOT entry day")
def _():
    assert IronCondorScanner.is_entry_day(_make_thursday()) is False


@test("gate: Friday NOT entry day")
def _():
    assert IronCondorScanner.is_entry_day(_make_friday()) is False


@test("gate: Saturday NOT entry day")
def _():
    assert IronCondorScanner.is_entry_day(_make_saturday()) is False


@test("gate: ENTRY_DAYS constant = {0, 1}")
def _():
    assert ENTRY_DAYS == {0, 1}


@test("gate: CLOSE_DAYS constant = {3, 4}")
def _():
    assert CLOSE_DAYS == {3, 4}


# ===== SCANNER SCAN TESTS =====
print("\n--- Scanner Scan ---")


@test("scan: non-entry day returns empty")
def _():
    scanner = IronCondorScanner(_base_config())
    result = scanner.scan(now_et=_make_wednesday())
    assert result == []


@test("scan: entry day calls _scan_ticker for all tickers")
def _():
    scanner = IronCondorScanner(_base_config())
    call_count = [0]
    def counting_scan_ticker(ticker):
        call_count[0] += 1
        return [{"ticker": ticker, "type": "iron_condor", "score": 80}]
    scanner._scan_ticker = counting_scan_ticker
    result = scanner.scan(now_et=_make_monday())
    assert call_count[0] == 6  # SPY,QQQ,TSLA,AMZN,META,GOOGL
    assert len(result) == 6


@test("scan: handles ticker error gracefully")
def _():
    scanner = IronCondorScanner(_base_config())
    def failing_scan_ticker(ticker):
        raise Exception("API error")
    scanner._scan_ticker = failing_scan_ticker
    result = scanner.scan(now_et=_make_monday())
    assert result == []


@test("scan: annotates alert_source=iron_condor")
def _():
    scanner = IronCondorScanner(_base_config())
    def mock_scan_ticker(ticker):
        return [{"ticker": ticker, "type": "iron_condor", "score": 80, "alert_source": "iron_condor"}]
    scanner._scan_ticker = mock_scan_ticker
    result = scanner.scan(now_et=_make_tuesday())
    for opp in result:
        assert opp.get("alert_source") == "iron_condor"


# ===== EXIT MONITOR TESTS =====
print("\n--- Exit Monitor ---")


class _MockPaperTrader:
    """Minimal paper trader mock."""
    def __init__(self, trades):
        self.open_trades = trades

    def _evaluate_position(self, trade, price, dte):
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


def _make_ic_trade(**overrides):
    base = {
        "id": "ic1",
        "ticker": "SPY",
        "strategy_type": "iron_condor",
        "total_credit": 200,
        "_mock_pnl": 0,
    }
    base.update(overrides)
    return base


@test("exit: 50% profit triggers alert")
def _():
    trade = _make_ic_trade(_mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    reasons = [t["reason"] for t in triggered]
    assert "profit_target" in reasons
    assert len(bot.sent) >= 1


@test("exit: 2x stop loss triggers alert")
def _():
    trade = _make_ic_trade(_mock_pnl=-410)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    reasons = [t["reason"] for t in triggered]
    assert "stop_loss" in reasons


@test("exit: below threshold → no alert")
def _():
    trade = _make_ic_trade(_mock_pnl=50)  # 25% < 50% target
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    # Use Monday to avoid weekly close alerts
    triggered = monitor.check_and_alert({"SPY": 550.0}, now_et=_make_monday())
    assert len(triggered) == 0
    assert len(bot.sent) == 0


@test("exit: exact 50% triggers")
def _():
    trade = _make_ic_trade(_mock_pnl=100)  # exactly 50% of 200
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    reasons = [t["reason"] for t in triggered]
    assert "profit_target" in reasons


@test("exit: exact 2x stop triggers")
def _():
    trade = _make_ic_trade(_mock_pnl=-400)  # exactly -2x
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    reasons = [t["reason"] for t in triggered]
    assert "stop_loss" in reasons


@test("exit: composite dedup allows profit + weekly_close_warning")
def _():
    trade = _make_ic_trade(_mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert(
        {"SPY": 550.0}, now_et=_make_thursday()
    )
    reasons = {t["reason"] for t in triggered}
    assert "profit_target" in reasons
    assert "weekly_close_warning" in reasons


@test("exit: duplicate suppression works per-reason")
def _():
    trade = _make_ic_trade(_mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    monitor.check_and_alert({"SPY": 550.0})
    triggered2 = monitor.check_and_alert({"SPY": 550.0})
    profit_alerts = [t for t in triggered2 if t["reason"] == "profit_target"]
    assert len(profit_alerts) == 0


@test("exit: non-condor positions skipped")
def _():
    trade = _make_ic_trade(strategy_type="bull_put_spread", _mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    assert len(triggered) == 0


@test("exit: missing trade_id skipped")
def _():
    trade = _make_ic_trade(id="", _mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    assert len(triggered) == 0


@test("exit: zero credit skipped")
def _():
    trade = _make_ic_trade(total_credit=0, _mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    assert len(triggered) == 0


@test("exit: missing price skipped")
def _():
    trade = _make_ic_trade(_mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"QQQ": 400.0})  # no SPY
    assert len(triggered) == 0


@test("exit: Thursday warning fires")
def _():
    trade = _make_ic_trade(_mock_pnl=30)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert(
        {"SPY": 550.0}, now_et=_make_thursday()
    )
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "weekly_close_warning"


@test("exit: Friday close_now fires")
def _():
    trade = _make_ic_trade(_mock_pnl=30)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert(
        {"SPY": 550.0}, now_et=_make_friday()
    )
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "weekly_close_now"


@test("exit: Thursday warning + Friday close_now both fire")
def _():
    trade = _make_ic_trade(_mock_pnl=30)
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    # Thursday
    thu = monitor.check_and_alert(
        {"SPY": 550.0}, now_et=_make_thursday()
    )
    assert any(t["reason"] == "weekly_close_warning" for t in thu)
    # Friday — warning already sent, close_now is new
    fri = monitor.check_and_alert(
        {"SPY": 550.0}, now_et=_make_friday()
    )
    assert any(t["reason"] == "weekly_close_now" for t in fri)


@test("exit: 'type' field fallback (no strategy_type)")
def _():
    trade = {
        "id": "ic2", "ticker": "QQQ", "type": "iron_condor",
        "total_credit": 150, "_mock_pnl": 80,
    }
    bot = _MockTelegramBot()
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    # Use Monday to avoid weekly close alerts
    triggered = monitor.check_and_alert({"QQQ": 400.0}, now_et=_make_monday())
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "profit_target"


@test("exit: telegram failure doesn't crash")
def _():
    trade = _make_ic_trade(_mock_pnl=110)
    bot = _MockTelegramBot()
    _original_send = bot.send_alert
    def failing_send(msg):
        raise Exception("network error")
    bot.send_alert = failing_send
    monitor = IronCondorExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 550.0})
    assert len(triggered) >= 1


# ===== RESULTS =====
print(f"\n{'='*60}")
print(f"Phase 3 tests: {_pass} passed, {_fail} failed")
if _errors:
    print("\nFailed tests:")
    for name, err in _errors:
        print(f"  {name}: {err}")
print(f"{'='*60}")

sys.exit(1 if _fail else 0)
