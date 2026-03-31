#!/usr/bin/env python3
"""Self-contained Phase 1 test runner.

This script mocks out all third-party dependencies (pandas, sklearn, etc.)
so the Phase 1 alert infrastructure tests can run in minimal environments.
"""

import importlib
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

# yfinance, requests
for mod_name in ["yfinance", "requests", "sentry_sdk"]:
    _mock_module(mod_name, {"init": lambda **k: None})

# shared.indicators (avoid pandas import)
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

# We need to pre-load ml.position_sizer BEFORE ml.__init__ tries to import
# heavy ml submodules (regime_detector, etc.).
# Strategy: mock the ml package, then load position_sizer directly.

# First, import shared.constants (it has no heavy deps)

# Create a fake ml package
ml_pkg = types.ModuleType("ml")
ml_pkg.__path__ = ["."]
sys.modules["ml"] = ml_pkg

# Now load the real ml.position_sizer (it only needs shared.types)
import importlib.util

_ps_spec = importlib.util.spec_from_file_location(
    "ml.position_sizer",
    "./ml/position_sizer.py",
)
_ps_mod = importlib.util.module_from_spec(_ps_spec)
sys.modules["ml.position_sizer"] = _ps_mod
_ps_spec.loader.exec_module(_ps_mod)

# Now import everything
from datetime import datetime, timedelta, timezone

from alerts.alert_position_sizer import AlertPositionSizer

# ----- Import modules under test -----
from alerts.alert_schema import (
    Alert,
    AlertStatus,
    AlertType,
    Confidence,
    Direction,
    Leg,
    SizeResult,
    TimeSensitivity,
)
from alerts.formatters.telegram import TelegramAlertFormatter
from compass.risk_gate import RiskGate
from shared.constants import (
    COOLDOWN_AFTER_STOP,
    DAILY_LOSS_LIMIT,
    MAX_CORRELATED_POSITIONS,
    MAX_RISK_PER_TRADE,
    MAX_TOTAL_EXPOSURE,
    WEEKLY_LOSS_LIMIT,
)

print("All Phase 1 modules imported successfully\n")

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


def _make_legs():
    return [
        Leg(strike=100.0, option_type="put", action="sell", expiration="2025-06-20"),
        Leg(strike=95.0, option_type="put", action="buy", expiration="2025-06-20"),
    ]


def _make_alert(**overrides):
    defaults = dict(
        type=AlertType.credit_spread,
        ticker="SPY",
        direction=Direction.bullish,
        legs=_make_legs(),
        entry_price=1.50,
        stop_loss=3.00,
        profit_target=0.75,
        risk_pct=0.02,
    )
    defaults.update(overrides)
    return Alert(**defaults)


def _clean_state(**overrides):
    base = {
        "account_value": 100_000,
        "open_positions": [],
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "recent_stops": [],
    }
    base.update(overrides)
    return base


# ===== ALERT SCHEMA TESTS =====
print("--- Alert Schema ---")


@test("enum: AlertType has 5 members")
def _():
    assert len(AlertType) == 5


@test("enum: Confidence has 3 members")
def _():
    assert len(Confidence) == 3


@test("enum: TimeSensitivity has 3 members")
def _():
    assert len(TimeSensitivity) == 3


@test("enum: Direction has 3 members")
def _():
    assert len(Direction) == 3


@test("alert: basic creation")
def _():
    a = _make_alert()
    assert a.ticker == "SPY"
    assert a.type == AlertType.credit_spread
    assert a.status == AlertStatus.pending
    assert len(a.legs) == 2


@test("alert: unique IDs")
def _():
    assert _make_alert().id != _make_alert().id


@test("validation: empty legs raises")
def _():
    try:
        _make_alert(legs=[])
        assert False, "Should have raised"
    except ValueError:
        pass


@test("validation: risk_pct=0 raises")
def _():
    try:
        _make_alert(risk_pct=0.0)
        assert False
    except ValueError:
        pass


@test("validation: risk_pct=0.06 raises")
def _():
    try:
        _make_alert(risk_pct=0.06)
        assert False
    except ValueError:
        pass


@test("validation: risk_pct=0.05 OK")
def _():
    a = _make_alert(risk_pct=0.05)
    assert a.risk_pct == 0.05


@test("validation: entry_price=0 raises")
def _():
    try:
        _make_alert(entry_price=0.0)
        assert False
    except ValueError:
        pass


@test("from_opportunity: bull_put_spread")
def _():
    opp = {
        "ticker": "AAPL", "type": "bull_put_spread", "expiration": "2025-07-18",
        "short_strike": 180.0, "long_strike": 175.0, "credit": 1.20,
        "stop_loss": 2.40, "profit_target": 0.60, "score": 72,
    }
    a = Alert.from_opportunity(opp)
    assert a.ticker == "AAPL"
    assert a.type == AlertType.credit_spread
    assert a.direction == Direction.bullish
    assert len(a.legs) == 2


@test("from_opportunity: iron_condor has 4 legs")
def _():
    opp = {
        "ticker": "SPY", "type": "iron_condor", "expiration": "2025-07-18",
        "short_strike": 540.0, "long_strike": 535.0,
        "call_short_strike": 560.0, "call_long_strike": 565.0,
        "credit": 2.50, "stop_loss": 5.0, "profit_target": 1.25, "score": 80,
    }
    a = Alert.from_opportunity(opp)
    assert a.type == AlertType.iron_condor
    assert a.direction == Direction.neutral
    assert len(a.legs) == 4


@test("from_opportunity: confidence mapping")
def _():
    opp = {
        "ticker": "X", "type": "bull_put_spread", "expiration": "2025-07-18",
        "short_strike": 10.0, "long_strike": 5.0, "credit": 0.50,
        "stop_loss": 1.0, "profit_target": 0.25,
    }
    opp["score"] = 85
    assert Alert.from_opportunity(opp).confidence == Confidence.HIGH
    opp["score"] = 65
    assert Alert.from_opportunity(opp).confidence == Confidence.MEDIUM
    opp["score"] = 40
    assert Alert.from_opportunity(opp).confidence == Confidence.SPECULATIVE


@test("to_dict: correct types")
def _():
    d = _make_alert().to_dict()
    assert d["type"] == "credit_spread"
    assert d["direction"] == "bullish"
    assert isinstance(d["created_at"], str)
    assert len(d["legs"]) == 2


# ===== RISK GATE TESTS =====
print("\n--- Risk Gate ---")


@test("constants match MASTERPLAN")
def _():
    assert MAX_RISK_PER_TRADE == 0.05
    assert MAX_TOTAL_EXPOSURE == 0.15
    assert DAILY_LOSS_LIMIT == 0.08
    assert WEEKLY_LOSS_LIMIT == 0.15
    assert MAX_CORRELATED_POSITIONS == 3
    assert COOLDOWN_AFTER_STOP == 1800


@test("rule 1: per-trade risk within limit")
def _():
    ok, _ = RiskGate().check(_make_alert(risk_pct=0.03), _clean_state())
    assert ok is True


@test("rule 1: per-trade risk at limit")
def _():
    ok, _ = RiskGate().check(_make_alert(risk_pct=0.05), _clean_state())
    assert ok is True


@test("rule 1: per-trade risk above limit")
def _():
    alert = _make_alert(risk_pct=0.05)
    object.__setattr__(alert, "risk_pct", 0.06)
    ok, reason = RiskGate().check(alert, _clean_state())
    assert ok is False
    assert "exceeds" in reason.lower()


@test("rule 2: total exposure within limit")
def _():
    state = _clean_state(open_positions=[
        {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.05},
    ])
    ok, _ = RiskGate().check(_make_alert(risk_pct=0.05), state)
    assert ok is True  # 5% + 5% = 10%


@test("rule 2: total exposure at limit")
def _():
    state = _clean_state(open_positions=[
        {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.10},
    ])
    ok, _ = RiskGate().check(_make_alert(risk_pct=0.05), state)
    assert ok is True  # 10% + 5% = 15% == limit


@test("rule 2: total exposure over limit")
def _():
    state = _clean_state(open_positions=[
        {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.05},
        {"ticker": "IWM", "direction": "bullish", "risk_pct": 0.05},
        {"ticker": "AAPL", "direction": "bearish", "risk_pct": 0.03},
    ])
    ok, reason = RiskGate().check(_make_alert(risk_pct=0.03), state)
    assert ok is False
    assert "exposure" in reason.lower()


@test("rule 3: daily loss OK")
def _():
    ok, _ = RiskGate().check(_make_alert(), _clean_state(daily_pnl_pct=-0.07))
    assert ok is True


@test("rule 3: daily loss at limit (allowed)")
def _():
    ok, _ = RiskGate().check(_make_alert(), _clean_state(daily_pnl_pct=-DAILY_LOSS_LIMIT))
    assert ok is True


@test("rule 3: daily loss breached")
def _():
    ok, reason = RiskGate().check(_make_alert(), _clean_state(daily_pnl_pct=-0.081))
    assert ok is False
    assert "daily" in reason.lower()


@test("rule 4: weekly loss does NOT block")
def _():
    ok, _ = RiskGate().check(_make_alert(), _clean_state(weekly_pnl_pct=-0.20))
    assert ok is True


@test("rule 4: weekly_loss_breach flag")
def _():
    gate = RiskGate()
    assert gate.weekly_loss_breach(_clean_state(weekly_pnl_pct=-0.10)) is False
    assert gate.weekly_loss_breach(_clean_state(weekly_pnl_pct=-WEEKLY_LOSS_LIMIT)) is False
    assert gate.weekly_loss_breach(_clean_state(weekly_pnl_pct=-0.16)) is True


@test("rule 5: correlated positions below limit")
def _():
    state = _clean_state(open_positions=[
        {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.02},
        {"ticker": "IWM", "direction": "bullish", "risk_pct": 0.02},
    ])
    ok, _ = RiskGate().check(_make_alert(direction=Direction.bullish), state)
    assert ok is True


@test("rule 5: correlated positions at limit")
def _():
    state = _clean_state(open_positions=[
        {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.02},
        {"ticker": "IWM", "direction": "bullish", "risk_pct": 0.02},
        {"ticker": "AAPL", "direction": "bullish", "risk_pct": 0.02},
    ])
    ok, reason = RiskGate().check(_make_alert(direction=Direction.bullish), state)
    assert ok is False
    assert "positions" in reason.lower()


@test("rule 5: different direction OK")
def _():
    state = _clean_state(open_positions=[
        {"ticker": "QQQ", "direction": "bullish", "risk_pct": 0.02},
        {"ticker": "IWM", "direction": "bullish", "risk_pct": 0.02},
        {"ticker": "AAPL", "direction": "bullish", "risk_pct": 0.02},
    ])
    ok, _ = RiskGate().check(_make_alert(direction=Direction.bearish), state)
    assert ok is True


@test("rule 6: no recent stops → OK")
def _():
    ok, _ = RiskGate().check(_make_alert(), _clean_state())
    assert ok is True


@test("rule 6: within cooldown → blocked")
def _():
    stopped_at = datetime.now(timezone.utc) - timedelta(seconds=COOLDOWN_AFTER_STOP - 60)
    state = _clean_state(recent_stops=[{"ticker": "SPY", "stopped_at": stopped_at}])
    ok, reason = RiskGate().check(_make_alert(ticker="SPY"), state)
    assert ok is False
    assert "cooldown" in reason.lower()


@test("rule 6: after cooldown → OK")
def _():
    stopped_at = datetime.now(timezone.utc) - timedelta(seconds=COOLDOWN_AFTER_STOP + 60)
    state = _clean_state(recent_stops=[{"ticker": "SPY", "stopped_at": stopped_at}])
    ok, _ = RiskGate().check(_make_alert(ticker="SPY"), state)
    assert ok is True


@test("rule 6: different ticker unaffected")
def _():
    stopped_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    state = _clean_state(recent_stops=[{"ticker": "QQQ", "stopped_at": stopped_at}])
    ok, _ = RiskGate().check(_make_alert(ticker="SPY"), state)
    assert ok is True


@test("rule 6: ISO string stopped_at handled")
def _():
    stopped_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    state = _clean_state(recent_stops=[{"ticker": "SPY", "stopped_at": stopped_at}])
    ok, _ = RiskGate().check(_make_alert(ticker="SPY"), state)
    assert ok is False


@test("no bypass: RiskGate has no config attr")
def _():
    gate = RiskGate()
    assert not hasattr(gate, "config")


# ===== POSITION SIZER TESTS =====
print("\n--- Alert Position Sizer ---")


@test("sizing: basic (100K, IVR 30)")
def _():
    sizer = AlertPositionSizer()
    result = sizer.size(
        alert=_make_alert(entry_price=1.50),
        account_value=100_000,
        iv_rank=30,
        current_portfolio_risk=0,
    )
    assert isinstance(result, SizeResult)
    assert result.contracts >= 0
    assert result.dollar_risk > 0
    # 2% of 100K = 2000, spread width 5, credit 1.50, max_loss/contract = 350
    # contracts = floor(2000/350) = 5 (capped at 5)
    assert result.contracts == 5


@test("sizing: low IV reduces contracts")
def _():
    sizer = AlertPositionSizer()
    result = sizer.size(
        alert=_make_alert(),
        account_value=100_000,
        iv_rank=10,
        current_portfolio_risk=0,
    )
    # 1% of 100K = 1000, 1000/350 = 2
    assert result.contracts == 2


@test("sizing: 5% hard cap enforced")
def _():
    sizer = AlertPositionSizer()
    result = sizer.size(
        alert=_make_alert(),
        account_value=10_000,
        iv_rank=100,
        current_portfolio_risk=0,
    )
    assert result.risk_pct <= MAX_RISK_PER_TRADE
    assert result.dollar_risk <= MAX_RISK_PER_TRADE * 10_000


@test("sizing: weekly loss 50% reduction")
def _():
    sizer = AlertPositionSizer()
    normal = sizer.size(_make_alert(), 100_000, 30, 0, weekly_loss_breach=False)
    reduced = sizer.size(_make_alert(), 100_000, 30, 0, weekly_loss_breach=True)
    assert abs(reduced.dollar_risk - normal.dollar_risk * 0.5) < 0.01
    assert reduced.contracts <= normal.contracts


@test("sizing: tiny account → 0 contracts")
def _():
    sizer = AlertPositionSizer()
    result = sizer.size(_make_alert(), 100, 30, 0)
    assert result.contracts == 0
    assert result.max_loss == 0


@test("sizing: iron condor spread width")
def _():
    legs = [
        Leg(95.0, "put", "buy", "2025-06-20"),
        Leg(100.0, "put", "sell", "2025-06-20"),
        Leg(110.0, "call", "sell", "2025-06-20"),
        Leg(115.0, "call", "buy", "2025-06-20"),
    ]
    alert = _make_alert(legs=legs)
    sizer = AlertPositionSizer()
    width, credit = sizer._extract_spread_params(alert)
    assert width == 5.0


# ===== TELEGRAM FORMATTER TESTS =====
print("\n--- Telegram Formatter ---")


@test("entry: contains ticker")
def _():
    fmt = TelegramAlertFormatter()
    msg = fmt.format_entry_alert(_make_alert(score=78))
    assert "SPY" in msg


@test("entry: contains type label")
def _():
    fmt = TelegramAlertFormatter()
    msg = fmt.format_entry_alert(_make_alert())
    assert "CREDIT SPREAD" in msg


@test("entry: type emojis correct")
def _():
    fmt = TelegramAlertFormatter()
    expected = {
        AlertType.credit_spread: "\U0001f7e2",
        AlertType.momentum_swing: "\U0001f535",
        AlertType.iron_condor: "\U0001f7e1",
        AlertType.earnings_play: "\U0001f7e0",
        AlertType.gamma_lotto: "\U0001f534",
    }
    for atype, emoji in expected.items():
        msg = fmt.format_entry_alert(_make_alert(type=atype))
        assert emoji in msg, f"Missing emoji for {atype.value}"


@test("entry: contains all 8 MASTERPLAN elements")
def _():
    fmt = TelegramAlertFormatter()
    alert = _make_alert(
        score=78,
        thesis="Test thesis here",
        management_instructions="Close at 50%",
        time_sensitivity=TimeSensitivity.TODAY,
        confidence=Confidence.HIGH,
    )
    msg = fmt.format_entry_alert(alert)
    assert "BULLISH" in msg                # direction
    assert "SELL $100.00 PUT" in msg       # legs
    assert "$1.50" in msg                  # entry price
    assert "$3.00" in msg                  # stop loss
    assert "$0.75" in msg                  # profit target
    assert "2.0%" in msg                   # risk
    assert "Test thesis here" in msg       # thesis
    assert "Close at 50%" in msg           # management
    assert "TODAY" in msg                  # time sensitivity
    assert "HIGH" in msg                   # confidence


@test("entry: sizing shown when present")
def _():
    fmt = TelegramAlertFormatter()
    alert = _make_alert()
    alert.sizing = SizeResult(risk_pct=0.02, contracts=3, dollar_risk=2000, max_loss=1050)
    msg = fmt.format_entry_alert(alert)
    assert "Contracts: 3" in msg
    assert "$1050.00" in msg


@test("exit: positive PnL format")
def _():
    fmt = TelegramAlertFormatter()
    msg = fmt.format_exit_alert("SPY", "CLOSE", 200.0, 50.0, "Target", "None")
    assert "+$200.00" in msg
    assert "+50.0%" in msg


@test("exit: negative PnL format")
def _():
    fmt = TelegramAlertFormatter()
    msg = fmt.format_exit_alert("SPY", "STOP", -300.0, -75.0, "Stop", "Review")
    assert "-$300.00" in msg


@test("summary: contains all fields")
def _():
    fmt = TelegramAlertFormatter()
    msg = fmt.format_daily_summary(
        date="2025-06-20", alerts_fired=8, closed_today=3, wins=2, losses=1,
        day_pnl=450.0, day_pnl_pct=0.9, open_positions=5, total_risk_pct=8.5,
        account_balance=50_450.0, pct_from_start=0.9, best="SPY +$200", worst="AAPL -$50",
    )
    assert "2025-06-20" in msg
    assert "W:2" in msg
    assert "L:1" in msg
    assert "$450.00" in msg
    assert "$50,450.00" in msg


# ===== RESULTS =====
print(f"\n{'='*60}")
print(f"Phase 1 tests: {_pass} passed, {_fail} failed")
if _errors:
    print("\nFailed tests:")
    for name, err in _errors:
        print(f"  {name}: {err}")
print(f"{'='*60}")

sys.exit(1 if _fail else 0)
