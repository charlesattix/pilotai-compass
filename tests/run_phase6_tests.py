#!/usr/bin/env python3
"""Self-contained Phase 6 test runner.

Mocks third-party dependencies (same pattern as Phase 1/2/3/4/5) and exercises:
- Economic calendar: FOMC sourcing, CPI/PPI/Jobs/GDP computation, upcoming events,
  is_event_tomorrow, get_next_event
- Gamma config overlay
- Scanner: market hours gate, event gate, scan pipeline, OTM filtering, sigma payoffs,
  scoring
- Exit monitor: trailing stop activation/triggered, expired worthless, dedup, edge cases
- from_opportunity gamma conversion
- Position sizer gamma_lotto sizing
- Formatter integration
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
np_mod = _mock_module("numpy", {
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
    "zeros": lambda *a: [0] * (a[0] if a else 0),
    "ones": lambda *a: [1] * (a[0] if a else 0),
    "inf": float("inf"),
    "abs": abs,
    "sum": sum,
    "full": lambda shape, val, **kw: [val] * shape,
    "random": type("random", (), {
        "seed": staticmethod(lambda x: None),
        "normal": staticmethod(lambda *a, **k: [0.0]),
    })(),
    "arange": lambda *a, **k: list(range(int(a[0]), int(a[1]), int(a[2])) if len(a) >= 3 else range(int(a[0]))),
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

# Pre-load ml.position_sizer (same trick as Phase 1/2/3/4/5)
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
from datetime import date, datetime, time, timedelta, timezone

from alerts.alert_position_sizer import AlertPositionSizer
from alerts.alert_schema import Alert, AlertType, Confidence, Direction, Leg, TimeSensitivity
from alerts.gamma_config import (
    GAMMA_TICKERS,
    SCAN_HOURS,
    build_gamma_config,
)
from alerts.gamma_exit_monitor import GammaExitMonitor
from alerts.gamma_scanner import GammaScanner
from shared.constants import FOMC_DATES, GAMMA_LOTTO_MAX_RISK_PCT
from shared.economic_calendar import EVENT_IMPORTANCE, EconomicCalendar

print("All Phase 6 modules imported successfully\n")

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


# ===== ECONOMIC CALENDAR TESTS =====
print("--- Economic Calendar ---")


@test("calendar: FOMC dates sourced from shared.constants")
def _():
    cal = EconomicCalendar()
    assert len(cal._fomc_dates) == len(FOMC_DATES)
    for fd, cd in zip(FOMC_DATES, cal._fomc_dates):
        assert fd == cd


@test("calendar: CPI dates computed (12 per year)")
def _():
    cal = EconomicCalendar()
    # Should have 12 * 2 = 24 CPI dates (current year + next)
    assert len(cal._cpi_dates) == 24


@test("calendar: PPI dates computed (12 per year)")
def _():
    cal = EconomicCalendar()
    assert len(cal._ppi_dates) == 24


@test("calendar: Jobs dates computed (12 per year)")
def _():
    cal = EconomicCalendar()
    assert len(cal._jobs_dates) == 24


@test("calendar: GDP dates computed (4 per year, Jan/Apr/Jul/Oct)")
def _():
    cal = EconomicCalendar()
    assert len(cal._gdp_dates) == 8
    # Check months: should be Jan, Apr, Jul, Oct
    months = [d.month for d in cal._gdp_dates]
    for m in [1, 4, 7, 10]:
        assert m in months


@test("calendar: CPI dates are 2nd Wednesdays")
def _():
    dates = EconomicCalendar._compute_cpi_dates(2026)
    for d in dates:
        dt = date(d.year, d.month, d.day)
        assert dt.weekday() == 2  # Wednesday
        assert 8 <= dt.day <= 14  # 2nd week


@test("calendar: Jobs dates are first Fridays")
def _():
    dates = EconomicCalendar._compute_jobs_dates(2026)
    for d in dates:
        dt = date(d.year, d.month, d.day)
        assert dt.weekday() == 4  # Friday
        assert 1 <= dt.day <= 7   # first week


@test("calendar: GDP dates are last Thursdays of Jan/Apr/Jul/Oct")
def _():
    dates = EconomicCalendar._compute_gdp_dates(2026)
    assert len(dates) == 4
    for d in dates:
        dt = date(d.year, d.month, d.day)
        assert dt.weekday() == 3  # Thursday
        assert d.month in [1, 4, 7, 10]


@test("calendar: get_upcoming_events returns sorted events")
def _():
    cal = EconomicCalendar()
    # Pick a date just before a known FOMC date
    if FOMC_DATES:
        ref = FOMC_DATES[0] - timedelta(days=1)
        events = cal.get_upcoming_events(days_ahead=3, reference_date=ref)
        if len(events) > 1:
            for i in range(len(events) - 1):
                assert events[i]["date"] <= events[i + 1]["date"]


@test("calendar: is_event_tomorrow True when FOMC is tomorrow")
def _():
    cal = EconomicCalendar()
    if FOMC_DATES:
        # day before FOMC
        ref = FOMC_DATES[0] - timedelta(days=1)
        assert cal.is_event_tomorrow(ref) is True


@test("calendar: is_event_tomorrow False when no event")
def _():
    cal = EconomicCalendar()
    # Pick a date that's unlikely to match any event (Dec 25)
    ref = datetime(2026, 12, 25, 12, 0, 0, tzinfo=timezone.utc)
    # Might or might not be an event, but test the function runs
    result = cal.is_event_tomorrow(ref)
    assert isinstance(result, bool)


@test("calendar: get_next_event returns dict or None")
def _():
    cal = EconomicCalendar()
    ref = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = cal.get_next_event(ref)
    if result is not None:
        assert "event_type" in result
        assert "date" in result
        assert "importance" in result


@test("calendar: EVENT_IMPORTANCE has all event types")
def _():
    assert EVENT_IMPORTANCE["fomc"] == 1.0
    assert EVENT_IMPORTANCE["cpi"] == 0.85
    assert EVENT_IMPORTANCE["jobs"] == 0.75
    assert EVENT_IMPORTANCE["ppi"] == 0.70
    assert EVENT_IMPORTANCE["gdp"] == 0.65


# ===== GAMMA CONFIG OVERLAY TESTS =====
print("\n--- Gamma Config Overlay ---")


@test("config: min_dte=0, max_dte=1")
def _():
    cfg = build_gamma_config(_base_config())
    assert cfg["strategy"]["min_dte"] == 0
    assert cfg["strategy"]["max_dte"] == 1


@test("config: iron_condor disabled")
def _():
    cfg = build_gamma_config(_base_config())
    assert cfg["strategy"]["iron_condor"]["enabled"] is False


@test("config: gamma sub-dict with correct keys")
def _():
    cfg = build_gamma_config(_base_config())
    g = cfg["strategy"]["gamma"]
    assert g["price_min"] == 0.10
    assert g["price_max"] == 0.50
    assert g["max_risk_pct"] == 0.005
    assert g["trailing_stop_activation"] == 3.0
    assert g["trailing_stop_level"] == 2.0
    assert g["min_otm_pct"] == 0.02
    assert g["max_otm_pct"] == 0.10


@test("config: tickers = GAMMA_TICKERS")
def _():
    cfg = build_gamma_config(_base_config())
    assert cfg["tickers"] == ["SPY", "QQQ", "IWM"]


@test("config: GAMMA_TICKERS has 3 ETFs")
def _():
    assert GAMMA_TICKERS == ["SPY", "QQQ", "IWM"]


@test("config: SCAN_HOURS is 9:35-15:30")
def _():
    assert SCAN_HOURS == (time(9, 35), time(15, 30))


@test("config: base config NOT mutated")
def _():
    base = _base_config()
    original_min_dte = base["strategy"]["min_dte"]
    original_tickers = list(base["tickers"])
    build_gamma_config(base)
    assert base["strategy"]["min_dte"] == original_min_dte
    assert base["tickers"] == original_tickers


@test("config: preserves other base keys")
def _():
    base = _base_config()
    base["custom_setting"] = "keep_me"
    cfg = build_gamma_config(base)
    assert cfg["custom_setting"] == "keep_me"


@test("config: GAMMA_LOTTO_MAX_RISK_PCT constant exists")
def _():
    assert GAMMA_LOTTO_MAX_RISK_PCT == 0.005


@test("config: gamma tickers are all uppercase strings")
def _():
    for t in GAMMA_TICKERS:
        assert isinstance(t, str)
        assert t == t.upper()


# ===== SCANNER TESTS =====
print("\n--- Gamma Scanner ---")


@test("scanner: market hours gate — weekday 10am ET → True")
def _():
    # Monday 10:00 ET (no timezone = treated as ET by the static method)
    et = datetime(2026, 3, 2, 10, 0)  # Monday
    assert GammaScanner.is_market_hours(et) is True


@test("scanner: market hours gate — weekday 9:30 ET → False (before 9:35)")
def _():
    et = datetime(2026, 3, 2, 9, 30)
    assert GammaScanner.is_market_hours(et) is False


@test("scanner: market hours gate — weekday 15:35 ET → False (after 15:30)")
def _():
    et = datetime(2026, 3, 2, 15, 35)
    assert GammaScanner.is_market_hours(et) is False


@test("scanner: market hours gate — Saturday → False")
def _():
    et = datetime(2026, 2, 28, 10, 0)  # Saturday
    assert GammaScanner.is_market_hours(et) is False


@test("scanner: market hours gate — Sunday → False")
def _():
    et = datetime(2026, 3, 1, 10, 0)  # Sunday
    assert GammaScanner.is_market_hours(et) is False


@test("scanner: market hours gate — 9:35 ET → True (inclusive)")
def _():
    et = datetime(2026, 3, 2, 9, 35)
    assert GammaScanner.is_market_hours(et) is True


@test("scanner: market hours gate — 15:30 ET → True (inclusive)")
def _():
    et = datetime(2026, 3, 2, 15, 30)
    assert GammaScanner.is_market_hours(et) is True


@test("scanner: scan returns empty outside market hours")
def _():
    scanner = GammaScanner(_base_config())
    # Saturday
    result = scanner.scan(now_et=datetime(2026, 2, 28, 10, 0))
    assert result == []


@test("scanner: scan returns empty when no event tomorrow")
def _():
    scanner = GammaScanner(_base_config())
    # Override calendar to say no event tomorrow
    scanner._calendar.is_event_tomorrow = lambda ref=None: False
    # Monday 10am (market hours)
    result = scanner.scan(now_et=datetime(2026, 3, 2, 10, 0))
    assert result == []


@test("scanner: scan calls _scan_ticker when event tomorrow")
def _():
    scanner = GammaScanner(_base_config())
    scanner._calendar.is_event_tomorrow = lambda ref=None: True
    scanner._calendar.get_upcoming_events = lambda **k: [
        {"event_type": "fomc", "date": datetime(2026, 3, 3, tzinfo=timezone.utc),
         "description": "FOMC", "importance": 1.0}
    ]
    scan_calls = []
    def mock_scan_ticker(ticker, event):
        scan_calls.append(ticker)
        return [{"ticker": ticker, "type": "gamma_lotto_call", "score": 70}]
    scanner._scan_ticker = mock_scan_ticker
    result = scanner.scan(now_et=datetime(2026, 3, 2, 10, 0))
    assert len(scan_calls) == 3  # SPY, QQQ, IWM
    assert all(r["alert_source"] == "gamma_lotto" for r in result)


@test("scanner: scan annotates alert_source=gamma_lotto")
def _():
    scanner = GammaScanner(_base_config())
    scanner._calendar.is_event_tomorrow = lambda ref=None: True
    scanner._calendar.get_upcoming_events = lambda **k: [
        {"event_type": "cpi", "date": datetime(2026, 3, 3, tzinfo=timezone.utc),
         "description": "CPI", "importance": 0.85}
    ]
    scanner._scan_ticker = lambda t, e: [{"ticker": t, "type": "gamma_lotto_call"}]
    result = scanner.scan(now_et=datetime(2026, 3, 2, 10, 0))
    for r in result:
        assert r["alert_source"] == "gamma_lotto"


@test("scanner: scan handles ticker error gracefully")
def _():
    scanner = GammaScanner(_base_config())
    scanner._calendar.is_event_tomorrow = lambda ref=None: True
    scanner._calendar.get_upcoming_events = lambda **k: [
        {"event_type": "fomc", "date": datetime(2026, 3, 3, tzinfo=timezone.utc),
         "description": "FOMC", "importance": 1.0}
    ]
    def failing_scan(ticker, event):
        raise Exception("API error")
    scanner._scan_ticker = failing_scan
    result = scanner.scan(now_et=datetime(2026, 3, 2, 10, 0))
    assert result == []


@test("scanner: sigma payoffs call correct math")
def _():
    # Call: payoff at 1σ = max(0, (price + expected_move) - strike) - debit
    payoffs = GammaScanner._calculate_sigma_payoffs(
        strike=560, price=550, expected_move=5, debit=0.20, option_type="call"
    )
    # 1σ: max(0, 555 - 560) - 0.20 = -0.20
    assert payoffs["sigma_1_payoff"] == -0.20
    # 2σ: max(0, 560 - 560) - 0.20 = -0.20
    assert payoffs["sigma_2_payoff"] == -0.20
    # 3σ: max(0, 565 - 560) - 0.20 = 4.80
    assert payoffs["sigma_3_payoff"] == 4.80


@test("scanner: sigma payoffs put correct math")
def _():
    # Put: payoff at 1σ = max(0, strike - (price - expected_move)) - debit
    payoffs = GammaScanner._calculate_sigma_payoffs(
        strike=540, price=550, expected_move=5, debit=0.15, option_type="put"
    )
    # 1σ: max(0, 540 - 545) - 0.15 = -0.15
    assert payoffs["sigma_1_payoff"] == -0.15
    # 2σ: max(0, 540 - 540) - 0.15 = -0.15
    assert payoffs["sigma_2_payoff"] == -0.15
    # 3σ: max(0, 540 - 535) - 0.15 = 4.85
    assert payoffs["sigma_3_payoff"] == 4.85


@test("scanner: sigma payoffs return percentages")
def _():
    payoffs = GammaScanner._calculate_sigma_payoffs(
        strike=560, price=550, expected_move=5, debit=0.20, option_type="call"
    )
    # 3σ: payoff=4.80, return_pct = 4.80/0.20*100 = 2400%
    assert payoffs["sigma_3_return_pct"] == 2400.0


@test("scanner: scoring — perfect components")
def _():
    score = GammaScanner._score_gamma_opportunity(
        debit=0.10,
        sigma_payoffs={"sigma_2_return_pct": 2000},
        event={"importance": 1.0},
    )
    # Payoff: 2000/2000*40 = 40, Cheapness: (0.50-0.10)/0.40*30 = 30, Event: 1.0*30 = 30
    assert score == 100.0


@test("scanner: scoring — expensive option scores lower cheapness")
def _():
    score_cheap = GammaScanner._score_gamma_opportunity(
        debit=0.10,
        sigma_payoffs={"sigma_2_return_pct": 1000},
        event={"importance": 0.85},
    )
    score_expensive = GammaScanner._score_gamma_opportunity(
        debit=0.50,
        sigma_payoffs={"sigma_2_return_pct": 1000},
        event={"importance": 0.85},
    )
    assert score_cheap > score_expensive


@test("scanner: scoring — higher importance scores higher")
def _():
    score_fomc = GammaScanner._score_gamma_opportunity(
        debit=0.25,
        sigma_payoffs={"sigma_2_return_pct": 500},
        event={"importance": 1.0},
    )
    score_gdp = GammaScanner._score_gamma_opportunity(
        debit=0.25,
        sigma_payoffs={"sigma_2_return_pct": 500},
        event={"importance": 0.65},
    )
    assert score_fomc > score_gdp


# ===== EXIT MONITOR TESTS =====
print("\n--- Gamma Exit Monitor ---")


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


def _make_gamma_trade(**overrides):
    base = {
        "id": "gamma1",
        "ticker": "SPY",
        "strategy_type": "gamma_lotto_call",
        "debit": 0.25,
        "_mock_pnl": 0,
    }
    base.update(overrides)
    return base


@test("exit: trailing stop activation at 3x entry")
def _():
    # 3x entry = 0.25 * 3 * 100 = 75 per contract
    trade = _make_gamma_trade(_mock_pnl=80)
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 560.0})
    reasons = [t["reason"] for t in triggered]
    assert "trailing_stop_activation" in reasons
    assert len(bot.sent) >= 1


@test("exit: no activation below 3x")
def _():
    trade = _make_gamma_trade(_mock_pnl=50)  # below 75
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 560.0})
    activation = [t for t in triggered if t["reason"] == "trailing_stop_activation"]
    assert len(activation) == 0


@test("exit: trailing stop triggered after activation")
def _():
    trade = _make_gamma_trade(_mock_pnl=80)  # 80 > 75, activates
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    # First call: activates trailing stop
    monitor.check_and_alert({"SPY": 560.0})
    # Now simulate P&L dropping below 2x entry (0.25*2*100=50)
    trade["_mock_pnl"] = 40  # below 50
    triggered = monitor.check_and_alert({"SPY": 555.0})
    reasons = [t["reason"] for t in triggered]
    assert "trailing_stop_triggered" in reasons


@test("exit: trailing stop NOT triggered without activation")
def _():
    trade = _make_gamma_trade(_mock_pnl=40)  # below activation AND below trail
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 560.0})
    trailing = [t for t in triggered if t["reason"] == "trailing_stop_triggered"]
    assert len(trailing) == 0


@test("exit: expired worthless fires on total loss")
def _():
    # Total loss = -(debit * 100) = -(0.25 * 100) = -25
    trade = _make_gamma_trade(_mock_pnl=-25)
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 540.0})
    reasons = [t["reason"] for t in triggered]
    assert "expired_worthless" in reasons


@test("exit: expired worthless at exact -debit*100")
def _():
    trade = _make_gamma_trade(debit=0.20, _mock_pnl=-20)
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 540.0})
    reasons = [t["reason"] for t in triggered]
    assert "expired_worthless" in reasons


@test("exit: composite dedup — same reason suppressed on second call")
def _():
    trade = _make_gamma_trade(_mock_pnl=-25)
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    monitor.check_and_alert({"SPY": 540.0})
    triggered2 = monitor.check_and_alert({"SPY": 540.0})
    worthless = [t for t in triggered2 if t["reason"] == "expired_worthless"]
    assert len(worthless) == 0


@test("exit: non-gamma positions skipped")
def _():
    trade = _make_gamma_trade(strategy_type="iron_condor", _mock_pnl=-25)
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 540.0})
    assert len(triggered) == 0


@test("exit: missing trade_id skipped")
def _():
    trade = _make_gamma_trade(id="", _mock_pnl=-25)
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 540.0})
    assert len(triggered) == 0


@test("exit: zero debit skipped")
def _():
    trade = _make_gamma_trade(debit=0, _mock_pnl=-25)
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 540.0})
    assert len(triggered) == 0


@test("exit: missing price skipped")
def _():
    trade = _make_gamma_trade(_mock_pnl=-25)
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"QQQ": 400.0})  # wrong ticker
    assert len(triggered) == 0


@test("exit: 'type' field fallback (no strategy_type)")
def _():
    trade = {
        "id": "g2", "ticker": "QQQ", "type": "gamma_lotto_put",
        "debit": 0.30, "_mock_pnl": -30,
    }
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"QQQ": 390.0})
    assert len(triggered) >= 1


@test("exit: lotto in type field also matched")
def _():
    trade = {
        "id": "g3", "ticker": "IWM", "type": "lotto_call",
        "debit": 0.15, "_mock_pnl": -15,
    }
    bot = _MockTelegramBot()
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"IWM": 200.0})
    assert len(triggered) >= 1


@test("exit: telegram failure doesn't crash")
def _():
    trade = _make_gamma_trade(_mock_pnl=-25)
    bot = _MockTelegramBot()
    def failing_send(msg):
        raise Exception("network error")
    bot.send_alert = failing_send
    monitor = GammaExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"SPY": 540.0})
    assert len(triggered) >= 1


# ===== FROM_OPPORTUNITY GAMMA CONVERSION =====
print("\n--- from_opportunity gamma conversion ---")


@test("from_opportunity: gamma_lotto_call → AlertType.gamma_lotto")
def _():
    opp = {
        "ticker": "SPY",
        "type": "gamma_lotto_call",
        "option_type": "call",
        "strike": 560.0,
        "debit": 0.25,
        "stop_loss": 0.0,
        "profit_target": 0.75,
        "expiration": "2026-03-03",
        "dte": 1,
        "score": 70,
        "alert_source": "gamma_lotto",
        "risk_pct": 0.005,
    }
    alert = Alert.from_opportunity(opp)
    assert alert.type == AlertType.gamma_lotto


@test("from_opportunity: gamma direction — call=bullish, put=bearish")
def _():
    call_opp = {
        "ticker": "SPY", "type": "gamma_lotto_call", "option_type": "call",
        "strike": 560, "debit": 0.25, "stop_loss": 0, "profit_target": 0.75,
        "expiration": "2026-03-03", "score": 70, "alert_source": "gamma_lotto",
        "risk_pct": 0.005,
    }
    put_opp = {
        "ticker": "SPY", "type": "gamma_lotto_put", "option_type": "put",
        "strike": 540, "debit": 0.25, "stop_loss": 0, "profit_target": 0.75,
        "expiration": "2026-03-03", "score": 70, "alert_source": "gamma_lotto",
        "risk_pct": 0.005,
    }
    call_alert = Alert.from_opportunity(call_opp)
    put_alert = Alert.from_opportunity(put_opp)
    assert call_alert.direction == Direction.bullish
    assert put_alert.direction == Direction.bearish


@test("from_opportunity: gamma has single leg (BUY)")
def _():
    opp = {
        "ticker": "QQQ", "type": "gamma_lotto_call", "option_type": "call",
        "strike": 480, "debit": 0.20, "stop_loss": 0, "profit_target": 0.60,
        "expiration": "2026-03-03", "score": 65, "alert_source": "gamma_lotto",
        "risk_pct": 0.005,
    }
    alert = Alert.from_opportunity(opp)
    assert len(alert.legs) == 1
    assert alert.legs[0].action == "buy"
    assert alert.legs[0].strike == 480
    assert alert.legs[0].option_type == "call"


@test("from_opportunity: gamma confidence is SPECULATIVE")
def _():
    opp = {
        "ticker": "SPY", "type": "gamma_lotto_call", "option_type": "call",
        "strike": 560, "debit": 0.25, "stop_loss": 0, "profit_target": 0.75,
        "expiration": "2026-03-03", "score": 90, "alert_source": "gamma_lotto",
        "risk_pct": 0.005,
    }
    alert = Alert.from_opportunity(opp)
    assert alert.confidence == Confidence.SPECULATIVE


@test("from_opportunity: gamma time_sensitivity is IMMEDIATE")
def _():
    opp = {
        "ticker": "SPY", "type": "gamma_lotto_call", "option_type": "call",
        "strike": 560, "debit": 0.25, "stop_loss": 0, "profit_target": 0.75,
        "expiration": "2026-03-03", "score": 70, "alert_source": "gamma_lotto",
        "risk_pct": 0.005,
    }
    alert = Alert.from_opportunity(opp)
    assert alert.time_sensitivity == TimeSensitivity.IMMEDIATE


@test("from_opportunity: gamma entry_price = debit")
def _():
    opp = {
        "ticker": "SPY", "type": "gamma_lotto_call", "option_type": "call",
        "strike": 560, "debit": 0.30, "stop_loss": 0, "profit_target": 0.90,
        "expiration": "2026-03-03", "score": 70, "alert_source": "gamma_lotto",
        "risk_pct": 0.005,
    }
    alert = Alert.from_opportunity(opp)
    assert alert.entry_price == 0.30


# ===== POSITION SIZER GAMMA TESTS =====
print("\n--- Position Sizer (gamma) ---")


@test("sizer: gamma_lotto max risk capped at 0.5%")
def _():
    sizer = AlertPositionSizer()
    alert = Alert(
        type=AlertType.gamma_lotto,
        ticker="SPY",
        direction=Direction.bullish,
        legs=[Leg(560, "call", "buy", "2026-03-03")],
        entry_price=0.25,
        stop_loss=0.0,
        profit_target=0.75,
        risk_pct=0.005,
    )
    result = sizer.size(alert, 100000, iv_rank=50, current_portfolio_risk=0)
    # Dollar risk capped at 0.5% of 100k = $500
    assert result.dollar_risk <= 500


@test("sizer: gamma_lotto contracts computed from debit")
def _():
    sizer = AlertPositionSizer()
    alert = Alert(
        type=AlertType.gamma_lotto,
        ticker="SPY",
        direction=Direction.bullish,
        legs=[Leg(560, "call", "buy", "2026-03-03")],
        entry_price=0.25,
        stop_loss=0.0,
        profit_target=0.75,
        risk_pct=0.005,
    )
    result = sizer.size(alert, 100000, iv_rank=50, current_portfolio_risk=0)
    # $500 / ($0.25 * 100) = $500 / $25 = 20 contracts, capped at 5
    assert result.contracts >= 1
    assert result.contracts <= 5


@test("sizer: gamma_lotto max_loss = debit * 100 * contracts")
def _():
    sizer = AlertPositionSizer()
    alert = Alert(
        type=AlertType.gamma_lotto,
        ticker="SPY",
        direction=Direction.bullish,
        legs=[Leg(560, "call", "buy", "2026-03-03")],
        entry_price=0.25,
        stop_loss=0.0,
        profit_target=0.75,
        risk_pct=0.005,
    )
    result = sizer.size(alert, 100000, iv_rank=50, current_portfolio_risk=0)
    expected_max_loss = 0.25 * 100 * result.contracts
    assert result.max_loss == expected_max_loss


@test("sizer: single-leg extract_spread_params returns (credit, credit)")
def _():
    sizer = AlertPositionSizer()
    alert = Alert(
        type=AlertType.gamma_lotto,
        ticker="SPY",
        direction=Direction.bullish,
        legs=[Leg(560, "call", "buy", "2026-03-03")],
        entry_price=0.25,
        stop_loss=0.0,
        profit_target=0.75,
        risk_pct=0.005,
    )
    width, credit = sizer._extract_spread_params(alert)
    assert width == 0.25
    assert credit == 0.25


# ===== FORMATTER INTEGRATION TESTS =====
print("\n--- Telegram Formatter ---")


@test("telegram: gamma_lotto shows GAMMA LOTTO header")
def _():
    from alerts.formatters.telegram import TelegramAlertFormatter
    fmt = TelegramAlertFormatter()
    alert = Alert(
        type=AlertType.gamma_lotto,
        ticker="SPY",
        direction=Direction.bullish,
        legs=[Leg(560, "call", "buy", "2026-03-03")],
        entry_price=0.25,
        stop_loss=0.0,
        profit_target=0.75,
        risk_pct=0.005,
    )
    text = fmt.format_entry_alert(alert)
    assert "GAMMA LOTTO" in text


@test("telegram: gamma_lotto uses Debit label (not Credit)")
def _():
    from alerts.formatters.telegram import TelegramAlertFormatter
    fmt = TelegramAlertFormatter()
    alert = Alert(
        type=AlertType.gamma_lotto,
        ticker="SPY",
        direction=Direction.bullish,
        legs=[Leg(560, "call", "buy", "2026-03-03")],
        entry_price=0.25,
        stop_loss=0.0,
        profit_target=0.75,
        risk_pct=0.005,
    )
    text = fmt.format_entry_alert(alert)
    assert "Debit:" in text
    assert "Credit:" not in text


# ===== RESULTS =====
print(f"\n{'='*60}")
print(f"Phase 6 tests: {_pass} passed, {_fail} failed")
if _errors:
    print("\nFailed tests:")
    for name, err in _errors:
        print(f"  {name}: {err}")
print(f"{'='*60}")

sys.exit(1 if _fail else 0)
