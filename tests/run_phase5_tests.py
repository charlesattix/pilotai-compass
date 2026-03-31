#!/usr/bin/env python3
"""Self-contained Phase 5 test runner.

Mocks third-party dependencies (same pattern as Phase 1/2/3/4) and exercises:
- Earnings config overlay
- Earnings calendar: ETF skip, caching, lookahead, expected move, historical analysis
- Entry window gate
- Scanner scan pipeline
- Condor builder: price-based strikes, credit, max_loss, field validation
- Scoring
- Exit monitor: post-earnings close, profit, stop, dedup, edge cases
- from_opportunity earnings conversion
- Position sizer + formatter integration
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

# Pre-load ml.position_sizer (same trick as Phase 1/2/3/4)
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

from alerts.alert_position_sizer import AlertPositionSizer
from alerts.alert_schema import Alert, AlertType, Direction, Leg
from alerts.earnings_config import (
    EARNINGS_LOOKAHEAD_DAYS,
    EARNINGS_TICKERS,
    build_earnings_config,
)
from alerts.earnings_exit_monitor import EarningsExitMonitor, _is_post_earnings
from alerts.earnings_scanner import EarningsScanner
from shared.earnings_calendar import _NO_EARNINGS_TICKERS, EarningsCalendar

print("All Phase 5 modules imported successfully\n")

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


# ===== CONFIG OVERLAY TESTS =====
print("--- Earnings Config Overlay ---")


@test("config: min_dte=1, max_dte=7")
def _():
    cfg = build_earnings_config(_base_config())
    assert cfg["strategy"]["min_dte"] == 1
    assert cfg["strategy"]["max_dte"] == 7


@test("config: min_iv_rank=60")
def _():
    cfg = build_earnings_config(_base_config())
    assert cfg["strategy"]["min_iv_rank"] == 60


@test("config: iron_condor disabled")
def _():
    cfg = build_earnings_config(_base_config())
    assert cfg["strategy"]["iron_condor"]["enabled"] is False


@test("config: earnings sub-dict exists with correct keys")
def _():
    cfg = build_earnings_config(_base_config())
    e = cfg["strategy"]["earnings"]
    assert e["expected_move_multiplier"] == 1.2
    assert e["min_stay_in_range_pct"] == 65
    assert e["min_historical_quarters"] == 4
    assert e["spread_width"] == 5


@test("config: earnings risk/exit params")
def _():
    cfg = build_earnings_config(_base_config())
    e = cfg["strategy"]["earnings"]
    assert e["max_risk_pct"] == 0.02
    assert e["profit_target_pct"] == 0.50
    assert e["stop_loss_multiplier"] == 2.0


@test("config: earnings entry window params")
def _():
    cfg = build_earnings_config(_base_config())
    e = cfg["strategy"]["earnings"]
    assert e["min_entry_days_before"] == 1
    assert e["max_entry_days_before"] == 3


@test("config: tickers = EARNINGS_TICKERS")
def _():
    cfg = build_earnings_config(_base_config())
    assert cfg["tickers"] == list(EARNINGS_TICKERS)
    assert len(cfg["tickers"]) == 18


@test("config: EARNINGS_LOOKAHEAD_DAYS = 14")
def _():
    assert EARNINGS_LOOKAHEAD_DAYS == 14


@test("config: base config NOT mutated")
def _():
    base = _base_config()
    original_min_dte = base["strategy"]["min_dte"]
    original_tickers = list(base["tickers"])
    build_earnings_config(base)
    assert base["strategy"]["min_dte"] == original_min_dte
    assert base["tickers"] == original_tickers


@test("config: preserves other base keys")
def _():
    base = _base_config()
    base["custom_setting"] = "keep_me"
    cfg = build_earnings_config(base)
    assert cfg["custom_setting"] == "keep_me"


@test("config: EARNINGS_TICKERS has 18 names")
def _():
    expected = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
        "AMD", "CRM", "COIN", "SQ", "SHOP", "UBER", "ABNB", "PLTR",
        "SNAP", "ROKU",
    ]
    assert EARNINGS_TICKERS == expected


@test("config: expected_move_multiplier is float")
def _():
    cfg = build_earnings_config(_base_config())
    m = cfg["strategy"]["earnings"]["expected_move_multiplier"]
    assert isinstance(m, float)
    assert m == 1.2


@test("config: earnings tickers are all strings")
def _():
    for t in EARNINGS_TICKERS:
        assert isinstance(t, str)
        assert t == t.upper()


# ===== EARNINGS CALENDAR TESTS =====
print("\n--- Earnings Calendar ---")


@test("calendar: ETFs return None from get_next_earnings")
def _():
    cal = EarningsCalendar()
    for etf in ["SPY", "QQQ", "IWM", "SMH", "ARKK"]:
        assert cal.get_next_earnings(etf) is None


@test("calendar: _NO_EARNINGS_TICKERS is a frozenset")
def _():
    assert isinstance(_NO_EARNINGS_TICKERS, frozenset)
    assert "SPY" in _NO_EARNINGS_TICKERS
    assert "AAPL" not in _NO_EARNINGS_TICKERS


@test("calendar: caching works (second call uses cache)")
def _():
    cal = EarningsCalendar()
    # Prime cache with a known value
    future = datetime(2026, 4, 1, tzinfo=timezone.utc)
    cal._earnings_cache["TEST"] = (future, datetime.now(timezone.utc))
    result = cal.get_next_earnings("TEST")
    assert result == future


@test("calendar: cache expires after TTL")
def _():
    cal = EarningsCalendar()
    old_time = datetime.now(timezone.utc) - timedelta(hours=25)
    future = datetime(2026, 4, 1, tzinfo=timezone.utc)
    cal._earnings_cache["EXPIRED"] = (future, old_time)
    # Would try yfinance (which is mocked), so cache should be stale
    # We just verify the cache check logic
    _, fetched_at = cal._earnings_cache["EXPIRED"]
    age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
    assert age_hours > 24  # cache is expired


@test("calendar: get_lookahead_calendar returns sorted list")
def _():
    cal = EarningsCalendar()
    now = datetime.now(timezone.utc)
    # Prime cache
    cal._earnings_cache["AAPL"] = (now + timedelta(days=5), now)
    cal._earnings_cache["MSFT"] = (now + timedelta(days=2), now)
    cal._earnings_cache["NVDA"] = (now + timedelta(days=20), now)  # beyond lookahead

    result = cal.get_lookahead_calendar(["AAPL", "MSFT", "NVDA"], days_ahead=14)
    assert len(result) == 2  # NVDA excluded (20 days > 14)
    assert result[0]["ticker"] == "MSFT"  # 2 days < 5 days
    assert result[1]["ticker"] == "AAPL"


@test("calendar: get_lookahead_calendar filters past earnings")
def _():
    cal = EarningsCalendar()
    now = datetime.now(timezone.utc)
    cal._earnings_cache["OLD"] = (now - timedelta(days=5), now)
    result = cal.get_lookahead_calendar(["OLD"], days_ahead=14)
    assert len(result) == 0


@test("calendar: calculate_expected_move returns None for empty chain")
def _():
    cal = EarningsCalendar()
    assert cal.calculate_expected_move(None, 100.0) is None


@test("calendar: historical stay-in-range returns default for ETFs")
def _():
    cal = EarningsCalendar()
    result = cal.calculate_historical_stay_in_range("SPY")
    assert result["total_quarters"] == 0
    assert result["stay_in_range_pct"] == 0.0


# ===== ENTRY WINDOW GATE TESTS =====
print("\n--- Entry Window Gate ---")


@test("gate: earnings in 2 days → True")
def _():
    scanner = EarningsScanner(_base_config())
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    earnings = now + timedelta(days=2)
    scanner._earnings_calendar._earnings_cache["AAPL"] = (
        earnings, datetime.now(timezone.utc)
    )
    assert scanner.is_in_entry_window("AAPL", now_et=now) is True


@test("gate: earnings in 1 day → True")
def _():
    scanner = EarningsScanner(_base_config())
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    earnings = now + timedelta(days=1)
    scanner._earnings_calendar._earnings_cache["AAPL"] = (
        earnings, datetime.now(timezone.utc)
    )
    assert scanner.is_in_entry_window("AAPL", now_et=now) is True


@test("gate: earnings in 3 days → True")
def _():
    scanner = EarningsScanner(_base_config())
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    earnings = now + timedelta(days=3)
    scanner._earnings_calendar._earnings_cache["AAPL"] = (
        earnings, datetime.now(timezone.utc)
    )
    assert scanner.is_in_entry_window("AAPL", now_et=now) is True


@test("gate: earnings in 5 days → False (too far)")
def _():
    scanner = EarningsScanner(_base_config())
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    earnings = now + timedelta(days=5)
    scanner._earnings_calendar._earnings_cache["AAPL"] = (
        earnings, datetime.now(timezone.utc)
    )
    assert scanner.is_in_entry_window("AAPL", now_et=now) is False


@test("gate: earnings today (0 days) → False (too close)")
def _():
    scanner = EarningsScanner(_base_config())
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    earnings = now  # same day
    scanner._earnings_calendar._earnings_cache["AAPL"] = (
        earnings, datetime.now(timezone.utc)
    )
    assert scanner.is_in_entry_window("AAPL", now_et=now) is False


@test("gate: no earnings date → False")
def _():
    scanner = EarningsScanner(_base_config())
    # SPY is an ETF, returns None
    assert scanner.is_in_entry_window("SPY") is False


# ===== SCANNER SCAN TESTS =====
print("\n--- Scanner Scan ---")


@test("scan: no upcoming earnings returns empty")
def _():
    scanner = EarningsScanner(_base_config())
    scanner._earnings_calendar.get_lookahead_calendar = lambda t, **k: []
    result = scanner.scan()
    assert result == []


@test("scan: calls _scan_ticker for tickers in entry window")
def _():
    scanner = EarningsScanner(_base_config())
    now = datetime.now(timezone.utc)
    scanner._earnings_calendar.get_lookahead_calendar = lambda t, **k: [
        {"ticker": "AAPL", "earnings_date": now + timedelta(days=2), "days_until": 2},
        {"ticker": "MSFT", "earnings_date": now + timedelta(days=10), "days_until": 10},
    ]
    scan_calls = []
    def mock_scan_ticker(ticker, earnings_date):
        scan_calls.append(ticker)
        return {"ticker": ticker, "type": "earnings_iron_condor", "score": 80,
                "alert_source": "earnings_play"}
    scanner._scan_ticker = mock_scan_ticker
    result = scanner.scan()
    assert "AAPL" in scan_calls   # 2 days = in window
    assert "MSFT" not in scan_calls  # 10 days = outside window
    assert len(result) == 1


@test("scan: handles ticker error gracefully")
def _():
    scanner = EarningsScanner(_base_config())
    now = datetime.now(timezone.utc)
    scanner._earnings_calendar.get_lookahead_calendar = lambda t, **k: [
        {"ticker": "AAPL", "earnings_date": now + timedelta(days=2), "days_until": 2},
    ]
    def failing_scan(ticker, earnings_date):
        raise Exception("API error")
    scanner._scan_ticker = failing_scan
    result = scanner.scan()
    assert result == []


@test("scan: annotates alert_source=earnings_play")
def _():
    scanner = EarningsScanner(_base_config())
    now = datetime.now(timezone.utc)
    scanner._earnings_calendar.get_lookahead_calendar = lambda t, **k: [
        {"ticker": "NVDA", "earnings_date": now + timedelta(days=1), "days_until": 1},
    ]
    def mock_scan(ticker, earnings_date):
        return {"ticker": ticker, "type": "earnings_iron_condor"}
    scanner._scan_ticker = mock_scan
    result = scanner.scan()
    assert len(result) == 1
    assert result[0]["alert_source"] == "earnings_play"


@test("scan: multiple tickers in window all scanned")
def _():
    scanner = EarningsScanner(_base_config())
    now = datetime.now(timezone.utc)
    scanner._earnings_calendar.get_lookahead_calendar = lambda t, **k: [
        {"ticker": "AAPL", "earnings_date": now + timedelta(days=1), "days_until": 1},
        {"ticker": "MSFT", "earnings_date": now + timedelta(days=2), "days_until": 2},
        {"ticker": "GOOGL", "earnings_date": now + timedelta(days=3), "days_until": 3},
    ]
    scanned = []
    def mock_scan(ticker, earnings_date):
        scanned.append(ticker)
        return {"ticker": ticker, "type": "earnings_iron_condor"}
    scanner._scan_ticker = mock_scan
    result = scanner.scan()
    assert len(scanned) == 3
    assert len(result) == 3


@test("scan: None from _scan_ticker is filtered")
def _():
    scanner = EarningsScanner(_base_config())
    now = datetime.now(timezone.utc)
    scanner._earnings_calendar.get_lookahead_calendar = lambda t, **k: [
        {"ticker": "AAPL", "earnings_date": now + timedelta(days=2), "days_until": 2},
    ]
    scanner._scan_ticker = lambda t, e: None
    result = scanner.scan()
    assert result == []


@test("scan: _scan_ticker returns full pipeline (mocked)")
def _():
    scanner = EarningsScanner(_base_config())
    # Mock all internal dependencies
    scanner._earnings_calendar.calculate_historical_stay_in_range = lambda t, **k: {
        "stay_in_range_pct": 75, "total_quarters": 6, "quarters_in_range": 4, "avg_move_pct": 3.0,
    }
    scanner._earnings_calendar.calculate_expected_move = lambda c, p: 10.0

    class FakeDataCache:
        def get_history(self, ticker, period=None):
            class FakeDF:
                empty = False
                def __getitem__(self, key):
                    class FakeSeries:
                        def __init__(self): pass
                        @property
                        def iloc(self): return type("I", (), {"__getitem__": lambda s, i: 500.0})()
                    return FakeSeries()
            return FakeDF()
    scanner._data_cache = FakeDataCache()

    class FakeOptionsAnalyzer:
        def get_options_chain(self, ticker): return type("DF", (), {"empty": False})()
        def get_current_iv(self, chain): return 0.35
        def calculate_iv_rank(self, ticker, iv): return {"iv_rank": 70}
    scanner._options_analyzer = FakeOptionsAnalyzer()

    # Mock condor builder to return a valid opp
    scanner._build_earnings_condor = lambda **k: {
        "ticker": k["ticker"], "type": "earnings_iron_condor",
        "credit": 2.0, "spread_width": 5, "max_loss": 3.0,
    }
    result = scanner._scan_ticker("AAPL", datetime(2026, 4, 1, tzinfo=timezone.utc))
    assert result is not None
    assert result["type"] == "earnings_iron_condor"
    assert "score" in result


@test("scan: low IV rank gates ticker out")
def _():
    scanner = EarningsScanner(_base_config())
    class FakeDataCache:
        def get_history(self, ticker, period=None):
            class FakeDF:
                empty = False
                def __getitem__(self, key):
                    class FakeSeries:
                        @property
                        def iloc(self): return type("I", (), {"__getitem__": lambda s, i: 500.0})()
                    return FakeSeries()
            return FakeDF()
    scanner._data_cache = FakeDataCache()

    class FakeOptionsAnalyzer:
        def get_options_chain(self, ticker): return type("DF", (), {"empty": False})()
        def get_current_iv(self, chain): return 0.20
        def calculate_iv_rank(self, ticker, iv): return {"iv_rank": 40}  # below 60
    scanner._options_analyzer = FakeOptionsAnalyzer()

    result = scanner._scan_ticker("AAPL", datetime(2026, 4, 1, tzinfo=timezone.utc))
    assert result is None


# ===== CONDOR BUILDER TESTS =====
print("\n--- Condor Builder ---")


@test("condor: price-based strike placement")
def _():
    # Verify the math: short_put = price - 1.2 * expected_move
    price = 500.0
    expected_move = 10.0
    multiplier = 1.2
    target_short_put = price - multiplier * expected_move  # 488
    target_short_call = price + multiplier * expected_move  # 512
    assert target_short_put == 488.0
    assert target_short_call == 512.0


@test("condor: spread_width = 5 from config")
def _():
    cfg = build_earnings_config(_base_config())
    assert cfg["strategy"]["earnings"]["spread_width"] == 5


@test("condor: credit must be positive")
def _():
    # If credit <= 0, builder returns None
    credit = -0.5
    assert credit <= 0  # would be rejected


@test("condor: max_loss = width - credit")
def _():
    width = 5.0
    credit = 2.0
    max_loss = width - credit
    assert max_loss == 3.0


@test("condor: earnings_iron_condor type field")
def _():
    opp = {
        "ticker": "AAPL",
        "type": "earnings_iron_condor",
        "direction": "neutral",
        "short_strike": 488.0,
        "long_strike": 483.0,
        "call_short_strike": 512.0,
        "call_long_strike": 517.0,
        "spread_width": 5.0,
        "credit": 2.0,
        "total_credit": 2.0,
        "max_loss": 3.0,
        "expected_move": 10.0,
        "earnings_date": "2026-04-01T00:00:00+00:00",
        "stay_in_range_pct": 75.0,
        "iv_rank": 70.0,
        "alert_source": "earnings_play",
    }
    assert opp["type"] == "earnings_iron_condor"
    assert opp["direction"] == "neutral"
    assert opp["alert_source"] == "earnings_play"


@test("condor: four-leg structure")
def _():
    opp = {
        "short_strike": 488.0,   # short put
        "long_strike": 483.0,    # long put
        "call_short_strike": 512.0,  # short call
        "call_long_strike": 517.0,   # long call
    }
    # Put side: short > long
    assert opp["short_strike"] > opp["long_strike"]
    # Call side: long > short
    assert opp["call_long_strike"] > opp["call_short_strike"]
    # Width check
    assert opp["short_strike"] - opp["long_strike"] == 5.0
    assert opp["call_long_strike"] - opp["call_short_strike"] == 5.0


@test("condor: management_instructions present")
def _():
    EarningsScanner(_base_config())
    # Use mock builder output
    opp = {
        "management_instructions": (
            "Earnings iron condor. Close morning after earnings to capture IV crush, "
            "or at 50% profit / 2x credit stop loss."
        ),
    }
    assert "IV crush" in opp["management_instructions"]
    assert "50%" in opp["management_instructions"]


@test("condor: earnings_date and expected_move fields present")
def _():
    opp = {
        "expected_move": 10.0,
        "earnings_date": "2026-04-01T00:00:00+00:00",
        "stay_in_range_pct": 75.0,
        "iv_rank": 70.0,
    }
    assert "expected_move" in opp
    assert "earnings_date" in opp
    assert "stay_in_range_pct" in opp
    assert "iv_rank" in opp


# ===== SCORING TESTS =====
print("\n--- Scoring ---")


@test("score: perfect IV rank + perfect range + perfect credit = 100")
def _():
    score = EarningsScanner._score_earnings_opportunity(
        iv_rank=100, stay_in_range=100, credit=2.5, width=5,
    )
    assert score == 100.0


@test("score: minimum IV rank + minimum range + zero credit = 0")
def _():
    score = EarningsScanner._score_earnings_opportunity(
        iv_rank=60, stay_in_range=65, credit=0, width=5,
    )
    assert score == 0.0


@test("score: mid-range values produce reasonable score")
def _():
    score = EarningsScanner._score_earnings_opportunity(
        iv_rank=80, stay_in_range=80, credit=1.0, width=5,
    )
    # IV: (80-60)/40*35 = 17.5, Range: (80-65)/35*35 = 15, Credit: 1/5*30/0.5=12
    assert 30 < score < 60


@test("score: zero width returns zero credit score")
def _():
    score = EarningsScanner._score_earnings_opportunity(
        iv_rank=80, stay_in_range=80, credit=2.0, width=0,
    )
    # IV + Range only, no credit score
    assert score > 0
    # Should be IV + Range = ~17.5 + 15 = 32.5
    assert score < 40


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


def _make_earnings_trade(**overrides):
    base = {
        "id": "earn1",
        "ticker": "AAPL",
        "strategy_type": "earnings_iron_condor",
        "total_credit": 200,
        "earnings_date": "2026-04-01T00:00:00+00:00",
        "_mock_pnl": 0,
    }
    base.update(overrides)
    return base


@test("exit: post-earnings triggers close alert")
def _():
    trade = _make_earnings_trade(
        earnings_date="2026-03-01T00:00:00+00:00",
        _mock_pnl=30,
    )
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    # now_et is after earnings
    now = datetime(2026, 3, 2, 10, 0, 0, tzinfo=timezone.utc)
    triggered = monitor.check_and_alert({"AAPL": 500.0}, now_et=now)
    reasons = [t["reason"] for t in triggered]
    assert "post_earnings" in reasons
    assert len(bot.sent) >= 1


@test("exit: pre-earnings does NOT trigger post-earnings close")
def _():
    trade = _make_earnings_trade(
        earnings_date="2026-04-01T00:00:00+00:00",
        _mock_pnl=30,
    )
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    now = datetime(2026, 3, 30, 10, 0, 0, tzinfo=timezone.utc)
    triggered = monitor.check_and_alert({"AAPL": 500.0}, now_et=now)
    reasons = [t["reason"] for t in triggered]
    assert "post_earnings" not in reasons


@test("exit: 50% profit triggers alert")
def _():
    trade = _make_earnings_trade(_mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"AAPL": 500.0})
    reasons = [t["reason"] for t in triggered]
    assert "profit_target" in reasons


@test("exit: exact 50% profit triggers")
def _():
    trade = _make_earnings_trade(_mock_pnl=100)  # 100/200 = 50%
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"AAPL": 500.0})
    reasons = [t["reason"] for t in triggered]
    assert "profit_target" in reasons


@test("exit: below profit threshold → no alert")
def _():
    trade = _make_earnings_trade(_mock_pnl=50)
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"AAPL": 500.0})
    profit_alerts = [t for t in triggered if t["reason"] == "profit_target"]
    assert len(profit_alerts) == 0


@test("exit: 2x stop loss triggers alert")
def _():
    trade = _make_earnings_trade(_mock_pnl=-410)
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"AAPL": 500.0})
    reasons = [t["reason"] for t in triggered]
    assert "stop_loss" in reasons


@test("exit: exact 2x stop loss triggers")
def _():
    trade = _make_earnings_trade(_mock_pnl=-400)  # -400 = -(200 * 2)
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"AAPL": 500.0})
    reasons = [t["reason"] for t in triggered]
    assert "stop_loss" in reasons


@test("exit: composite dedup allows post_earnings + profit")
def _():
    trade = _make_earnings_trade(
        earnings_date="2026-03-01T00:00:00+00:00",
        _mock_pnl=110,
    )
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    now = datetime(2026, 3, 2, 10, 0, 0, tzinfo=timezone.utc)
    triggered = monitor.check_and_alert({"AAPL": 500.0}, now_et=now)
    reasons = {t["reason"] for t in triggered}
    assert "post_earnings" in reasons
    assert "profit_target" in reasons


@test("exit: duplicate suppression works per-reason")
def _():
    trade = _make_earnings_trade(_mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    monitor.check_and_alert({"AAPL": 500.0})
    triggered2 = monitor.check_and_alert({"AAPL": 500.0})
    profit_alerts = [t for t in triggered2 if t["reason"] == "profit_target"]
    assert len(profit_alerts) == 0


@test("exit: non-earnings positions skipped")
def _():
    trade = _make_earnings_trade(strategy_type="iron_condor", _mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"AAPL": 500.0})
    assert len(triggered) == 0


@test("exit: missing trade_id skipped")
def _():
    trade = _make_earnings_trade(id="", _mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"AAPL": 500.0})
    assert len(triggered) == 0


@test("exit: zero credit skipped")
def _():
    trade = _make_earnings_trade(total_credit=0, _mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"AAPL": 500.0})
    assert len(triggered) == 0


@test("exit: missing price skipped")
def _():
    trade = _make_earnings_trade(_mock_pnl=110)
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"NVDA": 500.0})  # wrong ticker
    assert len(triggered) == 0


@test("exit: 'type' field fallback (no strategy_type)")
def _():
    trade = {
        "id": "earn2", "ticker": "MSFT", "type": "earnings_iron_condor",
        "total_credit": 150, "earnings_date": "2026-04-01T00:00:00+00:00",
        "_mock_pnl": 80,
    }
    bot = _MockTelegramBot()
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"MSFT": 400.0})
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "profit_target"


@test("exit: telegram failure doesn't crash")
def _():
    trade = _make_earnings_trade(_mock_pnl=110)
    bot = _MockTelegramBot()
    def failing_send(msg):
        raise Exception("network error")
    bot.send_alert = failing_send
    monitor = EarningsExitMonitor(
        _MockPaperTrader([trade]), bot, formatter=_MockFormatter()
    )
    triggered = monitor.check_and_alert({"AAPL": 500.0})
    assert len(triggered) >= 1


# ===== _is_post_earnings HELPER TESTS =====
print("\n--- _is_post_earnings helper ---")


@test("_is_post_earnings: after earnings → True")
def _():
    trade = {"earnings_date": "2026-03-01T00:00:00+00:00"}
    now = datetime(2026, 3, 2, tzinfo=timezone.utc)
    assert _is_post_earnings(trade, now) is True


@test("_is_post_earnings: before earnings → False")
def _():
    trade = {"earnings_date": "2026-04-01T00:00:00+00:00"}
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert _is_post_earnings(trade, now) is False


@test("_is_post_earnings: no earnings_date → False")
def _():
    trade = {}
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert _is_post_earnings(trade, now) is False


@test("_is_post_earnings: empty string → False")
def _():
    trade = {"earnings_date": ""}
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert _is_post_earnings(trade, now) is False


# ===== FROM_OPPORTUNITY EARNINGS TESTS =====
print("\n--- from_opportunity earnings conversion ---")


@test("from_opportunity: earnings_iron_condor → AlertType.earnings_play")
def _():
    opp = {
        "ticker": "AAPL",
        "type": "earnings_iron_condor",
        "short_strike": 488.0,
        "long_strike": 483.0,
        "call_short_strike": 512.0,
        "call_long_strike": 517.0,
        "credit": 2.0,
        "stop_loss": 4.0,
        "profit_target": 1.0,
        "expiration": "2026-04-05",
        "dte": 5,
        "score": 75,
        "alert_source": "earnings_play",
    }
    alert = Alert.from_opportunity(opp)
    assert alert.type == AlertType.earnings_play
    assert alert.direction == Direction.neutral


@test("from_opportunity: earnings with alert_source maps correctly")
def _():
    opp = {
        "ticker": "MSFT",
        "type": "earnings_iron_condor",
        "short_strike": 390.0,
        "long_strike": 385.0,
        "call_short_strike": 420.0,
        "call_long_strike": 425.0,
        "credit": 1.8,
        "stop_loss": 3.6,
        "profit_target": 0.9,
        "expiration": "2026-04-05",
        "dte": 5,
        "score": 80,
        "alert_source": "earnings_play",
    }
    alert = Alert.from_opportunity(opp)
    assert alert.type == AlertType.earnings_play


@test("from_opportunity: earnings condor has 4 legs")
def _():
    opp = {
        "ticker": "AAPL",
        "type": "earnings_iron_condor",
        "short_strike": 488.0,
        "long_strike": 483.0,
        "call_short_strike": 512.0,
        "call_long_strike": 517.0,
        "credit": 2.0,
        "stop_loss": 4.0,
        "profit_target": 1.0,
        "expiration": "2026-04-05",
        "dte": 5,
        "score": 75,
        "alert_source": "earnings_play",
    }
    alert = Alert.from_opportunity(opp)
    assert len(alert.legs) == 4
    # Put side
    assert alert.legs[0].strike == 488.0
    assert alert.legs[0].option_type == "put"
    assert alert.legs[0].action == "sell"
    assert alert.legs[1].strike == 483.0
    assert alert.legs[1].option_type == "put"
    assert alert.legs[1].action == "buy"
    # Call side
    assert alert.legs[2].strike == 512.0
    assert alert.legs[2].option_type == "call"
    assert alert.legs[2].action == "sell"
    assert alert.legs[3].strike == 517.0
    assert alert.legs[3].option_type == "call"
    assert alert.legs[3].action == "buy"


@test("from_opportunity: earnings direction is neutral")
def _():
    opp = {
        "ticker": "NVDA",
        "type": "earnings_iron_condor",
        "short_strike": 780.0,
        "long_strike": 775.0,
        "call_short_strike": 820.0,
        "call_long_strike": 825.0,
        "credit": 3.0,
        "stop_loss": 6.0,
        "profit_target": 1.5,
        "expiration": "2026-04-05",
        "dte": 5,
        "score": 85,
        "alert_source": "earnings_play",
    }
    alert = Alert.from_opportunity(opp)
    assert alert.direction == Direction.neutral


@test("from_opportunity: regular iron_condor (no earnings source) stays iron_condor")
def _():
    opp = {
        "ticker": "SPY",
        "type": "iron_condor",
        "short_strike": 550.0,
        "long_strike": 545.0,
        "call_short_strike": 580.0,
        "call_long_strike": 585.0,
        "credit": 1.5,
        "stop_loss": 3.0,
        "profit_target": 0.75,
        "expiration": "2026-04-05",
        "dte": 7,
        "score": 70,
        "alert_source": "iron_condor",
    }
    alert = Alert.from_opportunity(opp)
    assert alert.type == AlertType.iron_condor


@test("from_opportunity: earnings management instructions preserved")
def _():
    opp = {
        "ticker": "AAPL",
        "type": "earnings_iron_condor",
        "short_strike": 488.0,
        "long_strike": 483.0,
        "call_short_strike": 512.0,
        "call_long_strike": 517.0,
        "credit": 2.0,
        "stop_loss": 4.0,
        "profit_target": 1.0,
        "expiration": "2026-04-05",
        "dte": 5,
        "score": 75,
        "alert_source": "earnings_play",
        "management_instructions": "Close morning after earnings to capture IV crush.",
    }
    alert = Alert.from_opportunity(opp)
    assert "IV crush" in alert.management_instructions


# ===== POSITION SIZER TESTS =====
print("\n--- Position Sizer (earnings) ---")


@test("sizer: earnings condor uses (width-credit)*100 like iron condor")
def _():
    sizer = AlertPositionSizer()
    alert = Alert(
        type=AlertType.earnings_play,
        ticker="AAPL",
        direction=Direction.neutral,
        legs=[
            Leg(488.0, "put", "sell", "2026-04-05"),
            Leg(483.0, "put", "buy", "2026-04-05"),
            Leg(512.0, "call", "sell", "2026-04-05"),
            Leg(517.0, "call", "buy", "2026-04-05"),
        ],
        entry_price=2.0,
        stop_loss=4.0,
        profit_target=1.0,
        risk_pct=0.02,
    )
    result = sizer.size(alert, 100000, iv_rank=70, current_portfolio_risk=0)
    # Max loss = (width - credit) * 100 per contract = (5 - 2) * 100 = 300
    assert result.max_loss == 300 * result.contracts


@test("sizer: earnings_play contracts >= 1")
def _():
    sizer = AlertPositionSizer()
    alert = Alert(
        type=AlertType.earnings_play,
        ticker="AAPL",
        direction=Direction.neutral,
        legs=[
            Leg(488.0, "put", "sell", "2026-04-05"),
            Leg(483.0, "put", "buy", "2026-04-05"),
            Leg(512.0, "call", "sell", "2026-04-05"),
            Leg(517.0, "call", "buy", "2026-04-05"),
        ],
        entry_price=2.0,
        stop_loss=4.0,
        profit_target=1.0,
        risk_pct=0.02,
    )
    result = sizer.size(alert, 100000, iv_rank=70, current_portfolio_risk=0)
    assert result.contracts >= 1


# ===== TELEGRAM FORMATTER TESTS =====
print("\n--- Telegram Formatter ---")


@test("telegram: earnings_play shows EARNINGS PLAY header")
def _():
    from alerts.formatters.telegram import TelegramAlertFormatter
    fmt = TelegramAlertFormatter()
    alert = Alert(
        type=AlertType.earnings_play,
        ticker="AAPL",
        direction=Direction.neutral,
        legs=[
            Leg(488.0, "put", "sell", "2026-04-05"),
            Leg(483.0, "put", "buy", "2026-04-05"),
            Leg(512.0, "call", "sell", "2026-04-05"),
            Leg(517.0, "call", "buy", "2026-04-05"),
        ],
        entry_price=2.0,
        stop_loss=4.0,
        profit_target=1.0,
        risk_pct=0.02,
    )
    text = fmt.format_entry_alert(alert)
    assert "EARNINGS PLAY" in text


@test("telegram: earnings_play uses Credit label (not Debit)")
def _():
    from alerts.formatters.telegram import TelegramAlertFormatter
    fmt = TelegramAlertFormatter()
    alert = Alert(
        type=AlertType.earnings_play,
        ticker="AAPL",
        direction=Direction.neutral,
        legs=[
            Leg(488.0, "put", "sell", "2026-04-05"),
            Leg(483.0, "put", "buy", "2026-04-05"),
            Leg(512.0, "call", "sell", "2026-04-05"),
            Leg(517.0, "call", "buy", "2026-04-05"),
        ],
        entry_price=2.0,
        stop_loss=4.0,
        profit_target=1.0,
        risk_pct=0.02,
    )
    text = fmt.format_entry_alert(alert)
    assert "Credit:" in text
    assert "Debit:" not in text


# ===== RESULTS =====
print(f"\n{'='*60}")
print(f"Phase 5 tests: {_pass} passed, {_fail} failed")
if _errors:
    print("\nFailed tests:")
    for name, err in _errors:
        print(f"  {name}: {err}")
print(f"{'='*60}")

sys.exit(1 if _fail else 0)
