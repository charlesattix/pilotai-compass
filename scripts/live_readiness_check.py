#!/usr/bin/env python3
"""Live-Readiness Checklist.

Verifies that all required infrastructure is in place before moving from
paper trading to live execution. Produces a pass/fail checklist.

Usage:
    python scripts/live_readiness_check.py
    python scripts/live_readiness_check.py --config config.yaml --db data/pilotai.db
    python scripts/live_readiness_check.py --skip-telegram   # skip Telegram send test
"""

import argparse
import importlib
import inspect
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "results" / "paper_trading"

# ---------------------------------------------------------------------------
# Check result type
# ---------------------------------------------------------------------------

class CheckResult:
    """Single check outcome."""

    def __init__(self, name: str, passed: bool, detail: str, severity: str = "REQUIRED"):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.severity = severity  # REQUIRED | RECOMMENDED

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def __str__(self) -> str:
        mark = "[PASS]" if self.passed else "[FAIL]"
        sev = f" ({self.severity})" if not self.passed and self.severity == "RECOMMENDED" else ""
        return f"  {mark} {self.name}{sev}\n        {self.detail}"


# ---------------------------------------------------------------------------
# 1. Config file checks
# ---------------------------------------------------------------------------

def check_champion_config(config_path: str) -> List[CheckResult]:
    """Verify champion.json exists and is valid."""
    results = []
    path = Path(config_path)

    if not path.exists():
        results.append(CheckResult(
            "Champion config exists",
            False,
            f"File not found: {config_path}",
        ))
        return results

    results.append(CheckResult(
        "Champion config exists",
        True,
        str(config_path),
    ))

    try:
        with open(path) as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        results.append(CheckResult("Champion config valid JSON", False, str(e)))
        return results

    results.append(CheckResult("Champion config valid JSON", True, "Parsed successfully"))

    # Required keys
    has_strategies = "strategies" in cfg and len(cfg.get("strategies", [])) > 0
    has_params = "strategy_params" in cfg and len(cfg.get("strategy_params", {})) > 0
    results.append(CheckResult(
        "Champion config has strategies",
        has_strategies and has_params,
        f"strategies={cfg.get('strategies', [])}" if has_strategies else "Missing 'strategies' or 'strategy_params'",
    ))

    # Validation section
    validation = cfg.get("validation", {})
    robustness = validation.get("robustness_score", 0)
    results.append(CheckResult(
        "Champion config validated (ROBUST)",
        robustness >= 0.70,
        f"robustness_score={robustness:.3f}" + (" (ROBUST)" if robustness >= 0.70 else " (below 0.70 threshold)"),
    ))

    return results


def check_main_config(config_path: str) -> List[CheckResult]:
    """Verify main config.yaml exists and has required sections."""
    results = []
    path = Path(config_path)

    if not path.exists():
        results.append(CheckResult(
            "Main config exists",
            False,
            f"Not found: {config_path}",
        ))
        return results

    results.append(CheckResult("Main config exists", True, str(config_path)))

    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except ImportError:
        # Try raw parse if pyyaml missing
        results.append(CheckResult(
            "Main config parseable",
            False,
            "PyYAML not installed — cannot validate config.yaml",
        ))
        return results
    except Exception as e:
        results.append(CheckResult("Main config parseable", False, str(e)))
        return results

    results.append(CheckResult("Main config parseable", True, "YAML parsed successfully"))

    required = ["tickers", "strategy", "risk"]
    missing = [s for s in required if s not in cfg]
    results.append(CheckResult(
        "Main config required sections",
        len(missing) == 0,
        f"Present: {[s for s in required if s in cfg]}" if not missing else f"Missing: {missing}",
    ))

    # Alpaca config
    alpaca_cfg = cfg.get("alpaca", {})
    alpaca_enabled = alpaca_cfg.get("enabled", False)
    results.append(CheckResult(
        "Alpaca integration configured",
        alpaca_enabled,
        "alpaca.enabled=True" if alpaca_enabled else "alpaca.enabled=False or section missing",
    ))

    return results


def check_env_vars() -> List[CheckResult]:
    """Check required environment variables."""
    results = []

    required_vars = {
        "ALPACA_API_KEY": "Alpaca trading API key",
        "ALPACA_API_SECRET": "Alpaca trading API secret",
    }
    recommended_vars = {
        "POLYGON_API_KEY": "Polygon data API key",
        "TELEGRAM_BOT_TOKEN": "Telegram alert bot token",
        "TELEGRAM_CHAT_ID": "Telegram chat ID for alerts",
    }

    for var, desc in required_vars.items():
        val = os.environ.get(var)
        results.append(CheckResult(
            f"Env: {var}",
            bool(val),
            f"{desc} — {'set' if val else 'NOT SET'}",
        ))

    for var, desc in recommended_vars.items():
        val = os.environ.get(var)
        results.append(CheckResult(
            f"Env: {var}",
            bool(val),
            f"{desc} — {'set' if val else 'NOT SET'}",
            severity="RECOMMENDED",
        ))

    return results


# ---------------------------------------------------------------------------
# 2. Database checks
# ---------------------------------------------------------------------------

def check_database(db_path: str) -> List[CheckResult]:
    """Verify DB connectivity, schema, and data freshness."""
    results = []
    path = Path(db_path)

    if not path.exists():
        results.append(CheckResult(
            "Database file exists",
            False,
            f"Not found: {db_path}. Run init_db() or start paper trader first.",
        ))
        return results

    results.append(CheckResult("Database file exists", True, str(db_path)))

    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
    except Exception as e:
        results.append(CheckResult("Database connectable", False, str(e)))
        return results

    results.append(CheckResult("Database connectable", True, "SQLite connection OK"))

    # Check WAL mode
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        results.append(CheckResult(
            "Database WAL mode",
            mode.lower() == "wal",
            f"journal_mode={mode}" + ("" if mode.lower() == "wal" else " (should be WAL for concurrent access)"),
            severity="RECOMMENDED",
        ))
    except Exception:
        pass

    # Check required tables
    try:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    except Exception as e:
        results.append(CheckResult("Database schema", False, str(e)))
        conn.close()
        return results

    required_tables = {"trades", "scanner_state"}
    missing_tables = required_tables - tables
    results.append(CheckResult(
        "Database required tables",
        len(missing_tables) == 0,
        f"Found: {sorted(tables & required_tables)}" if not missing_tables else f"Missing: {sorted(missing_tables)}",
    ))

    # Data freshness — check most recent trade
    try:
        row = conn.execute(
            "SELECT MAX(updated_at) as latest FROM trades"
        ).fetchone()
        latest = row[0] if row else None
        if latest:
            try:
                latest_dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                age_days = (datetime.now(latest_dt.tzinfo) - latest_dt).days if latest_dt.tzinfo else (datetime.now() - latest_dt).days
            except Exception:
                age_days = -1
            fresh = age_days <= 7 if age_days >= 0 else False
            results.append(CheckResult(
                "Database data freshness",
                fresh,
                f"Latest trade update: {latest} ({age_days}d ago)" if age_days >= 0 else f"Latest: {latest} (age unknown)",
                severity="RECOMMENDED",
            ))
        else:
            results.append(CheckResult(
                "Database data freshness",
                False,
                "No trades in database",
                severity="RECOMMENDED",
            ))
    except Exception:
        pass

    # Trade count summary
    try:
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        open_count = conn.execute("SELECT COUNT(*) FROM trades WHERE status='open'").fetchone()[0]
        closed = total - open_count
        results.append(CheckResult(
            "Database trade history",
            total > 0,
            f"Total: {total} (open: {open_count}, closed: {closed})",
            severity="RECOMMENDED",
        ))
    except Exception:
        pass

    conn.close()
    return results


# ---------------------------------------------------------------------------
# 3. Buying power check (Lesson 005)
# ---------------------------------------------------------------------------

def check_buying_power_logic() -> List[CheckResult]:
    """Verify buying power check exists before order submission (Lesson 005)."""
    results = []

    # Check 1: ExecutionEngine has drawdown circuit breaker
    try:
        from execution.execution_engine import ExecutionEngine
        has_cb = hasattr(ExecutionEngine, "_check_drawdown_cb")
        results.append(CheckResult(
            "Drawdown circuit breaker exists",
            has_cb,
            "ExecutionEngine._check_drawdown_cb() found" if has_cb else "Missing _check_drawdown_cb method",
        ))

        # Check that submit_opportunity calls the CB
        src = inspect.getsource(ExecutionEngine.submit_opportunity)
        calls_cb = "_check_drawdown_cb" in src
        results.append(CheckResult(
            "Drawdown CB called before submission",
            calls_cb,
            "submit_opportunity() calls _check_drawdown_cb()" if calls_cb else "CB not called in submit path",
        ))
    except ImportError as e:
        results.append(CheckResult("ExecutionEngine importable", False, str(e)))

    # Check 2: AlpacaProvider exposes buying_power in get_account()
    try:
        from strategy.alpaca_provider import AlpacaProvider
        src = inspect.getsource(AlpacaProvider.get_account)
        has_bp = "buying_power" in src and "options_buying_power" in src
        results.append(CheckResult(
            "Account buying power fields exposed",
            has_bp,
            "get_account() returns buying_power + options_buying_power" if has_bp else "buying_power fields not in get_account()",
        ))
    except (ImportError, AttributeError) as e:
        results.append(CheckResult("AlpacaProvider importable", False, str(e)))

    # Check 3: Explicit pre-submission BP check (Lesson 005 gap)
    # Per Lesson 005: "Always check buying power BEFORE attempting order submission"
    try:
        from execution.execution_engine import ExecutionEngine
        src = inspect.getsource(ExecutionEngine.submit_opportunity)
        has_explicit_bp = "buying_power" in src or "options_buying_power" in src
        results.append(CheckResult(
            "Explicit buying power pre-check (Lesson 005)",
            has_explicit_bp,
            "submit_opportunity() checks buying power before order" if has_explicit_bp
            else "WARNING: No explicit buying_power check in submit_opportunity(). "
                 "Drawdown CB exists but doesn't check options_buying_power directly. "
                 "Lesson 005: rejected orders when BP ran out with 72 legs open.",
        ))
    except (ImportError, AttributeError):
        pass

    return results


# ---------------------------------------------------------------------------
# 4. Order failure/retry handling
# ---------------------------------------------------------------------------

def check_order_retry_logic() -> List[CheckResult]:
    """Verify order failure handling and retry logic."""
    results = []

    try:
        from strategy.alpaca_provider import _retry_with_backoff
        results.append(CheckResult(
            "Retry decorator exists",
            True,
            "_retry_with_backoff decorator found in alpaca_provider",
        ))
    except ImportError as e:
        results.append(CheckResult("Retry decorator exists", False, str(e)))
        return results

    # Check that key order methods use the decorator
    try:
        from strategy.alpaca_provider import AlpacaProvider
        methods_to_check = [
            "submit_credit_spread",
            "close_spread",
            "close_iron_condor",
        ]
        for method_name in methods_to_check:
            method = getattr(AlpacaProvider, method_name, None)
            if method is None:
                results.append(CheckResult(
                    f"Retry on {method_name}",
                    False,
                    f"Method {method_name} not found",
                ))
                continue
            # Check if wrapped (functools.wraps preserves __wrapped__)
            has_retry = hasattr(method, "__wrapped__") or "_retry" in str(getattr(method, "__qualname__", ""))
            if not has_retry:
                # Fallback: check source for decorator
                try:
                    src = inspect.getsource(method)
                    has_retry = "@_retry_with_backoff" in src
                except (OSError, TypeError):
                    # Can't get source of wrapped function — check the unwrapped
                    pass
            results.append(CheckResult(
                f"Retry on {method_name}",
                has_retry,
                f"{method_name} has retry wrapper" if has_retry else f"{method_name} may lack retry decorator",
            ))
    except ImportError as e:
        results.append(CheckResult("AlpacaProvider methods", False, str(e)))

    # Check non-retryable error handling
    try:
        import strategy.alpaca_provider as ap
        src = inspect.getsource(ap._retry_with_backoff)
        handles_429 = "429" in src or "rate_limit" in src.lower() or "retry_after" in src.lower()
        handles_4xx = "non_retryable" in src.lower() or "400" in src
        results.append(CheckResult(
            "Rate limit (429) handling",
            handles_429,
            "Retry-After / 429 handling found" if handles_429 else "No 429 handling detected",
        ))
        results.append(CheckResult(
            "Non-retryable error handling",
            handles_4xx,
            "Client errors (4xx) fail fast without retry" if handles_4xx else "Missing non-retryable error check",
        ))
    except Exception:
        pass

    # Check ExecutionEngine error return statuses
    try:
        from execution.execution_engine import ExecutionEngine
        src = inspect.getsource(ExecutionEngine.submit_opportunity)
        statuses = []
        for s in ("submitted", "dry_run", "error", "duplicate", "drawdown_blocked", "market_closed"):
            if f'"{s}"' in src or f"'{s}'" in src:
                statuses.append(s)
        results.append(CheckResult(
            "Order status coverage",
            len(statuses) >= 4,
            f"Statuses handled: {statuses}",
        ))
    except (ImportError, AttributeError):
        pass

    return results


# ---------------------------------------------------------------------------
# 5. Telegram alerting pipeline
# ---------------------------------------------------------------------------

def check_telegram(skip_send: bool = False) -> List[CheckResult]:
    """Verify Telegram alerting pipeline."""
    results = []

    try:
        from shared.telegram_alerts import is_configured, send_message
        results.append(CheckResult(
            "Telegram module importable",
            True,
            "shared.telegram_alerts imported successfully",
        ))
    except ImportError as e:
        results.append(CheckResult("Telegram module importable", False, str(e)))
        return results

    configured = is_configured()
    results.append(CheckResult(
        "Telegram credentials configured",
        configured,
        "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set" if configured
        else "Missing TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID env vars",
    ))

    # Check key alert functions exist
    try:
        from shared import telegram_alerts as ta
        funcs = ["notify_trade_open", "notify_trade_close", "notify_api_failure"]
        for fn_name in funcs:
            exists = hasattr(ta, fn_name) and callable(getattr(ta, fn_name))
            results.append(CheckResult(
                f"Alert function: {fn_name}",
                exists,
                f"{fn_name}() found" if exists else f"{fn_name}() missing",
            ))
    except Exception:
        pass

    # Live send test (optional)
    if configured and not skip_send:
        try:
            ok = send_message("Live-readiness check: Telegram connectivity test")
            results.append(CheckResult(
                "Telegram send test",
                ok,
                "Test message sent successfully" if ok else "send_message() returned False",
            ))
        except Exception as e:
            results.append(CheckResult("Telegram send test", False, f"Exception: {e}"))
    elif not configured:
        results.append(CheckResult(
            "Telegram send test",
            False,
            "Skipped — credentials not configured",
            severity="RECOMMENDED",
        ))

    return results


# ---------------------------------------------------------------------------
# 6. Position limits
# ---------------------------------------------------------------------------

def check_position_limits() -> List[CheckResult]:
    """Verify position limit configuration and enforcement."""
    results = []

    # Backtester limits
    try:
        from engine.portfolio_backtester import PortfolioBacktester
        sig = inspect.signature(PortfolioBacktester.__init__)
        params = sig.parameters

        for param_name, default_val in [
            ("max_positions", 10),
            ("max_positions_per_strategy", 5),
            ("max_portfolio_risk_pct", 0.40),
        ]:
            has_param = param_name in params
            actual_default = params[param_name].default if has_param else None
            results.append(CheckResult(
                f"Backtester: {param_name}",
                has_param,
                f"default={actual_default}" if has_param else "Parameter missing",
            ))

        # Check _can_accept enforcement
        has_accept = hasattr(PortfolioBacktester, "_can_accept")
        results.append(CheckResult(
            "Backtester position limit enforcement",
            has_accept,
            "_can_accept() method enforces limits" if has_accept else "_can_accept() missing",
        ))
    except ImportError as e:
        results.append(CheckResult("PortfolioBacktester importable", False, str(e)))

    # Constants
    try:
        from shared.constants import MAX_POSITIONS_PER_STRATEGY, MAX_CONTRACTS_PER_TRADE
        results.append(CheckResult(
            "Constants: position caps",
            True,
            f"MAX_POSITIONS_PER_STRATEGY={MAX_POSITIONS_PER_STRATEGY}, "
            f"MAX_CONTRACTS_PER_TRADE={MAX_CONTRACTS_PER_TRADE}",
        ))
    except ImportError as e:
        results.append(CheckResult("Constants importable", False, str(e)))

    # Live execution gap check
    try:
        from execution.execution_engine import ExecutionEngine
        src = inspect.getsource(ExecutionEngine.submit_opportunity)
        has_pos_limit = "max_position" in src.lower() or "position_limit" in src.lower() or "_can_accept" in src
        results.append(CheckResult(
            "Execution engine position limit enforcement",
            has_pos_limit,
            "Position limits enforced in submit_opportunity()" if has_pos_limit
            else "WARNING: ExecutionEngine.submit_opportunity() does not enforce position limits. "
                 "Limits are only in backtester — live path relies on external signal filtering.",
        ))
    except (ImportError, AttributeError):
        pass

    return results


# ---------------------------------------------------------------------------
# 7. Regime detector
# ---------------------------------------------------------------------------

def check_regime_detector() -> List[CheckResult]:
    """Verify regime detector is functional with current market data."""
    results = []

    try:
        from compass.regime import Regime, RegimeClassifier
        classifier = RegimeClassifier()
        results.append(CheckResult(
            "RegimeClassifier instantiable",
            True,
            "RegimeClassifier created successfully",
        ))
    except ImportError as e:
        results.append(CheckResult("RegimeClassifier importable", False, str(e)))
        return results
    except Exception as e:
        results.append(CheckResult("RegimeClassifier instantiable", False, str(e)))
        return results

    # Test with live market data
    try:
        import pandas as pd
        import yfinance as yf

        end = datetime.now()
        start = end - timedelta(days=120)

        spy = yf.download(
            "SPY",
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        vix = yf.download(
            "^VIX",
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if spy.empty or vix.empty:
            results.append(CheckResult(
                "Market data available",
                False,
                "Could not download SPY/VIX data from Yahoo Finance",
            ))
            return results

        # Flatten MultiIndex if needed
        for df in (spy, vix):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        results.append(CheckResult(
            "Market data available",
            True,
            f"SPY: {len(spy)} days, VIX: {len(vix)} days",
        ))

        vix_close = vix["Close"].dropna()
        regime_series = classifier.classify_series(spy, vix_close)

        if regime_series.empty:
            results.append(CheckResult("Regime classification", False, "Empty regime series"))
            return results

        # Current regime
        current_regime = regime_series.iloc[-1]
        current_date = regime_series.index[-1].strftime("%Y-%m-%d")
        regime_val = current_regime.value if isinstance(current_regime, Regime) else str(current_regime)

        results.append(CheckResult(
            "Current regime detection",
            True,
            f"Current regime: {regime_val} (as of {current_date})",
        ))

        # Regime distribution over period
        summary = classifier.summarize(regime_series)
        dist_parts = []
        for r, info in sorted(summary.get("distribution", {}).items()):
            pct = info.get("pct", 0) if isinstance(info, dict) else 0
            dist_parts.append(f"{r}={pct:.0f}%")
        transitions = summary.get("transitions", 0)

        results.append(CheckResult(
            "Regime distribution (120d)",
            True,
            f"{', '.join(dist_parts)} | transitions={transitions}",
        ))

        # Sanity: check regime isn't stuck on one value (would suggest broken classifier)
        unique_regimes = regime_series.nunique()
        results.append(CheckResult(
            "Regime diversity",
            unique_regimes >= 2,
            f"{unique_regimes} distinct regimes observed"
            + ("" if unique_regimes >= 2 else " — classifier may be stuck"),
            severity="RECOMMENDED",
        ))

    except Exception as e:
        results.append(CheckResult("Regime classification with live data", False, str(e)))

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(all_results: Dict[str, List[CheckResult]]) -> str:
    """Build human-readable report."""
    lines = []
    lines.append(f"LIVE-READINESS CHECKLIST — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    total_pass = 0
    total_fail = 0
    total_required_fail = 0
    action_items = []

    for section, checks in all_results.items():
        lines.append(f"\n{section}")
        lines.append("-" * len(section))
        for c in checks:
            lines.append(str(c))
            if c.passed:
                total_pass += 1
            else:
                total_fail += 1
                if c.severity == "REQUIRED":
                    total_required_fail += 1
                    action_items.append(f"[{section}] {c.name}: {c.detail}")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"TOTAL: {total_pass} passed, {total_fail} failed "
                 f"({total_required_fail} required, {total_fail - total_required_fail} recommended)")

    if total_required_fail == 0:
        lines.append("STATUS: READY FOR LIVE (all required checks pass)")
    else:
        lines.append(f"STATUS: NOT READY ({total_required_fail} required check(s) failing)")

    if action_items:
        lines.append("")
        lines.append("ACTION ITEMS (required failures):")
        for item in action_items:
            lines.append(f"  - {item}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live-readiness checklist for paper → live transition")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Main config.yaml path")
    parser.add_argument("--champion", default=str(ROOT / "configs" / "champion.json"), help="Champion config path")
    parser.add_argument("--db", default=None, help="SQLite DB path (auto-detected if not set)")
    parser.add_argument("--skip-telegram", action="store_true", help="Skip Telegram send test")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Auto-detect DB path
    db_path = args.db
    if not db_path:
        from shared.constants import DATA_DIR
        candidates = [
            os.environ.get("PILOTAI_DB_PATH", ""),
            os.path.join(DATA_DIR, "pilotai_champion.db"),
            os.path.join(DATA_DIR, "pilotai.db"),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                db_path = c
                break
        if not db_path:
            db_path = candidates[1] if len(candidates) > 1 else "data/pilotai.db"

    print(f"Running live-readiness checks...")
    print(f"  Config:   {args.config}")
    print(f"  Champion: {args.champion}")
    print(f"  DB:       {db_path}")
    print()

    # Run all checks
    all_results: Dict[str, List[CheckResult]] = {}

    all_results["1. CONFIG FILES"] = (
        check_champion_config(args.champion)
        + check_main_config(args.config)
        + check_env_vars()
    )

    all_results["2. DATABASE"] = check_database(db_path)

    all_results["3. BUYING POWER (Lesson 005)"] = check_buying_power_logic()

    all_results["4. ORDER FAILURE/RETRY"] = check_order_retry_logic()

    all_results["5. ALERTING (Telegram)"] = check_telegram(skip_send=args.skip_telegram)

    all_results["6. POSITION LIMITS"] = check_position_limits()

    print("Checking regime detector with live market data...")
    all_results["7. REGIME DETECTOR"] = check_regime_detector()

    # Build report
    report = build_report(all_results)

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "readiness_check.txt"
    with open(out_path, "w") as f:
        f.write(report)

    print()
    print(report)
    print()
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
