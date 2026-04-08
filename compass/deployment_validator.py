"""
Pre-flight deployment validator for paper trading.

Runs 8 check categories before launch:
  1. Module imports     — all required compass modules load
  2. Config schema      — paper_exp880.yaml has required fields
  3. Alpaca connectivity — API key valid, paper account reachable
  4. Model files        — trained model artifacts exist on disk
  5. Directory perms    — data/output dirs exist and are writable
  6. Telegram config    — bot token + chat ID present if enabled
  7. Dry trade sim      — signal→order pipeline executes without error
  8. Report generation  — HTML pass/fail summary

All checks are safe (read-only, no side-effects on broker).
"""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    category: str
    passed: bool
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class ValidationReport:
    checks: List[CheckResult]
    all_passed: bool
    n_passed: int
    n_failed: int
    total_duration_ms: float
    timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class DeploymentValidator:
    """Pre-flight validation for paper trading deployment.

    Args:
        project_root: Root directory of the project.
        config_path: Path to the YAML config file.
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        config_path: str = "configs/paper_exp880.yaml",
    ) -> None:
        self.root = Path(project_root) if project_root else Path.cwd()
        self.config_path = self.root / config_path
        self._checks: List[CheckResult] = []

    # ------------------------------------------------------------------
    # Check runner
    # ------------------------------------------------------------------

    def _run_check(self, name: str, category: str, fn: Callable) -> CheckResult:
        t0 = time.perf_counter()
        try:
            passed, detail = fn()
            ms = (time.perf_counter() - t0) * 1000
            result = CheckResult(name, category, passed, detail, ms)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            result = CheckResult(name, category, False, f"Exception: {e}", ms)
        self._checks.append(result)
        return result

    # ------------------------------------------------------------------
    # 1. Module imports
    # ------------------------------------------------------------------

    REQUIRED_MODULES = [
        "compass.regime",
        "compass.vol_forecaster",
        "compass.risk_orchestrator",
        "compass.portfolio_constructor",
        "compass.signal_backtester",
        "compass.drawdown_protection",
        "compass.performance_attribution",
        "compass.live_trading_blueprint",
        "compass.telegram_alerter",
        "compass.execution_analytics",
    ]

    def check_module_imports(self) -> CheckResult:
        """Verify all required compass modules import correctly."""
        def _check():
            failed = []
            for mod in self.REQUIRED_MODULES:
                try:
                    importlib.import_module(mod)
                except ImportError as e:
                    failed.append(f"{mod}: {e}")
            if failed:
                return False, f"{len(failed)} modules failed: {'; '.join(failed[:3])}"
            return True, f"All {len(self.REQUIRED_MODULES)} modules imported"
        return self._run_check("module_imports", "modules", _check)

    # ------------------------------------------------------------------
    # 2. Config schema validation
    # ------------------------------------------------------------------

    REQUIRED_CONFIG_KEYS = [
        "tickers", "strategy", "risk",
    ]

    REQUIRED_STRATEGY_KEYS = [
        "min_dte", "max_dte",
    ]

    def check_config_schema(self, config: Optional[Dict] = None) -> CheckResult:
        """Validate the paper trading config file."""
        def _check():
            if config is not None:
                cfg = config
            else:
                if not self.config_path.exists():
                    return False, f"Config not found: {self.config_path}"
                try:
                    import yaml
                    cfg = yaml.safe_load(self.config_path.read_text())
                except ImportError:
                    # Fallback: parse as simple key-value
                    text = self.config_path.read_text()
                    cfg = {}
                    for line in text.split("\n"):
                        if ":" in line and not line.strip().startswith("#"):
                            key = line.split(":")[0].strip()
                            if key:
                                cfg[key] = True

            missing_top = [k for k in self.REQUIRED_CONFIG_KEYS if k not in cfg]
            if missing_top:
                return False, f"Missing top-level keys: {missing_top}"

            strategy = cfg.get("strategy", {})
            if isinstance(strategy, dict):
                missing_strat = [k for k in self.REQUIRED_STRATEGY_KEYS if k not in strategy]
                if missing_strat:
                    return False, f"Missing strategy keys: {missing_strat}"

            return True, "Config schema valid"
        return self._run_check("config_schema", "config", _check)

    # ------------------------------------------------------------------
    # 3. Alpaca API connectivity
    # ------------------------------------------------------------------

    def check_alpaca_connectivity(self) -> CheckResult:
        """Test Alpaca API connection (paper account)."""
        def _check():
            api_key = os.environ.get("ALPACA_API_KEY", "")
            api_secret = os.environ.get("ALPACA_API_SECRET", "")

            if not api_key or not api_secret:
                return False, "ALPACA_API_KEY or ALPACA_API_SECRET not set"

            # Try importing alpaca client
            try:
                import requests
                url = "https://paper-api.alpaca.markets/v2/account"
                headers = {
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": api_secret,
                }
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    equity = data.get("equity", "?")
                    return True, f"Connected — equity: ${float(equity):,.2f}"
                return False, f"API returned {resp.status_code}: {resp.text[:100]}"
            except Exception as e:
                return False, f"Connection failed: {e}"

        return self._run_check("alpaca_connectivity", "broker", _check)

    # ------------------------------------------------------------------
    # 4. Model files
    # ------------------------------------------------------------------

    DEFAULT_MODEL_PATHS = [
        "ml/models",
        "data/models",
    ]

    def check_model_files(
        self, model_paths: Optional[List[str]] = None,
    ) -> CheckResult:
        """Verify trained model artifacts exist."""
        def _check():
            paths = model_paths or self.DEFAULT_MODEL_PATHS
            found = []
            missing = []
            for p in paths:
                full = self.root / p
                if full.exists() and full.is_dir():
                    files = list(full.glob("*.pkl")) + list(full.glob("*.joblib")) + list(full.glob("*.json"))
                    if files:
                        found.append(f"{p}: {len(files)} files")
                    else:
                        missing.append(f"{p}: dir exists but no model files")
                else:
                    missing.append(f"{p}: not found")

            if found:
                return True, "; ".join(found)
            if missing:
                return False, "; ".join(missing)
            return False, "No model directories configured"

        return self._run_check("model_files", "models", _check)

    # ------------------------------------------------------------------
    # 5. Directory permissions
    # ------------------------------------------------------------------

    REQUIRED_DIRS = ["data", "output", "reports", "logs"]

    def check_directory_permissions(
        self, dirs: Optional[List[str]] = None,
    ) -> CheckResult:
        """Verify data/output directories exist and are writable."""
        def _check():
            check_dirs = dirs or self.REQUIRED_DIRS
            issues = []
            for d in check_dirs:
                full = self.root / d
                if not full.exists():
                    try:
                        full.mkdir(parents=True, exist_ok=True)
                    except OSError as e:
                        issues.append(f"{d}: cannot create ({e})")
                        continue

                # Test write
                test_file = full / ".write_test"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                except OSError as e:
                    issues.append(f"{d}: not writable ({e})")

            if issues:
                return False, "; ".join(issues)
            return True, f"All {len(check_dirs)} directories writable"

        return self._run_check("directory_permissions", "filesystem", _check)

    # ------------------------------------------------------------------
    # 6. Telegram config
    # ------------------------------------------------------------------

    def check_telegram_config(self) -> CheckResult:
        """Validate Telegram bot credentials if configured."""
        def _check():
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

            if not token and not chat_id:
                return True, "Telegram not configured (optional — alerts will use logging)"

            if token and not chat_id:
                return False, "TELEGRAM_BOT_TOKEN set but TELEGRAM_CHAT_ID missing"
            if chat_id and not token:
                return False, "TELEGRAM_CHAT_ID set but TELEGRAM_BOT_TOKEN missing"

            if len(token) < 20:
                return False, "TELEGRAM_BOT_TOKEN looks invalid (too short)"

            return True, "Telegram credentials present"

        return self._run_check("telegram_config", "alerts", _check)

    # ------------------------------------------------------------------
    # 7. Dry trade simulation
    # ------------------------------------------------------------------

    def check_dry_trade(self) -> CheckResult:
        """Run a signal→order pipeline without hitting the broker."""
        def _check():
            from compass.live_trading_blueprint import (
                LiveTradingBlueprint, SimulatedBroker, StrategySignal, RiskCheckResult,
            )

            broker = SimulatedBroker(fill_rate=1.0)
            ltb = LiveTradingBlueprint(broker=broker, starting_equity=100000)

            signal = StrategySignal(
                signal_id="DRY-001", strategy="credit_spread",
                symbol="SPY", direction="short", confidence=0.85,
                target_contracts=2, spread_width=5.0, entry_price=1.50,
            )

            risk, order = ltb.process_signal(signal)

            if risk.result != RiskCheckResult.APPROVED:
                return False, f"Risk check rejected: {risk.reject_reasons}"

            if order is None:
                return False, "Order not created"

            if order.filled_qty == 0:
                return False, "Order not filled in simulation"

            if len(ltb.positions) == 0:
                return False, "No position created"

            if len(ltb.audit_trail) == 0:
                return False, "Audit trail empty"

            return True, (f"Dry trade OK: {order.status.value}, "
                          f"{len(ltb.positions)} position, "
                          f"{len(ltb.audit_trail)} audit entries")

        return self._run_check("dry_trade_simulation", "simulation", _check)

    # ------------------------------------------------------------------
    # 8. Full validation
    # ------------------------------------------------------------------

    def validate(
        self, config: Optional[Dict] = None,
        model_paths: Optional[List[str]] = None,
        dirs: Optional[List[str]] = None,
        skip_alpaca: bool = False,
    ) -> ValidationReport:
        """Run all pre-flight checks."""
        self._checks.clear()
        t0 = time.perf_counter()

        self.check_module_imports()
        self.check_config_schema(config)
        if not skip_alpaca:
            self.check_alpaca_connectivity()
        self.check_model_files(model_paths)
        self.check_directory_permissions(dirs)
        self.check_telegram_config()
        self.check_dry_trade()

        total_ms = (time.perf_counter() - t0) * 1000
        n_passed = sum(1 for c in self._checks if c.passed)
        n_failed = sum(1 for c in self._checks if not c.passed)

        return ValidationReport(
            checks=list(self._checks),
            all_passed=n_failed == 0,
            n_passed=n_passed,
            n_failed=n_failed,
            total_duration_ms=total_ms,
            timestamp=datetime.now(),
        )

    @property
    def checks(self) -> List[CheckResult]:
        return list(self._checks)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, report: ValidationReport,
        output_path: str = "reports/deployment_validation.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        status_color = "#059669" if report.all_passed else "#dc2626"
        status_text = "ALL CHECKS PASSED" if report.all_passed else f"{report.n_failed} CHECK(S) FAILED"

        rows = []
        for c in report.checks:
            color = "#059669" if c.passed else "#dc2626"
            icon = "PASS" if c.passed else "FAIL"
            rows.append(
                f"<tr><td style='text-align:left'>{c.name}</td>"
                f"<td>{c.category}</td>"
                f"<td style='color:{color};font-weight:700'>{icon}</td>"
                f"<td>{c.duration_ms:.1f}</td>"
                f"<td style='text-align:left'>{c.detail}</td></tr>")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Deployment Validation</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fff; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: .5rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #f1f5f9; padding: 10px 12px; text-align: right; border-bottom: 2px solid #e2e8f0; font-size: .9rem; }}
th:first-child {{ text-align: left; }}
td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #f1f5f9; font-size: .9rem; }}
td:first-child {{ text-align: left; font-weight: 500; }}
.banner {{ background: {status_color}; color: #fff; padding: 1rem 1.5rem; border-radius: 8px;
           font-size: 1.2rem; font-weight: 700; margin: 1rem 0; }}
.summary {{ color: #64748b; margin: .5rem 0; }}
</style></head><body>
<h1>Pre-Flight Deployment Validation</h1>
<div class="banner">{status_text}</div>
<p class="summary">{report.n_passed} passed, {report.n_failed} failed | {report.total_duration_ms:.0f}ms total |
{report.timestamp.strftime('%Y-%m-%d %H:%M') if report.timestamp else ''}</p>
<table>
<tr><th style='text-align:left'>Check</th><th>Category</th><th>Result</th><th>Time (ms)</th><th style='text-align:left'>Detail</th></tr>
{''.join(rows)}
</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)
