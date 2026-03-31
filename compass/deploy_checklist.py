from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from html import escape
from pathlib import Path
from typing import Dict, List, Optional


class CheckCategory(str, Enum):
    CONFIG = "config"
    CREDENTIALS = "credentials"
    DATA = "data"
    MODELS = "models"
    RISK = "risk"
    ALERTS = "alerts"
    LOGGING = "logging"
    PERFORMANCE = "performance"


@dataclass
class CheckResult:
    name: str
    status: str  # "pass", "fail", "warn"
    detail: str
    category: CheckCategory

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail", "warn"):
            raise ValueError(f"Invalid status: {self.status!r}")


@dataclass
class ManualCheck:
    status: str  # "pass", "fail", "pending"
    signed_by: str
    date: str


@dataclass
class DeployResult:
    checks: List[CheckResult]
    manual_checks: Dict[str, ManualCheck]
    go_no_go: bool
    blockers: List[str]
    generated_at: str


class DeployChecklist:
    """Pre-deploy verification checklist for PilotAI credit-spreads."""

    # Default key files that must exist in the project root.
    DEFAULT_KEY_FILES: List[str] = [
        "pytest.ini",
        "requirements.txt",
    ]

    # Patterns that indicate hardcoded credentials.
    CREDENTIAL_PATTERNS: List[str] = [
        r"""api_key\s*=\s*['"][^'"]+['"]""",
        r"""secret\s*=\s*['"][^'"]+['"]""",
        r"""password\s*=\s*['"][^'"]+['"]""",
    ]

    def __init__(
        self,
        project_root: Path,
        *,
        key_files: Optional[List[str]] = None,
        model_freshness_days: int = 7,
    ) -> None:
        self.project_root = Path(project_root)
        self.key_files = key_files if key_files is not None else list(self.DEFAULT_KEY_FILES)
        self.model_freshness_days = model_freshness_days
        self._manual_checks: Dict[str, ManualCheck] = {}

    # ------------------------------------------------------------------
    # Manual check tracker
    # ------------------------------------------------------------------

    def set_manual_check(
        self, name: str, status: str, signed_by: str, date: str
    ) -> None:
        if status not in ("pass", "fail", "pending"):
            raise ValueError(f"Invalid manual check status: {status!r}")
        self._manual_checks[name] = ManualCheck(
            status=status, signed_by=signed_by, date=date
        )

    def get_manual_checks(self) -> Dict[str, ManualCheck]:
        return dict(self._manual_checks)

    # ------------------------------------------------------------------
    # Automated checks
    # ------------------------------------------------------------------

    def config_files_exist(self) -> CheckResult:
        missing: List[str] = []
        for fname in self.key_files:
            if not (self.project_root / fname).exists():
                missing.append(fname)
        if missing:
            return CheckResult(
                name="config_files_exist",
                status="fail",
                detail=f"Missing config files: {', '.join(missing)}",
                category=CheckCategory.CONFIG,
            )
        return CheckResult(
            name="config_files_exist",
            status="pass",
            detail="All key config files present",
            category=CheckCategory.CONFIG,
        )

    def no_hardcoded_credentials(self) -> CheckResult:
        violations: List[str] = []
        py_files = list(self.project_root.rglob("*.py"))
        compiled = [re.compile(p) for p in self.CREDENTIAL_PATTERNS]
        for py_file in py_files:
            try:
                content = py_file.read_text(errors="ignore")
            except OSError:
                continue
            for pattern in compiled:
                matches = pattern.findall(content)
                for m in matches:
                    rel = py_file.relative_to(self.project_root)
                    violations.append(f"{rel}: {m.strip()}")
        if violations:
            return CheckResult(
                name="no_hardcoded_credentials",
                status="fail",
                detail=f"Found {len(violations)} hardcoded credential(s): {'; '.join(violations[:5])}",
                category=CheckCategory.CREDENTIALS,
            )
        return CheckResult(
            name="no_hardcoded_credentials",
            status="pass",
            detail="No hardcoded credentials detected",
            category=CheckCategory.CREDENTIALS,
        )

    def data_dirs_exist(self) -> CheckResult:
        required = ["data", "reports"]
        missing = [d for d in required if not (self.project_root / d).is_dir()]
        if missing:
            return CheckResult(
                name="data_dirs_exist",
                status="fail",
                detail=f"Missing directories: {', '.join(missing)}",
                category=CheckCategory.DATA,
            )
        return CheckResult(
            name="data_dirs_exist",
            status="pass",
            detail="Required data directories exist",
            category=CheckCategory.DATA,
        )

    def model_artifacts_exist(self) -> CheckResult:
        models_dir = self.project_root / "data" / "models"
        if not models_dir.is_dir():
            return CheckResult(
                name="model_artifacts_exist",
                status="fail",
                detail="data/models/ directory not found",
                category=CheckCategory.MODELS,
            )
        joblib_files = list(models_dir.glob("*.joblib"))
        if not joblib_files:
            return CheckResult(
                name="model_artifacts_exist",
                status="fail",
                detail="No .joblib model files found in data/models/",
                category=CheckCategory.MODELS,
            )
        return CheckResult(
            name="model_artifacts_exist",
            status="pass",
            detail=f"Found {len(joblib_files)} model artifact(s)",
            category=CheckCategory.MODELS,
        )

    def model_freshness(self) -> CheckResult:
        models_dir = self.project_root / "data" / "models"
        if not models_dir.is_dir():
            return CheckResult(
                name="model_freshness",
                status="fail",
                detail="data/models/ directory not found",
                category=CheckCategory.MODELS,
            )
        joblib_files = list(models_dir.glob("*.joblib"))
        if not joblib_files:
            return CheckResult(
                name="model_freshness",
                status="fail",
                detail="No .joblib files to check freshness",
                category=CheckCategory.MODELS,
            )
        now = time.time()
        threshold = self.model_freshness_days * 86400
        stale: List[str] = []
        for f in joblib_files:
            age_seconds = now - f.stat().st_mtime
            if age_seconds > threshold:
                age_days = int(age_seconds / 86400)
                stale.append(f"{f.name} ({age_days}d old)")
        if stale:
            return CheckResult(
                name="model_freshness",
                status="warn",
                detail=f"Stale models (>{self.model_freshness_days}d): {', '.join(stale)}",
                category=CheckCategory.MODELS,
            )
        return CheckResult(
            name="model_freshness",
            status="pass",
            detail=f"All models within {self.model_freshness_days}-day freshness window",
            category=CheckCategory.MODELS,
        )

    def risk_limits_configured(self) -> CheckResult:
        constants_path = self.project_root / "shared" / "constants.py"
        if not constants_path.exists():
            return CheckResult(
                name="risk_limits_configured",
                status="fail",
                detail="shared/constants.py not found",
                category=CheckCategory.RISK,
            )
        try:
            content = constants_path.read_text(errors="ignore")
        except OSError:
            return CheckResult(
                name="risk_limits_configured",
                status="fail",
                detail="Could not read shared/constants.py",
                category=CheckCategory.RISK,
            )
        risk_keywords = ["max_loss", "risk_limit", "max_position", "stop_loss", "max_drawdown"]
        found = [kw for kw in risk_keywords if kw.lower() in content.lower()]
        if not found:
            return CheckResult(
                name="risk_limits_configured",
                status="fail",
                detail="No risk-related configuration found in shared/constants.py",
                category=CheckCategory.RISK,
            )
        return CheckResult(
            name="risk_limits_configured",
            status="pass",
            detail=f"Risk config found: {', '.join(found)}",
            category=CheckCategory.RISK,
        )

    def kill_switch_available(self) -> CheckResult:
        path = self.project_root / "compass" / "crisis_hedge.py"
        if path.exists():
            return CheckResult(
                name="kill_switch_available",
                status="pass",
                detail="compass/crisis_hedge.py exists",
                category=CheckCategory.RISK,
            )
        return CheckResult(
            name="kill_switch_available",
            status="fail",
            detail="compass/crisis_hedge.py not found (kill switch missing)",
            category=CheckCategory.RISK,
        )

    def logging_configured(self) -> CheckResult:
        """Check that at least one .py file sets up logging."""
        py_files = list(self.project_root.rglob("*.py"))
        logging_pattern = re.compile(
            r"(logging\.basicConfig|logging\.getLogger|import logging)"
        )
        for py_file in py_files:
            try:
                content = py_file.read_text(errors="ignore")
            except OSError:
                continue
            if logging_pattern.search(content):
                return CheckResult(
                    name="logging_configured",
                    status="pass",
                    detail="Logging setup detected",
                    category=CheckCategory.LOGGING,
                )
        return CheckResult(
            name="logging_configured",
            status="fail",
            detail="No logging setup found in any .py file",
            category=CheckCategory.LOGGING,
        )

    def alert_integration(self) -> CheckResult:
        candidates = [
            self.project_root / "compass" / "anomaly_detector.py",
            self.project_root / "shared" / "telegram_alerts.py",
        ]
        found = [str(p.relative_to(self.project_root)) for p in candidates if p.exists()]
        if found:
            return CheckResult(
                name="alert_integration",
                status="pass",
                detail=f"Alert integration found: {', '.join(found)}",
                category=CheckCategory.ALERTS,
            )
        return CheckResult(
            name="alert_integration",
            status="fail",
            detail="No alert integration found (anomaly_detector.py or telegram_alerts.py)",
            category=CheckCategory.ALERTS,
        )

    def paper_trading_duration(self, days_traded: int = 0) -> CheckResult:
        if days_traded >= 14:
            return CheckResult(
                name="paper_trading_duration",
                status="pass",
                detail=f"Paper traded for {days_traded} days (>=14)",
                category=CheckCategory.PERFORMANCE,
            )
        if days_traded > 0:
            return CheckResult(
                name="paper_trading_duration",
                status="warn",
                detail=f"Only {days_traded} days of paper trading (<14 recommended)",
                category=CheckCategory.PERFORMANCE,
            )
        return CheckResult(
            name="paper_trading_duration",
            status="warn",
            detail="No paper trading days reported",
            category=CheckCategory.PERFORMANCE,
        )

    def test_suite_health(self, n_passed: int = 0, n_failed: int = 0) -> CheckResult:
        total = n_passed + n_failed
        if n_failed > 0:
            return CheckResult(
                name="test_suite_health",
                status="fail",
                detail=f"{n_failed}/{total} tests failed",
                category=CheckCategory.PERFORMANCE,
            )
        if total == 0:
            return CheckResult(
                name="test_suite_health",
                status="warn",
                detail="No test results provided",
                category=CheckCategory.PERFORMANCE,
            )
        return CheckResult(
            name="test_suite_health",
            status="pass",
            detail=f"All {n_passed} tests passed",
            category=CheckCategory.PERFORMANCE,
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_all(
        self,
        *,
        days_traded: int = 0,
        n_passed: int = 0,
        n_failed: int = 0,
    ) -> DeployResult:
        checks: List[CheckResult] = [
            self.config_files_exist(),
            self.no_hardcoded_credentials(),
            self.data_dirs_exist(),
            self.model_artifacts_exist(),
            self.model_freshness(),
            self.risk_limits_configured(),
            self.kill_switch_available(),
            self.logging_configured(),
            self.alert_integration(),
            self.paper_trading_duration(days_traded=days_traded),
            self.test_suite_health(n_passed=n_passed, n_failed=n_failed),
        ]

        blockers: List[str] = []
        for c in checks:
            if c.status == "fail":
                blockers.append(f"[{c.category.value}] {c.name}: {c.detail}")

        # Manual checks that are "fail" or "pending" are also blockers.
        for name, mc in self._manual_checks.items():
            if mc.status in ("fail", "pending"):
                blockers.append(f"[manual] {name}: {mc.status}")

        go_no_go = len(blockers) == 0

        return DeployResult(
            checks=checks,
            manual_checks=dict(self._manual_checks),
            go_no_go=go_no_go,
            blockers=blockers,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        result: DeployResult,
    ) -> str:
        status_colors = {
            "pass": "#28a745",
            "fail": "#dc3545",
            "warn": "#ffc107",
        }
        status_labels = {
            "pass": "PASS",
            "fail": "FAIL",
            "warn": "WARN",
        }

        rows = ""
        for c in result.checks:
            color = status_colors.get(c.status, "#6c757d")
            label = status_labels.get(c.status, c.status.upper())
            rows += (
                f"<tr>"
                f"<td>{escape(c.name)}</td>"
                f"<td>{escape(c.category.value)}</td>"
                f'<td style="background-color:{color};color:#fff;text-align:center;font-weight:bold;">{label}</td>'
                f"<td>{escape(c.detail)}</td>"
                f"</tr>\n"
            )

        # Manual checks
        manual_rows = ""
        for name, mc in result.manual_checks.items():
            color = status_colors.get(mc.status, "#6c757d")
            label = mc.status.upper()
            manual_rows += (
                f"<tr>"
                f"<td>{escape(name)}</td>"
                f'<td style="background-color:{color};color:#fff;text-align:center;font-weight:bold;">{label}</td>'
                f"<td>{escape(mc.signed_by)}</td>"
                f"<td>{escape(mc.date)}</td>"
                f"</tr>\n"
            )

        # Blockers
        blocker_items = ""
        for b in result.blockers:
            blocker_items += f"<li>{escape(b)}</li>\n"

        go_badge_color = "#28a745" if result.go_no_go else "#dc3545"
        go_badge_text = "GO" if result.go_no_go else "NO-GO"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Deploy Checklist Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
  th, td {{ border: 1px solid #dee2e6; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background-color: #f8f9fa; }}
  .badge {{ display: inline-block; padding: 0.5rem 1.5rem; border-radius: 0.25rem;
            color: #fff; font-size: 1.5rem; font-weight: bold; }}
</style>
</head>
<body>
<h1>Deploy Checklist Report</h1>
<p>Generated at: {escape(result.generated_at)}</p>

<h2>Go / No-Go Decision</h2>
<span class="badge" style="background-color:{go_badge_color};">{go_badge_text}</span>

<h2>Automated Checks</h2>
<table>
<thead><tr><th>Check</th><th>Category</th><th>Status</th><th>Detail</th></tr></thead>
<tbody>
{rows}</tbody>
</table>

<h2>Manual Checks</h2>
<table>
<thead><tr><th>Check</th><th>Status</th><th>Signed By</th><th>Date</th></tr></thead>
<tbody>
{manual_rows}</tbody>
</table>

<h2>Blockers</h2>
<ul>
{blocker_items}</ul>

</body>
</html>"""
        return html
