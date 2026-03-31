from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from compass.deploy_checklist import (
    CheckCategory,
    CheckResult,
    DeployChecklist,
    DeployResult,
    ManualCheck,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(tmp_path: Path, *, extras: dict[str, str] | None = None) -> Path:
    """Create a minimal valid project skeleton under tmp_path."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "requirements.txt").write_text("pandas\n")
    (root / "data").mkdir()
    (root / "data" / "models").mkdir()
    (root / "data" / "models" / "signal.joblib").write_bytes(b"fake")
    (root / "reports").mkdir()
    (root / "shared").mkdir()
    (root / "shared" / "constants.py").write_text("MAX_LOSS = 500\nMAX_DRAWDOWN = 0.1\n")
    (root / "compass").mkdir()
    (root / "compass" / "crisis_hedge.py").write_text("# kill switch\n")
    (root / "compass" / "anomaly_detector.py").write_text("# anomaly\n")
    (root / "shared" / "telegram_alerts.py").write_text("# alerts\n")
    # A file that sets up logging
    (root / "main.py").write_text("import logging\nlogging.basicConfig()\n")
    if extras:
        for rel, content in extras.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
    return root


# ---------------------------------------------------------------------------
# CheckResult dataclass
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_valid_statuses(self):
        for s in ("pass", "fail", "warn"):
            cr = CheckResult(name="t", status=s, detail="ok", category=CheckCategory.CONFIG)
            assert cr.status == s

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="Invalid status"):
            CheckResult(name="t", status="bad", detail="x", category=CheckCategory.CONFIG)


# ---------------------------------------------------------------------------
# CheckCategory
# ---------------------------------------------------------------------------

class TestCheckCategory:
    def test_all_categories_present(self):
        expected = {"config", "credentials", "data", "models", "risk", "alerts", "logging", "performance"}
        assert {c.value for c in CheckCategory} == expected


# ---------------------------------------------------------------------------
# config_files_exist
# ---------------------------------------------------------------------------

class TestConfigFilesExist:
    def test_pass_when_all_present(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.config_files_exist()
        assert r.status == "pass"
        assert r.category == CheckCategory.CONFIG

    def test_fail_when_missing(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "pytest.ini").unlink()
        cl = DeployChecklist(root)
        r = cl.config_files_exist()
        assert r.status == "fail"
        assert "pytest.ini" in r.detail

    def test_custom_key_files(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root, key_files=["nonexistent.cfg"])
        r = cl.config_files_exist()
        assert r.status == "fail"
        assert "nonexistent.cfg" in r.detail

    def test_empty_key_files_list(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root, key_files=[])
        r = cl.config_files_exist()
        assert r.status == "pass"


# ---------------------------------------------------------------------------
# no_hardcoded_credentials
# ---------------------------------------------------------------------------

class TestNoHardcodedCredentials:
    def test_pass_clean_project(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.no_hardcoded_credentials()
        assert r.status == "pass"

    def test_fail_api_key(self, tmp_path: Path):
        root = _make_project(tmp_path, extras={
            "bad.py": 'api_key = "sk-12345"\n',
        })
        cl = DeployChecklist(root)
        r = cl.no_hardcoded_credentials()
        assert r.status == "fail"
        assert "hardcoded credential" in r.detail.lower()

    def test_fail_secret(self, tmp_path: Path):
        root = _make_project(tmp_path, extras={
            "secrets.py": "secret = 'my_secret_value'\n",
        })
        cl = DeployChecklist(root)
        r = cl.no_hardcoded_credentials()
        assert r.status == "fail"

    def test_fail_password(self, tmp_path: Path):
        root = _make_project(tmp_path, extras={
            "config_loader.py": 'password = "hunter2"\n',
        })
        cl = DeployChecklist(root)
        r = cl.no_hardcoded_credentials()
        assert r.status == "fail"

    def test_env_var_not_flagged(self, tmp_path: Path):
        root = _make_project(tmp_path, extras={
            "safe.py": 'api_key = os.environ["KEY"]\n',
        })
        cl = DeployChecklist(root)
        r = cl.no_hardcoded_credentials()
        assert r.status == "pass"


# ---------------------------------------------------------------------------
# data_dirs_exist
# ---------------------------------------------------------------------------

class TestDataDirsExist:
    def test_pass(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        assert cl.data_dirs_exist().status == "pass"

    def test_fail_no_data(self, tmp_path: Path):
        root = _make_project(tmp_path)
        # Remove data dir contents then dir
        for item in (root / "data").rglob("*"):
            if item.is_file():
                item.unlink()
        for item in sorted((root / "data").rglob("*"), reverse=True):
            if item.is_dir():
                item.rmdir()
        (root / "data").rmdir()
        cl = DeployChecklist(root)
        r = cl.data_dirs_exist()
        assert r.status == "fail"
        assert "data" in r.detail

    def test_fail_no_reports(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "reports").rmdir()
        cl = DeployChecklist(root)
        r = cl.data_dirs_exist()
        assert r.status == "fail"
        assert "reports" in r.detail


# ---------------------------------------------------------------------------
# model_artifacts_exist
# ---------------------------------------------------------------------------

class TestModelArtifactsExist:
    def test_pass(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.model_artifacts_exist()
        assert r.status == "pass"
        assert "1" in r.detail

    def test_fail_no_models_dir(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "data" / "models" / "signal.joblib").unlink()
        (root / "data" / "models").rmdir()
        cl = DeployChecklist(root)
        r = cl.model_artifacts_exist()
        assert r.status == "fail"

    def test_fail_no_joblib_files(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "data" / "models" / "signal.joblib").unlink()
        cl = DeployChecklist(root)
        r = cl.model_artifacts_exist()
        assert r.status == "fail"
        assert "No .joblib" in r.detail


# ---------------------------------------------------------------------------
# model_freshness
# ---------------------------------------------------------------------------

class TestModelFreshness:
    def test_pass_recent_model(self, tmp_path: Path):
        root = _make_project(tmp_path)
        # File was just created so it's fresh
        cl = DeployChecklist(root, model_freshness_days=7)
        r = cl.model_freshness()
        assert r.status == "pass"

    def test_warn_stale_model(self, tmp_path: Path):
        root = _make_project(tmp_path)
        model = root / "data" / "models" / "signal.joblib"
        old_time = time.time() - (10 * 86400)
        os.utime(model, (old_time, old_time))
        cl = DeployChecklist(root, model_freshness_days=7)
        r = cl.model_freshness()
        assert r.status == "warn"
        assert "Stale" in r.detail

    def test_custom_freshness_threshold(self, tmp_path: Path):
        root = _make_project(tmp_path)
        model = root / "data" / "models" / "signal.joblib"
        old_time = time.time() - (3 * 86400)
        os.utime(model, (old_time, old_time))
        # With 2-day threshold it should warn
        cl = DeployChecklist(root, model_freshness_days=2)
        r = cl.model_freshness()
        assert r.status == "warn"
        # With 5-day threshold it should pass
        cl2 = DeployChecklist(root, model_freshness_days=5)
        r2 = cl2.model_freshness()
        assert r2.status == "pass"

    def test_fail_no_models_dir(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "data" / "models" / "signal.joblib").unlink()
        (root / "data" / "models").rmdir()
        cl = DeployChecklist(root)
        r = cl.model_freshness()
        assert r.status == "fail"

    def test_fail_no_joblib_files(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "data" / "models" / "signal.joblib").unlink()
        cl = DeployChecklist(root)
        r = cl.model_freshness()
        assert r.status == "fail"


# ---------------------------------------------------------------------------
# risk_limits_configured
# ---------------------------------------------------------------------------

class TestRiskLimitsConfigured:
    def test_pass(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.risk_limits_configured()
        assert r.status == "pass"
        assert "max_loss" in r.detail.lower() or "max_drawdown" in r.detail.lower()

    def test_fail_no_constants(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "shared" / "constants.py").unlink()
        cl = DeployChecklist(root)
        r = cl.risk_limits_configured()
        assert r.status == "fail"

    def test_fail_no_risk_keywords(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "shared" / "constants.py").write_text("FOO = 1\nBAR = 2\n")
        cl = DeployChecklist(root)
        r = cl.risk_limits_configured()
        assert r.status == "fail"


# ---------------------------------------------------------------------------
# kill_switch_available
# ---------------------------------------------------------------------------

class TestKillSwitchAvailable:
    def test_pass(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        assert cl.kill_switch_available().status == "pass"

    def test_fail(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "compass" / "crisis_hedge.py").unlink()
        cl = DeployChecklist(root)
        r = cl.kill_switch_available()
        assert r.status == "fail"
        assert "kill switch" in r.detail.lower()


# ---------------------------------------------------------------------------
# logging_configured
# ---------------------------------------------------------------------------

class TestLoggingConfigured:
    def test_pass(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        assert cl.logging_configured().status == "pass"

    def test_fail_no_logging(self, tmp_path: Path):
        root = tmp_path / "bare"
        root.mkdir()
        (root / "app.py").write_text("print('hello')\n")
        cl = DeployChecklist(root)
        r = cl.logging_configured()
        assert r.status == "fail"


# ---------------------------------------------------------------------------
# alert_integration
# ---------------------------------------------------------------------------

class TestAlertIntegration:
    def test_pass_anomaly_detector(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.alert_integration()
        assert r.status == "pass"

    def test_pass_telegram_only(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "compass" / "anomaly_detector.py").unlink()
        cl = DeployChecklist(root)
        r = cl.alert_integration()
        assert r.status == "pass"
        assert "telegram_alerts" in r.detail

    def test_fail_no_alerts(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "compass" / "anomaly_detector.py").unlink()
        (root / "shared" / "telegram_alerts.py").unlink()
        cl = DeployChecklist(root)
        r = cl.alert_integration()
        assert r.status == "fail"


# ---------------------------------------------------------------------------
# paper_trading_duration
# ---------------------------------------------------------------------------

class TestPaperTradingDuration:
    def test_pass_enough_days(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.paper_trading_duration(days_traded=20)
        assert r.status == "pass"

    def test_pass_exact_threshold(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.paper_trading_duration(days_traded=14)
        assert r.status == "pass"

    def test_warn_insufficient(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.paper_trading_duration(days_traded=5)
        assert r.status == "warn"
        assert "5 days" in r.detail

    def test_warn_zero_days(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.paper_trading_duration(days_traded=0)
        assert r.status == "warn"


# ---------------------------------------------------------------------------
# test_suite_health
# ---------------------------------------------------------------------------

class TestTestSuiteHealth:
    def test_pass_all_green(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.test_suite_health(n_passed=100, n_failed=0)
        assert r.status == "pass"

    def test_fail_with_failures(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.test_suite_health(n_passed=95, n_failed=5)
        assert r.status == "fail"
        assert "5/100" in r.detail

    def test_warn_no_results(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        r = cl.test_suite_health(n_passed=0, n_failed=0)
        assert r.status == "warn"


# ---------------------------------------------------------------------------
# Manual check tracking
# ---------------------------------------------------------------------------

class TestManualChecks:
    def test_set_and_get(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        cl.set_manual_check("code_review", "pass", "alice", "2026-03-30")
        checks = cl.get_manual_checks()
        assert "code_review" in checks
        assert checks["code_review"].status == "pass"
        assert checks["code_review"].signed_by == "alice"

    def test_invalid_manual_status(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        with pytest.raises(ValueError):
            cl.set_manual_check("code_review", "invalid", "bob", "2026-03-30")

    def test_multiple_manual_checks(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        cl.set_manual_check("code_review", "pass", "alice", "2026-03-30")
        cl.set_manual_check("security_audit", "pending", "bob", "2026-03-30")
        checks = cl.get_manual_checks()
        assert len(checks) == 2

    def test_pending_manual_is_blocker(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        cl.set_manual_check("code_review", "pending", "alice", "2026-03-30")
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        assert not result.go_no_go
        assert any("code_review" in b for b in result.blockers)

    def test_failed_manual_is_blocker(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        cl.set_manual_check("security_audit", "fail", "bob", "2026-03-30")
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        assert not result.go_no_go


# ---------------------------------------------------------------------------
# run_all / go-no-go / blockers
# ---------------------------------------------------------------------------

class TestRunAll:
    def test_go_all_pass(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        assert result.go_no_go is True
        assert result.blockers == []
        assert len(result.checks) == 11

    def test_no_go_with_failure(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "pytest.ini").unlink()
        cl = DeployChecklist(root)
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        assert result.go_no_go is False
        assert len(result.blockers) > 0

    def test_blocker_list_populated(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "pytest.ini").unlink()
        (root / "compass" / "crisis_hedge.py").unlink()
        cl = DeployChecklist(root)
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        assert len(result.blockers) >= 2
        names = " ".join(result.blockers)
        assert "config_files_exist" in names
        assert "kill_switch_available" in names

    def test_warn_does_not_block(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        # days_traded=5 gives a warn, not a fail
        result = cl.run_all(days_traded=5, n_passed=50, n_failed=0)
        assert result.go_no_go is True

    def test_test_failures_block(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        result = cl.run_all(days_traded=14, n_passed=95, n_failed=5)
        assert result.go_no_go is False
        assert any("test_suite_health" in b for b in result.blockers)

    def test_generated_at_present(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        assert result.generated_at is not None
        assert len(result.generated_at) > 0

    def test_deploy_result_fields(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        result = cl.run_all()
        assert isinstance(result, DeployResult)
        assert isinstance(result.checks, list)
        assert isinstance(result.manual_checks, dict)
        assert isinstance(result.blockers, list)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

class TestHTMLReport:
    def test_report_contains_checks(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        html = cl.generate_report(result)
        assert "config_files_exist" in html
        assert "model_freshness" in html

    def test_report_go_badge(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        html = cl.generate_report(result)
        assert ">GO<" in html
        assert "#28a745" in html  # green

    def test_report_no_go_badge(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "pytest.ini").unlink()
        cl = DeployChecklist(root)
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        html = cl.generate_report(result)
        assert "NO-GO" in html
        assert "#dc3545" in html  # red

    def test_report_includes_blockers(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "pytest.ini").unlink()
        cl = DeployChecklist(root)
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        html = cl.generate_report(result)
        assert "config_files_exist" in html
        assert "<li>" in html

    def test_report_includes_manual_checks(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        cl.set_manual_check("code_review", "pass", "alice", "2026-03-30")
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        html = cl.generate_report(result)
        assert "code_review" in html
        assert "alice" in html

    def test_report_is_valid_html(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        result = cl.run_all(days_traded=14, n_passed=50, n_failed=0)
        html = cl.generate_report(result)
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_report_status_colors(self, tmp_path: Path):
        root = _make_project(tmp_path)
        cl = DeployChecklist(root)
        # Force a warn by low paper trading days
        result = cl.run_all(days_traded=5, n_passed=50, n_failed=0)
        html = cl.generate_report(result)
        assert "#ffc107" in html  # yellow for warn
        assert "#28a745" in html  # green for pass


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_project_dir(self, tmp_path: Path):
        root = tmp_path / "empty"
        root.mkdir()
        cl = DeployChecklist(root)
        result = cl.run_all()
        assert result.go_no_go is False
        assert len(result.blockers) > 0

    def test_missing_all_dirs(self, tmp_path: Path):
        root = tmp_path / "bare"
        root.mkdir()
        cl = DeployChecklist(root)
        r = cl.data_dirs_exist()
        assert r.status == "fail"
        assert "data" in r.detail
        assert "reports" in r.detail

    def test_no_py_files_credentials_check(self, tmp_path: Path):
        root = tmp_path / "nopy"
        root.mkdir()
        cl = DeployChecklist(root)
        r = cl.no_hardcoded_credentials()
        assert r.status == "pass"

    def test_no_py_files_logging_check(self, tmp_path: Path):
        root = tmp_path / "nopy2"
        root.mkdir()
        cl = DeployChecklist(root)
        r = cl.logging_configured()
        assert r.status == "fail"

    def test_multiple_models_all_fresh(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "data" / "models" / "model_a.joblib").write_bytes(b"a")
        (root / "data" / "models" / "model_b.joblib").write_bytes(b"b")
        cl = DeployChecklist(root)
        r = cl.model_artifacts_exist()
        assert r.status == "pass"
        assert "3" in r.detail  # signal.joblib + model_a + model_b

    def test_hardcoded_creds_multiple_files(self, tmp_path: Path):
        root = _make_project(tmp_path, extras={
            "a.py": 'api_key = "key1"\n',
            "sub/b.py": 'password = "pw"\n',
        })
        cl = DeployChecklist(root)
        r = cl.no_hardcoded_credentials()
        assert r.status == "fail"
        assert "2" in r.detail  # 2 violations
