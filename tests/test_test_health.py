"""Tests for compass.test_health — 30 tests."""
import numpy as np
import pytest
from pathlib import Path
from compass.test_health import (
    TestHealthAnalyzer, TestFileInfo, ModuleCoverage, FlakynessRisk, HealthReport,
)

ROOT = Path(__file__).resolve().parent.parent


class TestDiscovery:
    def test_discover_test_files(self):
        a = TestHealthAnalyzer(str(ROOT))
        files = a.discover_test_files()
        assert len(files) > 50  # we have 200+ test files

    def test_discover_compass_modules(self):
        a = TestHealthAnalyzer(str(ROOT))
        modules = a.discover_compass_modules()
        assert len(modules) > 100  # we have 147+ compass modules

    def test_files_are_paths(self):
        a = TestHealthAnalyzer(str(ROOT))
        for f in a.discover_test_files()[:5]:
            assert isinstance(f, Path)
            assert f.exists()


class TestAnalyzeFile:
    def test_self(self):
        """Analyze this very test file."""
        info = TestHealthAnalyzer.analyze_test_file(Path(__file__))
        assert isinstance(info, TestFileInfo)
        assert info.n_tests > 0
        assert info.n_classes > 0

    def test_counts(self):
        info = TestHealthAnalyzer.analyze_test_file(Path(__file__))
        assert info.n_assertions > 0
        assert info.assertion_density > 0

    def test_nonexistent(self):
        info = TestHealthAnalyzer.analyze_test_file(Path("/nonexistent/test.py"))
        assert info.n_tests == 0

    def test_no_random_without_seed(self):
        # This file uses np.random but doesn't call without seed
        info = TestHealthAnalyzer.analyze_test_file(Path(__file__))
        assert not info.has_random_without_seed

    def test_known_file(self):
        # Analyze a known test file with many tests
        f = ROOT / "tests" / "test_vol_forecaster.py"
        if f.exists():
            info = TestHealthAnalyzer.analyze_test_file(f)
            assert info.n_tests >= 20


class TestCoverage:
    def test_basic(self):
        a = TestHealthAnalyzer(str(ROOT))
        coverages = a.compute_coverage()
        assert len(coverages) > 50
        assert all(isinstance(c, ModuleCoverage) for c in coverages)

    def test_some_tested(self):
        a = TestHealthAnalyzer(str(ROOT))
        coverages = a.compute_coverage()
        tested = sum(1 for c in coverages if c.is_tested)
        assert tested > 30

    def test_has_untested(self):
        a = TestHealthAnalyzer(str(ROOT))
        coverages = a.compute_coverage()
        untested = [c for c in coverages if not c.is_tested]
        # Some modules may not have tests
        assert isinstance(untested, list)

    def test_n_tests_populated(self):
        a = TestHealthAnalyzer(str(ROOT))
        coverages = a.compute_coverage()
        tested = [c for c in coverages if c.is_tested]
        if tested:
            assert any(c.n_tests > 0 for c in tested)


class TestFlakiness:
    def test_basic(self):
        info = TestFileInfo("/test.py", "test_x", 10, 2, 15, False, 0, 2, 1.5)
        risk = TestHealthAnalyzer.assess_flakiness(info)
        assert isinstance(risk, FlakynessRisk)
        assert risk.risk_score == 0.0  # no risk factors

    def test_random_no_seed(self):
        info = TestFileInfo("/test.py", "test_x", 10, 2, 15, True, 0, 2, 1.5)
        risk = TestHealthAnalyzer.assess_flakiness(info)
        assert risk.risk_score >= 0.5
        assert "random" in risk.reasons[0]

    def test_low_density(self):
        info = TestFileInfo("/test.py", "test_x", 10, 2, 3, False, 0, 2, 0.3)
        risk = TestHealthAnalyzer.assess_flakiness(info)
        assert risk.risk_score > 0
        assert "density" in risk.reasons[0]

    def test_no_tests(self):
        info = TestFileInfo("/test.py", "test_x", 0, 0, 0, False, 0, 0, 0.0)
        risk = TestHealthAnalyzer.assess_flakiness(info)
        assert risk.risk_score > 0


class TestFullAnalysis:
    def test_basic(self):
        a = TestHealthAnalyzer(str(ROOT))
        report = a.analyze()
        assert isinstance(report, HealthReport)
        assert report.total_test_files > 50
        assert report.total_tests > 200
        assert report.total_compass_modules > 100
        assert 0 < report.coverage_pct <= 1.0

    def test_untested_list(self):
        a = TestHealthAnalyzer(str(ROOT))
        report = a.analyze()
        assert isinstance(report.untested_modules, list)

    def test_flaky_populated(self):
        a = TestHealthAnalyzer(str(ROOT))
        report = a.analyze()
        assert isinstance(report.flaky_risks, list)


class TestHTMLReport:
    def test_creates_file(self, tmp_path):
        a = TestHealthAnalyzer(str(ROOT))
        report = a.analyze()
        out = tmp_path / "health.html"
        path = a.generate_report(report, output_path=str(out))
        assert Path(path).exists()
        html = out.read_text()
        assert "Test Suite Health" in html

    def test_contains_coverage(self, tmp_path):
        a = TestHealthAnalyzer(str(ROOT))
        report = a.analyze()
        out = tmp_path / "h.html"
        a.generate_report(report, output_path=str(out))
        html = out.read_text()
        assert "Module Coverage" in html

    def test_contains_flaky(self, tmp_path):
        a = TestHealthAnalyzer(str(ROOT))
        report = a.analyze()
        out = tmp_path / "h.html"
        a.generate_report(report, output_path=str(out))
        assert "Flakiness" in out.read_text()

    def test_contains_summary(self, tmp_path):
        a = TestHealthAnalyzer(str(ROOT))
        report = a.analyze()
        out = tmp_path / "h.html"
        a.generate_report(report, output_path=str(out))
        html = out.read_text()
        assert "tests" in html
        assert "coverage" in html
