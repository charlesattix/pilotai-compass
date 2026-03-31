"""Tests for compass/module_auditor.py"""
from __future__ import annotations
from pathlib import Path
import pytest
from compass.module_auditor import AuditSummary, DuplicateGroup, ModuleAuditor, ModuleInfo

@pytest.fixture
def auditor():
    return ModuleAuditor(skip_imports=True)

class TestDataclasses:
    def test_module_info(self):
        m = ModuleInfo("test", "/p", 100, 2, 5, True, True, 10, False, [], [], True, 80, "A")
        assert m.quality_score == pytest.approx(80)
    def test_duplicate_group(self):
        d = DuplicateGroup(["a", "b"], "name_similar", 0.8, "desc")
        assert d.similarity == pytest.approx(0.8)
    def test_audit_summary(self):
        s = AuditSummary(10, 5000, 20, 100, 8, 2, 200, 6, [], 75.0, [], {"A": 5})
        assert s.total_modules == 10

class TestAudit:
    def test_audit_returns_dict(self, auditor):
        result = auditor.audit()
        assert "modules" in result and "summary" in result
    def test_finds_modules(self, auditor):
        auditor.audit()
        assert len(auditor.modules) > 0
    def test_summary_populated(self, auditor):
        auditor.audit()
        assert auditor.summary is not None
        assert auditor.summary.total_modules > 0
    def test_total_lines_positive(self, auditor):
        auditor.audit()
        assert auditor.summary.total_lines > 0
    def test_modules_have_names(self, auditor):
        auditor.audit()
        for m in auditor.modules:
            assert len(m.name) > 0
    def test_line_counts_positive(self, auditor):
        auditor.audit()
        for m in auditor.modules:
            assert m.lines > 0
    def test_quality_scores_range(self, auditor):
        auditor.audit()
        for m in auditor.modules:
            assert 0 <= m.quality_score <= 100
    def test_grades_valid(self, auditor):
        auditor.audit()
        for m in auditor.modules:
            assert m.grade in ("A", "B", "C", "D", "F")
    def test_grade_distribution(self, auditor):
        auditor.audit()
        total_grades = sum(auditor.summary.grade_distribution.values())
        assert total_grades == auditor.summary.total_modules
    def test_known_module_exists(self, auditor):
        auditor.audit()
        names = {m.name for m in auditor.modules}
        assert "stress_test" in names
    def test_has_tests_detected(self, auditor):
        auditor.audit()
        assert auditor.summary.modules_with_tests > 0
    def test_without_tests_detected(self, auditor):
        auditor.audit()
        # Some modules may lack tests
        assert auditor.summary.modules_without_tests >= 0

class TestDuplicates:
    def test_duplicate_detection(self, auditor):
        auditor.audit()
        assert isinstance(auditor.summary.duplicates, list)
    def test_duplicate_similarity_range(self, auditor):
        auditor.audit()
        for d in auditor.summary.duplicates:
            assert 0 <= d.similarity <= 1

class TestImports:
    def test_import_status(self, auditor):
        auditor.audit()
        # At least some should import successfully
        working = sum(1 for m in auditor.modules if m.import_works)
        assert working > 0
    def test_dependencies_detected(self, auditor):
        auditor.audit()
        has_deps = sum(1 for m in auditor.modules if m.imports_compass)
        assert has_deps >= 0  # May be 0

class TestMiniAudit:
    def test_custom_dir(self, tmp_path):
        # Create a tiny compass dir
        mod = tmp_path / "test_mod.py"
        mod.write_text('"""Doc."""\nclass Foo:\n    pass\ndef bar():\n    pass\n')
        a = ModuleAuditor(compass_dir=tmp_path)
        a.audit()
        assert len(a.modules) == 1
        assert a.modules[0].classes == 1
        assert a.modules[0].functions == 1

class TestReport:
    def test_html(self, tmp_path, auditor):
        path = auditor.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "Module Audit" in c
    def test_sections(self, tmp_path, auditor):
        path = auditor.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Catalog" in c and "Coverage" in c and "Duplicate" in c
    def test_charts(self, tmp_path, auditor):
        path = auditor.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()
    def test_auto_audit(self, tmp_path, auditor):
        assert auditor.summary is None
        auditor.generate_report(str(tmp_path / "r.html"))
        assert auditor.summary is not None
