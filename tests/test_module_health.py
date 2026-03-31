"""Tests for compass.module_health using temporary directories with dummy modules."""
from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from compass.module_health import HealthResult, ModuleHealthChecker, ModuleStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def workspace(tmp_path: Path):
    """Create a minimal compass + tests workspace inside *tmp_path*."""
    compass = tmp_path / "fake_compass"
    compass.mkdir()
    tests = tmp_path / "fake_tests"
    tests.mkdir()
    # __init__.py so it is a package
    (compass / "__init__.py").write_text("")
    return compass, tests


def _write_module(compass: Path, name: str, code: str) -> Path:
    p = compass / f"{name}.py"
    p.write_text(textwrap.dedent(code))
    return p


def _write_test(tests: Path, module_name: str) -> Path:
    p = tests / f"test_{module_name}.py"
    p.write_text(f"# placeholder test for {module_name}\n")
    return p


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestModuleStatusDataclass:
    def test_defaults(self):
        s = ModuleStatus(name="foo", importable=True, has_tests=False)
        assert s.error == ""
        assert s.public_classes == []
        assert s.public_functions == []
        assert s.follows_conventions is True

    def test_custom_values(self):
        s = ModuleStatus(
            name="bar",
            importable=False,
            has_tests=True,
            error="boom",
            public_classes=["Bar"],
            public_functions=["do_stuff"],
            follows_conventions=False,
        )
        assert s.name == "bar"
        assert s.error == "boom"
        assert s.public_classes == ["Bar"]

    def test_mutable_defaults_independent(self):
        a = ModuleStatus(name="a", importable=True, has_tests=True)
        b = ModuleStatus(name="b", importable=True, has_tests=True)
        a.public_classes.append("X")
        assert b.public_classes == []


class TestHealthResultDataclass:
    def test_defaults(self):
        r = HealthResult()
        assert r.statuses == []
        assert r.n_total == 0
        assert r.import_errors == []

    def test_mutable_defaults_independent(self):
        r1 = HealthResult()
        r2 = HealthResult()
        r1.statuses.append(ModuleStatus(name="x", importable=True, has_tests=True))
        assert r2.statuses == []


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_discovers_py_files(self, workspace):
        compass, tests = workspace
        _write_module(compass, "alpha", "x = 1\n")
        _write_module(compass, "beta", "y = 2\n")
        checker = ModuleHealthChecker(compass, tests)
        found = checker._discover_modules()
        names = [p.stem for p in found]
        assert "alpha" in names
        assert "beta" in names

    def test_excludes_init(self, workspace):
        compass, tests = workspace
        _write_module(compass, "gamma", "z = 3\n")
        checker = ModuleHealthChecker(compass, tests)
        found = checker._discover_modules()
        assert all(p.name != "__init__.py" for p in found)

    def test_excludes_pycache(self, workspace):
        compass, tests = workspace
        cache = compass / "__pycache__"
        cache.mkdir()
        (cache / "cached.py").write_text("# cached\n")
        checker = ModuleHealthChecker(compass, tests)
        found = checker._discover_modules()
        assert all("__pycache__" not in str(p) for p in found)

    def test_empty_directory(self, workspace):
        compass, tests = workspace
        checker = ModuleHealthChecker(compass, tests)
        found = checker._discover_modules()
        assert found == []


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImport:
    def test_importable_module(self, workspace):
        compass, tests = workspace
        _write_module(compass, "good_mod", """\
            class GoodClass:
                pass

            def good_func():
                pass
        """)
        checker = ModuleHealthChecker(compass, tests)
        ok, err, mod = checker._try_import(compass / "good_mod.py")
        assert ok is True
        assert err == ""
        assert mod is not None

    def test_syntax_error_module(self, workspace):
        compass, tests = workspace
        (compass / "bad_syntax.py").write_text("def oops(\n")
        checker = ModuleHealthChecker(compass, tests)
        ok, err, mod = checker._try_import(compass / "bad_syntax.py")
        assert ok is False
        assert "SyntaxError" in err
        assert mod is None

    def test_runtime_error_module(self, workspace):
        compass, tests = workspace
        _write_module(compass, "runtime_err", """\
            raise ValueError("intentional")
        """)
        checker = ModuleHealthChecker(compass, tests)
        ok, err, mod = checker._try_import(compass / "runtime_err.py")
        assert ok is False
        assert "ValueError" in err


# ---------------------------------------------------------------------------
# Test file matching
# ---------------------------------------------------------------------------

class TestTestFileMatching:
    def test_has_test_true(self, workspace):
        compass, tests = workspace
        _write_test(tests, "alpha")
        checker = ModuleHealthChecker(compass, tests)
        assert checker._has_test_file("alpha") is True

    def test_has_test_false(self, workspace):
        compass, tests = workspace
        checker = ModuleHealthChecker(compass, tests)
        assert checker._has_test_file("nonexistent") is False


# ---------------------------------------------------------------------------
# Public members
# ---------------------------------------------------------------------------

class TestPublicMembers:
    def test_lists_classes_and_functions(self, workspace):
        compass, tests = workspace
        _write_module(compass, "members_mod", """\
            class PublicClass:
                pass

            class _PrivateClass:
                pass

            def public_func():
                pass

            def _private_func():
                pass
        """)
        checker = ModuleHealthChecker(compass, tests)
        ok, _, mod = checker._try_import(compass / "members_mod.py")
        assert ok
        classes, funcs = checker._public_members(mod)
        assert "PublicClass" in classes
        assert "_PrivateClass" not in classes
        assert "public_func" in funcs
        assert "_private_func" not in funcs

    def test_empty_module(self, workspace):
        compass, tests = workspace
        _write_module(compass, "empty_mod", "# nothing here\n")
        checker = ModuleHealthChecker(compass, tests)
        ok, _, mod = checker._try_import(compass / "empty_mod.py")
        assert ok
        classes, funcs = checker._public_members(mod)
        assert classes == []
        assert funcs == []


# ---------------------------------------------------------------------------
# Naming conventions
# ---------------------------------------------------------------------------

class TestNamingConventions:
    def test_snake_case_module_passes(self, workspace):
        compass, tests = workspace
        checker = ModuleHealthChecker(compass, tests)
        assert checker._check_conventions("my_module", []) is True

    def test_non_snake_case_fails(self, workspace):
        compass, tests = workspace
        checker = ModuleHealthChecker(compass, tests)
        assert checker._check_conventions("MyModule", []) is False
        assert checker._check_conventions("my-module", []) is False

    def test_pascal_case_class_passes(self, workspace):
        compass, tests = workspace
        checker = ModuleHealthChecker(compass, tests)
        assert checker._check_conventions("mod", ["MyClass", "Foo"]) is True

    def test_non_pascal_class_fails(self, workspace):
        compass, tests = workspace
        checker = ModuleHealthChecker(compass, tests)
        assert checker._check_conventions("mod", ["my_class"]) is False

    def test_single_word_module(self, workspace):
        compass, tests = workspace
        checker = ModuleHealthChecker(compass, tests)
        assert checker._check_conventions("alpha", ["Alpha"]) is True


# ---------------------------------------------------------------------------
# Circular import detection
# ---------------------------------------------------------------------------

class TestCircularImportDetection:
    def test_detects_circular_keyword(self):
        assert ModuleHealthChecker._is_circular_import(
            "ImportError: circular import detected"
        ) is True

    def test_detects_partially_initialized(self):
        msg = (
            "ImportError: cannot import name 'X' from partially initialized "
            "module 'foo'"
        )
        assert ModuleHealthChecker._is_circular_import(msg) is True

    def test_normal_error_not_circular(self):
        assert ModuleHealthChecker._is_circular_import(
            "ModuleNotFoundError: No module named 'xyz'"
        ) is False


# ---------------------------------------------------------------------------
# check_all integration
# ---------------------------------------------------------------------------

class TestCheckAll:
    def test_full_run(self, workspace):
        compass, tests = workspace
        _write_module(compass, "healthy_mod", """\
            class Healthy:
                pass

            def run():
                pass
        """)
        _write_test(tests, "healthy_mod")
        _write_module(compass, "broken_mod", "raise RuntimeError('boom')\n")

        checker = ModuleHealthChecker(compass, tests)
        result = checker.check_all()

        assert result.n_total == 2
        assert result.n_importable == 1
        assert result.n_import_errors == 1
        assert result.n_with_tests == 1
        assert "broken_mod" in result.modules_without_tests
        assert any("broken_mod" in e for e in result.import_errors)

    def test_empty_compass(self, workspace):
        compass, tests = workspace
        checker = ModuleHealthChecker(compass, tests)
        result = checker.check_all()
        assert result.n_total == 0
        assert result.statuses == []


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

class TestHTMLReport:
    def test_report_contains_key_sections(self, workspace):
        compass, tests = workspace
        _write_module(compass, "report_mod", "class Foo:\n    pass\n")
        _write_test(tests, "report_mod")
        checker = ModuleHealthChecker(compass, tests)
        checker.check_all()
        html = checker.generate_report()

        assert "<!DOCTYPE html>" in html
        assert "Module Health Report" in html
        assert "report_mod" in html
        assert "Total Modules" in html
        assert "Importable" in html

    def test_report_shows_errors(self, workspace):
        compass, tests = workspace
        _write_module(compass, "err_mod", "raise TypeError('oops')\n")
        checker = ModuleHealthChecker(compass, tests)
        checker.check_all()
        html = checker.generate_report()

        assert "Import Error Details" in html
        assert "err_mod" in html
        assert "oops" in html

    def test_report_escapes_html(self, workspace):
        compass, tests = workspace
        _write_module(compass, "html_mod", 'raise ValueError("<script>alert(1)</script>")\n')
        checker = ModuleHealthChecker(compass, tests)
        checker.check_all()
        html = checker.generate_report()

        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_report_without_check_raises(self, workspace):
        compass, tests = workspace
        checker = ModuleHealthChecker(compass, tests)
        with pytest.raises(RuntimeError, match="No results"):
            checker.generate_report()

    def test_report_accepts_explicit_result(self, workspace):
        compass, tests = workspace
        checker = ModuleHealthChecker(compass, tests)
        result = HealthResult(n_total=0)
        html = checker.generate_report(result)
        assert "Module Health Report" in html

    def test_green_badge_for_healthy(self, workspace):
        compass, tests = workspace
        _write_module(compass, "green_mod", "pass\n")
        _write_test(tests, "green_mod")
        checker = ModuleHealthChecker(compass, tests)
        checker.check_all()
        html = checker.generate_report()
        assert "background:green" in html

    def test_yellow_badge_for_no_tests(self, workspace):
        compass, tests = workspace
        _write_module(compass, "yellow_mod", "pass\n")
        checker = ModuleHealthChecker(compass, tests)
        checker.check_all()
        html = checker.generate_report()
        assert "background:yellow" in html

    def test_red_badge_for_import_error(self, workspace):
        compass, tests = workspace
        (compass / "red_mod.py").write_text("import nonexistent_xyz_pkg\n")
        checker = ModuleHealthChecker(compass, tests)
        checker.check_all()
        html = checker.generate_report()
        assert "background:red" in html
