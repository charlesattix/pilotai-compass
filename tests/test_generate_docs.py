"""Tests for compass/generate_docs.py — auto-documentation generator.

Covers:
  - AST extraction: params, functions, classes, imports, constants
  - parse_module: single file parsing
  - scan_compass_modules: directory scanning
  - build_dependency_matrix: import graph
  - generate_html: structure, content, completeness
  - generate_docs: end-to-end
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from compass.generate_docs import (
    QUICK_START,
    ClassInfo,
    FuncInfo,
    ModuleInfo,
    ParamInfo,
    build_dependency_matrix,
    generate_docs,
    generate_html,
    parse_module,
    scan_compass_modules,
)


# ── parse_module ─────────────────────────────────────────────────────────


class TestParseModule:
    def test_parse_real_module(self):
        p = Path("compass/signal_model.py")
        if not p.exists():
            pytest.skip("signal_model.py not found")
        info = parse_module(p)
        assert info.name == "signal_model"
        assert len(info.classes) >= 1
        assert any(c.name == "SignalModel" for c in info.classes)

    def test_parse_extracts_docstring(self):
        p = Path("compass/crisis_hedge.py")
        if not p.exists():
            pytest.skip()
        info = parse_module(p)
        assert len(info.docstring) > 0

    def test_parse_extracts_imports(self):
        p = Path("compass/stress_test.py")
        if not p.exists():
            pytest.skip()
        info = parse_module(p)
        assert "crisis_hedge" in info.imports_from

    def test_parse_synthetic_module(self, tmp_path):
        src = tmp_path / "sample.py"
        src.write_text(textwrap.dedent('''\
            """Module docstring."""
            MY_CONST = 42
            class Foo:
                """Foo class."""
                def __init__(self, x: int = 5):
                    self.x = x
                def bar(self, y: str) -> bool:
                    """Do bar."""
                    return True
            def helper(a: float, b: float = 1.0) -> float:
                """A helper."""
                return a + b
        '''))
        info = parse_module(src)
        assert info.name == "sample"
        assert info.docstring == "Module docstring."
        assert len(info.classes) == 1
        assert info.classes[0].name == "Foo"
        assert len(info.classes[0].methods) == 1  # bar (not __init__)
        assert info.classes[0].methods[0].name == "bar"
        assert len(info.classes[0].init_params) == 1
        assert info.classes[0].init_params[0].name == "x"
        assert len(info.functions) == 1
        assert info.functions[0].name == "helper"
        assert "MY_CONST" in info.constants

    def test_parse_handles_syntax_error(self, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("def broken(:\n  pass")
        info = parse_module(bad)
        assert info.name == "bad"
        assert info.classes == []


# ── scan_compass_modules ─────────────────────────────────────────────────


class TestScanModules:
    def test_finds_modules(self):
        modules = scan_compass_modules()
        assert len(modules) > 10
        names = [m.name for m in modules]
        assert "signal_model" in names
        assert "crisis_hedge" in names

    def test_excludes_init(self):
        modules = scan_compass_modules()
        names = [m.name for m in modules]
        assert "__init__" not in names

    def test_custom_dir(self, tmp_path):
        (tmp_path / "a.py").write_text('"""A."""\ndef foo(): pass')
        (tmp_path / "b.py").write_text('"""B."""\nclass Bar: pass')
        (tmp_path / "__init__.py").write_text("")
        modules = scan_compass_modules(tmp_path)
        assert len(modules) == 2
        names = [m.name for m in modules]
        assert "a" in names and "b" in names


# ── build_dependency_matrix ──────────────────────────────────────────────


class TestDependencyMatrix:
    def test_real_modules(self):
        modules = scan_compass_modules()
        matrix = build_dependency_matrix(modules)
        assert isinstance(matrix, dict)
        # stress_test imports from crisis_hedge
        assert "crisis_hedge" in matrix.get("stress_test", [])

    def test_empty(self):
        matrix = build_dependency_matrix([])
        assert matrix == {}


# ── generate_html ────────────────────────────────────────────────────────


class TestGenerateHTML:
    def _make_modules(self):
        return [ModuleInfo(
            filename="sample.py", name="sample",
            docstring="A sample module.",
            classes=[ClassInfo("MyClass", "A class.", ["Base"],
                               [FuncInfo("method", "Do stuff.", [ParamInfo("x", "int", "")],
                                         "bool", [], True)],
                               [ParamInfo("name", "str", "'default'")])],
            functions=[FuncInfo("helper", "Help.", [ParamInfo("a", "float", "")], "float", [])],
            imports_from=["other"],
            constants=["MY_CONST"],
        )]

    def test_valid_html(self):
        html = generate_html(self._make_modules(), {"sample": ["other"]})
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_module(self):
        html = generate_html(self._make_modules(), {"sample": ["other"]})
        assert "sample.py" in html
        assert "MyClass" in html
        assert "method" in html
        assert "helper" in html

    def test_contains_quick_start(self):
        html = generate_html(self._make_modules(), {})
        assert "Quick Start" in html
        for title in QUICK_START:
            assert title in html

    def test_contains_dependency_matrix(self):
        html = generate_html(self._make_modules(), {"sample": ["other"]})
        assert "Dependency Matrix" in html

    def test_no_external_resources(self):
        html = generate_html(self._make_modules(), {})
        assert "http://" not in html
        assert "https://" not in html


# ── generate_docs end-to-end ─────────────────────────────────────────────


class TestGenerateDocs:
    def test_end_to_end(self, tmp_path):
        out = str(tmp_path / "docs.html")
        path = generate_docs(output=out)
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "COMPASS" in content
        assert "SignalModel" in content
        assert len(content) > 10000

    def test_output_file_created(self, tmp_path):
        out = tmp_path / "docs.html"
        generate_docs(output=str(out))
        assert out.exists()
        assert out.stat().st_size > 5000
