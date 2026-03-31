"""Tests for compass/auto_docs.py — auto-documentation generator.

Covers:
  - AST extraction: parse_module (classes, functions, params, imports, constants)
  - detect_test_coverage: direct match, import-based detection
  - build_dependency_graph: structure
  - render_text_dependency_graph: output format
  - AutoDocGenerator: scan, coverage_summary, generate
  - HTML output: structure, content, self-containment
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from compass.auto_docs import (
    AutoDocGenerator,
    ClassInfo,
    CoverageInfo,
    MethodInfo,
    ModuleInfo,
    ParamInfo,
    build_dependency_graph,
    detect_test_coverage,
    parse_module,
    render_text_dependency_graph,
    _count_test_functions,
)


# ── parse_module ─────────────────────────────────────────────────────────


class TestParseModule:
    def test_synthetic_module(self, tmp_path):
        src = tmp_path / "sample.py"
        src.write_text(textwrap.dedent('''\
            """Sample module."""
            MY_CONST = 42
            class Foo:
                """Foo doc."""
                def __init__(self, x: int = 5): pass
                def bar(self, y: str) -> bool:
                    """Do bar."""
                    return True
                def _private(self): pass
            def helper(a: float, b: float = 1.0) -> float:
                """A helper."""
                return a + b
            def _internal(): pass
        '''))
        info = parse_module(src)
        assert info.name == "sample"
        assert info.docstring == "Sample module."
        assert len(info.classes) == 1
        assert info.classes[0].name == "Foo"
        assert len(info.classes[0].methods) == 1  # bar only (not __init__, not _private)
        assert info.classes[0].methods[0].name == "bar"
        assert len(info.classes[0].init_params) == 1
        assert info.classes[0].init_params[0].name == "x"
        assert info.classes[0].init_params[0].default == "5"
        assert len(info.functions) == 1  # helper only (not _internal)
        assert info.functions[0].name == "helper"
        assert info.functions[0].returns == "float"
        assert "MY_CONST" in info.constants

    def test_extracts_compass_imports(self, tmp_path):
        src = tmp_path / "importer.py"
        src.write_text("from compass.signal_model import SignalModel\nfrom compass.regime import Regime\n")
        info = parse_module(src)
        assert "signal_model" in info.imports_from_compass
        assert "regime" in info.imports_from_compass

    def test_handles_syntax_error(self, tmp_path):
        src = tmp_path / "bad.py"
        src.write_text("def broken(:\n  pass")
        info = parse_module(src)
        assert info.name == "bad"
        assert info.classes == []
        assert info.functions == []

    def test_line_count(self, tmp_path):
        src = tmp_path / "short.py"
        src.write_text("a = 1\nb = 2\nc = 3\n")
        info = parse_module(src)
        assert info.line_count >= 3

    def test_real_signal_model(self):
        p = Path("compass/signal_model.py")
        if not p.exists():
            pytest.skip()
        info = parse_module(p)
        assert any(c.name == "SignalModel" for c in info.classes)

    def test_static_and_classmethod(self, tmp_path):
        src = tmp_path / "decs.py"
        src.write_text(textwrap.dedent('''\
            class A:
                @staticmethod
                def s(): pass
                @classmethod
                def c(cls): pass
                @property
                def p(self): return 1
        '''))
        info = parse_module(src)
        methods = info.classes[0].methods
        names = {m.name: m for m in methods}
        assert names["s"].is_static
        assert names["c"].is_classmethod
        assert names["p"].is_property


# ── detect_test_coverage ─────────────────────────────────────────────────


class TestDetectCoverage:
    def test_direct_match(self, tmp_path):
        (tmp_path / "test_foo.py").write_text("def test_one(): pass\ndef test_two(): pass\n")
        cov = detect_test_coverage("foo", tmp_path)
        assert cov.has_tests
        assert "test_foo.py" in cov.test_files
        assert cov.test_count == 2

    def test_import_based_match(self, tmp_path):
        (tmp_path / "test_integration.py").write_text(
            "from compass.bar import BarClass\ndef test_bar(): pass\n"
        )
        cov = detect_test_coverage("bar", tmp_path)
        assert cov.has_tests
        assert "test_integration.py" in cov.test_files

    def test_no_tests(self, tmp_path):
        cov = detect_test_coverage("nonexistent", tmp_path)
        assert not cov.has_tests
        assert cov.test_count == 0

    def test_missing_tests_dir(self, tmp_path):
        cov = detect_test_coverage("foo", tmp_path / "nope")
        assert not cov.has_tests


# ── _count_test_functions ────────────────────────────────────────────────


class TestCountTests:
    def test_counts(self, tmp_path):
        f = tmp_path / "test_x.py"
        f.write_text("def test_a(): pass\ndef test_b(): pass\ndef helper(): pass\n")
        assert _count_test_functions(f) == 2

    def test_empty(self, tmp_path):
        f = tmp_path / "test_empty.py"
        f.write_text("")
        assert _count_test_functions(f) == 0


# ── build_dependency_graph ───────────────────────────────────────────────


class TestDependencyGraph:
    def test_structure(self):
        modules = [
            ModuleInfo("a.py", "a", imports_from_compass=["b", "c"]),
            ModuleInfo("b.py", "b", imports_from_compass=["c"]),
            ModuleInfo("c.py", "c", imports_from_compass=[]),
        ]
        g = build_dependency_graph(modules)
        assert g["a"] == ["b", "c"]
        assert g["c"] == []

    def test_empty(self):
        assert build_dependency_graph([]) == {}


class TestRenderTextGraph:
    def test_output(self):
        g = {"a": ["b"], "b": []}
        text = render_text_dependency_graph(g)
        assert "a → b" in text
        assert "no compass deps" in text


# ── AutoDocGenerator ─────────────────────────────────────────────────────


class TestAutoDocGenerator:
    def test_scan_real_modules(self):
        gen = AutoDocGenerator()
        modules = gen.scan()
        assert len(modules) > 10
        assert any(m.name == "signal_model" for m in modules)

    def test_coverage_summary(self):
        gen = AutoDocGenerator()
        gen.scan()
        cov = gen.coverage_summary()
        assert "total_modules" in cov
        assert "tested_modules" in cov
        assert "coverage_pct" in cov
        assert cov["coverage_pct"] >= 0

    def test_generate_html(self, tmp_path):
        gen = AutoDocGenerator()
        gen.scan()
        path = gen.generate(str(tmp_path / "docs.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "COMPASS" in content
        assert "Module Index" in content
        assert "Dependency Matrix" in content
        assert "Test Coverage" in content
        assert len(content) > 10000

    def test_html_no_external(self, tmp_path):
        gen = AutoDocGenerator()
        gen.scan()
        path = gen.generate(str(tmp_path / "docs.html"))
        content = open(path).read()
        assert "http://" not in content
        assert "https://" not in content

    def test_custom_dirs(self, tmp_path):
        cdir = tmp_path / "compass"
        tdir = tmp_path / "tests"
        cdir.mkdir()
        tdir.mkdir()
        (cdir / "a.py").write_text('"""A."""\nclass X: pass\n')
        (cdir / "b.py").write_text('"""B."""\nfrom compass.a import X\ndef foo(): pass\n')
        (tdir / "test_a.py").write_text("def test_x(): pass\n")
        gen = AutoDocGenerator(str(cdir), str(tdir))
        gen.scan()
        assert len(gen.modules) == 2
        assert gen.modules[0].has_tests or gen.modules[1].has_tests
        path = gen.generate(str(tmp_path / "docs.html"))
        assert Path(path).exists()
